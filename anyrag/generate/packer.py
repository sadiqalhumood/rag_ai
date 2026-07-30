"""Context packing under a token budget that is measured, never estimated.

Every number in this module comes from `anyrag.core.tokenizer`. A chars/4
heuristic is wrong by 2-4x on exactly the inputs this corpus is built from --
Arabic text, CJK, dense numeric tables, long identifier-like strings -- and the
failure mode is silent prompt overflow rather than a visible error, so there is
no cheap-estimate fast path here at all, not even as an optimisation.

Three behaviours the packer owes its caller:

* **Drop lowest-scored chunks first.** When the selection is over budget the
  worst-scoring chunk goes, then the next worst, until it fits. Not "skip
  whatever does not fit and keep going down the list" -- a top-scoring chunk is
  never sacrificed to make room for two mediocre ones.
* **Truncate rather than drop an oversized chunk.** A row wide enough to blow
  `max_chunk_tokens` still carries the answer in its first fields, so it is cut
  at a token boundary and marked, not discarded.
* **Refuse rather than crash.** If not even a single truncated chunk fits the
  whole budget, `pack` returns an empty context carrying a reason. The caller
  turns that into a refusal; nothing raises.

Packing is deterministic for a given input ordering: all ties break on input
index, and no set iteration reaches a sort key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..core.config import GenerationConfig
from ..core.errors import BudgetExceededError
from ..core.tokenizer import Tokenizer, TokenizerInfo, get_tokenizer
from ..core.types import Chunk, Hit, RowRef

__all__ = [
    "PackedChunk",
    "PackedContext",
    "pack",
    "render_block",
    "BLOCK_SEPARATOR",
    "TRUNCATION_MARK",
]

#: Joined between rendered blocks. Counted against the budget like everything
#: else.
BLOCK_SEPARATOR = "\n\n"

#: Appended to a chunk body that was cut at `max_chunk_tokens`, so both the
#: model and a human reading the trace can see the text is partial.
TRUNCATION_MARK = " …[truncated]"

#: How many row refs to name in a block header before eliding.
_MAX_HEADER_ROWS = 3


def _sanitize(text: str) -> str:
    """Neutralise square brackets inside chunk bodies.

    Citation markers are `[n]`. A field value that happens to contain "[2]"
    would otherwise be indistinguishable from a marker when the answer is parsed
    back, which is precisely the ambiguity the citation validator exists to
    prevent. Brackets become parentheses; nothing else is altered.
    """
    return text.replace("[", "(").replace("]", ")")


def _rows_label(row_refs: Sequence[RowRef]) -> str:
    if not row_refs:
        return ""
    shown = ", ".join(str(r) for r in row_refs[:_MAX_HEADER_ROWS])
    if len(row_refs) > _MAX_HEADER_ROWS:
        shown += f", +{len(row_refs) - _MAX_HEADER_ROWS} more"
    return f" rows={shown}"


def render_block(marker: str, chunk: Chunk, body: str, score: float) -> str:
    """One context block: a visible, stable citation marker plus provenance."""
    table = f" table={chunk.table}" if chunk.table else ""
    kind = getattr(chunk.kind, "value", str(chunk.kind))
    return (
        f"{marker} source={chunk.source_id} kind={kind}{table}"
        f"{_rows_label(chunk.row_refs)} score={score:.3f}\n{body}"
    )


@dataclass(frozen=True)
class PackedChunk:
    """A chunk that made it into the prompt, with its assigned marker."""

    marker: int
    chunk: Chunk
    score: float
    #: Body as rendered: bracket-sanitised, possibly truncated. This -- not
    #: `chunk.text` -- is what the generator actually saw, so citation spans are
    #: quoted from here.
    text: str
    truncated: bool = False
    rank: int = 0
    retriever: str = ""

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def source_id(self) -> str:
        return self.chunk.source_id

    @property
    def label(self) -> str:
        return f"[{self.marker}]"

    def render(self) -> str:
        return render_block(self.label, self.chunk, self.text, self.score)


@dataclass(frozen=True)
class PackedContext:
    """Selected chunks, the rendered context string, and its measured size."""

    blocks: tuple[PackedChunk, ...] = ()
    text: str = ""
    #: Measured token count of `text`, from the tokenizer -- never estimated.
    tokens: int = 0
    #: Tokens the caller set aside for prompt scaffolding and the question.
    reserved_tokens: int = 0
    #: `config.max_prompt_tokens` at pack time, carried for the trace.
    budget: int = 0
    dropped: tuple[str, ...] = ()
    truncated: tuple[str, ...] = ()
    #: Empty when packing succeeded; otherwise why it produced nothing.
    reason: str = ""
    tokenizer: TokenizerInfo | None = None

    @property
    def is_empty(self) -> bool:
        return not self.blocks

    @property
    def total_tokens(self) -> int:
        return self.tokens + self.reserved_tokens

    @property
    def marker_map(self) -> dict[int, str]:
        """Marker -> chunk_id. Kept as data; never re-parsed out of prose."""
        return {b.marker: b.chunk_id for b in self.blocks}

    def by_marker(self, marker: int) -> PackedChunk | None:
        for b in self.blocks:
            if b.marker == marker:
                return b
        return None

    def as_trace(self) -> dict[str, Any]:
        return {
            "packed_chunks": len(self.blocks),
            "context_tokens": self.tokens,
            "reserved_tokens": self.reserved_tokens,
            "total_tokens": self.total_tokens,
            "budget": self.budget,
            "dropped_chunk_ids": list(self.dropped),
            "truncated_chunk_ids": list(self.truncated),
            "marker_map": self.marker_map,
            "tokenizer": self.tokenizer.name if self.tokenizer else None,
            "tokenizer_degraded": bool(self.tokenizer.degraded)
            if self.tokenizer
            else None,
            "reason": self.reason,
        }


@dataclass(eq=False)  # identity comparison: two chunks may render identically
class _Entry:
    index: int
    hit: Hit
    body: str
    truncated: bool
    cost: int = 0


def _truncate_body(body: str, max_tokens: int, tok: Tokenizer) -> tuple[str, bool]:
    """Cut at a token boundary, leaving room for the truncation mark."""
    if max_tokens <= 0 or tok.count(body) <= max_tokens:
        return body, False
    mark_cost = tok.count(TRUNCATION_MARK)
    keep = max(1, max_tokens - mark_cost)
    cut = tok.truncate(body, keep).rstrip()
    return cut + TRUNCATION_MARK, True


def pack(
    hits: Sequence[Hit],
    config: GenerationConfig | None = None,
    *,
    reserved_tokens: int = 0,
    tokenizer: Tokenizer | None = None,
    strict: bool = False,
) -> PackedContext:
    """Select and render as many chunks as the measured budget allows.

    `reserved_tokens` is what the caller has already committed to prompt
    scaffolding and the question, so the invariant this enforces is
    `context_tokens + reserved_tokens <= config.max_prompt_tokens` -- the whole
    prompt, not just the context.

    With `strict=True` an unpackable input raises `BudgetExceededError` instead
    of returning an empty context. The default is False because the production
    path wants a refusal, not an exception.
    """
    config = config or GenerationConfig()
    tok = tokenizer or get_tokenizer()
    hits = list(hits or ())
    budget = config.max_prompt_tokens

    if not hits:
        empty = PackedContext(
            budget=budget,
            reserved_tokens=reserved_tokens,
            reason="no_hits",
            tokenizer=tok.info,
        )
        if strict:
            raise BudgetExceededError("no hits to pack")
        return empty

    entries: list[_Entry] = []
    for i, hit in enumerate(hits):
        body, was_cut = _truncate_body(
            _sanitize(hit.chunk.text or ""), config.max_chunk_tokens, tok
        )
        entries.append(_Entry(index=i, hit=hit, body=body, truncated=was_cut))

    # Cost each block once, using the widest marker any block could receive.
    # Overshooting the marker width can only make the estimate conservative,
    # never optimistic, and it keeps this loop O(n) tokenizations instead of
    # O(n^2) -- which matters at 500 chunks.
    width = len(str(len(entries)))
    placeholder = "[" + "9" * width + "]"
    sep_cost = tok.count(BLOCK_SEPARATOR)
    for e in entries:
        e.cost = tok.count(
            render_block(placeholder, e.hit.chunk, e.body, e.hit.score)
        )

    # Worst first: lowest score, and among equal scores the later input index.
    drop_order = sorted(entries, key=lambda e: (e.hit.score, -e.index))
    kept: list[_Entry] = list(entries)
    dropped: list[str] = []

    def estimate(sel: Sequence[_Entry]) -> int:
        if not sel:
            return 0
        return sum(e.cost for e in sel) + sep_cost * (len(sel) - 1)

    di = 0
    while kept and estimate(kept) + reserved_tokens > budget:
        victim = drop_order[di]
        di += 1
        if victim in kept:
            kept.remove(victim)
            dropped.append(victim.hit.chunk.chunk_id)

    def build(sel: Sequence[_Entry]) -> tuple[tuple[PackedChunk, ...], str]:
        blocks = tuple(
            PackedChunk(
                marker=n,
                chunk=e.hit.chunk,
                score=e.hit.score,
                text=e.body,
                truncated=e.truncated,
                rank=e.hit.rank,
                retriever=e.hit.retriever,
            )
            for n, e in enumerate(sel, start=1)
        )
        return blocks, BLOCK_SEPARATOR.join(b.render() for b in blocks)

    # The estimate is conservative but tokenizers are not strictly additive
    # across concatenation, so the final answer is a measured count of the
    # actual string. This loop normally does not execute at all.
    blocks, text = build(kept)
    measured = tok.count(text) if text else 0
    while kept and measured + reserved_tokens > budget:
        victim = drop_order[di]
        di += 1
        if victim in kept:
            kept.remove(victim)
            dropped.append(victim.hit.chunk.chunk_id)
        blocks, text = build(kept)
        measured = tok.count(text) if text else 0

    if not kept:
        reason = "context_budget_exhausted"
        if strict:
            raise BudgetExceededError(
                f"no chunk fits the {budget}-token prompt budget "
                f"({reserved_tokens} tokens reserved for the prompt scaffold)"
            )
        return PackedContext(
            budget=budget,
            reserved_tokens=reserved_tokens,
            dropped=tuple(dropped),
            reason=reason,
            tokenizer=tok.info,
        )

    return PackedContext(
        blocks=blocks,
        text=text,
        tokens=measured,
        reserved_tokens=reserved_tokens,
        budget=budget,
        dropped=tuple(dropped),
        truncated=tuple(b.chunk_id for b in blocks if b.truncated),
        tokenizer=tok.info,
    )
