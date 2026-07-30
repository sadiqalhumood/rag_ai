"""Citation extraction and enforcement.

Two failures this module exists to prevent:

* **A hallucinated marker becoming a citation.** `[7]` when five blocks were
  packed resolves to nothing. It is recorded as invalid, stripped from the
  answer text, and never becomes a `Citation`. Resolution is a dict lookup
  against `PackedContext`, so an out-of-range marker cannot be silently
  coerced into the nearest real one.

* **`CitationError` escaping into production.** `Answer.__post_init__` raises
  when a non-refusal carries no citations -- that invariant is the point, but it
  should be the last line of defence, not the mechanism. `finalize` checks the
  same condition first and converts to an explicit refusal with a reason, so the
  exception is unreachable from the generator path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..core.types import Answer, Citation, QueryRoute
from .packer import PackedChunk, PackedContext
from .refusal import unique_terms, weighted_overlap

__all__ = [
    "CitationResult",
    "MARKER_RE",
    "finalize",
    "parse_markers",
    "resolve",
    "MAX_QUOTED_SPAN",
]

#: Bounded to four digits: a "[123456]" in a field value is not a marker, and
#: chunk bodies have their brackets neutralised by the packer anyway.
MARKER_RE = re.compile(r"\[(\d{1,4})\]")

#: Citations carry a quoted span for provenance display, not the whole chunk.
MAX_QUOTED_SPAN = 240


@dataclass(frozen=True)
class CitationResult:
    citations: tuple[Citation, ...] = ()
    valid_markers: tuple[int, ...] = ()
    invalid_markers: tuple[int, ...] = ()
    #: Answer text with unresolvable markers removed.
    text: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.citations)

    def as_trace(self) -> dict[str, Any]:
        return {
            "valid_markers": list(self.valid_markers),
            "invalid_markers": list(self.invalid_markers),
            "n_citations": len(self.citations),
        }


def parse_markers(text: str) -> tuple[int, ...]:
    """Markers in order of first appearance, deduplicated."""
    seen: list[int] = []
    for match in MARKER_RE.finditer(text or ""):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    return tuple(seen)


def _quoted_span(block: PackedChunk, focus: str) -> str:
    """The line of the block that best matches `focus`, bounded in length.

    Deterministic: ties break on line order, and the overlap score iterates a
    sorted term list rather than a set.
    """
    lines = [ln.strip() for ln in (block.text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        span = (block.text or "").strip()
    else:
        focus_terms = unique_terms(focus)
        if focus_terms:
            scored = sorted(
                ((-weighted_overlap(focus_terms, ln, None), i, ln)
                 for i, ln in enumerate(lines))
            )
            span = scored[0][2]
        else:
            span = lines[0]
    if len(span) > MAX_QUOTED_SPAN:
        cut = span[:MAX_QUOTED_SPAN]
        pivot = cut.rfind(" ")
        span = (cut[:pivot] if pivot > MAX_QUOTED_SPAN // 2 else cut).rstrip() + "…"
    return span


def resolve(
    text: str,
    context: PackedContext,
    *,
    question: str = "",
) -> CitationResult:
    """Turn markers in `text` into `Citation` objects against `context`."""
    markers = parse_markers(text)
    valid: list[int] = []
    invalid: list[int] = []
    citations: list[Citation] = []

    for marker in markers:
        block = context.by_marker(marker)
        if block is None:
            invalid.append(marker)
            continue
        valid.append(marker)
        citations.append(
            Citation(
                chunk_id=block.chunk_id,
                source_id=block.source_id,
                row_refs=tuple(block.chunk.row_refs),
                score=block.score,
                quoted_span=_quoted_span(block, question or text),
            )
        )

    cleaned = text or ""
    if invalid:
        bad = set(invalid)

        def _strip(match: re.Match[str]) -> str:
            return "" if int(match.group(1)) in bad else match.group(0)

        cleaned = MARKER_RE.sub(_strip, cleaned)
        # Collapse the whitespace the removal left behind, without touching
        # line structure.
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
        cleaned = "\n".join(ln.rstrip() for ln in cleaned.splitlines())

    return CitationResult(
        citations=tuple(citations),
        valid_markers=tuple(valid),
        invalid_markers=tuple(invalid),
        text=cleaned.strip(),
    )


def finalize(
    text: str,
    context: PackedContext,
    *,
    question: str = "",
    route: QueryRoute | None = None,
    trace: Mapping[str, Any] | None = None,
    empty_reason: str = "no_valid_citations",
) -> Answer:
    """Build a cited `Answer`, or an explicit refusal when it cannot be cited.

    This is the only sanctioned way for a generator to produce an `Answer`:
    it guarantees `CitationError` is never how an uncited answer surfaces.
    """
    result = resolve(text, context, question=question)
    trace = dict(trace or {})
    trace["citations"] = result.as_trace()

    if not result.citations or not result.text:
        detail = (
            "generated answer cited no packed chunk"
            if not result.citations
            else "generated answer was empty after removing invalid citations"
        )
        if result.invalid_markers:
            detail += (
                "; rejected hallucinated marker(s) "
                + ", ".join(f"[{m}]" for m in result.invalid_markers)
            )
        return Answer.refusal(
            f"{empty_reason}: {detail}",
            route=route,
            trace=trace,
        )

    return Answer(
        text=result.text,
        citations=result.citations,
        refused=False,
        reason="",
        route=route,
        trace=trace,
    )
