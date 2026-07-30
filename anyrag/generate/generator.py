"""Answer generators.

`ExtractiveGenerator` is the default and the one the whole eval runs on. It is
offline, deterministic, and composes its answer out of the packed chunks by
selection alone -- it never writes a value that was not in the retrieved text.

Be clear-eyed about what that proves. An extractive composer *cannot* fabricate
a fact the way an LLM can, so the eval's false-answer rate over this generator
measures the refusal-threshold logic in `refusal.py`, not hallucination
resistance. It is still the right default: it makes the retrieval and refusal
behaviour measurable without a network call, and it is byte-for-byte
reproducible. `AnthropicGenerator` exists so the same questions can be run
through a model that *can* hallucinate, behind the same packing, the same
citation enforcement and the same refusal gates.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from ..core.config import GenerationConfig
from ..core.errors import ConfigError
from ..core.tokenizer import Tokenizer, get_tokenizer
from ..core.types import Answer, Hit, QueryRoute
from . import citations as _citations
from .packer import PackedContext, pack
from .prompt import REFUSAL_SENTINEL, Prompt, build_prompt, overhead_tokens
from .refusal import (
    RefusalDecision,
    RefusalPolicy,
    assess,
    term_weights,
    unique_terms,
    weighted_overlap,
)

__all__ = [
    "ExtractiveGenerator",
    "AnthropicGenerator",
    "get_generator",
    "available_generators",
    "DEFAULT_ANTHROPIC_MODEL",
]

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

#: Separators a verbalized row chunk plausibly uses between fields.
_SEGMENT_SPLITS = (" | ", " · ", "; ")

_MAX_EVIDENCE_BLOCKS = 3
_MAX_SEGMENTS_PER_BLOCK = 2
_MAX_SEGMENT_CHARS = 400

#: A block beyond the first is only cited when its weighted overlap with the
#: question reaches this fraction of the best block's. Without it, every packed
#: chunk gets cited and citation precision collapses: five near-identical
#: customer rows retrieved for a question about one of them would all be
#: presented as supporting evidence.
_EVIDENCE_RATIO = 0.6

_PREAMBLE = "From the retrieved records:"


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------


def _pack_for_prompt(
    question: str,
    hits: Sequence[Hit],
    config: GenerationConfig,
    tok: Tokenizer,
) -> tuple[PackedContext, Prompt]:
    """Pack, assemble, and verify the *whole* prompt against the budget.

    `overhead_tokens` measures the scaffolding exactly, but tokenizers merge
    across concatenation boundaries, so the assembled prompt is measured again
    and the reserve grown until it genuinely fits. Bounded to a few rounds; the
    first almost always succeeds.
    """
    reserve = overhead_tokens(question, tokenizer=tok)
    context = pack(hits, config, reserved_tokens=reserve, tokenizer=tok)
    prompt = build_prompt(question, context)

    for _ in range(4):
        measured = prompt.token_count(tok)
        if measured <= config.max_prompt_tokens or context.is_empty:
            return context, prompt
        # Grow the reserve by the observed excess and re-pack.
        reserve += max(1, measured - config.max_prompt_tokens)
        context = pack(hits, config, reserved_tokens=reserve, tokenizer=tok)
        prompt = build_prompt(question, context)
    return context, prompt


def _base_trace(
    name: str,
    tok: Tokenizer,
    decision: RefusalDecision,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "generator": name,
        "tokenizer": tok.info.name,
        "tokenizer_degraded": tok.info.degraded,
        "refusal": decision.as_trace(),
    }
    if extra:
        trace.update(extra)
    return trace


def _segments(text: str) -> list[str]:
    """Split a chunk body into citable units, preserving order."""
    parts: list[str] = []
    for line in (text or "").splitlines():
        pieces = [line]
        for sep in _SEGMENT_SPLITS:
            nxt: list[str] = []
            for piece in pieces:
                nxt.extend(piece.split(sep))
            pieces = nxt
        for piece in pieces:
            piece = piece.strip(" \t-•")
            if piece:
                parts.append(piece[:_MAX_SEGMENT_CHARS].rstrip())
    return parts


# --------------------------------------------------------------------------
# Extractive (default, offline, deterministic)
# --------------------------------------------------------------------------


class ExtractiveGenerator:
    """Compose an answer by selecting field values from the packed chunks.

    Deterministic by construction: every ordering decision breaks ties on an
    integer index, no set iteration reaches a sort key, and nothing consults the
    clock, the filesystem or the network.
    """

    name = "extractive"

    def __init__(
        self,
        *,
        policy: RefusalPolicy | None = None,
        tokenizer: Tokenizer | None = None,
        max_evidence_blocks: int = _MAX_EVIDENCE_BLOCKS,
        max_segments_per_block: int = _MAX_SEGMENTS_PER_BLOCK,
        evidence_ratio: float = _EVIDENCE_RATIO,
    ) -> None:
        self._policy = policy
        self._tokenizer = tokenizer
        self.max_evidence_blocks = max_evidence_blocks
        self.max_segments_per_block = max_segments_per_block
        self.evidence_ratio = evidence_ratio

    # -- interface ---------------------------------------------------------

    def generate(
        self,
        question: str,
        hits: Sequence[Hit],
        config: GenerationConfig | None = None,
        *,
        route: QueryRoute | None = None,
        policy: RefusalPolicy | None = None,
        **kwargs: Any,
    ) -> Answer:
        config = config or GenerationConfig()
        tok = self._tokenizer or get_tokenizer()
        pol = policy or self._policy or RefusalPolicy.from_config(config)

        decision = assess(question, hits, config, policy=pol)
        trace = _base_trace(self.name, tok, decision)
        if decision.refuse:
            return Answer.refusal(
                f"{decision.code}: {decision.detail}", route=route, trace=trace
            )

        context, _prompt = _pack_for_prompt(question, hits, config, tok)
        trace["packing"] = context.as_trace()
        if context.is_empty:
            return Answer.refusal(
                "context_budget_exhausted: no retrieved chunk fits the "
                f"{config.max_prompt_tokens}-token prompt budget",
                route=route,
                trace=trace,
            )

        body = self._compose(question, context, config, tok)
        if not body:
            return Answer.refusal(
                "no_extractable_answer: packed context contained no citable "
                "field values",
                route=route,
                trace=trace,
            )

        return _citations.finalize(
            body, context, question=question, route=route, trace=trace
        )

    # -- composition -------------------------------------------------------

    def _compose(
        self,
        question: str,
        context: PackedContext,
        config: GenerationConfig,
        tok: Tokenizer,
    ) -> str:
        q_terms = unique_terms(question)
        weights = term_weights([b.text for b in context.blocks])
        overlaps = {
            b.marker: weighted_overlap(q_terms, b.text, weights)
            for b in context.blocks
        }
        best_overlap = max(overlaps.values(), default=0.0)

        # Retrieval order decides *rank*; lexical overlap decides *inclusion*.
        # The top-scored block is always cited (the refusal gates already
        # accepted it); the rest must be comparably relevant to the question.
        ranked = sorted(context.blocks, key=lambda b: (-b.score, b.marker))
        cap = max(1, self.max_evidence_blocks)
        if best_overlap <= 0.0:
            ranked_blocks = ranked[:1]
        else:
            floor = best_overlap * self.evidence_ratio
            ranked_blocks = [ranked[0]] + [
                b for b in ranked[1:] if overlaps[b.marker] >= floor
            ]
            ranked_blocks = ranked_blocks[:cap]

        lines: list[str] = []
        for block in ranked_blocks:
            segs = _segments(block.text)
            if not segs:
                continue
            scored = sorted(
                (
                    (-weighted_overlap(q_terms, seg, weights), i, seg)
                    for i, seg in enumerate(segs)
                )
            )
            chosen = sorted(
                scored[: max(1, self.max_segments_per_block)], key=lambda s: s[1]
            )
            for _score, _i, seg in chosen:
                lines.append(f"- {seg} [{block.marker}]")

        if not lines:
            return ""

        # Grow the answer line by line under the measured answer budget, so a
        # citation marker is never cut in half by truncation.
        out = _PREAMBLE
        budget = max(1, config.max_answer_tokens)
        for line in lines:
            candidate = f"{out}\n{line}"
            if tok.count(candidate) > budget and out != _PREAMBLE:
                break
            out = candidate
        return "" if out == _PREAMBLE else out


# --------------------------------------------------------------------------
# Anthropic (optional, env-gated, lazily imported)
# --------------------------------------------------------------------------


class AnthropicGenerator:
    """LLM-backed generator behind the same interface.

    Env-gated on `ANTHROPIC_API_KEY`; model from `ANYRAG_ANTHROPIC_MODEL`. The
    `anthropic` package is imported inside `_client()`, not at module scope, so
    importing `anyrag.generate` offline -- or without the package installed at
    all -- can never fail.

    It runs the *same* refusal gates before the call (a question the retrieval
    cannot support is refused without spending a token) and the same citation
    enforcement after it, so a marker the model invents is rejected exactly as
    it would be from the extractive path.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        policy: RefusalPolicy | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.model = model or os.getenv("ANYRAG_ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client_obj = client
        self._policy = policy
        self._tokenizer = tokenizer
        if self._client_obj is None and not self._api_key:
            raise ConfigError(
                "AnthropicGenerator requires ANTHROPIC_API_KEY (or an explicit "
                "api_key/client); the offline default is ExtractiveGenerator"
            )

    @staticmethod
    def available() -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def _client(self) -> Any:
        if self._client_obj is None:
            try:
                import anthropic  # noqa: PLC0415 - deliberately lazy
            except Exception as exc:  # pragma: no cover - depends on env
                raise ConfigError(
                    f"the 'anthropic' package is not importable: {exc}"
                ) from exc
            self._client_obj = anthropic.Anthropic(api_key=self._api_key)
        return self._client_obj

    def generate(
        self,
        question: str,
        hits: Sequence[Hit],
        config: GenerationConfig | None = None,
        *,
        route: QueryRoute | None = None,
        policy: RefusalPolicy | None = None,
        **kwargs: Any,
    ) -> Answer:
        config = config or GenerationConfig()
        tok = self._tokenizer or get_tokenizer()
        pol = policy or self._policy or RefusalPolicy.from_config(config)

        decision = assess(question, hits, config, policy=pol)
        trace = _base_trace(self.name, tok, decision, {"model": self.model})
        if decision.refuse:
            return Answer.refusal(
                f"{decision.code}: {decision.detail}", route=route, trace=trace
            )

        context, prompt = _pack_for_prompt(question, hits, config, tok)
        trace["packing"] = context.as_trace()
        if context.is_empty:
            return Answer.refusal(
                "context_budget_exhausted: no retrieved chunk fits the "
                f"{config.max_prompt_tokens}-token prompt budget",
                route=route,
                trace=trace,
            )

        response = self._client().messages.create(
            model=self.model,
            max_tokens=config.max_answer_tokens,
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.user}],
        )

        stop_reason = getattr(response, "stop_reason", None)
        trace["stop_reason"] = stop_reason
        if stop_reason == "refusal":
            return Answer.refusal(
                "provider_refusal: the model declined to answer",
                route=route,
                trace=trace,
            )

        text = self._text_of(response)
        if not text.strip() or REFUSAL_SENTINEL in text:
            return Answer.refusal(
                "model_declared_insufficient_context: the model reported that "
                "the packed context does not support an answer",
                route=route,
                trace=trace,
            )

        return _citations.finalize(
            text, context, question=question, route=route, trace=trace
        )

    @staticmethod
    def _text_of(response: Any) -> str:
        blocks = getattr(response, "content", None) or []
        parts: list[str] = []
        for block in blocks:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", "") or "")
            elif isinstance(block, Mapping) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

_GENERATORS = {
    "extractive": ExtractiveGenerator,
    "anthropic": AnthropicGenerator,
    "claude": AnthropicGenerator,
}


def available_generators() -> dict[str, bool]:
    """Name -> whether it can actually run in this environment."""
    return {
        "extractive": True,
        "anthropic": AnthropicGenerator.available(),
    }


def get_generator(name: str | None = None, **kwargs: Any) -> Any:
    """Return a generator by name, defaulting to the offline extractive one."""
    key = (name or "extractive").strip().lower()
    if not key:
        key = "extractive"
    cls = _GENERATORS.get(key)
    if cls is None:
        known = ", ".join(sorted(set(_GENERATORS)))
        raise ConfigError(f"unknown generator {name!r}; known generators: {known}")
    return cls(**kwargs)
