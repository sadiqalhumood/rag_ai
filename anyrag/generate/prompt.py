"""Prompt assembly: context blocks, citation markers, and refusal instructions.

The marker-to-chunk mapping is carried as data on `PackedContext.marker_map` and
travels with the `Prompt`. Nothing downstream re-parses it out of the rendered
prose -- a citation is resolved by looking up an integer in a dict, so a marker
the generator invented has nowhere to resolve to and is rejected structurally
rather than by pattern-matching.

`overhead_tokens` exists so the packer can be told, before it packs anything,
exactly how much of `max_prompt_tokens` the scaffolding and the question will
consume. The generator then verifies the assembled prompt against the same
budget and re-packs if a BPE merge at a block boundary moved the count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..core.tokenizer import Tokenizer, get_tokenizer
from .packer import PackedContext

__all__ = [
    "Prompt",
    "build_prompt",
    "overhead_tokens",
    "SYSTEM_PROMPT",
    "REFUSAL_SENTINEL",
]

#: The exact string a model must emit when the context does not support an
#: answer. Checked literally, so it cannot be confused with hedging prose.
REFUSAL_SENTINEL = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""\
You answer questions strictly from the numbered context blocks you are given.

Rules, in priority order:

1. Every factual claim in your answer must end with the citation marker of the
   block it came from, written exactly as [1], [2], and so on. A sentence with
   no marker is not allowed.
2. Only cite markers that actually appear in the context below. Never invent a
   marker, never cite a block number you were not given, and never merge two
   blocks under one marker.
3. If the context does not contain the answer, reply with exactly
   {REFUSAL_SENTINEL} and nothing else. Do not guess, do not answer from prior
   knowledge, and do not offer a plausible-sounding substitute. A missing
   customer, product or date is a reason to refuse, not to answer about a
   similar one.
4. Quote values as they appear in the context. Do not compute, convert,
   round or reformat them.
5. Be brief. Surface the supporting field values and stop.\
"""

USER_TEMPLATE = """\
Question:
{question}

Context blocks:
{context}

Answer the question using only the blocks above, citing markers. If they do not
contain the answer, reply with exactly {sentinel}.\
"""

#: Small pad added to the measured overhead. The scaffolding is measured
#: exactly, but concatenating context into it can merge tokens at the seam; the
#: generator re-verifies the full prompt regardless, so this only saves a
#: re-pack round in the common case.
_SEAM_PAD = 8


@dataclass(frozen=True)
class Prompt:
    """A rendered prompt plus the marker mapping needed to validate citations."""

    system: str
    user: str
    context: PackedContext
    question: str = ""
    marker_map: Mapping[int, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """System and user text joined, for a single budget measurement."""
        return f"{self.system}\n\n{self.user}"

    def token_count(self, tokenizer: Tokenizer | None = None) -> int:
        return (tokenizer or get_tokenizer()).count(self.text)

    def as_trace(self) -> dict[str, Any]:
        return {"marker_map": dict(self.marker_map), "question": self.question}


def build_prompt(question: str, context: PackedContext) -> Prompt:
    return Prompt(
        system=SYSTEM_PROMPT,
        user=USER_TEMPLATE.format(
            question=question.strip(),
            context=context.text,
            sentinel=REFUSAL_SENTINEL,
        ),
        context=context,
        question=question,
        marker_map=context.marker_map,
    )


def overhead_tokens(question: str, *, tokenizer: Tokenizer | None = None) -> int:
    """Measured cost of everything in the prompt that is not context.

    Measured with the real tokenizer on the real template, not approximated
    from a character count.
    """
    tok = tokenizer or get_tokenizer()
    skeleton = build_prompt(question, PackedContext())
    return tok.count(skeleton.text) + _SEAM_PAD
