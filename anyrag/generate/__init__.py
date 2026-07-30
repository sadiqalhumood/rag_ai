"""Answer generation: packing, prompting, citation enforcement, refusal.

The pipeline is four steps, and three of them can end in a refusal:

    assess(question, hits, config)   -> refuse when the evidence is too weak
    pack(hits, config)               -> refuse when nothing fits the budget
    build_prompt(question, packed)   -> markers carried as data, not prose
    finalize(text, packed)           -> refuse when nothing valid was cited

`get_generator()` returns the offline, deterministic `ExtractiveGenerator` by
default. Importing this package never touches the network and never requires the
`anthropic` package to be installed.
"""

from __future__ import annotations

from .citations import CitationResult, finalize, parse_markers, resolve
from .generator import (
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicGenerator,
    ExtractiveGenerator,
    available_generators,
    get_generator,
)
from .packer import PackedChunk, PackedContext, pack, render_block
from .prompt import REFUSAL_SENTINEL, SYSTEM_PROMPT, Prompt, build_prompt, overhead_tokens
from .refusal import (
    RefusalDecision,
    RefusalPolicy,
    TermWeights,
    assess,
    entity_terms,
    term_weights,
    terms,
    weighted_overlap,
)

__all__ = [
    # packing
    "PackedChunk",
    "PackedContext",
    "pack",
    "render_block",
    # prompting
    "Prompt",
    "build_prompt",
    "overhead_tokens",
    "SYSTEM_PROMPT",
    "REFUSAL_SENTINEL",
    # citations
    "CitationResult",
    "finalize",
    "parse_markers",
    "resolve",
    # refusal
    "RefusalDecision",
    "RefusalPolicy",
    "TermWeights",
    "assess",
    "entity_terms",
    "term_weights",
    "terms",
    "weighted_overlap",
    # generators
    "ExtractiveGenerator",
    "AnthropicGenerator",
    "get_generator",
    "available_generators",
    "DEFAULT_ANTHROPIC_MODEL",
]
