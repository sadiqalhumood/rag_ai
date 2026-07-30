"""Configuration objects.

Every retrieval stage is individually toggleable because the eval harness
ablates them; that requirement drives the shape of RetrievalConfig more than
ergonomics does.

No credentials live here. Connection strings come from environment variables
(see .env.example); config carries only the *name* of a source URI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .errors import ConfigError
from .types import ChunkKind


@dataclass(frozen=True)
class MetadataFilter:
    """Pre-filter applied before scoring.

    `equals` matches chunk.meta exactly; `kinds` restricts chunk kinds;
    `tables` restricts to named tables. All conditions are ANDed.
    """

    kinds: frozenset[ChunkKind] | None = None
    tables: frozenset[str] | None = None
    source_ids: frozenset[str] | None = None
    equals: Mapping[str, Any] = field(default_factory=dict)

    def matches(self, chunk) -> bool:  # noqa: ANN001 - avoids import cycle
        if self.kinds is not None and chunk.kind not in self.kinds:
            return False
        if self.source_ids is not None and chunk.source_id not in self.source_ids:
            return False
        if self.tables is not None and chunk.meta.get("table") not in self.tables:
            return False
        for key, val in self.equals.items():
            if chunk.meta.get(key) != val:
                return False
        return True

    @property
    def is_empty(self) -> bool:
        return (
            self.kinds is None
            and self.tables is None
            and self.source_ids is None
            and not self.equals
        )


@dataclass(frozen=True)
class RetrievalConfig:
    """One cell of the ablation grid.

    dense/lexical/expansion/rerank are the four switches the harness sweeps;
    `chunk_kinds` is the third axis (row-chunks, schema-cards, or both).
    """

    dense: bool = True
    lexical: bool = True
    expansion: bool = False
    rerank: bool = False
    k: int = 10
    #: Candidates pulled from each retriever before fusion.
    candidate_k: int = 50
    #: Reciprocal-rank-fusion damping constant.
    rrf_k: int = 60
    chunk_kinds: frozenset[ChunkKind] = frozenset(
        {ChunkKind.ROW, ChunkKind.SCHEMA_CARD, ChunkKind.SQL_RESULT}
    )
    prefilter: MetadataFilter | None = None
    #: Minimum fused score for a hit to be considered supporting evidence.
    min_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.dense and not self.lexical:
            raise ConfigError(
                "at least one of dense/lexical retrieval must be enabled"
            )
        if self.k <= 0 or self.candidate_k <= 0:
            raise ConfigError("k and candidate_k must be positive")

    @property
    def name(self) -> str:
        """Stable identifier used for results filenames and table rows."""
        if self.dense and self.lexical:
            mode = "hybrid"
        elif self.dense:
            mode = "dense"
        else:
            mode = "bm25"
        # Any unmapped set MUST get its own name. Collapsing unknown sets to
        # "both" silently merged distinct ablation cells: a grid using
        # {ROW, SQL_RESULT} for its row cell named itself identically to the
        # both-kinds cell, and 18 cells became 12 with results files
        # overwriting each other. Names are identity here, not decoration.
        canonical = {
            frozenset({ChunkKind.ROW}): "row",
            frozenset({ChunkKind.SCHEMA_CARD}): "card",
            frozenset({ChunkKind.ROW, ChunkKind.SCHEMA_CARD}): "both",
        }
        kinds = canonical.get(self.chunk_kinds)
        if kinds is None:
            codes = {
                ChunkKind.ROW: "row",
                ChunkKind.SCHEMA_CARD: "card",
                ChunkKind.SQL_RESULT: "sql",
            }
            kinds = "+".join(sorted(codes[k] for k in self.chunk_kinds)) or "nokinds"
        bits = [mode, "rerank" if self.rerank else "norerank", kinds]
        if self.expansion:
            bits.append("expand")
        return "_".join(bits)

    def with_(self, **kw: Any) -> "RetrievalConfig":
        return replace(self, **kw)


@dataclass(frozen=True)
class GenerationConfig:
    """Token budget and refusal policy."""

    #: Total budget for the assembled prompt, measured with a real tokenizer.
    max_prompt_tokens: int = 3000
    #: Ceiling for any single chunk; longer chunks are truncated at a token
    #: boundary rather than dropped, so a huge row still contributes something.
    max_chunk_tokens: int = 512
    max_answer_tokens: int = 512
    #: A question is refused when no retrieved chunk clears this support score.
    min_support_score: float = 0.02
    #: Minimum number of supporting chunks required to answer.
    min_support_chunks: int = 1
    #: Refuse when the best hit's idf-weighted overlap with the question is
    #: below this.
    #:
    #: Calibrated for the *weighted* metric, not plain token overlap. Plain
    #: overlap does not discriminate: a distractor asking for a nonexistent
    #: customer scores ~0.50 purely on generic schema words ("email",
    #: "customer"), which is above any threshold low enough to admit real
    #: questions. Weighting terms by inverse document frequency over the
    #: retrieved set drops that distractor to ~0.29 while an answerable twin
    #: stays at ~1.0, so the gap sits either side of 0.35.
    min_overlap: float = 0.35
    #: How many top hits the overlap gate examines before giving up.
    overlap_top_k: int = 5
    #: Fraction of value-like question tokens (names, ids, numbers) that must
    #: actually appear in the retrieved text. Defence in depth behind
    #: min_overlap: catches a distractor that borrows enough schema vocabulary
    #: to clear the overlap gate.
    min_entity_coverage: float = 0.75
    #: Toggle for the entity gate, so the eval can ablate it in isolation.
    require_entity_coverage: bool = True
    #: Minimum length for a token in a caseless script (Arabic, CJK) to count
    #: as an entity. Without a floor, common function words are treated as
    #: entities and legitimate questions are refused.
    caseless_entity_min_length: int = 4
    generator: str = "extractive"

    def __post_init__(self) -> None:
        # Deliberately NOT rejecting a max_prompt_tokens below the instruction
        # scaffold. It was tempting -- such a budget can only ever refuse -- but
        # "every chunk is too large, so refuse gracefully instead of crashing"
        # is a behaviour worth testing, and a constructor guard would make that
        # test unwritable.
        if not 0.0 <= self.min_overlap <= 1.0:
            raise ConfigError("min_overlap must be in [0, 1]")
        if not 0.0 <= self.min_entity_coverage <= 1.0:
            raise ConfigError("min_entity_coverage must be in [0, 1]")


@dataclass(frozen=True)
class AnyRagConfig:
    source_uri: str = ""
    embedder: str = "auto"
    index_dir: str | None = None
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    row_chunks: bool = True
    schema_cards: bool = True

    @classmethod
    def from_env(cls, **overrides: Any) -> "AnyRagConfig":
        base = cls(
            source_uri=os.getenv("ANYRAG_SOURCE_URI", ""),
            embedder=os.getenv("ANYRAG_EMBEDDER", "auto"),
            index_dir=os.getenv("ANYRAG_INDEX_DIR") or None,
        )
        return replace(base, **overrides) if overrides else base
