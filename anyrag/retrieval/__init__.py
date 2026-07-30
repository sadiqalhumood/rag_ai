"""Retrieval: hybrid dense + BM25 search with fusion, filtering and reranking.

    expand -> (dense || lexical) -> prefilter -> RRF fuse -> rerank -> top-k

`HybridRetriever` runs that pipeline; every stage is switched by a field of
`anyrag.core.config.RetrievalConfig`, and switching a stage off means it does
not run. The eval harness sweeps an 18-cell grid over
{dense, bm25, hybrid} x {rerank on/off} x {row-chunks, schema-cards, both}, so a
stage that could not be turned off -- or that ran and discarded its results --
would make whole rows of that grid meaningless.

Defaults, and why
-----------------
* **Fusion is RRF** (`fusion.rrf`). Dense cosine and BM25 scores live on
  incomparable scales; rank is the only quantity both retrievers agree on.
  `fusion.normalized_score_fusion` is the alternative, so the choice is
  measurable rather than merely asserted.
* **Fused scores are normalised to ``(0, 1]``** by dividing by the maximum
  attainable score. That is division by a constant, so ordering is untouched,
  but it keeps the absolute thresholds downstream (`min_score`,
  `min_support_score`) meaning the same thing in every cell of the grid.
* **Reranking is `LexicalOverlapReranker`**: IDF-weighted query-term coverage
  over the candidate set plus contiguous-bigram evidence, blended with the fused
  score. Real signal, deterministic, no model download.
* **Expansion is rule-based and off by default**, with variants weighted at half
  the original query's vote.
* **Ties break on `chunk_id`** at every stage, as they already do inside the
  indexes, so ablation cells are reproducible run to run.

`Hit.rank` is 1-based throughout (`anyrag.index.FIRST_RANK`), so RRF is
``1 / (rrf_k + rank)`` with no off-by-one correction.
"""

from __future__ import annotations

from .expand import (
    DEFAULT_EXPANDER,
    EXPANDERS,
    TRANSLITERATIONS,
    DeterministicExpander,
    NoopExpander,
    get_expander,
)
from .fusion import (
    DEFAULT_FUSION,
    DEFAULT_RRF_K,
    FUSIONS,
    NORM_RETRIEVER,
    RRF_RETRIEVER,
    FusedHit,
    fuse,
    normalized_score_fusion,
    rrf,
)
from .pipeline import (
    DEFAULT_VARIANT_WEIGHT,
    DENSE_LABEL,
    LEXICAL_LABEL,
    HybridRetriever,
)
from .prefilter import (
    ALL_KINDS,
    build_filter,
    coerce_kinds,
    describe_filter,
    merge_filters,
    unfiltered_violations,
)
from .rerank import (
    CROSS_ENCODER_MODEL,
    DEFAULT_RERANKER,
    RERANK_RETRIEVER,
    RERANKERS,
    CrossEncoderReranker,
    LexicalOverlapReranker,
    NoopReranker,
    get_reranker,
)

__all__ = [
    # pipeline
    "HybridRetriever",
    "DENSE_LABEL",
    "LEXICAL_LABEL",
    "DEFAULT_VARIANT_WEIGHT",
    # fusion
    "rrf",
    "normalized_score_fusion",
    "fuse",
    "FusedHit",
    "FUSIONS",
    "DEFAULT_FUSION",
    "DEFAULT_RRF_K",
    "RRF_RETRIEVER",
    "NORM_RETRIEVER",
    # prefilter
    "build_filter",
    "merge_filters",
    "coerce_kinds",
    "describe_filter",
    "unfiltered_violations",
    "ALL_KINDS",
    # rerank
    "NoopReranker",
    "LexicalOverlapReranker",
    "CrossEncoderReranker",
    "get_reranker",
    "RERANKERS",
    "DEFAULT_RERANKER",
    "RERANK_RETRIEVER",
    "CROSS_ENCODER_MODEL",
    # expand
    "NoopExpander",
    "DeterministicExpander",
    "get_expander",
    "EXPANDERS",
    "DEFAULT_EXPANDER",
    "TRANSLITERATIONS",
]
