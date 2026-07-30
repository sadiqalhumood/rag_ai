"""The retrieval pipeline: expand -> (dense || lexical) -> fuse -> rerank -> top-k.

    expand        config.expansion   deterministic query variants, no LLM
    dense         config.dense       vector search, pre-filtered in the index
    lexical       config.lexical     BM25 search, pre-filtered in the index
    prefilter     config.chunk_kinds pushed *into* both searches, never after
    fuse          config.rrf_k       reciprocal rank fusion over every list
    min_score     config.min_score   absolute floor on the fused score
    rerank        config.rerank      lexical-overlap by default
    top-k         config.k           the cut, taken last

Every stage is switchable from `RetrievalConfig` and switching one off means the
work is *not done*, not that it is done and discarded. ``dense=False`` performs
no vector search and no query embedding at all; the same holds for `lexical`,
`expansion` and `rerank`. This is the property the 18-cell ablation grid rests
on: a "bm25 only" cell that quietly ran and threw away the dense results would
report dense-influenced latency and identical-looking traces, and the grid would
be measuring one system eighteen times.

Candidates and the cut
----------------------
Each enabled retriever is asked for `candidate_k` hits (per query variant),
everything is fused, and only the final list is cut to `k`. Reranking therefore
sees `candidate_k`-deep evidence rather than `k`-deep, which is the only reason
a reranker can add anything.

Pre-filtering
-------------
`chunk_kinds` and any caller `prefilter` are merged into a single
`MetadataFilter` and passed *into* `index.search()`. The indexes narrow their
candidate set with posting lists before scoring, so a filtered search still
returns `k` hits when `k` matching chunks exist. Filtering the result list
instead would return short lists and quietly deflate recall@k in exactly the
filtered cells of the ablation.

Trace
-----
`retrieve` records a JSON-safe `trace` dict on the retriever (also returned by
`retrieve_with_trace`) naming the stages that ran, the number of candidates each
produced, the query variants used, and whether a reranker fell back. The eval
harness reports it, and it is how "this cell really did skip dense retrieval"
is checked from outside.

Determinism
-----------
Ties break on `chunk_id` at every stage -- index, fusion, reranker -- so the
same query and config produce a bit-identical hit list every time and ablation
cells are reproducible.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from anyrag.core.config import MetadataFilter, RetrievalConfig
from anyrag.core.errors import ConfigError
from anyrag.core.types import Hit

from .expand import DeterministicExpander, NoopExpander
from .fusion import DEFAULT_FUSION, FusedHit, fuse
from .prefilter import build_filter, describe_filter
from .rerank import LexicalOverlapReranker, NoopReranker

__all__ = [
    "DENSE_LABEL",
    "LEXICAL_LABEL",
    "DEFAULT_VARIANT_WEIGHT",
    "HybridRetriever",
]

#: Fusion labels. These match `Hit.retriever` as the indexes set it, so a trace
#: and a `FusedHit.ranks` mapping talk about retrievers by the same names.
DENSE_LABEL = "dense"
LEXICAL_LABEL = "bm25"

#: Weight applied to results retrieved for an *expanded* query relative to the
#: original. Expansion should be able to rescue a result the original query
#: missed without being able to outvote the original -- half a vote does that
#: and keeps a bad expansion's damage bounded.
DEFAULT_VARIANT_WEIGHT = 0.5

#: Bound on the query-vector cache. The grid asks the same ~300 questions in 18
#: configurations; re-embedding each one 18 times is pure waste, and the cache
#: cannot change results because embedding is deterministic.
MAX_CACHED_QUERIES = 4096


class HybridRetriever:
    """Hybrid dense + lexical retriever with per-stage switches.

    Only the components a config actually uses need to exist: a BM25-only
    retriever can be built with ``HybridRetriever(lexical_index=idx)``. Asking
    for a stage whose component is missing raises `ConfigError` at retrieve
    time rather than silently returning fewer results -- an ablation cell that
    half ran is worse than one that failed outright.
    """

    def __init__(
        self,
        vector_index: Any | None = None,
        lexical_index: Any | None = None,
        embedder: Any | None = None,
        reranker: Any | None = None,
        expander: Any | None = None,
        *,
        fusion: str = DEFAULT_FUSION,
        variant_weight: float = DEFAULT_VARIANT_WEIGHT,
        cache_queries: bool = True,
    ) -> None:
        self.vector_index = vector_index
        self.lexical_index = lexical_index
        self.embedder = embedder
        #: Reranker used when ``config.rerank`` is true. Defaults to the
        #: deterministic offline reranker rather than to nothing, so
        #: "rerank on" is a real condition even for a caller that passed none.
        self.reranker = reranker if reranker is not None else LexicalOverlapReranker()
        #: Expander used when ``config.expansion`` is true, same reasoning.
        self.expander = expander if expander is not None else DeterministicExpander()
        self.fusion = fusion
        self.variant_weight = float(variant_weight)
        self.cache_queries = bool(cache_queries)
        self._noop_reranker = NoopReranker()
        self._noop_expander = NoopExpander()
        self._vector_cache: dict[str, Any] = {}
        #: Trace of the most recent `retrieve` call.
        self.trace: dict[str, Any] = {}

    # -- introspection ----------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"HybridRetriever(vector={type(self.vector_index).__name__}, "
            f"lexical={type(self.lexical_index).__name__}, fusion={self.fusion!r})"
        )

    def clear_cache(self) -> None:
        self._vector_cache.clear()

    # -- the pipeline -----------------------------------------------------

    def retrieve(self, query: str, config: RetrievalConfig | None = None) -> list[Hit]:
        """Top-`config.k` hits for `query`. See `retrieve_with_trace`."""
        hits, _trace = self.retrieve_with_trace(query, config)
        return hits

    def retrieve_with_trace(
        self, query: str, config: RetrievalConfig | None = None
    ) -> tuple[list[Hit], dict[str, Any]]:
        """Retrieve, and return the per-stage trace alongside the hits."""
        config = config or RetrievalConfig()
        started = time.perf_counter()
        trace: dict[str, Any] = {
            "config": config.name,
            "k": config.k,
            "candidate_k": config.candidate_k,
            "stages": [],
        }
        self.trace = trace
        self._check_components(config)

        if not query or not query.strip():
            trace["empty_query"] = True
            trace["returned"] = 0
            trace["latency_ms"] = (time.perf_counter() - started) * 1000.0
            return [], trace

        queries = self._expand(query, config, trace)
        flt = build_filter(config)
        trace["prefilter"] = describe_filter(flt)

        lists, labels, weights = self._search(queries, config, flt, trace)
        fused = self._fuse(lists, labels, weights, config, trace)
        kept = self._apply_min_score(fused, config, trace)
        hits = self._rerank(query, kept, config, trace)

        trace["returned"] = len(hits)
        trace["latency_ms"] = (time.perf_counter() - started) * 1000.0
        return hits, trace

    # -- stages -----------------------------------------------------------

    def _check_components(self, config: RetrievalConfig) -> None:
        if config.dense:
            if self.vector_index is None:
                raise ConfigError(
                    "config.dense is True but this retriever has no vector index"
                )
            if self.embedder is None:
                raise ConfigError(
                    "config.dense is True but this retriever has no embedder"
                )
        if config.lexical and self.lexical_index is None:
            raise ConfigError(
                "config.lexical is True but this retriever has no lexical index"
            )

    def _expand(
        self, query: str, config: RetrievalConfig, trace: dict[str, Any]
    ) -> list[str]:
        """Query variants, original first. Off means the expander is not called."""
        if not config.expansion:
            trace["expansion"] = {"ran": False, "n_queries": 1}
            return [query]
        expander = self.expander or self._noop_expander
        queries = list(expander.expand(query))
        if not queries or queries[0] != query:
            # An expander that drops or reorders the original would silently
            # change what "the query" means; put it back rather than trusting it.
            queries = [query] + [q for q in queries if q != query]
        trace["stages"].append("expand")
        entry: dict[str, Any] = {
            "ran": True,
            "expander": getattr(expander, "name", type(expander).__name__),
            "n_queries": len(queries),
            "variants": queries[1:],
            "variant_weight": self.variant_weight,
        }
        explain = getattr(expander, "explain", None)
        if callable(explain):
            entry["rules"] = [rule for rule, _variant in explain(query)]
        trace["expansion"] = entry
        return queries

    def _embed(self, query: str) -> Any:
        if not self.cache_queries:
            return self.embedder.embed_query(query)
        cached = self._vector_cache.get(query)
        if cached is None:
            cached = self.embedder.embed_query(query)
            if len(self._vector_cache) >= MAX_CACHED_QUERIES:
                self._vector_cache.clear()
            self._vector_cache[query] = cached
        return cached

    def _search(
        self,
        queries: Sequence[str],
        config: RetrievalConfig,
        flt: MetadataFilter | None,
        trace: dict[str, Any],
    ) -> tuple[list[list[Hit]], list[str], list[float]]:
        lists: list[list[Hit]] = []
        labels: list[str] = []
        weights: list[float] = []
        dense_label = getattr(self.vector_index, "retriever", DENSE_LABEL)
        lex_label = getattr(self.lexical_index, "retriever", LEXICAL_LABEL)
        dense_hits = lex_hits = 0
        dense_calls = lex_calls = 0

        for i, q in enumerate(queries):
            weight = 1.0 if i == 0 else self.variant_weight
            if config.dense:
                hits = list(
                    self.vector_index.search(
                        self._embed(q), k=config.candidate_k, flt=flt
                    )
                )
                dense_calls += 1
                dense_hits += len(hits)
                lists.append(hits)
                labels.append(dense_label)
                weights.append(weight)
            if config.lexical:
                hits = list(
                    self.lexical_index.search(q, k=config.candidate_k, flt=flt)
                )
                lex_calls += 1
                lex_hits += len(hits)
                lists.append(hits)
                labels.append(lex_label)
                weights.append(weight)

        if config.dense:
            trace["stages"].append("dense")
        if config.lexical:
            trace["stages"].append("lexical")
        trace["dense"] = {
            "ran": bool(config.dense),
            "searches": dense_calls,
            "candidates": dense_hits,
        }
        trace["lexical"] = {
            "ran": bool(config.lexical),
            "searches": lex_calls,
            "candidates": lex_hits,
        }
        return lists, labels, weights

    def _fuse(
        self,
        lists: Sequence[Sequence[Hit]],
        labels: Sequence[str],
        weights: Sequence[float],
        config: RetrievalConfig,
        trace: dict[str, Any],
    ) -> list[FusedHit]:
        fused = fuse(
            self.fusion,
            lists,
            k=config.rrf_k,
            labels=list(labels),
            weights=list(weights),
        )
        trace["stages"].append("fuse")
        trace["fusion"] = {
            "method": self.fusion,
            "rrf_k": config.rrf_k,
            "lists": len(lists),
            "candidates": len(fused),
            "from_both": sum(1 for h in fused if h.n_retrievers > 1),
        }
        return fused

    @staticmethod
    def _apply_min_score(
        fused: Sequence[FusedHit], config: RetrievalConfig, trace: dict[str, Any]
    ) -> list[FusedHit]:
        """Drop hits below the absolute floor, before reranking.

        The floor is defined on the *fused* score (`RetrievalConfig.min_score`
        says so), which also means it behaves identically whether or not
        reranking is enabled -- a threshold that moved with the reranker would
        confound the rerank axis of the grid with a recall change.
        """
        if config.min_score <= 0.0:
            trace["min_score"] = {"applied": False, "dropped": 0}
            return list(fused)
        kept = [h for h in fused if h.score >= config.min_score]
        trace["stages"].append("min_score")
        trace["min_score"] = {
            "applied": True,
            "threshold": config.min_score,
            "dropped": len(fused) - len(kept),
        }
        return kept

    def _rerank(
        self,
        query: str,
        hits: Sequence[Hit],
        config: RetrievalConfig,
        trace: dict[str, Any],
    ) -> list[Hit]:
        if not config.rerank:
            out = self._noop_reranker.rerank(query, hits, config.k)
            trace["stages"].append("topk")
            trace["rerank"] = {"ran": False, "reranker": self._noop_reranker.name}
            return out

        reranker = self.reranker or LexicalOverlapReranker()
        before = [h.chunk_id for h in hits][: config.k]
        out = list(reranker.rerank(query, hits, config.k))
        entry: dict[str, Any] = {
            "ran": True,
            "reranker": getattr(reranker, "name", type(reranker).__name__),
            "candidates": len(hits),
            "changed_order": [h.chunk_id for h in out] != before,
        }
        info = getattr(reranker, "info", None)
        if isinstance(info, Mapping):
            entry["info"] = dict(info)
            entry["fell_back"] = bool(info.get("fell_back"))
        trace["stages"].extend(["rerank", "topk"])
        trace["rerank"] = entry
        return out
