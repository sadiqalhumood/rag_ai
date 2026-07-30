"""The end-to-end pipeline, and the switches the ablation grid depends on.

Half of this file tests *negatives*: that a disabled stage did no work. Those
are asserted through spies that wrap the real indexes, because the failure mode
being guarded against -- computing dense results and then discarding them --
produces identical hit lists and is invisible from the outside.
"""

from __future__ import annotations

import itertools

import pytest

from anyrag.core.config import MetadataFilter, RetrievalConfig
from anyrag.core.errors import ConfigError
from anyrag.core.interfaces import Retriever
from anyrag.core.types import ChunkKind
from anyrag.index import FIRST_RANK, BM25Index, MemoryVectorIndex
from anyrag.retrieval import HybridRetriever
from anyrag.retrieval.expand import DeterministicExpander
from anyrag.retrieval.fusion import FusedHit
from anyrag.retrieval.prefilter import unfiltered_violations
from anyrag.retrieval.rerank import LexicalOverlapReranker, NoopReranker
from test_retrieval_fixtures import (
    BagEmbedder,
    SpyExpander,
    SpyLexicalIndex,
    SpyReranker,
    SpyVectorIndex,
    build_indexes,
    corpus,
    ids,
    make_chunk,
    many_chunks,
)

ROW = frozenset({ChunkKind.ROW})
CARD = frozenset({ChunkKind.SCHEMA_CARD})
BOTH = RetrievalConfig().chunk_kinds

QUERY = "email of customer Ahmed Al-Sayed"


def build(chunks=None, **kw) -> HybridRetriever:
    vector, lexical, embedder = build_indexes(chunks, extra_vocab=[QUERY])
    return HybridRetriever(vector, lexical, embedder, **kw)


def test_conforms_to_the_retriever_protocol() -> None:
    assert isinstance(build(), Retriever)


# --------------------------------------------------------------------------
# Stage switches -- the property the whole 18-cell grid rests on
# --------------------------------------------------------------------------


def test_dense_off_runs_no_vector_search_and_no_query_embedding() -> None:
    retriever = build()
    hits = retriever.retrieve(QUERY, RetrievalConfig(dense=False))
    assert hits
    assert retriever.vector_index.n_calls == 0
    assert retriever.embedder.query_calls == []  # not computed and discarded
    assert retriever.lexical_index.n_calls == 1
    assert retriever.trace["dense"] == {"ran": False, "searches": 0, "candidates": 0}
    assert "dense" not in retriever.trace["stages"]


def test_lexical_off_runs_no_bm25_search() -> None:
    retriever = build()
    hits = retriever.retrieve(QUERY, RetrievalConfig(lexical=False))
    assert hits
    assert retriever.lexical_index.n_calls == 0
    assert retriever.vector_index.n_calls == 1
    assert retriever.trace["lexical"]["ran"] is False


def test_hybrid_runs_both_exactly_once_per_query() -> None:
    retriever = build()
    retriever.retrieve(QUERY, RetrievalConfig())
    assert retriever.vector_index.n_calls == 1
    assert retriever.lexical_index.n_calls == 1
    assert retriever.trace["fusion"]["lists"] == 2


def test_expansion_off_never_calls_the_expander() -> None:
    expander = SpyExpander()
    retriever = build(expander=expander)
    retriever.retrieve(QUERY, RetrievalConfig(expansion=False))
    assert expander.calls == []
    assert retriever.trace["expansion"] == {"ran": False, "n_queries": 1}


def test_expansion_on_searches_every_variant() -> None:
    expander = SpyExpander(variant="email of customer Ahmad Al Sayed")
    retriever = build(expander=expander)
    retriever.retrieve(QUERY, RetrievalConfig(expansion=True))
    assert expander.calls == [QUERY]
    assert retriever.lexical_index.queries == [
        QUERY,
        "email of customer Ahmad Al Sayed",
    ]
    assert retriever.vector_index.n_calls == 2
    assert retriever.trace["expansion"]["n_queries"] == 2


def test_expansion_variants_cannot_outvote_the_original() -> None:
    # The variant is a perfect match for a different customer; the original
    # query still wins the top slot because variant lists vote at half weight.
    retriever = build(expander=SpyExpander(variant="customer Omar Farouk"))
    top = retriever.retrieve(QUERY, RetrievalConfig(expansion=True, k=3))[0]
    assert top.chunk_id == "cust:1"


def test_rerank_off_never_calls_the_reranker() -> None:
    reranker = SpyReranker()
    retriever = build(reranker=reranker)
    hits = retriever.retrieve(QUERY, RetrievalConfig(rerank=False))
    assert reranker.calls == []
    assert retriever.trace["rerank"] == {"ran": False, "reranker": "noop"}
    assert all(h.retriever == "rrf" for h in hits)


def test_rerank_on_calls_the_reranker_with_the_original_query() -> None:
    reranker = SpyReranker()
    retriever = build(reranker=reranker)
    config = RetrievalConfig(rerank=True, k=4)
    hits = retriever.retrieve(QUERY, config)
    assert len(reranker.calls) == 1
    query, n_candidates, k = reranker.calls[0]
    assert query == QUERY  # the original, not an expansion
    assert k == config.k
    assert n_candidates >= len(hits)
    assert retriever.trace["rerank"]["ran"] is True
    assert retriever.trace["rerank"]["changed_order"] is True


def test_rerank_sees_candidate_k_deep_evidence_not_k() -> None:
    reranker = SpyReranker(reverse=False)
    retriever = build(many_chunks(30), reranker=reranker)
    retriever.retrieve("customer number 7", RetrievalConfig(rerank=True, k=5,
                                                            candidate_k=20))
    _query, n_candidates, _k = reranker.calls[0]
    assert n_candidates > 5


def test_rerank_on_changes_the_answer_end_to_end() -> None:
    # c1 is a short chunk made entirely of the question's generic vocabulary,
    # which both length-normalised BM25 and a cosine over term presence adore.
    # c4 is the row that actually names Ahmed, diluted by a long note. Fusion
    # puts c1 first; the reranker sees that "ahmed" is the only query term that
    # is not in every candidate, and disagrees.
    query = "email and region and signup date of customer Ahmed"
    chunks = [
        make_chunk("c1", "customer email region signup_date"),
        make_chunk("c2", "customer Mona Hassan email mona@example.com region Cairo "
                         "signup_date is 2024-02-11"),
        make_chunk("c3", "customer Omar Farouk email omar@example.com region Giza "
                         "signup_date is 2024-07-30"),
        make_chunk("c4", "customer Ahmed Al-Sayed email ahmed@example.com region "
                         "Cairo signup_date is 2023-01-05 with notes about "
                         "deliveries preferences history complaints and other "
                         "unrelated verbiage recorded by the support team over "
                         "several years of activity"),
    ]
    retriever = build(chunks)
    off = retriever.retrieve(query, RetrievalConfig(rerank=False, k=4))
    on = retriever.retrieve(query, RetrievalConfig(rerank=True, k=4))
    assert off[0].chunk_id == "c1"
    assert on[0].chunk_id == "c4"
    assert ids(off) != ids(on)


def test_candidate_k_is_what_each_retriever_is_asked_for() -> None:
    retriever = build(many_chunks(40))
    config = RetrievalConfig(k=5, candidate_k=25)
    hits = retriever.retrieve("customer number 3", config)
    assert retriever.vector_index.calls[0]["k"] == 25
    assert retriever.lexical_index.calls[0]["k"] == 25
    assert len(hits) == 5


def test_min_score_drops_weak_hits() -> None:
    retriever = build()
    unfiltered = retriever.retrieve(QUERY, RetrievalConfig(k=8))
    threshold = unfiltered[1].score
    filtered = retriever.retrieve(QUERY, RetrievalConfig(k=8, min_score=threshold))
    assert len(filtered) < len(unfiltered)
    assert all(h.score >= threshold for h in filtered)
    assert retriever.trace["min_score"]["applied"] is True
    assert retriever.trace["min_score"]["dropped"] > 0
    assert [h.rank for h in filtered] == list(
        range(FIRST_RANK, FIRST_RANK + len(filtered))
    )


# --------------------------------------------------------------------------
# Pre-filtering
# --------------------------------------------------------------------------


def test_chunk_kinds_restriction_returns_only_the_requested_kind() -> None:
    retriever = build()
    for kinds, expected in ((ROW, ChunkKind.ROW), (CARD, ChunkKind.SCHEMA_CARD)):
        hits = retriever.retrieve(
            "customers", RetrievalConfig(chunk_kinds=kinds, k=10)
        )
        assert hits
        assert {h.chunk.kind for h in hits} == {expected}


def test_both_kinds_really_returns_both() -> None:
    hits = build().retrieve("customers", RetrievalConfig(chunk_kinds=BOTH, k=10))
    assert {h.chunk.kind for h in hits} == {ChunkKind.ROW, ChunkKind.SCHEMA_CARD}


def test_the_filter_is_pushed_into_the_index_call() -> None:
    retriever = build()
    config = RetrievalConfig(chunk_kinds=ROW)
    hits = retriever.retrieve("customers", config)
    for spy in (retriever.vector_index, retriever.lexical_index):
        pushed = spy.calls[0]["flt"]
        assert isinstance(pushed, MetadataFilter)
        assert pushed.kinds == ROW
    assert unfiltered_violations(hits, config.prefilter) == []


def test_prefiltering_still_returns_k_results_when_k_matches_exist() -> None:
    # 20 row chunks and 20 schema cards all match the query terms. Filtering
    # after retrieval would return roughly half of k here; filtering before
    # scoring returns k.
    chunks = many_chunks(20) + many_chunks(
        20, kind=ChunkKind.SCHEMA_CARD, prefix="card"
    )
    retriever = build(chunks)
    for kinds in (ROW, CARD):
        hits = retriever.retrieve(
            "customer of the shop", RetrievalConfig(chunk_kinds=kinds, k=10)
        )
        assert len(hits) == 10
        assert {h.chunk.kind for h in hits} == {next(iter(kinds))}


def test_caller_prefilter_is_honoured_alongside_chunk_kinds() -> None:
    config = RetrievalConfig(
        chunk_kinds=ROW, prefilter=MetadataFilter(tables=frozenset({"orders"})), k=10
    )
    hits = build().retrieve("customer Ahmed", config)
    assert hits
    assert {h.chunk.table for h in hits} == {"orders"}
    assert {h.chunk.kind for h in hits} == {ChunkKind.ROW}


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------


def test_empty_index_returns_no_hits_and_does_not_raise() -> None:
    embedder = BagEmbedder.over(QUERY)
    retriever = HybridRetriever(
        SpyVectorIndex(MemoryVectorIndex(dim=embedder.dim)),
        SpyLexicalIndex(BM25Index()),
        embedder,
    )
    for config in (
        RetrievalConfig(),
        RetrievalConfig(rerank=True),
        RetrievalConfig(expansion=True),
        RetrievalConfig(dense=False),
        RetrievalConfig(lexical=False),
    ):
        assert retriever.retrieve(QUERY, config) == []


def test_empty_query_returns_no_hits_and_searches_nothing() -> None:
    retriever = build()
    assert retriever.retrieve("", RetrievalConfig()) == []
    assert retriever.retrieve("   ", RetrievalConfig()) == []
    assert retriever.vector_index.n_calls == 0
    assert retriever.lexical_index.n_calls == 0
    assert retriever.trace["empty_query"] is True


def test_a_query_matching_nothing_lexically_still_returns_dense_hits() -> None:
    retriever = build()
    hits = retriever.retrieve("zzz", RetrievalConfig(k=3))
    assert isinstance(hits, list)  # no exception; may be empty


def test_missing_components_raise_rather_than_half_running() -> None:
    _vector, lexical, embedder = build_indexes()
    with pytest.raises(ConfigError):
        HybridRetriever(None, lexical, embedder).retrieve(QUERY, RetrievalConfig())
    with pytest.raises(ConfigError):
        HybridRetriever(_vector, lexical, None).retrieve(QUERY, RetrievalConfig())
    with pytest.raises(ConfigError):
        HybridRetriever(_vector, None, embedder).retrieve(QUERY, RetrievalConfig())


def test_a_lexical_only_retriever_needs_no_vector_index_at_all() -> None:
    _vector, lexical, _embedder = build_indexes()
    retriever = HybridRetriever(lexical_index=lexical)
    assert retriever.retrieve(QUERY, RetrievalConfig(dense=False))


# --------------------------------------------------------------------------
# Determinism and output shape
# --------------------------------------------------------------------------


def test_same_query_and_config_give_the_identical_hit_list_twice() -> None:
    retriever = build()
    config = RetrievalConfig(rerank=True, expansion=True, k=5)
    first = retriever.retrieve(QUERY, config)
    second = retriever.retrieve(QUERY, config)
    assert [(h.chunk_id, h.score, h.rank, h.retriever) for h in first] == [
        (h.chunk_id, h.score, h.rank, h.retriever) for h in second
    ]
    # ... and across retriever instances built from the same data.
    other = build()
    assert ids(other.retrieve(QUERY, config)) == ids(first)


def test_score_ties_break_on_chunk_id() -> None:
    # Two chunks with identical text: every retriever scores them the same, so
    # only the tie-break decides, and it must not depend on insertion order.
    text = "customer Ahmed Al-Sayed email ahmed@example.com"
    forwards = build([make_chunk("zeta", text), make_chunk("alpha", text)])
    backwards = build([make_chunk("alpha", text), make_chunk("zeta", text)])
    for retriever in (forwards, backwards):
        for config in (RetrievalConfig(), RetrievalConfig(rerank=True)):
            assert ids(retriever.retrieve(QUERY, config)) == ["alpha", "zeta"]


def test_hits_are_ranked_one_based_and_contiguous() -> None:
    for config in (RetrievalConfig(k=4), RetrievalConfig(k=4, rerank=True)):
        hits = build().retrieve(QUERY, config)
        assert len(hits) == 4
        assert [h.rank for h in hits] == [FIRST_RANK + i for i in range(4)]


def test_hits_carry_fusion_provenance() -> None:
    hits = build().retrieve(QUERY, RetrievalConfig(k=5))
    assert all(isinstance(h, FusedHit) for h in hits)
    from_both = [h for h in hits if h.n_retrievers == 2]
    assert from_both, "hybrid retrieval should agree on something"
    assert set(from_both[0].ranks) == {"dense", "bm25"}
    assert from_both[0].ranks["dense"] >= FIRST_RANK


def test_scores_are_descending_and_bounded() -> None:
    for config in (RetrievalConfig(k=6), RetrievalConfig(k=6, rerank=True)):
        hits = build().retrieve(QUERY, config)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 < s <= 1.0 for s in scores)


def test_query_vectors_are_cached_across_configs() -> None:
    retriever = build()
    for _ in range(3):
        retriever.retrieve(QUERY, RetrievalConfig())
    assert retriever.embedder.query_calls == [QUERY]
    retriever.clear_cache()
    retriever.retrieve(QUERY, RetrievalConfig())
    assert retriever.embedder.query_calls == [QUERY, QUERY]


def test_caching_does_not_change_results() -> None:
    cached = build().retrieve(QUERY, RetrievalConfig(k=5))
    uncached = build(cache_queries=False).retrieve(QUERY, RetrievalConfig(k=5))
    assert [(h.chunk_id, h.score) for h in cached] == [
        (h.chunk_id, h.score) for h in uncached
    ]


# --------------------------------------------------------------------------
# Trace
# --------------------------------------------------------------------------


def test_trace_names_the_stages_that_ran_and_counts_candidates() -> None:
    import json

    retriever = build()
    config = RetrievalConfig(rerank=True, expansion=True, k=3)
    hits, trace = retriever.retrieve_with_trace(QUERY, config)

    assert trace["config"] == config.name
    assert trace["stages"] == ["expand", "dense", "lexical", "fuse", "rerank", "topk"]
    assert trace["dense"]["searches"] == trace["expansion"]["n_queries"]
    assert trace["dense"]["candidates"] > 0
    assert trace["lexical"]["candidates"] > 0
    assert trace["fusion"]["method"] == "rrf"
    assert trace["fusion"]["candidates"] >= len(hits)
    assert trace["returned"] == len(hits)
    assert trace["prefilter"]["kinds"] is not None
    json.dumps(trace)  # the harness writes this to a results file


def test_trace_is_the_same_object_as_the_retrievers_attribute() -> None:
    retriever = build()
    _hits, trace = retriever.retrieve_with_trace(QUERY, RetrievalConfig())
    assert retriever.trace is trace


def test_trace_records_a_reranker_fallback() -> None:
    class FallingBack:
        name = "cross_encoder"
        info = {"fell_back": True, "reason": "no model", "degraded": True}

        def rerank(self, query, hits, k=10):
            return LexicalOverlapReranker().rerank(query, hits, k)

    retriever = build(reranker=FallingBack())
    retriever.retrieve(QUERY, RetrievalConfig(rerank=True))
    assert retriever.trace["rerank"]["fell_back"] is True
    assert retriever.trace["rerank"]["info"]["reason"] == "no model"


# --------------------------------------------------------------------------
# The grid itself
# --------------------------------------------------------------------------


def test_every_cell_of_the_ablation_grid_runs_and_is_distinctly_named() -> None:
    retriever = build(corpus() + many_chunks(12))
    cells = [
        RetrievalConfig(dense=dense, lexical=lexical, rerank=rerank,
                        chunk_kinds=kinds, k=5)
        for (dense, lexical), rerank, kinds in itertools.product(
            [(True, False), (False, True), (True, True)],
            [False, True],
            [ROW, CARD, BOTH],
        )
    ]
    assert len(cells) == 18
    assert len({c.name for c in cells}) == 18

    names = set()
    for config in cells:
        hits = retriever.retrieve("customer of the shop", config)
        assert len(hits) <= config.k
        assert all(h.chunk.kind in config.chunk_kinds for h in hits)
        names.add(retriever.trace["config"])
        assert retriever.trace["dense"]["ran"] is config.dense
        assert retriever.trace["lexical"]["ran"] is config.lexical
        assert retriever.trace["rerank"]["ran"] is config.rerank
    assert len(names) == 18


def test_dense_and_bm25_cells_do_not_return_the_same_thing() -> None:
    # If the two arms could not disagree, the hybrid axis of the grid would be
    # measuring nothing. `BagEmbedder` is lexical by construction, so the
    # disagreement is staged: the dense index is loaded with vectors for a
    # paraphrase of ord:2 rather than for its own text, which is what a real
    # embedder does when it matches meaning the literal tokens do not carry.
    chunks = corpus()
    embedder = BagEmbedder.over(*[c.text for c in chunks], QUERY)
    aliases = {"ord:2": QUERY}
    vector = MemoryVectorIndex(dim=embedder.dim)
    vector.upsert(
        chunks,
        embedder.embed([aliases.get(c.chunk_id, c.text) for c in chunks]),
    )
    lexical = BM25Index()
    lexical.upsert(chunks)
    retriever = HybridRetriever(
        SpyVectorIndex(vector), SpyLexicalIndex(lexical), embedder
    )

    dense_only = ids(retriever.retrieve(QUERY, RetrievalConfig(lexical=False, k=4)))
    bm25_only = ids(retriever.retrieve(QUERY, RetrievalConfig(dense=False, k=4)))
    hybrid = ids(retriever.retrieve(QUERY, RetrievalConfig(k=4)))
    assert dense_only[0] == "ord:2"
    assert bm25_only[0] == "cust:1"
    assert dense_only != bm25_only
    # Fusion is not just one arm wearing a hat.
    assert hybrid != dense_only and hybrid != bm25_only


def test_default_components_are_the_offline_defaults() -> None:
    retriever = build()
    assert isinstance(retriever.reranker, LexicalOverlapReranker)
    assert isinstance(retriever.expander, DeterministicExpander)
    assert isinstance(retriever._noop_reranker, NoopReranker)
    assert retriever.fusion == "rrf"


def test_score_fusion_can_be_selected_without_touching_the_config() -> None:
    hits = build(fusion="norm").retrieve(QUERY, RetrievalConfig(k=3))
    assert hits and all(h.retriever == "norm" for h in hits)
