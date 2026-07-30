"""Hand-computed cases for every metric.

Nothing here compares the implementation against itself. Each expected value is
either a small exact fraction or is assembled from two standard constants:

    1 / log2(2) = 1.0
    1 / log2(3) = 0.6309297535714574
    1 / log2(5) = 0.43067655807339306

A wrong nDCG invalidates the entire report and would not be visible in any
end-to-end test, so it is pinned here in the most explicit way available.
"""

from __future__ import annotations

import math

import pytest

from anyrag.core.types import Chunk, ChunkKind, Citation, Hit, QueryRoute, RowRef

from evals import metrics
from evals.metrics import GoldTarget, QuestionResult

INV_LOG2_2 = 1.0
INV_LOG2_3 = 0.6309297535714574
INV_LOG2_5 = 0.43067655807339306


def row_chunk(table: str, pk: str, cid: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=cid or f"{table}-{pk}",
        source_id="test",
        kind=ChunkKind.ROW,
        text=f"{table} {pk}",
        row_refs=(RowRef(table=table, pk=pk),),
        meta={"table": table},
    )


def card_chunk(table: str) -> Chunk:
    return Chunk(
        chunk_id=f"card-{table}",
        source_id="test",
        kind=ChunkKind.SCHEMA_CARD,
        text=f"schema for {table}",
        row_refs=(),
        meta={"table": table},
    )


def hits(*chunks: Chunk) -> list[Hit]:
    return [Hit(chunk=c, score=1.0 - i * 0.01, rank=i + 1) for i, c in enumerate(chunks)]


# --------------------------------------------------------------------------
# The constants themselves
# --------------------------------------------------------------------------


def test_log_constants_are_what_the_docstring_claims():
    assert 1 / math.log2(2) == pytest.approx(INV_LOG2_2, abs=1e-15)
    assert 1 / math.log2(3) == pytest.approx(INV_LOG2_3, abs=1e-15)
    assert 1 / math.log2(5) == pytest.approx(INV_LOG2_5, abs=1e-15)


# --------------------------------------------------------------------------
# GoldTarget
# --------------------------------------------------------------------------


def test_gold_keys_cover_rows_and_schema_cards():
    gold = GoldTarget(
        row_refs=frozenset({RowRef("customers", "1"), RowRef("orders", "9")}),
        schema_tables=frozenset({"products"}),
    )
    assert gold.n_gold == 3
    assert gold.keys == {("row", "customers", "1"), ("row", "orders", "9"), ("schema", "products")}


def test_schema_card_is_relevant_only_for_its_own_table():
    gold = GoldTarget.of_schema(["orders"])
    assert gold.is_relevant(card_chunk("orders"))
    assert not gold.is_relevant(card_chunk("customers"))
    # A *row* of the gold table is not evidence for a schema question.
    assert not gold.is_relevant(row_chunk("orders", "3"))


def test_rowref_pk_is_compared_as_a_string():
    # The int/str normalisation in RowRef is what makes the CSV adapter gradeable.
    gold = GoldTarget.of_rows([RowRef("customers", 7)])
    assert gold.is_relevant(row_chunk("customers", "7"))


# --------------------------------------------------------------------------
# recall / hit / MRR
# --------------------------------------------------------------------------


def test_recall_at_k_is_gold_coverage_not_a_hit_flag():
    gold = GoldTarget.of_rows(
        [RowRef("c", "1"), RowRef("c", "2"), RowRef("c", "3"), RowRef("c", "4")]
    )
    ranked = hits(
        row_chunk("c", "9"),   # 1: miss
        row_chunk("c", "1"),   # 2: covers 1/4
        row_chunk("c", "8"),   # 3: miss
        row_chunk("c", "7"),   # 4: miss
        row_chunk("c", "2"),   # 5: covers 2/4
    )
    scores = metrics.score_retrieval(ranked, gold, ks=(1, 5, 10))
    assert scores.recall_at[1] == 0.0
    assert scores.recall_at[5] == 0.5      # 2 of 4 gold rows
    assert scores.hit_at[1] == 0.0
    assert scores.hit_at[5] == 1.0         # at least one relevant chunk
    assert scores.mrr == pytest.approx(0.5)  # first relevant at rank 2
    assert scores.first_relevant_rank == 2


def test_mrr_is_one_third_when_first_relevant_is_third():
    gold = GoldTarget.of_rows([RowRef("c", "5")])
    ranked = hits(row_chunk("c", "1"), row_chunk("c", "2"), row_chunk("c", "5"))
    assert metrics.score_retrieval(ranked, gold).mrr == pytest.approx(1 / 3)


def test_no_relevant_hits_scores_zero_everywhere():
    gold = GoldTarget.of_rows([RowRef("c", "5")])
    ranked = hits(row_chunk("c", "1"), row_chunk("c", "2"))
    scores = metrics.score_retrieval(ranked, gold)
    assert scores.mrr == 0.0
    assert scores.first_relevant_rank is None
    assert scores.recall_at[10] == 0.0
    assert scores.ndcg_at[10] == 0.0


def test_empty_hit_list_is_zero_not_an_error():
    gold = GoldTarget.of_rows([RowRef("c", "5")])
    scores = metrics.score_retrieval([], gold)
    assert scores.n_hits == 0
    assert scores.recall_at[1] == 0.0
    assert scores.mrr == 0.0


# --------------------------------------------------------------------------
# nDCG
# --------------------------------------------------------------------------


def test_ndcg_perfect_ranking_is_exactly_one():
    gold = GoldTarget.of_rows([RowRef("c", "1"), RowRef("c", "2")])
    ranked = hits(row_chunk("c", "1"), row_chunk("c", "2"), row_chunk("c", "9"))
    assert metrics.ndcg_at_k(ranked, gold, 10) == pytest.approx(1.0)


def test_ndcg_hand_computed_two_gold_at_ranks_two_and_four():
    # relevance by rank: [miss, hit, miss, hit]
    #   DCG  = 1/log2(3) + 1/log2(5)
    #   IDCG = 1/log2(2) + 1/log2(3)      (2 gold rows, ideally at ranks 1 and 2)
    gold = GoldTarget.of_rows([RowRef("c", "1"), RowRef("c", "2")])
    ranked = hits(
        row_chunk("c", "8"),
        row_chunk("c", "1"),
        row_chunk("c", "7"),
        row_chunk("c", "2"),
    )
    expected = (INV_LOG2_3 + INV_LOG2_5) / (INV_LOG2_2 + INV_LOG2_3)
    assert metrics.ndcg_at_k(ranked, gold, 10) == pytest.approx(expected, abs=1e-12)
    assert expected == pytest.approx(0.65092, abs=1e-5)


def test_ndcg_single_gold_at_rank_three_is_inv_log2_4():
    gold = GoldTarget.of_rows([RowRef("c", "1")])
    ranked = hits(row_chunk("c", "8"), row_chunk("c", "9"), row_chunk("c", "1"))
    # DCG = 1/log2(4) = 0.5 ; IDCG = 1/log2(2) = 1.0
    assert metrics.ndcg_at_k(ranked, gold, 10) == pytest.approx(0.5, abs=1e-12)


def test_ndcg_cutoff_at_k_ignores_later_hits():
    gold = GoldTarget.of_rows([RowRef("c", "1")])
    ranked = hits(*[row_chunk("c", str(i)) for i in range(90, 100)], row_chunk("c", "1"))
    assert metrics.ndcg_at_k(ranked, gold, 10) == 0.0
    assert metrics.ndcg_at_k(ranked, gold, 11) == pytest.approx(
        (1 / math.log2(12)) / 1.0
    )


def test_ndcg_cannot_exceed_one_when_chunks_duplicate_the_same_gold_row():
    # Three distinct chunks all pointing at the same row. Naive binary relevance
    # would give DCG = 1 + 1/log2(3) + 1/log2(4) > IDCG = 1.
    gold = GoldTarget.of_rows([RowRef("c", "1")])
    ranked = hits(
        row_chunk("c", "1", cid="part-0"),
        row_chunk("c", "1", cid="part-1"),
        row_chunk("c", "1", cid="part-2"),
    )
    assert metrics.relevance_flags(ranked, gold) == [True, True, True]
    assert metrics.novel_coverage_flags(ranked, gold) == [True, False, False]
    assert metrics.ndcg_at_k(ranked, gold, 10) == pytest.approx(1.0)


def test_ndcg_ideal_is_capped_at_k_not_at_n_gold():
    # 20 gold rows, k=10: the best achievable is 10 hits in the top 10.
    gold = GoldTarget.of_rows([RowRef("c", str(i)) for i in range(20)])
    ranked = hits(*[row_chunk("c", str(i)) for i in range(10)])
    assert metrics.ndcg_at_k(ranked, gold, 10) == pytest.approx(1.0)
    # ... while recall@10 correctly reports that half the gold is missing.
    assert metrics.score_retrieval(ranked, gold).recall_at[10] == pytest.approx(0.5)


def test_dcg_is_the_textbook_sum():
    # ranks 1 and 3 -> 1/log2(2) + 1/log2(4)
    assert metrics.dcg([True, False, True], 3) == pytest.approx(1.0 + 0.5, abs=1e-12)
    # ranks 2 and 4 -> 1/log2(3) + 1/log2(5)
    assert metrics.dcg([False, True, False, True], 4) == pytest.approx(
        INV_LOG2_3 + INV_LOG2_5, abs=1e-12
    )
    # the k cutoff really cuts off
    assert metrics.dcg([False, True, False, True], 3) == pytest.approx(
        INV_LOG2_3, abs=1e-12
    )


# --------------------------------------------------------------------------
# Aggregate correctness
# --------------------------------------------------------------------------


def test_counts_must_match_exactly():
    assert metrics.aggregate_correct(42, 42)
    assert metrics.aggregate_correct(42.0, 42)     # a float-typed count is fine
    assert not metrics.aggregate_correct(43, 42)
    assert not metrics.aggregate_correct(42.0000001, 42)


def test_float_gold_uses_relative_tolerance():
    gold = 1234.5678
    assert metrics.aggregate_correct(gold * (1 + 1e-9), gold)
    assert metrics.aggregate_correct(gold + 1e-6, gold)          # within 1e-6 rel
    assert not metrics.aggregate_correct(gold * (1 + 1e-4), gold)


def test_float_tolerance_boundary_is_where_documented():
    gold = 1000.0
    assert metrics.aggregate_correct(gold + gold * metrics.AGG_REL_TOL * 0.99, gold)
    assert not metrics.aggregate_correct(gold + gold * metrics.AGG_REL_TOL * 10, gold)


def test_zero_gold_uses_the_absolute_floor():
    assert metrics.aggregate_correct(0.0, 0.0)
    assert not metrics.aggregate_correct(0.001, 0.0)


def test_missing_prediction_is_never_correct():
    assert not metrics.aggregate_correct(None, 5)
    assert not metrics.aggregate_correct(5, None)
    assert not metrics.aggregate_correct(float("nan"), 5)


# --------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------


def _cite(chunk: Chunk) -> Citation:
    return Citation(
        chunk_id=chunk.chunk_id,
        source_id=chunk.source_id,
        row_refs=chunk.row_refs,
    )


def test_citation_precision_is_the_fraction_that_are_gold():
    gold = GoldTarget.of_rows([RowRef("c", "1"), RowRef("c", "2")])
    cites = [_cite(row_chunk("c", "1")), _cite(row_chunk("c", "9")),
             _cite(row_chunk("c", "8")), _cite(row_chunk("c", "2"))]
    assert metrics.citation_precision(cites, gold) == pytest.approx(0.5)
    assert metrics.citation_recall(cites, gold) == pytest.approx(1.0)


def test_citation_recall_counts_gold_keys_not_citations():
    gold = GoldTarget.of_rows([RowRef("c", "1"), RowRef("c", "2"), RowRef("c", "3")])
    cites = [_cite(row_chunk("c", "1")), _cite(row_chunk("c", "1"))]
    assert metrics.citation_recall(cites, gold) == pytest.approx(1 / 3)
    assert metrics.citation_precision(cites, gold) == pytest.approx(1.0)


def test_no_citations_gives_undefined_precision_not_zero():
    gold = GoldTarget.of_rows([RowRef("c", "1")])
    assert metrics.citation_precision([], gold) is None


def test_schema_card_citation_needs_the_chunk_to_be_resolvable():
    gold = GoldTarget.of_schema(["orders"])
    card = card_chunk("orders")
    citation = Citation(chunk_id=card.chunk_id, source_id="test", row_refs=())
    # Without the chunk, a schema-card citation carries no row_refs to match on.
    assert metrics.citation_precision([citation], gold) == 0.0
    # With it, precision is 1.0. This is the missing-hook workaround.
    assert metrics.citation_precision([citation], gold, {card.chunk_id: card}) == 1.0


# --------------------------------------------------------------------------
# Summary rollups
# --------------------------------------------------------------------------


def _result(**kw) -> QuestionResult:
    base = dict(
        qid="q",
        qtype="entity_lookup",
        template_set="dev",
        answerable=True,
        expected_route=QueryRoute.LOOKUP.value,
    )
    base.update(kw)
    return QuestionResult(**base)


def test_false_answer_rate_is_non_refusals_over_unanswerables():
    results = [
        _result(qid="d1", qtype="distractor", answerable=False, refused=True),
        _result(qid="d2", qtype="distractor", answerable=False, refused=True),
        _result(qid="d3", qtype="distractor", answerable=False, refused=False),
        _result(qid="d4", qtype="distractor", answerable=False, refused=False),
        _result(qid="a1", answerable=True, refused=False),
    ]
    summary = metrics.summarize(results)
    assert summary["false_answer"]["n_unanswerable"] == 4
    assert summary["false_answer"]["n_false_answers"] == 2
    assert summary["false_answer"]["rate"] == pytest.approx(0.5)


def test_a_crashed_question_is_not_credited_as_a_refusal():
    results = [
        _result(qid="d1", qtype="distractor", answerable=False, refused=False, error="boom"),
    ]
    summary = metrics.summarize(results)
    assert summary["n_errors"] == 1
    assert summary["false_answer"]["rate"] == pytest.approx(1.0)


def test_router_accuracy_and_confusion():
    results = [
        _result(qid="1", expected_route="lookup", predicted_route="lookup"),
        _result(qid="2", expected_route="lookup", predicted_route="aggregate"),
        _result(qid="3", expected_route="aggregate", predicted_route="aggregate"),
        _result(qid="4", expected_route="hybrid", predicted_route="aggregate"),
    ]
    summary = metrics.summarize(results)
    assert summary["router"]["accuracy"] == pytest.approx(0.5)
    assert summary["router"]["confusion"]["lookup"] == {"lookup": 1, "aggregate": 1}


def test_sql_coverage_counts_aggregate_and_hybrid_questions_only():
    results = [
        _result(qid="1", expected_route="aggregate", sql_generated=True),
        _result(qid="2", expected_route="aggregate", sql_generated=False),
        _result(qid="3", expected_route="hybrid", sql_generated=True),
        _result(qid="4", expected_route="lookup", sql_generated=False),
    ]
    summary = metrics.summarize(results)
    assert summary["sql"]["n_aggregate_questions"] == 3
    assert summary["sql"]["coverage"] == pytest.approx(2 / 3)


def test_unanswerable_questions_are_excluded_from_sql_coverage():
    results = [
        _result(qid="1", expected_route="aggregate", answerable=False, sql_generated=False),
        _result(qid="2", expected_route="aggregate", sql_generated=True),
    ]
    assert metrics.summarize(results)["sql"]["coverage"] == pytest.approx(1.0)


def test_by_type_breakdown_carries_its_own_denominators():
    results = [
        _result(qid="1", qtype="entity_lookup", refused=False),
        _result(qid="2", qtype="entity_lookup", refused=True),
        _result(qid="3", qtype="distractor", answerable=False, refused=False),
    ]
    by_type = metrics.summarize(results)["by_type"]
    assert by_type["entity_lookup"]["n"] == 2
    assert by_type["entity_lookup"]["refusal_rate"] == pytest.approx(0.5)
    assert by_type["entity_lookup"]["false_answer_rate"] is None
    assert by_type["distractor"]["false_answer_rate"] == pytest.approx(1.0)


def test_retrieval_rollup_reports_the_n_it_used():
    gold = GoldTarget.of_rows([RowRef("c", "1")])
    scored = metrics.score_retrieval(hits(row_chunk("c", "1")), gold)
    results = [
        _result(qid="1", retrieval=scored),
        _result(qid="2", retrieval=None),           # count question: no row gold
        _result(qid="3", qtype="distractor", answerable=False, retrieval=None),
    ]
    summary = metrics.summarize(results)
    assert summary["retrieval"]["n"] == 1
    assert summary["retrieval"]["recall@10"] == pytest.approx(1.0)
