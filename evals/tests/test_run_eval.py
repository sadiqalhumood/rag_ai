"""End-to-end harness behaviour, driven against fakes.

The three fakes are fixed points: the oracle must score 1.0 and 0% false
answers, the always-answer engine must score 100% false answers, and the
crashing engine must not be able to launder a crash into a refusal. If the
harness cannot distinguish those three, it cannot distinguish anything.
"""

from __future__ import annotations

import json

import pytest

from anyrag.core.config import GenerationConfig, RetrievalConfig
from anyrag.core.types import Answer, ChunkKind, Citation, QueryRoute, RowRef

from evals import run_eval as R

from .conftest import TEST_SEED
from .fakes import (
    AlwaysAnswerEngine,
    CrashingEngine,
    LexicalFakeEngine,
    OracleEngine,
    build_corpus,
)


@pytest.fixture(scope="module")
def small(manifest, questions):
    """A stratified slice, so the full-fidelity tests stay fast."""
    from evals.questions import stratified_sample

    return stratified_sample(questions, 64, TEST_SEED)


# --------------------------------------------------------------------------
# Named configurations
# --------------------------------------------------------------------------


def test_named_configs_cover_the_whole_grid():
    configs = R.named_configs()
    distinct = {c.name for c in configs.values()}
    assert len(distinct) == 18
    assert "hybrid_rerank_both" in configs
    assert configs["hybrid_rerank_both"].dense and configs["hybrid_rerank_both"].lexical
    assert configs["bm25_norerank_row"].dense is False


# --------------------------------------------------------------------------
# The oracle: everything right
# --------------------------------------------------------------------------


def test_oracle_scores_perfectly_and_never_false_answers(manifest, small):
    engine = OracleEngine(manifest, small)
    payload = R.run_eval(
        engine, manifest=manifest, questions=small, seed=TEST_SEED,
        retrieval=R.named_configs()["hybrid_norerank_both"],
    )
    summary = payload["summary"]
    assert summary["retrieval"]["recall@10"] == pytest.approx(1.0)
    assert summary["retrieval"]["ndcg@10"] == pytest.approx(1.0)
    assert summary["retrieval"]["mrr"] == pytest.approx(1.0)
    assert summary["false_answer"]["rate"] == pytest.approx(0.0)
    assert summary["aggregate"]["accuracy"] == pytest.approx(1.0)
    assert summary["router"]["accuracy"] == pytest.approx(1.0)
    assert summary["citations"]["precision"] == pytest.approx(1.0)
    assert summary["n_errors"] == 0


def test_oracle_generates_sql_for_every_aggregate_question(manifest, small):
    engine = OracleEngine(manifest, small)
    payload = R.run_eval(engine, manifest=manifest, questions=small, seed=TEST_SEED)
    assert payload["summary"]["sql"]["coverage"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# The degenerate engines
# --------------------------------------------------------------------------


def test_always_answering_engine_scores_a_hundred_percent_false_answers(manifest, small):
    engine = AlwaysAnswerEngine(manifest)
    payload = R.run_eval(engine, manifest=manifest, questions=small, seed=TEST_SEED)
    fa = payload["summary"]["false_answer"]
    assert fa["rate"] == pytest.approx(1.0)
    assert fa["n_false_answers"] == fa["n_unanswerable"] > 0


def test_a_crash_is_recorded_as_an_error_not_as_a_refusal(manifest, small):
    payload = R.run_eval(
        CrashingEngine(), manifest=manifest, questions=small, seed=TEST_SEED
    )
    summary = payload["summary"]
    assert summary["n_errors"] == len(small)
    # Crucially: a system that crashes on every distractor must not be able to
    # report a perfect false-answer rate.
    assert summary["false_answer"]["rate"] == pytest.approx(1.0)
    assert summary["refusal"]["refusal_rate_unanswerable"] == pytest.approx(0.0)


def test_a_realistic_fake_lands_strictly_between_the_two_fixed_points(manifest, small):
    engine = LexicalFakeEngine(manifest)
    payload = R.run_eval(engine, manifest=manifest, questions=small, seed=TEST_SEED)
    summary = payload["summary"]
    assert 0.0 < summary["retrieval"]["recall@10"] < 1.0
    assert summary["n_errors"] == 0
    assert summary["false_answer"]["rate"] is not None


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


REQUIRED_PROVENANCE = (
    "embedder",
    "embedder_degraded",
    "generator",
    "tokenizer",
    "template_set",
    "n",
    "seed",
    "manifest_content_hash",
    "retrieval_config",
    "retrieval_config_name",
    "git_commit",
)


def test_every_result_records_full_provenance(manifest, small):
    engine = LexicalFakeEngine(manifest)
    payload = R.run_eval(
        engine,
        manifest=manifest,
        questions=small,
        seed=TEST_SEED,
        templates="dev",
        generation=GenerationConfig(generator="extractive"),
    )
    prov = payload["provenance"]
    for key in REQUIRED_PROVENANCE:
        assert key in prov, key
    assert prov["n"] == len(small)
    assert prov["seed"] == TEST_SEED
    assert prov["template_set"] == "dev"
    assert prov["generator"] == "extractive"
    assert prov["manifest_content_hash"] == manifest.meta["content_hash"]
    assert prov["embedder"]["name"] == "fake"
    assert prov["embedder_degraded"] is True
    assert sum(prov["n_by_type"].values()) == len(small)


def test_an_engine_without_an_embedder_is_marked_degraded_not_silent(manifest, small):
    class NoEmbedder(LexicalFakeEngine):
        pass

    engine = NoEmbedder(manifest)
    del engine.embedder
    prov = R.embedder_info_of(engine)
    assert prov["degraded"] is True
    assert prov["name"] == "unknown"
    assert "provenance" in prov["detail"]


def test_results_serialise_to_json(tmp_path, manifest, small):
    engine = LexicalFakeEngine(manifest)
    payload = R.run_eval(engine, manifest=manifest, questions=small, seed=TEST_SEED)
    path = R.write_results(payload, "unit-test", tmp_path)
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["summary"]["n_questions"] == len(small)
    assert len(reloaded["results"]) == len(small)
    assert reloaded["provenance"]["embedder"]["name"] == "fake"


def test_per_question_records_keep_the_gold_and_the_prediction(manifest, small):
    engine = OracleEngine(manifest, small)
    payload = R.run_eval(engine, manifest=manifest, questions=small, seed=TEST_SEED)
    aggregates = [r for r in payload["results"] if r["gold_scalar"] is not None]
    assert aggregates
    for record in aggregates:
        assert record["predicted_scalar"] is not None
        assert record["aggregate_ok"] is True
        assert record["scalar_source"].startswith("trace")


# --------------------------------------------------------------------------
# Answer introspection
# --------------------------------------------------------------------------


def _answer(**kw) -> Answer:
    base = dict(
        text="x",
        citations=(Citation(chunk_id="c", source_id="s"),),
        route=QueryRoute.AGGREGATE,
    )
    base.update(kw)
    return Answer(**base)


def test_scalar_is_read_from_the_trace_before_the_text():
    answer = _answer(text="the answer is 999", trace={"scalar": 42})
    value, source = R.extract_predicted_scalar(answer)
    assert value == 42
    assert source == "trace['scalar']"


def test_scalar_falls_back_to_parsing_the_text_and_says_so():
    answer = _answer(text="There are 1,234 orders.", trace={})
    value, source = R.extract_predicted_scalar(answer)
    assert value == 1234
    assert source == "text"


def test_scalar_is_read_from_a_query_result_object():
    from anyrag.core.types import QueryResult

    result = QueryResult(columns=("n",), rows=((7,),), sql="SELECT COUNT(*) FROM t")
    answer = _answer(text="seven", trace={"sql_result": result})
    assert R.extract_predicted_scalar(answer)[0] == 7
    assert R.extract_sql(answer)[0] == "SELECT COUNT(*) FROM t"


def test_sql_is_found_on_a_sql_result_chunk_when_the_trace_is_silent(manifest):
    from anyrag.core.types import Chunk

    chunk = Chunk(
        chunk_id="sqlres",
        source_id="s",
        kind=ChunkKind.SQL_RESULT,
        text="count=3",
        row_refs=(),
        meta={"sql": "SELECT COUNT(*) FROM orders"},
    )
    answer = _answer(citations=(Citation(chunk_id="sqlres", source_id="s"),), trace={})
    sql, source = R.extract_sql(answer, {"sqlres": chunk})
    assert sql == "SELECT COUNT(*) FROM orders"
    assert "meta" in source


def test_no_sql_anywhere_is_reported_as_no_coverage():
    answer = _answer(trace={})
    assert R.extract_sql(answer) == (None, "none")


# --------------------------------------------------------------------------
# Grading rules
# --------------------------------------------------------------------------


def test_refusing_an_answerable_aggregate_is_wrong_but_not_a_false_answer(manifest, questions):
    question = next(q for q in questions if q.gold_scalar is not None and q.answerable)
    answer = Answer.refusal("out of coverage", route=question.route)
    result = R.grade_question(question, answer, [])
    assert result.aggregate_ok is False
    assert result.refused is True
    assert result.false_answer is False


def test_unanswerable_questions_have_undefined_citation_precision(manifest, questions):
    question = next(q for q in questions if not q.answerable)
    answer = Answer(
        text="Sure.",
        citations=(Citation(chunk_id="c", source_id="s", row_refs=(RowRef("customers", "1"),)),),
        route=question.route,
    )
    result = R.grade_question(question, answer, [])
    assert result.citation_precision is None
    assert result.false_answer is True


def test_retrieval_is_graded_even_when_generation_fails(manifest, questions):
    question = next(q for q in questions if q.qtype == "entity_lookup")
    corpus = {c.chunk_id: c for c in build_corpus(manifest)}
    ref = next(iter(question.gold.row_refs))
    from anyrag.core.types import Hit

    chunk = next(c for c in corpus.values() if ref in c.row_refs)
    result = R.grade_question(
        question, None, [Hit(chunk=chunk, score=1.0, rank=1)], error="ask: boom"
    )
    assert result.error == "ask: boom"
    assert result.retrieval is not None
    assert result.retrieval.recall_at[1] == pytest.approx(1.0)
    assert result.refused is False


# --------------------------------------------------------------------------
# Subsets and limits
# --------------------------------------------------------------------------


def test_distractor_subset_selects_only_unanswerable_questions(questions):
    subset = R.select_subset(questions, "distractors")
    assert subset
    assert all(not q.answerable for q in subset)


def test_aggregate_subset_selects_aggregate_and_hybrid_routes(questions):
    subset = R.select_subset(questions, "aggregates")
    assert all(q.route in (QueryRoute.AGGREGATE, QueryRoute.HYBRID) for q in subset)
    assert all(q.answerable for q in subset)


def test_subset_can_name_a_question_type(questions):
    subset = R.select_subset(questions, "schema_question")
    assert subset and all(q.qtype == "schema_question" for q in subset)


def test_unknown_subset_is_an_error(questions):
    with pytest.raises(ValueError):
        R.select_subset(questions, "nonsense")


def test_limit_is_applied_as_a_stratified_sample(manifest):
    engine = LexicalFakeEngine(manifest)
    payload = R.run_eval(engine, manifest=manifest, seed=TEST_SEED, limit=40)
    assert payload["summary"]["n_questions"] == 40
    assert len(payload["summary"]["by_type"]) >= 6


# --------------------------------------------------------------------------
# The adapter seam
# --------------------------------------------------------------------------


def test_adapter_finds_a_working_call_shape_and_records_it(manifest, small):
    class KeywordOnly:
        def __init__(self, inner):
            self.inner = inner
            self.embedder = inner.embedder

        def retrieve(self, question, *, config=None):
            return self.inner.retrieve(question, config)

        def ask(self, question, *, retrieval=None, generation=None):
            return self.inner.ask(question, retrieval)

    engine = KeywordOnly(LexicalFakeEngine(manifest))
    adapter = R.EngineAdapter(engine, GenerationConfig())
    config = RetrievalConfig()
    assert adapter.retrieve(small[0].text, config) is not None
    adapter.ask(small[0].text, config)
    assert adapter.retrieve_shape == "retrieve(q, config=config)"
    assert adapter.ask_shape == "ask(q, retrieval=cfg, generation=gen)"


def test_adapter_raises_a_readable_error_when_nothing_fits(manifest):
    class Wrong:
        def retrieve(self, a, b, c, d):
            raise AssertionError("unreachable")

    adapter = R.EngineAdapter(Wrong())
    with pytest.raises(TypeError, match="no supported call shape"):
        adapter.retrieve("q", RetrievalConfig())
