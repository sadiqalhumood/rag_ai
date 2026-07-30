"""The grid must stay complete, comparable, and honestly labelled."""

from __future__ import annotations

import numpy as np
import pytest

from anyrag.embed.hashing import HashingEmbedder

from evals import ablate
from evals.questions import stratified_sample, type_counts

from .conftest import TEST_SEED
from .fakes import LexicalFakeEngine


@pytest.fixture(scope="module")
def tiny(questions):
    return stratified_sample(questions, 16, TEST_SEED)


# --------------------------------------------------------------------------
# Grid shape
# --------------------------------------------------------------------------


def test_the_grid_is_three_by_two_by_three():
    cells = ablate.grid_configs()
    assert len(cells) == 18
    assert len({(m, r, k) for m, r, k, _ in cells}) == 18
    assert {m for m, _, _, _ in cells} == {"dense", "bm25", "hybrid"}
    assert {r for _, r, _, _ in cells} == {"on", "off"}
    assert {k for _, _, k, _ in cells} == {"row-chunks", "schema-cards", "both"}


def test_every_cell_has_a_distinct_config_name():
    # Otherwise two cells overwrite each other's results file, and the grid
    # silently becomes 12 cells reported as 18.
    names = [cfg.name for _, _, _, cfg in ablate.grid_configs()]
    assert len(set(names)) == 18


def test_dense_and_bm25_cells_really_disable_the_other_retriever():
    by_mode = {m: cfg for m, r, k, cfg in ablate.grid_configs() if r == "off" and k == "both"}
    assert by_mode["dense"].dense and not by_mode["dense"].lexical
    assert by_mode["bm25"].lexical and not by_mode["bm25"].dense
    assert by_mode["hybrid"].dense and by_mode["hybrid"].lexical


# --------------------------------------------------------------------------
# Running the grid
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid_run(manifest, tiny):
    engine = LexicalFakeEngine(manifest)
    return ablate.run_grid(engine, tiny, seed=TEST_SEED, progress=False)


def test_all_eighteen_cells_are_reported(grid_run):
    assert len(grid_run.cells) == 18
    assert not grid_run.reduced


def test_every_cell_used_the_same_question_set(grid_run):
    assert len({cell.n for cell in grid_run.cells}) == 1
    assert {cell.summary["n_questions"] for cell in grid_run.cells} == {len(grid_run.questions)}


def test_the_chunk_kind_axis_actually_changes_retrieval(grid_run):
    row_cells = [c for c in grid_run.cells if c.kinds == "row-chunks"]
    card_cells = [c for c in grid_run.cells if c.kinds == "schema-cards"]
    # A row-chunks cell cannot retrieve a schema card, so schema questions must
    # score zero there. That is the point of the axis.
    for cell in row_cells:
        entry = cell.summary["by_type"].get("schema_question")
        if entry and entry["recall@10"] is not None:
            assert entry["recall@10"] == 0.0
    for cell in card_cells:
        entry = cell.summary["by_type"].get("entity_lookup")
        if entry and entry["recall@10"] is not None:
            assert entry["recall@10"] == 0.0


def test_the_retrieval_axis_produces_different_numbers(grid_run):
    both = {
        c.mode: c.summary["retrieval"]["recall@10"]
        for c in grid_run.cells
        if c.kinds == "both" and c.rerank == "off"
    }
    assert len(set(both.values())) > 1, both


# --------------------------------------------------------------------------
# Timebox (amendment 4)
# --------------------------------------------------------------------------


def test_blowing_the_timebox_reduces_n_and_keeps_all_eighteen_cells(manifest, tiny):
    engine = LexicalFakeEngine(manifest)
    run = ablate.run_grid(
        engine, tiny, timebox_seconds=1e-9, seed=TEST_SEED, sample_size=8, progress=False
    )
    assert run.reduced is True
    assert len(run.cells) == 18, "grid coverage beats question count"
    assert run.reduction["n_after"] == 8
    assert run.reduction["n_before"] == len(tiny)
    assert "seed" in run.reduction and "per_type" in run.reduction


def test_the_reduced_sample_is_identical_across_every_cell(manifest, tiny):
    engine = LexicalFakeEngine(manifest)
    run = ablate.run_grid(
        engine, tiny, timebox_seconds=1e-9, seed=TEST_SEED, sample_size=8, progress=False
    )
    # An inconsistent sample would make the cells incomparable, which is worse
    # than a smaller n.
    assert {cell.n for cell in run.cells} == {8}
    qids = [
        {r["qid"] for r in cell.results} for cell in run.cells
    ]
    assert all(q == qids[0] for q in qids)


def test_the_reduction_is_stratified_by_question_type(manifest, questions):
    engine = LexicalFakeEngine(manifest)
    sample = stratified_sample(questions, 40, TEST_SEED)
    run = ablate.run_grid(
        engine, sample, timebox_seconds=1e-9, seed=TEST_SEED, sample_size=24, progress=False
    )
    assert set(run.reduction["per_type"]) == set(type_counts(sample))


# --------------------------------------------------------------------------
# Embedding cache
# --------------------------------------------------------------------------


def test_cache_returns_the_same_vectors_as_the_inner_embedder(tmp_path):
    inner = HashingEmbedder(dim=64)
    cached = ablate.CachedEmbedder(inner, tmp_path)
    texts = ["orders in Riyadh", "الرياض", "Ahmed Al-Sayed"]
    expected = inner.embed(texts)
    np.testing.assert_allclose(cached.embed(texts), expected, rtol=0, atol=0)
    assert cached.misses == 3 and cached.hits == 0
    np.testing.assert_allclose(cached.embed(texts), expected, rtol=0, atol=0)
    assert cached.hits == 3


def test_cache_survives_a_process_boundary(tmp_path):
    inner = HashingEmbedder(dim=64)
    first = ablate.CachedEmbedder(inner, tmp_path)
    first.embed(["cached across invocations"])
    first.save()

    second = ablate.CachedEmbedder(HashingEmbedder(dim=64), tmp_path)
    second.embed(["cached across invocations"])
    assert second.hits == 1 and second.misses == 0


def test_cache_is_namespaced_by_embedder_so_a_fallback_cannot_poison_it(tmp_path):
    from evals.tests.fakes import _FakeEmbedder

    a = ablate.CachedEmbedder(HashingEmbedder(dim=64), tmp_path)
    b = ablate.CachedEmbedder(_FakeEmbedder(), tmp_path)
    assert a.namespace != b.namespace
    assert a.path != b.path


def test_cache_delegates_provenance_untouched(tmp_path):
    inner = HashingEmbedder(dim=64)
    cached = ablate.CachedEmbedder(inner, tmp_path)
    assert cached.info is inner.info
    assert cached.info.degraded is True


def test_installing_the_cache_does_not_change_reported_provenance(manifest, tmp_path):
    from evals.run_eval import embedder_info_of

    engine = LexicalFakeEngine(manifest)
    before = embedder_info_of(engine)
    ablate.install_cache(engine, tmp_path)
    assert embedder_info_of(engine) == before


def test_run_grid_persists_the_cache_it_is_given(manifest, tiny, tmp_path):
    engine = LexicalFakeEngine(manifest)
    cached = ablate.CachedEmbedder(HashingEmbedder(dim=32), tmp_path)
    cached.embed(["seeded before the grid"])
    run = ablate.run_grid(
        engine, tiny[:2], seed=TEST_SEED, progress=False, cache=cached
    )
    assert cached.path.exists(), "the grid must write the cache back to disk"
    assert run.cache_stats == cached.stats
    assert run.cache_stats["entries"] >= 1


def test_embed_query_goes_through_the_cache(tmp_path):
    cached = ablate.CachedEmbedder(HashingEmbedder(dim=64), tmp_path)
    v1 = cached.embed_query("how many orders")
    v2 = cached.embed_query("how many orders")
    np.testing.assert_allclose(v1, v2)
    assert cached.hits == 1


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _provenance(degraded: bool) -> dict:
    return {
        "embedder": {
            "name": "hashing" if degraded else "local",
            "model": "m",
            "revision": "abc123def456789",
            "degraded": degraded,
        },
        "embedder_degraded": degraded,
        "generator": "extractive",
        "tokenizer": {"name": "tiktoken"},
        "template_set": "dev",
        "seed": 7,
        "manifest_content_hash": "deadbeef",
        "git_commit": "cafebabe",
    }


def test_markdown_contains_every_cell_not_just_the_winner(grid_run):
    md = ablate.render_markdown(grid_run, _provenance(False))
    for cell in grid_run.cells:
        assert f"| {cell.mode} | {cell.rerank} | {cell.kinds} |" in md
    assert md.count("| dense |") + md.count("| bm25 |") + md.count("| hybrid |") == 18


def test_markdown_stamps_provenance_and_the_n_it_used(grid_run):
    md = ablate.render_markdown(grid_run, _provenance(False))
    assert "local:m" in md
    assert f"n (questions per cell): **{grid_run.cells[0].n}**" in md
    assert "deadbeef" in md
    assert "cafebabe" in md
    assert "no timebox reduction was applied" in md


def test_degraded_embedder_is_marked_inside_every_affected_cell(grid_run):
    md = ablate.render_markdown(grid_run, _provenance(True))
    assert "DEGRADED" in md
    # One marker per metric cell, not just one warning in the prose.
    assert md.count("⚠") > 18
    for line in md.splitlines():
        if line.startswith("| dense |") or line.startswith("| hybrid |"):
            assert "⚠" in line


def test_a_reduced_grid_says_so_in_the_header(manifest, tiny):
    engine = LexicalFakeEngine(manifest)
    run = ablate.run_grid(
        engine, tiny, timebox_seconds=1e-9, seed=TEST_SEED, sample_size=8, progress=False
    )
    md = ablate.render_markdown(run, _provenance(False))
    assert "Timebox reduction" in md
    assert "The grid is complete on the reduced set" in md
    assert "n (questions per cell): **8**" in md


def test_markdown_carries_the_false_answer_provider_caveat(grid_run):
    md = ablate.render_markdown(grid_run, _provenance(False))
    assert "refusal-threshold logic, not hallucination" in md
    assert "False-answer rate by cell" in md
