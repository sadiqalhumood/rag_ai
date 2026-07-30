"""Rerankers.

The load-bearing test in this file is
`test_lexical_reranker_promotes_the_chunk_the_rare_term_is_in`: if the offline
reranker cannot change the order for a reason a first-stage retriever missed,
then "rerank on" is not an experimental condition and a third of the ablation
grid is measuring nothing.
"""

from __future__ import annotations

import sys
import types
from typing import Sequence

import pytest

from anyrag.core.interfaces import Reranker
from anyrag.core.types import Hit
from anyrag.index import FIRST_RANK
from anyrag.retrieval.fusion import rrf
from anyrag.retrieval.rerank import (
    CrossEncoderReranker,
    LexicalOverlapReranker,
    NoopReranker,
    get_reranker,
)
from test_retrieval_fixtures import ids, make_chunk, make_hit


def hit(chunk_id: str, text: str, score: float, rank: int) -> Hit:
    return Hit(
        chunk=make_chunk(chunk_id, text), score=score, rank=rank, retriever="rrf"
    )


def test_all_three_conform_to_the_protocol() -> None:
    for reranker in (
        NoopReranker(),
        LexicalOverlapReranker(),
        CrossEncoderReranker(),
    ):
        assert isinstance(reranker, Reranker)
        assert isinstance(reranker.name, str) and reranker.name


# --------------------------------------------------------------------------
# Noop
# --------------------------------------------------------------------------


def test_noop_is_the_identity_up_to_truncation() -> None:
    hits = [hit(f"c{i}", f"text {i}", 1.0 - 0.1 * i, i + 1) for i in range(5)]
    out = NoopReranker().rerank("anything", hits, k=3)
    assert out == hits[:3]  # same objects, same scores, same ranks
    assert NoopReranker().rerank("anything", [], k=3) == []


# --------------------------------------------------------------------------
# Lexical overlap
# --------------------------------------------------------------------------


def test_lexical_reranker_promotes_the_chunk_the_rare_term_is_in() -> None:
    # Every candidate is a customer row, so "email" and "customer" are free.
    # Only one of them contains the name, and it is not the one fusion liked.
    query = "email of customer Ahmed Al-Sayed"
    hits = [
        hit("c1", "customer Mona Hassan email mona@example.com region Cairo", 0.9, 1),
        hit("c2", "customer Omar Farouk email omar@example.com region Giza", 0.8, 2),
        hit("c3", "customer Ahmed Al-Sayed email ahmed@example.com", 0.7, 3),
    ]
    before = ids(hits)
    after = ids(LexicalOverlapReranker().rerank(query, hits, k=3))
    assert before[0] == "c1"
    assert after[0] == "c3"
    assert after != before


def test_terms_present_in_every_candidate_do_not_decide_the_order() -> None:
    # c_long contains far more of the query's generic vocabulary; c_rare
    # contains the one term that is not in every candidate.
    query = "customer email region Zanzibar"
    hits = [
        hit("c_long", "customer email region customer email region", 0.9, 1),
        hit("c_rare", "customer email region Zanzibar", 0.9, 2),
    ]
    scores = dict(
        zip(ids(hits), LexicalOverlapReranker().overlap_scores(query, hits))
    )
    assert scores["c_rare"] > scores["c_long"]


def test_adjacency_separates_a_phrase_from_the_same_words_scattered() -> None:
    query = "Ahmed Al-Sayed"
    hits = [
        hit("phrase", "customer Ahmed Al-Sayed of Cairo", 0.5, 1),
        hit("scattered", "customer Ahmed Hassan and customer Mona Al-Sayed", 0.5, 2),
    ]
    reranked = LexicalOverlapReranker().rerank(query, hits, k=2)
    assert ids(reranked) == ["phrase", "scattered"]
    assert reranked[0].score > reranked[1].score


def test_blend_zero_leaves_the_fused_order_alone() -> None:
    query = "Ahmed"
    hits = [
        hit("a", "Mona Hassan", 0.9, 1),
        hit("b", "Ahmed Al-Sayed", 0.5, 2),
    ]
    assert ids(LexicalOverlapReranker(blend=0.0).rerank(query, hits, k=2)) == ["a", "b"]
    assert ids(LexicalOverlapReranker(blend=1.0).rerank(query, hits, k=2)) == ["b", "a"]


def test_the_fused_score_still_counts_at_the_default_blend() -> None:
    # Two chunks with identical overlap: first-stage evidence breaks the tie.
    query = "Ahmed"
    hits = [
        hit("low", "customer Ahmed one", 0.10, 2),
        hit("high", "customer Ahmed two", 0.90, 1),
    ]
    assert ids(LexicalOverlapReranker().rerank(query, hits, k=2)) == ["high", "low"]


def test_scores_stay_in_the_unit_interval() -> None:
    query = "customer Ahmed Al-Sayed email"
    hits = [
        hit("a", "customer Ahmed Al-Sayed email ahmed@example.com", 1.0, 1),
        hit("b", "order 1002 amount 42", 0.001, 2),
    ]
    for h in LexicalOverlapReranker().rerank(query, hits, k=2):
        assert 0.0 <= h.score <= 1.0


def test_reranked_hits_are_relabelled_and_reranked_contiguously() -> None:
    hits = [hit(f"c{i}", f"customer number {i}", 1.0 - 0.1 * i, i + 1) for i in range(4)]
    out = LexicalOverlapReranker().rerank("customer number 2", hits, k=3)
    assert [h.rank for h in out] == [FIRST_RANK, FIRST_RANK + 1, FIRST_RANK + 2]
    assert {h.retriever for h in out} == {"rerank"}


def test_reranking_preserves_fusion_provenance() -> None:
    fused = rrf(
        [
            [make_hit("a", 1, 0.9, "dense", text="customer Ahmed")],
            [make_hit("a", 2, 4.0, "bm25", text="customer Ahmed")],
        ]
    )
    out = LexicalOverlapReranker().rerank("Ahmed", fused, k=1)
    assert out[0].ranks == {"dense": 1, "bm25": 2}  # FusedHit survived replace()


def test_empty_candidates_and_empty_query_are_handled() -> None:
    reranker = LexicalOverlapReranker()
    assert reranker.rerank("anything", [], k=5) == []
    hits = [hit("a", "customer one", 0.9, 1), hit("b", "customer two", 0.8, 2)]
    # A query with no content terms cannot express a preference; the fused
    # order must survive rather than collapsing to an arbitrary one.
    assert ids(reranker.rerank("the of and", hits, k=2)) == ["a", "b"]


def test_reranking_is_deterministic_and_breaks_ties_on_chunk_id() -> None:
    query = "customer Ahmed"
    hits = [
        hit("zeta", "customer Ahmed", 0.5, 1),
        hit("alpha", "customer Ahmed", 0.5, 2),
    ]
    first = LexicalOverlapReranker().rerank(query, hits, k=2)
    second = LexicalOverlapReranker().rerank(query, hits, k=2)
    assert ids(first) == ids(second) == ["alpha", "zeta"]


def test_invalid_blend_is_rejected() -> None:
    with pytest.raises(ValueError):
        LexicalOverlapReranker(blend=1.5)
    with pytest.raises(ValueError):
        LexicalOverlapReranker(phrase_weight=-0.1)


# --------------------------------------------------------------------------
# Cross-encoder
# --------------------------------------------------------------------------


class _FakeCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder without the download."""

    def __init__(self, model_name: str, **kwargs: object) -> None:
        self.model_name = model_name

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        # Longer text wins, which no other component in this test file does,
        # so "the model was actually consulted" is observable.
        return [float(len(text)) for _query, text in pairs]


def install_fake(monkeypatch: pytest.MonkeyPatch, factory: object) -> None:
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_cross_encoder_orders_by_model_score_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake(monkeypatch, _FakeCrossEncoder)
    reranker = CrossEncoderReranker()
    hits = [hit("short", "Ahmed", 0.9, 1), hit("long", "a much longer text here", 0.1, 2)]
    out = reranker.rerank("Ahmed", hits, k=2)
    assert ids(out) == ["long", "short"]
    assert reranker.fell_back is False
    assert all(0.0 <= h.score <= 1.0 for h in out)  # logits squashed


def test_cross_encoder_is_imported_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[str] = []

    def factory(model_name: str, **kwargs: object) -> _FakeCrossEncoder:
        loaded.append(model_name)
        return _FakeCrossEncoder(model_name)

    install_fake(monkeypatch, factory)
    reranker = CrossEncoderReranker()
    assert loaded == []  # construction downloads nothing
    reranker.rerank("q", [hit("a", "text", 1.0, 1)], k=1)
    assert loaded == [reranker.model_name]


def test_cross_encoder_falls_back_loudly_when_the_model_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(model_name: str, **kwargs: object) -> None:
        raise OSError("model not found on disk and no network")

    install_fake(monkeypatch, explode)
    query = "email of customer Ahmed Al-Sayed"
    hits = [
        hit("c1", "customer Mona Hassan email mona@example.com", 0.9, 1),
        hit("c2", "customer Ahmed Al-Sayed email ahmed@example.com", 0.7, 2),
    ]
    reranker = CrossEncoderReranker()
    out = reranker.rerank(query, hits, k=2)

    assert reranker.fell_back is True
    assert "model not found" in reranker.fallback_reason
    assert reranker.info["degraded"] is True
    assert reranker.active_name == "cross_encoder->lexical_overlap"
    # The fallback is the lexical reranker, not an unranked passthrough.
    assert ids(out) == ids(LexicalOverlapReranker().rerank(query, hits, k=2))


def test_cross_encoder_reports_fallback_when_prediction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Broken(_FakeCrossEncoder):
        def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
            raise RuntimeError("CUDA is on fire")

    install_fake(monkeypatch, Broken)
    reranker = CrossEncoderReranker()
    out = reranker.rerank("q", [hit("a", "customer text", 1.0, 1)], k=1)
    assert reranker.fell_back is True
    assert "CUDA is on fire" in reranker.fallback_reason
    assert len(out) == 1


def test_cross_encoder_keeps_unscored_tail_instead_of_dropping_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake(monkeypatch, _FakeCrossEncoder)
    reranker = CrossEncoderReranker(max_pairs=2)
    hits = [hit(f"c{i}", "text " * (i + 1), 1.0 - 0.1 * i, i + 1) for i in range(4)]
    out = reranker.rerank("q", hits, k=10)
    assert len(out) == 4
    assert set(ids(out)) == {"c0", "c1", "c2", "c3"}
    assert [h.rank for h in out] == [1, 2, 3, 4]


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_get_reranker_builds_by_name_and_rejects_unknown() -> None:
    assert isinstance(get_reranker(), LexicalOverlapReranker)
    assert isinstance(get_reranker("noop"), NoopReranker)
    assert isinstance(get_reranker("cross_encoder"), CrossEncoderReranker)
    assert get_reranker("lexical", blend=0.25).blend == 0.25
    with pytest.raises(ValueError):
        get_reranker("vibes")
