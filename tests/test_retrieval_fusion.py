"""Reciprocal rank fusion and its score-normalisation alternative.

The property that matters is the one the whole hybrid claim rests on:
**agreement beats depth**. A chunk both retrievers put near the top must beat a
chunk only one retriever put at the very top, and it must do so for a stated
reason rather than by accident of scale.
"""

from __future__ import annotations

import pytest

from anyrag.core.types import Hit
from anyrag.index import FIRST_RANK
from anyrag.retrieval.fusion import (
    DEFAULT_RRF_K,
    FusedHit,
    fuse,
    normalized_score_fusion,
    rrf,
)
from test_retrieval_fixtures import ids, make_hit, ranked


def test_rank_convention_is_the_one_the_indexes_use() -> None:
    # Guards the off-by-one the whole fusion formula depends on.
    assert FIRST_RANK == 1
    assert ranked(["a", "b"])[0].rank == FIRST_RANK


def test_chunk_in_both_lists_beats_chunk_in_one() -> None:
    dense = ranked(["shared", "dense_only"], retriever="dense")
    lexical = [
        make_hit("lex_only", rank=1, retriever="bm25"),
        make_hit("shared", rank=2, retriever="bm25"),
    ]
    fused = rrf([dense, lexical])
    assert fused[0].chunk_id == "shared"
    # ... and it beats the chunks that were rank 1 in exactly one list.
    assert set(ids(fused)[1:]) == {"dense_only", "lex_only"}


def test_rrf_formula_is_the_textbook_one() -> None:
    dense = [make_hit("a", rank=1, retriever="dense")]
    lexical = [make_hit("a", rank=1, retriever="bm25")]
    fused = rrf([dense, lexical], normalize=False)
    assert fused[0].score == pytest.approx(2.0 / (DEFAULT_RRF_K + 1))

    solo = rrf([dense], normalize=False)
    assert solo[0].score == pytest.approx(1.0 / (DEFAULT_RRF_K + 1))


def test_rrf_k_damps_the_rank_penalty() -> None:
    dense = ranked([f"d{i}" for i in range(5)], retriever="dense")
    lexical = ranked([f"l{i}" for i in range(5)], retriever="bm25")
    tight = rrf([dense, lexical], 1, normalize=False)
    loose = rrf([dense, lexical], 1000, normalize=False)
    spread = lambda f: f[0].score / f[-1].score  # noqa: E731
    assert spread(tight) > spread(loose)


def test_normalisation_scales_without_reordering() -> None:
    dense = ranked(["a", "b", "c"], retriever="dense")
    lexical = ranked(["b", "c", "d"], retriever="bm25")
    raw = rrf([dense, lexical], normalize=False)
    normed = rrf([dense, lexical], normalize=True)
    assert ids(raw) == ids(normed)
    assert normed[0].score <= 1.0
    assert all(h.score > 0 for h in normed)
    # The divisor is the best attainable score, so rank-1-everywhere is exactly 1.
    both_first = rrf([[make_hit("x", 1, retriever="dense")],
                      [make_hit("x", 1, retriever="bm25")]])
    assert both_first[0].score == pytest.approx(1.0)


def test_normalised_scores_are_comparable_across_grid_cells() -> None:
    # The reason normalisation exists: an absolute floor like
    # GenerationConfig.min_support_score must not mean something different in
    # the dense-only cell than in the hybrid cell.
    dense_only = rrf([ranked(["a", "b"], retriever="dense")])
    hybrid = rrf([ranked(["a", "b"], retriever="dense"),
                  ranked(["a", "b"], retriever="bm25")])
    assert dense_only[0].score == pytest.approx(hybrid[0].score) == pytest.approx(1.0)


def test_fused_hits_record_per_retriever_ranks_and_scores() -> None:
    dense = [make_hit("a", rank=3, score=0.81, retriever="dense")]
    lexical = [make_hit("a", rank=2, score=11.4, retriever="bm25")]
    fused = rrf([dense, lexical])
    hit = fused[0]
    assert isinstance(hit, FusedHit) and isinstance(hit, Hit)
    assert hit.retriever == "rrf"
    assert hit.ranks == {"dense": 3, "bm25": 2}
    assert hit.raw_scores == {"dense": 0.81, "bm25": 11.4}
    assert hit.retrievers == ("bm25", "dense")
    assert hit.n_retrievers == 2
    assert hit.best_rank() == 2
    assert sum(hit.contributions.values()) == pytest.approx(hit.score)


def test_output_ranks_are_contiguous_and_one_based() -> None:
    fused = rrf([ranked(["a", "b", "c"]), ranked(["c", "d"], retriever="bm25")])
    assert [h.rank for h in fused] == [1, 2, 3, 4]


def test_ties_break_on_chunk_id() -> None:
    # Symmetric input: "zeta" and "alpha" are each rank 1 in exactly one list.
    dense = [make_hit("zeta", rank=1, retriever="dense")]
    lexical = [make_hit("alpha", rank=1, retriever="bm25")]
    fused = rrf([dense, lexical])
    assert fused[0].score == pytest.approx(fused[1].score)
    assert ids(fused) == ["alpha", "zeta"]
    # Reversing the input lists must not change the answer.
    assert ids(rrf([lexical, dense])) == ["alpha", "zeta"]


def test_weights_damp_a_lists_vote() -> None:
    dense = [make_hit("a", rank=1, retriever="dense")]
    variant = [make_hit("b", rank=1, retriever="dense")]
    assert ids(rrf([dense, variant], weights=[1.0, 0.5])) == ["a", "b"]
    assert ids(rrf([dense, variant], weights=[0.5, 1.0])) == ["b", "a"]


def test_repeated_labels_sum_contributions_and_keep_the_best_rank() -> None:
    # Two expansion variants searched with the same retriever.
    first = [make_hit("a", rank=5, retriever="dense")]
    second = [make_hit("a", rank=2, retriever="dense")]
    fused = rrf([first, second], normalize=False)
    hit = fused[0]
    assert hit.ranks == {"dense": 2}
    assert hit.score == pytest.approx(
        1 / (DEFAULT_RRF_K + 5) + 1 / (DEFAULT_RRF_K + 2)
    )
    assert hit.contributions["dense"] == pytest.approx(hit.score)


def test_unranked_hits_fall_back_to_list_position() -> None:
    # Hit.rank defaults to 0, which means "unranked", not "best".
    unranked = [
        Hit(chunk=make_hit("a", 1).chunk, score=9.0),
        Hit(chunk=make_hit("b", 1).chunk, score=8.0),
    ]
    fused = rrf([unranked], normalize=False)
    assert ids(fused) == ["a", "b"]
    assert fused[0].score == pytest.approx(1 / (DEFAULT_RRF_K + 1))
    assert fused[1].score == pytest.approx(1 / (DEFAULT_RRF_K + 2))


def test_single_list_fusion_preserves_the_retrievers_order() -> None:
    only = ranked(["a", "b", "c", "d"], retriever="bm25")
    assert ids(rrf([only])) == ids(only)


def test_empty_input_is_empty_output() -> None:
    assert rrf([]) == []
    assert rrf([[], []]) == []
    assert normalized_score_fusion([[], []]) == []


def test_labels_override_the_hit_retriever() -> None:
    fused = rrf([ranked(["a"], retriever="dense")], labels=["custom"])
    assert fused[0].ranks == {"custom": 1}


def test_mismatched_labels_or_weights_raise() -> None:
    lists = [ranked(["a"]), ranked(["b"])]
    with pytest.raises(ValueError):
        rrf(lists, labels=["only-one"])
    with pytest.raises(ValueError):
        rrf(lists, weights=[1.0])
    with pytest.raises(ValueError):
        rrf(lists, -1)


# --------------------------------------------------------------------------
# Score-normalisation fusion
# --------------------------------------------------------------------------


def test_score_fusion_keeps_margins_that_rrf_discards() -> None:
    # Dense is emphatic about "a" and lukewarm about "b"; BM25 puts "b" first
    # but by a hair. The ranks are symmetric, the margins are not.
    dense = [
        make_hit("a", rank=1, score=0.95, retriever="dense"),
        make_hit("b", rank=2, score=0.60, retriever="dense"),
        make_hit("c", rank=3, score=0.10, retriever="dense"),
    ]
    lexical = [
        make_hit("b", rank=1, score=5.10, retriever="bm25"),
        make_hit("a", rank=2, score=5.05, retriever="bm25"),
        make_hit("c", rank=3, score=4.90, retriever="bm25"),
    ]
    # RRF sees rank 1+2 against 2+1 and calls it a tie, broken by chunk_id.
    fused_rrf = rrf([dense, lexical])
    assert ids(fused_rrf)[:2] == ["a", "b"]
    assert fused_rrf[0].score == pytest.approx(fused_rrf[1].score)
    # Score fusion sees the margins and breaks the tie on evidence instead.
    scored = normalized_score_fusion([dense, lexical])
    assert ids(scored)[:2] == ["a", "b"]
    assert scored[0].score > scored[1].score


def test_score_fusion_normalises_per_list_and_bounds_the_result() -> None:
    dense = [
        make_hit("a", rank=1, score=0.9, retriever="dense"),
        make_hit("b", rank=2, score=0.1, retriever="dense"),
    ]
    lexical = [
        make_hit("a", rank=1, score=180.0, retriever="bm25"),
        make_hit("b", rank=2, score=20.0, retriever="bm25"),
    ]
    fused = normalized_score_fusion([dense, lexical])
    assert fused[0].chunk_id == "a"
    assert fused[0].score == pytest.approx(1.0)
    assert 0.0 <= fused[1].score <= 1.0
    assert fused[0].retriever == "norm"


def test_score_fusion_treats_a_flat_list_as_all_equal() -> None:
    flat = [
        make_hit("a", rank=1, score=1.0, retriever="dense"),
        make_hit("b", rank=2, score=1.0, retriever="dense"),
    ]
    fused = normalized_score_fusion([flat])
    assert [h.score for h in fused] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert ids(fused) == ["a", "b"]  # tie broken by chunk_id


def test_fuse_dispatches_by_name() -> None:
    lists = [ranked(["a", "b"]), ranked(["b"], retriever="bm25")]
    assert ids(fuse("rrf", lists, k=60)) == ids(rrf(lists, 60))
    assert fuse("norm", lists, k=60)[0].retriever == "norm"
    with pytest.raises(ValueError):
        fuse("nope", lists)


def test_fusion_is_deterministic() -> None:
    lists = [ranked(["a", "b", "c"]), ranked(["c", "a", "z"], retriever="bm25")]
    first = rrf(lists)
    second = rrf(lists)
    assert [(h.chunk_id, h.score, h.rank) for h in first] == [
        (h.chunk_id, h.score, h.rank) for h in second
    ]
