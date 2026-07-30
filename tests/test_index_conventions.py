"""The cross-index contract retrieval will consume.

Fusion code reads `Hit.rank` from a dense index and a lexical index in the same
breath. If the two disagreed about whether the best hit is rank 0 or rank 1, RRF
would silently weight one retriever above the other, and the ablation table would
be measuring an off-by-one. So the convention is pinned here, once, for all three
index classes rather than implicitly inside each one's own test module.

**The convention: `Hit.rank` is 1-based.** Best hit is rank 1, ranks are
contiguous within a search, ties are broken by `chunk_id`.
"""

from __future__ import annotations

import numpy as np
import pytest

from anyrag.index import FIRST_RANK, BM25Index, MemoryVectorIndex, PersistentVectorIndex
from test_index_fixtures import make_chunk, random_unit_vectors

DIM = 8
TEXTS = [
    "quarterly revenue for the northern region",
    "quarterly revenue for the southern region",
    "shipping and packaging costs",
    "customer contact details for gulf trading",
    "تقرير الإيرادات الفصلية للمنطقة الشمالية",
]


def dense(path: str | None = None):
    chunks = [make_chunk(f"c{i}", t) for i, t in enumerate(TEXTS)]
    vecs = random_unit_vectors(len(TEXTS), DIM, seed=21)
    idx = (
        MemoryVectorIndex(dim=DIM)
        if path is None
        else PersistentVectorIndex(path, dim=DIM)
    )
    idx.upsert(chunks, vecs)
    return idx, vecs


def lexical() -> BM25Index:
    idx = BM25Index()
    idx.upsert([make_chunk(f"c{i}", t) for i, t in enumerate(TEXTS)])
    return idx


def test_first_rank_constant_matches_the_documented_convention() -> None:
    assert FIRST_RANK == 1


@pytest.mark.parametrize("kind", ["memory", "persistent", "bm25"])
def test_every_index_ranks_from_one_contiguously(kind: str, tmp_path) -> None:
    if kind == "bm25":
        hits = lexical().search("quarterly revenue region costs", k=3)
    else:
        idx, vecs = dense(str(tmp_path / "ix") if kind == "persistent" else None)
        hits = idx.search(vecs[0], k=3)
    assert len(hits) == 3
    assert [h.rank for h in hits] == [FIRST_RANK, FIRST_RANK + 1, FIRST_RANK + 2]
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:]))


@pytest.mark.parametrize("kind", ["memory", "persistent", "bm25"])
def test_rank_is_contiguous_even_when_fewer_than_k_results_exist(
    kind: str, tmp_path
) -> None:
    if kind == "bm25":
        hits = lexical().search("northern", k=50)
    else:
        idx, vecs = dense(str(tmp_path / "ix") if kind == "persistent" else None)
        hits = idx.search(vecs[0], k=50)
    assert hits
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_retriever_labels_are_stable_across_index_types(tmp_path) -> None:
    mem, vecs = dense()
    per, _ = dense(str(tmp_path / "ix"))
    assert {h.retriever for h in mem.search(vecs[0], k=3)} == {"dense"}
    assert {h.retriever for h in per.search(vecs[0], k=3)} == {"dense"}
    assert {h.retriever for h in lexical().search("revenue", k=3)} == {"bm25"}


def test_rrf_over_both_retrievers_needs_no_off_by_one_fudge() -> None:
    """The shape retrieval will actually use: 1 / (rrf_k + rank), rank >= 1."""
    mem, vecs = dense()
    fused: dict[str, float] = {}
    for hits in (mem.search(vecs[0], k=5), lexical().search("quarterly revenue", k=5)):
        for hit in hits:
            assert hit.rank >= 1  # no zero-division, no negative contribution
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (60 + hit.rank)
    assert fused
    assert all(0 < v <= 2 / 61 for v in fused.values())


def test_hit_chunks_are_the_stored_objects_not_copies() -> None:
    """Retrieval reads `hit.chunk.row_refs` for citations and eval grading."""
    mem, vecs = dense()
    hit = mem.search(vecs[2], k=1)[0]
    assert hit.chunk is mem.get(hit.chunk_id)
    assert hit.chunk.row_refs
    assert hit.chunk_id == hit.chunk.chunk_id
    assert isinstance(hit.score, float) and not isinstance(hit.score, np.floating)
