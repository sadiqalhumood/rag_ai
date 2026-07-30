"""MemoryVectorIndex: incremental upsert, real deletion, pre-filtered search.

The headline property under test is that this is not a rebuild in disguise:
re-upserting the same ids must not change `count()`, and deleting must actually
give the slot back.
"""

from __future__ import annotations

import numpy as np
import pytest

from anyrag.core.config import MetadataFilter
from anyrag.core.errors import IndexError_
from anyrag.core.interfaces import VectorIndex
from anyrag.core.types import ChunkKind
from anyrag.index import MemoryVectorIndex
from test_index_fixtures import ids, make_chunk, random_unit_vectors, unit

DIM = 4


def build(n: int = 3, **kw) -> tuple[MemoryVectorIndex, list, np.ndarray]:
    chunks = [make_chunk(f"c{i}", f"row {i}", **kw) for i in range(n)]
    vecs = random_unit_vectors(n, DIM, seed=11)
    idx = MemoryVectorIndex(dim=DIM)
    idx.upsert(chunks, vecs)
    return idx, chunks, vecs


# -- protocol conformance --------------------------------------------------


def test_conforms_to_vector_index_protocol() -> None:
    assert isinstance(MemoryVectorIndex(dim=DIM), VectorIndex)


# -- the incremental-update contract ---------------------------------------


def test_reupserting_identical_chunks_leaves_count_unchanged() -> None:
    idx, chunks, vecs = build(5)
    assert idx.count() == 5
    idx.upsert(chunks, vecs)
    idx.upsert(chunks, vecs)
    assert idx.count() == 5
    assert sorted(c.chunk_id for c in idx.iter_chunks()) == [f"c{i}" for i in range(5)]


def test_upsert_with_changed_text_updates_in_place() -> None:
    idx, chunks, vecs = build(3)
    edited = [make_chunk("c1", "the revised text for row one")]
    idx.upsert(edited, vecs[1:2])
    assert idx.count() == 3
    assert idx.get("c1").text == "the revised text for row one"
    hits = idx.search(vecs[1], k=1)
    assert hits[0].chunk_id == "c1"
    assert hits[0].chunk.text == "the revised text for row one"


def test_upsert_with_changed_vector_moves_the_neighbourhood() -> None:
    idx = MemoryVectorIndex(dim=DIM)
    idx.upsert(
        [make_chunk("a"), make_chunk("b")],
        np.stack([unit([1, 0, 0, 0]), unit([0, 1, 0, 0])]),
    )
    query = unit([1, 0, 0, 0])
    assert idx.search(query, k=1)[0].chunk_id == "a"
    # Re-point "b" at the query direction; "a" keeps its old vector.
    idx.upsert([make_chunk("b")], unit([1, 0, 0, 0]).reshape(1, -1))
    assert idx.count() == 2
    assert ids(idx.search(query, k=2))[0] in {"a", "b"}
    assert idx.search(query, k=2)[1].score == pytest.approx(1.0, abs=1e-5)


def test_duplicate_ids_within_one_batch_last_wins() -> None:
    idx = MemoryVectorIndex(dim=DIM)
    written = idx.upsert(
        [make_chunk("dup", "first"), make_chunk("dup", "second")],
        np.stack([unit([1, 0, 0, 0]), unit([0, 1, 0, 0])]),
    )
    assert written == 2
    assert idx.count() == 1
    assert idx.get("dup").text == "second"


def test_empty_upsert_is_a_noop() -> None:
    idx = MemoryVectorIndex(dim=DIM)
    assert idx.upsert([], np.zeros((0, DIM), dtype=np.float32)) == 0
    assert idx.count() == 0


# -- deletion --------------------------------------------------------------


def test_delete_by_source_removes_exactly_those_chunks() -> None:
    idx = MemoryVectorIndex(dim=DIM)
    a = [make_chunk(f"a{i}", source_id="sqlite:A") for i in range(4)]
    b = [make_chunk(f"b{i}", source_id="sqlite:B") for i in range(3)]
    idx.upsert(a + b, random_unit_vectors(7, DIM, seed=3))
    assert idx.delete_by_source("sqlite:A") == 4
    assert idx.count() == 3
    assert sorted(c.chunk_id for c in idx.iter_chunks()) == ["b0", "b1", "b2"]
    assert idx.get("a0") is None
    assert idx.get("b0") is not None


def test_delete_by_source_for_unknown_source_removes_nothing() -> None:
    idx, _, _ = build(3)
    assert idx.delete_by_source("sqlite:nope") == 0
    assert idx.count() == 3


def test_delete_by_ids_counts_only_ids_that_were_present() -> None:
    idx, _, _ = build(5)
    removed = idx.delete_by_ids(["c0", "c4", "ghost", "c0"])
    assert removed == 2  # duplicate counted once, absent id not counted
    assert idx.count() == 3
    assert idx.delete_by_ids([]) == 0


def test_deletion_reclaims_slots_rather_than_tombstoning() -> None:
    idx = MemoryVectorIndex(dim=DIM)
    chunks = [make_chunk(f"c{i}") for i in range(500)]
    idx.upsert(chunks, random_unit_vectors(500, DIM, seed=5))
    fat = idx.capacity
    assert idx.delete_by_ids([f"c{i}" for i in range(490)]) == 490
    assert idx.count() == 10
    assert idx.capacity < fat  # storage actually given back
    # Re-adding does not grow past what a fresh index would need.
    idx.upsert(chunks[:490], random_unit_vectors(490, DIM, seed=5))
    assert idx.count() == 500


def test_deleted_chunks_disappear_from_search() -> None:
    idx = MemoryVectorIndex(dim=DIM)
    idx.upsert(
        [make_chunk("a"), make_chunk("b"), make_chunk("c")],
        np.stack([unit([1, 0, 0, 0]), unit([0.9, 0.1, 0, 0]), unit([0, 0, 1, 0])]),
    )
    query = unit([1, 0, 0, 0])
    assert ids(idx.search(query, k=3))[:2] == ["a", "b"]
    idx.delete_by_ids(["a"])
    assert ids(idx.search(query, k=3)) == ["b", "c"]


def test_clear_empties_the_index() -> None:
    idx, _, _ = build(4)
    idx.clear()
    assert idx.count() == 0
    assert idx.search(unit([1, 0, 0, 0]), k=3) == []


# -- search, ranking, filtering -------------------------------------------


def test_empty_index_search_returns_empty_list() -> None:
    assert MemoryVectorIndex(dim=DIM).search(unit([1, 0, 0, 0]), k=5) == []
    assert MemoryVectorIndex().search([1.0, 0.0], k=5) == []


def test_rank_is_one_based_contiguous_and_labelled_dense() -> None:
    idx, _, vecs = build(6)
    hits = idx.search(vecs[0], k=4)
    assert len(hits) == 4
    assert [h.rank for h in hits] == [1, 2, 3, 4]
    assert all(h.retriever == "dense" for h in hits)
    assert hits[0].chunk_id == "c0"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_scores_are_descending_and_ties_are_deterministic() -> None:
    idx = MemoryVectorIndex(dim=DIM)
    same = unit([1, 0, 0, 0])
    idx.upsert(
        [make_chunk("z"), make_chunk("m"), make_chunk("a")],
        np.stack([same, same, same]),
    )
    first = ids(idx.search(same, k=3))
    assert first == ["a", "m", "z"]  # tie broken by chunk_id
    assert first == ids(idx.search(same, k=3))


def test_k_larger_than_count_returns_everything() -> None:
    idx, _, vecs = build(3)
    hits = idx.search(vecs[0], k=99)
    assert len(hits) == 3
    assert [h.rank for h in hits] == [1, 2, 3]


def test_non_positive_k_returns_empty() -> None:
    idx, _, vecs = build(3)
    assert idx.search(vecs[0], k=0) == []
    assert idx.search(vecs[0], k=-1) == []


def test_filtered_search_returns_k_results_when_k_are_available() -> None:
    """The filter runs before ranking, so filtering must not starve top-k."""
    idx = MemoryVectorIndex(dim=DIM)
    chunks, vecs = [], []
    # 40 chunks from the wrong table sit closest to the query; 5 from the
    # right one sit further away. A post-filter would return zero of them.
    query = unit([1, 0, 0, 0])
    for i in range(40):
        chunks.append(make_chunk(f"near{i}", table="customers"))
        vecs.append(unit([1.0, 0.01 * i, 0, 0]))
    for i in range(5):
        chunks.append(make_chunk(f"far{i}", table="orders"))
        vecs.append(unit([0.1, 1.0, 0.05 * i, 0]))
    idx.upsert(chunks, np.stack(vecs))

    flt = MetadataFilter(tables=frozenset({"orders"}))
    hits = idx.search(query, k=5, flt=flt)
    assert len(hits) == 5
    assert all(h.chunk.table == "orders" for h in hits)
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]


def test_filter_by_kind_source_and_equals() -> None:
    idx = MemoryVectorIndex(dim=DIM)
    chunks = [
        make_chunk("row-a", source_id="sqlite:A", kind=ChunkKind.ROW, lang="en"),
        make_chunk("row-b", source_id="sqlite:B", kind=ChunkKind.ROW, lang="ar"),
        make_chunk(
            "card-a", source_id="sqlite:A", kind=ChunkKind.SCHEMA_CARD, lang="en"
        ),
    ]
    idx.upsert(chunks, random_unit_vectors(3, DIM, seed=2))
    query = unit([1, 1, 1, 1])

    by_kind = idx.search(query, k=5, flt=MetadataFilter(kinds=frozenset({ChunkKind.SCHEMA_CARD})))
    assert ids(by_kind) == ["card-a"]

    by_source = idx.search(query, k=5, flt=MetadataFilter(source_ids=frozenset({"sqlite:B"})))
    assert ids(by_source) == ["row-b"]

    by_equals = idx.search(query, k=5, flt=MetadataFilter(equals={"lang": "ar"}))
    assert ids(by_equals) == ["row-b"]

    combined = idx.search(
        query,
        k=5,
        flt=MetadataFilter(source_ids=frozenset({"sqlite:A"}), equals={"lang": "en"}),
    )
    assert sorted(ids(combined)) == ["card-a", "row-a"]


def test_filter_matching_nothing_returns_empty() -> None:
    idx, _, vecs = build(3)
    flt = MetadataFilter(tables=frozenset({"no_such_table"}))
    assert idx.search(vecs[0], k=3, flt=flt) == []


def test_filter_follows_the_chunk_after_an_upsert_changes_its_table() -> None:
    """Stale postings would keep a chunk visible under its old table."""
    idx = MemoryVectorIndex(dim=DIM)
    vec = unit([1, 0, 0, 0]).reshape(1, -1)
    idx.upsert([make_chunk("c", table="orders")], vec)
    idx.upsert([make_chunk("c", table="invoices")], vec)
    assert idx.count() == 1
    assert idx.search(vec[0], k=3, flt=MetadataFilter(tables=frozenset({"orders"}))) == []
    assert ids(idx.search(vec[0], k=3, flt=MetadataFilter(tables=frozenset({"invoices"})))) == ["c"]


# -- errors ----------------------------------------------------------------


def test_dimension_mismatch_on_upsert_raises() -> None:
    idx = MemoryVectorIndex(dim=DIM)
    idx.upsert([make_chunk("a")], np.zeros((1, DIM), dtype=np.float32))
    with pytest.raises(IndexError_, match="dimension mismatch"):
        idx.upsert([make_chunk("b")], np.zeros((1, DIM + 3), dtype=np.float32))


def test_dimension_mismatch_on_query_raises() -> None:
    idx, _, _ = build(2)
    with pytest.raises(IndexError_, match="dimension mismatch"):
        idx.search(np.zeros(DIM + 1, dtype=np.float32), k=1)


def test_dimension_is_adopted_from_the_first_upsert() -> None:
    idx = MemoryVectorIndex()
    idx.upsert([make_chunk("a")], np.zeros((1, 9), dtype=np.float32))
    assert idx.dim == 9
    with pytest.raises(IndexError_, match="dimension mismatch"):
        idx.upsert([make_chunk("b")], np.zeros((1, 8), dtype=np.float32))


def test_chunk_vector_count_mismatch_raises() -> None:
    idx = MemoryVectorIndex(dim=DIM)
    with pytest.raises(IndexError_, match="one-to-one"):
        idx.upsert([make_chunk("a"), make_chunk("b")], np.zeros((3, DIM), dtype=np.float32))


def test_get_returns_none_for_absent_id() -> None:
    idx, _, _ = build(2)
    assert idx.get("nope") is None
    assert "c0" in idx and "nope" not in idx


def test_persist_without_a_path_raises() -> None:
    with pytest.raises(IndexError_, match="no path"):
        MemoryVectorIndex(dim=DIM).persist()
