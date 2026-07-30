"""PersistentVectorIndex: a persist/load cycle must change nothing observable.

The load-bearing test is the round trip through a *fresh object* -- reusing the
same instance would pass even if `load` were a no-op.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from anyrag.core.config import MetadataFilter
from anyrag.core.errors import IndexError_
from anyrag.core.interfaces import VectorIndex
from anyrag.core.types import Chunk, ChunkKind, RowRef
from anyrag.index import CHUNKS_FILE, MANIFEST_FILE, VECTORS_FILE, PersistentVectorIndex
from test_index_fixtures import ids, make_chunk, random_unit_vectors, unit

DIM = 16
N = 60


def populate(path: str) -> tuple[PersistentVectorIndex, list, np.ndarray]:
    chunks = [
        make_chunk(
            f"c{i}",
            f"row {i} about widgets",
            source_id="sqlite:A" if i % 2 else "sqlite:B",
            table="orders" if i % 3 else "customers",
            pk=str(i),
        )
        for i in range(N)
    ]
    vecs = random_unit_vectors(N, DIM, seed=17)
    idx = PersistentVectorIndex(path, dim=DIM)
    idx.upsert(chunks, vecs)
    return idx, chunks, vecs


def test_conforms_to_vector_index_protocol(tmp_path) -> None:
    assert isinstance(PersistentVectorIndex(str(tmp_path / "ix"), dim=DIM), VectorIndex)


def test_persist_then_fresh_load_gives_identical_search_results(tmp_path) -> None:
    path = str(tmp_path / "ix")
    idx, _, vecs = populate(path)
    queries = random_unit_vectors(5, DIM, seed=99)
    before = [idx.search(q, k=10) for q in queries]
    idx.persist()

    fresh = PersistentVectorIndex(path, autoload=False)
    assert fresh.count() == 0
    fresh.load(path)

    assert fresh.count() == idx.count() == N
    assert fresh.dim == DIM
    for q, expected in zip(queries, before):
        got = fresh.search(q, k=10)
        assert ids(got) == ids(expected)
        assert [h.rank for h in got] == [h.rank for h in expected]
        assert [h.retriever for h in got] == ["dense"] * len(got)
        for a, b in zip(got, expected):
            assert a.score == pytest.approx(b.score, abs=1e-6)


def test_deletions_survive_a_persist_load_cycle(tmp_path) -> None:
    path = str(tmp_path / "ix")
    idx, _, _ = populate(path)
    removed = idx.delete_by_source("sqlite:A")
    assert removed == N // 2
    idx.delete_by_ids(["c0", "c2"])
    survivors = sorted(c.chunk_id for c in idx.iter_chunks())
    idx.persist()

    fresh = PersistentVectorIndex(path, autoload=False)
    fresh.load(path)
    assert sorted(c.chunk_id for c in fresh.iter_chunks()) == survivors
    assert fresh.count() == N - removed - 2
    assert fresh.get("c1") is None
    # The postings were rebuilt on load, so delete_by_source still works.
    assert fresh.delete_by_source("sqlite:A") == 0
    remaining = fresh.count()
    assert fresh.delete_by_source("sqlite:B") == remaining
    assert fresh.count() == 0


def test_row_refs_and_meta_round_trip_exactly(tmp_path) -> None:
    path = str(tmp_path / "ix")
    chunk = Chunk(
        chunk_id="rich",
        source_id="sqlite:A",
        kind=ChunkKind.SCHEMA_CARD,
        text="جدول الطلبات: رقم الطلب، المبلغ",
        row_refs=(RowRef(table="orders", pk=7), RowRef(table="orders", pk="8")),
        meta={
            "table": "orders",
            "columns": ("order_id", "amount"),
            "part_index": 0,
            "n_parts": 1,
            "lang": "ar",
            "nested": {"date_min": "2024-01-01", "tags": ("a", "b")},
            "ratio": 0.5,
            "flag": True,
            "nothing": None,
        },
    )
    idx = PersistentVectorIndex(path, dim=DIM)
    idx.upsert([chunk], random_unit_vectors(1, DIM, seed=1))
    idx.persist()

    fresh = PersistentVectorIndex(path)  # autoload
    got = fresh.get("rich")
    assert got == chunk
    assert got.row_refs == (RowRef("orders", "7"), RowRef("orders", "8"))
    # Tuples must stay tuples: MetadataFilter.equals compares with !=.
    assert got.meta["columns"] == ("order_id", "amount")
    assert isinstance(got.meta["nested"]["tags"], tuple)
    flt = MetadataFilter(equals={"columns": ("order_id", "amount")})
    assert ids(fresh.search(unit(np.ones(DIM)), k=1, flt=flt)) == ["rich"]


def test_autoload_picks_up_an_existing_directory(tmp_path) -> None:
    path = str(tmp_path / "ix")
    idx, _, vecs = populate(path)
    idx.persist()
    reopened = PersistentVectorIndex(path)
    assert reopened.count() == N
    assert reopened.search(vecs[3], k=1)[0].chunk_id == "c3"


def test_autoload_on_a_missing_directory_starts_empty(tmp_path) -> None:
    idx = PersistentVectorIndex(str(tmp_path / "nothing-here"))
    assert idx.count() == 0
    assert not PersistentVectorIndex.exists(str(tmp_path / "nothing-here"))


def test_autopersist_flushes_writes_and_deletes(tmp_path) -> None:
    path = str(tmp_path / "ix")
    idx = PersistentVectorIndex(path, dim=DIM, autopersist=True)
    idx.upsert([make_chunk("a"), make_chunk("b")], random_unit_vectors(2, DIM, seed=4))
    assert PersistentVectorIndex(path).count() == 2
    idx.delete_by_ids(["a"])
    assert sorted(c.chunk_id for c in PersistentVectorIndex(path).iter_chunks()) == ["b"]


def test_persisted_layout_is_inspectable_files_not_a_database(tmp_path) -> None:
    path = str(tmp_path / "ix")
    idx, _, _ = populate(path)
    idx.persist()
    assert sorted(os.listdir(path)) == sorted([VECTORS_FILE, CHUNKS_FILE, MANIFEST_FILE])
    manifest = json.loads(open(os.path.join(path, MANIFEST_FILE), encoding="utf-8").read())
    assert manifest["kind"] == "dense" and manifest["dim"] == DIM and manifest["count"] == N
    with open(os.path.join(path, CHUNKS_FILE), encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    assert len(lines) == N
    assert json.loads(lines[0])["chunk_id"] == "c0"
    assert np.load(os.path.join(path, VECTORS_FILE)).shape == (N, DIM)


def test_upserting_after_a_load_is_still_incremental(tmp_path) -> None:
    path = str(tmp_path / "ix")
    idx, chunks, vecs = populate(path)
    idx.persist()
    fresh = PersistentVectorIndex(path)
    fresh.upsert(chunks, vecs)  # same ids again
    assert fresh.count() == N
    fresh.upsert([make_chunk("brand-new")], random_unit_vectors(1, DIM, seed=8))
    assert fresh.count() == N + 1


def test_load_from_a_missing_directory_raises(tmp_path) -> None:
    with pytest.raises(IndexError_, match="no persisted index"):
        PersistentVectorIndex(dim=DIM).load(str(tmp_path / "absent"))


def test_load_rejects_an_index_of_the_wrong_kind(tmp_path) -> None:
    from anyrag.index import BM25Index

    path = str(tmp_path / "lex")
    lex = BM25Index(path=path)
    lex.upsert([make_chunk("a", "hello")])
    lex.persist()
    with pytest.raises(IndexError_, match="not 'dense'"):
        PersistentVectorIndex(dim=DIM).load(path)


def test_load_detects_a_truncated_sidecar(tmp_path) -> None:
    path = str(tmp_path / "ix")
    idx, _, _ = populate(path)
    idx.persist()
    with open(os.path.join(path, CHUNKS_FILE), encoding="utf-8") as fh:
        lines = fh.readlines()
    with open(os.path.join(path, CHUNKS_FILE), "w", encoding="utf-8") as fh:
        fh.writelines(lines[:-5])
    with pytest.raises(IndexError_, match="inconsistent"):
        PersistentVectorIndex(dim=DIM).load(path)
