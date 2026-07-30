"""Shared builders for the index tests, plus a couple of self-checks.

`anyrag.ingest` is written by another agent and may not exist yet, so these
tests construct `Chunk` objects directly rather than going through a chunker.
The helpers here exist so the individual index test modules can talk about
"twelve order rows from two sources" without twelve lines of boilerplate each.
"""

from __future__ import annotations

import numpy as np

from anyrag.core.types import Chunk, ChunkKind, RowRef


def make_chunk(
    chunk_id: str,
    text: str = "",
    *,
    source_id: str = "sqlite:test",
    kind: ChunkKind = ChunkKind.ROW,
    table: str | None = "orders",
    pk: str | None = None,
    **meta: object,
) -> Chunk:
    """A `Chunk` with sane defaults; extra kwargs land in `meta`."""
    full_meta: dict[str, object] = {"content_hash": f"h-{chunk_id}"}
    if table is not None:
        full_meta["table"] = table
    full_meta.update(meta)
    refs = ()
    if table is not None:
        refs = (RowRef(table=table, pk=pk if pk is not None else chunk_id),)
    return Chunk(
        chunk_id=chunk_id,
        source_id=source_id,
        kind=kind,
        text=text or f"row {chunk_id}",
        row_refs=refs,
        meta=full_meta,
    )


def unit(values: object) -> np.ndarray:
    """A unit-norm float32 vector, so dot product is cosine."""
    vec = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def random_unit_vectors(n: int, dim: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((n, dim)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return mat


def corpus(n: int, *, source_id: str = "sqlite:test", table: str = "orders"):
    return [
        make_chunk(f"{source_id}:{table}:{i}", f"order number {i}",
                   source_id=source_id, table=table, pk=str(i))
        for i in range(n)
    ]


def ids(hits) -> list[str]:
    return [h.chunk_id for h in hits]


# -- self-checks -----------------------------------------------------------


def test_make_chunk_defaults_are_coherent() -> None:
    chunk = make_chunk("c1")
    assert chunk.table == "orders"
    assert chunk.row_refs == (RowRef(table="orders", pk="c1"),)
    assert chunk.content_hash == "h-c1"


def test_unit_vectors_are_unit_norm() -> None:
    mat = random_unit_vectors(5, 8)
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-6)
