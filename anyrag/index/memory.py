"""Exact dense index over unit-norm vectors, held in a numpy matrix.

Embedders in this project emit unit-norm vectors, so a dot product *is* cosine
similarity and search is one matrix-vector multiply. At eval scale (tens of
thousands of chunks) exact search is both faster and more honest than an ANN
structure: there is no recall knob to tune and no index to rebuild.

The design constraint that shapes everything here is **incremental update**. A
`chunk_id -> row` map lets `upsert` overwrite a row in place, and deletion
swaps the last live row into the hole and shrinks, so re-upserting the same ids
leaves `count()` unchanged and deleting really reclaims the slot. Nothing in
this module ever rebuilds the matrix from scratch except `load`.

Rank convention
---------------
**`Hit.rank` is 1-based**: the best result of a search has ``rank == 1``, the
k-th has ``rank == k``. Ranks are assigned per call and are always contiguous
starting from 1. The `Hit` dataclass defaults `rank` to 0, which under this
convention reads as "not ranked yet" rather than as a spurious top position --
that is the reason for choosing 1-based. Every index in `anyrag.index` follows
this, so reciprocal-rank fusion can use ``1 / (rrf_k + hit.rank)`` directly.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator, Sequence

import numpy as np

from anyrag.core.config import MetadataFilter
from anyrag.core.errors import IndexError_
from anyrag.core.types import Chunk, Hit

from .filters import PostingLists, chunk_from_dict, chunk_to_dict

__all__ = ["MemoryVectorIndex"]

#: Sidecar filenames inside an index directory.
VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"

FORMAT = "anyrag-index/1"

_MIN_CAPACITY = 64


class MemoryVectorIndex:
    """In-memory `VectorIndex`: exact cosine search with incremental upsert.

    `dim` may be left unset and is then adopted from the first `upsert`; a later
    vector of a different width raises `IndexError_`.

    See the module docstring for the `Hit.rank` convention (1-based).
    """

    #: Value of `Hit.retriever` on every hit this index produces.
    retriever = "dense"

    def __init__(self, dim: int | None = None, path: str | None = None) -> None:
        self._dim: int | None = int(dim) if dim else None
        self._path = path
        self._buf: np.ndarray | None = None  # (capacity, dim) float32
        self._n = 0
        self._chunks: list[Chunk] = []  # row -> chunk, len == _n
        self._row_of: dict[str, int] = {}
        self._postings = PostingLists()
        if self._dim:
            self._buf = np.zeros((_MIN_CAPACITY, self._dim), dtype=np.float32)

    # -- introspection ----------------------------------------------------

    @property
    def dim(self) -> int | None:
        return self._dim

    @property
    def path(self) -> str | None:
        return self._path

    def count(self) -> int:
        return self._n

    def __len__(self) -> int:
        return self._n

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._row_of

    def get(self, chunk_id: str) -> Chunk | None:
        row = self._row_of.get(chunk_id)
        return None if row is None else self._chunks[row]

    def vector_of(self, chunk_id: str) -> np.ndarray | None:
        """The stored vector for `chunk_id`, as a copy, or None."""
        row = self._row_of.get(chunk_id)
        if row is None or self._buf is None:
            return None
        return np.array(self._buf[row], dtype=np.float32)

    def iter_chunks(self) -> Iterator[Chunk]:
        return iter(list(self._chunks))

    def sources(self) -> tuple[str, ...]:
        return self._postings.sources

    # -- writes -----------------------------------------------------------

    def upsert(self, chunks: Sequence[Chunk], vectors: Any) -> int:
        """Insert new chunk ids and overwrite existing ones, in place.

        Returns the number of chunks written (inserted plus updated). Re-upserting
        ids that are already present leaves `count()` unchanged -- that property,
        not the return value, is the contract. Within a single call the last
        occurrence of a repeated `chunk_id` wins.
        """
        chunks = list(chunks)
        if not chunks:
            return 0

        mat = self._as_matrix(vectors, len(chunks))
        for i, chunk in enumerate(chunks):
            self._write_row(chunk, mat[i])
        return len(chunks)

    def _as_matrix(self, vectors: Any, n_chunks: int) -> np.ndarray:
        try:
            mat = np.asarray(vectors, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise IndexError_(f"vectors are not array-like: {exc}") from exc
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        if mat.ndim != 2:
            raise IndexError_(
                f"vectors must be a 2-D (n, dim) array, got shape {mat.shape}"
            )
        if mat.shape[0] != n_chunks:
            raise IndexError_(
                f"got {n_chunks} chunks but {mat.shape[0]} vectors; "
                "they must correspond one-to-one and in order"
            )
        if self._dim is None:
            self._dim = int(mat.shape[1])
            if self._dim == 0:
                raise IndexError_("vectors must have a non-zero dimension")
            self._buf = np.zeros((_MIN_CAPACITY, self._dim), dtype=np.float32)
        elif mat.shape[1] != self._dim:
            raise IndexError_(
                f"vector dimension mismatch: index holds {self._dim}-d vectors, "
                f"got {mat.shape[1]}-d"
            )
        return mat

    def _write_row(self, chunk: Chunk, vector: np.ndarray) -> None:
        assert self._buf is not None  # _as_matrix guarantees it
        row = self._row_of.get(chunk.chunk_id)
        if row is None:
            self._grow(self._n + 1)
            row = self._n
            self._n += 1
            self._chunks.append(chunk)
            self._row_of[chunk.chunk_id] = row
        else:
            self._chunks[row] = chunk
        self._buf[row] = vector
        self._postings.add(chunk)

    def delete_by_ids(self, chunk_ids: Sequence[str]) -> int:
        """Remove the given ids; returns how many were actually present.

        Ids that are absent are counted zero times, and a duplicate id in the
        argument is counted once.
        """
        removed = 0
        seen: set[str] = set()
        for cid in chunk_ids:
            if cid in seen:
                continue
            seen.add(cid)
            if self._remove_one(cid):
                removed += 1
        if removed:
            self._maybe_shrink()
        return removed

    def delete_by_source(self, source_id: str) -> int:
        """Remove every chunk from `source_id`; returns how many were removed.

        Uses the `source_id` posting list, so the cost is proportional to the
        number of chunks deleted, not to the size of the index.
        """
        return self.delete_by_ids(sorted(self._postings.ids_for_source(source_id)))

    def _remove_one(self, chunk_id: str) -> bool:
        row = self._row_of.pop(chunk_id, None)
        if row is None:
            return False
        victim = self._chunks[row]
        last = self._n - 1
        if row != last:
            # Swap the final live row into the hole: O(1), and the matrix stays
            # dense so search never scores a tombstone.
            assert self._buf is not None
            self._buf[row] = self._buf[last]
            moved = self._chunks[last]
            self._chunks[row] = moved
            self._row_of[moved.chunk_id] = row
        self._chunks.pop()
        self._n -= 1
        self._postings.discard(victim)
        return True

    def clear(self) -> None:
        self._n = 0
        self._chunks = []
        self._row_of = {}
        self._postings.clear()
        if self._dim:
            self._buf = np.zeros((_MIN_CAPACITY, self._dim), dtype=np.float32)

    # -- capacity ---------------------------------------------------------

    def _grow(self, needed: int) -> None:
        assert self._buf is not None
        cap = self._buf.shape[0]
        if needed <= cap:
            return
        new_cap = max(_MIN_CAPACITY, cap * 2)
        while new_cap < needed:
            new_cap *= 2
        buf = np.zeros((new_cap, self._dim or 0), dtype=np.float32)
        buf[: self._n] = self._buf[: self._n]
        self._buf = buf

    def _maybe_shrink(self) -> None:
        """Give memory back after large deletions; slots must not leak."""
        if self._buf is None:
            return
        cap = self._buf.shape[0]
        if cap > _MIN_CAPACITY and self._n * 4 <= cap:
            new_cap = max(_MIN_CAPACITY, self._n * 2)
            buf = np.zeros((new_cap, self._dim or 0), dtype=np.float32)
            buf[: self._n] = self._buf[: self._n]
            self._buf = buf

    @property
    def capacity(self) -> int:
        return 0 if self._buf is None else int(self._buf.shape[0])

    # -- search -----------------------------------------------------------

    def search(
        self, vector: Any, k: int = 10, flt: MetadataFilter | None = None
    ) -> list[Hit]:
        """Top-`k` chunks by cosine similarity, filtered *before* ranking.

        The filter is applied to the candidate set rather than to the results,
        so a filtered search still returns `k` hits whenever `k` matching chunks
        exist. Hits come back in descending score order with `rank` 1..k (ties
        broken by `chunk_id` so the ordering is deterministic) and
        ``retriever == "dense"``. An empty index returns `[]`.
        """
        if k <= 0 or self._n == 0 or self._buf is None:
            return []

        try:
            q = np.asarray(vector, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise IndexError_(f"query vector is not array-like: {exc}") from exc
        if q.shape[0] != self._dim:
            raise IndexError_(
                f"vector dimension mismatch: index holds {self._dim}-d vectors, "
                f"got {q.shape[0]}-d query"
            )

        rows = self._candidate_rows(flt)
        if rows is None:
            sub = self._buf[: self._n]
            row_ids = None
        else:
            if not rows:
                return []
            row_ids = np.asarray(rows, dtype=np.intp)
            sub = self._buf[row_ids]

        scores = sub @ q
        n = int(scores.shape[0])
        take = min(k, n)
        if take < n:
            cand = np.argpartition(-scores, take - 1)[:take]
        else:
            cand = np.arange(n)

        def row_for(i: int) -> int:
            return int(i if row_ids is None else row_ids[i])

        order = sorted(
            (int(i) for i in cand),
            key=lambda i: (-float(scores[i]), self._chunks[row_for(i)].chunk_id),
        )
        return [
            Hit(
                chunk=self._chunks[row_for(i)],
                score=float(scores[i]),
                rank=rank,
                retriever=self.retriever,
            )
            for rank, i in enumerate(order, start=1)
        ]

    def _candidate_rows(self, flt: MetadataFilter | None) -> list[int] | None:
        """Rows surviving `flt`, or None meaning "every live row"."""
        ids, needs_verify = self._postings.candidates(flt)
        if ids is None:
            if not needs_verify:
                return None
            assert flt is not None
            return [r for r in range(self._n) if flt.matches(self._chunks[r])]
        rows: list[int] = []
        for cid in ids:
            row = self._row_of.get(cid)
            if row is None:
                continue
            if needs_verify:
                assert flt is not None
                if not flt.matches(self._chunks[row]):
                    continue
            rows.append(row)
        return rows

    # -- persistence ------------------------------------------------------

    def persist(self, path: str | None = None) -> None:
        """Write the index to a directory: vectors `.npy` + a JSONL sidecar.

        No external database and no pickle: `vectors.npy` holds the live rows in
        row order, `chunks.jsonl` holds the matching chunks one per line, and
        `manifest.json` records the format, dim and count so `load` can detect a
        truncated or mismatched directory.
        """
        target = self._resolve(path)
        os.makedirs(target, exist_ok=True)
        mat = (
            self._buf[: self._n]
            if self._buf is not None
            else np.zeros((0, 0), dtype=np.float32)
        )
        np.save(os.path.join(target, VECTORS_FILE), np.ascontiguousarray(mat))
        with open(os.path.join(target, CHUNKS_FILE), "w", encoding="utf-8") as fh:
            for chunk in self._chunks:
                fh.write(json.dumps(chunk_to_dict(chunk), ensure_ascii=False) + "\n")
        manifest = {
            "format": FORMAT,
            "kind": "dense",
            "dim": self._dim,
            "count": self._n,
        }
        with open(os.path.join(target, MANIFEST_FILE), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
        self._path = target

    def load(self, path: str | None = None) -> None:
        """Replace this index's contents with the ones persisted at `path`."""
        target = self._resolve(path)
        manifest = read_manifest(target, expect_kind="dense")
        vec_path = os.path.join(target, VECTORS_FILE)
        if not os.path.exists(vec_path):
            raise IndexError_(f"no {VECTORS_FILE} in index directory {target!r}")
        mat = np.load(vec_path).astype(np.float32, copy=False)
        chunks = read_chunks(os.path.join(target, CHUNKS_FILE))
        if mat.shape[0] != len(chunks):
            raise IndexError_(
                f"index at {target!r} is inconsistent: {mat.shape[0]} vectors "
                f"but {len(chunks)} chunks"
            )

        self._dim = manifest.get("dim") or (int(mat.shape[1]) if mat.size else None)
        self._n = 0
        self._chunks = []
        self._row_of = {}
        self._postings.clear()
        self._buf = np.zeros(
            (max(_MIN_CAPACITY, len(chunks)), self._dim or 1), dtype=np.float32
        )
        if chunks:
            self._buf[: len(chunks)] = mat
            self._n = len(chunks)
            self._chunks = chunks
            for row, chunk in enumerate(chunks):
                self._row_of[chunk.chunk_id] = row
                self._postings.add(chunk)
        self._path = target

    def _resolve(self, path: str | None) -> str:
        target = path or self._path
        if not target:
            raise IndexError_(
                f"{type(self).__name__} has no path: pass one to persist()/load() "
                "or set it on the constructor"
            )
        return target


# --------------------------------------------------------------------------
# Sidecar readers, shared with the BM25 index
# --------------------------------------------------------------------------


def read_manifest(directory: str, *, expect_kind: str) -> dict[str, Any]:
    manifest_path = os.path.join(directory, MANIFEST_FILE)
    if not os.path.isdir(directory) or not os.path.exists(manifest_path):
        raise IndexError_(f"no persisted index at {directory!r}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("format") != FORMAT:
        raise IndexError_(
            f"index at {directory!r} has format {manifest.get('format')!r}, "
            f"expected {FORMAT!r}"
        )
    if manifest.get("kind") != expect_kind:
        raise IndexError_(
            f"index at {directory!r} is a {manifest.get('kind')!r} index, "
            f"not {expect_kind!r}"
        )
    return manifest


def read_chunks(path: str) -> list[Chunk]:
    if not os.path.exists(path):
        raise IndexError_(f"missing chunk sidecar {path!r}")
    out: list[Chunk] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(chunk_from_dict(json.loads(line)))
    return out
