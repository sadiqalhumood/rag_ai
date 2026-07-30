"""Dense index bound to a directory on disk.

This is `MemoryVectorIndex` with a home: a default path, optional load-on-open,
and optional write-through so callers that care about durability do not have to
remember to call `persist`. Search, upsert and delete are inherited verbatim --
duplicating the scoring path would mean two places to get cosine, filtering and
the rank convention subtly different, which is exactly the bug a hybrid
retriever would surface as "bm25 and dense disagree about what rank 1 means".

Storage is a plain directory (`vectors.npy` + `chunks.jsonl` + `manifest.json`),
so an index can be inspected with `head` and no database has to exist for the
eval harness to run.

`Hit.rank` is 1-based here as everywhere in `anyrag.index`; see
`anyrag.index.memory` for the full statement of the convention.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from anyrag.core.types import Chunk

from .memory import MANIFEST_FILE, MemoryVectorIndex

__all__ = ["PersistentVectorIndex"]


class PersistentVectorIndex(MemoryVectorIndex):
    """A `MemoryVectorIndex` that survives process restart.

    Parameters
    ----------
    path:
        Directory the index lives in. Created on first `persist`.
    dim:
        Vector width, if known up front; otherwise adopted from the first upsert
        or from the manifest on load.
    autoload:
        When True (the default) an existing index at `path` is loaded during
        construction, so ``PersistentVectorIndex(path)`` in a fresh process sees
        the state the previous process left behind -- deletions included, since
        deleted rows are simply never written.
    autopersist:
        When True every mutating call flushes to disk. Off by default: ingest
        upserts in batches and one flush at the end is far cheaper. Deletions
        are covered too, so a crash cannot resurrect a deleted chunk.
    """

    def __init__(
        self,
        path: str | None = None,
        dim: int | None = None,
        *,
        autoload: bool = True,
        autopersist: bool = False,
    ) -> None:
        super().__init__(dim=dim, path=path)
        self.autopersist = autopersist
        if autoload and path and self.exists(path):
            self.load(path)

    @staticmethod
    def exists(path: str) -> bool:
        """True when `path` looks like a persisted index directory."""
        return os.path.isfile(os.path.join(path, MANIFEST_FILE))

    # -- write-through overrides -----------------------------------------

    def upsert(self, chunks: Sequence[Chunk], vectors: Any) -> int:
        written = super().upsert(chunks, vectors)
        if written and self.autopersist:
            self.persist()
        return written

    def delete_by_ids(self, chunk_ids: Sequence[str]) -> int:
        removed = super().delete_by_ids(chunk_ids)
        if removed and self.autopersist:
            self.persist()
        return removed

    def clear(self) -> None:
        super().clear()
        if self.autopersist and self.path:
            self.persist()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{type(self).__name__}(path={self.path!r}, dim={self.dim}, "
            f"count={self.count()})"
        )
