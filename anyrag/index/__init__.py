"""Indexes: dense (numpy) and lexical (BM25), both incrementally updatable.

Three implementations, one lifecycle contract
---------------------------------------------
* `MemoryVectorIndex` -- exact cosine search over a numpy matrix.
* `PersistentVectorIndex` -- the same index bound to a directory on disk.
* `BM25Index` -- Okapi BM25 over an inverted index, written here rather than
  taken from `rank_bm25` because that library rebuilds on every update.

All three implement `upsert / delete_by_source / delete_by_ids / search / get /
count / persist / load` from `anyrag.core.interfaces`, and all three treat
incremental update as the contract: re-upserting a chunk id that is already
present replaces it and leaves `count()` unchanged, and deletion reclaims the
storage rather than tombstoning it forever.

Rank convention (relied on by retrieval)
----------------------------------------
**`Hit.rank` is 1-based.** The top result of any `search()` has ``rank == 1``,
the k-th has ``rank == k``, and ranks are contiguous within a single call. Score
ties are broken by `chunk_id` so repeated searches over the same index return
identical orderings.

1-based was chosen over 0-based because `Hit.rank` defaults to 0 in the frozen
`anyrag.core.types.Hit`; under this convention that default reads as "unranked"
instead of silently claiming the top spot, and reciprocal-rank fusion can use
``1 / (rrf_k + hit.rank)`` with no off-by-one adjustment.

`Hit.retriever` is ``"dense"`` for both vector indexes and ``"bm25"`` for the
lexical one.

Filtering
---------
`MetadataFilter` is applied to the *candidate set*, before scoring, not to the
result list afterwards. A filtered `search(..., k=5)` therefore returns 5 hits
whenever 5 matching chunks exist. `PostingLists` in `anyrag.index.filters` keeps
`source_id`/`table`/`kind` inverted lists so neither pre-filtering nor
`delete_by_source` scans the whole index.

Persistence
-----------
An index is a directory: `vectors.npy` (dense only), `chunks.jsonl`,
`bm25.json` (lexical only) and `manifest.json`. No external database, no
pickle. `persist` then `load` into a fresh object round-trips exactly, including
`row_refs`, `meta` (tuples stay tuples) and any deletions performed before the
persist.
"""

from __future__ import annotations

from .bm25 import BM25Index, tokenize
from .filters import PostingLists, chunk_from_dict, chunk_to_dict
from .memory import CHUNKS_FILE, MANIFEST_FILE, VECTORS_FILE, MemoryVectorIndex
from .persistent import PersistentVectorIndex

#: The rank of the best hit returned by any index in this package.
FIRST_RANK = 1

__all__ = [
    "BM25Index",
    "MemoryVectorIndex",
    "PersistentVectorIndex",
    "PostingLists",
    "FIRST_RANK",
    "tokenize",
    "chunk_to_dict",
    "chunk_from_dict",
    "CHUNKS_FILE",
    "MANIFEST_FILE",
    "VECTORS_FILE",
]
