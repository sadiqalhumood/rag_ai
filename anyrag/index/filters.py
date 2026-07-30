"""Posting lists and (de)serialization helpers shared by every index.

Two jobs, both of them boring on purpose:

1. **Posting lists.** `delete_by_source` and pre-filtered search must not be
   linear scans over the whole corpus. `PostingLists` maintains
   ``source_id -> {chunk_id}``, ``table -> {chunk_id}`` and
   ``kind -> {chunk_id}`` so a `MetadataFilter` can be turned into a candidate
   set by intersecting a few small sets, and so `delete_by_source` is O(deleted)
   rather than O(indexed).

2. **Chunk (de)serialization.** Persistence is a JSONL sidecar in every index,
   dense or lexical, so the encoder lives here rather than being written twice.
   Round-tripping is *exact*: `RowRef` tuples come back as `RowRef` tuples and
   tuples nested in `meta` come back as tuples, not lists, because
   `MetadataFilter.equals` compares metadata values with `!=` and a tuple that
   silently became a list would stop matching after a reload.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from anyrag.core.config import MetadataFilter
from anyrag.core.errors import IndexError_
from anyrag.core.types import Chunk, ChunkKind, RowRef

__all__ = [
    "PostingLists",
    "chunk_to_dict",
    "chunk_from_dict",
]


# --------------------------------------------------------------------------
# Posting lists
# --------------------------------------------------------------------------


class PostingLists:
    """Inverted lists over the metadata dimensions `MetadataFilter` filters on.

    Only the dimensions with a bounded, hashable vocabulary get postings:
    `source_id`, `meta["table"]` and `kind`. `MetadataFilter.equals` is
    open-ended, so it is verified per-candidate instead -- see `candidates`.
    """

    __slots__ = ("_by_source", "_by_table", "_by_kind", "_ids")

    def __init__(self) -> None:
        self._by_source: dict[str, set[str]] = {}
        self._by_table: dict[Any, set[str]] = {}
        self._by_kind: dict[Any, set[str]] = {}
        self._ids: set[str] = set()

    # -- mutation ---------------------------------------------------------

    def add(self, chunk: Chunk) -> None:
        """Register `chunk`, replacing any earlier registration of that id.

        An upsert that changes a chunk's `source_id`, `table` or `kind` must
        not leave the old posting behind, so a re-add drops the id from every
        bucket first. The key vocabularies are bounded (sources, tables, kinds),
        so that sweep is cheap.
        """
        cid = chunk.chunk_id
        if cid in self._ids:
            self._drop_everywhere(cid)
        self._ids.add(cid)
        self._by_source.setdefault(chunk.source_id, set()).add(cid)
        self._by_table.setdefault(chunk.meta.get("table"), set()).add(cid)
        self._by_kind.setdefault(chunk.kind, set()).add(cid)

    def discard(self, chunk_or_id: Chunk | str) -> None:
        """Unregister a chunk (or bare id). Silent when it was never present."""
        cid = (
            chunk_or_id
            if isinstance(chunk_or_id, str)
            else chunk_or_id.chunk_id
        )
        if cid not in self._ids:
            return
        self._ids.discard(cid)
        self._drop_everywhere(cid)

    def _drop_everywhere(self, cid: str) -> None:
        for d in (self._by_source, self._by_table, self._by_kind):
            self._drop_from(d, cid)

    def clear(self) -> None:
        self._by_source.clear()
        self._by_table.clear()
        self._by_kind.clear()
        self._ids.clear()

    @staticmethod
    def _drop_from(d: dict[Any, set[str]], cid: str) -> None:
        empty = []
        for key, bucket in d.items():
            if cid in bucket:
                bucket.discard(cid)
                if not bucket:
                    empty.append(key)
        for key in empty:
            del d[key]

    # -- lookup -----------------------------------------------------------

    def ids_for_source(self, source_id: str) -> set[str]:
        return set(self._by_source.get(source_id, ()))

    def ids_for_table(self, table: str | None) -> set[str]:
        return set(self._by_table.get(table, ()))

    def ids_for_kind(self, kind: ChunkKind) -> set[str]:
        return set(self._by_kind.get(kind, ()))

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_source))

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._ids

    def __len__(self) -> int:
        return len(self._ids)

    def candidates(self, flt: MetadataFilter | None) -> tuple[set[str] | None, bool]:
        """Narrow `flt` to a candidate id set using the postings.

        Returns ``(ids, needs_verify)``:

        * ``ids is None`` means "no posting-backed restriction" -- the caller
          should consider every indexed chunk.
        * ``needs_verify`` is True when `flt` carries `equals` conditions that
          the postings cannot answer, so the caller must still run
          `flt.matches(chunk)` on each candidate.

        The narrowing is exact for `kinds`, `tables` and `source_ids`: a chunk
        outside the returned set can never match, and when `needs_verify` is
        False every chunk inside it does match.
        """
        if flt is None or flt.is_empty:
            return None, False

        buckets: list[set[str]] = []
        if flt.source_ids is not None:
            buckets.append(self._union(self._by_source, flt.source_ids))
        if flt.tables is not None:
            buckets.append(self._union(self._by_table, flt.tables))
        if flt.kinds is not None:
            buckets.append(self._union(self._by_kind, flt.kinds))

        needs_verify = bool(flt.equals)
        if not buckets:
            return None, needs_verify
        buckets.sort(key=len)
        out = set(buckets[0])
        for extra in buckets[1:]:
            out &= extra
            if not out:
                break
        return out, needs_verify

    @staticmethod
    def _union(d: dict[Any, set[str]], keys: Iterable[Any]) -> set[str]:
        out: set[str] = set()
        for key in keys:
            bucket = d.get(key)
            if bucket:
                out |= bucket
        return out


# --------------------------------------------------------------------------
# Chunk serialization
# --------------------------------------------------------------------------

_TUPLE_TAG = "__tuple__"


def _encode(value: Any) -> Any:
    """JSON-encode a metadata value, preserving tuple-ness."""
    if isinstance(value, tuple):
        return {_TUPLE_TAG: [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise IndexError_(
        f"chunk metadata value of type {type(value).__name__!r} is not JSON "
        "serialisable; indexes persist meta verbatim and refuse to lose it"
    )


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if _TUPLE_TAG in value and len(value) == 1:
            return tuple(_decode(v) for v in value[_TUPLE_TAG])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    """A JSON-safe dict that `chunk_from_dict` turns back into `chunk`."""
    try:
        meta = {str(k): _encode(v) for k, v in chunk.meta.items()}
    except IndexError_ as exc:
        raise IndexError_(f"chunk {chunk.chunk_id!r}: {exc}") from exc
    return {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "kind": chunk.kind.value,
        "text": chunk.text,
        "row_refs": [[r.table, r.pk] for r in chunk.row_refs],
        "meta": meta,
    }


def chunk_from_dict(data: Mapping[str, Any]) -> Chunk:
    try:
        return Chunk(
            chunk_id=data["chunk_id"],
            source_id=data["source_id"],
            kind=ChunkKind(data["kind"]),
            text=data["text"],
            row_refs=tuple(
                RowRef(table=r[0], pk=r[1]) for r in data.get("row_refs", ())
            ),
            meta={k: _decode(v) for k, v in (data.get("meta") or {}).items()},
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise IndexError_(f"corrupt chunk record in index sidecar: {exc}") from exc
