"""Metadata pre-filtering: turn a `RetrievalConfig` into a `MetadataFilter`.

The filter is *pushed down into the index calls*, never applied to the result
list. `anyrag.index` narrows its candidate set with inverted posting lists
before scoring, so ``search(..., k=10, flt=...)`` returns ten hits whenever ten
matching chunks exist. Filtering afterwards would silently return fewer than `k`
results and every recall@k number in the ablation would be wrong for the
filtered cells -- wrong in the direction that flatters the unfiltered ones.

Two sources of restriction are merged here:

* ``config.chunk_kinds`` -- the {row-chunks, schema-cards, both} ablation axis.
  This is a *closed* set: the axis is only meaningful if "row" cells never see a
  schema card, so kinds are intersected, never widened.
* ``config.prefilter`` -- an arbitrary caller-supplied `MetadataFilter`
  (tables, sources, `equals`).

Both are ANDed. An empty intersection is representable: ``kinds=frozenset()``
matches nothing, which is the honest answer to "row chunks that are also schema
cards" and is what the indexes will return zero hits for.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from anyrag.core.config import MetadataFilter, RetrievalConfig
from anyrag.core.types import ChunkKind, Hit

__all__ = [
    "ALL_KINDS",
    "build_filter",
    "coerce_kinds",
    "describe_filter",
    "merge_filters",
    "unfiltered_violations",
]

#: Every chunk kind. `chunk_kinds` equal to this still gets pushed down: the
#: cost is one posting-list union and it keeps the filtered/unfiltered code
#: paths identical, so a bug in filtering cannot hide in the "both" cells.
ALL_KINDS: frozenset[ChunkKind] = frozenset(ChunkKind)


def coerce_kinds(kinds: Iterable[Any] | None) -> frozenset[ChunkKind] | None:
    """Normalise an iterable of kinds (or their string values) to `ChunkKind`.

    `ChunkKind` is a `str` Enum, so a raw ``"row"`` currently compares and
    hashes equal to `ChunkKind.ROW` and would work by accident against the
    index posting lists. Two reasons not to rely on that: a typo like
    ``"schema-card"`` would then be a filter that silently matches nothing
    rather than an error, and `describe_filter` reads ``kind.value`` for the
    trace. Coercing validates and normalises in one place.
    """
    if kinds is None:
        return None
    return frozenset(k if isinstance(k, ChunkKind) else ChunkKind(k) for k in kinds)


def merge_filters(
    base: MetadataFilter | None, kinds: Iterable[Any] | None
) -> MetadataFilter | None:
    """AND `kinds` into `base`, intersecting rather than replacing.

    Returns None when the result would impose no restriction at all.
    """
    kinds_set = coerce_kinds(kinds)
    if base is None:
        if kinds_set is None:
            return None
        return MetadataFilter(kinds=kinds_set)

    base_kinds = coerce_kinds(base.kinds)
    if kinds_set is None:
        merged = base_kinds
    elif base_kinds is None:
        merged = kinds_set
    else:
        # Intersection, not union: a caller that asked for schema cards and a
        # config cell that asked for rows agree on nothing, and pretending
        # otherwise would leak the other kind into the cell.
        merged = base_kinds & kinds_set

    out = MetadataFilter(
        kinds=merged,
        tables=base.tables,
        source_ids=base.source_ids,
        equals=dict(base.equals),
    )
    return None if out.is_empty else out


def build_filter(config: RetrievalConfig) -> MetadataFilter | None:
    """The filter to hand to every index `search()` for this config."""
    return merge_filters(config.prefilter, config.chunk_kinds)


def describe_filter(flt: MetadataFilter | None) -> dict[str, Any]:
    """A JSON-safe summary of `flt`, for the retrieval trace."""
    if flt is None:
        return {"applied": False}
    return {
        "applied": True,
        "kinds": sorted(k.value for k in flt.kinds) if flt.kinds is not None else None,
        "tables": sorted(flt.tables) if flt.tables is not None else None,
        "source_ids": (
            sorted(flt.source_ids) if flt.source_ids is not None else None
        ),
        "equals": {str(k): v for k, v in flt.equals.items()},
    }


def unfiltered_violations(
    hits: Sequence[Hit], flt: MetadataFilter | None
) -> list[str]:
    """Chunk ids in `hits` that `flt` should have excluded.

    Deliberately *not* used to filter anything: it exists so tests can assert
    that the index honoured the pushed-down filter. If this ever returns a
    non-empty list the fix is in the index, not a post-filter here -- dropping
    the offenders after the fact is exactly the silent-short-list bug this
    module is written to avoid.
    """
    if flt is None:
        return []
    return [h.chunk_id for h in hits if not flt.matches(h.chunk)]
