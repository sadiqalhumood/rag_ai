"""PostingLists and chunk serialization.

These are the two shared pieces both indexes lean on: the postings are what make
`delete_by_source` and pre-filtering sub-linear, and the serializer is what makes
the "persist/load round-trips exactly" claim true rather than approximately true.
"""

from __future__ import annotations

import pytest

from anyrag.core.config import MetadataFilter
from anyrag.core.errors import IndexError_
from anyrag.core.types import Chunk, ChunkKind, RowRef
from anyrag.index import PostingLists, chunk_from_dict, chunk_to_dict
from test_index_fixtures import make_chunk


def filled() -> PostingLists:
    p = PostingLists()
    for chunk in (
        make_chunk("a1", source_id="sqlite:A", table="orders"),
        make_chunk("a2", source_id="sqlite:A", table="customers"),
        make_chunk("b1", source_id="sqlite:B", table="orders"),
        make_chunk("b2", source_id="sqlite:B", table="orders", kind=ChunkKind.SCHEMA_CARD),
    ):
        p.add(chunk)
    return p


# -- postings --------------------------------------------------------------


def test_lookup_by_source_table_and_kind() -> None:
    p = filled()
    assert len(p) == 4
    assert p.ids_for_source("sqlite:A") == {"a1", "a2"}
    assert p.ids_for_source("sqlite:missing") == set()
    assert p.ids_for_table("orders") == {"a1", "b1", "b2"}
    assert p.ids_for_kind(ChunkKind.SCHEMA_CARD) == {"b2"}
    assert p.sources == ("sqlite:A", "sqlite:B")


def test_adding_the_same_id_twice_does_not_double_count() -> None:
    p = PostingLists()
    chunk = make_chunk("c")
    p.add(chunk)
    p.add(chunk)
    assert len(p) == 1
    assert p.ids_for_source("sqlite:test") == {"c"}


def test_readding_a_moved_chunk_leaves_no_stale_posting() -> None:
    """An upsert may change a chunk's table, source or kind."""
    p = PostingLists()
    p.add(make_chunk("c", source_id="sqlite:A", table="orders"))
    p.add(make_chunk("c", source_id="sqlite:B", table="invoices",
                     kind=ChunkKind.SCHEMA_CARD))
    assert len(p) == 1
    assert p.ids_for_source("sqlite:A") == set()
    assert p.ids_for_table("orders") == set()
    assert p.ids_for_kind(ChunkKind.ROW) == set()
    assert p.ids_for_source("sqlite:B") == {"c"}
    assert p.ids_for_table("invoices") == {"c"}


def test_discard_is_idempotent_and_accepts_a_bare_id() -> None:
    p = filled()
    p.discard(make_chunk("a1", source_id="sqlite:A", table="orders"))
    assert len(p) == 3
    p.discard("a1")
    assert len(p) == 3
    p.discard("never-seen")
    assert len(p) == 3
    p.discard("b2")
    assert p.ids_for_kind(ChunkKind.SCHEMA_CARD) == set()
    p.clear()
    assert len(p) == 0 and p.sources == ()


def test_empty_or_absent_filter_means_no_restriction() -> None:
    p = filled()
    assert p.candidates(None) == (None, False)
    assert p.candidates(MetadataFilter()) == (None, False)


def test_candidates_intersects_the_constrained_dimensions() -> None:
    p = filled()
    ids, verify = p.candidates(
        MetadataFilter(source_ids=frozenset({"sqlite:B"}), tables=frozenset({"orders"}))
    )
    assert ids == {"b1", "b2"} and verify is False
    ids, verify = p.candidates(
        MetadataFilter(
            source_ids=frozenset({"sqlite:B"}),
            kinds=frozenset({ChunkKind.ROW}),
        )
    )
    assert ids == {"b1"} and verify is False
    ids, _ = p.candidates(MetadataFilter(tables=frozenset({"nope"})))
    assert ids == set()


def test_equals_conditions_are_reported_as_needing_verification() -> None:
    """`equals` is open-ended, so postings can only narrow, never decide."""
    p = filled()
    ids, verify = p.candidates(MetadataFilter(equals={"lang": "ar"}))
    assert ids is None and verify is True
    ids, verify = p.candidates(
        MetadataFilter(tables=frozenset({"orders"}), equals={"lang": "ar"})
    )
    assert ids == {"a1", "b1", "b2"} and verify is True


def test_chunks_without_a_table_are_indexed_under_none() -> None:
    p = PostingLists()
    p.add(make_chunk("no-table", table=None))
    assert p.ids_for_table(None) == {"no-table"}
    ids, _ = p.candidates(MetadataFilter(tables=frozenset({"orders"})))
    assert ids == set()


# -- serialization ---------------------------------------------------------


def test_chunk_round_trips_including_row_refs_and_nested_meta() -> None:
    chunk = Chunk(
        chunk_id="c1",
        source_id="files:exports",
        kind=ChunkKind.SQL_RESULT,
        text="إجمالي المبيعات 1200 ريال",
        row_refs=(RowRef("orders", "7"), RowRef("orders", 8)),
        meta={
            "table": "orders",
            "columns": ("a", "b"),
            "nested": {"tags": ("x",), "list": [1, 2]},
            "n": 3,
            "f": 1.5,
            "t": True,
            "none": None,
        },
    )
    back = chunk_from_dict(chunk_to_dict(chunk))
    assert back == chunk
    assert back.row_refs == (RowRef("orders", "7"), RowRef("orders", "8"))
    assert isinstance(back.meta["columns"], tuple)
    assert isinstance(back.meta["nested"]["tags"], tuple)
    assert isinstance(back.meta["nested"]["list"], list)


def test_serialized_form_is_plain_json_types() -> None:
    import json

    data = chunk_to_dict(make_chunk("c1", "hello"))
    assert json.loads(json.dumps(data)) == data
    assert data["kind"] == "row"
    assert data["row_refs"] == [["orders", "c1"]]


def test_unserialisable_metadata_is_refused_loudly() -> None:
    """Silently dropping meta would break filters after a reload."""
    chunk = Chunk(
        chunk_id="c1",
        source_id="s",
        kind=ChunkKind.ROW,
        text="x",
        meta={"when": object()},
    )
    with pytest.raises(IndexError_, match="c1"):
        chunk_to_dict(chunk)


def test_corrupt_record_raises_a_clear_error() -> None:
    with pytest.raises(IndexError_, match="corrupt chunk record"):
        chunk_from_dict({"chunk_id": "c1"})
    with pytest.raises(IndexError_, match="corrupt chunk record"):
        chunk_from_dict(
            {"chunk_id": "c1", "source_id": "s", "kind": "not-a-kind", "text": ""}
        )
