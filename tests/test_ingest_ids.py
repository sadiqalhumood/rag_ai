"""Chunk identity: the property the whole incremental-update story rests on.

The four requirements tested here are, in order of consequence:

1. same logical row twice -> same id (re-ingestion updates, never duplicates);
2. edited row -> same id, different content_hash (change is detectable without
   changing identity);
3. distinct rows/tables/sources/parts -> distinct ids (no silent overwrite);
4. composite keys join stably.
"""

from __future__ import annotations

import hashlib

import pytest

from anyrag.core.types import ChunkKind, TableRef
from anyrag.ingest.ids import (
    ID_LENGTH,
    NULL_PK,
    PK_SEP,
    UNIT_SEP,
    chunk_id,
    content_hash,
    id_payload,
    make_pk,
    normalize_pk_value,
    pk_for_row,
    row_chunk_id,
    schema_card_id,
)


def _id(**kw):
    base = dict(
        source_id="sqlite:///demo.db", kind=ChunkKind.ROW, table="customers",
        pk="41", part_index=0,
    )
    base.update(kw)
    return chunk_id(**base)


# -- format -----------------------------------------------------------------


def test_payload_is_the_documented_five_fields() -> None:
    payload = id_payload(
        source_id="s", kind=ChunkKind.ROW, table="customers", pk="41", part_index=2
    )
    assert payload == UNIT_SEP.join(("s", "row", "customers", "41", "2"))
    assert (
        chunk_id(source_id="s", kind=ChunkKind.ROW, table="customers", pk="41", part_index=2)
        == hashlib.sha256(payload.encode("utf-8")).hexdigest()[:ID_LENGTH]
    )


def test_kind_uses_enum_value_not_repr() -> None:
    """str(ChunkKind.ROW) is 'ChunkKind.ROW' on some Pythons; ids must not shift."""
    assert "ChunkKind" not in id_payload(
        source_id="s", kind=ChunkKind.SCHEMA_CARD, table="t", pk="", part_index=0
    )
    assert _id(kind=ChunkKind.ROW) == _id(kind="row")


def test_id_is_32_hex_chars() -> None:
    value = _id()
    assert len(value) == ID_LENGTH
    assert all(c in "0123456789abcdef" for c in value)


# -- 1. stability across re-ingestion ---------------------------------------


def test_same_logical_row_twice_gives_identical_id() -> None:
    first = row_chunk_id(source_id="s", table=TableRef("customers"), pk="41")
    second = row_chunk_id(source_id="s", table=TableRef("customers"), pk="41")
    assert first == second


def test_pk_type_does_not_affect_identity() -> None:
    """SQLite hands back 7, a CSV hands back '7'. Same row, same id."""
    assert make_pk(7) == make_pk("7") == "7"
    assert make_pk(7.0) == "7"
    assert _id(pk=make_pk(7)) == _id(pk=make_pk("7"))


# -- 2. content is not identity ---------------------------------------------


def test_content_is_absent_from_the_id_but_present_in_the_hash() -> None:
    before = "customers record 41: region is EMEA."
    after = "customers record 41: region is APAC."
    assert _id() == _id()  # id does not take content at all
    assert content_hash(before) != content_hash(after)
    assert content_hash(before) == content_hash(before)


def test_content_hash_is_full_length_sha256() -> None:
    digest = content_hash("anything")
    assert len(digest) == 64
    assert digest == hashlib.sha256(b"anything").hexdigest()


# -- 3. distinctness --------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_id", "sqlite:///other.db"),
        ("kind", ChunkKind.SCHEMA_CARD),
        ("table", "orders"),
        ("pk", "42"),
        ("part_index", 1),
    ],
)
def test_changing_any_id_field_changes_the_id(field: str, value) -> None:
    assert _id(**{field: value}) != _id()


def test_no_collisions_across_a_grid_of_rows() -> None:
    ids = {
        _id(source_id=s, table=t, pk=str(pk), part_index=p)
        for s in ("a", "b")
        for t in ("customers", "orders")
        for pk in range(50)
        for p in range(3)
    }
    assert len(ids) == 2 * 2 * 50 * 3


def test_separator_prevents_field_boundary_collisions() -> None:
    """('ab', 'c') and ('a', 'bc') must not hash to the same payload."""
    assert _id(table="ab", pk="c") != _id(table="a", pk="bc")


def test_schema_card_and_row_ids_differ_for_the_same_table() -> None:
    table = TableRef("customers")
    assert schema_card_id(source_id="s", table=table) != row_chunk_id(
        source_id="s", table=table, pk=""
    )


def test_schema_qualified_tables_are_distinct() -> None:
    assert _id(table=TableRef("customers", schema="sales")) != _id(
        table=TableRef("customers", schema="ops")
    )


# -- 4. composite keys ------------------------------------------------------


def test_composite_pk_joins_stably_and_order_matters() -> None:
    assert make_pk((900, 1)) == f"900{PK_SEP}1"
    assert make_pk([900, 1]) == make_pk(("900", "1"))
    assert make_pk((900, 1)) != make_pk((1, 900))


def test_pk_for_row_follows_the_column_order_given() -> None:
    row = {"order_id": 900, "line_no": 2, "sku": "B-2"}
    assert pk_for_row(row, ("order_id", "line_no")) == "900|2"
    assert pk_for_row(row, ("line_no", "order_id")) == "2|900"


def test_single_element_composite_has_no_separator() -> None:
    assert make_pk(["41"]) == "41"


def test_mapping_pk_uses_value_order() -> None:
    assert make_pk({"order_id": 900, "line_no": 1}) == "900|1"


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, "1"),
        (False, "0"),
        (None, NULL_PK),
        (b"\xff\x00", "ff00"),
        (3.5, "3.5"),
        ("سارة", "سارة"),
    ],
)
def test_pk_value_normalisation(value, expected: str) -> None:
    assert normalize_pk_value(value) == expected


def test_string_pk_is_not_treated_as_a_sequence_of_characters() -> None:
    assert make_pk("A-1") == "A-1"
