"""Row verbalisation: what the embedder actually sees."""

from __future__ import annotations

from datetime import date

import pytest
from test_ingest_fakes import (
    CUSTOMER_PROFILE,
    CUSTOMER_ROWS,
    CUSTOMER_SCHEMA,
    ORDER_PROFILE,
    ORDER_ROWS,
    ORDER_SCHEMA,
)

from anyrag.core.tokenizer import get_tokenizer
from anyrag.core.types import (
    ChunkKind,
    ColumnProfile,
    ColumnRole,
    ColumnSchema,
    RowRef,
    TableProfile,
    TableRef,
    TableSchema,
)
from anyrag.ingest.ids import content_hash
from anyrag.ingest.row_serializer import (
    NULL_PHRASE,
    RowSerializer,
    RowSerializerConfig,
    format_value,
)

ARABIC_NAME = "سارة عبد الله"
ARABIC_NOTES = "ملاحظات باللغة العربية عن العميل."


def customer_serializer(**cfg) -> RowSerializer:
    return RowSerializer(
        source_id="fake://demo",
        schema=CUSTOMER_SCHEMA,
        profile=CUSTOMER_PROFILE,
        config=RowSerializerConfig(**cfg) if cfg else None,
    )


# -- prose ------------------------------------------------------------------


def test_row_reads_as_a_sentence_with_column_names() -> None:
    text = customer_serializer().row_text(CUSTOMER_ROWS[0]).text
    assert text.startswith("customers record 41:")
    assert "name is Ahmed Al-Sayed" in text
    assert "region is EMEA" in text
    assert "signup_date is 2023-04-02" in text
    assert text.endswith(".")


def test_nulls_are_stated_not_omitted() -> None:
    text = customer_serializer().row_text(CUSTOMER_ROWS[1]).text
    assert f"region is {NULL_PHRASE}" in text
    assert "None" not in text


def test_null_free_text_column_also_gets_a_phrase() -> None:
    text = customer_serializer().row_text(CUSTOMER_ROWS[2]).text
    assert f"notes is {NULL_PHRASE}" in text


def test_arabic_is_preserved_byte_for_byte() -> None:
    rendered = customer_serializer().row_text(CUSTOMER_ROWS[1])
    assert ARABIC_NAME in rendered.text
    assert ARABIC_NOTES in rendered.text
    # Round-trips through UTF-8 unchanged: no transliteration, no escaping.
    assert rendered.text.encode("utf-8").decode("utf-8") == rendered.text
    assert "\\u" not in rendered.text
    assert rendered.lang in ("ar", "mixed")


def test_booleans_read_as_words() -> None:
    text = customer_serializer().row_text(CUSTOMER_ROWS[0]).text
    assert "active is yes" in text
    assert "active is True" not in text


SHIPMENTS = TableSchema(
    table=TableRef("shipments"),
    columns=(
        ColumnSchema("shipment_id", "INTEGER", nullable=False, is_primary_key=True),
        ColumnSchema("tracking_uuid", "TEXT"),
        ColumnSchema("carrier", "TEXT"),
    ),
    primary_key=("shipment_id",),
)
SHIPMENT_PROFILE = TableProfile(
    table=TableRef("shipments"),
    row_count=1,
    columns={
        "shipment_id": ColumnProfile("shipment_id", ColumnRole.ID, 1, 0, 1),
        "tracking_uuid": ColumnProfile("tracking_uuid", ColumnRole.ID, 1, 0, 1),
        "carrier": ColumnProfile("carrier", ColumnRole.CATEGORICAL, 1, 0, 1),
    },
)
SHIPMENT_ROW = {
    "shipment_id": 5,
    "tracking_uuid": "4f0a-11ee-be56-0242ac120002",
    "carrier": "DHL",
}


def test_surrogate_id_columns_are_kept_out_of_the_prose_but_not_lost() -> None:
    rendered = RowSerializer(
        source_id="s", schema=SHIPMENTS, profile=SHIPMENT_PROFILE
    ).row_text(SHIPMENT_ROW)
    assert "tracking_uuid is" not in rendered.text
    assert "carrier is DHL" in rendered.text
    # Dropped from the sentence, retained in meta for joins and filters.
    assert rendered.id_columns["tracking_uuid"] == "4f0a-11ee-be56-0242ac120002"
    assert "tracking_uuid" not in rendered.columns
    # The pk is in the header rather than repeated as a clause.
    assert rendered.text.startswith("shipments record 5:")
    assert "shipment_id is" not in rendered.text


def test_foreign_keys_stay_in_the_prose_even_though_they_are_ids() -> None:
    """Dropping the FK would make every join question unanswerable from rows."""
    rendered = RowSerializer(
        source_id="s", schema=ORDER_SCHEMA, profile=ORDER_PROFILE
    ).row_text(ORDER_ROWS[0])
    assert "customer_id is 41" in rendered.text
    assert "customer_id" in rendered.columns


def test_id_column_skipping_is_switchable() -> None:
    serializer = RowSerializer(
        source_id="s",
        schema=SHIPMENTS,
        profile=SHIPMENT_PROFILE,
        config=RowSerializerConfig(skip_id_columns=False),
    )
    text = serializer.row_text(SHIPMENT_ROW).text
    assert "tracking_uuid is 4f0a-11ee-be56-0242ac120002" in text
    assert "shipment_id is 5" in text


def test_composite_pk_appears_in_the_header() -> None:
    rendered = RowSerializer(
        source_id="s", schema=ORDER_SCHEMA, profile=ORDER_PROFILE
    ).row_text(ORDER_ROWS[1])
    assert rendered.pk == "900|2"
    assert rendered.text.startswith("orders record 900|2:")


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, NULL_PHRASE),
        (True, "yes"),
        (False, "no"),
        ("", "empty"),
        ("   ", "empty"),
        (date(2023, 4, 2), "2023-04-02"),
        (12.0, "12"),
        (12.5, "12.5"),
        (b"\x00\x01", "<2 bytes of binary data>"),
    ],
)
def test_value_formatting(value, expected: str) -> None:
    assert format_value(value) == expected


# -- chunks and meta --------------------------------------------------------


def test_chunk_carries_row_ref_and_required_meta() -> None:
    chunk = customer_serializer().chunks(CUSTOMER_ROWS[0])[0]
    assert chunk.kind is ChunkKind.ROW
    assert chunk.row_refs == (RowRef("customers", "41"),)
    for key in ("table", "columns", "part_index", "n_parts", "content_hash"):
        assert key in chunk.meta, key
    assert chunk.meta["table"] == "customers"
    assert chunk.meta["content_hash"] == content_hash(chunk.text)
    assert chunk.meta["truncated_fields"] == ()


def test_date_min_max_recorded_for_filtering() -> None:
    chunk = customer_serializer().chunks(CUSTOMER_ROWS[0])[0]
    assert chunk.meta["date_min"] == "2023-04-02"
    assert chunk.meta["date_max"] == "2023-04-02"


def test_editing_a_row_keeps_the_id_and_changes_the_content_hash() -> None:
    serializer = customer_serializer()
    before = serializer.chunks(CUSTOMER_ROWS[0])[0]
    edited = dict(CUSTOMER_ROWS[0], region="APAC")
    after = serializer.chunks(edited)[0]
    assert after.chunk_id == before.chunk_id
    assert after.meta["content_hash"] != before.meta["content_hash"]
    assert "region is APAC" in after.text


def test_oversized_row_splits_into_overlapping_parts_sharing_one_row_ref() -> None:
    row = dict(CUSTOMER_ROWS[0], notes="incident report " * 12_500)
    serializer = customer_serializer(
        max_chunk_tokens=128, overlap_tokens=32, max_field_tokens=4096
    )
    chunks = serializer.chunks(row)
    assert len(chunks) > 1
    assert {c.meta["n_parts"] for c in chunks} == {len(chunks)}
    assert [c.meta["part_index"] for c in chunks] == list(range(len(chunks)))
    assert {c.row_refs for c in chunks} == {(RowRef("customers", "41"),)}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    tok = get_tokenizer()
    assert all(tok.count(c.text) <= 128 for c in chunks)


def test_adversarial_field_is_truncated_and_recorded() -> None:
    row = dict(CUSTOMER_ROWS[0], notes="x" * 200_000)
    chunks = customer_serializer(max_field_tokens=32).chunks(row)
    assert len(chunks) == 1  # truncation kept the row inside one chunk
    assert chunks[0].meta["truncated_fields"] == ("notes",)
    assert "[truncated]" in chunks[0].text


def test_serializer_works_without_a_profile() -> None:
    """Roles fall back to declared types; nothing crashes, dates still found."""
    serializer = RowSerializer(source_id="s", schema=CUSTOMER_SCHEMA, profile=None)
    chunk = serializer.chunks(CUSTOMER_ROWS[0])[0]
    assert "name is Ahmed Al-Sayed" in chunk.text
    assert chunk.meta["date_min"] == "2023-04-02"


def test_table_without_declared_key_falls_back_to_ordinal() -> None:
    schema = TableSchema(
        table=TableRef("events"),
        columns=(ColumnSchema("what", "TEXT"), ColumnSchema("when", "DATE")),
    )
    serializer = RowSerializer(source_id="s", schema=schema)
    chunks = serializer.chunks({"what": "launch", "when": "2024-02-01"}, ordinal=7)
    assert chunks[0].row_refs == (RowRef("events", "7"),)


def test_extra_columns_not_in_the_schema_are_still_verbalised() -> None:
    row = dict(CUSTOMER_ROWS[0], surprise="unexpected column")
    text = customer_serializer().row_text(row).text
    assert "surprise is unexpected column" in text
