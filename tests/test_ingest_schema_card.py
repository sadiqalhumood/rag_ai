"""Schema cards: the chunks that make schema questions answerable."""

from __future__ import annotations

from test_ingest_fakes import (
    CUSTOMER_PROFILE,
    CUSTOMER_SCHEMA,
    ORDER_PROFILE,
    ORDER_SCHEMA,
)

from anyrag.core.tokenizer import get_tokenizer
from anyrag.core.types import ChunkKind, ColumnSchema, TableRef, TableSchema
from anyrag.ingest.ids import content_hash, schema_card_id
from anyrag.ingest.schema_card import (
    SchemaCardConfig,
    build_schema_card,
    schema_card_text,
)


def card_text(**kw) -> str:
    return schema_card_text(CUSTOMER_SCHEMA, CUSTOMER_PROFILE, **kw)


def test_card_names_every_column_and_its_declared_type() -> None:
    text = card_text()
    for column in CUSTOMER_SCHEMA.columns:
        assert column.name in text
        assert column.type_name in text


def test_card_answers_what_columns_does_this_table_have() -> None:
    text = card_text()
    assert "Columns of customers: " in text
    listing = text.split("Columns of customers: ")[1].split("\n")[0]
    for name in CUSTOMER_SCHEMA.column_names:
        assert name in listing


def test_card_states_roles_cardinality_and_null_fraction() -> None:
    text = card_text()
    assert "role categorical" in text
    assert "role free_text" in text
    assert "2 distinct values" in text
    assert "missing" in text


def test_card_includes_samples_and_category_vocabulary() -> None:
    text = card_text()
    assert "Values: APAC, EMEA." in text
    assert "Examples:" in text
    assert "Ahmed Al-Sayed" in text


def test_card_preserves_arabic_samples() -> None:
    assert "سارة عبد الله" in card_text()


def test_card_states_primary_key_and_foreign_keys() -> None:
    customers = card_text()
    assert "Primary key: customer_id." in customers
    assert "Foreign keys: none." in customers

    orders = schema_card_text(ORDER_SCHEMA, ORDER_PROFILE)
    assert "Primary key: order_id, line_no." in orders
    assert "orders.customer_id references customers.customer_id" in orders


def test_card_reports_row_count_when_profiled() -> None:
    assert "3 rows" in card_text()


def test_card_without_a_profile_still_lists_the_schema() -> None:
    text = schema_card_text(CUSTOMER_SCHEMA, None)
    assert "Columns of customers:" in text
    assert "signup_date: type DATE" in text
    assert "rows" not in text.split("\n")[0]


def test_card_chunk_has_no_row_refs() -> None:
    """A card describes a table, so crediting it with row provenance would
    corrupt the eval's gold-row matching."""
    chunks = build_schema_card(
        source_id="s", schema=CUSTOMER_SCHEMA, profile=CUSTOMER_PROFILE
    )
    assert len(chunks) == 1
    assert chunks[0].row_refs == ()
    assert chunks[0].kind is ChunkKind.SCHEMA_CARD


def test_card_meta_carries_types_roles_and_hash() -> None:
    chunk = build_schema_card(
        source_id="s", schema=CUSTOMER_SCHEMA, profile=CUSTOMER_PROFILE
    )[0]
    assert chunk.meta["table"] == "customers"
    assert chunk.meta["column_types"]["signup_date"] == "DATE"
    assert chunk.meta["column_roles"]["region"] == "categorical"
    assert chunk.meta["primary_key"] == ("customer_id",)
    assert chunk.meta["row_count"] == 3
    assert chunk.meta["part_index"] == 0
    assert chunk.meta["n_parts"] == 1
    assert chunk.meta["content_hash"] == content_hash(chunk.text)


def test_card_meta_lists_foreign_keys() -> None:
    chunk = build_schema_card(
        source_id="s", schema=ORDER_SCHEMA, profile=ORDER_PROFILE
    )[0]
    assert chunk.meta["foreign_keys"] == ("customer_id->customers.customer_id",)


def test_card_id_is_stable_and_content_independent() -> None:
    expected = schema_card_id(source_id="s", table=TableRef("customers"))
    first = build_schema_card(
        source_id="s", schema=CUSTOMER_SCHEMA, profile=CUSTOMER_PROFILE
    )[0]
    # Same table, different statistics (a day's worth of new rows).
    second = build_schema_card(source_id="s", schema=CUSTOMER_SCHEMA, profile=None)[0]
    assert first.chunk_id == second.chunk_id == expected
    assert first.meta["content_hash"] != second.meta["content_hash"]


def test_very_wide_table_splits_into_overlapping_parts() -> None:
    wide = TableSchema(
        table=TableRef("wide"),
        columns=tuple(
            ColumnSchema(f"column_number_{i}", "TEXT") for i in range(400)
        ),
    )
    chunks = build_schema_card(
        source_id="s",
        schema=wide,
        config=SchemaCardConfig(max_chunk_tokens=256, overlap_tokens=32),
    )
    tok = get_tokenizer()
    assert len(chunks) > 1
    assert [c.meta["part_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.meta["n_parts"] == len(chunks) for c in chunks)
    assert all(tok.count(c.text) <= 256 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
