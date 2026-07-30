"""The pipeline: ablation switches, determinism, and re-ingestion."""

from __future__ import annotations

import pytest
from test_ingest_fakes import CUSTOMERS, FakeSource

from anyrag.core.config import AnyRagConfig
from anyrag.core.errors import ConfigError, SourceError
from anyrag.core.interfaces import Chunker, DataSource
from anyrag.core.types import ChunkKind, TableRef
from anyrag.ingest.pipeline import (
    IngestConfig,
    IngestStats,
    TableChunker,
    ingest,
)


def kinds(chunks) -> set[ChunkKind]:
    return {c.kind for c in chunks}


# -- protocol conformance ---------------------------------------------------


def test_fake_source_satisfies_the_datasource_protocol() -> None:
    assert isinstance(FakeSource(), DataSource)


def test_table_chunker_satisfies_the_chunker_protocol() -> None:
    assert isinstance(TableChunker(), Chunker)


# -- ablation switches ------------------------------------------------------


def test_both_strategies_by_default() -> None:
    chunks = list(ingest(FakeSource()))
    assert kinds(chunks) == {ChunkKind.ROW, ChunkKind.SCHEMA_CARD}


def test_row_chunks_only() -> None:
    chunks = list(ingest(FakeSource(), schema_cards=False))
    assert kinds(chunks) == {ChunkKind.ROW}
    assert len(chunks) == 6  # 3 customers + 3 order lines


def test_schema_cards_only() -> None:
    chunks = list(ingest(FakeSource(), row_chunks=False))
    assert kinds(chunks) == {ChunkKind.SCHEMA_CARD}
    assert len(chunks) == 2  # one per table
    assert all(c.row_refs == () for c in chunks)


def test_neither_strategy_is_a_configuration_error_not_a_silent_empty_index() -> None:
    with pytest.raises(ConfigError):
        list(ingest(FakeSource(), row_chunks=False, schema_cards=False))


def test_switches_override_a_supplied_config() -> None:
    cfg = IngestConfig(row_chunks=True, schema_cards=True)
    chunks = list(ingest(FakeSource(), config=cfg, row_chunks=False))
    assert kinds(chunks) == {ChunkKind.SCHEMA_CARD}
    # The caller's config object is untouched.
    assert cfg.row_chunks is True


def test_the_three_ablation_cells_are_all_reachable_and_distinct() -> None:
    both = list(ingest(FakeSource()))
    rows = list(ingest(FakeSource(), schema_cards=False))
    cards = list(ingest(FakeSource(), row_chunks=False))
    assert len(both) == len(rows) + len(cards)
    assert {c.chunk_id for c in both} == {c.chunk_id for c in rows} | {
        c.chunk_id for c in cards
    }


def test_config_can_be_built_from_the_app_config() -> None:
    app = AnyRagConfig(row_chunks=True, schema_cards=False)
    cfg = IngestConfig.from_app_config(app)
    assert cfg.schema_cards is False
    assert cfg.max_chunk_tokens == app.generation.max_chunk_tokens


# -- determinism and re-ingestion -------------------------------------------


def test_ingesting_twice_yields_an_identical_set_of_chunk_ids() -> None:
    first = list(ingest(FakeSource()))
    second = list(ingest(FakeSource()))
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]
    # This is the property the index relies on: upserting run 2 over run 1
    # cannot change the index size.
    assert len({c.chunk_id for c in first} | {c.chunk_id for c in second}) == len(first)


def test_chunk_ids_are_unique_within_a_run() -> None:
    chunks = list(ingest(FakeSource()))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_editing_a_row_preserves_its_id_and_changes_its_content_hash() -> None:
    source = FakeSource()
    before = {c.chunk_id: c for c in ingest(source, schema_cards=False)}

    source.edit("customers", "customer_id", 42, region="APAC")
    after = {c.chunk_id: c for c in ingest(source, schema_cards=False)}

    assert set(before) == set(after)  # no new ids, so no duplicates on upsert
    changed = [k for k in before if before[k].content_hash != after[k].content_hash]
    assert len(changed) == 1
    assert "region is APAC" in after[changed[0]].text


def test_two_sources_over_the_same_data_do_not_collide() -> None:
    a = {c.chunk_id for c in ingest(FakeSource("sqlite:///a.db"))}
    b = {c.chunk_id for c in ingest(FakeSource("postgres://b"))}
    assert a.isdisjoint(b)


# -- table walking ----------------------------------------------------------


def test_schema_card_precedes_its_table_rows() -> None:
    chunks = list(ingest(FakeSource(), tables=["customers"]))
    assert chunks[0].kind is ChunkKind.SCHEMA_CARD
    assert all(c.kind is ChunkKind.ROW for c in chunks[1:])
    assert {c.meta["table"] for c in chunks} == {"customers"}


def test_tables_can_be_selected_by_name_or_ref() -> None:
    by_name = list(ingest(FakeSource(), tables=["orders"]))
    by_ref = list(ingest(FakeSource(), tables=[TableRef("orders")]))
    assert [c.chunk_id for c in by_name] == [c.chunk_id for c in by_ref]


def test_unknown_table_is_an_error_not_an_empty_result() -> None:
    with pytest.raises(SourceError):
        list(ingest(FakeSource(), tables=["nope"]))


def test_row_limit_is_respected() -> None:
    cfg = IngestConfig(max_rows_per_table=1)
    chunks = list(ingest(FakeSource(), config=cfg, schema_cards=False))
    assert len(chunks) == 2  # one row from each of the two tables


def test_batching_does_not_change_the_output() -> None:
    small = list(ingest(FakeSource(), config=IngestConfig(batch_size=1)))
    large = list(ingest(FakeSource(), config=IngestConfig(batch_size=1000)))
    assert [c.chunk_id for c in small] == [c.chunk_id for c in large]


def test_ingest_is_lazy() -> None:
    """A generator, not a list: a ten-million-row table must stream."""
    stream = ingest(FakeSource())
    assert next(stream).kind is ChunkKind.SCHEMA_CARD


# -- degraded sources -------------------------------------------------------


def test_source_without_profiles_still_ingests() -> None:
    stats = IngestStats()
    chunks = list(ingest(FakeSource(profiles=False), stats=stats))
    assert kinds(chunks) == {ChunkKind.ROW, ChunkKind.SCHEMA_CARD}
    assert set(stats.profile_failures) == {"customers", "orders"}
    assert "SchemaError" in stats.profile_failures["customers"]


def test_stats_count_what_was_built() -> None:
    stats = IngestStats()
    list(ingest(FakeSource(), stats=stats))
    assert stats.tables == 2
    assert stats.rows == 6
    assert stats.schema_cards == 2
    assert stats.row_chunks == 6
    assert stats.chunks == 8


def test_bad_config_is_rejected_up_front() -> None:
    with pytest.raises(ConfigError):
        IngestConfig(max_chunk_tokens=64, overlap_tokens=64)
    with pytest.raises(ConfigError):
        IngestConfig(batch_size=0)


def test_chunker_can_be_pointed_at_one_table() -> None:
    source = FakeSource()
    chunks = list(TableChunker().chunk_table(source, CUSTOMERS))
    assert {c.meta["table"] for c in chunks} == {"customers"}
