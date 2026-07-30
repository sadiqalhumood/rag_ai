"""Walk a source, emit chunks.

`ingest()` is a generator on purpose: a table with ten million rows must be
embeddable without ever holding ten million chunks in memory, and the index's
`upsert` takes batches anyway.

The two chunking strategies are independent switches, not a mode enum and not a
default that some caller can forget to turn off. The eval ablates over
{row-chunks, schema-cards, both} to show what each contributes, so hardwiring
either one on would silently delete a row of the results table.

This module knows nothing about any concrete adapter. It talks to the
`DataSource` Protocol, so a source that does not exist yet -- or a fake defined
inside a test file -- works identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Sequence

from ..core.config import AnyRagConfig
from ..core.errors import ConfigError, SourceError
from ..core.interfaces import DataSource
from ..core.tokenizer import Tokenizer, get_tokenizer
from ..core.types import Chunk, TableProfile, TableRef, TableSchema
from .row_serializer import RowSerializer, RowSerializerConfig
from .schema_card import SchemaCardConfig, build_schema_card


@dataclass(frozen=True)
class IngestConfig:
    """Everything the chunkers need, in one object the eval can vary."""

    #: Emit one chunk per row (possibly several parts for a huge row).
    row_chunks: bool = True
    #: Emit one card per table describing its schema and statistics.
    schema_cards: bool = True
    max_chunk_tokens: int = 512
    max_field_tokens: int = 96
    overlap_tokens: int = 48
    skip_id_columns: bool = True
    max_columns: int | None = None
    batch_size: int = 1000
    #: Cap rows per table (smoke runs and unit tests); None = every row.
    max_rows_per_table: int | None = None

    def __post_init__(self) -> None:
        if not self.row_chunks and not self.schema_cards:
            raise ConfigError(
                "at least one of row_chunks/schema_cards must be enabled; "
                "ingesting with both off would build an empty index silently"
            )
        if self.overlap_tokens >= self.max_chunk_tokens:
            raise ConfigError("overlap_tokens must be smaller than max_chunk_tokens")
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be positive")

    def with_(self, **kw: Any) -> "IngestConfig":
        return replace(self, **kw)

    @property
    def serializer_config(self) -> RowSerializerConfig:
        return RowSerializerConfig(
            max_chunk_tokens=self.max_chunk_tokens,
            max_field_tokens=self.max_field_tokens,
            overlap_tokens=self.overlap_tokens,
            skip_id_columns=self.skip_id_columns,
            max_columns=self.max_columns,
        )

    @property
    def card_config(self) -> SchemaCardConfig:
        return SchemaCardConfig(
            max_chunk_tokens=self.max_chunk_tokens,
            overlap_tokens=self.overlap_tokens,
        )

    @classmethod
    def from_app_config(cls, config: AnyRagConfig, **overrides: Any) -> "IngestConfig":
        base = cls(
            row_chunks=config.row_chunks,
            schema_cards=config.schema_cards,
            max_chunk_tokens=config.generation.max_chunk_tokens,
        )
        return base.with_(**overrides) if overrides else base


@dataclass
class IngestStats:
    """Counters for the run, so a caller can report what was actually built."""

    tables: int = 0
    rows: int = 0
    row_chunks: int = 0
    schema_cards: int = 0
    truncated_fields: int = 0
    split_rows: int = 0
    #: Tables whose profile could not be computed, with the reason.
    profile_failures: dict[str, str] = field(default_factory=dict)

    @property
    def chunks(self) -> int:
        return self.row_chunks + self.schema_cards


def _safe_profile(
    source: DataSource, table: TableRef, stats: IngestStats | None
) -> TableProfile | None:
    """Profiles are enrichment, not correctness.

    A source that cannot profile a table (permissions, an exotic column type, an
    adapter still being written) must still be ingestible: roles fall back to
    declared types and the schema card simply carries fewer statistics.
    """
    try:
        return source.profile(table)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        if stats is not None:
            stats.profile_failures[table.qualified] = f"{type(exc).__name__}: {exc}"
        return None


class TableChunker:
    """Implements the `Chunker` protocol: a source and a table in, chunks out."""

    def __init__(
        self,
        config: IngestConfig | None = None,
        *,
        tokenizer: Tokenizer | None = None,
        stats: IngestStats | None = None,
    ) -> None:
        self.config = config or IngestConfig()
        self.tokenizer = tokenizer or get_tokenizer()
        self.stats = stats if stats is not None else IngestStats()

    def chunk_table(self, source: DataSource, table: TableRef) -> Iterator[Chunk]:
        cfg = self.config
        schema: TableSchema = source.schema(table)
        profile = _safe_profile(source, table, self.stats)
        self.stats.tables += 1

        if cfg.schema_cards:
            for chunk in build_schema_card(
                source_id=source.source_id,
                schema=schema,
                profile=profile,
                config=cfg.card_config,
                tokenizer=self.tokenizer,
            ):
                self.stats.schema_cards += 1
                yield chunk

        if not cfg.row_chunks:
            return

        serializer = RowSerializer(
            source_id=source.source_id,
            schema=schema,
            profile=profile,
            config=cfg.serializer_config,
            tokenizer=self.tokenizer,
        )
        ordinal = 0
        limit = cfg.max_rows_per_table
        for batch in source.iter_rows(table, cfg.batch_size):
            for row in batch:
                if limit is not None and ordinal >= limit:
                    return
                chunks = serializer.chunks(row, ordinal=ordinal)
                ordinal += 1
                self.stats.rows += 1
                if len(chunks) > 1:
                    self.stats.split_rows += 1
                for chunk in chunks:
                    self.stats.row_chunks += 1
                    self.stats.truncated_fields += len(
                        chunk.meta.get("truncated_fields", ())
                    )
                    yield chunk


def ingest(
    source: DataSource,
    *,
    row_chunks: bool = True,
    schema_cards: bool = True,
    tables: Sequence[TableRef | str] | None = None,
    config: IngestConfig | None = None,
    tokenizer: Tokenizer | None = None,
    stats: IngestStats | None = None,
) -> Iterator[Chunk]:
    """Chunk every table of `source`.

    `row_chunks` / `schema_cards` are the ablation switches. When an explicit
    `config` is passed they act as overrides on it, so
    `ingest(src, config=cfg, schema_cards=False)` does what it reads like.
    """
    cfg = (config or IngestConfig()).with_(
        row_chunks=row_chunks, schema_cards=schema_cards
    )
    chunker = TableChunker(cfg, tokenizer=tokenizer, stats=stats)
    for table in _resolve_tables(source, tables):
        yield from chunker.chunk_table(source, table)


def ingest_tables(
    source: DataSource,
    tables: Sequence[TableRef | str],
    **kwargs: Any,
) -> Iterator[Chunk]:
    return ingest(source, tables=tables, **kwargs)


def _resolve_tables(
    source: DataSource, tables: Sequence[TableRef | str] | None
) -> list[TableRef]:
    available = list(source.tables())
    if tables is None:
        return available
    by_name: dict[str, TableRef] = {}
    for ref in available:
        by_name[ref.qualified] = ref
        by_name.setdefault(ref.name, ref)
    out: list[TableRef] = []
    for wanted in tables:
        if isinstance(wanted, TableRef):
            out.append(wanted)
            continue
        ref = by_name.get(str(wanted))
        if ref is None:
            raise SourceError(
                f"table {wanted!r} not found in source {source.source_id!r}; "
                f"available: {sorted(by_name)}"
            )
        out.append(ref)
    return out
