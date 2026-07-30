"""Chunking: source records in, `Chunk` objects out.

Two strategies, independently switchable because the eval ablates them:

* `row_serializer` turns each row into a natural-language sentence.
* `schema_card` turns each table into one card describing its columns.

`ids` is the load-bearing module: chunk identity is derived from
(source, kind, table, pk, part_index) and never from content, so re-ingesting
unchanged data is a no-op and re-ingesting edited data updates in place.
"""

from __future__ import annotations

from .ids import (
    ID_LENGTH,
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
from .pipeline import (
    IngestConfig,
    IngestStats,
    TableChunker,
    ingest,
    ingest_tables,
)
from .row_serializer import (
    NULL_PHRASE,
    RowSerializer,
    RowSerializerConfig,
    RowText,
    format_value,
)
from .schema_card import SchemaCardConfig, build_schema_card, schema_card_text
from .truncate import (
    TRUNCATION_MARKER,
    Truncation,
    shared_boundary,
    split_with_overlap,
    truncate_field,
)

__all__ = [
    # ids
    "ID_LENGTH", "PK_SEP", "UNIT_SEP", "chunk_id", "content_hash", "id_payload",
    "make_pk", "normalize_pk_value", "pk_for_row", "row_chunk_id",
    "schema_card_id",
    # pipeline
    "IngestConfig", "IngestStats", "TableChunker", "ingest", "ingest_tables",
    # rows
    "NULL_PHRASE", "RowSerializer", "RowSerializerConfig", "RowText",
    "format_value",
    # schema cards
    "SchemaCardConfig", "build_schema_card", "schema_card_text",
    # truncation
    "TRUNCATION_MARKER", "Truncation", "shared_boundary", "split_with_overlap",
    "truncate_field",
]
