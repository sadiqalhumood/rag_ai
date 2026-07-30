"""Deterministic chunk identity.

The single rule this module exists to enforce:

    chunk_id = sha256(source_id | kind | table | pk | part_index)[:32]

**Content is deliberately not part of the id.** A chunk's identity is *what it
is about* -- a particular part of a particular row of a particular table of a
particular source -- not what it currently says. Hashing the text in would mint
a fresh id every time a row is edited, so re-ingestion would append a duplicate
instead of updating in place, and the index would grow without bound on a
corpus that never changed size.

Change detection is still needed, so it lives beside identity rather than
inside it: `content_hash(text)` goes into `meta["content_hash"]`. An edited row
therefore keeps its `chunk_id` and changes its `content_hash`, which is exactly
what an incremental upsert needs to decide whether to re-embed.

Everything here is pure: no I/O, no tokenizer, no global state.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Iterable, Mapping, Sequence

from ..core.types import ChunkKind, Row, TableRef

#: Field separator inside the id payload. ASCII unit separator: it cannot occur
#: in a table name or in a sanely-typed primary key, so distinct field tuples
#: cannot collide by concatenation ("ab|c" vs "a|bc").
UNIT_SEP = "\x1f"

#: Separator between components of a composite primary key. Kept printable
#: because this string also becomes `RowRef.pk`, which humans read in citations
#: and the eval harness compares against gold row ids.
PK_SEP = "|"

#: Rendering of a NULL inside a primary key. PKs are non-null by definition, so
#: this only matters for degenerate sources; it is spelled unambiguously so it
#: cannot be confused with the literal string "None" or with an empty value.
NULL_PK = "\x00"

#: Truncation length of the hex digest. 32 hex chars = 128 bits; birthday
#: collision risk is negligible for any corpus that fits on a disk.
ID_LENGTH = 32


def _normalize_kind(kind: ChunkKind | str) -> str:
    """`ChunkKind` is a str-Enum, but `str(ChunkKind.ROW)` is 'ChunkKind.ROW'.

    Taking `.value` explicitly keeps ids stable across Python versions, which
    have changed str-Enum formatting more than once.
    """
    return kind.value if isinstance(kind, ChunkKind) else str(kind)


def _normalize_table(table: TableRef | str | None) -> str:
    if table is None:
        return ""
    if isinstance(table, TableRef):
        return table.qualified
    return str(table)


def normalize_pk_value(value: Any) -> str:
    """Render one primary-key component as a stable string.

    Sources disagree about types for the same logical key -- SQLite hands back
    `7`, a CSV reader `"7"`, Postgres a `bigint`, a Parquet file a `numpy.int64`
    -- and chunk ids must not depend on which adapter happened to load the data.
    """
    if value is None:
        return NULL_PK
    if isinstance(value, bool):
        # Before the int branch: bool is a subclass of int. 1/0 matches what
        # SQLite and CSV produce for the same column.
        return "1" if value else "0"
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    if isinstance(value, float):
        # 7.0 and 7 are the same key. Anything genuinely fractional keeps repr.
        return str(int(value)) if value.is_integer() else repr(value)
    return str(value)


def make_pk(values: Any) -> str:
    """Join one or more key components into the canonical pk string.

    A scalar passes through `normalize_pk_value`; a sequence or mapping becomes
    `"a|b|c"` in the order given. Callers must pass composite key columns in a
    fixed order (schema order) so the result is stable across runs.
    """
    if values is None:
        return NULL_PK
    if isinstance(values, Mapping):
        parts: Iterable[Any] = list(values.values())
    elif isinstance(values, (str, bytes)) or not isinstance(values, (Sequence, tuple, list)):
        return normalize_pk_value(values)
    else:
        parts = values
    rendered = [normalize_pk_value(v) for v in parts]
    if len(rendered) == 1:
        return rendered[0]
    return PK_SEP.join(rendered)


def pk_for_row(row: Row, key_columns: Sequence[str]) -> str:
    """Canonical pk for `row` given its key columns, in the order supplied."""
    return make_pk([row.get(c) for c in key_columns])


def id_payload(
    *,
    source_id: str,
    kind: ChunkKind | str,
    table: TableRef | str | None,
    pk: str,
    part_index: int = 0,
) -> str:
    """The exact string that gets hashed. Exposed so tests can pin the format."""
    return UNIT_SEP.join(
        (
            str(source_id),
            _normalize_kind(kind),
            _normalize_table(table),
            str(pk),
            str(int(part_index)),
        )
    )


def chunk_id(
    *,
    source_id: str,
    kind: ChunkKind | str,
    table: TableRef | str | None,
    pk: str = "",
    part_index: int = 0,
) -> str:
    """Deterministic chunk id. Content is intentionally absent -- see module doc."""
    payload = id_payload(
        source_id=source_id, kind=kind, table=table, pk=pk, part_index=part_index
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:ID_LENGTH]


def content_hash(text: str) -> str:
    """Full sha256 of chunk text -- the change detector, stored in `meta`."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_chunk_id(
    *, source_id: str, table: TableRef | str, pk: str, part_index: int = 0
) -> str:
    return chunk_id(
        source_id=source_id,
        kind=ChunkKind.ROW,
        table=table,
        pk=pk,
        part_index=part_index,
    )


def schema_card_id(
    *, source_id: str, table: TableRef | str, part_index: int = 0
) -> str:
    """Schema cards describe a table, not a row, so their pk slot is empty."""
    return chunk_id(
        source_id=source_id,
        kind=ChunkKind.SCHEMA_CARD,
        table=table,
        pk="",
        part_index=part_index,
    )
