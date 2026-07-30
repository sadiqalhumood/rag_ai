"""Source-independent introspection helpers.

Everything in this module is pure: no driver imports, no I/O, no SQL execution.
It holds the parts of "understanding a table" that do not depend on which
backend the table came from -- type-name normalisation, date sniffing,
identifier quoting, and primary-key string formatting.

Keeping this here rather than inside each adapter is what makes the
source-agnosticism claim cheap to honour: a new backend supplies rows and
declared type names, and inherits the rest.

Owned by source-eng. Imports only from `anyrag.core`.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any, Iterable, Sequence

from ..core.types import TableRef

# --------------------------------------------------------------------------
# Type families
# --------------------------------------------------------------------------
#
# A *family* is the coarse storage class of a declared type, normalised across
# dialects: SQLite's `VARCHAR(20)`, Postgres' `character varying`, and a CSV
# header's absent type all have to land somewhere comparable.

INTEGER = "integer"
FLOAT = "float"
NUMERIC = "numeric"
TEXT = "text"
BOOLEAN = "boolean"
DATE = "date"
TIMESTAMP = "timestamp"
TIME = "time"
BLOB = "blob"
JSON = "json"
UUID = "uuid"
OTHER = "other"
UNKNOWN = "unknown"

NUMERIC_FAMILIES = frozenset({INTEGER, FLOAT, NUMERIC})
TEMPORAL_FAMILIES = frozenset({DATE, TIMESTAMP, TIME})
#: Families whose values are strings we may want to read as prose.
TEXTUAL_FAMILIES = frozenset({TEXT})

# Exact matches take priority over substring rules, because several dialect
# type names are substrings of one another (`point` contains `int`, `timestamp`
# contains `time`, `serial` is an integer).
_EXACT_TYPES: dict[str, str] = {
    "": UNKNOWN,
    "int": INTEGER,
    "int2": INTEGER,
    "int4": INTEGER,
    "int8": INTEGER,
    "integer": INTEGER,
    "smallint": INTEGER,
    "bigint": INTEGER,
    "tinyint": INTEGER,
    "mediumint": INTEGER,
    "serial": INTEGER,
    "bigserial": INTEGER,
    "smallserial": INTEGER,
    "float": FLOAT,
    "float4": FLOAT,
    "float8": FLOAT,
    "real": FLOAT,
    "double": FLOAT,
    "double precision": FLOAT,
    "numeric": NUMERIC,
    "decimal": NUMERIC,
    "money": NUMERIC,
    "bool": BOOLEAN,
    "boolean": BOOLEAN,
    "date": DATE,
    "timestamp": TIMESTAMP,
    "timestamptz": TIMESTAMP,
    "datetime": TIMESTAMP,
    "timestamp with time zone": TIMESTAMP,
    "timestamp without time zone": TIMESTAMP,
    "time": TIME,
    "timetz": TIME,
    "time with time zone": TIME,
    "time without time zone": TIME,
    "text": TEXT,
    "varchar": TEXT,
    "character varying": TEXT,
    "char": TEXT,
    "character": TEXT,
    "bpchar": TEXT,
    "clob": TEXT,
    "string": TEXT,
    "citext": TEXT,
    "name": TEXT,
    "blob": BLOB,
    "bytea": BLOB,
    "binary": BLOB,
    "varbinary": BLOB,
    "json": JSON,
    "jsonb": JSON,
    "uuid": UUID,
    "point": OTHER,
    "interval": OTHER,
    "xml": OTHER,
    "array": OTHER,
}

# Ordered substring rules, applied only when no exact match was found.
_SUBSTRING_TYPES: tuple[tuple[str, str], ...] = (
    ("bool", BOOLEAN),
    ("timestamp", TIMESTAMP),
    ("datetime", TIMESTAMP),
    ("date", DATE),
    ("time", TIME),
    ("serial", INTEGER),
    ("int", INTEGER),
    ("double", FLOAT),
    ("float", FLOAT),
    ("real", FLOAT),
    ("numeric", NUMERIC),
    ("decimal", NUMERIC),
    ("money", NUMERIC),
    ("uuid", UUID),
    ("json", JSON),
    ("char", TEXT),
    ("text", TEXT),
    ("clob", TEXT),
    ("string", TEXT),
    ("blob", BLOB),
    ("bytea", BLOB),
    ("binary", BLOB),
)

_TYPE_PARAMS_RE = re.compile(r"\s*\(.*?\)\s*")
_ARRAY_SUFFIX_RE = re.compile(r"(\s*\[\s*\d*\s*\])+$")


def normalize_type(type_name: str | None) -> str:
    """Map a declared type name onto a family constant.

    Unrecognised names return UNKNOWN rather than guessing; callers fall back
    to inspecting values, which is the only honest option for an untyped
    source.
    """
    if not type_name:
        return UNKNOWN
    raw = str(type_name).strip().lower()
    raw = _ARRAY_SUFFIX_RE.sub("", raw)
    raw = _TYPE_PARAMS_RE.sub(" ", raw).strip()
    raw = re.sub(r"\s+", " ", raw)
    if raw.startswith("unsigned "):
        raw = raw[len("unsigned ") :]
    if raw in _EXACT_TYPES:
        return _EXACT_TYPES[raw]
    for needle, family in _SUBSTRING_TYPES:
        if needle in raw:
            return family
    return UNKNOWN


def family_from_values(values: Iterable[Any]) -> str:
    """Infer a family from observed Python values.

    Used for sources with no declared types (SQLite columns declared with an
    empty type, CSV headers) and as a tie-breaker when `normalize_type` gives
    UNKNOWN.
    """
    seen: set[str] = set()
    n = 0
    for v in values:
        if v is None:
            continue
        n += 1
        if isinstance(v, bool):
            seen.add(BOOLEAN)
        elif isinstance(v, int):
            seen.add(INTEGER)
        elif isinstance(v, float):
            seen.add(FLOAT)
        elif isinstance(v, str):
            seen.add(TEXT)
        elif isinstance(v, (bytes, bytearray, memoryview)):
            seen.add(BLOB)
        elif isinstance(v, _dt.datetime):
            seen.add(TIMESTAMP)
        elif isinstance(v, _dt.date):
            seen.add(DATE)
        elif isinstance(v, _dt.time):
            seen.add(TIME)
        else:
            seen.add(OTHER)
        if n >= 500:
            break
    if not seen:
        return UNKNOWN
    if len(seen) == 1:
        return seen.pop()
    # Mixed int/float is still numeric; anything else mixed is untyped soup.
    if seen <= {INTEGER, FLOAT}:
        return FLOAT
    if seen <= {DATE, TIMESTAMP}:
        return TIMESTAMP
    return UNKNOWN


# --------------------------------------------------------------------------
# Date sniffing
# --------------------------------------------------------------------------

# Deliberately conservative. A loose parser turns every zip code and version
# string into a DATE column, which downstream shows up as nonsense range
# filters -- far worse than leaving the column as text.
_ISO_DATE_RE = re.compile(
    r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})"
    r"(?:[ T](?P<H>\d{2}):(?P<M>\d{2})(?::(?P<S>\d{2})(?:\.\d+)?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)?$"
)
_SLASH_YMD_RE = re.compile(r"^(?P<y>\d{4})/(?P<m>\d{2})/(?P<d>\d{2})$")
_COMPACT_YMD_RE = re.compile(r"^(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})$")


def looks_like_date(value: Any) -> bool:
    """True when `value` is, or reads as, an ISO-ish calendar date."""
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (_dt.date, _dt.datetime)):
        return True
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    for rex in (_ISO_DATE_RE, _SLASH_YMD_RE, _COMPACT_YMD_RE):
        m = rex.match(text)
        if not m:
            continue
        try:
            _dt.date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
        except ValueError:
            return False
        return True
    return False


def date_parse_fraction(values: Sequence[Any]) -> float:
    """Fraction of non-null `values` that read as dates. 0.0 when all null."""
    considered = [v for v in values if v is not None]
    if not considered:
        return 0.0
    hits = sum(1 for v in considered if looks_like_date(v))
    return hits / len(considered)


# --------------------------------------------------------------------------
# Identifiers
# --------------------------------------------------------------------------


def quote_ident(name: str) -> str:
    """Double-quote an identifier, doubling embedded quotes.

    ANSI quoting, which SQLite and Postgres both accept. This is the only
    place identifiers get interpolated into SQL -- table and column names
    cannot be passed as bind parameters, so they must be quoted rather than
    trusted.
    """
    text = str(name)
    if "\x00" in text:
        raise ValueError(f"identifier contains a NUL byte: {text!r}")
    return '"' + text.replace('"', '""') + '"'


def qualified_ident(table: TableRef) -> str:
    """`"schema"."table"`, or just `"table"` when the ref has no schema."""
    if table.schema:
        return f"{quote_ident(table.schema)}.{quote_ident(table.name)}"
    return quote_ident(table.name)


def coerce_table_ref(
    table: TableRef | str, *, default_schema: str | None = None
) -> TableRef:
    """Accept either a TableRef or a `"schema.table"` / `"table"` string.

    The Protocol says TableRef, and adapters honour that; accepting a string
    as well removes a class of annoying call-site noise without weakening the
    contract.
    """
    if isinstance(table, TableRef):
        if table.schema is None and default_schema is not None:
            return TableRef(table.name, default_schema)
        return table
    text = str(table)
    if "." in text:
        schema, _, name = text.partition(".")
        return TableRef(name, schema)
    return TableRef(text, default_schema)


# --------------------------------------------------------------------------
# Primary keys
# --------------------------------------------------------------------------

#: Separator for composite primary keys. Chosen once, here, so that every
#: adapter (and the eval harness building gold RowRefs) produces byte-identical
#: keys for the same logical row.
PK_SEPARATOR = "|"


def pk_string(values: Sequence[Any]) -> str:
    """Format primary-key value(s) as the canonical string used by RowRef.

    Always a string, because sources disagree about types for the same logical
    key (SQLite INTEGER 7, CSV "7", Postgres bigint 7) and gold matching must
    not depend on which adapter loaded the data.
    """
    return PK_SEPARATOR.join(str(v) for v in values)


def row_pk_string(row: dict[str, Any] | Any, pk_columns: Sequence[str]) -> str:
    """`pk_string` applied to the pk columns of a row mapping."""
    return pk_string([row[c] if c in row else None for c in pk_columns])


_CAMEL_ID_RE = re.compile(r"[a-z0-9]Id$")


def looks_like_id_name(name: str) -> bool:
    """True for `id`, `user_id`, `userId` -- not for `paid` or `valid`."""
    text = str(name).strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered == "id" or lowered.endswith("_id"):
        return True
    # camelCase: a lowercase/digit immediately before a capital "Id".
    return bool(_CAMEL_ID_RE.search(text))


__all__ = [
    "BLOB",
    "BOOLEAN",
    "DATE",
    "FLOAT",
    "INTEGER",
    "JSON",
    "NUMERIC",
    "NUMERIC_FAMILIES",
    "OTHER",
    "PK_SEPARATOR",
    "TEMPORAL_FAMILIES",
    "TEXT",
    "TEXTUAL_FAMILIES",
    "TIME",
    "TIMESTAMP",
    "UNKNOWN",
    "UUID",
    "coerce_table_ref",
    "date_parse_fraction",
    "family_from_values",
    "looks_like_date",
    "looks_like_id_name",
    "normalize_type",
    "pk_string",
    "qualified_ident",
    "quote_ident",
    "row_pk_string",
]
