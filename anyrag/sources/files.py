"""Flat-file source adapter: a directory of CSV and/or Parquet files.

One file, one table; the table name is the filename stem. The files are read
with pyarrow and materialised into a private in-memory SQLite database, which
is then locked read-only. That gives the adapter a real SQL engine for
`execute_readonly` without inventing a second query dialect, and keeps this
module thin -- it is a *loader* plus the same schema/profile/iterate surface
every other adapter implements.

Three decisions worth naming:

* **Surrogate keys.** Flat files have no primary key. Inventing one from
  "looks unique in this sample" would be a landmine: the eval harness matches
  gold rows by `(table, pk)`, so a key that shifts when a column changes
  silently zeroes retrieval scores. Instead every table gets `_row_ordinal`,
  the 0-based position of the row *within its file*. It is reproducible for
  identical input, independent of column content, and stable under re-load
  because both readers (pyarrow CSV and Parquet) preserve file order.

* **Read-only is enforced after loading, not during.** Building the database
  requires writes, so the lockdown happens once, at the end of `__init__`:
  `PRAGMA query_only=ON` plus a SQLite authorizer that denies every write
  action code. Neither is ever lifted -- there is no lazy-load path that would
  need to reopen the door. `execute_readonly` still runs `lint_readonly`
  first, so a query has to pass the linter *and* the engine.

* **A bad file is skipped, not fatal.** A directory is a bag of files of
  varying quality; one ragged CSV must not take out the other nine tables.
  Failures are recorded in `load_errors` rather than swallowed, so a caller
  can report what was dropped.

Type handling differs by format on purpose. Parquet carries declared Arrow
types and a nullability flag, so those are used directly. CSV declares
nothing: pyarrow's own value inference supplies a storage type, and anything
it leaves as a string falls through to the shared value-based inference in
`profile.infer_role` -- which is how a `yes`/`no` column becomes BOOLEAN and a
`2025/03/04` column becomes DATE.

Owned by source-eng.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pa_pq

from ..core.errors import SourceError
from ..core.lint import lint_readonly
from ..core.registry import register_source
from ..core.types import (
    ColumnSchema,
    QueryResult,
    Row,
    TableProfile,
    TableRef,
    TableSchema,
)
from .introspect import (
    coerce_table_ref,
    pk_string,
    quote_ident,
    row_pk_string,
)
from .profile import build_table_profile

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Name of the synthesised surrogate key column. Leading underscore keeps it
#: out of the way of real CSV headers; a genuine collision is renamed instead
#: of clobbered (see `_unique_column_names`).
ROW_ORDINAL = "_row_ordinal"

#: Recognised delimited-text extensions and their field separator.
CSV_SUFFIXES: dict[str, str] = {".csv": ",", ".tsv": "\t"}
PARQUET_SUFFIXES: frozenset[str] = frozenset({".parquet", ".pq"})

#: Tables at or below this size are profiled by full scan (exact statistics).
DEFAULT_FULL_SCAN_MAX_ROWS = 20_000
#: Larger tables are profiled from a sample of this many rows.
DEFAULT_PROFILE_SAMPLE_ROWS = 5_000
#: Rows pulled from pyarrow per load batch, bounding peak memory during load.
LOAD_BATCH_ROWS = 8_192

# Deterministic pseudo-random sampling order. `ORDER BY RANDOM()` would make
# two profile runs over identical input disagree, and this adapter's whole
# reason for existing is that identical input produces identical output. A
# multiply-mod-prime permutation of the ordinal is stable *and* uncorrelated
# with file order, so it does not bias statistics toward the head of the file
# the way `LIMIT n` alone would.
_SHUFFLE_MULTIPLIER = 2_654_435_761
_SHUFFLE_MODULUS = 4_294_967_291

# Authorizer action codes that must never be permitted on the loaded database.
# Assembled with getattr so a future Python that renames one degrades to "one
# fewer denial" rather than an import error.
_DENIED_ACTIONS: frozenset[int] = frozenset(
    code
    for code in (
        getattr(sqlite3, name, None)
        for name in (
            "SQLITE_INSERT",
            "SQLITE_UPDATE",
            "SQLITE_DELETE",
            "SQLITE_CREATE_TABLE",
            "SQLITE_CREATE_TEMP_TABLE",
            "SQLITE_CREATE_INDEX",
            "SQLITE_CREATE_TEMP_INDEX",
            "SQLITE_CREATE_VIEW",
            "SQLITE_CREATE_TEMP_VIEW",
            "SQLITE_CREATE_TRIGGER",
            "SQLITE_CREATE_TEMP_TRIGGER",
            "SQLITE_DROP_TABLE",
            "SQLITE_DROP_TEMP_TABLE",
            "SQLITE_DROP_INDEX",
            "SQLITE_DROP_TEMP_INDEX",
            "SQLITE_DROP_VIEW",
            "SQLITE_DROP_TEMP_VIEW",
            "SQLITE_DROP_TRIGGER",
            "SQLITE_DROP_TEMP_TRIGGER",
            "SQLITE_ALTER_TABLE",
            "SQLITE_ATTACH",
            "SQLITE_DETACH",
            "SQLITE_TRANSACTION",
            "SQLITE_PRAGMA",
        )
    )
    if code is not None
)


# --------------------------------------------------------------------------
# Arrow -> (SQLite storage, declared type name, value converter)
# --------------------------------------------------------------------------


def _iso(value: Any) -> str:
    return value.isoformat()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


def _to_float(value: Any) -> float:
    return float(value) if isinstance(value, Decimal) else float(value)


def _to_int(value: Any) -> int:
    return int(bool(value))


def arrow_column_spec(dtype: pa.DataType) -> tuple[str, str, Callable[[Any], Any] | None]:
    """Map an Arrow type onto how this adapter stores and describes it.

    Returns `(sqlite_storage_class, declared_type_name, converter)`. The
    declared type name is deliberately spelled the way `introspect.normalize_type`
    expects (`bigint`, `double`, `timestamp`, ...) so that role inference gets
    the same declared-type signal from Parquet that it gets from a real
    database.

    Temporal values are stored as ISO strings rather than as Python
    `date`/`datetime` objects: sqlite3's implicit date adapters are deprecated,
    and ISO text sorts correctly, so nothing is lost and a deprecation warning
    is avoided.
    """
    if pa.types.is_dictionary(dtype):
        dtype = dtype.value_type

    if pa.types.is_boolean(dtype):
        return ("INTEGER", "boolean", _to_int)
    if pa.types.is_integer(dtype):
        return ("INTEGER", "bigint", None)
    if pa.types.is_floating(dtype):
        return ("REAL", "double", None)
    if pa.types.is_decimal(dtype):
        return ("REAL", "numeric", _to_float)
    if pa.types.is_date(dtype):
        return ("TEXT", "date", _iso)
    if pa.types.is_timestamp(dtype):
        return ("TEXT", "timestamp", _iso)
    if pa.types.is_time(dtype):
        return ("TEXT", "time", _iso)
    if pa.types.is_duration(dtype) or pa.types.is_interval(dtype):
        return ("TEXT", "interval", str)
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return ("TEXT", "text", None)
    if (
        pa.types.is_binary(dtype)
        or pa.types.is_large_binary(dtype)
        or pa.types.is_fixed_size_binary(dtype)
    ):
        return ("BLOB", "blob", None)
    if pa.types.is_null(dtype):
        # An all-null column carries no evidence at all. Calling it text would
        # be a guess; UNKNOWN is the honest classification and is what an empty
        # declared type produces downstream.
        return ("TEXT", "", None)
    if (
        pa.types.is_list(dtype)
        or pa.types.is_large_list(dtype)
        or pa.types.is_struct(dtype)
        or pa.types.is_map(dtype)
    ):
        return ("TEXT", "json", _json_text)
    return ("TEXT", "other", str)


def _unique_column_names(names: Iterable[str], reserved: Iterable[str]) -> list[str]:
    """De-duplicate column names, reserving the surrogate-key name.

    A CSV with two `amount` headers, or one that literally contains a
    `_row_ordinal` column, is a real thing to receive. Renaming the later
    occurrence keeps the table loadable; failing the whole file over a header
    quirk would lose data for no benefit.
    """
    seen = set(reserved)
    out: list[str] = []
    for raw in names:
        base = str(raw) if str(raw) else "column"
        candidate, n = base, 1
        while candidate in seen:
            n += 1
            candidate = f"{base}_{n}"
        seen.add(candidate)
        out.append(candidate)
    return out


# --------------------------------------------------------------------------
# Loaded-table bookkeeping
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedFile:
    """What the adapter knows about one successfully loaded file."""

    ref: TableRef
    path: str
    fmt: str
    schema: TableSchema
    row_count: int


@register_source("files")
class FilesSource:
    """A read-only DataSource over a directory of CSV and/or Parquet files.

    URI form: ``files:/path/to/directory`` (a single file path also works).

    Every table gains a `_row_ordinal` primary key: the 0-based position of the
    row within its file. See the module docstring for why a synthesised key is
    preferable to guessing at a natural one.
    """

    def __init__(
        self,
        locator: str,
        *,
        source_id: str | None = None,
        full_scan_max_rows: int = DEFAULT_FULL_SCAN_MAX_ROWS,
        profile_sample_rows: int = DEFAULT_PROFILE_SAMPLE_ROWS,
        recursive: bool = False,
        **_ignored: Any,
    ) -> None:
        self.locator = str(locator).strip()
        if not self.locator:
            raise SourceError("files source requires a path, e.g. files:/data/csvs")

        root = Path(self.locator).expanduser()
        if not root.exists():
            raise SourceError(f"files source path not found: {root}")

        if root.is_file():
            paths = [root.resolve()]
            display = root.resolve().parent.name or str(root.resolve().parent)
        else:
            resolved = root.resolve()
            globber = resolved.rglob if recursive else resolved.glob
            # Sorted so table discovery order -- and therefore everything
            # derived from it -- is identical on every platform and run.
            paths = sorted(
                (p for p in globber("*") if p.is_file() and _data_format(p)),
                key=lambda p: str(p),
            )
            display = resolved.name or str(resolved)

        self.root = str(root.resolve())
        self.source_id = source_id or f"files:{display}"
        self.full_scan_max_rows = int(full_scan_max_rows)
        self.profile_sample_rows = int(profile_sample_rows)

        #: filename -> "ExcType: message" for every file that could not load.
        self.load_errors: dict[str, str] = {}
        self._loaded: dict[str, LoadedFile] = {}

        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        self._conn: sqlite3.Connection | None = conn

        for path in paths:
            fmt = _data_format(path)
            if fmt is None:  # pragma: no cover - filtered above
                continue
            name = path.stem
            if not name:
                self.load_errors[path.name] = "ValueError: empty table name"
                continue
            if name in self._loaded:
                self.load_errors[path.name] = (
                    f"ValueError: table name {name!r} already taken by "
                    f"{self._loaded[name].path}"
                )
                continue
            try:
                self._loaded[name] = self._load_file(conn, path, name, fmt)
            except Exception as exc:  # noqa: BLE001 - one bad file, not a bad source
                try:
                    conn.execute(f"DROP TABLE IF EXISTS {quote_ident(name)}")
                except sqlite3.Error:  # pragma: no cover - best effort cleanup
                    pass
                self.load_errors[path.name] = f"{type(exc).__name__}: {exc}"

        conn.commit()
        # Lockdown, in this order: the pragma has to be set before the
        # authorizer starts denying PRAGMA.
        conn.execute("PRAGMA query_only=ON")
        conn.set_authorizer(_deny_writes)

    # -- loading ----------------------------------------------------------

    def _load_file(
        self, conn: sqlite3.Connection, path: Path, name: str, fmt: str
    ) -> LoadedFile:
        arrow_schema, batches = _open_reader(path, fmt)
        fields = list(arrow_schema)
        col_names = _unique_column_names((f.name for f in fields), (ROW_ORDINAL,))
        specs = [arrow_column_spec(f.type) for f in fields]

        ident = quote_ident(name)
        decls = [f"{quote_ident(ROW_ORDINAL)} INTEGER PRIMARY KEY"]
        decls += [
            f"{quote_ident(col)} {spec[0]}" for col, spec in zip(col_names, specs)
        ]
        conn.execute(f"CREATE TABLE {ident} ({', '.join(decls)})")

        columns_sql = ", ".join(
            quote_ident(c) for c in (ROW_ORDINAL, *col_names)
        )
        placeholders = ", ".join("?" * (len(col_names) + 1))
        insert_sql = f"INSERT INTO {ident} ({columns_sql}) VALUES ({placeholders})"

        converters = [spec[2] for spec in specs]
        null_counts = [0] * len(col_names)
        ordinal = 0
        for batch in batches:
            n_rows = batch.num_rows
            if not n_rows:
                continue
            batch_cols = [batch.column(i).to_pylist() for i in range(len(col_names))]
            payload: list[tuple[Any, ...]] = []
            for r in range(n_rows):
                values: list[Any] = [ordinal + r]
                for ci, col in enumerate(batch_cols):
                    value = col[r]
                    if value is None:
                        null_counts[ci] += 1
                    else:
                        conv = converters[ci]
                        if conv is not None:
                            value = conv(value)
                    values.append(value)
                payload.append(tuple(values))
            conn.executemany(insert_sql, payload)
            ordinal += n_rows

        columns = [
            ColumnSchema(
                name=ROW_ORDINAL,
                type_name="integer",
                nullable=False,
                is_primary_key=True,
            )
        ]
        for field, col, spec, nulls in zip(fields, col_names, specs, null_counts):
            # Parquet declares nullability; CSV does not, so for CSV the flag
            # reports what was *observed* rather than what was promised.
            declared_nullable = bool(field.nullable) if fmt == "parquet" else False
            columns.append(
                ColumnSchema(
                    name=col,
                    type_name=spec[1],
                    nullable=declared_nullable or nulls > 0,
                    is_primary_key=False,
                    # Flat files carry no referential metadata. Guessing at
                    # foreign keys from column names would fabricate joins the
                    # data never declared.
                    references=None,
                )
            )

        ref = TableRef(name, None)
        schema = TableSchema(
            table=ref, columns=tuple(columns), primary_key=(ROW_ORDINAL,)
        )
        return LoadedFile(
            ref=ref, path=str(path), fmt=fmt, schema=schema, row_count=ordinal
        )

    # -- plumbing ---------------------------------------------------------

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise SourceError(f"{self.source_id} is closed")
        return self._conn

    def _entry(self, table: TableRef | str) -> LoadedFile:
        ref = coerce_table_ref(table)
        entry = self._loaded.get(ref.name)
        if entry is None:
            known = ", ".join(sorted(self._loaded)) or "(none)"
            raise SourceError(
                f"{self.source_id}: no such table {ref.name!r}; loaded: {known}"
            )
        return entry

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Lint, execute, and materialise a small internal query."""
        conn = self._require_conn()
        clean = lint_readonly(sql)
        try:
            return conn.execute(clean, params).fetchall()
        except sqlite3.Error as exc:
            raise SourceError(
                f"{self.source_id}: query failed: {exc}\n{clean}"
            ) from exc

    # -- DataSource -------------------------------------------------------

    def tables(self) -> list[TableRef]:
        return [self._loaded[name].ref for name in sorted(self._loaded)]

    def schema(self, table: TableRef | str) -> TableSchema:
        return self._entry(table).schema

    def pk_columns(self, table: TableRef | str) -> tuple[str, ...]:
        """The effective key columns -- always the synthesised row ordinal."""
        return self._entry(table).schema.primary_key or (ROW_ORDINAL,)

    def row_pk(self, table: TableRef | str, row: Row) -> str:
        """The canonical RowRef pk string for `row`."""
        return row_pk_string(row, self.pk_columns(table))

    def row_count(self, table: TableRef | str) -> int:
        return self._entry(table).row_count

    def source_path(self, table: TableRef | str) -> str:
        """The file a table was loaded from -- useful provenance for callers."""
        return self._entry(table).path

    def profile(self, table: TableRef | str) -> TableProfile:
        entry = self._entry(table)
        ident = quote_ident(entry.ref.name)
        order = quote_ident(ROW_ORDINAL)
        total = entry.row_count

        if total <= self.full_scan_max_rows:
            sql = (
                f"SELECT * FROM {ident} ORDER BY {order} "
                f"LIMIT {self.full_scan_max_rows}"
            )
        else:
            sql = (
                f"SELECT * FROM {ident} ORDER BY "
                f"(({order} * {_SHUFFLE_MULTIPLIER}) % {_SHUFFLE_MODULUS}) "
                f"LIMIT {self.profile_sample_rows}"
            )

        rows = [dict(r) for r in self._query(sql)]
        return build_table_profile(
            table=entry.ref, schema=entry.schema, row_count=total, rows=rows
        )

    def iter_rows(
        self, table: TableRef | str, batch_size: int = 1000
    ) -> Iterator[list[Row]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        entry = self._entry(table)
        sql = lint_readonly(
            f"SELECT * FROM {quote_ident(entry.ref.name)} "
            f"ORDER BY {quote_ident(ROW_ORDINAL)}"
        )
        conn = self._require_conn()
        cur = conn.execute(sql)
        try:
            while True:
                chunk = cur.fetchmany(batch_size)
                if not chunk:
                    return
                yield [dict(r) for r in chunk]
        finally:
            cur.close()

    def execute_readonly(self, sql: str, max_rows: int = 1000) -> QueryResult:
        clean = lint_readonly(sql)
        conn = self._require_conn()
        try:
            cur = conn.execute(clean)
        except sqlite3.Error as exc:
            raise SourceError(
                f"{self.source_id}: query failed: {exc}\n{clean}"
            ) from exc
        try:
            columns = tuple(d[0] for d in (cur.description or ()))
            if max_rows is None or max_rows <= 0:
                fetched = cur.fetchall()
                truncated = False
            else:
                fetched = cur.fetchmany(max_rows + 1)
                truncated = len(fetched) > max_rows
                fetched = fetched[:max_rows]
        finally:
            cur.close()
        return QueryResult(
            columns=columns,
            rows=tuple(tuple(r) for r in fetched),
            sql=clean,
            truncated=truncated,
        )

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:  # pragma: no cover - close is best effort
                pass

    # -- niceties ---------------------------------------------------------

    def __enter__(self) -> "FilesSource":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        state = "closed" if self._conn is None else "open"
        return (
            f"<FilesSource {self.source_id} "
            f"({len(self._loaded)} tables, {len(self.load_errors)} skipped, {state})>"
        )


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------


def _data_format(path: Path) -> str | None:
    """'csv', 'tsv' or 'parquet' for a supported file, else None."""
    suffix = path.suffix.lower()
    if suffix in CSV_SUFFIXES:
        return suffix.lstrip(".")
    if suffix in PARQUET_SUFFIXES:
        return "parquet"
    return None


def _open_reader(path: Path, fmt: str) -> tuple[pa.Schema, Iterable[pa.RecordBatch]]:
    """Return `(arrow_schema, batch_iterator)` without materialising the file.

    Both readers expose their schema before the first batch is consumed, which
    is what lets the SQLite table be created up front and the rows streamed in.
    """
    if fmt == "parquet":
        pf = pa_pq.ParquetFile(str(path))
        return pf.schema_arrow, pf.iter_batches(batch_size=LOAD_BATCH_ROWS)

    reader = pa_csv.open_csv(
        str(path),
        parse_options=pa_csv.ParseOptions(delimiter=CSV_SUFFIXES[f".{fmt}"]),
        # Without this an empty cell in a text column arrives as "" rather than
        # NULL, which would make every null-fraction statistic wrong for
        # exactly the columns where missingness matters most.
        convert_options=pa_csv.ConvertOptions(strings_can_be_null=True),
    )
    return reader.schema, reader


def _deny_writes(action: int, *_args: Any) -> int:
    """SQLite authorizer: refuse every mutating action code.

    This is the second lock, independent of `PRAGMA query_only`. It also denies
    PRAGMA itself, so nothing that slips past the linter can turn `query_only`
    back off.
    """
    return sqlite3.SQLITE_DENY if action in _DENIED_ACTIONS else sqlite3.SQLITE_OK


__all__ = [
    "CSV_SUFFIXES",
    "DEFAULT_FULL_SCAN_MAX_ROWS",
    "DEFAULT_PROFILE_SAMPLE_ROWS",
    "PARQUET_SUFFIXES",
    "ROW_ORDINAL",
    "FilesSource",
    "LoadedFile",
    "arrow_column_spec",
    "pk_string",
]
