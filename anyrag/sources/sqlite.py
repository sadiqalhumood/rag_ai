"""SQLite source adapter.

Read-only is enforced twice, because one layer is one bug away from a write:

1. the connection is opened with a `file:...?mode=ro` URI, so the driver never
   acquires a write lock on the file; and
2. `PRAGMA query_only=ON` is set on the connection, so even a statement that
   somehow got past the linter is refused by the engine.

Schema introspection uses SQLite's table-valued pragma functions
(`pragma_table_info`, `pragma_foreign_key_list`) rather than `PRAGMA`
statements. That is not cosmetic: the table-valued form is a `SELECT`, so it
passes `lint_readonly` and every SQL string this adapter executes -- including
its own introspection -- goes through the same gate as user queries.

Owned by source-eng.
"""

from __future__ import annotations

import sqlite3
import urllib.parse
from pathlib import Path
from typing import Any, Iterator

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

#: Tables at or below this size are profiled by full scan (exact statistics).
DEFAULT_FULL_SCAN_MAX_ROWS = 20_000
#: Larger tables are profiled from a random sample of this many rows.
DEFAULT_PROFILE_SAMPLE_ROWS = 5_000
#: SQLite's own rowid, used as the effective key for tables without a PK.
ROWID = "rowid"


@register_source("sqlite")
class SQLiteSource:
    """A read-only DataSource over a SQLite database file.

    URI form: ``sqlite:/abs/path/to/database.sqlite``
    """

    def __init__(
        self,
        locator: str,
        *,
        source_id: str | None = None,
        full_scan_max_rows: int = DEFAULT_FULL_SCAN_MAX_ROWS,
        profile_sample_rows: int = DEFAULT_PROFILE_SAMPLE_ROWS,
        include_views: bool = False,
        timeout: float = 30.0,
        **_ignored: Any,
    ) -> None:
        self.locator = str(locator).strip()
        if not self.locator:
            raise SourceError("sqlite source requires a path, e.g. sqlite:/data/x.db")

        if self.locator.startswith("file:"):
            uri = self.locator
            display = self.locator
        else:
            path = Path(self.locator).expanduser()
            if not path.exists():
                raise SourceError(f"sqlite database not found: {path}")
            resolved = path.resolve()
            uri = "file:" + urllib.parse.quote(str(resolved)) + "?mode=ro"
            display = resolved.stem or resolved.name

        self.path = display
        self.source_id = source_id or f"sqlite:{display}"
        self.full_scan_max_rows = int(full_scan_max_rows)
        self.profile_sample_rows = int(profile_sample_rows)
        self.include_views = bool(include_views)

        try:
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=timeout)
        except sqlite3.Error as exc:
            raise SourceError(f"cannot open sqlite database {self.locator!r}: {exc}") from exc
        conn.row_factory = sqlite3.Row
        # Second lock on the door; see module docstring.
        conn.execute("PRAGMA query_only=ON")
        self._conn: sqlite3.Connection | None = conn
        self._schema_cache: dict[TableRef, TableSchema] = {}
        self._table_cache: list[TableRef] | None = None

    # -- plumbing ---------------------------------------------------------

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise SourceError(f"{self.source_id} is closed")
        return self._conn

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Lint, execute, and fully materialise a small internal query."""
        conn = self._require_conn()
        clean = lint_readonly(sql)
        try:
            return conn.execute(clean, params).fetchall()
        except sqlite3.Error as exc:
            raise SourceError(f"{self.source_id}: query failed: {exc}\n{clean}") from exc

    def _ref(self, table: TableRef | str) -> TableRef:
        ref = coerce_table_ref(table)
        # SQLite has no schemas in the Postgres sense; ignore any that arrives.
        return TableRef(ref.name, None)

    # -- DataSource -------------------------------------------------------

    def tables(self) -> list[TableRef]:
        if self._table_cache is None:
            kinds = "('table', 'view')" if self.include_views else "('table')"
            rows = self._query(
                "SELECT name FROM sqlite_master "
                f"WHERE type IN {kinds} AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
            self._table_cache = [TableRef(r["name"], None) for r in rows]
        return list(self._table_cache)

    def schema(self, table: TableRef | str) -> TableSchema:
        ref = self._ref(table)
        cached = self._schema_cache.get(ref)
        if cached is not None:
            return cached

        info = self._query(
            'SELECT "cid", "name", "type", "notnull", "pk" '
            "FROM pragma_table_info(?) ORDER BY \"cid\"",
            (ref.name,),
        )
        if not info:
            raise SourceError(f"{self.source_id}: no such table {ref.qualified!r}")

        pk_order: list[tuple[int, str]] = []
        references = self._foreign_keys(ref.name)
        columns: list[ColumnSchema] = []
        for row in info:
            pk_pos = int(row["pk"] or 0)
            if pk_pos:
                pk_order.append((pk_pos, row["name"]))
            columns.append(
                ColumnSchema(
                    name=row["name"],
                    type_name=(row["type"] or "").strip(),
                    # A PK column is never logically nullable, and SQLite
                    # reports notnull=0 for INTEGER PRIMARY KEY.
                    nullable=not (int(row["notnull"] or 0) or pk_pos),
                    is_primary_key=bool(pk_pos),
                    references=references.get(row["name"]),
                )
            )

        primary_key = tuple(name for _, name in sorted(pk_order))
        schema = TableSchema(table=ref, columns=tuple(columns), primary_key=primary_key)
        self._schema_cache[ref] = schema
        return schema

    def _foreign_keys(self, name: str) -> dict[str, str]:
        """column -> "other_table.other_column" for every declared FK."""
        rows = self._query(
            'SELECT "id", "seq", "table", "from", "to" '
            "FROM pragma_foreign_key_list(?) "
            'ORDER BY "id", "seq"',
            (name,),
        )
        out: dict[str, str] = {}
        implicit: list[tuple[str, str, int]] = []
        for row in rows:
            from_col, to_col, other = row["from"], row["to"], row["table"]
            if to_col is None:
                # `REFERENCES other` with no column list targets other's PK,
                # positionally. Resolve it rather than dropping the edge.
                implicit.append((from_col, other, int(row["seq"] or 0)))
                continue
            out[from_col] = f"{other}.{to_col}"
        for from_col, other, seq in implicit:
            target = self._primary_key_columns(other)
            if seq < len(target):
                out[from_col] = f"{other}.{target[seq]}"
        return out

    def _primary_key_columns(self, name: str) -> tuple[str, ...]:
        rows = self._query(
            'SELECT "name", "pk" FROM pragma_table_info(?) ORDER BY "pk"',
            (name,),
        )
        ordered = sorted(
            ((int(r["pk"] or 0), r["name"]) for r in rows if int(r["pk"] or 0))
        )
        return tuple(n for _, n in ordered)

    def pk_columns(self, table: TableRef | str) -> tuple[str, ...]:
        """The *effective* key columns used for ordering and RowRefs.

        Falls back to SQLite's implicit `rowid` when a table declares no
        primary key, so every row still has a stable, deterministic identity.
        """
        schema = self.schema(table)
        return schema.primary_key or (ROWID,)

    def row_pk(self, table: TableRef | str, row: Row) -> str:
        """The canonical RowRef pk string for `row`."""
        return row_pk_string(row, self.pk_columns(table))

    def row_count(self, table: TableRef | str) -> int:
        ref = self._ref(table)
        self.schema(ref)  # existence check
        rows = self._query(f"SELECT COUNT(*) AS n FROM {quote_ident(ref.name)}")
        return int(rows[0]["n"])

    def profile(self, table: TableRef | str) -> TableProfile:
        ref = self._ref(table)
        schema = self.schema(ref)
        total = self.row_count(ref)
        ident = quote_ident(ref.name)

        if total <= self.full_scan_max_rows:
            order = ", ".join(quote_ident(c) for c in self.pk_columns(ref))
            sql = f"SELECT * FROM {ident} ORDER BY {order} LIMIT {self.full_scan_max_rows}"
        else:
            # Head-of-table sampling would bias every statistic toward the
            # oldest rows, which is exactly where dates and categories drift.
            sql = f"SELECT * FROM {ident} ORDER BY RANDOM() LIMIT {self.profile_sample_rows}"

        rows = [dict(r) for r in self._query(sql)]
        return build_table_profile(
            table=ref, schema=schema, row_count=total, rows=rows
        )

    def iter_rows(
        self, table: TableRef | str, batch_size: int = 1000
    ) -> Iterator[list[Row]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        ref = self._ref(table)
        schema = self.schema(ref)
        keys = self.pk_columns(ref)
        ident = quote_ident(ref.name)
        order = ", ".join(quote_ident(c) for c in keys)

        if schema.primary_key:
            projection = "*"
        else:
            # Surface the surrogate key so callers can build a RowRef without
            # a second round trip.
            projection = f'{ROWID} AS {quote_ident(ROWID)}, *'

        sql = lint_readonly(f"SELECT {projection} FROM {ident} ORDER BY {order}")
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
            raise SourceError(f"{self.source_id}: query failed: {exc}\n{clean}") from exc
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

    def __enter__(self) -> "SQLiteSource":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        state = "closed" if self._conn is None else "open"
        return f"<SQLiteSource {self.source_id} ({state})>"


__all__ = ["SQLiteSource", "pk_string"]
