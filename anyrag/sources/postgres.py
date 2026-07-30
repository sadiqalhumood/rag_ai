"""PostgreSQL source adapter (psycopg 3).

Read-only is enforced at three levels, for the same reason as the SQLite
adapter: the linter is a filter, not a guarantee.

1. `lint_readonly` gates every SQL string, including this adapter's own
   introspection queries.
2. The session runs with `default_transaction_read_only=on` and
   `connection.read_only = True`, so psycopg opens every transaction as
   ``BEGIN READ ONLY`` and the server rejects writes outright.
3. A `statement_timeout` is set in the connection options, so a "read-only"
   query cannot become an availability incident. Autocommit is off; nothing
   here ever commits a write.

Credentials never appear in code or in a committed file: the locator is either
empty or the name of an environment variable (``postgres:$ANYRAG_PG_DSN``).

Owned by source-eng.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Iterator, Sequence

from ..core.errors import ConfigError, SourceError
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
    qualified_ident,
    quote_ident,
    row_pk_string,
)
from .profile import build_table_profile

try:  # pragma: no cover - exercised by absence, not by tests
    import psycopg
    from psycopg.rows import dict_row, tuple_row
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "the postgres adapter requires psycopg 3 (`pip install 'psycopg[binary]'`)"
    ) from exc

#: Environment variable consulted when the locator is empty.
DEFAULT_DSN_ENV = "ANYRAG_PG_DSN"
#: Schemas that are never user data.
SYSTEM_SCHEMAS = ("pg_catalog", "information_schema", "pg_toast")

DEFAULT_STATEMENT_TIMEOUT_MS = 30_000
DEFAULT_FULL_SCAN_MAX_ROWS = 20_000
DEFAULT_PROFILE_SAMPLE_ROWS = 5_000


def resolve_dsn(locator: str) -> str:
    """Turn a source-URI locator into a libpq connection string.

    Accepted forms:

    * ``""``                    -> read `$ANYRAG_PG_DSN`
    * ``"$SOME_VAR"``           -> read `$SOME_VAR`
    * ``"//user@host/db"``      -> from `postgres://user@host/db`, re-prefixed
    * anything else             -> used verbatim as a libpq DSN

    The environment-variable forms exist so that a connection string is never
    committed; they are the documented default in `.env.example`.
    """
    text = (locator or "").strip()
    if not text:
        env_name = DEFAULT_DSN_ENV
        value = os.environ.get(env_name, "").strip()
        if not value:
            raise ConfigError(
                f"postgres source has no locator and ${env_name} is unset; "
                f"use postgres:${env_name} with the variable set, or "
                "postgres:postgresql://..."
            )
        return value
    if text.startswith("$"):
        env_name = text[1:].strip() or DEFAULT_DSN_ENV
        value = os.environ.get(env_name, "").strip()
        if not value:
            raise ConfigError(
                f"postgres source points at ${env_name}, which is unset or empty"
            )
        return value
    if text.startswith("//"):
        # `open_source` splits on the first ':', so `postgres://host/db`
        # arrives here as `//host/db`.
        return "postgresql:" + text
    return text


@register_source("postgres")
class PostgresSource:
    """A read-only DataSource over a PostgreSQL database.

    URI forms::

        postgres:$ANYRAG_PG_DSN
        postgres:postgresql://user@host:5432/dbname
        postgres://user@host:5432/dbname
    """

    def __init__(
        self,
        locator: str = "",
        *,
        source_id: str | None = None,
        schemas: Sequence[str] | None = None,
        statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS,
        full_scan_max_rows: int = DEFAULT_FULL_SCAN_MAX_ROWS,
        profile_sample_rows: int = DEFAULT_PROFILE_SAMPLE_ROWS,
        connect_timeout: int = 10,
        **_ignored: Any,
    ) -> None:
        self.locator = locator or ""
        dsn = resolve_dsn(self.locator)
        self.schemas = tuple(schemas) if schemas else None
        self.full_scan_max_rows = int(full_scan_max_rows)
        self.profile_sample_rows = int(profile_sample_rows)

        options = (
            f"-c statement_timeout={int(statement_timeout_ms)} "
            "-c default_transaction_read_only=on"
        )
        try:
            conn = psycopg.connect(
                dsn,
                autocommit=False,
                options=options,
                connect_timeout=connect_timeout,
                row_factory=dict_row,
            )
        except Exception as exc:
            # Never echo the DSN: it may carry a password.
            raise SourceError(f"cannot connect to postgres: {exc}") from exc
        conn.read_only = True
        self._conn: Any = conn

        info = conn.info
        self.database = info.dbname
        self.source_id = source_id or f"postgres:{self.database}"
        self._schema_cache: dict[TableRef, TableSchema] = {}
        self._table_cache: list[TableRef] | None = None

    # -- plumbing ---------------------------------------------------------

    def _require_conn(self) -> Any:
        if self._conn is None:
            raise SourceError(f"{self.source_id} is closed")
        return self._conn

    def _query(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        """Lint, execute inside a read-only transaction, materialise, commit."""
        conn = self._require_conn()
        clean = lint_readonly(sql)
        try:
            with conn.cursor() as cur:
                cur.execute(clean, params)
                rows = cur.fetchall() if cur.description else []
            conn.commit()
            return list(rows)
        except Exception as exc:
            conn.rollback()
            if isinstance(exc, SourceError):
                raise
            raise SourceError(f"{self.source_id}: query failed: {exc}\n{clean}") from exc

    def _ref(self, table: TableRef | str) -> TableRef:
        ref = coerce_table_ref(table)
        if ref.schema:
            return ref
        for known in self.tables():
            if known.name == ref.name:
                return known
        return TableRef(ref.name, "public")

    # -- DataSource -------------------------------------------------------

    def tables(self) -> list[TableRef]:
        if self._table_cache is None:
            params: list[Any] = [list(SYSTEM_SCHEMAS)]
            clause = "t.table_schema <> ALL(%s) AND t.table_schema NOT LIKE 'pg_%%'"
            if self.schemas:
                clause += " AND t.table_schema = ANY(%s)"
                params.append(list(self.schemas))
            rows = self._query(
                "SELECT t.table_schema AS s, t.table_name AS n "
                "FROM information_schema.tables t "
                f"WHERE t.table_type = 'BASE TABLE' AND {clause} "
                "ORDER BY t.table_schema, t.table_name",
                params,
            )
            self._table_cache = [TableRef(r["n"], r["s"]) for r in rows]
        return list(self._table_cache)

    def schema(self, table: TableRef | str) -> TableSchema:
        ref = self._ref(table)
        cached = self._schema_cache.get(ref)
        if cached is not None:
            return cached

        cols = self._query(
            "SELECT c.column_name AS name, c.data_type AS dtype, "
            "c.udt_name AS udt, c.is_nullable AS nullable "
            "FROM information_schema.columns c "
            "WHERE c.table_schema = %s AND c.table_name = %s "
            "ORDER BY c.ordinal_position",
            (ref.schema, ref.name),
        )
        if not cols:
            raise SourceError(f"{self.source_id}: no such table {ref.qualified!r}")

        primary_key = self._primary_key_columns(ref)
        references = self._foreign_keys(ref)
        pk_set = set(primary_key)

        columns = tuple(
            ColumnSchema(
                name=c["name"],
                # `data_type` says "USER-DEFINED"/"ARRAY" for some types;
                # `udt_name` is the concrete one, so prefer it when data_type
                # is uninformative.
                type_name=(
                    c["udt"]
                    if str(c["dtype"]).upper() in ("USER-DEFINED", "ARRAY")
                    else c["dtype"]
                ),
                nullable=(str(c["nullable"]).upper() == "YES")
                and c["name"] not in pk_set,
                is_primary_key=c["name"] in pk_set,
                references=references.get(c["name"]),
            )
            for c in cols
        )
        schema = TableSchema(table=ref, columns=columns, primary_key=primary_key)
        self._schema_cache[ref] = schema
        return schema

    def _primary_key_columns(self, ref: TableRef) -> tuple[str, ...]:
        rows = self._query(
            "SELECT kcu.column_name AS name "
            "FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "  ON tc.constraint_name = kcu.constraint_name "
            " AND tc.constraint_schema = kcu.constraint_schema "
            "WHERE tc.constraint_type = 'PRIMARY KEY' "
            "  AND tc.table_schema = %s AND tc.table_name = %s "
            "ORDER BY kcu.ordinal_position",
            (ref.schema, ref.name),
        )
        return tuple(r["name"] for r in rows)

    def _foreign_keys(self, ref: TableRef) -> dict[str, str]:
        """column -> "other_table.other_column".

        Read from `pg_catalog` rather than `information_schema`, because the
        usual `constraint_column_usage` join produces a cartesian product for
        composite foreign keys and silently mispairs the columns.
        """
        rows = self._query(
            "SELECT att.attname AS col, cl2.relname AS ref_table, "
            "       ns2.nspname AS ref_schema, att2.attname AS ref_col "
            "FROM pg_constraint con "
            "JOIN pg_class cl ON cl.oid = con.conrelid "
            "JOIN pg_namespace ns ON ns.oid = cl.relnamespace "
            "JOIN LATERAL unnest(con.conkey, con.confkey) "
            "     WITH ORDINALITY AS u(attnum, refattnum, ord) ON true "
            "JOIN pg_attribute att "
            "  ON att.attrelid = con.conrelid AND att.attnum = u.attnum "
            "JOIN pg_class cl2 ON cl2.oid = con.confrelid "
            "JOIN pg_namespace ns2 ON ns2.oid = cl2.relnamespace "
            "JOIN pg_attribute att2 "
            "  ON att2.attrelid = con.confrelid AND att2.attnum = u.refattnum "
            "WHERE con.contype = 'f' AND ns.nspname = %s AND cl.relname = %s "
            "ORDER BY con.conname, u.ord",
            (ref.schema, ref.name),
        )
        out: dict[str, str] = {}
        for r in rows:
            target = r["ref_table"]
            if r["ref_schema"] not in (ref.schema, "public"):
                target = f"{r['ref_schema']}.{r['ref_table']}"
            out.setdefault(r["col"], f"{target}.{r['ref_col']}")
        return out

    def pk_columns(self, table: TableRef | str) -> tuple[str, ...]:
        """Effective key columns.

        Postgres has no usable surrogate (`ctid` moves on UPDATE and VACUUM),
        so a table with no primary key falls back to *all* columns. That keeps
        row identity deterministic and reproducible across ingests, which is
        what RowRef gold-matching needs, at the cost of a longer key.
        """
        schema = self.schema(table)
        return schema.primary_key or schema.column_names

    def row_pk(self, table: TableRef | str, row: Row) -> str:
        return row_pk_string(row, self.pk_columns(table))

    def row_count(self, table: TableRef | str) -> int:
        ref = self._ref(table)
        self.schema(ref)
        rows = self._query(f"SELECT COUNT(*) AS n FROM {qualified_ident(ref)}")
        return int(rows[0]["n"])

    def profile(self, table: TableRef | str) -> TableProfile:
        ref = self._ref(table)
        schema = self.schema(ref)
        total = self.row_count(ref)
        ident = qualified_ident(ref)

        if total <= self.full_scan_max_rows:
            order = ", ".join(quote_ident(c) for c in self.pk_columns(ref))
            sql = f"SELECT * FROM {ident} ORDER BY {order} LIMIT {self.full_scan_max_rows}"
        else:
            sql = f"SELECT * FROM {ident} ORDER BY random() LIMIT {self.profile_sample_rows}"

        rows = self._query(sql)
        return build_table_profile(
            table=ref, schema=schema, row_count=total, rows=rows
        )

    def iter_rows(
        self, table: TableRef | str, batch_size: int = 1000
    ) -> Iterator[list[Row]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        ref = self._ref(table)
        self.schema(ref)
        order = ", ".join(quote_ident(c) for c in self.pk_columns(ref))
        sql = lint_readonly(
            f"SELECT * FROM {qualified_ident(ref)} ORDER BY {order}"
        )
        conn = self._require_conn()
        # A named (server-side) cursor keeps memory flat regardless of table
        # size; a client-side cursor would buffer the whole result first.
        name = "anyrag_iter_" + uuid.uuid4().hex
        try:
            with conn.cursor(name=name) as cur:
                cur.itersize = batch_size
                cur.execute(sql)
                while True:
                    chunk = cur.fetchmany(batch_size)
                    if not chunk:
                        break
                    yield [dict(r) for r in chunk]
        except BaseException:
            # BaseException, not Exception: a consumer that stops iterating
            # early closes the generator with GeneratorExit, and leaving the
            # transaction open would pin an idle-in-transaction session.
            conn.rollback()
            raise
        else:
            conn.commit()

    def execute_readonly(self, sql: str, max_rows: int = 1000) -> QueryResult:
        clean = lint_readonly(sql)
        conn = self._require_conn()
        try:
            # tuple_row, not the connection's dict_row: a result may legally
            # repeat a column name, and a dict would silently drop one.
            with conn.cursor(row_factory=tuple_row) as cur:
                cur.execute(clean)
                columns = tuple(d.name for d in (cur.description or ()))
                if max_rows is None or max_rows <= 0:
                    fetched = cur.fetchall()
                    truncated = False
                else:
                    fetched = cur.fetchmany(max_rows + 1)
                    truncated = len(fetched) > max_rows
                    fetched = fetched[:max_rows]
            conn.commit()
        except Exception as exc:
            conn.rollback()
            raise SourceError(f"{self.source_id}: query failed: {exc}\n{clean}") from exc
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
            except Exception:  # pragma: no cover - close is best effort
                pass

    # -- niceties ---------------------------------------------------------

    def __enter__(self) -> "PostgresSource":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        state = "closed" if self._conn is None else "open"
        return f"<PostgresSource {self.source_id} ({state})>"


__all__ = ["DEFAULT_DSN_ENV", "PostgresSource", "pk_string", "resolve_dsn"]
