"""Integration tests for the PostgreSQL source adapter against a real server.

There is no mock in this file. PostgreSQL refuses to run as root, and this
environment *is* root, so the fixture initialises a throwaway cluster in a
temporary directory owned by an unprivileged user and starts it on a free
loopback port. If that cannot be done, every test here skips with the exact
reason -- a skipped test is honest, a mocked one that reports "passed" is not.

Point the tests at an existing server instead by setting `ANYRAG_PG_DSN`.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest

from anyrag.core.errors import ConfigError, SourceError, UnsafeQueryError
from anyrag.core.interfaces import DataSource
from anyrag.core.registry import available_sources, open_source
from anyrag.core.types import ColumnRole, QueryResult, TableRef
from test_sources_data import (
    EXPECTED_CUSTOMER_ROLES,
    EXPECTED_ORDER_ROLES,
    N_CUSTOMERS,
    populate_postgres,
)

psycopg = pytest.importorskip("psycopg", reason="psycopg 3 is not installed")

from anyrag.sources.postgres import (  # noqa: E402
    DEFAULT_DSN_ENV,
    PostgresSource,
    resolve_dsn,
)

TEST_DB = "anyrag_test"
CUSTOMERS = TableRef("customers", "public")
ORDERS = TableRef("orders", "public")
SHIPMENTS = TableRef("shipments", "public")
EVENTS = TableRef("events", "public")


# --------------------------------------------------------------------------
# Throwaway cluster
# --------------------------------------------------------------------------


def _bindir() -> str | None:
    """Locate PostgreSQL server binaries (not just the client)."""
    override = os.environ.get("ANYRAG_PG_BINDIR")
    if override and Path(override, "initdb").exists():
        return override
    candidates = sorted(Path("/usr/lib/postgresql").glob("*/bin"), reverse=True)
    candidates += sorted(Path("/usr/pgsql").glob("*/bin"), reverse=True)
    for path in candidates:
        if (path / "initdb").exists() and (path / "pg_ctl").exists():
            return str(path)
    which = shutil.which("initdb")
    return str(Path(which).parent) if which else None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _pick_unprivileged_user() -> str | None:
    """A non-root account that can own the cluster. None if we are not root."""
    if os.geteuid() != 0:
        return None
    import pwd

    for name in ("postgres", "ubuntu", "claude", "nobody"):
        try:
            entry = pwd.getpwnam(name)
        except KeyError:
            continue
        if entry.pw_uid != 0:
            return name
    return None


def _run(command: str, *, user: str | None, cwd: str) -> subprocess.CompletedProcess:
    if user:
        argv = ["su", "-s", "/bin/sh", user, "-c", f"cd {cwd} && {command}"]
    else:
        argv = ["/bin/sh", "-c", f"cd {cwd} && {command}"]
    return subprocess.run(argv, capture_output=True, text=True, timeout=180)


def _start_cluster() -> tuple[str, callable]:
    """initdb + start a cluster. Returns (dsn, stop_callable).

    Raises RuntimeError with the captured stderr when anything fails, so the
    skip message says what actually went wrong instead of "postgres missing".
    """
    bindir = _bindir()
    if not bindir:
        raise RuntimeError(
            "no PostgreSQL server binaries found (looked for initdb/pg_ctl under "
            "/usr/lib/postgresql/*/bin and on PATH); set ANYRAG_PG_BINDIR"
        )

    user = _pick_unprivileged_user()
    if os.geteuid() == 0 and not user:
        raise RuntimeError(
            "running as root and no unprivileged account is available to own "
            "the cluster; PostgreSQL refuses to start as root"
        )

    # Deliberately not pytest's tmp_path: /tmp/pytest-of-root is mode 0700 and
    # the unprivileged user cannot traverse into it, so initdb fails with a
    # confusing "could not access directory". /var/tmp is world-traversable.
    base = tempfile.mkdtemp(prefix="anyrag-pg-", dir="/var/tmp")
    datadir = os.path.join(base, "data")
    logfile = os.path.join(base, "server.log")
    port = _free_port()

    def cleanup() -> None:
        _run(
            f"{bindir}/pg_ctl -D {datadir} -m immediate stop",
            user=user,
            cwd="/var/tmp",
        )
        shutil.rmtree(base, ignore_errors=True)

    try:
        if user:
            shutil.chown(base, user=user)
        os.chmod(base, 0o700)

        proc = _run(
            f"{bindir}/initdb -D {datadir} -U postgres -A trust "
            "--encoding=UTF8 --locale=C",
            user=user,
            cwd=base,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"initdb failed: {proc.stderr.strip() or proc.stdout.strip()}")

        proc = _run(
            f"{bindir}/pg_ctl -D {datadir} "
            f'-o "-k {base} -h 127.0.0.1 -p {port}" '
            f"-l {logfile} -w -t 30 start",
            user=user,
            cwd=base,
        )
        if proc.returncode != 0:
            log = Path(logfile).read_text() if Path(logfile).exists() else ""
            raise RuntimeError(
                f"pg_ctl start failed: {proc.stderr.strip() or proc.stdout.strip()}\n{log}"
            )

        admin = f"postgresql://postgres@127.0.0.1:{port}/postgres"
        with psycopg.connect(admin, autocommit=True, connect_timeout=10) as conn:
            conn.execute(f"CREATE DATABASE {TEST_DB}")
        return f"postgresql://postgres@127.0.0.1:{port}/{TEST_DB}", cleanup
    except Exception:
        cleanup()
        raise


@pytest.fixture(scope="session")
def pg_dsn():
    """A DSN for a live Postgres carrying the fixture data, or skip."""
    external = os.environ.get(DEFAULT_DSN_ENV, "").strip()
    stop = None
    if external:
        dsn = external
    else:
        try:
            dsn, stop = _start_cluster()
        except Exception as exc:
            pytest.skip(
                "no PostgreSQL available for integration tests: "
                f"{exc}. Set {DEFAULT_DSN_ENV} to test against an existing server."
            )
    try:
        with psycopg.connect(dsn, autocommit=False, connect_timeout=10) as conn:
            populate_postgres(conn)
        yield dsn
    finally:
        if stop is not None:
            stop()


@pytest.fixture(scope="session")
def src(pg_dsn):
    source = PostgresSource(pg_dsn)
    yield source
    source.close()


# --------------------------------------------------------------------------
# DSN resolution -- pure, no server needed
# --------------------------------------------------------------------------


def test_adapter_is_discovered_by_scheme_without_any_registry_edit():
    assert available_sources()["postgres"] is PostgresSource


def test_resolve_dsn_reads_the_named_environment_variable(monkeypatch):
    monkeypatch.setenv("SOME_PG_DSN", "postgresql://u@h/db")
    assert resolve_dsn("$SOME_PG_DSN") == "postgresql://u@h/db"


def test_resolve_dsn_defaults_to_anyrag_pg_dsn(monkeypatch):
    monkeypatch.setenv(DEFAULT_DSN_ENV, "postgresql://u@h/db")
    assert resolve_dsn("") == "postgresql://u@h/db"
    assert resolve_dsn("$") == "postgresql://u@h/db"


def test_resolve_dsn_rebuilds_a_uri_split_by_the_registry():
    """`postgres://host/db` reaches the adapter as `//host/db`."""
    assert resolve_dsn("//user@host:5432/db") == "postgresql://user@host:5432/db"


def test_resolve_dsn_passes_a_libpq_string_through():
    assert resolve_dsn("host=h dbname=d") == "host=h dbname=d"


def test_resolve_dsn_errors_when_the_variable_is_unset(monkeypatch):
    monkeypatch.delenv(DEFAULT_DSN_ENV, raising=False)
    with pytest.raises(ConfigError):
        resolve_dsn("")
    with pytest.raises(ConfigError):
        resolve_dsn("$NO_SUCH_VARIABLE_ANYWHERE")


def test_no_credentials_are_committed_in_the_adapter():
    """Docstring examples are fine; a URL carrying `user:password@` is not."""
    import re

    text = Path(__file__).resolve().parents[1].joinpath(
        "anyrag/sources/postgres.py"
    ).read_text(encoding="utf-8")
    assert not re.search(r"postgres(?:ql)?://[^\s\"']*:[^\s\"'@/]*@", text)
    # And no hard-coded default host: the DSN always comes from the caller.
    assert "@localhost" not in text and "@127.0.0.1" not in text


# --------------------------------------------------------------------------
# Registration and conformance
# --------------------------------------------------------------------------


def test_open_source_builds_the_adapter_from_an_env_uri(pg_dsn, monkeypatch):
    monkeypatch.setenv(DEFAULT_DSN_ENV, pg_dsn)
    src = open_source(f"postgres:${DEFAULT_DSN_ENV}")
    try:
        assert isinstance(src, PostgresSource)
        assert isinstance(src, DataSource)
        assert src.source_id == f"postgres:{TEST_DB}"
    finally:
        src.close()


def test_bad_dsn_is_a_source_error():
    with pytest.raises(SourceError):
        PostgresSource("postgresql://nobody@127.0.0.1:1/none", connect_timeout=2)


# --------------------------------------------------------------------------
# Schema introspection
# --------------------------------------------------------------------------


def test_tables_lists_user_schemas_only(src):
    refs = src.tables()
    assert [t.name for t in refs] == ["customers", "events", "orders", "shipments"]
    assert {t.schema for t in refs} == {"public"}
    assert not any(t.schema in ("pg_catalog", "information_schema") for t in refs)


def test_schema_columns_types_and_nullability(src):
    schema = src.schema(CUSTOMERS)
    assert schema.column_names == (
        "customer_id",
        "full_name",
        "country",
        "signup_date",
        "is_active",
        "notes",
        "avatar",
    )
    assert schema.column("signup_date").type_name == "date"
    assert schema.column("is_active").type_name == "boolean"
    assert schema.column("full_name").nullable is False
    assert schema.column("country").nullable is True
    assert schema.column("customer_id").nullable is False


def test_composite_primary_key_keeps_declaration_order(src):
    schema = src.schema(ORDERS)
    assert schema.primary_key == ("order_id", "line_no")
    assert schema.column("order_id").is_primary_key is True
    assert schema.column("line_no").is_primary_key is True


def test_foreign_key_without_a_column_list_resolves_to_the_target_pk(src):
    assert src.schema(ORDERS).column("customer_id").references == "customers.customer_id"


def test_composite_foreign_key_pairs_columns_correctly(src):
    """The naive information_schema join mispairs these; pg_catalog does not."""
    schema = src.schema(SHIPMENTS)
    assert schema.column("order_id").references == "orders.order_id"
    assert schema.column("line_no").references == "orders.line_no"
    assert schema.column("carrier").references is None


def test_schema_accepts_qualified_and_bare_names(src):
    assert src.schema("public.customers") == src.schema(CUSTOMERS)
    assert src.schema("customers") == src.schema(CUSTOMERS)


def test_unknown_table_raises(src):
    with pytest.raises(SourceError):
        src.schema(TableRef("no_such_table", "public"))


# --------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------


def test_profile_row_count_is_exact(src):
    assert src.profile(CUSTOMERS).row_count == N_CUSTOMERS
    assert src.profile(ORDERS).row_count == N_CUSTOMERS * 2


def test_profile_counts_nulls_and_distincts(src):
    country = src.profile(CUSTOMERS).columns["country"]
    assert country.total_count == N_CUSTOMERS
    assert country.null_count == 10
    assert country.distinct_count == 5


@pytest.mark.parametrize("column,role", sorted(EXPECTED_CUSTOMER_ROLES.items()))
def test_customer_column_roles_match_sqlite(src, column, role):
    """Same logical data, native Postgres types -- the roles must not move."""
    assert src.profile(CUSTOMERS).role(column).value == role


@pytest.mark.parametrize("column,role", sorted(EXPECTED_ORDER_ROLES.items()))
def test_order_column_roles_match_sqlite(src, column, role):
    assert src.profile(ORDERS).role(column).value == role


def test_every_role_in_the_enum_is_exercised_by_the_fixture(src):
    seen = set()
    for table in src.tables():
        seen.update(p.role for p in src.profile(table).columns.values())
    assert seen == set(ColumnRole)


def test_categorical_columns_carry_their_vocabulary(src):
    country = src.profile(CUSTOMERS).columns["country"]
    assert country.role is ColumnRole.CATEGORICAL
    assert country.categories == ("AE", "EG", "JO", "MA", "SA")


def test_arabic_text_survives_the_round_trip(src):
    names = {r["full_name"] for b in src.iter_rows(CUSTOMERS, batch_size=500) for r in b}
    assert "أحمد السيد" in names
    assert "فاطمة الزهراء" in names
    assert {"Ahmad Al-Sayed", "Ahmed Al-Sayed", "Ahmad Al Sayed"} <= names


def test_large_table_is_sampled_rather_than_fully_scanned(pg_dsn):
    src = PostgresSource(pg_dsn, full_scan_max_rows=10, profile_sample_rows=25)
    try:
        prof = src.profile(CUSTOMERS)
        assert prof.row_count == N_CUSTOMERS
        assert prof.columns["customer_id"].total_count == 25
    finally:
        src.close()


# --------------------------------------------------------------------------
# Row iteration
# --------------------------------------------------------------------------


def test_iter_rows_batches_and_covers_every_row(src):
    batches = list(src.iter_rows(CUSTOMERS, batch_size=50))
    assert [len(b) for b in batches] == [50, 50, 20]


def test_iter_rows_is_ordered_by_primary_key(src):
    ids = [r["customer_id"] for b in src.iter_rows(CUSTOMERS, batch_size=7) for r in b]
    assert ids == list(range(1, N_CUSTOMERS + 1))


def test_iter_rows_composite_pk_order_is_deterministic(src):
    keys = [
        (r["order_id"], r["line_no"])
        for b in src.iter_rows(ORDERS, batch_size=13)
        for r in b
    ]
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))


def test_iter_rows_includes_primary_key_values(src):
    first = next(iter(src.iter_rows(ORDERS, batch_size=5)))[0]
    assert src.row_pk(ORDERS, first) == "1000|1"


def test_table_without_a_primary_key_falls_back_to_all_columns(src):
    assert src.pk_columns(EVENTS) == ("kind", "at")
    rows = [r for b in src.iter_rows(EVENTS, batch_size=4) for r in b]
    assert len(rows) == 15
    assert src.row_pk(EVENTS, rows[0]).count("|") == 1


def test_abandoning_iter_rows_early_does_not_pin_the_session(src):
    it = src.iter_rows(CUSTOMERS, batch_size=5)
    assert len(next(it)) == 5
    it.close()  # consumer walked away mid-scan
    # The session is usable and not stuck idle-in-transaction.
    assert src.execute_readonly("SELECT COUNT(*) FROM customers").scalar == N_CUSTOMERS
    assert src._conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE


def test_pk_string_matches_across_backends(src):
    """A bigint 1000 and a SQLite INTEGER 1000 must produce the same RowRef."""
    assert src.row_pk(ORDERS, {"order_id": 1000, "line_no": 2}) == "1000|2"


# --------------------------------------------------------------------------
# execute_readonly
# --------------------------------------------------------------------------


def test_execute_readonly_runs_a_real_select(src):
    res = src.execute_readonly(
        "SELECT country, COUNT(*) AS n FROM customers "
        "WHERE country IS NOT NULL GROUP BY country ORDER BY country"
    )
    assert isinstance(res, QueryResult)
    assert res.columns == ("country", "n")
    assert [r[0] for r in res.rows] == ["AE", "EG", "JO", "MA", "SA"]
    assert sum(r[1] for r in res.rows) == 110


def test_execute_readonly_rejects_a_stacked_drop(src):
    with pytest.raises(UnsafeQueryError):
        src.execute_readonly("SELECT 1; DROP TABLE customers")
    assert src.execute_readonly("SELECT COUNT(*) FROM customers").scalar == N_CUSTOMERS


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers",
        "UPDATE customers SET country = 'ZZ'",
        "DROP TABLE customers",
        "INSERT INTO customers (customer_id) VALUES (9999)",
        "WITH x AS (DELETE FROM customers RETURNING *) SELECT * FROM x",
        "SELECT pg_sleep(30)",
        "SELECT pg_read_file('/etc/passwd')",
        "COPY customers TO '/tmp/leak.csv'",
        "SELECT $$evil$$",
    ],
)
def test_execute_readonly_rejects_writes_and_filesystem_access(src, sql):
    with pytest.raises(UnsafeQueryError):
        src.execute_readonly(sql)


def test_execute_readonly_truncates_at_max_rows(src):
    res = src.execute_readonly("SELECT customer_id FROM customers", max_rows=10)
    assert len(res) == 10 and res.truncated is True
    res = src.execute_readonly("SELECT customer_id FROM customers", max_rows=N_CUSTOMERS)
    assert len(res) == N_CUSTOMERS and res.truncated is False


def test_bad_sql_becomes_a_source_error_and_leaves_the_session_usable(src):
    with pytest.raises(SourceError):
        src.execute_readonly("SELECT nonexistent_column FROM customers")
    # The aborted transaction was rolled back, not left poisoned.
    assert src.execute_readonly("SELECT 1").scalar == 1


def test_session_is_read_only_at_the_server(src):
    """Belt and braces: even bypassing the linter, the server refuses writes."""
    conn = src._conn
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        with conn.cursor() as cur:
            cur.execute("DELETE FROM customers")
    conn.rollback()
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        with conn.cursor() as cur:
            cur.execute("CREATE TABLE sneaky (x integer)")
    conn.rollback()


def test_statement_timeout_is_configured(src):
    res = src.execute_readonly("SELECT current_setting('statement_timeout') AS t")
    assert res.scalar not in ("0", "", None)


def test_data_is_unchanged_after_the_workout(src):
    assert src.execute_readonly("SELECT COUNT(*) FROM customers").scalar == N_CUSTOMERS
    assert src.execute_readonly("SELECT COUNT(*) FROM orders").scalar == N_CUSTOMERS * 2


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def test_close_is_idempotent(pg_dsn):
    src = PostgresSource(pg_dsn)
    src.close()
    src.close()
    with pytest.raises(SourceError):
        src.execute_readonly("SELECT 1")


def test_context_manager_closes(pg_dsn):
    with PostgresSource(pg_dsn) as src:
        assert src.execute_readonly("SELECT 1").scalar == 1
    with pytest.raises(SourceError):
        src.execute_readonly("SELECT 1")
