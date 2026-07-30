"""Integration tests for the SQLite source adapter against a real database."""

from __future__ import annotations

import sqlite3

import pytest

from anyrag.core.errors import SourceError, UnsafeQueryError
from anyrag.core.interfaces import DataSource
from anyrag.core.registry import available_sources, open_source
from anyrag.core.types import ColumnRole, QueryResult, TableRef
from anyrag.sources.sqlite import SQLiteSource
from test_sources_data import (
    EXPECTED_CUSTOMER_ROLES,
    EXPECTED_ORDER_ROLES,
    N_CUSTOMERS,
    build_sqlite_db,
)

CUSTOMERS = TableRef("customers", None)
ORDERS = TableRef("orders", None)
SHIPMENTS = TableRef("shipments", None)
EVENTS = TableRef("events", None)


@pytest.fixture(scope="module")
def db_path(tmp_path_factory) -> str:
    return build_sqlite_db(str(tmp_path_factory.mktemp("sqlite") / "fixture.sqlite"))


@pytest.fixture(scope="module")
def src(db_path):
    source = SQLiteSource(db_path)
    yield source
    source.close()


# --------------------------------------------------------------------------
# Registration and conformance
# --------------------------------------------------------------------------


def test_adapter_is_discovered_by_scheme_without_any_registry_edit():
    assert available_sources()["sqlite"] is SQLiteSource


def test_open_source_builds_the_adapter_from_a_uri(db_path):
    src = open_source(f"sqlite:{db_path}")
    try:
        assert isinstance(src, SQLiteSource)
        assert isinstance(src, DataSource)
        assert src.source_id.startswith("sqlite:")
    finally:
        src.close()


def test_missing_file_is_a_source_error(tmp_path):
    with pytest.raises(SourceError):
        SQLiteSource(str(tmp_path / "nope.sqlite"))


def test_empty_locator_is_a_source_error():
    with pytest.raises(SourceError):
        SQLiteSource("   ")


# --------------------------------------------------------------------------
# Schema introspection
# --------------------------------------------------------------------------


def test_tables_lists_user_tables_only(src):
    names = [t.name for t in src.tables()]
    assert names == ["customers", "events", "orders", "shipments"]
    assert all(t.schema is None for t in src.tables())
    assert not any(n.startswith("sqlite_") for n in names)


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
    assert schema.column("customer_id").type_name == "INTEGER"
    assert schema.column("notes").type_name == "TEXT"
    assert schema.column("full_name").nullable is False  # declared NOT NULL
    assert schema.column("country").nullable is True
    # A primary key is not nullable even though SQLite reports notnull=0.
    assert schema.column("customer_id").nullable is False


def test_single_column_primary_key(src):
    schema = src.schema(CUSTOMERS)
    assert schema.primary_key == ("customer_id",)
    assert schema.column("customer_id").is_primary_key is True
    assert schema.column("full_name").is_primary_key is False


def test_composite_primary_key_keeps_declaration_order(src):
    schema = src.schema(ORDERS)
    assert schema.primary_key == ("order_id", "line_no")
    assert schema.column("order_id").is_primary_key is True
    assert schema.column("line_no").is_primary_key is True


def test_foreign_key_without_a_column_list_resolves_to_the_target_pk(src):
    """`REFERENCES customers` names no column; it means the target's PK."""
    assert src.schema(ORDERS).column("customer_id").references == "customers.customer_id"


def test_composite_foreign_key_pairs_columns_correctly(src):
    schema = src.schema(SHIPMENTS)
    assert schema.column("order_id").references == "orders.order_id"
    assert schema.column("line_no").references == "orders.line_no"
    assert schema.column("carrier").references is None


def test_schema_accepts_a_plain_string(src):
    assert src.schema("customers") == src.schema(CUSTOMERS)


def test_unknown_table_raises(src):
    with pytest.raises(SourceError):
        src.schema(TableRef("no_such_table"))


# --------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------


def test_profile_row_count_is_exact(src):
    assert src.profile(CUSTOMERS).row_count == N_CUSTOMERS
    assert src.profile(ORDERS).row_count == N_CUSTOMERS * 2


def test_profile_counts_nulls_and_distincts(src):
    prof = src.profile(CUSTOMERS)
    country = prof.columns["country"]
    assert country.total_count == N_CUSTOMERS
    assert country.null_count == 10  # every 12th row of 120
    assert country.distinct_count == 5
    assert country.null_fraction == pytest.approx(10 / 120)

    pk = prof.columns["customer_id"]
    assert pk.null_count == 0
    assert pk.distinct_count == N_CUSTOMERS
    assert pk.cardinality_ratio == 1.0
    assert (pk.min_value, pk.max_value) == (1, N_CUSTOMERS)


@pytest.mark.parametrize("column,role", sorted(EXPECTED_CUSTOMER_ROLES.items()))
def test_customer_column_roles(src, column, role):
    assert src.profile(CUSTOMERS).role(column).value == role


@pytest.mark.parametrize("column,role", sorted(EXPECTED_ORDER_ROLES.items()))
def test_order_column_roles(src, column, role):
    assert src.profile(ORDERS).role(column).value == role


def test_every_role_in_the_enum_is_exercised_by_the_fixture(src):
    """If a role never appears, the fixture is not testing what it claims."""
    seen = set()
    for table in src.tables():
        prof = src.profile(table)
        seen.update(p.role for p in prof.columns.values())
    assert seen == set(ColumnRole)


def test_categorical_columns_carry_their_vocabulary(src):
    country = src.profile(CUSTOMERS).columns["country"]
    assert country.role is ColumnRole.CATEGORICAL
    assert country.categories == ("AE", "EG", "JO", "MA", "SA")

    status = src.profile(ORDERS).columns["status"]
    assert set(status.categories) == {"pending", "shipped", "delivered", "cancelled"}


def test_free_text_columns_do_not_carry_a_vocabulary(src):
    notes = src.profile(CUSTOMERS).columns["notes"]
    assert notes.role is ColumnRole.FREE_TEXT
    assert notes.categories == ()
    assert notes.mean_length is not None and notes.mean_length > 64


def test_samples_are_bounded_and_real(src):
    prof = src.profile(CUSTOMERS)
    names = prof.columns["full_name"]
    assert 0 < len(names.samples) <= 10
    assert all(isinstance(s, str) for s in names.samples)


def test_arabic_text_survives_profiling(src):
    samples = set()
    for batch in src.iter_rows(CUSTOMERS, batch_size=500):
        samples.update(r["full_name"] for r in batch)
    assert "أحمد السيد" in samples
    assert "فاطمة الزهراء" in samples


def test_near_duplicate_names_are_distinct_values(src):
    names = set()
    for batch in src.iter_rows(CUSTOMERS, batch_size=500):
        names.update(r["full_name"] for r in batch)
    assert {"Ahmad Al-Sayed", "Ahmed Al-Sayed", "Ahmad Al Sayed"} <= names


def test_large_table_is_sampled_rather_than_fully_scanned(db_path):
    """Profiling degrades to a bounded sample instead of loading everything."""
    src = SQLiteSource(db_path, full_scan_max_rows=10, profile_sample_rows=25)
    try:
        prof = src.profile(CUSTOMERS)
        assert prof.row_count == N_CUSTOMERS  # exact, from COUNT(*)
        # ...but per-column stats were computed over the sample only, and say so.
        assert prof.columns["customer_id"].total_count == 25
    finally:
        src.close()


# --------------------------------------------------------------------------
# Row iteration
# --------------------------------------------------------------------------


def test_iter_rows_batches_and_covers_every_row(src):
    batches = list(src.iter_rows(CUSTOMERS, batch_size=50))
    assert [len(b) for b in batches] == [50, 50, 20]
    assert sum(len(b) for b in batches) == N_CUSTOMERS


def test_iter_rows_is_ordered_by_primary_key(src):
    ids = [r["customer_id"] for b in src.iter_rows(CUSTOMERS, batch_size=7) for r in b]
    assert ids == sorted(ids)
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
    assert "order_id" in first and "line_no" in first
    assert src.row_pk(ORDERS, first) == "1000|1"


def test_iter_rows_repeats_identically(src):
    a = [r["customer_id"] for b in src.iter_rows(CUSTOMERS, batch_size=11) for r in b]
    b = [r["customer_id"] for b_ in src.iter_rows(CUSTOMERS, batch_size=97) for r in b_]
    assert a == b


def test_table_without_a_primary_key_uses_rowid_as_a_surrogate(src):
    assert src.pk_columns(EVENTS) == ("rowid",)
    batches = list(src.iter_rows(EVENTS, batch_size=4))
    rows = [r for b in batches for r in b]
    assert len(rows) == 15
    assert [r["rowid"] for r in rows] == list(range(1, 16))
    assert src.row_pk(EVENTS, rows[0]) == "1"


def test_iter_rows_rejects_a_nonpositive_batch(src):
    with pytest.raises(ValueError):
        next(src.iter_rows(CUSTOMERS, batch_size=0))


def test_pk_columns_and_row_pk(src):
    assert src.pk_columns(CUSTOMERS) == ("customer_id",)
    assert src.pk_columns(ORDERS) == ("order_id", "line_no")
    assert src.row_pk(CUSTOMERS, {"customer_id": 7}) == "7"
    assert src.row_pk(ORDERS, {"order_id": 1, "line_no": 2}) == "1|2"


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
    assert res.truncated is False


def test_execute_readonly_scalar(src):
    assert src.execute_readonly("SELECT COUNT(*) FROM customers").scalar == N_CUSTOMERS


def test_execute_readonly_rejects_a_stacked_drop(src):
    with pytest.raises(UnsafeQueryError):
        src.execute_readonly("SELECT 1; DROP TABLE customers")
    # ...and the table is still there.
    assert src.execute_readonly("SELECT COUNT(*) FROM customers").scalar == N_CUSTOMERS


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers",
        "UPDATE customers SET country = 'ZZ'",
        "DROP TABLE customers",
        "INSERT INTO customers (customer_id) VALUES (9999)",
        "PRAGMA writable_schema = ON",
        "ATTACH DATABASE '/tmp/evil.db' AS evil",
        "WITH x AS (DELETE FROM customers RETURNING *) SELECT * FROM x",
        "SELECT 1 -- \n; DROP TABLE customers",
        "",
    ],
)
def test_execute_readonly_rejects_writes(src, sql):
    with pytest.raises(UnsafeQueryError):
        src.execute_readonly(sql)


def test_execute_readonly_accepts_a_literal_that_merely_looks_dangerous(src):
    res = src.execute_readonly("SELECT 'please delete from customers' AS t")
    assert res.scalar == "please delete from customers"


def test_execute_readonly_truncates_at_max_rows(src):
    res = src.execute_readonly("SELECT customer_id FROM customers", max_rows=10)
    assert len(res) == 10
    assert res.truncated is True

    res = src.execute_readonly("SELECT customer_id FROM customers", max_rows=N_CUSTOMERS)
    assert len(res) == N_CUSTOMERS
    assert res.truncated is False


def test_execute_readonly_reports_the_linted_sql(src):
    res = src.execute_readonly("SELECT 1 AS one;")
    assert res.sql == "SELECT 1 AS one"


def test_bad_sql_becomes_a_source_error_not_a_driver_error(src):
    with pytest.raises(SourceError):
        src.execute_readonly("SELECT nonexistent_column FROM customers")


def test_connection_itself_is_read_only(src):
    """Belt and braces: even bypassing the linter, the driver refuses writes."""
    with pytest.raises(sqlite3.OperationalError):
        src._conn.execute("CREATE TABLE sneaky (x INTEGER)")
    with pytest.raises(sqlite3.OperationalError):
        src._conn.execute("DELETE FROM customers")


def test_source_file_is_unchanged_after_a_full_workout(db_path):
    import hashlib
    from pathlib import Path

    before = hashlib.sha256(Path(db_path).read_bytes()).hexdigest()
    src = SQLiteSource(db_path)
    try:
        src.tables()
        src.schema(CUSTOMERS)
        src.profile(ORDERS)
        list(src.iter_rows(CUSTOMERS, batch_size=10))
        src.execute_readonly("SELECT * FROM customers")
        with pytest.raises(UnsafeQueryError):
            src.execute_readonly("DELETE FROM customers")
    finally:
        src.close()
    assert hashlib.sha256(Path(db_path).read_bytes()).hexdigest() == before


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def test_close_is_idempotent(db_path):
    src = SQLiteSource(db_path)
    src.close()
    src.close()
    src.close()
    with pytest.raises(SourceError):
        src.execute_readonly("SELECT 1")


def test_context_manager_closes(db_path):
    with SQLiteSource(db_path) as src:
        assert src.execute_readonly("SELECT 1").scalar == 1
    with pytest.raises(SourceError):
        src.execute_readonly("SELECT 1")
