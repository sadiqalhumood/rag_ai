"""Integration tests for the flat-file source adapter.

The fixture directory deliberately mixes formats and quality: a CSV and a
Parquet file that must both introspect correctly, a ragged CSV that must be
skipped rather than take the source down, and a non-data file that must be
ignored silently.

The two data files carry the same shapes of column -- id, categorical, free
text, date, boolean, numeric, nulls, Arabic script -- through two very
different type systems (CSV declares nothing; Parquet declares everything).
Role inference reaching the same conclusions from both is the point.
"""

from __future__ import annotations

import datetime as dt
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from anyrag.core.errors import SourceError, UnsafeQueryError
from anyrag.core.interfaces import DataSource
from anyrag.core.registry import available_sources, open_source
from anyrag.core.types import ColumnRole, QueryResult, RowRef, TableRef
from anyrag.sources.files import (
    ROW_ORDINAL,
    FilesSource,
    arrow_column_spec,
)

PEOPLE = TableRef("people", None)
ORDERS = TableRef("orders", None)

N_PEOPLE = 60
N_ORDERS = 40

#: Near-duplicate spellings plus Arabic script, so `name` is a vocabulary
#: rather than a category set.
NAMES = [
    "Ahmad Al-Sayed",
    "Ahmed Al-Sayed",
    "Ahmad Al Sayed",
    "Fatima Al-Zahra",
    "Fatimah Al-Zahra",
    "أحمد السيد",
    "فاطمة الزهراء",
    "محمد عبد الله",
    "Mohammed Abdullah",
    "Mohamed Abdallah",
    "Layla Hassan",
    "Laila Hasan",
    "Youssef Ibrahim",
    "Yusuf Ibrahim",
    "Nour El-Din",
    "Noor Eldin",
    "خالد منصور",
    "Khaled Mansour",
    "Sara Khalil",
    "Sarah Khaleel",
    "عمر الفاروق",
    "Omar Al-Farouq",
    "Hana Mahmoud",
    "Hanaa Mahmud",
]

COUNTRIES = ["EG", "SA", "AE", "MA", "JO"]
CUSTOMERS = ["acme", "globex", "شركة النور", "initech"]


# --------------------------------------------------------------------------
# Fixture construction
# --------------------------------------------------------------------------


def _write_people_csv(path) -> None:
    lines = [
        "person_id,name,country,signup_date,renewed_on,active,score,bio",
    ]
    for i in range(N_PEOPLE):
        name = NAMES[i % len(NAMES)]
        # Every 7th row drops its country, so `country` has real nulls.
        country = "" if i % 7 == 0 else COUNTRIES[i % len(COUNTRIES)]
        signup = "" if i % 11 == 0 else f"2024-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}"
        # Slash-formatted on purpose: pyarrow will NOT infer a date type, so
        # this column can only be classified by looking at the values.
        renewed = "" if i % 9 == 0 else f"2025/{(i % 12) + 1:02d}/{(i % 27) + 1:02d}"
        active = "yes" if i % 3 else "no"
        score = "" if i % 5 == 0 else f"{i * 1.5:.2f}"
        bio = (
            f"Row {i}: a deliberately long free-text biography that comfortably "
            f"exceeds the categorical mean-length ceiling and is unique per row."
        )
        lines.append(
            f"{i + 1},{name},{country},{signup},{renewed},{active},{score},{bio}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_orders_parquet(path) -> None:
    schema = pa.schema(
        [
            # Declared NOT NULL, so the adapter has a real nullability signal
            # to read -- unlike CSV, where it can only report observation.
            pa.field("order_id", pa.int64(), nullable=False),
            pa.field("customer", pa.string(), nullable=True),
            pa.field("amount", pa.float64(), nullable=True),
            pa.field("ordered_on", pa.date32(), nullable=True),
            pa.field("note", pa.string(), nullable=True),
            pa.field("flag", pa.bool_(), nullable=True),
            pa.field("payload", pa.binary(), nullable=True),
        ]
    )
    table = pa.table(
        {
            "order_id": [i + 1 for i in range(N_ORDERS)],
            "customer": [CUSTOMERS[i % len(CUSTOMERS)] for i in range(N_ORDERS)],
            "amount": [
                None if i % 6 == 0 else round(10.5 + i * 3.25, 2)
                for i in range(N_ORDERS)
            ],
            "ordered_on": [
                None if i % 8 == 0 else dt.date(2023, (i % 12) + 1, (i % 28) + 1)
                for i in range(N_ORDERS)
            ],
            "note": [
                f"طلب رقم {i}: ملاحظة نصية حرة وطويلة بما يكفي لتجاوز حد التصنيف."
                for i in range(N_ORDERS)
            ],
            "flag": [bool(i % 2) for i in range(N_ORDERS)],
            "payload": [bytes([i % 251, (i * 7) % 251]) for i in range(N_ORDERS)],
        },
        schema=schema,
    )
    pq.write_table(table, str(path))


def build_files_dir(root):
    """Materialise the fixture directory and return it."""
    _write_people_csv(root / "people.csv")
    _write_orders_parquet(root / "orders.parquet")
    # Ragged: row 2 has three fields where the header promises two.
    (root / "broken.csv").write_text("a,b\n1,2\n3,4,5\n", encoding="utf-8")
    # Not a data file at all; must be ignored without an error entry.
    (root / "notes.txt").write_text("just prose\n", encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def files_dir(tmp_path_factory):
    return build_files_dir(tmp_path_factory.mktemp("files"))


@pytest.fixture(scope="module")
def src(files_dir):
    source = FilesSource(str(files_dir))
    yield source
    source.close()


# --------------------------------------------------------------------------
# Registration and conformance
# --------------------------------------------------------------------------


def test_adapter_is_discovered_by_scheme_without_any_registry_edit():
    assert available_sources()["files"] is FilesSource


def test_open_source_builds_the_adapter_from_a_uri(files_dir):
    source = open_source(f"files:{files_dir}")
    try:
        assert isinstance(source, FilesSource)
        assert isinstance(source, DataSource)
        assert source.source_id.startswith("files:")
    finally:
        source.close()


def test_missing_directory_is_a_source_error(tmp_path):
    with pytest.raises(SourceError):
        FilesSource(str(tmp_path / "nope"))


def test_empty_locator_is_a_source_error():
    with pytest.raises(SourceError):
        FilesSource("   ")


# --------------------------------------------------------------------------
# Discovery: one table per file, bad files skipped
# --------------------------------------------------------------------------


def test_one_table_per_file_named_from_the_stem(src):
    assert src.tables() == [ORDERS, PEOPLE]


def test_unreadable_file_is_skipped_and_recorded_not_fatal(src):
    assert "broken.csv" in src.load_errors
    assert src.load_errors["broken.csv"]
    # The good tables loaded anyway -- one bad file is not a bad source.
    assert src.row_count(PEOPLE) == N_PEOPLE
    assert src.row_count(ORDERS) == N_ORDERS


def test_unsupported_extension_is_ignored_silently(src):
    assert "notes.txt" not in src.load_errors
    assert TableRef("notes", None) not in src.tables()


def test_unknown_table_is_a_source_error(src):
    with pytest.raises(SourceError):
        src.schema("does_not_exist")


def test_name_collision_between_formats_keeps_the_first_and_records_the_second(
    tmp_path,
):
    (tmp_path / "dupe.csv").write_text("a\n1\n", encoding="utf-8")
    pq.write_table(pa.table({"a": [1, 2]}), str(tmp_path / "dupe.parquet"))
    with FilesSource(str(tmp_path)) as source:
        assert source.tables() == [TableRef("dupe", None)]
        assert source.row_count("dupe") == 1  # the CSV won, sorted first
        assert "dupe.parquet" in source.load_errors


def test_a_single_file_locator_works(files_dir):
    with FilesSource(str(files_dir / "people.csv")) as source:
        assert source.tables() == [PEOPLE]
        assert source.row_count(PEOPLE) == N_PEOPLE


# --------------------------------------------------------------------------
# Schema introspection
# --------------------------------------------------------------------------


def test_csv_schema_columns_and_synthesised_primary_key(src):
    schema = src.schema(PEOPLE)
    assert schema.primary_key == (ROW_ORDINAL,)
    assert schema.column_names == (
        ROW_ORDINAL,
        "person_id",
        "name",
        "country",
        "signup_date",
        "renewed_on",
        "active",
        "score",
        "bio",
    )
    ordinal = schema.column(ROW_ORDINAL)
    assert ordinal.is_primary_key is True
    assert ordinal.nullable is False


def test_parquet_schema_columns_and_synthesised_primary_key(src):
    schema = src.schema(ORDERS)
    assert schema.primary_key == (ROW_ORDINAL,)
    assert schema.column_names == (
        ROW_ORDINAL,
        "order_id",
        "customer",
        "amount",
        "ordered_on",
        "note",
        "flag",
        "payload",
    )


def test_flat_files_declare_no_foreign_keys(src):
    for table in (PEOPLE, ORDERS):
        assert all(c.references is None for c in src.schema(table).columns)


def test_parquet_nullability_comes_from_the_declared_schema(src):
    schema = src.schema(ORDERS)
    assert schema.column("order_id").nullable is False
    assert schema.column("amount").nullable is True


def test_csv_nullability_reports_observed_nulls(src):
    schema = src.schema(PEOPLE)
    assert schema.column("person_id").nullable is False  # never empty
    assert schema.column("score").nullable is True  # every 5th row empty
    assert schema.column("country").nullable is True


def test_declared_type_names_are_normalisable(src):
    orders = src.schema(ORDERS)
    assert orders.column("order_id").type_name == "bigint"
    assert orders.column("amount").type_name == "double"
    assert orders.column("ordered_on").type_name == "date"
    assert orders.column("flag").type_name == "boolean"
    assert orders.column("payload").type_name == "blob"

    people = src.schema(PEOPLE)
    # CSV declares nothing; these come from pyarrow's value inference.
    assert people.column("person_id").type_name == "bigint"
    assert people.column("score").type_name == "double"
    assert people.column("renewed_on").type_name == "text"


def test_arrow_column_spec_maps_the_families_we_care_about():
    assert arrow_column_spec(pa.int32())[:2] == ("INTEGER", "bigint")
    assert arrow_column_spec(pa.float32())[:2] == ("REAL", "double")
    assert arrow_column_spec(pa.decimal128(9, 2))[:2] == ("REAL", "numeric")
    assert arrow_column_spec(pa.bool_())[:2] == ("INTEGER", "boolean")
    assert arrow_column_spec(pa.date32())[:2] == ("TEXT", "date")
    assert arrow_column_spec(pa.timestamp("us"))[:2] == ("TEXT", "timestamp")
    assert arrow_column_spec(pa.large_string())[:2] == ("TEXT", "text")
    assert arrow_column_spec(pa.binary())[:2] == ("BLOB", "blob")
    assert arrow_column_spec(pa.list_(pa.int64()))[:2] == ("TEXT", "json")
    # A dictionary-encoded column is classified by what it decodes to.
    assert arrow_column_spec(
        pa.dictionary(pa.int8(), pa.string())
    )[:2] == ("TEXT", "text")


# --------------------------------------------------------------------------
# Role inference
# --------------------------------------------------------------------------


EXPECTED_PEOPLE_ROLES = {
    ROW_ORDINAL: ColumnRole.ID,
    "person_id": ColumnRole.ID,
    "name": ColumnRole.FREE_TEXT,
    "country": ColumnRole.CATEGORICAL,
    "signup_date": ColumnRole.DATE,
    "renewed_on": ColumnRole.DATE,
    "active": ColumnRole.BOOLEAN,
    "score": ColumnRole.NUMERIC,
    "bio": ColumnRole.FREE_TEXT,
}

EXPECTED_ORDER_ROLES = {
    ROW_ORDINAL: ColumnRole.ID,
    "order_id": ColumnRole.ID,
    "customer": ColumnRole.CATEGORICAL,
    "amount": ColumnRole.NUMERIC,
    "ordered_on": ColumnRole.DATE,
    "note": ColumnRole.FREE_TEXT,
    "flag": ColumnRole.BOOLEAN,
    # Binary is genuinely undeterminable, and saying so is the right answer.
    "payload": ColumnRole.UNKNOWN,
}


@pytest.mark.parametrize("column,role", sorted(EXPECTED_PEOPLE_ROLES.items()))
def test_csv_role_inference(src, column, role):
    assert src.profile(PEOPLE).role(column) is role


@pytest.mark.parametrize("column,role", sorted(EXPECTED_ORDER_ROLES.items()))
def test_parquet_role_inference(src, column, role):
    assert src.profile(ORDERS).role(column) is role


def test_value_based_date_inference_carries_a_column_pyarrow_left_as_text(src):
    """`renewed_on` is `YYYY/MM/DD`, which pyarrow does not type as a date."""
    assert src.schema(PEOPLE).column("renewed_on").type_name == "text"
    assert src.profile(PEOPLE).role("renewed_on") is ColumnRole.DATE


def test_value_based_boolean_inference_handles_yes_no(src):
    assert src.schema(PEOPLE).column("active").type_name == "text"
    profile = src.profile(PEOPLE).columns["active"]
    assert profile.role is ColumnRole.BOOLEAN
    assert set(profile.categories) == {"yes", "no"}


# --------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------


def test_profile_counts_rows_and_nulls(src):
    profile = src.profile(PEOPLE)
    assert profile.row_count == N_PEOPLE
    country = profile.columns["country"]
    assert country.total_count == N_PEOPLE
    assert country.null_count == len([i for i in range(N_PEOPLE) if i % 7 == 0])
    assert 0.0 < country.null_fraction < 1.0


def test_categorical_vocabulary_is_captured(src):
    country = src.profile(PEOPLE).columns["country"]
    assert set(country.categories) == set(COUNTRIES)
    customer = src.profile(ORDERS).columns["customer"]
    assert set(customer.categories) == set(CUSTOMERS)


def test_free_text_columns_get_no_vocabulary(src):
    bio = src.profile(PEOPLE).columns["bio"]
    assert bio.categories == ()
    assert bio.distinct_count == N_PEOPLE
    assert bio.mean_length and bio.mean_length > 64


def test_numeric_min_max(src):
    score = src.profile(PEOPLE).columns["score"]
    assert score.min_value == pytest.approx(1.5)
    assert score.max_value == pytest.approx(88.5)


def test_dates_round_trip_as_sortable_iso_text(src):
    expected = sorted(
        dt.date(2023, (i % 12) + 1, (i % 28) + 1).isoformat()
        for i in range(N_ORDERS)
        if i % 8 != 0
    )
    ordered_on = src.profile(ORDERS).columns["ordered_on"]
    assert ordered_on.min_value == expected[0]
    assert ordered_on.max_value == expected[-1]
    # Lexicographic order over ISO text is chronological order.
    assert all(isinstance(v, str) for v in ordered_on.samples)


def test_arabic_text_survives_the_load(src):
    names = {row["name"] for batch in src.iter_rows(PEOPLE) for row in batch}
    assert "أحمد السيد" in names
    notes = {row["note"] for batch in src.iter_rows(ORDERS) for row in batch}
    assert any(n.startswith("طلب رقم") for n in notes)


# --------------------------------------------------------------------------
# Surrogate keys: determinism is the whole contract
# --------------------------------------------------------------------------


def _all_rows(source, table):
    return [row for batch in source.iter_rows(table, batch_size=13) for row in batch]


def test_row_ordinal_is_the_zero_based_position_within_the_file(src):
    rows = _all_rows(src, PEOPLE)
    assert [r[ROW_ORDINAL] for r in rows] == list(range(N_PEOPLE))
    # Ordinal tracks file order, so it lines up with the CSV's own id column.
    assert [r["person_id"] for r in rows] == list(range(1, N_PEOPLE + 1))


def test_pk_assignment_is_identical_across_two_independent_loads(files_dir):
    def fingerprint(table):
        with FilesSource(str(files_dir)) as source:
            return [
                (source.row_pk(table, row), row.get("person_id") or row.get("order_id"))
                for row in _all_rows(source, table)
            ]

    for table in (PEOPLE, ORDERS):
        first = fingerprint(table)
        second = fingerprint(table)
        assert first == second
        assert len(first) == len({pk for pk, _ in first})


def test_row_pk_is_a_string_usable_directly_as_a_rowref(src):
    row = _all_rows(src, ORDERS)[7]
    pk = src.row_pk(ORDERS, row)
    assert pk == "7"
    ref = RowRef(ORDERS.name, pk)
    assert str(ref) == "orders#7"


def test_pk_columns_is_the_synthesised_ordinal(src):
    assert src.pk_columns(PEOPLE) == (ROW_ORDINAL,)
    assert src.pk_columns(ORDERS) == (ROW_ORDINAL,)


def test_sampled_profile_is_deterministic_and_not_head_biased(files_dir):
    """Above the full-scan threshold the profile samples -- deterministically."""

    def sampled():
        with FilesSource(
            str(files_dir), full_scan_max_rows=10, profile_sample_rows=20
        ) as source:
            profile = source.profile(PEOPLE)
            assert profile.row_count == N_PEOPLE  # exact, even when sampling
            ordinals = profile.columns[ROW_ORDINAL]
            assert ordinals.total_count == 20  # examined, not total
            return ordinals.samples

    first = sampled()
    assert first == sampled()
    # `LIMIT n` over file order would have handed back 0..19 and biased every
    # statistic toward the head of the file.
    assert set(first) != set(range(len(first)))


def test_profile_is_deterministic_across_loads(files_dir):
    def roles():
        with FilesSource(str(files_dir)) as source:
            return {
                name: prof.role for name, prof in source.profile(PEOPLE).columns.items()
            }

    assert roles() == roles()


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------


def test_iter_rows_batches_at_the_requested_size(src):
    batches = list(src.iter_rows(PEOPLE, batch_size=7))
    assert [len(b) for b in batches] == [7] * (N_PEOPLE // 7) + [N_PEOPLE % 7]
    assert sum(len(b) for b in batches) == N_PEOPLE


def test_iter_rows_is_ordered_by_the_surrogate_key(src):
    ordinals = [r[ROW_ORDINAL] for r in _all_rows(src, ORDERS)]
    assert ordinals == sorted(ordinals) == list(range(N_ORDERS))


def test_iter_rows_includes_the_key_in_every_row(src):
    assert all(ROW_ORDINAL in row for row in _all_rows(src, PEOPLE))


def test_iter_rows_rejects_a_nonpositive_batch_size(src):
    with pytest.raises(ValueError):
        list(src.iter_rows(PEOPLE, batch_size=0))


# --------------------------------------------------------------------------
# execute_readonly
# --------------------------------------------------------------------------


def test_execute_readonly_accepts_a_real_select(src):
    result = src.execute_readonly('SELECT COUNT(*) AS n FROM "people"')
    assert isinstance(result, QueryResult)
    assert result.columns == ("n",)
    assert result.scalar == N_PEOPLE
    assert result.truncated is False


def test_execute_readonly_rejects_stacked_statements(src):
    with pytest.raises(UnsafeQueryError):
        src.execute_readonly("SELECT 1; DROP TABLE t")


@pytest.mark.parametrize(
    "sql",
    [
        'DELETE FROM "people"',
        'UPDATE "people" SET "name" = \'x\'',
        'INSERT INTO "people" ("name") VALUES (\'x\')',
        'DROP TABLE "people"',
        "PRAGMA query_only=OFF",
        'ATTACH DATABASE \'/tmp/x.db\' AS x',
    ],
)
def test_execute_readonly_rejects_writes_at_the_linter(src, sql):
    with pytest.raises(UnsafeQueryError):
        src.execute_readonly(sql)


def test_the_engine_refuses_writes_even_when_the_linter_is_bypassed(src):
    """Second lock: `query_only` plus an authorizer, not just the linter."""
    conn = src._conn
    assert conn is not None
    for sql in (
        'DELETE FROM "people"',
        'UPDATE "people" SET "name" = \'x\'',
        'CREATE TABLE "evil" ("a" INTEGER)',
        'DROP TABLE "orders"',
    ):
        with pytest.raises(sqlite3.Error):
            conn.execute(sql)
    # ...and the data is still there.
    assert src.execute_readonly('SELECT COUNT(*) FROM "people"').scalar == N_PEOPLE


def test_execute_readonly_truncates_and_flags_it(src):
    result = src.execute_readonly('SELECT * FROM "people"', max_rows=5)
    assert len(result) == 5
    assert result.truncated is True


def test_execute_readonly_surfaces_engine_errors_as_source_errors(src):
    with pytest.raises(SourceError):
        src.execute_readonly('SELECT * FROM "no_such_table"')


def test_execute_readonly_can_join_across_two_files(src):
    """The point of loading into one database: CSV and Parquet in one query."""
    result = src.execute_readonly(
        'SELECT p."name", o."customer" FROM "people" p '
        'JOIN "orders" o ON o."order_id" = p."person_id" '
        'ORDER BY p."person_id" LIMIT 3'
    )
    assert result.columns == ("name", "customer")
    assert len(result) == 3


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def test_close_is_idempotent_and_then_queries_fail(files_dir):
    source = FilesSource(str(files_dir))
    source.close()
    source.close()
    with pytest.raises(SourceError):
        source.execute_readonly('SELECT 1')


def test_context_manager_closes(files_dir):
    with FilesSource(str(files_dir)) as source:
        assert source.tables()
    with pytest.raises(SourceError):
        source.execute_readonly("SELECT 1")
