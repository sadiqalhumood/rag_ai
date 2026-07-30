"""Unit tests for the source-independent introspection and profiling logic.

These pin the *thresholds*. Downstream routing keys off ColumnRole, so a change
here is a behaviour change for the whole system, and it should have to break a
test to happen.
"""

from __future__ import annotations

import datetime as dt

import pytest

from anyrag.core.types import ColumnRole, ColumnSchema, TableRef, TableSchema
from anyrag.sources import introspect as I
from anyrag.sources import profile as P


# --------------------------------------------------------------------------
# Type normalisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "declared,family",
    [
        ("INTEGER", I.INTEGER),
        ("int8", I.INTEGER),
        ("BIGSERIAL", I.INTEGER),
        ("SMALLINT", I.INTEGER),
        ("REAL", I.FLOAT),
        ("double precision", I.FLOAT),
        ("NUMERIC(10, 2)", I.NUMERIC),
        ("numeric", I.NUMERIC),
        ("money", I.NUMERIC),
        ("VARCHAR(255)", I.TEXT),
        ("character varying", I.TEXT),
        ("TEXT", I.TEXT),
        ("BOOLEAN", I.BOOLEAN),
        ("bool", I.BOOLEAN),
        ("DATE", I.DATE),
        ("timestamp without time zone", I.TIMESTAMP),
        ("timestamptz", I.TIMESTAMP),
        ("DATETIME", I.TIMESTAMP),
        ("time without time zone", I.TIME),
        ("BLOB", I.BLOB),
        ("bytea", I.BLOB),
        ("jsonb", I.JSON),
        ("uuid", I.UUID),
        ("text[]", I.TEXT),
        ("", I.UNKNOWN),
        (None, I.UNKNOWN),
        ("wibble", I.UNKNOWN),
    ],
)
def test_normalize_type(declared, family):
    assert I.normalize_type(declared) == family


def test_point_is_not_mistaken_for_an_integer():
    """`point` contains the substring `int`; the exact-match table wins."""
    assert I.normalize_type("point") == I.OTHER


def test_timestamp_wins_over_time():
    assert I.normalize_type("timestamp") == I.TIMESTAMP


@pytest.mark.parametrize(
    "values,family",
    [
        ([1, 2, 3], I.INTEGER),
        ([1.5, 2.5], I.FLOAT),
        ([1, 2.5], I.FLOAT),
        (["a", "b"], I.TEXT),
        ([True, False], I.BOOLEAN),
        ([b"\x00"], I.BLOB),
        ([dt.date(2024, 1, 1)], I.DATE),
        ([dt.datetime(2024, 1, 1)], I.TIMESTAMP),
        ([], I.UNKNOWN),
        ([None, None], I.UNKNOWN),
        ([1, "a"], I.UNKNOWN),
    ],
)
def test_family_from_values(values, family):
    assert I.family_from_values(values) == family


# --------------------------------------------------------------------------
# Date sniffing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "2024-01-31",
        "2024-01-31T14:05",
        "2024-01-31 14:05:09",
        "2024-01-31T14:05:09.123Z",
        "2024-01-31T14:05:09+02:00",
        "2024/01/31",
        "20240131",
        dt.date(2024, 1, 31),
        dt.datetime(2024, 1, 31, 14, 5),
    ],
)
def test_looks_like_date_accepts(value):
    assert I.looks_like_date(value) is True


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "   ",
        "not a date",
        "2024-13-01",  # month 13
        "2024-02-31",  # day out of range
        "1.2.3",
        "12345",
        "99999999",
        3,
        True,
        "31-01-2024",  # day-first is ambiguous; we refuse rather than guess
    ],
)
def test_looks_like_date_rejects(value):
    assert I.looks_like_date(value) is False


def test_date_parse_fraction():
    assert I.date_parse_fraction([]) == 0.0
    assert I.date_parse_fraction([None, None]) == 0.0
    assert I.date_parse_fraction(["2024-01-01", "nope"]) == 0.5
    # Nulls are excluded from the denominator, not counted as failures.
    assert I.date_parse_fraction(["2024-01-01", None]) == 1.0


# --------------------------------------------------------------------------
# Identifiers and keys
# --------------------------------------------------------------------------


def test_quote_ident_doubles_embedded_quotes():
    assert I.quote_ident("plain") == '"plain"'
    assert I.quote_ident('we"ird') == '"we""ird"'
    # The classic injection attempt becomes an (absurd) single identifier.
    assert I.quote_ident('t"; DROP TABLE t; --') == '"t""; DROP TABLE t; --"'


def test_quote_ident_rejects_nul():
    with pytest.raises(ValueError):
        I.quote_ident("a\x00b")


def test_qualified_ident():
    assert I.qualified_ident(TableRef("orders")) == '"orders"'
    assert I.qualified_ident(TableRef("orders", "public")) == '"public"."orders"'


def test_coerce_table_ref():
    assert I.coerce_table_ref("orders") == TableRef("orders", None)
    assert I.coerce_table_ref("sales.orders") == TableRef("orders", "sales")
    assert I.coerce_table_ref("orders", default_schema="public") == TableRef(
        "orders", "public"
    )
    ref = TableRef("orders", "sales")
    assert I.coerce_table_ref(ref, default_schema="public") is ref


def test_pk_string_is_deterministic_and_stringly_typed():
    assert I.pk_string([7]) == "7"
    assert I.pk_string(["7"]) == "7"
    assert I.pk_string([1000, 2]) == "1000|2"
    assert I.pk_string([]) == ""


def test_row_pk_string():
    row = {"order_id": 1000, "line_no": 2, "status": "shipped"}
    assert I.row_pk_string(row, ("order_id", "line_no")) == "1000|2"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("id", True),
        ("customer_id", True),
        ("customerId", True),
        ("ID", True),
        ("paid", False),
        ("valid", False),
        ("identity", False),
        ("idle", False),
        ("", False),
    ],
)
def test_looks_like_id_name(name, expected):
    assert I.looks_like_id_name(name) is expected


# --------------------------------------------------------------------------
# Role inference
# --------------------------------------------------------------------------


def _role(**kw) -> ColumnRole:
    """infer_role with sane defaults, so each test states only what it means."""
    values = kw.pop("values", ())
    base = dict(
        name="col",
        type_name="TEXT",
        values=values,
        total_count=kw.pop("total_count", len(values)),
        null_count=0,
        distinct_count=kw.pop("distinct_count", len(set(map(str, values)))),
        mean_length=None,
    )
    base.update(kw)
    return P.infer_role(**base)


def test_primary_key_is_always_id():
    assert _role(name="anything", type_name="TEXT", is_primary_key=True) is ColumnRole.ID


def test_near_unique_id_named_integer_is_id():
    n = 200
    assert (
        _role(
            name="customer_id",
            type_name="INTEGER",
            total_count=n,
            distinct_count=n,
            values=[1, 2, 3],
        )
        is ColumnRole.ID
    )


def test_id_cardinality_threshold_is_pinned_at_095():
    n = 100
    at = _role(
        name="customer_id",
        type_name="INTEGER",
        total_count=n,
        distinct_count=95,
        values=[1, 2],
    )
    below = _role(
        name="customer_id",
        type_name="INTEGER",
        total_count=n,
        distinct_count=94,
        values=[1, 2],
    )
    assert P.ID_MIN_CARDINALITY_RATIO == 0.95
    assert at is ColumnRole.ID
    assert below is not ColumnRole.ID


def test_repeating_foreign_key_is_id_even_though_it_is_not_unique():
    """A declared FK is a key, not a measurement -- averaging it is nonsense."""
    role = _role(
        name="customer_id",
        type_name="INTEGER",
        total_count=1000,
        distinct_count=40,
        references="customers.customer_id",
        values=[1, 2, 3],
    )
    assert role is ColumnRole.ID


def test_id_named_column_without_fk_and_with_low_cardinality_is_not_id():
    role = _role(
        name="customer_id",
        type_name="INTEGER",
        total_count=1000,
        distinct_count=40,
        values=[1, 2, 3],
    )
    assert role is ColumnRole.NUMERIC


def test_declared_boolean_is_boolean():
    assert _role(name="flag", type_name="BOOLEAN") is ColumnRole.BOOLEAN


@pytest.mark.parametrize(
    "pair", [(0, 1), ("true", "false"), ("YES", "no"), ("t", "F"), ("y", "n")]
)
def test_two_valued_column_is_boolean(pair):
    role = _role(
        name="is_active",
        type_name="INTEGER",
        values=list(pair) * 5,
        total_count=10,
        distinct_count=2,
    )
    assert role is ColumnRole.BOOLEAN


def test_three_values_is_not_boolean():
    role = _role(
        name="is_active",
        type_name="INTEGER",
        values=[0, 1, 2] * 4,
        total_count=12,
        distinct_count=3,
    )
    assert role is ColumnRole.NUMERIC


def test_two_non_boolean_values_is_not_boolean():
    role = _role(
        name="tier",
        type_name="TEXT",
        values=["gold", "silver"] * 10,
        total_count=20,
        distinct_count=2,
    )
    assert role is ColumnRole.CATEGORICAL


def test_declared_date_and_timestamp_are_date():
    assert _role(name="d", type_name="DATE") is ColumnRole.DATE
    assert _role(name="d", type_name="timestamptz") is ColumnRole.DATE


def test_time_alone_is_not_a_date():
    assert _role(name="t", type_name="time without time zone") is not ColumnRole.DATE


def test_text_column_of_iso_dates_is_date():
    values = [f"2024-01-{d:02d}" for d in range(1, 21)]
    assert (
        _role(name="signup", type_name="TEXT", values=values, distinct_count=20)
        is ColumnRole.DATE
    )


def test_date_parse_fraction_threshold_is_pinned_at_08():
    assert P.DATE_MIN_PARSE_FRACTION == 0.8
    ok = [f"2024-01-{d:02d}" for d in range(1, 9)] + ["x", "y"]  # 8/10
    bad = [f"2024-01-{d:02d}" for d in range(1, 8)] + ["x", "y", "z"]  # 7/10
    assert _role(name="d", type_name="TEXT", values=ok, distinct_count=10) is ColumnRole.DATE
    assert _role(name="d", type_name="TEXT", values=bad, distinct_count=10) is not ColumnRole.DATE


def test_numeric_declared_type_falls_through_to_numeric():
    assert _role(name="amount", type_name="REAL", total_count=100, distinct_count=90) is (
        ColumnRole.NUMERIC
    )


def test_categorical_ratio_threshold():
    assert P.CATEGORICAL_MAX_RATIO == 0.2
    # 200 rows -> limit 40 distinct.
    assert P.is_categorical(distinct_count=40, non_null=200, mean_length=5.0) is True
    assert P.is_categorical(distinct_count=41, non_null=200, mean_length=5.0) is False


def test_categorical_absolute_cap():
    assert P.CATEGORICAL_MAX_DISTINCT == 50
    # 10_000 rows: 0.2 * 10_000 = 2000, but the absolute cap wins.
    assert P.is_categorical(distinct_count=50, non_null=10_000, mean_length=5.0) is True
    assert P.is_categorical(distinct_count=51, non_null=10_000, mean_length=5.0) is False


def test_categorical_small_table_floor():
    """Without a floor, 0.2 * 12 rows = 2 and small tables have no categories."""
    assert P.CATEGORICAL_SMALL_TABLE_FLOOR == 8
    assert P.is_categorical(distinct_count=4, non_null=12, mean_length=5.0) is True
    assert P.is_categorical(distinct_count=8, non_null=12, mean_length=5.0) is True
    assert P.is_categorical(distinct_count=9, non_null=12, mean_length=5.0) is False


def test_all_values_unique_is_never_categorical():
    assert P.is_categorical(distinct_count=8, non_null=8, mean_length=3.0) is False


def test_long_values_are_never_categorical():
    assert P.CATEGORICAL_MAX_MEAN_LENGTH == 64
    assert P.is_categorical(distinct_count=3, non_null=100, mean_length=64.0) is True
    assert P.is_categorical(distinct_count=3, non_null=100, mean_length=64.1) is False


def test_repeated_long_paragraphs_are_free_text_not_categorical():
    para = "x" * 400
    role = _role(
        name="notes",
        type_name="TEXT",
        values=[para, para + "y"] * 50,
        total_count=100,
        distinct_count=2,
        mean_length=400.0,
    )
    assert role is ColumnRole.FREE_TEXT


def test_high_cardinality_text_is_free_text():
    values = [f"name-{i}" for i in range(100)]
    role = _role(name="full_name", type_name="TEXT", values=values, total_count=100, distinct_count=100)
    assert role is ColumnRole.FREE_TEXT


def test_blob_is_unknown():
    assert _role(name="avatar", type_name="BLOB", values=[b"\x00"]) is ColumnRole.UNKNOWN


def test_untyped_empty_column_is_unknown():
    assert _role(name="mystery", type_name=None, values=[], total_count=0) is (
        ColumnRole.UNKNOWN
    )


def test_untyped_column_infers_family_from_values():
    """SQLite lets a column be declared with no type at all."""
    assert (
        _role(name="n", type_name="", values=[1, 2, 3], total_count=100, distinct_count=90)
        is ColumnRole.NUMERIC
    )


def test_all_null_text_column_is_free_text_not_a_crash():
    role = _role(
        name="notes", type_name="TEXT", values=[], total_count=50, null_count=50,
        distinct_count=0,
    )
    assert role is ColumnRole.FREE_TEXT


# --------------------------------------------------------------------------
# Accumulator / profile assembly
# --------------------------------------------------------------------------


def test_accumulator_counts_and_samples():
    acc = P.ColumnAccumulator("c")
    for v in [1, 2, 2, None, 3]:
        acc.add(v)
    assert acc.total == 5
    assert acc.nulls == 1
    assert acc.non_null == 4
    assert acc.distinct_count == 3
    assert acc.min_max() == (1, 3)
    assert acc.samples == (1, 2, 3)  # distinct, in encounter order


def test_accumulator_sample_bound():
    acc = P.ColumnAccumulator("c")
    for v in range(1000):
        acc.add(v)
    assert len(acc.samples) == P.MAX_SAMPLE_VALUES


def test_accumulator_survives_mixed_types():
    acc = P.ColumnAccumulator("c")
    for v in [1, "two", 3.0]:
        acc.add(v)
    # Incomparable values: a missing range beats an exception.
    assert acc.min_max() == (None, None)
    assert acc.distinct_count == 3


def test_accumulator_survives_unhashable_values():
    acc = P.ColumnAccumulator("c")
    acc.add({"a": 1})
    acc.add([1, 2])
    assert acc.distinct_count == 2


def test_categories_are_bounded():
    acc = P.ColumnAccumulator("c")
    for i in range(500):
        acc.add(f"v{i}")
    assert len(acc.categories()) <= P.MAX_CATEGORIES


def test_build_table_profile_end_to_end():
    schema = TableSchema(
        table=TableRef("t"),
        columns=(
            ColumnSchema("id", "INTEGER", nullable=False, is_primary_key=True),
            ColumnSchema("status", "TEXT"),
            ColumnSchema("absent", "TEXT"),
        ),
        primary_key=("id",),
    )
    rows = [{"id": i, "status": ["a", "b"][i % 2]} for i in range(20)]
    prof = P.build_table_profile(
        table=TableRef("t"), schema=schema, row_count=20, rows=rows
    )
    assert prof.row_count == 20
    assert prof.role("id") is ColumnRole.ID
    assert prof.role("status") is ColumnRole.CATEGORICAL
    assert prof.columns["status"].categories == ("a", "b")
    # A column missing from every row is all-null, not absent from the profile.
    assert prof.columns["absent"].null_count == 20
    assert prof.columns["id"].cardinality_ratio == 1.0
    assert prof.columns["status"].null_fraction == 0.0
