"""The synthetic source must be reproducible and deliberately awkward.

If any of these fail, the gold answers are still internally consistent but the
harness has stopped testing what it claims to test -- a corpus with no nulls, no
confusable names and no Arabic would flatter every retriever equally.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from evals import gen_db
from evals.gen_db import DATE_MAX, DATE_MIN, SCHEMA, TABLE_ORDER, Manifest

from .conftest import TEST_SEED, TEST_SIZES

ARABIC = range(0x0600, 0x0700)


def _has_arabic(text: str) -> bool:
    return any(ord(ch) in ARABIC for ch in str(text))


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_same_seed_produces_byte_identical_manifests():
    a = gen_db.build_manifest(seed=11, sizes=TEST_SIZES)
    b = gen_db.build_manifest(seed=11, sizes=TEST_SIZES)
    assert a["meta"]["content_hash"] == b["meta"]["content_hash"]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_different_seeds_produce_different_data():
    a = gen_db.build_manifest(seed=11, sizes=TEST_SIZES)
    b = gen_db.build_manifest(seed=12, sizes=TEST_SIZES)
    assert a["meta"]["content_hash"] != b["meta"]["content_hash"]


def test_manifest_carries_no_wall_clock_timestamp():
    # A timestamp would make the content hash useless as a reproducibility proof.
    meta = gen_db.build_manifest(seed=3, sizes=TEST_SIZES)["meta"]
    assert not any("time" in k or "date" in k and k.startswith("created") for k in meta)
    assert "content_hash" in meta


# --------------------------------------------------------------------------
# The five deliberate awkwardnesses
# --------------------------------------------------------------------------


def test_nulls_are_present_at_varying_rates(manifest: Manifest):
    rates: dict[str, float] = {}
    for table in TABLE_ORDER:
        rows = manifest.rows(table)
        for col in manifest.columns(table):
            nulls = sum(1 for r in rows if r[col] is None)
            if nulls:
                rates[f"{table}.{col}"] = nulls / len(rows)
    assert len(rates) >= 5, f"expected nulls in several columns, got {rates}"
    # "Varying" means varying: a single uniform null rate would not exercise
    # aggregate null-skipping against lookup null-handling.
    assert max(rates.values()) - min(rates.values()) > 0.2, rates


def test_declared_nullability_matches_the_data(manifest: Manifest):
    for table in TABLE_ORDER:
        rows = manifest.rows(table)
        for col in SCHEMA[table]:
            if not col.nullable:
                assert all(r[col.name] is not None for r in rows), f"{table}.{col.name}"


def test_confusable_transliterations_exist_as_distinct_rows(manifest: Manifest):
    names = {c["full_name"] for c in manifest.rows("customers")}
    clusters = 0
    for first, alt_first, particle, last in gen_db.CONFUSABLE_BASES:
        variants = set(gen_db._confusable_names(first, alt_first, particle, last))
        present = variants & names
        if len(present) >= 2:
            clusters += 1
            assert len(present) == len({n.lower() for n in present})
    assert clusters >= 8, "too few confusable name clusters to make retrieval disagree"


def test_every_customer_name_is_unique(manifest: Manifest):
    # Entity-lookup gold is "the row with exactly this name"; a duplicate would
    # silently turn a one-row question into a two-row one.
    names = [c["full_name"] for c in manifest.rows("customers")]
    assert len(names) == len(set(names))


def test_arabic_and_english_free_text_are_both_populated(manifest: Manifest):
    customers = manifest.rows("customers")
    assert any(_has_arabic(c["full_name_ar"]) for c in customers)
    assert any(c["notes_ar"] and _has_arabic(c["notes_ar"]) for c in customers)
    assert any(c["notes"] and not _has_arabic(c["notes"]) for c in customers)
    products = manifest.rows("products")
    assert any(_has_arabic(p["name_ar"]) for p in products)
    assert any(p["description_ar"] and _has_arabic(p["description_ar"]) for p in products)


def test_dates_span_about_two_years(manifest: Manifest):
    dates = [o["order_date"] for o in manifest.rows("orders")]
    assert min(dates) >= DATE_MIN.isoformat()
    assert max(dates) <= DATE_MAX.isoformat()
    # Both halves of the span must be populated or date-range questions degenerate.
    assert any(d < "2023-07-01" for d in dates)
    assert any(d > "2024-07-01" for d in dates)


def test_foreign_keys_all_resolve(manifest: Manifest):
    for table in TABLE_ORDER:
        for col in SCHEMA[table]:
            if not col.references:
                continue
            ref_table, ref_col = col.references.split(".")
            targets = {str(r[ref_col]) for r in manifest.rows(ref_table)}
            for row in manifest.rows(table):
                assert str(row[col.name]) in targets, f"{table}.{col.name} -> {row[col.name]}"


def test_the_database_has_a_few_thousand_rows():
    manifest = gen_db.build_manifest(seed=TEST_SEED)  # default sizes
    assert 3000 <= manifest["meta"]["total_rows"] <= 20000
    assert len(manifest["meta"]["row_counts"]) == 5


# --------------------------------------------------------------------------
# The emitted SQLite file
# --------------------------------------------------------------------------


def test_sqlite_matches_the_manifest_row_for_row(conn, manifest: Manifest):
    for table in TABLE_ORDER:
        n = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        assert n == len(manifest.rows(table))


def test_sqlite_enforces_the_declared_foreign_keys(db_path):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        assert violations == []
    finally:
        conn.close()


def test_a_sampled_row_is_identical_in_both_representations(conn, manifest: Manifest):
    for table in TABLE_ORDER:
        pk = manifest.pk_col(table)
        row = manifest.rows(table)[0]
        db_row = conn.execute(
            f'SELECT * FROM "{table}" WHERE "{pk}" = ?', (row[pk],)
        ).fetchone()
        assert db_row is not None
        for col in manifest.columns(table):
            assert db_row[col] == pytest.approx(row[col]) if isinstance(
                row[col], float
            ) else db_row[col] == row[col]


def test_manifest_round_trips_through_json(tmp_path, manifest: Manifest):
    path = tmp_path / "m.json"
    gen_db.write_manifest(manifest.data, path)
    reloaded = Manifest.load(path)
    assert reloaded.meta["content_hash"] == manifest.meta["content_hash"]
    assert reloaded.rows("customers")[0] == manifest.rows("customers")[0]


def test_exports_are_readable_by_pyarrow(tmp_path, manifest: Manifest):
    pa_parquet = pytest.importorskip("pyarrow.parquet")
    gen_db.write_exports(manifest.data, tmp_path)
    for table in TABLE_ORDER:
        assert (tmp_path / f"{table}.csv").exists()
        arrow = pa_parquet.read_table(tmp_path / f"{table}.parquet")
        assert arrow.num_rows == len(manifest.rows(table))
