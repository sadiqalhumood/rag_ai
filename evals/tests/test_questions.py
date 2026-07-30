"""Gold answers, verified a second time by an independent path.

`questions.py` computes gold in plain Python over `manifest.json`. These tests
recompute the same answers with SQL against the generated SQLite file. The two
paths share only the generator, so an error in the Python gold shows up here as
a disagreement -- which is the only reason to trust a self-grading harness.
"""

from __future__ import annotations

import sqlite3

import pytest

from anyrag.core.types import QueryRoute

from evals import questions as Q
from evals.gen_db import DATE_MAX, DATE_MIN, Manifest, build_manifest
from evals.questions import build_questions, type_counts

from .conftest import TEST_SEED

MIN_QUESTIONS = 300
MIN_TYPES = 6


@pytest.fixture(scope="module")
def _full_manifest():
    return Manifest(build_manifest(seed=TEST_SEED))


@pytest.fixture(scope="module")
def full_questions(_full_manifest):
    """The real, full-size question set -- the one the size floor applies to."""
    return build_questions(_full_manifest, "dev", TEST_SEED)


@pytest.fixture(scope="module")
def full_heldout(_full_manifest):
    return build_questions(_full_manifest, "heldout", TEST_SEED)


# --------------------------------------------------------------------------
# Shape of the set
# --------------------------------------------------------------------------


def test_at_least_three_hundred_questions_across_eight_types(full_questions):
    counts = type_counts(full_questions)
    assert len(full_questions) >= MIN_QUESTIONS, counts
    assert len(counts) >= MIN_TYPES
    assert set(counts) == set(Q.QTYPES), counts
    for qtype, n in counts.items():
        assert n >= 15, f"{qtype} has only {n} questions"


def test_question_ids_are_unique_and_namespaced(full_questions):
    qids = [q.qid for q in full_questions]
    assert len(qids) == len(set(qids))
    assert all(q.qid.startswith("dev-") for q in full_questions)


def test_no_question_is_asked_twice_within_a_set(full_questions, full_heldout):
    """A duplicate is not a second measurement, it is a double-weighted one."""
    for questions in (full_questions, full_heldout):
        texts = [q.text for q in questions]
        assert len(set(texts)) == len(texts), [
            t for t in texts if texts.count(t) > 1
        ][:5]


def test_building_twice_gives_identical_questions(manifest):
    a = build_questions(manifest, "dev", 7)
    b = build_questions(manifest, "dev", 7)
    assert [q.as_dict() for q in a] == [q.as_dict() for q in b]


def test_a_different_seed_gives_a_different_set(manifest):
    a = build_questions(manifest, "dev", 7)
    b = build_questions(manifest, "dev", 8)
    assert [q.text for q in a] != [q.text for q in b]


def test_every_answerable_question_is_gradeable_on_something(full_questions):
    for q in full_questions:
        if not q.answerable:
            continue
        assert (not q.gold.is_empty) or q.gold_scalar is not None, q.qid


def test_routes_are_assigned_per_type(full_questions):
    expected = {
        "entity_lookup": {QueryRoute.LOOKUP},
        "multi_filter_lookup": {QueryRoute.LOOKUP},
        "count_by_category": {QueryRoute.AGGREGATE},
        "numeric_aggregate": {QueryRoute.AGGREGATE},
        "date_range_aggregate": {QueryRoute.AGGREGATE},
        "join_relationship": {QueryRoute.HYBRID},
        "schema_question": {QueryRoute.LOOKUP},
    }
    for qtype, allowed in expected.items():
        got = {q.route for q in full_questions if q.qtype == qtype}
        assert got <= allowed, (qtype, got)


# --------------------------------------------------------------------------
# Held-out set is deliberately absent
# --------------------------------------------------------------------------


def test_heldout_set_is_written_and_comparable_in_size(full_heldout):
    assert Q.HELDOUT_AVAILABLE is True
    counts = type_counts(full_heldout)
    assert len(full_heldout) >= MIN_QUESTIONS, counts
    assert set(counts) == set(Q.QTYPES), counts
    for qtype, n in counts.items():
        assert n >= 15, f"{qtype} has only {n} held-out questions"


def test_all_resolves_to_both_sets():
    assert Q.resolve_sets("all") == ("dev", "heldout")


def test_dev_and_heldout_template_keys_are_disjoint():
    dev_keys = {t.key for t in Q.DEV_TEMPLATES}
    heldout_keys = {t.key for t in Q.HELDOUT_TEMPLATES}
    assert dev_keys & heldout_keys == set()
    assert len(dev_keys) == len(Q.DEV_TEMPLATES)
    assert len(heldout_keys) == len(Q.HELDOUT_TEMPLATES)


def test_no_question_text_appears_in_both_sets(full_questions, full_heldout):
    """Disjoint means disjoint. A shared phrasing is leakage by definition."""
    dev_texts = {q.text for q in full_questions}
    heldout_texts = {q.text for q in full_heldout}
    assert dev_texts & heldout_texts == set()


def test_heldout_phrasings_are_not_dev_phrasings_with_the_words_moved(
    full_questions, full_heldout
):
    """A held-out set of near-duplicates would measure nothing.

    Compared on token *multisets* per question type: if held-out questions were
    dev questions reordered, the bag of words would be near-identical.
    """
    import re as _re

    def shapes(questions, qtype):
        out = set()
        for q in questions:
            if q.qtype != qtype:
                continue
            # Strip the interpolated entity, keeping the sentence frame.
            frame = _re.sub(r"[A-Z][\w'-]*|\d[\d.-]*|'[^']*'", "~", q.text)
            out.add(" ".join(frame.split()))
        return out

    for qtype in Q.QTYPES:
        dev_frames = shapes(full_questions, qtype)
        heldout_frames = shapes(full_heldout, qtype)
        assert dev_frames and heldout_frames
        overlap = dev_frames & heldout_frames
        assert not overlap, f"{qtype} reuses dev sentence frames: {overlap}"


def test_qids_are_namespaced_by_template_set(full_heldout):
    assert all(q.qid.startswith("heldout-") for q in full_heldout)
    assert all(q.template_set == "heldout" for q in full_heldout)


def test_building_both_sets_together_keeps_qids_unique(manifest):
    combined = build_questions(manifest, "all", TEST_SEED)
    qids = [q.qid for q in combined]
    assert len(qids) == len(set(qids))
    assert {q.template_set for q in combined} == {"dev", "heldout"}


def test_a_qid_means_the_same_question_however_it_was_built(manifest):
    """A qid in a results file must identify a question on its own.

    With a counter shared across template sets, `heldout-distractor-0001` is a
    different question in a heldout-only run than in a `--templates all` run,
    and results files stop being joinable to the question set that produced
    them.
    """
    combined = {q.qid: q.text for q in build_questions(manifest, "all", TEST_SEED)}
    for name in ("dev", "heldout"):
        alone = build_questions(manifest, name, TEST_SEED)
        assert alone, name
        for q in alone:
            assert combined[q.qid] == q.text, q.qid


# --------------------------------------------------------------------------
# The router freeze (amendment 3)
# --------------------------------------------------------------------------


def test_the_router_freeze_is_verifiable():
    """Re-check the frozen files against the recorded blob hashes.

    The freeze is what makes the held-out number mean anything, so it is
    verified mechanically on every test run rather than trusted. If any of the
    three frozen files is edited after the freeze, this fails and the held-out
    run is invalidated -- which is the intended behaviour, not an inconvenience.

    The freeze commit was rewritten by a `--reset-author` rebase after it was
    communicated. Blob hashes are checked rather than the commit SHA precisely
    because they survive that kind of rewrite: they pin the *content* that was
    frozen, which is the thing that actually matters.
    """
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    for path, expected in Q.ROUTER_FREEZE["frozen_blobs"].items():
        target = repo / path
        assert target.exists(), path
        out = subprocess.run(
            ["git", "hash-object", str(target)],
            cwd=repo, capture_output=True, text=True, timeout=10,
        )
        assert out.returncode == 0, out.stderr
        actual = out.stdout.strip()
        assert actual == expected, (
            f"{path} has changed since the router freeze "
            f"({actual} != {expected}); the held-out run is invalidated "
            "and the router must be re-frozen"
        )


def test_the_freeze_records_the_superseded_commit():
    # A quietly-corrected hash is indistinguishable from one chosen after the
    # fact, so both are kept.
    assert Q.ROUTER_FREEZE["commit"] != Q.ROUTER_FREEZE["superseded_commit"]
    assert "reset-author" in Q.ROUTER_FREEZE["supersession"]


# --------------------------------------------------------------------------
# Gold verified against SQL
# --------------------------------------------------------------------------


def _scalar(conn: sqlite3.Connection, sql: str, params=()) -> object:
    return conn.execute(sql, params).fetchone()[0]


def test_count_by_category_gold_matches_sql(conn, questions):
    checked = 0
    for q in questions:
        if q.qtype != "count_by_category":
            continue
        table, column, value = q.meta["table"], q.meta["column"], q.meta["value"]
        n = _scalar(conn, f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" = ?', (value,))
        assert n == q.gold_scalar, (q.qid, q.text, n, q.gold_scalar)
        checked += 1
    assert checked >= 10


def test_numeric_aggregate_gold_matches_sql(conn, questions):
    checked = 0
    for q in questions:
        if q.qtype != "numeric_aggregate" or "column" not in q.meta:
            continue
        table, column = q.meta["table"], q.meta["column"]
        fn = q.gold_scalar_kind.upper()
        group = q.meta.get("group") or {}
        where = " AND ".join(f'"{k}" = ?' for k in group)
        sql = f'SELECT {fn}("{column}") FROM "{table}"'
        if where:
            sql += f" WHERE {where}"
        got = _scalar(conn, sql, tuple(group.values()))
        # SQL skips nulls; so does the gold. That agreement is the point.
        assert got == pytest.approx(q.gold_scalar, rel=1e-9), (q.qid, q.text)
        checked += 1
    assert checked >= 10


def test_date_range_gold_matches_sql(conn, questions):
    checked = 0
    for q in questions:
        if q.qtype != "date_range_aggregate":
            continue
        table, column = q.meta["table"], q.meta["column"]
        lo, hi = q.meta["window"]
        n = _scalar(
            conn,
            f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" BETWEEN ? AND ?',
            (lo, hi),
        )
        assert n == q.gold_scalar, (q.qid, q.text)
        checked += 1
    assert checked >= 10


def test_entity_lookup_gold_row_holds_the_expected_value(conn, questions, manifest):
    checked = 0
    for q in questions:
        if q.qtype != "entity_lookup":
            continue
        refs = q.gold_row_refs
        assert len(refs) == 1, q.qid
        table = q.meta["table"]
        pk = manifest.pk_col(table)
        row = conn.execute(
            f'SELECT * FROM "{table}" WHERE "{pk}" = ?', (refs[0].pk,)
        ).fetchone()
        assert row is not None, q.qid
        expected = q.meta["expected_value"]
        got = row[q.meta["attribute"]]
        if isinstance(expected, float):
            assert got == pytest.approx(expected)
        else:
            assert got == expected, (q.qid, q.text)
        # The entity named in the question is the one the gold points at.
        assert str(row[1]) in q.text or str(row[1]).lower() in q.text.lower()
        checked += 1
    assert checked >= 20


def test_multi_filter_gold_row_set_matches_sql(conn, questions, manifest):
    checked = 0
    column_for = {
        "region": None,  # resolved through regions.name
        "segment": "segment",
        "loyalty_tier": "loyalty_tier",
        "category": "category",
        "in_stock": "in_stock",
        "status": "status",
        "channel": "channel",
    }
    for q in questions:
        if q.qtype != "multi_filter_lookup":
            continue
        table = q.meta["table"]
        pk = manifest.pk_col(table)
        clauses: list[str] = []
        params: list[object] = []
        for key, value in q.meta["filters"].items():
            if key == "region":
                clauses.append(
                    '"region_id" = (SELECT "region_id" FROM "regions" WHERE "name" = ?)'
                )
                params.append(value)
            elif key == "unit_price_lt":
                clauses.append('"unit_price" < ?')
                params.append(value)
            elif key == "discount_gt":
                clauses.append('"discount_pct" > ?')
                params.append(value)
            else:
                clauses.append(f'"{column_for.get(key, key)}" = ?')
                params.append(value)
        rows = conn.execute(
            f'SELECT "{pk}" FROM "{table}" WHERE {" AND ".join(clauses)}', params
        ).fetchall()
        assert {str(r[0]) for r in rows} == {r.pk for r in q.gold_row_refs}, (q.qid, q.text)
        checked += 1
    assert checked >= 10


def test_join_gold_matches_sql(conn, questions, manifest):
    checked = 0
    for q in questions:
        if q.qtype != "join_relationship":
            continue
        if q.meta.get("region") is not None:
            n = _scalar(
                conn,
                'SELECT COUNT(*) FROM "orders" o JOIN "customers" c '
                'ON o."customer_id" = c."customer_id" '
                'JOIN "regions" r ON c."region_id" = r."region_id" '
                'WHERE r."name" = ?',
                (q.meta["region"],),
            )
            assert n == q.gold_scalar, (q.qid, q.text)
            checked += 1
        elif q.meta.get("product") is not None:
            n = _scalar(
                conn,
                'SELECT SUM(i."quantity") FROM "order_items" i '
                'JOIN "products" p ON i."product_id" = p."product_id" '
                'WHERE p."name" = ?',
                (q.meta["product"],),
            )
            assert n == q.gold_scalar, (q.qid, q.text)
            checked += 1
        elif q.meta.get("order_id") is not None:
            # order -> customer linkage must be the real foreign key.
            cid = _scalar(
                conn,
                'SELECT "customer_id" FROM "orders" WHERE "order_id" = ?',
                (q.meta["order_id"],),
            )
            refs = {(r.table, r.pk) for r in q.gold_row_refs}
            assert ("customers", str(cid)) in refs, (q.qid, refs)
            checked += 1
    assert checked >= 10


def test_schema_question_gold_matches_pragma(conn, questions):
    checked = 0
    for q in questions:
        if q.qtype != "schema_question":
            continue
        table = q.meta["table"]
        info = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        expected = q.meta["expected_value"]
        if q.meta["aspect"] == "columns":
            assert [r["name"] for r in info] == expected, q.qid
        elif q.meta["aspect"] == "primary_key":
            assert [r["name"] for r in info if r["pk"]] == [expected], q.qid
        elif q.meta["aspect"] == "nullable":
            assert [r["name"] for r in info if not r["notnull"]] == expected, q.qid
        else:
            fks = conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
            assert sorted(r["from"] for r in fks) == sorted(expected), q.qid
        # Gold is the schema card, never a row.
        assert q.gold.row_refs == frozenset()
        assert q.gold.schema_tables == frozenset({table})
        checked += 1
    assert checked >= 10


# --------------------------------------------------------------------------
# Distractors: the answer really must be absent
# --------------------------------------------------------------------------


def test_every_distractor_is_marked_unanswerable_and_has_no_gold(questions):
    distractors = [q for q in questions if q.qtype == "distractor"]
    assert len(distractors) >= 40
    for q in distractors:
        assert q.answerable is False
        assert q.gold.is_empty
        assert q.gold_scalar is None


def test_only_distractors_are_unanswerable(questions):
    for q in questions:
        if not q.answerable:
            assert q.qtype == "distractor", q.qid


def test_distractor_entities_do_not_exist_in_the_database(conn, questions):
    name_column = {"customers": "full_name", "products": "name", "regions": "name"}
    checked = 0
    for q in questions:
        if q.meta.get("kind") not in ("near_miss_entity", "near_miss_region"):
            continue
        table = q.meta["table"]
        column = name_column[table]
        n = _scalar(
            conn, f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" = ?', (q.meta["fake_value"],)
        )
        assert n == 0, (q.qid, q.meta["fake_value"])
        checked += 1
    assert checked >= 10


def test_distractor_entities_are_genuinely_near_misses(questions):
    """A distractor nobody could confuse with real data proves nothing."""
    for q in questions:
        if q.meta.get("kind") != "near_miss_entity":
            continue
        fake, real = q.meta["fake_value"], q.meta["nearest_real"]
        assert fake != real
        assert abs(len(fake) - len(real)) <= 2, (fake, real)
        shared = sum(1 for a, b in zip(fake.lower(), real.lower()) if a == b)
        assert shared / max(len(fake), len(real)) > 0.7, (fake, real)


def test_distractor_categories_and_enums_do_not_exist(conn, questions, manifest):
    checked = 0
    for q in questions:
        if q.meta.get("kind") not in ("near_miss_category", "near_miss_enum"):
            continue
        table = q.meta["table"]
        value = q.meta["fake_value"]
        # Held-out "unanchored" distractors name a value with no column word
        # beside it, so there is no single column to check -- the value must be
        # absent from *every* column of the table.
        columns = (
            [q.meta["column"]] if "column" in q.meta else list(manifest.columns(table))
        )
        for column in columns:
            n = _scalar(
                conn, f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" = ?', (value,)
            )
            assert n == 0, (q.qid, value, column)
        checked += 1
    assert checked >= 8


def test_unanchored_distractors_really_have_no_adjacent_column_word(questions, manifest):
    """The probe only means something if the column word is genuinely absent.

    If "How many orders were refunded?" happened to contain the word `status`,
    it would be a dev-style anchored distractor wearing a held-out label, and a
    refusal would prove nothing about generalisation.
    """
    checked = 0
    for q in questions:
        if q.meta.get("probe") != "value_without_adjacent_column_word":
            continue
        words = set(q.text.lower().replace("?", "").split())
        for table in manifest.table_names:
            for column in manifest.columns(table):
                for variant in {column, column.rstrip("s"), column.replace("_", " ")}:
                    assert variant.lower() not in words, (q.qid, variant)
        # The template carries one deliberate control whose value *does* sit
        # beside a table name; everything else must have no table word either,
        # or the probe is not testing what it claims to test.
        if not q.meta.get("anchored_on_table_name"):
            for table in manifest.table_names:
                for variant in {table, table.rstrip("s"), table.replace("_", " ")}:
                    idx = q.text.lower().split()
                    if variant.lower() in idx:
                        pos = idx.index(variant.lower())
                        prev = idx[pos - 1] if pos else ""
                        assert prev in ("many", "the", ""), (q.qid, variant, prev)
        checked += 1
    assert checked >= 8
    anchored = [
        q for q in questions
        if q.meta.get("probe") == "value_without_adjacent_column_word"
        and q.meta.get("anchored_on_table_name")
    ]
    assert len(anchored) == 1, "expected exactly one in-template control"


def test_distractor_attributes_are_not_columns(conn, questions):
    checked = 0
    for q in questions:
        if q.meta.get("kind") != "missing_attribute":
            continue
        info = conn.execute(f'PRAGMA table_info("{q.meta["table"]}")').fetchall()
        names = {r["name"].lower() for r in info}
        attr = q.meta["attribute"].lower()
        assert attr not in names
        assert attr.replace(" ", "_") not in names
        # ... but the entity it asks about is real, which is what makes it hard.
        assert q.meta["real_entity"] in q.text
        checked += 1
    assert checked >= 5


def test_distractor_tables_do_not_exist(conn, questions):
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    checked = 0
    for q in questions:
        if q.meta.get("kind") != "missing_table":
            continue
        assert q.meta["table"] not in tables
        checked += 1
    assert checked >= 4


def test_out_of_range_date_distractors_are_outside_the_data(conn, questions):
    checked = 0
    for q in questions:
        if q.meta.get("kind") != "out_of_range_date":
            continue
        lo, hi = q.meta["window"]
        n = _scalar(
            conn,
            'SELECT COUNT(*) FROM "orders" WHERE "order_date" BETWEEN ? AND ?',
            (lo, hi),
        )
        assert n == 0, (q.qid, q.meta["window"])
        assert hi < DATE_MIN.isoformat() or lo > DATE_MAX.isoformat()
        checked += 1
    assert checked >= 5


def test_no_out_of_range_count_question_is_called_a_distractor(questions):
    """Zero is the correct answer to "how many orders in March 2025".

    Counting that as unanswerable would let a system that answers "0" be scored
    as hallucinating, and would make the headline number dishonest.
    """
    for q in questions:
        if q.meta.get("kind") != "out_of_range_date":
            continue
        assert not q.text.lower().startswith("how many")


def test_distractor_kinds_are_varied(questions):
    kinds = {q.meta.get("kind") for q in questions if not q.answerable}
    assert len(kinds) >= 6, kinds


# --------------------------------------------------------------------------
# Stratified sampling (used identically across all 18 ablation cells)
# --------------------------------------------------------------------------


def test_stratified_sample_is_deterministic_and_keeps_every_type(full_questions):
    a = Q.stratified_sample(full_questions, 80, seed=7)
    b = Q.stratified_sample(full_questions, 80, seed=7)
    assert [q.qid for q in a] == [q.qid for q in b]
    assert len(a) == 80
    assert set(type_counts(a)) == set(type_counts(full_questions))


def test_stratified_sample_keeps_unanswerable_questions(full_questions):
    sample = Q.stratified_sample(full_questions, 60, seed=7)
    assert any(not q.answerable for q in sample)


def test_limit_uses_the_stratified_sample(manifest):
    limited = build_questions(manifest, "dev", TEST_SEED, limit=50)
    assert len(limited) == 50
    assert len(type_counts(limited)) >= MIN_TYPES
