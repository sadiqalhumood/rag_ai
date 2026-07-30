"""Adversarial tests for the read-only SQL linter.

The linter is the only thing standing between a generated string and the user's
database, so it is tested against attacks rather than happy paths.
"""

from __future__ import annotations

import pytest

from anyrag.core.errors import UnsafeQueryError
from anyrag.core.lint import is_readonly, lint_readonly, strip_noncode

# --------------------------------------------------------------------------
# Must be REJECTED
# --------------------------------------------------------------------------

MALICIOUS = [
    # Plain DML / DDL
    "DELETE FROM customers",
    "delete from customers",
    "INSERT INTO customers VALUES (1)",
    "UPDATE customers SET name = 'x'",
    "DROP TABLE customers",
    "ALTER TABLE customers ADD COLUMN x INT",
    "CREATE TABLE evil (id INT)",
    "TRUNCATE customers",
    "GRANT ALL ON customers TO PUBLIC",
    # Stacked statements
    "SELECT 1; DELETE FROM customers",
    "SELECT 1;DELETE FROM customers",
    "SELECT 1 ; DROP TABLE t ;",
    "SELECT * FROM t;;",
    # Comment-obfuscated DML
    "SELECT 1; -- harmless\nDELETE FROM customers",
    "SELECT 1 /* nothing to see */ ; DROP TABLE t",
    "/* leading */ DELETE FROM t",
    "SELECT/**/1;/**/DROP/**/TABLE/**/t",
    # DML hidden in a CTE -- a real Postgres write that starts with WITH
    "WITH x AS (DELETE FROM customers RETURNING *) SELECT * FROM x",
    "WITH x AS (INSERT INTO t VALUES (1) RETURNING id) SELECT * FROM x",
    "WITH x AS (UPDATE t SET a=1 RETURNING *) SELECT * FROM x",
    # SELECT INTO creates a table in Postgres
    "SELECT * INTO evil FROM customers",
    # Locking / session state
    "SELECT * FROM t FOR UPDATE",
    "SET search_path = evil",
    "BEGIN",
    "COMMIT",
    "LOCK TABLE customers",
    # Filesystem and OS reach-through
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT lo_import('/etc/passwd')",
    "SELECT writefile('/tmp/x', 'y')",
    "SELECT load_extension('evil.so')",
    "SELECT pg_sleep(60)",
    "SELECT pg_terminate_backend(1)",
    # SQLite specifics
    "PRAGMA writable_schema = 1",
    "ATTACH DATABASE '/tmp/evil.db' AS evil",
    "VACUUM",
    "REPLACE INTO t VALUES (1)",
    # Procedural bodies
    "DO $$ BEGIN PERFORM 1; END $$",
    "SELECT $$ hello $$",
    "CALL some_procedure()",
    "EXECUTE stmt",
    # Malformed / ambiguous input the linter must not try to guess through
    "SELECT 'unterminated",
    'SELECT "unterminated',
    "SELECT 1 /* unterminated",
    "",
    "   ",
    "NOT EVEN SQL",
    "SELECT 1\x00; DROP TABLE t",
]


@pytest.mark.parametrize("sql", MALICIOUS, ids=range(len(MALICIOUS)))
def test_malicious_queries_are_rejected(sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        lint_readonly(sql)
    assert is_readonly(sql) is False


# --------------------------------------------------------------------------
# Must be ACCEPTED
# --------------------------------------------------------------------------

BENIGN = [
    "SELECT 1",
    "select 1",
    "SELECT * FROM customers",
    "SELECT * FROM customers;",
    "  SELECT * FROM customers  ",
    "SELECT COUNT(*) FROM orders WHERE region = 'EMEA'",
    "SELECT c.name, COUNT(o.id) FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.name",
    "WITH recent AS (SELECT * FROM orders WHERE created_at > '2024-01-01') SELECT COUNT(*) FROM recent",
    "SELECT * FROM (VALUES (1), (2)) AS v(x)",
    "(SELECT 1) UNION (SELECT 2)",
    "SELECT CASE WHEN a > 1 THEN 'big' ELSE 'small' END FROM t",
    "SELECT AVG(total) FROM orders WHERE created_at BETWEEN '2024-01-01' AND '2024-06-30'",
    "SELECT * FROM orders ORDER BY total DESC LIMIT 10 OFFSET 5",
    # String literals that merely *contain* dangerous words must be fine --
    # over-blocking these is how naive linters become useless.
    "SELECT * FROM t WHERE note = 'please delete me'",
    "SELECT * FROM t WHERE cmd = 'DROP TABLE users'",
    "SELECT 'INSERT INTO' AS phrase",
    "SELECT * FROM logs WHERE msg LIKE '%truncate%'",
    # Identifiers that contain dangerous words as substrings
    "SELECT is_deleted FROM t",
    "SELECT deleted_at, created_at FROM t",
    "SELECT offset_value FROM t",
    "SELECT dataset_id FROM t",
    # Quoted identifiers
    'SELECT "select" FROM "table"',
    'SELECT t."group" FROM t',
    # REPLACE as a scalar function is legitimate read-only SQL
    "SELECT REPLACE(name, 'a', 'b') FROM customers",
    # Escaped quote inside a literal
    "SELECT * FROM t WHERE name = 'O''Brien'",
    # Comments in benign positions
    "SELECT 1 -- trailing comment",
    "SELECT /* inline */ 1",
    "SELECT 1 /* nested /* comment */ still comment */",
    # Non-ASCII content
    "SELECT * FROM t WHERE name = 'مرحبا'",
]


@pytest.mark.parametrize("sql", BENIGN, ids=range(len(BENIGN)))
def test_benign_queries_are_accepted(sql: str) -> None:
    out = lint_readonly(sql)
    assert out
    assert not out.endswith(";")
    assert is_readonly(sql) is True


def test_backslash_is_not_a_string_escape() -> None:
    """Failing closed: we must not be fooled into thinking we are inside a string.

    Honouring `\\'` as an escape would let an attacker keep the scanner inside a
    literal while the database had already terminated it.
    """
    with pytest.raises(UnsafeQueryError):
        lint_readonly(r"SELECT 'a\'; DROP TABLE t; --'")


def test_strip_noncode_blanks_literals_but_keeps_structure() -> None:
    code = strip_noncode("SELECT 'delete from t' FROM \"my table\" -- hi")
    assert "delete" not in code.lower()
    assert "SELECT" in code
    assert "FROM" in code


def test_error_carries_reason_and_fragment() -> None:
    with pytest.raises(UnsafeQueryError) as exc:
        lint_readonly("SELECT 1; DELETE FROM t")
    assert exc.value.reason
    assert exc.value.fragment


def test_non_string_input_rejected() -> None:
    for bad in (None, 123, ["SELECT 1"]):
        with pytest.raises(UnsafeQueryError):
            lint_readonly(bad)  # type: ignore[arg-type]


def test_trailing_semicolon_is_stripped_not_rejected() -> None:
    assert lint_readonly("SELECT 1;") == "SELECT 1"
