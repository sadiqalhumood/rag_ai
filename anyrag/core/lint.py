"""Read-only SQL linter.

Every string that reaches a source's `execute_readonly` passes through
`lint_readonly` first. The contract is narrow on purpose: a single `SELECT`, or
a single `WITH ... SELECT`, and nothing else.

Design notes that matter for security:

* Keyword scanning runs only over *code* segments. String literals and quoted
  identifiers are excised first, so `SELECT 'please delete me'` is accepted
  while `SELECT 1; DELETE FROM t` is not. Getting this backwards is the classic
  way naive linters both over- and under-block.

* Where the scanner is ambiguous it fails *closed*. In particular backslash is
  NOT treated as a string escape: standard SQL escapes a quote by doubling it,
  and honouring `\\'` would let an attacker keep the scanner inside a "string"
  while the database had already left it. Ending a string too early merely
  causes more text to be scanned as code, which can only produce extra
  rejections -- the safe direction.

* Dollar-quoting (`$$ ... $$`) is rejected outright rather than parsed. No
  read-only SELECT we generate needs it, and it is the standard vehicle for
  smuggling procedural code into Postgres.

This module is orchestrator-owned and is used by both source adapters and the
NL->SQL router.
"""

from __future__ import annotations

import re
from typing import Iterator, Literal

from .errors import UnsafeQueryError

SegmentKind = Literal["code", "string", "ident", "comment"]

# Statement keywords that must never appear outside a string literal. Anything
# that writes, locks, changes session state, or executes procedural code.
_DENY_WORDS: tuple[str, ...] = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "GRANT", "REVOKE", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX",
    "CLUSTER", "COPY", "MERGE", "UPSERT", "CALL", "DO", "EXECUTE", "PREPARE",
    "DEALLOCATE", "LOCK", "UNLOCK", "NOTIFY", "LISTEN", "UNLISTEN", "DISCARD",
    "RESET", "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE", "START",
    "RETURNING", "INTO", "SET", "REFRESH", "IMPORT", "LOAD", "INSTALL",
    "ATTACHED", "SHUTDOWN", "CHECKPOINT",
)

# Functions that read or write the filesystem / OS, or that let a "read-only"
# query become a denial of service.
_DENY_FUNCS: tuple[str, ...] = (
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "pg_logdir_ls", "lo_import", "lo_export", "lo_put",
    "pg_terminate_backend", "pg_cancel_backend", "pg_sleep", "pg_sleep_for",
    "dblink", "dblink_exec", "postgres_fdw_handler",
    "writefile", "readfile", "edit", "load_extension", "fts3_tokenizer",
    "system", "shell", "eval", "exec", "xp_cmdshell",
)

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")

# `REPLACE` is special-cased: `REPLACE(a, b, c)` is a legitimate read-only
# scalar function in both SQLite and Postgres, while `REPLACE INTO` is SQLite
# DML. Denying the bare word would break valid SELECTs, so we deny the
# statement form only (the INTO deny-word also catches it, belt and braces).
_REPLACE_INTO_RE = re.compile(r"\bREPLACE\s+INTO\b", re.IGNORECASE)


def _scan(sql: str) -> Iterator[tuple[SegmentKind, str]]:
    """Split SQL into code / string / identifier / comment segments.

    Raises UnsafeQueryError on an unterminated literal or comment: an
    unterminated construct means we cannot reason about the rest of the text,
    and guessing is exactly what we must not do.
    """
    i, n = 0, len(sql)
    buf: list[str] = []

    def flush() -> Iterator[tuple[SegmentKind, str]]:
        if buf:
            yield "code", "".join(buf)
            buf.clear()

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        # -- line comment
        if ch == "-" and nxt == "-":
            yield from flush()
            end = sql.find("\n", i)
            end = n if end == -1 else end
            yield "comment", sql[i:end]
            i = end
            continue

        # /* block comment */ -- Postgres allows nesting.
        if ch == "/" and nxt == "*":
            yield from flush()
            depth, j = 1, i + 2
            while j < n and depth:
                if sql[j] == "/" and j + 1 < n and sql[j + 1] == "*":
                    depth += 1
                    j += 2
                elif sql[j] == "*" and j + 1 < n and sql[j + 1] == "/":
                    depth -= 1
                    j += 2
                else:
                    j += 1
            if depth:
                raise UnsafeQueryError("unterminated block comment", sql)
            yield "comment", sql[i:j]
            i = j
            continue

        # Dollar quoting: rejected wholesale, see module docstring.
        if ch == "$":
            m = re.match(r"\$[A-Za-z_0-9]*\$", sql[i:])
            if m:
                raise UnsafeQueryError(
                    "dollar-quoted string is not permitted", sql, m.group(0)
                )

        # 'string literal' with '' doubling as the only escape.
        if ch == "'":
            yield from flush()
            j = i + 1
            while j < n:
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                raise UnsafeQueryError("unterminated string literal", sql)
            yield "string", sql[i : j + 1]
            i = j + 1
            continue

        # "quoted identifier" / `backticked identifier`
        if ch in '"`':
            yield from flush()
            close = ch
            j = i + 1
            while j < n:
                if sql[j] == close:
                    if close == '"' and j + 1 < n and sql[j + 1] == '"':
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                raise UnsafeQueryError("unterminated quoted identifier", sql)
            yield "ident", sql[i : j + 1]
            i = j + 1
            continue

        buf.append(ch)
        i += 1

    yield from flush()


def strip_noncode(sql: str) -> str:
    """Return only the code segments, with literals/idents/comments blanked.

    Literals collapse to `''` and identifiers to `"x"` so that token structure
    (and therefore statement separators) is preserved.
    """
    out: list[str] = []
    for kind, text in _scan(sql):
        if kind == "code":
            out.append(text)
        elif kind == "string":
            out.append("''")
        elif kind == "ident":
            out.append('"x"')
        else:  # comment -> whitespace, so `SELECT/**/1` stays two tokens
            out.append(" ")
    return "".join(out)


def lint_readonly(sql: str) -> str:
    """Validate `sql` as a single read-only statement.

    Returns the original SQL stripped of leading/trailing whitespace and any
    single trailing semicolon. Raises UnsafeQueryError otherwise.
    """
    if sql is None or not isinstance(sql, str):
        raise UnsafeQueryError("query must be a string", str(sql))

    raw = sql.strip()
    if not raw:
        raise UnsafeQueryError("empty query", sql)

    # Control characters (other than ordinary whitespace) are never legitimate
    # and are a common obfuscation vector.
    for chq in raw:
        if ord(chq) < 32 and chq not in "\t\n\r":
            raise UnsafeQueryError(
                "control character in query", sql, repr(chq)
            )

    code = strip_noncode(raw)

    # Exactly one statement. A single trailing `;` is tolerated; anything else
    # is stacking.
    stripped_code = code.strip()
    if stripped_code.endswith(";"):
        stripped_code = stripped_code[:-1]
    if ";" in stripped_code:
        raise UnsafeQueryError(
            "multiple statements are not permitted", sql, ";"
        )

    upper = stripped_code.upper()

    # First keyword must be SELECT or WITH. Leading parens are allowed:
    # `(SELECT 1) UNION (SELECT 2)` is a legal read-only query.
    first = _WORD_RE.search(stripped_code)
    if not first:
        raise UnsafeQueryError("no SQL keyword found", sql)
    head = first.group(0).upper()
    if head not in ("SELECT", "WITH"):
        raise UnsafeQueryError(
            f"only SELECT and WITH queries are permitted, got {head}", sql, head
        )

    if head == "WITH" and not re.search(r"\bSELECT\b", upper):
        raise UnsafeQueryError("WITH clause contains no SELECT", sql, "WITH")

    if _REPLACE_INTO_RE.search(stripped_code):
        raise UnsafeQueryError("REPLACE INTO is not permitted", sql, "REPLACE INTO")

    # Deny-word scan over code segments only. This is what catches DML hidden
    # in a CTE -- `WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x` is a
    # real Postgres write that starts with the word WITH.
    for word in _DENY_WORDS:
        if re.search(rf"\b{word}\b", upper):
            raise UnsafeQueryError(
                f"forbidden keyword {word} in query", sql, word
            )

    lowered = stripped_code.lower()
    for func in _DENY_FUNCS:
        if re.search(rf"\b{re.escape(func)}\b", lowered):
            raise UnsafeQueryError(
                f"forbidden function {func} in query", sql, func
            )

    return raw[:-1].strip() if raw.endswith(";") else raw


def is_readonly(sql: str) -> bool:
    """Boolean form of `lint_readonly`, for filtering rather than enforcing."""
    try:
        lint_readonly(sql)
        return True
    except UnsafeQueryError:
        return False
