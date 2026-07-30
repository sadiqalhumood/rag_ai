"""Exception hierarchy for anyrag.

Kept in its own module so every other module can import it without cycles.
"""

from __future__ import annotations


class AnyRagError(Exception):
    """Base class for every error raised by anyrag."""


class UnsafeQueryError(AnyRagError):
    """A SQL string failed the read-only linter.

    Carries the offending fragment so callers can log *why* without re-running
    the linter.
    """

    def __init__(self, reason: str, sql: str = "", fragment: str = "") -> None:
        self.reason = reason
        self.sql = sql
        self.fragment = fragment
        detail = f"{reason}"
        if fragment:
            detail += f" (offending fragment: {fragment!r})"
        super().__init__(detail)


class CitationError(AnyRagError):
    """An answer claimed to be grounded but carried no citations.

    This is deliberately an error and not a warning: an uncited answer is a
    product failure, so it must be impossible to construct one.
    """


class SourceError(AnyRagError):
    """A DataSource could not be opened, introspected, or queried."""


class SchemaError(SourceError):
    """A table or column was referenced that does not exist in the source."""


class IndexError_(AnyRagError):
    """An index operation failed (dimension mismatch, missing persistence, ...)."""


class ConfigError(AnyRagError):
    """Invalid configuration."""


class BudgetExceededError(AnyRagError):
    """Context could not be packed within the token budget."""


class SqlGenerationError(AnyRagError):
    """The NL->SQL generator could not produce SQL for a question.

    This is an expected, non-fatal outcome: the router converts it into a
    refusal rather than guessing.
    """
