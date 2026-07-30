"""NL -> SQL for aggregate questions (Phase 0 stub).

Every generated statement passes through `lint_readonly` before it reaches a
source, and out-of-coverage questions raise `SqlGenerationError` so the caller
refuses instead of guessing.
"""

from __future__ import annotations

from ..core.errors import SqlGenerationError


class HeuristicSqlGenerator:
    """Placeholder. Phase 2 fills this in from the schema profile."""

    name = "heuristic"

    def __init__(self, source=None) -> None:  # noqa: ANN001
        self.source = source

    def generate(self, question: str, source=None) -> str:  # noqa: ANN001
        raise SqlGenerationError("SQL generation not implemented until Phase 2")
