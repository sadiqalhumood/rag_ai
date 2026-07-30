"""Question routing: LOOKUP vs AGGREGATE vs HYBRID.

Orchestrator-owned. Phase 0 ships the interface only; the real classifier and
NL->SQL generator land in Phase 2 and are then frozen (their commit SHA is
recorded in DECISIONS.md) before the held-out eval templates are written.
"""

from __future__ import annotations

from .router import RouteDecision, Router, classify  # noqa: F401
from .schema_lexicon import SchemaLexicon  # noqa: F401
from .sqlgen import HeuristicSqlGenerator, SqlPlan  # noqa: F401

__all__ = [
    "RouteDecision",
    "Router",
    "classify",
    "SchemaLexicon",
    "HeuristicSqlGenerator",
    "SqlPlan",
]
