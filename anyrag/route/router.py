"""Question router (Phase 0 stub -- real implementation lands in Phase 2).

Structured-source retrieval is not document RAG: "how many orders in region X"
cannot be answered by fetching five similar rows, no matter how good the
embeddings are. The router decides whether a question is answerable by
retrieval (LOOKUP), needs generated SQL (AGGREGATE), or both (HYBRID).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.types import QueryRoute


@dataclass(frozen=True)
class RouteDecision:
    route: QueryRoute
    confidence: float = 0.0
    reason: str = ""


def classify(question: str, profiles=None) -> RouteDecision:  # noqa: ANN001
    """Placeholder classifier: everything is a LOOKUP until Phase 2."""
    return RouteDecision(
        route=QueryRoute.LOOKUP, confidence=0.0, reason="phase-0 stub"
    )


class Router:
    """Placeholder router preserving the Phase 2 call signature."""

    def __init__(self, source=None, sql_generator=None) -> None:  # noqa: ANN001
        self.source = source
        self.sql_generator = sql_generator

    def classify(self, question: str) -> RouteDecision:
        return classify(question)
