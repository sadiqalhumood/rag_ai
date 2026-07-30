"""Question router: LOOKUP vs AGGREGATE vs HYBRID.

Structured-source retrieval is not document RAG. "How many orders came from
EMEA" is not answerable by fetching the five most similar rows, however good the
embeddings are -- the answer is a number that exists in no single row. Fetching
similar rows and letting a generator count them produces a confident wrong
answer, which is worse than a refusal.

So each question is classified:

* **LOOKUP** -- retrieval answers it. Entity lookups, attribute filters, and
  schema questions ("what columns does X have", answered from schema cards).
* **AGGREGATE** -- needs generated SQL. The SQL is linted read-only, executed
  against the source, and its result becomes a citable chunk.
* **HYBRID** -- spans more than one table; both a query result and retrieved
  rows are citable.

The classifier reads only `TableSchema` and `TableProfile`. It was written
before the evaluation question templates existed, so its cue lists are generic
English aggregation vocabulary rather than anything fitted to the harness.

Orchestrator-owned. Frozen before the held-out eval templates are written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ..core.types import ChunkKind, ColumnRole, QueryRoute, TableRef
from .schema_lexicon import SchemaLexicon, tokenize

# Generic English aggregation vocabulary. Weighted: an explicit "how many" is a
# stronger signal than a bare "per".
_AGG_CUES: dict[str, float] = {
    "how many": 1.0, "how much": 0.9, "number of": 0.9, "count": 0.9,
    "total": 0.85, "sum": 0.85, "average": 0.9, "avg": 0.9, "mean": 0.7,
    "median": 0.8, "highest": 0.8, "lowest": 0.8, "maximum": 0.85,
    "minimum": 0.85, "max": 0.7, "min": 0.7, "most": 0.6, "least": 0.6,
    "per": 0.5, "for each": 0.7, "each": 0.35, "breakdown": 0.8,
    "aggregate": 0.8, "grouped by": 0.8, "group by": 0.8, "overall": 0.4,
    "combined": 0.5, "altogether": 0.6, "in total": 0.9, "tally": 0.8,
}

_SCHEMA_CUES: tuple[str, ...] = (
    "what columns", "which columns", "what fields", "which fields",
    "what attributes", "which attributes", "columns does", "columns are in",
    "fields does", "schema of", "schema for", "structure of", "describe the",
    "what does the", "columns of", "fields of", "data type", "column types",
)

_SUPERLATIVE = re.compile(
    r"\b(highest|lowest|largest|smallest|biggest|greatest|most|least|top|bottom)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteDecision:
    route: QueryRoute
    confidence: float = 0.0
    reason: str = ""
    tables: tuple[TableRef, ...] = ()
    #: Chunk kinds worth retrieving for this question.
    preferred_kinds: frozenset[ChunkKind] = field(
        default_factory=lambda: frozenset(
            {ChunkKind.ROW, ChunkKind.SCHEMA_CARD, ChunkKind.SQL_RESULT}
        )
    )

    @property
    def needs_sql(self) -> bool:
        return self.route in (QueryRoute.AGGREGATE, QueryRoute.HYBRID)


def _aggregate_score(question: str) -> tuple[float, list[str]]:
    lowered = " " + " ".join(tokenize(question)) + " "
    hits: list[str] = []
    score = 0.0
    for cue, weight in _AGG_CUES.items():
        if f" {cue} " in lowered:
            hits.append(cue)
            score = max(score, weight)
    return score, hits


def _is_schema_question(question: str) -> bool:
    lowered = question.lower()
    return any(cue in lowered for cue in _SCHEMA_CUES)


def classify(
    question: str, lexicon: SchemaLexicon | None = None
) -> RouteDecision:
    """Classify a question. Without a lexicon, only lexical cues are used."""
    if not question or not question.strip():
        return RouteDecision(QueryRoute.LOOKUP, 0.0, "empty question")

    # Schema questions are answered from schema cards, never from SQL: asking
    # the database to describe itself would work, but the schema card is both
    # cheaper and directly citable.
    if _is_schema_question(question):
        return RouteDecision(
            route=QueryRoute.LOOKUP,
            confidence=0.9,
            reason="schema question -> schema cards",
            tables=tuple(lexicon.match_tables(question)) if lexicon else (),
            preferred_kinds=frozenset({ChunkKind.SCHEMA_CARD}),
        )

    agg_score, agg_hits = _aggregate_score(question)

    tables: list[TableRef] = []
    if lexicon is not None:
        tables = list(lexicon.match_tables(question))
        for match in lexicon.match_columns(question):
            if match.table not in tables:
                tables.append(match.table)
        for match in lexicon.match_values(question):
            if match.table not in tables:
                tables.append(match.table)

    # More than one table in play means a join: retrieval alone cannot combine
    # rows across tables, so SQL is needed, but the individual rows are still
    # worth retrieving and citing.
    if len(tables) >= 2:
        return RouteDecision(
            route=QueryRoute.HYBRID,
            confidence=0.7 + min(0.25, 0.1 * len(tables)),
            reason=f"question spans {len(tables)} tables: "
            + ", ".join(t.name for t in tables),
            tables=tuple(tables),
        )

    if agg_score >= 0.5:
        return RouteDecision(
            route=QueryRoute.AGGREGATE,
            confidence=agg_score,
            reason="aggregation cue(s): " + ", ".join(sorted(agg_hits)),
            tables=tuple(tables),
            preferred_kinds=frozenset({ChunkKind.SQL_RESULT, ChunkKind.SCHEMA_CARD}),
        )

    # A superlative without an explicit aggregate word ("which region had the
    # most orders") still needs SQL: ranking requires seeing every row.
    if _SUPERLATIVE.search(question):
        return RouteDecision(
            route=QueryRoute.AGGREGATE,
            confidence=0.55,
            reason="superlative implies ranking over all rows",
            tables=tuple(tables),
            preferred_kinds=frozenset({ChunkKind.SQL_RESULT, ChunkKind.SCHEMA_CARD}),
        )

    return RouteDecision(
        route=QueryRoute.LOOKUP,
        confidence=0.6 if tables else 0.4,
        reason="no aggregation or join cue; retrievable from rows",
        tables=tuple(tables),
        preferred_kinds=frozenset({ChunkKind.ROW, ChunkKind.SCHEMA_CARD}),
    )


class Router:
    """Binds a classifier to a concrete source and its lexicon."""

    def __init__(self, source=None, lexicon: SchemaLexicon | None = None) -> None:  # noqa: ANN001
        self.source = source
        if lexicon is None and source is not None:
            lexicon = SchemaLexicon.from_source(source)
        self.lexicon = lexicon

    def classify(self, question: str) -> RouteDecision:
        return classify(question, self.lexicon)
