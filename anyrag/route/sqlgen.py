"""Natural language -> SQL for aggregate questions.

`HeuristicSqlGenerator` is slot-filling driven entirely by the schema profile:
it discovers aggregate function, target column, group-by, and filter predicates
by matching the question against table names, column names, and the *observed
value vocabulary* of categorical columns. It has never seen the evaluation
question templates.

Its coverage is deliberately incomplete, and that is reported rather than
hidden. The important property is the failure mode: when the generator cannot
confidently build a query it raises `SqlGenerationError`, and the caller turns
that into a refusal. A plausible wrong number is far worse than "I don't know".

Every statement it produces is passed through `lint_readonly` before it is
returned, so a malformed or malicious slot value cannot escape as executable
SQL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ..core.errors import SqlGenerationError
from ..core.lint import lint_readonly
from ..core.types import ColumnRole, TableRef
from .schema_lexicon import SchemaLexicon, tokenize

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_AGG_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("COUNT", ("how many", "number of", "count of", "count", "how much", "tally")),
    ("AVG", ("average", "avg", "mean", "typical")),
    ("SUM", ("total", "sum", "combined", "altogether", "in total")),
    ("MAX", ("maximum", "max", "highest", "largest", "biggest", "greatest", "most")),
    ("MIN", ("minimum", "min", "lowest", "smallest", "least")),
)

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_GROUPBY_CUE = re.compile(
    r"\b(?:per|by|for each|grouped by|group by|broken down by)\s+(.{1,40})",
    re.IGNORECASE,
)


def _quote_literal(value: str) -> str:
    """Single-quote a literal, doubling embedded quotes.

    Slot values come from the *database's own* categorical vocabulary rather
    than from user text, but they are escaped anyway and the finished statement
    is linted -- defence in depth is cheap here.
    """
    return "'" + str(value).replace("'", "''") + "'"


def _quote_ident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""):
        raise SqlGenerationError(f"unsafe identifier: {name!r}")
    return f'"{name}"'


@dataclass
class SqlPlan:
    """The slots recovered from a question, before rendering to SQL."""

    table: TableRef
    agg: str = "COUNT"
    target: str | None = None
    group_by: str | None = None
    filters: list[str] = field(default_factory=list)
    join: tuple[TableRef, str, str] | None = None
    explanation: str = ""


class HeuristicSqlGenerator:
    """Schema-profile-driven NL->SQL. Refuses when out of coverage."""

    name = "heuristic"

    def __init__(self, source=None, lexicon: SchemaLexicon | None = None) -> None:  # noqa: ANN001
        self.source = source
        if lexicon is None and source is not None:
            lexicon = SchemaLexicon.from_source(source)
        self.lexicon = lexicon

    # -- slot detection -----------------------------------------------------

    @staticmethod
    def _detect_agg(question: str) -> tuple[str, str]:
        lowered = " " + " ".join(tokenize(question)) + " "
        for agg, cues in _AGG_PATTERNS:
            for cue in cues:
                if f" {cue} " in lowered:
                    return agg, cue
        raise SqlGenerationError(
            "no aggregate function detected in question"
        )

    def _detect_dates(
        self, question: str, lex: SchemaLexicon, table: TableRef
    ) -> list[str]:
        date_cols = lex.date_columns(table)
        if not date_cols:
            return []
        col = _quote_ident(date_cols[0])
        lowered = question.lower()

        iso = _ISO_DATE.findall(question)
        if len(iso) >= 2:
            spans = ["-".join(m) for m in iso[:2]]
            return [f"{col} BETWEEN {_quote_literal(spans[0])} AND {_quote_literal(spans[1])}"]
        if len(iso) == 1:
            date = "-".join(iso[0])
            if "before" in lowered or "prior to" in lowered or "up to" in lowered:
                return [f"{col} < {_quote_literal(date)}"]
            if "after" in lowered or "since" in lowered:
                return [f"{col} > {_quote_literal(date)}"]
            return [f"{col} = {_quote_literal(date)}"]

        years = [m.group(0) for m in _YEAR.finditer(question)]
        month = next(
            (num for name, num in _MONTHS.items() if f" {name} " in f" {lowered} "),
            None,
        )
        if years and month:
            year = years[0]
            last = _month_end(int(year), month)
            return [
                f"{col} BETWEEN {_quote_literal(f'{year}-{month:02d}-01')} "
                f"AND {_quote_literal(f'{year}-{month:02d}-{last:02d}')}"
            ]
        if len(years) >= 2:
            lo, hi = sorted(years[:2])
            return [
                f"{col} BETWEEN {_quote_literal(lo + '-01-01')} "
                f"AND {_quote_literal(hi + '-12-31')}"
            ]
        if years:
            year = years[0]
            if "before" in lowered:
                return [f"{col} < {_quote_literal(year + '-01-01')}"]
            if "after" in lowered or "since" in lowered:
                return [f"{col} > {_quote_literal(year + '-12-31')}"]
            return [
                f"{col} BETWEEN {_quote_literal(year + '-01-01')} "
                f"AND {_quote_literal(year + '-12-31')}"
            ]
        return []

    def _detect_group_by(
        self, question: str, lex: SchemaLexicon, table: TableRef
    ) -> str | None:
        match = _GROUPBY_CUE.search(question)
        if not match:
            return None
        tail = match.group(1)
        for cand in lex.match_columns(tail):
            if cand.table == table and cand.role in (
                ColumnRole.CATEGORICAL,
                ColumnRole.DATE,
                ColumnRole.BOOLEAN,
                ColumnRole.ID,
            ):
                return cand.column
        return None

    def _pick_table(
        self, question: str, lex: SchemaLexicon
    ) -> tuple[TableRef, tuple[TableRef, str, str] | None]:
        """Choose the table the question is about, plus an optional FK join."""
        scores: dict[str, tuple[TableRef, float]] = {}

        def bump(table: TableRef, amount: float) -> None:
            key = table.qualified
            prev = scores.get(key, (table, 0.0))[1]
            scores[key] = (table, prev + amount)

        for table in lex.match_tables(question):
            bump(table, 2.0)
        for cmatch in lex.match_columns(question):
            bump(cmatch.table, 1.0)
        for vmatch in lex.match_values(question):
            bump(vmatch.table, 1.5)

        if not scores:
            raise SqlGenerationError(
                "question does not reference any known table or column"
            )

        ordered = sorted(scores.values(), key=lambda kv: (-kv[1], kv[0].name))
        primary = ordered[0][0]

        join = None
        if len(ordered) > 1:
            other = ordered[1][0]
            path = lex.foreign_key_path(primary, other)
            if path is not None:
                join = (other, path[0], path[1])
        return primary, join

    def _detect_target(
        self, question: str, lex: SchemaLexicon, table: TableRef, agg: str
    ) -> str | None:
        if agg == "COUNT":
            return None
        candidates = [
            m for m in lex.match_columns(question)
            if m.table == table and m.role is ColumnRole.NUMERIC
        ]
        if candidates:
            return max(candidates, key=lambda m: m.span).column
        numeric = lex.numeric_columns(table)
        if len(numeric) == 1:
            # Unambiguous: only one thing in this table can be summed.
            return numeric[0]
        raise SqlGenerationError(
            f"{agg} needs a numeric column but the question names none "
            f"(candidates: {numeric or 'none'})"
        )

    # -- generation ---------------------------------------------------------

    def plan(self, question: str, source=None) -> SqlPlan:  # noqa: ANN001
        lex = self.lexicon
        if lex is None and source is not None:
            lex = SchemaLexicon.from_source(source)
        if lex is None:
            raise SqlGenerationError("no schema lexicon available")

        agg, cue = self._detect_agg(question)
        table, join = self._pick_table(question, lex)
        target = self._detect_target(question, lex, table, agg)
        group_by = self._detect_group_by(question, lex, table)

        filters: list[str] = []
        for vmatch in lex.match_values(question):
            if vmatch.table == table:
                filters.append(
                    f"{_quote_ident(vmatch.column)} = {_quote_literal(vmatch.value)}"
                )
            elif join is not None and vmatch.table == join[0]:
                filters.append(
                    f"{_quote_ident(join[0].name)}.{_quote_ident(vmatch.column)} "
                    f"= {_quote_literal(vmatch.value)}"
                )
        filters.extend(self._detect_dates(question, lex, table))

        return SqlPlan(
            table=table,
            agg=agg,
            target=target,
            group_by=group_by,
            filters=filters,
            join=join,
            explanation=f"matched aggregate cue {cue!r} on table {table.name}",
        )

    def render(self, plan: SqlPlan) -> str:
        table = _quote_ident(plan.table.name)
        if plan.agg == "COUNT":
            select_expr = "COUNT(*)"
        else:
            if not plan.target:
                raise SqlGenerationError(f"{plan.agg} requires a target column")
            select_expr = f"{plan.agg}({table}.{_quote_ident(plan.target)})"

        parts = ["SELECT"]
        if plan.group_by:
            gb = f"{table}.{_quote_ident(plan.group_by)}"
            parts.append(f"{gb}, {select_expr} AS value")
        else:
            parts.append(f"{select_expr} AS value")
        parts.append(f"FROM {table}")

        if plan.join is not None:
            other, left_col, right_col = plan.join
            oth = _quote_ident(other.name)
            parts.append(
                f"JOIN {oth} ON {table}.{_quote_ident(left_col)} "
                f"= {oth}.{_quote_ident(right_col)}"
            )

        # Filters are rendered unqualified against the primary table unless they
        # already carry a table prefix.
        rendered_filters = [
            f if f.lstrip().startswith('"' + plan.table.name + '"') or "." in f.split("=")[0]
            else f"{table}.{f}"
            for f in plan.filters
        ]
        if rendered_filters:
            parts.append("WHERE " + " AND ".join(rendered_filters))
        if plan.group_by:
            parts.append(f"GROUP BY {table}.{_quote_ident(plan.group_by)}")
            parts.append("ORDER BY value DESC")

        return " ".join(parts)

    def generate(self, question: str, source=None) -> str:  # noqa: ANN001
        sql = self.render(self.plan(question, source))
        # Never hand back a statement that has not passed the read-only linter.
        return lint_readonly(sql)

    def can_generate(self, question: str, source=None) -> bool:  # noqa: ANN001
        try:
            self.generate(question, source)
            return True
        except SqlGenerationError:
            return False


def _month_end(year: int, month: int) -> int:
    if month == 12:
        return 31
    import calendar

    return calendar.monthrange(year, month)[1]
