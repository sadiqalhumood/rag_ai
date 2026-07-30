"""A searchable lexicon built from a source's schema and column profiles.

Both the router and the SQL generator need the same thing: given a question in
English, decide which tables, columns, and *literal values* it is talking about.
That mapping is derived entirely from `TableSchema` and `TableProfile` -- never
from question templates -- which is what keeps the router honest when it is
later evaluated on held-out phrasings.

Orchestrator-owned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..core.types import ColumnRole, TableProfile, TableRef, TableSchema

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

#: A categorical value directly after one of these is a verb, not a filter.
_AUXILIARIES = frozenset(
    {
        "was", "were", "is", "are", "be", "been", "being", "am",
        "has", "have", "had", "do", "does", "did", "get", "gets", "got",
        "getting", "become", "became", "gets",
    }
)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def _singular(word: str) -> str:
    """Crude English de-pluralisation, enough to match `orders` to `order`."""
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("ses"):
        return word[:-2]
    if len(word) > 2 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def name_variants(name: str) -> set[str]:
    """Surface forms a column or table name might appear as in a question."""
    parts = tokenize(name.replace("_", " "))
    out: set[str] = set()
    if parts:
        joined = " ".join(parts)
        out.add(joined)
        out.add(" ".join(_singular(p) for p in parts))
        # Trailing `_id` rarely appears in prose: "customer_id" -> "customer".
        if len(parts) > 1 and parts[-1] == "id":
            stem = " ".join(parts[:-1])
            out.add(stem)
            out.add(" ".join(_singular(p) for p in parts[:-1]))
    return {o for o in out if o}


@dataclass(frozen=True)
class ColumnMatch:
    table: TableRef
    column: str
    role: ColumnRole
    #: How much of the question the match consumed, in tokens.
    span: int
    matched_text: str


@dataclass(frozen=True)
class ValueMatch:
    """A literal value from a categorical column that appears in the question."""

    table: TableRef
    column: str
    value: str
    span: int
    #: Token index where the match starts, so callers can inspect the
    #: surrounding words. Needed to tell a filter literal from a verb that
    #: happens to collide with a category value.
    start: int = 0


@dataclass
class SchemaLexicon:
    schemas: Mapping[str, TableSchema] = field(default_factory=dict)
    profiles: Mapping[str, TableProfile] = field(default_factory=dict)
    #: surface form -> [(table, column)]
    _column_index: dict[str, list[tuple[TableRef, str]]] = field(
        default_factory=dict, repr=False
    )
    #: surface form -> table
    _table_index: dict[str, TableRef] = field(default_factory=dict, repr=False)
    #: lowercase value -> [(table, column, original value)]
    _value_index: dict[str, list[tuple[TableRef, str, str]]] = field(
        default_factory=dict, repr=False
    )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_source(cls, source) -> "SchemaLexicon":  # noqa: ANN001
        schemas: dict[str, TableSchema] = {}
        profiles: dict[str, TableProfile] = {}
        for table in source.tables():
            key = table.qualified
            try:
                schemas[key] = source.schema(table)
            except Exception:
                continue
            try:
                profiles[key] = source.profile(table)
            except Exception:
                pass
        lex = cls(schemas=schemas, profiles=profiles)
        lex.build()
        return lex

    def build(self) -> None:
        self._column_index.clear()
        self._table_index.clear()
        self._value_index.clear()

        for key, schema in self.schemas.items():
            for variant in name_variants(schema.table.name):
                self._table_index.setdefault(variant, schema.table)
            profile = self.profiles.get(key)
            for col in schema.columns:
                for variant in name_variants(col.name):
                    self._column_index.setdefault(variant, []).append(
                        (schema.table, col.name)
                    )
                if profile is None:
                    continue
                cprof = profile.columns.get(col.name)
                if cprof is None or cprof.role is not ColumnRole.CATEGORICAL:
                    continue
                for value in cprof.categories:
                    if value is None:
                        continue
                    text = str(value).strip()
                    if not text:
                        continue
                    # Key on the tokenized form, not raw lowercase: questions
                    # are tokenized too, so "Cote d'Ivoire" must index under
                    # "cote d ivoire" or it can never be matched.
                    key_form = " ".join(tokenize(text))
                    if not key_form:
                        continue
                    self._value_index.setdefault(key_form, []).append(
                        (schema.table, col.name, text)
                    )

    # -- lookup -------------------------------------------------------------

    def role_of(self, table: TableRef, column: str) -> ColumnRole:
        profile = self.profiles.get(table.qualified)
        return profile.role(column) if profile else ColumnRole.UNKNOWN

    def _ngrams(self, tokens: Sequence[str], max_n: int = 4):
        for n in range(min(max_n, len(tokens)), 0, -1):
            for i in range(len(tokens) - n + 1):
                yield i, n, " ".join(tokens[i : i + n])

    def match_tables(self, question: str) -> list[TableRef]:
        tokens = tokenize(question)
        found: list[TableRef] = []
        for _, _, gram in self._ngrams(tokens):
            table = self._table_index.get(gram)
            if table is not None and table not in found:
                found.append(table)
        return found

    def match_columns(self, question: str) -> list[ColumnMatch]:
        """Longest-first column matches, each token consumed at most once."""
        tokens = tokenize(question)
        used: set[int] = set()
        out: list[ColumnMatch] = []
        for i, n, gram in self._ngrams(tokens):
            if any(p in used for p in range(i, i + n)):
                continue
            entries = self._column_index.get(gram)
            if not entries:
                continue
            for table, column in entries:
                out.append(
                    ColumnMatch(
                        table=table,
                        column=column,
                        role=self.role_of(table, column),
                        span=n,
                        matched_text=gram,
                    )
                )
            used.update(range(i, i + n))
        return out

    def match_values(self, question: str) -> list[ValueMatch]:
        """Categorical literals appearing in the question, longest match first.

        Only CATEGORICAL columns are indexed. Matching free-text values here
        would produce nonsense equality predicates on prose columns.

        A match immediately preceded by an auxiliary verb is dropped. "How many
        orders were **placed** in 2024" collides with the `status` value
        'placed', and filtering on it turned a true answer of 725 into a
        confident 145. After an auxiliary, a word like that is a past
        participle, not a filter literal.
        """
        tokens = tokenize(question)
        used: set[int] = set()
        out: list[ValueMatch] = []
        for i, n, gram in self._ngrams(tokens, max_n=6):
            if any(p in used for p in range(i, i + n)):
                continue
            entries = self._value_index.get(gram)
            if not entries:
                continue
            if i > 0 and tokens[i - 1] in _AUXILIARIES:
                # Consume the span so a shorter sub-gram cannot re-match it,
                # but emit nothing.
                used.update(range(i, i + n))
                continue
            for table, column, value in entries:
                out.append(
                    ValueMatch(
                        table=table, column=column, value=value, span=n, start=i
                    )
                )
            used.update(range(i, i + n))
        return out

    def date_columns(self, table: TableRef) -> list[str]:
        profile = self.profiles.get(table.qualified)
        if not profile:
            return []
        return list(profile.columns_with_role(ColumnRole.DATE))

    def numeric_columns(self, table: TableRef) -> list[str]:
        profile = self.profiles.get(table.qualified)
        if not profile:
            return []
        return list(profile.columns_with_role(ColumnRole.NUMERIC))

    def primary_key(self, table: TableRef) -> tuple[str, ...]:
        schema = self.schemas.get(table.qualified)
        return schema.primary_key if schema else ()

    def foreign_key_path(
        self, left: TableRef, right: TableRef
    ) -> tuple[str, str] | None:
        """Direct FK join between two tables, as (left_column, right_column)."""
        for a, b in ((left, right), (right, left)):
            schema = self.schemas.get(a.qualified)
            if not schema:
                continue
            for col in schema.columns:
                if not col.references:
                    continue
                ref_table, _, ref_col = col.references.partition(".")
                if ref_table == b.name:
                    return (col.name, ref_col) if a is left else (ref_col, col.name)
        return None

    def tables(self) -> list[TableRef]:
        return [s.table for s in self.schemas.values()]
