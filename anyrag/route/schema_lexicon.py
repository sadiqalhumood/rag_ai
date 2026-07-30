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

#: Words that never form part of a filter value, used to bound the phrase
#: captured either side of a column name.
_FUNCTION_WORDS = frozenset(
    {
        "the", "a", "an", "each", "every", "per", "this", "that", "these",
        "those", "which", "what", "whose", "any", "all", "some", "no", "none",
        "of", "in", "on", "at", "by", "for", "and", "or", "but", "with",
        "from", "to", "their", "its", "his", "her", "our", "your", "my",
        "how", "many", "much", "count", "number", "total", "average", "avg",
        "sum", "list", "show", "give", "find", "there", "are", "is", "was",
        "were", "do", "does", "did", "have", "has", "had", "be", "been",
        "same", "different", "other", "within", "into", "over", "under",
    }
)

#: Date/number vocabulary, which the date slot handles rather than the
#: categorical matcher.
_TEMPORAL_WORDS = frozenset(
    {
        "between", "before", "after", "since", "until", "during", "from",
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec", "year", "years", "month", "months", "day", "days",
        "quarter", "week", "weeks", "date", "dates",
    }
)

#: Skipped when reading a value that follows a column name ("status is X").
_COPULAS = frozenset({"is", "are", "was", "were", "equals", "equal", "of", "to"})

#: Explicit attribute requests: "what is the X of ...", "how long is the X on
#: ...", "which X does ... belong to", "in which X is ...".
_ATTR_REQUEST = re.compile(
    r"\b(?:what|which)\s+(?:is|are|was|were)\s+the\s+(?P<attr>[\w' -]{2,40}?)\s+(?:of|for|on|in)\b"
    r"|\bhow\s+(?:long|much|heavy|big|old)\s+is\s+the\s+(?P<attr2>[\w' -]{2,40}?)\s+(?:of|for|on|in)\b"
    r"|\bin\s+which\s+(?P<attr3>[\w' -]{2,40}?)\s+(?:is|are|was|were)\b"
    r"|\bwhich\s+(?P<attr4>[\w' -]{2,40}?)\s+does\b",
    re.IGNORECASE,
)

#: Entity-type requests: "how many <thing>", "which <thing> <verb>".
_ENTITY_REQUEST = re.compile(
    r"\bhow\s+many\s+(?P<ent>[\w' -]{2,30}?)\s+"
    r"(?:are|is|was|were|do|does|did|have|has|had|in|with|from|by|deliver|"
    r"delivers|ship|ships|stock|stocks|belong|belongs)\b"
    r"|\bwhich\s+(?P<ent2>[\w' -]{2,30}?)\s+"
    r"(?:stores|store|manages|manage|supplies|supply|handles|handle)\b",
    re.IGNORECASE,
)

#: Attribute words that are *about* the schema rather than a column in it.
_META_ATTRIBUTES = frozenset(
    {"column", "columns", "field", "fields", "attribute", "attributes",
     "schema", "structure", "type", "types", "row", "rows", "table", "tables"}
)

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


def _looks_like_a_value(phrase: str) -> bool:
    """Reject phrases that cannot be a categorical literal.

    Dates and bare numbers are handled by the date/numeric slots, not by
    category matching; treating "between 2023 03 01" as an unknown category
    turned a valid date-range query into a refusal.
    """
    words = phrase.split()
    if not words:
        return False
    if not any(any(ch.isalpha() for ch in w) for w in words):
        return False
    if all(w.isdigit() or w in _TEMPORAL_WORDS for w in words):
        return False
    return True


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

    def unresolved_constraints(self, question: str) -> list[tuple[str, str]]:
        """Find filter constraints that name a value the data does not contain.

        This is the guard against the worst failure this system can produce.
        "How many products are in the Groceries category?" names a category that
        does not exist; silently dropping the predicate answers a *different*
        question — the unfiltered total — with total confidence. Measured on the
        eval's distractor set, that single behaviour accounted for most of a
        65.6% false-answer rate.

        Detects both orderings around a known categorical column:
            "<value> <column>"   -- "in the Groceries category"
            "<column> <value>"   -- "status 'refunded'"

        Returns (value, column) pairs that could not be resolved. An empty list
        means every constraint the question names exists in the data.
        """
        tokens = tokenize(question)
        problems: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        # A table name is an anchor too: "in the Dubay region" points at the
        # `regions` table, whose label column holds the vocabulary, even though
        # no column is literally called "region".
        # A column name can precede its value ("status 'refunded'") or follow
        # it ("Groceries category"). A *table* name only ever follows it: "the
        # Dubay region". Reading forward from a table name picks up the verb,
        # which flagged "shipments were carried by Maersk" as an unknown value
        # 'carried'.
        col_anchors: dict[str, list[tuple[TableRef, str]]] = {}
        for variant, entries in self._column_index.items():
            cat = [
                (t, c) for (t, c) in entries
                if self.role_of(t, c) is ColumnRole.CATEGORICAL
            ]
            if cat:
                col_anchors.setdefault(variant, []).extend(cat)

        table_anchors: dict[str, list[tuple[TableRef, str]]] = {}
        for variant, table in self._table_index.items():
            profile = self.profiles.get(table.qualified)
            if not profile:
                continue
            cat = [
                (table, name)
                for name, prof in profile.columns.items()
                if prof.role is ColumnRole.CATEGORICAL
            ]
            if cat:
                table_anchors.setdefault(variant, []).extend(cat)

        for anchors, both_directions in ((col_anchors, True), (table_anchors, False)):
            for variant, cat in anchors.items():
                vt = variant.split()
                n = len(vt)
                for i in range(len(tokens) - n + 1):
                    if tokens[i : i + n] != vt:
                        continue
                    phrases = [self._phrase_before(tokens, i)]
                    if both_directions:
                        phrases.append(self._phrase_after(tokens, i + n))
                    for phrase in phrases:
                        if not phrase or not _looks_like_a_value(phrase):
                            continue
                        if self._resolves(phrase, cat):
                            continue
                        key = (phrase, variant)
                        if key not in seen:
                            seen.add(key)
                            problems.append(key)
        return problems

    def unknown_entities(self, question: str) -> list[str]:
        """Entity types the question counts or selects that do not exist.

        "How many suppliers deliver to the Dammam region?" names a real region
        but there is no suppliers table. Without this, the generator falls back
        to whatever table it can find and answers about something else.
        """
        out: list[str] = []
        for match in _ENTITY_REQUEST.finditer(question):
            raw = next((g for g in match.groups() if g), None)
            if not raw:
                continue
            phrase = " ".join(
                t for t in tokenize(raw) if t not in _FUNCTION_WORDS
            )
            if not phrase or phrase in _META_ATTRIBUTES:
                continue
            if self._names_something(phrase):
                continue
            out.append(phrase)
        return out

    def date_range_outside_data(
        self, lo: str, hi: str
    ) -> bool:
        """True if [lo, hi] lies entirely outside every date column's range.

        The profile knows the observed min and max of each date column, so a
        question about 2026 over a corpus spanning 2023-2024 is answerable only
        as "there is no such data".
        """
        seen_any = False
        for profile in self.profiles.values():
            for prof in profile.columns.values():
                if prof.role is not ColumnRole.DATE:
                    continue
                lo_v, hi_v = prof.min_value, prof.max_value
                if lo_v is None or hi_v is None:
                    continue
                seen_any = True
                if not (hi < str(lo_v)[:10] or lo > str(hi_v)[:10]):
                    return False
        return seen_any

    def unresolved_attributes(self, question: str) -> list[str]:
        """Attributes the question asks for that no column provides.

        "What is the shipping weight of product X" names a real product but an
        attribute the schema does not have. Retrieval happily returns the
        product row and a generator will compose *something* from it — an
        answer to a question nobody asked.

        Only fires on explicit attribute-request phrasings, and only when the
        attribute resolves to no column and no table. Schema questions ("what
        columns does X have") must be excluded by the caller, since their
        attribute word is deliberately meta.
        """
        out: list[str] = []
        scope = self.match_tables(question)
        for match in _ATTR_REQUEST.finditer(question):
            raw = next(
                (
                    g
                    for g in (
                        match.group("attr"),
                        match.group("attr2"),
                        match.group("attr3"),
                        match.group("attr4"),
                    )
                    if g
                ),
                None,
            )
            if not raw:
                continue
            phrase = " ".join(
                t for t in tokenize(raw) if t not in _FUNCTION_WORDS
            )
            if not phrase or phrase in _META_ATTRIBUTES:
                continue
            if self._names_something(phrase, scope):
                continue
            out.append(phrase)
        return out

    def _names_something(
        self, phrase: str, scope: Sequence[TableRef] = ()
    ) -> bool:
        """True if any sub-phrase names a known column or table.

        When the question names an entity type ("...of the product 'X'"), the
        attribute must exist on *that* table. Checking globally accepts "in
        which country is the product manufactured" purely because
        `regions.country` exists — the attribute is real, just not for the
        thing being asked about.
        """
        words = phrase.split()
        allowed = {t.qualified for t in scope}
        for size in range(len(words), 0, -1):
            for start in range(len(words) - size + 1):
                gram = " ".join(words[start : start + size])
                if gram in self._table_index:
                    return True
                entries = self._column_index.get(gram)
                if not entries:
                    continue
                if not allowed:
                    return True
                if any(t.qualified in allowed for t, _ in entries):
                    return True
        return False

    @staticmethod
    def _phrase_before(tokens: Sequence[str], idx: int) -> str:
        out: list[str] = []
        j = idx - 1
        while j >= 0 and len(out) < 4 and tokens[j] not in _FUNCTION_WORDS:
            out.insert(0, tokens[j])
            j -= 1
        return " ".join(out)

    @staticmethod
    def _phrase_after(tokens: Sequence[str], idx: int) -> str:
        out: list[str] = []
        j = idx
        while j < len(tokens) and tokens[j] in _COPULAS:
            j += 1
        while j < len(tokens) and len(out) < 4 and tokens[j] not in _FUNCTION_WORDS:
            out.append(tokens[j])
            j += 1
        return " ".join(out)

    def _resolves(self, phrase: str, cat: Sequence[tuple[TableRef, str]]) -> bool:
        """True if `phrase`, or any contiguous sub-phrase, is a known value."""
        words = phrase.split()
        cols = {(t.qualified, c) for t, c in cat}
        for size in range(len(words), 0, -1):
            for start in range(len(words) - size + 1):
                gram = " ".join(words[start : start + size])
                for table, column, _ in self._value_index.get(gram, ()):
                    if (table.qualified, column) in cols:
                        return True
        return False

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
