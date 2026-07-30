"""Column profiling and role inference, shared by every source adapter.

The division of labour is deliberate: an adapter is responsible only for
producing (a) an exact row count and (b) a bounded set of rows. Everything
statistical -- null fractions, cardinality, value vocabularies, and the
`ColumnRole` decision -- happens here, in pure Python, over those rows.

That is why `sqlite.py` and `postgres.py` are thin, and why a third adapter
does not get to invent its own idea of what CATEGORICAL means. Downstream code
routes on these roles: ingest decides how to verbalize a column, and the router
decides whether a question named a categorical value or a free-text span. Two
adapters disagreeing about `status` would be a silent retrieval bug.

Thresholds are module constants rather than literals so tests can pin them and
an ablation can move them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..core.types import (
    ColumnProfile,
    ColumnRole,
    ColumnSchema,
    Row,
    TableProfile,
    TableRef,
    TableSchema,
)
from .introspect import (
    BOOLEAN,
    DATE,
    NUMERIC_FAMILIES,
    TEMPORAL_FAMILIES,
    TEXTUAL_FAMILIES,
    TIMESTAMP,
    UNKNOWN,
    UUID,
    date_parse_fraction,
    family_from_values,
    looks_like_id_name,
    normalize_type,
)

# --------------------------------------------------------------------------
# Tunable thresholds
# --------------------------------------------------------------------------

#: How many example values to keep on a ColumnProfile.
MAX_SAMPLE_VALUES = 10

#: Hard cap on the CATEGORICAL value vocabulary. A category list longer than
#: this stops being useful as a prompt hint and starts being a token bill.
MAX_CATEGORIES = 50

#: A column is CATEGORICAL only if its distinct count is at or below both an
#: absolute cap and a fraction of the rows examined...
CATEGORICAL_MAX_DISTINCT = MAX_CATEGORIES
CATEGORICAL_MAX_RATIO = 0.2
#: ...with a floor, because on a 12-row table `0.2 * 12 = 2.4` would classify
#: essentially nothing as categorical. Small tables are the common case in
#: tests and demos, and the ratio rule alone is unusable there.
CATEGORICAL_SMALL_TABLE_FLOOR = 8
#: Long values are not category labels, however few of them there are.
CATEGORICAL_MAX_MEAN_LENGTH = 64

#: An `*_id` column is only an ID if it is nearly unique. A foreign key like
#: `orders.customer_id` repeats, and would fail this -- see `infer_role` for
#: why declared foreign keys are handled separately.
ID_MIN_CARDINALITY_RATIO = 0.95

#: Fraction of non-null sampled values that must parse as dates before a text
#: column is called DATE.
DATE_MIN_PARSE_FRACTION = 0.8

#: Values (case-folded) that a two-valued column may contain to be BOOLEAN.
BOOLEAN_TOKENS = frozenset(
    {"0", "1", "true", "false", "t", "f", "yes", "no", "y", "n"}
)

#: Upper bound on the distinct-value set we track per column, so profiling a
#: wide table of unique strings cannot blow up memory.
DISTINCT_TRACK_LIMIT = 100_000


# --------------------------------------------------------------------------
# Accumulation
# --------------------------------------------------------------------------


def _hashable(value: Any) -> Any:
    """A stable hash key for a value, falling back to its repr.

    JSON columns hand back dicts and lists; bytes are hashable but unbounded.
    Neither should crash a profile run.
    """
    try:
        hash(value)
    except TypeError:
        return ("__unhashable__", repr(value))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ("__bytes__", bytes(value)[:64])
    return value


@dataclass
class ColumnAccumulator:
    """Single-pass statistics for one column."""

    name: str
    total: int = 0
    nulls: int = 0
    length_sum: int = 0
    length_n: int = 0
    _distinct: set[Any] = field(default_factory=set)
    _distinct_saturated: bool = False
    _samples: list[Any] = field(default_factory=list)
    _sample_keys: set[Any] = field(default_factory=set)
    #: Kept for value-level inference (date sniffing, boolean tokens). Bounded.
    _values: list[Any] = field(default_factory=list)

    #: How many non-null values to retain for value-level inference.
    value_budget: int = 2000

    def add(self, value: Any) -> None:
        self.total += 1
        if value is None:
            self.nulls += 1
            return
        key = _hashable(value)
        if not self._distinct_saturated:
            self._distinct.add(key)
            if len(self._distinct) >= DISTINCT_TRACK_LIMIT:
                self._distinct_saturated = True
        if len(self._samples) < MAX_SAMPLE_VALUES and key not in self._sample_keys:
            self._samples.append(value)
            self._sample_keys.add(key)
        if len(self._values) < self.value_budget:
            self._values.append(value)
        try:
            self.length_sum += len(str(value))
            self.length_n += 1
        except Exception:  # pragma: no cover - str() of a hostile object
            pass

    # -- derived ----------------------------------------------------------

    @property
    def non_null(self) -> int:
        return self.total - self.nulls

    @property
    def distinct_count(self) -> int:
        return len(self._distinct)

    @property
    def mean_length(self) -> float | None:
        return (self.length_sum / self.length_n) if self.length_n else None

    @property
    def values(self) -> list[Any]:
        return self._values

    @property
    def samples(self) -> tuple[Any, ...]:
        return tuple(self._samples)

    def min_max(self) -> tuple[Any, Any]:
        """Min/max over distinct values, or (None, None) if incomparable.

        SQLite columns can legitimately hold mixed types, and comparing a str
        to an int raises. A profile that crashes on messy data is worse than a
        profile with a missing range.
        """
        vals = [v for v in self._distinct if v is not None and not isinstance(v, tuple)]
        if not vals:
            return (None, None)
        try:
            return (min(vals), max(vals))
        except TypeError:
            return (None, None)

    def categories(self, limit: int = MAX_CATEGORIES) -> tuple[str, ...]:
        """Sorted, bounded string vocabulary of the observed values."""
        if self._distinct_saturated:
            return ()
        out = sorted({str(v) for v in self._distinct if not isinstance(v, tuple)})
        return tuple(out[:limit])


# --------------------------------------------------------------------------
# Role inference
# --------------------------------------------------------------------------


def _categorical_limit(non_null: int) -> int:
    """Largest distinct count still considered categorical for `non_null` rows."""
    return min(
        CATEGORICAL_MAX_DISTINCT,
        max(CATEGORICAL_SMALL_TABLE_FLOOR, int(CATEGORICAL_MAX_RATIO * non_null)),
    )


def is_categorical(
    *, distinct_count: int, non_null: int, mean_length: float | None
) -> bool:
    if non_null <= 0 or distinct_count <= 0:
        return False
    if distinct_count >= non_null:
        # Every value unique: a vocabulary, not a category set.
        return False
    if mean_length is not None and mean_length > CATEGORICAL_MAX_MEAN_LENGTH:
        return False
    return distinct_count <= _categorical_limit(non_null)


def _boolean_by_value(values: Sequence[Any], distinct_count: int) -> bool:
    if distinct_count != 2 or not values:
        return False
    tokens = {str(v).strip().lower() for v in values if v is not None}
    if len(tokens) != 2:
        return False
    return tokens <= BOOLEAN_TOKENS


def infer_role(
    *,
    name: str,
    type_name: str | None,
    values: Sequence[Any] = (),
    total_count: int = 0,
    null_count: int = 0,
    distinct_count: int = 0,
    mean_length: float | None = None,
    is_primary_key: bool = False,
    references: str | None = None,
) -> ColumnRole:
    """Decide what a column *means* from its declared type plus statistics.

    Precedence is ID -> BOOLEAN -> DATE -> NUMERIC -> CATEGORICAL -> FREE_TEXT,
    with UNKNOWN reserved for columns that are genuinely undeterminable
    (binary blobs, exotic types, untyped columns with no observed values).

    One deliberate extension beyond "primary key or near-unique `*_id`": a
    column that is a *declared foreign key* and is named like an id is ID even
    though it repeats. `orders.customer_id` is a key, not a measurement, and
    calling it NUMERIC would invite downstream code to average it.
    """
    family = normalize_type(type_name)
    if family == UNKNOWN:
        family = family_from_values(values)

    non_null = max(0, total_count - null_count)
    ratio = (distinct_count / non_null) if non_null else 0.0

    # 1. ID
    if is_primary_key:
        return ColumnRole.ID
    if looks_like_id_name(name):
        if references:
            return ColumnRole.ID
        if (
            family in NUMERIC_FAMILIES or family == UUID
        ) and non_null and ratio >= ID_MIN_CARDINALITY_RATIO:
            return ColumnRole.ID

    # 2. BOOLEAN
    if family == BOOLEAN:
        return ColumnRole.BOOLEAN
    if _boolean_by_value(values, distinct_count):
        return ColumnRole.BOOLEAN

    # 3. DATE  (TIME alone is not a calendar date and stays out)
    if family in (DATE, TIMESTAMP):
        return ColumnRole.DATE
    if family in TEXTUAL_FAMILIES and values:
        if date_parse_fraction(values) >= DATE_MIN_PARSE_FRACTION:
            return ColumnRole.DATE

    # 4. NUMERIC
    if family in NUMERIC_FAMILIES:
        return ColumnRole.NUMERIC

    # 5. CATEGORICAL
    if family in TEXTUAL_FAMILIES or family in TEMPORAL_FAMILIES:
        if is_categorical(
            distinct_count=distinct_count, non_null=non_null, mean_length=mean_length
        ):
            return ColumnRole.CATEGORICAL

    # 6. FREE_TEXT
    if family in TEXTUAL_FAMILIES:
        return ColumnRole.FREE_TEXT

    # 7. Genuinely undeterminable: blobs, geometry, empty untyped columns.
    return ColumnRole.UNKNOWN


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_column_profile(
    acc: ColumnAccumulator, column: ColumnSchema | None = None
) -> ColumnProfile:
    mean_length = acc.mean_length
    role = infer_role(
        name=acc.name,
        type_name=column.type_name if column else None,
        values=acc.values,
        total_count=acc.total,
        null_count=acc.nulls,
        distinct_count=acc.distinct_count,
        mean_length=mean_length,
        is_primary_key=bool(column and column.is_primary_key),
        references=column.references if column else None,
    )
    lo, hi = acc.min_max()
    categories: tuple[str, ...] = ()
    if role in (ColumnRole.CATEGORICAL, ColumnRole.BOOLEAN):
        categories = acc.categories(MAX_CATEGORIES)
    return ColumnProfile(
        name=acc.name,
        role=role,
        total_count=acc.total,
        null_count=acc.nulls,
        distinct_count=acc.distinct_count,
        samples=acc.samples,
        min_value=lo,
        max_value=hi,
        mean_length=mean_length,
        categories=categories,
    )


def build_table_profile(
    *,
    table: TableRef,
    schema: TableSchema,
    row_count: int,
    rows: Iterable[Row],
) -> TableProfile:
    """Profile `table` from an already-fetched, bounded set of `rows`.

    `TableProfile.row_count` is the exact table cardinality supplied by the
    adapter. Each `ColumnProfile.total_count` is the number of rows actually
    *examined*, which equals `row_count` for small tables and the sample size
    for large ones -- so `null_fraction` and `cardinality_ratio` stay
    meaningful either way, and never silently claim a full scan happened.
    """
    columns_by_name: Mapping[str, ColumnSchema] = {c.name: c for c in schema.columns}
    accs = {name: ColumnAccumulator(name) for name in columns_by_name}

    for row in rows:
        for name, acc in accs.items():
            acc.add(row[name] if name in row else None)

    profiles = {
        name: build_column_profile(acc, columns_by_name.get(name))
        for name, acc in accs.items()
    }
    return TableProfile(table=table, row_count=row_count, columns=profiles)


__all__ = [
    "BOOLEAN_TOKENS",
    "CATEGORICAL_MAX_DISTINCT",
    "CATEGORICAL_MAX_MEAN_LENGTH",
    "CATEGORICAL_MAX_RATIO",
    "CATEGORICAL_SMALL_TABLE_FLOOR",
    "DATE_MIN_PARSE_FRACTION",
    "ID_MIN_CARDINALITY_RATIO",
    "MAX_CATEGORIES",
    "MAX_SAMPLE_VALUES",
    "ColumnAccumulator",
    "build_column_profile",
    "build_table_profile",
    "infer_role",
    "is_categorical",
]
