"""Frozen data contracts shared by every anyrag component.

Nothing in this module knows about embeddings, indexes, or retrieval. It is the
vocabulary that lets six independently-written subsystems agree on what a table,
a row, a chunk, and an answer are.

Ownership note: this file is orchestrator-owned. Subsystem packages import from
here and never edit it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .errors import CitationError, SchemaError

# --------------------------------------------------------------------------
# Sources: tables, schemas, profiles
# --------------------------------------------------------------------------


class ColumnRole(str, Enum):
    """What a column *means*, as opposed to how it is stored.

    Inferred by the source adapter from declared type plus column statistics.
    Ingest uses this to decide how to verbalize a column; the router uses it to
    decide whether a question names a categorical value or a free-text span.
    """

    FREE_TEXT = "free_text"
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    DATE = "date"
    ID = "id"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


@dataclass(frozen=True, order=True)
class TableRef:
    """Identifies a table within a source. `schema` is None for SQLite/files."""

    name: str
    schema: str | None = None

    @property
    def qualified(self) -> str:
        return f"{self.schema}.{self.name}" if self.schema else self.name

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.qualified


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    type_name: str
    nullable: bool = True
    is_primary_key: bool = False
    #: "other_table.other_column" when this column is a foreign key.
    references: str | None = None


@dataclass(frozen=True)
class TableSchema:
    table: TableRef
    columns: tuple[ColumnSchema, ...]
    primary_key: tuple[str, ...] = ()

    def column(self, name: str) -> ColumnSchema:
        for c in self.columns:
            if c.name == name:
                return c
        raise SchemaError(f"no column {name!r} on {self.table.qualified}")

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


@dataclass(frozen=True)
class ColumnProfile:
    """Per-column statistics, computed by sampling at introspection time."""

    name: str
    role: ColumnRole
    total_count: int = 0
    null_count: int = 0
    distinct_count: int = 0
    samples: tuple[Any, ...] = ()
    min_value: Any | None = None
    max_value: Any | None = None
    mean_length: float | None = None
    #: For CATEGORICAL columns: the observed value vocabulary (bounded).
    categories: tuple[str, ...] = ()

    @property
    def null_fraction(self) -> float:
        return (self.null_count / self.total_count) if self.total_count else 0.0

    @property
    def cardinality_ratio(self) -> float:
        return (self.distinct_count / self.total_count) if self.total_count else 0.0


@dataclass(frozen=True)
class TableProfile:
    table: TableRef
    row_count: int
    columns: Mapping[str, ColumnProfile] = field(default_factory=dict)

    def role(self, column: str) -> ColumnRole:
        prof = self.columns.get(column)
        return prof.role if prof else ColumnRole.UNKNOWN

    def columns_with_role(self, role: ColumnRole) -> tuple[str, ...]:
        return tuple(n for n, p in self.columns.items() if p.role is role)


#: A single record as returned by a source. Keys are column names.
Row = Mapping[str, Any]


@dataclass(frozen=True)
class QueryResult:
    """The outcome of a linted, read-only query."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    sql: str = ""
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def scalar(self) -> Any:
        """The single value of a 1x1 result, else None.

        Aggregate questions overwhelmingly produce 1x1 results; this keeps the
        happy path readable without hiding the general case.
        """
        if len(self.rows) == 1 and len(self.rows[0]) == 1:
            return self.rows[0][0]
        return None

    def as_dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, r)) for r in self.rows]


# --------------------------------------------------------------------------
# Chunks
# --------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class RowRef:
    """A pointer from a chunk back to the source row(s) that produced it.

    This is the load-bearing field for evaluation: the eval harness generated
    the data, so it knows the gold row IDs for every question, and grades
    retrieval by intersecting gold RowRefs with the RowRefs of retrieved chunks.

    `pk` is always a *string*. Sources disagree about types for the same logical
    key (SQLite INTEGER 7, CSV "7", Postgres bigint 7), and gold matching must
    not depend on which adapter happened to load the data.
    """

    table: str
    pk: str

    def __post_init__(self) -> None:
        if not isinstance(self.pk, str):
            object.__setattr__(self, "pk", str(self.pk))

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.table}#{self.pk}"


class ChunkKind(str, Enum):
    ROW = "row"
    SCHEMA_CARD = "schema_card"
    SQL_RESULT = "sql_result"


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text plus its provenance.

    `chunk_id` must be deterministic across re-ingestion of unchanged data --
    see anyrag.ingest.ids. Content is deliberately NOT part of the id; the
    content hash lives in `meta["content_hash"]` for change detection.
    """

    chunk_id: str
    source_id: str
    kind: ChunkKind
    text: str
    row_refs: tuple[RowRef, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def table(self) -> str | None:
        return self.meta.get("table")

    @property
    def content_hash(self) -> str | None:
        return self.meta.get("content_hash")


@dataclass(frozen=True)
class Hit:
    """A chunk retrieved by some retriever, with its score."""

    chunk: Chunk
    score: float
    rank: int = 0
    #: Which retriever produced this hit ("dense", "bm25", "rrf", "rerank", ...).
    retriever: str = ""

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


# --------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------


class QueryRoute(str, Enum):
    LOOKUP = "lookup"
    AGGREGATE = "aggregate"
    HYBRID = "hybrid"


# Documented keys for `Answer.trace`. These are a contract, not an internal
# convention: the eval harness reads them to score SQL coverage, and if the
# producer and the reader disagree on a key name, coverage silently reads 0%
# instead of failing loudly.
TRACE_SQL = "sql"                 # the generated SQL string, if any
TRACE_SQL_ERROR = "sql_error"     # why SQL generation or execution failed
TRACE_SCALAR = "scalar"           # the aggregate result (mirrors Answer.value)
TRACE_ROUTE = "route"             # QueryRoute value as a string
TRACE_ROUTE_REASON = "route_reason"
TRACE_RETRIEVAL = "retrieval"     # per-stage retrieval trace
TRACE_REFUSAL = "refusal"         # structured refusal decision


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    source_id: str
    row_refs: tuple[RowRef, ...] = ()
    score: float = 0.0
    quoted_span: str = ""
    #: Kind of the cited chunk. A schema-card citation legitimately has no row
    #: refs, so without this a scorer cannot tell "cited a schema card" from
    #: "cited nothing useful" and has to cross-reference the retrieval result
    #: to find out.
    kind: ChunkKind | None = None


@dataclass(frozen=True)
class Answer:
    """A generated answer.

    Invariant enforced in the constructor: a non-refusal must carry at least one
    citation. An uncited answer is a failure, so it is made unconstructible
    rather than merely discouraged.
    """

    text: str
    citations: tuple[Citation, ...] = ()
    refused: bool = False
    reason: str = ""
    route: QueryRoute | None = None
    trace: Mapping[str, Any] = field(default_factory=dict)
    #: The computed result of an AGGREGATE question, when there is one.
    #:
    #: Without a typed field, a scorer has to regex the answer prose for a
    #: number -- and "shipped 3 of 12 items" parses to 3. An aggregate result
    #: is structured data and should not survive only as text.
    value: Any | None = None

    def __post_init__(self) -> None:
        if not self.refused and not self.citations:
            raise CitationError(
                "answer is not a refusal but carries zero citations; "
                "either cite retrieved chunks or refuse explicitly"
            )

    @classmethod
    def refusal(
        cls,
        reason: str,
        *,
        route: QueryRoute | None = None,
        text: str = "",
        trace: Mapping[str, Any] | None = None,
        citations: Sequence[Citation] = (),
    ) -> "Answer":
        return cls(
            text=text or "I cannot answer that from the available data.",
            citations=tuple(citations),
            refused=True,
            reason=reason,
            route=route,
            trace=dict(trace or {}),
        )

    @property
    def cited_chunk_ids(self) -> tuple[str, ...]:
        return tuple(c.chunk_id for c in self.citations)

    @property
    def cited_row_refs(self) -> tuple[RowRef, ...]:
        out: list[RowRef] = []
        for c in self.citations:
            out.extend(c.row_refs)
        return tuple(out)
