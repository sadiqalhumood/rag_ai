"""Protocols every pluggable component implements.

These are the seams of the system. The source-agnosticism claim reduces to:
`DataSource` is small enough that a new backend is one class and nothing else.

All Protocols are runtime_checkable so tests can assert conformance without
importing concrete implementations.
"""

from __future__ import annotations

from typing import Any, Iterator, Protocol, Sequence, runtime_checkable

from .config import GenerationConfig, MetadataFilter, RetrievalConfig
from .types import (
    Answer,
    Chunk,
    Hit,
    QueryResult,
    Row,
    TableProfile,
    TableRef,
    TableSchema,
)


@runtime_checkable
class DataSource(Protocol):
    """A read-only view over some tabular backend.

    Implementations must never mutate the underlying store, and must route
    every SQL string through `anyrag.core.lint.lint_readonly` before execution.
    """

    source_id: str

    def tables(self) -> list[TableRef]: ...

    def schema(self, table: TableRef) -> TableSchema: ...

    def profile(self, table: TableRef) -> TableProfile: ...

    def iter_rows(self, table: TableRef, batch_size: int = 1000) -> Iterator[list[Row]]: ...

    def execute_readonly(self, sql: str, max_rows: int = 1000) -> QueryResult: ...

    def close(self) -> None: ...


@runtime_checkable
class Embedder(Protocol):
    """Maps text to unit-norm vectors."""

    dim: int

    def embed(self, texts: Sequence[str]) -> Any:  # -> np.ndarray (n, dim)
        ...

    def embed_query(self, text: str) -> Any:  # -> np.ndarray (dim,)
        ...


@runtime_checkable
class VectorIndex(Protocol):
    """Dense index. Incremental update is part of the contract, not a bonus."""

    def upsert(self, chunks: Sequence[Chunk], vectors: Any) -> int: ...

    def delete_by_source(self, source_id: str) -> int: ...

    def delete_by_ids(self, chunk_ids: Sequence[str]) -> int: ...

    def search(
        self, vector: Any, k: int = 10, flt: MetadataFilter | None = None
    ) -> list[Hit]: ...

    def get(self, chunk_id: str) -> Chunk | None: ...

    def count(self) -> int: ...

    def persist(self, path: str | None = None) -> None: ...

    def load(self, path: str | None = None) -> None: ...


@runtime_checkable
class LexicalIndex(Protocol):
    """Sparse/BM25 index, same lifecycle contract as VectorIndex."""

    def upsert(self, chunks: Sequence[Chunk]) -> int: ...

    def delete_by_source(self, source_id: str) -> int: ...

    def delete_by_ids(self, chunk_ids: Sequence[str]) -> int: ...

    def search(
        self, query: str, k: int = 10, flt: MetadataFilter | None = None
    ) -> list[Hit]: ...

    def get(self, chunk_id: str) -> Chunk | None: ...

    def count(self) -> int: ...

    def persist(self, path: str | None = None) -> None: ...

    def load(self, path: str | None = None) -> None: ...


@runtime_checkable
class Chunker(Protocol):
    """Turns source records into chunks."""

    def chunk_table(
        self, source: DataSource, table: TableRef
    ) -> Iterator[Chunk]: ...


@runtime_checkable
class Reranker(Protocol):
    """Reorders candidate hits. `NoopReranker` is the identity."""

    name: str

    def rerank(self, query: str, hits: Sequence[Hit], k: int = 10) -> list[Hit]: ...


@runtime_checkable
class QueryExpander(Protocol):
    def expand(self, query: str) -> list[str]: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, config: RetrievalConfig) -> list[Hit]: ...


@runtime_checkable
class SqlGenerator(Protocol):
    """NL -> SQL for AGGREGATE questions.

    Must raise `SqlGenerationError` rather than returning a guess when the
    question is out of coverage: an unanswerable aggregate has to become a
    refusal, not a plausible wrong number.
    """

    name: str

    def generate(self, question: str, source: DataSource) -> str: ...


@runtime_checkable
class AnswerGenerator(Protocol):
    """Composes a cited answer from packed context, or refuses."""

    name: str

    def generate(
        self,
        question: str,
        hits: Sequence[Hit],
        config: GenerationConfig,
        **kwargs: Any,
    ) -> Answer: ...
