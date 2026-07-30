"""Shared contracts. Orchestrator-owned; subsystem packages import, never edit."""

from __future__ import annotations

from .config import AnyRagConfig, GenerationConfig, MetadataFilter, RetrievalConfig
from .errors import (
    AnyRagError,
    BudgetExceededError,
    CitationError,
    ConfigError,
    SchemaError,
    SourceError,
    SqlGenerationError,
    UnsafeQueryError,
)
from .lint import is_readonly, lint_readonly
from .tokenizer import count_tokens, get_tokenizer, truncate_to_tokens
from .types import (
    Answer,
    Chunk,
    ChunkKind,
    Citation,
    ColumnProfile,
    ColumnRole,
    ColumnSchema,
    Hit,
    QueryResult,
    QueryRoute,
    Row,
    RowRef,
    TableProfile,
    TableRef,
    TableSchema,
)

__all__ = [
    "AnyRagConfig", "GenerationConfig", "MetadataFilter", "RetrievalConfig",
    "AnyRagError", "BudgetExceededError", "CitationError", "ConfigError",
    "SchemaError", "SourceError", "SqlGenerationError", "UnsafeQueryError",
    "is_readonly", "lint_readonly",
    "count_tokens", "get_tokenizer", "truncate_to_tokens",
    "Answer", "Chunk", "ChunkKind", "Citation", "ColumnProfile", "ColumnRole",
    "ColumnSchema", "Hit", "QueryResult", "QueryRoute", "Row", "RowRef",
    "TableProfile", "TableRef", "TableSchema",
]
