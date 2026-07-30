"""anyrag -- source-agnostic retrieval-augmented generation.

Point it at a database, ask questions in natural language, get answers with
citations back to specific rows.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .core import (  # noqa: F401
    Answer,
    AnyRagConfig,
    Chunk,
    ChunkKind,
    Citation,
    GenerationConfig,
    Hit,
    QueryRoute,
    RetrievalConfig,
    RowRef,
    lint_readonly,
)

__all__ = [
    "__version__",
    "Answer",
    "AnyRagConfig",
    "Chunk",
    "ChunkKind",
    "Citation",
    "GenerationConfig",
    "Hit",
    "QueryRoute",
    "RetrievalConfig",
    "RowRef",
    "lint_readonly",
]
