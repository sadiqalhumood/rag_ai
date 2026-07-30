"""Embedder selection with explicit, recorded fallback.

`get_embedder("auto")` prefers the local sentence-transformers model and falls
back to hashing. The fallback is never silent: it is written to
`FALLBACK_REASON` and surfaced through `EmbedderInfo.degraded`.
"""

from __future__ import annotations

import os
from typing import Any

from ..core.errors import ConfigError
from .base import BaseEmbedder, EmbedderInfo, normalize
from .hashing import HashingEmbedder

#: Populated when auto-selection had to fall back, for BLOCKERS.md / HANDOFF.md.
FALLBACK_REASON: str | None = None

__all__ = [
    "BaseEmbedder",
    "EmbedderInfo",
    "HashingEmbedder",
    "get_embedder",
    "normalize",
    "FALLBACK_REASON",
]


def get_embedder(name: str = "auto", **kwargs: Any) -> BaseEmbedder:
    global FALLBACK_REASON
    name = (name or "auto").lower()

    if name == "hashing":
        return HashingEmbedder(**kwargs)

    if name == "remote":
        from .remote import RemoteEmbedder

        return RemoteEmbedder(**kwargs)

    if name == "local":
        from .local import LocalEmbedder

        return LocalEmbedder(**kwargs)

    if name != "auto":
        raise ConfigError(f"unknown embedder {name!r}")

    if os.getenv("ANYRAG_FORCE_HASHING_EMBEDDER"):
        FALLBACK_REASON = "ANYRAG_FORCE_HASHING_EMBEDDER was set"
        return HashingEmbedder(**kwargs)

    try:
        from .local import LocalEmbedder

        return LocalEmbedder(**kwargs)
    except Exception as exc:
        FALLBACK_REASON = f"{type(exc).__name__}: {exc}"
        emb = HashingEmbedder()
        emb.info = EmbedderInfo(
            name="hashing",
            model=emb.info.model,
            dim=emb.dim,
            degraded=True,
            detail=f"fell back from local model: {FALLBACK_REASON}",
        )
        return emb
