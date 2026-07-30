"""Local sentence-transformers embedder -- the intended default.

Runs entirely offline once the model is cached, which is what makes the eval
suite network-free.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from ..core.registry import register_embedder
from .base import BaseEmbedder, EmbedderInfo

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#: Pinned so provenance is reproducible; verified in Phase 0.
DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@register_embedder("local")
class LocalEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        # Method was renamed in sentence-transformers 5.x; support both.
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            getattr(self._model, "get_sentence_embedding_dimension", None),
        )
        self.dim = int(get_dim())
        self.info = EmbedderInfo(
            name="local",
            model=model_name,
            dim=self.dim,
            revision=_resolve_revision(model_name),
            degraded=False,
        )

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )


def _resolve_revision(model_name: str) -> str:
    """Best-effort commit hash of the cached snapshot, for provenance."""
    try:
        cache = os.getenv(
            "HF_HUB_CACHE",
            os.path.join(
                os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                "hub",
            ),
        )
        ref = os.path.join(
            cache, "models--" + model_name.replace("/", "--"), "refs", "main"
        )
        with open(ref, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except Exception:
        return DEFAULT_REVISION
