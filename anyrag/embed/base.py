"""Embedder base types and provenance.

Provenance is not decoration. A fallback embedder keeps the eval *runnable* but
makes its numbers non-comparable, so `EmbedderInfo` travels with every results
JSON and is stamped into the ablation table itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class EmbedderInfo:
    """Which embedder actually produced a set of vectors."""

    name: str
    model: str
    dim: int
    revision: str = ""
    #: True when this is a fallback rather than the intended model. Any metric
    #: computed with a degraded embedder must be labelled as such at the point
    #: of display, not only in prose.
    degraded: bool = False
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def label(self) -> str:
        base = f"{self.name}:{self.model}"
        if self.revision:
            base += f"@{self.revision[:12]}"
        return base + (" (DEGRADED)" if self.degraded else "")


class BaseEmbedder:
    """Shared plumbing: batching and L2 normalisation.

    Vectors are unit-norm so that dot product is cosine similarity, which lets
    every index implementation use a single scoring path.
    """

    info: EmbedderInfo
    dim: int

    def _encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._encode(texts)
        return normalize(np.asarray(vecs, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalise rows, leaving all-zero rows untouched (no NaNs)."""
    if v.ndim == 1:
        n = float(np.linalg.norm(v))
        return v if n == 0.0 else (v / n).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (v / norms).astype(np.float32)
