"""Dependency-free fallback embedder.

Hashed character n-grams projected into a fixed-width space. This is a real
(weak) semantic signal: it captures lexical overlap and morphology, which is
enough for the pipeline to run end to end, and it is fully deterministic across
processes -- unlike Python's salted `hash()`.

It exists so that a failed model download degrades result *quality* rather than
breaking the eval harness. Anything it produces is marked degraded.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

from ..core.registry import register_embedder
from .base import BaseEmbedder, EmbedderInfo

_WORD = re.compile(r"\w+", re.UNICODE)


@register_embedder("hashing")
class HashingEmbedder(BaseEmbedder):
    def __init__(self, dim: int = 384, ngram: tuple[int, int] = (3, 5)) -> None:
        self.dim = dim
        self._lo, self._hi = ngram
        self.info = EmbedderInfo(
            name="hashing",
            model=f"charngram-{self._lo}-{self._hi}",
            dim=dim,
            degraded=True,
            detail="deterministic hashing fallback; not a learned model",
        )

    @staticmethod
    def _bucket(token: str, dim: int) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        val = int.from_bytes(digest, "big")
        # Low bit picks the sign, which keeps unrelated tokens from all adding
        # constructively into the same direction.
        return val % dim, 1.0 if (val >> 63) & 1 else -1.0

    def _features(self, text: str) -> list[str]:
        text = text.lower()
        feats: list[str] = []
        for word in _WORD.findall(text):
            feats.append(f"w:{word}")
            padded = f"^{word}$"
            for n in range(self._lo, self._hi + 1):
                if len(padded) < n:
                    continue
                for i in range(len(padded) - n + 1):
                    feats.append(f"g:{padded[i : i + n]}")
        return feats

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for feat in self._features(text or ""):
                idx, sign = self._bucket(feat, self.dim)
                out[i, idx] += sign
        return out
