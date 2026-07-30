"""Optional remote embedder, behind the same interface.

Never used by the offline eval path. It exists to demonstrate that the provider
seam is real: swapping in a hosted model changes one config value, not any
calling code. Requires network and an API key, so it is env-gated and imports
its client lazily.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from ..core.errors import ConfigError
from ..core.registry import register_embedder
from .base import BaseEmbedder, EmbedderInfo


@register_embedder("remote")
class RemoteEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str | None = None,
        api_key_env: str = "ANYRAG_EMBED_API_KEY",
        endpoint_env: str = "ANYRAG_EMBED_ENDPOINT",
        dim: int = 1536,
    ) -> None:
        self._api_key = os.getenv(api_key_env)
        self._endpoint = os.getenv(endpoint_env)
        if not self._api_key or not self._endpoint:
            raise ConfigError(
                f"remote embedder needs {api_key_env} and {endpoint_env} to be set"
            )
        self._model = model_name or os.getenv("ANYRAG_EMBED_MODEL", "unset")
        self.dim = dim
        self.info = EmbedderInfo(
            name="remote",
            model=self._model,
            dim=dim,
            degraded=False,
            detail=f"endpoint from ${endpoint_env}",
        )

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        import json
        import urllib.request

        req = urllib.request.Request(
            self._endpoint,
            data=json.dumps({"model": self._model, "input": list(texts)}).encode(),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read())
        rows = [item["embedding"] for item in payload["data"]]
        return np.asarray(rows, dtype=np.float32)
