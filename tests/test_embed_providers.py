"""Embedder contract and provenance tests.

The provenance assertions matter as much as the vector ones: a degraded
embedder that fails to announce itself would silently poison every reported
number.
"""

from __future__ import annotations

import numpy as np
import pytest

from anyrag.embed import get_embedder
from anyrag.embed.base import normalize
from anyrag.embed.hashing import HashingEmbedder


def test_hashing_embedder_is_deterministic_across_instances() -> None:
    """Must not depend on Python's per-process salted hash()."""
    a = HashingEmbedder().embed(["hello world"])
    b = HashingEmbedder().embed(["hello world"])
    np.testing.assert_allclose(a, b)


def test_hashing_embedder_declares_itself_degraded() -> None:
    assert HashingEmbedder().info.degraded is True
    assert "DEGRADED" in HashingEmbedder().info.label


def test_vectors_are_unit_norm() -> None:
    vecs = HashingEmbedder().embed(["alpha beta", "gamma delta"])
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-5)


def test_empty_batch_returns_correct_shape() -> None:
    emb = HashingEmbedder()
    out = emb.embed([])
    assert out.shape == (0, emb.dim)


def test_empty_string_does_not_produce_nan() -> None:
    vecs = HashingEmbedder().embed(["", "real text"])
    assert not np.isnan(vecs).any()


def test_lexical_overlap_beats_unrelated_for_hashing() -> None:
    emb = HashingEmbedder()
    v = emb.embed(
        ["customer Ahmed Al-Sayed from Riyadh", "customer Ahmed Al-Sayed", "photosynthesis"]
    )
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


def test_embed_query_matches_embed_batch() -> None:
    emb = HashingEmbedder()
    np.testing.assert_allclose(emb.embed_query("hello"), emb.embed(["hello"])[0])


def test_normalize_leaves_zero_vector_alone() -> None:
    z = np.zeros((2, 4), dtype=np.float32)
    out = normalize(z)
    assert not np.isnan(out).any()


def test_auto_embedder_reports_provenance() -> None:
    emb = get_embedder("auto")
    info = emb.info
    assert info.name and info.model and info.dim > 0
    assert isinstance(info.degraded, bool)


@pytest.mark.parametrize("name", ["hashing", "auto"])
def test_embedders_conform_to_protocol(name: str) -> None:
    emb = get_embedder(name)
    vecs = emb.embed(["a", "b"])
    assert vecs.shape == (2, emb.dim)


def test_unknown_embedder_name_raises() -> None:
    from anyrag.core.errors import ConfigError

    with pytest.raises(ConfigError):
        get_embedder("no-such-embedder")


def test_local_embedder_semantics_when_available() -> None:
    """The real model must rank a paraphrase above an unrelated sentence."""
    emb = get_embedder("auto")
    if emb.info.degraded:
        pytest.skip("local model unavailable; running degraded")
    v = emb.embed(
        [
            "the cat sat on the mat",
            "a feline rested on the rug",
            "quarterly revenue rose 12% in Q3",
        ]
    )
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])
