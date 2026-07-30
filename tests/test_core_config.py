"""Config invariants the ablation harness depends on."""

from __future__ import annotations

import pytest

from anyrag.core.config import MetadataFilter, RetrievalConfig
from anyrag.core.errors import ConfigError
from anyrag.core.types import Chunk, ChunkKind


def _chunk(**kw) -> Chunk:
    base = dict(
        chunk_id="c1",
        source_id="sqlite:test",
        kind=ChunkKind.ROW,
        text="t",
        meta={"table": "orders"},
    )
    base.update(kw)
    return Chunk(**base)  # type: ignore[arg-type]


def test_config_names_are_stable_and_unique() -> None:
    """Ablation rows are keyed by name; collisions would merge grid cells."""
    names = set()
    for dense, lexical in ((True, False), (False, True), (True, True)):
        for rerank in (False, True):
            for kinds in (
                frozenset({ChunkKind.ROW}),
                frozenset({ChunkKind.SCHEMA_CARD}),
                frozenset({ChunkKind.ROW, ChunkKind.SCHEMA_CARD}),
            ):
                cfg = RetrievalConfig(
                    dense=dense, lexical=lexical, rerank=rerank, chunk_kinds=kinds
                )
                names.add(cfg.name)
    assert len(names) == 18


def test_disabling_both_retrievers_is_rejected() -> None:
    with pytest.raises(ConfigError):
        RetrievalConfig(dense=False, lexical=False)


def test_nonpositive_k_rejected() -> None:
    with pytest.raises(ConfigError):
        RetrievalConfig(k=0)
    with pytest.raises(ConfigError):
        RetrievalConfig(candidate_k=-1)


def test_config_naming_covers_modes() -> None:
    assert RetrievalConfig(dense=True, lexical=False).name.startswith("dense")
    assert RetrievalConfig(dense=False, lexical=True).name.startswith("bm25")
    assert RetrievalConfig(dense=True, lexical=True).name.startswith("hybrid")


def test_with_returns_a_modified_copy() -> None:
    cfg = RetrievalConfig()
    assert cfg.with_(k=99).k == 99
    assert cfg.k == 10


def test_metadata_filter_matches_kinds_and_tables() -> None:
    flt = MetadataFilter(kinds=frozenset({ChunkKind.ROW}))
    assert flt.matches(_chunk())
    assert not flt.matches(_chunk(kind=ChunkKind.SCHEMA_CARD))

    flt = MetadataFilter(tables=frozenset({"orders"}))
    assert flt.matches(_chunk())
    assert not flt.matches(_chunk(meta={"table": "customers"}))


def test_metadata_filter_conditions_are_anded() -> None:
    flt = MetadataFilter(
        kinds=frozenset({ChunkKind.ROW}), tables=frozenset({"customers"})
    )
    assert not flt.matches(_chunk())


def test_empty_filter_is_reported_empty() -> None:
    assert MetadataFilter().is_empty
    assert not MetadataFilter(tables=frozenset({"t"})).is_empty


def test_equals_filter_on_arbitrary_meta() -> None:
    flt = MetadataFilter(equals={"region": "EMEA"})
    assert flt.matches(_chunk(meta={"region": "EMEA"}))
    assert not flt.matches(_chunk(meta={"region": "APAC"}))
