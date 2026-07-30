"""Metadata pre-filter construction.

The end-to-end property (a filtered search still returns `k` results) is tested
in `test_retrieval_pipeline.py` against real indexes. What is tested here is the
translation from a `RetrievalConfig` into the `MetadataFilter` that gets pushed
down -- in particular that the `chunk_kinds` ablation axis is *exact*, since a
"schema cards only" cell that leaked one row chunk would silently make the
comparison meaningless.
"""

from __future__ import annotations

import pytest

from anyrag.core.config import MetadataFilter, RetrievalConfig
from anyrag.core.types import ChunkKind
from anyrag.retrieval.prefilter import (
    ALL_KINDS,
    build_filter,
    coerce_kinds,
    describe_filter,
    merge_filters,
    unfiltered_violations,
)
from test_retrieval_fixtures import make_chunk, make_hit

ROW = frozenset({ChunkKind.ROW})
CARD = frozenset({ChunkKind.SCHEMA_CARD})


def test_default_config_filters_on_the_default_kinds() -> None:
    flt = build_filter(RetrievalConfig())
    assert flt is not None
    assert flt.kinds == RetrievalConfig().chunk_kinds


def test_each_ablation_axis_value_is_exact() -> None:
    for kinds in (ROW, CARD, ALL_KINDS):
        flt = build_filter(RetrievalConfig(chunk_kinds=kinds))
        assert flt is not None and flt.kinds == kinds
        row = make_chunk("r", kind=ChunkKind.ROW)
        card = make_chunk("c", kind=ChunkKind.SCHEMA_CARD)
        assert flt.matches(row) is (ChunkKind.ROW in kinds)
        assert flt.matches(card) is (ChunkKind.SCHEMA_CARD in kinds)


def test_caller_prefilter_is_anded_not_replaced() -> None:
    base = MetadataFilter(tables=frozenset({"orders"}), equals={"lang": "ar"})
    flt = build_filter(RetrievalConfig(chunk_kinds=ROW, prefilter=base))
    assert flt is not None
    assert flt.kinds == ROW
    assert flt.tables == frozenset({"orders"})
    assert flt.equals == {"lang": "ar"}


def test_kinds_intersect_rather_than_widen() -> None:
    base = MetadataFilter(kinds=frozenset({ChunkKind.ROW, ChunkKind.SCHEMA_CARD}))
    merged = merge_filters(base, ROW)
    assert merged is not None and merged.kinds == ROW


def test_disjoint_kinds_match_nothing_rather_than_everything() -> None:
    merged = merge_filters(MetadataFilter(kinds=CARD), ROW)
    assert merged is not None and merged.kinds == frozenset()
    assert not merged.matches(make_chunk("r", kind=ChunkKind.ROW))
    assert not merged.matches(make_chunk("c", kind=ChunkKind.SCHEMA_CARD))


def test_no_restriction_at_all_is_none() -> None:
    assert merge_filters(None, None) is None
    assert merge_filters(MetadataFilter(), None) is None


def test_string_kinds_are_coerced_to_enum_members() -> None:
    # A raw "row" happens to compare and hash equal to ChunkKind.ROW, so an
    # uncoerced string filter would work -- until someone writes "schema-card",
    # which would then be a filter that silently matches nothing instead of an
    # error. Coercion turns that into a raised exception.
    coerced = coerce_kinds({"row", "schema_card"})
    assert coerced == frozenset({ChunkKind.ROW, ChunkKind.SCHEMA_CARD})
    assert all(isinstance(k, ChunkKind) for k in coerced)
    assert coerce_kinds(None) is None
    with pytest.raises(ValueError):
        coerce_kinds({"not_a_kind"})


def test_describe_filter_is_json_safe() -> None:
    import json

    assert describe_filter(None) == {"applied": False}
    described = describe_filter(build_filter(RetrievalConfig(chunk_kinds=ROW)))
    assert described["applied"] is True
    assert described["kinds"] == ["row"]
    json.dumps(described)  # must not raise


def test_violations_detect_an_index_that_ignored_the_filter() -> None:
    flt = build_filter(RetrievalConfig(chunk_kinds=ROW))
    good = [make_hit("r1", 1)]
    bad = [make_hit("c1", 1, kind=ChunkKind.SCHEMA_CARD)]
    assert unfiltered_violations(good, flt) == []
    assert unfiltered_violations(bad, flt) == ["c1"]
    assert unfiltered_violations(bad, None) == []
