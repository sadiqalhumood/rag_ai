"""The citation invariant and the RowRef normalisation contract."""

from __future__ import annotations

import pytest

from anyrag.core.errors import CitationError, SchemaError
from anyrag.core.types import (
    Answer,
    Chunk,
    ChunkKind,
    Citation,
    ColumnSchema,
    QueryResult,
    QueryRoute,
    RowRef,
    TableRef,
    TableSchema,
)


def test_uncited_answer_is_unconstructible() -> None:
    """An answer with no citation is a failure, not a warning."""
    with pytest.raises(CitationError):
        Answer(text="The total is 42.", citations=())


def test_refusal_needs_no_citations() -> None:
    ans = Answer.refusal("no supporting evidence")
    assert ans.refused is True
    assert ans.citations == ()
    assert ans.reason


def test_cited_answer_is_fine() -> None:
    ans = Answer(
        text="The total is 42.",
        citations=(Citation(chunk_id="c1", source_id="sqlite:test"),),
    )
    assert ans.refused is False
    assert ans.cited_chunk_ids == ("c1",)


def test_rowref_pk_is_always_a_string() -> None:
    """Gold matching must not depend on which adapter loaded the data."""
    assert RowRef("customers", 7).pk == "7"
    assert RowRef("customers", "7").pk == "7"
    assert RowRef("customers", 7) == RowRef("customers", "7")
    assert hash(RowRef("customers", 7)) == hash(RowRef("customers", "7"))


def test_rowref_is_hashable_for_set_intersection() -> None:
    gold = {RowRef("orders", 1), RowRef("orders", 2)}
    retrieved = {RowRef("orders", "2"), RowRef("orders", "3")}
    assert gold & retrieved == {RowRef("orders", 2)}


def test_answer_collects_row_refs_across_citations() -> None:
    ans = Answer(
        text="x",
        citations=(
            Citation("c1", "s", row_refs=(RowRef("t", 1),)),
            Citation("c2", "s", row_refs=(RowRef("t", 2),)),
        ),
    )
    assert set(ans.cited_row_refs) == {RowRef("t", 1), RowRef("t", 2)}


def test_table_ref_qualification() -> None:
    assert TableRef("orders").qualified == "orders"
    assert TableRef("orders", "public").qualified == "public.orders"


def test_schema_lookup_raises_on_missing_column() -> None:
    schema = TableSchema(
        table=TableRef("t"),
        columns=(ColumnSchema("id", "INTEGER", is_primary_key=True),),
        primary_key=("id",),
    )
    assert schema.column("id").is_primary_key
    assert schema.column_names == ("id",)
    with pytest.raises(SchemaError):
        schema.column("nope")


def test_query_result_scalar() -> None:
    assert QueryResult(("n",), ((42,),)).scalar == 42
    assert QueryResult(("a", "b"), ((1, 2),)).scalar is None
    assert QueryResult(("n",), ()).scalar is None
    assert QueryResult(("n",), ((1,), (2,))).scalar is None


def test_query_result_as_dicts() -> None:
    res = QueryResult(("a", "b"), ((1, 2), (3, 4)))
    assert res.as_dicts() == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    assert len(res) == 2


def test_chunk_accessors() -> None:
    chunk = Chunk(
        chunk_id="abc",
        source_id="sqlite:test",
        kind=ChunkKind.ROW,
        text="hello",
        row_refs=(RowRef("t", 1),),
        meta={"table": "t", "content_hash": "deadbeef"},
    )
    assert chunk.table == "t"
    assert chunk.content_hash == "deadbeef"


def test_query_route_values() -> None:
    assert {r.value for r in QueryRoute} == {"lookup", "aggregate", "hybrid"}
