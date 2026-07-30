"""Citation resolution, hallucinated-marker rejection, and the refusal fallback.

The load-bearing property here is that `CitationError` -- which
`Answer.__post_init__` raises for an uncited non-refusal -- is never how an
uncited answer surfaces. `finalize` catches the condition first and returns an
explicit refusal, so the exception stays a backstop.
"""

from __future__ import annotations

import pytest

from anyrag.core.config import GenerationConfig
from anyrag.core.errors import CitationError
from anyrag.core.types import Answer, Chunk, ChunkKind, Hit, QueryRoute, RowRef
from anyrag.generate.citations import (
    MAX_QUOTED_SPAN,
    finalize,
    parse_markers,
    resolve,
)
from anyrag.generate.packer import PackedContext, pack


def make_hit(chunk_id: str, text: str, score: float, pk: str = "1") -> Hit:
    return Hit(
        chunk=Chunk(
            chunk_id=chunk_id,
            source_id="src",
            kind=ChunkKind.ROW,
            text=text,
            row_refs=(RowRef(table="customers", pk=pk),),
            meta={"table": "customers"},
        ),
        score=score,
    )


HITS = [
    make_hit("c1", "customer_id: 7\nname: Ahmed Al-Sayed\nemail: ahmed@example.com",
             0.91, pk="7"),
    make_hit("c2", "customer_id: 8\nname: Fatima Nasser\nemail: fatima@example.com",
             0.55, pk="8"),
]


def packed() -> PackedContext:
    return pack(HITS, GenerationConfig())


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_parse_markers_preserves_first_appearance_order_and_dedupes() -> None:
    assert parse_markers("a [2] b [1] c [2] d") == (2, 1)


def test_parse_markers_on_text_without_markers() -> None:
    assert parse_markers("no markers at all") == ()
    assert parse_markers("") == ()


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def test_valid_markers_resolve_to_citations_with_provenance() -> None:
    ctx = packed()
    result = resolve("The email is ahmed@example.com [1].", ctx,
                     question="What is Ahmed Al-Sayed's email?")

    assert result.valid_markers == (1,)
    assert result.invalid_markers == ()
    (citation,) = result.citations
    assert citation.chunk_id == "c1"
    assert citation.source_id == "src"
    assert citation.row_refs == (RowRef(table="customers", pk="7"),)
    assert citation.score == pytest.approx(0.91)
    assert citation.quoted_span


def test_hallucinated_marker_is_rejected_not_silently_accepted() -> None:
    ctx = packed()
    assert len(ctx.blocks) == 2

    result = resolve("Ahmed [1] also appears in the archive [7].", ctx)

    assert result.valid_markers == (1,)
    assert result.invalid_markers == (7,)
    assert [c.chunk_id for c in result.citations] == ["c1"]
    assert "[7]" not in result.text


def test_hallucinated_marker_does_not_resolve_to_a_nearby_block() -> None:
    ctx = packed()
    result = resolve("claim [3]", ctx)

    assert result.citations == ()
    assert result.invalid_markers == (3,)


def test_quoted_span_is_bounded_and_drawn_from_the_packed_text() -> None:
    long_row = "note: " + ("some long field value " * 60)
    ctx = pack([make_hit("big", long_row, 0.9)], GenerationConfig())
    result = resolve("see [1]", ctx, question="what is the note")

    (citation,) = result.citations
    assert len(citation.quoted_span) <= MAX_QUOTED_SPAN + 1  # +1 for the ellipsis
    assert citation.quoted_span.rstrip("…").strip() in ctx.blocks[0].text


def test_quoted_span_prefers_the_line_matching_the_question() -> None:
    ctx = packed()
    result = resolve("answer [1]", ctx, question="what is the email address?")
    assert "email" in result.citations[0].quoted_span


# --------------------------------------------------------------------------
# finalize: the guard that keeps CitationError unreachable
# --------------------------------------------------------------------------


def test_finalize_builds_a_cited_answer() -> None:
    answer = finalize("email: ahmed@example.com [1]", packed(),
                      question="email of Ahmed?", route=QueryRoute.LOOKUP)

    assert answer.refused is False
    assert answer.cited_chunk_ids == ("c1",)
    assert answer.route is QueryRoute.LOOKUP
    assert answer.trace["citations"]["n_citations"] == 1


def test_zero_valid_citations_becomes_a_refusal_never_an_uncited_answer() -> None:
    answer = finalize("The email is ahmed@example.com.", packed())

    assert answer.refused is True
    assert answer.reason.startswith("no_valid_citations")
    assert answer.citations == ()


def test_only_hallucinated_citations_becomes_a_refusal() -> None:
    answer = finalize("The archive says so [7][8].", packed())

    assert answer.refused is True
    assert "hallucinated marker" in answer.reason
    assert "[7]" in answer.reason and "[8]" in answer.reason


def test_text_emptied_by_marker_stripping_becomes_a_refusal() -> None:
    answer = finalize("[9]", packed())
    assert answer.refused is True


def test_finalize_never_raises_citation_error() -> None:
    for text in ["", "   ", "no markers", "[99]", "[0]", "marker-free [abc]"]:
        answer = finalize(text, packed())
        assert answer.refused is True


def test_answer_constructor_still_enforces_the_invariant() -> None:
    """The backstop is real; finalize just makes sure nothing reaches it."""
    with pytest.raises(CitationError):
        Answer(text="uncited claim", citations=(), refused=False)


def test_finalize_on_an_empty_context_refuses() -> None:
    answer = finalize("something [1]", PackedContext())
    assert answer.refused is True
    assert answer.citations == ()
