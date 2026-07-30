"""Packing under a measured token budget, including adversarial input.

Every budget assertion here re-measures with the tokenizer rather than trusting
the packer's own bookkeeping, and one test pins the gap between a real token
count and a chars/4 estimate so a regression to a character heuristic fails
loudly instead of silently overflowing prompts.
"""

from __future__ import annotations

import pytest

from anyrag.core.config import GenerationConfig
from anyrag.core.errors import BudgetExceededError
from anyrag.core.tokenizer import RegexTokenizer, get_tokenizer
from anyrag.core.types import Chunk, ChunkKind, Hit, RowRef
from anyrag.generate.packer import TRUNCATION_MARK, pack

TOKENIZERS = [get_tokenizer(), RegexTokenizer()]
IDS = ["default", "regex-fallback"]

ARABIC = "العميل أحمد السيد من القاهرة اشترى ثلاثة منتجات في مارس"
CJK = "客户购买了三件产品在三月份东京仓库发货"


def make_hit(chunk_id: str, text: str, score: float, *, table: str = "customers",
             pk: str | None = None) -> Hit:
    return Hit(
        chunk=Chunk(
            chunk_id=chunk_id,
            source_id="src",
            kind=ChunkKind.ROW,
            text=text,
            row_refs=(RowRef(table=table, pk=pk or chunk_id),),
            meta={"table": table},
        ),
        score=score,
    )


# --------------------------------------------------------------------------
# Budget is measured, and respected
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_budget_respected_with_200k_character_field(tok) -> None:  # noqa: ANN001
    """A single 200k-character field must truncate into budget, not overflow."""
    config = GenerationConfig(max_prompt_tokens=600, max_chunk_tokens=200)
    hits = [make_hit("huge", "value: " + ("x9y8z7 " * 30_000), 0.9)]
    assert len(hits[0].chunk.text) > 200_000

    packed = pack(hits, config, reserved_tokens=50, tokenizer=tok)

    assert not packed.is_empty
    assert tok.count(packed.text) == packed.tokens
    assert packed.total_tokens <= config.max_prompt_tokens
    assert packed.truncated == ("huge",)


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_budget_respected_with_500_small_chunks(tok) -> None:  # noqa: ANN001
    config = GenerationConfig(max_prompt_tokens=1500, max_chunk_tokens=64)
    hits = [
        make_hit(f"c{i:03d}", f"order_id: {i} | status: shipped | total: {i * 3}.50",
                 score=1.0 - i / 1000.0)
        for i in range(500)
    ]

    packed = pack(hits, config, reserved_tokens=120, tokenizer=tok)

    assert not packed.is_empty
    assert tok.count(packed.text) == packed.tokens
    assert packed.total_tokens <= config.max_prompt_tokens
    assert len(packed.dropped) == 500 - len(packed.blocks)


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_budget_respected_with_mixed_arabic_english_cjk(tok) -> None:  # noqa: ANN001
    config = GenerationConfig(max_prompt_tokens=800, max_chunk_tokens=128)
    hits = [
        make_hit("ar", f"name: أحمد السيد | note: {ARABIC}", 0.9),
        make_hit("cjk", f"name: 田中 | note: {CJK}", 0.8),
        make_hit("mix", f"note: {ARABIC} / {CJK} / mixed ascii tail", 0.7),
        make_hit("en", "name: John Smith | note: plain english row", 0.6),
    ]

    packed = pack(hits, config, reserved_tokens=100, tokenizer=tok)

    assert tok.count(packed.text) == packed.tokens
    assert packed.total_tokens <= config.max_prompt_tokens


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_token_count_is_not_a_character_heuristic(tok) -> None:  # noqa: ANN001
    """Arabic is where chars/4 goes wrong; pin that the counts really differ."""
    naive = len(ARABIC) / 4.0
    measured = tok.count(ARABIC)
    assert abs(measured - naive) / max(naive, 1.0) > 0.25


def test_reported_tokens_match_a_fresh_measurement() -> None:
    tok = get_tokenizer()
    config = GenerationConfig(max_prompt_tokens=2000, max_chunk_tokens=256)
    hits = [make_hit(f"c{i}", f"field_{i}: value {i}", 0.5) for i in range(20)]

    packed = pack(hits, config, tokenizer=tok)

    assert packed.tokens == tok.count(packed.text)
    assert packed.tokenizer is not None


# --------------------------------------------------------------------------
# What gets dropped, and what gets truncated
# --------------------------------------------------------------------------


def test_lowest_scored_chunks_are_dropped_first() -> None:
    tok = get_tokenizer()
    body = "field: " + ("token " * 60)
    hits = [
        make_hit("worst", body, 0.10),
        make_hit("best", body, 0.95),
        make_hit("middle", body, 0.50),
    ]
    per_chunk = tok.count(body)
    # Room for roughly two of the three blocks.
    config = GenerationConfig(max_prompt_tokens=int(per_chunk * 2.4),
                              max_chunk_tokens=1000)

    packed = pack(hits, config, tokenizer=tok)

    kept = [b.chunk_id for b in packed.blocks]
    assert "best" in kept
    assert "worst" not in kept
    assert "worst" in packed.dropped


def test_dropping_walks_up_the_score_order() -> None:
    hits = [make_hit(f"c{i}", "field: " + ("word " * 40), score=i / 10.0)
            for i in range(10)]
    config = GenerationConfig(max_prompt_tokens=200, max_chunk_tokens=500)

    packed = pack(hits, config)

    kept_scores = [b.score for b in packed.blocks]
    dropped_scores = [
        h.score for h in hits if h.chunk.chunk_id in packed.dropped
    ]
    assert kept_scores, "at least the top-scored chunk should survive"
    assert min(kept_scores) >= max(dropped_scores)


def test_oversized_chunk_is_truncated_not_dropped() -> None:
    tok = get_tokenizer()
    config = GenerationConfig(max_prompt_tokens=4000, max_chunk_tokens=40)
    long_body = "customer_id: 7 | " + ("filler_field: filler_value | " * 500)
    hits = [make_hit("wide", long_body, 0.9)]

    packed = pack(hits, config, tokenizer=tok)

    assert [b.chunk_id for b in packed.blocks] == ["wide"]
    block = packed.blocks[0]
    assert block.truncated is True
    assert block.text.endswith(TRUNCATION_MARK)
    assert tok.count(block.text) <= config.max_chunk_tokens
    # The head of the row -- where the answer usually is -- survives.
    assert "customer_id: 7" in block.text


def test_truncation_is_at_a_token_boundary() -> None:
    tok = get_tokenizer()
    config = GenerationConfig(max_prompt_tokens=4000, max_chunk_tokens=30)
    hits = [make_hit("ar", ARABIC * 40, 0.9)]

    packed = pack(hits, config, tokenizer=tok)
    body = packed.blocks[0].text

    assert "�" not in body  # no split multi-byte character
    assert tok.count(body) <= config.max_chunk_tokens


# --------------------------------------------------------------------------
# Unpackable input is a refusal, not a crash
# --------------------------------------------------------------------------


def test_chunk_larger_than_the_whole_budget_yields_a_refusal_not_a_crash() -> None:
    config = GenerationConfig(max_prompt_tokens=40, max_chunk_tokens=100_000)
    hits = [make_hit("giant", "value " * 50_000, 0.9)]

    packed = pack(hits, config)

    assert packed.is_empty
    assert packed.reason == "context_budget_exhausted"
    assert packed.blocks == ()
    assert packed.text == ""


def test_every_chunk_too_large_yields_a_refusal() -> None:
    config = GenerationConfig(max_prompt_tokens=30, max_chunk_tokens=100_000)
    hits = [make_hit(f"c{i}", "word " * 5_000, 0.9 - i / 10) for i in range(5)]

    packed = pack(hits, config)

    assert packed.is_empty
    assert len(packed.dropped) == 5


def test_reserved_tokens_alone_can_exhaust_the_budget() -> None:
    config = GenerationConfig(max_prompt_tokens=100, max_chunk_tokens=50)
    hits = [make_hit("c1", "small row", 0.9)]

    packed = pack(hits, config, reserved_tokens=100)

    assert packed.is_empty
    assert packed.reason == "context_budget_exhausted"


def test_strict_mode_raises_budget_exceeded() -> None:
    config = GenerationConfig(max_prompt_tokens=20, max_chunk_tokens=100_000)
    hits = [make_hit("giant", "word " * 5_000, 0.9)]

    with pytest.raises(BudgetExceededError):
        pack(hits, config, strict=True)


def test_no_hits_packs_to_an_empty_context() -> None:
    packed = pack([], GenerationConfig())
    assert packed.is_empty
    assert packed.reason == "no_hits"


# --------------------------------------------------------------------------
# Determinism and rendering
# --------------------------------------------------------------------------


def test_packing_is_deterministic_for_a_given_input_ordering() -> None:
    config = GenerationConfig(max_prompt_tokens=400, max_chunk_tokens=64)
    hits = [make_hit(f"c{i}", f"order_id: {i} | note: repeated text here", 0.5)
            for i in range(40)]

    first = pack(hits, config)
    second = pack(hits, config)

    assert first.text == second.text
    assert first.marker_map == second.marker_map
    assert first.dropped == second.dropped


def test_markers_are_sequential_and_map_to_chunk_ids() -> None:
    config = GenerationConfig(max_prompt_tokens=4000, max_chunk_tokens=128)
    hits = [make_hit("a", "row a", 0.9), make_hit("b", "row b", 0.8)]

    packed = pack(hits, config)

    assert [b.marker for b in packed.blocks] == [1, 2]
    assert packed.marker_map == {1: "a", 2: "b"}
    assert "[1]" in packed.text and "[2]" in packed.text
    assert packed.by_marker(3) is None


def test_brackets_in_chunk_text_cannot_masquerade_as_markers() -> None:
    config = GenerationConfig(max_prompt_tokens=4000, max_chunk_tokens=128)
    hits = [make_hit("a", "note: see [7] in the appendix", 0.9)]

    packed = pack(hits, config)

    assert "[7]" not in packed.text
    assert "(7)" in packed.blocks[0].text
