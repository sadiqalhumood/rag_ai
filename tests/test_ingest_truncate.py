"""Oversized fields and overlapping splits, measured in tokens.

Run against both tokenizers: the token-budget logic must not depend on tiktoken
being downloadable, and the fallback's different vocabulary is exactly the kind
of difference that turns "budgeted" into "overflowed" if a chars/4 heuristic
sneaks back in.
"""

from __future__ import annotations

import pytest

from anyrag.core.errors import ConfigError
from anyrag.core.tokenizer import RegexTokenizer, get_tokenizer
from anyrag.ingest.truncate import (
    MAX_CHARS_PER_TOKEN,
    TRUNCATION_MARKER,
    shared_boundary,
    split_with_overlap,
    truncate_field,
)

TOKENIZERS = [get_tokenizer(), RegexTokenizer()]
IDS = ["default", "regex-fallback"]

#: The adversarial input the brief asks for.
HUGE = "incident report " * 12_500  # 200k characters
#: An unbroken 200k-character string: `RegexTokenizer` segments [A-Za-z]+ as a
#: single piece, so this counts as *one token* and is the case a purely
#: token-based check cannot see.
HUGE_UNBROKEN = "x" * 200_000
#: Non-repetitive, so a longest-common-substring reconstruction cannot get
#: lucky (or unlucky) on accidental matches away from the real boundary.
VARIED = " ".join(f"row {i} scored {i * 7} in region {i % 13}" for i in range(3_000))
ARABIC = "الشركة سجلت نموا كبيرا في الإيرادات خلال الربع الثالث من العام. " * 200


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_short_value_is_untouched(tok) -> None:
    result = truncate_field("region is EMEA", 100, tokenizer=tok)
    assert result.text == "region is EMEA"
    assert result.truncated is False


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_200k_character_field_is_cut_to_budget(tok) -> None:
    assert len(HUGE) >= 200_000
    result = truncate_field(HUGE, 64, tokenizer=tok)
    assert result.truncated is True
    assert result.original_tokens > 10_000
    # The marker is budgeted for, not added on top of the budget.
    assert tok.count(result.text) <= 64
    assert result.text.endswith(TRUNCATION_MARKER)


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_unbroken_200k_string_cannot_slip_through_a_token_only_check(tok) -> None:
    """The fallback tokenizer counts this as one token; it must still be cut.

    Tokens stay the budget; MAX_CHARS_PER_TOKEN is the backstop for input the
    tokenizer's segmentation cannot see.
    """
    result = truncate_field(HUGE_UNBROKEN, 20, tokenizer=tok)
    assert result.truncated is True
    assert len(result.text) <= 20 * MAX_CHARS_PER_TOKEN
    assert "[truncated]" in result.text


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_unbroken_string_splits_into_bounded_parts(tok) -> None:
    parts = split_with_overlap(HUGE_UNBROKEN, 128, 32, tokenizer=tok)
    assert len(parts) > 1
    assert all(len(p) <= 128 * MAX_CHARS_PER_TOKEN for p in parts)


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_truncation_is_visible_in_the_text_not_only_in_meta(tok) -> None:
    result = truncate_field("incident " * 5000, 20, tokenizer=tok)
    assert "[truncated]" in result.text


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_arabic_truncation_leaves_no_replacement_characters(tok) -> None:
    result = truncate_field(ARABIC, 40, tokenizer=tok)
    assert result.truncated is True
    assert "�" not in result.text
    # Still Arabic, not transliterated or stripped.
    assert any("؀" <= ch <= "ۿ" for ch in result.text)


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_truncation_is_deterministic(tok) -> None:
    a = truncate_field(HUGE, 50, tokenizer=tok)
    b = truncate_field(HUGE, 50, tokenizer=tok)
    assert a.text == b.text


# -- splitting --------------------------------------------------------------


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_text_under_budget_is_a_single_part(tok) -> None:
    assert split_with_overlap("short row", 100, 10, tokenizer=tok) == ["short row"]


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_every_part_respects_the_budget(tok) -> None:
    parts = split_with_overlap(HUGE, 200, 40, tokenizer=tok)
    assert len(parts) > 1
    assert all(tok.count(p) <= 200 for p in parts)


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_consecutive_parts_share_exactly_the_configured_overlap(tok) -> None:
    """The last N tokens of part i must be the first N tokens of part i+1."""
    max_tokens, overlap = 128, 32
    stride = max_tokens - overlap
    parts = split_with_overlap(VARIED, max_tokens, overlap, tokenizer=tok)
    assert len(parts) > 3

    ids = list(tok.encode(VARIED))
    for i in range(len(parts) - 1):
        start = i * stride
        expected = tok.decode(ids[start + stride : start + max_tokens])
        assert parts[i].endswith(expected), f"part {i} does not end with the overlap"
        assert parts[i + 1].startswith(expected), f"part {i+1} does not start with it"
        # And the shared region is genuinely `overlap` tokens wide, not a
        # coincidental few characters.
        assert tok.count(shared_boundary(parts[i], parts[i + 1])) >= overlap - 2


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_zero_overlap_partitions_without_repetition(tok) -> None:
    parts = split_with_overlap(VARIED, 150, 0, tokenizer=tok)
    assert "".join(parts) == VARIED


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_overlapping_parts_still_cover_the_whole_text(tok) -> None:
    """Nothing is dropped: stripping each part's overlap rebuilds the input."""
    max_tokens, overlap = 128, 32
    stride = max_tokens - overlap
    parts = split_with_overlap(VARIED, max_tokens, overlap, tokenizer=tok)
    ids = list(tok.encode(VARIED))

    rebuilt = parts[0]
    for i, nxt in enumerate(parts[1:], start=1):
        start = i * stride
        shared = tok.decode(ids[start : start + overlap])
        assert nxt.startswith(shared)
        rebuilt += nxt[len(shared) :]
    assert rebuilt == VARIED


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_arabic_splits_without_mojibake(tok) -> None:
    parts = split_with_overlap(ARABIC, 64, 16, tokenizer=tok)
    assert len(parts) > 1
    assert all("�" not in p for p in parts)


def test_overlap_must_be_smaller_than_the_budget() -> None:
    with pytest.raises(ConfigError):
        split_with_overlap("text", 32, 32)
    with pytest.raises(ConfigError):
        split_with_overlap("text", 32, 64)
    with pytest.raises(ConfigError):
        split_with_overlap("text", 0, 0)


def test_shared_boundary_finds_nothing_when_there_is_nothing() -> None:
    assert shared_boundary("abc", "xyz") == ""
    assert shared_boundary("hello world", "world peace") == "world"
