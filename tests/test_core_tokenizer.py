"""Token counting must be real, including on adversarial input."""

from __future__ import annotations

import pytest

from anyrag.core.tokenizer import (
    RegexTokenizer,
    count_tokens,
    get_tokenizer,
    truncate_to_tokens,
)

TOKENIZERS = [get_tokenizer(), RegexTokenizer()]
IDS = ["default", "regex-fallback"]


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_roundtrip_and_count(tok) -> None:  # noqa: ANN001
    text = "The quarterly revenue rose 12% in Q3 2024."
    assert tok.count(text) > 0
    assert tok.count(text) == len(tok.encode(text))


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_truncate_respects_budget(tok) -> None:  # noqa: ANN001
    text = "word " * 5000
    out = tok.truncate(text, 100)
    assert tok.count(out) <= 100
    assert len(out) < len(text)


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_truncate_noop_when_under_budget(tok) -> None:  # noqa: ANN001
    text = "short text"
    assert tok.truncate(text, 10_000) == text


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_zero_and_negative_budget(tok) -> None:  # noqa: ANN001
    assert tok.truncate("anything", 0) == ""
    assert tok.truncate("anything", -5) == ""


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_empty_input(tok) -> None:  # noqa: ANN001
    assert tok.count("") == 0
    assert tok.truncate("", 10) == ""


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_adversarially_long_single_field(tok) -> None:  # noqa: ANN001
    """A 200k-character field must truncate cleanly, not blow up."""
    text = "x" * 200_000
    out = tok.truncate(text, 50)
    assert tok.count(out) <= 50
    # The budget assertion above is satisfied trivially by a tokenizer that
    # counts the whole string as one token, so pin the count itself: an
    # unbroken run must cost roughly its length, not O(1).
    assert tok.count(text) >= 1000
    assert len(out) < 10_000


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
@pytest.mark.parametrize(
    ("text", "max_chars_per_token"),
    [
        ("x" * 200_000, 64),
        ("9" * 100_000, 64),
        # Whitespace legitimately compresses far harder in a real BPE
        # (cl100k encodes 100k spaces as ~782 tokens), so it gets a looser
        # bound. The point is that it must not be O(1).
        (" " * 100_000, 512),
        ("ب" * 50_000, 64),
        ("一" * 50_000, 64),
    ],
    ids=["latin", "digits", "spaces", "arabic", "cjk"],
)
def test_unbroken_runs_are_not_counted_as_one_token(
    tok, text, max_chars_per_token
) -> None:  # noqa: ANN001
    """No single unbroken run may collapse to a handful of tokens.

    A budget enforced with such a tokenizer would admit a context many times
    larger than the model can accept -- exactly the failure that a "real
    tokenizer, not a heuristic" requirement exists to prevent. The regression
    this pins: an unbounded `[A-Za-z]+` rule counted 200k characters as ONE
    token, and the budget test above still passed because it only asserted the
    count was *under* budget.
    """
    assert tok.count(text) >= len(text) / max_chars_per_token


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_arabic_is_not_char_over_four(tok) -> None:  # noqa: ANN001
    """Arabic is where a chars/4 heuristic goes badly wrong.

    We assert the real count differs from the naive estimate, which is the whole
    reason this module exists.
    """
    arabic = "مرحبا بالعالم هذا نص عربي طويل نسبيا لاختبار عد الرموز"
    naive = max(1, len(arabic) // 4)
    real = tok.count(arabic)
    assert real > 0
    assert real != naive


@pytest.mark.parametrize("tok", TOKENIZERS, ids=IDS)
def test_truncation_is_monotonic(tok) -> None:  # noqa: ANN001
    text = "alpha beta gamma delta epsilon zeta eta theta " * 50
    counts = [tok.count(tok.truncate(text, n)) for n in (10, 25, 50, 100)]
    assert counts == sorted(counts)


def test_module_level_helpers() -> None:
    assert count_tokens("hello world") > 0
    assert truncate_to_tokens("hello world " * 100, 5) != ""


def test_tokenizer_reports_provenance() -> None:
    info = get_tokenizer().info
    assert info.name
    assert info.encoding
    assert isinstance(info.degraded, bool)


def test_regex_fallback_marks_itself_degraded() -> None:
    assert RegexTokenizer().info.degraded is True


def test_mixed_script_truncation_does_not_corrupt() -> None:
    tok = get_tokenizer()
    text = "revenue إيرادات 收入 " * 200
    out = tok.truncate(text, 30)
    assert tok.count(out) <= 30
    assert "�" not in out
