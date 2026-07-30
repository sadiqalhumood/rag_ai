"""Token-boundary truncation and overlapping splits.

Two separate jobs, deliberately kept apart:

* **Field truncation** bounds one pathological *value* (a 200k-character notes
  column) before it drowns out the twenty other columns in the same row.
* **Chunk splitting** bounds the *serialized row* after all fields have been
  rendered, emitting overlapping parts so a fact that straddles a boundary is
  still wholly present in at least one part.

Both measure with `anyrag.core.tokenizer`, never with `len(text)`. A character
budget is wrong by 2-4x on Arabic and on dense numeric text, which is precisely
the content this corpus is made of; the failure mode is a context window that
overflows only on the rows that matter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..core.errors import ConfigError
from ..core.tokenizer import Tokenizer, get_tokenizer

#: Truncation is marked in the text, not just in metadata: a reader (human or
#: model) of a truncated value must be able to see that it is truncated without
#: consulting the chunk's meta dict.
TRUNCATION_MARKER = " …[truncated]"

#: tiktoken emits U+FFFD when a decoded token prefix splits a multi-byte
#: character. Stripping it keeps Arabic text clean at part boundaries.
_REPLACEMENT = "�"

#: Safety valve, not a budget. `RegexTokenizer` segments `[A-Za-z]+` as one
#: piece, so a 200k-character unbroken string counts as *one token* and would
#: sail through a purely token-based check -- a degraded run could then emit a
#: 200k-character chunk. Tokens remain the measurement; this only bounds the
#: pathological case a tokenizer cannot see. 64 chars/token is far above any
#: real token, so it never fires on natural text in either vocabulary.
MAX_CHARS_PER_TOKEN = 64


@dataclass(frozen=True)
class Truncation:
    """Result of truncating a single field value."""

    text: str
    truncated: bool
    original_tokens: int
    kept_tokens: int


def _clean(text: str) -> str:
    return text.replace(_REPLACEMENT, "") if _REPLACEMENT in text else text


def _certainly_fits(text: str, max_tokens: int) -> bool:
    """True when `text` cannot possibly exceed `max_tokens`, without encoding it.

    Every token covers at least one character, so an ASCII string no longer than
    the budget is always within it. The ASCII guard is not decorative: tiktoken
    works on bytes, so one Arabic character can become two tokens and the
    inequality would not hold. Ingestion calls this once per row, so skipping
    the encode on ordinary short rows is the difference between one pass over
    the corpus and two.
    """
    return len(text) <= max_tokens and text.isascii()


def truncate_field(
    value: str,
    max_tokens: int,
    *,
    tokenizer: Tokenizer | None = None,
    marker: str = TRUNCATION_MARKER,
) -> Truncation:
    """Cut `value` to `max_tokens` tokens at a token boundary.

    The marker is budgeted for, not appended on top, so the returned text
    honours `max_tokens` rather than overshooting it by the marker's length.
    """
    tok = tokenizer or get_tokenizer()
    if max_tokens <= 0:
        return Truncation("", bool(value), tok.count(value) if value else 0, 0)
    total = tok.count(value)
    char_limit = max_tokens * MAX_CHARS_PER_TOKEN
    if total <= max_tokens and len(value) <= char_limit:
        return Truncation(value, False, total, total)

    marker_tokens = tok.count(marker)
    keep = max(1, max_tokens - marker_tokens)
    # Cheap char pre-cut (the backstop, budgeting for the marker), then the
    # real, token-boundary cut inside it. For natural text the pre-cut is far
    # wider than `keep` tokens and so has no effect on where the cut lands.
    prefix = value[: max(1, char_limit - len(marker))]
    head = _clean(tok.truncate(prefix, keep))
    text = head + marker
    return Truncation(text, True, total, tok.count(text))


def split_with_overlap(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
    *,
    tokenizer: Tokenizer | None = None,
) -> list[str]:
    """Split `text` into parts of <= `max_tokens`, sharing `overlap_tokens`.

    Consecutive parts share their boundary tokens exactly: the last
    `overlap_tokens` tokens of part *i* are the first `overlap_tokens` tokens of
    part *i+1*. That is what stops a `name is X; region is Y` clause pair from
    being severed such that neither part supports the question that needs both.

    Returns `[text]` unchanged when it already fits, so the common case costs
    one token count and no decode.
    """
    if max_tokens <= 0:
        raise ConfigError("max_tokens must be positive")
    if overlap_tokens < 0:
        raise ConfigError("overlap_tokens must not be negative")
    if overlap_tokens >= max_tokens:
        raise ConfigError(
            f"overlap_tokens ({overlap_tokens}) must be smaller than "
            f"max_tokens ({max_tokens}); otherwise splitting cannot advance"
        )

    tok = tokenizer or get_tokenizer()
    if not text:
        return [""]
    char_limit = max_tokens * MAX_CHARS_PER_TOKEN
    if _certainly_fits(text, max_tokens):
        return [text]
    ids: Sequence[int] = tok.encode(text)
    if len(ids) <= max_tokens and len(text) <= char_limit:
        return [text]

    stride = max_tokens - overlap_tokens
    parts: list[str] = []
    start = 0
    n = len(ids)
    while start < n:
        window = ids[start : start + max_tokens]
        parts.append(_clean(tok.decode(window)))
        if start + max_tokens >= n:
            break
        start += stride

    # Safety valve (see MAX_CHARS_PER_TOKEN): a tokenizer that segments a huge
    # unbroken string into a handful of pieces cannot produce parts of bounded
    # size, so any part it leaves oversized is cut on characters.
    if any(len(p) > char_limit for p in parts):
        char_overlap = min(
            overlap_tokens * MAX_CHARS_PER_TOKEN, max(1, char_limit // 4)
        )
        expanded: list[str] = []
        for part in parts:
            expanded.extend(_char_windows(part, char_limit, char_overlap))
        parts = expanded
    return parts


def _char_windows(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    stride = max(1, size - overlap)
    out: list[str] = []
    start = 0
    while start < len(text):
        out.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += stride
    return out


def shared_boundary(first: str, second: str) -> str:
    """Longest suffix of `first` that is also a prefix of `second`.

    Only used by tests and diagnostics, but it lives here so the overlap
    contract is checkable from outside without re-deriving the token windows.
    """
    limit = min(len(first), len(second))
    for length in range(limit, 0, -1):
        if first.endswith(second[:length]):
            return second[:length]
    return ""
