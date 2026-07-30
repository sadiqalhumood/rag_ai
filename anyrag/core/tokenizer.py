"""Real token counting.

Token budgeting is measured, never estimated. A chars/4 heuristic is wrong by
2-4x on exactly the inputs that matter here -- Arabic text, long identifier-like
strings, and dense numeric tables -- which is how context packers silently
overflow.

`get_tokenizer()` prefers tiktoken's cl100k_base and falls back to a
deterministic regex tokenizer if the BPE file cannot be fetched. The fallback is
still a real tokenizer (it segments and counts actual units); it is simply a
different vocabulary, and it reports itself as degraded so provenance travels
with any number computed from it.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class TokenizerInfo:
    name: str
    encoding: str
    degraded: bool = False
    detail: str = ""


@runtime_checkable
class Tokenizer(Protocol):
    info: TokenizerInfo

    def encode(self, text: str) -> Sequence[int]: ...
    def decode(self, tokens: Sequence[int]) -> str: ...
    def count(self, text: str) -> int: ...
    def truncate(self, text: str, max_tokens: int) -> str: ...


class TiktokenTokenizer:
    """cl100k_base, the encoding used by current-generation chat models."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding_name)
        self.info = TokenizerInfo(name="tiktoken", encoding=encoding_name)

    def encode(self, text: str) -> Sequence[int]:
        return self._enc.encode(text, disallowed_special=())

    def decode(self, tokens: Sequence[int]) -> str:
        return self._enc.decode(list(tokens))

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        toks = self.encode(text)
        if len(toks) <= max_tokens:
            return text
        # Decoding a prefix can split a multi-byte character; tiktoken handles
        # this by emitting the replacement char, which we strip.
        return self.decode(toks[:max_tokens]).replace("�", "")


class RegexTokenizer:
    """Deterministic fallback: words, numbers, CJK/Arabic runs, punctuation.

    Not byte-pair encoding, so counts differ from a model tokenizer, but it is
    stable, dependency-free, and never under-counts by an order of magnitude the
    way a character heuristic does.
    """

    # Every run is length-bounded. An unbounded `[A-Za-z]+` counts a 200k-char
    # unbroken string as ONE token, which would let a pathological field sail
    # through a token budget that is doing exactly what it was told. Bounding
    # the runs keeps the count within a small constant factor of a real BPE
    # tokenizer instead of being wrong by five orders of magnitude.
    _MAX_RUN = 24

    _PATTERN = re.compile(
        rf"[A-Za-z]{{1,{_MAX_RUN}}}"
        rf"|\d{{1,{_MAX_RUN}}}"
        rf"|[؀-ۿ]{{1,{_MAX_RUN}}}"
        r"|[一-鿿]"
        rf"|\s{{1,{_MAX_RUN}}}"
        r"|[^\sA-Za-z\d]",
    )

    def __init__(self) -> None:
        self.info = TokenizerInfo(
            name="regex-fallback",
            encoding="regex-v1",
            degraded=True,
            detail="tiktoken unavailable; counts are approximate",
        )
        self._vocab: dict[str, int] = {}
        self._inverse: list[str] = []

    def _piece_ids(self, pieces: list[str]) -> list[int]:
        out = []
        for p in pieces:
            if p not in self._vocab:
                self._vocab[p] = len(self._inverse)
                self._inverse.append(p)
            out.append(self._vocab[p])
        return out

    def _pieces(self, text: str) -> list[str]:
        # Whitespace runs attach to the following token rather than counting
        # separately, mirroring BPE behaviour closely enough for budgeting.
        raw = self._PATTERN.findall(text)
        pieces: list[str] = []
        pending = ""
        for tok in raw:
            if tok.isspace():
                pending += tok
                # Flush long whitespace runs so they cannot accumulate into a
                # single piece; otherwise bounding the regex achieves nothing
                # for an all-whitespace input.
                if len(pending) >= self._MAX_RUN:
                    pieces.append(pending)
                    pending = ""
            else:
                pieces.append(pending + tok)
                pending = ""
        if pending:
            pieces.append(pending)
        return pieces

    def encode(self, text: str) -> Sequence[int]:
        return self._piece_ids(self._pieces(text))

    def decode(self, tokens: Sequence[int]) -> str:
        return "".join(self._inverse[t] for t in tokens)

    def count(self, text: str) -> int:
        return len(self._pieces(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        pieces = self._pieces(text)
        if len(pieces) <= max_tokens:
            return text
        return "".join(pieces[:max_tokens])


_LOCK = threading.Lock()
_CACHED: Tokenizer | None = None


def get_tokenizer(force_fallback: bool = False) -> Tokenizer:
    """Return the process-wide tokenizer, building it on first use."""
    global _CACHED
    with _LOCK:
        if _CACHED is not None and not force_fallback:
            return _CACHED
        tok: Tokenizer
        if force_fallback:
            return RegexTokenizer()
        try:
            tok = TiktokenTokenizer()
        except Exception as exc:  # network, missing wheel, corrupt cache
            tok = RegexTokenizer()
            tok.info = TokenizerInfo(
                name="regex-fallback",
                encoding="regex-v1",
                degraded=True,
                detail=f"tiktoken unavailable: {type(exc).__name__}: {exc}",
            )
        _CACHED = tok
        return tok


def count_tokens(text: str) -> int:
    return get_tokenizer().count(text)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    return get_tokenizer().truncate(text, max_tokens)
