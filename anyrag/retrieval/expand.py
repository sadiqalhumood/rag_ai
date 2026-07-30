"""Query expansion: deterministic, offline, no LLM.

Expansion here means "the same question, spelled the way the corpus might have
spelled it". Every variant is produced by a named rule, in a fixed order, with
no randomness and no network -- an expansion that needed a model would make one
axis of the ablation depend on an API key.

The rules, in the order they are tried
--------------------------------------
``punctuation``
    Punctuation becomes whitespace: ``Ahmed Al-Sayed`` -> ``Ahmed Al Sayed``,
    ``customer_id`` -> ``customer id``. The cheapest and safest rule, and on
    this corpus also the one that bridges the hyphenated/spaced name split.
``arabic``
    Orthographic normalisation of the Arabic script: alef forms (أ إ آ ٱ) to
    ا, alef maksura ى to ي, ta marbuta ة to ه, diacritics and tatweel removed.
    This is normalisation, not stemming -- no letters are added or dropped --
    and it is the single most common reason two spellings of the same Arabic
    name fail to match.
``number``
    ``1,234.50`` -> ``1234.5``. Row text renders decimals normalised
    (``Decimal.normalize``), so a question written with thousands separators
    tokenises to ``1 234 50`` and matches nothing at all.
``date``
    ``5 January 2023`` / ``January 5, 2023`` -> ``2023-01-05``, and
    ``January 2023`` -> ``2023-01``. Dates are serialised ISO in row text and
    the index tokenizer splits on the hyphens, so the ISO form is what actually
    matches.
``al_prefix``
    ``Al-Sayed`` / ``Al Sayed`` / ``Alsayed`` are the same surname written three
    ways, and this corpus contains all three as *different customers*.
``translit``
    A small closed table of Arabic-name transliteration variants
    (Ahmed/Ahmad, Mohamed/Mohammed/Muhammad, ...). One substitution per variant,
    never a cross product.
``acronym``
    A run of three or more consecutive capitalised words becomes its initialism
    (``Gulf National Bank`` -> ``GNB``). Gated at three words so that ordinary
    two-word personal names do not generate noise.

Conservatism
------------
`max_variants` (default 4) caps how many expansions are produced, and the rules
are ordered most-reliable-first so the cap trims the speculative tail rather
than the safe head. The original query is always first in the returned list and
is never dropped; the pipeline additionally weights variants below it during
fusion, so an expansion can add a result the original missed but cannot outvote
it. Expansion that hurts is a legitimate finding for the ablation to report --
expansion that is obviously reckless is not.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

__all__ = [
    "DEFAULT_EXPANDER",
    "DeterministicExpander",
    "EXPANDERS",
    "NoopExpander",
    "TRANSLITERATIONS",
    "get_expander",
]

# --------------------------------------------------------------------------
# Rule tables
# --------------------------------------------------------------------------

_PUNCT = re.compile(r"[-_/\\.,;:!?()\[\]{}\"'’«»]+")
_WS = re.compile(r"\s+")

_ARABIC_MARKS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_ARABIC_FOLD = str.maketrans(
    {
        "أ": "ا",  # أ -> ا
        "إ": "ا",  # إ -> ا
        "آ": "ا",  # آ -> ا
        "ٱ": "ا",  # ٱ -> ا
        "ى": "ي",  # ى -> ي
        "ة": "ه",  # ة -> ه
    }
)
_HAS_ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")

_MONTHS: dict[str, int] = {}
for _i, _names in enumerate(
    [
        ("january", "jan"),
        ("february", "feb"),
        ("march", "mar"),
        ("april", "apr"),
        ("may",),
        ("june", "jun"),
        ("july", "jul"),
        ("august", "aug"),
        ("september", "sep", "sept"),
        ("october", "oct"),
        ("november", "nov"),
        ("december", "dec"),
    ],
    start=1,
):
    for _name in _names:
        _MONTHS[_name] = _i

_DAY_MONTH_YEAR = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([A-Za-z]{3,9})\.?,?\s+(\d{4})\b"
)
_MONTH_DAY_YEAR = re.compile(
    r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b"
)
_MONTH_YEAR = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{4})\b")

_GROUPED_NUMBER = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")
_DECIMAL = re.compile(r"\b\d+\.\d*0\b")

_AL_HYPHEN = re.compile(r"\b([Aa]l)-(\w)")
_AL_SPACE = re.compile(r"\b([Aa]l)\s+(\w)")

#: Closed transliteration table. Every group is a set of spellings of one name;
#: a token matching any member expands to the others. Kept small and specific on
#: purpose -- a general vowel-folding rule would collapse distinct names (Hassan
#: / Hussein) and the corpus deliberately contains near-duplicates that must stay
#: distinguishable.
TRANSLITERATIONS: tuple[tuple[str, ...], ...] = (
    ("ahmed", "ahmad"),
    ("mohamed", "mohammed", "muhammad", "mohammad"),
    ("mahmoud", "mahmud"),
    ("hussein", "hussain", "husayn"),
    ("hassan", "hasan"),
    ("youssef", "yousef", "yusuf"),
    ("ibrahim", "ebrahim"),
    ("khaled", "khalid"),
    ("omar", "umar"),
    ("othman", "osman", "uthman"),
    ("sayed", "sayyed", "sayyid"),
    ("abdullah", "abdallah", "abdulla"),
    ("fatima", "fatma", "fatimah"),
    ("aisha", "aysha", "aicha"),
    ("nour", "noor", "nur"),
    ("layla", "leila", "laila"),
    ("saeed", "said", "sayid"),
    ("tarek", "tariq", "tarik"),
    ("gamal", "jamal"),
    ("gaber", "jaber", "jabir"),
)

_VARIANTS_OF: dict[str, tuple[str, ...]] = {
    form: tuple(f for f in group if f != form)
    for group in TRANSLITERATIONS
    for form in group
}

#: Words that never start an initialism, so "The National Bank of X" does not
#: become "TNB".
_ACRONYM_SKIP = frozenset({"the", "a", "an", "of", "and", "al", "el", "for"})


def _norm_space(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _key(text: str) -> str:
    """Dedup key: whitespace- and case-insensitive."""
    return _norm_space(text).casefold()


# --------------------------------------------------------------------------
# Individual rules -- each returns zero or more variants of `query`
# --------------------------------------------------------------------------


def _rule_punctuation(query: str) -> list[str]:
    out = _norm_space(_PUNCT.sub(" ", query))
    return [out] if out else []


def _rule_arabic(query: str) -> list[str]:
    if not _HAS_ARABIC.search(query):
        return []
    text = unicodedata.normalize("NFKC", query)
    text = _ARABIC_MARKS.sub("", text).translate(_ARABIC_FOLD)
    return [_norm_space(text)]


def _rule_number(query: str) -> list[str]:
    def strip_groups(m: re.Match[str]) -> str:
        return m.group(0).replace(",", "")

    text = _GROUPED_NUMBER.sub(strip_groups, query)

    def trim_zeros(m: re.Match[str]) -> str:
        value = m.group(0).rstrip("0").rstrip(".")
        return value or m.group(0)

    text = _DECIMAL.sub(trim_zeros, text)
    return [text] if text != query else []


def _iso(year: str, month: int, day: str | None = None) -> str:
    if day is None:
        return f"{int(year):04d}-{month:02d}"
    return f"{int(year):04d}-{month:02d}-{int(day):02d}"


def _rule_date(query: str) -> list[str]:
    text = query

    def dmy(m: re.Match[str]) -> str:
        month = _MONTHS.get(m.group(2).lower())
        return m.group(0) if month is None else _iso(m.group(3), month, m.group(1))

    def mdy(m: re.Match[str]) -> str:
        month = _MONTHS.get(m.group(1).lower())
        return m.group(0) if month is None else _iso(m.group(3), month, m.group(2))

    def my(m: re.Match[str]) -> str:
        month = _MONTHS.get(m.group(1).lower())
        return m.group(0) if month is None else _iso(m.group(2), month)

    text = _DAY_MONTH_YEAR.sub(dmy, text)
    text = _MONTH_DAY_YEAR.sub(mdy, text)
    text = _MONTH_YEAR.sub(my, text)
    return [text] if text != query else []


def _rule_al_prefix(query: str) -> list[str]:
    out: list[str] = []
    spaced = _AL_HYPHEN.sub(r"\1 \2", query)
    if spaced != query:
        out.append(spaced)
    joined = _AL_SPACE.sub(r"\1\2", _AL_HYPHEN.sub(r"\1\2", query))
    if joined != query:
        out.append(joined)
    hyphenated = _AL_SPACE.sub(r"\1-\2", query)
    if hyphenated != query:
        out.append(hyphenated)
    return out


def _rule_translit(query: str) -> list[str]:
    tokens = query.split()
    out: list[str] = []
    for i, token in enumerate(tokens):
        core = _PUNCT.sub("", token).lower()
        for alt in _VARIANTS_OF.get(core, ()):
            swapped = list(tokens)
            swapped[i] = _preserve_case(token, core, alt)
            out.append(" ".join(swapped))
    return out


def _preserve_case(original: str, core: str, replacement: str) -> str:
    """Substitute `replacement` for `core` inside `original`, keeping its shape."""
    lowered = original.lower()
    start = lowered.find(core)
    if start < 0:
        return replacement
    piece = original[start : start + len(core)]
    if piece.isupper():
        replacement = replacement.upper()
    elif piece[:1].isupper():
        replacement = replacement.capitalize()
    return original[:start] + replacement + original[start + len(core) :]


def _rule_acronym(query: str) -> list[str]:
    tokens = query.split()
    runs: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        word = _PUNCT.sub("", token)
        if word[:1].isupper() and word.isalpha() and word.lower() not in _ACRONYM_SKIP:
            current.append(word)
        else:
            if len(current) >= 3:
                runs.append(current)
            current = []
    if len(current) >= 3:
        runs.append(current)
    out: list[str] = []
    for run in runs:
        acronym = "".join(w[0] for w in run).upper()
        if 3 <= len(acronym) <= 6:
            phrase = " ".join(run)
            out.append(query.replace(phrase, acronym, 1))
    return out


#: Rules in application order: most reliable first, so `max_variants` trims the
#: speculative tail rather than the safe head.
_RULES: tuple[tuple[str, Callable[[str], list[str]]], ...] = (
    ("punctuation", _rule_punctuation),
    ("arabic", _rule_arabic),
    ("number", _rule_number),
    ("date", _rule_date),
    ("al_prefix", _rule_al_prefix),
    ("translit", _rule_translit),
    ("acronym", _rule_acronym),
)


# --------------------------------------------------------------------------
# Expanders
# --------------------------------------------------------------------------


class NoopExpander:
    """`QueryExpander` that expands nothing. What ``expansion=False`` means."""

    name = "noop"

    def expand(self, query: str) -> list[str]:
        return [query]

    def explain(self, query: str) -> list[tuple[str, str]]:
        return []

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "NoopExpander()"


@dataclass(frozen=True)
class DeterministicExpander:
    """Rule-based `QueryExpander`. The original query is always first.

    Variants are deduplicated case- and whitespace-insensitively against the
    original and against each other, so a rule that fires without changing
    anything costs nothing.
    """

    name: str = "deterministic"
    #: Maximum number of variants *beyond* the original query.
    max_variants: int = 4
    #: Rules to run, by name; None means all of them, in `_RULES` order.
    rules: tuple[str, ...] | None = None

    def _active(self) -> tuple[tuple[str, Callable[[str], list[str]]], ...]:
        if self.rules is None:
            return _RULES
        wanted = set(self.rules)
        unknown = wanted - {name for name, _fn in _RULES}
        if unknown:
            raise ValueError(f"unknown expansion rules: {sorted(unknown)}")
        return tuple((n, fn) for n, fn in _RULES if n in wanted)

    def explain(self, query: str) -> list[tuple[str, str]]:
        """``(rule, variant)`` pairs, in the order the variants are returned."""
        seen = {_key(query)}
        out: list[tuple[str, str]] = []
        if not query or not query.strip():
            return out
        for rule_name, fn in self._active():
            for variant in fn(query):
                variant = _norm_space(variant)
                key = _key(variant)
                if not variant or key in seen:
                    continue
                seen.add(key)
                out.append((rule_name, variant))
                if len(out) >= self.max_variants:
                    return out
        return out

    def expand(self, query: str) -> list[str]:
        return [query] + [variant for _rule, variant in self.explain(query)]


DEFAULT_EXPANDER = "deterministic"

EXPANDERS: dict[str, Callable[..., object]] = {
    "noop": NoopExpander,
    "none": NoopExpander,
    "deterministic": DeterministicExpander,
}


def get_expander(name: str = DEFAULT_EXPANDER, **kwargs: object) -> object:
    """Build an expander by name. Unknown names raise rather than defaulting."""
    try:
        factory = EXPANDERS[name]
    except KeyError:
        raise ValueError(
            f"unknown expander {name!r}; known: {sorted(EXPANDERS)}"
        ) from None
    return factory(**kwargs)  # type: ignore[arg-type]
