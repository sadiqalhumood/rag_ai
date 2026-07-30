"""Refusal policy: decide whether the retrieved context supports an answer.

This is the single most important behaviour in the system. The eval's headline
metric is the false-answer rate on questions whose answers are genuinely absent
from the data, and every one of those is won or lost here.

The design problem is that the obvious signal -- plain lexical overlap between
the question and the best chunk -- does not discriminate. Consider a database of
customers and these two questions:

    "What is the email of customer Ahmed Al-Sayed?"   (answerable)
    "What is the email of customer Zanzibar Petrov?"  (distractor)

Both retrieve customer rows. Both share {email, customer} with the best chunk.
Plain overlap gives roughly 0.5 for the distractor -- comfortably above any
threshold low enough to let real questions through. The generic schema words
carry the score, and the one term that actually decides answerability is a
rounding error.

So overlap here is **inverse-document-frequency weighted**, with the document
frequencies computed over the retrieved chunk set itself (no corpus statistics
required, nothing to persist). Terms that appear in most retrieved chunks --
"email", "customer", "order" -- are nearly free; a term that appears in none of
them carries the maximum weight. That inverts the failure above: the distractor's
weight mass sits on the two terms that are missing.

Weighted overlap alone still leaves the distractor uncomfortably close to the
threshold, so a second, harder gate follows it: **entity coverage**. Question
tokens that look like literal values rather than grammar -- capitalised
mid-sentence, containing digits, or written in a non-Latin script -- must
actually occur somewhere in the retrieved text. A question that names a customer
who does not exist fails this outright, while "how many orders were placed in
2023" passes as long as 2023 appears in the data.

Four gates, all of which must pass before an answer is attempted:

    G1 support score    best hit score           >= min_support_score
    G2 support count    #hits over that score    >= min_support_chunks
    G3 lexical overlap  best weighted overlap    >= min_overlap
    G4 entity coverage  grounded entity fraction >= min_entity_coverage

G1-G3 are tunable through `GenerationConfig`. G4 has no home in that frozen
dataclass, so it lives on `RefusalPolicy`, which `from_config` derives from a
`GenerationConfig` and which callers may override field by field.

Over-refusing is bad; answering a distractor is worse. Where the two trade off,
this module refuses.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from ..core.config import GenerationConfig
from ..core.types import Hit

__all__ = [
    "RefusalPolicy",
    "RefusalDecision",
    "TermWeights",
    "assess",
    "entity_terms",
    "term_weights",
    "terms",
    "weighted_overlap",
]


# --------------------------------------------------------------------------
# Lexical analysis
# --------------------------------------------------------------------------

#: Word characters in any script, underscores excluded so snake_case column
#: names split into their parts ("customer_id" -> customer, id).
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

#: CJK has no word delimiters, so a regex "word" is a whole run of characters.
#: Runs are split per character: matching is then per-ideograph, which is the
#: closest cheap analogue of word matching for those scripts.
_CJK_RANGES = (
    (0x3040, 0x30FF),  # kana
    (0x3400, 0x4DBF),  # CJK ext A
    (0x4E00, 0x9FFF),  # CJK unified
    (0xF900, 0xFAFF),  # compatibility ideographs
    (0xAC00, 0xD7AF),  # hangul syllables
)

_ARABIC_DIACRITICS = re.compile(r"[ً-ْٰـ]")

#: Deliberately generous. A stopword that slips through only costs a little
#: precision in the overlap score; a content word wrongly listed here silently
#: removes the term that decides answerability.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then than that this these those of in on at by for
    with without from to into over under between among during about as is are
    was were be been being am do does did doing have has had having will would
    shall should can could may might must not no nor its it he she they them
    his her their we you your our i me my us who whom whose which what when
    where why how many much more most less least any all each every both few
    some other another such only own same so too very just also there here
    please tell show list find give name display return get fetch provide
    total number count sum average avg min max mean number_of
    من في على عن الى إلى ما ماذا كم هل هذا هذه ذلك التي الذي و ال او أو
    """.split()
)

#: Multi-character CJK stopwords are not attempted; single characters are too
#: information-dense to drop.


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _normalize(token: str) -> str:
    """Casefold, strip Arabic diacritics, unify Arabic letter variants.

    "Ahmed Al-Sayed" and "ahmed al sayed" must produce the same terms, and the
    corpus deliberately contains near-duplicate Arabic spellings.
    """
    tok = unicodedata.normalize("NFKC", token).casefold()
    tok = _ARABIC_DIACRITICS.sub("", tok)
    tok = (
        tok.replace("أ", "ا")  # alef hamza above -> alef
        .replace("إ", "ا")  # alef hamza below -> alef
        .replace("آ", "ا")  # alef madda -> alef
        .replace("ة", "ه")  # teh marbuta -> heh
        .replace("ى", "ي")  # alef maksura -> yeh
    )
    # Crude singularisation: "orders" and "order" must match. Only applied to
    # tokens long enough that the trailing s is unlikely to be part of a stem.
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        tok = tok[:-1]
    return tok


def _split_cjk(token: str) -> list[str]:
    """Split a raw token into per-character CJK pieces and Latin/digit runs."""
    if not any(_is_cjk(c) for c in token):
        return [token]
    out: list[str] = []
    buf = ""
    for ch in token:
        if _is_cjk(ch):
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def _raw_tokens(text: str) -> list[str]:
    out: list[str] = []
    for match in _WORD_RE.finditer(text or ""):
        out.extend(_split_cjk(match.group(0)))
    return out


def _keep(term: str) -> bool:
    if not term or term in STOPWORDS:
        return False
    if len(term) >= 2:
        return True
    # Single characters survive only when they carry real information: a CJK
    # ideograph or a digit.
    return _is_cjk(term) or term.isdigit()


def terms(text: str) -> tuple[str, ...]:
    """Content terms of `text`, normalised, stopwords removed, order preserved."""
    return tuple(t for t in (_normalize(r) for r in _raw_tokens(text)) if _keep(t))


def unique_terms(text: str) -> tuple[str, ...]:
    """`terms` with duplicates removed, sorted for deterministic iteration."""
    return tuple(sorted(set(terms(text))))


def _is_caseless(token: str) -> bool:
    """True for scripts with no capitalisation: Arabic, CJK, Hebrew, Thai."""
    return token.lower() == token.upper()


#: Minimum length for a token in a caseless script to be treated as an entity.
#: Arabic function words are short (ما, هو, في) while names and places are not
#: (أحمد, السيد, القاهرة). Without this floor the entity gate fires on the
#: grammar of every Arabic question and refuses the lot.
CASELESS_ENTITY_MIN_LENGTH = 4


def entity_terms(
    question: str, *, caseless_min_length: int = CASELESS_ENTITY_MIN_LENGTH
) -> tuple[str, ...]:
    """Question tokens that look like literal *values* rather than grammar.

    Three cheap surface cues, none of which needs a tagger:

    * an embedded digit -- years, ids, quantities;
    * an initial capital, in a script that *has* capitals -- proper nouns, and
      the corpus's person, product and region names all carry one;
    * in a caseless script, a token long enough not to be a function word.

    Sentence-initial capitals are *not* excluded, because a question may open
    with the entity ("Ahmed Al-Sayed's email?"). The leading interrogatives and
    imperatives that would otherwise be false positives are in STOPWORDS.

    Two consequences worth knowing. An all-lowercase Latin question with no
    digits yields no entity terms, so gate G4 abstains and G3 carries the
    decision alone. And in caseless scripts the length floor is a blunt
    proxy -- a long common noun will be treated as an entity -- which is why G4
    is a *coverage ratio* rather than a demand that every entity be grounded.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in _raw_tokens(question):
        norm = _normalize(raw)
        if not _keep(norm) or norm in seen:
            continue
        if any(c.isdigit() for c in raw):
            entityish = True
        elif _is_caseless(raw):
            entityish = len(norm) >= caseless_min_length
        else:
            entityish = raw[:1].isupper()
        if entityish:
            seen.add(norm)
            out.append(norm)
    return tuple(out)


class TermWeights:
    """Inverse-document-frequency weights over a small set of texts.

    Document frequencies come from the retrieved chunks themselves, which is the
    whole point: it takes no corpus statistics, no persistence and no second
    pass, and it is exactly the distribution that decides whether a question term
    is generic *for this retrieval* or is the one term that matters.
    """

    __slots__ = ("_df", "_n", "_max")

    def __init__(self, texts: Sequence[str]) -> None:
        self._n = len(texts)
        df: dict[str, int] = {}
        for text in texts:
            for term in set(terms(text)):
                df[term] = df.get(term, 0) + 1
        self._df = df
        self._max = math.log(1.0 + float(self._n)) if self._n else 1.0

    def __len__(self) -> int:
        return self._n

    def weight(self, term: str) -> float:
        if not self._n:
            return 1.0
        df = self._df.get(term, 0)
        return math.log(1.0 + self._n / (1.0 + df))

    def document_frequency(self, term: str) -> int:
        return self._df.get(term, 0)

    @property
    def max_weight(self) -> float:
        return self._max


def term_weights(texts: Sequence[str]) -> TermWeights:
    return TermWeights(list(texts))


def weighted_overlap(
    question_terms: Sequence[str],
    text: str,
    weights: TermWeights | None = None,
) -> float:
    """Fraction of the question's *weight mass* that `text` covers.

    Returns 0.0 for an empty question. With `weights` omitted this degrades to
    plain unweighted term coverage, which is useful for ranking segments inside
    a single chunk where idf carries no signal.

    Iteration is over a sorted unique list rather than a set: Python randomises
    string hashing per process, and float addition is not associative, so set
    iteration would make the score -- and any sort keyed on it -- differ between
    runs of an otherwise deterministic generator.
    """
    q_unique = sorted(set(question_terms))
    if not q_unique:
        return 0.0
    present = set(terms(text))
    if weights is None:
        return sum(1.0 for t in q_unique if t in present) / float(len(q_unique))
    denom = 0.0
    numer = 0.0
    for t in q_unique:
        w = weights.weight(t)
        denom += w
        if t in present:
            numer += w
    return (numer / denom) if denom else 0.0


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RefusalPolicy:
    """Thresholds for the four gates.

    `from_config` takes the three that `GenerationConfig` owns; the rest are
    defaults here because that dataclass is frozen and orchestrator-owned.
    Every field is overridable per call, so the eval harness can sweep them.
    """

    #: G1/G2. A hit below this score is not evidence.
    min_support_score: float = 0.02
    #: G2. How many hits must clear `min_support_score`.
    min_support_chunks: int = 1
    #: G3. Applied to the *idf-weighted* overlap, not plain term overlap.
    min_overlap: float = 0.18
    #: G4. Fraction of the question's entity-like terms that must appear
    #: somewhere in the retrieved text. 0.75 means a two-entity question
    #: tolerates zero misses and a four-entity question tolerates one.
    min_entity_coverage: float = 0.75
    #: G4 off switch, for ablations and for corpora with no proper nouns.
    require_entity_coverage: bool = True
    #: G3 considers this many top-scoring hits, not just the single best.
    overlap_top_k: int = 5
    #: G4, caseless scripts only. See `entity_terms`.
    caseless_entity_min_length: int = CASELESS_ENTITY_MIN_LENGTH
    #: Optional floor on *unweighted* overlap. Off by default: the weighted
    #: metric is strictly more informative, and stacking both over-refuses.
    min_plain_overlap: float = 0.0

    @classmethod
    def from_config(cls, config: GenerationConfig, **overrides: Any) -> "RefusalPolicy":
        base = cls(
            min_support_score=config.min_support_score,
            min_support_chunks=config.min_support_chunks,
            min_overlap=config.min_overlap,
        )
        return replace(base, **overrides) if overrides else base

    def with_(self, **kw: Any) -> "RefusalPolicy":
        return replace(self, **kw)


@dataclass(frozen=True)
class RefusalDecision:
    """Structured outcome of the gates.

    `code` is stable and machine-readable (it is what the eval groups on);
    `detail` is the human sentence that ends up in `Answer.reason`.
    """

    refuse: bool
    code: str
    detail: str
    best_score: float = 0.0
    supporting_chunks: int = 0
    best_overlap: float = 0.0
    plain_overlap: float = 0.0
    entity_coverage: float = 1.0
    ungrounded_entities: tuple[str, ...] = ()
    n_hits: int = 0
    gates: Mapping[str, bool] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.gates is None:
            object.__setattr__(self, "gates", {})

    def as_trace(self) -> dict[str, Any]:
        return {
            "refused": self.refuse,
            "code": self.code,
            "detail": self.detail,
            "best_score": round(self.best_score, 6),
            "supporting_chunks": self.supporting_chunks,
            "best_weighted_overlap": round(self.best_overlap, 6),
            "best_plain_overlap": round(self.plain_overlap, 6),
            "entity_coverage": round(self.entity_coverage, 6),
            "ungrounded_entities": list(self.ungrounded_entities),
            "n_hits": self.n_hits,
            "gates": dict(self.gates),
        }


def _ordered_hits(hits: Sequence[Hit]) -> list[Hit]:
    """Score-descending, with input order as a total tie-break."""
    return [h for _, h in sorted(
        enumerate(hits), key=lambda p: (-p[1].score, p[1].rank, p[0])
    )]


def assess(
    question: str,
    hits: Sequence[Hit],
    config: GenerationConfig | None = None,
    *,
    policy: RefusalPolicy | None = None,
) -> RefusalDecision:
    """Run the four gates. Returns a decision; never raises on odd input."""
    if policy is None:
        policy = RefusalPolicy.from_config(config or GenerationConfig())

    hits = list(hits or ())
    gates: dict[str, bool] = {}

    if not hits:
        return RefusalDecision(
            refuse=True,
            code="no_hits",
            detail="retrieval returned no candidate chunks for this question",
            gates={"support_score": False, "support_count": False,
                   "lexical_overlap": False, "entity_coverage": False},
        )

    ordered = _ordered_hits(hits)
    best_score = ordered[0].score
    supporting = [h for h in hits if h.score >= policy.min_support_score]

    weights = term_weights([h.chunk.text or "" for h in hits])
    q_terms = unique_terms(question)

    considered = ordered[: max(1, policy.overlap_top_k)]
    if q_terms:
        best_overlap = max(
            weighted_overlap(q_terms, h.chunk.text or "", weights) for h in considered
        )
        plain_overlap = max(
            weighted_overlap(q_terms, h.chunk.text or "", None) for h in considered
        )
    else:
        # A question made entirely of stopwords carries no lexical signal at
        # all; the overlap gates abstain rather than refuse everything.
        best_overlap = 0.0
        plain_overlap = 0.0

    ents = entity_terms(
        question, caseless_min_length=policy.caseless_entity_min_length
    )
    if ents:
        grounded = tuple(e for e in ents if weights.document_frequency(e) > 0)
        ungrounded = tuple(e for e in ents if weights.document_frequency(e) == 0)
        coverage = len(grounded) / float(len(ents))
    else:
        ungrounded = ()
        coverage = 1.0

    gates["support_score"] = best_score >= policy.min_support_score
    gates["support_count"] = len(supporting) >= policy.min_support_chunks
    gates["lexical_overlap"] = (
        (not q_terms)
        or (best_overlap >= policy.min_overlap
            and plain_overlap >= policy.min_plain_overlap)
    )
    gates["entity_coverage"] = (
        (not policy.require_entity_coverage)
        or (not ents)
        or coverage >= policy.min_entity_coverage
    )

    def decide(code: str, detail: str) -> RefusalDecision:
        return RefusalDecision(
            refuse=code != "supported",
            code=code,
            detail=detail,
            best_score=best_score,
            supporting_chunks=len(supporting),
            best_overlap=best_overlap,
            plain_overlap=plain_overlap,
            entity_coverage=coverage,
            ungrounded_entities=ungrounded,
            n_hits=len(hits),
            gates=gates,
        )

    if not gates["support_score"]:
        return decide(
            "low_support_score",
            f"best retrieved chunk scored {best_score:.4f}, "
            f"below the support floor of {policy.min_support_score}",
        )
    if not gates["support_count"]:
        return decide(
            "insufficient_support_chunks",
            f"only {len(supporting)} chunk(s) cleared the support floor; "
            f"{policy.min_support_chunks} required",
        )
    if not gates["lexical_overlap"]:
        return decide(
            "low_lexical_overlap",
            f"best chunk covers {best_overlap:.3f} of the question's weighted "
            f"terms, below the required {policy.min_overlap}",
        )
    if not gates["entity_coverage"]:
        return decide(
            "ungrounded_entities",
            "the retrieved data does not mention "
            + ", ".join(repr(e) for e in ungrounded),
        )
    return decide("supported", "retrieved context supports an answer")
