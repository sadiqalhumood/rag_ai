"""Rerankers: reorder fused candidates before the top-`k` cut.

Three implementations behind one interface (`anyrag.core.interfaces.Reranker`):

* `NoopReranker` -- identity, what runs when ``rerank=False``.
* `LexicalOverlapReranker` -- the offline default.
* `CrossEncoderReranker` -- optional ms-marco cross-encoder, lazily imported,
  falling back to the lexical reranker (loudly) when the model is unavailable.

Why the lexical reranker is not a no-op with a different name
-------------------------------------------------------------
"rerank on/off" is one axis of an 18-cell ablation. If the offline reranker only
shuffled ties, half the grid would be measuring nothing, so it has to carry a
signal the first-stage retrievers do not.

It does, in three ways:

1. **Query terms are IDF-weighted over the candidate set**, not counted. The
   candidates for "what is the email of customer Ahmed Al-Sayed" are all customer
   rows, so *email*, *customer* and the column vocabulary appear in every one of
   them and carry ~0 weight, while *Al-Sayed* appears in a handful and carries
   nearly all of it. Plain overlap ranks by how much schema vocabulary a chunk
   happens to contain -- which is close to ranking by chunk length. Dense
   retrieval cannot make this distinction at all (the discriminating token is a
   rare name, which is exactly what a 384-d sentence embedding smooths away),
   and BM25 makes it with *corpus* IDF, which is a different and coarser
   statistic than IDF over the retrieved set.
2. **Adjacency**. BM25 is a bag of words: "Ahmed Al-Sayed" and a row mentioning
   Ahmed Hassan and Mona Al-Sayed score identically. The reranker gives credit
   for query bigrams occurring *contiguously* in the chunk, which is the cheapest
   real form of phrase evidence and the one that separates the near-duplicate
   names this corpus is built around.
3. **The fused score is blended, not replaced** (`blend`, default 0.5), so
   first-stage evidence still counts and the reranker cannot promote a chunk no
   retriever liked.

Score scale
-----------
Rerankers emit scores in ``[0, 1]``: the fused score is divided by the best
fused score in the candidate set before blending (see `_scale`, which explains
why that is not min-max), and the overlap term is a coverage fraction. Fused
scores are already normalised to ``(0, 1]`` by
`anyrag.retrieval.fusion`, so "rerank on" and "rerank off" produce comparable
magnitudes and the absolute thresholds downstream (`min_support_score`) do not
silently mean different things in different cells of the grid.

Determinism
-----------
No randomness, no corpus statistics, no model state: the same query and the same
candidate list always produce the same order, with ties broken by `chunk_id`
exactly as the indexes and the fusion do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

from anyrag.core.types import Hit
from anyrag.index import FIRST_RANK, tokenize

__all__ = [
    "CrossEncoderReranker",
    "CROSS_ENCODER_MODEL",
    "DEFAULT_RERANKER",
    "LexicalOverlapReranker",
    "NoopReranker",
    "RERANK_RETRIEVER",
    "RERANKERS",
    "STOPWORDS",
    "get_reranker",
]

#: `Hit.retriever` on reranked hits. The pre-rerank provenance survives on
#: `FusedHit.ranks` / `.raw_scores`, so relabelling loses nothing.
RERANK_RETRIEVER = "rerank"

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: Deliberately small. Retrieval stopwords only need to stop *grammar* from
#: dominating the weighted overlap; the IDF weighting already flattens
#: schema vocabulary, and dropping a content word here would remove the term
#: that decides the ranking. Arabic function words are included because the
#: corpus is bilingual and the same argument applies to them.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then than that this these those of in on at by for
    with without from to into over under between among during about as is are
    was were be been being am do does did doing have has had having will would
    shall should can could may might must not no nor it its he she they them
    his her their we you your our i me my us who whom whose which what when
    where why how any all each every both some such only same so too very just
    also there here please tell show list find give name display return get
    من في على عن الى إلى ما ماذا كم هل هذا هذه ذلك التي الذي و ال أو او
    """.split()
)


def _content_terms(text: str) -> list[str]:
    """Tokens in order, stopwords removed, using the index tokenizer.

    Sharing `anyrag.index.tokenize` is deliberate: the reranker must agree with
    BM25 about what a token *is* (Unicode words, Arabic diacritics stripped,
    ISO dates split into their parts), otherwise it would rescore a document
    representation the retriever never saw.
    """
    return [t for t in tokenize(text) if t not in STOPWORDS]


def _bigrams(tokens: Sequence[str]) -> set[tuple[str, str]]:
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _scale(values: Sequence[float]) -> list[float]:
    """Divide by the best score in the candidate set, clamping negatives to 0.

    Deliberately *not* min-max. Min-max maps the worst candidate to 0 whatever
    its score, so a candidate set the first stage considers nearly tied
    (0.90, 0.89, 0.88) is stretched across the whole interval and the first
    stage ends up dominating a blend it barely has an opinion in. Dividing by
    the maximum preserves the ratios the first stage actually expressed: near
    ties stay near ties, and a genuine gap stays a gap.
    """
    if not values:
        return []
    hi = max(values)
    if hi <= 0.0:
        return [1.0] * len(values)
    return [max(0.0, v) / hi for v in values]


def _restamp(hit: Hit, score: float, rank: int, retriever: str) -> Hit:
    """Rewrite a hit's score/rank/retriever, preserving its concrete type.

    `dataclasses.replace` keeps `FusedHit` a `FusedHit`, so per-retriever ranks
    and raw scores survive reranking and the trace can still explain where a
    result came from.
    """
    return replace(hit, score=float(score), rank=rank, retriever=retriever)


def _rank(scored: Sequence[tuple[float, Hit]], retriever: str, k: int) -> list[Hit]:
    ordered = sorted(scored, key=lambda pair: (-pair[0], pair[1].chunk_id))
    if k > 0:
        ordered = ordered[:k]
    return [
        _restamp(hit, score, rank, retriever)
        for rank, (score, hit) in enumerate(ordered, start=FIRST_RANK)
    ]


# --------------------------------------------------------------------------
# Noop
# --------------------------------------------------------------------------


class NoopReranker:
    """Identity reranker: the order (and the scores) come out unchanged.

    Used when ``rerank=False``. It truncates to `k` and nothing else -- in
    particular it does not restamp scores, so a no-rerank cell reports exactly
    the fused score the fusion produced.
    """

    name = "noop"

    def rerank(self, query: str, hits: Sequence[Hit], k: int = 10) -> list[Hit]:
        out = list(hits)
        return out[:k] if k > 0 else out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "NoopReranker()"


# --------------------------------------------------------------------------
# Lexical overlap
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LexicalOverlapReranker:
    """Rescore candidates by IDF-weighted query-term coverage plus adjacency.

    ``final = (1 - blend) * minmax(fused) + blend * overlap`` where

    ``overlap = (1 - phrase_weight) * covered_weight/total_weight
                + phrase_weight * covered_bigrams/total_bigrams``

    and term weights are ``ln(1 + (N - df + 0.5)/(df + 0.5))`` with `df` counted
    over the candidate list itself -- the same non-negative Lucene IDF the BM25
    index uses, so a term present in every candidate contributes ~0 rather than
    a negative amount.

    Both components are fractions in ``[0, 1]``, so `final` is too.
    """

    name: str = "lexical_overlap"
    #: How much of the final score comes from overlap rather than first-stage
    #: evidence. 0.0 makes this a no-op; 1.0 discards the retrievers' opinion.
    blend: float = 0.5
    #: Share of the overlap term carried by contiguous query bigrams.
    phrase_weight: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError(f"blend must be in [0, 1], got {self.blend}")
        if not 0.0 <= self.phrase_weight <= 1.0:
            raise ValueError(
                f"phrase_weight must be in [0, 1], got {self.phrase_weight}"
            )

    # -- scoring ----------------------------------------------------------

    def overlap_scores(
        self, query: str, hits: Sequence[Hit]
    ) -> list[float]:
        """The overlap component alone, in the order of `hits`. Public for tests."""
        q_terms = _content_terms(query)
        q_unique = list(dict.fromkeys(q_terms))
        q_bigrams = _bigrams(q_terms)
        if not q_unique:
            return [0.0] * len(hits)

        docs = [_content_terms(h.chunk.text) for h in hits]
        doc_sets = [set(d) for d in docs]
        n = len(hits)

        df = {t: sum(1 for s in doc_sets if t in s) for t in q_unique}
        weights = {
            t: math.log(1.0 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in q_unique
        }
        total_w = sum(weights.values())
        if total_w <= 0.0:
            # Every query term occurs in every candidate: IDF says they are all
            # equally (un)informative, so fall back to plain coverage rather
            # than dividing by zero and calling everything a perfect match.
            weights = {t: 1.0 for t in q_unique}
            total_w = float(len(q_unique))

        out: list[float] = []
        for doc, doc_set in zip(docs, doc_sets):
            covered = sum(weights[t] for t in q_unique if t in doc_set)
            term_score = covered / total_w
            if q_bigrams:
                d_bigrams = _bigrams(doc)
                phrase_score = len(q_bigrams & d_bigrams) / len(q_bigrams)
                score = (
                    1.0 - self.phrase_weight
                ) * term_score + self.phrase_weight * phrase_score
            else:
                score = term_score
            out.append(score)
        return out

    def rerank(self, query: str, hits: Sequence[Hit], k: int = 10) -> list[Hit]:
        hits = list(hits)
        if not hits:
            return []
        overlaps = self.overlap_scores(query, hits)
        base = _scale([float(h.score) for h in hits])
        scored = [
            ((1.0 - self.blend) * b + self.blend * o, hit)
            for hit, b, o in zip(hits, base, overlaps)
        ]
        return _rank(scored, RERANK_RETRIEVER, k)


# --------------------------------------------------------------------------
# Cross-encoder (optional)
# --------------------------------------------------------------------------


class CrossEncoderReranker:
    """Cross-encoder reranker, imported lazily and degrading loudly.

    The model is only imported and downloaded on the first `rerank` call, so
    constructing one costs nothing and an environment without
    sentence-transformers can still build the pipeline. If loading or scoring
    fails for any reason, the call is served by `fallback` (a
    `LexicalOverlapReranker` unless told otherwise) and the failure is recorded
    on `fell_back` / `fallback_reason` / `info` rather than raised: a missing
    model must cost result *quality*, visibly, not the whole eval run.

    Raw cross-encoder outputs are logits on an arbitrary scale, so they are
    squashed with a logistic into ``[0, 1]`` -- monotone, so the ordering is the
    model's, but the scale matches the other rerankers and the absolute
    thresholds downstream.
    """

    name = "cross_encoder"

    def __init__(
        self,
        model_name: str = CROSS_ENCODER_MODEL,
        *,
        fallback: Any | None = None,
        max_pairs: int = 100,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.fallback = fallback if fallback is not None else LexicalOverlapReranker()
        self.max_pairs = int(max_pairs)
        self.device = device
        self.fell_back = False
        self.fallback_reason = ""
        self._model: Any | None = None
        self._load_attempted = False

    # -- lazy model -------------------------------------------------------

    def _load(self) -> Any | None:
        if self._model is not None or self._load_attempted:
            return self._model
        self._load_attempted = True
        try:  # pragma: no cover - depends on the environment
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            kwargs = {"device": self.device} if self.device else {}
            self._model = CrossEncoder(self.model_name, **kwargs)
        except Exception as exc:  # pragma: no cover - environment dependent
            self._note_fallback(f"{type(exc).__name__}: {exc}")
            self._model = None
        return self._model

    def _note_fallback(self, reason: str) -> None:
        self.fell_back = True
        self.fallback_reason = reason

    @property
    def active_name(self) -> str:
        """Which reranker actually produced the last result."""
        if self.fell_back:
            return f"{self.name}->{getattr(self.fallback, 'name', 'fallback')}"
        return self.name

    @property
    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model_name,
            "active": self.active_name,
            "fell_back": self.fell_back,
            "reason": self.fallback_reason,
            "degraded": self.fell_back,
        }

    # -- scoring ----------------------------------------------------------

    def rerank(self, query: str, hits: Sequence[Hit], k: int = 10) -> list[Hit]:
        hits = list(hits)
        if not hits:
            return []
        model = self._load()
        if model is None:
            return self.fallback.rerank(query, hits, k)

        scoring = hits[: self.max_pairs] if self.max_pairs > 0 else hits
        try:  # pragma: no cover - environment dependent
            raw = model.predict([(query, h.chunk.text) for h in scoring])
            values = [float(v) for v in raw]
        except Exception as exc:  # pragma: no cover - environment dependent
            self._note_fallback(f"predict failed: {type(exc).__name__}: {exc}")
            return self.fallback.rerank(query, hits, k)

        scored = [
            (1.0 / (1.0 + math.exp(-v)), hit) for hit, v in zip(scoring, values)
        ]
        # Candidates beyond max_pairs were not scored by the model; they keep
        # their first-stage order below everything that was, rather than being
        # dropped silently.
        tail = hits[len(scoring) :]
        ranked = _rank(scored, RERANK_RETRIEVER, 0)
        base = len(ranked)
        ranked.extend(
            _restamp(h, 0.0, base + i + FIRST_RANK, RERANK_RETRIEVER)
            for i, h in enumerate(tail)
        )
        return ranked[:k] if k > 0 else ranked

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CrossEncoderReranker(model={self.model_name!r}, fell_back={self.fell_back})"


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

#: The reranker used when a config turns reranking on without naming one.
DEFAULT_RERANKER = "lexical_overlap"

RERANKERS: dict[str, Callable[[], Any]] = {
    "noop": NoopReranker,
    "none": NoopReranker,
    "lexical": LexicalOverlapReranker,
    "lexical_overlap": LexicalOverlapReranker,
    "cross_encoder": CrossEncoderReranker,
    "cross-encoder": CrossEncoderReranker,
}


def get_reranker(name: str = DEFAULT_RERANKER, **kwargs: Any) -> Any:
    """Build a reranker by name. Unknown names raise rather than defaulting."""
    try:
        factory = RERANKERS[name]
    except KeyError:
        raise ValueError(
            f"unknown reranker {name!r}; known: {sorted(RERANKERS)}"
        ) from None
    return factory(**kwargs)
