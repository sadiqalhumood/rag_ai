"""Fusing several ranked lists into one.

Two fusion functions, one default
---------------------------------
* `rrf` -- reciprocal rank fusion, ``sum_l w_l / (k + rank_l)``. **The default.**
* `normalized_score_fusion` -- min-max normalise each list's scores, then sum.

RRF is the default because the two retrievers being fused produce scores on
incomparable scales: dense search returns a cosine in ``[-1, 1]`` while BM25
returns an unbounded sum of IDF-weighted term contributions whose magnitude
depends on the corpus, the query length and the document length. Any score-level
combination has to invent a mapping between those scales, and min-max does it
per query, so the mapping changes from question to question -- a BM25 top score
of 18 becomes 1.0 for a one-word query and 1.0 again for a five-word query with
a much better match. Rank is the one quantity both retrievers agree on.
`normalized_score_fusion` is implemented anyway so the ablation can *measure*
that claim instead of asserting it.

Rank convention
---------------
`Hit.rank` is 1-based (`anyrag.index.FIRST_RANK`), so the RRF term is
``1 / (rrf_k + hit.rank)`` with no off-by-one adjustment and no division by zero.
A hit whose `rank` is left at the `Hit` default of 0 is treated as unranked and
gets its position in the list instead.

Score scale
-----------
Both functions normalise by the maximum attainable fused score (``normalize=True``
by default), i.e. the score of a chunk that placed first in every list. That is
division by a constant, so the ordering and every tie are bit-identical to the
un-normalised computation -- but it puts fused scores in ``(0, 1]`` regardless of
how many lists were fused. That matters downstream: `RetrievalConfig.min_score`
and `GenerationConfig.min_support_score` are *absolute* thresholds, and raw RRF
would make them mean something different in every cell of the ablation grid
(raw top score with one list is ``1/61 = 0.016``, with two lists ``0.033``,
with expansion on, something else again). Pass ``normalize=False`` for the
textbook quantity.

Provenance
----------
Fused hits carry ``retriever="rrf"`` and are `FusedHit` instances: a `Hit`
subclass that also records, per contributing retriever, the best rank, the raw
score, and how much that retriever contributed to the fused score. Nothing in
`anyrag.core` changes -- `FusedHit` is a `Hit` everywhere a `Hit` is expected.

Determinism
-----------
Ties are broken by `chunk_id`, matching what the indexes already do, so every
ablation cell is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from anyrag.core.types import Chunk, Hit
from anyrag.index import FIRST_RANK

__all__ = [
    "DEFAULT_FUSION",
    "DEFAULT_RRF_K",
    "FUSIONS",
    "FusedHit",
    "NORM_RETRIEVER",
    "RRF_RETRIEVER",
    "fuse",
    "normalized_score_fusion",
    "rrf",
]

#: `Hit.retriever` on the output of each fusion function.
RRF_RETRIEVER = "rrf"
NORM_RETRIEVER = "norm"

DEFAULT_RRF_K = 60
DEFAULT_FUSION = "rrf"


@dataclass(frozen=True)
class FusedHit(Hit):
    """A `Hit` that remembers which retrievers produced it, and where.

    `ranks` maps retriever label -> best (lowest) rank that retriever gave the
    chunk; `raw_scores` the score at that rank; `contributions` how much each
    retriever added to `score`. Callers that want to explain a result -- or an
    eval that wants to count how often fusion changed the winner -- read these
    instead of re-running the retrievers.
    """

    ranks: Mapping[str, int] = field(default_factory=dict)
    raw_scores: Mapping[str, float] = field(default_factory=dict)
    contributions: Mapping[str, float] = field(default_factory=dict)

    @property
    def retrievers(self) -> tuple[str, ...]:
        return tuple(sorted(self.ranks))

    @property
    def n_retrievers(self) -> int:
        return len(self.ranks)

    def best_rank(self) -> int:
        return min(self.ranks.values()) if self.ranks else 0


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------


class _Acc:
    """Accumulator for one chunk across every list it appears in."""

    __slots__ = ("chunk", "score", "ranks", "raw", "contrib")

    def __init__(self, chunk: Chunk) -> None:
        self.chunk = chunk
        self.score = 0.0
        self.ranks: dict[str, int] = {}
        self.raw: dict[str, float] = {}
        self.contrib: dict[str, float] = {}

    def observe(self, label: str, rank: int, raw: float, contribution: float) -> None:
        self.score += contribution
        self.contrib[label] = self.contrib.get(label, 0.0) + contribution
        prev = self.ranks.get(label)
        if prev is None or rank < prev:
            self.ranks[label] = rank
            self.raw[label] = raw


def _prepare(
    ranked_lists: Sequence[Sequence[Hit]],
    labels: Sequence[str] | None,
    weights: Sequence[float] | None,
) -> list[tuple[str, float, Sequence[Hit]]]:
    lists = [list(lst) for lst in ranked_lists]
    if labels is not None and len(labels) != len(lists):
        raise ValueError(
            f"got {len(labels)} labels for {len(lists)} ranked lists"
        )
    if weights is not None and len(weights) != len(lists):
        raise ValueError(
            f"got {len(weights)} weights for {len(lists)} ranked lists"
        )
    out: list[tuple[str, float, Sequence[Hit]]] = []
    for i, hits in enumerate(lists):
        if labels is not None:
            label = labels[i]
        else:
            label = hits[0].retriever if hits and hits[0].retriever else f"list{i}"
        weight = float(weights[i]) if weights is not None else 1.0
        out.append((label, weight, hits))
    return out


def _rank_of(hit: Hit, position: int) -> int:
    """`hit.rank` when the producer set one, else the list position.

    `Hit.rank` defaults to 0, which under the 1-based convention means
    "unranked" rather than "best" -- so an unranked list is ranked by position
    instead of every hit claiming ``1/(k+0)``.
    """
    return hit.rank if hit.rank >= FIRST_RANK else position + FIRST_RANK


def _finalise(
    acc: dict[str, _Acc], retriever: str, divisor: float
) -> list[FusedHit]:
    if divisor and divisor != 1.0:
        for entry in acc.values():
            entry.score /= divisor
            entry.contrib = {k: v / divisor for k, v in entry.contrib.items()}
    ordered = sorted(acc.items(), key=lambda kv: (-kv[1].score, kv[0]))
    return [
        FusedHit(
            chunk=entry.chunk,
            score=float(entry.score),
            rank=rank,
            retriever=retriever,
            ranks=dict(entry.ranks),
            raw_scores=dict(entry.raw),
            contributions=dict(entry.contrib),
        )
        for rank, (_cid, entry) in enumerate(ordered, start=FIRST_RANK)
    ]


# --------------------------------------------------------------------------
# Reciprocal rank fusion
# --------------------------------------------------------------------------


def rrf(
    ranked_lists: Sequence[Sequence[Hit]],
    k: int = DEFAULT_RRF_K,
    *,
    labels: Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
    normalize: bool = True,
    retriever: str = RRF_RETRIEVER,
) -> list[FusedHit]:
    """Reciprocal rank fusion: ``score(c) = sum_l w_l / (k + rank_l(c))``.

    A chunk that both retrievers rank highly necessarily beats one that only a
    single retriever ranks highly, as long as the second retriever's rank is
    better than ``k`` positions worse -- which is the whole point of the damping
    constant. With the default ``k=60``, a chunk at rank 1 in both lists scores
    ``2/61``, while a chunk at rank 1 in one list alone scores ``1/61``: no rank
    in a single list can beat agreement between two.

    `weights` lets the caller discount lists it trusts less -- the pipeline uses
    it to damp query-expansion variants so an expansion cannot outvote the
    original query. Lists sharing a label (the same retriever run over several
    expanded queries) sum their contributions and keep the best rank.

    Returns `FusedHit`s in descending score order, `rank` starting at
    `FIRST_RANK`, ties broken by `chunk_id`.
    """
    if k < 0:
        raise ValueError(f"rrf k must be non-negative, got {k}")
    prepared = _prepare(ranked_lists, labels, weights)

    acc: dict[str, _Acc] = {}
    for label, weight, hits in prepared:
        for position, hit in enumerate(hits):
            rank = _rank_of(hit, position)
            entry = acc.get(hit.chunk_id)
            if entry is None:
                entry = acc[hit.chunk_id] = _Acc(hit.chunk)
            entry.observe(label, rank, float(hit.score), weight / (k + rank))

    divisor = 1.0
    if normalize:
        best = sum(w for _l, w, hits in prepared if hits) / (k + FIRST_RANK)
        divisor = best or 1.0
    return _finalise(acc, retriever, divisor)


# --------------------------------------------------------------------------
# Score-normalisation fusion (the alternative)
# --------------------------------------------------------------------------


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        # Every hit in the list scored the same. Calling them all equally good
        # (1.0) keeps the list's vote intact; spreading them 0..1 would invent
        # an ordering the retriever did not express.
        return [1.0] * len(values)
    span = hi - lo
    return [(v - lo) / span for v in values]


def normalized_score_fusion(
    ranked_lists: Sequence[Sequence[Hit]],
    *,
    labels: Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
    normalize: bool = True,
    retriever: str = NORM_RETRIEVER,
) -> list[FusedHit]:
    """Fuse on min-max normalised scores instead of ranks.

    Each list's scores are mapped onto ``[0, 1]`` and summed with `weights`;
    a chunk missing from a list contributes nothing from it. This keeps score
    *margins* that RRF throws away -- a dense hit at 0.91 against a runner-up at
    0.42 says more than "rank 1 vs rank 2" -- at the cost of making the mapping
    query-dependent, since the normalisation constants come from the candidate
    list itself. Offered as the fusion alternative so the grid can measure the
    trade-off rather than argue about it.
    """
    prepared = _prepare(ranked_lists, labels, weights)

    acc: dict[str, _Acc] = {}
    for label, weight, hits in prepared:
        normed = _minmax([float(h.score) for h in hits])
        for position, (hit, value) in enumerate(zip(hits, normed)):
            rank = _rank_of(hit, position)
            entry = acc.get(hit.chunk_id)
            if entry is None:
                entry = acc[hit.chunk_id] = _Acc(hit.chunk)
            entry.observe(label, rank, float(hit.score), weight * value)

    divisor = 1.0
    if normalize:
        divisor = sum(w for _l, w, hits in prepared if hits) or 1.0
    return _finalise(acc, retriever, divisor)


#: Fusion strategies addressable by name, for config-driven sweeps.
FUSIONS: dict[str, Callable[..., list[FusedHit]]] = {
    "rrf": rrf,
    "norm": normalized_score_fusion,
    "score": normalized_score_fusion,
}


def fuse(
    name: str,
    ranked_lists: Sequence[Sequence[Hit]],
    **kwargs: Any,
) -> list[FusedHit]:
    """Dispatch to a fusion function by name (``"rrf"`` or ``"norm"``)."""
    try:
        fn = FUSIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown fusion {name!r}; known: {sorted(FUSIONS)}"
        ) from None
    if fn is not rrf:
        kwargs.pop("k", None)
    return fn(ranked_lists, **kwargs)
