"""Exact metrics. No model in the loop, anywhere.

Every number here is arithmetic over known-gold sets. The conventions below are
spelled out because a metric whose definition is implicit is a metric nobody can
check -- and a wrong nDCG silently invalidates an entire report, which is why
`tests/test_metrics.py` pins each one against hand-computed cases.

Conventions
-----------

**Gold keys.** A question's gold is a set of *keys*, not just rows, so that
schema questions grade on the same machinery as row questions:

* ``("row", table, pk)``   -- a source row the answer must rest on.
* ``("schema", table)``    -- the schema card for a table.

**Relevance.** A retrieved chunk is *relevant* iff it covers at least one gold
key: its ``row_refs`` intersect the gold row refs, or it is a ``SCHEMA_CARD``
for a gold table.

**recall@k** is *gold coverage*: the fraction of gold keys covered by the top-k
chunks. For a single-row gold it degenerates to the usual hit rate; for a
multi-row gold it correctly refuses to call one row out of eight a success.
``hit@k`` (did any relevant chunk appear in the top k) is reported alongside,
because the two answer different questions and averaging only one of them hides
the multi-row cases.

**Rank** is taken from the *position in the returned list*, not from
``Hit.rank``. The convention (D13) is that ``rank`` is 1-based with 0 meaning
"unranked", so a retriever that forgets to populate it would otherwise make
every hit look like a tie. List order is what the caller actually sees.

**MRR** is the reciprocal rank of the first relevant chunk, 0.0 if none.

**nDCG@k** uses binary gains over *novel* coverage: a chunk scores 1 only if it
covers a gold key no higher-ranked chunk already covered. Without that, two
chunks of the same row would both score, and DCG could exceed the ideal, pushing
nDCG above 1.0. The ideal is ``min(n_gold, k)`` chunks at ranks 1..n, which is
achievable exactly when the corpus contains a chunk per gold key -- which it
does, since the harness generated the rows.

**Aggregate correctness.** Integer gold (counts) must match *exactly*: a count
is either right or wrong. Float gold uses relative tolerance
``AGG_REL_TOL = 1e-6`` with an absolute floor of ``AGG_ABS_TOL = 1e-9`` so a
gold of 0.0 is comparable. 1e-6 is far tighter than any real error and far
looser than float summation-order noise on ~4k rounded values.

**False-answer rate** is the headline: of the questions whose answer is
genuinely absent from the data, what fraction got a non-refusal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from anyrag.core.types import Chunk, ChunkKind, Citation, Hit, QueryRoute, RowRef

#: Relative tolerance for float aggregate comparison. Counts are exact.
AGG_REL_TOL = 1e-6
#: Absolute floor, so a gold value of 0.0 is still comparable.
AGG_ABS_TOL = 1e-9

DEFAULT_KS: tuple[int, ...] = (1, 5, 10)
NDCG_K = 10

#: ("row", table, pk) or ("schema", table)
GoldKey = tuple[str, ...]


# --------------------------------------------------------------------------
# Gold targets
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldTarget:
    """What a question's answer must be grounded in.

    `row_refs` are source rows; `schema_tables` are tables whose *schema card*
    is the right evidence. Both reduce to gold keys so one set of metrics covers
    both kinds of question.
    """

    row_refs: frozenset[RowRef] = frozenset()
    schema_tables: frozenset[str] = frozenset()

    @staticmethod
    def of_rows(refs: Iterable[RowRef]) -> "GoldTarget":
        return GoldTarget(row_refs=frozenset(refs))

    @staticmethod
    def of_schema(tables: Iterable[str]) -> "GoldTarget":
        return GoldTarget(schema_tables=frozenset(tables))

    @property
    def keys(self) -> frozenset[GoldKey]:
        out: set[GoldKey] = {("row", r.table, r.pk) for r in self.row_refs}
        out |= {("schema", t) for t in self.schema_tables}
        return frozenset(out)

    @property
    def n_gold(self) -> int:
        return len(self.keys)

    @property
    def is_empty(self) -> bool:
        return not self.row_refs and not self.schema_tables

    def covered_by_chunk(self, chunk: Chunk) -> frozenset[GoldKey]:
        """The gold keys this chunk supplies evidence for."""
        keys = self.keys
        got: set[GoldKey] = set()
        for ref in getattr(chunk, "row_refs", ()) or ():
            key = ("row", ref.table, str(ref.pk))
            if key in keys:
                got.add(key)
        if self.schema_tables and getattr(chunk, "kind", None) == ChunkKind.SCHEMA_CARD:
            table = (chunk.meta or {}).get("table")
            if table in self.schema_tables:
                got.add(("schema", str(table)))
        return frozenset(got)

    def covered_by_citation(
        self,
        citation: Citation,
        chunks_by_id: Mapping[str, Chunk] | None = None,
    ) -> frozenset[GoldKey]:
        """Gold keys a citation supplies evidence for.

        `Citation` carries `row_refs` but not `kind`/`meta`, so a schema-card
        citation can only be resolved when the corresponding chunk is available
        (the harness passes the retrieval result). Without it, schema questions
        would score zero citation precision for a purely structural reason.
        """
        chunk = (chunks_by_id or {}).get(citation.chunk_id)
        if chunk is not None:
            covered = self.covered_by_chunk(chunk)
            if covered:
                return covered
        keys = self.keys
        return frozenset(
            ("row", r.table, str(r.pk))
            for r in (citation.row_refs or ())
            if ("row", r.table, str(r.pk)) in keys
        )

    def is_relevant(self, chunk: Chunk) -> bool:
        return bool(self.covered_by_chunk(chunk))


# --------------------------------------------------------------------------
# Retrieval metrics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalScores:
    recall_at: Mapping[int, float]
    hit_at: Mapping[int, float]
    mrr: float
    ndcg_at: Mapping[int, float]
    n_gold: int
    n_hits: int
    first_relevant_rank: int | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "mrr": self.mrr,
            "n_gold": self.n_gold,
            "n_hits": self.n_hits,
            "first_relevant_rank": self.first_relevant_rank,
        }
        for k, v in self.recall_at.items():
            out[f"recall@{k}"] = v
        for k, v in self.hit_at.items():
            out[f"hit@{k}"] = v
        for k, v in self.ndcg_at.items():
            out[f"ndcg@{k}"] = v
        return out


def relevance_flags(hits: Sequence[Hit], gold: GoldTarget) -> list[bool]:
    """Per-rank relevance: does this chunk cover any gold key at all."""
    return [bool(gold.covered_by_chunk(h.chunk)) for h in hits]


def novel_coverage_flags(hits: Sequence[Hit], gold: GoldTarget) -> list[bool]:
    """Per-rank relevance counting only gold keys not already credited.

    This is what nDCG is computed over; see the module docstring.
    """
    seen: set[GoldKey] = set()
    flags: list[bool] = []
    for hit in hits:
        covered = gold.covered_by_chunk(hit.chunk) - seen
        flags.append(bool(covered))
        seen |= covered
    return flags


def dcg(flags: Sequence[bool], k: int) -> float:
    return sum(1.0 / math.log2(i + 2) for i, f in enumerate(flags[:k]) if f)


def ndcg_at_k(hits: Sequence[Hit], gold: GoldTarget, k: int = NDCG_K) -> float:
    n_gold = gold.n_gold
    if n_gold == 0:
        return 0.0
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n_gold, k)))
    if ideal == 0.0:  # pragma: no cover - unreachable while n_gold > 0
        return 0.0
    return dcg(novel_coverage_flags(hits, gold), k) / ideal


def score_retrieval(
    hits: Sequence[Hit],
    gold: GoldTarget,
    ks: Sequence[int] = DEFAULT_KS,
    ndcg_ks: Sequence[int] = (NDCG_K,),
) -> RetrievalScores:
    """All retrieval metrics for one question, from one ranked hit list."""
    hits = list(hits)
    n_gold = gold.n_gold
    flags = relevance_flags(hits, gold)

    first_rank: int | None = None
    for i, f in enumerate(flags):
        if f:
            first_rank = i + 1
            break

    recall_at: dict[int, float] = {}
    hit_at: dict[int, float] = {}
    for k in ks:
        prefix = hits[:k]
        if n_gold:
            covered: set[GoldKey] = set()
            for h in prefix:
                covered |= gold.covered_by_chunk(h.chunk)
            recall_at[k] = len(covered) / n_gold
        else:
            recall_at[k] = 0.0
        hit_at[k] = 1.0 if any(flags[:k]) else 0.0

    return RetrievalScores(
        recall_at=recall_at,
        hit_at=hit_at,
        mrr=(1.0 / first_rank) if first_rank else 0.0,
        ndcg_at={k: ndcg_at_k(hits, gold, k) for k in ndcg_ks},
        n_gold=n_gold,
        n_hits=len(hits),
        first_relevant_rank=first_rank,
    )


# --------------------------------------------------------------------------
# Aggregate correctness
# --------------------------------------------------------------------------


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (
        isinstance(x, float) and (math.isnan(x) or math.isinf(x))
    )


def aggregate_correct(
    predicted: Any,
    gold: Any,
    *,
    rel_tol: float = AGG_REL_TOL,
    abs_tol: float = AGG_ABS_TOL,
) -> bool:
    """Exact for integer gold (counts), relative-tolerance for float gold."""
    if predicted is None or gold is None:
        return False
    if not _is_number(predicted) or not _is_number(gold):
        return str(predicted).strip() == str(gold).strip()
    if isinstance(gold, int):
        # A count is exactly right or it is wrong. 5.0 == 5 is accepted because
        # a source may hand back a float; 5.0000001 is not.
        return float(predicted) == float(gold)
    return abs(float(predicted) - float(gold)) <= max(
        abs_tol, rel_tol * abs(float(gold))
    )


# --------------------------------------------------------------------------
# Citation metrics
# --------------------------------------------------------------------------


def citation_precision(
    citations: Sequence[Citation],
    gold: GoldTarget,
    chunks_by_id: Mapping[str, Chunk] | None = None,
) -> float | None:
    """Fraction of cited chunks that are actually gold-relevant.

    None when nothing was cited: an answer with no citations has no citation
    precision, and scoring it 0.0 would let refusals drag the mean down and be
    mistaken for bad grounding.
    """
    if not citations:
        return None
    good = sum(
        1 for c in citations if gold.covered_by_citation(c, chunks_by_id)
    )
    return good / len(citations)


def citation_recall(
    citations: Sequence[Citation],
    gold: GoldTarget,
    chunks_by_id: Mapping[str, Chunk] | None = None,
) -> float | None:
    """Fraction of gold keys covered by the citations."""
    if gold.n_gold == 0:
        return None
    covered: set[GoldKey] = set()
    for c in citations:
        covered |= gold.covered_by_citation(c, chunks_by_id)
    return len(covered) / gold.n_gold


# --------------------------------------------------------------------------
# Per-question result record
# --------------------------------------------------------------------------


@dataclass
class QuestionResult:
    """Everything graded for one question. Serialised verbatim into results."""

    qid: str
    qtype: str
    template_set: str
    answerable: bool
    expected_route: str
    predicted_route: str | None = None
    refused: bool = False
    refusal_reason: str = ""
    retrieval: RetrievalScores | None = None
    gold_scalar: Any = None
    predicted_scalar: Any = None
    aggregate_ok: bool | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
    n_citations: int = 0
    sql: str | None = None
    sql_generated: bool = False
    #: Where the predicted scalar / SQL was recovered from. Recorded because the
    #: last-resort path is parsing the answer text, and a report should never
    #: hide that it fell back to that.
    scalar_source: str = ""
    sql_source: str = ""
    error: str | None = None
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        out = {
            "qid": self.qid,
            "qtype": self.qtype,
            "template_set": self.template_set,
            "answerable": self.answerable,
            "expected_route": self.expected_route,
            "predicted_route": self.predicted_route,
            "route_correct": self.route_correct,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "gold_scalar": self.gold_scalar,
            "predicted_scalar": self.predicted_scalar,
            "aggregate_ok": self.aggregate_ok,
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
            "n_citations": self.n_citations,
            "sql": self.sql,
            "sql_generated": self.sql_generated,
            "scalar_source": self.scalar_source,
            "sql_source": self.sql_source,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 3),
        }
        if self.retrieval is not None:
            out["retrieval"] = self.retrieval.as_dict()
        return out

    @property
    def route_correct(self) -> bool | None:
        if self.predicted_route is None:
            return None
        return self.predicted_route == self.expected_route

    @property
    def false_answer(self) -> bool:
        """An unanswerable question that got a non-refusal. The headline failure."""
        return (not self.answerable) and (not self.refused)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


AGGREGATE_ROUTES = {QueryRoute.AGGREGATE.value, QueryRoute.HYBRID.value}


def _mean(values: Sequence[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def summarize(results: Sequence[QuestionResult]) -> dict[str, Any]:
    """Roll per-question records up into the reportable numbers.

    Every rate carries the `n` it was computed over. A rate without its
    denominator is not reportable -- 0% false answers on 3 distractors is noise,
    on 64 it is a finding.
    """
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    retrieval_scored = [r for r in answerable if r.retrieval is not None and r.retrieval.n_gold]
    agg_graded = [r for r in results if r.aggregate_ok is not None]
    routed = [r for r in results if r.predicted_route is not None]
    routed_answerable = [r for r in routed if r.answerable]
    agg_route = [r for r in results if r.expected_route in AGGREGATE_ROUTES and r.answerable]

    ks = sorted({k for r in retrieval_scored for k in r.retrieval.recall_at})
    ndcg_ks = sorted({k for r in retrieval_scored for k in r.retrieval.ndcg_at})

    retrieval: dict[str, Any] = {"n": len(retrieval_scored)}
    for k in ks:
        retrieval[f"recall@{k}"] = _mean([r.retrieval.recall_at[k] for r in retrieval_scored])
        retrieval[f"hit@{k}"] = _mean([r.retrieval.hit_at[k] for r in retrieval_scored])
    for k in ndcg_ks:
        retrieval[f"ndcg@{k}"] = _mean([r.retrieval.ndcg_at[k] for r in retrieval_scored])
    retrieval["mrr"] = _mean([r.retrieval.mrr for r in retrieval_scored])

    summary: dict[str, Any] = {
        "n_questions": len(results),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "n_errors": sum(1 for r in results if r.error),
        "retrieval": retrieval,
        "aggregate": {
            "n": len(agg_graded),
            "accuracy": _mean([1.0 if r.aggregate_ok else 0.0 for r in agg_graded]),
        },
        "citations": {
            "n_with_citations": sum(1 for r in results if r.citation_precision is not None),
            "precision": _mean([r.citation_precision for r in results]),
            "recall": _mean([r.citation_recall for r in results if r.answerable]),
        },
        "false_answer": {
            "n_unanswerable": len(unanswerable),
            "n_false_answers": sum(1 for r in unanswerable if r.false_answer),
            # THE headline number.
            "rate": (
                sum(1 for r in unanswerable if r.false_answer) / len(unanswerable)
                if unanswerable
                else None
            ),
        },
        "refusal": {
            "n": len(results),
            "refusal_rate_answerable": _mean(
                [1.0 if r.refused else 0.0 for r in answerable]
            ),
            "refusal_rate_unanswerable": _mean(
                [1.0 if r.refused else 0.0 for r in unanswerable]
            ),
        },
        "router": {
            "n": len(routed),
            "accuracy": _mean([1.0 if r.route_correct else 0.0 for r in routed]),
            "n_answerable": len(routed_answerable),
            "accuracy_answerable": _mean(
                [1.0 if r.route_correct else 0.0 for r in routed_answerable]
            ),
            "confusion": _confusion(routed),
        },
        "sql": {
            # Coverage is "SQL was produced at all", deliberately distinct from
            # "SQL produced the right number" -- an uncovered aggregate should
            # refuse, and refusing is not the same failure as answering wrongly.
            "n_aggregate_questions": len(agg_route),
            "coverage": _mean([1.0 if r.sql_generated else 0.0 for r in agg_route]),
            "n_generated": sum(1 for r in agg_route if r.sql_generated),
        },
        "by_type": _by_type(results),
    }
    return summary


def _confusion(results: Sequence[QuestionResult]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in results:
        row = out.setdefault(r.expected_route, {})
        row[str(r.predicted_route)] = row.get(str(r.predicted_route), 0) + 1
    return out


def _by_type(results: Sequence[QuestionResult]) -> dict[str, dict[str, Any]]:
    types = sorted({r.qtype for r in results})
    out: dict[str, dict[str, Any]] = {}
    for t in types:
        subset = [r for r in results if r.qtype == t]
        scored = [r for r in subset if r.retrieval is not None and r.retrieval.n_gold]
        agg = [r for r in subset if r.aggregate_ok is not None]
        routed = [r for r in subset if r.predicted_route is not None]
        unanswerable = [r for r in subset if not r.answerable]
        out[t] = {
            "n": len(subset),
            "recall@10": _mean([r.retrieval.recall_at.get(10) for r in scored]) if scored else None,
            "ndcg@10": _mean([r.retrieval.ndcg_at.get(10) for r in scored]) if scored else None,
            "mrr": _mean([r.retrieval.mrr for r in scored]) if scored else None,
            "aggregate_accuracy": _mean([1.0 if r.aggregate_ok else 0.0 for r in agg]) if agg else None,
            "router_accuracy": _mean([1.0 if r.route_correct else 0.0 for r in routed]) if routed else None,
            "citation_precision": _mean([r.citation_precision for r in subset]),
            "refusal_rate": _mean([1.0 if r.refused else 0.0 for r in subset]),
            "false_answer_rate": (
                sum(1 for r in unanswerable if r.false_answer) / len(unanswerable)
                if unanswerable
                else None
            ),
        }
    return out
