"""Test doubles implementing the public eval contract.

The harness must be buildable and testable while `anyrag/app.py` is still being
written, so everything here implements exactly the surface the judge is allowed
to touch:

    ask(question, config) -> Answer
    retrieve(question, config) -> list[Hit]
    chunk.row_refs -> tuple[RowRef]

`LexicalFakeEngine` is a *plausible* system: it builds chunks from the manifest
and ranks them by overlap, and has no access to the gold answers. `OracleEngine`
and `AlwaysAnswerEngine` are degenerate on purpose -- they are the fixed points
that let the tests assert the harness reports 0% and 100% correctly. An oracle
that is handed the gold is a legitimate way to test a *scorer*; it would not be
a legitimate way to test a retriever.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

from anyrag.core.config import RetrievalConfig
from anyrag.core.types import (
    Answer,
    Chunk,
    ChunkKind,
    Citation,
    Hit,
    QueryRoute,
    RowRef,
)

from evals.gen_db import Manifest
from evals.questions import EvalQuestion

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class FakeEmbedderInfo:
    name: str = "fake"
    model: str = "token-overlap"
    dim: int = 0
    revision: str = ""
    degraded: bool = True
    detail: str = "test double; not a real embedder"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "dim": self.dim,
            "revision": self.revision,
            "degraded": self.degraded,
            "detail": self.detail,
        }


class _FakeEmbedder:
    info = FakeEmbedderInfo()


def _chunk_id(source_id: str, kind: str, table: str, pk: str) -> str:
    raw = f"{source_id}|{kind}|{table}|{pk}|0"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def build_corpus(manifest: Manifest) -> list[Chunk]:
    """One ROW chunk per row, one SCHEMA_CARD chunk per table."""
    chunks: list[Chunk] = []
    source_id = manifest.source_id
    for table in manifest.table_names:
        pk_col = manifest.pk_col(table)
        for row in manifest.rows(table):
            pk = str(row[pk_col])
            body = "; ".join(
                f"{k}={v}" for k, v in row.items() if v is not None
            )
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(source_id, "row", table, pk),
                    source_id=source_id,
                    kind=ChunkKind.ROW,
                    text=f"{table}: {body}",
                    row_refs=(RowRef(table=table, pk=pk),),
                    meta={"table": table, "pk": pk},
                )
            )
        specs = manifest.column_specs(table)
        card = ", ".join(f"{c['name']} ({c['type']})" for c in specs)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(source_id, "schema_card", table, "-"),
                source_id=source_id,
                kind=ChunkKind.SCHEMA_CARD,
                text=(
                    f"Table {table}. Columns: {card}. "
                    f"Primary key: {pk_col}."
                ),
                row_refs=(),
                meta={"table": table, "columns": [c["name"] for c in specs]},
            )
        )
    return chunks


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _trigrams(text: str) -> set[str]:
    low = text.lower()
    return {low[i : i + 3] for i in range(max(0, len(low) - 2))}


class LexicalFakeEngine:
    """A believable retriever with no knowledge of the gold answers.

    It honours every switch on `RetrievalConfig` (dense / lexical / rerank /
    chunk_kinds / k) so the 18-cell ablation produces 18 genuinely different
    result sets when driven against it.
    """

    def __init__(
        self,
        manifest: Manifest,
        *,
        refusal_threshold: float = 0.30,
        generator: str = "fake-extractive",
    ) -> None:
        self.manifest = manifest
        self.corpus = build_corpus(manifest)
        self.embedder = _FakeEmbedder()
        self.generator = generator
        self.refusal_threshold = refusal_threshold
        self._tokens = {c.chunk_id: set(_tokens(c.text)) for c in self.corpus}
        self._trigrams = {c.chunk_id: _trigrams(c.text) for c in self.corpus}

    # -- retrieval -------------------------------------------------------

    def _lexical(self, q_tokens: set[str], pool: Sequence[Chunk]) -> list[tuple[Chunk, float]]:
        scored: list[tuple[Chunk, float]] = []
        for chunk in pool:
            toks = self._tokens[chunk.chunk_id]
            if not toks:
                continue
            overlap = len(q_tokens & toks)
            if overlap:
                scored.append((chunk, overlap / math.sqrt(len(toks))))
        return scored

    def _dense(self, q_grams: set[str], pool: Sequence[Chunk]) -> list[tuple[Chunk, float]]:
        scored: list[tuple[Chunk, float]] = []
        for chunk in pool:
            grams = self._trigrams[chunk.chunk_id]
            if not grams:
                continue
            inter = len(q_grams & grams)
            if inter:
                scored.append((chunk, inter / math.sqrt(len(grams) * len(q_grams) or 1)))
        return scored

    @staticmethod
    def _rank(scored: Sequence[tuple[Chunk, float]]) -> list[Chunk]:
        return [
            c for c, _ in sorted(scored, key=lambda cs: (-cs[1], cs[0].chunk_id))
        ]

    def retrieve(self, question: str, config: RetrievalConfig | None = None) -> list[Hit]:
        config = config or RetrievalConfig()
        pool = [c for c in self.corpus if c.kind in config.chunk_kinds]
        q_tokens = set(_tokens(question))
        q_grams = _trigrams(question)

        lists: list[list[Chunk]] = []
        if config.lexical:
            lists.append(self._rank(self._lexical(q_tokens, pool))[: config.candidate_k])
        if config.dense:
            lists.append(self._rank(self._dense(q_grams, pool))[: config.candidate_k])

        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for ranked in lists:
            for i, chunk in enumerate(ranked):
                by_id[chunk.chunk_id] = chunk
                fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (
                    config.rrf_k + i + 1
                )

        order = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        if config.rerank:
            # Deterministic "reranker": exact-token containment beats fusion rank.
            def boost(chunk_id: str) -> float:
                return len(q_tokens & self._tokens[chunk_id]) / (len(q_tokens) or 1)

            order = sorted(order, key=lambda kv: (-(kv[1] + boost(kv[0])), kv[0]))

        return [
            Hit(chunk=by_id[cid], score=score, rank=i + 1, retriever="fake")
            for i, (cid, score) in enumerate(order[: config.k])
        ]

    # -- generation ------------------------------------------------------

    @staticmethod
    def _route(question: str) -> QueryRoute:
        low = question.lower()
        if any(cue in low for cue in ("what columns", "which fields", "primary key")):
            return QueryRoute.LOOKUP
        if any(cue in low for cue in ("how many", "total", "average", "highest", "lowest", "count")):
            return QueryRoute.AGGREGATE
        return QueryRoute.LOOKUP

    def ask(self, question: str, config: RetrievalConfig | None = None) -> Answer:
        config = config or RetrievalConfig()
        hits = self.retrieve(question, config)
        route = self._route(question)
        if not hits or hits[0].score < self.refusal_threshold * max(
            h.score for h in hits
        ) or not hits[0].score:
            return Answer.refusal("no supporting evidence", route=route)

        q_tokens = set(_tokens(question))
        best = hits[0]
        overlap = len(q_tokens & self._tokens[best.chunk.chunk_id]) / (len(q_tokens) or 1)
        if overlap < 0.30:
            return Answer.refusal("insufficient overlap with any chunk", route=route)

        citations = tuple(
            Citation(
                chunk_id=h.chunk.chunk_id,
                source_id=h.chunk.source_id,
                row_refs=h.chunk.row_refs,
                score=h.score,
            )
            for h in hits[:3]
        )
        return Answer(
            text=f"Based on {len(citations)} chunks: {best.chunk.text[:160]}",
            citations=citations,
            refused=False,
            route=route,
            trace={"n_hits": len(hits)},
        )


class OracleEngine:
    """Perfect on every axis. Used only to prove the harness reports 1.0/0%.

    It is handed the question set, which a real system never is. That is the
    point: it pins the *upper* fixed point of every metric.
    """

    def __init__(self, manifest: Manifest, questions: Sequence[EvalQuestion]) -> None:
        self.manifest = manifest
        self.corpus = build_corpus(manifest)
        self.embedder = _FakeEmbedder()
        self._by_text = {q.text: q for q in questions}
        self._by_ref: dict[tuple[str, str], Chunk] = {}
        self._cards: dict[str, Chunk] = {}
        for chunk in self.corpus:
            if chunk.kind == ChunkKind.SCHEMA_CARD:
                self._cards[chunk.meta["table"]] = chunk
            for ref in chunk.row_refs:
                self._by_ref[(ref.table, ref.pk)] = chunk

    def _gold_chunks(self, question: EvalQuestion) -> list[Chunk]:
        out = [
            self._by_ref[(r.table, r.pk)]
            for r in sorted(question.gold.row_refs)
            if (r.table, r.pk) in self._by_ref
        ]
        out += [self._cards[t] for t in sorted(question.gold.schema_tables) if t in self._cards]
        return out

    def retrieve(self, question: str, config: RetrievalConfig | None = None) -> list[Hit]:
        config = config or RetrievalConfig()
        q = self._by_text.get(question)
        if q is None:
            return []
        chunks = [c for c in self._gold_chunks(q) if c.kind in config.chunk_kinds]
        return [
            Hit(chunk=c, score=1.0 - i * 1e-3, rank=i + 1, retriever="oracle")
            for i, c in enumerate(chunks[: config.k])
        ]

    def ask(self, question: str, config: RetrievalConfig | None = None) -> Answer:
        config = config or RetrievalConfig()
        q = self._by_text.get(question)
        if q is None or not q.answerable:
            return Answer.refusal("not answerable from the available data")
        hits = self.retrieve(question, config)
        citations = tuple(
            Citation(
                chunk_id=h.chunk.chunk_id,
                source_id=h.chunk.source_id,
                row_refs=h.chunk.row_refs,
                score=h.score,
            )
            for h in hits
        )
        if not citations:
            # Aggregates have a scalar gold but no row gold: cite the schema card
            # of the table the question is about, which is what a SQL answer
            # would carry alongside its SQL_RESULT chunk.
            table = q.meta.get("table") or (q.meta.get("tables") or ["orders"])[0]
            card = self._cards.get(table)
            if card is not None:
                citations = (
                    Citation(
                        chunk_id=card.chunk_id,
                        source_id=card.source_id,
                        row_refs=(),
                        score=1.0,
                    ),
                )
        trace: dict[str, Any] = {}
        if q.gold_scalar is not None:
            trace["scalar"] = q.gold_scalar
        if q.route in (QueryRoute.AGGREGATE, QueryRoute.HYBRID):
            trace["sql"] = q.meta.get("sql", "SELECT 1")
        if not citations:
            return Answer.refusal("no citable evidence", route=q.route)
        return Answer(
            text=f"{q.gold_scalar}" if q.gold_scalar is not None else "see citations",
            citations=citations,
            refused=False,
            route=q.route,
            trace=trace,
        )


class AlwaysAnswerEngine:
    """Never refuses. Pins the false-answer rate at 100%."""

    def __init__(self, manifest: Manifest) -> None:
        self.corpus = build_corpus(manifest)
        self.embedder = _FakeEmbedder()

    def retrieve(self, question: str, config: RetrievalConfig | None = None) -> list[Hit]:
        config = config or RetrievalConfig()
        pool = [c for c in self.corpus if c.kind in config.chunk_kinds][: config.k]
        return [
            Hit(chunk=c, score=0.5, rank=i + 1, retriever="always")
            for i, c in enumerate(pool)
        ]

    def ask(self, question: str, config: RetrievalConfig | None = None) -> Answer:
        hits = self.retrieve(question, config)
        citations = tuple(
            Citation(
                chunk_id=h.chunk.chunk_id,
                source_id=h.chunk.source_id,
                row_refs=h.chunk.row_refs,
                score=h.score,
            )
            for h in hits[:2]
        )
        return Answer(
            text="Yes, certainly: 42.",
            citations=citations,
            refused=False,
            route=QueryRoute.LOOKUP,
            trace={"scalar": 42},
        )


class CrashingEngine:
    """Raises on every call. An exception must not be scored as a refusal."""

    def __init__(self) -> None:
        self.embedder = _FakeEmbedder()

    def retrieve(self, question: str, config: Any = None) -> list[Hit]:
        raise RuntimeError("index unavailable")

    def ask(self, question: str, config: Any = None) -> Answer:
        raise RuntimeError("generator unavailable")
