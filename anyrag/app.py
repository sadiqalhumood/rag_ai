"""The `AnyRAG` facade: ingest a source, then ask questions of it.

This is the entire public surface the evaluation harness grades through:

    AnyRAG.ingest()                      -> IngestReport
    AnyRAG.retrieve(question, config)    -> list[Hit]
    AnyRAG.ask(question, config)         -> Answer

Routing lives here because it is the thing that makes database RAG different
from document RAG. A LOOKUP question is answered from retrieved rows. An
AGGREGATE question is answered by generating SQL, linting it read-only,
executing it, and citing the result — because the answer is a number that
exists in no single row, and retrieving five similar rows to "count" would
produce a confident wrong answer. HYBRID does both.

Orchestrator-owned.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .core.config import AnyRagConfig, GenerationConfig, RetrievalConfig
from .core.errors import ConfigError, SqlGenerationError, UnsafeQueryError
from .core.registry import open_source
from .core.types import (
    TRACE_REFUSAL,
    TRACE_RETRIEVAL,
    TRACE_ROUTE,
    TRACE_ROUTE_REASON,
    TRACE_SCALAR,
    TRACE_SQL,
    TRACE_SQL_ERROR,
    Answer,
    Chunk,
    ChunkKind,
    Citation,
    Hit,
    QueryResult,
    QueryRoute,
)
from .embed import get_embedder
from .generate import get_generator
from .index import BM25Index, MemoryVectorIndex, PersistentVectorIndex
from .ingest import ingest as ingest_chunks
from .retrieval import HybridRetriever, get_expander, get_reranker
from .route.router import Router
from .route.schema_lexicon import SchemaLexicon
from .route.sqlgen import HeuristicSqlGenerator


_DATE_PAIR = __import__("re").compile(
    r"(\d{4}-\d{2}-\d{2})\D{1,20}?(\d{4}-\d{2}-\d{2})"
)


def _explicit_date_window(question: str) -> tuple[str, str] | None:
    """Extract an explicit ISO date range from a question, if it has one."""
    match = _DATE_PAIR.search(question)
    if not match:
        return None
    lo, hi = match.group(1), match.group(2)
    return (lo, hi) if lo <= hi else (hi, lo)


@dataclass
class IngestReport:
    """What an ingest run actually did.

    `embedded` vs `skipped` is the observable proof that re-ingestion is
    incremental: a second run over unchanged data must embed nothing.
    """

    chunks: int = 0
    embedded: int = 0
    skipped: int = 0
    tables: int = 0
    kinds: dict[str, int] = field(default_factory=dict)

    @property
    def index_size(self) -> int:
        return self.chunks


class AnyRAG:
    """Ties a source, an index, a retriever, and a generator together."""

    def __init__(
        self,
        source,  # noqa: ANN001 - DataSource protocol
        *,
        config: AnyRagConfig | None = None,
        embedder=None,  # noqa: ANN001
        vector_index=None,  # noqa: ANN001
        lexical_index=None,  # noqa: ANN001
        generator=None,  # noqa: ANN001
        reranker=None,  # noqa: ANN001
        expander=None,  # noqa: ANN001
        sql_generator=None,  # noqa: ANN001
        build_lexicon: bool = True,
    ) -> None:
        # `AnyRAG(cfg)` is a natural call shape and should not be a crash. When
        # the first positional is a config carrying a source URI, open it.
        if isinstance(source, AnyRagConfig):
            if config is not None and config is not source:
                raise ConfigError(
                    "received an AnyRagConfig positionally and another via "
                    "config=; pass exactly one"
                )
            config = source
            if not config.source_uri:
                raise ConfigError(
                    "AnyRAG(config) needs config.source_uri set, or pass a "
                    "DataSource as the first argument"
                )
            source = open_source(config.source_uri)

        self.source = source
        self.config = config or AnyRagConfig()
        self.embedder = embedder if embedder is not None else get_embedder(self.config.embedder)

        if vector_index is None:
            vector_index = (
                PersistentVectorIndex(path=self.config.index_dir)
                if self.config.index_dir
                else MemoryVectorIndex(dim=self.embedder.dim)
            )
        self.vector_index = vector_index
        self.lexical_index = lexical_index if lexical_index is not None else BM25Index()

        self.retriever = HybridRetriever(
            vector_index=self.vector_index,
            lexical_index=self.lexical_index,
            embedder=self.embedder,
            reranker=reranker if reranker is not None else get_reranker(),
            expander=expander if expander is not None else get_expander(),
        )
        self.generator = (
            generator
            if generator is not None
            else get_generator(self.config.generation.generator)
        )

        # The lexicon is what lets the router and SQL generator talk about this
        # particular database. Building it touches the source, so it is
        # optional for callers that only want retrieval.
        self.lexicon = SchemaLexicon.from_source(source) if build_lexicon else None
        self.router = Router(source=source, lexicon=self.lexicon)
        self.sql_generator = (
            sql_generator
            if sql_generator is not None
            else HeuristicSqlGenerator(source=source, lexicon=self.lexicon)
        )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_uri(cls, uri: str, **kwargs: Any) -> "AnyRAG":
        """Open a source by URI (`sqlite:...`, `postgres:...`, `files:...`)."""
        return cls(open_source(uri), **kwargs)

    # -- ingestion ----------------------------------------------------------

    def ingest(
        self,
        *,
        row_chunks: bool | None = None,
        schema_cards: bool | None = None,
        batch_size: int = 256,
        **kwargs: Any,
    ) -> IngestReport:
        """Chunk the source and load both indexes.

        Re-ingesting unchanged data must not duplicate: chunk IDs are stable by
        construction, and a chunk whose `content_hash` matches what is already
        indexed is neither re-embedded nor re-upserted.
        """
        report = IngestReport()
        pending: list[Chunk] = []

        def flush() -> None:
            if not pending:
                return
            vectors = self.embedder.embed([c.text for c in pending])
            self.vector_index.upsert(pending, vectors)
            self.lexical_index.upsert(pending)
            report.embedded += len(pending)
            pending.clear()

        stream = ingest_chunks(
            self.source,
            row_chunks=self.config.row_chunks if row_chunks is None else row_chunks,
            schema_cards=(
                self.config.schema_cards if schema_cards is None else schema_cards
            ),
            **kwargs,
        )
        tables: set[str] = set()
        for chunk in stream:
            report.chunks += 1
            report.kinds[chunk.kind.value] = report.kinds.get(chunk.kind.value, 0) + 1
            if chunk.table:
                tables.add(chunk.table)

            existing = self.vector_index.get(chunk.chunk_id)
            if existing is not None and existing.content_hash == chunk.content_hash:
                report.skipped += 1
                continue

            pending.append(chunk)
            if len(pending) >= batch_size:
                flush()
        flush()
        report.tables = len(tables)
        return report

    # -- retrieval ----------------------------------------------------------

    @staticmethod
    def _split_config(
        config: RetrievalConfig | AnyRagConfig | None,
        default: AnyRagConfig,
        generation: GenerationConfig | None,
    ) -> tuple[RetrievalConfig, GenerationConfig]:
        """Accept either config object explicitly, rather than guessing.

        The eval harness had to probe five call shapes to discover this; being
        strict about the two legitimate ones is kinder than being permissive
        and silently misbehaving when handed the wrong type.
        """
        if config is None:
            retrieval = default.retrieval
            gen = generation or default.generation
        elif isinstance(config, AnyRagConfig):
            retrieval = config.retrieval
            gen = generation or config.generation
        elif isinstance(config, RetrievalConfig):
            retrieval = config
            gen = generation or default.generation
        else:
            raise ConfigError(
                "config must be a RetrievalConfig or an AnyRagConfig, got "
                f"{type(config).__name__}"
            )
        return retrieval, gen

    def retrieve(
        self, question: str, config: RetrievalConfig | AnyRagConfig | None = None
    ) -> list[Hit]:
        retrieval, _ = self._split_config(config, self.config, None)
        return self.retriever.retrieve(question, retrieval)

    # -- SQL path -----------------------------------------------------------

    def _sql_result_chunk(self, sql: str, result: QueryResult) -> Chunk:
        """Wrap a query result as a citable chunk.

        The result of a linted, executed query is evidence in exactly the same
        sense a retrieved row is, so it becomes a `Chunk` and gets cited rather
        than being spliced into prose uncited.
        """
        digest = hashlib.sha256(sql.encode("utf-8")).hexdigest()[:32]
        rows = result.as_dicts()
        if len(rows) == 1 and len(result.columns) == 1:
            body = f"{result.columns[0]} = {rows[0][result.columns[0]]}"
        else:
            body = "; ".join(
                ", ".join(f"{k} is {v}" for k, v in row.items()) for row in rows[:25]
            )
        text = f"Query result for: {sql}\n{body}"
        return Chunk(
            chunk_id=f"sql:{digest}",
            source_id=getattr(self.source, "source_id", "unknown"),
            kind=ChunkKind.SQL_RESULT,
            text=text,
            row_refs=(),
            meta={
                "sql": sql,
                "columns": list(result.columns),
                "row_count": len(result.rows),
                "truncated": result.truncated,
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            },
        )

    def _run_sql(self, question: str) -> tuple[Chunk | None, str, str, Any]:
        """Generate, lint, and execute SQL. Never raises for expected failures.

        Returns (chunk, sql, error, scalar). An out-of-coverage question yields
        an error rather than a guess, and the caller turns that into a refusal.
        """
        try:
            sql = self.sql_generator.generate(question, self.source)
        except SqlGenerationError as exc:
            return None, "", f"sql_generation: {exc}", None
        except UnsafeQueryError as exc:  # pragma: no cover - defence in depth
            return None, "", f"unsafe_sql: {exc}", None

        try:
            result = self.source.execute_readonly(sql)
        except UnsafeQueryError as exc:
            return None, sql, f"unsafe_sql: {exc}", None
        except Exception as exc:
            return None, sql, f"sql_execution: {type(exc).__name__}: {exc}", None

        return self._sql_result_chunk(sql, result), sql, "", result.scalar

    # -- the main entry point ----------------------------------------------

    def ask(
        self,
        question: str,
        config: RetrievalConfig | AnyRagConfig | None = None,
        *,
        generation: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Answer:
        retrieval, gen_config = self._split_config(config, self.config, generation)
        decision = self.router.classify(question)
        trace: dict[str, Any] = {
            TRACE_ROUTE: decision.route.value,
            TRACE_ROUTE_REASON: decision.reason,
        }

        # A question asking for an attribute the schema does not have is
        # unanswerable regardless of route. Retrieval will happily return the
        # named entity's row and a generator will compose something plausible
        # from it -- an answer to a question nobody asked. Schema questions are
        # exempt: their attribute word ("columns") is deliberately meta.
        # Exempt only true schema questions, which the router marks by
        # preferring schema cards *exclusively*. An ordinary LOOKUP also lists
        # SCHEMA_CARD among its preferred kinds, so testing for membership
        # rather than equality disabled this guard entirely.
        is_schema_question = decision.preferred_kinds == frozenset(
            {ChunkKind.SCHEMA_CARD}
        )
        if self.lexicon is not None and not is_schema_question:
            missing = self.lexicon.unresolved_attributes(question)
            if missing:
                trace["unresolved_attributes"] = missing
                return Answer.refusal(
                    "the data has no attribute matching "
                    + ", ".join(repr(m) for m in missing),
                    route=decision.route,
                    trace=trace,
                )
            unknown = self.lexicon.unknown_entities(question)
            if unknown:
                trace["unknown_entities"] = unknown
                return Answer.refusal(
                    "the data has no entity matching "
                    + ", ".join(repr(u) for u in unknown),
                    route=decision.route,
                    trace=trace,
                )

        # A date window entirely outside the data's observed range cannot be
        # answered from the data, whatever the route.
        if self.lexicon is not None:
            window = _explicit_date_window(question)
            if window and self.lexicon.date_range_outside_data(*window):
                trace["date_window_outside_data"] = list(window)
                return Answer.refusal(
                    f"the data covers no dates between {window[0]} and {window[1]}",
                    route=decision.route,
                    trace=trace,
                )

        hits = self.retriever.retrieve(question, retrieval)
        trace[TRACE_RETRIEVAL] = getattr(self.retriever, "last_trace", {}) or {}

        if decision.route is QueryRoute.LOOKUP:
            return self._answer_from_hits(
                question, hits, gen_config, decision.route, trace, **kwargs
            )

        sql_chunk, sql, sql_error, scalar = self._run_sql(question)
        if sql:
            trace[TRACE_SQL] = sql
        if sql_error:
            trace[TRACE_SQL_ERROR] = sql_error

        if sql_chunk is None:
            # An aggregate we could not compute must refuse. Falling back to
            # retrieval here is what produces confident wrong numbers.
            if decision.route is QueryRoute.AGGREGATE:
                return Answer.refusal(
                    f"could not compute an aggregate answer ({sql_error})",
                    route=decision.route,
                    trace=trace,
                )
            return self._answer_from_hits(
                question, hits, gen_config, decision.route, trace, **kwargs
            )

        # A NULL scalar means the aggregate is undefined over the matched rows
        # (AVG/MAX/MIN of nothing). There is no value to report, so reporting
        # one would be inventing it. COUNT is exempt: zero is a real answer.
        if scalar is None and len(sql_chunk.meta.get("columns", ())) == 1:
            if sql_chunk.meta.get("row_count", 0) == 0 or "COUNT(" not in sql.upper():
                if sql_chunk.meta.get("row_count", 0) <= 1:
                    return Answer.refusal(
                        "the query matched no rows, so the aggregate is undefined",
                        route=decision.route,
                        trace=trace,
                    )

        trace[TRACE_SCALAR] = scalar

        # A successfully executed, linted query IS the evidence: the answer is
        # grounded by construction, so it does not go through the lexical
        # support gates that exist to catch ungrounded retrieval.
        citations = [
            Citation(
                chunk_id=sql_chunk.chunk_id,
                source_id=sql_chunk.source_id,
                row_refs=(),
                score=1.0,
                quoted_span=sql_chunk.text[:280],
                kind=ChunkKind.SQL_RESULT,
            )
        ]
        supporting: list[Hit] = []
        if decision.route is QueryRoute.HYBRID:
            supporting = [h for h in hits if h.chunk.kind is ChunkKind.ROW][:3]
            citations.extend(
                Citation(
                    chunk_id=h.chunk.chunk_id,
                    source_id=h.chunk.source_id,
                    row_refs=h.chunk.row_refs,
                    score=float(h.score),
                    quoted_span=h.chunk.text[:280],
                    kind=h.chunk.kind,
                )
                for h in supporting
            )

        text = self._render_sql_answer(question, sql_chunk, scalar)
        return Answer(
            text=text,
            citations=tuple(citations),
            refused=False,
            route=decision.route,
            trace=trace,
            value=scalar,
        )

    @staticmethod
    def _render_sql_answer(question: str, chunk: Chunk, scalar: Any) -> str:
        if scalar is not None:
            return f"{scalar} [1]"
        body = chunk.text.split("\n", 1)[-1]
        return f"{body} [1]"

    def _answer_from_hits(
        self,
        question: str,
        hits: Sequence[Hit],
        gen_config: GenerationConfig,
        route: QueryRoute,
        trace: dict[str, Any],
        **kwargs: Any,
    ) -> Answer:
        answer = self.generator.generate(
            question, hits, gen_config, route=route, **kwargs
        )
        merged = dict(answer.trace or {})
        merged.update(trace)
        if answer.reason:
            merged.setdefault(TRACE_REFUSAL, answer.reason)
        return Answer(
            text=answer.text,
            citations=answer.citations,
            refused=answer.refused,
            reason=answer.reason,
            route=route,
            trace=merged,
            value=answer.value,
        )

    def close(self) -> None:
        close = getattr(self.source, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "AnyRAG":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
