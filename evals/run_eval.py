"""Run one configuration end to end and write a provenance-stamped result.

    python -m evals.run_eval --config hybrid_rerank_both
    python -m evals.run_eval --templates dev --subset distractors --limit 64

The engine is injected rather than constructed inline, so the harness can be
exercised against a fake implementation of the public contract:

    AnyRAG.ask(question, config)      -> Answer
    AnyRAG.retrieve(question, config) -> list[Hit]
    chunk.row_refs                    -> tuple[RowRef]

That is the entire surface the judge touches. It never reaches inside the
library, and it never asks the library what the right answer is.

Provenance
----------
A number without provenance is not reportable, so every results file records the
active embedder (including its `degraded` flag), the generator, the tokenizer,
the template set, `n`, the seed, and the content hash of the manifest the gold
answers were computed from. Two runs whose provenance blocks differ are not
comparable, and the file says so on its face.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from anyrag.core.config import GenerationConfig, RetrievalConfig
from anyrag.core.types import Answer, Chunk, ChunkKind, Hit, QueryRoute

from . import metrics
from .gen_db import DEFAULT_MANIFEST_PATH, DEFAULT_SEED, Manifest
from .questions import EvalQuestion, build_questions, type_counts

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"

# The three points of the chunk-kind axis.
#
# These sets are chosen to match `RetrievalConfig.name`, which recognises
# exactly `{ROW}` -> "row" and `{SCHEMA_CARD}` -> "card" and collapses every
# other set to "both". Adding SQL_RESULT to the row cell would silently name it
# "both" and overwrite the real both-cell's results file. SQL_RESULT chunks are
# minted at answer time rather than at ingest, so they ride with the `both` cell.
ROW = frozenset({ChunkKind.ROW})
CARD = frozenset({ChunkKind.SCHEMA_CARD})
BOTH = frozenset({ChunkKind.ROW, ChunkKind.SCHEMA_CARD, ChunkKind.SQL_RESULT})


# --------------------------------------------------------------------------
# Named configurations
# --------------------------------------------------------------------------


def named_configs() -> dict[str, RetrievalConfig]:
    """The 18 ablation cells plus a couple of convenience aliases."""
    out: dict[str, RetrievalConfig] = {}
    modes = {
        "dense": dict(dense=True, lexical=False),
        "bm25": dict(dense=False, lexical=True),
        "hybrid": dict(dense=True, lexical=True),
    }
    kinds = {"row": ROW, "card": CARD, "both": BOTH}
    for mode, flags in modes.items():
        for rerank in (False, True):
            for kind_name, kind_set in kinds.items():
                cfg = RetrievalConfig(
                    **flags, rerank=rerank, chunk_kinds=kind_set, k=10
                )
                out[cfg.name] = cfg
                out[f"{mode}_{'rerank' if rerank else 'norerank'}_{kind_name}"] = cfg
    out["default"] = out["hybrid_norerank_both"]
    return out


# --------------------------------------------------------------------------
# The engine seam
# --------------------------------------------------------------------------


class EvalEngine(Protocol):
    """Exactly the public surface described in the eval contract."""

    def ask(self, question: str, config: Any) -> Answer: ...

    def retrieve(self, question: str, config: Any) -> list[Hit]: ...


class EngineAdapter:
    """Tolerant caller for `ask` / `retrieve`.

    The contract fixes the *names* of the two methods and that a config is
    passed, but not whether generation settings ride along as a second
    positional, a keyword, or are held on the app. Rather than guess once and
    fail late, each shape is tried in order and the one that worked is recorded
    in the results file, so the reader can see which call the numbers came from.
    """

    def __init__(self, engine: Any, generation: GenerationConfig | None = None) -> None:
        self.engine = engine
        self.generation = generation or GenerationConfig()
        self.ask_shape: str = "untried"
        self.retrieve_shape: str = "untried"

    def retrieve(self, question: str, config: RetrievalConfig) -> list[Hit]:
        attempts = (
            ("retrieve(q, config)", lambda: self.engine.retrieve(question, config)),
            ("retrieve(q, config=config)", lambda: self.engine.retrieve(question, config=config)),
            ("retrieve(q)", lambda: self.engine.retrieve(question)),
        )
        return list(self._first_working(attempts, "retrieve_shape"))

    def ask(self, question: str, config: RetrievalConfig) -> Answer:
        gen = self.generation
        attempts = (
            ("ask(q, config)", lambda: self.engine.ask(question, config)),
            ("ask(q, config, generation)", lambda: self.engine.ask(question, config, gen)),
            ("ask(q, config, generation=gen)", lambda: self.engine.ask(question, config, generation=gen)),
            ("ask(q, retrieval=cfg, generation=gen)",
             lambda: self.engine.ask(question, retrieval=config, generation=gen)),
            ("ask(q)", lambda: self.engine.ask(question)),
        )
        return self._first_working(attempts, "ask_shape")

    def _first_working(self, attempts, slot: str):
        recorded = getattr(self, slot)
        if recorded not in ("untried",):
            for name, fn in attempts:
                if name == recorded:
                    return fn()
        errors: list[str] = []
        for name, fn in attempts:
            try:
                result = fn()
            except TypeError as exc:
                errors.append(f"{name}: {exc}")
                continue
            setattr(self, slot, name)
            return result
        raise TypeError(
            f"no supported call shape for {slot.split('_')[0]}; tried:\n  "
            + "\n  ".join(errors)
        )


def build_engine(
    source_uri: str,
    *,
    embedder: str = "auto",
    generator: str = "extractive",
    index_dir: str | None = None,
    generation: GenerationConfig | None = None,
    on_built=None,
):
    """Construct the real `AnyRAG` facade and ingest the source.

    Imported lazily: the harness must be importable, testable and unit-runnable
    while `anyrag/app.py` is still being written.
    """
    try:
        from anyrag.app import AnyRAG  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on build order
        raise ImportError(
            "anyrag.app is not importable yet, so there is nothing to grade "
            f"({exc}). The harness itself is complete and testable without it: "
            "`.venv/bin/python -m pytest evals -q` runs the whole suite against "
            "test doubles implementing ask()/retrieve()."
        ) from exc
    from anyrag.core.config import AnyRagConfig  # noqa: PLC0415

    generation = generation or GenerationConfig(generator=generator)
    cfg = AnyRagConfig(
        source_uri=source_uri,
        embedder=embedder,
        index_dir=index_dir,
        generation=generation,
    )
    errors: list[str] = []
    for build in (
        lambda: AnyRAG(cfg),
        lambda: AnyRAG(config=cfg),
        lambda: AnyRAG(source_uri=source_uri, embedder=embedder),
    ):
        try:
            app = build()
            break
        except TypeError as exc:
            errors.append(str(exc))
    else:  # pragma: no cover - reported, not swallowed
        raise TypeError("cannot construct AnyRAG; tried:\n  " + "\n  ".join(errors))

    # Hook run *before* ingest, which is the only useful moment to wrap the
    # embedder: ingest is where the chunk vectors are computed, so a wrapper
    # installed afterwards can only ever see query embeddings -- and only if the
    # sub-components read `app.embedder` rather than holding their own
    # reference from construction.
    if on_built is not None:
        on_built(app)

    ingest = getattr(app, "ingest", None)
    if callable(ingest):
        try:
            ingest()
        except TypeError:
            ingest(source_uri)
    return app


# --------------------------------------------------------------------------
# Answer introspection
#
# `Answer` carries text, citations, route and a free-form `trace`. It has no
# structured field for "the number this aggregate produced" or "the SQL that was
# run", so both are recovered here from a documented list of places, and the
# path that worked is recorded per question. See the final report: a typed hook
# on `Answer` would remove the text-parsing fallback entirely.
# --------------------------------------------------------------------------

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def _coerce_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        match = _NUMBER.search(value)
        if match:
            raw = match.group(0).replace(",", "")
            try:
                return float(raw) if "." in raw else int(raw)
            except ValueError:  # pragma: no cover - regex guarantees parseable
                return None
    return None


def extract_predicted_scalar(answer: Answer) -> tuple[Any, str]:
    """(value, where_it_came_from). Text parsing is the last resort."""
    trace = dict(answer.trace or {})

    for key in ("scalar", "value", "aggregate_value", "answer_value"):
        if key in trace:
            num = _coerce_number(trace[key])
            if num is not None:
                return num, f"trace[{key!r}]"

    for key in ("sql_result", "result", "query_result"):
        obj = trace.get(key)
        if obj is None:
            continue
        scalar = getattr(obj, "scalar", None)
        num = _coerce_number(scalar)
        if num is not None:
            return num, f"trace[{key!r}].scalar"
        rows = getattr(obj, "rows", None) or (obj if isinstance(obj, (list, tuple)) else None)
        if rows:
            first = rows[0]
            cell = first[0] if isinstance(first, (list, tuple)) else first
            num = _coerce_number(cell)
            if num is not None:
                return num, f"trace[{key!r}].rows[0][0]"

    num = _coerce_number(answer.text)
    if num is not None:
        return num, "text"
    return None, "none"


def extract_sql(answer: Answer, chunks_by_id: Mapping[str, Chunk] | None = None) -> tuple[str | None, str]:
    trace = dict(answer.trace or {})
    for key in ("sql", "generated_sql", "sql_generated", "query"):
        val = trace.get(key)
        if isinstance(val, str) and val.strip():
            return val, f"trace[{key!r}]"
    for key in ("sql_result", "result", "query_result"):
        obj = trace.get(key)
        sql = getattr(obj, "sql", None)
        if isinstance(sql, str) and sql.strip():
            return sql, f"trace[{key!r}].sql"
    for citation in answer.citations:
        chunk = (chunks_by_id or {}).get(citation.chunk_id)
        if chunk is None:
            continue
        if chunk.kind == ChunkKind.SQL_RESULT:
            sql = (chunk.meta or {}).get("sql")
            if isinstance(sql, str) and sql.strip():
                return sql, "citation.chunk.meta['sql']"
            return "<sql_result chunk without sql in meta>", "citation.kind"
    return None, "none"


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------


def grade_question(
    question: EvalQuestion,
    answer: Answer | None,
    hits: Sequence[Hit],
    *,
    error: str | None = None,
    latency_ms: float = 0.0,
    ks: Sequence[int] = metrics.DEFAULT_KS,
) -> metrics.QuestionResult:
    """One question, fully graded. No model, no system-under-test opinion."""
    chunks_by_id = {h.chunk.chunk_id: h.chunk for h in hits}

    result = metrics.QuestionResult(
        qid=question.qid,
        qtype=question.qtype,
        template_set=question.template_set,
        answerable=question.answerable,
        expected_route=question.route.value,
        gold_scalar=question.gold_scalar,
        error=error,
        latency_ms=latency_ms,
    )

    if not question.gold.is_empty:
        result.retrieval = metrics.score_retrieval(hits, question.gold, ks=ks)

    if answer is None:
        # An exception is not a refusal. Counting it as one would let a crashing
        # system post a perfect false-answer rate.
        result.refused = False
        return result

    result.refused = bool(answer.refused)
    result.refusal_reason = answer.reason or ""
    result.predicted_route = answer.route.value if answer.route is not None else None
    result.n_citations = len(answer.citations)

    sql, sql_source = extract_sql(answer, chunks_by_id)
    result.sql = sql
    result.sql_generated = bool(sql)

    if question.answerable:
        # Citation grading needs gold *evidence*, which pure aggregates do not
        # have: nothing says which chunk a COUNT ought to cite. Those questions
        # are graded on their scalar instead, and their citation precision is
        # undefined rather than zero.
        if answer.citations and not question.gold.is_empty:
            result.citation_precision = metrics.citation_precision(
                answer.citations, question.gold, chunks_by_id
            )
            result.citation_recall = metrics.citation_recall(
                answer.citations, question.gold, chunks_by_id
            )
        if question.gold_scalar is not None:
            predicted, source = extract_predicted_scalar(answer)
            result.predicted_scalar = predicted
            # A refusal on an answerable aggregate is not a correct answer, but
            # it is a different failure from a wrong number -- the refusal rate
            # and SQL coverage separate the two.
            result.aggregate_ok = (
                False
                if answer.refused
                else metrics.aggregate_correct(predicted, question.gold_scalar)
            )
            result.scalar_source = source
    # Unanswerable questions have no gold, so citation precision is undefined
    # for them; their failure mode is the false-answer rate.

    result.sql_source = sql_source
    return result


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def _as_plain(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _as_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (set, frozenset)):
        return sorted(str(x) for x in obj)
    if isinstance(obj, (list, tuple)):
        return [_as_plain(x) for x in obj]
    if isinstance(obj, Mapping):
        return {str(k): _as_plain(v) for k, v in obj.items()}
    return str(obj)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=EVALS_DIR.parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # pragma: no cover - git may be absent
        return ""


def embedder_info_of(engine: Any) -> dict[str, Any]:
    """The embedder that actually produced the vectors, however it is reached.

    Falls back to a marker rather than to silence: an unknown embedder makes a
    retrieval number uninterpretable, and that must be visible in the file.
    """
    for path in ("embedder", "_embedder", "index.embedder", "retriever.embedder"):
        obj: Any = engine
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        info = getattr(obj, "info", None)
        if info is not None:
            as_dict = getattr(info, "as_dict", None)
            return as_dict() if callable(as_dict) else _as_plain(info)
    return {
        "name": "unknown",
        "model": "unknown",
        "dim": -1,
        "degraded": True,
        "detail": "engine exposed no `.embedder.info`; provenance could not be recorded",
    }


def tokenizer_info() -> dict[str, Any]:
    try:
        from anyrag.core.tokenizer import get_tokenizer  # noqa: PLC0415

        return _as_plain(get_tokenizer().info)
    except Exception as exc:  # pragma: no cover - tokenizer has a fallback
        return {"name": "unknown", "detail": f"{type(exc).__name__}: {exc}"}


def build_provenance(
    *,
    engine: Any,
    manifest: Manifest,
    questions: Sequence[EvalQuestion],
    retrieval: RetrievalConfig,
    generation: GenerationConfig,
    templates: str,
    seed: int,
    source_uri: str,
    adapter: EngineAdapter | None = None,
) -> dict[str, Any]:
    embedder = embedder_info_of(engine)
    import anyrag  # noqa: PLC0415

    return {
        "embedder": embedder,
        "embedder_degraded": bool(embedder.get("degraded")),
        "generator": generation.generator,
        "tokenizer": tokenizer_info(),
        "template_set": templates,
        "n": len(questions),
        "n_by_type": type_counts(questions),
        "seed": seed,
        "source_uri": source_uri,
        "manifest_seed": manifest.seed,
        "manifest_content_hash": manifest.meta.get("content_hash"),
        "manifest_row_counts": dict(manifest.meta.get("row_counts", {})),
        "retrieval_config": _as_plain(retrieval),
        "retrieval_config_name": retrieval.name,
        "generation_config": _as_plain(generation),
        "engine_class": type(engine).__name__,
        "ask_shape": getattr(adapter, "ask_shape", None),
        "retrieve_shape": getattr(adapter, "retrieve_shape", None),
        "anyrag_version": getattr(anyrag, "__version__", "unknown"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "harness": "evals.run_eval",
    }


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def select_subset(questions: Sequence[EvalQuestion], subset: str | None) -> list[EvalQuestion]:
    if not subset or subset == "all":
        return list(questions)
    if subset in ("distractors", "unanswerable"):
        return [q for q in questions if not q.answerable]
    if subset == "answerable":
        return [q for q in questions if q.answerable]
    if subset == "aggregates":
        return [
            q
            for q in questions
            if q.route in (QueryRoute.AGGREGATE, QueryRoute.HYBRID) and q.answerable
        ]
    matching = [q for q in questions if q.qtype == subset]
    if not matching:
        raise ValueError(f"unknown subset {subset!r}")
    return matching


def run_questions(
    engine: Any,
    questions: Sequence[EvalQuestion],
    retrieval: RetrievalConfig,
    generation: GenerationConfig | None = None,
    *,
    adapter: EngineAdapter | None = None,
    progress: bool = False,
) -> tuple[list[metrics.QuestionResult], EngineAdapter]:
    adapter = adapter or EngineAdapter(engine, generation)
    results: list[metrics.QuestionResult] = []
    for i, question in enumerate(questions, start=1):
        started = time.perf_counter()
        hits: list[Hit] = []
        answer: Answer | None = None
        error: str | None = None
        try:
            hits = adapter.retrieve(question.text, retrieval)
        except Exception as exc:
            error = f"retrieve: {type(exc).__name__}: {exc}"
        try:
            answer = adapter.ask(question.text, retrieval)
        except Exception as exc:
            error = ((error + " | ") if error else "") + f"ask: {type(exc).__name__}: {exc}"
        latency = (time.perf_counter() - started) * 1000.0
        results.append(
            grade_question(question, answer, hits, error=error, latency_ms=latency)
        )
        if progress and i % 25 == 0:
            print(f"  {i}/{len(questions)}", file=sys.stderr)
    return results, adapter


def run_eval(
    engine: Any,
    *,
    manifest: Manifest | None = None,
    templates: str = "dev",
    subset: str | None = None,
    seed: int = DEFAULT_SEED,
    limit: int | None = None,
    retrieval: RetrievalConfig | None = None,
    generation: GenerationConfig | None = None,
    source_uri: str = "",
    questions: Sequence[EvalQuestion] | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    manifest = manifest or Manifest.load()
    retrieval = retrieval or named_configs()["default"]
    generation = generation or GenerationConfig()

    if questions is None:
        questions = build_questions(manifest, templates, seed, limit=None)
    questions = select_subset(questions, subset)
    if limit is not None and limit < len(questions):
        from .questions import stratified_sample  # noqa: PLC0415

        questions = stratified_sample(questions, limit, seed)

    started = time.perf_counter()
    results, adapter = run_questions(
        engine, questions, retrieval, generation, progress=progress
    )
    elapsed = time.perf_counter() - started

    return {
        "provenance": build_provenance(
            engine=engine,
            manifest=manifest,
            questions=questions,
            retrieval=retrieval,
            generation=generation,
            templates=templates,
            seed=seed,
            source_uri=source_uri,
            adapter=adapter,
        ),
        "subset": subset or "all",
        "wall_seconds": round(elapsed, 3),
        "summary": metrics.summarize(results),
        "results": [r.as_dict() for r in results],
    }


def write_results(payload: Mapping[str, Any], name: str, out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        fh.write("\n")
    return path


def print_summary(payload: Mapping[str, Any]) -> None:
    prov = payload["provenance"]
    summary = payload["summary"]
    degraded = " [DEGRADED EMBEDDER]" if prov["embedder_degraded"] else ""
    print(f"embedder   : {prov['embedder'].get('name')}:{prov['embedder'].get('model')}{degraded}")
    print(f"generator  : {prov['generator']}")
    print(f"tokenizer  : {prov['tokenizer'].get('name')}")
    print(f"templates  : {prov['template_set']}   n={prov['n']}   seed={prov['seed']}")
    print(f"config     : {prov['retrieval_config_name']}")
    print()
    ret = summary["retrieval"]
    print(f"retrieval (n={ret['n']}):")
    for key in ("recall@1", "recall@5", "recall@10", "hit@10", "mrr", "ndcg@10"):
        if ret.get(key) is not None:
            print(f"  {key:<12} {ret[key]:.4f}")
    fa = summary["false_answer"]
    rate = fa["rate"]
    print()
    print(
        f"FALSE-ANSWER RATE : "
        f"{'n/a' if rate is None else f'{rate:.1%}'} "
        f"({fa['n_false_answers']}/{fa['n_unanswerable']} unanswerable)"
    )
    agg = summary["aggregate"]
    if agg["accuracy"] is not None:
        print(f"aggregate accuracy: {agg['accuracy']:.1%} (n={agg['n']})")
    cit = summary["citations"]
    if cit["precision"] is not None:
        print(f"citation precision: {cit['precision']:.1%} (n={cit['n_with_citations']})")
    rou = summary["router"]
    if rou["accuracy"] is not None:
        print(f"router accuracy   : {rou['accuracy']:.1%} (n={rou['n']})")
    sql = summary["sql"]
    if sql["coverage"] is not None:
        print(
            f"SQL coverage      : {sql['coverage']:.1%} "
            f"({sql['n_generated']}/{sql['n_aggregate_questions']} aggregate questions)"
        )


def main(argv: Sequence[str] | None = None) -> int:
    configs = named_configs()
    ap = argparse.ArgumentParser(description="Run one eval configuration.")
    ap.add_argument("--config", default="default", choices=sorted(configs))
    ap.add_argument("--templates", default="dev", choices=("dev", "heldout", "all"))
    ap.add_argument("--subset", default=None)
    ap.add_argument("--source", default=None, help="source URI, e.g. sqlite:evals/data/eval.sqlite")
    ap.add_argument("--generator", default="extractive")
    ap.add_argument("--embedder", default="auto")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    ap.add_argument("--name", default=None, help="results filename stem")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = ap.parse_args(argv)

    manifest = Manifest.load(args.manifest)
    source_uri = args.source or f"sqlite:{EVALS_DIR / 'data' / 'eval.sqlite'}"
    retrieval = configs[args.config]
    generation = GenerationConfig(generator=args.generator)

    engine = build_engine(
        source_uri,
        embedder=args.embedder,
        generator=args.generator,
        generation=generation,
    )
    payload = run_eval(
        engine,
        manifest=manifest,
        templates=args.templates,
        subset=args.subset,
        seed=args.seed,
        limit=args.limit,
        retrieval=retrieval,
        generation=generation,
        source_uri=source_uri,
        progress=True,
    )
    name = args.name or f"{args.config}__{args.templates}" + (
        f"__{args.subset}" if args.subset else ""
    )
    path = write_results(payload, name, Path(args.out_dir))
    print_summary(payload)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
