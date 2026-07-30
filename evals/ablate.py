"""The full 3 x 2 x 3 ablation grid -> ABLATIONS.md.

    {dense, bm25, hybrid} x {rerank on, off} x {row-chunks, schema-cards, both}

Three rules this file exists to enforce:

1. **The whole grid is reported, never just the winner.** Every one of the 18
   cells appears in the table, including the ones that lose badly -- a sweep
   that only publishes its best cell is a press release, not an ablation.
2. **Embeddings are computed once.** The engine is built and ingested a single
   time and only `RetrievalConfig` varies across cells, and query/chunk vectors
   are additionally cached to `evals/.cache/` so a second invocation is nearly
   free. On four cores this is the difference between affordable and not.
3. **A partial grid is never presented as a complete one.** Past the timebox the
   run drops to a fixed stratified sample -- one seeded sample reused
   *identically* across all 18 cells, because an inconsistent sample makes cells
   incomparable, which is worse than a smaller n -- restarts, and records the
   reduction in the file header.

If the active embedder is degraded, every affected metric cell is marked in the
table itself (``0.412 ⚠``), not only in the prose around it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from anyrag.core.config import GenerationConfig, RetrievalConfig
from anyrag.core.types import ChunkKind

from . import metrics
from .gen_db import DEFAULT_MANIFEST_PATH, DEFAULT_SEED, Manifest
from .questions import EvalQuestion, build_questions, stratified_sample, type_counts
from .run_eval import (
    BOTH,
    CARD,
    EVALS_DIR,
    ROW,
    build_engine,
    build_provenance,
    run_questions,
    write_results,
)

CACHE_DIR = EVALS_DIR / ".cache"
ABLATIONS_MD = EVALS_DIR.parent / "ABLATIONS.md"
RESULTS_SUBDIR = EVALS_DIR / "results" / "ablation"

DEFAULT_TIMEBOX_MINUTES = 90.0

MODES: tuple[tuple[str, dict[str, bool]], ...] = (
    ("dense", {"dense": True, "lexical": False}),
    ("bm25", {"dense": False, "lexical": True}),
    ("hybrid", {"dense": True, "lexical": True}),
)

KINDS: tuple[tuple[str, frozenset[ChunkKind]], ...] = (
    ("row-chunks", ROW),
    ("schema-cards", CARD),
    ("both", BOTH),
)


def grid_configs() -> list[tuple[str, str, str, RetrievalConfig]]:
    """All 18 cells, in a stable order."""
    out: list[tuple[str, str, str, RetrievalConfig]] = []
    for mode_name, flags in MODES:
        for rerank in (False, True):
            for kind_name, kind_set in KINDS:
                out.append(
                    (
                        mode_name,
                        "on" if rerank else "off",
                        kind_name,
                        RetrievalConfig(
                            **flags, rerank=rerank, chunk_kinds=kind_set, k=10
                        ),
                    )
                )
    return out


# --------------------------------------------------------------------------
# Embedding cache
# --------------------------------------------------------------------------


class CachedEmbedder:
    """Disk-memoised wrapper around any object satisfying the Embedder protocol.

    It wraps the public interface -- `embed`, `embed_query`, `info`, `dim` -- and
    changes no library code. `info` is delegated untouched so provenance still
    reports the embedder that actually produced the vectors, and the cache
    namespace is keyed on that label so a degraded fallback can never serve
    vectors to a real-model run or vice versa.
    """

    def __init__(self, inner: Any, cache_dir: Path = CACHE_DIR) -> None:
        self.inner = inner
        self.info = getattr(inner, "info", None)
        self.dim = getattr(inner, "dim", 0)
        label = getattr(self.info, "label", None) or str(getattr(self.info, "name", "unknown"))
        self.namespace = hashlib.sha256(str(label).encode("utf-8")).hexdigest()[:16]
        self.path = Path(cache_dir) / f"emb-{self.namespace}.npz"
        self.hits = 0
        self.misses = 0
        self._cache: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with np.load(self.path, allow_pickle=False) as data:
                keys = data["keys"]
                vecs = data["vectors"]
            self._cache = {str(k): vecs[i] for i, k in enumerate(keys)}
        except Exception:  # pragma: no cover - a corrupt cache is not fatal
            self._cache = {}

    def save(self) -> None:
        if not self._cache:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._cache)
        vecs = np.vstack([self._cache[k] for k in keys]).astype(np.float32)
        np.savez_compressed(self.path, keys=np.array(keys), vectors=vecs)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        keys = [self._key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            fresh = np.asarray(self.inner.embed([texts[i] for i in missing]), dtype=np.float32)
            for slot, i in enumerate(missing):
                self._cache[keys[i]] = fresh[slot]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if not texts:
            return np.zeros((0, self.dim or 1), dtype=np.float32)
        return np.vstack([self._cache[k] for k in keys]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._cache)}


def install_cache(engine: Any, cache_dir: Path = CACHE_DIR) -> CachedEmbedder | None:
    """Wrap the engine's embedder in the disk cache, if it exposes one."""
    inner = getattr(engine, "embedder", None)
    if inner is None or isinstance(inner, CachedEmbedder):
        return inner if isinstance(inner, CachedEmbedder) else None
    cached = CachedEmbedder(inner, cache_dir)
    try:
        engine.embedder = cached
    except Exception:  # pragma: no cover - read-only attribute
        return None
    return cached


# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------


@dataclass
class Cell:
    mode: str
    rerank: str
    kinds: str
    config_name: str
    summary: dict[str, Any]
    seconds: float
    n: int
    results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.mode}__{'rerank' if self.rerank == 'on' else 'norerank'}__{self.kinds}"


@dataclass
class GridRun:
    cells: list[Cell]
    questions: Sequence[EvalQuestion]
    reduced: bool
    reduction: dict[str, Any] | None
    seconds: float
    cache_stats: dict[str, int] | None


def run_grid(
    engine: Any,
    questions: Sequence[EvalQuestion],
    *,
    generation: GenerationConfig | None = None,
    timebox_seconds: float = DEFAULT_TIMEBOX_MINUTES * 60,
    seed: int = DEFAULT_SEED,
    sample_size: int | None = None,
    progress: bool = True,
    cache: CachedEmbedder | None = None,
) -> GridRun:
    """Run all 18 cells. Reduces the question set rather than the grid."""
    generation = generation or GenerationConfig()
    configs = grid_configs()
    started = time.perf_counter()
    reduction: dict[str, Any] | None = None
    working = list(questions)

    while True:
        cells: list[Cell] = []
        aborted = False
        for i, (mode, rerank, kinds, config) in enumerate(configs, start=1):
            cell_start = time.perf_counter()
            results, _ = run_questions(engine, working, config, generation)
            seconds = time.perf_counter() - cell_start
            cells.append(
                Cell(
                    mode=mode,
                    rerank=rerank,
                    kinds=kinds,
                    config_name=config.name,
                    summary=metrics.summarize(results),
                    seconds=seconds,
                    n=len(working),
                    results=[r.as_dict() for r in results],
                )
            )
            if progress:
                print(
                    f"  [{i:>2}/18] {mode:<6} rerank={rerank:<3} {kinds:<12} "
                    f"{seconds:6.1f}s  n={len(working)}",
                    file=sys.stderr,
                )

            # Project the full grid from what has actually been measured. The
            # decision is taken early and the grid restarts, so every cell in the
            # published table is computed on the same question set.
            elapsed = time.perf_counter() - started
            projected = elapsed / i * len(configs)
            if reduction is None and projected > timebox_seconds and i < len(configs):
                target = sample_size or max(
                    40, int(len(working) * timebox_seconds / projected * 0.8)
                )
                target = min(target, len(working) - 1) if target >= len(working) else target
                sampled = stratified_sample(working, target, seed)
                reduction = {
                    "trigger": (
                        f"projected {projected / 60:.1f} min for the full grid exceeds "
                        f"the {timebox_seconds / 60:.0f} min timebox after {i} cells"
                    ),
                    "n_before": len(working),
                    "n_after": len(sampled),
                    "per_type": type_counts(sampled),
                    "seed": seed,
                    "note": (
                        "one seeded stratified sample, reused identically across "
                        "all 18 cells; the grid was restarted so no cell is "
                        "computed on a different question set"
                    ),
                }
                if progress:
                    print(f"  ! timebox: {reduction['trigger']}", file=sys.stderr)
                    print(
                        f"  ! restarting the full grid on n={len(sampled)}",
                        file=sys.stderr,
                    )
                working = sampled
                aborted = True
                break
        if not aborted:
            break

    if cache is not None:
        cache.save()

    return GridRun(
        cells=cells,
        questions=working,
        reduced=reduction is not None,
        reduction=reduction,
        seconds=time.perf_counter() - started,
        cache_stats=cache.stats if cache else None,
    )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _fmt(value: Any, degraded: bool, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    text = f"{value:.{digits}f}"
    return f"{text} ⚠" if degraded else text


def _pct(value: Any, degraded: bool = False) -> str:
    if value is None:
        return "n/a"
    text = f"{value * 100:.1f}%"
    return f"{text} ⚠" if degraded else text


def render_markdown(run: GridRun, provenance: Mapping[str, Any]) -> str:
    emb = provenance["embedder"]
    degraded = bool(provenance["embedder_degraded"])
    n = run.cells[0].n if run.cells else 0

    lines: list[str] = []
    lines.append("# ABLATIONS")
    lines.append("")
    lines.append(
        f"Embedder: **{emb.get('name')}:{emb.get('model')}**"
        + (f" @ `{emb.get('revision', '')[:12]}`" if emb.get("revision") else "")
        + ("  — **DEGRADED**" if degraded else "")
    )
    lines.append("")
    lines.append(f"- generator: `{provenance['generator']}`")
    lines.append(f"- tokenizer: `{provenance['tokenizer'].get('name')}`")
    lines.append(f"- template set: `{provenance['template_set']}`")
    lines.append(f"- n (questions per cell): **{n}**")
    lines.append(f"- seed: `{provenance['seed']}`")
    lines.append(f"- manifest content hash: `{provenance['manifest_content_hash']}`")
    lines.append(f"- git commit: `{provenance['git_commit'] or 'unknown'}`")
    lines.append(f"- grid wall time: {run.seconds / 60:.1f} min over {len(run.cells)} cells")
    if run.cache_stats:
        lines.append(
            f"- embedding cache: {run.cache_stats['hits']} hits / "
            f"{run.cache_stats['misses']} misses, {run.cache_stats['entries']} entries"
        )
    lines.append("")

    if degraded:
        lines.append(
            "> **Every metric below was computed with a degraded (fallback) embedder.** "
            "Cells are marked `⚠` individually so no single row can be quoted without "
            "the caveat. These numbers are runnable, not comparable to a real-model run."
        )
        lines.append("")

    if run.reduced and run.reduction:
        r = run.reduction
        lines.append("## Timebox reduction (amendment 4)")
        lines.append("")
        lines.append(f"- trigger: {r['trigger']}")
        lines.append(f"- question set reduced: {r['n_before']} → **{r['n_after']}**")
        lines.append(f"- sampling: {r['note']} (seed `{r['seed']}`)")
        lines.append("- per-type counts in the sample:")
        lines.append("")
        for qtype, count in r["per_type"].items():
            lines.append(f"  - `{qtype}`: {count}")
        lines.append("")
        lines.append(
            "**The grid is complete on the reduced set.** Grid coverage was preserved "
            "in preference to question count; all 18 cells use the identical sample."
        )
        lines.append("")
    else:
        lines.append(
            f"The complete 18-cell grid ran on the full question set (n={n}); "
            "no timebox reduction was applied."
        )
        lines.append("")

    lines.append("## The full grid (all 18 cells)")
    lines.append("")
    header = (
        "| retrieval | rerank | chunks | recall@1 | recall@5 | recall@10 | MRR | "
        "nDCG@10 | cite prec | **false-answer** | router acc | SQL cov | n | degraded |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 14)
    for cell in run.cells:
        s = cell.summary
        ret = s["retrieval"]
        lines.append(
            "| {mode} | {rerank} | {kinds} | {r1} | {r5} | {r10} | {mrr} | {ndcg} "
            "| {cite} | {fa} | {router} | {sql} | {n} | {deg} |".format(
                mode=cell.mode,
                rerank=cell.rerank,
                kinds=cell.kinds,
                r1=_fmt(ret.get("recall@1"), degraded),
                r5=_fmt(ret.get("recall@5"), degraded),
                r10=_fmt(ret.get("recall@10"), degraded),
                mrr=_fmt(ret.get("mrr"), degraded),
                ndcg=_fmt(ret.get("ndcg@10"), degraded),
                cite=_pct(s["citations"]["precision"], degraded),
                fa=_pct(s["false_answer"]["rate"], degraded),
                router=_pct(s["router"]["accuracy"]),
                sql=_pct(s["sql"]["coverage"]),
                n=cell.n,
                deg="⚠ yes" if degraded else "no",
            )
        )
    lines.append("")
    lines.append(
        f"Retrieval columns are averaged over the questions with row/schema gold "
        f"(n={run.cells[0].summary['retrieval']['n'] if run.cells else 0} of {n}); "
        "count and pure-aggregate questions have a scalar gold rather than a "
        "retrievable one and are excluded from them by construction, not by "
        "selection. Router accuracy and SQL coverage do not depend on the "
        "retrieval axis and are shown per cell for completeness."
    )
    lines.append("")

    lines.append("## recall@10 by question type")
    lines.append("")
    qtypes = sorted(
        {t for cell in run.cells for t in cell.summary["by_type"]}
    )
    lines.append("| cell | " + " | ".join(f"`{t}`" for t in qtypes) + " |")
    lines.append("|" + "---|" * (len(qtypes) + 1))
    for cell in run.cells:
        row = [f"{cell.mode}/{cell.rerank}/{cell.kinds}"]
        for qtype in qtypes:
            entry = cell.summary["by_type"].get(qtype, {})
            row.append(_fmt(entry.get("recall@10"), degraded))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(
        "The `schema_question` column is the reason the chunk-kind axis exists: "
        "a `row-chunks` cell cannot retrieve a schema card at all, so it scores "
        "zero there however good the embedder is."
    )
    lines.append("")

    lines.append("## False-answer rate by cell (headline)")
    lines.append("")
    lines.append("| cell | false-answer rate | n unanswerable | refusal rate (answerable) |")
    lines.append("|---|---|---|---|")
    for cell in run.cells:
        s = cell.summary
        lines.append(
            f"| {cell.mode}/{cell.rerank}/{cell.kinds} "
            f"| {_pct(s['false_answer']['rate'], degraded)} "
            f"| {s['false_answer']['n_unanswerable']} "
            f"| {_pct(s['refusal']['refusal_rate_answerable'])} |"
        )
    lines.append("")
    lines.append(
        "Over an extractive generator the false-answer rate measures "
        "**refusal-threshold logic, not hallucination**: an extractive composer "
        "cannot invent facts the way an LLM can, so this number does not transfer "
        "to an LLM-backed configuration."
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the 18-cell ablation grid.")
    ap.add_argument("--templates", default="dev", choices=("dev", "heldout", "all"))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--limit", type=int, default=None, help="cap n before the grid starts")
    ap.add_argument("--timebox-minutes", type=float, default=DEFAULT_TIMEBOX_MINUTES)
    ap.add_argument("--sample-size", type=int, default=None, help="n to fall back to")
    ap.add_argument("--source", default=None)
    ap.add_argument("--generator", default="extractive")
    ap.add_argument("--embedder", default="auto")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    ap.add_argument("--out", default=str(ABLATIONS_MD))
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    manifest = Manifest.load(args.manifest)
    source_uri = args.source or f"sqlite:{EVALS_DIR / 'data' / 'eval.sqlite'}"
    generation = GenerationConfig(generator=args.generator)

    questions = build_questions(manifest, args.templates, args.seed, limit=args.limit)

    # Wrapped before ingest, so the chunk vectors go through the cache too --
    # installing it afterwards would only ever catch query embeddings.
    holder: dict[str, CachedEmbedder | None] = {"cache": None}

    def _wrap(app: Any) -> None:
        if not args.no_cache:
            holder["cache"] = install_cache(app)

    engine = build_engine(
        source_uri,
        embedder=args.embedder,
        generator=args.generator,
        generation=generation,
        on_built=_wrap,
    )
    cache = holder["cache"]
    if cache is None and not args.no_cache:
        print(
            "note: the engine exposes no writable `.embedder`, so the disk cache "
            "is inactive. The grid still embeds once: the engine is built and "
            "ingested a single time and only RetrievalConfig varies per cell.",
            file=sys.stderr,
        )

    print(f"running 18 cells over n={len(questions)} questions", file=sys.stderr)
    run = run_grid(
        engine,
        questions,
        generation=generation,
        timebox_seconds=args.timebox_minutes * 60,
        seed=args.seed,
        sample_size=args.sample_size,
        cache=cache,
    )

    provenance = build_provenance(
        engine=engine,
        manifest=manifest,
        questions=run.questions,
        retrieval=grid_configs()[0][3],
        generation=generation,
        templates=args.templates,
        seed=args.seed,
        source_uri=source_uri,
    )

    RESULTS_SUBDIR.mkdir(parents=True, exist_ok=True)
    for cell in run.cells:
        write_results(
            {
                "provenance": {**provenance, "retrieval_config_name": cell.config_name},
                "cell": {"mode": cell.mode, "rerank": cell.rerank, "kinds": cell.kinds},
                "summary": cell.summary,
                "results": cell.results,
            },
            cell.key,
            RESULTS_SUBDIR,
        )

    markdown = render_markdown(run, provenance)
    Path(args.out).write_text(markdown + "\n", encoding="utf-8")
    print(f"\nwrote {args.out} ({len(run.cells)} cells)", file=sys.stderr)
    print(f"wrote per-cell results to {RESULTS_SUBDIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
