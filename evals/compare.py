"""Dev vs held-out, side by side. The leakage check (amendment 3).

    python -m evals.compare

Router accuracy and SQL coverage are reported **separately per template set and
never pooled**. A large dev -> held-out gap is not noise to average away: it is
a finding about heuristics fitted to phrasings that were visible during
development, and pooling the two sets is exactly how such a finding disappears.

The per-probe breakdown is the diagnostic half. Each held-out distractor carries
a `probe` tag naming the hypothesis it tests, and several templates contain both
a phrasing the guard's pattern covers and one it does not. Comparing those two
inside a single template separates "the guard is wrong about the schema" from
"the guard never fired", which the headline rate alone cannot do.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .gen_db import DEFAULT_SEED, Manifest
from .questions import build_questions
from .run_eval import RESULTS_DIR

LEAKAGE_MD = RESULTS_DIR / "LEAKAGE.md"

#: (label, path into summary, is it a percentage, does higher mean better)
METRICS: tuple[tuple[str, tuple[str, ...], bool, bool], ...] = (
    ("false-answer rate", ("false_answer", "rate"), True, False),
    ("aggregate accuracy", ("aggregate", "accuracy"), True, True),
    ("router accuracy", ("router", "accuracy"), True, True),
    ("SQL coverage", ("sql", "coverage"), True, True),
    ("citation precision", ("citations", "precision"), True, True),
    ("recall@10", ("retrieval", "recall@10"), False, True),
    ("nDCG@10", ("retrieval", "ndcg@10"), False, True),
    ("MRR", ("retrieval", "mrr"), False, True),
)


def _dig(obj: Mapping[str, Any], path: Sequence[str]) -> Any:
    for key in path:
        if not isinstance(obj, Mapping):
            return None
        obj = obj.get(key)  # type: ignore[assignment]
    return obj


def _fmt(value: Any, pct: bool) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%" if pct else f"{value:.4f}"


def compare(dev: Mapping[str, Any], heldout: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    dp, hp = dev["provenance"], heldout["provenance"]

    lines.append("# Dev vs held-out (leakage check)")
    lines.append("")
    lines.append(
        f"Embedder: **{dp['embedder'].get('name')}:{dp['embedder'].get('model')}**"
        + ("  — **DEGRADED**" if dp["embedder_degraded"] else "")
    )
    lines.append(f"- generator: `{dp['generator']}`   tokenizer: `{dp['tokenizer'].get('name')}`")
    lines.append(f"- config: `{dp['retrieval_config_name']}`   seed: `{dp['seed']}`")
    lines.append(f"- n: dev **{dp['n']}**, held-out **{hp['n']}**")
    lines.append(f"- manifest content hash: `{dp['manifest_content_hash']}`")
    lines.append(f"- git commit: `{dp['git_commit'] or 'unknown'}`")
    lines.append("")
    if dp["embedder"] != hp["embedder"]:
        lines.append(
            "> **The two runs used different embedders, so they are not comparable.**"
        )
        lines.append("")

    lines.append("| metric | dev | held-out | gap |")
    lines.append("|---|---|---|---|")
    for label, path, pct, higher_better in METRICS:
        a, b = _dig(dev["summary"], path), _dig(heldout["summary"], path)
        if a is None or b is None:
            continue
        gap = b - a
        worse = (gap < 0) if higher_better else (gap > 0)
        marker = ""
        if pct and abs(gap) >= 0.10:
            marker = " **←— large**" if worse else " *(held-out better)*"
        gap_text = f"{gap * 100:+.1f} pts" if pct else f"{gap:+.4f}"
        lines.append(f"| {label} | {_fmt(a, pct)} | {_fmt(b, pct)} | {gap_text}{marker} |")
    lines.append("")
    lines.append(
        "Never pooled. Each column is one template set, run separately with the "
        "same retrieval config, seed and manifest."
    )
    lines.append("")
    return lines


def distractor_breakdown(
    payload: Mapping[str, Any], questions: Mapping[str, Any], title: str
) -> list[str]:
    agg: dict[tuple[str, str], list[int]] = collections.defaultdict(lambda: [0, 0])
    for record in payload["results"]:
        if record["answerable"]:
            continue
        question = questions.get(record["qid"])
        if question is None:
            continue
        key = (
            str(question.meta.get("kind")),
            str(question.meta.get("probe") or "-"),
        )
        agg[key][1] += 1
        if not record["refused"]:
            agg[key][0] += 1

    lines = [f"### {title}", ""]
    lines.append("| distractor kind | probe | false answers | rate |")
    lines.append("|---|---|---|---|")
    for key in sorted(agg, key=lambda k: (-agg[k][0] / agg[k][1], k)):
        bad, n = agg[key]
        lines.append(f"| `{key[0]}` | `{key[1]}` | {bad}/{n} | {bad / n * 100:.0f}% |")
    lines.append("")
    return lines


def frame_probe(payload: Mapping[str, Any], questions: Mapping[str, Any]) -> list[str]:
    """Within one template: does the guard's regex frame decide the outcome?"""
    lines: list[str] = []
    for field, caption in (
        ("dev_style_frame", "attribute guard (`unresolved_attributes`)"),
        ("pattern_covered", "entity guard (`unknown_entities`)"),
    ):
        agg: dict[bool, list[int]] = collections.defaultdict(lambda: [0, 0])
        for record in payload["results"]:
            if record["answerable"]:
                continue
            question = questions.get(record["qid"])
            if question is None or field not in question.meta:
                continue
            covered = bool(question.meta[field])
            agg[covered][1] += 1
            if not record["refused"]:
                agg[covered][0] += 1
        if not agg:
            continue
        lines.append(f"**{caption}** — identical semantics, different sentence frame:")
        lines.append("")
        lines.append("| frame matches the guard's regex? | false answers | rate |")
        lines.append("|---|---|---|")
        for covered in (True, False):
            if covered not in agg:
                continue
            bad, n = agg[covered]
            lines.append(
                f"| {'yes' if covered else 'no'} | {bad}/{n} | {bad / n * 100:.0f}% |"
            )
        lines.append("")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare dev and held-out results.")
    ap.add_argument("--config", default="hybrid_rerank_both")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=str(LEAKAGE_MD))
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    paths = {
        name: results_dir / f"{args.config}__{name}.json" for name in ("dev", "heldout")
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print(
            "missing results file(s): "
            + ", ".join(missing)
            + "\nrun: python -m evals.run_eval --config "
            + f"{args.config} --templates dev  (and --templates heldout)",
            file=sys.stderr,
        )
        return 1

    payloads = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in paths.items()}
    manifest = Manifest.load(args.manifest) if args.manifest else Manifest.load()
    questions = {
        name: {q.qid: q for q in build_questions(manifest, name, args.seed)}
        for name in ("dev", "heldout")
    }

    lines = compare(payloads["dev"], payloads["heldout"])
    lines.append("## Where the held-out distractors failed")
    lines.append("")
    lines += distractor_breakdown(payloads["dev"], questions["dev"], "dev")
    lines += distractor_breakdown(payloads["heldout"], questions["heldout"], "held-out")
    lines += frame_probe(payloads["heldout"], questions["heldout"])

    markdown = "\n".join(lines)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
