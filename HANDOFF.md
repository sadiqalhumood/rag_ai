# HANDOFF

**Active embedder: `local:sentence-transformers/all-MiniLM-L6-v2` @ `1110a243fdf4` (dim 384) — the real model, NOT degraded.** Verified before any code was written: cos(paraphrase)=0.5643 vs cos(unrelated)=-0.0289. The `HashingEmbedder` fallback is implemented and unit-tested but was never used, so no number in this repo carries a degraded-embedder caveat.

> ⚠️ **Numbers below marked (provisional) were measured while `evals/questions.py` was being edited concurrently** to add the held-out template set. The dev set shifted from n=343 to n=338 mid-run. The ablation grid is internally consistent (single run, n=343, commit `061a5e7`), but the dev single-config numbers and the ablation were measured against slightly different question sets. **Re-run both to reconcile** — commands in §Reproduce.

---

## Ten bullets

1. **What works end to end.** Point it at SQLite, Postgres, or a directory of CSV/Parquet; it ingests, indexes (dense + BM25), routes each question LOOKUP/AGGREGATE/HYBRID, generates and lints SQL for aggregates, and returns answers with citations back to specific rows. 882 tests pass.

2. **The headline number: false-answer rate 12.5–13.3%** on unanswerable questions (8 false answers), down from **65.6% on the first real eval run**. Read the caveat in bullet 3 before quoting it.

3. **That number measures refusal-threshold logic, not hallucination resistance.** The default `ExtractiveGenerator` composes answers by selection only — a test asserts every emitted line is a verbatim substring of retrieved text — so it *cannot* fabricate the way an LLM can. **This number does not transfer to an LLM-backed config.** `AnthropicGenerator` exists behind the same interface and applies the same gates, but **`ANTHROPIC_API_KEY` was not set in this environment, so the side-by-side LLM comparison was never run.** There is no LLM false-answer number in this repo.

4. **What the ablations showed.** Best cell: hybrid + rerank + both chunk kinds (recall@1 0.603, nDCG@10 0.695). But the **chunk-kind axis dominates the retriever axis**: row-chunks cells score exactly **0.000** on schema questions, because they cannot retrieve a schema card at all. Rerank gives a consistent modest lift (hybrid/both: 0.556 → 0.603 recall@1). BM25 alone beats dense alone on citation precision (59.5% vs 45.8%) at similar recall. Whole grid is in `ABLATIONS.md`; all 18 cells ran on the full n=343 with **no timebox reduction**.

5. **A cell that looks best and isn't.** Schema-cards-only posts the lowest false-answer rate (6.2% vs 12.5%) while refusing **40.9% of answerable questions**. Reporting the entire grid rather than the winning row is what makes that visible.

6. **Source-agnosticism, measured not asserted.** The CSV/Parquet adapter was written last, deliberately. Diff: **exactly two new files** (`anyrag/sources/files.py`, `tests/test_sources_files.py`), nothing else modified, no registry edit — `pkgutil` discovery picked up the scheme with zero edits and `DISCOVERY_ERRORS` is empty. `introspect.py`/`profile.py` were reused unchanged.

7. **What broke on the second source, and why.** Running the same pipeline over Parquet: source-independent metrics identical or better (false-answer 12.5% same, router 64.4% same, SQL coverage 71.2% same, aggregate accuracy 69.9% → **78.7%**), but **retrieval recall@1 collapsed 0.603 → 0.168** and citation precision 44.8% → 8.7%. Cause confirmed, not inferred: SQLite emits `RowRef customers#1` from the real `customer_id`; the files adapter emits `customers#0` from a row ordinal, because flat files have no natural key. **The pipeline is source-agnostic; gold *identity* is not portable.** Both components are individually correct — the defect is in the contract between them.

8. **Chunk-ID stability holds on real data.** 5,777 chunks from the 5,767-row synthetic DB; a second ingest embedded **0** and left index size unchanged. Same property verified independently on the Parquet source. IDs deliberately exclude content (content hash lives in `meta`), which is what makes re-ingestion an update rather than a duplication.

9. **What is stubbed or unproven.** (a) No LLM generator run — no API key. (b) The ablation logged **0 embedding-cache hits and 0 misses**, so the cache the design called for was never exercised; it didn't matter at this scale (4.3 min for 18 cells) but the requirement is not demonstrated. (c) NL→SQL is heuristic and single-hop only: `orders → regions` needs two FK hops and is unsupported, so SQL coverage is 71.2%, not ~100%. (d) Router accuracy is **64.4%** — the weakest headline number and the most obvious place to improve. (e) `RemoteEmbedder` and `CrossEncoderReranker` are implemented but unexercised.

10. **The leakage risk, stated plainly.** Three rounds of guards took the dev false-answer rate 65.6% → 46.9% → 31.2% → 12.5%, and **every round was driven by inspecting dev-set failures**. That is textbook overfitting. `route/` was frozen at commit `60adca7` and the held-out templates written only afterwards, precisely so the gap is measurable. **Held-out numbers were pending at the time of writing — see §Held-out.** A large dev→held-out gap is a real finding about these guards, not noise to average away.

---

## Held-out results

*Pending at time of writing.* Run:

```bash
.venv/bin/python -m evals.run_eval --config hybrid_rerank_both --templates heldout
```

Report router accuracy and SQL coverage **separately** from dev; never pool them. If the held-out false-answer rate is materially worse than dev's, the guards in §Reproduce keyed on dev phrasing rather than on schema structure, and that is the honest conclusion.

---

## Reproduce every number

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only

# 882 unit tests (incl. ~45 adversarial SQL-linter cases, 60 live Postgres tests)
.venv/bin/python -m pytest -q

# Synthetic corpus: 5,767 rows, 5 FK-linked tables, nulls, Arabic+English, 2023-24 dates
.venv/bin/python -m evals.gen_db --seed 7

# Single config -> evals/results/<name>.json  (bullets 2, 3)
.venv/bin/python -m evals.run_eval --config hybrid_rerank_both --templates dev

# Full 18-cell grid -> ABLATIONS.md  (bullets 4, 5)
.venv/bin/python -m evals.ablate

# Held-out set  (bullet 10, section above)
.venv/bin/python -m evals.run_eval --config hybrid_rerank_both --templates heldout

# Source-agnosticism: export to CSV/Parquet, run the same pipeline  (bullets 6, 7)
.venv/bin/python -m evals.gen_db --seed 7 --export-dir evals/exports
mkdir -p /tmp/pq && cp evals/exports/*.parquet /tmp/pq/
.venv/bin/python -m evals.run_eval --config hybrid_rerank_both --templates dev \
    --source "files:/tmp/pq" --name files_source__dev

# Chunk-ID stability  (bullet 8)
.venv/bin/python -c "
from anyrag.app import AnyRAG
app = AnyRAG.from_uri('sqlite:evals/data/eval.sqlite')
a = app.ingest(); b = app.ingest()
print(a.chunks, 'chunks; re-ingest embedded', b.embedded, '-> size', app.vector_index.count())"
```

Optional, requires a key (bullet 3):
```bash
ANTHROPIC_API_KEY=... .venv/bin/python -m evals.run_eval \
    --subset distractors --generator anthropic --config hybrid_rerank_both
```

## Where the reasoning lives

`DECISIONS.md` records every judgement call with the alternative rejected — including the bugs found in my own code by the harness and by subagents (a tokenizer that counted 200k characters as one token; an ablation-cell naming collapse that would have silently turned 18 cells into 12; an RRF/threshold units mismatch that would have made all 12 single-retriever cells refuse everything and look like a finding). `BLOCKERS.md` records what could not be done. `PLAN.md` is the design as approved.
