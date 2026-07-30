# HANDOFF

**Active embedder: `local:sentence-transformers/all-MiniLM-L6-v2` @ `1110a243fdf4` (dim 384) — the real model, NOT degraded.** Verified before any code was written: cos(paraphrase)=0.5643 vs cos(unrelated)=-0.0289. The `HashingEmbedder` fallback is implemented and unit-tested but was never used, so no number in this repo carries a degraded-embedder caveat.

> ## ⚠️ Read this before quoting any number
>
> **The dev false-answer rate of 13.3% does not generalise. On held-out templates it is 43.9% — a +30.6 point gap.** Three rounds of dev-driven tuning moved dev from 65.6% to 12.5%, and nearly all of that gain was phrasing-specific: held-out sits closer to the *first* round (46.9%) than the last. **43.9% is the number to quote.** Details in bullet 1 and `evals/results/LEAKAGE.md`.
>
> The ablation grid (n=343, commit `061a5e7`) was measured before eval-eng deduplicated questions and stabilised qids; single-config dev numbers are now n=338. The grid's *relative* comparisons stand — every cell shares the same question set — but its absolute false-answer column reads 12.5% where the corrected dev figure is 13.3%.

---

## Ten bullets

1. **The most important result is a negative one: the leakage check fired.** Dev false-answer rate 13.3%, held-out **43.9%**, gap **+30.6 points**. The gap is precisely localised, which makes it diagnostic rather than merely bad news — retrieval is flat (recall@10 0.729 → 0.736) and router accuracy is *better* on held-out (64.8% → 74.8%), so the embedder, retriever and route classifier all generalise. **Only the refusal layer collapsed.** Verified directly against the frozen generator:
   ```
   "How many orders have the status 'expedited'?"  -> REFUSED (unknown value)
   "How many orders were expedited?"               -> SELECT COUNT(*) FROM orders
   ```
   Identical semantics, identical schema facts, different sentence frame; the second returns the unfiltered total with full confidence. The guard checks for an **adjacent column word** — it does not check the schema, despite my claim in DECISIONS D29 that it did. Guards that consult *data* (out-of-range dates, NULL aggregates) transferred; guards that match *sentence shape* did not. That is the whole lesson.

2. **What works end to end.** Point it at SQLite, Postgres, or a directory of CSV/Parquet; it ingests, indexes (dense + BM25), routes each question LOOKUP/AGGREGATE/HYBRID, generates and lints SQL for aggregates, and returns answers with citations back to specific rows. 882 unit tests + 135 eval-harness tests pass.

3. **That number measures refusal-threshold logic, not hallucination resistance.** The default `ExtractiveGenerator` composes answers by selection only — a test asserts every emitted line is a verbatim substring of retrieved text — so it *cannot* fabricate the way an LLM can. **This number does not transfer to an LLM-backed config.** `AnthropicGenerator` exists behind the same interface and applies the same gates, but **`ANTHROPIC_API_KEY` was not set in this environment, so the side-by-side LLM comparison was never run.** There is no LLM false-answer number in this repo.

4. **What the ablations showed.** Best cell: hybrid + rerank + both chunk kinds (recall@1 0.603, nDCG@10 0.695). But the **chunk-kind axis dominates the retriever axis**: row-chunks cells score exactly **0.000** on schema questions, because they cannot retrieve a schema card at all. Rerank gives a consistent modest lift (hybrid/both: 0.556 → 0.603 recall@1). BM25 alone beats dense alone on citation precision (59.5% vs 45.8%) at similar recall. Whole grid is in `ABLATIONS.md`; all 18 cells ran on the full n=343 with **no timebox reduction**.

5. **A cell that looks best and isn't.** Schema-cards-only posts the lowest false-answer rate (6.2% vs 12.5%) while refusing **40.9% of answerable questions**. Reporting the entire grid rather than the winning row is what makes that visible.

6. **Source-agnosticism, measured not asserted.** The CSV/Parquet adapter was written last, deliberately. Diff: **exactly two new files** (`anyrag/sources/files.py`, `tests/test_sources_files.py`), nothing else modified, no registry edit — `pkgutil` discovery picked up the scheme with zero edits and `DISCOVERY_ERRORS` is empty. `introspect.py`/`profile.py` were reused unchanged.

7. **What broke on the second source, and why.** Running the same pipeline over Parquet: source-independent metrics identical or better (false-answer 12.5% same, router 64.4% same, SQL coverage 71.2% same, aggregate accuracy 69.9% → **78.7%**), but **retrieval recall@1 collapsed 0.603 → 0.168** and citation precision 44.8% → 8.7%. Cause confirmed, not inferred: SQLite emits `RowRef customers#1` from the real `customer_id`; the files adapter emits `customers#0` from a row ordinal, because flat files have no natural key. **The pipeline is source-agnostic; gold *identity* is not portable.** Both components are individually correct — the defect is in the contract between them.

8. **Chunk-ID stability holds on real data.** 5,777 chunks from the 5,767-row synthetic DB; a second ingest embedded **0** and left index size unchanged. Same property verified independently on the Parquet source. IDs deliberately exclude content (content hash lives in `meta`), which is what makes re-ingestion an update rather than a duplication.

9. **What is stubbed or unproven.** (a) No LLM generator run — no API key. (b) The ablation's embedding cache never intercepted (it wrapped the embedder after ingest had already run); eval-eng has since fixed it, but the `ABLATIONS.md` in this repo was produced with `0 hits / 0 misses`. It was never load-bearing — building the engine once and varying only `RetrievalConfig` is what gets the grid to 4.3 min. (c) NL→SQL is heuristic and single-hop only: `orders → regions` needs two FK hops and is unsupported, so SQL coverage is 71.2%, not ~100%. (d) Router accuracy is **64.4%** — the weakest headline number and the most obvious place to improve. (e) `RemoteEmbedder` and `CrossEncoderReranker` are implemented but unexercised.

10. **The fix is scoped but deliberately not applied.** Match candidate values against the full categorical vocabulary regardless of adjacency, and replace the regex frames with a real notion of "restrictive modifier on the counted entity". Tuning against the held-out set would repeat exactly the error it just exposed and leave nothing clean to validate on; doing it properly needs a *third* template set written after a re-freeze. The diagnosis, the mechanism and the exact failing pair are worth more than a fix that cannot be honestly validated.

---

## Held-out results (the leakage check)

| metric | dev (n=338) | held-out (n=341) | gap |
|---|---|---|---|
| **false-answer rate** | **13.3%** (8/60) | **43.9%** (29/66) | **+30.6 pts** |
| aggregate accuracy | 69.9% | 51.5% | −18.4 |
| citation precision | 44.3% | 34.1% | −10.2 |
| SQL coverage | 71.2% | 63.9% | −7.3 |
| router accuracy | 64.8% | 74.8% | **+10.0** |
| recall@10 | 0.7294 | 0.7361 | +0.007 |
| nDCG@10 | 0.6933 | 0.6753 | −0.018 |

Reported separately and never pooled; `summarize()` emits `by_template_set` so even `--templates all` cannot merge them.

**Which guards transferred.** Failed: unanchored values 9/10, possessive/imperative attribute frames 10/12, absent *relationships* 6/8 (no guard covers those at all). Transferred: fake table names 0/6, possessive near-miss 0/4, untouched column vocabulary 1/8, prose dates 1/8.

**Caveat on the held-out set itself, passed through from eval-eng:** it read the frozen router before writing these templates. Nothing was contorted to break the guards — every question is a form a real user would type — but it knew where to look. **Treat 43.9% as a well-targeted probe of known weak spots, not an unbiased estimate of production traffic.** The in-template controls are what make it diagnostic: identical semantics either side of a sentence-frame boundary, differing 9/9 vs 0/1.

**A separate coverage gap, not leakage:** `ho_date_periods` scores 0/16 — "the first half of 2024" silently widens to the whole year and returns a confident wrong number. Dev's two date forms both parse, so dev never probed it.

**The freeze is now enforced mechanically**, not by discipline: `questions.py` records the frozen files' blob hashes (which survive a rebase, unlike a commit SHA) and a test re-checks them on every run, so editing a frozen file fails the suite instead of silently invalidating these numbers.

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
