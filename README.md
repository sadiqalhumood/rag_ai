# anyrag

Point it at a database. Ask questions in natural language. Get answers with
citations back to specific rows.

`anyrag` is a source-agnostic retrieval-augmented generation system. Its quality
is established by an offline evaluation harness that generates its own ground
truth — not by reading outputs and deciding they look reasonable.

## What makes it different from document RAG

Retrieval over a database is not retrieval over documents:

- **"How many orders came from EMEA" has no answer in any single row.** Fetching
  the five most similar rows and letting a model count them produces a
  confident wrong number. `anyrag` classifies each question as LOOKUP (retrieval
  answers it), AGGREGATE (needs generated SQL), or HYBRID, and routes
  accordingly. Generated SQL passes a read-only linter, executes against the
  source, and its result becomes a citable chunk.
- **Uncited answers are unconstructible**, not discouraged. `Answer` raises
  `CitationError` if a non-refusal carries no citations.
- **Refusal is a first-class outcome.** The headline eval metric is the
  false-answer rate on questions whose answers are genuinely absent from the
  data.

## Design constraints

| Constraint | How it is met |
|---|---|
| Source-agnostic | One `DataSource` Protocol; adapters self-register via `pkgutil` discovery, so a new backend is one new file with no registry table to edit |
| Fully offline | Local `all-MiniLM-L6-v2` embeddings, local vector + BM25 indexes, deterministic extractive generator. Remote embedding/LLM providers are optional plugins behind the same interfaces |
| Read-only at the source | `lint_readonly` accepts only a single SELECT/WITH, and the driver is constrained too (SQLite `mode=ro` + `query_only`, Postgres read-only transaction) |
| Every answer cited | Citations carry `chunk_id`, `source_id`, and `row_refs` pointing at specific source rows |

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# CPU-only torch avoids pulling CUDA wheels:
#   .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

cp .env.example .env      # then edit; .env is gitignored

.venv/bin/python -m pytest -q
```

## Layout

```
anyrag/
  core/        frozen shared contracts: types, interfaces, config, SQL linter, tokenizer
  embed/       local (sentence-transformers), hashing fallback, optional remote
  sources/     DataSource adapters: sqlite, postgres, files (CSV/Parquet)
  ingest/      row serialization + schema cards, deterministic chunk IDs
  index/       vector index (memory + persistent) and BM25, incremental upsert
  retrieval/   hybrid dense+BM25, RRF fusion, prefilter, rerank, expansion
  route/       LOOKUP/AGGREGATE/HYBRID router and schema-driven NL->SQL
  generate/    token-budgeted packing, citation enforcement, refusal path
evals/         synthetic corpus, programmatic gold answers, metrics, ablations
```

## Documentation

- `PLAN.md` — the design, including the eval contract and file-ownership map
- `DECISIONS.md` — judgement calls made during an unattended build, with the
  alternatives rejected and why
- `BLOCKERS.md` — what could not be made to work, and what it cost
- `ABLATIONS.md` — the full retrieval ablation grid
- `HANDOFF.md` — what works, what is stubbed, and the commands that reproduce
  every reported number

## Configuration

All configuration is via environment variables; see `.env.example`. No
credentials or connection strings are committed — the Postgres adapter reads its
DSN from `ANYRAG_PG_DSN`.
