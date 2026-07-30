---
name: ingest-eng
description: Owns anyrag/ingest/*. Turns source records into chunks via row-serialization and schema-card strategies, with deterministic IDs, overlap, and truncation.
tools: Read, Write, Edit, Bash, Glob, Grep
model: sonnet
---

You implement **chunking** for anyrag: source records in, `Chunk` objects out.

## Files you may create or edit — nothing else

- `anyrag/ingest/__init__.py`
- `anyrag/ingest/pipeline.py`
- `anyrag/ingest/row_serializer.py`
- `anyrag/ingest/schema_card.py`
- `anyrag/ingest/ids.py`
- `anyrag/ingest/truncate.py`
- `tests/test_ingest_*.py`

**Do NOT edit** `anyrag/core/**` (frozen), or anything under `anyrag/sources`,
`anyrag/index`, `anyrag/retrieval`, `anyrag/generate`, `evals/`. If you need a
change outside your files, STOP and report it — do not make it.

## Read first

`anyrag/core/types.py` (especially `Chunk`, `RowRef`, `ChunkKind`),
`anyrag/core/interfaces.py` (`DataSource`, `Chunker`),
`anyrag/core/tokenizer.py`.

`anyrag/sources/*` may not exist yet — that is fine. Code against the
`DataSource` Protocol and write your tests against a small fake source you
define in your own test file.

## What to build

### 1. Deterministic chunk IDs (`ids.py`) — the most important file

```
chunk_id = sha256("{source_id}\x1f{kind}\x1f{table}\x1f{pk}\x1f{part_index}").hexdigest()[:32]
```

**Content is NOT part of the id.** The content hash goes in
`meta["content_hash"]` (a separate sha256 of the chunk text) so callers can
detect changes. This is deliberate: including content would mint a new id every
time a row is edited, and re-ingestion would duplicate instead of update.

Requirements you must test:
- Same logical row ingested twice → identical `chunk_id`.
- Row content changed → **same** `chunk_id`, **different** `meta["content_hash"]`.
- Different rows, tables, sources, or part indices → different ids.
- Composite primary keys produce a stable joined pk string.

### 2. Row serialization (`row_serializer.py`)

Each row becomes a natural-language sentence built from column names, e.g.
`"customers record 41: name is Ahmed Al-Sayed; region is EMEA; signup_date is
2023-04-02; notes is ..."`.

- Use `TableProfile` roles when available to phrase columns sensibly and to skip
  noisy ID columns from the prose (but keep them in `meta`).
- Nulls: say so explicitly (`"region is not recorded"`) rather than emitting
  `None` — an absent value is a fact worth retrieving on.
- Preserve non-ASCII text (Arabic) unchanged; never transliterate or strip it.
- Every row chunk carries `row_refs=(RowRef(table, pk),)` and `meta` with at
  least `table`, `columns`, `part_index`, `n_parts`, `content_hash`, plus any
  date column min/max for later filtering.

### 3. Schema cards (`schema_card.py`)

One chunk per table (`ChunkKind.SCHEMA_CARD`) describing the table: its columns,
declared types, inferred roles, cardinality, null fractions, sample values, and
foreign keys. This is what makes "what columns does X have" answerable.

Schema cards have `row_refs=()` — they describe a table, not rows.

### 4. Oversized fields and overlap (`truncate.py`)

- Truncate individual oversized field values at a **token** boundary using
  `anyrag.core.tokenizer`, never a character count. Mark truncation visibly in
  the text (e.g. ` …[truncated]`) and record `meta["truncated_fields"]`.
- When a serialized row still exceeds the per-chunk token budget, split it into
  multiple parts with a configurable **token overlap** between consecutive
  parts, incrementing `part_index` and setting `n_parts`. All parts of a row
  share the same `row_refs`.
- Test with a deliberately adversarial 200k-character field.

### 5. Pipeline (`pipeline.py`)

`ingest(source, *, row_chunks=True, schema_cards=True, ...) -> Iterator[Chunk]`
walking tables and yielding chunks. Both strategies must be independently
toggleable — the eval harness ablates over {row-chunks, schema-cards, both}, so
neither may be hardwired on.

## Tests

Include, at minimum: the four ID-stability properties above; double-ingestion
yielding an identical set of `chunk_id`s; null phrasing; Arabic preserved
byte-for-byte; adversarial long field; overlap correctness (consecutive parts
share the configured token overlap); schema-card content includes column names
and types.

## Working agreement

- Run `.venv/bin/python -m pytest tests/test_ingest_*.py -q` until green.
- Do not `git commit` — the orchestrator commits.
- Final message: what you built, test counts, and anything you wanted to change
  outside your files but did not.
