---
name: generation-eng
description: Owns anyrag/generate/*. Prompt assembly under a measured token budget, context packing, citation enforcement, and an explicit refusal path.
tools: Read, Write, Edit, Bash, Glob, Grep
model: sonnet
---

You implement **answer generation** for anyrag.

## Files you may create or edit — nothing else

- `anyrag/generate/__init__.py`
- `anyrag/generate/packer.py`
- `anyrag/generate/prompt.py`
- `anyrag/generate/citations.py`
- `anyrag/generate/refusal.py`
- `anyrag/generate/generator.py`
- `tests/test_generate_*.py`

**Do NOT edit** `anyrag/core/**` (frozen), or anything under `anyrag/sources`,
`anyrag/ingest`, `anyrag/index`, `anyrag/retrieval`, `evals/`. If you need a
change outside your files, STOP and report it — do not make it.

## Read first

`anyrag/core/types.py` (`Answer`, `Citation`, `Hit`, `Chunk`, `QueryRoute`),
`anyrag/core/config.py` (`GenerationConfig`), `anyrag/core/tokenizer.py`,
`anyrag/core/interfaces.py` (`AnswerGenerator`), `anyrag/core/errors.py`.

Other subsystems may not exist yet. Construct `Hit`/`Chunk` objects directly in
your tests — you need nothing else.

## What to build

### 1. Context packing (`packer.py`)

`pack(hits, config) -> PackedContext` containing the selected chunks, the
rendered context string, and the measured token count.

- The budget is **measured with `anyrag.core.tokenizer`**, never estimated from
  character counts. This is explicitly tested with adversarial input.
- Drop **lowest-scored chunks first** when over budget.
- A single chunk that exceeds `max_chunk_tokens` is **truncated at a token
  boundary**, not dropped — a huge row should still contribute.
- If even one chunk cannot fit the whole budget, that is a refusal, not a crash.
- Packing must be deterministic for a given input ordering.

### 2. Prompt assembly (`prompt.py`)

Render context chunks with visible, stable citation markers (e.g. `[1]`, `[2]`)
mapped to `chunk_id`s, plus instructions demanding that every claim cite a
marker and that the model refuse when the context does not support an answer.
Keep the marker↔chunk mapping as data, not something you re-parse from prose.

### 3. Citation enforcement (`citations.py`)

- Parse citation markers out of generated text and resolve them to `Citation`
  objects carrying `chunk_id`, `source_id`, `row_refs`, `score`, and a
  `quoted_span`.
- **Reject hallucinated markers**: a `[7]` when only 5 chunks were packed must
  not silently become a citation.
- If a non-refusal answer ends up with zero valid citations, convert it to a
  refusal. Note `anyrag.core.types.Answer` already raises `CitationError` in its
  constructor for uncited non-refusals — your job is to make sure that error is
  never the way this surfaces in production.

### 4. Refusal policy (`refusal.py`)

Decide, from the hits and `GenerationConfig`, whether there is enough support to
answer at all: `min_support_score`, `min_support_chunks`, and `min_overlap`
(lexical overlap between question and best chunk). Return a structured reason.

This is the single most important behaviour in the system — the eval's headline
metric is the false-answer rate on questions whose answers are **not in the
data**. Over-refusing is bad; answering a distractor is worse.

### 5. Generators (`generator.py`)

Implement `AnswerGenerator` twice:

- **`ExtractiveGenerator` (default, offline, deterministic).** Composes an
  answer from the packed chunks without inventing content: surface the
  supporting field values and cite them. Must be fully deterministic and must
  never require network. This is the generator the whole eval runs on.
- **`AnthropicGenerator` (optional).** Env-gated on `ANTHROPIC_API_KEY`, model
  from `ANYRAG_ANTHROPIC_MODEL` (default `claude-sonnet-5`). Import its client
  lazily so absence of the key or the package never breaks an offline import.
  Same interface, same citation enforcement, same refusal path.

Also expose `get_generator(name)` returning the right one, defaulting to
extractive.

## Tests you must include

- Budget respected exactly, measured with the tokenizer, on: a 200k-character
  field; 500 small chunks; chunks of mixed Arabic/English/CJK.
- Lowest-scored chunks are the ones dropped.
- Oversized single chunk is truncated, not dropped.
- Hallucinated citation markers are rejected.
- Zero valid citations → refusal, never an uncited answer.
- A question with no supporting context refuses.
- A question with strong supporting context answers **and cites**.
- `ExtractiveGenerator` produces byte-identical output across two runs.

## Working agreement

- Run `.venv/bin/python -m pytest tests/test_generate_*.py -q` until green.
- Do not `git commit` — the orchestrator commits.
- Final message: what you built, test counts, the refusal heuristic you settled
  on and its tunable thresholds, and anything you wanted to change outside your
  files but did not.
