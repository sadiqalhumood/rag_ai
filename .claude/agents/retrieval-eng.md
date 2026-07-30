---
name: retrieval-eng
description: Owns anyrag/retrieval/*. Hybrid dense+BM25 search with reciprocal rank fusion, metadata pre-filtering, reranking behind an interface, and query expansion — every stage toggleable.
tools: Read, Write, Edit, Bash, Glob, Grep
model: sonnet
---

You implement **retrieval** for anyrag.

## Files you may create or edit — nothing else

- `anyrag/retrieval/__init__.py`
- `anyrag/retrieval/pipeline.py`
- `anyrag/retrieval/fusion.py`
- `anyrag/retrieval/rerank.py`
- `anyrag/retrieval/expand.py`
- `anyrag/retrieval/prefilter.py`
- `tests/test_retrieval_*.py`

**Do NOT edit** `anyrag/core/**` (frozen), or anything under `anyrag/sources`,
`anyrag/ingest`, `anyrag/index`, `anyrag/generate`, `evals/`. If you need a
change outside your files, STOP and report it — do not make it.

## Read first

`anyrag/core/config.py` (`RetrievalConfig`, `MetadataFilter`),
`anyrag/core/interfaces.py` (`VectorIndex`, `LexicalIndex`, `Reranker`,
`QueryExpander`, `Retriever`), `anyrag/core/types.py` (`Hit`), and the **already
implemented** `anyrag/index/*` — read it, do not reimplement it.

## What to build

### 1. Pipeline (`pipeline.py`)

```
expand → (dense ∥ lexical) → prefilter → RRF fuse → rerank → top-k
```

`HybridRetriever(vector_index, lexical_index, embedder, reranker=None,
expander=None)` with `retrieve(query, config: RetrievalConfig) -> list[Hit]`.

**Every stage must be independently switchable from `RetrievalConfig`**
(`dense`, `lexical`, `expansion`, `rerank`, `chunk_kinds`, `prefilter`, `k`,
`candidate_k`, `rrf_k`, `min_score`). The eval harness sweeps an 18-cell grid
over these; a stage that cannot be turned off is a bug. Setting `dense=False`
must mean *no vector search runs at all*, not that its results are discarded.

Pull `candidate_k` from each enabled retriever, fuse, then return `k`.

Expose a `trace` (dict) on the retriever or return value recording which stages
ran and how many candidates each produced — the harness reports this.

### 2. Reciprocal rank fusion (`fusion.py`)

`rrf(ranked_lists, k=60) -> list[Hit]` scoring `sum(1 / (k + rank))` across
lists. Get the rank convention right — read the convention `anyrag/index/*`
actually uses rather than assuming — and test that a chunk ranked highly by both
retrievers beats one ranked highly by only one.

Fused hits should carry `retriever="rrf"` and record the per-retriever ranks in
a way callers can inspect.

Also implement a score-normalisation fusion as an alternative and note in your
final message which you made the default and why. RRF is the default unless you
have evidence otherwise.

### 3. Pre-filtering (`prefilter.py`)

Metadata filtering applied **before** scoring, pushed down into the index calls
(the index API already takes a `MetadataFilter`). Filtering after retrieval
would silently return fewer than `k` results, which would corrupt recall@k.

Support restricting by `chunk_kinds` — this is the ablation axis for
{row-chunks, schema-cards, both}, so it must be exact.

### 4. Reranking (`rerank.py`)

An interface with three implementations:

- `NoopReranker` — identity, used when `rerank=False`.
- `LexicalOverlapReranker` — **the offline default**. Deterministic: rescore by
  token overlap / coverage of the query terms in the chunk, blended with the
  fused score. Must be meaningful enough that "rerank on" is a real experimental
  condition offline, not a no-op with a different name.
- `CrossEncoderReranker` — optional, `cross-encoder/ms-marco-MiniLM-L-6-v2` via
  sentence-transformers. Import lazily; if the model is unavailable, fall back to
  the lexical reranker and **record that it fell back** rather than failing.

### 5. Query expansion (`expand.py`)

Offline and deterministic: no LLM. Useful expansions for this corpus include
splitting on punctuation, generating an acronym/initialism form, adding
number/date normalisations, and transliteration-insensitive variants for Arabic
and English near-duplicate names (the corpus deliberately contains
"Ahmed Al-Sayed" vs "Ahmad Al Sayed"). Expanded queries are searched and their
results fused.

Keep it conservative — expansion that adds noise will show up as a *loss* in the
ablation, which is a legitimate finding, but do not make it obviously bad.

## Tests you must include

- Each stage toggle actually changes behaviour (`dense=False` runs no vector
  search — assert via a spy/fake index that records calls).
- RRF ranks a both-lists chunk above a one-list chunk.
- Pre-filtering returns `k` results when `k` matching chunks exist.
- `chunk_kinds` restriction returns only the requested kinds.
- Reranking with `LexicalOverlapReranker` changes order in a case you construct.
- Empty index → `[]`, no exception.
- Determinism: the same query and config give the identical hit list twice.
- Score ties are broken deterministically (by `chunk_id`), so ablation cells are
  reproducible.

## Working agreement

- Run `.venv/bin/python -m pytest tests/test_retrieval_*.py -q` until green.
- Do not `git commit` — the orchestrator commits.
- Final message: what you built, test counts, your fusion default and why, and
  anything you wanted to change outside your files but did not.
