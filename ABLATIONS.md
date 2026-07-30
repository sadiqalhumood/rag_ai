# ABLATIONS

Embedder: **local:sentence-transformers/all-MiniLM-L6-v2** @ `1110a243fdf4`

- generator: `extractive`
- tokenizer: `tiktoken`
- template set: `dev`
- n (questions per cell): **343**
- seed: `7`
- manifest content hash: `7f12436d90929ce0dd073fe556ebaba05d83c6217fc083b36bbcc61b2b81e06d`
- git commit: `061a5e74061b56c4af74e5a334950ecb22bdc89f`
- grid wall time: 4.3 min over 18 cells
- embedding cache: 0 hits / 0 misses, 0 entries

The complete 18-cell grid ran on the full question set (n=343); no timebox reduction was applied.

## The full grid (all 18 cells)

| retrieval | rerank | chunks | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 | cite prec | **false-answer** | router acc | SQL cov | n | degraded |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | off | row-chunks | 0.398 | 0.473 | 0.504 | 0.489 | 0.466 | 45.8% | 12.5% | 64.4% | 71.2% | 343 | no |
| dense | off | schema-cards | 0.132 | 0.145 | 0.145 | 0.138 | 0.140 | 16.0% | 6.2% | 64.4% | 71.2% | 343 | no |
| dense | off | both | 0.517 | 0.612 | 0.642 | 0.617 | 0.597 | 43.0% | 12.5% | 64.4% | 71.2% | 343 | no |
| dense | on | row-chunks | 0.456 | 0.499 | 0.531 | 0.571 | 0.515 | 50.7% | 12.5% | 64.4% | 71.2% | 343 | no |
| dense | on | schema-cards | 0.145 | 0.145 | 0.145 | 0.145 | 0.145 | 15.4% | 6.2% | 64.4% | 71.2% | 343 | no |
| dense | on | both | 0.594 | 0.638 | 0.669 | 0.710 | 0.654 | 47.6% | 12.5% | 64.4% | 71.2% | 343 | no |
| bm25 | off | row-chunks | 0.455 | 0.532 | 0.572 | 0.582 | 0.540 | 59.5% | 12.5% | 64.4% | 71.2% | 343 | no |
| bm25 | off | schema-cards | 0.057 | 0.145 | 0.145 | 0.092 | 0.106 | 12.4% | 6.2% | 64.4% | 71.2% | 343 | no |
| bm25 | off | both | 0.493 | 0.644 | 0.687 | 0.649 | 0.619 | 52.4% | 12.5% | 64.4% | 71.2% | 343 | no |
| bm25 | on | row-chunks | 0.462 | 0.531 | 0.572 | 0.584 | 0.542 | 57.8% | 12.5% | 64.4% | 71.2% | 343 | no |
| bm25 | on | schema-cards | 0.138 | 0.145 | 0.145 | 0.142 | 0.142 | 15.4% | 6.2% | 64.4% | 71.2% | 343 | no |
| bm25 | on | both | 0.575 | 0.642 | 0.688 | 0.698 | 0.656 | 51.6% | 12.5% | 64.4% | 71.2% | 343 | no |
| hybrid | off | row-chunks | 0.455 | 0.546 | 0.584 | 0.593 | 0.545 | 50.2% | 12.5% | 64.4% | 71.2% | 343 | no |
| hybrid | off | schema-cards | 0.107 | 0.145 | 0.145 | 0.125 | 0.130 | 16.0% | 6.2% | 64.4% | 71.2% | 343 | no |
| hybrid | off | both | 0.556 | 0.678 | 0.724 | 0.710 | 0.668 | 44.9% | 12.5% | 64.4% | 71.2% | 343 | no |
| hybrid | on | row-chunks | 0.474 | 0.554 | 0.593 | 0.616 | 0.562 | 48.7% | 12.5% | 64.4% | 71.2% | 343 | no |
| hybrid | on | schema-cards | 0.145 | 0.145 | 0.145 | 0.145 | 0.145 | 15.4% | 6.2% | 64.4% | 71.2% | 343 | no |
| hybrid | on | both | 0.603 | 0.686 | 0.731 | 0.748 | 0.695 | 44.8% | 12.5% | 64.4% | 71.2% | 343 | no |

Retrieval columns are averaged over the questions with row/schema gold (n=159 of 343); count and pure-aggregate questions have a scalar gold rather than a retrievable one and are excluded from them by construction, not by selection. Router accuracy and SQL coverage do not depend on the retrieval axis and are shown per cell for completeness.

## recall@10 by question type

| cell | `count_by_category` | `date_range_aggregate` | `distractor` | `entity_lookup` | `join_relationship` | `multi_filter_lookup` | `numeric_aggregate` | `schema_question` |
|---|---|---|---|---|---|---|---|---|
| dense/off/row-chunks | n/a | n/a | n/a | 0.964 | 0.438 | 0.214 | n/a | 0.000 |
| dense/off/schema-cards | n/a | n/a | n/a | 0.000 | 0.000 | 0.000 | n/a | 1.000 |
| dense/off/both | n/a | n/a | n/a | 0.964 | 0.438 | 0.214 | n/a | 0.957 |
| dense/on/row-chunks | n/a | n/a | n/a | 1.000 | 0.450 | 0.259 | n/a | 0.000 |
| dense/on/schema-cards | n/a | n/a | n/a | 0.000 | 0.000 | 0.000 | n/a | 1.000 |
| dense/on/both | n/a | n/a | n/a | 1.000 | 0.450 | 0.259 | n/a | 0.957 |
| bm25/off/row-chunks | n/a | n/a | n/a | 1.000 | 0.525 | 0.347 | n/a | 0.000 |
| bm25/off/schema-cards | n/a | n/a | n/a | 0.000 | 0.000 | 0.000 | n/a | 1.000 |
| bm25/off/both | n/a | n/a | n/a | 1.000 | 0.525 | 0.330 | n/a | 0.826 |
| bm25/on/row-chunks | n/a | n/a | n/a | 1.000 | 0.525 | 0.347 | n/a | 0.000 |
| bm25/on/schema-cards | n/a | n/a | n/a | 0.000 | 0.000 | 0.000 | n/a | 1.000 |
| bm25/on/both | n/a | n/a | n/a | 1.000 | 0.525 | 0.336 | n/a | 0.826 |
| hybrid/off/row-chunks | n/a | n/a | n/a | 1.000 | 0.571 | 0.352 | n/a | 0.000 |
| hybrid/off/schema-cards | n/a | n/a | n/a | 0.000 | 0.000 | 0.000 | n/a | 1.000 |
| hybrid/off/both | n/a | n/a | n/a | 1.000 | 0.571 | 0.355 | n/a | 0.957 |
| hybrid/on/row-chunks | n/a | n/a | n/a | 1.000 | 0.600 | 0.359 | n/a | 0.000 |
| hybrid/on/schema-cards | n/a | n/a | n/a | 0.000 | 0.000 | 0.000 | n/a | 1.000 |
| hybrid/on/both | n/a | n/a | n/a | 1.000 | 0.600 | 0.356 | n/a | 0.957 |

The `schema_question` column is the reason the chunk-kind axis exists: a `row-chunks` cell cannot retrieve a schema card at all, so it scores zero there however good the embedder is.

## False-answer rate by cell (headline)

| cell | false-answer rate | n unanswerable | refusal rate (answerable) |
|---|---|---|---|
| dense/off/row-chunks | 12.5% | 64 | 34.8% |
| dense/off/schema-cards | 6.2% | 64 | 40.9% |
| dense/off/both | 12.5% | 64 | 30.1% |
| dense/on/row-chunks | 12.5% | 64 | 33.7% |
| dense/on/schema-cards | 6.2% | 64 | 40.9% |
| dense/on/both | 12.5% | 64 | 29.0% |
| bm25/off/row-chunks | 12.5% | 64 | 31.5% |
| bm25/off/schema-cards | 6.2% | 64 | 40.9% |
| bm25/off/both | 12.5% | 64 | 27.2% |
| bm25/on/row-chunks | 12.5% | 64 | 29.4% |
| bm25/on/schema-cards | 6.2% | 64 | 40.9% |
| bm25/on/both | 12.5% | 64 | 24.7% |
| hybrid/off/row-chunks | 12.5% | 64 | 30.8% |
| hybrid/off/schema-cards | 6.2% | 64 | 40.9% |
| hybrid/off/both | 12.5% | 64 | 26.2% |
| hybrid/on/row-chunks | 12.5% | 64 | 27.6% |
| hybrid/on/schema-cards | 6.2% | 64 | 40.9% |
| hybrid/on/both | 12.5% | 64 | 23.3% |

Over an extractive generator the false-answer rate measures **refusal-threshold logic, not hallucination**: an extractive composer cannot invent facts the way an LLM can, so this number does not transfer to an LLM-backed configuration.

