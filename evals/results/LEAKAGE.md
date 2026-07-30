# Dev vs held-out (leakage check)

Embedder: **local:sentence-transformers/all-MiniLM-L6-v2**
- generator: `extractive`   tokenizer: `tiktoken`
- config: `hybrid_rerank_card+row+sql`   seed: `7`
- n: dev **338**, held-out **341**
- manifest content hash: `7f12436d90929ce0dd073fe556ebaba05d83c6217fc083b36bbcc61b2b81e06d`
- git commit: `061a5e74061b56c4af74e5a334950ecb22bdc89f`

| metric | dev | held-out | gap |
|---|---|---|---|
| false-answer rate | 13.3% | 43.9% | +30.6 pts **←— large** |
| aggregate accuracy | 69.9% | 51.5% | -18.4 pts **←— large** |
| router accuracy | 64.8% | 74.8% | +10.0 pts |
| SQL coverage | 71.2% | 63.9% | -7.3 pts |
| citation precision | 44.3% | 34.1% | -10.2 pts **←— large** |
| recall@10 | 0.7294 | 0.7361 | +0.0067 |
| nDCG@10 | 0.6933 | 0.6753 | -0.0180 |
| MRR | 0.7461 | 0.7189 | -0.0272 |

Never pooled. Each column is one template set, run separately with the same retrieval config, seed and manifest.

## Where the held-out distractors failed

### dev

| distractor kind | probe | false answers | rate |
|---|---|---|---|
| `near_miss_enum` | `-` | 2/6 | 33% |
| `near_miss_entity` | `-` | 5/20 | 25% |
| `missing_table` | `-` | 1/6 | 17% |
| `missing_attribute` | `-` | 0/10 | 0% |
| `near_miss_category` | `-` | 0/8 | 0% |
| `near_miss_region` | `-` | 0/4 | 0% |
| `out_of_range_date` | `-` | 0/6 | 0% |

### held-out

| distractor kind | probe | false answers | rate |
|---|---|---|---|
| `near_miss_enum` | `value_without_adjacent_column_word` | 9/10 | 90% |
| `missing_attribute` | `possessive_or_imperative_frame` | 10/12 | 83% |
| `missing_relationship` | `real_nouns_absent_edge` | 6/8 | 75% |
| `missing_table` | `entity_verb_outside_pattern` | 2/10 | 20% |
| `near_miss_category` | `untouched_column_vocabulary` | 1/8 | 12% |
| `out_of_range_date` | `prose_date_not_iso_pair` | 1/8 | 12% |
| `missing_table` | `fresh_fake_tables` | 0/6 | 0% |
| `near_miss_entity` | `possessive_free_near_miss` | 0/4 | 0% |

**attribute guard (`unresolved_attributes`)** — identical semantics, different sentence frame:

| frame matches the guard's regex? | false answers | rate |
|---|---|---|
| yes | 1/3 | 33% |
| no | 9/9 | 100% |

**entity guard (`unknown_entities`)** — identical semantics, different sentence frame:

| frame matches the guard's regex? | false answers | rate |
|---|---|---|
| yes | 0/4 | 0% |
| no | 2/6 | 33% |

