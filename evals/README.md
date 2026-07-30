# `evals/` — the self-grading harness

The harness generates its own data, so it knows every gold answer by direct
computation. **No model is ever asked what the right answer is, and nothing is
queried through the system under test.** That is the only property that makes a
self-grading report worth reading.

```
gen_db.py     seeded synthetic SQLite + manifest.json (schema + every row)
questions.py  343 questions across 8 types, gold computed over the manifest
metrics.py    exact retrieval / aggregate / citation / refusal metrics
run_eval.py   one config -> evals/results/<name>.json, provenance-stamped
ablate.py     the 18-cell grid -> ABLATIONS.md
tests/        the harness grading itself
```

## Commands

```bash
.venv/bin/python -m evals.gen_db --seed 7                     # db + manifest
.venv/bin/python -m evals.gen_db --seed 7 --export-dir evals/exports   # + CSV/Parquet
.venv/bin/python -m evals.questions --templates dev --show 10  # inspect the set
.venv/bin/python -m evals.run_eval --config hybrid_rerank_both
.venv/bin/python -m evals.run_eval --subset distractors --generator anthropic
.venv/bin/python -m evals.ablate --timebox-minutes 90
.venv/bin/python -m pytest evals -q                            # 124 tests
```

Everything is offline and a pure function of `--seed`.
`manifest["meta"]["content_hash"]` is the reproducibility proof: two runs with
the same seed produce the same hash, and every results file records it.

## The public surface the judge touches

```python
AnyRAG.ask(question, config)      -> Answer
AnyRAG.retrieve(question, config) -> list[Hit]
chunk.row_refs                    -> tuple[RowRef]
```

Nothing else. `evals/tests/fakes.py` implements exactly that surface, which is
why the harness was buildable and fully testable before `anyrag/app.py` existed.

## Question types

| type | n | route | gold |
|---|---|---|---|
| `entity_lookup` | 56 | LOOKUP | one row |
| `multi_filter_lookup` | 40 | LOOKUP | 1–8 rows |
| `count_by_category` | 40 | AGGREGATE | integer scalar |
| `numeric_aggregate` | 44 | AGGREGATE | float scalar |
| `date_range_aggregate` | 36 | AGGREGATE | integer scalar |
| `join_relationship` | 40 | HYBRID | 1–3 rows, sometimes + scalar |
| `schema_question` | 23 | LOOKUP | a schema card |
| `distractor` | 64 | varies | **none — refusal is correct** |

343 total. Gold for every one is computed in plain Python over
`manifest.json`; `tests/test_questions.py` recomputes a sample of the same
answers with SQL against the generated SQLite file, so two independent paths
have to agree before any of it is trusted.

## Metric conventions

These are the choices a reader has to know to interpret a number.

- **Relevance.** A retrieved chunk is relevant iff it covers a gold *key*:
  its `row_refs` intersect the gold row refs, or it is the `SCHEMA_CARD` of a
  gold table. Schema questions grade on the same machinery as row questions.
- **recall@k is gold coverage**, not a hit flag — the fraction of gold keys
  covered by the top k. For single-row gold it degenerates to the usual hit
  rate; for eight-row gold it correctly refuses to call one row a success.
  `hit@k` is reported alongside.
- **nDCG@10 uses novel-coverage binary gains.** A chunk scores only if it covers
  a gold key no higher-ranked chunk already covered, and the ideal is
  `min(n_gold, 10)`. Without the novelty rule, two chunks of the same row both
  score and nDCG can exceed 1.0.
- **Rank is list position**, not `Hit.rank`. `rank=0` means "unranked" (D13), so
  trusting the field would make an unpopulated result look like an all-way tie.
- **Aggregate tolerance.** Integer gold (counts) must match **exactly**. Float
  gold uses relative tolerance `AGG_REL_TOL = 1e-6` with an absolute floor of
  `1e-9` so a gold of 0.0 is comparable. Gold aggregates skip nulls, exactly as
  SQL does — `customers.credit_limit` is ~18% null on purpose.
- **Citation precision** is graded only where there is gold evidence to check
  against. A pure `COUNT` has a scalar gold and no row gold: nothing says which
  chunk a count *ought* to cite, so its citation precision is `None`, not 0.
- **Unanswerable questions have no citation precision** either. Their failure
  mode is the false-answer rate, and scoring them 0 would double-count it.
- **A crash is not a refusal.** An exception is recorded as an error and the
  question counts as a non-refusal, so a system that crashes on every distractor
  cannot post a perfect false-answer rate.
- **Refusing an answerable aggregate** is scored incorrect, but it is not a
  false answer. The refusal rate and SQL coverage separate the two failures.

## Distractors, and why they are the way they are

The false-answer rate is the headline, so a trivially refusable distractor set
would make the whole report meaningless. The 64 distractors are:

| kind | n | why it is hard |
|---|---|---|
| `near_miss_entity` | 20 | plausible alternative *spellings* ("Lina Nassar" for "Lina Nasser"), not random typos |
| `missing_attribute` | 10 | a **real** customer or product, an attribute no table has |
| `out_of_range_date` | 10 | windows entirely outside the two-year span |
| `near_miss_category` | 8 | "Stationary" for "Stationery" — a real English word |
| `near_miss_enum` | 6 | "refunded", "kiosk", "Diamond" tier |
| `missing_table` | 6 | suppliers, warehouses, invoices |
| `near_miss_region` | 4 | "Riyad", "Dubay" |

Two rules are enforced in code and in tests:

1. **Absence is verified against a normalised form** of every real value
   (casefolded, non-alphanumerics stripped). Plain equality is not enough: the
   data deliberately contains both "Ahmed Al-Sayed" and "Ahmad Al Sayed", so a
   perturbation differing only in punctuation would be scored unanswerable while
   any reasonable system finds it.
2. **"How many orders were placed in March 2025" is not a distractor.** Zero is
   the correct answer to it. Only questions whose empty result is genuinely
   *undefined* (an average over an empty set) or which ask to enumerate and cite
   rows that do not exist are used. Counting a legitimate zero as a
   hallucination would make the headline number dishonest in our favour.

## Dev vs held-out

`DEV_TEMPLATES` is written. `HELDOUT_TEMPLATES` is **deliberately empty** until
`route/router.py` and `route/sqlgen.py` are frozen and the freeze SHA is in
`DECISIONS.md`. A held-out set written against a router that does not exist yet
measures nothing. `--templates heldout` raises rather than silently returning an
empty set, and `--templates all` resolves to dev-only while that holds.

Dropping the held-out set in requires adding `@heldout_template(...)` functions
and flipping `HELDOUT_AVAILABLE` — one edit, in one file, no restructuring.

## Provenance

Every results JSON records: the active embedder (with its `degraded` flag), the
generator, the tokenizer, the template set, `n`, `n` per type, the seed, the
manifest content hash, the retrieval and generation configs, the git commit, and
which `ask`/`retrieve` call shape the adapter actually used. A number without
provenance is not reportable; two runs whose provenance blocks differ are not
comparable, and the file says so on its face.

If the embedder is degraded, `ABLATIONS.md` marks **each affected metric cell**
inside the table (`0.412 ⚠`) plus a `degraded` column — not only in prose.

## Timebox

`ablate.py` projects the full grid cost after each cell. If it will exceed 90
minutes, it cuts to one seeded stratified sample, **restarts the grid from cell
one** on that sample, and records the trigger, sizes, per-type counts and seed
in the `ABLATIONS.md` header. Grid coverage beats question count: all 18 cells
always use the identical sample, because an inconsistent sample makes cells
incomparable, which is worse than a smaller n.
