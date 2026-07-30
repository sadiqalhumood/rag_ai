# anyrag — source-agnostic RAG with a self-grading eval oracle

## Context

`sadiqalhumood/rag_ai` is an empty repository on branch `claude/anyrag-rag-system-xmjvm7`
(no commits). We are building `anyrag` from scratch: a RAG system that can be pointed at an
*arbitrary* database (SQLite, Postgres, a directory of CSV/Parquet) and answer natural-language
questions over it with citations. The point of the exercise is not a demo — it is that quality
is *proven* by an offline eval harness that generates its own ground truth, so no human
judgment and no model-in-the-loop is needed to score retrieval, aggregate correctness,
citation precision, and (most importantly) the false-answer rate on unanswerable questions.

Environment facts confirmed by probing:
- Python 3.11.15, pip 24.0, no numpy/sentence-transformers yet → will use a `.venv`.
- PyPI reachable (200), HuggingFace reachable (307) → local MiniLM embeddings are feasible.
- **PostgreSQL 16 is installed** (`/usr/bin/psql`, `/usr/lib/postgresql/16`) → the Postgres
  adapter can be integration-tested for real, not just stubbed. Postgres refuses to run as
  root, so a cluster must be `initdb`'d under an unprivileged user; if that fails it becomes
  a BLOCKERS.md entry with `pytest.mark.skipif` guards.
- 4 cores, 15 GB RAM, 30 GB free disk → the 18-cell ablation matrix is affordable **only** if
  embeddings are computed once and cached; this is a design requirement, not an optimization.

Unattended run: no questions will be asked. Every ambiguous call gets resolved to the most
defensible option and logged in `DECISIONS.md`.

---

## The central architectural decision

**All shared contracts live in `anyrag/core/` and are written by the orchestrator (me) in
Phase 0, before any subagent is spawned.** Subagents import from `anyrag/core` and never edit
it. This is what makes "no two subagents write the same file" actually hold — the usual failure
mode is two agents both editing a shared `base.py`.

Anything cross-cutting is also mine, not a subagent's: the SQL linter (used by both sources and
the router), the tokenizer, the embedder implementations (needed by both ingest and retrieval),
the LOOKUP/AGGREGATE/HYBRID router, and the top-level facade.

---

## Layout and file-ownership map

```
anyrag/
  core/          [ORCHESTRATOR] types.py interfaces.py config.py errors.py
                                lint.py tokenizer.py registry.py
  embed/         [ORCHESTRATOR] base.py local.py hashing.py remote.py
  route/         [ORCHESTRATOR] router.py sqlgen.py          # Phase 2 hard part
  app.py         [ORCHESTRATOR] AnyRAG facade: ingest() / ask()
  sources/       [source-eng]   sqlite.py postgres.py files.py introspect.py profile.py
  ingest/        [ingest-eng]   pipeline.py row_serializer.py schema_card.py ids.py truncate.py
  index/         [index-eng]    memory.py persistent.py bm25.py filters.py
  retrieval/     [retrieval-eng] pipeline.py fusion.py rerank.py expand.py prefilter.py
  generate/      [generation-eng] packer.py prompt.py citations.py refusal.py generator.py
tests/
  test_core_*.py test_lint_*.py test_embed_*.py test_route_*.py test_app_*.py   [ORCHESTRATOR]
  test_sources_*.py [source-eng]   test_ingest_*.py [ingest-eng]
  test_index_*.py   [index-eng]    test_retrieval_*.py [retrieval-eng]
  test_generate_*.py [generation-eng]
evals/           [eval-eng]     gen_db.py questions.py metrics.py run_eval.py ablate.py
docs / root      [ORCHESTRATOR] PLAN.md DECISIONS.md BLOCKERS.md HANDOFF.md
                                ABLATIONS.md requirements.txt .env.example README.md
```

Hard rule enforced in every agent prompt: *"You may create/edit only files matching
`<your globs>`. If you believe you need a change outside them, stop and report it — do not
edit."* eval-eng additionally owns nothing under `anyrag/` — the judge cannot edit what it grades.

---

## Frozen contracts (Phase 0, `anyrag/core/`)

**DataSource** — the whole source-agnosticism claim rests on this being small:
```python
class DataSource(Protocol):
    source_id: str                                  # "sqlite:evaldb", "files:exports"
    def tables(self) -> list[TableRef]
    def schema(self, t: TableRef) -> TableSchema     # cols, types, pk, nullable, fks
    def profile(self, t: TableRef) -> TableProfile   # row count, per-col cardinality,
                                                     # null frac, samples, ColumnRole
    def iter_rows(self, t, batch=1000) -> Iterator[list[Row]]
    def execute_readonly(self, sql: str) -> QueryResult   # MUST call lint_readonly first
    def close(self) -> None
```
`ColumnRole = FREE_TEXT | CATEGORICAL | NUMERIC | DATE | ID | BOOLEAN`, inferred from type +
cardinality/length statistics. Adding a source = one class implementing this Protocol.

**Chunk** — `row_refs` is the load-bearing field: it is how the eval links a retrieved chunk
back to the gold row IDs it generated.
```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str; source_id: str; kind: ChunkKind  # ROW | SCHEMA_CARD | SQL_RESULT
    text: str
    row_refs: tuple[RowRef, ...]                    # (table, pk_value) — gold linkage
    meta: Mapping[str, Any]                         # table, columns, part_index/n_parts,
                                                    # date_min/max, lang, content_hash
```

**Stable chunk IDs**: `sha256(source_id | kind | table | pk | part_index)[:32]` — **content is
deliberately excluded**. Including content would mint a new ID on every edit and duplicate on
re-ingest; instead `content_hash` lives in `meta` for change detection. Test: ingest twice,
assert `len(index)` unchanged and no `chunk_id` collisions across distinct rows.

**Indexes**: `VectorIndex` and `LexicalIndex` both expose
`upsert(chunks) / delete_by_source(source_id) / delete_by_ids(ids) / search(q, k, filter) / count() / persist() / load()`.
Incremental upsert is part of the interface, not a rebuild in disguise.

**Retrieval config** — every stage individually toggleable, because the ablation needs it:
`RetrievalConfig(dense: bool, lexical: bool, expansion: bool, rerank: bool, rrf_k=60, k=10, prefilter=None, chunk_kinds={ROW, SCHEMA_CARD})`.
Pipeline stages: `expand → (dense ∥ lexical) → prefilter → RRF fuse → rerank → top-k`.

**Answer**: `Answer(text, citations: list[Citation], refused: bool, reason, route, trace)`.
`Citation(chunk_id, source_id, row_refs, score, quoted_span)`. An `Answer` with
`refused=False` and zero citations raises `CitationError` — enforced in the constructor, so
"no citation" is structurally impossible rather than a lint warning.

**Eval contract**: harness calls `AnyRAG.ask(question, config) -> Answer` plus
`AnyRAG.retrieve(question, config) -> list[Hit]`, and reads `chunk.row_refs` for grading.
That's the entire surface — the judge never reaches inside the library.

---

## Offline-first providers (the blocker mitigation, decided up front)

Each of these is an interface with a deterministic offline default *and* an optional remote
plugin, so nothing in the eval path can be broken by a failed download or a missing API key.

**Provenance is a first-class output, not a footnote.** Fallbacks make results *runnable*, not
*comparable*, so which provider was actually active must travel with every number:

- **Embedder (amendment 1).** Download and verify `all-MiniLM-L6-v2` as the *very first* action
  of Phase 0, before writing any code — a late discovery that the model is unavailable would
  silently reshape every retrieval number. Verification = load the model, embed two known
  strings, assert dim 384 and that cosine(similar pair) > cosine(unrelated pair). The active
  embedder id + model revision is recorded as the **first line of HANDOFF.md** and in an
  `embedder` field of **every** results JSON. If `HashingEmbedder` is active, each retrieval
  metric cell in `ABLATIONS.md` is labelled degraded **inside the table** (e.g.
  `0.412 (degraded)` plus a `⚠ degraded` column), never in surrounding prose only.
- **Generator (amendment 2).** See the eval section — the false-answer rate is provider-relative
  and must be reported as such.

| Concern | Offline default | Optional plugin |
|---|---|---|
| Embeddings | `LocalEmbedder` (sentence-transformers `all-MiniLM-L6-v2`) with automatic fallback to `HashingEmbedder` (deterministic char-ngram → projected vector) if the model won't download | `RemoteEmbedder` (env-gated) |
| Reranking | `LexicalOverlapReranker` (deterministic) | `CrossEncoderReranker` (ms-marco MiniLM) |
| NL→SQL for AGGREGATE | `HeuristicSqlGenerator` — schema-profile-driven slot filling (agg fn + target col + group-by + filter/date predicates, using value dictionaries built from `TableProfile`) | `LLMSqlGenerator` (Anthropic, env-gated) |
| Answer synthesis | `ExtractiveGenerator` — deterministic composition over packed chunks, always cites | `AnthropicGenerator` (env-gated) |

`HeuristicSqlGenerator` is built from the *schema profile*, never from the eval's question
templates — it must not peek at gold. Its coverage will be imperfect; the eval reports a
**SQL-generation coverage rate** alongside accuracy so the limitation is visible instead of
hidden. Uncovered AGGREGATE questions must **refuse**, not guess.

---

## Read-only enforcement

`anyrag/core/lint.py::lint_readonly(sql)` → raises `UnsafeQueryError` unless the statement is a
*single* `SELECT` or `WITH …SELECT`. Implementation: strip comments (`--`, `/* */`, nested),
reject multiple statements after trailing-semicolon normalization, reject a denylist of
keywords appearing outside string literals (`INSERT UPDATE DELETE DROP ALTER CREATE TRUNCATE
GRANT ATTACH PRAGMA VACUUM COPY REPLACE MERGE SET CALL DO ...`), reject `;` inside the body,
reject CTEs containing DML (`WITH x AS (DELETE … RETURNING)` — a real Postgres escape),
reject `INTO`/`SELECT … INTO`, `pg_read_file`, `lo_import`, sqlite `writefile()`.
Belt and braces: SQLite connects via `file:…?mode=ro` URI + `query_only` pragma; Postgres uses
a read-only transaction with a statement timeout. Unit tests feed ~30 malicious inputs
(stacked statements, comment-obfuscated DML, unicode/whitespace tricks, DML-in-CTE, string
literals that merely *contain* "delete" and must be **accepted**).

---

## The router (Phase 2, orchestrator-owned)

`classify(question, schema_profile) -> LOOKUP | AGGREGATE | HYBRID` using aggregate cue words
("how many", "total", "average", "per", "between <dates>"), superlatives, and whether the
question names a categorical *value* vs a free-text span. Routing:
- **LOOKUP** → hybrid retrieval → generate with citations.
- **AGGREGATE** → generate SQL → `lint_readonly` → `execute_readonly` → wrap the result set as
  a `SQL_RESULT` chunk (carrying the SQL and the contributing row refs) → that chunk is what
  gets cited. Retrieval still runs to supply schema context.
- **HYBRID** → both; SQL result and retrieved rows are both citable.
Router accuracy vs the harness's known `qtype` is itself a reported metric.

---

## Token budgeting

`core/tokenizer.py`: `tiktoken` (`cl100k_base`) with a fallback tokenizer; **never** a
chars/4 heuristic. `generate/packer.py` packs chunks under a real measured budget, dropping
lowest-scored chunks first, and truncates individual oversized fields at token boundaries.
Tests use adversarial inputs: a single 200k-char field, chunks that individually exceed the
whole budget (must yield a refusal, not a crash), and CJK/Arabic text where bytes ≠ tokens.

---

## Eval harness (`evals/`, eval-eng)

1. `gen_db.py` — seeded synthetic SQLite: ~5 related tables (`customers`, `orders`,
   `order_items`, `products`, `regions`), a few thousand rows, deliberate nulls,
   near-duplicate names ("Ahmed Al-Sayed" vs "Ahmad Al Sayed"), mixed Arabic/English text
   fields, dates spanning ~2 years. Emits `manifest.json` (schema + full row table) so gold
   answers are computed by direct pandas/SQL over the generator's own data.
2. `questions.py` — ≥300 questions across 8 types (≥6 required):
   entity-lookup-by-name, multi-attribute-filter lookup, count-by-category, numeric
   aggregate (sum/avg/min/max) by group, date-range aggregate, join/relationship question,
   schema question (targets schema-card chunks), and **distractor/unanswerable**. Each carries
   gold `row_refs` and/or a gold scalar — computed programmatically, no model involved.
3. `metrics.py` — exact recall@{1,5,10}, MRR, nDCG@10 for retrieval; exact-match / relative
   numeric tolerance for AGGREGATE; **citation precision** = fraction of cited chunks whose
   `row_refs` intersect gold; **false-answer rate** = fraction of unanswerable questions that
   got a non-refusal (the headline number); plus router accuracy and SQL coverage.
4. `run_eval.py` — one config → `evals/results/<config>.json`, each stamped with the active
   embedder and generator.
5. `ablate.py` — the full **3 × 2 × 3 = 18** grid {dense, bm25, hybrid} × {rerank on/off} ×
   {row-chunks, schema-cards, both} → `ABLATIONS.md`. The **whole grid** is reported, not the
   winner. Embeddings for chunks and queries are computed once and cached to disk
   (`evals/.cache/`) and reused across all 18 cells.

### Amendment 2 — what the false-answer rate actually measures

Over `ExtractiveGenerator` the false-answer rate measures **refusal-threshold logic, not
hallucination**, because an extractive composer cannot invent facts the way an LLM can. This
number therefore does **not** transfer to an LLM-backed config, and that sentence goes in
`HANDOFF.md` immediately next to the number — not in a caveats appendix.

If `ANTHROPIC_API_KEY` is present, additionally run **the distractor subset only** (not the
grid — 18 LLM cells is neither affordable nor necessary) through `AnthropicGenerator` at the
single best retrieval config, and report both false-answer rates side by side:
`extractive: X% | anthropic: Y% (distractor subset, n=N)`. If no key is present, HANDOFF.md says
so explicitly rather than leaving the LLM row blank or absent.

### Amendment 3 — held-out question templates

eval-eng writes **two disjoint template sets**:
- a **dev set**, which it may share with the orchestrator and which router/sqlgen development
  may look at;
- a **held-out set**, which eval-eng writes **only after `route/router.py` and `route/sqlgen.py`
  are committed and frozen** (frozen = the commit SHA is recorded in DECISIONS.md; any later
  change to those two files invalidates the held-out run and forces a re-freeze).

Router accuracy and SQL coverage are reported **separately on each set**, never pooled. A large
dev→held-out gap is not noise to average away — it is a **leakage finding** about heuristics
overfitted to templates I could see, and it is written up in HANDOFF.md as such, with both
numbers and the gap stated. This is the single most likely way for the whole exercise to fool
itself, which is why the freeze is mechanical rather than a matter of good intentions.

### Amendment 4 — ablation timebox

The grid is timeboxed to **90 minutes** from the start of `ablate.py`. If it has not finished by
then: cut to a **fixed stratified sample** of questions (stratified by question type, one seeded
sample reused identically across **all 18 cells** — an inconsistent sample would make cells
incomparable, which is worse than a smaller n), finish the **complete** grid on that sample, and
record the reduction (trigger, sample size, per-type counts, seed) in DECISIONS.md and in the
`ABLATIONS.md` header. **Grid coverage beats question count**; a partial grid is never presented
as a complete one, and every table states the n it was computed on.

---

## Execution sequence

**Phase 0 (me, no subagents)** — in this order:
0. **Install deps and download + verify `all-MiniLM-L6-v2` first** (amendment 1), before any
   other code. Record the outcome in DECISIONS.md immediately; if it fails, BLOCKERS.md gets the
   entry and the degraded-labelling path is switched on from the start rather than retrofitted.
1. `PLAN.md` (this document, amendments folded in), `DECISIONS.md`, `BLOCKERS.md`,
   `requirements.txt`, `.env.example`.
2. All of `anyrag/core/`, `anyrag/embed/`, plus stub `route/` + `app.py` so downstream agents
   have something importable. Linter + tokenizer tests green. **Commit.**

**Phase 1** — write `.claude/agents/{source,ingest,index,retrieval,generation,eval}-eng.md`
with tight, file-scoped prompts. Commit. Then:
- **Parallel batch A**: source-eng (SQLite + Postgres only), ingest-eng, index-eng —
  disjoint files, disjoint concepts.
- **Concurrent**: generation-eng and eval-eng, coding against the frozen core contracts.
  eval-eng builds the **dev** template set now and is explicitly instructed *not* to write the
  held-out set until told (amendment 3).
- **After index-eng lands**: retrieval-eng.
- **Last**: source-eng returns for `sources/files.py` (CSV/Parquet). I then record
  `git diff --stat` for that commit as the source-agnosticism proof — expected to touch only
  `sources/files.py`, `sources/__init__.py`'s registry entry, and `tests/test_sources_files.py`.
  Whatever it actually touches goes in HANDOFF.md verbatim, including failures.

**Phase 2 (me)** — real router + `HeuristicSqlGenerator`; wire `app.py`; stable-ID
double-ingest test; adversarial token-budget tests. Then **freeze `route/router.py` and
`route/sqlgen.py`**, record the freeze commit SHA in DECISIONS.md, and only then tell eval-eng
to write the held-out template set (amendment 3).

**Phase 3/4** — actually run: full test suite, then `evals/run_eval.py`, then the 18-cell
`ablate.py`, then the same pipeline against the CSV/Parquet export of the same data. Numbers in
the report come from real runs; nothing is inferred from unit tests. **Commit after each phase**,
never leave the tree broken.

---

## Verification

- `.venv/bin/pytest -q` — all unit tests, including ~30 linter attack cases, double-ingest
  ID stability, incremental upsert/delete, adversarial token budgets.
- `.venv/bin/python -m evals.gen_db --seed 7` → synthetic DB + manifest.
- `.venv/bin/python -m evals.run_eval --config hybrid_rerank_both` → metrics JSON.
- `.venv/bin/python -m evals.ablate` → `ABLATIONS.md` with all 18 rows, each stamped with
  embedder provenance, degraded labels if applicable, and the n it was computed on.
- `.venv/bin/python -m evals.run_eval --templates heldout` → router accuracy + SQL coverage on
  the held-out set, reported beside the dev-set numbers.
- `.venv/bin/python -m evals.run_eval --subset distractors --generator anthropic` → the
  side-by-side false-answer rate, when a key is present.
- `.venv/bin/python -m evals.run_eval --source files:evals/exports` → source-agnosticism run.
- Postgres: `initdb` a throwaway cluster under an unprivileged user and run
  `tests/test_sources_postgres.py`; skip-with-reason + BLOCKERS.md entry if it can't start.
- Every command above is reproduced verbatim in `HANDOFF.md` next to the number it produces.

## Risks, pre-decided

- **Model download fails** → detected in Phase 0 step 0, not late. `HashingEmbedder` keeps all 18
  cells runnable; every affected cell is labelled degraded *in the table*, and HANDOFF.md line 1
  names the active embedder.
- **Postgres won't start as root** → skipif + BLOCKERS.md; SQLite and files adapters still
  prove the interface.
- **Heuristic NL→SQL under-covers** → uncovered AGGREGATEs refuse; coverage is reported on dev
  and held-out sets separately.
- **Heuristics overfit to templates I could see** → the held-out freeze (amendment 3) is
  designed to expose exactly this; the gap gets reported, not smoothed.
- **18 cells too slow** → embedding cache is mandatory; past the 90-minute box, drop to a fixed
  stratified sample shared across all cells and finish the whole grid on it (amendment 4).

## Deliverables

`PLAN.md`, `DECISIONS.md`, `BLOCKERS.md`, the `anyrag` library, unit tests, `evals/`,
`ABLATIONS.md` (full 18-row grid, provenance-stamped), and `HANDOFF.md` — 10 bullets covering
what works, what is stubbed, what the ablations showed, the false-answer rate with its
provider caveat and LLM comparison, the dev/held-out gap, and the exact commands that reproduce
every number. HANDOFF.md's **first line** is the active embedder.
