# Decisions

Running log of judgement calls made during an unattended build. Each entry
records the choice, the alternative rejected, and why. Newest phase last.

---

## Phase 0 — design and shared contracts

### D1. Shared contracts are orchestrator-owned, not subagent-owned
`anyrag/core/` (types, interfaces, config, errors, lint, tokenizer, registry) is
written before any subagent is spawned, and no subagent may edit it.

*Alternative rejected:* letting `source-eng` own a `sources/base.py` that
everyone imports. That is the standard way "no two agents write the same file"
quietly fails — every other agent eventually needs a field added to the shared
dataclass, and they all reach for the same file.

### D2. `RowRef.pk` is always a string
Normalised in `__post_init__`. SQLite returns `7`, a CSV returns `"7"`, and
Postgres returns a `bigint`. Gold-answer matching in the eval intersects sets of
`RowRef`, so if the type leaked through, retrieval would score ~0 on the
CSV/Parquet adapter for reasons that have nothing to do with retrieval.

### D3. Chunk IDs exclude content
`sha256(source_id | kind | table | pk | part_index)`, with `content_hash` stored
separately in `meta`.

*Alternative rejected:* hashing the content into the ID. That makes IDs
"honest" but guarantees the Phase 2 requirement fails: re-ingesting edited data
would mint new IDs and duplicate rather than update. Identity should track *what
the chunk is about*, not what it currently says.

### D4. An uncited answer is unconstructible
`Answer.__post_init__` raises `CitationError` when `refused=False` and
`citations` is empty. The spec calls an uncited answer a failure rather than a
warning; the cheapest way to honour that is to make the object impossible to
build, so no code path can forget to check.

### D5. The linter fails closed on ambiguity
Three specific choices, all in the same direction:
- **Backslash is not a string escape.** SQL escapes quotes by doubling them.
  Honouring `\'` would let an attacker keep the scanner inside a "string" while
  the database had already left it. Terminating a string too early only causes
  more text to be scanned as code, which can only add rejections.
- **Dollar-quoting is rejected outright**, not parsed. No read-only SELECT we
  generate needs `$$`, and it is the standard vehicle for smuggling procedural
  code into Postgres.
- **Unterminated literals/comments are rejected**, not repaired.

### D6. Keyword scanning runs only over code segments
String literals and quoted identifiers are excised before the deny-word scan, so
`SELECT * FROM t WHERE note = 'please delete me'` is accepted while
`SELECT 1; DELETE FROM t` is not. Both directions are tested. `REPLACE` is
special-cased: `REPLACE(a,b,c)` is a legitimate scalar function, so only
`REPLACE INTO` is denied.

### D7. Adapter discovery is automatic, so a new source really is one file
`anyrag/sources/__init__.py` walks its own package with `pkgutil` and imports
every module; adapters bind to a URI scheme with `@register_source`.

This is a strengthening of the brief. The obvious implementation keeps a
`{"sqlite": SqliteSource, ...}` table, which would mean the third adapter
requires editing a shared file — and the Phase 4 proof of source-agnosticism
would be "one file plus a registry line" rather than "one file". The discovery
walk is orchestrator-owned infrastructure; the adapter modules are source-eng's.

### D8. Fallbacks are loud
Every provider (embedder, tokenizer) carries an `info` object with a `degraded`
flag, and `anyrag.embed.FALLBACK_REASON` records why auto-selection fell back.
Per amendment 1, degraded runs are labelled inside result tables, not in prose.

### D9. Token counting has a real fallback, not a heuristic
If tiktoken is unavailable, `RegexTokenizer` segments words, numbers, Arabic and
CJK runs, and punctuation. It reports itself degraded. A chars/4 heuristic was
rejected outright: it is wrong by 2-4x on exactly the Arabic and numeric-table
inputs this corpus is built from.

### D10. Embedder verified before any code was written (amendment 1)
`sentence-transformers/all-MiniLM-L6-v2` downloaded and verified as the first
action of Phase 0: dim 384, cosine(paraphrase)=0.5643 vs cosine(unrelated)=
-0.0289, Arabic encodes without error. Revision pinned to
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. **The real model is active; no
degraded-labelling path is in use.**

### D11. Postgres will be tested for real
PostgreSQL 16 is installed in this environment, so the Postgres adapter gets a
live integration test rather than a mock. It refuses to run as root, so the test
fixture initdb's a throwaway cluster under an unprivileged user. If that proves
impossible it becomes a BLOCKERS.md entry with a skip guard — but it will not be
quietly mocked and reported as passing.

### D12a. Subagents dispatched via `general-purpose` with their definition file loaded
The custom agent types in `.claude/agents/` were not resolvable by the Agent
tool: this session's agent registry was snapshotted at startup, before those
files existed. Rather than inline the prompts (which would have made
`.claude/agents/*.md` decorative), each subagent is dispatched as
`general-purpose` and instructed to read its own definition file first and treat
it as binding operating instructions. The file-ownership rules therefore still
come from the committed definitions, which is what the brief asked for.

### D12b. The router was written *before* the eval templates existed
Amendment 3 requires that router/sqlgen not be overfitted to question templates
the orchestrator could see. Rather than rely on a promise not to look, the
router and `HeuristicSqlGenerator` were written concurrently with eval-eng's
first pass — at a point when `evals/questions.py` did not yet exist on disk.
Both are derived from `TableProfile`/`TableSchema` only. The held-out set will
still be written after the freeze, so the leakage check remains meaningful; this
just removes the most obvious way for it to be defeated in advance.

### D13. `Hit.rank` is 1-based (cross-agent contract)
The top result of any `search()` has `rank == 1`; ranks are contiguous within a
call. Raised by index-eng as a decision that belongs in this log rather than in
one package's docstring, and they were right — retrieval, fusion, and the eval
harness all depend on it.

Rationale: the frozen `Hit` dataclass defaults `rank=0`. Under a 1-based
convention that default reads as "unranked" instead of silently claiming the top
position, and RRF becomes `1 / (rrf_k + rank)` with no off-by-one fudge and no
division-by-zero edge. `anyrag.index.FIRST_RANK` is exported so downstream code
asserts against it rather than hardcoding.

### D14. Tuples in `Chunk.meta` survive persistence round-trips
`MetadataFilter.equals` compares with `!=`, so a tuple silently becoming a list
after a persist/load cycle would break filtering in a way that looks like a
retrieval quality problem rather than a serialization bug. The JSONL chunk
sidecar therefore type-tags tuples. Non-JSON-able metadata raises rather than
being dropped.

### D15. BM25 is hand-written rather than taking `rank_bm25`
Verified: the library is not installed and not imported; the two mentions in
`anyrag/index/bm25.py` are docstring prose explaining the choice. The library's
API forces a full corpus rebuild on every update, which is incompatible with the
incremental-upsert requirement. Maintaining document frequencies and average
document length per-document instead means upsert and delete touch only the
affected document.

### D16. Fallback tokenizer bug: unbounded runs counted as one token
Found by ingest-eng, confirmed and fixed. `RegexTokenizer`'s `[A-Za-z]+` rule
counted a 200,000-character unbroken string as **one token** — against tiktoken's
25,000. Any purely token-based budget check would have passed it straight
through, which is precisely the failure the "measure with a real tokenizer, never
a character heuristic" requirement exists to prevent. In a degraded run
(tiktoken unavailable) `generate/packer.py` shared the exposure.

Two things were wrong, and the second is the more embarrassing:
1. The regex runs were unbounded. Now capped at 24 characters per piece, and
   long whitespace runs are flushed rather than accumulated.
2. **My own adversarial test passed trivially.** It asserted only that the count
   after truncation was *under* budget — which is satisfied perfectly by a
   tokenizer that thinks everything is one token. The test now pins the count
   itself: an unbroken run must cost roughly proportional to its length.

Whitespace gets a looser bound (512 chars/token vs 64) because real BPE
genuinely compresses it hard — cl100k encodes 100k spaces as 782 tokens. The
requirement is that it not be O(1), not that it match content density.

ingest-eng had contained the bug locally with a character backstop and flagged
that the exposure was not theirs alone. Fixing it at the tokenizer means every
consumer benefits rather than each defending separately.

### D17. Refusal overlap threshold raised 0.18 -> 0.35, before any eval was run
generation-eng found that plain token overlap does not discriminate: a
distractor asking for a nonexistent customer scores ~0.50 purely on generic
schema words ("email", "customer"), above any threshold low enough to admit real
questions. They built an idf-weighted overlap (weights computed over the
retrieved set, no corpus stats) which drops that distractor to ~0.29 while an
answerable twin stays at ~1.0.

`min_overlap` had been calibrated for the *plain* metric. Left at 0.18 against
the weighted metric, the entity-coverage gate would have been the sole
discriminator; at 0.35 the overlap gate catches the distractor by itself and
entity coverage becomes defence in depth.

**Timing matters for honesty here:** this was decided from a controlled
two-example comparison before the eval harness had produced a single number, so
it is calibration rather than tuning against the test set. The four entity-gate
knobs were also lifted into `GenerationConfig` so the ablation can sweep them.

### D18. Rejected: a `max_prompt_tokens` floor
Tempting — a budget below the ~260-token instruction scaffold can only ever
refuse. But it broke four of generation-eng's tests, and those tests were right:
"every chunk is too large, so refuse gracefully rather than crash" is behaviour
worth testing, and a constructor guard makes that test unwritable. The config
records why the guard is absent.

### D19. Composite primary keys join on `"|"` with no escaping
`pk_string` joins composite key values with `"|"`. Values containing a literal
`"|"` could in principle collide. Kept unescaped because it is a cross-agent
contract that ingest, sources, and the eval's gold answers must all reproduce
byte-for-byte, and an escaping scheme is one more thing for three
implementations to agree on. Raised independently by both source-eng and
ingest-eng. Rather than add an unused helper in `core` and hope everyone adopts
it, integration will assert that ingest-produced `RowRef`s for composite-key
tables match the eval's gold refs exactly — a test beats a convention.

### D20. `RetrievalConfig.name` no longer collapses unknown chunk-kind sets
Found by eval-eng. The name mapping handled `{ROW}` and `{SCHEMA_CARD}` and sent
everything else to `"both"`. A grid whose row cell was `{ROW, SQL_RESULT}` named
itself identically to the both-kinds cell, so **18 ablation cells silently became
12 and their results files overwrote each other**. The sweep would have reported
a complete grid that was nothing of the kind.

Names are identity here, not decoration: they key results files and table rows.
Unmapped sets now get their own composed name. The three canonical ablation
names are preserved so existing cell names do not shift.

### D21. Aggregate results are structured data, not prose
Added `Answer.value` and documented `TRACE_*` keys in `core/types.py`. eval-eng
had been recovering an aggregate's number by regex-parsing the answer text,
where "shipped 3 of 12 items" parses to 3. A scorer should never have to guess.
The trace keys are a contract for the same reason: if producer and reader
disagree on a key name, SQL coverage silently reads 0% rather than failing.

Also added `Citation.kind`, so a schema-card citation (which legitimately has no
row refs) is distinguishable from a useless one without cross-referencing the
retrieval result.

### D22. Known blind spot: date-range questions score like distractors
Found by generation-eng while validating the 0.35 threshold. A legitimate
date-range aggregate scores **0.134** weighted overlap — identical to a genuine
distractor — so *no* value of `min_overlap` separates them. The cause is
vocabulary mismatch rather than calibration: "April" never appears literally in
`order_date: 2023-04-02`, and "revenue" never appears in `total_amount`, so all
three terms carry maximum idf weight precisely because they are absent.

Deliberately **not** patched. The obvious fix (month-name to number aliasing)
was rejected without evidence: "04" as a month collides with "04" as a day, and
there was no eval yet to measure the false-accept cost. The exposure is limited
because date-range aggregates route to SQL, where an uncovered aggregate is
supposed to refuse anyway; HYBRID questions are where it bites. To be revisited
once the harness can actually score the tradeoff.

### D23. "How many orders in March 2025" is not a distractor
eval-eng's call, and the right one. Zero is the *correct* answer to that
question, so scoring it as unanswerable would count a correct "0" as a
hallucination and flatter the headline false-answer rate. Only
undefined-on-empty aggregates and enumerate-and-cite questions are used as
distractors. Enforced by a test rather than left to authorial discipline.

### D24. Fused scores are on a (0,1] scale — cross-agent contract
Raised by retrieval-eng, and the most consequential near-miss in the build.

`GenerationConfig.min_support_score = 0.02` is an absolute threshold on
`Hit.score`, but nothing in `core/` said what scale a fused score is on. Textbook
RRF gives a top score of `1/(60+1) = 0.0164` when only one retriever
contributes — **below the gate**. Every dense-only and bm25-only configuration
would therefore have refused every question.

That is 12 of the 18 ablation cells returning uniform zeros, and the failure is
worse than the number suggests: the grid would have looked like a *finding*
("single retrievers are useless, hybrid is essential") rather than a units
mismatch between two subsystems. It would have been easy to write up and
completely wrong.

Fused scores are now normalised by dividing by the maximum attainable score.
That is division by a constant, so ordering and ties are bit-identical to raw
RRF (`normalize=False` yields the textbook value and is tested). Verified on the
real MiniLM embedder across five configs.

**The contract: any component producing `Hit.score` for consumption by the
refusal gates must emit values on (0,1].** A future fusion change that ignores
this makes the entire system refuse everything, silently.

### D25. RRF is the fusion default; score fusion is implemented for the grid
Dense cosine lives in [-1,1]; BM25 is an unbounded IDF sum whose magnitude
depends on query and document length. Any score-level mix has to invent a
per-query mapping between the two. RRF discards magnitude and uses only rank, so
it needs no such invention. `normalized_score_fusion` exists so the grid can
measure the trade-off rather than the choice resting on assertion.

### D12. `SET` and `INTO` are deny-listed despite false-positive risk
A column literally named `set` or `into` would be rejected. Accepted: the
deny-words only match on word boundaries (so `offset`, `dataset_id`,
`is_deleted` are all fine), and the failure mode is a rejected query rather than
an executed write.
