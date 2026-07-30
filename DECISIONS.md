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

### D12. `SET` and `INTO` are deny-listed despite false-positive risk
A column literally named `set` or `into` would be rejected. Accepted: the
deny-words only match on word boundaries (so `offset`, `dataset_id`,
`is_deleted` are all fine), and the failure mode is a rejected query rather than
an executed write.
