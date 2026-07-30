# Blockers

Things that could not be made to work, what was stubbed behind the interface
instead, and what it costs. Empty sections are load-bearing: they mean the thing
was actually verified, not skipped.

---

## Phase 0

**None.** Every dependency installed and every provider verified live:

| Dependency | Status | Evidence |
|---|---|---|
| numpy 2.4.6 | OK | imported |
| torch 2.13.0+cpu | OK | CPU wheel, no CUDA pull |
| sentence-transformers 5.6.1 | OK | model loaded and encoded |
| `all-MiniLM-L6-v2` | **OK — real model active** | dim 384; cos(paraphrase)=0.5643 > cos(unrelated)=-0.0289; Arabic encodes |
| tiktoken 0.13.0 | OK | `cl100k_base` loaded |
| psycopg 3.3.4 | OK | imported |
| pyarrow 25.0.0 | OK | imported |
| PostgreSQL 16 server | Present | `/usr/bin/psql`, `/usr/lib/postgresql/16` |

Fallback paths (`HashingEmbedder`, `RegexTokenizer`) are implemented and unit
tested, but **are not in use**. They exist so that a future environment without
network degrades result quality instead of breaking the harness.

### Known open risk, not yet a blocker
PostgreSQL refuses to run as root. The integration-test fixture must `initdb` a
cluster under an unprivileged user. If that fails, the Postgres adapter tests
will be skipped with a reason and recorded here — they will not be mocked and
reported as passing.
