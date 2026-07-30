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

### Phase 0 open risk — now RESOLVED
PostgreSQL refuses to run as root, so it was unclear whether the Postgres
adapter could be integration-tested at all.

**Resolved in Phase 1.** The test fixture successfully `initdb`s and starts a
throwaway cluster under an unprivileged user on a loopback port. **59 Postgres
tests run against a live server; none are skipped and none are mocked.**
Verified independently of the agent's report: `pytest tests/test_sources_postgres.py -q -rs`
shows 59 passed with zero skip reasons, and `ps` shows live `postgres`
processes.

The fixture still degrades honestly: if the binaries are absent or the cluster
cannot start, every test in that file skips with the specific underlying error
rather than a generic message.
