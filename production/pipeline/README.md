# Production data foundation

Status on 16 November 2026 — migrations 000–002 are deployed and verified on the dedicated Liquidity Risk production branch.

## Files and responsibilities

| Path | Responsibility |
| --- | --- |
| `config.json`, `config.py` | Approved project identity, timeouts and root `.env.local` loading |
| `database.py` | TLS-verified Psycopg transactions and server identity gate |
| `repository.py` | Canonical universe registration, run plans, OHLCV upserts, provenance, batch replay and checkpoints |
| `migrate.py` | Offline migration plan and explicit transactional migration application |
| `migrations/000_bootstrap.sql` | Dedicated schema and migration checksum ledger |
| `migrations/001_foundation.sql` | Production tables, constraints and indexes |
| `migrations/002_integrity_guards.sql` | Forward-only coverage, checkpoint, scoring and plan integrity guards |
| `requirements.txt`, `requirements.lock` | Pinned production dependencies and Windows/Ubuntu distribution hashes |
| `tests/` | Mocked connection tests and isolated in-memory PostgreSQL integration tests |
| `.gitattributes` | Stable migration bytes and checksums across checkouts |

The reviewed architecture is `reports/production_architecture_final_handoff.md`. The Phase 0 findings and verified database mapping are recorded in the session. The existing Phase 1 report is `production/phase1/PHASE1_VALIDATION_REPORT.md`.

## Schema

The `liquidity` schema contains 12 tables, including its migration ledger.

- `schema_migrations` records applied versions and SHA-256 checksums.
- `instruments` holds stocks, equity indices and the volatility index.
- `stocks` retains the approved stock dimension and frozen-universe provenance.
- `ingestion_runs` records request identity, attempt status and planned range.
- `ingestion_run_instruments` records each instrument's explicit ordered trading-session plan.
- `data_provenance` records provider/version, adjustment mode, retrieval time and canonical input checksum.
- `ingestion_batches` records committed per-instrument batches and idempotency keys.
- `market_eod` stores OHLCV with instrument/date uniqueness and source lineage.
- `instrument_checkpoints` records committed coverage and optimistic revision state.
- `daily_features`, `pipeline_runs` and `daily_predictions` retain the approved downstream table roles.

Six explicit indexes support date reads, ingestion coverage/status and one successful scoring run per session. Primary keys and additional unique constraints create their associated indexes.

Migration `002` adds three supporting indexes for instrument plans, predictions by pipeline run, and pipelines by ingestion run. The schema still has 12 tables and the same 37 frozen feature columns.

The architecture's stock-only OHLCV foreign key is extended through `instruments` so benchmark rows have a valid parent. The feature table includes all 37 distinct frozen model inputs, including IF's `volume_ratio`. Floating-point storage preserves OHLCV input precision. HGBT probability and numerical scores use double precision. Scoring bands are supplied by the frozen methodology. The probability/flag constraint preserves the approved 0.534 threshold.

## Integrity review corrections (migration 002)

The original `000` and `001` SQL files and their checksums are preserved. Apply `002` only through the existing identity-checked runner after separate authorization. No migration was applied to Neon during this implementation.

- Completion checks every planned session against persisted observations, as well as each instrument's final batch receipt. A checkpoint does not substitute for historical coverage. Before advancing a batch, the repository requires earlier planned observations to exist or be included in that atomic batch.
- Checkpoint foreign keys now tie the instrument, batch and exact batch endpoint together. A trigger rejects backward movement, resets, row deletion, identity changes and invalid revision changes. Advancing requires the next revision and a newer date.
- A composite foreign key binds prediction date to scoring-run date. `pipeline_runs.finished_at` and state constraints require successful runs to have a completion time, finite nonnegative duration, a positive stock count and no error. Failed runs require a completion time and error code. Running runs cannot carry completion metrics. Terminal statuses cannot be reset for retries.
- Deferred constraint triggers require successful ingestion, exact prediction counts for successful scoring runs, exact expected-instrument membership/count, and planned-session bounds within the run. Session arrays must be nonempty, one-dimensional, finite, null-free, unique and increasing. Instrument arrays must be nonempty, one-dimensional, null-free, nonblank and unique. Plans and run identities are immutable; changed plans use a new request key.
- A stock-type column with a check and composite foreign key prevents benchmark instruments from entering the stock dimension and prevents retyping a referenced stock. Observation deletion or re-keying cannot remove coverage from successful ingestion.

Cross-table checks are deferred until commit so callers can assemble a complete plan or insert predictions and mark success in one transaction. Callers must handle commit failures and roll back. The design uses [PostgreSQL constraint triggers](https://www.postgresql.org/docs/18/sql-createtrigger.html) for these checks; ordinary [CHECK constraints](https://www.postgresql.org/docs/18/ddl-constraints.html) are limited to row-local values.

The forward migration validates existing rows and aborts atomically on invalid legacy state. It never invents completion timestamps, metrics or missing historical observations. An older database containing successful scoring runs without the newly required metrics needs separately reviewed remediation before this migration can pass. Row guards protect ordinary DML; privileged DDL or trigger disabling remains an administrative responsibility.

## Connection isolation

Credentials are loaded exclusively from `LIQUIDITY_NEON_DATABASE_URL` in the repository-root `.env.local`. Process-level database URL variables are ignored. Host, database, port and URI parameters are checked before connecting. The nonsecret project mapping is anchored to the verified target.

Every application transaction checks server-reported database, user, project, branch and endpoint before exposing a cursor. Absent or mismatched identity fails closed. TLS uses certificate and hostname verification, client certificate authentication is disabled, and password-file fallback is disabled. Prepared-statement caching is disabled for pooled connections. Errors expose a SQLSTATE or a fixed connection-failure code.

The pooled endpoint rejects the libpq startup `options` parameter. The implementation therefore uses Psycopg's transaction read-only control and transaction-local session settings. A live read-only smoke check passed against `neondb`, user `neondb_owner`, PostgreSQL 18.7, with transaction read-only on. The project, branch and endpoint identity gate passed in that same transaction.

## Persistence contract

`Repository.register_universe()` reads the protected canonical JSON directly and registers the 300 stocks and three configured market instruments by default. Passing `market_tickers=("^NSEI", "^INDIAVIX")` selects the 302 instruments required for the historical snapshot import. Registration is transactional and idempotent.

## Historical snapshot import

`historical_import.py` reads the three approved cleaned Parquet files, validates their source-observed sessions, and delegates all persistence to `Repository`. Run the offline plan first, then use `--apply` for an authorized import or `--validate-only` for an independent read-only comparison.

```powershell
python -B -m production.pipeline.historical_import --report production/pipeline/import_results/historical_import_plan.json
python -B -m production.pipeline.historical_import --apply --report production/pipeline/import_results/historical_import_execution.json
python -B -m production.pipeline.historical_import --validate-only --report production/pipeline/import_results/historical_import_validation.json
```

The manifest binds the three source-file hashes, universe hash, adjustment convention and batch size. The default batch limit is 4096 observations per instrument. Repeating the same command uses the same request and batch keys and replays committed receipts. An interrupted run stays `RUNNING` for recovery. Each uncommitted batch rolls back, and its next retry uses the original checkpoint expectation. Source conflicts or changed content produce explicit validation failures.

Provenance identifies `cleaned_parquet` and the exact Parquet SHA-256. Its `retrieved_at` records the run's stable snapshot-import timestamp. The original Yahoo retrieval timestamp is unavailable in these snapshots. The approved acquisition configuration specifies adjusted prices. Float32 source prices convert exactly to database double precision, date values retain the source trading day, zero volumes are retained, and absent turnover maps to SQL NULL. Auxiliary returns and quality columns retain their original values in the source Parquet files.

The independent validator reloads the Parquet sources and computes a SHA-256 digest of every date and OHLCV value per instrument in date order. PostgreSQL independently hashes the same lossless binary representation through its core `int8send`, `float8send` and `sha256` functions. This preserves exact float8 and int8 values and distinguishes nullable turnover from zero. It returns compact instrument profiles while reading every database observation. Its report includes per-instrument counts, bounds, value digests, null/zero-volume counts, duplicates, session plans, provenance, deterministic batch checksums and checkpoint revisions. Reports are written under `production/pipeline/`. The source plan also records coverage differences against observed NIFTY sessions, including listings with short histories and special sessions.

`start_run(request_key, mode, session_plan)` requires a mapping of ticker to sorted, unique Python `date` values. The caller must supply an approved trading-session plan. This foundation stores and enforces that plan. Trading-calendar construction and provider collection belong to the subsequent pipeline stage.

`persist_batch(...)` requires a stable batch key, explicit new sessions, the checkpoint observed during planning and timezone-aware source provenance. It validates coverage before persistence, locks the run and checkpoint, checks the stored session sequence, and writes bars, provenance, batch receipt and checkpoint in one transaction. A failure rolls back all batch effects. Overlapping earlier observations can be corrected alongside new sessions. Changes in adjustment convention are rejected.

The batch checksum covers canonical normalized bars, planned sessions, prior checkpoint and supplied provenance metadata. It describes the submitted batch. Replaying identical content returns its existing receipt. Reusing the key for changed content fails. A stale checkpoint fails, and checkpoints reference both an existing observation and a committed batch for the same instrument.

`complete_run()` requires every planned observation, each instrument's final batch receipt, and the expected instrument count. `fail_run()` accepts a bounded error code. A failed run remains an audit record. A subsequent attempt uses a new request key and the currently committed checkpoints. Completed batch receipts remain replayable.

## Validation results

The original foundation suite passed 34 tests. The integrity-guard suite passed all 55 tests (16.937 seconds on 16 November 2026), including repository bypasses through direct SQL, historical holes before a checkpoint, exact batch endpoints, backward/reset prevention, scoring dates/status/metrics/counts, instrument types, plan arrays/membership, and valid and invalid populated-database upgrades. Historical import adds seven tests to the complete 62-test suite. These tests cover exact float32 preservation, structural zero volume, invalid and duplicate source observations, source-observed session gaps, 302-instrument registration, interrupted import recovery, repeated batch replay, lossless nullable-turnover encoding, and PostgreSQL/source digest reconciliation detecting a 0.000000001 price discrepancy. Existing tests cover credential-source isolation, secret-safe failures, every identity mismatch, transaction cleanup, source registration, duplicate input, key replay, changed-content rejection, overlap updates, skipped sessions, chunked completion, stale checkpoints, adjustment conventions and full rollback after a mid-batch failure. Migration tests cover repeat application, checksum drift and rollback of all schema DDL after an injected migration error.

Integration tests use PGlite 0.5.16 with PostgreSQL 18.6 in memory. Node v26.13.0 and Python 3.13.5 were used. Tests use synthetic market observations and the existing canonical universe. Before deployment, 11 independent-connection concurrency tests and the original 55-test suite also passed against an isolated native PostgreSQL 18.7 server. The schema and statement semantics are exercised by PostgreSQL rather than a SQL translation layer.

An isolated Python environment installed all 16 pinned production packages successfully. The production specification reuses the Phase 1 runtime and adds Psycopg 3.2.9, psycopg-binary 3.2.9, certifi 2026.7.22 and typing_extensions 4.15.0. The complete Ubuntu x64 wheel set resolved and passed hash validation. Actual Ubuntu model-parity acceptance remains recorded as pending in the existing Phase 1 report.

Temporary installation and test evidence is stored under `C:\Users\vighn\AppData\Local\Temp\liquidity-foundation-l3ar7vhq`, including `install.json` and `tests-final.txt`. Final protected-file comparisons are reported in the session.

## Repeating local tests

Use an isolated Python 3.13.5 environment. From the repository root install the production lock and run dependency checks.

```powershell
python -m pip install --require-hashes --only-binary=:all: -r production/pipeline/requirements.lock
python -m pip check
```

Copy `tests/package.json` and `tests/package-lock.json` into a fresh temporary Node dependency directory. Run `npm ci --prefix <temporary-node-directory> --ignore-scripts --no-audit --no-fund`. Set `LIQUIDITY_TEST_PGLITE_MODULE` to that directory's `node_modules/@electric-sql/pglite/dist/index.js`, then run from the repository root.

```powershell
python -B -m unittest discover -s production/pipeline/tests -v
```

## Migration runner

The offline plan command only reads the SQL files and prints checksums.

```powershell
python -B -m production.pipeline.migrate
```

Versions `000_bootstrap`, `001_foundation` and `002_integrity_guards` are deployed to the verified `liquidity-risk-production` project, `production` branch. The runner skips versions already recorded with matching checksums. The explicit application entry point is shown below for an authorized migration deployment.

```powershell
python -B -m production.pipeline.migrate --apply
```

The runner checks identity, obtains a transaction-scoped migration lock, verifies recorded checksums and applies pending SQL atomically. Any schema conflict or error aborts the transaction. Historical snapshot import uses the repository entry points documented above.

## Approved production session exclusion - 16 November 2026

The user explicitly excluded **8 November 2026** (the NSE Muhurat session) from this production pipeline. This is an application policy exception, not a claim that the exchange was closed. The reviewed calendar retains the exchange special-session metadata and separately records the exclusion and approval date in `calendars/2026.json`.

`NSECalendar` applies this single exclusion to catch-up and daily plans, latest-session selection, provider request ranges and reconciliation. Yahoo request windows are split across the excluded date; it is neither fetched nor expected. `Repository.persist_session` rejects checkpoint advancement for an excluded day. Reconciliation ignores any existing rows on that date while reporting their retained count. No existing historical observations are deleted or rewritten. All other regular NSE sessions remain mandatory, including 6 and 9 November; the published 10 November holiday remains a non-session.

The failed GitHub run `43815823549` reached the collector after a successful Python 3.13.5 installation and dependency check. Its dedicated Neon environment value was empty because the repository Actions secret did not exist. After separate user approval, `LIQUIDITY_NEON_DATABASE_URL` was encrypted with GitHub's repository public key and configured as the dedicated Actions secret. It is scoped to the collector step. The local workflow also checks for a missing/empty secret explicitly; permissions, pinned dependencies, weekday 10:30 UTC schedule and manual dispatch remain unchanged. Local changes require approval before commit/push; Actions manual dispatch uses the existing remote `main` until then.

## Consistent production price basis (Phase 2 final execution)

Production consumers MUST read liquidity.market_eod_consistent for historical and incremental OHLCV. liquidity.market_eod is the immutable observation archive and must not be used as a coherent adjusted-price series. Original research Parquets, feature definitions, frozen models and historical archive rows are not rewritten.

Yahoo auto_adjust uses the current Adj Close / Close ratio. Later dividends/splits or vendor corrections can change old adjusted OHLC values. Matching only the last raw checkpoint, ignoring revisions, or multiplying new prices by a guessed factor does not solve this problem.

Migration 003 adds append-only generation receipts and actual historical provider revision rows. The collector fetches all required new regular sessions for all 302 instruments. Initial transition validates the latest 20 available archive observations per instrument against provider data and obtains an explicit dividend/split event ledger. Any overlap change or post-checkpoint corporate action requires complete provider history for that instrument. Missing historical keys on a changed basis block activation. Instruments with an exactly matching overlap and an empty post-checkpoint event ledger retain their independently hashed approved archive under explicit UNCHANGED_BASIS evidence. Their retained observations are identified as archived observations. This handles Yahoo history truncation for an unchanged instrument without inventing missing provider observations. All values use the existing float32 convention. Changed-basis fetch endpoints are re-read to detect source instability. Extra historical provider dates outside frozen coverage are not backfilled.

Only changed historical observations are appended as revision evidence. A staged generation is invisible. Its activation shares the existing first complete-session transaction, which also persists all 302 observations, provenance, receipts and checkpoints. PostgreSQL independently hashes every historical date and OHLCV value and requires 302 complete, classified basis receipts before activation can commit. Migration 004 distinguishes actual FULL_PROVIDER historical replacements from retained UNCHANGED_BASIS archives with matching checkpoint hashes and an empty corporate-action ledger. Missing or wrong hashes roll back activation and the entire session. Activated evidence is immutable. Production's canonical view resolves actual activated provider revisions while leaving original observations intact. Subsequent sessions use that same collected provider basis. Later provider revisions require another fully validated generation.

The existing session repository batches writes through Psycopg executemany while retaining advisory locks, all existing database constraints and per-instrument receipts. Deferred ingestion validation is deduplicated only until a subsequent row mutation invalidates its transaction-local marker. Direct SQL and concurrent-writer regressions verify those guarantees.

8 November 2026 remains permanently excluded from fetching, new observation requirements and canonical production coverage. Yahoo request ranges are split around that date. The original observation archive remains preserved, including any pre-existing excluded-date records.

Yahoo does not expose a contractual point-in-time snapshot/SLA. Provider completeness, endpoint stability checks, source/version/timestamps and independent value hashes are recorded rather than assuming stable historical adjustment vintages. A missing or unstable changed-basis history blocks ingestion. The request rate is bounded by a 0.5-second inter-request delay, with three attempts and the existing bounded backoff. The collector validates every required new regular session.
