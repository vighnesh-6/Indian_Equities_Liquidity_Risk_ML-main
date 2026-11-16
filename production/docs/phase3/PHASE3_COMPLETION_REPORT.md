# Phase 3 offline implementation report

**Status - Offline validation passed. Production migration 005 awaits separate approval.**

All implementation changes use the approved `production/features/` and `production/docs/phase3/` paths, plus the two approved production migration/ingestion files. Repository HEAD is `c8e71a06ff079d842120db55acac2c012946069f`.

## Completed implementation

- All 55 frozen core features and three supplementary India VIX features are implemented with exact approved numerical functions, column ordering, nine quality flags, float32 outputs, cap-segment metadata and warm-up tagging.
- Chronological, isolated stock processing uses the canonical production market-data view. Missing observations retain their source absence. Approved mathematical nulls persist as SQL NULL.
- Historical backfill retains all observed stock dates. Current production backfill target from the approved read-only plan is 569,690 stock rows through 16 November 2026. Historical calculations retain warm-up rows.
- Incremental calculation retains bounded 100-row rolling context, exact recursive Wilder RSI state and the Amihud float64 input history required for exact pandas variance continuation. All 300 new daily stock outputs and checkpoints persist in one transaction. Private historical blocks publish only after complete observed-date coverage.
- Version identity binds frozen implementation, configuration and canonical price vintage. Full SHA-256 identities are retained in the registry with stable compact UUID relational keys. Source, benchmark, VIX and prior-state hashes are recorded.
- Immutable output versions, source foreign keys, coverage guards, complete-session publication, checkpoint continuity, idempotency, rollback and recovery are implemented.
- A wrong 300-stock membership set is rejected even when its count remains 300. 8 November 2026 remains excluded.

## Offline validation results

| Check | Result | Evidence |
|---|---|---|
| Current Phase 3 tests | PASS | 31 unique tests across the portable suite and final integrity checks |
| Existing pipeline regressions | PASS | 85 tests |
| Prior independent-connection concurrency suite | PASS | 12 tests, including migration locks and rollback |
| Research historical parity | PASS | 300 stocks, 565,190 rows, 32,781,020 feature values, exact float32 values and identical null masks |
| Real-data incremental parity | PASS | 300 stocks, 6,000 rows, 348,000 values, zero observed differences |
| 300-stock normal entry point | PASS | Backfill, atomic daily commit, injected complete-session failure, checkpoint rollback, safe retry and NOOP |
| Causality and ticker isolation | PASS | Future stock, benchmark and VIX perturbations and mixed-ticker rejection |
| Historical database reconciliation | PASS | 36,300 rows across 21 representative stock histories, 2018-01-01 through 2026-10-23 |
| Feature migration locking | PASS | Independent connections install once and replay |
| Production migration and feature persistence | PENDING APPROVAL | Offline execution made zero production connections or writes |

The total is 128 unique tests, including three new Phase 3 independent-connection cases within its test count and the 12 prior concurrency cases. Tests ran under Python 3.13.5 and native PostgreSQL 18.7 on Windows x64. Each disposable cluster was bound to localhost and stopped after validation. The pinned production lock includes Windows and Ubuntu wheels. Hosted Phase 3 feature execution remains pending the approved deployment sequence.

Initial test failures are retained in the validation directory. Two assertions in the reused temporary Phase 2 harness expected five market rows despite two dates for each of two instruments. The Phase 3-only adapter corrects the expected count to four and records the original source hash. All locking and timeout assertions were retained. The final 12-case rerun passed.

A separate initial incremental comparison found a 0.0000513792 Amihud z-score difference for Patanjali. Pandas rolling variance retains numerical history from previous windows. Retaining its exact float64 input history resolved the regression. The prescribed comparison remained rtol=1e-6 and atol=1e-8, with exact missing-value masks and exact float32 RSI. All 300 actual stock histories ultimately showed zero incremental differences.

## Migration 005

The final SQL is `production/features/migrations/005_feature_store.sql`.

SHA-256 `4c2055a8ddafe45fbfb6aca7ed2656f14de039df467d6e61571b2ef32c0b7dc1`.

It creates five new tables (`feature_versions`, `feature_runs`, `feature_values`, `feature_checkpoints`, `feature_publications`), the `feature_model_inputs` view, two explicit lookup indexes plus primary/unique indexes, and integrity functions/triggers. The feature value table stores 58 `real` columns and nine non-null quality flags. Existing tables supply foreign-key targets and canonical source coverage. Applying the runner records migration 005 in the existing ledger.

The two approved existing-file changes add optional feature-migration discovery and retain mandatory checksum validation for every installed migration. Ingestion continues requiring 000-004, while feature operations require 005. Missing required versions, unknown versions, gaps and altered checksums fail validation. Default migration execution retains its core-only scope unless `--include-features` is specified.

The migration was applied and tested exclusively on isolated PostgreSQL databases. Schema application on production Neon is the next approval gate. Production calculation and reconciliation follow successful approved schema application.

## Storage estimate

The measured representative backfill contains 36,300 feature rows across 21 stocks. Full `feature_values` storage is projected at 262,786,353 bytes, including its indexes. The 300 checkpoint states are projected at 10,883,658 bytes. Combined with the 127,860,736-byte production baseline recorded during the read-only plan, the projected database total is **383.3 MiB** (401,923,963 bytes).

The initial text-key projection was 479.66 MiB. Compact UUID keys reduce repeated heap/index version identifiers while the registry preserves the complete SHA-256 identity. The final measurement includes the added rolling-history cache. These figures are extrapolations from real PostgreSQL heap, TOAST and index sizes. Future data, extra calculation versions, WAL, branch retention and subsequent phases require additional capacity. Production account quota and billing guarantees were outside this offline execution.

## Reproducibility and scope

`FEATURE_CONTRACT.md` documents frozen definitions, state, mathematical missing values, source-vintage limitations and execution commands. `CHANGED_FILES.json` lists every repository file created or modified. `EXECUTION_LEDGER.json` records completed and pending stages. `validation/SUMMARY.json` combines evidence from all final checks and preserves prior failed attempts.

Historical calculations on retrospective adjusted-price vintages establish formula causality, while original point-in-time provider knowledge for past years remains unavailable. The frozen `is_short_history` flag is run-scoped dataset metadata and is excluded from model inputs. Future unrecognized corporate-action classifications remain governed by the frozen registry and data-quality workflow.

## Approval gate

Requested next action is approval to apply the exact reviewed migration 005 to `liquidity-risk-production`, project `super-flower-30495255`, branch `production` / `br-winter-thunder-azx88rt3`, endpoint `ep-misty-shadow-azik30ky`, database `neondb`, schema `liquidity`.


The branch executed by the existing ingestion workflow must contain the reviewed compatibility changes and migration 005 file before 005 is applied. A collector checkout that only recognizes 000-004 will reject the extended ledger. Publishing these local changes requires separate commit/push authorization under the current instructions. Code publication therefore precedes production DDL in the approved deployment sequence.

The existing runner will re-verify identity, acquire its migration advisory lock, check original migration checksums and apply the forward SQL transactionally. After successful approved application, Phase 3 production backfill and independent reconciliation remain pending. Phase 4 awaits the Phase 3 completion gate.

Text-source fingerprints normalize CRLF to LF for Windows/Ubuntu reproducibility. Scoped Git attributes pin the new SQL and Python/JSON files to LF. Binary reference and preservation fingerprints remain raw-byte SHA-256. The same SQL statements were executed on the isolated test cluster before and after the key/storage revisions. Production DDL is pending approval.

Final portability validation passed all 30 Phase 3 tests and all 12 prior independent-connection concurrency cases. This includes actual application of LF-pinned migration SQL on the isolated PostgreSQL cluster, source-fingerprint line-ending equivalence, the full 300-equity normal entry point and the strict universe-membership guard. The existing 85 pipeline regression results remain recorded in validation/offline_tests.json.

The final migration additionally enforces exact recursive context length and its last observed date, alongside previous-close and Amihud-history coverage checks. All 15 final persistence, migration-scope and concurrency tests passed with that SQL. Combining the final portable suite and the focused integrity cases gives 31 unique Phase 3 tests. Package versions and a successful read-only pip check are recorded in validation/runtime.json.
