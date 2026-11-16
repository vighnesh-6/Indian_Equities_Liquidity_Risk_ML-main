# Phase 2 workflow fix and operational validation - 16 November 2026

**Phase 2 is NOT operationally complete. Catch-up is paused under the user's explicit decision to keep strict overlap blocking.**

## Original failure and fix

[Failed run 43815823549](https://github.com/vighnesh-6/Indian_Equities_Liquidity_Risk_ML-main/actions/runs/43815823549) exited with code 2 before connecting to Neon. The exact error was: ConfigurationError: Exactly one nonempty Liquidity Risk database value is required.

The dedicated environment value was empty and the repository Actions secret list was empty. Python 3.13.5, installation of all 32 pinned packages and pip check had succeeded. All 11 files in the complete log archive were inspected. Metadata and hashes are recorded in phase2_results/operational_workflow_diagnosis.json. Its publication field records the state at the initial diagnosis, before the later approved push.

After separate user approval, LIQUIDITY_NEON_DATABASE_URL was encrypted using GitHub's repository public key and configured from the identity-verified local value. GitHub returned HTTP 201. The credential was neither printed nor committed. The dedicated workflow now also reports a missing/empty secret explicitly. Existing TLS verification, target isolation, dependency pins, action commit pins and restricted permissions were preserved.

After separate commit/push approval, the 19 reviewed implementation and evidence files were published to existing main in commit **54075937c3dc6f279db499f325446d5598191be3**: fix: enforce Phase 2 session exclusion and workflow preflight. No history was rewritten and no branch was changed.

## Approved session exclusion

Only **2026-11-08** is excluded by explicit user-approved production policy. This was a real NSE Muhurat session; its exchange metadata is retained separately from the production exclusion.

The shared calendar applies the exclusion to daily/catch-up planning, session selection, provider request windows, reconciliation and session checkpoint advancement. Provider ranges are split so that 8 November is never requested. The session repository rejects checkpoint advancement for that date. Reconciliation preserves any pre-existing rows without requiring new observations. No other regular session is excluded; tests retain 6 and 9 November and the published 10 November holiday.

The production database currently has no rows on the excluded date. A PostgreSQL regression separately proved preservation of a synthetic pre-existing excluded-date record while all regular sessions reconciled.

## Catch-up coverage and all checkpoint dates

| Measure | Result |
| --- | --- |
| Latest completed/publication-ready session at execution | 2026-11-13 |
| Canonical instruments | 300 equities + NIFTY 50 + INDIA VIX = 302 |
| Required new regular sessions | 14 |
| Required new observations | 4,228 |
| Provider validation | Complete for all 302 instruments and all 14 sessions |
| New observations persisted | 0 |
| New successful sessions | 0 |
| All 302 checkpoint dates | 2026-10-23 |
| Catch-up result | BLOCKED / OVERLAP_REVISION |
| Equities with checkpoint-price revisions | 39 |

Required dates: 2026-10-26, 2026-10-27, 2026-10-28, 2026-10-29, 2026-10-30, 2026-11-02, 2026-11-03, 2026-11-04, 2026-11-05, 2026-11-06, 2026-11-09, 2026-11-11, 2026-11-12, 2026-11-13.

All provider histories were fetched and validated. Before any production write, the exact-overlap guard found price differences at the existing 23 October checkpoint for 39 equities. All volumes were unchanged and OHLC prices showed approximately uniform scaling. This is consistent with adjusted-price revisions; individual corporate-action causes were not independently confirmed. Field-level evidence is in phase2_results/operational_overlap_revisions.json.

The user expressly selected **Keep strict overlap blocking; leave catch-up paused**. That guard remains unchanged. No prices were rescaled, historical records rewritten, observations fabricated or checkpoints forced forward. There will be no additional catch-up attempt without a separately approved resolution.

Every instrument's checkpoint date and revision is listed in phase2_results/operational_checkpoints.json. All 302 remain at 23 October 2026.

## Daily automation and manual GitHub validation

The collector continues to use existing identity-checked database and repository APIs, bounded retries, provenance, transactional persistence, idempotency, rollback and checkpoint recovery. An entire 302-instrument session must validate before committing. Incomplete responses and overlap conflicts exit nonzero; they are not reported as successful sessions.

The schedule remains **weekdays at 16:00 IST / 10:30 UTC**, with manual dispatch. Permissions remain contents: read. Concurrent jobs are serialized without cancelling a running job, and the dedicated secret is exposed only to the collector step.

[Manual run 43819767605](https://github.com/vighnesh-6/Indian_Equities_Liquidity_Risk_ML-main/actions/runs/43819767605) executed the approved commit on Ubuntu 24.04 with Python 3.13.5. Final GitHub status: **failure**. Collector status: **BLOCKED / OVERLAP_REVISION**.

Runtime installation, dependency checks, secret configuration and the Neon identity gate passed. The runner validated all 302 provider histories, then blocked on exactly the same 39 checkpoint revisions as the local execution. No new sessions were committed. The workflow summary and audit-artifact upload succeeded. All 11 files in the manual run's complete log archive were read and its JSON audit artifact was retrieved. Final step results, log hashes and the collector artifact are embedded in phase2_results/operational_manual_workflow.json.

## Tests and independent database verification

| Check | Result |
| --- | --- |
| Complete suite on isolated native PostgreSQL 18.7 | 76 passed, 0 failures/errors |
| Independent-connection concurrency tests | 12 passed, 0 failures/errors |
| Strengthened excluded-row preservation test on PostgreSQL 18.7 | Passed separately |
| Session suite on PGlite PostgreSQL | 13 passed |
| Pinned runtime installation and pip check on GitHub Ubuntu | PASS |
| Final independent read-only production audit after manual run | PASS |
| Historical observations matching original Parquet sources | 569,517 |
| Total database observations | 569,517 |
| Duplicate keys | 0 |
| Invalid constraints / disabled triggers / invalid indexes | 0 / 0 / 0 |
| Frozen feature columns | 37 |
| Original successful historical ingestion runs | 1 |
| New successful ingestion/checkpoint advancement | Not achieved; paused |

Only project liquidity-risk-production (super-flower-30495255), production branch (br-winter-thunder-azx88rt3), endpoint ep-misty-shadow-azik30ky, database neondb, schema liquidity was accessed. No Sector Rotation database or credentials were accessed.

## Preservation, files and remaining blocker

Changes are confined to production/pipeline/ and the dedicated workflow. Existing migrations, dependency pins, root configurations, research datasets, notebooks, research source, feature definitions, models, model artifacts, strategies, research reports and other workflows were preserved. The two pre-existing untracked processed Parquet datasets were excluded from staging and remain unchanged.

Final byte-integrity verification and exact modified/new file inventory are in phase2_results/operational_preservation.json. The final report and post-publication validation evidence remain local; they were not automatically committed or pushed.

The remaining blocker is a separately approved, data-consistent resolution of the 39 checkpoint-price revisions. Until then, 14 regular sessions and 4,228 observations remain unpersisted, all 302 checkpoints remain at 23 October, and the required successful ingestion/workflow acceptance is unmet. **Phase 2 is not declared operationally complete.**
