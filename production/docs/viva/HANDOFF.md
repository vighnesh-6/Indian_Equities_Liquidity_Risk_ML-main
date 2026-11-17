# Production intelligence handoff ? 17 November 2026

Phases 4?7 are implemented and populated. The six-page Phase 8 application passes its production build and all 14 desktop/mobile checks locally. Publication and the updated hosted workflow are explicitly approved and await actual execution; no live deployment is claimed.

## Verified production evidence

Dedicated target: liquidity-risk-production / super-flower-30495255 / production / br-winter-thunder-azx88rt3 / ep-misty-shadow-azik30ky / neondb / liquidity. No other project is used.

- Migration 006 applied using its locked transactional runner; SHA-256: `0060751009c239098695867223d31e9a9432909eb5edf495cd30ee3444e0deaf`.
- 569,690 intelligence observations; 544,026 matured HRE-5D labels; 1,500 trailing pending horizons.
- 545,690 frozen HGBT probabilities and 545,690 frozen Isolation Forest scores. Warm-up/unavailable inputs remain null.
- 2,191 complete prediction publication dates, through 2026-11-16; 25 source-verified market events.
- The original 574,047 market observations and all 302 checkpoints at 2026-11-16 remain preserved.
- Independent 826-row prediction sample matched frozen probability/raw score/scaled score at rtol=1e-7, atol=1e-9; binary decisions matched exactly.
- Four focused automated label/inference tests passed. Five isolated PostgreSQL 18.7 integration checks passed (migration idempotency, COPY/recovery, mature labels, immutable values, deferred publication rollback).
- Final production build/type checking passed; 14 Playwright desktop/mobile page and historical/filter checks passed (32.9 seconds). Visual screenshots reviewed locally; ignored browser artifacts are not publication files.
- Local connected daily invocation succeeded with existing features and inference NOOP; pending-label window was recomputed safely. This is not evidence that the updated GitHub workflow has executed.
- 372 protected baseline files are byte-identical. The only existing tracked-file change is the explicitly scoped daily workflow. No model fit, model save, research rewrite or historical feature backfill was performed.

## Operations

Use Python 3.13.5 and the existing hash-pinned `production/pipeline/requirements-collector.lock` (also referenced by the intelligence lock). Models are loaded by the existing Phase 1 interfaces and verified against the frozen hashes.

After successful market ingestion, including ingestion NOOP recovery, run:

```text
python -B -m production.intelligence.daily --report <temporary-output-path>
```

The existing workflow now invokes this automatically after ingestion. Its weekday 10:30 UTC / 16:00 IST schedule, restricted permissions, pinned Actions, complete-session requirements and permanent 2026-11-08 exclusion are preserved. A failed ingestion prevents downstream publication. Database provenance, active price-vintage checks and atomic cohort guards remain mandatory.

Web root: `production/web`. Install with `npm ci`, build with `npm run build`. The only server credential is `LIQUIDITY_NEON_DATABASE_URL`; never use a NEXT_PUBLIC-prefixed secret. For a secure local preview, use `python -B -m production.intelligence.serve_web --production --port 3003` from the repository root after building. No local credential file is created by that helper.

Pages: `/`, `/screener`, `/stock`, `/anomalies`, `/events`, `/status`. APIs: `/api/overview`, `/api/screener`, `/api/stock`, `/api/anomalies`, `/api/events`, `/api/status`, `/api/dates`. Date selection uses YYYY-MM-DD. The stock API includes historical OHLCV. All database API transactions are read-only, identity checked and use the active price vintage. Transient HTTP transport failures receive at most one retry; integrity and identity failures are never bypassed.

## Publication gate and limitations

The user explicitly approved one-file-per-commit publication to existing main, Vercel account access, dedicated project creation/linking, secure server-side production-secret transfer, deployment and manual GitHub workflow execution. This document records the pre-publication validation snapshot; hosted outcomes must be verified after publication.

Only reviewed production source, lockfiles, tests, migration, this handoff and the dedicated workflow should be staged. Exclude datasets, model binaries, `.env.local`, node_modules, `.next`, caches and browser outputs. The web-local ignore rules explicitly unignore its required `lib/` source while preserving root ignore rules.

Historical inference before 2025-10-20 is explicitly labeled TRAINING_PERIOD_REPLAY, not a strict point-in-time backtest. HRE outcomes require five future stock observations and remain pending when unavailable. Events are source-verified registry entries, with no fabricated causal impact estimates. Future-year calendar coverage requires an approved calendar update; unknown sessions are not invented.

Database size measured 513.11 MiB after population (configured server cap 1 GB). This does not establish the account billing allowance or guarantee zero cost. A one-time compaction of the new intelligence table is declined by the user; storage remains unchanged and compaction has not executed. Hosted workflow execution and the Vercel live URL remain outstanding until publication is executed and verified.
