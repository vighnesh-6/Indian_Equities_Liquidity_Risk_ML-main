# Phase 3 production feature contract

The numerical authority is `docs/phase5_feature_engineering_specification.md`, section 13, approved decisions C1-C10, the frozen feature configuration, and the audited research implementations. `production/features/contract.json` pins their exact SHA-256 identities, output order, reference artifact and model-input mappings.

## Calculations and inputs

`kernels.py` reproduces the approved numerical function bodies. An AST-equivalence test verifies every function against its frozen source. The production orchestrator avoids importing the research package entry points and their filesystem logging side effects.

Inputs are chronological observations for one stock, NIFTY 50 and INDIA VIX. Per-stock windows count observed rows. Benchmark levels use the frozen backward-only alignment rules. Relative volume retains nulls on missing or zero benchmark volume. Ratios and z-scores preserve the approved zero-denominator protections. Volatility annualization is sqrt(252), sample standard deviations use ddof=1, and Amihud's 20-row aggregates require 15 valid observations.

Calculations use the frozen float64 intermediate arithmetic and emit all 58 columns as float32. PostgreSQL uses `real`. Legitimate NaN outputs become SQL NULL and retain their source quality flags. Warm-up tagging retains the first 80 observed rows for each stock. The frozen `is_short_history` flag describes the dataset available to its calculation run, uses the approved 200-row threshold and is metadata rather than a model input. Historical flag records retain their run context as the dataset grows.

Nine quality flags are retained: `is_zero_volume`, `is_stale_price`, `is_corporate_action_gap`, `is_market_shock`, `is_short_history`, `is_warmup`, `is_benchmark_available`, `is_muhurat_session`, and `is_valid_ohlcv`. Frozen historical classifications and thresholds are preserved. Cap segments are obtained from the frozen universe dimension.

## Historical and incremental state

Backfill processes stock histories independently and publishes only dates whose observed stock cohort is complete. Daily incremental persistence includes all 300 equities in one transaction. Price-vintage changes require a new calculation version and a targeted historical rebuild. The permanent 2026-11-08 exclusion is enforced at input validation and in PostgreSQL.

Wilder RSI retains its exact initialization gains/losses, previous close and recursive float64 averages. The rolling input context retains at most 100 observations. The Amihud float64 input stream is also retained because pandas 2.2.3 rolling variance carries numerical accumulator history across earlier windows. Replaying that one cached stream reproduces the frozen historical variance calculation. This cache grows by one float64 value per observed stock row and is included in checkpoint coverage validation and storage measurement.

An initial real-data incremental check found a 0.0000513792 maximum difference in Patanjali's Amihud z-score. The numerical-history cache addresses that regression. The original acceptance tolerance remains rtol=1e-6 and atol=1e-8, with exact null masks and exact float32 RSI. Research backfill comparison requires exact float32 outputs for all 58 features.

[Pandas 2.2.3 rolling accumulator implementation](https://github.com/pandas-dev/pandas/blob/v2.2.3/pandas/_libs/window/aggregations.pyx)

## Versioned persistence

Migration 005 adds five tables: `feature_versions`, `feature_runs`, `feature_values`, `feature_checkpoints`, and `feature_publications`. The `feature_model_inputs` view exposes the 37 distinct model inputs for published dates. Model feature ordering remains recorded separately for HGBT and Isolation Forest.

The full SHA-256 calculation identity binds implementation hashes and canonical price vintage. Stable UUID relational keys reduce repeated index storage. The registry retains the full SHA-256 and rejects compact-key collisions. Input lineage includes stock, NIFTY, VIX and prior recursive-state hashes. The existing identity-checked database connection and locked migration runner are reused.

Feature observations and successful evidence are immutable. New calculation versions preserve prior versions. Uniqueness is `(ticker,date,feature_version)`. Database guards enforce observed source coverage, instrument types, run/date/version agreement, finite feature values, checkpoint continuity, state coverage and complete publication. Expected mathematical missing values remain nullable.

The market source is `liquidity.market_eod_consistent`. Original market observations and active revision chains retain their established provenance. Calculation versions identify the price basis used. Historical causal formulas evaluated on a retrospective adjusted-price vintage cannot establish complete point-in-time market-data knowledge for past years.

## Feature ordering

| ID | Frozen column |
|---|---|
| F01 | `daily_return` |
| F02 | `log_return` |
| F03 | `return_5d` |
| F04 | `return_20d` |
| F05 | `return_60d` |
| F06 | `return_z5` |
| F07 | `return_z20` |
| F08 | `abs_return` |
| F09 | `price_rel_ma20` |
| F10 | `price_rel_ma60` |
| F11 | `vol_5d` |
| F12 | `vol_20d` |
| F13 | `vol_60d` |
| F14 | `vol_ratio_5_20` |
| F15 | `parkinson_vol` |
| F16 | `true_range` |
| F17 | `atr_20d` |
| F18 | `downside_vol_20d` |
| F19 | `sortino_component` |
| F20 | `max_drawdown_20d` |
| F21 | `rsi_14` |
| F22 | `vol_of_vol` |
| F23 | `log_volume` |
| F24 | `volume_ma20` |
| F25 | `volume_ratio` |
| F26 | `volume_z20` |
| F27 | `amihud_ratio` |
| F28 | `amihud_ma20` |
| F29 | `amihud_z20` |
| F30 | `turnover_value` |
| F31 | `log_turnover` |
| F32 | `turnover_ma20` |
| F33 | `turnover_ratio` |
| F34 | `zero_return_flag` |
| F35 | `zero_vol_flag` |
| F36 | `low_volume_streak` |
| F37 | `daily_range` |
| F38 | `daily_range_ma20` |
| F39 | `range_z20` |
| F40 | `overnight_gap` |
| F41 | `overnight_gap_abs` |
| F42 | `gap_z20` |
| F43 | `cs_spread_est` |
| F44 | `cs_spread_ma10` |
| F45 | `intraday_return` |
| F46 | `excess_return_1d` |
| F47 | `excess_return_20d` |
| F48 | `rolling_beta_60d` |
| F49 | `rolling_corr_60d` |
| F50 | `idiosyncratic_vol` |
| F51 | `relative_volume` |
| F52 | `nifty_vol_20d` |
| F53 | `nifty_vol_ratio` |
| F54 | `nifty_return_20d` |
| F55 | `vol_beta` |
| Supplementary | `vix_close` |
| Supplementary | `vix_return_1d` |
| Supplementary | `vix_z20` |

## Execution and approval

The pinned dependency files reuse the complete production runtime closure, including Python 3.13.5, NumPy 2.1.3, pandas 2.2.3, PyArrow 19.0.0 and Psycopg 3.2.9. Existing research dependency declarations retain their original files.

After separate production migration approval, the exact migration runner command is:

```powershell
python -B -m production.pipeline.migrate --include-features --apply
```

After the approved schema is installed and the production calculation execution is authorized:

```powershell
python -B -m production.features.run --through 2026-11-16 --apply --report production/docs/phase3/validation/production_backfill.json
python -B -m production.features.run --through YYYY-MM-DD --incremental --apply --report production/docs/phase3/validation/production_incremental.json
```

The offline implementation gate precedes production migration application and feature backfill. Phase 4 follows completion and approval of Phase 3.
