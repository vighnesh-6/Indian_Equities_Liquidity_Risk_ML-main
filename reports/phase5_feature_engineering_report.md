# Phase 5: Feature Engineering Implementation & Validation Report

**Project:** ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities  
**Contract Reference:** `docs/phase5_feature_engineering_specification.md` (§11 Decisions C1–C10, §13 Mathematical Contract)  
**Primary Output:** `data/processed/features.parquet`  

---

## 1. Executive Summary

Phase 5 feature engineering has been implemented across `src/features/`, synchronized across configuration and documentation files, verified with analytical closed-form and edge-case tests in `tests/test_features.py`, and executed across the full 300-stock NSE universe.

- **Pipeline Execution Time:** `49.31` seconds (single-stock chronological streaming, 25 tickers per row group)
- **Output Artifact:** `data/processed/features.parquet` (`156.86 MB`, Snappy compression, `12` Parquet row groups)
- **Dataset Dimensions:** `565,190` rows × `70` columns (`300` tickers, `2018-01-01` to `2026-10-23`, `0` duplicate `(ticker, date)` keys)
- **Schema Breakdown:**
  - `3` metadata columns (`date`, `ticker`, `cap_segment` — mapped from `configs/universe.json`)
  - `9` retained Phase 4.2 quality flags (`is_zero_volume`, `is_stale_price`, `is_corporate_action_gap`, `is_market_shock`, `is_short_history`, `is_warmup`, `is_benchmark_available`, `is_muhurat_session`, `is_valid_ohlcv`)
  - `55` core features (`F01`–`F55`), all stored in `float32`
  - `3` supplementary India VIX features (`vix_close`, `vix_return_1d`, `vix_z20`), kept separate from `F01`–`F55` and stored in `float32`
- **Data Integrity Check:** `0` non-`float32` feature columns and `0` `inf` / `-inf` values across the entire `565,190`-row matrix.

---

## 2. Implemented Modules (`src/features/`)

| Module | Scope | Key Implementation Details |
|---|---|---|
| `src/features/_common.py` | Shared numerical & masking primitives | Float64 `safe_divide`, `rolling_zscore` (with IEEE-754 ULP zero-std floor), `compound_returns`, corporate-action transition window masking |
| `src/features/returns.py` | Category 1 (`F01`–`F10`) | Masked `daily_return`/`log_return`, compounded multi-day returns (`F03`–`F05`), rolling z-scores (`F06`–`F07`), transition-masked `price_rel_ma20`/`ma60` (`F09`–`F10`) |
| `src/features/volatility.py` | Category 2 (`F11`–`F22`) | Annualized `vol_5d`/`20d`/`60d`, `parkinson_vol`, masked `true_range`/`atr_20d`, target-0 `downside_vol_20d` (**C4**), `sortino_component` (**C4**), exact `[t-19, t]` `max_drawdown_20d` (**C10**), Wilder RSI-14 with preserved EWM state across gaps (**C4**, **C6**), `vol_of_vol` |
| `src/features/liquidity.py` | Category 3 (`F23`–`F36`) | Volume/turnover MAs & ratios, Amihud illiquidity (`1e6` scalar, `NaN` on `Volume==0` & corp gaps, `min_periods=15` for `F28`/`F29` per **C7**), `zero_return_flag`, `zero_vol_flag`, `NaN`-aware `low_volume_streak` (`F36`) |
| `src/features/gaps.py` | Category 4 (`F37`–`F45`) | Intraday range metrics, corporate-action-masked `overnight_gap`/`overnight_gap_abs`/`gap_z20`, overflow-safe `2*tanh(alpha/2)` Corwin-Schultz spread (`F43`) floored at `0.0` with 2-day pair masking (**C6**), `cs_spread_ma10`, `intraday_return` |
| `src/features/market_relative.py` | Category 5 (`F46`–`F51`) | NIFTY 50 relative returns, 60d rolling beta & correlation (clipped to `[-1, 1]`), idiosyncratic volatility (`F50`), and `relative_volume` (`F51`) sourcing `^NSEI` volume from `nifty50_clean.parquet` (**C5**) |
| `src/features/market_stress.py` | Category 6 (`F52`–`F55`) + VIX | Annualized `nifty_vol_5d`/`nifty_vol_20d` (**C3**), `nifty_vol_ratio`, `nifty_return_20d`, `vol_beta`, and optional supplementary `vix_close`/`vix_return_1d`/`vix_z20` (**C1**) |
| `src/features/engineer.py` | `FeatureEngineer` orchestrator | Per-stock causal isolation, optional `vix_path` support in `process_universe()` (**C9**), cap-segment mapping, streaming atomic Snappy Parquet output via `StreamingParquetWriter` |

---

## 3. Contract & Edge-Case Verification Results

Observed on `data/processed/features.parquet`:

- **Row & Partition Coverage:** `565,190` total rows (`24,000` `is_warmup=True`, `541,190` post-warmup `is_warmup=False`); cap-segment split: `mid = 278,971`, `large = 202,581`, `small = 83,638`.
- **Short-History Tickers (`VAML.NS`, `VOGL.NS`, `VISL.NS`):** Executed cleanly across 91 rows each, yielding `33` post-warmup rows (`11` per ticker) with valid 60d/80d features (`F05`, `F10`, `F13`, `F22`, `F48`–`F50`).
- **Corporate-Action Masking (**C6**):** All `3` confirmed corporate-action gap rows (`MOTILALOFS.NS` `2024-01-01`, `PATANJALI.NS` `2020-01-27`, `VEDL.NS` `2026-04-30`) strictly have `NaN` for `F01`, `F02`, `F08`, `F16`, `F21`, `F27`, `F34`, `F40`, `F41`, `F43`, with clean window propagation (`3` post-warmup nulls on `daily_return`, `15` on `return_5d`, `60` on `return_20d`, `57` on `price_rel_ma20` / `max_drawdown_20d`, `6` on `cs_spread_est`, `33` on `cs_spread_ma10`).
- **Market Shock Preservation (**C6**):** Both `YESBANK.NS` `is_market_shock=True` observations retain finite return, gap, volatility, and Amihud values.
- **Zero-Volume Handling (**C7**):** All `751` `is_zero_volume=True` rows have `amihud_ratio = NaN` and `zero_vol_flag = 1.0`. Setting `min_periods=15` for `amihud_ma20` (`F28`) keeps post-warmup nulls down to `103` rows (`0.0190%`).
- **Benchmark Unmatched Dates & `F51` (**C5**):** Across all `1,219` rows where `is_benchmark_available=False`, `F51 relative_volume` is `100% NaN` while market stress context (`nifty_vol_20d`) safely carries backward `.ffill()` (`996` post-20-day rows populated, `0` post-warmup nulls).
- **India VIX Supplementary Features (**C1**):** `vix_close`, `vix_return_1d`, and `vix_z20` achieve `0` post-warmup nulls across all `541,190` post-warmup rows.
- **Range Bounds:** `rsi_14` lies in `[3.0222, 99.9966]`; `rolling_corr_60d` lies in `[-0.403133, 0.939611]`; `cs_spread_est` minimum is `0.000000`.

---

## 4. Test & Configuration Synchronization Verification

- **Pytest Suite:** `pytest tests -v` executed from project root: **`61 passed in 3.58s`** (covering `test_features.py`, `test_events.py`, `test_fetcher.py`, `test_interim_market.py`, `test_lrs.py`, `test_storage.py`, `test_targets.py`, `test_universe.py`, `test_validator.py`).
- **Synchronized Configs & Documentation:**
  - `configs/feature_config.yaml`: Updated `F11` (`annualized: true`), `F28`/`F29` (`min_periods: 15`), Category 6 count `5 -> 4`, and `india_vix_features` (`enabled: true`, `status: "VERIFIED"`).
  - `configs/pipeline_config.yaml`: Added `processed_features: "data/processed/features.parquet"`.
  - `README.md`, `docs/DATA_DICTIONARY_AND_FEATURES.md`, `docs/PROJECT_STATUS.md`, `docs/ARCHITECTURE.md`, and `docs/phase5_feature_engineering_specification.md`: Updated to reflect Decision C1–C10, standardized `features.parquet`, removed stale Garman-Klass/Yang-Zhang references, and checked off Phase 5 completion.
  - Raw (`data/raw/`) and interim (`data/interim/`) datasets remain untouched and unmodified.

