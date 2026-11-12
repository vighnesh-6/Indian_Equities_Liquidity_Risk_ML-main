# Phase 5: Feature Engineering & Benchmark Specification Audit
## ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities

**Document ID:** `PHASE-5-FEATURE-SPEC-AUDIT`  
**Status:** Specification Frozen — Decisions C1–C10 Approved (Awaiting Implementation Approval)  
**Target Package:** `src/features/` (`FeatureEngineer` and category submodules)  
**Primary Sources Audited:** `configs/feature_config.yaml`, `configs/pipeline_config.yaml`, `configs/universe.json`, `docs/DATA_DICTIONARY_AND_FEATURES.md`, `docs/DATA_SOURCE_AUDIT.md`, `docs/ML_METHODOLOGY.md`, `docs/ARCHITECTURE.md`, `docs/PROJECT_STATUS.md`, `docs/RISKS_AND_ASSUMPTIONS.md`, `reports/phase4_2_cleaning_eda.md`, `reports/data_coverage_audit.md`, `src/features/*.py`, `data/interim/` Parquet schemas, `notebooks/03_feature_exploration.ipynb`, `tests/test_features.py`.

---

## 1. Document Purpose and Scope

### 1.1 Purpose
This specification defines the complete, auditable mathematical, data-alignment, quality-masking, and architectural requirements for **Phase 5: Feature Engineering** before any implementation code is written in `src/features/`. It reconciles all existing repository documentation, configuration files, and Phase 4.2 interim datasets into a **frozen pre-implementation mathematical contract** incorporating approved Decisions **C1–C10**.

### 1.2 Scope
- **In Scope:**
  - Complete specification of the **55 core features (`F01`–`F55`)** across 6 categories.
  - Complete specification of the **3 supplementary India VIX features** (`vix_close`, `vix_return_1d`, `vix_z20`) kept separate from `F01`–`F55` (**C1**).
  - Complete audit of all **3 market index / benchmark series** (`^NSEI`, `^NSEBANK`, `^INDIAVIX`), with `^NSEBANK` explicitly excluded from `F01`–`F55` (**C8**).
  - Exact mapping to Phase 4.2 interim Parquet inputs (`data/interim/stocks/ohlcv_clean.parquet` and `data/interim/market/*_clean.parquet`).
  - Phase 4.2 quality masking rules, short-history handling, warm-up tagging, leakage prevention, and `float32` memory constraints.
  - Configuration and documentation reconciliation (**C1–C10**).
- **Out of Scope (Deferred to Later Phases):**
  - Implementation of `src/features/*.py` (pending separate Phase 5 implementation approval).
  - Composite Liquidity-Risk Score (`LRS`) and forward binary target (`HRE_5d`) computation (Phase 7: `src/labels/`).
  - Walk-forward scaling, model training, and Isolation Forest fitting (Phase 8: `src/modeling/`).
  - Event-study window analysis (Phase 9: `notebooks/09_event_analysis.ipynb`).

---

## 2. Feature Engineering Architecture

### 2.1 Processing & Memory Architecture
Per `configs/pipeline_config.yaml`, `docs/ARCHITECTURE.md`, `docs/SCALABILITY_GUIDE.md`, and **Decisions C1 & C9**, the Phase 5 feature pipeline follows a **sequential, per-stock streaming design**:

1. **Market Context Preparation (Once per Run):**
   - Load `data/interim/market/nifty50_clean.parquet` (`^NSEI`) and `data/interim/market/india_vix_clean.parquet` (`^INDIAVIX`). (`^NSEBANK` is not loaded in Phase 5 per **C8**.)
   - Compute market-level rolling benchmark series (`^NSEI` returns, annualized 5d/20d volatility, `nifty_volume_ratio`, and `^INDIAVIX` supplementary features) once on the ~2,156–2,171 row index tables.
2. **Per-Stock Sequential Processing (`process_per_stock: true`):**
   - Iterate over the 300 tickers individually (never run rolling `.rolling()` across a stacked multi-ticker DataFrame without per-ticker isolation, preventing cross-stock contamination).
   - For each ticker, sort chronologically by `date` ascending.
   - Compute Categories 1–4 (`returns.py`, `volatility.py`, `liquidity.py`, `gaps.py`) and stock-level `F55` (`vol_beta`).
   - Join / align market-relative (`market_relative.py`) and market-stress + VIX (`market_stress.py`) features by `date`.
   - Preserve all Phase 4.2 quality flags and `is_warmup` (`True` for the initial `80` trading days per stock).
   - Downcast all engineered feature columns to `float32` (`output_dtype: "float32"`).
3. **Chunked Atomic Output (**C9**):**
   - Stream per-stock feature blocks via `StreamingParquetWriter` (`src/data/storage.py`) into **`data/processed/features.parquet`** using Snappy compression and atomic `.tmp.parquet` replacement.

### 2.2 Module Responsibility Mapping

| Module | Function / Class | Owned Features | Approved Contract Notes |
|---|---|---|---|
| `src/features/engineer.py` | `FeatureEngineer.compute_stock_features`, `process_universe` | Orchestration, warm-up flag, `float32` cast, Parquet streaming to `data/processed/features.parquet` | `process_universe` accepts optional `vix_path` parameter (**C1**, **C9**) |
| `src/features/returns.py` | `compute_returns_features` | `F01`–`F10` (10 features) | Corporate-action masking applied to `F01`–`F10` (**C6**) |
| `src/features/volatility.py` | `compute_volatility_features` | `F11`–`F22` (12 features) | `F11` annualized (**C3**); `F18`/`F19`/`F21` formulas locked (**C4**); `F20` `[t-19, t]` (**C10**); Garman-Klass/Yang-Zhang excluded (**C9**) |
| `src/features/liquidity.py` | `compute_liquidity_features` | `F23`–`F36` (14 features) | `F27` masked on `Volume=0` & corp gaps (**C6**); `F28` `min_periods=15` (**C7**) |
| `src/features/gaps.py` | `compute_gap_features` | `F37`–`F45` (9 features) | `F40`–`F42` & `F43`–`F44` masked on corp-action gaps (**C6**) |
| `src/features/market_relative.py` | `compute_market_relative_features` | `F46`–`F51` (6 features) | `F51` reads `^NSEI` volume from `nifty50_clean.parquet`, `NaN` on zero/unmatched (**C5**) |
| `src/features/market_stress.py` | `compute_market_stress_features` | `F52`–`F55` (4 features) + 3 supplementary VIX features | `nifty_vol_5d` annualized (**C3**); VIX enabled outside `F01`–`F55` (**C1**) |

---

## 3. Complete F01–F55 Feature Catalogue

### 3.1 Category Distribution Summary (**C2**)

| # | Category | Config Key (`feature_config.yaml`) | Feature IDs | Approved Count | Config Alignment Note |
|---|---|---|---|---:|---|
| 1 | Returns & Price Behaviour | `returns_and_price_behaviour` | `F01`–`F10` | **10** | Matches config (`10`) |
| 2 | Volatility & Downside Risk | `volatility_and_downside_risk` | `F11`–`F22` | **12** | Matches config (`12`) |
| 3 | Liquidity & Trading Activity | `liquidity_and_trading_activity` | `F23`–`F36` | **14** | Matches config (`14`) |
| 4 | Price-Gap & Price-Impact | `price_gap_and_price_impact` | `F37`–`F45` | **9** | Matches config (`9`) |
| 5 | Market-Relative Behaviour | `market_relative_behaviour` | `F46`–`F51` | **6** | Matches config (`6`) |
| 6 | Market-Stress & Correlation | `market_stress_and_correlation` | `F52`–`F55` | **4** | Config/docs to be updated from `5` to `4` (**C2**) |
| **Total** | **Core Feature Set** | — | **`F01`–`F55`** | **55** | Supplementary India VIX (3 features) kept separate (**C1**) |

---

### 3.2 Full Feature-by-Feature Specification (`F01`–`F55`)

> **Legend:** All 55 core features have **Daily (post-market, 1 row per stock-trading day)** calculation frequency. **Scope** is either `Per-stock` (computed per ticker) or `Market-level` (computed once per trading date on `^NSEI` and broadcast onto stock rows by `date`). All 55 features have **Approved** specifications under Decisions **C1–C10**.

#### Category 1: Returns & Price Behaviour (`F01`–`F10`)

| ID | Feature Name | Category | Mathematical Formula | Required Input Columns | Rolling / Lookback Window | Freq | Scope | Benchmark Dep. | Missing-Value & Warm-Up Handling | Corp-Action & Zero-Vol Handling | Output Dtype | Financial Interpretation & Purpose | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **F01** | `daily_return` | Returns | $\frac{C_t - C_{t-1}}{C_{t-1}}$, masked to `NaN` if `is_corporate_action_gap` | `close`, `is_corporate_action_gap` | 1 day ($t-1, t$) | Daily | Per-stock | None | Row 0 ($t=0$) is `NaN`; inside 80d `is_warmup` | **Mask to `NaN` when `is_corporate_action_gap=True` (C6)**; retain `is_market_shock=True`; keep price-derived return on `is_zero_volume` | `float32` | Simple 1-day percentage return; baseline price movement | **Approved (C6)** |
| **F02** | `log_return` | Returns | $\ln\!\left(\frac{C_t}{C_{t-1}}\right)$, masked to `NaN` if `is_corporate_action_gap` | `close`, `is_corporate_action_gap` | 1 day ($t-1, t$) | Daily | Per-stock | None | Row 0 ($t=0$) is `NaN` | **Mask to `NaN` when `is_corporate_action_gap=True` (C6)**; retain `is_market_shock=True` | `float32` | Time-additive logarithmic return for volatility estimators | **Approved (C6)** |
| **F03** | `return_5d` | Returns | $\prod_{i=0}^{4}(1 + r_{t-i}) - 1$, where $r_t = \text{F01}$ | Masked `daily_return` (`F01`) | 5 days | Daily | Per-stock | None | Rows $0..4$ are `NaN` | **C6:** Compound valid masked daily returns; yields `NaN` if any day in `[t-4, t]` has a masked corporate-action return | `float32` | 1-week (5-session) price momentum | **Approved (C6)** |
| **F04** | `return_20d` | Returns | $\prod_{i=0}^{19}(1 + r_{t-i}) - 1$, where $r_t = \text{F01}$ | Masked `daily_return` (`F01`) | 20 days | Daily | Per-stock | None | Rows $0..19$ are `NaN` | **C6:** Compound valid masked daily returns; yields `NaN` if any day in `[t-19, t]` has a masked corporate-action return | `float32` | 1-month (~20-session) momentum; feeds `F19` and `F47` | **Approved (C6)** |
| **F05** | `return_60d` | Returns | $\prod_{i=0}^{59}(1 + r_{t-i}) - 1$, where $r_t = \text{F01}$ | Masked `daily_return` (`F01`) | 60 days | Daily | Per-stock | None | Rows $0..59$ are `NaN`; short-history stocks (91 rows) yield 31 values, 11 post-warmup | **C6:** Compound valid masked daily returns; yields `NaN` if any day in `[t-59, t]` has a masked corporate-action return | `float32` | 3-month medium-term momentum | **Approved (C6)** |
| **F06** | `return_z5` | Returns | $\frac{r_t - \mu_{5}(r_t)}{\sigma_{5}(r_t)}$ where $r_t = \text{F01}$ | Masked `daily_return` (`F01`) | 5-day rolling | Daily | Per-stock | None | Rows $0..5$ `NaN`; if $\sigma_5 = 0$, return `NaN` | Uses masked `F01`; $\sigma_5=0$ on flat stale segments guarded | `float32` | 5-day return standardized by recent 1-week dispersion | **Approved (C6)** |
| **F07** | `return_z20` | Returns | $\frac{r_t - \mu_{20}(r_t)}{\sigma_{20}(r_t)}$ where $r_t = \text{F01}$ | Masked `daily_return` (`F01`) | 20-day rolling | Daily | Per-stock | None | Rows $0..20$ `NaN`; if $\sigma_{20}=0$ (e.g., `FORCEMOT.NS` 41d halt), return `NaN` | Uses masked `F01`; $\sigma_{20}=0$ guarded | `float32` | 20-day standardized return; detects abnormal daily moves (Isolation Forest) | **Approved (C6)** |
| **F08** | `abs_return` | Returns | $\lvert r_t \rvert$ where $r_t = \text{F01}$ | Masked `daily_return` (`F01`) | 1 day | Daily | Per-stock | None | Row 0 is `NaN` | `NaN` on `is_corporate_action_gap=True` (**C6**) | `float32` | Unsigned return magnitude; primary numerator for Amihud (`F27`) | **Approved (C6)** |
| **F09** | `price_rel_ma20` | Returns | $\frac{C_t}{\text{SMA}_{20}(C)_t} - 1$ (`NaN` if window has corp-action transition) | `close`, `is_corporate_action_gap` | 20-day rolling | Daily | Per-stock | None | Rows $0..18$ are `NaN` (requires 20 closes) | **C6:** Mask to `NaN` whenever `[t-19, t]` (excluding row $t-19$ entry only if transition is inside `[t-18, t]` — strictly mask if any `is_corporate_action_gap` in `[t-18, t]` or `[t-19, t]`; contract uses `[t-18, t]` price transitions or full 20-row mask, see §13) | `float32` | Short-term trend extension relative to 20-day moving average | **Approved (C6)** |
| **F10** | `price_rel_ma60` | Returns | $\frac{C_t}{\text{SMA}_{60}(C)_t} - 1$ (`NaN` if window has corp-action transition) | `close`, `is_corporate_action_gap` | 60-day rolling | Daily | Per-stock | None | Rows $0..58$ are `NaN`; short-history stocks (91 rows) yield 32 values | **C6:** Mask to `NaN` whenever the 60-session price window includes an affected corporate-action transition | `float32` | Medium-term trend deviation from 60-day moving average | **Approved (C6)** |

#### Category 2: Volatility & Downside Risk (`F11`–`F22`)

| ID | Feature Name | Category | Mathematical Formula | Required Input Columns | Rolling / Lookback Window | Freq | Scope | Benchmark Dep. | Missing-Value & Warm-Up Handling | Corp-Action & Zero-Vol Handling | Output Dtype | Financial Interpretation & Purpose | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **F11** | `vol_5d` | Volatility | $\sigma_5(\text{log\_return}_t) \times \sqrt{252}$ | Masked `log_return` (`F02`) | 5-day rolling | Daily | Per-stock | None | First 5 return rows `NaN` | Uses masked `F02` (**C6**); annualized by $\sqrt{252}$ (**C3**) | `float32` | Short-term 5-day annualized realized volatility | **Approved (C3, C6)** |
| **F12** | `vol_20d` | Volatility | $\sigma_{20}(\text{log\_return}_t) \times \sqrt{252}$ | Masked `log_return` (`F02`) | 20-day rolling | Daily | Per-stock | None | First 20 return rows `NaN` | Uses masked `F02` (**C6**); stale streaks compress vol toward 0 | `float32` | Annualized 20-day realized volatility; feeds `F14`, `F22`, `F50`, `F55`, `LRS`, Isolation Forest | **Approved (C6)** |
| **F13** | `vol_60d` | Volatility | $\sigma_{60}(\text{log\_return}_t) \times \sqrt{252}$ | Masked `log_return` (`F02`) | 60-day rolling | Daily | Per-stock | None | First 60 return rows `NaN` | Uses masked `F02` (**C6**) | `float32` | Longer-term 60-day baseline annualized volatility | **Approved (C6)** |
| **F14** | `vol_ratio_5_20` | Volatility | $\frac{\text{vol\_5d}}{\text{vol\_20d}}$ | `vol_5d` (`F11`), `vol_20d` (`F12`) | 20 days (5+20) | Daily | Per-stock | None | `NaN` during first 20 returns or when `vol_20d == 0` | Both numerator & denominator annualized by $\sqrt{252}$ (**C3**); guard `vol_20d == 0` -> `NaN` | `float32` | Dimensionless volatility regime shift (`>1` = short-term spike) | **Approved (C3)** |
| **F15** | `parkinson_vol` | Volatility | $\sqrt{\frac{1}{4\ln 2}\cdot \frac{1}{20}\sum_{i=0}^{19}\left(\ln\frac{H_{t-i}}{L_{t-i}}\right)^2}\times\sqrt{252}$ | `high`, `low` | 20-day rolling | Daily | Per-stock | None | First 19 rows `NaN` | Unaffected by close-to-close corp-action gaps (`H >= L > 0` verified 100%); equals `0.0` on flat `H==L` zero-vol bars | `float32` | Range-based annualized volatility (Parkinson 1980) | **Approved** |
| **F16** | `true_range` | Volatility | $\max\!\left(H_t - L_t,\; \lvert H_t - C_{t-1}\rvert,\; \lvert L_t - C_{t-1}\rvert\right)$, masked if `is_corporate_action_gap` | `high`, `low`, `close`, `is_corporate_action_gap` | 1 day ($t-1, t$) | Daily | Per-stock | None | Row 0 ($C_{t-1}=\text{NaN}$) is `NaN` | **C6:** Mask `true_range` to `NaN` on `is_corporate_action_gap=True` to prevent artificial step-drop contamination | `float32` | Single-day price range including overnight gap (₹); ATR component | **Approved (C6)** |
| **F17** | `atr_20d` | Volatility | $\frac{1}{20}\sum_{i=0}^{19} \text{true\_range}_{t-i}$ | `true_range` (`F16`) | 20-day rolling | Daily | Per-stock | None | First 20 rows `NaN` | **C6:** Uses masked `F16` with full 20-obs window (`min_periods=20`), yielding `NaN` while a corporate-action gap sits in the window | `float32` | 20-day Average True Range (₹) | **Approved (C6)** |
| **F18** | `downside_vol_20d` | Volatility | $\sqrt{\frac{1}{20}\sum_{i=0}^{19}\min(\text{log\_return}_{t-i},\; 0)^2} \times \sqrt{252}$ | Masked `log_return` (`F02`) | 20-day rolling | Daily | Per-stock | None | First 20 return rows `NaN` (requires 20 valid `log_return`s) | **C4, C6:** Target-zero downside semi-deviation over full 20-observation window, annualized by $\sqrt{252}$; equals `0.0` if all 20 returns $\ge 0$ | `float32` | Annualized target-zero downside semi-deviation | **Approved (C4, C6)** |
| **F19** | `sortino_component` | Volatility | $\frac{\text{return\_20d}}{\text{downside\_vol\_20d}}$ | `return_20d` (`F04`), `downside_vol_20d` (`F18`) | 20 days | Daily | Per-stock | None | `NaN` during first 20 rows or when `downside_vol_20d == 0` | **C4:** Retained as `return_20d / downside_vol_20d`; explicitly documented as a Sortino component (not a conventional annualized Sortino ratio); guard `downside_vol_20d == 0` -> `NaN` | `float32` | 20-day return per unit of annualized downside semi-deviation (Sortino component) | **Approved (C4)** |
| **F20** | `max_drawdown_20d` | Volatility | $\frac{C_t - \max(C_{t-19:t})}{\max(C_{t-19:t})}$ (`NaN` if window has corp-action transition) | `close`, `is_corporate_action_gap` | **20 observations `[t-19, t]` (C10)** | Daily | Per-stock | None | First 19 rows `NaN`; always $\le 0$ | **C6, C10:** Strictly 20 observations `[t-19, t]`; masked to `NaN` if the 20-session window includes an affected corporate-action transition | `float32` | Maximum drawdown from trailing 20-session peak ($\le 0$) | **Approved (C6, C10)** |
| **F21** | `rsi_14` | Volatility | Wilder RSI: $100 - \frac{100}{1 + \text{avg\_gain}_{14}/\text{avg\_loss}_{14}}$, $\alpha = \frac{1}{14}$ | `close`, `is_corporate_action_gap` | 14-day Wilder EWM | Daily | Per-stock | None | First 14 price changes (`rows 0..13` or `0..14`) `NaN`; initialized using first 14 price changes | **C4, C6:** Mask price change $\Delta C_t$ on `is_corporate_action_gap=True`; explicit edge cases: `avg_loss==0 & avg_gain>0 -> 100.0`, `avg_gain==0 & avg_loss>0 -> 0.0`, `avg_gain==0 & avg_loss==0 -> 50.0` | `float32` | Wilder Relative Strength Index in $[0,100]$; `<30` oversold distress | **Approved (C4, C6)** |
| **F22** | `vol_of_vol` | Volatility | $\sigma_{60}(\text{vol\_20d}_t)$ | `vol_20d` (`F12`) | 60d rolling (total 80d warm-up) | Daily | Per-stock | None | First ~79–80 rows `NaN`; short-history stocks (91 rows) yield ~11–12 valid rows | Inherits masked `F12` (**C6**) | `float32` | Volatility-of-volatility; regime instability signal | **Approved (C6)** |

#### Category 3: Liquidity & Trading-Activity Proxies (`F23`–`F36`)

| ID | Feature Name | Category | Mathematical Formula | Required Input Columns | Rolling / Lookback Window | Freq | Scope | Benchmark Dep. | Missing-Value & Warm-Up Handling | Corp-Action & Zero-Vol Handling | Output Dtype | Financial Interpretation & Purpose | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **F23** | `log_volume` | Liquidity | $\ln(\text{Volume}_t + 1)$ | `volume` | 1 day | Daily | Per-stock | None | No `NaN`s (`Volume = 0` -> `0.0`) | `Volume = 0` yields `0.0` cleanly | `float32` | Log-transformed share volume; compresses 5-decade cross-sectional skew | **Approved** |
| **F24** | `volume_ma20` | Liquidity | $\frac{1}{20}\sum_{i=0}^{19}\text{Volume}_{t-i}$ | `volume` | 20-day rolling | Daily | Per-stock | None | First 19 rows `NaN` | Retains `Volume = 0` rows in rolling baseline | `float32` | 20-day average daily share volume | **Approved** |
| **F25** | `volume_ratio` | Liquidity | $\frac{\text{Volume}_t}{\text{volume\_ma20}_t}$ | `volume`, `volume_ma20` (`F24`) | 20 days | Daily | Per-stock | None | First 19 rows `NaN`; if `volume_ma20 == 0`, return `NaN` | `Volume = 0` with positive `ma20` yields `0.0`; `ma20 == 0` guarded -> `NaN` | `float32` | Relative volume; `<0.20` triggers HRE volume drought; feeds `F36`, `F51` | **Approved** |
| **F26** | `volume_z20` | Liquidity | $\frac{\text{Volume}_t - \text{volume\_ma20}_t}{\sigma_{20}(\text{Volume})_t}$ | `volume`, `volume_ma20` (`F24`) | 20-day rolling | Daily | Per-stock | None | First 19 rows `NaN`; if $\sigma_{20}=0$, return `NaN` | Guard $\sigma_{20} = 0$ on multi-week zero-vol halts | `float32` | 20-day volume z-score; abnormal activity / drought detector | **Approved** |
| **F27** | `amihud_ratio` | Liquidity | $\frac{\lvert r_t \rvert}{\text{Volume}_t \times \text{Close}_t} \times 10^6$ (`NaN` if `Volume_t == 0` or masked) | Masked `daily_return` (`F01`), `volume`, `close` | 1 day | Daily | Per-stock | None | Row 0 `NaN`; **`NaN` when `Volume_t == 0`** | **Mandatory:** `NaN` when `is_zero_volume=True` (751 rows) and when `is_corporate_action_gap=True` (**C6**); `1e6` scalar retained (validated in Phase 6 EDA) | `float32` | Amihud (2002) price impact per ₹10^6 turnover; higher = more illiquid | **Approved (C6)** |
| **F28** | `amihud_ma20` | Liquidity | $\text{mean}_{20}(\text{amihud\_ratio})$ with **`min_periods=15`** | `amihud_ratio` (`F27`) | 20-day rolling (`min_periods=15`) | Daily | Per-stock | None | **C7:** Preserves `NaN` if $<15$ valid non-`NaN` observations exist in `[t-19, t]` | Inherits `F27` masking (`Volume=0` and corp gaps excluded as `NaN`) | `float32` | Smoothed 20-day Amihud illiquidity; 25% LRS weight | **Approved (C7)** |
| **F29** | `amihud_z20` | Liquidity | $\frac{\text{amihud\_ratio}_t - \text{amihud\_ma20}_t}{\sigma_{20}(\text{amihud\_ratio})_t}$ | `amihud_ratio` (`F27`), `amihud_ma20` (`F28`) | 20-day rolling (`min_periods=15`) | Daily | Per-stock | None | `NaN` on `Volume=0`, when `<15` valid obs, or when $\sigma_{20}=0$ | Guard $\sigma_{20} = 0$ -> `NaN` | `float32` | Z-score of Amihud; detects sudden illiquidity spikes | **Approved (C7)** |
| **F30** | `turnover_value` | Liquidity | $\text{Volume}_t \times \text{Close}_t$ | `volume`, `close` | 1 day | Daily | Per-stock | None | No `NaN`s | `Volume = 0` yields `0.0` | `float32` | Daily traded value in ₹ (ADTV building block) | **Approved** |
| **F31** | `log_turnover` | Liquidity | $\ln(\text{turnover\_value}_t + 1)$ | `turnover_value` (`F30`) | 1 day | Daily | Per-stock | None | No `NaN`s | `Volume = 0` yields `0.0` | `float32` | Log-transformed rupee turnover | **Approved** |
| **F32** | `turnover_ma20` | Liquidity | $\frac{1}{20}\sum_{i=0}^{19}\text{turnover\_value}_{t-i}$ | `turnover_value` (`F30`) | 20-day rolling | Daily | Per-stock | None | First 19 rows `NaN` | Retains `Volume = 0` days (`0.0` turnover) | `float32` | 20-day baseline rupee turnover | **Approved** |
| **F33** | `turnover_ratio` | Liquidity | $\frac{\text{turnover\_value}_t}{\text{turnover\_ma20}_t}$ | `turnover_value` (`F30`), `turnover_ma20` (`F32`) | 20 days | Daily | Per-stock | None | First 19 rows `NaN`; guard `turnover_ma20 == 0` -> `NaN` | `Volume = 0` with positive `ma20` gives `0.0`; feeds LRS volume drought | `float32` | Turnover relative to 20-day average | **Approved** |
| **F34** | `zero_return_flag` | Liquidity | $\mathbb{I}(\lvert \text{daily\_return}_t \rvert < 10^{-4})$ (`NaN` if `F01` is `NaN`) | Masked `daily_return` (`F01`) | 1 day | Daily | Per-stock | None | Row 0 (`F01=NaN`) -> `NaN` | `is_corporate_action_gap` (`F01=NaN`) -> `NaN` (**C6**) | `float32` | Indicator of unchanged/stale closing price | **Approved (C6)** |
| **F35** | `zero_vol_flag` | Liquidity | $\mathbb{I}(\text{Volume}_t == 0)$ | `volume` (or `is_zero_volume`) | 1 day | Daily | Per-stock | None | No `NaN`s | Directly flags the 751 zero-volume observations as `1.0` | `float32` | Zero-volume trading halt / illiquidity indicator | **Approved** |
| **F36** | `low_volume_streak` | Liquidity | $\sum_{i=0}^{4}\mathbb{I}(\text{volume\_ratio}_{t-i} < 0.3)$ | `volume_ratio` (`F25`) | 5-day rolling (effective 24d warm-up) | Daily | Per-stock | None | First 23 rows `NaN`; if any `volume_ratio` in 5d window is `NaN`, returns `NaN` | Includes `Volume=0` (`volume_ratio=0 < 0.3`) | `float32` | Count of very-low-volume days in last 5 sessions; drought signal | **Approved** |

#### Category 4: Price-Gap & Price-Impact Proxies (`F37`–`F45`)

| ID | Feature Name | Category | Mathematical Formula | Required Input Columns | Rolling / Lookback Window | Freq | Scope | Benchmark Dep. | Missing-Value & Warm-Up Handling | Corp-Action & Zero-Vol Handling | Output Dtype | Financial Interpretation & Purpose | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **F37** | `daily_range` | Price Gaps | $\frac{H_t - L_t}{C_t}$ | `high`, `low`, `close` | 1 day | Daily | Per-stock | None | No `NaN`s ($C_t > 0$, $H_t \ge L_t$ verified 100%) | Unaffected by close-to-close corp gaps; `0.0` on locked `H==L` days | `float32` | Normalized intraday range; spread & dispersion proxy | **Approved** |
| **F38** | `daily_range_ma20` | Price Gaps | $\frac{1}{20}\sum_{i=0}^{19}\text{daily\_range}_{t-i}$ | `daily_range` (`F37`) | 20-day rolling | Daily | Per-stock | None | First 19 rows `NaN` | Continuous across zero-vol days | `float32` | 20-day baseline intraday range | **Approved** |
| **F39** | `range_z20` | Price Gaps | $\frac{\text{daily\_range}_t - \text{daily\_range\_ma20}_t}{\sigma_{20}(\text{daily\_range})_t}$ | `daily_range` (`F37`), `daily_range_ma20` (`F38`) | 20-day rolling | Daily | Per-stock | None | First 19 rows `NaN`; guard $\sigma_{20}=0$ -> `NaN` | Guard $\sigma_{20} = 0$ on locked-price halts | `float32` | Range z-score; detects abnormal intraday dispersion | **Approved** |
| **F40** | `overnight_gap` | Price Gaps | $\frac{O_t - C_{t-1}}{C_{t-1}}$, masked to `NaN` if `is_corporate_action_gap` | `open`, `close`, `is_corporate_action_gap` | 1 day ($t-1, t$) | Daily | Per-stock | None | Row 0 ($t=0$) is `NaN` | **C6:** Mask to `NaN` when `is_corporate_action_gap=True`; retain `is_market_shock=True` | `float32` | Signed overnight return discontinuity | **Approved (C6)** |
| **F41** | `overnight_gap_abs` | Price Gaps | $\lvert \text{overnight\_gap}_t \rvert$ | Masked `overnight_gap` (`F40`) | 1 day | Daily | Per-stock | None | Row 0 is `NaN` | Inherits `F40` corp-action mask (**C6**); `> 0.05` triggers HRE gap rule; 10% LRS weight | `float32` | Unsigned overnight gap shock magnitude | **Approved (C6)** |
| **F42** | `gap_z20` | Price Gaps | $\frac{\text{overnight\_gap}_t - \mu_{20}(\text{gap})}{\sigma_{20}(\text{gap})}$ | Masked `overnight_gap` (`F40`) | 20-day rolling | Daily | Per-stock | None | First 20 rows `NaN`; guard $\sigma_{20}=0$ -> `NaN` | Inherits `F40` corp-action mask (**C6**) | `float32` | Overnight gap z-score | **Approved (C6)** |
| **F43** | `cs_spread_est` | Price Gaps | Corwin & Schultz (2012) 2-day estimator floored at `0.0` (§5.2), masked if either bar $(t-1, t)$ has `is_corporate_action_gap` | `high`, `low`, `is_corporate_action_gap` | 2 days ($t-1, t$) | Daily | Per-stock | None | Row 0 is `NaN`; negative estimates floored at `0.0` | **C6:** Mask to `NaN` when either bar $t-1$ or $t$ has `is_corporate_action_gap=True` | `float32` | Effective bid-ask spread proxy in percentage terms (CS 2012) | **Approved (C6)** |
| **F44** | `cs_spread_ma10` | Price Gaps | $\frac{1}{10}\sum_{i=0}^{9}\text{cs\_spread\_est}_{t-i}$ | `cs_spread_est` (`F43`) | 10d rolling (12d config lookback) | Daily | Per-stock | None | First 10 rows `NaN` | Inherits `F43` corp-action mask (**C6**); 20% LRS weight | `float32` | Smoothed 10-day Corwin-Schultz spread estimate | **Approved (C6)** |
| **F45** | `intraday_return` | Price Gaps | $\frac{C_t - O_t}{O_t}$ | `open`, `close` | 1 day | Daily | Per-stock | None | No `NaN`s ($O_t > 0$ verified 100%) | Unaffected by prior-day close gaps; `0.0` when `O == C` | `float32` | Open-to-close intraday price pressure | **Approved** |

#### Category 5: Market-Relative Behaviour (`F46`–`F51`)

| ID | Feature Name | Category | Mathematical Formula | Required Input Columns | Rolling / Lookback Window | Freq | Scope | Benchmark Dep. | Missing-Value & Warm-Up Handling | Corp-Action & Zero-Vol Handling | Output Dtype | Financial Interpretation & Purpose | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **F46** | `excess_return_1d` | Market Relative | $r_{\text{stock}, t} - r_{\text{nifty}, t}$ | Masked `daily_return` (`F01`), `benchmark_return_fallback` | 1 day | Daily | Per-stock vs. `^NSEI` | **Yes (`^NSEI`)** | Row 0 is `NaN`; on 5 unmatched dates (`1,219` rows), `benchmark_return_fallback = 0.0` preserves continuity | Uses masked `F01` (**C6**) | `float32` | 1-day market-relative alpha | **Approved (C6)** |
| **F47** | `excess_return_20d` | Market Relative | $R_{\text{stock}, 20, t} - R_{\text{nifty}, 20, t}$ (`F04 - F54`) | `return_20d` (`F04`), `nifty_return_20d` (`F54`) | 20 days | Daily | Per-stock vs. `^NSEI` | **Yes (`^NSEI`)** | First 20 rows `NaN`; aligned on `date` | Inherits `F04` corp-action handling (**C6**) | `float32` | 20-day excess return over Nifty 50 | **Approved (C6)** |
| **F48** | `rolling_beta_60d` | Market Relative | $\frac{\text{Cov}_{60}(r_{\text{stock}}, r_{\text{nifty}})}{\text{Var}_{60}(r_{\text{nifty}})}$ | Masked `daily_return` (`F01`), `benchmark_return_fallback` | 60-day rolling | Daily | Per-stock vs. `^NSEI` | **Yes (`^NSEI`)** | First 60 rows `NaN`; short-history stocks (91 rows) yield ~31 values, 11 post-warmup | Uses masked `F01` (**C6**); guard $\text{Var}_{60}(r_{\text{nifty}}) == 0$ -> `NaN` | `float32` | 60-day systematic market beta | **Approved (C6)** |
| **F49** | `rolling_corr_60d` | Market Relative | $\text{Corr}_{60}(r_{\text{stock}}, r_{\text{nifty}})$ | Masked `daily_return` (`F01`), `benchmark_return_fallback` | 60-day rolling | Daily | Per-stock vs. `^NSEI` | **Yes (`^NSEI`)** | First 60 rows `NaN`; if $\sigma_{60}(r_{\text{stock}}) == 0$, return `NaN` | Uses masked `F01` (**C6**); clip to $[-1, 1]$ for `float32` rounding safety | `float32` | 60-day Pearson co-movement with Nifty 50 | **Approved (C6)** |
| **F50** | `idiosyncratic_vol` | Market Relative | $\text{vol\_20d}_t \times \sqrt{\max(0, 1 - \text{rolling\_corr\_60d}_t^2)}$ | `vol_20d` (`F12`), `rolling_corr_60d` (`F49`) | 60+20 days (80d config lookback) | Daily | Per-stock vs. `^NSEI` | **Yes (`^NSEI`)** | First 60–80 rows `NaN` (bounded by 80d `is_warmup`) | Inherits `F12` & `F49` masking (**C6**); 10% LRS weight | `float32` | Stock-specific (non-market) annualized volatility | **Approved (C6)** |
| **F51** | `relative_volume` | Market Relative | $\frac{\text{volume\_ratio}_t}{\text{nifty\_volume\_ratio}_t}$ | Stock `volume_ratio` (`F25`), `^NSEI` `volume` from `nifty50_clean.parquet`, `is_benchmark_available` | 20-day rolling | Daily | Per-stock vs. `^NSEI` | **Yes (`^NSEI` volume)** | **C5:** Set to `NaN` if `is_benchmark_available == False` (5 unmatched dates), or `nifty_volume_ratio` is `<= 0`, `NaN`, or invalid | **C5:** Read `^NSEI` `volume` directly from `nifty50_clean.parquet`; never fabricate benchmark volume (`27` zero-vol `^NSEI` rows guarded -> `NaN`) | `float32` | Stock trading-activity surge relative to aggregate Nifty 50 volume | **Approved (C5)** |

#### Category 6: Market-Stress & Correlation (`F52`–`F55`, **4 Features per C2**)

| ID | Feature Name | Category | Mathematical Formula | Required Input Columns | Rolling / Lookback Window | Freq | Scope | Benchmark Dep. | Missing-Value & Warm-Up Handling | Corp-Action & Zero-Vol Handling | Output Dtype | Financial Interpretation & Purpose | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **F52** | `nifty_vol_20d` | Market Stress | $\sigma_{20}\!\left(\ln\frac{\text{Nifty}_t}{\text{Nifty}_{t-1}}\right)\times\sqrt{252}$ | `^NSEI` `close` (`nifty50_clean.parquet`) | 20-day rolling | Daily | **Market-level** (broadcast by `date`) | **Yes (`^NSEI`)** | First 20 index days `NaN`; stocks listing after 2018-01-30 inherit valid market vol from day 0 | Compute on `nifty50_clean.parquet`, left-join by `date` with strictly backward `.ffill()` onto the 5 unmatched stock dates | `float32` | Broad-market 20d realized volatility regime; used by Isolation Forest | **Approved** |
| **F53** | `nifty_vol_ratio` | Market Stress | $\frac{\text{nifty\_vol\_5d}_t}{\text{nifty\_vol\_20d}_t}$, where `nifty_vol_5d` $= \sigma_5(\ln\frac{\text{Nifty}_t}{\text{Nifty}_{t-1}})\times\sqrt{252}$ | `^NSEI` `close` (`nifty50_clean.parquet`) | 5+20 days | Daily | **Market-level** (broadcast by `date`) | **Yes (`^NSEI`)** | First 20 index days `NaN`; guard `nifty_vol_20d == 0` -> `NaN` | **C3:** `nifty_vol_5d` annualized by $\sqrt{252}$ so ratio is dimensionless and centered near `1.0` | `float32` | Market volatility acceleration (`>1` = escalating stress) | **Approved (C3)** |
| **F54** | `nifty_return_20d` | Market Stress | $\frac{\text{Nifty}_t - \text{Nifty}_{t-20}}{\text{Nifty}_{t-20}}$ | `^NSEI` `close` (`nifty50_clean.parquet`) | 20-day rolling | Daily | **Market-level** (broadcast by `date`) | **Yes (`^NSEI`)** | First 20 index days `NaN` | Compute on `nifty50_clean.parquet`, left-join by `date` with backward `.ffill()` on unmatched dates | `float32` | 20-day broad-market trend context | **Approved** |
| **F55** | `vol_beta` | Market Stress | $\frac{\text{vol\_20d}_t - \text{vol\_20d}_{t-20}}{\text{vol\_20d}_{t-20}}$ | Stock `vol_20d` (`F12`) | 40d (20d vol + 20d lag) | Daily | **Per-stock** *(no `^NSEI` input)* | **No** | First 40 return rows `NaN`; guard `vol_20d_{t-20} == 0` (stale streak 20d ago) -> `NaN` | Inherits masked `F12` (**C6**) | `float32` | Stock volatility acceleration rate over trailing 20 sessions | **Approved (C6)** |

---

## 4. Benchmark Input & Benchmark-Derived Feature Audit

### 4.1 Available Benchmark & Market Index Series in `data/interim/market/`

All three series exist on disk in `data/interim/market/`, audited in `reports/data_coverage_audit.md` and `tests/test_interim_market.py`:

| Index | Ticker | Interim File | Rows | Date Span | Schema Columns | Zero-Vol Rows | Frozen Phase 5 Role |
|---|---|---|---:|---|---|---:|---|
| **NIFTY 50** | `^NSEI` | `data/interim/market/nifty50_clean.parquet` | 2,171 | `2018-01-02` to `2026-10-23` | `date`, `ticker`, `open`, `high`, `low`, `close`, `volume`, `prev_close`, `daily_return`, `is_zero_volume`, `is_extreme_move`, `is_valid_ohlcv` | 27 | **Mandatory core broad-market benchmark** (`F46`–`F54`: 9 features total) |
| **NIFTY Bank** | `^NSEBANK` | `data/interim/market/banknifty_clean.parquet` | 2,172 | `2018-01-01` to `2026-10-23` | Same 12 columns as `nifty50_clean.parquet` | 662 | **Excluded from Core `F01`–`F55`**; reserved for possible future supplementary sector analysis (**C8**) |
| **India VIX** | `^INDIAVIX` | `data/interim/market/india_vix_clean.parquet` | 2,156 | `2018-01-01` to `2026-10-23` | Same 12 columns + `daily_change` (`float32`, 13 total) | 2,156 *(structural)* | **VERIFIED & Enabled** (**C1**): produces 3 supplementary features outside `F01`–`F55` |

---

### 4.2 Detailed Role & Feature Mapping by Index

#### A. NIFTY 50 (`^NSEI`) — Core Benchmark
- **Source Columns Used:**
  - `ohlcv_clean.parquet` carries pre-joined columns: `benchmark_close`, `benchmark_return`, `nifty_close`, `nifty_return`, `is_benchmark_available`, `is_muhurat_session`, `benchmark_close_fallback`, `benchmark_return_fallback`.
  - `nifty50_clean.parquet` supplies canonical index `close` (across index trading dates) and `volume` (strictly required for `F51 relative_volume` per **C5**).
- **Benchmark-Level Metrics Derived on `^NSEI`:**
  - `nifty_log_return_1d` = $\ln(\text{NiftyClose}_t / \text{NiftyClose}_{t-1})$ (intermediate for `F52`, `F53`).
  - `nifty_vol_5d` = $\sigma_5(\text{nifty\_log\_return\_1d}) \times \sqrt{252}$ (**C3**).
  - **F52 `nifty_vol_20d`** = $\sigma_{20}(\text{nifty\_log\_return\_1d}) \times \sqrt{252}$.
  - **F53 `nifty_vol_ratio`** = `nifty_vol_5d / nifty_vol_20d` (**C3**).
  - **F54 `nifty_return_20d`** = $(\text{NiftyClose}_t - \text{NiftyClose}_{t-20}) / \text{NiftyClose}_{t-20}$.
  - `nifty_volume_ma20` = 20-day rolling mean of `^NSEI` `volume`; `nifty_volume_ratio` = `volume / nifty_volume_ma20` (set to `NaN` when `volume == 0` or `ma20 <= 0`, feeding **F51** per **C5**).
- **Relative-Strength & Correlation Features (`F46`–`F51`):**
  - Relative returns: `F46 excess_return_1d`, `F47 excess_return_20d`.
  - Sensitivity & correlation: `F48 rolling_beta_60d`, `F49 rolling_corr_60d`, `F50 idiosyncratic_vol` (uses `F49` and `F12`).
  - Relative activity: `F51 relative_volume` (stock `volume_ratio` / `nifty_volume_ratio`; `NaN` on zero/invalid/unmatched benchmark volume per **C5**).
- **Alignment & Fallback Methodology:**
  - `ohlcv_clean.parquet` has **5 stock-trading dates** (`2018-01-01`, `2019-01-01`, `2019-10-27` Muhurat, `2020-11-14` Muhurat, `2026-01-15` — **1,219 stock-day rows total**) absent from `^NSEI`.
  - Compute `F52`, `F53`, `F54`, and `nifty_volume_ratio` on `nifty50_clean.parquet` first, then left-join by `date` onto each stock.
  - For market **level/regime** features (`F52`, `F53`, `F54`), apply strictly backward `.ffill()` on unmatched dates (no lookahead). For `F51`, **do not forward-fill or fabricate volume**: set `F51 = NaN` on unmatched dates (`is_benchmark_available == False`) and whenever `^NSEI` `volume == 0` or `nifty_volume_ratio <= 0` (**C5**).
  - For daily returns on unmatched dates (`F46`, `F48`, `F49`), use `benchmark_return_fallback` (`0.0` after row 0).

#### B. NIFTY Bank (`^NSEBANK`) — Reserved Sector Benchmark (**C8**)
- **Verified Availability:** `banknifty_clean.parquet` exists (`2,172` rows, `2018-01-01` to `2026-10-23`, 0 nulls, 662 zero-volume rows).
- **Approved Disposition (**C8**):** **Excluded from Core `F01`–`F55` and not used in Phase 5.** Reserved unmodified in `data/interim/market/` for possible future supplementary sector analysis.

#### C. India VIX (`^INDIAVIX`) — Supplementary Market-Fear Features (**C1**)
- **Verified Availability & Status (**C1**):** `india_vix_clean.parquet` (`2,156` rows, `2018-01-01` to `2026-10-23`, 0 nulls) is **VERIFIED** and **ENABLED**.
- **Supplementary Benchmark-Derived Features (3 Features, Outside `F01`–`F55`):**

| Feature Name | Mathematical Formula | Source Columns (`india_vix_clean.parquet`) | Rolling / Lookback | Scope | Alignment & Missing-Data Treatment | Output Dtype | Status |
|---|---|---|---|---|---|---|---|
| `vix_close` | $\text{VIX}_t$ | `close` | 1 day | Market-level (broadcast by `date`) | Left-join by `date`; strictly backward `.ffill()` on active equity dates absent from VIX feed (no `bfill`) | `float32` | **Approved (C1)** |
| `vix_return_1d` | $\frac{\text{VIX}_t - \text{VIX}_{t-1}}{\text{VIX}_{t-1}}$ | `close` (or precomputed `daily_return`) | 1 day ($t-1, t$) | Market-level (broadcast by `date`) | Row 0 of VIX series is `NaN`; assign `0.0` on unmatched equity-only dates after row 0 (consistent with `benchmark_return_fallback`) | `float32` | **Approved (C1)** |
| `vix_z20` | $\frac{\text{VIX}_t - \text{SMA}_{20}(\text{VIX})_t}{\sigma_{20}(\text{VIX})_t}$ | `close` | 20-day rolling | Market-level (broadcast by `date`) | Compute on `india_vix_clean.parquet` first, left-join by `date` with strictly backward `.ffill()` on unmatched dates; guard $\sigma_{20}=0 \to \text{NaN}$ | `float32` | **Approved (C1)** |

---

## 5. Mathematical Definitions and Notation

### 5.1 General Notation
- Let $t \in \{0, 1, \dots, T-1\}$ index chronologically sorted trading days for a single stock ($T \in [91, 2176]$).
- Raw OHLCV at day $t$: $O_t = \text{open}_t$, $H_t = \text{high}_t$, $L_t = \text{low}_t$, $C_t = \text{close}_t$, $V_t = \text{volume}_t$.
- Rolling window of length $W$ ending at $t$: $[t - W + 1, t]$ (strictly backward-looking, inclusive of $t$).
- Sample standard deviation $\sigma_W(x)_t$: rolling sample std (`ddof=1`) over window $W$.
- Annualization factor: $\sqrt{252}$ trading days/year.

### 5.2 Key Microstructure Estimator Equations (Frozen)

1. **Amihud (2002) Illiquidity Ratio (`F27`) & Moving Average (`F28`, **C7**):**
   $$\text{ILLIQ}_t = \begin{cases} \dfrac{\lvert r_t \rvert}{V_t \cdot C_t} \times 10^6, & \text{if } V_t > 0 \text{ and not masked} \\ \text{NaN}, & \text{if } V_t = 0 \text{ or } \text{is\_corporate\_action\_gap}_t = \text{True} \end{cases}$$
   $$\text{amihud\_ma20}_t = \text{mean}\!\left(\{\text{ILLIQ}_{t-19}, \dots, \text{ILLIQ}_t\}\right) \quad \text{if valid count} \ge 15, \text{ else NaN}$$

2. **Parkinson (1980) High-Low Volatility (`F15`):**
   $$\sigma_{\text{Parkinson}, 20, t} = \sqrt{\frac{1}{4 \ln 2} \cdot \frac{1}{20} \sum_{i=0}^{19} \left(\ln \frac{H_{t-i}}{L_{t-i}}\right)^2} \times \sqrt{252}$$

3. **Target-Zero Downside Semi-Deviation (`F18`, **C4**) & Sortino Component (`F19`, **C4**):**
   $$\sigma_{\text{down}, 20, t} = \sqrt{\frac{1}{20} \sum_{i=0}^{19} \min\!\left(\text{log\_return}_{t-i},\; 0\right)^2} \times \sqrt{252}$$
   $$\text{sortino\_component}_t = \begin{cases} \dfrac{\text{return\_20d}_t}{\sigma_{\text{down}, 20, t}}, & \text{if } \sigma_{\text{down}, 20, t} > 0 \\ \text{NaN}, & \text{otherwise} \end{cases}$$

4. **Wilder RSI-14 (`F21`, **C4**, **C6**):**
   Let $\Delta C_t = C_t - C_{t-1}$ (masked to `NaN` on `is_corporate_action_gap = True`). With $G_t = \max(\Delta C_t, 0)$ and $L_t = \max(-\Delta C_t, 0)$, compute smoothed averages with $\alpha = \frac{1}{14}$ initialized as the arithmetic mean of the first 14 valid price changes (`ewm(alpha=1/14, adjust=False)` after 14-change initialization):
   $$\text{RSI}_{14, t} = \begin{cases} 100.0, & \text{if } \overline{L}_{14,t} = 0 \text{ and } \overline{G}_{14,t} > 0 \\ 0.0, & \text{if } \overline{G}_{14,t} = 0 \text{ and } \overline{L}_{14,t} > 0 \\ 50.0, & \text{if } \overline{G}_{14,t} = 0 \text{ and } \overline{L}_{14,t} = 0 \\ 100 - \dfrac{100}{1 + \overline{G}_{14,t} / \overline{L}_{14,t}}, & \text{otherwise} \end{cases}$$

5. **Corwin-Schultz (2012) High-Low Spread Estimator (`F43`, **C6**):**
   For consecutive bars $(t-1, t)$ where neither bar has `is_corporate_action_gap = True`:
   $$\beta_t = \left(\ln \frac{H_{t-1}}{L_{t-1}}\right)^2 + \left(\ln \frac{H_t}{L_t}\right)^2, \qquad \gamma_t = \left(\ln \frac{\max(H_{t-1}, H_t)}{\min(L_{t-1}, L_t)}\right)^2$$
   $$\alpha_t = \frac{\sqrt{2\beta_t} - \sqrt{\beta_t}}{3 - 2\sqrt{2}} - \sqrt{\frac{\gamma_t}{3 - 2\sqrt{2}}}, \qquad S_t = \max\!\left(0.0,\; \frac{2\left(e^{\alpha_t} - 1\right)}{1 + e^{\alpha_t}}\right)$$

6. **Idiosyncratic Volatility (`F50`):**
   $$\sigma_{\text{idio}, t} = \text{vol\_20d}_t \cdot \sqrt{\max\!\left(0.0,\; 1 - \rho_{60, t}^2\right)}$$

---

## 6. Input Dataset and Column Mapping

### 6.1 Primary Stock Input: `data/interim/stocks/ohlcv_clean.parquet`
Verified PyArrow schema (`565,190` rows, `27` columns, `300` tickers, `2018-01-01` to `2026-10-23`):

| Column Group | Exact Columns & Physical Types | Usage in Phase 5 |
|---|---|---|
| **Keys** | `date` (`timestamp[ns]`), `ticker` (`string`) | Grouping, chronological sort, output primary key |
| **Core OHLCV** | `open`, `high`, `low`, `close` (`float`), `volume` (`int64`) | Inputs to `F01`–`F45`, `F55` |
| **Precomputed Return** | `prev_close` (`float`), `daily_return` (`float`) | Baseline for `F01` (masked when `is_corporate_action_gap=True`) |
| **Quality / Regime Flags** | `is_zero_volume`, `is_zero_volume_nonzero_return`, `is_stale_price`, `is_extreme_move`, `is_corporate_action_gap`, `is_market_shock`, `is_short_history`, `is_warmup`, `is_valid_ohlcv` (`bool`), `stale_price_streak` (`int32`) | Corporate-action masking, zero-volume protection, warm-up preservation, short-history safety |
| **Pre-Joined `^NSEI`** | `benchmark_close`, `benchmark_return`, `nifty_close`, `nifty_return`, `benchmark_close_fallback`, `benchmark_return_fallback` (`float`), `is_benchmark_available`, `is_muhurat_session` (`bool`) | Market-relative returns (`F46`–`F49`) and unmatched-date detection (`F51`) |

*Note:* `cap_segment` (`large`/`mid`/`small`) is mapped from `configs/universe.json` via `UniverseManager.get_segment_mapping()` if included as a metadata column in `data/processed/features.parquet`.

### 6.2 Market Index Inputs (`data/interim/market/`)
- **`nifty50_clean.parquet`** (`2,171` rows, `12` cols): Supplies canonical `close`, `daily_return` (`F52`–`F54`), and `volume` (strictly required for `F51`, **C5**).
- **`india_vix_clean.parquet`** (`2,156` rows, `13` cols): Supplies `close`, `daily_return`, `daily_change` for the 3 enabled supplementary VIX features (**C1**).
- **`banknifty_clean.parquet`** (`2,172` rows, `12` cols): **Not consumed** in Phase 5 (**C8**).

---

## 7. Rolling-Window and Alignment Methodology

1. **Per-Stock Isolation:** Every stock series is sliced, sorted by `date` ascending, and passed independently to feature functions so rolling windows never bleed across ticker boundaries.
2. **Row-Index Windows:** Rolling lookbacks ($5, 10, 14, 20, 40, 60, 80$) operate on **consecutive observed trading rows** for that ticker, not calendar days.
3. **20-Observation Window Exactness (**C10**):** All 20-day rolling windows (including `F20 max_drawdown_20d`) use `[t-19, t]` (exactly 20 observations).
4. **Market-to-Stock Alignment (`^NSEI`, `^INDIAVIX`):**
   - Compute `F52`–`F54`, `nifty_volume_ratio`, and `vix_close` / `vix_return_1d` / `vix_z20` on their respective index series first, then left-join onto each stock by `date`.
   - On unmatched stock-only dates (`5` dates for `^NSEI`, ~`20` dates for `^INDIAVIX`), market **level/regime** features (`F52`–`F54`, `vix_close`, `vix_z20`) use strictly backward `.ffill()` (no lookahead); market **daily returns** use `0.0` (`benchmark_return_fallback`); **`F51` `relative_volume` strictly returns `NaN`** (**C5**).

---

## 8. Data Quality and Masking Rules

| Phase 4.2 Flag / Constraint | Observed Count | Features Affected | Frozen Phase 5 Rule |
|---|---|---|---|
| **`is_corporate_action_gap`** | **3 rows** (`MOTILALOFS.NS` `2024-01-01`, `PATANJALI.NS` `2020-01-27`, `VEDL.NS` `2026-04-30`) | `F01`–`F14`, `F16`–`F22`, `F27`–`F29`, `F34`, `F40`–`F44`, `F46`–`F50`, `F55` | **C6:** Mask `F01`, `F02`, `F08`, `F16`, `F27`, `F40`, `F41` on flagged rows; compound `F03`–`F05` from masked returns (yielding `NaN` if any day in window is masked); mask `F09`, `F10`, `F17`, `F20` whenever their window spans an affected transition; mask `F43` if either bar $t-1$ or $t$ is flagged; mask $\Delta C_t$ in `F21`. |
| **`is_market_shock`** | **2 rows** (`YESBANK.NS` `2020-03-06` & `2020-03-17`) | All return, vol, gap, and liquidity features | **C6:** **RETAIN unchanged.** Genuine liquidity crises are primary tail-risk signals. |
| **`is_zero_volume`** | **751 rows** (incl. 6 with `is_zero_volume_nonzero_return=True`) | `F27`–`F29` (`amihud_*`), `F23`–`F26`, `F30`–`F33`, `F35`–`F36` | **`F27` MUST be `NaN` when `Volume == 0`.** Retain rows (`F35 = 1.0`, `F23 = 0.0`, `F30 = 0.0`). **`F28` uses `min_periods=15` (C7).** |
| **`is_stale_price`** | **330 rows** across 44 tickers (`stale_price_streak >= 3`, incl. `FORCEMOT.NS` 41d halt) | `F06`–`F07`, `F11`–`F14`, `F18`–`F19`, `F25`–`F26`, `F29`, `F39`, `F42`, `F49`–`F50`, `F55` | Guard zero std / zero denominator in all z-scores, ratios, and correlations so flat segments yield `NaN` rather than `inf`/`-inf`. |
| **`benchmark_close_fallback` / `benchmark_return_fallback`** | **1,219 unmatched stock-days** (5 dates) | `F46`–`F54` | Use fallback columns for return/level continuity (`F46`–`F50`, `F52`–`F54`); **strictly set `F51 = NaN`** when `is_benchmark_available == False` or `^NSEI` `volume <= 0` (**C5**). |
| **`is_warmup`** | **24,000 rows** (80 rows × 300 stocks) | All `F01`–`F55` + VIX | Retain all rows in `data/processed/features.parquet` with `is_warmup` flag preserved; downstream training excludes `is_warmup == True`. |
| **Short-history tickers** | **273 rows** (`VAML.NS`, `VOGL.NS`, `VISL.NS`, 91 rows each) | `F05`, `F10`, `F13`, `F22`, `F48`–`F50` | Execute cleanly on 91-row series (~11 post-warmup rows per ticker) and retain `is_short_history = True`. |

---

## 9. Missing-Value and Warm-Up Policy

1. **No Fabricated Historical Rows:** Missing dates (pre-IPO histories or trading suspensions) and missing benchmark volume (`F51`) are **never** fabricated or backfilled.
2. **Expected Warm-Up `NaN`s:** Features with lookback $W$ naturally produce `NaN` for the first $W - 1$ (or $W$) rows per stock; up to 80 trading days per stock are covered by `is_warmup = True`.
3. **Zero-Denominator Protection:** Any division by zero (`std == 0`, `ma20 == 0`, `vol_20d == 0`, `downside_vol_20d == 0`, `nifty_volume_ratio == 0`) evaluates to **`NaN`**, never `inf` or `-inf`.
4. **No Backward-Fill Imputation:** Backward-filling (`bfill`) is strictly prohibited. Forward-fill is permitted only for market-level index context (`benchmark_close_fallback`, `F52`–`F54`, `vix_close`, `vix_z20`) on active equity dates absent from index feeds.

---

## 10. Leakage Prevention Rules

1. **Strict Causal Windows:** Every feature at time $t$ uses data in $[t - W + 1, t]$ only (future shifting `shift(-k)` is strictly forbidden in `src/features/`).
2. **No Cross-Stock Contamination:** Rolling calculations execute strictly per-ticker.
3. **No Cross-Sectional Standardization in Phase 5:** `StandardScaler` and cross-sectional LRS percentile ranks belong to Phases 7–8 and must fit on training folds only.
4. **No Target Computation in Phase 5:** `HRE_5d` (which uses $t+1 \dots t+5$) must not be computed or merged in `src/features/`.

---

## 11. Configuration Inconsistencies & Decisions Approved (**C1–C10**)

| # | Topic | Pre-Audit Conflict / Gap | Approved Resolution (Frozen) | Pending Sync Targets (On Impl. Approval) |
|---|---|---|---|---|
| **C1** | **India VIX** | `feature_config.yaml` had `enabled: false` vs. `pipeline_config.yaml` `enabled: true` (`VERIFIED`). | **Enable `india_vix_features` and mark `VERIFIED`.** Implement `vix_close`, `vix_return_1d`, `vix_z20` outside `F01`–`F55`. | `configs/feature_config.yaml`, `src/features/market_stress.py` |
| **C2** | **Feature Count** | `feature_config.yaml`, `README.md`, `DATA_DICTIONARY_AND_FEATURES.md` listed Category 6 count as `5` instead of `4`. | **Correct Category 6 from `5` to `4`.** Core total remains **55** (`F01`–`F55`). | `configs/feature_config.yaml`, `README.md`, `docs/DATA_DICTIONARY_AND_FEATURES.md` |
| **C3** | **Annualization** | `vol_5d` (`F11`) & `nifty_vol_5d` omitted `× sqrt(252)` while `vol_20d` included it. | **Annualize `F11` (`vol_5d`) and `nifty_vol_5d` using $\sqrt{252}$** so `F14` & `F53` are dimensionless and consistent. | `configs/feature_config.yaml`, `docs/DATA_DICTIONARY_AND_FEATURES.md`, `src/features/` |
| **C4** | **`F18`, `F19`, `F21`** | Semi-deviation vs. filtered std (`F18`), ratio naming (`F19`), Wilder vs. SMA & zero edge cases (`F21`). | **`F18`:** Target-zero semi-deviation over full 20-obs window $\times \sqrt{252}$. **`F19`:** Retain `return_20d / downside_vol_20d` documented as Sortino component. **`F21`:** Wilder RSI ($\alpha=1/14$), 14-change init, explicit zero-gain/loss cases. | `docs/DATA_DICTIONARY_AND_FEATURES.md`, `src/features/volatility.py` |
| **C5** | **`F51` (`relative_volume`)** | `^NSEI` `volume` absent from `ohlcv_clean.parquet`; 27 zero-vol rows & 5 unmatched dates. | **Read `^NSEI` `volume` from `nifty50_clean.parquet`.** Set `F51 = NaN` when `nifty_volume_ratio` is zero, invalid, or unavailable (incl. unmatched dates). Never fabricate volume. | `src/features/market_relative.py`, `src/features/engineer.py` |
| **C6** | **Corporate-Action Masking** | Only `daily_return` masking was documented; multi-day/TR/CS rules unspecified. | **Mask `F01`/`F02`, compound `F03`–`F05` (`NaN` if window has masked return), mask `F09`–`F10`/`F20`/`F16`–`F17`/`F43` when window/pair includes affected transition; retain `YESBANK.NS` shocks.** | `src/features/returns.py`, `volatility.py`, `liquidity.py`, `gaps.py` |
| **C7** | **`F28` (`amihud_ma20`)** | Default `rolling(20)` would drop 20 days per zero-vol `NaN`. | **Set `min_periods=15`** for `amihud_ma20` (`F28`); preserve `NaN` for $<15$ valid observations. | `src/features/liquidity.py`, `configs/feature_config.yaml` |
| **C8** | **NIFTY Bank** | `banknifty_clean.parquet` exists but 0 features specified. | **Exclude `^NSEBANK` from core `F01`–`F55`**; reserve for possible future supplementary sector analysis. | `docs/DATA_SOURCE_AUDIT.md`, `src/features/` |
| **C9** | **Architecture & Output** | `process_universe` lacked `vix_path`; `features.parquet` vs `features_YYYYMMDD.parquet`; stale Garman-Klass/Yang-Zhang in `PROJECT_STATUS.md`. | **Add optional `vix_path` to `process_universe()`. Standardize output as `data/processed/features.parquet`. Reconcile stale Garman-Klass/Yang-Zhang docs; do not implement them.** | `src/features/engineer.py`, `docs/PROJECT_STATUS.md`, `docs/ARCHITECTURE.md` |
| **C10** | **`F20` Window** | `DATA_DICTIONARY_AND_FEATURES.md` wrote `Close_{t-20:t}` (21 obs) vs. `lookback: 20`. | **Use exactly 20 observations: `[t-19, t]`.** | `docs/DATA_DICTIONARY_AND_FEATURES.md`, `src/features/volatility.py` |

---

## 12. Final Implementation Readiness Checklist

- [x] **Decision C1 (Approved):** Enable `india_vix_features` as `VERIFIED` and implement `vix_close`, `vix_return_1d`, `vix_z20` outside `F01`–`F55`.
- [x] **Decision C2 (Approved):** Correct Category 6 count from `5` to `4`; total core feature count remains `55`.
- [x] **Decision C3 (Approved):** Annualize `F11` (`vol_5d`) and `nifty_vol_5d` using $\sqrt{252}$.
- [x] **Decision C4 (Approved):** Lock `F18` (target-0 semi-deviation $\times \sqrt{252}$), `F19` (`return_20d / downside_vol_20d` Sortino component), and `F21` (Wilder RSI $\alpha=1/14$ with explicit edge cases).
- [x] **Decision C5 (Approved):** Source `^NSEI` `volume` from `nifty50_clean.parquet`; set `F51 = NaN` on zero, invalid, or unmatched benchmark volume.
- [x] **Decision C6 (Approved):** Lock corporate-action masking across `F01`–`F05`, `F09`–`F10`, `F16`–`F17`, `F20`, `F40`–`F44`, and `F27`–`F29`, while retaining `is_market_shock` (`YESBANK.NS`).
- [x] **Decision C7 (Approved):** Set `min_periods=15` for `F28` (`amihud_ma20`).
- [x] **Decision C8 (Approved):** Exclude `^NSEBANK` from `F01`–`F55` and reserve for future supplementary sector analysis.
- [x] **Decision C9 (Approved):** Add optional `vix_path` to `FeatureEngineer.process_universe()`, standardize output path to `data/processed/features.parquet`, and reconcile stale Garman-Klass/Yang-Zhang references.
- [x] **Decision C10 (Approved):** Use exact 20-observation window `[t-19, t]` for `F20` (`max_drawdown_20d`).
- [x] **Implementation & Verification Gate (Completed):** Implemented `src/features/*.py`, synchronized configs/docs, generated `data/processed/features.parquet` (565,190 rows × 70 cols, `float32`, zero `inf` values), and passed all 61 unit and integration tests (`tests/`). See `reports/phase5_feature_engineering_report.md`.

---

## 13. Approved Phase 5 Mathematical Contract

This section constitutes the **immutable mathematical and engineering contract** governing Phase 5 implementation.

### 13.1 Output Scope & Schema Contract
- **Primary Artifact:** `data/processed/features.parquet` (Snappy-compressed Parquet written atomically via `StreamingParquetWriter`).
- **Row Preservation:** Exact 1-to-1 correspondence with `data/interim/stocks/ohlcv_clean.parquet` (**565,190 rows**, `300` tickers, zero dropped rows, zero fabricated historical dates).
- **Feature Columns:**
  - **55 Core Features (`F01`–`F55`):** Exact column names from `configs/feature_config.yaml`, all cast to **`float32`**.
  - **3 Supplementary Market-Fear Features (Separate from `F01`–`F55`):** `vix_close`, `vix_return_1d`, `vix_z20`, all cast to **`float32`**.
  - **Retained Metadata & Phase 4.2 Flags:** `date`, `ticker`, `cap_segment` (from `configs/universe.json`), plus Phase 4.2 quality flags (`is_zero_volume`, `is_stale_price`, `is_corporate_action_gap`, `is_market_shock`, `is_short_history`, `is_warmup`, `is_benchmark_available`, `is_muhurat_session`, `is_valid_ohlcv`).
- **Excluded Inputs:** `^NSEBANK` (`banknifty_clean.parquet`) is **not** read or emitted; Garman-Klass and Yang-Zhang estimators are **not** implemented.

### 13.2 Corporate-Action & Market-Shock Masking Contract
Let $g_t = \text{is\_corporate\_action\_gap}_t$ (True on the 3 confirmed unadjusted corporate-action rows; False on the 2 `YESBANK.NS` `is_market_shock` rows, which are preserved intact):
1. **1-Day Return / Gap / Impact Metrics (`F01`, `F02`, `F08`, `F16`, `F27`, `F34`, `F40`, `F41`):**
   - When $g_t = \text{True}$: set `F01`, `F02`, `F08`, `F16`, `F27`, `F34`, `F40`, `F41` to `NaN`.
2. **Compounded Multi-Day Returns (`F03`, `F04`, `F05`):**
   - Computed from masked `F01` ($r_t$): $R_{k,t} = \prod_{i=0}^{k-1}(1 + r_{t-i}) - 1$. If **any** $r_{t-i}$ in $[t-k+1, t]$ is `NaN` (including $g_{t-i}=\text{True}$), $R_{k,t} = \text{NaN}$.
3. **Price-Level Rolling Windows (`F09`, `F10`, `F20`):**
   - A price transition from $t-1$ to $t$ is affected whenever $g_t = \text{True}$. For a $W$-observation price window $[t-W+1, t]$ ($W=20$ for `F09`/`F20`, $W=60$ for `F10`), the window spans an affected transition if $\sum_{i=0}^{W-2} \mathbb{I}(g_{t-i}) > 0$ (i.e., any day after the first bar of the window has $g = \text{True}$); in that case, mask the feature at day $t$ to `NaN`.
   - `F20` window is strictly $[t-19, t]$ (`20` observations).
4. **Two-Day Corwin-Schultz Pair (`F43`) & `F44`:**
   - `F43` at day $t$ is masked to `NaN` if $g_{t-1} = \text{True}$ **or** $g_t = \text{True}$. Negative valid estimates are clipped at `0.0`. `F44` is the 10-day rolling mean of `F43` (`min_periods=10`, so `NaN` propagates while masked `F43` sits in `[t-9, t]`).
5. **True Range & ATR (`F16`, `F17`):**
   - `F16` (`true_range`) is `NaN` at $t=0$ and when $g_t = \text{True}$. `F17` (`atr_20d`) is the 20-day rolling mean of `F16` with `min_periods=20` (yielding `NaN` whenever a masked corporate-action transition falls inside `[t-19, t]`).
6. **Wilder RSI-14 (`F21`):**
   - Compute $\Delta C_t = C_t - C_{t-1}$, masking $\Delta C_t = \text{NaN}$ when $g_t = \text{True}$ (and at $t=0$). Apply Wilder smoothing ($\alpha = 1/14$) initialized on the first 14 valid price changes; apply the explicit zero-gain/zero-loss edge cases in §5.2(4).

### 13.3 Volatility, Liquidity & Benchmark Contract
1. **Annualization (`F11`–`F15`, `F18`, `F52`–`F53`):**
   - `F11` (`vol_5d`), `F12` (`vol_20d`), `F13` (`vol_60d`), `F15` (`parkinson_vol`), `F18` (`downside_vol_20d`), `nifty_vol_5d`, and `F52` (`nifty_vol_20d`) are all annualized by $\sqrt{252}$.
   - `F14` (`vol_5d / vol_20d`) and `F53` (`nifty_vol_5d / nifty_vol_20d`) are dimensionless ratios (`NaN` if denominator $\le 0$).
2. **Downside Semi-Deviation (`F18`) & Sortino Component (`F19`):**
   - `F18` uses target $0$ over all 20 observations in $[t-19, t]$ (`min_periods=20` valid `log_return`s).
   - `F19 = return_20d / downside_vol_20d` (`NaN` if `downside_vol_20d <= 0` or `return_20d` is `NaN`).
3. **Zero-Volume & Amihud (`F27`–`F29`):**
   - `F27` is `NaN` when `Volume == 0` or $g_t = \text{True}$.
   - `F28` (`amihud_ma20`) uses `rolling(20, min_periods=15).mean()`.
   - `F29` (`amihud_z20`) uses `rolling(20, min_periods=15).std()` and evaluates to `NaN` if `F27` is `NaN`, valid count $<15$, or $\sigma_{20} = 0$.
4. **NIFTY 50 Relative Volume (`F51`):**
   - `nifty_volume_ma20` and `nifty_volume_ratio` are computed on `nifty50_clean.parquet`.
   - `F51 = NaN` whenever `is_benchmark_available == False`, `^NSEI` `volume <= 0`, `nifty_volume_ma20 <= 0`, or `nifty_volume_ratio` is `NaN`/$\le 0$.
5. **India VIX Supplementary Features (`vix_close`, `vix_return_1d`, `vix_z20`):**
   - Computed on `india_vix_clean.parquet`, left-joined by `date` onto stock series with strictly backward `.ffill()` for levels/z-score (`vix_close`, `vix_z20`) and `0.0` for `vix_return_1d` on unmatched active equity dates after row 0.
6. **Zero-Denominator & Precision Guarantee:**
   - Every ratio or z-score denominator $\le 0$ or `== 0` produces `NaN` (zero `inf`/`-inf` values allowed across the entire 565,190-row output).
   - Every engineered feature column is stored as `float32`.
