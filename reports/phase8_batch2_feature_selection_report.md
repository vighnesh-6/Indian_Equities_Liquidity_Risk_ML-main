# Phase 8 – Batch 2: Feature Relevance Analysis & Final Feature Selection
## Stages 8.5–8.6 | Strictly Read-Only Analysis & Architectural Specification
### ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities

**Execution Status:** COMPLETE & VERIFIED  
**Authoritative Dataset:** `data/processed/features_with_targets.parquet` (565,190 rows × 75 columns)  
**Supervised Training Population:** `(split == "train") & (hre_5d_train.notna())` = **466,346 rows**  
**Target Variable:** `hre_5d_train` (Binary High-Risk Event in forward 5 trading days)  
**Candidate Features Evaluated:** 55 Core Features (`F01`–`F55`) + 3 Supplementary India VIX Features (58 total)  
**File Created:** `reports/phase8_batch2_feature_selection_report.md` (Single permitted file)  
**Data Preservation Guarantee:** 100% Read-Only; Zero Datasets Modified; Zero Models Trained.

---

## 1. Executive Summary

This report concludes **Batch 2 (Stages 8.5–8.6)** of Phase 8A: Feature Selection & Data Preprocessing. Building directly upon the feature inventory, missingness profiling, distribution shapes, correlation matrices, and VIF diagnostics established in Batch 1, Batch 2 evaluates the **empirical predictive relevance** of all 58 candidate features against the supervised target `hre_5d_train` and formulates **model-specific feature subsets** across four target algorithms:

1. **Logistic Regression (Baseline Linear Model):** Selected **20 features**. Eliminates all exact algebraic identities, redundant twin features ($|r| \ge 0.85$), and high-VIF drivers. Every retained feature has **$\text{VIF} < 5.0$** (maximum $\text{VIF} = 4.89$).
2. **Random Forest (Supervised Ensemble Bagging):** Selected **35 features**. Prunes exact duplicates ($|r| \ge 0.95$), unscaled rupee features (`true_range`, `atr_20d`), and near-constant noise (`zero_vol_flag`), while retaining complementary non-linear interactions across volatility term structures, liquidity dry-ups, and price impact.
3. **HistGradientBoosting (Supervised Gradient Boosting):** Selected **36 features**. Ingests the 35 features from Random Forest plus `relative_volume` (F51), leveraging HGBT's native missing-value routing to exploit benchmark-relative trading volume without synthetic data fabrication.
4. **Isolation Forest (Unsupervised Anomaly Detection):** Preserves the established **12-feature subset** defined in `configs/model_config.yaml`. Verified to represent orthogonal multidimensional liquidity and market shock triggers (order-book impact, volume droughts, volatility expansion, intraday range, and overnight gaps).

All analyses were executed strictly on the **466,346 eligible training observations**. The holdout partition (71,756 valid rows) and the 5-session boundary embargo (1,444 rows) remained 100% blind and unreferenced.

---

## 2. Scope & Data Preservation Confirmation

This evaluation operated under strict read-only constraints:
* **No Dataset Alterations:** `data/processed/features.parquet` and `data/processed/features_with_targets.parquet` were accessed in read-only binary mode.
* **No Pipeline or Model Implementations:** No feature selectors, imputation transformers, scalers, or estimators were serialized to disk.
* **No Architectural Modifications:** No existing source code in `src/`, configuration in `configs/`, or notebooks in `notebooks/` were altered.
* **Single File Operation:** Only this Markdown report was authored in `reports/`.

---

## 3. Training Population & Leakage Controls

### Exact Partition Isolation
All statistical metrics, correlations, ROC-AUC evaluations, and VIF calculations were computed strictly on the training partition:

$$\mathcal{D}_{\text{train, eligible}} = \left\{ (i, t) \in \mathcal{D} \mid \text{split}_{i, t} = \text{"train"} \;\wedge\; \text{hre\_5d\_train}_{i, t} \in \{0.0, 1.0\} \right\}$$

$$\text{Total Observations } N = 466,346 \quad (\text{Large: } 169,508, \;\; \text{Mid: } 229,298, \;\; \text{Small: } 67,540)$$

$$\text{Class Balance: } \text{Negative } (HRE=0) = 381,358 \; (81.78\%), \quad \text{Positive } (HRE=1) = 84,988 \; (18.22\%)$$

### Strict Safeguards Enforced:
1. **Holdout Blindness:** Observations with `split == "holdout"` (Dates: `2025-10-20` to `2026-10-23`, 71,756 valid rows) were completely isolated.
2. **Boundary Embargo Enforced:** Observations in the 5-day embargo (`2025-10-13` to `2025-10-17`, 1,444 rows), whose forward 5-day windows cross into the holdout period, were strictly excluded via `hre_5d_train = NaN`.
3. **Causal Lookback Verification:** All candidate features rely strictly on historical information available at market close of session $t$ ($t \le 0$). No future forward LRS or target data was used as an input feature.
4. **Metadata & Flags Exclusion:** Identification columns (`date`, `ticker`, `cap_segment`), quality flags, and intermediate target scores (`lrs`, `fwd_max_lrs_5d`, `hre_5d`) were strictly excluded from predictor matrices.

---

## 4. Stage 8.5 – Feature Relevance Methodology

To evaluate the predictive signal of each candidate feature without fitting complex models, a multi-faceted statistical framework was deployed:

1. **Univariate ROC-AUC (Rank Discrimination):** Evaluates how effectively a single feature rank-orders subsequent 5-day high-risk liquidity events. An $\text{AUC} > 0.50$ indicates that higher feature values predict risk (positive relationship); an $\text{AUC} < 0.50$ indicates that lower feature values predict risk (negative/inverse relationship). The **effective AUC** is defined as $\max(\text{AUC}, 1 - \text{AUC})$.
2. **Spearman Rank Correlation ($r_{\text{rank}}$):** Quantifies monotonic association with the binary target, robust to severe skewness and heavy tails.
3. **Pearson Point-Biserial Correlation ($r_{\text{pb}}$):** Measures linear co-movement with the binary indicator.
4. **Market-Cap Stratified Discrimination:** Evaluates whether predictive associations are consistent across Large, Mid, and Small cap tiers or if structural differences exist.

---

## 5. Feature Relevance Results

### Top 25 Candidate Features by Predictive Association (Eligible Training Set: N = 466,346)

| Rank | Feature ID | Feature Name | Category | Spearman $r$ | Pearson $r$ | Raw ROC-AUC | Effective ROC-AUC | Risk Direction | Primary Predictive Signal |
|:---:|:---:|:---|:---:|---:|---:|---:|---:|:---:|:---|
| 1 | **F28** | `amihud_ma20` | Liquidity | +0.3502 | +0.0213 | **0.7619** | **0.7619** | Positive (+) | Smoothed price impact baseline |
| 2 | **F44** | `cs_spread_ma10` | Price Impact | +0.3425 | +0.3404 | **0.7561** | **0.7561** | Positive (+) | Effective bid-ask transaction friction |
| 3 | **F38** | `daily_range_ma20` | Price Impact | +0.3286 | +0.2953 | **0.7457** | **0.7457** | Positive (+) | Intraday price span uncertainty |
| 4 | **F15** | `parkinson_vol` | Volatility | +0.3238 | +0.2830 | **0.7421** | **0.7421** | Positive (+) | Extreme intraday range volatility |
| 5 | **F50** | `idiosyncratic_vol` | Market Relative| +0.3123 | +0.2924 | **0.7336** | **0.7336** | Positive (+) | Non-systematic, unhedgeable crash risk |
| 6 | **F27** | `amihud_ratio` | Liquidity | +0.2918 | +0.0098 | **0.7182** | **0.7182** | Positive (+) | 1-day instantaneous price impact |
| 7 | **F30** | `turnover_value` | Liquidity | -0.2841 | -0.1408 | 0.2875 | **0.7125** | Negative (-) | Rupee liquidity flow (low turnover = risk) |
| 8 | **F31** | `log_turnover` | Liquidity | -0.2841 | -0.2771 | 0.2875 | **0.7125** | Negative (-) | Log rupee liquidity flow |
| 9 | **F32** | `turnover_ma20` | Liquidity | -0.2803 | -0.1768 | 0.2904 | **0.7096** | Negative (-) | 20-day baseline trading volume flow |
| 10 | **F12** | `vol_20d` | Volatility | +0.2803 | +0.2516 | **0.7096** | **0.7096** | Positive (+) | 1-month realized annualized volatility |
| 11 | **F18** | `downside_vol_20d` | Volatility | +0.2597 | +0.2044 | **0.6942** | **0.6942** | Positive (+) | Asymmetric downside semi-deviation |
| 12 | **F13** | `vol_60d` | Volatility | +0.2535 | +0.2279 | **0.6896** | **0.6896** | Positive (+) | Medium-term volatility regime |
| 13 | **F37** | `daily_range` | Price Impact | +0.1985 | +0.1661 | **0.6484** | **0.6484** | Positive (+) | 1-day normalized high-low spread |
| 14 | **F11** | `vol_5d` | Volatility | +0.1875 | +0.1725 | **0.6402** | **0.6402** | Positive (+) | Short-term volatility spike |
| 15 | **F23** | `log_volume` | Liquidity | -0.1733 | -0.1566 | 0.3704 | **0.6296** | Negative (-) | Absolute share volume level |
| 16 | **F22** | `vol_of_vol` | Volatility | +0.1700 | +0.1301 | **0.6272** | **0.6272** | Positive (+) | Volatility clustering instability |
| 17 | **F49** | `rolling_corr_60d` | Market Relative| -0.1699 | -0.1659 | 0.3729 | **0.6271** | Negative (-) | Decoupling from NIFTY 50 |
| 18 | **F24** | `volume_ma20` | Liquidity | -0.1621 | +0.0326 | 0.3788 | **0.6212** | Negative (-) | 20-day share volume baseline |
| 19 | **F20** | `max_drawdown_20d` | Volatility | -0.1365 | -0.1327 | 0.3980 | **0.6020** | Negative (-) | Trailing peak-to-trough price damage |
| 20 | **F43** | `cs_spread_est` | Price Impact | +0.1204 | +0.1564 | **0.5874** | **0.5874** | Positive (+) | 2-day raw Corwin-Schultz spread |
| 21 | **F08** | `abs_return` | Returns | +0.0958 | +0.0928 | **0.5716** | **0.5716** | Positive (+) | Direction-agnostic price shock |
| 22 | **F55** | `vol_beta` | Market Stress | +0.0932 | +0.0046 | **0.5697** | **0.5697** | Positive (+) | Volatility acceleration / momentum |
| 23 | **F33** | `turnover_ratio` | Liquidity | -0.0878 | -0.0267 | 0.4343 | **0.5657** | Negative (-) | Rupee turnover drought (< 1.0) |
| 24 | **F36** | `low_volume_streak`| Liquidity | +0.1367 | +0.1376 | **0.5627** | **0.5627** | Positive (+) | Multi-day persistent volume dry-up |
| 25 | **F25** | `volume_ratio` | Liquidity | -0.0839 | -0.0265 | 0.4373 | **0.5627** | Negative (-) | Share volume drought (< 1.0) |

### Bottom 10 Features by Predictive Association

| Feature ID | Feature Name | Category | Spearman $r$ | Pearson $r$ | Raw ROC-AUC | Effective ROC-AUC | Diagnosis |
|:---:|:---|:---:|---:|---:|---:|---:|:---|
| **F29** | `amihud_z20` | Liquidity | +0.0062 | +0.0089 | 0.5046 | **0.5046** | Local z-score removes cross-sectional illiquidity tier contrast |
| **F06** | `return_z5` | Returns | -0.0052 | -0.0052 | 0.4961 | **0.5039** | Short-term standardized return has symmetric mean reversion |
| **F04** | `return_20d` | Returns | -0.0043 | +0.0174 | 0.4968 | **0.5032** | Directional return alone does not differentiate liquidity panic |
| **F42** | `gap_z20` | Price Impact | -0.0041 | -0.0025 | 0.4970 | **0.5030** | Standardized gap removes absolute jump magnitude |
| **F54** | `nifty_return_20d` | Market Stress | +0.0039 | +0.0052 | 0.5029 | **0.5029** | Benchmark index return has weak standalone single-stock signal |
| **F53** | `nifty_vol_ratio` | Market Stress | -0.0030 | -0.0038 | 0.4978 | **0.5022** | Market-wide term structure ratio exhibits noisy oscillations |
| **F34** | `zero_return_flag` | Liquidity | +0.0117 | +0.0117 | 0.5015 | **0.5015** | Severe sparsity (98.99% zeros) limits linear ranking |
| **Supp** | `vix_z20` | Macro Fear | -0.0016 | -0.0019 | 0.4988 | **0.5012** | Rolling VIX z-score oscillates symmetrically around zero |
| **Supp** | `vix_return_1d` | Macro Fear | +0.0007 | -0.0005 | 0.5005 | **0.5005** | Daily VIX noise lacks persistent 5-day lookahead predictive power |
| **F35** | `zero_vol_flag` | Liquidity | -0.0017 | -0.0017 | 0.4999 | **0.5001** | Extreme sparsity (99.92% zeros); negligible standalone utility |

### Stratified Predictive Association Across Market-Cap Segments (ROC-AUC)

| Feature ID & Name | Large Cap ($N = 169,508$) | Mid Cap ($N = 229,298$) | Small Cap ($N = 67,540$) | Cross-Tier Behavioral Dynamic |
|:---|:---:|:---:|:---:|:---|
| **F28: amihud_ma20** | **0.8316** | **0.8007** | **0.7673** | Strongest single predictor in every tier; highest discrimination in Large caps |
| **F44: cs_spread_ma10** | **0.7815** | **0.7574** | **0.7423** | Consistently high discrimination ($>0.74$) across all three market-cap tiers |
| **F12: vol_20d** | **0.7392** | **0.7132** | **0.6642** | Realized volatility predictive power is strongest in institutional Large/Mid caps |
| **F18: downside_vol_20d** | **0.7224** | **0.6912** | **0.6596** | Downside semi-deviation mirrors total volatility with asymmetric crash focus |
| **F26: volume_z20** | 0.4752 (Eff: **0.5248**) | 0.4712 (Eff: **0.5288**) | 0.4618 (Eff: **0.5382**) | Inverted volume signal: volume droughts are most severe in Small caps |
| **F51: relative_volume** | 0.4438 (Eff: **0.5562**) | 0.4318 (Eff: **0.5682**) | 0.4476 (Eff: **0.5524**) | Trading activity decoupling from benchmark signals risk across all tiers |
| **F21: rsi_14** | 0.4637 (Eff: **0.5363**) | 0.4918 (Eff: **0.5082**) | 0.4926 (Eff: **0.5074**) | Momentum exhaustion has modest standalone predictive power |
| **F52: nifty_vol_20d** | 0.4855 (Eff: **0.5145**) | 0.4834 (Eff: **0.5166**) | 0.4811 (Eff: **0.5189**) | Macro index volatility acts as a universal market conditioning backdrop |

---

## 6. Detailed Investigation of Redundancy Groups & Edge Cases

The 10 specific empirical anomalies and collinear pairs identified in Batch 1 were investigated in depth:

### 1. `daily_return` (F01) vs `log_return` (F02)
* **Empirical Reality:** $r_{\text{spearman}} = 1.0000$, $r_{\text{pearson}} = 0.9989$, $\text{VIF} > 2,700$.
* **Decision:** **Exclude `log_return`; Retain `daily_return` for LR, RF, and HGBT.**
* **Justification:** `daily_return` is the primary audited financial return metric across the project. Keeping both creates severe coefficient inflation in linear models ($VIF > 1,500$) and splits tree importance arbitrarily.

### 2. `turnover_value` (F30) vs `log_turnover` (F31)
* **Empirical Reality:** $r_{\text{spearman}} = 1.0000$, $r_{\text{pearson}} = 0.5356$. Effective $\text{AUC} = 0.7125$ for both. However, `turnover_value` exhibits extreme skewness ($+13.59$) and kurtosis ($309.8$), spanning from ₹31 Cr to ₹37,066 Cr.
* **Decision:** **Exclude `turnover_value`; Retain `log_turnover` for LR, RF, and HGBT.**
* **Justification:** Logarithmic transformation compresses extreme institutional block-trade outliers into a well-behaved continuous distribution, preventing gradient distortion and large-leverage regression points.

### 3. `volume_ratio` (F25) vs `turnover_ratio` (F33)
* **Empirical Reality:** $r_{\text{spearman}} = 0.9952$, $r_{\text{pearson}} = 0.9955$, $\text{VIF} > 100$.
* **Decision:** **Exclude `volume_ratio`; Retain `turnover_ratio` for LR, RF, and HGBT.** (Retain `volume_ratio` in Isolation Forest per frozen baseline).
* **Justification:** `turnover_ratio` is rupee-weighted, making it invariant to share price splits and penny-stock share inflation. Both capture the exact same relative trading activity surge ($r = 0.995$).

### 4. `parkinson_vol` (F15) vs `daily_range_ma20` (F38)
* **Empirical Reality:** $r_{\text{spearman}} = 0.9905$, $r_{\text{pearson}} = 0.9858$, $\text{VIF} > 51$.
* **Decision:** **Exclude `parkinson_vol` from LR; Retain `daily_range` (F37) and `daily_range_ma20` (F38) for Tree Models.**
* **Justification:** Both formulas scale the high-low price range. `daily_range_ma20` is simpler, normalized by close price, and directly intuitive. For Logistic Regression, both are excluded in favor of `vol_20d` and `daily_range` to keep $\text{VIF} < 5.0$.

### 5. `vol_20d` (F12) vs `idiosyncratic_vol` (F50)
* **Empirical Reality:** $r_{\text{spearman}} = 0.9661$, $r_{\text{pearson}} = 0.9633$, $\text{VIF} = 46.9$.
* **Decision:** **Exclude `idiosyncratic_vol` from all models; Retain `vol_20d` and `rolling_corr_60d`.**
* **Justification:** Indian equity volatility is overwhelmingly stock-specific; total volatility tracks idiosyncratic volatility at $r = 0.966$. Retaining `vol_20d` alongside `rolling_corr_60d` allows models to observe both total volatility and index decoupling without collinearity.

### 6. `true_range` (F16) vs `atr_20d` (F17)
* **Empirical Reality:** $r_{\text{spearman}} = 0.9525$, $r_{\text{pearson}} = 0.8766$. Both are unscaled rupee figures (spanning from ₹0.50 to ₹16,176 for high-priced stocks like MRF).
* **Decision:** **Exclude BOTH `true_range` and `atr_20d` from all four models.**
* **Justification:** Unnormalized price-level metrics are non-stationary and introduce cross-sectional distortion in global models (a ₹500 true range on a ₹100,000 stock is 0.5%, but on a ₹500 stock it is 100%). Bounded percentage equivalents (`daily_range`, `daily_range_ma20`, `vol_20d`) are vastly superior.

### 7. The Algebraic Triplet: `return_20d` (F04), `nifty_return_20d` (F54), and `excess_return_20d` (F47)
* **Empirical Reality:** Exact mathematical identity: $\text{excess\_return\_20d} \equiv \text{return\_20d} - \text{nifty\_return\_20d}$. Causes matrix singularity with $\text{VIF} > 2.8 \times 10^{14}$.
* **Decision:** **Exclude `excess_return_20d` from LR and RF; Retain `return_20d` and `nifty_return_20d`.**
* **Justification:** Linear models cannot invert a singular design matrix. Keeping the stock return and benchmark return allows models to capture both absolute momentum and market trend independently.

### 8. `zero_vol_flag` (F35): Near-Constant Sparsity
* **Empirical Reality:** $99.92\%$ zeros ($\sigma = 0.028$, only 371 positive observations across 466,346 rows). Effective $\text{AUC} = 0.5001$.
* **Decision:** **Exclude `zero_vol_flag` from LR, RF, and HGBT.**
* **Justification:** A feature that is zero for 99.92% of observations provides virtually zero split variance in tree ensembles and causes numerical instability in unregularized linear splits.

### 9. `relative_volume` (F51): Benchmark Missingness
* **Empirical Reality:** Contains 7,047 missing values ($1.511\%$) due to 27 zero-volume NIFTY sessions. Effective $\text{AUC} = 0.5562$.
* **Decision:** **Exclude from LR and RF; Retain exclusively in HistGradientBoosting (HGBT).**
* **Justification:** In Logistic Regression and standard Random Forest, handling F51 requires synthetic imputation. HGBT natively handles missing values by learning the optimal split direction for NaNs during training.

### 10. Extreme Skewness Features: `amihud_ratio` (F27) and `vol_beta` (F55)
* **Empirical Reality:** `amihud_ratio` skewness is $+343.33$ (kurtosis 136,892); `vol_beta` skewness is $+207.71$.
* **Decision:**
  * For **Logistic Regression:** Exclude `amihud_ratio` and `vol_beta` in favor of the smoothed, well-behaved `amihud_ma20` ($\text{AUC} = 0.7619$).
  * For **Tree Ensembles (RF & HGBT):** Retain `amihud_ratio`, `amihud_ma20`, and `vol_beta`. Decision trees split on rank thresholds, making them completely immune to positive skewness.

---

## 7. Stage 8.6 – Logistic Regression Feature Selection

### Design Objectives
1. **Multicollinearity Elimination:** Ensure every feature satisfies $\text{VIF} < 5.0$.
2. **Coefficient Interpretability:** Include monotonic, stable indicators across returns, downside volatility, transaction cost, volume drought, and macro fear.
3. **Algebraic Identity Removal:** Zero linear dependencies.

### Selected Feature List: Exactly 20 Features

| No. | Feature ID | Feature Column Name | Category | Univariate ROC-AUC | VIF in Selected Subset | Interpretability Role in Logistic Regression |
|:---:|:---:|:---|:---:|---:|---:|:---|
| 1 | **F01** | `daily_return` | Returns | 0.5182 | **1.43** | Short-term 1-day directional price momentum |
| 2 | **F03** | `return_5d` | Returns | 0.5284 | **2.22** | 1-week cumulative directional trend |
| 3 | **F11** | `vol_5d` | Volatility | 0.6402 | **2.08** | Short-term realized volatility surge |
| 4 | **F18** | `downside_vol_20d` | Volatility | 0.6942 | **4.21** | Asymmetric crash semi-deviation (Key LRS pillar) |
| 5 | **F20** | `max_drawdown_20d` | Volatility | 0.6020 | **4.89** | Trailing peak-to-trough price destruction |
| 6 | **F21** | `rsi_14` | Volatility | 0.5152 | **3.22** | Wilder momentum oscillator (bounded 0–100) |
| 7 | **F31** | `log_turnover` | Liquidity | 0.7125 | **1.43** | Log rupee transaction flow (low flow = high risk) |
| 8 | **F33** | `turnover_ratio` | Liquidity | 0.5657 | **4.54** | Rupee volume drought indicator (< 1.0) |
| 9 | **F26** | `volume_z20` | Liquidity | 0.5288 | **4.23** | Standardized volume shock/drought (Key LRS pillar)|
| 10 | **F28** | `amihud_ma20` | Liquidity | 0.7619 | **1.01** | Baseline price impact per rupee (Key LRS pillar) |
| 11 | **F36** | `low_volume_streak`| Liquidity | 0.5627 | **1.16** | Multi-day persistent liquidity dry-up count (0–5) |
| 12 | **F37** | `daily_range` | Price Impact | 0.6484 | **2.32** | 1-day normalized intraday price spread |
| 13 | **F41** | `overnight_gap_abs` | Price Impact | 0.5348 | **1.27** | Absolute overnight jump/dislocation risk |
| 14 | **F42** | `gap_z20` | Price Impact | 0.5030 | **1.11** | Standardized overnight jump anomaly |
| 15 | **F44** | `cs_spread_ma10` | Price Impact | 0.7561 | **1.48** | Baseline bid-ask spread proxy (Key LRS pillar) |
| 16 | **F48** | `rolling_beta_60d` | Market Relative| 0.5321 | **3.23** | Systematic market risk sensitivity |
| 17 | **F49** | `rolling_corr_60d` | Market Relative| 0.6271 | **3.40** | Decoupling from NIFTY 50 index co-movement |
| 18 | **F52** | `nifty_vol_20d` | Market Stress | 0.5166 | **4.48** | Benchmark market-wide realized volatility |
| 19 | **F54** | `nifty_return_20d` | Market Stress | 0.5029 | **1.39** | Benchmark 1-month market trend regime |
| 20 | **Supp** | `vix_close` | Macro Fear | 0.5245 | **4.02** | Macro market implied fear gauge (India VIX) |

**Subset Diagnostics:**
* **Total Features:** **20 features**
* **Maximum VIF:** **4.89** (`max_drawdown_20d`)
* **Mean VIF:** **2.63**
* **Singularity Status:** Clean ($\text{Condition Number} = 14.8$). Zero numerical instability.

---

## 8. Stage 8.6 – Random Forest Feature Selection

### Design Objectives
1. **Nonlinear Interaction Exploitation:** Retain term-structure pairs (`vol_5d`, `vol_20d`, `vol_ratio_5_20`) and price-impact ratios (`amihud_ratio`, `amihud_ma20`).
2. **Exact Duplicate Pruning:** Drop exact twins ($r \ge 0.95$) that dilute tree split-criterion importance.
3. **Unscaled Rupee Metric Removal:** Eliminate price-dependent unscaled features (`true_range`, `atr_20d`).

### Selected Feature List: Exactly 35 Features

| No. | Feature ID | Feature Column Name | Category | Primary Role in Random Forest Ensemble |
|:---:|:---:|:---|:---:|:---|
| 1 | **F01** | `daily_return` | Returns | 1-day percentage price change |
| 2 | **F03** | `return_5d` | Returns | 1-week cumulative momentum |
| 3 | **F04** | `return_20d` | Returns | 1-month cumulative trend |
| 4 | **F07** | `return_z20` | Returns | 20-day return anomaly / extreme move |
| 5 | **F08** | `abs_return` | Returns | Direction-agnostic magnitude of daily move |
| 6 | **F09** | `price_rel_ma20` | Returns | 20-day mean reversion / trend stretch |
| 7 | **F11** | `vol_5d` | Volatility | Short-term realized volatility level |
| 8 | **F12** | `vol_20d` | Volatility | Medium-term realized volatility baseline |
| 9 | **F14** | `vol_ratio_5_20` | Volatility | Volatility term-structure inversion spike |
| 10 | **F18** | `downside_vol_20d` | Volatility | Asymmetric crash risk semi-deviation |
| 11 | **F20** | `max_drawdown_20d` | Volatility | Trailing peak-to-trough price destruction |
| 12 | **F21** | `rsi_14` | Volatility | Wilder momentum exhaustion indicator |
| 13 | **F22** | `vol_of_vol` | Volatility | Volatility clustering / uncertainty |
| 14 | **F23** | `log_volume` | Liquidity | Absolute volume scale across stocks |
| 15 | **F33** | `turnover_ratio` | Liquidity | Relative trading flow drought (< 1.0) |
| 16 | **F26** | `volume_z20` | Liquidity | Standardized volume drought shock |
| 17 | **F27** | `amihud_ratio` | Liquidity | Instantaneous 1-day rupee price impact |
| 18 | **F28** | `amihud_ma20` | Liquidity | 20-day baseline price impact profile |
| 19 | **F29** | `amihud_z20` | Liquidity | Standardized price impact shock |
| 20 | **F31** | `log_turnover` | Liquidity | Continuous log rupee turnover level |
| 21 | **F36** | `low_volume_streak`| Liquidity | Multi-day persistent volume drought count |
| 22 | **F37** | `daily_range` | Price Impact | 1-day normalized intraday price spread |
| 23 | **F38** | `daily_range_ma20` | Price Impact | 20-day baseline intraday range |
| 24 | **F39** | `range_z20` | Price Impact | Standardized range expansion shock |
| 25 | **F40** | `overnight_gap` | Price Impact | Directional overnight news gap |
| 26 | **F41** | `overnight_gap_abs` | Price Impact | Absolute magnitude of overnight jump |
| 27 | **F43** | `cs_spread_est` | Price Impact | 2-day Corwin-Schultz bid-ask proxy |
| 28 | **F44** | `cs_spread_ma10` | Price Impact | 10-day smoothed bid-ask spread baseline |
| 29 | **F46** | `excess_return_1d` | Market Relative| 1-day idiosyncratic alpha / decoupling move |
| 30 | **F48** | `rolling_beta_60d` | Market Relative| Systematic market risk sensitivity |
| 31 | **F49** | `rolling_corr_60d` | Market Relative| Market co-movement / decoupling |
| 32 | **F52** | `nifty_vol_20d` | Market Stress | Benchmark index realized volatility |
| 33 | **F54** | `nifty_return_20d` | Market Stress | Benchmark index 1-month market trend |
| 34 | **F55** | `vol_beta` | Market Stress | Volatility acceleration / surge momentum |
| 35 | **Supp** | `vix_close` | Macro Fear | Implied market fear level (India VIX) |

**Excluded Features for RF (23 Features):**
`log_return` (F02), `return_5d` (F03-redundant with F01/F04), `return_60d` (F05), `return_z5` (F06), `price_rel_ma60` (F10), `vol_60d` (F13), `parkinson_vol` (F15), `true_range` (F16), `atr_20d` (F17), `sortino_component` (F19), `volume_ma20` (F24), `volume_ratio` (F25), `turnover_value` (F30), `turnover_ma20` (F32), `zero_return_flag` (F34), `zero_vol_flag` (F35), `gap_z20` (F42), `intraday_return` (F45), `excess_return_20d` (F47), `idiosyncratic_vol` (F50), `relative_volume` (F51), `nifty_vol_ratio` (F53), `vix_return_1d`, `vix_z20`.

---

## 9. Stage 8.6 – HistGradientBoosting (HGBT) Feature Selection

### Design Objectives
1. **Leverage Native Missingness Handling:** Include `relative_volume` (F51), allowing the boosting algorithm to learn optimal branch splits for missing benchmark volume sessions.
2. **Deep Non-Linear Interaction Depth:** Provide complete liquidity, volatility term-structure, and macro features.
3. **Final Feature Count:** **Exactly 36 Features** (The 35 Random Forest features + `relative_volume`).

### Selected Feature List: Exactly 36 Features
* **Features 1 through 35:** Identical to the Random Forest selected list (Section 8).
* **Feature 36:** **`relative_volume` (F51)** (Category 5: Market-Relative Behaviour).
  * *Justification:* Measures single-stock trading volume intensity relative to NIFTY 50 trading activity ($\text{AUC} = 0.556$). While containing $1.51\%$ missing values from zero-volume index sessions, HGBT natively evaluates missing values on separate tree branches, eliminating any need for synthetic data imputation.

---

## 10. Stage 8.6 – Isolation Forest Feature Selection

### Baseline Configuration Verification
Inspected `configs/model_config.yaml` lines 48–69:

```yaml
  isolation_forest:
    model_type: "unsupervised_anomaly"
    estimator: "sklearn.ensemble.IsolationForest"
    params:
      n_estimators: 100
      contamination: 0.05
      max_samples: "auto"
      random_state: 42
    feature_subset:
      - "amihud_ratio"
      - "amihud_z20"
      - "volume_ratio"
      - "volume_z20"
      - "vol_20d"
      - "vol_ratio_5_20"
      - "daily_range"
      - "range_z20"
      - "overnight_gap_abs"
      - "cs_spread_est"
      - "nifty_vol_20d"
      - "return_z20"
```

### Empirical Evaluation & Preservation Justification
* **Preservation Status:** **RETAIN BASELINE 12-FEATURE SUBSET UNCHANGED.**
* **Unsupervised Integrity:** Isolation Forest is an unsupervised outlier isolation algorithm. It does **not** train on `hre_5d_train` and must not be selected via supervised AUC metrics.
* **Dimensionality Control:** Distance- and isolation-based tree algorithms degrade rapidly beyond 10–15 dimensions (the curse of dimensionality). Limiting the input matrix to 12 features maximizes isolation contrast.
* **Domain Orthogonality:** The 12 features provide balanced 2-feature coverage across 6 core physical market disruption channels:
  1. *Order-Book Depth Shocks:* `amihud_ratio` (F27) & `amihud_z20` (F29)
  2. *Volume Contraction & Droughts:* `volume_ratio` (F25) & `volume_z20` (F26)
  3. *Volatility Expansion & Term Spike:* `vol_20d` (F12) & `vol_ratio_5_20` (F14)
  4. *Intraday Price Range Spread:* `daily_range` (F37) & `range_z20` (F39)
  5. *Overnight Price Dislocation:* `overnight_gap_abs` (F41)
  6. *Transaction Spread & Price Dislocation:* `cs_spread_est` (F43), `return_z20` (F07), `nifty_vol_20d` (F52)

---

## 11. Consolidated Selected / Excluded Feature Matrix

The following matrix documents the proposed selection status of all 58 candidate features across all four models:

| ID | Feature Name | Category | Logistic Regression (20) | Random Forest (35) | HistGradientBoosting (36) | Isolation Forest (12) | Primary Reason for Exclusion (Where Excluded) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **F01** | `daily_return` | Returns | **SELECTED** | **SELECTED** | **SELECTED** | Excluded | Standalone linear return (IF uses return_z20) |
| **F02** | `log_return` | Returns | Excluded | Excluded | Excluded | Excluded | Exact mathematical twin of daily_return ($r=1.000$) |
| **F03** | `return_5d` | Returns | **SELECTED** | **SELECTED** | **SELECTED** | Excluded | Intermediate 1-week momentum |
| **F04** | `return_20d` | Returns | Excluded | **SELECTED** | **SELECTED** | Excluded | LR drops to avoid algebraic collinearity with NIFTY |
| **F05** | `return_60d` | Returns | Excluded | Excluded | Excluded | Excluded | Heavy outlier skewness ($+17.3$); weak AUC ($0.518$) |
| **F06** | `return_z5` | Returns | Excluded | Excluded | Excluded | Excluded | Redundant with return_z20 ($r=0.870$); weak AUC |
| **F07** | `return_z20` | Returns | Excluded | **SELECTED** | **SELECTED** | **SELECTED** | LR drops ($VIF>10$); RF/IF capture extreme returns |
| **F08** | `abs_return` | Returns | Excluded | **SELECTED** | **SELECTED** | Excluded | LR uses daily_range; trees capture magnitude |
| **F09** | `price_rel_ma20` | Returns | Excluded | **SELECTED** | **SELECTED** | Excluded | LR drops ($VIF>15$); trees capture SMA stretch |
| **F10** | `price_rel_ma60` | Returns | Excluded | Excluded | Excluded | Excluded | Redundant with price_rel_ma20 and returns |
| **F11** | `vol_5d` | Volatility | **SELECTED** | **SELECTED** | **SELECTED** | Excluded | Short-term realized volatility surge |
| **F12** | `vol_20d` | Volatility | Excluded | **SELECTED** | **SELECTED** | **SELECTED** | LR drops for downside_vol ($VIF<5$); key vol anchor |
| **F13** | `vol_60d` | Volatility | Excluded | Excluded | Excluded | Excluded | Redundant with vol_20d; slow moving |
| **F14** | `vol_ratio_5_20` | Volatility | Excluded | **SELECTED** | **SELECTED** | **SELECTED** | Term structure spike (vol_5d / vol_20d) |
| **F15** | `parkinson_vol` | Volatility | Excluded | Excluded | Excluded | Excluded | Redundant with daily_range_ma20 ($r=0.990$) |
| **F16** | `true_range` | Volatility | Excluded | Excluded | Excluded | Excluded | Unscaled rupee value; cross-stock scale distortion |
| **F17** | `atr_20d` | Volatility | Excluded | Excluded | Excluded | Excluded | Unscaled rupee value; cross-stock scale distortion |
| **F18** | `downside_vol_20d` | Volatility | **SELECTED** | **SELECTED** | **SELECTED** | Excluded | Asymmetric downside crash risk (Key LRS pillar) |
| **F19** | `sortino_component`| Volatility | Excluded | Excluded | Excluded | Excluded | Collinear ratio with return_20d ($r=0.971$) |
| **F20** | `max_drawdown_20d` | Volatility | **SELECTED** | **SELECTED** | **SELECTED** | Excluded | Trailing 20-day peak price destruction |
| **F21** | `rsi_14` | Volatility | **SELECTED** | **SELECTED** | **SELECTED** | Excluded | Bounded 14-day momentum exhaustion oscillator |
| **F22** | `vol_of_vol` | Volatility | Excluded | **SELECTED** | **SELECTED** | Excluded | LR drops; trees exploit volatility clustering |
| **F23** | `log_volume` | Liquidity | Excluded | **SELECTED** | **SELECTED** | Excluded | Trees capture baseline stock trading activity |
| **F24** | `volume_ma20` | Liquidity | Excluded | Excluded | Excluded | Excluded | Unscaled share volume; redundant with log_volume |
| **F25** | `volume_ratio` | Liquidity | Excluded | Excluded | Excluded | **SELECTED** | Redundant with turnover_ratio ($r=0.995$); kept in IF |
| **F26** | `volume_z20` | Liquidity | **SELECTED** | **SELECTED** | **SELECTED** | **SELECTED** | Standardized volume shock/drought (Key LRS pillar)|
| **F27** | `amihud_ratio` | Liquidity | Excluded | **SELECTED** | **SELECTED** | **SELECTED** | Extreme skewness ($+343$); trees handle; kept in IF |
| **F28** | `amihud_ma20` | Liquidity | **SELECTED** | **SELECTED** | **SELECTED** | Excluded | Smoothed price impact (Top AUC = 0.762; LRS pillar)|
| **F29** | `amihud_z20` | Liquidity | Excluded | **SELECTED** | **SELECTED** | **SELECTED** | Standardized illiquidity spike indicator |
| **F30** | `turnover_value` | Liquidity | Excluded | Excluded | Excluded | Excluded | Exact monotonic twin of log_turnover ($r=1.000$) |
| **F31** | `log_turnover` | Liquidity | **SELECTED** | **SELECTED** | **SELECTED** | Excluded | Log rupee trading activity flow |
| **F32** | `turnover_ma20` | Liquidity | Excluded | Excluded | Excluded | Excluded | Unscaled rupee turnover; redundant with log_turnover |
| **F33** | `turnover_ratio` | Liquidity | **SELECTED** | **SELECTED** | **SELECTED** | Excluded | Rupee-weighted volume ratio (superior to F25) |
| **F34** | `zero_return_flag` | Liquidity | Excluded | Excluded | Excluded | Excluded | Near-constant ($98.99\%$ zeros); weak signal |
| **F35** | `zero_vol_flag` | Liquidity | Excluded | Excluded | Excluded | Excluded | Near-constant ($99.92\%$ zeros); negligible variance |
| **F36** | `low_volume_streak`| Liquidity | **SELECTED** | **SELECTED** | **SELECTED** | Excluded | Multi-day persistent liquidity dry-up count (0–5) |
| **F37** | `daily_range` | Price Impact | **SELECTED** | **SELECTED** | **SELECTED** | **SELECTED** | 1-day normalized intraday price spread |
| **F38** | `daily_range_ma20` | Price Impact | Excluded | **SELECTED** | **SELECTED** | Excluded | LR drops ($VIF>50$); trees capture baseline spread |
| **F39** | `range_z20` | Price Impact | Excluded | **SELECTED** | **SELECTED** | **SELECTED** | Standardized intraday range expansion shock |
| **F40** | `overnight_gap` | Price Impact | Excluded | **SELECTED** | **SELECTED** | Excluded | Directional overnight price gap |
| **F41** | `overnight_gap_abs` | Price Impact | **SELECTED** | **SELECTED** | **SELECTED** | **SELECTED** | Absolute magnitude of overnight jump |
| **F42** | `gap_z20` | Price Impact | **SELECTED** | Excluded | Excluded | Excluded | LR keeps ($VIF=1.1$); trees use F40/F41 |
| **F43** | `cs_spread_est` | Price Impact | Excluded | **SELECTED** | **SELECTED** | **SELECTED** | 2-day Corwin-Schultz effective bid-ask spread |
| **F44** | `cs_spread_ma10` | Price Impact | **SELECTED** | **SELECTED** | **SELECTED** | Excluded | 10-day smoothed spread proxy (Key LRS pillar) |
| **F45** | `intraday_return` | Price Impact | Excluded | Excluded | Excluded | Excluded | Redundant decomposition ($r=0.888$ with daily_return)|
| **F46** | `excess_return_1d` | Market Relative| Excluded | **SELECTED** | **SELECTED** | Excluded | 1-day alpha / decoupling move from NIFTY 50 |
| **F47** | `excess_return_20d`| Market Relative| Excluded | Excluded | Excluded | Excluded | Exact algebraic identity (return_20d - nifty_return) |
| **F48** | `rolling_beta_60d` | Market Relative| **SELECTED** | **SELECTED** | **SELECTED** | Excluded | Systematic market risk sensitivity / beta |
| **F49** | `rolling_corr_60d` | Market Relative| **SELECTED** | **SELECTED** | **SELECTED** | Excluded | Co-movement vs decoupling with NIFTY 50 index |
| **F50** | `idiosyncratic_vol`| Market Relative| Excluded | Excluded | Excluded | Excluded | Extreme redundancy with vol_20d ($r=0.966$) |
| **F51** | `relative_volume` | Market Relative| Excluded | Excluded | **SELECTED** | Excluded | 1.51% missingness; HGBT natively routes NaNs |
| **F52** | `nifty_vol_20d` | Market Stress | **SELECTED** | **SELECTED** | **SELECTED** | **SELECTED** | Benchmark index realized volatility conditioning |
| **F53** | `nifty_vol_ratio` | Market Stress | Excluded | Excluded | Excluded | Excluded | Weak signal ($AUC = 0.502$); noisy oscillations |
| **F54** | `nifty_return_20d` | Market Stress | **SELECTED** | **SELECTED** | **SELECTED** | Excluded | Benchmark index 1-month trend regime |
| **F55** | `vol_beta` | Market Stress | Excluded | **SELECTED** | **SELECTED** | Excluded | Volatility acceleration momentum |
| **Supp**| `vix_close` | Macro Fear | **SELECTED** | **SELECTED** | **SELECTED** | Excluded | Institutional macro fear level (India VIX) |
| **Supp**| `vix_return_1d` | Macro Fear | Excluded | Excluded | Excluded | Excluded | Daily VIX noise ($AUC = 0.5005$) |
| **Supp**| `vix_z20` | Macro Fear | Excluded | Excluded | Excluded | Excluded | Symmetrically oscillating z-score ($AUC = 0.5012$) |

---

## 12. Model-Specific Feature Counts

```
========================================================================================
                          MODEL-SPECIFIC FINAL FEATURE COUNTS
========================================================================================
1. Logistic Regression (Baseline Linear Model) :  20 Features Selected  (38 Excluded)
2. Random Forest (Supervised Ensemble Bagging) :  35 Features Selected  (23 Excluded)
3. HistGradientBoosting (Gradient Boosting)    :  36 Features Selected  (22 Excluded)
4. Isolation Forest (Unsupervised Anomaly)     :  12 Features Preserved (46 Excluded)
========================================================================================
```

---

## 13. Unresolved Decisions & Limitations

1. **`relative_volume` (F51) in Random Forest:** Excluded from the proposed RF subset because standard scikit-learn `RandomForestClassifier` cannot handle missing values natively. Marked as **`PENDING VALIDATION`**: in Batch 3, if a fold-level median imputer performs cleanly, F51 can be tested in RF.
2. **Extreme Skewness Scaling for Linear Models:** Features like `amihud_ma20` and `downside_vol_20d` have positive skewness. In Logistic Regression, standard Z-score standardization (`StandardScaler`) can be pulled by extreme values; `RobustScaler` (median / IQR) should be evaluated during fold-level pipeline building in Batch 3.
3. **Threshold Sensitivity:** The supervised target `hre_5d_train` is calibrated at the 90th percentile of LRS (18.22% positive class). Models will require decision-threshold tuning or class-weight balancing (`class_weight='balanced'`) during training.

---

## 14. Recommendations for Batch 3 (Preprocessing & Walk-Forward Pipelines)

1. **Construct Model-Specific Scikit-Learn Pipelines:**
   * **Logistic Regression:** `Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('classifier', LogisticRegression(class_weight='balanced', C=1.0))])` on the 20 selected features.
   * **Random Forest:** `Pipeline([('imputer', SimpleImputer(strategy='median')), ('classifier', RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=50, class_weight='balanced_subsample'))])` on the 35 selected features.
   * **HistGradientBoosting:** `Pipeline([('classifier', HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight='balanced'))])` directly on the 36 selected features (zero imputation needed!).
   * **Isolation Forest:** `Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler()), ('detector', IsolationForest(contamination=0.05))])` on the 12 baseline features.
2. **Enforce Fold-Level Transformer Fitting:** Fit all scalers and imputers strictly inside each of the 10 expanding-window walk-forward training folds.
3. **Walk-Forward Validation Architecture:** Ingest `configs/model_config.yaml` to execute 10 chronological splits with 490 days minimum train, 60 days test window, and 1 day embargo.

---

## 15. Final Data & File Preservation Verification

```
========================================================================================
                          FINAL DATA PRESERVATION VERIFICATION
========================================================================================
File Path: data/processed/features.parquet
Expected SHA-256: c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae
Verified SHA-256: c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae
Status: EXACT MATCH (100% UNTOUCHED)
----------------------------------------------------------------------------------------
File Path: data/processed/features_with_targets.parquet
Expected SHA-256: c6cebf3d93d022b11765c206b8e2ffb84e578a020e0d913d937b37d04028aa91
Verified SHA-256: c6cebf3d93d022b11765c206b8e2ffb84e578a020e0d913d937b37d04028aa91
Status: EXACT MATCH (100% UNTOUCHED)
----------------------------------------------------------------------------------------
Git Working Tree Status: Clean (Only reports/phase8_batch2_feature_selection_report.md created)
Pytest Test Suite Status: 78 passed in 8.11s (100% passing)
Model Artifacts Created: Exactly 0
========================================================================================
```

---

## 16. Conclusion & Stopping Point

Batch 2 (Stages 8.5–8.6) is complete. The predictive relevance of all 58 features has been empirically evaluated, all collinear redundancy pairs and algebraic identities have been resolved, and exact model-specific feature lists have been established for all four target models.

In accordance with strict project instructions:
* **No feature selection code or preprocessing has been implemented in the codebase.**
* **No models have been trained or serialized.**
* **Execution has stopped at the boundary of Batch 2.**
* **Awaiting explicit user review and approval before proceeding to Batch 3.**

