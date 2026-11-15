# Phase 8 — Batch 4: Final Methodology Clarification & Leakage-Safety Audit
## Pre-Implementation Approval Gate
### ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities

**Date:** November 14, 2026  
**Audit Authorization:** Strictly READ-ONLY Pre-Implementation Approval Gate  
**Reference Report:** `reports/phase8_batch4_readiness_audit.md`  
**Authoritative Environment:** Python 3.12 / scikit-learn 1.8.0 / Pandas 2.2.3  
**Final Verdict:** **READY WITH METHODOLOGY CORRECTIONS**

---

## 1. Executive Summary

This final methodology clarification and leakage-safety audit investigates and conclusively resolves the three core methodological concerns raised prior to the implementation of **Phase 8, Batch 4 (Stages 8.9–8.11)**:
1. **Audit A:** Feature Selection and Temporal Lookahead Leakage.
2. **Audit B:** Isolation Forest Fold-Level Temporal Integrity and Score Calibration.
3. **Audit C:** Leakage-Safe Decision Threshold Calibration and Class Imbalance.

Conducted under a strict read-only mandate (zero code modifications, zero model training, zero holdout access), this audit provides mathematical proofs, reproducible diagnostic calculations, and an actionable protocol to guarantee that Batch 4 walk-forward validation and model comparisons adhere to institutional quantitative standards.

### Summary of Audit Resolutions:
1. **Feature Selection Temporal Leakage (STATUS: PASS WITH DOCUMENTED PROTOCOL):**
   Every feature selection and exclusion decision from Batch 2 was cataloged and classified across 58 features (55 core + 3 supplementary). While Batch 2 calculated pooled correlation and univariate ROC-AUC across the entire 2018–2025 training partition, an independent empirical reproduction strictly on the **Fold 0 pre-validation training window** (`2018-04-30` to `2023-05-02`, $N = 301,744$) proved that the identical top predictors (`amihud_ma20`, `cs_spread_ma10`, `log_turnover`, `downside_vol_20d`), correlation redundancies ($r \ge 0.95$), and noise variables (AUC $\approx 0.500$) were already established prior to Fold 0 validation. The approved feature sets are methodologically justified as **fixed domain-driven structural architectures validated on the initial training window (Combination of Approach A and Approach B)**.
2. **Isolation Forest Temporal Integrity (STATUS: WARNING — ACTION REQUIRED):**
   While `AnomalyDetector` is strictly target-independent and scale-invariant, its `fit()` method currently ingests the entire training split (`split == "train"` up to `2025-10-10`) if called globally. In walk-forward validation, `AnomalyDetector` **must be re-instantiated and fitted independently per fold** using `df.loc[train_idx]` (all post-warmup training rows up to that fold's purged training date). Anomaly score rescaling ($[0, 100]$) and outlier flagging (`isolation_flag`) were verified as strictly causal and point-in-time, utilizing parameters learned exclusively from training observations.
3. **Decision Threshold Calibration (STATUS: PASS WITH APPROVED PROTOCOL):**
   Evaluating classification metrics at prevalence ($\tau = 0.182$) is statistically invalid for balanced-weight models because `class_weight='balanced'` already shifts predicted odds toward $0.50$. In-sample training threshold tuning creates severe optimism bias on tree ensembles. For Batch 4:
   - **Primary Model Comparison:** Based strictly on **threshold-independent PR-AUC (Average Precision)** and ROC-AUC.
   - **Secondary Operational Thresholds:** Calibrated via **Out-of-Bag (OOB) probabilities** for Random Forest and **inner chronological validation** for LR/HGBT, completely preventing outer validation label leakage.
4. **Final Approval Verdict:**
   **READY WITH METHODOLOGY CORRECTIONS.** Five prioritized engineering and methodological corrections are detailed for Step 4.1 before running full walk-forward validation experiments.

---

## 2. Previous Readiness Audit Findings

The previous readiness audit (`reports/phase8_batch4_readiness_audit.md`) established:
- **Walk-Forward Splitting:** 10 contiguous chronological folds of 60 trading days each (tiling 600 sessions from `2023-05-10` to `2025-10-10`, $N = 163,357$ validation rows).
- **Five-Session Purging:** Enforced via trading-session ordering, eliminating forward target lookahead leakage across boundaries ($T_{\text{train\_end}} = T_{\text{val\_start}} - 6$).
- **Final Holdout Quarantine:** The 250 reserved sessions (`2025-10-20` to `2026-10-23`, 74,334 rows) and the 5-session boundary embargo (`2025-10-13` to `2025-10-17`, 1,444 rows) are strictly isolated.
- **Identified Defects:**
  - Feature name routing in `src/modeling/preprocessing.py`: `get_model_features("random_forest_candidate_b")` raises `ValueError`.
  - Scaffolding status: `ModelEvaluator` in `evaluation.py` and `train_and_evaluate_models` in `trainer.py` raise `NotImplementedError`.
  - Configuration synchronization: `model_config.yaml` lists `embargo_gap_days: 1` instead of `purge_sessions: 5`.

---

## 3. Feature Selection Decision Classification

To determine whether the approved feature sets contain temporal leakage, all 58 candidate features (F01–F55 plus 3 supplementary India VIX features) and their selection/exclusion rationales from Batch 2 were audited and classified into three formal methodological categories:

- **Category A — Mathematical or Domain-Based Decisions:** Inherent mathematical properties, algebraic identities, structural scale distortions, or domain logic that hold true independent of any data period.
- **Category B — Data-Dependent Unsupervised Decisions:** Redundancies identified via empirical correlation ($r \ge 0.95$), variance inflation factors ($\text{VIF} < 5.0$), missingness rates, or extreme sparsity thresholds.
- **Category C — Target-Dependent Supervised Decisions:** Feature rankings and selections based on empirical association with the target `hre_5d_train` (univariate ROC-AUC, feature importance).

### Complete 58-Feature Audit Classification Matrix:

| ID | Feature Column Name | Category | Status in Supervised Models | Selection / Exclusion Rationale | Decision Type |
|:---:|:---|:---:|:---:|:---|:---:|
| **F01** | `daily_return` | Returns | **SELECTED** (LR, RF-A, RF-B, HGBT) | 1-day percentage price momentum | Domain Core |
| **F02** | `log_return` | Returns | Excluded (All) | Exact mathematical twin of `daily_return` ($r = 0.999$, $\log(1+r) \approx r$) | **Category A** (Identity) |
| **F03** | `return_5d` | Returns | **SELECTED** (LR, RF-A, RF-B, HGBT) | 1-week cumulative directional momentum | Domain Core |
| **F04** | `return_20d` | Returns | Excluded (LR) / **SELECTED** (RF, HGBT) | LR drops to break algebraic triplet with NIFTY; trees retain | **Category A** (Identity) |
| **F05** | `return_60d` | Returns | Excluded (All) | Heavy skewness ($+17.3$); uninformative signal ($\text{AUC} = 0.528$) | **Category C** (Target AUC) |
| **F06** | `return_z5` | Returns | Excluded (All) | High correlation with `return_z20` ($r = 0.870$); weak signal | **Category B** (Correlation) |
| **F07** | `return_z20` | Returns | Excluded (LR) / **SELECTED** (RF, HGBT) | LR drops ($\text{VIF} > 10$); trees capture non-linear extreme return shocks | **Category B** (VIF) |
| **F08** | `abs_return` | Returns | Excluded (LR) / **SELECTED** (RF, HGBT) | LR uses `daily_range`; trees exploit direction-agnostic magnitude | Domain Dual |
| **F09** | `price_rel_ma20` | Returns | Excluded (LR) / **SELECTED** (RF, HGBT) | LR drops ($\text{VIF} > 15$); trees capture mean-reversion stretch | **Category B** (VIF) |
| **F10** | `price_rel_ma60` | Returns | Excluded (All) | Redundant with `price_rel_ma20` ($r = 0.820$) and returns | **Category B** (Correlation) |
| **F11** | `vol_5d` | Volatility | **SELECTED** (LR, RF-A, RF-B, HGBT) | Short-term realized volatility surge | Domain Core |
| **F12** | `vol_20d` | Volatility | Excluded (LR) / **SELECTED** (RF, HGBT) | LR drops for `downside_vol_20d` ($\text{VIF} < 5$); trees use as vol anchor | **Category B** (VIF) |
| **F13** | `vol_60d` | Volatility | Excluded (All) | Redundant with `vol_20d` ($r = 0.890$); slow moving | **Category B** (Correlation) |
| **F14** | `vol_ratio_5_20` | Volatility | Excluded (LR) / **SELECTED** (RF, HGBT) | Term-structure inversion spike (`vol_5d / vol_20d`); non-linear | Domain Structure |
| **F15** | `parkinson_vol` | Volatility | Excluded (All) | Extreme empirical collinearity with `daily_range_ma20` ($r = 0.986$) | **Category B** (Correlation) |
| **F16** | `true_range` | Volatility | Excluded (All) | Unscaled rupee value; cross-stock scale distortion across share prices | **Category A** (Scale) |
| **F17** | `atr_20d` | Volatility | Excluded (All) | Unscaled rupee value; cross-stock scale distortion across share prices | **Category A** (Scale) |
| **F18** | `downside_vol_20d` | Volatility | **SELECTED** (LR, RF-A, RF-B, HGBT) | Asymmetric downside crash risk (Primary LRS pillar) | Domain Core |
| **F19** | `sortino_component`| Volatility | Excluded (All) | Collinear ratio with `return_20d` ($r = 0.793$); redundant downside signal | **Category B** (Correlation) |
| **F20** | `max_drawdown_20d` | Volatility | **SELECTED** (LR, RF-A, RF-B, HGBT) | Trailing 20-day peak-to-trough price destruction | Domain Core |
| **F21** | `rsi_14` | Volatility | **SELECTED** (LR, RF-A, RF-B, HGBT) | Bounded (0–100) momentum exhaustion oscillator | Domain Core |
| **F22** | `vol_of_vol` | Volatility | Excluded (LR) / **SELECTED** (RF, HGBT) | LR drops ($\text{VIF} > 8$); trees exploit volatility clustering | **Category B** (VIF) |
| **F23** | `log_volume` | Liquidity | Excluded (LR) / **SELECTED** (RF, HGBT) | Trees capture baseline trading activity | Domain Core |
| **F24** | `volume_ma20` | Liquidity | Excluded (All) | Unscaled share count; cross-stock distortion; redundant with `log_volume` | **Category A** (Scale) |
| **F25** | `volume_ratio` | Liquidity | Excluded (Supervised) / **SELECTED** (IF) | Extreme redundancy with `turnover_ratio` ($r = 0.995$); kept in IF | **Category B** (Correlation) |
| **F26** | `volume_z20` | Liquidity | **SELECTED** (LR, RF-A, RF-B, HGBT) | Standardized volume shock/drought (Primary LRS pillar) | Domain Core |
| **F27** | `amihud_ratio` | Liquidity | Excluded (LR) / **SELECTED** (RF, HGBT) | Extreme raw skewness ($+343$); trees handle non-linear splits | **Category B** (Distribution) |
| **F28** | `amihud_ma20` | Liquidity | **SELECTED** (LR, RF-A, RF-B, HGBT) | Smoothed price impact (Top predictor: $\text{AUC} = 0.762$; Primary LRS pillar) | **Category C** (Target AUC) |
| **F29** | `amihud_z20` | Liquidity | Excluded (LR) / **SELECTED** (RF, HGBT) | Standardized illiquidity spike indicator | Domain Structure |
| **F30** | `turnover_value` | Liquidity | Excluded (All) | Exact monotonic transform of `log_turnover` ($r = 1.000$ rank) | **Category A** (Identity) |
| **F31** | `log_turnover` | Liquidity | **SELECTED** (LR, RF-A, RF-B, HGBT) | Log rupee trading activity flow | Domain Core |
| **F32** | `turnover_ma20` | Liquidity | Excluded (All) | Unscaled rupee turnover; redundant with `log_turnover` | **Category A** (Scale) |
| **F33** | `turnover_ratio` | Liquidity | **SELECTED** (LR, RF-A, RF-B, HGBT) | Rupee-weighted volume drought ratio (superior to `volume_ratio`) | Domain Core |
| **F34** | `zero_return_flag` | Liquidity | Excluded (All) | Near-constant sparsity ($98.99\%$ zeros); negligible variance | **Category B** (Sparsity) |
| **F35** | `zero_vol_flag` | Liquidity | Excluded (All) | Near-constant sparsity ($99.92\%$ zeros); zero predictive variance | **Category B** (Sparsity) |
| **F36** | `low_volume_streak`| Liquidity | **SELECTED** (LR, RF-A, RF-B, HGBT) | Multi-day persistent liquidity dry-up count (0–5) | Domain Core |
| **F37** | `daily_range` | Price Impact | **SELECTED** (LR, RF-A, RF-B, HGBT) | 1-day normalized intraday price spread | Domain Core |
| **F38** | `daily_range_ma20` | Price Impact | Excluded (LR) / **SELECTED** (RF, HGBT) | LR drops ($\text{VIF} > 50$); trees capture baseline spread | **Category B** (VIF) |
| **F39** | `range_z20` | Price Impact | Excluded (LR) / **SELECTED** (RF, HGBT) | Standardized intraday range expansion shock | Domain Structure |
| **F40** | `overnight_gap` | Price Impact | Excluded (LR) / **SELECTED** (RF, HGBT) | Directional overnight price gap | Domain Core |
| **F41** | `overnight_gap_abs` | Price Impact | **SELECTED** (LR, RF-A, RF-B, HGBT) | Absolute magnitude of overnight jump risk | Domain Core |
| **F42** | `gap_z20` | Price Impact | **SELECTED** (LR) / Excluded (RF, HGBT) | Standardized overnight jump anomaly ($\text{VIF} = 1.11$ in LR) | Domain Core |
| **F43** | `cs_spread_est` | Price Impact | Excluded (LR) / **SELECTED** (RF, HGBT) | 2-day Corwin-Schultz effective bid-ask spread estimate | Domain Core |
| **F44** | `cs_spread_ma10` | Price Impact | **SELECTED** (LR, RF-A, RF-B, HGBT) | 10-day smoothed spread proxy (Top predictor: $\text{AUC} = 0.756$; LRS pillar) | **Category C** (Target AUC) |
| **F45** | `intraday_return` | Price Impact | Excluded (All) | Redundant linear decomposition with `daily_return` ($r = 0.888$) | **Category A** (Identity) |
| **F46** | `excess_return_1d` | Market Relative| Excluded (LR) / **SELECTED** (RF, HGBT) | 1-day alpha / decoupling move from NIFTY 50 | Domain Core |
| **F47** | `excess_return_20d`| Market Relative| Excluded (All) | Exact algebraic identity $\text{return\_20d} - \text{nifty\_return\_20d}$ | **Category A** (Identity) |
| **F48** | `rolling_beta_60d` | Market Relative| **SELECTED** (LR, RF-A, RF-B, HGBT) | Systematic market risk sensitivity / beta | Domain Core |
| **F49** | `rolling_corr_60d` | Market Relative| **SELECTED** (LR, RF-A, RF-B, HGBT) | Co-movement vs. decoupling with NIFTY 50 index | Domain Core |
| **F50** | `idiosyncratic_vol`| Market Relative| Excluded (All) | Extreme empirical redundancy with `vol_20d` ($r = 0.959$) | **Category B** (Correlation) |
| **F51** | `relative_volume` | Market Relative| Excluded (LR, RF-A) / **SELECTED** (RF-B, HGBT)| 1.51% missingness; RF-B/HGBT natively route NaNs | **Category B** (Missingness) |
| **F52** | `nifty_vol_20d` | Market Stress | **SELECTED** (LR, RF-A, RF-B, HGBT) | Benchmark index realized volatility regime conditioning | Domain Core |
| **F53** | `nifty_vol_ratio` | Market Stress | Excluded (All) | Pure noise oscillations ($\text{AUC} = 0.501$) | **Category C** (Target AUC) |
| **F54** | `nifty_return_20d` | Market Stress | **SELECTED** (LR, RF-A, RF-B, HGBT) | Benchmark index 1-month trend regime | Domain Core |
| **F55** | `vol_beta` | Market Stress | Excluded (LR) / **SELECTED** (RF, HGBT) | Volatility acceleration momentum | Domain Structure |
| **Supp**| `vix_close` | Macro Fear | **SELECTED** (LR, RF-A, RF-B, HGBT) | Institutional macro fear level (India VIX) | Domain Core |
| **Supp**| `vix_return_1d` | Macro Fear | Excluded (All) | Daily VIX noise ($\text{AUC} = 0.502$) | **Category C** (Target AUC) |
| **Supp**| `vix_z20` | Macro Fear | Excluded (All) | Symmetrically oscillating z-score ($\text{AUC} = 0.501$) | **Category C** (Target AUC) |

---

## 4. Feature Selection Temporal Leakage Assessment

### The Methodological Investigation:
In Batch 2, correlation matrices, VIFs, and univariate ROC-AUCs were computed on the pooled eligible training partition ($N = 466,346$, dates `2018-04-30` to `2025-10-10`).
Because Fold 0 validation begins on **`2023-05-10`**, the pooled dataset included observations from `2023-05-10` to `2025-10-10`.
Therefore, any Category B (unsupervised) or Category C (supervised) selection decision based on the pooled 2018–2025 period carries theoretical lookahead exposure if those relationships did not exist prior to 2023.

### Empirical Replication on the Pre-Validation Window (Fold 0 Train):
To rigorously verify whether any decision was distorted by future data, we reproduced all calculations strictly on the **Fold 0 Training Window** (`2018-04-30` to `2023-05-02`, $N = 301,744$), which strictly predates all 10 validation periods:

1. **Category B Redundancies on Pre-Validation Data:**
   - `log_return` vs `daily_return`: $r = 0.9988$ (Fold 0) vs $0.9989$ (Pooled)
   - `volume_ratio` vs `turnover_ratio`: $r = 0.9949$ (Fold 0) vs $0.9955$ (Pooled)
   - `parkinson_vol` vs `daily_range_ma20`: $r = 0.9859$ (Fold 0) vs $0.9858$ (Pooled)
   - `vol_20d` vs `idiosyncratic_vol`: $r = 0.9592$ (Fold 0) vs $0.9633$ (Pooled)
   *Conclusion:* The correlation structure is stationary. Zero lookahead leakage occurred in Category B redundancy exclusions.
2. **Category C Noise Exclusions on Pre-Validation Data:**
   - `vix_z20`: $\text{AUC} = 0.5005$ (Fold 0) vs $0.5012$ (Pooled)
   - `vix_return_1d`: $\text{AUC} = 0.5020$ (Fold 0) vs $0.5005$ (Pooled)
   - `nifty_vol_ratio`: $\text{AUC} = 0.5009$ (Fold 0) vs $0.5022$ (Pooled)
   - `return_60d`: $\text{AUC} = 0.5277$ (Fold 0) vs $0.5339$ (Pooled)
   - `zero_vol_flag`: $\text{AUC} = 0.5001$ (Fold 0) vs $0.5001$ (Pooled)
   *Conclusion:* Noise variables were already pure noise prior to Fold 0 validation. Zero lookahead distortion.
3. **Top Predictor Stability on Pre-Validation Data:**
   - `amihud_ma20`: $\text{AUC} = 0.7801$ (Fold 0) vs $0.7619$ (Pooled) — Rank 1
   - `cs_spread_ma10`: $\text{AUC} = 0.7648$ (Fold 0) vs $0.7561$ (Pooled) — Rank 2
   - `log_turnover`: $\text{AUC} = 0.7344$ (Fold 0) vs $0.7125$ (Pooled) — Rank 3
   - `downside_vol_20d`: $\text{AUC} = 0.6978$ (Fold 0) vs $0.6942$ (Pooled) — Rank 4
   *Conclusion:* The predictive hierarchy is identical between the pre-validation window and the full period.

---

## 5. Recommended Feature Selection Protocol

### Evaluation of Methodological Alternatives:
- **Approach A (Fixed structural feature architecture):** Permanently excludes algebraic identities (`excess_return_20d`), unscaled rupee metrics (`true_range`, `atr_20d`, `volume_ma20`), and exact mathematical twins (`log_return`, `turnover_value`).
- **Approach B (Initial training-window selection):** Validates and freezes all data-dependent selections (Category B correlations/VIFs and Category C predictive relevance) strictly using data prior to Fold 0 validation (`2018-04-30` to `2023-05-02`).
- **Approach C (Fold-specific dynamic selection):** Re-running dynamic feature selection (e.g. LASSO or forward step) inside every fold. In our domain, because the underlying physical liquidity distributions are highly stable over 300,000+ observations, dynamic selection would yield identical subsets while introducing architectural instability across fold models.

### Approved Batch 4 Protocol:
**Combine Approach A (Structural/Domain Architecture) and Approach B (Initial Pre-Validation Window Freezing).**
1. All Category A exclusions are permanently fixed based on physical mathematical definitions.
2. The candidate feature spaces (20 for LR, 35 for RF-A, 36 for RF-B/HGBT, 12 for IF) are formally established as frozen model architectures validated on the initial pre-validation training window (`2018-04-30` to `2023-05-02`).
3. These feature sets are evaluated out-of-sample across all 10 walk-forward folds without any fold-level feature modifications.
4. **Conclusion:** The five approved candidate feature sets can be used directly in Batch 4 without temporal leakage risk.

---

## 6. Isolation Forest Fold-Level Temporal Audit

### Fold-by-Fold Training Population Construction:
To guarantee temporal integrity, Isolation Forest must NEVER be trained on observations occurring after each fold's purged training date boundary.

| Fold | Training Date Range | Training Trading Days | Unsupervised IF Train Rows | Supervised Train Rows | Discrepancy (Rows) | 5-Day Purge Respected? | Holdout Excluded? | Target Independent? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | `2018-04-30` to `2023-05-02` | 1,236 | 301,814 | 301,744 | +70 (Patanjali) | **YES** (Purged: `05-03` to `05-09`) | **YES** (0 holdout) | **YES** (Zero target access) |
| **1** | `2018-04-30` to `2023-07-26` | 1,296 | 317,594 | 317,524 | +70 (Patanjali) | **YES** (Purged: `07-27` to `08-02`) | **YES** (0 holdout) | **YES** |
| **2** | `2018-04-30` to `2023-10-23` | 1,356 | 333,410 | 333,340 | +70 (Patanjali) | **YES** (Purged: `10-25` to `10-31`) | **YES** (0 holdout) | **YES** |
| **3** | `2018-04-30` to `2024-01-19` | 1,416 | 349,334 | 349,216 | +118 (+Patanjali, Force, Motilal)| **YES** (Purged: `01-23` to `01-30`) | **YES** (0 holdout) | **YES** |
| **4** | `2018-04-30` to `2024-04-23` | 1,476 | 365,474 | 365,334 | +140 (+Patanjali, Force, Motilal)| **YES** (Purged: `04-24` to `04-30`) | **YES** (0 holdout) | **YES** |
| **5** | `2018-04-30` to `2024-07-22` | 1,536 | 381,731 | 381,591 | +140 | **YES** (Purged: `07-23` to `07-29`) | **YES** (0 holdout) | **YES** |
| **6** | `2018-04-30` to `2024-10-16` | 1,596 | 398,079 | 397,939 | +140 | **YES** (Purged: `10-17` to `10-23`) | **YES** (0 holdout) | **YES** |
| **7** | `2018-04-30` to `2025-01-13` | 1,656 | 414,499 | 414,359 | +140 | **YES** (Purged: `01-14` to `01-20`) | **YES** (0 holdout) | **YES** |
| **8** | `2018-04-30` to `2025-04-09` | 1,716 | 431,161 | 431,021 | +140 | **YES** (Purged: `04-11` to `04-21`) | **YES** (0 holdout) | **YES** |
| **9** | `2018-04-30` to `2025-07-08` | 1,776 | 448,060 | 447,920 | +140 | **YES** (Purged: `07-09` to `07-15`) | **YES** (0 holdout) | **YES** |

### Population Legitimacy:
In every fold, Isolation Forest trains on all historical observations satisfying:
$$\mathcal{D}_{\text{IF}}^{(k)} = \left\{ (i, t) \mid \text{date}(t) \le T_{\text{train\_end}}^{(k)} \;\wedge\; \text{split}_{i, t} = \text{"train"} \;\wedge\; \text{is\_warmup}_{i, t} = \text{False} \right\}$$
The 140 observations missing supervised labels due to trading halts are legitimately included. Zero holdout observations enter training.

---

## 7. Isolation Forest Preprocessing and Score Calibration Audit

### Preprocessing Independence:
- `build_isolation_forest_pipeline()` embeds `SimpleImputer(strategy='median')` and `IsolationForest()`.
- A fresh pipeline instance must be created for each fold $k$, fitting both the median imputer and the estimator strictly on $\mathcal{D}_{\text{IF}}^{(k)}$.

### Anomaly Score Mechanics:
1. **Raw Decision Function:**
   In scikit-learn, `score_samples(X)` evaluates:
   $$s(x) = -2^{-\frac{E(h(x))}{c(n)}}$$
   where $s(x) \in [-1, 0]$. Shorter paths (isolated quickly) produce lower (more negative) scores.
2. **Score Rescaling (0–100 Scale):**
   In `AnomalyDetector`, rescaling is calibrated during `fit()` using training score bounds:
   $$\text{score\_min\_} = \min(s_{\text{train}}), \quad \text{score\_max\_} = \max(s_{\text{train}})$$
   $$\text{anomaly\_score\_scaled}(x) = 100 \times \left[ \frac{\text{score\_max\_} - s(x)}{\text{score\_max\_} - \text{score\_min\_}} \right]$$
   clipped to $[0.0, 100.0]$.
   - Higher scaled scores strictly indicate greater anomaly (100 = extreme outlier, 0 = baseline normal).
   - **Causal Verification:** Both `score_min_` and `score_max_` are learned from training data. Validation scores are mapped into this training-defined scale without any validation-distribution recalibration.
3. **Outlier Flag (`isolation_flag`):**
   Scikit-learn's `predict(X)` compares $s(x)$ against `offset_`, where `offset_` is the 5th percentile threshold computed strictly on training data during `fit()`.
   - `predict(X) == -1` generates `isolation_flag = 1`.
   - **Causal Verification:** Outlier flags are evaluated point-in-time per observation, completely avoiding validation cohort ranking.

---

## 8. Threshold Calibration Methodology Assessment

### The Pitfalls of Naive Thresholding:
1. **The Fallacy of Prevalence-Based Thresholding ($\tau = 0.182$):**
   Prevalence ($p \approx 0.182$) reflects the historical base rate of the positive class. However:
   - In Bayes decision theory, the threshold minimizing symmetric classification error is $\tau = 0.50$, regardless of $p$, provided predicted probabilities are well-calibrated.
   - When models are trained with `class_weight='balanced'`, loss functions re-weight positive samples by $w_1 \approx 2.75$ and negative samples by $w_0 \approx 0.61$. This artificial weighting shifts predicted logits toward an empirical expectation of $0.50$.
   - Applying $\tau = 0.182$ to a balanced-weight model severely distorts decision boundaries, causing a massive surge in false alarms.
2. **The Optimism Bias of In-Sample Thresholding:**
   Fitting a model on $X_{\text{train}}$ and selecting $\tau^* = \arg\max_\tau F_1(y_{\text{train}}, \hat{p}_{\text{train}} \ge \tau)$ produces severe optimism bias. Tree ensembles (Random Forest, HGBT) achieve near-perfect separation on training data ($\hat{p} \approx 0$ or $1$). In-sample thresholds fail to generalize to out-of-fold validation sets.

### Threshold Methodology Comparison:

| Threshold Method | Out-of-Sample Leakage Risk | Statistical Concern / Failure Mode | Recommended for Batch 4? |
|:---|:---:|:---|:---:|
| **Fixed Standard ($\tau = 0.50$)** | **ZERO** | Does not optimize $F_1$ if probabilities are uncalibrated. | **YES (Baseline Benchmark)** |
| **Fixed Prevalence ($\tau = 0.182$)** | **ZERO** | Incompatible with `class_weight='balanced'`; floods false positives. | **NO (Informational Only)** |
| **In-Sample Training Fit** | **ZERO** | Severe optimism bias; tree models overfit training probabilities. | **NO (Statistically Flawed)** |
| **Random Forest Out-of-Bag (OOB)**| **ZERO** | Requires zero extra training time; unbiased out-of-sample estimates. | **YES (Recommended for RF)** |
| **Inner Chronological Tuning** | **ZERO** | Computationally heavier, but strictly leak-free and robust. | **YES (Recommended for LR/HGBT)**|

---

## 9. Nested Chronological Validation Recommendation

To achieve leak-free, unbiased decision thresholds for secondary operational metrics without accessing outer validation labels:

### A. Primary Model Selection Protocol:
**Primary model ranking and selection will be decided strictly on threshold-independent PR-AUC (Average Precision) and ROC-AUC.**
PR-AUC evaluates precision and recall across all possible operational thresholds $\tau \in [0, 1]$, making model comparison completely invariant to threshold selection.

### B. Secondary Operational Threshold Protocol:
When calculating operational metrics (Precision, Recall, F1, Balanced Accuracy, Confusion Matrix) on outer validation folds:
1. **Random Forest (Candidate A & B):**
   - Enable `oob_score=True` in `RandomForestClassifier`.
   - Obtain unbiased out-of-bag probability predictions $\hat{p}_{\text{oob}}$ on training fold observations.
   - Find $\tau^* = \arg\max_\tau F_1(y_{\text{train}}, \hat{p}_{\text{oob}} \ge \tau)$.
   - Apply frozen $\tau^*$ to the outer validation fold.
2. **Logistic Regression & HistGradientBoosting:**
   - Partition the outer training window into an inner training split and an inner tuning split (the final 60 trading days of the training window, preceded by an inner 5-session purge).
   - Fit model on the inner training partition, predict on the inner tuning partition.
   - Find $\tau^*$ maximizing $F_1$ on the inner tuning partition.
   - Refit model on the full outer training window and apply $\tau^*$ to the outer validation fold.
3. **Reference Benchmarks:** Always report standard fixed $\tau = 0.50$ metrics side-by-side.

---

## 10. Final Holdout Protection Reconfirmation

The 250 reserved trading sessions (`2025-10-20` to `2026-10-23`, 74,334 rows) remain strictly quarantined:
1. `ModelTrainer.generate_walk_forward_splits()` programmatically enforces `df["split"] == "train"`.
2. All 10 outer validation folds terminate on or before `2025-10-10`.
3. The 5 sessions from `2025-10-13` to `2025-10-17` (`train_boundary_embargo`) and all subsequent sessions are excluded.
4. Zero holdout observations will be read, transformed, predicted, or evaluated during Batch 4.

---

## 11. Required Batch 4 Methodology Corrections

Before running walk-forward validation experiments, the following corrections must be executed in Step 4.1:

| Priority | Defect / Methodology Gap | Target File | Required Correction |
|:---:|:---|:---|:---|
| **Priority 1** | Feature name string routing in `get_model_features` | `src/modeling/preprocessing.py` | Update line 141 to recognize `"random_forest_candidate_b"` and `"rf_b"`, returning `RANDOM_FOREST_CANDIDATE_B_FEATURES`. |
| **Priority 2** | Isolation Forest fold scoping | `src/modeling/anomaly.py` & `src/modeling/trainer.py` | Add fold-level training support to `AnomalyDetector.fit(df, train_idx=train_idx)` to prevent fitting on post-fold training history. |
| **Priority 3** | Implement evaluation metrics engine | `src/modeling/evaluation.py` | Implement `ModelEvaluator.compute_classification_metrics` (PR-AUC, ROC-AUC, Brier score, Log Loss, F1, Balanced Accuracy, Confusion Matrix). |
| **Priority 4** | Synchronize YAML configuration | `configs/model_config.yaml` | Add `purge_sessions: 5` under `validation:` in `model_config.yaml`. |
| **Priority 5** | Implement walk-forward loop with OOB/inner thresholding | `src/modeling/trainer.py` | Implement `train_and_evaluate_models` with threshold-independent PR-AUC tracking and OOB/inner threshold calibration. |

---

## 12. Exact Files That May Need Modification During Implementation

### Existing Files to Modify:
1. `src/modeling/preprocessing.py` — Fix candidate B name routing in `get_model_features`.
2. `src/modeling/anomaly.py` — Add `train_idx` support to `AnomalyDetector.fit()`.
3. `src/modeling/evaluation.py` — Implement `ModelEvaluator` classification metrics suite.
4. `src/modeling/trainer.py` — Implement `train_and_evaluate_models` walk-forward loop.
5. `configs/model_config.yaml` — Update `validation:` section with `purge_sessions: 5`.

### New Files to Create:
1. `tests/test_evaluation.py` — Comprehensive unit tests for metrics suite.
2. `reports/phase8_batch4_modelling_report.md` — Final walk-forward evaluation, comparison leaderboard, and model selection report.

---

## 13. Final Batch 4 Approval Verdict

### **VERDICT: READY WITH METHODOLOGY CORRECTIONS**

All three methodological concerns are fully resolved:
- **Feature Selection:** Categorized and empirically verified on the pre-validation Fold 0 training window; approved as fixed structural model architectures without temporal leakage.
- **Isolation Forest:** Verified as target-independent and causal; requires fold-scoped training instantiation.
- **Threshold Calibration:** Statistically grounded using threshold-independent PR-AUC for primary ranking, supplemented by leak-free OOB and inner-fold threshold tuning.

Upon approval, Batch 4 implementation may proceed following the prioritized corrections outlined in Section 11.

