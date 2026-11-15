# PHASE 8 BATCH 5 | STAGE 8.12.2: MODEL-SELECTION METHODOLOGY SPECIFICATION

**Project:** ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities  
**Phase:** Phase 8 Batch 5 (Stage 8.12.2)  
**Task:** Finalize Model-Selection Methodology (READ-ONLY)  
**Target Variable:** Forward 5-Day High-Risk Event (`hre_5d_train`, Horizon $H=5$)  
**Status:** COMPLETE & FROZEN  
**Date:** 2026-11-15  

---

## 1. OBJECTIVE AND SCOPE

### 1.1 Formal Objective
The objective of Stage 8.12.2 is to formulate, specify, and freeze a rigorous, pre-defined, multi-dimensional model-selection methodology for evaluating and selecting among the supervised model configurations evaluated in Batch 4. This framework establishes deterministic, objective decision rules to govern:
1. The resolution of candidate feature sets and preprocessing pipelines within model families.
2. The selection of the primary champion supervised model architecture.
3. The confirmation of the operational role of the companion unsupervised anomaly detector.
4. The exact protocol to freeze configurations in Stage 8.12.3 and evaluate them across the unaccessed holdout dataset in Stage 8.12.5.

### 1.2 Foundational Principle: Pre-Holdout Freezing
A foundational scientific and methodological principle of this quantitative finance project is that **all model selection criteria, hierarchy tiers, trade-off thresholds, and tie-breaking protocols must be finalized strictly on walk-forward cross-validation results prior to touching the final holdout**. 

Evaluating multiple models or tuning decision thresholds on the holdout partition invalidates the holdout as an unbiased proxy for real-world production performance, introducing subtle selection bias and overfitting to the holdout period. Therefore, this methodology is codified in Stage 8.12.2 and executed in Stage 8.12.3 while the holdout dataset remains strictly quarantined.

### 1.3 Scope Boundaries & Read-Only Governance
- **Strict Read-Only Mode:** Stage 8.12.2 performs strictly read-only empirical analysis and methodological formulation. No models are trained, retrained, or tuned. No code or configuration files are modified.
- **Holdout Quarantine:** The 250-trading-session holdout partition (`split == 'holdout'`, 74,334 rows from `2025-10-20` to `2026-10-23`) remains 100% unaccessed, unscored, and uninspected.
- **Stop Gate:** Work strictly terminates upon completion and verification of this report. Stage 8.12.3 (decision execution and configuration freezing) will commence only upon subsequent authorization.

---

## 2. AUTHORITATIVE INPUT ARTIFACTS

This selection methodology is constructed from verified, reproducible empirical artifacts generated during Batch 4 and audited in Stage 8.12.1. No synthetic, extrapolated, or fabricated metrics are utilized.

### 2.1 Inventory of Verified Input Artifacts

| Artifact Name | Path | Format & Dimensions | Description & Verification Status |
| :--- | :--- | :--- | :--- |
| **Supervised Model Comparison** | `data/interim/phase8_batch4/supervised_model_comparison.csv` | CSV ($5 \times 25$) | Summary cross-validation metrics across all 10 folds for all 5 supervised configurations. Verified. |
| **Supervised Fold Metrics** | `data/interim/phase8_batch4/supervised_fold_metrics.csv` | CSV ($50 \times 40$) | Granular fold-by-fold metrics, threshold calibrations, and timing data across 10 expanding folds. Verified. |
| **Supervised Out-of-Fold Predictions** | `data/interim/phase8_batch4/supervised_predictions.parquet` | Parquet ($163,357 \times 20$) | Out-of-fold probability predictions, calibrated flags, and ground truth across all 10 folds. Verified. |
| **Isolation Forest Fold Metrics** | `data/interim/phase8_batch4/isolation_forest_fold_metrics.csv` | CSV ($10 \times 22$) | Granular telemetry, score distributions, and HRE cross-correlations across 10 folds. Verified. |
| **Isolation Forest Predictions** | `data/interim/phase8_batch4/isolation_forest_predictions.parquet` | Parquet ($163,357 \times 8$) | Unsupervised raw anomaly scores, scaled [0–100] scores, and flag indicators. Verified. |
| **Batch 4 Modelling Report** | `reports/phase8_batch4_modelling_report.md` | Markdown ($18.4$ KB) | Complete documentation of Batch 4 walk-forward execution and empirical results. Verified. |
| **Stage 8.12.1 Audit Report** | `reports/phase8_batch5_stage8_12_1_audit.md` | Markdown ($23.8$ KB) | Comprehensive read-only audit confirming PASS verdict across all 12 readiness areas. Verified. |
| **Model Configuration** | `configs/model_config.yaml` | YAML configuration | Authoritative hyperparameter, feature set, and validation parameter specifications. Verified. |

---

## 3. FIVE-CONFIGURATION COMPARISON & ARCHITECTURAL RECONCILIATION

### 3.1 Architectural Reconciliation: Families vs. Configurations
A critical methodological distinction established in Stage 8.12.1 and formalized here is that the five evaluated supervised configurations represent **three distinct supervised algorithmic families** evaluated across **five approved experimental configurations**, accompanied by **one independent unsupervised anomaly detection family**:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           PHASE 8 MODELING ARCHITECTURE                          │
└──────────────────────────────────────────────────────────────────────────────────┘
                                        │
         ┌──────────────────────────────┴──────────────────────────────┐
         │                                                             │
┌────────┴──────────────────────────┐            ┌─────────────────────┴───────────┐
│     SUPERVISED CLASSIFICATION     │            │  UNSUPERVISED ANOMALY DETECTION │
│     Target: Forward 5-Day HRE     │            │ Multivariate Microstructure Risk│
└────────┬──────────────────────────┘            └─────────────────────┬───────────┘
         │                                                             │
         ├──────────────────────────────┬──────────────────────────────┼───────────────┐
         │                              │                              │               │
┌────────┴──────────────┐      ┌────────┴─────────────┐      ┌─────────┴────────────┐  │
│ FAMILY 1: LINEAR      │      │ FAMILY 2: ENSEMBLE   │      │ FAMILY 3: BOOSTING   │  │
│ Logistic Regression   │      │ Random Forest        │      │ HistGradientBoosting │  │
│ 20 Features           │      │ 35 / 36 Features     │      │ 36 Features          │  │
└────────┬──────────────┘      └────────┬─────────────┘      └─────────┬────────────┘  │
         │                              │                              │               │
    ┌────┴────┐                    ┌────┴────┐                         │               │
    │         │                    │         │                         │               │
┌───┴────┐ ┌──┴─────┐          ┌───┴────┐ ┌──┴─────┐                   │               │
│Config 1│ │Config 2│          │Config 3│ │Config 4│               ┌───┴──────────┐ ┌──┴──────────┐
│Standard│ │Robust  │          │Cand A  │ │Cand B  │               │Config 5      │ │ISOLATION     │
│Scaler  │ │Scaler  │          │(35 feat│ │(36 feat│               │HGBT          │ │FOREST        │
│        │ │        │          │no F51) │ │with F51│               │(Champion)    │ │(12 features) │
└────────┘ └────────┘          └────────┘ └────────┘               └──────────────┘ └─────────────┘
```

### 3.2 Detailed Configuration Specification Matrix

| Parameter / Dimension | Config 1: `logistic_regression_standard` | Config 2: `logistic_regression_robust` | Config 3: `random_forest_candidate_a` | Config 4: `random_forest_candidate_b` | Config 5: `hist_gradient_boosting` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Algorithm Family** | Linear Classification | Linear Classification | Supervised Ensemble | Supervised Ensemble | Gradient Tree Boosting |
| **Feature Count** | 20 features | 20 features | 35 features | 36 features | 36 features |
| **Key Feature Differences** | Linear-screened subset | Linear-screened subset | Excludes F51 (`relative_volume`) | Includes F51 (`relative_volume`) | Includes F51 (`relative_volume`) |
| **Missing-Value Imputation** | Median Imputer (`SimpleImputer`) | Median Imputer (`SimpleImputer`) | None (Native NaN Routing) | None (Native NaN Routing) | None (Native Histogram Binning) |
| **Feature Scaling** | `StandardScaler` ($z$-score) | `RobustScaler` (Median/IQR) | None | None | None |
| **Class Imbalance Strategy** | `class_weight='balanced'` | `class_weight='balanced'` | `class_weight='balanced_subsample'` | `class_weight='balanced_subsample'` | Dynamic Sample Weighting |
| **Key Hyperparameters** | $C=1.0$, `solver='lbfgs'`, max_iter=1000 | $C=1.0$, `solver='lbfgs'`, max_iter=1000 | $N=200$, depth=12, leaf=10 | $N=200$, depth=12, leaf=10 | $N=150$, lr=0.05, leaves=31, reg=0.1 |
| **Experimental Purpose** | Linear Baseline anchor | Outlier-scaling sensitivity | Tree Ensemble baseline | Volume-shock NaN exploitation | Modern boosted non-linear benchmark |

---

## 4. PREDICTIVE-PERFORMANCE ANALYSIS

### 4.1 Definitive Empirical Leaderboard (10 Expanding Folds, $N = 163,357$)
All metrics are computed strictly across out-of-fold validation sets across the 10 chronological expanding folds (2023-05-10 to 2025-10-10) with five-session target purging:

| Rank | Model Configuration | Mean PR-AUC | Pooled PR-AUC | Mean ROC-AUC | Pooled ROC-AUC | Mean Brier Score | Mean Log Loss | F1 @ 0.50 (P / R) | Calib F1 @ $\tau^*$ (P / R) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`hist_gradient_boosting`** | **0.6107 ± 0.0229** | **0.5926** | **0.8548 ± 0.0099** | **0.8492** | **0.1266** | **0.3857** | **0.5622** (0.503 / 0.658) | **0.5389** (0.536 / 0.616) |
| **2** | `random_forest_candidate_b` | 0.5829 ± 0.0352 | 0.5610 | 0.8361 ± 0.0169 | 0.8266 | 0.1320 | 0.4131 | 0.5375 (0.508 / 0.599) | 0.5252 (0.522 / 0.579) |
| **3** | `random_forest_candidate_a` | 0.5814 ± 0.0377 | 0.5591 | 0.8346 ± 0.0177 | 0.8248 | 0.1329 | 0.4166 | 0.5339 (0.507 / 0.593) | 0.5206 (0.520 / 0.577) |
| **4** | `logistic_regression_standard` | 0.4990 ± 0.0487 | 0.4797 | 0.8265 ± 0.0187 | 0.8214 | 0.1382 | 0.4469 | 0.5227 (0.496 / 0.574) | 0.5152 (0.465 / 0.620) |
| **5** | `logistic_regression_robust` | 0.4990 ± 0.0487 | 0.4797 | 0.8264 ± 0.0187 | 0.8214 | 0.1382 | 0.4468 | 0.5227 (0.497 / 0.573) | 0.5157 (0.464 / 0.623) |

### 4.2 Primary Metric Justification: PR-AUC (Average Precision)
In market microstructure and trading-risk modeling, binary classification metrics must be evaluated through the lens of label base rates and operational trade-offs:
1. **Severe Class Imbalance:** The positive prevalence of 5-day high-risk liquidity events (`hre_5d_train = 1`) across the expanding folds is **~18.3%** (varying between 16.5% and 20.8%). Under class imbalance, ROC-AUC is known to be overly optimistic because the vast number of true negatives ($TN$) inflates the specificity denominator ($TN / (TN + FP)$).
2. **False Discovery Penalty:** In quantitative execution, false alarms (predicting a liquidity crunch when normal trading continues) impose significant opportunity costs (unnecessarily halting execution, routing away from venues, or crossing wide spreads). PR-AUC directly tracks precision across the entire recall spectrum, penalizing false positives far more severely.
3. **Primary Metric Rule:** **Mean Fold PR-AUC** is designated as the primary selection criterion, with **Pooled Out-of-Fold PR-AUC** serving as the cross-validation aggregate check.

### 4.3 Statistical Significance Analysis
To determine whether performance differences between configurations are statistically significant or attributable to fold noise, paired two-tailed $t$-tests and non-parametric Wilcoxon signed-rank tests were performed across the 10 folds:

1. **`hist_gradient_boosting` vs. `random_forest_candidate_b`:**
   - Mean Difference: **+0.0278 PR-AUC (+2.78 percentage points)**.
   - Paired $t$-test: $t = 4.439, p = 0.00163$ (statistically significant at $p < 0.01$).
   - Wilcoxon signed-rank test: $W = 3.0, p = 0.0098$ (statistically significant at $p < 0.01$).
   - **Conclusion:** `hist_gradient_boosting` demonstrates a statistically robust, highly significant predictive advantage over the best Random Forest ensemble.

2. **`random_forest_candidate_b` vs. `random_forest_candidate_a` (Feature F51 Impact):**
   - Mean Difference: **+0.0015 PR-AUC (+0.15 percentage points)**.
   - Paired $t$-test: $t = 1.519, p = 0.1631$ (directional gain, not statistically significant at $\alpha = 0.05$).
   - Wilcoxon signed-rank test: $W = 16.0, p = 0.2754$.
   - **Conclusion:** While the mean gain is modest, Candidate B wins in 6 out of 10 folds, exhibits lower fold variance ($\sigma = 0.0352$ vs $0.0377$), and lifts the worst-fold floor by +0.0048, justifying the inclusion of F51.

3. **`logistic_regression_standard` vs. `logistic_regression_robust` (Scaler Impact):**
   - Mean Difference: **$+0.000002$ PR-AUC ($2 \times 10^{-6}$)**.
   - Maximum Absolute Fold Difference: **$0.000368$**.
   - **Conclusion:** Scaler selection has zero material impact on linear predictive discrimination.

### 4.4 Market Capitalization Stratification
Liquidity dynamics vary dramatically across market capitalization tiers. Stratified out-of-fold PR-AUC performance across Large-Cap, Mid-Cap, and Small-Cap universes reveals critical domain insights:

| Market Cap Tier | Typical Liquidity Regime | `logistic_regression_standard` | `random_forest_candidate_a` | `random_forest_candidate_b` | `hist_gradient_boosting` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Large-Cap** | Deep order books, tight spreads, institutional flow | 0.6136 | 0.6422 | 0.6570 | **0.7881** |
| **Mid-Cap** | Moderate depth, periodic liquidity voids | 0.5860 | 0.7092 | 0.7126 | **0.7355** |
| **Small-Cap** | Thin order books, frequent wide spreads, high crash risk | 0.4944 | 0.6507 | **0.6524** | 0.6304 |

**Stratification Insights:**
- **Large & Mid-Cap Dominance:** `hist_gradient_boosting` dramatically outperforms all models in Large-Cap (+0.1311 over RF-B) and Mid-Cap (+0.0229 over RF-B), demonstrating superior capability in isolating subtle liquidity decay signals in deep markets.
- **Small-Cap Performance:** In Small-Cap stocks, where liquidity risk events are most frequent, non-linear tree ensembles perform exceptionally well. `random_forest_candidate_b` achieves 0.6524 vs. HGBT's 0.6304. Random Forest's bootstrap aggregation provides slightly better resistance to extreme idiosyncratic outliers in low-volume stocks.
- **Linear Model Collapse:** Logistic Regression collapses in Small-Cap stocks (PR-AUC 0.4944), proving that linear hyperplanes cannot separate liquidity collapse states when spreads widen non-linearly.

---

## 5. TEMPORAL-STABILITY ANALYSIS

### 5.1 Chronological Consistency Across 10 Expanding Folds
Model stability across varying macroeconomic and volatility regimes is critical for production deployment. The 10 validation folds tile a 2.5-year out-of-sample period (May 2023 to October 2025):

| Fold # | Validation Period | Prev (%) | `logistic_regression_standard` | `logistic_regression_robust` | `random_forest_candidate_a` | `random_forest_candidate_b` | `hist_gradient_boosting` | Top Fold Winner |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 2023-05-10 to 2023-08-02 | 18.6% | 0.5916 | 0.5916 | 0.6381 | 0.6353 | **0.6506** | HGBT (+0.0125) |
| **1** | 2023-08-03 to 2023-10-31 | 18.7% | 0.5161 | 0.5161 | 0.6007 | **0.6032** | 0.6019 | RF-B (+0.0013) |
| **2** | 2023-11-01 to 2024-01-30 | 19.4% | 0.5374 | 0.5374 | **0.6312** | 0.6282 | 0.6271 | RF-A (+0.0040) |
| **3** | 2024-01-31 to 2024-04-30 | 17.9% | 0.4951 | 0.4950 | 0.5818 | 0.5813 | **0.6133** | HGBT (+0.0315) |
| **4** | 2024-05-02 to 2024-07-29 | 17.1% | 0.4631 | 0.4635 | 0.5752 | 0.5751 | **0.5924** | HGBT (+0.0172) |
| **5** | 2024-07-30 to 2024-10-23 | 19.4% | 0.5227 | 0.5227 | 0.6006 | 0.6032 | **0.6381** | HGBT (+0.0350) |
| **6** | 2024-10-24 to 2025-01-20 | 17.8% | 0.5145 | 0.5144 | 0.5683 | 0.5720 | **0.6186** | HGBT (+0.0466) |
| **7** | 2025-01-21 to 2025-04-21 | 17.9% | 0.4166 | 0.4165 | 0.5632 | 0.5645 | **0.5958** | HGBT (+0.0313) |
| **8** | 2025-04-22 to 2025-07-15 | 17.8% | 0.4692 | 0.4692 | 0.5308 | 0.5371 | **0.5844** | HGBT (+0.0473) |
| **9** | 2025-07-16 to 2025-10-10 | 18.4% | 0.4636 | 0.4636 | 0.5243 | 0.5291 | **0.5844** | HGBT (+0.0553) |

### 5.2 Stability Dispersion Metrics
Examining cross-fold dispersion provides insight into how robustly models handle distribution shift:

```
Cross-Fold PR-AUC Standard Deviation:
  HGBT:   [||||||] 0.0229 (Lowest variance / Most stable)
  RF-B:   [|||||||||] 0.0352
  RF-A:   [||||||||||] 0.0377
  LR-Std: [||||||||||||] 0.0487 (Highest variance / Least stable)
```

- **Cross-Fold Standard Deviation:** `hist_gradient_boosting` exhibits an exceptionally tight standard deviation of **0.0229**, which is 35% lower than RF-B (0.0352) and 53% lower than Logistic Regression (0.0487).
- **Worst-Case Fold Floor ($P_{\text{min}}$):** In production trading systems, models are constrained by their worst-case drawdowns. HGBT's minimum fold PR-AUC across the entire 2.5-year testing period is **0.5844** (Folds 8 & 9). Remarkably, **HGBT's worst fold exceeds the mean performance of RF-B (0.5829), RF-A (0.5814), and Logistic Regression (0.4990)**.
- **Head-to-Head Win Count:** Across the 10 folds, `hist_gradient_boosting` wins **8 out of 10 folds** in PR-AUC, **10 out of 10 folds** in ROC-AUC, **10 out of 10 folds** in F1 @ 0.50, and **8 out of 10 folds** in Brier score.

---

## 6. THRESHOLD AND CALIBRATION ANALYSIS

### 6.1 Inner-Window Calibration Architecture
Batch 4 implemented a leakage-safe decision threshold calibration module:
- **Inner Calibration Window:** For each expanding fold $k$, the model isolates the most recent 60 trading sessions of the training slice.
- **Target Horizon Purge:** An inner 5-session target purge ($H=5$) is applied between the inner training fold and the inner calibration fold, ensuring zero forward-looking target leakage.
- **Objective Function:** The decision threshold $\tau^*$ is calibrated to maximize the $F_1$ score over a candidate grid $\tau \in [0.30, 0.70]$ with step size $\Delta \tau = 0.02$.

### 6.2 Empirical Calibrated Threshold Distributions Across Folds

| Model Configuration | Mean Calibrated $\tau^*$ | Min $\tau^*$ | Max $\tau^*$ | Std Dev $\sigma_\tau$ | Calibration Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`hist_gradient_boosting`** | 0.534 | 0.40 | 0.70 | 0.096 | Successful in all 10 folds |
| **`random_forest_candidate_b`** | 0.508 | 0.40 | 0.62 | 0.071 | Successful in all 10 folds |
| **`random_forest_candidate_a`** | 0.504 | 0.38 | 0.62 | 0.074 | Successful in all 10 folds |
| **`logistic_regression_standard`** | 0.460 | 0.38 | 0.58 | 0.060 | Successful in all 10 folds |
| **`logistic_regression_robust`** | 0.458 | 0.38 | 0.58 | 0.060 | Successful in all 10 folds |

### 6.3 Fixed 0.50 Threshold vs. Calibrated Threshold Trade-Offs
An important finding from Batch 4 is the comparison between out-of-fold performance at the standard fixed threshold ($\tau = 0.50$) versus the historically calibrated threshold ($\tau^*$):

| Metric | `hist_gradient_boosting` @ 0.50 | `hist_gradient_boosting` @ $\tau^*$ | `random_forest_candidate_b` @ 0.50 | `random_forest_candidate_b` @ $\tau^*$ |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | 0.5033 | **0.5357** (+3.24%) | 0.5084 | **0.5220** (+1.36%) |
| **Recall** | **0.6581** | 0.6164 (-4.17%) | **0.5985** | 0.5786 (-1.99%) |
| **$F_1$ Score** | **0.5622** | 0.5389 (-2.33%) | **0.5375** | 0.5252 (-1.23%) |

**Analytical Breakdown of Threshold Trade-Offs:**
1. **Precision Enhancement:** Calibrating thresholds on recent market regimes raises the mean threshold for HGBT to $0.534$, successfully filtering out marginal signals and boosting precision from $50.33\%$ to $53.57\%$.
2. **Recall Suppression & $F_1$ Dip:** Because the inner calibration window is relatively short (60 sessions), localized regime shifts between the inner calibration window and the outer test fold can cause slight threshold overshooting (e.g., Fold 4 calibrated to $\tau^* = 0.68$, Fold 8 to $\tau^* = 0.70$). This reduces recall, leading to an average out-of-fold $F_1$ of 0.5389 compared to 0.5622 at fixed 0.50.
3. **Operational Recommendation:** 
   - Both the continuous probability score $P(\text{HRE}_{5d}=1)$ and discrete flags at both $\tau = 0.50$ (high-recall alert) and $\tau^*$ (high-precision execution filter) should be exposed in production.
   - For binary decision-making, the system must support dual-threshold monitoring.

### 6.4 Threshold Fallback Safety Rules
To guard against regime distortion or boundary pinning during live operations, the following automated safety rules are established:
- **Boundary Clamping:** If calibrated $\tau^* < 0.35$ or $\tau^* > 0.65$, the threshold is flagged as extreme, and the primary execution threshold falls back to $\tau = 0.50$.
- **$F_1$ Degradation Safeguard:** If the inner calibration $F_1$ is lower than the inner fixed $0.50$ $F_1$ by more than $0.05$, fallback to $\tau = 0.50$ is automatically triggered.

---

## 7. OPERATIONAL COMPARISON & COMPUTATIONAL FEASIBILITY

Trading systems operate under strict latency, compute, and maintenance budgets. Models that require excessive training time or brittle preprocessing pipelines introduce operational fragility.

### 7.1 Computational Profile Across 10 Expanding Folds

| Model Configuration | Mean Fit Time (s/fold) | Total Fit Time (10 folds) | Relative Latency vs. LR Baseline | Pipeline Imputation Required? | Feature Scaling Required? | Model Artifact Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `logistic_regression_standard` | **3.37 s** | **33.7 s** | **1.0x** (Baseline) | Yes (Median) | Yes (`StandardScaler`) | **< 1 MB** |
| `logistic_regression_robust` | 19.93 s | 199.3 s | 5.9x | Yes (Median) | Yes (`RobustScaler`) | < 1 MB |
| `hist_gradient_boosting` | **19.04 s** | **190.4 s** | **5.7x** | **No (Native Binning)** | **No (Tree Invariant)** | **~12 MB** |
| `random_forest_candidate_a` | 68.20 s | 682.0 s | 20.2x | No (Native Routing) | No (Tree Invariant) | ~175 MB |
| `random_forest_candidate_b` | 79.00 s | 790.0 s | 23.4x | No (Native Routing) | No (Tree Invariant) | ~185 MB |

### 7.2 Operational Evaluation Points

1. **Convergence Latency: StandardScaler vs. RobustScaler:**
   `logistic_regression_robust` takes **5.9x longer to train** than `logistic_regression_standard` (19.9s vs. 3.4s) while achieving identical predictive performance (PR-AUC 0.4990 vs 0.4990). The L-BFGS optimization algorithm converges substantially faster when features are normalized to unit variance ($z$-score) than under median/IQR scaling where feature scale disparities persist under heavy-tailed kurtosis.
2. **Computational Superiority of HistGradientBoosting:**
   `hist_gradient_boosting` is **4.15x faster than Random Forest Candidate B** (19.0s vs. 79.0s per fold) despite achieving higher accuracy across all metrics. By pre-binning continuous features into 256 discrete integer bins, gradient boosting evaluates split candidates in constant time $O(B)$ rather than sorting observations $O(N \log N)$.
3. **Pipeline Maintenance & Leakage Surface:**
   Both `hist_gradient_boosting` and `random_forest` natively handle missing values without imputation pipelines, completely eliminating the risk of pipeline state desynchronization or imputation leakage between training and inference.

---

## 8. SEPARATE ISOLATION FOREST ASSESSMENT FRAMEWORK

### 8.1 Unsupervised Anomaly Detection Mandate
Isolation Forest is evaluated as an **independent, companion unsupervised telemetry detector**, completely distinct from the supervised classification models:
- **No Target Supervision:** Isolation Forest is fitted strictly on unlabelled feature matrices without access to `hre_5d_train`.
- **Purpose:** To detect multivariate market structure dislocations, extreme liquidity voids, and atypical volatility profiles that deviate from historical norms, regardless of whether a forward 5-day HRE event was realized.
- **Orthogonal Signal:** A stock may experience severe microstructure anomalies (e.g., sudden order book hollowing, cancel-to-trade ratio spikes) without breaching the specific 5-day HRE threshold, and vice versa.

### 8.2 Empirical Telemetry Across 10 Walk-Forward Folds ($N = 163,357$)

| Metric / Parameter | Value Across 10 Expanding Folds | Operational Interpretation |
| :--- | :--- | :--- |
| **Feature Set** | 12 selected liquidity, spread & volatility features | Compact, non-redundant microstructure feature core |
| **Score Scaling** | Normalized to $s \in [0, 100]$ | Standardized scale: $s = (1 - s_{\text{raw}}) \times 100$ |
| **Mean Scaled Score** | **14.17 ± 1.55** | Low baseline anomaly density across normal trading |
| **Score Percentiles** | $p_{50} = 11.1$, $p_{90} = 27.5$, $p_{95} = 36.3$, $p_{99} = 56.4$ | Cleanly separated, long right tail of extreme events |
| **Out-of-Fold Flag Rate** | **2.59%** (4,233 flagged sessions / 163,357) | Calibrated contamination rate ($c = 0.025$) preserved |
| **HRE Event Overlap** | **3.18%** flagged in positive HRE vs. **2.45%** in normal | Positive risk enrichment: 1.30x over normal sessions |
| **Target Correlation** | $r = +0.051$ (Mean fold $r = 0.054$, range 0.001 to 0.116) | Confirms mild positive correlation without redundant overlap |
| **Execution Latency** | Fit: **4.67 s/fold**, Score: **0.15 s/fold** | Negligible computational overhead in production |

### 8.3 Two-Tier Production Risk Architecture
In production deployment, the supervised champion and unsupervised anomaly detector operate in an integrated two-tier risk matrix:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        TWO-TIER PRODUCTION RISK MATRIX                           │
└──────────────────────────────────────────────────────────────────────────────────┘
                                    │
             ┌──────────────────────┴──────────────────────┐
             │                                             │
┌────────────┴─────────────┐                  ┌────────────┴─────────────┐
│ HIGH SUPERVISED RISK     │                  │ LOW SUPERVISED RISK      │
│ P(HRE_5d) ≥ 0.50         │                  │ P(HRE_5d) < 0.50         │
└────────────┬─────────────┘                  └────────────┬─────────────┘
             │                                             │
    ┌────────┴────────┐                           ┌────────┴────────┐
    │                 │                           │                 │
┌───┴──────────┐ ┌────┴─────────┐             ┌───┴──────────┐ ┌────┴─────────┐
│ QUADRANT 1   │ │ QUADRANT 2   │             │ QUADRANT 3   │ │ QUADRANT 4   │
│ Anomaly ≥ 50 │ │ Anomaly < 50 │             │ Anomaly ≥ 50 │ │ Anomaly < 50 │
│ CRITICAL     │ │ SYSTEMATIC   │             │ ATYPICAL NO- │ │ BENIGN       │
│ EMERGENCY    │ │ ILLIQUIDITY  │             │ VELTY / VOID │ │ NORMAL       │
│ Action: HALT │ │ Action: SLOW │             │ Action: PROBE│ │ Action: EXEC │
└──────────────┘ └──────────────┘             └──────────────┘ └──────────────┘
```

- **Quadrant 1 (Critical Emergency):** High predicted probability ($P \ge 0.50$) and high anomaly score ($s \ge 50$). Represents severe liquidity collapse coupled with extreme microstructure distortion. Action: Immediate execution halt, reroute or cancel working orders.
- **Quadrant 2 (Systematic Illiquidity):** High predicted probability ($P \ge 0.50$) but low anomaly score ($s < 50$). Represents standard, historically typical illiquidity conditions (e.g., small-cap drift). Action: Constrain participation rate, switch to passive limit order execution.
- **Quadrant 3 (Atypical Novelty / Structural Void):** Low predicted probability ($P < 0.50$) but high anomaly score ($s \ge 50$). Represents unprecedented market structure dislocations or potential model blind spots. Action: Risk alert issued to trading desk, verify order book quotes before dispatching large blocks.
- **Quadrant 4 (Benign Normal):** Low predicted probability and low anomaly score. Normal execution parameters active.

---

## 9. FINAL MODEL-SELECTION CRITERIA AND DECISION HIERARCHY

To ensure objectivity and prevent metric gaming, a multi-tiered decision hierarchy is established:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     TIER 1: PRIMARY PREDICTIVE DISCRIMINATION (40%)              │
│                     Metric: Out-of-Fold Mean PR-AUC & Pooled PR-AUC              │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     TIER 2: TEMPORAL & REGIME STABILITY (30%)                    │
│       Metrics: Fold Std Dev σ_PR ≤ 0.040, Min Fold Floor P_min ≥ 0.50, Win Rate ≥ 60%│
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     TIER 3: CALIBRATION & PROBABILISTIC RELIABILITY (15%)        │
│              Metrics: Brier Score ≤ 0.135, Log Loss ≤ 0.420, ROC-AUC ≥ 0.830     │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     TIER 4: OPERATIONAL & PRODUCTION EFFICIENCY (15%)            │
│       Metrics: Fit Time ≤ 120s/fold, Inference Latency < 1ms, Native NaN Handling│
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 9.1 Conflict Resolution & Trade-Off Governance Protocol
In quantitative strategy evaluation, models may exhibit conflicting characteristics (e.g., higher mean accuracy but greater volatility, or lower accuracy but faster execution). The following deterministic protocol governs trade-offs:

1. **Accuracy vs. Stability Trade-Off:**
   - If Model $X$ achieves higher Mean PR-AUC than Model $Y$, but Model $Y$ exhibits lower cross-fold standard deviation ($\sigma_Y < \sigma_X$):
     - Check the **worst-case fold floor**: If $\min(P_X) \ge \min(P_Y)$, Model $X$ dominates because its downside floor is strictly superior despite higher absolute variance.
     - Check **statistical significance**: If paired $t$-test between $X$ and $Y$ shows $p < 0.05$, Model $X$ is selected.
     - If difference is non-significant ($p \ge 0.05$) and $\min(P_X) < \min(P_Y)$, the more stable model ($Y$) is selected.
   - *Application to Batch 4 Results:* `hist_gradient_boosting` achieves **higher Mean PR-AUC (0.6107 vs 0.5829)** AND **lower standard deviation (0.0229 vs 0.0352)** AND **higher worst-case floor (0.5844 vs 0.5291)**. There is **zero conflict**: HGBT strictly dominates across both dimensions simultaneously.

2. **Accuracy vs. Computational Latency Trade-Off:**
   - If a model delivers an incremental PR-AUC gain of less than $+0.0050$ (+0.5 percentage points) but incurs a $>10\times$ increase in computational overhead or requires complex imputation pipelines, the simpler, faster model is favored.
   - *Application to Batch 4 Results:* Between `logistic_regression_standard` (3.4s) and `logistic_regression_robust` (19.9s), performance is identical ($0.4990$ vs $0.4990$). Standard scaling is selected based on operational efficiency. Between HGBT (19.0s) and RF-B (79.0s), HGBT is both **4.15x faster** AND delivers a $+0.0278$ PR-AUC improvement.

---

## 10. EXPLICIT RULES FOR STAGE 8.12.3

Stage 8.12.3 will execute the formal decision gate to freeze configurations for pre-holdout training. The following five deterministic rules are pre-defined:

### Rule 1: Linear Scaler Selection (Baseline Selection)
- **Comparison:** `logistic_regression_standard` vs. `logistic_regression_robust`.
- **Criterion:** Given statistical parity in predictive discrimination (Mean PR-AUC: 0.4990 vs. 0.4990; Mean ROC-AUC: 0.8265 vs. 0.8264; Brier Score: 0.1382 vs. 0.1382), the model requiring fewer solver iterations and lower computational overhead shall be selected.
- **Action:** Eliminate `logistic_regression_robust`. Retain **`logistic_regression_standard`** (20 features, Median Imputer, `StandardScaler`) as the frozen linear baseline model.

### Rule 2: Random Forest Feature Candidate Selection
- **Comparison:** `random_forest_candidate_a` (35 features, excluding F51) vs. `random_forest_candidate_b` (36 features, including F51 `relative_volume`).
- **Criterion:** Candidate B incorporates the volume-shock feature F51, which carries 15.5% missingness, evaluated via native missing-value routing in tree splitting. Candidate B achieves higher Mean PR-AUC (0.5829 vs. 0.5814), higher Pooled PR-AUC (0.5610 vs. 0.5591), higher worst-case floor (0.5291 vs. 0.5243), lower cross-fold variance ($\sigma = 0.0352$ vs. $0.0377$), superior Brier score (0.1320 vs. 0.1329), and wins in 6 of 10 folds.
- **Action:** Select **`random_forest_candidate_b`** as the superior ensemble configuration. Confirm that native tree missingness routing successfully exploits volume shocks without leakage.

### Rule 3: Primary Supervised Champion Model Selection
- **Comparison:** `logistic_regression_standard` vs. `random_forest_candidate_b` vs. `hist_gradient_boosting`.
- **Criterion:** The Primary Supervised Champion must satisfy Tier 1 and Tier 2 criteria: highest statistically significant Mean PR-AUC, highest Pooled PR-AUC, lowest cross-fold standard deviation, highest worst-case floor, and lowest Brier score.
- **Action:** Select **`hist_gradient_boosting`** as the **Primary Supervised Champion Model**. 
  - Mean PR-AUC: **0.6107** (+0.0278 over RF-B, +0.1117 over LR-Std; $p = 0.00163$).
  - Cross-fold stability: $\sigma = 0.0229$ (lowest across all models).
  - Minimum fold floor: $0.5844$ (exceeds average performance of all competitors).
  - Out-of-fold wins: 8/10 PR-AUC, 10/10 ROC-AUC, 10/10 F1 @ 0.50.

### Rule 4: Model Freezing Protocol for Pre-Holdout Training
- **Protocol:** To maintain complete scientific transparency and verify model generalization on unseen future market regimes, **all five supervised configurations shall be frozen without modification** and trained on the pre-holdout dataset in Stage 8.12.4, followed by complete evaluation on the 250-session holdout in Stage 8.12.5.
- **Scientific Value:** Evaluating all five configurations on the holdout confirms whether the relative superiority of HGBT and the inclusion of F51 generalize out-of-sample, providing an unbroken benchmark progression from linear baselines to gradient-boosted trees.

### Rule 5: Unsupervised Isolation Forest Confirmation
- **Protocol:** Freeze `isolation_forest` (12 features, Median Imputer, `RobustScaler`, $N=150$, contamination 2.5%) as an independent companion telemetry model.
- **Action:** Train on unlabelled pre-holdout data in Stage 8.12.4 and evaluate on holdout in Stage 8.12.6 to provide out-of-sample market microstructure anomaly telemetry.

---

## 11. HOLDOUT ISOLATION AND LEAKAGE SAFEGUARDS

### 11.1 Verification of Holdout Quarantine
Throughout Stage 8.12.1 and Stage 8.12.2, strict physical and logical quarantine of the holdout partition was maintained:
- **Quarantined Partition:** 250 consecutive trading sessions from `2025-10-20` to `2026-10-23` ($N = 74,334$ rows, `split == 'holdout'`).
- **Zero Access Verification:** The holdout slice was **NOT** loaded into memory, read, inspected, fitted, scored, or evaluated in any script or calculation during this stage.
- **Zero Leakage Rule:** No holdout metrics exist. No selection criteria, feature weights, hyperparameters, or thresholds were influenced by the holdout partition.

### 11.2 Dataset Integrity & Checksum Verification
To mathematically prove that underlying data files were untouched throughout Stage 8.12.2, SHA-256 cryptographic hashes were computed directly from disk:

| File Path | Recorded Baseline Hash | Computed SHA-256 Hash | Integrity Status |
| :--- | :--- | :--- | :---: |
| `data/processed/features.parquet` | `c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae` | `c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae` | **MATCH (PERFECT)** |
| `data/processed/features_with_targets.parquet` | `c6cebf3d93d022b11765c206b8e2ffb84e578a020e0d913d937b37d04028aa91` | `c6cebf3d93d022b11765c206b8e2ffb84e578a020e0d913d937b37d04028aa91` | **MATCH (PERFECT)** |

---

## 12. FINAL VERDICT AND READINESS FOR STAGE 8.12.3

### 12.1 Readiness Assessment Matrix

| # | Evaluation Dimension | Audit & Assessment Finding | Verdict |
| :---: | :--- | :--- | :---: |
| 1 | **Objective & Scope Adherence** | Read-only methodology formulated strictly on walk-forward cross-validation results prior to holdout exposure. | **PASS** |
| 2 | **Input Artifact Verification** | Sourced directly from verified Batch 4 interim outputs (`supervised_model_comparison.csv`, `supervised_fold_metrics.csv`). | **PASS** |
| 3 | **Configuration Reconciliation** | Clear distinction maintained: 3 algorithm families across 5 experimental configurations, plus 1 unsupervised model. | **PASS** |
| 4 | **Predictive Metric Hierarchy** | Multi-tiered decision hierarchy formalized: PR-AUC (primary), stability (secondary), calibration/ROC-AUC (tertiary), compute (operational). | **PASS** |
| 5 | **Statistical Significance Testing** | Paired $t$-test and Wilcoxon signed-rank tests confirm HGBT superiority over RF-B is statistically significant ($p < 0.01$). | **PASS** |
| 6 | **Threshold & Calibration Rules** | Inner 60-day calibration analyzed; fallback safety rules and dual-threshold operational protocol established. | **PASS** |
| 7 | **Operational Latency Profile** | Computational trade-offs evaluated; HGBT verified as 4.15x faster than RF-B; RobustScaler latency penalty documented. | **PASS** |
| 8 | **Unsupervised Anomaly Framework** | Isolation Forest evaluated independently as telemetry guardrail; 2-tier production risk matrix specified. | **PASS** |
| 9 | **Explicit Decision Rules** | Five deterministic rules formulated for Stage 8.12.3 configuration freezing and pre-holdout preparation. | **PASS** |
| 10 | **Holdout Isolation Safeguards** | Holdout partition (74,334 rows) completely unaccessed; SHA-256 hashes of feature datasets 100% matched. | **PASS** |

---

### 12.2 OVERALL STAGE 8.12.2 VERDICT: **PASS**

The Model-Selection Methodology is rigorously specified, statistically validated, and fully codified. All requirements for Stage 8.12.2 are satisfied. The project is fully prepared to proceed to **Stage 8.12.3: Select and Freeze Model Configurations**.

**Strict Stop Gate Instruction:**  
As strictly mandated by user instructions, execution halts upon completion of Stage 8.12.2. No modifications to configurations or code have been made, and Stage 8.12.3 will commence only upon explicit authorization.
