# PHASE 8 BATCH 5 | STAGE 8.12.8: FINAL MODEL SELECTION AND RESEARCH CONCLUSIONS REPORT

**Project:** ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities  
**Phase:** Phase 8 Batch 5 (Stage 8.12.8)  
**Task:** Final Model Selection, Methodological Conclusions & Research Architecture Formalization  
**Authoritative Evidence:**  
- Model Selection Methodology: `reports/phase8_batch5_stage8_12_2_selection_methodology.md`  
- Configuration Freeze Manifest: `reports/phase8_batch5/configuration_freeze/model_configuration_manifest.json`  
- Pre-Holdout Training Report: `reports/phase8_batch5/stage_8_12_3_4_training_report.md`  
- Supervised Holdout Report: `reports/phase8_batch5/stage_8_12_5_supervised_holdout_report.md`  
- Isolation Forest Holdout Report: `reports/phase8_batch5/stage_8_12_6_isolation_forest_holdout_report.md`  
- Walk-Forward vs Holdout Report: `reports/phase8_batch5/stage_8_12_7_walkforward_vs_holdout_report.md`  
- Batch 2 Consolidated Summary: `reports/phase8_batch5/stage_8_12_5_6_7_batch2_summary.md`  
**Status:** COMPLETE, FORMALLY RATIFIED & FROZEN  
**Date:** 2026-11-15  

---

## 1. EXECUTIVE DECISION

Following the completion of the 10-fold walk-forward validation (Batch 4) and the out-of-sample evaluation across the quarantined 250-trading-session holdout (`2025-10-20` to `2026-10-23`, $N=71,756$ valid rows), this report documents the formal model selection decision according to the criteria pre-registered in Stage 8.12.2.

### Formal Selection Outcome:
1. **Primary Champion Supervised Model:** **`hist_gradient_boosting` (Configuration `CFG_05_HGBT`)** is formally selected as the preferred production-grade supervised predictive model for forward 5-day High-Risk Events (`hre_5d`).
   - **Holdout PR-AUC:** **0.5723** (93.72% retention of walk-forward mean PR-AUC 0.6107).
   - **Holdout ROC-AUC:** **0.8460** (vs walk-forward mean 0.8548).
   - **Holdout Calibration:** Lowest Brier score (**0.1167**) and lowest Log Loss (**0.3625**).
   - **Temporal Consistency:** Won **13 out of 13 monthly holdout subperiods (100% win rate)** against all competing configurations.
   - **Margin:** Exceeds the nearest alternative by **+0.0343 PR-AUC (+3.43 percentage points)**.
2. **Strongest Supervised Alternative / Benchmark:** **`random_forest_candidate_b` (Configuration `CFG_04_RF_CANDIDATE_B`, 36 features)** is designated as the primary reference ensemble.
   - **Holdout PR-AUC:** **0.5380** (92.30% retention).
   - Validates the inclusion of feature F51 (`relative_volume`), outperforming Candidate A by +0.0062 PR-AUC and winning 12 of 13 monthly periods.
3. **Linear Baselines:** **`logistic_regression_standard` (Configuration `CFG_01_LOGREG_STANDARD`, 20 features)** is retained strictly as the baseline benchmark (Holdout PR-AUC: **0.4051**). Scaler choice between Standard and Robust demonstrated complete empirical indifference ($\Delta = 0.000070$).
4. **Independent Anomaly Telemetry:** **`isolation_forest` (Configuration `CFG_06_IF_UNSUPERVISED`, 12 features)** is retained strictly as an orthogonal, unsupervised risk-telemetry signal (holdout flag rate: 2.34%, HRE enrichment: 1.18x). It is unranked against the supervised classifiers.

> [!IMPORTANT]
> **Methodological Boundary:**
> Holdout performance metrics serve strictly as terminal evaluation evidence for ratifying this pre-defined selection decision. They must NOT be utilized for further hyperparameter tuning, feature pruning, threshold re-optimization, or model refitting.

---

## 2. RECAPITULATION OF APPROVED SELECTION METHODOLOGY

As codified in `reports/phase8_batch5_stage8_12_2_selection_methodology.md`, model selection follows a deterministic 5-tier evaluation hierarchy:

1. **Tier 1 (Primary Criterion): Out-of-Sample Precision-Recall AUC (PR-AUC)**  
   Given the empirical positive class imbalance (base rate ~16.55% on holdout, ~18.5% historical), PR-AUC is the definitive primary metric measuring precision-recall trade-offs without rewarding true negative inflation.
2. **Tier 2 (Secondary Discrimination): Receiver Operating Characteristic AUC (ROC-AUC)**  
   Measures overall cross-threshold ranking power. A minimum threshold of ROC-AUC $\ge 0.75$ is required for institutional risk models.
3. **Tier 3 (Probability Calibration): Brier Score and Log Loss**  
   Evaluates reliability of probability outputs $P(y=1 \mid X_t)$. Lower Brier score and Log Loss ensure that generated probability outputs correspond accurately to empirical frequencies.
4. **Tier 4 (Temporal Stability & Regime Resilience): Monthly Win Rate & Generalization Retention**  
   Evaluates performance consistency across market cycles:
   - Generalization Retention $\% = 100 \times \frac{\text{Holdout PR-AUC}}{\text{Walk-Forward Mean PR-AUC}}$.
   - Monthly Win Rate across the 13 monthly subperiods of the holdout.
5. **Tier 5 (Operational Feasibility): Inference Latency & Architectural Parsimony**  
   Strict latency constraint: Full cross-sectional inference for 300 tickers must complete in $< 1.0$ second.

---

## 3. FIVE-CONFIGURATION COMPARISON ACROSS SELECTION CRITERIA

The table below summarizes the authoritative empirical performance across both walk-forward cross-validation and the untouched holdout partition ($N_{\text{holdout}} = 71,756$ valid rows):

| Model Configuration | Feature Count | WF Mean PR-AUC | Holdout PR-AUC | PR-AUC Retention | WF Mean ROC-AUC | Holdout ROC-AUC | Holdout Brier | Holdout Log Loss | Monthly Holdout Wins | Holdout Inference Latency | Overall Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`hist_gradient_boosting`** | **36** | **0.6107** | **0.5723** | **93.72%** | **0.8548** | **0.8460** | **0.1167** | **0.3625** | **13 / 13 (100%)** | **0.22 s** | **1 (Champion)** |
| **`random_forest_candidate_b`** | 36 | 0.5829 | 0.5380 | 92.30% | 0.8361 | 0.8211 | 0.1249 | 0.4036 | 0 / 13 (0%) | 0.11 s | **2 (Runner-Up)** |
| **`random_forest_candidate_a`** | 35 | 0.5814 | 0.5318 | 91.46% | 0.8346 | 0.8186 | 0.1261 | 0.4082 | 0 / 13 (0%) | 0.12 s | **3** |
| **`logistic_regression_standard`**| 20 | 0.4990 | 0.4051 | 81.18% | 0.8265 | 0.7938 | 0.1298 | 0.4349 | 0 / 13 (0%) | 0.02 s | **4 (Baseline)** |
| **`logistic_regression_robust`**  | 20 | 0.4990 | 0.4050 | 81.17% | 0.8264 | 0.7937 | 0.1298 | 0.4349 | 0 / 13 (0%) | 0.02 s | **5** |

---

## 4. FINAL SUPERVISED MODEL DECISION & RATIONALE

### 4.1 Why `hist_gradient_boosting` Is Preferred
`hist_gradient_boosting` satisfies every pre-registered selection tier decisively:
1. **Unambiguous Predictive Superiority:** Achieved a Holdout PR-AUC of **0.5723**, outperforming `random_forest_candidate_b` by **+0.0343** (+3.43 percentage points) and `logistic_regression_standard` by **+0.1672** (+16.72 percentage points).
2. **Minimal Generalization Decay:** Retained **93.72%** of its walk-forward PR-AUC, exhibiting the lowest degradation ($\Delta = -0.0383$) among all evaluated architectures.
3. **Superior Probability Calibration:** Generated the lowest Brier score (**0.1167**) and lowest Log Loss (**0.3625**). Its calibrated probabilities reflect true event frequencies without systematic over-confidence.
4. **Perfect Temporal Consistency:** Won **13 out of 13 monthly subperiods** in the holdout year. Regardless of whether equity markets experienced high-volatility shocks or low-volatility grinding regimes, HGBT remained the top-performing model.
5. **Large-Cap Institutional Edge:** In large-cap equities (the highest-volume segment of Indian markets), HGBT delivered a Holdout PR-AUC of **0.7079**, outperforming Random Forest by +0.0870.
6. **Operational Efficiency:** Evaluated 74,334 holdout rows in **0.22 seconds** (~3.0 microseconds per record), comfortably surpassing the 1.0-second institutional latency threshold.

---

## 5. STRONGEST ALTERNATIVE: RANDOM FOREST CANDIDATE B

### 5.1 Role as Primary Benchmark
`random_forest_candidate_b` (36 features) is confirmed as the strongest alternative to the champion:
- Achieved Holdout PR-AUC of **0.5380** with 92.30% retention.
- Demonstrates competitive resilience in mid-cap equities (Holdout PR-AUC of **0.7097**, marginally edging HGBT's 0.6890).
- Serves as an essential reference architecture: as a bagging ensemble, it provides an independent point of comparison against boosting dynamics.

### 5.2 Resolution of Feature F51 (`relative_volume`) Hypothesis
- Candidate B included feature F51 (`relative_volume`), while Candidate A excluded it.
- **Walk-Forward Result:** Candidate B led Candidate A by +0.0015 PR-AUC (0.5829 vs 0.5814).
- **Holdout Result:** Candidate B widened its lead to **+0.0062 PR-AUC** (0.5380 vs 0.5318) and outperformed Candidate A in **12 out of 13 monthly subperiods (92.3% win rate)**.
- **Empirical Resolution:** Feature F51 provides demonstrable incremental signal by capturing volume surges and liquidity stress during abnormal trading sessions. It is permanently retained in the tree-based feature schema.

---

## 6. LOGISTIC REGRESSION BASELINE CONCLUSIONS

1. **Failure of Linear Hyperplanes in Microstructure Regimes:** Both regularized Logistic Regression configurations suffered substantial out-of-sample degradation, retaining only **81.18%** of their walk-forward PR-AUC (dropping from 0.4990 to 0.4051). This confirms that linear models cannot adequately represent complex cross-feature interactions (e.g., elevated bid-ask spread interacting with order-flow shock).
2. **Scaler Invariance Resolution:**
   - StandardScaler Holdout PR-AUC: **0.405109**
   - RobustScaler Holdout PR-AUC: **0.405039**
   - Absolute Difference: **0.000070** (0.007 percentage points)
   - **Conclusion:** Under median imputation and L2 regularization ($C=1.0$), scaling methodology has zero practical or operational impact. StandardScaler is preserved as the standard baseline pipeline.

---

## 7. ISOLATION FOREST: INDEPENDENT ANOMALY SURVEILLANCE & THRESHOLD AUDIT

### 7.1 Independent Role & Complementary Utility
Isolation Forest operates as an unsupervised multidimensional outlier detector, fitted on 12 features without target labels. It is **not ranked** against the supervised classifiers because its objective is structural dislocation detection, not forward conditional risk prediction.
- Holdout mean scaled score: **13.91** (vs Walk-Forward mean: **14.25**), confirming stable score distributions across forward regimes.
- At an anomaly flag rate of **2.34%** (1,738 / 74,334 sessions), the realized HRE rate was **19.38%** vs **16.48%** for unflagged sessions (**1.18x enrichment**, $r = +0.0264$).
- **Surveillance Function:** Isolation Forest provides an orthogonal early-warning tripwire for extreme microstructure dislocations (e.g., massive block trades or quote drops) that do not necessarily produce directional price drift.

### 7.2 Methodological Audit of Anomaly Threshold (Provenance of 43.12)
An audit of the Isolation Forest implementation and documentation revealed the following:
1. In the pre-holdout training metadata (`reports/phase8_batch5/training_metadata/isolation_forest_training_metadata.json`), the model was fitted with `contamination=0.05` on training data.
2. The operational flag generation in `src/modeling/anomaly.py` and `scripts/evaluate_holdout.py` relies strictly on scikit-learn's internal `predict()` method, which uses the fitted training-set decision offset `offset_`. On the holdout dataset, this decision boundary corresponds to a scaled score cutoff of **44.44**, flagging exactly 1,738 sessions (2.34%).
3. The threshold value **43.12** reported in early Batch 2 narrative documentation corresponds to the empirical ~97.41th percentile of the scaled scores on the holdout partition itself ($1 - 0.0259$).
4. **Methodological Note:** Describing 43.12 as an "independently pre-calibrated threshold" was an informal shorthand in early summary text; the actual algorithmic flagging applied the pre-trained model's internal training-set `offset_` without referencing holdout percentiles. To preserve audit trail integrity, existing artifacts remain untouched, but future deployment specifications must reference the model's native `offset_` / binary prediction output directly.

---

## 8. FINAL RESEARCH ARCHITECTURE

The finalized Phase 8 model architecture is formally specified as:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FINAL PHASE 8 RESEARCH ARCHITECTURE                             │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               │                                                         │
┌──────────────┴──────────────────────────┐               ┌──────────────┴───────────────┐
│     PRIMARY CHAMPION SUPERVISED MODEL   │               │    INDEPENDENT SURVEILLANCE  │
│  HistGradientBoosting (36 Features)     │               │  Isolation Forest (12 Feat)  │
│  Config: CFG_05_HGBT                    │               │  Config: CFG_06_IF           │
│  Holdout PR-AUC: 0.5723                 │               │  Holdout Flag Rate: 2.34%    │
│  Holdout ROC-AUC: 0.8460                │               │  Enrichment Ratio: 1.18x     │
│  Calibrated Threshold: tau* = 0.534     │               │  Role: Microstructure Outlier│
│  Role: Primary Conditional Risk Predict │               │        Dislocation Tripwire  │
└──────────────┬──────────────────────────┘               └──────────────────────────────┘
               │
               ├─────────────────────────────────────────┐
               │                                         │
┌──────────────┴──────────────────────────┐ ┌────────────┴───────────────────────────────┐
│     STRONGEST ALTERNATIVE BENCHMARK     │ │           LINEAR BASELINE BENCHMARK        │
│  Random Forest Candidate B (36 Features)│ │  Logistic Regression Standard (20 Features)│
│  Config: CFG_04_RF_CANDIDATE_B          │ │  Config: CFG_01_LOGREG_STANDARD            │
│  Holdout PR-AUC: 0.5380                 │ │  Holdout PR-AUC: 0.4051                    │
│  Holdout ROC-AUC: 0.8211                │ │  Holdout ROC-AUC: 0.7938                   │
│  Role: Primary Bagging Reference Model  │ │  Role: Linear Baseline Benchmark           │
└─────────────────────────────────────────┘ └────────────────────────────────────────────┘
```

---

## 9. GENERALIZATION, RETENTION & OVERFITTING ASSESSMENT

### 9.1 Generalization Degradation (Retention Gap)
- Tree-based architectures demonstrated acceptable generalization bounds:
  - HGBT: **93.72% retention** ($-0.0383$ PR-AUC delta).
  - RF-B: **92.30% retention** ($-0.0449$ PR-AUC delta).
  - RF-A: **91.46% retention** ($-0.0497$ PR-AUC delta).
- A ~6–8% drop in PR-AUC from expanding-window cross-validation to a completely unaccessed 1-year forward holdout is well within theoretical expectations for financial time-series, driven by regime variation, non-stationarity, and macroeconomic shifts.
- Conversely, linear baselines dropped by **18.82%** (retention 81.18%), demonstrating structural underfitting to complex non-linear dynamics.

### 9.2 Overfitting Diagnosis: PASS
The evidence indicates that `hist_gradient_boosting` is **not overfitted**:
- Kendall's rank correlation across all 5 models was **+1.000** (perfect rank preservation).
- Brier score actually improved on holdout (0.1266 $\rightarrow$ 0.1167).
- Log Loss improved on holdout (0.3857 $\rightarrow$ 0.3625).
- Model won 100% of monthly holdout periods.

---

## 10. METHODOLOGICAL & OPERATIONAL LIMITATIONS

1. **Finite 250-Session Holdout:** The holdout spans one calendar year (October 2025 to October 2026). While 74,334 ticker-sessions provide strong cross-sectional breadth, the sample represents a single annual macroeconomic regime.
2. **Missing Boundary Forward Labels:** Out of 74,334 holdout rows, 2,578 (3.47%) lacked valid forward labels (1,500 due to the 5-session forward boundary lookahead and 1,078 due to trading suspensions/delistings).
3. **Cross-Sectional Pooling:** Models were evaluated globally across large, mid, and small-cap stocks. While stratified subgroup metrics were reported, production deployment must account for liquidity differentials across market capitalization tiers.
4. **Predictive Intelligence vs. Trading Profitability:** This research establishes predictive performance for liquidity risk intelligence. It does **not** prove execution profitability or evaluate slippage, trading fees, market impact, or portfolio management constraints.
5. **No Universal Optimality Claim:** `hist_gradient_boosting` is the preferred configuration *within the approved experimental design and evaluated feature space*. It is not claimed to be universally optimal under untested architectures or market regimes.

---

## 11. RESEARCH CONCLUSIONS & READINESS

1. **Primary Research Goal Achieved:** Machine learning models (specifically gradient boosted decision trees) provide actionable, leakage-safe, out-of-sample predictive intelligence for forward liquidity risk in Indian equities.
2. **Feature Engineering Validation:** Multi-horizon volatility ratios, turnover metrics, and volume-shock features (F51) provide robust out-of-sample predictive power.
3. **Phase 8 Research Phase Complete:** All experimental stages (8.9 through 8.12.8) have been executed with strict data preservation, zero contamination, and full reproducibility.
4. **Readiness:** The repository is fully prepared for final notebook synchronization (Stage 8.12.9) followed by test suite verification and formal Phase 8 closure.
