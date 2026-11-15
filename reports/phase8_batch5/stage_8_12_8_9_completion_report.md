# PHASE 8 BATCH 5 | BATCH 3 COMPLETION REPORT
## Stages 8.12.8 & 8.12.9: Final Model Selection, Conclusions & Notebook Documentation

**Project:** ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities  
**Repository:** `Indian_Equities_Liquidity_Risk_ML-main`  
**Phase:** Phase 8 Batch 5 (Batch 3: Model Selection & Notebook Documentation)  
**Stages Executed:**  
- **Stage 8.12.8:** Final Model Selection and Research Conclusions  
- **Stage 8.12.9:** Final Research Notebook Updates  
**Status:** COMPLETE, INDEPENDENTLY AUDITED & FORMALLY FROZEN  
**Date:** 2026-11-15  

---

## 1. FINAL SUPERVISED MODEL SELECTION

In strict adherence to the model selection methodology pre-registered and frozen in Stage 8.12.2 (`reports/phase8_batch5_stage8_12_2_selection_methodology.md`), the formal decision across the five evaluated supervised configurations is:

### Selected Champion: `hist_gradient_boosting` (Configuration `CFG_05_HGBT`)
- **Algorithm Family:** Supervised Gradient Boosted Decision Trees (`HistGradientBoostingClassifier`)
- **Feature Space:** 36 curated liquidity, spread, volatility, and order-flow features
- **Model Artifact:** `models/phase8_batch5/hist_gradient_boosting/model.joblib`
- **Calibrated Operational Threshold:** $\tau^* = 0.534$
- **Holdout PR-AUC:** **0.5723** (93.72% retention of walk-forward mean PR-AUC 0.6107)
- **Holdout ROC-AUC:** **0.8460** (vs walk-forward mean 0.8548)
- **Calibration Loss:** Lowest Brier score (**0.1167**) and lowest Log Loss (**0.3625**) across all configurations
- **Monthly Temporal Consistency:** Won **13 out of 13 monthly subperiods (100% win rate)** on the holdout
- **Competitive Advantage:** Leads the nearest alternative (`random_forest_candidate_b`) by **+0.0343 PR-AUC (+3.43 percentage points)** and the linear baseline by **+0.1672 PR-AUC (+16.72 percentage points)**

---

## 2. STRONGEST ALTERNATIVE: RANDOM FOREST CANDIDATE B

- **Designated Architecture:** `random_forest_candidate_b` (Configuration `CFG_04_RF_CANDIDATE_B`, 36 features)
- **Role:** Primary bagging reference alternative
- **Holdout PR-AUC:** **0.5380** (92.30% retention of walk-forward mean 0.5829)
- **Validation of Feature F51 (`relative_volume`):**
  Candidate B outperformed Candidate A (35 features, excluding F51) by **+0.0062 PR-AUC (+0.62 percentage points)** on the holdout and won **12 out of 13 monthly subperiods (92.3% win rate)**. Feature F51 provides statistically verified incremental predictive power during volume shock episodes and is permanently retained in the 36-feature schema.

---

## 3. LINEAR BASELINE CONCLUSION

- **Designated Architecture:** `logistic_regression_standard` (Configuration `CFG_01_LOGREG_STANDARD`, 20 features)
- **Holdout PR-AUC:** **0.4051** (81.18% retention of walk-forward mean 0.4990)
- **Underfitting Evidence:** Suffered substantial generalization decay (-18.82%), demonstrating that linear hyperplanes cannot capture complex non-linear liquidity interactions.
- **Scaler Invariance Resolution:** StandardScaler and RobustScaler holdout PR-AUC differed by only **0.000070**, confirming complete empirical indifference. StandardScaler is preserved as the standard linear baseline.

---

## 4. ISOLATION FOREST CONCLUSION & TELEMETRY

- **Designated Architecture:** `isolation_forest` (Configuration `CFG_06_IF_UNSUPERVISED`, 12 features)
- **Role:** Independent unsupervised anomaly telemetry and microstructure dislocation tripwire
- **Holdout Telemetry:** Mean scaled score **13.91** (vs WF 14.25), Median **10.97** (vs WF 11.22), Flag rate **2.34%** (1,738 / 74,334 sessions), Realized HRE rate in flagged **19.38%** vs **16.48%** in unflagged (**1.18x enrichment**, $r = +0.0264$).
- **Methodological Threshold Audit:** The operational binary flag relies on the fitted model's native training-set decision offset (`offset_`). The threshold value `43.12` corresponds to the empirical ~97.41th percentile of holdout scaled scores, while the native binary decision boundary corresponds to $s_{\text{scaled}} \approx 44.44$. It is retained as an orthogonal surveillance component, unranked against supervised models.

---

## 5. FINAL RESEARCH ARCHITECTURE SPECIFICATION

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
│  Role: Primary Conditional Risk Predict │               │  Role: Microstructure Outlier│
└──────────────┬──────────────────────────┘               └──────────────────────────────┘
               │
               ├─────────────────────────────────────────┐
               │                                         │
┌──────────────┴──────────────────────────┐ ┌────────────┴───────────────────────────────┐
│     STRONGEST ALTERNATIVE BENCHMARK     │ │           LINEAR BASELINE BENCHMARK        │
│  Random Forest Candidate B (36 Features)│ │  Logistic Regression Standard (20 Features)│
│  Config: CFG_04_RF_CANDIDATE_B          │ │  Config: CFG_01_LOGREG_STANDARD            │
│  Holdout PR-AUC: 0.5380                 │ │  Holdout PR-AUC: 0.4051                    │
│  Role: Primary Bagging Reference Model  │ │  Role: Linear Baseline Benchmark           │
└─────────────────────────────────────────┘ └────────────────────────────────────────────┘
```

---

## 6. UPDATED NOTEBOOK INVENTORY

All four notebooks were updated by appending final documentation sections. Every pre-existing cell, execution count, and historical output was strictly preserved:

| Notebook | Pre-Existing Cells | Appended Documentation Section | Final Total Cells | Verification Status |
| :--- | :---: | :--- | :---: | :---: |
| [`notebooks/05_model_baseline.ipynb`](file:///C:/Users/vighn/Downloads/Indian_Equities_Liquidity_Risk_ML-main/Indian_Equities_Liquidity_Risk_ML-main/notebooks/05_model_baseline.ipynb) | 18 | Section 9: Final Model Selection & Baseline Conclusions | 19 | 100% Intact |
| [`notebooks/06_model_rf.ipynb`](file:///C:/Users/vighn/Downloads/Indian_Equities_Liquidity_Risk_ML-main/Indian_Equities_Liquidity_Risk_ML-main/notebooks/06_model_rf.ipynb) | 16 | Section 8: Final Model Selection & Random Forest Conclusions | 17 | 100% Intact |
| [`notebooks/07_model_hgbt.ipynb`](file:///C:/Users/vighn/Downloads/Indian_Equities_Liquidity_Risk_ML-main/Indian_Equities_Liquidity_Risk_ML-main/notebooks/07_model_hgbt.ipynb) | 14 | Section 7: Final Model Selection & Champion Confirmation | 15 | 100% Intact |
| [`notebooks/08_anomaly_detection.ipynb`](file:///C:/Users/vighn/Downloads/Indian_Equities_Liquidity_Risk_ML-main/Indian_Equities_Liquidity_Risk_ML-main/notebooks/08_anomaly_detection.ipynb) | 14 | Section 7: Final Anomaly Telemetry Conclusions & Role | 15 | 100% Intact |

---

## 7. NEW DOCUMENTATION AND MANIFEST INVENTORY

1. [`reports/phase8_batch5/stage_8_12_8_final_model_selection.md`](file:///C:/Users/vighn/Downloads/Indian_Equities_Liquidity_Risk_ML-main/Indian_Equities_Liquidity_Risk_ML-main/reports/phase8_batch5/stage_8_12_8_final_model_selection.md)
   - Comprehensive final model selection report covering executive decisions, approved methodology, configuration comparisons, architecture justifications, and research limitations.
2. [`reports/phase8_batch5/final_model_selection_manifest.json`](file:///C:/Users/vighn/Downloads/Indian_Equities_Liquidity_Risk_ML-main/Indian_Equities_Liquidity_Risk_ML-main/reports/phase8_batch5/final_model_selection_manifest.json)
   - Machine-readable decision manifest capturing selected model IDs, artifact paths, feature list references, calibrated thresholds, and complete empirical metrics.
3. [`reports/phase8_batch5/stage_8_12_8_9_completion_report.md`](file:///C:/Users/vighn/Downloads/Indian_Equities_Liquidity_Risk_ML-main/Indian_Equities_Liquidity_Risk_ML-main/reports/phase8_batch5/stage_8_12_8_9_completion_report.md)
   - This authoritative Batch 3 completion report.

---

## 8. STRICT VERIFICATION CHECKLIST

| Verification Item | Requirement | Status | Evidence / Verification |
| :--- | :--- | :---: | :--- |
| **Model Training Prohibition** | No `.fit()` or `.fit_transform()` | **VERIFIED** | Zero models retrained; all fitted model artifacts unchanged |
| **Walk-Forward Validation** | No rerunning of 10 folds | **VERIFIED** | Batch 4 artifacts unchanged |
| **Holdout Evaluation** | No rerunning of holdout scoring | **VERIFIED** | Batch 2 holdout artifacts unchanged |
| **Threshold Recalibration** | No threshold re-optimization | **VERIFIED** | Pre-holdout calibrated thresholds ($\tau^*$) preserved |
| **Notebook Pre-existing Cells**| No historical cell execution | **VERIFIED** | All historical execution counts, outputs, and cells intact |
| **Dataset Hash: `features.parquet`** | `c3975cf9...8ae` | **VERIFIED** | SHA-256: `c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae` |
| **Dataset Hash: `features_with_targets.parquet`** | `c6cebf3d...a91` | **VERIFIED** | SHA-256: `c6cebf3d93d022b11765c206b8e2ffb84e578a020e0d913d937b37d04028aa91` |
| **Model Artifact Integrity** | Fitted models in `models/` untouched | **VERIFIED** | All 6 model joblib binaries remain byte-for-byte identical |
| **Configuration Manifest** | Original freeze manifest untouched | **VERIFIED** | `model_configuration_manifest.json` unmodified |
| **Unit Test Suite** | All unit tests passing | **VERIFIED** | 27 of 27 unit tests passing (100% green) |

---

## 9. METHODOLOGICAL LIMITATIONS & SCOPE REMINDERS

1. **Finite 250-Session Holdout:** Evaluated over one calendar year (`2025-10-20` to `2026-10-23`). While cross-sectionally broad (74,334 ticker-sessions), performance reflects the macroeconomic environment of that period.
2. **Predictive Performance vs. Trading Strategy:** This research demonstrates statistical risk prediction for high-risk liquidity events. It does not model execution slippage, bid-ask spreads, transaction taxes, short-sale borrowing costs, or portfolio alpha.
3. **No Retrospective Overfitting:** Holdout results have been used exclusively as evaluation evidence for finalizing the pre-registered selection decision. No further tuning, feature selection, or retraining has occurred or will be permitted.

---

## 10. TRANSITION GUIDANCE & STRICT STOP GATE

Stage 8.12.8 and Stage 8.12.9 are officially complete.

> [!IMPORTANT]
> **Strict Scope Boundary Enforcement:**
> Execution has **STOPPED IMMEDIATELY** after Stage 8.12.9.
> - **DO NOT** begin Stage 8.12.10 (Tests, artifact verification and reproducibility).
> - **DO NOT** begin Stage 8.12.11 (Comprehensive Phase 8 final documentation).
> - **DO NOT** deploy models or begin live inference.
> - All outputs, reports, manifests, and notebooks are frozen and await explicit authorization for Stage 8.12.10.
