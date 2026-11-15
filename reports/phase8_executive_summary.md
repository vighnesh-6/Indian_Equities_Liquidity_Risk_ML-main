# PHASE 8 EXECUTIVE SUMMARY
## ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities

**Project Title:** ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities  
**Phase:** Phase 8 — ML Modelling, Walk-Forward Validation & Model Selection  
**Target:** Forward 5-Day High-Risk Event (`hre_5d`, Horizon $H=5$)  
**Status:** COMPLETE, AUDITED & FROZEN  
**Date:** 2026-11-15  

---

### 1. Executive Overview
Phase 8 successfully designed, validated, and evaluated an institutional-grade machine learning system to forecast forward liquidity and trading risk across the NIFTY 500 equity universe. Using an expanding-window walk-forward validation framework with strict chronological purging and a quarantined 250-session final holdout (October 2025 – October 2026, $N=71,756$ valid labeled rows), five supervised model configurations and one unsupervised anomaly detector were rigorously evaluated.

---

### 2. Final Selected Model Architecture

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
│  Holdout PR-AUC: 0.5723 (93.7% Ret.)    │               │  Holdout Flag Rate: 2.34%    │
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
│  Holdout PR-AUC: 0.5380 (92.3% Ret.)    │ │  Holdout PR-AUC: 0.4051 (81.2% Ret.)       │
│  Holdout ROC-AUC: 0.8211                │ │  Holdout ROC-AUC: 0.7938                   │
│  Role: Primary Bagging Reference Model  │ │  Role: Linear Baseline Benchmark           │
└─────────────────────────────────────────┘ └────────────────────────────────────────────┘
```

---

### 3. Key Empirical Results

| Model Configuration | Feature Count | Walk-Forward Mean PR-AUC | Holdout PR-AUC | PR-AUC Retention % | Holdout ROC-AUC | Holdout Brier Score | Monthly Holdout Win Rate | Inference Latency | Selected Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`hist_gradient_boosting`** | **36** | **0.6107** | **0.5723** | **93.72%** | **0.8460** | **0.1167** | **13 / 13 (100%)** | **0.22 s** | **Champion Model** |
| **`random_forest_candidate_b`** | 36 | 0.5829 | 0.5380 | 92.30% | 0.8211 | 0.1249 | 0 / 13 (0%) | 0.11 s | **Strongest Alternative** |
| **`random_forest_candidate_a`** | 35 | 0.5814 | 0.5318 | 91.46% | 0.8186 | 0.1261 | 0 / 13 (0%) | 0.12 s | Baseline Tree |
| **`logistic_regression_standard`**| 20 | 0.4990 | 0.4051 | 81.18% | 0.7938 | 0.1298 | 0 / 13 (0%) | 0.02 s | **Linear Baseline** |
| **`logistic_regression_robust`**  | 20 | 0.4990 | 0.4050 | 81.17% | 0.7937 | 0.1298 | 0 / 13 (0%) | 0.02 s | Linear Baseline |

---

### 4. Core Takeaways & Insights
1. **Decisive Boosting Supremacy:** `hist_gradient_boosting` outperformed all competing architectures across every evaluation dimension, winning 100% of monthly holdout periods and retaining 93.72% of cross-validated performance.
2. **Rank Preservation:** Model rankings established in walk-forward validation were reproduced identically on the untouched holdout partition (Kendall's $\tau = +1.000$).
3. **Volume-Shock Predictive Power:** The inclusion of feature F51 (`relative_volume`) in Random Forest Candidate B delivered an empirical gain of **+0.62 percentage points in PR-AUC** over Candidate A, winning 12 of 13 monthly periods.
4. **Scaler Parity in Linear Models:** StandardScaler and RobustScaler yielded identical holdout PR-AUC ($\Delta = 0.000070$), confirming that scaling methodology has no material effect on L2-regularized linear models under median imputation.
5. **Orthogonal Surveillance Value:** Isolation Forest successfully flagged 2.34% of holdout sessions with an HRE enrichment ratio of **1.18x**, providing an independent microstructure dislocation tripwire.

---

### 5. Methodological Qualifications
- **Holdout Context:** Holdout performance was utilized as evaluation evidence for finalizing the pre-registered model selection decision; it must not be interpreted as an unbiased post-selection estimate.
- **Empirical Findings:** The superiority of Candidate B over Candidate A is an empirical observation across the evaluated holdout, not a formal statistical significance claim.
- **Threshold Provenance:** The operational Isolation Forest binary flag relies on the model's native training-set decision offset (`offset_`). The threshold value `43.12` corresponds to the empirical ~97.41th percentile on the holdout partition.
- **Research vs. Production:** Phase 8 concludes the research modelling and validation phase. Live execution, real-time feature streaming, and production deployment represent separate subsequent engineering milestones.

---

### 6. Deliverable & Artifact Verification
- **Dataset Checksums:** Verified 100% byte-for-byte identical (`features.parquet` & `features_with_targets.parquet`).
- **Trained Model Binaries:** All 6 model joblib files verified intact in `models/phase8_batch5/`.
- **Test Suite:** 27 of 27 automated unit tests passed (100% green).
- **Notebooks:** All 4 research notebooks (`05`–`08`) preserved read-only without re-execution.
- **Authoritative Report:** Comprehensive documentation codified in [`reports/phase8_final_report.md`](file:///C:/Users/vighn/Downloads/Indian_Equities_Liquidity_Risk_ML-main/Indian_Equities_Liquidity_Risk_ML-main/reports/phase8_final_report.md).
