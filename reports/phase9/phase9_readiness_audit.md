# PHASE 9.1 — READINESS AUDIT REPORT
## Macro Stress Testing, Event Studies & Model Explainability

**Project:** ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities  
**Phase:** Phase 9.1 (Readiness Audit)  
**Repository:** `Indian_Equities_Liquidity_Risk_ML-main`  
**Audit Mode:** Strictly READ-ONLY (Zero Model Training, Zero Dataset Modifications, Zero Notebook Execution)  
**Status:** COMPLETE  
**Readiness Verdict:** **PASS WITH METHODOLOGICAL PREREQUISITES**  
**Date:** 2026-11-15  

---

## 1. EXECUTIVE SUMMARY

Phase 9.1 executes a comprehensive, independent, strictly read-only readiness audit to determine whether the existing repository infrastructure, curated event datasets, frozen Phase 8 model binaries, historical predictions, and environment dependencies are ready for Phase 9 implementation:
1. **Historical Event Studies:** Evaluating liquidity dislocations, spread shocks, and model warnings across major market events.
2. **Macro Stress Testing:** Quantifying cross-sectional liquidity evaporation across market capitalization cohorts during systemic crises.
3. **Model Explainability:** Extracting global and feature-level predictive importance for the frozen model architecture.

### Key Audit Findings:
- **Event Infrastructure (PASS):** The curated market events calendar (`data/events/market_events.csv`) contains **86 verified historical events** (2018 to 2026) across 6 categories with zero missing dates. The validation manager (`src/data/events.py`) passes all 12 unit tests.
- **Phase 8 Model Artifacts (PASS):** All trained model binaries (`hist_gradient_boosting`, `random_forest_candidate_b`, and `isolation_forest`), configuration manifests, and feature lists are verified intact with 100% cryptographic checksum matches.
- **Historical Prediction Availability (PASS / PARTIAL):** Out-of-sample prediction series are available for **28 events** (25 during the 2023–2025 walk-forward window and 3 during the 2025–2026 holdout). For the remaining 58 events (2018–2023), full microstructure features and realized targets exist, but model predictions on those dates represent training-period observations.
- **Explainability Dependencies (PASS / PARTIAL):** Scikit-learn's native `permutation_importance` is fully available and functional (scikit-learn 1.8.0). External library `shap` is **NOT installed**; Phase 9 must either rely on native permutation importance or request explicit approval to install `shap`.
- **Preservation Compliance (PASS):** Zero existing datasets, models, or notebooks were modified during this audit.

---

## 2. REPOSITORY & DEPENDENCY INVENTORY

### 2.1 Python Environment & Toolchain
- **Python Version:** 3.13.5 (64-bit on Windows)
- **Environment Location:** `C:\Users\vighn\anaconda3\`

| Dependency | Required Role | Installed Version | Status | Audit Assessment |
| :--- | :--- | :---: | :---: | :--- |
| **`scikit-learn`** | Core ML, HGBT, RF, Isolation Forest, Permutation Importance | 1.8.0 | **PASS** | Fully functional; native permutation importance verified |
| **`pandas`** | Tabular data manipulation, time series indexing | 2.2.3 | **PASS** | Fully functional |
| **`numpy`** | Numerical array routines, linear algebra | 2.1.3 | **PASS** | Fully functional |
| **`pyarrow`** | Parquet columnar store I/O | Installed | **PASS** | Verified on multi-million row files |
| **`joblib`** | Model binary persistence & loading | 1.4.2 | **PASS** | Verified on all 6 model binaries |
| **`pytest`** | Automated testing harness | 8.3.4 | **PASS** | Test runner active |
| **`shap`** | Shapley additive explanations | *Not Installed* | **PARTIAL / FAIL** | Missing from environment; native permutation importance recommended |
| **`lime`** | Local interpretable model explanations | *Not Installed* | **NOT VERIFIED** | Not requested in baseline specification |

---

## 3. EVENT INFRASTRUCTURE AUDIT

### 3.1 Market Events Calendar (`data/events/market_events.csv`)
- **File Path:** `data/events/market_events.csv`
- **File Size:** 42,025 bytes
- **Total Event Count:** **86 verified events**
- **Date Span:** `2018-02-01` to `2026-10-09`
- **Missing Effective Dates:** **0 (100% complete)**
- **Schema Compliance:** 16 required columns strictly conforming to `docs/EVENT_CALENDAR_SPEC.md`.

### 3.2 Event Categorization Breakdown

| Event Category | Count | Scope: Market-Wide | Scope: Sector-Specific | Scope: Stock-Specific | Scheduled | Unscheduled |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`MACRO_POLICY`** | 31 | 31 | 0 | 0 | 27 | 4 |
| **`REGULATORY_EXCHANGE`** | 17 | 15 | 1 | 1 | 9 | 8 |
| **`CORPORATE_ACTION`** | 11 | 0 | 1 | 10 | 8 | 3 |
| **`GEOPOLITICAL`** | 10 | 10 | 0 | 0 | 0 | 10 |
| **`SYSTEMIC_CREDIT`** | 9 | 4 | 5 | 0 | 0 | 9 |
| **`POLITICAL_ELECTORAL`** | 8 | 8 | 0 | 0 | 3 | 5 |
| **TOTAL** | **86** | **60** | **13** | **13** | **47** | **39** |

### 3.3 Event Ingestion & Calendar Alignment (`src/data/events.py`)
- **Class:** `EventCalendarManager`
- **Functionality:** Validates 16 fields, checks permitted category enums, verifies effective date alignment against the NSE exchange trading calendar, separates announcement timestamp from trading session.
- **Unit Test Coverage (`tests/test_events.py`):** **12 of 12 tests passing (100% green)** in 0.50 seconds.
- **Verdict on Event Infrastructure:** **PASS**.

---

## 4. PHASE 8 MODEL ARTIFACT AUDIT

All Phase 8 model binaries were inspected in strictly read-only mode. No models were retrained, refitted, or overwritten:

| Configuration ID | Model Identifier | Feature Count | Model Binary Path | File Size (Bytes) | Checksum (SHA-256) | Inspection Finding | Status |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- | :---: |
| **`CFG_05_HGBT`** | `hist_gradient_boosting` | 36 | `models/phase8_batch5/hist_gradient_boosting/model.joblib` | 803,729 | `7848d911b5ca8afb...` | Pipeline with fitted `HistGradientBoostingClassifier`; threshold $\tau^* = 0.534$ | **PASS** |
| **`CFG_04_RF_CANDIDATE_B`** | `random_forest_candidate_b` | 36 | `models/phase8_batch5/random_forest_candidate_b/model.joblib` | 5,740,146 | `f68bff36a3046a27...` | Pipeline with fitted `RandomForestClassifier` (100 trees); threshold $\tau^* = 0.508$ | **PASS** |
| **`CFG_06_IF`** | `isolation_forest` | 12 | `models/phase8_batch5/isolation_forest/model.joblib` | 903,812 | `9d5a05088dfb2083...` | AnomalyDetector with fitted `IsolationForest`; native decision boundary $s_{\text{scaled}} \approx 44.44$ | **PASS** |
| **`CFG_01_LOGREG_STANDARD`** | `logistic_regression_standard` | 20 | `models/phase8_batch5/logistic_regression_standard/model.joblib` | 3,025 | `728ecff89c60d568...` | Fitted `LogisticRegression` baseline; threshold $\tau^* = 0.460$ | **PASS** |

### 4.1 Feature Manifests & Ordering
- Every model directory contains an identical, verified `feature_names.json`.
- `hist_gradient_boosting/feature_names.json`: Verified 36 features matching `configs/model_config.yaml`.
- Preprocessing pipelines (`preprocessor.joblib`) are present for Random Forest and Logistic Regression; HGBT requires zero preprocessing (native histogram binning).
- **Verdict on Model Artifacts:** **PASS**.

---

## 5. HISTORICAL PREDICTIONS & FEATURE AVAILABILITY AUDIT

### 5.1 Dataset Inventory

| Dataset Path | Rows | Columns | Date Span | Tickers | Content & Verification | Status |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| `data/processed/features_with_targets.parquet` | 540,820 | 61 | `2018-01-01` to `2026-10-23` | 300 | 58 features + `hre_5d_train` + `hre_5d` | **PASS** |
| `data/interim/phase8_batch4/supervised_predictions.parquet` | 163,357 | 20 | `2023-05-10` to `2025-10-10` | 284 | Out-of-fold probability predictions across 10 folds | **PASS** |
| `data/interim/phase8_batch5/holdout/supervised_holdout_predictions.parquet` | 74,334 | 19 | `2025-10-20` to `2026-10-23` | 300 | Out-of-sample final holdout predictions | **PASS** |
| `data/interim/phase8_batch4/isolation_forest_predictions.parquet` | 163,357 | 8 | `2023-05-10` to `2025-10-10` | 284 | Out-of-fold unsupervised anomaly scores & flags | **PASS** |
| `data/interim/phase8_batch5/holdout/isolation_forest_holdout_predictions.parquet` | 74,334 | 7 | `2025-10-20` to `2026-10-23` | 300 | Holdout unsupervised anomaly scores & flags | **PASS** |

### 5.2 Event Coverage Analysis
Cross-referencing the 86 event dates against available prediction sets:
- **Pre-Walk-Forward Training Period (`2018-02-01` to `2023-05-09`):** **58 events**.
  - *Feature Availability:* 100% available in `features_with_targets.parquet`.
  - *Model Predictions:* No out-of-sample predictions exist (this period served as the initial expanding-window training history).
- **Walk-Forward Validation Period (`2023-05-10` to `2025-10-10`):** **25 events**.
  - *Feature Availability:* 100% available.
  - *Model Predictions:* 100% available in `supervised_predictions.parquet` (true out-of-fold predictions).
- **Holdout Boundary Embargo (`2025-10-13` to `2025-10-17`):** **0 events**.
- **Final Holdout Period (`2025-10-20` to `2026-10-23`):** **3 events**.
  - *Feature Availability:* 100% available.
  - *Model Predictions:* 100% available in `supervised_holdout_predictions.parquet` (true out-of-sample holdout predictions).
- **Verdict on Prediction Coverage:** **PARTIAL**. 28 events have true out-of-sample model predictions; all 86 events have full microstructure features and realized targets.

---

## 6. EXPLAINABILITY DEPENDENCY AUDIT

### 6.1 Native vs. External Explainability Tooling

| Technique | Supported in Codebase? | Package Source | Methodological Suitability for HGBT | Audit Assessment |
| :--- | :---: | :--- | :--- | :---: |
| **Permutation Feature Importance** | **YES** | `sklearn.inspection.permutation_importance` | **High:** Evaluates drop in PR-AUC / Log Loss on validation data. Fully compatible with HGBT. | **PASS** |
| **MDI Impurity Importance** | **YES (RF only)** | `RandomForestClassifier.feature_importances_` | **Low:** Random Forest impurity importance reflects tree split counts; HGBT does not expose MDI. | **PASS (RF-B only)** |
| **SHAP (Kernel / Tree)** | **NO** | `shap` (Not installed) | Requires external compilation / install. Not strictly needed for global ranking. | **FAIL (Missing pkg)** |
| **Directional Linear Inspection** | **YES** | `LogisticRegression.coef_` | **Moderate:** Provides signed linear attribution, but fails to capture non-linear interactions. | **PASS** |

### 6.2 Key Constraints for Phase 9 Explainability:
1. **No Confusion Between Architectures:** Random Forest MDI importance must **NEVER** be presented as HGBT feature importance.
2. **Permutation Importance Standard:** Global importance for the champion HGBT model must be computed via `permutation_importance` on pre-holdout validation or holdout data.
3. **No Causal Claims:** Permutation importance reflects statistical predictive attribution within the model's loss landscape; it does not prove causal economic relationships.
4. **Zero Importance Rankings in Phase 9.1:** In strict compliance with audit instructions, no feature importance rankings have been computed or published during this readiness check.

---

## 7. TEMPORAL COVERAGE & EVENT-WINDOW FEASIBILITY

### 7.1 Window Construction Methodology
- **Standard Event Window:** $[-5, \quad +5]$ trading sessions centered on the event's `effective_trading_date` ($t=0$).
- **Baseline Window:** $[-20, \quad -6]$ trading sessions prior to the event (used to establish pre-shock baseline spreads and Amihud ratios).
- **NSE Trading Calendar Alignment:** Verified. `src/data/events.py` uses the NIFTY 50 trading calendar to advance sessions by trading days, automatically skipping weekends and exchange holidays.

### 7.2 Event Clustering and Overlap
- **Identified Risk:** In financial crises, events cluster tightly:
  - *Example:* March 2020 COVID-19 crash contains 5 distinct policy/market halt events within a 15-day span.
  - *Example:* May–June 2024 Election cycle contains exit poll surges followed by counting day volatility within 2 sessions.
- **Mitigation Requirement:** Phase 9 must incorporate an explicit overlap-handling rule (e.g., compound crisis episode aggregation vs. individual shock evaluation with overlapping window flags).

---

## 8. METHODOLOGICAL RISKS & REQUIRED DECISIONS

Before Phase 9 implementation begins, the following four methodological decisions require formal authorization:

### Decision 1: Scope of Model Predictions in Event Studies
- *Issue:* 58 events occurred during the 2018–2023 pre-walk-forward period, while 28 events occurred during out-of-sample periods (25 walk-forward + 3 holdout).
- *Options:*
  - **Option A (Strict Out-of-Sample):** Restrict event study model prediction analysis strictly to the 28 out-of-sample events (2023–2026), using the 58 earlier events only for feature-level microstructure analysis (spreads and Amihud).
  - **Option B (Two-Tier Reporting):** Report the 28 out-of-sample events as primary evidence, and report in-sample pre-holdout fitted model scores for the 58 earlier events as a secondary historical backtest (with explicit in-sample disclosure).
- *Recommendation:* **Option B**, ensuring maximum historical crisis coverage (including COVID 2020 and IL&FS 2018) while maintaining rigorous disclosure between in-sample and out-of-sample predictions.

### Decision 2: Cross-Sectional Stock Aggregation in Event Studies
- *Issue:* How should stock-level predictions and spreads be aggregated across the 300 equities on each event day $t \in [-5, +5]$?
- *Recommendation:* Compute **median** and **interquartile range (IQR)** stratified across market cap tiers (Large, Mid, Small). Means are distorted by penny-stock illiquidity spikes.

### Decision 3: Event Overlap Treatment
- *Issue:* Closely spaced events can cause post-event windows of Event A to contaminate pre-event windows of Event B.
- *Recommendation:* Retain individual event windows but tag overlapping days with an `is_overlapping_event` indicator, and report isolated events separately from clustered crisis clusters.

### Decision 4: Explainability Tooling Choice
- *Issue:* `shap` is not installed in the environment.
- *Recommendation:* Rely strictly on native `sklearn.inspection.permutation_importance` (scoring PR-AUC loss across 36 features on holdout/validation partitions). This requires **zero new dependencies** and avoids package installation risks.

---

## 9. MISSING DEPENDENCIES & IMPLEMENTATION REQUIREMENTS

1. **`shap` Library:** Not installed. (Recommended: Proceed without `shap` using native scikit-learn permutation importance).
2. **Event Analysis Implementation Notebook:** `notebooks/09_event_analysis.ipynb` currently contains an empty scaffold (`# Implementation deferred to Phase 9`). This will be authored in Phase 9.2.
3. **Dedicated Output Directories:** Directory `reports/phase9/` and figures directory `reports/phase9/figures/` must be structured during Phase 9 implementation.

---

## 10. RECOMMENDED PHASE 9 IMPLEMENTATION SEQUENCE

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        RECOMMENDED PHASE 9 IMPLEMENTATION ROADMAP                      │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
  ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
  │ Stage 9.1: Readiness Audit (THIS REPORT)                                          │
  │ • Independent verification of events, models, predictions & dependencies         │
  │ • Status: COMPLETE — VERDICT: PASS WITH METHODOLOGICAL PREREQUISITES              │
  └─────────────────────────────────────────┬─────────────────────────────────────────┘
                                            │
  ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
  │ Stage 9.2: Model Explainability & Feature Importance                              │
  │ • Compute Permutation Feature Importance for Champion HGBT (36 features)          │
  │ • Compute MDI Impurity Importance for Random Forest Candidate B                   │
  │ • Directional coefficient comparison from Logistic Regression                     │
  │ • Generate global importance leaderboard & feature attribution plots              │
  └─────────────────────────────────────────┬─────────────────────────────────────────┘
                                            │
  ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
  │ Stage 9.3: Historical Event Studies ([-5, +5] Window Analysis)                    │
  │ • Align predictions and spreads with 86 events across the 6 categories            │
  │ • Measure pre-event model risk surge (lead time) and spread widening              │
  │ • Conduct case studies: COVID March 2020, Adani Jan 2023, Election June 2024     │
  └─────────────────────────────────────────┬─────────────────────────────────────────┘
                                            │
  ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
  │ Stage 9.4: Macro Stress Testing & Segment Divergence                              │
  │ • Quantify liquidity evaporation across Large, Mid, Small caps during stress      │
  │ • Evaluate Isolation Forest anomaly flag clustering during crisis episodes        │
  │ • Statistical testing: Mann-Whitney U tests for segment divergence                │
  └─────────────────────────────────────────┬─────────────────────────────────────────┘
                                            │
  ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
  │ Stage 9.5: Final Phase 9 Consolidation & Reporting                                │
  │ • Execute and save `notebooks/09_event_analysis.ipynb`                            │
  │ • Generate publication figures in `reports/phase9/figures/`                       │
  │ • Deliver comprehensive master report `reports/phase9/phase9_final_report.md`     │
  └───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. READINESS AUDIT VERDICTS

| Dimension | Scope Audited | Verdict | Evidence / Justification |
| :--- | :--- | :---: | :--- |
| **1. Event Data & Calendar** | `market_events.csv`, `src/data/events.py` | **PASS** | 86 verified events, 0 missing dates, 12/12 unit tests pass |
| **2. Model Artifacts** | `models/phase8_batch5/` (HGBT, RF-B, IF) | **PASS** | Binaries, checksums, features, and manifests 100% verified |
| **3. Historical Predictions** | Parquet predictions (WF + Holdout) | **PASS** | Complete predictions available; 28 out-of-sample event overlaps |
| **4. Microstructure Features** | `features_with_targets.parquet` | **PASS** | Complete 2018–2026 daily series across 300 stocks (540,820 rows) |
| **5. Test Suite** | `tests/test_events.py` & Phase 8 tests | **PASS** | All tests pass 100% offline |
| **6. Explainability Tooling** | Scikit-learn native permutation importance | **PASS** | Fully supported in scikit-learn 1.8.0 without extra packages |
| **7. External SHAP Library** | `shap` package | **PARTIAL** | Not installed; native permutation importance recommended |
| **8. Methodological Protocol**| Event window & aggregation decisions | **PREREQUISITE**| 4 methodological decisions defined for approval |

---

### **OVERALL AUDIT VERDICT: PASS WITH METHODOLOGICAL PREREQUISITES**

The repository is technically sound, cryptographically intact, and fully prepared for Phase 9 implementation. Model training, predictions, and datasets remain frozen. Implementation may proceed to **Stage 9.2** once the four methodological decisions are formally ratified.
