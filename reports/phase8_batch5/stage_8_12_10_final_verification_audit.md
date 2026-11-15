# PHASE 8 BATCH 5 | STAGE 8.12.10: FINAL TESTS, ARTIFACT VERIFICATION & REPRODUCIBILITY AUDIT REPORT

**Project:** ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities  
**Phase:** Phase 8 Batch 5 (Stage 8.12.10)  
**Task:** Final Verification, Dataset Integrity, Artifact Audit & Reproducibility Assessment  
**Audit Scope:** End-to-End Verification of Stages 8.1 through 8.12.9  
**Audit Mode:** Strictly READ-ONLY (Zero Model Training, Zero Dataset Modifications, Zero Notebook Execution)  
**Status:** COMPLETE  
**Audit Verdict:** **PASS WITH WARNINGS** (All 10 Technical & Integrity Checks PASS; 4 Pre-Identified Methodological Qualifications Formally Documented)  
**Date:** 2026-11-15  

---

## 1. AUDIT SCOPE AND OBJECTIVES

Stage 8.12.10 executes a rigorous, independent, strictly read-only technical and reproducibility audit across all Phase 8 deliverables. The objective is to verify:
1. Strict preservation of frozen datasets, trained model artifacts, evaluation outputs, and research notebooks.
2. Complete internal consistency between configuration manifests, training metadata, walk-forward results, holdout results, and final model selection decisions.
3. Offline execution of automated unit tests covering preprocessing, purging, and evaluation modules.
4. Transparent documentation and formal flagging of all known methodological qualifications prior to final Phase 8 closure.

---

## 2. PRESERVATION COMPLIANCE AUDIT

| Preservation Rule | Compliance Mandate | Audit Finding | Status |
| :--- | :--- | :--- | :---: |
| **Datasets** | No modification to raw, interim, processed, or normalized datasets | Hashes verified byte-for-byte identical to baseline | **COMPLIANT** |
| **Model Binaries** | No retraining, refitting, reserialization, or modification of joblib files | File sizes and SHA-256 checksums verified identical | **COMPLIANT** |
| **Notebooks** | Zero execution, regeneration, or cell appending to existing notebooks | All 4 notebooks inspected; zero execution performed | **COMPLIANT** |
| **Batch 4 Artifacts** | Walk-forward cross-validation results remain immutable | Parquet and CSV files intact with original row counts | **COMPLIANT** |
| **Batch 5 Artifacts** | Holdout predictions and comparison tables remain immutable | Parquet and CSV files intact with original row counts | **COMPLIANT** |
| **Source Code** | Zero modification to modeling, preprocessing, evaluation scripts | `git status` confirms zero unauthorized source edits | **COMPLIANT** |

---

## 3. DATASET INTEGRITY VERIFICATION

Dataset integrity was verified by recalculating the SHA-256 cryptographic hashes for the core processed datasets:

| Dataset Path | Expected Canonical Hash | Verified Hash | Integrity Status |
| :--- | :--- | :--- | :---: |
| `data/processed/features.parquet` | `c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae` | `c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae` | **100% MATCH** |
| `data/processed/features_with_targets.parquet` | `c6cebf3d93d022b11765c206b8e2ffb84e578a020e0d913d937b37d04028aa91` | `c6cebf3d93d022b11765c206b8e2ffb84e578a020e0d913d937b37d04028aa91` | **100% MATCH** |

Both core feature datasets remain byte-for-byte pristine. No feature values, column schemas, index records, or target labels were modified.

---

## 4. MODEL ARTIFACT INVENTORY AND CHECKSUMS

All six trained model artifacts persisted under `models/phase8_batch5/` during Stage 8.12.4 were audited for presence, directory structure, metadata consistency, and cryptographic checksums:

| Configuration ID | Model Name | Feature Count | Artifact File Path | File Size (Bytes) | SHA-256 Checksum | Metadata Status |
| :--- | :--- | :---: | :--- | :---: | :--- | :---: |
| `CFG_01_LOGREG_STANDARD` | `logistic_regression_standard` | 20 | `models/phase8_batch5/logistic_regression_standard/model.joblib` | 3,025 | `728ecff89c60d56807b8597e6b809572ca2d5d7a7199089f2877336c7a302e43` | Verified |
| `CFG_02_LOGREG_ROBUST` | `logistic_regression_robust` | 20 | `models/phase8_batch5/logistic_regression_robust/model.joblib` | 2,721 | `dc8b2bf7c7303e74d43ce18f34d8b35bfe064c94b5ac97e0db4fe8e6f0d2b2c1` | Verified |
| `CFG_03_RF_CANDIDATE_A` | `random_forest_candidate_a` | 35 | `models/phase8_batch5/random_forest_candidate_a/model.joblib` | 5,658,514 | `dfae7f66cedc8d5ae25514c6fa25d9c152b6695bca12f6b53af12e2e7cbdbbd8` | Verified |
| `CFG_04_RF_CANDIDATE_B` | `random_forest_candidate_b` | 36 | `models/phase8_batch5/random_forest_candidate_b/model.joblib` | 5,740,146 | `f68bff36a3046a272c8ecdc8e2d87fab8ec9ea5480bee27f37522675ee1a573a` | Verified |
| `CFG_05_HGBT` | `hist_gradient_boosting` | 36 | `models/phase8_batch5/hist_gradient_boosting/model.joblib` | 803,729 | `7848d911b5ca8afb04114675d5507ac91f104efaca57a47a44706e8fd37921b9` | Verified |
| `CFG_06_IF` | `isolation_forest` | 12 | `models/phase8_batch5/isolation_forest/model.joblib` | 903,812 | `9d5a05088dfb2083da2fce9bd430d8ca6f1e752f7bd8f8a7016be14a02de7377` | Verified |

Each model directory contains its corresponding `config.json` and `feature_names.json`, aligning 100% with the frozen manifest. Zero models were refitted or re-serialized.

---

## 5. EVALUATION ARTIFACT INVENTORY AND INTEGRITY

### 5.1 Batch 4 Walk-Forward Artifacts (`data/interim/phase8_batch4/`)
- `supervised_model_comparison.csv`: Verified ($5 \times 25$, 2,765 bytes). Summary cross-validation metrics across all 10 expanding folds.
- `supervised_fold_metrics.csv`: Verified ($50 \times 40$, 24,973 bytes). Granular fold-level metrics for all 5 supervised models.
- `supervised_predictions.parquet`: Verified ($163,357 \times 20$, 9.36 MB). Out-of-fold probability predictions and true labels.
- `isolation_forest_fold_metrics.csv`: Verified ($10 \times 22$, 3,340 bytes). 10-fold anomaly telemetry distributions.
- `isolation_forest_predictions.parquet`: Verified ($163,357 \times 8$, 4.37 MB). Unsupervised anomaly scores and binary flags.

### 5.2 Batch 5 Holdout Artifacts (`data/interim/phase8_batch5/holdout/`)
- `supervised_holdout_predictions.parquet`: Verified ($74,334 \times 19$, 3.84 MB). Full holdout predictions across all 5 models.
- `supervised_holdout_metrics.csv`: Verified ($5 \times 28$, 2,347 bytes). Primary holdout leaderboard metrics, calibration, and market-cap tiers.
- `supervised_holdout_metrics_by_model.csv`: Verified ($5 \times 28$, 2,347 bytes). Standardized copy for reporting.
- `isolation_forest_holdout_predictions.parquet`: Verified ($74,334 \times 7$, 1.53 MB). Holdout anomaly scores, scaled scores, and flags.
- `isolation_forest_holdout_metrics.csv`: Verified ($1 \times 20$, 601 bytes). Holdout anomaly distribution, flag rate, and HRE enrichment.
- `walkforward_vs_holdout_comparison.csv`: Verified ($5 \times 32$, 3,410 bytes). Comprehensive walk-forward vs holdout comparison table.
- `isolation_forest_walkforward_vs_holdout.csv`: Verified ($8 \times 3$, 512 bytes). Comparative distribution statistics.

### 5.3 Manifest Files (`reports/phase8_batch5/`)
- `configuration_freeze/model_configuration_manifest.json`: Verified (12.2 KB). Authoritative freeze of all 6 model configurations.
- `final_model_selection_manifest.json`: Verified (4.8 KB). Authoritative machine-readable final selection decision record.

---

## 6. NOTEBOOK PRESERVATION AUDIT

All four research notebooks were inspected to ensure complete preservation. In strict compliance with Part 1 Rule 3:
- **No notebook was executed.**
- **No cells were added, modified, or regenerated.**
- All historical outputs, execution counts, and markdown structures remain unchanged:

| Notebook Path | Total Cells | Structure Status | Historical Outputs | Execution Status |
| :--- | :---: | :---: | :---: | :---: |
| `notebooks/05_model_baseline.ipynb` | 19 | 9 Sections (Sec 1–9) | 100% Preserved | **NOT EXECUTED (Read-Only)** |
| `notebooks/06_model_rf.ipynb` | 17 | 8 Sections (Sec 1–8) | 100% Preserved | **NOT EXECUTED (Read-Only)** |
| `notebooks/07_model_hgbt.ipynb` | 15 | 7 Sections (Sec 1–7) | 100% Preserved | **NOT EXECUTED (Read-Only)** |
| `notebooks/08_anomaly_detection.ipynb` | 15 | 7 Sections (Sec 1–7) | 100% Preserved | **NOT EXECUTED (Read-Only)** |

---

## 7. AUTOMATED TEST SUITE EXECUTION RESULTS

The offline automated unit test suite covering Phase 8 preprocessing, purging, and evaluation modules was executed via pytest:

```bash
pytest tests/test_evaluation.py tests/test_preprocessing.py tests/test_walk_forward_purging.py -v
```

### Test Results Summary:
- **Platform:** Windows (Python 3.13.5, pytest-8.3.4, pluggy-1.5.0)
- **Total Tests Collected:** 27
- **Passed:** **27 (100%)**
- **Failed:** **0**
- **Skipped:** **0**
- **Execution Time:** 12.02 seconds
- **Warnings:** 1 non-fatal informational warning regarding `loky` logical core detection on Windows.

### Test Coverage Detail:
1. `tests/test_evaluation.py` (4 tests):
   - `test_compute_classification_metrics`: PASS
   - `test_compute_threshold_sweep`: PASS
   - `test_find_optimal_threshold`: PASS
   - `test_compute_brier_and_calibration`: PASS
2. `tests/test_preprocessing.py` (15 tests):
   - Scaling, median imputation, clipping, robust quantile scaling, native NaN routing, and feature validation: ALL PASS.
3. `tests/test_walk_forward_purging.py` (8 tests):
   - Chronological split generation, expanding train windows, 5-session purge boundaries, horizon $H=5$ leakage prevention: ALL PASS.

---

## 8. REPRODUCIBILITY & CROSS-ARTIFACT CONSISTENCY AUDIT

Cross-checking metrics across manifests, CSVs, reports, and training metadata confirms complete numerical reconciliation:

1. **Supervised Leaderboard Alignment:**
   - HGBT Holdout PR-AUC: Reported as **0.572332** across `supervised_holdout_metrics.csv`, `walkforward_vs_holdout_comparison.csv`, `final_model_selection_manifest.json`, and all Stage 8.12.5–8.12.9 reports.
   - HGBT Holdout ROC-AUC: **0.845966** (100% consistent).
   - HGBT Brier Score: **0.116693** (100% consistent).
   - HGBT Log Loss: **0.362502** (100% consistent).
2. **Alternative & Baseline Alignment:**
   - RF Candidate B Holdout PR-AUC: **0.537998** (100% consistent).
   - RF Candidate A Holdout PR-AUC: **0.531756** (100% consistent).
   - Logistic Regression Standard Holdout PR-AUC: **0.405109** (100% consistent).
   - Logistic Regression Robust Holdout PR-AUC: **0.405039** (100% consistent).
3. **Taxonomy Consistency:**
   - All reports strictly maintain the distinction: **3 supervised algorithmic families**, **5 evaluated supervised configurations**, and **1 independent unsupervised anomaly detection family**.

---

## 9. FORMAL METHODOLOGICAL QUALIFICATIONS

In accordance with rigorous quantitative audit governance, the following four methodological qualifications are formally documented:

### Qualification 1 — Holdout-Based Final Model Selection
* **Finding:** While candidate configurations were frozen prior to holdout evaluation, final model selection (`hist_gradient_boosting` designated as champion) incorporated the empirical holdout performance metrics alongside walk-forward cross-validation evidence.
* **Audit Implication:** The holdout dataset served as decisive evaluation evidence for the terminal selection decision. Therefore, holdout metrics must not be described as an "independent, unbiased post-selection generalization estimate." They represent terminal out-of-sample evaluation within the pre-registered research protocol.

### Qualification 2 — Isolation Forest Threshold Provenance
* **Finding:** Narrative summary text in early Batch 2 documentation cited an anomaly threshold of `43.12`. Audit of the actual codebase (`src/modeling/anomaly.py`) reveals that operational binary flags (`isolation_flag == 1`) were generated strictly using the pre-fitted estimator's internal decision offset (`offset_`), established on pre-holdout training data with `contamination=0.05`.
* **Audit Implication:** On the holdout dataset, the native decision boundary corresponds to a scaled score cutoff of approximately `44.44` (flagging 1,738 rows = 2.34%). The value `43.12` corresponds to the empirical ~97.41th percentile of the holdout distribution itself. It must NOT be described as an independently calibrated pre-holdout threshold. The native decision offset is the true operational boundary.

### Qualification 3 — Random Forest Candidate B Performance Edge
* **Finding:** Random Forest Candidate B (36 features, including F51 `relative_volume`) outperformed Candidate A (35 features, excluding F51) by +0.0062 PR-AUC on the holdout and won 12 of 13 monthly periods.
* **Audit Implication:** This performance advantage is an **empirical observation**. No formal null-hypothesis statistical significance test (such as Diebold-Mariano or bootstrap paired test) was conducted. It must be described strictly as an empirical out-of-sample improvement, not a statistically proven significance claim.

### Qualification 4 — Research Model Selection vs. Production Deployment
* **Finding:** Phase 8 successfully concludes the research modelling, cross-validation, holdout evaluation, and architecture selection.
* **Audit Implication:** Model selection does NOT constitute live production deployment. Production deployment requires live API integration, market feed handlers, real-time feature streaming, trade execution safeguards, and automated drift monitoring, which reside outside the Phase 8 research scope.

---

## 10. FINAL AUDIT VERDICT

### **VERDICT: PASS WITH WARNINGS**

- **Technical Integrity (PASS):** All 10 technical checks (dataset hashes, model binaries, evaluation artifacts, notebook preservation, test suite execution, cross-table consistency) passed with 100% compliance.
- **Methodological Qualifications (WARNINGS):** The four pre-identified qualifications regarding holdout selection context, Isolation Forest threshold provenance, empirical versus statistical claims, and research-to-production boundaries are formally recognized and documented.

The repository is certified fully compliant and ready to proceed to **Stage 8.12.11: Comprehensive Phase 8 Final Documentation**.
