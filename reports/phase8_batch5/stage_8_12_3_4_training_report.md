# PHASE 8 BATCH 5 | STAGES 8.12.3 & 8.12.4: CONFIGURATION FREEZE & PRE-HOLDOUT TRAINING REPORT

**Project:** ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities  
**Phase:** Phase 8 Batch 5 (Batch 1: Stages 8.12.3 & 8.12.4)  
**Execution Date:** 2026-11-15  
**Target Variable:** Forward 5-Day High-Risk Event (`hre_5d_train`, Horizon $H=5$)  
**Status:** COMPLETE & VERIFIED  

---

## 1. STAGE 8.12.3: CONFIGURATION-FREEZE SUMMARY

Stage 8.12.3 formally executed the configuration freeze across all candidate models based on the approved selection methodology established in `reports/phase8_batch5_stage8_12_2_selection_methodology.md`. 

Five supervised configurations (representing three distinct algorithmic families) and one independent unsupervised anomaly detection configuration were locked without alteration. A versioned, machine-readable configuration manifest was generated and persisted to:
`reports/phase8_batch5/configuration_freeze/model_configuration_manifest.json`

The companion freeze report was documented at:
`reports/phase8_batch5/stage_8_12_3_configuration_freeze.md`

All feature inventories, imputation pipelines, scaling transformations, and hyperparameter dictionaries were locked to prevent post-hoc tuning or methodology drift.

---

## 2. ALL FIVE SUPERVISED CONFIGURATIONS AND THEIR FROZEN SETTINGS

| Parameter / Dimension | Configuration 1 | Configuration 2 | Configuration 3 | Configuration 4 | Configuration 5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Identifier** | `logistic_regression_standard` | `logistic_regression_robust` | `random_forest_candidate_a` | `random_forest_candidate_b` | `hist_gradient_boosting` |
| **Config ID** | `CFG_01_LR_STANDARD` | `CFG_02_LR_ROBUST` | `CFG_03_RF_CAND_A` | `CFG_04_RF_CAND_B` | `CFG_05_HGBT` |
| **Algorithm Family** | Linear Classification | Linear Classification | Supervised Ensemble | Supervised Ensemble | Gradient Tree Boosting |
| **Estimator Class** | `LogisticRegression` | `LogisticRegression` | `RandomForestClassifier` | `RandomForestClassifier` | `HistGradientBoostingClassifier` |
| **Feature Count** | 20 features | 20 features | 35 features | 36 features | 36 features |
| **Feature Subset** | Linear-screened subset | Linear-screened subset | Excludes F51 (`relative_volume`) | Includes F51 (`relative_volume`) | Includes F51 (`relative_volume`) |
| **Missing-Value Strategy** | `SimpleImputer(median)` | `SimpleImputer(median)` | Native Tree Routing (NaN) | Native Tree Routing (NaN) | Native Histogram Binning |
| **Feature Scaling** | `StandardScaler` ($z$-score) | `RobustScaler` (Median/IQR) | None (Scale-invariant) | None (Scale-invariant) | None (Scale-invariant) |
| **Class Weighting** | `balanced` | `balanced` | `balanced_subsample` | `balanced_subsample` | `balanced` |
| **Key Hyperparameters** | $C=1.0$, L2, max_iter=1000 | $C=1.0$, L2, max_iter=1000 | $N=150$, depth=8, leaf=50 | $N=150$, depth=8, leaf=50 | $N=200$, lr=0.05, leaves=31, reg=0.1 |
| **Random Seed** | 42 | 42 | 42 | 42 | 42 |
| **Role in Project** | Linear Benchmark Baseline | Scaling Sensitivity Baseline | Tree Baseline (no F51) | Tree Candidate (with F51) | **Primary Supervised Champion** |

---

## 3. SEPARATE ISOLATION FOREST CONFIGURATION

Isolation Forest is frozen strictly as an **independent, companion unsupervised anomaly detector**, completely uncoupled from the supervised classification models:
- **Configuration Identifier:** `CFG_06_IF_UNSUPERVISED`
- **Model Name:** `isolation_forest`
- **Algorithm Family:** Unsupervised Anomaly Detection (`sklearn.ensemble.IsolationForest`)
- **Feature Count:** 12 curated liquidity, spread, volume, and volatility features:
  `amihud_ratio`, `amihud_z20`, `volume_ratio`, `volume_z20`, `vol_20d`, `vol_ratio_5_20`, `daily_range`, `range_z20`, `overnight_gap_abs`, `cs_spread_est`, `nifty_vol_20d`, `return_z20`.
- **Target Independence:** **Zero target supervision**. The detector is fitted completely without `hre_5d_train` labels, preserving unlabelled microstructure dynamics.
- **Preprocessing Pipeline:** `SimpleImputer(strategy='median')` to guarantee finite feature bounds for random orthogonal partitioning. No feature scaling is applied (axis-aligned partitioning is strictly monotonic-scale invariant).
- **Hyperparameters:** $N=100$ isolation trees, `contamination=0.05`, `max_samples='auto'`, `random_state=42`.
- **Output Mapping:** Decision scores are normalized to a standardized $[0, 100]$ scaled score range ($s = 100 \times \frac{s_{\text{max}} - s_{\text{raw}}}{s_{\text{max}} - s_{\text{min}}}$) based on pre-holdout score extrema.

---

## 4. STAGE 8.12.4: TRAINING METHODOLOGY

Stage 8.12.4 executed the full-population model training on all eligible pre-holdout observations. This represents final model training, **not walk-forward validation**.

### 4.1 Supervised Population Construction
Supervised models were fitted on observations satisfying all of the following rules:
1. Historical training split: `split == "train"`.
2. Post-warmup period: `is_warmup == False` (observations on or after `2018-04-30`).
3. Valid target label: `hre_5d_train.notna()`.
4. Strict embargo exclusion: Observations during the 5-session boundary embargo (`2025-10-13` to `2025-10-17`, 1,444 rows, `split == "train_boundary_embargo"`) were completely excluded.
5. Strict holdout exclusion: Observations in the 250-session holdout (`2025-10-20` to `2026-10-23`, 74,334 rows, `split == "holdout"`) were completely quarantined.

### 4.2 Unsupervised Isolation Forest Population Construction
Isolation Forest was fitted on observations satisfying:
1. `split == "train"`.
2. `is_warmup == False` (observations on or after `2018-04-30`).
3. Complete target independence: Valid unlabelled trading sessions (140 suspension/delisting sessions where forward 5-day return could not be computed) were retained in the training matrix.
4. Embargo and holdout observations were strictly excluded.

---

## 5. EXACT TRAINING DATE RANGE AND ELIGIBLE ROW/SESSION COUNTS

| Training Cohort | Filter Criteria | Date Range (Start to End) | Unique Sessions | Total Eligible Rows | Positive Labels ($y=1$) | Negative Labels ($y=0$) | Positive Prevalence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Supervised Models** | `split == 'train' & ~is_warmup & hre_5d_train.notna()` | **2018-04-30 to 2025-10-10** | **1,841** | **466,346** | **84,988** | **381,358** | **18.22%** |
| **Isolation Forest** | `split == 'train' & ~is_warmup` | **2018-04-30 to 2025-10-10** | **1,841** | **466,486** | *N/A (Unsupervised)* | *N/A (Unsupervised)* | *N/A* |

*Note on population difference:* The 140-row difference between the unsupervised population (466,486) and supervised population (466,346) consists of trading sessions where stocks were quoted and generated valid historical feature vectors, but subsequently suspended trading or were delisted within the 5-day forward horizon, making the forward target undefined.

---

## 6. TRAINING DURATION FOR EACH MODEL

All models were trained in the project Python environment (`C:\Users\vighn\anaconda3\python.exe`, Python 3.13.5, scikit-learn 1.8.0, pandas 2.2.3, numpy 2.2.2). Fitting times:

| Model Configuration | Feature Count | Eligible Rows | Fit Time (Seconds) | Execution Throughput |
| :--- | :---: | :---: | :---: | :---: |
| `logistic_regression_standard` | 20 | 466,346 | **1.75 s** | 266,483 rows/sec |
| `logistic_regression_robust` | 20 | 466,346 | **8.24 s** | 56,595 rows/sec |
| `random_forest_candidate_a` | 35 | 466,346 | **48.47 s** | 9,621 rows/sec |
| `random_forest_candidate_b` | 36 | 466,346 | **38.24 s** | 12,195 rows/sec |
| `hist_gradient_boosting` | 36 | 466,346 | **10.20 s** | 45,720 rows/sec |
| `isolation_forest` | 12 | 466,486 | **8.73 s** | 53,435 rows/sec |
| **Total Pipeline Fitting Time** | — | — | **115.63 s (< 2.0 min)** | — |

---

## 7. MODEL ARTIFACT LOCATIONS

All serialized models and configuration files are persisted to separate subdirectories under `models/phase8_batch5/`:

```text
models/phase8_batch5/
├── hist_gradient_boosting/
│   ├── config.json               (Configuration dictionary)
│   ├── feature_names.json        (36 feature names & ordering)
│   └── model.joblib              (Fitted HistGradientBoosting pipeline)
├── isolation_forest/
│   ├── config.json               (Configuration dictionary)
│   ├── feature_names.json        (12 feature names & ordering)
│   └── model.joblib              (Fitted AnomalyDetector instance)
├── logistic_regression_robust/
│   ├── config.json               (Configuration dictionary)
│   ├── feature_names.json        (20 feature names & ordering)
│   └── model.joblib              (Fitted Median Imputer + RobustScaler + LR pipeline)
├── logistic_regression_standard/
│   ├── config.json               (Configuration dictionary)
│   ├── feature_names.json        (20 feature names & ordering)
│   └── model.joblib              (Fitted Median Imputer + StandardScaler + LR pipeline)
├── random_forest_candidate_a/
│   ├── config.json               (Configuration dictionary)
│   ├── feature_names.json        (35 feature names & ordering)
│   └── model.joblib              (Fitted Random Forest Candidate A pipeline)
└── random_forest_candidate_b/
    ├── config.json               (Configuration dictionary)
    ├── feature_names.json        (36 feature names & ordering)
    └── model.joblib              (Fitted Random Forest Candidate B pipeline)
```

---

## 8. TRAINING METADATA LOCATIONS

Authoritative, machine-readable training metadata is saved to `reports/phase8_batch5/training_metadata/`:

| Metadata Artifact | Content & Verification |
| :--- | :--- |
| `logistic_regression_standard_training_metadata.json` | Full training audit: dates, rows, class counts, fit time (1.75s), library versions. Verified. |
| `logistic_regression_robust_training_metadata.json` | Full training audit: dates, rows, class counts, fit time (8.24s), library versions. Verified. |
| `random_forest_candidate_a_training_metadata.json` | Full training audit: 35 features, fit time (48.47s), native NaN routing. Verified. |
| `random_forest_candidate_b_training_metadata.json` | Full training audit: 36 features (with F51), fit time (38.24s). Verified. |
| `hist_gradient_boosting_training_metadata.json` | Full training audit: Primary Champion, 36 features, fit time (10.20s). Verified. |
| `isolation_forest_training_metadata.json` | Full training audit: 12 features, target-independent, raw score bounds. Verified. |
| `overall_training_metadata_summary.json` | Comprehensive consolidation across all 6 models, quarantine status. Verified. |

---

## 9. NOTEBOOK CELLS EXECUTED AND EXPLICITLY SKIPPED

In strict compliance with Part C5 instructions:
- **Walk-forward validation cells:** Cells 0 through 11 in `notebooks/05_model_baseline.ipynb` and `notebooks/06_model_rf.ipynb`, and Cells 0 through 9 in `notebooks/07_model_hgbt.ipynb` and `notebooks/08_anomaly_detection.ipynb` were **explicitly skipped and not rerun**. All historical Batch 4 outputs, charts, and fold metric tables remain preserved intact.
- **Stage 8.12.4 verification sections:** A clearly distinguished Stage 8.12.4 verification section was appended to each notebook:
  - `notebooks/05_model_baseline.ipynb`: Cells 12 (markdown) and 13 (code). Executed in isolation.
  - `notebooks/06_model_rf.ipynb`: Cells 12 (markdown) and 13 (code). Executed in isolation.
  - `notebooks/07_model_hgbt.ipynb`: Cells 10 (markdown) and 11 (code). Executed in isolation.
  - `notebooks/08_anomaly_detection.ipynb`: Cells 10 (markdown) and 11 (code). Executed in isolation.
- Each executed cell loaded the persisted model artifacts, verified their metadata, and displayed an authoritative summary table.

---

## 10. CONFIRMATION THAT NO WALK-FORWARD VALIDATION WAS RERUN

- **Walk-forward loop execution:** 0 walk-forward loops executed.
- **Fold retraining:** 0 validation folds retrained.
- **Threshold calibration:** 0 inner calibration loops executed.
- **OOF predictions:** 0 out-of-fold predictions regenerated.
- **Batch 4 artifacts:** All 5 Batch 4 interim artifacts in `data/interim/phase8_batch4/` remain byte-for-byte identical to their pre-execution state.

---

## 11. CONFIRMATION THAT NO DATASETS WERE MODIFIED

All raw, interim, and processed datasets in the repository were treated as strictly read-only:
- No records added, modified, deleted, or re-indexed.
- No dataset schemas altered.
- No files regenerated in `data/raw/`, `data/interim/`, or `data/processed/`.

---

## 12. BEFORE-AND-AFTER SHA-256 VERIFICATION

Cryptographic checksums of primary feature datasets were verified before and after execution:

| Dataset Path | Pre-Execution SHA-256 Hash | Post-Execution SHA-256 Hash | Status |
| :--- | :--- | :--- | :---: |
| `data/processed/features.parquet` | `c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae` | `c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae` | **MATCH (PERFECT)** |
| `data/processed/features_with_targets.parquet` | `c6cebf3d93d022b11765c206b8e2ffb84e578a020e0d913d937b37d04028aa91` | `c6cebf3d93d022b11765c206b8e2ffb84e578a020e0d913d937b37d04028aa91` | **MATCH (PERFECT)** |

---

## 13. HOLDOUT QUARANTINE VERIFICATION

- **Quarantine Period:** 250 consecutive trading sessions from `2025-10-20` to `2026-10-23` ($N = 74,334$ rows, `split == "holdout"`).
- **Access Status:** **Zero access**.
- No holdout labels or feature rows entered any model fitting, transformer fitting, imputer, scaler, or metadata generation.
- No holdout scoring or metric calculation was performed.

---

## 14. BOUNDARY EMBARGO VERIFICATION

- **Embargo Window:** 5 trading sessions from `2025-10-13` to `2025-10-17` ($N = 1,444$ rows, `split == "train_boundary_embargo"`).
- **Access Status:** **Strictly excluded**.
- Target values for these sessions were engineered as `NaN` in Phase 7 (`hre_5d_train == NaN`), preventing any 5-day lookahead overlap into the holdout partition.

---

## 15. TEST RESULTS

The full repository test suite was executed post-implementation:
- **Test Command:** `python -m pytest -q`
- **Result:** **105 passed in 6.64s (100% green)**.
- **Coverage:** Unit tests verified preprocessing pipelines, native NaN routing, threshold calibration logic, label engineering, and data validator constraints.

---

## 16. ISSUES, DEVIATIONS OR LIMITATIONS

1. **Joblib Loky Physical Cores Warning on Windows:** Under Windows 11, joblib generates a minor warning regarding `wmic.exe` absence when determining physical core counts, falling back cleanly to logical cores. This does not impact parallelism or numerical output.
2. **Zero Methodological Deviations:** All approved configurations, parameters, and population constraints were respected without deviation.

---

## 17. FINAL VERDICT AND READINESS FOR BATCH 2

### Summary Checklist:
- [x] All five supervised configurations frozen in manifest.
- [x] Isolation Forest frozen as independent anomaly detector.
- [x] All six models fitted strictly on eligible pre-holdout data.
- [x] All model artifacts serialized in `models/phase8_batch5/`.
- [x] All training metadata persisted in `reports/phase8_batch5/training_metadata/`.
- [x] All four notebooks updated with Stage 8.12.4 sections and verified outputs.
- [x] Historical walk-forward results preserved with zero reruns.
- [x] Zero dataset modifications; SHA-256 hashes verified.
- [x] Final holdout (74,334 rows) completely unaccessed and quarantined.
- [x] 105/105 tests passing green.

### OVERALL VERDICT: **PASS (READY FOR BATCH 2: STAGES 8.12.5 & 8.12.6)**

The project is fully prepared to proceed to **Stage 8.12.5 (Evaluate All Five Supervised Configurations on Untouched Holdout)** and **Stage 8.12.6 (Evaluate Isolation Forest on Holdout)** upon authorization.
