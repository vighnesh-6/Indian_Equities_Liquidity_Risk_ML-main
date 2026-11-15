# Phase 8 — Batch 3: Implementation Report
## Missing-Value Treatment & Model-Specific Preprocessing Pipelines
### ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities

**Date:** November 14, 2026  
**Status:** **BATCH 3 COMPLETED**  
**Execution Context:** Strictly Implementation and Engineering Phase (No ML validation or performance evaluation experiments executed)  
**Authoritative Environment:** Python 3.12 / scikit-learn 1.8.0

---

## 1. Executive Summary

Phase 8 Batch 3 implements four independent, model-specific preprocessing pipelines and integrates them into the core modeling architecture of the Indian Equities Liquidity and Trading-Risk Intelligence system. 

Following the pre-implementation specifications established in `reports/phase8_batch3_readiness_audit.md` and `reports/phase8_batch3_final_clarification_audit.md`, this batch successfully transitions the repository from high-level architectural scaffolds into a functional, modular, and leak-free machine learning preprocessing framework without modifying any historical dataset or running out-of-scope model performance competitions.

### Key Deliverables Completed:
1. **Model-Specific Preprocessing Engine (`src/modeling/preprocessing.py`):**
   - **Logistic Regression (Baseline):** 20 features, `SimpleImputer(strategy='median')`, configurable `StandardScaler` (default) vs. `RobustScaler`.
   - **Random Forest (Ensemble):** Configurable Candidate A (35 features, baseline excluding F51) and Candidate B (36 features, experimental including F51 `relative_volume`), leveraging scikit-learn 1.8.0 native missing-value routing.
   - **HistGradientBoosting (Boosting):** 36 features (including F51 `relative_volume`), leveraging native tree split missing-value routing without artificial imputation or scaling.
   - **Isolation Forest (Unsupervised Anomaly Detector):** 12 features from `configs/model_config.yaml`, strictly target-independent (zero access to `hre_5d_train`), raw features post-median imputation (proven scale-invariant).
2. **Expanding-Window Walk-Forward Purging Safeguard (`src/modeling/trainer.py`):**
   - Implemented `generate_walk_forward_splits` with an explicit, configurable 5-trading-session purge buffer (`purge_sessions=5`).
   - Prevents forward-target lookahead contamination where the 5-session forward label evaluation window $[t+1, t+5]$ would otherwise peek across fold boundaries into validation sessions.
   - Strictly enforces trading-session calendar ordering rather than calendar days.
3. **Unsupervised Anomaly Engine (`src/modeling/anomaly.py`):**
   - Integrated `build_isolation_forest_pipeline` into `AnomalyDetector`.
   - Strictly preserves unsupervised target independence across the eligible historical training population (466,486 rows), excluding holdout and initial warmup data without filtering on forward target availability.
4. **Comprehensive Unit and Integration Test Suite (`tests/`):**
   - Added 21 dedicated unit and integration tests across `tests/test_preprocessing.py` and `tests/test_walk_forward_purging.py`.
   - Full test suite execution: **99 passed, 0 failed** in 9.42 seconds (100% green).
5. **Absolute Dataset and Artifact Preservation:**
   - SHA-256 hashes for `data/processed/features.parquet` and `data/processed/features_with_targets.parquet` verified identical to bit-level precision.
   - Zero historical research datasets, Phase 5 feature modules, or Phase 7 target modules modified.

---

## 2. Implementation Scope

Batch 3 was strictly confined to software engineering, pipeline instantiation, leakage prevention, and architectural integration.

| Architectural Component | In-Scope Actions (Completed) | Out-of-Scope Actions (Deferred to Phase 8B) |
|:---|:---|:---|
| **Preprocessing Pipelines** | Implement factory builders, imputer/scaler configurations, and feature validations in `src/modeling/preprocessing.py` | Training models to measure AUC/PR-AUC impact of scalers or imputation strategies |
| **Random Forest NaN Support** | Support Candidate A (35 feats) and Candidate B (36 feats) using scikit-learn 1.8.0 native NaN routing | Comparing out-of-fold generalization performance between Candidate A and Candidate B |
| **Walk-Forward Splits** | Implement expanding-window split generator with 5-session target purging in `src/modeling/trainer.py` | Running walk-forward model training loops to produce prediction leaderboards |
| **Leakage Controls** | Enforce trading-session purging, fit-on-train-only rules, and holdout exclusion tests | Performing hyperparameter tuning or grid search across folds |
| **Isolation Forest** | Integrate unsupervised pipeline, 12 features, and 0–100 anomaly rescaling in `src/modeling/anomaly.py` | Benchmarking anomaly detection F1 or evaluating anomaly-risk correlation |
| **Final Holdout** | Enforce programmatic quarantine excluding the 250-session holdout (`2025-10-20` to `2026-10-23`) | Evaluating any model or transformer on the final holdout partition |

---

## 3. Final Source Architecture

The modeling subpackage `src/modeling/` has been upgraded from abstract scaffolds to a modular, decoupled architecture:

```
src/modeling/
├── __init__.py           # Exports pipeline builders, feature lists, and modeling classes
├── preprocessing.py      # [NEW] Model-specific pipeline builders and feature constants
├── trainer.py            # [UPDATED] Walk-forward split generator with 5-session purging & fold execution
├── anomaly.py            # [UPDATED] Unsupervised Isolation Forest pipeline integration & score scaling
├── scorer.py             # Inference engine interface (Phase 10)
├── evaluation.py         # Evaluation metrics interface (Phase 8B/9)
└── explainability.py     # Interpretability interface (Fixed typing import)
```

### Module Responsibilities:
- `src/modeling/preprocessing.py`: Serves as the single authoritative source of truth for model-specific candidate feature lists (`LOGISTIC_REGRESSION_FEATURES`, `RANDOM_FOREST_CANDIDATE_A_FEATURES`, `RANDOM_FOREST_CANDIDATE_B_FEATURES`, `HGBT_FEATURES`, `ISOLATION_FOREST_FEATURES`) and encapsulates pipeline creation logic.
- `src/modeling/trainer.py`: Manages temporal fold creation, trading-session index calculation, purging of boundary-overlapping labels, and fold-level training orchestration.
- `src/modeling/anomaly.py`: Manages the independent unsupervised anomaly detection workflow, guaranteeing target-neutral training and scoring.

---

## 4. Model-Specific Preprocessing Decisions

A foundational requirement of Phase 8 is avoiding a "one-size-fits-all" preprocessing pipeline. Different machine learning algorithms exhibit fundamentally different mathematical sensitivities to missing values, scale, and collinearity.

| Model | Role | Feature Count | Missing-Value Treatment | Feature Scaler | Theoretical Justification |
|:---|:---|:---:|:---|:---|:---|
| **Logistic Regression** | Linear Baseline | **20** | `SimpleImputer(strategy='median')` | Configurable: `StandardScaler` (default) vs. `RobustScaler` | Linear models cannot ingest NaNs. Regularization ($L_2$) penalizes unscaled coefficients disproportionately; median imputation is robust to extreme distributional tails. |
| **Random Forest (Candidate A)** | Non-linear Ensemble (Primary) | **35** | Native NaN split routing | None (Raw features) | Scikit-learn 1.8.0 natively routes NaNs during node splitting. Tree splits are invariant to monotonic scaling; excludes F51 to eliminate benchmark missingness. |
| **Random Forest (Candidate B)** | Non-linear Ensemble (Experimental) | **36** | Native NaN split routing | None (Raw features) | Evaluates whether including F51 (`relative_volume`, 1.51% missing) adds predictive signal via native NaN routing without imputation distortion. |
| **HistGradientBoosting** | Gradient Boosting | **36** | Native NaN split routing | None (Raw features) | Binned gradient boosting natively assigns NaNs to the child node maximizing gradient split gain; completely scale-invariant. |
| **Isolation Forest** | Unsupervised Anomaly Detector | **12** | `SimpleImputer(strategy='median')` | None (Raw features) | Tree-based random axis-aligned partitioning is scale-invariant. Median imputation ensures finite, well-defined $[min, max]$ sampling intervals. Strictly target-independent. |

---

## 5. Logistic Regression Scaler Configuration

### Mathematical Formulation
Logistic Regression minimizes an $L_2$-penalized log-loss objective:
$$\min_{w, b} -\sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i) \right] + \frac{1}{2C} \sum_{j=1}^{20} w_j^2$$

Because the penalty term $\sum w_j^2$ applies an identical isotropic shrinkage penalty across all coefficients, features on larger raw numerical scales (e.g., `log_turnover` $\approx 15.0$ vs. `daily_return` $\approx 0.001$) would be penalized unequally without standardization.

### Implemented Interface
In `src/modeling/preprocessing.py`, `build_logistic_regression_pipeline(scaler_type="standard", **kwargs)` supports:
1. `scaler_type="standard"` (Primary baseline):
   $$z_j = \frac{x_j - \mu_j}{\sigma_j}$$
   Fits standard normal deviations, aligning with maximum-likelihood assumptions under Gaussian errors.
2. `scaler_type="robust"` (Alternative candidate):
   $$z_j = \frac{x_j - \text{median}(x_j)}{\text{IQR}(x_j)}$$
   Uses median centering and interquartile range scaling, preventing heavy-tailed outliers in financial return distributions from compressing feature variance.

**Strict Pre-Implementation Rule:** The pipeline builder rejects any unsupported scaler with a descriptive `ValueError`. Model comparison between `StandardScaler` and `RobustScaler` is strictly reserved for Phase 8B walk-forward validation experiments.

---

## 6. Random Forest Candidate A and B

In Phase 8 Batch 2, F51 (`relative_volume`) was flagged for having 1.51% missing values (7,025 observations) due to benchmark trading halts and new stock listings.

### Candidate Configurations:
- **RF Candidate A (35 Features):** Excludes F51 (`relative_volume`). Serves as the primary, high-integrity baseline operating on 35 fully populated or low-missingness features.
- **RF Candidate B (36 Features):** Includes F51 (`relative_volume`). Serves as an experimental candidate to test whether scikit-learn 1.8.0's native missingness routing recovers valuable signal from volume decoupling.

### Verification of scikit-learn 1.8.0 Runtime Capabilities:
Both candidates utilize scikit-learn 1.8.0's native missing-value support:
- `RandomForestClassifier` handles NaNs directly during tree building by evaluating split criteria on present values and sending missing values to the child branch that minimizes node impurity.
- Unit test `test_rf_native_nan_handling` in `tests/test_preprocessing.py` verified that synthetic matrices with missing values across multiple features train and produce valid probabilities without an imputer step.

---

## 7. HistGradientBoosting NaN Handling

`HistGradientBoostingClassifier` natively bins continuous features into integer bins (up to 256 bins), reserving bin index 255 specifically for missing values (`NaN`). During tree traversal:
1. All missing values are grouped into their own dedicated bin.
2. For every potential threshold split, the algorithm computes gradient gains when routing the missing bin to the left child versus the right child.
3. The routing yielding the greatest split criterion gain is learned and saved into the tree structure.

Consequently, `build_hgbt_pipeline()` includes **no imputer** and **no scaler**, feeding all 36 features directly to the booster. Unit test `test_hgbt_native_nan_handling` validated this behavior on multi-column missing data.

---

## 8. Isolation Forest Preprocessing

Isolation Forest partitions data using random, axis-aligned orthogonal cuts:
1. A feature $j \in \{1, \dots, 12\}$ is chosen uniformly at random.
2. A split value $v \in [\min(x_j), \max(x_j)]$ is selected uniformly at random.
3. The space is partitioned into $x_j < v$ and $x_j \ge v$ until points are isolated.

### Preprocessing Architecture:
- **Imputation:** `SimpleImputer(strategy="median")` is applied. While scikit-learn's Isolation Forest supports missing values, imputing medians ensures well-conditioned $[\min, \max]$ sampling bounds.
- **No Scaler:** Random orthogonal cuts are strictly invariant to monotonic linear and affine transformations. Empirical tests in the clarification audit demonstrated bit-level identical anomaly scores ($r = 1.000000$) between raw and scaled inputs.
- **Anomaly Score Rescaling:** In `AnomalyDetector.score()`, raw decision scores (where lower numbers represent higher anomaly) are rescaled to a calibrated 0–100 scale:
  $$\text{anomaly\_score\_scaled} = 100 \times \frac{\max(s_{\text{train}}) - s}{\max(s_{\text{train}}) - \min(s_{\text{train}})}$$
  clipped to $[0.0, 100.0]$.

---

## 9. Five-Session Purging Implementation

### The Boundary Overlap Problem
The forward target `hre_5d_train` evaluates the maximum Liquidity Risk Score over the subsequent 5 trading sessions:
$$\text{fwd\_max\_lrs\_5d}(i, t) = \max_{k \in [1, 5]} \text{LRS}(i, t+k)$$
If a training fold ends on session $T_{\text{train}}$ and validation begins on session $T_{\text{val}} = T_{\text{train}} + 1$:
- The label for $T_{\text{train}}$ depends on $[T_{\text{train}}+1, T_{\text{train}}+5]$, which are **Sessions 1, 2, 3, 4, 5 of validation**.
- The label for $T_{\text{train}}-1$ peeks into **Sessions 1, 2, 3, 4 of validation**.
- The label for $T_{\text{train}}-4$ peeks into **Session 1 of validation**.
- Only observations at $t \le T_{\text{train}}-5$ have target windows $[t+1, t+5]$ ending on or before $T_{\text{train}}$, remaining strictly self-contained.

```
Training Window:  ... [T-5] [T-4] [T-3] [T-2] [T-1] [ T ] | [T+1] [T+2] [T+3] [T+4] [T+5] ... (Validation)
Target at T:                   └────────────────────────────► [  1  ] [  2  ] [  3  ] [  4  ] [  5  ] (LEAKAGE!)
Target at T-1:           └────────────────────────────► [  1  ] [  2  ] [  3  ] [  4  ]         (LEAKAGE!)
Target at T-2:     └────────────────────────────► [  1  ] [  2  ] [  3  ]                 (LEAKAGE!)
Target at T-3: └────────────────────────────► [  1  ] [  2  ]                         (LEAKAGE!)
Target at T-4: └──────────────────────► [  1  ]                                 (LEAKAGE!)
Target at T-5: └────────────────► [ T ] (CLEAN — ZERO OVERLAP)
```

### Technical Solution
In `ModelTrainer.generate_walk_forward_splits()`, parameter `purge_sessions: int = 5` excludes the final 5 trading sessions immediately preceding every validation window:
```python
val_start_idx = val_starts[i]
val_end_idx = min(val_start_idx + self.test_window_days - 1, n_dates - 1)
train_end_idx = val_start_idx - p - 1  # Exactly p sessions purged

train_dates = set(eligible_dates[0 : train_end_idx + 1])
val_dates = set(eligible_dates[val_start_idx : val_end_idx + 1])
```
Unit test `test_five_session_purging_prevents_target_overlap` verified that for all generated folds, $\text{last\_train\_session} + 5 < \text{first\_val\_session}$, mathematically guaranteeing zero label lookahead into validation.

---

## 10. Walk-Forward Leakage Safeguards

Batch 3 enforces four programmatic safeguards against data leakage:

1. **Trading-Session Ordering Over Calendar Days:**
   Purging operates on ordered unique trading sessions (`date_series.unique()`), not calendar day intervals. Holiday periods and weekend gaps are naturally handled without under-purging or over-purging (verified in `test_trading_session_counts_used_rather_than_calendar_days`).
2. **Identical Chronological Fold Boundaries:**
   Fold split boundaries depend exclusively on temporal calendar ordering, ensuring Logistic Regression, Random Forest, and HistGradientBoosting receive mathematically identical training and validation date partitions (verified in `test_identical_fold_boundaries_across_models`).
3. **Strict Information Isolation in Preprocessing:**
   Transformers (`SimpleImputer`, `StandardScaler`, `RobustScaler`) are fitted strictly on `df.loc[train_idx]`. Validation observations are transformed without fitting. In `test_no_preprocessing_fit_on_validation_observations`, injecting massive outlier values ($+10^6$) into validation features caused zero shift in the fitted scaler's mean or standard deviation.
4. **Final Holdout Quarantine:**
   Observations where `split == "holdout"` (the 250 reserved trading sessions from `2025-10-20` to `2026-10-23`) and `split == "train_boundary_embargo"` are programmatically masked from both `train_idx` and `val_idx` across all walk-forward folds (verified in `test_holdout_and_embargo_excluded_from_training_populations`).

---

## 11. Unit and Integration Test Results

A test suite containing 21 new tests was designed and executed.

### Preprocessing Suite (`tests/test_preprocessing.py`)
| Test ID | Test Function Name | Architectural Requirement Verified | Status |
|:---:|:---|:---|:---:|
| 1 | `test_lr_builder_fresh_instances` | LR builder returns independent Pipeline instances with fresh steps | **PASSED** |
| 2 | `test_lr_pipeline_uses_median_imputer` | LR pipeline integrates `SimpleImputer(strategy='median')` | **PASSED** |
| 3 | `test_lr_supports_standard_and_robust_scalers` | LR pipeline supports both `StandardScaler` and `RobustScaler` | **PASSED** |
| 4 | `test_lr_builder_rejects_invalid_scaler` | LR builder raises `ValueError` on invalid scaler names | **PASSED** |
| 5 | `test_lr_feature_count_and_selection` | LR features exactly 20; algebraic triplets excluded; fits on NaNs | **PASSED** |
| 6 | `test_rf_candidate_a_features` | RF Candidate A features exactly 35; excludes F51; no imputer/scaler | **PASSED** |
| 7 | `test_rf_candidate_b_features` | RF Candidate B features exactly 36; includes F51; no imputer/scaler | **PASSED** |
| 8 | `test_rf_candidate_b_includes_f51` | Candidate B minus Candidate A is exactly `{'relative_volume'}` | **PASSED** |
| 9 | `test_rf_native_nan_handling` | RF natively fits and predicts probabilities on missing values | **PASSED** |
| 10 | `test_hgbt_native_nan_handling` | HGBT natively fits and predicts on 36 features with NaNs | **PASSED** |
| 11 | `test_isolation_forest_feature_count` | Isolation Forest features exactly 12 candidate features | **PASSED** |
| 12 | `test_isolation_forest_target_independence` | Isolation Forest fits with zero target labels; outputs scaled scores | **PASSED** |
| - | `test_validate_features_present` | Feature validation helper catches missing columns | **PASSED** |

### Walk-Forward and Preservation Suite (`tests/test_walk_forward_purging.py`)
| Test ID | Test Function Name | Architectural Requirement Verified | Status |
|:---:|:---|:---|:---:|
| 13 | `test_five_session_purging_prevents_target_overlap` | 5-session purge prevents target window overlap across folds | **PASSED** |
| 14 | `test_trading_session_counts_used_rather_than_calendar_days` | Purge counts trading sessions, ignoring calendar gaps/holidays | **PASSED** |
| 15 | `test_identical_fold_boundaries_across_models` | Date boundaries identical across LR, RF, and HGBT | **PASSED** |
| 16 | `test_no_preprocessing_fit_on_validation_observations` | Transformers fit strictly on train; validation does not alter statistics | **PASSED** |
| 17 | `test_fresh_preprocessing_instances_per_fold` | New pipeline instances created per fold without state leakage | **PASSED** |
| 18 | `test_holdout_and_embargo_excluded_from_training_populations` | Final 250-session holdout and embargo excluded from all folds | **PASSED** |
| 19 | `test_dataset_hashes_unchanged` | Parquet dataset SHA-256 hashes match bit-level specification | **PASSED** |
| 20 | `test_phase5_and_phase7_artifacts_untouched` | All frozen Phase 5 and Phase 7 artifacts exist and untouched | **PASSED** |

### Full Test Suite Execution Summary
```
============================== test session starts ==============================
platform win32 -- Python 3.12.8, pytest-8.3.4, pluggy-1.5.0
rootdir: C:\Users\vighn\Downloads\Indian_Equities_Liquidity_Risk_ML-main\Indian_Equities_Liquidity_Risk_ML-main
configfile: pyproject.toml
collected 99 items

tests/test_cli.py .................                                        [ 17%]
tests/test_data_acquisition.py .......                                     [ 24%]
tests/test_eda.py ...........                                              [ 35%]
tests/test_features.py ................                                    [ 51%]
tests/test_labels.py ....................                                  [ 71%]
tests/test_preprocessing.py .............                                  [ 84%]
tests/test_universe.py .......                                             [ 91%]
tests/test_walk_forward_purging.py ........                                [100%]

============================== 99 passed in 9.42s ===============================
```

---

## 12. Dataset Hash Verification

All historical datasets remain untouched. Cryptographic SHA-256 checksums were verified before and after implementation:

| Dataset File | Expected SHA-256 Hash | Post-Implementation SHA-256 Hash | Integrity Status |
|:---|:---|:---|:---:|
| `data/processed/features.parquet` | `c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae` | `c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae` | **MATCH** |
| `data/processed/features_with_targets.parquet` | `c6cebf3d93d022b11765c206b8e2ffb84e578a020e0d913d937b37d04028aa91` | `c6cebf3d93d022b11765c206b8e2ffb84e578a020e0d913d937b37d04028aa91` | **MATCH** |

---

## 13. Files Created and Modified

### Files Created:
1. `src/modeling/preprocessing.py` (338 lines) — Model-specific preprocessing builders and candidate feature constants.
2. `tests/test_preprocessing.py` (221 lines) — Unit tests for pipeline builders, NaN handling, and feature dimensions.
3. `tests/test_walk_forward_purging.py` (248 lines) — Integration tests for walk-forward splits, 5-session purging, and holdout exclusion.
4. `reports/phase8_batch3_implementation_report.md` (This document) — Comprehensive implementation and verification report.

### Files Modified:
1. `src/modeling/trainer.py` — Implemented `generate_walk_forward_splits` with 5-session purging and fold execution methods.
2. `src/modeling/anomaly.py` — Integrated `build_isolation_forest_pipeline`, target-neutral fitting, and 0–100 score rescaling.
3. `src/modeling/__init__.py` — Exported preprocessing builders and feature constants.
4. `src/modeling/explainability.py` — Fixed missing `Any` typing import.

---

## 14. Limitations and Deferred Phase 8B Work

To ensure strict compliance with project boundaries, all predictive modeling experiments were halted at the implementation gate:

1. **No Performance Evaluations:** No models were fitted on the full training dataset to measure AUC, PR-AUC, F1, precision, or recall.
2. **No Model Comparison:** No competition was conducted between Logistic Regression, Random Forest, and HistGradientBoosting.
3. **No Scaler Selection:** `StandardScaler` vs. `RobustScaler` for Logistic Regression remains unselected; both are implemented and configurable for Phase 8B empirical evaluation.
4. **No RF Candidate Selection:** RF Candidate A vs. Candidate B remains unselected; both pipelines are implemented and ready for walk-forward testing.
5. **No Final Holdout Evaluation:** The final 250 trading sessions (`2025-10-20` to `2026-10-23`) remain strictly quarantined and unread by any modeling routine.

---

## 15. Final Batch 3 Completion Verdict

### **VERDICT: BATCH 3 COMPLETED**

The Phase 8 Batch 3 missing-value treatment and model-specific preprocessing pipelines are fully implemented, verified, and integrated. All 20 required verification criteria and all 99 automated tests pass with 100% compliance. The modeling codebase is completely prepared for Phase 8B walk-forward validation experiments.

