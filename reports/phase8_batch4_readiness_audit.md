# Phase 8 — Batch 4: Pre-Implementation Readiness & Methodology Audit
## ML Modelling, Walk-Forward Validation, Evaluation & Model Comparison
### ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities

**Date:** November 14, 2026  
**Audit Authorization:** Strictly READ-ONLY Pre-Implementation Gate  
**Execution Context:** No models trained, no performance evaluated, no holdout accessed, no existing code modified  
**Authoritative Environment:** Python 3.12 / scikit-learn 1.8.0 / Pandas 2.2.3  
**Final Verdict:** **READY WITH CORRECTIONS**

---

## 1. Executive Summary

This readiness and methodology audit conducts an exhaustive, read-only evaluation of the machine learning modeling architecture, model-specific preprocessing pipelines, feature configurations, expanding-window walk-forward validation engine, leakage safeguards, and evaluation framework for **Phase 8, Batch 4 (Stages 8.9–8.11)** of the project: *ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities*.

Following the completion of Batch 1 (Feature Inventory & Distributions), Batch 2 (Feature Relevance & Selection), and Batch 3 (Preprocessing Pipelines & 5-Session Purging), the repository stands at the threshold of actual ML model training, walk-forward cross-validation, and multi-model benchmarking.

### Core Audit Findings:
1. **Walk-Forward Validation Engine & 5-Session Purge (STATUS: PASS):**
   The chronological expanding-window walk-forward generator in `src/modeling/trainer.py` generates exactly 10 non-overlapping validation windows of 60 trading days each (tiling 600 trading sessions from `2023-05-10` to `2025-10-10`, comprising 163,357 out-of-fold validation observations). The 5-trading-session purge buffer is strictly enforced using trading-session calendar ordering, mathematically guaranteeing zero overlap between the forward target window $[t+1, t+5]$ and validation sessions ($\text{last\_train\_session} + 5 < \text{first\_val\_session}$).
2. **Model-Specific Preprocessing Pipelines (STATUS: PASS / WARNING):**
   All four model configurations (Logistic Regression, Random Forest A/B, HistGradientBoosting, Isolation Forest) are implemented as independent scikit-learn `Pipeline` objects in `src/modeling/preprocessing.py`. They correctly implement median imputation and dual scaling (LR), native NaN routing without scalers (RF-A, RF-B, HGBT), and scale-invariant unsupervised anomaly detection (IF). **Defect Identified (WARNING):** In `get_model_features()`, passing `model_name="random_forest_candidate_b"` raises a `ValueError` because the string matcher only checks `("random_forest", "rf")`. This must be corrected during implementation.
3. **Feature Selection Leakage Assessment (STATUS: PASS):**
   The candidate feature sets (20 for LR, 35 for RF-A, 36 for RF-B/HGBT, 12 for IF) represent **pre-approved candidate model architectures** (domain-driven structural pruning). An empirical audit conducted on Fold 0 training data alone (`2018-04-30` to `2023-05-02`, $N = 301,744$), which strictly predates all 10 validation periods, verified that the top univariate predictors (`amihud_ma20` $\text{AUC} = 0.7801$, `cs_spread_ma10` $\text{AUC} = 0.7648$, `log_turnover` $\text{AUC} = 0.7344$) and their ranking are identical to the pooled Batch 2 analysis, proving zero post-2023 lookahead distortion.
4. **Final Holdout Isolation (STATUS: PASS):**
   The 250 reserved trading sessions (`2025-10-20` to `2026-10-23`, 74,334 rows) and the 5-session boundary embargo (`2025-10-13` to `2025-10-17`, 1,444 rows) are strictly quarantined by programmatic masks in `trainer.py` and `anomaly.py`. Zero holdout rows enter training or validation folds.
5. **Evaluation Architecture Gaps (STATUS: WARNING):**
   `src/modeling/evaluation.py` (`ModelEvaluator`) and `ModelTrainer.train_and_evaluate_models` currently raise `NotImplementedError`. Concrete implementation of metric computation (PR-AUC, ROC-AUC, Brier score, Log Loss, threshold-tuned F1, and cap-stratified evaluation) must be completed as the first engineering step of Batch 4.
6. **Configuration Synchronization (STATUS: WARNING):**
   `configs/model_config.yaml` still specifies `embargo_gap_days: 1` under `validation:`, whereas `trainer.py` implements `purge_sessions: 5`. The YAML configuration must be updated to explicitly include `purge_sessions: 5`.

---

## 2. Files and Components Inspected

| Component | File Path | Type | Inspection Status | Integrity / Hash |
|:---|:---|:---|:---:|:---|
| **Features Dataset** | `data/processed/features.parquet` | Processed Data | Read-Only Verified | `c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae` |
| **Targets Dataset** | `data/processed/features_with_targets.parquet` | Labeled Data | Read-Only Verified | `c6cebf3d93d022b11765c206b8e2ffb84e578a020e0d913d937b37d04028aa91` |
| **Model Config** | `configs/model_config.yaml` | YAML Config | Inspected | 79 lines |
| **Feature Config** | `configs/feature_config.yaml` | YAML Config | Inspected | 244 lines |
| **Preprocessing Engine** | `src/modeling/preprocessing.py` | Python Source | Inspected | 381 lines |
| **Training Engine** | `src/modeling/trainer.py` | Python Source | Inspected | 306 lines |
| **Anomaly Module** | `src/modeling/anomaly.py` | Python Source | Inspected | 174 lines |
| **Evaluation Module** | `src/modeling/evaluation.py` | Python Source | Inspected (Scaffold) | 54 lines (`NotImplementedError`) |
| **Explainability Module**| `src/modeling/explainability.py` | Python Source | Inspected | 52 lines |
| **Inference Module** | `src/modeling/scorer.py` | Python Source | Inspected (Scaffold) | 44 lines (`NotImplementedError`) |
| **Modeling Init** | `src/modeling/__init__.py` | Python Source | Inspected | 35 lines |
| **Preprocessing Tests** | `tests/test_preprocessing.py` | Pytest Module | Executed & Verified | 13 passed |
| **Purging Tests** | `tests/test_walk_forward_purging.py` | Pytest Module | Executed & Verified | 8 passed |
| **Batch 2 Report** | `reports/phase8_batch2_feature_selection_report.md` | Markdown Report | Reference Verified | Complete |
| **Batch 3 Report** | `reports/phase8_batch3_implementation_report.md` | Markdown Report | Reference Verified | Complete |

---

## 3. Existing Architecture Assessment

```
src/modeling/
├── __init__.py           [PASS] Exports all public classes, functions, and feature lists.
├── preprocessing.py      [PASS] Independent builders for LR, RF-A, RF-B, HGBT, IF.
├── trainer.py            [PASS] Chronological walk-forward generator with 5-session purge.
│                                [WARNING] train_and_evaluate_models raises NotImplementedError.
├── anomaly.py            [PASS] Unsupervised IF pipeline, 12 features, target-independent.
├── evaluation.py         [WARNING] Scaffold only; compute_classification_metrics is NotImplementedError.
├── scorer.py             [PASS] Phase 10 inference scaffold; untouched.
└── explainability.py     [PASS] SHAP/Permutation importance scaffold; typing verified.
```

### Architectural Integrity Verification:
- **Coupling:** Low. Preprocessing pipelines are completely decoupled from model training orchestration. Each pipeline can be instantiated, fitted, and serialized independently.
- **Independence:** High. Each algorithm receives a model-specific feature subset and its own dedicated preprocessing steps.
- **Reproducibility:** High. Random seeds (`random_state=42`) are explicitly parameterized across all estimators and splitters.
- **Runtime Environment:** Verified against scikit-learn 1.8.0. Native NaN routing in `RandomForestClassifier` and `HistGradientBoostingClassifier` verified operational.

---

## 4. Preprocessing Pipeline Verification

### A. Logistic Regression (Linear Baseline) — STATUS: PASS
- **Input Feature Set:** Exactly 20 approved features from `LOGISTIC_REGRESSION_FEATURES`.
  - Collinearity eliminated: Maximum VIF in selected subset is 4.89 (`max_drawdown_20d`), condition number 14.8.
  - Algebraic triplet dropped: `return_20d` and `excess_return_20d` excluded; `nifty_return_20d` retained.
- **Missing-Value Handling:** `SimpleImputer(strategy='median')`.
  - In financial time-series, median imputation is robust to heavy-tailed return spikes.
  - Imputer is fit strictly on training fold observations; validation missing values are replaced with training medians.
- **Feature Scaling:** Dual configurable builders supported via `build_logistic_regression_pipeline(scaler_type=...)`:
  - `scaler_type="standard"`: `StandardScaler()`. Mean centering and unit variance scaling.
  - `scaler_type="robust"`: `RobustScaler()`. Median centering and interquartile range (IQR) scaling.
  - Any invalid scaler name raises an explicit `ValueError`.
- **Estimator Configuration:** `LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42)`.
- **Leakage Safeguard:** In `test_no_preprocessing_fit_on_validation_observations`, injecting $+10^6$ into validation features resulted in zero change to fitted scaler means.

### B. Random Forest Candidate A (Primary Ensemble Baseline) — STATUS: PASS
- **Input Feature Set:** Exactly 35 approved features from `RANDOM_FOREST_CANDIDATE_A_FEATURES`.
  - Excludes F51 `relative_volume` (eliminating the 1.51% missingness from benchmark halts).
  - Prunes exact collinear duplicates ($r \ge 0.95$) and unscaled rupee metrics (`true_range`, `atr_20d`).
- **Missing-Value Handling:** Native NaN handling in scikit-learn 1.8.0.
  - No `SimpleImputer` step in the pipeline.
  - Verified on synthetic matrices: tree split growers evaluate impurity on present observations and route missing observations to the child node minimizing impurity.
- **Scaling:** None. Tree-based decision boundaries are invariant to monotonic scaling.
- **Estimator Configuration:** `RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=50, max_features='sqrt', class_weight='balanced_subsample', n_jobs=-1, random_state=42)`.

### C. Random Forest Candidate B (Experimental Candidate) — STATUS: PASS
- **Input Feature Set:** Exactly 36 approved features (`RANDOM_FOREST_CANDIDATE_B_FEATURES`).
  - Includes F51 `relative_volume` inserted at index 31.
  - Candidate B minus Candidate A is verified as exactly `{'relative_volume'}`.
- **Missing-Value Handling:** Identical native NaN routing as Candidate A.
- **Purpose in Batch 4:** Isolates the exact marginal predictive contribution of benchmark-relative volume decoupling.

### D. HistGradientBoosting (Gradient Boosting) — STATUS: PASS
- **Input Feature Set:** Exactly 36 approved features (`HGBT_FEATURES`), identical to RF Candidate B.
- **Missing-Value Handling:** Native histogram binning.
  - Missing values are assigned to a dedicated missing bin (bin 255).
  - Split optimizer learns direction (left or right) for missing values that maximizes gradient gain.
- **Scaling:** None. Gradient boosting on binned features is strictly scale-invariant.
- **Estimator Configuration:** `HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=0.05, l2_regularization=0.1, class_weight='balanced', random_state=42)`.

### E. Isolation Forest (Unsupervised Anomaly Detector) — STATUS: PASS
- **Input Feature Set:** Exactly 12 features from `configs/model_config.yaml` (`ISOLATION_FOREST_FEATURES`).
- **Target Independence:** Strictly never accesses or filters by `hre_5d_train`.
- **Training Population:** Post-warmup training rows (`split == "train"` & `~is_warmup`, $N = 466,486$). The 140 observations missing target labels due to trading halts are legitimately included.
- **Missing-Value Treatment:** `SimpleImputer(strategy='median')` ensures finite $[\min, \max]$ sampling bounds for random cuts.
- **Scaling:** None. Scale-invariance verified empirically ($r = 1.000000$ between raw and scaled scores).
- **Score Calibration:** Rescales negative decision scores to $[0, 100]$ where 100 = extreme anomaly.

---

## 5. Model Configuration Verification

| Model Identifier | Configuration Source | Feature Count | Estimator Class | Hyperparameter Alignment | Status |
|:---|:---|:---:|:---|:---|:---:|
| `logistic_regression` (std) | `configs/model_config.yaml` | 20 | `LogisticRegression` | $C=1.0$, solver=lbfgs, max_iter=1000, class_weight=balanced | **PASS** |
| `logistic_regression` (rob) | `src/modeling/preprocessing.py` | 20 | `LogisticRegression` | Identical, with `RobustScaler` | **PASS** |
| `random_forest` (Cand. A) | `configs/model_config.yaml` | 35 | `RandomForestClassifier` | $N=150$, depth=8, min_leaf=50, class_weight=balanced_subsample | **PASS** |
| `random_forest` (Cand. B) | Batch 2/3 Specification | 36 | `RandomForestClassifier` | Identical, with F51 included | **PASS** |
| `hist_gradient_boosting` | `configs/model_config.yaml` | 36 | `HistGradientBoostingClassifier`| iter=200, leaves=31, lr=0.05, l2=0.1, class_weight=balanced | **PASS** |
| `isolation_forest` | `configs/model_config.yaml` | 12 | `IsolationForest` | $N=100$, contamination=0.05, max_samples=auto | **PASS** |

### Configuration Gap Identified:
- In `configs/model_config.yaml` (lines 4–11):
  ```yaml
  validation:
    strategy: "expanding_window_walk_forward"
    n_splits: 10
    min_train_days: 490
    test_window_days: 60
    embargo_gap_days: 1 # <--- GAP: Insufficient for 5-day forward target
    holdout_fraction_or_days: 250
    random_state: 42
  ```
  `model_config.yaml` specifies `embargo_gap_days: 1`. In `ModelTrainer`, `purge_sessions: 5` is implemented as an explicit parameter.
  **Recommendation:** Update `configs/model_config.yaml` to specify `purge_sessions: 5` under `validation:` so that configuration loaders instantiate `ModelTrainer` with matching parameters.

---

## 6. Walk-Forward Validation Audit

An independent, date-level audit was executed on `data/processed/features_with_targets.parquet`.

### Exact 10-Fold Chronological Schedule:
The post-warmup training partition comprises 1,841 unique trading dates (`2018-04-30` to `2025-10-10`). With `min_train_days=490`, `test_window_days=60`, and `purge_sessions=5`, the expanding-window split generator produces exactly 10 contiguous validation folds:

| Fold | Training Date Range | Train Days | Train Rows | Train Pos. | Train Prev. | Purged Dates (Count) | Validation Date Range | Val Days | Val Rows | Val Pos. | Val Prev. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | `2018-04-30` to `2023-05-02` | 1,236 | 301,744 | 54,853 | 18.18% | `2023-05-03` to `2023-05-09` (5 d) | `2023-05-10` to `2023-08-02` | 60 | 15,780 | 2,937 | 18.61% |
| **1** | `2018-04-30` to `2023-07-26` | 1,296 | 317,524 | 57,831 | 18.21% | `2023-07-27` to `2023-08-02` (5 d) | `2023-08-03` to `2023-10-31` | 60 | 15,821 | 2,959 | 18.71% |
| **2** | `2018-04-30` to `2023-10-23` | 1,356 | 333,340 | 60,770 | 18.23% | `2023-10-25` to `2023-10-31` (5 d) | `2023-11-01` to `2024-01-30` | 60 | 15,939 | 3,100 | 19.51% |
| **3** | `2018-04-30` to `2024-01-19` | 1,416 | 349,216 | 63,908 | 18.30% | `2024-01-23` to `2024-01-30` (5 d) | `2024-01-31` to `2024-04-30` | 60 | 16,157 | 2,887 | 17.89% |
| **4** | `2018-04-30` to `2024-04-23` | 1,476 | 365,334 | 66,774 | 18.28% | `2024-04-24` to `2024-04-30` (5 d) | `2024-05-02` to `2024-07-29` | 60 | 16,260 | 2,773 | 17.05% |
| **5** | `2018-04-30` to `2024-07-22` | 1,536 | 381,591 | 69,550 | 18.23% | `2024-07-23` to `2024-07-29` (5 d) | `2024-07-30` to `2024-10-23` | 60 | 16,358 | 3,167 | 19.36% |
| **6** | `2018-04-30` to `2024-10-16` | 1,596 | 397,939 | 72,678 | 18.26% | `2024-10-17` to `2024-10-23` (5 d) | `2024-10-24` to `2025-01-20` | 60 | 16,435 | 2,919 | 17.76% |
| **7** | `2018-04-30` to `2025-01-13` | 1,656 | 414,359 | 75,626 | 18.25% | `2025-01-14` to `2025-01-20` (5 d) | `2025-01-21` to `2025-04-21` | 60 | 16,685 | 2,989 | 17.91% |
| **8** | `2018-04-30` to `2025-04-09` | 1,716 | 431,021 | 78,609 | 18.24% | `2025-04-11` to `2025-04-21` (5 d) | `2025-04-22` to `2025-07-15` | 60 | 16,911 | 3,003 | 17.76% |
| **9** | `2018-04-30` to `2025-07-08` | 1,776 | 447,920 | 81,607 | 18.22% | `2025-07-09` to `2025-07-15` (5 d) | `2025-07-16` to `2025-10-10` | 60 | 17,011 | 3,136 | 18.44% |
| **Total / Pooled** | — | — | — | — | — | — | **`2023-05-10` to `2025-10-10`** | **600** | **163,357** | **29,870** | **18.28%** |

### Key Validation Observations:
1. **Expanding Window:** Training data expands monotonically from 301,744 rows (Fold 0) to 447,920 rows (Fold 9), ensuring the models learn from an ever-increasing history while testing on unseen forward quarters.
2. **Prevalence Stability:** Positive class prevalence in training folds is remarkably consistent at $\sim 18.22\% \pm 0.04\%$. Validation prevalence varies between $17.05\%$ and $19.51\%$, mirroring real-world macro market liquidity cycles without extreme distribution collapse.
3. **Contiguous Coverage:** The 10 validation periods tile exactly 600 trading sessions without a single missing day or overlap.

---

## 7. Five-Session Purge Verification

### Mathematical Overlap Proof:
The target variable is defined as:
$$\text{hre\_5d\_train}(i, t) = \mathbb{I}\left( \max_{k \in [1, 5]} \text{LRS}(i, t+k) \ge \theta_{90, s} \right)$$
For any training observation on trading day $T_k$, the forward window covers sessions $\{T_{k+1}, T_{k+2}, T_{k+3}, T_{k+4}, T_{k+5}\}$.
Let the validation window start at session index $V_{\text{start}}$.
- If $k = V_{\text{start}} - 1$, its window is $\{V_{\text{start}}, \dots, V_{\text{start}}+4\}$ (5 sessions overlap).
- If $k = V_{\text{start}} - 2$, its window is $\{V_{\text{start}}-1, \dots, V_{\text{start}}+3\}$ (4 sessions overlap).
- If $k = V_{\text{start}} - 3$, its window is $\{V_{\text{start}}-2, \dots, V_{\text{start}}+2\}$ (3 sessions overlap).
- If $k = V_{\text{start}} - 4$, its window is $\{V_{\text{start}}-3, \dots, V_{\text{start}}+1\}$ (2 sessions overlap).
- If $k = V_{\text{start}} - 5$, its window is $\{V_{\text{start}}-4, \dots, V_{\text{start}}\}$ (1 session overlap at $V_{\text{start}}$).
- For $k \le V_{\text{start}} - 6$, the final forward session is $k+5 \le V_{\text{start}} - 1 < V_{\text{start}}$. **Zero overlap.**

### Verification Across All 10 Folds:
In every fold generated by `ModelTrainer`:
$$T_{\text{train\_end}} = V_{\text{start}} - 6$$
The gap between the last training date and the first validation date contains exactly 5 trading sessions:
$$\{ V_{\text{start}}-5, V_{\text{start}}-4, V_{\text{start}}-3, V_{\text{start}}-2, V_{\text{start}}-1 \}$$
These 5 dates are completely omitted from both the training fold and the validation fold.
- **Calendar-Day Independence:** In Fold 0, the purged sessions are `2023-05-03` (Wed), `2023-05-04` (Thu), `2023-05-05` (Fri), `2023-05-08` (Mon), and `2023-05-09` (Tue). The weekend (`May 6–7`) is skipped naturally.
- **Double-Embargo Verification:** In Fold 9, validation ends on `2025-10-10`. The Phase 7 boundary embargo (`2025-10-13` to `2025-10-17`) occurs *after* Fold 9 validation and *before* holdout (`2025-10-20`). The Phase 7 embargo is NOT double-applied to walk-forward folds.

---

## 8. Feature Selection Leakage Assessment

### The Methodological Dilemma:
In Batch 2, univariate ROC-AUC, correlation matrices, and VIFs were computed across the pooled eligible training partition ($N = 466,346$, dates `2018-04-30` to `2025-10-10`). The user prompt raises the critical concern: *Does using feature sets selected from the pooled 2018–2025 training partition introduce lookahead leakage into earlier walk-forward folds (e.g. Folds 0–4)?*

### Distinction Between Feature Space Types:
1. **Category A — Pre-Approved Candidate Feature Sets (Current Architecture):**
   - The selected feature sets (20 for LR, 35 for RF-A, 36 for RF-B/HGBT, 12 for IF) were pruned primarily based on:
     - Mathematical identities: Dropping algebraic triplets ($\text{excess\_return\_20d} \equiv \text{return\_20d} - \text{nifty\_return\_20d}$).
     - Sparsity: Dropping `zero_vol_flag` (99.92% zeros).
     - Collinearity: Enforcing $\text{VIF} < 5.0$ and dropping exact twins ($r \ge 0.95$).
     - Scale invariance: Dropping unscaled rupee metrics (`true_range`, `atr_20d`).
   - None of these structural exclusions depend on forward target relationships.
2. **Category B — Data-Dependent Dynamic Feature Selection:**
   - Performing automated feature selection (e.g., forward selection, recursive feature elimination, or LASSO coefficient thresholding) dynamically within the walk-forward loop.

### Empirical Audit: Initial-Window Feature Relevance (Zero-Leakage Benchmark)
To test whether the top predictive features were an artifact of post-2023 data, we evaluated the univariate ROC-AUC of candidate features strictly on the **Fold 0 Training Window** (`2018-04-30` to `2023-05-02`, $N = 301,744$), which strictly predates all 10 validation periods:

| Feature Name | Fold 0 Train AUC (2018–2023) | Pooled Batch 2 AUC (2018–2025) | Ranking Stability | Lookahead Distortion? |
|:---|:---:|:---:|:---:|:---:|
| `amihud_ma20` | **0.7801** | 0.7619 | Rank 1 $\rightarrow$ Rank 1 | **NO (Signal already dominant in 2018–2023)** |
| `cs_spread_ma10` | **0.7648** | 0.7561 | Rank 2 $\rightarrow$ Rank 2 | **NO** |
| `log_turnover` | **0.7344** | 0.7125 | Rank 3 $\rightarrow$ Rank 3 | **NO** |
| `downside_vol_20d`| **0.6978** | 0.6942 | Rank 4 $\rightarrow$ Rank 4 | **NO** |
| `daily_range` | **0.6550** | 0.6484 | Rank 5 $\rightarrow$ Rank 5 | **NO** |
| `vol_5d` | **0.6427** | 0.6402 | Rank 6 $\rightarrow$ Rank 6 | **NO** |
| `rolling_corr_60d`| **0.6235** | 0.6271 | Rank 7 $\rightarrow$ Rank 7 | **NO** |
| `max_drawdown_20d`| **0.6041** | 0.6020 | Rank 8 $\rightarrow$ Rank 8 | **NO** |
| `turnover_ratio` | **0.5738** | 0.5657 | Rank 9 $\rightarrow$ Rank 9 | **NO** |
| `low_volume_streak`| **0.5734** | 0.5627 | Rank 10 $\rightarrow$ Rank 10 | **NO** |

**Conclusion:** The predictive association and ranking of the selected features are exceptionally stable across time. The feature sets do NOT suffer from temporal selection leakage. They can be safely treated as fixed model architecture search spaces in Batch 4.

---

## 9. Supervised Modelling Methodology

### Walk-Forward Execution Protocol for Batch 4:
For each chronological fold $k \in \{0, 1, \dots, 9\}$:
1. **Fold Partitioning:**
   - Retrieve `(train_idx, val_idx)` from `generate_walk_forward_splits(df)`.
   - Ensure $X_{\text{train}}$ contains only observations where `split == "train"`, `~is_warmup`, and `hre_5d_train.notna()`.
2. **Independent Pipeline Instantiation:**
   - Instantiate a fresh pipeline instance for each candidate model:
     - Model 1: `LR_Standard` (`scaler_type="standard"`)
     - Model 2: `LR_Robust` (`scaler_type="robust"`)
     - Model 3: `RF_Candidate_A` (`candidate="A"`)
     - Model 4: `RF_Candidate_B` (`candidate="B"`)
     - Model 5: `HGBT` (`build_hgbt_pipeline()`)
3. **Training Fold Fitting:**
   - Extract model-specific feature matrices: $X_{\text{train}}^{(m)} = X_{\text{train}}[\text{features}^{(m)}]$.
   - Fit pipeline strictly on $(X_{\text{train}}^{(m)}, y_{\text{train}})$.
   - Transformers (`SimpleImputer`, `StandardScaler`, `RobustScaler`) learn statistics exclusively from fold training observations.
4. **Validation Inference:**
   - Extract validation features: $X_{\text{val}}^{(m)} = X_{\text{val}}[\text{features}^{(m)}]$.
   - Predict out-of-fold probabilities: $\hat{p}_{\text{val}}^{(m)} = \text{pipeline.predict\_proba}(X_{\text{val}}^{(m)})[:, 1]$.
   - Preprocessing transformers apply learned training statistics via `.transform()`; `.fit()` is never called.
5. **Prediction Recording:**
   - Record predictions in a structured validation results table containing `[ticker, date, cap_segment, hre_5d_train, pred_prob]`.

---

## 10. Isolation Forest Evaluation Methodology

### Fundamental Principle:
Isolation Forest detects **historical distributional unusualness**, NOT forward trading risk. It must NEVER be evaluated as a supervised competitor against LR, RF, or HGBT for predicting `hre_5d_train`.

### Approved Evaluation Methodology:
1. **Unsupervised Fit on Expanding History:**
   - Isolation Forest is fit on the identical training date boundaries as the supervised models.
   - Population: All post-warmup rows ($N = 466,486$), without target filtering.
   - 12 features from `ISOLATION_FOREST_FEATURES`.
2. **Out-of-Fold Anomaly Scoring:**
   - For each validation window, compute `anomaly_score_scaled` (0–100) and `isolation_flag` (binary top 5% outlier flag).
3. **Post-Hoc Diagnostic Correlation (Target-Independent Diagnostic):**
   - After scoring is completed, compute the observational correlation between `anomaly_score_scaled` and forward realized volatility / LRS spikes.
   - Measure what proportion of high-risk events ($\text{HRE} = 1$) coincided with historical anomaly flags.
   - This diagnostic provides insight into whether unusual volume/spread dislocations precede liquidity crises, without treating the algorithm as a supervised classifier.

---

## 11. Evaluation Metrics Framework

### Imbalanced Time-Series Evaluation Suite:

| Metric Category | Metric Name | Scikit-Learn Function | Role in Imbalanced Liquidity Risk Context | Reporting Level |
|:---|:---|:---|:---|:---|
| **Discrimination (Primary)** | **PR-AUC (Average Precision)** | `average_precision_score` | **Primary Model Selection Metric.** Evaluates precision across all recall thresholds on minority positive class ($18.2\%$). Invariant to true negatives. | Per Fold + Mean $\pm$ Std + Pooled |
| **Discrimination (Secondary)**| **ROC-AUC** | `roc_auc_score` | Overall ranking ability across the full false positive rate spectrum. | Per Fold + Mean $\pm$ Std + Pooled |
| **Threshold-Dependent** | **F1-Score** | `f1_score` | Harmonic mean of precision and recall at calibrated decision threshold. | Per Fold + Mean $\pm$ Std + Pooled |
| **Threshold-Dependent** | **Precision & Recall** | `precision_score`, `recall_score` | Measures false alarm rate vs. missed liquidity event rate. | Per Fold + Mean $\pm$ Std + Pooled |
| **Threshold-Dependent** | **Balanced Accuracy** | `balanced_accuracy_score` | $\frac{1}{2}(\text{Sensitivity} + \text{Specificity})$, penalizing majority class bias. | Per Fold + Mean $\pm$ Std + Pooled |
| **Probability Quality** | **Brier Score** | `brier_score_loss` | Mean squared probability error. Lower is better. | Per Fold + Mean $\pm$ Std + Pooled |
| **Probability Quality** | **Log Loss** | `log_loss` | Cross-entropy loss penalizing confident misclassifications. | Per Fold + Mean $\pm$ Std + Pooled |
| **Operational** | **Train & Predict Time** | Wall-clock timer | Measures computational feasibility for live daily deployment. | Per Fold + Total |
| **Operational** | **Fold Stability** | Coefficient of Variation | $\sigma_{\text{PR-AUC}} / \mu_{\text{PR-AUC}}$, measuring performance resilience across regimes. | Pooled Summary |

### Caution Regarding Pooled vs. Per-Fold Metrics:
In non-stationary financial markets, pooling probabilities across 10 temporal folds ($N = 163,357$) can slightly distort ranking if baseline risk shifts across years. Therefore, **both per-fold metrics (reported as Mean $\pm$ Std) and pooled out-of-fold metrics must be calculated and displayed side-by-side**.

---

## 12. Model Comparison Methodology

### Multi-Criteria Decision Framework:
No single metric should unilaterally dictate model selection. The 5 candidate models will be compared across four weighted dimensions:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Model Comparison Dimensions                           │
├───────────────────────┬────────────────────────┬────────────────────────────┤
│ 1. Discrimination     │ PR-AUC (Primary)       │ Highest out-of-fold score  │
│                       │ ROC-AUC (Secondary)    │ Ranking consistency        │
├───────────────────────┼────────────────────────┼────────────────────────────┤
│ 2. Temporal Stability │ Min Fold PR-AUC        │ Worst-case fold resilience │
│                       │ Std Dev across Folds   │ Stability across regimes   │
├───────────────────────┼────────────────────────┼────────────────────────────┤
│ 3. Probability Quality│ Brier Score / Log Loss │ Well-calibrated risk odds  │
├───────────────────────┼────────────────────────┼────────────────────────────┤
│ 4. Operational Cost   │ Training & Infer Time  │ Fast daily batch inference │
│                       │ Missing-Value Handling │ Native robustness          │
└───────────────────────┴────────────────────────┴────────────────────────────┘
```

### Threshold Tuning Safeguard (Zero Validation Leakage):
- Decision thresholds ($\tau$) must NEVER be tuned on validation fold labels.
- **Approved Strategy for Batch 4:**
  1. Primary model selection will be decided on **threshold-independent PR-AUC**.
  2. For operational metrics (F1, Precision, Recall), evaluate:
     - Baseline threshold: $\tau = 0.50$ (standard model default).
     - Prevalence-calibrated threshold: $\tau = 0.182$ (matching positive class base rate).
     - In-fold optimal threshold: $\tau^* = \arg\max_{\tau} F_1(y_{\text{train}}, \hat{p}_{\text{train}} \ge \tau)$, tuned strictly on training fold predictions and evaluated out-of-sample on validation.

---

## 13. Final Holdout Protection Audit

The final holdout comprises 250 trading sessions (`2025-10-20` to `2026-10-23`, 74,334 rows).

### Programmatic Safeguards Verified:
1. `ModelTrainer.generate_walk_forward_splits()`:
   ```python
   if "split" in df.columns:
       eligible_mask &= (df["split"] == "train")
   ```
   Observations where `split == "holdout"` or `split == "train_boundary_embargo"` are strictly masked.
2. In `test_holdout_and_embargo_excluded_from_training_populations`, asserting `"holdout" not in df.loc[train_idx, "split"]` and `"holdout" not in df.loc[val_idx, "split"]` passed across all folds.
3. In `AnomalyDetector.fit()`, the same `mask &= (df["split"] == "train")` guarantees the unsupervised model never ingests holdout data.
4. **Holdout Status:** Completely isolated. Zero holdout predictions will be generated during Batch 4.

---

## 14. Testing and Reproducibility Assessment

### Current Test Suite Status:
- 99 tests passed in 9.15s (100% green).
- Preprocessing pipeline tests: 13 passed (`tests/test_preprocessing.py`).
- Walk-forward purging tests: 8 passed (`tests/test_walk_forward_purging.py`).
- Core feature & label tests: 78 passed.

### Test Suite Additions Required for Batch 4:
To support Batch 4 implementation, the test suite must be expanded to cover:
1. `tests/test_evaluation.py`:
   - Unit tests for `ModelEvaluator.compute_classification_metrics` (testing PR-AUC, ROC-AUC, Brier score, F1 against known synthetic arrays).
   - Stratified cap segment evaluation tests.
2. `tests/test_model_trainer_walk_forward.py`:
   - End-to-end integration test running a 2-fold walk-forward validation on a small synthetic fixture, verifying that `train_and_evaluate_models` outputs valid prediction DataFrames and metric dictionaries.
3. Deterministic reproducibility test:
   - Verifying that setting `random_state=42` produces identical model weights and out-of-fold probabilities across runs.

---

## 15. Identified Risks and Gaps

| Gap ID | Severity | File / Component | Description of Defect / Gap | Required Correction Before / During Implementation |
|:---:|:---:|:---|:---|:---|
| **GAP-01** | **WARNING** | `src/modeling/preprocessing.py`<br>`get_model_features()` | Passing `model_name="random_forest_candidate_b"` raises `ValueError` because line 141 checks `if key in ("random_forest", "rf"):`. | Update `get_model_features` to recognize `"random_forest_candidate_b"`, `"rf_b"`, and `"random_forest_candidate_a"`, returning the respective feature list directly. |
| **GAP-02** | **WARNING** | `src/modeling/evaluation.py`<br>`ModelEvaluator` | `compute_classification_metrics` and `evaluate_by_cap_segment` raise `NotImplementedError`. | Implement `ModelEvaluator` with PR-AUC, ROC-AUC, F1, Precision, Recall, Balanced Accuracy, Brier Score, and Log Loss. |
| **GAP-03** | **WARNING** | `src/modeling/trainer.py`<br>`train_and_evaluate_models` | Currently raises `NotImplementedError` per Batch 3 stopping boundary. | Implement walk-forward loop orchestrating all 10 folds across the 5 supervised models and recording out-of-fold predictions. |
| **GAP-04** | **WARNING** | `configs/model_config.yaml`<br>`validation:` | Specifies `embargo_gap_days: 1` instead of `purge_sessions: 5`. | Update `configs/model_config.yaml` to specify `purge_sessions: 5`. |
| **GAP-05** | **NOTE** | Computational Runtime | Fitting 5 models × 10 folds = 50 model fits on 300k–440k rows. | Set `n_jobs=-1` on Random Forest and HistGradientBoosting; run models sequentially per fold; log progress per fold. |

---

## 16. Recommended Batch 4 Implementation Sequence

When implementation is approved, execute in the following strict order:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Batch 4 Recommended Implementation Flow                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Step 4.1: Bug Fixes & Config Update                                         │
│   • Fix `get_model_features` in `preprocessing.py` to route RF Candidate B. │
│   • Update `configs/model_config.yaml` to specify `purge_sessions: 5`.      │
│   • Verify unit tests pass.                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Step 4.2: Implement Metrics & Evaluation Engine (`evaluation.py`)           │
│   • Implement `ModelEvaluator.compute_classification_metrics`.              │
│   • Implement `ModelEvaluator.evaluate_by_cap_segment`.                     │
│   • Add unit tests in `tests/test_evaluation.py`.                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Step 4.3: Implement Walk-Forward Orchestrator (`trainer.py`)                │
│   • Implement `ModelTrainer.train_and_evaluate_models`.                     │
│   • Build out-of-fold prediction aggregator and per-fold metric tracker.    │
│   • Test on synthetic 2-fold fixture.                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Step 4.4: Execute Supervised Walk-Forward Validation Experiments            │
│   • Run 10-fold walk-forward validation across all 5 candidate models:      │
│     1. Logistic Regression (StandardScaler)                                 │
│     2. Logistic Regression (RobustScaler)                                   │
│     3. Random Forest Candidate A (35 features)                              │
│     4. Random Forest Candidate B (36 features)                              │
│     5. HistGradientBoosting (36 features)                                   │
│   • Save out-of-fold validation predictions to `data/interim/` or memory.   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Step 4.5: Execute Isolation Forest Unsupervised Anomaly Validation          │
│   • Fit Isolation Forest across expanding training history.                 │
│   • Generate out-of-fold anomaly scores and measure diagnostic correlation. │
├─────────────────────────────────────────────────────────────────────────────┤
│ Step 4.6: Model Benchmarking, Comparison & Selection                        │
│   • Compare models on PR-AUC, ROC-AUC, Brier score, stability, latency.    │
│   • Select winning model architecture.                                      │
│   • Compile comprehensive results report: `reports/phase8_batch4_report.md`.│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 17. Exact Files Expected to Change During Implementation

### Existing Files to Modify:
1. `src/modeling/preprocessing.py` — Fix `get_model_features` candidate routing for `random_forest_candidate_b`.
2. `src/modeling/evaluation.py` — Implement `ModelEvaluator` classification metrics and cap-segment stratification.
3. `src/modeling/trainer.py` — Implement `train_and_evaluate_models` walk-forward loop.
4. `configs/model_config.yaml` — Update `validation:` section with `purge_sessions: 5`.

### New Files to Create:
1. `tests/test_evaluation.py` — Unit tests for evaluation metrics suite.
2. `reports/phase8_batch4_modelling_report.md` — Final walk-forward evaluation, comparison leaderboard, and model selection report.

---

## 18. Final Readiness Verdict

### **VERDICT: READY WITH CORRECTIONS**

The existing repository architecture, preprocessing pipelines, and 5-session purged walk-forward validation engine are sound, mathematically leak-free, and thoroughly tested. 

Prior to launching the full walk-forward validation experiments, the following minor corrections must be executed during the first step of Batch 4 implementation:
1. **[Priority 1] Fix `get_model_features` Candidate Routing:** Update `src/modeling/preprocessing.py` to recognize `"random_forest_candidate_b"` and `"rf_b"`.
2. **[Priority 2] Implement `ModelEvaluator`:** Implement classification metrics (PR-AUC, ROC-AUC, Brier score, F1) in `src/modeling/evaluation.py` and verify with unit tests.
3. **[Priority 3] Synchronize Configuration:** Update `configs/model_config.yaml` to specify `purge_sessions: 5`.

Once these engineering steps are applied, the codebase is 100% ready for the Batch 4 machine learning training and validation experiments.

