# Phase 8 Batch 3 — Final Methodology Clarification Audit
## Pre-Implementation Approval Gate & Methodological Specification
### ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities

**Audit Execution Date:** November 13, 2026  
**Audit Scope:** Pre-Implementation Resolution of Batch 3 Methodological Gates  
**Target Document Audited:** `reports/phase8_batch3_readiness_audit.md`  
**Authoritative Dataset:** `data/processed/features_with_targets.parquet` (565,190 rows × 75 columns)  
**Supervised Training Population:** `(split == "train") & (hre_5d_train.notna())` = **466,346 rows**  
**Unsupervised Training Population:** `(split == "train") & (~is_warmup)` = **466,486 rows**  
**Scikit-Learn Version Verified:** `1.8.0`  
**File Created:** `reports/phase8_batch3_final_clarification_audit.md` (Single permitted file)  
**Final Readiness Verdict:** **READY FOR BATCH 3 IMPLEMENTATION**

---

## 1. Executive Summary

This final clarification audit resolves the remaining methodology gates identified during the readiness review of Phase 8 Batch 3: Missing-Value Treatment & Preprocessing Pipelines. 

### Key Audit Resolutions:
1. **Random Forest Native NaN Support & F51 (`relative_volume`) (Audit A):** Empirically verified against the active `scikit-learn 1.8.0` runtime that `RandomForestClassifier` has native support for missing values (NaNs). At each split point, the tree grower assigns missing values to whichever child branch minimizes impurity. To ensure an unbiased evaluation, two candidate feature sets are formalized: **Candidate A (35 features, baseline excluding F51)** and **Candidate B (36 features, incorporating F51 natively)**.
2. **5-Day Target Horizon Overlap & Walk-Forward Embargo/Purging (Audit B):** Derived mathematically from `src/labels/targets.py` that because `hre_5d_train` evaluates the rolling maximum LRS over the forward interval $[t+1, t+5]$, any training observation within 5 sessions of a validation boundary peeks into validation period LRS values. The existing `embargo_gap_days: 1` in `configs/model_config.yaml` is mathematically insufficient. A **5-session purging rule** (or 5-session embargo buffer) is strictly required to ensure zero forward lookahead leakage across walk-forward folds, exactly mirroring the `train_boundary_embargo` established in Phase 7.
3. **Isolation Forest 140-Observation Difference (Audit C):** Empirically traced the exact 140-observation difference between the supervised population (466,346) and the unsupervised training population (466,486) to trading suspensions/halts in three stocks: `PATANJALI.NS` (70 sessions), `FORCEMOT.NS` (46 sessions), and `MOTILALOFS.NS` (24 sessions). These are valid historical sessions where forward lookahead targets were uncalculable (`hre_5d_train = NaN`), but historical feature values exist. Isolation Forest correctly trains on all **466,486 post-warmup training rows** without target-label filtering, fully respecting the final holdout quarantine.
4. **Logistic Regression Scaler Configuration (Audit D):** Verified that `build_logistic_regression_pipeline` should provide a configurable interface: `scaler_type = "standard"` (baseline) with `'robust'` as an option for comparison within cross-validation folds.
5. **Isolation Forest Scale Invariance (Audit E):** Proven mathematically and confirmed empirically on synthetic and market data that axis-aligned uniform random splits in Isolation Forest are **100% invariant to linear scaling** ($r = 1.000$, bit-level identical anomaly scores). Adding scalers is computationally redundant; raw feature ingestion post-imputation is optimal.

---

## 2. Scope & Preservation Confirmation

This audit was conducted under strict read-only protocols:
* **No File Mutations:** `features.parquet`, `features_with_targets.parquet`, existing source modules in `src/`, notebooks in `notebooks/`, configurations in `configs/`, and prior reports remain 100% byte-for-byte identical to their pre-audit state.
* **No Pipeline or Model Artifacts:** Zero estimators, scalers, imputers, or transformers were created or serialized.
* **Single Permitted File Creation:** Only this audit document (`reports/phase8_batch3_final_clarification_audit.md`) has been generated.
* **Dataset Hash Integrity:**
  * `features.parquet`: `c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae` (**VERIFIED MATCH**)
  * `features_with_targets.parquet`: `c6cebf3d93d022b11765c206b8e2ffb84e578a020e0d913d937b37d04028aa91` (**VERIFIED MATCH**)
* **Pytest Test Suite:** 78 passed in 3.74s (100% green).

---

## 3. Audit A — Random Forest NaN Support & F51 (`relative_volume`)

### 1. Verification of Runtime Estimator Capabilities
Inspected the docstrings and runtime behavior of `scikit-learn 1.8.0`:
* **`HistGradientBoostingClassifier`:** Native NaN support. Bins NaNs into a dedicated category and routes missing values to the child branch that minimizes gradient loss during training.
* **`RandomForestClassifier`:** **Native NaN support confirmed.** Authoritative library docstring:
  > *"This estimator has native support for missing values (NaNs). During training, the tree grower learns at each split point whether samples with missing values are assigned to the left or right child consequently. If no missing values were encountered for a given feature during training, then samples with missing values are mapped to whichever child has the most samples."*

### 2. Random Forest Evaluation Candidates
Because scikit-learn 1.8.0 handles missing values natively without raising errors, the exclusion of `relative_volume` (F51) in Random Forest is no longer a technical limitation, but an empirical modeling question:

| Candidate | Feature Count | Input Features | Missingness Profile | Methodological Hypothesis |
|:---|:---:|:---|:---|:---|
| **RF Candidate A (Baseline)** | **35 Features** | Excludes `relative_volume` (F51). | All 35 features have $< 0.08\%$ missingness. | Pure, clean, low-missingness baseline; avoids benchmark volume noise. |
| **RF Candidate B (Experimental)**| **36 Features** | Includes `relative_volume` (F51). | F51 contains 7,047 NaNs ($1.51\%$). | Evaluates whether native tree routing of benchmark volume adds predictive lift in RF. |

### 3. Recommendation
* **Primary Pipeline Architecture:** In `src/modeling/preprocessing.py`, configure `build_random_forest_pipeline` to accept either candidate list.
* **Walk-Forward Validation Protocol:** Evaluate **Candidate A (35 features)** as the authoritative baseline. Candidate B (36 features) will be evaluated alongside it under identical fold boundaries to determine if F51 provides incremental out-of-fold PR-AUC lift.

---

## 4. Audit B — Five-Day Target Horizon & Walk-Forward Embargo/Purging

### 1. Mathematical Formulation of the Forward Target Horizon
Inspected `src/labels/targets.py` lines 8–15:
$$fwd\_max\_lrs\_5d(i, t) = \max_{k \in [1, 5]} \text{LRS}(i, t+k)$$
The binary target `hre_5d_train` classifies whether this forward maximum exceeds the market-cap threshold $\theta_{90, s}$. 
* Observation day $t$ is strictly excluded.
* Future days $t+1, t+2, t+3, t+4, t+5$ are required.

### 2. The Boundary Overlap Problem
Consider a walk-forward training fold ending at session $T_{\text{train}}$. If validation begins at session $T_{\text{val\_start}} = T_{\text{train}} + 1$:
* The label for $T_{\text{train}}$ depends on LRS from $[T_{\text{train}}+1, T_{\text{train}}+5]$, which are **Sessions 1, 2, 3, 4, 5 of the validation period**!
* The label for $T_{\text{train}}-1$ depends on LRS from $[T_{\text{train}}, T_{\text{train}}+4]$, touching **Sessions 1, 2, 3, 4 of validation**.
* The label for $T_{\text{train}}-2$ touches **Sessions 1, 2, 3 of validation**.
* The label for $T_{\text{train}}-3$ touches **Sessions 1, 2 of validation**.
* The label for $T_{\text{train}}-4$ touches **Session 1 of validation**.
* The label for $T_{\text{train}}-5$ depends on LRS from $[T_{\text{train}}-4, T_{\text{train}}]$, strictly contained within the training period.

```
Training Sessions:   ... [T-5] [T-4] [T-3] [T-2] [T-1] [ T ] | [T+1] [T+2] [T+3] [T+4] [T+5] ... (Validation)
Label at T:                      └────────────────────────────► [  1  ] [  2  ] [  3  ] [  4  ] [  5  ] (LEAKAGE!)
Label at T-1:              └────────────────────────────► [  1  ] [  2  ] [  3  ] [  4  ]         (LEAKAGE!)
Label at T-2:        └────────────────────────────► [  1  ] [  2  ] [  3  ]                 (LEAKAGE!)
Label at T-3:  └────────────────────────────► [  1  ] [  2  ]                         (LEAKAGE!)
Label at T-4:  └──────────────────────► [  1  ]                                 (LEAKAGE!)
Label at T-5:  └────────────────► [ T ] (SELF-CONTAINED - ZERO LEAKAGE)
```

### 3. Inadequacy of the Existing 1-Day Embargo
In `configs/model_config.yaml`, line 9 specifies `embargo_gap_days: 1`. 
* If only 1 trading session is embargoed, $T_{\text{val\_start}} = T_{\text{train}} + 2$.
* The label at $T_{\text{train}}$ still looks into $T+2, T+3, T+4, T+5$, contaminating the first 4 sessions of validation.
* **Therefore, a 1-day embargo does NOT prevent target lookahead leakage.**

### 4. Methodological Distinction: Purging vs. Embargo
Per quantitative finance standards (López de Prado, 2018):
* **Purging:** Eliminating training observations whose forward label evaluation window extends into the validation/testing period.
* **Embargo:** Eliminating training/validation observations immediately *after* an evaluation window to prevent autoregressive feature leakage.

### 5. Derived Fold Construction Rule
Because the forward target spans exactly 5 trading sessions, a **5-session purging rule** is mathematically mandatory:
$$\text{For every training fold ending at date } D_{\text{train\_end}}, \text{ mask or drop the final 5 trading sessions } [D_{\text{train\_end}-4}, D_{\text{train\_end}}].$$

Alternatively, set the embargo gap between training and validation to **at least 5 trading sessions** (`embargo_gap_days: 5`).
* **Precedent in Project Architecture:** This derives directly from Phase 7, where the exact same 5-session boundary (`2025-10-13` to `2025-10-17`, 1,444 rows) was embargoed and masked to `NaN` as `train_boundary_embargo` before the holdout began on `2025-10-20`.

---

## 5. Audit C — Isolation Forest Training Population

### 1. Empirical Origin of the 140-Observation Difference
Inspected the differences between `(split == "train") & (hre_5d_train.notna())` (466,346 rows) and `(split == "train") & (~is_warmup)` (466,486 rows):

```
Ticker Breakdown of the 140 Rows:
1. PATANJALI.NS : 70 sessions (2019-11-14 to 2020-02-20) [Trading suspension during Ruchi Soya acquisition]
2. FORCEMOT.NS  : 46 sessions (2023-10-26 to 2024-03-01) [Trading halt / corporate restructuring]
3. MOTILALOFS.NS: 24 sessions (2023-12-22 to 2024-01-29) [Trading suspension]
Total Discrepancy: Exactly 140 rows
```

### 2. Why Are Targets Missing for These Rows?
In Phase 7, `compute_forward_max_lrs` requires 5 contiguous trading sessions (`min_periods=5`). When trading was halted for 24 to 70 consecutive sessions, the forward lookahead window contained NaNs, forcing `fwd_max_lrs_5d` and `hre_5d_train` to evaluate to `NaN`.

### 3. Population Legitimacy for Isolation Forest
* **Are these rows legitimate historical training observations?** **YES.** They are within the training date boundaries (`2018-01-01` to `2025-10-10`) and occurred long after the initial 80-day warmup window (`is_warmup == False`).
* **Do they belong to the holdout?** **NO.** 0 rows belong to the holdout.
* **Target Independence Principle:** Supervised models cannot train on these 140 rows because they lack a target label. However, Isolation Forest is **unsupervised**. Trading halts and post-halt resumptions represent physical market anomalies (zero volume, frozen prices). Filtering Isolation Forest by `hre_5d_train.notna()` would violate target-neutrality.
* **Authoritative Unsupervised Population:**
  $$\mathcal{D}_{\text{unsupervised, train}} = \left\{ (i, t) \in \mathcal{D} \mid \text{split}_{i, t} = \text{"train"} \;\wedge\; \text{is\_warmup}_{i, t} = \text{False} \right\} \quad (N = 466,486)$$
  Inside each walk-forward fold $k$, Isolation Forest trains on all post-warmup stock observations up to the fold's purged training date boundary.

---

## 6. Audit D — Logistic Regression Scaler Configuration

### 1. Scaling Physics in Regularized Linear Models
Logistic Regression minimizes an $L_2$-penalized objective:
$$\mathcal{L}(w, b) = -\sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i) \right] + \frac{1}{2C} \sum_{j=1}^{20} w_j^2$$
Because the penalty $\frac{1}{2C} w_j^2$ is uniform across all coefficients, features with larger natural scales (e.g., `rsi_14` spanning $[0, 100]$ vs `daily_return` spanning $[-0.10, +0.10]$) will have their coefficients artificially suppressed unless standardized.

### 2. StandardScaler vs. RobustScaler Trade-Offs
* **`StandardScaler()` (Baseline):** Centers to mean zero, divides by standard deviation. Preserves canonical financial interpretability: $\beta_j$ represents the increase in log-odds of a liquidity shock per 1 standard deviation move. In Batch 2, extreme unscaled features were pruned, yielding well-behaved condition numbers ($14.8$).
* **`RobustScaler()` (Alternative):** Centers to median, divides by Interquartile Range ($\text{IQR} = q_{75} - q_{25}$). Highly resilient to heavy-tailed distributions where extreme kurtosis inflates the standard deviation.

### 3. Recommended Interface Design
In `src/modeling/preprocessing.py`:
```python
def build_logistic_regression_pipeline(
    features: List[str],
    scaler_type: str = "standard",  # "standard" | "robust" | None
    imputer_strategy: str = "median",
    C: float = 1.0,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> Pipeline:
    # Supports comparative evaluation within walk-forward folds without pipeline refactoring
```

---

## 7. Audit E — Isolation Forest Scaling

### 1. Mathematical Scale Invariance Proof
An Isolation Tree selects a feature $j$ uniformly at random, and chooses a split point $p \sim \text{Uniform}(\min(X_j), \max(X_j))$.
For any positive affine transformation $X'_j = a X_j + b$ ($a > 0$):
$$\min(X'_j) = a \min(X_j) + b, \quad \max(X'_j) = a \max(X_j) + b$$
$$p' \sim \text{Uniform}(a \min(X_j) + b, a \max(X_j) + b) \iff p' = a p + b$$
The branching decision:
$$X'_{i, j} \le p' \iff a X_{i, j} + b \le a p + b \iff X_{i, j} \le p$$
**The tree partition and path length $h(x)$ are mathematically invariant to linear scaling.**

### 2. Empirical Verification
Tested on multiscale synthetic and financial matrices:
* $\text{Correlation}(\text{Scores}_{\text{raw}}, \text{Scores}_{\text{StandardScaler}}) = \mathbf{1.0000000000000000}$
* $\text{Correlation}(\text{Scores}_{\text{raw}}, \text{Scores}_{\text{RobustScaler}}) = \mathbf{1.0000000000000000}$
* All anomaly scores are bit-level identical.

### 3. Recommendation
* **Baseline:** Use raw features post-median imputation. Adding `StandardScaler` or `RobustScaler` provides zero mathematical difference and consumes unnecessary CPU cycles.
* **Pipeline:** `Pipeline([('imputer', SimpleImputer(strategy='median')), ('detector', IsolationForest(...))])`.

---

## 8. Consolidated Implementation Decisions

| Decision Domain | Approved Outcome | Empirical Rationale & Mathematical Justification |
|:---|:---|:---|
| **RF Missing-Value Handling** | **Native Handling Supported; Median Imputer Optional** | Verified that `scikit-learn 1.8.0` supports NaNs natively in `RandomForestClassifier`. `SimpleImputer` can be included for deterministic stability. |
| **RF F51 Inclusion** | **Evaluate Candidate A (35) as Baseline; Compare Candidate B (36)** | Candidate A is the clean baseline. Candidate B evaluates whether native handling of `relative_volume` adds predictive lift across walk-forward folds. |
| **HGBT Missing-Value Handling**| **Native Handling (No Imputer)** | HGBT natively routes missing values to optimal gradient split branches; optimal for structural index volume gaps. |
| **Walk-Forward Overlap/Purge** | **Mandatory 5-Session Purge Buffer** | `hre_5d_train` forward lookahead $[t+1, t+5]$ peeks into validation if $<5$ sessions are purged. Existing 1-day embargo is insufficient; 5 sessions required. |
| **Isolation Forest Population** | **466,486 Rows (Post-Warmup Train)** | Decoupled from `hre_5d_train.notna()`. Includes 140 historical halt sessions without label contamination; strictly obeys holdout quarantine. |
| **Logistic Regression Scaler** | **Configurable (`StandardScaler` default)** | Mandatory for $L_2$ regularization. Factory function parameterizes `scaler_type='standard'` vs `'robust'`. |
| **Isolation Forest Scaler** | **Raw Features Post-Imputation** | Proven mathematically and empirically that tree-based random partitioning is 100% scale-invariant ($r = 1.000$). |
| **Holdout Protection** | **100% Quarantined (250 sessions)** | Oct 20, 2025 to Oct 23, 2026 (71,756 valid rows) remains completely blind and untouched throughout Phase 8. |

---

## 9. Final Implementation Architecture

The architecture confirmed across all audits:

```
src/modeling/
├── preprocessing.py      # Factory builders: build_logistic_regression_pipeline, 
│                         # build_random_forest_pipeline, build_hgbt_pipeline,
│                         # build_isolation_forest_pipeline
├── trainer.py            # Supervised walk-forward training orchestrator (10 expanding folds)
│                         # with 5-session target purging and out-of-fold metrics tracking
├── anomaly.py            # Encapsulates Isolation Forest pipeline, unsupervised fit, 0-100 scaling
├── evaluation.py         # PR-AUC, ROC-AUC, Brier score, and market-cap tier evaluations
├── scorer.py             # Model artifact loading and production inference
└── explainability.py     # Feature importances and stability tracking across folds
```

### Execution Rules Enforced:
1. **Independent Pipelines:** Each model has its own pipeline with dedicated transformers and estimators.
2. **Fold-Level Fitting:** Inside each fold $k$, pipelines are instantiated fresh and fitted strictly on `X_train_k`.
3. **Validation Transform Only:** `X_val_k` is evaluated using `predict_proba()` / `decision_function()`, calling `.transform()` only.
4. **Target Decoupling:** Isolation Forest receives zero target labels.

---

## 10. Final Implementation Readiness Verdict

$$\mathbf{READY \; FOR \; BATCH \; 3 \; IMPLEMENTATION}$$

### Justification:
* All 5 methodological gates (RF NaNs/F51, 5-day target overlap purging, Isolation Forest population, LR scaling, and Isolation Forest scale-invariance) are mathematically and empirically resolved.
* The scikit-learn 1.8.0 environment capabilities are documented and verified.
* The walk-forward purging rule protects against all temporal lookahead leakage.
* Dataset integrity is 100% verified.

---

## 11. Final Data Preservation Verification

```
========================================================================================
                          FINAL DATA PRESERVATION VERIFICATION
========================================================================================
File Path: data/processed/features.parquet
Expected SHA-256: c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae
Verified SHA-256: c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae
Status: EXACT MATCH (100% UNTOUCHED)
----------------------------------------------------------------------------------------
File Path: data/processed/features_with_targets.parquet
Expected SHA-256: c6cebf3d93d022b11765c206b8e2ffb84e578a020e0d913d937b37d04028aa91
Verified SHA-256: c6cebf3d93d022b11765c206b8e2ffb84e578a020e0d913d937b37d04028aa91
Status: EXACT MATCH (100% UNTOUCHED)
----------------------------------------------------------------------------------------
Git Working Tree Status: Clean (Only reports/phase8_batch3_final_clarification_audit.md created)
Pytest Test Suite Status: 78 passed in 3.74s (100% green)
Existing Artifacts Modified: Exactly 0
New Artifacts Created: Exactly 1 (reports/phase8_batch3_final_clarification_audit.md)
========================================================================================
```

---

## 12. Stopping Condition

**AUDIT COMPLETE.** In accordance with strict instructions:
- No pipelines have been implemented.
- No model code has been written.
- No models have been trained or serialized.
- Execution has stopped at the final pre-implementation gate.

**Awaiting your explicit authorization before proceeding to Phase 8 Batch 3 implementation.**

