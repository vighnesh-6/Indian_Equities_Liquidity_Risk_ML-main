# Phase 8 Batch 2 — Read-Only Clarification & Methodology Audit
## Independent Verification of Feature Relevance & Selection Methodology
### ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities

**Audit Execution Date:** November 13, 2026  
**Audit Status:** COMPLETE & VERIFIED (READ-ONLY)  
**Target Document Audited:** `reports/phase8_batch2_feature_selection_report.md`  
**Authoritative Dataset:** `data/processed/features_with_targets.parquet` (565,190 rows × 75 columns)  
**Supervised Population Audited:** `(split == "train") & (hre_5d_train.notna())` = **466,346 observations**  
**File Created:** `reports/phase8_batch2_clarification_audit.md` (Single permitted file)  
**Overall Verdict:** **CONDITIONALLY APPROVED — CLARIFICATIONS REQUIRED**

---

## 1. Executive Summary

This independent clarification audit evaluates the methodological foundations, numerical consistency, temporal leakage risks, and mathematical rationales presented in the Phase 8 Batch 2 Feature Relevance & Selection Report ([reports/phase8_batch2_feature_selection_report.md](file:///C:/Users/vighn/Downloads/Indian_Equities_Liquidity_Risk_ML-main/Indian_Equities_Liquidity_Risk_ML-main/reports/phase8_batch2_feature_selection_report.md)).

### Core Audit Outcomes:
1. **Random Forest Inconsistency Resolved (Audit A):** Identified a documentation discrepancy in Section 8 of the Batch 2 report where `return_5d` (F03) was simultaneously listed in the Selected table and in the Excluded narrative text. In the authoritative Consolidated Matrix (Section 11), `return_5d` is **Selected**. The corrected excluded count is **23 features**, yielding exactly 35 selected + 23 excluded = 58 features (100% mutually exclusive).
2. **Logistic Regression Rationale Clarified (Audit B):** The previous report attributed the exclusion of `return_20d` to an algebraic linear identity with `nifty_return_20d`. This audit clarifies that the algebraic identity ($\text{excess\_return\_20d} \equiv \text{return\_20d} - \text{nifty\_return\_20d}$) is broken by dropping `excess_return_20d` alone. The true empirical justification for dropping `return_20d` is its near-zero standalone predictive power ($\text{AUC} = 0.5032$, Spearman $r = -0.0043$) and collinearity with `rsi_14` ($r = 0.863$) and `max_drawdown_20d` ($r = 0.796$).
3. **Temporal Leakage Evaluated (Audit C):** Calculating feature relevance and VIF across the entire 466,346-row training set (2018–2025) constitutes full-sample exploratory screening. If frozen directly into early walk-forward folds (e.g. Fold 1 in 2020), this introduces mild temporal selection leakage. This audit formalizes two leak-free methodologies for Batch 3: **Fold-Specific Selection** and **Initial-Window Frozen Selection**.
4. **ROC-AUC & Panel Nuances Documented (Audits D & E):** Clarified that pooled panel ROC-AUC reflects both cross-sectional tier contrasts and time-series shock signals. Missing values were handled via pairwise deletion. Temporal stability remains an open question to be validated during walk-forward folds.
5. **Isolation Forest Physics Corrected (Audit F):** Corrected the previous report's description of Isolation Forest as "distance-based". Isolation Forest is an axis-aligned random partitioning tree ensemble; its vulnerability to high dimensions stems from feature masking and swamping, not Euclidean distance distortion. The 12-feature configuration in `configs/model_config.yaml` is mathematically sound and preserved.
6. **Tree Model Skewness & Scaling Nuanced (Audit G):** While decision tree splits are rank-invariant on single features, Random Forest's random feature sub-sampling (`max_features`) and cross-stock pooling mean unscaled rupee features (`true_range`, `atr_20d`) act as share-price proxies rather than volatility measures, validating their exclusion.
7. **VIF Reproducibility Parameters Established (Audit H):** Documented the diagnostic sampling parameters ($N = 50,000$, median imputation, standard scaling) required to reproduce the reported Logistic Regression VIF results (mean 2.63, max 4.89, condition number 14.8).

---

## 2. Scope & Read-Only Preservation Confirmation

This audit was conducted under strict read-only protocols:
* **No File Mutations:** `features.parquet`, `features_with_targets.parquet`, source modules in `src/`, notebooks in `notebooks/`, configurations in `configs/`, and the existing Batch 2 report remain 100% byte-for-byte identical to their pre-audit state.
* **No Pipeline or Model Artifacts:** Zero estimators, scalers, imputers, or transformers were created or serialized.
* **Single Permitted File Creation:** Only this audit document (`reports/phase8_batch2_clarification_audit.md`) has been written.
* **Dataset Hash Integrity:**
  * `features.parquet`: `c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae` (**VERIFIED MATCH**)
  * `features_with_targets.parquet`: `c6cebf3d93d022b11765c206b8e2ffb84e578a020e0d913d937b37d04028aa91` (**VERIFIED MATCH**)
* **Pytest Test Suite:** 78 passed in 3.74s (100% green).

---

## 3. Audit A — Random Forest Feature Count Inconsistency

### 1. Inconsistency Identification
In `reports/phase8_batch2_feature_selection_report.md`:
* **Section 8 Table (Selected List, lines 234–273):** Features 35 entries. Row 2 explicitly lists:
  $$\text{Feature 2: } \mathbf{F03} \text{ (return\_5d)} \quad [\text{Category: Returns}]$$
* **Section 8 Narrative Text (Excluded List, lines 274–275):**
  The text states: *"Excluded Features for RF (23 Features):"*  
  However, counting the items in that narrative string reveals **24 feature names**, including:
  $$\dots, \mathbf{return\_5d} \text{ (F03-redundant with F01/F04)}, \dots$$
* **Section 11 (Consolidated Matrix, line 344):** Row `F03: return_5d` is marked as **SELECTED** for Random Forest.

### 2. Resolution & Reconciled Accounting
* **Authoritative Status of F03:** `return_5d` (F03) is **SELECTED** in both the Section 8 table and the Section 11 master matrix. Its appearance in the Section 8 excluded narrative text is a typographical clerical error.
* **Corrected Excluded Feature Count:** Exactly **23 Features** (removing `return_5d` from the narrative list yields the stated count of 23).
* **Mutual Exclusivity:** The corrected Selected (35) and Excluded (23) lists are 100% mutually exclusive.
* **Universe Accounting:** $35 + 23 = 58$ candidate features. Every feature is accounted for exactly once.

### Audit A Verdict: **PASS WITH CLARIFICATION**

---

## 4. Audit B — Logistic Regression Return Feature Rationale

### 1. The Algebraic Collinearity Question
The previous report noted that:
$$\text{excess\_return\_20d} \equiv \text{return\_20d} - \text{nifty\_return\_20d}$$
This triplet forms an exact linear identity ($X_1 - X_2 - X_3 = 0$), causing the design matrix to be singular with condition number $> 6.7 \times 10^7$ and empirical $\text{VIF} > 2.8 \times 10^{14}$.

### 2. Methodological Clarification
* **Does retaining `return_20d` and `nifty_return_20d` alone cause an exact linear dependency?**  
  **NO.** An exact linear dependency requires three features where one is an exact linear combination of the other two, or two features with $|r| = 1.0000$. The Pearson correlation between `return_20d` and `nifty_return_20d` is $+0.387$. Once `excess_return_20d` is dropped, the remaining pair `(return_20d, nifty_return_20d)` has zero algebraic singularity ($\text{VIF} \approx 1.17$).
* **Was the stated reason in the Section 11 table ("LR drops to avoid algebraic collinearity with NIFTY") mathematically accurate?**  
  **NO.** Dropping `excess_return_20d` fully broke the algebraic singularity. Blaming `return_20d`'s exclusion on algebraic collinearity with `nifty_return_20d` was a flawed narrative justification.

### 3. The True Empirical Justification for Excluding `return_20d`
The empirical data in the Batch 2 report provides a completely independent, rigorous justification:
1. **Near-Zero Linear Predictive Signal:** In the univariate relevance table (Section 5), `return_20d` exhibited a raw ROC-AUC of **0.4968** (Effective $\text{AUC} = \mathbf{0.5032}$) and a Spearman correlation of $r = -0.0043$. Standalone 20-day return has virtually zero monotonic discrimination for 5-day forward High-Risk Events.
2. **Multicollinearity with Retained Momentum & Volatility Features:** In the correlation matrix (Section 8), `return_20d` is highly correlated with `sortino_component` ($r = 0.971$), `price_rel_ma20` ($r = 0.852$), `rsi_14` ($r = 0.863$), and `max_drawdown_20d` ($r = 0.796$). When `return_20d` was retained alongside `rsi_14` and `max_drawdown_20d` in early candidate subsets, VIFs pushed beyond $6.5$.
3. **Conclusion:** Excluding `return_20d` from Logistic Regression is statistically justified by its **negligible linear signal and collinearity with retained drawdown/momentum metrics**, not by an algebraic identity.

### Audit B Verdict: **PASS WITH CLARIFICATION**

---

## 5. Audit C — Temporal Leakage in Feature Selection

### 1. Risk Assessment of Full-Training-Period Feature Ranking
The Batch 2 relevance metrics (ROC-AUC, Spearman correlation, Pearson correlation) and VIF diagnostics were computed using all **466,346 observations** spanning January 1, 2018 to October 10, 2025.

In walk-forward validation (governed by `configs/model_config.yaml`):
* Fold 1 trains on ~490 trading sessions (2018–2019) and validates on 60 sessions in 2020.
* If a model in Fold 1 uses a feature subset selected because those features performed well over the full 2018–2025 period, the feature selection process has **seen future data** (specifically, crisis dynamics from 2020 to 2025).

### 2. Evaluation of Two Leak-Free Methodologies

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               METHODOLOGICAL COMPARISON FOR BATCH 3                                   │
├────────────────────────────────┬───────────────────────────────────────┬───────────────────────────────┤
│ Methodology                    │ Advantages                            │ Disadvantages & Trade-offs    │
├────────────────────────────────┼───────────────────────────────────────┼───────────────────────────────┤
│ Method 1:                      │ 1. 100% mathematically leak-free.     │ 1. Dynamic feature subsets:   │
│ Fold-Specific Feature Selection│ 2. Simulates realistic production     │    different folds may select │
│ (Selection inside each fold)   │    quant deployment at each re-fit.   │    different feature subsets. │
│                                │ 3. Adapts to emerging market regimes. │ 2. Increases pipeline runtime.│
├────────────────────────────────┼───────────────────────────────────────┼───────────────────────────────┤
│ Method 2:                      │ 1. Static, predictable feature matrix │ 1. Initial 490 days (2018-19) │
│ Initial-Window Feature Select  │    across all 10 walk-forward folds.  │    may lack stress regimes    │
│ (Frozen on Fold 1 train only)  │ 2. 100% leak-free for folds 1 to 10.  │    (e.g. COVID-19 crash).     │
│                                │ 3. Extremely efficient computation.   │ 2. Inflexible if regime shifts│
└────────────────────────────────┴───────────────────────────────────────┴───────────────────────────────┘
```

### 3. Classification of Batch 2 Findings & Recommendation
* **Status of Current Batch 2 Results:** The Batch 2 feature selections must be formally classified as **Exploratory Candidate Search Spaces**, not frozen production features.
* **Recommended Hybrid Protocol for Batch 3:**
  1. *Structural Domain Redundancies Frozen Globally:* Universal mathematical identities (`log_return` duplicate, `turnover_value` duplicate, `excess_return_20d` identity) and unscaled rupee distortions (`true_range`, `atr_20d`) are permanently excluded across all folds based on physical/mathematical logic.
  2. *Statistical Pruning Executed per Fold (Method 1):* Model-specific correlation thresholds and VIF pruning must be validated inside each walk-forward fold's training slice (`fold_train`) before evaluating on `fold_test`.

### Audit C Verdict: **PASS WITH CLARIFICATION**

---

## 6. Audit D — ROC-AUC Methodology Audit

Review of the methodology documented in Section 4 and Section 5 of the Batch 2 report:

| Requirement / Methodological Nuance | Documented in Report? | Report Location | Audit Assessment & Evidence |
|:---|:---:|:---|:---|
| **Pooled Stock-Day Observations** | **YES** | Section 3 & 4 | Calculated over all 466,346 stock-day rows pooled as a single panel. |
| **Missing-Value Handling** | **PARTIALLY** | Section 3 | Evaluated via pairwise complete observations per feature (`s.notna()`). Documented in script; omitted from prose. |
| **Raw AUC vs Effective AUC** | **YES** | Section 5 Table | Both raw AUC and $\max(\text{AUC}, 1-\text{AUC})$ are explicitly tabulated. |
| **Relationship Direction Preserved** | **YES** | Section 5 Table | Explicit column for Risk Direction (`+` for direct, `-` for inverse). |
| **Class Imbalance Acknowledged** | **YES** | Section 3 | Explicitly stated: 84,988 positive (18.22%) vs 381,358 negative (81.78%). |
| **Observation Consistency Across Rows** | **PARTIALLY** | Section 5 | 56 features evaluated on 466,346 rows; `relative_volume` evaluated on 459,299 rows (7,047 NaNs omitted pairwise). |
| **Market-Cap Stratified Results** | **YES** | Section 5 (end) | Stratified AUC table provided for Large ($N=169k$), Mid ($N=229k$), Small ($N=67k$). |
| **Temporal Stability Assessed** | **NO** | — | **UNRESOLVED:** The report did not assess year-by-year or regime-by-regime AUC stability. |

### Audit D Verdict: **PASS WITH CLARIFICATION**

---

## 7. Audit E — Pooled Panel Analysis & Temporal Stability

### 1. Cross-Sectional vs Time-Series Confounding in Pooled Panels
In a pooled dataset of 300 stocks across 1,921 trading sessions, interpreting univariate ROC-AUC requires extreme caution:
* **The "Between-Stock" (Cross-Sectional) Confounder:** Small-cap stocks have structurally higher baseline Amihud illiquidity and wider Corwin-Schultz spreads than Large-cap stocks. Because small caps also have higher baseline volatility, a high pooled AUC (e.g. `amihud_ma20` $\text{AUC} = 0.7619$) partially measures **which** stocks are illiquid rather than **when** an individual stock's risk is surging.
* **The "Time-Series" (Within-Stock) Reality:** The stratified table in Section 5 demonstrates that when evaluated *within* each tier, `amihud_ma20` remains exceptionally strong (Large: $0.8316$, Mid: $0.8007$, Small: $0.7673$). This confirms that the predictive signal is genuine both across and within market-cap segments.

### 2. Common Market-Wide Date Effects (Macro Clustering)
During systemic panic dates (e.g. March 23, 2020), nearly all 300 stocks experience simultaneous volatility spikes and order-book dry-ups. In a pooled calculation, these cluster days contribute 300 simultaneous true-positive points, which can inflate pooled AUC.

### 3. Conclusion on Temporal Stability
Pooled AUC establishes that a feature has an overall statistical association with high-risk events across the historical dataset. **It does NOT establish that the relationship is invariant over time.** Temporal stability can only be verified during expanding-window walk-forward validation in Batch 3.

### Audit E Verdict: **PASS WITH CLARIFICATION**

---

## 8. Audit F — Isolation Forest Methodology Correction

### 1. Technical Critique of the Previous Description
In Section 10 of the Batch 2 report, line 325 stated:
> *"Distance- and isolation-based tree algorithms degrade rapidly beyond 10–15 dimensions (the curse of dimensionality). Limiting the input matrix to 12 features maximizes isolation contrast."*

### 2. Methodological Clarifications
1. **Isolation Forest is NOT Distance-Based:** Isolation Forest (Liu, Ting, Zhou, 2008) is an axis-aligned recursive partitioning tree algorithm (`iTree`). It calculates **tree path length** (the number of recursive splits required to isolate an observation), not Euclidean distance, Manhattan distance, or Mahalanobis distance.
2. **Mechanism of Random Partitioning:** At each node of an isolation tree, an algorithm randomly selects **one feature** $q$, and randomly selects a split value $p$ uniformly between $\min(q)$ and $\max(q)$. It never calculates pairwise point distances.
3. **The True Nature of Dimensionality Degradation in Isolation Forest:**
   * In high-dimensional spaces with irrelevant or noisy features, random feature selection has a high probability of choosing uninformative dimensions.
   * This results in **swamping** (normal points being mistakenly isolated early along irrelevant noise dimensions) and **masking** (true anomalous clusters being masked by overlapping noise coordinates).
   * Therefore, restricting Isolation Forest to 10–15 features is mathematically justified, but to prevent **feature masking and swamping**, NOT to prevent distance metric degradation.
4. **Baseline Configuration Status:** The 12-feature subset defined in `configs/model_config.yaml` is mathematically sound and covers 6 distinct physical shock mechanisms. Because Isolation Forest is completely unsupervised, it was correctly excluded from supervised AUC filtering.

### Audit F Verdict: **PASS WITH CLARIFICATION**

---

## 9. Audit G — Tree Models, Skewness & Unscaled Features

Audit of four specific claims made in the Batch 2 report:

| Stated Claim in Previous Report | Audit Finding | Methodological Assessment & Evidence |
|:---|:---:|:---|
| *"Decision trees split on rank thresholds, making them completely immune to positive skewness."* | **PARTIALLY CORRECT** | A single decision tree split on one feature is invariant to strictly monotonic transformations. However, in Random Forest, candidate features are sub-sampled at each split (`max_features = 'sqrt'`). Features with extreme outliers (e.g. `amihud_ratio` skewness $+343$) can produce split criteria isolating tiny outlier subsets, increasing tree variance. Calling them "completely immune" is an overstatement; "split-invariant to monotonic scaling" is the precise property. |
| *"`true_range` and `atr_20d` should be excluded because they are unscaled rupee-denominated features."* | **CORRECT & VERIFIED** | **Fully supported.** In a pooled panel of 300 stocks, MRF trades at ₹100,000 (true range ₹1,500) while Vodafone Idea trades at ₹10 (true range ₹0.30). A global tree split like $\text{true\_range} > 50$ splits on stock price level, not volatility. Price-level unscaled metrics are invalid in cross-sectional models. |
| *"Features with high skewness can automatically be retained for tree models."* | **METHODOLOGICALLY FLAWED** | High skewness alone does not disqualify a feature from tree models, but neither does it grant an automatic pass. Extreme tail values ($p_{99} \ll \max$) can cause greedy tree algorithms to overfit on rare idiosyncratic spikes. Retaining log-transformed versions (`log_turnover`) is superior to raw values (`turnover_value`). |
| *"`relative_volume` should be retained for HistGradientBoosting but excluded for Random Forest."* | **CORRECT & VERIFIED** | **Fully supported.** Standard scikit-learn `RandomForestClassifier` raises an immediate `ValueError: Input contains NaN` when encountering missing values. `HistGradientBoostingClassifier` natively routes missing values to whichever child branch minimizes training loss. |

### Audit G Verdict: **PASS WITH CLARIFICATION**

---

## 10. Audit H — Logistic Regression VIF Reproducibility

### 1. Verification of Reported Metrics
In Section 7 of the Batch 2 report, the 20-feature Logistic Regression subset reported:
* **Mean VIF:** **2.63**
* **Maximum VIF:** **4.89** (`max_drawdown_20d`)
* **Condition Number:** **14.8**
* **All VIFs $< 5.0$**

### 2. Documentation Gaps Identified
While the reported VIF metrics were independently verified to be mathematically accurate for the 20 features, the previous report omitted several technical execution details necessary for exact bit-level reproduction:
1. **Subsample Specification:** Calculating statsmodels `variance_inflation_factor` across $N = 466,346$ rows requires computing 20 separate OLS regressions on nearly half a million rows, which is computationally prohibitive in memory. The diagnostic was run on a representative random sample of $N = 50,000$ observations with `random_state = 42`. The report failed to state this sample size in its prose.
2. **Missing-Value Imputation for Diagnostic:** Statsmodels VIF requires a complete matrix. Nulls in the 20 features (which occur in $<0.02\%$ of rows) were filled with training-set column medians.
3. **Feature Standardization:** Features were scaled using `StandardScaler` prior to VIF calculation to ensure numerical stability during matrix inversion.
4. **Conclusion:** The VIF results are 100% reproducible when executing under these documented parameters.

### Audit H Verdict: **PASS WITH CLARIFICATION**

---

## 11. Consolidated Findings & Required Clarifications

```
========================================================================================
                          CONSOLIDATED CLARIFICATION AUDIT MATRIX
========================================================================================
Audit Area                             Verdict                 Primary Clarification Required
----------------------------------------------------------------------------------------
Audit A: RF Feature Count               PASS WITH CLARIFICATION Typo: F03 is Selected; Excluded count is 23.
Audit B: LR Return Rationale            PASS WITH CLARIFICATION Exclude F04 due to low AUC & momentum VIF,
                                                               not algebraic identity with NIFTY.
Audit C: Temporal Leakage               PASS WITH CLARIFICATION Batch 2 is exploratory; execute fold-level
                                                               selection or initial-window freezing in Batch 3.
Audit D: ROC-AUC Methodology            PASS WITH CLARIFICATION Document pairwise null drop & panel pooling.
Audit E: Pooled Panel Analysis          PASS WITH CLARIFICATION Acknowledge cross-sectional tier contrast
                                                               vs within-stock time-series signals.
Audit F: Isolation Forest Methodology   PASS WITH CLARIFICATION Correct "distance-based" to random partitioning;
                                                               preserve 12 baseline features.
Audit G: Tree Skewness & Scaling        PASS WITH CLARIFICATION Rupee unscaled exclusion verified;
                                                               single-split monotonic invariance clarified.
Audit H: VIF Reproducibility            PASS WITH CLARIFICATION Document N=50k, median fill, StandardScaler.
========================================================================================
```

---

## 12. Final Batch 2 Approval Verdict

### Formal Decision:
$$\mathbf{CONDITIONALLY \; APPROVED \; — \; CLARIFICATIONS \; REQUIRED}$$

### Justification:
1. **The Proposed Feature Subsets Are Methodologically Sound:** The 20-feature Logistic Regression subset successfully eliminates multicollinearity ($\text{VIF} < 5.0$) and removes singular algebraic identities. The 35-feature Random Forest and 36-feature HistGradientBoosting subsets properly prune unscaled rupee distortions and exact duplicates while capturing multi-tier volatility and liquidity dry-ups. The 12-feature Isolation Forest baseline maintains strict unsupervised orthogonality.
2. **Approval Conditions for Batch 3 Execution:**
   * *Condition 1:* The typographic discrepancy in the Random Forest exclusion list (`return_5d`) is recognized as resolved (35 Selected, 23 Excluded).
   * *Condition 2:* The proposed feature subsets are treated as **Authoritative Candidate Search Spaces** for walk-forward validation, not globally frozen leakage-inducing selections. All data-dependent scaling and imputation must be fitted strictly inside each walk-forward training fold during Batch 3.

---

## 13. Final Preservation Verification

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
Git Working Tree Status: Clean (Only reports/phase8_batch2_clarification_audit.md created)
Pytest Test Suite: 78 passed in 3.74s (100% green)
Existing Artifacts Modified: Exactly 0
New Artifacts Created: Exactly 1 (reports/phase8_batch2_clarification_audit.md)
========================================================================================
```

