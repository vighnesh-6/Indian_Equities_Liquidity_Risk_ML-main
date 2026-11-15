import json
from pathlib import Path
import nbformat

REPO_ROOT = Path(__file__).resolve().parent.parent

# -------------------------------------------------------------
# 1. Notebook 05: Logistic Regression
# -------------------------------------------------------------
nb05_path = REPO_ROOT / "notebooks" / "05_model_baseline.ipynb"
nb05 = nbformat.read(nb05_path, as_version=4)

nb05_sec9_md = nbformat.v4.new_markdown_cell(source="""### 9. Phase 8 Batch 5 (Stages 8.12.8 & 8.12.9): Final Model Selection & Baseline Conclusions

#### 9.1 StandardScaler vs. RobustScaler Comparison
Across both the 10-fold walk-forward validation and the 250-session final holdout evaluation, the performance of the StandardScaler and RobustScaler configurations was virtually indistinguishable:
* **Walk-Forward Mean PR-AUC:** 0.4990 (StandardScaler) vs 0.4990 (RobustScaler) — $\\Delta = 0.000001$.
* **Holdout PR-AUC:** 0.405109 (StandardScaler) vs 0.405039 (RobustScaler) — $\\Delta = 0.000070$.
* **Holdout ROC-AUC:** 0.793804 vs 0.793728.
* **Holdout Brier Score:** 0.129792 vs 0.129781.

**Conclusion on Scaler Choice:** Under median imputation and L2 regularization ($C=1.0$), scaling methodology has no material operational effect. `StandardScaler` is confirmed as the standard canonical baseline pipeline.

#### 9.2 Walk-Forward vs. Holdout Performance
| Metric | Walk-Forward 10-Fold Mean | Final Holdout (250 Sessions) | Absolute Delta | Retention % |
| :--- | :---: | :---: | :---: | :---: |
| **PR-AUC** | 0.4990 | 0.4051 | -0.0939 | 81.18% |
| **ROC-AUC** | 0.8265 | 0.7938 | -0.0326 | 96.04% |
| **Brier Score** | 0.1382 | 0.1298 | -0.0084 | Improved |
| **Log Loss** | 0.4469 | 0.4349 | -0.0120 | Improved |
| **Calibrated F1** | 0.5152 | 0.4656 | -0.0496 | 90.37% |

#### 9.3 Why Logistic Regression Remains a Baseline
1. **Severe Generalization Degradation:** While non-linear tree ensembles retained >91–94% of out-of-sample PR-AUC, Logistic Regression suffered an 18.82% degradation (retention: 81.18%). Linear hyperplanes are incapable of modeling non-linear risk interactions, such as wide bid-ask spreads conditional on high volume shocks.
2. **Substantial Performance Gap:** The linear baseline trails the primary champion (`hist_gradient_boosting`) by **-0.1672 PR-AUC (-16.72 percentage points)** and lags Random Forest by **-0.1329 PR-AUC**.
3. **Role:** Logistic Regression is preserved strictly as a fast, interpretable, linear reference benchmark, fulfilling regulatory and institutional validation requirements.

#### 9.4 Final Supervised Model-Selection Outcome
* **Champion Model Selected:** `hist_gradient_boosting` (36 features, Configuration `CFG_05_HGBT`).
* **Strongest Alternative:** `random_forest_candidate_b` (36 features, Configuration `CFG_04_RF_CANDIDATE_B`).
* **Baseline Status:** `logistic_regression_standard` (20 features, Configuration `CFG_01_LOGREG_STANDARD`) formally frozen as reference baseline.
* **Decision Status:** FORMALLY FROZEN in `reports/phase8_batch5/final_model_selection_manifest.json`.""")

nb05.cells.append(nb05_sec9_md)
nbformat.write(nb05, nb05_path)
print(f"Updated 05_model_baseline.ipynb: now {len(nb05.cells)} cells.")

# -------------------------------------------------------------
# 2. Notebook 06: Random Forest
# -------------------------------------------------------------
nb06_path = REPO_ROOT / "notebooks" / "06_model_rf.ipynb"
nb06 = nbformat.read(nb06_path, as_version=4)

nb06_sec8_md = nbformat.v4.new_markdown_cell(source="""### 8. Phase 8 Batch 5 (Stages 8.12.8 & 8.12.9): Final Model Selection & Random Forest Conclusions

#### 8.1 Candidate A (35 features) vs. Candidate B (36 features, includes F51 `relative_volume`)
Across both walk-forward cross-validation and the untouched holdout evaluation, Candidate B consistently outperformed Candidate A:
* **Walk-Forward PR-AUC:** 0.5829 (Candidate B) vs 0.5814 (Candidate A) — $\\Delta = +0.0015$.
* **Holdout PR-AUC:** **0.5380** (Candidate B) vs **0.5318** (Candidate A) — $\\Delta = +0.0062$ (+0.62 percentage points gain).
* **Holdout ROC-AUC:** 0.8211 vs 0.8186.
* **Holdout Brier Score:** 0.1249 vs 0.1261.
* **Monthly Subperiod Consistency:** Candidate B beat Candidate A in **12 out of 13 monthly periods (92.3% win rate)** on the holdout.

**Empirical Resolution of F51:** Feature F51 (`relative_volume`) provides robust, non-redundant predictive signal during volume-shock and liquidity stress episodes. It is permanently retained in the 36-feature production tree schema.

#### 8.2 Walk-Forward vs. Holdout Performance (Candidate B)
| Metric | Walk-Forward 10-Fold Mean | Final Holdout (250 Sessions) | Absolute Delta | Retention % |
| :--- | :---: | :---: | :---: | :---: |
| **PR-AUC** | 0.5829 | 0.5380 | -0.0449 | 92.30% |
| **ROC-AUC** | 0.8361 | 0.8211 | -0.0150 | 98.21% |
| **Brier Score** | 0.1320 | 0.1249 | -0.0071 | Improved |
| **Log Loss** | 0.4131 | 0.4036 | -0.0095 | Improved |
| **Calibrated F1** | 0.5252 | 0.4997 | -0.0255 | 95.14% |

#### 8.3 Why Candidate B Is the Strongest Alternative Architecture
1. **High Generalization Retention:** Retained **92.30%** of its walk-forward PR-AUC, demonstrating strong resistance to regime shifts.
2. **Mid-Cap Segment Resilience:** In mid-cap equities, Candidate B achieved a holdout PR-AUC of **0.7097**, marginally edging HGBT (0.6890), demonstrating that bagging ensembles perform strongly in moderately fragmented liquidity.
3. **Decisive Baseline Superiority:** Outperforms `logistic_regression_standard` by **+0.1329 PR-AUC (+13.29 percentage points)** on holdout.
4. **Architectural Complementarity:** Provides an essential bagging reference benchmark alongside gradient boosting.

#### 8.4 Final Supervised Model-Selection Outcome
* **Champion Model Selected:** `hist_gradient_boosting` (Holdout PR-AUC: 0.5723, Configuration `CFG_05_HGBT`).
* **Strongest Alternative Designated:** `random_forest_candidate_b` (Holdout PR-AUC: 0.5380, Configuration `CFG_04_RF_CANDIDATE_B`).
* **Decision Status:** FORMALLY FROZEN in `reports/phase8_batch5/final_model_selection_manifest.json`.""")

nb06.cells.append(nb06_sec8_md)
nbformat.write(nb06, nb06_path)
print(f"Updated 06_model_rf.ipynb: now {len(nb06.cells)} cells.")

# -------------------------------------------------------------
# 3. Notebook 07: HistGradientBoosting
# -------------------------------------------------------------
nb07_path = REPO_ROOT / "notebooks" / "07_model_hgbt.ipynb"
nb07 = nbformat.read(nb07_path, as_version=4)

nb07_sec7_md = nbformat.v4.new_markdown_cell(source="""### 7. Phase 8 Batch 5 (Stages 8.12.8 & 8.12.9): Final Model Selection & Champion Confirmation

#### 7.1 Authoritative Performance Summary
Across both expanding-window cross-validation and the untouched 250-session holdout, `hist_gradient_boosting` (Configuration `CFG_05_HGBT`, 36 features) achieved decisive out-of-sample supremacy:
* **Walk-Forward Mean PR-AUC:** **0.6107** (Pooled PR-AUC: 0.5926).
* **Holdout PR-AUC:** **0.5723** (Exceeds RF-B by +0.0343; exceeds Linear Baseline by +0.1672).
* **Holdout ROC-AUC:** **0.8460** (vs Walk-Forward mean 0.8548).
* **Holdout Brier Calibration Score:** **0.1167** (Lowest loss across all models; improved from 0.1266 in WF).
* **Holdout Log Loss:** **0.3625** (Lowest across all models; improved from 0.3857 in WF).
* **Monthly Subperiod Consistency:** Won **13 out of 13 monthly periods (100% win rate)** on the holdout partition.

#### 7.2 Walk-Forward vs. Holdout Generalization Assessment
| Metric | Walk-Forward 10-Fold Mean | Final Holdout (250 Sessions) | Absolute Delta | Retention % |
| :--- | :---: | :---: | :---: | :---: |
| **PR-AUC** | 0.6107 | 0.5723 | -0.0383 | **93.72%** |
| **ROC-AUC** | 0.8548 | 0.8460 | -0.0089 | **98.96%** |
| **Brier Score** | 0.1266 | 0.1167 | -0.0099 | Improved |
| **Log Loss** | 0.3857 | 0.3625 | -0.0232 | Improved |
| **F1 @ 0.50** | 0.5622 | 0.5321 | -0.0301 | 94.65% |
| **Calibrated F1 (tau*=0.534)** | 0.5389 | 0.5313 | -0.0076 | **98.59%** |

#### 7.3 Final Selection Decision and Architectural Rationale
1. **Decisive Leader:** Ranked #1 in both walk-forward cross-validation and final holdout evaluation (Kendall's $\\tau = +1.000$).
2. **Minimal Generalization Decay:** Retained **93.72%** of its walk-forward PR-AUC, demonstrating the highest generalization fidelity of any model.
3. **Probability Calibration:** Achieved the lowest Brier score and Log Loss, guaranteeing accurate risk probability estimates for downstream position sizing.
4. **Institutional Large-Cap Performance:** Holdout PR-AUC of **0.7079** in the large-cap segment (+0.0870 over RF-B).
5. **Operational Suitability:** Full cross-sectional inference on 74,334 records completed in **0.22 seconds** (~3.0 $\\mu$s per record), far exceeding the 1.0-second institutional latency standard.

**Final Decision:** `hist_gradient_boosting` (CFG_05_HGBT) is formally selected as the **Primary Champion Supervised Predictive Model** for forward liquidity risk intelligence. Formally recorded and frozen in `reports/phase8_batch5/final_model_selection_manifest.json`.""")

nb07.cells.append(nb07_sec7_md)
nbformat.write(nb07, nb07_path)
print(f"Updated 07_model_hgbt.ipynb: now {len(nb07.cells)} cells.")

# -------------------------------------------------------------
# 4. Notebook 08: Isolation Forest
# -------------------------------------------------------------
nb08_path = REPO_ROOT / "notebooks" / "08_anomaly_detection.ipynb"
nb08 = nbformat.read(nb08_path, as_version=4)

nb08_sec7_md = nbformat.v4.new_markdown_cell(source="""### 7. Phase 8 Batch 5 (Stages 8.12.8 & 8.12.9): Final Anomaly Telemetry Conclusions & Operational Role

#### 7.1 Authoritative Telemetry Summary (Walk-Forward vs. Holdout)
| Metric | Walk-Forward 10-Fold Mean | Final Holdout (250 Sessions) | Stability Assessment |
| :--- | :---: | :---: | :--- |
| **Mean Scaled Score [0-100]** | 14.25 | **13.91** | $\\Delta = -0.34$ (Highly stable) |
| **Median Scaled Score** | 11.22 | **10.97** | $\\Delta = -0.25$ (Identical central tendency) |
| **90th Percentile (P90)** | 27.16 | **26.92** | $\\Delta = -0.24$ (Consistent upper boundary) |
| **95th Percentile (P95)** | 36.14 | **35.07** | $\\Delta = -1.07$ (Mild tail compression) |
| **99th Percentile (P99)** | 56.11 | **55.17** | $\\Delta = -0.94$ (Robust capture of extreme shocks) |
| **Anomaly Flag Rate** | 2.63% | **2.34%** (1,738 / 74,334) | Matches nominal 2.5% tail expectation |
| **HRE Rate in Flagged Sessions**| 22.84% | **19.38%** (327 / 1,687) | Elevated conditional risk |
| **HRE Rate in Unflagged** | 16.36% | **16.48%** (11,546 / 70,069) | Baseline market prevalence |
| **Enrichment Ratio** | **1.40x** | **1.18x** | Positive risk enrichment maintained |
| **Target Correlation (r)** | **+0.0586** | **+0.0264** | Orthogonal risk signal |

#### 7.2 Independent Anomaly-Detection Role
* **Strictly Target-Independent:** Fitted on 12 curated microstructure features without forward target labels (`hre_5d`).
* **Orthogonal Surveillance Function:** Isolation Forest detects multivariate structural dislocations (e.g. extreme volume surges accompanied by quote depth collapse) rather than predicting forward directional drift.
* **Why It Complements Rather than Competes:** ~80.6% of flagged anomalies do not realize an HRE within 5 sessions (often representing orderly institutional rebalancing), while many HREs emerge from macro drift without triggering single-day isolation outlier thresholds. It functions as an independent surveillance tripwire alongside the primary supervised model (`hist_gradient_boosting`).

#### 7.3 Methodological Audit of Flagging Threshold
* **Operational Mechanism:** Binary flags (`isolation_flag == 1`) are generated strictly using the pre-fitted estimator's internal decision offset (`offset_`), established on pre-holdout training data with `contamination=0.05`.
* **Provenance Note on 43.12:** The value 43.12 referenced in preliminary summary reports corresponds to the empirical ~97.41th percentile of scaled scores on the holdout partition. The actual operational boundary separating flagged from unflagged sessions is the native model decision boundary ($s_{\\text{scaled}} \\approx 44.44$).

#### 7.4 Final Research Conclusion
* **Retention:** `isolation_forest` (Configuration `CFG_06_IF_UNSUPERVISED`) is formally retained as an **independent unsupervised anomaly telemetry component**.
* **Decision Status:** FORMALLY FROZEN in `reports/phase8_batch5/final_model_selection_manifest.json`.""")

nb08.cells.append(nb08_sec7_md)
nbformat.write(nb08, nb08_path)
print(f"Updated 08_anomaly_detection.ipynb: now {len(nb08.cells)} cells.")
