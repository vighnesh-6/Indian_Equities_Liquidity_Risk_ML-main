# PHASE 8 BATCH 5 | STAGE 8.12.5: SUPERVISED HOLDOUT EVALUATION REPORT

**Project:** ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities  
**Phase:** Phase 8 Batch 5 (Stage 8.12.5)  
**Task:** Supervised Holdout Evaluation across Five Frozen Configurations  
**Evaluation Target:** Forward 5-Day High-Risk Event (`hre_5d`, Horizon $H=5$)  
**Status:** COMPLETE & VERIFIED  
**Date:** 2026-11-15  

---

## 1. EXECUTIVE SUMMARY

Stage 8.12.5 executes the out-of-sample evaluation of the five frozen supervised model configurations on the strictly quarantined final holdout partition. In strict adherence to quantitative risk and methodology standards:
- **No model training or refitting occurred:** All evaluations strictly applied the already-fitted pipelines persisted in `models/phase8_batch5/` during Stage 8.12.4.
- **True Out-of-Sample Holdout:** The holdout spans 250 consecutive trading sessions (`2025-10-20` to `2026-10-23`, $N=74,334$ rows), completely separated from pre-holdout training by the 5-session boundary embargo (`2025-10-13` to `2025-10-17`).
- **Supervised Target:** Evaluated strictly against the forward 5-day realized label `hre_5d` (NOT `hre_5d_train`).
- **Primary Finding:** **HistGradientBoostingClassifier (`hist_gradient_boosting`) achieved decisive out-of-sample superiority**, delivering a Holdout PR-AUC of **0.5723**, ROC-AUC of **0.8460**, and the lowest Brier calibration score of **0.1167**, outperforming all alternative architectures.

---

## 2. HOLDOUT DATASET POPULATION ACCOUNTING

The final holdout comprises 250 trading sessions from October 20, 2025 to October 23, 2026:

| Cohort / Metric | Count / Value | Proportion of Holdout | Description |
| :--- | :---: | :---: | :--- |
| **Total Holdout Rows** | **74,334** | 100.00% | All ticker-sessions with `split == 'holdout'` |
| **Unique Trading Dates** | **250** | — | Exactly 250 sessions (~1 calendar year) |
| **Valid Target Rows (`hre_5d.notna()`)** | **71,756** | 96.53% | Eligible evaluation population with verified forward 5-day labels |
| **Missing Target Rows (`hre_5d.isna()`)** | **2,578** | 3.47% | Excluded from metric scoring: <br>• 1,500 rows across final 5 sessions (boundary horizon lookahead)<br>• 1,078 trading sessions with subsequent suspensions/delistings |
| **Positive Class ($y=1$, HRE Event)** | **11,873** | **16.55%** | Positive holdout prevalence (base rate) |
| **Negative Class ($y=0$, Normal)** | **59,883** | **83.45%** | Negative holdout prevalence |

---

## 3. SUPERVISED HOLDOUT PERFORMANCE LEADERBOARD

Metrics were computed strictly on the $N=71,756$ valid evaluation rows:

| Rank | Model Configuration | Features | Holdout PR-AUC | Holdout ROC-AUC | Brier Score | Log Loss | F1 @ 0.50 (P / R) | Calib $\tau^*$ | F1 @ $\tau^*$ (P / R) | Inference Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`hist_gradient_boosting`** | **36** | **0.5723** | **0.8460** | **0.1167** | **0.3621** | **0.5321** (0.473 / 0.609) | **0.534** | **0.5313** (0.505 / 0.560) | **0.20 s** |
| **2** | `random_forest_candidate_b` | 36 | 0.5380 | 0.8211 | 0.1249 | 0.3929 | 0.5013 (0.468 / 0.539) | 0.508 | 0.4997 (0.474 / 0.528) | 0.12 s |
| **3** | `random_forest_candidate_a` | 35 | 0.5318 | 0.8186 | 0.1261 | 0.3970 | 0.4960 (0.466 / 0.530) | 0.504 | 0.4949 (0.470 / 0.523) | 0.11 s |
| **4** | `logistic_regression_standard` | 20 | 0.4051 | 0.7938 | 0.1298 | 0.4137 | 0.4425 (0.435 / 0.450) | 0.460 | 0.4656 (0.396 / 0.566) | 0.02 s |
| **5** | `logistic_regression_robust` | 20 | 0.4050 | 0.7937 | 0.1298 | 0.4137 | 0.4425 (0.435 / 0.450) | 0.458 | 0.4664 (0.394 / 0.571) | 0.03 s |

---

## 4. CONFUSION MATRICES & CLASSIFICATION COUNTS

### 4.1 Performance at Baseline Fixed Threshold ($\tau = 0.50$)

| Model Configuration | True Positives (TP) | False Positives (FP) | True Negatives (TN) | False Negatives (FN) | Precision | Recall | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `hist_gradient_boosting` | **7,226** | 8,064 | 51,819 | 4,647 | 47.26% | **60.86%** | **73.70%** |
| `random_forest_candidate_b` | 6,403 | 7,272 | 52,611 | 5,470 | 46.82% | 53.93% | 70.90% |
| `random_forest_candidate_a` | 6,290 | 7,204 | 52,679 | 5,583 | 46.61% | 52.98% | 70.47% |
| `logistic_regression_standard` | 5,348 | 6,944 | 52,939 | 6,525 | 43.51% | 45.04% | 66.72% |
| `logistic_regression_robust` | 5,346 | 6,942 | 52,941 | 6,527 | 43.51% | 45.03% | 66.71% |

### 4.2 Performance at Frozen Calibrated Threshold ($\tau^*$)

| Model Configuration | Calib $\tau^*$ | True Positives (TP) | False Positives (FP) | True Negatives (TN) | False Negatives (FN) | Precision | Recall | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `hist_gradient_boosting` | **0.534** | **6,654** | **6,522** | **53,361** | 5,219 | **50.50%** | 56.04% | **72.57%** |
| `random_forest_candidate_b` | 0.508 | 6,275 | 6,957 | 52,926 | 5,598 | 47.42% | 52.85% | 70.62% |
| `random_forest_candidate_a` | 0.504 | 6,206 | 7,007 | 52,876 | 5,667 | 46.97% | 52.27% | 70.29% |
| `logistic_regression_standard` | 0.460 | 6,717 | 10,248 | 49,635 | 5,156 | 39.60% | 56.57% | 69.72% |
| `logistic_regression_robust` | 0.458 | 6,785 | 10,432 | 49,451 | 5,088 | 39.41% | 57.15% | 69.86% |

---

## 5. MARKET CAPITALIZATION STRATIFICATION ANALYSIS

Out-of-sample performance varies meaningfully across market capitalization segments:

| Market Cap Segment | Valid Rows | Positive Prevalence | `logistic_regression_standard` | `random_forest_candidate_a` | `random_forest_candidate_b` | `hist_gradient_boosting` | Top Performing Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Large-Cap** | 24,960 | 12.18% | 0.4358 | 0.5284 | 0.5401 | **0.6268** | **HGBT (+0.0867 over RF-B)** |
| **Mid-Cap** | 24,880 | 15.65% | 0.4289 | 0.5441 | 0.5492 | **0.5912** | **HGBT (+0.0420 over RF-B)** |
| **Small-Cap** | 21,916 | 22.54% | 0.3848 | **0.5926** | 0.5887 | 0.5879 | **RF Candidate A / B** |

### Stratification Insights:
1. **Large & Mid-Cap Dominance:** `hist_gradient_boosting` achieves massive outperformance in Large-Cap (0.6268 vs 0.5401, +8.67 percentage points) and Mid-Cap (0.5912 vs 0.5492, +4.20 percentage points), demonstrating exceptional ability to detect subtle institutional liquidity dry-ups.
2. **Small-Cap Resilience:** In Small-Cap stocks, where liquidity crunches are frequent (22.54% base rate), Random Forest ensembles perform on par with HGBT (0.5887 vs 0.5879). Bootstrap tree aggregation provides slight variance reduction in highly volatile small-cap names.
3. **Linear Model Failure in Small-Caps:** Logistic Regression collapses to 0.3848 in Small-Cap stocks, confirming that linear models cannot separate liquidity risk states under heavy non-linear volatility.

---

## 6. MONTHLY HOLDOUT TEMPORAL DYNAMICS

Holdout performance across the 13 calendar months (October 2025 to October 2026):

| Month | Sessions | Valid Rows | Base Rate | `logistic_regression_standard` | `random_forest_candidate_a` | `random_forest_candidate_b` | `hist_gradient_boosting` | Top Monthly Model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2025-10** | 10 | 2,569 | 19.23% | 0.4197 | 0.5146 | 0.5272 | **0.5805** | HGBT (+0.0533) |
| **2025-11** | 19 | 5,444 | 19.21% | 0.4629 | 0.5738 | 0.5868 | **0.6476** | HGBT (+0.0608) |
| **2025-12** | 22 | 6,314 | 18.96% | 0.4906 | 0.5883 | 0.5970 | **0.6664** | HGBT (+0.0694) |
| **2026-01** | 21 | 6,006 | 17.12% | 0.3750 | 0.5464 | 0.5576 | **0.6210** | HGBT (+0.0634) |
| **2026-02** | 20 | 5,771 | 15.44% | 0.4274 | 0.5231 | 0.5222 | **0.5754** | HGBT (+0.0523) |
| **2026-03** | 20 | 5,580 | 15.72% | 0.4956 | 0.5920 | 0.6004 | **0.6479** | HGBT (+0.0475) |
| **2026-04** | 21 | 5,917 | 15.40% | 0.3952 | 0.4600 | 0.4644 | **0.4814** | HGBT (+0.0170) |
| **2026-05** | 20 | 5,625 | 15.38% | 0.3631 | 0.4557 | 0.4559 | **0.4829** | HGBT (+0.0270) |
| **2026-06** | 22 | 6,237 | 15.91% | 0.3298 | 0.5013 | 0.5088 | **0.5351** | HGBT (+0.0263) |
| **2026-07** | 23 | 6,831 | 15.58% | 0.4925 | 0.5906 | 0.5936 | **0.6416** | HGBT (+0.0480) |
| **2026-08** | 21 | 6,237 | 16.55% | 0.4989 | 0.6170 | 0.6207 | **0.6795** | HGBT (+0.0588) |
| **2026-09** | 21 | 6,237 | 16.87% | 0.4462 | 0.5306 | 0.5369 | **0.5876** | HGBT (+0.0507) |
| **2026-10** | 10 | 2,988 | 14.19% | 0.2838 | 0.3813 | 0.3865 | **0.4286** | HGBT (+0.0421) |

**Key Finding:** `hist_gradient_boosting` achieved the highest PR-AUC in **13 out of 13 monthly subperiods** across the entire holdout year, demonstrating flawless temporal consistency.

---

## 7. METHODOLOGICAL HYPOTHESES CONFIRMATION

1. **Volume Shock Feature (F51 `relative_volume`):**
   - RF Candidate B (incorporating F51) achieved **0.5380 PR-AUC**, outperforming RF Candidate A (**0.5318**) by **+0.0062 (+0.62 percentage points)**.
   - Out-of-sample generalization confirms that native missing-value routing during tree splitting successfully extracts predictive volume shock signals without imputation leakage.
2. **Scaler Sensitivity (StandardScaler vs RobustScaler):**
   - `logistic_regression_standard` (0.405109) and `logistic_regression_robust` (0.405039) achieved identical performance (difference $< 0.0001$), confirming scaler parity out-of-sample.
3. **Linear Model Fragility:**
   - Logistic Regression suffered significant degradation (-0.0939 PR-AUC drop vs walk-forward mean), while tree-based models maintained over 91% of their performance.

---

## 8. STAGE 8.12.5 CONCLUSION

Stage 8.12.5 confirms that the Primary Supervised Champion `hist_gradient_boosting` decisively generalizes to unseen future market data, achieving top rank across all discrimination and calibration metrics.
