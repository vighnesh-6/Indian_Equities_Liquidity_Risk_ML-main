# PHASE 8 BATCH 5 | STAGE 8.12.3: CONFIGURATION FREEZE REPORT

**Project:** ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities  
**Phase:** Phase 8 Batch 5 (Stage 8.12.3)  
**Task:** Select and Freeze Model Configurations  
**Target Variable:** Forward 5-Day High-Risk Event (`hre_5d_train`, Horizon $H=5$)  
**Status:** COMPLETE & FROZEN  
**Date:** 2026-11-15  

---

## 1. EXECUTIVE SUMMARY

Stage 8.12.3 formally establishes, registers, and freezes the model configurations for pre-holdout training and subsequent holdout evaluation. All configurations adhere strictly to the methodology defined in `reports/phase8_batch5_stage8_12_2_selection_methodology.md` and the empirical baseline established in Batch 4.

Five supervised configurations (representing three distinct algorithmic families) and one independent unsupervised anomaly detector have been frozen. A machine-readable configuration manifest has been saved to:
`reports/phase8_batch5/configuration_freeze/model_configuration_manifest.json`

---

## 2. FROZEN MODEL CONFIGURATIONS SPECIFICATION

### 2.1 Supervised Model Configurations

| Config ID | Model Identifier | Family | Feature Count | Scaling / Imputation | Key Hyperparameters | Random Seed |
| :--- | :--- | :--- | :---: | :--- | :--- | :---: |
| **CFG_01** | `logistic_regression_standard` | Linear Classification | 20 | Median Imputer + `StandardScaler` | $C=1.0$, L2, `solver='lbfgs'`, max_iter=1000, `class_weight='balanced'` | 42 |
| **CFG_02** | `logistic_regression_robust` | Linear Classification | 20 | Median Imputer + `RobustScaler` | $C=1.0$, L2, `solver='lbfgs'`, max_iter=1000, `class_weight='balanced'` | 42 |
| **CFG_03** | `random_forest_candidate_a` | Supervised Ensemble | 35 | Native tree NaN routing (No scaler) | $N=150$, depth=8, leaf=50, max_feat='sqrt', `class_weight='balanced_subsample'` | 42 |
| **CFG_04** | `random_forest_candidate_b` | Supervised Ensemble | 36 | Native tree NaN routing (No scaler) | $N=150$, depth=8, leaf=50, max_feat='sqrt', `class_weight='balanced_subsample'` | 42 |
| **CFG_05** | `hist_gradient_boosting` | Gradient Boosting | 36 | Native histogram binning (No scaler) | $N=200$, lr=0.05, leaf_nodes=31, l2_reg=0.1, `class_weight='balanced'` | 42 |

### 2.2 Independent Unsupervised Anomaly Detection

| Config ID | Model Identifier | Role | Feature Count | Preprocessing | Key Hyperparameters | Target Dependency |
| :--- | :--- | :--- | :---: | :--- | :--- | :---: |
| **CFG_06** | `isolation_forest` | Microstructure Telemetry | 12 | Median Imputer (No scaler) | $N=100$, contamination=0.05, max_samples='auto' | **NONE (Target-Independent)** |

---

## 3. EXACT FEATURE INVENTORIES AND ORDERING

### 3.1 Logistic Regression (20 Features)
`daily_return`, `return_5d`, `vol_5d`, `downside_vol_20d`, `max_drawdown_20d`, `rsi_14`, `log_turnover`, `turnover_ratio`, `volume_z20`, `amihud_ma20`, `low_volume_streak`, `daily_range`, `overnight_gap_abs`, `gap_z20`, `cs_spread_ma10`, `rolling_beta_60d`, `rolling_corr_60d`, `nifty_vol_20d`, `nifty_return_20d`, `vix_close`.

### 3.2 Random Forest Candidate A (35 Features)
`daily_return`, `return_5d`, `return_20d`, `return_z20`, `abs_return`, `price_rel_ma20`, `vol_5d`, `vol_20d`, `vol_ratio_5_20`, `downside_vol_20d`, `max_drawdown_20d`, `rsi_14`, `vol_of_vol`, `log_volume`, `volume_z20`, `amihud_ratio`, `amihud_ma20`, `amihud_z20`, `log_turnover`, `turnover_ratio`, `low_volume_streak`, `daily_range`, `daily_range_ma20`, `range_z20`, `overnight_gap`, `overnight_gap_abs`, `cs_spread_est`, `cs_spread_ma10`, `excess_return_1d`, `rolling_beta_60d`, `rolling_corr_60d`, `nifty_vol_20d`, `nifty_return_20d`, `vol_beta`, `vix_close`.

### 3.3 Random Forest Candidate B & HistGradientBoosting (36 Features)
Identical to Candidate A plus **F51 `relative_volume`** inserted at position 31:
`daily_return`, `return_5d`, `return_20d`, `return_z20`, `abs_return`, `price_rel_ma20`, `vol_5d`, `vol_20d`, `vol_ratio_5_20`, `downside_vol_20d`, `max_drawdown_20d`, `rsi_14`, `vol_of_vol`, `log_volume`, `volume_z20`, `amihud_ratio`, `amihud_ma20`, `amihud_z20`, `log_turnover`, `turnover_ratio`, `low_volume_streak`, `daily_range`, `daily_range_ma20`, `range_z20`, `overnight_gap`, `overnight_gap_abs`, `cs_spread_est`, `cs_spread_ma10`, `excess_return_1d`, `rolling_beta_60d`, `rolling_corr_60d`, `relative_volume`, `nifty_vol_20d`, `nifty_return_20d`, `vol_beta`, `vix_close`.

### 3.4 Isolation Forest (12 Features)
`amihud_ratio`, `amihud_z20`, `volume_ratio`, `volume_z20`, `vol_20d`, `vol_ratio_5_20`, `daily_range`, `range_z20`, `overnight_gap_abs`, `cs_spread_est`, `nifty_vol_20d`, `return_z20`.

---

## 4. PRE-TRAINING VERIFICATION GATE AUDIT

Prior to commencing Stage 8.12.4 training, all six verification criteria were audited:

| # | Verification Criterion | Status | Audit Findings |
| :---: | :--- | :---: | :--- |
| 1 | **All 5 Supervised Configurations Verified** | **PASS** | Logistic Regression (Standard/Robust), Random Forest (Cand A/B), and HGBT match specifications. |
| 2 | **Isolation Forest Verified Separately** | **PASS** | 12 features, target-independent, 0–100 scaled score mapping confirmed. |
| 3 | **Feature Counts & Identities Exact** | **PASS** | Counts match 20, 20, 35, 36, 36, and 12 features exactly with zero ordering discrepancies. |
| 4 | **Hyperparameters & Preprocessing Exact** | **PASS** | Imputation, scaling, tree parameters, and seeds match `configs/model_config.yaml` and Batch 4. |
| 5 | **Configuration Manifest Complete** | **PASS** | `reports/phase8_batch5/configuration_freeze/model_configuration_manifest.json` generated and valid. |
| 6 | **Zero Existing Dataset Modifications** | **PASS** | SHA-256 hashes of `features.parquet` and `features_with_targets.parquet` match baseline byte-for-byte. |

---

## 5. TRANSITION TO STAGE 8.12.4

Stage 8.12.3 is complete and verified. The configuration freeze is finalized and locked. Training of all five supervised configurations and Isolation Forest on eligible pre-holdout data may proceed under Stage 8.12.4.
