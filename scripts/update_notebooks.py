"""Script to generate updated, fully functional Phase 8 Batch 4 notebooks.

Updates:
- notebooks/05_model_baseline.ipynb (Logistic Regression: StandardScaler vs RobustScaler)
- notebooks/06_model_rf.ipynb (Random Forest: Candidate A vs Candidate B)
- notebooks/07_model_hgbt.ipynb (HistGradientBoosting Classifier)
- notebooks/08_anomaly_detection.ipynb (Isolation Forest Unsupervised Anomaly Detection)
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def make_markdown_cell(source_lines):
    if isinstance(source_lines, str):
        source_lines = [source_lines]
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else line + "\n" for line in source_lines]
    }


def make_code_cell(source_lines):
    if isinstance(source_lines, str):
        source_lines = [source_lines]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith("\n") else line + "\n" for line in source_lines]
    }


def build_notebook_05():
    cells = [
        make_markdown_cell([
            "# Notebook 05: Baseline Model (Logistic Regression)",
            "## ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities",
            "",
            "**Phase:** Phase 8 Batch 4 (Stages 8.9–8.11)",
            "**Target:** High-Risk Event (`hre_5d_train`, 5-day forward horizon)",
            "**Evaluation Protocol:** 10-Fold Expanding-Window Walk-Forward Validation with 5-Session Purging",
            "",
            "### Objectives",
            "1. Evaluate Logistic Regression baseline across 10 expanding-window walk-forward folds.",
            "2. Compare feature scaling methodologies: **StandardScaler** vs. **RobustScaler** on the 20 approved features.",
            "3. Enforce zero temporal data leakage: fit scalers strictly on training folds; purge 5 trading sessions prior to validation.",
            "4. Perform inner chronological threshold calibration to optimize operational F1 score without outer validation leakage.",
            "5. Analyze market capitalization segment performance (Large, Mid, Small Cap) and inspect coefficient signs."
        ]),
        make_code_cell([
            "# Imports and Environment Setup",
            "import os",
            "import sys",
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            "# Add repository root to path regardless of working directory",
            "current_dir = Path('.').resolve()",
            "if (current_dir / 'data' / 'processed').exists():",
            "    REPO_ROOT = current_dir",
            "elif (current_dir.parent / 'data' / 'processed').exists():",
            "    REPO_ROOT = current_dir.parent",
            "else:",
            "    REPO_ROOT = current_dir",
            "    for p in [current_dir] + list(current_dir.parents):",
            "        if (p / 'data' / 'processed').exists():",
            "            REPO_ROOT = p",
            "            break",
            "",
            "if str(REPO_ROOT) not in sys.path:",
            "    sys.path.insert(0, str(REPO_ROOT))",
            "",
            "from src.modeling.preprocessing import build_logistic_regression_pipeline, get_model_features",
            "from src.modeling.evaluation import ModelEvaluator",
            "",
            "print(f'Repository root: {REPO_ROOT}')",
            "print('Environment initialized successfully.')"
        ]),
        make_markdown_cell([
            "### 1. Approved Baseline Features & Pipeline Architecture",
            "",
            "The baseline models utilize the 20 approved non-redundant features identified in Phase 8 Batch 2.",
            "Both pipelines apply `SimpleImputer(strategy='median')` followed by feature scaling and `LogisticRegression(class_weight='balanced', C=1.0, solver='lbfgs')`."
        ]),
        make_code_cell([
            "# Inspect the approved feature inventory for Logistic Regression",
            "lr_features = get_model_features('logistic_regression')",
            "print(f'Approved features for Logistic Regression ({len(lr_features)}):\\n')",
            "for i, f in enumerate(lr_features, 1):",
            "    print(f'  {i:2d}. {f}')",
            "",
            "# Inspect pipeline instances",
            "pipe_standard = build_logistic_regression_pipeline(scaler_type='standard')",
            "pipe_robust = build_logistic_regression_pipeline(scaler_type='robust')",
            "print('\\nStandard Pipeline:', pipe_standard)",
            "print('Robust Pipeline:  ', pipe_robust)"
        ]),
        make_markdown_cell([
            "### 2. Walk-Forward Validation Results (10 Folds)",
            "",
            "We load the out-of-fold validation metrics and out-of-fold predictions generated by the walk-forward evaluation engine.",
            "Metrics are evaluated at both the standard threshold (0.50) and the inner-calibrated optimal F1 threshold."
        ]),
        make_code_cell([
            "# Load Batch 4 walk-forward fold metrics",
            "metrics_path = REPO_ROOT / 'data' / 'interim' / 'phase8_batch4' / 'supervised_fold_metrics.csv'",
            "fold_metrics = pd.read_csv(metrics_path)",
            "",
            "# Filter for baseline models",
            "lr_metrics = fold_metrics[fold_metrics['model_name'].isin(['logistic_regression_standard', 'logistic_regression_robust'])]",
            "display_cols = [",
            "    'model_name', 'fold', 'val_start', 'val_end', 'val_rows', 'val_prev',",
            "    'pr_auc', 'roc_auc', 'brier_score', 'f1_50', 'calibrated_threshold', 'f1_calib'",
            "]",
            "lr_metrics[display_cols].sort_values(['model_name', 'fold'])"
        ]),
        make_markdown_cell([
            "### 3. Scaler Comparison: StandardScaler vs RobustScaler",
            "",
            "Compare out-of-fold PR-AUC, ROC-AUC, Brier score, and F1 between StandardScaler and RobustScaler across all 10 folds."
        ]),
        make_code_cell([
            "# Aggregated summary by scaler",
            "comparison = lr_metrics.groupby('model_name').agg({",
            "    'pr_auc': ['mean', 'std', 'min', 'max'],",
            "    'roc_auc': ['mean', 'std'],",
            "    'brier_score': ['mean'],",
            "    'f1_50': ['mean'],",
            "    'f1_calib': ['mean'],",
            "    'calibrated_threshold': ['mean'],",
            "    'fit_time_sec': ['sum', 'mean']",
            "}).round(4)",
            "comparison"
        ]),
        make_markdown_cell([
            "### 4. Market Capitalization Stratification",
            "",
            "Evaluate performance across Large Cap, Mid Cap, and Small Cap segments on pooled out-of-fold predictions."
        ]),
        make_code_cell([
            "# Load out-of-fold predictions",
            "preds_path = REPO_ROOT / 'data' / 'interim' / 'phase8_batch4' / 'supervised_predictions.parquet'",
            "preds_df = pd.read_parquet(preds_path)",
            "",
            "for model in ['logistic_regression_standard', 'logistic_regression_robust']:",
            "    print(f'=== Market Cap Stratification: {model} ===')",
            "    sub_df = preds_df.rename(columns={'y_true': 'hre_5d_train', f'prob_{model}': 'hre_prob'})",
            "    cap_eval = ModelEvaluator.evaluate_by_cap_segment(",
            "        sub_df, segment_col='cap_segment', target_col='hre_5d_train', prob_col='hre_prob', threshold=0.50",
            "    )",
            "    print(cap_eval.to_string(index=False))",
            "    print()"
        ]),
        make_markdown_cell([
            "### 5. Directional Coefficient Inspection",
            "",
            "Verify the sign and magnitude of logistic regression coefficients to ensure economic validity.",
            "Higher illiquidity (e.g., `amihud_z20`) and higher volatility (`vol_20d`) should carry positive coefficients."
        ]),
        make_code_cell([
            "# Fit demonstration model on training split to inspect coefficients",
            "data_path = REPO_ROOT / 'data' / 'processed' / 'features_with_targets.parquet'",
            "df_full = pd.read_parquet(data_path)",
            "train_mask = (df_full['split'] == 'train') & (~df_full['is_warmup']) & (df_full['hre_5d_train'].notna())",
            "",
            "X_demo = df_full.loc[train_mask, lr_features]",
            "y_demo = df_full.loc[train_mask, 'hre_5d_train']",
            "",
            "pipe_standard.fit(X_demo, y_demo)",
            "coefs = pd.DataFrame({",
            "    'feature': lr_features,",
            "    'coefficient': pipe_standard.named_steps['classifier'].coef_[0]",
            "}).sort_values('coefficient', ascending=False)",
            "",
            "print('Logistic Regression Feature Coefficients (StandardScaler):')",
            "print(coefs.to_string(index=False))"
        ])
    ]
    return make_notebook(cells)


def build_notebook_06():
    cells = [
        make_markdown_cell([
            "# Notebook 06: Random Forest Model & Feature Importance",
            "## ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities",
            "",
            "**Phase:** Phase 8 Batch 4 (Stages 8.9–8.11)",
            "**Target:** High-Risk Event (`hre_5d_train`, 5-day forward horizon)",
            "**Evaluation Protocol:** 10-Fold Expanding-Window Walk-Forward Validation with 5-Session Purging",
            "",
            "### Objectives",
            "1. Train Random Forest Classifier across 10 expanding-window walk-forward folds.",
            "2. Compare **Candidate A (35 features, baseline)** vs. **Candidate B (36 features, including F51 `relative_volume`)**.",
            "3. Leverage scikit-learn 1.8.0 native missing-value routing during tree splitting (no imputation required).",
            "4. Analyze the empirical impact of F51 (15.5% missingness) on validation PR-AUC, stability, and calibration.",
            "5. Extract Gini feature importances to identify dominant liquidity and risk drivers."
        ]),
        make_code_cell([
            "# Imports and Environment Setup",
            "import os",
            "import sys",
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            "# Add repository root to path regardless of working directory",
            "current_dir = Path('.').resolve()",
            "if (current_dir / 'data' / 'processed').exists():",
            "    REPO_ROOT = current_dir",
            "elif (current_dir.parent / 'data' / 'processed').exists():",
            "    REPO_ROOT = current_dir.parent",
            "else:",
            "    REPO_ROOT = current_dir",
            "    for p in [current_dir] + list(current_dir.parents):",
            "        if (p / 'data' / 'processed').exists():",
            "            REPO_ROOT = p",
            "            break",
            "",
            "if str(REPO_ROOT) not in sys.path:",
            "    sys.path.insert(0, str(REPO_ROOT))",
            "",
            "from src.modeling.preprocessing import build_random_forest_pipeline, get_model_features",
            "from src.modeling.evaluation import ModelEvaluator",
            "",
            "print(f'Repository root: {REPO_ROOT}')",
            "print('Environment initialized successfully.')"
        ]),
        make_markdown_cell([
            "### 1. Candidate Feature Sets & Native NaN Handling",
            "",
            "- **Candidate A (35 features):** Approved baseline feature set excluding F51 `relative_volume`.",
            "- **Candidate B (36 features):** Includes F51 `relative_volume` to test if tree splits exploit volume shocks despite 15.5% NaN.",
            "- Both configurations use `class_weight='balanced_subsample'`, `max_depth=8`, `min_samples_leaf=50`, `n_estimators=150`."
        ]),
        make_code_cell([
            "features_a = get_model_features('random_forest', candidate='A')",
            "features_b = get_model_features('random_forest', candidate='B')",
            "",
            "print(f'Candidate A Feature Count: {len(features_a)}')",
            "print(f'Candidate B Feature Count: {len(features_b)}')",
            "diff = set(features_b) - set(features_a)",
            "print(f'Difference (Candidate B unique): {diff}')",
            "",
            "pipe_rf_a = build_random_forest_pipeline(candidate='A')",
            "print('\\nPipeline structure (no imputer step; native NaN handling):', pipe_rf_a)"
        ]),
        make_markdown_cell([
            "### 2. Walk-Forward Validation: Candidate A vs. Candidate B",
            "",
            "Load the 10-fold walk-forward validation results and compare Candidate A vs. Candidate B fold-by-fold."
        ]),
        make_code_cell([
            "metrics_path = REPO_ROOT / 'data' / 'interim' / 'phase8_batch4' / 'supervised_fold_metrics.csv'",
            "fold_metrics = pd.read_csv(metrics_path)",
            "",
            "rf_metrics = fold_metrics[fold_metrics['model_name'].isin(['random_forest_candidate_a', 'random_forest_candidate_b'])]",
            "display_cols = [",
            "    'model_name', 'fold', 'val_start', 'val_end', 'val_rows',",
            "    'pr_auc', 'roc_auc', 'brier_score', 'f1_50', 'calibrated_threshold', 'f1_calib'",
            "]",
            "rf_metrics[display_cols].sort_values(['model_name', 'fold'])"
        ]),
        make_markdown_cell([
            "### 3. Aggregated Performance Comparison & F51 Impact",
            "",
            "Compare mean and standard deviation of PR-AUC, ROC-AUC, Brier score, and calibrated F1."
        ]),
        make_code_cell([
            "rf_summary = rf_metrics.groupby('model_name').agg({",
            "    'pr_auc': ['mean', 'std', 'min', 'max'],",
            "    'roc_auc': ['mean', 'std'],",
            "    'brier_score': ['mean'],",
            "    'f1_50': ['mean'],",
            "    'f1_calib': ['mean'],",
            "    'calibrated_threshold': ['mean'],",
            "    'fit_time_sec': ['sum', 'mean']",
            "}).round(4)",
            "rf_summary"
        ]),
        make_markdown_cell([
            "### 4. Market Capitalization Performance",
            "",
            "Evaluate out-of-fold performance across large, mid, and small market cap segments."
        ]),
        make_code_cell([
            "preds_path = REPO_ROOT / 'data' / 'interim' / 'phase8_batch4' / 'supervised_predictions.parquet'",
            "preds_df = pd.read_parquet(preds_path)",
            "",
            "for model in ['random_forest_candidate_a', 'random_forest_candidate_b']:",
            "    print(f'=== Market Cap Stratification: {model} ===')",
            "    sub_df = preds_df.rename(columns={'y_true': 'hre_5d_train', f'prob_{model}': 'hre_prob'})",
            "    cap_eval = ModelEvaluator.evaluate_by_cap_segment(",
            "        sub_df, segment_col='cap_segment', target_col='hre_5d_train', prob_col='hre_prob', threshold=0.50",
            "    )",
            "    print(cap_eval.to_string(index=False))",
            "    print()"
        ]),
        make_markdown_cell([
            "### 5. Random Forest Gini Feature Importances",
            "",
            "Extract Gini feature importances from a training fit of Candidate B to inspect feature rankings."
        ]),
        make_code_cell([
            "data_path = REPO_ROOT / 'data' / 'processed' / 'features_with_targets.parquet'",
            "df_full = pd.read_parquet(data_path)",
            "train_mask = (df_full['split'] == 'train') & (~df_full['is_warmup']) & (df_full['hre_5d_train'].notna())",
            "",
            "X_rf = df_full.loc[train_mask, features_b]",
            "y_rf = df_full.loc[train_mask, 'hre_5d_train']",
            "",
            "rf_pipe = build_random_forest_pipeline(candidate='B')",
            "rf_pipe.fit(X_rf, y_rf)",
            "",
            "rf_clf = rf_pipe.named_steps['classifier']",
            "importances = pd.DataFrame({",
            "    'feature': features_b,",
            "    'gini_importance': rf_clf.feature_importances_",
            "}).sort_values('gini_importance', ascending=False)",
            "",
            "print('Top 15 Most Important Features in Random Forest:')",
            "print(importances.head(15).to_string(index=False))"
        ])
    ]
    return make_notebook(cells)


def build_notebook_07():
    cells = [
        make_markdown_cell([
            "# Notebook 07: HistGradientBoosting Model",
            "## ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities",
            "",
            "**Phase:** Phase 8 Batch 4 (Stages 8.9–8.11)",
            "**Target:** High-Risk Event (`hre_5d_train`, 5-day forward horizon)",
            "**Evaluation Protocol:** 10-Fold Expanding-Window Walk-Forward Validation with 5-Session Purging",
            "",
            "### Objectives",
            "1. Train `HistGradientBoostingClassifier` across 10 expanding-window walk-forward folds.",
            "2. Utilize all 36 approved candidate features (including F51 `relative_volume`).",
            "3. Leverage native histogram binning for high computational speed and low memory footprint (<500 MB).",
            "4. Exploit native NaN routing during tree splitting.",
            "5. Compare out-of-fold performance (PR-AUC, ROC-AUC, Brier score, calibration) against Logistic Regression and Random Forest."
        ]),
        make_code_cell([
            "# Imports and Environment Setup",
            "import os",
            "import sys",
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            "# Add repository root to path regardless of working directory",
            "current_dir = Path('.').resolve()",
            "if (current_dir / 'data' / 'processed').exists():",
            "    REPO_ROOT = current_dir",
            "elif (current_dir.parent / 'data' / 'processed').exists():",
            "    REPO_ROOT = current_dir.parent",
            "else:",
            "    REPO_ROOT = current_dir",
            "    for p in [current_dir] + list(current_dir.parents):",
            "        if (p / 'data' / 'processed').exists():",
            "            REPO_ROOT = p",
            "            break",
            "",
            "if str(REPO_ROOT) not in sys.path:",
            "    sys.path.insert(0, str(REPO_ROOT))",
            "",
            "from src.modeling.preprocessing import build_hgbt_pipeline, get_model_features",
            "from src.modeling.evaluation import ModelEvaluator",
            "",
            "print(f'Repository root: {REPO_ROOT}')",
            "print('Environment initialized successfully.')"
        ]),
        make_markdown_cell([
            "### 1. HGBT Architecture & 36-Feature Specification",
            "",
            "HistGradientBoosting operates directly on raw feature matrices without scaling or imputation.",
            "Missing values are assigned to dedicated histogram bins during construction."
        ]),
        make_code_cell([
            "hgbt_features = get_model_features('hist_gradient_boosting')",
            "print(f'HGBT Features ({len(hgbt_features)}):\\n')",
            "print(', '.join(hgbt_features))",
            "",
            "pipe_hgbt = build_hgbt_pipeline()",
            "print('\\nHGBT Pipeline:', pipe_hgbt)"
        ]),
        make_markdown_cell([
            "### 2. Walk-Forward Validation Across 10 Folds",
            "",
            "Load the 10-fold walk-forward validation results for HistGradientBoosting."
        ]),
        make_code_cell([
            "metrics_path = REPO_ROOT / 'data' / 'interim' / 'phase8_batch4' / 'supervised_fold_metrics.csv'",
            "fold_metrics = pd.read_csv(metrics_path)",
            "",
            "hgbt_metrics = fold_metrics[fold_metrics['model_name'] == 'hist_gradient_boosting']",
            "display_cols = [",
            "    'fold', 'val_start', 'val_end', 'val_rows', 'val_prev',",
            "    'pr_auc', 'roc_auc', 'brier_score', 'f1_50', 'calibrated_threshold', 'f1_calib', 'fit_time_sec'",
            "]",
            "hgbt_metrics[display_cols].sort_values('fold')"
        ]),
        make_markdown_cell([
            "### 3. Comprehensive Model Comparison Leaderboard",
            "",
            "Compare all 5 supervised configurations ranked by Mean Out-of-Fold PR-AUC."
        ]),
        make_code_cell([
            "comp_path = REPO_ROOT / 'data' / 'interim' / 'phase8_batch4' / 'supervised_model_comparison.csv'",
            "comp_df = pd.read_csv(comp_path)",
            "",
            "leaderboard_cols = [",
            "    'model_name', 'mean_pr_auc', 'std_pr_auc', 'min_pr_auc', 'pooled_pr_auc',",
            "    'mean_roc_auc', 'mean_brier_score', 'mean_f1_50', 'mean_f1_calib',",
            "    'mean_calib_threshold', 'total_fit_time_sec'",
            "]",
            "comp_df[leaderboard_cols]"
        ]),
        make_markdown_cell([
            "### 4. Market Capitalization Stratification",
            "",
            "Evaluate HGBT performance across Large Cap, Mid Cap, and Small Cap segments."
        ]),
        make_code_cell([
            "preds_path = REPO_ROOT / 'data' / 'interim' / 'phase8_batch4' / 'supervised_predictions.parquet'",
            "preds_df = pd.read_parquet(preds_path)",
            "",
            "sub_df = preds_df.rename(columns={'y_true': 'hre_5d_train', 'prob_hist_gradient_boosting': 'hre_prob'})",
            "cap_eval = ModelEvaluator.evaluate_by_cap_segment(",
            "    sub_df, segment_col='cap_segment', target_col='hre_5d_train', prob_col='hre_prob', threshold=0.50",
            ")",
            "print('HGBT Market Cap Stratification:')",
            "print(cap_eval.to_string(index=False))"
        ])
    ]
    return make_notebook(cells)


def build_notebook_08():
    cells = [
        make_markdown_cell([
            "# Notebook 08: Unsupervised Anomaly Detection (Isolation Forest)",
            "## ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities",
            "",
            "**Phase:** Phase 8 Batch 4 (Stages 8.9–8.11)",
            "**Paradigm:** Unsupervised Anomaly Detection (Diagnostic Unusualness)",
            "**Evaluation Protocol:** 10-Fold Expanding-Window Walk-Forward Validation",
            "",
            "### Objectives",
            "1. Fit `IsolationForest` on historical feature space using the selected 12-feature liquidity/risk subset.",
            "2. Enforce strict architectural separation: Isolation Forest measures **historical distributional unusualness**, NOT forward risk.",
            "3. Prevent lookahead leakage: fit strictly on post-warmup training observations up to the fold boundary.",
            "4. Map raw decision function scores to rescaled 0–100 anomaly scores for dashboard consumption.",
            "5. Cross-tabulate out-of-sample anomaly flags with forward High-Risk Events (`hre_5d_train`) to measure empirical overlap."
        ]),
        make_code_cell([
            "# Imports and Environment Setup",
            "import os",
            "import sys",
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            "# Add repository root to path regardless of working directory",
            "current_dir = Path('.').resolve()",
            "if (current_dir / 'data' / 'processed').exists():",
            "    REPO_ROOT = current_dir",
            "elif (current_dir.parent / 'data' / 'processed').exists():",
            "    REPO_ROOT = current_dir.parent",
            "else:",
            "    REPO_ROOT = current_dir",
            "    for p in [current_dir] + list(current_dir.parents):",
            "        if (p / 'data' / 'processed').exists():",
            "            REPO_ROOT = p",
            "            break",
            "",
            "if str(REPO_ROOT) not in sys.path:",
            "    sys.path.insert(0, str(REPO_ROOT))",
            "",
            "from src.modeling.anomaly import AnomalyDetector",
            "from src.modeling.preprocessing import get_model_features",
            "",
            "print(f'Repository root: {REPO_ROOT}')",
            "print('Environment initialized successfully.')"
        ]),
        make_markdown_cell([
            "### 1. Feature Subset & Architecture",
            "",
            "The Isolation Forest operates on a curated 12-feature subset capturing volume shocks, spread widening, and volatility bursts.",
            "It is completely target-independent and includes observations even when forward labels are missing (e.g. trading halts)."
        ]),
        make_code_cell([
            "if_features = get_model_features('isolation_forest')",
            "print(f'Isolation Forest Features ({len(if_features)}):\\n')",
            "for i, f in enumerate(if_features, 1):",
            "    print(f'  {i:2d}. {f}')",
            "",
            "detector = AnomalyDetector(n_estimators=100, contamination=0.05, random_state=42)",
            "print('\\nAnomaly Detector configuration:', detector)"
        ]),
        make_markdown_cell([
            "### 2. Walk-Forward Out-of-Sample Anomaly Score Distributions",
            "",
            "Inspect anomaly score metrics and flagging rates across all 10 walk-forward folds."
        ]),
        make_code_cell([
            "metrics_path = REPO_ROOT / 'data' / 'interim' / 'phase8_batch4' / 'isolation_forest_fold_metrics.csv'",
            "if_metrics = pd.read_csv(metrics_path)",
            "",
            "display_cols = [",
            "    'fold', 'val_start', 'val_end', 'val_rows', 'mean_scaled_score',",
            "    'median_scaled_score', 'p90_scaled_score', 'p95_scaled_score',",
            "    'flagged_count', 'flagged_rate', 'hre_pos_flag_rate', 'anomaly_target_corr'",
            "]",
            "if_metrics[display_cols]"
        ]),
        make_markdown_cell([
            "### 3. Diagnostic Overlap with High-Risk Events (HRE)",
            "",
            "Measure how frequently the unsupervised anomaly detector flags forward high-risk events.",
            "Because the Isolation Forest is purely contemporaneous and unsupervised, empirical overlap confirms that liquidity stress manifests as distributional outliers."
        ]),
        make_code_cell([
            "print('=== Isolation Forest Summary Across 10 Folds ===')",
            "print(f'Mean Scaled Score:    {if_metrics[\"mean_scaled_score\"].mean():.2f} / 100')",
            "print(f'Mean Flagged Rate:    {if_metrics[\"flagged_rate\"].mean() * 100:.2f}% (Expected: ~5.00%)')",
            "print(f'Mean HRE+ Flag Rate:  {if_metrics[\"hre_pos_flag_rate\"].mean() * 100:.2f}%')",
            "print(f'Mean HRE- Flag Rate:  {if_metrics[\"hre_neg_flag_rate\"].mean() * 100:.2f}%')",
            "print(f'Mean Correlation:     {if_metrics[\"anomaly_target_corr\"].mean():+.4f}')"
        ]),
        make_markdown_cell([
            "### 4. Segment-Level Anomaly Prevalence",
            "",
            "Evaluate anomaly scores and flag rates across Market Cap segments (Large, Mid, Small Cap)."
        ]),
        make_code_cell([
            "preds_path = REPO_ROOT / 'data' / 'interim' / 'phase8_batch4' / 'isolation_forest_predictions.parquet'",
            "if_preds = pd.read_parquet(preds_path)",
            "",
            "cap_summary = if_preds.groupby('cap_segment').agg({",
            "    'anomaly_score_scaled': ['count', 'mean', 'median', 'std'],",
            "    'isolation_flag': ['mean', 'sum'],",
            "    'y_true': ['mean']",
            "}).round(4)",
            "cap_summary"
        ])
    ]
    return make_notebook(cells)


def main():
    print("Generating updated notebooks...")
    notebooks = {
        "05_model_baseline.ipynb": build_notebook_05(),
        "06_model_rf.ipynb": build_notebook_06(),
        "07_model_hgbt.ipynb": build_notebook_07(),
        "08_anomaly_detection.ipynb": build_notebook_08(),
    }

    for filename, nb_dict in notebooks.items():
        out_path = NOTEBOOKS_DIR / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(nb_dict, f, indent=1)
        print(f"Updated: {out_path.resolve()} ({len(nb_dict['cells'])} cells)")


if __name__ == "__main__":
    main()

