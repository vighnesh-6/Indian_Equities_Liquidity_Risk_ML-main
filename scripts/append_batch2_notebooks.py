"""Appends Stage 8.12.5, 8.12.6, and 8.12.7 sections to the four modeling notebooks.

CRITICAL NON-NEGOTIABLE COMPLIANCE:
- ZERO pre-existing cells are executed.
- All pre-existing cells, outputs, execution counts, and metadata are 100% preserved.
- Appends self-contained new sections.
- Executes ONLY newly appended code cells in isolation.
"""

from pathlib import Path
import io
import sys
import contextlib
import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell, new_output

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NOTEBOOK_BATCH2_SECTIONS = {
    "notebooks/05_model_baseline.ipynb": [
        {
            "section_title": "### 7. Phase 8 Batch 5 (Stage 8.12.5): Supervised Holdout Evaluation — Logistic Regression Baseline",
            "markdown": (
                "### 7. Phase 8 Batch 5 (Stage 8.12.5): Supervised Holdout Evaluation — Logistic Regression Baseline\n\n"
                "In Stage 8.12.5, the pre-trained Logistic Regression baseline models (`logistic_regression_standard` and "
                "`logistic_regression_robust`) were evaluated on the 250-trading-session holdout (`2025-10-20` to `2026-10-23`, "
                "74,334 total rows, 71,756 valid `hre_5d` rows) strictly using already-fitted pipelines without refitting.\n\n"
                "- Evaluated at standard fixed threshold $\\tau = 0.50$.\n"
                "- Evaluated at frozen historical calibrated thresholds (StandardScaler $\\tau^* = 0.460$, RobustScaler $\\tau^* = 0.458$)."
            ),
            "code": (
                "# Stage 8.12.5: Inspect Supervised Holdout Evaluation for Baseline Logistic Regression\n"
                "from pathlib import Path\n"
                "import pandas as pd\n\n"
                "metrics_path = REPO_ROOT / 'data' / 'interim' / 'phase8_batch5' / 'holdout' / 'supervised_holdout_metrics.csv'\n"
                "holdout_metrics = pd.read_csv(metrics_path)\n"
                "lr_holdout = holdout_metrics[holdout_metrics['model_name'].str.contains('logistic_regression')].copy()\n\n"
                "display_cols = [\n"
                "    'model_name', 'holdout_pr_auc', 'holdout_roc_auc', 'holdout_brier_score',\n"
                "    'calibrated_threshold', 'f1_50', 'precision_50', 'recall_50',\n"
                "    'f1_calib', 'precision_calib', 'recall_calib', 'evaluated_rows'\n"
                "]\n"
                "print('=' * 80)\n"
                "print('STAGE 8.12.5: BASELINE LOGISTIC REGRESSION HOLDOUT PERFORMANCE')\n"
                "print('=' * 80)\n"
                "print(lr_holdout[display_cols].to_string(index=False))\n"
            ),
        },
        {
            "section_title": "### 8. Phase 8 Batch 5 (Stage 8.12.7): Walk-Forward vs Holdout Performance Comparison",
            "markdown": (
                "### 8. Phase 8 Batch 5 (Stage 8.12.7): Walk-Forward vs Holdout Performance Comparison\n\n"
                "Comprehensive comparison between 10-fold walk-forward cross-validation performance and final holdout "
                "performance across all five supervised candidate configurations.\n\n"
                "- Compares Mean Fold PR-AUC, Pooled PR-AUC, ROC-AUC, Brier score, and F1 scores.\n"
                "- Assesses model ranking preservation, out-of-sample generalization retention, and calibration stability."
            ),
            "code": (
                "# Stage 8.12.7: Walk-Forward vs Holdout Performance Comparison Across All Supervised Models\n"
                "from pathlib import Path\n"
                "import pandas as pd\n\n"
                "comp_path = REPO_ROOT / 'data' / 'interim' / 'phase8_batch5' / 'holdout' / 'walkforward_vs_holdout_comparison.csv'\n"
                "wf_vs_holdout = pd.read_csv(comp_path)\n\n"
                "summary_cols = [\n"
                "    'model_name', 'wf_rank', 'holdout_rank', 'wf_mean_pr_auc', 'holdout_pr_auc',\n"
                "    'pr_auc_delta_vs_mean', 'pr_auc_retention_pct', 'wf_mean_roc_auc', 'holdout_roc_auc',\n"
                "    'wf_brier_score', 'holdout_brier_score', 'wf_f1_50', 'holdout_f1_50'\n"
                "]\n"
                "print('=' * 80)\n"
                "print('STAGE 8.12.7: COMPREHENSIVE WALK-FORWARD VS HOLDOUT COMPARISON LEADERBOARD')\n"
                "print('=' * 80)\n"
                "print(wf_vs_holdout[summary_cols].to_string(index=False))\n"
            ),
        },
    ],
    "notebooks/06_model_rf.ipynb": [
        {
            "section_title": "### 7. Phase 8 Batch 5 (Stage 8.12.5): Supervised Holdout Evaluation — Random Forest Candidates A & B",
            "markdown": (
                "### 7. Phase 8 Batch 5 (Stage 8.12.5): Supervised Holdout Evaluation — Random Forest Candidates A & B\n\n"
                "In Stage 8.12.5, both Random Forest candidate configurations were evaluated on the 250-trading-session holdout "
                "(`2025-10-20` to `2026-10-23`, 71,756 valid rows):\n\n"
                "- **Candidate A (35 features):** Baseline tree ensemble excluding F51 `relative_volume`.\n"
                "- **Candidate B (36 features):** Experimental tree ensemble incorporating F51 `relative_volume` with native NaN routing.\n"
                "- Evaluates the out-of-sample impact of volume-shock feature F51 on unseen future data."
            ),
            "code": (
                "# Stage 8.12.5: Inspect Supervised Holdout Evaluation for Random Forest Candidates A & B\n"
                "from pathlib import Path\n"
                "import pandas as pd\n\n"
                "metrics_path = REPO_ROOT / 'data' / 'interim' / 'phase8_batch5' / 'holdout' / 'supervised_holdout_metrics.csv'\n"
                "holdout_metrics = pd.read_csv(metrics_path)\n"
                "rf_holdout = holdout_metrics[holdout_metrics['model_name'].str.contains('random_forest')].copy()\n\n"
                "display_cols = [\n"
                "    'model_name', 'holdout_pr_auc', 'holdout_roc_auc', 'holdout_brier_score',\n"
                "    'calibrated_threshold', 'f1_50', 'precision_50', 'recall_50',\n"
                "    'f1_calib', 'precision_calib', 'recall_calib', 'evaluated_rows'\n"
                "]\n"
                "print('=' * 80)\n"
                "print('STAGE 8.12.5: RANDOM FOREST CANDIDATES A & B HOLDOUT PERFORMANCE')\n"
                "print('=' * 80)\n"
                "print(rf_holdout[display_cols].to_string(index=False))\n\n"
                "f51_gain = float(rf_holdout.loc[rf_holdout['model_name'] == 'random_forest_candidate_b', 'holdout_pr_auc'].values[0] - \\\n"
                "                 rf_holdout.loc[rf_holdout['model_name'] == 'random_forest_candidate_a', 'holdout_pr_auc'].values[0])\n"
                "print(f'\\nOut-of-sample F51 relative_volume PR-AUC Gain on Holdout: {f51_gain:+.4f} ({f51_gain*100:+.2f} percentage points)')\n"
            ),
        },
    ],
    "notebooks/07_model_hgbt.ipynb": [
        {
            "section_title": "### 6. Phase 8 Batch 5 (Stage 8.12.5): Supervised Holdout Evaluation — Primary Champion (HistGradientBoosting)",
            "markdown": (
                "### 6. Phase 8 Batch 5 (Stage 8.12.5): Supervised Holdout Evaluation — Primary Champion (HistGradientBoosting)\n\n"
                "In Stage 8.12.5, the Primary Supervised Champion (`hist_gradient_boosting`, 36 features) was evaluated on the "
                "250-trading-session holdout (`2025-10-20` to `2026-10-23`, 71,756 valid rows):\n\n"
                "- Evaluates out-of-sample discrimination (PR-AUC, ROC-AUC) and calibration (Brier score, Log Loss).\n"
                "- Evaluates market-cap segment performance across Large-Cap, Mid-Cap, and Small-Cap universes.\n"
                "- Confirms whether HGBT maintains its decisive superiority over all competitor models on unseen data."
            ),
            "code": (
                "# Stage 8.12.5: Inspect Supervised Holdout Evaluation for Primary Champion HGBT\n"
                "from pathlib import Path\n"
                "import pandas as pd\n\n"
                "metrics_path = REPO_ROOT / 'data' / 'interim' / 'phase8_batch5' / 'holdout' / 'supervised_holdout_metrics.csv'\n"
                "holdout_metrics = pd.read_csv(metrics_path)\n"
                "hgbt_holdout = holdout_metrics[holdout_metrics['model_name'] == 'hist_gradient_boosting'].copy()\n\n"
                "display_cols = [\n"
                "    'model_name', 'holdout_pr_auc', 'holdout_roc_auc', 'holdout_brier_score',\n"
                "    'calibrated_threshold', 'f1_50', 'precision_50', 'recall_50',\n"
                "    'f1_calib', 'precision_calib', 'recall_calib',\n"
                "    'large_cap_pr_auc', 'mid_cap_pr_auc', 'small_cap_pr_auc'\n"
                "]\n"
                "print('=' * 80)\n"
                "print('STAGE 8.12.5: HISTGRADIENTBOOSTING CHAMPION HOLDOUT PERFORMANCE')\n"
                "print('=' * 80)\n"
                "print(hgbt_holdout[display_cols].to_string(index=False))\n"
            ),
        },
    ],
    "notebooks/08_anomaly_detection.ipynb": [
        {
            "section_title": "### 6. Phase 8 Batch 5 (Stages 8.12.6 & 8.12.7): Isolation Forest Holdout Evaluation & Comparison",
            "markdown": (
                "### 6. Phase 8 Batch 5 (Stages 8.12.6 & 8.12.7): Isolation Forest Holdout Evaluation & Comparison\n\n"
                "In Stage 8.12.6 and 8.12.7, the pre-trained unsupervised Isolation Forest model (12 features) was evaluated "
                "on the 250-trading-session holdout (`2025-10-20` to `2026-10-23`, 74,334 total rows):\n\n"
                "- Evaluates the holdout anomaly score distribution, tail percentiles (P90, P95, P99), and flag rate.\n"
                "- Assesses the empirical relationship and enrichment ratio between anomaly flags and realized forward HRE events.\n"
                "- Compares walk-forward telemetry characteristics against holdout telemetry characteristics."
            ),
            "code": (
                "# Stages 8.12.6 & 8.12.7: Isolation Forest Holdout Evaluation & Walk-Forward Comparison\n"
                "from pathlib import Path\n"
                "import pandas as pd\n\n"
                "holdout_if_path = REPO_ROOT / 'data' / 'interim' / 'phase8_batch5' / 'holdout' / 'isolation_forest_holdout_metrics.csv'\n"
                "comp_if_path = REPO_ROOT / 'data' / 'interim' / 'phase8_batch5' / 'holdout' / 'isolation_forest_walkforward_vs_holdout.csv'\n\n"
                "if_holdout = pd.read_csv(holdout_if_path)\n"
                "if_comp = pd.read_csv(comp_if_path)\n\n"
                "print('=' * 80)\n"
                "print('STAGE 8.12.6: ISOLATION FOREST HOLDOUT METRICS')\n"
                "print('=' * 80)\n"
                "print(if_holdout.to_string(index=False))\n\n"
                "print('\\n' + '=' * 80)\n"
                "print('STAGE 8.12.7: ISOLATION FOREST WALK-FORWARD VS HOLDOUT COMPARISON')\n"
                "print('=' * 80)\n"
                "print(if_comp.to_string(index=False))\n"
            ),
        },
    ],
}


def main():
    print("Appending and executing Batch 2 sections in notebooks...")

    for rel_path, sections in NOTEBOOK_BATCH2_SECTIONS.items():
        nb_path = REPO_ROOT / rel_path
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        for section in sections:
            has_section = any(
                section["section_title"] in cell.get("source", "")
                for cell in nb.cells
            )

            if has_section:
                print(f"  {rel_path}: '{section['section_title'][:40]}' already present. Skipping.")
                continue

            code_str = section["code"]
            stdout_buf = io.StringIO()
            exec_scope = {"REPO_ROOT": REPO_ROOT}

            with contextlib.redirect_stdout(stdout_buf):
                exec(code_str, exec_scope)

            captured_text = stdout_buf.getvalue()

            max_exec = max(
                (cell.get("execution_count") or 0 for cell in nb.cells if cell.cell_type == "code"),
                default=0,
            )
            new_exec_count = max_exec + 1

            md_cell = new_markdown_cell(section["markdown"])
            out_stream = new_output("stream", name="stdout", text=captured_text)
            cd_cell = new_code_cell(code_str, execution_count=new_exec_count, outputs=[out_stream])

            nb.cells.extend([md_cell, cd_cell])
            print(f"  {rel_path}: Appended and executed '{section['section_title'][:40]}...' (cells {len(nb.cells)-2} & {len(nb.cells)-1})")

        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

    print("\nAll four notebooks successfully updated with Batch 2 sections!")


if __name__ == "__main__":
    main()
