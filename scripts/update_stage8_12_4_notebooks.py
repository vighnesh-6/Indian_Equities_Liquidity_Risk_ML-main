"""Updates notebooks for Stage 8.12.4 without re-running walk-forward validation.

Preserves all historical Batch 4 cells and outputs intact.
Appends a clearly distinguished Stage 8.12.4 pre-holdout training verification section.
Executes only the newly appended cell to capture authoritative outputs.
"""

from pathlib import Path
import json
import io
import sys
import contextlib
import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell, new_output
import pandas as pd
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NOTEBOOK_UPDATES = {
    "notebooks/05_model_baseline.ipynb": {
        "section_title": "### 6. Phase 8 Batch 5 (Stage 8.12.4): Pre-Holdout Full-Population Training",
        "markdown": (
            "### 6. Phase 8 Batch 5 (Stage 8.12.4): Pre-Holdout Full-Population Training\n\n"
            "In accordance with the Stage 8.12.3 configuration freeze and Stage 8.12.4 pre-holdout training protocol:\n"
            "- All historical walk-forward validation experiments above are preserved and strictly **not rerun**.\n"
            "- Both Logistic Regression baseline configurations (`logistic_regression_standard` and `logistic_regression_robust`) "
            "were fitted on the full eligible pre-holdout training population (`2018-04-30` to `2025-10-10`, 466,346 rows, 1,841 sessions).\n"
            "- The 5-session boundary embargo (`2025-10-13` to `2025-10-17`, 1,444 rows) and 250-session holdout "
            "(`2025-10-20` to `2026-10-23`, 74,334 rows) were strictly excluded with zero access."
        ),
        "code": (
            "# Phase 8 Batch 5 (Stage 8.12.4): Inspect Pre-Holdout Baseline Model Artifacts\n"
            "import json\n"
            "from pathlib import Path\n"
            "import joblib\n"
            "import pandas as pd\n\n"
            "models_dir = REPO_ROOT / 'models' / 'phase8_batch5'\n"
            "meta_dir = REPO_ROOT / 'reports' / 'phase8_batch5' / 'training_metadata'\n\n"
            "baseline_records = []\n"
            "for m_name in ['logistic_regression_standard', 'logistic_regression_robust']:\n"
            "    meta_file = meta_dir / f'{m_name}_training_metadata.json'\n"
            "    with open(meta_file, 'r', encoding='utf-8') as f:\n"
            "        meta = json.load(f)\n"
            "    art_path = models_dir / m_name / 'model.joblib'\n"
            "    pipeline = joblib.load(art_path)\n"
            "    baseline_records.append({\n"
            "        'Model Identifier': meta['model_identifier'],\n"
            "        'Algorithm': meta['algorithm'],\n"
            "        'Feature Count': meta['feature_count'],\n"
            "        'Training Start': meta['training_start_date'],\n"
            "        'Training End': meta['training_end_date'],\n"
            "        'Eligible Rows': f\"{meta['eligible_training_rows']:,}\",\n"
            "        'Pos Class (HRE=1)': f\"{meta['positive_class_count']:,} ({meta['positive_class_prevalence']:.2%})\",\n"
            "        'Fit Time (s)': f\"{meta['fit_time_seconds']:.2f}s\",\n"
            "        'Artifact Path': str(art_path.relative_to(REPO_ROOT)),\n"
            "        'Status': meta['training_completion_status'],\n"
            "    })\n\n"
            "print('=' * 80)\n"
            "print('STAGE 8.12.4: PRE-HOLDOUT LOGISTIC REGRESSION BASELINE TRAINING SUMMARY')\n"
            "print('=' * 80)\n"
            "baseline_df = pd.DataFrame(baseline_records)\n"
            "print(baseline_df.to_string(index=False))\n"
        ),
    },
    "notebooks/06_model_rf.ipynb": {
        "section_title": "### 6. Phase 8 Batch 5 (Stage 8.12.4): Pre-Holdout Full-Population Training",
        "markdown": (
            "### 6. Phase 8 Batch 5 (Stage 8.12.4): Pre-Holdout Full-Population Training\n\n"
            "In accordance with the Stage 8.12.3 configuration freeze and Stage 8.12.4 pre-holdout training protocol:\n"
            "- All historical walk-forward validation experiments above are preserved and strictly **not rerun**.\n"
            "- Both Random Forest configurations (`random_forest_candidate_a` with 35 features and `random_forest_candidate_b` "
            "with 36 features including F51 `relative_volume`) were fitted on the full eligible pre-holdout population "
            "(`2018-04-30` to `2025-10-10`, 466,346 rows, 1,841 sessions).\n"
            "- Missing values in `relative_volume` (7,047 rows) were natively handled during tree splitting without imputation.\n"
            "- The boundary embargo (1,444 rows) and final holdout (74,334 rows) were strictly excluded with zero access."
        ),
        "code": (
            "# Phase 8 Batch 5 (Stage 8.12.4): Inspect Pre-Holdout Random Forest Artifacts\n"
            "import json\n"
            "from pathlib import Path\n"
            "import joblib\n"
            "import pandas as pd\n\n"
            "models_dir = REPO_ROOT / 'models' / 'phase8_batch5'\n"
            "meta_dir = REPO_ROOT / 'reports' / 'phase8_batch5' / 'training_metadata'\n\n"
            "rf_records = []\n"
            "for m_name in ['random_forest_candidate_a', 'random_forest_candidate_b']:\n"
            "    meta_file = meta_dir / f'{m_name}_training_metadata.json'\n"
            "    with open(meta_file, 'r', encoding='utf-8') as f:\n"
            "        meta = json.load(f)\n"
            "    art_path = models_dir / m_name / 'model.joblib'\n"
            "    pipeline = joblib.load(art_path)\n"
            "    rf_records.append({\n"
            "        'Model Identifier': meta['model_identifier'],\n"
            "        'Algorithm': meta['algorithm'],\n"
            "        'Feature Count': meta['feature_count'],\n"
            "        'Includes F51': 'Yes (relative_volume)' if 'relative_volume' in meta['feature_list'] else 'No',\n"
            "        'Training Start': meta['training_start_date'],\n"
            "        'Training End': meta['training_end_date'],\n"
            "        'Eligible Rows': f\"{meta['eligible_training_rows']:,}\",\n"
            "        'Pos Class (HRE=1)': f\"{meta['positive_class_count']:,} ({meta['positive_class_prevalence']:.2%})\",\n"
            "        'Fit Time (s)': f\"{meta['fit_time_seconds']:.2f}s\",\n"
            "        'Artifact Path': str(art_path.relative_to(REPO_ROOT)),\n"
            "        'Status': meta['training_completion_status'],\n"
            "    })\n\n"
            "print('=' * 80)\n"
            "print('STAGE 8.12.4: PRE-HOLDOUT RANDOM FOREST TRAINING SUMMARY')\n"
            "print('=' * 80)\n"
            "rf_df = pd.DataFrame(rf_records)\n"
            "print(rf_df.to_string(index=False))\n"
        ),
    },
    "notebooks/07_model_hgbt.ipynb": {
        "section_title": "### 5. Phase 8 Batch 5 (Stage 8.12.4): Pre-Holdout Full-Population Training",
        "markdown": (
            "### 5. Phase 8 Batch 5 (Stage 8.12.4): Pre-Holdout Full-Population Training\n\n"
            "In accordance with the Stage 8.12.3 configuration freeze and Stage 8.12.4 pre-holdout training protocol:\n"
            "- All historical walk-forward validation experiments above are preserved and strictly **not rerun**.\n"
            "- The Primary Champion architecture (`hist_gradient_boosting`, 36 features including F51 `relative_volume`) "
            "was fitted on the full eligible pre-holdout population (`2018-04-30` to `2025-10-10`, 466,346 rows, 1,841 sessions).\n"
            "- Native integer histogram binning and missingness routing were applied without feature scaling or imputation.\n"
            "- The boundary embargo (1,444 rows) and final holdout (74,334 rows) were strictly quarantined with zero access."
        ),
        "code": (
            "# Phase 8 Batch 5 (Stage 8.12.4): Inspect Pre-Holdout Champion HGBT Artifacts\n"
            "import json\n"
            "from pathlib import Path\n"
            "import joblib\n"
            "import pandas as pd\n\n"
            "models_dir = REPO_ROOT / 'models' / 'phase8_batch5'\n"
            "meta_dir = REPO_ROOT / 'reports' / 'phase8_batch5' / 'training_metadata'\n\n"
            "m_name = 'hist_gradient_boosting'\n"
            "meta_file = meta_dir / f'{m_name}_training_metadata.json'\n"
            "with open(meta_file, 'r', encoding='utf-8') as f:\n"
            "    meta = json.load(f)\n"
            "art_path = models_dir / m_name / 'model.joblib'\n"
            "pipeline = joblib.load(art_path)\n"
            "hgbt_rec = [{\n"
            "    'Model Identifier': meta['model_identifier'],\n"
            "    'Role': 'PRIMARY SUPERVISED CHAMPION',\n"
            "    'Algorithm': meta['algorithm'],\n"
            "    'Feature Count': meta['feature_count'],\n"
            "    'Training Start': meta['training_start_date'],\n"
            "    'Training End': meta['training_end_date'],\n"
            "    'Eligible Rows': f\"{meta['eligible_training_rows']:,}\",\n"
            "    'Pos Class (HRE=1)': f\"{meta['positive_class_count']:,} ({meta['positive_class_prevalence']:.2%})\",\n"
            "    'Fit Time (s)': f\"{meta['fit_time_seconds']:.2f}s\",\n"
            "    'Artifact Path': str(art_path.relative_to(REPO_ROOT)),\n"
            "    'Status': meta['training_completion_status'],\n"
            "}]\n\n"
            "print('=' * 80)\n"
            "print('STAGE 8.12.4: PRE-HOLDOUT HISTGRADIENTBOOSTING CHAMPION SUMMARY')\n"
            "print('=' * 80)\n"
            "hgbt_df = pd.DataFrame(hgbt_rec)\n"
            "print(hgbt_df.to_string(index=False))\n"
        ),
    },
    "notebooks/08_anomaly_detection.ipynb": {
        "section_title": "### 5. Phase 8 Batch 5 (Stage 8.12.4): Pre-Holdout Full-Population Training",
        "markdown": (
            "### 5. Phase 8 Batch 5 (Stage 8.12.4): Pre-Holdout Full-Population Training\n\n"
            "In accordance with the Stage 8.12.3 configuration freeze and Stage 8.12.4 pre-holdout training protocol:\n"
            "- All historical walk-forward validation experiments above are preserved and strictly **not rerun**.\n"
            "- Unsupervised Isolation Forest (`isolation_forest`, 12 features) was fitted on the full eligible target-independent "
            "pre-holdout population (`2018-04-30` to `2025-10-10`, 466,486 rows, 1,841 sessions).\n"
            "- Zero target dependency: fitted completely without `hre_5d_train` labels, retaining valid unlabelled suspension sessions.\n"
            "- Final holdout (74,334 rows) strictly quarantined with zero access."
        ),
        "code": (
            "# Phase 8 Batch 5 (Stage 8.12.4): Inspect Pre-Holdout Isolation Forest Artifacts\n"
            "import json\n"
            "from pathlib import Path\n"
            "import joblib\n"
            "import pandas as pd\n\n"
            "models_dir = REPO_ROOT / 'models' / 'phase8_batch5'\n"
            "meta_dir = REPO_ROOT / 'reports' / 'phase8_batch5' / 'training_metadata'\n\n"
            "m_name = 'isolation_forest'\n"
            "meta_file = meta_dir / f'{m_name}_training_metadata.json'\n"
            "with open(meta_file, 'r', encoding='utf-8') as f:\n"
            "    meta = json.load(f)\n"
            "art_path = models_dir / m_name / 'model.joblib'\n"
            "detector = joblib.load(art_path)\n"
            "if_rec = [{\n"
            "    'Model Identifier': meta['model_identifier'],\n"
            "    'Role': 'UNSUPERVISED ANOMALY TELEMETRY',\n"
            "    'Algorithm': meta['algorithm'],\n"
            "    'Target Independent': 'YES (Zero Target Labels)',\n"
            "    'Feature Count': meta['feature_count'],\n"
            "    'Training Start': meta['training_start_date'],\n"
            "    'Training End': meta['training_end_date'],\n"
            "    'Eligible Rows': f\"{meta['eligible_training_rows']:,}\",\n"
            "    'Score Min / Max (Raw)': f\"{meta['score_min_raw']:.4f} / {meta['score_max_raw']:.4f}\",\n"
            "    'Fit Time (s)': f\"{meta['fit_time_seconds']:.2f}s\",\n"
            "    'Artifact Path': str(art_path.relative_to(REPO_ROOT)),\n"
            "    'Status': meta['training_completion_status'],\n"
            "}]\n\n"
            "print('=' * 80)\n"
            "print('STAGE 8.12.4: PRE-HOLDOUT ISOLATION FOREST TELEMETRY SUMMARY')\n"
            "print('=' * 80)\n"
            "if_df = pd.DataFrame(if_rec)\n"
            "print(if_df.to_string(index=False))\n"
        ),
    },
}

def main():
    print("Updating notebooks with Stage 8.12.4 pre-holdout sections...")
    for rel_path, update in NOTEBOOK_UPDATES.items():
        nb_path = REPO_ROOT / rel_path
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # Check if already appended
        has_section = any(
            update["section_title"] in cell.get("source", "")
            for cell in nb.cells
        )

        if has_section:
            print(f"  {rel_path}: Stage 8.12.4 section already present. Skipping append.")
            continue

        # Execute only the new code cell in isolation to capture its real output
        code_str = update["code"]
        stdout_buf = io.StringIO()
        exec_scope = {"REPO_ROOT": REPO_ROOT}

        with contextlib.redirect_stdout(stdout_buf):
            exec(code_str, exec_scope)

        captured_text = stdout_buf.getvalue()

        # Compute next execution count
        max_exec = max(
            (cell.get("execution_count") or 0 for cell in nb.cells if cell.cell_type == "code"),
            default=0,
        )
        new_exec_count = max_exec + 1

        md_cell = new_markdown_cell(update["markdown"])
        out_stream = new_output("stream", name="stdout", text=captured_text)
        cd_cell = new_code_cell(code_str, execution_count=new_exec_count, outputs=[out_stream])

        nb.cells.extend([md_cell, cd_cell])

        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        print(f"  {rel_path}: Appended Stage 8.12.4 section (cells {len(nb.cells)-2} and {len(nb.cells)-1}). Historical outputs preserved.")

    print("All four notebooks successfully updated!")

if __name__ == "__main__":
    main()
