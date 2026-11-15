"""Script to execute Stage 9.4 Model Explainability & Feature Importance."""

import json
import os
from pathlib import Path
import sys
import time

# Configure loky CPU count
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import joblib
import pandas as pd

from src.modeling.phase9.explainability import (
    compare_feature_importances,
    compute_hgbt_permutation_importance,
    compute_rf_mdi_importance,
    extract_top_10_risk_drivers,
    generate_explainability_figures,
)


def main():
    print("=== STAGE 9.4: EXECUTING MODEL EXPLAINABILITY & FEATURE IMPORTANCE ===")
    t0 = time.time()

    out_dir = REPO_ROOT / "data" / "interim" / "phase9" / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = REPO_ROOT / "reports" / "phase9" / "figures" / "feature_importance"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model binaries & feature manifest
    print("1. Loading frozen model artifacts and feature names...")
    hgbt_model_path = REPO_ROOT / "models" / "phase8_batch5" / "hist_gradient_boosting" / "model.joblib"
    hgbt_model = joblib.load(hgbt_model_path)

    rf_model_path = REPO_ROOT / "models" / "phase8_batch5" / "random_forest_candidate_b" / "model.joblib"
    rf_model = joblib.load(rf_model_path)

    feat_manifest_path = REPO_ROOT / "models" / "phase8_batch5" / "hist_gradient_boosting" / "feature_names.json"
    with open(feat_manifest_path) as f:
        feature_names = json.load(f)["features"]
    print(f"   Loaded CFG_05_HGBT and CFG_04_RF_CANDIDATE_B with {len(feature_names)} features.")

    # 2. Load evaluation datasets
    feat_targets_path = REPO_ROOT / "data" / "processed" / "features_with_targets.parquet"
    print(f"2. Loading evaluation features from {feat_targets_path}...")
    df = pd.read_parquet(feat_targets_path)

    wf_df = df[(df["date"] >= "2023-05-10") & (df["date"] <= "2025-10-10")].dropna(subset=["hre_5d"]).copy()
    ho_df = df[(df["date"] >= "2025-10-20") & (df["date"] <= "2026-10-23")].dropna(subset=["hre_5d"]).copy()
    print(f"   Walk-Forward OOS evaluation rows: {len(wf_df)}")
    print(f"   Holdout evaluation rows: {len(ho_df)}")

    # 3. Compute HGBT Permutation Importance on Walk-Forward OOS data
    print("3. Computing HGBT permutation importance on Walk-Forward OOS data (PR-AUC scoring)...")
    hgbt_wf_imp = compute_hgbt_permutation_importance(
        model=hgbt_model,
        features_df=wf_df,
        feature_names=feature_names,
        target_col="hre_5d",
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        sample_size=20000,
        regime_label="OUT_OF_SAMPLE_WALK_FORWARD",
    )
    wf_csv = out_dir / "hgbt_permutation_importance_wf.csv"
    hgbt_wf_imp.to_csv(wf_csv, index=False)
    print(f"   Saved Walk-Forward importance table to {wf_csv.name}.")

    # Also compute on Holdout data as post-selection holdout analysis
    print("   Computing HGBT permutation importance on Final Holdout data (post-selection)...")
    hgbt_ho_imp = compute_hgbt_permutation_importance(
        model=hgbt_model,
        features_df=ho_df,
        feature_names=feature_names,
        target_col="hre_5d",
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        sample_size=20000,
        regime_label="POST_SELECTION_HOLDOUT",
    )
    ho_csv = out_dir / "hgbt_permutation_importance_holdout.csv"
    hgbt_ho_imp.to_csv(ho_csv, index=False)
    print(f"   Saved Holdout importance table to {ho_csv.name}.")

    # 4. Extract RF Candidate B native MDI importance
    print("4. Extracting Random Forest Candidate B native MDI importance...")
    rf_mdi_imp = compute_rf_mdi_importance(
        model=rf_model,
        feature_names=feature_names,
        model_name="CFG_04_RF_CANDIDATE_B",
    )
    rf_csv = out_dir / "rf_candidate_b_mdi_importance.csv"
    rf_mdi_imp.to_csv(rf_csv, index=False)
    print(f"   Saved RF MDI importance table to {rf_csv.name}.")

    # 5. Comparative ranking analysis
    print("5. Comparing feature rankings across HGBT and RF Candidate B...")
    comp_df = compare_feature_importances(hgbt_wf_imp, rf_mdi_imp)
    comp_csv = out_dir / "feature_importance_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"   Saved comparison table to {comp_csv.name}.")

    # 6. Extract Top 10 Risk Drivers
    print("6. Extracting Top 10 Risk Drivers with financial interpretations...")
    top10_df = extract_top_10_risk_drivers(hgbt_wf_imp)
    top10_csv = out_dir / "top_10_risk_drivers.csv"
    top10_df.to_csv(top10_csv, index=False)
    print(f"   Saved Top 10 Risk Drivers to {top10_csv.name}.")

    # 7. Generate publication-quality figures
    print("7. Generating figures in reports/phase9/figures/feature_importance/...")
    generate_explainability_figures(
        hgbt_imp_df=hgbt_wf_imp,
        comparison_df=comp_df,
        out_dir=fig_dir,
    )

    elapsed = time.time() - t0
    print(f"=== STAGE 9.4 COMPLETED IN {elapsed:.2f}s ===")


if __name__ == "__main__":
    main()
