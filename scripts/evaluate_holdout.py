"""Phase 8 Batch 5 (Batch 2): Final Holdout Evaluation & Walk-Forward Comparison.

Stages:
- Stage 8.12.5: Supervised Holdout Evaluation (5 models)
- Stage 8.12.6: Isolation Forest Holdout Evaluation
- Stage 8.12.7: Walk-Forward vs Holdout Comparison

Strictly READ-ONLY regarding models and datasets:
- Loads fitted models from models/phase8_batch5/
- ZERO .fit() or .fit_transform() calls
- Computes metrics on holdout (split == 'holdout')
- Generates artifacts in data/interim/phase8_batch5/holdout/
- Generates plots in reports/phase8_batch5/figures/
"""

from pathlib import Path
import json
import time
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.calibration import calibration_curve

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Frozen historical thresholds from Batch 4
FROZEN_THRESHOLDS = {
    "hist_gradient_boosting": 0.534,
    "random_forest_candidate_b": 0.508,
    "random_forest_candidate_a": 0.504,
    "logistic_regression_standard": 0.460,
    "logistic_regression_robust": 0.458,
}

SUPERVISED_MODELS = [
    "logistic_regression_standard",
    "logistic_regression_robust",
    "random_forest_candidate_a",
    "random_forest_candidate_b",
    "hist_gradient_boosting",
]


def run_holdout_evaluation():
    print("=" * 80)
    print("PHASE 8 BATCH 5 (BATCH 2): FINAL HOLDOUT EVALUATION & COMPARISON")
    print("=" * 80)

    # 1. Output directories
    out_dir = REPO_ROOT / "data" / "interim" / "phase8_batch5" / "holdout"
    fig_dir = REPO_ROOT / "reports" / "phase8_batch5" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load dataset
    print("\n[Step 1] Loading features_with_targets.parquet...")
    data_path = REPO_ROOT / "data" / "processed" / "features_with_targets.parquet"
    df = pd.read_parquet(data_path)
    print(f"  Total dataset shape: {df.shape}")

    holdout_df = df[df["split"] == "holdout"].copy()
    print(f"  Holdout rows: {len(holdout_df):,}, dates: {holdout_df['date'].min()} to {holdout_df['date'].max()}")
    print(f"  Unique trading sessions: {holdout_df['date'].nunique()}")

    target_col = "hre_5d"
    valid_mask = holdout_df[target_col].notna()
    print(f"  Valid target rows: {valid_mask.sum():,} ({valid_mask.mean():.2%})")
    print(f"  Missing target rows: {(~valid_mask).sum():,} ({(~valid_mask).mean():.2%})")

    y_true_all = holdout_df.loc[valid_mask, target_col].values.astype(int)
    pos_count = int(np.sum(y_true_all == 1))
    neg_count = int(np.sum(y_true_all == 0))
    prevalence = float(pos_count / len(y_true_all))
    print(f"  Positive class (HRE=1): {pos_count:,} ({prevalence:.2%})")
    print(f"  Negative class (HRE=0): {neg_count:,} ({1 - prevalence:.2%})")

    # 3. Supervised Model Evaluations
    print("\n[Step 2] Evaluating Supervised Models on Holdout...")
    models_dir = REPO_ROOT / "models" / "phase8_batch5"

    sup_preds_df = holdout_df[["date", "ticker", "cap_segment", "hre_5d"]].copy()
    sup_metrics_list = []

    for model_name in SUPERVISED_MODELS:
        m_dir = models_dir / model_name
        art_path = m_dir / "model.joblib"
        feat_path = m_dir / "feature_names.json"

        print(f"\n  Evaluating {model_name}...")
        features = json.load(open(feat_path, "r", encoding="utf-8"))["features"]
        pipeline = joblib.load(art_path)

        t0_pred = time.perf_counter()
        probs = pipeline.predict_proba(holdout_df[features])[:, 1]
        pred_time = time.perf_counter() - t0_pred
        print(f"    Inference completed in {pred_time:.2f}s ({len(probs)/pred_time:,.0f} rows/s)")

        tau_calib = FROZEN_THRESHOLDS[model_name]
        preds_50 = (probs >= 0.50).astype(int)
        preds_calib = (probs >= tau_calib).astype(int)

        sup_preds_df[f"prob_{model_name}"] = probs
        sup_preds_df[f"pred_50_{model_name}"] = preds_50
        sup_preds_df[f"pred_calib_{model_name}"] = preds_calib

        # Metrics on valid evaluation subset
        y_prob = probs[valid_mask]
        pr_auc = float(average_precision_score(y_true_all, y_prob))
        roc_auc = float(roc_auc_score(y_true_all, y_prob))
        brier = float(brier_score_loss(y_true_all, y_prob))
        clipped_p = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
        ll = float(log_loss(y_true_all, clipped_p))

        # Metrics @ 0.50
        p_50 = preds_50[valid_mask]
        f1_50 = float(f1_score(y_true_all, p_50, zero_division=0))
        prec_50 = float(precision_score(y_true_all, p_50, zero_division=0))
        rec_50 = float(recall_score(y_true_all, p_50, zero_division=0))
        bal_acc_50 = float(balanced_accuracy_score(y_true_all, p_50))
        cm_50 = confusion_matrix(y_true_all, p_50, labels=[0, 1])
        tn_50, fp_50, fn_50, tp_50 = [int(v) for v in cm_50.ravel()]

        # Metrics @ calibrated threshold
        p_calib = preds_calib[valid_mask]
        f1_calib = float(f1_score(y_true_all, p_calib, zero_division=0))
        prec_calib = float(precision_score(y_true_all, p_calib, zero_division=0))
        rec_calib = float(recall_score(y_true_all, p_calib, zero_division=0))
        bal_acc_calib = float(balanced_accuracy_score(y_true_all, p_calib))
        cm_calib = confusion_matrix(y_true_all, p_calib, labels=[0, 1])
        tn_calib, fp_calib, fn_calib, tp_calib = [int(v) for v in cm_calib.ravel()]

        # Cap segment stratification
        cap_pr_auc = {}
        for seg in ["large", "mid", "small"]:
            seg_mask = valid_mask & (holdout_df["cap_segment"] == seg)
            if seg_mask.sum() > 0 and len(np.unique(holdout_df.loc[seg_mask, target_col])) > 1:
                cap_pr_auc[seg] = float(average_precision_score(
                    holdout_df.loc[seg_mask, target_col].values.astype(int),
                    probs[seg_mask.values]
                ))
            else:
                cap_pr_auc[seg] = np.nan

        rec = {
            "model_name": model_name,
            "holdout_pr_auc": pr_auc,
            "holdout_roc_auc": roc_auc,
            "holdout_brier_score": brier,
            "holdout_log_loss": ll,
            "calibrated_threshold": tau_calib,
            "f1_50": f1_50,
            "precision_50": prec_50,
            "recall_50": rec_50,
            "balanced_accuracy_50": bal_acc_50,
            "tp_50": tp_50,
            "fp_50": fp_50,
            "tn_50": tn_50,
            "fn_50": fn_50,
            "f1_calib": f1_calib,
            "precision_calib": prec_calib,
            "recall_calib": rec_calib,
            "balanced_accuracy_calib": bal_acc_calib,
            "tp_calib": tp_calib,
            "fp_calib": fp_calib,
            "tn_calib": tn_calib,
            "fn_calib": fn_calib,
            "large_cap_pr_auc": cap_pr_auc.get("large", np.nan),
            "mid_cap_pr_auc": cap_pr_auc.get("mid", np.nan),
            "small_cap_pr_auc": cap_pr_auc.get("small", np.nan),
            "inference_time_sec": pred_time,
            "evaluated_rows": int(valid_mask.sum()),
            "total_holdout_rows": len(holdout_df),
        }
        sup_metrics_list.append(rec)
        print(f"    PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | Brier: {brier:.4f} | F1@0.50: {f1_50:.4f} | F1@calib ({tau_calib:.3f}): {f1_calib:.4f}")

    sup_metrics_df = pd.DataFrame(sup_metrics_list).sort_values("holdout_pr_auc", ascending=False)
    sup_metrics_df.to_csv(out_dir / "supervised_holdout_metrics.csv", index=False)
    sup_metrics_df.to_csv(out_dir / "supervised_holdout_metrics_by_model.csv", index=False)
    sup_preds_df.to_parquet(out_dir / "supervised_holdout_predictions.parquet", index=False)
    print(f"\n  Saved supervised holdout metrics to: {out_dir / 'supervised_holdout_metrics.csv'}")
    print(f"  Saved supervised holdout predictions to: {out_dir / 'supervised_holdout_predictions.parquet'}")

    # 4. Isolation Forest Holdout Evaluation
    print("\n[Step 3] Evaluating Isolation Forest on Holdout...")
    if_dir = models_dir / "isolation_forest"
    if_detector = joblib.load(if_dir / "model.joblib")

    t0_if = time.perf_counter()
    if_scores_df = if_detector.score(holdout_df)
    if_score_time = time.perf_counter() - t0_if
    print(f"  Isolation Forest scored in {if_score_time:.2f}s")

    raw_scores = if_scores_df["isolation_score"].values
    scaled_scores = if_scores_df["anomaly_score_scaled"].values
    flags = if_scores_df["isolation_flag"].values

    flagged_count = int(np.sum(flags == 1))
    flagged_rate = float(np.mean(flags == 1))

    # Overlap with holdout target on valid rows
    flags_valid = flags[valid_mask]
    scaled_valid = scaled_scores[valid_mask]
    y_valid = y_true_all

    tp_anomaly = int(np.sum((flags_valid == 1) & (y_valid == 1)))
    fp_anomaly = int(np.sum((flags_valid == 1) & (y_valid == 0)))
    tn_anomaly = int(np.sum((flags_valid == 0) & (y_valid == 0)))
    fn_anomaly = int(np.sum((flags_valid == 0) & (y_valid == 1)))

    hre_rate_flagged = float(tp_anomaly / (tp_anomaly + fp_anomaly)) if (tp_anomaly + fp_anomaly) > 0 else np.nan
    hre_rate_unflagged = float(fn_anomaly / (fn_anomaly + tn_anomaly)) if (fn_anomaly + tn_anomaly) > 0 else np.nan
    enrichment_ratio = float(hre_rate_flagged / hre_rate_unflagged) if hre_rate_unflagged > 0 else np.nan
    corr_val = float(np.corrcoef(scaled_valid, y_valid)[0, 1])

    if_metrics = {
        "model_name": "isolation_forest",
        "total_holdout_rows": len(holdout_df),
        "valid_target_rows": int(valid_mask.sum()),
        "score_time_sec": if_score_time,
        "mean_scaled_score": float(np.mean(scaled_scores)),
        "median_scaled_score": float(np.median(scaled_scores)),
        "std_scaled_score": float(np.std(scaled_scores)),
        "p90_scaled_score": float(np.percentile(scaled_scores, 90)),
        "p95_scaled_score": float(np.percentile(scaled_scores, 95)),
        "p99_scaled_score": float(np.percentile(scaled_scores, 99)),
        "flagged_count": flagged_count,
        "flagged_rate": flagged_rate,
        "tp_anomaly": tp_anomaly,
        "fp_anomaly": fp_anomaly,
        "tn_anomaly": tn_anomaly,
        "fn_anomaly": fn_anomaly,
        "hre_rate_flagged": hre_rate_flagged,
        "hre_rate_unflagged": hre_rate_unflagged,
        "enrichment_ratio": enrichment_ratio,
        "anomaly_target_corr": corr_val,
    }

    if_metrics_df = pd.DataFrame([if_metrics])
    if_metrics_df.to_csv(out_dir / "isolation_forest_holdout_metrics.csv", index=False)

    if_preds_df = holdout_df[["date", "ticker", "cap_segment", "hre_5d"]].copy()
    if_preds_df["isolation_score"] = raw_scores
    if_preds_df["anomaly_score_scaled"] = scaled_scores
    if_preds_df["isolation_flag"] = flags
    if_preds_df.to_parquet(out_dir / "isolation_forest_holdout_predictions.parquet", index=False)
    print(f"  Saved IF holdout metrics to: {out_dir / 'isolation_forest_holdout_metrics.csv'}")
    print(f"  Saved IF holdout predictions to: {out_dir / 'isolation_forest_holdout_predictions.parquet'}")
    print(f"  IF Scaled Mean: {if_metrics['mean_scaled_score']:.2f} | Flag Rate: {flagged_rate:.2%} | Enrichment: {enrichment_ratio:.2f}x | Corr: {corr_val:.4f}")

    # 5. Walk-Forward vs Holdout Comparison (Stage 8.12.7)
    print("\n[Step 4] Computing Walk-Forward vs. Holdout Comparison...")
    wf_comp_path = REPO_ROOT / "data" / "interim" / "phase8_batch4" / "supervised_model_comparison.csv"
    wf_comp = pd.read_csv(wf_comp_path)

    comparison_records = []
    for _, h_row in sup_metrics_df.iterrows():
        m_name = h_row["model_name"]
        wf_sub = wf_comp[wf_comp["model_name"] == m_name]
        if wf_sub.empty:
            continue
        w = wf_sub.iloc[0]

        comp_rec = {
            "model_name": m_name,
            "wf_mean_pr_auc": w["mean_pr_auc"],
            "wf_pooled_pr_auc": w["pooled_pr_auc"],
            "holdout_pr_auc": h_row["holdout_pr_auc"],
            "pr_auc_delta_vs_mean": h_row["holdout_pr_auc"] - w["mean_pr_auc"],
            "pr_auc_retention_pct": (h_row["holdout_pr_auc"] / w["mean_pr_auc"]) * 100.0,
            "wf_mean_roc_auc": w["mean_roc_auc"],
            "wf_pooled_roc_auc": w["pooled_roc_auc"],
            "holdout_roc_auc": h_row["holdout_roc_auc"],
            "roc_auc_delta": h_row["holdout_roc_auc"] - w["mean_roc_auc"],
            "wf_brier_score": w["mean_brier_score"],
            "holdout_brier_score": h_row["holdout_brier_score"],
            "brier_delta": h_row["holdout_brier_score"] - w["mean_brier_score"],
            "wf_log_loss": w["mean_log_loss"],
            "holdout_log_loss": h_row["holdout_log_loss"],
            "wf_f1_50": w["mean_f1_50"],
            "holdout_f1_50": h_row["f1_50"],
            "wf_f1_calib": w["mean_f1_calib"],
            "holdout_f1_calib": h_row["f1_calib"],
            "calibrated_threshold": h_row["calibrated_threshold"],
            "holdout_precision_50": h_row["precision_50"],
            "holdout_recall_50": h_row["recall_50"],
            "holdout_precision_calib": h_row["precision_calib"],
            "holdout_recall_calib": h_row["recall_calib"],
            "wf_large_cap_pr": w["large_cap_pr_auc"],
            "holdout_large_cap_pr": h_row["large_cap_pr_auc"],
            "wf_mid_cap_pr": w["mid_cap_pr_auc"],
            "holdout_mid_cap_pr": h_row["mid_cap_pr_auc"],
            "wf_small_cap_pr": w["small_cap_pr_auc"],
            "holdout_small_cap_pr": h_row["small_cap_pr_auc"],
        }
        comparison_records.append(comp_rec)

    comp_df = pd.DataFrame(comparison_records).sort_values("holdout_pr_auc", ascending=False)
    comp_df["wf_rank"] = [wf_comp[wf_comp["model_name"] == m].index[0] + 1 for m in comp_df["model_name"]]
    comp_df["holdout_rank"] = range(1, len(comp_df) + 1)
    comp_df.to_csv(out_dir / "walkforward_vs_holdout_comparison.csv", index=False)
    print(f"  Saved comparison to: {out_dir / 'walkforward_vs_holdout_comparison.csv'}")

    # Isolation Forest WF vs Holdout Comparison
    wf_if_path = REPO_ROOT / "data" / "interim" / "phase8_batch4" / "isolation_forest_fold_metrics.csv"
    wf_if = pd.read_csv(wf_if_path)

    if_comp_rec = {
        "metric": [
            "Mean Scaled Score",
            "Median Scaled Score",
            "P90 Scaled Score",
            "P95 Scaled Score",
            "P99 Scaled Score",
            "Flag Rate",
            "Enrichment Ratio (Flagged/Unflagged HRE)",
            "Target Correlation",
        ],
        "walk_forward_mean": [
            float(wf_if["mean_scaled_score"].mean()),
            float(wf_if["median_scaled_score"].mean()),
            float(wf_if["p90_scaled_score"].mean()),
            float(wf_if["p95_scaled_score"].mean()),
            float(wf_if["p99_scaled_score"].mean()),
            float(wf_if["flagged_rate"].mean()),
            float((wf_if["hre_pos_flag_rate"] / wf_if["hre_neg_flag_rate"]).mean()),
            float(wf_if["anomaly_target_corr"].mean()),
        ],
        "holdout_value": [
            if_metrics["mean_scaled_score"],
            if_metrics["median_scaled_score"],
            if_metrics["p90_scaled_score"],
            if_metrics["p95_scaled_score"],
            if_metrics["p99_scaled_score"],
            if_metrics["flagged_rate"],
            if_metrics["enrichment_ratio"],
            if_metrics["anomaly_target_corr"],
        ],
    }
    if_comp_df = pd.DataFrame(if_comp_rec)
    if_comp_df.to_csv(out_dir / "isolation_forest_walkforward_vs_holdout.csv", index=False)
    print(f"  Saved IF comparison to: {out_dir / 'isolation_forest_walkforward_vs_holdout.csv'}")

    # 6. Generate Figures
    print("\n[Step 5] Generating Comparison Plots...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: PR Curves
    plt.figure(figsize=(10, 7), dpi=300)
    for model_name in SUPERVISED_MODELS:
        y_prob = sup_preds_df.loc[valid_mask, f"prob_{model_name}"].values
        prec, rec, _ = precision_recall_curve(y_true_all, y_prob)
        score = average_precision_score(y_true_all, y_prob)
        plt.plot(rec, prec, label=f"{model_name} (PR-AUC = {score:.4f})", linewidth=2)
    plt.axhline(prevalence, color="red", linestyle="--", label=f"Holdout Base Rate ({prevalence:.2%})")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Phase 8 Holdout: Precision-Recall Curves (250 Trading Sessions, N=71,756)", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(fig_dir / "holdout_pr_curves.png")
    plt.close()

    # Figure 2: ROC Curves
    plt.figure(figsize=(10, 7), dpi=300)
    for model_name in SUPERVISED_MODELS:
        y_prob = sup_preds_df.loc[valid_mask, f"prob_{model_name}"].values
        fpr, tpr, _ = roc_curve(y_true_all, y_prob)
        score = roc_auc_score(y_true_all, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (ROC-AUC = {score:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Chance (0.50)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Phase 8 Holdout: ROC Curves (250 Trading Sessions, N=71,756)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(fig_dir / "holdout_roc_curves.png")
    plt.close()

    # Figure 3: Calibration Curves
    plt.figure(figsize=(10, 7), dpi=300)
    for model_name in SUPERVISED_MODELS:
        y_prob = sup_preds_df.loc[valid_mask, f"prob_{model_name}"].values
        frac_pos, mean_pred = calibration_curve(y_true_all, y_prob, n_bins=10, strategy="uniform")
        plt.plot(mean_pred, frac_pos, "s-", label=f"{model_name}", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives", fontsize=12)
    plt.title("Phase 8 Holdout: Probability Calibration Curves (Reliability Diagram)", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(fig_dir / "holdout_calibration_curves.png")
    plt.close()

    # Figure 4: WF vs Holdout PR-AUC Comparison Bar Chart
    plt.figure(figsize=(12, 6), dpi=300)
    bar_width = 0.35
    idx = np.arange(len(comp_df))
    plt.bar(idx - bar_width/2, comp_df["wf_mean_pr_auc"], bar_width, label="Walk-Forward (Mean Fold)", color="#4A90E2", alpha=0.85)
    plt.bar(idx + bar_width/2, comp_df["holdout_pr_auc"], bar_width, label="Holdout (250 Sessions)", color="#E94E77", alpha=0.85)
    plt.xticks(idx, comp_df["model_name"], rotation=15, ha="right", fontsize=10)
    plt.ylabel("PR-AUC (Average Precision)", fontsize=12)
    plt.title("Generalization Check: Walk-Forward Mean vs. Final Holdout PR-AUC", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(fig_dir / "walkforward_vs_holdout_comparison.png")
    plt.close()

    # Figure 5: Isolation Forest Score Distribution on Holdout
    plt.figure(figsize=(10, 6), dpi=300)
    plt.hist(scaled_scores, bins=50, color="#50E3C2", edgecolor="black", alpha=0.75, density=True)
    plt.axvline(np.percentile(scaled_scores, 95), color="orange", linestyle="--", label=f"P95 ({np.percentile(scaled_scores, 95):.1f})")
    plt.axvline(np.percentile(scaled_scores, 99), color="red", linestyle="--", label=f"P99 ({np.percentile(scaled_scores, 99):.1f})")
    plt.xlabel("Scaled Anomaly Score [0 - 100]", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Holdout: Unsupervised Isolation Forest Anomaly Score Distribution", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(fig_dir / "isolation_forest_holdout_distribution.png")
    plt.close()

    print("  All figures saved to:", fig_dir.resolve())
    print("\nHoldout evaluation and comparison successfully executed!")


if __name__ == "__main__":
    run_holdout_evaluation()
