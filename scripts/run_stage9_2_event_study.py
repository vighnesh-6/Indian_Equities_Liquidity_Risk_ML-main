"""Script to execute Stage 9.2 Historical Event Studies across the 86 market events."""

import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.modeling.phase9.event_study import (
    build_event_windows,
    compute_cross_sectional_event_metrics,
    compute_event_impact_summary,
    extract_event_stock_data,
    load_and_cluster_events,
)

def main():
    print("=== STAGE 9.2: EXECUTING HISTORICAL EVENT STUDIES ===")
    t0 = time.time()

    # 1. Load and cluster events
    print("1. Loading and clustering market events...")
    events_df = load_and_cluster_events(
        events_csv_path=REPO_ROOT / "data" / "events" / "market_events.csv",
        calendar_features_path=REPO_ROOT / "data" / "processed" / "features_with_targets.parquet",
    )
    print(f"   Loaded {len(events_df)} events: {events_df['is_overlapping_event'].sum()} overlapping in clusters, {(~events_df['is_overlapping_event']).sum()} isolated.")

    # 2. Get trading calendar
    print("2. Extracting exchange trading calendar...")
    feat_dates = pd.read_parquet(REPO_ROOT / "data" / "processed" / "features_with_targets.parquet", columns=["date"])
    trading_cal = sorted(feat_dates["date"].unique())
    print(f"   Calendar spans {len(trading_cal)} trading days ({trading_cal[0].strftime('%Y-%m-%d')} to {trading_cal[-1].strftime('%Y-%m-%d')}).")

    # 3. Build event windows [-5, +5]
    print("3. Building [-5, +5] event window schedule...")
    windows_df = build_event_windows(events_df, trading_cal, window_pre=5, window_post=5)
    print(f"   Generated {len(windows_df)} event-session records.")

    out_win_dir = REPO_ROOT / "data" / "interim" / "phase9" / "event_windows"
    out_win_dir.mkdir(parents=True, exist_ok=True)
    windows_df.to_parquet(out_win_dir / "event_windows_schedule.parquet", index=False)

    # 4. Load features and predictions
    print("4. Loading features and existing predictions...")
    features_df = pd.read_parquet(REPO_ROOT / "data" / "processed" / "features_with_targets.parquet")
    wf_preds_df = pd.read_parquet(REPO_ROOT / "data" / "interim" / "phase8_batch4" / "supervised_predictions.parquet")
    ho_preds_df = pd.read_parquet(REPO_ROOT / "data" / "interim" / "phase8_batch5" / "holdout" / "supervised_holdout_predictions.parquet")
    wf_if_df = pd.read_parquet(REPO_ROOT / "data" / "interim" / "phase8_batch4" / "isolation_forest_predictions.parquet")
    ho_if_df = pd.read_parquet(REPO_ROOT / "data" / "interim" / "phase8_batch5" / "holdout" / "isolation_forest_holdout_predictions.parquet")

    # Load frozen model binaries strictly for in-sample pre-2023 evaluation
    hgbt_model = joblib.load(REPO_ROOT / "models" / "phase8_batch5" / "hist_gradient_boosting" / "model.joblib")
    with open(REPO_ROOT / "models" / "phase8_batch5" / "hist_gradient_boosting" / "feature_names.json") as f:
        hgbt_feats = json.load(f)["features"]

    rf_model = joblib.load(REPO_ROOT / "models" / "phase8_batch5" / "random_forest_candidate_b" / "model.joblib")
    with open(REPO_ROOT / "models" / "phase8_batch5" / "random_forest_candidate_b" / "feature_names.json") as f:
        rf_feats = json.load(f)["features"]

    import sys
    sys.path.insert(0, str(REPO_ROOT))
    if_detector = joblib.load(REPO_ROOT / "models" / "phase8_batch5" / "isolation_forest" / "model.joblib")

    # 5. Extract stock-level event window observations
    print("5. Merging event windows with stock features and predictions...")
    event_stocks_df = extract_event_stock_data(
        windows_df,
        features_df,
        wf_preds_df,
        ho_preds_df,
        wf_if_df,
        ho_if_df,
        hgbt_model=hgbt_model,
        hgbt_features=hgbt_feats,
        rf_model=rf_model,
        rf_features=rf_feats,
        if_detector=if_detector,
    )
    print(f"   Merged {len(event_stocks_df)} stock-event observations across 300 tickers.")
    event_stocks_df.to_parquet(out_win_dir / "event_stock_observations.parquet", index=False)

    # 6. Compute cross-sectional metrics
    print("6. Computing cross-sectional statistics (median, IQR, cap tiers)...")
    cs_metrics_df = compute_cross_sectional_event_metrics(event_stocks_df)
    out_metrics_dir = REPO_ROOT / "data" / "interim" / "phase9" / "event_study_metrics"
    out_metrics_dir.mkdir(parents=True, exist_ok=True)
    cs_metrics_df.to_parquet(out_metrics_dir / "cross_sectional_metrics.parquet", index=False)
    print(f"   Computed {len(cs_metrics_df)} cross-sectional summary rows.")

    # 7. Compute impact summary
    print("7. Computing event impact summary...")
    impact_summary_df = compute_event_impact_summary(cs_metrics_df)
    impact_summary_df.to_csv(out_metrics_dir / "event_impact_summary.csv", index=False)
    print(f"   Summary generated for {len(impact_summary_df)} events.")

    # 8. Generate publication figures
    print("8. Generating publication figures...")
    fig_dir = REPO_ROOT / "reports" / "phase9" / "figures" / "event_window_analysis"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Average trajectory of CS-Spread and Amihud across [-5, +5] for OOS vs In-Sample
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    mw = cs_metrics_df[cs_metrics_df["segment"] == "market_wide"]

    for regime, label, color in [
        ("OUT_OF_SAMPLE_WF", "Out-of-Sample Walk-Forward (25 Evts)", "#1f77b4"),
        ("OUT_OF_SAMPLE_HOLDOUT", "Out-of-Sample Holdout (3 Evts)", "#2ca02c"),
        ("IN_SAMPLE_PRETRAIN", "In-Sample Pre-Train (58 Evts)", "#d62728"),
    ]:
        sub = mw[mw["prediction_regime"] == regime]
        traj_spread = sub.groupby("relative_day")["cs_spread_est_median"].median() * 10000.0
        traj_amihud = sub.groupby("relative_day")["amihud_ratio_median"].median() * 1e6
        days = traj_spread.index

        axes[0].plot(days, traj_spread, marker="o", label=label, color=color, linewidth=2)
        axes[1].plot(days, traj_amihud, marker="s", label=label, color=color, linewidth=2)

    axes[0].axvline(0, color="black", linestyle="--", alpha=0.7, label="Event Day (t=0)")
    axes[0].set_title("Corwin-Schultz Bid-Ask Spread Around Events", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Relative Trading Day (t)")
    axes[0].set_ylabel("Median CS Spread (Basis Points)")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=9)

    axes[1].axvline(0, color="black", linestyle="--", alpha=0.7, label="Event Day (t=0)")
    axes[1].set_title("Amihud Price Impact Ratio Around Events", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Relative Trading Day (t)")
    axes[1].set_ylabel("Median Amihud Ratio (x 1e-6)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_dir / "event_window_spread_and_amihud.png")
    plt.close()

    # Plot 2: Model High-Risk Probability Trajectory
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    for regime, label, color in [
        ("OUT_OF_SAMPLE_WF", "OOS Walk-Forward (25 Evts)", "#1f77b4"),
        ("OUT_OF_SAMPLE_HOLDOUT", "OOS Holdout (3 Evts)", "#2ca02c"),
        ("IN_SAMPLE_PRETRAIN", "In-Sample Pre-Train (58 Evts)", "#d62728"),
    ]:
        sub = mw[mw["prediction_regime"] == regime]
        traj_prob = sub.groupby("relative_day")["prob_hist_gradient_boosting_median"].median()
        ax.plot(traj_prob.index, traj_prob.values, marker="^", label=label, color=color, linewidth=2)

    ax.axvline(0, color="black", linestyle="--", alpha=0.7, label="Event Day (t=0)")
    ax.axhline(0.534, color="gray", linestyle=":", label="Operating Threshold (tau*=0.534)")
    ax.set_title("Champion HGBT High-Risk Probability Across Event Window", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Trading Day (t)")
    ax.set_ylabel("Median P(hre_5d = 1)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "event_window_hgbt_risk_surge.png")
    plt.close()

    # Plot 3: Market-Cap Tier Divergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    caps = [("large", "Large-Cap", "#1f77b4"), ("mid", "Mid-Cap", "#ff7f0e"), ("small", "Small-Cap", "#2ca02c")]
    for cap, label, color in caps:
        sub = cs_metrics_df[cs_metrics_df["segment"] == cap]
        traj_spread = sub.groupby("relative_day")["cs_spread_est_median"].median() * 10000.0
        traj_prob = sub.groupby("relative_day")["prob_hist_gradient_boosting_median"].median()
        axes[0].plot(traj_spread.index, traj_spread.values, marker="o", label=label, color=color, linewidth=2)
        axes[1].plot(traj_prob.index, traj_prob.values, marker="^", label=label, color=color, linewidth=2)

    axes[0].axvline(0, color="black", linestyle="--", alpha=0.7)
    axes[0].set_title("CS-Spread Divergence by Market Cap", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Relative Trading Day (t)")
    axes[0].set_ylabel("Median CS Spread (bp)")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=9)

    axes[1].axvline(0, color="black", linestyle="--", alpha=0.7)
    axes[1].axhline(0.534, color="gray", linestyle=":")
    axes[1].set_title("HGBT Risk Probability by Market Cap", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Relative Trading Day (t)")
    axes[1].set_ylabel("Median P(hre_5d = 1)")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_dir / "event_window_cap_divergence.png")
    plt.close()

    elapsed = time.time() - t0
    print(f"=== STAGE 9.2 COMPLETED IN {elapsed:.2f}s ===")

if __name__ == "__main__":
    main()
