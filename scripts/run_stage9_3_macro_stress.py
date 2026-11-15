"""Script to execute Stage 9.3 Macro Stress Testing across market-cap segments."""

import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.modeling.phase9.macro_stress import (
    REPRESENTATIVE_STRESS_EPISODES,
    compute_isolation_forest_stress_clustering,
    compute_segment_divergence,
    compute_stress_period_metrics,
    define_stress_episodes,
    run_statistical_significance_tests,
)


def plot_stress_period_spread_widening(stress_metrics_df: pd.DataFrame, out_path: Path):
    """Plot Pre vs During vs Post spread widening across market cap segments."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    episodes = [
        ("EP_01_COVID_CRASH", "COVID-19 Lower Circuit Halts (Mar 2020)"),
        ("EP_05_ELECTION_2024_SHOCK", "Lok Sabha 2024 Election Shock (Jun 2024)"),
        ("EP_06_SEBI_FROTH_WARNING", "SEBI Small/Mid-Cap Froth Warning (Mar 2024)"),
    ]

    segments = ["LARGE", "MID", "SMALL"]
    colors = {"PRE": "#2ca02c", "DURING": "#d62728", "POST": "#1f77b4"}

    for ax, (ep_id, title) in zip(axes, episodes):
        ep_sub = stress_metrics_df[stress_metrics_df["episode_id"] == ep_id]
        if ep_sub.empty:
            continue

        bar_width = 0.25
        x = np.arange(len(segments))

        for i, period in enumerate(["PRE", "DURING", "POST"]):
            p_vals = []
            for seg in segments:
                row = ep_sub[(ep_sub["segment"] == seg) & (ep_sub["period"] == period)]
                val = row["cs_spread_est_median"].iloc[0] * 100.0 if not row.empty else 0.0
                p_vals.append(val)

            ax.bar(x + (i - 1) * bar_width, p_vals, width=bar_width, label=period, color=colors[period], alpha=0.85)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(segments, fontweight="bold")
        ax.set_xlabel("Market Cap Segment", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    axes[0].set_ylabel("Median Corwin-Schultz Spread (%)", fontsize=11, fontweight="bold")
    axes[0].legend(title="Stress Period", frameon=True)
    plt.suptitle("Cross-Sectional Bid-Ask Spread Widening Across Historical Stress Episodes", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_stress_segment_divergence(divergence_df: pd.DataFrame, out_path: Path):
    """Plot Small-cap vs Large-cap differential spread shock across episodes."""
    div_sub = divergence_df[divergence_df["segment"] == "DIVERGENCE_SMALL_VS_LARGE"].copy()
    if div_sub.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    ep_names = [name[:30] + ("..." if len(name) > 30 else "") for name in div_sub["episode_name"]]
    y_pos = np.arange(len(ep_names))
    spread_div_bp = div_sub["spread_shock_abs"] * 10000.0  # Basis points

    bar_colors = ["#d62728" if v > 0 else "#1f77b4" for v in spread_div_bp]

    bars = ax.barh(y_pos, spread_div_bp, color=bar_colors, alpha=0.85, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ep_names, fontsize=10)
    ax.set_xlabel("Small-Cap Minus Large-Cap Spread Widening Differential (Basis Points)", fontsize=11, fontweight="bold")
    ax.set_title("Liquidity Evaporation Asymmetry: Small-Cap Excess Spread Widening", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linestyle="-", linewidth=1.0)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    for bar, val in zip(bars, spread_div_bp):
        if np.isnan(val):
            continue
        ha = "left" if val >= 0 else "right"
        offset = 5 if val >= 0 else -5
        ax.annotate(f"{val:+.1f} bps",
                    xy=(val, bar.get_y() + bar.get_height() / 2),
                    xytext=(offset, 0), textcoords="offset points",
                    ha=ha, va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_isolation_forest_clustering(if_clustering_df: pd.DataFrame, out_path: Path):
    """Plot Isolation Forest anomaly flag surge during stress periods."""
    fig, ax = plt.subplots(figsize=(13, 6))

    mkt_sub = if_clustering_df[if_clustering_df["segment"] == "MARKET_WIDE"].copy()
    if mkt_sub.empty:
        return

    x = np.arange(len(mkt_sub))
    bar_width = 0.25

    ax.bar(x - bar_width, mkt_sub["pre_flag_rate_pct"], width=bar_width, label="Pre-Stress Flag Rate", color="#2ca02c", alpha=0.85)
    ax.bar(x, mkt_sub["during_flag_rate_pct"], width=bar_width, label="During-Stress Flag Rate", color="#d62728", alpha=0.85)
    ax.bar(x + bar_width, mkt_sub["post_flag_rate_pct"], width=bar_width, label="Post-Stress Flag Rate", color="#1f77b4", alpha=0.85)

    ep_labels = [name[:25] for name in mkt_sub["episode_name"]]
    ax.set_xticks(x)
    ax.set_xticklabels(ep_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Market-Wide Anomaly Flag Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Isolation Forest Anomaly Clustering Across Historical Stress Episodes", fontsize=13, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Annotate enrichment ratios above During bars
    for i, (_, row) in enumerate(mkt_sub.iterrows()):
        ratio = row["stress_enrichment_ratio"]
        if not np.isnan(ratio):
            ax.annotate(f"{ratio:.1f}x",
                        xy=(i, row["during_flag_rate_pct"]),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold", color="#d62728")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("=== STAGE 9.3: EXECUTING MACRO STRESS TESTING & SEGMENT DIVERGENCE ===")
    t0 = time.time()

    # 1. Load event stock observations
    obs_path = REPO_ROOT / "data" / "interim" / "phase9" / "event_windows" / "event_stock_observations.parquet"
    print(f"1. Loading stock observations from {obs_path}...")
    event_stocks_df = pd.read_parquet(obs_path)
    print(f"   Loaded {len(event_stocks_df)} records across {event_stocks_df['ticker'].nunique()} tickers.")

    # 2. Define stress episodes
    events_csv = REPO_ROOT / "data" / "events" / "market_events.csv"
    print(f"2. Cataloging representative stress episodes against {events_csv}...")
    events_df = pd.read_csv(events_csv)
    episodes_df = define_stress_episodes(events_df)
    print(f"   Cataloged {len(episodes_df)} representative stress episodes across {len(REPRESENTATIVE_STRESS_EPISODES)} configurations.")

    # 3. Compute stress period metrics
    print("3. Aggregating Pre, During, and Post stress period metrics by segment...")
    stress_metrics_df = compute_stress_period_metrics(event_stocks_df)
    out_dir = REPO_ROOT / "data" / "interim" / "phase9" / "macro_stress_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_parquet = out_dir / "macro_stress_summary.parquet"
    summary_csv = out_dir / "macro_stress_summary.csv"
    stress_metrics_df.to_parquet(summary_parquet, index=False)
    stress_metrics_df.to_csv(summary_csv, index=False)
    print(f"   Saved {len(stress_metrics_df)} aggregated summary rows to {summary_parquet.name} and {summary_csv.name}.")

    # 4. Compute segment divergence
    print("4. Computing Small-cap vs Large-cap divergence and recovery ratios...")
    divergence_df = compute_segment_divergence(stress_metrics_df)
    div_csv = out_dir / "stress_segment_divergence.csv"
    divergence_df.to_csv(div_csv, index=False)
    print(f"   Saved {len(divergence_df)} divergence records to {div_csv.name}.")

    # 5. Run statistical tests (Mann-Whitney U, Wilcoxon, Bootstrap CIs)
    print("5. Running non-parametric significance tests and 1000-sample bootstrap CIs...")
    stat_tests_df = run_statistical_significance_tests(
        event_stocks_df,
        n_bootstrap=1000,
        random_state=42,
    )
    tests_csv = out_dir / "stress_statistical_tests.csv"
    stat_tests_df.to_csv(tests_csv, index=False)
    print(f"   Saved {len(stat_tests_df)} statistical test results to {tests_csv.name}.")

    # 6. Compute Isolation Forest anomaly flag clustering
    print("6. Computing Isolation Forest anomaly clustering and enrichment ratios...")
    if_clustering_df = compute_isolation_forest_stress_clustering(event_stocks_df)
    if_csv = out_dir / "isolation_forest_stress_clustering.csv"
    if_clustering_df.to_csv(if_csv, index=False)
    print(f"   Saved {len(if_clustering_df)} anomaly clustering records to {if_csv.name}.")

    # 7. Generate publication-quality figures
    fig_dir = REPO_ROOT / "reports" / "phase9" / "figures" / "liquidity_stress"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("7. Generating figures in reports/phase9/figures/liquidity_stress/...")
    plot_stress_period_spread_widening(
        stress_metrics_df,
        fig_dir / "stress_period_spread_widening.png",
    )
    plot_stress_segment_divergence(
        divergence_df,
        fig_dir / "stress_segment_divergence.png",
    )
    plot_isolation_forest_clustering(
        if_clustering_df,
        fig_dir / "isolation_forest_stress_clustering.png",
    )

    elapsed = time.time() - t0
    print(f"=== STAGE 9.3 COMPLETED IN {elapsed:.2f}s ===")


if __name__ == "__main__":
    main()
