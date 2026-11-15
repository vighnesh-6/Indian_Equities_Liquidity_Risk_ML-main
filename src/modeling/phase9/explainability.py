"""Model Explainability & Feature Importance Analysis (Phase 9.4).

This module computes empirical feature importance and risk attribution for the
frozen Phase 8 models:
1. Permutation Feature Importance for CFG_05_HGBT (measuring PR-AUC loss).
2. Native Mean Decrease in Impurity (MDI) for CFG_04_RF_CANDIDATE_B.
3. Comparative importance analysis across tree-based architectures.
4. Top 10 risk driver identification with financial domain interpretations.

Strict Methodological Principles:
- Native scikit-learn functionality only (zero external SHAP dependency).
- Evaluated on untouched out-of-sample data (Walk-Forward and Holdout).
- Zero model retraining or parameter modification.
- Explicit non-causal interpretation (attribution of predictive signal, not causal mechanisms).
"""

import os

# Prevent joblib loky physical core detection warning on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

# Mapping of the 36 engineered features to their functional microstructure families
FEATURE_FAMILY_MAP: Dict[str, str] = {
    # Illiquidity & Price Impact
    "amihud_ratio": "ILLIQUIDITY",
    "amihud_ma20": "ILLIQUIDITY",
    "amihud_z20": "ILLIQUIDITY",
    "low_volume_streak": "ILLIQUIDITY",
    # Spread & Price Range
    "cs_spread_est": "SPREAD",
    "cs_spread_ma10": "SPREAD",
    "daily_range": "SPREAD",
    "daily_range_ma20": "SPREAD",
    "range_z20": "SPREAD",
    # Volatility & Downside Risk
    "vol_5d": "VOLATILITY",
    "vol_20d": "VOLATILITY",
    "vol_ratio_5_20": "VOLATILITY",
    "downside_vol_20d": "VOLATILITY",
    "max_drawdown_20d": "VOLATILITY",
    "vol_of_vol": "VOLATILITY",
    # Volume & Turnover Dynamics
    "log_volume": "VOLUME_TURNOVER",
    "volume_z20": "VOLUME_TURNOVER",
    "log_turnover": "VOLUME_TURNOVER",
    "turnover_ratio": "VOLUME_TURNOVER",
    "relative_volume": "VOLUME_TURNOVER",
    # Returns, Gaps & Momentum
    "daily_return": "MOMENTUM_RETURN",
    "return_5d": "MOMENTUM_RETURN",
    "return_20d": "MOMENTUM_RETURN",
    "return_z20": "MOMENTUM_RETURN",
    "abs_return": "MOMENTUM_RETURN",
    "price_rel_ma20": "MOMENTUM_RETURN",
    "rsi_14": "MOMENTUM_RETURN",
    "overnight_gap": "MOMENTUM_RETURN",
    "overnight_gap_abs": "MOMENTUM_RETURN",
    "excess_return_1d": "MOMENTUM_RETURN",
    # Macro, Benchmark & Systemic Conditioning
    "rolling_beta_60d": "MACRO_MARKET",
    "rolling_corr_60d": "MACRO_MARKET",
    "nifty_vol_20d": "MACRO_MARKET",
    "nifty_return_20d": "MACRO_MARKET",
    "vol_beta": "MACRO_MARKET",
    "vix_close": "MACRO_MARKET",
}

# Domain interpretations for key features in Indian equities
FEATURE_INTERPRETATIONS: Dict[str, Tuple[str, str]] = {
    "amihud_ma20": (
        "20-day moving average of price impact per unit of trading turnover.",
        "Persistent illiquidity strongly elevates probability of future liquidity freeze.",
    ),
    "cs_spread_ma10": (
        "10-day moving average of Corwin-Schultz estimated bid-ask spread.",
        "Elevated baseline transaction costs reflect wide market-maker quotes.",
    ),
    "downside_vol_20d": (
        "20-day realized downside semivariance (negative return volatility).",
        "Asymmetric downside selling pressure directly signals liquidation stress.",
    ),
    "log_volume": (
        "Log-transformed total daily traded share volume.",
        "Thinly traded equities possess lower market-maker depth, amplifying freeze risk.",
    ),
    "vix_close": (
        "India VIX index closing value reflecting market-wide implied volatility.",
        "Macro regime indicator that conditions institutional hedging and margin constraints.",
    ),
    "vol_20d": (
        "20-day realized volatility of daily log returns.",
        "Total price variability increases execution uncertainty and inventory holding costs.",
    ),
    "max_drawdown_20d": (
        "Maximum peak-to-trough price decline over the preceding 20 trading sessions.",
        "Severe drawdowns trigger margin calls and stop-loss sell cascades.",
    ),
    "relative_volume": (
        "Ratio of current trading volume relative to stock's 20-day historical mean.",
        "Volume dry-ups indicate buyers stepping back, causing order book thinning.",
    ),
    "range_z20": (
        "Standardized intraday high-low range relative to its 20-day rolling distribution.",
        "Intraday price whipsaws signal unstable equilibrium in market-maker inventory.",
    ),
    "amihud_ratio": (
        "Single-day ratio of absolute return to turnover value.",
        "Immediate price sensitivity to order flow on the active trading day.",
    ),
}


def compute_hgbt_permutation_importance(
    model,
    features_df: pd.DataFrame,
    feature_names: List[str],
    target_col: str = "hre_5d",
    scoring: str = "average_precision",
    n_repeats: int = 5,
    random_state: int = 42,
    sample_size: Optional[int] = 25000,
    regime_label: str = "OUT_OF_SAMPLE_WALK_FORWARD",
) -> pd.DataFrame:
    """Compute permutation feature importance for CFG_05_HGBT.

    Measures the deterioration in PR-AUC (average precision) when each feature
    is randomly shuffled across samples.

    Parameters
    ----------
    model : Pipeline or HistGradientBoostingClassifier
        Fitted model artifact.
    features_df : pd.DataFrame
        Evaluation dataset containing features and target column.
    feature_names : List[str]
        Exact ordered list of 36 features.
    target_col : str
        Target label column name ('hre_5d').
    scoring : str
        Scoring metric name (default 'average_precision').
    n_repeats : int
        Number of permutation repeats (default 5).
    random_state : int
        Seed for reproducibility.
    sample_size : Optional[int]
        If specified, samples N rows deterministically for stable, efficient computation.
    regime_label : str
        Dataset provenance description.

    Returns
    -------
    pd.DataFrame
        Table with feature, importance_mean, importance_std, rank, family, etc.
    """
    clean_df = features_df.dropna(subset=[target_col]).copy()
    if sample_size is not None and len(clean_df) > sample_size:
        clean_df = clean_df.sample(n=sample_size, random_state=random_state)

    X = clean_df[feature_names]
    y = clean_df[target_col].astype(int)

    # Run scikit-learn permutation importance
    res = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )

    records = []
    for i, feat in enumerate(feature_names):
        records.append({
            "feature": feat,
            "feature_family": FEATURE_FAMILY_MAP.get(feat, "OTHER"),
            "importance_mean": float(res.importances_mean[i]),
            "importance_std": float(res.importances_std[i]),
            "scoring_metric": scoring,
            "n_repeats": n_repeats,
            "sample_size": len(clean_df),
            "regime_label": regime_label,
        })

    df = pd.DataFrame(records)
    # Sort descending by importance_mean
    df = df.sort_values(by="importance_mean", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def compute_rf_mdi_importance(
    model,
    feature_names: List[str],
    model_name: str = "CFG_04_RF_CANDIDATE_B",
) -> pd.DataFrame:
    """Extract native Mean Decrease in Impurity (MDI) feature importances from Random Forest.

    Parameters
    ----------
    model : Pipeline or RandomForestClassifier
        Fitted Random Forest model artifact.
    feature_names : List[str]
        Exact ordered list of 36 features.
    model_name : str
        Model identifier.

    Returns
    -------
    pd.DataFrame
        Table with feature, importance_mdi, rank, family.
    """
    # Extract classifier step if Pipeline
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        estimator = model.named_steps["classifier"]
    else:
        estimator = model

    if not hasattr(estimator, "feature_importances_"):
        raise ValueError("Estimator does not expose feature_importances_.")

    importances = estimator.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError(f"Length mismatch: {len(importances)} importances vs {len(feature_names)} features.")

    records = []
    for feat, imp in zip(feature_names, importances):
        records.append({
            "feature": feat,
            "feature_family": FEATURE_FAMILY_MAP.get(feat, "OTHER"),
            "importance_mdi": float(imp),
            "model_name": model_name,
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by="importance_mdi", ascending=False).reset_index(drop=True)
    df["rank_mdi"] = df.index + 1
    return df


def compare_feature_importances(
    hgbt_imp_df: pd.DataFrame,
    rf_imp_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge and compare HGBT permutation importance with RF MDI importance.

    Parameters
    ----------
    hgbt_imp_df : pd.DataFrame
        Output of compute_hgbt_permutation_importance.
    rf_imp_df : pd.DataFrame
        Output of compute_rf_mdi_importance.

    Returns
    -------
    pd.DataFrame
        Comparison table with ranks, rank differences, and agreements.
    """
    merged = pd.merge(
        hgbt_imp_df[["feature", "feature_family", "importance_mean", "importance_std", "rank"]],
        rf_imp_df[["feature", "importance_mdi", "rank_mdi"]],
        on="feature",
        suffixes=("_hgbt", "_rf"),
    )
    merged = merged.rename(columns={"rank": "rank_hgbt"})
    merged["rank_diff"] = (merged["rank_hgbt"] - merged["rank_mdi"]).abs()
    merged = merged.sort_values(by="rank_hgbt").reset_index(drop=True)
    return merged


def extract_top_10_risk_drivers(
    hgbt_imp_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract top 10 risk drivers with detailed financial interpretations.

    Parameters
    ----------
    hgbt_imp_df : pd.DataFrame
        HGBT permutation importance DataFrame.

    Returns
    -------
    pd.DataFrame
        Top 10 table formatted for research reporting.
    """
    top10 = hgbt_imp_df.head(10).copy()

    records = []
    for _, row in top10.iterrows():
        feat = row["feature"]
        desc, fin_meaning = FEATURE_INTERPRETATIONS.get(
            feat,
            (f"Engineered microstructure metric for {feat}.", "Contributes predictive signal to risk estimation."),
        )

        records.append({
            "rank": row["rank"],
            "feature": feat,
            "family": row["feature_family"],
            "importance_mean_prauc_loss": row["importance_mean"],
            "importance_std": row["importance_std"],
            "feature_description": desc,
            "financial_interpretation": fin_meaning,
            "methodological_limitations": "Permutation attribution reflects marginal predictive loss, not causal causality.",
        })

    return pd.DataFrame(records)


def generate_explainability_figures(
    hgbt_imp_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    out_dir: Path,
):
    """Generate publication-quality feature importance and comparison charts.

    Parameters
    ----------
    hgbt_imp_df : pd.DataFrame
        HGBT permutation importance DataFrame.
    comparison_df : pd.DataFrame
        Feature importance comparison DataFrame.
    out_dir : Path
        Directory to save figures.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Top 15 HGBT Permutation Importance with Error Bars
    top15 = hgbt_imp_df.head(15).iloc[::-1]  # Reverse for horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color code by family
    family_colors = {
        "ILLIQUIDITY": "#d62728",
        "SPREAD": "#ff7f0e",
        "VOLATILITY": "#9467bd",
        "VOLUME_TURNOVER": "#1f77b4",
        "MOMENTUM_RETURN": "#2ca02c",
        "MACRO_MARKET": "#8c564b",
    }
    bar_colors = [family_colors.get(f, "#7f7f7f") for f in top15["feature_family"]]

    bars = ax.barh(
        top15["feature"],
        top15["importance_mean"],
        xerr=top15["importance_std"],
        color=bar_colors,
        alpha=0.85,
        capsize=4,
        edgecolor="black",
    )

    ax.set_xlabel("Permutation Importance: Mean PR-AUC Loss (Drop in Average Precision)", fontsize=11, fontweight="bold")
    ax.set_title("CFG_05_HGBT Champion: Top 15 Predictive Risk Drivers (Out-of-Sample Walk-Forward)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    # Add legend for families
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, edgecolor="black", label=fam)
        for fam, c in family_colors.items()
        if fam in top15["feature_family"].values
    ]
    ax.legend(handles=legend_elements, title="Microstructure Family", frameon=True, loc="lower right")

    plt.tight_layout()
    fig1_path = out_dir / "hgbt_permutation_importance.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Ranking Comparison HGBT Permutation vs RF MDI
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        comparison_df["rank_hgbt"],
        comparison_df["rank_mdi"],
        s=80,
        c="#1f77b4",
        alpha=0.8,
        edgecolors="black",
    )

    # 45-degree diagonal line
    max_rank = max(comparison_df["rank_hgbt"].max(), comparison_df["rank_mdi"].max())
    ax.plot([1, max_rank], [1, max_rank], color="red", linestyle="--", alpha=0.7, label="Perfect Rank Agreement")

    # Label top features
    for _, row in comparison_df.head(10).iterrows():
        ax.annotate(
            row["feature"],
            xy=(row["rank_hgbt"], row["rank_mdi"]),
            xytext=(6, 2),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlabel("HGBT Permutation Importance Rank (1 = Most Critical)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Random Forest MDI Importance Rank (1 = Most Critical)", fontsize=11, fontweight="bold")
    ax.set_title("Cross-Model Feature Importance Concordance: HGBT vs. Random Forest Candidate B", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)
    ax.set_xlim(0, max_rank + 2)
    ax.set_ylim(0, max_rank + 2)

    plt.tight_layout()
    fig2_path = out_dir / "importance_comparison_hgbt_vs_rf.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig2_path}")
