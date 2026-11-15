"""Macro Stress Testing & Market-Cap Liquidity Divergence Analysis (Phase 9.3).

This module implements cross-sectional stress testing of equity liquidity across
market-capitalization tiers (Large, Mid, Small caps) during major historical
macroeconomic and systemic market disruptions in Indian equities.

Key components:
1. Stress episode definitions and mapping across market events.
2. Pre vs. During vs. Post period liquidity and model metric aggregation.
3. Market-cap segment divergence quantification (differential spread widening).
4. Non-parametric statistical testing (Mann-Whitney U, Wilcoxon signed-rank, Bootstrap CIs).
5. Isolation Forest anomaly flag clustering and co-movement analysis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StressEpisode:
    """Specification of a historical macro/systemic stress episode."""
    episode_id: str
    episode_name: str
    category: str
    event_ids: List[str]
    description: str
    regime: str  # 'IN_SAMPLE' or 'OUT_OF_SAMPLE'


# Authoritative representative historical stress episodes curated from 86 market events
REPRESENTATIVE_STRESS_EPISODES: List[StressEpisode] = [
    StressEpisode(
        episode_id="EP_01_COVID_CRASH",
        episode_name="COVID-19 Pandemic Lower Circuit Halts",
        category="REGULATORY_EXCHANGE",
        event_ids=["EVT_2020_005", "EVT_2020_007", "EVT_2020_008"],
        description="First and second 10% lower circuit market trading halts and emergency short-selling curbs (March 2020).",
        regime="IN_SAMPLE",
    ),
    StressEpisode(
        episode_id="EP_02_ILFS_NBFC_CRISIS",
        episode_name="IL&FS Default & NBFC Liquidity Freeze",
        category="SYSTEMIC_CREDIT",
        event_ids=["EVT_2018_004", "EVT_2018_005"],
        description="IL&FS commercial paper default triggering NBFC Black Monday market panic (Sept 2018).",
        regime="IN_SAMPLE",
    ),
    StressEpisode(
        episode_id="EP_03_UKRAINE_INVASION",
        episode_name="Russia-Ukraine Invasion Market Shock",
        category="GEOPOLITICAL",
        event_ids=["EVT_2022_003"],
        description="Russian military invasion of Ukraine, commodity spike, and 4.8% single-day index drop (Feb 2022).",
        regime="IN_SAMPLE",
    ),
    StressEpisode(
        episode_id="EP_04_ADANI_HINDENBURG",
        episode_name="Adani-Hindenburg Governance & FPO Shock",
        category="MACRO_POLICY",
        event_ids=["EVT_2023_001", "EVT_2023_003"],
        description="Hindenburg Research report release and Adani Enterprises FPO cancellation (Jan-Feb 2023).",
        regime="IN_SAMPLE",
    ),
    StressEpisode(
        episode_id="EP_05_ELECTION_2024_SHOCK",
        episode_name="Lok Sabha 2024 Election Result Whiplash",
        category="POLITICAL_ELECTORAL",
        event_ids=["EVT_2024_007", "EVT_2024_008", "EVT_2024_009"],
        description="Exit poll euphoria (+3.4%) followed by counting day shock plunge (-5.9%) and coalition rebound (June 2024).",
        regime="OUT_OF_SAMPLE",
    ),
    StressEpisode(
        episode_id="EP_06_SEBI_FROTH_WARNING",
        episode_name="SEBI Small/Mid-Cap Froth Regulatory Warning",
        category="REGULATORY_EXCHANGE",
        event_ids=["EVT_2024_004"],
        description="SEBI Chairperson warning on irrational exuberance and froth in small/mid-cap mutual funds (March 2024).",
        regime="OUT_OF_SAMPLE",
    ),
    StressEpisode(
        episode_id="EP_07_YEN_CARRY_UNWIND",
        episode_name="Global Black Monday Yen Carry Trade Unwind",
        category="GEOPOLITICAL",
        event_ids=["EVT_2024_012"],
        description="Bank of Japan rate hike triggering global margin calls and multi-asset liquidation (Aug 2024).",
        regime="OUT_OF_SAMPLE",
    ),
]


def define_stress_episodes(
    events_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return a DataFrame cataloging representative stress episodes and their metadata.

    Parameters
    ----------
    events_df : Optional[pd.DataFrame]
        Optional events DataFrame to cross-reference event existence.

    Returns
    -------
    pd.DataFrame
        Table of stress episodes with episode_id, episode_name, category, event_ids,
        description, and regime.
    """
    records = []
    for ep in REPRESENTATIVE_STRESS_EPISODES:
        records.append({
            "episode_id": ep.episode_id,
            "episode_name": ep.episode_name,
            "category": ep.category,
            "event_ids": ",".join(ep.event_ids),
            "num_events": len(ep.event_ids),
            "description": ep.description,
            "regime": ep.regime,
        })
    df = pd.DataFrame(records)
    if events_df is not None and "event_id" in events_df.columns:
        known_ids = set(events_df["event_id"].unique())
        for ep in REPRESENTATIVE_STRESS_EPISODES:
            missing = set(ep.event_ids) - known_ids
            if missing:
                raise ValueError(f"Episode {ep.episode_id} references unknown events: {missing}")
    return df


def compute_stress_period_metrics(
    event_stocks_df: pd.DataFrame,
    episodes: Optional[List[StressEpisode]] = None,
) -> pd.DataFrame:
    """Aggregate stock observations into Pre, During, and Post periods for each stress episode.

    Computes cross-sectional median, 25th percentile, 75th percentile, and IQR
    for key liquidity and model metrics across Large, Mid, Small caps and Market-Wide.

    Parameters
    ----------
    event_stocks_df : pd.DataFrame
        Stock-level event observations from Stage 9.2 (event_stock_observations.parquet).
    episodes : Optional[List[StressEpisode]]
        Episodes to evaluate. Defaults to REPRESENTATIVE_STRESS_EPISODES.

    Returns
    -------
    pd.DataFrame
        Aggregated stress metrics table.
    """
    if episodes is None:
        episodes = REPRESENTATIVE_STRESS_EPISODES

    metrics_to_agg = [
        "cs_spread_est",
        "amihud_ratio",
        "vol_20d",
        "downside_vol_20d",
        "volume_ratio",
        "prob_hist_gradient_boosting",
        "anomaly_score_scaled",
        "isolation_flag",
    ]
    # Filter to metrics present in input
    available_metrics = [m for m in metrics_to_agg if m in event_stocks_df.columns]

    rows = []
    for ep in episodes:
        ep_df = event_stocks_df[event_stocks_df["event_id"].isin(ep.event_ids)].copy()
        if ep_df.empty:
            continue

        # Standardize period: support both PRE_EVENT/EVENT_DAY/POST_EVENT and PRE/DURING/POST
        period_map = {
            "PRE_EVENT": "PRE",
            "EVENT_DAY": "DURING",
            "POST_EVENT": "POST",
            "EVENT": "DURING",
        }
        if "period" in ep_df.columns:
            ep_df["period"] = ep_df["period"].replace(period_map)
        elif "relative_day" in ep_df.columns:
            ep_df["period"] = ep_df["relative_day"].apply(
                lambda r: "PRE" if r < 0 else ("DURING" if r == 0 else "POST")
            )

        if "cap_segment" in ep_df.columns:
            ep_df["cap_segment"] = ep_df["cap_segment"].astype(str).str.upper()
        else:
            ep_df["cap_segment"] = "MARKET_WIDE"

        # Groupings: By segment (Large, Mid, Small) and Market-Wide
        segments = ["LARGE", "MID", "SMALL"]

        for period in ["PRE", "DURING", "POST"]:
            period_sub = ep_df[ep_df["period"] == period]
            if period_sub.empty:
                continue

            for seg in segments + ["MARKET_WIDE"]:
                if seg == "MARKET_WIDE":
                    seg_sub = period_sub
                else:
                    seg_sub = period_sub[period_sub["cap_segment"] == seg]

                n_obs = len(seg_sub)
                n_stocks = seg_sub["ticker"].nunique() if "ticker" in seg_sub.columns else 0

                row = {
                    "episode_id": ep.episode_id,
                    "episode_name": ep.episode_name,
                    "category": ep.category,
                    "regime": ep.regime,
                    "period": period,
                    "segment": seg,
                    "n_obs": n_obs,
                    "n_stocks": n_stocks,
                }

                for m in available_metrics:
                    vals = seg_sub[m].dropna()
                    if len(vals) > 0:
                        med = float(np.median(vals))
                        q25 = float(np.percentile(vals, 25))
                        q75 = float(np.percentile(vals, 75))
                        iqr = float(q75 - q25)
                        mean_val = float(np.mean(vals))
                    else:
                        med = q25 = q75 = iqr = mean_val = np.nan

                    row[f"{m}_median"] = med
                    row[f"{m}_q25"] = q25
                    row[f"{m}_q75"] = q75
                    row[f"{m}_iqr"] = iqr
                    row[f"{m}_mean"] = mean_val

                rows.append(row)

    return pd.DataFrame(rows)


def compute_segment_divergence(
    stress_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute market-cap segment divergence across stress episodes.

    Quantifies the differential impact: Small-cap metric shock vs. Large-cap metric shock.
    Specifically measures:
    1. During vs. Pre spread change (basis points / absolute).
    2. During vs. Pre Amihud illiquidity % change.
    3. During vs. Pre HGBT high-risk probability surge.
    4. Post vs. During recovery ratio (post-stress reversion).
    5. Divergence: Small-cap shock minus Large-cap shock.

    Parameters
    ----------
    stress_metrics_df : pd.DataFrame
        Output of compute_stress_period_metrics.

    Returns
    -------
    pd.DataFrame
        Divergence summary table per episode and segment.
    """
    records = []
    episode_ids = stress_metrics_df["episode_id"].unique()

    for ep_id in episode_ids:
        ep_sub = stress_metrics_df[stress_metrics_df["episode_id"] == ep_id]
        ep_name = ep_sub["episode_name"].iloc[0]
        cat = ep_sub["category"].iloc[0]
        regime = ep_sub["regime"].iloc[0]

        # Get metrics by segment and period
        segment_data = {}
        for seg in ["LARGE", "MID", "SMALL", "MARKET_WIDE"]:
            seg_rows = ep_sub[ep_sub["segment"] == seg].set_index("period")
            segment_data[seg] = seg_rows

        # Calculate shocks for each segment
        seg_shocks = {}
        for seg in ["LARGE", "MID", "SMALL", "MARKET_WIDE"]:
            s_df = segment_data[seg]
            if "PRE" in s_df.index and "DURING" in s_df.index:
                pre_spread = s_df.loc["PRE", "cs_spread_est_median"]
                dur_spread = s_df.loc["DURING", "cs_spread_est_median"]
                spread_shock_abs = dur_spread - pre_spread
                spread_shock_pct = (dur_spread / pre_spread - 1.0) * 100.0 if pre_spread > 0 else np.nan

                pre_amihud = s_df.loc["PRE", "amihud_ratio_median"]
                dur_amihud = s_df.loc["DURING", "amihud_ratio_median"]
                amihud_shock_pct = (dur_amihud / pre_amihud - 1.0) * 100.0 if pre_amihud > 0 else np.nan

                pre_hgbt = s_df.loc["PRE", "prob_hist_gradient_boosting_median"]
                dur_hgbt = s_df.loc["DURING", "prob_hist_gradient_boosting_median"]
                hgbt_shock_abs = dur_hgbt - pre_hgbt

                pre_if = s_df.loc["PRE", "isolation_flag_mean"]
                dur_if = s_df.loc["DURING", "isolation_flag_mean"]
                if_shock_abs = dur_if - pre_if

                # Recovery ratio (post vs during): < 1.0 indicates mean-reversion
                if "POST" in s_df.index:
                    post_spread = s_df.loc["POST", "cs_spread_est_median"]
                    # If spread widened, how much of that widening unwound by post-window?
                    reversion_pct = (
                        (dur_spread - post_spread) / spread_shock_abs * 100.0
                        if abs(spread_shock_abs) > 1e-6 else np.nan
                    )
                else:
                    reversion_pct = np.nan

                seg_shocks[seg] = {
                    "pre_spread_median": pre_spread,
                    "during_spread_median": dur_spread,
                    "spread_shock_abs": spread_shock_abs,
                    "spread_shock_pct": spread_shock_pct,
                    "pre_amihud_median": pre_amihud,
                    "during_amihud_median": dur_amihud,
                    "amihud_shock_pct": amihud_shock_pct,
                    "hgbt_shock_abs": hgbt_shock_abs,
                    "if_flag_shock_abs": if_shock_abs,
                    "spread_reversion_pct": reversion_pct,
                }
            else:
                seg_shocks[seg] = {}

        # Add rows for each segment
        for seg in ["LARGE", "MID", "SMALL", "MARKET_WIDE"]:
            shk = seg_shocks.get(seg, {})
            row = {
                "episode_id": ep_id,
                "episode_name": ep_name,
                "category": cat,
                "regime": regime,
                "segment": seg,
                **shk,
            }
            records.append(row)

        # Cross-segment divergence (Small minus Large)
        small_shk = seg_shocks.get("SMALL", {})
        large_shk = seg_shocks.get("LARGE", {})
        if small_shk and large_shk:
            div_spread_abs = small_shk.get("spread_shock_abs", np.nan) - large_shk.get("spread_shock_abs", np.nan)
            div_amihud_pct = small_shk.get("amihud_shock_pct", np.nan) - large_shk.get("amihud_shock_pct", np.nan)
            div_hgbt_abs = small_shk.get("hgbt_shock_abs", np.nan) - large_shk.get("hgbt_shock_abs", np.nan)
            records.append({
                "episode_id": ep_id,
                "episode_name": ep_name,
                "category": cat,
                "regime": regime,
                "segment": "DIVERGENCE_SMALL_VS_LARGE",
                "spread_shock_abs": div_spread_abs,
                "spread_shock_pct": np.nan,
                "amihud_shock_pct": div_amihud_pct,
                "hgbt_shock_abs": div_hgbt_abs,
                "if_flag_shock_abs": small_shk.get("if_flag_shock_abs", np.nan) - large_shk.get("if_flag_shock_abs", np.nan),
                "spread_reversion_pct": np.nan,
            })

    return pd.DataFrame(records)


def run_statistical_significance_tests(
    event_stocks_df: pd.DataFrame,
    episodes: Optional[List[StressEpisode]] = None,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run non-parametric statistical tests comparing liquidity shocks across segments.

    1. Mann-Whitney U test comparing stock-level During vs Pre spread changes between Small and Large caps.
    2. Wilcoxon signed-rank test for paired (During - Pre) spread changes within each market-cap segment.
    3. Bootstrap 95% Confidence Intervals for median spread shock by segment.

    Parameters
    ----------
    event_stocks_df : pd.DataFrame
        Stock-level event observations.
    episodes : Optional[List[StressEpisode]]
        Episodes to evaluate.
    n_bootstrap : int
        Number of bootstrap iterations (default 1000).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Statistical test results and confidence intervals.
    """
    if episodes is None:
        episodes = REPRESENTATIVE_STRESS_EPISODES

    rng = np.random.default_rng(random_state)
    results = []

    for ep in episodes:
        ep_df = event_stocks_df[event_stocks_df["event_id"].isin(ep.event_ids)].copy()
        if ep_df.empty:
            continue

        period_map = {
            "PRE_EVENT": "PRE",
            "EVENT_DAY": "DURING",
            "POST_EVENT": "POST",
            "EVENT": "DURING",
        }
        if "period" in ep_df.columns:
            ep_df["period"] = ep_df["period"].replace(period_map)
        elif "relative_day" in ep_df.columns:
            ep_df["period"] = ep_df["relative_day"].apply(
                lambda r: "PRE" if r < 0 else ("DURING" if r == 0 else "POST")
            )

        if "cap_segment" in ep_df.columns:
            ep_df["cap_segment"] = ep_df["cap_segment"].astype(str).str.upper()
        else:
            ep_df["cap_segment"] = "MARKET_WIDE"

        # Pivot to compute per-stock During minus Pre change
        pre_stock = (
            ep_df[ep_df["period"] == "PRE"]
            .groupby(["ticker", "cap_segment"], as_index=False)[["cs_spread_est", "amihud_ratio", "prob_hist_gradient_boosting"]]
            .median()
        )
        during_stock = (
            ep_df[ep_df["period"] == "DURING"]
            .groupby(["ticker", "cap_segment"], as_index=False)[["cs_spread_est", "amihud_ratio", "prob_hist_gradient_boosting"]]
            .median()
        )

        merged = pd.merge(
            pre_stock, during_stock,
            on=["ticker", "cap_segment"],
            suffixes=("_pre", "_during"),
        )
        if merged.empty:
            continue

        merged["delta_spread"] = merged["cs_spread_est_during"] - merged["cs_spread_est_pre"]
        merged["delta_amihud"] = merged["amihud_ratio_during"] - merged["amihud_ratio_pre"]
        merged["delta_hgbt"] = merged["prob_hist_gradient_boosting_during"] - merged["prob_hist_gradient_boosting_pre"]

        # 1. Segment-level paired Wilcoxon tests & Bootstrap CIs
        for seg in ["LARGE", "MID", "SMALL"]:
            seg_data = merged[merged["cap_segment"] == seg]
            n_stocks = len(seg_data)
            if n_stocks < 5:
                continue

            spread_deltas = seg_data["delta_spread"].dropna().values
            if len(spread_deltas) >= 5 and not np.all(spread_deltas == 0):
                # Paired Wilcoxon test against null of zero median change
                try:
                    w_res = stats.wilcoxon(spread_deltas, alternative="greater")
                    wilc_stat, wilc_pval = float(w_res.statistic), float(w_res.pvalue)
                except Exception:
                    wilc_stat, wilc_pval = np.nan, np.nan
            else:
                wilc_stat, wilc_pval = np.nan, np.nan

            # Bootstrap 95% CI for median delta_spread
            if len(spread_deltas) >= 5:
                boot_medians = [
                    np.median(rng.choice(spread_deltas, size=len(spread_deltas), replace=True))
                    for _ in range(n_bootstrap)
                ]
                ci_lower = float(np.percentile(boot_medians, 2.5))
                ci_upper = float(np.percentile(boot_medians, 97.5))
                observed_median = float(np.median(spread_deltas))
            else:
                ci_lower, ci_upper, observed_median = np.nan, np.nan, np.nan

            results.append({
                "episode_id": ep.episode_id,
                "episode_name": ep.episode_name,
                "category": ep.category,
                "regime": ep.regime,
                "test_type": "PAIRED_WILCOXON_AND_BOOTSTRAP_CI",
                "comparison": f"{seg}_DURING_VS_PRE",
                "segment": seg,
                "sample_size": n_stocks,
                "statistic": wilc_stat,
                "p_value": wilc_pval,
                "observed_delta_median": observed_median,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "is_significant_005": bool(wilc_pval < 0.05) if not np.isnan(wilc_pval) else False,
            })

        # 2. Mann-Whitney U test: Small-cap vs Large-cap differential shock
        small_deltas = merged[merged["cap_segment"] == "SMALL"]["delta_spread"].dropna().values
        large_deltas = merged[merged["cap_segment"] == "LARGE"]["delta_spread"].dropna().values

        if len(small_deltas) >= 5 and len(large_deltas) >= 5:
            mwu_res = stats.mannwhitneyu(small_deltas, large_deltas, alternative="greater")
            mwu_stat, mwu_pval = float(mwu_res.statistic), float(mwu_res.pvalue)
            small_med = float(np.median(small_deltas))
            large_med = float(np.median(large_deltas))
            diff_med = small_med - large_med

            results.append({
                "episode_id": ep.episode_id,
                "episode_name": ep.episode_name,
                "category": ep.category,
                "regime": ep.regime,
                "test_type": "MANN_WHITNEY_U_DIVERGENCE",
                "comparison": "SMALL_VS_LARGE_DELTA_SPREAD",
                "segment": "SMALL_VS_LARGE",
                "sample_size": len(small_deltas) + len(large_deltas),
                "statistic": mwu_stat,
                "p_value": mwu_pval,
                "observed_delta_median": diff_med,
                "ci_95_lower": np.nan,
                "ci_95_upper": np.nan,
                "is_significant_005": bool(mwu_pval < 0.05),
            })

    return pd.DataFrame(results)


def compute_isolation_forest_stress_clustering(
    event_stocks_df: pd.DataFrame,
    episodes: Optional[List[StressEpisode]] = None,
) -> pd.DataFrame:
    """Analyze Isolation Forest anomaly flag concentration during stress episodes.

    Computes:
    - Universe anomaly flag rate (%) during Pre vs. During vs. Post periods.
    - Anomaly rate breakdown across Large, Mid, Small caps.
    - Ratio of stress-period anomaly rate to baseline pre-event anomaly rate (enrichment factor).

    Parameters
    ----------
    event_stocks_df : pd.DataFrame
        Stock-level event observations.
    episodes : Optional[List[StressEpisode]]
        Episodes to evaluate.

    Returns
    -------
    pd.DataFrame
        Clustering metrics table.
    """
    if episodes is None:
        episodes = REPRESENTATIVE_STRESS_EPISODES

    records = []
    for ep in episodes:
        ep_df = event_stocks_df[event_stocks_df["event_id"].isin(ep.event_ids)].copy()
        if ep_df.empty or "isolation_flag" not in ep_df.columns:
            continue

        period_map = {
            "PRE_EVENT": "PRE",
            "EVENT_DAY": "DURING",
            "POST_EVENT": "POST",
            "EVENT": "DURING",
        }
        if "period" in ep_df.columns:
            ep_df["period"] = ep_df["period"].replace(period_map)
        elif "relative_day" in ep_df.columns:
            ep_df["period"] = ep_df["relative_day"].apply(
                lambda r: "PRE" if r < 0 else ("DURING" if r == 0 else "POST")
            )

        if "cap_segment" in ep_df.columns:
            ep_df["cap_segment"] = ep_df["cap_segment"].astype(str).str.upper()
        else:
            ep_df["cap_segment"] = "MARKET_WIDE"

        for seg in ["MARKET_WIDE", "LARGE", "MID", "SMALL"]:
            seg_df = ep_df if seg == "MARKET_WIDE" else ep_df[ep_df["cap_segment"] == seg]
            if seg_df.empty:
                continue

            pre_sub = seg_df[seg_df["period"] == "PRE"]
            dur_sub = seg_df[seg_df["period"] == "DURING"]
            post_sub = seg_df[seg_df["period"] == "POST"]

            pre_rate = float(pre_sub["isolation_flag"].mean() * 100.0) if len(pre_sub) > 0 else np.nan
            dur_rate = float(dur_sub["isolation_flag"].mean() * 100.0) if len(dur_sub) > 0 else np.nan
            post_rate = float(post_sub["isolation_flag"].mean() * 100.0) if len(post_sub) > 0 else np.nan

            enrichment = (dur_rate / pre_rate) if (pre_rate and pre_rate > 0) else np.nan

            records.append({
                "episode_id": ep.episode_id,
                "episode_name": ep.episode_name,
                "category": ep.category,
                "regime": ep.regime,
                "segment": seg,
                "pre_flag_rate_pct": pre_rate,
                "during_flag_rate_pct": dur_rate,
                "post_flag_rate_pct": post_rate,
                "stress_enrichment_ratio": enrichment,
                "flag_rate_surge_pp": (dur_rate - pre_rate) if not (np.isnan(dur_rate) or np.isnan(pre_rate)) else np.nan,
            })

    return pd.DataFrame(records)
