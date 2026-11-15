"""Unit tests for Phase 9.3 Macro Stress Testing & Segment Divergence."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.phase9.macro_stress import (
    REPRESENTATIVE_STRESS_EPISODES,
    StressEpisode,
    compute_isolation_forest_stress_clustering,
    compute_segment_divergence,
    compute_stress_period_metrics,
    define_stress_episodes,
    run_statistical_significance_tests,
)


@pytest.fixture
def mock_event_stocks_df() -> pd.DataFrame:
    """Create a realistic mock stock-level event observations dataset."""
    rng = np.random.default_rng(42)
    tickers = [f"TICKER_{i:03d}" for i in range(30)]
    records = []

    for ep in REPRESENTATIVE_STRESS_EPISODES[:3]:
        for evt_id in ep.event_ids:
            for rel_day in range(-5, 6):
                period = "PRE" if rel_day < 0 else ("DURING" if rel_day == 0 else "POST")
                for i, ticker in enumerate(tickers):
                    # Assign segment
                    seg = "LARGE" if i < 10 else ("MID" if i < 20 else "SMALL")
                    # Small caps have higher spread and larger stress shock
                    base_spread = 0.002 if seg == "LARGE" else (0.005 if seg == "MID" else 0.012)
                    shock_mult = 1.2 if seg == "LARGE" else (1.6 if seg == "MID" else 2.5)

                    spread = base_spread * (shock_mult if rel_day == 0 else (1.1 if rel_day > 0 else 1.0))
                    spread += rng.normal(0, 0.0002)

                    amihud = 0.05 * shock_mult if rel_day == 0 else 0.05
                    prob_hgbt = 0.2 if rel_day < 0 else (0.45 if rel_day == 0 else 0.25)
                    iso_flag = 1 if (rel_day == 0 and rng.random() < 0.6) else (1 if rng.random() < 0.15 else 0)

                    records.append({
                        "event_id": evt_id,
                        "relative_day": rel_day,
                        "period": period,
                        "ticker": ticker,
                        "cap_segment": seg,
                        "cs_spread_est": max(0.0001, spread),
                        "amihud_ratio": max(0.001, amihud),
                        "vol_20d": 0.02,
                        "downside_vol_20d": 0.015,
                        "volume_ratio": 1.0,
                        "prob_hist_gradient_boosting": prob_hgbt,
                        "anomaly_score_scaled": 50.0 if rel_day == 0 else 40.0,
                        "isolation_flag": iso_flag,
                    })

    return pd.DataFrame(records)


def test_define_stress_episodes():
    """Verify stress episodes catalog structure and content."""
    df = define_stress_episodes()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(REPRESENTATIVE_STRESS_EPISODES)
    assert set(df.columns) >= {"episode_id", "episode_name", "category", "event_ids", "regime"}
    assert all(df["num_events"] > 0)
    assert set(df["regime"].unique()) == {"IN_SAMPLE", "OUT_OF_SAMPLE"}


def test_compute_stress_period_metrics(mock_event_stocks_df):
    """Verify stress period metrics aggregation across pre, during, post and segments."""
    metrics_df = compute_stress_period_metrics(
        mock_event_stocks_df,
        episodes=REPRESENTATIVE_STRESS_EPISODES[:2],
    )
    assert not metrics_df.empty
    assert "period" in metrics_df.columns
    assert "segment" in metrics_df.columns
    assert set(metrics_df["period"].unique()) == {"PRE", "DURING", "POST"}
    assert set(metrics_df["segment"].unique()) == {"LARGE", "MID", "SMALL", "MARKET_WIDE"}

    # Verify summary statistics presence
    for m in ["cs_spread_est", "amihud_ratio", "prob_hist_gradient_boosting"]:
        assert f"{m}_median" in metrics_df.columns
        assert f"{m}_iqr" in metrics_df.columns
        assert f"{m}_q25" in metrics_df.columns
        assert f"{m}_q75" in metrics_df.columns

    # Verify that Small caps have higher spread than Large caps
    dur_small = metrics_df[(metrics_df["period"] == "DURING") & (metrics_df["segment"] == "SMALL")]["cs_spread_est_median"].iloc[0]
    dur_large = metrics_df[(metrics_df["period"] == "DURING") & (metrics_df["segment"] == "LARGE")]["cs_spread_est_median"].iloc[0]
    assert dur_small > dur_large


def test_compute_segment_divergence(mock_event_stocks_df):
    """Verify market-cap segment divergence computation."""
    metrics_df = compute_stress_period_metrics(
        mock_event_stocks_df,
        episodes=REPRESENTATIVE_STRESS_EPISODES[:2],
    )
    div_df = compute_segment_divergence(metrics_df)
    assert not div_df.empty
    assert "DIVERGENCE_SMALL_VS_LARGE" in div_df["segment"].values

    div_rows = div_df[div_df["segment"] == "DIVERGENCE_SMALL_VS_LARGE"]
    assert len(div_rows) == 2
    # Small caps had larger shock in mock data, so divergence should be positive
    assert all(div_rows["spread_shock_abs"] > 0)


def test_run_statistical_significance_tests(mock_event_stocks_df):
    """Verify non-parametric statistical tests (Mann-Whitney U and Wilcoxon)."""
    test_res = run_statistical_significance_tests(
        mock_event_stocks_df,
        episodes=REPRESENTATIVE_STRESS_EPISODES[:2],
        n_bootstrap=200,
        random_state=42,
    )
    assert not test_res.empty
    assert set(test_res["test_type"].unique()) == {"PAIRED_WILCOXON_AND_BOOTSTRAP_CI", "MANN_WHITNEY_U_DIVERGENCE"}

    # Verify p-values are valid probabilities in [0, 1]
    p_vals = test_res["p_value"].dropna()
    assert (p_vals >= 0.0).all() and (p_vals <= 1.0).all()

    # Verify bootstrap CI lower <= upper
    wilc_rows = test_res[test_res["test_type"] == "PAIRED_WILCOXON_AND_BOOTSTRAP_CI"]
    for _, r in wilc_rows.iterrows():
        if not np.isnan(r["ci_95_lower"]):
            assert r["ci_95_lower"] <= r["ci_95_upper"]


def test_compute_isolation_forest_stress_clustering(mock_event_stocks_df):
    """Verify Isolation Forest anomaly clustering calculations."""
    if_df = compute_isolation_forest_stress_clustering(
        mock_event_stocks_df,
        episodes=REPRESENTATIVE_STRESS_EPISODES[:2],
    )
    assert not if_df.empty
    assert "pre_flag_rate_pct" in if_df.columns
    assert "during_flag_rate_pct" in if_df.columns
    assert "stress_enrichment_ratio" in if_df.columns

    # In mock data, during-stress had 60% flags vs 15% pre, so enrichment should be > 1.0
    mkt_rows = if_df[if_df["segment"] == "MARKET_WIDE"]
    assert all(mkt_rows["during_flag_rate_pct"] > mkt_rows["pre_flag_rate_pct"])
    assert all(mkt_rows["stress_enrichment_ratio"] > 1.0)
