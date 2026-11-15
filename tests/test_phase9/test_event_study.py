"""Unit tests for Phase 9 Event Study module."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.modeling.phase9.event_study import (
    build_event_windows,
    compute_cross_sectional_event_metrics,
    compute_event_impact_summary,
    load_and_cluster_events,
)


@pytest.fixture
def mock_calendar():
    """Returns 25 consecutive trading dates."""
    return pd.date_range("2024-01-01", periods=25, freq="B").tolist()


@pytest.fixture
def sample_events_df(mock_calendar):
    """Returns sample clustered events."""
    return pd.DataFrame(
        {
            "event_id": ["EVT_001", "EVT_002", "EVT_003"],
            "event_name": ["Shock 1", "Shock 2 (Cluster)", "Shock 3 (Isolated)"],
            "event_category": ["MACRO_POLICY", "GEOPOLITICAL", "SYSTEMIC_CREDIT"],
            "scope": ["MARKET_WIDE", "MARKET_WIDE", "MARKET_WIDE"],
            "is_scheduled": [True, False, False],
            "effective_trading_date": [mock_calendar[5], mock_calendar[7], mock_calendar[15]],
            "cal_idx": [5, 7, 15],
            "event_cluster_id": ["CLUSTER_01", "CLUSTER_01", "ISOLATED_EVT_003"],
            "is_overlapping_event": [True, True, False],
        }
    )


def test_load_and_cluster_events_real_data():
    """Tests loading the actual market events calendar."""
    events = load_and_cluster_events("data/events/market_events.csv")
    assert len(events) == 86
    assert "event_cluster_id" in events.columns
    assert "is_overlapping_event" in events.columns
    assert events["is_overlapping_event"].sum() > 0
    assert (~events["is_overlapping_event"]).sum() > 0


def test_build_event_windows(sample_events_df, mock_calendar):
    """Tests [-5, +5] event window generation."""
    windows = build_event_windows(sample_events_df, mock_calendar, window_pre=5, window_post=5)

    # 3 events * 11 days = 33 window records
    assert len(windows) == 33
    assert set(windows["relative_day"]) == set(range(-5, 6))

    # Check periods
    assert set(windows[windows["relative_day"] < 0]["period"]) == {"PRE_EVENT"}
    assert set(windows[windows["relative_day"] == 0]["period"]) == {"EVENT_DAY"}
    assert set(windows[windows["relative_day"] > 0]["period"]) == {"POST_EVENT"}

    # Relative day 0 matches effective_trading_date
    evt1_windows = windows[windows["event_id"] == "EVT_001"]
    day0 = evt1_windows[evt1_windows["relative_day"] == 0].iloc[0]
    assert day0["session_date"] == sample_events_df.iloc[0]["effective_trading_date"]


def test_compute_cross_sectional_metrics():
    """Tests median, IQR, and market-cap stratified aggregation."""
    # Create synthetic stock observations for 1 event day
    data = []
    for ticker_idx in range(30):
        cap = "large" if ticker_idx < 10 else ("mid" if ticker_idx < 20 else "small")
        data.append(
            {
                "event_id": "EVT_TEST",
                "event_name": "Test Event",
                "event_category": "MACRO_POLICY",
                "scope": "MARKET_WIDE",
                "is_scheduled": True,
                "is_overlapping_event": False,
                "event_cluster_id": "ISOLATED_EVT_TEST",
                "prediction_regime": "OUT_OF_SAMPLE_WF",
                "period": "EVENT_DAY",
                "relative_day": 0,
                "session_date": pd.Timestamp("2024-06-04"),
                "ticker": f"TICKER_{ticker_idx}",
                "cap_segment": cap,
                "cs_spread_est": 0.01 * (ticker_idx + 1),
                "amihud_ratio": 0.0001 * (ticker_idx + 1),
                "prob_hist_gradient_boosting": 0.50 + 0.01 * ticker_idx,
                "hre_5d": 1 if ticker_idx % 2 == 0 else 0,
            }
        )
    df = pd.DataFrame(data)

    cs_summary = compute_cross_sectional_event_metrics(
        df, metrics=["cs_spread_est", "amihud_ratio", "prob_hist_gradient_boosting", "hre_5d"]
    )

    # Should contain 3 cap tiers + market_wide = 4 segments
    assert len(cs_summary) == 4
    assert set(cs_summary["segment"]) == {"large", "mid", "small", "market_wide"}

    # Verify median and IQR
    mw = cs_summary[cs_summary["segment"] == "market_wide"].iloc[0]
    assert mw["stock_count"] == 30
    assert mw["cs_spread_est_median"] > 0
    assert mw["cs_spread_est_iqr"] > 0
    assert mw["cs_spread_est_q75"] > mw["cs_spread_est_q25"]


def test_compute_event_impact_summary():
    """Tests summary impact calculation between pre-event, event day, and post-event."""
    # Synthetic cross-sectional summary with relative_day -5 to +5
    records = []
    for rel_day in range(-5, 6):
        records.append(
            {
                "event_id": "EVT_SAMPLE",
                "event_name": "Sample Shock",
                "event_category": "MACRO_POLICY",
                "scope": "MARKET_WIDE",
                "is_scheduled": True,
                "is_overlapping_event": False,
                "event_cluster_id": "ISOLATED_EVT_SAMPLE",
                "prediction_regime": "OUT_OF_SAMPLE_WF",
                "period": "PRE_EVENT" if rel_day < 0 else ("EVENT_DAY" if rel_day == 0 else "POST_EVENT"),
                "relative_day": rel_day,
                "session_date": pd.Timestamp("2024-06-04") + pd.Timedelta(days=rel_day),
                "segment": "market_wide",
                "stock_count": 300,
                "cs_spread_est_median": 0.008 if rel_day < 0 else (0.015 if rel_day == 0 else 0.010),
                "amihud_ratio_median": 0.00002 if rel_day < 0 else 0.00005,
                "prob_hist_gradient_boosting_median": 0.35 if rel_day < 0 else 0.65,
                "anomaly_score_scaled_median": 12.0 if rel_day < 0 else 35.0,
            }
        )
    cs_df = pd.DataFrame(records)

    summary = compute_event_impact_summary(cs_df)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["event_id"] == "EVT_SAMPLE"
    assert row["spread_shock_pct"] > 0  # 0.015 vs 0.008 baseline is positive shock
    assert row["hgbt_prob_delta"] > 0  # 0.65 vs 0.35 baseline is positive delta
