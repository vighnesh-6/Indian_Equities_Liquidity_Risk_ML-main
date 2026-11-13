"""Unit tests for Composite Liquidity-Risk Score (LRS) module."""

import numpy as np
import pandas as pd
import pytest

from src.labels.lrs import (
    DEFAULT_LRS_WEIGHTS,
    FROZEN_LRS_THRESHOLDS,
    calibrate_lrs_thresholds,
    compute_lrs,
)


def test_lrs_callable():
    """Verifies that compute_lrs is callable."""
    assert callable(compute_lrs)


def test_lrs_weights_validation():
    """Verifies that invalid weights sum raises ValueError."""
    df = pd.DataFrame({
        "date": ["2023-01-02"],
        "ticker": ["ABC.NS"],
        "cap_segment": ["large"],
        "amihud_ma20": [0.1],
        "cs_spread_ma10": [0.5],
        "volume_z20": [0.0],
        "downside_vol_20d": [0.02],
    })
    invalid_weights = {"amihud_ma20": 0.5, "cs_spread_ma10": 0.5, "volume_z20": 0.5, "downside_vol_20d": 0.5}
    with pytest.raises(ValueError, match="must sum to 1.0"):
        compute_lrs(df, weights=invalid_weights)


def test_lrs_missing_columns_validation():
    """Verifies that missing columns raise ValueError."""
    df = pd.DataFrame({
        "date": ["2023-01-02"],
        "ticker": ["ABC.NS"],
        # Missing cap_segment and features
    })
    with pytest.raises(ValueError, match="missing required"):
        compute_lrs(df)


def test_lrs_calculation_toy_dataset():
    """Verifies rank calculation, inverse volume direction, and score bounds on toy data."""
    # Day 1: 3 large cap stocks
    df = pd.DataFrame({
        "date": ["2023-01-02", "2023-01-02", "2023-01-02"],
        "ticker": ["STOCK_LOW", "STOCK_MID", "STOCK_HIGH"],
        "cap_segment": ["large", "large", "large"],
        # amihud (asc): LOW=10, MID=20, HIGH=30 -> ranks: 1/3, 2/3, 3/3
        "amihud_ma20": [10.0, 20.0, 30.0],
        # cs_spread (asc): LOW=0.01, MID=0.02, HIGH=0.03 -> ranks: 1/3, 2/3, 3/3
        "cs_spread_ma10": [0.01, 0.02, 0.03],
        # volume_z20 (desc / inv): LOW has high volume (+2.0), HIGH has dried volume (-2.0)
        # Drying volume = higher risk -> HIGH ranks highest (100%), LOW ranks lowest (33.3%)
        "volume_z20": [2.0, 0.0, -2.0],
        # downside_vol (asc): LOW=0.01, MID=0.02, HIGH=0.03 -> ranks: 1/3, 2/3, 3/3
        "downside_vol_20d": [0.01, 0.02, 0.03],
    })

    lrs = compute_lrs(df)
    assert len(lrs) == 3
    assert lrs.name == "lrs"
    # STOCK_HIGH should have maximum risk score (100.0)
    assert np.isclose(lrs.iloc[2], 100.0)
    # STOCK_LOW should have lowest risk score (33.333...)
    assert np.isclose(lrs.iloc[0], 100.0 / 3.0)
    # STOCK_MID should be between LOW and HIGH
    assert lrs.iloc[0] < lrs.iloc[1] < lrs.iloc[2]
    # All bounded [0, 100]
    assert (lrs >= 0.0).all() and (lrs <= 100.0).all()


def test_lrs_missing_values_propagation():
    """Verifies that missing feature values propagate NaN to LRS."""
    df = pd.DataFrame({
        "date": ["2023-01-02", "2023-01-02"],
        "ticker": ["A.NS", "B.NS"],
        "cap_segment": ["large", "large"],
        "amihud_ma20": [10.0, np.nan],  # B has NaN
        "cs_spread_ma10": [0.01, 0.02],
        "volume_z20": [1.0, -1.0],
        "downside_vol_20d": [0.01, 0.02],
    })
    lrs = compute_lrs(df)
    assert not np.isnan(lrs.iloc[0])
    assert np.isnan(lrs.iloc[1])


def test_lrs_segment_isolation():
    """Verifies that ranking is computed independently within each cap segment."""
    df = pd.DataFrame({
        "date": ["2023-01-02", "2023-01-02", "2023-01-02", "2023-01-02"],
        "ticker": ["L1", "L2", "S1", "S2"],
        "cap_segment": ["large", "large", "small", "small"],
        # Large stocks have low amihud (0.1, 0.2)
        # Small stocks have high amihud (10.0, 20.0)
        "amihud_ma20": [0.1, 0.2, 10.0, 20.0],
        "cs_spread_ma10": [0.01, 0.02, 0.05, 0.10],
        "volume_z20": [1.0, -1.0, 1.0, -1.0],
        "downside_vol_20d": [0.01, 0.02, 0.05, 0.10],
    })
    lrs = compute_lrs(df)
    # Within large segment: L1 is lowest (50%), L2 is highest (100%)
    # Within small segment: S1 is lowest (50%), S2 is highest (100%)
    # Despite S1 having 100x higher raw illiquidity than L2, within-tier ranking isolates them:
    assert np.isclose(lrs.iloc[0], 50.0)
    assert np.isclose(lrs.iloc[1], 100.0)
    assert np.isclose(lrs.iloc[2], 50.0)
    assert np.isclose(lrs.iloc[3], 100.0)


def test_frozen_thresholds_match_audited_constants():
    """Verifies that frozen threshold values match the audited contract."""
    assert np.isclose(FROZEN_LRS_THRESHOLDS["large"], 73.8298, atol=1e-4)
    assert np.isclose(FROZEN_LRS_THRESHOLDS["mid"], 72.3251, atol=1e-4)
    assert np.isclose(FROZEN_LRS_THRESHOLDS["small"], 72.4359, atol=1e-4)


def test_calibrate_lrs_thresholds_toy():
    """Verifies calibrate_lrs_thresholds calculates segment percentiles."""
    # Create 10 observations for large, mid, small
    df = pd.DataFrame({
        "date": ["2023-01-02"] * 30,
        "ticker": [f"TICK_{i}" for i in range(30)],
        "cap_segment": ["large"] * 10 + ["mid"] * 10 + ["small"] * 10,
        "amihud_ma20": list(range(10)) + list(range(10)) + list(range(10)),
        "cs_spread_ma10": list(range(10)) + list(range(10)) + list(range(10)),
        "volume_z20": list(range(10)) + list(range(10)) + list(range(10)),
        "downside_vol_20d": list(range(10)) + list(range(10)) + list(range(10)),
    })
    thresh = calibrate_lrs_thresholds(df, train_cutoff="2023-01-05", quantile=0.90)
    assert set(thresh.keys()) == {"large", "mid", "small"}
    for k in thresh:
        assert 0.0 <= thresh[k] <= 100.0

