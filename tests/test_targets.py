"""Unit tests for supervised target construction module."""

import numpy as np
import pandas as pd
import pytest

from src.labels.targets import (
    apply_training_embargo_split,
    build_features_with_targets,
    compute_forward_max_lrs,
    construct_hre_target,
)


def test_target_builder_callable():
    """Verifies that construct_hre_target is callable."""
    assert callable(construct_hre_target)


def test_compute_forward_max_lrs_shift_and_window():
    """Verifies that forward max excludes day t and requires 5 future observations."""
    dates = pd.date_range("2023-01-01", periods=8, freq="B").strftime("%Y-%m-%d").tolist()
    # 8 observations for single ticker
    df = pd.DataFrame({
        "date": dates,
        "ticker": ["ABC.NS"] * 8,
        "lrs": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    })
    fwd = compute_forward_max_lrs(df, forward_window=5)

    # Observation 0 (day 0, lrs=10) looks at days 1..5: [20, 30, 40, 50, 60] -> max is 60.0
    assert np.isclose(fwd.iloc[0], 60.0)
    # Observation 1 (day 1, lrs=20) looks at days 2..6: [30, 40, 50, 60, 70] -> max is 70.0
    assert np.isclose(fwd.iloc[1], 70.0)
    # Observation 2 (day 2, lrs=30) looks at days 3..7: [40, 50, 60, 70, 80] -> max is 80.0
    assert np.isclose(fwd.iloc[2], 80.0)
    # Final 5 observations (indices 3, 4, 5, 6, 7) cannot form full 5-day future window -> NaN
    assert fwd.iloc[3:].isna().all()
    assert len(fwd.iloc[3:]) == 5


def test_compute_forward_max_lrs_ticker_isolation():
    """Verifies that forward windows do NOT cross between tickers."""
    dates = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
    df = pd.DataFrame({
        "date": dates * 2,
        "ticker": ["STOCK_A"] * 6 + ["STOCK_B"] * 6,
        "lrs": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0],
    })
    fwd = compute_forward_max_lrs(df, forward_window=5)

    # For STOCK_A index 0: looks at A's days 1..5: [20, 30, 40, 50, 60] -> max is 60.0 (NOT 95 from B!)
    assert np.isclose(fwd.iloc[0], 60.0)
    # For STOCK_B index 6: looks at B's days 1..5: [91, 92, 93, 94, 95] -> max is 95.0
    assert np.isclose(fwd.iloc[6], 95.0)


def test_construct_hre_target_threshold_mapping():
    """Verifies binary classification with segment-specific thresholds."""
    df = pd.DataFrame({
        "date": ["2023-01-01"] * 4,
        "ticker": ["L1", "M1", "S1", "L2"],
        "cap_segment": ["large", "mid", "small", "large"],
        # Frozen thresholds: Large=73.8298, Mid=72.3251, Small=72.4359
        "fwd_max_lrs_5d": [75.0, 71.0, 73.0, np.nan],
    })
    hre = construct_hre_target(df, fwd_max_col="fwd_max_lrs_5d")

    # L1: 75.0 >= 73.8298 -> 1.0
    assert hre.iloc[0] == 1.0
    # M1: 71.0 < 72.3251 -> 0.0
    assert hre.iloc[1] == 0.0
    # S1: 73.0 >= 72.4359 -> 1.0
    assert hre.iloc[2] == 1.0
    # L2: NaN -> NaN
    assert np.isnan(hre.iloc[3])


def test_apply_training_embargo_split_dates_and_masking():
    """Verifies partition split creation and embargo masking in hre_5d_train."""
    # Create 15 trading days
    dates = pd.date_range("2025-10-01", periods=15, freq="B").strftime("%Y-%m-%d").tolist()
    # Assume train_cutoff is 10th day, embargo is 5 days (days 6..10), holdout is days 11..15
    train_cutoff = dates[9]
    embargo_days = 5
    holdout_start = dates[10]

    df = pd.DataFrame({
        "date": dates,
        "ticker": ["A.NS"] * 15,
        "hre_5d": [1.0] * 15,
    })

    split_df = apply_training_embargo_split(
        df,
        train_cutoff=train_cutoff,
        embargo_days=embargo_days,
        holdout_start=holdout_start,
        hre_col="hre_5d",
    )

    # Check split categories
    assert (split_df["split"].iloc[:5] == "train").all()
    assert (split_df["split"].iloc[5:10] == "train_boundary_embargo").all()
    assert (split_df["split"].iloc[10:] == "holdout").all()

    # Check hre_5d_train: valid only in train, NaN in embargo and holdout
    assert (split_df["hre_5d_train"].iloc[:5] == 1.0).all()
    assert split_df["hre_5d_train"].iloc[5:].isna().all()


def test_build_features_with_targets_toy():
    """Verifies end-to-end dataset builder on toy features DataFrame."""
    dates = pd.date_range("2025-10-01", periods=12, freq="B").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({
        "date": dates * 2,
        "ticker": ["T1"] * 12 + ["T2"] * 12,
        "cap_segment": ["large"] * 24,
        "amihud_ma20": [0.1] * 24,
        "cs_spread_ma10": [0.01] * 24,
        "volume_z20": [1.0] * 24,
        "downside_vol_20d": [0.02] * 24,
    })

    enriched = build_features_with_targets(
        df,
        train_cutoff=dates[7],
        embargo_days=3,
        holdout_start=dates[8],
    )

    # Verify all expected columns exist
    expected_cols = ["lrs", "fwd_max_lrs_5d", "hre_5d", "split", "hre_5d_train"]
    for col in expected_cols:
        assert col in enriched.columns
    # Verify original rows preserved
    assert len(enriched) == len(df)
