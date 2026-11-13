"""Validation tests for the generated features_with_targets.parquet dataset."""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.utils.constants import (
    DEFAULT_BENCHMARK,
    DEFAULT_LRS_WEIGHTS,
    DEFAULT_TARGET_UNIVERSE_SIZE,
    EMBARGO_START_DATE,
    FORWARD_TARGET_WINDOW_DAYS,
    FROZEN_LRS_THRESHOLDS,
    HOLDOUT_START_DATE,
    TRAIN_CUTOFF_DATE,
    WARMUP_WINDOW_DAYS,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_WITH_TARGETS_PATH = BASE_DIR / "data" / "processed" / "features_with_targets.parquet"
ORIGINAL_FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
ORIGINAL_OHLCV_PATH = BASE_DIR / "data" / "interim" / "stocks" / "ohlcv_clean.parquet"
ORIGINAL_EVENTS_PATH = BASE_DIR / "data" / "events" / "market_events.csv"

EXPECTED_FEATURES_HASH = "c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae"
EXPECTED_OHLCV_HASH = "feaefaac2ded41478f067ece25ad3860affc3aa683026b8a925ba431a1b93421"
EXPECTED_EVENTS_HASH = "ad9026e0a50c5e1471c04a089f19bf8cfe7fb04963b132fcae7e641bc8c5e554"


def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def test_baseline_dataset_hashes_unchanged():
    """Confirms 100% bit-for-bit immutability of all protected data files."""
    assert get_sha256(ORIGINAL_FEATURES_PATH) == EXPECTED_FEATURES_HASH
    assert get_sha256(ORIGINAL_OHLCV_PATH) == EXPECTED_OHLCV_HASH
    assert get_sha256(ORIGINAL_EVENTS_PATH) == EXPECTED_EVENTS_HASH


def test_features_with_targets_dimensions_and_schema():
    """Verifies that features_with_targets.parquet exists, has 565,190 rows and 75 columns."""
    assert FEATURES_WITH_TARGETS_PATH.exists()
    df = pd.read_parquet(FEATURES_WITH_TARGETS_PATH)
    assert len(df) == 565190
    assert df.shape[1] == 75
    assert df["ticker"].nunique() == DEFAULT_TARGET_UNIVERSE_SIZE

    expected_new = ["lrs", "fwd_max_lrs_5d", "hre_5d", "split", "hre_5d_train"]
    for col in expected_new:
        assert col in df.columns


def test_features_with_targets_conservation_identity():
    """Verifies the exact conservation identity across all 541,190 post-warmup observations."""
    df = pd.read_parquet(FEATURES_WITH_TARGETS_PATH)
    warmup_rows = int(df["is_warmup"].sum())
    assert warmup_rows == 24000
    post_warmup = df[~df["is_warmup"]]
    assert len(post_warmup) == 541190

    # 1. Eligible training labels (hre_5d_train notna)
    eligible_train = post_warmup["hre_5d_train"].notna().sum()
    assert eligible_train == 466346

    # 2. Embargo observations
    embargo_rows = (post_warmup["split"] == "train_boundary_embargo").sum()
    assert embargo_rows == 1424

    # 3. Training internal forward nulls
    is_train_split = post_warmup["split"] == "train"
    train_internal_nulls = (is_train_split & post_warmup["hre_5d_train"].isna()).sum()
    assert train_internal_nulls == 140

    # 4. Valid holdout labels
    is_holdout = post_warmup["split"] == "holdout"
    assert is_holdout.sum() == 73280
    valid_holdout = (is_holdout & post_warmup["hre_5d"].notna()).sum()
    assert valid_holdout == 71756

    # 5. Tail nulls (300 stocks * 5 final rows)
    tail_5_nulls = post_warmup.groupby("ticker").tail(5)["fwd_max_lrs_5d"].isna().sum()
    assert tail_5_nulls == 1500

    # 6. Holdout internal forward nulls
    holdout_internal_nulls = is_holdout.sum() - valid_holdout - tail_5_nulls
    assert holdout_internal_nulls == 24

    # Conservation identity
    conservation_sum = (
        eligible_train + embargo_rows + train_internal_nulls +
        valid_holdout + holdout_internal_nulls + tail_5_nulls
    )
    assert conservation_sum == 541190


def test_features_with_targets_leakage_and_embargo():
    """Verifies strict zero information leakage across the training-holdout boundary."""
    df = pd.read_parquet(FEATURES_WITH_TARGETS_PATH)

    # All embargo rows must have hre_5d_train = NaN
    embargo_mask = df["split"] == "train_boundary_embargo"
    assert df.loc[embargo_mask, "hre_5d_train"].isna().all()

    # All holdout rows must have hre_5d_train = NaN
    holdout_mask = df["split"] == "holdout"
    assert df.loc[holdout_mask, "hre_5d_train"].isna().all()

    # Exactly 250 trading sessions in holdout
    assert df.loc[holdout_mask, "date"].nunique() == 250


def test_features_with_targets_class_balance():
    """Verifies actual empirical class balance matches audited figures."""
    df = pd.read_parquet(FEATURES_WITH_TARGETS_PATH)
    post_warmup = df[~df["is_warmup"]]

    # Training class balance
    eligible_train = post_warmup["hre_5d_train"].dropna()
    pos_train = (eligible_train == 1.0).sum()
    neg_train = (eligible_train == 0.0).sum()
    assert pos_train == 84988
    assert neg_train == 381358
    assert np.isclose(pos_train / len(eligible_train) * 100.0, 18.2242, atol=1e-2)

    # Holdout class balance
    valid_holdout = post_warmup.loc[post_warmup["split"] == "holdout", "hre_5d"].dropna()
    pos_holdout = (valid_holdout == 1.0).sum()
    neg_holdout = (valid_holdout == 0.0).sum()
    assert pos_holdout == 11873
    assert neg_holdout == 59883
    assert np.isclose(pos_holdout / len(valid_holdout) * 100.0, 16.5464, atol=1e-2)

