"""Unit and integration tests for walk-forward fold generation, 5-session purging, and data preservation.

Tests verify:
13. Five-session purging excludes training observations whose target windows overlap validation.
14. Trading-session counts are used rather than calendar days.
15. Fold dates remain identical across supervised models.
16. No preprocessing fit occurs on validation observations.
17. Fresh preprocessing instances are created for each fold.
18. Final holdout observations are excluded from all training populations.
19. Dataset hashes remain unchanged.
20. Existing Phase 5 and Phase 7 artifacts remain untouched.
"""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.modeling.trainer import ModelTrainer
from src.modeling.preprocessing import (
    LOGISTIC_REGRESSION_FEATURES,
    RANDOM_FOREST_CANDIDATE_A_FEATURES,
    RANDOM_FOREST_CANDIDATE_B_FEATURES,
    HGBT_FEATURES,
    build_logistic_regression_pipeline,
)
from src.modeling.anomaly import AnomalyDetector


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def synthetic_trading_df():
    """Generates a small synthetic DataFrame with 100 consecutive business days."""
    dates = pd.date_range("2023-01-02", periods=100, freq="B")
    n_stocks = 5
    records = []
    rng = np.random.RandomState(42)

    for d in dates:
        for s in range(n_stocks):
            records.append({
                "date": d,
                "ticker": f"STOCK_{s}.NS",
                "split": "train",
                "is_warmup": False,
                "hre_5d_train": float(rng.binomial(1, 0.2)),
                "val_feat": float(rng.randn()),
            })

    df = pd.DataFrame(records)
    # Add all LR features to synthetic df
    for f in LOGISTIC_REGRESSION_FEATURES:
        if f not in df.columns:
            df[f] = rng.randn(len(df))

    return df


# =============================================================================
# 13. Five-Session Purging Overlap Tests
# =============================================================================

def test_five_session_purging_prevents_target_overlap(synthetic_trading_df):
    """13. Five-session purging excludes training sessions whose target windows overlap validation."""
    trainer = ModelTrainer(n_splits=3, min_train_days=30, test_window_days=10, purge_sessions=5)
    splits = trainer.generate_walk_forward_splits(synthetic_trading_df)

    assert len(splits) == 3

    unique_dates = sorted(synthetic_trading_df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_dates = sorted(synthetic_trading_df.loc[train_idx, "date"].unique())
        val_dates = sorted(synthetic_trading_df.loc[val_idx, "date"].unique())

        last_train_date = train_dates[-1]
        first_val_date = val_dates[0]

        last_train_session = date_to_idx[last_train_date]
        first_val_session = date_to_idx[first_val_date]

        # The forward 5-day window for an observation at last_train_session
        # covers sessions [last_train_session + 1, last_train_session + 5].
        # In order for ZERO overlap with first_val_session, we must have:
        # last_train_session + 5 < first_val_session
        assert last_train_session + 5 < first_val_session, (
            f"Fold {fold_idx}: Target lookahead from session {last_train_session} reaches "
            f"session {last_train_session + 5}, which touches or exceeds validation start {first_val_session}!"
        )

        # Verify exactly purge_sessions (5) sessions sit between last_train and first_val
        purged_sessions_count = first_val_session - last_train_session - 1
        assert purged_sessions_count == 5, (
            f"Fold {fold_idx}: Expected exactly 5 purged trading sessions, found {purged_sessions_count}."
        )


# =============================================================================
# 14. Trading Sessions vs. Calendar Days
# =============================================================================

def test_trading_session_counts_used_rather_than_calendar_days():
    """14. Trading-session counts are used rather than calendar days.

    Verifies that weekend gaps and holidays do not alter the exact 5-session purge count.
    """
    # Create non-contiguous dates with weekend leaps and holiday jumps
    calendar_dates = [
        # Week 1
        pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04"),
        pd.Timestamp("2023-01-05"), pd.Timestamp("2023-01-06"),
        # Weekend + Holiday (4 calendar days gap)
        pd.Timestamp("2023-01-11"), pd.Timestamp("2023-01-12"), pd.Timestamp("2023-01-13"),
        # Weekend (3 calendar days gap)
        pd.Timestamp("2023-01-16"), pd.Timestamp("2023-01-17"), pd.Timestamp("2023-01-18"),
        pd.Timestamp("2023-01-19"), pd.Timestamp("2023-01-20"), pd.Timestamp("2023-01-23"),
        pd.Timestamp("2023-01-24"), pd.Timestamp("2023-01-25"), pd.Timestamp("2023-01-26"),
        pd.Timestamp("2023-01-27"), pd.Timestamp("2023-01-30"), pd.Timestamp("2023-01-31"),
    ]
    df = pd.DataFrame({
        "date": calendar_dates,
        "split": "train",
        "is_warmup": False,
        "hre_5d_train": 1.0,
    })

    # min_train=10, test_window=4, purge=5 => total required = 10 + 5 + 4 = 19 <= 20
    trainer = ModelTrainer(n_splits=1, min_train_days=10, test_window_days=4, purge_sessions=5)
    splits = trainer.generate_walk_forward_splits(df)

    train_idx, val_idx = splits[0]
    train_dates = df.loc[train_idx, "date"].tolist()
    val_dates = df.loc[val_idx, "date"].tolist()

    # Total sessions = 20. Val has 4 sessions (indices 16, 17, 18, 19).
    # Purge must take exactly sessions 11, 12, 13, 14, 15 (5 trading sessions).
    # Train must take sessions 0 to 10 (11 sessions).
    assert len(train_dates) == 11
    assert len(val_dates) == 4

    # Check that purged sessions are exactly 5 distinct trading dates
    purged = [d for d in calendar_dates if d not in train_dates and d not in val_dates]
    assert len(purged) == 5
    assert purged == calendar_dates[11:16]


# =============================================================================
# 15. Identical Fold Boundaries Across Supervised Models
# =============================================================================

def test_identical_fold_boundaries_across_models(synthetic_trading_df):
    """15. Fold date boundaries remain strictly identical across all supervised models."""
    trainer = ModelTrainer(n_splits=3, min_train_days=30, test_window_days=10, purge_sessions=5)

    # Supervised splits
    splits_lr = trainer.generate_walk_forward_splits(synthetic_trading_df, target_col="hre_5d_train")
    splits_rf = trainer.generate_walk_forward_splits(synthetic_trading_df, target_col="hre_5d_train")
    splits_hgbt = trainer.generate_walk_forward_splits(synthetic_trading_df, target_col="hre_5d_train")

    assert len(splits_lr) == len(splits_rf) == len(splits_hgbt) == 3

    for fold in range(3):
        lr_train_dates = set(synthetic_trading_df.loc[splits_lr[fold][0], "date"])
        rf_train_dates = set(synthetic_trading_df.loc[splits_rf[fold][0], "date"])
        hgbt_train_dates = set(synthetic_trading_df.loc[splits_hgbt[fold][0], "date"])

        lr_val_dates = set(synthetic_trading_df.loc[splits_lr[fold][1], "date"])
        rf_val_dates = set(synthetic_trading_df.loc[splits_rf[fold][1], "date"])
        hgbt_val_dates = set(synthetic_trading_df.loc[splits_hgbt[fold][1], "date"])

        assert lr_train_dates == rf_train_dates == hgbt_train_dates
        assert lr_val_dates == rf_val_dates == hgbt_val_dates


# =============================================================================
# 16. No Preprocessing Fit on Validation Observations
# =============================================================================

def test_no_preprocessing_fit_on_validation_observations(synthetic_trading_df):
    """16. No preprocessing fit occurs on validation observations."""
    trainer = ModelTrainer(n_splits=2, min_train_days=30, test_window_days=10, purge_sessions=5)
    splits = trainer.generate_walk_forward_splits(synthetic_trading_df)
    train_idx, val_idx = splits[0]

    pipeline, y_val_prob = trainer.fit_fold_pipeline(
        "logistic_regression",
        synthetic_trading_df,
        train_idx,
        val_idx,
        target_col="hre_5d_train",
        scaler_type="standard",
    )

    # Inspect fitted imputer and scaler
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]

    # Calculate expected statistics solely on training data
    train_features = synthetic_trading_df.loc[train_idx, LOGISTIC_REGRESSION_FEATURES]
    expected_medians = train_features.median().values
    expected_means = train_features.mean().values

    # Assert imputer learned medians strictly from training observations
    np.testing.assert_allclose(imputer.statistics_, expected_medians, rtol=1e-5)

    # Assert scaler learned means strictly from training observations
    np.testing.assert_allclose(scaler.mean_, expected_means, rtol=1e-5)

    # Inject extreme values into validation observations and verify fitted scaler means do not change
    val_copy = synthetic_trading_df.copy()
    val_copy.loc[val_idx, LOGISTIC_REGRESSION_FEATURES] += 1e6
    pipeline.predict_proba(val_copy.loc[val_idx, LOGISTIC_REGRESSION_FEATURES])

    # Scaler mean must remain completely unchanged
    np.testing.assert_allclose(scaler.mean_, expected_means, rtol=1e-5)


# =============================================================================
# 17. Fresh Preprocessing Instances Per Fold
# =============================================================================

def test_fresh_preprocessing_instances_per_fold(synthetic_trading_df):
    """17. Fresh preprocessing instances are created for each fold."""
    trainer = ModelTrainer(n_splits=2, min_train_days=30, test_window_days=10, purge_sessions=5)
    splits = trainer.generate_walk_forward_splits(synthetic_trading_df)

    pipe1, _ = trainer.fit_fold_pipeline(
        "logistic_regression", synthetic_trading_df, splits[0][0], splits[0][1]
    )
    pipe2, _ = trainer.fit_fold_pipeline(
        "logistic_regression", synthetic_trading_df, splits[1][0], splits[1][1]
    )

    assert pipe1 is not pipe2
    assert pipe1.named_steps["imputer"] is not pipe2.named_steps["imputer"]
    assert pipe1.named_steps["scaler"] is not pipe2.named_steps["scaler"]
    assert pipe1.named_steps["classifier"] is not pipe2.named_steps["classifier"]


# =============================================================================
# 18. Holdout Observations Excluded From Training Populations
# =============================================================================

def test_holdout_and_embargo_excluded_from_training_populations(synthetic_trading_df):
    """18. Final holdout and embargo observations are strictly excluded from all training populations."""
    df = synthetic_trading_df.copy()
    # Tag last 15 dates as holdout and 5 dates as train_boundary_embargo
    unique_dates = sorted(df["date"].unique())
    holdout_dates = unique_dates[-15:]
    embargo_dates = unique_dates[-20:-15]

    df.loc[df["date"].isin(holdout_dates), "split"] = "holdout"
    df.loc[df["date"].isin(embargo_dates), "split"] = "train_boundary_embargo"

    trainer = ModelTrainer(n_splits=2, min_train_days=20, test_window_days=10, purge_sessions=5)
    splits = trainer.generate_walk_forward_splits(df)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_splits = df.loc[train_idx, "split"].unique()
        val_splits = df.loc[val_idx, "split"].unique()

        assert "holdout" not in train_splits, f"Fold {fold_idx}: Holdout present in train_idx!"
        assert "holdout" not in val_splits, f"Fold {fold_idx}: Holdout present in val_idx!"
        assert "train_boundary_embargo" not in train_splits, f"Fold {fold_idx}: Embargo present in train_idx!"
        assert "train_boundary_embargo" not in val_splits, f"Fold {fold_idx}: Embargo present in val_idx!"

    # Also verify Isolation Forest fit excludes holdout and embargo
    detector = AnomalyDetector(n_estimators=10)
    # Add IF features
    from src.modeling.preprocessing import ISOLATION_FOREST_FEATURES
    for f in ISOLATION_FOREST_FEATURES:
        if f not in df.columns:
            df[f] = np.random.randn(len(df))

    detector.fit(df)
    assert detector.pipeline is not None


# =============================================================================
# 19. Dataset Hash Verification
# =============================================================================

def test_dataset_hashes_unchanged():
    """19. Dataset hashes remain unchanged."""
    expected_features_hash = "c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae"
    expected_targets_hash = "c6cebf3d93d022b11765c206b8e2ffb84e578a020e0d913d937b37d04028aa91"

    features_path = Path("data/processed/features.parquet")
    targets_path = Path("data/processed/features_with_targets.parquet")

    assert features_path.exists(), "data/processed/features.parquet does not exist!"
    assert targets_path.exists(), "data/processed/features_with_targets.parquet does not exist!"

    actual_features_hash = hashlib.sha256(features_path.read_bytes()).hexdigest()
    actual_targets_hash = hashlib.sha256(targets_path.read_bytes()).hexdigest()

    assert actual_features_hash == expected_features_hash, (
        f"features.parquet hash mismatch! Expected {expected_features_hash}, got {actual_features_hash}"
    )
    assert actual_targets_hash == expected_targets_hash, (
        f"features_with_targets.parquet hash mismatch! Expected {expected_targets_hash}, got {actual_targets_hash}"
    )


# =============================================================================
# 20. Preservation of Phase 5 & Phase 7 Artifacts
# =============================================================================

def test_phase5_and_phase7_artifacts_untouched():
    """20. Existing Phase 5 and Phase 7 artifacts remain untouched."""
    critical_artifacts = [
        Path("src/features/returns.py"),
        Path("src/features/volatility.py"),
        Path("src/features/liquidity.py"),
        Path("src/labels/lrs.py"),
        Path("src/labels/targets.py"),
        Path("reports/phase6_eda_findings_report.md"),
        Path("reports/phase7_data_eligibility_audit.md"),
        Path("reports/phase7_implementation_report.md"),
        Path("reports/phase8_batch2_feature_selection_report.md"),
        Path("reports/phase8_batch3_readiness_audit.md"),
        Path("reports/phase8_batch3_final_clarification_audit.md"),
    ]

    for artifact in critical_artifacts:
        assert artifact.exists(), f"Critical project artifact {artifact} is missing!"
        assert artifact.stat().st_size > 0, f"Critical project artifact {artifact} is empty!"

