"""Unit tests for Phase 8 Batch 3 model-specific preprocessing pipelines.

Tests verify:
1. Logistic Regression builder returns fresh Pipeline instances.
2. Logistic Regression pipeline uses median SimpleImputer.
3. Logistic Regression supports both StandardScaler and RobustScaler.
4. Logistic Regression builder rejects invalid scaler types.
5. Logistic Regression receives only its 20 candidate features.
6. Random Forest Candidate A builder returns a pipeline with 35 features.
7. Random Forest Candidate B builder returns a pipeline with 36 features.
8. Random Forest Candidate B includes F51 relative_volume.
9. Random Forest native NaN handling works without imputer.
10. HGBT native NaN handling works.
11. Isolation Forest receives only its configured 12 features.
12. Isolation Forest receives no target labels.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest

from src.modeling.preprocessing import (
    LOGISTIC_REGRESSION_FEATURES,
    RANDOM_FOREST_CANDIDATE_A_FEATURES,
    RANDOM_FOREST_CANDIDATE_B_FEATURES,
    HGBT_FEATURES,
    ISOLATION_FOREST_FEATURES,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    build_hgbt_pipeline,
    build_isolation_forest_pipeline,
    get_model_features,
    validate_features_present,
)
from src.modeling.anomaly import AnomalyDetector


# =============================================================================
# 1. Logistic Regression Builder & Feature Tests
# =============================================================================

def test_lr_builder_fresh_instances():
    """1. Logistic Regression builder returns fresh, independent Pipeline instances."""
    p1 = build_logistic_regression_pipeline(scaler_type="standard")
    p2 = build_logistic_regression_pipeline(scaler_type="standard")

    assert isinstance(p1, Pipeline)
    assert isinstance(p2, Pipeline)
    assert p1 is not p2
    assert p1.named_steps["imputer"] is not p2.named_steps["imputer"]
    assert p1.named_steps["scaler"] is not p2.named_steps["scaler"]
    assert p1.named_steps["classifier"] is not p2.named_steps["classifier"]


def test_lr_pipeline_uses_median_imputer():
    """2. Logistic Regression pipeline uses SimpleImputer with strategy='median'."""
    p = build_logistic_regression_pipeline(scaler_type="standard")
    imputer = p.named_steps["imputer"]

    assert isinstance(imputer, SimpleImputer)
    assert imputer.strategy == "median"


def test_lr_supports_standard_and_robust_scalers():
    """3. Logistic Regression supports both StandardScaler and RobustScaler."""
    p_std = build_logistic_regression_pipeline(scaler_type="standard")
    assert isinstance(p_std.named_steps["scaler"], StandardScaler)

    p_rob = build_logistic_regression_pipeline(scaler_type="robust")
    assert isinstance(p_rob.named_steps["scaler"], RobustScaler)


def test_lr_builder_rejects_invalid_scaler():
    """4. Logistic Regression builder rejects invalid scaler types with ValueError."""
    with pytest.raises(ValueError, match="Invalid scaler_type 'minmax'"):
        build_logistic_regression_pipeline(scaler_type="minmax")

    with pytest.raises(ValueError, match="Invalid scaler_type 'none'"):
        build_logistic_regression_pipeline(scaler_type="none")


def test_lr_feature_count_and_selection():
    """5. Logistic Regression receives exactly its 20 candidate features."""
    features = get_model_features("logistic_regression")
    assert len(features) == 20
    assert features == LOGISTIC_REGRESSION_FEATURES

    # Verify algebraic triplet exclusion (F04 return_20d and F47 excess_return_20d dropped)
    assert "return_20d" not in features
    assert "excess_return_20d" not in features
    assert "nifty_return_20d" in features

    # Test fitting and predicting on synthetic data with NaNs
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(50, 20), columns=features)
    X.iloc[0, 1] = np.nan
    X.iloc[5, 8] = np.nan
    y = pd.Series(rng.binomial(1, 0.3, size=50))

    pipeline = build_logistic_regression_pipeline(scaler_type="standard")
    pipeline.fit(X, y)
    probs = pipeline.predict_proba(X)
    assert probs.shape == (50, 2)
    assert np.all(np.isfinite(probs))


# =============================================================================
# 2. Random Forest Candidate A & B Tests
# =============================================================================

def test_rf_candidate_a_features():
    """6. Random Forest Candidate A builder returns a pipeline configured for 35 features."""
    features_a = get_model_features("random_forest", candidate="A")
    assert len(features_a) == 35
    assert features_a == RANDOM_FOREST_CANDIDATE_A_FEATURES
    assert "relative_volume" not in features_a

    pipeline = build_random_forest_pipeline(candidate="A")
    assert isinstance(pipeline, Pipeline)
    assert "classifier" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["classifier"], RandomForestClassifier)
    # Native NaN handling: no imputer or scaler in steps
    assert "imputer" not in pipeline.named_steps
    assert "scaler" not in pipeline.named_steps


def test_rf_candidate_b_features():
    """7. Random Forest Candidate B builder returns a pipeline configured for 36 features."""
    features_b = get_model_features("random_forest", candidate="B")
    assert len(features_b) == 36
    assert features_b == RANDOM_FOREST_CANDIDATE_B_FEATURES

    pipeline = build_random_forest_pipeline(candidate="B")
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["classifier"], RandomForestClassifier)


def test_rf_candidate_b_includes_f51():
    """8. Random Forest Candidate B includes F51 relative_volume."""
    features_a = set(RANDOM_FOREST_CANDIDATE_A_FEATURES)
    features_b = set(RANDOM_FOREST_CANDIDATE_B_FEATURES)

    diff = features_b - features_a
    assert diff == {"relative_volume"}
    assert "relative_volume" in RANDOM_FOREST_CANDIDATE_B_FEATURES


def test_rf_native_nan_handling():
    """9. Random Forest native NaN handling works without imputer."""
    pipeline = build_random_forest_pipeline(candidate="A", n_estimators=10, random_state=42)

    rng = np.random.RandomState(42)
    cols = RANDOM_FOREST_CANDIDATE_A_FEATURES
    X = pd.DataFrame(rng.randn(60, len(cols)), columns=cols)
    # Inject missing values across multiple columns
    X.iloc[2, 5] = np.nan
    X.iloc[10:15, 12] = np.nan
    X.iloc[20, 30] = np.nan
    y = pd.Series(rng.binomial(1, 0.25, size=60))

    pipeline.fit(X, y)
    probs = pipeline.predict_proba(X)
    assert probs.shape == (60, 2)
    assert np.all(np.isfinite(probs))


# =============================================================================
# 3. HistGradientBoosting Tests
# =============================================================================

def test_hgbt_native_nan_handling():
    """10. HGBT native NaN handling works without imputer or scaler."""
    pipeline = build_hgbt_pipeline(max_iter=15, random_state=42)
    assert isinstance(pipeline.named_steps["classifier"], HistGradientBoostingClassifier)
    assert "imputer" not in pipeline.named_steps
    assert "scaler" not in pipeline.named_steps

    cols = HGBT_FEATURES
    assert len(cols) == 36
    assert "relative_volume" in cols

    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(60, len(cols)), columns=cols)
    # Inject missing values including in relative_volume
    X.loc[X.index[:10], "relative_volume"] = np.nan
    X.iloc[5, 2] = np.nan
    y = pd.Series(rng.binomial(1, 0.25, size=60))

    pipeline.fit(X, y)
    probs = pipeline.predict_proba(X)
    assert probs.shape == (60, 2)
    assert np.all(np.isfinite(probs))


# =============================================================================
# 4. Isolation Forest Tests
# =============================================================================

def test_isolation_forest_feature_count():
    """11. Isolation Forest receives only its configured 12 features."""
    features = get_model_features("isolation_forest")
    assert len(features) == 12
    assert features == ISOLATION_FOREST_FEATURES

    expected = [
        "amihud_ratio",
        "amihud_z20",
        "volume_ratio",
        "volume_z20",
        "vol_20d",
        "vol_ratio_5_20",
        "daily_range",
        "range_z20",
        "overnight_gap_abs",
        "cs_spread_est",
        "nifty_vol_20d",
        "return_z20",
    ]
    assert features == expected


def test_isolation_forest_target_independence():
    """12. Isolation Forest receives no target labels and fits strictly unsupervised."""
    detector = AnomalyDetector(n_estimators=15, random_state=42)
    assert detector.feature_subset == ISOLATION_FOREST_FEATURES

    # Synthetic training dataframe with and without target
    rng = np.random.RandomState(42)
    df = pd.DataFrame(rng.randn(50, 12), columns=ISOLATION_FOREST_FEATURES)
    df.iloc[3, 0] = np.nan  # Median imputer handles this

    # Verify fit works with NO target column present whatsoever
    assert "hre_5d_train" not in df.columns
    detector.fit(df)
    assert detector.pipeline is not None
    assert detector.model is not None

    scores_df = detector.score(df)
    assert "isolation_score" in scores_df.columns
    assert "isolation_flag" in scores_df.columns
    assert "anomaly_score_scaled" in scores_df.columns
    assert len(scores_df) == 50
    assert scores_df["anomaly_score_scaled"].between(0.0, 100.0).all()
    assert set(scores_df["isolation_flag"].unique()).issubset({0, 1})


def test_validate_features_present():
    """Helper test: validate_features_present raises informative ValueError on missing columns."""
    df = pd.DataFrame({"feat_a": [1, 2], "feat_b": [3, 4]})
    validate_features_present(df, ["feat_a", "feat_b"])

    with pytest.raises(ValueError, match="Missing 1 required feature columns"):
        validate_features_present(df, ["feat_a", "feat_b", "feat_c"])


def test_model_identifier_feature_routing():
    """Regression test: all model identifiers resolve to exact expected feature sets."""
    # Logistic Regression variants (20 features)
    for name in ["logistic_regression", "lr", "baseline", "logistic_regression_standard", "lr_standard", "logistic_regression_robust", "lr_robust"]:
        feats = get_model_features(name)
        assert len(feats) == 20
        assert feats == LOGISTIC_REGRESSION_FEATURES

    # Random Forest Candidate A variants (35 features, excludes F51)
    for name in ["random_forest", "rf", "random_forest_candidate_a", "rf_a", "rf_candidate_a"]:
        feats = get_model_features(name, candidate="A")
        assert len(feats) == 35
        assert feats == RANDOM_FOREST_CANDIDATE_A_FEATURES
        assert "relative_volume" not in feats

    # Random Forest Candidate B variants (36 features, includes F51)
    for name in ["random_forest_candidate_b", "rf_b", "rf_candidate_b"]:
        feats = get_model_features(name)
        assert len(feats) == 36
        assert feats == RANDOM_FOREST_CANDIDATE_B_FEATURES
        assert "relative_volume" in feats

    # HistGradientBoosting variants (36 features)
    for name in ["hist_gradient_boosting", "hgbt", "boosting"]:
        feats = get_model_features(name)
        assert len(feats) == 36
        assert feats == HGBT_FEATURES

    # Isolation Forest variants (12 features)
    for name in ["isolation_forest", "if", "anomaly"]:
        feats = get_model_features(name)
        assert len(feats) == 12
        assert feats == ISOLATION_FOREST_FEATURES


def test_isolation_forest_fold_scoped_fitting():
    """Tests that AnomalyDetector.fit with train_idx fits exclusively on the scoped slice."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame(rng.randn(100, 12), columns=ISOLATION_FOREST_FEATURES)
    df["split"] = "train"
    df["is_warmup"] = False

    # Define a fold slice of first 40 rows
    train_idx = df.index[:40]
    detector = AnomalyDetector(n_estimators=10, random_state=42)
    detector.fit(df, train_idx=train_idx)

    assert detector.pipeline is not None
    assert detector.score_min_ is not None
    assert detector.score_max_ is not None

    # Score remaining 60 rows
    val_slice = df.iloc[40:]
    scores = detector.score(val_slice)
    assert len(scores) == 60
    assert "anomaly_score_scaled" in scores.columns
    assert scores["anomaly_score_scaled"].between(0.0, 100.0).all()
