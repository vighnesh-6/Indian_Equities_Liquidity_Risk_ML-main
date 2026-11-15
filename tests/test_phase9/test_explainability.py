"""Unit tests for Phase 9.4 Model Explainability & Feature Importance."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.modeling.phase9.explainability import (
    FEATURE_FAMILY_MAP,
    compare_feature_importances,
    compute_hgbt_permutation_importance,
    compute_rf_mdi_importance,
    extract_top_10_risk_drivers,
)


@pytest.fixture
def mock_dataset():
    """Create a synthetic dataset for testing explainability functions."""
    rng = np.random.default_rng(42)
    n = 200
    feature_names = [f"feat_{i:02d}" for i in range(12)]
    X = rng.normal(size=(n, 12))
    # Make feat_00 strongly predictive
    prob = 1 / (1 + np.exp(-(X[:, 0] * 2.0 + X[:, 1] * 1.0)))
    y = (rng.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame(X, columns=feature_names)
    df["hre_5d"] = y
    return df, feature_names


@pytest.fixture
def mock_hgbt_pipeline(mock_dataset):
    """Fit a mock HGBT pipeline for testing."""
    df, feature_names = mock_dataset
    X = df[feature_names]
    y = df["hre_5d"]
    clf = HistGradientBoostingClassifier(random_state=42, max_iter=20)
    pipeline = Pipeline([("classifier", clf)])
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def mock_rf_pipeline(mock_dataset):
    """Fit a mock Random Forest pipeline for testing."""
    df, feature_names = mock_dataset
    X = df[feature_names]
    y = df["hre_5d"]
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    pipeline = Pipeline([("classifier", clf)])
    pipeline.fit(X, y)
    return pipeline


def test_feature_family_mapping():
    """Verify that feature family mapping covers core microstructure categories."""
    assert len(FEATURE_FAMILY_MAP) == 36
    valid_families = {"ILLIQUIDITY", "SPREAD", "VOLATILITY", "VOLUME_TURNOVER", "MOMENTUM_RETURN", "MACRO_MARKET"}
    for feat, fam in FEATURE_FAMILY_MAP.items():
        assert fam in valid_families


def test_compute_hgbt_permutation_importance(mock_hgbt_pipeline, mock_dataset):
    """Verify permutation feature importance computation."""
    df, feature_names = mock_dataset
    imp_df = compute_hgbt_permutation_importance(
        model=mock_hgbt_pipeline,
        features_df=df,
        feature_names=feature_names,
        target_col="hre_5d",
        scoring="average_precision",
        n_repeats=3,
        random_state=42,
        sample_size=100,
        regime_label="TEST_DATASET",
    )
    assert not imp_df.empty
    assert len(imp_df) == len(feature_names)
    assert set(imp_df.columns) >= {"feature", "feature_family", "importance_mean", "importance_std", "rank"}
    # feat_00 was strongly predictive, so it should be near the top
    assert imp_df["rank"].iloc[0] == 1
    assert imp_df["importance_mean"].iloc[0] >= imp_df["importance_mean"].iloc[-1]


def test_compute_rf_mdi_importance(mock_rf_pipeline, mock_dataset):
    """Verify Random Forest MDI importance extraction."""
    _, feature_names = mock_dataset
    mdi_df = compute_rf_mdi_importance(
        model=mock_rf_pipeline,
        feature_names=feature_names,
    )
    assert not mdi_df.empty
    assert len(mdi_df) == len(feature_names)
    assert set(mdi_df.columns) >= {"feature", "importance_mdi", "rank_mdi"}
    # MDI importances should sum approximately to 1.0
    assert abs(mdi_df["importance_mdi"].sum() - 1.0) < 1e-4
    assert (mdi_df["importance_mdi"] >= 0.0).all()


def test_compare_feature_importances(mock_hgbt_pipeline, mock_rf_pipeline, mock_dataset):
    """Verify comparative ranking analysis across models."""
    df, feature_names = mock_dataset
    hgbt_df = compute_hgbt_permutation_importance(
        model=mock_hgbt_pipeline,
        features_df=df,
        feature_names=feature_names,
        n_repeats=2,
    )
    rf_df = compute_rf_mdi_importance(
        model=mock_rf_pipeline,
        feature_names=feature_names,
    )
    comp_df = compare_feature_importances(hgbt_df, rf_df)
    assert not comp_df.empty
    assert len(comp_df) == len(feature_names)
    assert "rank_diff" in comp_df.columns
    assert (comp_df["rank_diff"] >= 0).all()


def test_extract_top_10_risk_drivers(mock_hgbt_pipeline, mock_dataset):
    """Verify top 10 risk driver table extraction and enrichment."""
    df, feature_names = mock_dataset
    hgbt_df = compute_hgbt_permutation_importance(
        model=mock_hgbt_pipeline,
        features_df=df,
        feature_names=feature_names,
        n_repeats=2,
    )
    top10_df = extract_top_10_risk_drivers(hgbt_df)
    assert len(top10_df) == 10
    assert set(top10_df.columns) >= {
        "rank", "feature", "family", "importance_mean_prauc_loss",
        "financial_interpretation", "methodological_limitations"
    }
    assert all(top10_df["financial_interpretation"].str.len() > 10)
