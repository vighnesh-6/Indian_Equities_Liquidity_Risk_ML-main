"""Unit tests for ModelEvaluator metrics suite, threshold tuning, and stratified evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.evaluation import ModelEvaluator


def test_compute_classification_metrics_perfect():
    """Verifies metrics computation on perfect predictions."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.15, 0.85, 0.90, 0.95])

    metrics = ModelEvaluator.compute_classification_metrics(y_true, y_prob, threshold=0.50)

    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["tp"] == 3
    assert metrics["fp"] == 0
    assert metrics["tn"] == 3
    assert metrics["fn"] == 0
    assert metrics["n_obs"] == 6
    assert metrics["n_pos"] == 3
    assert metrics["brier_score"] < 0.05


def test_compute_classification_metrics_edge_cases():
    """Verifies safe handling of edge cases (single class, all NaNs)."""
    # Single class only
    y_all_zero = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])
    m_zero = ModelEvaluator.compute_classification_metrics(y_all_zero, y_prob)
    assert np.isnan(m_zero["roc_auc"])
    assert np.isnan(m_zero["pr_auc"])
    assert m_zero["n_pos"] == 0

    # All NaNs
    m_nan = ModelEvaluator.compute_classification_metrics(np.array([np.nan]), np.array([np.nan]))
    assert np.isnan(m_nan["pr_auc"])
    assert m_nan["tp"] == 0


def test_find_optimal_f1_threshold():
    """Verifies that find_optimal_f1_threshold finds the threshold maximizing F1."""
    # Synthetic case where positive class is at lower predicted probability (~0.35)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.10, 0.15, 0.20, 0.25, 0.35, 0.40, 0.45, 0.50])

    # At threshold 0.50, only one positive is flagged (recall = 0.25, F1 = 0.40)
    # At threshold 0.30, all positives and no negatives are flagged (recall = 1.0, precision = 1.0, F1 = 1.0)
    th, f1 = ModelEvaluator.find_optimal_f1_threshold(y_true, y_prob, min_threshold=0.10, max_threshold=0.60, step=0.05)

    assert th <= 0.35
    assert f1 == pytest.approx(1.0)


def test_evaluate_by_cap_segment():
    """Verifies stratified evaluation across market-cap segments."""
    df = pd.DataFrame({
        "cap_segment": ["Large", "Large", "Mid", "Mid", "Small", "Small"],
        "hre_5d_train": [0, 1, 0, 1, 0, 1],
        "hre_prob": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
    })

    res = ModelEvaluator.evaluate_by_cap_segment(
        df, segment_col="cap_segment", target_col="hre_5d_train", prob_col="hre_prob", threshold=0.50
    )

    assert isinstance(res, pd.DataFrame)
    assert "cap_segment" in res.columns
    assert set(res["cap_segment"].unique()) == {"large", "mid", "small", "ALL"}
    assert len(res) == 4
    # All segments had perfect separation at threshold 0.50
    assert (res["f1"] == 1.0).all()

