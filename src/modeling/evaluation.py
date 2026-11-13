"""Model evaluation and metrics computation module.

Responsible for:
- Computing discrimination metrics: ROC-AUC, PR-AUC (Average Precision).
- Computing threshold metrics: Precision, Recall, F1-Score, Balanced Accuracy, Confusion Matrix.
- Computing probability quality: Brier Score, Log Loss.
- Supporting inner-fold decision threshold optimization for imbalanced classification.
- Stratified performance evaluation across market-cap subgroups (Large, Mid, Small).
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelEvaluator:
    """Evaluates classification performance with focus on imbalanced time-series."""

    @staticmethod
    def compute_classification_metrics(
        y_true: Union[np.ndarray, pd.Series],
        y_prob: Union[np.ndarray, pd.Series],
        threshold: float = 0.50,
    ) -> Dict[str, float]:
        """Computes comprehensive classification metrics suite.

        Args:
            y_true: Ground truth binary target (0 or 1).
            y_prob: Predicted probabilities for positive class (1).
            threshold: Binary classification decision threshold.

        Returns:
            Dict[str, float]: Dictionary containing PR-AUC, ROC-AUC, F1, Precision,
                Recall, Balanced Accuracy, Brier score, Log Loss, and confusion counts.
        """
        y_true_arr = np.asarray(y_true, dtype=float)
        y_prob_arr = np.asarray(y_prob, dtype=float)

        # Filter valid paired observations
        valid_mask = np.isfinite(y_true_arr) & np.isfinite(y_prob_arr)
        if not np.any(valid_mask):
            return {
                "pr_auc": np.nan,
                "roc_auc": np.nan,
                "threshold": float(threshold),
                "f1": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "balanced_accuracy": np.nan,
                "brier_score": np.nan,
                "log_loss": np.nan,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
            }

        y_t = y_true_arr[valid_mask].astype(int)
        y_p = y_prob_arr[valid_mask]

        n_pos = int(np.sum(y_t == 1))
        n_neg = int(np.sum(y_t == 0))

        # Check edge cases: if only one class exists in partition
        if n_pos == 0 or n_neg == 0:
            pr_auc = np.nan
            roc_auc = np.nan
        else:
            pr_auc = float(average_precision_score(y_t, y_p))
            roc_auc = float(roc_auc_score(y_t, y_p))

        # Threshold-dependent binary predictions
        y_pred = (y_p >= threshold).astype(int)

        f1 = float(f1_score(y_t, y_pred, zero_division=0))
        precision = float(precision_score(y_t, y_pred, zero_division=0))
        recall = float(recall_score(y_t, y_pred, zero_division=0))
        if n_pos == 0 or n_neg == 0:
            bal_acc = np.nan
        else:
            bal_acc = float(balanced_accuracy_score(y_t, y_pred))

        cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        # Probability quality metrics
        brier = float(brier_score_loss(y_t, y_p))
        clipped_prob = np.clip(y_p, 1e-15, 1.0 - 1e-15)
        # log_loss requires both classes in labels argument if one class is missing in y_t
        try:
            ll = float(log_loss(y_t, clipped_prob, labels=[0, 1]))
        except Exception:
            ll = np.nan

        return {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "threshold": float(threshold),
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "balanced_accuracy": bal_acc,
            "brier_score": brier,
            "log_loss": ll,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "n_obs": int(len(y_t)),
            "n_pos": n_pos,
            "prevalence": float(n_pos / len(y_t)) if len(y_t) > 0 else np.nan,
        }

    @staticmethod
    def find_optimal_f1_threshold(
        y_true: Union[np.ndarray, pd.Series],
        y_prob: Union[np.ndarray, pd.Series],
        min_threshold: float = 0.05,
        max_threshold: float = 0.95,
        step: float = 0.01,
        fallback_threshold: float = 0.50,
    ) -> Tuple[float, float]:
        """Finds the decision threshold maximizing F1 on the provided predictions.

        Deterministic tie-breaking:
            If multiple candidate thresholds produce the identical maximum F1 score,
            selects the threshold closest to 0.50 (minimum deviation from center).

        Args:
            y_true: Ground truth binary labels.
            y_prob: Predicted probability array.
            min_threshold: Lower threshold search bound (default: 0.05).
            max_threshold: Upper threshold search bound (default: 0.95).
            step: Search resolution (default: 0.01).
            fallback_threshold: Fallback if optimization is impossible (default: 0.50).

        Returns:
            Tuple[float, float]: (optimal_threshold, maximum_f1).
        """
        y_true_arr = np.asarray(y_true, dtype=float)
        y_prob_arr = np.asarray(y_prob, dtype=float)
        valid_mask = np.isfinite(y_true_arr) & np.isfinite(y_prob_arr)

        if not np.any(valid_mask):
            return fallback_threshold, 0.0

        y_t = y_true_arr[valid_mask].astype(int)
        y_p = y_prob_arr[valid_mask]

        if np.sum(y_t == 1) == 0:
            return fallback_threshold, 0.0

        thresholds = np.arange(min_threshold, max_threshold + step / 2, step)
        best_threshold = fallback_threshold
        best_f1 = -1.0
        best_dist = float("inf")

        for th in thresholds:
            y_pred = (y_p >= th).astype(int)
            score = f1_score(y_t, y_pred, zero_division=0)
            dist_to_center = abs(th - 0.50)

            if score > best_f1:
                best_f1 = score
                best_threshold = float(round(th, 4))
                best_dist = dist_to_center
            elif np.isclose(score, best_f1, atol=1e-6) and dist_to_center < best_dist:
                best_threshold = float(round(th, 4))
                best_dist = dist_to_center

        return best_threshold, float(best_f1)

    @staticmethod
    def evaluate_by_cap_segment(
        df: pd.DataFrame,
        segment_col: str = "cap_segment",
        target_col: str = "hre_5d_train",
        prob_col: str = "hre_prob",
        threshold: float = 0.50,
    ) -> pd.DataFrame:
        """Evaluates model performance stratified by market cap segment.

        Args:
            df: Scored validation/test DataFrame.
            segment_col: Cap segment column name ('large', 'mid', 'small').
            target_col: Ground truth target column.
            prob_col: Prediction probability column.
            threshold: Binary decision threshold.

        Returns:
            pd.DataFrame: Table of per-segment performance metrics.
        """
        if segment_col not in df.columns:
            raise ValueError(f"Column '{segment_col}' not found in DataFrame.")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        if prob_col not in df.columns:
            raise ValueError(f"Probability column '{prob_col}' not found in DataFrame.")

        records = []
        segments = sorted(df[segment_col].dropna().astype(str).str.lower().unique())

        for seg in segments:
            seg_df = df[df[segment_col].astype(str).str.lower() == seg]
            metrics = ModelEvaluator.compute_classification_metrics(
                seg_df[target_col], seg_df[prob_col], threshold=threshold
            )
            metrics[segment_col] = seg
            records.append(metrics)

        # Overall summary row
        overall = ModelEvaluator.compute_classification_metrics(
            df[target_col], df[prob_col], threshold=threshold
        )
        overall[segment_col] = "ALL"
        records.append(overall)

        res_df = pd.DataFrame(records)
        cols = [segment_col] + [c for c in res_df.columns if c != segment_col]
        return res_df[cols]
