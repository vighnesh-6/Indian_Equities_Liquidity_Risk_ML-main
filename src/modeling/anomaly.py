"""Unsupervised anomaly detection module.

Responsible for:
- Fitting `IsolationForest` on historical feature subspace (12 features from model_config.yaml).
- Generating `isolation_score`, `isolation_flag`, and `anomaly_score_scaled` (0–100 range).
- STRICT ARCHITECTURAL PRINCIPLE: Isolation Forest detects historical unusualness,
  NOT forward risk. It strictly never receives supervised target labels (`hre_5d_train`).
- Scale-invariant: Random axis-aligned partitioning is invariant to monotonic transformations,
  so raw features post-median imputation are used.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.modeling.preprocessing import (
    ISOLATION_FOREST_FEATURES,
    build_isolation_forest_pipeline,
    validate_features_present,
)


class AnomalyDetector:
    """Unsupervised anomaly detection using Isolation Forest."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        random_state: int = 42,
        feature_subset: Optional[List[str]] = None,
    ) -> None:
        """Initializes the AnomalyDetector.

        Args:
            n_estimators: Number of isolation trees (default: 100).
            contamination: Expected proportion of outliers in the data (default: 0.05).
            random_state: Random state for reproducibility (default: 42).
            feature_subset: List of feature names to use. Defaults to approved
                12-feature subspace from configs/model_config.yaml.
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.feature_subset = list(feature_subset) if feature_subset is not None else list(ISOLATION_FOREST_FEATURES)
        self.pipeline: Optional[Pipeline] = None
        self.model: Optional[Any] = None
        self.score_min_: Optional[float] = None
        self.score_max_: Optional[float] = None

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """Fits Isolation Forest on historical feature observations.

        Population Rules Enforced:
            - Strictly excludes holdout observations ('split' == 'holdout').
            - Strictly excludes initial warmup observations ('is_warmup' == True).
            - Strictly DOES NOT filter by forward target labels (`hre_5d_train.notna()`).
              Maintains complete target independence (466,486 training rows).

        Args:
            df: Historical training feature matrix.

        Returns:
            AnomalyDetector: Fitted instance.
        """
        validate_features_present(df, self.feature_subset)

        # Select eligible historical training population
        mask = pd.Series(True, index=df.index)
        if "split" in df.columns:
            mask &= (df["split"] == "train")
        if "is_warmup" in df.columns:
            mask &= (~df["is_warmup"])

        X_train = df.loc[mask, self.feature_subset]

        # Build fresh independent pipeline
        self.pipeline = build_isolation_forest_pipeline(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )

        # Fit strictly on historical features
        self.pipeline.fit(X_train)
        self.model = self.pipeline.named_steps["model"]

        # Establish training score bounds for 0-100 rescaling
        train_scores = self.pipeline.score_samples(X_train)
        self.score_min_ = float(np.min(train_scores))
        self.score_max_ = float(np.max(train_scores))

        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scores observations and returns rescaled 0-100 anomaly scores.

        Inference only: strictly applies the fitted pipeline without refitting.

        Args:
            df: Feature DataFrame to score.

        Returns:
            pd.DataFrame: Contains:
                - `isolation_score`: Raw decision score from score_samples (lower = more anomalous).
                - `isolation_flag`: Binary indicator (1 = anomalous, 0 = normal).
                - `anomaly_score_scaled`: Rescaled score in 0–100 range (higher = more anomalous).
        """
        if self.pipeline is None:
            raise RuntimeError("AnomalyDetector has not been fitted yet. Call fit() first.")

        validate_features_present(df, self.feature_subset)
        X = df[self.feature_subset]

        # Raw scores: higher is more normal, lower is more anomalous
        raw_scores = self.pipeline.score_samples(X)
        preds = self.pipeline.predict(X)

        # Isolation flag: predict returns -1 for outlier, 1 for inlier
        isolation_flag = (preds == -1).astype(int)

        # Rescale to 0-100 where 100 = most anomalous, 0 = most normal
        if self.score_min_ is not None and self.score_max_ is not None and self.score_max_ != self.score_min_:
            denom = self.score_max_ - self.score_min_
            scaled_scores = 100.0 * (self.score_max_ - raw_scores) / denom
            scaled_scores = np.clip(scaled_scores, 0.0, 100.0)
        else:
            scaled_scores = np.zeros_like(raw_scores)

        return pd.DataFrame(
            {
                "isolation_score": raw_scores,
                "isolation_flag": isolation_flag,
                "anomaly_score_scaled": scaled_scores,
            },
            index=df.index,
        )

    def save_artifact(self, dest_path: Path | str) -> Path:
        """Saves fitted Isolation Forest model artifact.

        Args:
            dest_path: Destination file path.

        Returns:
            Path: Written artifact path.
        """
        if self.pipeline is None:
            raise RuntimeError("Cannot save un-fitted AnomalyDetector.")
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        return dest

    @classmethod
    def load_artifact(cls, src_path: Path | str) -> "AnomalyDetector":
        """Loads serialized AnomalyDetector artifact from disk.

        Args:
            src_path: Path to serialized artifact.

        Returns:
            AnomalyDetector: Restored instance.
        """
        path = Path(src_path)
        if not path.exists():
            raise FileNotFoundError(f"AnomalyDetector artifact not found at '{src_path}'.")
        instance = joblib.load(path)
        if not isinstance(instance, cls):
            raise TypeError(f"Loaded object is of type {type(instance)}, expected {cls.__name__}.")
        return instance
