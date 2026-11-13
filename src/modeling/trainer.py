"""Walk-forward model training orchestrator.

Responsible for:
- Implementing expanding-window walk-forward validation (10 chronological folds).
- Enforcing a 5-trading-session purge buffer to eliminate forward target lookahead leakage.
- Maintaining identical chronological fold boundaries across supervised models.
- Managing model-specific preprocessing pipelines (Logistic Regression, Random Forest A/B, HistGradientBoosting).
- Enforcing strict information isolation: fitting transformers exclusively on training folds,
  transforming validation folds without refitting.
- Preserving the 250-session final holdout completely untouched.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

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


class ModelTrainer:
    """Orchestrates expanding-window walk-forward training across candidate models."""

    def __init__(
        self,
        n_splits: int = 10,
        min_train_days: int = 490,
        test_window_days: int = 60,
        embargo_gap_days: int = 1,
        purge_sessions: int = 5,
        models_dir: Path | str = "models",
    ) -> None:
        """Initializes the ModelTrainer.

        Args:
            n_splits: Number of walk-forward chronological splits (default: 10).
            min_train_days: Minimum training trading sessions required (default: 490).
            test_window_days: Size of each validation test window in trading sessions (default: 60).
            embargo_gap_days: Autoregressive embargo gap (default: 1).
            purge_sessions: Target lookahead purge buffer in trading sessions (default: 5).
                Eliminates training observations whose 5-session forward window [t+1, t+5]
                touches the validation period.
            models_dir: Directory where trained artifacts will be saved.
        """
        self.n_splits = n_splits
        self.min_train_days = min_train_days
        self.test_window_days = test_window_days
        self.embargo_gap_days = embargo_gap_days
        self.purge_sessions = purge_sessions
        self.models_dir = Path(models_dir)

    def generate_walk_forward_splits(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        target_col: Optional[str] = "hre_5d_train",
        purge_sessions: Optional[int] = None,
    ) -> List[Tuple[pd.Index, pd.Index]]:
        """Generates chronological train/validation index splits with 5-session target purging.

        Mathematical Leakage Safeguard:
            The forward target `hre_5d_train` evaluates LRS over [t+1, t+5].
            For a validation period starting at trading session T_val, any training session
            in [T_val - 5, T_val - 1] has a forward evaluation window extending into the
            validation period.
            The purge buffer drops these 5 sessions strictly using trading-session ordering.

        Population Rules Enforced:
            - Excludes holdout observations ('split' == 'holdout').
            - Excludes train boundary embargo observations ('split' == 'train_boundary_embargo').
            - Excludes feature warmup observations ('is_warmup' == True).
            - For supervised models (when target_col is provided), requires target_col.notna().

        Args:
            df: DataFrame containing date, features, and optionally partition/target columns.
            date_col: Name of date column (or uses DatetimeIndex if absent).
            target_col: Target column name for supervised eligibility filtering.
                Set to None for unsupervised population (e.g. Isolation Forest).
            purge_sessions: Override for target lookahead purge sessions (defaults to self.purge_sessions).

        Returns:
            List[Tuple[pd.Index, pd.Index]]: List of (train_idx, val_idx) row indices.

        Raises:
            ValueError: If trading dates are insufficient for the configured fold parameters.
        """
        p = purge_sessions if purge_sessions is not None else self.purge_sessions

        # Extract dates
        if date_col in df.columns:
            date_series = pd.to_datetime(df[date_col])
        elif isinstance(df.index, pd.DatetimeIndex):
            date_series = pd.Series(df.index, index=df.index)
        else:
            raise ValueError(f"Date column '{date_col}' not found and index is not a DatetimeIndex.")

        # Filter candidate dates to eligible historical training dates
        eligible_mask = pd.Series(True, index=df.index)
        if "split" in df.columns:
            # Strictly exclude holdout and train_boundary_embargo
            eligible_mask &= (df["split"] == "train")
        if "is_warmup" in df.columns:
            # Strictly exclude initial warmup window
            eligible_mask &= (~df["is_warmup"])

        eligible_dates = sorted(date_series[eligible_mask].unique())
        n_dates = len(eligible_dates)

        min_val_start = self.min_train_days + p
        max_val_start = n_dates - self.test_window_days

        if max_val_start < min_val_start:
            raise ValueError(
                f"Insufficient eligible trading dates ({n_dates}) for min_train_days="
                f"{self.min_train_days}, purge_sessions={p}, and test_window_days="
                f"{self.test_window_days}."
            )

        # Compute validation window starting positions
        # If contiguous windows ending at the final date satisfy min_train_days, tile contiguously
        contiguous_first_val_start = n_dates - self.n_splits * self.test_window_days
        if contiguous_first_val_start >= min_val_start:
            val_starts = [
                n_dates - (self.n_splits - i) * self.test_window_days
                for i in range(self.n_splits)
            ]
        else:
            # Evenly space the validation windows across available date range
            if self.n_splits == 1:
                val_starts = [max_val_start]
            else:
                step = (max_val_start - min_val_start) / (self.n_splits - 1)
                val_starts = [int(round(min_val_start + i * step)) for i in range(self.n_splits)]

        splits: List[Tuple[pd.Index, pd.Index]] = []

        for val_start_idx in val_starts:
            val_end_idx = min(val_start_idx + self.test_window_days - 1, n_dates - 1)
            train_end_idx = val_start_idx - p - 1

            train_dates = set(eligible_dates[0 : train_end_idx + 1])
            val_dates = set(eligible_dates[val_start_idx : val_end_idx + 1])

            # Validation mask: rows on validation dates
            val_mask = date_series.isin(val_dates)
            if "split" in df.columns:
                val_mask &= (df["split"] == "train")
            if "is_warmup" in df.columns:
                val_mask &= (~df["is_warmup"])

            # Training mask: rows on training dates
            train_mask = date_series.isin(train_dates)
            if "split" in df.columns:
                train_mask &= (df["split"] == "train")
            if "is_warmup" in df.columns:
                train_mask &= (~df["is_warmup"])
            if target_col is not None and target_col in df.columns:
                # Supervised eligibility requires non-null target
                train_mask &= df[target_col].notna()

            train_idx = df.index[train_mask]
            val_idx = df.index[val_mask]

            splits.append((train_idx, val_idx))

        return splits

    def get_model_pipeline(self, model_name: str, **kwargs: Any) -> Pipeline:
        """Returns a fresh un-fitted Pipeline instance for the requested model.

        Args:
            model_name: Identifier ('logistic_regression', 'random_forest',
                'random_forest_candidate_b', 'hist_gradient_boosting', 'isolation_forest').
            **kwargs: Hyperparameters forwarded to the specific pipeline builder.

        Returns:
            Pipeline: Fresh un-fitted sklearn Pipeline.

        Raises:
            ValueError: If model_name is unrecognized.
        """
        key = model_name.strip().lower()
        if key in ("logistic_regression", "lr", "baseline"):
            return build_logistic_regression_pipeline(**kwargs)
        elif key in ("random_forest", "rf", "random_forest_candidate_a", "rf_a"):
            return build_random_forest_pipeline(candidate="A", **kwargs)
        elif key in ("random_forest_candidate_b", "rf_b"):
            return build_random_forest_pipeline(candidate="B", **kwargs)
        elif key in ("hist_gradient_boosting", "hgbt", "boosting"):
            return build_hgbt_pipeline(**kwargs)
        elif key in ("isolation_forest", "if", "anomaly"):
            return build_isolation_forest_pipeline(**kwargs)
        else:
            raise ValueError(
                f"Unrecognized model_name '{model_name}'. Supported: "
                "'logistic_regression', 'random_forest', 'random_forest_candidate_b', "
                "'hist_gradient_boosting', 'isolation_forest'."
            )

    def get_model_features(self, model_name: str, candidate: str = "A") -> List[str]:
        """Returns the approved feature list for a given model identifier."""
        return get_model_features(model_name=model_name, candidate=candidate)

    def fit_fold_pipeline(
        self,
        model_name: str,
        df: pd.DataFrame,
        train_idx: pd.Index,
        val_idx: pd.Index,
        target_col: str = "hre_5d_train",
        **pipeline_kwargs: Any,
    ) -> Tuple[Pipeline, pd.Series]:
        """Fits an independent model pipeline strictly on a training fold and predicts on validation.

        Enforces strict leakage boundaries:
            - Instantiates a fresh pipeline instance.
            - Fits transformers and estimator strictly on `df.loc[train_idx]`.
            - Calls `predict_proba` on `df.loc[val_idx]` without fitting on validation.

        Args:
            model_name: Model identifier.
            df: Full DataFrame containing features and target.
            train_idx: Permitted training row indices for this fold.
            val_idx: Validation row indices for this fold.
            target_col: Target column name.
            **pipeline_kwargs: Hyperparameters for pipeline builder.

        Returns:
            Tuple[Pipeline, pd.Series]: Fitted pipeline and validation predicted probabilities.
        """
        candidate = "B" if "candidate_b" in model_name.lower() or model_name.lower() == "rf_b" else "A"
        features = self.get_model_features(model_name, candidate=candidate)
        validate_features_present(df, features)

        pipeline = self.get_model_pipeline(model_name, **pipeline_kwargs)

        X_train = df.loc[train_idx, features]
        y_train = df.loc[train_idx, target_col]
        X_val = df.loc[val_idx, features]

        # Fit strictly on training observations
        pipeline.fit(X_train, y_train)

        # Predict out-of-fold validation probabilities without fitting
        y_val_prob = pd.Series(
            pipeline.predict_proba(X_val)[:, 1],
            index=val_idx,
            name=f"{model_name}_prob",
        )

        return pipeline, y_val_prob

    def train_and_evaluate_models(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "hre_5d_train",
    ) -> Dict[str, Any]:
        """Runs walk-forward evaluation across LogisticRegression, RF, and HGBT.

        NOTE: Walk-forward model training experiments and performance comparisons
        are deferred to Phase 8B per Batch 3 pre-implementation authorization.

        Args:
            df: Labeled feature matrix.
            feature_cols: Optional override list of feature columns.
            target_col: Target column name.

        Raises:
            NotImplementedError: Batch 3 is strictly an implementation/engineering phase.
        """
        raise NotImplementedError(
            "Walk-forward model training experiments and performance comparisons "
            "are deferred to Phase 8B per Batch 3 pre-implementation authorization."
        )

    def fit_final_model(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        best_model_name: str,
        target_col: str = "hre_5d_train",
    ) -> Path:
        """Fits best model on full training set and serializes artifact with metadata.

        NOTE: Final model fitting and artifact serialization are deferred to Phase 8B/10.

        Raises:
            NotImplementedError: Batch 3 is strictly an implementation/engineering phase.
        """
        raise NotImplementedError(
            "Final model training and serialization are deferred to Phase 8B/10."
        )
