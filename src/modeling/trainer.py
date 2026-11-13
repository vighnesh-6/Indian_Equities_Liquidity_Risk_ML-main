"""Walk-forward model training orchestrator.

Responsible for:
- Implementing expanding-window walk-forward validation (10 chronological folds).
- Enforcing a 5-trading-session purge buffer to eliminate forward target lookahead leakage.
- Maintaining identical chronological fold boundaries across supervised models.
- Managing model-specific preprocessing pipelines (Logistic Regression, Random Forest A/B, HistGradientBoosting).
- Enforcing strict information isolation: fitting transformers exclusively on training folds,
  transforming validation folds without refitting.
- Enforcing inner chronological threshold calibration to avoid outer validation label leakage.
- Preserving the 250-session final holdout completely untouched.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
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
from src.modeling.evaluation import ModelEvaluator
from src.modeling.anomaly import AnomalyDetector


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
            model_name: Identifier ('logistic_regression', 'logistic_regression_standard',
                'logistic_regression_robust', 'random_forest_candidate_a',
                'random_forest_candidate_b', 'hist_gradient_boosting', 'isolation_forest').
            **kwargs: Hyperparameters forwarded to the specific pipeline builder.

        Returns:
            Pipeline: Fresh un-fitted sklearn Pipeline.

        Raises:
            ValueError: If model_name is unrecognized.
        """
        key = model_name.strip().lower()
        if key in ("logistic_regression", "lr", "baseline", "logistic_regression_standard", "lr_standard"):
            scaler = kwargs.pop("scaler_type", "standard")
            return build_logistic_regression_pipeline(scaler_type=scaler, **kwargs)
        elif key in ("logistic_regression_robust", "lr_robust"):
            scaler = kwargs.pop("scaler_type", "robust")
            return build_logistic_regression_pipeline(scaler_type=scaler, **kwargs)
        elif key in ("random_forest", "rf", "random_forest_candidate_a", "rf_a", "rf_candidate_a"):
            cand = kwargs.pop("candidate", "A")
            return build_random_forest_pipeline(candidate=cand, **kwargs)
        elif key in ("random_forest_candidate_b", "rf_b", "rf_candidate_b"):
            cand = kwargs.pop("candidate", "B")
            return build_random_forest_pipeline(candidate=cand, **kwargs)
        elif key in ("hist_gradient_boosting", "hgbt", "boosting"):
            return build_hgbt_pipeline(**kwargs)
        elif key in ("isolation_forest", "if", "anomaly"):
            return build_isolation_forest_pipeline(**kwargs)
        else:
            raise ValueError(
                f"Unrecognized model_name '{model_name}'. Supported: "
                "'logistic_regression_standard', 'logistic_regression_robust', "
                "'random_forest_candidate_a', 'random_forest_candidate_b', "
                "'hist_gradient_boosting', 'isolation_forest'."
            )

    def get_model_features(self, model_name: str, candidate: str = "A") -> List[str]:
        """Returns the approved feature list for a given model identifier."""
        return get_model_features(model_name=model_name, candidate=candidate)

    def calibrate_inner_threshold(
        self,
        model_name: str,
        df: pd.DataFrame,
        train_dates: List[pd.Timestamp],
        target_col: str = "hre_5d_train",
        calib_window_days: int = 60,
        purge_sessions: int = 5,
        **pipeline_kwargs: Any,
    ) -> Tuple[float, float, str]:
        """Calibrates classification threshold strictly on inner chronological training history.

        Zero Outer Leakage Protocol:
            1. Reserves the final `calib_window_days` of the outer training window as an inner calibration window.
            2. Enforces a 5-session purge prior to the inner calibration window.
            3. Fits a fresh model on inner training observations preceding the purge.
            4. Predicts out-of-sample probabilities on the inner calibration window.
            5. Selects threshold maximizing F1 on inner calibration data.
            6. The outer validation fold is never accessed or influenced.

        Args:
            model_name: Model identifier.
            df: Full labeled DataFrame.
            train_dates: Sorted list of trading dates in outer training fold.
            target_col: Target column name.
            calib_window_days: Size of inner calibration window (default: 60).
            purge_sessions: Inner purge sessions (default: 5).
            **pipeline_kwargs: Model builder arguments.

        Returns:
            Tuple[float, float, str]: (calibrated_threshold, inner_calib_f1, calibration_status).
        """
        n_dates = len(train_dates)
        min_required = calib_window_days + purge_sessions + 60

        if n_dates < min_required:
            return 0.50, float("nan"), "fallback_insufficient_history"

        # Chronological inner window boundaries
        inner_calib_dates = set(train_dates[-calib_window_days:])
        inner_purge_dates = set(train_dates[-calib_window_days - purge_sessions : -calib_window_days])
        inner_train_dates = set(train_dates[: -calib_window_days - purge_sessions])

        # Date series
        date_series = pd.to_datetime(df["date"])

        inner_train_mask = date_series.isin(inner_train_dates)
        inner_calib_mask = date_series.isin(inner_calib_dates)

        if "split" in df.columns:
            inner_train_mask &= (df["split"] == "train")
            inner_calib_mask &= (df["split"] == "train")
        if "is_warmup" in df.columns:
            inner_train_mask &= (~df["is_warmup"])
            inner_calib_mask &= (~df["is_warmup"])

        inner_train_mask &= df[target_col].notna()
        inner_calib_mask &= df[target_col].notna()

        y_calib = df.loc[inner_calib_mask, target_col].values
        if len(y_calib) == 0 or np.sum(y_calib == 1) == 0 or np.sum(y_calib == 0) == 0:
            return 0.50, float("nan"), "fallback_single_class"

        # Fit fresh inner pipeline
        candidate = "B" if ("candidate_b" in model_name.lower() or model_name.lower() == "rf_b") else "A"
        features = self.get_model_features(model_name, candidate=candidate)

        inner_pipeline = self.get_model_pipeline(model_name, **pipeline_kwargs)
        X_inner_train = df.loc[inner_train_mask, features]
        y_inner_train = df.loc[inner_train_mask, target_col]

        inner_pipeline.fit(X_inner_train, y_inner_train)

        # Predict out-of-fold on inner calibration window
        X_inner_calib = df.loc[inner_calib_mask, features]
        prob_calib = inner_pipeline.predict_proba(X_inner_calib)[:, 1]

        best_th, best_f1 = ModelEvaluator.find_optimal_f1_threshold(
            y_calib, prob_calib, min_threshold=0.10, max_threshold=0.90, step=0.02
        )

        return best_th, best_f1, "calibrated"

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
        candidate = "B" if ("candidate_b" in model_name.lower() or model_name.lower() == "rf_b") else "A"
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
        models: Optional[List[str]] = None,
        target_col: str = "hre_5d_train",
        output_dir: Optional[Union[str, Path]] = None,
        eval_isolation_forest: bool = True,
        calibrate_thresholds: bool = True,
        **model_kwargs: Any,
    ) -> Dict[str, Any]:
        """Executes full 10-fold expanding-window walk-forward validation across candidate models.

        Methodological Invariants Enforced:
            - 10 chronological expanding folds (60 trading days validation each).
            - 5-trading-session purge between training and validation.
            - Identical outer fold boundaries across all models.
            - Preprocessing objects instantiated fresh and fit strictly on training data.
            - Threshold calibration conducted strictly on inner training partitions.
            - Final holdout (250 sessions) strictly excluded.

        Args:
            df: Full features and targets DataFrame.
            models: List of model identifiers to evaluate. Defaults to 5 approved supervised models.
            target_col: Target column name ('hre_5d_train').
            output_dir: Destination path to persist prediction parquet and summary csv files.
            eval_isolation_forest: Whether to evaluate unsupervised Isolation Forest.
            calibrate_thresholds: Whether to perform inner chronological threshold calibration.
            **model_kwargs: Overrides passed to model builders.

        Returns:
            Dict[str, Any]: Structured results containing fold metrics, out-of-fold predictions,
                model comparison tables, and Isolation Forest diagnostics.
        """
        if models is None:
            models = [
                "logistic_regression_standard",
                "logistic_regression_robust",
                "random_forest_candidate_a",
                "random_forest_candidate_b",
                "hist_gradient_boosting",
            ]

        out_path = Path(output_dir) if output_dir else None
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)

        # Generate outer walk-forward splits
        splits_supervised = self.generate_walk_forward_splits(df, target_col=target_col)
        if eval_isolation_forest:
            splits_unsupervised = self.generate_walk_forward_splits(df, target_col=None)
        else:
            splits_unsupervised = []

        all_train_dates = sorted(
            df[(df["split"] == "train") & (~df["is_warmup"])]["date"].unique()
        )

        supervised_fold_records: List[Dict[str, Any]] = []
        all_supervised_preds: Dict[str, List[pd.DataFrame]] = {m: [] for m in models}
        if_fold_records: List[Dict[str, Any]] = []
        if_preds_list: List[pd.DataFrame] = []

        # Load existing artifacts if resuming
        if out_path:
            metrics_csv = out_path / "supervised_fold_metrics.csv"
            preds_parquet = out_path / "supervised_predictions.parquet"
            if metrics_csv.exists() and preds_parquet.exists():
                try:
                    loaded_metrics = pd.read_csv(metrics_csv).to_dict("records")
                    loaded_preds = pd.read_parquet(preds_parquet)
                    # Find folds completed by ALL models
                    completed_folds = set()
                    available_folds = set(int(r.get("fold")) for r in loaded_metrics)
                    for f in available_folds:
                        if all(any(r.get("model_name") == m and int(r.get("fold")) == f for r in loaded_metrics) for m in models):
                            completed_folds.add(f)

                    for rec in loaded_metrics:
                        m = rec.get("model_name")
                        f = int(rec.get("fold"))
                        if m in models and f in completed_folds:
                            supervised_fold_records.append(rec)

                    for m in models:
                        for f in sorted(completed_folds):
                            sub_pred = loaded_preds[loaded_preds["fold"] == f]
                            if not sub_pred.empty and f"prob_{m}" in sub_pred.columns:
                                p_df = pd.DataFrame(
                                    {
                                        "fold": f,
                                        "date": sub_pred["date"].values,
                                        "ticker": sub_pred["ticker"].values,
                                        "cap_segment": sub_pred["cap_segment"].values if "cap_segment" in sub_pred.columns else "unknown",
                                        "y_true": sub_pred["y_true"].values,
                                        f"prob_{m}": sub_pred[f"prob_{m}"].values,
                                        f"pred_50_{m}": sub_pred[f"pred_50_{m}"].values,
                                        f"pred_calib_{m}": sub_pred[f"pred_calib_{m}"].values,
                                    },
                                    index=sub_pred.index,
                                )
                                all_supervised_preds[m].append(p_df)
                    print(f"Loaded {len(supervised_fold_records)} existing model-fold metrics across completed folds {sorted(completed_folds)} from {metrics_csv}", flush=True)
                except Exception as e:
                    print(f"Warning: Could not resume from existing files: {e}", flush=True)

            if_metrics_csv = out_path / "isolation_forest_fold_metrics.csv"
            if_preds_parquet = out_path / "isolation_forest_predictions.parquet"
            if if_metrics_csv.exists() and if_preds_parquet.exists():
                try:
                    loaded_if_recs = pd.read_csv(if_metrics_csv).to_dict("records")
                    if_loaded_preds = pd.read_parquet(if_preds_parquet)
                    for rec in loaded_if_recs:
                        f = int(rec.get("fold"))
                        if f in completed_folds:
                            if_fold_records.append(rec)
                            sub_if = if_loaded_preds[if_loaded_preds["fold"] == f]
                            if not sub_if.empty:
                                if_preds_list.append(sub_if)
                    print(f"Loaded {len(if_fold_records)} existing Isolation Forest fold metrics from {if_metrics_csv}", flush=True)
                except Exception as e:
                    print(f"Warning: Could not resume IF files: {e}", flush=True)

        def save_checkpoints() -> None:
            if not out_path:
                return
            if supervised_fold_records:
                sf_df = pd.DataFrame(supervised_fold_records)
                sf_df.to_csv(out_path / "supervised_fold_metrics.csv", index=False)

                # Aggregate model comparison
                comp_records = []
                for m in models:
                    m_sub = sf_df[sf_df["model_name"] == m]
                    if m_sub.empty or len(all_supervised_preds[m]) == 0:
                        continue
                    m_pred = pd.concat(all_supervised_preds[m]).sort_values(["date", "ticker"])
                    pooled_50 = ModelEvaluator.compute_classification_metrics(m_pred["y_true"], m_pred[f"prob_{m}"], threshold=0.50)
                    cap_metrics = ModelEvaluator.evaluate_by_cap_segment(
                        m_pred.rename(columns={"y_true": "hre_5d_train", f"prob_{m}": "hre_prob"}),
                        segment_col="cap_segment",
                        target_col="hre_5d_train",
                        prob_col="hre_prob",
                        threshold=0.50,
                    )
                    rec = {
                        "model_name": m,
                        "mean_pr_auc": float(m_sub["pr_auc"].mean()),
                        "std_pr_auc": float(m_sub["pr_auc"].std()) if len(m_sub) > 1 else float("nan"),
                        "min_pr_auc": float(m_sub["pr_auc"].min()),
                        "max_pr_auc": float(m_sub["pr_auc"].max()),
                        "pooled_pr_auc": pooled_50["pr_auc"],
                        "mean_roc_auc": float(m_sub["roc_auc"].mean()),
                        "std_roc_auc": float(m_sub["roc_auc"].std()) if len(m_sub) > 1 else float("nan"),
                        "pooled_roc_auc": pooled_50["roc_auc"],
                        "mean_brier_score": float(m_sub["brier_score"].mean()),
                        "pooled_brier_score": pooled_50["brier_score"],
                        "mean_log_loss": float(m_sub["log_loss"].mean()),
                        "pooled_log_loss": pooled_50["log_loss"],
                        "mean_f1_50": float(m_sub["f1_50"].mean()),
                        "mean_precision_50": float(m_sub["precision_50"].mean()),
                        "mean_recall_50": float(m_sub["recall_50"].mean()),
                        "mean_f1_calib": float(m_sub["f1_calib"].mean()),
                        "mean_precision_calib": float(m_sub["precision_calib"].mean()),
                        "mean_recall_calib": float(m_sub["recall_calib"].mean()),
                        "mean_calib_threshold": float(m_sub["calibrated_threshold"].mean()),
                        "total_fit_time_sec": float(m_sub["fit_time_sec"].sum()),
                        "mean_fit_time_sec": float(m_sub["fit_time_sec"].mean()),
                        "large_cap_pr_auc": float(
                            cap_metrics.loc[cap_metrics["cap_segment"] == "large", "pr_auc"].values[0]
                            if "large" in cap_metrics["cap_segment"].values else np.nan
                        ),
                        "mid_cap_pr_auc": float(
                            cap_metrics.loc[cap_metrics["cap_segment"] == "mid", "pr_auc"].values[0]
                            if "mid" in cap_metrics["cap_segment"].values else np.nan
                        ),
                        "small_cap_pr_auc": float(
                            cap_metrics.loc[cap_metrics["cap_segment"] == "small", "pr_auc"].values[0]
                            if "small" in cap_metrics["cap_segment"].values else np.nan
                        ),
                    }
                    comp_records.append(rec)
                if comp_records:
                    pd.DataFrame(comp_records).sort_values("mean_pr_auc", ascending=False).to_csv(
                        out_path / "supervised_model_comparison.csv", index=False
                    )

                # Predictions parquet: only save if all active models have the same number of prediction dfs
                active_models = [m for m in models if len(all_supervised_preds[m]) > 0]
                if active_models:
                    min_len = min(len(all_supervised_preds[m]) for m in active_models)
                    max_len = max(len(all_supervised_preds[m]) for m in active_models)
                    if min_len == max_len and min_len > 0:
                        first_m = active_models[0]
                        base_p = pd.concat(all_supervised_preds[first_m]).sort_values(["date", "ticker"])
                        consolidated = base_p[["fold", "date", "ticker", "cap_segment", "y_true"]].copy()
                        for m in active_models:
                            m_p = pd.concat(all_supervised_preds[m]).sort_values(["date", "ticker"])
                            consolidated[f"prob_{m}"] = m_p[f"prob_{m}"].values
                            consolidated[f"pred_50_{m}"] = m_p[f"pred_50_{m}"].values
                            consolidated[f"pred_calib_{m}"] = m_p[f"pred_calib_{m}"].values
                        consolidated.to_parquet(out_path / "supervised_predictions.parquet")

            if if_fold_records:
                pd.DataFrame(if_fold_records).to_csv(out_path / "isolation_forest_fold_metrics.csv", index=False)
            if if_preds_list:
                pd.concat(if_preds_list).sort_values(["date", "ticker"]).to_parquet(out_path / "isolation_forest_predictions.parquet")

        print(f"\n{'='*80}\nBEGINNING WALK-FORWARD VALIDATION ({len(splits_supervised)} FOLDS x {len(models)} MODELS)\n{'='*80}", flush=True)

        # Iterate through outer walk-forward folds
        for fold_idx in range(len(splits_supervised)):
            train_idx, val_idx = splits_supervised[fold_idx]
            val_df = df.loc[val_idx]
            train_df = df.loc[train_idx]

            t_dates = sorted(train_df["date"].unique())
            v_dates = sorted(val_df["date"].unique())

            idx_t_end = all_train_dates.index(t_dates[-1])
            idx_v_start = all_train_dates.index(v_dates[0])
            purged_dates = [str(d)[:10] for d in all_train_dates[idx_t_end + 1 : idx_v_start]]

            print(
                f"\n>>> FOLD {fold_idx + 1}/10: Train [{str(t_dates[0])[:10]} to {str(t_dates[-1])[:10]}] "
                f"({len(train_idx):,} rows, pos={int(train_df[target_col].sum()):,}) | "
                f"Purge ({len(purged_dates)} d: {purged_dates}) | "
                f"Val [{str(v_dates[0])[:10]} to {str(v_dates[-1])[:10]}] ({len(val_idx):,} rows, pos={int(val_df[target_col].sum()):,})",
                flush=True,
            )

            # Evaluate each supervised model on this fold
            for model_name in models:
                already_done = any(
                    r.get("model_name") == model_name and int(r.get("fold")) == fold_idx
                    for r in supervised_fold_records
                )
                if already_done:
                    print(f"  [{model_name:28s}] Reusing cached Fold {fold_idx} results.", flush=True)
                    continue

                t0_fit = time.perf_counter()

                # Inner chronological threshold calibration
                if calibrate_thresholds:
                    calib_th, inner_f1, calib_status = self.calibrate_inner_threshold(
                        model_name=model_name,
                        df=df,
                        train_dates=t_dates,
                        target_col=target_col,
                        calib_window_days=self.test_window_days,
                        purge_sessions=self.purge_sessions,
                        **model_kwargs,
                    )
                else:
                    calib_th, inner_f1, calib_status = 0.50, float("nan"), "uncalibrated_0.50"

                # Fit outer pipeline strictly on training fold
                pipeline, val_prob = self.fit_fold_pipeline(
                    model_name=model_name,
                    df=df,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    target_col=target_col,
                    **model_kwargs,
                )
                fit_time = time.perf_counter() - t0_fit

                t0_pred = time.perf_counter()
                y_val_true = val_df[target_col].values
                val_prob_arr = val_prob.values
                predict_time = time.perf_counter() - t0_pred

                # Compute classification metrics at baseline 0.50 and calibrated threshold
                m_50 = ModelEvaluator.compute_classification_metrics(y_val_true, val_prob_arr, threshold=0.50)
                m_calib = ModelEvaluator.compute_classification_metrics(y_val_true, val_prob_arr, threshold=calib_th)

                print(
                    f"  [{model_name:28s}] Fit: {fit_time:5.1f}s | "
                    f"PR-AUC: {m_50['pr_auc']:.4f} | ROC-AUC: {m_50['roc_auc']:.4f} | "
                    f"Brier: {m_50['brier_score']:.4f} | F1@0.50: {m_50['f1']:.4f} | "
                    f"CalibTh: {calib_th:.2f} (F1: {m_calib['f1']:.4f})",
                    flush=True,
                )

                # Store fold metrics record
                record = {
                    "model_name": model_name,
                    "fold": fold_idx,
                    "train_start": str(t_dates[0])[:10],
                    "train_end": str(t_dates[-1])[:10],
                    "train_days": len(t_dates),
                    "train_rows": len(train_idx),
                    "train_pos": int(train_df[target_col].sum()),
                    "train_prev": float(train_df[target_col].mean()),
                    "val_start": str(v_dates[0])[:10],
                    "val_end": str(v_dates[-1])[:10],
                    "val_days": len(v_dates),
                    "val_rows": len(val_idx),
                    "val_pos": int(val_df[target_col].sum()),
                    "val_prev": float(val_df[target_col].mean()),
                    "purged_sessions_count": len(purged_dates),
                    "calibrated_threshold": calib_th,
                    "inner_calib_f1": inner_f1,
                    "calibration_status": calib_status,
                    "fit_time_sec": fit_time,
                    "predict_time_sec": predict_time,
                    "pr_auc": m_50["pr_auc"],
                    "roc_auc": m_50["roc_auc"],
                    "brier_score": m_50["brier_score"],
                    "log_loss": m_50["log_loss"],
                    "f1_50": m_50["f1"],
                    "precision_50": m_50["precision"],
                    "recall_50": m_50["recall"],
                    "balanced_accuracy_50": m_50["balanced_accuracy"],
                    "tp_50": m_50["tp"],
                    "fp_50": m_50["fp"],
                    "tn_50": m_50["tn"],
                    "fn_50": m_50["fn"],
                    "f1_calib": m_calib["f1"],
                    "precision_calib": m_calib["precision"],
                    "recall_calib": m_calib["recall"],
                    "balanced_accuracy_calib": m_calib["balanced_accuracy"],
                    "tp_calib": m_calib["tp"],
                    "fp_calib": m_calib["fp"],
                    "tn_calib": m_calib["tn"],
                    "fn_calib": m_calib["fn"],
                }
                supervised_fold_records.append(record)

                # Store out-of-fold prediction rows
                pred_df = pd.DataFrame(
                    {
                        "fold": fold_idx,
                        "date": val_df["date"].values,
                        "ticker": val_df["ticker"].values,
                        "cap_segment": val_df["cap_segment"].values if "cap_segment" in val_df.columns else "unknown",
                        "y_true": y_val_true,
                        f"prob_{model_name}": val_prob_arr,
                        f"pred_50_{model_name}": (val_prob_arr >= 0.50).astype(int),
                        f"pred_calib_{model_name}": (val_prob_arr >= calib_th).astype(int),
                    },
                    index=val_idx,
                )
                all_supervised_preds[model_name].append(pred_df)

            # Evaluate unsupervised Isolation Forest on this fold
            if eval_isolation_forest:
                already_if_done = any(int(r.get("fold")) == fold_idx for r in if_fold_records)
                if already_if_done:
                    print(f"  [isolation_forest             ] Reusing cached Fold {fold_idx} results.", flush=True)
                else:
                    t0_if = time.perf_counter()
                    unsup_train_idx, unsup_val_idx = splits_unsupervised[fold_idx]
                    unsup_val_df = df.loc[unsup_val_idx]

                    if_detector = AnomalyDetector(n_estimators=100, contamination=0.05, random_state=42)
                    if_detector.fit(df, train_idx=unsup_train_idx)
                    if_fit_time = time.perf_counter() - t0_if

                    t0_if_score = time.perf_counter()
                    val_anomaly_scores = if_detector.score(unsup_val_df)
                    if_score_time = time.perf_counter() - t0_if_score

                    # Compute anomaly distribution and diagnostic overlap with target
                    raw_s = val_anomaly_scores["isolation_score"].values
                    scaled_s = val_anomaly_scores["anomaly_score_scaled"].values
                    flags = val_anomaly_scores["isolation_flag"].values

                    flagged_count = int(np.sum(flags == 1))
                    flagged_rate = float(np.mean(flags == 1))

                    # Diagnostic overlap with supervised target
                    y_val_int = unsup_val_df[target_col].fillna(0).astype(int).values
                    tp_anomaly = int(np.sum((flags == 1) & (y_val_int == 1)))
                    hre_pos_count = int(np.sum(y_val_int == 1))
                    hre_neg_count = int(np.sum(y_val_int == 0))
                    hre_pos_flag_rate = float(tp_anomaly / hre_pos_count) if hre_pos_count > 0 else np.nan
                    hre_neg_flag_rate = (
                        float(np.sum((flags == 1) & (y_val_int == 0)) / hre_neg_count) if hre_neg_count > 0 else np.nan
                    )

                    # Anomaly correlation with forward target
                    valid_t_mask = unsup_val_df[target_col].notna().values
                    if np.any(valid_t_mask):
                        corr_val = float(np.corrcoef(scaled_s[valid_t_mask], unsup_val_df.loc[valid_t_mask, target_col].values)[0, 1])
                    else:
                        corr_val = np.nan

                    if_record = {
                        "fold": fold_idx,
                        "train_start": str(t_dates[0])[:10],
                        "train_end": str(t_dates[-1])[:10],
                        "unsup_train_rows": len(unsup_train_idx),
                        "val_start": str(v_dates[0])[:10],
                        "val_end": str(v_dates[-1])[:10],
                        "val_rows": len(unsup_val_df),
                        "fit_time_sec": if_fit_time,
                        "score_time_sec": if_score_time,
                        "mean_scaled_score": float(np.mean(scaled_s)),
                        "median_scaled_score": float(np.median(scaled_s)),
                        "std_scaled_score": float(np.std(scaled_s)),
                        "p90_scaled_score": float(np.percentile(scaled_s, 90)),
                        "p95_scaled_score": float(np.percentile(scaled_s, 95)),
                        "p99_scaled_score": float(np.percentile(scaled_s, 99)),
                        "flagged_count": flagged_count,
                        "flagged_rate": flagged_rate,
                        "hre_pos_count": hre_pos_count,
                        "hre_pos_flagged_count": tp_anomaly,
                        "hre_pos_flag_rate": hre_pos_flag_rate,
                        "hre_neg_flag_rate": hre_neg_flag_rate,
                        "anomaly_target_corr": corr_val,
                    }
                    if_fold_records.append(if_record)

                    print(
                        f"  [isolation_forest             ] Fit: {if_fit_time:5.1f}s | "
                        f"MeanScaled: {if_record['mean_scaled_score']:.1f} | Flagged: {flagged_count:,} ({flagged_rate*100:.2f}%) | "
                        f"HRE+ Flagged: {tp_anomaly:,}/{hre_pos_count:,} ({hre_pos_flag_rate*100:.1f}%) | Corr: {corr_val:+.4f}",
                        flush=True,
                    )

                    # Store IF prediction rows
                    if_pred_df = pd.DataFrame(
                        {
                            "fold": fold_idx,
                            "date": unsup_val_df["date"].values,
                            "ticker": unsup_val_df["ticker"].values,
                            "cap_segment": unsup_val_df["cap_segment"].values if "cap_segment" in unsup_val_df.columns else "unknown",
                            "y_true": unsup_val_df[target_col].values,
                            "isolation_score": raw_s,
                            "anomaly_score_scaled": scaled_s,
                            "isolation_flag": flags,
                        },
                        index=unsup_val_idx,
                    )
                    if_preds_list.append(if_pred_df)

            save_checkpoints()

        print(f"\n{'='*80}\nWALK-FORWARD VALIDATION COMPLETED ACROSS ALL 10 FOLDS\n{'='*80}")

        # Assemble summary tables and comparisons
        supervised_folds_df = pd.DataFrame(supervised_fold_records)
        if_folds_df = pd.DataFrame(if_fold_records) if if_fold_records else pd.DataFrame()

        # Combine predictions per model
        combined_preds: Dict[str, pd.DataFrame] = {}
        for m in models:
            combined_preds[m] = pd.concat(all_supervised_preds[m]).sort_values(["date", "ticker"])

        # Model comparison aggregation across 10 folds
        model_comparison_records = []
        for m in models:
            m_sub = supervised_folds_df[supervised_folds_df["model_name"] == m]
            m_pred = combined_preds[m]

            # Pooled out-of-fold metrics
            pooled_50 = ModelEvaluator.compute_classification_metrics(m_pred["y_true"], m_pred[f"prob_{m}"], threshold=0.50)

            # Cap-segment stratified metrics on pooled out-of-fold predictions
            cap_metrics = ModelEvaluator.evaluate_by_cap_segment(
                m_pred.rename(columns={"y_true": "hre_5d_train", f"prob_{m}": "hre_prob"}),
                segment_col="cap_segment",
                target_col="hre_5d_train",
                prob_col="hre_prob",
                threshold=0.50,
            )

            rec = {
                "model_name": m,
                "mean_pr_auc": float(m_sub["pr_auc"].mean()),
                "std_pr_auc": float(m_sub["pr_auc"].std()),
                "min_pr_auc": float(m_sub["pr_auc"].min()),
                "max_pr_auc": float(m_sub["pr_auc"].max()),
                "pooled_pr_auc": pooled_50["pr_auc"],
                "mean_roc_auc": float(m_sub["roc_auc"].mean()),
                "std_roc_auc": float(m_sub["roc_auc"].std()),
                "pooled_roc_auc": pooled_50["roc_auc"],
                "mean_brier_score": float(m_sub["brier_score"].mean()),
                "pooled_brier_score": pooled_50["brier_score"],
                "mean_log_loss": float(m_sub["log_loss"].mean()),
                "pooled_log_loss": pooled_50["log_loss"],
                "mean_f1_50": float(m_sub["f1_50"].mean()),
                "mean_precision_50": float(m_sub["precision_50"].mean()),
                "mean_recall_50": float(m_sub["recall_50"].mean()),
                "mean_f1_calib": float(m_sub["f1_calib"].mean()),
                "mean_precision_calib": float(m_sub["precision_calib"].mean()),
                "mean_recall_calib": float(m_sub["recall_calib"].mean()),
                "mean_calib_threshold": float(m_sub["calibrated_threshold"].mean()),
                "total_fit_time_sec": float(m_sub["fit_time_sec"].sum()),
                "mean_fit_time_sec": float(m_sub["fit_time_sec"].mean()),
                "large_cap_pr_auc": float(
                    cap_metrics.loc[cap_metrics["cap_segment"] == "large", "pr_auc"].values[0]
                    if "large" in cap_metrics["cap_segment"].values
                    else np.nan
                ),
                "mid_cap_pr_auc": float(
                    cap_metrics.loc[cap_metrics["cap_segment"] == "mid", "pr_auc"].values[0]
                    if "mid" in cap_metrics["cap_segment"].values
                    else np.nan
                ),
                "small_cap_pr_auc": float(
                    cap_metrics.loc[cap_metrics["cap_segment"] == "small", "pr_auc"].values[0]
                    if "small" in cap_metrics["cap_segment"].values
                    else np.nan
                ),
            }
            model_comparison_records.append(rec)

        model_comparison_df = pd.DataFrame(model_comparison_records).sort_values("mean_pr_auc", ascending=False)

        # Print model comparison table
        print(f"\n{'='*80}\nFINAL MODEL COMPARISON LEADERBOARD (RANKED BY MEAN PR-AUC ACROSS 10 FOLDS)\n{'='*80}")
        disp_cols = [
            "model_name",
            "mean_pr_auc",
            "std_pr_auc",
            "min_pr_auc",
            "pooled_pr_auc",
            "mean_roc_auc",
            "mean_brier_score",
            "mean_f1_50",
            "mean_f1_calib",
            "mean_calib_threshold",
            "total_fit_time_sec",
        ]
        print(model_comparison_df[disp_cols].to_string(index=False))

        # Save artifacts to output directory if specified
        if out_path:
            supervised_folds_df.to_csv(out_path / "supervised_fold_metrics.csv", index=False)
            model_comparison_df.to_csv(out_path / "supervised_model_comparison.csv", index=False)

            # Consolidate all model predictions into single table
            base_pred = combined_preds[models[0]][["fold", "date", "ticker", "cap_segment", "y_true"]].copy()
            for m in models:
                base_pred[f"prob_{m}"] = combined_preds[m][f"prob_{m}"]
                base_pred[f"pred_50_{m}"] = combined_preds[m][f"pred_50_{m}"]
                base_pred[f"pred_calib_{m}"] = combined_preds[m][f"pred_calib_{m}"]

            base_pred.to_parquet(out_path / "supervised_predictions.parquet")

            if eval_isolation_forest and not if_folds_df.empty:
                if_folds_df.to_csv(out_path / "isolation_forest_fold_metrics.csv", index=False)
                combined_if_preds = pd.concat(if_preds_list).sort_values(["date", "ticker"])
                combined_if_preds.to_parquet(out_path / "isolation_forest_predictions.parquet")

            print(f"\nSaved Batch 4 evaluation outputs to: {out_path.resolve()}")

        return {
            "supervised_fold_metrics": supervised_folds_df,
            "model_comparison": model_comparison_df,
            "supervised_predictions": combined_preds,
            "isolation_forest_fold_metrics": if_folds_df,
        }

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
            NotImplementedError: Deferred to Phase 8 Batch 5 final model selection.
        """
        raise NotImplementedError(
            "Final model training and serialization are deferred to Phase 8 Batch 5."
        )
