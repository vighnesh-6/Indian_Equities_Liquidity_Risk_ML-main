"""Model-specific preprocessing pipelines and feature configurations.

Phase 8 Batch 3 Architecture:
Prepares independent, model-specific preprocessing pipelines for:
1. Logistic Regression (Baseline): 20 features, SimpleImputer(strategy='median'),
   configurable StandardScaler (default) or RobustScaler.
2. Random Forest (Ensemble): Configurable Candidate A (35 features, baseline excluding F51)
   and Candidate B (36 features, experimental including F51 relative_volume), leveraging
   scikit-learn 1.8.0 native NaN handling (no imputer, no scaler).
3. HistGradientBoosting (Boosting): 36 features (including F51 relative_volume),
   native NaN handling, scale-invariant (no imputer, no scaler).
4. Isolation Forest (Unsupervised Anomaly Detector): 12 features from model_config.yaml,
   strictly target-independent, SimpleImputer(strategy='median'), scale-invariant (no scaler).

Strict Leakage Safeguards:
- Preprocessing objects are always instantiated fresh per model and per fold.
- Transformers are fit strictly on training observations; validation observations are only transformed.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest

# =============================================================================
# Approved Feature Lists (Phase 8 Batch 2 & Batch 3 Clarification Audits)
# =============================================================================

LOGISTIC_REGRESSION_FEATURES: List[str] = [
    "daily_return",
    "return_5d",
    "vol_5d",
    "downside_vol_20d",
    "max_drawdown_20d",
    "rsi_14",
    "log_turnover",
    "turnover_ratio",
    "volume_z20",
    "amihud_ma20",
    "low_volume_streak",
    "daily_range",
    "overnight_gap_abs",
    "gap_z20",
    "cs_spread_ma10",
    "rolling_beta_60d",
    "rolling_corr_60d",
    "nifty_vol_20d",
    "nifty_return_20d",
    "vix_close",
]

RANDOM_FOREST_CANDIDATE_A_FEATURES: List[str] = [
    "daily_return",
    "return_5d",
    "return_20d",
    "return_z20",
    "abs_return",
    "price_rel_ma20",
    "vol_5d",
    "vol_20d",
    "vol_ratio_5_20",
    "downside_vol_20d",
    "max_drawdown_20d",
    "rsi_14",
    "vol_of_vol",
    "log_volume",
    "volume_z20",
    "amihud_ratio",
    "amihud_ma20",
    "amihud_z20",
    "log_turnover",
    "turnover_ratio",
    "low_volume_streak",
    "daily_range",
    "daily_range_ma20",
    "range_z20",
    "overnight_gap",
    "overnight_gap_abs",
    "cs_spread_est",
    "cs_spread_ma10",
    "excess_return_1d",
    "rolling_beta_60d",
    "rolling_corr_60d",
    "nifty_vol_20d",
    "nifty_return_20d",
    "vol_beta",
    "vix_close",
]

# Candidate B includes F51 'relative_volume' (36 features)
RANDOM_FOREST_CANDIDATE_B_FEATURES: List[str] = (
    RANDOM_FOREST_CANDIDATE_A_FEATURES[:31]
    + ["relative_volume"]
    + RANDOM_FOREST_CANDIDATE_A_FEATURES[31:]
)

# HistGradientBoosting uses the identical 36 features including F51
HGBT_FEATURES: List[str] = list(RANDOM_FOREST_CANDIDATE_B_FEATURES)

# Isolation Forest features (12 features from configs/model_config.yaml)
ISOLATION_FOREST_FEATURES: List[str] = [
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


# =============================================================================
# Helper Functions
# =============================================================================

def get_model_features(model_name: str, candidate: str = "A") -> List[str]:
    """Returns the approved feature list for a given model identifier.

    Args:
        model_name: Model identifier ('logistic_regression', 'random_forest',
            'hist_gradient_boosting', 'isolation_forest').
        candidate: Candidate version for models with variants (e.g. 'A' or 'B' for RF).

    Returns:
        List[str]: List of feature column names.

    Raises:
        ValueError: If model_name or candidate is unrecognized.
    """
    key = model_name.strip().lower()
    if key in ("logistic_regression", "lr", "baseline"):
        return list(LOGISTIC_REGRESSION_FEATURES)
    elif key in ("random_forest", "rf"):
        cand = candidate.strip().upper()
        if cand == "A":
            return list(RANDOM_FOREST_CANDIDATE_A_FEATURES)
        elif cand == "B":
            return list(RANDOM_FOREST_CANDIDATE_B_FEATURES)
        else:
            raise ValueError(f"Invalid Random Forest candidate '{candidate}'. Supported: 'A', 'B'.")
    elif key in ("hist_gradient_boosting", "hgbt", "boosting"):
        return list(HGBT_FEATURES)
    elif key in ("isolation_forest", "if", "anomaly"):
        return list(ISOLATION_FOREST_FEATURES)
    else:
        raise ValueError(
            f"Unrecognized model_name '{model_name}'. Supported: "
            "'logistic_regression', 'random_forest', 'hist_gradient_boosting', 'isolation_forest'."
        )


def validate_features_present(df: pd.DataFrame, features: List[str]) -> None:
    """Validates that all expected features are present in the DataFrame.

    Args:
        df: DataFrame to validate.
        features: Expected feature column names.

    Raises:
        ValueError: If any expected feature is missing from df.columns.
    """
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required feature columns in DataFrame: {missing}"
        )


# =============================================================================
# Model-Specific Pipeline Builders
# =============================================================================

def build_logistic_regression_pipeline(
    scaler_type: str = "standard",
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    class_weight: str = "balanced",
    random_state: int = 42,
    **kwargs: Any,
) -> Pipeline:
    """Builds an independent Logistic Regression pipeline with median imputation and scaling.

    Args:
        scaler_type: Type of feature scaler ('standard' or 'robust').
        C: Inverse regularization strength.
        solver: Optimization algorithm ('lbfgs').
        max_iter: Maximum solver iterations.
        class_weight: Class weight scheme ('balanced').
        random_state: Random state for reproducibility.
        **kwargs: Additional keyword arguments forwarded to LogisticRegression.

    Returns:
        Pipeline: Fresh un-fitted sklearn Pipeline containing:
            1. 'imputer': SimpleImputer(strategy='median')
            2. 'scaler': StandardScaler() or RobustScaler()
            3. 'classifier': LogisticRegression(...)

    Raises:
        ValueError: If scaler_type is not 'standard' or 'robust'.
    """
    st = scaler_type.strip().lower()
    if st == "standard":
        scaler = StandardScaler()
    elif st == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(
            f"Invalid scaler_type '{scaler_type}'. Supported values are 'standard' and 'robust'."
        )

    imputer = SimpleImputer(strategy="median")
    classifier = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        **kwargs,
    )

    return Pipeline([
        ("imputer", imputer),
        ("scaler", scaler),
        ("classifier", classifier),
    ])


def build_random_forest_pipeline(
    candidate: str = "A",
    n_estimators: int = 150,
    max_depth: Optional[int] = 8,
    min_samples_leaf: int = 50,
    max_features: Any = "sqrt",
    class_weight: str = "balanced_subsample",
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs: Any,
) -> Pipeline:
    """Builds an independent Random Forest pipeline leveraging native NaN handling.

    Candidate Configurations:
    - Candidate A (Baseline): 35 features, excludes F51 relative_volume.
    - Candidate B (Experimental): 36 features, includes F51 relative_volume.

    Missing-value handling:
    scikit-learn 1.8.0 native NaN handling routes missing values during tree splitting.
    No imputer or scaler is applied.

    Args:
        candidate: 'A' (35 features, baseline) or 'B' (36 features, including F51).
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        min_samples_leaf: Minimum samples per leaf.
        max_features: Feature subset size per split.
        class_weight: Class weight adjustment.
        random_state: Random state.
        n_jobs: Number of parallel jobs.
        **kwargs: Additional keyword arguments forwarded to RandomForestClassifier.

    Returns:
        Pipeline: Fresh un-fitted sklearn Pipeline containing:
            1. 'classifier': RandomForestClassifier(...)

    Raises:
        ValueError: If candidate is not 'A' or 'B'.
    """
    cand = candidate.strip().upper()
    if cand not in ("A", "B"):
        raise ValueError(
            f"Invalid Random Forest candidate '{candidate}'. Supported values are 'A' and 'B'."
        )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs,
    )

    return Pipeline([
        ("classifier", classifier),
    ])


def build_hgbt_pipeline(
    max_iter: int = 200,
    max_leaf_nodes: int = 31,
    learning_rate: float = 0.05,
    l2_regularization: float = 0.1,
    class_weight: str = "balanced",
    random_state: int = 42,
    **kwargs: Any,
) -> Pipeline:
    """Builds an independent HistGradientBoosting pipeline leveraging native NaN handling.

    Input: 36 candidate features (including F51 relative_volume).
    Missing-value handling: HistGradientBoostingClassifier natively routes missing values
    to whichever child node yields the highest split gain.
    Scale-invariant: No scaler is applied.

    Args:
        max_iter: Maximum boosting iterations.
        max_leaf_nodes: Maximum leaves per tree.
        learning_rate: Shrinkage rate.
        l2_regularization: L2 penalty parameter.
        class_weight: Class weight adjustment.
        random_state: Random state.
        **kwargs: Additional keyword arguments forwarded to HistGradientBoostingClassifier.

    Returns:
        Pipeline: Fresh un-fitted sklearn Pipeline containing:
            1. 'classifier': HistGradientBoostingClassifier(...)
    """
    classifier = HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        class_weight=class_weight,
        random_state=random_state,
        **kwargs,
    )

    return Pipeline([
        ("classifier", classifier),
    ])


def build_isolation_forest_pipeline(
    n_estimators: int = 100,
    contamination: float = 0.05,
    max_samples: Any = "auto",
    random_state: int = 42,
    **kwargs: Any,
) -> Pipeline:
    """Builds an independent Isolation Forest pipeline for unsupervised anomaly detection.

    Input: 12 candidate features from configs/model_config.yaml.
    Target Independence: Isolation Forest strictly never receives supervised target labels.
    Missing-value handling: SimpleImputer(strategy='median') guarantees clean finite
    min/max bounds for random orthogonal hyperplane partitioning.
    Scale-invariant: Random partitioning is strictly invariant to monotonic scaling; no scaler is applied.

    Args:
        n_estimators: Number of isolation trees.
        contamination: Expected proportion of outliers in the data.
        max_samples: Number of samples drawn from X to train each tree.
        random_state: Random state.
        **kwargs: Additional keyword arguments forwarded to IsolationForest.

    Returns:
        Pipeline: Fresh un-fitted sklearn Pipeline containing:
            1. 'imputer': SimpleImputer(strategy='median')
            2. 'model': IsolationForest(...)
    """
    imputer = SimpleImputer(strategy="median")
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        **kwargs,
    )

    return Pipeline([
        ("imputer", imputer),
        ("model", model),
    ])
