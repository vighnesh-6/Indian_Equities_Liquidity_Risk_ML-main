"""Machine learning package for walk-forward training, scoring, anomaly detection, and evaluation."""

from src.modeling.trainer import ModelTrainer
from src.modeling.scorer import ModelScorer
from src.modeling.anomaly import AnomalyDetector
from src.modeling.evaluation import ModelEvaluator
from src.modeling.explainability import ModelExplainer
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

__all__ = [
    "ModelTrainer",
    "ModelScorer",
    "AnomalyDetector",
    "ModelEvaluator",
    "ModelExplainer",
    "LOGISTIC_REGRESSION_FEATURES",
    "RANDOM_FOREST_CANDIDATE_A_FEATURES",
    "RANDOM_FOREST_CANDIDATE_B_FEATURES",
    "HGBT_FEATURES",
    "ISOLATION_FOREST_FEATURES",
    "build_logistic_regression_pipeline",
    "build_random_forest_pipeline",
    "build_hgbt_pipeline",
    "build_isolation_forest_pipeline",
    "get_model_features",
    "validate_features_present",
]
