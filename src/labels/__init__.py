"""Labels and Risk-Score computation package (LRS and HRE_5d target)."""

from src.labels.lrs import (
    DEFAULT_LRS_WEIGHTS,
    FROZEN_LRS_THRESHOLDS,
    calibrate_lrs_thresholds,
    compute_lrs,
)
from src.labels.targets import (
    apply_training_embargo_split,
    build_features_with_targets,
    compute_forward_max_lrs,
    construct_hre_target,
)

__all__ = [
    "compute_lrs",
    "calibrate_lrs_thresholds",
    "DEFAULT_LRS_WEIGHTS",
    "FROZEN_LRS_THRESHOLDS",
    "compute_forward_max_lrs",
    "construct_hre_target",
    "apply_training_embargo_split",
    "build_features_with_targets",
]
