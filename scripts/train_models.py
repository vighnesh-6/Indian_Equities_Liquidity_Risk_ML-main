"""Offline model training and walk-forward validation script (Phase 8 Batch 4).

Responsible for:
- Loading labeled feature matrix from `data/processed/features_with_targets.parquet`.
- Enforcing strict 250-session holdout quarantine and 5-session boundary embargo.
- Executing 10-fold expanding-window walk-forward validation across 5 approved supervised configurations:
    1. logistic_regression_standard (StandardScaler, 20 features)
    2. logistic_regression_robust (RobustScaler, 20 features)
    3. random_forest_candidate_a (35 features, baseline excluding F51 relative_volume)
    4. random_forest_candidate_b (36 features, experimental including F51 relative_volume)
    5. hist_gradient_boosting (36 features, including F51 relative_volume)
- Executing fold-scoped unsupervised Isolation Forest across the same 10 folds (12 features).
- Enforcing inner chronological threshold calibration (60-day inner calib, 5-day inner purge).
- Persisting Batch 4 evaluation artifacts to `data/interim/phase8_batch4/`.
"""

import os
import sys
import time
from pathlib import Path

# Silence Loky physical cores warning on Windows
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

import pandas as pd

# Add repository root to python path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.modeling.trainer import ModelTrainer


def main() -> None:
    print("=" * 80)
    print("PHASE 8 BATCH 4: LEAKAGE-SAFE WALK-FORWARD MODELLING & EVALUATION")
    print("=" * 80)

    data_path = repo_root / "data" / "processed" / "features_with_targets.parquet"
    output_dir = repo_root / "data" / "interim" / "phase8_batch4"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading feature matrix from: {data_path}")
    t0 = time.perf_counter()
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns in {time.perf_counter() - t0:.2f}s")

    # Check splits
    split_counts = df["split"].value_counts().to_dict()
    print(f"Data splits: {split_counts}")

    # Initialize ModelTrainer with approved 10-fold configuration
    trainer = ModelTrainer(
        n_splits=10,
        min_train_days=490,
        test_window_days=60,
        purge_sessions=5,
    )

    models_to_evaluate = [
        "logistic_regression_standard",
        "logistic_regression_robust",
        "random_forest_candidate_a",
        "random_forest_candidate_b",
        "hist_gradient_boosting",
    ]

    print(f"\nStarting walk-forward validation across {len(models_to_evaluate)} supervised models + Isolation Forest...")
    results = trainer.train_and_evaluate_models(
        df=df,
        models=models_to_evaluate,
        target_col="hre_5d_train",
        output_dir=str(output_dir),
        eval_isolation_forest=True,
        calibrate_thresholds=True,
    )

    print("\n" + "=" * 80)
    print("BATCH 4 EXECUTION COMPLETE")
    print(f"All artifacts persisted to: {output_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()

