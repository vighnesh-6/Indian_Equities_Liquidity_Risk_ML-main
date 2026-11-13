"""Composite Liquidity-Risk Score (LRS) module.

Phase 7 Mathematical Specification:
Computes the rule-based, formula-driven composite score (0–100) per stock-day
using daily cross-sectional percentile ranking independently within each static
market-capitalization tier (Large, Mid, Small).

Approved 4-Pillar Formulation (D7.1 & D7.2):
    LRS(i, t, s) = 0.35 * R_{t, s}(amihud_ma20)
                 + 0.25 * R_{t, s}(cs_spread_ma10)
                 + 0.20 * R_inv_{t, s}(volume_z20)
                 + 0.20 * R_{t, s}(downside_vol_20d)

where:
    - R_{t, s}(X): Daily cross-sectional percentile rank (ascending, higher = higher risk).
    - R_inv_{t, s}(volume_z20): Daily cross-sectional percentile rank (descending, drying up volume = higher risk).
    - Weights sum exactly to 1.00.
    - LRS is strictly bounded within [0.0, 100.0] wherever calculable, or NaN if inputs are missing.

Approved Frozen 90th-Percentile Calibration Thresholds (D7.4):
    - Large Cap: 73.8298
    - Mid Cap:   72.3251
    - Small Cap: 72.4359
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


# Approved Phase 7 component weights
DEFAULT_LRS_WEIGHTS: Dict[str, float] = {
    "amihud_ma20": 0.35,
    "cs_spread_ma10": 0.25,
    "volume_z20": 0.20,
    "downside_vol_20d": 0.20,
}

# Approved frozen 90th-percentile calibration thresholds (Training period <= 2025-10-17)
FROZEN_LRS_THRESHOLDS: Dict[str, float] = {
    "large": 73.8298,
    "mid": 72.3251,
    "small": 72.4359,
}

# Required input feature columns (F28, F44, F26, F18)
REQUIRED_LRS_FEATURES: List[str] = [
    "amihud_ma20",
    "cs_spread_ma10",
    "volume_z20",
    "downside_vol_20d",
]

# Required grouping and identifier columns
REQUIRED_METADATA_COLUMNS: List[str] = [
    "date",
    "ticker",
    "cap_segment",
]


def compute_lrs(
    features_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    calibration_quantiles: Optional[Dict[str, Any]] = None,
) -> pd.Series:
    """Computes the rule-based Liquidity Risk Score (0-100) per stock-day.

    Performs daily cross-sectional percentile ranking independently within
    each static market-cap segment (Large, Mid, Small).

    Args:
        features_df: DataFrame containing date, ticker, cap_segment,
            and the 4 required feature columns:
            `amihud_ma20`, `cs_spread_ma10`, `volume_z20`, `downside_vol_20d`.
        weights: Optional dictionary of component weights overriding DEFAULT_LRS_WEIGHTS.
        calibration_quantiles: Optional precomputed calibration distribution (preserved
            for API compatibility).

    Returns:
        pd.Series: Continuous LRS values bounded [0.0, 100.0], with NaN for
            uncalculable or missing observations, indexed identically to features_df.

    Raises:
        ValueError: If required columns are missing or weights do not sum to 1.0.
    """
    w = weights or DEFAULT_LRS_WEIGHTS

    # Validate weight sum
    weight_sum = sum(w.values())
    if not np.isclose(weight_sum, 1.0, atol=1e-5):
        raise ValueError(f"LRS component weights must sum to 1.0, got {weight_sum:.6f}")

    # Validate required metadata columns
    missing_meta = [c for c in REQUIRED_METADATA_COLUMNS if c not in features_df.columns]
    if missing_meta:
        raise ValueError(f"features_df is missing required metadata columns: {missing_meta}")

    # Validate required feature columns
    missing_features = [c for c in w.keys() if c not in features_df.columns]
    if missing_features:
        raise ValueError(f"features_df is missing required feature columns: {missing_features}")

    # Prepare working DataFrame with normalized grouping keys and features
    cap_seg = features_df["cap_segment"].astype(str).str.lower()
    df_work = pd.DataFrame({
        "date": features_df["date"],
        "cap_segment": cap_seg,
        "amihud_ma20": features_df["amihud_ma20"],
        "cs_spread_ma10": features_df["cs_spread_ma10"],
        "volume_z20": features_df["volume_z20"],
        "downside_vol_20d": features_df["downside_vol_20d"],
    }, index=features_df.index)

    # Daily cross-sectional percentile ranking within (date, cap_segment)
    # amihud_ma20: higher illiquidity = higher risk (ascending)
    r_amihud = df_work.groupby(["date", "cap_segment"])["amihud_ma20"].rank(
        pct=True, ascending=True
    ) * 100.0

    # cs_spread_ma10: higher bid-ask spread = higher risk (ascending)
    r_cs = df_work.groupby(["date", "cap_segment"])["cs_spread_ma10"].rank(
        pct=True, ascending=True
    ) * 100.0

    # volume_z20: drying up volume = higher risk (descending / inverse ranking)
    r_vol_inv = df_work.groupby(["date", "cap_segment"])["volume_z20"].rank(
        pct=True, ascending=False
    ) * 100.0

    # downside_vol_20d: higher downside volatility = higher risk (ascending)
    r_downside = df_work.groupby(["date", "cap_segment"])["downside_vol_20d"].rank(
        pct=True, ascending=True
    ) * 100.0

    # Compute weighted composite score
    lrs = (
        w["amihud_ma20"] * r_amihud +
        w["cs_spread_ma10"] * r_cs +
        w["volume_z20"] * r_vol_inv +
        w["downside_vol_20d"] * r_downside
    )

    # Ensure strict numeric boundaries [0.0, 100.0] wherever calculable
    lrs = lrs.clip(lower=0.0, upper=100.0)
    lrs.name = "lrs"

    return lrs


def calibrate_lrs_thresholds(
    features_df: pd.DataFrame,
    lrs_series: Optional[pd.Series] = None,
    train_cutoff: str = "2025-10-17",
    quantile: float = 0.90,
) -> Dict[str, float]:
    """Calculates within-tier percentile thresholds strictly from training observations.

    Args:
        features_df: DataFrame with date, cap_segment, and features.
        lrs_series: Precomputed LRS series; if None, compute_lrs(features_df) is run.
        train_cutoff: Latest date included in training calibration (inclusive).
        quantile: Percentile quantile to calculate (default: 0.90 for 90th percentile).

    Returns:
        Dict[str, float]: Segment thresholds mapping segment name ('large', 'mid', 'small')
            to calibrated threshold value.
    """
    if lrs_series is None:
        lrs = compute_lrs(features_df)
    else:
        lrs = lrs_series

    dates = pd.to_datetime(features_df["date"])
    cutoff = pd.to_datetime(train_cutoff)
    train_mask = dates <= cutoff

    seg_series = features_df["cap_segment"].astype(str).str.lower()
    thresholds: Dict[str, float] = {}

    for seg in ["large", "mid", "small"]:
        seg_mask = train_mask & (seg_series == seg) & lrs.notna()
        seg_vals = lrs[seg_mask]
        if len(seg_vals) == 0:
            raise ValueError(f"No valid training observations found for segment: {seg}")
        thresholds[seg] = float(np.percentile(seg_vals, quantile * 100.0))

    return thresholds
