"""
Category 3 — Liquidity & Trading-Activity Proxies (F23 to F36)
==============================================================

Audit Status (Phase 5 Contract §13): APPROVED & IMPLEMENTED
Features:
    F23: log_volume         — log(Volume + 1)
    F24: volume_ma20        — 20-day rolling mean share volume
    F25: volume_ratio       — Volume / volume_ma20
    F26: volume_z20         — 20-day rolling z-score of volume
    F27: amihud_ratio       — |r_t| / (Volume * Close) * 1e6 (NaN on Vol==0 or corp gaps)
    F28: amihud_ma20        — 20-day Amihud MA with min_periods=15 (C7)
    F29: amihud_z20         — 20-day Amihud z-score with min_periods=15
    F30: turnover_value     — Volume * Close (INR)
    F31: log_turnover       — log(turnover_value + 1)
    F32: turnover_ma20      — 20-day rolling mean rupee turnover
    F33: turnover_ratio     — turnover_value / turnover_ma20
    F34: zero_return_flag   — 1.0 if |daily_return| < 1e-4 else 0.0 (NaN if F01 is NaN)
    F35: zero_vol_flag      — 1.0 if Volume == 0 else 0.0
    F36: low_volume_streak  — 5-day rolling count of volume_ratio < 0.3 (NaN-aware)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features._common import (
    compute_masked_daily_return,
    rolling_zscore,
    safe_divide,
)

AMIHUD_SCALING_FACTOR: float = 1_000_000.0
AMIHUD_MIN_PERIODS: int = 15
ZERO_RETURN_EPSILON: float = 1e-4
LOW_VOLUME_RATIO_THRESHOLD: float = 0.3


def compute_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Category 3 features F23 to F36 for a single stock.

    Args:
        df: Chronologically sorted single-stock OHLCV DataFrame containing
            ``close`` and ``volume`` (and optionally ``is_corporate_action_gap``).

    Returns:
        DataFrame indexed identically to ``df`` containing float64 columns
        ``log_volume`` through ``low_volume_streak`` (F23-F36).
    """
    close = df["close"].astype("float64")
    volume = df["volume"].astype("float64")
    daily_return = compute_masked_daily_return(df)

    log_volume = pd.Series(np.nan, index=df.index, dtype="float64")
    valid_vol = np.isfinite(volume) & (volume >= 0.0)
    log_volume.loc[valid_vol] = np.log1p(volume.loc[valid_vol])

    volume_ma20 = volume.rolling(window=20, min_periods=20).mean()
    volume_ratio = safe_divide(volume, volume_ma20)
    volume_z20 = rolling_zscore(volume, 20)

    turnover_value = volume * close
    log_turnover = pd.Series(np.nan, index=df.index, dtype="float64")
    valid_turn = np.isfinite(turnover_value) & (turnover_value >= 0.0)
    log_turnover.loc[valid_turn] = np.log1p(turnover_value.loc[valid_turn])

    turnover_ma20 = turnover_value.rolling(window=20, min_periods=20).mean()
    turnover_ratio = safe_divide(turnover_value, turnover_ma20)

    # F27: safe_divide returns NaN whenever volume == 0 (turnover_value == 0) or daily_return is NaN
    amihud_ratio = safe_divide(daily_return.abs() * AMIHUD_SCALING_FACTOR, turnover_value)
    amihud_ma20 = amihud_ratio.rolling(window=20, min_periods=AMIHUD_MIN_PERIODS).mean()
    amihud_z20 = rolling_zscore(amihud_ratio, 20, min_periods=AMIHUD_MIN_PERIODS)

    zero_return_flag = pd.Series(
        np.where(
            daily_return.isna(),
            np.nan,
            (daily_return.abs() < ZERO_RETURN_EPSILON).astype("float64"),
        ),
        index=df.index,
        dtype="float64",
    )
    zero_vol_flag = pd.Series(
        np.where(volume.isna(), np.nan, (volume == 0.0).astype("float64")),
        index=df.index,
        dtype="float64",
    )

    is_low_vol = pd.Series(
        np.where(
            volume_ratio.isna(),
            np.nan,
            (volume_ratio < LOW_VOLUME_RATIO_THRESHOLD).astype("float64"),
        ),
        index=df.index,
        dtype="float64",
    )
    low_volume_streak = is_low_vol.rolling(window=5, min_periods=5).sum()

    out = pd.DataFrame(
        {
            "log_volume": log_volume,
            "volume_ma20": volume_ma20,
            "volume_ratio": volume_ratio,
            "volume_z20": volume_z20,
            "amihud_ratio": amihud_ratio,
            "amihud_ma20": amihud_ma20,
            "amihud_z20": amihud_z20,
            "turnover_value": turnover_value,
            "log_turnover": log_turnover,
            "turnover_ma20": turnover_ma20,
            "turnover_ratio": turnover_ratio,
            "zero_return_flag": zero_return_flag,
            "zero_vol_flag": zero_vol_flag,
            "low_volume_streak": low_volume_streak,
        },
        index=df.index,
        dtype="float64",
    )
    return out
