"""Shared numerical and masking helpers for Phase 5 feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR: float = 252.0
ANNUALIZATION_FACTOR: float = float(np.sqrt(TRADING_DAYS_PER_YEAR))


def safe_divide(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
    *,
    allow_negative_denom: bool = False,
) -> pd.Series:
    """Divides two series in float64, returning NaN on zero/invalid denominators.

    Args:
        numerator: Dividend series or array.
        denominator: Divisor series or array.
        allow_negative_denom: When False (default), denominators <= 0 yield NaN.
            When True, only denominators == 0 or non-finite yield NaN.

    Returns:
        Float64 Series with NaN wherever division is undefined or non-finite.
    """
    num_s = pd.Series(numerator, dtype="float64")
    den_s = pd.Series(denominator, index=num_s.index, dtype="float64")

    if allow_negative_denom:
        valid = np.isfinite(num_s) & np.isfinite(den_s) & (den_s != 0.0)
    else:
        valid = np.isfinite(num_s) & np.isfinite(den_s) & (den_s > 0.0)

    out = pd.Series(np.nan, index=num_s.index, dtype="float64")
    out.loc[valid] = num_s.loc[valid] / den_s.loc[valid]
    return out


def rolling_zscore(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Computes rolling sample z-score with strict zero-std protection.

    Args:
        series: Input float64 Series.
        window: Rolling lookback window length.
        min_periods: Minimum non-NaN observations required (defaults to ``window``).

    Returns:
        Float64 Series with NaN when valid count < min_periods or rolling std <= 0.
    """
    mp = window if min_periods is None else min_periods
    s = series.astype("float64")
    mean = s.rolling(window=window, min_periods=mp).mean()
    std = s.rolling(window=window, min_periods=mp).std(ddof=1)
    # Guard IEEE-754 floating-point ULP noise on near-constant series (e.g. 1.01**t)
    ulp_floor = np.maximum(1e-15, 1e-12 * mean.abs())
    std_clean = std.where(std > ulp_floor, np.nan)
    return safe_divide(s - mean, std_clean)


def get_corp_action_mask(df: pd.DataFrame) -> pd.Series:
    """Extracts boolean corporate-action gap mask aligned to ``df.index``."""
    if "is_corporate_action_gap" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df["is_corporate_action_gap"].fillna(False).astype(bool)


def transition_window_mask(corp_gap_mask: pd.Series, window: int) -> pd.Series:
    """Identifies windows spanning an interior corporate-action price transition.

    Per Phase 5 Contract §13.2(3), for a W-observation price window [t-W+1, t],
    a transition from t-1 to t affects the window whenever any bar in
    [t-W+2, t] (the last W-1 bars) has ``is_corporate_action_gap == True``.
    """
    if window <= 1:
        return corp_gap_mask.astype(bool)
    return (
        corp_gap_mask.astype("int64")
        .rolling(window=window - 1, min_periods=1)
        .sum()
        .gt(0)
    )


def compute_masked_daily_return(df: pd.DataFrame) -> pd.Series:
    """Computes C_t / C_{t-1} - 1 masked on corporate-action gap rows."""
    close = df["close"].astype("float64")
    prev_close = close.shift(1)
    ret = safe_divide(close - prev_close, prev_close)
    corp_gap = get_corp_action_mask(df)
    return ret.mask(corp_gap, np.nan)


def compute_masked_log_return(df: pd.DataFrame) -> pd.Series:
    """Computes ln(C_t / C_{t-1}) masked on corporate-action gap rows."""
    close = df["close"].astype("float64")
    prev_close = close.shift(1)
    valid = (close > 0.0) & (prev_close > 0.0) & np.isfinite(close) & np.isfinite(prev_close)
    log_ret = pd.Series(np.nan, index=df.index, dtype="float64")
    log_ret.loc[valid] = np.log(close.loc[valid] / prev_close.loc[valid])
    corp_gap = get_corp_action_mask(df)
    return log_ret.mask(corp_gap, np.nan)


def compound_returns(daily_return: pd.Series, window: int) -> pd.Series:
    """Compounds ``window`` consecutive masked daily returns: prod(1 + r) - 1."""
    gross = 1.0 + daily_return.astype("float64")
    return gross.rolling(window=window, min_periods=window).apply(np.prod, raw=True) - 1.0
