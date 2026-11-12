"""
Category 2 — Volatility & Downside Risk (F11 to F22)
====================================================

Audit Status (Phase 5 Contract §13): APPROVED & IMPLEMENTED
Features:
    F11: vol_5d            — 5-day realized log-return vol, annualized * sqrt(252) (C3)
    F12: vol_20d           — 20-day realized log-return vol, annualized * sqrt(252)
    F13: vol_60d           — 60-day realized log-return vol, annualized * sqrt(252)
    F14: vol_ratio_5_20    — Dimensionless vol_5d / vol_20d (C3)
    F15: parkinson_vol     — 20-day Parkinson high-low volatility, annualized
    F16: true_range        — 1-day True Range (masked on corp gaps) (C6)
    F17: atr_20d           — 20-day Average True Range (full 20-obs window) (C6)
    F18: downside_vol_20d  — 20-day target-0 semi-deviation, annualized (C4)
    F19: sortino_component — return_20d / downside_vol_20d (C4)
    F20: max_drawdown_20d  — Trailing [t-19, t] max drawdown (C6, C10)
    F21: rsi_14            — 14-day Wilder RSI with explicit edge cases (C4, C6)
    F22: vol_of_vol        — 60-day rolling std of vol_20d
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features._common import (
    ANNUALIZATION_FACTOR,
    compound_returns,
    compute_masked_daily_return,
    compute_masked_log_return,
    get_corp_action_mask,
    safe_divide,
    transition_window_mask,
)

_PARKINSON_DENOM: float = float(4.0 * np.log(2.0))


def _compute_parkinson_vol(high: pd.Series, low: pd.Series) -> pd.Series:
    """Computes annualized 20-day Parkinson high-low range volatility."""
    valid = (high > 0.0) & (low > 0.0) & (high >= low) & np.isfinite(high) & np.isfinite(low)
    log_hl_sq = pd.Series(np.nan, index=high.index, dtype="float64")
    log_hl_sq.loc[valid] = np.log(high.loc[valid] / low.loc[valid]) ** 2
    mean_sq = log_hl_sq.rolling(window=20, min_periods=20).mean()
    return np.sqrt(mean_sq / _PARKINSON_DENOM) * ANNUALIZATION_FACTOR


def _compute_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    corp_gap: pd.Series,
) -> pd.Series:
    """Computes 1-day True Range, NaN at t=0 and on corporate-action gap rows."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)
    return tr.mask(corp_gap, np.nan)


def _compute_max_drawdown_20d(close: pd.Series, corp_gap: pd.Series) -> pd.Series:
    """Computes strictly [t-19, t] max drawdown, masked across corp-action gaps."""
    rolling_peak = close.rolling(window=20, min_periods=20).max()
    dd = safe_divide(close - rolling_peak, rolling_peak)
    affected = transition_window_mask(corp_gap, 20)
    return dd.mask(affected, np.nan)


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    """Evaluates Wilder RSI with explicit C4 zero-gain/zero-loss edge handling."""
    if avg_loss == 0.0 and avg_gain > 0.0:
        return 100.0
    if avg_gain == 0.0 and avg_loss > 0.0:
        return 0.0
    if avg_gain == 0.0 and avg_loss == 0.0:
        return 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_wilder_rsi14(close: pd.Series, corp_gap: pd.Series) -> pd.Series:
    """Computes 14-day Wilder RSI (alpha=1/14), masking corp-gap price changes."""
    delta = close.diff().mask(corp_gap, np.nan).to_numpy(dtype="float64")
    n = len(delta)
    rsi = np.full(n, np.nan, dtype="float64")

    init_gains: list[float] = []
    init_losses: list[float] = []
    initialized = False
    avg_gain = 0.0
    avg_loss = 0.0

    for i in range(n):
        d = delta[i]
        if not np.isfinite(d):
            # Preserve prior EWM state across corporate-action discontinuities
            continue

        gain = d if d > 0.0 else 0.0
        loss = -d if d < 0.0 else 0.0

        if not initialized:
            init_gains.append(gain)
            init_losses.append(loss)
            if len(init_gains) == 14:
                avg_gain = float(sum(init_gains) / 14.0)
                avg_loss = float(sum(init_losses) / 14.0)
                initialized = True
                rsi[i] = _rsi_from_averages(avg_gain, avg_loss)
        else:
            avg_gain = (13.0 * avg_gain + gain) / 14.0
            avg_loss = (13.0 * avg_loss + loss) / 14.0
            rsi[i] = _rsi_from_averages(avg_gain, avg_loss)

    return pd.Series(rsi, index=close.index, dtype="float64")


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Category 2 features F11 to F22 for a single stock.

    Args:
        df: Chronologically sorted single-stock OHLCV DataFrame containing
            ``high``, ``low``, ``close`` (and optionally ``is_corporate_action_gap``).

    Returns:
        DataFrame indexed identically to ``df`` containing float64 columns
        ``vol_5d`` through ``vol_of_vol`` (F11-F22).
    """
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    close = df["close"].astype("float64")
    corp_gap = get_corp_action_mask(df)

    daily_return = compute_masked_daily_return(df)
    log_return = compute_masked_log_return(df)
    return_20d = compound_returns(daily_return, 20)

    vol_5d = log_return.rolling(window=5, min_periods=5).std(ddof=1) * ANNUALIZATION_FACTOR
    vol_20d = log_return.rolling(window=20, min_periods=20).std(ddof=1) * ANNUALIZATION_FACTOR
    vol_60d = log_return.rolling(window=60, min_periods=60).std(ddof=1) * ANNUALIZATION_FACTOR

    true_range = _compute_true_range(high, low, close, corp_gap)
    downside_sq = log_return.clip(upper=0.0) ** 2
    downside_vol_20d = (
        np.sqrt(downside_sq.rolling(window=20, min_periods=20).mean())
        * ANNUALIZATION_FACTOR
    )

    out = pd.DataFrame(
        {
            "vol_5d": vol_5d,
            "vol_20d": vol_20d,
            "vol_60d": vol_60d,
            "vol_ratio_5_20": safe_divide(vol_5d, vol_20d),
            "parkinson_vol": _compute_parkinson_vol(high, low),
            "true_range": true_range,
            "atr_20d": true_range.rolling(window=20, min_periods=20).mean(),
            "downside_vol_20d": downside_vol_20d,
            "sortino_component": safe_divide(return_20d, downside_vol_20d),
            "max_drawdown_20d": _compute_max_drawdown_20d(close, corp_gap),
            "rsi_14": _compute_wilder_rsi14(close, corp_gap),
            "vol_of_vol": vol_20d.rolling(window=60, min_periods=60).std(ddof=1),
        },
        index=df.index,
        dtype="float64",
    )
    return out
