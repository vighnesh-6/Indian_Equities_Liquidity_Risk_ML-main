"""
Category 4 — Price-Gap & Price-Impact Proxies (F37 to F45)
==========================================================

Audit Status (Phase 5 Contract §13): APPROVED & IMPLEMENTED
Features:
    F37: daily_range        — (High - Low) / Close
    F38: daily_range_ma20   — 20-day rolling mean of daily_range
    F39: range_z20          — 20-day rolling z-score of daily_range
    F40: overnight_gap      — (Open_t - Close_{t-1}) / Close_{t-1} (masked on corp gaps)
    F41: overnight_gap_abs  — |overnight_gap| (masked on corp gaps)
    F42: gap_z20            — 20-day rolling z-score of overnight_gap
    F43: cs_spread_est      — Corwin-Schultz (2012) spread floored at 0.0 (2-day pair mask)
    F44: cs_spread_ma10     — 10-day rolling mean of cs_spread_est
    F45: intraday_return    — (Close - Open) / Open
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features._common import (
    get_corp_action_mask,
    rolling_zscore,
    safe_divide,
)

_CS_CONSTANT: float = float(3.0 - 2.0 * np.sqrt(2.0))


def _compute_corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
    corp_gap: pd.Series,
) -> pd.Series:
    """Computes Corwin-Schultz (2012) 2-day effective spread floored at 0.0."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    valid_today = (high > 0.0) & (low > 0.0) & (high >= low) & np.isfinite(high) & np.isfinite(low)
    valid_pair = (
        valid_today
        & (prev_high > 0.0)
        & (prev_low > 0.0)
        & (prev_high >= prev_low)
        & np.isfinite(prev_high)
        & np.isfinite(prev_low)
    )

    log_hl = pd.Series(np.nan, index=high.index, dtype="float64")
    log_hl.loc[valid_today] = np.log(high.loc[valid_today] / low.loc[valid_today])
    beta = (log_hl ** 2) + (log_hl.shift(1) ** 2)

    two_day_high = pd.concat([high, prev_high], axis=1).max(axis=1, skipna=False)
    two_day_low = pd.concat([low, prev_low], axis=1).min(axis=1, skipna=False)
    gamma = pd.Series(np.nan, index=high.index, dtype="float64")
    gamma.loc[valid_pair] = np.log(two_day_high.loc[valid_pair] / two_day_low.loc[valid_pair]) ** 2

    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_CONSTANT - np.sqrt(gamma / _CS_CONSTANT)
    # 2*(exp(alpha) - 1)/(1 + exp(alpha)) == 2*tanh(alpha/2) without overflow risk
    spread = np.maximum(0.0, 2.0 * np.tanh(alpha / 2.0))

    pair_masked = corp_gap | corp_gap.shift(1, fill_value=False)
    return spread.mask(~valid_pair | pair_masked, np.nan)


def compute_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Category 4 features F37 to F45 for a single stock.

    Args:
        df: Chronologically sorted single-stock OHLCV DataFrame containing
            ``open``, ``high``, ``low``, ``close`` (and optionally
            ``is_corporate_action_gap``).

    Returns:
        DataFrame indexed identically to ``df`` containing float64 columns
        ``daily_range`` through ``intraday_return`` (F37-F45).
    """
    open_ = df["open"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    close = df["close"].astype("float64")
    corp_gap = get_corp_action_mask(df)

    daily_range = safe_divide(high - low, close)
    daily_range_ma20 = daily_range.rolling(window=20, min_periods=20).mean()
    range_z20 = rolling_zscore(daily_range, 20)

    prev_close = close.shift(1)
    overnight_gap = safe_divide(open_ - prev_close, prev_close).mask(corp_gap, np.nan)
    overnight_gap_abs = overnight_gap.abs()
    gap_z20 = rolling_zscore(overnight_gap, 20)

    cs_spread_est = _compute_corwin_schultz_spread(high, low, corp_gap)
    cs_spread_ma10 = cs_spread_est.rolling(window=10, min_periods=10).mean()
    intraday_return = safe_divide(close - open_, open_)

    out = pd.DataFrame(
        {
            "daily_range": daily_range,
            "daily_range_ma20": daily_range_ma20,
            "range_z20": range_z20,
            "overnight_gap": overnight_gap,
            "overnight_gap_abs": overnight_gap_abs,
            "gap_z20": gap_z20,
            "cs_spread_est": cs_spread_est,
            "cs_spread_ma10": cs_spread_ma10,
            "intraday_return": intraday_return,
        },
        index=df.index,
        dtype="float64",
    )
    return out
