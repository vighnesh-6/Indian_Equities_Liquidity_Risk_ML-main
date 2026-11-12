"""
Category 1 — Returns & Price Behaviour (F01 to F10)
====================================================

Audit Status (Phase 5 Contract §13): APPROVED & IMPLEMENTED
Features:
    F01: daily_return      — 1-day percentage close return (masked on corp gaps)
    F02: log_return        — 1-day log close return (masked on corp gaps)
    F03: return_5d         — 5-day compounded return (from masked F01)
    F04: return_20d        — 20-day compounded return (from masked F01)
    F05: return_60d        — 60-day compounded return (from masked F01)
    F06: return_z5         — 5-day rolling z-score of daily return
    F07: return_z20        — 20-day rolling z-score of daily return
    F08: abs_return        — Absolute 1-day return (masked on corp gaps)
    F09: price_rel_ma20    — Close / SMA(20) - 1 (masked on corp-gap transition)
    F10: price_rel_ma60    — Close / SMA(60) - 1 (masked on corp-gap transition)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features._common import (
    compound_returns,
    compute_masked_daily_return,
    compute_masked_log_return,
    get_corp_action_mask,
    rolling_zscore,
    safe_divide,
    transition_window_mask,
)


def _price_relative_to_sma(
    close: pd.Series,
    corp_gap_mask: pd.Series,
    window: int,
) -> pd.Series:
    """Computes Close / SMA(window) - 1, masking windows across corp-action gaps."""
    sma = close.rolling(window=window, min_periods=window).mean()
    rel = safe_divide(close, sma) - 1.0
    affected = transition_window_mask(corp_gap_mask, window)
    return rel.mask(affected, np.nan)


def compute_returns_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Category 1 features F01 to F10 for a single stock.

    Args:
        df: Chronologically sorted single-stock OHLCV DataFrame containing at
            minimum ``close`` (and optionally ``is_corporate_action_gap``).

    Returns:
        DataFrame indexed identically to ``df`` containing float64 columns
        ``daily_return`` through ``price_rel_ma60`` (F01-F10).
    """
    close = df["close"].astype("float64")
    corp_gap = get_corp_action_mask(df)

    daily_return = compute_masked_daily_return(df)
    log_return = compute_masked_log_return(df)

    out = pd.DataFrame(
        {
            "daily_return": daily_return,
            "log_return": log_return,
            "return_5d": compound_returns(daily_return, 5),
            "return_20d": compound_returns(daily_return, 20),
            "return_60d": compound_returns(daily_return, 60),
            "return_z5": rolling_zscore(daily_return, 5),
            "return_z20": rolling_zscore(daily_return, 20),
            "abs_return": daily_return.abs(),
            "price_rel_ma20": _price_relative_to_sma(close, corp_gap, 20),
            "price_rel_ma60": _price_relative_to_sma(close, corp_gap, 60),
        },
        index=df.index,
        dtype="float64",
    )
    return out
