"""
Category 5 — Market-Relative Behaviour Features (F46 to F51)
============================================================

Audit Status (Phase 5 Contract §13): APPROVED & IMPLEMENTED
Features:
    F46: excess_return_1d  — Stock masked daily_return - benchmark daily return
    F47: excess_return_20d — Stock 20d compounded return - NIFTY 20d return
    F48: rolling_beta_60d  — Cov_60(r_stock, r_nifty) / Var_60(r_nifty)
    F49: rolling_corr_60d  — Corr_60(r_stock, r_nifty), clipped to [-1, 1]
    F50: idiosyncratic_vol — vol_20d * sqrt(max(0, 1 - rolling_corr_60d^2))
    F51: relative_volume   — stock volume_ratio / nifty_volume_ratio (NaN per C5)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features._common import (
    ANNUALIZATION_FACTOR,
    compound_returns,
    compute_masked_daily_return,
    compute_masked_log_return,
    safe_divide,
)
from src.features.market_stress import (
    align_market_columns_by_date,
    prepare_benchmark_features,
)


def compute_market_relative_features(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
) -> pd.DataFrame:
    """Computes Category 5 features F46 to F51 for a single stock.

    Args:
        stock_df: Chronologically sorted single-stock OHLCV DataFrame with
            ``date``, ``close``, ``volume`` (and optionally ``is_corporate_action_gap``,
            ``benchmark_return_fallback``, ``is_benchmark_available``).
        benchmark_df: NIFTY 50 OHLCV DataFrame (raw or prepared via
            ``prepare_benchmark_features``).

    Returns:
        DataFrame indexed identically to ``stock_df`` containing float64 columns
        ``excess_return_1d`` through ``relative_volume`` (F46-F51).
    """
    bench_prep = prepare_benchmark_features(benchmark_df)
    aligned = align_market_columns_by_date(
        stock_df["date"],
        bench_prep,
        ffill_cols=("nifty_return_20d",),
        zero_after_start_cols=("nifty_daily_return",),
        exact_cols=("nifty_volume_ratio",),
    )

    daily_return = compute_masked_daily_return(stock_df)
    return_20d = compound_returns(daily_return, 20)
    log_return = compute_masked_log_return(stock_df)
    vol_20d = log_return.rolling(window=20, min_periods=20).std(ddof=1) * ANNUALIZATION_FACTOR

    if "benchmark_return_fallback" in stock_df.columns:
        bench_ret = stock_df["benchmark_return_fallback"].astype("float64")
    else:
        bench_ret = aligned["nifty_daily_return"]

    excess_return_1d = daily_return - bench_ret
    excess_return_20d = return_20d - aligned["nifty_return_20d"]

    cov_60 = daily_return.rolling(window=60, min_periods=60).cov(bench_ret)
    var_bench_60 = bench_ret.rolling(window=60, min_periods=60).var(ddof=1)
    rolling_beta_60d = safe_divide(cov_60, var_bench_60)

    std_stock_60 = daily_return.rolling(window=60, min_periods=60).std(ddof=1)
    std_bench_60 = bench_ret.rolling(window=60, min_periods=60).std(ddof=1)
    rolling_corr_60d = safe_divide(cov_60, std_stock_60 * std_bench_60).clip(
        lower=-1.0, upper=1.0
    )

    idiosyncratic_vol = vol_20d * np.sqrt(np.maximum(0.0, 1.0 - (rolling_corr_60d ** 2)))

    # F51: strictly NaN if is_benchmark_available is False or benchmark volume ratio invalid (C5)
    volume = stock_df["volume"].astype("float64")
    volume_ma20 = volume.rolling(window=20, min_periods=20).mean()
    volume_ratio = safe_divide(volume, volume_ma20)

    nifty_vol_ratio = aligned["nifty_volume_ratio"]
    if "is_benchmark_available" in stock_df.columns:
        bench_avail = stock_df["is_benchmark_available"].fillna(False).astype(bool)
        nifty_vol_ratio = nifty_vol_ratio.where(bench_avail, np.nan)

    relative_volume = safe_divide(volume_ratio, nifty_vol_ratio)

    out = pd.DataFrame(
        {
            "excess_return_1d": excess_return_1d,
            "excess_return_20d": excess_return_20d,
            "rolling_beta_60d": rolling_beta_60d,
            "rolling_corr_60d": rolling_corr_60d,
            "idiosyncratic_vol": idiosyncratic_vol,
            "relative_volume": relative_volume,
        },
        index=stock_df.index,
        dtype="float64",
    )
    return out
