"""
Category 6 — Market-Stress & Correlation Features (F52 to F55)
plus Supplementary India VIX Features (C1, outside F01-F55)
================================================================

Audit Status (Phase 5 Contract §13): APPROVED & IMPLEMENTED
Core Features (Category 6 — 4 Features per C2):
    F52: nifty_vol_20d    — 20-day NIFTY 50 realized log-return vol, annualized
    F53: nifty_vol_ratio  — nifty_vol_5d / nifty_vol_20d (both annualized per C3)
    F54: nifty_return_20d — 20-day NIFTY 50 simple return
    F55: vol_beta         — (vol_20d_t - vol_20d_{t-20}) / vol_20d_{t-20}

Supplementary Features (C1 — Separate from F01-F55):
    vix_close             — India VIX closing level
    vix_return_1d         — 1-day percentage change in India VIX
    vix_z20               — 20-day rolling z-score of India VIX close
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.features._common import (
    ANNUALIZATION_FACTOR,
    compute_masked_log_return,
    rolling_zscore,
    safe_divide,
)

CORE_MARKET_STRESS_COLUMNS: tuple[str, ...] = (
    "nifty_vol_20d",
    "nifty_vol_ratio",
    "nifty_return_20d",
    "vol_beta",
)
VIX_SUPPLEMENTARY_COLUMNS: tuple[str, ...] = (
    "vix_close",
    "vix_return_1d",
    "vix_z20",
)


def _normalize_date_series(dates: pd.Series) -> pd.Series:
    """Normalizes date column to tz-naive midnight datetime64[ns] for joins."""
    dt = pd.to_datetime(dates)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    return dt.dt.normalize()


def prepare_benchmark_features(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Precomputes NIFTY 50 level, volatility, return, and volume series by date.

    Args:
        benchmark_df: NIFTY 50 OHLCV DataFrame (from ``nifty50_clean.parquet``)
            with columns ``date``, ``close``, ``volume``.

    Returns:
        DataFrame indexed by normalized unique trading ``date`` containing:
        ``nifty_close``, ``nifty_daily_return``, ``nifty_vol_5d``,
        ``nifty_vol_20d``, ``nifty_vol_ratio``, ``nifty_return_20d``,
        and ``nifty_volume_ratio`` (NaN when index volume <= 0 per C5).
    """
    if "_prepared_benchmark" in getattr(benchmark_df, "attrs", {}):
        return benchmark_df

    work = benchmark_df.copy()
    work["_norm_date"] = _normalize_date_series(work["date"])
    work = work.sort_values("_norm_date", kind="mergesort").drop_duplicates(
        subset=["_norm_date"], keep="last"
    )
    work = work.set_index("_norm_date")

    close = work["close"].astype("float64")
    volume = work["volume"].astype("float64") if "volume" in work.columns else pd.Series(np.nan, index=work.index, dtype="float64")

    prev_close = close.shift(1)
    valid_ret = (close > 0.0) & (prev_close > 0.0) & np.isfinite(close) & np.isfinite(prev_close)
    nifty_log_ret = pd.Series(np.nan, index=work.index, dtype="float64")
    nifty_log_ret.loc[valid_ret] = np.log(close.loc[valid_ret] / prev_close.loc[valid_ret])

    nifty_daily_return = safe_divide(close - prev_close, prev_close)
    nifty_vol_5d = nifty_log_ret.rolling(window=5, min_periods=5).std(ddof=1) * ANNUALIZATION_FACTOR
    nifty_vol_20d = nifty_log_ret.rolling(window=20, min_periods=20).std(ddof=1) * ANNUALIZATION_FACTOR
    nifty_vol_ratio = safe_divide(nifty_vol_5d, nifty_vol_20d)
    nifty_return_20d = safe_divide(close - close.shift(20), close.shift(20))

    # C5: Do not fabricate benchmark volume; volume <= 0 yields NaN ratio
    nifty_volume_ma20 = volume.rolling(window=20, min_periods=20).mean()
    pos_vol = volume.where(volume > 0.0, np.nan)
    nifty_volume_ratio = safe_divide(pos_vol, nifty_volume_ma20)

    prepared = pd.DataFrame(
        {
            "nifty_close": close,
            "nifty_daily_return": nifty_daily_return,
            "nifty_vol_5d": nifty_vol_5d,
            "nifty_vol_20d": nifty_vol_20d,
            "nifty_vol_ratio": nifty_vol_ratio,
            "nifty_return_20d": nifty_return_20d,
            "nifty_volume_ratio": nifty_volume_ratio,
        },
        index=work.index,
        dtype="float64",
    )
    prepared.attrs["_prepared_benchmark"] = True
    return prepared


def prepare_vix_features(vix_df: pd.DataFrame) -> pd.DataFrame:
    """Precomputes supplementary India VIX series (C1) indexed by date.

    Args:
        vix_df: India VIX DataFrame (from ``india_vix_clean.parquet``) with
            columns ``date`` and ``close``.

    Returns:
        DataFrame indexed by normalized unique trading ``date`` containing:
        ``vix_close``, ``vix_return_1d``, and ``vix_z20``.
    """
    if "_prepared_vix" in getattr(vix_df, "attrs", {}):
        return vix_df

    work = vix_df.copy()
    work["_norm_date"] = _normalize_date_series(work["date"])
    work = work.sort_values("_norm_date", kind="mergesort").drop_duplicates(
        subset=["_norm_date"], keep="last"
    )
    work = work.set_index("_norm_date")

    vix_close = work["close"].astype("float64")
    vix_return_1d = safe_divide(vix_close - vix_close.shift(1), vix_close.shift(1))
    vix_z20 = rolling_zscore(vix_close, 20)

    prepared = pd.DataFrame(
        {
            "vix_close": vix_close,
            "vix_return_1d": vix_return_1d,
            "vix_z20": vix_z20,
        },
        index=work.index,
        dtype="float64",
    )
    prepared.attrs["_prepared_vix"] = True
    return prepared


def align_market_columns_by_date(
    stock_dates: pd.Series,
    market_prepared: pd.DataFrame,
    *,
    ffill_cols: Iterable[str] = (),
    zero_after_start_cols: Iterable[str] = (),
    exact_cols: Iterable[str] = (),
) -> pd.DataFrame:
    """Aligns precomputed market columns onto ``stock_dates`` without lookahead.

    - ``ffill_cols`` use strictly backward forward-fill over the union of market
      and stock dates (never backward-filled).
    - ``zero_after_start_cols`` use exact match and fill unmatched dates strictly
      after ``market_prepared.index.min()`` with ``0.0``.
    - ``exact_cols`` use exact date match only (preserving ``NaN`` on gaps).
    """
    target_dates = _normalize_date_series(stock_dates)
    out = pd.DataFrame(index=stock_dates.index)

    ffill_list = list(ffill_cols)
    zero_list = list(zero_after_start_cols)
    exact_list = list(exact_cols)

    if ffill_list and not market_prepared.empty:
        union_idx = market_prepared.index.union(pd.Index(target_dates.unique())).sort_values()
        ffilled = market_prepared[ffill_list].reindex(union_idx).ffill()
        aligned = ffilled.reindex(target_dates)
        for col in ffill_list:
            out[col] = aligned[col].to_numpy(dtype="float64")
    else:
        for col in ffill_list:
            out[col] = np.nan

    if zero_list:
        first_market_date = market_prepared.index.min() if not market_prepared.empty else pd.NaT
        for col in zero_list:
            if market_prepared.empty:
                out[col] = np.nan
                continue
            vals = market_prepared[col].reindex(target_dates).to_numpy(dtype="float64")
            unmatched_after_start = (~target_dates.isin(market_prepared.index)) & (
                target_dates > first_market_date
            )
            vals = np.where(unmatched_after_start.to_numpy(), 0.0, vals)
            out[col] = vals

    if exact_list:
        for col in exact_list:
            if market_prepared.empty:
                out[col] = np.nan
            else:
                out[col] = market_prepared[col].reindex(target_dates).to_numpy(dtype="float64")

    return out


def compute_market_stress_features(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Computes Category 6 features (F52-F55) and optional India VIX features.

    Args:
        stock_df: Chronologically sorted single-stock OHLCV DataFrame with
            ``date``, ``close`` (and optionally ``is_corporate_action_gap``).
        benchmark_df: NIFTY 50 OHLCV DataFrame (raw or prepared).
        vix_df: Optional India VIX DataFrame (raw or prepared). When provided,
            emits ``vix_close``, ``vix_return_1d``, and ``vix_z20``.

    Returns:
        DataFrame indexed identically to ``stock_df`` containing float64 columns
        ``nifty_vol_20d``, ``nifty_vol_ratio``, ``nifty_return_20d``, ``vol_beta``,
        and (if ``vix_df`` is provided) ``vix_close``, ``vix_return_1d``, ``vix_z20``.
    """
    bench_prep = prepare_benchmark_features(benchmark_df)
    aligned_bench = align_market_columns_by_date(
        stock_df["date"],
        bench_prep,
        ffill_cols=("nifty_vol_20d", "nifty_vol_ratio", "nifty_return_20d"),
    )

    log_return = compute_masked_log_return(stock_df)
    vol_20d = log_return.rolling(window=20, min_periods=20).std(ddof=1) * ANNUALIZATION_FACTOR
    prev_vol_20d = vol_20d.shift(20)
    vol_beta = safe_divide(vol_20d - prev_vol_20d, prev_vol_20d)

    out = pd.DataFrame(
        {
            "nifty_vol_20d": aligned_bench["nifty_vol_20d"],
            "nifty_vol_ratio": aligned_bench["nifty_vol_ratio"],
            "nifty_return_20d": aligned_bench["nifty_return_20d"],
            "vol_beta": vol_beta,
        },
        index=stock_df.index,
        dtype="float64",
    )

    if vix_df is not None:
        vix_prep = prepare_vix_features(vix_df)
        aligned_vix = align_market_columns_by_date(
            stock_df["date"],
            vix_prep,
            ffill_cols=("vix_close", "vix_z20"),
            zero_after_start_cols=("vix_return_1d",),
        )
        for col in VIX_SUPPLEMENTARY_COLUMNS:
            out[col] = aligned_vix[col]

    return out
