"""Frozen Phase 5 numerical kernels, isolated from research I/O entry points.

Function bodies are reproduced verbatim from the approved source modules.
Their source hashes are pinned in contract.json.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

import pandas as pd


# Approved numerical functions from src/features/_common.py

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


# Approved numerical functions from src/features/returns.py

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


# Approved numerical functions from src/features/volatility.py

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


# Approved numerical functions from src/features/liquidity.py

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


# Approved numerical functions from src/features/gaps.py

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


# Approved numerical functions from src/features/market_stress.py

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


# Approved numerical functions from src/features/market_relative.py

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
    mean_stock_60 = daily_return.rolling(window=60, min_periods=60).mean().abs()
    mean_bench_60 = bench_ret.rolling(window=60, min_periods=60).mean().abs()
    std_stock_60 = std_stock_60.where(std_stock_60 > np.maximum(1e-15, 1e-12 * mean_stock_60), np.nan)
    std_bench_60 = std_bench_60.where(std_bench_60 > np.maximum(1e-15, 1e-12 * mean_bench_60), np.nan)
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
