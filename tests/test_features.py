"""Comprehensive unit and integration tests for Phase 5 feature engineering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    CORE_FEATURE_COLUMNS,
    FeatureEngineer,
)
from src.features.gaps import compute_gap_features
from src.features.liquidity import compute_liquidity_features
from src.features.market_relative import compute_market_relative_features
from src.features.market_stress import (
    VIX_SUPPLEMENTARY_COLUMNS,
    compute_market_stress_features,
)
from src.features.returns import compute_returns_features
from src.features.volatility import compute_volatility_features


def _make_ohlcv(
    n_rows: int = 120,
    *,
    ticker: str = "TEST.NS",
    start_date: str = "2020-01-01",
    seed: int = 42,
    freq: str = "B",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_rows, freq=freq)
    rets = rng.normal(0.0005, 0.015, size=n_rows)
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = close * (1.0 + rng.normal(0.0, 0.003, size=n_rows))
    spread_hi = np.abs(rng.normal(0.01, 0.004, size=n_rows))
    spread_lo = np.abs(rng.normal(0.01, 0.004, size=n_rows))
    high = np.maximum(open_, close) * (1.0 + spread_hi)
    low = np.minimum(open_, close) * (1.0 - spread_lo)
    volume = rng.integers(10_000, 500_000, size=n_rows, dtype=np.int64)

    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ticker,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "is_corporate_action_gap": False,
            "is_market_shock": False,
            "is_zero_volume": False,
            "is_stale_price": False,
            "is_short_history": n_rows < 252,
            "is_warmup": np.arange(n_rows) < 80,
            "is_benchmark_available": True,
            "is_muhurat_session": False,
            "is_valid_ohlcv": True,
        }
    )


def _assert_no_inf(df: pd.DataFrame) -> None:
    num = df.select_dtypes(include=[np.number]).to_numpy(dtype="float64")
    assert not np.isinf(num).any(), "Found unexpected +/-inf values in feature output."


def test_core_feature_inventory_count_and_names() -> None:
    assert len(CORE_FEATURE_COLUMNS) == 55
    assert len(set(CORE_FEATURE_COLUMNS)) == 55
    for vix_col in VIX_SUPPLEMENTARY_COLUMNS:
        assert vix_col not in CORE_FEATURE_COLUMNS


def test_returns_closed_form_and_compound() -> None:
    # Constant +1% daily return -> exact compounded 5d/20d/60d and price_rel_ma
    n = 80
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 * (1.01 ** np.arange(n))
    df = pd.DataFrame(
        {"date": dates, "close": close, "is_corporate_action_gap": False}
    )
    out = compute_returns_features(df)
    _assert_no_inf(out)

    assert np.isnan(out["daily_return"].iloc[0])
    assert np.isclose(out["daily_return"].iloc[1], 0.01, atol=1e-12)
    assert np.isclose(out["log_return"].iloc[1], np.log(1.01), atol=1e-12)
    assert np.isnan(out["return_5d"].iloc[4])
    assert np.isclose(out["return_5d"].iloc[5], (1.01 ** 5) - 1.0, atol=1e-12)
    assert np.isclose(out["return_20d"].iloc[20], (1.01 ** 20) - 1.0, atol=1e-12)
    assert np.isclose(out["return_60d"].iloc[60], (1.01 ** 60) - 1.0, atol=1e-12)
    # Zero variance of constant +1% returns must guard z-scores safely to NaN
    assert out["return_z5"].iloc[10:].isna().all()
    assert out["return_z20"].iloc[25:].isna().all()

    expected_rel20 = close[19] / np.mean(close[:20]) - 1.0
    assert np.isnan(out["price_rel_ma20"].iloc[18])
    assert np.isclose(out["price_rel_ma20"].iloc[19], expected_rel20, atol=1e-12)


def test_volatility_annualization_downside_mdd20_and_wilder_rsi() -> None:
    # Alternating returns so both vol_5d and vol_20d are well-defined and positive
    n = 90
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    closes = [100.0]
    for i in range(1, n):
        step = 1.02 if i % 2 == 1 else 0.99
        closes.append(closes[-1] * step)
    close = np.array(closes)
    # Constant H/L ratio exp(0.02) -> exact Parkinson vol
    low = close * 0.99
    high = low * np.exp(0.02)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "is_corporate_action_gap": False,
        }
    )
    out = compute_volatility_features(df)
    _assert_no_inf(out)

    # C3: F11 vol_5d annualized by sqrt(252) -> matches manual std(log_ret)*sqrt(252)
    lr = np.log(close[1:] / close[:-1])
    expected_vol5 = np.std(lr[1:6], ddof=1) * np.sqrt(252.0)
    assert np.isclose(out["vol_5d"].iloc[6], expected_vol5, atol=1e-12)

    expected_park = 0.02 / np.sqrt(4.0 * np.log(2.0)) * np.sqrt(252.0)
    assert np.isclose(out["parkinson_vol"].iloc[19], expected_park, atol=1e-12)

    # C4: F18 target-0 downside semi-deviation & F19 sortino_component
    w_lr = lr[1:21]
    expected_down = np.sqrt(np.mean(np.minimum(w_lr, 0.0) ** 2)) * np.sqrt(252.0)
    assert np.isclose(out["downside_vol_20d"].iloc[21], expected_down, atol=1e-12)
    ret_20d = compute_returns_features(df)["return_20d"].iloc[21]
    assert np.isclose(out["sortino_component"].iloc[21], ret_20d / expected_down, atol=1e-12)

    # C10: F20 max_drawdown_20d uses strictly [t-19, t] (20 observations)
    # Put an isolated high peak at t=0 so by t=20 window is [1, 20] (peak at t=0 excluded!)
    c_mdd = np.full(25, 100.0)
    c_mdd[0] = 200.0  # t=0 peak
    c_mdd[1:20] = 120.0
    c_mdd[20] = 90.0  # Window [1, 20] has max=120, so DD=(90-120)/120=-0.25 (NOT (90-200)/200=-0.55)
    df_mdd = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=25, freq="B"),
            "open": c_mdd,
            "high": c_mdd,
            "low": c_mdd,
            "close": c_mdd,
            "is_corporate_action_gap": False,
        }
    )
    mdd_out = compute_volatility_features(df_mdd)
    assert np.isclose(mdd_out["max_drawdown_20d"].iloc[20], -0.25, atol=1e-12)

    # C4: Wilder RSI explicit edge cases (100.0 all gains, 0.0 all losses, 50.0 flat)
    for delta, expected_rsi in [(1.0, 100.0), (-1.0, 0.0), (0.0, 50.0)]:
        base = 100.0 + np.arange(20, dtype="float64") * delta
        edge_df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=20, freq="B"),
                "open": base,
                "high": base,
                "low": base,
                "close": base,
            }
        )
        rsi_series = compute_volatility_features(edge_df)["rsi_14"]
        assert np.isnan(rsi_series.iloc[13])
        assert np.isclose(rsi_series.iloc[14], expected_rsi, atol=1e-12)
        assert np.isclose(rsi_series.iloc[19], expected_rsi, atol=1e-12)


def test_liquidity_amihud_min_periods_zero_vol_and_low_vol_streak() -> None:
    df = _make_ohlcv(50, seed=7)
    # Zero volume at rows 5..14 (10 consecutive rows): within [20..39] window, valid count fluctuates
    df.loc[15:21, "volume"] = 0  # 7 zero-vol rows inside early [10..29] -> <15 valid obs!
    out = compute_liquidity_features(df)
    _assert_no_inf(out)

    # Zero volume -> F27 amihud_ratio NaN, F35 zero_vol_flag == 1.0, F23 log_volume == 0.0
    assert out.loc[15:21, "amihud_ratio"].isna().all()
    assert (out.loc[15:21, "zero_vol_flag"] == 1.0).all()
    assert (out.loc[15:21, "log_volume"] == 0.0).all()
    assert (out.loc[15:21, "turnover_value"] == 0.0).all()

    # C7: amihud_ma20 NaN when valid count < 15 (e.g. row 25 has returns starting row 1: rows 1..25 minus 7 zero-vol = 18 overall, check within last 20 rows [6..25] -> 13 valid < 15!)
    assert np.isnan(out["amihud_ma20"].iloc[25])
    assert np.isnan(out["amihud_z20"].iloc[25])
    # Once >= 15 valid observations exist in trailing 20 rows (row 35 has [16..35] -> 6 zero-vol -> 14 valid; row 36 has [17..36] -> 5 zero-vol -> 15 valid!)
    assert np.isnan(out["amihud_ma20"].iloc[35])
    assert np.isfinite(out["amihud_ma20"].iloc[36])

    # F36 low_volume_streak: rows 0..22 MUST be NaN (never treat NaN volume_ratio < 0.3 as False/0)
    clean_df = _make_ohlcv(40, seed=11)
    clean_out = compute_liquidity_features(clean_df)
    assert clean_out["low_volume_streak"].iloc[:23].isna().all()
    assert np.isfinite(clean_out["low_volume_streak"].iloc[23])


def test_stale_price_and_constant_volume_no_inf() -> None:
    df = _make_ohlcv(60, seed=99)
    # Locked-price trading halt for 35 sessions
    df.loc[20:54, ["open", "high", "low", "close"]] = 150.0
    df.loc[20:54, "volume"] = 0

    bench = _make_ohlcv(60, ticker="^NSEI", seed=101)
    vix = _make_ohlcv(60, ticker="^INDIAVIX", seed=102)
    eng = FeatureEngineer()
    res = eng.compute_stock_features(df, bench, vix)
    _assert_no_inf(res)

    # Division-by-zero guarded columns on locked segment -> NaN, never inf
    assert np.isnan(res["return_z20"].iloc[50])
    assert np.isnan(res["range_z20"].iloc[50])
    assert np.isnan(res["volume_z20"].iloc[50])
    assert np.isnan(res["volume_ratio"].iloc[50])
    assert np.isnan(res["vol_ratio_5_20"].iloc[50])
    assert np.isnan(res["sortino_component"].iloc[50])
    assert np.isnan(res["rolling_corr_60d"].iloc[59])
    assert res["parkinson_vol"].iloc[50] == pytest.approx(0.0)


def test_corwin_schultz_flooring_and_gaps() -> None:
    df = _make_ohlcv(40, seed=13)
    # Two narrow non-nested gapping bars make gamma >> beta -> alpha < 0 -> floored at 0.0
    df.loc[10, ["high", "low"]] = [101.0, 100.0]
    df.loc[11, ["high", "low"]] = [110.0, 108.9]
    out = compute_gap_features(df)
    _assert_no_inf(out)

    assert np.isnan(out["cs_spread_est"].iloc[0])
    assert out["cs_spread_est"].iloc[11] == pytest.approx(0.0)
    assert out["cs_spread_est"].dropna().ge(0.0).all()
    assert out["cs_spread_ma10"].iloc[:10].isna().all()
    assert np.isfinite(out["cs_spread_ma10"].iloc[10])


def test_corporate_action_masking_and_market_shock_retention() -> None:
    df = _make_ohlcv(140, seed=21)
    bench = _make_ohlcv(140, ticker="^NSEI", seed=22)

    # Inject a 2:1 unadjusted split gap at row k = 85
    k = 85
    df.loc[k:, ["open", "high", "low", "close"]] *= 0.5
    df.loc[k, "is_corporate_action_gap"] = True

    # Inject genuine market shock (YESBANK style) at row m = 130
    m = 130
    df.loc[m, ["open", "high", "low", "close"]] *= 0.45
    df.loc[m, "is_market_shock"] = True

    eng = FeatureEngineer()
    res = eng.compute_stock_features(df, bench)
    _assert_no_inf(res)

    # 1-day masked features at row k
    for col in (
        "daily_return",
        "log_return",
        "abs_return",
        "true_range",
        "amihud_ratio",
        "zero_return_flag",
        "overnight_gap",
        "overnight_gap_abs",
        "rsi_14",
    ):
        assert np.isnan(res[col].iloc[k]), f"{col} should be NaN at corp-gap row {k}"

    # Multi-day compounded returns F03-F05 recover right after window leaves row k
    assert res["return_5d"].iloc[k : k + 5].isna().all()
    assert np.isfinite(res["return_5d"].iloc[k + 5])
    assert res["return_20d"].iloc[k : k + 20].isna().all()
    assert np.isfinite(res["return_20d"].iloc[k + 20])

    # Price-level windows F09, F20 mask across [k .. k+18] and recover at k+19 (19 post-split closes)
    assert res["price_rel_ma20"].iloc[k : k + 19].isna().all()
    assert np.isfinite(res["price_rel_ma20"].iloc[k + 19])
    assert res["max_drawdown_20d"].iloc[k : k + 19].isna().all()
    assert np.isfinite(res["max_drawdown_20d"].iloc[k + 19])

    # True range ATR-20 masks across [k .. k+19] and recovers at k+20
    assert res["atr_20d"].iloc[k : k + 20].isna().all()
    assert np.isfinite(res["atr_20d"].iloc[k + 20])

    # Corwin-Schultz 2-day pair mask: both k and k+1 are NaN, k+2 recovers
    assert np.isnan(res["cs_spread_est"].iloc[k])
    assert np.isnan(res["cs_spread_est"].iloc[k + 1])
    assert np.isfinite(res["cs_spread_est"].iloc[k + 2])

    # Wilder RSI preserves prior EWM state past k
    assert np.isfinite(res["rsi_14"].iloc[k + 1])

    # Genuine market shock at row m MUST be preserved cleanly (not masked)
    for col in ("daily_return", "log_return", "true_range", "amihud_ratio", "overnight_gap", "rsi_14"):
        assert np.isfinite(res[col].iloc[m]), f"{col} must NOT be masked on is_market_shock row {m}"


def test_market_relative_perfect_tracking_and_f51_guards() -> None:
    bench = _make_ohlcv(130, ticker="^NSEI", seed=55)
    stock = bench.copy()
    stock["ticker"] = "TRACK.NS"
    stock["close"] = bench["close"] * 2.0
    stock["open"] = bench["open"] * 2.0
    stock["high"] = bench["high"] * 2.0
    stock["low"] = bench["low"] * 2.0
    stock["volume"] = bench["volume"] * 3

    # Remove one date from benchmark (row 40) and set benchmark volume == 0 on row 50
    bench_gapped = bench.drop(index=40).reset_index(drop=True)
    bench_gapped.loc[bench_gapped["date"] == bench["date"].iloc[50], "volume"] = 0
    stock.loc[40, "is_benchmark_available"] = False
    stock["benchmark_return_fallback"] = bench["close"].pct_change().fillna(0.0)
    stock.loc[40, "benchmark_return_fallback"] = 0.0

    rel = compute_market_relative_features(stock, bench_gapped)
    _assert_no_inf(rel)

    assert np.isclose(rel["excess_return_1d"].iloc[30], 0.0, atol=1e-12)
    # Window [66..125] excludes the unmatched date at row 40 -> exact unit beta/corr and zero idio vol
    assert np.isclose(rel["rolling_beta_60d"].iloc[125], 1.0, atol=1e-10)
    assert np.isclose(rel["rolling_corr_60d"].iloc[125], 1.0, atol=1e-10)
    assert np.isclose(rel["idiosyncratic_vol"].iloc[125], 0.0, atol=1e-6)
    assert rel["rolling_corr_60d"].dropna().between(-1.0, 1.0).all()

    # C5: F51 MUST be NaN on unmatched date (row 40) and zero-volume benchmark row (row 50)
    assert np.isnan(rel["relative_volume"].iloc[40])
    assert np.isnan(rel["relative_volume"].iloc[50])
    # Rows 30 and 75 both have clean 20-day volume windows matching scaled benchmark volumes
    assert np.isclose(rel["relative_volume"].iloc[30], 1.0, atol=1e-6)
    assert np.isclose(rel["relative_volume"].iloc[75], 1.0, atol=1e-6)


def test_vix_alignment_and_market_stress() -> None:
    stock = _make_ohlcv(50, ticker="VIXT.NS", seed=61)
    bench = _make_ohlcv(50, ticker="^NSEI", seed=62)
    vix = _make_ohlcv(50, ticker="^INDIAVIX", seed=63)
    # Drop row 30 from VIX feed to test backward .ffill() & 0.0 return fallback
    vix_gapped = vix.drop(index=30).reset_index(drop=True)

    stress = compute_market_stress_features(stock, bench, vix_df=vix_gapped)
    _assert_no_inf(stress)
    assert list(stress.columns) == [
        "nifty_vol_20d",
        "nifty_vol_ratio",
        "nifty_return_20d",
        "vol_beta",
        "vix_close",
        "vix_return_1d",
        "vix_z20",
    ]
    assert np.isclose(stress["vix_close"].iloc[30], stress["vix_close"].iloc[29], atol=1e-12)
    assert np.isclose(stress["vix_z20"].iloc[30], stress["vix_z20"].iloc[29], atol=1e-12)
    assert stress["vix_return_1d"].iloc[30] == 0.0


def test_no_lookahead_and_ticker_isolation_and_short_history(tmp_path: Path) -> None:
    s1 = _make_ohlcv(120, ticker="INFY.NS", seed=71)
    s2 = _make_ohlcv(91, ticker="VAML.NS", seed=72)  # Short-history 91 rows
    bench = _make_ohlcv(120, ticker="^NSEI", seed=73)
    vix = _make_ohlcv(120, ticker="^INDIAVIX", seed=74)

    eng = FeatureEngineer()
    full_s1 = eng.compute_stock_features(s1, bench, vix)
    trunc_s1 = eng.compute_stock_features(s1.iloc[:90].copy(), bench.iloc[:90].copy(), vix.iloc[:90].copy())
    pd.testing.assert_frame_equal(
        full_s1.iloc[:90].reset_index(drop=True),
        trunc_s1.reset_index(drop=True),
        check_exact=False,
        atol=1e-6,
    )

    # Multi-ticker process_universe streaming Parquet check
    ohlcv_path = tmp_path / "ohlcv.parquet"
    bench_path = tmp_path / "nifty50.parquet"
    vix_path = tmp_path / "vix.parquet"
    out_path = tmp_path / "features.parquet"

    combined = pd.concat([s1, s2], ignore_index=True)
    combined.to_parquet(ohlcv_path, index=False)
    bench.to_parquet(bench_path, index=False)
    vix.to_parquet(vix_path, index=False)

    eng.process_universe(
        clean_ohlcv_path=ohlcv_path,
        benchmark_path=bench_path,
        output_path=out_path,
        vix_path=vix_path,
        batch_tickers=1,
    )
    assert out_path.exists()
    out_df = pd.read_parquet(out_path)
    _assert_no_inf(out_df)

    assert len(out_df) == len(s1) + len(s2)
    for col in list(CORE_FEATURE_COLUMNS) + list(VIX_SUPPLEMENTARY_COLUMNS):
        assert col in out_df.columns
        assert out_df[col].dtype == np.float32

    # Short-history ticker produces finite F05/F10/F13/F22/F48-F50 in post-warmup rows 80..90
    vaml_post = out_df[(out_df["ticker"] == "VAML.NS") & (~out_df["is_warmup"])]
    assert len(vaml_post) == 11
    for col in ("return_60d", "price_rel_ma60", "vol_60d", "vol_of_vol", "rolling_beta_60d", "rolling_corr_60d"):
        assert vaml_post[col].notna().all()
