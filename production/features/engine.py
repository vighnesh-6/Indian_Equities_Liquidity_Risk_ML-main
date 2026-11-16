"""Per-instrument production orchestration of the frozen numerical contract."""

import hashlib
import json

import numpy as np
import pandas as pd

from . import kernels
from .contract import FEATURES, FLAGS, ROOT
from .inputs import DataValidator


def validate_market_frame(frame, *, stock=False):
    required = ["date", "open", "high", "low", "close", "volume"]
    if stock:
        required.append("ticker")
    if frame.empty or not set(required).issubset(frame):
        raise ValueError("Nonempty observed OHLCV and dates are required")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    if result.date.dt.tz is not None or not result.date.eq(result.date.dt.normalize()).all():
        raise ValueError("Trading dates must be timezone-naive midnight values")
    if result.date.duplicated().any() or result.date.isna().any():
        raise ValueError("Duplicate or missing trading dates")
    if result.date.eq(pd.Timestamp("2026-11-08")).any():
        raise ValueError("2026-11-08 is explicitly excluded")
    if stock and (result.ticker.nunique() != 1 or result.ticker.isna().any()):
        raise ValueError("Feature calculations require exactly one ticker")
    values = result[["open", "high", "low", "close", "volume"]].to_numpy(dtype="float64")
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any():
        raise ValueError("Nonfinite or nonpositive OHLCV")
    if (values[:, 4] < 0).any() or (values[:, 4] != np.floor(values[:, 4])).any():
        raise ValueError("Invalid observed volume")
    if not (result.open.between(result.low, result.high)
            & result.close.between(result.low, result.high)).all():
        raise ValueError("Invalid OHLC bounds")
    return result.sort_values("date", kind="mergesort").reset_index(drop=True)


def prepare_stock(stock, benchmark):
    stock = validate_market_frame(stock, stock=True)
    benchmark = validate_market_frame(benchmark)
    # Reference inputs already contain the approved frozen quality enrichment.
    if set(FLAGS).issubset(stock) and "benchmark_return_fallback" in stock:
        return stock
    validator = DataValidator()
    return validator.clean_and_enrich_stock(
        stock, validator.prepare_benchmark_series(benchmark), warmup_rows=80)


def calculate_prepared(stock, benchmark, vix):
    benchmark = kernels.prepare_benchmark_features(benchmark)
    vix = kernels.prepare_vix_features(vix)
    parts = [kernels.compute_returns_features(stock),
             kernels.compute_volatility_features(stock),
             kernels.compute_liquidity_features(stock),
             kernels.compute_gap_features(stock),
             kernels.compute_market_relative_features(stock, benchmark),
             kernels.compute_market_stress_features(stock, benchmark, vix_df=vix)]
    values = pd.concat(parts, axis=1).loc[:, list(FEATURES)].astype("float32")
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Infinite engineered feature")
    metadata = stock.loc[:, ["date", "ticker", *FLAGS]].copy()
    if "cap_segment" in stock:
        metadata["cap_segment"] = stock.cap_segment
    else:
        universe = json.loads((ROOT / "configs/universe.json").read_text(encoding="utf-8"))
        groups = universe["tickers"]
        mapping = {item["ticker"]: segment for key, segment in
                   [("large_cap_nifty_100", "large"), ("mid_cap_nifty_midcap_150", "mid"),
                    ("small_cap_nifty_smallcap_250_selected", "small")]
                   for item in groups[key]}
        metadata["cap_segment"] = stock.ticker.map(mapping).fillna("unknown")
    return pd.concat([metadata, values], axis=1)


def calculate(stock, benchmark, vix):
    prepared = prepare_stock(stock, benchmark)
    return calculate_prepared(prepared, validate_market_frame(benchmark), validate_market_frame(vix))


def frame_hash(frame):
    """Hash sorted dates, exact float32 values and their missing-value masks."""
    digest = hashlib.sha256()
    for row in frame.sort_values(["ticker", "date"]).itertuples(index=False):
        digest.update(str(row.ticker).encode() + b"\0" + str(row.date).encode())
    values = frame.sort_values(["ticker", "date"])[list(FEATURES)].to_numpy(dtype="float32")
    missing = np.isnan(values)
    digest.update(missing.tobytes())
    digest.update(np.where(missing, 0, values).astype(">f4").tobytes())
    return digest.hexdigest()


def input_hash(frame):
    work = frame.copy()
    work["date"] = pd.to_datetime(work.date).dt.strftime("%Y-%m-%d")
    records = work.replace({np.nan: None}).to_dict("records")
    return hashlib.sha256(json.dumps(records, sort_keys=True, default=str,
                                     allow_nan=False).encode()).hexdigest()
