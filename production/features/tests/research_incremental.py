"""Incremental parity against independent frozen outputs on real histories."""

import numpy as np
import pandas as pd

from production.features.contract import ROOT, FEATURES, FLOAT32_RTOL, FLOAT32_ATOL
from production.features.state import initialize, update, FeatureState


def check_research_incremental(*, all_tickers=False):
    names = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "COCHINSHIP.NS", "SCI.NS", "FORCEMOT.NS",
             "MOTILALOFS.NS", "PATANJALI.NS", "YESBANK.NS", "VAML.NS", "VOGL.NS", "VISL.NS"]
    filters = None if all_tickers else [("ticker", "in", names)]
    stock = pd.read_parquet(ROOT / "data/interim/stocks/ohlcv_clean.parquet", filters=filters)
    reference = pd.read_parquet(ROOT / "data/processed/features.parquet", filters=filters)
    benchmark = pd.read_parquet(ROOT / "data/interim/market/nifty50_clean.parquet")
    vix = pd.read_parquet(ROOT / "data/interim/market/india_vix_clean.parquet")
    issues, rows, maxima = [], 0, {}
    for ticker, group in stock.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        expected = reference.loc[reference.ticker.eq(ticker)].sort_values("date").reset_index(drop=True)
        base = len(group) - 20
        _, state = initialize(group.iloc[:base], benchmark, vix, source_version="frozen-reference")
        for start, stop in [(base, base+1), (base+1, base+5), (base+5, len(group))]:
            actual, state = update(FeatureState.from_json(state.to_json()), group.iloc[start:stop],
                                   benchmark, vix, source_version="frozen-reference")
            values = actual[list(FEATURES)].to_numpy()
            target = expected.iloc[start:stop][list(FEATURES)].to_numpy()
            if not np.array_equal(np.isnan(values), np.isnan(target)) or not np.allclose(
                    values, target, rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL, equal_nan=True):
                issues.append(ticker + ":" + str(start))
            if not np.array_equal(actual.rsi_14.to_numpy(), expected.rsi_14.iloc[start:stop].to_numpy(), equal_nan=True):
                issues.append(ticker + ":RSI")
            for index, name in enumerate(FEATURES):
                difference = np.abs(values[:, index] - target[:, index])
                finite = difference[np.isfinite(difference)]
                maxima[name] = max(maxima.get(name, 0), float(finite.max()) if len(finite) else 0)
            rows += len(actual)
    return {"result": "PASS" if not issues else "FAIL", "tickers": int(stock.ticker.nunique()),
            "incremental_rows": rows, "feature_values_compared": rows * 58,
            "tolerance": {"rtol": FLOAT32_RTOL, "atol": FLOAT32_ATOL, "null_masks": "exact", "Wilder_RSI": "exact float32"},
            "maximum_absolute_differences": {name: value for name, value in maxima.items() if value},
            "discrepancies": issues}
