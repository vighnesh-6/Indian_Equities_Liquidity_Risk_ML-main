"""Offline research parity and independent persisted-feature reconciliation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contract import CONTRACT, FEATURES, FLAGS, ROOT, verify_contract
from .engine import calculate


def research_parity(tickers=None, progress=print):
    verify_contract()
    filters = [("ticker", "in", list(tickers))] if tickers else None
    stock = pd.read_parquet(ROOT / "data/interim/stocks/ohlcv_clean.parquet", filters=filters)
    expected = pd.read_parquet(ROOT / CONTRACT["reference_features"], filters=filters)
    benchmark = pd.read_parquet(ROOT / "data/interim/market/nifty50_clean.parquet")
    vix = pd.read_parquet(ROOT / "data/interim/market/india_vix_clean.parquet")
    problems, rows = [], 0
    for number, (ticker, group) in enumerate(stock.groupby("ticker", sort=True), 1):
        actual = calculate(group, benchmark, vix)
        reference = expected.loc[expected.ticker.eq(ticker)].sort_values("date").reset_index(drop=True)
        if not actual.date.equals(reference.date):
            problems.append(ticker + ":DATES")
        for name in [*FEATURES, *FLAGS, "cap_segment"]:
            if not np.array_equal(actual[name].to_numpy(), reference[name].to_numpy(), equal_nan=name in FEATURES):
                problems.append(ticker + ":" + name)
        rows += len(actual)
        if number % 25 == 0:
            progress(f"Research parity verified for {number} tickers")
    return {"result": "PASS" if not problems else "FAIL", "tickers": stock.ticker.nunique(),
            "rows": rows, "feature_values_compared": rows * 58, "comparison": "exact float32 with identical null masks",
            "reference_sha256": CONTRACT["reference_sha256"], "discrepancies": problems}


def reconcile(database, expected, version):
    """Read stored float32 values independently from the persistence method."""
    problems = []
    tickers = sorted(expected.ticker.unique())
    with database.transaction(read_only=True) as cursor:
        cursor.execute("SELECT ticker,date," + ",".join([*FLAGS, *FEATURES])
                       + " FROM liquidity.feature_values WHERE feature_version=%s AND ticker=ANY(%s) ORDER BY ticker,date",
                       (version, tickers))
        records = cursor.fetchall()
    actual = pd.DataFrame(records)
    reference = expected.sort_values(["ticker", "date"]).reset_index(drop=True)
    if len(actual) != len(reference):
        return {"result": "FAIL", "rows": len(actual), "discrepancies": ["ROW_COUNT"]}
    actual["date"] = pd.to_datetime(actual.date)
    if not actual[["ticker", "date"]].equals(reference[["ticker", "date"]]):
        problems.append("KEYS")
    for name in FEATURES:
        values = pd.to_numeric(actual[name], errors="raise").to_numpy(dtype="float32", na_value=np.nan)
        if not np.array_equal(values, reference[name].to_numpy(), equal_nan=True):
            problems.append(name)
    for name in FLAGS:
        if not np.array_equal(actual[name].to_numpy(dtype=bool), reference[name].to_numpy()):
            problems.append(name)
    return {"result": "PASS" if not problems else "FAIL", "rows": len(actual), "discrepancies": problems}
