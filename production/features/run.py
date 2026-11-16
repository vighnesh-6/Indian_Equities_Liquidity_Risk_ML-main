"""Historical and incremental feature calculation with explicit persistence."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from production.pipeline.config import ROOT
from production.pipeline.database import Database
from production.pipeline.ingestion import canonical_tickers
from .contract import verify_contract
from .engine import input_hash
from .repository import FeatureRepository, source_version
from .state import initialize, update


def read_history(database, ticker, through):
    with database.transaction(read_only=True) as cursor:
        cursor.execute("SELECT date,ticker,open,high,low,close,volume FROM liquidity.market_eod_consistent "
                       "WHERE ticker=%s AND date<=%s ORDER BY date", (ticker, through))
        frame = pd.DataFrame(cursor.fetchall())
    if frame.empty:
        raise ValueError("Instrument has no observed canonical input: " + ticker)
    frame["date"] = pd.to_datetime(frame.date)
    return frame


def process(database, through, *, incremental=False, persist=False, progress=print):
    """Use existing canonical market data and write only the selected feature scope."""
    verify_contract()
    with database.transaction(read_only=True) as cursor:
        FeatureRepository.require_schema(cursor)
        cursor.execute("SELECT ticker FROM liquidity.stocks WHERE is_active ORDER BY ticker")
        tickers = [row["ticker"] for row in cursor.fetchall()]
        expected = set(canonical_tickers()) - {"^NSEI", "^INDIAVIX"}
        if set(tickers) != expected:
            raise ValueError("The equity universe must match the frozen 300-stock definition")
        cursor.execute("SELECT last_date,count(*) AS n FROM liquidity.instrument_checkpoints GROUP BY last_date")
        checkpoints = cursor.fetchall()
        if len(checkpoints) != 1 or checkpoints[0]["n"] != 302 or through > checkpoints[0]["last_date"]:
            raise ValueError("Features require complete committed market observations through the requested date")
        source = source_version(cursor)
    benchmark = read_history(database, "^NSEI", through)
    vix = read_history(database, "^INDIAVIX", through)
    market_hashes = {"benchmark": input_hash(benchmark), "vix": input_hash(vix)}
    repository = FeatureRepository(database)
    from .contract import calculation_version
    version = calculation_version(source)
    if persist:
        repository.register_version(source)
    calculated = []
    receipts = []
    states = {}
    input_hashes = {}
    previous_dates = set()
    for ticker in tickers:
        stock = read_history(database, ticker, through)
        state = repository.load_state(ticker, version) if incremental else None
        previous_state_hash = hashlib.sha256(state.to_json().encode()).hexdigest() if state else None
        if incremental and state is None:
            raise ValueError("Incremental processing requires a completed historical checkpoint for " + ticker)
        if state:
            previous_dates.add(state.last_date)
            stock = stock.loc[stock.date.gt(pd.Timestamp(state.last_date))].copy()
            if stock.empty:
                continue
            features, state = update(state, stock, benchmark, vix, source_version=source)
        else:
            features, state = initialize(stock, benchmark, vix, source_version=source)
        calculated.append(features)
        states[ticker] = state
        lineage = {"stock": input_hash(stock), **market_hashes, "previous_state": previous_state_hash}
        input_hashes[ticker] = hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()
        if persist and not incremental:
            # Historical ticker blocks are private until complete dates publish.
            key = "features:" + hashlib.sha256((ticker + version + str(through)).encode()).hexdigest()
            receipts.append(repository.persist(features, {ticker: state}, version=version, request_key=key,
                                               input_sha256=input_hashes[ticker], provenance={"source_version": source,
                                               "market_view": "liquidity.market_eod_consistent", "input_hashes": lineage}))
        progress("Calculated " + ticker)
    if not calculated:
        return {"status": "NOOP", "feature_version": version, "rows": 0}
    combined = pd.concat(calculated, ignore_index=True)
    if incremental and len(previous_dates) != 1:
        raise ValueError("Incremental feature checkpoints must agree")
    if persist and incremental:
        if len(states) != 300:
            raise ValueError("A new feature session requires all 300 equities")
        checksum = hashlib.sha256(json.dumps(input_hashes, sort_keys=True).encode()).hexdigest()
        key = "feature-sessions:" + hashlib.sha256((version + str(through) + checksum).encode()).hexdigest()
        receipts.append(repository.persist(combined, states, version=version, request_key=key,
                                           input_sha256=checksum, provenance={"source_version": source}, publish=True))
    elif persist:
        repository.publish(version, sorted(combined.date.dt.date.unique()))
    return {"status": "SUCCESS" if persist else "CALCULATED_ONLY", "feature_version": version,
            "source_version": source, "rows": len(combined), "tickers": combined.ticker.nunique(), "receipts": receipts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", required=True, type=lambda value: pd.Timestamp(value).date())
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Persist only after separate migration and backfill authorization")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    destination = args.report.resolve()
    if not destination.is_relative_to(ROOT / "production/docs/phase3/validation") or destination.suffix != ".json":
        raise ValueError("Feature reports require the approved Phase 3 validation directory")
    result = process(Database(), args.through, incremental=args.incremental, persist=args.apply)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
