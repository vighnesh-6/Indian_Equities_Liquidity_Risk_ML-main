"""Import the three approved historical snapshots through Repository APIs.

Run without --apply for an offline source plan. --validate-only performs a fresh
read-only database/source comparison. An interrupted --apply resumes the same
deterministic run and replays committed batch receipts.
"""

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import ROOT
from .database import Database
from .migrate import migration_plan
from .repository import Bar, PersistenceError, Provenance, Repository, prepare_batch

SOURCE_PATHS = (
    "data/interim/stocks/ohlcv_clean.parquet",
    "data/interim/market/nifty50_clean.parquet",
    "data/interim/market/india_vix_clean.parquet",
)
MARKET_TICKERS = ("^NSEI", "^INDIAVIX")
VALUE_COLUMNS = ("open", "high", "low", "close", "volume", "turnover")


class ImportValidationError(PersistenceError):
    pass


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_bytes(day, values):
    """Lossless encoding of PostgreSQL DATE, float8, int8 and nullable turnover."""
    prices = values[:4]
    volume, turnover = values[4:]
    encoded = struct.pack(">qddddq", day.toordinal(), *prices, int(volume))
    return encoded + (b"N" if turnover is None else b"V" + struct.pack(">d", turnover))


def source_records(frame):
    for values in frame.loc[:, ["date", *VALUE_COLUMNS]].itertuples(index=False, name=None):
        day = values[0].date()
        turnover = None if pd.isna(values[6]) else float(values[6])
        yield day, (*[float(value) for value in values[1:5]], int(values[5]), turnover)


def source_summary(frame):
    digest = hashlib.sha256()
    for day, values in source_records(frame):
        digest.update(row_bytes(day, values))
    return {
        "rows": len(frame), "start": str(frame.date.iloc[0].date()),
        "end": str(frame.date.iloc[-1].date()), "values_sha256": digest.hexdigest(),
        "zero_volume_rows": int(frame.volume.eq(0).sum()),
    }


def validate_frame(frame, expected_tickers):
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ImportValidationError("Source requires nonempty ticker/date/OHLCV columns")
    if frame.loc[:, sorted(required)].isna().any().any():
        raise ImportValidationError("Required source values contain nulls")
    if not pd.api.types.is_datetime64_any_dtype(frame.date) or frame.date.dt.tz is not None:
        raise ImportValidationError("Source dates must be timezone-naive trading dates")
    if not frame.date.eq(frame.date.dt.normalize()).all():
        raise ImportValidationError("Source dates contain an intraday time")
    if set(frame.ticker) != set(expected_tickers):
        raise ImportValidationError("Source ticker set differs from the approved instrument mapping")
    if frame.duplicated(["ticker", "date"]).any():
        raise ImportValidationError("Duplicate source keys require explicit reconciliation")
    prices = frame.loc[:, ["open", "high", "low", "close"]]
    if not np.isfinite(prices.to_numpy()).all() or not prices.gt(0).all().all():
        raise ImportValidationError("Source prices must be finite and positive")
    if not (frame.high.ge(frame.low) & frame.open.between(frame.low, frame.high)
            & frame.close.between(frame.low, frame.high)).all():
        raise ImportValidationError("Source OHLC bounds violate the persistence contract")
    if not pd.api.types.is_integer_dtype(frame.volume) or not frame.volume.between(0, 2**63 - 1).all():
        raise ImportValidationError("Source volume must be a nonnegative int64")
    if "turnover" in frame:
        present = frame.turnover.dropna()
        if not np.isfinite(present).all() or present.lt(0).any():
            raise ImportValidationError("Source turnover must be finite, nonnegative or null")


@dataclass
class Snapshot:
    root: Path
    groups: dict
    sources: list
    ticker_sources: dict
    manifest: dict
    request_key: str
    batch_size: int
    adjustment_mode: str

    def plan(self):
        benchmark = set(self.groups["^NSEI"].date.dt.date)
        coverage = {}
        for ticker, frame in self.groups.items():
            days = set(frame.date.dt.date)
            first_day, last_day = min(days), max(days)
            relevant = {day for day in benchmark if first_day <= day <= last_day}
            coverage[ticker] = {
                **source_summary(frame),
                "benchmark_sessions_without_source_rows": len(relevant - days),
                "source_sessions_outside_benchmark": len(days - benchmark),
            }
        return {
            "request_key": self.request_key, "batch_size": self.batch_size,
            "instruments": len(self.groups), "rows": sum(len(frame) for frame in self.groups.values()),
            "batches": sum(math.ceil(len(frame) / self.batch_size) for frame in self.groups.values()),
            "sources": self.sources, "coverage": coverage,
            "mapping": "ticker/date/OHLCV -> liquidity.market_eod; absent turnover -> NULL",
            "session_policy": "Only source-observed dates, preserving instrument-specific gaps",
            "adjustment_mode": self.adjustment_mode,
        }

    def verify_source_hashes(self):
        if any(file_sha256(self.root / item["path"]) != item["sha256"] for item in self.sources):
            raise ImportValidationError("A source snapshot changed after inspection")
        if file_sha256(self.root / "configs/universe.json") != self.manifest["universe_sha256"]:
            raise ImportValidationError("The canonical universe changed after inspection")


def load_snapshot(root=ROOT, batch_size=4096):
    root = Path(root)
    if not 1 <= batch_size <= 4096:
        raise ImportValidationError("Batch size must be between 1 and 4096")
    universe = json.loads((root / "configs/universe.json").read_text(encoding="utf-8"))
    stocks = {entry["ticker"] for group in universe["tickers"].values() for entry in group}
    if len(stocks) != 300 or set(universe["flat_ticker_list"]) != stocks:
        raise ImportValidationError("The canonical universe must contain exactly 300 stocks")
    config = yaml.safe_load((root / "configs/pipeline_config.yaml").read_text(encoding="utf-8"))
    adjusted = config["data_acquisition"]["yfinance"]["auto_adjust"]
    if type(adjusted) is not bool:
        raise ImportValidationError("The approved adjustment convention is missing")
    sources, groups, ticker_sources = [], {}, {}
    for relative, expected in zip(SOURCE_PATHS, (stocks, {"^NSEI"}, {"^INDIAVIX"})):
        path = root / relative
        checksum = file_sha256(path)
        frame = pd.read_parquet(path)
        validate_frame(frame, expected)
        if file_sha256(path) != checksum:
            raise ImportValidationError("A Parquet snapshot changed while being read")
        sources.append({
            "path": relative, "sha256": checksum, "rows": len(frame),
            "start": str(frame.date.min().date()), "end": str(frame.date.max().date()),
            "nulls": {name: int(count) for name, count in frame.isna().sum().items()},
            "auxiliary_columns": [name for name in frame if name not in {"ticker", "date", *VALUE_COLUMNS}],
        })
        if "turnover" not in frame:
            frame = frame.assign(turnover=np.nan)
        for ticker, group in frame.groupby("ticker", sort=True):
            groups[ticker] = group.sort_values("date").reset_index(drop=True)
            ticker_sources[ticker] = checksum
    manifest = {
        "format": "historical-cleaned-parquet-v1", "batch_size": batch_size,
        "adjustment_mode": "adjusted" if adjusted else "unadjusted",
        "sources": [{"path": item["path"], "sha256": item["sha256"]} for item in sources],
        "universe_sha256": file_sha256(root / "configs/universe.json"),
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return Snapshot(root, groups, sources, ticker_sources, manifest,
                    "historical-parquet-v1:" + digest, batch_size, manifest["adjustment_mode"])


def database_coverage(cursor):
    """Hash the same lossless row_bytes encoding using PostgreSQL core functions.

    int8send/float8send use network byte order, matching struct's > encoding.
    Each aggregate covers one instrument in date order. DATE ordinals start at 1.
    This returns 302 small profiles while hashing every stored key and value.
    """
    cursor.execute(
        "SELECT ticker,count(*) AS rows,min(date)::text AS start,max(date)::text AS end,"
        "count(*) FILTER (WHERE volume=0) AS zero_volume_rows,"
        "count(*) FILTER (WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL) AS required_nulls,"
        "count(*) FILTER (WHERE turnover IS NULL) AS turnover_nulls,"
        "encode(sha256(string_agg("
        "int8send((date-DATE '0001-01-01')::bigint+1)||float8send(open)||float8send(high)||"
        "float8send(low)||float8send(close)||int8send(volume)||"
        "CASE WHEN turnover IS NULL THEN decode('4e','hex') ELSE decode('56','hex')||float8send(turnover) END,"
        "''::bytea ORDER BY date)),'hex') AS values_sha256 "
        "FROM liquidity.market_eod GROUP BY ticker ORDER BY ticker"
    )
    profiles = {}
    for row in cursor.fetchall():
        ticker = row.pop("ticker")
        for name in ["rows", "zero_volume_rows", "required_nulls", "turnover_nulls"]:
            row[name] = int(row[name])
        profiles[ticker] = row
    return profiles


def reconcile(database, snapshot, *, require_complete=True):
    """Independently compare binary database digests with re-readable source rows."""
    discrepancies = []
    actual = {}
    total = 0
    with database.transaction(read_only=True) as cursor:
        cursor.execute("SELECT version,sha256 FROM liquidity.schema_migrations ORDER BY version")
        ledger = cursor.fetchall()
        expected_ledger = [{"version": item["version"], "sha256": item["sha256"]} for item in migration_plan()]
        if ledger != expected_ledger:
            raise ImportValidationError("Production migration ledger differs from local SQL")
        actual = database_coverage(cursor)
        total = sum(state["rows"] for state in actual.values())
        if set(actual) - set(snapshot.groups):
            discrepancies.append({"issue": "unexpected_instruments", "tickers": sorted(set(actual) - set(snapshot.groups))})
        for ticker, frame in snapshot.groups.items():
            expected = source_summary(frame)
            found = actual.get(ticker, {})
            if found and not require_complete:
                # Restart receipts contain whole instrument batches. Every committed
                # prefix must match the corresponding immutable source prefix.
                prefix = frame.iloc[:found["rows"]]
                expected = source_summary(prefix)
            mismatch = any(found.get(key) != value for key, value in expected.items())
            if (require_complete or found) and mismatch:
                discrepancies.append({"ticker": ticker, "issue": "coverage_or_digest_mismatch", "expected": expected, "actual": found})
        cursor.execute("SELECT ticker,instrument_type FROM liquidity.instruments ORDER BY ticker")
        instruments = cursor.fetchall()
        expected_types = {ticker: ("volatility_index" if ticker == "^INDIAVIX" else "index" if ticker == "^NSEI" else "stock") for ticker in snapshot.groups}
        if any(expected_types.get(row["ticker"]) != row["instrument_type"] for row in instruments):
            discrepancies.append({"issue": "unexpected_dimension_mapping"})
        if require_complete and len(instruments) != 302:
            discrepancies.append({"issue": "instrument_dimension_count", "actual": len(instruments)})
        cursor.execute("SELECT count(*) AS n FROM liquidity.stocks")
        stock_count = int(cursor.fetchone()["n"])
        if require_complete and stock_count != 300:
            discrepancies.append({"issue": "stock_dimension_count", "actual": stock_count})
        cursor.execute("SELECT ticker,last_date,last_batch_id,revision FROM liquidity.instrument_checkpoints ORDER BY ticker")
        checkpoints = cursor.fetchall()
        if require_complete:
            expected_dates = {ticker: frame.date.iloc[-1].date() for ticker, frame in snapshot.groups.items()}
            if {row["ticker"]: row["last_date"] for row in checkpoints} != expected_dates:
                discrepancies.append({"issue": "checkpoint_coverage_mismatch"})
        cursor.execute("SELECT count(*) AS n FROM (SELECT ticker,date FROM liquidity.market_eod GROUP BY ticker,date HAVING count(*)>1) duplicates")
        duplicates = int(cursor.fetchone()["n"])
        if duplicates:
            discrepancies.append({"issue": "duplicate_database_keys", "count": duplicates})
        cursor.execute("SELECT run_id,request_key,status,started_at,finished_at FROM liquidity.ingestion_runs ORDER BY started_at")
        runs = cursor.fetchall()
        if any(row["request_key"] != snapshot.request_key for row in runs):
            discrepancies.append({"issue": "existing_run_belongs_to_a_different_snapshot"})
        run = next((row for row in runs if row["request_key"] == snapshot.request_key), None)
        if require_complete and (run is None or run["status"] != "SUCCESS"):
            discrepancies.append({"issue": "historical_run_not_successful"})
        cursor.execute("SELECT count(*) AS batches,coalesce(sum(row_count),0) AS rows FROM liquidity.ingestion_batches")
        receipts = cursor.fetchone()
        cursor.execute("SELECT count(*) AS n FROM liquidity.market_eod m JOIN liquidity.data_provenance p ON p.provenance_id=m.provenance_id "
                       "JOIN liquidity.ingestion_runs r ON r.run_id=p.run_id WHERE r.request_key<>%s OR p.source_name<>'cleaned_parquet' OR p.adjustment_mode<>%s",
                       (snapshot.request_key, snapshot.adjustment_mode))
        if cursor.fetchone()["n"]:
            discrepancies.append({"issue": "unexpected_data_provenance"})
        if require_complete and (int(receipts["rows"]) != total or len(checkpoints) != 302):
            discrepancies.append({"issue": "batch_or_checkpoint_totals"})
        if require_complete and run is not None:
            cursor.execute("SELECT ticker,planned_sessions FROM liquidity.ingestion_run_instruments WHERE run_id=%s", (run["run_id"],))
            observed_plans = {row["ticker"]: row["planned_sessions"] for row in cursor.fetchall()}
            expected_plans = {ticker: list(frame.date.dt.date) for ticker, frame in snapshot.groups.items()}
            if observed_plans != expected_plans:
                discrepancies.append({"issue": "stored_session_plan_differs_from_source"})
            cursor.execute(
                "SELECT b.ticker,b.batch_key,b.batch_id,b.row_count,b.range_start,b.range_end,b.payload_sha256,"
                "p.source_name,p.source_version,p.adjustment_mode,p.retrieved_at,p.payload_sha256 AS provenance_sha256 "
                "FROM liquidity.ingestion_batches b JOIN liquidity.data_provenance p ON p.provenance_id=b.provenance_id "
                "WHERE b.run_id=%s", (run["run_id"],),
            )
            stored_batches = {(row["ticker"], row["batch_key"]): row for row in cursor.fetchall()}
            expected_keys = set()
            checkpoint_map = {row["ticker"]: row for row in checkpoints}
            for ticker, frame in snapshot.groups.items():
                previous = None
                version = "sha256." + snapshot.ticker_sources[ticker]
                provenance = Provenance("cleaned_parquet", version, snapshot.adjustment_mode, run["started_at"])
                final_receipt = None
                for offset in range(0, len(frame), snapshot.batch_size):
                    batch = list(source_records(frame.iloc[offset:offset + snapshot.batch_size]))
                    days = [day for day, _ in batch]
                    _, _, checksum = prepare_batch([Bar(day, *values) for day, values in batch], days, previous, provenance)
                    key = f"parquet:{snapshot.ticker_sources[ticker]}:{days[0]}:{days[-1]}"
                    expected_keys.add((ticker, key))
                    stored = stored_batches.get((ticker, key))
                    expected = {"row_count": len(batch), "range_start": days[0], "range_end": days[-1],
                                "payload_sha256": checksum, "provenance_sha256": checksum,
                                "source_name": "cleaned_parquet", "source_version": version,
                                "adjustment_mode": snapshot.adjustment_mode, "retrieved_at": run["started_at"]}
                    if stored is None or any(stored[name] != value for name, value in expected.items()):
                        discrepancies.append({"ticker": ticker, "batch": key, "issue": "batch_receipt_or_provenance_mismatch"})
                    final_receipt = stored
                    previous = days[-1]
                checkpoint = checkpoint_map.get(ticker)
                if (checkpoint is None or final_receipt is None
                        or checkpoint["last_batch_id"] != final_receipt["batch_id"]
                        or checkpoint["revision"] != math.ceil(len(frame) / snapshot.batch_size)):
                    discrepancies.append({"ticker": ticker, "issue": "checkpoint_receipt_or_revision_mismatch"})
            if set(stored_batches) != expected_keys:
                discrepancies.append({"issue": "unexpected_or_missing_batch_receipts"})
    return {"result": "PASS" if not discrepancies else "FAIL", "method": "PostgreSQL binary SHA-256 per instrument versus independently decoded Parquet rows", "rows": total,
            "instruments": len(actual), "stock_dimension_rows": stock_count,
            "instrument_dimension_rows": len(instruments), "duplicate_keys": duplicates,
            "checkpoints": len(checkpoints), "receipts": receipts, "run": run,
            "coverage": actual, "discrepancies": discrepancies}


def import_snapshot(database, snapshot, progress=print):
    snapshot.verify_source_hashes()
    preflight = reconcile(database, snapshot, require_complete=False)
    if preflight["discrepancies"]:
        raise ImportValidationError("Pre-existing database values or metadata conflict with this snapshot")
    repository = Repository(database)
    progress("Registering canonical stocks and the two selected market instruments")
    repository.register_universe(market_tickers=MARKET_TICKERS)
    sessions = {ticker: list(frame.date.dt.date) for ticker, frame in snapshot.groups.items()}
    progress("Registering the source-observed historical session plan")
    run = repository.start_run(snapshot.request_key, "bootstrap", sessions)
    if run["status"] == "FAILED":
        raise ImportValidationError("The historical run is terminal FAILED and requires reviewed recovery")
    committed = replayed = 0
    for number, (ticker, frame) in enumerate(sorted(snapshot.groups.items()), 1):
        previous = None
        provenance = Provenance("cleaned_parquet", "sha256." + snapshot.ticker_sources[ticker],
                                snapshot.adjustment_mode, run["started_at"])
        for offset in range(0, len(frame), snapshot.batch_size):
            batch = list(source_records(frame.iloc[offset:offset + snapshot.batch_size]))
            days = [day for day, _ in batch]
            bars = [Bar(day, *values) for day, values in batch]
            key = f"parquet:{snapshot.ticker_sources[ticker]}:{days[0]}:{days[-1]}"
            receipt = repository.persist_batch(run["run_id"], ticker, key, bars, days, previous, provenance)
            replayed += int(receipt["replayed"])
            committed += int(not receipt["replayed"])
            previous = days[-1]
        progress(f"Instrument {number}/302 {ticker}: {len(frame)} source rows committed or replayed")
    snapshot.verify_source_hashes()
    repository.complete_run(run["run_id"])
    return {"run_id": str(run["run_id"]), "committed_batches": committed, "replayed_batches": replayed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and not args.report.resolve().is_relative_to(ROOT / "production/pipeline"):
        raise ImportValidationError("Import reports must be stored under production/pipeline")
    snapshot = load_snapshot(batch_size=args.batch_size)
    report = {"plan": snapshot.plan()}
    if args.apply or args.validate_only:
        database = Database()
        if args.apply:
            report["execution"] = import_snapshot(database, snapshot, progress=lambda text: print(text, flush=True))
        # Re-read the immutable source files for the independent comparison.
        fresh = load_snapshot(batch_size=args.batch_size)
        report["validation"] = reconcile(database, fresh)
        fresh.verify_source_hashes()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    summary = {"rows": report["plan"]["rows"], "instruments": report["plan"]["instruments"],
               "batches": report["plan"]["batches"], "request_key": snapshot.request_key}
    if "execution" in report:
        summary["execution"] = report["execution"]
    if "validation" in report:
        summary["validation"] = {key: value for key, value in report["validation"].items() if key != "coverage"}
    print(json.dumps(summary, default=str), flush=True)
    if "validation" in report and report["validation"]["result"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
