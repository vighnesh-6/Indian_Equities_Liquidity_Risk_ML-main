"""Transactional OHLCV persistence, provenance, idempotency and checkpoints."""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from numbers import Integral
from uuid import uuid4

from .config import ROOT


class PersistenceError(ValueError):
    pass


class CheckpointConflict(PersistenceError):
    pass


@dataclass(frozen=True)
class Bar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    turnover: float | None = None


@dataclass(frozen=True)
class Provenance:
    source_name: str
    source_version: str
    adjustment_mode: str
    retrieved_at: datetime


def prepare_batch(bars, sessions, prior_checkpoint, provenance):
    """Validate the caller's planned session coverage before opening a transaction."""
    bars = list(bars)
    sessions = list(sessions)
    if not bars or not sessions or any(type(day) is not date for day in sessions):
        raise PersistenceError("Nonempty bars and explicit trading-session dates are required")
    if sessions != sorted(set(sessions)):
        raise PersistenceError("Planned sessions must be unique and ordered")
    if prior_checkpoint is not None and (type(prior_checkpoint) is not date or sessions[0] <= prior_checkpoint):
        raise PersistenceError("Planned new sessions must follow the expected checkpoint")
    if len({bar.date for bar in bars}) != len(bars):
        raise PersistenceError("Duplicate dates within a batch are ambiguous")
    for bar in bars:
        prices = [bar.open, bar.high, bar.low, bar.close]
        if type(bar.date) is not date or not all(math.isfinite(value) and value > 0 for value in prices):
            raise PersistenceError("Bars require session dates and finite positive prices")
        if not bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high:
            raise PersistenceError("OHLC price bounds are inconsistent")
        if isinstance(bar.volume, bool) or not isinstance(bar.volume, Integral) or bar.volume < 0:
            raise PersistenceError("Volume must be a nonnegative integer")
        if bar.turnover is not None and (not math.isfinite(bar.turnover) or bar.turnover < 0):
            raise PersistenceError("Turnover must be finite and nonnegative")
    dates = {bar.date for bar in bars}
    if not set(sessions).issubset(dates) or max(dates) > sessions[-1]:
        raise PersistenceError("Batch does not cover the full planned session range")
    if {day for day in dates if prior_checkpoint is None or day > prior_checkpoint} != set(sessions):
        raise PersistenceError("Every new bar must belong to the planned session range")
    if provenance.adjustment_mode not in {"adjusted", "unadjusted"}:
        raise PersistenceError("Unknown price adjustment mode")
    if provenance.retrieved_at.tzinfo is None or provenance.retrieved_at.utcoffset() is None:
        raise PersistenceError("Provenance retrieval time must include a timezone")
    if not all(re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value) for value in [provenance.source_name, provenance.source_version]):
        raise PersistenceError("Source name and version require bounded identifiers")
    bars.sort(key=lambda bar: bar.date)
    payload = {"bars": [asdict(bar) for bar in bars], "sessions": sessions,
               "prior_checkpoint": prior_checkpoint, "provenance": asdict(provenance)}
    checksum = hashlib.sha256(json.dumps(payload, default=str, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return bars, sessions, checksum


class Repository:
    def __init__(self, database):
        self.database = database

    def register_universe(self, *, market_tickers=None):
        """Read an already-loaded frozen universe without changing its source file."""
        universe = json.loads((ROOT / "configs" / "universe.json").read_text(encoding="utf-8"))
        groups = universe["tickers"]
        stocks = [entry for group in groups.values() for entry in group]
        counts = [len(groups[key]) for key in ["large_cap_nifty_100", "mid_cap_nifty_midcap_150", "small_cap_nifty_smallcap_250_selected"]]
        if counts != [100, 150, 50] or len({stock["ticker"] for stock in stocks}) != 300:
            raise PersistenceError("Canonical universe must contain 300 unique stocks in approved segments")
        checksum = hashlib.sha256(json.dumps(universe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        metadata = universe["metadata"]
        indices = [metadata["benchmark"], *metadata["supplementary_market_indices"].values()]
        if market_tickers is not None:
            requested = list(market_tickers)
            available = {index["ticker"] for index in indices}
            if len(requested) != len(set(requested)) or not set(requested).issubset(available):
                raise PersistenceError("Market instrument selection must be unique and configured")
            indices = [index for index in indices if index["ticker"] in requested]
        with self.database.transaction() as cursor:
            for stock in stocks:
                cursor.execute(
                    "INSERT INTO liquidity.instruments(ticker, instrument_type, name) VALUES (%s, 'stock', %s) "
                    "ON CONFLICT (ticker) DO UPDATE SET name = EXCLUDED.name WHERE liquidity.instruments.instrument_type = 'stock' RETURNING ticker",
                    (stock["ticker"], stock["company_name"]),
                )
                if cursor.fetchone() is None:
                    raise PersistenceError("Existing instrument type conflicts with canonical universe")
                cursor.execute(
                    "INSERT INTO liquidity.stocks(ticker, company_name, cap_segment, sector, isin, universe_version, universe_sha256) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (ticker) DO UPDATE SET company_name=EXCLUDED.company_name, "
                    "cap_segment=EXCLUDED.cap_segment, sector=EXCLUDED.sector, universe_version=EXCLUDED.universe_version, "
                    "universe_sha256=EXCLUDED.universe_sha256 WHERE liquidity.stocks.isin=EXCLUDED.isin RETURNING ticker",
                    (stock["ticker"], stock["company_name"], stock["cap_segment"], stock.get("industry"), stock["isin"], metadata["version"], checksum),
                )
                if cursor.fetchone() is None:
                    raise PersistenceError("Existing ISIN conflicts with canonical universe")
            for index in indices:
                kind = "volatility_index" if index["ticker"] == "^INDIAVIX" else "index"
                cursor.execute(
                    "INSERT INTO liquidity.instruments(ticker, instrument_type, name) VALUES (%s,%s,%s) "
                    "ON CONFLICT (ticker) DO UPDATE SET name=EXCLUDED.name WHERE liquidity.instruments.instrument_type=EXCLUDED.instrument_type RETURNING ticker",
                    (index["ticker"], kind, index["name"]),
                )
                if cursor.fetchone() is None:
                    raise PersistenceError("Existing instrument type conflicts with benchmark")

    def get_checkpoint(self, ticker):
        with self.database.transaction(read_only=True) as cursor:
            cursor.execute("SELECT last_date FROM liquidity.instrument_checkpoints WHERE ticker=%s", (ticker,))
            row = cursor.fetchone()
            return row["last_date"] if row else None

    def start_run(self, request_key, mode, session_plan):
        instruments = sorted(session_plan)
        if not request_key or len(request_key) > 200 or not instruments or mode not in {"bootstrap", "catchup", "daily"}:
            raise PersistenceError("Invalid ingestion run plan")
        for sessions in session_plan.values():
            if not sessions or any(type(day) is not date for day in sessions) or list(sessions) != sorted(set(sessions)):
                raise PersistenceError("Each instrument requires a unique ordered trading-session plan")
        all_sessions = [day for sessions in session_plan.values() for day in sessions]
        range_start, range_end = min(all_sessions), max(all_sessions)
        plan_hash = hashlib.sha256(json.dumps(session_plan, default=str, sort_keys=True).encode()).hexdigest()
        run_id = uuid4()
        with self.database.transaction() as cursor:
            cursor.execute("SELECT ticker FROM liquidity.instruments WHERE ticker = ANY(%s)", (instruments,))
            if {row["ticker"] for row in cursor.fetchall()} != set(instruments):
                raise PersistenceError("Run plan contains an unregistered instrument")
            cursor.execute(
                "INSERT INTO liquidity.ingestion_runs(run_id, request_key, plan_sha256, mode, range_start, range_end, expected_instruments) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (request_key) DO NOTHING",
                (run_id, request_key, plan_hash, mode, range_start, range_end, instruments),
            )
            cursor.execute("SELECT * FROM liquidity.ingestion_runs WHERE request_key=%s", (request_key,))
            run = cursor.fetchone()
            if run["mode"] != mode or run["plan_sha256"] != plan_hash:
                raise PersistenceError("Request key was already used for a different run plan")
            for ticker in instruments:
                cursor.execute(
                    "INSERT INTO liquidity.ingestion_run_instruments(run_id,ticker,planned_sessions) VALUES (%s,%s,%s) "
                    "ON CONFLICT (run_id,ticker) DO NOTHING", (run["run_id"], ticker, list(session_plan[ticker])),
                )
            return run

    def persist_batch(self, run_id, ticker, batch_key, bars, sessions, prior_checkpoint, provenance):
        bars, sessions, checksum = prepare_batch(bars, sessions, prior_checkpoint, provenance)
        if not batch_key or len(batch_key) > 200:
            raise PersistenceError("A bounded batch idempotency key is required")
        with self.database.transaction() as cursor:
            cursor.execute("SELECT * FROM liquidity.ingestion_runs WHERE run_id=%s FOR UPDATE", (run_id,))
            run = cursor.fetchone()
            if run is None or ticker not in run["expected_instruments"]:
                raise PersistenceError("Batch instrument is outside its registered run plan")
            cursor.execute("SELECT * FROM liquidity.ingestion_batches WHERE run_id=%s AND ticker=%s AND batch_key=%s", (run_id, ticker, batch_key))
            existing = cursor.fetchone()
            if existing is not None:
                if existing["payload_sha256"] != checksum:
                    raise PersistenceError("Batch key was already used for different content")
                return {"batch_id": existing["batch_id"], "last_date": existing["range_end"], "replayed": True}
            if run["status"] != "RUNNING" or sessions[0] < run["range_start"] or sessions[-1] > run["range_end"]:
                raise PersistenceError("Batch is outside an active run's date range")
            cursor.execute("SELECT planned_sessions FROM liquidity.ingestion_run_instruments WHERE run_id=%s AND ticker=%s", (run_id, ticker))
            planned = cursor.fetchone()["planned_sessions"]
            remaining = [day for day in planned if prior_checkpoint is None or day > prior_checkpoint]
            if sessions != remaining[:len(sessions)] or any(bar.date not in planned for bar in bars):
                raise PersistenceError("Batch skips a planned session or includes an unplanned observation")
            cursor.execute("INSERT INTO liquidity.instrument_checkpoints(ticker) VALUES (%s) ON CONFLICT (ticker) DO NOTHING", (ticker,))
            cursor.execute("SELECT last_date FROM liquidity.instrument_checkpoints WHERE ticker=%s FOR UPDATE", (ticker,))
            if cursor.fetchone()["last_date"] != prior_checkpoint:
                raise CheckpointConflict("Instrument checkpoint changed after batch planning")
            cursor.execute(
                "SELECT s.day FROM unnest(%s::date[]) AS s(day) WHERE s.day <= %s "
                "AND NOT (s.day = ANY(%s::date[])) AND NOT EXISTS "
                "(SELECT 1 FROM liquidity.market_eod m WHERE m.ticker=%s AND m.date=s.day)",
                (planned, prior_checkpoint, [bar.date for bar in bars], ticker),
            )
            if cursor.fetchall():
                raise PersistenceError("Historical planned observations are missing before the checkpoint")
            cursor.execute(
                "SELECT DISTINCT p.adjustment_mode FROM liquidity.market_eod m JOIN liquidity.data_provenance p "
                "ON p.provenance_id=m.provenance_id WHERE m.ticker=%s", (ticker,),
            )
            if any(row["adjustment_mode"] != provenance.adjustment_mode for row in cursor.fetchall()):
                raise PersistenceError("Price adjustment mode differs from stored instrument history")
            provenance_id, batch_id = uuid4(), uuid4()
            cursor.execute(
                "INSERT INTO liquidity.data_provenance(provenance_id, run_id, source_name, source_version, adjustment_mode, retrieved_at, payload_sha256) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (provenance_id, run_id, provenance.source_name, provenance.source_version, provenance.adjustment_mode, provenance.retrieved_at, checksum),
            )
            cursor.executemany(
                    "INSERT INTO liquidity.market_eod(ticker,date,open,high,low,close,volume,turnover,provenance_id) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (ticker,date) DO UPDATE SET "
                    "open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume, "
                    "turnover=EXCLUDED.turnover, provenance_id=EXCLUDED.provenance_id, updated_at=now()",
                    [(ticker, bar.date, bar.open, bar.high, bar.low, bar.close, int(bar.volume), bar.turnover, provenance_id) for bar in bars],
            )
            cursor.execute(
                "INSERT INTO liquidity.ingestion_batches(batch_id,run_id,ticker,batch_key,payload_sha256,provenance_id,range_start,range_end,row_count) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (batch_id, run_id, ticker, batch_key, checksum, provenance_id, sessions[0], sessions[-1], len(bars)),
            )
            cursor.execute(
                "UPDATE liquidity.instrument_checkpoints SET last_date=%s, last_batch_id=%s, revision=revision+1, updated_at=now() WHERE ticker=%s",
                (sessions[-1], batch_id, ticker),
            )
            return {"batch_id": batch_id, "last_date": sessions[-1], "replayed": False}

    def complete_run(self, run_id):
        with self.database.transaction() as cursor:
            cursor.execute("SELECT * FROM liquidity.ingestion_runs WHERE run_id=%s FOR UPDATE", (run_id,))
            run = cursor.fetchone()
            if run is None or run["status"] == "FAILED":
                raise PersistenceError("Run cannot be completed")
            cursor.execute(
                "SELECT p.ticker FROM liquidity.ingestion_run_instruments p WHERE p.run_id=%s AND ("
                "EXISTS (SELECT 1 FROM unnest(p.planned_sessions) AS s(day) WHERE NOT EXISTS "
                "(SELECT 1 FROM liquidity.market_eod m WHERE m.ticker=p.ticker AND m.date=s.day)) "
                "OR NOT EXISTS (SELECT 1 FROM liquidity.ingestion_batches b WHERE b.run_id=p.run_id AND b.ticker=p.ticker "
                "AND b.range_end=p.planned_sessions[cardinality(p.planned_sessions)]))",
                (run_id,),
            )
            if cursor.fetchall():
                raise PersistenceError("Run still has instruments without complete planned coverage")
            cursor.execute("SELECT count(*) AS n FROM liquidity.ingestion_run_instruments WHERE run_id=%s", (run_id,))
            if cursor.fetchone()["n"] != len(run["expected_instruments"]):
                raise PersistenceError("Run plan does not match its expected instrument count")
            cursor.execute("UPDATE liquidity.ingestion_runs SET status='SUCCESS', finished_at=COALESCE(finished_at,now()) WHERE run_id=%s", (run_id,))

    def fail_run(self, run_id, error_code):
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", error_code):
            raise PersistenceError("Failure recording requires a sanitized error code")
        with self.database.transaction() as cursor:
            cursor.execute(
                "UPDATE liquidity.ingestion_runs SET status='FAILED', finished_at=now(), error_code=%s "
                "WHERE run_id=%s AND status='RUNNING' RETURNING run_id", (error_code, run_id),
            )
            if cursor.fetchone() is None:
                raise PersistenceError("Only an active run can be marked failed")
