"""Shared session collector for catch-up and scheduled daily ingestion."""

import hashlib
import json
import time
from datetime import date, datetime, timedelta, timezone

from .config import ROOT
from .database import DatabaseError
from .historical_import import database_coverage, row_bytes
from .migrate import migration_plan
from .nse_calendar import NSECalendar
from .provider import ProviderIncomplete
from .repository import Bar, PersistenceError, Repository


def canonical_tickers():
    universe = json.loads((ROOT / "configs/universe.json").read_text(encoding="utf-8"))
    stocks = [item["ticker"] for group in universe["tickers"].values() for item in group]
    if len(stocks) != 300 or len(set(stocks)) != 300 or set(stocks) != set(universe["flat_ticker_list"]):
        raise PersistenceError("Canonical universe must contain exactly 300 stocks")
    return sorted([*stocks, "^NSEI", "^INDIAVIX"])


def inspect_state(database):
    expected = set(canonical_tickers())
    with database.transaction(read_only=True) as cursor:
        cursor.execute("SELECT version,sha256 FROM liquidity.schema_migrations ORDER BY version")
        if cursor.fetchall() != [{"version": row["version"], "sha256": row["sha256"]} for row in migration_plan()]:
            raise PersistenceError("Migration ledger differs from the approved local SQL")
        cursor.execute("SELECT ticker,instrument_type FROM liquidity.instruments")
        types = {row["ticker"]: row["instrument_type"] for row in cursor.fetchall()}
        expected_types = {ticker: "volatility_index" if ticker == "^INDIAVIX" else "index" if ticker == "^NSEI" else "stock" for ticker in expected}
        if types != expected_types:
            raise PersistenceError("Registered universe differs from the canonical 302 instruments")
        cursor.execute("SELECT ticker,last_date FROM liquidity.instrument_checkpoints")
        checkpoints = {row["ticker"]: row["last_date"] for row in cursor.fetchall()}
        if set(checkpoints) != expected or None in checkpoints.values() or len(set(checkpoints.values())) != 1:
            raise PersistenceError("All 302 committed checkpoints must agree before incremental collection")
    return checkpoints


def validate_window(database, quotes, sessions):
    """Independent persisted-value digests for a completely fetched date window."""
    expected = {}
    for ticker, records in quotes.items():
        digest = hashlib.sha256()
        for day in sessions:
            bar = records[day]
            digest.update(row_bytes(day, (bar.open, bar.high, bar.low, bar.close, bar.volume, bar.turnover)))
        expected[ticker] = digest.hexdigest()
    issues = []
    excluded = NSECalendar().exclusions(sessions[0], sessions[-1])
    with database.transaction(read_only=True) as cursor:
        profiles = database_coverage(cursor, start=sessions[0], end=sessions[-1], exclude_dates=excluded)
        if set(profiles) != set(expected):
            issues.append("INSTRUMENT_COVERAGE")
        for ticker, checksum in expected.items():
            actual = profiles.get(ticker, {})
            if actual.get("rows") != len(sessions) or actual.get("values_sha256") != checksum:
                issues.append("VALUES_OR_SESSIONS:" + ticker)
        cursor.execute("SELECT date,count(*) AS rows FROM liquidity.market_eod WHERE date>=%s AND date<=%s AND date <> ALL(%s::date[]) GROUP BY date ORDER BY date", (sessions[0], sessions[-1], excluded))
        dates = {row["date"]: int(row["rows"]) for row in cursor.fetchall()}
        if dates != {day: 302 for day in sessions}:
            issues.append("DAILY_COUNTS")
    return {"result": "PASS" if not issues else "FAIL", "rows": sum(row["rows"] for row in profiles.values()), "sessions": list(map(str, sessions)), "discrepancies": issues}


class SessionIngestor:
    def __init__(self, database, provider, *, calendar=None, sleeper=time.sleep):
        self.database = database
        self.provider = provider
        self.calendar = calendar or NSECalendar()
        self.repository = Repository(database)
        self.sleep = sleeper

    def plan(self, now=None):
        now = now or datetime.now(timezone.utc)
        checkpoints = inspect_state(self.database)
        base = next(iter(checkpoints.values()))
        completed = self.calendar.latest(now)
        ready = self.calendar.latest(now, ready=True)
        if base > completed:
            raise PersistenceError("Checkpoint is ahead of the latest completed NSE session")
        sessions = self.calendar.sessions(base + timedelta(days=1), ready)
        return checkpoints, {"base": str(base), "latest_completed": str(completed), "target": str(ready),
                             "sessions": list(map(str, sessions)), "expected_new_rows": 302 * len(sessions),
                             "instruments": 302, "publication_pending": completed > ready,
                             "excluded_sessions": list(map(str, self.calendar.exclusions(base, ready)))}

    def persist_with_recovery(self, key, mode, day, observations, previous, provenance):
        for attempt in range(3):
            try:
                return self.repository.persist_session(key, mode, day, observations, previous, provenance)
            except DatabaseError as error:
                transient = any(code in str(error) for code in ["CONNECTION_FAILURE", "080", "40001", "40P01", "55P03", "57014", "57P01"])
                if not transient or attempt == 2:
                    raise
                current = inspect_state(self.database)
                if set(current.values()) == {day}:
                    # A lost COMMIT acknowledgement is resolved by source-vs-DB
                    # verification, then the successful receipt is replayed.
                    verdict = validate_window(self.database, {ticker: {day: bar} for ticker, bar in observations.items()}, [day])
                    if verdict["result"] != "PASS":
                        raise PersistenceError("Uncertain commit differs from validated provider observations") from None
                    with self.database.transaction(read_only=True) as cursor:
                        cursor.execute("SELECT run_id,status FROM liquidity.ingestion_runs WHERE request_key=%s", (key,))
                        receipt = cursor.fetchone()
                    if receipt is None or receipt["status"] != "SUCCESS":
                        raise PersistenceError("Uncertain commit has no successful session receipt") from None
                    return {"run_id": receipt["run_id"], "replayed": True}
                if current != previous:
                    raise PersistenceError("Checkpoints changed during failed session recovery") from None
                self.sleep(2 ** (attempt + 1))

    def collect(self, mode, *, now=None, progress=print):
        previous, plan = self.plan(now)
        report = {"mode": mode, "plan": plan, "completed_sessions": [], "status": "RUNNING"}
        sessions = [date.fromisoformat(value) for value in plan["sessions"]]
        if not sessions:
            report["status"] = "WAITING_PUBLICATION" if plan["publication_pending"] else "NOOP"
            return report
        base = date.fromisoformat(plan["base"])
        window = self.calendar.sessions(base, sessions[-1])
        quotes, provenance = {}, {}
        try:
            for number, ticker in enumerate(sorted(previous), 1):
                quotes[ticker], provenance[ticker] = self.provider.fetch(ticker, window)
                progress(f"Validated provider history {number}/302 {ticker}")
        except ProviderIncomplete as error:
            report.update(status="BLOCKED", issue=error.report(), provider_retries=getattr(self.provider, "retry_events", []))
            return report
        # Overlap is checked after the entire provider window is complete. Existing
        # adjusted history is preserved when Yahoo revises an older observation.
        with self.database.transaction(read_only=True) as cursor:
            cursor.execute("SELECT ticker,date,open,high,low,close,volume,turnover FROM liquidity.market_eod WHERE date=%s", (base,))
            existing = {row["ticker"]: Bar(row["date"], row["open"], row["high"], row["low"], row["close"], row["volume"], row["turnover"]) for row in cursor.fetchall()}
        revisions = [ticker for ticker in previous if existing.get(ticker) != quotes[ticker][base]]
        if revisions:
            report.update(status="BLOCKED", issue={"code": "OVERLAP_REVISION", "tickers": revisions, "date": str(base)})
            return report
        universe_sha = hashlib.sha256(json.dumps(sorted(previous)).encode()).hexdigest()[:24]
        for day in sessions:
            observations = {ticker: quotes[ticker][day] for ticker in previous}
            key = f"nse-session-v1:{day}:{universe_sha}"
            try:
                receipt = self.persist_with_recovery(key, mode, day, observations, previous, provenance)
            except (DatabaseError, PersistenceError) as error:
                report.update(status="PARTIAL" if report["completed_sessions"] else "FAILED",
                              issue={"code": "SESSION_PERSISTENCE_FAILED", "type": type(error).__name__, "message": str(error), "session": str(day)})
                return report
            report["completed_sessions"].append({"date": str(day), "run_id": str(receipt["run_id"]), "replayed": receipt["replayed"]})
            previous = {ticker: day for ticker in previous}
            progress(f"Committed complete session {day} for 302 instruments")
        report["validation"] = validate_window(self.database, quotes, sessions)
        final = inspect_state(self.database)
        report["status"] = "SUCCESS" if report["validation"]["result"] == "PASS" and set(final.values()) == {sessions[-1]} else "FAILED"
        return report
