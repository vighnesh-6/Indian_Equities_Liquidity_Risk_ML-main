"""Independent read-only frozen-history and incremental-session reconciliation."""

import argparse
import json
from datetime import timedelta
from pathlib import Path

from .database import Database
from .config import ROOT
from .historical_import import database_coverage, load_snapshot, source_summary
from .ingestion import inspect_state
from .nse_calendar import NSECalendar


def audit_database(database, *, now=None):
    snapshot = load_snapshot()
    cutoff = max(frame.date.iloc[-1].date() for frame in snapshot.groups.values())
    checkpoints = inspect_state(database)
    current = next(iter(checkpoints.values()))
    calendar = NSECalendar()
    latest = calendar.latest(now, ready=True)
    issues = []
    with database.transaction(read_only=True) as cursor:
        cursor.execute("SELECT current_database() AS database,current_user AS username,current_setting('neon.project_id',true) AS project_id,"
                       "current_setting('neon.branch_id',true) AS branch_id,current_setting('neon.endpoint_id',true) AS endpoint_id,"
                       "current_setting('transaction_read_only') AS read_only")
        identity = cursor.fetchone()
        original = database_coverage(cursor, end=cutoff)
        if set(original) != set(snapshot.groups):
            issues.append("FROZEN_INSTRUMENT_SET")
        for ticker, frame in snapshot.groups.items():
            expected = source_summary(frame)
            if any(original.get(ticker, {}).get(name) != value for name, value in expected.items()):
                issues.append("FROZEN_HISTORY_MISMATCH:" + ticker)
        cursor.execute("SELECT date,count(*) AS rows FROM liquidity.market_eod WHERE date>%s GROUP BY date ORDER BY date", (cutoff,))
        observed_incremental = {row["date"]: int(row["rows"]) for row in cursor.fetchall()}
        excluded = calendar.exclusions(cutoff, max(current, latest))
        incremental = {day: count for day, count in observed_incremental.items() if day not in excluded}
        retained_excluded = {str(day): count for day, count in observed_incremental.items() if day in excluded}
        required = calendar.sessions(cutoff + timedelta(days=1), current)
        if incremental != {day: 302 for day in required}:
            issues.append("INCREMENTAL_SESSION_COVERAGE")
        cursor.execute("SELECT count(*) AS rows FROM liquidity.market_eod")
        total = int(cursor.fetchone()["rows"])
        cursor.execute("SELECT count(*) AS n FROM (SELECT ticker,date FROM liquidity.market_eod GROUP BY ticker,date HAVING count(*)>1) d")
        duplicates = int(cursor.fetchone()["n"])
        if duplicates:
            issues.append("DUPLICATE_KEYS")
        cursor.execute("SELECT count(*) AS n FROM pg_constraint WHERE connamespace='liquidity'::regnamespace AND NOT convalidated")
        invalid_constraints = int(cursor.fetchone()["n"])
        cursor.execute("SELECT count(*) AS n FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid WHERE c.relnamespace='liquidity'::regnamespace AND t.tgenabled<>'O'")
        disabled_triggers = int(cursor.fetchone()["n"])
        cursor.execute("SELECT count(*) AS n FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid WHERE c.relnamespace='liquidity'::regnamespace AND (NOT i.indisvalid OR NOT i.indisready)")
        invalid_indexes = int(cursor.fetchone()["n"])
        if invalid_constraints or disabled_triggers or invalid_indexes:
            issues.append("SCHEMA_INTEGRITY")
        cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='liquidity' AND table_name='daily_features'")
        actual_features = {row["column_name"] for row in cursor.fetchall()} - {"ticker", "date", "created_at"}
        frozen_features = set()
        for model in ["hist_gradient_boosting", "isolation_forest"]:
            frozen_features.update(json.loads((ROOT / f"models/phase8_batch5/{model}/feature_names.json").read_text())["features"])
        if actual_features != frozen_features or len(actual_features) != 37:
            issues.append("FROZEN_FEATURE_SCHEMA")
        cursor.execute("SELECT status,count(*) AS runs FROM liquidity.ingestion_runs GROUP BY status ORDER BY status")
        runs = cursor.fetchall()
    snapshot.verify_source_hashes()
    return {"result": "PASS" if not issues else "FAIL", "identity": identity,
            "frozen_rows": sum(row["rows"] for row in original.values()), "total_rows": total,
            "incremental_rows": sum(incremental.values()), "instruments": len(checkpoints),
            "excluded_sessions": list(map(str, excluded)), "excluded_session_rows_preserved": retained_excluded,
            "checkpoint": str(current), "latest_ready_session": str(latest), "up_to_date": current == latest,
            "pending_sessions": list(map(str, calendar.sessions(current + timedelta(days=1), latest))),
            "duplicate_keys": duplicates, "invalid_constraints": invalid_constraints,
            "disabled_triggers": disabled_triggers, "invalid_indexes": invalid_indexes,
            "frozen_feature_count": len(actual_features),
            "ingestion_runs": runs, "discrepancies": issues}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.suffix != ".json" or not args.report.resolve().is_relative_to(ROOT / "production/pipeline/phase2_results"):
        raise ValueError("Database audit reports require production/pipeline/phase2_results JSON paths")
    result = audit_database(Database())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, default=str))
    if result["result"] != "PASS":
        raise SystemExit(2)
