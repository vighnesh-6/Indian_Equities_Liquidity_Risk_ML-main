"""Execute catch-up through the latest completed and publication-ready NSE day."""

import argparse
import json
import os
from pathlib import Path

from .config import ROOT, ConfigurationError
from .database import Database, DatabaseError
from .ingestion import SessionIngestor
from .provider import YahooProvider
from .nse_calendar import CalendarUnavailable
from .repository import PersistenceError


def run_cli(mode):
    parser = argparse.ArgumentParser(description=f"Validated NSE {mode} collection through existing production APIs")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    # Reports may be in the approved production tree or ephemeral CI storage.
    import tempfile
    target = args.report.resolve()
    ci_temp = os.environ.get("RUNNER_TEMP") if os.environ.get("GITHUB_ACTIONS") == "true" else None
    allowed = [ROOT / "production/pipeline/phase2_results", Path(tempfile.gettempdir()).resolve()]
    if ci_temp:
        allowed.append(Path(ci_temp).resolve())
    if target.suffix != ".json" or not any(target.is_relative_to(directory) for directory in allowed):
        raise ValueError("Collector reports require production/pipeline or temporary storage")
    try:
        database = Database()
        provider = None if args.plan_only else YahooProvider()
        collector = SessionIngestor(database, provider)
        if args.plan_only:
            _, plan = collector.plan()
            report = {"status": "PLAN", "mode": mode, "plan": plan}
        else:
            report = collector.collect(mode, progress=lambda text: print(text, flush=True))
    except (ConfigurationError, CalendarUnavailable, DatabaseError, PersistenceError) as error:
        report = {"status": "FAILED", "mode": mode,
                  "issue": {"code": type(error).__name__, "message": str(error)}}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, default=str), flush=True)
    if report["status"] not in {"SUCCESS", "NOOP", "PLAN"}:
        raise SystemExit(2)


if __name__ == "__main__":
    run_cli("catchup")
