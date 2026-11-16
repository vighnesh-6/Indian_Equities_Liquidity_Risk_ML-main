"""Run Phase 3 validation on a disposable PostgreSQL 18 cluster."""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "production/pipeline/tests"))
import support
from production.features.tests.native import LocalPostgres, NativeDatabase
from production.features.audit import research_parity


class RecordedResults(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        self.successful.append(test.id())

    def startTestRun(self):
        self.successful = []


def execute(suite):
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResults).run(suite)
    return {"result": "PASS" if result.wasSuccessful() else "FAIL", "tests": result.testsRun,
            "passed": result.successful, "failures": [(test.id(), trace) for test, trace in result.failures],
            "errors": [(test.id(), trace) for test, trace in result.errors],
            "skipped": [(test.id(), why) for test, why in result.skipped]}


def load_prior_concurrency(path):
    """Retain prior lock tests and correct their two documented fixture counts."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    corrections = []
    names = {"test_different_instruments_in_same_run_serialize_safely",
             "test_independent_instruments_can_write_in_parallel"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            for call in ast.walk(node):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "assertEqual" and len(call.args) == 2
                        and isinstance(call.args[1], ast.Constant) and call.args[1].value == 5):
                    call.args[1].value = 4
                    corrections.append(node.name)
    if set(corrections) != names:
        raise ValueError("Legacy fixture structure differs from the reviewed two-date plans")
    module = types.ModuleType("prior_phase2_concurrency")
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module, {"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "corrected_tests": corrections, "expected_rows": 4,
                    "reason": "Two explicitly planned dates for each of two instruments. Original temporary file and all locking assertions are preserved."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postgres-bin", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--no-research", action="store_true")
    parser.add_argument("--no-pipeline", action="store_true")
    parser.add_argument("--storage", action="store_true")
    parser.add_argument("--legacy-concurrency-script", type=Path)
    args = parser.parse_args()
    report_path = args.report.resolve()
    if not report_path.is_relative_to(ROOT / "production/docs/phase3/validation"):
        raise ValueError("Reports must use the approved Phase 3 validation directory")
    report = {"python": sys.version.split()[0], "production_database_access": False,
              "incremental_tolerance": {"rtol": 1e-6, "atol": 1e-8, "null_masks": "exact", "rsi": "exact float32"}}
    started = time.monotonic()
    with LocalPostgres(args.postgres_bin) as server:
        print("Started isolated PostgreSQL 18 on localhost", flush=True)
        with server.connect() as connection:
            report["postgres"] = connection.execute("SELECT version() AS version").fetchone()["version"]
        loader = unittest.TestLoader()
        suite = loader.discover(str(ROOT / "production/features/tests"), pattern="test_*.py", top_level_dir=str(ROOT))
        report["phase3_tests"] = execute(suite)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not args.no_pipeline:
            support.MemoryDatabase = NativeDatabase
            report["pipeline_regression"] = execute(loader.discover(str(ROOT / "production/pipeline/tests")))
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.legacy_concurrency_script:
            module, corrections = load_prior_concurrency(args.legacy_concurrency_script)
            report["legacy_fixture_corrections"] = corrections
            module.NativeDatabase = NativeDatabase
            module.admin = server.connect
            report["pipeline_concurrency"] = execute(loader.loadTestsFromTestCase(module.ConcurrencyTests))
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.storage:
            from production.features.tests.storage import validate_storage
            report["storage"] = validate_storage()
        if not args.no_research:
            report["research_parity"] = research_parity()
    report["temporary_server_stopped"] = True
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["result"] = "PASS" if all(r.get("result") == "PASS" for r in report.values()
                                        if isinstance(r, dict) and "result" in r) else "FAIL"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("FINAL", json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}), flush=True)
    if report["result"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
