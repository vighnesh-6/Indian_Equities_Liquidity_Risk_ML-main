"""Normal production entry points exercised against 302 isolated instruments."""

import unittest

from production.pipeline.migrate import apply_migrations
from production.pipeline.ingestion import canonical_tickers
from production.features.run import process
from production.features.tests.native import NativeDatabase
from production.features.tests.test_engine import fixture
from production.features.tests.test_persistence import seed


class EntrypointTests(unittest.TestCase):
    def test_300_stock_backfill_atomic_daily_failure_recovery_and_noop(self):
        database = NativeDatabase()
        try:
            apply_migrations(database, include_features=True)
            groups = {ticker: fixture(ticker, rows=3) for ticker in canonical_tickers()}
            seed(database, groups)
            start = groups["RELIANCE.NS"].date.iloc[1].date()
            latest = groups["RELIANCE.NS"].date.iloc[2].date()
            historical = process(database, start, persist=True, progress=lambda _: None)
            self.assertEqual(historical["rows"], 600)
            self.assertEqual(historical["tickers"], 300)
            injected = []

            def fail_after_complete_session_is_prepared():
                database.execute("SELECT count(*) AS n FROM liquidity.feature_values WHERE date=%s", (latest,))
                if database.fetchone()["n"] == 300 and not injected:
                    injected.append(True)
                    raise RuntimeError("INJECTED_COMPLETE_FEATURE_SESSION_FAILURE")

            database.before_commit = fail_after_complete_session_is_prepared
            with self.assertRaises(RuntimeError):
                process(database, latest, incremental=True, persist=True, progress=lambda _: None)
            database.before_commit = None
            database.execute("SELECT count(*) AS n FROM liquidity.feature_values WHERE date=%s", (latest,))
            self.assertEqual(database.fetchone()["n"], 0)
            database.execute("SELECT count(*) AS n FROM liquidity.feature_checkpoints WHERE last_date=%s", (start,))
            self.assertEqual(database.fetchone()["n"], 300)
            recovered = process(database, latest, incremental=True, persist=True, progress=lambda _: None)
            self.assertEqual(recovered["rows"], 300)
            self.assertEqual(len(recovered["receipts"]), 1)
            database.execute("SELECT count(*) AS n FROM liquidity.feature_checkpoints WHERE last_date=%s", (latest,))
            self.assertEqual(database.fetchone()["n"], 300)
            repeated = process(database, latest, incremental=True, persist=True, progress=lambda _: None)
            self.assertEqual(repeated["status"], "NOOP")
            self.assertEqual(repeated["rows"], 0)
        finally:
            database.before_commit = None
            database.close()
