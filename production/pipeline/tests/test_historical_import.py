import json
import unittest
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd

from production.pipeline.config import ROOT
from production.pipeline.historical_import import (
    ImportValidationError, Snapshot, file_sha256, import_snapshot,
    reconcile, row_bytes, source_records, validate_frame,
)
from production.pipeline.migrate import apply_migrations
from production.pipeline.repository import PersistenceError, Repository
from support import MemoryDatabase, SqlError


def frame(ticker="^NSEI"):
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-10-26", "2026-10-27"]),
        "ticker": [ticker, ticker], "open": np.array([10, 11], dtype="float32"),
        "high": np.array([12, 12], dtype="float32"),
        "low": np.array([8, 8], dtype="float32"),
        "close": np.array([10.1, 11.2], dtype="float32"),
        "volume": np.array([0, 100], dtype="int64"), "turnover": [np.nan, np.nan],
    })


class SourceValidationTests(unittest.TestCase):
    def test_nullable_turnover_has_a_distinct_lossless_binary_encoding(self):
        day = date(2026, 10, 26)
        self.assertNotEqual(row_bytes(day, (10, 12, 8, 10, 0, None)),
                            row_bytes(day, (10, 12, 8, 10, 0, 0.0)))

    def test_float32_prices_and_zero_volume_are_preserved_exactly(self):
        source = frame()
        validate_frame(source, {"^NSEI"})
        records = list(source_records(source))
        self.assertEqual(records[0][1][3], float(source.close.iloc[0]))
        self.assertEqual(records[0][1][4:], (0, None))
        self.assertNotEqual(row_bytes(*records[0]), row_bytes(*records[1]))

    def test_duplicate_null_and_malformed_values_are_rejected(self):
        duplicate = pd.concat([frame(), frame().iloc[:1]])
        null = frame()
        null.loc[0, "close"] = np.nan
        bounds = frame()
        bounds.loc[0, "high"] = 1
        fractions = frame()
        fractions["volume"] = [0.5, 100.0]
        time = frame()
        time.loc[0, "date"] += pd.Timedelta(hours=1)
        for source in [duplicate, null, bounds, fractions, time]:
            with self.subTest(source=source), self.assertRaises(ImportValidationError):
                validate_frame(source, {"^NSEI"})

    def test_source_dates_are_not_expanded_to_fill_missing_sessions(self):
        source = frame()
        source.loc[1, "date"] = pd.Timestamp("2026-10-30")
        validate_frame(source, {"^NSEI"})
        self.assertEqual([record[0] for record in source_records(source)], [date(2026, 10, 26), date(2026, 10, 30)])


class HistoricalPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = MemoryDatabase()
        apply_migrations(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def setUp(self):
        self.db.fail_on_write = None
        self.db.write_count = 0
        self.db.execute("TRUNCATE liquidity.ingestion_runs,liquidity.instruments CASCADE")
        self.repo = Repository(self.db)
        groups = {ticker: frame(ticker) for ticker in ["^INDIAVIX", "^NSEI"]}
        self.snapshot = Snapshot(ROOT, groups, [], {ticker: "a" * 64 for ticker in groups},
                                 {"universe_sha256": file_sha256(ROOT / "configs/universe.json")},
                                 "test-historical-restart", 2, "adjusted")

    def test_registration_can_select_exactly_the_requested_302_instruments(self):
        self.repo.register_universe(market_tickers=("^NSEI", "^INDIAVIX"))
        self.db.execute("SELECT count(*) AS n FROM liquidity.instruments")
        self.assertEqual(int(self.db.fetchone()["n"]), 302)
        self.db.execute("SELECT ticker FROM liquidity.instruments WHERE ticker='^NSEBANK'")
        self.assertIsNone(self.db.fetchone())
        for selected in [("^NSEI", "^NSEI"), ("UNKNOWN",)]:
            with self.assertRaises(PersistenceError):
                self.repo.register_universe(market_tickers=selected)

    def test_interrupted_import_resumes_with_committed_receipts(self):
        # Two rows commit for the first instrument. The next instrument rolls back.
        self.db.fail_on_write = 4
        with patch("production.pipeline.historical_import.reconcile", return_value={"discrepancies": []}):
            with self.assertRaisesRegex(SqlError, "INJECTED"):
                import_snapshot(self.db, self.snapshot, progress=lambda _: None)
            self.db.execute("SELECT count(*) AS n FROM liquidity.market_eod")
            self.assertEqual(int(self.db.fetchone()["n"]), 2)
            self.db.execute("SELECT status FROM liquidity.ingestion_runs")
            self.assertEqual(self.db.fetchone()["status"], "RUNNING")
            self.db.fail_on_write = None
            outcome = import_snapshot(self.db, self.snapshot, progress=lambda _: None)
            self.assertEqual(outcome["replayed_batches"], 1)
            self.assertEqual(outcome["committed_batches"], 1)
            second = import_snapshot(self.db, self.snapshot, progress=lambda _: None)
            self.assertEqual(second["replayed_batches"], 2)
            self.assertEqual(second["committed_batches"], 0)
        self.db.execute("SELECT count(*) AS n FROM liquidity.market_eod")
        self.assertEqual(int(self.db.fetchone()["n"]), 4)
        self.db.execute("SELECT count(*) AS n FROM liquidity.data_provenance")
        self.assertEqual(int(self.db.fetchone()["n"]), 2)
        self.db.execute("SELECT revision FROM liquidity.instrument_checkpoints")
        self.assertEqual([int(row["revision"]) for row in self.db.fetchall()], [1, 1])

    def test_independent_reconciliation_detects_an_exact_value_mismatch(self):
        universe = json.loads((ROOT / "configs/universe.json").read_text())
        tickers = [*universe["flat_ticker_list"], "^NSEI", "^INDIAVIX"]
        self.snapshot.groups = {ticker: frame(ticker) for ticker in tickers}
        self.snapshot.groups["^NSEI"]["turnover"] = [0.0, 101.000000001]
        self.snapshot.ticker_sources = {ticker: "a" * 64 for ticker in tickers}
        with patch("production.pipeline.historical_import.reconcile", return_value={"discrepancies": []}):
            import_snapshot(self.db, self.snapshot, progress=lambda _: None)
        matched = reconcile(self.db, self.snapshot)
        self.assertEqual(matched["discrepancies"], [])
        self.assertEqual(matched["rows"], 604)
        self.db.execute("UPDATE liquidity.market_eod SET close=close+0.000000001 WHERE ticker='^NSEI' AND date=%s",
                        (date(2026, 10, 26),))
        changed = reconcile(self.db, self.snapshot)
        self.assertEqual(changed["result"], "FAIL")
        self.assertIn("coverage_or_digest_mismatch", {item["issue"] for item in changed["discrepancies"]})
