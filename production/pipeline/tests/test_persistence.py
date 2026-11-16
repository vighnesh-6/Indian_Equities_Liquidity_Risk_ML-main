import json
import unittest
from datetime import date, datetime, timezone
from unittest.mock import patch
from uuid import uuid4

from production.pipeline.config import ROOT
from production.pipeline.migrate import MigrationError, apply_migrations, migration_plan
from production.pipeline.repository import Bar, CheckpointConflict, PersistenceError, Provenance, Repository
from support import MemoryDatabase, SqlError

D1, D2, D3 = date(2026, 10, 26), date(2026, 10, 27), date(2026, 10, 28)
SOURCE = Provenance("yfinance", "1.2.0", "adjusted", datetime(2026, 10, 28, tzinfo=timezone.utc))


def bar(day, close=10):
    return Bar(day, 10, 12, 8, close, 100)


class PersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = MemoryDatabase()
        apply_migrations(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def setUp(self):
        self.db.execute("TRUNCATE liquidity.ingestion_runs, liquidity.instruments CASCADE")
        self.db.fail_on_write = None
        self.db.write_count = 0
        self.db.execute("INSERT INTO liquidity.instruments VALUES ('TEST.NS','stock','Test',true), ('^INDIAVIX','volatility_index','VIX',true)")
        self.repo = Repository(self.db)

    def run_plan(self, days=(D1, D2), key="test", instruments=("TEST.NS",)):
        return self.repo.start_run(key, "catchup", {ticker: list(days) for ticker in instruments})["run_id"]

    def persist(self, run, days=(D1, D2), key="batch", previous=None, bars=None, provenance=SOURCE):
        return self.repo.persist_batch(run, "TEST.NS", key, bars or [bar(day) for day in days], list(days), previous, provenance)

    def count(self, table):
        self.db.execute(f"SELECT count(*) AS n FROM liquidity.{table}")
        return int(self.db.fetchone()["n"])

    def test_migrations_reapply_as_noop_and_detect_checksum_drift(self):
        self.assertEqual(apply_migrations(self.db), [])
        changed = migration_plan()
        changed[-1]["sha256"] = "0" * 64
        with patch("production.pipeline.migrate.migration_plan", return_value=changed):
            with self.assertRaises(MigrationError):
                apply_migrations(self.db)
        self.assertEqual(self.count("schema_migrations"), 5)

    def test_canonical_universe_registration_is_idempotent(self):
        self.db.execute("TRUNCATE liquidity.ingestion_runs, liquidity.instruments CASCADE")
        self.repo.register_universe()
        self.repo.register_universe()
        self.assertEqual(self.count("stocks"), 300)
        self.assertEqual(self.count("instruments"), 303)

    def test_success_persists_provenance_and_checkpoint_atomically(self):
        run = self.run_plan()
        receipt = self.persist(run)
        self.assertFalse(receipt["replayed"])
        self.assertEqual(self.count("market_eod"), 2)
        self.assertEqual(self.count("data_provenance"), 1)
        self.assertEqual(self.repo.get_checkpoint("TEST.NS"), D2)
        self.repo.complete_run(run)
        self.db.execute("SELECT status FROM liquidity.ingestion_runs WHERE run_id=%s", (run,))
        self.assertEqual(self.db.fetchone()["status"], "SUCCESS")

    def test_run_request_key_reuse_is_checked(self):
        run = self.run_plan()
        self.assertEqual(run, self.run_plan())
        with self.assertRaises(PersistenceError):
            self.run_plan((D1, D3))
        self.assertEqual(self.count("ingestion_runs"), 1)

    def test_unknown_instrument_cannot_start_run(self):
        with self.assertRaises(PersistenceError):
            self.run_plan(instruments=("UNKNOWN.NS",))
        self.assertEqual(self.count("ingestion_runs"), 0)

    def test_duplicate_batch_replay_has_no_extra_rows_or_checkpoint_increment(self):
        run = self.run_plan()
        first = self.persist(run)
        second = self.persist(run)
        self.assertEqual(first["batch_id"], second["batch_id"])
        self.assertTrue(second["replayed"])
        for table, count in [("market_eod", 2), ("data_provenance", 1), ("ingestion_batches", 1)]:
            self.assertEqual(self.count(table), count)
        self.db.execute("SELECT revision FROM liquidity.instrument_checkpoints WHERE ticker='TEST.NS'")
        self.assertEqual(int(self.db.fetchone()["revision"]), 1)

    def test_changed_content_cannot_reuse_batch_key(self):
        run = self.run_plan()
        self.persist(run)
        with self.assertRaises(PersistenceError):
            self.persist(run, bars=[bar(D1, 11), bar(D2)])
        self.assertEqual(self.count("market_eod"), 2)

    def test_duplicate_input_dates_rejected(self):
        run = self.run_plan()
        with self.assertRaises(PersistenceError):
            self.persist(run, bars=[bar(D1), bar(D1), bar(D2)])
        self.assertEqual(self.count("market_eod"), 0)

    def test_missing_planned_observation_blocks_checkpoint(self):
        run = self.run_plan()
        with self.assertRaises(PersistenceError):
            self.persist(run, bars=[bar(D1)])
        self.assertIsNone(self.repo.get_checkpoint("TEST.NS"))

    def test_skipped_session_in_batch_plan_rejected(self):
        run = self.run_plan((D1, D2, D3))
        with self.assertRaises(PersistenceError):
            self.persist(run, days=(D1, D3))
        self.assertIsNone(self.repo.get_checkpoint("TEST.NS"))

    def test_stale_planning_snapshot_rejected(self):
        first = self.run_plan((D1,))
        self.persist(first, days=(D1,))
        second = self.run_plan((D2,), key="second")
        with self.assertRaises(CheckpointConflict):
            self.persist(second, days=(D2,))
        self.assertEqual(self.repo.get_checkpoint("TEST.NS"), D1)

    def test_overlap_updates_existing_row_without_duplication(self):
        first = self.run_plan((D1,))
        self.persist(first, days=(D1,))
        second = self.run_plan((D1, D2), key="second")
        self.persist(second, days=(D2,), previous=D1, bars=[bar(D1, 11), bar(D2)])
        self.assertEqual(self.count("market_eod"), 2)
        self.db.execute("SELECT close FROM liquidity.market_eod WHERE ticker='TEST.NS' AND date=%s", (D1,))
        self.assertEqual(float(self.db.fetchone()["close"]), 11)
        self.assertEqual(self.repo.get_checkpoint("TEST.NS"), D2)
        self.persist(first, days=(D1,))
        self.assertEqual(self.repo.get_checkpoint("TEST.NS"), D2)

    def test_late_statement_failure_rolls_back_data_provenance_and_checkpoint(self):
        run = self.run_plan()
        self.db.fail_on_write = 2
        with self.assertRaises(SqlError):
            self.persist(run)
        for table in ["market_eod", "data_provenance", "ingestion_batches", "instrument_checkpoints"]:
            self.assertEqual(self.count(table), 0)
        self.db.fail_on_write = None
        self.persist(run)
        self.assertEqual(self.repo.get_checkpoint("TEST.NS"), D2)

    def test_successful_run_requires_all_instruments_to_finish(self):
        run = self.run_plan(instruments=("TEST.NS", "^INDIAVIX"))
        self.persist(run)
        with self.assertRaises(PersistenceError):
            self.repo.complete_run(run)
        self.repo.persist_batch(run, "^INDIAVIX", "index", [Bar(day, 10, 12, 8, 10, 0) for day in [D1, D2]], [D1, D2], None, SOURCE)
        self.repo.complete_run(run)

    def test_failed_run_preserved_and_new_batches_rejected(self):
        run = self.run_plan()
        self.repo.fail_run(run, "SOURCE_UNAVAILABLE")
        with self.assertRaises(PersistenceError):
            self.persist(run)
        with self.assertRaises(PersistenceError):
            self.repo.complete_run(run)
        with self.assertRaises(PersistenceError):
            self.repo.fail_run(run, "postgres://SYNTHETIC_SECRET")

    def test_adjustment_mode_change_rejected(self):
        first = self.run_plan((D1,))
        self.persist(first, days=(D1,))
        second = self.run_plan((D2,), key="second")
        source = Provenance("yfinance", "1.2.0", "unadjusted", SOURCE.retrieved_at)
        with self.assertRaises(PersistenceError):
            self.persist(second, days=(D2,), previous=D1, provenance=source)
        self.assertEqual(self.repo.get_checkpoint("TEST.NS"), D1)

    def test_foreign_keys_price_constraints_and_unique_keys(self):
        run = self.run_plan()
        self.persist(run)
        self.db.execute("SELECT provenance_id FROM liquidity.data_provenance")
        provenance_id = self.db.fetchone()["provenance_id"]
        sql = "INSERT INTO liquidity.market_eod(ticker,date,open,high,low,close,volume,provenance_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
        for values, code in [
            (("TEST.NS", D1, 10, 12, 8, 10, 100, provenance_id), "23505"),
            (("UNKNOWN", D3, 10, 12, 8, 10, 100, provenance_id), "23503"),
            (("TEST.NS", D3, -1, 12, 8, 10, 100, provenance_id), "23514"),
            (("TEST.NS", D3, 10, 12, 8, 10, -1, provenance_id), "23514"),
            (("TEST.NS", D3, 10, 12, 8, 15, 100, provenance_id), "23514"),
            (("TEST.NS", D3, "NaN", 12, 8, 10, 100, provenance_id), "23514"),
        ]:
            with self.subTest(values=values), self.assertRaisesRegex(SqlError, code):
                self.db.execute(sql, values)

    def test_feature_schema_matches_both_frozen_feature_contracts(self):
        expected = set()
        for name in ["hist_gradient_boosting", "isolation_forest"]:
            expected.update(json.loads((ROOT / f"models/phase8_batch5/{name}/feature_names.json").read_text())["features"])
        self.db.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='liquidity' AND table_name='daily_features'")
        actual = {row["column_name"] for row in self.db.fetchall()} - {"ticker", "date", "created_at"}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 37)

    def test_checkpoint_cannot_reference_another_instrument_batch(self):
        run = self.run_plan()
        batch = self.persist(run)["batch_id"]
        with self.assertRaisesRegex(SqlError, "23503"):
            self.db.execute("INSERT INTO liquidity.instrument_checkpoints(ticker,last_date,last_batch_id,revision) VALUES ('^INDIAVIX',%s,%s,1)", (D2, batch))

    def test_checkpoint_cannot_move_backwards(self):
        run = self.run_plan()
        self.persist(run)
        next_run = self.run_plan((D1,), key="regression")
        with self.assertRaises(PersistenceError):
            self.persist(next_run, days=(D1,), previous=D2)
        self.assertEqual(self.repo.get_checkpoint("TEST.NS"), D2)

    def test_checkpoint_requires_a_persisted_observation(self):
        run = self.run_plan()
        self.persist(run)
        with self.assertRaisesRegex(SqlError, "23503"):
            self.db.execute("UPDATE liquidity.instrument_checkpoints SET last_date=%s, revision=revision+1 WHERE ticker='TEST.NS'", (D3,))
        self.assertEqual(self.repo.get_checkpoint("TEST.NS"), D2)

    def test_chunked_run_cannot_complete_before_final_planned_session(self):
        run = self.run_plan((D1, D2, D3))
        self.persist(run, days=(D1,), key="first")
        with self.assertRaises(PersistenceError):
            self.repo.complete_run(run)
        self.persist(run, days=(D2, D3), key="remaining", previous=D1)
        self.repo.complete_run(run)
        self.assertEqual(self.repo.get_checkpoint("TEST.NS"), D3)

    def test_prediction_threshold_and_one_success_per_date_constraints(self):
        self.repo.register_universe()
        universe = json.loads((ROOT / "configs/universe.json").read_text())
        ticker = universe["flat_ticker_list"][0]
        run = self.repo.start_run("prediction", "daily", {ticker: [D1]})["run_id"]
        self.repo.persist_batch(run, ticker, "bar", [bar(D1)], [D1], None, SOURCE)
        self.db.execute("INSERT INTO liquidity.daily_features(ticker,date) VALUES (%s,%s)", (ticker, D1))
        pipeline_run = uuid4()
        self.repo.complete_run(run)
        self.db.execute("INSERT INTO liquidity.pipeline_runs(run_id,run_date,request_key,ingestion_run_id,triggered_by,status) VALUES (%s,%s,'score',%s,'manual','RUNNING')", (pipeline_run, D1, run))
        sql = "INSERT INTO liquidity.daily_predictions(ticker,date,pipeline_run_id,lrs_score,lrs_band,lrs_90p_exceeded,hre_prob,hre_flag,isolation_score,anomaly_score_scaled,is_anomaly,model_version) VALUES (%s,%s,%s,50,'provided_by_frozen_methodology',false,0.534,%s,-0.5,20,false,'CFG_05_HGBT')"
        with self.assertRaisesRegex(SqlError, "23514"):
            self.db.execute(sql, (ticker, D1, pipeline_run, 0))
        self.db.execute(sql, (ticker, D1, pipeline_run, 1))
        self.db.execute("UPDATE liquidity.pipeline_runs SET status='SUCCESS', finished_at=now(), duration_seconds=0, stocks_processed=1 WHERE run_id=%s", (pipeline_run,))
        with self.assertRaisesRegex(SqlError, "23505"):
            self.db.execute("INSERT INTO liquidity.pipeline_runs(run_id,run_date,request_key,ingestion_run_id,triggered_by,status,finished_at,duration_seconds,stocks_processed) VALUES (%s,%s,'second-score',%s,'manual','SUCCESS',now(),0,1)", (uuid4(), D1, run))


class MigrationAtomicityTests(unittest.TestCase):
    def test_failed_migration_rolls_back_all_schema_objects(self):
        database = MemoryDatabase()
        try:
            plan = migration_plan()
            plan[-1]["sql"] += "\nSELECT missing_function_for_rollback_test();"
            with patch("production.pipeline.migrate.migration_plan", return_value=plan):
                with self.assertRaises(SqlError):
                    apply_migrations(database)
            database.execute("SELECT count(*) AS n FROM information_schema.schemata WHERE schema_name='liquidity'")
            self.assertEqual(int(database.fetchone()["n"]), 0)
        finally:
            database.close()
