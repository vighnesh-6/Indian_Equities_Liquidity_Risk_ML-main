"""Regression tests against isolated PostgreSQL, including repository bypasses."""
import unittest
from uuid import uuid4

import test_persistence as fixtures
from production.pipeline.migrate import apply_migrations, migration_plan
from production.pipeline.repository import PersistenceError
from support import MemoryDatabase, SqlError

D1, D2, D3 = fixtures.D1, fixtures.D2, fixtures.D3


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = MemoryDatabase()
        apply_migrations(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    setUp = fixtures.PersistenceTests.setUp
    run_plan = fixtures.PersistenceTests.run_plan
    persist = fixtures.PersistenceTests.persist
    count = fixtures.PersistenceTests.count

    def finish_ingestion(self, run):
        self.db.execute("UPDATE liquidity.ingestion_runs SET status='SUCCESS',finished_at=now() WHERE run_id=%s", (run,))

    def pipeline(self, run, day=D1):
        target = uuid4()
        self.db.execute(
            "INSERT INTO liquidity.pipeline_runs(run_id,run_date,request_key,ingestion_run_id,triggered_by,status) "
            "VALUES (%s,%s,%s,%s,'test','RUNNING')", (target, day, str(target), run))
        return target

    def prediction(self, target, day=D1):
        self.db.execute(
            "INSERT INTO liquidity.daily_predictions(ticker,date,pipeline_run_id,lrs_score,lrs_band,lrs_90p_exceeded,"
            "hre_prob,hre_flag,isolation_score,anomaly_score_scaled,is_anomaly,model_version) "
            "VALUES ('TEST.NS',%s,%s,50,'frozen-upstream',false,0.534,1,-0.5,20,false,'CFG_05_HGBT')", (day, target))

    def scoring_fixture(self, complete=True):
        self.db.execute("INSERT INTO liquidity.stocks(ticker,company_name,cap_segment,isin,universe_version,universe_sha256) "
                        "VALUES ('TEST.NS','Test','large','TESTISIN','test',repeat('a',64))")
        run = self.run_plan()
        self.persist(run)
        if complete:
            self.repo.complete_run(run)
        self.db.execute("INSERT INTO liquidity.daily_features(ticker,date) VALUES ('TEST.NS',%s),('TEST.NS',%s)", (D1, D2))
        return run, self.pipeline(run)

    def finish_pipeline(self, target, count=1):
        self.db.execute("UPDATE liquidity.pipeline_runs SET status='SUCCESS',finished_at=now(),"
                        "duration_seconds=0,stocks_processed=%s WHERE run_id=%s", (count, target))

    def test_historical_hole_before_checkpoint_blocks_next_batch(self):
        first = self.run_plan((D2,))
        self.persist(first, days=(D2,))
        catchup = self.run_plan((D1, D2, D3), key="catchup")
        with self.assertRaisesRegex(PersistenceError, "Historical"):
            self.persist(catchup, days=(D3,), previous=D2)
        self.assertEqual(self.repo.get_checkpoint("TEST.NS"), D2)
        self.assertEqual(self.count("market_eod"), 1)

    def test_historical_hole_can_be_filled_alongside_advancement(self):
        first = self.run_plan((D2,))
        self.persist(first, days=(D2,))
        catchup = self.run_plan((D1, D2, D3), key="catchup")
        self.persist(catchup, days=(D3,), previous=D2, bars=[fixtures.bar(D1), fixtures.bar(D3)])
        self.repo.complete_run(catchup)
        self.assertEqual(self.count("market_eod"), 3)

    def test_repository_completion_checks_earlier_sessions(self):
        run = self.run_plan()
        self.persist(run)
        self.db.execute("DELETE FROM liquidity.market_eod WHERE date=%s", (D1,))
        with self.assertRaises(PersistenceError):
            self.repo.complete_run(run)

    def test_direct_success_cannot_hide_missing_earlier_session(self):
        run = self.run_plan()
        self.persist(run)
        self.db.execute("DELETE FROM liquidity.market_eod WHERE date=%s", (D1,))
        with self.assertRaisesRegex(SqlError, "23514"):
            self.finish_ingestion(run)
        self.db.execute("SELECT status FROM liquidity.ingestion_runs WHERE run_id=%s", (run,))
        self.assertEqual(self.db.fetchone()["status"], "RUNNING")

    def test_successful_coverage_cannot_be_deleted_or_rekeyed(self):
        run = self.run_plan()
        self.persist(run)
        self.repo.complete_run(run)
        for sql in ["DELETE FROM liquidity.market_eod WHERE date=%s",
                    "UPDATE liquidity.market_eod SET date=date-1 WHERE date=%s"]:
            with self.subTest(sql=sql), self.assertRaisesRegex(SqlError, "23514"):
                self.db.execute(sql, (D1,))

    def test_direct_checkpoint_backward_and_reset_rejected(self):
        run = self.run_plan()
        self.persist(run, days=(D1,), key="one")
        self.db.execute("SELECT last_batch_id FROM liquidity.instrument_checkpoints")
        old_batch = self.db.fetchone()["last_batch_id"]
        self.persist(run, days=(D2,), previous=D1, key="two")
        for sql, params in [
            ("UPDATE liquidity.instrument_checkpoints SET last_date=%s,last_batch_id=%s,revision=revision+1", (D1, old_batch)),
            ("UPDATE liquidity.instrument_checkpoints SET last_date=NULL,last_batch_id=NULL,revision=0", ()),
            ("DELETE FROM liquidity.instrument_checkpoints", ()),
            ("UPDATE liquidity.instrument_checkpoints SET revision=revision+1", ()),
        ]:
            with self.subTest(sql=sql), self.assertRaisesRegex(SqlError, "23514"):
                self.db.execute(sql, params)
        self.assertEqual(self.repo.get_checkpoint("TEST.NS"), D2)

    def test_checkpoint_date_must_equal_batch_endpoint(self):
        run = self.run_plan()
        self.persist(run)
        self.db.execute("INSERT INTO liquidity.market_eod(ticker,date,open,high,low,close,volume,provenance_id) "
                        "SELECT ticker,%s,open,high,low,close,volume,provenance_id FROM liquidity.market_eod WHERE date=%s", (D3, D2))
        with self.assertRaisesRegex(SqlError, "23503"):
            self.db.execute("UPDATE liquidity.instrument_checkpoints SET last_date=%s,revision=revision+1", (D3,))
        with self.assertRaisesRegex(SqlError, "23503"):
            self.db.execute("UPDATE liquidity.ingestion_batches SET range_end=%s", (D1,))

    def test_prediction_date_must_match_pipeline_date(self):
        _, target = self.scoring_fixture()
        with self.assertRaisesRegex(SqlError, "23503"):
            self.prediction(target, D2)
        self.prediction(target)
        with self.assertRaisesRegex(SqlError, "23503"):
            self.db.execute("UPDATE liquidity.pipeline_runs SET run_date=%s WHERE run_id=%s", (D2, target))

    def test_success_requires_finished_time_duration_count_and_no_error(self):
        _, target = self.scoring_fixture()
        self.prediction(target)
        for fields in ["status='SUCCESS'", "status='SUCCESS',finished_at=now()",
                       "status='SUCCESS',finished_at=now(),duration_seconds=0",
                       "status='SUCCESS',finished_at=now(),duration_seconds=0,stocks_processed=0",
                       "status='SUCCESS',finished_at=now(),duration_seconds=-1,stocks_processed=1",
                       "status='SUCCESS',finished_at=now(),duration_seconds=0,stocks_processed=301",
                       "status='SUCCESS',finished_at=now(),duration_seconds=0,stocks_processed=1,error_code='ERROR'",
                       "status='SUCCESS',finished_at=created_at-interval '1 second',duration_seconds=0,stocks_processed=1"]:
            with self.subTest(fields=fields), self.assertRaisesRegex(SqlError, "23514"):
                self.db.execute(f"UPDATE liquidity.pipeline_runs SET {fields} WHERE run_id=%s", (target,))
        self.finish_pipeline(target)

    def test_success_count_matches_predictions_and_ingestion_must_succeed(self):
        run, target = self.scoring_fixture(complete=False)
        self.prediction(target)
        with self.assertRaisesRegex(SqlError, "23514"):
            self.finish_pipeline(target)
        self.repo.complete_run(run)
        with self.assertRaisesRegex(SqlError, "23514"):
            self.finish_pipeline(target, count=2)
        self.finish_pipeline(target)
        with self.assertRaisesRegex(SqlError, "23514"):
            self.db.execute("DELETE FROM liquidity.daily_predictions WHERE pipeline_run_id=%s", (target,))
        with self.assertRaisesRegex(SqlError, "23514"):
            self.db.execute("UPDATE liquidity.pipeline_runs SET status='RUNNING',finished_at=NULL,"
                            "duration_seconds=NULL,stocks_processed=NULL WHERE run_id=%s", (target,))

    def test_failed_and_running_pipeline_state_consistency(self):
        _, target = self.scoring_fixture()
        for fields in ["status='FAILED'", "status='FAILED',finished_at=now()", "error_code='ERROR'", "finished_at=now()"]:
            with self.subTest(fields=fields), self.assertRaisesRegex(SqlError, "23514"):
                self.db.execute(f"UPDATE liquidity.pipeline_runs SET {fields} WHERE run_id=%s", (target,))
        self.db.execute("UPDATE liquidity.pipeline_runs SET status='FAILED',finished_at=now(),error_code='ERROR' WHERE run_id=%s", (target,))

    def test_atomic_success_and_prediction_insertion_is_supported(self):
        _, target = self.scoring_fixture()
        with self.db.transaction():
            self.finish_pipeline(target)
            self.prediction(target)

    def test_stock_type_checked_on_stock_insert_and_parent_type_change(self):
        with self.assertRaisesRegex(SqlError, "23503"):
            self.db.execute("INSERT INTO liquidity.stocks(ticker,company_name,cap_segment,isin,universe_version,universe_sha256) "
                            "VALUES ('^INDIAVIX','VIX','large','VIX','test',repeat('a',64))")
        self.scoring_fixture()
        with self.assertRaisesRegex(SqlError, "23503"):
            self.db.execute("UPDATE liquidity.instruments SET instrument_type='index' WHERE ticker='TEST.NS'")
        with self.assertRaisesRegex(SqlError, "23514"):
            self.db.execute("UPDATE liquidity.stocks SET instrument_type='index'")

    def test_invalid_expected_arrays_rejected_by_direct_sql(self):
        for instruments in [["TEST.NS", "TEST.NS"], [None], [""], []]:
            with self.subTest(instruments=instruments), self.assertRaisesRegex(SqlError, "23514"):
                self.db.execute("INSERT INTO liquidity.ingestion_runs(run_id,request_key,plan_sha256,mode,range_start,range_end,expected_instruments) "
                                "VALUES (%s,%s,repeat('a',64),'daily',%s,%s,%s::text[])", (uuid4(), str(uuid4()), D1, D2, instruments))

    def test_direct_plan_requires_exact_expected_count_and_membership(self):
        for instruments, ticker in [(["TEST.NS"], None), (["TEST.NS", "^INDIAVIX"], "TEST.NS"), (["TEST.NS"], "^INDIAVIX")]:
            with self.subTest(instruments=instruments, ticker=ticker), self.assertRaisesRegex(SqlError, "23514"):
                with self.db.transaction():
                    run = uuid4()
                    self.db.execute("INSERT INTO liquidity.ingestion_runs(run_id,request_key,plan_sha256,mode,range_start,range_end,expected_instruments) "
                                    "VALUES (%s,%s,repeat('a',64),'daily',%s,%s,%s::text[])", (run, str(run), D1, D2, instruments))
                    if ticker:
                        self.db.execute("INSERT INTO liquidity.ingestion_run_instruments VALUES (%s,%s,%s::date[])", (run, ticker, [D1]))

    def test_direct_plan_rejects_unordered_duplicate_null_and_out_of_range_sessions(self):
        for sessions in [[D2, D1], [D1, D1], [None], [], [D3]]:
            with self.subTest(sessions=sessions), self.assertRaisesRegex(SqlError, "23514"):
                with self.db.transaction():
                    run = uuid4()
                    self.db.execute("INSERT INTO liquidity.ingestion_runs(run_id,request_key,plan_sha256,mode,range_start,range_end,expected_instruments) "
                                    "VALUES (%s,%s,repeat('a',64),'daily',%s,%s,ARRAY['TEST.NS'])", (run, str(run), D1, D2))
                    self.db.execute("INSERT INTO liquidity.ingestion_run_instruments VALUES (%s,'TEST.NS',%s::date[])", (run, sessions))

    def test_direct_plan_rejects_unregistered_instrument(self):
        with self.assertRaisesRegex(SqlError, "23503"):
            with self.db.transaction():
                run = uuid4()
                self.db.execute("INSERT INTO liquidity.ingestion_runs(run_id,request_key,plan_sha256,mode,range_start,range_end,expected_instruments) "
                                "VALUES (%s,%s,repeat('a',64),'daily',%s,%s,ARRAY['UNKNOWN'])", (run, str(run), D1, D2))
                self.db.execute("INSERT INTO liquidity.ingestion_run_instruments VALUES (%s,'UNKNOWN',%s::date[])", (run, [D1]))

    def test_committed_plan_and_terminal_status_cannot_be_rewritten(self):
        run = self.run_plan()
        for sql in ["DELETE FROM liquidity.ingestion_run_instruments WHERE run_id=%s",
                    "UPDATE liquidity.ingestion_run_instruments SET planned_sessions=ARRAY[DATE '2026-10-26'] WHERE run_id=%s",
                    "UPDATE liquidity.ingestion_runs SET expected_instruments=ARRAY['^INDIAVIX'] WHERE run_id=%s"]:
            with self.subTest(sql=sql), self.assertRaisesRegex(SqlError, "23514"):
                self.db.execute(sql, (run,))
        self.persist(run)
        self.repo.complete_run(run)
        with self.assertRaisesRegex(SqlError, "23514"):
            self.db.execute("UPDATE liquidity.ingestion_runs SET status='RUNNING',finished_at=NULL WHERE run_id=%s", (run,))


class ForwardMigrationTests(unittest.TestCase):
    def test_inconsistent_legacy_coverage_aborts_forward_migration_atomically(self):
        from unittest.mock import patch
        from production.pipeline.repository import Repository
        db = MemoryDatabase()
        try:
            with patch('production.pipeline.migrate.migration_plan', return_value=migration_plan()[:2]):
                apply_migrations(db)
            db.execute("INSERT INTO liquidity.instruments VALUES ('TEST.NS','stock','Test',true)")
            repo = Repository(db)
            run = repo.start_run('legacy-hole', 'catchup', {'TEST.NS': [D1, D2]})['run_id']
            repo.persist_batch(run, 'TEST.NS', 'one', [fixtures.bar(D1), fixtures.bar(D2)], [D1, D2], None, fixtures.SOURCE)
            repo.complete_run(run)
            db.execute('DELETE FROM liquidity.market_eod WHERE date=%s', (D1,))
            with self.assertRaisesRegex(SqlError, '23514'):
                apply_migrations(db)
            db.execute('SELECT count(*) AS n FROM liquidity.schema_migrations')
            self.assertEqual(db.fetchone()['n'], 2)
            db.execute("SELECT count(*) AS n FROM information_schema.columns WHERE table_schema='liquidity' "
                       "AND table_name='pipeline_runs' AND column_name='finished_at'")
            self.assertEqual(db.fetchone()['n'], 0)
            self.assertEqual(repo.get_checkpoint('TEST.NS'), D2)
        finally:
            db.close()

    def test_legacy_success_without_completion_metrics_is_not_fabricated(self):
        from unittest.mock import patch
        from production.pipeline.repository import Repository
        db = MemoryDatabase()
        try:
            with patch('production.pipeline.migrate.migration_plan', return_value=migration_plan()[:2]):
                apply_migrations(db)
            db.execute("INSERT INTO liquidity.instruments VALUES ('TEST.NS','stock','Test',true)")
            run = Repository(db).start_run('legacy-score', 'daily', {'TEST.NS': [D1]})['run_id']
            db.execute("INSERT INTO liquidity.pipeline_runs(run_id,run_date,request_key,ingestion_run_id,triggered_by,status) "
                       "VALUES (%s,%s,'legacy-score',%s,'test','SUCCESS')", (uuid4(), D1, run))
            with self.assertRaisesRegex(SqlError, '23514'):
                apply_migrations(db)
            db.execute('SELECT count(*) AS n FROM liquidity.schema_migrations')
            self.assertEqual(db.fetchone()['n'], 2)
        finally:
            db.close()

    def test_upgrade_from_original_migrations_preserves_market_data(self):
        from unittest.mock import patch
        db = MemoryDatabase()
        try:
            original = migration_plan()[:2]
            with patch('production.pipeline.migrate.migration_plan', return_value=original):
                apply_migrations(db)
            db.execute("INSERT INTO liquidity.instruments VALUES ('TEST.NS','stock','Test',true)")
            from production.pipeline.repository import Repository
            repo = Repository(db)
            run = repo.start_run('upgrade', 'daily', {'TEST.NS': [D1]})['run_id']
            repo.persist_batch(run, 'TEST.NS', 'one', [fixtures.bar(D1)], [D1], None, fixtures.SOURCE)
            repo.complete_run(run)
            self.assertEqual(apply_migrations(db), ['002_integrity_guards', '003_price_basis'])
            self.assertEqual(repo.get_checkpoint('TEST.NS'), D1)
            db.execute('SELECT count(*) AS n FROM liquidity.market_eod')
            self.assertEqual(db.fetchone()['n'], 1)
            self.assertEqual(apply_migrations(db), [])
        finally:
            db.close()
