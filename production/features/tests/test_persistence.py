from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import threading
import time
import unittest

import pandas as pd
import support

from production.pipeline.migrate import apply_migrations, migration_plan, verify_history, MigrationError
from production.pipeline.repository import Repository, Bar, Provenance
from production.features.audit import reconcile
from production.features.engine import input_hash
from production.features.repository import FeatureRepository
from production.features.state import initialize, update
from production.features.tests.native import NativeDatabase
from production.features.tests.test_engine import fixture


def seed(database, groups):
    repository = Repository(database)
    repository.register_universe(market_tickers=("^NSEI", "^INDIAVIX"))
    plan = {ticker: list(pd.to_datetime(group.date).dt.date) for ticker, group in groups.items()}
    run = repository.start_run("offline-market-source", "bootstrap", plan)
    provenance = Provenance("offline_fixture", "phase3", "adjusted", datetime.now(timezone.utc))
    for ticker, group in groups.items():
        bars = [Bar(row.date.date(), float(row.open), float(row.high), float(row.low),
                    float(row.close), int(row.volume)) for row in group.itertuples(index=False)]
        repository.persist_batch(run["run_id"], ticker, "offline", bars, plan[ticker], None, provenance)
    repository.complete_run(run["run_id"])


class LedgerTests(unittest.TestCase):
    def test_core_collector_accepts_optional_005_with_checksums_intact(self):
        core = [{"version": r["version"], "sha256": r["sha256"]} for r in migration_plan()]
        full = [{"version": r["version"], "sha256": r["sha256"]} for r in migration_plan(include_features=True)]
        verify_history(core)
        verify_history(full)
        with self.assertRaises(MigrationError):
            verify_history(core, require_features=True)
        damaged = [dict(r) for r in full]
        damaged[-1]["sha256"] = "0" * 64
        with self.assertRaises(MigrationError):
            verify_history(damaged)
        with self.assertRaises(MigrationError):
            verify_history(full[:2] + full[3:])


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.db = NativeDatabase()
        apply_migrations(self.db, include_features=True)
        self.groups = {t: fixture(t, rows=20) for t in ["RELIANCE.NS", "TCS.NS"]}
        seed(self.db, self.groups)
        self.benchmark = fixture("^NSEI", rows=20)
        self.vix = fixture("^INDIAVIX", rows=20)
        self.frames, self.states = {}, {}
        for ticker, group in self.groups.items():
            self.frames[ticker], self.states[ticker] = initialize(group, self.benchmark, self.vix,
                                                                 source_version="canonical-basis-0")
        self.frame = pd.concat(self.frames.values(), ignore_index=True)
        self.repo = FeatureRepository(self.db)
        self.version = self.repo.register_version("canonical-basis-0")
        self.checksum = input_hash(pd.concat(self.groups.values(), ignore_index=True))
        self.peers = []

    def tearDown(self):
        self.db.before_commit = None
        for peer in self.peers:
            peer.close()
        self.db.close()

    def persist(self, repository=None, *, frame=None, states=None, key="offline-features", publish=True):
        return (repository or self.repo).persist(self.frame if frame is None else frame,
                self.states if states is None else states, version=self.version, request_key=key,
                input_sha256=self.checksum, provenance={"test": "isolated"}, publish=publish)

    def test_exact_roundtrip_nulls_publication_and_duplicate_replay(self):
        first = self.persist()
        second = self.persist()
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertTrue(second["replayed"])
        self.assertEqual(reconcile(self.db, self.frame, self.version)["result"], "PASS")
        self.db.execute("SELECT count(*) AS n FROM liquidity.feature_model_inputs")
        self.assertEqual(self.db.fetchone()["n"], 40)

    def test_atomic_rollback_after_all_writes_and_safe_retry(self):
        def fail():
            raise RuntimeError("INJECTED_PRECOMMIT_FAILURE")
        self.db.before_commit = fail
        with self.assertRaises(RuntimeError):
            self.persist()
        self.db.before_commit = None
        for table in ["feature_values", "feature_runs", "feature_checkpoints", "feature_publications"]:
            self.db.execute("SELECT count(*) AS n FROM liquidity." + table)
            self.assertEqual(self.db.fetchone()["n"], 0)
        self.assertFalse(self.persist()["replayed"])

    def test_direct_immutability_and_backward_checkpoint_writes_rejected(self):
        self.persist()
        for query in ["UPDATE liquidity.feature_values SET daily_return=0",
                      "DELETE FROM liquidity.feature_values",
                      "UPDATE liquidity.feature_checkpoints SET last_date=last_date-1",
                      "DELETE FROM liquidity.feature_publications",
                      "UPDATE liquidity.feature_versions SET source_version='other'"]:
            with self.assertRaises(support.SqlError):
                self.db.execute(query)

    def test_incomplete_session_publication_rolls_back_every_write(self):
        ticker = "RELIANCE.NS"
        with self.assertRaises(support.SqlError):
            self.persist(frame=self.frames[ticker], states={ticker: self.states[ticker]}, publish=True)
        self.db.execute("SELECT count(*) AS n FROM liquidity.feature_values")
        self.assertEqual(self.db.fetchone()["n"], 0)

    def test_private_backfill_blocks_publish_until_other_instrument_arrives(self):
        ticker = "RELIANCE.NS"
        self.persist(frame=self.frames[ticker], states={ticker: self.states[ticker]}, publish=False)
        days = list(self.frames[ticker].date.dt.date)
        with self.assertRaises(support.SqlError):
            self.repo.publish(self.version, days)
        ticker = "TCS.NS"
        self.persist(frame=self.frames[ticker], states={ticker: self.states[ticker]}, key="second", publish=False)
        self.repo.publish(self.version, days)
        self.assertEqual(reconcile(self.db, self.frame, self.version)["result"], "PASS")

    def test_replay_conflicting_values_and_stale_state_rejected(self):
        self.persist()
        changed = self.frame.copy()
        changed.loc[1, "daily_return"] += 1
        with self.assertRaises(ValueError):
            self.persist(frame=changed)
        with self.assertRaises(support.SqlError):
            self.persist(key="different-key-same-observations")

    def test_direct_missing_historical_coverage_cannot_succeed(self):
        from uuid import uuid4
        with self.assertRaises(support.SqlError):
            with self.db.transaction() as cursor:
                cursor.execute("INSERT INTO liquidity.feature_runs(run_id,request_key,feature_version,range_start,range_end,"
                               "expected_tickers,expected_rows,input_sha256,output_sha256,provenance,status,finished_at) "
                               "VALUES (%s,'direct',%s,'2020-01-01','2020-01-28',%s,20,%s,%s,'{}','SUCCESS',now())",
                               (uuid4(), self.version, ["RELIANCE.NS"], "0"*64, "0"*64))

    def test_recursive_checkpoint_wrong_previous_close_rolls_back(self):
        self.states["RELIANCE.NS"].previous_close *= 2
        with self.assertRaises(support.SqlError):
            self.persist()
        self.db.execute("SELECT count(*) AS n FROM liquidity.feature_values")
        self.assertEqual(self.db.fetchone()["n"], 0)

    def test_truncated_recursive_context_cannot_advance_checkpoint(self):
        self.states["RELIANCE.NS"].context = []
        with self.assertRaises(support.SqlError):
            self.persist()
        self.db.execute("SELECT count(*) AS n FROM liquidity.feature_checkpoints")
        self.assertEqual(self.db.fetchone()["n"], 0)

    def test_direct_nonfinite_feature_insertion_rejected(self):
        from uuid import uuid4
        from production.features.contract import FLAGS
        run = uuid4()
        with self.assertRaises(support.SqlError):
            with self.db.transaction() as cursor:
                cursor.execute("INSERT INTO liquidity.feature_runs(run_id,request_key,feature_version,range_start,range_end,"
                               "expected_tickers,expected_rows,input_sha256,output_sha256,provenance,status) "
                               "VALUES (%s,'nan-direct',%s,'2020-01-01','2020-01-28',%s,20,%s,%s,'{}','RUNNING')",
                               (run, self.version, ["RELIANCE.NS"], "0"*64, "0"*64))
                columns = ["ticker", "date", "feature_version", "run_id", *FLAGS, "daily_return"]
                cursor.execute("INSERT INTO liquidity.feature_values(" + ",".join(columns) + ") VALUES ("
                               + ",".join(["%s"]*len(columns)) + ")",
                               ("RELIANCE.NS", "2020-01-01", self.version, run, *([False]*len(FLAGS)), float("nan")))

    def test_normal_entrypoint_rejects_same_count_wrong_universe(self):
        from production.features.run import process
        self.db.execute("INSERT INTO liquidity.instruments(ticker,instrument_type,name) VALUES ('BROKEN.NS','stock','Fixture')")
        self.db.execute("UPDATE liquidity.stocks SET ticker='BROKEN.NS' WHERE ticker='360ONE.NS'")
        with self.assertRaisesRegex(ValueError, "match the frozen 300-stock"):
            process(self.db, self.groups["RELIANCE.NS"].date.iloc[-1].date(), persist=False, progress=lambda _: None)

    def test_concurrent_conflicting_keys_cannot_duplicate_or_overwrite(self):
        peer = self.db.peer()
        self.peers.append(peer)
        with ThreadPoolExecutor(2) as pool:
            attempts = [pool.submit(self.persist, key="first-key"),
                        pool.submit(self.persist, FeatureRepository(peer), key="second-key")]
            successful, failed = 0, 0
            for future in attempts:
                try:
                    future.result(20)
                    successful += 1
                except support.SqlError:
                    failed += 1
        self.assertEqual((successful, failed), (1, 1))
        self.assertEqual(reconcile(self.db, self.frame, self.version)["result"], "PASS")

    def test_independent_connections_concurrent_identical_replay_once(self):
        peer = self.db.peer()
        self.peers.append(peer)
        held, release = threading.Event(), threading.Event()
        def hold():
            held.set()
            if not release.wait(12):
                raise RuntimeError("Test release timed out")
        self.db.before_commit = hold
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(self.persist)
            self.assertTrue(held.wait(8))
            second = pool.submit(self.persist, FeatureRepository(peer))
            try:
                deadline = time.monotonic() + 8
                waiting = False
                while time.monotonic() < deadline:
                    with self.db.server.connect(self.db.name) as connection:
                        waiting = connection.execute("SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE pid=ANY(%s) "
                                                     "AND wait_event_type='Lock') AS waiting", (list(peer.pids),)).fetchone()["waiting"]
                    if waiting:
                        break
                    time.sleep(0.05)
                self.assertTrue(waiting)
            finally:
                release.set()
            a, b = first.result(15), second.result(15)
            self.assertEqual(a["run_id"], b["run_id"])
            self.assertTrue(b["replayed"])
        self.db.before_commit = None
        self.assertTrue(self.db.pids.isdisjoint(peer.pids))
        self.db.execute("SELECT count(*) AS n FROM liquidity.feature_values")
        self.assertEqual(self.db.fetchone()["n"], 40)


class MigrationConcurrencyTests(unittest.TestCase):
    def test_two_migration_connections_install_once_and_replay(self):
        database = NativeDatabase()
        peer = database.peer()
        try:
            with ThreadPoolExecutor(2) as pool:
                a = pool.submit(apply_migrations, database, include_features=True)
                b = pool.submit(apply_migrations, peer, include_features=True)
                results = [a.result(30), b.result(30)]
            self.assertEqual(sorted(map(len, results)), [0, 6])
            database.execute("SELECT count(*) AS n FROM liquidity.schema_migrations")
            self.assertEqual(database.fetchone()["n"], 6)
        finally:
            peer.close()
            database.close()
