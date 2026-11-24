import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from unittest.mock import patch

from production.pipeline.database import DatabaseError
from production.pipeline.ingestion import SessionIngestor, inspect_state
from production.pipeline.migrate import apply_migrations
from production.pipeline.nse_calendar import IST
from production.pipeline.price_basis import load_history, stage_generation, reconcile_generation
from production.pipeline.repository import Bar, Repository
from support import MemoryDatabase, SqlError
from test_sessions import BASE, MONDAY, SOURCE, FakeProvider, bar


class FullProvider(FakeProvider):
    supports_full_history = True

    def fetch(self, ticker, sessions, *, historical_cutoff=None):
        values, source = super().fetch(ticker, sessions)
        if ticker == '360ONE.NS':
            values = {day: Bar(day, 5, 6, 4, 5.5, 100) for day in sessions}
        return values, source


class PriceBasisTests(unittest.TestCase):
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
        self.db.execute('TRUNCATE liquidity.price_basis_generations,liquidity.ingestion_runs,liquidity.instruments CASCADE')
        self.repo = Repository(self.db)
        self.repo.register_universe(market_tickers=('^NSEI','^INDIAVIX'))
        from production.pipeline.ingestion import canonical_tickers
        self.tickers = canonical_tickers()
        self.repo.persist_session('seed','bootstrap',BASE,
                                  {t:bar(BASE,t) for t in self.tickers}, {t:None for t in self.tickers},
                                  {t:SOURCE for t in self.tickers})

    def stage(self):
        history = load_history(self.db, BASE)
        quotes, sources = {}, {}
        provider = FullProvider()
        for ticker in self.tickers:
            quotes[ticker], sources[ticker] = provider.fetch(ticker,[BASE,MONDAY])
        generation, _ = stage_generation(self.db,history,quotes,sources,BASE,MONDAY)
        return generation, {t:q[MONDAY] for t,q in quotes.items()}, sources

    def test_actual_provider_revision_preserves_original_and_reconciles_every_historical_key(self):
        collector = SessionIngestor(self.db,FullProvider(),sleeper=lambda _:None)
        original = collector.repository.persist_session
        lost = []
        def lose_commit_ack(*args,**kwargs):
            result = original(*args,**kwargs)
            if not lost:
                lost.append(True)
                raise DatabaseError('PostgreSQL operation failed (CONNECTION_FAILURE)')
            return result
        collector.repository.persist_session = lose_commit_ack
        result = collector.collect('catchup',now=datetime(2026,10,26,16,tzinfo=IST),progress=lambda _:None)
        self.assertEqual(result['status'],'SUCCESS')
        self.assertTrue(result['completed_sessions'][0]['replayed'])
        self.assertEqual(result['historical_reconciliation']['result'],'PASS')
        self.assertEqual(result['price_basis']['actual_provider_revision_rows'],1)
        self.db.execute("SELECT close FROM liquidity.market_eod WHERE ticker='360ONE.NS' AND date=%s",(BASE,))
        self.assertEqual(self.db.fetchone()['close'],11)
        self.db.execute("SELECT close FROM liquidity.market_eod_consistent WHERE ticker='360ONE.NS' AND date=%s",(BASE,))
        self.assertEqual(self.db.fetchone()['close'],5.5)
        self.assertEqual(set(inspect_state(self.db).values()),{MONDAY})
        with self.assertRaises(SqlError):
            self.db.execute('UPDATE liquidity.price_basis_revisions SET close=5 WHERE generation_id=%s',(result['price_basis']['generation_id'],))

    def test_missing_historical_key_blocks_activation_and_all_checkpoint_advancement(self):
        provider = FullProvider(missing=BASE)
        result = SessionIngestor(self.db,provider).collect('catchup',now=datetime(2026,10,26,16,tzinfo=IST),progress=lambda _:None)
        self.assertEqual(result['status'],'BLOCKED')
        self.assertEqual(set(inspect_state(self.db).values()),{BASE})
        self.db.execute("SELECT count(*) AS n FROM liquidity.price_basis_generations WHERE state='ACTIVE'")
        self.assertEqual(int(self.db.fetchone()['n']),0)

    def test_missing_only_recovery_blocks_initial_historical_basis_build(self):
        with patch('production.pipeline.price_basis.fetch_full_history') as fetch, \
             patch('production.pipeline.price_basis.stage_generation') as stage:
            result = SessionIngestor(self.db, FullProvider()).collect(
                'daily', now=datetime(2026,10,26,16,tzinfo=IST),
                progress=lambda _:None, allow_basis_refresh=False)
        self.assertEqual(result['issue']['code'], 'PRICE_BASIS_REFRESH_REQUIRED')
        self.assertEqual(result['completed_sessions'], [])
        fetch.assert_not_called()
        stage.assert_not_called()
        self.assertEqual(set(inspect_state(self.db).values()), {BASE})
        self.db.execute('SELECT count(*) AS n FROM liquidity.price_basis_generations')
        self.assertEqual(int(self.db.fetchone()['n']), 0)

    def test_storage_preflight_failure_prevents_basis_staging_and_ingestion(self):
        from production.pipeline.repository import PersistenceError
        def reject_storage(history, quotes):
            self.assertEqual(set(history), set(quotes))
            raise PersistenceError('Insufficient capacity')
        with self.assertRaises(PersistenceError), \
             patch('production.pipeline.price_basis.stage_generation') as stage:
            SessionIngestor(self.db, FullProvider()).collect(
                'daily', now=datetime(2026,10,26,16,tzinfo=IST),
                progress=lambda _:None, before_basis_stage=reject_storage)
        stage.assert_not_called()
        self.assertEqual(set(inspect_state(self.db).values()), {BASE})

    def test_missing_only_recovery_blocks_revisions_of_an_active_basis(self):
        with patch('production.pipeline.price_basis.active_generation', return_value=1), \
             patch('production.pipeline.price_basis.fetch_full_history') as fetch:
            result = SessionIngestor(self.db, FullProvider()).collect(
                'daily', now=datetime(2026,10,26,16,tzinfo=IST),
                progress=lambda _:None, allow_basis_refresh=False)
        self.assertEqual(result['status'], 'BLOCKED')
        self.assertEqual(result['issue']['revised_checkpoint_instruments'], ['360ONE.NS'])
        self.assertEqual(result['issue']['revised_checkpoint_count'], 1)
        fetch.assert_not_called()
        self.db.execute('SELECT count(*) AS n FROM liquidity.market_eod')
        self.assertEqual(int(self.db.fetchone()['n']), 302)
        self.assertEqual(set(inspect_state(self.db).values()), {BASE})

    def test_missing_only_recovery_appends_when_active_overlap_matches(self):
        generation, observations, sources = self.stage()
        self.repo.persist_session('activate', 'catchup', MONDAY, observations,
                                  inspect_state(self.db), sources,
                                  price_basis_generation=generation)
        with patch('production.pipeline.price_basis.fetch_full_history') as fetch:
            result = SessionIngestor(self.db, FullProvider()).collect(
                'daily', now=datetime(2026,10,27,16,tzinfo=IST),
                progress=lambda _:None, allow_basis_refresh=False)
        self.assertEqual(result['status'], 'SUCCESS')
        self.assertEqual(result['validation']['rows'], 302)
        fetch.assert_not_called()
        self.db.execute('SELECT count(*) AS n FROM liquidity.market_eod')
        self.assertEqual(int(self.db.fetchone()['n']), 906)

    def test_direct_activation_without_complete_receipts_is_rejected(self):
        with self.assertRaises(SqlError):
            with self.db.transaction() as c:
                c.execute("INSERT INTO liquidity.price_basis_generations(request_key,base_date,target_date) VALUES (%s,%s,%s) RETURNING generation_id",('0'*64,BASE,MONDAY))
                generation = c.fetchone()['generation_id']
                c.execute("UPDATE liquidity.price_basis_generations SET state='ACTIVE',activated_at=now() WHERE generation_id=%s",(generation,))
        self.assertEqual(set(inspect_state(self.db).values()),{BASE})

    def test_wrong_provider_digest_rolls_back_activation_and_whole_session(self):
        generation,observations,sources = self.stage()
        self.db.execute("UPDATE liquidity.price_basis_receipts SET provider_sha256=%s WHERE generation_id=%s AND ticker='360ONE.NS'",('0'*64,generation))
        with self.assertRaises(SqlError):
            self.repo.persist_session('bad-digest','catchup',MONDAY,observations,inspect_state(self.db),sources,price_basis_generation=generation)
        self.assertEqual(set(inspect_state(self.db).values()),{BASE})
        self.db.execute('SELECT state FROM liquidity.price_basis_generations WHERE generation_id=%s',(generation,))
        self.assertEqual(self.db.fetchone()['state'],'STAGED')
        self.db.execute('SELECT count(*) AS n FROM liquidity.market_eod WHERE date=%s',(MONDAY,))
        self.assertEqual(int(self.db.fetchone()['n']),0)

    def test_concurrent_activation_replays_identical_complete_session(self):
        if not hasattr(self.db,'peer'):
            self.skipTest('Independent connections require the native PostgreSQL harness')
        generation,observations,sources = self.stage()
        previous = inspect_state(self.db)
        peer = self.db.peer()
        def persist(repository):
            return repository.persist_session('basis-race','catchup',MONDAY,observations,previous,sources,price_basis_generation=generation)
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(persist,self.repo)
            second = pool.submit(persist,Repository(peer))
            results = [first.result(30),second.result(30)]
        self.assertEqual(results[0]['run_id'],results[1]['run_id'])
        self.assertEqual(sorted(r['replayed'] for r in results),[False,True])
        self.assertEqual(reconcile_generation(self.db,generation)['result'],'PASS')
        self.assertEqual(set(inspect_state(self.db).values()),{MONDAY})

    def test_deferred_validation_cache_is_invalidated_by_later_direct_writes(self):
        from test_sessions import TUESDAY
        ticker = self.tickers[0]
        run = self.repo.start_run('dirty-cache','catchup',{ticker:[MONDAY,TUESDAY]})['run_id']
        self.repo.persist_batch(run,ticker,'both',[bar(MONDAY,ticker),bar(TUESDAY,ticker)],[MONDAY,TUESDAY],BASE,SOURCE)
        with self.assertRaises(SqlError):
            with self.db.transaction() as c:
                c.execute('SET CONSTRAINTS ALL IMMEDIATE')
                c.execute("UPDATE liquidity.ingestion_runs SET status='RUNNING' WHERE run_id=%s",(run,))
                c.execute('DELETE FROM liquidity.market_eod WHERE ticker=%s AND date=%s',(ticker,TUESDAY))
                c.execute("UPDATE liquidity.ingestion_runs SET status='SUCCESS',finished_at=now() WHERE run_id=%s",(run,))
        self.assertEqual(self.repo.get_checkpoint(ticker),TUESDAY)

    def test_unchanged_basis_retains_archive_with_explicit_overlap_and_action_evidence(self):
        class StableProvider(FakeProvider):
            supports_full_history = True
            def __init__(self):
                super().__init__()
                self.action_evidence = {}
            def fetch(self,ticker,sessions,*,historical_cutoff=None):
                result = super().fetch(ticker,sessions)
                self.action_evidence[ticker] = {'complete':True,'events':[]}
                return result
        result = SessionIngestor(self.db,StableProvider()).collect('catchup',now=datetime(2026,10,26,16,tzinfo=IST),progress=lambda _:None)
        self.assertEqual(result['status'],'SUCCESS')
        self.assertEqual(result['price_basis']['unchanged_archive_instruments'],302)
        self.assertEqual(result['price_basis']['full_provider_instruments'],0)
        self.assertEqual(result['price_basis']['provider_historical_sha256'],{})
        self.assertEqual(reconcile_generation(self.db,result['price_basis']['generation_id'])['result'],'PASS')

    def test_missing_basis_evidence_kind_is_rejected_by_postgresql(self):
        self.db.execute('INSERT INTO liquidity.price_basis_generations(request_key,base_date,target_date) VALUES (%s,%s,%s) RETURNING generation_id',('f'*64,BASE,MONDAY))
        generation = self.db.fetchone()['generation_id']
        with self.assertRaises(SqlError):
            self.db.execute("INSERT INTO liquidity.price_basis_receipts(generation_id,ticker,expected_rows,baseline_sha256,provider_sha256,source_version,retrieved_at,basis_evidence) VALUES (%s,%s,1,%s,%s,'test',%s,'{}'::jsonb)",
                            (generation,self.tickers[0],'0'*64,'0'*64,SOURCE.retrieved_at))
