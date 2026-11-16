import unittest
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

from production.pipeline.config import ConfigurationError, load_settings
from production.pipeline.database import DatabaseError
from production.pipeline.ingestion import SessionIngestor, canonical_tickers, inspect_state
from production.pipeline.migrate import apply_migrations
from production.pipeline.nse_calendar import CalendarUnavailable, IST, NSECalendar
from production.pipeline.provider import ProviderIncomplete, YahooProvider, normalize_history
from production.pipeline.repository import Bar, PersistenceError, Provenance, Repository
from support import MemoryDatabase, SqlError

BASE, MONDAY, TUESDAY = date(2026, 10, 23), date(2026, 10, 26), date(2026, 10, 27)
SOURCE = Provenance("yfinance", "1.2.0", "adjusted", datetime(2026, 10, 28, tzinfo=timezone.utc))


def bar(day, ticker):
    return Bar(day, 10, 12, 8, 11, 0 if ticker.startswith('^') else 100)


class CalendarProviderTests(unittest.TestCase):
    def test_ci_secret_source_is_explicit_and_target_isolated(self):
        host = 'ep-misty-shadow-azik30ky-pooler.c-3.ap-southeast-1.aws.neon.tech'
        uri = f'postgresql://test_role:SYNTHETIC_SECRET@{host}/neondb?sslmode=require'
        with patch.dict('os.environ', {'GITHUB_ACTIONS': 'true', 'LIQUIDITY_NEON_DATABASE_URL': uri,
                                       'DATABASE_URL': 'postgresql://unrelated-project.invalid/unrelated'}):
            settings = load_settings(from_environment=True)
            self.assertEqual(settings.hostname, host)
            self.assertNotIn('SYNTHETIC_SECRET', repr(settings))
        for value in ['', 'postgresql://test_role:SYNTHETIC_SECRET@unrelated-project.invalid/neondb']:
            with patch.dict('os.environ', {'GITHUB_ACTIONS': 'true', 'LIQUIDITY_NEON_DATABASE_URL': value}):
                with self.assertRaises(ConfigurationError) as caught:
                    load_settings(from_environment=True)
                self.assertNotIn('SYNTHETIC_SECRET', str(caught.exception))
        with patch.dict('os.environ', {'GITHUB_ACTIONS': 'false', 'LIQUIDITY_NEON_DATABASE_URL': uri}):
            with self.assertRaises(ConfigurationError):
                load_settings(from_environment=True)
    def test_special_sunday_and_holiday_are_authoritative(self):
        calendar = NSECalendar()
        sessions = calendar.sessions(BASE, date(2026, 11, 13))
        self.assertEqual(len(sessions), 16)
        self.assertIn(date(2026, 11, 8), sessions)
        self.assertNotIn(date(2026, 11, 10), sessions)
        self.assertNotIn(date(2026, 11, 7), sessions)
        self.assertFalse(calendar.is_session(date(2026, 1, 15)))

    def test_open_session_publication_guard_and_holiday(self):
        calendar = NSECalendar()
        self.assertEqual(calendar.latest(datetime(2026, 11, 16, 10, tzinfo=IST)), date(2026, 11, 13))
        now = datetime(2026, 11, 16, 15, 45, tzinfo=IST)
        self.assertEqual(calendar.latest(now), date(2026, 11, 16))
        self.assertEqual(calendar.latest(now, ready=True), date(2026, 11, 13))
        self.assertEqual(calendar.latest(datetime(2026, 11, 16, 16, tzinfo=IST), ready=True), date(2026, 11, 16))
        self.assertEqual(calendar.latest(datetime(2026, 11, 10, 16, tzinfo=IST)), date(2026, 11, 9))
        with self.assertRaises(CalendarUnavailable):
            calendar.is_session(date(2027, 1, 1))

    def test_nulls_duplicates_missing_dates_and_structural_zero(self):
        frame = pd.DataFrame({'Open': [10.0], 'High': [12.0], 'Low': [8.0], 'Close': [11.0], 'Volume': [0]}, index=pd.DatetimeIndex([MONDAY], tz='Asia/Kolkata'))
        self.assertEqual(normalize_history('^INDIAVIX', frame, [MONDAY])[MONDAY].volume, 0)
        self.assertEqual(len(normalize_history('^INDIAVIX', pd.concat([frame, frame]), [MONDAY])), 1)
        with self.assertRaises(ProviderIncomplete):
            normalize_history('TEST.NS', frame, [MONDAY, TUESDAY])
        missing = frame.copy()
        missing['Volume'] = float('nan')
        with self.assertRaises(ProviderIncomplete):
            normalize_history('TEST.NS', missing, [MONDAY])
        conflicting = pd.concat([frame, frame.assign(Close=10)])
        with self.assertRaises(ProviderIncomplete):
            normalize_history('TEST.NS', conflicting, [MONDAY])
        tiny_conflict = pd.concat([frame, frame.assign(Close=11.000000001)])
        with self.assertRaises(ProviderIncomplete):
            normalize_history('TEST.NS', tiny_conflict, [MONDAY])

    def test_network_and_incomplete_responses_have_bounded_retries(self):
        provider = YahooProvider.__new__(YahooProvider)
        provider.version = '1.2.0'
        provider.attempts, provider.delay, provider.backoff, provider.timeout = 3, 0, 2, 1
        provider.sleep = MagicMock()
        provider.calls, provider.retry_events = 0, []
        provider.yf = MagicMock()
        frame = pd.DataFrame({'Open': [10], 'High': [12], 'Low': [8], 'Close': [11], 'Volume': [100]}, index=pd.DatetimeIndex([MONDAY]))
        provider.yf.Ticker.return_value.history.side_effect = [TimeoutError(), pd.DataFrame(), frame]
        observations, _ = provider.fetch('TEST.NS', [MONDAY])
        self.assertEqual(len(observations), 1)
        self.assertEqual(provider.calls, 3)
        provider.yf.Ticker.return_value.history.side_effect = [frame, frame, frame]
        with self.assertRaises(ProviderIncomplete):
            provider.fetch('TEST.NS', [MONDAY, TUESDAY])
        self.assertEqual(provider.calls, 6)


class FakeProvider:
    retry_events = []

    def __init__(self, missing=None):
        self.missing = missing
        self.calls = 0

    def fetch(self, ticker, sessions):
        self.calls += 1
        if self.missing in sessions:
            raise ProviderIncomplete(ticker, 'MISSING_SESSIONS', missing=[self.missing])
        return {day: bar(day, ticker) for day in sessions}, SOURCE


class AtomicSessionTests(unittest.TestCase):
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
        self.db.execute('TRUNCATE liquidity.ingestion_runs,liquidity.instruments CASCADE')
        self.repo = Repository(self.db)
        self.repo.register_universe(market_tickers=('^NSEI', '^INDIAVIX'))
        self.tickers = canonical_tickers()
        self.repo.persist_session('seed', 'bootstrap', BASE,
                                  {ticker: bar(BASE, ticker) for ticker in self.tickers},
                                  {ticker: None for ticker in self.tickers},
                                  {ticker: SOURCE for ticker in self.tickers})

    def test_all_302_checkpoints_rollback_then_session_replay_is_idempotent(self):
        previous = inspect_state(self.db)
        observations = {ticker: bar(MONDAY, ticker) for ticker in self.tickers}
        provenance = {ticker: SOURCE for ticker in self.tickers}
        self.db.fail_on_write = self.db.write_count + 200
        with self.assertRaises(SqlError):
            self.repo.persist_session('monday', 'daily', MONDAY, observations, previous, provenance)
        self.assertEqual(set(inspect_state(self.db).values()), {BASE})
        self.db.execute('SELECT count(*) AS n FROM liquidity.market_eod')
        self.assertEqual(int(self.db.fetchone()['n']), 302)
        self.db.fail_on_write = None
        first = self.repo.persist_session('monday', 'daily', MONDAY, observations, previous, provenance)
        replay = self.repo.persist_session('monday', 'catchup', MONDAY, observations, previous, provenance)
        self.assertEqual(first['run_id'], replay['run_id'])
        self.assertTrue(replay['replayed'])
        self.db.execute('SELECT count(*) AS n FROM liquidity.market_eod')
        self.assertEqual(int(self.db.fetchone()['n']), 604)
        with self.assertRaises(PersistenceError):
            self.repo.persist_session('incomplete', 'daily', TUESDAY, dict(list(observations.items())[:301]), previous, provenance)

    def test_missing_special_session_blocks_all_writes(self):
        collector = SessionIngestor(self.db, FakeProvider(missing=date(2026, 11, 8)))
        report = collector.collect('catchup', now=datetime(2026, 11, 16, 10, tzinfo=IST), progress=lambda _: None)
        self.assertEqual(report['status'], 'BLOCKED')
        self.assertEqual(report['issue']['missing_sessions'], ['2026-11-08'])
        self.assertEqual(set(inspect_state(self.db).values()), {BASE})

    def test_catchup_reconciliation_daily_noop_and_recovery_after_lost_ack(self):
        collector = SessionIngestor(self.db, FakeProvider(), sleeper=lambda _: None)
        original = self.repo.persist_session
        lost = []
        def lose_ack(*args, **kwargs):
            result = original(*args, **kwargs)
            if not lost:
                lost.append(True)
                raise DatabaseError('PostgreSQL operation failed (CONNECTION_FAILURE)')
            return result
        collector.repository.persist_session = lose_ack
        now = datetime(2026, 10, 27, 16, tzinfo=IST)
        report = collector.collect('catchup', now=now, progress=lambda _: None)
        self.assertEqual(report['status'], 'SUCCESS')
        self.assertTrue(report['completed_sessions'][0]['replayed'])
        self.assertEqual(report['validation']['rows'], 604)
        self.assertEqual(collector.collect('daily', now=now)['status'], 'NOOP')
        self.assertEqual(set(inspect_state(self.db).values()), {TUESDAY})

    def test_daily_holiday_noop_uses_no_provider_and_incomplete_daily_stays_blocked(self):
        collector = SessionIngestor(self.db, FakeProvider(missing=MONDAY), sleeper=lambda _: None)
        holiday = collector.collect('daily', now=datetime(2026, 10, 24, 16, tzinfo=IST))
        self.assertEqual(holiday['status'], 'NOOP')
        self.assertEqual(collector.provider.calls, 0)
        report = collector.collect('daily', now=datetime(2026, 10, 26, 16, tzinfo=IST), progress=lambda _: None)
        self.assertEqual(report['status'], 'BLOCKED')
        self.assertEqual(set(inspect_state(self.db).values()), {BASE})
        collector.provider = FakeProvider()
        complete = collector.collect('daily', now=datetime(2026, 10, 26, 16, tzinfo=IST), progress=lambda _: None)
        self.assertEqual(complete['status'], 'SUCCESS')
        self.assertEqual(complete['validation']['rows'], 302)

    def test_completed_session_survives_later_failure_and_daily_recovers(self):
        collector = SessionIngestor(self.db, FakeProvider(), sleeper=lambda _: None)
        original = collector.repository.persist_session
        def fail_tuesday(key, mode, day, *args):
            if day == TUESDAY:
                raise DatabaseError('PostgreSQL operation failed (23514)')
            return original(key, mode, day, *args)
        collector.repository.persist_session = fail_tuesday
        now = datetime(2026, 10, 27, 16, tzinfo=IST)
        partial = collector.collect('catchup', now=now, progress=lambda _: None)
        self.assertEqual(partial['status'], 'PARTIAL')
        self.assertEqual(set(inspect_state(self.db).values()), {MONDAY})
        collector.repository.persist_session = original
        recovered = collector.collect('daily', now=now, progress=lambda _: None)
        self.assertEqual(recovered['status'], 'SUCCESS')
        self.assertEqual(set(inspect_state(self.db).values()), {TUESDAY})

    def test_provider_adjustment_revision_preserves_frozen_overlap(self):
        provider = FakeProvider()
        fetch = provider.fetch
        def revised(ticker, sessions):
            records, provenance = fetch(ticker, sessions)
            if ticker == self.tickers[0]:
                records[BASE] = Bar(BASE, 10, 12, 8, 10, 100)
            return records, provenance
        provider.fetch = revised
        collector = SessionIngestor(self.db, provider)
        report = collector.collect('daily', now=datetime(2026, 10, 26, 16, tzinfo=IST), progress=lambda _: None)
        self.assertEqual(report['status'], 'BLOCKED')
        self.assertEqual(report['issue']['code'], 'OVERLAP_REVISION')
        self.assertEqual(set(inspect_state(self.db).values()), {BASE})
