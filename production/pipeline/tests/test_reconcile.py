import unittest
from pathlib import Path
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from production.pipeline.reconcile import RESERVE_BYTES, initialize_features, read_committed_features, report_path, require_capacity, run, storage_estimate
from production.pipeline.config import ROOT
from production.pipeline.ingestion import canonical_tickers
from production.pipeline.repository import PersistenceError


class ReconciliationTests(unittest.TestCase):
    def test_ci_runner_temp_outside_os_temp_is_accepted_only_in_actions(self):
        runner = ROOT / 'uncreated-ci-temporary-directory'
        target = runner / 'liquidity-daily.json'
        with patch.dict('os.environ', {'GITHUB_ACTIONS': 'true', 'RUNNER_TEMP': str(runner)}):
            self.assertEqual(report_path(target), target.resolve())
        with patch.dict('os.environ', {'GITHUB_ACTIONS': 'false', 'RUNNER_TEMP': str(runner)}):
            with self.assertRaises(ValueError):
                report_path(target)

    def test_ci_report_guard_rejects_traversal_and_non_json_outputs(self):
        runner = ROOT / 'uncreated-ci-temporary-directory'
        with patch.dict('os.environ', {'GITHUB_ACTIONS': 'true', 'RUNNER_TEMP': str(runner)}):
            for target in [runner / '..' / 'outside.json', runner / 'report.txt']:
                with self.subTest(target=target), self.assertRaises(ValueError):
                    report_path(target)

    def test_paged_verification_preserves_every_value_flag_and_null(self):
        from production.features.contract import FEATURES, FLAGS
        days = pd.date_range('2020-01-01', periods=1101).date
        rows = [{'ticker': 'ONE.NS', 'date': day,
                 **{name: True for name in FLAGS},
                 **{name: (None if n % 7 == 0 else n * 0.125) for name in FEATURES}}
                for n, day in enumerate(days)]
        database = MagicMock()
        cursor = database.transaction.return_value.__enter__.return_value
        cursor.fetchall.side_effect = [rows[:500], rows[500:1000], rows[1000:]]
        actual = read_committed_features(database, 'version', 'ONE.NS', days[0], days[-1])
        pd.testing.assert_frame_equal(actual, pd.DataFrame(rows))
        calls = cursor.execute.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertTrue(all('ORDER BY date LIMIT 500' in call.args[0] for call in calls))
        self.assertEqual(calls[1].args[1][2], days[499])
        self.assertEqual(calls[2].args[1][2], days[999])
        self.assertTrue(all(call.kwargs == {'read_only': True} for call in database.transaction.call_args_list))

    def test_storage_estimate_retains_current_allocation_and_budgets_both_versions(self):
        sizes = {'feature_values': 4000, 'intelligence_observations': 2000,
                 'feature_checkpoints': 300 * 40000, 'price_basis_revisions': 2400,
                 'market_eod': 2000}
        counts = {'feature_values': 10, 'intelligence_observations': 10,
                  'feature_checkpoints': 300, 'price_basis_revisions': 10, 'market_eod': 10}
        expected = 1000 + 20 * 400 + 20 * 200 + 300 * 40000 + 2 * 240 + 5 * 200 + 5 * 1024 * 1024
        self.assertEqual(storage_estimate(1000, sizes, counts, 20, 5, 2), expected)

    def test_unknown_capacity_stops_without_writes(self):
        database = MagicMock()
        cursor = database.transaction.return_value.__enter__.return_value
        cursor.fetchone.side_effect = [{'bytes': 100, 'capacity': None}] + [{'bytes': 100, 'rows': 1}] * 5
        with self.assertRaises(PersistenceError):
            require_capacity(database, 300)
        database.transaction.assert_called_once_with(read_only=True)

    def test_capacity_requires_reserved_headroom_without_deleting_anything(self):
        database = MagicMock()
        cursor = database.transaction.return_value.__enter__.return_value
        cursor.fetchone.side_effect = [{'bytes': 100, 'capacity': RESERVE_BYTES}] + [{'bytes': 100, 'rows': 1}] * 5
        with self.assertRaises(PersistenceError):
            require_capacity(database, 300)
        self.assertTrue(all(call.args[0].startswith('SELECT') for call in cursor.execute.call_args_list))

    def test_provider_block_does_not_start_feature_processing(self):
        with patch('production.pipeline.reconcile.SessionIngestor') as collector, \
             patch('production.pipeline.reconcile.initialize_features') as features:
            collector.return_value.collect.return_value = {'status': 'BLOCKED', 'completed_sessions': []}
            result = run(MagicMock(), MagicMock(), progress=lambda _: None)
        self.assertEqual(result['status'], 'BLOCKED')
        features.assert_not_called()
        self.assertTrue(collector.return_value.collect.call_args.kwargs['allow_basis_refresh'])
        self.assertTrue(callable(collector.return_value.collect.call_args.kwargs['before_basis_stage']))

    def test_capacity_reconnect_does_not_repeat_provider_collection(self):
        from production.pipeline.database import DatabaseError
        from production.pipeline.repository import Bar
        base, following = date(2026,11,16), date(2026,11,17)
        history = {'ONE.NS': {base: Bar(base, 10, 12, 8, 11, 100)}}
        quotes = {'ONE.NS': {**history['ONE.NS'], following: Bar(following, 11, 13, 9, 12, 100)}}
        def collect(*args, **kwargs):
            kwargs['before_basis_stage'](history, quotes)
            return {'status': 'BLOCKED', 'completed_sessions': []}
        with patch('production.pipeline.reconcile.SessionIngestor') as collector, \
             patch('production.pipeline.reconcile.require_capacity', side_effect=[
                 DatabaseError('PostgreSQL operation failed (CONNECTION_FAILURE)'),
                 {'current_bytes': 100}]) as capacity, \
             patch('production.intelligence.database.time.sleep'):
            collector.return_value.collect.side_effect = collect
            result = run(MagicMock(), MagicMock(), progress=lambda _: None)
        self.assertEqual(capacity.call_count, 2)
        collector.return_value.collect.assert_called_once()
        self.assertEqual(result['storage']['current_bytes'], 100)

    def test_existing_complete_feature_cohort_is_reused_without_calculation(self):
        database = MagicMock()
        cursor = database.transaction.return_value.__enter__.return_value
        cursor.fetchone.return_value = {'generation': 3}
        cursor.fetchall.return_value = [{'ticker': ticker, 'last_date': date(2026, 11, 23)}
                                       for ticker in canonical_tickers() if not ticker.startswith('^')]
        with patch('production.pipeline.reconcile.initialize') as calculate, \
             patch('production.pipeline.reconcile.read_history') as read:
            result = initialize_features(database, date(2026, 11, 23))
        self.assertEqual(result['status'], 'NOOP')
        calculate.assert_not_called()
        read.assert_not_called()

    def test_success_preserves_ingestion_summary_and_initializes_matching_target(self):
        through = date(2026, 11, 23)
        ingestion = {'status': 'SUCCESS', 'plan': {'target': str(through)},
                     'completed_sessions': [{'date': str(through)}]}
        with patch('production.pipeline.reconcile.SessionIngestor') as collector, \
             patch('production.pipeline.reconcile.initialize_features', return_value={'status': 'SUCCESS'}) as features:
            collector.return_value.collect.return_value = ingestion
            database = MagicMock()
            result = run(database, MagicMock(), progress=lambda _: None)
        self.assertEqual(result['completed_sessions'], ingestion['completed_sessions'])
        features.assert_called_once_with(database, through, progress=unittest.mock.ANY)

    def test_resume_calculates_only_missing_ticker_with_real_frozen_engine(self):
        dates = pd.bdate_range('2026-01-01', periods=120)
        through = dates[-1].date()
        tickers = [t for t in canonical_tickers() if not t.startswith('^')]
        missing = tickers[0]
        def history(database, ticker, end):
            close = 100 + np.arange(len(dates)) * 0.2 + np.sin(np.arange(len(dates)))
            return pd.DataFrame({'date': dates, 'ticker': ticker, 'open': close - 0.1,
                                 'high': close + 1, 'low': close - 1, 'close': close,
                                 'volume': 100000 + np.arange(len(dates)) * 100})
        database = MagicMock()
        cursor = database.transaction.return_value.__enter__.return_value
        cursor.fetchone.side_effect = [{'generation': 3}, {'n': 300, 'first': through, 'last': through}]
        stored = {}
        reads = iter(['checkpoints', 'values', 'dates'])
        def fetchall():
            stage = next(reads)
            if stage == 'checkpoints':
                return [{'ticker': t, 'last_date': through} for t in tickers if t != missing]
            if stage == 'values':
                return stored['values'].to_dict('records')
            return [{'date': day.date()} for day in dates]
        cursor.fetchall.side_effect = fetchall
        with patch('production.pipeline.reconcile.FeatureRepository') as repository, \
             patch('production.pipeline.reconcile.read_history', side_effect=history):
            def persist(values, states, **kwargs):
                from production.features.repository import validate_frame
                validate_frame(values)
                self.assertEqual(set(states), {missing})
                stored['values'] = values.copy()
            repository.return_value.persist.side_effect = persist
            result = initialize_features(database, through, progress=lambda _: None)
        self.assertEqual(result['tickers_processed'], 1)
        self.assertEqual(result['rows_verified'], len(dates))
        repository.return_value.persist.assert_called_once()
        repository.return_value.publish.assert_called_once()


if __name__ == '__main__':
    unittest.main()
