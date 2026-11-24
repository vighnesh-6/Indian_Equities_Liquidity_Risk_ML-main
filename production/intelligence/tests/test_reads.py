import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from production.intelligence.inference import MODEL_FEATURES, read_inference_features
from production.intelligence.worker import read_features


class BoundedReadTests(unittest.TestCase):
    def test_inference_reads_preserve_all_keys_nulls_and_float32_inputs(self):
        days = list(pd.bdate_range('2026-11-02', periods=7).date)
        rows = [{'ticker': ticker, 'date': day, 'is_warmup': False, 'is_valid_ohlcv': True,
                 **{name: (None if n == 2 else n * 0.125) for name in MODEL_FEATURES}}
                for ticker in ['B.NS', 'A.NS'] for n, day in enumerate(days)]
        database = MagicMock()
        cursor = database.transaction.return_value.__enter__.return_value
        cursor.fetchall.side_effect = [[r for r in rows if r['date'] in days[start:start+3]]
                                      for start in range(0, len(days), 3)]
        actual = read_inference_features(database, 'version', days)
        expected = pd.DataFrame(rows).sort_values(['ticker', 'date']).reset_index(drop=True)
        expected['date'] = pd.to_datetime(expected.date)
        expected[MODEL_FEATURES] = expected[MODEL_FEATURES].astype('float32')
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual([call.args[1][1] for call in cursor.execute.call_args_list],
                         [days[:3], days[3:6], days[6:]])
        self.assertTrue(all(call.kwargs == {'read_only': True} for call in database.transaction.call_args_list))

    def test_label_reads_keep_each_ticker_history_complete(self):
        names = MODEL_FEATURES[:3]
        days = pd.bdate_range('2026-11-02', periods=7).date
        rows = [[{'ticker': ticker, 'date': day, 'cap_segment': 'large',
                  'is_warmup': False, 'is_valid_ohlcv': True,
                  **{name: (None if n == 2 else n * 0.125) for name in names}}
                 for n, day in enumerate(days)] for ticker in ['A.NS', 'B.NS']]
        database = MagicMock()
        cursor = database.transaction.return_value.__enter__.return_value
        cursor.fetchall.side_effect = rows
        with patch('production.intelligence.worker.canonical_tickers', return_value=['A.NS', 'B.NS', '^NSEI', '^INDIAVIX']):
            actual = read_features(database, 'version', columns=names)
        expected = pd.DataFrame(rows[0] + rows[1])
        expected['date'] = pd.to_datetime(expected.date)
        expected[names] = expected[names].astype('float32')
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual([call.args[1][1] for call in cursor.execute.call_args_list], [['A.NS'], ['B.NS']])


if __name__ == '__main__':
    unittest.main()
