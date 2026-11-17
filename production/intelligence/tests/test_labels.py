import unittest

import numpy as np
import pandas as pd

from production.intelligence.labels import calculate_labels
from src.labels.targets import build_features_with_targets


class LabelTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for ticker, segment in [('A', 'large'), ('B', 'large'), ('C', 'small')]:
            for index, day in enumerate(pd.bdate_range('2020-01-01', periods=30)):
                value = index + ord(ticker)
                rows.append(dict(date=day, ticker=ticker, cap_segment=segment, is_warmup=index<3,
                                 is_valid_ohlcv=True, amihud_ma20=value, cs_spread_ma10=value,
                                 volume_z20=-value, downside_vol_20d=value))
        return pd.DataFrame(rows)

    def test_research_parity_and_exact_tail(self):
        source = self.fixture()
        actual = calculate_labels(source)
        expected = build_features_with_targets(source)
        for name, research_name in [('lrs', 'lrs'), ('forward_lrs', 'fwd_max_lrs_5d'), ('hre_label', 'hre_5d')]:
            np.testing.assert_array_equal(actual[name], expected[research_name])
        for _, group in actual.groupby('ticker'):
            self.assertTrue(group.tail(5).hre_label.isna().all())
            self.assertTrue(group.head(3).lrs.isna().all())
            self.assertEqual(group.iloc[3].horizon_end, group.iloc[8].date)

    def test_present_excluded_and_ticker_isolation(self):
        source = self.fixture()
        actual = calculate_labels(source)
        for _, group in actual.groupby('ticker'):
            post = group.loc[~group.is_warmup]
            for index in range(len(post)-5):
                self.assertEqual(post.iloc[index].forward_lrs, post.iloc[index+1:index+6].lrs.max())
        changed = source.copy()
        changed.loc[changed.date.gt(pd.Timestamp('2020-01-20')), 'amihud_ma20'] *= 9
        result = calculate_labels(changed)
        np.testing.assert_array_equal(actual.loc[actual.date.le('2020-01-20'), 'lrs'], result.loc[result.date.le('2020-01-20'), 'lrs'])

    def test_duplicates_and_excluded_day_fail(self):
        frame = self.fixture()
        with self.assertRaises(ValueError):
            calculate_labels(pd.concat([frame, frame.iloc[:1]]))
        frame.loc[0, 'date'] = pd.Timestamp('2026-11-08')
        with self.assertRaises(ValueError):
            calculate_labels(frame)
