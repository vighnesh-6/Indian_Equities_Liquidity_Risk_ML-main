import unittest
import numpy as np

from production.features.contract import FEATURES, FLOAT32_ATOL, FLOAT32_RTOL
from production.features.engine import calculate
from production.features.state import FeatureState, initialize, update
from production.features.tests.test_engine import fixture


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.stock = fixture(rows=320)
        self.benchmark = fixture("^NSEI", rows=320)
        self.vix = fixture("^INDIAVIX", rows=320)

    def test_daily_equivalence_serialization_and_recursive_state(self):
        _, state = initialize(self.stock.iloc[:180], self.benchmark, self.vix, source_version="fixture")
        expected = calculate(self.stock, self.benchmark, self.vix)
        for start in [180, 190, 230, 280]:
            stop = {180: 190, 190: 230, 230: 280, 280: 320}[start]
            state = FeatureState.from_json(state.to_json())
            actual, state = update(state, self.stock.iloc[start:stop], self.benchmark, self.vix,
                                   source_version="fixture")
            ref = expected.iloc[start:stop][list(FEATURES)].to_numpy()
            np.testing.assert_array_equal(np.isnan(actual[list(FEATURES)]), np.isnan(ref))
            np.testing.assert_allclose(actual[list(FEATURES)], ref, rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL, equal_nan=True)
            np.testing.assert_array_equal(actual.rsi_14, expected.rsi_14.iloc[start:stop])
            self.assertLessEqual(len(state.context), 100)
        self.assertEqual(state.count, 320)

    def test_single_session_and_original_state_preserved(self):
        _, state = initialize(self.stock.iloc[:200], self.benchmark, self.vix, source_version="fixture")
        snapshot = state.to_json()
        actual, revised = update(state, self.stock.iloc[200:201], self.benchmark, self.vix, source_version="fixture")
        self.assertEqual(state.to_json(), snapshot)
        self.assertEqual(revised.count, 201)
        np.testing.assert_allclose(actual[list(FEATURES)], calculate(self.stock, self.benchmark, self.vix).iloc[200:201][list(FEATURES)],
                                   rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL, equal_nan=True)

    def test_changed_vintage_and_backward_date_rejected(self):
        _, state = initialize(self.stock.iloc[:200], self.benchmark, self.vix, source_version="fixture")
        for frame, version in [(self.stock.iloc[199:201], "fixture"), (self.stock.iloc[200:201], "changed")]:
            with self.assertRaises(ValueError):
                update(state, frame, self.benchmark, self.vix, source_version=version)

    def test_recursive_rsi_survives_masked_corporate_action(self):
        stock = fixture("PATANJALI.NS", rows=260)
        gap = stock.index[stock.date.eq("2020-01-27")][0]
        stock.loc[gap:, ["open", "high", "low", "close"]] *= np.float32(0.05)
        _, state = initialize(stock.iloc[:gap], self.benchmark, self.vix, source_version="fixture")
        actual, revised = update(state, stock.iloc[gap:], self.benchmark, self.vix, source_version="fixture")
        expected = calculate(stock, self.benchmark, self.vix).iloc[gap:]
        np.testing.assert_array_equal(actual.rsi_14, expected.rsi_14)
        np.testing.assert_allclose(actual[list(FEATURES)], expected[list(FEATURES)],
                                   rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL, equal_nan=True)
        self.assertEqual(revised.count, len(stock))

    def test_real_reference_preserves_pandas_variance_history(self):
        from production.features.tests.research_incremental import check_research_incremental
        result = check_research_incremental()
        self.assertEqual(result["result"], "PASS", result)
        self.assertEqual(result["maximum_absolute_differences"], {})
