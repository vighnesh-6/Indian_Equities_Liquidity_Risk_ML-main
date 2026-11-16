import ast
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from production.features.contract import CONTRACT, FEATURES, FLAGS, ROOT, verify_contract
from production.features.engine import calculate


def fixture(ticker="RELIANCE.NS", rows=260, start="2020-01-01"):
    dates = pd.bdate_range(start, periods=rows)
    sequence = np.arange(rows)
    close = (100 * np.exp(0.0007 * sequence + 0.02 * np.sin(sequence / 7))).astype("float32")
    return pd.DataFrame({"date": dates, "ticker": ticker, "open": close,
                         "high": close * np.float32(1.02), "low": close * np.float32(0.98),
                         "close": close, "volume": (1000 + sequence % 17 * 101).astype("int64")})


class ContractTests(unittest.TestCase):
    def test_frozen_hashes_and_exact_58_names(self):
        verify_contract()
        self.assertEqual(len(FEATURES), 58)
        self.assertEqual(len(CONTRACT["core_features"]), 55)

    def test_numerical_function_bodies_match_approved_sources(self):
        target = ast.parse((ROOT / "production/features/kernels.py").read_text(encoding="utf-8"))
        functions = {n.name: ast.dump(n, include_attributes=False) for n in target.body if isinstance(n, ast.FunctionDef)}
        for name in ["_common", "returns", "volatility", "liquidity", "gaps", "market_relative", "market_stress"]:
            original = ast.parse((ROOT / f"src/features/{name}.py").read_text(encoding="utf-8"))
            for node in original.body:
                if isinstance(node, ast.FunctionDef):
                    self.assertEqual(functions[node.name], ast.dump(node, include_attributes=False))

    def test_text_identity_is_independent_of_checkout_line_endings(self):
        import tempfile
        from production.features.contract import text_sha256
        with tempfile.TemporaryDirectory(prefix="liquidity-phase3-text-") as directory:
            path = Path(directory) / "source.txt"
            path.write_bytes(b"frozen\ncontract\n")
            expected = text_sha256(path)
            path.write_bytes(b"frozen\r\ncontract\r\n")
            self.assertEqual(text_sha256(path), expected)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.stock = fixture()
        self.benchmark = fixture("^NSEI")
        self.vix = fixture("^INDIAVIX")

    def test_output_order_dtype_warmup_and_null_boundaries(self):
        out = calculate(self.stock, self.benchmark, self.vix)
        self.assertEqual([c for c in out if c in FEATURES], list(FEATURES))
        self.assertTrue(all(out[c].dtype == np.dtype("float32") for c in FEATURES))
        self.assertEqual(out.is_warmup.sum(), 80)
        self.assertTrue(np.isnan(out.return_20d.iloc[19]))
        self.assertTrue(np.isfinite(out.return_20d.iloc[20]))
        self.assertFalse(np.isinf(out[list(FEATURES)].to_numpy()).any())

    def test_causality_future_price_and_volume_perturbation(self):
        baseline = calculate(self.stock.iloc[:180], self.benchmark.iloc[:180], self.vix.iloc[:180])
        changed = self.stock.copy()
        changed.loc[180:, ["open", "high", "low", "close", "volume"]] *= 2
        result = calculate(changed, self.benchmark, self.vix).iloc[:180]
        np.testing.assert_array_equal(baseline[list(FEATURES)], result[list(FEATURES)])

    def test_causality_future_benchmark_and_vix_perturbation(self):
        baseline = calculate(self.stock, self.benchmark, self.vix).iloc[:180]
        benchmark, vix = self.benchmark.copy(), self.vix.copy()
        for frame in [benchmark, vix]:
            frame.loc[180:, ["open", "high", "low", "close", "volume"]] *= 3
        changed = calculate(self.stock, benchmark, vix).iloc[:180]
        np.testing.assert_array_equal(baseline[list(FEATURES)], changed[list(FEATURES)])

    def test_ticker_isolation_duplicate_dates_and_excluded_day(self):
        with self.assertRaises(ValueError):
            calculate(pd.concat([self.stock, fixture("TCS.NS")]), self.benchmark, self.vix)
        with self.assertRaises(ValueError):
            calculate(pd.concat([self.stock, self.stock.iloc[:1]]), self.benchmark, self.vix)
        changed = self.stock.copy()
        changed.loc[0, "date"] = pd.Timestamp("2026-11-08")
        with self.assertRaisesRegex(ValueError, "excluded"):
            calculate(changed, self.benchmark, self.vix)

    def test_zero_volume_and_benchmark_missingness(self):
        self.stock.loc[100, "volume"] = 0
        self.benchmark.loc[120, "volume"] = 0
        out = calculate(self.stock, self.benchmark, self.vix)
        self.assertTrue(np.isnan(out.amihud_ratio.iloc[100]))
        self.assertEqual(out.zero_vol_flag.iloc[100], 1)
        self.assertTrue(np.isnan(out.relative_volume.iloc[120]))
        benchmark = self.benchmark.drop(index=130)
        out = calculate(self.stock, benchmark, self.vix)
        self.assertFalse(out.is_benchmark_available.iloc[130])
        self.assertTrue(np.isnan(out.relative_volume.iloc[130]))

    def test_approved_corporate_gap_masks_without_dropping_rows(self):
        stock = fixture("PATANJALI.NS")
        index = stock.index[stock.date.eq("2020-01-27")][0]
        stock.loc[index:, ["open", "high", "low", "close"]] *= np.float32(0.05)
        out = calculate(stock, self.benchmark, self.vix)
        self.assertTrue(out.is_corporate_action_gap.iloc[index])
        for name in ["daily_return", "true_range", "amihud_ratio", "overnight_gap", "cs_spread_est", "rsi_14"]:
            self.assertTrue(np.isnan(out[name].iloc[index]), name)
        self.assertEqual(len(out), len(stock))

    def test_wilder_rsi_edges(self):
        for slope, expected in [(1, 100), (-1, 0), (0, 50)]:
            stock = self.stock.iloc[:40].copy()
            close = 100 + slope * np.arange(40)
            for name in ["open", "high", "low", "close"]:
                stock[name] = close.astype("float32")
            out = calculate(stock, self.benchmark, self.vix)
            self.assertEqual(out.rsi_14.iloc[-1], expected)
