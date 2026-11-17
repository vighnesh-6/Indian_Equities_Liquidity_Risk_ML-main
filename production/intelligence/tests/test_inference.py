import json
import unittest

import numpy as np
import pandas as pd

from production.pipeline.config import ROOT
from production.intelligence.inference import FrozenInference


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FrozenInference()

    def test_frozen_reference_probabilities_and_scores(self):
        fixtures = json.loads((ROOT/'production/phase1/inputs.json').read_text())
        reference = json.loads((ROOT/'production/phase1/windows_reference.json').read_text())
        frame = pd.DataFrame(fixtures['values'][:20], columns=fixtures['columns'])
        frame['ticker'] = 'RELIANCE.NS'
        frame['date'] = pd.bdate_range('2026-01-01', periods=len(frame))
        frame['is_warmup'] = False
        frame['is_valid_ohlcv'] = True
        actual = self.engine.predict(frame)
        np.testing.assert_allclose(actual.hre_probability, reference['predictions']['hgbt_probability'][:20], rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(actual.isolation_score, reference['predictions']['isolation_score'][:20], rtol=1e-7, atol=1e-9)
        frame['is_warmup'] = True
        self.assertTrue(self.engine.predict(frame).hre_probability.isna().all())
