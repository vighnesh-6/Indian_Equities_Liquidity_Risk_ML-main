"""Inference-only adapters for the two hash-verified frozen models."""

import hashlib
import json
import time
from uuid import uuid4

import numpy as np
import pandas as pd

from production.pipeline.config import ROOT
from production.phase1.validate_runtime import load_models
from .database import retry
from .worker import active_version, register_version

CONTRACT = json.loads((ROOT/'production/phase1/contract.json').read_text())
MODEL_FEATURES = list(dict.fromkeys(CONTRACT['features']['hist_gradient_boosting'] + CONTRACT['features']['isolation_forest']))


class FrozenInference:
    def __init__(self):
        self.hgbt, self.detector = load_models()
        if list(self.hgbt.feature_names_in_) != CONTRACT['features']['hist_gradient_boosting']:
            raise ValueError('HGBT feature order differs from the frozen contract')
        if self.detector.feature_subset != CONTRACT['features']['isolation_forest']:
            raise ValueError('Isolation Forest feature order differs from the frozen contract')
        if [self.detector.score_min_, self.detector.score_max_] != CONTRACT['if_score_bounds']:
            raise ValueError('Isolation Forest fitted score bounds differ')

    def predict(self, frame):
        if not set(MODEL_FEATURES).issubset(frame):
            raise ValueError('Frozen model inputs are incomplete')
        if np.isinf(frame[MODEL_FEATURES].to_numpy(dtype='float64')).any():
            raise ValueError('Infinite model input')
        output = frame[['ticker', 'date']].copy()
        output['hre_probability'] = np.nan
        output['isolation_score'] = np.nan
        eligible = ~frame.is_warmup & frame.is_valid_ohlcv
        hgbt_ready = eligible & frame[CONTRACT['features']['hist_gradient_boosting']].notna().any(axis=1)
        if_ready = eligible & frame[CONTRACT['features']['isolation_forest']].notna().any(axis=1)
        if hgbt_ready.any():
            values = self.hgbt.predict_proba(frame.loc[hgbt_ready, CONTRACT['features']['hist_gradient_boosting']])
            output.loc[hgbt_ready, 'hre_probability'] = values[:, list(self.hgbt.classes_).index(1.0)]
        if if_ready.any():
            output.loc[if_ready, 'isolation_score'] = self.detector.score(frame.loc[if_ready])['isolation_score']
        return output


def pending_dates(database, version):
    with database.transaction(read_only=True) as cursor:
        cursor.execute('SELECT date FROM liquidity.intelligence_publications WHERE feature_version=%s AND predictions_at IS NULL ORDER BY date', (version,))
        return [row['date'] for row in cursor.fetchall()]


def persist_predictions(database, frame, version):
    with database.transaction() as cursor:
        cursor.execute('SELECT pg_advisory_xact_lock(6381402503)')
        days = sorted(set(frame.date.dt.date))
        cursor.execute('SELECT date FROM liquidity.intelligence_publications WHERE feature_version=%s AND date=ANY(%s) AND predictions_at IS NULL', (version, days))
        pending = {row['date'] for row in cursor.fetchall()}
        selected = frame.loc[frame.date.dt.date.isin(pending)]
        for start in range(0, len(selected), 2000):
            group = selected.iloc[start:start+2000]
            values = []
            for row in group.itertuples(index=False):
                values.extend((version, row.ticker, row.date.date(),
                               None if pd.isna(row.hre_probability) else float(row.hre_probability),
                               None if pd.isna(row.isolation_score) else float(row.isolation_score)))
            placeholders = ','.join(['(%s::uuid,%s::text,%s::date,%s::float8,%s::float8)'] * len(group))
            cursor.execute('UPDATE liquidity.intelligence_observations i SET hre_probability=v.probability,isolation_score=v.score '
                           'FROM (VALUES '+placeholders+') AS v(version,ticker,date,probability,score) '
                           'WHERE i.feature_version=v.version AND i.ticker=v.ticker AND i.date=v.date', values)
            if cursor.rowcount != len(group):
                raise ValueError('Prediction keys differ from the complete label cohort')
        cursor.execute('UPDATE liquidity.intelligence_publications SET predictions_at=clock_timestamp() WHERE feature_version=%s AND date=ANY(%s) AND predictions_at IS NULL', (version, list(pending)))
    return len(selected)


def run_inference(database, *, incremental=False):
    version, through = retry(lambda: active_version(database))
    retry(lambda: register_version(database, version))
    days = retry(lambda: pending_dates(database, version))
    if not days:
        return {'status': 'NOOP', 'feature_version': str(version), 'through': str(through), 'rows': 0}
    engine = FrozenInference()
    run_id = uuid4()
    with database.transaction() as cursor:
        cursor.execute("INSERT INTO liquidity.intelligence_runs(run_id,feature_version,stage,through_date,status) VALUES (%s,%s,'inference',%s,'RUNNING')", (run_id, version, through))
    processed = 0
    try:
        # Date batches keep serverless memory, query time and transaction size bounded.
        for offset in range(0, len(days), 40):
            batch_days = days[offset:offset+40]
            def read():
                with database.transaction(read_only=True) as cursor:
                    cursor.execute('SELECT f.ticker,f.date,f.is_warmup,f.is_valid_ohlcv,'+','.join('f.'+name for name in MODEL_FEATURES)+
                                   ' FROM liquidity.feature_values f WHERE f.feature_version=%s AND f.date=ANY(%s) ORDER BY f.ticker,f.date', (version, batch_days))
                    result = pd.DataFrame(cursor.fetchall())
                result['date'] = pd.to_datetime(result.date)
                for name in MODEL_FEATURES:
                    result[name] = pd.to_numeric(result[name]).astype('float32')
                return result
            features = retry(read)
            predictions = engine.predict(features)
            processed += retry(lambda: persist_predictions(database, predictions, version))
            print(json.dumps({'stage': 'inference', 'dates_processed': min(offset+40, len(days)), 'dates': len(days), 'rows': processed}), flush=True)
        with database.transaction() as cursor:
            cursor.execute("UPDATE liquidity.intelligence_runs SET status='SUCCESS',finished_at=clock_timestamp(),rows_processed=%s,details=%s::jsonb WHERE run_id=%s", (processed, json.dumps({'model_hashes': CONTRACT['model_hashes'], 'hgbt_threshold': 0.534}), run_id))
    except Exception as error:
        try:
            with database.transaction() as cursor:
                cursor.execute("UPDATE liquidity.intelligence_runs SET status='FAILED',finished_at=clock_timestamp(),rows_processed=%s,details=%s::jsonb WHERE run_id=%s", (processed, json.dumps({'error_type': type(error).__name__}), run_id))
        except Exception:
            pass
        raise
    return {'status': 'SUCCESS', 'feature_version': str(version), 'through': str(through), 'rows': processed, 'run_id': str(run_id)}
