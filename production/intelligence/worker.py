"""Historical and incremental labels and frozen inference, with atomic date batches."""

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from production.features.contract import calculation_version
from production.features.repository import source_version
from production.pipeline.config import ROOT
from production.pipeline.ingestion import canonical_tickers
from .database import WorkerDatabase, retry
from .labels import calculate_labels, contract_hash
from .migrate import SQL


def require_schema(cursor):
    cursor.execute('SELECT version,sha256 FROM liquidity.intelligence_schema_migrations ORDER BY version')
    if cursor.fetchall() != [{'version': SQL.stem, 'sha256': hashlib.sha256(SQL.read_bytes()).hexdigest()}]:
        raise ValueError('Intelligence schema differs from reviewed migration')


def active_version(database):
    with database.transaction(read_only=True) as cursor:
        require_schema(cursor)
        version = calculation_version(source_version(cursor))
        cursor.execute('SELECT min(last_date) AS through,count(*) AS n FROM liquidity.feature_checkpoints WHERE feature_version=%s', (version,))
        row = cursor.fetchone()
        if row['n'] != 300 or row['through'] is None:
            raise ValueError('The active price vintage requires all 300 feature checkpoints')
        cursor.execute('SELECT feature_names FROM liquidity.feature_versions WHERE feature_version=%s', (version,))
        if cursor.fetchone() is None:
            raise ValueError('Unknown feature version')
    return version, row['through']


def read_features(database, version, *, columns=None, start=None, through=None):
    names = columns or ['amihud_ma20', 'cs_spread_ma10', 'volume_z20', 'downside_vol_20d']
    tickers = [ticker for ticker in canonical_tickers() if ticker not in ('^NSEI', '^INDIAVIX')]
    pieces = []
    for offset in range(0, len(tickers)):
        selected = tickers[offset:offset+1]
        def read_batch():
            with database.transaction(read_only=True) as cursor:
                cursor.execute('SELECT f.ticker,f.date,s.cap_segment,f.is_warmup,f.is_valid_ohlcv,' +
                               ','.join('f.'+name for name in names) +
                               ' FROM liquidity.feature_values f JOIN liquidity.stocks s USING(ticker) '
                               'JOIN liquidity.feature_publications p USING(date,feature_version) '
                               'WHERE f.feature_version=%s AND f.ticker=ANY(%s) AND (%s::date IS NULL OR f.date>=%s) '
                               'AND (%s::date IS NULL OR f.date<=%s) ORDER BY f.ticker,f.date',
                               (version, selected, start, start, through, through))
                return pd.DataFrame(cursor.fetchall())
        pieces.append(retry(read_batch))
    frame = pd.concat(pieces, ignore_index=True)
    if frame.empty:
        raise ValueError('No published features in requested date range')
    frame['date'] = pd.to_datetime(frame.date)
    for name in names:
        frame[name] = pd.to_numeric(frame[name], errors='raise').astype('float32')
    return frame


def register_version(database, version):
    contract = json.loads((ROOT/'production/phase1/contract.json').read_text())
    with database.transaction() as cursor:
        require_schema(cursor)
        cursor.execute('INSERT INTO liquidity.intelligence_versions(feature_version,contract_sha256,model_hashes,thresholds) VALUES (%s,%s,%s::jsonb,%s::jsonb) ON CONFLICT DO NOTHING',
                       (version, contract_hash(), json.dumps(contract['model_hashes']), json.dumps({'hgbt': 0.534, 'large': 73.8298, 'mid': 72.3251, 'small': 72.4359})))
        cursor.execute('SELECT contract_sha256,model_hashes FROM liquidity.intelligence_versions WHERE feature_version=%s', (version,))
        row = cursor.fetchone()
        if row['contract_sha256'] != contract_hash() or row['model_hashes'] != contract['model_hashes']:
            raise ValueError('Frozen intelligence contract changed for an existing version')


def number(value):
    return None if pd.isna(value) else float(value)


def persist_labels(database, frame, version):
    """Insert each complete date once and mature previously pending outcomes."""
    with database.transaction() as cursor:
        cursor.execute('SELECT pg_advisory_xact_lock(6381402503)')
        require_schema(cursor)
        days = sorted(set(frame.date.dt.date))
        cursor.execute('SELECT date FROM liquidity.intelligence_publications WHERE feature_version=%s AND date=ANY(%s)', (version, days))
        existing = {row['date'] for row in cursor.fetchall()}
        new = frame.loc[~frame.date.dt.date.isin(existing)]
        if len(new):
            with cursor.copy('COPY liquidity.intelligence_observations(feature_version,date,ticker,lrs,forward_lrs,hre_label,horizon_end) FROM STDIN') as copy:
                for row in new.itertuples(index=False):
                    label = None if pd.isna(row.hre_label) else int(row.hre_label)
                    end = None if pd.isna(row.horizon_end) else row.horizon_end.date()
                    copy.write_row((version, row.date.date(), row.ticker, number(row.lrs), number(row.forward_lrs), label, end))
        cursor.execute('SELECT ticker,date FROM liquidity.intelligence_observations WHERE feature_version=%s AND date=ANY(%s) AND hre_label IS NULL', (version, list(existing)))
        pending = {(row['ticker'], row['date']) for row in cursor.fetchall()}
        updates = [(float(row.forward_lrs), int(row.hre_label), row.horizon_end.date(), version, row.ticker, row.date.date())
                   for row in frame.loc[frame.hre_label.notna()].itertuples(index=False)
                   if (row.ticker, row.date.date()) in pending]
        if updates:
            cursor.executemany('UPDATE liquidity.intelligence_observations SET forward_lrs=%s,hre_label=%s,horizon_end=%s WHERE feature_version=%s AND ticker=%s AND date=%s AND hre_label IS NULL', updates)
        for day, group in new.groupby(new.date.dt.date):
            cursor.execute('INSERT INTO liquidity.intelligence_publications(feature_version,date,rows) VALUES (%s,%s,%s)', (version, day, len(group)))
    return len(new)


def run_labels(database, *, incremental=False):
    version, through = retry(lambda: active_version(database))
    retry(lambda: register_version(database, version))
    start = None
    if incremental:
        with database.transaction(read_only=True) as cursor:
            cursor.execute('SELECT min(date) AS start FROM (SELECT date FROM liquidity.intelligence_publications WHERE feature_version=%s ORDER BY date DESC LIMIT 12) d', (version,))
            start = cursor.fetchone()['start']
    frame = read_features(database, version, start=start, through=through)
    result = calculate_labels(frame)
    days = sorted(result.date.unique())
    inserted = 0
    for offset in range(0, len(days), 20):
        batch = result.loc[result.date.isin(days[offset:offset+20])]
        inserted += retry(lambda: persist_labels(database, batch, version))
        if offset % 200 == 0 or offset+20 >= len(days):
            print(json.dumps({'stage': 'labels', 'dates_processed': min(offset+20, len(days)), 'dates': len(days), 'inserted': inserted}), flush=True)
    return {'feature_version': str(version), 'through': str(through), 'rows_calculated': len(result), 'inserted': inserted,
            'matured_labels': int(result.hre_label.notna().sum()), 'pending_horizons': int((~result.is_warmup & result.horizon_end.isna()).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['labels', 'inference'], required=True)
    parser.add_argument('--incremental', action='store_true')
    args = parser.parse_args()
    database = WorkerDatabase()
    try:
        if args.stage == 'labels':
            result = run_labels(database, incremental=args.incremental)
        else:
            from .inference import run_inference
            result = run_inference(database, incremental=args.incremental)
        print(json.dumps(result), flush=True)
    finally:
        database.close()


if __name__ == '__main__':
    main()
