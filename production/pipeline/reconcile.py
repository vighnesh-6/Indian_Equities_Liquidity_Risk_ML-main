"""Append-only operational reconciliation using the frozen production engines.

Original market observations and previously published feature/model versions
are retained. A source-vintage change initializes a separate feature version;
ordinary sessions continue through the existing incremental feature engine.
"""

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from production.features.contract import FEATURES, FLAGS, calculation_version, verify_contract
from production.features.engine import frame_hash, input_hash
from production.features.repository import FeatureRepository, source_version
from production.features.run import read_history
from production.features.state import initialize, update
from production.intelligence.database import WorkerDatabase, retry
from .config import ROOT
from .ingestion import SessionIngestor, canonical_tickers
from .provider import YahooProvider
from .repository import PersistenceError

RESERVE_BYTES = 64 * 1024 * 1024


def storage_estimate(current, sizes, counts, stock_rows, new_rows, revisions):
    """Conservative allocated-size estimate; never assumes VACUUM or deletion."""
    def estimate(table, rows, fallback):
        return math.ceil(rows * max(fallback, sizes.get(table, 0) / max(counts.get(table, 0), 1)))

    added = sum((estimate('feature_values', stock_rows, 256),
                 estimate('intelligence_observations', stock_rows, 128),
                 estimate('feature_checkpoints', 300, 32768),
                 estimate('price_basis_revisions', revisions, 192),
                 estimate('market_eod', new_rows, 160), 5 * 1024 * 1024))
    return current + added


def require_capacity(database, stock_rows, *, new_rows=0, revisions=0):
    with database.transaction(read_only=True) as cursor:
        cursor.execute("SELECT pg_database_size(current_database()) AS bytes,"
                       "pg_size_bytes(current_setting('neon.max_cluster_size',true)) AS capacity")
        storage = cursor.fetchone()
        names = ['feature_values', 'intelligence_observations', 'feature_checkpoints',
                 'price_basis_revisions', 'market_eod']
        sizes, counts = {}, {}
        for name in names:
            # Identifiers are fixed internal constants, never request input.
            cursor.execute('SELECT pg_total_relation_size(%s::regclass) AS bytes, '
                           '(SELECT count(*) FROM liquidity.' + name + ') AS rows',
                           ('liquidity.' + name,))
            row = cursor.fetchone()
            sizes[name], counts[name] = row['bytes'], row['rows']
    projected = storage_estimate(storage['bytes'], sizes, counts, stock_rows, new_rows, revisions)
    if storage['capacity'] is None or projected + RESERVE_BYTES > storage['capacity']:
        raise PersistenceError('Insufficient verified capacity for append-only reconciliation; '
                               'original records retained and no historical deletion attempted')
    return {'current_bytes': storage['bytes'], 'projected_bytes': projected,
            'capacity_bytes': storage['capacity'], 'reserved_bytes': RESERVE_BYTES}


def initialize_features(database, through, *, progress=print):
    """Resume ticker commits without replacing old source versions or formulas."""
    verify_contract()
    repository = FeatureRepository(database)
    tickers = [ticker for ticker in canonical_tickers() if not ticker.startswith('^')]
    with database.transaction(read_only=True) as cursor:
        source = source_version(cursor)
        version = calculation_version(source)
        cursor.execute('SELECT ticker,last_date FROM liquidity.feature_checkpoints WHERE feature_version=%s', (version,))
        checkpoints = {row['ticker']: row['last_date'] for row in cursor.fetchall()}
    if set(checkpoints) == set(tickers) and set(checkpoints.values()) == {through}:
        return {'status': 'NOOP', 'feature_version': version, 'rows': 0}
    if any(day > through for day in checkpoints.values()):
        raise PersistenceError('Feature checkpoint is ahead of the market target')
    retry(lambda: repository.register_version(source))
    benchmark = retry(lambda: read_history(database, '^NSEI', through))
    vix = retry(lambda: read_history(database, '^INDIAVIX', through))
    market_hashes = {'benchmark': input_hash(benchmark), 'vix': input_hash(vix)}
    processed = 0
    verified_rows = 0
    for number, ticker in enumerate(tickers, 1):
        if checkpoints.get(ticker) != through:
            stock = retry(lambda: read_history(database, ticker, through))
            state = retry(lambda: repository.load_state(ticker, version)) if ticker in checkpoints else None
            previous = hashlib.sha256(state.to_json().encode()).hexdigest() if state else None
            if state:
                stock = stock.loc[stock.date.gt(pd.Timestamp(state.last_date))].copy()
                values, state = update(state, stock, benchmark, vix, source_version=source)
            else:
                values, state = initialize(stock, benchmark, vix, source_version=source)
            lineage = {'stock': input_hash(stock), **market_hashes, 'previous_state': previous}
            input_sha = hashlib.sha256(json.dumps(lineage, sort_keys=True).encode()).hexdigest()
            key = 'reconcile-features:' + hashlib.sha256(
                (ticker + version + str(through) + input_sha).encode()).hexdigest()
            retry(lambda: repository.persist(values, {ticker: state}, version=version,
                  request_key=key, input_sha256=input_sha,
                  provenance={'source_version': source, 'market_view': 'liquidity.market_eod_consistent',
                              'input_hashes': lineage}))
            # Verify the committed values and null masks before publication.
            actual = read_committed_features(database, version, ticker, values.date.min().date(), through)
            actual['date'] = pd.to_datetime(actual.date)
            for name in FEATURES:
                actual[name] = pd.to_numeric(actual[name]).astype('float32')
            if len(actual) != len(values) or frame_hash(actual) != frame_hash(values):
                raise PersistenceError('Committed feature values or null masks differ: ' + ticker)
            if not actual[list(FLAGS)].reset_index(drop=True).equals(values[list(FLAGS)].reset_index(drop=True)):
                raise PersistenceError('Committed quality flags differ: ' + ticker)
            verified_rows += len(values)
            processed += 1
        if number % 25 == 0:
            progress(f'Feature histories complete {number}/300; newly processed {processed}')
    with database.transaction(read_only=True) as cursor:
        cursor.execute('SELECT count(*) AS n,min(last_date) AS first,max(last_date) AS last '
                       'FROM liquidity.feature_checkpoints WHERE feature_version=%s', (version,))
        row = cursor.fetchone()
        if row['n'] != 300 or row['first'] != through or row['last'] != through:
            raise PersistenceError('Feature checkpoints are not a complete 300-equity cohort')
        cursor.execute('SELECT DISTINCT date FROM liquidity.feature_values WHERE feature_version=%s ORDER BY date', (version,))
        days = [row['date'] for row in cursor.fetchall()]
    retry(lambda: repository.publish(version, days))
    return {'status': 'SUCCESS', 'feature_version': version, 'rows_verified': verified_rows,
            'tickers_processed': processed, 'published_dates': len(days)}


def read_committed_features(database, version, ticker, first, through):
    """Read every committed value without an oversized serverless result."""
    from datetime import timedelta
    previous = first - timedelta(days=1)
    rows = []
    while True:
        def read():
            with database.transaction(read_only=True) as cursor:
                cursor.execute('SELECT ticker,date,' + ','.join([*FLAGS, *FEATURES]) +
                               ' FROM liquidity.feature_values WHERE feature_version=%s AND ticker=%s '
                               'AND date>%s AND date<=%s ORDER BY date LIMIT 500',
                               (version, ticker, previous, through))
                return cursor.fetchall()
        batch = retry(read)
        rows.extend(batch)
        if len(batch) < 500:
            return pd.DataFrame(rows)
        previous = batch[-1]['date']


def run(database, provider, *, now=None, progress=print):
    capacity = {}

    def before_stage(history, quotes):
        stock_rows = sum(len(rows) for ticker, rows in quotes.items() if not ticker.startswith('^'))
        revisions = sum(bar != quotes[ticker][day] for ticker, rows in history.items()
                        for day, bar in rows.items())
        new_rows = sum(len(quotes[ticker]) - len(rows) for ticker, rows in history.items())
        capacity.update(retry(lambda: require_capacity(
            database, stock_rows, new_rows=new_rows, revisions=revisions)))
        progress('Append-only storage preflight: ' + json.dumps(capacity))

    ingestion = SessionIngestor(database, provider).collect(
        'daily', now=now, progress=progress, allow_basis_refresh=True, before_basis_stage=before_stage)
    result = {**ingestion, 'storage': capacity}
    if ingestion['status'] not in {'SUCCESS', 'NOOP'}:
        return result
    through = date.fromisoformat(ingestion['plan']['target'])
    result['features'] = initialize_features(database, through, progress=progress)
    result['status'] = 'SUCCESS'
    return result


def report_path(path):
    target = path.resolve()
    allowed = [Path(tempfile.gettempdir()).resolve(), ROOT / 'production/pipeline/phase2_results']
    ci_temp = os.environ.get('RUNNER_TEMP') if os.environ.get('GITHUB_ACTIONS') == 'true' else None
    if ci_temp:
        allowed.append(Path(ci_temp).resolve())
    if target.suffix != '.json' or not any(target.is_relative_to(root) for root in allowed):
        raise ValueError('Operational reports require temporary or approved production storage')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    target = report_path(args.report)
    database = WorkerDatabase()
    result = None
    try:
        result = run(database, YahooProvider())
    except Exception as error:
        result = {'status': 'FAILED', 'issue': {'code': type(error).__name__, 'message': str(error)}}
    finally:
        database.close()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, default=str, indent=2), encoding='utf-8')
        print(json.dumps(result, default=str), flush=True)
    if result['status'] != 'SUCCESS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
