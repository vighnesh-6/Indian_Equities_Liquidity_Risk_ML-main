"""Append actual, fully reconciled Yahoo revisions; never rescale old prices.

market_eod is the immutable observation archive. Production readers use
market_eod_consistent, whose activated overlays contain actual provider bars.
Every historical key must be fetched before a generation can be activated.
"""

import hashlib
import json
from collections import defaultdict

from .historical_import import row_bytes
from .repository import Bar, PersistenceError


def digest(records):
    result = hashlib.sha256()
    for day, bar in sorted(records.items()):
        result.update(row_bytes(day, (bar.open, bar.high, bar.low, bar.close, bar.volume, bar.turnover)))
    return result.hexdigest()


def active_generation(database):
    with database.transaction(read_only=True) as cursor:
        cursor.execute("SELECT max(generation_id) AS id FROM liquidity.price_basis_generations WHERE state='ACTIVE'")
        return cursor.fetchone()['id']


def load_history(database, base):
    groups = defaultdict(dict)
    with database.transaction(read_only=True) as cursor:
        cursor.execute("SELECT ticker,date,open,high,low,close,volume,turnover FROM liquidity.market_eod_consistent WHERE date<=%s ORDER BY ticker,date", (base,))
        for row in cursor.fetchall():
            groups[row['ticker']][row['date']] = Bar(row['date'], row['open'], row['high'], row['low'], row['close'], row['volume'], row['turnover'])
    if len(groups) != 302 or any(not records or max(records) != base for records in groups.values()):
        raise PersistenceError('Full price-basis transition requires complete 302-instrument checkpoint history')
    return dict(groups)


def fetch_full_history(provider, history, sessions, base, progress):
    quotes, sources = {}, {}
    for number, ticker in enumerate(sorted(history), 1):
        expected = sorted(set(history[ticker]) | set(sessions))
        quotes[ticker], sources[ticker] = provider.fetch(ticker, expected, historical_cutoff=base)
        # Yahoo has no point-in-time snapshot parameter. Re-read both ends to
        # reject an observed adjustment change during the multi-window fetch.
        for anchor in (base, sessions[-1]):
            probe, _ = provider.fetch(ticker, [anchor])
            if probe[anchor] != quotes[ticker][anchor]:
                raise PersistenceError('Provider adjustment changed during full-history collection: ' + ticker)
        progress(f'Validated full historical price basis {number}/302 {ticker}')
    return quotes, sources


def profiles(cursor, base):
    cursor.execute('SELECT * FROM liquidity.price_basis_profiles(%s)', (base,))
    return {row['ticker']: (int(row['rows']), row['values_sha256']) for row in cursor.fetchall()}


def stage_generation(database, history, quotes, sources, base, target):
    if set(history) != set(quotes) or len(history) != 302:
        raise PersistenceError('A price-basis generation requires all 302 historical instruments')
    baseline = {ticker: (len(rows), digest(rows)) for ticker, rows in history.items()}
    canonical = {}
    changes = []
    for ticker, old in history.items():
        actual = {day: quotes[ticker][day] for day in old}
        canonical[ticker] = (len(actual), digest(actual))
        for day, bar in actual.items():
            if bar != old[day]:
                changes.append((ticker, day, bar.open, bar.high, bar.low, bar.close, bar.volume, bar.turnover))
    with database.transaction() as cursor:
        cursor.execute('SELECT pg_advisory_xact_lock(6381402502)')
        if profiles(cursor, base) != baseline:
            raise PersistenceError('Canonical history changed while the provider snapshot was collected')
        cursor.execute("SELECT max(generation_id) AS id FROM liquidity.price_basis_generations WHERE state='ACTIVE'")
        parent = cursor.fetchone()['id']
        request = hashlib.sha256(json.dumps({'parent': parent, 'base': str(base), 'target': str(target), 'baseline': baseline, 'provider': canonical}, sort_keys=True).encode()).hexdigest()
        cursor.execute("INSERT INTO liquidity.price_basis_generations(request_key,parent_id,base_date,target_date) VALUES (%s,%s,%s,%s) ON CONFLICT(request_key) DO NOTHING RETURNING generation_id", (request, parent, base, target))
        created = cursor.fetchone()
        if created:
            generation = created['generation_id']
            cursor.executemany('INSERT INTO liquidity.price_basis_receipts(generation_id,ticker,expected_rows,baseline_sha256,provider_sha256,source_version,retrieved_at) VALUES (%s,%s,%s,%s,%s,%s,%s)',
                               [(generation, ticker, baseline[ticker][0], baseline[ticker][1], canonical[ticker][1], sources[ticker].source_version, sources[ticker].retrieved_at) for ticker in sorted(history)])
            for offset in range(0, len(changes), 1000):
                cursor.executemany('INSERT INTO liquidity.price_basis_revisions(generation_id,ticker,date,open,high,low,close,volume,turnover) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                                   [(generation, *row) for row in changes[offset:offset + 1000]])
        else:
            cursor.execute('SELECT generation_id FROM liquidity.price_basis_generations WHERE request_key=%s', (request,))
            generation = cursor.fetchone()['generation_id']
    return generation, {'generation_id': generation, 'historical_rows': sum(count for count, _ in canonical.values()),
                        'actual_provider_revision_rows': len(changes), 'original_observations_modified': 0,
                        'basis': 'Actual Yahoo adjusted OHLCV; complete historical-key reconciliation; no arithmetic rescaling',
                        'provider_historical_sha256': {ticker: value[1] for ticker, value in canonical.items()}}


def activate_generation(cursor, generation):
    cursor.execute('SELECT * FROM liquidity.price_basis_generations WHERE generation_id=%s FOR UPDATE', (generation,))
    row = cursor.fetchone()
    if row is None:
        raise PersistenceError('Unknown price-basis generation')
    if row['state'] == 'ACTIVE':
        return
    cursor.execute('SELECT ticker,expected_rows,baseline_sha256 FROM liquidity.price_basis_receipts WHERE generation_id=%s', (generation,))
    baseline = {r['ticker']: (int(r['expected_rows']), r['baseline_sha256']) for r in cursor.fetchall()}
    if len(baseline) != 302 or profiles(cursor, row['base_date']) != baseline:
        raise PersistenceError('Price-basis baseline is stale or incomplete')
    cursor.execute("UPDATE liquidity.price_basis_generations SET state='ACTIVE',activated_at=now() WHERE generation_id=%s", (generation,))


def reconcile_generation(database, generation):
    with database.transaction(read_only=True) as cursor:
        cursor.execute('SELECT base_date,state FROM liquidity.price_basis_generations WHERE generation_id=%s', (generation,))
        row = cursor.fetchone()
        cursor.execute('SELECT ticker,expected_rows,provider_sha256 FROM liquidity.price_basis_receipts WHERE generation_id=%s', (generation,))
        expected = {r['ticker']: (int(r['expected_rows']), r['provider_sha256']) for r in cursor.fetchall()}
        actual = profiles(cursor, row['base_date'])
    return {'result': 'PASS' if row['state'] == 'ACTIVE' and len(expected) == 302 and expected == actual else 'FAIL',
            'generation_id': generation, 'historical_rows': sum(value[0] for value in actual.values()),
            'provider_profiles_match': expected == actual}
