"""Persist only already-verified events from the protected research registry."""

import csv
import hashlib
import json

from production.pipeline.config import ROOT
from .database import WorkerDatabase


def import_verified(database):
    source = ROOT/'data/events/market_events.csv'
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(encoding='utf-8-sig', newline='') as handle:
        records = [row for row in csv.DictReader(handle) if row['verification_status'] == 'VERIFIED']
    values = [(row['event_id'], row['effective_trading_date'], row['event_name'], row['event_category'],
               row['event_description'], row['source_name'], row['source_url'], 'VERIFIED', checksum,
               json.dumps(row, sort_keys=True)) for row in records]
    with database.transaction() as cursor:
        cursor.executemany('INSERT INTO liquidity.market_events(event_id,event_date,name,category,description,source_name,source_url,verification_status,source_sha256,metadata) '
                           'VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING', values)
    return {'verified_registry_events': len(values), 'source_sha256': checksum}


if __name__ == '__main__':
    database = WorkerDatabase()
    try:
        print(json.dumps(import_verified(database)))
    finally:
        database.close()
