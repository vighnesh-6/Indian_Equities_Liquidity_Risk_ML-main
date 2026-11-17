"""Locked forward extension migration, leaving the collector ledger intact."""

import argparse
import hashlib
from pathlib import Path

from production.features.repository import FeatureRepository
from .database import WorkerDatabase

SQL = Path(__file__).with_name('migrations') / '006_intelligence.sql'


def apply(database):
    checksum = hashlib.sha256(SQL.read_bytes()).hexdigest()
    with database.transaction() as cursor:
        cursor.execute('SELECT pg_advisory_xact_lock(6381402501)')
        FeatureRepository.require_schema(cursor)
        cursor.execute("SELECT to_regclass('liquidity.intelligence_schema_migrations') AS ledger")
        if cursor.fetchone()['ledger'] is not None:
            cursor.execute('SELECT version,sha256 FROM liquidity.intelligence_schema_migrations ORDER BY version')
            if cursor.fetchall() != [{'version': SQL.stem, 'sha256': checksum}]:
                raise ValueError('Intelligence migration checksum mismatch')
            return []
        cursor.execute(SQL.read_text(), prepare=False)
        cursor.execute('INSERT INTO liquidity.intelligence_schema_migrations(version,sha256) VALUES (%s,%s)', (SQL.stem, checksum))
    return [SQL.stem]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.apply:
        database = WorkerDatabase()
        try:
            print('Applied', apply(database))
        finally:
            database.close()
    else:
        print(SQL.name, hashlib.sha256(SQL.read_bytes()).hexdigest())
