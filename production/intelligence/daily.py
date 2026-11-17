"""Connect market ingestion's completed checkpoint to published intelligence."""

import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

from production.features.run import process as process_features
from production.features.contract import calculation_version
from production.features.repository import source_version
from production.pipeline.ingestion import inspect_state
from .database import WorkerDatabase, retry
from .inference import run_inference
from .worker import active_version, run_labels


def run(database):
    checkpoints = retry(lambda: inspect_state(database))
    through = min(checkpoints.values())
    with database.transaction(read_only=True) as cursor:
        version = calculation_version(source_version(cursor))
        cursor.execute('SELECT count(*) AS n,min(last_date) AS earliest,max(last_date) AS latest FROM liquidity.feature_checkpoints WHERE feature_version=%s', (version,))
        current = cursor.fetchone()
        cursor.execute('SELECT count(*) AS n FROM liquidity.feature_publications WHERE feature_version=%s AND date=%s', (version, through))
        published = cursor.fetchone()['n'] == 1
    if current['n'] == 300 and current['earliest'] == through and current['latest'] == through and published:
        features = {'status': 'NOOP', 'feature_version': str(version), 'rows': 0}
    else:
        features = retry(lambda: process_features(database, through, incremental=True, persist=True,
                                                 progress=lambda _: None))
    labels = run_labels(database, incremental=True)
    predictions = run_inference(database, incremental=True)
    version, feature_date = active_version(database)
    with database.transaction() as cursor:
        cursor.execute('SELECT count(*) AS n FROM liquidity.intelligence_publications WHERE feature_version=%s AND date=%s AND predictions_at IS NOT NULL', (version, through))
        if cursor.fetchone()['n'] != 1:
            raise ValueError('Latest complete market session has no complete intelligence publication')
        result = {'status': 'SUCCESS', 'through': str(through), 'feature_version': str(version),
                  'features': features, 'labels': labels, 'inference': predictions,
                  'trigger': 'github_actions' if os.environ.get('GITHUB_ACTIONS') == 'true' else 'local'}
        cursor.execute("INSERT INTO liquidity.intelligence_runs(run_id,feature_version,stage,through_date,status,finished_at,rows_processed,details) VALUES (%s,%s,'daily',%s,'SUCCESS',clock_timestamp(),%s,%s::jsonb)",
                       (uuid4(), version, through, predictions.get('rows',0), json.dumps(result, default=str)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    database = WorkerDatabase()
    try:
        result = run(database)
    except Exception as error:
        result = {'status': 'FAILED', 'error_type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        database.close()
        if 'result' in locals():
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
            print(json.dumps(result, default=str), flush=True)


if __name__ == '__main__':
    main()
