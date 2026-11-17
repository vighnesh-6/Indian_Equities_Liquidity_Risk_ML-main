"""Start the local web app with the validated project-local credential in memory."""

import argparse
import os
import subprocess
from urllib.parse import quote

from production.pipeline.config import ROOT, load_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--production', action='store_true')
    parser.add_argument('--port', default='3000')
    args = parser.parse_args()
    settings = load_settings()
    environment = os.environ.copy()
    environment['LIQUIDITY_NEON_DATABASE_URL'] = (
        'postgresql://'+quote(settings.username, safe='')+':'+quote(settings.password, safe='')+'@'+settings.hostname+
        '/'+settings.database+'?sslmode=verify-full&channel_binding=require')
    environment['NEXT_TELEMETRY_DISABLED'] = '1'
    command = ['npm.cmd' if os.name == 'nt' else 'npm', 'run', 'start' if args.production else 'dev', '--', '--port', args.port]
    raise SystemExit(subprocess.call(command, cwd=ROOT/'production/web', env=environment))


if __name__ == '__main__':
    main()
