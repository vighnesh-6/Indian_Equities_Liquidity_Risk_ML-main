import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from production.pipeline.config import ROOT


class DailyWorkflowTests(unittest.TestCase):
    def test_production_entry_point_disallows_historical_basis_refresh(self):
        from production.pipeline.catchup import run_cli
        with tempfile.TemporaryDirectory(prefix='liquidity-recovery-cli-') as directory, \
             patch('production.pipeline.catchup.Database'), \
             patch('production.pipeline.catchup.YahooProvider'), \
             patch('production.pipeline.catchup.SessionIngestor') as collector, \
             patch('sys.argv', ['daily', '--report', str(Path(directory) / 'audit.json')]), \
             patch('builtins.print'):
            collector.return_value.collect.return_value = {'status': 'NOOP'}
            run_cli('daily')
            self.assertFalse(collector.return_value.collect.call_args.kwargs['allow_basis_refresh'])

    def test_schedule_permissions_secret_and_entry_point(self):
        path = ROOT / '.github/workflows/liquidity-risk-daily-ingestion.yml'
        text = path.read_text(encoding='utf-8')
        workflow = yaml.load(text, Loader=yaml.BaseLoader)
        self.assertEqual(workflow['on']['schedule'][0]['cron'], '30 10 * * 1-5')
        self.assertIn('workflow_dispatch', workflow['on'])
        self.assertEqual(workflow['permissions'], {'contents': 'read'})
        self.assertEqual(workflow['concurrency']['cancel-in-progress'], 'false')
        job = workflow['jobs']['collect']
        self.assertEqual(job['runs-on'], 'ubuntu-24.04')
        secret_steps = [step for step in job['steps'] if 'LIQUIDITY_NEON_DATABASE_URL' in step.get('env', {})]
        self.assertEqual(
            {step['name'] for step in secret_steps},
            {'Collect and validate complete NSE sessions',
             'Features, matured labels, frozen inference and risk intelligence'},
        )
        self.assertEqual(len(secret_steps), 2)
        for step in secret_steps:
            self.assertEqual(step['env']['LIQUIDITY_NEON_DATABASE_URL'], '${{ secrets.LIQUIDITY_NEON_DATABASE_URL }}')
        self.assertIn('-m production.pipeline.reconcile', secret_steps[0]['run'])
        self.assertIn('-m production.intelligence.daily', secret_steps[1]['run'])
        self.assertIn('${LIQUIDITY_NEON_DATABASE_URL:-}', secret_steps[0]['run'])
        self.assertIn('missing or empty', secret_steps[0]['run'])
        self.assertNotIn('DATABASE_URL:', text.replace('LIQUIDITY_NEON_DATABASE_URL:', ''))
        for step in job['steps']:
            if 'uses' in step:
                self.assertRegex(step['uses'], r'@[0-9a-f]{40}$')
        self.assertIn('--require-hashes', text)
        self.assertNotIn('pull_request', workflow['on'])
