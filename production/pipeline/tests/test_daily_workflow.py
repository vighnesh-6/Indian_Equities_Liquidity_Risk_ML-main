import unittest

import yaml

from production.pipeline.config import ROOT


class DailyWorkflowTests(unittest.TestCase):
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
        self.assertEqual(len(secret_steps), 1)
        self.assertEqual(secret_steps[0]['env']['LIQUIDITY_NEON_DATABASE_URL'], '${{ secrets.LIQUIDITY_NEON_DATABASE_URL }}')
        self.assertIn('-m production.pipeline.daily', secret_steps[0]['run'])
        self.assertNotIn('DATABASE_URL:', text.replace('LIQUIDITY_NEON_DATABASE_URL:', ''))
        for step in job['steps']:
            if 'uses' in step:
                self.assertRegex(step['uses'], r'@[0-9a-f]{40}$')
        self.assertIn('--require-hashes', text)
        self.assertNotIn('pull_request', workflow['on'])
