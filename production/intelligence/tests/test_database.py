import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import psycopg

from production.intelligence.database import WorkerDatabase
from production.pipeline.database import DatabaseError, IdentityError


class WorkerTransportTests(unittest.TestCase):
    def setUp(self):
        with patch('production.intelligence.database.Database.__init__', return_value=None):
            self.database = WorkerDatabase()
        self.database.settings = SimpleNamespace(
            hostname='approved.example', database='neondb', username='test_user',
            password='dummy-not-a-credential', project_id='test_project',
            branch_id='test_branch', endpoint_id='test_endpoint',
            connect_timeout_seconds=1, statement_timeout_ms=1000, lock_timeout_ms=1000)

    def connection(self):
        connection = MagicMock()
        connection.closed = False
        connection.info.host = self.database.settings.hostname
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = dict(database='neondb', username='test_user',
            project_id='test_project', branch_id='test_branch', endpoint_id='test_endpoint')
        return connection, cursor

    def test_stale_setup_connection_is_replaced_before_yield(self):
        stale, first = self.connection()
        first.execute.side_effect = psycopg.OperationalError('lost connection')
        fresh, second = self.connection()
        self.database.connection = stale
        with patch('production.intelligence.database.psycopg.connect', return_value=fresh) as connect, \
             patch('production.intelligence.database.time.sleep'):
            with self.database.transaction(read_only=True) as cursor:
                self.assertIs(cursor, second)
                cursor.execute('SELECT 1')
        stale.close.assert_called_once()
        connect.assert_called_once()
        self.assertEqual(first.execute.call_count, 1)

    def test_wrong_project_identity_is_never_retried(self):
        connection, cursor = self.connection()
        cursor.fetchone.return_value['project_id'] = 'wrong_project'
        with patch('production.intelligence.database.psycopg.connect', return_value=connection) as connect, \
             patch('production.intelligence.database.time.sleep') as sleep:
            with self.assertRaises(IdentityError):
                with self.database.transaction():
                    self.fail('Unverified connection must never be yielded')
        connect.assert_called_once()
        sleep.assert_not_called()

    def test_body_failure_is_not_replayed(self):
        connection, _ = self.connection()
        calls = []
        with patch('production.intelligence.database.psycopg.connect', return_value=connection) as connect:
            with self.assertRaisesRegex(DatabaseError, 'transaction body or commit'):
                with self.database.transaction():
                    calls.append('write attempted')
                    raise psycopg.OperationalError('lost write acknowledgement')
        self.assertEqual(calls, ['write attempted'])
        connect.assert_called_once()

    def test_uncertain_commit_is_not_replayed(self):
        connection, _ = self.connection()
        connection.transaction.return_value.__exit__.side_effect = psycopg.OperationalError('lost commit acknowledgement')
        calls = []
        with patch('production.intelligence.database.psycopg.connect', return_value=connection) as connect:
            with self.assertRaises(DatabaseError):
                with self.database.transaction():
                    calls.append('write attempted')
        self.assertEqual(calls, ['write attempted'])
        connect.assert_called_once()

    def test_setup_retries_are_bounded_and_errors_do_not_expose_credentials(self):
        connections = [self.connection() for _ in range(3)]
        for _, cursor in connections:
            cursor.execute.side_effect = psycopg.OperationalError('dummy-not-a-credential')
        with patch('production.intelligence.database.psycopg.connect', side_effect=[c for c, _ in connections]) as connect, \
             patch('production.intelligence.database.time.sleep') as sleep:
            with self.assertRaises(DatabaseError) as failure:
                with self.database.transaction():
                    self.fail('Failed setup must not yield a cursor')
        self.assertEqual(connect.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertNotIn('dummy-not-a-credential', str(failure.exception))
