import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import psycopg

from production.pipeline.config import ConfigurationError, ROOT, Settings, load_settings
from production.pipeline.database import Database, DatabaseError, IdentityError


def settings():
    config = json.loads((ROOT / "production/pipeline/config.json").read_text())
    return Settings(**config, username="test_role", password="SYNTHETIC_SECRET")


class ConfigurationTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory(prefix="liquidity-config-test-") as directory:
            root = Path(directory)
            (root / ".env.local").write_text(text)
            return load_settings(root)

    def test_local_file_only_and_password_repr(self):
        host = settings().hostname
        with patch.dict("os.environ", {"DATABASE_URL": "postgres://other-project", "LIQUIDITY_NEON_DATABASE_URL": "postgres://other-project"}):
            value = self.load(f'LIQUIDITY_NEON_DATABASE_URL="postgresql://role:SYNTHETIC_SECRET@{host}/neondb?sslmode=require"\n')
        self.assertEqual(value.username, "role")
        self.assertNotIn("SYNTHETIC_SECRET", repr(value))

    def test_wrong_endpoint_rejected_without_credentials_in_error(self):
        with self.assertRaises(ConfigurationError) as caught:
            self.load("LIQUIDITY_NEON_DATABASE_URL=postgresql://role:SYNTHETIC_SECRET@other-project.neon.tech/neondb")
        self.assertNotIn("SYNTHETIC_SECRET", str(caught.exception))

    def test_missing_empty_and_duplicate_values_rejected(self):
        for text in ["", "LIQUIDITY_NEON_DATABASE_URL=", "LIQUIDITY_NEON_DATABASE_URL=x\nLIQUIDITY_NEON_DATABASE_URL=y"]:
            with self.subTest(text=text), self.assertRaises(ConfigurationError):
                self.load(text)

    def test_configuration_cannot_retarget_another_project(self):
        config = json.loads((ROOT / "production/pipeline/config.json").read_text())
        config["project_id"] = "other-project"
        with patch("production.pipeline.config.json.loads", return_value=config):
            with self.assertRaises(ConfigurationError):
                self.load("LIQUIDITY_NEON_DATABASE_URL=x")

    def test_override_database_port_and_service_parameters_rejected(self):
        host = settings().hostname
        for suffix in [f"{host}/other_database", f"{host}:5433/neondb", f"{host}/neondb?host=other-project", f"{host}/neondb?service=other-project"]:
            with self.subTest(target=suffix), self.assertRaises(ConfigurationError):
                self.load(f"LIQUIDITY_NEON_DATABASE_URL=postgres://role:SYNTHETIC_SECRET@{suffix}")


class ConnectionTests(unittest.TestCase):
    def fixtures(self, mismatch=None):
        config = settings()
        identity = {"database": config.database, "username": config.username,
                    "project_id": config.project_id, "branch_id": config.branch_id,
                    "endpoint_id": config.endpoint_id}
        if mismatch:
            identity[mismatch] = "UNEXPECTED_IDENTITY"
        connection = MagicMock()
        connection.info.host = config.hostname
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = identity
        return config, connection, cursor

    def test_identity_precedes_application_queries_and_tls_enforced(self):
        config, connection, cursor = self.fixtures()
        with patch("production.pipeline.database.load_settings", return_value=config), patch("production.pipeline.database.psycopg.connect", return_value=connection) as connect:
            with Database().transaction() as application:
                application.execute("SELECT 42")
        sql = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertIn("neon.project_id", sql[0])
        self.assertEqual(sql[-1], "SELECT 42")
        self.assertEqual(connect.call_args.kwargs["sslmode"], "verify-full")
        self.assertEqual(connect.call_args.kwargs["host"], config.hostname)
        self.assertIsNone(connect.call_args.kwargs["prepare_threshold"])
        connection.close.assert_called_once()

    def test_each_identity_mismatch_blocks_application_access(self):
        for field in ["database", "username", "project_id", "branch_id", "endpoint_id"]:
            config, connection, cursor = self.fixtures(field)
            with self.subTest(field=field), patch("production.pipeline.database.load_settings", return_value=config), patch("production.pipeline.database.psycopg.connect", return_value=connection):
                with self.assertRaises(IdentityError):
                    with Database().transaction():
                        self.fail("Application cursor exposed after identity mismatch")
            self.assertEqual(cursor.execute.call_count, 1)
            self.assertIs(connection.transaction.return_value.__exit__.call_args.args[0], IdentityError)
            connection.close.assert_called_once()

    def test_read_only_operation_uses_read_only_transaction(self):
        config, connection, cursor = self.fixtures()
        with patch("production.pipeline.database.load_settings", return_value=config), patch("production.pipeline.database.psycopg.connect", return_value=connection):
            with Database().transaction(read_only=True):
                pass
        self.assertTrue(connection.read_only)
        self.assertIn("neon.project_id", cursor.execute.call_args_list[0].args[0])

    def test_driver_failures_are_sanitized(self):
        with patch("production.pipeline.database.load_settings", return_value=settings()), patch("production.pipeline.database.psycopg.connect", side_effect=psycopg.OperationalError("SYNTHETIC_SECRET")):
            with self.assertRaises(DatabaseError) as caught:
                with Database().transaction():
                    pass
        self.assertNotIn("SYNTHETIC_SECRET", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_application_exception_rolls_back_and_closes(self):
        config, connection, cursor = self.fixtures()
        with patch("production.pipeline.database.load_settings", return_value=config), patch("production.pipeline.database.psycopg.connect", return_value=connection):
            with self.assertRaises(ValueError):
                with Database().transaction():
                    raise ValueError("synthetic application failure")
        self.assertIs(connection.transaction.return_value.__exit__.call_args.args[0], ValueError)
        connection.close.assert_called_once()
