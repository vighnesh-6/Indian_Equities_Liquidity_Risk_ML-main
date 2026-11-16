"""Identity-checked PostgreSQL transactions with sanitized failures."""

import os
from contextlib import contextmanager

import certifi
import psycopg
from psycopg.rows import dict_row

from .config import load_settings


class DatabaseError(RuntimeError):
    """Sanitized database error safe for application logs."""


class IdentityError(DatabaseError):
    """The connected backend does not match the configured target."""


IDENTITY_SQL = """
SELECT pg_catalog.current_database() AS database,
       current_user AS username,
       pg_catalog.current_setting('neon.project_id', true) AS project_id,
       pg_catalog.current_setting('neon.branch_id', true) AS branch_id,
       pg_catalog.current_setting('neon.endpoint_id', true) AS endpoint_id
"""


class Database:
    def __init__(self):
        self.settings = load_settings()

    @contextmanager
    def transaction(self, *, read_only=False):
        """Verify server identity before yielding any application cursor."""
        connection = None
        try:
            settings = self.settings
            connection = psycopg.connect(
                host=settings.hostname, port=5432, dbname=settings.database,
                user=settings.username, password=settings.password, passfile=os.devnull,
                sslmode="verify-full", sslrootcert=certifi.where(), sslcertmode="disable",
                channel_binding="require", connect_timeout=settings.connect_timeout_seconds,
                application_name="liquidity-risk-production", autocommit=True,
                prepare_threshold=None, row_factory=dict_row,
            )
            if connection.info.host != settings.hostname:
                raise IdentityError("Connected endpoint differs from the approved hostname")
            connection.read_only = read_only
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(IDENTITY_SQL)
                    actual = cursor.fetchone()
                    expected = {
                        "database": settings.database, "project_id": settings.project_id,
                        "username": settings.username,
                        "branch_id": settings.branch_id, "endpoint_id": settings.endpoint_id,
                    }
                    if actual != expected:
                        raise IdentityError("Server identity differs from the approved Liquidity Risk mapping")
                    cursor.execute("SELECT pg_catalog.set_config('search_path', 'pg_catalog', true)")
                    cursor.execute("SELECT set_config('statement_timeout', %s, true)", (str(settings.statement_timeout_ms),))
                    cursor.execute("SELECT set_config('lock_timeout', %s, true)", (str(settings.lock_timeout_ms),))
                    yield cursor
        except psycopg.Error as error:
            code = error.sqlstate or "CONNECTION_FAILURE"
            raise DatabaseError(f"PostgreSQL operation failed ({code})") from None
        finally:
            if connection is not None:
                connection.close()
