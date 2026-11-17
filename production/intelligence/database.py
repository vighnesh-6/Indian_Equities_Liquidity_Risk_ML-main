"""Identity-checked persistent transport for the bounded production worker."""

import time
from contextlib import contextmanager

import certifi
import psycopg
from psycopg.rows import dict_row

from production.pipeline.database import Database, DatabaseError, IDENTITY_SQL, IdentityError


class WorkerDatabase(Database):
    """Reuse TLS while preserving target verification for every transaction."""

    def __init__(self):
        super().__init__()
        self.connection = None

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    @contextmanager
    def transaction(self, *, read_only=False):
        settings = self.settings
        try:
            if self.connection is None or self.connection.closed:
                self.connection = psycopg.connect(
                    host=settings.hostname, port=5432, dbname=settings.database,
                    user=settings.username, password=settings.password,
                    sslmode='verify-full', sslrootcert=certifi.where(), channel_binding='require',
                    passfile=__import__('os').devnull, sslcertmode='disable',
                    connect_timeout=settings.connect_timeout_seconds, autocommit=True,
                    prepare_threshold=None, row_factory=dict_row,
                    application_name='liquidity-risk-production', keepalives=1,
                    keepalives_idle=15, keepalives_interval=5, keepalives_count=3,
                    tcp_user_timeout=30000)
            connection = self.connection
            if connection.info.host != settings.hostname:
                raise IdentityError('Approved endpoint mismatch')
            connection.read_only = read_only
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(IDENTITY_SQL)
                    expected = dict(database=settings.database, username=settings.username,
                                    project_id=settings.project_id, branch_id=settings.branch_id,
                                    endpoint_id=settings.endpoint_id)
                    if cursor.fetchone() != expected:
                        raise IdentityError('Approved Liquidity Risk server identity mismatch')
                    cursor.execute("SELECT set_config('search_path','pg_catalog',true),set_config('statement_timeout',%s,true),set_config('lock_timeout',%s,true)",
                                   (str(settings.statement_timeout_ms), str(settings.lock_timeout_ms)))
                    yield cursor
        except psycopg.Error as error:
            self.close()
            raise DatabaseError('PostgreSQL operation failed (' + (error.sqlstate or 'CONNECTION_FAILURE') + ')') from None
        except BaseException:
            self.close()
            raise


def retry(operation, *, attempts=5):
    for attempt in range(attempts):
        try:
            return operation()
        except DatabaseError as error:
            if attempt == attempts-1 or not any(code in str(error) for code in ['CONNECTION_FAILURE', '55P03', '40001', '40P01']):
                raise
            time.sleep(min(2**attempt, 8))
