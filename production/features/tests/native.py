"""Disposable PostgreSQL 18 test cluster bound exclusively to localhost."""

from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import threading
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
import support

ACTIVE_SERVER = None


class LocalPostgres:
    def __init__(self, binaries):
        self.binaries = Path(binaries)
        self.root = Path(tempfile.mkdtemp(prefix="liquidity-phase3-offline-"))
        self.data = self.root / "cluster"
        self.user = "liquidity_phase3_test"
        self.password = secrets.token_urlsafe(32)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    def __enter__(self):
        global ACTIVE_SERVER
        if os.name == "nt":
            ctypes.windll.kernel32.SetErrorMode(3)
        password_file = self.root / "init-password"
        password_file.write_text(self.password, encoding="utf-8")
        try:
            subprocess.run([str(self.binaries / "initdb.exe"), "-D", str(self.data), "-U", self.user,
                            "--pwfile=" + str(password_file), "--auth=scram-sha-256", "--encoding=UTF8", "--locale=C"],
                           check=True, capture_output=True, creationflags=self.flags)
        finally:
            password_file.unlink()
        options = f"-h 127.0.0.1 -p {self.port} -c max_connections=30 -c cluster_name=liquidity_phase3_isolated"
        subprocess.run([str(self.binaries / "pg_ctl.exe"), "-D", str(self.data), "-l", str(self.root / "postgres.log"),
                        "-o", options, "-w", "start"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True, timeout=45, creationflags=self.flags)
        ACTIVE_SERVER = self
        return self

    def connect(self, database="postgres"):
        connection = psycopg.connect(host="127.0.0.1", port=self.port, user=self.user,
                                     password=self.password, dbname=database, sslmode="disable",
                                     passfile=os.devnull, autocommit=True, row_factory=dict_row)
        identity = connection.execute("SELECT current_setting('data_directory') AS directory,"
                                      "current_setting('server_version_num')::int AS version").fetchone()
        if Path(identity["directory"]).resolve() != self.data.resolve() or not 180000 <= identity["version"] < 190000:
            connection.close()
            raise RuntimeError("Isolated test cluster identity differs")
        return connection

    def __exit__(self, *args):
        global ACTIVE_SERVER
        ACTIVE_SERVER = None
        subprocess.run([str(self.binaries / "pg_ctl.exe"), "-D", str(self.data), "-m", "fast", "-w", "stop"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
                       timeout=35, creationflags=self.flags)


class NativeDatabase:
    def __init__(self, name=None):
        if ACTIVE_SERVER is None:
            raise RuntimeError("Offline test server is not running")
        self.server = ACTIVE_SERVER
        self.owned = name is None
        self.name = name or "liquidity_phase3_" + uuid4().hex
        if not self.name.startswith("liquidity_phase3_"):
            raise ValueError("Unexpected isolated test database")
        if self.owned:
            with self.server.connect() as connection:
                connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.name)))
        self.local = threading.local()
        self.fail_on_write = None
        self.write_count = 0
        self.after_advisory = self.before_commit = self.after_write = None
        self.pids = set()
        self.pid_lock = threading.Lock()
        self.pooled_connections = []

    def peer(self):
        return NativeDatabase(self.name)

    def connect(self):
        connection = self.server.connect(self.name)
        connection.execute("SET statement_timeout='30s'")
        connection.execute("SET lock_timeout='15s'")
        with self.pid_lock:
            self.pids.add(connection.info.backend_pid)
        return connection

    def execute(self, query, params=(), *, prepare=None):
        connection = getattr(self.local, "connection", None)
        own = connection is None
        if own:
            connection = self.connect()
        try:
            if query.startswith("INSERT INTO liquidity.market_eod"):
                self.write_count += 1
                if self.write_count == self.fail_on_write:
                    raise support.SqlError("INJECTED_FAILURE")
            with connection.cursor() as cursor:
                cursor.execute(query, params or None, prepare=prepare)
                self.local.rows = cursor.fetchall() if cursor.description else []
            if query.startswith("SELECT pg_advisory_xact_lock") and self.after_advisory:
                self.after_advisory()
            if query.startswith("INSERT INTO liquidity.market_eod") and self.after_write:
                self.after_write()
            return self
        except psycopg.Error as error:
            raise support.SqlError(f"{error.sqlstate}: {error.diag.message_primary}") from None
        finally:
            if own:
                connection.close()

    def executemany(self, query, rows):
        connection = getattr(self.local, "connection", None)
        if connection is not None and query.startswith("INSERT INTO liquidity.feature_values"):
            try:
                with connection.cursor() as cursor:
                    cursor.executemany(query, rows)
            except psycopg.Error as error:
                raise support.SqlError(f"{error.sqlstate}: {error.diag.message_primary}") from None
        else:
            for row in rows:
                self.execute(query, row)

    def fetchone(self):
        rows = getattr(self.local, "rows", [])
        return rows.pop(0) if rows else None

    def fetchall(self):
        rows = getattr(self.local, "rows", [])
        self.local.rows = []
        return rows

    @contextmanager
    def transaction(self, *, read_only=False):
        # Each test-database instance owns one connection per worker thread.
        # Peers therefore retain genuinely independent PostgreSQL backends.
        connection = getattr(self.local, "pooled", None)
        if connection is None or connection.closed:
            connection = self.connect()
            self.local.pooled = connection
            with self.pid_lock:
                self.pooled_connections.append(connection)
        self.local.connection = connection
        try:
            connection.read_only = read_only
            with connection.transaction():
                yield self
                if self.before_commit:
                    self.before_commit()
        except psycopg.Error as error:
            raise support.SqlError(f"{error.sqlstate}: {error.diag.message_primary}") from None
        finally:
            self.local.connection = None

    def close(self):
        for connection in self.pooled_connections:
            connection.close()
        if self.owned:
            with self.server.connect() as connection:
                connection.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(self.name)))
