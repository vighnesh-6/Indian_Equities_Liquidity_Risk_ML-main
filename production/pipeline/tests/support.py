"""Test-only DB-API adapter around a separate in-memory PGlite process."""

import json
import os
import subprocess
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from uuid import UUID


class SqlError(RuntimeError):
    pass


class MemoryDatabase:
    def __init__(self):
        if not os.environ.get("LIQUIDITY_TEST_PGLITE_MODULE"):
            raise RuntimeError("Set LIQUIDITY_TEST_PGLITE_MODULE to the temporary PGlite installation")
        self.process = subprocess.Popen(
            ["node", str(Path(__file__).with_name("pglite_bridge.mjs"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8",
        )
        ready = json.loads(self.process.stdout.readline())
        assert ready["ready"]
        self.version = ready["version"]
        self.rows = []
        self.fail_on_write = None
        self.write_count = 0

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=20)
        self.process.stdout.close()
        self.process.stderr.close()

    def execute(self, sql, params=(), *, prepare=None):
        if sql.startswith("INSERT INTO liquidity.market_eod"):
            self.write_count += 1
            if self.write_count == self.fail_on_write:
                raise SqlError("INJECTED_FAILURE")
        if prepare is not False:
            fragments = sql.split("%s")
            sql = "".join(part + (f"${index + 1}" if index < len(fragments) - 1 else "")
                          for index, part in enumerate(fragments))
        request = {"sql": sql, "params": params, "multi": prepare is False}
        self.process.stdin.write(json.dumps(request, default=str, allow_nan=False) + "\n")
        self.process.stdin.flush()
        result = json.loads(self.process.stdout.readline())
        if not result["ok"]:
            raise SqlError(f"{result['code']}: {result.get('message', '')} (position {result.get('position', '')})")
        data = result["result"]
        if isinstance(data, list):
            self.rows = []
            return self
        self.rows = data.get("rows", [])
        types = {item["name"]: item["dataTypeID"] for item in data.get("fields", [])}
        for row in self.rows:
            for name, value in row.items():
                if value is None:
                    continue
                if types.get(name) == 1082:
                    row[name] = date.fromisoformat(value[:10])
                elif types.get(name) == 1182:
                    row[name] = [date.fromisoformat(item[:10]) for item in value]
                elif types.get(name) == 2950:
                    row[name] = UUID(value)
                elif types.get(name) == 1184:
                    row[name] = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return self

    def executemany(self, sql, rows):
        for row in rows:
            self.execute(sql, row)

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def fetchall(self):
        rows, self.rows = self.rows, []
        return rows

    @contextmanager
    def transaction(self, *, read_only=False):
        self.execute("BEGIN READ ONLY" if read_only else "BEGIN")
        try:
            yield self
            self.execute("COMMIT")
        except Exception:
            self.execute("ROLLBACK")
            raise
