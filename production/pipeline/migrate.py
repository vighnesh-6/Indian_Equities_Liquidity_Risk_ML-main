"""Plan offline, or explicitly apply versioned SQL after deployment approval."""

import argparse
import hashlib
from pathlib import Path

from .database import Database

MIGRATIONS = Path(__file__).with_name("migrations")


class MigrationError(RuntimeError):
    pass


def migration_plan():
    return [
        {"version": path.stem, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
         "sql": path.read_text(encoding="utf-8")}
        for path in sorted(MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql"))
    ]


def apply_migrations(database):
    """Apply all pending versions atomically under a transaction-scoped lock."""
    plan = migration_plan()
    applied = []
    with database.transaction() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(6381402501)")
        cursor.execute(plan[0]["sql"], prepare=False)
        cursor.execute("SELECT version, sha256 FROM liquidity.schema_migrations ORDER BY version")
        history = {row["version"]: row["sha256"] for row in cursor.fetchall()}
        expected = {item["version"]: item["sha256"] for item in plan}
        if any(version not in expected or expected[version] != checksum for version, checksum in history.items()):
            raise MigrationError("Migration history differs from the reviewed SQL files")
        if history and list(history) != list(expected)[:len(history)]:
            raise MigrationError("Migration history contains a version gap")
        for item in plan:
            if item["version"] in history:
                continue
            if item is not plan[0]:
                cursor.execute(item["sql"], prepare=False)
            cursor.execute(
                "INSERT INTO liquidity.schema_migrations(version, sha256) VALUES (%s, %s)",
                (item["version"], item["sha256"]),
            )
            applied.append(item["version"])
    return applied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply reviewed SQL to the verified Liquidity Risk database")
    args = parser.parse_args()
    if args.apply:
        print("Applied versions", apply_migrations(Database()))
    else:
        for item in migration_plan():
            print(item["version"], item["sha256"])


if __name__ == "__main__":
    main()
