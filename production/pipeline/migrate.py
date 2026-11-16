"""Plan offline, or explicitly apply versioned SQL after deployment approval."""

import argparse
import hashlib
from pathlib import Path

from .database import Database

MIGRATIONS = Path(__file__).with_name("migrations")


class MigrationError(RuntimeError):
    pass


def migration_plan(*, include_features=False):
    paths = list(MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql"))
    if include_features:
        feature_directory = MIGRATIONS.parent.parent / "features" / "migrations"
        paths.extend(feature_directory.glob("[0-9][0-9][0-9]_*.sql"))
    if len({path.stem for path in paths}) != len(paths):
        raise MigrationError("Duplicate migration versions")
    return [
        {"version": path.stem, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
         "sql": path.read_text(encoding="utf-8")}
        for path in sorted(paths, key=lambda item: item.stem)
    ]


def verify_history(rows, *, require_core=True, require_features=False):
    """Verify every applied checksum and the explicitly required migration scope."""
    known = migration_plan(include_features=True)
    actual = [{"version": row["version"], "sha256": row["sha256"]} for row in rows]
    expected = [{"version": row["version"], "sha256": row["sha256"]} for row in known]
    if actual != expected[:len(actual)]:
        raise MigrationError("Migration history differs from reviewed SQL or contains a gap")
    required = known if require_features else migration_plan()
    if require_core and len(actual) < len(required):
        raise MigrationError("Required migration scope is not installed")


def apply_migrations(database, *, include_features=False):
    """Apply all pending versions atomically under a transaction-scoped lock."""
    plan = migration_plan(include_features=include_features)
    applied = []
    with database.transaction() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(6381402501)")
        cursor.execute(plan[0]["sql"], prepare=False)
        cursor.execute("SELECT version, sha256 FROM liquidity.schema_migrations ORDER BY version")
        rows = cursor.fetchall()
        verify_history(rows, require_core=False)
        history = {row["version"]: row["sha256"] for row in rows}
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
    parser.add_argument("--include-features", action="store_true", help="Include separately approved feature migration 005")
    args = parser.parse_args()
    if args.apply:
        print("Applied versions", apply_migrations(Database(), include_features=args.include_features))
    else:
        for item in migration_plan(include_features=args.include_features):
            print(item["version"], item["sha256"])


if __name__ == "__main__":
    main()
