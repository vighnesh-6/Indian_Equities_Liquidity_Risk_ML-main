"""Versioned feature persistence through the approved PostgreSQL connection."""

import hashlib
import json
from uuid import uuid4

import numpy as np
import pandas as pd

from production.pipeline.migrate import verify_history
from .contract import CONTRACT, FEATURES, FLAGS, calculation_identity, calculation_version, implementation_hash
from .engine import frame_hash


def source_version(cursor):
    cursor.execute("SELECT max(generation_id) AS generation FROM liquidity.price_basis_generations WHERE state='ACTIVE'")
    return "canonical-basis-" + str(cursor.fetchone()["generation"] or 0)


def validate_frame(frame):
    required = {"ticker", "date", *FEATURES, *FLAGS}
    if frame.empty or not required.issubset(frame) or frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Nonempty unique dated feature observations are required")
    if frame.ticker.isna().any() or frame.date.isna().any():
        raise ValueError("Feature keys cannot be null")
    if any(frame[name].dtype != np.dtype("float32") for name in FEATURES):
        raise ValueError("All 58 engineered columns must be float32")
    if np.isinf(frame[list(FEATURES)].to_numpy()).any():
        raise ValueError("Infinite feature values")
    for flag in FLAGS:
        if frame[flag].dtype != np.dtype("bool") or frame[flag].isna().any():
            raise ValueError("Quality flags must be non-null booleans")
    dates = pd.to_datetime(frame.date)
    if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Feature keys require normalized trading dates")
    if dates.eq(pd.Timestamp("2026-11-08")).any():
        raise ValueError("Excluded production session")


class FeatureRepository:
    def __init__(self, database):
        self.database = database

    @staticmethod
    def require_schema(cursor):
        cursor.execute("SELECT version,sha256 FROM liquidity.schema_migrations ORDER BY version")
        verify_history(cursor.fetchall(), require_features=True)

    def register_version(self, source):
        version = calculation_version(source)
        with self.database.transaction() as cursor:
            self.require_schema(cursor)
            if source_version(cursor) != source:
                raise ValueError("Canonical input vintage changed before registration")
            cursor.execute(
                "INSERT INTO liquidity.feature_versions(feature_version,calculation_sha256,implementation_sha256,source_version,feature_names,contract) "
                "VALUES (%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING",
                (version, calculation_identity(source), implementation_hash(), source, list(FEATURES), json.dumps(CONTRACT)))
            cursor.execute("SELECT calculation_sha256 FROM liquidity.feature_versions WHERE feature_version=%s", (version,))
            if cursor.fetchone()["calculation_sha256"] != calculation_identity(source):
                raise ValueError("Compact feature version collides with a different complete SHA-256 identity")
        return version

    def load_state(self, ticker, version):
        from .state import FeatureState
        with self.database.transaction(read_only=True) as cursor:
            self.require_schema(cursor)
            cursor.execute("SELECT state FROM liquidity.feature_checkpoints WHERE ticker=%s AND feature_version=%s",
                           (ticker, version))
            row = cursor.fetchone()
        return FeatureState.from_json(json.dumps(row["state"])) if row else None

    def persist(self, frame, states, *, version, request_key, input_sha256, provenance, publish=False):
        validate_frame(frame)
        frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
        tickers = sorted(frame.ticker.unique())
        if set(states) != set(tickers):
            raise ValueError("Every processed instrument requires its recursive checkpoint state")
        state_data = {ticker: json.loads(state.to_json()) for ticker, state in states.items()}
        payload = {"values": frame_hash(frame), "flags": frame[list(FLAGS)].to_numpy().tolist(),
                   "states": state_data, "version": version}
        output_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        days = pd.to_datetime(frame.date).dt.date
        run_id = uuid4()
        with self.database.transaction() as cursor:
            self.require_schema(cursor)
            cursor.execute("SELECT pg_advisory_xact_lock(6381402502)")
            cursor.execute("SELECT pg_advisory_xact_lock(6381402503)")
            cursor.execute("SELECT source_version FROM liquidity.feature_versions WHERE feature_version=%s", (version,))
            registered = cursor.fetchone()
            if registered is None or registered["source_version"] != source_version(cursor):
                raise ValueError("Feature version is absent or its canonical source vintage changed")
            if any(state.source_version != registered["source_version"] for state in states.values()):
                raise ValueError("Recursive state vintage differs from the registered feature version")
            cursor.execute("SELECT run_id,status,input_sha256,output_sha256 FROM liquidity.feature_runs WHERE request_key=%s",
                           (request_key,))
            existing = cursor.fetchone()
            if existing:
                if existing["status"] != "SUCCESS" or existing["input_sha256"] != input_sha256 or existing["output_sha256"] != output_hash:
                    raise ValueError("Idempotency key conflicts with committed feature evidence")
                return {"run_id": str(existing["run_id"]), "replayed": True, "rows": len(frame)}
            cursor.execute(
                "INSERT INTO liquidity.feature_runs(run_id,request_key,feature_version,range_start,range_end,expected_tickers,"
                "expected_rows,input_sha256,output_sha256,provenance,status) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,'RUNNING')",
                (run_id, request_key, version, min(days), max(days), tickers, len(frame), input_sha256, output_hash,
                 json.dumps(provenance, sort_keys=True, allow_nan=False)))
            columns = ["ticker", "date", "feature_version", "run_id", *FLAGS, *FEATURES]
            rows = []
            for record in frame.to_dict("records"):
                values = [None if pd.isna(record[name]) else float(record[name]) for name in FEATURES]
                rows.append((record["ticker"], pd.Timestamp(record["date"]).date(), version, run_id,
                             *[bool(record[name]) for name in FLAGS], *values))
            cursor.executemany("INSERT INTO liquidity.feature_values(" + ",".join(columns) + ") VALUES ("
                               + ",".join(["%s"] * len(columns)) + ")", rows)
            for ticker, state in states.items():
                if state.ticker != ticker or state.last_date != str(max(days[frame.ticker.eq(ticker)])):
                    raise ValueError("Recursive checkpoint endpoint differs from feature observations")
                cursor.execute(
                    "INSERT INTO liquidity.feature_checkpoints(ticker,feature_version,last_date,state) VALUES (%s,%s,%s,%s::jsonb) "
                    "ON CONFLICT(ticker,feature_version) DO UPDATE SET last_date=excluded.last_date,state=excluded.state",
                    (ticker, version, state.last_date, state.to_json()))
            cursor.execute("UPDATE liquidity.feature_runs SET status='SUCCESS',finished_at=now() WHERE run_id=%s", (run_id,))
            if publish:
                for day in sorted(set(days)):
                    cursor.execute("INSERT INTO liquidity.feature_publications(date,feature_version) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                                   (day, version))
        return {"run_id": str(run_id), "replayed": False, "rows": len(frame)}

    def publish(self, version, days):
        with self.database.transaction() as cursor:
            self.require_schema(cursor)
            cursor.execute("SELECT pg_advisory_xact_lock(6381402502)")
            cursor.execute("SELECT pg_advisory_xact_lock(6381402503)")
            cursor.executemany("INSERT INTO liquidity.feature_publications(date,feature_version) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                               [(day, version) for day in days])
