"""Measure actual PostgreSQL storage using a representative historical backfill."""

import json
import math
import pandas as pd

from production.pipeline.migrate import apply_migrations
from production.features.contract import ROOT
from production.features.engine import input_hash
from production.features.audit import reconcile
from production.features.state import initialize
from production.features.repository import FeatureRepository
from production.features.tests.native import NativeDatabase
from production.features.tests.test_persistence import seed


def validate_storage():
    universe = json.loads((ROOT / "configs/universe.json").read_text(encoding="utf-8"))
    selected = {entry["ticker"] for group in universe["tickers"].values() for entry in group[:5]}
    selected.update(["VAML.NS", "VOGL.NS", "VISL.NS", "FORCEMOT.NS", "PATANJALI.NS", "YESBANK.NS"])
    stock = pd.read_parquet(ROOT / "data/interim/stocks/ohlcv_clean.parquet", filters=[("ticker", "in", sorted(selected))])
    benchmark = pd.read_parquet(ROOT / "data/interim/market/nifty50_clean.parquet")
    vix = pd.read_parquet(ROOT / "data/interim/market/india_vix_clean.parquet")
    groups = {ticker: frame.reset_index(drop=True) for ticker, frame in stock.groupby("ticker")}
    db = NativeDatabase()
    try:
        apply_migrations(db, include_features=True)
        seed(db, groups)
        repository = FeatureRepository(db)
        version = repository.register_version("canonical-basis-0")
        features = []
        for ticker, group in groups.items():
            output, state = initialize(group, benchmark, vix, source_version="canonical-basis-0")
            repository.persist(output, {ticker: state}, version=version, request_key="storage-" + ticker,
                               input_sha256=input_hash(group), provenance={"purpose": "isolated historical storage validation"})
            features.append(output)
        combined = pd.concat(features, ignore_index=True)
        repository.publish(version, sorted(combined.date.dt.date.unique()))
        verdict = reconcile(db, combined, version)
        db.execute("SELECT relname,pg_total_relation_size(oid) AS bytes FROM pg_class WHERE relnamespace='liquidity'::regnamespace "
                   "AND relkind='r' AND relname LIKE 'feature_%' ORDER BY relname")
        sizes = {r["relname"]: r["bytes"] for r in db.fetchall()}
        projected = math.ceil(sizes["feature_values"] / len(combined) * 569690)
        checkpoints = math.ceil(sizes["feature_checkpoints"] / len(groups) * 300)
        other = sum(value for name, value in sizes.items() if name not in ["feature_values", "feature_checkpoints"])
        baseline = 127860736
        return {"result": verdict["result"], "sample_rows": len(combined), "sample_tickers": len(groups),
                "date_start": str(combined.date.min().date()), "date_end": str(combined.date.max().date()),
                "reconciliation": verdict, "actual_relation_bytes": sizes,
                "projected_full_feature_values_bytes": projected,
                "projected_300_checkpoint_bytes": checkpoints,
                "projected_database_bytes": baseline + projected + checkpoints + other,
                "projected_database_mib": round((baseline + projected + checkpoints + other) / 2**20, 2),
                "production_baseline_bytes_from_approved_read_only_plan": baseline,
                "limitation": "Extrapolation from actual representative PostgreSQL 18 heap, TOAST and indexes. Future versions, WAL, branch retention and later phases require additional capacity."}
    finally:
        db.close()
