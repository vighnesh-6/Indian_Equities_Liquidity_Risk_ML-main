"""
Feature Engineering Orchestrator (Phase 5 — F01 to F55 + India VIX)
====================================================================

Coordinates computation of all 55 audited core features (F01-F55) plus the
3 supplementary India VIX features (Decision C1) cleanly per ticker, with
strict causal lookback isolation and float32 streaming Parquet output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa

from src.data.storage import StreamingParquetWriter
from src.data.universe import UniverseManager
from src.features.gaps import compute_gap_features
from src.features.liquidity import compute_liquidity_features
from src.features.market_relative import compute_market_relative_features
from src.features.market_stress import (
    CORE_MARKET_STRESS_COLUMNS,
    VIX_SUPPLEMENTARY_COLUMNS,
    compute_market_stress_features,
    prepare_benchmark_features,
    prepare_vix_features,
)
from src.features.returns import compute_returns_features
from src.features.volatility import compute_volatility_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

CORE_FEATURE_COLUMNS: tuple[str, ...] = (
    # Category 1: Returns & Price Behaviour (F01-F10)
    "daily_return",
    "log_return",
    "return_5d",
    "return_20d",
    "return_60d",
    "return_z5",
    "return_z20",
    "abs_return",
    "price_rel_ma20",
    "price_rel_ma60",
    # Category 2: Volatility & Downside Risk (F11-F22)
    "vol_5d",
    "vol_20d",
    "vol_60d",
    "vol_ratio_5_20",
    "parkinson_vol",
    "true_range",
    "atr_20d",
    "downside_vol_20d",
    "sortino_component",
    "max_drawdown_20d",
    "rsi_14",
    "vol_of_vol",
    # Category 3: Liquidity & Trading Activity (F23-F36)
    "log_volume",
    "volume_ma20",
    "volume_ratio",
    "volume_z20",
    "amihud_ratio",
    "amihud_ma20",
    "amihud_z20",
    "turnover_value",
    "log_turnover",
    "turnover_ma20",
    "turnover_ratio",
    "zero_return_flag",
    "zero_vol_flag",
    "low_volume_streak",
    # Category 4: Price Gap & Price Impact (F37-F45)
    "daily_range",
    "daily_range_ma20",
    "range_z20",
    "overnight_gap",
    "overnight_gap_abs",
    "gap_z20",
    "cs_spread_est",
    "cs_spread_ma10",
    "intraday_return",
    # Category 5: Market-Relative Behaviour (F46-F51)
    "excess_return_1d",
    "excess_return_20d",
    "rolling_beta_60d",
    "rolling_corr_60d",
    "idiosyncratic_vol",
    "relative_volume",
    # Category 6: Market Stress & Correlation (F52-F55)
    *CORE_MARKET_STRESS_COLUMNS,
)

QUALITY_FLAG_COLUMNS: tuple[str, ...] = (
    "is_zero_volume",
    "is_stale_price",
    "is_corporate_action_gap",
    "is_market_shock",
    "is_short_history",
    "is_warmup",
    "is_benchmark_available",
    "is_muhurat_session",
    "is_valid_ohlcv",
)


def _build_parquet_schema(
    columns: List[str],
    feature_cols: List[str],
    flag_cols: List[str],
    output_dtype: str,
) -> pa.Schema:
    """Builds an explicit PyArrow schema for deterministic streaming writes."""
    pa_float_type = pa.float32() if output_dtype == "float32" else pa.float64()
    feature_set = set(feature_cols)
    flag_set = set(flag_cols)

    fields: list[pa.Field] = []
    for col in columns:
        if col == "date":
            fields.append(pa.field("date", pa.timestamp("ns")))
        elif col in {"ticker", "cap_segment"}:
            fields.append(pa.field(col, pa.string()))
        elif col in flag_set:
            fields.append(pa.field(col, pa.bool_()))
        elif col in feature_set:
            fields.append(pa.field(col, pa_float_type))
        else:
            fields.append(pa.field(col, pa.string()))
    return pa.schema(fields)


class FeatureEngineer:
    """Orchestrates feature engineering across all 55 core features + India VIX."""

    def __init__(
        self,
        warmup_rows: int = 80,
        output_dtype: str = "float32",
        universe_config_path: str | Path = "configs/universe.json",
    ) -> None:
        if output_dtype not in {"float32", "float64"}:
            raise ValueError(f"Unsupported output_dtype: {output_dtype}")
        self.warmup_rows = warmup_rows
        self.output_dtype = output_dtype
        self.universe_config_path = Path(universe_config_path)
        self._segment_map: Optional[Dict[str, str]] = None

    def _get_segment_mapping(self) -> Dict[str, str]:
        """Lazily loads ticker -> cap_segment mapping from universe config."""
        if self._segment_map is None:
            if self.universe_config_path.exists():
                self._segment_map = UniverseManager(
                    config_path=self.universe_config_path
                ).get_segment_mapping()
            else:
                self._segment_map = {}
        return self._segment_map

    def compute_stock_features(
        self,
        stock_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Computes all 55 core features (and optional VIX features) for one stock.

        Args:
            stock_df: Single-stock OHLCV DataFrame (sorted chronologically by ``date``
                internally if needed).
            benchmark_df: NIFTY 50 OHLCV DataFrame (raw or prepared).
            vix_df: Optional India VIX DataFrame (raw or prepared).

        Returns:
            DataFrame containing metadata (``date``, ``ticker``, ``cap_segment``),
            retained Phase 4.2 quality flags, all 55 core features (F01-F55), and
            supplementary VIX features (when ``vix_df`` is provided), cast to
            ``self.output_dtype``.
        """
        if stock_df.empty:
            raise ValueError("stock_df must not be empty.")

        sorted_stock = stock_df.sort_values("date", kind="mergesort").copy()
        bench_prep = prepare_benchmark_features(benchmark_df)
        vix_prep = prepare_vix_features(vix_df) if vix_df is not None else None

        cat1 = compute_returns_features(sorted_stock)
        cat2 = compute_volatility_features(sorted_stock)
        cat3 = compute_liquidity_features(sorted_stock)
        cat4 = compute_gap_features(sorted_stock)
        cat5 = compute_market_relative_features(sorted_stock, bench_prep)
        cat6 = compute_market_stress_features(sorted_stock, bench_prep, vix_df=vix_prep)

        meta = pd.DataFrame(index=sorted_stock.index)
        meta["date"] = pd.to_datetime(sorted_stock["date"])
        if getattr(meta["date"].dt, "tz", None) is not None:
            meta["date"] = meta["date"].dt.tz_localize(None)

        if "ticker" in sorted_stock.columns:
            tickers = sorted_stock["ticker"].astype(str)
            meta["ticker"] = tickers
            seg_map = self._get_segment_mapping()
            meta["cap_segment"] = tickers.map(seg_map).fillna("unknown").astype(str)

        present_flags: list[str] = [
            col for col in QUALITY_FLAG_COLUMNS if col in sorted_stock.columns
        ]
        for col in present_flags:
            meta[col] = sorted_stock[col].fillna(False).astype(bool)

        if "is_warmup" not in meta.columns:
            meta["is_warmup"] = np.arange(len(sorted_stock)) < self.warmup_rows

        feature_frames = [cat1, cat2, cat3, cat4, cat5, cat6]
        features_df = pd.concat(feature_frames, axis=1)

        expected_feature_cols = list(CORE_FEATURE_COLUMNS)
        if vix_prep is not None:
            expected_feature_cols.extend(VIX_SUPPLEMENTARY_COLUMNS)
        features_df = features_df[expected_feature_cols].astype(self.output_dtype)

        return pd.concat([meta, features_df], axis=1)

    def process_universe(
        self,
        clean_ohlcv_path: str | Path,
        benchmark_path: str | Path,
        output_path: str | Path = "data/processed/features.parquet",
        vix_path: Optional[str | Path] = None,
        batch_tickers: int = 25,
    ) -> None:
        """Streams all 300 universe stocks through the feature pipeline.

        Args:
            clean_ohlcv_path: Path to ``data/interim/stocks/ohlcv_clean.parquet``.
            benchmark_path: Path to ``data/interim/market/nifty50_clean.parquet``.
            output_path: Target Parquet path (default ``data/processed/features.parquet``).
            vix_path: Optional path to ``data/interim/market/india_vix_clean.parquet``.
            batch_tickers: Number of stocks buffered per Parquet row-group write.
        """
        logger.info("Loading interim inputs for Phase 5 feature engineering...")
        ohlcv_df = pd.read_parquet(clean_ohlcv_path)
        benchmark_df = pd.read_parquet(benchmark_path)
        bench_prep = prepare_benchmark_features(benchmark_df)

        vix_prep: Optional[pd.DataFrame] = None
        if vix_path is not None:
            vix_raw = pd.read_parquet(vix_path)
            vix_prep = prepare_vix_features(vix_raw)
            logger.info("India VIX supplementary features enabled (%s rows).", len(vix_prep))

        feature_cols = list(CORE_FEATURE_COLUMNS)
        if vix_prep is not None:
            feature_cols.extend(VIX_SUPPLEMENTARY_COLUMNS)
        flag_cols = [c for c in QUALITY_FLAG_COLUMNS if c in ohlcv_df.columns]
        if "is_warmup" not in flag_cols:
            flag_cols.append("is_warmup")

        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        unique_tickers = ohlcv_df["ticker"].drop_duplicates().tolist()
        logger.info(
            "Processing %s tickers (%s total rows) -> %s",
            len(unique_tickers),
            len(ohlcv_df),
            dest,
        )

        schema: Optional[pa.Schema] = None
        buffer: list[pd.DataFrame] = []

        # Pre-warm universe segment map before streaming
        self._get_segment_mapping()

        grouped = ohlcv_df.groupby("ticker", sort=False)
        writer: Optional[StreamingParquetWriter] = None

        try:
            for idx, (_, stock_df) in enumerate(grouped, start=1):
                feat_df = self.compute_stock_features(
                    stock_df=stock_df,
                    benchmark_df=bench_prep,
                    vix_df=vix_prep,
                )
                buffer.append(feat_df)

                if len(buffer) >= batch_tickers or idx == len(unique_tickers):
                    chunk = pd.concat(buffer, axis=0, ignore_index=True)
                    if schema is None:
                        schema = _build_parquet_schema(
                            columns=list(chunk.columns),
                            feature_cols=feature_cols,
                            flag_cols=flag_cols,
                            output_dtype=self.output_dtype,
                        )
                        writer = StreamingParquetWriter(
                            dest_path=dest,
                            schema=schema,
                            compression="snappy",
                        )
                    writer.write_dataframe(chunk, schema=schema)
                    buffer.clear()

                if idx % 50 == 0 or idx == len(unique_tickers):
                    logger.info("Processed %s / %s tickers...", idx, len(unique_tickers))
        except Exception:
            if writer is not None:
                writer.close()
                if writer.tmp_path.exists():
                    writer.tmp_path.unlink()
            raise

        if writer is not None:
            writer.__exit__(None, None, None)

        logger.info("Phase 5 feature Parquet written successfully to %s", dest)
