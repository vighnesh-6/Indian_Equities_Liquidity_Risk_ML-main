"""Approved pure quality enrichment, isolated from research file entry points."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import pandas as pd

CONFIRMED_CORPORATE_ACTIONS: Dict[Tuple[str, str], str] = {
    ("MOTILALOFS.NS", "2024-01-01"): "BONUS_ISSUE_3_FOR_1",
    ("PATANJALI.NS", "2020-01-27"): "NCLT_CAPITAL_REDUCTION",
    ("VEDL.NS", "2026-04-30"): "DEMERGER_SPINOFF_4_WAY",
}

CONFIRMED_MARKET_SHOCKS: Dict[Tuple[str, str], str] = {
    ("YESBANK.NS", "2020-03-06"): "RBI_MORATORIUM_COLLAPSE",
    ("YESBANK.NS", "2020-03-17"): "SBI_BAILOUT_RECONSTRUCTION_SURGE",
}

VERIFIED_MUHURAT_DATES: set = {
    "2019-10-27",
    "2020-11-14",
}

class DataValidator:
    def __init__(
        self,
        min_valid_trading_days: int = 200,
        stale_price_threshold: int = 3,
        extreme_move_threshold: float = 0.50,
    ) -> None:
        self.min_valid_trading_days = min_valid_trading_days
        self.stale_price_threshold = stale_price_threshold
        self.extreme_move_threshold = extreme_move_threshold
        self.required_columns = ["date", "ticker", "open", "high", "low", "close", "volume"]


    def calculate_stale_streaks(self, close_series: pd.Series) -> pd.Series:
        """Calculates point-in-time consecutive unchanged close streak for each observation.

        Streak definition:
        - Row 0: streak = 1
        - Row t: streak = streak_{t-1} + 1 if close_t == close_{t-1} else 1

        Returns:
            pd.Series: Integer streak count with matching index (int32).
        """
        n = len(close_series)
        if n == 0:
            return pd.Series(dtype="int32")

        values = close_series.values
        streaks = np.ones(n, dtype=np.int32)
        for i in range(1, n):
            if values[i] == values[i - 1]:
                streaks[i] = streaks[i - 1] + 1
            else:
                streaks[i] = 1
        return pd.Series(streaks, index=close_series.index, dtype="int32")


    def classify_extreme_moves(
        self,
        ticker: str,
        date_series: pd.Series,
        return_series: pd.Series,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Classifies extreme single-day returns (>50%) into corporate actions or genuine market shocks.

        Args:
            ticker: Stock symbol.
            date_series: Series of timestamps.
            return_series: Series of daily returns.

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (is_extreme_move, is_corporate_action_gap, is_market_shock)
        """
        is_extreme = return_series.abs() > self.extreme_move_threshold
        is_corp_action = pd.Series(False, index=return_series.index, dtype=bool)
        is_market_shock = pd.Series(False, index=return_series.index, dtype=bool)

        if not is_extreme.any():
            return is_extreme, is_corp_action, is_market_shock

        date_strings = pd.to_datetime(date_series).dt.strftime("%Y-%m-%d")
        for idx in return_series[is_extreme].index:
            d_str = date_strings.loc[idx]
            key = (ticker, d_str)
            if key in CONFIRMED_CORPORATE_ACTIONS:
                is_corp_action.loc[idx] = True
            elif key in CONFIRMED_MARKET_SHOCKS:
                is_market_shock.loc[idx] = True

        return is_extreme, is_corp_action, is_market_shock


    def prepare_benchmark_series(self, benchmark_df: pd.DataFrame) -> pd.DataFrame:
        """Prepares benchmark (Nifty 50) DataFrame for left joining with stock series.

        Computes benchmark daily returns and standardizes schema:
        - date: datetime64[ns]
        - benchmark_close: float32
        - benchmark_return: float32

        Args:
            benchmark_df: Raw benchmark DataFrame (must contain 'date' and 'close').

        Returns:
            pd.DataFrame: Deduplicated benchmark DataFrame sorted by date.
        """
        bm = benchmark_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(bm["date"]):
            bm["date"] = pd.to_datetime(bm["date"])

        bm = bm.sort_values(by="date", ascending=True).drop_duplicates(subset=["date"]).reset_index(drop=True)
        bm["benchmark_close"] = bm["close"].astype("float32")
        bm["benchmark_return"] = bm["benchmark_close"].pct_change().astype("float32")
        return bm[["date", "benchmark_close", "benchmark_return"]]


    def clean_and_enrich_stock(
        self,
        stock_df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        ticker: Optional[str] = None,
        warmup_rows: int = 80,
    ) -> pd.DataFrame:
        """Performs non-destructive cleaning, anomaly flagging, and benchmark integration.

        Guarantees:
        1. Row retention: Exactly len(stock_df) rows returned in chronological order.
        2. Immutability of raw values: Original OHLCV values are preserved exactly.
        3. Zero volume handling: Flagged via `is_zero_volume`. Return is computed directly
           from prices; non-zero returns on zero-volume days are flagged via
           `is_zero_volume_nonzero_return`.
        4. Stale price streaks: Strictly backward-looking streak count in `stale_price_streak`
           and flagged via `is_stale_price = (streak >= stale_price_threshold)`.
        5. Extreme move disambiguation: Flagged via `is_extreme_move`. Separates confirmed
           corporate actions (`is_corporate_action_gap`) from genuine market liquidity
           crises (`is_market_shock`).
        6. Short history / Warm-up: Retains short histories (`is_short_history = True` when
           total stock rows < min_valid_trading_days), and flags `is_warmup = True` for
           the first warmup_rows observations.
        7. Benchmark integration: Left-joined with ^NSEI. Missing benchmark dates preserve
           NaN in observed columns (`benchmark_close`, `benchmark_return`), flag
           `is_benchmark_available = False`, and provide strictly backward-looking
           forward-filled fallback values in `benchmark_close_fallback` and
           `benchmark_return_fallback`. Verified Diwali Muhurat trading dates are
           flagged via `is_muhurat_session = True`.

        Args:
            stock_df: Stock OHLCV DataFrame.
            benchmark_df: Optional prepared benchmark DataFrame (from prepare_benchmark_series).
            ticker: Stock symbol (if None, taken from stock_df['ticker'].iloc[0]).
            warmup_rows: Number of initial rows flagged as warmup (default: 80).

        Returns:
            pd.DataFrame: Enriched DataFrame with all quality flags and benchmark columns.
        """
        if stock_df is None or stock_df.empty:
            return pd.DataFrame()

        df = stock_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        df = df.sort_values(by="date", ascending=True).reset_index(drop=True)

        if ticker is None:
            ticker = str(df["ticker"].iloc[0]) if "ticker" in df.columns else "UNKNOWN"

        total_rows = len(df)

        # Standardize numeric types
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype("float32")
        df["volume"] = df["volume"].astype("int64")

        # 1. Price changes and return calculation (strictly backward-looking)
        df["prev_close"] = df["close"].shift(1).astype("float32")
        df["daily_return"] = ((df["close"] - df["prev_close"]) / df["prev_close"]).astype("float32")

        # 2. Quality flags
        df["is_zero_volume"] = (df["volume"] == 0).astype(bool)
        df["is_zero_volume_nonzero_return"] = (
            df["is_zero_volume"] & (df["daily_return"] != 0.0) & df["daily_return"].notnull()
        ).astype(bool)

        df["stale_price_streak"] = self.calculate_stale_streaks(df["close"])
        df["is_stale_price"] = (df["stale_price_streak"] >= self.stale_price_threshold).astype(bool)

        is_ext, is_corp, is_shock = self.classify_extreme_moves(ticker, df["date"], df["daily_return"])
        df["is_extreme_move"] = is_ext.astype(bool)
        df["is_corporate_action_gap"] = is_corp.astype(bool)
        df["is_market_shock"] = is_shock.astype(bool)

        df["is_short_history"] = (total_rows < self.min_valid_trading_days)
        df["is_warmup"] = (pd.Series(range(total_rows), index=df.index) < warmup_rows).astype(bool)

        # OHLCV validity check
        df["is_valid_ohlcv"] = (
            (df["high"] >= df["low"])
            & (df["open"] > 0)
            & (df["high"] > 0)
            & (df["low"] > 0)
            & (df["close"] > 0)
            & (df["volume"] >= 0)
            & (df["open"] >= df["low"])
            & (df["open"] <= df["high"])
            & (df["close"] >= df["low"])
            & (df["close"] <= df["high"])
        ).astype(bool)

        # 3. Benchmark Integration
        if benchmark_df is not None and not benchmark_df.empty:
            merged = pd.merge(df, benchmark_df, on="date", how="left")
            assert len(merged) == total_rows, f"Benchmark merge row mismatch: {len(merged)} vs {total_rows}"
            df["benchmark_close"] = merged["benchmark_close"].astype("float32")
            df["benchmark_return"] = merged["benchmark_return"].astype("float32")
            df["nifty_close"] = df["benchmark_close"]
            df["nifty_return"] = df["benchmark_return"]
            df["is_benchmark_available"] = df["benchmark_close"].notnull().astype(bool)

            date_str_series = df["date"].dt.strftime("%Y-%m-%d")
            df["is_muhurat_session"] = date_str_series.isin(VERIFIED_MUHURAT_DATES).astype(bool)

            # Strictly backward-looking fallback (forward-fill observed benchmark close)
            # No future data leakage: only uses same date or earlier
            df["benchmark_close_fallback"] = df["benchmark_close"].ffill().astype("float32")

            # Fallback return: 0.0 on sessions where equity traded but benchmark was unavailable
            # (unless it's row 0, where daily_return is also NaN)
            bm_ret_fallback = df["benchmark_return"].copy()
            # For unmatched sessions after row 0, impute 0.0 in fallback column
            missing_bm_mask = ~df["is_benchmark_available"]
            if total_rows > 0 and missing_bm_mask.iloc[0]:
                # If first row itself is missing benchmark, both stock and benchmark return are NaN
                bm_ret_fallback.iloc[1:] = bm_ret_fallback.iloc[1:].fillna(0.0)
            else:
                bm_ret_fallback = bm_ret_fallback.fillna(0.0)

            df["benchmark_return_fallback"] = bm_ret_fallback.astype("float32")
        else:
            df["benchmark_close"] = np.nan
            df["benchmark_return"] = np.nan
            df["nifty_close"] = np.nan
            df["nifty_return"] = np.nan
            df["is_benchmark_available"] = False
            df["is_muhurat_session"] = False
            df["benchmark_close_fallback"] = np.nan
            df["benchmark_return_fallback"] = np.nan

        return df

