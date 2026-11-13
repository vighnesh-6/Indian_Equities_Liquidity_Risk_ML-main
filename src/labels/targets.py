"""Supervised target construction module.

Phase 7 Mathematical Specification:
Constructs the forward-looking binary label `HRE_5d` (High-Risk Event in t+1 to t+5)
and the leak-free training target `hre_5d_train` with temporal boundary embargo.

Approved Formulation (D7.3 & D7.4):
    1. Forward Rolling Maximum (Strictly Ticker-Isolated):
       fwd_max_lrs_5d(i, t) = max_{k in [1, 5]} LRS(i, t+k)
       - Strictly excludes observation day t.
       - Grouped strictly by ticker.
       - Requires full 5-session forward window (min_periods=5); any missing
         session or trading freeze inside [t+1, t+5] yields NaN.
       - The final 5 observations per stock strictly evaluate to NaN.

    2. Binary Target Classification:
       hre_5d(i, t) = 1.0 if fwd_max_lrs_5d(i, t) >= theta_{90, s}
                      0.0 if fwd_max_lrs_5d(i, t) <  theta_{90, s}
                      NaN if fwd_max_lrs_5d(i, t) is NaN

       where theta_{90, s} are the frozen segment thresholds:
       - Large Cap: 73.8298
       - Mid Cap:   72.3251
       - Small Cap: 72.4359

    3. Temporal Partition & Boundary Embargo (Requirement 2 & Section 12):
       - 'train':                  Date <= 2025-10-10 (Self-contained forward windows)
       - 'train_boundary_embargo': 2025-10-13 to 2025-10-17 (5 sessions peeking into holdout)
       - 'holdout':                Date >= 2025-10-20 (250 reserved trading sessions)
       - hre_5d_train:             Equal to hre_5d in 'train'; masked to NaN in embargo & holdout.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.labels.lrs import FROZEN_LRS_THRESHOLDS, compute_lrs


def compute_forward_max_lrs(
    df: pd.DataFrame,
    forward_window: int = 5,
    lrs_col: str = "lrs",
    ticker_col: str = "ticker",
    date_col: str = "date",
) -> pd.Series:
    """Computes the forward maximum LRS over the next [t+1, t+window] observations.

    Strictly groups by ticker and excludes observation day t. Requires a full
    window of future observations. The final `forward_window` observations
    per ticker evaluate to NaN.

    Args:
        df: DataFrame containing ticker, date, and LRS column.
        forward_window: Number of forward trading sessions (default: 5).
        lrs_col: Column name of Liquidity Risk Score.
        ticker_col: Column name of ticker identifier.
        date_col: Column name of trading date.

    Returns:
        pd.Series: Forward maximum LRS series named 'fwd_max_lrs_5d', indexed
            identically to input df.
    """
    if lrs_col not in df.columns:
        raise ValueError(f"Column '{lrs_col}' not found in DataFrame.")
    if ticker_col not in df.columns or date_col not in df.columns:
        raise ValueError(f"Columns '{ticker_col}' and '{date_col}' are required.")

    # Create temporary sorting helper to guarantee chronological order per stock
    work_df = pd.DataFrame({
        "ticker": df[ticker_col],
        "date_dt": pd.to_datetime(df[date_col]),
        "lrs": df[lrs_col],
        "_orig_idx": df.index,
    })

    # Sort by ticker and date
    work_df = work_df.sort_values(["ticker", "date_dt"])

    def _fwd_max_single_ticker(s: pd.Series) -> pd.Series:
        # Reverse series, rolling max of window size requiring all valid observations, shift by 1, reverse back
        return s.iloc[::-1].rolling(window=forward_window, min_periods=forward_window).max().shift(1).iloc[::-1]

    # Grouped rolling transform strictly by ticker
    work_df["fwd_max"] = work_df.groupby("ticker")["lrs"].transform(_fwd_max_single_ticker)

    # Reorder back to original input index
    work_df = work_df.sort_values("_orig_idx")
    result = work_df["fwd_max"].copy()
    result.index = df.index
    result.name = f"fwd_max_lrs_{forward_window}d"

    return result


def construct_hre_target(
    df: pd.DataFrame,
    forward_window: int = 5,
    thresholds: Optional[Dict[str, float]] = None,
    lrs_col: str = "lrs",
    fwd_max_col: Optional[str] = None,
    cap_col: str = "cap_segment",
    # Legacy parameters preserved for backward compatibility
    amihud_threshold_quantile: float = 0.95,
    volume_ratio_threshold: float = 0.20,
    gap_threshold: float = 0.05,
    lrs_threshold: float = 75.0,
    **kwargs: Any,
) -> pd.Series:
    """Constructs the HRE_5d binary target strictly using future data (t+1 to t+5).

    Evaluates binary target based on segment-specific 90th-percentile thresholds:
        HRE = 1.0 if fwd_max_lrs_5d >= threshold_s
              0.0 if fwd_max_lrs_5d <  threshold_s
              NaN if fwd_max_lrs_5d is NaN

    Args:
        df: DataFrame containing date, ticker, cap_segment, and LRS or forward max LRS.
        forward_window: Lookahead trading days (default: 5).
        thresholds: Dictionary mapping cap segment ('large', 'mid', 'small') to threshold.
            Defaults to approved FROZEN_LRS_THRESHOLDS.
        lrs_col: Column name of LRS.
        fwd_max_col: Column name of precomputed forward max LRS; if None or absent in df,
            it is computed using `compute_forward_max_lrs`.
        cap_col: Column name of market-cap segment.
        **kwargs: Unused legacy keyword arguments preserved for backward compatibility.

    Returns:
        pd.Series: Binary target series (0.0 or 1.0 for valid observations; NaN for
            missing windows or tail rows), indexed identically to df.
    """
    th = thresholds or FROZEN_LRS_THRESHOLDS

    # Validate cap_segment column
    if cap_col not in df.columns:
        raise ValueError(f"Column '{cap_col}' is required for segment-specific threshold evaluation.")

    # Obtain or compute forward maximum LRS
    col_name = fwd_max_col or f"fwd_max_lrs_{forward_window}d"
    if col_name in df.columns:
        fwd_max = df[col_name]
    else:
        if lrs_col not in df.columns:
            raise ValueError(f"Neither '{col_name}' nor '{lrs_col}' found in DataFrame.")
        fwd_max = compute_forward_max_lrs(df, forward_window=forward_window, lrs_col=lrs_col)

    # Normalize cap segments to lowercase
    seg_series = df[cap_col].astype(str).str.lower()
    thresh_series = seg_series.map(th)

    # Construct binary target
    target = pd.Series(np.nan, index=df.index, dtype=float, name=f"hre_{forward_window}d")

    valid_mask = fwd_max.notna() & thresh_series.notna()
    target[valid_mask] = (fwd_max[valid_mask] >= thresh_series[valid_mask]).astype(float)

    return target


def apply_training_embargo_split(
    df: pd.DataFrame,
    train_cutoff: str = "2025-10-17",
    embargo_days: int = 5,
    holdout_start: str = "2025-10-20",
    date_col: str = "date",
    hre_col: str = "hre_5d",
) -> pd.DataFrame:
    """Creates the explicit partition column 'split' and leak-free training target 'hre_5d_train'.

    Partition categories:
        - 'train':                  Historical training dates strictly self-contained (t <= 2025-10-10).
        - 'train_boundary_embargo': The final 5 trading days of training (2025-10-13 to 2025-10-17),
                                    whose 5-day forward lookahead touches holdout data.
        - 'holdout':                The 250 reserved holdout trading sessions (t >= 2025-10-20).

    Target 'hre_5d_train' masks embargo rows and holdout rows to NaN, ensuring
    strict information isolation during model training.

    Args:
        df: DataFrame containing date and target columns.
        train_cutoff: Final date of training period (inclusive, default: '2025-10-17').
        embargo_days: Number of trading sessions to embargo before cutoff (default: 5).
        holdout_start: First trading date of holdout partition (default: '2025-10-20').
        date_col: Date column name.
        hre_col: Target column name.

    Returns:
        pd.DataFrame: DataFrame with two columns: 'split' (object) and 'hre_5d_train' (float64),
            indexed identically to df.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found.")
    if hre_col not in df.columns:
        raise ValueError(f"Target column '{hre_col}' not found.")

    dates = pd.to_datetime(df[date_col])
    unique_dates = sorted(dates.unique())

    # Identify trading dates up to train_cutoff
    t_cutoff = pd.to_datetime(train_cutoff)
    train_dates = [d for d in unique_dates if d <= t_cutoff]

    if len(train_dates) < embargo_days:
        raise ValueError(f"Insufficient training dates ({len(train_dates)}) for embargo of {embargo_days} days.")

    # Embargo dates: the final `embargo_days` sessions of training
    embargo_dates = set(train_dates[-embargo_days:])
    embargo_start_date = train_dates[-embargo_days]
    pre_embargo_dates = set(train_dates[:-embargo_days])

    t_holdout_start = pd.to_datetime(holdout_start)

    # Assign split column
    split = pd.Series(index=df.index, dtype=object, name="split")
    split[dates >= t_holdout_start] = "holdout"
    split[dates.isin(embargo_dates)] = "train_boundary_embargo"
    split[dates.isin(pre_embargo_dates)] = "train"

    # Fill any dates strictly prior to train cutoff that weren't in pre_embargo_dates
    split[(dates <= t_cutoff) & split.isna()] = "train"

    # Assign hre_5d_train: valid only in 'train', NaN in embargo and holdout
    hre_train = pd.Series(np.nan, index=df.index, dtype=float, name=f"{hre_col}_train")
    train_eligible_mask = (split == "train") & df[hre_col].notna()
    hre_train[train_eligible_mask] = df.loc[train_eligible_mask, hre_col]

    return pd.DataFrame({"split": split, f"{hre_col}_train": hre_train}, index=df.index)


def build_features_with_targets(
    features_df: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
    train_cutoff: str = "2025-10-17",
    embargo_days: int = 5,
    holdout_start: str = "2025-10-20",
) -> pd.DataFrame:
    """Builds the complete enriched dataset with all Phase 7 target and partition columns.

    Preserves 100% of input features, metadata, flags, and observations (including warmup).
    Appends:
        - 'lrs': Composite Liquidity-Risk Score (0–100)
        - 'fwd_max_lrs_5d': Forward 5-day rolling maximum LRS
        - 'hre_5d': Global binary High-Risk Event target
        - 'split': Categorical partition ('train', 'train_boundary_embargo', 'holdout')
        - 'hre_5d_train': Supervised training target with embargo masked to NaN

    Args:
        features_df: Original engineered features DataFrame (565,190 rows).
        thresholds: Optional segment thresholds; defaults to FROZEN_LRS_THRESHOLDS.
        train_cutoff: Training period cutoff date.
        embargo_days: Number of boundary embargo trading sessions.
        holdout_start: Holdout period start date.

    Returns:
        pd.DataFrame: Enriched dataset with original columns plus the 5 Phase 7 columns.
    """
    df_out = features_df.copy()

    # Determine post-warmup rows (warmup rows must not distort cross-sectional percentiles)
    if "is_warmup" in df_out.columns:
        post_mask = ~df_out["is_warmup"].astype(bool)
    else:
        post_mask = pd.Series(True, index=df_out.index)

    # 1. Compute LRS for valid post-warmup observations
    df_out["lrs"] = np.nan
    df_out.loc[post_mask, "lrs"] = compute_lrs(df_out.loc[post_mask])

    # 2. Compute forward max LRS (5-day window) for post-warmup observations
    df_out["fwd_max_lrs_5d"] = np.nan
    df_out.loc[post_mask, "fwd_max_lrs_5d"] = compute_forward_max_lrs(
        df_out.loc[post_mask], forward_window=5, lrs_col="lrs"
    )

    # 3. Construct global HRE_5d target
    df_out["hre_5d"] = np.nan
    df_out.loc[post_mask, "hre_5d"] = construct_hre_target(
        df_out.loc[post_mask],
        forward_window=5,
        thresholds=thresholds,
        fwd_max_col="fwd_max_lrs_5d",
    )

    # 4. Apply training embargo split and create hre_5d_train
    split_df = apply_training_embargo_split(
        df_out,
        train_cutoff=train_cutoff,
        embargo_days=embargo_days,
        holdout_start=holdout_start,
        hre_col="hre_5d",
    )
    df_out["split"] = split_df["split"]
    df_out["hre_5d_train"] = split_df["hre_5d_train"]

    return df_out
