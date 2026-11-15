"""Historical Event Studies Module for Phase 9.

Implements:
- Event calendar parsing, validation, and NSE trading calendar alignment.
- Event clustering and overlap detection (5-session proximity threshold).
- Event window construction across [-5, +5] trading sessions.
- Stock-level microstructure extraction (Amihud, Corwin-Schultz spread, volume, volatility).
- Model risk behavior integration (HGBT probability, RF Candidate B, Isolation Forest).
- Cross-sectional aggregation (Median, 25th percentile, 75th percentile, IQR) stratified by cap tier.
- Strict segregation of Out-of-Sample (OOS) vs In-Sample (pre-train) predictions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import json
import numpy as np
import pandas as pd


# Key microstructure and model metrics to analyze
MICROSTRUCTURE_METRICS = [
    "amihud_ratio",
    "amihud_z20",
    "cs_spread_est",
    "volume_ratio",
    "volume_z20",
    "log_volume",
    "relative_volume",
    "vol_20d",
    "vol_5d",
    "daily_range",
    "range_z20",
    "downside_vol_20d",
    "max_drawdown_20d",
    "overnight_gap_abs",
    "daily_return",
]

PREDICTION_METRICS = [
    "prob_hist_gradient_boosting",
    "pred_calib_hist_gradient_boosting",
    "prob_random_forest_candidate_b",
    "pred_calib_random_forest_candidate_b",
    "anomaly_score_scaled",
    "isolation_flag",
    "hre_5d",
]


def load_and_cluster_events(
    events_csv_path: Path | str = "data/events/market_events.csv",
    calendar_dates: Optional[List[pd.Timestamp]] = None,
    calendar_features_path: Optional[Path | str] = "data/processed/features_with_targets.parquet",
    max_gap_sessions: int = 5,
) -> pd.DataFrame:
    """Loads market events calendar and identifies event clusters based on trading session proximity.

    Args:
        events_csv_path: Path to curated market events CSV.
        calendar_dates: Ordered list of unique trading dates from exchange calendar.
        calendar_features_path: Path to feature dataset to extract trading calendar if calendar_dates is None.
        max_gap_sessions: Trading session distance threshold for event clustering (default: 5).

    Returns:
        pd.DataFrame: Validated events DataFrame with cluster and overlap indicators.
    """
    events_path = Path(events_csv_path)
    if not events_path.exists():
        raise FileNotFoundError(f"Event calendar not found at '{events_csv_path}'.")

    events = pd.read_csv(events_path)
    required_cols = [
        "event_id",
        "event_name",
        "event_category",
        "scope",
        "effective_trading_date",
        "is_scheduled",
    ]
    missing = [c for c in required_cols if c not in events.columns]
    if missing:
        raise ValueError(f"Event calendar missing required columns: {missing}")

    events["effective_trading_date"] = pd.to_datetime(events["effective_trading_date"])

    if calendar_dates is None and calendar_features_path is not None and Path(calendar_features_path).exists():
        feat_dates = pd.read_parquet(calendar_features_path, columns=["date"])
        calendar_dates = sorted(feat_dates["date"].unique())

    if calendar_dates is not None:
        cal_map = {pd.to_datetime(d): i for i, d in enumerate(calendar_dates)}
        events["cal_idx"] = events["effective_trading_date"].map(cal_map)
        if events["cal_idx"].isna().any():
            missing_dates = events[events["cal_idx"].isna()]["effective_trading_date"].tolist()
            raise ValueError(f"Some event dates not in trading calendar: {missing_dates}")
    else:
        # Fallback to business day difference from origin
        origin = events["effective_trading_date"].min()
        events["cal_idx"] = [
            len(pd.bdate_range(origin, d)) for d in events["effective_trading_date"]
        ]

    events = events.sort_values("cal_idx").reset_index(drop=True)

    # Detect clusters (gap <= max_gap_sessions)
    events["event_cluster_id"] = ""
    events["is_overlapping_event"] = False

    curr_cluster = 0
    n = len(events)
    for i in range(n):
        diff_prev = int(events.loc[i, "cal_idx"] - events.loc[i - 1, "cal_idx"]) if i > 0 else 999
        diff_next = int(events.loc[i + 1, "cal_idx"] - events.loc[i, "cal_idx"]) if i < n - 1 else 999

        is_overlap = (diff_prev <= max_gap_sessions) or (diff_next <= max_gap_sessions)
        events.loc[i, "is_overlapping_event"] = is_overlap

        if diff_prev <= max_gap_sessions:
            events.loc[i, "event_cluster_id"] = f"CLUSTER_{curr_cluster:02d}"
        elif diff_next <= max_gap_sessions:
            curr_cluster += 1
            events.loc[i, "event_cluster_id"] = f"CLUSTER_{curr_cluster:02d}"
        else:
            events.loc[i, "event_cluster_id"] = f"ISOLATED_{events.loc[i, 'event_id']}"

    return events


def build_event_windows(
    events_df: pd.DataFrame,
    trading_calendar: List[pd.Timestamp],
    window_pre: int = 5,
    window_post: int = 5,
) -> pd.DataFrame:
    """Builds [-window_pre, +window_post] event window schedule for all events.

    Args:
        events_df: Clustered events DataFrame.
        trading_calendar: Ordered list of unique trading sessions.
        window_pre: Sessions before event (default: 5).
        window_post: Sessions after event (default: 5).

    Returns:
        pd.DataFrame: Event windows schedule (86 events * 11 days = 946 window days).
    """
    cal_map = {d: i for i, d in enumerate(trading_calendar)}
    records = []

    wf_start = pd.Timestamp("2023-05-10")
    wf_end = pd.Timestamp("2025-10-10")
    ho_start = pd.Timestamp("2025-10-20")
    ho_end = pd.Timestamp("2026-10-23")

    for _, evt in events_df.iterrows():
        evt_id = evt["event_id"]
        evt_date = evt["effective_trading_date"]
        evt_idx = cal_map[evt_date]

        for rel_day in range(-window_pre, window_post + 1):
            target_idx = evt_idx + rel_day
            if target_idx < 0 or target_idx >= len(trading_calendar):
                continue

            session_date = trading_calendar[target_idx]

            # Determine prediction regime
            if wf_start <= session_date <= wf_end:
                regime = "OUT_OF_SAMPLE_WF"
            elif ho_start <= session_date <= ho_end:
                regime = "OUT_OF_SAMPLE_HOLDOUT"
            elif wf_end < session_date < ho_start:
                regime = "BOUNDARY_EMBARGO"
            else:
                regime = "IN_SAMPLE_PRETRAIN"

            # Window period label
            if rel_day < 0:
                period = "PRE_EVENT"
            elif rel_day == 0:
                period = "EVENT_DAY"
            else:
                period = "POST_EVENT"

            records.append(
                {
                    "event_id": evt_id,
                    "event_name": evt["event_name"],
                    "event_category": evt["event_category"],
                    "scope": evt["scope"],
                    "is_scheduled": evt["is_scheduled"],
                    "effective_trading_date": evt_date,
                    "event_cluster_id": evt["event_cluster_id"],
                    "is_overlapping_event": evt["is_overlapping_event"],
                    "relative_day": rel_day,
                    "period": period,
                    "session_date": session_date,
                    "prediction_regime": regime,
                }
            )

    return pd.DataFrame(records)


def extract_event_stock_data(
    event_windows_df: pd.DataFrame,
    features_df: pd.DataFrame,
    wf_preds_df: pd.DataFrame,
    ho_preds_df: pd.DataFrame,
    wf_if_df: pd.DataFrame,
    ho_if_df: pd.DataFrame,
    hgbt_model: Optional[Any] = None,
    hgbt_features: Optional[List[str]] = None,
    rf_model: Optional[Any] = None,
    rf_features: Optional[List[str]] = None,
    if_detector: Optional[Any] = None,
) -> pd.DataFrame:
    """Merges event window schedule with stock-level features and historical predictions.

    Args:
        event_windows_df: Event window schedule.
        features_df: Full feature matrix with targets.
        wf_preds_df: Batch 4 walk-forward supervised predictions.
        ho_preds_df: Batch 5 holdout supervised predictions.
        wf_if_df: Batch 4 Isolation Forest predictions.
        ho_if_df: Batch 5 Isolation Forest holdout predictions.
        hgbt_model: Optional pre-fitted HGBT model for in-sample scoring.
        hgbt_features: Feature list for HGBT.
        rf_model: Optional pre-fitted RF model for in-sample scoring.
        rf_features: Feature list for RF.
        if_detector: Optional pre-fitted AnomalyDetector.

    Returns:
        pd.DataFrame: Merged stock-level event window observations.
    """
    # 1. Prepare unified out-of-sample predictions
    # Standardize column names
    wf_sup = wf_preds_df[
        [
            "date",
            "ticker",
            "cap_segment",
            "prob_hist_gradient_boosting",
            "pred_calib_hist_gradient_boosting",
            "prob_random_forest_candidate_b",
            "pred_calib_random_forest_candidate_b",
        ]
    ].copy()

    ho_sup = ho_preds_df[
        [
            "date",
            "ticker",
            "cap_segment",
            "prob_hist_gradient_boosting",
            "pred_calib_hist_gradient_boosting",
            "prob_random_forest_candidate_b",
            "pred_calib_random_forest_candidate_b",
        ]
    ].copy()

    oos_sup = pd.concat([wf_sup, ho_sup], ignore_index=True)

    wf_anom = wf_if_df[["date", "ticker", "anomaly_score_scaled", "isolation_flag"]].copy()
    ho_anom = ho_if_df[["date", "ticker", "anomaly_score_scaled", "isolation_flag"]].copy()
    oos_anom = pd.concat([wf_anom, ho_anom], ignore_index=True)

    oos_preds = pd.merge(oos_sup, oos_anom, on=["date", "ticker"], how="outer")

    # 2. Extract needed feature columns
    feat_cols = ["date", "ticker", "cap_segment", "hre_5d"] + [
        c for c in MICROSTRUCTURE_METRICS if c in features_df.columns
    ]
    sub_features = features_df[feat_cols].copy()

    # 3. Merge features with OOS predictions
    stock_market_data = pd.merge(sub_features, oos_preds, on=["date", "ticker", "cap_segment"], how="left")

    # 4. Score in-sample observations for pre-2023 events if models provided
    if hgbt_model is not None and hgbt_features is not None:
        mask_missing_hgbt = stock_market_data["prob_hist_gradient_boosting"].isna()
        if mask_missing_hgbt.sum() > 0:
            # Score strictly in inference mode without refitting
            X_in = features_df.loc[mask_missing_hgbt, hgbt_features]
            probs_in = hgbt_model.predict_proba(X_in)[:, 1]
            stock_market_data.loc[mask_missing_hgbt, "prob_hist_gradient_boosting"] = probs_in
            stock_market_data.loc[mask_missing_hgbt, "pred_calib_hist_gradient_boosting"] = (
                probs_in >= 0.534
            ).astype(int)

    if rf_model is not None and rf_features is not None:
        mask_missing_rf = stock_market_data["prob_random_forest_candidate_b"].isna()
        if mask_missing_rf.sum() > 0:
            X_in_rf = features_df.loc[mask_missing_rf, rf_features]
            probs_in_rf = rf_model.predict_proba(X_in_rf)[:, 1]
            stock_market_data.loc[mask_missing_rf, "prob_random_forest_candidate_b"] = probs_in_rf
            stock_market_data.loc[mask_missing_rf, "pred_calib_random_forest_candidate_b"] = (
                probs_in_rf >= 0.508
            ).astype(int)

    if if_detector is not None:
        mask_missing_if = stock_market_data["anomaly_score_scaled"].isna()
        if mask_missing_if.sum() > 0:
            if_scores = if_detector.score(features_df.loc[mask_missing_if])
            stock_market_data.loc[mask_missing_if, "anomaly_score_scaled"] = if_scores[
                "anomaly_score_scaled"
            ].values
            stock_market_data.loc[mask_missing_if, "isolation_flag"] = if_scores["isolation_flag"].values

    # 5. Join event windows with stock market data on date
    event_windows_df["date"] = event_windows_df["session_date"]
    merged = pd.merge(event_windows_df, stock_market_data, on="date", how="inner")

    return merged


def compute_cross_sectional_event_metrics(
    event_stock_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Computes cross-sectional summary statistics (median, q25, q75, iqr, mean, std) across stocks.

    Generates stratified summaries for:
    - Large-Cap
    - Mid-Cap
    - Small-Cap
    - Market-Wide (pooled across all segments)

    Args:
        event_stock_df: Stock-level observations merged with event windows.
        metrics: List of metrics to summarize.

    Returns:
        pd.DataFrame: Cross-sectional summary table.
    """
    if metrics is None:
        metrics = [
            "amihud_ratio",
            "cs_spread_est",
            "volume_ratio",
            "vol_20d",
            "daily_range",
            "daily_return",
            "prob_hist_gradient_boosting",
            "pred_calib_hist_gradient_boosting",
            "prob_random_forest_candidate_b",
            "anomaly_score_scaled",
            "isolation_flag",
            "hre_5d",
        ]

    # Filter available metrics
    available_metrics = [m for m in metrics if m in event_stock_df.columns]

    group_keys = [
        "event_id",
        "event_name",
        "event_category",
        "scope",
        "is_scheduled",
        "is_overlapping_event",
        "event_cluster_id",
        "prediction_regime",
        "period",
        "relative_day",
        "session_date",
    ]

    records = []

    # 1. Stratified by cap_segment
    for key_tuple, grp in event_stock_df.groupby(group_keys + ["cap_segment"]):
        row = dict(zip(group_keys + ["cap_segment"], key_tuple))
        row["segment"] = row.pop("cap_segment")
        row["stock_count"] = len(grp["ticker"].unique())

        for m in available_metrics:
            series = grp[m].dropna()
            if len(series) > 0:
                row[f"{m}_median"] = float(series.median())
                row[f"{m}_q25"] = float(series.quantile(0.25))
                row[f"{m}_q75"] = float(series.quantile(0.75))
                row[f"{m}_iqr"] = row[f"{m}_q75"] - row[f"{m}_q25"]
                row[f"{m}_mean"] = float(series.mean())
                row[f"{m}_std"] = float(series.std()) if len(series) > 1 else 0.0
            else:
                row[f"{m}_median"] = np.nan
                row[f"{m}_q25"] = np.nan
                row[f"{m}_q75"] = np.nan
                row[f"{m}_iqr"] = np.nan
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan

        records.append(row)

    # 2. Market-wide pooled
    for vals, grp in event_stock_df.groupby(group_keys):
        row = dict(zip(group_keys, vals))
        row["segment"] = "market_wide"
        row["stock_count"] = len(grp["ticker"].unique())

        for m in available_metrics:
            series = grp[m].dropna()
            if len(series) > 0:
                row[f"{m}_median"] = float(series.median())
                row[f"{m}_q25"] = float(series.quantile(0.25))
                row[f"{m}_q75"] = float(series.quantile(0.75))
                row[f"{m}_iqr"] = row[f"{m}_q75"] - row[f"{m}_q25"]
                row[f"{m}_mean"] = float(series.mean())
                row[f"{m}_std"] = float(series.std()) if len(series) > 1 else 0.0
            else:
                row[f"{m}_median"] = np.nan
                row[f"{m}_q25"] = np.nan
                row[f"{m}_q75"] = np.nan
                row[f"{m}_iqr"] = np.nan
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan

        records.append(row)

    summary_df = pd.DataFrame(records)
    summary_df = summary_df.sort_values(["event_id", "segment", "relative_day"]).reset_index(drop=True)
    return summary_df


def compute_event_impact_summary(cross_sectional_df: pd.DataFrame) -> pd.DataFrame:
    """Computes event-level impact metrics comparing Pre-Event vs Event-Day and Post-Event.

    Args:
        cross_sectional_df: Output from compute_cross_sectional_event_metrics.

    Returns:
        pd.DataFrame: Table summarizing spread shock, Amihud surge, risk surge for every event.
    """
    mw = cross_sectional_df[cross_sectional_df["segment"] == "market_wide"].copy()
    events = mw["event_id"].unique()

    records = []
    for evt_id in events:
        evt_data = mw[mw["event_id"] == evt_id].sort_values("relative_day")
        if len(evt_data) == 0:
            continue

        first_row = evt_data.iloc[0]

        # Pre-event baseline (relative_day in [-5, -1])
        pre = evt_data[evt_data["relative_day"] < 0]
        event_day = evt_data[evt_data["relative_day"] == 0]
        post = evt_data[evt_data["relative_day"] > 0]

        pre_spread = pre["cs_spread_est_median"].mean() if len(pre) > 0 else np.nan
        evt_spread = event_day["cs_spread_est_median"].values[0] if len(event_day) > 0 else np.nan
        post_spread = post["cs_spread_est_median"].mean() if len(post) > 0 else np.nan

        pre_amihud = pre["amihud_ratio_median"].mean() if len(pre) > 0 else np.nan
        evt_amihud = event_day["amihud_ratio_median"].values[0] if len(event_day) > 0 else np.nan
        post_amihud = post["amihud_ratio_median"].mean() if len(post) > 0 else np.nan

        pre_prob = pre["prob_hist_gradient_boosting_median"].mean() if len(pre) > 0 else np.nan
        evt_prob = event_day["prob_hist_gradient_boosting_median"].values[0] if len(event_day) > 0 else np.nan
        post_prob = post["prob_hist_gradient_boosting_median"].mean() if len(post) > 0 else np.nan

        pre_anom = pre["anomaly_score_scaled_median"].mean() if len(pre) > 0 else np.nan
        evt_anom = event_day["anomaly_score_scaled_median"].values[0] if len(event_day) > 0 else np.nan
        post_anom = post["anomaly_score_scaled_median"].mean() if len(post) > 0 else np.nan

        spread_shock_pct = (
            100.0 * (evt_spread - pre_spread) / pre_spread if (pre_spread and pre_spread > 0) else np.nan
        )
        amihud_surge_pct = (
            100.0 * (evt_amihud - pre_amihud) / pre_amihud if (pre_amihud and pre_amihud > 0) else np.nan
        )
        prob_delta = (evt_prob - pre_prob) if (evt_prob is not None and pre_prob is not None) else np.nan

        records.append(
            {
                "event_id": evt_id,
                "event_name": first_row["event_name"],
                "event_category": first_row["event_category"],
                "scope": first_row["scope"],
                "is_scheduled": first_row["is_scheduled"],
                "is_overlapping_event": first_row["is_overlapping_event"],
                "event_cluster_id": first_row["event_cluster_id"],
                "prediction_regime": first_row["prediction_regime"],
                "session_date": first_row["session_date"],
                "pre_event_spread_bp": pre_spread * 10000.0 if pre_spread is not None else np.nan,
                "event_day_spread_bp": evt_spread * 10000.0 if evt_spread is not None else np.nan,
                "post_event_spread_bp": post_spread * 10000.0 if post_spread is not None else np.nan,
                "spread_shock_pct": spread_shock_pct,
                "pre_event_amihud": pre_amihud,
                "event_day_amihud": evt_amihud,
                "amihud_surge_pct": amihud_surge_pct,
                "pre_event_hgbt_prob": pre_prob,
                "event_day_hgbt_prob": evt_prob,
                "post_event_hgbt_prob": post_prob,
                "hgbt_prob_delta": prob_delta,
                "pre_event_anomaly_score": pre_anom,
                "event_day_anomaly_score": evt_anom,
                "post_event_anomaly_score": post_anom,
            }
        )

    return pd.DataFrame(records)
