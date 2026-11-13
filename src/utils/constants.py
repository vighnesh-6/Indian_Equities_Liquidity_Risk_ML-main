"""Global constants and enumeration mappings."""

RISK_BANDS = {
    "LOW": {"min": 0.0, "max": 25.0, "label": "Low Risk", "color": "#28a745"},
    "MODERATE": {"min": 25.0, "max": 50.0, "label": "Moderate Risk", "color": "#ffc107"},
    "HIGH": {"min": 50.0, "max": 75.0, "label": "High Risk", "color": "#fd7e14"},
    "VERY_HIGH": {"min": 75.0, "max": 100.0, "label": "Very High Risk", "color": "#dc3545"},
}

FEATURE_CATEGORIES = {
    1: "Returns and Price Behaviour (F01-F10)",
    2: "Volatility and Downside Risk (F11-F22)",
    3: "Liquidity and Trading-Activity Proxies (F23-F36)",
    4: "Price-Gap and Price-Impact Proxies (F37-F45)",
    5: "Market-Relative Behaviour (F46-F51)",
    6: "Market-Stress and Correlation (F52-F55)",
}

TOTAL_AUDITED_FEATURES = 55
DEFAULT_BENCHMARK = "^NSEI"
CANDIDATE_INDIA_VIX = "^INDIAVIX"
WARMUP_WINDOW_DAYS = 80
DEFAULT_TARGET_UNIVERSE_SIZE = 300

# Phase 7: Target & Label Engineering Constants (Frozen Contract D7.1 - D7.4)
DEFAULT_LRS_WEIGHTS = {
    "amihud_ma20": 0.35,
    "cs_spread_ma10": 0.25,
    "volume_z20": 0.20,
    "downside_vol_20d": 0.20,
}

FROZEN_LRS_THRESHOLDS = {
    "large": 73.8298,
    "mid": 72.3251,
    "small": 72.4359,
}

FORWARD_TARGET_WINDOW_DAYS = 5
TRAIN_CUTOFF_DATE = "2025-10-17"
EMBARGO_START_DATE = "2025-10-13"
HOLDOUT_START_DATE = "2025-10-20"

