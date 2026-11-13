# ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Architecture Status](https://img.shields.io/badge/Architecture-Audited%20%26%20Implementation--Ready-brightgreen.svg)](docs/ARCHITECTURE.md)
[![Project Status](https://img.shields.io/badge/Status-Phase%204.2%20Complete%20(Interim%20Dataset)-brightgreen.svg)](docs/PROJECT_STATUS.md)

An institutional-grade decision-support platform designed to quantify, forecast, and visualize liquidity risk and abnormal trading regimes across 300 Indian equities listed on the National Stock Exchange (NSE).

> **Important Disclaimer:** This system is a quantitative decision-support tool for risk analysis. It is **not** a stock-price predictor, trading signal generator, or automated trading robot.

---

## 1. Project Objectives

1. **Rule-Based Liquidity Risk Score (LRS, 0–100):** Calculate a daily, transparent, formula-driven composite score characterizing the instantaneous liquidity state of each stock using academically cited microstructure proxies (Amihud illiquidity, Corwin-Schultz high-low spread estimator, volume drought, and range dispersion).
2. **Supervised Risk Prediction (`HRE_5d`):** Train classical machine learning classifiers (Logistic Regression baseline, Random Forest, and HistGradientBoosting) using strictly chronological walk-forward validation to forecast the probability of a High-Risk Event within the next five trading days.
3. **Unsupervised Anomaly Detection:** Deploy an Isolation Forest model to flag historically unusual stock-day trading regimes, operating as a statistical outlier detector independent of future labels.
4. **Market Stress & Major Market Events Analysis:** Conduct event-study analyses around major macroeconomic, electoral, geopolitical, and regulatory market shocks using a curated event calendar and supplementary market-fear index (India VIX).
5. **Interactive Streamlit Dashboard:** Present multi-tier intelligence across market overview, stock deep-dives, portfolio aggregation, and data quality monitoring.
6. **Automated Post-Market Refresh:** Orchestrate automated daily ingestion, feature calculation, scoring, and manifest logging.

---

## 2. Target Stakeholders

| Stakeholder | Use Case & Core Value | Primary Output / View |
|---|---|---|
| **Retail Investors** | Screen illiquid or trap counters before entering/exiting positions | Overview risk ranks & stock detail drilldowns |
| **Portfolio Managers** | Audit aggregate liquidity concentration across holdings | Portfolio Risk Aggregator |
| **Risk Analysts** | Detect systemic illiquidity contagion across market-cap tiers | Anomaly Log & Market Stress alerts |
| **Quant Researchers** | Benchmark microstructure proxies against Indian equities | Feature dataset & walk-forward metrics |
| **Academic Evaluators** | Verify leakage-free temporal modeling and econometric rigor | Methodology documentation & test suite |

---

## 3. Data Scope & Sources

- **Universe (300 Stocks):** Stratified sample spanning the NSE market-cap spectrum:
  - **100 Large-Cap:** All constituents of the Nifty 100 index (liquid baseline anchor).
  - **150 Mid-Cap:** All constituents of the Nifty Midcap 150 index (core segment for risk variation).
  - **50 Small-Cap:** Selected from Nifty Smallcap 250 using 6-month trailing ADTV ranking.
- **Benchmark Index:** Nifty 50 (`^NSEI`) for market-relative features (beta, correlation, excess returns).
- **Supplementary Fear Index:** India VIX (`^INDIAVIX`) for market-wide expected volatility.
- **Market-Wide Event Calendar:** Curated dates covering national elections, Union Budgets, geopolitical escalations, credit freezes, and exchange halts (`data/events/market_events_template.csv`).
- **Data Ingestion Standard:** Daily OHLCV via `yfinance` (`.NS` tickers, `auto_adjust=True`).
- **Explicit Exclusions:** No scraping of NSE India market pages, no Indian broker APIs, and strictly no individual company earnings, news, transcripts, or NLP/sentiment.
- **Target Time Horizon:** Historical window from January 2018 to October 2026.

---

## 4. Methodological Foundations

### Dual-Output Architecture + Anomaly Detection
```text
                          ┌────────────────────────────────────────────────────────┐
                          │   Processed Feature Matrix (F01–F55 + 3 VIX features)  │
                          │        data/processed/features.parquet (float32)       │
                          └──────────────────────────┬─────────────────────────────┘
                                                     │
         ┌───────────────────────────────────────────┼───────────────────────────────────────────┐
         ▼                                           ▼                                           ▼
┌─────────────────────────────┐             ┌─────────────────────────────┐             ┌─────────────────────────────┐
│  Composite LRS (Rule-Based) │             │ Forward HRE Prediction (ML) │             │  Isolation Forest (Anomaly) │
│  - Range: 0 to 100          │             │  - Target: HRE_5d (binary)  │             │  - Unsupervised outlier     │
│  - 6 academic proxies       │             │  - Models: LR, RF, HGBT     │             │  - 12 designated features   │
│  - Instantaneous State      │             │  - Walk-Forward Validation  │             │  - Historical unusualness   │
│  - Cross-sectional ranks    │             │  - Evaluated by PR-AUC      │             │  - NOT forward prediction   │
└──────────────┬──────────────┘             └──────────────┬──────────────┘             └──────────────┬──────────────┘
               │                                           │                                           │
               └───────────────────────────────────────────┼───────────────────────────────────────────┘
                                                           ▼
                                      ┌───────────────────────────────────────────┐
                                      │   Presentation Tier: Streamlit Dashboard  │
                                      │   (5 Interactive Multi-Page Risk Views)   │
                                      └───────────────────────────────────────────┘
```

### Audited 55-Feature Engine (F01 to F55)
- **Category 1: Returns & Price Behaviour (10 features):** Daily returns, log returns, multi-period momentum (5d, 20d, 60d), rolling return z-scores, absolute returns, MA deviations.
- **Category 2: Volatility & Downside Risk (12 features):** Realized volatility (5d, 20d, 60d), Parkinson high-low volatility, Average True Range (ATR), semi-deviation downside vol, Sortino ratio component, rolling max drawdown, RSI-14, vol-of-vol.
- **Category 3: Liquidity & Trading Activity Proxies (14 features):** Log volume, volume moving averages, volume ratio, volume z-scores, Amihud (2002) illiquidity ratio ($10^6$ scaled), turnover value, zero-volume and stale-price flags, low-volume streaks.
- **Category 4: Price-Gap & Price-Impact Proxies (9 features):** Normalized daily range, overnight gap returns, Corwin-Schultz (2012) high-low spread estimator (floored at zero), intraday return.
- **Category 5: Market-Relative Behaviour (6 features):** Excess return over Nifty 50, rolling 60d beta, rolling 60d correlation, idiosyncratic volatility, relative volume ratio.
- **Category 6: Market-Stress Features (4 features + 3 supplementary India VIX features):** Benchmark realized volatility (`nifty_vol_20d`), market volatility regime ratio (`nifty_vol_ratio`), market 20d return (`nifty_return_20d`), volatility acceleration (`vol_beta`), plus separate supplementary India VIX features (`vix_close`, `vix_return_1d`, `vix_z20`).

### Temporal Leakage Prevention
- **Expanding-Window Walk-Forward Validation:** Chronological splits; no future data in training sets.
- **Fold-Specific Scaling:** `StandardScaler` fit strictly on training folds.
- **Strict Target Separation:** `HRE_5d` requires future outcomes ($t+1$ to $t+5$); live scored rows strictly assign `HRE_5d = NaN`.
- **Warm-Up Isolation:** First 80 trading days per stock tagged `is_warmup=True` and excluded from model fitting.

---

## 5. Limitations (Transparently Acknowledged)

1. **Proxy-Only Liquidity:** Daily OHLCV data cannot directly observe order-book depth, bid-ask quotes, or execution costs. All measures are academically validated proxies.
2. **Survivorship Bias:** Current index constituents reflect survivor companies. Delisted or failed companies are inaccessible in free public feeds.
3. **Daily Granularity:** Risk metrics evaluate close-to-close behavior; intraday flash illiquidity is not observable.
4. **Third-Party API Dependency:** Yahoo Finance is an undocumented public endpoint subject to rate limits and symbol adjustments.

---

## 6. System Architecture & Directory Layout

```text
liquidity-risk-intelligence/
├── configs/                  # Version-controlled configuration files
│   ├── universe.json         # 300 tickers, cap tiers, benchmark, VIX candidate
│   ├── pipeline_config.yaml  # Paths, fetch params, retry logic, date ranges
│   ├── feature_config.yaml   # F01-F55 feature toggles, lookback windows, LRS weights
│   ├── model_config.yaml     # Walk-forward splits, hyperparameters, decision threshold
│   └── dashboard_config.yaml # Streamlit UI layout, risk bands, color scheme
├── data/                     # Data stores (market data excluded from Git)
│   ├── raw/                  # Downloaded Parquet files (300 stocks, 3 market indices)
│   ├── interim/              # Cleaned interim datasets (stocks/ and market/ subfolders)
│   │   ├── stocks/           # data/interim/stocks/ohlcv_clean.parquet (565,190 rows x 27 cols)
│   │   └── market/           # Cleaned Nifty 50, Bank Nifty, and India VIX indices
│   ├── processed/            # Engineered feature matrix (data/processed/features.parquet: 565,190 x 70)
│   ├── predictions/          # Scored daily predictions (LRS, HRE_prob, anomaly_score)
│   ├── manifests/            # Data collection and pipeline audit manifests
│   └── events/               # Curated market events calendar (market_events.csv, 86 events)
├── audit/                    # Cryptographic verification tools and universe audit evidence
│   ├── compute_raw_sha256.py # Recomputes SHA-256 digests and verifies raw Merkle root
│   ├── raw_data_sha256.json  # Cryptographic baseline hashes for all 303 raw files
│   └── smallcap250_adtv_ranked.csv # 6-month ADTV ranking for smallcap selection
├── notebooks/                # Sequential exploratory notebooks (01_data_collection & 03_feature_exploration executed)
├── src/                      # Production Python package
│   ├── data/                 # Universe loader, fetcher, validator, storage manager, events
│   ├── features/             # Feature orchestrator, returns, vol, liquidity, gaps, stress (F01-F55 + VIX)
│   ├── labels/               # Composite LRS calculator, forward HRE_5d target builder
│   ├── modeling/             # Walk-forward trainer, scorer, Isolation Forest anomaly, evaluator
│   ├── pipeline/             # Daily refresh pipeline, system health checks
│   ├── reporting/            # Validation report generator
│   └── utils/                # Logger, config loader, calendar dates, constants
├── app/                      # Streamlit multi-page application
│   ├── streamlit_app.py      # Entrypoint navigation
│   ├── pages/                # 1_Overview, 2_Stock_Detail, 3_Portfolio_Risk, 4_Anomaly, 5_Health
│   └── components/           # Plotly charts, styled tables, UI filter widgets
├── models/                   # Serialized model artifacts (*.joblib) & model_registry.json
├── reports/                  # Markdown audit reports and presentation figures
│   ├── phase4_2_cleaning_eda.md          # Data cleaning and interim dataset audit report
│   ├── phase5_feature_engineering_report.md # Phase 5 feature validation & engineering report
│   └── figures/              # Generated high-resolution visualization charts
├── logs/                     # Daily runtime logs (pipeline_YYYYMMDD.log)
├── scripts/                  # Standalone execution runners
│   ├── backfill_history.py           # Historical data acquisition runner
│   ├── generate_interim_dataset.py   # Phase 4.2 data cleaning & interim Parquet generator
│   ├── generate_phase4_2_eda.py      # Phase 4.2 basic EDA metrics and figure generator
│   └── train_models.py               # Model training runner
├── tests/                    # Pytest offline test suite (61 unit tests passing)
├── docs/                     # Architectural and methodology documentation
│   └── ARCHITECTURE.md       # Canonical system architecture document
├── requirements.txt          # Pinned dependencies
├── pyproject.toml            # Project packaging metadata
└── .gitignore                # Exclusions for raw data, models, logs, caches, and secrets
```

---

## 7. Software Design & Efficiency Principles

- **Sequential Streaming Processing:** Processing stock data sequentially stock-by-stock keeps process memory footprint low and predictable regardless of universe size.
- **Chunked Parquet Writes:** Data pipeline scripts use chunked PyArrow streaming writers to stream records directly to disk, avoiding full-dataset memory accumulation.
- **Numeric Precision Optimization:** Numeric price and return columns are cast to `float32`, cutting in-memory footprint in half while retaining precision for quantitative modeling.
- **Atomic File Replacement:** File writing utilities write first to temporary files before atomically replacing target destinations, eliminating partial file corruption.

---

## 8. Installation & Setup

### Prerequisites
- Python 3.10+ installed.

### Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd ads-project-quant

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Install pinned dependencies
pip install -r requirements.txt
```

### Running Test Suite
```bash
pytest tests/
```

### Generating Clean Interim Dataset
```bash
python scripts/generate_interim_dataset.py
```

### Running Exploratory Notebook
```bash
jupyter notebook notebooks/01_data_collection.ipynb
```

---

## 9. Current Project Status

- **Phase 1 (Data Acquisition Design):** Completed. Universe defined in `configs/universe.json`.
- **Phase 2 (Microstructure Proxy Formulation):** Completed. F01–F55 documented in `configs/feature_config.yaml`.
- **Phase 3 (Pipeline Infrastructure):** Completed. Modular fetcher, validator, and storage managers implemented.
- **Phase 4.1 (Historical Ingestion):** Completed. 300 stocks (565,190 rows) and 3 market indices acquired and verified in `data/raw/`.
- **Phase 4.2 (Data Cleaning & Interim Dataset):** Completed. Validated interim datasets generated at `data/interim/stocks/ohlcv_clean.parquet` and `data/interim/market/` with comprehensive EDA report in `reports/phase4_2_cleaning_eda.md`.
- **Phase 5 (Feature Engineering Pipeline):** Completed. Implemented all 55 core features (`F01`–`F55`) + 3 India VIX supplementary features in `src/features/`, generated `data/processed/features.parquet` (565,190 rows × 70 columns, `float32`, Snappy), and verified with 61 passing tests in `tests/`.
- **Next Step:** **Phase 6: Post-Engineering EDA & Regime Analysis / Phase 7: LRS & HRE Target Engineering** (`src/labels/`).
