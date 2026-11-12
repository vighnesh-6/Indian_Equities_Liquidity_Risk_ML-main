## Advanced Data Science : Pre-Submission Summary

## 1. Project and Group

- Project Title & Group: ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities | Group Number: 8

- Team Members & Student IDs: 1. AARYAMAN PRATAP SINGH (240356) | 2. ADITYA NADAR (240368) | 3. ANANSHI SHARMA (240377) | 4. VIGHNESH SOMANI (240946)

## 2. Stakeholder and Problem

- Stakeholders & Domain: Retail equity investors, portfolio managers, and risk analysts active in Indian equities (NSE).

- Trading-Risk Problem & Impact: Severe liquidity fragmentation across market-cap tiers leaves non-institutional participants vulnerable to sudden liquidity droughts, spread blowouts, and price gaps. Lacking costly real-time order books, stakeholders face severe execution slippage, portfolio concentration traps, and an inability to exit during stress.

## 3. Research Questions and Hypotheses

- Research Questions: Can daily OHLCV bars and proxy estimators reliably forecast high-risk liquidity regimes without intraday order books? How does liquidity risk vary across market caps, and does macro volatility (India VIX) amplify it?

- Initial Testable Hypotheses: H1: Amihud illiquidity positively correlates with future high-risk liquidity events. H2: Volume droughts (<20% of 20-day MA) precede increased price gaps. H3: Small-caps exhibit systematically higher Amihud ratios than large-caps. H4: Macro-volatility shocks amplify single-stock illiquidity.

## 4. Data Sources

- Sources & Scale: Daily OHLCV via Yahoo Finance API (yfinance, NSE .NS suffix) for 300 curated stocks (100 Large, 150 Mid, 50 Small-Cap) and 3 indices (NIFTY 50 [^NSEI], Bank Nifty [^NSEBANK], India VIX [^INDIAVIX]), plus regulatory event archives (SEBI, NSE, RBI).

- Coverage, Scale & Constraints: 2018-01-01 to 2026-10-23 . 565,190 raw stock rows (223 full-history stocks, 77 legitimate post- 2018 listings) +(NIFTY50 , BANKNIFTY ,INDIAVIX) 6,499 Market index rows (571,689 total). Variables: OHLCV, returns, volatility, event distances. Constraints: No Level-2 order books; 4 special Saturday sessions omitted by data provider.

## 5. Integration Plan (SSOT)

- SSOT Architecture & Join Keys: Unified tabular Single Source of Truth (SSOT) joining stock OHLCV with market indices and event indicators using composite key (ticker, date) synchronized to the exchange calendar.

- Data Quality & Cleaning: Audited all 303 Parquets: 0 duplicate dates, 0 missing OHLCV values across 3.95M cells, 0 negative prices, 0 spread inversions. 100% raw-to-interim row preservation (565,190 stock rows). Extreme moves (>300%) and zero volume flagged; forward-filled benchmark alignment handles non-trading index gaps. Remaining: Merging 55 features into processed dataset.

## 6. Methods and BADIR Workflow

- BADIR Framework Alignment: 1. Business Understanding: Scoped dual-output design (Composite LRS + ML Classifier). 2. Data Acquisition & 3. Data Integration: Automated ingestion with retry logic; completed Phase 4 cleaning, calendar synchronization, and SSOT structuring.

- Analysis, Modeling & Reporting: 55 quantitative features (Amihud, Corwin-Schultz spread, volume z-scores, Parkinson volatility, beta). Rule-based Composite Liquidity-Risk Score (LRS, 0-100) for current state. Supervised classifiers (Logistic Regression baseline, Random Forest, HistGradientBoosting) predicting 5-day forward High-Risk Events (hre_5d). Unsupervised Isolation Forest for historical anomalies. Streamlit dashboard for monitoring.

## 7. Validation and Decision Value

- Validation Strategy: Strict chronological walk-forward validation (10 rolling folds) preventing temporal leakage (no random shuffling). Precision-Recall AUC (PR-AUC) and Brier score evaluated alongside ROC-AUC to address class imbalance (~10-20% positive class). Mann-Whitney U tests for segment divergence.

- Decision Value: Delivers an actionable early-warning signal (LRS bands, HRE probabilities) enabling stakeholders to dynamically resize positions, mitigate slippage, and avoid illiquid exit traps during market stress.

## 8. Limitations and Next Steps

- Assumptions & Limitations: Reliance on daily proxy estimators rather than true Level-2 spreads; survivorship bias from current index constituents (mitigated by tracking); provider omission of special Saturday sessions.

- Milestones & Deviations: 1. Generate 55 features across all 300 stocks (Phase 5). 2. Execute formal EDA and hypothesis testing (Phase 6). 3. Calibrate LRS weights and label hre_5d (Phase 7). 4. Train walk-forward ML models (Phase 8). Deviations: None; strictly adheres to approved blueprint.
