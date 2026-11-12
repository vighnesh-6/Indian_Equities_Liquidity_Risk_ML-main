## ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities

Project overview and methodology reference

What this project is: A quantitative decision-support platform that measures liquidity risk, estimates the likelihood of a high-risk trading event over the next five trading days, identifies unusual historical stock-day behaviour, and helps users inspect market stress through an interactive dashboard.

The system is designed for risk analysis—not to predict stock prices, issue buy/sell signals, or execute trades automatically.

## 1. Project purpose and questions

The project studies daily trading data for a broad cross-section of Indian equities to turn observable price and volume behaviour into interpretable risk information. It combines transparent financial rules, supervised machine learning, unsupervised anomaly detection, and market-event analysis.

- Liquidity: How difficult might a stock be to trade, based on observable daily market-data proxies?

- Forward risk: Given information available today, how likely is a defined high-risk event to occur during the next five trading days?

- Unusual behaviour: Is a stock-day observation statistically unusual compared with historical patterns, without needing a future-event label?

- Market stress: How do liquidity and trading-risk measures behave around major market-wide shocks?

- Practical presentation: How can these results be explored at market, individual-stock, portfolio, and data-quality levels?

## 2. Data universe and sources

The README defines a 300-stock universe designed to cover different market-cap segments, with a broad-market benchmark and a volatility-based fear index.

| Component | Scope / role |
| --- | --- |
| 100 large-cap stocks | All constituents of the Nifty 100; a liquid baseline anchor. |
| 150 mid-cap stocks | All constituents of the Nifty Midcap 150; the core segment for risk variation. |
| 50 small-cap stocks | Selected from Nifty Smallcap 250 using a six-month trailing average daily traded value (ADTV) |
|   | ranking. |
| Nifty 50 | Market benchmark for excess returns, beta, correlation, and other market-relative features. |
| India VIX | Supplementary indicator of market-wide expected volatility. |
| Market-event calendar | Curated dates for elections, Union Budgets, geopolitical escalations, credit freezes, and |
|   | exchange halts. |

The stated historical target window is January 2018 to October 2026. Daily OHLCV data are obtained through yfinance using adjusted prices (auto_adjust=True). The README explicitly excludes scraping NSE India pages, Indian broker APIs, company earnings/news/transcripts, and NLP or sentiment analysis.

OHLCV means open, high, low, close, and volume. These daily observations form the basis for feature calculations; they do not provide full order-book or intraday execution information.


## 3. Overall methodology: three complementary outputs

Each stock-day observation is represented by a feature vector, documented as F01–F55. The pipeline then branches into three outputs that answer different questions.

| Output | Method | What it tells us |
| --- | --- | --- |
| Liquidity Risk Score | Transparent, rule-based composite score from | A description of the stock’s current/instantaneous |
| (LRS) | 0 to 100 using percentile-ranked liquidity | liquidity-risk state. |
|   | proxies. |   |
| High-Risk Event | Supervised classifiers trained on a binary | Estimated probability of a defined high-risk event |
| probability | forward target, HRE_5d. | within the next five trading days. |
| Anomaly score / flag | Unsupervised Isolation Forest. | Whether today’s observed pattern looks historically |
|   |   | unusual; not a forecast of a future event. |

These outputs are intentionally distinct. The LRS is a formula-based score, HRE_5d is a supervised forward-risk estimate, and Isolation Forest detects outliers without relying on future labels.

## 4. The 55-feature engine (F01–F55)

The README groups the 55 engineered features into six categories. Together, they represent price behaviour, volatility, liquidity proxies, price gaps and impact proxies, market-relative behaviour, and broad market stress.

| Feature category | Count | Examples / purpose |
| --- | --- | --- |
| Returns & price behaviour | 10 | Daily and log returns, 5/20/60-day momentum, rolling return z-scores, absolute |
|   |   | returns, moving-average deviations. |
| Volatility & downside risk | 12 | Realized volatility over multiple windows, Parkinson high-low volatility, ATR, downside |
|   |   | semi-deviation, drawdown, RSI-14, volatility-of-volatility. |
| Liquidity & trading activity | 14 | Volume measures and z-scores, Amihud illiquidity, turnover, zero-volume/stale-price |
|   |   | flags, low-volume streaks. |
| Price-gap & price-impact | 9 | Normalized range, overnight gaps, Corwin–Schultz high-low spread estimate, intraday |
| proxies |   | return. |
| Market-relative behaviour | 6 | Excess return vs Nifty 50, rolling beta and correlation, idiosyncratic volatility, relative |
|   |   | volume. |
| Market-stress features | 5 | Benchmark volatility, market volatility-regime ratio, market return, volatility |
|   |   | acceleration, VIX integration. |

The feature design includes established microstructure-inspired proxies, notably the Amihud illiquidity ratio and the Corwin–Schultz high-low spread estimator. They are estimates derived from daily OHLCV, not direct measurements of bid–ask quotes, order-book depth, or realized execution costs.

## 5. How the Liquidity Risk Score works

The LRS is a transparent, formula-driven composite score ranging from 0 to 100. It combines percentile-ranked proxies intended to characterize the stock’s liquidity state on a given day.

- Amihud illiquidity proxy: relates absolute price movement to traded value, approximating price impact per unit of trading activity.

- Corwin–Schultz spread estimator: estimates a bid–ask spread proxy from daily high and low prices; the README says the estimate is floored at zero.

- Volume drought: captures unusually weak trading activity relative to a stock’s own recent history.

- Range dispersion: uses price-range behaviour as an additional liquidity-related signal.


A higher score is intended to indicate greater liquidity risk under the implemented scoring convention. The README refers to configurable LRS weights in feature configuration but does not specify the actual numerical weights in the README itself.

## 6. Supervised prediction of HRE_5d

The supervised task is to estimate the probability of a High-Risk Event occurring within the next five trading days.

The target HRE_5d is binary and depends on future observations from t+1 through t+5; the README does not define the exact event threshold in the README text.

## Models listed

- Logistic Regression: interpretable baseline classifier for estimating event probability.

- Random Forest: ensemble of decision trees that can capture nonlinear relationships and feature interactions.

- HistGradientBoosting: gradient-boosted tree classifier designed to model complex patterns efficiently.

## Validation and leakage prevention

- Expanding-window walk-forward validation: models train on earlier dates and are evaluated on later dates, preserving chronological order.

- Fold-specific scaling: StandardScaler is fit only on the training portion of each fold, then applied to its validation portion.

- Target separation: because HRE_5d requires future outcomes, rows intended for live scoring have HRE_5d set to NaN rather than using unknown future labels.

- Warm-up period: the first 80 trading days for each stock are marked is_warmup=True and excluded from model fitting.

This design is intended to make evaluation more realistic by preventing future information from leaking into model training. Actual predictive performance still needs to be assessed using the project's completed evaluation outputs.

## 7. Unsupervised anomaly detection

Isolation Forest is used to identify stock-day observations that look unusual relative to the patterns learned from historical feature data. It is independent of the future HRE_5d label.

- The model produces an anomaly score and/or flag for unusual behaviour.

- An anomaly is not automatically a loss, a liquidity crisis, or a predicted future event; it is a statistical signal for further inspection.

- The anomaly output complements the LRS and HRE_5d probability rather than replacing either one.

## 8. Market-stress and event-study analysis

The project includes a curated calendar of market-wide events—such as elections, Union Budgets, geopolitical escalations, credit freezes, and exchange halts—and uses India VIX as a supplementary market-fear/expected-volatility measure.

Event-study analysis can compare how market and stock-level risk measures behave around event dates. The README identifies this as an objective, but it does not state specific event windows, statistical tests, or final findings; those should be documented when implemented.

## 9. Dashboard and intended user workflow

The planned Streamlit dashboard organizes outputs into five pages:

- Overview: market-wide risk rankings and summary view.


- Stock Detail: drill-down into an individual stock’s risk measures and history.

- Portfolio Risk: aggregate liquidity-risk concentration across a set of holdings.

- Anomaly: inspect flagged unusual stock-day observations.

- Health / Data Quality: monitor dataset quality and pipeline status.

The goal is to make quantitative risk measures inspectable and useful for analysis, rather than present a black-box buy/sell recommendation.

## 10. Data pipeline, storage, and engineering design

The architecture separates configuration, raw data, cleaned interim data, engineered features, predictions, model artifacts, reports, tests, and the dashboard.

- Raw layer: downloaded Parquet files for 300 stocks and three market indices; retained separately from cleaned data.

- Interim layer: validated and cleaned OHLCV datasets, including stock and market data.

- Processed layer: the F01–F55 feature matrix plus warm-up indicator.

- Prediction layer: daily outputs including LRS, HRE probability, and anomaly score.

- Audit and manifest layer: data-collection manifests and SHA-256-based raw-file integrity verification.

- Model and reporting layers: serialized model artifacts, model registry, Markdown reports, and figures.

## Efficiency and reliability choices

- Sequential stock-by-stock processing to keep memory use predictable.

- Chunked Parquet writes to avoid accumulating the full dataset in memory.

- float32 for price and return columns to reduce memory footprint.

- Atomic file replacement to reduce the risk of partially written/corrupted outputs.

- A pytest offline test suite, with the README reporting 34 passing unit tests.

## 11. Limitations and interpretation

- Proxy-only liquidity: daily OHLCV cannot directly observe order-book depth, quoted spreads, or actual execution costs.

- Survivorship bias: current index constituents may exclude companies that were delisted or failed during the historical period.

- Daily granularity: close-to-close measures cannot capture intraday flash illiquidity.

- Third-party data dependency: yfinance/Yahoo Finance is an undocumented public endpoint subject to rate limits and symbol adjustments.

Consequently, the outputs should be interpreted as estimates and decision-support indicators, not as a guarantee of liquidity, a forecast of returns, or financial advice.


## 12. Current implementation status (as stated in the README)

| Phase | README status |
| --- | --- |
| Phase 1 — Data acquisition design | Completed: universe defined in configs/universe.json. |
| Phase 2 — Microstructure proxy formulation | Completed: F01–F55 documented in configs/feature_config.yaml. |
| Phase 3 — Pipeline infrastructure | Completed: modular fetcher, validator, and storage managers implemented. |
| Phase 4.1 — Historical ingestion | Completed: 300 stocks (565,190 rows) and three market indices acquired and verified. |
| Phase 4.2 — Cleaning & interim dataset | Completed: cleaned interim Parquet datasets, EDA report, and 34 passing unit tests. |
| Next — Phase 5 | Feature Engineering Pipeline (F01–F55), producing data/processed/features.parquet. |

The row count and phase status above reproduce the README’s stated status. This overview does not independently audit the repository or infer completion of phases that the README marks as future work.

## 13. End-to-end workflow

| Step | Activity | Output / purpose |
| --- | --- | --- |
| 1 | Acquire and verify daily OHLCV | Audited raw stock and index data. |
| 2 | Clean and align data | Validated interim datasets and EDA quality report. |
| 3 | Engineer F01–F55 | Stock-day feature matrix and warm-up flags. |
| 4 | Calculate LRS | Rule-based current liquidity-risk score. |
| 5 | Build HRE_5d target and train classifiers | Forward high-risk-event probability estimates with walk-forward |
|   |   | evaluation. |
| 6 | Fit Isolation Forest | Unsupervised unusualness scores/flags. |
| 7 | Analyse market events and stress | Event-period comparisons and market context. |
| 8 | Present results in Streamlit | Overview, stock, portfolio, anomaly, and data-quality views. |
| 9 | Automate post-market refresh | Updated data, features, scores, predictions, and manifest logs. |

## 14. Key terms

| Term | Meaning in this project |
| --- | --- |
| ADTV | Average Daily Traded Value; used to rank small-cap candidates by trailing trading activity. |
| OHLCV | Open, High, Low, Close, and Volume. |
| LRS | Liquidity Risk Score; rule-based 0–100 composite score. |
| HRE_5d | Binary target indicating a defined high-risk event within the next five trading days. |
| Walk-forward validation | Chronological evaluation in which training data precede the validation period. |
| Feature leakage | Use of information during training that would not have been available at prediction time. |
| Isolation Forest | Unsupervised model that identifies observations that are relatively isolated in feature space. |
| India VIX | Market volatility index used as a supplementary indicator of expected volatility. |

Source basis: README.md supplied with the project. This document expands its descriptions into a readable project reference while preserving the methods, scope, stated limitations, and status that the README supports. Where the README does not specify a


formula, threshold, weight, or result, this overview says so rather than inventing one.
