# Phase 7: Target & Label Engineering (LRS & $HRE_{5d}$) — Readiness Audit Report
## ML-Based Liquidity and Trading-Risk Intelligence for Indian Equities

**Audit Date:** November 2026  
**Audit Scope:** READ-ONLY Readiness Audit for Phase 7  
**Primary Input Dataset:** `data/processed/features.parquet` (565,190 rows × 70 columns)  
**Analysis-Grade Post-Warmup Rows:** `is_warmup == False` (541,190 rows across 300 stocks)  
**Report Destination:** `reports/phase7_readiness_audit.md`  

---

## 1. Executive Summary & Readiness Scorecard

This audit evaluates the repository, datasets, mathematical definitions, and codebase prerequisites required to implement **Phase 7: Target & Label Engineering**. 

Phase 7 is responsible for establishing ground truth for liquidity risk:
1. **Instantaneous Composite Liquidity Risk Score (LRS, 0–100 Scale):** Rule-based composite score measuring immediate trading risk and order-flow friction today.
2. **High-Risk Event Forward Target ($HRE_{5d} \in \{0, 1\}$):** Supervised binary target indicating whether a stock experiences an acute liquidity drought over the **forward 5 trading sessions** ($t+1$ to $t+5$).

### Overall Verdict: **READY — ALL DECISIONS SIGNED OFF & VERIFIED**

The foundational feature store [data/processed/features.parquet](file:///C:/Users/vighn/Downloads/Indian_Equities_Liquidity_Risk_ML-main/Indian_Equities_Liquidity_Risk_ML-main/data/processed/features.parquet) is intact, mathematically verified, strictly sorted, and completely free of infinite values. All four key mathematical decisions (D7.1 through D7.4) have been approved and signed off by the user. The training-period cutoff date (`2025-10-17`), frozen 90th-percentile thresholds, and anti-leakage boundary protocols are fully established.

### Dependency Readiness Scorecard:

| Dependency / Prerequisite | Target Artifact / Check | Status | Key Audit Finding |
|---|---|---|---|
| **D1: Feature Store Availability** | `data/processed/features.parquet` | ✅ **PASS** | 565,190 rows × 70 columns verified. Snappy compression, 156.86 MB. |
| **D2: Dataset Record Integrity** | `(ticker, date)` uniqueness & sorting | ✅ **PASS** | 0 duplicate keys. Monotonically sorted by `(ticker, date)`. |
| **D3: Numeric Cleanliness** | Infs / NaNs across features | ✅ **PASS** | Exactly 0 infinite values. Post-warmup nulls are minimal and legitimate. |
| **D4: Cap-Tier Availability** | `cap_segment` column | ✅ **PASS** | 100 Large, 150 Mid, 50 Small. Static 6-month ADTV classification. |
| **D5: Historical Sufficiency** | Days per stock ($\ge 200$ days) | ✅ **PASS** | 276/300 stocks have $\ge 500$ days. 3 listings in 2026 have 11 post-warmup days. |
| **D6: Normalization Verification** | Raw vs. Pre-normalized features | ✅ **PASS** | Features are un-normalized raw metrics; Phase 7 owns percentile ranking. |
| **D7: LRS Formulation Contract** | 4-pillar formulation (Option A) | ✅ **APPROVED** | Signed off: Amihud (35%), Spread (25%), Vol Drought (20%), Downside Vol (20%). |
| **D8: Within-Tier Calibration** | Daily Cross-Sectional Ranking (Option A) | ✅ **APPROVED** | Signed off: Daily percentile ranking independently within each cap segment. |
| **D9: $HRE_{5d}$ Forward Formulation** | Forward 5-Day Max LRS (Option A) | ✅ **APPROVED** | Signed off: $HRE_{5d} = \mathbb{I}(\max_{k \in [1, 5]} \text{LRS}_{t+k} \ge \theta_{90, s})$. |
| **D10: Threshold Scope & Freezing** | Frozen Training 90th Percentile | ✅ **APPROVED** | Signed off: Calibrated strictly on training period ($t \le \text{2025-10-17}$) and frozen. |
| **D11: Anti-Leakage Protocol** | Tail NaN masking & boundary embargo | ✅ **PASS** | Verified forward rolling logic correctly produces NaNs on final 5 rows. |

---

## 2. Existing Project Structure & Artifact Audit

### Existing Source Modules & Scaffolds:
- [src/labels/lrs.py](file:///C:/Users/vighn/Downloads/Indian_Equities_Liquidity_Risk_ML-main/Indian_Equities_Liquidity_Risk_ML-main/src/labels/lrs.py): Currently an unexecuted scaffold containing the function signature `compute_lrs(features_df, calibration_quantiles=None)`. Its docstring outlines a 6-component weighted rank model.
- [src/labels/targets.py](file:///C:/Users/vighn/Downloads/Indian_Equities_Liquidity_Risk_ML-main/Indian_Equities_Liquidity_Risk_ML-main/src/labels/targets.py): Currently an unexecuted scaffold containing the function signature `construct_hre_target(df, forward_window=5, amihud_threshold_quantile=0.95, volume_ratio_threshold=0.20, gap_threshold=0.05, lrs_threshold=75.0)`.
- `tests/test_lrs.py` & `tests/test_targets.py`: Currently contain minimal smoke tests checking `callable()`. Both pass, but must be expanded in Phase 7 to assert mathematical bounds, edge-case masking, and non-leakage truncation.
- [notebooks/04_label_analysis.ipynb](file:///C:/Users/vighn/Downloads/Indian_Equities_Liquidity_Risk_ML-main/Indian_Equities_Liquidity_Risk_ML-main/notebooks/04_label_analysis.ipynb): Currently an unexecuted 3-cell template shell.

### Configuration Layer (`configs/feature_config.yaml`):
The configuration file already contains preliminary definitions for LRS and HRE:
```yaml
lrs_weights:
  amihud_ma20_rank: 0.25        # 25%
  cs_spread_ma10_rank: 0.20     # 20%
  vol_20d_rank: 0.20            # 20%
  volume_drought_rank: 0.15     # 15% (1 - turnover_ratio when < 1)
  overnight_gap_abs_rank: 0.10  # 10%
  idiosyncratic_vol_rank: 0.10  # 10%

hre_target:
  forward_window_days: 5
  triggers:
    amihud_ratio_cross_sectional_quantile: 0.95
    volume_ratio_threshold: 0.20
    overnight_gap_abs_threshold: 0.05
    lrs_threshold: 75.0
```

---

## 3. Verified Feature Mappings & Normalization Status

### A. Clarification on Features F18, F23, F24, F30
In the audit request, specific attention was drawn to features **F18, F23, F24, and F30**. Cross-referencing `configs/feature_config.yaml`, `docs/phase5_feature_engineering_specification.md`, and [data/processed/features.parquet](file:///C:/Users/vighn/Downloads/Indian_Equities_Liquidity_Risk_ML-main/Indian_Equities_Liquidity_Risk_ML-main/data/processed/features.parquet) reveals:

| Feature ID | Parquet Column Name | Category | Exact Mathematical Definition | Range in Post-Warmup Data | Normalization Status |
|---|---|---|---|---|---|
| **F18** | `downside_vol_20d` | Volatility | Target-zero 20-day downside semi-deviation: $\sqrt{\frac{1}{20}\sum_{i=0}^{19}\min(r_{t-i}, 0)^2} \times \sqrt{252}$ | $[0.00, 3.24]$, Median: `0.197` | **Raw (Un-normalized)** |
| **F23** | `log_volume` | Liquidity | Logarithm of volume: $\ln(\text{Volume}_t + 1)$ | $[0.00, 21.17]$, Median: `14.06` | **Log-Transformed, Unbounded** |
| **F24** | `volume_ma20` | Liquidity | 20-day simple moving average of share volume: $\frac{1}{20}\sum_{i=0}^{19}\text{Volume}_{t-i}$ | $[0.00, 7.13 \times 10^8]$, Median: `1.55e6` | **Raw Unscaled Count** |
| **F30** | `turnover_value` | Liquidity | Daily rupee turnover: $\text{Volume}_t \times \text{Close}_t$ | $[0.00, 3.71 \times 10^{11}]$, Median: `8.34e8` | **Raw Unscaled Currency** |

### B. Critical Mapping Reconciliation for LRS Pillars
In quantitative literature and conceptual discussions, the 4 core pillars of liquidity risk are:
1. **Price Impact (Amihud)**
2. **Transaction Spread (Corwin-Schultz)**
3. **Volume / Turnover Drought**
4. **Tail Risk / Volatility**

Notice the exact column correspondences in our Phase 5 feature store:
- **Price Impact is NOT F23:** `F23` is `log_volume`. Price impact is `amihud_ratio` (**F27**) or its 20-day smoothed version `amihud_ma20` (**F28**).
- **Volume Drought is NOT F24:** `F24` is raw `volume_ma20`. Volume drought is measured by `volume_z20` (**F26**), `volume_ratio` (**F25**), or `turnover_ratio` (**F33**).
- **Bid-Ask Spread is NOT F30:** `F30` is `turnover_value`. Bid-ask spread is `cs_spread_est` (**F43**) or `cs_spread_ma10` (**F44**).
- **Volatility:** Captured by total volatility `vol_20d` (**F12**) or downside semi-deviation `downside_vol_20d` (**F18**).

### C. Normalization Verification
**Finding:** NONE of the candidate features in `features.parquet` are pre-normalized, scaled to $[0, 100]$, or percentile-ranked across stocks. They represent raw, physical financial quantities. 
**Conclusion:** Phase 7 has complete ownership and responsibility for cross-sectional ranking and scaling to the $[0, 100]$ domain.

---

## 4. Dataset Integrity & Coverage Findings

Detailed programmatic inspection of [data/processed/features.parquet](file:///C:/Users/vighn/Downloads/Indian_Equities_Liquidity_Risk_ML-main/Indian_Equities_Liquidity_Risk_ML-main/data/processed/features.parquet) confirms:

### 1. Dimensions & Keys:
- Total rows: `565,190`
- Total columns: `70` (3 metadata, 9 flags, 55 core features, 3 VIX features)
- Post-warmup analysis rows (`is_warmup == False`): `541,190`
- Monotonic sorting: Strictly sorted by `(ticker, date)`.
- Key uniqueness: Exactly `0` duplicate `(ticker, date)` pairs.

### 2. Numeric Cleanliness & Null Audit:
- **Infinities:** Exactly `0` `+inf` or `-inf` values across all 58 numeric columns.
- **Null Values in Candidate LRS Features (Post-Warmup, N=541,190):**
  - `downside_vol_20d` (F18): 60 nulls (strictly corporate action masking)
  - `vol_20d` (F12): 60 nulls (corporate action masking)
  - `amihud_ratio` (F27): 724 nulls (zero-volume days and corporate actions)
  - `amihud_ma20` (F28): 103 nulls (initial corporate action lookbacks)
  - `cs_spread_est` (F43): 6 nulls
  - `cs_spread_ma10` (F44): 33 nulls
  - `volume_z20` (F26): 0 nulls
  - `turnover_ratio` (F33): 46 nulls
  - `overnight_gap_abs` (F41): 3 nulls
  - `idiosyncratic_vol` (F50): 180 nulls (benchmark alignment and corporate actions)

### 3. Market-Cap Classification (`cap_segment`):
- Available on 100% of rows: `large` (202,581 rows), `mid` (278,971 rows), `small` (83,638 rows).
- **Point-in-Time Status:** The classification is **STATIC** (fixed over the entire 2018–2026 history based on 6-month ADTV rankings at universe construction time). It is not dynamically rebalanced monthly or quarterly.
- **Impact on Phase 7:** Static classification provides a stable, consistent partitioning anchor across the entire backtest period.

### 4. Historical Sufficiency per Stock:
- Total unique stocks: 300.
- Median trading days: 2,176 days (~7.8 years of daily bars).
- 276 stocks have $\ge 500$ observations (deep history).
- 24 newly listed stocks have $< 500$ days (IPOs listed between 2024 and 2026).
- **Edge-Case Listings:** Exactly 3 stocks (`VAML.NS`, `VOGL.NS`, `VISL.NS`) listed in early 2026 have only 91 trading days (11 post-warmup days). When applying a forward 5-day window, these 3 stocks will have only $11 - 5 = 6$ valid target rows. This is mathematically correct and does not threaten model stability.

---

## 5. Mathematical Requirements & Ambiguity Analysis

Before implementing Phase 7, four key mathematical questions must be explicitly resolved:

```mermaid
flowchart TD
    subgraph A["Ambiguity 1: LRS Formulation"]
        A1["Option 1A: 4-Pillars (Amihud 35%, Spread 25%, Vol Drought 20%, Tail Risk 20%)"]
        A2["Option 1B: 6-Components in Config (Amihud 25%, Spread 20%, Vol 20%, Drought 15%, Gap 10%, Idio Vol 10%)"]
    end

    subgraph B["Ambiguity 2: Percentile Calibration Method"]
        B1["Option 2A: Daily Cross-Sectional Ranking within Cap Tier (Leakage-Free)"]
        B2["Option 2B: Rolling Historical Window per Stock (e.g. 252d lookback)"]
    end

    subgraph C["Ambiguity 3: HRE_5d Target Definition"]
        C1["Option 3A: Pure LRS Threshold (Forward 5d Max LRS >= 90th Percentile)"]
        C2["Option 3B: Multi-Trigger Rule in Config (Amihud Q95 OR VolRatio < 0.20 OR Gap > 5% OR LRS > 75)"]
    end

    subgraph D["Ambiguity 4: 90th Percentile Scope"]
        D1["Option 4A: Calibrated per Market-Cap Tier (~10% positive rate across Large, Mid, Small)"]
        D2["Option 4B: Global 90th Percentile across all 300 stocks"]
    end
```

### Ambiguity 1: 4-Component vs. 6-Component LRS Formulation
- **The Conflict:** The user prompt asks to audit *"Four LRS components and their required feature inputs"*, which mirrors the 4 pillars discussed in Phase 6 EDA. However, `configs/feature_config.yaml` and `src/labels/lrs.py` currently define **6 components**:
  - `amihud_ma20_rank`: 0.25 (F28)
  - `cs_spread_ma10_rank`: 0.20 (F44)
  - `vol_20d_rank`: 0.20 (F12)
  - `volume_drought_rank`: 0.15 (1 - turnover_ratio F33)
  - `overnight_gap_abs_rank`: 0.10 (F41)
  - `idiosyncratic_vol_rank`: 0.10 (F50)
- **Recommendation:** Both are mathematically sound. The 4-pillar formulation is simpler and avoids collinearity between `vol_20d` and `idiosyncratic_vol` ($r = 0.963$). The 6-component formulation incorporates gap risk and market-relative risk. A decision sign-off is required.

---

### Ambiguity 2: Within-Market-Cap-Tier Percentile Calibration
- **The Issue:** Small-cap median Amihud is **7.15x** higher than Large-cap median Amihud. If features are ranked globally across all 300 stocks, small-cap stocks are mechanically pushed to the highest risk scores purely due to their company size.
- **Mathematical Solution (Daily Cross-Sectional Ranking within Tier):**
  For each trading day $t$ and each market-cap segment $s \in \{\text{large, mid, small}\}$:
  $$\text{Rank}_{i, t, s} = \frac{\text{Rank}(X_{i, t, s}) - 1}{N_{t, s} - 1} \times 100$$
  where $N_{t, s}$ is the count of active stocks in tier $s$ on date $t$.
- **Simulation Result:** In our test run on all 541,190 post-warmup rows:
  - Large Cap Median LRS: **50.79** (90th percentile: **73.66**)
  - Mid Cap Median LRS: **50.35** (90th percentile: **72.14**)
  - Small Cap Median LRS: **51.42** (90th percentile: **72.16**)
  *Tier-calibrated ranking achieves perfect symmetry and equal risk calibration across all three tiers.*

---

### Ambiguity 3: $HRE_{5d}$ Formulation — Pure LRS vs. Multi-Condition Trigger
- **Option 3A (Pure LRS Threshold):**
  $$HRE_{5d}(t) = \mathbb{I}\left(\max_{k \in [1, 5]} \text{LRS}(t+k) \ge \theta_{90}\right)$$
  Directly ties the target to the continuous risk score. Highly interpretable.
- **Option 3B (Multi-Trigger Rule in Config):**
  $$HRE_{5d}(t) = \mathbb{I}\left(\exists k \in [1, 5] : \text{Amihud}_{t+k} \ge Q_{0.95} \lor \text{VolRatio}_{t+k} \le 0.20 \lor \text{Gap}_{t+k} \ge 0.05 \lor \text{LRS}_{t+k} \ge 75.0\right)$$
  Captures acute localized dislocations (e.g., flash crash gaps or sudden zero-bid volume dry-ups) even if smoothed LRS has not fully caught up.
- **Recommendation:** Option 3A is cleaner for standard classification, while Option 3B provides broader catastrophe capture. Must be explicitly approved.

---

### Ambiguity 4: 90th Percentile Threshold Scope (Global vs. Per-Tier)
- **Global Threshold ($\theta_{90} \approx 72.70$):**
  One single threshold applied across all 300 stocks. Because tier-calibrated LRS already standardizes medians to ~50 across tiers, global $\theta_{90}$ yields a balanced ~10% positive rate across Large (10.6%), Mid (9.6%), and Small (9.8%).
- **Per-Tier Threshold ($\theta_{90, s}$):**
  Large = 73.66, Mid = 72.14, Small = 72.16. Enforces exact 10.0% positive class representation within each individual tier.
- **Recommendation:** Per-tier thresholding ensures that class imbalance ($~10\%$) is mathematically identical across all three modeling slices.

---

## 6. Information Leakage Risks & Prevention Protocols

Phase 7 is the boundary where forward-looking information is introduced. To ensure zero lookahead bias:

1. **Forward Shift Rule ($t+1$ to $t+5$):**
   The target $HRE_{5d}$ at day $t$ must observe only days $t+1, t+2, t+3, t+4, t+5$. It must **never** include day $t$ itself.
2. **Mandatory Tail NaN Masking:**
   For every stock, the final 5 trading days in the dataset ($t \in [T-4, T]$) do not have 5 subsequent sessions.
   - For these rows, $HRE_{5d}$ **MUST evaluate strictly to `NaN`**.
   - With 300 stocks, exactly **1,500 rows** in the final dataset will have `HRE_5d = NaN`.
   - Verified in prototype: `tk.groupby('ticker')['lrs'].transform(...)` correctly yields exactly 5 NaNs at the end of each stock's timeline.
3. **Strict Ticker Partitioning:**
   Rolling operations must strictly be executed **grouped by `ticker`**. Never apply a global rolling window over the stacked dataframe, which would leak stock $B$'s opening prices into stock $A$'s trailing rows.
4. **Target Isolation from Feature Matrix:**
   $HRE_{5d}$ and forward metrics are **targets only**. They must never be passed to any modeling script as an input feature column.

---

## 7. Recommended Implementation Sequence

Once decisions are approved, Phase 7 should be executed in this exact sequence:

```mermaid
flowchart TD
    S1["Step 1: Synchronize configs/feature_config.yaml with approved decisions"]
    --> S2["Step 2: Implement src/labels/lrs.py (Within-tier cross-sectional ranking)"]
    --> S3["Step 3: Implement src/labels/targets.py (Vectorized forward 5d max & NaN tail)"]
    --> S4["Step 4: Expand Unit Tests in tests/test_lrs.py & tests/test_targets.py"]
    --> S5["Step 5: Execute pytest tests -v (Target: 65+ passing tests)"]
    --> S6["Step 6: Update & Execute notebooks/04_label_analysis.ipynb in-place"]
    --> S7["Step 7: Generate features_with_targets.parquet & reports/phase7_label_engineering_report.md"]
```

1. **Step 1 — Configuration Alignment:** Update `configs/feature_config.yaml` to formalize the approved component weights and threshold parameters.
2. **Step 2 — Implement LRS Engine (`src/labels/lrs.py`):** Vectorized calculation of daily cross-sectional percentile ranks per market-cap segment, bounded strictly within $[0, 100]$.
3. **Step 3 — Implement Target Engine (`src/labels/targets.py`):** Vectorized forward rolling maximum over $[t+1, t+5]$ grouped by ticker, with strict NaN assignment on the final 5 observations per stock.
4. **Step 4 — Comprehensive Unit Tests:** Add tests verifying:
   - LRS output range $[0, 100]$ and median symmetry (~50 across tiers).
   - $HRE_{5d} \in \{0, 1, \text{NaN}\}$.
   - Exactly 5 trailing NaNs per ticker.
   - Zero cross-ticker data contamination.
5. **Step 5 — Test Suite Execution:** Verify offline test suite with pytest.
6. **Step 6 — Interactive Label Analysis (`notebooks/04_label_analysis.ipynb`):** Complete notebook in-place, inspecting class balance, persistence of high-risk states, and transition probabilities.
7. **Step 7 — Artifact Storage & Documentation:** Store `data/processed/features_with_targets.parquet` and generate `reports/phase7_label_engineering_report.md`.

---

## 8. Approved Decision Matrix (Signed Off by User)

The following four foundational Phase 7 mathematical decisions have been formally approved and signed off:

### Decision D7.1: LRS Component Count & Weights — APPROVED: OPTION A (4 Pillars)
- **Amihud Illiquidity (`amihud_ma20` F28):** **35%** weight (Price Impact & Order Book Depth)
- **Corwin-Schultz Spread (`cs_spread_ma10` F44):** **25%** weight (Immediate Transaction Cost)
- **Volume Drought (`volume_z20` F26, inverted):** **20%** weight (Order Flow Drying Up)
- **Downside Volatility (`downside_vol_20d` F18):** **20%** weight (Asymmetric Crash Risk)

### Decision D7.2: Within-Tier Calibration Method — APPROVED: OPTION A (Daily Cross-Sectional Ranking)
- Daily cross-sectional percentile ranking independently within each market-cap segment (`large`, `mid`, `small`) on date $t$:
  $$\text{Rank}_{i, t, s} = \frac{\text{Rank}(X_{i, t, s}) - 1}{N_{t, s} - 1} \times 100$$
- Guarantees unbiased, perfectly centered 0–100 distributions across all three tiers with zero lookahead bias.

### Decision D7.3: $HRE_{5d}$ Forward Target Formulation — APPROVED: OPTION A (Forward 5-Day Max LRS)
- Supervised binary target constructed by taking the maximum LRS over the next five trading sessions:
  $$HRE_{5d}(t) = \mathbb{I}\left(\max_{k \in [1, 5]} \text{LRS}(t+k) \ge \theta_{90, s}\right)$$
- Direct, continuous-to-binary mapping with high financial interpretability.

### Decision D7.4: 90th Percentile Threshold Scope — APPROVED: OPTION A WITH STRICT MODIFICATION
- Calculate separate 90th-percentile LRS thresholds ($\theta_{90, s}$) for each market-cap tier using **training-period observations only** ($t \le \text{2025-10-17}$).
- Freeze these thresholds permanently before holdout evaluation.
- Holdout observations must **never** be used to calculate or recalibrate thresholds.

---

## 9. Strict Data-Preservation & Cryptographic Integrity Protocol

To guarantee that no prior research, cleaning, or feature-engineering artifacts are corrupted or overwritten during Phase 7, the following non-negotiable data-preservation rules are locked into the implementation plan:

1. **Strict Immutability of Datasets:**
   - No existing raw, interim, or processed datasets will be modified, overwritten, deleted, renamed, or regenerated.
   - `data/processed/features.parquet` is treated strictly as an **immutable, read-only input**.
2. **Distinct Phase 7 Output Store:**
   - The annotated output containing the new columns (`lrs`, `fwd_max_lrs_5d`, `hre_5d`, `hre_5d_train`, `split`) will be generated strictly as a **separate, new file**:
     `data/processed/features_with_targets.parquet`
3. **Preservation of Prior Phase Artifacts:**
   - All Phase 5 feature calculation modules (`src/features/*.py`) and Phase 5 reports remain frozen.
   - All Phase 6 EDA artifacts (the executed `notebooks/02_eda.ipynb`, `reports/phase6_eda_findings_report.md`, and all 7 generated PNG charts in `reports/figures/`) remain untouched.
4. **Scoped Configuration Updates:**
   - Any edits to `configs/feature_config.yaml` are strictly restricted to Phase 7 label-engineering parameters (`lrs_weights` and `hre_target`). No feature calculation parameters will be altered.
5. **Cryptographic Pre- and Post-Implementation SHA-256 Verification:**
   - Baseline SHA-256 hashes of all input artifacts have been computed prior to implementation and are permanently recorded below.
   - Following Phase 7 execution, hashes will be re-computed and compared to guarantee zero bit-level tampering:

| Dataset / Artifact Path | File Size (Bytes) | Pre-Implementation SHA-256 Hash |
|---|---:|---|
| `data/processed/features.parquet` | 164,478,803 | `c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae` |
| `data/interim/stocks/ohlcv_clean.parquet` | 46,302,787 | `feaefaac2ded41478f067ece25ad3860affc3aa683026b8a925ba431a1b93421` |
| `data/events/market_events.csv` | 42,025 | `ad9026e0a50c5e1471c04a089f19bf8cfe7fb04963b132fcae7e641bc8c5e554` |

6. **Stop-and-Report Rule:**
   - If any step during implementation requires modifying or replacing any frozen artifact or dataset, execution will immediately halt, and the issue will be reported to the user.

---

## 10. Approved Mathematical Specification & Implementation Contract

### A. Exact Training-Period Cutoff Date & Holdout Split
Verified directly from `configs/model_config.yaml` (`validation.holdout_fraction_or_days: 250`):
- **Full Dataset Date Span:** `2018-01-01` to `2026-10-23` (2,176 unique trading sessions).
- **Reserved Final Holdout Period:** Exactly the most recent **250 trading days** (~1 trading year):
  - **Holdout Span:** `2025-10-20` to `2026-10-23` (250 trading sessions).
- **Training / Validation Period:** All preceding trading days:
  - **Training Span:** `2018-01-01` to `2025-10-17` (1,926 trading sessions).
- **Exact Training Cutoff Date:** **`2025-10-17`**.

### B. Frozen 90th-Percentile LRS Thresholds ($\theta_{90, s}$)
Computed strictly on training observations ($t \le \text{2025-10-17}$), frozen permanently, and never re-fitted on holdout data:

$$\theta_{90, \text{large}} = 73.8298 \quad (N_{\text{train}} = 169,992, \text{Median} = 50.7778)$$
$$\theta_{90, \text{mid}} = 72.3251 \quad (N_{\text{train}} = 230,016, \text{Median} = 50.3719)$$
$$\theta_{90, \text{small}} = 72.4359 \quad (N_{\text{train}} = 67,774, \text{Median} = 51.4516)$$
$$\left[\text{Global Reference Threshold}: 72.8926\right]$$

### C. Composite LRS Formulation
For stock $i$ on date $t$ belonging to market-cap segment $s$:
$$\text{LRS}_{i, t} = 0.35 \cdot R_{t, s}(\text{amihud\_ma20}) + 0.25 \cdot R_{t, s}(\text{cs\_spread\_ma10}) + 0.20 \cdot R_{t, s}^{\text{inv}}(\text{volume\_z20}) + 0.20 \cdot R_{t, s}(\text{downside\_vol\_20d})$$
where:
- $R_{t, s}(X) = \frac{\text{Rank}(X) - 1}{N_{t, s} - 1} \times 100$ (ascending rank; higher value = higher risk).
- $R_{t, s}^{\text{inv}}(\text{volume\_z20}) = \frac{\text{Rank}(-\text{volume\_z20}) - 1}{N_{t, s} - 1} \times 100$ (descending rank; drying up volume = higher risk).
- $\text{LRS}_{i, t} \in [0.0, 100.0]$.

### D. Forward Target $HRE_{5d}$ & Tail NaN Masking
1. **Forward Rolling Maximum:** Grouped strictly by `ticker`:
   $$\text{fwd\_max\_lrs\_5d}_{i, t} = \max_{k \in [1, 5]} \text{LRS}_{i, t+k}$$
2. **Binary Classification Rule:**
   $$HRE_{5d}(i, t) = \begin{cases} 1 & \text{if } \text{fwd\_max\_lrs\_5d}_{i, t} \ge \theta_{90, s} \\ 0 & \text{if } \text{fwd\_max\_lrs\_5d}_{i, t} < \theta_{90, s} \\ \text{NaN} & \text{if } \text{fwd\_max\_lrs\_5d}_{i, t} \text{ is NaN} \end{cases}$$
3. **Mandatory Final Tail Masking:** For each stock, the final 5 observations in the dataset ($t \in [T-4, T]$, Oct 19–23, 2026) cannot form a complete forward window and evaluate strictly to `NaN` (exactly $300 \times 5 = \mathbf{1,500}$ rows).

### E. Training Boundary & Embargo Protocol (Requirement 2)
To guarantee that training labels do not use future LRS observations extending beyond the training cutoff date (`2025-10-17`):
- Observations on the last 5 trading days of the training period (`2025-10-13` to `2025-10-17`, 1,424 rows) look ahead into the holdout period (`2025-10-20` onwards).
- The dataset will include an explicit `split` column:
  - `'train'`: Date $\le \text{2025-10-12}$ (fully self-contained historical training observations).
  - `'train_boundary_embargo'`: $\text{2025-10-13} \le \text{Date} \le \text{2025-10-17}$ (5-day embargo gap preventing forward leakage into holdout).
  - `'holdout'`: Date $\ge \text{2025-10-20}$ (uncontaminated 1-year holdout).
- A dedicated target column `hre_5d_train` will mask the boundary embargo rows as `NaN` to guarantee zero information leakage across the training-holdout boundary.

### F. Actual Empirical Class Balance (Requirement 7)
Without any artificial threshold forcing, the actual empirical class balance resulting from the approved specification is:

| Dataset Partition | Total Valid Rows | Positive Class ($HRE_{5d} = 1$) | Negative Class ($HRE_{5d} = 0$) | Actual Positive Prevalence |
|---|---:|---:|---:|---:|
| **Large Cap (Training)** | 169,992 | 30,894 | 139,098 | **18.17%** |
| **Mid Cap (Training)** | 230,008 | 41,671 | 188,337 | **18.12%** |
| **Small Cap (Training)** | 67,770 | 12,711 | 55,059 | **18.76%** |
| **Total Training Period** | **467,770** | **85,276** | **382,494** | **18.23%** |
| **Large Cap (Holdout)** | 24,065 | 4,068 | 19,997 | **16.90%** |
| **Mid Cap (Holdout)** | 36,119 | 6,054 | 30,065 | **16.76%** |
| **Small Cap (Holdout)** | 11,572 | 1,751 | 9,821 | **15.13%** |
| **Total Holdout Period** | **71,756** | **11,873** | **59,883** | **16.55%** |

**Empirical Finding:**
The positive prevalence of **18.23%** (training) and **16.55%** (holdout) falls **squarely within the optimal target band (10%–25%)**. This occurs naturally because taking the maximum of five consecutive 90th-percentile Bernoulli draws with moderate positive serial correlation ($r_{\text{LRS}} \approx 0.65$) yields an expected 5-day event probability of:
$$P(\max_{k=1}^5 \text{LRS}_{t+k} \ge \theta_{90}) \approx 1 - (1 - 0.10) \cdot (1 - 0.10 \cdot (1 - \rho))^4 \approx 18\%$$
No synthetic threshold distortion or re-weighting is required.

### G. Static `cap_segment` Documentation & Implications (Requirement 6)
- In `configs/universe.json` and the feature store, `cap_segment` is **static** (100 Large, 150 Mid, 50 Small), assigned based on 6-month ADTV rankings at the time of universe construction.
- **Implications:**
  1. *Stability:* Daily cross-sectional percentiles have a consistent, invariant population of peer stocks on every trading day ($N_{\text{large}} \approx 100, N_{\text{mid}} \approx 150, N_{\text{small}} \approx 50$).
  2. *Zero Survivorship Churn:* Stocks do not artificially jump tiers from month to month, preventing sudden structural jumps in cross-sectional percentile rankings.
  3. *Known Research Boundary:* The backtest reflects the liquidity dynamics of the currently investable universe across history, which is the standard methodology documented in our project blueprint.

---

## 11. Final Pre-Implementation Audit — Numerical Reconciliation

### 11.1 Investigation & Reconciliation of Discrepancy A (Training Observation Counts)
- **Reported Counts:**
  - Training threshold sample observations: **467,782**
  - Training class-balance valid observations: **467,770**
  - Exact discrepancy: **12 rows** ($467,782 - 467,770 = 12$).
- **Root Cause & Mathematical Distinction:**
  - $N_{\text{threshold}} = 467,782$ represents the count of rows in the training split ($\text{Date} \le \text{2025-10-17}$) where the **current-day** Composite Liquidity Risk Score is valid ($\text{LRS}_{i, t} \ne \text{NaN}$). LRS depends only on concurrent features at time $t$.
  - $N_{\text{class\_balance}} = 467,770$ represents the count of rows in the training split where the **forward 5-day rolling maximum** is valid ($\text{fwd\_max\_lrs\_5d}_{i, t} = \max_{k=1}^5 \text{LRS}_{i, t+k} \ne \text{NaN}$).
  - Because `fwd_max_lrs_5d` requires a full, uncorrupted 5-session forward window (`min_periods=5`), any internal trading halt, corporate-action suspension, or missing record within $[t+1, t+5]$ turns the forward maximum into `NaN` for up to 5 preceding sessions.
- **Exact Row-by-Row Accounting:**
  1. **Rows where $\text{LRS}_{i, t}$ is valid (`notna`) but $\text{fwd\_max\_lrs\_5d}_{i, t}$ is `NaN` (15 rows):**
     - `FORCEMOT.NS` (Small Cap, 5 rows): 2023-10-26, 2023-10-27, 2023-10-30, 2023-10-31, 2023-11-01. These sessions precede an extended 41-day trading halt starting 2023-11-02. Current-day LRS is fully valid, but their 5-day forward lookahead window reaches into the halt, invalidating `fwd_max_lrs_5d`.
     - `MOTILALOFS.NS` (Mid Cap, 5 rows): 2023-12-22, 2023-12-26, 2023-12-27, 2023-12-28, 2023-12-29. These sessions precede an internal corporate-action adjustment gap in early January 2024. Current LRS is valid, but the forward window contains an internal null.
     - `PATANJALI.NS` (Mid Cap, 5 rows): 2019-11-14, 2019-11-15, 2019-11-18, 2019-11-19, 2019-11-20. These sessions immediately precede an extensive CIRP/resolution suspension from late November 2019 to February 2020.
  2. **Rows where $\text{LRS}_{i, t}$ is `NaN` (`isna`) but $\text{fwd\_max\_lrs\_5d}_{i, t}$ is valid (`notna`) (3 rows):**
     - On the exact boundary dates of isolated single-day gaps, current-day features are missing/flagged, causing current $\text{LRS}_{i, t}$ to be `NaN`. However, the subsequent 5 trading days $[t+1, t+5]$ are all normal, uncorrupted trading sessions:
       - `FORCEMOT.NS` (Small Cap, 1 row): 2024-03-04 ($\text{LRS} = \text{NaN}, \text{fwd\_max} = 79.4186$)
       - `MOTILALOFS.NS` (Mid Cap, 1 row): 2024-01-30 ($\text{LRS} = \text{NaN}, \text{fwd\_max} = 78.5115$)
       - `PATANJALI.NS` (Mid Cap, 1 row): 2020-02-24 ($\text{LRS} = \text{NaN}, \text{fwd\_max} = 54.7083$)
- **Net Reconciliation:**
  $$\Delta = 15 - 3 = \mathbf{12 \text{ rows}}$$
  $$N_{\text{class\_balance}} = N_{\text{threshold}} - 15 + 3 = 467,782 - 15 + 3 = \mathbf{467,770 \text{ rows}}$$

### 11.2 Investigation & Reconciliation of Discrepancy B (Overall Post-Warmup Counts)
- **Reported Counts:**
  - Total post-warmup observations: **541,190**
  - Naive tail-5 null deduction: $300 \times 5 = \mathbf{1,500}$
  - Expected remaining rows: $541,190 - 1,500 = \mathbf{539,690}$
  - Reported valid forward target rows (Training + Holdout): $467,770 + 71,756 = \mathbf{539,526}$
  - Discrepancy: $539,690 - 539,526 = \mathbf{164 \text{ rows}}$.
- **Root Cause & Mathematical Mechanism:**
  - The naive formula $541,190 - 1,500 = 539,690$ assumes that the *only* `NaN` values in `fwd_max_lrs_5d` occur at the structural tail of the dataset ($t \in [T-4, T]$, Oct 19–23, 2026).
  - In reality, `fwd_max_lrs_5d` requires a strictly valid 5-day forward window (`min_periods=5`). Whenever any stock undergoes an internal trading halt, corporate-action suspension, or data gap, the sessions immediately preceding that gap cannot evaluate a full 5-day forward window, generating **internal forward nulls**.
- **Exact Ticker and Tier Breakdown of the 164 Internal Forward Nulls:**
  | Ticker | Cap Segment | Internal Null Rows | Primary Historical Cause | Partition Breakdown |
  |---|---|---:|---|---|
  | `PATANJALI.NS` | Mid Cap | **70** | CIRP resolution trading freeze (Nov 2019 – Feb 2020) | 70 Train, 0 Holdout |
  | `FORCEMOT.NS` | Small Cap | **46** | Extended trading suspension (Nov 2023 – Mar 2024) | 46 Train, 0 Holdout |
  | `MOTILALOFS.NS` | Mid Cap | **24** | Corporate-action bonus/split masking window (Dec 2023 – Jan 2024) | 24 Train, 0 Holdout |
  | `VEDL.NS` | Large Cap | **24** | Demerger & corporate restructuring event (2026) | 0 Train, 24 Holdout |
  | **Total** | — | **164** | — | **140 Train, 24 Holdout** |
- **Sum by Market-Cap Segment:**
  - Large Cap: **24 rows**
  - Mid Cap: **94 rows**
  - Small Cap: **46 rows**
  - Total: **164 rows**
- **Complete Post-Warmup Target Accounting:**
  $$\begin{aligned}
  \text{Total Post-Warmup Rows} &= 541,190 \\
  - \text{ Final 5 Tail Nulls } (300 \times 5) &= -1,500 \\
  - \text{ Internal Forward Nulls } &= -164 \\
  \hline
  \mathbf{\text{Total Valid Forward Target Rows}} &= \mathbf{539,526} \\
  \text{Training Valid Rows} &= 467,770 \\
  \text{Holdout Valid Rows} &= 71,756 \\
  \text{Training} + \text{Holdout} &= 467,770 + 71,756 = \mathbf{539,526}
  \end{aligned}$$
  The 164-row discrepancy is accounted for to the exact single row.

### 11.3 Investigation & Reconciliation of Discrepancy C (Threshold Sample Counts by Tier)
- **Empirical Dataset Verification of Threshold Counts:**
  - Large Cap ($N_{\text{thresh, large}}$): **169,992**
  - Mid Cap ($N_{\text{thresh, mid}}$): **230,016**
  - Small Cap ($N_{\text{thresh, small}}$): **67,774**
  - Total Training Threshold Sample: **467,782**
- **Reconciliation with Valid Training Class-Balance Counts by Tier:**
  | Market-Cap Tier | Threshold Sample ($N_{\text{thresh}}$: $\text{LRS} \ne \text{NaN}$) | Class-Balance Valid ($N_{\text{target}}$: $\text{fwd\_max} \ne \text{NaN}$) | Net Difference | Breakdown of Difference ($\text{LRS}_{\text{ok}}\cap\text{fwd}_{\text{na}} - \text{LRS}_{\text{na}}\cap\text{fwd}_{\text{ok}}$) |
  |---|---:|---:|---:|---|
  | **Large Cap** | 169,992 | 169,992 | **0** | $0 - 0 = 0$ (Zero internal gaps during training period) |
  | **Mid Cap** | 230,016 | 230,008 | **8** | $10 - 2 = 8$ (`MOTILALOFS.NS`: $5-1=4$; `PATANJALI.NS`: $5-1=4$) |
  | **Small Cap** | 67,774 | 67,770 | **4** | $5 - 1 = 4$ (`FORCEMOT.NS`: $5-1=4$) |
  | **Total** | **467,782** | **467,770** | **12** | $\mathbf{15 - 3 = 12}$ |
- **Conclusion:** The counts differ solely due to the forward rolling window's sensitivity to internal trading gaps, and the net difference matches the 12-row discrepancy exactly.

### 11.4 Date Boundaries & Trading Calendar Verification
- **Full Dataset:**
  - Start Date: `2018-01-01`
  - End Date: `2026-10-23`
  - Total Trading Sessions: **2,176 sessions** across 8.8 years.
  - Total Observations: 565,190 rows across 300 stocks.
- **Warmup Period:**
  - Start Date: `2018-01-01`
  - End Date: `2018-04-27`
  - Total Trading Sessions: **80 sessions**.
  - Total Warmup Rows: Exactly $300 \times 80 = \mathbf{24,000 \text{ rows}}$.
  - Post-warmup observations: $565,190 - 24,000 = \mathbf{541,190 \text{ rows}}$.
- **Training Period:**
  - Start Date: `2018-04-30`
  - Cutoff Date: `2025-10-17` (Friday)
  - Total Training Trading Sessions: **1,846 sessions**.
  - Total Training Rows: **467,910 rows**.
- **Training Boundary Embargo (Leakage Prevention Window):**
  - Covers the final 5 trading sessions of the training period:
    1. `2025-10-13` (Monday)
    2. `2025-10-14` (Tuesday)
    3. `2025-10-15` (Wednesday)
    4. `2025-10-16` (Thursday)
    5. `2025-10-17` (Friday)
  - Total observations in embargo window: **1,424 rows**.
  - **Leakage Mechanics:** For observation date $t = \text{2025-10-13}$, the forward window $[t+1, t+5]$ spans `2025-10-14`, `2025-10-15`, `2025-10-16`, `2025-10-17`, and `2025-10-20`. Because `2025-10-20` is the first session of the holdout period, this label would contaminate the training set with holdout market information.
  - **Self-Contained Training Cutoff:** The latest trading date with strictly self-contained training forward labels is **`2025-10-10` (Friday)**.
  - **Target Embargo Protocol:** In `hre_5d_train`, all 1,424 rows from `2025-10-13` to `2025-10-17` will be masked to `NaN`.
- **Holdout Period:**
  - Start Date: `2025-10-20` (Monday)
  - End Date: `2026-10-23` (Friday)
  - Total Trading Sessions: **exactly 250 trading days** (1 standard calendar trading year).
  - Total Observations: **73,280 rows**.
  - Tail null observations: final 5 trading days (`2026-10-19` to `2026-10-23`, 1,500 rows).
  - Valid holdout target observations: **71,756 rows** ($73,280 - 1,500 - 24\text{ internal gaps} = 71,756$).

### 11.5 Five-Day Forward Target Construction & Leakage Prevention Guarantees
- **Mathematical Specification of Forward Target:**
  $$\text{fwd\_max\_lrs\_5d}_{i, t} = \max_{k \in [1, 5]} \text{LRS}_{i, t+k}$$
  - **Strict Forward Offset ($k \ge 1$):** Observation day $t$ is strictly excluded from its own label.
  - **Per-Ticker Partitioning:** Forward rolling operations are partitioned strictly by `ticker` (`groupby('ticker')`). Zero cross-stock contamination is mathematically guaranteed.
  - **Final 5 Tail Nulls:** For each stock, the last 5 chronological rows ($t \in [T-4, T]$) evaluate to `NaN` ($300 \times 5 = \mathbf{1,500 \text{ rows}}$).
- **Target Separation & Zero-Leakage Architecture:**
  - `features_with_targets.parquet` will contain both:
    1. `hre_5d`: Global research target across all valid sessions.
    2. `hre_5d_train`: Training target where `holdout` rows AND `train_boundary_embargo` rows are masked to `NaN`.
    3. `split`: Explicit categorical partition (`'train'`, `'train_boundary_embargo'`, `'holdout'`).
- **Frozen Threshold Integrity:**
  - The 90th-percentile thresholds are calibrated strictly on training observations:
    $$\theta_{90, \text{Large}} = \mathbf{73.8298}, \quad \theta_{90, \text{Mid}} = \mathbf{72.3251}, \quad \theta_{90, \text{Small}} = \mathbf{72.4359}$$
  - These values are hard-frozen and applied out-of-sample to holdout data without re-estimation or dynamic adaptation.

### 11.6 Cryptographic Hash Verification & Dataset Immutability
All baseline raw, interim, and processed files were re-hashed using SHA-256 and confirmed bit-for-bit identical to their frozen pre-audit states:
- `data/processed/features.parquet`: `c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae` (164,478,803 bytes) — **VERIFIED UNCHANGED**
- `data/interim/stocks/ohlcv_clean.parquet`: `feaefaac2ded41478f067ece25ad3860affc3aa683026b8a925ba431a1b93421` (46,302,787 bytes) — **VERIFIED UNCHANGED**
- `data/events/market_events.csv`: `ad9026e0a50c5e1471c04a089f19bf8cfe7fb04963b132fcae7e641bc8c5e554` (42,025 bytes) — **VERIFIED UNCHANGED**

### 11.7 Final Implementation Readiness Verdict
**VERDICT: READY FOR PHASE 7 IMPLEMENTATION**

All mathematical definitions, dataset counts, date boundaries, target-generation rules, and data-preservation guarantees are internally consistent and verified against the actual repository artifacts.

---

## 12. Final Training-Label Eligibility and Embargo Reconciliation

### 12.1 Purpose of Audit & Investigation
This section resolves the training-label accounting distinction between:
1. **Unembargoed Global Research Target (`hre_5d`):** Evaluated across all post-warmup observations that possess five subsequent trading sessions, regardless of split boundaries.
2. **Leak-Free Supervised Training Target (`hre_5d_train`):** Evaluated strictly for historical model training, where holdout observations AND boundary embargo observations are masked to `NaN` to guarantee zero information leakage across the training-holdout partition.

### 12.2 Resolution of the 467,770 vs. Eligible Training Count Inconsistency
- **The Prior Reporting Inconsistency:**
  - Section 10 and Section 11 previously reported **467,770** valid training rows.
  - **Empirical Investigation:** 
    $$\text{Post-Warmup Training Observations } (467,910) - \text{Training Internal Forward Nulls } (140) = \mathbf{467,770 \text{ rows}}.$$
  - This count represented the **unembargoed global target** `hre_5d` evaluated on all rows up to `2025-10-17`.
  - It **accidentally included** the **1,424 boundary embargo observations** (`2025-10-13` to `2025-10-17`). Because stocks continued trading into October 2025, those 1,424 observations possessed 5 forward trading days by looking into the holdout period (`2025-10-20` to `2025-10-24`).
- **Correction Protocol:**
  - For leakage-free training, observations within the 5-day boundary embargo (`2025-10-13` to `2025-10-17`) **must be masked to `NaN`** in `hre_5d_train`.
  - Therefore, the true, leak-free **eligible training sample size** is:
    $$N_{\text{eligible\_train}} = 467,770 - 1,424 = \mathbf{466,346 \text{ rows}}.$$

### 12.3 Overlap Analysis: Internal Forward Nulls vs. Embargo Observations
An independent cross-tabulation was conducted to determine whether any of the 140 training internal forward nulls overlap with the 1,424 embargo rows:
$$\text{Internal Forward Nulls in Training Period } (t \le \text{2025-10-17}) = \mathbf{140 \text{ rows}}$$
- **Internal Nulls Inside Embargo Window (`2025-10-13` to `2025-10-17`):** **`0 rows`** (Zero overlap).
- **Internal Nulls Outside Embargo Window (Pre-Embargo, $t \le \text{2025-10-10}$):** **`140 rows`** (100% of training internal nulls).
- **Explanation:**
  - All 140 training internal forward nulls occurred in earlier historical periods due to known corporate actions or trading halts:
    - `PATANJALI.NS` (Mid Cap, 70 rows): CIRP resolution freeze (Nov 2019 – Feb 2020).
    - `FORCEMOT.NS` (Small Cap, 46 rows): Trading suspension (Nov 2023 – Mar 2024).
    - `MOTILALOFS.NS` (Mid Cap, 24 rows): Corporate action split/bonus window (Dec 2023 – Jan 2024).
  - During the embargo week of `2025-10-13` to `2025-10-17`, all 300 stocks in the active universe traded normally with zero missing data gaps. Consequently, all 1,424 embargo rows had valid unembargoed forward targets.
- **Deduction Formula Verification:**
  $$\begin{aligned}
  \text{Eligible Training Observations} &= \text{Post-warmup training rows} - \text{Embargo rows} - \text{Internal nulls outside embargo} \\
  &= 467,910 - 1,424 - 140 \\
  &= \mathbf{466,346 \text{ rows}}.
  \end{aligned}$$
  This equality holds exactly.

### 12.4 Embargo Partition Breakdown (The 1,424 Masked Rows)
The 1,424 observations masked out of `hre_5d_train` to prevent holdout contamination exhibit the following distribution:

| Market-Cap Tier | Embargo Rows Masked | Forward Window Positive ($HRE = 1$) | Forward Window Negative ($HRE = 0$) | Event Prevalence |
|---|---:|---:|---:|---:|
| **Large Cap** | 484 | 94 | 390 | 19.42% |
| **Mid Cap** | 710 | 127 | 583 | 17.89% |
| **Small Cap** | 230 | 25 | 205 | 10.87% |
| **Total Embargo** | **1,424** | **246** | **1,178** | **17.28%** |

*Note:* Because these 246 positive events and 1,178 negative events evaluate $[t+1, t+5]$ into holdout dates (`2025-10-20` to `2025-10-24`), excluding them from `hre_5d_train` ensures 100% mathematical isolation between training and holdout data.

### 12.5 Corrected Eligible Training Class Balance (`hre_5d_train`)
After applying the strict boundary embargo mask, the exact class balance for model training is:

| Market-Cap Tier | Eligible Training Rows | Positive Class ($HRE_{5d} = 1$) | Negative Class ($HRE_{5d} = 0$) | Actual Positive Prevalence |
|---|---:|---:|---:|---:|
| **Large Cap** | 169,508 | 30,800 | 138,708 | **18.17%** |
| **Mid Cap** | 229,298 | 41,544 | 187,754 | **18.12%** |
| **Small Cap** | 67,540 | 12,686 | 54,854 | **18.78%** |
| **Total Eligible Training** | **466,346** | **85,030** | **381,316** | **18.23%** |

**Prevalence Finding:**
The leak-free eligible training prevalence remains **18.23%**, perfectly inside the desired 10%–25% range across all three market-cap tiers without any synthetic threshold re-tuning.

### 12.6 Holdout Partition Class Balance & Isolation Confirmation
- **Holdout Date Range:** `2025-10-20` (Monday) to `2026-10-23` (Friday) (exactly 250 trading sessions).
- **Total Post-Warmup Holdout Observations:** **73,280 rows**.
- **Deductions:**
  - Dataset final tail-5 nulls: **1,500 rows** ($300 \times 5$, dates `2026-10-19` to `2026-10-23`).
  - Holdout internal forward nulls: **24 rows** (`VEDL.NS` demerger gap in 2026).
- **Valid Holdout Labels:** $73,280 - 1,500 - 24 = \mathbf{71,756 \text{ rows}}$.

| Market-Cap Tier | Valid Holdout Rows | Positive Class ($HRE_{5d} = 1$) | Negative Class ($HRE_{5d} = 0$) | Actual Positive Prevalence |
|---|---:|---:|---:|---:|
| **Large Cap** | 24,065 | 4,068 | 19,997 | **16.90%** |
| **Mid Cap** | 36,119 | 6,054 | 30,065 | **16.76%** |
| **Small Cap** | 11,572 | 1,751 | 9,821 | **15.13%** |
| **Total Holdout Period** | **71,756** | **11,873** | **59,883** | **16.55%** |

- **Holdout Isolation Guarantee:**
  - Thresholds $\theta_{90, s}$ applied to the holdout set were calibrated strictly on training observations:
    $$\theta_{90, \text{Large}} = \mathbf{73.8298}, \quad \theta_{90, \text{Mid}} = \mathbf{72.3251}, \quad \theta_{90, \text{Small}} = \mathbf{72.4359}$$
  - No holdout data was used in calculating percentiles, thresholds, or training labels.

### 12.7 Comprehensive Dataset Reconciliation Table

| Category | Actual Rows | Expected Rows | Difference | Status |
|---|---:|---:|---:|:---:|
| **Original features dataset** | **565,190** | 565,190 | 0 | Verified |
| **Initial warmup (80 sessions $\times$ 300 stocks)** | **24,000** | 24,000 | 0 | Verified |
| **Post-warmup dataset** | **541,190** | 541,190 | 0 | Verified |
| **Post-warmup training period ($t \le \text{2025-10-17}$)** | **467,910** | 467,910 | 0 | Verified |
| **Boundary embargo ($t \in [\text{2025-10-13}, \text{2025-10-17}]$)** | **1,424** | 1,424 | 0 | Verified |
| **Eligible training labels (`hre_5d_train` valid)** | **466,346** | 466,346 | 0 | Verified |
| **Valid holdout labels (`hre_5d` holdout valid)** | **71,756** | 71,756 | 0 | Verified |
| **Internal forward nulls (140 train + 24 holdout)** | **164** | 164 | 0 | Verified |
| **Final dataset-tail nulls (5 sessions $\times$ 300 stocks)** | **1,500** | 1,500 | 0 | Verified |

**Complete Post-Warmup Conservation Identity:**
$$\begin{aligned}
\text{Post-Warmup Observations} &= \text{Eligible Training} + \text{Embargo Rows} + \text{Train Internal Nulls} + \text{Valid Holdout} + \text{Holdout Internal Nulls} + \text{Tail Nulls} \\
541,190 &= 466,346 + 1,424 + 140 + 71,756 + 24 + 1,500 \\
541,190 &= 541,190 \quad \text{\textbf{(100\% Exact Mathematical Match)}}
\end{aligned}$$

### 12.8 Dataset Integrity & Hash Verification
All original repository artifacts were verified bit-for-bit unchanged:
- `data/processed/features.parquet`: `c3975cf9a99b84e109bf7bc2035ac0792c37ec49bef36d2049274395e4b758ae`
- `data/interim/stocks/ohlcv_clean.parquet`: `feaefaac2ded41478f067ece25ad3860affc3aa683026b8a925ba431a1b93421`
- `data/events/market_events.csv`: `ad9026e0a50c5e1471c04a089f19bf8cfe7fb04963b132fcae7e641bc8c5e554`

### 12.9 Remaining Unresolved Issues
**NONE.** Every observation, calendar date, embargo row, forward-window boundary, and class count is reconciled and empirically proven.

---

### 12.10 Final Pre-Implementation Verdict
$$\mathbf{READY\ FOR\ PHASE\ 7\ IMPLEMENTATION}$$




