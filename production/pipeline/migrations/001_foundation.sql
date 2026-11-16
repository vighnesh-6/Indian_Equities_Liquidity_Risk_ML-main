-- Forward migration. Apply through the identity-checked migration runner.
CREATE TABLE liquidity.instruments (
    ticker text PRIMARY KEY,
    instrument_type text NOT NULL CHECK (instrument_type IN ('stock', 'index', 'volatility_index')),
    name text NOT NULL,
    is_active boolean NOT NULL DEFAULT true
);
CREATE TABLE liquidity.stocks (
    ticker text PRIMARY KEY REFERENCES liquidity.instruments(ticker),
    company_name text NOT NULL,
    cap_segment text NOT NULL CHECK (cap_segment IN ('large', 'mid', 'small')),
    sector text,
    isin text NOT NULL UNIQUE,
    universe_version text NOT NULL,
    universe_sha256 text NOT NULL CHECK (universe_sha256 ~ '^[a-f0-9]{64}$'),
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE liquidity.ingestion_runs (
    run_id uuid PRIMARY KEY,
    request_key text NOT NULL UNIQUE,
    plan_sha256 text NOT NULL CHECK (plan_sha256 ~ '^[a-f0-9]{64}$'),
    mode text NOT NULL CHECK (mode IN ('bootstrap', 'catchup', 'daily')),
    range_start date NOT NULL,
    range_end date NOT NULL CHECK (range_end >= range_start),
    expected_instruments text[] NOT NULL CHECK (cardinality(expected_instruments) > 0),
    status text NOT NULL DEFAULT 'RUNNING' CHECK (status IN ('RUNNING', 'SUCCESS', 'FAILED')),
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    error_code text CHECK (error_code ~ '^[A-Z][A-Z0-9_]{0,63}$'),
    CHECK ((status = 'RUNNING' AND finished_at IS NULL AND error_code IS NULL)
        OR (status = 'SUCCESS' AND finished_at IS NOT NULL AND error_code IS NULL)
        OR (status = 'FAILED' AND finished_at IS NOT NULL AND error_code IS NOT NULL))
);
CREATE TABLE liquidity.ingestion_run_instruments (
    run_id uuid NOT NULL REFERENCES liquidity.ingestion_runs(run_id),
    ticker text NOT NULL REFERENCES liquidity.instruments(ticker),
    planned_sessions date[] NOT NULL CHECK (cardinality(planned_sessions) > 0 AND array_position(planned_sessions, NULL) IS NULL),
    PRIMARY KEY (run_id, ticker)
);
CREATE TABLE liquidity.data_provenance (
    provenance_id uuid PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES liquidity.ingestion_runs(run_id),
    source_name text NOT NULL,
    source_version text NOT NULL,
    adjustment_mode text NOT NULL CHECK (adjustment_mode IN ('adjusted', 'unadjusted')),
    retrieved_at timestamptz NOT NULL,
    payload_sha256 text NOT NULL CHECK (payload_sha256 ~ '^[a-f0-9]{64}$'),
    UNIQUE (provenance_id, run_id)
);
CREATE TABLE liquidity.ingestion_batches (
    batch_id uuid PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES liquidity.ingestion_runs(run_id),
    ticker text NOT NULL REFERENCES liquidity.instruments(ticker),
    batch_key text NOT NULL,
    payload_sha256 text NOT NULL CHECK (payload_sha256 ~ '^[a-f0-9]{64}$'),
    provenance_id uuid NOT NULL,
    range_start date NOT NULL,
    range_end date NOT NULL CHECK (range_end >= range_start),
    row_count integer NOT NULL CHECK (row_count > 0),
    completed_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (run_id, ticker, batch_key),
    FOREIGN KEY (run_id, ticker) REFERENCES liquidity.ingestion_run_instruments(run_id, ticker),
    FOREIGN KEY (provenance_id, run_id) REFERENCES liquidity.data_provenance(provenance_id, run_id),
    UNIQUE (batch_id, ticker)
);
CREATE TABLE liquidity.market_eod (
    ticker text NOT NULL REFERENCES liquidity.instruments(ticker),
    date date NOT NULL,
    open double precision NOT NULL CHECK (open > 0 AND open < 'Infinity'::float8),
    high double precision NOT NULL CHECK (high > 0 AND high < 'Infinity'::float8),
    low double precision NOT NULL CHECK (low > 0 AND low < 'Infinity'::float8),
    close double precision NOT NULL CHECK (close > 0 AND close < 'Infinity'::float8),
    volume bigint NOT NULL CHECK (volume >= 0),
    turnover double precision CHECK (turnover >= 0 AND turnover < 'Infinity'::float8),
    provenance_id uuid NOT NULL REFERENCES liquidity.data_provenance(provenance_id),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, date),
    CHECK (high >= low AND open BETWEEN low AND high AND close BETWEEN low AND high)
);
CREATE TABLE liquidity.instrument_checkpoints (
    ticker text PRIMARY KEY REFERENCES liquidity.instruments(ticker),
    last_date date,
    last_batch_id uuid,
    revision bigint NOT NULL DEFAULT 0 CHECK (revision >= 0),
    updated_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (last_batch_id, ticker) REFERENCES liquidity.ingestion_batches(batch_id, ticker),
    FOREIGN KEY (ticker, last_date) REFERENCES liquidity.market_eod(ticker, date),
    CHECK ((last_date IS NULL AND last_batch_id IS NULL AND revision = 0)
        OR (last_date IS NOT NULL AND last_batch_id IS NOT NULL AND revision > 0))
);
CREATE TABLE liquidity.daily_features (
    ticker text NOT NULL REFERENCES liquidity.stocks(ticker),
    date date NOT NULL,
    daily_return real,
    return_5d real,
    return_20d real,
    return_z20 real,
    abs_return real,
    price_rel_ma20 real,
    vol_5d real,
    vol_20d real,
    vol_ratio_5_20 real,
    downside_vol_20d real,
    max_drawdown_20d real,
    rsi_14 real,
    vol_of_vol real,
    log_volume real,
    volume_z20 real,
    amihud_ratio real,
    amihud_ma20 real,
    amihud_z20 real,
    log_turnover real,
    turnover_ratio real,
    low_volume_streak real,
    daily_range real,
    daily_range_ma20 real,
    range_z20 real,
    overnight_gap real,
    overnight_gap_abs real,
    cs_spread_est real,
    cs_spread_ma10 real,
    excess_return_1d real,
    rolling_beta_60d real,
    rolling_corr_60d real,
    relative_volume real,
    nifty_vol_20d real,
    nifty_return_20d real,
    vol_beta real,
    vix_close real,
    volume_ratio real,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, date),
    FOREIGN KEY (ticker, date) REFERENCES liquidity.market_eod(ticker, date)
);
CREATE TABLE liquidity.pipeline_runs (
    run_id uuid PRIMARY KEY,
    run_date date NOT NULL,
    request_key text NOT NULL UNIQUE,
    ingestion_run_id uuid NOT NULL REFERENCES liquidity.ingestion_runs(run_id),
    triggered_by text NOT NULL,
    status text NOT NULL CHECK (status IN ('RUNNING', 'SUCCESS', 'FAILED')),
    duration_seconds double precision CHECK (duration_seconds >= 0 AND duration_seconds < 'Infinity'::float8),
    stocks_processed integer CHECK (stocks_processed BETWEEN 0 AND 300),
    error_code text CHECK (error_code ~ '^[A-Z][A-Z0-9_]{0,63}$'),
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX pipeline_one_success_per_date ON liquidity.pipeline_runs(run_date) WHERE status = 'SUCCESS';
CREATE TABLE liquidity.daily_predictions (
    ticker text NOT NULL,
    date date NOT NULL,
    pipeline_run_id uuid NOT NULL REFERENCES liquidity.pipeline_runs(run_id),
    lrs_score double precision NOT NULL CHECK (lrs_score BETWEEN 0 AND 100),
    lrs_band text NOT NULL,
    lrs_90p_exceeded boolean NOT NULL,
    hre_prob double precision NOT NULL CHECK (hre_prob BETWEEN 0 AND 1),
    hre_flag smallint NOT NULL CHECK (hre_flag IN (0, 1)),
    isolation_score double precision NOT NULL CHECK (isolation_score > '-Infinity'::float8 AND isolation_score < 'Infinity'::float8),
    anomaly_score_scaled double precision NOT NULL CHECK (anomaly_score_scaled BETWEEN 0 AND 100),
    is_anomaly boolean NOT NULL,
    model_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, date),
    FOREIGN KEY (ticker, date) REFERENCES liquidity.daily_features(ticker, date),
    CHECK (hre_flag = CASE WHEN hre_prob >= 0.534 THEN 1 ELSE 0 END)
);
CREATE INDEX market_eod_date ON liquidity.market_eod(date DESC);
CREATE INDEX daily_features_date ON liquidity.daily_features(date DESC);
CREATE INDEX daily_predictions_date ON liquidity.daily_predictions(date DESC);
CREATE INDEX ingestion_batches_coverage ON liquidity.ingestion_batches(run_id, ticker, range_end DESC);
CREATE INDEX ingestion_runs_status ON liquidity.ingestion_runs(status, started_at DESC);
