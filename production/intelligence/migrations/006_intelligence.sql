-- Independent extension ledger preserves the deployed 000-005 collector contract.
CREATE TABLE liquidity.intelligence_schema_migrations (
    version text PRIMARY KEY,
    sha256 text NOT NULL CHECK (sha256 ~ '^[a-f0-9]{64}$'),
    applied_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE liquidity.intelligence_versions (
    feature_version uuid PRIMARY KEY REFERENCES liquidity.feature_versions(feature_version),
    contract_sha256 text NOT NULL CHECK (contract_sha256 ~ '^[a-f0-9]{64}$'),
    model_hashes jsonb NOT NULL,
    thresholds jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TRIGGER intelligence_version_immutable BEFORE UPDATE OR DELETE ON liquidity.intelligence_versions
FOR EACH ROW EXECUTE FUNCTION liquidity.feature_immutable();
CREATE TABLE liquidity.intelligence_observations (
    feature_version uuid NOT NULL REFERENCES liquidity.intelligence_versions(feature_version),
    date date NOT NULL CHECK (date <> DATE '2026-11-08'),
    ticker text NOT NULL,
    lrs double precision CHECK (lrs BETWEEN 0 AND 100),
    hre_probability double precision CHECK (hre_probability BETWEEN 0 AND 1),
    isolation_score double precision CHECK (isolation_score > '-Infinity'::float8 AND isolation_score < 'Infinity'::float8),
    forward_lrs double precision CHECK (forward_lrs BETWEEN 0 AND 100),
    hre_label smallint CHECK (hre_label IN (0,1)),
    horizon_end date,
    PRIMARY KEY(feature_version,date,ticker),
    FOREIGN KEY(ticker,date,feature_version) REFERENCES liquidity.feature_values(ticker,date,feature_version),
    CHECK (horizon_end IS NULL OR horizon_end>date),
    CHECK ((hre_label IS NULL AND forward_lrs IS NULL) OR (hre_label IS NOT NULL AND forward_lrs IS NOT NULL AND horizon_end IS NOT NULL))
);
CREATE TABLE liquidity.intelligence_publications (
    feature_version uuid NOT NULL REFERENCES liquidity.intelligence_versions(feature_version),
    date date NOT NULL,
    rows integer NOT NULL CHECK (rows BETWEEN 1 AND 300),
    labels_at timestamptz NOT NULL DEFAULT now(),
    predictions_at timestamptz,
    PRIMARY KEY(feature_version,date),
    FOREIGN KEY(date,feature_version) REFERENCES liquidity.feature_publications(date,feature_version)
);
CREATE TABLE liquidity.intelligence_runs (
    run_id uuid PRIMARY KEY,
    feature_version uuid NOT NULL REFERENCES liquidity.intelligence_versions(feature_version),
    stage text NOT NULL CHECK (stage IN ('labels','inference','daily')),
    through_date date NOT NULL,
    status text NOT NULL CHECK (status IN ('RUNNING','SUCCESS','FAILED')),
    started_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    finished_at timestamptz,
    rows_processed integer NOT NULL DEFAULT 0 CHECK (rows_processed>=0),
    details jsonb NOT NULL DEFAULT '{}',
    CHECK ((status='RUNNING' AND finished_at IS NULL) OR (status<>'RUNNING' AND finished_at IS NOT NULL))
);
CREATE TABLE liquidity.market_events (
    event_id text PRIMARY KEY,
    event_date date NOT NULL,
    name text NOT NULL,
    category text NOT NULL,
    description text NOT NULL,
    source_name text NOT NULL,
    source_url text NOT NULL CHECK (source_url ~ '^https?://'),
    verification_status text NOT NULL CHECK (verification_status='VERIFIED'),
    source_sha256 text NOT NULL CHECK (source_sha256 ~ '^[a-f0-9]{64}$'),
    metadata jsonb NOT NULL
);
CREATE INDEX market_events_date ON liquidity.market_events(event_date DESC);
CREATE FUNCTION liquidity.guard_intelligence_observation() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog AS $$
DECLARE threshold float8; final_day date;
BEGIN
    IF TG_OP='DELETE' THEN
        RAISE EXCEPTION 'Intelligence history cannot be deleted' USING ERRCODE='23514';
    END IF;
    IF TG_OP='UPDATE' AND (NEW.feature_version<>OLD.feature_version OR NEW.ticker<>OLD.ticker OR NEW.date<>OLD.date
       OR NEW.lrs IS DISTINCT FROM OLD.lrs
       OR (OLD.hre_probability IS NOT NULL AND NEW.hre_probability IS DISTINCT FROM OLD.hre_probability)
       OR (OLD.isolation_score IS NOT NULL AND NEW.isolation_score IS DISTINCT FROM OLD.isolation_score)
       OR (OLD.hre_label IS NOT NULL AND (NEW.hre_label IS DISTINCT FROM OLD.hre_label OR NEW.forward_lrs IS DISTINCT FROM OLD.forward_lrs OR NEW.horizon_end IS DISTINCT FROM OLD.horizon_end))) THEN
        RAISE EXCEPTION 'Committed intelligence evidence is immutable' USING ERRCODE='23514';
    END IF;
    IF NEW.horizon_end IS NOT NULL THEN
        SELECT date INTO final_day FROM liquidity.feature_values
          WHERE ticker=NEW.ticker AND feature_version=NEW.feature_version AND date>NEW.date ORDER BY date OFFSET 4 LIMIT 1;
        IF final_day IS DISTINCT FROM NEW.horizon_end THEN
            RAISE EXCEPTION 'HRE horizon requires the exact next five observations' USING ERRCODE='23514';
        END IF;
    END IF;
    IF NEW.hre_label IS NOT NULL THEN
        SELECT CASE cap_segment WHEN 'large' THEN 73.8298 WHEN 'mid' THEN 72.3251 ELSE 72.4359 END INTO threshold
          FROM liquidity.stocks WHERE ticker=NEW.ticker;
        IF NEW.hre_label<>(NEW.forward_lrs>=threshold)::int THEN
            RAISE EXCEPTION 'HRE label differs from frozen threshold' USING ERRCODE='23514';
        END IF;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER intelligence_observation_guard BEFORE INSERT OR UPDATE OR DELETE ON liquidity.intelligence_observations
FOR EACH ROW EXECUTE FUNCTION liquidity.guard_intelligence_observation();
CREATE FUNCTION liquidity.guard_intelligence_publication() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog AS $$
DECLARE available bigint; persisted bigint;
BEGIN
    SELECT count(*) INTO available FROM liquidity.feature_values WHERE date=NEW.date AND feature_version=NEW.feature_version;
    SELECT count(*) INTO persisted FROM liquidity.intelligence_observations WHERE date=NEW.date AND feature_version=NEW.feature_version;
    IF available<>persisted OR persisted<>NEW.rows THEN
        RAISE EXCEPTION 'Partial intelligence session cannot publish' USING ERRCODE='23514';
    END IF;
    IF NEW.predictions_at IS NOT NULL AND EXISTS (
        SELECT 1 FROM liquidity.feature_values f JOIN liquidity.intelligence_observations i USING(ticker,date,feature_version)
        WHERE f.date=NEW.date AND f.feature_version=NEW.feature_version AND NOT f.is_warmup AND f.is_valid_ohlcv
          AND (i.hre_probability IS NULL OR i.isolation_score IS NULL)) THEN
        RAISE EXCEPTION 'Eligible predictions are incomplete' USING ERRCODE='23514';
    END IF;
    RETURN NULL;
END $$;
CREATE CONSTRAINT TRIGGER intelligence_complete AFTER INSERT OR UPDATE ON liquidity.intelligence_publications
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION liquidity.guard_intelligence_publication();
CREATE VIEW liquidity.risk_intelligence AS
SELECT i.*,s.company_name,s.cap_segment,s.sector,
    CASE WHEN i.lrs IS NULL THEN NULL WHEN i.lrs<25 THEN 'LOW' WHEN i.lrs<50 THEN 'MODERATE' WHEN i.lrs<75 THEN 'HIGH' ELSE 'VERY_HIGH' END AS lrs_band,
    i.lrs>=CASE s.cap_segment WHEN 'large' THEN 73.8298 WHEN 'mid' THEN 72.3251 ELSE 72.4359 END AS lrs_90p_exceeded,
    CASE WHEN i.hre_probability IS NULL THEN NULL ELSE (i.hre_probability>=0.534)::int END AS hre_flag,
    CASE WHEN i.isolation_score IS NULL THEN NULL ELSE (i.isolation_score < -0.5366245497779618) END AS is_anomaly,
    CASE WHEN i.isolation_score IS NULL THEN NULL ELSE greatest(0,least(100,100*(-0.34533976879598105-i.isolation_score)/(-0.34533976879598105-(-0.7758002614394625)))) END AS anomaly_score_scaled,
    CASE WHEN f.is_warmup THEN 'WARMUP' WHEN NOT f.is_valid_ohlcv THEN 'INVALID_INPUT' WHEN p.predictions_at IS NULL THEN 'PENDING_INFERENCE' WHEN i.hre_probability IS NULL THEN 'UNAVAILABLE_INPUTS' ELSE 'AVAILABLE' END AS prediction_status,
    CASE WHEN f.is_warmup THEN 'WARMUP' WHEN i.horizon_end IS NULL THEN 'PENDING_HORIZON' WHEN i.hre_label IS NULL THEN 'UNAVAILABLE_INPUTS' ELSE 'MATURED' END AS label_status,
    CASE WHEN i.date<'2025-10-20' THEN 'TRAINING_PERIOD_REPLAY' ELSE 'OUT_OF_SAMPLE' END AS model_period,
    p.predictions_at
FROM liquidity.intelligence_observations i JOIN liquidity.stocks s USING(ticker)
JOIN liquidity.feature_values f USING(ticker,date,feature_version)
JOIN liquidity.intelligence_publications p USING(feature_version,date);
