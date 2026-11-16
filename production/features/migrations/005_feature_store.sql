-- Phase 3 forward migration. Separate production deployment approval is required.
-- Apply with the existing identity-checked migration runner and advisory lock.
CREATE TABLE liquidity.feature_versions (
    feature_version uuid PRIMARY KEY,
    calculation_sha256 text NOT NULL UNIQUE CHECK (calculation_sha256 ~ '^[a-f0-9]{64}$'),
    implementation_sha256 text NOT NULL CHECK (implementation_sha256 ~ '^[a-f0-9]{64}$'),
    source_version text NOT NULL,
    feature_names text[] NOT NULL CHECK (feature_names = ARRAY['daily_return','log_return','return_5d','return_20d','return_60d','return_z5','return_z20','abs_return','price_rel_ma20','price_rel_ma60','vol_5d','vol_20d','vol_60d','vol_ratio_5_20','parkinson_vol','true_range','atr_20d','downside_vol_20d','sortino_component','max_drawdown_20d','rsi_14','vol_of_vol','log_volume','volume_ma20','volume_ratio','volume_z20','amihud_ratio','amihud_ma20','amihud_z20','turnover_value','log_turnover','turnover_ma20','turnover_ratio','zero_return_flag','zero_vol_flag','low_volume_streak','daily_range','daily_range_ma20','range_z20','overnight_gap','overnight_gap_abs','gap_z20','cs_spread_est','cs_spread_ma10','intraday_return','excess_return_1d','excess_return_20d','rolling_beta_60d','rolling_corr_60d','idiosyncratic_vol','relative_volume','nifty_vol_20d','nifty_vol_ratio','nifty_return_20d','vol_beta','vix_close','vix_return_1d','vix_z20']::text[]),
    contract jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE liquidity.feature_runs (
    run_id uuid PRIMARY KEY,
    request_key text NOT NULL UNIQUE,
    feature_version uuid NOT NULL REFERENCES liquidity.feature_versions(feature_version),
    range_start date NOT NULL,
    range_end date NOT NULL CHECK (range_end>=range_start),
    expected_tickers text[] NOT NULL CHECK (cardinality(expected_tickers) BETWEEN 1 AND 300 AND array_position(expected_tickers,NULL) IS NULL),
    expected_rows integer NOT NULL CHECK (expected_rows>0),
    input_sha256 text NOT NULL CHECK (input_sha256 ~ '^[a-f0-9]{64}$'),
    output_sha256 text NOT NULL CHECK (output_sha256 ~ '^[a-f0-9]{64}$'),
    provenance jsonb NOT NULL,
    status text NOT NULL CHECK (status IN ('RUNNING','SUCCESS')),
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    CHECK ((status='RUNNING' AND finished_at IS NULL) OR (status='SUCCESS' AND finished_at IS NOT NULL)),
    UNIQUE(run_id,feature_version)
);
CREATE TABLE liquidity.feature_values (
    ticker text NOT NULL REFERENCES liquidity.stocks(ticker),
    date date NOT NULL CHECK (date<>DATE '2026-11-08'),
    feature_version uuid NOT NULL REFERENCES liquidity.feature_versions(feature_version),
    run_id uuid NOT NULL,
    is_zero_volume boolean NOT NULL,
    is_stale_price boolean NOT NULL,
    is_corporate_action_gap boolean NOT NULL,
    is_market_shock boolean NOT NULL,
    is_short_history boolean NOT NULL,
    is_warmup boolean NOT NULL,
    is_benchmark_available boolean NOT NULL,
    is_muhurat_session boolean NOT NULL,
    is_valid_ohlcv boolean NOT NULL,
    daily_return real CHECK (daily_return IS NULL OR (daily_return > '-Infinity'::real AND daily_return < 'Infinity'::real)),
    log_return real CHECK (log_return IS NULL OR (log_return > '-Infinity'::real AND log_return < 'Infinity'::real)),
    return_5d real CHECK (return_5d IS NULL OR (return_5d > '-Infinity'::real AND return_5d < 'Infinity'::real)),
    return_20d real CHECK (return_20d IS NULL OR (return_20d > '-Infinity'::real AND return_20d < 'Infinity'::real)),
    return_60d real CHECK (return_60d IS NULL OR (return_60d > '-Infinity'::real AND return_60d < 'Infinity'::real)),
    return_z5 real CHECK (return_z5 IS NULL OR (return_z5 > '-Infinity'::real AND return_z5 < 'Infinity'::real)),
    return_z20 real CHECK (return_z20 IS NULL OR (return_z20 > '-Infinity'::real AND return_z20 < 'Infinity'::real)),
    abs_return real CHECK (abs_return IS NULL OR (abs_return > '-Infinity'::real AND abs_return < 'Infinity'::real)),
    price_rel_ma20 real CHECK (price_rel_ma20 IS NULL OR (price_rel_ma20 > '-Infinity'::real AND price_rel_ma20 < 'Infinity'::real)),
    price_rel_ma60 real CHECK (price_rel_ma60 IS NULL OR (price_rel_ma60 > '-Infinity'::real AND price_rel_ma60 < 'Infinity'::real)),
    vol_5d real CHECK (vol_5d IS NULL OR (vol_5d > '-Infinity'::real AND vol_5d < 'Infinity'::real)),
    vol_20d real CHECK (vol_20d IS NULL OR (vol_20d > '-Infinity'::real AND vol_20d < 'Infinity'::real)),
    vol_60d real CHECK (vol_60d IS NULL OR (vol_60d > '-Infinity'::real AND vol_60d < 'Infinity'::real)),
    vol_ratio_5_20 real CHECK (vol_ratio_5_20 IS NULL OR (vol_ratio_5_20 > '-Infinity'::real AND vol_ratio_5_20 < 'Infinity'::real)),
    parkinson_vol real CHECK (parkinson_vol IS NULL OR (parkinson_vol > '-Infinity'::real AND parkinson_vol < 'Infinity'::real)),
    true_range real CHECK (true_range IS NULL OR (true_range > '-Infinity'::real AND true_range < 'Infinity'::real)),
    atr_20d real CHECK (atr_20d IS NULL OR (atr_20d > '-Infinity'::real AND atr_20d < 'Infinity'::real)),
    downside_vol_20d real CHECK (downside_vol_20d IS NULL OR (downside_vol_20d > '-Infinity'::real AND downside_vol_20d < 'Infinity'::real)),
    sortino_component real CHECK (sortino_component IS NULL OR (sortino_component > '-Infinity'::real AND sortino_component < 'Infinity'::real)),
    max_drawdown_20d real CHECK (max_drawdown_20d IS NULL OR (max_drawdown_20d > '-Infinity'::real AND max_drawdown_20d < 'Infinity'::real)),
    rsi_14 real CHECK (rsi_14 IS NULL OR (rsi_14 > '-Infinity'::real AND rsi_14 < 'Infinity'::real)),
    vol_of_vol real CHECK (vol_of_vol IS NULL OR (vol_of_vol > '-Infinity'::real AND vol_of_vol < 'Infinity'::real)),
    log_volume real CHECK (log_volume IS NULL OR (log_volume > '-Infinity'::real AND log_volume < 'Infinity'::real)),
    volume_ma20 real CHECK (volume_ma20 IS NULL OR (volume_ma20 > '-Infinity'::real AND volume_ma20 < 'Infinity'::real)),
    volume_ratio real CHECK (volume_ratio IS NULL OR (volume_ratio > '-Infinity'::real AND volume_ratio < 'Infinity'::real)),
    volume_z20 real CHECK (volume_z20 IS NULL OR (volume_z20 > '-Infinity'::real AND volume_z20 < 'Infinity'::real)),
    amihud_ratio real CHECK (amihud_ratio IS NULL OR (amihud_ratio > '-Infinity'::real AND amihud_ratio < 'Infinity'::real)),
    amihud_ma20 real CHECK (amihud_ma20 IS NULL OR (amihud_ma20 > '-Infinity'::real AND amihud_ma20 < 'Infinity'::real)),
    amihud_z20 real CHECK (amihud_z20 IS NULL OR (amihud_z20 > '-Infinity'::real AND amihud_z20 < 'Infinity'::real)),
    turnover_value real CHECK (turnover_value IS NULL OR (turnover_value > '-Infinity'::real AND turnover_value < 'Infinity'::real)),
    log_turnover real CHECK (log_turnover IS NULL OR (log_turnover > '-Infinity'::real AND log_turnover < 'Infinity'::real)),
    turnover_ma20 real CHECK (turnover_ma20 IS NULL OR (turnover_ma20 > '-Infinity'::real AND turnover_ma20 < 'Infinity'::real)),
    turnover_ratio real CHECK (turnover_ratio IS NULL OR (turnover_ratio > '-Infinity'::real AND turnover_ratio < 'Infinity'::real)),
    zero_return_flag real CHECK (zero_return_flag IS NULL OR (zero_return_flag > '-Infinity'::real AND zero_return_flag < 'Infinity'::real)),
    zero_vol_flag real CHECK (zero_vol_flag IS NULL OR (zero_vol_flag > '-Infinity'::real AND zero_vol_flag < 'Infinity'::real)),
    low_volume_streak real CHECK (low_volume_streak IS NULL OR (low_volume_streak > '-Infinity'::real AND low_volume_streak < 'Infinity'::real)),
    daily_range real CHECK (daily_range IS NULL OR (daily_range > '-Infinity'::real AND daily_range < 'Infinity'::real)),
    daily_range_ma20 real CHECK (daily_range_ma20 IS NULL OR (daily_range_ma20 > '-Infinity'::real AND daily_range_ma20 < 'Infinity'::real)),
    range_z20 real CHECK (range_z20 IS NULL OR (range_z20 > '-Infinity'::real AND range_z20 < 'Infinity'::real)),
    overnight_gap real CHECK (overnight_gap IS NULL OR (overnight_gap > '-Infinity'::real AND overnight_gap < 'Infinity'::real)),
    overnight_gap_abs real CHECK (overnight_gap_abs IS NULL OR (overnight_gap_abs > '-Infinity'::real AND overnight_gap_abs < 'Infinity'::real)),
    gap_z20 real CHECK (gap_z20 IS NULL OR (gap_z20 > '-Infinity'::real AND gap_z20 < 'Infinity'::real)),
    cs_spread_est real CHECK (cs_spread_est IS NULL OR (cs_spread_est > '-Infinity'::real AND cs_spread_est < 'Infinity'::real)),
    cs_spread_ma10 real CHECK (cs_spread_ma10 IS NULL OR (cs_spread_ma10 > '-Infinity'::real AND cs_spread_ma10 < 'Infinity'::real)),
    intraday_return real CHECK (intraday_return IS NULL OR (intraday_return > '-Infinity'::real AND intraday_return < 'Infinity'::real)),
    excess_return_1d real CHECK (excess_return_1d IS NULL OR (excess_return_1d > '-Infinity'::real AND excess_return_1d < 'Infinity'::real)),
    excess_return_20d real CHECK (excess_return_20d IS NULL OR (excess_return_20d > '-Infinity'::real AND excess_return_20d < 'Infinity'::real)),
    rolling_beta_60d real CHECK (rolling_beta_60d IS NULL OR (rolling_beta_60d > '-Infinity'::real AND rolling_beta_60d < 'Infinity'::real)),
    rolling_corr_60d real CHECK (rolling_corr_60d IS NULL OR (rolling_corr_60d > '-Infinity'::real AND rolling_corr_60d < 'Infinity'::real)),
    idiosyncratic_vol real CHECK (idiosyncratic_vol IS NULL OR (idiosyncratic_vol > '-Infinity'::real AND idiosyncratic_vol < 'Infinity'::real)),
    relative_volume real CHECK (relative_volume IS NULL OR (relative_volume > '-Infinity'::real AND relative_volume < 'Infinity'::real)),
    nifty_vol_20d real CHECK (nifty_vol_20d IS NULL OR (nifty_vol_20d > '-Infinity'::real AND nifty_vol_20d < 'Infinity'::real)),
    nifty_vol_ratio real CHECK (nifty_vol_ratio IS NULL OR (nifty_vol_ratio > '-Infinity'::real AND nifty_vol_ratio < 'Infinity'::real)),
    nifty_return_20d real CHECK (nifty_return_20d IS NULL OR (nifty_return_20d > '-Infinity'::real AND nifty_return_20d < 'Infinity'::real)),
    vol_beta real CHECK (vol_beta IS NULL OR (vol_beta > '-Infinity'::real AND vol_beta < 'Infinity'::real)),
    vix_close real CHECK (vix_close IS NULL OR (vix_close > '-Infinity'::real AND vix_close < 'Infinity'::real)),
    vix_return_1d real CHECK (vix_return_1d IS NULL OR (vix_return_1d > '-Infinity'::real AND vix_return_1d < 'Infinity'::real)),
    vix_z20 real CHECK (vix_z20 IS NULL OR (vix_z20 > '-Infinity'::real AND vix_z20 < 'Infinity'::real)),
    PRIMARY KEY(ticker,date,feature_version),
    FOREIGN KEY(ticker,date) REFERENCES liquidity.market_eod(ticker,date),
    FOREIGN KEY(run_id,feature_version) REFERENCES liquidity.feature_runs(run_id,feature_version)
);
CREATE TABLE liquidity.feature_checkpoints (
    ticker text NOT NULL,
    feature_version uuid NOT NULL,
    last_date date NOT NULL,
    state jsonb NOT NULL,
    PRIMARY KEY(ticker,feature_version),
    FOREIGN KEY(ticker,last_date,feature_version) REFERENCES liquidity.feature_values(ticker,date,feature_version)
);
CREATE TABLE liquidity.feature_publications (
    date date NOT NULL CHECK (date<>DATE '2026-11-08'),
    feature_version uuid NOT NULL REFERENCES liquidity.feature_versions(feature_version),
    published_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(date,feature_version)
);
CREATE INDEX feature_values_date_version ON liquidity.feature_values(date,feature_version,ticker);
CREATE INDEX feature_values_run ON liquidity.feature_values(run_id);

CREATE FUNCTION liquidity.feature_immutable() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog AS $$
BEGIN
    RAISE EXCEPTION 'Published feature evidence is immutable' USING ERRCODE='23514';
END $$;
CREATE TRIGGER feature_version_immutable BEFORE UPDATE OR DELETE ON liquidity.feature_versions
FOR EACH ROW EXECUTE FUNCTION liquidity.feature_immutable();
CREATE TRIGGER feature_value_immutable BEFORE UPDATE OR DELETE ON liquidity.feature_values
FOR EACH ROW EXECUTE FUNCTION liquidity.feature_immutable();
CREATE TRIGGER feature_publication_immutable BEFORE UPDATE OR DELETE ON liquidity.feature_publications
FOR EACH ROW EXECUTE FUNCTION liquidity.feature_immutable();

CREATE FUNCTION liquidity.feature_mutation_guard() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog AS $$
DECLARE run liquidity.feature_runs%ROWTYPE; target uuid;
BEGIN
    IF TG_TABLE_NAME='feature_runs' THEN
        IF TG_OP='UPDATE' THEN
            IF OLD.status='SUCCESS' OR (to_jsonb(NEW)-'status'-'finished_at')<>(to_jsonb(OLD)-'status'-'finished_at') THEN
                RAISE EXCEPTION 'Feature run identity is immutable' USING ERRCODE='23514';
            END IF;
        END IF;
        target:=NEW.run_id;
    ELSE
        SELECT * INTO run FROM liquidity.feature_runs WHERE run_id=NEW.run_id;
        IF run.status<>'RUNNING' OR NEW.date<run.range_start OR NEW.date>run.range_end OR NOT NEW.ticker=ANY(run.expected_tickers) THEN
            RAISE EXCEPTION 'Feature observation differs from its running plan' USING ERRCODE='23514';
        END IF;
        target:=NEW.run_id;
    END IF;
    PERFORM set_config('liquidity.feature_checked_'||replace(target::text,'-',''),'0',true);
    RETURN NEW;
END $$;
CREATE TRIGGER feature_run_mutation BEFORE INSERT OR UPDATE ON liquidity.feature_runs
FOR EACH ROW EXECUTE FUNCTION liquidity.feature_mutation_guard();
CREATE TRIGGER feature_value_mutation BEFORE INSERT ON liquidity.feature_values
FOR EACH ROW EXECUTE FUNCTION liquidity.feature_mutation_guard();
CREATE TRIGGER feature_run_no_delete BEFORE DELETE ON liquidity.feature_runs
FOR EACH ROW EXECUTE FUNCTION liquidity.feature_immutable();

CREATE FUNCTION liquidity.assert_feature_run(target uuid) RETURNS void
LANGUAGE plpgsql SET search_path=pg_catalog AS $$
DECLARE run liquidity.feature_runs%ROWTYPE; expected bigint; actual bigint; key text;
BEGIN
    key:='liquidity.feature_checked_'||replace(target::text,'-','');
    IF current_setting(key,true)='1' THEN RETURN; END IF;
    SELECT * INTO run FROM liquidity.feature_runs WHERE run_id=target;
    IF run.status<>'SUCCESS' THEN
        RAISE EXCEPTION 'An incomplete feature run cannot commit' USING ERRCODE='23514';
    END IF;
    IF cardinality(run.expected_tickers)<>(SELECT count(DISTINCT t) FROM unnest(run.expected_tickers) t)
       OR EXISTS(SELECT 1 FROM unnest(run.expected_tickers) t LEFT JOIN liquidity.stocks s ON s.ticker=t WHERE s.ticker IS NULL) THEN
        RAISE EXCEPTION 'Invalid feature instrument plan' USING ERRCODE='23514';
    END IF;
    SELECT count(*) INTO expected FROM liquidity.market_eod m
      WHERE m.ticker=ANY(run.expected_tickers) AND m.date BETWEEN run.range_start AND run.range_end AND m.date<>DATE '2026-11-08';
    SELECT count(*) INTO actual FROM liquidity.feature_values WHERE run_id=target;
    IF expected<>run.expected_rows OR actual<>expected THEN
        RAISE EXCEPTION 'Feature coverage differs from observed source coverage' USING ERRCODE='23514';
    END IF;
    PERFORM set_config(key,'1',true);
END $$;
CREATE FUNCTION liquidity.feature_completion_guard() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog AS $$
BEGIN
    PERFORM liquidity.assert_feature_run(NEW.run_id);
    RETURN NULL;
END $$;
CREATE CONSTRAINT TRIGGER feature_run_completion AFTER INSERT OR UPDATE ON liquidity.feature_runs
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION liquidity.feature_completion_guard();
CREATE CONSTRAINT TRIGGER feature_values_completion AFTER INSERT ON liquidity.feature_values
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION liquidity.feature_completion_guard();

CREATE FUNCTION liquidity.feature_checkpoint_guard() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog AS $$
DECLARE n bigint; available bigint; source text; checkpoint_close double precision;
BEGIN
    IF TG_OP='DELETE' THEN RAISE EXCEPTION 'Feature checkpoint deletion is forbidden' USING ERRCODE='23514'; END IF;
    IF TG_OP='UPDATE' AND (NEW.ticker<>OLD.ticker OR NEW.feature_version<>OLD.feature_version OR NEW.last_date<=OLD.last_date) THEN
        RAISE EXCEPTION 'Feature checkpoint must advance within its version' USING ERRCODE='23514';
    END IF;
    SELECT count(*) INTO n FROM liquidity.feature_values WHERE ticker=NEW.ticker AND feature_version=NEW.feature_version AND date<=NEW.last_date;
    SELECT count(*) INTO available FROM liquidity.market_eod WHERE ticker=NEW.ticker AND date<=NEW.last_date AND date<>DATE '2026-11-08';
    SELECT source_version INTO source FROM liquidity.feature_versions WHERE feature_version=NEW.feature_version;
    SELECT close INTO checkpoint_close FROM liquidity.market_eod_consistent WHERE ticker=NEW.ticker AND date=NEW.last_date;
    IF jsonb_typeof(NEW.state->'context') IS DISTINCT FROM 'array' OR jsonb_array_length(NEW.state->'context')<>least(100,n)
       OR NOT COALESCE(NEW.state->'context'->-1->>'date'=NEW.last_date::text,false)
       OR NOT COALESCE((NEW.state->>'previous_close')::double precision=checkpoint_close,false) THEN
        RAISE EXCEPTION 'Feature recursive state close or context differs' USING ERRCODE='23514';
    END IF;
    IF jsonb_typeof(NEW.state->'amihud_history') IS DISTINCT FROM 'array'
       OR jsonb_array_length(NEW.state->'amihud_history')<>n THEN
        RAISE EXCEPTION 'Feature rolling-variance history differs from coverage' USING ERRCODE='23514';
    END IF;
    IF n<>available OR NOT COALESCE((NEW.state->>'count')::bigint=n
       AND NEW.state->>'ticker'=NEW.ticker AND NEW.state->>'last_date'=NEW.last_date::text
       AND NEW.state->>'source_version'=source,false) THEN
        RAISE EXCEPTION 'Feature checkpoint coverage or recursive state differs' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER feature_checkpoint_consistency BEFORE INSERT OR UPDATE OR DELETE ON liquidity.feature_checkpoints
FOR EACH ROW EXECUTE FUNCTION liquidity.feature_checkpoint_guard();

CREATE FUNCTION liquidity.feature_publication_guard() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog AS $$
BEGIN
    IF NOT EXISTS(SELECT 1 FROM liquidity.market_eod m JOIN liquidity.stocks s USING(ticker) WHERE m.date=NEW.date)
       OR EXISTS(SELECT 1 FROM liquidity.market_eod m JOIN liquidity.stocks s USING(ticker)
          LEFT JOIN liquidity.feature_values f ON f.ticker=m.ticker AND f.date=m.date AND f.feature_version=NEW.feature_version
          WHERE m.date=NEW.date AND f.ticker IS NULL) THEN
        RAISE EXCEPTION 'Incomplete feature session cannot be published' USING ERRCODE='23514';
    END IF;
    RETURN NULL;
END $$;
CREATE CONSTRAINT TRIGGER feature_publication_complete AFTER INSERT ON liquidity.feature_publications
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION liquidity.feature_publication_guard();

CREATE VIEW liquidity.feature_model_inputs AS
SELECT f.ticker,f.date,f.feature_version,f.daily_return,f.return_5d,f.return_20d,f.return_z20,f.abs_return,f.price_rel_ma20,f.vol_5d,f.vol_20d,f.vol_ratio_5_20,f.downside_vol_20d,f.max_drawdown_20d,f.rsi_14,f.vol_of_vol,f.log_volume,f.volume_z20,f.amihud_ratio,f.amihud_ma20,f.amihud_z20,f.log_turnover,f.turnover_ratio,f.low_volume_streak,f.daily_range,f.daily_range_ma20,f.range_z20,f.overnight_gap,f.overnight_gap_abs,f.cs_spread_est,f.cs_spread_ma10,f.excess_return_1d,f.rolling_beta_60d,f.rolling_corr_60d,f.relative_volume,f.nifty_vol_20d,f.nifty_return_20d,f.vol_beta,f.vix_close,f.volume_ratio
FROM liquidity.feature_values f JOIN liquidity.feature_publications p USING(date,feature_version);
