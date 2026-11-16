-- Forward-only hardening. Never edit 000/001 or their recorded checksums.
-- Cross-table invariants are deferred to permit atomic plan/output assembly.
CREATE FUNCTION liquidity.valid_instruments(items text[]) RETURNS boolean
LANGUAGE sql IMMUTABLE SET search_path = pg_catalog AS $$
    SELECT coalesce(array_ndims(items) = 1 AND array_lower(items, 1) = 1
        AND cardinality(items) > 0 AND NOT EXISTS (
            SELECT 1 FROM unnest(items) AS t(item) WHERE item IS NULL OR btrim(item) = '')
        AND cardinality(items) = (SELECT count(DISTINCT item) FROM unnest(items) AS t(item)), false)
$$;
CREATE FUNCTION liquidity.valid_sessions(items date[]) RETURNS boolean
LANGUAGE sql IMMUTABLE SET search_path = pg_catalog AS $$
    SELECT coalesce(array_ndims(items) = 1 AND array_lower(items, 1) = 1
        AND cardinality(items) > 0 AND NOT EXISTS (
            SELECT 1 FROM (
                SELECT item, lag(item) OVER (ORDER BY ordinal) AS previous
                FROM unnest(items) WITH ORDINALITY AS t(item, ordinal)
            ) s WHERE item IS NULL OR NOT isfinite(item) OR item <= previous), false)
$$;
ALTER TABLE liquidity.ingestion_runs ADD CONSTRAINT ingestion_expected_valid
    CHECK (liquidity.valid_instruments(expected_instruments));
ALTER TABLE liquidity.ingestion_run_instruments ADD CONSTRAINT ingestion_sessions_valid
    CHECK (liquidity.valid_sessions(planned_sessions));

-- A stock dimension row can only refer to a stock instrument, in either write direction.
ALTER TABLE liquidity.instruments ADD CONSTRAINT instruments_ticker_type_key UNIQUE (ticker, instrument_type);
ALTER TABLE liquidity.stocks ADD COLUMN instrument_type text NOT NULL DEFAULT 'stock'
    CHECK (instrument_type = 'stock');
ALTER TABLE liquidity.stocks ADD CONSTRAINT stocks_instrument_type_fk
    FOREIGN KEY (ticker, instrument_type) REFERENCES liquidity.instruments(ticker, instrument_type);

ALTER TABLE liquidity.ingestion_batches ADD CONSTRAINT batches_endpoint_key UNIQUE (batch_id, ticker, range_end);
ALTER TABLE liquidity.instrument_checkpoints ADD CONSTRAINT checkpoints_batch_endpoint_fk
    FOREIGN KEY (last_batch_id, ticker, last_date)
    REFERENCES liquidity.ingestion_batches(batch_id, ticker, range_end);
CREATE FUNCTION liquidity.guard_checkpoint() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Checkpoint deletion requires an explicitly reviewed maintenance operation' USING ERRCODE = '23514';
    END IF;
    IF TG_OP = 'INSERT' THEN
        IF NEW.revision <> (CASE WHEN NEW.last_date IS NULL THEN 0 ELSE 1 END) THEN
            RAISE EXCEPTION 'Invalid initial checkpoint revision' USING ERRCODE = '23514';
        END IF;
    ELSE
        IF NEW.ticker IS DISTINCT FROM OLD.ticker
            OR (OLD.last_date IS NOT NULL AND (NEW.last_date IS NULL OR NEW.last_date < OLD.last_date)) THEN
            RAISE EXCEPTION 'Checkpoint cannot change identity or move backwards' USING ERRCODE = '23514';
        END IF;
        IF ROW(NEW.last_date, NEW.last_batch_id) IS DISTINCT FROM ROW(OLD.last_date, OLD.last_batch_id) THEN
            IF NEW.last_date IS NOT DISTINCT FROM OLD.last_date OR NEW.revision <> OLD.revision + 1 THEN
                RAISE EXCEPTION 'Checkpoint advancement requires a newer date and next revision' USING ERRCODE = '23514';
            END IF;
        ELSIF NEW.revision <> OLD.revision THEN
            RAISE EXCEPTION 'Checkpoint revision requires advancement' USING ERRCODE = '23514';
        END IF;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER checkpoint_guard BEFORE INSERT OR UPDATE OR DELETE ON liquidity.instrument_checkpoints
    FOR EACH ROW EXECUTE FUNCTION liquidity.guard_checkpoint();

CREATE INDEX ingestion_plan_instrument ON liquidity.ingestion_run_instruments(ticker, run_id);
CREATE FUNCTION liquidity.assert_ingestion(target uuid) RETURNS void
LANGUAGE plpgsql SET search_path = pg_catalog AS $$
DECLARE r liquidity.ingestion_runs%ROWTYPE;
BEGIN
    SELECT * INTO r FROM liquidity.ingestion_runs WHERE run_id = target FOR UPDATE;
    IF NOT FOUND THEN RETURN; END IF;
    IF (SELECT count(*) FROM liquidity.ingestion_run_instruments WHERE run_id = target)
            <> cardinality(r.expected_instruments)
        OR EXISTS (SELECT 1 FROM liquidity.ingestion_run_instruments p WHERE p.run_id = target
            AND (NOT p.ticker = ANY(r.expected_instruments)
                OR p.planned_sessions[1] < r.range_start
                OR p.planned_sessions[cardinality(p.planned_sessions)] > r.range_end)) THEN
        RAISE EXCEPTION 'Ingestion plan membership, count or date range is inconsistent' USING ERRCODE = '23514';
    END IF;
    IF r.status = 'SUCCESS' AND EXISTS (
        SELECT 1 FROM liquidity.ingestion_run_instruments p WHERE p.run_id = target AND (
            EXISTS (SELECT 1 FROM unnest(p.planned_sessions) AS s(day)
                WHERE NOT EXISTS (SELECT 1 FROM liquidity.market_eod m WHERE m.ticker = p.ticker AND m.date = s.day))
            OR NOT EXISTS (SELECT 1 FROM liquidity.ingestion_batches b WHERE b.run_id = target AND b.ticker = p.ticker
                AND b.range_end = p.planned_sessions[cardinality(p.planned_sessions)]))) THEN
        RAISE EXCEPTION 'Successful ingestion requires every planned observation and final batch receipt' USING ERRCODE = '23514';
    END IF;
END $$;
CREATE FUNCTION liquidity.check_ingestion() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
    IF TG_OP <> 'INSERT' THEN PERFORM liquidity.assert_ingestion(OLD.run_id); END IF;
    IF TG_OP <> 'DELETE' THEN PERFORM liquidity.assert_ingestion(NEW.run_id); END IF;
    RETURN NULL;
END $$;
CREATE CONSTRAINT TRIGGER ingestion_run_integrity AFTER INSERT OR UPDATE OR DELETE ON liquidity.ingestion_runs
    DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION liquidity.check_ingestion();
CREATE CONSTRAINT TRIGGER ingestion_plan_integrity AFTER INSERT OR UPDATE OR DELETE ON liquidity.ingestion_run_instruments
    DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION liquidity.check_ingestion();
CREATE CONSTRAINT TRIGGER ingestion_batch_integrity AFTER INSERT OR UPDATE OR DELETE ON liquidity.ingestion_batches
    DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION liquidity.check_ingestion();

-- Protect an already-successful run when an observation is removed or re-keyed.
-- Lock its parent before removing coverage, serializing against completion.
CREATE FUNCTION liquidity.guard_observation_removal() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog AS $$
DECLARE target uuid;
BEGIN
    IF TG_OP = 'UPDATE' AND ROW(NEW.ticker, NEW.date) IS NOT DISTINCT FROM ROW(OLD.ticker, OLD.date) THEN
        RETURN NEW;
    END IF;
    FOR target IN SELECT run_id FROM liquidity.ingestion_run_instruments
        WHERE ticker = OLD.ticker AND OLD.date = ANY(planned_sessions) ORDER BY run_id LOOP
        PERFORM 1 FROM liquidity.ingestion_runs WHERE run_id = target FOR UPDATE;
        IF EXISTS (SELECT 1 FROM liquidity.ingestion_runs WHERE run_id = target AND status = 'SUCCESS') THEN
            RAISE EXCEPTION 'Cannot remove coverage from successful ingestion' USING ERRCODE = '23514';
        END IF;
    END LOOP;
    IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER observation_coverage_guard BEFORE DELETE OR UPDATE ON liquidity.market_eod
    FOR EACH ROW EXECUTE FUNCTION liquidity.guard_observation_removal();

-- Plans are immutable after creation; a changed plan must use a new request key.
CREATE FUNCTION liquidity.guard_plan_change() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
    IF TG_TABLE_NAME = 'ingestion_runs' THEN
        IF ROW(NEW.run_id, NEW.request_key, NEW.plan_sha256, NEW.mode, NEW.range_start, NEW.range_end, NEW.expected_instruments)
            IS DISTINCT FROM ROW(OLD.run_id, OLD.request_key, OLD.plan_sha256, OLD.mode, OLD.range_start, OLD.range_end, OLD.expected_instruments)
            OR (OLD.status <> 'RUNNING' AND NEW.status <> OLD.status) THEN
            RAISE EXCEPTION 'Run identity, plan and terminal status are immutable' USING ERRCODE = '23514';
        END IF;
    ELSIF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Ingestion plans cannot be deleted' USING ERRCODE = '23514';
    ELSIF NEW IS DISTINCT FROM OLD THEN
        RAISE EXCEPTION 'Ingestion plans are immutable' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER ingestion_identity_guard BEFORE UPDATE ON liquidity.ingestion_runs
    FOR EACH ROW EXECUTE FUNCTION liquidity.guard_plan_change();
CREATE TRIGGER ingestion_plan_guard BEFORE UPDATE OR DELETE ON liquidity.ingestion_run_instruments
    FOR EACH ROW EXECUTE FUNCTION liquidity.guard_plan_change();

ALTER TABLE liquidity.pipeline_runs ADD COLUMN finished_at timestamptz;
ALTER TABLE liquidity.pipeline_runs ADD CONSTRAINT pipeline_completion_state CHECK (
    (status = 'RUNNING' AND finished_at IS NULL AND error_code IS NULL
        AND duration_seconds IS NULL AND stocks_processed IS NULL)
    OR (status = 'SUCCESS' AND finished_at IS NOT NULL AND error_code IS NULL
        AND duration_seconds IS NOT NULL AND stocks_processed IS NOT NULL AND stocks_processed >= 1)
    OR (status = 'FAILED' AND finished_at IS NOT NULL AND error_code IS NOT NULL));
ALTER TABLE liquidity.pipeline_runs ADD CONSTRAINT pipeline_finished_time CHECK (finished_at >= created_at);
ALTER TABLE liquidity.pipeline_runs ADD CONSTRAINT pipeline_run_date_key UNIQUE (run_id, run_date);
ALTER TABLE liquidity.daily_predictions ADD CONSTRAINT predictions_run_date_fk
    FOREIGN KEY (pipeline_run_id, date) REFERENCES liquidity.pipeline_runs(run_id, run_date);
CREATE INDEX predictions_pipeline_run ON liquidity.daily_predictions(pipeline_run_id);
CREATE INDEX pipeline_ingestion_run ON liquidity.pipeline_runs(ingestion_run_id);
CREATE FUNCTION liquidity.guard_pipeline_status() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
    IF OLD.status <> 'RUNNING' AND NEW.status <> OLD.status THEN
        RAISE EXCEPTION 'Terminal pipeline status is immutable; retries require a new run' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER pipeline_status_guard BEFORE UPDATE ON liquidity.pipeline_runs
    FOR EACH ROW EXECUTE FUNCTION liquidity.guard_pipeline_status();
CREATE FUNCTION liquidity.assert_pipeline(target uuid) RETURNS void
LANGUAGE plpgsql SET search_path = pg_catalog AS $$
DECLARE p liquidity.pipeline_runs%ROWTYPE;
BEGIN
    SELECT * INTO p FROM liquidity.pipeline_runs WHERE run_id = target FOR UPDATE;
    IF NOT FOUND OR p.status <> 'SUCCESS' THEN RETURN; END IF;
    PERFORM 1 FROM liquidity.ingestion_runs WHERE run_id = p.ingestion_run_id FOR UPDATE;
    IF NOT EXISTS (SELECT 1 FROM liquidity.ingestion_runs WHERE run_id = p.ingestion_run_id AND status = 'SUCCESS')
        OR p.stocks_processed <> (SELECT count(*) FROM liquidity.daily_predictions WHERE pipeline_run_id = target) THEN
        RAISE EXCEPTION 'Successful pipeline requires successful ingestion and matching prediction count' USING ERRCODE = '23514';
    END IF;
END $$;
CREATE FUNCTION liquidity.check_pipeline() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog AS $$
BEGIN
    IF TG_TABLE_NAME = 'pipeline_runs' THEN
        IF TG_OP <> 'INSERT' THEN PERFORM liquidity.assert_pipeline(OLD.run_id); END IF;
        IF TG_OP <> 'DELETE' THEN PERFORM liquidity.assert_pipeline(NEW.run_id); END IF;
    ELSE
        IF TG_OP <> 'INSERT' THEN PERFORM liquidity.assert_pipeline(OLD.pipeline_run_id); END IF;
        IF TG_OP <> 'DELETE' THEN PERFORM liquidity.assert_pipeline(NEW.pipeline_run_id); END IF;
    END IF;
    RETURN NULL;
END $$;
CREATE CONSTRAINT TRIGGER pipeline_integrity AFTER INSERT OR UPDATE OR DELETE ON liquidity.pipeline_runs
    DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION liquidity.check_pipeline();
CREATE CONSTRAINT TRIGGER predictions_pipeline_integrity AFTER INSERT OR UPDATE OR DELETE ON liquidity.daily_predictions
    DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION liquidity.check_pipeline();

-- Constraint triggers do not retroactively validate old rows. Abort rather than
-- invent coverage or completion metrics if an existing database is inconsistent.
DO $$
DECLARE target uuid;
BEGIN
    FOR target IN SELECT run_id FROM liquidity.ingestion_runs LOOP
        PERFORM liquidity.assert_ingestion(target);
    END LOOP;
    FOR target IN SELECT run_id FROM liquidity.pipeline_runs LOOP
        PERFORM liquidity.assert_pipeline(target);
    END LOOP;
END $$;
