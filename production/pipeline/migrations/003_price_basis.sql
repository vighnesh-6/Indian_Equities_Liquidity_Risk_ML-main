-- Original market_eod observations remain immutable. Actual provider revisions
-- are appended separately and become visible only after full-history validation.
-- Deduplicate deferred checks only while no intervening row mutation occurred.
-- Immediate dirty triggers invalidate the transaction-local validation marker.
CREATE FUNCTION liquidity.mark_ingestion_dirty() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog AS $$
BEGIN
 IF TG_OP<>'INSERT' THEN PERFORM set_config('liquidity.run_'||replace(OLD.run_id::text,'-',''),'dirty',true); END IF;
 IF TG_OP<>'DELETE' THEN PERFORM set_config('liquidity.run_'||replace(NEW.run_id::text,'-',''),'dirty',true); END IF;
 RETURN NULL;
END $$;
CREATE TRIGGER aaa_ingestion_dirty AFTER INSERT OR UPDATE OR DELETE ON liquidity.ingestion_runs FOR EACH ROW EXECUTE FUNCTION liquidity.mark_ingestion_dirty();
CREATE TRIGGER aaa_ingestion_dirty AFTER INSERT OR UPDATE OR DELETE ON liquidity.ingestion_run_instruments FOR EACH ROW EXECUTE FUNCTION liquidity.mark_ingestion_dirty();
CREATE TRIGGER aaa_ingestion_dirty AFTER INSERT OR UPDATE OR DELETE ON liquidity.ingestion_batches FOR EACH ROW EXECUTE FUNCTION liquidity.mark_ingestion_dirty();
CREATE OR REPLACE FUNCTION liquidity.check_ingestion() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog AS $$
DECLARE target uuid; marker text;
BEGIN
 FOR target IN SELECT DISTINCT id FROM unnest(ARRAY[
  CASE WHEN TG_OP<>'INSERT' THEN OLD.run_id END,
  CASE WHEN TG_OP<>'DELETE' THEN NEW.run_id END]) AS t(id) WHERE id IS NOT NULL LOOP
  marker := 'liquidity.run_'||replace(target::text,'-','');
  IF current_setting(marker,true) IS DISTINCT FROM 'clean' THEN
   PERFORM liquidity.assert_ingestion(target);
   PERFORM set_config(marker,'clean',true);
  END IF;
 END LOOP;
 RETURN NULL;
END $$;
CREATE TABLE liquidity.price_basis_generations (
    generation_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    request_key text NOT NULL UNIQUE CHECK (request_key ~ '^[a-f0-9]{64}$'),
    parent_id bigint REFERENCES liquidity.price_basis_generations(generation_id),
    base_date date NOT NULL,
    target_date date NOT NULL CHECK (target_date > base_date),
    state text NOT NULL DEFAULT 'STAGED' CHECK (state IN ('STAGED','ACTIVE')),
    created_at timestamptz NOT NULL DEFAULT now(),
    activated_at timestamptz,
    CHECK ((state='STAGED' AND activated_at IS NULL) OR (state='ACTIVE' AND activated_at IS NOT NULL)),
    CHECK (parent_id IS NULL OR parent_id < generation_id)
);
CREATE TABLE liquidity.price_basis_receipts (
    generation_id bigint NOT NULL REFERENCES liquidity.price_basis_generations(generation_id),
    ticker text NOT NULL REFERENCES liquidity.instruments(ticker),
    expected_rows bigint NOT NULL CHECK (expected_rows > 0),
    baseline_sha256 text NOT NULL CHECK (baseline_sha256 ~ '^[a-f0-9]{64}$'),
    provider_sha256 text NOT NULL CHECK (provider_sha256 ~ '^[a-f0-9]{64}$'),
    source_version text NOT NULL,
    retrieved_at timestamptz NOT NULL,
    PRIMARY KEY (generation_id,ticker)
);
CREATE TABLE liquidity.price_basis_revisions (
    generation_id bigint NOT NULL REFERENCES liquidity.price_basis_generations(generation_id),
    ticker text NOT NULL,
    date date NOT NULL CHECK (date <> DATE '2026-11-08'),
    open double precision NOT NULL CHECK (open > 0 AND open < 'Infinity'::float8),
    high double precision NOT NULL CHECK (high > 0 AND high < 'Infinity'::float8),
    low double precision NOT NULL CHECK (low > 0 AND low < 'Infinity'::float8),
    close double precision NOT NULL CHECK (close > 0 AND close < 'Infinity'::float8),
    volume bigint NOT NULL CHECK (volume >= 0),
    turnover double precision CHECK (turnover >= 0 AND turnover < 'Infinity'::float8),
    PRIMARY KEY (generation_id,ticker,date),
    FOREIGN KEY (ticker,date) REFERENCES liquidity.market_eod(ticker,date),
    FOREIGN KEY (generation_id,ticker) REFERENCES liquidity.price_basis_receipts(generation_id,ticker),
    CHECK (high >= low AND open BETWEEN low AND high AND close BETWEEN low AND high)
);
CREATE INDEX price_basis_revision_lookup ON liquidity.price_basis_revisions(ticker,date,generation_id DESC);

CREATE VIEW liquidity.market_eod_consistent AS
WITH revisions AS (
    SELECT DISTINCT ON (r.ticker,r.date) r.*
    FROM liquidity.price_basis_revisions r
    JOIN liquidity.price_basis_generations g USING(generation_id)
    WHERE g.state='ACTIVE'
    ORDER BY r.ticker,r.date,r.generation_id DESC
)
SELECT m.ticker,m.date,coalesce(r.open,m.open) AS open,
       coalesce(r.high,m.high) AS high,coalesce(r.low,m.low) AS low,
       coalesce(r.close,m.close) AS close,coalesce(r.volume,m.volume) AS volume,
       CASE WHEN r.generation_id IS NULL THEN m.turnover ELSE r.turnover END AS turnover,
       m.provenance_id AS original_provenance_id,r.generation_id AS price_basis_generation
FROM liquidity.market_eod m LEFT JOIN revisions r USING(ticker,date)
WHERE m.date <> DATE '2026-11-08';

CREATE FUNCTION liquidity.price_basis_profiles(cutoff date)
RETURNS TABLE(ticker text,rows bigint,values_sha256 text)
LANGUAGE sql STABLE SET search_path=pg_catalog AS $$
 SELECT m.ticker,count(*),encode(sha256(string_agg(
  int8send((m.date-DATE '0001-01-01')::bigint+1)||float8send(m.open)||float8send(m.high)||
  float8send(m.low)||float8send(m.close)||int8send(m.volume)||
  CASE WHEN m.turnover IS NULL THEN decode('4e','hex') ELSE decode('56','hex')||float8send(m.turnover) END,
  ''::bytea ORDER BY m.date)),'hex')
 FROM liquidity.market_eod_consistent m WHERE m.date<=cutoff GROUP BY m.ticker
$$;

CREATE FUNCTION liquidity.guard_price_basis_rows() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog AS $$
DECLARE gid bigint; g liquidity.price_basis_generations%ROWTYPE;
BEGIN
 IF TG_OP<>'INSERT' AND EXISTS (SELECT 1 FROM liquidity.price_basis_generations WHERE generation_id=OLD.generation_id AND state='ACTIVE') THEN
  RAISE EXCEPTION 'Activated price-basis evidence is immutable' USING ERRCODE='23514';
 END IF;
 gid := CASE WHEN TG_OP='DELETE' THEN OLD.generation_id ELSE NEW.generation_id END;
 SELECT * INTO g FROM liquidity.price_basis_generations WHERE generation_id=gid FOR UPDATE;
 IF g.state='ACTIVE' THEN RAISE EXCEPTION 'Activated price-basis evidence is immutable' USING ERRCODE='23514'; END IF;
 IF TG_TABLE_NAME='price_basis_revisions' AND TG_OP<>'DELETE' THEN
  IF NEW.date>g.base_date THEN RAISE EXCEPTION 'Historical revision exceeds generation base date' USING ERRCODE='23514'; END IF;
 END IF;
 IF TG_OP='DELETE' THEN RETURN OLD; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER price_basis_receipts_immutable BEFORE INSERT OR UPDATE OR DELETE ON liquidity.price_basis_receipts FOR EACH ROW EXECUTE FUNCTION liquidity.guard_price_basis_rows();
CREATE TRIGGER price_basis_revisions_immutable BEFORE INSERT OR UPDATE OR DELETE ON liquidity.price_basis_revisions FOR EACH ROW EXECUTE FUNCTION liquidity.guard_price_basis_rows();

CREATE FUNCTION liquidity.guard_price_basis_header() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog AS $$
BEGIN
 IF TG_OP='DELETE' OR OLD.state='ACTIVE' THEN
  RAISE EXCEPTION 'Price-basis generation cannot be deleted or rewritten after activation' USING ERRCODE='23514';
 END IF;
 IF (NEW.request_key,NEW.parent_id,NEW.base_date,NEW.target_date) IS DISTINCT FROM
    (OLD.request_key,OLD.parent_id,OLD.base_date,OLD.target_date) THEN
  RAISE EXCEPTION 'Price-basis identity cannot be rewritten' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER price_basis_header_immutable BEFORE UPDATE OR DELETE ON liquidity.price_basis_generations FOR EACH ROW EXECUTE FUNCTION liquidity.guard_price_basis_header();

CREATE FUNCTION liquidity.validate_price_basis_activation() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog AS $$
DECLARE prior bigint;
BEGIN
 IF NEW.state<>'ACTIVE' THEN RETURN NULL; END IF;
 PERFORM pg_advisory_xact_lock(6381402502);
 SELECT max(generation_id) INTO prior FROM liquidity.price_basis_generations WHERE state='ACTIVE' AND generation_id<>NEW.generation_id;
 IF prior IS DISTINCT FROM NEW.parent_id THEN RAISE EXCEPTION 'Stale price-basis parent' USING ERRCODE='23514'; END IF;
 IF (SELECT count(*) FROM liquidity.price_basis_receipts WHERE generation_id=NEW.generation_id)<>302
 OR EXISTS (SELECT 1 FROM liquidity.instruments i WHERE NOT EXISTS
  (SELECT 1 FROM liquidity.price_basis_receipts r WHERE r.generation_id=NEW.generation_id AND r.ticker=i.ticker))
 OR EXISTS (
  SELECT 1 FROM liquidity.price_basis_receipts r
  LEFT JOIN liquidity.price_basis_profiles(NEW.base_date) p USING(ticker)
  WHERE r.generation_id=NEW.generation_id AND
   (p.rows IS DISTINCT FROM r.expected_rows OR p.values_sha256 IS DISTINCT FROM r.provider_sha256)
 ) THEN RAISE EXCEPTION 'Incomplete or inconsistent full-history price basis' USING ERRCODE='23514'; END IF;
 RETURN NULL;
END $$;
CREATE CONSTRAINT TRIGGER price_basis_complete AFTER INSERT OR UPDATE ON liquidity.price_basis_generations
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION liquidity.validate_price_basis_activation();
