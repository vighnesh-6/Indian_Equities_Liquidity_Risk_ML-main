-- Distinguish complete provider replacement from an unchanged archived basis.
ALTER TABLE liquidity.price_basis_receipts ADD COLUMN basis_evidence jsonb NOT NULL DEFAULT '{"kind":"FULL_PROVIDER"}'::jsonb;
ALTER TABLE liquidity.price_basis_receipts ADD CONSTRAINT price_basis_evidence_kind CHECK (coalesce(
 basis_evidence->>'kind' IN ('FULL_PROVIDER','UNCHANGED_BASIS') AND
 (basis_evidence->>'kind'<>'UNCHANGED_BASIS' OR
  (baseline_sha256=provider_sha256 AND basis_evidence->'post_checkpoint_actions'='[]'::jsonb
   AND (basis_evidence->>'verified_overlap_rows')::integer>=1)),false));
COMMENT ON COLUMN liquidity.price_basis_receipts.provider_sha256 IS
 'Expected canonical historical digest. FULL_PROVIDER means actual replacement observations. UNCHANGED_BASIS retains the independently hashed original archive, supported by matching provider overlap and an empty post-checkpoint corporate-action ledger.';
CREATE FUNCTION liquidity.validate_unchanged_price_basis() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog AS $$
BEGIN
 IF NEW.state<>'ACTIVE' THEN RETURN NULL; END IF;
 IF EXISTS (
  SELECT 1 FROM liquidity.price_basis_receipts r
  JOIN liquidity.market_eod_consistent m ON m.ticker=r.ticker AND m.date=NEW.base_date
  WHERE r.generation_id=NEW.generation_id AND r.basis_evidence->>'kind'='UNCHANGED_BASIS' AND (
   r.basis_evidence->>'checkpoint_sha256' IS DISTINCT FROM encode(sha256(
    int8send((m.date-DATE '0001-01-01')::bigint+1)||float8send(m.open)||float8send(m.high)||float8send(m.low)||float8send(m.close)||int8send(m.volume)||
    CASE WHEN m.turnover IS NULL THEN decode('4e','hex') ELSE decode('56','hex')||float8send(m.turnover) END),'hex')
   OR EXISTS (SELECT 1 FROM liquidity.price_basis_revisions x WHERE x.generation_id=r.generation_id AND x.ticker=r.ticker)
  )
 ) THEN RAISE EXCEPTION 'Unchanged-basis evidence conflicts with observations or revisions' USING ERRCODE='23514'; END IF;
 RETURN NULL;
END $$;
CREATE CONSTRAINT TRIGGER unchanged_basis_complete AFTER INSERT OR UPDATE ON liquidity.price_basis_generations
DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION liquidity.validate_unchanged_price_basis();
