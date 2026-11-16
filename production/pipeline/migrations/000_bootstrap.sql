CREATE SCHEMA IF NOT EXISTS liquidity;
CREATE TABLE IF NOT EXISTS liquidity.schema_migrations (
    version text PRIMARY KEY,
    sha256 text NOT NULL CHECK (sha256 ~ '^[a-f0-9]{64}$'),
    applied_at timestamptz NOT NULL DEFAULT now()
);
