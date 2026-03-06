-- NOTE: In production this should be backed by DynamoDB (or similar key-value store)
--       with a TTL attribute for automatic expiry. Using PostgreSQL here for POC simplicity.

DROP TABLE IF EXISTS rest_cache;

CREATE TABLE rest_cache (
  id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  url              TEXT        NOT NULL,
  params_hash      TEXT        NOT NULL,    -- SHA-256 of canonically sorted params JSON
  params_payload   JSONB       NOT NULL DEFAULT '{}'::jsonb,
  response_payload JSONB       NOT NULL DEFAULT '{}'::jsonb,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at       TIMESTAMPTZ NOT NULL,

  CONSTRAINT uq_rest_cache_url_hash UNIQUE (url, params_hash)
);

CREATE INDEX idx_rest_cache_url ON rest_cache(url);
CREATE INDEX idx_rest_cache_expires_at ON rest_cache(expires_at);
