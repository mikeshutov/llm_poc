-- Initial database schema. This replaces the former db/setup bootstrap files.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- The baseline is destructive. Explicitly remove application tables that may
-- predate this migration so historical CREATE TABLE IF NOT EXISTS statements
-- cannot preserve incompatible schemas during a clean replay.
DROP TABLE IF EXISTS
  job_result_artifacts,
  artifacts,
  job_results,
  job_runs,
  jobs,
  tool_cache_entries,
  tools,
  user_agent_model_config,
  agent_model_config,
  user_agent,
  agents,
  agent_versions,
  user_attributes,
  user_profile,
  tone,
  conversation_model_config,
  conversation_event,
  conversation_message,
  llm_call,
  evidence_views,
  roundtrip_prompts,
  roundtrip_feedback,
  memories
CASCADE;

DROP TYPE IF EXISTS plan_status CASCADE;

DROP TABLE IF EXISTS products CASCADE;

CREATE TABLE products (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  category TEXT,
  color TEXT,
  style TEXT,
  gender TEXT,
  season TEXT,
  year INT,
  price NUMERIC,
  image_url TEXT,
  embedding vector(1536)
);

CREATE INDEX products_category_idx ON products (category);
CREATE INDEX products_color_idx ON products (color);
CREATE INDEX products_price_idx ON products (price);
CREATE INDEX IF NOT EXISTS products_embedding_idx ON products USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);

-- This baseline is intentionally destructive: existing data and dependent
-- legacy tables are disposable for a fresh refactor install.
DROP TABLE IF EXISTS conversation CASCADE;

CREATE TABLE conversation (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NULL,
  user_id TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  tone_state JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS files (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  file_path TEXT NOT NULL,
  file_name TEXT NOT NULL,
  file_type TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}',
  uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS file_chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  file_id UUID NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  embedding VECTOR(1536),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_file_chunk UNIQUE (file_id, chunk_index)
);

DROP TABLE IF EXISTS conversation_roundtrip CASCADE;

CREATE TABLE conversation_roundtrip (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
  message_index INTEGER NOT NULL,
  user_prompt TEXT NOT NULL,
  generated_response TEXT NOT NULL,
  response_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  relevant_evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
  parsed_query JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  CONSTRAINT uq_roundtrip_conversation_turn UNIQUE (conversation_id, message_index)
);

DROP TABLE IF EXISTS conversation_summary CASCADE;

CREATE TABLE conversation_summary (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
  summary TEXT NOT NULL,
  message_index_cutoff INTEGER NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

DROP TABLE IF EXISTS plans CASCADE;
DROP TYPE IF EXISTS plan_status;

CREATE TYPE plan_status AS ENUM ('pending', 'running', 'completed', 'failed');

CREATE TABLE plans (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  roundtrip_id UUID NOT NULL REFERENCES conversation_roundtrip(id) ON DELETE CASCADE,
  steps JSONB NOT NULL DEFAULT '[]'::jsonb,
  current_step_index INTEGER NOT NULL DEFAULT 0,
  status plan_status NOT NULL DEFAULT 'pending',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_plans_roundtrip_id ON plans(roundtrip_id);

DROP TABLE IF EXISTS tool_calls CASCADE;

CREATE TABLE tool_calls (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  roundtrip_id UUID NOT NULL REFERENCES conversation_roundtrip(id) ON DELETE CASCADE,
  plan_id UUID NULL REFERENCES plans(id) ON DELETE SET NULL,
  plan_step_id UUID NULL,
  step_index INTEGER NULL,
  tool_name TEXT NOT NULL,
  status TEXT NOT NULL,
  input_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  output_payload JSONB NULL,
  error_message TEXT NULL,
  duration_ms INTEGER NULL,
  goal TEXT NULL,
  summary TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_tool_calls_plan_step UNIQUE (roundtrip_id, plan_step_id)
);

CREATE INDEX idx_tool_calls_roundtrip_id ON tool_calls(roundtrip_id);
CREATE INDEX idx_tool_calls_plan_id ON tool_calls(plan_id);
CREATE INDEX idx_tool_calls_plan_step_id ON tool_calls(plan_step_id);
CREATE INDEX idx_tool_calls_tool_name ON tool_calls(tool_name);

DROP TABLE IF EXISTS rest_cache CASCADE;

CREATE TABLE rest_cache (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  url TEXT NOT NULL,
  params_hash TEXT NOT NULL,
  params_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  response_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at TIMESTAMPTZ NOT NULL,
  CONSTRAINT uq_rest_cache_url_hash UNIQUE (url, params_hash)
);

CREATE INDEX idx_rest_cache_url ON rest_cache(url);
CREATE INDEX idx_rest_cache_expires_at ON rest_cache(expires_at);
