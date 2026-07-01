DROP TABLE IF EXISTS plans;
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
