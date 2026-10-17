DROP TABLE IF EXISTS tool_calls;

CREATE TABLE tool_calls (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  roundtrip_id UUID NOT NULL REFERENCES conversation_roundtrip(id) ON DELETE CASCADE,
  plan_id UUID NULL REFERENCES plans(id) ON DELETE SET NULL,
  plan_step_id UUID NULL,      -- matches PlanStep.db_id stored in plans.steps JSONB
  step_index INTEGER NULL,     -- position of the step within the plan for ordering
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