DROP TABLE IF EXISTS tool_calls;

CREATE TABLE tool_calls (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  roundtrip_id UUID NOT NULL REFERENCES conversation_roundtrip(id) ON DELETE CASCADE,
  call_index INTEGER NOT NULL,
  tool_name TEXT NOT NULL,
  status TEXT NOT NULL,
  reason TEXT NULL,
  input_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  output_payload JSONB NULL,
  error_message TEXT NULL,
  duration_ms INTEGER NULL,
  goal TEXT NULL,
  done BOOLEAN NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  CONSTRAINT uq_tool_calls_roundtrip_idx UNIQUE (roundtrip_id, call_index)
);

CREATE INDEX idx_tool_calls_roundtrip_id ON tool_calls(roundtrip_id);
CREATE INDEX idx_tool_calls_tool_name ON tool_calls(tool_name);