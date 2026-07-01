DROP TABLE IF EXISTS roundtrip_feedback;

CREATE TABLE roundtrip_feedback (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  roundtrip_id UUID NOT NULL UNIQUE REFERENCES conversation_roundtrip(id) ON DELETE CASCADE,
  met_expectation BOOLEAN NOT NULL,
  reason TEXT,
  expected_answer TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  model TEXT
);
