DROP TABLE IF EXISTS tool_calls;
DROP TABLE IF EXISTS plans;
DROP TABLE IF EXISTS roundtrip_feedback;
DROP TABLE IF EXISTS conversation_roundtrip;

CREATE TABLE conversation_roundtrip (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
  message_index INTEGER NOT NULL,        -- turn_index within a conversation_id
  user_prompt TEXT NOT NULL,
  generated_response TEXT NOT NULL,
  roundtrip_summary TEXT,
  roundtrip_summary_embedding vector(1536),
  response_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  parsed_query JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  model TEXT,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

  CONSTRAINT uq_roundtrip_conversation_turn UNIQUE (conversation_id, message_index) -- an index can only appear once within a conversation
);
CREATE INDEX IF NOT EXISTS conversation_roundtrip_summary_embedding_idx
  ON conversation_roundtrip USING ivfflat (roundtrip_summary_embedding vector_l2_ops) WITH (lists = 100);

