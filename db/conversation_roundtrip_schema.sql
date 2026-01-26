DROP TABLE IF EXISTS conversation_roundtrip;

CREATE TABLE conversation_roundtrip (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
  message_index INTEGER NOT NULL,        -- turn_index within a conversation_id
  user_prompt TEXT NOT NULL,
  generated_response TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

  CONSTRAINT uq_roundtrip_conversation_turn UNIQUE (conversation_id, message_index) -- an index can only appear once within a conversation
);
