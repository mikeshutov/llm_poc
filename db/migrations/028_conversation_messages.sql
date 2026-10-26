-- Make the ordered conversation message stream canonical for roundtrips.
DROP TABLE IF EXISTS evidence_views CASCADE;
DROP TABLE IF EXISTS tool_calls CASCADE;
DROP TABLE IF EXISTS conversation_event CASCADE;
DROP TABLE IF EXISTS roundtrip_prompts CASCADE;
DROP TABLE IF EXISTS llm_call CASCADE;

ALTER TABLE conversation_roundtrip
    DROP COLUMN IF EXISTS user_prompt,
    DROP COLUMN IF EXISTS generated_response,
    DROP COLUMN IF EXISTS response_payload,
    DROP COLUMN IF EXISTS relevant_evidence,
    DROP COLUMN IF EXISTS parsed_query,
    DROP COLUMN IF EXISTS assistant_follow_up,
    DROP COLUMN IF EXISTS model;

ALTER TABLE conversation_roundtrip
    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'pending';

CREATE TABLE conversation_message (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  roundtrip_id UUID NOT NULL REFERENCES conversation_roundtrip(id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('user', 'assistant', 'tool_call', 'tool_result')),
  visibility TEXT NOT NULL DEFAULT 'visible' CHECK (visibility IN ('visible', 'internal')),
  role TEXT NOT NULL,
  content TEXT NOT NULL DEFAULT '',
  payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  parent_message_id UUID NULL REFERENCES conversation_message(id) ON DELETE SET NULL,
  embedding vector(1536) NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_conversation_message_sequence UNIQUE (roundtrip_id, sequence)
);

CREATE INDEX idx_conversation_message_roundtrip
  ON conversation_message(roundtrip_id, sequence);
CREATE INDEX idx_conversation_message_kind
  ON conversation_message(roundtrip_id, kind);
