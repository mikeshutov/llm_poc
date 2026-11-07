ALTER TABLE conversation_message
    ADD COLUMN IF NOT EXISTS is_subagent BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS subagent_name TEXT NULL;

CREATE INDEX IF NOT EXISTS idx_conversation_message_context
    ON conversation_message(roundtrip_id, is_subagent, sequence);
