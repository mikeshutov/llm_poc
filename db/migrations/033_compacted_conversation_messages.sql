ALTER TABLE conversation_message
    ADD COLUMN IF NOT EXISTS compacted BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_conversation_message_uncompacted_context
    ON conversation_message(roundtrip_id, sequence)
    WHERE NOT compacted AND NOT is_subagent;
