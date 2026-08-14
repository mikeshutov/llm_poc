CREATE TABLE IF NOT EXISTS conversation_event (
    id BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
    roundtrip_id UUID REFERENCES conversation_roundtrip(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    agent_name TEXT NOT NULL DEFAULT '',
    node_name TEXT NOT NULL DEFAULT '',
    step_id TEXT NOT NULL DEFAULT '',
    iteration INTEGER,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversation_event_conversation_id
    ON conversation_event(conversation_id);

CREATE INDEX IF NOT EXISTS idx_conversation_event_roundtrip_id
    ON conversation_event(roundtrip_id);

CREATE INDEX IF NOT EXISTS idx_conversation_event_event_type
    ON conversation_event(event_type);
