CREATE TABLE IF NOT EXISTS conversation_model_config (
    conversation_id UUID NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
    agent TEXT NOT NULL,
    stage TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (conversation_id, agent, stage)
);

CREATE INDEX IF NOT EXISTS idx_conversation_model_config_conversation_id
    ON conversation_model_config(conversation_id);
