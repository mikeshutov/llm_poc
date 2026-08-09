CREATE TABLE IF NOT EXISTS llm_call (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversation(id) ON DELETE CASCADE,
    roundtrip_id UUID REFERENCES conversation_roundtrip(id) ON DELETE CASCADE,
    agent TEXT,
    stage TEXT,
    callsite TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    input_price_per_million_tokens NUMERIC(12, 6) NOT NULL,
    output_price_per_million_tokens NUMERIC(12, 6) NOT NULL,
    computed_input_cost NUMERIC(18, 10) NOT NULL,
    computed_output_cost NUMERIC(18, 10) NOT NULL,
    computed_total_cost NUMERIC(18, 10) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_llm_call_conversation_id
    ON llm_call(conversation_id);

CREATE INDEX IF NOT EXISTS idx_llm_call_roundtrip_id
    ON llm_call(roundtrip_id);
