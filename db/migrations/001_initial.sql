ALTER TABLE conversation
    ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT '';

ALTER TABLE conversation_roundtrip
    ADD COLUMN IF NOT EXISTS roundtrip_summary TEXT,
    ADD COLUMN IF NOT EXISTS roundtrip_summary_embedding vector(1536),
    ADD COLUMN IF NOT EXISTS model TEXT;

CREATE INDEX IF NOT EXISTS conversation_roundtrip_summary_embedding_idx
    ON conversation_roundtrip USING ivfflat (roundtrip_summary_embedding vector_l2_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS roundtrip_prompts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    roundtrip_id UUID NOT NULL REFERENCES conversation_roundtrip(id) ON DELETE CASCADE,
    agent TEXT NOT NULL,
    prompt_step TEXT NOT NULL,
    prompt TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_roundtrip_prompts_roundtrip_id
    ON roundtrip_prompts(roundtrip_id);

CREATE INDEX IF NOT EXISTS idx_roundtrip_prompts_roundtrip_agent_step
    ON roundtrip_prompts(roundtrip_id, agent, prompt_step);

CREATE TABLE IF NOT EXISTS roundtrip_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    roundtrip_id UUID NOT NULL REFERENCES conversation_roundtrip(id) ON DELETE CASCADE,
    met_expectation BOOLEAN,
    reason TEXT,
    expected_answer TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    model TEXT
);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'roundtrip_feedback'
          AND column_name = 'rating'
    ) AND NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'roundtrip_feedback'
          AND column_name = 'met_expectation'
    ) THEN
        ALTER TABLE roundtrip_feedback ADD COLUMN met_expectation BOOLEAN;

        UPDATE roundtrip_feedback
        SET met_expectation = CASE
            WHEN rating IN ('up', 'true', '1') THEN TRUE
            ELSE FALSE
        END;
    END IF;
END $$;

UPDATE roundtrip_feedback
SET met_expectation = FALSE
WHERE met_expectation IS NULL;

ALTER TABLE roundtrip_feedback
    ALTER COLUMN met_expectation SET NOT NULL;

ALTER TABLE roundtrip_feedback
    DROP CONSTRAINT IF EXISTS roundtrip_feedback_rating_check;

ALTER TABLE roundtrip_feedback
    DROP COLUMN IF EXISTS rating;

CREATE UNIQUE INDEX IF NOT EXISTS roundtrip_feedback_roundtrip_id_key
    ON roundtrip_feedback(roundtrip_id);

CREATE TABLE IF NOT EXISTS conversation_summary (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    tool_summary TEXT NOT NULL DEFAULT '',
    message_index_cutoff INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE conversation_summary
    ADD COLUMN IF NOT EXISTS tool_summary TEXT NOT NULL DEFAULT '';
