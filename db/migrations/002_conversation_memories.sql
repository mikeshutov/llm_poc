ALTER TABLE conversation
    ADD COLUMN IF NOT EXISTS summary_embedding vector(1536);

CREATE INDEX IF NOT EXISTS conversation_summary_embedding_idx
    ON conversation USING ivfflat (summary_embedding vector_l2_ops) WITH (lists = 100);
