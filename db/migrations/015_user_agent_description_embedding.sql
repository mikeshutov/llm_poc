ALTER TABLE user_agent
ADD COLUMN IF NOT EXISTS description_embedding vector(1536);

CREATE INDEX IF NOT EXISTS user_agent_description_embedding_idx
    ON user_agent USING ivfflat (description_embedding vector_cosine_ops) WITH (lists = 100);
