CREATE TABLE IF NOT EXISTS user_attributes (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id text NULL,
    attribute_text text NOT NULL,
    attribute_embedding vector(1536),
    attribute_type text NULL,
    source text NULL,
    source_conversation_id uuid NULL REFERENCES conversation(id),
    source_roundtrip_id uuid NULL REFERENCES conversation_roundtrip(id),
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    confidence double precision NULL,
    importance double precision NULL
);

CREATE INDEX IF NOT EXISTS user_attributes_attribute_embedding_idx
    ON user_attributes USING ivfflat (attribute_embedding vector_l2_ops) WITH (lists = 100);
