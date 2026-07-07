CREATE TABLE IF NOT EXISTS memories (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id text NULL,
    memory_text text NOT NULL,
    memory_embedding vector(1536),
    memory_type text NULL,
    source text NULL,
    source_conversation_id uuid NULL REFERENCES conversation(id),
    source_roundtrip_id uuid NULL REFERENCES conversation_roundtrip(id),
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    confidence double precision NULL,
    importance double precision NULL
);

CREATE INDEX IF NOT EXISTS memories_memory_embedding_idx
    ON memories USING ivfflat (memory_embedding vector_l2_ops) WITH (lists = 100);
