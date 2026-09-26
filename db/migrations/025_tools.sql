CREATE TABLE IF NOT EXISTS tools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    description TEXT NOT NULL DEFAULT '',
    tool_embedding vector(1536),
    result_type TEXT NOT NULL DEFAULT 'generic',
    rate_limit_key TEXT,
    retry_policy JSONB NOT NULL DEFAULT '{}'::jsonb,
    rate_limit_policy JSONB,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT tools_name_not_blank CHECK (BTRIM(name) <> ''),
    CONSTRAINT tools_version_positive CHECK (version > 0),
    CONSTRAINT tools_result_type_not_blank CHECK (BTRIM(result_type) <> ''),
    CONSTRAINT tools_retry_policy_object CHECK (jsonb_typeof(retry_policy) = 'object'),
    CONSTRAINT tools_rate_limit_policy_object CHECK (
        rate_limit_policy IS NULL OR jsonb_typeof(rate_limit_policy) = 'object'
    ),
    CONSTRAINT tools_metadata_object CHECK (jsonb_typeof(metadata) = 'object'),
    CONSTRAINT tools_name_key UNIQUE (name)
);

CREATE INDEX IF NOT EXISTS idx_tools_active ON tools(is_active);

CREATE INDEX IF NOT EXISTS idx_tools_tool_embedding
    ON tools USING ivfflat (tool_embedding vector_cosine_ops) WITH (lists = 100);
