CREATE TABLE IF NOT EXISTS tool_cache_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_key TEXT NOT NULL UNIQUE,
    tool_name TEXT NOT NULL,
    tool_version INTEGER NOT NULL,
    input_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_payload JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT tool_cache_entries_version_positive CHECK (tool_version > 0),
    CONSTRAINT tool_cache_entries_expiry_valid CHECK (expires_at > created_at)
);

CREATE INDEX IF NOT EXISTS idx_tool_cache_entries_expiry
    ON tool_cache_entries(expires_at);
CREATE INDEX IF NOT EXISTS idx_tool_cache_entries_tool
    ON tool_cache_entries(tool_name, tool_version);
