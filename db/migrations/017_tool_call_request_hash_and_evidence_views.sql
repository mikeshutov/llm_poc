ALTER TABLE tool_calls
ADD COLUMN IF NOT EXISTS request_hash TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_tool_calls_request_hash
    ON tool_calls(request_hash);

CREATE TABLE IF NOT EXISTS evidence_views (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tool_call_id UUID NOT NULL REFERENCES tool_calls(id) ON DELETE CASCADE,
    evidence_id TEXT NOT NULL,
    step_id TEXT NOT NULL DEFAULT '',
    item_id TEXT NOT NULL DEFAULT '',
    tool_name TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    urls JSONB NOT NULL DEFAULT '[]'::jsonb,
    image_url TEXT NOT NULL DEFAULT '',
    published_at TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    entity_type TEXT NOT NULL DEFAULT '',
    location_name TEXT NOT NULL DEFAULT '',
    hash TEXT NOT NULL DEFAULT '',
    llm_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_payload JSONB NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT evidence_views_tool_call_evidence_key UNIQUE (tool_call_id, evidence_id)
);

CREATE INDEX IF NOT EXISTS idx_evidence_views_tool_call_id
    ON evidence_views(tool_call_id);

CREATE INDEX IF NOT EXISTS idx_evidence_views_evidence_id
    ON evidence_views(evidence_id);

CREATE INDEX IF NOT EXISTS idx_evidence_views_hash
    ON evidence_views(hash);
