ALTER TABLE tool_calls
    ADD COLUMN IF NOT EXISTS hypothesis TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS call_metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE tool_calls
    ADD CONSTRAINT tool_calls_call_metadata_object
    CHECK (jsonb_typeof(call_metadata) = 'object');

