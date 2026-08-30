ALTER TABLE conversation_roundtrip
ADD COLUMN IF NOT EXISTS relevant_evidence JSONB NOT NULL DEFAULT '{}'::jsonb;
