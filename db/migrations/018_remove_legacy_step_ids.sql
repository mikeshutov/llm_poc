ALTER TABLE conversation_event
DROP COLUMN IF EXISTS step_id;

ALTER TABLE evidence_views
DROP COLUMN IF EXISTS step_id;

ALTER TABLE evidence_views
DROP COLUMN IF EXISTS evidence_id;
