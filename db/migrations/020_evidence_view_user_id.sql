ALTER TABLE evidence_views
ADD COLUMN IF NOT EXISTS user_id TEXT NULL;

CREATE INDEX IF NOT EXISTS idx_evidence_views_user_id
    ON evidence_views(user_id);
