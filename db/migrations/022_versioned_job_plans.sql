ALTER TABLE jobs ADD COLUMN IF NOT EXISTS current_plan_id UUID;

ALTER TABLE plans ALTER COLUMN roundtrip_id DROP NOT NULL;
ALTER TABLE plans ADD COLUMN IF NOT EXISTS user_id TEXT REFERENCES user_profile(user_id) ON DELETE CASCADE;
ALTER TABLE plans ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES jobs(id) ON DELETE CASCADE;
ALTER TABLE plans ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1;
ALTER TABLE plans ADD COLUMN IF NOT EXISTS prompt_hash TEXT;
ALTER TABLE plans ADD COLUMN IF NOT EXISTS plan_kind TEXT NOT NULL DEFAULT 'interactive';

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'plans_version_positive') THEN
        ALTER TABLE plans ADD CONSTRAINT plans_version_positive CHECK (version > 0);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'plans_kind_valid') THEN
        ALTER TABLE plans ADD CONSTRAINT plans_kind_valid CHECK (plan_kind IN ('interactive', 'job'));
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS idx_plans_job_version ON plans(job_id, version) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_plans_job_id ON plans(job_id);
