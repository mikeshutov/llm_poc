CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL REFERENCES user_profile(user_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    plan JSONB NOT NULL,
    plan_version INTEGER NOT NULL DEFAULT 1,
    plan_generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    plan_status TEXT NOT NULL DEFAULT 'ready',
    plan_prompt_hash TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    days_of_week SMALLINT[] NOT NULL,
    run_time TIME NOT NULL,
    timezone TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT jobs_user_name_key UNIQUE (user_id, name),
    CONSTRAINT jobs_name_not_blank CHECK (BTRIM(name) <> ''),
    CONSTRAINT jobs_prompt_not_blank CHECK (BTRIM(prompt) <> ''),
    CONSTRAINT jobs_plan_object CHECK (jsonb_typeof(plan) = 'object'),
    CONSTRAINT jobs_plan_version_positive CHECK (plan_version > 0),
    CONSTRAINT jobs_plan_status_valid CHECK (plan_status IN ('ready', 'stale', 'failed')),
    CONSTRAINT jobs_plan_prompt_hash_not_blank CHECK (BTRIM(plan_prompt_hash) <> ''),
    CONSTRAINT jobs_enabled_plan_valid CHECK (NOT enabled OR (plan_status = 'ready' AND COALESCE(jsonb_array_length(plan->'steps'), 0) > 0)),
    CONSTRAINT jobs_days_not_empty CHECK (cardinality(days_of_week) > 0),
    CONSTRAINT jobs_days_valid CHECK (days_of_week <@ ARRAY[1, 2, 3, 4, 5, 6, 7]::SMALLINT[]),
    CONSTRAINT jobs_timezone_not_blank CHECK (BTRIM(timezone) <> '')
);

CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_user_enabled ON jobs(user_id, enabled);

CREATE TABLE IF NOT EXISTS job_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    scheduled_for TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    queue_message_id TEXT,
    deduplication_key TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT job_runs_job_schedule_key UNIQUE (job_id, scheduled_for),
    CONSTRAINT job_runs_id_job_key UNIQUE (id, job_id),
    CONSTRAINT job_runs_status_valid CHECK (status IN ('pending', 'enqueued', 'running', 'succeeded', 'failed')),
    CONSTRAINT job_runs_deduplication_key_unique UNIQUE (deduplication_key)
);

CREATE INDEX IF NOT EXISTS idx_job_runs_job_id ON job_runs(job_id);

CREATE TABLE IF NOT EXISTS job_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    job_run_id UUID NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    text TEXT NOT NULL DEFAULT '',
    error_text TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    CONSTRAINT job_results_run_key UNIQUE (job_run_id),
    CONSTRAINT job_results_run_job_fk FOREIGN KEY (job_run_id, job_id)
        REFERENCES job_runs(id, job_id) ON DELETE CASCADE,
    CONSTRAINT job_results_status_valid CHECK (status IN ('pending', 'succeeded', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_job_results_job_id ON job_results(job_id);

CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL REFERENCES user_profile(user_id) ON DELETE CASCADE,
    artifact_type TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT artifacts_type_not_blank CHECK (BTRIM(artifact_type) <> ''),
    CONSTRAINT artifacts_id_not_blank CHECK (BTRIM(artifact_id) <> ''),
    CONSTRAINT artifacts_user_reference_key UNIQUE (user_id, artifact_type, artifact_id)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_user_id ON artifacts(user_id);

CREATE TABLE IF NOT EXISTS job_result_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_result_id UUID NOT NULL REFERENCES job_results(id) ON DELETE CASCADE,
    artifact_id UUID NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT job_result_artifacts_unique_reference UNIQUE (job_result_id, artifact_id)
);

CREATE INDEX IF NOT EXISTS idx_job_result_artifacts_result_id
    ON job_result_artifacts(job_result_id);
