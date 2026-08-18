CREATE TABLE IF NOT EXISTS user_agent (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL REFERENCES user_profile(user_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    allowed_categories TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    planner_instruction TEXT NOT NULL,
    planner_rules TEXT NOT NULL DEFAULT '',
    max_turns INTEGER NOT NULL DEFAULT 10,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT user_agent_name_not_blank CHECK (BTRIM(name) <> ''),
    CONSTRAINT user_agent_max_turns_positive CHECK (max_turns > 0),
    CONSTRAINT user_agent_user_name_key UNIQUE (user_id, name)
);

CREATE INDEX IF NOT EXISTS idx_user_agent_user_id
    ON user_agent(user_id);

CREATE INDEX IF NOT EXISTS idx_user_agent_user_active
    ON user_agent(user_id, is_active);
