ALTER TABLE user_agent
ADD COLUMN IF NOT EXISTS execution_strategy TEXT NOT NULL DEFAULT 'planner_executor_evaluator';

CREATE TABLE IF NOT EXISTS user_agent_model_config (
    user_agent_id UUID NOT NULL REFERENCES user_agent(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_agent_id, stage),
    CONSTRAINT user_agent_model_config_stage_not_blank CHECK (BTRIM(stage) <> ''),
    CONSTRAINT user_agent_model_config_provider_not_blank CHECK (BTRIM(provider) <> ''),
    CONSTRAINT user_agent_model_config_model_not_blank CHECK (BTRIM(model) <> '')
);

CREATE INDEX IF NOT EXISTS idx_user_agent_model_config_user_agent_id
    ON user_agent_model_config(user_agent_id);
