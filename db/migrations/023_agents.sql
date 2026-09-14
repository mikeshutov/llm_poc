ALTER TABLE user_agent RENAME TO agents;
ALTER TABLE user_agent_model_config RENAME TO agent_model_config;
ALTER TABLE agent_model_config RENAME COLUMN user_agent_id TO agent_id;

ALTER TABLE agents ADD COLUMN agent_type TEXT NOT NULL DEFAULT 'user';
ALTER TABLE agents ALTER COLUMN user_id DROP NOT NULL;

ALTER TABLE agents RENAME CONSTRAINT user_agent_name_not_blank TO agents_name_not_blank;
ALTER TABLE agents RENAME CONSTRAINT user_agent_max_turns_positive TO agents_max_turns_positive;
ALTER TABLE agents RENAME CONSTRAINT user_agent_user_name_key TO agents_user_name_key;
ALTER TABLE agents ADD CONSTRAINT agents_type_valid CHECK (agent_type IN ('system', 'user'));
ALTER TABLE agents ADD CONSTRAINT agents_type_user_id_consistent CHECK (
    (agent_type = 'system' AND user_id IS NULL)
    OR (agent_type = 'user' AND user_id IS NOT NULL AND BTRIM(user_id) <> '')
);

DROP INDEX IF EXISTS idx_user_agent_user_id;
DROP INDEX IF EXISTS idx_user_agent_user_active;
ALTER INDEX IF EXISTS user_agent_description_embedding_idx RENAME TO agents_description_embedding_idx;

CREATE INDEX idx_agents_user_id ON agents(user_id);
CREATE INDEX idx_agents_user_active ON agents(user_id, is_active);
CREATE INDEX idx_agents_type_active ON agents(agent_type, is_active);

ALTER TABLE agent_model_config RENAME CONSTRAINT user_agent_model_config_stage_not_blank TO agent_model_config_stage_not_blank;
ALTER TABLE agent_model_config RENAME CONSTRAINT user_agent_model_config_provider_not_blank TO agent_model_config_provider_not_blank;
ALTER TABLE agent_model_config RENAME CONSTRAINT user_agent_model_config_model_not_blank TO agent_model_config_model_not_blank;
ALTER TABLE agent_model_config RENAME CONSTRAINT user_agent_model_config_pkey TO agent_model_config_pkey;
ALTER TABLE agent_model_config RENAME CONSTRAINT user_agent_model_config_user_agent_id_fkey TO agent_model_config_agent_id_fkey;

ALTER INDEX idx_user_agent_model_config_user_agent_id RENAME TO idx_agent_model_config_agent_id;
