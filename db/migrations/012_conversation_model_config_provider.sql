ALTER TABLE conversation_model_config
ADD COLUMN IF NOT EXISTS provider TEXT;

UPDATE conversation_model_config
SET provider = 'openai'
WHERE provider IS NULL OR BTRIM(provider) = '';

ALTER TABLE conversation_model_config
ALTER COLUMN provider SET NOT NULL;

ALTER TABLE conversation_model_config
ALTER COLUMN provider SET DEFAULT 'openai';
