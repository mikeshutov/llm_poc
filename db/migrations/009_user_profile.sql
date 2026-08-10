CREATE TABLE IF NOT EXISTS user_profile (
    user_id TEXT PRIMARY KEY,
    first_name TEXT NULL,
    last_name TEXT NULL,
    display_name TEXT NULL,
    email TEXT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE files
    ADD COLUMN IF NOT EXISTS user_id TEXT NULL,
    ADD COLUMN IF NOT EXISTS conversation_id UUID NULL REFERENCES conversation(id) ON DELETE SET NULL;

ALTER TABLE llm_call
    ADD COLUMN IF NOT EXISTS user_id TEXT NULL;

CREATE INDEX IF NOT EXISTS idx_conversation_user_id
    ON conversation(user_id);

CREATE INDEX IF NOT EXISTS idx_user_attributes_user_id
    ON user_attributes(user_id);

CREATE INDEX IF NOT EXISTS idx_files_user_id
    ON files(user_id);

CREATE INDEX IF NOT EXISTS idx_files_conversation_id
    ON files(conversation_id);

CREATE INDEX IF NOT EXISTS idx_llm_call_user_id
    ON llm_call(user_id);

ALTER TABLE conversation
    ADD CONSTRAINT fk_conversation_user_profile
    FOREIGN KEY (user_id)
    REFERENCES user_profile(user_id)
    ON UPDATE CASCADE
    ON DELETE SET NULL;

ALTER TABLE user_attributes
    ADD CONSTRAINT fk_user_attributes_user_profile
    FOREIGN KEY (user_id)
    REFERENCES user_profile(user_id)
    ON UPDATE CASCADE
    ON DELETE SET NULL;

ALTER TABLE files
    ADD CONSTRAINT fk_files_user_profile
    FOREIGN KEY (user_id)
    REFERENCES user_profile(user_id)
    ON UPDATE CASCADE
    ON DELETE SET NULL;

ALTER TABLE llm_call
    ADD CONSTRAINT fk_llm_call_user_profile
    FOREIGN KEY (user_id)
    REFERENCES user_profile(user_id)
    ON UPDATE CASCADE
    ON DELETE SET NULL;
