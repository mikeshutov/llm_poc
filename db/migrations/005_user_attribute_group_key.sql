ALTER TABLE user_attributes
    ADD COLUMN IF NOT EXISTS group_key text NULL;

ALTER TABLE user_attributes
    DROP COLUMN IF EXISTS source_conversation_id,
    DROP COLUMN IF EXISTS source_roundtrip_id;
