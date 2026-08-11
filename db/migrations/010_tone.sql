CREATE TABLE IF NOT EXISTS tone (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL REFERENCES user_profile(user_id) ON UPDATE CASCADE ON DELETE CASCADE,
    conversation_id UUID NULL REFERENCES conversation(id) ON UPDATE CASCADE ON DELETE CASCADE,
    tone_type TEXT NOT NULL,
    verbosity TEXT NULL,
    formality TEXT NULL,
    directness TEXT NULL,
    humor TEXT NULL,
    technical_depth TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_tone_type
        CHECK (tone_type IN ('profile', 'conversation')),
    CONSTRAINT chk_tone_scope
        CHECK (
            (tone_type = 'profile' AND conversation_id IS NULL)
            OR (tone_type = 'conversation' AND conversation_id IS NOT NULL)
        )
);

CREATE INDEX IF NOT EXISTS idx_tone_user_id
    ON tone(user_id);

CREATE INDEX IF NOT EXISTS idx_tone_conversation_id
    ON tone(conversation_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_tone_profile_unique
    ON tone(user_id)
    WHERE tone_type = 'profile' AND conversation_id IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_tone_conversation_unique
    ON tone(user_id, conversation_id)
    WHERE tone_type = 'conversation' AND conversation_id IS NOT NULL;
