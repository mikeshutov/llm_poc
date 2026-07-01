DROP TABLE IF EXISTS conversation;

CREATE TABLE conversation (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NULL,
  -- user_id may not be used for now but you probably want to have conversations for users
  user_id TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  tone_state JSONB NOT NULL DEFAULT '{}'::jsonb
);
