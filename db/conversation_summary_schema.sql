DROP TABLE IF EXISTS conversation_summary;

CREATE TABLE conversation_summary (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
  summary TEXT NOT NULL,
  message_index_cutoff INTEGER NOT NULL, -- summarized conversation up to this message index
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);