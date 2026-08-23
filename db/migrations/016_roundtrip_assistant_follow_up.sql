ALTER TABLE conversation_roundtrip
ADD COLUMN IF NOT EXISTS assistant_follow_up TEXT NOT NULL DEFAULT '';

UPDATE conversation_roundtrip
SET assistant_follow_up = COALESCE(response_payload ->> 'next_question', '')
WHERE BTRIM(assistant_follow_up) = ''
  AND BTRIM(COALESCE(response_payload ->> 'next_question', '')) <> '';
