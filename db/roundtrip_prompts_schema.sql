DROP TABLE IF EXISTS roundtrip_prompts;

CREATE TABLE roundtrip_prompts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  roundtrip_id UUID NOT NULL REFERENCES conversation_roundtrip(id) ON DELETE CASCADE,
  agent TEXT NOT NULL,
  prompt_step TEXT NOT NULL,
  prompt TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_roundtrip_prompts_roundtrip_id
  ON roundtrip_prompts(roundtrip_id);

CREATE INDEX idx_roundtrip_prompts_roundtrip_agent_step
  ON roundtrip_prompts(roundtrip_id, agent, prompt_step);
