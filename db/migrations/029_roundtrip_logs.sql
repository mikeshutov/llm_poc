CREATE TABLE IF NOT EXISTS node_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  scope_type TEXT NOT NULL,
  scope_id UUID NOT NULL,
  sequence INTEGER NOT NULL,
  node_name TEXT NOT NULL,
  stage TEXT NOT NULL DEFAULT '',
  event_type TEXT NOT NULL,
  level TEXT NOT NULL DEFAULT 'info' CHECK (level IN ('debug', 'info', 'warning', 'error')),
  message TEXT NOT NULL DEFAULT '',
  payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  duration_ms INTEGER NULL,
  correlation_id UUID NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_node_log_sequence UNIQUE (scope_type, scope_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_node_log_scope_sequence
  ON node_log(scope_type, scope_id, sequence);
CREATE INDEX IF NOT EXISTS idx_node_log_scope_created
  ON node_log(scope_type, scope_id, created_at);
CREATE INDEX IF NOT EXISTS idx_node_log_event_type
  ON node_log(scope_type, scope_id, event_type);
