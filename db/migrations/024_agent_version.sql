ALTER TABLE agents
ADD COLUMN version INTEGER NOT NULL DEFAULT 1;

ALTER TABLE agents
ADD CONSTRAINT agents_version_positive CHECK (version > 0);
