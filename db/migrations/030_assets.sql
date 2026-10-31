CREATE TABLE IF NOT EXISTS assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NULL REFERENCES user_profile(user_id) ON DELETE SET NULL,
    asset_type TEXT NOT NULL,
    origin TEXT NOT NULL,
    name TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT assets_type_not_blank CHECK (BTRIM(asset_type) <> ''),
    CONSTRAINT assets_origin_not_blank CHECK (BTRIM(origin) <> ''),
    CONSTRAINT assets_name_not_blank CHECK (BTRIM(name) <> '')
);

CREATE INDEX IF NOT EXISTS idx_assets_user_type ON assets(user_id, asset_type);
CREATE UNIQUE INDEX IF NOT EXISTS assets_user_type_name_key
    ON assets(user_id, asset_type, name);

CREATE TABLE IF NOT EXISTS file_assets (
    asset_id UUID PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
    storage_path TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    size_bytes BIGINT,
    checksum TEXT
);

CREATE TABLE IF NOT EXISTS http_pages (
    asset_id UUID PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    fetched_at TIMESTAMPTZ
);

INSERT INTO assets (id, user_id, asset_type, origin, name, metadata, created_at)
SELECT f.id, f.user_id, 'file', 'uploaded', f.file_name, f.metadata, f.uploaded_at
FROM files f
ON CONFLICT (id) DO NOTHING;

INSERT INTO file_assets (asset_id, storage_path, mime_type)
SELECT f.id, f.file_path, f.file_type
FROM files f
ON CONFLICT (asset_id) DO NOTHING;

CREATE TABLE IF NOT EXISTS asset_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    asset_type TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_asset_chunk UNIQUE (asset_id, chunk_index)
);

INSERT INTO asset_chunks (id, asset_id, asset_type, chunk_index, content, embedding, created_at)
SELECT c.id, c.file_id, a.asset_type, c.chunk_index, c.content, c.embedding, c.created_at
FROM file_chunks c
JOIN assets a ON a.id = c.file_id
ON CONFLICT (asset_id, chunk_index) DO NOTHING;

CREATE TABLE IF NOT EXISTS job_result_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_result_id UUID NOT NULL REFERENCES job_results(id) ON DELETE CASCADE,
    asset_id UUID NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT job_result_assets_unique_reference UNIQUE (job_result_id, asset_id)
);

INSERT INTO assets (id, user_id, asset_type, origin, name, metadata, created_at, updated_at)
SELECT id, user_id, artifact_type, 'generated', artifact_id, metadata, created_at, updated_at
FROM artifacts
ON CONFLICT (id) DO NOTHING;

INSERT INTO job_result_assets (job_result_id, asset_id, created_at)
SELECT job_result_id, artifact_id, created_at
FROM job_result_artifacts
ON CONFLICT (job_result_id, asset_id) DO NOTHING;

DROP TABLE IF EXISTS job_result_artifacts;
DROP TABLE IF EXISTS artifacts;
DROP TABLE IF EXISTS file_chunks;
DROP TABLE IF EXISTS files;

COMMENT ON COLUMN assets.user_id IS
    'NULL means globally visible; a value scopes the asset to that user.';
