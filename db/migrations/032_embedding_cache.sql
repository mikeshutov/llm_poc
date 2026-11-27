CREATE TABLE IF NOT EXISTS embedding_cache_entries (
    model TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    embedding DOUBLE PRECISION[] NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model, input_hash),
    CONSTRAINT embedding_cache_entries_vector_valid CHECK (
        array_ndims(embedding) = 1
        AND cardinality(embedding) > 0
        AND array_position(embedding, NULL) IS NULL
    )
);
