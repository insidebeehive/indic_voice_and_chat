-- Run this ONCE as the postgres superuser before starting the app with
-- vector_store.provider = pgvector.
--
-- Replace <app_user> with the username from DATABASE_URL.

-- 1. Extension (requires superuser)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Table (in the voicebot schema, same as all other app tables)
CREATE TABLE IF NOT EXISTS voicebot.knowledge_chunks (
    id        TEXT  PRIMARY KEY,
    content   TEXT  NOT NULL,
    metadata  JSONB NOT NULL DEFAULT '{}',
    embedding vector(384),
    tenant_id TEXT          -- NULL = platform / global docs
);

-- 3. Indexes
CREATE INDEX IF NOT EXISTS knowledge_chunks_tenant_idx
    ON voicebot.knowledge_chunks (tenant_id);

-- HNSW requires pgvector >= 0.5 (Northflank managed Postgres ships with it)
CREATE INDEX IF NOT EXISTS knowledge_chunks_emb_hnsw
    ON voicebot.knowledge_chunks USING hnsw (embedding vector_cosine_ops);

-- 4. App user grants
GRANT SELECT, INSERT, UPDATE, DELETE ON voicebot.knowledge_chunks TO <app_user>;
