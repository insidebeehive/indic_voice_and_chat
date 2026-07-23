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
    tenant_id TEXT,         -- set = tenant-scoped chunk; NULL = not tenant-scoped
    crm_id    TEXT          -- set = CRM-scoped chunk; NULL = not CRM-scoped
                             -- (exactly one of tenant_id/crm_id is set per row)
);

-- 2b. If this table already existed from before the CRM-level KB change
-- (docs/superpowers/specs/2026-07-23-crm-kb-design.md), add the new column
-- here — safe to re-run, a no-op if the column already exists:
ALTER TABLE voicebot.knowledge_chunks ADD COLUMN IF NOT EXISTS crm_id TEXT;

-- 3. Indexes
CREATE INDEX IF NOT EXISTS knowledge_chunks_tenant_idx
    ON voicebot.knowledge_chunks (tenant_id);
CREATE INDEX IF NOT EXISTS knowledge_chunks_crm_idx
    ON voicebot.knowledge_chunks (crm_id);

-- HNSW requires pgvector >= 0.5 (Northflank managed Postgres ships with it)
CREATE INDEX IF NOT EXISTS knowledge_chunks_emb_hnsw
    ON voicebot.knowledge_chunks USING hnsw (embedding vector_cosine_ops);

-- 4. App user grants
GRANT SELECT, INSERT, UPDATE, DELETE ON voicebot.knowledge_chunks TO <app_user>;

-- 5. One-time backfill for any pre-existing platform-wide rows (tenant_id IS
-- NULL) into the CRM that now owns them (see alembic/versions/
-- 0010_crm_kb_documents.py — this migration attempts this same UPDATE
-- automatically if this table/column already exist at migration time; this
-- line is here as a manual fallback if that automatic attempt was skipped,
-- e.g. because this SQL file was applied AFTER the migration ran). Replace
-- <crm_id> with the actual CRM id (e.g. 'betstudio'):
-- UPDATE voicebot.knowledge_chunks SET crm_id = '<crm_id>'
--   WHERE tenant_id IS NULL AND crm_id IS NULL;
