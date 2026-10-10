-- GCOR schema for buzzG
-- Requires: pgvector extension (pgvector/pgvector:pg16 image or manual install)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE SCHEMA IF NOT EXISTS gcor;

-- ── Documents ────────────────────────────────────────────────────────────────
-- One row per ingested file / message / git-patch.
CREATE TABLE IF NOT EXISTS gcor.documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title           TEXT NOT NULL,
    source_type     TEXT NOT NULL DEFAULT 'file',   -- file | message | git_event
    source_ref      TEXT,                            -- buzz event id, git sha, etc.
    sha256          TEXT NOT NULL,                   -- content hash for dedup
    minio_key       TEXT NOT NULL,                   -- object path in MinIO
    access_level    TEXT NOT NULL DEFAULT 'private', -- public | team | private
    agent_id        TEXT,                            -- owning agent (nullable = global)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_from      TIMESTAMPTZ,
    valid_to        TIMESTAMPTZ,
    metadata        JSONB NOT NULL DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS documents_sha256_idx ON gcor.documents (sha256);
CREATE INDEX IF NOT EXISTS documents_agent_idx       ON gcor.documents (agent_id);
CREATE INDEX IF NOT EXISTS documents_access_idx      ON gcor.documents (access_level);

-- ── Chunks ───────────────────────────────────────────────────────────────────
-- Fixed-size text windows produced by the chunking step.
CREATE TABLE IF NOT EXISTS gcor.chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES gcor.documents(id) ON DELETE CASCADE,
    chunk_index     INT  NOT NULL,
    content         TEXT NOT NULL,
    embedding       VECTOR(1536),                    -- dim must match EMBEDDING_DIMS env var
    access_level    TEXT NOT NULL DEFAULT 'private',
    agent_id        TEXT,
    confidence      FLOAT NOT NULL DEFAULT 1.0,
    valid_from      TIMESTAMPTZ,
    valid_to        TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW index for approximate nearest-neighbour cosine search
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON gcor.chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_document_idx   ON gcor.chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_agent_idx      ON gcor.chunks (agent_id);
CREATE INDEX IF NOT EXISTS chunks_access_idx     ON gcor.chunks (access_level);

-- ── Graph nodes ──────────────────────────────────────────────────────────────
-- Node types: Document | Chunk | Memory | Belief | Inference | Goal | Event | Concept
CREATE TABLE IF NOT EXISTS gcor.nodes (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_type       TEXT NOT NULL,                   -- see vocabulary above
    label           TEXT NOT NULL,
    properties      JSONB NOT NULL DEFAULT '{}',
    agent_id        TEXT,
    confidence      FLOAT NOT NULL DEFAULT 1.0,
    access_level    TEXT NOT NULL DEFAULT 'private',
    valid_from      TIMESTAMPTZ,
    valid_to        TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS nodes_type_idx   ON gcor.nodes (node_type);
CREATE INDEX IF NOT EXISTS nodes_agent_idx  ON gcor.nodes (agent_id);
CREATE INDEX IF NOT EXISTS nodes_props_gin  ON gcor.nodes USING gin(properties);

-- ── Graph edges ──────────────────────────────────────────────────────────────
-- Relation types: CONTAINS | DERIVED_FROM | HOLDS | ABOUT | CONTRADICTS |
--                 SUPPORTS | RELATES_TO | PRECEDES | FOLLOWS
CREATE TABLE IF NOT EXISTS gcor.edges (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id       UUID NOT NULL REFERENCES gcor.nodes(id) ON DELETE CASCADE,
    target_id       UUID NOT NULL REFERENCES gcor.nodes(id) ON DELETE CASCADE,
    relation        TEXT NOT NULL,
    weight          FLOAT NOT NULL DEFAULT 1.0,
    properties      JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS edges_source_idx    ON gcor.edges (source_id);
CREATE INDEX IF NOT EXISTS edges_target_idx    ON gcor.edges (target_id);
CREATE INDEX IF NOT EXISTS edges_relation_idx  ON gcor.edges (relation);

-- ── Chunk ↔ node join ────────────────────────────────────────────────────────
-- Links a chunk to the graph node that represents it.
CREATE TABLE IF NOT EXISTS gcor.chunk_nodes (
    chunk_id   UUID NOT NULL REFERENCES gcor.chunks(id) ON DELETE CASCADE,
    node_id    UUID NOT NULL REFERENCES gcor.nodes(id)  ON DELETE CASCADE,
    PRIMARY KEY (chunk_id, node_id)
);
