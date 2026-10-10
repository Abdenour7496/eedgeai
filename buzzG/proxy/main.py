"""
gcor-proxy  –  FastAPI ingestion & retrieval service for buzzG.

Endpoints
---------
POST /api/ingest          ingest file (multipart) or raw text
POST /api/retrieve        two-phase GCOR retrieval
GET  /api/collections     list documents with chunk counts
GET  /health              liveness
GET  /metrics             Prometheus metrics
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import asyncpg
import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from minio import Minio
from pgvector.asyncpg import register_vector
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel
from pydantic_settings import BaseSettings


# ── Settings ──────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "buzz"
    postgres_password: str = "buzz"
    postgres_db: str = "buzz"

    minio_host: str = "localhost:9000"
    minio_root_user: str = "minioadmin"
    minio_root_password: str = "minioadmin"
    minio_bucket: str = "gcor-objects"
    minio_secure: bool = False

    embedding_backend: str = "openai"   # openai | ollama
    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    chunk_size: int = 512               # tokens per chunk
    chunk_overlap: int = 64

    ingest_webhook_secret: str = ""

    class Config:
        env_file = ".env"


cfg = Settings()

# ── Prometheus metrics ────────────────────────────────────────────────────────

INGEST_TOTAL = Counter("gcor_ingest_total", "Total ingest calls", ["status"])
RAG_TOTAL    = Counter("gcor_rag_requests_total", "Total retrieve calls", ["status"])
RAG_LATENCY  = Histogram("gcor_rag_duration_seconds", "Retrieve latency")

# ── Lifespan ──────────────────────────────────────────────────────────────────

_pool: asyncpg.Pool | None = None
_minio: Minio | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool, _minio
    _pool = await asyncpg.create_pool(
        host=cfg.postgres_host,
        port=cfg.postgres_port,
        user=cfg.postgres_user,
        password=cfg.postgres_password,
        database=cfg.postgres_db,
        init=register_vector,
    )
    _minio = Minio(
        cfg.minio_host,
        access_key=cfg.minio_root_user,
        secret_key=cfg.minio_root_password,
        secure=cfg.minio_secure,
    )
    if not _minio.bucket_exists(cfg.minio_bucket):
        _minio.make_bucket(cfg.minio_bucket)
    yield
    if _pool:
        await _pool.close()


app = FastAPI(title="gcor-proxy", lifespan=lifespan)

# ── Embedding ─────────────────────────────────────────────────────────────────

async def embed(texts: list[str]) -> list[list[float]]:
    if cfg.embedding_backend == "openai":
        import openai
        client = openai.AsyncOpenAI(api_key=cfg.openai_api_key)
        resp = await client.embeddings.create(model=cfg.embedding_model, input=texts)
        return [d.embedding for d in resp.data]
    # ollama
    async with httpx.AsyncClient(base_url=cfg.ollama_base_url, timeout=60) as c:
        results = []
        for t in texts:
            r = await c.post("/api/embeddings", json={"model": cfg.embedding_model, "prompt": t})
            r.raise_for_status()
            results.append(r.json()["embedding"])
        return results

# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = cfg.chunk_size, overlap: int = cfg.chunk_overlap) -> list[str]:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + size, len(tokens))
            chunks.append(enc.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start += size - overlap
        return chunks
    except Exception:
        # Fallback: character-based split
        words = text.split()
        chunks, buf = [], []
        for w in words:
            buf.append(w)
            if len(buf) >= size:
                chunks.append(" ".join(buf))
                buf = buf[-overlap:]
        if buf:
            chunks.append(" ".join(buf))
        return chunks

# ── Ingest ────────────────────────────────────────────────────────────────────

@app.post("/api/ingest")
async def ingest(
    request: Request,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    title: str = Form("untitled"),
    access_level: str = Form("private"),
    agent_id: Optional[str] = Form(None),
    source_type: str = Form("file"),
    source_ref: Optional[str] = Form(None),
):
    # Optional webhook secret check
    if cfg.ingest_webhook_secret:
        secret = request.headers.get("X-Webhook-Secret", "")
        if secret != cfg.ingest_webhook_secret:
            raise HTTPException(status_code=401, detail="invalid webhook secret")

    try:
        if file is not None:
            raw = await file.read()
        elif text is not None:
            raw = text.encode()
        else:
            raise HTTPException(status_code=422, detail="provide file or text")

        sha = hashlib.sha256(raw).hexdigest()
        minio_key = f"gcor/{sha[:2]}/{sha}"

        # Dedup: skip if already ingested
        async with _pool.acquire() as conn:
            existing = await conn.fetchval(
                "SELECT id FROM gcor.documents WHERE sha256 = $1", sha
            )
        if existing:
            INGEST_TOTAL.labels(status="dedup").inc()
            return {"status": "dedup", "document_id": str(existing)}

        # Upload to MinIO
        _minio.put_object(
            cfg.minio_bucket,
            minio_key,
            io.BytesIO(raw),
            length=len(raw),
        )

        # Chunk + embed
        body = raw.decode(errors="replace")
        chunks = chunk_text(body)
        embeddings = await embed(chunks)

        async with _pool.acquire() as conn:
            async with conn.transaction():
                doc_id = await conn.fetchval(
                    """
                    INSERT INTO gcor.documents
                        (title, source_type, source_ref, sha256, minio_key, access_level, agent_id)
                    VALUES ($1,$2,$3,$4,$5,$6,$7)
                    RETURNING id
                    """,
                    title, source_type, source_ref, sha, minio_key, access_level, agent_id,
                )

                # Create a Document graph node
                doc_node_id = await conn.fetchval(
                    """
                    INSERT INTO gcor.nodes (node_type, label, properties, agent_id, access_level)
                    VALUES ('Document', $1, $2::jsonb, $3, $4)
                    RETURNING id
                    """,
                    title,
                    json.dumps({"document_id": str(doc_id), "source_type": source_type}),
                    agent_id, access_level,
                )

                for i, (chunk_text_val, emb) in enumerate(zip(chunks, embeddings)):
                    chunk_id = await conn.fetchval(
                        """
                        INSERT INTO gcor.chunks
                            (document_id, chunk_index, content, embedding, access_level, agent_id)
                        VALUES ($1,$2,$3,$4,$5,$6)
                        RETURNING id
                        """,
                        doc_id, i, chunk_text_val, emb, access_level, agent_id,
                    )

                    # Chunk graph node
                    chunk_node_id = await conn.fetchval(
                        """
                        INSERT INTO gcor.nodes (node_type, label, properties, agent_id, access_level)
                        VALUES ('Chunk', $1, $2::jsonb, $3, $4)
                        RETURNING id
                        """,
                        f"{title}[{i}]",
                        json.dumps({"chunk_id": str(chunk_id), "chunk_index": i}),
                        agent_id, access_level,
                    )

                    # chunk_nodes join
                    await conn.execute(
                        "INSERT INTO gcor.chunk_nodes (chunk_id, node_id) VALUES ($1,$2)",
                        chunk_id, chunk_node_id,
                    )

                    # CONTAINS edge: Document → Chunk
                    await conn.execute(
                        """
                        INSERT INTO gcor.edges (source_id, target_id, relation)
                        VALUES ($1,$2,'CONTAINS')
                        """,
                        doc_node_id, chunk_node_id,
                    )

        INGEST_TOTAL.labels(status="ok").inc()
        return {"status": "ok", "document_id": str(doc_id), "chunks": len(chunks)}

    except HTTPException:
        INGEST_TOTAL.labels(status="error").inc()
        raise
    except Exception as exc:
        INGEST_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Retrieve ──────────────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 8
    hops: int = 2
    access_level: Optional[str] = None
    agent_id: Optional[str] = None
    min_confidence: float = 0.0


@app.post("/api/retrieve")
async def retrieve(req: RetrieveRequest):
    t0 = time.perf_counter()
    try:
        [q_emb] = await embed([req.query])

        async with _pool.acquire() as conn:
            # ── Phase 1: semantic search ───────────────────────────────────────
            filters = ["c.valid_to IS NULL OR c.valid_to > now()"]
            params: list[Any] = [q_emb, req.top_k]
            p = 3
            if req.access_level:
                filters.append(f"c.access_level = ${p}"); params.append(req.access_level); p += 1
            if req.agent_id:
                filters.append(f"(c.agent_id IS NULL OR c.agent_id = ${p})"); params.append(req.agent_id); p += 1
            if req.min_confidence > 0:
                filters.append(f"c.confidence >= ${p}"); params.append(req.min_confidence); p += 1

            where = " AND ".join(filters)
            rows = await conn.fetch(
                f"""
                SELECT c.id, c.content, c.document_id,
                       1 - (c.embedding <=> $1) AS score
                FROM gcor.chunks c
                WHERE {where}
                ORDER BY c.embedding <=> $1
                LIMIT $2
                """,
                *params,
            )
            chunks = [dict(r) for r in rows]

            # ── Phase 2: graph expansion ───────────────────────────────────────
            chunk_ids = [r["id"] for r in rows]
            graph_nodes: list[dict] = []

            if chunk_ids:
                seed_node_rows = await conn.fetch(
                    "SELECT node_id FROM gcor.chunk_nodes WHERE chunk_id = ANY($1::uuid[])",
                    chunk_ids,
                )
                seed_ids = [r["node_id"] for r in seed_node_rows]

                if seed_ids:
                    hops_graph = await conn.fetch(
                        f"""
                        WITH RECURSIVE expand(id, depth) AS (
                            SELECT id, 0 FROM gcor.nodes WHERE id = ANY($1::uuid[])
                            UNION
                            SELECT e.target_id, x.depth + 1
                            FROM gcor.edges e
                            JOIN expand x ON x.id = e.source_id
                            WHERE x.depth < $2
                        )
                        SELECT DISTINCT n.id, n.node_type, n.label, n.properties
                        FROM expand x
                        JOIN gcor.nodes n ON n.id = x.id
                        ORDER BY n.node_type, n.label
                        """,
                        seed_ids, req.hops,
                    )
                    graph_nodes = [dict(r) for r in hops_graph]

        RAG_TOTAL.labels(status="ok").inc()
        RAG_LATENCY.observe(time.perf_counter() - t0)

        # Reflection check
        if not graph_nodes and not chunks:
            return {"status": "empty", "chunks": [], "graph_nodes": []}

        return {
            "status": "ok",
            "chunks": [
                {"id": str(c["id"]), "content": c["content"],
                 "document_id": str(c["document_id"]), "score": c["score"]}
                for c in chunks
            ],
            "graph_nodes": [
                {"id": str(g["id"]), "type": g["node_type"],
                 "label": g["label"], "properties": g["properties"]}
                for g in graph_nodes
            ],
        }

    except Exception as exc:
        RAG_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Collections ───────────────────────────────────────────────────────────────

@app.get("/api/collections")
async def collections():
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.id, d.title, d.source_type, d.created_at,
                   COUNT(c.id) AS chunk_count
            FROM gcor.documents d
            LEFT JOIN gcor.chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.created_at DESC
            """
        )
    return [
        {
            "id": str(r["id"]),
            "title": r["title"],
            "source_type": r["source_type"],
            "created_at": r["created_at"].isoformat(),
            "chunk_count": r["chunk_count"],
        }
        for r in rows
    ]


# ── Health & metrics ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        async with _pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"status": "ok"}
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(exc)})


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
