# EedgeAI

EedgeAI is a local-first agent and retrieval stack. Graphiti provides temporal knowledge extraction and hybrid retrieval, FalkorDB stores the graph and vectors, OpenClaw provides agent tools, Open WebUI provides chat, and SeaweedFS (S3-compatible) preserves original documents.

## Current architecture

```text
Open WebUI / Knowledge UI / OpenClaw
                  |
             RAG proxy
                  |
              Graphiti
        (episodes, entities, facts)
                  |
              FalkorDB
          (graph + vectors)
```

Neo4j, Qdrant, and the Buzz collaboration stack are no longer active services. Their old Docker volumes are intentionally retained by name for rollback/export; the new services do not mount or modify them.

## Feasibility and trade-offs

Graphiti officially supports FalkorDB and combines graph traversal, semantic similarity, keyword search, temporal validity, and episode provenance in one retrieval layer. This simplifies the former dual-write Qdrant/Neo4j path and removes cross-database consistency failures.

The material constraint is extraction quality: Graphiti requires an LLM that reliably returns structured JSON. The default local model is `qwen2.5:7b`; smaller local models were not reliable enough in validation. Allocate at least 12 GB RAM to Docker (16 GB recommended). For better extraction quality, point `GRAPHITI_API_BASE_URL`, `GRAPHITI_API_KEY`, and `GRAPHITI_MODEL` at an OpenAI-compatible hosted model.

This cutover does not automatically transform legacy Neo4j/Qdrant data. Re-ingest source documents to populate Graphiti. Legacy volumes are retained until migration has been accepted.

## Start

```bash
cp .env.example .env
docker compose -f docker-compose.unified.yml up -d --build --remove-orphans
```

The first start downloads Qwen 2.5 7B, Nomic Embed Text, Llama 3.2, and LLaVA, so it can take several minutes.

## Services

| Service | URL / port | Purpose |
|---|---|---|
| Knowledge UI / proxy | http://localhost:5001/knowledge | ingest, browse, search, RAG API |
| Open WebUI | http://localhost:8080 | chat UI |
| Graphiti REST | http://localhost:8000/docs | knowledge ingestion and retrieval |
| FalkorDB Browser | http://localhost:3001 | graph inspection |
| FalkorDB | localhost:6379 | graph/vector database |
| SeaweedFS S3 API | http://localhost:9000 | original documents (S3) |
| SeaweedFS admin UI | http://localhost:9001 | storage admin (set `SEAWEEDFS_ADMIN_PASSWORD`) |
| OpenClaw | http://localhost:18799 | agent gateway |
| Prometheus | http://localhost:9090 | metrics |
| Grafana | http://localhost:3000 | dashboards |

## Knowledge API

```bash
curl -F "file=@report.pdf" -F "collection=documents" http://localhost:5001/api/ingest
curl "http://localhost:5001/api/search?q=your+question&collection=documents"
```

The ingest watcher also processes objects placed under `documents/inbox/`.

## Operations

```bash
docker compose -f docker-compose.unified.yml ps
docker compose -f docker-compose.unified.yml logs -f graphiti graphiti-mcp falkordb proxy
curl http://localhost:8000/healthcheck
docker exec aiedge-falkordb-1 redis-cli PING
```

Graphiti groups replace Qdrant collections. A group is materialized on first ingest. Renaming groups is intentionally unsupported; deleting a group deletes its Graphiti episodes and facts.

## Configuration

See `.env.example`. The important settings are `GRAPHITI_API_BASE_URL`, `GRAPHITI_MODEL`, `GRAPHITI_EMBEDDING_MODEL`, `GRAPHITI_EMBEDDING_DIM`, and `GRAPHITI_GROUP_ID`. Do not commit `.env` or credentials.

## Upgrading from the MinIO-based stack

MinIO stopped publishing container images in October 2025 and its repository was
archived in April 2026, so document storage now runs on SeaweedFS. Existing
documents live in the old `docs_minio_data` volume and are copied across once:

```bash
# 1. Back up the Open WebUI and FalkorDB volumes (v0.13 runs DB migrations).
# 2. Start the new stack, then copy the old objects into SeaweedFS
#    (needs a MinIO image still cached locally; set LEGACY_MINIO_IMAGE if needed):
docker compose -f docker-compose.unified.yml --profile migrate run --rm storage-migrate
docker compose -f docker-compose.unified.yml --profile migrate rm -sf minio-legacy
```

The copy is idempotent (re-running skips objects already copied) and never
modifies the source volume. Remove `docs_minio_data` only after verifying.

FalkorDB stays on 4.x for now: 6.x has no supported upgrade path for 4.x data
until 6.2.
