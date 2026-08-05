# buzzG

**GCOR (Graph-Centric Orchestrated Retrieval) ingestion &amp; retrieval pipeline for <a href="https://github.com/block/buzz">Buzz</a>, running entirely on Buzz's own Postgres and MinIO.**

This is the Postgres-native successor to the proxy component in <a href="https://github.com/Abdenour7496/eedgeai">eedgeai</a> (which used Qdrant + Neo4j). Same two-phase cognitive retrieval model — semantic search over embedded chunks, then graph expansion over Memory/Belief/Inference/Concept nodes — collapsed onto one substrate so Buzz doesn't need extra databases.

---

## Why

Buzz is a self-hostable workspace where humans and AI agents share channels, threads, git events, and workflows — all as signed events in one Postgres-backed log. `buzzG` adds a memory layer on top: every message, git patch, and document that flows through Buzz gets chunked, embedded, and linked into a graph, so agents can ask "have we seen this before?" and get back reasoned context instead of just keyword hits.

- **Vector store:** `pgvector` on Buzz's Postgres — no Qdrant.
- **Graph store:** adjacency tables (`gcor.nodes` / `gcor.edges`) on the same Postgres — no Neo4j.
- **Object store:** Buzz's existing MinIO/S3 bucket — raw ingested files live there, content-addressed by SHA-256.
- **Agent surface:** an MCP server (`mcp-postgres-gcor`) exposing `semantic_search`, `graph_expand`, and `ingest_document` as tools, reachable from any Buzz agent (Goose, Codex, Claude Code) via `buzz-acp`.

---

## Architecture

```
Buzz (relay, channels, git events, workflows)
   │
   │  buzz-workflow webhook trigger (message / git_event)
   ▼
gcor-proxy  (FastAPI)
   ├─► chunk + embed
   ├─► upload raw bytes → MinIO (content-addressed, dedup by hash)
   └─► write metadata + embeddings + graph nodes/edges → Postgres (gcor schema)

Buzz agent (Goose / Codex / Claude Code)
   │  buzz-acp (ACP ↔ MCP)
   ▼
mcp-postgres-gcor  ──HTTP──►  gcor-proxy  /api/retrieve
   semantic_search              │
   graph_expand                 ├─► pgvector cosine search (chunks)
   ingest_document               └─► recursive CTE graph expansion (nodes/edges)
```

Retrieval is two-phase, same as the original GCOR pipeline:

1. **Semantic phase** — embed the query, cosine-search `gcor.chunks` via the `pgvector` HNSW index, filtered by `access_level`, `confidence`, temporal validity (`valid_from`/`valid_to`), and optionally `agent_id`.
2. **Structural phase** — from the matched chunks' graph nodes, expand N hops through `gcor.edges` (`CONTAINS`, `DERIVED_FROM`, `HOLDS`, `ABOUT`, `CONTRADICTS`, ...) to pull in related Memory/Belief/Inference/Concept nodes.
3. **Reflection check** — falls back to chunk text only if the graph expansion is empty, and to the plain LLM if both are empty.

---

## Repo layout

```
buzzG/
├── migrations/
│   └── 0001_gcor_schema.sql      ← pgvector + gcor schema (documents, chunks, nodes, edges)
├── proxy/                         ← gcor-proxy: FastAPI ingestion + retrieval service
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── mcp-postgres-gcor/              ← MCP (SSE) server: semantic_search / graph_expand / ingest_document
│   ├── server.py
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.gcor.yml        ← add-on stack, attaches to Buzz's existing network/DB/MinIO
├── .env.example
└── example-buzz-workflow.gcor-ingest.yml   ← sample webhook trigger for buzz-workflow
```

---

## Prerequisites

- A running <a href="https://github.com/block/buzz">Buzz</a> deployment (`just setup && just dev`, or the production Compose bundle in `buzz/deploy/compose/`), with Postgres and MinIO up.
- `pgvector` extension available on that Postgres image (use `pgvector/pgvector:pg16` as the base, or install the extension package, if Buzz's default Postgres image doesn't already have it).
- Docker + Docker Compose.
- An embedding provider: OpenAI API key, or a local Ollama instance.

---

## Setup

**1. Point this stack at your Buzz deployment**

```bash
cp .env.example .env
```

Edit `.env`:

| Variable | Description |
|---|---|
| `BUZZ_NETWORK` | Docker network name Buzz's Compose project created (check with `docker network ls`) |
| `POSTGRES_HOST` / `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Buzz's Postgres connection details |
| `MINIO_HOST` / `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` | Buzz's MinIO connection details |
| `EMBEDDING_BACKEND` | `openai` or `ollama` |
| `EMBEDDING_MODEL` | e.g. `text-embedding-3-small` (must match `EMBEDDING_DIMS` in the migration/schema) |
| `OPENAI_API_KEY` | required if `EMBEDDING_BACKEND=openai` |
| `INGEST_WEBHOOK_SECRET` | shared secret checked on inbound webhook calls from `buzz-workflow` |

**2. Bring up the pipeline**

```bash
docker compose -f docker-compose.gcor.yml up -d
```

This runs the migration once (`gcor-migrate`), then starts `gcor-proxy` (port `5001`) and `mcp-postgres-gcor` (port `8765`).

**3. Wire Buzz to ingest automatically**

Drop `example-buzz-workflow.gcor-ingest.yml` into Buzz's workflow directory (see `buzz/VISION_AGENT.md` / workflow docs for the exact path and trigger schema for your Buzz version) so new channel messages and NIP-34 git patch events get POSTed to `gcor-proxy`'s `/api/ingest` automatically.

**4. Register the MCP server with agents**

Point `buzz-acp` at `mcp-postgres-gcor` (`http://mcp-postgres-gcor:8765`, SSE transport) so agents in Buzz get `semantic_search`, `graph_expand`, and `ingest_document` as normal tool calls.

---

## API reference (gcor-proxy)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/ingest` | Ingest a file (`multipart/form-data`) or raw text; chunks, embeds, uploads to MinIO, writes graph nodes |
| `POST` | `/api/retrieve` | Run the two-phase GCOR pipeline for a query; returns matched chunks + expanded graph nodes |
| `GET` | `/api/collections` | List ingested documents with chunk counts |
| `GET` | `/health` | Liveness/readiness check |
| `GET` | `/metrics` | Prometheus metrics (`gcor_rag_requests_total`, `gcor_rag_duration_seconds`, `gcor_ingest_total`, etc.) |

---

## Manual ingest / query examples

```bash
# Ingest a file
curl -X POST http://localhost:5001/api/ingest \
  -F "file=@report.pdf" \
  -F "title=Q4 Report" \
  -F "access_level=public"

# Retrieve
curl -X POST http://localhost:5001/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "what did we decide about the release cadence?", "top_k": 8}'
```

---

## Relationship to eedgeai

`buzzG` reuses the cognitive data model from <a href="https://github.com/Abdenour7496/eedgeai">eedgeai</a> (confidence scores, temporal validity windows, agent-scoped ACL, the `Document/Chunk/Memory/Inference/Belief/Goal/Event/Concept` node vocabulary) but replaces the Qdrant + Neo4j backend with a single Postgres schema (`gcor.*`, using `pgvector` + recursive CTEs) and routes raw object storage through Buzz's own MinIO instead of a dedicated ingest volume.

## License

Match Buzz's license (Apache-2.0) unless you have a reason to diverge.
