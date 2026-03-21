# EedgeAI — Cognitive AI Stack

A production-ready AI agent stack built on **GCOR** (Graph-Centric Orchestrated Retrieval): Neo4j as the cognitive backbone, Qdrant as the semantic perception layer, OpenClaw as the agentic interface, and OpenWebUI as the chat frontend — all wired together with a cognitive RAG proxy and full observability stack.

---

## Architecture

```
User
 ├─► Knowledge UI  (localhost:5001)     ← document ingest, search, browse, collection management
 ├─► Open WebUI    (localhost:8080)     ← chat interface
 │     └─► GCOR Proxy (port 5001)
 │           ├─► Qdrant  (port 6333)   ← semantic search  (vector perception)
 │           ├─► Neo4j   (port 7687)   ← knowledge graph  (cognitive backbone)
 │           └─► LLM API (OpenAI / Anthropic)
 │
 └─► OpenClaw Agent (localhost:18799)  ← agentic interface with tools
       ├─► mcp-qdrant (port 8765)  ──► Qdrant
       └─► mcp-neo4j  (port 8766)  ──► Neo4j

Monitoring
 ├─► Prometheus (localhost:9090)   ← scrapes proxy + qdrant metrics every 15s
 └─► Grafana    (localhost:3000)   ← auto-provisioned GCOR observability dashboard
```

### GCOR Retrieval Pipeline (every chat message via OpenWebUI)

1. **Intent classification** — keyword-based: `factual | planning | dependency | memory | semantic | inference | belief`
2. **Semantic phase** — embed query → top-K Qdrant hits, filtered by confidence, temporal validity, and access level
3. **Structural phase** — intent-specific Neo4j Cypher expansion using the `neo4j_element_id` from each Qdrant hit
4. **Reflection check** — fallback to chunk text when graph is empty; pure LLM when both are empty
5. **Context injection** — structured system message with confidence scores, temporal badges, reasoning traces
6. **LLM call** — forwarded to OpenAI or Anthropic with enriched context

### Cognitive Infrastructure

Every node in Neo4j and every vector point in Qdrant carries:

| Property | Purpose |
|---|---|
| `confidence` | Float 0.0–1.0 — certainty score, filterable at query time |
| `valid_from` / `valid_to` | ISO-8601 temporal validity window — expired knowledge is excluded |
| `agent_id` | Agent partition — scopes memory/belief/inference to one agent |
| `access_level` | ACL: `public` \| `restricted` \| `agent:<id>` |

Cognitive node types supported in Neo4j:

| Label | Description |
|---|---|
| `:Document` | Ingested source document |
| `:Chunk` | Text segment of a Document, linked via `CONTAINS` |
| `:Memory` | Agent observations, reflections, plans, facts |
| `:Inference` | Reasoned conclusion with `reasoning_trace` and `DERIVED_FROM` sources |
| `:Belief` | Agent epistemic state with `HOLDS`, `ABOUT`, and `CONTRADICTS` relationships |
| `:Goal` | Agent objectives |
| `:Event` | Timestamped occurrences |
| `:Concept` | Named concepts linked to memories and beliefs |
| `:ArchivedCollection` | Archive metadata for deleted Qdrant collections (used by restore) |

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- At least **8 GB RAM** allocated to Docker
- Ports `3000, 5001, 6333, 7474, 7687, 8080, 9090, 18799, 18801` free

---

## Step 1 — Configure Environment Variables

Copy `.env.example` and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key — used for LLM calls and embeddings |
| `ANTHROPIC_API_KEY` | Anthropic API key — optional fallback LLM |
| `OPENCLAW_GATEWAY_TOKEN` | Token for OpenClaw gateway authentication |
| `OPENCLAW_GATEWAY_PASSWORD` | Password for OpenClaw gateway |

Optional cognitive knobs:

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `openai` | Primary LLM: `openai` or `anthropic` |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | OpenAI model for chat completions |
| `ANTHROPIC_CHAT_MODEL` | `claude-sonnet-4-6` | Anthropic model for chat completions |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `QDRANT_COLLECTION` | `documents` | Default Qdrant collection name |
| `QDRANT_TOP_K` | `8` | Number of semantic search results |
| `ENABLE_RAG` | `true` | Set `false` to disable GCOR and use plain LLM |
| `CONFIDENCE_THRESHOLD` | `0.0` | Drop knowledge below this confidence (0.0–1.0) |
| `AGENT_ID` | _(empty)_ | Scope retrieval to a specific agent partition |
| `DEFAULT_ACCESS_LEVEL` | `public` | Default ACL for new nodes |

> **Security:** Never commit `.env` to version control.

---

## Step 2 — Project Structure

```
eedgeai/
├── docker-compose.unified.yml
├── .env.example
├── openclaw-config/
│   └── openclaw.json              ← OpenClaw config (gateway mode + default model)
├── openclaw/
│   ├── Dockerfile                 ← extends openclaw base image
│   ├── package.json               ← neo4j-driver, qdrant client, pdf-parse, mammoth
│   ├── neo4j.js                   ← neo4j-cli  (shell tool for OpenClaw agent)
│   ├── qdrant.js                  ← qdrant-cli (shell tool for OpenClaw agent)
│   └── ingest.js                  ← ingest-cli (document ingestion tool)
├── openclaw-relay/
│   └── relay.js                   ← TCP relay: 18799→18789, 18801→18790
├── mcp-servers/
│   ├── qdrant/
│   │   ├── Dockerfile
│   │   └── server.py              ← GCOR-aware MCP server for Qdrant (SSE :8765)
│   └── neo4j/
│       ├── Dockerfile
│       └── server.py              ← Cognitive MCP server for Neo4j (SSE :8766)
├── proxy/
│   ├── Dockerfile
│   ├── main.py                    ← GCOR proxy + Knowledge UI + ingest API + metrics
│   ├── requirements.txt
│   └── templates/
│       └── knowledge.html         ← Knowledge management UI
├── monitoring/
│   ├── prometheus.yml             ← scrape config (proxy + qdrant, 15s interval)
│   ├── grafana-dashboard.json     ← auto-provisioned GCOR observability dashboard
│   └── grafana-provisioning/
│       ├── datasources/
│       │   └── prometheus.yaml    ← Prometheus datasource (uid: prometheus)
│       └── dashboards/
│           └── default.yaml       ← dashboard file provider config
└── agentic-stack/
    └── agentic_stack/
        └── graph/
            └── schema.py          ← Neo4j constraints and indexes
```

---

## Step 3 — Start the Stack

```bash
docker compose -f docker-compose.unified.yml up -d
```

First run pulls all images and builds custom containers (~5–10 min). Services start in dependency order: Neo4j and Qdrant → MCP servers → OpenClaw → Proxy → OpenWebUI.

Watch logs:
```bash
docker compose -f docker-compose.unified.yml logs -f
```

---

## Step 4 — Verify All Services

```bash
docker compose -f docker-compose.unified.yml ps
```

| Service | Port(s) | Description |
|---|---|---|
| `openwebui` | 8080 | Chat UI — select **openclaw** model |
| `proxy` | 5001 | GCOR pipeline + Knowledge UI + `/metrics` |
| `openclaw` | 18799 | OpenClaw agent web interface |
| `openclaw-relay` | (shared) | TCP relay (18799→18789, 18801→18790) |
| `mcp-qdrant` | 8765 (internal) | MCP server — Qdrant tools for OpenClaw |
| `mcp-neo4j` | 8766 (internal) | MCP server — Neo4j tools for OpenClaw |
| `neo4j` | 7474, 7687 | Graph database |
| `qdrant` | 6333 | Vector database + `/metrics` |
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3000 | Observability dashboard |

---

## Step 5 — Ingest Documents

### Via the Knowledge UI (recommended)

Open **http://localhost:5001** in your browser. It redirects to the Knowledge page:

- **Cards** show each Qdrant collection with document count, chunk count, and recent files
- **New Collection** button — create an empty Qdrant collection with custom name and vector size
- **Archives** button — browse and restore previously deleted collections
- **Ingest button** (per card) — drag-and-drop or click to upload; supports `.txt` `.md` `.pdf` `.docx` `.json` `.csv`
- **Test button** — run a semantic search query against the collection
- **View button** — browse all ingested documents with chunk counts and timestamps
- **Rename** (✏ icon) — rename a collection (scroll-copies all vectors to the new name)
- **Delete** (🗑 icon / Delete button) — archive a collection: copies all Qdrant vectors to `_archived_<name>_<ts>` and marks Neo4j `Document` nodes as `archived=true`; the data is never destroyed and can be restored

Each upload:
1. Extracts text from the file
2. Chunks it into ~2000-character segments with overlap
3. Creates a `:Document` node → `:Chunk` nodes in Neo4j (linked via `CONTAINS`)
4. Embeds each chunk and upserts it to Qdrant with the Neo4j `elementId` as `neo4j_element_id`

### Collection Management API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/collections` | List all collections with doc/vector stats |
| `POST` | `/api/collections` | Create a new empty collection |
| `PATCH` | `/api/collections/{name}` | Rename a collection (scroll-copy + delete) |
| `DELETE` | `/api/collections/{name}` | Archive a collection (preserves all data) |
| `GET` | `/api/archives` | List all archived collections |
| `POST` | `/api/archives/{archive_name}/restore` | Restore an archive (optionally under a new name) |

### Via OpenClaw agent (ingest-cli)

Inside the OpenClaw chat, ask the agent to ingest a file using the built-in `ingest-cli` tool:

```bash
# Ingest a file from the workspace
ingest-cli /path/to/document.pdf --title "Q4 Report"

# Ingest a Word document
ingest-cli /path/to/spec.docx --title "Technical Spec"

# Ingest with agent scoping and access control
ingest-cli /path/to/doc.txt --agent-id "my-agent" --access-level restricted

# Pipe text directly
echo "content here" | ingest-cli --stdin --title "Quick Note"
```

### Via API

```bash
curl -X POST http://localhost:5001/api/ingest \
  -F "file=@report.pdf" \
  -F "title=Q4 Report" \
  -F "access_level=public"
```

---

## Step 6 — Chat via OpenWebUI

Open **http://localhost:8080**:

1. Create an account on first visit
2. Select **openclaw** from the model dropdown
3. Every message is automatically enriched via the GCOR pipeline before reaching the LLM

---

## Step 7 — Chat via OpenClaw

Open **http://localhost:18799**:

- Full agentic interface with tool use
- Native access to `neo4j-cli`, `qdrant-cli`, and `ingest-cli` as shell tools
- Connects to Neo4j and Qdrant via MCP sidecar servers
- Default model: `openai/gpt-4o` (configured in `openclaw-config/openclaw.json`)

---

## Step 8 — Browse the Knowledge Graph

Open **http://localhost:7474** (Neo4j Browser):
- **Username:** `neo4j` · **Password:** `test1234`

Useful Cypher queries:

```cypher
// All documents
MATCH (d:Document) RETURN d ORDER BY d.created_at DESC

// Document with its chunks
MATCH (d:Document)-[:CONTAINS]->(c:Chunk)
WHERE d.title CONTAINS "report"
RETURN d, c

// All cognitive nodes for an agent
MATCH (n) WHERE n.agent_id = "my-agent"
RETURN labels(n), n.confidence, n.created_at LIMIT 50

// Archived collections
MATCH (a:ArchivedCollection) RETURN a ORDER BY a.archived_at DESC
```

---

## Step 9 — Monitor (Grafana)

Open **http://localhost:3000** · `admin / admin`

The **EedgeAI — GCOR Full Observability** dashboard is auto-provisioned with 39 panels across 7 rows:

| Row | What you see |
|---|---|
| **Service Health** | Proxy / Neo4j / Qdrant / OpenClaw up/down, RAG requests (24h), ingests (24h) |
| **GCOR RAG Pipeline** | Request rate by intent, latency p50/p95/p99, intent distribution pie, fallback mode donut, avg Qdrant hits + Neo4j records per request |
| **LLM & Embeddings** | LLM call rate by backend (OpenAI / Anthropic), latency p50/p95, embedding API latency, error and exception counters |
| **Ingest Operations** | Ingest rate (success / error), avg chunks per document, total documents ingested, collection ops (1h) |
| **HTTP API (Proxy)** | Request rate by handler, 4xx / 5xx error rate, latency p50/p95/p99 |
| **Qdrant Vector DB** | REST request rate by endpoint, REST latency p95, gRPC request rate, collections total, vectors total |
| **Prometheus Health** | Scrape target availability over time, scrape duration per job |

### Prometheus Scrape Targets

| Target | Job | Metrics path |
|---|---|---|
| `proxy:5001` | `proxy` | `/metrics` |
| `qdrant:6333` | `qdrant` | `/metrics` |

> **Neo4j metrics:** Prometheus scraping of Neo4j requires **Neo4j Enterprise Edition**. Community edition does not expose a Prometheus endpoint. To enable when using Enterprise: add `NEO4J_server_metrics_prometheus_enabled: "true"` to the `neo4j` service in `docker-compose.unified.yml`, expose port `2004`, and uncomment the `neo4j` job in `monitoring/prometheus.yml`.

### Proxy Metrics (`/metrics`)

The proxy exposes the following custom metric families at `http://localhost:5001/metrics`:

| Metric | Type | Labels | Description |
|---|---|---|---|
| `gcor_rag_requests_total` | Counter | `intent`, `fallback_mode` | Every GCOR RAG pipeline invocation |
| `gcor_rag_duration_seconds` | Histogram | — | End-to-end pipeline latency (embed → Qdrant → Neo4j → build) |
| `gcor_qdrant_hits` | Histogram | — | Qdrant hits after cognitive filters, per request |
| `gcor_neo4j_records` | Histogram | — | Neo4j graph records expanded, per request |
| `gcor_embed_duration_seconds` | Histogram | — | OpenAI embedding API call latency |
| `gcor_llm_requests_total` | Counter | `backend`, `status` | LLM API calls (openai / anthropic) |
| `gcor_llm_duration_seconds` | Histogram | `backend` | LLM call latency (non-streaming) |
| `gcor_ingest_total` | Counter | `status` | Document ingest operations |
| `gcor_ingest_chunks` | Histogram | — | Chunks produced per ingested document |
| `gcor_collection_ops_total` | Counter | `operation` | Collection create / rename / archive / restore |

Standard HTTP metrics (`http_requests_total`, `http_request_duration_seconds`, etc.) are auto-instrumented by `prometheus-fastapi-instrumentator` for every endpoint.

`fallback_mode` values: `full_gcor` (graph + semantic), `qdrant_text` (semantic only, graph empty), `llm_only` (both empty), `disabled` (RAG off), `none` (no query).

---

## Stopping the Stack

```bash
docker compose -f docker-compose.unified.yml down

# Also remove all stored data
docker compose -f docker-compose.unified.yml down -v
```

---

## Rebuilding After Code Changes

```bash
docker compose -f docker-compose.unified.yml build <service>
docker compose -f docker-compose.unified.yml up -d <service>

# After restarting openclaw, always recreate the relay:
docker compose -f docker-compose.unified.yml up -d --force-recreate openclaw-relay
```

> ⚠️ **Important:** Every time the `openclaw` container restarts, the `openclaw-relay` must be force-recreated. The relay shares openclaw's network namespace — when openclaw gets a new namespace on restart, the relay's reference goes stale (`ERR_EMPTY_RESPONSE` on port 18799). Always run the force-recreate command above after any openclaw restart.

> ⚠️ **Important:** When Grafana is force-recreated (fresh container), it starts with an empty database and re-provisions the datasource and dashboard from the `monitoring/grafana-provisioning/` files automatically.

---

## Troubleshooting

**OpenClaw shows `ERR_EMPTY_RESPONSE` on port 18799:**
```bash
docker compose -f docker-compose.unified.yml up -d --force-recreate openclaw-relay
```
This happens every time the `openclaw` container is restarted. The relay must be recreated to attach to the new network namespace.

**OpenWebUI shows no models / only one model:**
```bash
curl http://localhost:5001/v1/models | python3 -m json.tool | grep '"id"' | head -3
```
The first model should be `"openclaw"`. If not, check that the proxy is healthy:
```bash
curl http://localhost:5001/health
```

**Knowledge UI shows no collections:**
The `documents` Qdrant collection is created automatically on first ingest. If Qdrant was wiped, ingest any document via the Knowledge UI to recreate it.

**Qdrant search returns 0 results:**
```bash
curl http://localhost:6333/collections/documents
```
Check `points_count`. If 0, the collection exists but is empty — ingest documents first.

**Neo4j connection refused:**
Neo4j takes 30–60 s to initialise. Wait and retry. Check health:
```bash
docker compose -f docker-compose.unified.yml logs neo4j | tail -20
```

**OpenClaw LLM errors (credit / billing):**
OpenClaw defaults to `openai/gpt-4o`. To switch models:
```bash
# From inside the container
docker exec eedgeai-openclaw-1 openclaw models set openai/gpt-4o
# Or edit openclaw-config/openclaw.json:
# { "gateway": { "mode": "local" }, "agents": { "defaults": { "model": "openai/gpt-4o" } } }
```
> Note: `openclaw-config/openclaw.json` is mounted read-only. Edit the host file and restart openclaw + relay.

**MCP servers not responding (OpenClaw tools fail):**
```bash
docker compose -f docker-compose.unified.yml logs mcp-qdrant
docker compose -f docker-compose.unified.yml logs mcp-neo4j
docker compose -f docker-compose.unified.yml restart mcp-qdrant mcp-neo4j
```

**RAG returns 0 context (proxy logs show 0 chunks / 0 records):**
```bash
docker compose -f docker-compose.unified.yml logs proxy --tail 20
```
- `Qdrant collection 'documents' not found` → ingest at least one document
- `Neo4j: 0 graph records expanded` → graph is empty, ingest documents first
- To disable RAG temporarily: set `ENABLE_RAG=false` in `.env` and restart the proxy

**Grafana shows no data / datasource not found:**
```bash
# Force-recreate Grafana to reset its database and re-provision from config files
docker compose -f docker-compose.unified.yml up -d --force-recreate grafana
```
Then verify Prometheus targets are up at `http://localhost:9090/targets` — `proxy` and `qdrant` should both show `UP`.

**Grafana dashboard not appearing:**
Check that the provisioning files are mounted correctly:
```bash
docker exec eedgeai-grafana-1 ls /var/lib/grafana/dashboards/
docker exec eedgeai-grafana-1 ls /etc/grafana/provisioning/datasources/
```

**Proxy `/metrics` endpoint not responding:**
```bash
curl http://localhost:5001/metrics | head -5
```
If the proxy container is healthy but `/metrics` 404s, rebuild the proxy image:
```bash
docker compose -f docker-compose.unified.yml build proxy
docker compose -f docker-compose.unified.yml up -d proxy
```
