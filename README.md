# EedgeAI Agentic Stack

A unified AI agent stack combining graph database (Neo4j), vector search (Qdrant), RAG orchestration (OpenClaw), and a chat UI (Open WebUI) with monitoring.

---

## Architecture

```
User
 └─► Open WebUI (port 8080)
       └─► Proxy / RAG Pipeline (port 5001)
             ├─► Qdrant (port 6333)      ← semantic search for context
             ├─► Neo4j (port 7687)       ← knowledge graph for context
             └─► LLM API (OpenAI / Anthropic) ← answers with injected context

OpenClaw Agent (ports 18799 / 18801)   ← standalone agent interface (separate from OpenWebUI)
 ├─► MCP Server: Neo4j  (port 8766) ──► Neo4j (port 7687)
 └─► MCP Server: Qdrant (port 8765) ──► Qdrant (port 6333)

Monitoring
 ├─► Prometheus (port 9090)
 └─► Grafana (port 3000)
```

### How OpenWebUI gets RAG context (proxy pipeline)

The proxy at port 5001 is the core RAG engine for OpenWebUI:

1. **Embed** — the user's message is embedded using `text-embedding-3-small` (OpenAI)
2. **Qdrant search** — the proxy queries Qdrant for the top-k most similar document chunks
3. **Neo4j search** — the proxy queries Neo4j for nodes whose properties match the query keywords
4. **Context injection** — retrieved chunks and graph nodes are prepended as a system message
5. **LLM call** — the enriched request is forwarded to OpenAI or Anthropic and the response is streamed back

The proxy also handles `/v1/embeddings` so OpenWebUI's own built-in RAG (for uploaded documents) works correctly.

### OpenClaw (separate agent interface)

OpenClaw is a standalone AI coding assistant. It connects to Neo4j and Qdrant through MCP sidecar
services (`mcp-qdrant`, `mcp-neo4j`). Its web interface is available at `http://localhost:18801`.
It operates independently of OpenWebUI.

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- At least 8 GB RAM allocated to Docker
- Ports 3000, 5001, 6333, 7474, 7687, 8080, 9090, 18799, 18801 free on your machine

---

## Step 1 — Configure Environment Variables

Copy `.env` and fill in your credentials:

```bash
cp .env .env.local   # or edit .env directly
```

Required variables:

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI API key (also used for embeddings) |
| `OPENCLAW_GATEWAY_TOKEN` | Token for OpenClaw gateway authentication |
| `OPENCLAW_GATEWAY_PASSWORD` | Password for OpenClaw gateway |

> **Security:** Never commit `.env` to version control. Add it to `.gitignore`.

---

## Step 2 — Project Structure

```
eedgeai/
├── docker-compose.unified.yml
├── .env
├── openclaw-config/
│   └── openclaw.json          ← MCP server wiring (Qdrant + Neo4j)
├── mcp-servers/
│   ├── qdrant/
│   │   └── Dockerfile         ← mcp-server-qdrant (SSE on :8765)
│   └── neo4j/
│       └── Dockerfile         ← mcp-neo4j-cypher  (SSE on :8766)
├── proxy/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── openclaw-relay/
│   └── relay.js
├── monitoring/
│   ├── prometheus.yml
│   ├── grafana-dashboard.json
│   └── grafana-provisioning/
│       ├── datasources/
│       │   └── prometheus.yaml
│       └── dashboards/
│           └── default.yaml
└── agentic-stack/
    └── agentic_stack/
        ├── config/
        │   ├── agents.yaml
        │   └── model_registry.yaml
        └── ingestion/
            └── ingest.py
```

---

## Step 3 — Start the Stack

Open a terminal in the `eedgeai/` folder and run:

```bash
docker compose -f docker-compose.unified.yml up -d
```

The first run will pull all images and build the proxy (~5–10 minutes depending on internet speed).

Services start in dependency order: Neo4j and Qdrant become healthy first, then OpenClaw, then the proxy, then Open WebUI.

To watch logs in real time:

```bash
docker compose -f docker-compose.unified.yml logs -f
```

---

## Step 4 — Verify All Services Are Running

```bash
docker compose -f docker-compose.unified.yml ps
```

All 10 services should show `Up` or `healthy`:

| Service         | Port            | Description                                        |
|-----------------|-----------------|----------------------------------------------------|
| openwebui       | 8080            | Chat UI                                            |
| proxy           | 5001            | RAG pipeline — embeds, retrieves, calls LLM        |
| openclaw        | 18799, 18801    | Standalone AI agent (its own chat interface)       |
| openclaw-relay  | (shared)        | TCP relay (18799→18789, 18801→18790)               |
| mcp-qdrant      | 8765 (internal) | MCP server — exposes Qdrant to OpenClaw agent      |
| mcp-neo4j       | 8766 (internal) | MCP server — exposes Neo4j to OpenClaw agent       |
| neo4j           | 7474, 7687      | Graph database                                     |
| quarant         | 6333            | Qdrant vector database                             |
| prometheus      | 9090            | Metrics collection                                 |
| grafana         | 3000            | Metrics dashboard                                  |

---

## Step 5 — Access the Chat UI

Open your browser and go to:

```
http://localhost:8080
```

1. Create an account on the first visit (local only, no internet required).
2. The UI is pre-configured to route queries through the proxy to OpenClaw.
3. Select the **openclaw** model from the model dropdown.
4. Start chatting — queries are answered using the knowledge graph and vector store.

---

## Step 6 — Ingest Data (Optional)

To populate the vector store with your own documents, use the ingestion script:

```bash
cd eedgeai/agentic-stack/agentic_stack/ingestion

# Install dependencies first
pip install httpx pyyaml openai          # for OpenAI/Azure models
pip install sentence-transformers        # for local models

# Ingest a directory of documents (txt, md, json, yaml)
python ingest.py --input ./docs --model openai_large

# Use a local model (no API key needed)
python ingest.py --input ./docs --model sentence_transformer

# Target a specific Qdrant collection
python ingest.py --input ./docs --model openai_large --collection my_docs
```

The script:
1. Reads all `.txt`, `.md`, `.rst`, `.json`, `.yaml` files under `--input`
2. Splits them into overlapping chunks
3. Embeds each chunk using the selected model
4. Creates the Qdrant collection if it doesn't exist
5. Upserts all vectors with filename + text as payload

Available embedding models (configured in `config/agents.yaml`):

| Model Key              | Type             | Description                    |
|------------------------|------------------|--------------------------------|
| `openai_large`         | OpenAI API       | text-embedding-3-large         |
| `openai_small`         | OpenAI API       | text-embedding-3-small         |
| `sentence_transformer` | Local            | all-MiniLM-L6-v2 (no API key) |
| `code_embedding`       | Local            | codebert-base                  |
| `azure_openai`         | Azure OpenAI API | text-embedding-3-large         |

Environment variables used by the ingestion script:

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `documents` | Default collection name |
| `OPENAI_API_KEY` | — | Required for OpenAI/Azure models |

To change the default active model, edit `config/agents.yaml`:

```yaml
active_model: sentence_transformer
```

---

## Step 7 — Browse the Knowledge Graph (Neo4j)

Open your browser and go to:

```
http://localhost:7474
```

- **Username:** `neo4j`
- **Password:** `test1234`

Run Cypher queries to explore the graph, e.g.:

```cypher
MATCH (n) RETURN n LIMIT 25
```

---

## Step 8 — Monitor the Stack (Grafana)

Open your browser and go to:

```
http://localhost:3000
```

- **Username:** `admin`
- **Password:** `admin` (you will be prompted to change on first login)

The **Agent System Metrics** dashboard is auto-provisioned on startup. It includes:

- Service up/down status for proxy, Neo4j, OpenClaw, and Qdrant
- HTTP request rate through the proxy
- HTTP 5xx error rate

Prometheus scrapes metrics from:

| Target | Job |
|---|---|
| `proxy:5001` | `proxy` |
| `neo4j:2004` | `neo4j` |
| `openclaw:9091` | `openclaw` |
| `quarant:6333` | `qdrant` |

---

## Stopping the Stack

```bash
docker compose -f docker-compose.unified.yml down
```

To also remove all stored data (volumes):

```bash
docker compose -f docker-compose.unified.yml down -v
```

---

## Restarting / Updating

```bash
# Pull latest images and restart
docker compose -f docker-compose.unified.yml pull
docker compose -f docker-compose.unified.yml up -d --build
```

---

## Troubleshooting

**A container keeps restarting:**
```bash
docker compose -f docker-compose.unified.yml logs <service-name>
# e.g.
docker compose -f docker-compose.unified.yml logs openclaw
```

**Port already in use:**
Edit `docker-compose.unified.yml` and change the left-hand port number, e.g. `"8081:8080"` for Open WebUI.

**OpenWebUI shows no models:**
Confirm the proxy is running and healthy:
```bash
curl http://localhost:5001/health
curl http://localhost:5001/v1/models
```
Expected response from `/v1/models`: `{"object":"list","data":[{"id":"openclaw",...}]}`

If the proxy can't reach OpenClaw it will return a fallback model list and log a warning — check proxy logs:
```bash
docker compose -f docker-compose.unified.yml logs proxy
```

**Neo4j connection refused:**
Neo4j can take 30–60 seconds to fully initialise. The stack uses health checks so dependent services wait automatically, but you may still need to wait before hitting port 7474 from your browser.

**Grafana shows no data:**
- Confirm Prometheus is up: `http://localhost:9090/targets`
- All targets should show `UP`. If a target is down, check that the relevant container is running.

**OpenWebUI chat has no RAG context (answers don't use Neo4j / Qdrant data):**

The proxy handles RAG automatically — each chat message triggers an embed + Qdrant search + Neo4j lookup before calling the LLM. Check the proxy logs to see what's happening:
```bash
docker compose -f docker-compose.unified.yml logs proxy
```
You should see lines like:
```
INFO: Qdrant: 5 chunk(s) retrieved
INFO: Neo4j:  2 node(s) retrieved
```
If Qdrant returns 0 chunks, the collection is likely empty — run the ingestion script first (see Step 6).
If Neo4j returns 0 nodes, the graph is empty — add data via the Neo4j browser at `http://localhost:7474`.
To disable RAG temporarily (answer without context), set `ENABLE_RAG=false` in `.env` and restart the proxy.

**OpenClaw can't query Neo4j or Qdrant (OpenClaw's own interface):**

OpenClaw uses MCP sidecar services for its native interface. Check them:
```bash
docker compose -f docker-compose.unified.yml logs mcp-qdrant
docker compose -f docker-compose.unified.yml logs mcp-neo4j
```
If they failed to start, restart them:
```bash
docker compose -f docker-compose.unified.yml restart mcp-qdrant mcp-neo4j
```

**Ingestion script fails:**
- Ensure required pip packages are installed (`openai`, `httpx`, `pyyaml`, or `sentence-transformers` depending on the model).
- For OpenAI models, confirm `OPENAI_API_KEY` is set in your environment.
- Confirm Qdrant is reachable at `QDRANT_URL` (default `http://localhost:6333`).
