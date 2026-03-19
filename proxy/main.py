"""
Graph-Centric Orchestrated Retrieval (GCOR) proxy for OpenWebUI.

Architecture
────────────
  Neo4j   = cognitive backbone   (primary truth, structure, reasoning)
  Qdrant  = semantic perception  (index over Neo4j nodes, similarity only)

Retrieval flow for every /v1/chat/completions request
─────────────────────────────────────────────────────
  1. INTENT CLASSIFICATION
     Classify the query: factual / planning / dependency / memory / semantic

  2. SEMANTIC PHASE  (Qdrant)
     Embed the query → top-K candidate Chunk nodes
     Each hit carries neo4j_element_id — not the answer, just a pointer

  3. STRUCTURAL PHASE  (Neo4j)
     For each candidate element_id, expand the graph:
       factual    → node + immediate relationships + source document
       planning   → node + DEPENDS_ON chain (2 hops) + sibling goals
       dependency → full DEPENDS_ON path (3 hops)
       memory     → node + ABOUT concepts + ABOUT memories + timeline
       semantic   → node + MENTIONS concepts + FOLLOWS neighbors

  4. REFLECTION CHECK  (lightweight)
     If Neo4j returns nothing but Qdrant has hits, fall back to chunk text.
     If both return nothing, answer from LLM general knowledge.

  5. CONTEXT INJECTION
     Build a structured system message and call the LLM.

Other endpoints
───────────────
  GET  /v1/models      → live OpenAI model list (with fallback)
  POST /v1/embeddings  → proxied to OpenAI (for OpenWebUI's own RAG)
  GET  /health         → liveness check
"""

import json
import logging
import os

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from neo4j import AsyncGraphDatabase

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# ── LLM backends ──────────────────────────────────────────────────────────────
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL    = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_CHAT_MODEL = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-6")
LLM_BACKEND          = os.getenv("LLM_BACKEND", "openai")

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── Qdrant (semantic perception layer) ────────────────────────────────────────
QDRANT_URL        = os.getenv("QDRANT_URL",        "http://quarant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
QDRANT_TOP_K      = int(os.getenv("QDRANT_TOP_K",  "8"))

# ── Neo4j (cognitive backbone) ────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://neo4j:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() in ("true", "1", "yes")


# ── Step 1: Intent classification ─────────────────────────────────────────────

_INTENT_KEYWORDS = {
    "planning":    ["plan", "schedule", "next step", "roadmap", "workflow",
                    "milestone", "sprint", "agenda", "timeline", "sequence"],
    "dependency":  ["depends", "requires", "blocked", "prerequisite", "constraint",
                    "before", "needs", "relies on", "linked to", "after"],
    "memory":      ["remember", "recall", "history", "last time", "previously",
                    "past", "earlier", "before we", "you said", "i told"],
    "factual":     ["what is", "how does", "why", "explain", "define", "describe",
                    "tell me about", "how do", "what are"],
}

def classify_intent(query: str) -> str:
    q = query.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent
    return "semantic"


# ── Step 2a: Semantic phase (Qdrant) ──────────────────────────────────────────

async def embed_text(text: str) -> list | None:
    if not OPENAI_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            resp = await c.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={"input": text[:8000], "model": EMBEDDING_MODEL},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
    except Exception as exc:
        logger.warning("Embedding failed: %s", exc)
        return None


async def qdrant_search(vector: list) -> list:
    """Return top-K Qdrant hits; each hit includes the full payload (with neo4j_element_id)."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            resp = await c.post(
                f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
                json={"vector": vector, "limit": QDRANT_TOP_K, "with_payload": True},
            )
            if resp.status_code == 404:
                logger.info("Qdrant collection '%s' not found.", QDRANT_COLLECTION)
                return []
            resp.raise_for_status()
            return resp.json().get("result", [])
    except Exception as exc:
        logger.warning("Qdrant search failed: %s", exc)
        return []


# ── Step 2b: Structural phase (Neo4j) ─────────────────────────────────────────

# Intent-specific Cypher — all return {node, labels, document, [extras]}
_EXPAND_CYPHER = {
    "factual": """
        MATCH (n) WHERE elementId(n) IN $ids
        OPTIONAL MATCH (n)-[r]-(related)
        OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document)
        RETURN
            properties(n) AS node,
            labels(n)     AS labels,
            properties(doc) AS document,
            collect(DISTINCT {
                rel:   type(r),
                props: properties(related),
                lbls:  labels(related)
            })[..6] AS related
        LIMIT 20
    """,
    "planning": """
        MATCH (n) WHERE elementId(n) IN $ids
        OPTIONAL MATCH (n)-[:DEPENDS_ON*1..2]->(dep)
        OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document)
        OPTIONAL MATCH (g:Goal)-[:DEPENDS_ON]->(n)
        RETURN
            properties(n) AS node,
            labels(n)     AS labels,
            properties(doc) AS document,
            collect(DISTINCT properties(dep))[..5] AS dependencies,
            collect(DISTINCT properties(g))[..5]   AS blocking_goals
        LIMIT 20
    """,
    "dependency": """
        MATCH (n) WHERE elementId(n) IN $ids
        OPTIONAL MATCH path = (n)-[:DEPENDS_ON*1..3]->(dep)
        OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document)
        RETURN
            properties(n)  AS node,
            labels(n)      AS labels,
            properties(doc) AS document,
            [x IN nodes(path) | properties(x)][..8] AS dependency_chain
        LIMIT 20
    """,
    "memory": """
        MATCH (n) WHERE elementId(n) IN $ids
        OPTIONAL MATCH (n)-[:MENTIONS]->(concept:Concept)
        OPTIONAL MATCH (mem:Memory)-[:ABOUT]->(concept)
        OPTIONAL MATCH (evt:Event)-[:INVOLVES]->(concept)
        OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document)
        RETURN
            properties(n)   AS node,
            labels(n)       AS labels,
            properties(doc) AS document,
            collect(DISTINCT properties(concept))[..5] AS concepts,
            collect(DISTINCT properties(mem))[..5]     AS memories,
            collect(DISTINCT properties(evt))[..5]     AS events
        ORDER BY evt.timestamp DESC
        LIMIT 20
    """,
    "semantic": """
        MATCH (n) WHERE elementId(n) IN $ids
        OPTIONAL MATCH (n)-[:MENTIONS]->(concept:Concept)
        OPTIONAL MATCH (n)-[:FOLLOWS]-(neighbor:Chunk)
        OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document)
        RETURN
            properties(n)   AS node,
            labels(n)       AS labels,
            properties(doc) AS document,
            collect(DISTINCT properties(concept))[..5]  AS concepts,
            collect(DISTINCT properties(neighbor))[..3] AS neighbors
        LIMIT 20
    """,
}

_FALLBACK_CYPHER = """
    MATCH (n)
    WHERE any(k IN keys(n) WHERE toLower(toString(n[k])) CONTAINS toLower($q))
    RETURN properties(n) AS node, labels(n) AS labels, null AS document
    LIMIT 10
"""


async def neo4j_expand(element_ids: list, intent: str, query: str) -> list:
    """Expand the graph around Qdrant candidate nodes based on intent."""
    cypher = _EXPAND_CYPHER.get(intent, _EXPAND_CYPHER["semantic"])
    params = {"ids": element_ids} if element_ids else {}

    if not element_ids:
        cypher = _FALLBACK_CYPHER
        params = {"q": query[:100]}

    try:
        driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                result = await session.run(cypher, params)
                return await result.data()
        finally:
            await driver.close()
    except Exception as exc:
        logger.warning("Neo4j expansion failed: %s", exc)
        return []


# ── Step 3: Reflection ─────────────────────────────────────────────────────────

def _reflection_fallback(qdrant_hits: list, graph_records: list) -> list:
    """If graph is empty but Qdrant has hits, use raw chunk text as fallback context."""
    if graph_records:
        return graph_records
    if qdrant_hits:
        logger.info("Reflection: Neo4j empty, falling back to Qdrant chunk text.")
        return [
            {
                "node":     {"text": h["payload"].get("text", ""), "document_id": h["payload"].get("document_id", "")},
                "labels":   ["Chunk"],
                "document": {"title": h["payload"].get("document_title", "")},
            }
            for h in qdrant_hits
            if h.get("payload", {}).get("text")
        ]
    return []


# ── Context builder ────────────────────────────────────────────────────────────

def build_gcor_context(intent: str, qdrant_hits: list, graph_records: list) -> str:
    """Build a structured system-message context block from GCOR results."""
    records = _reflection_fallback(qdrant_hits, graph_records)
    if not records:
        return ""

    lines = [
        f"## Retrieved Context  [intent: {intent} | "
        f"{len(qdrant_hits)} semantic candidates → {len(records)} graph records]",
        "",
        "Neo4j is the primary source. Qdrant identified the entry points.",
        "",
    ]

    for rec in records:
        node   = rec.get("node") or {}
        labels = rec.get("labels") or []
        doc    = rec.get("document") or {}

        node_type = labels[0] if labels else "Node"
        content   = (
            node.get("text") or node.get("content") or
            node.get("description") or node.get("title") or
            str(node)
        )[:600]

        lines.append(f"### [{node_type}]  {content}")

        if doc.get("title") or doc.get("source"):
            lines.append(f"Source: {doc.get('title') or doc.get('source', '')}")

        # Intent-specific extras
        if intent == "planning":
            deps = [d for d in (rec.get("dependencies") or []) if d]
            if deps:
                lines.append("Dependencies: " + " → ".join(
                    d.get("description") or d.get("title") or str(d) for d in deps
                ))
            blocking = [g for g in (rec.get("blocking_goals") or []) if g]
            if blocking:
                lines.append("Blocked by: " + ", ".join(
                    g.get("description") or str(g) for g in blocking
                ))

        elif intent == "dependency":
            chain = [x for x in (rec.get("dependency_chain") or []) if x]
            if len(chain) > 1:
                lines.append("Dependency path: " + " → ".join(
                    x.get("description") or x.get("title") or str(x) for x in chain
                ))

        elif intent == "memory":
            mems = [m for m in (rec.get("memories") or []) if m]
            if mems:
                lines.append("Related memories: " + "; ".join(
                    m.get("content") or str(m) for m in mems[:3]
                ))
            evts = [e for e in (rec.get("events") or []) if e]
            if evts:
                lines.append("Related events: " + "; ".join(
                    f"{e.get('timestamp', '')} {e.get('description', '')}" for e in evts[:3]
                ))

        elif intent == "semantic":
            concepts = [c for c in (rec.get("concepts") or []) if c]
            if concepts:
                lines.append("Concepts: " + ", ".join(c.get("name", "") for c in concepts))
            neighbors = [n for n in (rec.get("neighbors") or []) if n]
            if neighbors:
                lines.append("Adjacent chunks: " + " | ".join(
                    n.get("text", "")[:100] for n in neighbors
                ))

        else:  # factual
            related = [r for r in (rec.get("related") or []) if r and r.get("props")]
            if related:
                lines.append("Related: " + "; ".join(
                    f"{r['rel']} → {r['props'].get('name') or r['props'].get('text', '')[:80]}"
                    for r in related[:4]
                ))

        lines.append("")

    lines.append(
        "Instruction: use the retrieved context above when relevant. "
        "If the context is insufficient, apply general knowledge."
    )
    return "\n".join(lines)


def inject_context(messages: list, context: str) -> list:
    if not context:
        return messages
    result, merged = [], False
    for m in messages:
        if m.get("role") == "system" and not merged:
            result.append({"role": "system", "content": context + "\n\n---\n\n" + m.get("content", "")})
            merged = True
        else:
            result.append(m)
    if not merged:
        result = [{"role": "system", "content": context}] + messages
    return result


def last_user_text(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
    return ""


# ── LLM routing ───────────────────────────────────────────────────────────────

def _resolve_model(model: str) -> tuple:
    if model.startswith("claude-"):
        return model, "anthropic"
    if model.startswith(("gpt-", "o1", "o3")):
        return model, "openai"
    if LLM_BACKEND == "anthropic":
        return ANTHROPIC_CHAT_MODEL, "anthropic"
    return OPENAI_CHAT_MODEL, "openai"


async def call_openai(body: dict):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    if body.get("stream"):
        async def stream():
            try:
                async with httpx.AsyncClient(timeout=120) as c:
                    async with c.stream("POST", "https://api.openai.com/v1/chat/completions",
                                        json=body, headers=headers) as resp:
                        if resp.status_code != 200:
                            err = await resp.aread()
                            logger.error("OpenAI %s: %s", resp.status_code, err[:200])
                            yield f'data: {{"error": "OpenAI {resp.status_code}"}}\n\n'.encode()
                            return
                        async for chunk in resp.aiter_bytes():
                            yield chunk
            except Exception as exc:
                logger.error("OpenAI stream error: %s", exc)
                yield b'data: {"error": "proxy error"}\n\n'
        return StreamingResponse(stream(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=120) as c:
            resp = await c.post("https://api.openai.com/v1/chat/completions",
                                json=body, headers=headers)
            resp.raise_for_status()
            return JSONResponse(content=resp.json(), status_code=resp.status_code)


async def call_anthropic(body: dict):
    messages   = body.get("messages", [])
    system_txt = next((m["content"] for m in messages if m.get("role") == "system"), None)
    user_msgs  = [m for m in messages if m.get("role") != "system"]
    ant_body   = {"model": body.get("model", ANTHROPIC_CHAT_MODEL),
                  "max_tokens": body.get("max_tokens", 4096), "messages": user_msgs}
    if system_txt:
        ant_body["system"] = system_txt
    headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
               "Content-Type": "application/json"}

    if body.get("stream"):
        ant_body["stream"] = True
        async def stream():
            try:
                async with httpx.AsyncClient(timeout=120) as c:
                    async with c.stream("POST", "https://api.anthropic.com/v1/messages",
                                        json=ant_body, headers=headers) as resp:
                        if resp.status_code != 200:
                            err = await resp.aread()
                            logger.error("Anthropic %s: %s", resp.status_code, err[:200])
                            yield f'data: {{"error": "Anthropic {resp.status_code}"}}\n\n'.encode()
                            return
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            raw = line[6:]
                            if raw == "[DONE]":
                                yield b"data: [DONE]\n\n"; continue
                            try:
                                evt = json.loads(raw)
                                if evt.get("type") == "content_block_delta":
                                    txt = evt.get("delta", {}).get("text", "")
                                    yield f'data: {json.dumps({"choices": [{"delta": {"content": txt}, "finish_reason": None}]})}\n\n'.encode()
                                elif evt.get("type") == "message_stop":
                                    yield b"data: [DONE]\n\n"
                            except json.JSONDecodeError:
                                pass
            except Exception as exc:
                logger.error("Anthropic stream error: %s", exc)
                yield b'data: {"error": "proxy error"}\n\n'
        return StreamingResponse(stream(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=120) as c:
            resp = await c.post("https://api.anthropic.com/v1/messages",
                                json=ant_body, headers=headers)
            resp.raise_for_status()
            ant = resp.json()
            usage = ant.get("usage", {})
            return JSONResponse(content={
                "id": ant.get("id", ""), "object": "chat.completion",
                "model": ant.get("model", ant_body["model"]),
                "choices": [{"index": 0,
                             "message": {"role": "assistant",
                                         "content": ant.get("content", [{}])[0].get("text", "")},
                             "finish_reason": ant.get("stop_reason", "stop")}],
                "usage": {"prompt_tokens": usage.get("input_tokens", 0),
                          "completion_tokens": usage.get("output_tokens", 0),
                          "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)},
            })


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    models = []
    if OPENAI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                resp = await c.get("https://api.openai.com/v1/models",
                                   headers={"Authorization": f"Bearer {OPENAI_API_KEY}"})
                if resp.status_code == 200:
                    models = resp.json().get("data", [])
        except Exception as exc:
            logger.warning("Could not fetch OpenAI models: %s", exc)
    if not models:
        models = [
            {"id": OPENAI_CHAT_MODEL,    "object": "model", "owned_by": "openai"},
            {"id": ANTHROPIC_CHAT_MODEL, "object": "model", "owned_by": "anthropic"},
            {"id": "openclaw",           "object": "model", "owned_by": "eedgeai"},
        ]
    return {"object": "list", "data": models}


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """Proxy embedding requests to OpenAI (for OpenWebUI's built-in RAG)."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            resp = await c.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json=body,
            )
            resp.raise_for_status()
            return JSONResponse(content=resp.json())
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail="Embeddings upstream error")
    except Exception as exc:
        logger.error("Embeddings failed: %s", exc)
        raise HTTPException(status_code=502, detail="Embeddings request failed")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body     = await request.json()
    messages = list(body.get("messages", []))

    if ENABLE_RAG and messages:
        query = last_user_text(messages)

        if query:
            # ── Step 1: Intent ─────────────────────────────────────────────────
            intent = classify_intent(query)
            logger.info("GCOR intent: %s", intent)

            # ── Step 2a: Semantic phase (Qdrant → element IDs) ─────────────────
            vector       = await embed_text(query)
            qdrant_hits  = await qdrant_search(vector) if vector else []
            element_ids  = [
                h["payload"]["neo4j_element_id"]
                for h in qdrant_hits
                if h.get("payload", {}).get("neo4j_element_id")
            ]
            logger.info("Qdrant: %d candidates, %d with neo4j_element_id",
                        len(qdrant_hits), len(element_ids))

            # ── Step 2b: Structural phase (Neo4j graph expansion) ──────────────
            graph_records = await neo4j_expand(element_ids, intent, query)
            logger.info("Neo4j: %d graph records expanded", len(graph_records))

            # ── Step 3: Reflection + context build ─────────────────────────────
            context  = build_gcor_context(intent, qdrant_hits, graph_records)
            messages = inject_context(messages, context)
            body     = {**body, "messages": messages}

    # ── Route to LLM ──────────────────────────────────────────────────────────
    model, backend = _resolve_model(body.get("model", "openclaw"))
    body = {**body, "model": model}
    try:
        if backend == "anthropic" and ANTHROPIC_API_KEY:
            return await call_anthropic(body)
        elif OPENAI_API_KEY:
            return await call_openai(body)
        else:
            raise HTTPException(status_code=503, detail="No LLM API key configured")
    except httpx.HTTPStatusError as exc:
        logger.error("LLM upstream error: %s", exc)
        raise HTTPException(status_code=exc.response.status_code, detail="LLM upstream error")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Chat completion failed: %s", exc)
        raise HTTPException(status_code=502, detail="Chat completion failed")


@app.get("/health")
async def health():
    return {"status": "ok"}
