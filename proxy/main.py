"""
RAG-enhanced OpenAI-compatible proxy for OpenWebUI.

Request flow for /v1/chat/completions:
  1. Embed the last user message (OpenAI Embeddings API)
  2. Search Qdrant for the top-k most relevant document chunks
  3. Query Neo4j for related knowledge graph nodes
  4. Inject the retrieved context into the system message
  5. Call the configured LLM backend (OpenAI or Anthropic)
  6. Stream / return the response to OpenWebUI

Other endpoints:
  GET  /v1/models      → list available models
  POST /v1/embeddings  → proxy to OpenAI Embeddings (for OpenWebUI's built-in RAG)
  GET  /health         → liveness check
"""

import json
import logging
import os

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# ── LLM backends ──────────────────────────────────────────────────────────────
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL    = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_CHAT_MODEL = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-6")
# "openai" or "anthropic" – used when the model name is "openclaw" (UI default)
LLM_BACKEND          = os.getenv("LLM_BACKEND", "openai")

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_URL        = os.getenv("QDRANT_URL", "http://quarant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
QDRANT_TOP_K      = int(os.getenv("QDRANT_TOP_K", "5"))

# ── Neo4j ─────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")

# ── Feature flags ─────────────────────────────────────────────────────────────
ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() in ("true", "1", "yes")


# ── RAG helpers ───────────────────────────────────────────────────────────────

async def embed_text(text: str) -> list | None:
    """Return an embedding vector for *text* using the OpenAI Embeddings API."""
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
    """Return top-k text chunks from Qdrant closest to *vector*."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            resp = await c.post(
                f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
                json={"vector": vector, "limit": QDRANT_TOP_K, "with_payload": True},
            )
            if resp.status_code == 404:
                logger.info("Qdrant collection '%s' not found – skipping.", QDRANT_COLLECTION)
                return []
            resp.raise_for_status()
            return [
                hit["payload"]["text"]
                for hit in resp.json().get("result", [])
                if hit.get("payload", {}).get("text")
            ]
    except Exception as exc:
        logger.warning("Qdrant search failed: %s", exc)
        return []


async def neo4j_search(query: str) -> list:
    """Return text from Neo4j nodes whose properties contain keywords from *query*."""
    try:
        from neo4j import AsyncGraphDatabase

        driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            async with driver.session() as session:
                # Try a full-text index first; fall back to a property scan.
                try:
                    result = await session.run(
                        "CALL db.index.fulltext.queryNodes('nodeIndex', $q) "
                        "YIELD node RETURN toString(properties(node)) AS text LIMIT 5",
                        q=query[:200],
                    )
                    records = await result.data()
                    texts = [r["text"] for r in records if r.get("text")]
                except Exception:
                    result = await session.run(
                        "MATCH (n) "
                        "WHERE any(k IN keys(n) WHERE toString(n[k]) CONTAINS $q) "
                        "RETURN toString(properties(n)) AS text LIMIT 5",
                        q=query[:100],
                    )
                    records = await result.data()
                    texts = [r["text"] for r in records if r.get("text")]
                return texts
        finally:
            await driver.close()
    except Exception as exc:
        logger.warning("Neo4j search failed: %s", exc)
        return []


def last_user_text(messages: list) -> str:
    """Extract the text of the last user message."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
    return ""


def inject_context(messages: list, qdrant_hits: list, neo4j_hits: list) -> list:
    """Prepend retrieved context into the system message (or create one)."""
    if not qdrant_hits and not neo4j_hits:
        return messages

    sections = []
    if qdrant_hits:
        chunks = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(qdrant_hits))
        sections.append(f"### Retrieved document chunks\n{chunks}")
    if neo4j_hits:
        entries = "\n".join(f"- {t}" for t in neo4j_hits)
        sections.append(f"### Related knowledge graph nodes\n{entries}")

    ctx = (
        "Use the following retrieved context when answering. "
        "If the context is not relevant, rely on your general knowledge.\n\n"
        + "\n\n".join(sections)
    )

    result, merged = [], False
    for m in messages:
        if m.get("role") == "system" and not merged:
            result.append({"role": "system", "content": ctx + "\n\n---\n\n" + m.get("content", "")})
            merged = True
        else:
            result.append(m)
    if not merged:
        result = [{"role": "system", "content": ctx}] + messages
    return result


# ── LLM forwarding ────────────────────────────────────────────────────────────

def _resolve_model(model: str) -> tuple[str, str]:
    """Return (resolved_model_name, backend) for a given model string."""
    if model.startswith("claude-"):
        return model, "anthropic"
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        return model, "openai"
    # "openclaw" or any other placeholder → use configured backend
    if LLM_BACKEND == "anthropic":
        return ANTHROPIC_CHAT_MODEL, "anthropic"
    return OPENAI_CHAT_MODEL, "openai"


async def call_openai(body: dict):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    if body.get("stream"):
        async def stream():
            try:
                async with httpx.AsyncClient(timeout=120) as c:
                    async with c.stream(
                        "POST",
                        "https://api.openai.com/v1/chat/completions",
                        json=body,
                        headers=headers,
                    ) as resp:
                        if resp.status_code != 200:
                            err = await resp.aread()
                            logger.error("OpenAI error %s: %s", resp.status_code, err[:200])
                            yield f'data: {{"error": "OpenAI error {resp.status_code}"}}\n\n'.encode()
                            return
                        async for chunk in resp.aiter_bytes():
                            yield chunk
            except Exception as exc:
                logger.error("OpenAI stream error: %s", exc)
                yield b'data: {"error": "proxy error"}\n\n'

        return StreamingResponse(stream(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=120) as c:
            resp = await c.post(
                "https://api.openai.com/v1/chat/completions",
                json=body,
                headers=headers,
            )
            resp.raise_for_status()
            return JSONResponse(content=resp.json(), status_code=resp.status_code)


async def call_anthropic(body: dict):
    """Convert an OpenAI-format request to Anthropic's /v1/messages and back."""
    messages = body.get("messages", [])
    system_text = next((m["content"] for m in messages if m.get("role") == "system"), None)
    user_messages = [m for m in messages if m.get("role") != "system"]

    ant_body = {
        "model": body.get("model", ANTHROPIC_CHAT_MODEL),
        "max_tokens": body.get("max_tokens", 4096),
        "messages": user_messages,
    }
    if system_text:
        ant_body["system"] = system_text

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    if body.get("stream"):
        ant_body["stream"] = True

        async def stream():
            try:
                async with httpx.AsyncClient(timeout=120) as c:
                    async with c.stream(
                        "POST",
                        "https://api.anthropic.com/v1/messages",
                        json=ant_body,
                        headers=headers,
                    ) as resp:
                        if resp.status_code != 200:
                            err = await resp.aread()
                            logger.error("Anthropic error %s: %s", resp.status_code, err[:200])
                            yield f'data: {{"error": "Anthropic error {resp.status_code}"}}\n\n'.encode()
                            return
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            raw = line[6:]
                            if raw == "[DONE]":
                                yield b"data: [DONE]\n\n"
                                continue
                            try:
                                evt = json.loads(raw)
                                if evt.get("type") == "content_block_delta":
                                    delta_text = evt.get("delta", {}).get("text", "")
                                    chunk = {"choices": [{"delta": {"content": delta_text}, "finish_reason": None}]}
                                    yield f"data: {json.dumps(chunk)}\n\n".encode()
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
            resp = await c.post(
                "https://api.anthropic.com/v1/messages",
                json=ant_body,
                headers=headers,
            )
            resp.raise_for_status()
            ant = resp.json()
            content = ant.get("content", [{}])[0].get("text", "")
            usage = ant.get("usage", {})
            return JSONResponse(content={
                "id": ant.get("id", ""),
                "object": "chat.completion",
                "model": ant.get("model", ant_body["model"]),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": ant.get("stop_reason", "stop"),
                }],
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                },
            })


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    models = []
    if OPENAI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                resp = await c.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                )
                if resp.status_code == 200:
                    models = resp.json().get("data", [])
        except Exception as exc:
            logger.warning("Could not fetch OpenAI model list: %s", exc)

    if not models:
        models = [
            {"id": OPENAI_CHAT_MODEL,    "object": "model", "owned_by": "openai"},
            {"id": ANTHROPIC_CHAT_MODEL, "object": "model", "owned_by": "anthropic"},
            {"id": "openclaw",           "object": "model", "owned_by": "eedgeai"},
        ]

    return {"object": "list", "data": models}


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """Proxy embedding requests to OpenAI so OpenWebUI's built-in RAG works."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            resp = await c.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
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
    body = await request.json()
    messages = list(body.get("messages", []))

    # ── RAG context retrieval ──────────────────────────────────────────────────
    if ENABLE_RAG and messages:
        user_text = last_user_text(messages)
        qdrant_hits, neo4j_hits = [], []

        if user_text:
            vector = await embed_text(user_text)
            if vector:
                qdrant_hits = await qdrant_search(vector)
                logger.info("Qdrant: %d chunk(s) retrieved", len(qdrant_hits))

            neo4j_hits = await neo4j_search(user_text)
            logger.info("Neo4j:  %d node(s) retrieved", len(neo4j_hits))

        messages = inject_context(messages, qdrant_hits, neo4j_hits)
        body = {**body, "messages": messages}

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
