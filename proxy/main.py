"""
Graph-Centric Orchestrated Retrieval (GCOR) proxy — Cognitive Infrastructure.

Architecture
────────────
  Neo4j   = cognitive backbone   (primary truth, structure, reasoning)
  Qdrant  = semantic perception  (index over Neo4j nodes, similarity only)

─────────────────────────────────────────────────────────────────────────────
Cognitive infrastructure additions over baseline GCOR:

  Confidence filtering
    • Qdrant hits below CONFIDENCE_THRESHOLD are dropped before expansion
    • Neo4j expansion uses $min_conf to filter Memory/Inference/Belief

  Temporal validity
    • Qdrant hits with expired valid_to are dropped (dead knowledge)
    • Neo4j expansion passes $now so queries respect valid_to

  Agent partitioning
    • AGENT_ID env var scopes Memory/Belief/Inference to one agent
    • Access-control filter drops hits whose access_level forbids $AGENT_ID

  New intents
    • inference   → resolve :Inference chains and their evidence
    • belief      → retrieve :Belief nodes and contradiction graph

─────────────────────────────────────────────────────────────────────────────
Retrieval flow for every /v1/chat/completions request
─────────────────────────────────────────────────────
  1. INTENT CLASSIFICATION
     factual | planning | dependency | memory | semantic | inference | belief

  2. SEMANTIC PHASE  (Qdrant)
     Embed → top-K candidate nodes
     Filter by: score, confidence, temporal validity, access control

  3. STRUCTURAL PHASE  (Neo4j)
     Intent-specific Cypher with confidence + temporal + agent_id params

  4. REFLECTION CHECK
     Neo4j empty but Qdrant has hits → fall back to chunk text
     Both empty → LLM general knowledge

  5. CONTEXT INJECTION
     Structured system message including confidence scores

Other endpoints
───────────────
  GET  /v1/models      → live OpenAI model list (with fallback)
  POST /v1/embeddings  → proxied to OpenAI
  GET  /health         → liveness check
"""

import hashlib
import io
import json
import logging
import os
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from neo4j import AsyncGraphDatabase

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# ── LLM backends ───────────────────────────────────────────────────────────────
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL    = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_CHAT_MODEL = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-sonnet-4-6")
LLM_BACKEND          = os.getenv("LLM_BACKEND", "openai")

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── Qdrant (semantic perception layer) ────────────────────────────────────────
QDRANT_URL        = os.getenv("QDRANT_URL",        "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
QDRANT_TOP_K      = int(os.getenv("QDRANT_TOP_K",  "8"))

# ── Neo4j (cognitive backbone) ────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://neo4j:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() in ("true", "1", "yes")

# ── Cognitive infrastructure knobs ────────────────────────────────────────────
# Minimum confidence (0.0–1.0) — hits below this are discarded
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.0"))
# Agent id scopes memory/belief/inference retrieval to a single partition
AGENT_ID = os.getenv("AGENT_ID", "")


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
    "inference":   ["infer", "conclude", "deduce", "reasoning", "derive",
                    "implication", "therefore", "follows that", "suggests",
                    "implies", "logical", "because of"],
    "belief":      ["believe", "think", "opinion", "doubt", "uncertain",
                    "probably", "likely", "assume", "suppose", "suspect",
                    "my view", "i think"],
}


def classify_intent(query: str) -> str:
    q = query.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent
    return "semantic"


# ── Step 2a: Semantic phase (Qdrant) ──────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    """Return top-K Qdrant hits; each hit includes the full payload."""
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


def _filter_hits(hits: list) -> list:
    """
    Apply cognitive filters to raw Qdrant hits:
      1. Confidence threshold — payload.confidence below minimum is discarded
      2. Temporal validity   — expired valid_to is discarded
      3. Access control      — access_level incompatible with AGENT_ID is discarded
    """
    now = _now_iso()
    filtered = []
    for h in hits:
        p = h.get("payload") or {}

        # ── confidence ────────────────────────────────────────────────────────
        node_confidence = p.get("confidence")
        if node_confidence is not None and node_confidence < CONFIDENCE_THRESHOLD:
            logger.debug("Dropping hit %s: confidence %.2f < %.2f",
                         h.get("id"), node_confidence, CONFIDENCE_THRESHOLD)
            continue

        # ── temporal validity ─────────────────────────────────────────────────
        valid_to = p.get("valid_to")
        if valid_to and valid_to < now:
            logger.debug("Dropping hit %s: expired valid_to %s", h.get("id"), valid_to)
            continue

        # ── access control ────────────────────────────────────────────────────
        access_level = p.get("access_level", "public")
        if access_level == "restricted":
            logger.debug("Dropping hit %s: restricted access", h.get("id"))
            continue
        if access_level.startswith("agent:") and AGENT_ID:
            owner = access_level.split(":", 1)[1]
            if owner != AGENT_ID:
                logger.debug("Dropping hit %s: owned by %s, not %s", h.get("id"), owner, AGENT_ID)
                continue

        filtered.append(h)
    return filtered


# ── Step 2b: Structural phase (Neo4j) ─────────────────────────────────────────

_EXPAND_CYPHER = {
    "factual": """
        MATCH (n) WHERE elementId(n) IN $ids
        OPTIONAL MATCH (n)-[r]-(related)
        OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document)
        RETURN
            properties(n) AS node, labels(n) AS labels,
            properties(doc) AS document,
            collect(DISTINCT {
                rel: type(r), props: properties(related), lbls: labels(related)
            })[..6] AS related
        LIMIT 20
    """,
    "planning": """
        MATCH (n) WHERE elementId(n) IN $ids
        OPTIONAL MATCH (n)-[:DEPENDS_ON*1..2]->(dep)
        OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document)
        OPTIONAL MATCH (g:Goal)-[:DEPENDS_ON]->(n)
        OPTIONAL MATCH (inf:Inference)-[:SUPPORTS]->(n)
            WHERE inf.confidence >= $min_conf
        RETURN
            properties(n) AS node, labels(n) AS labels,
            properties(doc) AS document,
            collect(DISTINCT properties(dep))[..5]   AS dependencies,
            collect(DISTINCT properties(g))[..5]     AS blocking_goals,
            collect(DISTINCT properties(inf))[..3]   AS supporting_inferences
        LIMIT 20
    """,
    "dependency": """
        MATCH (n) WHERE elementId(n) IN $ids
        OPTIONAL MATCH path = (n)-[:DEPENDS_ON*1..3]->(dep)
        OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document)
        RETURN
            properties(n)  AS node, labels(n) AS labels,
            properties(doc) AS document,
            [x IN nodes(path) | properties(x)][..8] AS dependency_chain
        LIMIT 20
    """,
    "memory": """
        MATCH (n) WHERE elementId(n) IN $ids
        OPTIONAL MATCH (n)-[:MENTIONS]->(concept:Concept)
        OPTIONAL MATCH (mem:Memory)-[:ABOUT]->(concept)
            WHERE ($agent_id = '' OR mem.agent_id = $agent_id)
              AND mem.confidence >= $min_conf
              AND (mem.valid_to IS NULL OR mem.valid_to >= $now)
        OPTIONAL MATCH (bel:Belief)-[:ABOUT]->(concept)
            WHERE ($agent_id = '' OR bel.agent_id = $agent_id)
              AND bel.confidence >= $min_conf
              AND (bel.valid_to IS NULL OR bel.valid_to >= $now)
        OPTIONAL MATCH (evt:Event)-[:INVOLVES]->(concept)
        OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document)
        RETURN
            properties(n)   AS node, labels(n) AS labels,
            properties(doc) AS document,
            collect(DISTINCT properties(concept))[..5] AS concepts,
            collect(DISTINCT properties(mem))[..5]     AS memories,
            collect(DISTINCT properties(bel))[..3]     AS beliefs,
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
            properties(n)   AS node, labels(n) AS labels,
            properties(doc) AS document,
            collect(DISTINCT properties(concept))[..5]  AS concepts,
            collect(DISTINCT properties(neighbor))[..3] AS neighbors
        LIMIT 20
    """,
    "inference": """
        MATCH (n) WHERE elementId(n) IN $ids
        OPTIONAL MATCH (n)-[:DERIVED_FROM]->(src)
        OPTIONAL MATCH (n)-[:SUPPORTS]->(tgt)
        OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document)
        OPTIONAL MATCH (downstream:Inference)-[:DERIVED_FROM]->(n)
            WHERE downstream.confidence >= $min_conf
              AND ($agent_id = '' OR downstream.agent_id = $agent_id)
              AND (downstream.valid_to IS NULL OR downstream.valid_to >= $now)
        RETURN
            properties(n)   AS node, labels(n) AS labels,
            properties(doc) AS document,
            collect(DISTINCT properties(src))[..5]        AS sources,
            collect(DISTINCT properties(tgt))[..5]        AS supports,
            collect(DISTINCT properties(downstream))[..3] AS downstream_inferences
        LIMIT 20
    """,
    "belief": """
        MATCH (n) WHERE elementId(n) IN $ids
        OPTIONAL MATCH (a:Agent)-[:HOLDS]->(n)
        OPTIONAL MATCH (n)-[:ABOUT]->(concept:Concept)
        OPTIONAL MATCH (n)-[:CONTRADICTS]-(other:Belief)
            WHERE other.confidence >= $min_conf
        OPTIONAL MATCH (inf:Inference)-[:SUPPORTS]->(n)
            WHERE inf.confidence >= $min_conf
              AND (inf.valid_to IS NULL OR inf.valid_to >= $now)
        RETURN
            properties(n)   AS node, labels(n) AS labels, null AS document,
            properties(a)   AS agent,
            collect(DISTINCT properties(concept))[..5] AS concepts,
            collect(DISTINCT properties(other))[..3]   AS contradictions,
            collect(DISTINCT properties(inf))[..3]     AS supporting_inferences
        LIMIT 20
    """,
}

_FALLBACK_CYPHER = """
    MATCH (n)
    WHERE any(k IN keys(n) WHERE toLower(toString(n[k])) CONTAINS toLower($q))
      AND coalesce(n.confidence, 1.0) >= $min_conf
      AND (n.valid_to IS NULL OR n.valid_to >= $now)
    RETURN properties(n) AS node, labels(n) AS labels, null AS document
    LIMIT 10
"""


async def neo4j_expand(
    element_ids: list,
    intent: str,
    query: str,
) -> list:
    """Expand the graph around Qdrant candidate nodes based on intent."""
    cypher = _EXPAND_CYPHER.get(intent, _EXPAND_CYPHER["semantic"])
    now = _now_iso()
    params = {
        "ids":      element_ids,
        "agent_id": AGENT_ID,
        "min_conf": CONFIDENCE_THRESHOLD,
        "now":      now,
    }

    if not element_ids:
        cypher = _FALLBACK_CYPHER
        params  = {"q": query[:100], "min_conf": CONFIDENCE_THRESHOLD, "now": now}

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
    if graph_records:
        return graph_records
    if qdrant_hits:
        logger.info("Reflection: Neo4j empty, falling back to Qdrant chunk text.")
        return [
            {
                "node": {
                    "text":        h["payload"].get("text", ""),
                    "document_id": h["payload"].get("document_id", ""),
                    "confidence":  h["payload"].get("confidence"),
                    "valid_from":  h["payload"].get("valid_from"),
                    "valid_to":    h["payload"].get("valid_to"),
                },
                "labels":   [h["payload"].get("node_type", "Chunk")],
                "document": {"title": h["payload"].get("document_title", "")},
            }
            for h in qdrant_hits
            if h.get("payload", {}).get("text")
        ]
    return []


# ── Context builder ────────────────────────────────────────────────────────────

def _confidence_badge(props: dict) -> str:
    c = props.get("confidence")
    if c is None:
        return ""
    pct = int(c * 100)
    if pct >= 90:
        tier = "high"
    elif pct >= 60:
        tier = "medium"
    else:
        tier = "low"
    return f"  [confidence: {pct}% / {tier}]"


def _temporal_badge(props: dict) -> str:
    parts = []
    vf = props.get("valid_from")
    vt = props.get("valid_to")
    if vf:
        parts.append(f"from {vf[:10]}")
    if vt:
        parts.append(f"to {vt[:10]}")
    return f"  [valid {', '.join(parts)}]" if parts else ""


def build_gcor_context(intent: str, qdrant_hits: list, graph_records: list) -> str:
    records = _reflection_fallback(qdrant_hits, graph_records)
    if not records:
        return ""

    confidence_note = (
        f"confidence ≥ {int(CONFIDENCE_THRESHOLD*100)}% "
        if CONFIDENCE_THRESHOLD > 0 else ""
    )
    agent_note = f"agent partition: {AGENT_ID} " if AGENT_ID else ""

    lines = [
        f"## Retrieved Context  [intent: {intent} | "
        f"{len(qdrant_hits)} semantic candidates → {len(records)} graph records"
        + (f" | {confidence_note}{agent_note}".rstrip() if (confidence_note or agent_note) else "")
        + "]",
        "",
        "Neo4j is the primary source of truth. Qdrant identified the entry points.",
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
            node.get("subject") or str(node)
        )[:600]

        conf_badge    = _confidence_badge(node)
        temporal_badge = _temporal_badge(node)
        lines.append(f"### [{node_type}]{conf_badge}{temporal_badge}")
        lines.append(content)

        if doc.get("title") or doc.get("source"):
            lines.append(f"Source: {doc.get('title') or doc.get('source', '')}")

        # Reasoning trace for Inference nodes
        if node.get("reasoning_trace"):
            lines.append(f"Reasoning: {node['reasoning_trace'][:300]}")

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
            inferences = [i for i in (rec.get("supporting_inferences") or []) if i]
            if inferences:
                lines.append("Supporting inferences: " + "; ".join(
                    f"{i.get('text', '')[:80]} ({int(i.get('confidence', 0)*100)}%)"
                    for i in inferences
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
                lines.append("Memories: " + "; ".join(
                    f"{m.get('content', '')[:80]}{_confidence_badge(m)}"
                    for m in mems[:3]
                ))
            beliefs = [b for b in (rec.get("beliefs") or []) if b]
            if beliefs:
                lines.append("Beliefs: " + "; ".join(
                    f"{b.get('content', '')[:80]}{_confidence_badge(b)}"
                    for b in beliefs[:3]
                ))
            evts = [e for e in (rec.get("events") or []) if e]
            if evts:
                lines.append("Events: " + "; ".join(
                    f"{e.get('timestamp', '')[:10]} {e.get('description', '')}"
                    for e in evts[:3]
                ))

        elif intent == "inference":
            srcs = [s for s in (rec.get("sources") or []) if s]
            if srcs:
                lines.append("Derived from: " + "; ".join(
                    (s.get("text") or s.get("content") or str(s))[:80]
                    for s in srcs[:3]
                ))
            supports = [t for t in (rec.get("supports") or []) if t]
            if supports:
                lines.append("Supports: " + "; ".join(
                    (t.get("description") or t.get("content") or str(t))[:80]
                    for t in supports[:3]
                ))
            downstream = [d for d in (rec.get("downstream_inferences") or []) if d]
            if downstream:
                lines.append("Downstream inferences: " + "; ".join(
                    f"{d.get('text', '')[:60]}{_confidence_badge(d)}"
                    for d in downstream[:2]
                ))

        elif intent == "belief":
            contradictions = [c for c in (rec.get("contradictions") or []) if c]
            if contradictions:
                lines.append("Contradicts: " + "; ".join(
                    f"{c.get('content', '')[:80]}{_confidence_badge(c)}"
                    for c in contradictions[:2]
                ))
            inferences = [i for i in (rec.get("supporting_inferences") or []) if i]
            if inferences:
                lines.append("Supported by: " + "; ".join(
                    f"{i.get('text', '')[:60]}{_confidence_badge(i)}"
                    for i in inferences[:2]
                ))
            agent = rec.get("agent") or {}
            if agent.get("id"):
                lines.append(f"Held by agent: {agent['id']}")

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
        "Respect the confidence scores — lower-confidence information should be "
        "presented with appropriate uncertainty. If the context is insufficient, "
        "apply general knowledge."
    )
    return "\n".join(lines)


def inject_context(messages: list, context: str) -> list:
    if not context:
        return messages
    result, merged = [], False
    for m in messages:
        if m.get("role") == "system" and not merged:
            result.append({"role": "system",
                           "content": context + "\n\n---\n\n" + m.get("content", "")})
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
                "usage": {"prompt_tokens":    usage.get("input_tokens", 0),
                          "completion_tokens": usage.get("output_tokens", 0),
                          "total_tokens":      usage.get("input_tokens", 0) + usage.get("output_tokens", 0)},
            })


# ── Routes ─────────────────────────────────────────────────────────────────────

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
        ]
    # Always surface the cognitive RAG model at the top of the list
    models = [{"id": "openclaw", "object": "model", "created": 1700000000, "owned_by": "eedgeai"}] + models
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
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                         "Content-Type": "application/json"},
                json=body,
            )
            resp.raise_for_status()
            return JSONResponse(content=resp.json())
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code,
                            detail="Embeddings upstream error")
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
            # ── Step 1: Intent ──────────────────────────────────────────────
            intent = classify_intent(query)
            logger.info("GCOR intent: %s", intent)

            # ── Step 2a: Semantic (Qdrant → element IDs) ────────────────────
            vector      = await embed_text(query)
            raw_hits    = await qdrant_search(vector) if vector else []
            qdrant_hits = _filter_hits(raw_hits)
            element_ids = [
                h["payload"]["neo4j_element_id"]
                for h in qdrant_hits
                if h.get("payload", {}).get("neo4j_element_id")
            ]
            logger.info(
                "Qdrant: %d raw → %d after cognitive filters, %d with neo4j_element_id",
                len(raw_hits), len(qdrant_hits), len(element_ids),
            )

            # ── Step 2b: Structural (Neo4j graph expansion) ─────────────────
            graph_records = await neo4j_expand(element_ids, intent, query)
            logger.info("Neo4j: %d graph records expanded", len(graph_records))

            # ── Step 3: Reflection + context build ──────────────────────────
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


_TEMPLATES = os.path.join(os.path.dirname(__file__), "templates")


# ── Knowledge UI ───────────────────────────────────────────────────────────────

@app.get("/", response_class=RedirectResponse, status_code=302)
async def root():
    return "/knowledge"


@app.get("/knowledge", response_class=HTMLResponse)
async def knowledge_ui():
    with open(os.path.join(_TEMPLATES, "knowledge.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/collections")
async def api_collections():
    """List all Qdrant collections with Neo4j document stats."""
    # Fetch Qdrant collections
    collections = []
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            resp = await c.get(f"{QDRANT_URL}/collections")
            resp.raise_for_status()
            for col in resp.json().get("result", {}).get("collections", []):
                name = col["name"]
                info_resp = await c.get(f"{QDRANT_URL}/collections/{name}")
                info = info_resp.json().get("result", {})
                collections.append({
                    "name":            name,
                    "points_count":    info.get("points_count", 0),
                    "doc_count":       0,
                    "recent_docs":     [],
                    "embedding_model": EMBEDDING_MODEL,
                })
    except Exception as exc:
        logger.warning("Qdrant collection fetch failed: %s", exc)

    # If the configured collection is missing from Qdrant, still show it
    names = {c["name"] for c in collections}
    if QDRANT_COLLECTION not in names:
        collections.insert(0, {
            "name":            QDRANT_COLLECTION,
            "points_count":    0,
            "doc_count":       0,
            "recent_docs":     [],
            "embedding_model": EMBEDDING_MODEL,
        })

    # Enrich with Neo4j document counts
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        async with driver.session(database=NEO4J_DATABASE) as s:
            result = await s.run(
                "MATCH (d:Document) "
                "RETURN d.document_id AS doc_id, d.title AS title, "
                "       d.created_at AS created_at, d.access_level AS access_level "
                "ORDER BY d.created_at DESC LIMIT 100"
            )
            records = await result.data()
        doc_count   = len(records)
        recent_docs = [
            {"doc_id": r["doc_id"], "title": r["title"] or r["doc_id"],
             "created_at": r["created_at"], "access_level": r["access_level"]}
            for r in records[:5]
        ]
        for col in collections:
            if col["name"] == QDRANT_COLLECTION:
                col["doc_count"]   = doc_count
                col["recent_docs"] = recent_docs
    except Exception as exc:
        logger.warning("Neo4j doc count failed: %s", exc)
    finally:
        await driver.close()

    return {"collections": collections, "embedding_model": EMBEDDING_MODEL}


@app.get("/api/collections/{name}/docs")
async def api_collection_docs(name: str):
    """List all Document nodes in Neo4j for a given collection."""
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        async with driver.session(database=NEO4J_DATABASE) as s:
            result = await s.run(
                "MATCH (d:Document) "
                "OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk) "
                "RETURN d.document_id AS doc_id, d.title AS title, "
                "       d.created_at AS created_at, d.access_level AS access_level, "
                "       count(c) AS chunk_count "
                "ORDER BY d.created_at DESC LIMIT 200"
            )
            records = await result.data()
        docs = [
            {"doc_id": r["doc_id"], "title": r["title"] or r["doc_id"],
             "created_at": r["created_at"], "access_level": r["access_level"],
             "chunk_count": r["chunk_count"]}
            for r in records
        ]
        return {"collection": name, "docs": docs}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    finally:
        await driver.close()


@app.get("/api/search")
async def api_search(
    collection: str = Query(...),
    q: str = Query(...),
    top_k: int = Query(8),
):
    """Semantic search against a Qdrant collection."""
    vector = await embed_text(q)
    if not vector:
        raise HTTPException(status_code=503, detail="Embedding unavailable — check OPENAI_API_KEY")
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            resp = await c.post(
                f"{QDRANT_URL}/collections/{collection}/points/search",
                json={"vector": vector, "limit": top_k, "with_payload": True},
            )
            if resp.status_code == 404:
                return {"results": [], "error": f"Collection '{collection}' not found"}
            resp.raise_for_status()
            hits = resp.json().get("result", [])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    results = [
        {
            "score":            h.get("score", 0),
            "text":             h.get("payload", {}).get("text", ""),
            "neo4j_element_id": h.get("payload", {}).get("neo4j_element_id", ""),
            "doc_title":        h.get("payload", {}).get("document_title", ""),
            "position":         h.get("payload", {}).get("position"),
        }
        for h in hits
    ]
    return {"collection": collection, "query": q, "results": results}


@app.post("/api/ingest")
async def api_ingest(
    file: UploadFile = File(...),
    title: str = Form(""),
    agent_id: str = Form(""),
    access_level: str = Form("public"),
):
    """JSON-returning ingest endpoint used by the knowledge UI."""
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _INGEST_ACCEPT:
        raise HTTPException(status_code=415, detail=f"Unsupported type '{ext}'")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    doc_title = title.strip() or filename
    doc_id    = hashlib.md5(f"{filename}-{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    try:
        text = _extract_text(filename, data).strip()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Extraction failed: {exc}")
    if not text:
        raise HTTPException(status_code=422, detail="No text extracted")
    chunks = _chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks produced")
    logger.info("API ingest '%s': %d chunks", doc_title, len(chunks))
    try:
        _, chunk_eids = await _neo4j_ingest(doc_id, doc_title, filename, chunks, agent_id, access_level)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Neo4j: {exc}")
    try:
        upserted = await _qdrant_ingest(QDRANT_COLLECTION, chunks, chunk_eids, doc_id, doc_title, agent_id, access_level)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant: {exc}")
    return {"status": "ok", "document_id": doc_id, "title": doc_title,
            "filename": filename, "chunks": len(chunks), "qdrant_points": upserted,
            "collection": QDRANT_COLLECTION}


@app.get("/health")
async def health():
    return {
        "status":               "ok",
        "rag":                  ENABLE_RAG,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "agent_id":             AGENT_ID or None,
    }


# ── Document Ingest ────────────────────────────────────────────────────────────

_INGEST_ACCEPT = {".txt", ".md", ".pdf", ".docx", ".json", ".csv"}

_UPLOAD_FORM = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>EedgeAI — Ingest Document</title>
  <style>
    body{{font-family:system-ui,sans-serif;max-width:640px;margin:60px auto;padding:0 24px;background:#0f172a;color:#e2e8f0}}
    h1{{color:#38bdf8;margin-bottom:4px}}
    p{{color:#94a3b8;margin-top:0}}
    form{{background:#1e293b;padding:28px;border-radius:12px;margin-top:24px}}
    label{{display:block;font-size:.85rem;color:#94a3b8;margin-bottom:4px}}
    input,select{{width:100%;padding:8px 12px;border:1px solid #334155;border-radius:6px;
      background:#0f172a;color:#e2e8f0;font-size:.9rem;box-sizing:border-box;margin-bottom:16px}}
    button{{background:#0ea5e9;color:#fff;border:none;padding:10px 28px;
      border-radius:6px;font-size:1rem;cursor:pointer;width:100%}}
    button:hover{{background:#38bdf8}}
    .hint{{font-size:.78rem;color:#64748b;margin-top:-12px;margin-bottom:16px}}
    .accepted{{color:#64748b;font-size:.8rem;margin-top:16px}}
  </style>
</head>
<body>
  <h1>Ingest Document</h1>
  <p>Upload a file to index it into Neo4j + Qdrant for GCOR retrieval.</p>
  <form method="POST" enctype="multipart/form-data">
    <label>File</label>
    <input type="file" name="file" accept=".txt,.md,.pdf,.docx,.json,.csv" required>
    <p class="accepted">Accepted: .txt .md .pdf .docx .json .csv</p>
    <label>Title (optional — defaults to filename)</label>
    <input type="text" name="title" placeholder="My Document">
    <label>Agent ID (optional — leave blank for shared knowledge)</label>
    <input type="text" name="agent_id" placeholder="">
    <label>Access level</label>
    <select name="access_level">
      <option value="public">public</option>
      <option value="restricted">restricted</option>
    </select>
    <button type="submit">Ingest</button>
  </form>
</body>
</html>"""

_RESULT_TMPL = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Ingest result</title>
  <style>
    body{{font-family:system-ui,sans-serif;max-width:640px;margin:60px auto;padding:0 24px;background:#0f172a;color:#e2e8f0}}
    h1{{color:{color}}}
    pre{{background:#1e293b;padding:20px;border-radius:8px;overflow-x:auto;font-size:.85rem}}
    a{{color:#38bdf8}}
  </style>
</head>
<body>
  <h1>{heading}</h1>
  <pre>{body}</pre>
  <p><a href="/ingest">&larr; Ingest another</a></p>
</body>
</html>"""


def _extract_text(filename: str, data: bytes) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    if ext == ".docx":
        from docx import Document as DocxDocument
        doc = DocxDocument(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    if ext == ".json":
        try:
            return json.dumps(json.loads(data), indent=2)
        except Exception:
            pass
    return data.decode("utf-8", errors="replace")


def _chunk_text(text: str, size: int = 2000, overlap: int = 200) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for sep in ("\n\n", "\n", " "):
                pos = text.rfind(sep, start + size // 2, end)
                if pos != -1:
                    end = pos
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start >= len(text) - overlap:
            break
    return chunks


async def _neo4j_ingest(doc_id: str, title: str, source: str,
                        chunks: list[str], agent_id: str, access_level: str):
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    now = _now_iso()
    try:
        async with driver.session(database=NEO4J_DATABASE) as s:
            doc_rec = await s.run(
                """CREATE (d:Document {
                     document_id: $doc_id, title: $title, source: $source,
                     agent_id: $agent_id, access_level: $access_level,
                     chunk_count: $cc, created_at: $now
                   }) RETURN elementId(d) AS eid""",
                doc_id=doc_id, title=title, source=source,
                agent_id=agent_id, access_level=access_level,
                cc=len(chunks), now=now,
            )
            doc_eid = (await doc_rec.single())["eid"]

            chunk_eids = []
            for i, text in enumerate(chunks):
                cid = f"{doc_id}-chunk-{i}"
                c_rec = await s.run(
                    """MATCH (d:Document {document_id: $doc_id})
                       CREATE (c:Chunk {
                         chunk_id: $cid, text: $text, position: $pos,
                         document_id: $doc_id, document_title: $title,
                         agent_id: $agent_id, access_level: $access_level,
                         confidence: 1.0, created_at: $now
                       })
                       CREATE (d)-[:CONTAINS]->(c)
                       RETURN elementId(c) AS eid""",
                    doc_id=doc_id, cid=cid, text=text, pos=i,
                    title=title, agent_id=agent_id, access_level=access_level, now=now,
                )
                chunk_eids.append((await c_rec.single())["eid"])
        return doc_eid, chunk_eids
    finally:
        await driver.close()


async def _qdrant_ingest(collection: str, chunks: list[str], chunk_eids: list[str],
                         doc_id: str, title: str, agent_id: str, access_level: str):
    now = _now_iso()

    # Embed in batches of 96
    all_vectors: list[list[float]] = []
    for i in range(0, len(chunks), 96):
        batch = chunks[i:i + 96]
        async with httpx.AsyncClient(timeout=60) as c:
            resp = await c.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={"input": batch, "model": EMBEDDING_MODEL},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            data.sort(key=lambda x: x["index"])
            all_vectors.extend(d["embedding"] for d in data)

    # Ensure collection exists
    async with httpx.AsyncClient(timeout=30) as c:
        chk = await c.get(f"{QDRANT_URL}/collections/{collection}")
        if chk.status_code == 404:
            await c.put(
                f"{QDRANT_URL}/collections/{collection}",
                json={"vectors": {"size": len(all_vectors[0]), "distance": "Cosine"}},
            )

    # Upsert points
    points = []
    for i, (text, eid, vec) in enumerate(zip(chunks, chunk_eids, all_vectors)):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, eid))
        points.append({
            "id":      point_id,
            "vector":  vec,
            "payload": {
                "text":             text,
                "neo4j_element_id": eid,
                "node_type":        "Chunk",
                "document_id":      doc_id,
                "document_title":   title,
                "position":         i,
                "agent_id":         agent_id,
                "access_level":     access_level,
                "confidence":       1.0,
                "valid_from":       now,
                "valid_to":         None,
            },
        })

    async with httpx.AsyncClient(timeout=60) as c:
        resp = await c.put(
            f"{QDRANT_URL}/collections/{collection}/points",
            params={"wait": "true"},
            json={"points": points},
        )
        resp.raise_for_status()

    return len(points)


@app.get("/ingest", response_class=HTMLResponse)
async def ingest_form():
    return _UPLOAD_FORM


@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    title: str = Form(""),
    agent_id: str = Form(""),
    access_level: str = Form("public"),
):
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _INGEST_ACCEPT:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Accepted: {', '.join(sorted(_INGEST_ACCEPT))}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    doc_title = title.strip() or filename
    doc_id = hashlib.md5(f"{filename}-{datetime.now().isoformat()}".encode()).hexdigest()[:16]

    try:
        text = _extract_text(filename, data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {exc}")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="No text could be extracted from the file")

    chunks = _chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="File produced no chunks after extraction")

    logger.info("Ingest '%s': %d chars → %d chunks", doc_title, len(text), len(chunks))

    try:
        doc_eid, chunk_eids = await _neo4j_ingest(
            doc_id, doc_title, filename, chunks, agent_id, access_level
        )
    except Exception as exc:
        logger.error("Neo4j ingest failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Neo4j ingest failed: {exc}")

    try:
        upserted = await _qdrant_ingest(
            QDRANT_COLLECTION, chunks, chunk_eids, doc_id, doc_title, agent_id, access_level
        )
    except Exception as exc:
        logger.error("Qdrant ingest failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Qdrant ingest failed: {exc}")

    result = {
        "status":        "ok",
        "document_id":   doc_id,
        "document_eid":  doc_eid,
        "title":         doc_title,
        "filename":      filename,
        "chunks":        len(chunks),
        "qdrant_points": upserted,
        "collection":    QDRANT_COLLECTION,
    }
    logger.info("Ingest complete: %s", result)

    return HTMLResponse(_RESULT_TMPL.format(
        color="#4ade80",
        heading=f"✓ Ingested \"{doc_title}\"",
        body=json.dumps(result, indent=2),
    ))
