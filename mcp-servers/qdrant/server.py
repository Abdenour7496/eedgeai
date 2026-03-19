"""
Qdrant MCP server — GCOR Cognitive Infrastructure edition.

GCOR principle: Qdrant is a semantic index over Neo4j.
Every point carries neo4j_element_id (and cognitive metadata) so the proxy
can hop from a semantic hit straight back to the graph.

─────────────────────────────────────────────────────────────────────────────
Cognitive infrastructure additions:

  Confidence scoring
    • confidence (0.0–1.0) stored in every point's payload
    • qdrant_search filters by min_confidence parameter

  Temporal validity
    • valid_from / valid_to in every point's payload
    • qdrant_search_temporal — filter to a specific time window
    • Expired points are returned but flagged (the proxy drops them)

  Agent partitioning
    • agent_id in payload — restricts search to one agent's cognitive space
    • qdrant_search and qdrant_search_temporal both accept agent_id filter

  Access control
    • access_level ("public" | "restricted" | "agent:<id>") in payload
    • qdrant_search accepts access_level filter

  Self-reflective node types
    • node_type supports: Chunk | Memory | Inference | Belief | Goal | Event
    • All GCOR-aware tools filter by node_type

─────────────────────────────────────────────────────────────────────────────
Tools
  qdrant_list_collections      list all collections
  qdrant_collection_info       vector size, point count, status
  qdrant_count                 count points (optionally by node_type / agent_id)
  qdrant_search                semantic search with cognitive filters
  qdrant_search_temporal       search restricted to a temporal validity window
  qdrant_search_by_neo4j_id    look up Qdrant point by graph element ID
  qdrant_upsert_chunk          GCOR-aware upsert (all cognitive fields)
  qdrant_upsert                free-form upsert (ad-hoc / legacy)
  qdrant_delete_points         delete by IDs
  qdrant_delete_collection     drop a collection
  qdrant_gcor_status           health check: counts per node_type
  sync_neo4j_to_qdrant         pull nodes from Neo4j → embed → index with full
                               cognitive payload (confidence, temporal, ACL)

Exposed to OpenClaw via SSE on 0.0.0.0:8765.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone

import httpx
from mcp.server.fastmcp import FastMCP
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    Range,
    VectorParams,
)

# ── Config ─────────────────────────────────────────────────────────────────────
QDRANT_URL      = os.getenv("QDRANT_URL",        "http://qdrant:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY",    "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME",   "documents")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY",    "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL",   "text-embedding-3-small")
MAX_RETRY       = int(os.getenv("QDRANT_MAX_RETRIES", "3"))

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://neo4j:7687")
NEO4J_USER     = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

HOST = os.getenv("QDRANT_MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("QDRANT_MCP_PORT", "8765"))

DEFAULT_ACCESS = os.getenv("DEFAULT_ACCESS_LEVEL", "public")

# ── Qdrant client (singleton) ─────────────────────────────────────────────────
_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
            timeout=10,
        )
    return _client


# ── Retry helper ───────────────────────────────────────────────────────────────
async def with_retry(coro_fn, retries: int = MAX_RETRY):
    for attempt in range(retries):
        try:
            return await coro_fn()
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)


# ── Embedding ──────────────────────────────────────────────────────────────────
async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set — cannot embed text")
    async with httpx.AsyncClient(timeout=60) as c:
        resp = await c.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"input": [t[:8000] for t in texts], "model": EMBEDDING_MODEL},
        )
        resp.raise_for_status()
        return [d["embedding"] for d in resp.json()["data"]]


async def embed(text: str) -> list[float]:
    return (await embed_texts([text]))[0]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Collection bootstrap ───────────────────────────────────────────────────────
async def ensure_collection(name: str, vector_size: int) -> None:
    client = get_client()
    try:
        await client.get_collection(name)
    except Exception:
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


# ── Payload builder ────────────────────────────────────────────────────────────
def _build_cognitive_payload(
    text: str,
    neo4j_element_id: str = "",
    node_type: str = "Chunk",
    document_id: str = "",
    document_title: str = "",
    position: int = 0,
    agent_id: str = "",
    confidence: float = 1.0,
    valid_from: str = "",
    valid_to: str = "",
    access_level: str = "",
    extra: dict = None,
) -> dict:
    now = _now()
    return {
        # GCOR graph reference
        "neo4j_element_id": neo4j_element_id,
        "node_type":        node_type,
        # Content
        "text":             text,
        "document_id":      document_id,
        "document_title":   document_title,
        "position":         position,
        # Cognitive infrastructure
        "confidence":       confidence,
        "valid_from":       valid_from or now,
        "valid_to":         valid_to or None,
        "agent_id":         agent_id,
        "access_level":     access_level or DEFAULT_ACCESS,
        "created_at":       now,
        **(extra or {}),
    }


def _format_hit(hit) -> dict:
    """Normalise a search hit — always surface cognitive fields at top level."""
    p = hit.payload or {}
    return {
        "id":                hit.id,
        "score":             hit.score,
        # GCOR graph link
        "neo4j_element_id":  p.get("neo4j_element_id"),
        "node_type":         p.get("node_type", "Chunk"),
        # Cognitive metadata
        "confidence":        p.get("confidence"),
        "valid_from":        p.get("valid_from"),
        "valid_to":          p.get("valid_to"),
        "agent_id":          p.get("agent_id"),
        "access_level":      p.get("access_level", "public"),
        # Content
        "text":              p.get("text", ""),
        "document_id":       p.get("document_id"),
        "document_title":    p.get("document_title"),
        "position":          p.get("position"),
        "payload":           p,
    }


# ── Payload filter builder ─────────────────────────────────────────────────────
def _build_filter(
    node_type: str = "",
    agent_id: str = "",
    access_level: str = "",
    min_confidence: float = 0.0,
) -> Filter | None:
    must = []
    if node_type:
        must.append(FieldCondition(key="node_type", match=MatchValue(value=node_type)))
    if agent_id:
        must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
    if access_level:
        must.append(FieldCondition(key="access_level", match=MatchValue(value=access_level)))
    if min_confidence > 0.0:
        must.append(FieldCondition(key="confidence", range=Range(gte=min_confidence)))
    return Filter(must=must) if must else None


# ── MCP server ─────────────────────────────────────────────────────────────────
mcp = FastMCP("Qdrant-Cognitive")


@mcp.tool()
async def qdrant_list_collections() -> str:
    """List all Qdrant collections."""
    try:
        client = get_client()
        result = await with_retry(client.get_collections)
        return json.dumps([{"name": c.name} for c in result.collections], indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def qdrant_collection_info(collection: str) -> str:
    """Return detailed information about a Qdrant collection.
    Args:
        collection: Collection name.
    """
    try:
        client = get_client()
        info = await with_retry(lambda: client.get_collection(collection))
        return json.dumps({
            "name":          collection,
            "vectors_count": info.vectors_count,
            "points_count":  info.points_count,
            "status":        str(info.status),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def qdrant_count(
    collection: str,
    node_type: str = "",
    agent_id: str = "",
) -> str:
    """Return the number of points in a collection, optionally filtered.
    Args:
        collection: Collection name.
        node_type:  Optional GCOR node type filter (Chunk, Memory, Inference, Belief…).
        agent_id:   Optional — restrict to one agent's partition.
    """
    try:
        client = get_client()
        f = _build_filter(node_type=node_type, agent_id=agent_id)
        result = await with_retry(lambda: client.count(collection, count_filter=f))
        return json.dumps({
            "collection": collection,
            "node_type":  node_type or "all",
            "agent_id":   agent_id or "all",
            "count":      result.count,
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def qdrant_search(
    collection: str,
    query: str,
    limit: int = 5,
    node_type: str = "",
    agent_id: str = "",
    min_confidence: float = 0.0,
    access_level: str = "",
) -> str:
    """
    Semantic search with full cognitive filters.

    Results always include neo4j_element_id, confidence, temporal fields,
    and access_level so callers can apply GCOR graph expansion.

    Args:
        collection:     Collection to search.
        query:          Natural-language query text.
        limit:          Number of results (default 5).
        node_type:      Optional node type filter (Chunk, Memory, Inference, Belief…).
        agent_id:       Optional — restrict to one agent's partition.
        min_confidence: Minimum confidence score (0.0 = any).
        access_level:   Optional access_level filter (public | restricted | agent:<id>).
    """
    try:
        vector = await embed(query)
        client = get_client()
        f = _build_filter(
            node_type=node_type,
            agent_id=agent_id,
            access_level=access_level,
            min_confidence=min_confidence,
        )
        results = await with_retry(
            lambda: client.search(
                collection_name=collection,
                query_vector=vector,
                limit=limit,
                with_payload=True,
                query_filter=f,
            )
        )
        return json.dumps([_format_hit(r) for r in results], indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def qdrant_search_temporal(
    collection: str,
    query: str,
    at_time: str = "",
    limit: int = 5,
    node_type: str = "",
    agent_id: str = "",
    min_confidence: float = 0.0,
) -> str:
    """
    Semantic search restricted to nodes valid at a specific point in time.

    Uses Qdrant payload filters on valid_from / valid_to so only temporally
    active knowledge is returned.

    Args:
        collection:     Collection to search.
        query:          Natural-language query.
        at_time:        ISO-8601 timestamp (default: now).
        limit:          Number of results (default 5).
        node_type:      Optional node type filter.
        agent_id:       Optional agent partition filter.
        min_confidence: Minimum confidence (default 0.0).
    """
    try:
        t = at_time or _now()
        vector = await embed(query)
        client = get_client()

        # Build combined filter: cognitive fields + temporal window
        must = []
        if node_type:
            must.append(FieldCondition(key="node_type", match=MatchValue(value=node_type)))
        if agent_id:
            must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
        if min_confidence > 0.0:
            must.append(FieldCondition(key="confidence", range=Range(gte=min_confidence)))
        # valid_from <= at_time (or null)
        must.append(FieldCondition(key="valid_from", range=Range(lte=t)))

        query_filter = Filter(must=must) if must else None

        results = await with_retry(
            lambda: client.search(
                collection_name=collection,
                query_vector=vector,
                limit=limit * 3,  # over-fetch to account for valid_to filter
                with_payload=True,
                query_filter=query_filter,
            )
        )

        # Post-filter: exclude expired (valid_to < at_time)
        active = []
        for r in results:
            vt = (r.payload or {}).get("valid_to")
            if vt and vt < t:
                continue
            active.append(_format_hit(r))
            if len(active) >= limit:
                break

        return json.dumps(active, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def qdrant_search_by_neo4j_id(
    collection: str,
    neo4j_element_id: str,
) -> str:
    """
    Retrieve Qdrant point(s) that reference a specific Neo4j element ID.

    Use this in the GCOR structural phase to verify that a graph node is
    indexed and to retrieve its embedding metadata.

    Args:
        collection:       Collection name.
        neo4j_element_id: Neo4j elementId string (e.g. "4:abc:0").
    """
    try:
        client = get_client()
        f = Filter(must=[FieldCondition(
            key="neo4j_element_id",
            match=MatchValue(value=neo4j_element_id),
        )])
        points, _ = await with_retry(
            lambda: client.scroll(
                collection_name=collection,
                scroll_filter=f,
                with_payload=True,
                with_vectors=False,
                limit=10,
            )
        )
        return json.dumps(
            [{"id": p.id, "payload": p.payload} for p in points],
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def qdrant_upsert_chunk(
    collection: str,
    text: str,
    neo4j_element_id: str,
    node_type: str = "Chunk",
    document_id: str = "",
    document_title: str = "",
    position: int = 0,
    agent_id: str = "",
    confidence: float = 1.0,
    valid_from: str = "",
    valid_to: str = "",
    access_level: str = "",
    extra_metadata: str = "{}",
) -> str:
    """
    GCOR-aware upsert: embed text and store it with full cognitive metadata.

    The neo4j_element_id links this Qdrant point back to its authoritative
    Neo4j node. Confidence, temporal validity, agent partition, and
    access_level are stored in the payload for downstream filtering.

    Args:
        collection:       Target Qdrant collection.
        text:             Text to embed and store.
        neo4j_element_id: elementId() of the Neo4j node (required).
        node_type:        GCOR type: Chunk | Memory | Inference | Belief | Goal | Event.
        document_id:      Source document id.
        document_title:   Human-readable document title.
        position:         Chunk position within the document.
        agent_id:         Agent partition (empty = shared).
        confidence:       Certainty score 0.0–1.0 (default 1.0).
        valid_from:       ISO-8601 validity start (default: now).
        valid_to:         ISO-8601 expiry (default: null = forever).
        access_level:     ACL string (public | restricted | agent:<id>).
        extra_metadata:   Optional JSON object for additional fields.
    """
    try:
        if not neo4j_element_id:
            return json.dumps({"error": "neo4j_element_id is required for GCOR upsert"})
        if not (0.0 <= confidence <= 1.0):
            return json.dumps({"error": "confidence must be between 0.0 and 1.0"})

        vector = await embed(text)
        await ensure_collection(collection, len(vector))

        extra = json.loads(extra_metadata)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, neo4j_element_id))

        payload = _build_cognitive_payload(
            text=text,
            neo4j_element_id=neo4j_element_id,
            node_type=node_type,
            document_id=document_id,
            document_title=document_title,
            position=position,
            agent_id=agent_id,
            confidence=confidence,
            valid_from=valid_from,
            valid_to=valid_to,
            access_level=access_level,
            extra=extra,
        )
        point = PointStruct(id=point_id, vector=vector, payload=payload)
        client = get_client()
        result = await with_retry(
            lambda: client.upsert(collection_name=collection, points=[point])
        )
        return json.dumps({
            "status":            str(result.status),
            "id":                point_id,
            "neo4j_element_id":  neo4j_element_id,
            "confidence":        confidence,
            "valid_from":        payload["valid_from"],
            "valid_to":          valid_to or None,
            "access_level":      payload["access_level"],
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def qdrant_upsert(collection: str, text: str, metadata: str = "{}") -> str:
    """
    Free-form upsert: embed and store a text document.

    Prefer qdrant_upsert_chunk for GCOR-managed content.
    Use this for ad-hoc, non-graph content only.

    Args:
        collection: Target collection name.
        text:       Document text to embed and store.
        metadata:   Optional JSON object of extra payload fields.
    """
    try:
        vector = await embed(text)
        await ensure_collection(collection, len(vector))
        extra = json.loads(metadata)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": text, "access_level": DEFAULT_ACCESS, **extra},
        )
        client = get_client()
        result = await with_retry(
            lambda: client.upsert(collection_name=collection, points=[point])
        )
        return json.dumps({"status": str(result.status), "id": point.id}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def qdrant_delete_points(collection: str, point_ids: str) -> str:
    """Delete specific points from a collection by their IDs.
    Args:
        collection: Collection name.
        point_ids:  JSON array of point ID strings.
    """
    try:
        ids = json.loads(point_ids)
        client = get_client()
        result = await with_retry(
            lambda: client.delete(
                collection_name=collection,
                points_selector=PointIdsList(points=ids),
            )
        )
        return json.dumps({"status": str(result.status), "deleted_ids": ids}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def qdrant_delete_collection(collection: str) -> str:
    """Permanently delete a Qdrant collection and all its points.
    Args:
        collection: Collection name to delete.
    """
    try:
        client = get_client()
        result = await with_retry(lambda: client.delete_collection(collection))
        return json.dumps({"deleted": result}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def qdrant_gcor_status(collection: str = COLLECTION_NAME) -> str:
    """
    GCOR health check: per-node-type and per-access-level point counts.

    Args:
        collection: Collection name (default: COLLECTION_NAME env var).
    """
    node_types = ["Chunk", "Memory", "Inference", "Belief", "Goal", "Event",
                  "Document", "Concept"]
    try:
        client = get_client()
        type_counts = {}
        for nt in node_types:
            f = Filter(must=[FieldCondition(key="node_type", match=MatchValue(value=nt))])
            r = await with_retry(lambda f=f: client.count(collection, count_filter=f))
            type_counts[nt] = r.count

        total_r = await with_retry(lambda: client.count(collection))

        # Count by access level
        access_counts = {}
        for level in ["public", "restricted"]:
            f = Filter(must=[FieldCondition(key="access_level", match=MatchValue(value=level))])
            r = await with_retry(lambda f=f: client.count(collection, count_filter=f))
            access_counts[level] = r.count

        return json.dumps({
            "collection":     collection,
            "total":          total_r.count,
            "by_node_type":   type_counts,
            "by_access_level": access_counts,
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def sync_neo4j_to_qdrant(
    collection: str = COLLECTION_NAME,
    batch_size: int = 32,
    node_types: str = "Chunk",
    limit: int = 500,
) -> str:
    """
    GCOR sync: pull nodes from Neo4j, embed their text, and upsert into Qdrant
    with the full cognitive payload (neo4j_element_id, confidence, temporal,
    agent_id, access_level).

    Args:
        collection: Target Qdrant collection.
        batch_size: Texts to embed per OpenAI API call (default 32).
        node_types: Comma-separated node type labels (default "Chunk").
        limit:      Maximum nodes to sync per label (default 500).
    """
    if not OPENAI_API_KEY:
        return json.dumps({"error": "OPENAI_API_KEY not set — cannot embed"})

    try:
        from neo4j import AsyncGraphDatabase

        labels = [t.strip() for t in node_types.split(",") if t.strip()]
        all_nodes = []

        driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                for label in labels:
                    cypher = f"""
                        MATCH (n:{label})
                        WHERE n.text IS NOT NULL OR n.content IS NOT NULL
                              OR n.description IS NOT NULL OR n.name IS NOT NULL
                        RETURN
                            elementId(n)                          AS element_id,
                            coalesce(n.text, n.content,
                                     n.description, n.name)       AS text,
                            n.document_id                         AS document_id,
                            n.document_title                      AS document_title,
                            coalesce(n.position, 0)               AS position,
                            n.agent_id                            AS agent_id,
                            coalesce(n.confidence, 1.0)           AS confidence,
                            n.valid_from                          AS valid_from,
                            n.valid_to                            AS valid_to,
                            coalesce(n.access_level, $access)     AS access_level,
                            '{label}'                             AS node_type
                        LIMIT {limit}
                    """
                    result = await session.run(cypher, {"access": DEFAULT_ACCESS})
                    records = await result.data()
                    all_nodes.extend(records)
        finally:
            await driver.close()

        if not all_nodes:
            return json.dumps({"synced": 0,
                               "message": "No nodes with text found for given types"})

        sample = await embed_texts([all_nodes[0]["text"]])
        await ensure_collection(collection, len(sample[0]))

        total_upserted = 0
        client = get_client()

        for i in range(0, len(all_nodes), batch_size):
            batch   = all_nodes[i : i + batch_size]
            vectors = await embed_texts([n["text"] for n in batch])

            points = []
            for node, vec in zip(batch, vectors):
                eid = node["element_id"]
                payload = _build_cognitive_payload(
                    text=node["text"],
                    neo4j_element_id=eid,
                    node_type=node["node_type"],
                    document_id=node.get("document_id") or "",
                    document_title=node.get("document_title") or "",
                    position=node.get("position", 0),
                    agent_id=node.get("agent_id") or "",
                    confidence=float(node.get("confidence", 1.0)),
                    valid_from=node.get("valid_from") or "",
                    valid_to=node.get("valid_to") or "",
                    access_level=node.get("access_level") or DEFAULT_ACCESS,
                    extra={"source": "neo4j_sync"},
                )
                points.append(PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, eid)),
                    vector=vec,
                    payload=payload,
                ))

            await with_retry(
                lambda pts=points: client.upsert(collection_name=collection, points=pts)
            )
            total_upserted += len(points)

        return json.dumps({
            "synced":     total_upserted,
            "collection": collection,
            "node_types": labels,
        }, indent=2)

    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="sse", host=HOST, port=PORT)
