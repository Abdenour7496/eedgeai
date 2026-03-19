"""
Qdrant MCP server for OpenClaw — GCOR Edition.

GCOR principle: Qdrant is a semantic index over Neo4j.
Every point carries neo4j_element_id in its payload so the proxy
can hop from a semantic hit straight back to the graph.

Uses the official qdrant-client (async) with:
  - Persistent AsyncQdrantClient (connection pool managed by the SDK)
  - Exponential-backoff retry on transient errors
  - Full GCOR tool set:
      qdrant_list_collections      — enumerate collections
      qdrant_collection_info       — vector size, point count, status
      qdrant_count                 — count points (optionally filtered)
      qdrant_search                — semantic search, always returns neo4j_element_id
      qdrant_search_by_neo4j_id    — look up Qdrant point by graph element ID
      qdrant_upsert_chunk          — GCOR-aware upsert (requires neo4j_element_id)
      qdrant_upsert                — free-form upsert (legacy / ad-hoc)
      qdrant_delete_points         — delete by IDs
      qdrant_delete_collection     — drop a collection
      sync_neo4j_to_qdrant         — pull Chunk nodes from Neo4j → embed → index
      qdrant_gcor_status           — health check: counts per node_type in collection

Exposed to OpenClaw via SSE on 0.0.0.0:8765.
"""

import asyncio
import json
import os
import uuid

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
    VectorParams,
)

# ── Config ────────────────────────────────────────────────────────────────────
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


# ── Retry helper ──────────────────────────────────────────────────────────────
async def with_retry(coro_fn, retries: int = MAX_RETRY):
    for attempt in range(retries):
        try:
            return await coro_fn()
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)


# ── Embedding ─────────────────────────────────────────────────────────────────
async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via OpenAI and return a list of vectors."""
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


# ── Collection bootstrap ──────────────────────────────────────────────────────
async def ensure_collection(name: str, vector_size: int) -> None:
    client = get_client()
    try:
        await client.get_collection(name)
    except Exception:
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


# ── Payload helper ────────────────────────────────────────────────────────────
def _format_hit(hit) -> dict:
    """Normalise a search hit — always surface neo4j_element_id at top level."""
    payload = hit.payload or {}
    return {
        "id":                hit.id,
        "score":             hit.score,
        "neo4j_element_id":  payload.get("neo4j_element_id"),
        "node_type":         payload.get("node_type", "Chunk"),
        "text":              payload.get("text", ""),
        "document_id":       payload.get("document_id"),
        "document_title":    payload.get("document_title"),
        "position":          payload.get("position"),
        "payload":           payload,
    }


# ── MCP server ────────────────────────────────────────────────────────────────
mcp = FastMCP("Qdrant-GCOR")


@mcp.tool()
async def qdrant_list_collections() -> str:
    """List all Qdrant collections."""
    try:
        client = get_client()
        result = await with_retry(client.get_collections)
        return json.dumps(
            [{"name": c.name} for c in result.collections],
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def qdrant_collection_info(collection: str) -> str:
    """
    Return detailed information about a Qdrant collection.

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
async def qdrant_count(collection: str, node_type: str = "") -> str:
    """
    Return the number of points in a collection, optionally filtered by node_type.

    Args:
        collection: Collection name.
        node_type:  Optional GCOR node type filter (e.g. "Chunk", "Memory", "Goal").
    """
    try:
        client = get_client()
        scroll_filter = None
        if node_type:
            scroll_filter = Filter(
                must=[FieldCondition(key="node_type", match=MatchValue(value=node_type))]
            )
        result = await with_retry(
            lambda: client.count(collection, count_filter=scroll_filter)
        )
        return json.dumps(
            {"collection": collection, "node_type": node_type or "all", "count": result.count},
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def qdrant_search(
    collection: str,
    query: str,
    limit: int = 5,
    node_type: str = "",
) -> str:
    """
    Semantic search in a Qdrant collection.

    Results always include neo4j_element_id so callers can hop to the graph.

    Args:
        collection: Collection name to search.
        query:      Natural-language query text.
        limit:      Number of results (default 5).
        node_type:  Optional GCOR filter — restrict to one node type
                    (e.g. "Chunk", "Memory", "Goal", "Event").
    """
    try:
        vector = await embed(query)
        client = get_client()

        query_filter = None
        if node_type:
            query_filter = Filter(
                must=[FieldCondition(key="node_type", match=MatchValue(value=node_type))]
            )

        results = await with_retry(
            lambda: client.search(
                collection_name=collection,
                query_vector=vector,
                limit=limit,
                with_payload=True,
                query_filter=query_filter,
            )
        )
        return json.dumps([_format_hit(r) for r in results], indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def qdrant_search_by_neo4j_id(
    collection: str,
    neo4j_element_id: str,
) -> str:
    """
    Retrieve the Qdrant point(s) that reference a specific Neo4j element ID.

    Use this in the GCOR structural phase to verify that a graph node is
    indexed and to retrieve its embedding metadata.

    Args:
        collection:       Collection name.
        neo4j_element_id: The elementId string from Neo4j (e.g. "4:abc:0").
    """
    try:
        client = get_client()
        scroll_filter = Filter(
            must=[FieldCondition(
                key="neo4j_element_id",
                match=MatchValue(value=neo4j_element_id),
            )]
        )
        points, _ = await with_retry(
            lambda: client.scroll(
                collection_name=collection,
                scroll_filter=scroll_filter,
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
    document_id: str = "",
    document_title: str = "",
    position: int = 0,
    node_type: str = "Chunk",
    extra_metadata: str = "{}",
) -> str:
    """
    GCOR-aware upsert: embed text and store it with a Neo4j graph reference.

    The neo4j_element_id links this Qdrant point back to its authoritative
    Neo4j node — required by the GCOR architecture.

    Args:
        collection:       Target Qdrant collection.
        text:             Text to embed and store.
        neo4j_element_id: elementId() of the Neo4j node this indexes (required).
        document_id:      Source document identifier.
        document_title:   Human-readable document title.
        position:         Chunk position within the document.
        node_type:        GCOR node type label (default "Chunk").
        extra_metadata:   Optional JSON object for additional payload fields.
    """
    try:
        if not neo4j_element_id:
            return json.dumps({"error": "neo4j_element_id is required for GCOR upsert"})

        vector = await embed(text)
        await ensure_collection(collection, len(vector))

        extra = json.loads(extra_metadata)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, neo4j_element_id))

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "neo4j_element_id": neo4j_element_id,
                "node_type":        node_type,
                "text":             text,
                "document_id":      document_id,
                "document_title":   document_title,
                "position":         position,
                **extra,
            },
        )
        client = get_client()
        result = await with_retry(
            lambda: client.upsert(collection_name=collection, points=[point])
        )
        return json.dumps(
            {"status": str(result.status), "id": point_id, "neo4j_element_id": neo4j_element_id},
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def qdrant_upsert(collection: str, text: str, metadata: str = "{}") -> str:
    """
    Free-form upsert: embed and store a text document.

    Prefer qdrant_upsert_chunk for GCOR-managed content (requires neo4j_element_id).
    Use this tool for ad-hoc, non-graph content only.

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
            payload={"text": text, **extra},
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
    """
    Delete specific points from a collection by their IDs.

    Args:
        collection: Collection name.
        point_ids:  JSON array of point ID strings, e.g. '["id1","id2"]'.
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
    """
    Permanently delete a Qdrant collection and all its points.

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
    GCOR health check: return per-node-type point counts for a collection.

    Helps verify that Neo4j nodes are properly indexed in Qdrant.

    Args:
        collection: Collection name (default: COLLECTION_NAME env var).
    """
    node_types = ["Chunk", "Memory", "Goal", "Event", "Document", "Concept"]
    try:
        client = get_client()
        totals = {}
        for nt in node_types:
            f = Filter(must=[FieldCondition(key="node_type", match=MatchValue(value=nt))])
            r = await with_retry(lambda f=f: client.count(collection, count_filter=f))
            totals[nt] = r.count
        all_r = await with_retry(lambda: client.count(collection))
        totals["_total"] = all_r.count
        return json.dumps({"collection": collection, "counts": totals}, indent=2)
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
    with neo4j_element_id in every point's payload.

    Uses the canonical GCOR Cypher query that returns elementId() so that every
    Qdrant point carries a proper graph reference.

    Args:
        collection: Target Qdrant collection (default: COLLECTION_NAME env var).
        batch_size: Texts to embed per OpenAI API call (default 32).
        node_types: Comma-separated node type labels to sync (default "Chunk").
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
                    # Fetch nodes with their elementId and a text representation
                    cypher = f"""
                        MATCH (n:{label})
                        WHERE n.text IS NOT NULL OR n.content IS NOT NULL
                              OR n.description IS NOT NULL OR n.name IS NOT NULL
                        RETURN elementId(n)          AS element_id,
                               coalesce(n.text, n.content, n.description, n.name) AS text,
                               n.document_id         AS document_id,
                               n.document_title      AS document_title,
                               coalesce(n.position, 0) AS position,
                               '{label}'             AS node_type
                        LIMIT {limit}
                    """
                    result = await session.run(cypher)
                    records = await result.data()
                    all_nodes.extend(records)
        finally:
            await driver.close()

        if not all_nodes:
            return json.dumps({"synced": 0, "message": "No nodes with text found for given types"})

        # Ensure collection exists (probe vector size with first embed)
        sample = await embed_texts([all_nodes[0]["text"]])
        await ensure_collection(collection, len(sample[0]))

        # Embed + upsert in batches
        total_upserted = 0
        client = get_client()

        for i in range(0, len(all_nodes), batch_size):
            batch = all_nodes[i : i + batch_size]
            texts = [n["text"] for n in batch]
            vectors = await embed_texts(texts)

            points = []
            for node, vec in zip(batch, vectors):
                eid = node["element_id"]
                points.append(PointStruct(
                    # Deterministic ID — upsert is idempotent
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, eid)),
                    vector=vec,
                    payload={
                        "neo4j_element_id": eid,
                        "node_type":        node["node_type"],
                        "text":             node["text"],
                        "document_id":      node.get("document_id") or "",
                        "document_title":   node.get("document_title") or "",
                        "position":         node.get("position", 0),
                        "source":           "neo4j_sync",
                    },
                ))

            await with_retry(lambda pts=points: client.upsert(collection_name=collection, points=pts))
            total_upserted += len(points)

        return json.dumps({
            "synced":     total_upserted,
            "collection": collection,
            "node_types": labels,
        }, indent=2)

    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="sse", host=HOST, port=PORT)
