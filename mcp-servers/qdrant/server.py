"""
Qdrant MCP server for OpenClaw.

Uses the official qdrant-client (async) with:
  - Persistent AsyncQdrantClient (connection pool managed by the SDK)
  - Exponential-backoff retry on transient errors
  - Richer tool set: list, info, count, search, upsert, delete_collection
  - Sync tool: pull all Neo4j node text into Qdrant for cross-DB RAG

Exposed to OpenClaw via SSE on 0.0.0.0:8765.
"""

import asyncio
import json
import os
import uuid

import httpx
from mcp.server.fastmcp import FastMCP
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_URL        = os.getenv("QDRANT_URL",        "http://quarant:6333")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY",    "")
COLLECTION_NAME   = os.getenv("COLLECTION_NAME",   "documents")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",    "")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL",   "text-embedding-3-small")
MAX_RETRY         = int(os.getenv("QDRANT_MAX_RETRIES", "3"))

NEO4J_URI         = os.getenv("NEO4J_URI",      "bolt://neo4j:7687")
NEO4J_USER        = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
NEO4J_PASSWORD    = os.getenv("NEO4J_PASSWORD", "test1234")
NEO4J_DATABASE    = os.getenv("NEO4J_DATABASE", "neo4j")

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
async def embed(text: str) -> list:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set — cannot embed text")
    async with httpx.AsyncClient(timeout=30) as c:
        resp = await c.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"input": text[:8000], "model": EMBEDDING_MODEL},
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

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

# ── MCP server ────────────────────────────────────────────────────────────────
mcp = FastMCP("Qdrant")


@mcp.tool()
async def qdrant_list_collections() -> str:
    """List all Qdrant collections with their point counts."""
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
async def qdrant_count(collection: str) -> str:
    """
    Return the number of points stored in a collection.

    Args:
        collection: Collection name.
    """
    try:
        client = get_client()
        result = await with_retry(lambda: client.count(collection))
        return json.dumps({"collection": collection, "count": result.count}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def qdrant_search(collection: str, query: str, limit: int = 5) -> str:
    """
    Semantic search in a Qdrant collection using a natural-language query.

    Args:
        collection: Collection name to search.
        query:      Natural-language query text.
        limit:      Number of results to return (default 5).
    """
    try:
        vector = await embed(query)
        client = get_client()
        results = await with_retry(
            lambda: client.search(
                collection_name=collection,
                query_vector=vector,
                limit=limit,
                with_payload=True,
            )
        )
        return json.dumps(
            [{"score": r.score, "payload": r.payload} for r in results],
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def qdrant_upsert(collection: str, text: str, metadata: str = "{}") -> str:
    """
    Embed and store a text document in a Qdrant collection.
    The collection is created automatically if it does not exist.

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
async def sync_neo4j_to_qdrant(
    collection: str = COLLECTION_NAME,
    cypher: str = "MATCH (n) WHERE size(keys(n)) > 0 RETURN toString(properties(n)) AS text LIMIT 500",
    batch_size: int = 32,
) -> str:
    """
    Sync Neo4j node data into Qdrant for cross-database RAG.

    Runs a Cypher query, embeds each result row's text, and upserts it into
    the target Qdrant collection. The collection is created if it doesn't exist.

    Args:
        collection: Target Qdrant collection (default: COLLECTION_NAME env var).
        cypher:     Cypher query whose rows must contain a 'text' column.
        batch_size: Number of texts to embed per OpenAI API call.
    """
    if not OPENAI_API_KEY:
        return json.dumps({"error": "OPENAI_API_KEY not set — cannot embed"})

    try:
        from neo4j import AsyncGraphDatabase

        driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            async with driver.session(database=NEO4J_DATABASE) as session:
                result = await session.run(cypher)
                records = await result.data()
        finally:
            await driver.close()

        texts = [r.get("text", "") for r in records if r.get("text")]
        if not texts:
            return json.dumps({"synced": 0, "message": "No 'text' rows returned by Cypher query"})

        # Embed in batches
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            async with httpx.AsyncClient(timeout=60) as c:
                resp = await c.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={"input": batch, "model": EMBEDDING_MODEL},
                )
                resp.raise_for_status()
                vectors.extend([d["embedding"] for d in resp.json()["data"]])

        await ensure_collection(collection, len(vectors[0]))

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"text": txt, "source": "neo4j"},
            )
            for txt, vec in zip(texts, vectors)
        ]

        client = get_client()
        await with_retry(lambda: client.upsert(collection_name=collection, points=points))

        return json.dumps({"synced": len(points), "collection": collection}, indent=2)

    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="sse", host=HOST, port=PORT)
