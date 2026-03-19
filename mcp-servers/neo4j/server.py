"""
Neo4j MCP server for OpenClaw.

Uses the official neo4j Python async driver with:
  - Persistent connection pool (max 10 connections)
  - Exponential-backoff retry on transient errors
  - Richer tool set: query, search, create_node, create_relationship, schema, stats

Exposed to OpenClaw via SSE on 0.0.0.0:8766.
"""

import asyncio
import json
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

# ── Config ────────────────────────────────────────────────────────────────────
URI       = os.getenv("NEO4J_URI",      "bolt://neo4j:7687")
USER      = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
PASSWORD  = os.getenv("NEO4J_PASSWORD", "test1234")
DATABASE  = os.getenv("NEO4J_DATABASE", "neo4j")
MAX_RETRY = int(os.getenv("NEO4J_MAX_RETRIES", "3"))

HOST = os.getenv("NEO4J_MCP_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("NEO4J_MCP_SERVER_PORT", "8766"))

# ── Driver (connection pool) ──────────────────────────────────────────────────
_driver = None

def get_driver():
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            URI,
            auth=(USER, PASSWORD),
            max_connection_pool_size=10,
            connection_acquisition_timeout=5.0,
        )
    return _driver

# ── Retry helper ──────────────────────────────────────────────────────────────
_RETRYABLE = (ServiceUnavailable, SessionExpired, TransientError)

async def run_with_retry(cypher: str, params: dict = None, retries: int = MAX_RETRY) -> list:
    params = params or {}
    driver = get_driver()
    for attempt in range(retries):
        try:
            async with driver.session(database=DATABASE) as session:
                result = await session.run(cypher, params)
                return await result.data()
        except _RETRYABLE:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)

# ── MCP server ────────────────────────────────────────────────────────────────
mcp = FastMCP("Neo4j")


@mcp.tool()
async def neo4j_query(cypher: str, params: str = "{}") -> str:
    """
    Run any Cypher query against Neo4j and return results as JSON.

    Args:
        cypher: Cypher statement to execute.
        params: Optional JSON object of query parameters, e.g. '{"name": "Alice"}'.
    """
    try:
        p = json.loads(params)
        records = await run_with_retry(cypher, p)
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_search(query: str, limit: int = 10) -> str:
    """
    Full-text search across all Neo4j node properties.

    Args:
        query: Text to search for.
        limit: Maximum number of matching nodes to return.
    """
    try:
        records = await run_with_retry(
            "MATCH (n) "
            "WHERE any(k IN keys(n) WHERE toString(n[k]) CONTAINS $q) "
            "RETURN labels(n) AS labels, properties(n) AS props "
            "LIMIT $limit",
            {"q": query[:200], "limit": limit},
        )
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_create_node(label: str, properties: str) -> str:
    """
    Create a new node with the given label and properties.

    Args:
        label: Node label, e.g. "Document" or "Concept".
        properties: JSON object of node properties, e.g. '{"title": "RAG intro", "source": "web"}'.
    """
    try:
        props = json.loads(properties)
        records = await run_with_retry(
            f"CREATE (n:`{label}` $props) "
            "RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props",
            {"props": props},
        )
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_create_relationship(
    from_id: int,
    rel_type: str,
    to_id: int,
    properties: str = "{}",
) -> str:
    """
    Create a relationship between two existing nodes.

    Args:
        from_id: Internal Neo4j id of the source node.
        rel_type: Relationship type, e.g. "RELATED_TO" or "CITES".
        to_id: Internal Neo4j id of the target node.
        properties: Optional JSON object of relationship properties.
    """
    try:
        props = json.loads(properties)
        records = await run_with_retry(
            f"MATCH (a) WHERE id(a) = $from "
            f"MATCH (b) WHERE id(b) = $to "
            f"CREATE (a)-[r:`{rel_type}` $props]->(b) "
            "RETURN type(r) AS type, properties(r) AS props",
            {"from": from_id, "to": to_id, "props": props},
        )
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_schema() -> str:
    """
    Return the current database schema: node labels, relationship types, and property keys.
    """
    try:
        labels, rels, keys = await asyncio.gather(
            run_with_retry("CALL db.labels() YIELD label RETURN collect(label) AS labels"),
            run_with_retry("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS types"),
            run_with_retry("CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) AS keys"),
        )
        return json.dumps({
            "labels":             labels[0].get("labels", []) if labels else [],
            "relationship_types": rels[0].get("types",  []) if rels   else [],
            "property_keys":      keys[0].get("keys",   []) if keys   else [],
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_stats() -> str:
    """
    Return database statistics: total node count and relationship count.
    """
    try:
        nodes, rels = await asyncio.gather(
            run_with_retry("MATCH (n) RETURN count(n) AS count"),
            run_with_retry("MATCH ()-[r]->() RETURN count(r) AS count"),
        )
        return json.dumps({
            "node_count":         nodes[0].get("count", 0) if nodes else 0,
            "relationship_count": rels[0].get("count",  0) if rels  else 0,
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="sse", host=HOST, port=PORT)
