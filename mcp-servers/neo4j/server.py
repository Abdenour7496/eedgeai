"""
Neo4j MCP server — GCOR cognitive backbone.

Neo4j is primary truth. Every entity of interest (Document, Chunk, Concept,
Agent, Goal, Event, Memory) is a node.  Qdrant only holds pointers (element IDs)
back to these nodes.

Tools
─────
  Core query
    neo4j_query                run any Cypher
    neo4j_search               full-text search across node properties
    neo4j_schema / neo4j_stats database introspection

  Graph construction
    neo4j_create_node          generic node creation
    neo4j_create_relationship  connect two nodes

  GCOR-specific
    neo4j_expand_context       expand graph around Qdrant candidate element IDs
    neo4j_provenance           trace where a Chunk / Concept came from

  Semantic node types
    neo4j_create_memory        store an Agent memory
    neo4j_create_goal          create a Goal with optional dependency
    neo4j_create_event         log an Event and link involved concepts
    neo4j_get_agent_context    fetch everything related to an Agent
    neo4j_goal_dependencies    full DEPENDS_ON tree for a Goal
"""

import asyncio
import json
import os
from datetime import datetime, timezone

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
            URI, auth=(USER, PASSWORD),
            max_connection_pool_size=10,
            connection_acquisition_timeout=5.0,
        )
    return _driver

_RETRYABLE = (ServiceUnavailable, SessionExpired, TransientError)

async def run(cypher: str, params: dict = None, retries: int = MAX_RETRY) -> list:
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

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

# ── MCP server ────────────────────────────────────────────────────────────────
mcp = FastMCP("Neo4j")


# ── Core query tools ──────────────────────────────────────────────────────────

@mcp.tool()
async def neo4j_query(cypher: str, params: str = "{}") -> str:
    """Run any Cypher query against Neo4j and return results as JSON.
    Args:
        cypher: Cypher statement.
        params: Optional JSON parameter object, e.g. '{"name": "Alice"}'.
    """
    try:
        records = await run(cypher, json.loads(params))
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_search(query: str, limit: int = 10) -> str:
    """Full-text search across all node properties.
    Args:
        query: Search text.
        limit: Max results.
    """
    try:
        # Try full-text index first, fall back to property scan
        try:
            records = await run(
                "CALL db.index.fulltext.queryNodes('nodeIndex', $q) "
                "YIELD node RETURN labels(node) AS labels, properties(node) AS props LIMIT $limit",
                {"q": query[:200], "limit": limit},
            )
        except Exception:
            records = await run(
                "MATCH (n) WHERE any(k IN keys(n) WHERE toLower(toString(n[k])) CONTAINS toLower($q)) "
                "RETURN labels(n) AS labels, properties(n) AS props LIMIT $limit",
                {"q": query[:100], "limit": limit},
            )
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_schema() -> str:
    """Return node labels, relationship types, and property keys."""
    try:
        labels, rels, keys = await asyncio.gather(
            run("CALL db.labels() YIELD label RETURN collect(label) AS labels"),
            run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS types"),
            run("CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) AS keys"),
        )
        return json.dumps({
            "labels":             labels[0].get("labels", []) if labels else [],
            "relationship_types": rels[0].get("types",   []) if rels   else [],
            "property_keys":      keys[0].get("keys",    []) if keys   else [],
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_stats() -> str:
    """Return total node count and relationship count."""
    try:
        nodes, rels = await asyncio.gather(
            run("MATCH (n) RETURN count(n) AS count"),
            run("MATCH ()-[r]->() RETURN count(r) AS count"),
        )
        return json.dumps({
            "node_count":         nodes[0].get("count", 0) if nodes else 0,
            "relationship_count": rels[0].get("count",  0) if rels  else 0,
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Graph construction ────────────────────────────────────────────────────────

@mcp.tool()
async def neo4j_create_node(label: str, properties: str) -> str:
    """Create a new node.
    Args:
        label:      Node label, e.g. "Document", "Concept".
        properties: JSON object of properties.
    """
    try:
        props = json.loads(properties)
        records = await run(
            f"CREATE (n:`{label}` $props) "
            "RETURN elementId(n) AS element_id, labels(n) AS labels, properties(n) AS props",
            {"props": props},
        )
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_create_relationship(
    from_element_id: str, rel_type: str, to_element_id: str, properties: str = "{}"
) -> str:
    """Create a relationship between two nodes using their elementIds.
    Args:
        from_element_id: elementId of the source node.
        rel_type:        Relationship type, e.g. "DEPENDS_ON".
        to_element_id:   elementId of the target node.
        properties:      Optional JSON object of relationship properties.
    """
    try:
        props = json.loads(properties)
        records = await run(
            f"MATCH (a) WHERE elementId(a) = $from "
            f"MATCH (b) WHERE elementId(b) = $to "
            f"CREATE (a)-[r:`{rel_type}` $props]->(b) "
            "RETURN type(r) AS type, properties(r) AS props",
            {"from": from_element_id, "to": to_element_id, "props": props},
        )
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── GCOR retrieval tools ──────────────────────────────────────────────────────

@mcp.tool()
async def neo4j_expand_context(element_ids: str, intent: str = "semantic") -> str:
    """
    Expand the graph around a set of node elementIds (typically from Qdrant results).

    This is the structural phase of GCOR: Qdrant provides candidate IDs,
    Neo4j provides the rich context around them.

    Args:
        element_ids: JSON array of Neo4j elementId strings.
        intent:      One of: factual | planning | dependency | memory | semantic
    """
    _QUERIES = {
        "factual": (
            "MATCH (n) WHERE elementId(n) IN $ids "
            "OPTIONAL MATCH (n)-[r]-(rel) "
            "OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document) "
            "RETURN properties(n) AS node, labels(n) AS labels, "
            "properties(doc) AS document, "
            "collect(DISTINCT {rel: type(r), props: properties(rel)})[..6] AS related "
            "LIMIT 20"
        ),
        "planning": (
            "MATCH (n) WHERE elementId(n) IN $ids "
            "OPTIONAL MATCH (n)-[:DEPENDS_ON*1..2]->(dep) "
            "OPTIONAL MATCH (block:Goal)-[:DEPENDS_ON]->(n) "
            "OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document) "
            "RETURN properties(n) AS node, labels(n) AS labels, "
            "properties(doc) AS document, "
            "collect(DISTINCT properties(dep))[..5] AS dependencies, "
            "collect(DISTINCT properties(block))[..5] AS blocking_goals LIMIT 20"
        ),
        "dependency": (
            "MATCH (n) WHERE elementId(n) IN $ids "
            "OPTIONAL MATCH path = (n)-[:DEPENDS_ON*1..3]->(dep) "
            "OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document) "
            "RETURN properties(n) AS node, labels(n) AS labels, "
            "properties(doc) AS document, "
            "[x IN nodes(path) | properties(x)][..8] AS dependency_chain LIMIT 20"
        ),
        "memory": (
            "MATCH (n) WHERE elementId(n) IN $ids "
            "OPTIONAL MATCH (n)-[:MENTIONS]->(concept:Concept) "
            "OPTIONAL MATCH (mem:Memory)-[:ABOUT]->(concept) "
            "OPTIONAL MATCH (evt:Event)-[:INVOLVES]->(concept) "
            "OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document) "
            "RETURN properties(n) AS node, labels(n) AS labels, "
            "properties(doc) AS document, "
            "collect(DISTINCT properties(concept))[..5] AS concepts, "
            "collect(DISTINCT properties(mem))[..5] AS memories, "
            "collect(DISTINCT properties(evt))[..5] AS events LIMIT 20"
        ),
        "semantic": (
            "MATCH (n) WHERE elementId(n) IN $ids "
            "OPTIONAL MATCH (n)-[:MENTIONS]->(concept:Concept) "
            "OPTIONAL MATCH (n)-[:FOLLOWS]-(neighbor:Chunk) "
            "OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document) "
            "RETURN properties(n) AS node, labels(n) AS labels, "
            "properties(doc) AS document, "
            "collect(DISTINCT properties(concept))[..5] AS concepts, "
            "collect(DISTINCT properties(neighbor))[..3] AS neighbors LIMIT 20"
        ),
    }
    try:
        ids  = json.loads(element_ids)
        cyph = _QUERIES.get(intent, _QUERIES["semantic"])
        records = await run(cyph, {"ids": ids})
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_provenance(element_id: str) -> str:
    """Trace the origin of a node: which Document and Chunk it came from.
    Args:
        element_id: Neo4j elementId of any node.
    """
    try:
        records = await run(
            "MATCH (n) WHERE elementId(n) = $eid "
            "OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document) "
            "OPTIONAL MATCH (n)-[:FOLLOWS*0..]->(first:Chunk)<-[:HAS_CHUNK]-(root:Document) "
            "OPTIONAL MATCH (n)-[:MENTIONS]->(concept:Concept) "
            "RETURN properties(n) AS node, labels(n) AS labels, "
            "properties(doc) AS document, properties(root) AS root_document, "
            "collect(properties(concept)) AS concepts",
            {"eid": element_id},
        )
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Semantic node type tools ──────────────────────────────────────────────────

@mcp.tool()
async def neo4j_create_memory(
    agent_id: str, content: str, memory_type: str = "observation", concepts: str = "[]"
) -> str:
    """Store a Memory node for an Agent and link it to related Concepts.
    Args:
        agent_id:    Agent id string.
        content:     Memory content text.
        memory_type: e.g. "observation", "reflection", "plan", "fact".
        concepts:    JSON array of concept names to link via ABOUT.
    """
    try:
        import uuid as _uuid
        mem_id = str(_uuid.uuid4())
        await run(
            "MERGE (a:Agent {id: $agent_id}) "
            "CREATE (m:Memory {id: $mem_id, content: $content, type: $type, "
            "agent_id: $agent_id, created_at: $now}) "
            "CREATE (a)-[:HAS_MEMORY]->(m) "
            "RETURN elementId(m) AS element_id",
            {"agent_id": agent_id, "mem_id": mem_id, "content": content,
             "type": memory_type, "now": _now()},
        )
        for name in json.loads(concepts):
            await run(
                "MERGE (c:Concept {name: $name}) "
                "WITH c MATCH (m:Memory {id: $mid}) "
                "MERGE (m)-[:ABOUT]->(c)",
                {"name": name, "mid": mem_id},
            )
        return json.dumps({"memory_id": mem_id, "agent_id": agent_id, "type": memory_type}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_create_goal(
    agent_id: str,
    description: str,
    status: str = "active",
    depends_on_goal_id: str = "",
) -> str:
    """Create a Goal node for an Agent with optional DEPENDS_ON link.
    Args:
        agent_id:            Agent id.
        description:         Goal description.
        status:              active | blocked | done | cancelled.
        depends_on_goal_id:  Optional goal id this goal depends on.
    """
    try:
        import uuid as _uuid
        goal_id = str(_uuid.uuid4())
        await run(
            "MERGE (a:Agent {id: $agent_id}) "
            "CREATE (g:Goal {id: $goal_id, description: $desc, status: $status, "
            "agent_id: $agent_id, created_at: $now}) "
            "CREATE (a)-[:PURSUES]->(g)",
            {"agent_id": agent_id, "goal_id": goal_id, "desc": description,
             "status": status, "now": _now()},
        )
        if depends_on_goal_id:
            await run(
                "MATCH (g:Goal {id: $gid}), (dep:Goal {id: $dep}) "
                "MERGE (g)-[:DEPENDS_ON]->(dep)",
                {"gid": goal_id, "dep": depends_on_goal_id},
            )
        return json.dumps({"goal_id": goal_id, "status": status}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_create_event(
    description: str,
    event_type: str = "system",
    agent_id: str = "",
    concepts: str = "[]",
) -> str:
    """Log a time-stamped Event and link it to concepts and an optional agent.
    Args:
        description: What happened.
        event_type:  e.g. "user_action", "system", "tool_call", "result".
        agent_id:    Optional agent who triggered/observed the event.
        concepts:    JSON array of concept names the event involves.
    """
    try:
        import uuid as _uuid
        evt_id = str(_uuid.uuid4())
        await run(
            "CREATE (e:Event {id: $id, description: $desc, type: $type, "
            "timestamp: $now}) "
            "RETURN elementId(e) AS eid",
            {"id": evt_id, "desc": description, "type": event_type, "now": _now()},
        )
        if agent_id:
            await run(
                "MERGE (a:Agent {id: $aid}) "
                "WITH a MATCH (e:Event {id: $eid}) "
                "MERGE (a)-[:PARTICIPATED_IN]->(e)",
                {"aid": agent_id, "eid": evt_id},
            )
        for name in json.loads(concepts):
            await run(
                "MERGE (c:Concept {name: $name}) "
                "WITH c MATCH (e:Event {id: $eid}) "
                "MERGE (e)-[:INVOLVES]->(c)",
                {"name": name, "eid": evt_id},
            )
        return json.dumps({"event_id": evt_id, "timestamp": _now()}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_get_agent_context(agent_id: str) -> str:
    """Fetch everything related to an Agent: goals, memories, events.
    Args:
        agent_id: Agent id string.
    """
    try:
        records = await run(
            "MATCH (a:Agent {id: $id}) "
            "OPTIONAL MATCH (a)-[:PURSUES]->(g:Goal) "
            "OPTIONAL MATCH (a)-[:HAS_MEMORY]->(m:Memory) "
            "OPTIONAL MATCH (a)-[:PARTICIPATED_IN]->(e:Event) "
            "RETURN properties(a) AS agent, "
            "collect(DISTINCT properties(g)) AS goals, "
            "collect(DISTINCT properties(m)) AS memories, "
            "collect(DISTINCT properties(e)) AS events",
            {"id": agent_id},
        )
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_goal_dependencies(goal_id: str) -> str:
    """Return the full DEPENDS_ON tree for a Goal (up to 5 hops).
    Args:
        goal_id: Goal id string.
    """
    try:
        records = await run(
            "MATCH (g:Goal {id: $id}) "
            "OPTIONAL MATCH path = (g)-[:DEPENDS_ON*1..5]->(dep) "
            "RETURN properties(g) AS goal, "
            "[x IN nodes(path) | properties(x)] AS dependency_chain",
            {"id": goal_id},
        )
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="sse", host=HOST, port=PORT)
