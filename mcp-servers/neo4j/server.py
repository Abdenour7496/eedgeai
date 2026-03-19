"""
Neo4j MCP server — GCOR Cognitive Infrastructure backbone.

Neo4j is primary truth.  Every entity of interest is a first-class node.
Qdrant only holds pointers (element IDs) back to these nodes.

─────────────────────────────────────────────────────────────────────────────
Cognitive infrastructure additions over baseline GCOR:

  Temporal edges
    • valid_from / valid_to  on Memory, Inference, Belief, Goal
    • updated_at             on all mutable nodes

  Confidence scoring
    • confidence (0.0–1.0)  on Memory, Inference, Belief
    • neo4j_update_confidence  — patch confidence on any node

  Agent-specific memory partitions
    • agent_id on Memory, Inference, Belief, Goal
    • neo4j_agent_partition   — all cognitive nodes for one agent

  Graph-based access control
    • access_level ("public" | "restricted" | "agent:<id>") on all nodes
    • neo4j_access_check      — verify read permission

  Self-reflective node types
    • neo4j_create_inference  — :Inference derived from evidence
    • neo4j_create_belief     — :Belief held by an Agent
    • neo4j_resolve_beliefs   — agent beliefs filtered by confidence + time
    • neo4j_check_contradictions — CONTRADICTS neighbours
    • neo4j_inferences_for_goal  — Inferences that SUPPORT a Goal

─────────────────────────────────────────────────────────────────────────────
Tools
  Core query
    neo4j_query                run any Cypher
    neo4j_search               full-text search across node properties
    neo4j_schema / neo4j_stats database introspection

  Graph construction
    neo4j_create_node          generic node creation
    neo4j_create_relationship  connect two nodes

  GCOR structural retrieval
    neo4j_expand_context       expand graph around Qdrant element IDs
    neo4j_provenance           trace document/chunk origin of a node

  Semantic node types (existing)
    neo4j_create_memory
    neo4j_create_goal
    neo4j_create_event
    neo4j_get_agent_context
    neo4j_goal_dependencies

  Cognitive infrastructure (new)
    neo4j_create_inference
    neo4j_create_belief
    neo4j_resolve_beliefs
    neo4j_check_contradictions
    neo4j_inferences_for_goal
    neo4j_temporal_query
    neo4j_update_confidence
    neo4j_agent_partition
    neo4j_access_check
"""

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

# ── Config ─────────────────────────────────────────────────────────────────────
URI       = os.getenv("NEO4J_URI",      "bolt://neo4j:7687")
USER      = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
PASSWORD  = os.getenv("NEO4J_PASSWORD", "test1234")
DATABASE  = os.getenv("NEO4J_DATABASE", "neo4j")
MAX_RETRY = int(os.getenv("NEO4J_MAX_RETRIES", "3"))
HOST = os.getenv("NEO4J_MCP_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("NEO4J_MCP_SERVER_PORT", "8766"))

# Default access level for newly created nodes
DEFAULT_ACCESS = os.getenv("DEFAULT_ACCESS_LEVEL", "public")

# ── Driver (connection pool) ───────────────────────────────────────────────────
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


# ── MCP server ─────────────────────────────────────────────────────────────────
mcp = FastMCP("Neo4j-Cognitive")


# ── Core query tools ───────────────────────────────────────────────────────────

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
    """Return node / relationship counts and per-label breakdown."""
    try:
        nodes, rels, label_counts = await asyncio.gather(
            run("MATCH (n) RETURN count(n) AS count"),
            run("MATCH ()-[r]->() RETURN count(r) AS count"),
            run(
                "CALL db.labels() YIELD label "
                "CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) AS c', {}) YIELD value "
                "RETURN label, value.c AS count "
                "ORDER BY count DESC"
            ),
        )
        return json.dumps({
            "node_count":         nodes[0].get("count", 0) if nodes else 0,
            "relationship_count": rels[0].get("count",  0) if rels  else 0,
            "per_label":          label_counts,
        }, indent=2)
    except Exception:
        # Fallback without APOC
        nodes, rels = await asyncio.gather(
            run("MATCH (n) RETURN count(n) AS count"),
            run("MATCH ()-[r]->() RETURN count(r) AS count"),
        )
        return json.dumps({
            "node_count":         nodes[0].get("count", 0) if nodes else 0,
            "relationship_count": rels[0].get("count",  0) if rels  else 0,
        }, indent=2)


# ── Graph construction ─────────────────────────────────────────────────────────

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
        properties:      Optional JSON object of relationship properties
                         (may include valid_at, confidence, etc.).
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


# ── GCOR structural retrieval ──────────────────────────────────────────────────

@mcp.tool()
async def neo4j_expand_context(
    element_ids: str,
    intent: str = "semantic",
    agent_id: str = "",
    min_confidence: float = 0.0,
) -> str:
    """
    Expand the graph around Qdrant candidate element IDs (GCOR structural phase).

    Respects confidence thresholds and agent partitions for Memory/Belief/Inference.

    Args:
        element_ids:    JSON array of Neo4j elementId strings.
        intent:         factual | planning | dependency | memory |
                        semantic | inference | belief
        agent_id:       Optional — restrict Memory/Belief/Inference to this agent.
        min_confidence: Minimum confidence for Memory/Inference/Belief nodes (0.0 = any).
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
            "OPTIONAL MATCH (inf:Inference)-[:SUPPORTS]->(n) "
            "  WHERE inf.confidence >= $min_conf "
            "RETURN properties(n) AS node, labels(n) AS labels, "
            "properties(doc) AS document, "
            "collect(DISTINCT properties(dep))[..5]   AS dependencies, "
            "collect(DISTINCT properties(block))[..5] AS blocking_goals, "
            "collect(DISTINCT properties(inf))[..3]   AS supporting_inferences "
            "LIMIT 20"
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
            "  WHERE ($agent_id = '' OR mem.agent_id = $agent_id) "
            "    AND mem.confidence >= $min_conf "
            "    AND (mem.valid_to IS NULL OR mem.valid_to >= $now) "
            "OPTIONAL MATCH (bel:Belief)-[:ABOUT]->(concept) "
            "  WHERE ($agent_id = '' OR bel.agent_id = $agent_id) "
            "    AND bel.confidence >= $min_conf "
            "OPTIONAL MATCH (evt:Event)-[:INVOLVES]->(concept) "
            "OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document) "
            "RETURN properties(n) AS node, labels(n) AS labels, "
            "properties(doc) AS document, "
            "collect(DISTINCT properties(concept))[..5] AS concepts, "
            "collect(DISTINCT properties(mem))[..5]     AS memories, "
            "collect(DISTINCT properties(bel))[..3]     AS beliefs, "
            "collect(DISTINCT properties(evt))[..5]     AS events "
            "ORDER BY evt.timestamp DESC LIMIT 20"
        ),
        "semantic": (
            "MATCH (n) WHERE elementId(n) IN $ids "
            "OPTIONAL MATCH (n)-[:MENTIONS]->(concept:Concept) "
            "OPTIONAL MATCH (n)-[:FOLLOWS]-(neighbor:Chunk) "
            "OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document) "
            "RETURN properties(n) AS node, labels(n) AS labels, "
            "properties(doc) AS document, "
            "collect(DISTINCT properties(concept))[..5]  AS concepts, "
            "collect(DISTINCT properties(neighbor))[..3] AS neighbors LIMIT 20"
        ),
        "inference": (
            "MATCH (n) WHERE elementId(n) IN $ids "
            "OPTIONAL MATCH (n)-[:DERIVED_FROM]->(src) "
            "OPTIONAL MATCH (n)-[:SUPPORTS]->(tgt) "
            "OPTIONAL MATCH (n)<-[:HAS_CHUNK]-(doc:Document) "
            "OPTIONAL MATCH (downstream:Inference)-[:DERIVED_FROM]->(n) "
            "  WHERE downstream.confidence >= $min_conf "
            "RETURN properties(n) AS node, labels(n) AS labels, "
            "properties(doc) AS document, "
            "collect(DISTINCT properties(src))[..5]        AS sources, "
            "collect(DISTINCT properties(tgt))[..5]        AS supports, "
            "collect(DISTINCT properties(downstream))[..3] AS downstream_inferences "
            "LIMIT 20"
        ),
        "belief": (
            "MATCH (n) WHERE elementId(n) IN $ids "
            "OPTIONAL MATCH (a:Agent)-[:HOLDS]->(n) "
            "OPTIONAL MATCH (n)-[:ABOUT]->(concept:Concept) "
            "OPTIONAL MATCH (n)-[:CONTRADICTS]-(other:Belief) "
            "  WHERE other.confidence >= $min_conf "
            "OPTIONAL MATCH (inf:Inference)-[:SUPPORTS]->(n) "
            "  WHERE inf.confidence >= $min_conf "
            "RETURN properties(n) AS node, labels(n) AS labels, null AS document, "
            "properties(a) AS agent, "
            "collect(DISTINCT properties(concept))[..5] AS concepts, "
            "collect(DISTINCT properties(other))[..3]   AS contradictions, "
            "collect(DISTINCT properties(inf))[..3]     AS supporting_inferences "
            "LIMIT 20"
        ),
    }
    try:
        ids = json.loads(element_ids)
        cyph = _QUERIES.get(intent, _QUERIES["semantic"])
        records = await run(cyph, {
            "ids":       ids,
            "agent_id":  agent_id,
            "min_conf":  min_confidence,
            "now":       _now(),
        })
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
            "OPTIONAL MATCH (n)-[:DERIVED_FROM]->(src) "
            "RETURN properties(n) AS node, labels(n) AS labels, "
            "properties(doc) AS document, properties(root) AS root_document, "
            "collect(properties(concept)) AS concepts, "
            "collect(properties(src)) AS sources",
            {"eid": element_id},
        )
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Semantic node types ────────────────────────────────────────────────────────

@mcp.tool()
async def neo4j_create_memory(
    agent_id: str,
    content: str,
    memory_type: str = "observation",
    concepts: str = "[]",
    confidence: float = 1.0,
    valid_from: str = "",
    valid_to: str = "",
    access_level: str = "",
) -> str:
    """Store a Memory node for an Agent and link it to related Concepts.
    Args:
        agent_id:     Agent id string.
        content:      Memory content text.
        memory_type:  observation | reflection | plan | fact | experience.
        concepts:     JSON array of concept names to link via ABOUT.
        confidence:   Certainty score 0.0–1.0 (default 1.0).
        valid_from:   ISO-8601 start of validity (default: now).
        valid_to:     ISO-8601 end of validity (default: null = forever).
        access_level: ACL string (default: DEFAULT_ACCESS_LEVEL env var).
    """
    try:
        mem_id = str(uuid.uuid4())
        now = _now()
        await run(
            "MERGE (a:Agent {id: $agent_id}) "
            "CREATE (m:Memory {id: $mem_id, content: $content, type: $type, "
            "  agent_id: $agent_id, confidence: $confidence, "
            "  valid_from: $valid_from, valid_to: $valid_to, "
            "  access_level: $access_level, "
            "  created_at: $now, updated_at: $now}) "
            "CREATE (a)-[:HAS_MEMORY]->(m) "
            "RETURN elementId(m) AS element_id",
            {
                "agent_id":     agent_id,
                "mem_id":       mem_id,
                "content":      content,
                "type":         memory_type,
                "confidence":   confidence,
                "valid_from":   valid_from or now,
                "valid_to":     valid_to or None,
                "access_level": access_level or DEFAULT_ACCESS,
                "now":          now,
            },
        )
        for name in json.loads(concepts):
            await run(
                "MERGE (c:Concept {name: $name}) "
                "WITH c MATCH (m:Memory {id: $mid}) "
                "MERGE (m)-[:ABOUT]->(c)",
                {"name": name, "mid": mem_id},
            )
        return json.dumps({
            "memory_id": mem_id, "agent_id": agent_id,
            "type": memory_type, "confidence": confidence,
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_create_goal(
    agent_id: str,
    description: str,
    status: str = "active",
    depends_on_goal_id: str = "",
    confidence: float = 1.0,
    access_level: str = "",
) -> str:
    """Create a Goal node for an Agent with optional DEPENDS_ON link.
    Args:
        agent_id:            Agent id.
        description:         Goal description.
        status:              active | blocked | done | cancelled.
        depends_on_goal_id:  Optional goal id this goal depends on.
        confidence:          Confidence that this goal is achievable (0.0–1.0).
        access_level:        ACL string.
    """
    try:
        goal_id = str(uuid.uuid4())
        now = _now()
        await run(
            "MERGE (a:Agent {id: $agent_id}) "
            "CREATE (g:Goal {id: $goal_id, description: $desc, status: $status, "
            "  agent_id: $agent_id, confidence: $confidence, "
            "  access_level: $access_level, "
            "  valid_from: $now, created_at: $now, updated_at: $now}) "
            "CREATE (a)-[:PURSUES]->(g)",
            {
                "agent_id":     agent_id,
                "goal_id":      goal_id,
                "desc":         description,
                "status":       status,
                "confidence":   confidence,
                "access_level": access_level or DEFAULT_ACCESS,
                "now":          now,
            },
        )
        if depends_on_goal_id:
            await run(
                "MATCH (g:Goal {id: $gid}), (dep:Goal {id: $dep}) "
                "MERGE (g)-[:DEPENDS_ON]->(dep)",
                {"gid": goal_id, "dep": depends_on_goal_id},
            )
        return json.dumps({"goal_id": goal_id, "status": status, "confidence": confidence}, indent=2)
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
        event_type:  user_action | system | tool_call | result | observation.
        agent_id:    Optional agent who triggered/observed the event.
        concepts:    JSON array of concept names the event involves.
    """
    try:
        evt_id = str(uuid.uuid4())
        now = _now()
        await run(
            "CREATE (e:Event {id: $id, description: $desc, type: $type, "
            "  timestamp: $now, access_level: $access_level}) "
            "RETURN elementId(e) AS eid",
            {"id": evt_id, "desc": description, "type": event_type,
             "now": now, "access_level": DEFAULT_ACCESS},
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
        return json.dumps({"event_id": evt_id, "timestamp": now}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_get_agent_context(agent_id: str, min_confidence: float = 0.0) -> str:
    """Fetch everything related to an Agent: goals, memories, beliefs, inferences, events.
    Args:
        agent_id:       Agent id string.
        min_confidence: Minimum confidence for memories/beliefs/inferences (0.0 = all).
    """
    try:
        now = _now()
        records = await run(
            "MATCH (a:Agent {id: $id}) "
            "OPTIONAL MATCH (a)-[:PURSUES]->(g:Goal) "
            "OPTIONAL MATCH (a)-[:HAS_MEMORY]->(m:Memory) "
            "  WHERE m.confidence >= $min_conf "
            "    AND (m.valid_to IS NULL OR m.valid_to >= $now) "
            "OPTIONAL MATCH (a)-[:HOLDS]->(bel:Belief) "
            "  WHERE bel.confidence >= $min_conf "
            "    AND (bel.valid_to IS NULL OR bel.valid_to >= $now) "
            "OPTIONAL MATCH (inf:Inference {agent_id: $id}) "
            "  WHERE inf.confidence >= $min_conf "
            "    AND (inf.valid_to IS NULL OR inf.valid_to >= $now) "
            "OPTIONAL MATCH (a)-[:PARTICIPATED_IN]->(e:Event) "
            "RETURN properties(a)  AS agent, "
            "collect(DISTINCT properties(g))[..20]   AS goals, "
            "collect(DISTINCT properties(m))[..20]   AS memories, "
            "collect(DISTINCT properties(bel))[..20] AS beliefs, "
            "collect(DISTINCT properties(inf))[..20] AS inferences, "
            "collect(DISTINCT properties(e))[..10]   AS events",
            {"id": agent_id, "min_conf": min_confidence, "now": now},
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


# ── Cognitive infrastructure — Inference ───────────────────────────────────────

@mcp.tool()
async def neo4j_create_inference(
    text: str,
    agent_id: str,
    confidence: float,
    source_element_ids: str = "[]",
    supports_element_ids: str = "[]",
    reasoning_trace: str = "",
    valid_from: str = "",
    valid_to: str = "",
    access_level: str = "",
) -> str:
    """
    Create an :Inference node — a reasoned conclusion derived from evidence.

    Inferences are self-reflective: they record not just what was concluded but
    why (reasoning_trace) and how certain the agent is (confidence).

    Args:
        text:                 The conclusion in natural language.
        agent_id:             Agent who derived this inference.
        confidence:           Certainty score 0.0–1.0.
        source_element_ids:   JSON array of elementIds this inference was DERIVED_FROM.
        supports_element_ids: JSON array of Goal/Belief elementIds this SUPPORTS.
        reasoning_trace:      Optional step-by-step reasoning text.
        valid_from:           ISO-8601 validity start (default: now).
        valid_to:             ISO-8601 validity end (default: null = forever).
        access_level:         ACL string.
    """
    try:
        inf_id = str(uuid.uuid4())
        now = _now()
        records = await run(
            "MERGE (a:Agent {id: $agent_id}) "
            "CREATE (inf:Inference { "
            "  id: $inf_id, text: $text, agent_id: $agent_id, "
            "  confidence: $confidence, reasoning_trace: $reasoning_trace, "
            "  valid_from: $valid_from, valid_to: $valid_to, "
            "  access_level: $access_level, "
            "  created_at: $now, updated_at: $now "
            "}) "
            "RETURN elementId(inf) AS element_id",
            {
                "agent_id":       agent_id,
                "inf_id":         inf_id,
                "text":           text,
                "confidence":     confidence,
                "reasoning_trace": reasoning_trace,
                "valid_from":     valid_from or now,
                "valid_to":       valid_to or None,
                "access_level":   access_level or DEFAULT_ACCESS,
                "now":            now,
            },
        )
        element_id = records[0]["element_id"] if records else None

        # Link to evidence sources
        for src_eid in json.loads(source_element_ids):
            await run(
                "MATCH (inf:Inference {id: $inf_id}) "
                "MATCH (src) WHERE elementId(src) = $src_eid "
                "MERGE (inf)-[:DERIVED_FROM]->(src)",
                {"inf_id": inf_id, "src_eid": src_eid},
            )

        # Link to goals/beliefs this inference supports
        for tgt_eid in json.loads(supports_element_ids):
            await run(
                "MATCH (inf:Inference {id: $inf_id}) "
                "MATCH (tgt) WHERE elementId(tgt) = $tgt_eid "
                "MERGE (inf)-[:SUPPORTS]->(tgt)",
                {"inf_id": inf_id, "tgt_eid": tgt_eid},
            )

        return json.dumps({
            "inference_id": inf_id,
            "element_id":   element_id,
            "agent_id":     agent_id,
            "confidence":   confidence,
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Cognitive infrastructure — Belief ──────────────────────────────────────────

@mcp.tool()
async def neo4j_create_belief(
    agent_id: str,
    subject: str,
    content: str,
    confidence: float,
    concepts: str = "[]",
    contradicts_belief_id: str = "",
    valid_from: str = "",
    valid_to: str = "",
    access_level: str = "",
) -> str:
    """
    Create a :Belief node — an Agent's subjective epistemic state.

    Beliefs are distinct from memories (which record observations) and
    inferences (which record conclusions). A belief represents what the
    agent currently holds to be true, with a confidence score.

    Args:
        agent_id:              Agent who holds this belief.
        subject:               Short label for what this belief is about.
        content:               Full belief statement.
        confidence:            Certainty 0.0–1.0.
        concepts:              JSON array of concept names to link via ABOUT.
        contradicts_belief_id: Optional id of a :Belief this one contradicts.
        valid_from:            ISO-8601 validity start (default: now).
        valid_to:              ISO-8601 expiry (default: null).
        access_level:          ACL string.
    """
    try:
        bel_id = str(uuid.uuid4())
        now = _now()
        await run(
            "MERGE (a:Agent {id: $agent_id}) "
            "CREATE (b:Belief { "
            "  id: $bel_id, subject: $subject, content: $content, "
            "  agent_id: $agent_id, confidence: $confidence, "
            "  valid_from: $valid_from, valid_to: $valid_to, "
            "  access_level: $access_level, "
            "  created_at: $now, updated_at: $now "
            "}) "
            "CREATE (a)-[:HOLDS]->(b) "
            "RETURN elementId(b) AS element_id",
            {
                "agent_id":     agent_id,
                "bel_id":       bel_id,
                "subject":      subject,
                "content":      content,
                "confidence":   confidence,
                "valid_from":   valid_from or now,
                "valid_to":     valid_to or None,
                "access_level": access_level or DEFAULT_ACCESS,
                "now":          now,
            },
        )
        for name in json.loads(concepts):
            await run(
                "MERGE (c:Concept {name: $name}) "
                "WITH c MATCH (b:Belief {id: $bid}) "
                "MERGE (b)-[:ABOUT]->(c)",
                {"name": name, "bid": bel_id},
            )
        if contradicts_belief_id:
            await run(
                "MATCH (b:Belief {id: $bid}), (other:Belief {id: $other}) "
                "MERGE (b)-[:CONTRADICTS]->(other)",
                {"bid": bel_id, "other": contradicts_belief_id},
            )
        return json.dumps({
            "belief_id": bel_id, "agent_id": agent_id,
            "subject": subject, "confidence": confidence,
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_resolve_beliefs(
    agent_id: str,
    subject_contains: str = "",
    min_confidence: float = 0.5,
    at_time: str = "",
) -> str:
    """
    Retrieve an Agent's active beliefs, filtered by confidence and temporal validity.

    Args:
        agent_id:         Agent id.
        subject_contains: Optional substring filter on the subject field.
        min_confidence:   Minimum confidence to include (default 0.5).
        at_time:          ISO-8601 timestamp to evaluate validity (default: now).
    """
    try:
        t = at_time or _now()
        records = await run(
            "MATCH (a:Agent {id: $id})-[:HOLDS]->(b:Belief) "
            "WHERE b.confidence >= $min_conf "
            "  AND (b.valid_from IS NULL OR b.valid_from <= $t) "
            "  AND (b.valid_to   IS NULL OR b.valid_to   >= $t) "
            "  AND ($subject = '' OR toLower(b.subject) CONTAINS toLower($subject)) "
            "OPTIONAL MATCH (b)-[:ABOUT]->(c:Concept) "
            "OPTIONAL MATCH (inf:Inference)-[:SUPPORTS]->(b) "
            "RETURN properties(b)  AS belief, "
            "collect(DISTINCT c.name) AS concepts, "
            "collect(DISTINCT properties(inf))[..3] AS supporting_inferences "
            "ORDER BY b.confidence DESC",
            {
                "id":       agent_id,
                "min_conf": min_confidence,
                "t":        t,
                "subject":  subject_contains,
            },
        )
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_check_contradictions(belief_id: str) -> str:
    """
    Find beliefs that CONTRADICT a given belief (bidirectional).

    Args:
        belief_id: Id of the :Belief node to check.
    """
    try:
        records = await run(
            "MATCH (b:Belief {id: $id})-[:CONTRADICTS]-(other:Belief) "
            "OPTIONAL MATCH (a:Agent)-[:HOLDS]->(other) "
            "RETURN properties(b) AS belief, properties(other) AS contradicts, "
            "properties(a) AS held_by",
            {"id": belief_id},
        )
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def neo4j_inferences_for_goal(
    goal_id: str,
    min_confidence: float = 0.5,
) -> str:
    """
    Return all :Inference nodes that SUPPORT a given Goal.

    Args:
        goal_id:        Goal id string.
        min_confidence: Minimum confidence threshold (default 0.5).
    """
    try:
        records = await run(
            "MATCH (g:Goal {id: $id}) "
            "OPTIONAL MATCH (inf:Inference)-[:SUPPORTS]->(g) "
            "  WHERE inf.confidence >= $min_conf "
            "OPTIONAL MATCH (inf)-[:DERIVED_FROM]->(src) "
            "RETURN properties(g) AS goal, "
            "collect(DISTINCT properties(inf))[..20] AS inferences, "
            "collect(DISTINCT properties(src))[..20] AS sources",
            {"id": goal_id, "min_conf": min_confidence},
        )
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Cognitive infrastructure — Temporal ────────────────────────────────────────

@mcp.tool()
async def neo4j_temporal_query(
    label: str,
    at_time: str = "",
    agent_id: str = "",
    min_confidence: float = 0.0,
    limit: int = 20,
) -> str:
    """
    Query nodes of a given label that are valid at a specific point in time.

    Supports all temporal node types: Memory, Inference, Belief, Goal.

    Args:
        label:          Node label: Memory | Inference | Belief | Goal.
        at_time:        ISO-8601 timestamp (default: now).
        agent_id:       Optional — restrict to one agent's partition.
        min_confidence: Minimum confidence (default 0.0 = all).
        limit:          Max results (default 20).
    """
    try:
        allowed = {"Memory", "Inference", "Belief", "Goal"}
        if label not in allowed:
            return json.dumps({"error": f"label must be one of {allowed}"})
        t = at_time or _now()
        records = await run(
            f"MATCH (n:{label}) "
            "WHERE (n.valid_from IS NULL OR n.valid_from <= $t) "
            "  AND (n.valid_to   IS NULL OR n.valid_to   >= $t) "
            "  AND ($agent_id = '' OR n.agent_id = $agent_id) "
            "  AND coalesce(n.confidence, 1.0) >= $min_conf "
            "RETURN properties(n) AS node, labels(n) AS labels "
            "ORDER BY coalesce(n.confidence, 1.0) DESC "
            "LIMIT $limit",
            {"t": t, "agent_id": agent_id, "min_conf": min_confidence, "limit": limit},
        )
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Cognitive infrastructure — Confidence ──────────────────────────────────────

@mcp.tool()
async def neo4j_update_confidence(
    element_id: str,
    confidence: float,
    reason: str = "",
) -> str:
    """
    Update the confidence score of any node by its elementId.

    Also stamps updated_at so the change is traceable in the temporal log.

    Args:
        element_id: Neo4j elementId of the target node.
        confidence: New confidence value 0.0–1.0.
        reason:     Optional text explaining why confidence changed.
    """
    try:
        if not (0.0 <= confidence <= 1.0):
            return json.dumps({"error": "confidence must be between 0.0 and 1.0"})
        records = await run(
            "MATCH (n) WHERE elementId(n) = $eid "
            "SET n.confidence = $conf, n.updated_at = $now "
            + ("SET n.confidence_reason = $reason " if reason else "") +
            "RETURN labels(n) AS labels, n.confidence AS confidence, n.updated_at AS updated_at",
            {"eid": element_id, "conf": confidence, "now": _now(), "reason": reason},
        )
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Cognitive infrastructure — Agent partitions ────────────────────────────────

@mcp.tool()
async def neo4j_agent_partition(
    agent_id: str,
    min_confidence: float = 0.0,
    include_expired: bool = False,
) -> str:
    """
    Return all cognitive nodes (Memory, Inference, Belief, Goal) owned by an Agent.

    Args:
        agent_id:        Agent id string.
        min_confidence:  Minimum confidence filter (default 0.0 = all).
        include_expired: If false, exclude nodes where valid_to < now (default false).
    """
    try:
        now = _now()
        expiry_filter = "" if include_expired else "AND (n.valid_to IS NULL OR n.valid_to >= $now) "
        records = await run(
            "MATCH (n) WHERE n.agent_id = $agent_id "
            "  AND coalesce(n.confidence, 1.0) >= $min_conf "
            + expiry_filter +
            "RETURN labels(n) AS labels, properties(n) AS node "
            "ORDER BY coalesce(n.confidence, 1.0) DESC, n.created_at DESC",
            {"agent_id": agent_id, "min_conf": min_confidence, "now": now},
        )
        return json.dumps(records, default=str, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Cognitive infrastructure — Access control ──────────────────────────────────

@mcp.tool()
async def neo4j_access_check(element_id: str, agent_id: str) -> str:
    """
    Check whether an Agent is permitted to read a node.

    Access rules (evaluated on node.access_level):
      "public"        → allowed for any agent
      "restricted"    → denied for regular agents (system agents only)
      "agent:<id>"    → allowed only if agent_id matches

    Args:
        element_id: Neo4j elementId of the node to check.
        agent_id:   Agent requesting access.
    """
    try:
        records = await run(
            "MATCH (n) WHERE elementId(n) = $eid "
            "RETURN labels(n) AS labels, "
            "coalesce(n.access_level, 'public') AS access_level",
            {"eid": element_id},
        )
        if not records:
            return json.dumps({"allowed": False, "reason": "node not found"})

        access_level = records[0]["access_level"]
        if access_level == "public":
            allowed, reason = True, "public node"
        elif access_level == "restricted":
            allowed, reason = False, "restricted — system agents only"
        elif access_level.startswith("agent:"):
            owner = access_level.split(":", 1)[1]
            allowed = owner == agent_id
            reason = "agent-private — owner match" if allowed else f"agent-private — owned by {owner}"
        else:
            allowed, reason = True, f"unknown access_level '{access_level}' — defaulting to allow"

        return json.dumps({
            "allowed":      allowed,
            "reason":       reason,
            "access_level": access_level,
            "labels":       records[0]["labels"],
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="sse", host=HOST, port=PORT)
