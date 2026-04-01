"""
Neo4j Graph Schema — EedgeAI GCOR Cognitive Infrastructure.

Node types (primary truth):

  (:Document)   source document (file, URL, API response)
  (:Chunk)      paragraph/section from a Document
  (:Concept)    named entity / key idea extracted from Chunks
  (:Agent)      AI agent or human user
  (:Goal)       task / objective an Agent pursues
  (:Event)      time-stamped occurrence in the world or system
  (:Memory)     an Agent's stored observation or experience
  (:Inference)  a reasoned conclusion derived from evidence  ← COGNITIVE
  (:Belief)     an Agent's subjective belief state            ← COGNITIVE

─────────────────────────────────────────────────────────────────────────────
Temporal properties  (on all mutable node types)
  valid_from    ISO-8601 — when this fact became valid (null = from creation)
  valid_to      ISO-8601 — when this fact expires      (null = still valid)
  updated_at    ISO-8601 — last mutation timestamp

Confidence scoring  (Memory, Inference, Belief)
  confidence    float 0.0–1.0  (1.0 = fully certain)

Agent-specific partitioning
  agent_id      on Memory, Goal, Inference, Belief — isolates per-agent state

Graph-based access control
  access_level  string on all nodes:
                  "public"        — any agent may read
                  "restricted"    — system / admin agents only
                  "agent:<id>"    — that specific agent only
─────────────────────────────────────────────────────────────────────────────

Relationships:
  (:Document)-[:HAS_CHUNK]->(:Chunk)
  (:Chunk)-[:FOLLOWS]->(:Chunk)               sequential order within Document
  (:Chunk)-[:MENTIONS]->(:Concept)
  (:Agent)-[:HAS_MEMORY]->(:Memory)
  (:Agent)-[:PURSUES]->(:Goal)
  (:Agent)-[:PARTICIPATED_IN]->(:Event)
  (:Agent)-[:HOLDS]->(:Belief)                agent holds a belief
  (:Goal)-[:DEPENDS_ON]->(:Goal)
  (:Goal)-[:PRODUCES]->(:Concept)
  (:Memory)-[:ABOUT]->(:Concept)
  (:Event)-[:INVOLVES]->(:Concept)
  (:Event)-[:PRECEDES]->(:Event)
  (:Inference)-[:DERIVED_FROM]->(:Chunk|Memory|Belief|Inference)
  (:Inference)-[:SUPPORTS]->(:Goal|Belief)
  (:Belief)-[:ABOUT]->(:Concept)
  (:Belief)-[:CONTRADICTS]->(:Belief)

Run once to bootstrap the schema:
  python schema.py
  NEO4J_URI=bolt://localhost:7687 python schema.py
"""

import os
import logging

from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")


# ── Uniqueness constraints ─────────────────────────────────────────────────────
CONSTRAINTS = [
    # Core document graph
    "CREATE CONSTRAINT document_id  IF NOT EXISTS FOR (n:Document)  REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id     IF NOT EXISTS FOR (n:Chunk)     REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (n:Concept)   REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT entity_name  IF NOT EXISTS FOR (n:Entity)    REQUIRE n.name IS UNIQUE",
    # Agent world
    "CREATE CONSTRAINT agent_id     IF NOT EXISTS FOR (n:Agent)     REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT goal_id      IF NOT EXISTS FOR (n:Goal)      REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT event_id     IF NOT EXISTS FOR (n:Event)     REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT memory_id    IF NOT EXISTS FOR (n:Memory)    REQUIRE n.id IS UNIQUE",
    # Cognitive / self-reflective nodes
    "CREATE CONSTRAINT inference_id IF NOT EXISTS FOR (n:Inference) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT belief_id    IF NOT EXISTS FOR (n:Belief)    REQUIRE n.id IS UNIQUE",
]


# ── Property indexes ───────────────────────────────────────────────────────────
INDEXES = [
    # ── document graph ────────────────────────────────────────────────────────
    "CREATE INDEX chunk_document  IF NOT EXISTS FOR (n:Chunk)  ON (n.document_id)",
    "CREATE INDEX chunk_position  IF NOT EXISTS FOR (n:Chunk)  ON (n.position)",
    "CREATE INDEX entity_type     IF NOT EXISTS FOR (n:Entity) ON (n.type)",

    # ── event / time ──────────────────────────────────────────────────────────
    "CREATE INDEX event_timestamp IF NOT EXISTS FOR (n:Event)  ON (n.timestamp)",
    "CREATE INDEX event_type      IF NOT EXISTS FOR (n:Event)  ON (n.type)",

    # ── agent world ───────────────────────────────────────────────────────────
    "CREATE INDEX memory_type     IF NOT EXISTS FOR (n:Memory) ON (n.type)",
    "CREATE INDEX memory_agent    IF NOT EXISTS FOR (n:Memory) ON (n.agent_id)",
    "CREATE INDEX goal_status     IF NOT EXISTS FOR (n:Goal)   ON (n.status)",
    "CREATE INDEX goal_agent      IF NOT EXISTS FOR (n:Goal)   ON (n.agent_id)",

    # ── confidence scoring ────────────────────────────────────────────────────
    "CREATE INDEX memory_confidence    IF NOT EXISTS FOR (n:Memory)    ON (n.confidence)",
    "CREATE INDEX inference_confidence IF NOT EXISTS FOR (n:Inference) ON (n.confidence)",
    "CREATE INDEX belief_confidence    IF NOT EXISTS FOR (n:Belief)    ON (n.confidence)",

    # ── temporal validity windows ─────────────────────────────────────────────
    "CREATE INDEX memory_valid_from    IF NOT EXISTS FOR (n:Memory)    ON (n.valid_from)",
    "CREATE INDEX inference_valid_from IF NOT EXISTS FOR (n:Inference) ON (n.valid_from)",
    "CREATE INDEX inference_valid_to   IF NOT EXISTS FOR (n:Inference) ON (n.valid_to)",
    "CREATE INDEX belief_valid_from    IF NOT EXISTS FOR (n:Belief)    ON (n.valid_from)",
    "CREATE INDEX belief_valid_to      IF NOT EXISTS FOR (n:Belief)    ON (n.valid_to)",
    "CREATE INDEX goal_valid_from      IF NOT EXISTS FOR (n:Goal)      ON (n.valid_from)",
    "CREATE INDEX memory_valid_to      IF NOT EXISTS FOR (n:Memory)    ON (n.valid_to)",

    # ── agent partitions (cognitive owners) ───────────────────────────────────
    "CREATE INDEX inference_agent IF NOT EXISTS FOR (n:Inference) ON (n.agent_id)",
    "CREATE INDEX belief_agent    IF NOT EXISTS FOR (n:Belief)    ON (n.agent_id)",

    # ── graph-based access control ────────────────────────────────────────────
    "CREATE INDEX chunk_access     IF NOT EXISTS FOR (n:Chunk)     ON (n.access_level)",
    "CREATE INDEX memory_access    IF NOT EXISTS FOR (n:Memory)    ON (n.access_level)",
    "CREATE INDEX inference_access IF NOT EXISTS FOR (n:Inference) ON (n.access_level)",
    "CREATE INDEX belief_access    IF NOT EXISTS FOR (n:Belief)    ON (n.access_level)",
    "CREATE INDEX goal_access      IF NOT EXISTS FOR (n:Goal)      ON (n.access_level)",
    "CREATE INDEX document_access  IF NOT EXISTS FOR (n:Document)  ON (n.access_level)",
]


# ── Full-text index (keyword fallback search) ──────────────────────────────────
FULLTEXT_INDEX = (
    "CREATE FULLTEXT INDEX nodeIndex IF NOT EXISTS "
    "FOR (n:Document|Chunk|Concept|Memory|Goal|Event|Inference|Belief) "
    "ON EACH [n.text, n.title, n.name, n.description, n.content, "
    "         n.reasoning_trace, n.subject]"
)


def setup_schema(uri: str = URI, user: str = USER, password: str = PASSWORD) -> None:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            logger.info("Creating uniqueness constraints…")
            for stmt in CONSTRAINTS:
                session.run(stmt)
                label_hint = stmt.split("FOR")[1].strip()[:55]
                logger.info("  ✓ %s", label_hint)

            logger.info("Creating property indexes…")
            for stmt in INDEXES:
                session.run(stmt)
                label_hint = stmt.split("FOR")[1].strip()[:55]
                logger.info("  ✓ %s", label_hint)

            logger.info("Creating full-text index…")
            try:
                session.run(FULLTEXT_INDEX)
                logger.info("  ✓ nodeIndex (full-text, all cognitive node types)")
            except Exception as exc:
                logger.warning("  ⚠ Full-text index: %s", exc)

        logger.info("Schema setup complete.")
    finally:
        driver.close()


if __name__ == "__main__":
    setup_schema()
