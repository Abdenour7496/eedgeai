"""
Neo4j Graph Schema — EedgeAI GCOR Architecture.

Node types (primary truth — Qdrant only indexes these):

  (:Document)  source document (file, URL, API response)
  (:Chunk)     paragraph/section from a Document
  (:Concept)   named entity / key idea extracted from Chunks
  (:Agent)     AI agent or human user
  (:Goal)      task / objective an Agent pursues
  (:Event)     time-stamped occurrence in the world or system
  (:Memory)    an Agent's stored observation or experience

Relationships:
  (:Document)-[:HAS_CHUNK]->(:Chunk)
  (:Chunk)-[:FOLLOWS]->(:Chunk)          sequential order within Document
  (:Chunk)-[:MENTIONS]->(:Concept)
  (:Agent)-[:HAS_MEMORY]->(:Memory)
  (:Agent)-[:PURSUES]->(:Goal)
  (:Agent)-[:PARTICIPATED_IN]->(:Event)
  (:Goal)-[:DEPENDS_ON]->(:Goal)
  (:Goal)-[:PRODUCES]->(:Concept)
  (:Memory)-[:ABOUT]->(:Concept)
  (:Event)-[:INVOLVES]->(:Concept)
  (:Event)-[:PRECEDES]->(:Event)

Run once to bootstrap the schema:
  python schema.py
  # or against a running container:
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

# ── Uniqueness constraints ────────────────────────────────────────────────────
CONSTRAINTS = [
    "CREATE CONSTRAINT document_id  IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id     IF NOT EXISTS FOR (n:Chunk)    REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (n:Concept)  REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT agent_id     IF NOT EXISTS FOR (n:Agent)    REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT goal_id      IF NOT EXISTS FOR (n:Goal)     REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT event_id     IF NOT EXISTS FOR (n:Event)    REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT memory_id    IF NOT EXISTS FOR (n:Memory)   REQUIRE n.id IS UNIQUE",
]

# ── Property indexes (for frequent filter patterns) ───────────────────────────
INDEXES = [
    "CREATE INDEX chunk_document  IF NOT EXISTS FOR (n:Chunk)  ON (n.document_id)",
    "CREATE INDEX chunk_position  IF NOT EXISTS FOR (n:Chunk)  ON (n.position)",
    "CREATE INDEX event_timestamp IF NOT EXISTS FOR (n:Event)  ON (n.timestamp)",
    "CREATE INDEX memory_type     IF NOT EXISTS FOR (n:Memory) ON (n.type)",
    "CREATE INDEX memory_agent    IF NOT EXISTS FOR (n:Memory) ON (n.agent_id)",
    "CREATE INDEX goal_status     IF NOT EXISTS FOR (n:Goal)   ON (n.status)",
    "CREATE INDEX goal_agent      IF NOT EXISTS FOR (n:Goal)   ON (n.agent_id)",
]

# ── Full-text index (for fallback keyword search) ─────────────────────────────
FULLTEXT_INDEX = (
    "CREATE FULLTEXT INDEX nodeIndex IF NOT EXISTS "
    "FOR (n:Document|Chunk|Concept|Memory|Goal|Event) "
    "ON EACH [n.text, n.title, n.name, n.description, n.content]"
)


def setup_schema(uri: str = URI, user: str = USER, password: str = PASSWORD) -> None:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            logger.info("Creating uniqueness constraints…")
            for stmt in CONSTRAINTS:
                session.run(stmt)
                logger.info("  ✓ %s", stmt.split("FOR")[1].strip()[:50])

            logger.info("Creating property indexes…")
            for stmt in INDEXES:
                session.run(stmt)
                logger.info("  ✓ %s", stmt.split("FOR")[1].strip()[:50])

            logger.info("Creating full-text index…")
            try:
                session.run(FULLTEXT_INDEX)
                logger.info("  ✓ nodeIndex (full-text)")
            except Exception as exc:
                logger.warning("  ⚠ Full-text index: %s", exc)

        logger.info("Schema setup complete.")
    finally:
        driver.close()


if __name__ == "__main__":
    setup_schema()
