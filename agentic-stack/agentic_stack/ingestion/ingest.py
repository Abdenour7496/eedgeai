"""
Graph-Centric Ingestion Pipeline — GCOR Architecture.

PRINCIPLE: Neo4j is primary truth. Qdrant is a semantic index over Neo4j.

Flow
────
1. Parse document → overlapping text chunks
2. Create (:Document) and (:Chunk) nodes in Neo4j
   - Link chunks to document:  (:Document)-[:HAS_CHUNK]->(:Chunk)
   - Link sequential chunks:   (:Chunk)-[:FOLLOWS]->(:Chunk)
3. Extract concepts (noun phrases) → MERGE (:Concept) nodes
   - (:Chunk)-[:MENTIONS]->(:Concept)
4. Embed each chunk text
5. Upsert to Qdrant with neo4j_element_id in payload
   - Qdrant never holds primary data — only indexes it

Qdrant payload per point:
  {
    "neo4j_element_id": "4:abc:0",   ← graph node reference
    "neo4j_node_id":    <int>,
    "node_type":        "Chunk",
    "text":             "...",
    "document_id":      "...",
    "document_title":   "...",
    "position":         <int>,
    "created_at":       "ISO-8601"
  }

Usage
─────
  pip install httpx pyyaml openai neo4j       # for OpenAI embedding
  pip install sentence-transformers neo4j     # for local embedding

  python ingest.py --input ./docs --model openai_large
  python ingest.py --input ./docs --model sentence_transformer
"""

import argparse
import hashlib
import logging
import os
import pathlib
import re
import uuid
from datetime import datetime, timezone

import httpx
import yaml
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH   = pathlib.Path(__file__).parent.parent / "config" / "agents.yaml"
NEO4J_URI     = os.getenv("NEO4J_URI",     "bolt://localhost:7687")
NEO4J_USER    = os.getenv("NEO4J_USER",    "neo4j")
NEO4J_PASSWORD= os.getenv("NEO4J_PASSWORD","test1234")
QDRANT_URL    = os.getenv("QDRANT_URL",    "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Embedding ──────────────────────────────────────────────────────────────────

def load_model(name: str):
    """Return a callable embed(texts: list[str]) -> list[list[float]]."""
    config  = load_config()
    models  = config.get("embedding_models", {})
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Available: {list(models)}")

    spec       = models[name]
    model_type = spec["type"]
    logger.info("Loading embedding model '%s' (type=%s)", name, model_type)

    if model_type == "openai":
        def embed(texts):
            resp = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={"input": texts, "model": spec["model"]},
                timeout=60,
            )
            resp.raise_for_status()
            return [d["embedding"] for d in resp.json()["data"]]
        return embed

    elif model_type == "azure":
        import os as _os
        key = _os.getenv("AZURE_OPENAI_API_KEY", OPENAI_API_KEY)
        endpoint = spec["endpoint"].rstrip("/")
        deployment = spec["deployment"]
        def embed(texts):
            resp = httpx.post(
                f"{endpoint}/openai/deployments/{deployment}/embeddings?api-version=2024-02-01",
                headers={"api-key": key},
                json={"input": texts},
                timeout=60,
            )
            resp.raise_for_status()
            return [d["embedding"] for d in resp.json()["data"]]
        return embed

    elif model_type == "local":
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(spec["model"])
        def embed(texts):
            return st.encode(texts, convert_to_numpy=True).tolist()
        return embed

    raise ValueError(f"Unsupported model type: {model_type}")


# ── Document reading ───────────────────────────────────────────────────────────

def read_documents(input_path: str):
    """Yield (filename, text) tuples."""
    p = pathlib.Path(input_path)
    if p.is_file():
        yield p.name, p.read_text(encoding="utf-8", errors="replace")
    elif p.is_dir():
        exts = {".txt", ".md", ".rst", ".json", ".yaml", ".yml"}
        for fp in sorted(p.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in exts:
                logger.info("Reading %s", fp)
                yield str(fp.relative_to(p)), fp.read_text(encoding="utf-8", errors="replace")
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> list:
    """Split text into overlapping chunks."""
    step = max_chars - overlap
    chunks = []
    for i in range(0, max(1, len(text)), step):
        c = text[i : i + max_chars].strip()
        if c:
            chunks.append(c)
    return chunks


# ── Concept extraction ────────────────────────────────────────────────────────

_STOP = frozenset(
    "the a an is are was were be been being have has had do does did "
    "will would could should may might shall can and or but not in on at "
    "to for of with by from up about into through during before after above "
    "below between this that these those it its i we you he she they them "
    "what which who whom whose how when where why all both each few more "
    "most other some such no nor only same so than too very just".split()
)

def extract_concepts(text: str, max_concepts: int = 10) -> list:
    """Extract candidate concept phrases using simple heuristics."""
    words = re.findall(r"[A-Za-z][a-z]*(?:\s+[A-Z][a-z]+)*", text)
    seen, concepts = set(), []
    for w in words:
        lower = w.lower().strip()
        if len(lower) > 3 and lower not in _STOP and lower not in seen:
            seen.add(lower)
            concepts.append(w.strip())
            if len(concepts) >= max_concepts:
                break
    return concepts


# ── Neo4j: graph-first writes ─────────────────────────────────────────────────

def neo4j_create_document(session, doc_id: str, title: str, source: str) -> str:
    """MERGE a Document node and return its elementId."""
    result = session.run(
        """
        MERGE (d:Document {id: $id})
        ON CREATE SET d.title    = $title,
                      d.source   = $source,
                      d.created_at = $now
        ON MATCH  SET d.updated_at = $now
        RETURN elementId(d) AS eid
        """,
        id=doc_id, title=title, source=source, now=datetime.now(timezone.utc).isoformat(),
    )
    return result.single()["eid"]


def neo4j_create_chunk(session, chunk_id: str, doc_id: str,
                        text: str, position: int) -> str:
    """Create a Chunk node linked to its Document; return elementId."""
    result = session.run(
        """
        MERGE (c:Chunk {id: $id})
        ON CREATE SET c.text        = $text,
                      c.position    = $position,
                      c.document_id = $doc_id,
                      c.created_at  = $now
        WITH c
        MATCH (d:Document {id: $doc_id})
        MERGE (d)-[:HAS_CHUNK]->(c)
        RETURN elementId(c) AS eid
        """,
        id=chunk_id, text=text, position=position,
        doc_id=doc_id, now=datetime.now(timezone.utc).isoformat(),
    )
    return result.single()["eid"]


def neo4j_link_sequential_chunks(session, prev_id: str, curr_id: str) -> None:
    """Create FOLLOWS relationship between consecutive chunks."""
    session.run(
        """
        MATCH (a:Chunk {id: $prev}), (b:Chunk {id: $curr})
        MERGE (a)-[:FOLLOWS]->(b)
        """,
        prev=prev_id, curr=curr_id,
    )


def neo4j_create_concepts(session, chunk_id: str, concepts: list) -> None:
    """MERGE Concept nodes and link them to the Chunk."""
    for name in concepts:
        session.run(
            """
            MERGE (concept:Concept {name: $name})
            WITH concept
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (c)-[:MENTIONS]->(concept)
            """,
            name=name, chunk_id=chunk_id,
        )


# ── Qdrant: semantic index ────────────────────────────────────────────────────

def qdrant_ensure_collection(collection: str, vector_size: int) -> None:
    url = f"{QDRANT_URL}/collections/{collection}"
    resp = httpx.get(url, timeout=10)
    if resp.status_code == 200:
        return
    resp2 = httpx.put(
        url,
        json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        timeout=10,
    )
    resp2.raise_for_status()
    logger.info("Created Qdrant collection '%s' (dim=%d)", collection, vector_size)


def qdrant_upsert_points(collection: str, points: list) -> None:
    resp = httpx.put(
        f"{QDRANT_URL}/collections/{collection}/points",
        json={"points": points},
        timeout=60,
    )
    resp.raise_for_status()


# ── Main ingestion ─────────────────────────────────────────────────────────────

def ingest(
    embed_fn,
    input_path: str,
    collection: str = QDRANT_COLLECTION,
    batch_size: int = 32,
) -> None:
    docs = list(read_documents(input_path))
    if not docs:
        logger.warning("No documents found in '%s'.", input_path)
        return

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    all_chunks = []   # list of {chunk_id, doc_id, doc_title, text, position, element_id}

    # ── Phase 1: write graph (Neo4j is primary truth) ──────────────────────────
    with driver.session() as session:
        for filename, text in docs:
            doc_id    = hashlib.sha1(filename.encode()).hexdigest()
            doc_title = pathlib.Path(filename).stem
            neo4j_create_document(session, doc_id, doc_title, filename)
            logger.info("Document '%s' (id=%s)", filename, doc_id)

            chunks   = chunk_text(text)
            prev_id  = None

            for pos, chunk_text_val in enumerate(chunks):
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{pos}"))

                # Create Chunk node, get elementId
                element_id = neo4j_create_chunk(session, chunk_id, doc_id, chunk_text_val, pos)

                # Chain sequential chunks
                if prev_id:
                    neo4j_link_sequential_chunks(session, prev_id, chunk_id)
                prev_id = chunk_id

                # Extract and link concepts
                concepts = extract_concepts(chunk_text_val)
                neo4j_create_concepts(session, chunk_id, concepts)

                all_chunks.append({
                    "chunk_id":       chunk_id,
                    "element_id":     element_id,  # Neo4j element ID (GCOR reference)
                    "doc_id":         doc_id,
                    "doc_title":      doc_title,
                    "text":           chunk_text_val,
                    "position":       pos,
                })

    driver.close()
    logger.info("Graph phase complete: %d chunks written to Neo4j.", len(all_chunks))

    # ── Phase 2: embed + index in Qdrant (secondary — references Neo4j) ────────
    sample_vectors = embed_fn([all_chunks[0]["text"]])
    vector_size    = len(sample_vectors[0])
    qdrant_ensure_collection(collection, vector_size)

    points = []
    for i in range(0, len(all_chunks), batch_size):
        batch   = all_chunks[i : i + batch_size]
        vectors = embed_fn([c["text"] for c in batch])

        for chunk, vec in zip(batch, vectors):
            points.append({
                "id":     chunk["chunk_id"],
                "vector": vec,
                "payload": {
                    # GCOR: reference back to the Neo4j node
                    "neo4j_element_id": chunk["element_id"],
                    "neo4j_node_id":    None,          # integer id deprecated in Neo4j 5
                    "node_type":        "Chunk",
                    # Searchable metadata
                    "text":             chunk["text"],
                    "document_id":      chunk["doc_id"],
                    "document_title":   chunk["doc_title"],
                    "position":         chunk["position"],
                    "created_at":       datetime.now(timezone.utc).isoformat(),
                },
            })

        logger.info("  Embedded %d/%d chunks", min(i + batch_size, len(all_chunks)), len(all_chunks))

    qdrant_upsert_points(collection, points)
    logger.info(
        "Index phase complete: %d vectors upserted to Qdrant collection '%s'.",
        len(points), collection,
    )
    logger.info(
        "GCOR ingestion done. Neo4j = primary truth (%d chunks). "
        "Qdrant = semantic index (%d points).",
        len(all_chunks), len(points),
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph-Centric GCOR ingestion pipeline.")
    parser.add_argument("--input",      required=True, help="Path to file or directory")
    parser.add_argument("--model",      default="openai_large", help="Embedding model from agents.yaml")
    parser.add_argument("--collection", default=QDRANT_COLLECTION, help="Qdrant collection name")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    args = parser.parse_args()

    embed_fn = load_model(args.model)
    ingest(embed_fn, args.input, collection=args.collection, batch_size=args.batch_size)
