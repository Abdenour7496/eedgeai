"""
Ingestion pipeline for EedgeAI.

Reads documents from a directory (or stdin), embeds them using the model
specified in agents.yaml (or via --model CLI flag), and upserts the vectors
into the Qdrant collection.

Usage:
    python ingest.py --input ./docs --model openai_large
    python ingest.py --input ./docs --model sentence_transformer
"""

import argparse
import json
import logging
import os
import pathlib
import uuid

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config" / "agents.yaml"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_model(name: str):
    """Return a callable embed(texts: list[str]) -> list[list[float]]."""
    config = load_config()
    models = config.get("embedding_models", {})

    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Available: {list(models)}")

    spec = models[name]
    model_type = spec["type"]
    logger.info("Loading embedding model '%s' (type=%s, model=%s)", name, model_type, spec.get("model"))

    if model_type == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        client = OpenAI(api_key=OPENAI_API_KEY)
        model_id = spec["model"]

        def embed(texts):
            resp = client.embeddings.create(input=texts, model=model_id)
            return [item.embedding for item in resp.data]

        return embed

    elif model_type == "azure":
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY", OPENAI_API_KEY),
            azure_endpoint=spec["endpoint"],
            api_version="2024-02-01",
        )
        deployment = spec["deployment"]

        def embed(texts):
            resp = client.embeddings.create(input=texts, model=deployment)
            return [item.embedding for item in resp.data]

        return embed

    elif model_type == "local":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
        st_model = SentenceTransformer(spec["model"])

        def embed(texts):
            return st_model.encode(texts, convert_to_numpy=True).tolist()

        return embed

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def read_documents(input_path: str):
    """Yield (filename, text) tuples from a directory or single file."""
    p = pathlib.Path(input_path)
    if p.is_file():
        yield p.name, p.read_text(encoding="utf-8", errors="replace")
    elif p.is_dir():
        for fp in sorted(p.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in {".txt", ".md", ".rst", ".json", ".yaml", ".yml"}:
                logger.info("Reading %s", fp)
                yield str(fp.relative_to(p)), fp.read_text(encoding="utf-8", errors="replace")
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")


def chunk_text(text: str, max_chars: int = 1000) -> list:
    """Split text into overlapping chunks."""
    chunks = []
    step = max_chars - 100  # 100-char overlap
    for i in range(0, max(1, len(text)), step):
        chunk = text[i : i + max_chars].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def ensure_collection(collection: str, vector_size: int):
    """Create the Qdrant collection if it does not exist."""
    try:
        import httpx
    except ImportError:
        raise ImportError("Install httpx: pip install httpx")

    url = f"{QDRANT_URL}/collections/{collection}"
    resp = httpx.get(url)
    if resp.status_code == 200:
        logger.info("Collection '%s' already exists.", collection)
        return

    payload = {
        "vectors": {
            "size": vector_size,
            "distance": "Cosine",
        }
    }
    resp = httpx.put(url, json=payload)
    resp.raise_for_status()
    logger.info("Created collection '%s' with vector size %d.", collection, vector_size)


def upsert_points(collection: str, points: list):
    """Upsert a batch of {id, vector, payload} points into Qdrant."""
    try:
        import httpx
    except ImportError:
        raise ImportError("Install httpx: pip install httpx")

    url = f"{QDRANT_URL}/collections/{collection}/points"
    resp = httpx.put(url, json={"points": points}, timeout=30)
    resp.raise_for_status()


def ingest(model_fn, input_path: str, collection: str = QDRANT_COLLECTION, batch_size: int = 32):
    docs = list(read_documents(input_path))
    if not docs:
        logger.warning("No documents found in '%s'.", input_path)
        return

    all_chunks = []
    for filename, text in docs:
        for chunk in chunk_text(text):
            all_chunks.append({"filename": filename, "text": chunk})

    logger.info("Embedding %d chunks from %d documents...", len(all_chunks), len(docs))

    # Embed first chunk to detect vector size
    sample_vectors = model_fn([all_chunks[0]["text"]])
    vector_size = len(sample_vectors[0])
    ensure_collection(collection, vector_size)

    points = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        vectors = model_fn(texts)
        for chunk, vec in zip(batch, vectors):
            points.append({
                "id": str(uuid.uuid4()),
                "vector": vec,
                "payload": {"filename": chunk["filename"], "text": chunk["text"]},
            })
        logger.info("  Embedded %d/%d chunks", min(i + batch_size, len(all_chunks)), len(all_chunks))

    upsert_points(collection, points)
    logger.info("Ingestion complete: %d vectors upserted into '%s'.", len(points), collection)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant vector store.")
    parser.add_argument("--model", default="openai_large", help="Embedding model name from agents.yaml")
    parser.add_argument("--input", required=True, help="Path to file or directory of documents")
    parser.add_argument("--collection", default=QDRANT_COLLECTION, help="Qdrant collection name")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    args = parser.parse_args()

    embed_fn = load_model(args.model)
    ingest(embed_fn, args.input, collection=args.collection, batch_size=args.batch_size)
