"""
Neo4j Prometheus exporter for Community Edition.
Connects via Bolt, runs lightweight count queries, exposes /metrics on port 9188.
"""
import os, time, logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

NEO4J_URI  = os.getenv("NEO4J_URI",      "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "test1234")
NEO4J_DB   = os.getenv("NEO4J_DATABASE", "neo4j")
PORT       = int(os.getenv("EXPORTER_PORT", "9188"))
SCRAPE_SEC = int(os.getenv("SCRAPE_INTERVAL", "15"))

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ── Metrics cache ──────────────────────────────────────────────────────────────

_cache: dict = {}
_last_scrape: float = 0
_up: int = 0

def _collect():
    global _cache, _last_scrape, _up
    now = time.monotonic()
    if now - _last_scrape < SCRAPE_SEC:
        return
    _last_scrape = now
    try:
        with driver.session(database=NEO4J_DB) as s:
            # Node / relationship counts
            r = s.run("MATCH (n) RETURN count(n) AS c").single()
            _cache["neo4j_nodes_total"] = r["c"] if r else 0

            r = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()
            _cache["neo4j_relationships_total"] = r["c"] if r else 0

            # Domain-specific counts
            r = s.run("MATCH (d:Document) WHERE NOT d.archived RETURN count(d) AS c").single()
            _cache["neo4j_documents_active"] = r["c"] if r else 0

            r = s.run("MATCH (d:Document) WHERE d.archived RETURN count(d) AS c").single()
            _cache["neo4j_documents_archived"] = r["c"] if r else 0

            r = s.run("MATCH (c:Chunk) RETURN count(c) AS c").single()
            _cache["neo4j_chunks_total"] = r["c"] if r else 0

            r = s.run("MATCH (a:ArchivedDocument) RETURN count(a) AS c").single()
            _cache["neo4j_archived_documents_total"] = r["c"] if r else 0

            r = s.run("MATCH (a:ArchivedCollection) RETURN count(a) AS c").single()
            _cache["neo4j_archived_collections_total"] = r["c"] if r else 0

            # Label counts
            r = s.run("CALL db.labels() YIELD label RETURN count(label) AS c").single()
            _cache["neo4j_label_types_total"] = r["c"] if r else 0

            # Relationship type count
            r = s.run("CALL db.relationshipTypes() YIELD relationshipType RETURN count(relationshipType) AS c").single()
            _cache["neo4j_relationship_types_total"] = r["c"] if r else 0

        _up = 1
    except Exception as e:
        log.warning("Neo4j scrape failed: %s", e)
        _up = 0

# ── Prometheus text format ─────────────────────────────────────────────────────

HELP = {
    "neo4j_up":                          ("gauge",   "1 if Neo4j is reachable"),
    "neo4j_nodes_total":                 ("gauge",   "Total nodes in the graph"),
    "neo4j_relationships_total":         ("gauge",   "Total relationships in the graph"),
    "neo4j_documents_active":            ("gauge",   "Active (non-archived) Document nodes"),
    "neo4j_documents_archived":          ("gauge",   "Archived Document nodes"),
    "neo4j_chunks_total":                ("gauge",   "Total Chunk nodes"),
    "neo4j_archived_documents_total":    ("gauge",   "ArchivedDocument records"),
    "neo4j_archived_collections_total":  ("gauge",   "ArchivedCollection records"),
    "neo4j_label_types_total":           ("gauge",   "Distinct node label types"),
    "neo4j_relationship_types_total":    ("gauge",   "Distinct relationship types"),
}

def _render() -> str:
    _collect()
    lines = []
    for name, (kind, desc) in HELP.items():
        lines.append(f"# HELP {name} {desc}")
        lines.append(f"# TYPE {name} {kind}")
        val = _cache.get(name, 0) if name != "neo4j_up" else _up
        lines.append(f"{name} {val}")
    return "\n".join(lines) + "\n"

# ── HTTP server ────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass  # suppress request logs
    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404); self.end_headers(); return
        body = _render().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

if __name__ == "__main__":
    log.info("Neo4j exporter listening on :%d", PORT)
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
