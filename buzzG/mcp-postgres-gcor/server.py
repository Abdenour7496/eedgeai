"""
mcp-postgres-gcor  –  MCP (SSE) server exposing three tools to Buzz agents:

  semantic_search    – vector similarity search over gcor.chunks
  graph_expand       – recursive graph expansion from seed nodes
  ingest_document    – proxy call to gcor-proxy /api/ingest
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from sse_starlette.sse import EventSourceResponse


# ── Settings ──────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    gcor_proxy_url: str = "http://gcor-proxy:5001"
    server_host: str = "0.0.0.0"
    server_port: int = 8765

    class Config:
        env_file = ".env"


cfg = Settings()
app = FastAPI(title="mcp-postgres-gcor")

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "semantic_search",
        "description": (
            "Embed the query and cosine-search gcor.chunks via pgvector. "
            "Returns the top-k most relevant text chunks plus expanded graph nodes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":        {"type": "string",  "description": "Natural-language query"},
                "top_k":        {"type": "integer", "description": "Number of chunks to return", "default": 8},
                "hops":         {"type": "integer", "description": "Graph expansion depth", "default": 2},
                "access_level": {"type": "string",  "description": "Filter: public | team | private"},
                "agent_id":     {"type": "string",  "description": "Filter by agent scope"},
                "min_confidence": {"type": "number","description": "Minimum confidence threshold", "default": 0.0},
            },
            "required": ["query"],
        },
    },
    {
        "name": "graph_expand",
        "description": (
            "Recursively expand N hops from given node UUIDs through gcor.edges "
            "(CONTAINS, DERIVED_FROM, HOLDS, ABOUT, CONTRADICTS, ...). "
            "Returns related Memory/Belief/Inference/Concept nodes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Seed node UUIDs to expand from",
                },
                "hops":   {"type": "integer", "description": "Expansion depth", "default": 2},
            },
            "required": ["node_ids"],
        },
    },
    {
        "name": "ingest_document",
        "description": "Ingest raw text into the GCOR pipeline (chunk, embed, store in Postgres + MinIO).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text":         {"type": "string",  "description": "Document text to ingest"},
                "title":        {"type": "string",  "description": "Human-readable title"},
                "access_level": {"type": "string",  "description": "public | team | private", "default": "private"},
                "agent_id":     {"type": "string",  "description": "Agent that owns this document"},
                "source_type":  {"type": "string",  "description": "file | message | git_event", "default": "message"},
                "source_ref":   {"type": "string",  "description": "Buzz event id / git sha / etc."},
            },
            "required": ["text", "title"],
        },
    },
]

# ── Tool dispatch ─────────────────────────────────────────────────────────────

async def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=cfg.gcor_proxy_url, timeout=60) as client:
        if name == "semantic_search":
            r = await client.post("/api/retrieve", json=args)
            r.raise_for_status()
            return r.json()

        elif name == "graph_expand":
            # Direct graph expansion: POST /api/retrieve with empty query + seed nodes
            # We re-use the retrieve endpoint with a sentinel; alternatively hit DB directly.
            # For simplicity, proxy via retrieve with an explicit node_ids hint.
            node_ids = args.get("node_ids", [])
            hops     = args.get("hops", 2)
            r = await client.post(
                "/api/retrieve",
                json={"query": "", "top_k": 0, "hops": hops, "_seed_node_ids": node_ids},
            )
            r.raise_for_status()
            return r.json()

        elif name == "ingest_document":
            data = {
                "text":         args["text"],
                "title":        args.get("title", "untitled"),
                "access_level": args.get("access_level", "private"),
                "source_type":  args.get("source_type", "message"),
            }
            if "agent_id"   in args: data["agent_id"]   = args["agent_id"]
            if "source_ref" in args: data["source_ref"] = args["source_ref"]
            r = await client.post("/api/ingest", data=data)
            r.raise_for_status()
            return r.json()

        else:
            return {"error": f"unknown tool: {name}"}


# ── MCP SSE endpoint ──────────────────────────────────────────────────────────

@app.get("/sse")
async def sse_endpoint(request: Request):
    async def event_generator() -> AsyncGenerator[dict, None]:
        # Send tool list on connect
        yield {
            "event": "tools",
            "data": json.dumps({"tools": TOOLS}),
        }
        # Keep connection alive; tool calls arrive via POST /call
        while True:
            if await request.is_disconnected():
                break
            import asyncio
            await asyncio.sleep(15)
            yield {"event": "ping", "data": "{}"}

    return EventSourceResponse(event_generator())


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = {}


@app.post("/call")
async def call_endpoint(req: ToolCallRequest):
    result = await call_tool(req.name, req.arguments)
    return {"result": result}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=cfg.server_host, port=cfg.server_port, reload=False)
