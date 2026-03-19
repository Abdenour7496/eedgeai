import logging
import os

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

OPENCLAW_URL = os.getenv("OPENCLAW_URL", "http://openclaw:18799")
OPENCLAW_TOKEN = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")


def auth_headers():
    if OPENCLAW_TOKEN:
        return {"Authorization": f"Bearer {OPENCLAW_TOKEN}"}
    return {}


@app.get("/v1/models")
async def list_models():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{OPENCLAW_URL}/v1/models", headers=auth_headers())
            if resp.status_code == 200:
                return resp.json()
            logger.warning("Upstream /v1/models returned status %s", resp.status_code)
    except Exception as exc:
        logger.warning("Could not reach OpenClaw for model list: %s", exc)
    return {
        "object": "list",
        "data": [{"id": "openclaw", "object": "model", "owned_by": "openclaw"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    headers = {"Content-Type": "application/json", **auth_headers()}
    stream = body.get("stream", False)

    if stream:
        async def event_stream():
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    async with client.stream(
                        "POST",
                        f"{OPENCLAW_URL}/v1/chat/completions",
                        json=body,
                        headers=headers,
                    ) as resp:
                        if resp.status_code != 200:
                            logger.error("Upstream streaming error: status %s", resp.status_code)
                            yield f"data: {{\"error\": \"upstream error {resp.status_code}\"}}\n\n".encode()
                            return
                        async for chunk in resp.aiter_bytes():
                            yield chunk
            except Exception as exc:
                logger.error("Streaming request failed: %s", exc)
                yield f"data: {{\"error\": \"proxy error\"}}\n\n".encode()

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{OPENCLAW_URL}/v1/chat/completions",
                    json=body,
                    headers=headers,
                )
                resp.raise_for_status()
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except httpx.HTTPStatusError as exc:
            logger.error("Upstream returned error: %s", exc)
            raise HTTPException(status_code=exc.response.status_code, detail="Upstream error")
        except Exception as exc:
            logger.error("Chat completion request failed: %s", exc)
            raise HTTPException(status_code=502, detail="Could not reach OpenClaw")


@app.get("/health")
async def health():
    return {"status": "ok"}
