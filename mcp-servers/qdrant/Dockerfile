FROM python:3.12-slim

RUN pip install --no-cache-dir mcp-server-qdrant

EXPOSE 8765

CMD ["mcp-server-qdrant", "--transport", "sse", "--host", "0.0.0.0", "--port", "8765"]
