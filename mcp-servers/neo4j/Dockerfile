FROM python:3.12-slim

RUN pip install --no-cache-dir mcp-neo4j-cypher

EXPOSE 8766

# NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
# are supplied via docker-compose environment block.
CMD ["mcp-neo4j-cypher", "--transport", "sse"]
