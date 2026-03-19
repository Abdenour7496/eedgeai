FROM ghcr.io/openclaw/openclaw:latest

USER root

# Install Neo4j and Qdrant Node.js drivers alongside the driver CLI scripts.
# Files land in /opt/oc-drivers so they don't collide with OpenClaw internals.
WORKDIR /opt/oc-drivers
COPY package.json .
RUN npm install --production && npm cache clean --force

COPY neo4j.js qdrant.js ./

# Expose the scripts as globally callable commands.
RUN chmod +x neo4j.js qdrant.js && \
    ln -s /opt/oc-drivers/neo4j.js /usr/local/bin/neo4j-cli && \
    ln -s /opt/oc-drivers/qdrant.js /usr/local/bin/qdrant-cli

# Reset WORKDIR to /app so the base CMD ("node openclaw.mjs ...") still resolves
# the relative path correctly.
WORKDIR /app
USER node
