FROM ghcr.io/openclaw/openclaw:latest

USER root

# Install Neo4j and Qdrant Node.js drivers alongside the driver CLI scripts.
# Files land in /opt/oc-drivers so they don't collide with OpenClaw internals.
WORKDIR /opt/oc-drivers
COPY openclaw/package.json .
RUN npm install --production && npm cache clean --force

COPY openclaw/neo4j.js openclaw/qdrant.js openclaw/ingest.js ./

# Expose the scripts as globally callable commands.
RUN chmod +x neo4j.js qdrant.js ingest.js && \
    ln -s /opt/oc-drivers/neo4j.js   /usr/local/bin/neo4j-cli  && \
    ln -s /opt/oc-drivers/qdrant.js  /usr/local/bin/qdrant-cli && \
    ln -s /opt/oc-drivers/ingest.js  /usr/local/bin/ingest-cli

# Embed the relay so it runs in the same container (avoids network-namespace
# orphan issues when the openclaw container is recreated).
WORKDIR /opt/oc-relay
COPY openclaw-relay/relay.js .

# Wrapper entrypoint: starts relay in background then execs openclaw.
COPY openclaw/entrypoint.sh /usr/local/bin/oc-entrypoint.sh
RUN chmod +x /usr/local/bin/oc-entrypoint.sh

# Reset WORKDIR to /app so the base CMD ("node openclaw.mjs ...") still resolves
# the relative path correctly.
WORKDIR /app
USER node

ENTRYPOINT ["oc-entrypoint.sh"]
CMD ["node", "openclaw.mjs", "gateway", "--allow-unconfigured"]
