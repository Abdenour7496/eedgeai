FROM ghcr.io/openclaw/openclaw:2026.11.1

USER root

# Claude Code CLI — the openclaw Anthropic plugin's "claude-cli" backend shells
# out to this for subscription-based auth (Claude Pro/Max login), as opposed
# to its separate metered-API-key auth path (ANTHROPIC_API_KEY). Login itself
# is an interactive OAuth flow and has to be run by a human against a real TTY
# (docker exec -it ... claude login) — this just makes the binary available.
RUN npm install -g @anthropic-ai/claude-code && npm cache clean --force

# `claude login` writes two things: files under ~/.claude/ (covered by the
# openclaw_claude_config volume) AND a top-level ~/.claude.json — which isn't,
# so it silently reverted to a fresh/logged-out state on every container
# recreate. Symlink it into the volume-backed directory so it persists too.
RUN mkdir -p /home/node/.claude && \
    ln -sf /home/node/.claude/.claude.json /home/node/.claude.json && \
    chown -R node:node /home/node/.claude /home/node/.claude.json

# Install document ingestion and MinIO dependencies outside OpenClaw internals.
WORKDIR /opt/oc-drivers
COPY openclaw/package.json .
RUN npm install --production && npm cache clean --force

COPY openclaw/ingest.js openclaw/pdf.js openclaw/image-extractor.js ./

# Expose the scripts as globally callable commands.
RUN chmod +x ingest.js pdf.js && \
    ln -s /opt/oc-drivers/ingest.js  /usr/local/bin/ingest-cli && \
    ln -s /opt/oc-drivers/pdf.js     /usr/local/bin/pdf-cli

# Embed the relay so it runs in the same container (avoids network-namespace
# orphan issues when the openclaw container is recreated).
WORKDIR /opt/oc-relay
COPY openclaw-relay/relay.js .

# Patch the bundled OpenClaw runtime so stale assistant-side provider errors
# do not get misattributed to later fallback candidates.
# NOTE (2026.11.x): upstream rewrote these code paths (bundles are now .mjs,
# per-attempt assistant errors, typed Copilot auth errors), so all three
# patches log a warning and skip. Kept for older base images; the Copilot
# premium-allowance -> rate_limit classification is no longer applied.
COPY openclaw/hotfix-openclaw-runtime.js /tmp/hotfix-openclaw-runtime.js
RUN node /tmp/hotfix-openclaw-runtime.js && rm -f /tmp/hotfix-openclaw-runtime.js

# Wrapper entrypoint: starts relay in background then execs openclaw.
COPY openclaw/entrypoint.sh /usr/local/bin/oc-entrypoint.sh
RUN chmod +x /usr/local/bin/oc-entrypoint.sh

# Reset WORKDIR to /app so the base CMD ("node openclaw.mjs ...") still resolves
# the relative path correctly.
WORKDIR /app
USER node

ENTRYPOINT ["oc-entrypoint.sh"]
CMD ["node", "openclaw.mjs", "gateway", "--allow-unconfigured"]
