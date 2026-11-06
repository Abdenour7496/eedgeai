# Builds the upstream block/buzz binaries our Buzz integration needs, plus our
# own deterministic reply adapter (see reply_adapter.py):
#   buzz-acp     — connects to the Buzz relay (WebSocket + Nostr auth), watches
#                  whatever channel(s) channel_supervisor.sh currently passes it,
#                  and bridges messages to an ACP agent over stdio.
#   channel_supervisor.sh — computes buzz-acp's --channels from this identity's
#                  live channel membership (open channels self-joined
#                  automatically; private ones need a one-time owner invite)
#                  and relaunches buzz-acp only when that set changes.
#   reply_adapter.py — the actual agent_command. Deterministically calls the GCOR
#                  proxy and always posts the answer via `buzz messages send` —
#                  no LLM tool-call decision in the loop (see file header for why:
#                  buzz-agent + a shell tool answered questions but often never
#                  chose to call the "post" tool).
#   buzz (aka "buzz-cli" here) — same binary as the `buzz` CLI; used by
#                  reply_adapter.py to post, and for our own one-off admin
#                  tasks (`users set-profile`, `channels add-member`, ...).
#   buzz-agent, buzz-dev-mcp — still built and present, unused by default; kept
#                  around in case the LLM-tool-calling approach is revisited.
#   curl         — reply_adapter.py shells out to it for multipart uploads to
#                  the proxy's /v1/buzz/ingest (image attachments, storage_key
#                  lookups); avoids hand-rolling multipart/form-data in stdlib.
# Pinned to a release tag so this doesn't move under us on rebuild.
FROM rust:1.99-bookworm AS builder

ARG BUZZ_REF=v0.4.24
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake nasm pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --branch ${BUZZ_REF} --depth 1 https://github.com/block/buzz.git .

# Build only the crates we need — the workspace also contains buzz-relay,
# buzz-db, etc., which pull in Postgres/heavier deps we don't want here.
RUN cargo build --release -p buzz-acp -p buzz-agent -p buzz-cli -p buzz-dev-mcp

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates python3 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 buzzbridge

COPY --from=builder /build/target/release/buzz-acp     /usr/local/bin/buzz-acp
COPY --from=builder /build/target/release/buzz-agent   /usr/local/bin/buzz-agent
COPY --from=builder /build/target/release/buzz-dev-mcp /usr/local/bin/buzz-dev-mcp
# The buzz-cli crate's binary is named `buzz` upstream — install it under both names.
COPY --from=builder /build/target/release/buzz /usr/local/bin/buzz
RUN ln -s /usr/local/bin/buzz /usr/local/bin/buzz-cli
COPY reply_adapter.py /usr/local/bin/reply_adapter.py
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY channel_supervisor.sh /usr/local/bin/channel_supervisor.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/reply_adapter.py /usr/local/bin/channel_supervisor.sh

USER buzzbridge
WORKDIR /home/buzzbridge

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
