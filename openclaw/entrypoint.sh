#!/bin/sh
# Start the relay in the background, then exec the original openclaw entrypoint.
CONFIG_DIR="${OPENCLAW_CONFIG_DIR:-/home/node/.openclaw}"
RUNTIME_CONFIG="$CONFIG_DIR/openclaw.json"
SEED_CONFIG="/opt/openclaw-seed/openclaw.json"

mkdir -p "$CONFIG_DIR"

if [ ! -s "$RUNTIME_CONFIG" ] && [ -f "$SEED_CONFIG" ]; then
	cp "$SEED_CONFIG" "$RUNTIME_CONFIG"
fi

node /opt/oc-relay/relay.js &
exec docker-entrypoint.sh "$@"
