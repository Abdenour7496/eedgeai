#!/bin/sh
# Start the relay in the background, then exec the original openclaw entrypoint.
node /opt/oc-relay/relay.js &
exec docker-entrypoint.sh "$@"
