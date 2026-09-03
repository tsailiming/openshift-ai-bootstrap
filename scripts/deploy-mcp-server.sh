#!/bin/sh

set -eu

usage() {
    echo "Usage: $0 <namespace> <mcp-server-name> <manifest>" >&2
    exit 1
}

[ "$#" -eq 3 ] || usage

NAMESPACE="$1"
MCP_SERVER="$2"
MANIFEST="$3"

TIMEOUT="${MCP_WAIT_TIMEOUT:-120s}"

echo "Applying MCPServerRegistration from $MANIFEST..."

CLUSTER_DOMAIN="${CLUSTER_DOMAIN:?CLUSTER_DOMAIN must be set}" \
    envsubst < "$MANIFEST" | oc apply -f -

echo "Waiting for MCPServerRegistration/$MCP_SERVER to be ready (timeout: $TIMEOUT)..."

if ! oc wait \
    -n "$NAMESPACE" \
    "mcpserverregistration/$MCP_SERVER" \
    --for=condition=Ready \
    --timeout="$TIMEOUT"
then
    echo "ERROR: MCPServerRegistration/$MCP_SERVER did not become ready." >&2
    oc get mcpserverregistration "$MCP_SERVER" \
        -n "$NAMESPACE" \
        -o yaml >&2 || true
    exit 1
fi

echo "MCPServerRegistration/$MCP_SERVER is ready."