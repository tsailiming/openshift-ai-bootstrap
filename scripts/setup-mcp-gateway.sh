#!/bin/sh

set -eu

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Installing MCP Gateway Operator"
oc apply -f "${BASE}/yaml/rhoai/mcp-gateway.yaml"

"${BASE}/scripts/check-operator-install-status.sh" \
    mcp-gateway \
    mcp-gateway


# ----------------------------------------------------------------------
# Get OpenShift cluster domain
# ----------------------------------------------------------------------

CLUSTER_DOMAIN=$(
    oc get ingress.config.openshift.io cluster \
        -o jsonpath='{.spec.domain}'
)

if [ -z "$CLUSTER_DOMAIN" ]; then
    echo "ERROR: Could not determine OpenShift cluster domain." >&2
    exit 1
fi


# ----------------------------------------------------------------------
# Get default ingress certificate
# ----------------------------------------------------------------------

CERT_NAME=$(
    oc get ingresscontroller default \
        -n openshift-ingress-operator \
        -o jsonpath='{.spec.defaultCertificate.name}' \
        2>/dev/null || true
)

[ -z "$CERT_NAME" ] && CERT_NAME="router-certs-default"

export CLUSTER_DOMAIN
export CERT_NAME

echo "Cluster domain: $CLUSTER_DOMAIN"
echo "Certificate:    $CERT_NAME"


# ----------------------------------------------------------------------
# Create Gateway
# ----------------------------------------------------------------------

echo "Creating MCP Gateway..."

envsubst '${CLUSTER_DOMAIN} ${CERT_NAME}' \
    < "${BASE}/yaml/rhoai/mcp-default-gateway.yaml.tmpl" \
    | oc apply -f -


# ----------------------------------------------------------------------
# Create MCPGatewayExtension and ReferenceGrant
# ----------------------------------------------------------------------

echo "Creating MCP Gateway Extension..."

oc apply -f "${BASE}/yaml/rhoai/mcp-gateway-extension.yaml"

echo "Creating ReferenceGrant..."

oc apply -f "${BASE}/yaml/rhoai/mcp-gateway-ref-grant.yaml"


# ----------------------------------------------------------------------
# Wait for mcp-gateway Service
# ----------------------------------------------------------------------

echo "Waiting for mcp-gateway Service..."

if ! oc wait \
    --for=jsonpath='{.spec.ports[?(@.port==8080)].port}'=8080 \
    service/mcp-gateway \
    -n mcp-gateway \
    --timeout=120s \
    >/dev/null 2>&1
then
    echo "ERROR: mcp-gateway Service is not ready." >&2
    oc get service mcp-gateway -n mcp-gateway
    exit 1
fi

echo "mcp-gateway Service is ready."


# ----------------------------------------------------------------------
# Wait for mcp-gateway Service endpoints
# ----------------------------------------------------------------------

echo "Waiting for mcp-gateway endpoints..."

ENDPOINTS=""

for _ in $(seq 1 60); do
    ENDPOINTS=$(
        oc get endpoints mcp-gateway \
            -n mcp-gateway \
            -o jsonpath='{.subsets[*].addresses[*].ip}' \
            2>/dev/null || true
    )

    if [ -n "$ENDPOINTS" ]; then
        break
    fi

    sleep 2
done

if [ -z "$ENDPOINTS" ]; then
    echo "ERROR: mcp-gateway has no ready endpoints." >&2
    oc get endpoints mcp-gateway -n mcp-gateway -o yaml
    exit 1
fi

echo "mcp-gateway endpoint is ready."


# ----------------------------------------------------------------------
# Wait for Gateway to be programmed
# ----------------------------------------------------------------------

echo "Waiting for Gateway to be programmed..."

if ! oc wait \
    --for=condition=Programmed \
    gateway/mcp-default-gateway \
    -n openshift-ingress \
    --timeout=120s \
    >/dev/null 2>&1
then
    echo "ERROR: Gateway is not programmed." >&2
    oc get gateway mcp-default-gateway \
        -n openshift-ingress \
        -o yaml
    exit 1
fi

echo "Gateway is programmed."


# ----------------------------------------------------------------------
# Wait for HTTPRoute to be created
# ----------------------------------------------------------------------

echo "Waiting for MCP HTTPRoute..."

ROUTE=""

for _ in $(seq 1 60); do
    ROUTE=$(
        oc get httproute -A \
            -o jsonpath='{range .items[?(@.spec.parentRefs[0].name=="mcp-default-gateway")]}{.metadata.namespace}{" "}{.metadata.name}{"\n"}{end}' \
            2>/dev/null \
            | head -n 1
    )

    if [ -n "$ROUTE" ]; then
        break
    fi

    sleep 2
done

if [ -z "$ROUTE" ]; then
    echo "ERROR: MCP HTTPRoute was not created." >&2
    oc get httproute -A
    exit 1
fi

ROUTE_NAMESPACE=$(echo "$ROUTE" | awk '{print $1}')
ROUTE_NAME=$(echo "$ROUTE" | awk '{print $2}')

echo "HTTPRoute: ${ROUTE_NAMESPACE}/${ROUTE_NAME}"


# ----------------------------------------------------------------------
# Wait for HTTPRoute Accepted + ResolvedRefs
# ----------------------------------------------------------------------

echo "Waiting for HTTPRoute to be accepted..."

ACCEPTED=""
RESOLVED=""

for _ in $(seq 1 60); do

    ACCEPTED=$(
        oc get httproute "$ROUTE_NAME" \
            -n "$ROUTE_NAMESPACE" \
            -o jsonpath='{.status.parents[0].conditions[?(@.type=="Accepted")].status}' \
            2>/dev/null || true
    )

    RESOLVED=$(
        oc get httproute "$ROUTE_NAME" \
            -n "$ROUTE_NAMESPACE" \
            -o jsonpath='{.status.parents[0].conditions[?(@.type=="ResolvedRefs")].status}' \
            2>/dev/null || true
    )

    if [ "$ACCEPTED" = "True" ] && [ "$RESOLVED" = "True" ]; then
        break
    fi

    sleep 2
done

if [ "$ACCEPTED" != "True" ] || [ "$RESOLVED" != "True" ]; then
    echo "ERROR: HTTPRoute is not ready." >&2

    oc get httproute "$ROUTE_NAME" \
        -n "$ROUTE_NAMESPACE" \
        -o yaml

    exit 1
fi

echo "HTTPRoute is accepted and references are resolved."


# ----------------------------------------------------------------------
# Test MCP endpoint
# ----------------------------------------------------------------------

MCP_URL="http://mcp.${CLUSTER_DOMAIN}:8080/mcp"

echo "Testing MCP Gateway: ${MCP_URL}"

if ! curl \
    --silent \
    --show-error \
    --fail \
    --output /dev/null \
    -X POST "$MCP_URL" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d '{"jsonrpc": "2.0", "id": 1, "method": "initialize"}'
then
    echo "ERROR: MCP Gateway test failed." >&2
    exit 1
fi

echo "MCP Gateway test succeeded."
echo "MCP Gateway is ready."