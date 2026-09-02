#!/bin/sh

set -eu

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Installing MCP Gateway Operator"
oc apply -f ${BASE}/yaml/rhoai/mcp-gateway.yaml
${BASE}/scripts/check-operator-install-status.sh mcp-gateway mcp-gateway


# Get the OpenShift ingress domain.
CLUSTER_DOMAIN=$(
    oc get ingress.config.openshift.io cluster \
        -o jsonpath='{.spec.domain}'
)

if [ -z "$CLUSTER_DOMAIN" ]; then
    echo "ERROR: Could not determine OpenShift cluster domain." >&2
    exit 1
fi

# Get the default ingress certificate secret.
CERT_NAME=$(
    oc get ingresscontroller default -n openshift-ingress-operator \
        -o jsonpath='{.spec.defaultCertificate.name}' 2>/dev/null || true
)

[ -z "$CERT_NAME" ] && CERT_NAME="router-certs-default"

export CLUSTER_DOMAIN
export CERT_NAME

echo "Cluster domain: $CLUSTER_DOMAIN"
echo "Certificate:    $CERT_NAME"

envsubst '${CLUSTER_DOMAIN} ${CERT_NAME}' < $BASE/yaml/rhoai/mcp-default-gateway.yaml.tmpl | oc apply -f -
oc apply -f ${BASE}/yaml/rhoai/mcp-gateway-extension.yaml
oc apply -f ${BASE}/yaml/rhoai/mcp-gateway-ref-grant.yaml