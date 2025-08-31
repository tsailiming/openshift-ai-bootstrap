#!/bin/sh
# update-openwebui-from-isvc.sh
# Usage: ./update-openwebui-from-isvc.sh <isvc_name> [namespace]
# Example: ./update-openwebui-from-isvc.sh qwen25-vl-7b-instruct demo

set -e

ISVC_NAME=${1:?Error: ISVC name required}
NAMESPACE=${2:-demo}
CONFIGMAP_NAME="openwebui-config"
DEPLOYMENT_NAME="open-webui"

SERVICE_URL=$(oc get isvc "$ISVC_NAME" -n "$NAMESPACE" -o jsonpath='{.status.url}')
FULL_URL="${SERVICE_URL}:8080/v1"

# Get the first pod for this predictor
POD_NAME=$(oc get pod -l "serving.kserve.io/inferenceservice=$ISVC_NAME" -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}')
MODEL_ID=$(oc rsh "$POD_NAME" /usr/bin/curl -s -k "$FULL_URL/models" | jq -r '.data[0].id')

echo "Model URL: $FULL_URL"
echo "Model ID: $MODEL_ID"
echo "Updating OpenWebUI ConfigMap..."

oc patch cm "$CONFIGMAP_NAME" -n "$NAMESPACE" \
  --type merge \
  -p "{\"data\":{\"OPENAI_API_BASE_URLS\":\"$FULL_URL\",\"OPENAI_API_KEYS\":\"\"}}"

echo "Restarting OpenWebUI deployment..."
oc rollout restart deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE"