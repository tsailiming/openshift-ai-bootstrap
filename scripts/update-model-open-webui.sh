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
SERVICE_URL="http://${ISVC_NAME}-predictor.${NAMESPACE}.svc.cluster.local"
FULL_URL="${SERVICE_URL}:8080/v1"

# Get the first pod for this predictor
POD_NAME=$(oc get pod -l "serving.kserve.io/inferenceservice=$ISVC_NAME" -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}')
MODEL_ID=$(oc rsh -c kserve-container "$POD_NAME" /usr/bin/curl -s "http://localhost:8080/v1/models" | jq -r '.data[0].id')

echo "Model URL: $FULL_URL"
echo "Model ID: $MODEL_ID"

# Check if FULL_URL or MODEL_ID is empty
if [[ -z "$FULL_URL" || -z "$MODEL_ID" ]]; then
  echo "Error: Model URL or Model ID is empty."
  exit 1
fi

# Get current URLs from the ConfigMap
CURRENT_URLS=$(oc get cm "$CONFIGMAP_NAME" -n "$NAMESPACE" -o jsonpath='{.data.OPENAI_API_BASE_URLS}')

# Check if FULL_URL is already in the list
if [[ "$CURRENT_URLS" == *"$FULL_URL"* ]]; then
  echo "URL already exists in OPENAI_API_BASE_URLS."
else
  # Append FULL_URL to existing URLs, using semicolon as separator
  if [[ -z "$CURRENT_URLS" ]]; then
    UPDATED_URLS="$FULL_URL"
  else
    UPDATED_URLS="$CURRENT_URLS;$FULL_URL"
  fi

  echo "Updating ConfigMap with new model URL"
  oc patch cm "$CONFIGMAP_NAME" -n "$NAMESPACE" \
    --type merge \
    -p "{\"data\":{\"OPENAI_API_BASE_URLS\":\"$UPDATED_URLS\",\"OPENAI_API_KEYS\":\"\"}}"

  echo "Restarting OpenWebUI deployment..."
  oc rollout restart deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE"
fi

# oc patch cm "$CONFIGMAP_NAME" -n "$NAMESPACE" \
#   --type merge \
#   -p "{\"data\":{\"OPENAI_API_BASE_URLS\":\"$FULL_URL\",\"OPENAI_API_KEYS\":\"\"}}"



