#!/bin/sh

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NAMESPACE=demo
export MODE=${1:-s3} # Default to "s3" if not provided
export NAME=$2     # qwen2.5-vl-7b-instruct
export M_PATH=$3
export MODEL_PATH="${M_PATH#/}" # Qwen/Qwen2.5-VL-7B-Instruct/

if [ -z "$NAME" ] || [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 [mode] <name> <model_path>"
  echo "Example: $0 s3 qwen2.5-vl-7b-instruct Qwen/Qwen2.5-VL-7B-Instruct/ "
  echo "Modes: s3 (default), pvc"
  exit 1
fi

k8s_safe_name() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/-\+/-/g' | sed 's/^-//' | sed 's/-$//'
}

export NAME=$(k8s_safe_name "$NAME")
echo "Deploying NAME=${NAME} with PATH=${MODEL_PATH} using MODE=${MODE}"
echo

# Clean up existing resources
oc delete isvc/$NAME -n ${NAMESPACE} --ignore-not-found
oc delete servingruntime/$NAME -n ${NAMESPACE} --ignore-not-found

# Apply serving runtime template
envsubst '$NAME, $MODEL_PATH' \
    < ${BASE}/yaml/demo/sr.yaml.tmpl | oc apply -n ${NAMESPACE} -f -

# Branch based on MODE
if [ "$MODE" = "pvc" ]; then
  echo "Using PVC mode for model storage"
  envsubst '$NAME, $MODEL_PATH' \
    < ${BASE}/yaml/infra/data-connection-pvc.yaml.tmpl | oc apply -n ${NAMESPACE} -f -

  envsubst '$NAME, $MODEL_PATH' \
    < ${BASE}/yaml/demo/isvc-pvc.yaml.tmpl | oc apply -n ${NAMESPACE} -f -

else
  echo "Using S3 mode for model storage"
  envsubst '$NAME, $MODEL_PATH' \
    < ${BASE}/yaml/demo/isvc-s3.yaml.tmpl | oc apply -n ${NAMESPACE} -f -
fi
