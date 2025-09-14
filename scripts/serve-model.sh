#!/bin/sh

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NAMESPACE=demo
export MODE=${1:-s3} # Default to "s3" if not provided
export NAME=$2     # qwen2.5-vl-7b-instruct
export M_PATH=$3
export MODEL_PATH="${M_PATH#/}" # Qwen/Qwen2.5-VL-7B-Instruct/

if [ -z "$NAME" ] || [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 [mode] <name> <model_path>"
  echo "Example: $0 s3 qwen2.5-vl-7b-instruct Qwen/Qwen2.5-VL-7B-Instruct/"
  echo "Example: $0 pvc qwen2.5-7b-instruct Qwen/Qwen2.5-VL-7B-Instruct/"
  echo "Example: $0 oci qwen2.5-7b-instruct registry.redhat.io/rhelai1/modelcar-qwen2-5-7b-instruct-fp8-dynamic:1.5"
  echo "Modes: s3 (default), pvc, oci"
  exit 1
fi

k8s_safe_name() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/-\+/-/g' | sed 's/^-//' | sed 's/-$//'
}

export NAME=$(k8s_safe_name "$NAME")

if [ "$MODE" = "oci" ]; then
  export MODEL_PATH=oci://${MODEL_PATH}
fi

echo "Deploying NAME=${NAME} with PATH=${MODEL_PATH} using MODE=${MODE}"
echo

# Clean up existing resources
oc delete isvc/$NAME -n ${NAMESPACE} --ignore-not-found
oc delete servingruntime/$NAME -n ${NAMESPACE} --ignore-not-found

# Apply serving runtime template
envsubst '$NAME, $MODEL_PATH' \
    < ${BASE}/yaml/infra/sr.yaml.tmpl | oc apply -n ${NAMESPACE} -f -

# Branch based on MODE
if [ "$MODE" = "pvc" ]; then
  echo "Using PVC for model storage"
  envsubst '$NAME, $MODEL_PATH' \
    < ${BASE}/yaml/infra/data-connection-pvc.yaml.tmpl | oc apply -n ${NAMESPACE} -f -

  envsubst '$NAME, $MODEL_PATH' \
    < ${BASE}/yaml/infra/isvc-pvc.yaml.tmpl | oc apply -n ${NAMESPACE} -f -

elif [ "$MODE" = "oci" ]; then
  echo "Using OCI for model storage"
  
  export B64_MODEL_PATH=$(echo "$MODEL_PATH" | base64 -b 0)
  envsubst '$NAME, $B64_MODEL_PATH' \
    < ${BASE}/yaml/infra/data-connection-oci.yaml.tmpl | oc apply -n ${NAMESPACE} -f -

  envsubst '$NAME, $MODEL_PATH' \
    < ${BASE}/yaml/infra/isvc-oci.yaml.tmpl | oc apply -n ${NAMESPACE} -f -

elif [ "$MODE" = "s3" ]; then
  echo "Using S3 for model storage"
  envsubst '$NAME, $MODEL_PATH' \
    < ${BASE}/yaml/infra/isvc-s3.yaml.tmpl | oc apply -n ${NAMESPACE} -f -
fi
