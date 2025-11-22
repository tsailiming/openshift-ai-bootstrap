#!/bin/sh

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NAMESPACE=demo
export MODE=${1:-s3} # Default to "s3" if not provided
export NAME=$2     # qwen2.5-vl-7b-instruct
export M_PATH=$3
export EXTRA_VLLM_ARGS="$4"
export MODEL_PATH="${M_PATH#/}" # Qwen/Qwen2.5-VL-7B-Instruct/

if [ -z "$NAME" ] || [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 [mode] <name> <model_path>"
  echo "Example: $0 s3 qwen2.5-vl-7b-instruct Qwen/Qwen2.5-VL-7B-Instruct/ \"--max-model-len 4096 --gpu-memory-utilization 0.93\""
  echo "Example: $0 pvc qwen2.5-7b-instruct Qwen/Qwen2.5-VL-7B-Instruct/"
  echo "Example: $0 oci qwen2.5-7b-instruct oci://registry.redhat.io/rhelai1/modelcar-qwen2-5-7b-instruct-fp8-dynamic:1.5"
  echo "Modes: s3 (default), pvc, oci"
  exit 1
fi

k8s_safe_name() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/-\+/-/g' | sed 's/^-//' | sed 's/-$//'
}

export NAME=$(k8s_safe_name "$NAME")

# if [ "$MODE" = "oci" ]; then
#   export MODEL_PATH=${MODEL_PATH}
# fi

IFS=' ' read -r -a ARG_ARRAY <<< "$EXTRA_VLLM_ARGS"
VLLM_ARGS=$(printf '%s\n' $EXTRA_VLLM_ARGS | jq -R . | jq -s .)

printf "%-15s-+-%s\n" "---------------" "-----------------------------"
printf "%-15s | %s\n" "ISVC Name" "$NAME"
printf "%-15s | %s\n" "Storage Mode" "$MODE"
printf "%-15s | %s\n" "Model Path" "$M_PATH"
printf "%-15s | %s\n" "Extra vLLM Args" "$EXTRA_VLLM_ARGS"
printf "%-15s-+-%s\n" "---------------" "-----------------------------"

echo

# Clean up existing resources. Deleting so the old pod gets terminated
oc delete isvc/$NAME -n ${NAMESPACE} --ignore-not-found
oc delete servingruntime/$NAME -n ${NAMESPACE} --ignore-not-found

yq eval '
  .metadata.annotations."openshift.io/display-name" = env(NAME) |
  .metadata.name = env(NAME)
' ${BASE}/yaml/infra/sr.yaml.tmpl | oc apply -n ${NAMESPACE} -f -

# Branch based on MODE
if [ "$MODE" = "pvc" ]; then
  echo "Using PVC for model storage"
  
  oc delete secret/model-pvc-${NAME} -n ${NAMESPACE} --ignore-not-found

  yq eval '
  .metadata.name = "model-pvc-" + env(NAME) |
  .metadata.annotations."openshift.io/display-name" = "model-pvc-" + env(NAME) |
  .dataString.URI = "pvc://models-pvc/" + env(MODEL_PATH)
  ' ${BASE}/yaml/infra/data-connection-pvc.yaml.tmpl | oc apply -n ${NAMESPACE} -f -

  yq eval '
    .metadata.annotations."openshift.io/display-name" = env(NAME) |
    .metadata.name = env(NAME) |
    .spec.predictor.model.runtime = env(NAME) |
    .spec.predictor.model.storageUri = "pvc://models-pvc/" + env(MODEL_PATH) |
    .spec.predictor.model.args = env(VLLM_ARGS) |
    .spec.predictor.model.args style=""
  ' ${BASE}/yaml/infra/isvc-pvc.yaml.tmpl | oc apply -n ${NAMESPACE} -f -

  
elif [ "$MODE" = "oci" ]; then
  echo "Using OCI for model storage"
    
  oc delete secret/${NAME} -n ${NAMESPACE} --ignore-not-found  
  
  yq eval '
  .metadata.annotations."openshift.io/display-name" = env(NAME) |
  .metadata.name = env(NAME) |
  .data.URI = (env(MODEL_PATH) | @base64)
  ' ${BASE}/yaml/infra/data-connection-oci.yaml.tmpl | oc apply -n ${NAMESPACE} -f -

  yq eval '
    .metadata.annotations."openshift.io/display-name" = env(NAME) |
    .metadata.name = env(NAME) |
    .spec.predictor.model.runtime = env(NAME) |
    .spec.predictor.model.storageUri = env(MODEL_PATH) |
    .spec.predictor.model.args = env(VLLM_ARGS) |
    .spec.predictor.model.args style=""
  ' ${BASE}/yaml/infra/isvc-oci.yaml.tmpl | oc apply -n ${NAMESPACE} -f -

  
elif [ "$MODE" = "s3" ]; then
  echo "Using S3 for model storage"
  
  yq eval '
  .metadata.annotations."openshift.io/display-name" = env(NAME) |
  .metadata.name = env(NAME) |
  .spec.predictor.model.runtime = env(NAME) |
  .spec.predictor.model.storage.path = env(MODEL_PATH) |
  .spec.predictor.model.args = env(VLLM_ARGS) |
  .spec.predictor.model.args style=""
' ${BASE}/yaml/infra/isvc-s3.yaml.tmpl | oc apply -n ${NAMESPACE} -f -

fi
