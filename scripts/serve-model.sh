#!/usr/bin/env bash

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NAMESPACE=demo

export MODE=${1:-s3}
export NAME=$2
export M_PATH=$3
export EXTRA_VLLM_ARGS="$4"
export MODEL_PATH="${M_PATH#/}"

if [ -z "$NAME" ] || [ -z "$MODEL_PATH" ] || [ -z "$EXTRA_VLLM_ARGS" ]; then
  echo "Usage: $0 [mode] <name> <model_path> <extra_vllm_args>"
  echo
  echo "Examples:"
  echo "  $0 s3 qwen2.5-vl-7b-instruct Qwen/Qwen2.5-VL-7B-Instruct/ \"--tensor-parallel-size 1\""
  echo "  $0 s3 qwen2.5-vl-7b-instruct Qwen/Qwen2.5-VL-7B-Instruct/ \"--tensor-parallel-size 2 --max-model-len 4096\""
  echo "  $0 pvc qwen2.5-7b-instruct Qwen/Qwen2.5-VL-7B-Instruct/ \"--tensor-parallel-size 1\""
  echo "  $0 oci qwen2.5-7b-instruct oci://registry.redhat.io/rhelai1/modelcar-qwen2-5-7b-instruct-fp8-dynamic:1.5 \"--tensor-parallel-size 4\""
  echo
  echo "Modes: s3 (default), pvc, oci"
  exit 1
fi

# --------------------------------------------------------------------
# Extract tensor parallel size from vLLM arguments
# --------------------------------------------------------------------

IFS=' ' read -r -a ARG_ARRAY <<< "$EXTRA_VLLM_ARGS"

TP_SIZE=""

for ((i=0; i<${#ARG_ARRAY[@]}; i++)); do
  case "${ARG_ARRAY[$i]}" in
    --tensor-parallel-size)
      if (( i + 1 >= ${#ARG_ARRAY[@]} )); then
        echo "ERROR: --tensor-parallel-size requires a value"
        exit 1
      fi

      TP_SIZE="${ARG_ARRAY[$((i + 1))]}"
      ;;

    --tensor-parallel-size=*)
      TP_SIZE="${ARG_ARRAY[$i]#*=}"
      ;;
  esac
done

if [ -z "$TP_SIZE" ]; then
  echo "ERROR: --tensor-parallel-size is required"
  echo "       The number of GPUs is derived from --tensor-parallel-size."
  exit 1
fi

# Validate tensor parallel size
case "$TP_SIZE" in
  ''|*[!0-9]*|0)
    echo "ERROR: --tensor-parallel-size must be a positive integer"
    exit 1
    ;;
esac

# --------------------------------------------------------------------
# Kubernetes-safe name
# --------------------------------------------------------------------

k8s_safe_name() {
    echo "$1" |
      tr '[:upper:]' '[:lower:]' |
      sed 's/[^a-z0-9]/-/g' |
      sed 's/-\+/-/g' |
      sed 's/^-//' |
      sed 's/-$//'
}

export NAME=$(k8s_safe_name "$NAME")

# --------------------------------------------------------------------
# Convert vLLM arguments to JSON array
# --------------------------------------------------------------------

export VLLM_ARGS=$(printf '%s\n' $EXTRA_VLLM_ARGS | jq -R . | jq -s .)

# --------------------------------------------------------------------
# Display configuration
# --------------------------------------------------------------------

printf "%-20s-+-%s\n" "--------------------" "-----------------------------"
printf "%-20s | %s\n" "ISVC Name" "$NAME"
printf "%-20s | %s\n" "Storage Mode" "$MODE"
printf "%-20s | %s\n" "Model Path" "$M_PATH"
printf "%-20s | %s\n" "Tensor Parallel Size" "$TP_SIZE"
printf "%-20s | %s\n" "NVIDIA GPU Count" "$TP_SIZE"
printf "%-20s | %s\n" "Extra vLLM Args" "$EXTRA_VLLM_ARGS"
printf "%-20s-+-%s\n" "--------------------" "-----------------------------"

echo

# --------------------------------------------------------------------
# Clean up existing resources.
# Deleting so the old pod gets terminated.
# --------------------------------------------------------------------

oc delete isvc/$NAME -n ${NAMESPACE} --ignore-not-found
oc delete servingruntime/$NAME -n ${NAMESPACE} --ignore-not-found

# --------------------------------------------------------------------
# ServingRuntime
# --------------------------------------------------------------------

yq eval '
  .metadata.annotations."openshift.io/display-name" = env(NAME) |
  .metadata.name = env(NAME)
' ${BASE}/yaml/infra/sr.yaml.tmpl |
  oc apply -n ${NAMESPACE} -f -

# --------------------------------------------------------------------
# PVC
# --------------------------------------------------------------------

if [ "$MODE" = "pvc" ]; then

  echo "Using PVC for model storage"

  oc delete secret/model-pvc-${NAME} \
    -n ${NAMESPACE} \
    --ignore-not-found

  yq eval '
    .metadata.name = "model-pvc-" + env(NAME) |
    .metadata.annotations."openshift.io/display-name" = "model-pvc-" + env(NAME) |
    .dataString.URI = "pvc://models-pvc/" + env(MODEL_PATH)
  ' ${BASE}/yaml/infra/data-connection-pvc.yaml.tmpl |
    oc apply -n ${NAMESPACE} -f -

  yq eval '
    .metadata.annotations."openshift.io/display-name" = env(NAME) |
    .metadata.name = env(NAME) |
    .spec.predictor.model.runtime = env(NAME) |
    .spec.predictor.model.storageUri = "pvc://models-pvc/" + env(MODEL_PATH) |
    .spec.predictor.model.args = env(VLLM_ARGS) |
    .spec.predictor.model.args style="" |
    .spec.predictor.model.resources.limits."nvidia.com/gpu" = env(TP_SIZE) |
    .spec.predictor.model.resources.requests."nvidia.com/gpu" = env(TP_SIZE)
  ' ${BASE}/yaml/infra/isvc-pvc.yaml.tmpl |
    oc apply -n ${NAMESPACE} -f -

# --------------------------------------------------------------------
# OCI
# --------------------------------------------------------------------

elif [ "$MODE" = "oci" ]; then

  echo "Using OCI for model storage"

  oc delete secret/${NAME} \
    -n ${NAMESPACE} \
    --ignore-not-found

  yq eval '
    .metadata.annotations."openshift.io/display-name" = env(NAME) |
    .metadata.name = env(NAME) |
    .data.URI = (env(MODEL_PATH) | @base64)
  ' ${BASE}/yaml/infra/data-connection-oci.yaml.tmpl |
    oc apply -n ${NAMESPACE} -f -

  yq eval '
    .metadata.annotations."openshift.io/display-name" = env(NAME) |
    .metadata.name = env(NAME) |
    .spec.predictor.model.runtime = env(NAME) |
    .spec.predictor.model.storageUri = env(MODEL_PATH) |
    .spec.predictor.model.args = env(VLLM_ARGS) |
    .spec.predictor.model.args style="" |
    .spec.predictor.model.resources.limits."nvidia.com/gpu" = env(TP_SIZE) |
    .spec.predictor.model.resources.requests."nvidia.com/gpu" = env(TP_SIZE)
  ' ${BASE}/yaml/infra/isvc-oci.yaml.tmpl |
    oc apply -n ${NAMESPACE} -f -

# --------------------------------------------------------------------
# S3
# --------------------------------------------------------------------

elif [ "$MODE" = "s3" ]; then

  echo "Using S3 for model storage"

  yq eval '
    .metadata.annotations."openshift.io/display-name" = env(NAME) |
    .metadata.name = env(NAME) |
    .spec.predictor.model.runtime = env(NAME) |
    .spec.predictor.model.storage.path = env(MODEL_PATH) |
    .spec.predictor.model.args = env(VLLM_ARGS) |
    .spec.predictor.model.args style="" |
    .spec.predictor.model.resources.limits."nvidia.com/gpu" = env(TP_SIZE) |
    .spec.predictor.model.resources.requests."nvidia.com/gpu" = env(TP_SIZE)
  ' ${BASE}/yaml/infra/isvc-s3.yaml.tmpl |
    oc apply -n ${NAMESPACE} -f -

# --------------------------------------------------------------------
# Invalid mode
# --------------------------------------------------------------------

else

  echo "ERROR: Unknown mode '$MODE'"
  echo "Modes: s3, pvc, oci"
  exit 1

fi