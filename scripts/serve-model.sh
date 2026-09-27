#!/usr/bin/env bash
#
# Deploy a model with classic InferenceService (default) or MaaS + llm-d
# (baseRefs → v3-5-0-kserve-config-llm-single-node-template-nvidia-cuda on maas-default-gateway).
#
# Usage:
#   scripts/serve-model.sh [options] <name> <uri>
#
# Storage scheme is taken from the URI: pvc:// | oci://
#

set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TARGET="${TARGET:-isvc}"
REPLICAS="${REPLICAS:-1}"
VLLM_ARGS_STR="${VLLM_ARGS_STR:-}"
NAMESPACE="${NAMESPACE:-demo}"
MAAS_NS="${MAAS_NS:-models-as-a-service}"
OWNER_GROUP="${OWNER_GROUP:-system:authenticated}"
TOKEN_LIMIT="${TOKEN_LIMIT:-100000}"
TOKEN_WINDOW="${TOKEN_WINDOW:-1h}"
PRIORITY="${PRIORITY:-99}"
# RHOAI 3.5 ships version-prefixed LLMInferenceServiceConfig names in redhat-ods-applications.
BASE_REF="${BASE_REF:-v3-5-0-kserve-config-llm-single-node-template-nvidia-cuda}"

usage() {
  cat <<EOF
Usage:
  $0 [options] <name> <uri>

Arguments:
  <name>                 Model name / served-model ID.
                         Kubernetes resource names are derived from this.

  <uri>                  Model source:
                           pvc://models-pvc/<path>
                           oci://<image>

Options:
  --target TARGET        Deployment target: isvc or maas
                         Default: isvc

  --namespace NS         Kubernetes namespace
                         Default: demo

  --replicas N           Number of model replicas
                         Default: 1

  --vllm-args ARGS       Additional vLLM arguments.
                         Example:
                           --vllm-args "--tensor-parallel-size 2 --max-model-len 4096"

  -h, --help             Show this help

Examples:

  # Deploy to an InferenceService
  $0 \\
    qwen25-7b \\
    oci://registry.redhat.io/rhelai1/modelcar-qwen2-5-7b-instruct-fp8-dynamic:1.5

  # Deploy with vLLM configuration
  $0 \\
    qwen25-7b \\
    oci://registry.redhat.io/rhelai1/modelcar-qwen2-5-7b-instruct-fp8-dynamic:1.5 \\
    --vllm-args "--tensor-parallel-size 2 --max-model-len 4096"

  # Deploy to MaaS
  $0 \\
    --target maas \\
    gpt-oss-20b \\
    oci://registry.redhat.io/rhelai1/modelcar-gpt-oss-20b:1.5 \\
    --vllm-args "--tensor-parallel-size 2 --max-model-len 4096"

  # Deploy a PVC-backed model to MaaS
  $0 \\
    --target maas \\
    --namespace llm \\
    qwen35-27b-fp8 \\
    pvc://models-pvc/Qwen/Qwen3.5-27B-FP8 \\
    --vllm-args "--tensor-parallel-size 2 --max-model-len 4096"
EOF
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="$2"
      shift 2
      ;;
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --replicas)
      REPLICAS="$2"
      shift 2
      ;;
    --vllm-args)
      VLLM_ARGS_STR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "ERROR: Unknown option: $1"
      usage
      exit 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ne 2 ]]; then
  echo "ERROR: expected <name> <uri>"
  usage
  exit 1
fi

RAW_NAME="${POSITIONAL[0]}"
MODEL_URI="${POSITIONAL[1]}"

case "$TARGET" in
  isvc|maas) ;;
  *)
    echo "ERROR: --target must be isvc or maas (got: $TARGET)"
    exit 1
    ;;
esac

case "$REPLICAS" in
  ''|*[!0-9]*|0)
    echo "ERROR: --replicas must be a positive integer"
    exit 1
    ;;
esac

# --------------------------------------------------------------------
# Infer storage mode from URI scheme
# --------------------------------------------------------------------

case "$MODEL_URI" in
  pvc://*)
    MODE=pvc
    PVC_NAME="${MODEL_URI#pvc://}"
    PVC_NAME="${PVC_NAME%%/*}"
    if [ -z "$PVC_NAME" ]; then
      echo "ERROR: pvc:// URI must include a PVC name (e.g. pvc://models-pvc/<path>)"
      exit 1
    fi
    if ! oc get pvc "${PVC_NAME}" -n "${NAMESPACE}" >/dev/null 2>&1; then
      echo "ERROR: PVC '${PVC_NAME}' not found in namespace '${NAMESPACE}'"
      echo "       Create it first, or pass --namespace where the PVC exists."
      exit 1
    fi
    ;;
  oci://*)
    MODE=oci
    ;;
  *)
    echo "ERROR: uri must start with pvc:// or oci:// (got: $MODEL_URI)"
    usage
    exit 1
    ;;
esac

# --------------------------------------------------------------------
# Extract tensor parallel size from vLLM arguments (default: 1)
# --------------------------------------------------------------------

TP_SIZE="1"
if [ -n "$VLLM_ARGS_STR" ]; then
  # shellcheck disable=SC2206
  ARG_ARRAY=($VLLM_ARGS_STR)

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
fi

case "$TP_SIZE" in
  ''|*[!0-9]*|0)
    echo "ERROR: --tensor-parallel-size must be a positive integer (got: $TP_SIZE)"
    exit 1
    ;;
esac

# --------------------------------------------------------------------
# Kubernetes-safe name; served / display name = positional name
# --------------------------------------------------------------------

k8s_safe_name() {
  echo "$1" |
    tr '[:upper:]' '[:lower:]' |
    sed 's/[^a-z0-9]/-/g' |
    sed 's/-\+/-/g' |
    sed 's/^-//' |
    sed 's/-$//'
}

export NAME
NAME=$(k8s_safe_name "$RAW_NAME")
export NAME

export SERVED_NAME="$RAW_NAME"
export DISPLAY_NAME="$RAW_NAME"
export REPLICAS
export TP_SIZE
export GPUS="$TP_SIZE"
export NAMESPACE
# Templates historically used MAAS_MODEL_NS for the model/workload namespace.
export MAAS_MODEL_NS="$NAMESPACE"
export MAAS_NS
export OWNER_GROUP
export TOKEN_LIMIT
export TOKEN_WINDOW
export PRIORITY
export BASE_REF
export MODEL_URI
export MODE

# User vLLM args (--enable-force-include-usage is always prepended in the maas tmpl)
export VLLM_EXTRA_ARGS="$VLLM_ARGS_STR"

# --------------------------------------------------------------------
# Convert vLLM arguments to JSON array (isvc path)
# --------------------------------------------------------------------

if [ -n "$VLLM_ARGS_STR" ]; then
  export VLLM_ARGS
  VLLM_ARGS=$(printf '%s\n' $VLLM_ARGS_STR | jq -R . | jq -s .)
  export VLLM_ARGS
else
  export VLLM_ARGS='[]'
fi

# --------------------------------------------------------------------
# Display configuration
# --------------------------------------------------------------------

printf "%-22s-+-%s\n" "----------------------" "-----------------------------"
printf "%-22s | %s\n" "Target" "$TARGET"
printf "%-22s | %s\n" "Name" "$NAME"
printf "%-22s | %s\n" "Served name" "$SERVED_NAME"
printf "%-22s | %s\n" "URI" "$MODEL_URI"
printf "%-22s | %s\n" "Storage" "$MODE"
printf "%-22s | %s\n" "Namespace" "$NAMESPACE"
printf "%-22s | %s\n" "Replicas" "$REPLICAS"
printf "%-22s | %s\n" "Tensor Parallel Size" "$TP_SIZE"
printf "%-22s | %s\n" "NVIDIA GPU / replica" "$TP_SIZE"
printf "%-22s | %s\n" "Extra vLLM Args" "${VLLM_ARGS_STR:-(none)}"
if [ "$TARGET" = "maas" ]; then
  printf "%-22s | %s\n" "baseRefs" "$BASE_REF"
  printf "%-22s | %s\n" "Gateway" "maas-default-gateway"
fi
printf "%-22s-+-%s\n" "----------------------" "-----------------------------"
echo

# ====================================================================
# Target: maas (LLMInferenceService + MaaS governance)
# ====================================================================

deploy_maas() {
  echo "Ensuring namespace ${NAMESPACE}..."
  oc get ns "${NAMESPACE}" >/dev/null 2>&1 || oc create ns "${NAMESPACE}"
  oc label ns "${NAMESPACE}" \
    maas.opendatahub.io/gateway-access=true \
    opendatahub.io/dashboard=true \
    opendatahub.io/generated-namespace=true \
    --overwrite >/dev/null

  echo "Removing prior LLMInferenceService/${NAME} (if any)..."
  oc delete llminferenceservice/"${NAME}" -n "${NAMESPACE}" --ignore-not-found

  echo "Applying LLMInferenceService (baseRefs: ${BASE_REF})..."
  # shellcheck disable=SC2016
  envsubst '$NAME $SERVED_NAME $DISPLAY_NAME $MODEL_URI $REPLICAS $GPUS $MAAS_MODEL_NS $VLLM_EXTRA_ARGS $BASE_REF' \
    < "${BASE}/yaml/infra/llmisvc-maas.yaml.tmpl" |
    oc apply -f -

  echo "Applying MaaSModelRef..."
  # shellcheck disable=SC2016
  envsubst '$NAME $SERVED_NAME $DISPLAY_NAME $MAAS_MODEL_NS' \
    < "${BASE}/yaml/infra/maas-modelref.yaml.tmpl" |
    oc apply -f -

  echo "Applying MaaSAuthPolicy..."
  # shellcheck disable=SC2016
  envsubst '$NAME $DISPLAY_NAME $MAAS_MODEL_NS $MAAS_NS $OWNER_GROUP' \
    < "${BASE}/yaml/infra/maas-auth-policy.yaml.tmpl" |
    oc apply -f -

  echo "Applying MaaSSubscription..."
  # shellcheck disable=SC2016
  envsubst '$NAME $DISPLAY_NAME $MAAS_MODEL_NS $MAAS_NS $OWNER_GROUP $TOKEN_LIMIT $TOKEN_WINDOW $PRIORITY' \
    < "${BASE}/yaml/infra/maas-subscription.yaml.tmpl" |
    oc apply -f -

  echo
  echo "Deployed. Check status with:"
  echo "  oc get llmisvc ${NAME} -n ${NAMESPACE}"
  echo "  oc get maasmodelref ${NAME} -n ${NAMESPACE}"
}

if [ "$TARGET" = "maas" ]; then
  echo "Using ${MODE}: ${MODEL_URI}"
  if [ "$MODE" = "pvc" ]; then
    echo "NOTE: PVC must exist in namespace ${NAMESPACE} (or be cluster-accessible as referenced)."
  fi
  deploy_maas
  exit 0
fi

# ====================================================================
# Target: isvc (existing InferenceService path)
# ====================================================================

oc delete isvc/"$NAME" -n "${NAMESPACE}" --ignore-not-found
oc delete servingruntime/"$NAME" -n "${NAMESPACE}" --ignore-not-found

yq eval '
  .metadata.annotations."openshift.io/display-name" = env(NAME) |
  .metadata.name = env(NAME)
' "${BASE}/yaml/infra/sr.yaml.tmpl" |
  oc apply -n "${NAMESPACE}" -f -

if [ "$MODE" = "pvc" ]; then

  echo "Using PVC for model storage: ${MODEL_URI}"

  oc delete secret/model-pvc-"${NAME}" \
    -n "${NAMESPACE}" \
    --ignore-not-found

  yq eval '
    .metadata.name = "model-pvc-" + env(NAME) |
    .metadata.annotations."openshift.io/display-name" = "model-pvc-" + env(NAME) |
    .dataString.URI = env(MODEL_URI)
  ' "${BASE}/yaml/infra/data-connection-pvc.yaml.tmpl" |
    oc apply -n "${NAMESPACE}" -f -

  yq eval '
    .metadata.annotations."openshift.io/display-name" = env(NAME) |
    .metadata.name = env(NAME) |
    .spec.predictor.minReplicas = (env(REPLICAS) | tonumber) |
    .spec.predictor.maxReplicas = (env(REPLICAS) | tonumber) |
    .spec.predictor.model.runtime = env(NAME) |
    .spec.predictor.model.storageUri = env(MODEL_URI) |
    .spec.predictor.model.args = env(VLLM_ARGS) |
    .spec.predictor.model.args style="" |
    .spec.predictor.model.resources.limits."nvidia.com/gpu" = env(TP_SIZE) |
    .spec.predictor.model.resources.requests."nvidia.com/gpu" = env(TP_SIZE)
  ' "${BASE}/yaml/infra/isvc-pvc.yaml.tmpl" |
    oc apply -n "${NAMESPACE}" -f -

elif [ "$MODE" = "oci" ]; then

  echo "Using OCI for model storage: ${MODEL_URI}"

  oc delete secret/"${NAME}" \
    -n "${NAMESPACE}" \
    --ignore-not-found

  yq eval '
    .metadata.annotations."openshift.io/display-name" = env(NAME) |
    .metadata.name = env(NAME) |
    .data.URI = (env(MODEL_URI) | @base64)
  ' "${BASE}/yaml/infra/data-connection-oci.yaml.tmpl" |
    oc apply -n "${NAMESPACE}" -f -

  yq eval '
    .metadata.annotations."openshift.io/display-name" = env(NAME) |
    .metadata.name = env(NAME) |
    .spec.predictor.minReplicas = (env(REPLICAS) | tonumber) |
    .spec.predictor.maxReplicas = (env(REPLICAS) | tonumber) |
    .spec.predictor.model.runtime = env(NAME) |
    .spec.predictor.model.storageUri = env(MODEL_URI) |
    .spec.predictor.model.args = env(VLLM_ARGS) |
    .spec.predictor.model.args style="" |
    .spec.predictor.model.resources.limits."nvidia.com/gpu" = env(TP_SIZE) |
    .spec.predictor.model.resources.requests."nvidia.com/gpu" = env(TP_SIZE)
  ' "${BASE}/yaml/infra/isvc-oci.yaml.tmpl" |
    oc apply -n "${NAMESPACE}" -f -

fi
