#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [--list] | <old_machineset> <new_instance_type> [--dry-run] [--on-demand]

Options:
  --list             Show all current MachineSets in openshift-machine-api and exit
  --dry-run          Optional flag to print YAML without applying it
  --on-demand        Use on-demand instances instead of spot
  --help, -h         Show this help message

Clone an OpenShift MachineSet to a new instance type. By default, the new MachineSet
will use spot instances unless --on-demand is specified. The generated MachineSet will
always have replicas=0 and a name suffix based on GPU count and type.

Examples:
  $0 --list
  $0 ocp-c6bsh-bxjl2-worker-ap-northeast-1a g6.4xlarge --dry-run
  $0 ocp-c6bsh-bxjl2-worker-ap-northeast-1a p4d.24xlarge --on-demand
EOF
}

# Check dependencies
for cmd in oc aws jq yq; do
  if ! command -v $cmd &>/dev/null; then
    echo "Error: Required command '$cmd' is not installed or not in PATH."
    exit 1
  fi
done

# --- Argument parsing ---
DRY_RUN="false"
USE_ON_DEMAND="false"

ARGS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN="true"
      ;;
    --on-demand)
      USE_ON_DEMAND="true"
      ;;
    --help|-h)
  usage
  exit 0
      ;;
    --list)
  echo "Listing all MachineSets in namespace 'openshift-machine-api':"
  oc get machinesets -n openshift-machine-api
  exit 0
      ;;
    *)
      ARGS+=("$arg")
      ;;
  esac
done

# After parsing flags, assign positional arguments
if [[ ${#ARGS[@]} -lt 2 ]]; then
  echo "Error: Missing required arguments."
  usage
  exit 1
fi

OLD_MS="${ARGS[0]}"
NEW_TYPE="${ARGS[1]}"

# --- Fetch old MachineSet YAML ---
echo "Fetching MachineSet: $OLD_MS"
if ! OLD_MS_YAML=$(oc get machineset "$OLD_MS" -n openshift-machine-api -o yaml 2>/dev/null); then
  echo "Error: Failed to fetch MachineSet '$OLD_MS'."
  exit 1
fi

# Fetch GPU information from AWS
echo "Checking GPU info for instance type: $NEW_TYPE"
GPU_INFO=$(aws ec2 describe-instance-types \
  --instance-types "$NEW_TYPE" \
  --query "InstanceTypes[0].GpuInfo" \
  --output json)

GPU_TYPE=$(echo "$GPU_INFO" | jq -r '.Gpus[0].Name // "none"')
GPU_COUNT=$(echo "$GPU_INFO" | jq -r '.Gpus[0].Count // 0')

# Normalize GPU type (lowercase, spaces replaced with '-')
GPU_TYPE_SHORT=$(echo "$GPU_TYPE" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
if [[ "$GPU_COUNT" -gt 0 ]]; then
  GPU_SUFFIX="x${GPU_COUNT}-${GPU_TYPE_SHORT}"
else
  GPU_SUFFIX="x0-none"
fi

echo "GPU INFO: x$GPU_COUNT $GPU_TYPE"

# Append suffix to original MachineSet name
NEW_MS="${OLD_MS}-${GPU_SUFFIX}"
echo "Generated new MachineSet name: $NEW_MS"

# --- Build yq expression dynamically ---
YQ_EXPR="
    .metadata.name = \"${NEW_MS}\" |
    del(.metadata.resourceVersion, .metadata.uid, .metadata.creationTimestamp, .metadata.generation, .metadata.managedFields) |
    .metadata.annotations = {} |
    del(.status) |
    .spec.replicas = 0 |
    .spec.selector.matchLabels.\"machine.openshift.io/cluster-api-machineset\" = \"${NEW_MS}\" |
    .spec.template.metadata.labels.\"machine.openshift.io/cluster-api-machineset\" = \"${NEW_MS}\" |
    .spec.template.spec.providerSpec.value.instanceType = \"${NEW_TYPE}\"
"

# If NOT on-demand, add spotMarketOptions
if [[ "$USE_ON_DEMAND" == "false" ]]; then
  echo "Using spot instance"
  YQ_EXPR="${YQ_EXPR} | .spec.template.spec.providerSpec.value.spotMarketOptions = {}"
else
  echo "Using on-demand instance"
fi

# --- Generate final YAML ---
NEW_MS_YAML=$(echo "$OLD_MS_YAML" | yq eval "$YQ_EXPR" -)

# --- Dry-run or apply ---
if [[ "$DRY_RUN" == "true" ]]; then
  echo
  echo "Dry-run mode: showing generated MachineSet YAML only."
  echo "===================================================="
  echo "$NEW_MS_YAML"
else
  echo
  echo "Applying new MachineSet: $NEW_MS with 0 replica"
  echo "$NEW_MS_YAML" | oc apply -f -
fi
