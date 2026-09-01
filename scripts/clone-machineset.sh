#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="openshift-machine-api"

usage() {
  cat <<EOF
Usage:
  $0 [--list] | <new_instance_type> [--dry-run] [--on-demand]
  $0 <old_machineset> <new_instance_type> [--dry-run] [--on-demand]

Options:
  --list             Show all current MachineSets in openshift-machine-api and exit
  --dry-run          Print generated YAML without applying it
  --on-demand        Use on-demand instances instead of spot
  --help, -h         Show this help message

MachineSet selection:
  - If an old MachineSet is provided, it will be used directly.
  - If only an instance type is provided, the script automatically selects
    the first MachineSet whose Availability Zone ends in "a".

Replica behavior:
  - New MachineSet: replicas=0
  - Existing MachineSet: existing replica count is preserved

Naming:
  - GPU:     <old>-x<gpu_count>-<gpu_type>
  - Non-GPU: <old>-<instance_type>

Examples:
  $0 --list

  # Automatically find MachineSet in the first AZ ("a")
  $0 g6.4xlarge

  # Automatically find MachineSet and use on-demand
  $0 p4d.24xlarge --on-demand

  # Automatically find MachineSet and dry-run
  $0 g6.4xlarge --dry-run

  # Explicitly specify source MachineSet
  $0 ocp-c6bsh-bxjl2-worker-ap-northeast-1a g6.4xlarge --dry-run
EOF
}

# ------------------------------------------------------------
# Check dependencies
# ------------------------------------------------------------

for cmd in oc aws jq yq awk head; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "Error: Required command '$cmd' is not installed or not in PATH."
    exit 1
  fi
done

# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------

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
      echo "Listing all MachineSets in namespace '$NAMESPACE':"
      oc get machinesets -n "$NAMESPACE"
      exit 0
      ;;

    *)
      ARGS+=("$arg")
      ;;
  esac
done

# ------------------------------------------------------------
# Determine source MachineSet and instance type
#
# Supported:
#
#   script.sh <instance_type>
#   script.sh <machineset> <instance_type>
#
# If only one argument is provided, automatically select the
# first MachineSet whose AZ ends in "a".
# ------------------------------------------------------------

if [[ ${#ARGS[@]} -eq 1 ]]; then

  # Only instance type supplied.
  NEW_TYPE="${ARGS[0]}"

  echo "No source MachineSet specified."
  echo "Finding first MachineSet in an Availability Zone ending in 'a'..."

  MACHINESET_INFO=$(oc get machinesets \
    -n "$NAMESPACE" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.template.spec.providerSpec.value.placement.availabilityZone}{"\n"}{end}' \
    | awk '$2 ~ /a$/ {print $1 "\t" $2; exit}')

  if [[ -z "$MACHINESET_INFO" ]]; then
    echo "ERROR: No MachineSet found in an Availability Zone ending in 'a'."
    exit 1
  fi

  OLD_MS=$(echo "$MACHINESET_INFO" | awk '{print $1}')
  AWS_AZ=$(echo "$MACHINESET_INFO" | awk '{print $2}')

  echo "Found MachineSet: $OLD_MS"
  echo "Availability Zone: $AWS_AZ"

elif [[ ${#ARGS[@]} -ge 2 ]]; then

  # Explicit MachineSet and instance type supplied.
  OLD_MS="${ARGS[0]}"
  NEW_TYPE="${ARGS[1]}"

  echo "Using explicitly specified MachineSet: $OLD_MS"

else

  echo "Error: Missing required instance type."
  echo
  usage
  exit 1

fi

# ------------------------------------------------------------
# Fetch old MachineSet YAML
# ------------------------------------------------------------

echo
echo "Fetching source MachineSet: $OLD_MS"

if ! OLD_MS_YAML=$(oc get machineset "$OLD_MS" \
    -n "$NAMESPACE" \
    -o yaml 2>/dev/null); then

  echo "Error: Failed to fetch MachineSet '$OLD_MS'."
  exit 1
fi

# If MachineSet was explicitly provided, get its AZ for display.
if [[ -z "${AWS_AZ:-}" ]]; then
  AWS_AZ=$(echo "$OLD_MS_YAML" |
    yq eval '.spec.template.spec.providerSpec.value.placement.availabilityZone // ""' -)
fi

echo "Source MachineSet AZ: ${AWS_AZ:-unknown}"

# ------------------------------------------------------------
# Fetch GPU information from AWS
# ------------------------------------------------------------

echo
echo "Checking GPU info for instance type: $NEW_TYPE"

if ! GPU_INFO=$(aws ec2 describe-instance-types \
    --instance-types "$NEW_TYPE" \
    --query "InstanceTypes[0].GpuInfo" \
    --output json); then

  echo "Error: Failed to get AWS instance information for '$NEW_TYPE'."
  exit 1
fi

GPU_TYPE=$(echo "$GPU_INFO" | jq -r '.Gpus[0].Name // "none"')
GPU_COUNT=$(echo "$GPU_INFO" | jq -r '.Gpus[0].Count // 0')

# ------------------------------------------------------------
# Generate MachineSet name
# ------------------------------------------------------------

GPU_TYPE_SHORT=$(
  echo "$GPU_TYPE" |
    tr '[:upper:]' '[:lower:]' |
    tr ' ' '-'
)

if [[ "$GPU_COUNT" -gt 0 ]]; then

  GPU_SUFFIX="x${GPU_COUNT}-${GPU_TYPE_SHORT}"

else

  # Replace "." with "-" because MachineSet names must be
  # valid Kubernetes DNS names.
  GPU_SUFFIX="${NEW_TYPE//./-}"

fi

echo "GPU INFO: x${GPU_COUNT} ${GPU_TYPE}"

NEW_MS="${OLD_MS}-${GPU_SUFFIX}"

echo "Generated target MachineSet name: $NEW_MS"

# ------------------------------------------------------------
# Check whether target MachineSet already exists
# ------------------------------------------------------------

NEW_MS_EXISTS="false"
EXISTING_REPLICAS="0"

if EXISTING_REPLICAS=$(oc get machineset "$NEW_MS" \
    -n "$NAMESPACE" \
    -o jsonpath='{.spec.replicas}' 2>/dev/null); then

  NEW_MS_EXISTS="true"

  EXISTING_REPLICAS="${EXISTING_REPLICAS:-0}"

  echo
  echo "Target MachineSet already exists:"
  echo "  Name:     $NEW_MS"
  echo "  Replicas: $EXISTING_REPLICAS"
  echo
  echo "Existing replica count will be PRESERVED."

else

  echo
  echo "Target MachineSet does not exist:"
  echo "  Name:     $NEW_MS"
  echo "  Replicas: 0"
  echo
  echo "The new MachineSet will be created with 0 replicas."

fi

# ------------------------------------------------------------
# Determine replica expression
# ------------------------------------------------------------

if [[ "$NEW_MS_EXISTS" == "true" ]]; then
  REPLICA_EXPR=".spec.replicas = ${EXISTING_REPLICAS}"
else
  REPLICA_EXPR=".spec.replicas = 0"
fi

# ------------------------------------------------------------
# Build yq expression
# ------------------------------------------------------------

YQ_EXPR="
    .metadata.name = \"${NEW_MS}\" |
    del(
      .metadata.resourceVersion,
      .metadata.uid,
      .metadata.creationTimestamp,
      .metadata.generation,
      .metadata.managedFields
    ) |
    .metadata.annotations = {} |
    del(.status) |
    ${REPLICA_EXPR} |
    .spec.selector.matchLabels.\"machine.openshift.io/cluster-api-machineset\" = \"${NEW_MS}\" |
    .spec.template.metadata.labels.\"machine.openshift.io/cluster-api-machineset\" = \"${NEW_MS}\" |
    .spec.template.spec.providerSpec.value.instanceType = \"${NEW_TYPE}\"
"

# ------------------------------------------------------------
# Add NVIDIA GPU taint only for GPU instances
# ------------------------------------------------------------

if [[ "$GPU_COUNT" -gt 0 ]]; then

  echo "Adding NVIDIA GPU taint"

  YQ_EXPR="${YQ_EXPR} |
    .spec.template.spec.taints = [
      {
        \"key\": \"nvidia.com/gpu\",
        \"value\": \"true\",
        \"effect\": \"NoSchedule\"
      }
    ]"

else

  echo "No GPU detected"

  # Remove NVIDIA GPU taints when cloning to a non-GPU instance.
  YQ_EXPR="${YQ_EXPR} |
    del(.spec.template.spec.taints)"

fi

# ------------------------------------------------------------
# Configure spot / on-demand
# ------------------------------------------------------------

if [[ "$USE_ON_DEMAND" == "false" ]]; then

  echo "Using spot instance"

  YQ_EXPR="${YQ_EXPR} |
    .spec.template.spec.providerSpec.value.spotMarketOptions = {}"

else

  echo "Using on-demand instance"

  # Make sure a cloned spot MachineSet does not remain configured
  # with spotMarketOptions.
  YQ_EXPR="${YQ_EXPR} |
    del(.spec.template.spec.providerSpec.value.spotMarketOptions)"

fi

# ------------------------------------------------------------
# Generate final YAML
# ------------------------------------------------------------

NEW_MS_YAML=$(echo "$OLD_MS_YAML" | yq eval "$YQ_EXPR" -)

# ------------------------------------------------------------
# Dry run
# ------------------------------------------------------------

if [[ "$DRY_RUN" == "true" ]]; then

  echo
  echo "===================================================="
  echo "Dry-run mode"
  echo "===================================================="
  echo
  echo "Source MachineSet : $OLD_MS"
  echo "Source AZ         : ${AWS_AZ:-unknown}"
  echo "Target MachineSet : $NEW_MS"
  echo "Instance type     : $NEW_TYPE"
  echo "GPU               : x${GPU_COUNT} ${GPU_TYPE}"
  echo "On-demand         : $USE_ON_DEMAND"

  if [[ "$NEW_MS_EXISTS" == "true" ]]; then
    echo "Target exists     : yes"
    echo "Replicas          : ${EXISTING_REPLICAS} (preserved)"
  else
    echo "Target exists     : no"
    echo "Replicas          : 0 (new MachineSet)"
  fi

  echo
  echo "===================================================="
  echo "Generated MachineSet YAML"
  echo "===================================================="
  echo
  echo "$NEW_MS_YAML"

  exit 0

fi

# ------------------------------------------------------------
# Apply
# ------------------------------------------------------------

echo
echo "===================================================="

if [[ "$NEW_MS_EXISTS" == "true" ]]; then

  echo "Updating existing MachineSet"
  echo "  Name     : $NEW_MS"
  echo "  Replicas : $EXISTING_REPLICAS (PRESERVED)"

else

  echo "Creating new MachineSet"
  echo "  Name     : $NEW_MS"
  echo "  Replicas : 0"

fi

echo "===================================================="
echo

echo "$NEW_MS_YAML" | oc apply -f -

echo
echo "MachineSet '$NEW_MS' successfully applied."

if [[ "$NEW_MS_EXISTS" == "true" ]]; then
  echo "Existing replica count was preserved at ${EXISTING_REPLICAS}."
else
  echo "New MachineSet was created with 0 replicas."
fi