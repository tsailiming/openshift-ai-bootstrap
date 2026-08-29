#!/bin/sh

set -eu

NAMESPACE="openshift-sandboxed-containers-operator"
KATA_MCP="kata-oc"
KATA_RUNTIMECLASS="${KATA_RUNTIMECLASS:-kata}"
PATCH_MANIFEST="https://raw.githubusercontent.com/opendatahub-io/agent-ops/main/manifests/kata-nftables-patch-job.yaml"

die()
{
    echo "ERROR: $*" >&2
    exit 1
}

echo "=========================================="
echo "Checking OpenShift/Kata health"
echo "=========================================="

# ----------------------------------------------------------------------
# Check kata-oc MachineConfigPool
# ----------------------------------------------------------------------

echo "Checking MachineConfigPool/$KATA_MCP..."

oc get mcp "$KATA_MCP" >/dev/null 2>&1 ||
    die "MachineConfigPool/$KATA_MCP does not exist."

UPDATED=$(oc get mcp "$KATA_MCP" \
    -o jsonpath='{.status.conditions[?(@.type=="Updated")].status}')

UPDATING=$(oc get mcp "$KATA_MCP" \
    -o jsonpath='{.status.conditions[?(@.type=="Updating")].status}')

DEGRADED=$(oc get mcp "$KATA_MCP" \
    -o jsonpath='{.status.conditions[?(@.type=="Degraded")].status}')

MACHINE_COUNT=$(oc get mcp "$KATA_MCP" \
    -o jsonpath='{.status.machineCount}')

READY_COUNT=$(oc get mcp "$KATA_MCP" \
    -o jsonpath='{.status.readyMachineCount}')

UPDATED_COUNT=$(oc get mcp "$KATA_MCP" \
    -o jsonpath='{.status.updatedMachineCount}')

echo "  Updated:        ${UPDATED}"
echo "  Updating:       ${UPDATING}"
echo "  Degraded:       ${DEGRADED}"
echo "  Machines:       ${MACHINE_COUNT}"
echo "  Ready:          ${READY_COUNT}"
echo "  Updated nodes:  ${UPDATED_COUNT}"

if [ "$UPDATED" != "True" ] ||
   [ "$UPDATING" != "False" ] ||
   [ "$DEGRADED" != "False" ] ||
   [ "$MACHINE_COUNT" != "$READY_COUNT" ] ||
   [ "$MACHINE_COUNT" != "$UPDATED_COUNT" ]
then
    oc get mcp "$KATA_MCP"
    die "MachineConfigPool/$KATA_MCP is not healthy."
fi

echo "MachineConfigPool/$KATA_MCP: healthy"

# ----------------------------------------------------------------------
# Check Kata runtime class
# ----------------------------------------------------------------------

echo "Checking Kata RuntimeClass/$KATA_RUNTIMECLASS..."

oc get runtimeclass "$KATA_RUNTIMECLASS" >/dev/null 2>&1 ||
    die "RuntimeClass/$KATA_RUNTIMECLASS does not exist."

echo "RuntimeClass/$KATA_RUNTIMECLASS: available"

# ----------------------------------------------------------------------
# kata-remote does not require the initramfs patch.
# ----------------------------------------------------------------------

if [ "$KATA_RUNTIMECLASS" = "kata-remote" ]; then
    echo ""
    echo "KATA_RUNTIMECLASS=kata-remote"
    echo "Skipping Kata initramfs nftables patch."
    echo "Peer-pod VMs use an RHEL image that already includes nf_tables."
    echo ""
    exit 0
fi

# ----------------------------------------------------------------------
# Verify the operator namespace is accessible
# ----------------------------------------------------------------------

echo "Checking namespace/$NAMESPACE..."

oc get namespace "$NAMESPACE" >/dev/null 2>&1 ||
    die "Namespace/$NAMESPACE does not exist."

echo "Namespace/$NAMESPACE: available"

echo ""
echo "=========================================="
echo "Kata health check PASSED"
echo "=========================================="
echo ""

# ----------------------------------------------------------------------
# Grant privileged SCC
# ----------------------------------------------------------------------

echo "Granting privileged SCC to kata-install..."

oc adm policy add-scc-to-user privileged \
    -z kata-install \
    -n "$NAMESPACE"

# Make sure SCC is revoked even if the script exits unexpectedly.
cleanup()
{
    echo ""
    echo "Cleaning up Kata nftables patch..."

    if oc delete -f "$PATCH_MANIFEST" >/dev/null 2>&1; then
        echo "Patch DaemonSet deleted."
    else
        echo "WARNING: Patch DaemonSet may already be deleted."
    fi

    if oc adm policy remove-scc-from-user privileged \
        -z kata-install \
        -n "$NAMESPACE" >/dev/null 2>&1
    then
        echo "Privileged SCC revoked from kata-install."
    else
        echo "WARNING: Failed to revoke privileged SCC from kata-install." >&2
    fi
}

trap cleanup EXIT INT TERM

# ----------------------------------------------------------------------
# Apply nftables patch
# ----------------------------------------------------------------------

echo "Applying Kata nftables patch..."

oc apply -f "$PATCH_MANIFEST"

# ----------------------------------------------------------------------
# Wait for all patch pods to exist and become Running/Ready
# ----------------------------------------------------------------------

echo "Waiting for all Kata nftables patch pods to be Running and Ready..."

while true; do
    PODS=$(oc get pods \
        -n "$NAMESPACE" \
        -l app=kata-nftables-patch \
        --no-headers 2>/dev/null || true)

    if [ -z "$PODS" ]; then
        echo "Waiting for patch pods to be created..."
        sleep 5
        continue
    fi

    TOTAL=$(printf '%s\n' "$PODS" | wc -l | tr -d ' ')

    READY=$(printf '%s\n' "$PODS" |
        awk '$2 == "1/1" && $3 == "Running" {
            count++
        }
        END {
            print count+0
        }')

    echo "Patch pods ready: $READY/$TOTAL"

    if [ "$READY" -eq "$TOTAL" ]; then
        break
    fi

    sleep 5
done

echo "All patch pods are Running/Ready."

# ----------------------------------------------------------------------
# Verify every patch pod completed successfully
# ----------------------------------------------------------------------

echo "Waiting for all Kata initramfs patches to complete..."

while true; do
    PODS=$(oc get pods \
        -n "$NAMESPACE" \
        -l app=kata-nftables-patch \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
        2>/dev/null || true)

    TOTAL=$(printf '%s\n' "$PODS" | sed '/^$/d' | wc -l | tr -d ' ')

    if [ "$TOTAL" -eq 0 ]; then
        echo "Waiting for patch pods..."
        sleep 5
        continue
    fi

    COMPLETE=0

    for POD in $PODS; do
        LOGS=$(oc logs \
            -n "$NAMESPACE" \
            "$POD" 2>/dev/null || true)

        printf '%s\n' "$LOGS"

        if printf '%s\n' "$LOGS" |
            grep -q '\[+] Patch complete\. Sleeping\.'
        then
            COMPLETE=$((COMPLETE + 1))
        fi
    done

    echo "Patch complete: $COMPLETE/$TOTAL"

    if [ "$COMPLETE" -eq "$TOTAL" ]; then
        break
    fi

    sleep 5
done

echo "All Kata initramfs patches completed successfully."

# ----------------------------------------------------------------------
# Cleanup is performed by EXIT trap:
#
#   oc delete -f <manifest>
#   oc adm policy remove-scc-from-user privileged ...
# ----------------------------------------------------------------------

echo ""
echo "Kata initramfs patch completed successfully."
echo "The patch DaemonSet and privileged SCC will now be removed."
