#!/bin/bash
set -euo pipefail

echo "Creating temp dir..."
TMPDIR=$(mktemp -d)
echo "Using temp dir: $TMPDIR"

# Restore default storage class if it was changed
DEFAULT_SC=""

for sc in $(oc get storageclass -o name); do
    isdefault=$(oc get "$sc" -o jsonpath='{.metadata.annotations.storageclass\.kubernetes\.io/is-default-class}' 2>/dev/null)
    if [ "$isdefault" = "true" ]; then
        DEFAULT_SC=$(echo "$sc" | cut -d/ -f2)
        break
    fi
done

# Clone repo and run install
git clone https://github.com/tsailiming/openshift-nfs-server "$TMPDIR/openshift-nfs-server"
cd "$TMPDIR/openshift-nfs-server"
./scripts/install.sh

oc patch pvc nfs-server-nfs-server-0 -n nfs \
  -p '{"spec":{"resources":{"requests":{"storage":"100Gi"}}}}'

# Set all storageclasses to false
for sc in $(oc get storageclass -o name); do
    sc_name=$(echo "$sc" | cut -d/ -f2)
    echo "Setting $sc_name default=false"
    oc patch storageclass "$sc_name" \
        -p '{"metadata":{"annotations":{"storageclass.kubernetes.io/is-default-class":"false"}}}' || true
done

oc patch storageclass "$DEFAULT_SC" \
  -p '{"metadata":{"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'

echo "Current storage classes:"
oc get storageclass
