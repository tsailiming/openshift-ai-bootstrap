#!/bin/sh

set -eu

NODE="${1:-}"

if [ -z "$NODE" ]; then
    echo "Usage: $0 <node-name>"
    exit 1
fi

echo "Checking node: $NODE"
oc get node "$NODE" >/dev/null

echo
echo "Pods stuck in Terminating on $NODE:"
echo

PODS=$(oc get pods -A \
    --field-selector "spec.nodeName=$NODE" \
    -o jsonpath='{range .items[?(@.metadata.deletionTimestamp)]}{.metadata.namespace}{" "}{.metadata.name}{"\n"}{end}')

if [ -z "$PODS" ]; then
    echo "No terminating pods found on $NODE."
    exit 0
fi

echo "$PODS"
echo

printf "Force-delete ALL of these pods? [y/N] "
read CONFIRM

case "$CONFIRM" in
    y|Y|yes|YES)
        ;;
    *)
        echo "Aborted."
        exit 0
        ;;
esac

echo
echo "Force-deleting pods..."

echo "$PODS" | while read NAMESPACE POD; do
    if [ -n "$POD" ]; then
        echo "  Deleting $NAMESPACE/$POD"

        oc delete pod "$POD" \
            -n "$NAMESPACE" \
            --grace-period=0 \
            --force \
            --ignore-not-found
    fi
done

echo
echo "Done."
echo
echo "Remaining pods on $NODE:"
oc get pods -A \
    --field-selector "spec.nodeName=$NODE" \
    -o wide || true
