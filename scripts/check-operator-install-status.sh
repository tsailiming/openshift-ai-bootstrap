#!/usr/bin/env bash
# GPT created
# This script does not automatically approve upgrades so the operator will maintain
# the starting CSV version as defined in the subscription
set -euo pipefail

SUB_NAME="$1"
NAMESPACE="$2"
TIMEOUT="${3:-300}"
SLEEP=5

start_time=$(date +%s)

echo "📦 Checking Operator Subscription: $SUB_NAME in namespace: $NAMESPACE is fully installed"

# ------------------------------------------------------------
# STEP 1 — Approve InstallPlan from subscription
# ------------------------------------------------------------
echo "⏳ Waiting for Subscription to report InstallPlan..."

while true; do
    IP_NAME=$(oc get subscription "$SUB_NAME" -n "$NAMESPACE" \
          -o jsonpath='{.status.installPlanRef.name}' 2>/dev/null || true)

    if [[ -n "$IP_NAME" ]]; then
            echo "📄 InstallPlan: $IP_NAME"
        break
    else
        echo "   • InstallPlan not yet populated (waiting...)"
    fi

    if (( $(date +%s) - start_time >= TIMEOUT )); then
        echo "❌ Timeout waiting for InstallPlan to appear"
        exit 1
    fi

    sleep "$SLEEP"
done

CSV=$(oc get subscription "$SUB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.installedCSV}' 2>/dev/null || true)

CSV_PHASE=""
if [[ -n "$CSV" ]]; then
    CSV_PHASE=$(oc get csv "$CSV" -n "$NAMESPACE" \
        -o jsonpath='{.status.phase}' 2>/dev/null || true)
fi

echo "   • installedCSV: ${CSV:-<missing>}"
echo "   • CSV phase: ${CSV_PHASE:-<missing>}"

# Approve if CSV is missing or has not succeeded
if [[ -z "$CSV" || "$CSV_PHASE" != "Succeeded" ]]; then
    APPROVAL_MODE=$(oc get installplan "$IP_NAME" -n "$NAMESPACE" \
        -o jsonpath='{.spec.approval}' 2>/dev/null || true)
    APPROVED=$(oc get installplan "$IP_NAME" -n "$NAMESPACE" \
        -o jsonpath='{.spec.approved}' 2>/dev/null || true)

    echo "   • approval mode: $APPROVAL_MODE"
    echo "   • approved: $APPROVED"

    if [[ "$APPROVAL_MODE" == "Manual" && "$APPROVED" != "true" ]]; then
        echo "📝 InstallPlan requires manual approval → approving..."
        oc patch installplan "$IP_NAME" -n "$NAMESPACE" \
            --type merge -p '{"spec":{"approved":true}}'
    else
        echo "✔️ InstallPlan already approved or auto-approved."        
    fi
else
    echo "✔️ CSV exists and is Succeeded → nothing to do anymore."
    exit 0
fi

echo "⏳ Waiting for CSV to reach phase: Succeeded..."

while true; do

    elapsed=$(( $(date +%s) - start_time ))

    CSV=$(oc get subscription "$SUB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.installedCSV}' 2>/dev/null || true)

    if [[ -n "$CSV" ]]; then
        echo "📦 Target CSV: $CSV"        
    else
        echo "   • installedCSV not yet populated (waiting...)"        

        if (( elapsed >= TIMEOUT )); then
            echo "❌ Timeout waiting for installedCSV to appear"
            exit 1
        else
            sleep "$SLEEP"
            continue
        fi
    fi

    PHASE=$(oc get csv "$CSV" -n "$NAMESPACE" \
            -o jsonpath='{.status.phase}' 2>/dev/null || echo "")

    if [[ "$PHASE" == "Succeeded" ]]; then
        echo "✅ CSV '$CSV' is Succeeded"s
        exit 0

    elif [[ "$PHASE" == "Failed" ]]; then
        echo "❌ CSV '$CSV' entered Failed phase"       
        exit 1    
    fi

    if [[ -n "$PHASE" ]]; then
        echo "   • phase: $PHASE (waiting...)"
    else
        echo "   • CSV not visible yet (waiting...)" # This should never happen
    fi

    if (( elapsed >= TIMEOUT )); then
        echo "❌ Timeout waiting for CSV to reach Succeeded"
        exit 1
    fi

    sleep "$SLEEP"
done