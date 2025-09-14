#!/bin/sh

set -e

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NAMESPACE=demo
JOB_NAME=download-models
MODE="$1"          # first arg: s3 or pvc
shift
export MODEL_LIST="$@"   # remaining args: list of models

if [ -z "$MODE" ] || [ -z "$MODEL_LIST" ]; then
  echo "Usage: $0 <s3|pvc> \"<model1> <model2> ...\""
  exit 1
fi

if [ "$MODE" = "s3" ]; then
  TEMPLATE="download-models-s3.yaml.tmpl"
  JOB_NAME="${JOB_NAME}-s3"
elif [ "$MODE" = "pvc" ]; then
  TEMPLATE="download-models-pvc.yaml.tmpl"
  JOB_NAME="${JOB_NAME}-pvc"
else
  echo "Invalid mode: $MODE (must be 's3' or 'pvc')"
  exit 1
fi

# Cleanup old job
oc delete job/${JOB_NAME} -n ${NAMESPACE} --ignore-not-found

# Create job from template
envsubst '$MODEL_LIST $HF_TOKEN' < ${BASE}/yaml/infra/${TEMPLATE} | oc create -n ${NAMESPACE} -f -

echo "Waiting for job ${JOB_NAME} to complete..."

until oc get job ${JOB_NAME} -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' | grep -q "True"; do
    if oc get job ${JOB_NAME} -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' | grep -q "True"; then
        echo "Job ${JOB_NAME} failed."
        exit 1
    fi
    echo "Job ${JOB_NAME} is still running..."
    sleep 5
done

oc delete job/${JOB_NAME} -n ${NAMESPACE}
echo "Job ${JOB_NAME} completed successfully."