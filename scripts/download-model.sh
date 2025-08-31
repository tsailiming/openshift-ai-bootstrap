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

  if ! oc get pvc models-pvc -n demo >/dev/null 2>&1; then
    cat <<EOF | oc create -f -
  apiVersion: v1
  kind: PersistentVolumeClaim
  metadata:
    name: models-pvc
    namespace: demo
  spec:
    accessModes:
      - ReadWriteOnce
    resources:
      requests:
        storage: 100Gi
EOF

else
  echo "PVC models-pvc already exists in namespace demo. Skipping creation."
fi

else
  echo "Invalid mode: $MODE (must be 's3' or 'pvc')"
  exit 1
fi

# Cleanup old job (PVC is deleted only in s3 mode)
oc delete job/${JOB_NAME} -n ${NAMESPACE} --ignore-not-found

if [ "$MODE" = "s3" ]; then
  oc delete pvc/models-pvc -n ${NAMESPACE} --ignore-not-found
fi

# Create job from template
envsubst '$MODEL_LIST' < ${BASE}/yaml/infra/${TEMPLATE} | oc create -n ${NAMESPACE} -f -

echo "Waiting for job ${JOB_NAME} to complete..."

until oc get job ${JOB_NAME} -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' | grep -q "True"; do
    if oc get job ${JOB_NAME} -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' | grep -q "True"; then
        echo "Job ${JOB_NAME} failed."
        exit 1
    fi
    echo "Job ${JOB_NAME} is still running..."
    sleep 10
done

# Cleanup job (PVC is deleted only in s3 mode)
oc delete job/${JOB_NAME} -n ${NAMESPACE}
if [ "$MODE" = "s3" ]; then
  oc delete pvc/models-pvc -n ${NAMESPACE}
fi

echo "Job ${JOB_NAME} completed successfully."
