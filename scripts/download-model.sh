#!/bin/sh

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NAMESPACE=demo
JOB_NAME=download-models-s3 
export MODEL_LIST="$1"   # first argument to script

if [ -z "$MODEL_LIST" ]; then
  echo "Usage: $0 \"<model1> <model2> ...\""
  exit 1
fi

oc delete job/${JOB_NAME} -n ${NAMESPACE}
oc delete pvc/models-pvc -n ${NAMESPACE}

envsubst '$MODEL_LIST' < ${BASE}/yaml/infra/download-models-s3.tmpl.yaml | oc create -n ${NAMESPACE} -f -

echo "Waiting for job ${JOB_NAME} to complete..."

until oc get job ${JOB_NAME} -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' | grep -q "True"; do \
    if oc get job ${JOB_NAME} -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' | grep -q "True"; then \
        echo "Job ${JOB_NAME} failed."; \
        exit 1; \
    fi; \
    echo "Job ${JOB_NAME} is still running..."; \
    sleep 10; \
done	

oc delete job/${JOB_NAME} -n ${NAMESPACE}
oc delete pvc/models-pvc -n ${NAMESPACE}