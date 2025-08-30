#!/bin/sh

YAML_FILE=$1
NAMESPACE=$2
JOB_NAME=$3
SECRET_NAME=$4

echo "NAMESPACE: ${NAMESPACE}"
echo "JOB_NAME: ${JOB_NAME}"
echo "SECERT_NAME: ${SECRET_NAME}"
echo

if oc get job "${JOB_NAME}" -n "${NAMESPACE}" &> /dev/null; then
    echo "Job $JOB_NAME exists in namespace $NAMESPACE."
    oc delete -f ${YAML_FILE}    
fi

AWS_S3_BUCKET=\${AWS_S3_BUCKET} \
AWS_SECRET_NAME=${SECRET_NAME} \
envsubst < yaml/infra/setup-s3.yaml.tmpl | oc create -n ${NAMESPACE} -f -

echo "Waiting for job ${JOB_NAME} to complete..."

until oc get job ${JOB_NAME} -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' | grep -q "True"; do \
    if oc get job ${JOB_NAME} -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' | grep -q "True"; then \
        echo "Job ${JOB_NAME} failed."; \
        exit 1; \
    fi; \
    echo "Job ${JOB_NAME} is still running..."; \
    sleep 10; \
done	

oc delete -f ${YAML_FILE} -n ${NAMESPACE}