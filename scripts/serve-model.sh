#!/bin/sh

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NAMESPACE=demo
export NAME=$1 #qwen2.5-vl-7b-instruct
export MODEL_PATH=$2 #Qwen/Qwen2.5-VL-7B-Instruct/

if [ -z "$NAME" ] || [ -z "$MODEL_PATH" ]; then
  echo "Usage: $0 <name> <model_path>"
  echo "Example: $0 qwen2.5-vl-7b-instruct Qwen/Qwen2.5-VL-7B-Instruct/"
  exit 1
fi


oc delete isvc/$NAME -n ${NAMESPACE}
oc delete servingruntime/$NAME -n ${NAMESPACE}

envsubst '$NAME, $MODEL_PATH' \
    < ${BASE}/yaml/demo/sr.yaml.tmpl | oc apply -n ${NAMESPACE} -f -

envsubst '$NAME, $MODEL_PATH' \
    < ${BASE}/yaml/demo/isvc.yaml.tmpl | oc apply -n ${NAMESPACE} -f -