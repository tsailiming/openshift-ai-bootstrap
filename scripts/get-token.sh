#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="demo"
SA_NAME="cluster-admin-sa"

oc get ns "${NAMESPACE}" >/dev/null 2>&1 || \
    oc create ns "${NAMESPACE}"

oc get sa "${SA_NAME}" -n "${NAMESPACE}" >/dev/null 2>&1 || \
    oc create sa "${SA_NAME}" -n "${NAMESPACE}"

oc adm policy add-cluster-role-to-user \
    cluster-admin \
    -z "${SA_NAME}" \
    -n "${NAMESPACE}" >/dev/null

oc create token "${SA_NAME}" -n "${NAMESPACE}"