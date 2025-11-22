BASE:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SHELL=/bin/sh
NAMESPACE=demo

.PHONY: setup-rhoai
setup-rhoai: add-gpu-operator add-nfs-provisioner
	oc apply -f $(BASE)/yaml/rhoai/kueue.yaml
	
	@until oc get crd kueues.kueue.openshift.io >/dev/null 2>&1; do \
    	echo "Wait until CRD kueues.kueue.openshift.io is ready..."; \
		sleep 10; \
	done

	@echo "Set Red Hat build of Kueue operator to be upgraded manually instead of automatic"
	@oc patch subscription kueue-operator \
    -n openshift-kueue-operator \
    --type=merge \
    -p '{"spec": {"installPlanApproval": "Manual"}}'


	oc apply -f ${BASE}/yaml/rhoai/rhoai.yaml
	@until oc get DSCInitialization/default-dsci -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' | grep -q "True"; do \
		echo "Waiting for DSCInitialization to be ready..."; \
		sleep 10; \
	done
	
	oc apply -f ${BASE}/yaml/rhoai/rhoai-cr.yaml	
	@until oc get DataScienceCluster/default-dsc -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' | grep -q "True"; do \
		echo "Waiting for DataScienceCluster to be ready..."; \
		sleep 10; \
	done

	@echo "Set RHOAI operator to be upgraded manually instead of automatic"
	@oc patch subscription rhods-operator \
    -n redhat-ods-operator \
    --type=merge \
    -p '{"spec": {"installPlanApproval": "Manual"}}'

	oc apply -f ${BASE}/yaml/rhoai/odhdashboardconfig.yaml

	oc delete pods -l app=rhods-dashboard -n redhat-ods-applications
	oc rollout status deployment/rhods-dashboard -n redhat-ods-applications

	#oc apply -f ${BASE}/yaml/rhoai/group.yaml
	oc apply -f ${BASE}/yaml/rhoai/template-rhaiis.yaml	
	oc apply -f ${BASE}/yaml/rhoai/hardwareprofile.yaml
	oc apply -f ${BASE}/yaml/rhoai/uwm.yaml

.PHONY: add-nfs-provisioner
add-nfs-provisioner:
	@$(BASE)/scripts/install-nfs-provisioner.sh

.PHONY: add-gpu-operator
add-gpu-operator:
	oc apply -f $(BASE)/yaml/rhoai/nfd.yaml

	@until oc get crd nodefeaturediscoveries.nfd.openshift.io >/dev/null 2>&1; do \
    	echo "Wait until CRD nodefeaturediscoveries.nfd.openshift.io is ready..."; \
		sleep 10; \
	done

	oc apply -f $(BASE)/yaml/rhoai/nfd-cr.yaml
	oc apply -f $(BASE)/yaml/rhoai/nvidia.yaml

	@until oc get crd clusterpolicies.nvidia.com>/dev/null 2>&1; do \
    	echo "Wait until CRD clusterpolicies.nvidia.com is ready..."; \
		sleep 10; \
	done

	oc apply -f $(BASE)/yaml/rhoai/nvidia-cr.yaml

.PHONY: setup-demo
setup-demo: setup-namespace deploy-minio setup-odh-tec deploy-pipline

	@oc apply -f $(BASE)/yaml/infra/model-pvc.yaml
	@oc apply -f $(BASE)/yaml/infra/llmcompressor-is.yaml
	#@oc apply -f $(BASE)/yaml/demo/anythingllm-wb.yaml 
	#@oc apply -f $(BASE)/yaml/demo/llama-cpp-wb.yaml
	@oc apply -f $(BASE)/yaml/demo/guidellm.yaml  -n ${NAMESPACE}
	@oc apply -f $(BASE)/yaml/demo/benchmark-arena.yaml -n ${NAMESPACE}
	@oc apply -f $(BASE)/yaml/demo/ai-toolkit.yaml -n ${NAMESPACE}
	@oc apply -f https://raw.githubusercontent.com/tsailiming/openshift-open-webui/refs/heads/main/open-webui.yaml -n ${NAMESPACE}
	@oc set env deploy/open-webui ENABLE_PERSISTENT_CONFIG=False -n ${NAMESPACE}

	@oc apply -f $(BASE)/yaml/demo/custom-model-catalog.yaml

	oc delete pods -l app=rhods-dashboard -n redhat-ods-applications
	oc rollout status deployment/rhods-dashboard -n redhat-ods-applications

.PHONY: download-models
download-models:
	@echo "Downloading Qwen/Qwen3-VL-8B-Instruct"
	@$(BASE)/scripts/download-model.sh s3 Qwen/Qwen3-VL-8B-Instruct

	@echo "Downloading openai/gpt-oss-20b"
	@$(BASE)/scripts/download-model.sh s3 openai/gpt-oss-20b

	@echo "Downloading RedHatAI/whisper-large-v3-turbo-FP8-dynamic"
	@$(BASE)/scripts/download-model.sh s3 RedHatAI/whisper-large-v3-turbo-FP8-dynamic

.PHONY: teardown-namespace
teardown-namespace:
	-oc delete project $(NAMESPACE)

.PHONY: setup-namespace
setup-namespace:
	-oc new-project $(NAMESPACE)
	@oc label namespace $(NAMESPACE) \
		maistra.io/member-of=istio-system \
		modelmesh-enabled=false \
		opendatahub.io/dashboard=true

.PHONY: setup-odh-tec
setup-odh-tec:
	@oc apply -f $(BASE)/yaml/infra/odh-tec.yaml -n $(NAMESPACE)
	
	@ODH_ROUTE=$$(oc get route odh-tec -n $(NAMESPACE) -o jsonpath='{.spec.host}') && \
	echo "S3 Browser: $${ODH_ROUTE}"

.PHONY: show-odh-tec
show-odh-tec:
	@ODH_ROUTE=$$(oc get route odh-tec -n $(NAMESPACE) -o jsonpath='{.spec.host}') && open https://$${ODH_ROUTE}

.PHONY: teardown-odh-tec
teardown-odh-tec:
	@oc delete -f $(BASE)/yaml/infra/odh-tec.yaml -n $(NAMESPACE)
	
.PHONY: teardown-all
teardown-all: teardown-minio teardown-odh-tec teardown-namespace
		
.PHONY: teardown-minio
teardown-minio:
	-oc delete -f $(BASE)/yaml/infra/minio.yaml -n $(NAMESPACE)
	
	@PV_NAME=$$(oc get pvc data-minio-0 -n $(NAMESPACE) -o jsonpath='{.spec.volumeName}' 2>/dev/null); \
	oc delete pvc data-minio-0 -n $(NAMESPACE); \
	if [ -z "$$PV_NAME" ]; then \
		echo "PVC data-minio-0 already deleted or has no PV bound."; \
	else \
		echo "Waiting for PV $$PV_NAME to be deleted..."; \
		until ! oc get pv $$PV_NAME >/dev/null 2>&1; do \
			echo "PV $$PV_NAME still exists..."; \
			sleep 2; \
		done; \
		echo "PV $$PV_NAME deleted."; \
	fi

.PHONY: deploy-minio
deploy-minio: teardown-minio
	@oc apply -f $(BASE)/yaml/infra/minio.yaml -n $(NAMESPACE)

	@until oc get statefulset minio -n $(NAMESPACE) -o jsonpath='{.status.readyReplicas}' | grep -q '1'; do \
		echo "Waiting for StatefulSet minio to have 1 ready replica..."; \
		sleep 10; \
	done
	@echo "StatefulSet minio has 1 ready replica."

	-oc delete secret aws-connection-my-storage -n $(NAMESPACE)

	@AWS_ACCESS_KEY_ID=$$(oc extract secret/minio  --to=- --keys=MINIO_ROOT_USER -n $(NAMESPACE) 2>/dev/null | tr -d '\n' | base64 ) \
	AWS_SECRET_ACCESS_KEY=$$(oc extract secret/minio  --to=- --keys=MINIO_ROOT_PASSWORD -n $(NAMESPACE) 2>/dev/null | tr -d '\n' | base64) \
	AWS_S3_ENDPOINT=minio.$(NAMESPACE).svc.cluster.local \
	AWS_ENDPOINT_URL=minio.$(NAMESPACE).svc.cluster.local \
		envsubst < $(BASE)/yaml/infra/data-connection-s3.yaml.tmpl | oc apply -n $(NAMESPACE) -f -	
		
	@$(BASE)/scripts/run-job.sh $(BASE)/yaml/infra/setup-s3.yaml.tmpl $(NAMESPACE) setup-s3-job aws-connection-my-storage

.PHONY: teardown-pipeline
teardown-pipeline: 
	-oc delete -f $(BASE)/yaml/infra/dashboard-dspa-secret.yaml -n $(NAMESPACE)
	-oc delete -f $(BASE)/yaml/infra/dspa.yaml -n $(NAMESPACE)

.PHONY: deploy-pipeline
deploy-pipline: teardown-pipeline	
	@AWS_ACCESS_KEY_ID=$$(oc extract secret/minio  --to=- --keys=MINIO_ROOT_USER -n $(NAMESPACE) 2>/dev/null | tr -d '\n' | base64 ) \
	AWS_SECRET_ACCESS_KEY=$$(oc extract secret/minio  --to=- --keys=MINIO_ROOT_PASSWORD -n $(NAMESPACE) 2>/dev/null | tr -d '\n' | base64) \
	AWS_S3_ENDPOINT=minio.$(NAMESPACE).svc.cluster.local \
	AWS_ENDPOINT_URL=minio.$(NAMESPACE).svc.cluster.local \
	  envsubst < $(BASE)/yaml/infra/pipeline-connection-s3.yaml.tmpl | oc apply -n $(NAMESPACE) -f -	

	@oc apply -f $(BASE)/yaml/infra/dashboard-dspa-secret.yaml -n $(NAMESPACE)
	@oc apply -f $(BASE)/yaml/infra/dspa.yaml -n $(NAMESPACE)