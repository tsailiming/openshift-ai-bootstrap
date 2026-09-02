BASE:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
SHELL=/bin/sh
NAMESPACE=demo
RHAIIS_IMAGE=registry.redhat.io/rhaii-early-access/vllm-cuda-rhel9:3.5.0-ea.2
RHAIIS_VLLM_VERSION=0.19.1
AWS_METAL_INSTANCE=c5n.metal
AWS_AZ=ap-northeast-1a

.PHONY: rhoai-prereq
rhoai-prereq:
	@echo "Installing kueue operator"
	oc apply -f $(BASE)/yaml/rhoai/kueue.yaml
	@$(BASE)/scripts/check-operator-install-status.sh kueue-operator openshift-kueue-operator	
	oc apply -f $(BASE)/yaml/rhoai/kueue-cr.yaml
	
	@echo "Installing leader worker set operator"
	oc apply -f $(BASE)/yaml/rhoai/lws.yaml
	@$(BASE)/scripts/check-operator-install-status.sh leader-worker-set openshift-lws-operator 
	
	@echo "Installing leader jobset operator"
	oc apply -f $(BASE)/yaml/rhoai/jobset.yaml
	@$(BASE)/scripts/check-operator-install-status.sh job-set openshift-jobset-operator
	oc apply -f $(BASE)/yaml/rhoai/jobset-cr.yaml

	@echo "Enabling user workload monitoring"
	oc apply -f ${BASE}/yaml/rhoai/uwm.yaml
	
	@echo "Installing cluster observability operator"
	oc apply -f $(BASE)/yaml/rhoai/coo.yaml
	@$(BASE)/scripts/check-operator-install-status.sh cluster-observability-operator openshift-cluster-observability-operator
  
	@echo "Installing tempo operator"
	oc apply -f $(BASE)/yaml/rhoai/tempo.yaml
	@$(BASE)/scripts/check-operator-install-status.sh tempo-product openshift-tempo-operator

	@echo "Installing otel operator"
	oc apply -f ${BASE}/yaml/rhoai/otel.yaml
	@$(BASE)/scripts/check-operator-install-status.sh opentelemetry-product openshift-opentelemetry-operator
	
	@echo "Installing Red Hat Connectivity Link"
	oc apply -f $(BASE)/yaml/rhoai/kuadrant.yaml
	@$(BASE)/scripts/check-operator-install-status.sh rhcl-operator openshift-operators		
	@echo "Enable connectivity link console plugin"
	@oc patch console.operator.openshift.io cluster --type=json -p='[{"op":"add","path":"/spec/plugins/-","value":"kuadrant-console-plugin"}]'

	@echo "Installing Red Hat build of Agent Sandbox"
	oc apply -f $(BASE)/yaml/rhoai/agent-sandbox.yaml
	@$(BASE)/scripts/check-operator-install-status.sh agent-sandbox-operator agent-sandbox-system
	
	@echo "Installing Red Hat OpenShift Pipelines"
	oc apply -f $(BASE)/yaml/rhoai/pipeline.yaml
	@$(BASE)/scripts/check-operator-install-status.sh openshift-pipelines-operator-rh openshift-pipelines

	@echo "Enable pipeline console plugin"
	@oc patch console.operator.openshift.io cluster --type=json -p='[{"op":"add","path":"/spec/plugins/-","value":"pipelines-console-plugin"}]'

.PHONY: setup-rhoai
setup-rhoai: add-gpu-operator add-nfs-provisioner rhoai-prereq
	
	oc apply -f ${BASE}/yaml/rhoai/rhoai.yaml
	@$(BASE)/scripts/check-operator-install-status.sh rhods-operator redhat-ods-operator
	@until oc get DSCInitialization/default-dsci -o jsonpath='{.status.conditions[?(@.type=="Available")].status}' | grep -q "True"; do \
		echo "Waiting for DSCInitialization to be ready..."; \
		sleep 10; \
	done
	
	@oc patch dsci default-dsci --type=merge \
  	-p '{"spec":{"monitoring":{"managementState":"Managed","namespace":"redhat-ods-monitoring","alerting":{},"metrics":{"replicas":1,"storage":{"size":"5Gi","retention":"90d"},"exporters":{}},"traces":{"sampleRatio":"0.1","storage":{"backend":"pv","retention":"2160h"},"exporters":{}}}}}'

	@CSV=$$(oc get subscription rhods-operator -n redhat-ods-operator -o jsonpath='{.status.installedCSV}' 2>/dev/null); \
	if [ -z "$$CSV" ]; then \
		echo "No installed CSV found for subscription rhods-operator"; \
		exit 1; \
	fi; \
	oc patch csv "$$CSV" -n redhat-ods-operator --type=json \
		-p="[{\"op\":\"replace\",\"path\":\"/spec/install/spec/deployments/0/spec/replicas\",\"value\":1}]";	

	@oc scale deployment rhods-operator \
		-n redhat-ods-operator \
		--replicas=1
	@oc rollout restart deployment/rhods-operator -n redhat-ods-operator
	@oc rollout status deployment/rhods-operator -n redhat-ods-operator
	
	@oc apply -f ${BASE}/yaml/rhoai/rhoai-cr.yaml	
	@until oc get DataScienceCluster/default-dsc -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' | grep -q "True"; do \
		echo "Waiting for DataScienceCluster to be ready..."; \
		sleep 10; \
	done

	@oc apply -f ${BASE}/yaml/rhoai/odhdashboardconfig.yaml
	
	@echo "Waiting for deployment/rhods-dashboard to appear..."
	@until oc get deployment/rhods-dashboard -n redhat-ods-applications >/dev/null 2>&1; do \
		sleep 2; \
	done
	@echo "deployment/rhods-dashboard found. Restarting..."

	@oc scale deployment rhods-dashboard \
		-n redhat-ods-applications \
		--replicas=1
	
	@oc rollout restart deployment/rhods-dashboard -n redhat-ods-applications
	@oc rollout status deployment/rhods-dashboard -n redhat-ods-applications
	
	@echo "Patch to increase memory for data-science-gateway istio-proxy"
	@oc apply -f ${BASE}/yaml/rhoai/data-science-gateway-config.yaml

	@RHAIIS_IMAGE=$(RHAIIS_IMAGE) \
		RHAIIS_VLLM_VERSION=$(RHAIIS_VLLM_VERSION) \
		envsubst < $(BASE)/yaml/rhoai/template-rhaiis.yaml.tmpl | oc apply -n redhat-ods-applications -f -

	oc apply -f ${BASE}/yaml/rhoai/hardwareprofile.yaml
	oc apply -f ${BASE}/yaml/rhoai/mlflow-cr.yaml
	oc apply -f ${BASE}/yaml/rhoai/evalhub-cr.yaml
	
	@echo "Installing grafana operator"
	@oc apply -f ${BASE}/yaml/rhoai/grafana.yaml
	@$(BASE)/scripts/check-operator-install-status.sh grafana user-grafana	
	
	@echo "Configuring the NVIDIA DCGM Exporter Dashboard"
	@curl -L https://raw.githubusercontent.com/NVIDIA/dcgm-exporter/main/grafana/dcgm-exporter-dashboard.json \
	| oc create configmap nvidia-dcgm-exporter-dashboard \
		-n openshift-config-managed \
		--from-file=dcgm-exporter-dashboard.json=/dev/fd/0 \
		--dry-run=client -o yaml \
	| oc apply -f -

	oc label configmap nvidia-dcgm-exporter-dashboard -n openshift-config-managed \
	  console.openshift.io/dashboard=true --overwrite	

.PHONY: setup-osc
setup-osc:
	@echo "OpenShift sandboxed containers Operator"
	oc apply -f $(BASE)/yaml/rhoai/osc.yaml
	@$(BASE)/scripts/check-operator-install-status.sh sandboxed-containers-operator openshift-sandboxed-containers-operator
	
	@oc apply -f $(BASE)/yaml/rhoai/kata-config.yaml

	@machineset=$$(oc get machinesets -n openshift-machine-api \
		-o jsonpath='{range .items[?(@.spec.template.spec.providerSpec.value.placement.availabilityZone=="$(AWS_AZ)")]}{.metadata.name}{"\n"}{end}' | head -1); \
	if [ -z "$$machineset" ]; then \
		echo "ERROR: No MachineSet found in $(AWS_AZ)"; \
		exit 1; \
	fi; \
	echo "Found MachineSet: $$machineset"; \
	$(BASE)/scripts/clone-machineset.sh "$$machineset" $(AWS_METAL_INSTANCE) --on-demand; \
	oc patch machineset "$$machineset" -n openshift-machine-api --type=merge \
		-p '{"spec":{"template":{"metadata":{"labels":{"feature.node.kubernetes.io/runtime.kata":"true"}}}}}'

.PHONY: setup-mcp-gateway
setup-mcp-gateway:
	@$(BASE)/scripts/setup-mcp-gateway.sh

.PHONY: setup-openshell
setup-openshell: setup-osc

	@set -eu; \
	\
	if helm status openshell -n openshell >/dev/null 2>&1; then \
		echo "OpenShell is already installed in namespace openshell."; \
		echo "Nothing to do."; \
		exit 0; \
	fi; \
	\
	TMPDIR=$$(mktemp -d); \
	echo "TMPDIR=$$TMPDIR"; \
	trap 'rm -rf "$$TMPDIR"' EXIT; \
	\
	echo "Cloning agent-ops repository..."; \
	git clone https://github.com/opendatahub-io/agent-ops.git "$$TMPDIR/agent-ops"; \
	\
	echo "Running deploy-openshell.sh..."; \
	cd "$$TMPDIR/agent-ops"; \
	./scripts/deploy-openshell.sh

	@echo "Setting openshell default runtimeclass to kata"
	@helm upgrade openshell oci://ghcr.io/nvidia/openshell/helm-chart \
	--version 0.0.85 \
	--namespace openshell \
	--reuse-values \
	--set supervisor.topology=sidecar \
	--set supervisor.sidecar.processBinaryAwareNetworkPolicy=true \
	--set server.defaultRuntimeClassName=kata

.PHONY: setup-maas
setup-maas:
	@set -eu; \
	TMPDIR=$$(mktemp -d); \
	echo "TMPDIR=$$TMPDIR"; \
	TMP_KUBECONFIG="$$TMPDIR/kubeconfig"; \
	trap 'rm -rf "$$TMPDIR"' EXIT; \
	\
	SERVER=$$(oc whoami --show-server); \
	\
	if TOKEN=$$(oc whoami -t 2>/dev/null); then \
		echo "Using existing OAuth token"; \
	else \
		echo "No OAuth token found. Creating cluster-admin ServiceAccount token..."; \
		TOKEN=$$($(BASE)/scripts/get-token.sh); \
		echo "Token length: $${#TOKEN}"; \
		test -n "$$TOKEN"; \
		export KUBECONFIG="$$TMP_KUBECONFIG"; \
		oc login \
			--token="$$TOKEN" \
			--server="$$SERVER" \
			--insecure-skip-tls-verify=true; \
	fi; \
	\
	echo "Setting modelsAsService to Managed in DSC"; \
	oc patch datasciencecluster default-dsc --type='merge' \
		-p '{"spec":{"components":{"aigateway":{"modelsAsAService":{"managementState":"Managed"}}}}}'; \
	echo "Cloning repository..."; \
	git clone https://github.com/rh-aiservices-bu/rhoai-maas-guide.git "$$TMPDIR/rhoai-maas-guide"; \
	\
	echo "Running setup-maas.sh..."; \
	cd "$$TMPDIR/rhoai-maas-guide"; \
	./scripts/setup-maas.sh 

	@oc patch maastenantconfig default-tenant \
		-n models-as-a-service \
		--type=merge \
		-p '{"spec":{"telemetry":{"enabled":true,"metrics":{"captureGroup":true,"captureModelUsage":true,"captureOrganization":true,"captureUser":true}}}}'

	@oc patch maassubscription gpt-oss-20b-free \
		-n models-as-a-service \
		--type=merge \
		-p '{"spec":{"tokenMetadata":{"costCenter":"101","organizationId":"APAC AI"}}}'

	@oc patch maassubscription gpt-oss-20b-premium \
		-n models-as-a-service \
		--type=merge \
		-p '{"spec":{"tokenMetadata":{"costCenter":"101","organizationId":"APAC AI"}}}'
		
.PHONY: add-nfs-provisioner
add-nfs-provisioner:
	@$(BASE)/scripts/install-nfs-provisioner.sh

.PHONY: add-gpu-operator
add-gpu-operator:
	oc apply -f $(BASE)/yaml/rhoai/nfd.yaml

	@$(BASE)/scripts/check-operator-install-status.sh nfd openshift-nfd
	
	oc apply -f $(BASE)/yaml/rhoai/nfd-cr.yaml
	oc apply -f $(BASE)/yaml/rhoai/nvidia.yaml

	@$(BASE)/scripts/check-operator-install-status.sh gpu-operator-certified nvidia-gpu-operator

	oc apply -f $(BASE)/yaml/rhoai/nvidia-cr.yaml

.PHONY: setup-demo
setup-demo: setup-namespace deploy-minio setup-odh-tec deploy-pipline

	oc apply -f $(BASE)/yaml/infra/model-pvc.yaml
	oc apply -f $(BASE)/yaml/infra/llmcompressor-is.yaml
	#@oc apply -f $(BASE)/yaml/demo/anythingllm-wb.yaml 
	#@oc apply -f $(BASE)/yaml/demo/llama-cpp-wb.yaml
	oc apply -f $(BASE)/yaml/demo/guidellm.yaml  -n ${NAMESPACE}
	oc apply -f $(BASE)/yaml/demo/benchmark-arena.yaml -n ${NAMESPACE}
	oc apply -f $(BASE)/yaml/demo/ai-toolkit.yaml -n ${NAMESPACE}
	oc apply -f https://raw.githubusercontent.com/tsailiming/openshift-open-webui/refs/heads/main/open-webui.yaml -n ${NAMESPACE}
	oc set env deploy/open-webui ENABLE_PERSISTENT_CONFIG=False -n ${NAMESPACE}
	oc apply -f $(BASE)/yaml/demo/custom-model-catalog.yaml

	oc delete pods -l app.kubernetes.io/name=model-catalog -n rhoai-model-registries

.PHONY: setup-ai-playground
setup-ai-playground: 
# 	@echo "Serving llama-32-3b-instruct"
# 	@$(BASE)/scripts/serve-model.sh oci llama-32-3b-instruct oci://quay.io/redhat-ai-services/modelcar-catalog:llama-3.2-3b-instruct "--max-model-len 32768 --enable-auto-tool-choice --tool-call-parser=llama3_json --chat-template=/opt/app-root/template/tool_chat_template_llama3.2_json.jinja"
	
# 	@echo "Downloading and deploying Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
# 	@$(BASE)/scripts/download-model.sh pvc Qwen/Qwen3-30B-A3B-Thinking-2507-FP8
# 	@$(BASE)/scripts/scripts/serve-model.sh pvc qwen3.5-35b-a3b-fp8-dynamic RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic/ "--max-model-len 32768 --trust-remote-code --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --mm-encoder-tp-mode data"

	oc apply -f $(BASE)/yaml/demo/mcp-kubernetes.yaml -n ${NAMESPACE}
	oc apply -f $(BASE)/yaml/demo/mcp-weather.yaml -n ${NAMESPACE}
	oc apply -f $(BASE)/yaml/demo/lsd-mcp-cm.yaml
	oc apply -f $(BASE)/yaml/demo/llama-stack-cm.yaml -n ${NAMESPACE}
	oc apply -f $(BASE)/yaml/demo/ogx-ai-playground.yaml -n ${NAMESPACE}

	oc delete pod -l app=llama-stack -n ${NAMESPACE} --ignore-not-found  
	oc rollout status deployment/lsd-genai-playground -n ${NAMESPACE}

.PHONY: download-and-serve-models
download-and-serve-models:
	#@echo "Downloading RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic"
	#@$(BASE)/scripts/download-model.sh pvc RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic
	#@$(BASE)/scripts/serve-model.sh pvc qwen35-35b-A3b-fp8-dynamic RedHatAI/Qwen3.5-35B-A3B-FP8-dynamic "--max-model-len 4096"

	@echo "Downloading Qwen/Qwen3.5-27B-FP8"
	@$(BASE)/scripts/download-model.sh pvc Qwen/Qwen3.5-27B-FP8
	@$(BASE)/scripts/serve-model.sh pvc qwen35-27b-fp8 Qwen/Qwen3.5-27B-FP8 "--max-model-len 2048 --gpu-memory-utilization 0.97 --kv-cache-dtype fp8"

	@echo "Downloading openai/gpt-oss-20b"
	@$(BASE)/scripts/download-model.sh pvc openai/gpt-oss-20b
	@$(BASE)/scripts/serve-model.sh pvc gpt-oss-20b openai/gpt-oss-20b "--max-model-len 2048 --gpu-memory-utilization 0.97 --kv-cache-dtype fp8"

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

.PHONY: setup-kueue-demo
setup-kueue-demo:
	@oc apply -k "github.com/tsailiming/openshift-ai-kueue//yaml?ref=main"

.PHONY: setup-multi-user
setup-multi-user:
	@HASH=$$(oc get secret kubeadmin -n kube-system \
		-o jsonpath='{.data.kubeadmin}' | base64 --decode); \
	\
	echo "Using same kubeadmin bcrypt hash $$HASH"; \
	rm -f /tmp/users.htpasswd; \
	echo "user1:$$HASH" >> /tmp/users.htpasswd; \
	echo "user2:$$HASH" >> /tmp/users.htpasswd; \
	\
	oc create secret generic htpasswd-secret \
		--from-file=htpasswd=/tmp/users.htpasswd \
		-n openshift-config --dry-run=client -o yaml | oc apply -f -

	@oc apply -f $(BASE)/yaml/infra/oauth.yaml

	-oc new-project user1
	@oc label namespace user1 \
		maistra.io/member-of=istio-system \
		modelmesh-enabled=false \
		opendatahub.io/dashboard=true

	-oc new-project user2
	@oc label namespace user2 \
		maistra.io/member-of=istio-system \
		modelmesh-enabled=false \
		opendatahub.io/dashboard=true

	@oc adm policy add-role-to-user edit user1 -n user1
	@oc adm policy add-role-to-user edit user2 -n user2
