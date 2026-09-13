# Enable NeMo Guardrails in Models as a Service (MaaS)

This guide shows how to enable **NeMo Guardrails** for Models as a Service (MaaS) by connecting the default NeMo Guardrails deployment to MaaS **Ingress Payload Processing (IPP)**.

The example uses:

* NeMo Guardrails deployed in the `demo` namespace.
* MaaS IPP running in `openshift-ingress`.
* An OpenAI-compatible model endpoint used by NeMo Guardrails for its LLM-based rails.
* An OpenShift Route to expose the NeMo Guardrails checks API to MaaS IPP.

## How it works

There are two separate URLs involved, and they serve different purposes.

### 1. Guardrails LLM endpoint

`OPENAI_BASE_URL` is configured in the NeMo Guardrails deployment.

It tells NeMo Guardrails where to send LLM requests for rails such as `self_check_input` and `self_check_output`.

**This must be a direct OpenAI-compatible model endpoint. It must not point to MaaS.**

For example:

```text
https://<model-endpoint>/v1
```

The endpoint could be an in-cluster model Service, a KServe/LLMInferenceService workload endpoint, or another OpenAI-compatible endpoint.

### 2. Guardrails checks endpoint

`--nemo-guardrail-check-url` is configured in MaaS IPP.

It tells IPP where to send requests for guardrail evaluation.

This should point to the **NeMo Guardrails OpenShift Route**, for example:

```text
https://nemo-guardrails-<namespace>.<cluster-domain>
```

IPP then calls:

```text
POST /v1/guardrail/checks
```

### Request flow

For a MaaS request, the flow is approximately:

```text
Client
  |
  | MaaS request
  v
MaaS Route
  |
  v
payload-processing / IPP
  |
  | guardrail check
  v
NeMo Guardrails Route
  |
  v
NeMo Guardrails
  |
  | LLM request
  v
Direct model endpoint
```

The important point is that the Guardrails LLM request does **not** go back through MaaS IPP.

## Current limitation

IPP stores a single NeMo Guardrails checks URL in `payload-processing-plugins`.

After the NeMo plugins are enabled, **all models and tenants using MaaS payload processing use the same NeMo Guardrails endpoint and configuration**.

This bootstrap example does not provide per-model or per-route guardrail policies.

## Prerequisites

* `oc` CLI logged in to the OpenShift cluster.
* Permission to modify resources in the `demo` and `openshift-ingress` namespaces.
* MaaS and the `payload-processing` deployment running in `openshift-ingress`.
* OpenShift AI 3.5+ with the default MaaS gateway.
* [`uv`](https://docs.astral.sh/uv/) for running `scripts/enable-ipp-nemo.py`.

---

## Step 1: Deploy and configure NeMo Guardrails

The `setup-guardrail` Make target:

1. Creates the API key Secret used by the guardrails configuration.
2. Renders the NeMo Guardrails ConfigMap from its template.
3. Applies the `NemoGuardrails` custom resource.
4. Deploys the guardrails workload in the `demo` namespace.

### Required environment variables

| Variable            | Purpose                                               |
| ------------------- | ----------------------------------------------------- |
| `OPENAI_API_KEY`    | API key used by the NeMo Guardrails LLM configuration |
| `OPENAI_MODEL_NAME` | Model name used by the guardrails configuration       |
| `OPENAI_BASE_URL`   | Direct OpenAI-compatible endpoint for that model      |

### Example

From the repository root:

```bash
export OPENAI_API_KEY='<your-key>'
export OPENAI_MODEL_NAME='<model-id>'
export OPENAI_BASE_URL='https://<direct-model-endpoint>/v1'

make setup-guardrail
```

The exact model and endpoint are deployment-specific. For example, the model could be served by an in-cluster KServe workload or by another OpenAI-compatible endpoint.

For an in-cluster model, use its workload Service or another direct OpenAI-compatible endpoint.

For example:

```text
http://<model-service>.<namespace>.svc.cluster.local:<port>/v1
```

The exact Service name, namespace, and port depend on how the model is deployed.

---

## Step 2: Get the NeMo Guardrails Route

The NeMo Guardrails checks API is exposed through an OpenShift Route.

Find the Route:

```bash
oc get route -n demo
```

If the Route is named `nemo-guardrails`, retrieve its host with:

```bash
GUARDRAIL_HOST="$(oc get route nemo-guardrails -n demo -o jsonpath='{.spec.host}')"

export GUARDRAIL_BASE_URL="https://${GUARDRAIL_HOST}"
```

The resulting URL should look similar to:

```text
https://nemo-guardrails-demo.<cluster-domain>
```

`GUARDRAIL_BASE_URL` is the **base URL**. The checks API is:

```text
${GUARDRAIL_BASE_URL}/v1/guardrail/checks
```

If your Route has a different name, use:

```bash
oc get route -n demo
```

and identify the Route exposing the NeMo Guardrails service.

### Why use the Route?

MaaS IPP runs in `openshift-ingress`, and its NetworkPolicy restricts the destinations that the payload-processing workload can reach.

The supported path for the NeMo check is therefore the HTTPS Route:

```text
payload-processing
       |
       | HTTPS / 443
       v
NeMo Guardrails Route
       |
       v
NeMo Guardrails
```

Do not configure IPP to use an internal Service URL such as:

```text
http://<service>.demo.svc.cluster.local:80
```

The Route and the direct model endpoint are **not the same thing**:

| URL                                                 | Used by         | Purpose                               |
| --------------------------------------------------- | --------------- | ------------------------------------- |
| `OPENAI_BASE_URL`                                   | NeMo Guardrails | Calls the LLM used by guardrails      |
| `GUARDRAIL_BASE_URL` / `--nemo-guardrail-check-url` | MaaS IPP        | Calls the NeMo Guardrails checks API  |
| `MAAS_BASE_URL`                                     | MaaS clients    | Sends inference requests through MaaS |

---

## Step 3: Enable NeMo Guardrails in MaaS IPP

`scripts/enable-ipp-nemo.py` updates the live configmap:

```text
openshift-ingress/payload-processing-plugins
```

It can enable:

* `nemo-request-guard` for input/request rails
* `nemo-response-guard` for output/response rails

After updating the configuration, the script restarts `payload-processing` and waits for the rollout to complete.

### Enable request/input guardrails

If neither `--request` nor `--response` is specified, request guardrails are enabled by default:

```bash
uv run scripts/enable-ipp-nemo.py \
  --nemo-guardrail-check-url "${GUARDRAIL_BASE_URL}"
```

### Enable both request and response guardrails

```bash
uv run scripts/enable-ipp-nemo.py \
  --request \
  --response \
  --nemo-guardrail-check-url "${GUARDRAIL_BASE_URL}"
```

If the supplied URL does not contain a path, the script appends:

```text
/v1/guardrail/checks
```

Therefore the following is also valid:

```bash
uv run scripts/enable-ipp-nemo.py \
  --request \
  --response \
  --nemo-guardrail-check-url "${GUARDRAIL_BASE_URL}/v1/guardrail/checks"
```

### Dry run

To preview the resulting configuration without modifying the cluster:

```bash
uv run scripts/enable-ipp-nemo.py \
  --request \
  --response \
  --nemo-guardrail-check-url "${GUARDRAIL_BASE_URL}" \
  --dry-run
```

### Script options

| Flag                         | Description                                                                                                                                              |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--nemo-guardrail-check-url` | **Required.** NeMo Guardrails checks endpoint. Defaults to HTTPS if no scheme is specified. `/v1/guardrail/checks` is appended when no path is provided. |
| `--request`                  | Enable `nemo-request-guard` for input/request rails.                                                                                                     |
| `--response`                 | Enable `nemo-response-guard` for output/response rails.                                                                                                  |
| `--dry-run`                  | Print the resulting configuration without applying it or restarting IPP.                                                                                 |

The default timeout for guardrail HTTP calls is 10 seconds.

---

## Step 4: Test the configuration

### Test NeMo Guardrails directly

`scripts/test-guardrail.sh` calls:

```text
POST /v1/guardrail/checks
```

using `GUARDRAIL_BASE_URL`.

Set the Route URL:

```bash
export GUARDRAIL_BASE_URL="https://${GUARDRAIL_HOST}"
```

Then run:

```bash
./scripts/test-guardrail.sh "<model-name>" "What is the capital of France?"
```

The script requires:

* `curl`
* `jq`

This test verifies that the NeMo Guardrails API and its configured LLM endpoint are working independently of MaaS.

### Test MaaS end-to-end

`scripts/test-maas.sh` sends a request through MaaS and therefore exercises the IPP guardrails after they have been enabled.

```bash
export MAAS_BASE_URL='https://<maas-route-host>'
export MAAS_TOKEN='<bearer-token>'
export MAAS_MODEL_NAME='<model-id>'

./scripts/test-maas.sh
```

Use the same MaaS Route that a client would use.

The request flow is:

```text
MaaS client
    |
    v
MaaS Route
    |
    v
payload-processing / IPP
    |
    +-- NeMo request guard
    |       |
    |       v
    |   NeMo Guardrails
    |
    v
Model
    |
    +-- NeMo response guard
    |       |
    |       v
    |   NeMo Guardrails
    |
    v
MaaS client
```

The script requires:

* `curl`
* `jq`

---

## Reference

| Path                           | Purpose                                                              |
| ------------------------------ | -------------------------------------------------------------------- |
| `Makefile` (`setup-guardrail`) | Creates the Secret and ConfigMap and applies the `NemoGuardrails` CR |
| `yaml/demo/nemo-cm.yaml.tmpl`  | NeMo Guardrails configuration and `OPENAI_*` substitution            |
| `yaml/demo/nemo-cr.yaml`       | `NemoGuardrails` instance                                            |
| `scripts/enable-ipp-nemo.py`   | Enables NeMo plugins in MaaS IPP                                     |
| `scripts/test-guardrail.sh`    | Tests the NeMo Guardrails checks API                                 |
| `scripts/test-maas.sh`         | Tests MaaS `/v1/models` and `/v1/chat/completions`                   |

### Related repositories

1. ai-gateway-payload-processing [source code](https://github.com/opendatahub-io/ai-gateway-payload-processing/tree/main/examples)
2. Same configuration: [nemo-maas](https://github.com/eformat/nemo-maas/tree/main), [IPP NeMo example](https://github.com/opendatahub-io/ai-gateway-payload-processing/tree/main/examples/nemo)
3. [Sample guardrail config (banking)](https://github.com/trustyai-explainability/trustyai-llm-demo/tree/main/nemo-guardrails-config-collection/banking)
4. [Models as a Service source (models-as-a-service)](https://github.com/opendatahub-io/models-as-a-service)

---

## Troubleshooting

| Symptom                                            | Check                                                                                                                |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Guardrail checks time out from MaaS                | Verify `--nemo-guardrail-check-url` points to the NeMo Guardrails **Route** and uses HTTPS.                          |
| Guardrails work directly but not from MaaS         | Check NetworkPolicy and connectivity from `payload-processing` to the Route.                                         |
| Recursive requests or unexpected nested MaaS calls | Verify `OPENAI_BASE_URL` does **not** point to MaaS. It must point directly to the backing model endpoint.           |
| `oc patch` or rollout fails                        | Verify your `oc` context and permissions and confirm `payload-processing-plugins` exists in `openshift-ingress`.     |
| Guardrails return errors when calling the model    | Verify `OPENAI_BASE_URL`, `OPENAI_MODEL_NAME`, and `OPENAI_API_KEY` in the NeMo Guardrails configuration.            |
| 403 or connection errors from IPP                  | Verify the NeMo Guardrails Route is reachable over HTTPS and that the relevant NetworkPolicy permits the connection. |