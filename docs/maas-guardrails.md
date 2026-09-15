# Enable NeMo Guardrails in Models as a Service (MaaS)

This guide shows how to enable **NeMo Guardrails** for Models as a Service (MaaS) by connecting the default NeMo Guardrails deployment to MaaS **Ingress Payload Processing (IPP)**.

The example uses:

* NeMo Guardrails deployed in the `demo` namespace.
* MaaS IPP running in `openshift-ingress`.
* An OpenAI-compatible model endpoint used by NeMo Guardrails for its LLM-based rails.
* An OpenShift Route to expose the NeMo Guardrails checks API to MaaS IPP.

## IMPORTANT: `payload-processing` image version (TrustyAI `/v1/guardrail/checks`)

> Read this before enabling guardrails or debugging MaaS 500 errors.
>
> This bootstrap wires MaaS IPP to **TrustyAI NeMo Guardrails**, which calls:
>
> `POST /v1/guardrail/checks`
>
> and returns a JSON body whose top-level `status` is **`success`**, **`blocked`**, or **`error`** (not NVIDIA upstream `/v1/checks`, which used **`passed`**).
>
> Older `payload-processing` images (built from [ai-gateway-payload-processing](https://github.com/opendatahub-io/ai-gateway-payload-processing) **before** [PR #434](https://github.com/opendatahub-io/ai-gateway-payload-processing/pull/434)) only treat **`passed`** as “allow”. When NeMo correctly returns **`success`**, IPP fails closed with:
>
> ```text
> unknown NeMo guardrails status "success"
> ```
>
> and MaaS returns **500 Internal Server Error** even though NeMo allowed the request.
>
> **What works vs what breaks on those images**
>
> | NeMo result | Old IPP (pre-#434) | IPP with #434+ |
> | ----------- | ------------------ | -------------- |
> | Input blocked (`status: blocked`) | 403 (looks fine) | 403 |
> | Input allowed (`status: success`) | **500** (false failure) | Request continues to the model |
> | NeMo processing error (`status: error`) | Often **500** | **503** (fail-closed) |
>
> **What you should do**
>
> 1. Upgrade the **`payload-processing`** deployment in `openshift-ingress` to an image that includes merge commit **`d32e434`** or later (merged 2026-08-21 in opendatahub-io `main`).
> 2. Confirm on the cluster: `oc get deploy payload-processing -n openshift-ingress -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'` and match it to your RHOAI/MaaS release notes or a build **after** #434.
> 3. Direct NeMo tests (`scripts/test-guardrail.sh`, `curl` to `/v1/guardrail/checks`) can **pass** while MaaS still **500** until IPP is upgraded — that is expected with this mismatch.
>
> Enabling plugins via `scripts/enable-ipp-nemo.py` only changes IPP **configuration**; it does **not** replace the container image. NeMo and bootstrap config can be correct while IPP is still too old.
>
> Reference fix: [fix(guardrails): align NeMo status constants with /v1/guardrail/checks endpoint (#434)](https://github.com/opendatahub-io/ai-gateway-payload-processing/pull/434).

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
* A `payload-processing` image that includes [ai-gateway-payload-processing PR #434](https://github.com/opendatahub-io/ai-gateway-payload-processing/pull/434) (`success` / `error` status handling for `/v1/guardrail/checks`). See the disclaimer above if allowed requests return 500.
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

| Variable                   | Purpose                                                                 |
| -------------------------- | ----------------------------------------------------------------------- |
| `OPENAI_API_KEY`           | API key used by the NeMo Guardrails LLM configuration                   |
| `OPENAI_MODEL_NAME`        | Model name for NeMo’s **main** generation path in `config.yaml`         |
| `OPENAI_BASE_URL`          | Direct OpenAI-compatible base URL for that model (must **not** be MaaS) |
| `GUARDRAIL_LLM_BASE_URL`   | Direct OpenAI-compatible base URL for `self_check` rails (see template) |
| `GUARDRAIL_LLM_MODEL_NAME` | Model name NeMo uses for `self_check_input` / `self_check_output`       |

`GUARDRAIL_LLM_*` must point at a **direct** model endpoint, not the MaaS Route (same recursion rule as `OPENAI_BASE_URL`).

### Example

From the repository root:

```bash
export OPENAI_API_KEY='<your-key>'
export OPENAI_MODEL_NAME='<model-id>'
export OPENAI_BASE_URL='https://<direct-model-endpoint>/v1'
export GUARDRAIL_LLM_BASE_URL='https://<direct-self-check-endpoint>/v1'
export GUARDRAIL_LLM_MODEL_NAME='<self-check-model-id>'

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
| `GUARDRAIL_BASE_URL` / `--nemo-guardrail-check-url` | MaaS IPP, `test-guardrail.sh` | NeMo Guardrails **Route** base URL (checks API under `/v1/guardrail/checks`) |
| `SELF_CHECK_LLM_URL` / `SELF_CHECK_LLM_NAME`       | `test-guardrail.sh` only      | Same self-check model as `GUARDRAIL_LLM_*` — used to verify that endpoint directly |
| `MAAS_BASE_URL` / `MAAS_TOKEN`                     | `test-maas.sh`, MaaS clients  | MaaS Route and bearer token for `/v1/models` and `/v1/chat/completions` |

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

Both test scripts take **two positional arguments**: `<model-name>` and `<prompt>`. 

| Script | Usage | Help |
| ------ | ----- | ---- |
| `scripts/test-guardrail.sh` | `./scripts/test-guardrail.sh <model-name> <prompt>` | `./scripts/test-guardrail.sh -h` |
| `scripts/test-maas.sh` | `./scripts/test-maas.sh <model-name> <prompt>` | `./scripts/test-maas.sh -h` |

Use a **banking-related** prompt with the demo gatekeeper (for example `What is the bank rate?`). Prompts such as “Say hello in one short sentence.” are expected to be **blocked** on the MaaS path when input guardrails are enabled.

### Environment variables for `test-guardrail.sh`

| Variable | Required | Purpose |
| -------- | -------- | ------- |
| `GUARDRAIL_BASE_URL` | Yes | NeMo Guardrails Route base URL (no path suffix). Example: `https://nemo-guardrails-demo.<cluster-domain>` |
| `SELF_CHECK_LLM_URL` | Yes | Base URL of the model used for NeMo `self_check` rails. |
| `SELF_CHECK_LLM_NAME` | Yes | Model id for the  `SELF_CHECK_LLM_URL` endpoint. |

Example (after Step 2):

```bash
export GUARDRAIL_BASE_URL="https://${GUARDRAIL_HOST}"

# Match the GUARDRAIL_LLM_* values used in make setup-guardrail
export SELF_CHECK_LLM_URL='https://<direct-self-check-endpoint>'   # or in-cluster service URL
export SELF_CHECK_LLM_NAME='<self-check-model-id>'

./scripts/test-guardrail.sh "gpt-oss-20b" "What is the bank rate?"
```

The script runs three checks (in order):

1. **`POST ${GUARDRAIL_BASE_URL}/v1/guardrail/checks`** — input rails only (same API shape IPP uses, plus optional `guardrails.options` for logging).
2. **`POST ${GUARDRAIL_BASE_URL}/v1/chat/completions`** — full NeMo chat path through guardrails (generation + output rails in the demo config).
3. **`POST ${SELF_CHECK_LLM_URL}/v1/chat/completions`** — connectivity smoke test to the self-check model. This sends your **user prompt only**; it does **not** run NeMo’s `self_check_input` template.

Dependencies: `curl`, `jq`.

This verifies NeMo and its backing LLMs **without** MaaS or IPP.

### Environment variables for `test-maas.sh`

| Variable | Required | Purpose |
| -------- | -------- | ------- |
| `MAAS_BASE_URL` | Yes | MaaS Route base URL (same host clients use for OpenAI-compatible APIs) |
| `MAAS_TOKEN` | Yes | Bearer token for `Authorization: Bearer …` |

The **model** and **prompt** are the two script arguments.

Example:

```bash
export MAAS_BASE_URL='https://<maas-route-host>'
export MAAS_TOKEN='<bearer-token>'

./scripts/test-maas.sh "gpt-oss-20b" "What is the bank rate?"
```

The script:

1. Calls **`GET ${MAAS_BASE_URL}/v1/models`** and lists model ids.
2. Calls **`POST ${MAAS_BASE_URL}/v1/chat/completions`** with a single user message (`max_tokens: 1024`).

Run this only **after** `scripts/enable-ipp-nemo.py` (and after upgrading `payload-processing` per the disclaimer above) so the request flows through IPP and NeMo input (and output, if enabled) guardrails.

Dependencies: `curl`, `jq`.

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

---

## Reference

| Path                           | Purpose                                                              |
| ------------------------------ | -------------------------------------------------------------------- |
| `Makefile` (`setup-guardrail`) | Creates the Secret and ConfigMap and applies the `NemoGuardrails` CR |
| `yaml/demo/nemo-cm.yaml.tmpl`  | NeMo Guardrails configuration and `OPENAI_*` / `GUARDRAIL_LLM_*` substitution |
| `yaml/demo/nemo-cr.yaml`       | `NemoGuardrails` instance                                            |
| `scripts/enable-ipp-nemo.py`   | Enables NeMo plugins in MaaS IPP                                     |
| `scripts/test-guardrail.sh`    | Direct NeMo tests: `/v1/guardrail/checks`, `/v1/chat/completions`, self-check LLM |
| `scripts/test-maas.sh`         | MaaS E2E: `/v1/models` and `/v1/chat/completions` via IPP guardrails |

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
| `unknown NeMo guardrails status "success"` (IPP 500) | NeMo allowed the request; **upgrade `payload-processing`** to an image with [PR #434](https://github.com/opendatahub-io/ai-gateway-payload-processing/pull/434). Pre-#434 plugins expect top-level `passed`, not TrustyAI `success`. |
| NeMo `/v1/guardrail/checks` returns `success` but MaaS fails | Same as above — not a NeMo config bug. Check `payload-processing` image age vs #434. |
| 403 on “Say hello…” but banking prompts work in `test-guardrail.sh` | Expected with the demo gatekeeper: small talk is blocked. Use an on-topic prompt in `scripts/test-maas.sh` for a happy-path E2E test. |