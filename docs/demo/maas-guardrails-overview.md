# MaaS and NeMo Guardrails

Full procedure: [Enable NeMo Guardrails in MaaS](../maas-guardrails.md).

> **Before you test end-to-end:** TrustyAI NeMo returns `status: success` on allow. Older `payload-processing` images (before [ai-gateway-payload-processing PR #434](https://github.com/opendatahub-io/ai-gateway-payload-processing/pull/434)) treat that as an error and MaaS returns **500** (`unknown NeMo guardrails status "success"`). See the [full disclaimer in maas-guardrails.md](../maas-guardrails.md#important-payload-processing-image-version-trustyai-v1guardrailchecks).

NeMo Guardrails is **not** installed by `make setup-demo`. Deploy and wire it with:

```bash
export OPENAI_BASE_URL="..."          # direct model endpoint, not MaaS
export OPENAI_MODEL_NAME="..."
export OPENAI_API_KEY="..."
export GUARDRAIL_LLM_BASE_URL="..."   # direct self-check model endpoint, not MaaS
export GUARDRAIL_LLM_MODEL_NAME="..."
make setup-guardrail
```

Then configure MaaS IPP with `scripts/enable-ipp-nemo.py`.

**Test scripts** (see [Step 4](../maas-guardrails.md#step-4-test-the-configuration) in the full guide):

```bash
# Direct NeMo (env vars + model + prompt)
export GUARDRAIL_BASE_URL="https://<nemo-route-host>"
export SELF_CHECK_LLM_URL="..."      # same idea as GUARDRAIL_LLM_BASE_URL
export SELF_CHECK_LLM_NAME="..."       # same as GUARDRAIL_LLM_MODEL_NAME
./scripts/test-guardrail.sh "<model-id>" "What is the bank rate?"

# Through MaaS / IPP (env vars + model + prompt)
export MAAS_BASE_URL="https://<maas-route-host>"
export MAAS_TOKEN="<token>"
./scripts/test-maas.sh "<model-id>" "What is the bank rate?"
```

Use the OpenShift **Route** in `demo` for `GUARDRAIL_BASE_URL` (see the linked guide).
