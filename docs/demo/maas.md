# Models-as-a-Service (MaaS)

MaaS provides multi-tenant model serving with usage quotas, API key management, and centralized model access.

## Prerequisites

- Complete [setup-rhoai](../../README.md#step-2--run-the-platform-installer)
- GPU worker nodes available

## Enable MaaS

Run the setup target:

```bash
make setup-maas
```

This target:
- Enables `modelsAsAService` in the DataScienceCluster
- Clones and runs the [rhoai-maas-guide](https://github.com/rh-aiservices-bu/rhoai-maas-guide) setup script
- Configures tenant telemetry for usage tracking

## Deploy a model to MaaS

Use `serve-model.sh` with `--target maas` to deploy models to the MaaS gateway:

### From OCI registry

```bash
scripts/serve-model.sh \
  --target maas \
  gpt-oss-20b \
  oci://registry.redhat.io/rhelai1/modelcar-gpt-oss-20b:1.5 \
  --vllm-args "--max-model-len 4096"
```

### From PVC

First download the model:

```bash
scripts/download-model.sh pvc Qwen/Qwen3.5-27B-FP8
```

Then deploy to MaaS:

```bash
scripts/serve-model.sh \
  --target maas \
  qwen35-27b-fp8 \
  pvc://models-pvc/Qwen/Qwen3.5-27B-FP8 \
  --vllm-args "--max-model-len 2048 --gpu-memory-utilization 0.97"
```

## Test the MaaS endpoint

### Using test script

```bash
scripts/test-maas.sh gpt-oss-20b "Hello, how are you?"
```

### Using curl

```bash
# Get the MaaS route
MAAS_URL=$(oc get route maas -n models-as-a-service -o jsonpath='{.spec.host}')

# List models
curl -s "https://$MAAS_URL/v1/models" \
  -H "Authorization: Bearer <your-api-key>" | jq .

# Test chat completion
curl -s "https://$MAAS_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-api-key>" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }' | jq .
```

## serve-model.sh options for MaaS

| Option | Description | Default |
| :--- | :--- | :--- |
| `--target maas` | Deploy to MaaS instead of InferenceService | `isvc` |
| `--namespace NS` | Kubernetes namespace | `demo` |
| `--replicas N` | Number of model replicas | `1` |
| `--vllm-args "..."` | Additional vLLM arguments | none |

## Notes

- MaaS models are accessible via a centralized gateway route
- API keys are managed through MaaS subscriptions
- Usage metrics are captured when telemetry is enabled

> **NeMo Guardrails:** Integration with MaaS guardrails is experimental and not fully functional. See [maas-guardrails.md](../maas-guardrails.md) for details.
