# OpenShift AI Bootstrap Guide

This guide walks you through setting up **Red Hat OpenShift AI (RHOAI)** on OpenShift and exploring its capabilities: GPU-accelerated model serving with vLLM, benchmarking, Models-as-a-Service (MaaS), guardrails, MCP integration, and more.

## Changelog

| Date | Description |
| :--- | :--- |
| 27 Sep 2026 | Reorganized as step-by-step guide |
| 31 Aug 2026 | Based on RHOAI 3.5 and OpenShift 4.21 |
| 26 Mar 2026 | Based on RHOAI 3.4 EA1 |
| 4 Feb 2026 | Based on RHOAI 3.2 |
| 23 Nov 2025 | Based on RHOAI 3.0 |
| 30 Aug 2025 | Based on RHOAI 2.23 |

## What you will learn

By following this guide, you will:

1. **Set up OpenShift AI** — Install RHOAI with GPU operators, observability, and supporting infrastructure
2. **Serve LLM models** — Deploy models from Hugging Face or OCI registries using vLLM and KServe
3. **Benchmark inference** — Measure throughput and latency with GuideLLM and compare models side-by-side
4. **Build AI applications** — Use Open WebUI, Llama Stack, and MCP servers to create interactive AI experiences
5. **Enable Models-as-a-Service** — Deploy models to MaaS for multi-tenant access with usage quotas

---

## Before you begin

> **Disclaimer**
>
> 1. These examples showcase OpenShift AI capabilities — they are not a substitute for official product documentation.
> 2. Some components use upstream projects and may not be covered by Red Hat support.
> 3. Confirm supportability with Red Hat or your account team before production use.

### Prerequisites

You will need:

- An OpenShift cluster with `cluster-admin` access
- NVIDIA GPUs attached to worker nodes (see [GPU requirements](docs/requirements.md#nvidia-gpu))
- CLI tools: `oc`, `make`, `envsubst`, and `helm` (for optional targets)
- Registry access to `registry.redhat.io` and `quay.io`

Review the full [Requirements](docs/requirements.md) for cluster sizing, GPU compatibility, and storage.

---

## Part 1: Install OpenShift AI

### Step 1 — Clone the repository

```bash
git clone https://github.com/tsailiming/openshift-ai-bootstrap.git
cd openshift-ai-bootstrap
```

### Step 2 — Run the platform installer

The `setup-rhoai` target installs and configures the complete OpenShift AI stack:

| Component | What it does |
| :--- | :--- |
| NFD + NVIDIA GPU Operator | Detects and enables GPU resources on worker nodes |
| NFS provisioner | Provides RWX storage for model sharing |
| Kueue, LWS, JobSet | Enables workload scheduling and distributed training |
| COO, Tempo, OTel | Adds observability, tracing, and metrics |
| Kuadrant | Installs Red Hat Connectivity Link for API management |
| Agent Sandbox, Pipelines | Enables sandboxed execution and ML pipelines |
| RHOAI operator | Deploys DataScienceCluster with dashboard and model serving |
| MLflow, EvalHub, Grafana | Adds experiment tracking, evaluation, and dashboards |

Run the installer:

```bash
make setup-rhoai
```

**Note:** This takes 10–20 minutes depending on image pulls. The Makefile waits for `DSCInitialization` and `DataScienceCluster` to reach `Ready` status.

### Step 3 — Verify the installation

Once complete, confirm the dashboard is accessible:

```bash
oc get route rhods-dashboard -n redhat-ods-applications -o jsonpath='{.spec.host}'
```

Open the URL in your browser to access the OpenShift AI console.

---

## Part 2: Set up the demo environment

The demo environment provides tools and sample workloads for exploring model serving.

### Step 4 — Deploy the demo namespace

```bash
make setup-demo
oc project demo
```

This creates the `demo` project with:

- **SeaweedFS** — S3-compatible object storage for pipelines and artifacts
- **ODH-TEC** — Browser-based storage explorer
- **Data Science Pipelines** — ML workflow orchestration
- **Model PVC** — Shared storage for downloaded models
- **Open WebUI** — Chat interface for interacting with models
- **GuideLLM + Benchmark Arena** — Performance testing tools
- **Custom model catalog** — Pre-configured model sources

---

## Part 3: Serve your first model

Now you're ready to deploy a model.

### Step 5 — Download a model to the PVC

```bash
scripts/download-model.sh pvc Qwen/Qwen2.5-7B-Instruct
```

For gated models requiring authentication:

```bash
HF_TOKEN=<your-token> scripts/download-model.sh pvc meta-llama/Llama-3.1-8B-Instruct
```

### Step 6 — Deploy the model as an InferenceService

```bash
scripts/serve-model.sh qwen25-7b-instruct \
  pvc://models-pvc/Qwen/Qwen2.5-7B-Instruct \
  --vllm-args "--max-model-len 4096"
```

Or deploy directly from an OCI registry (no download needed):

```bash
scripts/serve-model.sh qwen25-7b-fp8 \
  oci://registry.redhat.io/rhelai1/modelcar-qwen2-5-7b-instruct-fp8-dynamic:1.5
```

### Step 7 — Test the model endpoint

```bash
# Get the model URL
MODEL_URL=$(oc get isvc qwen25-7b-instruct -o jsonpath='{.status.url}')

# Send a test request
curl -s "$MODEL_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen25-7b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }' | jq .
```

For the complete model serving walkthrough, see [Bring your own model](docs/demo/bring-your-own-model.md).

---

## Part 4: Explore more capabilities

With the platform running, explore these topics based on your interests:

### Model serving and optimization

| Topic | Description | Guide |
| :--- | :--- | :--- |
| Model catalog | Browse and deploy models from the dashboard | [model-catalog.md](docs/demo/model-catalog.md) |
| Compressed models | Deploy FP8 quantized models for better performance | [compressed-models.md](docs/demo/compressed-models.md) |
| Multi-GPU serving | Scale models across multiple GPUs with tensor parallelism | [multi-gpu.md](docs/demo/multi-gpu.md) |
| Context length tuning | Adjust KV cache and sequence limits | [context-length.md](docs/demo/context-length.md) |
| Distributed inference | Deploy with llm-d for high-throughput serving | [llm-d.md](docs/demo/llm-d.md) |

### Benchmarking and evaluation

| Topic | Description | Guide |
| :--- | :--- | :--- |
| GuideLLM benchmarking | Measure throughput, latency, and token rates | [benchmarking.md](docs/demo/benchmarking.md) |
| Benchmark arena | Compare models side-by-side | [benchmark-arena.md](docs/demo/benchmark-arena.md) |
| Observability | Monitor with Grafana and user workload metrics | [observability.md](docs/demo/observability.md) |

### Applications and integrations

| Topic | Description | Guide |
| :--- | :--- | :--- |
| Open WebUI | Chat interface with model switching | [open-webui-integration.md](docs/demo/open-webui-integration.md) |
| Whisper STT | Speech-to-text transcription | [whisper.md](docs/demo/whisper.md) |
| AI playground | Llama Stack, MCP servers, and RAG | [ai-playground.md](docs/demo/ai-playground.md) |
| LLM Compressor | Quantize models in a workbench | [llm-compressor.md](docs/demo/llm-compressor.md) |

### Enterprise features

| Topic | Description | Guide |
| :--- | :--- | :--- |
| Models-as-a-Service | Multi-tenant model serving with quotas | [maas.md](docs/demo/maas.md) |
| NeMo Guardrails | Policy enforcement, PII masking, and content filtering | [guardrails.md](docs/demo/guardrails.md) |

---

## Optional: Additional setup targets

These targets enable specific features not included in the base installation:

| Target | Description |
| :--- | :--- |
| `setup-maas` | Enable Models-as-a-Service with multi-tenant quotas |
| `setup-mcp-gateway` | Deploy MCP gateway for tool integrations |
| `setup-ai-playground` | Set up AI playground with Llama Stack and MCP servers |
| `download-and-serve-models` | Deploy sample models (Qwen, GPT-OSS) |
| `setup-guardrail` | Configure NeMo Guardrails (experimental, requires `OPENAI_*` and `GUARDRAIL_LLM_*` env vars) |
| `setup-osc` | Install OpenShift Sandboxed Containers (Kata) |
| `setup-openshell` | Deploy OpenShell on Kata for secure execution |
| `setup-kueue-demo` | Apply external Kueue demo manifests |
| `setup-multi-user` | Create HTPasswd users with separate projects |

See [Make targets](docs/make-targets.md) for the complete reference.

---

## Helper scripts

The `scripts/` directory contains utilities you can run directly:

| Script | Description |
| :--- | :--- |
| `download-model.sh` | Download models from Hugging Face to PVC. Usage: `scripts/download-model.sh pvc <model-id>` |
| `serve-model.sh` | Deploy a model as an InferenceService or to MaaS. Usage: `scripts/serve-model.sh <name> <uri>` |
| `update-model-open-webui.sh` | Register an InferenceService endpoint in Open WebUI. Usage: `scripts/update-model-open-webui.sh <isvc-name>` |
| `clone-machineset.sh` | Clone a MachineSet to a GPU instance type (AWS). Usage: `scripts/clone-machineset.sh <instance-type>` |
| `test-mcp-server.sh` | Verify MCP server connectivity. Usage: `scripts/test-mcp-server.sh <namespace> <server-name>` |
| `test-guardrail.sh` | Test NeMo guardrail endpoints. Usage: `scripts/test-guardrail.sh <model> <prompt>` |
| `test-maas.sh` | Test MaaS endpoints end-to-end. Usage: `scripts/test-maas.sh <model> <prompt>` |
| `evalhub-cli.py` | CLI wrapper for EvalHub SDK (requires `uv`) |
| `clean-terminating-pods.sh` | Force-delete Terminating pods on a node. Usage: `scripts/clean-terminating-pods.sh <node-name>` |

---

## Disconnected environments

If your cluster cannot reach public registries or Hugging Face, follow the [Disconnected environments](docs/disconnected.md) guide after completing Parts 1 and 2.

---

## Appendix

- [Appendix](docs/appendix.md) — Open WebUI configuration tips, network policies, AWS GPU notes
