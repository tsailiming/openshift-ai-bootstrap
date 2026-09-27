# Make targets

All targets are defined in the repository `Makefile`. Run from the repo root with a logged-in `oc` session (`cluster-admin` recommended).

Variables you may override:

| Variable | Default | Meaning |
| :---- | :---- | :---- |
| `NAMESPACE` | `demo` | Demo project for workloads |
| `EVALHUB_NAMESPACE` | `evalhub` | EvalHub tenant namespace |
| `RHAIIS_IMAGE` | `registry.redhat.io/rhaii-early-access/vllm-cuda-rhel9:3.5.0-ea.2` | vLLM ServingRuntime image |
| `RHAIIS_VLLM_VERSION` | `0.19.1` | vLLM version stamped in the ServingRuntime template |

## Core platform

| Target | Description |
| :---- | :---- |
| `setup-rhoai` | Full RHOAI install: GPU operator, NFS, Kueue/LWS/JobSet, COO/Tempo/OTel, Kuadrant, Agent Sandbox, Pipelines, DSC/DSCI, dashboard config, MLflow, EvalHub, Grafana, ServingRuntime template, hardware profiles |
| `rhoai-prereq` | Operators only (subset used by `setup-rhoai`) |
| `add-gpu-operator` | NFD + NVIDIA GPU Operator |
| `add-nfs-provisioner` | NFS subdir provisioner for RWX volumes |

## Demo namespace

| Target | Description |
| :---- | :---- |
| `setup-demo` | `setup-namespace` + SeaweedFS + ODH-TEC + pipelines + model PVC, GuideLLM, benchmark arena, ai-toolkit, Open WebUI, custom model catalog, EvalHub RBAC |
| `setup-namespace` | Creates `demo` and `evalhub` projects with required labels |
| `deploy-seaweedfs` / `teardown-seaweedfs` | SeaweedFS S3 storage via Helm chart and data connection (pipelines) |
| `setup-odh-tec` / `show-odh-tec` / `teardown-odh-tec` | Object storage browser (ODH-TEC) |
| `deploy-pipeline` / `teardown-pipeline` | Data Science Pipelines (DSPA) |
| `teardown-namespace` / `teardown-all` | Remove demo resources |

## Models and playground

| Target | Description |
| :---- | :---- |
| `download-and-serve-models` | Example: Qwen3.5-27B-FP8 and gpt-oss-20b to PVC + InferenceService |
| `setup-ai-playground` | MCP servers, Llama Stack CM, gen AI playground deployment |
| `setup-guardrail` | NeMo Guardrails in `demo` (requires `OPENAI_BASE_URL`, `OPENAI_MODEL_NAME`, `OPENAI_API_KEY`, `GUARDRAIL_LLM_BASE_URL`, `GUARDRAIL_LLM_MODEL_NAME`) |

## MaaS, MCP, sandbox

| Target | Description |
| :---- | :---- |
| `setup-maas` | Enables MaaS in DSC, runs [rhoai-maas-guide](https://github.com/rh-aiservices-bu/rhoai-maas-guide) `setup-maas.sh`, patches tenant telemetry and sample subscriptions |
| `setup-mcp-gateway` | MCP gateway extension + sample OCP MCP server in `demo` |
| `setup-osc` | Sandboxed Containers (Kata) + optional metal worker via `clone-machineset.sh` |
| `setup-openshell` | OpenShell on Kata (clones agent-ops, Helm upgrade) |

## Other

| Target | Description |
| :---- | :---- |
| `setup-kueue-demo` | Applies external Kueue demo manifests |
| `setup-multi-user` | HTPasswd users `user1` / `user2` with separate projects |
| `restart` | Restarts EvalHub and RHOAI dashboard deployments |

## Typical sequences

**Inference lab only**

```bash
make setup-rhoai
make setup-demo
make download-and-serve-models   # optional samples
```

**MaaS + guardrails**

```bash
make setup-rhoai
make setup-maas
make setup-guardrail             # set OPENAI_* first
# then scripts/enable-ipp-nemo.py — see docs/maas-guardrails.md
```

**Gen AI playground**

```bash
make setup-rhoai
make setup-demo
make setup-ai-playground
```
