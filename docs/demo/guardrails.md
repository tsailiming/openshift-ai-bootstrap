# NeMo Guardrails

NeMo Guardrails adds a safety layer to your LLM deployments. It intercepts requests and responses to enforce policies—blocking off-topic prompts, masking sensitive data (PII), validating tool calls, and filtering unsafe outputs before they reach users.

## Prerequisites

- Complete [setup-rhoai](../../README.md#step-2--run-the-platform-installer)
- A deployed model endpoint (InferenceService or external)
- A self-check model for guardrail evaluation

## Environment variables

The `setup-guardrail` target substitutes these variables into the NeMo ConfigMap (`yaml/demo/nemo-cm.yaml.tmpl`):

### Main model (handles user queries)

These configure the primary LLM that processes user requests:

```yaml
# From nemo-cm.yaml.tmpl
models:
  - type: main
    engine: openai
    model: "${OPENAI_MODEL_NAME}"
    api_key_env_var: OPENAI_API_KEY
    parameters:
      base_url: "${OPENAI_BASE_URL}"
```

| Variable | Description |
| :--- | :--- |
| `OPENAI_BASE_URL` | Base URL of the main model endpoint (e.g., `https://<isvc-host>/v1`) |
| `OPENAI_MODEL_NAME` | Model ID served at that endpoint |
| `OPENAI_API_KEY` | API key for authentication |

### Self-check model (evaluates guardrail rules)

NeMo uses a separate LLM to evaluate whether inputs/outputs pass the guardrail rules (e.g., the `self_check_input` prompt that decides if a query should be blocked):

```yaml
# From nemo-cm.yaml.tmpl
models:
  - type: self_check
    engine: openai
    model: "${GUARDRAIL_LLM_MODEL_NAME}"
    parameters:
      base_url: "${GUARDRAIL_LLM_BASE_URL}"
```

| Variable | Description |
| :--- | :--- |
| `GUARDRAIL_LLM_BASE_URL` | Base URL for the self-check LLM |
| `GUARDRAIL_LLM_MODEL_NAME` | Model ID for guardrail evaluation |

> **Tip:** The self-check model can be the same as the main model, or a smaller/faster model optimized for classification tasks.

## Deploy NeMo Guardrails

```bash
# Set environment variables
export OPENAI_BASE_URL="https://<model-endpoint>/v1"
export OPENAI_MODEL_NAME="<model-name>"
export OPENAI_API_KEY="<api-key>"
export GUARDRAIL_LLM_BASE_URL="https://<self-check-endpoint>/v1"
export GUARDRAIL_LLM_MODEL_NAME="<self-check-model>"

# Deploy guardrails
make setup-guardrail
```

This deploys:
- NeMo Guardrails ConfigMap with your model configuration
- NeMo Guardrails custom resource in the `demo` namespace

## Test guardrails

The `test-guardrail.sh` script tests the guardrail endpoints directly.

### Required environment variables for testing

```bash
export GUARDRAIL_BASE_URL="https://<guardrail-route>/v1"
export SELF_CHECK_LLM_URL="https://<self-check-endpoint>/v1"
export SELF_CHECK_LLM_NAME="<self-check-model>"
```

### Run test

```bash
scripts/test-guardrail.sh "<model-name>" "<prompt>"

# Example
scripts/test-guardrail.sh "gpt-oss-20b" "What is the bank rate?"
```

## What the default configuration does

The included `nemo-cm.yaml.tmpl` configures:

**Input rails:**
- `mask sensitive data on input` — Detects and masks PII (credit cards, SSN, bank account numbers, etc.)
- `self check input` — Uses the self-check LLM to block off-topic queries (small talk, recipes, entertainment)

**Output rails:**
- `append boston weather` — Example rail that appends weather data to responses

**Tool output rails:**
- `validate tool choice` — Only allows specific tools (`get_account_balance`, `get_transaction_history`, `get_exchange_rate`)
- `check valid account` — Validates account numbers in tool calls

## Customization

Edit `yaml/demo/nemo-cm.yaml.tmpl` to customize:

- Allowed/blocked topics in `prompts.yml`
- PII entity types to mask
- Tool validation logic in `actions.py`
- Custom rails in `rails.co`

See the [NeMo Guardrails documentation](https://docs.nvidia.com/nemo/guardrails/) for configuration options.
