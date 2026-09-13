#!/bin/sh

set -eu

usage() {
    echo "Usage: $0 <model-name> <prompt>" >&2
    echo "Example: $0 test \"What is the capital of France?\"" >&2
    exit 1
}

[ "$#" -eq 2 ] || usage

MODEL_NAME="$1"
PROMPT="$2"

echo "=== Checking environment variables ==="

if [ -z "${GUARDRAIL_BASE_URL:-}" ]; then
    echo "❌ GUARDRAIL_BASE_URL is not set or empty"
    exit 1
fi

echo "✅ GUARDRAIL_BASE_URL=$GUARDRAIL_BASE_URL"
echo "✅ MODEL_NAME=$MODEL_NAME"
echo "✅ PROMPT=$PROMPT"
echo

echo "=== Checking dependencies ==="

if ! command -v curl >/dev/null 2>&1; then
    echo "❌ curl is not installed"
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "❌ jq is not installed"
    exit 1
fi

echo "✅ curl is installed"
echo "✅ jq is installed"
echo

echo "=== Testing NeMo Guardrails API: /v1/guardrail/checks ==="

GUARDRAIL_URL="${GUARDRAIL_BASE_URL%/}/v1/guardrail/checks"

echo "URL: $GUARDRAIL_URL"
echo

PAYLOAD=$(jq -n \
    --arg model "$MODEL_NAME" \
    --arg prompt "$PROMPT" \
    '{
        model: $model,
        messages: [
            {
                role: "user",
                content: $prompt
            }
        ]
    }')

if ! response=$(curl -k -fsS \
    -X POST \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    "$GUARDRAIL_URL"); then

    echo "❌ Guardrail request failed: $GUARDRAIL_URL"
    exit 1
fi

echo "✅ Guardrail request succeeded"
echo

echo "Response:"
echo "$response" | jq .
