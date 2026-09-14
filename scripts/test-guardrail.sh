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

GUARDRAIL_URL="${GUARDRAIL_BASE_URL%/}/v1/guardrail/checks"
CHAT_URL="${GUARDRAIL_BASE_URL%/}/v1/chat/completions"

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

response_file=$(mktemp)
trap 'rm -f "$response_file"' EXIT

GUARDRAIL_FAILED=0
CHAT_FAILED=0

# --------------------------------------------------------------------
# Guardrail check
# --------------------------------------------------------------------

echo "=== Testing NeMo Guardrails API: /v1/guardrail/checks ==="

echo "URL: $GUARDRAIL_URL"
echo

if ! http_status=$(curl -k -sS \
    -o "$response_file" \
    -w "%{http_code}" \
    -X POST \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    "$GUARDRAIL_URL"); then

    echo "❌ Guardrail request failed: $GUARDRAIL_URL"
    GUARDRAIL_FAILED=1
else
    response=$(cat "$response_file")

    echo "HTTP status: $http_status"

    if [ "$http_status" -eq 200 ]; then
        echo "✅ Guardrail HTTP request succeeded"
    else
        echo "❌ Guardrail HTTP request returned HTTP $http_status"
        GUARDRAIL_FAILED=1
    fi

    echo
    echo "Response:"

    if formatted=$(printf '%s' "$response" | jq . 2>&1); then
        printf '%s\n' "$formatted"

        echo
        echo "Guardrail status:"

        guardrail_status=$(printf '%s' "$response" | \
            jq -r '.rails_status["self check input"].status // "unknown"')

        case "$guardrail_status" in
            success)
                echo "🟢 self check input: SUCCESS"
                ;;
            *)
                echo "🔴 self check input: $guardrail_status"
                ;;
        esac
    else
        echo "❌ Response is NOT valid JSON"
        echo
        printf '%s\n' "$formatted"
        echo
        echo "Raw response:"
        printf '%s\n' "$response"
    fi
fi

echo

# --------------------------------------------------------------------
# Chat completion
# --------------------------------------------------------------------

echo "=== Testing NeMo Guardrails API: /v1/chat/completions ==="

echo "URL: $CHAT_URL"
echo

: > "$response_file"

if ! http_status=$(curl -k -sS \
    -o "$response_file" \
    -w "%{http_code}" \
    -X POST \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    "$CHAT_URL"); then

    echo "❌ Chat completion request failed: $CHAT_URL"
    CHAT_FAILED=1
else
    response=$(cat "$response_file")

    echo "HTTP status: $http_status"

    if [ "$http_status" -eq 200 ]; then
        echo "✅ Chat completion HTTP request succeeded"
    else
        echo "❌ Chat completion HTTP request returned HTTP $http_status"
        CHAT_FAILED=1
    fi

    echo
    echo "Response:"

    if formatted=$(printf '%s' "$response" | jq . 2>&1); then
        printf '%s\n' "$formatted"
    else
        echo "❌ Response is NOT valid JSON"
        echo
        printf '%s\n' "$formatted"
        echo
        echo "Raw response:"
        printf '%s\n' "$response"
        CHAT_FAILED=1
    fi

    echo
    echo "Assistant response:"

    if assistant=$(printf '%s' "$response" | jq -er '.choices[0].message.content' 2>&1); then
        printf '%s\n' "$assistant"
    else
        echo "❌ Could not extract .choices[0].message.content"
        printf '%s\n' "$assistant"
        CHAT_FAILED=1
    fi
fi

echo
echo "=== Test Summary ==="

if [ "$GUARDRAIL_FAILED" -eq 0 ]; then
    echo "✅ Guardrail check: HTTP 200"
else
    echo "❌ Guardrail check: FAILED"
fi

if [ "$CHAT_FAILED" -eq 0 ]; then
    echo "✅ Chat completion: HTTP 200"
else
    echo "❌ Chat completion: FAILED"
fi

if [ "$GUARDRAIL_FAILED" -ne 0 ] || [ "$CHAT_FAILED" -ne 0 ]; then
    echo
    echo "❌ One or more API checks failed"
    exit 1
fi

echo
echo "✅ All API checks passed"