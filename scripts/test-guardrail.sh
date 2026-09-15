#!/bin/sh

set -eu

usage() {
    echo "Usage: $0 <model-name> <prompt>" >&2
    echo "Example: $0 \"gpt-oss-20b\" \"What is the bank rate?\"" >&2
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

if [ -z "${SELF_CHECK_LLM_URL:-}" ]; then
    echo "❌ SELF_CHECK_LLM_URL is not set or empty"
    exit 1
fi

if [ -z "${SELF_CHECK_LLM_NAME:-}" ]; then
    echo "❌ SELF_CHECK_LLM_NAME is not set or empty"
    exit 1
fi

echo "✅ GUARDRAIL_BASE_URL=$GUARDRAIL_BASE_URL"
echo "✅ SELF_CHECK_LLM_URL=$SELF_CHECK_LLM_URL"
echo "✅ SELF_CHECK_LLM_NAME=$SELF_CHECK_LLM_NAME"
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
SELF_CHECK_LLM_CHAT_URL="${SELF_CHECK_LLM_URL%/}/v1/chat/completions"

# Request sent through NeMo Guardrails.
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
        ],
        guardrails: {
            options: {
                llm_output: true,
                output_vars: true,
                log: {
                    activated_rails: true,
                    llm_calls: true,
                    internal_events: false,
                    colang_history: false
                }
            }
        }
    }')

# Direct request to the LLM configured for NeMo self-check.
#
# This is intentionally just a basic endpoint/model test. It does NOT
# reproduce NeMo Guardrails self_check_input because NeMo supplies its
# own self_check_input prompt.
SELF_CHECK_LLM_PAYLOAD=$(jq -n \
    --arg model "$SELF_CHECK_LLM_NAME" \
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
SELF_CHECK_LLM_FAILED=0

test_chat_completion() {
    label="$1"
    url="$2"
    payload="$3"

    echo "=== Testing $label ==="
    echo "URL: $url"
    echo

    echo "Payload (prompt omitted):"

    printf '%s' "$payload" | \
        jq '.messages[]?.content = "<prompt omitted>"'

    echo

    : > "$response_file"

    if ! http_status=$(curl -k -sS \
        -o "$response_file" \
        -w "%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$url"); then

        echo "❌ $label request failed: $url"
        return 1
    fi

    response=$(cat "$response_file")

    echo "HTTP status: $http_status"

    if [ "$http_status" -eq 200 ]; then
        echo "✅ $label HTTP request succeeded"
    else
        echo "❌ $label HTTP request returned HTTP $http_status"
        echo
        echo "Response:"
        printf '%s\n' "$response"
        return 1
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
        return 1
    fi

    echo
    echo "Assistant response:"

    if assistant=$(printf '%s' "$response" | \
        jq -er '.choices[0].message.content' 2>&1); then
        printf '%s\n' "$assistant"
    else
        echo "❌ Could not extract .choices[0].message.content"
        printf '%s\n' "$assistant"
        return 1
    fi

    return 0
}

test_self_check_llm() {
    label="$1"
    url="$2"
    payload="$3"

    echo "=== Testing $label ==="
    echo "URL: $url"
    echo

    echo "Payload (prompt omitted):"

    printf '%s' "$payload" | \
        jq '.messages[]?.content = "<prompt omitted>"'

    echo

    : > "$response_file"

    if ! http_status=$(curl -k -sS \
        -o "$response_file" \
        -w "%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$url"); then

        echo "❌ $label request failed: $url"
        return 1
    fi

    response=$(cat "$response_file")

    echo "HTTP status: $http_status"

    if [ "$http_status" -eq 200 ]; then
        echo "✅ $label HTTP request succeeded"
    else
        echo "❌ $label HTTP request returned HTTP $http_status"
        echo
        echo "Response:"
        printf '%s\n' "$response"
        return 1
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
        return 1
    fi

    echo
    echo "Self-check LLM response:"

    # The self-check LLM is normally expected to return a verdict when
    # invoked by NeMo with its self_check_input prompt. This direct test
    # does not send that prompt, so we only display whatever the model
    # actually returned.
    self_check_response=$(printf '%s' "$response" | jq -r '
        if ((.choices[0].message.content // "") | length) > 0 then
            .choices[0].message.content
        elif ((.choices[0].message.reasoning_content // "") | length) > 0 then
            .choices[0].message.reasoning_content
        elif ((.choices[0].text // "") | length) > 0 then
            .choices[0].text
        else
            "<no response content found>"
        end
    ')

    printf '%s\n' "$self_check_response"

    return 0
}

echo "=== Testing NeMo Guardrails API: /v1/guardrail/checks ==="
echo "URL: $GUARDRAIL_URL"
echo

: > "$response_file"

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
        GUARDRAIL_FAILED=1
    fi
fi

echo

if ! test_chat_completion \
    "NeMo Guardrails API: /v1/chat/completions" \
    "$CHAT_URL" \
    "$PAYLOAD"; then
    CHAT_FAILED=1
fi

echo

if ! test_self_check_llm \
    "NeMo LLM Self-Check Model API: /v1/chat/completions" \
    "$SELF_CHECK_LLM_CHAT_URL" \
    "$SELF_CHECK_LLM_PAYLOAD"; then
    SELF_CHECK_LLM_FAILED=1
fi

echo
echo "=== Test Summary ==="

if [ "$GUARDRAIL_FAILED" -eq 0 ]; then
    echo "✅ Guardrail check: HTTP 200"
else
    echo "❌ Guardrail check: FAILED"
fi

if [ "$CHAT_FAILED" -eq 0 ]; then
    echo "✅ Guardrail chat completion: HTTP 200"
else
    echo "❌ Guardrail chat completion: FAILED"
fi

if [ "$SELF_CHECK_LLM_FAILED" -eq 0 ]; then
    echo "✅ NeMo LLM self-check model chat completion: HTTP 200"
else
    echo "❌ NeMo LLM self-check model chat completion: FAILED"
fi

if [ "$GUARDRAIL_FAILED" -ne 0 ] ||
   [ "$CHAT_FAILED" -ne 0 ] ||
   [ "$SELF_CHECK_LLM_FAILED" -ne 0 ]; then
    echo
    echo "❌ One or more API checks failed"
    exit 1
fi

echo
echo "✅ All API checks passed"