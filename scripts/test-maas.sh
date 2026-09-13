#!/bin/sh

echo "=== Checking environment variables ==="

ERROR=0

[ -z "${MAAS_TOKEN:-}" ] && {
    echo "❌ MAAS_TOKEN is not set or empty"
    ERROR=1
}

[ -z "${MAAS_MODEL_NAME:-}" ] && {
    echo "❌ MAAS_MODEL_NAME is not set or empty"
    ERROR=1
}

[ -z "${MAAS_BASE_URL:-}" ] && {
    echo "❌ MAAS_BASE_URL is not set or empty"
    ERROR=1
}

if [ "$ERROR" -ne 0 ]; then
    exit 1
fi

echo "✅ MAAS_TOKEN is set"
echo "✅ MAAS_MODEL_NAME=$MAAS_MODEL_NAME"
echo "✅ MAAS_BASE_URL=$MAAS_BASE_URL"
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

echo "=== Testing MAAS API: /v1/models ==="

URL="${MAAS_BASE_URL%/}/v1/models"

response=$(curl -sS \
    -w '\n%{http_code}' \
    -H "Authorization: Bearer $MAAS_TOKEN" \
    "$URL") || {
    echo "❌ curl request failed: $URL"
    exit 1
}

HTTP_STATUS=$(printf '%s\n' "$response" | tail -n 1)
BODY=$(printf '%s\n' "$response" | sed '$d')

echo "HTTP status: $HTTP_STATUS"
echo

case "$HTTP_STATUS" in
    2??)
        echo "Available models:"
        printf '%s\n' "$BODY" | jq -r '.data[].id'
        ;;

    *)
        echo "Response:"
        printf '%s\n' "$BODY" | jq . 2>/dev/null || printf '%s\n' "$BODY"
        echo
        echo "❌ API request failed: HTTP $HTTP_STATUS"
        exit 1
        ;;
esac

echo
echo "=== Testing MAAS API: /v1/chat/completions ==="

CHAT_URL="${MAAS_BASE_URL%/}/v1/chat/completions"

response=$(curl -sS \
    -w '\n%{http_code}' \
    -H "Authorization: Bearer $MAAS_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MAAS_MODEL_NAME\",
        \"messages\": [
            {
                \"role\": \"user\",
                \"content\": \"Say hello in one short sentence.\"
            }
        ],
        \"max_tokens\": 1024
    }" \
    "$CHAT_URL") || {
    echo "❌ curl request failed: $CHAT_URL"
    exit 1
}

HTTP_STATUS=$(printf '%s\n' "$response" | tail -n 1)
BODY=$(printf '%s\n' "$response" | sed '$d')

echo "HTTP status: $HTTP_STATUS"
echo

case "$HTTP_STATUS" in
    2??)
        echo "Response:"
        printf '%s\n' "$BODY" | jq . 2>/dev/null || printf '%s\n' "$BODY"
        echo
        echo "✅ chat completion request succeeded"
        ;;

    *)
        echo "❌ chat completion request failed: HTTP $HTTP_STATUS"
        echo
        echo "Error payload:"
        printf '%s\n' "$BODY" | jq . 2>/dev/null || printf '%s\n' "$BODY"
        exit 1
        ;;
esac

echo
echo "🎉 All tests passed!"