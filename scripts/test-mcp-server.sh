#!/bin/sh

set -eu

usage() {
    printf 'Usage: %s <namespace> <mcp-server-name>\n' "$0" >&2
    exit 1
}

[ "$#" -eq 2 ] || usage

NAMESPACE="$1"
MCP_SERVER="$2"

# --------------------------------------------------------------------
# Terminal colors
# --------------------------------------------------------------------
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    RESET='\033[0m'
else
    GREEN=''
    RED=''
    RESET=''
fi

PASS="${GREEN}✓${RESET}"
FAIL="${RED}✗${RESET}"

# --------------------------------------------------------------------
# Requirements
# --------------------------------------------------------------------
for command in oc curl jq; do
    if ! command -v "$command" >/dev/null 2>&1; then
        printf '%b ERROR: %s is required\n' "$FAIL" "$command" >&2
        exit 1
    fi
done

# --------------------------------------------------------------------
# Temporary files
# --------------------------------------------------------------------
HEADERS="$(mktemp)"
BODY="$(mktemp)"

cleanup() {
    rm -f "$HEADERS" "$BODY"
}

trap cleanup EXIT INT TERM

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
print_separator() {
    printf '%s\n' '────────────────────────────────────────'
}

ready_status() {
    printf '%s' "$1" |
        jq -r '
            (.status.conditions // [])
            | map(select(.type == "Ready"))
            | .[0].status // "Unknown"
        '
}

ready_reason() {
    printf '%s' "$1" |
        jq -r '
            (.status.conditions // [])
            | map(select(.type == "Ready"))
            | .[0].reason // "Unknown"
        '
}

ready_message() {
    printf '%s' "$1" |
        jq -r '
            (.status.conditions // [])
            | map(select(.type == "Ready"))
            | .[0].message // "Unknown"
        '
}

# --------------------------------------------------------------------
# 1. Find the OpenShift cluster domain
# --------------------------------------------------------------------
CLUSTER_DOMAIN="$(
    oc get ingress.config.openshift.io cluster \
        -o jsonpath='{.spec.domain}'
)"

if [ -z "$CLUSTER_DOMAIN" ]; then
    printf '%b ERROR: Could not determine OpenShift cluster domain\n' \
        "$FAIL" >&2
    exit 1
fi

MCP_URL="http://mcp.${CLUSTER_DOMAIN}:8080/mcp"

echo
echo "MCP Gateway"
print_separator
printf '  Cluster domain : %s\n' "$CLUSTER_DOMAIN"
printf '  MCP URL        : %s\n' "$MCP_URL"

# --------------------------------------------------------------------
# 2. Get MCPServerRegistrations
#
# Fetch them once. The same JSON is used for:
#   - requested server validation
#   - registration listing
#   - registration count
#   - discovered tool count
# --------------------------------------------------------------------
ALL_REGISTRATIONS="$(
    oc get mcpserverregistration \
        -n "$NAMESPACE" \
        -o json
)"

REGISTRATION_COUNT="$(
    printf '%s' "$ALL_REGISTRATIONS" |
        jq '.items | length'
)"

TOTAL_DISCOVERED_TOOLS="$(
    printf '%s' "$ALL_REGISTRATIONS" |
        jq '
            [.items[].status.discoveredTools // 0]
            | add // 0
        '
)"

# --------------------------------------------------------------------
# 3. Display registrations
# --------------------------------------------------------------------
echo
echo
echo "MCPServerRegistrations"
printf 'Namespace: %s\n' "$NAMESPACE"
print_separator

printf '%s' "$ALL_REGISTRATIONS" |
    jq -r '
        .items[] |
        [
            .metadata.name,
            (.status.discoveredTools // 0),
            (
                (.status.conditions // [])
                | map(select(.type == "Ready"))
                | .[0].status // "Unknown"
            )
        ] |
        @tsv
    ' |
    while IFS="$(printf '\t')" read -r NAME TOOLS STATUS; do
        if [ "$STATUS" = "True" ]; then
            printf '  %b %-35s %4s tools    Ready\n' \
                "$PASS" "$NAME" "$TOOLS"
        else
            printf '  %b %-35s %4s tools    Not Ready\n' \
                "$FAIL" "$NAME" "$TOOLS"
        fi
    done

echo
printf '  Registrations : %s\n' "$REGISTRATION_COUNT"
printf '  Discovered    : %s\n' "$TOTAL_DISCOVERED_TOOLS"

# --------------------------------------------------------------------
# 4. Validate requested MCPServerRegistration
# --------------------------------------------------------------------
REGISTRATION_JSON="$(
    printf '%s' "$ALL_REGISTRATIONS" |
        jq -c --arg name "$MCP_SERVER" '
            .items[]
            | select(.metadata.name == $name)
        '
)"

if [ -z "$REGISTRATION_JSON" ]; then
    echo
    printf '  %b ERROR: MCPServerRegistration "%s" not found\n' \
        "$FAIL" "$MCP_SERVER"
    exit 1
fi

READY="$(ready_status "$REGISTRATION_JSON")"

if [ "$READY" != "True" ]; then
    echo
    printf '  %b ERROR: MCPServerRegistration "%s" is not Ready\n' \
        "$FAIL" "$MCP_SERVER"

    printf '  Reason : %s\n' "$(ready_reason "$REGISTRATION_JSON")"
    printf '  Message: %s\n' "$(ready_message "$REGISTRATION_JSON")"

    exit 1
fi

# --------------------------------------------------------------------
# 5. Initialize MCP session
# --------------------------------------------------------------------
echo
echo
echo "MCP Session"
print_separator
echo "  Initializing MCP gateway..."

curl -sS --fail \
    -D "$HEADERS" \
    -o "$BODY" \
    -X POST "$MCP_URL" \
    -H "Content-Type: application/json" \
    -d '{
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }'

# --------------------------------------------------------------------
# 6. Extract MCP session ID
# --------------------------------------------------------------------
SESSION_ID="$(
    awk '
        BEGIN { IGNORECASE = 1 }
        /^mcp-session-id:/ {
            sub(/^[^:]*:[[:space:]]*/, "")
            sub(/[[:space:]\r]*$/, "")
            print
            exit
        }
    ' "$HEADERS"
)"

if [ -z "$SESSION_ID" ]; then
    printf '  %b ERROR: MCP gateway did not return mcp-session-id\n' \
        "$FAIL" >&2

    echo
    echo "Response headers:" >&2
    cat "$HEADERS" >&2

    echo
    echo "Response body:" >&2
    cat "$BODY" >&2

    exit 1
fi

printf '  %b MCP session initialized\n' "$PASS"

# --------------------------------------------------------------------
# 7. List gateway tools
# --------------------------------------------------------------------
echo
echo
echo "Gateway Tools"
print_separator

TOOLS_RESPONSE="$(
    curl -sS --fail \
        -X POST "$MCP_URL" \
        -H "Content-Type: application/json" \
        -H "mcp-session-id: $SESSION_ID" \
        -d '{
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/list"
        }'
)"

if ! printf '%s' "$TOOLS_RESPONSE" |
    jq -e '.result.tools' >/dev/null 2>&1
then
    printf '  %b ERROR: Invalid tools/list response\n' "$FAIL" >&2
    printf '%s\n' "$TOOLS_RESPONSE" >&2
    exit 1
fi

GATEWAY_TOOL_COUNT="$(
    printf '%s' "$TOOLS_RESPONSE" |
        jq '.result.tools | length'
)"

printf '  Gateway returned %s tools\n' "$GATEWAY_TOOL_COUNT"
echo

printf '%s' "$TOOLS_RESPONSE" |
    jq -r '.result.tools[]?.name' |
    while IFS= read -r TOOL; do
        printf '  • %s\n' "$TOOL"
    done

# --------------------------------------------------------------------
# 8. Account for gateway meta tools
# --------------------------------------------------------------------
#
# These tools are provided by the gateway itself and are not included
# in MCPServerRegistration.status.discoveredTools.
#
GATEWAY_META_TOOLS='
discover_tools
select_tools
'

GATEWAY_META_COUNT="$(
    printf '%s' "$TOOLS_RESPONSE" |
        jq --argjson meta_tools '
            ["discover_tools", "select_tools"]
        ' '
            [
                .result.tools[]?.name
                | select(. as $name | $meta_tools | index($name))
            ]
            | length
        '
)"

GATEWAY_DISCOVERED_TOOL_COUNT=$(
    printf '%s' "$GATEWAY_TOOL_COUNT - $GATEWAY_META_COUNT" |
        bc
)

# --------------------------------------------------------------------
# 9. Final validation
# --------------------------------------------------------------------
echo
echo
echo "MCP Tool Count Validation"
print_separator

printf '  Registrations               : %s\n' \
    "$REGISTRATION_COUNT"

printf '  Discovered tools            : %s\n' \
    "$TOTAL_DISCOVERED_TOOLS"

printf '  Gateway tools               : %s\n' \
    "$GATEWAY_TOOL_COUNT"

printf '  Gateway meta tools          : %s\n' \
    "$GATEWAY_META_COUNT"

printf '  Gateway discovered tools    : %s\n' \
    "$GATEWAY_DISCOVERED_TOOL_COUNT"

echo

if [ "$TOTAL_DISCOVERED_TOOLS" -eq "$GATEWAY_DISCOVERED_TOOL_COUNT" ]; then
    printf '  %b PASS  Tool counts match\n' "$PASS"
    exit 0
fi

printf '  %b FAIL  Tool counts do not match\n' "$FAIL"
printf '          Expected: %s\n' "$TOTAL_DISCOVERED_TOOLS"
printf '          Actual:   %s\n' "$GATEWAY_DISCOVERED_TOOL_COUNT"

exit 1
