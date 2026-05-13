#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

task="${1:-}"

if [[ -z "${task}" ]]; then
    echo "usage: ai_docs/query.sh <task>"
    echo
    echo "tasks:"
    jq -r '.task' task-routes.jsonl
    exit 0
fi

route="$(jq -c --arg task "${task}" 'select(.task == $task)' task-routes.jsonl)"

if [[ -z "${route}" ]]; then
    echo "unknown task: ${task}" >&2
    echo "known tasks:" >&2
    jq -r '.task' task-routes.jsonl >&2
    exit 1
fi

echo "route:"
echo "${route}" | jq .
echo
echo "matching invariants:"

case "${task}" in
    disciplined-component)
        jq -c 'select(any(.applies_to[]?; . == "disciplined-component" or . == "hot-path"))' invariants.jsonl
        ;;
    sans-io)
        jq -c 'select(any(.applies_to[]?; . == "sans-io" or . == "parser" or . == "codec"))' invariants.jsonl
        ;;
    runtime-prime)
        jq -c 'select(any(.applies_to[]?; . == "runtime" or . == "hot-path"))' invariants.jsonl
        ;;
    bench-evidence)
        jq -c 'select(any(.read_when[]?; . == "bench-evidence"))' index.jsonl
        ;;
    decomposition)
        jq -c 'select(any(.read_when[]?; . == "crate-ownership" or . == "cycle-analysis"))' index.jsonl
        ;;
    *)
        jq -c . invariants.jsonl
        ;;
esac
