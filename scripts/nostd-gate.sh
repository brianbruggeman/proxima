#!/usr/bin/env bash
# nostd-gate.sh — discipline gate for dualcore crate "core" subtrees.
#
# Walks production code lines (everything before the bottom-of-file `#[cfg(test)]`
# tests module) and flags std:: drift: qualified `std::` paths that are NOT
# reachable behind a `#[cfg(feature = "std")]` gate, plus every `thread_local!`
# site (qualified or bare — it resolves to std via the prelude) regardless of
# gating, since raw TLS is itself the deferred debt being tracked.
#
# cfg-awareness: a small line-oriented state machine (nostd-gate.awk, next
# to this script) tracks whether the current line sits inside an item/block
# whose nearest enclosing `#[cfg(feature = "std")]` (or
# `#[cfg(all(..., feature = "std", ...))]`) attribute is active.
# `#[cfg(not(feature = "std"))]` (and any predicate containing that literal)
# never activates the gate. The gate propagates through a single item (a
# `use`, a `fn` body, an `impl` block, a `mod`, a bare `{ }` block, a
# `thread_local! { }` block, ...) via brace-depth tracking, so a gated `fn`
# body's std:: calls are covered without needing per-line allow-list entries.
#
# Allow-list: `thread_local!` is intentionally exempt from cfg-awareness (see
# above) — known deferred-debt TLS sites are tolerated by content match (not
# line number, so they survive surrounding-code drift) until C1
# (thread-identity-trait) lands and routes them through ThreadIdentity.
#
# Usage:
#   bash scripts/nostd-gate.sh                # default scope: prime/src/core/
#   bash scripts/nostd-gate.sh <file> [...]   # explicit files
#
# Exits 0 if clean, 1 if any violation found.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
gate_awk="${script_dir}/nostd-gate.awk"

declare -a paths
if [ "$#" -eq 0 ]; then
    paths=(
        prime/src/core/inbox.rs
        prime/src/core/local_executor.rs
        prime/src/core/timer.rs
        prime/src/core/inline_task.rs
        prime/src/core/sized.rs
    )
else
    paths=("$@")
fi

# Known deferred-debt TLS sites — drop once C1 (thread-identity-trait) lands
# and the sites route through ThreadIdentity instead of raw thread_local!.
# Matched by (file, content-substring) rather than line number, so drift in
# surrounding code never requires renumbering these entries.
declare -a allowed_files
declare -a allowed_content
allowed_files+=("prime/src/core/inbox.rs")
allowed_content+=("std::thread_local! {")
allowed_files+=("prime/src/core/local_executor.rs")
allowed_content+=("thread_local! {")

is_allowed() {
    local file="$1" rest="$2"
    local index
    for index in "${!allowed_files[@]}"; do
        if [ "${allowed_files[$index]}" = "$file" ] && [[ "$rest" == *"${allowed_content[$index]}"* ]]; then
            return 0
        fi
    done
    return 1
}

violations_log=$(mktemp)
trap 'rm -f "$violations_log"' EXIT

for file in "${paths[@]}"; do
    if [ ! -f "$file" ]; then
        printf 'warning: %s does not exist\n' "$file" >&2
        continue
    fi

    test_line=$(grep -n '^#\[cfg(test)\]' "$file" | head -1 | cut -d: -f1 || true)
    if [ -z "${test_line:-}" ]; then
        end_line=$(wc -l < "$file" | tr -d ' ')
    else
        end_line=$((test_line - 1))
    fi

    matches=$(head -n "$end_line" "$file" | awk -f "$gate_awk" || true)
    if [ -z "$matches" ]; then
        continue
    fi

    while IFS= read -r match; do
        line_num=${match%%:*}
        rest=${match#*:}
        if is_allowed "$file" "$rest"; then
            continue
        fi
        printf '%s:%s:%s\n' "$file" "$line_num" "$rest" >> "$violations_log"
    done <<< "$matches"
done

if [ -s "$violations_log" ]; then
    cat "$violations_log" >&2
    count=$(wc -l < "$violations_log" | tr -d ' ')
    printf '\n== nostd-gate: %d violation(s) ==\n' "$count" >&2
    exit 1
fi

printf 'nostd-gate: clean (checked %d file(s))\n' "${#paths[@]}"
