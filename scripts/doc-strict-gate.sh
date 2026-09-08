#!/usr/bin/env bash
# doc-strict-gate.sh — one cell, workspace-wide: rustdoc resolves every
# intra-doc link across every crate, denied not warned.
#
# Why it exists: main went clean under `RUSTDOCFLAGS='-D warnings' cargo doc
# --workspace --no-deps` on 2026-09-08 after ~250 link fixes across 12 crates
# (d759627). Nothing re-proves that on every commit — the per-crate gates
# (proxima-tensor-gate.sh, proxima-auth-gate.sh, proxima-config-gate.sh, ...)
# each carry their own rustdoc cell, but none of them spans the whole
# workspace in one command, so a link broken by crate A referencing crate B
# can pass every affected-gated per-crate job and still break `cargo doc
# --workspace`. This is that missing cell, unconditional on every push/PR
# (mirrors algebra-lint.yml's shape, not the affected-gated per-crate gates).
#
# Default features only, deliberately: `--all-features` is impossible here.
# prime's `runtime-prime-inbox-alloc` and `runtime-prime-inbox-const` are
# mutually exclusive by design (const-generic vs. alloc-backed inbox storage
# for the same slot), so no single `--all-features` invocation can unify the
# workspace. Default features is the floor every consumer actually builds
# docs against.
#
# Usage:  bash scripts/doc-strict-gate.sh
# Exits 0 if the cell passes, non-zero otherwise.

set -euo pipefail

export CARGO_TERM_COLOR=never

declare -a cells=(
    "doc-strict|RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps"
)

passed=0
failed=0
declare -a failures

for cell in "${cells[@]}"; do
    label="${cell%%|*}"
    command="${cell#*|}"

    printf '\n== %s ==\n%s\n' "$label" "$command"

    if eval "$command"; then
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
        failures+=("$label")
    fi
done

printf '\n== doc-strict-gate summary ==\n'
printf '   passed: %d\n' "$passed"
printf '   failed: %d\n' "$failed"

if [ "$failed" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "$label"
    done
    exit 1
fi

printf '\ndoc-strict-gate: all green.\n'
