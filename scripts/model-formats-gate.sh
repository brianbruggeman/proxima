#!/usr/bin/env bash
# model-formats-gate.sh — local mirror of proxima-tensor.yml's `model-formats`
# matrix job for the five sans-IO model-format crates.
#
# Why it exists: measured 2026-08-29, CI's `model-formats` job runs
# `clippy --all-targets --all-features`, `nextest --all-features`, and
# `doc --no-deps --all-features` per crate, and NOTHING local ran that union.
# The 33 existing gate scripts use explicit feature lists (proxima-tensor-gate
# and proxima-tokenizer-gate cover only their own crate and never touch
# proxima-gguf/proxima-safetensors/proxima-onnx/proxima-model-interop at all),
# so an `--all-features` rustdoc break — an ambiguous `[`f16`]` module/primitive
# link in proxima-gguf, and 25 public-doc-links-to-private-item errors across
# proxima-model-interop — landed on main and was caught only by CI. This
# script is the mechanical fix: it runs the exact three CI commands, per
# crate, so `bash scripts/model-formats-gate.sh` fails locally the same way
# the CI cell did.
#
# The alloc-tier cell mirrors the CI job's `tier: alloc` matrix arm
# (`clippy --no-default-features --features alloc`, clippy only — CI does not
# run nextest/doc under that tier either, so this script does not invent
# coverage CI itself does not have). proxima-model-interop has no `alloc`
# feature of its own -- its `default = []` already IS that floor -- so its
# cell drops `--features alloc`, matching the workflow's own per-crate branch
# (proxima-tensor.yml's `model-formats` job, fixed 2026-08-29: `--features
# alloc` there errors "the package does not contain this feature").
#
# Usage:  bash scripts/model-formats-gate.sh
# Exits 0 if every cell passes, non-zero (after running them all) otherwise.

set -euo pipefail

declare -a crates=(
    proxima-gguf
    proxima-safetensors
    proxima-onnx
    proxima-model-interop
    proxima-tokenizer
)

declare -a cells=()
for crate in "${crates[@]}"; do
    cells+=("${crate} clippy --all-features|cargo clippy -p ${crate} --all-targets --all-features -- -D warnings")
    cells+=("${crate} rustdoc --all-features|cargo doc -p ${crate} --no-deps --all-features")
    if [ "${crate}" = "proxima-model-interop" ]; then
        cells+=("${crate} alloc tier clippy|cargo clippy -p ${crate} --no-default-features -- -D warnings")
    else
        cells+=("${crate} alloc tier clippy|cargo clippy -p ${crate} --no-default-features --features alloc -- -D warnings")
    fi
done

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

# nextest --all-features gets its own loop, count asserted nonzero: an
# --all-features run whose feature union silently compiles zero tests for a
# crate (a misconfigured `required-features`, a cfg'd-out test module) exits 0
# and reads as green the same way `cargo test --doc` does. mirror
# proxima-tensor-gate.sh's doctest cell and proxima-tokenizer-gate.sh's
# ignored-fixture cell rather than trusting nextest's own exit code alone.
for crate in "${crates[@]}"; do
    printf '\n== %s nextest --all-features (count asserted nonzero) ==\n' "$crate"
    nextest_log="$(mktemp)"
    if cargo nextest run -p "$crate" --all-features --no-fail-fast > "$nextest_log" 2>&1; then
        cat "$nextest_log"
        test_count="$(grep -oE '[0-9]+ tests run' "$nextest_log" | tail -1 | awk '{ print $1 }')"
        test_count="${test_count:-0}"
        printf '%s tests run: %s\n' "$crate" "$test_count"
        if [ "$test_count" -lt 1 ]; then
            printf 'RED: nextest ran %s tests for %s. an empty match exits 0 and reads as green.\n' "$test_count" "$crate"
            failed=$((failed + 1))
            failures+=("${crate} nextest ran zero")
        else
            passed=$((passed + 1))
        fi
    else
        cat "$nextest_log"
        failed=$((failed + 1))
        failures+=("${crate} nextest --all-features")
    fi
    rm -f "$nextest_log"
done

printf '\n== model-formats-gate summary ==\n'
printf '   passed: %d\n' "$passed"
printf '   failed: %d\n' "$failed"

if [ "$failed" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "$label"
    done
    exit 1
fi

printf '\nmodel-formats-gate: all green.\n'
