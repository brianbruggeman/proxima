#!/usr/bin/env bash
# proxima-tokenizer-gate.sh — feature matrix gate for proxima-tokenizer.
#
# Why it exists: measured 2026-08-19, `default = ["std"]` does not pull in
# `gguf`, so `cargo test -p proxima-tokenizer` (the crate's own documented
# command) compiles none of gguf.rs's tests and reports GREEN having run zero
# of them. Those tests include the freshly-captured llama.cpp oracle parity
# fixtures, marked `#[ignore]` because they read a real GGUF file from disk —
# the crate's single most load-bearing test suite is invisible to both its
# default command AND a bare `--features gguf` run unless `-- --ignored` is
# also passed. This is the sixth instance of a gate reporting success because
# it never processed the case; this script is the mechanical fix, mirroring
# proxima-tensor-gate.sh's shape (per-feature cells, run each feature ALONE,
# assert a nonzero count rather than trust an exit code).
#
# `gguf` is deliberately NOT added to `default` (see the feature's doc
# comment in proxima-tokenizer/Cargo.toml): turning it on for every consumer
# would pull `proxima-gguf` and its non-optional deps (arrayvec, half, libm)
# into every default build of this crate, when the crate's whole point is
# "no hard dependency on the gguf reader". A gate script that reaches the
# feature is the smaller, correctly-scoped fix.
#
# Usage:  bash scripts/proxima-tokenizer-gate.sh
# Exits 0 if every cell passes, non-zero (after running them all) otherwise.

set -euo pipefail

# CI sets CARGO_TERM_COLOR=always; ANSI escapes wrapped around digits break
# any grep/awk that counts cargo/nextest summary output (the class
# proxima-tokenizer-gate.sh and proxima-test-gate.sh hit) -- force it off
# for every invocation in this script, present and future.
export CARGO_TERM_COLOR=never

declare -a cells=(
    # what the crate's own documented commands cover
    "default check|cargo check -p proxima-tokenizer --all-targets"
    "default tests|cargo nextest run -p proxima-tokenizer --no-fail-fast"
    "default clippy|cargo clippy -p proxima-tokenizer --all-targets -- -D warnings"
    "default rustdoc|cargo doc -p proxima-tokenizer --no-deps"

    # the no_std + alloc floor this crate claims (also proven on the
    # thumbv7m-none-eabi cliff by scripts/thumbv7m-cliff-gate.sh via
    # scripts/_floor-crate-matrix.sh -- this cell is the fast host-side check).
    "alloc tier check|cargo check -p proxima-tokenizer --no-default-features --features alloc"
    "alloc tier clippy|cargo clippy -p proxima-tokenizer --no-default-features --features alloc -- -D warnings"

    # each feature ALONE -- unification with the others must not be what
    # makes it compile.
    "std alone clippy|cargo clippy -p proxima-tokenizer --all-targets --no-default-features --features std -- -D warnings"
    "gguf alone clippy|cargo clippy -p proxima-tokenizer --all-targets --no-default-features --features gguf -- -D warnings"

    # the gguf feature's own test suite, unignored. this is the cell the
    # default cell above never reaches: `gguf` is off by default, so its
    # tests compile nowhere else in this script.
    "gguf tests|cargo nextest run -p proxima-tokenizer --no-default-features --features gguf --no-fail-fast"
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

# the llama.cpp oracle parity fixtures: `#[ignore]` because they read a real
# GGUF file from disk, so neither the default cell nor the plain `gguf tests`
# cell above ever runs them. this is the exact defect closed here -- count
# the passes and fail the cell if it processed zero tests, the same shape as
# proxima-tensor-gate.sh's doctest cell.
# `--color never`: CI sets `CARGO_TERM_COLOR: always` (proxima-tensor.yml),
# which wraps nextest's summary digits in ANSI escape codes even though this
# output is captured to a file, not a terminal. That silently breaks the
# `grep -oE '[0-9]+ passed'` below (no match, count defaults to 0, RED) —
# reproduced on a real ubuntu host with CARGO_TERM_COLOR=always set.
printf '\n== gguf oracle fixtures, --ignored (count asserted nonzero) ==\n'
ignored_log="$(mktemp)"
if cargo nextest run -p proxima-tokenizer --no-default-features --features gguf --run-ignored ignored-only --no-fail-fast --color never \
    > "$ignored_log" 2>&1; then
    cat "$ignored_log"
    ignored_count="$(grep -oE '[0-9]+ passed' "$ignored_log" | tail -1 | awk '{ print $1 }')"
    ignored_count="${ignored_count:-0}"
    printf 'ignored tests run: %s\n' "$ignored_count"
    if [ "$ignored_count" -lt 1 ]; then
        printf 'RED: nextest ran %s ignored tests. an empty match exits 0 and reads as green.\n' "$ignored_count"
        failed=$((failed + 1))
        failures+=("gguf oracle fixtures ran zero")
    else
        passed=$((passed + 1))
    fi
else
    cat "$ignored_log"
    failed=$((failed + 1))
    failures+=("gguf oracle fixtures")
fi
rm -f "$ignored_log"

printf '\n== proxima-tokenizer-gate summary ==\n'
printf '   passed: %d\n' "$passed"
printf '   failed: %d\n' "$failed"

if [ "$failed" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "$label"
    done
    exit 1
fi

printf '\nproxima-tokenizer-gate: all green.\n'
