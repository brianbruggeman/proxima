#!/usr/bin/env bash
# proxima-test-gate.sh — feature-matrix gate for proxima-test.
#
# Why it exists: on 2026-08-05 `cargo check -p proxima-test --all-targets`
# exited 101 with sixteen deny-level errors, and had for some time. The crate
# is `default = []`; both drivers (`tokio-driver`, `test-prime`) are default
# off, so with neither one linked the panic capture, the fd-limit raise,
# `CatchUnwind` and the ctx builder were all dead code and `OnceLock` was an
# unused import. No workspace command could see it: `cargo check --workspace`
# unifies a driver in from `proxima`, and every crate that dev-deps this one
# forwards a driver too. The broken cell is reachable only by building the
# crate ALONE — which is exactly what `proxima/test-support` links, since that
# feature is runtime-agnostic and forwards no driver at all.
#
# So each driver gets a cell of its OWN, plus the bare cell. `tokio-driver`
# and `test-prime` select different `run` definitions (`cfg(all(not(test-
# prime), tokio-driver))`), so a build with both is NOT evidence about either.
#
# The last cell is the one that matters most and is the one a check-only gate
# cannot give: proxima-macros writes the call to these entry points, and only
# RUNNING `tests/e2e/proxima_test_smoke.rs` proves the macro and the harness
# still agree on the argument list, the cassette path and the fixture cell.
#
# Usage:  bash scripts/proxima-test-gate.sh
# Exits 0 if every cell passes, non-zero (after running them all) otherwise.

set -euo pipefail

# `cargo test --doc` exits 0 when it matched NOTHING, so the exit code alone
# cannot tell "all passed" from "there were none". Assert the count.
doctests_ran_and_passed() {
    local output
    if ! output="$(cargo test -p proxima-test --doc 2>&1)"; then
        printf '%s\n' "$output"
        return 1
    fi
    printf '%s\n' "$output"
    grep -qE 'test result: ok\. [1-9][0-9]* passed' <<<"$output"
}

# same trap one level up: a filtered nextest run whose filter matches nothing
# exits 0. `proxima_test_smoke.rs` is `#![cfg(feature = "test-support")]` and
# every cassette case carries a runtime cfg on top, so "0 tests run" is a very
# reachable false green here. Assert a nonzero count.
#
# `--color never`: CI sets `CARGO_TERM_COLOR: always`, which makes nextest
# wrap the digits in its "N tests run: N passed" summary line in ANSI escape
# codes even though the output is captured to a file, not a terminal. That
# breaks the contiguous-text grep below silently (exit 1, read as "0 tests
# ran"), which is exactly the false-negative this cell exists to catch —
# reproduced on a real ubuntu host with CARGO_TERM_COLOR=always set.
macro_smoke_ran_and_passed() {
    local output
    if ! output="$(cargo nextest run -p proxima --test e2e \
        --features test-prime,http-hyper,http3-quinn-compat \
        -E 'test(proxima_test_smoke)' --no-fail-fast --color never 2>&1)"; then
        printf '%s\n' "$output"
        return 1
    fi
    printf '%s\n' "$output"
    grep -qE 'Summary.*[1-9][0-9]* tests run: [1-9][0-9]* passed' <<<"$output"
}

declare -a cells=(
    # the bare cell — no driver linked. This is the one that was red.
    "bare check|cargo check -p proxima-test --all-targets"
    "bare tests|cargo nextest run -p proxima-test --no-fail-fast"
    "bare clippy|cargo clippy -p proxima-test --all-targets -- -D warnings"
    "bare rustdoc|cargo doc -p proxima-test --no-deps"
    "bare doctests, count asserted|doctests_ran_and_passed"

    # each driver ALONE — they select different `run` bodies, so the union
    # below is not evidence about either one.
    "tokio-driver alone clippy|cargo clippy -p proxima-test --all-targets --features tokio-driver -- -D warnings"
    "tokio-driver alone tests|cargo nextest run -p proxima-test --features tokio-driver --no-fail-fast"
    "test-prime alone clippy|cargo clippy -p proxima-test --all-targets --features test-prime -- -D warnings"
    "test-prime alone tests|cargo nextest run -p proxima-test --features test-prime --no-fail-fast"
    "test-prime-tokio-compat clippy|cargo clippy -p proxima-test --all-targets --features test-prime-tokio-compat -- -D warnings"
    "test-prime-tokio-compat tests|cargo nextest run -p proxima-test --features test-prime-tokio-compat --no-fail-fast"

    # the union
    "all-features clippy|cargo clippy -p proxima-test --all-targets --all-features -- -D warnings"
    "all-features tests|cargo nextest run -p proxima-test --all-features --no-fail-fast"
    "all-features rustdoc|cargo doc -p proxima-test --no-deps --all-features"

    # the consumer contract, actually RUN
    "macro smoke, count asserted|macro_smoke_ran_and_passed"
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

printf '\n== proxima-test-gate summary ==\n'
printf '   passed: %d\n' "$passed"
printf '   failed: %d\n' "$failed"

if [ "$failed" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "$label"
    done
    exit 1
fi

printf '\nproxima-test-gate: all green.\n'
