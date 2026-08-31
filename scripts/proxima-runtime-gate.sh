#!/usr/bin/env bash
# proxima-runtime-gate.sh — feature-matrix gate for proxima-runtime.
#
# Why it exists: `cargo nextest run -p proxima-runtime` and
# `cargo clippy -p proxima-runtime --all-targets` see only `default =
# ["std"]`, which compiles lib.rs, ext.rs and primitives.rs and NOTHING
# else. concurrency/ (2.5k LOC), tokio/ (0.6k) and background_rayon.rs
# are all default-off, so the crate's own commands proved a quarter of
# it. When the 2026-08-04 consistency audit first ran the same commands
# with `--all-features` they failed on three denied `unwrap()`s, a
# `needless_update`, and five redundant rustdoc link targets — none of
# which any gate could have caught.
#
# Each feature also gets a cell of its OWN, not just a place in the
# --all-features union: a feature that only ever compiles unified with
# its siblings is a feature nobody has proven.
#
# The no_std tiers are NOT duplicated here — scripts/thumbv7m-cliff-gate.sh
# and scripts/tokio-free-floor.sh already carry the `proxima-runtime` and
# `proxima-runtime-bare-no-alloc` cells (guiding-principle 1: one source,
# no forked copies). The host-side floor builds below are the cheap
# pre-check those two gates then prove on the cliff.
#
# Usage:  bash scripts/proxima-runtime-gate.sh
# Exits 0 if every cell passes, non-zero (after running them all) otherwise.

set -euo pipefail

# CI sets CARGO_TERM_COLOR=always; ANSI escapes wrapped around digits break
# any grep/awk that counts cargo/nextest summary output (the class
# proxima-tokenizer-gate.sh and proxima-test-gate.sh hit) -- force it off
# for every invocation in this script, present and future.
export CARGO_TERM_COLOR=never

# `cargo test --doc` exits 0 when it matched NOTHING, so the exit code alone
# cannot tell "all passed" from "there were none" — the crate had zero doctests
# until 2026-08-04 and every gate read that as green. Assert the count.
doctests_ran_and_passed() {
    local output
    if ! output="$(cargo test -p proxima-runtime --all-features --doc 2>&1)"; then
        printf '%s\n' "$output"
        return 1
    fi
    printf '%s\n' "$output"
    grep -qE 'test result: ok\. [1-9][0-9]* passed' <<<"$output"
}

declare -a cells=(
    # what the crate's own documented commands cover
    "default check|cargo check -p proxima-runtime --all-targets"
    "default tests|cargo nextest run -p proxima-runtime --no-fail-fast"
    "default clippy|cargo clippy -p proxima-runtime --all-targets -- -D warnings"

    # the union — everything compiled together
    "all-features tests|cargo nextest run -p proxima-runtime --all-features --no-fail-fast"
    "all-features clippy|cargo clippy -p proxima-runtime --all-targets --all-features -- -D warnings"
    "all-features rustdoc|cargo doc -p proxima-runtime --all-features --no-deps"
    "default rustdoc|cargo doc -p proxima-runtime --no-deps"
    "all-features doctests, count asserted|doctests_ran_and_passed"

    # the tiers, host-side (the cliff proof lives in thumbv7m-cliff-gate.sh)
    "no_std no-alloc floor|cargo build -p proxima-runtime --no-default-features"
    "no_std + alloc floor|cargo build -p proxima-runtime --no-default-features --features alloc"

    # each feature ALONE — unification must not be what makes it compile
    "tokio alone clippy|cargo clippy -p proxima-runtime --all-targets --no-default-features --features tokio -- -D warnings"
    "tokio alone tests|cargo nextest run -p proxima-runtime --no-default-features --features tokio --no-fail-fast"
    "concurrency alone clippy|cargo clippy -p proxima-runtime --all-targets --no-default-features --features concurrency -- -D warnings"
    "concurrency alone tests|cargo nextest run -p proxima-runtime --no-default-features --features concurrency --no-fail-fast"
    "rayon alone clippy|cargo clippy -p proxima-runtime --all-targets --no-default-features --features rayon -- -D warnings"
    "rayon alone tests|cargo nextest run -p proxima-runtime --no-default-features --features rayon --no-fail-fast"
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

printf '\n== proxima-runtime-gate summary ==\n'
printf '   passed: %d\n' "$passed"
printf '   failed: %d\n' "$failed"

if [ "$failed" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "$label"
    done
    exit 1
fi

printf '\nproxima-runtime-gate: all green.\n'
