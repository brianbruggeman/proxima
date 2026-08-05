#!/usr/bin/env bash
# proxima-config-gate.sh — feature-matrix gate for proxima-config.
#
# Why it exists: `cargo nextest run -p proxima-config` and `cargo clippy -p
# proxima-config --all-targets` see only `default = ["std"]`, which is the
# config-format registry and nothing else. `sugar`, `schema`, `schema-std`,
# `schema-derive`, `store` and `store-std` — three former satellite crates
# folded into this one — are all default-off, so the crate's own commands ran
# 17 of its 89 tests and compiled none of the folded code. When the 2026-08-05
# consistency audit first ran the same commands per feature, seven of the eight
# cells failed to build their test target at all (`lib.rs`'s `#[cfg(test)] mod
# tests` carried no feature gate while every case in it needs `std`; two test
# modules took `String`/`Vec`/`vec!` from the std prelude at the crate's own
# no_std tier), and `cargo doc --all-features` exited 101 on twelve errors.
# None of that was reachable from a default-feature command.
#
# Each feature also gets a cell of its OWN, not just a place in the
# --all-features union: a feature that only ever compiles unified with its
# siblings is a feature nobody has proven. That is how the `schema-derive`
# rustdoc ambiguity (`Schema` is both the IR enum and the re-exported derive)
# stayed invisible.
#
# The no_std tiers are NOT duplicated here — scripts/thumbv7m-cliff-gate.sh and
# scripts/tokio-free-floor.sh carry the `proxima-config`, `-sugar`, `-schema`
# and `-store` cells (guiding-principle 1: one source, no forked copies). The
# host-side floor builds below are the cheap pre-check those two then prove on
# the cliff.
#
# Usage:  bash scripts/proxima-config-gate.sh
# Exits 0 if every cell passes, non-zero (after running them all) otherwise.

set -euo pipefail

# `cargo test --doc` exits 0 when it matched NOTHING, so the exit code alone
# cannot tell "all passed" from "there were none" — and proxima-config's only
# doctest lives in the default-OFF `sugar` module, so the default-feature run
# reports zero and every gate reading exit codes calls that green. Assert the
# count.
doctests_ran_and_passed() {
    local output
    if ! output="$(cargo test -p proxima-config --all-features --doc 2>&1)"; then
        printf '%s\n' "$output"
        return 1
    fi
    printf '%s\n' "$output"
    grep -qE 'test result: ok\. [1-9][0-9]* passed' <<<"$output"
}

declare -a cells=(
    # what the crate's own documented commands cover
    "default check|cargo check -p proxima-config --all-targets"
    "default tests|cargo nextest run -p proxima-config --no-fail-fast"
    "default clippy|cargo clippy -p proxima-config --all-targets -- -D warnings"
    "default rustdoc|cargo doc -p proxima-config --no-deps"

    # the union — everything compiled together
    "all-features tests|cargo nextest run -p proxima-config --all-features --no-fail-fast"
    "all-features clippy|cargo clippy -p proxima-config --all-targets --all-features -- -D warnings"
    "all-features rustdoc|cargo doc -p proxima-config --no-deps --all-features"
    "all-features doctests, count asserted|doctests_ran_and_passed"

    # the tiers, host-side (the cliff proof lives in thumbv7m-cliff-gate.sh).
    # bare is a real cell here even though it is not one on the cliff: with no
    # features the crate must still COMPILE, and it did not until 2026-08-05.
    "no_std bare|cargo clippy -p proxima-config --all-targets --no-default-features -- -D warnings"
    "no_std bare rustdoc|cargo doc -p proxima-config --no-deps --no-default-features"
    "no_std + alloc floor|cargo clippy -p proxima-config --all-targets --no-default-features --features alloc -- -D warnings"
    "no_std + alloc rustdoc|cargo doc -p proxima-config --no-deps --no-default-features --features alloc"

    # each feature ALONE — unification must not be what makes it compile
    "sugar alone clippy|cargo clippy -p proxima-config --all-targets --no-default-features --features sugar -- -D warnings"
    "sugar alone tests|cargo nextest run -p proxima-config --no-default-features --features sugar --no-fail-fast"
    "schema alone clippy|cargo clippy -p proxima-config --all-targets --no-default-features --features schema -- -D warnings"
    "schema alone tests|cargo nextest run -p proxima-config --no-default-features --features schema --no-fail-fast"
    "schema-std alone clippy|cargo clippy -p proxima-config --all-targets --no-default-features --features schema-std -- -D warnings"
    "schema-std alone tests|cargo nextest run -p proxima-config --no-default-features --features schema-std --no-fail-fast"
    "schema-derive alone clippy|cargo clippy -p proxima-config --all-targets --no-default-features --features schema-derive -- -D warnings"
    "schema-derive alone tests|cargo nextest run -p proxima-config --no-default-features --features schema-derive --no-fail-fast"
    "schema-derive alone rustdoc|cargo doc -p proxima-config --no-deps --no-default-features --features schema-derive"
    "store alone clippy|cargo clippy -p proxima-config --all-targets --no-default-features --features store -- -D warnings"
    "store alone tests|cargo nextest run -p proxima-config --no-default-features --features store --no-fail-fast"
    "store-std alone clippy|cargo clippy -p proxima-config --all-targets --no-default-features --features store-std -- -D warnings"
    "store-std alone tests|cargo nextest run -p proxima-config --no-default-features --features store-std --no-fail-fast"

    # proxima-primitives is the one workspace crate that consumes the schema
    # module directly (pipe/validate.rs), so a change to the IR that compiles
    # here can still break there.
    "primitives still builds on it|cargo check -p proxima-primitives --all-targets"
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

printf '\n== proxima-config-gate summary ==\n'
printf '   passed: %d\n' "$passed"
printf '   failed: %d\n' "$failed"

if [ "$failed" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "$label"
    done
    exit 1
fi

printf '\nproxima-config-gate: all green.\n'
