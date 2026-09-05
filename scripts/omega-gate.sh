#!/usr/bin/env bash
# omega-gate.sh
# Mechanical gate for omega (MSL kernel emission from the proxima-tensor
# BoundOp descriptor -- the GPU half of the bound-addressing seam, plus the
# macOS-only Metal execution driver behind the `metal` feature). No gate
# script existed for this crate before -- `cargo nextest run -p omega` and
# `cargo test --doc` both exit 0 whether they ran real work or nothing, so
# this asserts the nonzero count explicitly instead of trusting the exit
# code alone.
#
# `metal` links objc2/objc2-metal/objc2-foundation and is gated
# `target_os = "macos"` internally; this host is Darwin, so --all-features
# is provable here and is exercised directly (unlike proxima-net's dpdk or
# proxima-tensor's ggml-bench, which need external toolchains this host
# does not have).
#
# usage: bash scripts/omega-gate.sh

set -euo pipefail

# CI sets CARGO_TERM_COLOR=always; ANSI escapes wrapped around digits break
# any grep/awk that counts cargo/nextest summary output (the class
# proxima-tokenizer-gate.sh and proxima-test-gate.sh hit) -- force it off
# for every invocation in this script, present and future.
export CARGO_TERM_COLOR=never

crate="omega"

printf '\n== %s gate ==\n' "${crate}"

# Prints the count of rustc invocations FOR THIS CRATE under the given
# feature flags -- a tier build that compiles zero of the crate's own
# modules (every module gated off) would still exit 0, which reads exactly
# like a real build. N==0 here is the same false-green shape this script's
# own module doc warns about for nextest/doctest counts.
assert_tier_builds() {
    local label="$1"
    shift
    # forces a real rustc invocation for THIS crate every run -- an
    # incremental cache hit from a prior identical-flags build would leave
    # zero "Running" lines below and read as a false N==0.
    cargo clean -p "${crate}" > /dev/null 2>&1
    local verbose_output
    verbose_output="$(cargo build -p "${crate}" "$@" -v 2>&1)"
    printf '%s\n' "${verbose_output}"
    local rustc_count
    rustc_count="$(printf '%s\n' "${verbose_output}" | grep -c "Running \`.*rustc.*${crate}")"
    if [ "${rustc_count}" -eq 0 ]; then
        printf 'ERROR: %s tier build compiled zero rustc invocations for %s\n' "${label}" "${crate}" >&2
        exit 1
    fi
    printf '   %s tier: rustc invocations for %s = %s\n' "${label}" "${crate}" "${rustc_count}"
}

printf '\n[1/8] bare tier compiles (no_std, no alloc -- error + sized only)\n'
assert_tier_builds bare --no-default-features

printf '\n[2/8] no_std + alloc tier compiles (adds msl emission)\n'
assert_tier_builds alloc --no-default-features --features alloc

printf '\n[3/8] std tier compiles (adds backend, no concrete driver)\n'
assert_tier_builds std --no-default-features --features std

printf '\n[4/8] all-features (std + metal) build\n'
cargo build -p "${crate}" --all-targets --all-features

printf '\n[5/8] all-features tests green, count asserted\n'
nextest_output="$(cargo nextest run -p "${crate}" --all-features --no-fail-fast 2>&1)"
printf '%s\n' "${nextest_output}"
ran_count="$(printf '%s\n' "${nextest_output}" | grep -oE '[0-9]+ tests run:' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${ran_count}" ] || [ "${ran_count}" -eq 0 ]; then
    printf 'ERROR: %s nextest reported zero tests run -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   tests run: %s\n' "${ran_count}"

printf '\n[6/8] clippy pedantic clean (bare alloc + all-features)\n'
# --lib only for the bare-alloc arm: the integration tests under omega/tests/
# exercise the Metal execution driver and assume `metal` is present (it is a
# default feature -- see the note at the bottom of
# omega/tests/metal_compile_gate.rs), so they are not part of the alloc-tier
# claim this crate's own module docs make ("alloc: the whole crate" refers to
# the emission library, not the Metal-only integration suite). --all-targets
# here would fail to compile those tests for lacking `metal`, which is a
# feature-scope mismatch in the gate, not a source defect.
cargo clippy -p "${crate}" --lib --no-default-features --features alloc -- -D warnings
cargo clippy -p "${crate}" --all-targets --all-features -- -D warnings

printf '\n[7/8] rustdoc resolves (bare alloc + all-features)\n'
cargo doc -p "${crate}" --no-deps --no-default-features --features alloc
cargo doc -p "${crate}" --no-deps --all-features

# `cargo test --doc` exits 0 on a vacuous "0 passed" run, so grep for a
# nonzero count explicitly instead of trusting the exit code alone.
printf '\n[8/8] doctests (all-features), count asserted\n'
doctest_output="$(cargo test --doc -p "${crate}" --all-features 2>&1)"
printf '%s\n' "${doctest_output}"
passed_count="$(printf '%s\n' "${doctest_output}" | grep -oE '^test result: ok\. [0-9]+ passed' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${passed_count}" ] || [ "${passed_count}" -eq 0 ]; then
    printf 'ERROR: %s doctests reported zero passed -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   doctests passed: %s\n' "${passed_count}"

printf '\n== %s gate: PASS ==\n' "${crate}"
