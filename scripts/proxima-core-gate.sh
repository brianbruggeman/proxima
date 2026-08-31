#!/usr/bin/env bash
# proxima-core-gate.sh
# Mechanical gate for proxima-core (foundation primitives: error, ring,
# arena, buffer pool, the io-async seam, registry/config/live, the time
# tiers). No gate script existed for this crate before -- `cargo nextest run
# -p proxima-core` and `cargo test --doc` both exit 0 whether they ran real
# work or nothing, so this asserts the nonzero count explicitly instead of
# trusting the exit code alone. Most of the crate's surface (io-async,
# io-async-compat, registry, config, live, park) is default-off, so a
# default-feature run alone misses most of it -- mirrors the
# proxima-config-gate.sh rationale.
#
# usage: bash scripts/proxima-core-gate.sh

set -euo pipefail

# CI sets CARGO_TERM_COLOR=always; ANSI escapes wrapped around digits break
# any grep/awk that counts cargo/nextest summary output (the class
# proxima-tokenizer-gate.sh and proxima-test-gate.sh hit) -- force it off
# for every invocation in this script, present and future.
export CARGO_TERM_COLOR=never

crate="proxima-core"

printf '\n== %s gate ==\n' "${crate}"

printf '\n[1/7] no_std + alloc floor compiles (no default features)\n'
cargo build -p "${crate}" --no-default-features --features alloc

printf '\n[2/7] default (std) feature set compiles\n'
cargo build -p "${crate}" --all-targets

printf '\n[3/7] default (std) tests green, count asserted\n'
nextest_output="$(cargo nextest run -p "${crate}" --no-fail-fast 2>&1)"
printf '%s\n' "${nextest_output}"
ran_count="$(printf '%s\n' "${nextest_output}" | grep -oE '[0-9]+ tests run:' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${ran_count}" ] || [ "${ran_count}" -eq 0 ]; then
    printf 'ERROR: %s nextest reported zero tests run -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   tests run: %s\n' "${ran_count}"

# --all-features turns on loom too, which is exactly what tests/loom_signal.rs
# (required-features = ["loom", "std"]) needs to even compile -- a feature
# that only ever compiles unified with its siblings is a feature nobody has
# proven, so this is the ONLY place that test target runs.
printf '\n[4/7] all-features (registry/config/live/io-async*/park/loom) build + tests, count asserted\n'
all_output="$(cargo nextest run -p "${crate}" --all-features --no-fail-fast 2>&1)"
printf '%s\n' "${all_output}"
all_count="$(printf '%s\n' "${all_output}" | grep -oE '[0-9]+ tests run:' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${all_count}" ] || [ "${all_count}" -eq 0 ]; then
    printf 'ERROR: %s all-features nextest reported zero tests run -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   all-features tests run: %s\n' "${all_count}"

printf '\n[5/7] clippy pedantic clean (bare alloc + default std + all-features)\n'
cargo clippy -p "${crate}" --all-targets --no-default-features --features alloc -- -D warnings
cargo clippy -p "${crate}" --all-targets -- -D warnings
cargo clippy -p "${crate}" --all-targets --all-features -- -D warnings

printf '\n[6/7] rustdoc resolves (bare alloc + default std + all-features)\n'
cargo doc -p "${crate}" --no-deps --no-default-features --features alloc
cargo doc -p "${crate}" --no-deps
cargo doc -p "${crate}" --no-deps --all-features

# `cargo test --doc` exits 0 on a vacuous "0 passed" run, so grep for a
# nonzero count explicitly instead of trusting the exit code alone.
printf '\n[7/7] doctests (all-features), count asserted\n'
doctest_output="$(cargo test --doc -p "${crate}" --all-features 2>&1)"
printf '%s\n' "${doctest_output}"
passed_count="$(printf '%s\n' "${doctest_output}" | grep -oE '^test result: ok\. [0-9]+ passed' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${passed_count}" ] || [ "${passed_count}" -eq 0 ]; then
    printf 'ERROR: %s doctests reported zero passed -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   doctests passed: %s\n' "${passed_count}"

printf '\n== %s gate: PASS ==\n' "${crate}"
