#!/usr/bin/env bash
# proxima-clock-gate.sh
# Mechanical gate for proxima-clock (tick sources, wall-clock anchoring, the
# shared-time cell, all expressed as proxima_primitives pipes). No gate
# script existed for this crate before -- `cargo nextest run -p
# proxima-clock` and `cargo test --doc` both exit 0 whether they ran real
# work or nothing, so this asserts the nonzero count explicitly instead of
# trusting the exit code alone.
#
# usage: bash scripts/proxima-clock-gate.sh

set -euo pipefail

crate="proxima-clock"

printf '\n== %s gate ==\n' "${crate}"

printf '\n[1/6] default (std) feature set compiles\n'
cargo build -p "${crate}" --all-targets

printf '\n[2/6] default tests green, count asserted\n'
nextest_output="$(cargo nextest run -p "${crate}" --no-fail-fast 2>&1)"
printf '%s\n' "${nextest_output}"
ran_count="$(printf '%s\n' "${nextest_output}" | grep -oE '[0-9]+ tests run:' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${ran_count}" ] || [ "${ran_count}" -eq 0 ]; then
    printf 'ERROR: %s nextest reported zero tests run -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   tests run: %s\n' "${ran_count}"

printf "\n[3/6] all-features (adds 'config': bon + conflaguration + serde) build + tests, count asserted\n"
all_output="$(cargo nextest run -p "${crate}" --all-features --no-fail-fast 2>&1)"
printf '%s\n' "${all_output}"
all_count="$(printf '%s\n' "${all_output}" | grep -oE '[0-9]+ tests run:' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${all_count}" ] || [ "${all_count}" -eq 0 ]; then
    printf 'ERROR: %s all-features nextest reported zero tests run -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   all-features tests run: %s\n' "${all_count}"

printf '\n[4/6] clippy pedantic clean (default + all-features)\n'
cargo clippy -p "${crate}" --all-targets -- -D warnings
cargo clippy -p "${crate}" --all-targets --all-features -- -D warnings

printf '\n[5/6] rustdoc resolves\n'
cargo doc -p "${crate}" --no-deps
cargo doc -p "${crate}" --no-deps --all-features

# `cargo test --doc` exits 0 on a vacuous "0 passed" run, so grep for a
# nonzero count explicitly instead of trusting the exit code alone.
printf '\n[6/6] doctests (all-features), count asserted\n'
doctest_output="$(cargo test --doc -p "${crate}" --all-features 2>&1)"
printf '%s\n' "${doctest_output}"
passed_count="$(printf '%s\n' "${doctest_output}" | grep -oE '^test result: ok\. [0-9]+ passed' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${passed_count}" ] || [ "${passed_count}" -eq 0 ]; then
    printf 'ERROR: %s doctests reported zero passed -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   doctests passed: %s\n' "${passed_count}"

printf '\n== %s gate: PASS ==\n' "${crate}"
