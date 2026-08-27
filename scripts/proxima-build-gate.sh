#!/usr/bin/env bash
# proxima-build-gate.sh
# Mechanical gate for proxima-build (the build.rs helper crate: conflaguration
# profile resolution + generated cfg module). No gate script existed for this
# crate before -- `cargo nextest run -p proxima-build` and `cargo test --doc`
# both exit 0 whether they ran real work or nothing, so this asserts the
# nonzero count explicitly instead of trusting the exit code alone.
#
# usage: bash scripts/proxima-build-gate.sh

set -euo pipefail

crate="proxima-build"

printf '\n== %s gate ==\n' "${crate}"

printf '\n[1/5] crate + bin (dump-profile) build clean\n'
cargo build -p "${crate}" --all-targets

printf '\n[2/5] tests green, count asserted\n'
nextest_output="$(cargo nextest run -p "${crate}" --no-fail-fast 2>&1)"
printf '%s\n' "${nextest_output}"
ran_count="$(printf '%s\n' "${nextest_output}" | grep -oE '[0-9]+ tests run:' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${ran_count}" ] || [ "${ran_count}" -eq 0 ]; then
    printf 'ERROR: %s nextest reported zero tests run -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   tests run: %s\n' "${ran_count}"

printf '\n[3/5] clippy pedantic clean\n'
cargo clippy -p "${crate}" --all-targets -- -D warnings

printf '\n[4/5] rustdoc resolves\n'
cargo doc -p "${crate}" --no-deps

# `cargo test --doc` exits 0 on a vacuous "0 passed" run, so grep for a
# nonzero count explicitly instead of trusting the exit code alone.
printf '\n[5/5] doctests, count asserted\n'
doctest_output="$(cargo test --doc -p "${crate}" 2>&1)"
printf '%s\n' "${doctest_output}"
passed_count="$(printf '%s\n' "${doctest_output}" | grep -oE '^test result: ok\. [0-9]+ passed' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${passed_count}" ] || [ "${passed_count}" -eq 0 ]; then
    printf 'ERROR: %s doctests reported zero passed -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   doctests passed: %s\n' "${passed_count}"

printf '\n== %s gate: PASS ==\n' "${crate}"
