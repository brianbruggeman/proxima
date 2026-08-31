#!/usr/bin/env bash
# proxima-macros-gate.sh
# Mechanical gate for proxima-macros (proc-macro support: #[proxima::test],
# runtime entrypoints, telemetry spans, schema description, error derives).
# No gate script existed for this crate before -- `cargo nextest run -p
# proxima-macros` and `cargo test --doc` both exit 0 whether they ran real
# work or nothing, so this asserts the nonzero count explicitly instead of
# trusting the exit code alone.
#
# usage: bash scripts/proxima-macros-gate.sh

set -euo pipefail

# CI sets CARGO_TERM_COLOR=always; ANSI escapes wrapped around digits break
# any grep/awk that counts cargo/nextest summary output (the class
# proxima-tokenizer-gate.sh and proxima-test-gate.sh hit) -- force it off
# for every invocation in this script, present and future.
export CARGO_TERM_COLOR=never

crate="proxima-macros"

printf '\n== %s gate ==\n' "${crate}"

printf '\n[1/5] proc-macro crate + trybuild fixtures build clean\n'
cargo build -p "${crate}" --all-targets

printf '\n[2/5] tests green (unit tests + trybuild pass/fail fixtures), count asserted\n'
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

# cargo 1.98's merged-doctest support runs `proc-macro = true` crate
# doctests against a downstream token stream (each fence expands the macro
# for real, e.g. via the `proxima` dev-dependency), so this is no longer the
# structural zero it once was -- assert the nonzero count explicitly, same
# shape every other crate in this fleet uses.
printf '\n[5/5] doctests: proc-macro crate doctests run via merged doctests, count asserted\n'
doctest_output="$(cargo test --doc -p "${crate}" 2>&1)"
printf '%s\n' "${doctest_output}"
doctest_count="$(printf '%s\n' "${doctest_output}" | grep -oE '^test result: ok\. [0-9]+ passed' | grep -oE '[0-9]+' | awk '{sum+=$1} END{print sum+0}')"
if [ -z "${doctest_count}" ] || [ "${doctest_count}" -eq 0 ]; then
    printf 'ERROR: %s doctest run reported zero doctests passed -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   doctests: %s passed\n' "${doctest_count}"

printf '\n== %s gate: PASS ==\n' "${crate}"
