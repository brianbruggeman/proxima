#!/usr/bin/env bash
# proxima-intercept-gate.sh
# Mechanical gate for proxima-intercept (the MITM interception pipe: TLS
# CONNECT proxy, per-host cert generation, capture/replay, transparent QUIC
# intercept). No gate script existed for this crate before -- `cargo nextest
# run -p proxima-intercept` and `cargo test --doc` both exit 0 whether they
# ran real work or nothing, so this asserts the nonzero count explicitly
# instead of trusting the exit code alone. `default = []` here, so most of
# the crate's own test modules (capture, replay, config, quic_relay) sit
# behind intercept-capture/intercept-replay/intercept-config/quic-intercept
# and are invisible to a default-feature run.
#
# usage: bash scripts/proxima-intercept-gate.sh

set -euo pipefail

# CI sets CARGO_TERM_COLOR=always; ANSI escapes wrapped around digits break
# any grep/awk that counts cargo/nextest summary output (the class
# proxima-tokenizer-gate.sh and proxima-test-gate.sh hit) -- force it off
# for every invocation in this script, present and future.
export CARGO_TERM_COLOR=never

crate="proxima-intercept"

printf '\n== %s gate ==\n' "${crate}"

printf '\n[1/6] default (empty) feature set compiles\n'
cargo build -p "${crate}" --all-targets

printf '\n[2/6] all-features (intercept-capture/replay/config + quic-intercept + delta-tee) build\n'
cargo build -p "${crate}" --all-targets --all-features

printf '\n[3/6] all-features tests green, count asserted\n'
nextest_output="$(cargo nextest run -p "${crate}" --all-features --no-fail-fast 2>&1)"
printf '%s\n' "${nextest_output}"
ran_count="$(printf '%s\n' "${nextest_output}" | grep -oE '[0-9]+ tests run:' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${ran_count}" ] || [ "${ran_count}" -eq 0 ]; then
    printf 'ERROR: %s nextest reported zero tests run -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   tests run: %s\n' "${ran_count}"

printf '\n[4/6] clippy pedantic clean (default + all-features)\n'
cargo clippy -p "${crate}" --all-targets -- -D warnings
cargo clippy -p "${crate}" --all-targets --all-features -- -D warnings

printf '\n[5/6] rustdoc resolves (default + all-features)\n'
cargo doc -p "${crate}" --no-deps
cargo doc -p "${crate}" --no-deps --all-features

# this crate's src/ carries zero ``` fences (verified by
# `grep -rc '```' proxima-intercept/src | awk -F: '{sum+=$2} END{print sum+0}'`
# = 0) -- "0 expected, 0 found" is a real, explicit assertion. If a doctest
# is ever added, `cargo test --doc` starts reporting a nonzero pass count
# and this cell must be swapped for the standard nonzero-count assertion
# (see scripts/proxima-clock-gate.sh's [6/6] for that shape) or it will
# silently stop proving the new doctest runs.
printf '\n[6/6] doctests: 0 fences in src/ -- asserting 0 expected, 0 found\n'
doctest_output="$(cargo test --doc -p "${crate}" --all-features 2>&1)"
printf '%s\n' "${doctest_output}"
if ! printf '%s\n' "${doctest_output}" | grep -qE '^test result: ok\. 0 passed'; then
    printf 'ERROR: %s doctest run no longer matches the expected 0-fence baseline -- update this gate\n' "${crate}" >&2
    exit 1
fi
printf '   doctests: 0 expected, 0 found (matches src/ fence count)\n'

printf '\n== %s gate: PASS ==\n' "${crate}"
