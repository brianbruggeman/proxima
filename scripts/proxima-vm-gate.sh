#!/usr/bin/env bash
# proxima-vm-gate.sh
# Mechanical gate for tools/proxima-vm (the minimal VM-backed proxima Pipe
# proof surface for scratch guests over KVM / Hypervisor.framework). No gate
# script existed for this crate before -- `cargo nextest run -p proxima-vm`
# and `cargo test --doc` both exit 0 whether they ran real work or nothing,
# so this asserts the nonzero count explicitly instead of trusting the exit
# code alone.
#
# usage: bash scripts/proxima-vm-gate.sh

set -euo pipefail

# CI sets CARGO_TERM_COLOR=always; ANSI escapes wrapped around digits break
# any grep/awk that counts cargo/nextest summary output (the class
# proxima-tokenizer-gate.sh and proxima-test-gate.sh hit) -- force it off
# for every invocation in this script, present and future.
export CARGO_TERM_COLOR=never

crate="proxima-vm"

printf '\n== %s gate ==\n' "${crate}"

printf '\n[1/6] no_std + alloc floor compiles (no default features)\n'
cargo build -p "${crate}" --no-default-features --features alloc

printf '\n[2/6] default (std, `hello` bin) feature set compiles\n'
cargo build -p "${crate}" --all-targets

printf '\n[3/6] default tests green, count asserted\n'
nextest_output="$(cargo nextest run -p "${crate}" --no-fail-fast 2>&1)"
printf '%s\n' "${nextest_output}"
ran_count="$(printf '%s\n' "${nextest_output}" | grep -oE '[0-9]+ tests run:' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${ran_count}" ] || [ "${ran_count}" -eq 0 ]; then
    printf 'ERROR: %s nextest reported zero tests run -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   tests run: %s\n' "${ran_count}"

printf '\n[4/6] clippy pedantic clean (bare alloc + default std)\n'
cargo clippy -p "${crate}" --all-targets --no-default-features --features alloc -- -D warnings
cargo clippy -p "${crate}" --all-targets -- -D warnings

printf '\n[5/6] rustdoc resolves (bare alloc + default std)\n'
cargo doc -p "${crate}" --no-deps --no-default-features --features alloc
cargo doc -p "${crate}" --no-deps

# this crate's src/ carries zero ``` fences (verified by
# `grep -rc '```' tools/proxima-vm/src | awk -F: '{sum+=$2} END{print sum+0}'`
# = 0) -- "0 expected, 0 found" is a real, explicit assertion. If a doctest
# is ever added, swap this for the standard nonzero-count assertion (see
# scripts/proxima-clock-gate.sh's last step) or it will silently stop
# proving the new doctest runs.
printf '\n[6/6] doctests: 0 fences in src/ -- asserting 0 expected, 0 found\n'
doctest_output="$(cargo test --doc -p "${crate}" 2>&1)"
printf '%s\n' "${doctest_output}"
if ! printf '%s\n' "${doctest_output}" | grep -qE '^test result: ok\. 0 passed'; then
    printf 'ERROR: %s doctest run no longer matches the expected 0-fence baseline -- update this gate\n' "${crate}" >&2
    exit 1
fi
printf '   doctests: 0 expected, 0 found (matches src/ fence count)\n'

printf '\n== %s gate: PASS ==\n' "${crate}"
