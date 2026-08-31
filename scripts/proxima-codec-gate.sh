#!/usr/bin/env bash
# proxima-codec-gate.sh
# Mechanical gate for proxima-codec (the pluggable codec registry: FrameCodec/
# MessageCodec/Datagram traits at the no_std+alloc tier, JsonCodec + the
# registry/factory at std). No gate script existed for this crate before --
# `cargo nextest run -p proxima-codec` and `cargo test --doc` both exit 0
# whether they ran real work or nothing, so this asserts the nonzero count
# explicitly instead of trusting the exit code alone.
#
# usage: bash scripts/proxima-codec-gate.sh

set -euo pipefail

# CI sets CARGO_TERM_COLOR=always; ANSI escapes wrapped around digits break
# any grep/awk that counts cargo/nextest summary output (the class
# proxima-tokenizer-gate.sh and proxima-test-gate.sh hit) -- force it off
# for every invocation in this script, present and future.
export CARGO_TERM_COLOR=never

crate="proxima-codec"

printf '\n== %s gate ==\n' "${crate}"

# the no_std tier itself is NOT proven here -- a host build links libstd
# regardless. This step only proves the bare (alloc-only) feature set still
# compiles; scripts/thumbv7m-cliff-gate.sh's `proxima-codec` cell owns the
# bare-metal proof.
printf '\n[1/6] no_std + alloc floor compiles (no default features)\n'
cargo build -p "${crate}" --no-default-features --features alloc

printf '\n[2/6] default (std) feature set compiles\n'
cargo build -p "${crate}" --all-targets

printf '\n[3/6] default (std) tests green, count asserted\n'
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

# `cargo test --doc` exits 0 on a vacuous "0 passed" run, so grep for a
# nonzero count explicitly instead of trusting the exit code alone.
printf '\n[6/6] doctests, count asserted\n'
doctest_output="$(cargo test --doc -p "${crate}" 2>&1)"
printf '%s\n' "${doctest_output}"
passed_count="$(printf '%s\n' "${doctest_output}" | grep -oE '^test result: ok\. [0-9]+ passed' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${passed_count}" ] || [ "${passed_count}" -eq 0 ]; then
    printf 'ERROR: %s doctests reported zero passed -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   doctests passed: %s\n' "${passed_count}"

printf '\n== %s gate: PASS ==\n' "${crate}"
