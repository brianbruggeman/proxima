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

crate="omega"

printf '\n== %s gate ==\n' "${crate}"

printf '\n[1/6] no_std + alloc floor compiles (no default features)\n'
cargo build -p "${crate}" --no-default-features --features alloc

printf '\n[2/6] all-features (std + metal) build\n'
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

printf '\n[4/6] clippy pedantic clean (bare alloc + all-features)\n'
cargo clippy -p "${crate}" --all-targets --no-default-features --features alloc -- -D warnings
cargo clippy -p "${crate}" --all-targets --all-features -- -D warnings

printf '\n[5/6] rustdoc resolves (bare alloc + all-features)\n'
cargo doc -p "${crate}" --no-deps --no-default-features --features alloc
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
