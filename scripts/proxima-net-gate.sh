#!/usr/bin/env bash
# proxima-net-gate.sh
# Mechanical gate for proxima-net (UDP PacketListener + addressing helpers,
# plus every platform network backend -- prime/tokio/wasm/dpdk/xdp -- as a
# feature-gated module). No gate script existed for this crate before --
# `cargo nextest run -p proxima-net` and `cargo test --doc` both exit 0
# whether they ran real work or nothing, so this asserts the nonzero count
# explicitly instead of trusting the exit code alone.
#
# `dpdk` is DELIBERATELY excluded from every feature set below: its build.rs
# shells out to `pkg-config --silence-errors <flag> libdpdk` and asserts the
# call succeeds (build.rs:159-167), so `--features dpdk` (and therefore
# `--all-features`) cannot build on a host without a dpdk toolchain -- this
# box does not have one. `cargo nextest run --workspace --all-features` is
# broken workspace-wide for exactly this reason; this gate proves every
# feature this host CAN build instead of pretending the union is provable
# here. dpdk is proven by its own dedicated host / CI lane, out of scope for
# this gate.
#
# usage: bash scripts/proxima-net-gate.sh

set -euo pipefail

crate="proxima-net"
non_dpdk_features="prime,runtime-prime-inbox-alloc,tokio,wasm,xdp"

printf '\n== %s gate ==\n' "${crate}"

printf '\n[1/6] default (std) feature set compiles\n'
cargo build -p "${crate}" --all-targets

printf '\n[2/6] every non-dpdk feature union builds (prime/tokio/wasm/xdp)\n'
cargo build -p "${crate}" --all-targets --no-default-features --features "${non_dpdk_features}"

printf '\n[3/6] non-dpdk feature union tests green, count asserted\n'
nextest_output="$(cargo nextest run -p "${crate}" --no-default-features --features "${non_dpdk_features}" --no-fail-fast 2>&1)"
printf '%s\n' "${nextest_output}"
ran_count="$(printf '%s\n' "${nextest_output}" | grep -oE '[0-9]+ tests run:' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${ran_count}" ] || [ "${ran_count}" -eq 0 ]; then
    printf 'ERROR: %s nextest reported zero tests run -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   tests run: %s\n' "${ran_count}"

printf '\n[4/6] clippy pedantic clean (default + non-dpdk union)\n'
cargo clippy -p "${crate}" --all-targets -- -D warnings
cargo clippy -p "${crate}" --all-targets --no-default-features --features "${non_dpdk_features}" -- -D warnings

printf '\n[5/6] rustdoc resolves (default + non-dpdk union)\n'
cargo doc -p "${crate}" --no-deps
cargo doc -p "${crate}" --no-deps --no-default-features --features "${non_dpdk_features}"

# `cargo test --doc` exits 0 on a vacuous "0 passed" run, so grep for a
# nonzero count explicitly instead of trusting the exit code alone.
printf '\n[6/6] doctests (non-dpdk union), count asserted\n'
doctest_output="$(cargo test --doc -p "${crate}" --no-default-features --features "${non_dpdk_features}" 2>&1)"
printf '%s\n' "${doctest_output}"
passed_count="$(printf '%s\n' "${doctest_output}" | grep -oE '^test result: ok\. [0-9]+ passed' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${passed_count}" ] || [ "${passed_count}" -eq 0 ]; then
    printf 'ERROR: %s doctests reported zero passed -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   doctests passed: %s\n' "${passed_count}"

printf '\n== %s gate: PASS ==\n' "${crate}"
