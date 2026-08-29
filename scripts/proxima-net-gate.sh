#!/usr/bin/env bash
# proxima-net-gate.sh
# Mechanical gate for proxima-net (UDP PacketListener + addressing helpers,
# plus every platform network backend -- prime/tokio/wasm/dpdk/xdp -- as a
# feature-gated module). No gate script existed for this crate before --
# `cargo nextest run -p proxima-net` and `cargo test --doc` both exit 0
# whether they ran real work or nothing, so this asserts the count
# explicitly instead of trusting the exit code alone -- nonzero for
# nextest (real tests exist), and the documented 0-fence baseline for the
# doctest step (this crate's src/ carries no runnable doc fences).
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

# `cargo test --doc` exits 0 on a vacuous "0 passed" run, so this asserts
# the count explicitly rather than trusting the exit code alone. This
# crate's src/ carries zero ```rust fences under the non-dpdk union
# (verified by `git grep -c '```rust' -- proxima-net/src` = 0; the only
# fences present are the ```text pair in wasm/mod.rs, which rustdoc never
# runs as a doctest) -- "0 expected, 0 found" is therefore the real,
# explicit assertion, same as scripts/proxima-vm-gate.sh's own zero-fence
# step. If a runnable doc fence is ever added, swap this for the standard
# nonzero-count assertion (see scripts/proxima-clock-gate.sh's last step)
# or it will silently stop proving the new doctest runs.
printf '\n[6/6] doctests: 0 rust fences in src/ -- asserting 0 expected, 0 found\n'
doctest_output="$(cargo test --doc -p "${crate}" --no-default-features --features "${non_dpdk_features}" 2>&1)"
printf '%s\n' "${doctest_output}"
if ! printf '%s\n' "${doctest_output}" | grep -qE '^test result: ok\. 0 passed'; then
    printf 'ERROR: %s doctest run no longer matches the expected 0-fence baseline -- update this gate\n' "${crate}" >&2
    exit 1
fi
printf '   doctests: 0 expected, 0 found (matches src/ fence count)\n'

printf '\n== %s gate: PASS ==\n' "${crate}"
