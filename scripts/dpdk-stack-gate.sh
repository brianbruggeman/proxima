#!/usr/bin/env bash
# dpdk-stack-gate.sh
# Disciplined-component gate for the DPDK userspace-stack sans-IO layer.
# Re-proves the correctness claims from scratch (guiding-principle 16), with no
# one's memory: the L2-L4 wire codec (RFC 1071 checksum worked examples, eth/
# ipv4/tcp/udp parse+build) and the RFC 793 connection control FSM (the Fig 6
# transition table as worked-example tests) — plus the hard contract that both
# modules hold no_std + no-alloc on a bare-metal target.
#
# usage: bash scripts/dpdk-stack-gate.sh
#
# proxima-inet-codec and proxima-tcp folded into proxima-protocols as the
# `inet` and `tcp` features (protocols-fold); steps, per feature:
#   1. default build + clippy --all-targets -D warnings (pedantic, workspace lints)
#   2. default test suite (the RFC worked examples ARE the tests)
#   3. bare-metal no_std + no-alloc build (--no-default-features, thumbv7em)
#
# this script never modifies a discipline log; sealing a row is a manual read.

set -euo pipefail

crate="proxima-protocols"
features=(inet tcp)
bare_target="thumbv7em-none-eabihf"

printf '\n== dpdk-stack sans-IO gate ==\n'

for feature in "${features[@]}"; do
    printf '\n-- %s (feature=%s): build + clippy --\n' "${crate}" "${feature}"
    cargo build -p "${crate}" --no-default-features --features "${feature}"
    cargo clippy -p "${crate}" --no-default-features --features "${feature}" --all-targets -- -D warnings

    printf '\n-- %s (feature=%s): test suite (worked examples) --\n' "${crate}" "${feature}"
    cargo nextest run -p "${crate}" --no-default-features --features "${feature}"

    printf '\n-- %s (feature=%s): bare-metal no_std + no-alloc (%s) --\n' "${crate}" "${feature}" "${bare_target}"
    cargo build -p "${crate}" --no-default-features --features "${feature}" --target "${bare_target}"
done

printf '\n== dpdk-stack gate: all green ==\n'
