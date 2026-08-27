#!/usr/bin/env bash
# rekt-gate.sh
# Mechanical gate for tools/rekt (the load tester on proxima: the engine,
# scenario, and report modules, plus the scheduler/client-protocol features
# that pull proxima in). No gate script existed for this crate before --
# `cargo nextest run -p rekt` and `cargo test --doc` both exit 0 whether
# they ran real work or nothing, so this asserts the nonzero count
# explicitly instead of trusting the exit code alone.
#
# `default = []` here by design (the "default-off firewall": a plain
# `cargo build`/`cargo test` stays free of the whole proxima tree). rekt's
# --all-features union does NOT touch proxima-net's `dpdk` feature (rekt's
# own [features] table never names it, and cargo --all-features only
# enables the crate's OWN declared features), so this is safe on a host
# without a dpdk toolchain -- unlike scripts/proxima-net-gate.sh, which must
# exclude dpdk explicitly because it IS a feature proxima-net itself
# declares.
#
# usage: bash scripts/rekt-gate.sh

set -euo pipefail

crate="rekt"
pkg="-p ${crate}"

printf '\n== %s gate ==\n' "${crate}"

printf '\n[1/6] default (empty, tokio-free firewall) feature set compiles\n'
cargo build ${pkg} --all-targets

printf '\n[2/6] all-features (scheduler + tokio-compare + all-client-protocols) build\n'
cargo build ${pkg} --all-targets --all-features

printf '\n[3/6] tests green (default is tokio-free so has no test surface of its own;\n'
printf '      all-features exercises engine/scenario/report), count asserted\n'
nextest_output="$(cargo nextest run ${pkg} --all-features --no-fail-fast 2>&1)"
printf '%s\n' "${nextest_output}"
ran_count="$(printf '%s\n' "${nextest_output}" | grep -oE '[0-9]+ tests run:' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${ran_count}" ] || [ "${ran_count}" -eq 0 ]; then
    printf 'ERROR: %s nextest reported zero tests run -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   tests run: %s\n' "${ran_count}"

printf '\n[4/6] clippy clean (default + all-features; this crate denies warnings +\n'
printf '      unwrap/expect at [lints] itself, so this is largely re-proving the\n'
printf '      crates own lint gate under -D warnings too)\n'
cargo clippy ${pkg} --all-targets -- -D warnings
cargo clippy ${pkg} --all-targets --all-features -- -D warnings

printf '\n[5/6] rustdoc resolves (default + all-features)\n'
cargo doc ${pkg} --no-deps
cargo doc ${pkg} --no-deps --all-features

# this crate's src/ carries zero ``` fences (verified by
# `grep -rc '```' tools/rekt/src | awk -F: '{sum+=$2} END{print sum+0}'` = 0)
# -- "0 expected, 0 found" is a real, explicit assertion. If a doctest is
# ever added, swap this for the standard nonzero-count assertion (see
# scripts/proxima-clock-gate.sh's last step) or it will silently stop
# proving the new doctest runs.
printf '\n[6/6] doctests: 0 fences in src/ -- asserting 0 expected, 0 found\n'
doctest_output="$(cargo test --doc ${pkg} --all-features 2>&1)"
printf '%s\n' "${doctest_output}"
if ! printf '%s\n' "${doctest_output}" | grep -qE '^test result: ok\. 0 passed'; then
    printf 'ERROR: %s doctest run no longer matches the expected 0-fence baseline -- update this gate\n' "${crate}" >&2
    exit 1
fi
printf '   doctests: 0 expected, 0 found (matches src/ fence count)\n'

printf '\n== %s gate: PASS ==\n' "${crate}"
