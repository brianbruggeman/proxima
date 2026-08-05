#!/usr/bin/env bash
# proxima-storage-gate.sh — feature-matrix gate for proxima-storage.
#
# Why it exists: the crate is `default = []`, so `cargo nextest run -p
# proxima-storage` and `cargo clippy -p proxima-storage --all-targets` see the
# `pmem` module and nothing else. Measured 2026-08-05 on the pre-audit tree:
# the default command ran 21 tests, all of them `pmem::cow::tests`; the six
# `nvme::engine` tests and the nine `dax` tests never ran, and neither
# `nvme/uio.rs` nor — off Linux — `dax/{region,store}.rs` was ever compiled by
# a crate-scoped command. Three folded-in former crates (proxima-nvme,
# proxima-pmem, proxima-pmem-dax) with two thirds of them invisible to the
# commands the crate documents.
#
# Each feature also gets a cell of its OWN, not just a place in the
# --all-features union: a feature that only ever compiles unified with its
# siblings is a feature nobody has proven.
#
# The two no_std tiers are NOT duplicated here — scripts/thumbv7m-cliff-gate.sh
# carries the `proxima-storage-bare-no-alloc` and `proxima-storage-nvme` cells
# (guiding-principle 1: one source, no forked copies). The host-side cells
# below are the cheap pre-check that gate then proves on the cliff.
#
# No doctest cell: the crate has zero ``` fences, so a `cargo test --doc` cell
# would be the zero-passed-reads-as-all-passed trap AGENTS.md warns about. The
# runnable example below is what stands in for it — `cargo check --all-targets`
# compiles examples but never runs them.
#
# Usage:  bash scripts/proxima-storage-gate.sh
# Exits 0 if every cell passes, non-zero (after running them all) otherwise.

set -euo pipefail

declare -a cells=(
    # what the crate's own documented commands cover — the pmem leaf alone
    "default check|cargo check -p proxima-storage --all-targets"
    "default tests|cargo nextest run -p proxima-storage --no-fail-fast"
    "default clippy|cargo clippy -p proxima-storage --all-targets -- -D warnings"
    "default rustdoc|cargo doc -p proxima-storage --no-deps"

    # the union — every folded module compiled and tested together
    "all-features tests|cargo nextest run -p proxima-storage --all-features --no-fail-fast"
    "all-features clippy|cargo clippy -p proxima-storage --all-targets --all-features -- -D warnings"
    "all-features rustdoc|cargo doc -p proxima-storage --no-deps --all-features"

    # each feature ALONE — unification must not be what makes it compile
    "std alone clippy|cargo clippy -p proxima-storage --all-targets --no-default-features --features std -- -D warnings"
    "nvme alone clippy|cargo clippy -p proxima-storage --all-targets --no-default-features --features nvme -- -D warnings"
    "nvme+std tests|cargo nextest run -p proxima-storage --no-default-features --features nvme,std --no-fail-fast"
    "nvme-uio alone clippy|cargo clippy -p proxima-storage --all-targets --no-default-features --features nvme-uio -- -D warnings"
    "dax alone clippy|cargo clippy -p proxima-storage --all-targets --no-default-features --features dax -- -D warnings"
    "dax alone tests|cargo nextest run -p proxima-storage --no-default-features --features dax --no-fail-fast"

    # examples are compiled by --all-targets and never RUN by it. cow_walkthrough
    # drives every UpdateState transition and asserts the recovered value, so
    # running it is a real check; uio_rw needs an NVMe controller bound to
    # uio_pci_generic and is deliberately left to the hardware run.
    "cow_walkthrough example RUNS|cargo run -p proxima-storage --example cow_walkthrough"
)

passed=0
failed=0
declare -a failures

for cell in "${cells[@]}"; do
    label="${cell%%|*}"
    command="${cell#*|}"

    printf '\n== %s ==\n%s\n' "$label" "$command"

    if eval "$command"; then
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
        failures+=("$label")
    fi
done

printf '\n== proxima-storage-gate summary ==\n'
printf '   passed: %d\n' "$passed"
printf '   failed: %d\n' "$failed"

if [ "$failed" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "$label"
    done
    exit 1
fi

printf '\nproxima-storage-gate: all green.\n'
