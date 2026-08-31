#!/usr/bin/env bash
# thumbv7m-cliff-gate.sh -- CI gate: every crate that claims a no_std +
# alloc floor tier must actually compile on a real embedded target, not
# just under `--no-default-features --features alloc` on the host.
# Mirrors scripts/quic-h3-gate.sh's "thumbv7m cliff" cells (the same
# `cargo build ... --target thumbv7m-none-eabi` shape already proven for
# proxima-protocols' quic/http3_codec modules there) -- generalized over
# the full floor crate set instead of just those two modules
# (guiding-principle 1 / RISC: reuse the pattern, don't reinvent it).
#
# Shares its crate/feature matrix with tokio-free-floor.sh via
# _floor-crate-matrix.sh, so both gates check the identical floor
# definition from two angles (tokio-free dependency graph vs. actually
# compiles on the embedded cliff).
#
# Requires the thumbv7m-none-eabi target installed:
#   rustup target add thumbv7m-none-eabi
#
# Usage:
#   bash scripts/thumbv7m-cliff-gate.sh
#
# Exits 0 if every cell builds clean on thumbv7m-none-eabi, non-zero
# (after running every cell, reporting all failures) otherwise.

set -euo pipefail

# CI sets CARGO_TERM_COLOR=always; ANSI escapes wrapped around digits break
# any grep/awk that counts cargo/nextest summary output (the class
# proxima-tokenizer-gate.sh and proxima-test-gate.sh hit) -- force it off
# for every invocation in this script, present and future.
export CARGO_TERM_COLOR=never

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_floor-crate-matrix.sh
source "${script_dir}/_floor-crate-matrix.sh"

target="thumbv7m-none-eabi"

if ! rustup target list --installed | grep -q "^${target}\$"; then
    printf 'FAIL: %s target not installed. Run: rustup target add %s\n' "$target" "$target" >&2
    exit 1
fi

passed=0
failed=0
declare -a failures

for cell in "${FLOOR_CRATE_CELLS[@]}"; do
    label="${cell%%|*}"
    rest="${cell#*|}"
    crate="${rest%%|*}"
    features="${rest#*|}"

    printf '\n== %s (crate=%s, features=%s, target=%s) ==\n' "$label" "$crate" "$features" "$target"

    if cargo build -p "$crate" --no-default-features --features "$features" --target "$target"; then
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
        failures+=("$label")
    fi
done

printf '\n== thumbv7m-cliff-gate summary ==\n'
printf '   passed: %d\n' "$passed"
printf '   failed: %d\n' "$failed"

if [ "$failed" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "$label"
    done
    exit 1
fi

printf '\nthumbv7m-cliff-gate: all green.\n'
