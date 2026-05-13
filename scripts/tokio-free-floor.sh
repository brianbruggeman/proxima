#!/usr/bin/env bash
# tokio-free-floor.sh -- CI gate: the no_std + alloc floor tier carries
# zero transitive tokio, for every crate docs/pipe-to-metal/edges.md's
# "tokio/futures compat-layer sweep" scoping pass (2026-07-16) names
# clean (prime, proxima-primitives, proxima-net, proxima-runtime,
# proxima-protocols, proxima-core).
#
# Mechanism generalized, not reinvented (guiding-principle 1 / RISC):
# the same `cargo tree -p <crate> ... -i tokio` + "nothing to print"
# shape already proven in .github/workflows/proxima-pgwire.yml's
# tokio-gate job and proxima-redis.yml's equivalent, and already
# drafted (disabled) in scripts/quic-h3-gate.sh's TOKIO_FREE_FACADE_ENFORCE
# block -- this script is that same shape, matrixed over the floor
# crate set instead of one crate.
#
# Locks in an ALREADY-clean invariant (P16 proof substrate): every
# cell below is expected to pass today; a future regression (a floor
# crate's `alloc` feature accidentally reaching tokio) fails CI
# immediately instead of drifting unnoticed.
#
# Usage:
#   bash scripts/tokio-free-floor.sh
#
# Exits 0 if every cell's floor-tier dependency graph is tokio-free,
# non-zero (with the offending `cargo tree` output) otherwise.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_floor-crate-matrix.sh
source "${script_dir}/_floor-crate-matrix.sh"

passed=0
failed=0
declare -a failures

# the no_std+alloc floor cells PLUS the std-tier cells that are tokio-free-
# checkable but not thumbv7m-buildable (see _floor-crate-matrix.sh) -- this
# gate asks "is it tokio-free?", meaningful at BOTH tiers; the cliff gate asks
# "does it compile bare-metal?", meaningful only for the floor, so it iterates
# FLOOR_CRATE_CELLS alone.
for cell in "${FLOOR_CRATE_CELLS[@]}" "${TOKIO_FREE_EXTRA_CELLS[@]}"; do
    label="${cell%%|*}"
    rest="${cell#*|}"
    crate="${rest%%|*}"
    features="${rest#*|}"

    printf '\n== %s (crate=%s, features=%s) ==\n' "$label" "$crate" "$features"

    leaked="$(cargo tree -p "$crate" --no-default-features --features "$features" -e no-dev -i tokio 2>/dev/null || true)"
    if printf '%s' "$leaked" | grep -q '^tokio'; then
        printf 'FAIL: tokio leaked into %s at --no-default-features --features %s:\n%s\n' "$crate" "$features" "$leaked"
        failed=$((failed + 1))
        failures+=("$label")
    else
        printf 'ok: no tokio in %s --no-default-features --features %s\n' "$crate" "$features"
        passed=$((passed + 1))
    fi
done

printf '\n== tokio-free-floor summary ==\n'
printf '   passed: %d\n' "$passed"
printf '   failed: %d\n' "$failed"

if [ "$failed" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "$label"
    done
    exit 1
fi

printf '\ntokio-free-floor: all green.\n'
