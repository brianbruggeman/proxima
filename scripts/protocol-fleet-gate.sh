#!/usr/bin/env bash
# protocol-fleet-gate.sh
# Mechanical gate for the five protocol-fleet crates: proxima-memcached,
# proxima-dns, proxima-kafka, proxima-mqtt, proxima-amqp. Mirrors
# scripts/proxima-redis-gate.sh's shape, matrixed over the fleet instead of
# a single crate. Re-proves, per crate, without anyone's memory
# (guiding-principle 16): the sans-IO codec builds bare (no_std/alloc tier),
# the all-features build (client + listen) is green, the listener e2e suites
# (gated behind `required-features = ["listen"]` in memcached/kafka/mqtt) run
# under --all-features, clippy is warning-free across that same surface, and
# the HARD invariant — the bare sans-IO codec graph carries zero tokio.
#
# usage:
#   bash scripts/protocol-fleet-gate.sh            # every crate in the fleet
#   bash scripts/protocol-fleet-gate.sh proxima-dns # a single crate

set -euo pipefail

fleet=(proxima-memcached proxima-dns proxima-kafka proxima-mqtt proxima-amqp)
if [[ $# -gt 0 ]]; then
    fleet=("$@")
fi

gate_one() {
    local crate="$1"

    printf '\n== %s gate ==\n' "${crate}"

    printf '\n[1/5] sans-IO codec builds no_std + alloc (no default features)\n'
    cargo build -p "${crate}" --no-default-features

    printf '\n[2/5] crate builds clean with client + listen (all features)\n'
    cargo build -p "${crate}" --all-features

    printf '\n[3/5] codec + client + listener e2e tests green (all features)\n'
    cargo nextest run -p "${crate}" --all-features --no-fail-fast

    printf '\n[4/5] clippy pedantic clean across client + listen\n'
    cargo clippy -p "${crate}" --all-features --all-targets -- -D warnings

    printf '\n[5/5] TOKIO GATE — the bare sans-IO codec graph must carry zero tokio\n'
    local leaked
    leaked="$(cargo tree -p "${crate}" --no-default-features -e normal -i tokio 2>/dev/null || true)"
    if printf '%s' "${leaked}" | grep -q '^tokio'; then
        printf '   FAIL: tokio leaked into the no-default-features graph:\n%s\n' "${leaked}" >&2
        exit 1
    fi
    printf '   ok: no tokio in the bare %s graph\n' "${crate}"

    printf '\n== %s gate: PASS ==\n' "${crate}"
}

for crate in "${fleet[@]}"; do
    gate_one "${crate}"
done

printf '\n== protocol-fleet gate: PASS (%s) ==\n' "${fleet[*]}"
