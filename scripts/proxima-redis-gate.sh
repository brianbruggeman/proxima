#!/usr/bin/env bash
# proxima-redis-gate.sh
# Mechanical gate for the proxima-redis stack (docs/proxima-redis/discipline.md).
# Re-proves every discipline-log row from the artifact alone, without anyone's
# memory (guiding-principle 16): the sans-IO RESP codec, the client session FSM
# + config + Pipe, the vendored real-server corpus (parity vs the canonical
# incumbent), and the HARD invariant — the bare sans-IO codec graph carries zero
# tokio (the bare-metal / DPDK embedding contract).
#
# usage: bash scripts/proxima-redis-gate.sh
#
# Live differential parity (real redis:7 + valkey) is a separate CI job with
# service containers; this gate runs everything that needs no server.

set -euo pipefail

crate="proxima-redis"

printf '\n== proxima-redis gate ==\n'

printf '\n[1/7] sans-IO codec builds no_std + alloc (no default features)\n'
cargo build -p "${crate}" --no-default-features

printf '\n[2/7] crate builds clean with the client (all features)\n'
cargo build -p "${crate}" --all-features

printf '\n[3/7] codec + vendored corpus green (the RESP codec now lives in proxima-protocols::redis)\n'
cargo nextest run -p proxima-protocols --no-default-features --features redis --no-fail-fast

printf '\n[4/7] client facade tests green (ClientSession FSM + config + Pipe)\n'
cargo nextest run -p "${crate}" --features client --no-fail-fast

printf '\n[5/7] clippy pedantic clean across the feature matrix\n'
cargo clippy -p "${crate}" --all-targets -- -D warnings
cargo clippy -p "${crate}" --all-targets --features client -- -D warnings
cargo clippy -p "${crate}" --lib --no-default-features -- -D warnings

printf '\n[6/7] TOKIO GATE — the bare sans-IO codec graph must carry zero tokio\n'
leaked="$(cargo tree -p "${crate}" --no-default-features -e normal -i tokio 2>/dev/null || true)"
if printf '%s' "${leaked}" | grep -q '^tokio'; then
    printf '   FAIL: tokio leaked into the no-default-features graph:\n%s\n' "${leaked}" >&2
    exit 1
fi
printf '   ok: no tokio in the bare proxima-redis graph\n'

printf '\n[7/7] bench scaffolding compiles (records data, does NOT seal)\n'
cargo build -p "${crate}" --benches

printf '\n== proxima-redis gate: PASS ==\n'
