#!/usr/bin/env bash
# proxima-redis-gate.sh
# Mechanical gate for the proxima-redis stack.
# Re-proves every discipline-log row from the artifact alone, without anyone's
# memory (guiding-principle 16): the sans-IO RESP codec, the client session FSM
# + config + Pipe, the sans-IO listener (RedisConnectionPipe/RedisAnyProtocol),
# the vendored real-server corpus (parity vs the canonical incumbent), and the
# HARD invariant — the bare sans-IO codec graph carries zero tokio (the
# bare-metal / DPDK embedding contract).
#
# usage: bash scripts/proxima-redis-gate.sh
#
# Live differential parity (real redis:7 + valkey) is a separate CI job with
# service containers; this gate runs everything that needs no server.

set -euo pipefail

crate="proxima-redis"

printf '\n== proxima-redis gate ==\n'

printf '\n[1/9] sans-IO codec builds no_std + alloc (no default features)\n'
cargo build -p "${crate}" --no-default-features

printf '\n[2/9] crate builds clean with the client (all features)\n'
cargo build -p "${crate}" --all-features

printf '\n[3/9] codec + vendored corpus green (the RESP codec now lives in proxima-protocols::redis)\n'
cargo nextest run -p proxima-protocols --no-default-features --features redis --no-fail-fast

printf '\n[4/9] client facade tests green (ClientSession FSM + config + Pipe)\n'
cargo nextest run -p "${crate}" --features client --no-fail-fast

# the sans-IO listener (RedisConnectionPipe/RedisAnyProtocol) is a separate
# feature from `client` -- step [2/9]'s --all-features build compiles it, but
# nothing ran its tests until now: a feature-gated suite that compiles but
# never executes is invisible the same way the umbrella proxima crate's
# http1-native determinism proof was (see scripts/prime-serve-gate.sh).
printf '\n[5/9] listener tests green (RedisConnectionPipe/RedisAnyProtocol)\n'
cargo nextest run -p "${crate}" --features listen --no-fail-fast

printf '\n[6/9] clippy pedantic clean across the feature matrix\n'
cargo clippy -p "${crate}" --all-targets -- -D warnings
cargo clippy -p "${crate}" --all-targets --features client -- -D warnings
cargo clippy -p "${crate}" --lib --no-default-features -- -D warnings

printf '\n[7/9] TOKIO GATE — the bare sans-IO codec graph must carry zero tokio\n'
leaked="$(cargo tree -p "${crate}" --no-default-features -e normal -i tokio 2>/dev/null || true)"
if printf '%s' "${leaked}" | grep -q '^tokio'; then
    printf '   FAIL: tokio leaked into the no-default-features graph:\n%s\n' "${leaked}" >&2
    exit 1
fi
printf '   ok: no tokio in the bare proxima-redis graph\n'

printf '\n[8/9] bench scaffolding compiles (records data, does NOT seal)\n'
cargo build -p "${crate}" --benches

# nextest skips doctests by design; the client facade's own examples (e.g.
# the Subscribed compile_fail proof) only run here. `cargo test --doc` exits
# 0 on a vacuous "0 passed" run, so grep for a nonzero count explicitly
# instead of trusting the exit code alone.
printf '\n[9/9] client facade doctests (features=client)\n'
doctest_output="$(cargo test --doc -p "${crate}" --features client 2>&1)"
printf '%s\n' "${doctest_output}"
passed_count="$(printf '%s\n' "${doctest_output}" | grep -oE '^test result: ok\. [0-9]+ passed' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${passed_count}" ] || [ "${passed_count}" -eq 0 ]; then
    printf 'ERROR: %s doctests reported zero passed -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf 'doctests passed: %s\n' "${passed_count}"

printf '\n== proxima-redis gate: PASS ==\n'
