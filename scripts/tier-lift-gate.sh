#!/usr/bin/env bash
# tier-lift-gate.sh — discipline gate for the no_std tier-lift initiative.
#
# Re-proves, from scratch, the tier claims of the nostd-tier-lift branch so
# a DONE row is a contract CI can reverify, not a hypothesis (principle 16).
# Each cell is a build/test I verified passing on the committed tree; a cell
# that is expected to FAIL by design (a documented boundary) is NOT included
# and is noted below instead.
#
# Boundaries deliberately NOT gated (they fail by construction):
#   - proxima-primitives `blocking` module on thumbv7m: the atomic-wait futex has
#     no bare-metal (no-OS) backend. A sync blocking lock needs an OS wait
#     primitive; that is exactly why the async-mutex gate exists (principle
#     20). The async-mutex feature IS gated on thumbv7m below — it reaches
#     bare metal.
#   - proxima-primitives `blocking::futex` has no isolated nextest cell (unlike its
#     former proxima-lock crate, which had one): proxima-primitives carries a
#     PRE-EXISTING dev-dependency on the std-only `proxima` umbrella crate
#     (needed by semaphore.rs/runtime_shaped.rs's pre-existing
#     `#[proxima::test]` usage, unrelated to the Workstream F fold), so
#     `cargo nextest run -p proxima-primitives --no-default-features --features
#     blocking` unifies proxima-primitives's own `std` feature back on via that
#     dev-dependency's transitive requirement — `blocking::futex` (gated
#     `not(std)`) then silently fails to compile into the test binary at
#     all, so the command "passes" by running the std-tier tests instead
#     (a false green). This matches the workspace's PREVAILING no_std
#     validation pattern elsewhere in this file (every other nextest cell
#     below also runs WITH std active; no_std reach is proven by the build
#     cells, not by nextest) — proxima-lock's isolated futex nextest was the
#     one exception, only possible because proxima-lock had no such
#     dev-dependency. `blocking::futex`'s logic is unchanged byte-for-byte
#     from the already-tested proxima-lock crate (re-verified in isolation
#     during the fold); the build cell below is the no_std proof for this
#     module going forward.
#
# Usage:  bash scripts/tier-lift-gate.sh
# Exits 0 if clean, non-zero if any cell fails; each cell prints its command.

set -euo pipefail

declare -a cells=(
    # proxima-core io-async — the RISC async IO seam, all 4 tiers (futures::io
    # cannot compile at any no_std corner; that asymmetry is the whole point)
    "proxima-core io-async std+alloc|cargo build -p proxima-core --features io-async"
    "proxima-core io-async no_std+alloc|cargo build -p proxima-core --no-default-features --features alloc,io-async"
    "proxima-core io-async no_std+no_alloc floor|cargo build -p proxima-core --no-default-features --features io-async"
    "proxima-core io-async no_std+no_alloc on thumbv7m|cargo build -p proxima-core --no-default-features --features io-async --lib --target thumbv7m-none-eabi"
    "proxima-core io-async Prepend adapter on thumbv7m (no_std+alloc)|cargo build -p proxima-core --no-default-features --features alloc,io-async --lib --target thumbv7m-none-eabi"
    "proxima-core io-async tests|cargo nextest run -p proxima-core --features io-async"
    # NOTE (docs/pipe-to-metal/edges.md, 2026-07-16 concentration entry): this
    # cell used to re-prove `proxima-net --features prime,io-async,...`'s
    # `PrimeTcpConnection: proxima_core::io::{AsyncRead,AsyncWrite}` impl — a
    # redundant second AsyncRead/AsyncWrite on a socket type that already,
    # correctly, implements the canonical std-tier `futures::io` (which
    # `prime::os::net::TcpStream` itself implements). That impl (and the
    # `proxima-net/io-async` feature gating it) was removed as over-expansion;
    # `prime::tests::prime_tcp_upstream_connects_and_round_trips_bytes`
    # (no feature flag needed) is the real-byte round-trip proof now.

    # proxima-primitives::blocking / proxima-primitives::AsyncMutex — the canonical
    # tier-resolved mutexes (folded from proxima-lock, Workstream F)
    "proxima-primitives std (blocking::Mutex parking_lot passthrough)|cargo build -p proxima-primitives"
    "proxima-primitives no_std+OS futex (blocking, host)|cargo build -p proxima-primitives --no-default-features --features blocking"
    "proxima-primitives async-mutex gate (host)|cargo build -p proxima-primitives --no-default-features --features async-mutex"
    "proxima-primitives async-mutex gate on thumbv7m (bare metal, no futex)|cargo build -p proxima-primitives --no-default-features --features async-mutex --lib --target thumbv7m-none-eabi"
    "proxima-primitives async-mutex stress test|cargo nextest run -p proxima-primitives --features async-mutex"

    # bare-metal lifts — lib must compile no_std+alloc on thumbv7m
    "proxima-patterns control_plane std (folded proxima-control-plane)|cargo build -p proxima-patterns"
    "proxima-patterns control_plane no_std+alloc on thumbv7m|cargo build -p proxima-patterns --no-default-features --features alloc,control_plane --lib --target thumbv7m-none-eabi"
    "proxima-patterns control_plane nextest|cargo nextest run -p proxima-patterns -E 'test(control_plane::)'"

    "proxima-primitives shutdown std (folded proxima-shutdown)|cargo build -p proxima-primitives"
    "proxima-primitives shutdown ResourceRegistry no_std+alloc on thumbv7m|cargo build -p proxima-primitives --no-default-features --features alloc --lib --target thumbv7m-none-eabi"
    "proxima-primitives shutdown nextest|cargo nextest run -p proxima-primitives -E 'test(shutdown::)'"

    "proxima-recording std|cargo build -p proxima-recording"
    "proxima-recording no_std+alloc on thumbv7m|cargo build -p proxima-recording --no-default-features --features alloc --lib --target thumbv7m-none-eabi"
    "proxima-recording nextest (format base)|cargo nextest run -p proxima-recording"
    "proxima-recording nextest (pipe feature)|cargo nextest run -p proxima-recording --features pipe"
    "proxima-recording nextest (replay feature)|cargo nextest run -p proxima-recording --features replay"

    # proxima-listen admission FSM (sans-IO accept-layer, folded from the
    # former proxima-listen-core; stream sibling of quic-proto EndpointDemux).
    # Bounded connection table reaches bare metal as proxima-listen's no_std
    # base tier; the reactor/serving adapter stays std.
    "proxima-listen admission std|cargo build -p proxima-listen"
    "proxima-listen admission no_std+alloc on thumbv7m|cargo build -p proxima-listen --no-default-features --features alloc --lib --target thumbv7m-none-eabi"
    "proxima-listen admission no_std+no_alloc floor on thumbv7m|cargo build -p proxima-listen --no-default-features --lib --target thumbv7m-none-eabi"
    "proxima-listen admission FSM tests|cargo nextest run -p proxima-listen -E 'test(admission::)'"

    # transport — the fan-out dedup: Replay/tap survive, std tier green
    "proxima-primitives std|cargo build -p proxima-primitives"
    "proxima-primitives nextest|cargo nextest run -p proxima-primitives"

    # conflaguration-first-class — the bridge tests (defaults track the sized
    # floor AND equal the former magic constants) re-run here
    "proxima-telemetry log_buffer (folded log-buffer) conflag bridge|cargo nextest run -p proxima-telemetry -E 'test(log_buffer::)'"
    "proxima-listen stream (folded listeners-stream) conflag bridge|cargo nextest run -p proxima-listen --features stream -E 'test(stream::)'"
    "proxima-listen conflag bridge|cargo nextest run -p proxima-listen"
    "proxima-http listener (folded listeners-http) conflag bridge|cargo nextest run -p proxima-http --features http-listener -E 'test(listener::)'"
)

passed=0
failed=0
declare -a failures

for cell in "${cells[@]}"; do
    label="${cell%%|*}"
    cmd="${cell#*|}"
    printf '\n== %s ==\n' "$label"
    printf '   $ %s\n' "$cmd"
    if bash -c "$cmd"; then
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
        failures+=("$label")
    fi
done

printf '\n== tier-lift-gate summary ==\n'
printf '   passed: %d\n' "$passed"
printf '   failed: %d\n' "$failed"

if [ "$failed" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "$label"
    done
    exit 1
fi

printf '\ntier-lift-gate: all green.\n'
