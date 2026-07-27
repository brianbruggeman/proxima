#!/usr/bin/env bash
# prime-serve-gate.sh
# Disciplined-component gate for the prime-default serve runtime.
# Verifies the default flip holds and the tokio opt-out still works,
# then runs the parity + reactor-absence proofs.
#
# usage: bash scripts/prime-serve-gate.sh
#
# steps:
#   1. default build = PRIME (no flags) builds + clippy clean
#   2. tokio opt-out builds (runtime-tokio + http-hyper)
#   3. minimal --no-default-features compiles
#   4. serve_parity runs: byte-parity prime==tokio + the 2 MiB streaming
#      vector + the reactor-absence proof (Handle::try_current is Err on
#      the prime serve path)
#   5. full umbrella suite is green on the prime default
#   6. doctests must be fully green on default features — the command a
#      contributor actually runs (`cargo test --doc -p proxima`) never
#      shows red for a known, accepted gap; doctests whose backing
#      feature is off are `ignore`d at the source (`#[cfg_attr]` on the
#      fence), not silently failing
#   7. those same doctests execute and pass for real under the feature
#      set that registers every listener/client example (`nextest`
#      skips doctests by design, so steps 6/7 are the only place they
#      run); either step reporting zero passed fails the gate instead
#      of looking like success
#
# this script never modifies the discipline log; sealing a row is a
# manual step that reads the bench output. the compare-bench itself is
# `cargo bench --bench bench_serve_prime_vs_tokio`.

set -euo pipefail

# the prime runtime feature cluster the serve path + serve_parity need.
prime_feats="runtime-prime-executor,runtime-prime-inbox-alloc,runtime-prime-reactor,runtime-prime-bgpool,http1"

# additive on top of default features: http2 is already default, http1-native
# is the tokio-free base that registers the "http" listen protocol so the
# `.http()`/`.https()`/`.tcp()`/`.grpc()` doc examples can actually serve.
doctest_feats="http1-native,http2"

# runs `cargo test --doc -p proxima` (plus any extra args), then fails
# loudly if the reported passed count is zero or absent — `cargo test --doc`
# exits 0 on a vacuous "0 passed" run (verified empirically), which is
# exactly how a feature-gating mistake hides as green.
doctest_check() {
    local label="$1"
    shift
    printf '\n-- doctests (%s) --\n' "${label}"
    local doctest_output
    doctest_output="$(cargo test --doc -p proxima "$@" 2>&1)"
    printf '%s\n' "${doctest_output}"
    local passed_count
    passed_count="$(printf '%s\n' "${doctest_output}" | grep -oE '^test result: ok\. [0-9]+ passed' | grep -oE '[0-9]+' | tail -1)"
    if [ -z "${passed_count}" ] || [ "${passed_count}" -eq 0 ]; then
        printf 'ERROR: proxima doctests (%s) reported zero passed -- an empty run is not a pass\n' "${label}" >&2
        exit 1
    fi
    printf 'doctests passed (%s): %s\n' "${label}" "${passed_count}"
}

printf '\n== prime-serve gate ==\n'

printf '\n-- 1. default build = prime --\n'
cargo build -p proxima
cargo clippy -p proxima --all-targets

printf '\n-- 2. tokio opt-out builds --\n'
cargo build -p proxima --no-default-features \
    --features "runtime-tokio,http-hyper,tcp,udp,http1,http2,histogram,macros"

# a thin prime build (no tls/udp/http3). bare --no-default-features is a
# pre-existing umbrella gap (lib.rs imports proxima_h2 unconditionally) and
# is out of scope for this gate.
printf '\n-- 3. lean prime build (h1+h2, no tls/udp/http3) compiles --\n'
cargo build -p proxima --no-default-features \
    --features "serve-prime,tcp,http1,http2,histogram,macros"

printf '\n-- 4. serve_parity: byte-parity + streaming-on-prime + reactor-absence --\n'
cargo nextest run -p proxima --test serve_parity --features "${prime_feats}"

printf '\n-- 5. full umbrella suite on the prime default --\n'
cargo nextest run -p proxima

printf '\n-- 6. doctests on default features (must be fully green) --\n'
doctest_check "default features"

printf '\n-- 7. doctests under %s (exercises the feature-gated examples) --\n' "${doctest_feats}"
doctest_check "${doctest_feats}" --features "${doctest_feats}"

printf '\n== prime-serve gate: PASS ==\n'
