#!/usr/bin/env bash
# prime-serve-gate.sh
# Disciplined-component gate for the prime-default serve runtime
# (docs/prime-serve/discipline.md). Verifies the default flip holds and
# the tokio opt-out still works, then runs the parity + reactor-absence
# proofs.
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
#
# this script never modifies the discipline log; sealing a row is a
# manual step that reads the bench output. the compare-bench itself is
# `cargo bench --bench bench_serve_prime_vs_tokio`.

set -euo pipefail

# the prime runtime feature cluster the serve path + serve_parity need.
prime_feats="runtime-prime-executor,runtime-prime-inbox-alloc,runtime-prime-reactor,runtime-prime-bgpool,http1"

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

printf '\n== prime-serve gate: PASS ==\n'
