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
#   5. full umbrella suite is green on the prime default — this runs with
#      NO features, so anything gated behind a feature off by default
#      (e.g. http1-native) never compiles here and is invisible to this
#      step alone; step 6 exists because of that
#   6. the single-connection determinism proof (upstreams::record::tests::
#      determinism) runs under http1-native, the feature it needs to even
#      compile — step 5 alone silently never builds it, let alone runs it
#   7. doctests must be fully green on default features — the command a
#      contributor actually runs (`cargo test --doc -p proxima`) never
#      shows red for a known, accepted gap; doctests whose backing
#      feature is off are `ignore`d at the source (`#[cfg_attr]` on the
#      fence), not silently failing
#   8. those same doctests execute and pass for real under the feature
#      set that registers every listener/client example (`nextest`
#      skips doctests by design, so steps 7/8 are the only place they
#      run); either step reporting zero passed fails the gate instead
#      of looking like success
#
# this script never modifies the discipline log; sealing a row is a
# manual step that reads the bench output. the compare-bench itself is
# `cargo bench --bench bench_serve_prime_vs_tokio`.

set -euo pipefail

# CI sets CARGO_TERM_COLOR=always; ANSI escapes wrapped around digits break
# any grep/awk that counts cargo/nextest summary output (the class
# proxima-tokenizer-gate.sh and proxima-test-gate.sh hit) -- force it off
# for every invocation in this script, present and future.
export CARGO_TERM_COLOR=never

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

# runs an `-E`-filtered `cargo nextest run -p proxima` (plus any extra
# args), then fails loudly if the reported passed count is zero or absent.
# Verified empirically: with nextest's current default (`--no-tests=warn`),
# a filter matching zero tests still exits 0 ("warning: no tests to run --
# this will become an error in the future") — the same failure mode
# `doctest_check` above guards against, just on the nextest side instead of
# `cargo test --doc`. This is how a feature-gated test module (needing
# http1-native to even compile) went unnoticed by step 5's bare nextest
# run: nothing about that run's exit code or "PASS" banner distinguishes
# "343 tests, all green" from "343 tests, all green, plus 1 more that was
# never in the binary at all".
nextest_filter_check() {
    local label="$1" filter="$2"
    shift 2
    printf '\n-- nextest (%s) --\n' "${label}"
    local nextest_output
    # --color never: nextest force-colours when it detects CI, and the escape codes
    # land between the count and the word ("3<ESC>[0m <ESC>[32;1mpassed"), so the
    # summary grep below matches locally and silently fails on a CI runner.
    nextest_output="$(cargo nextest run --color never -p proxima -E "${filter}" "$@" 2>&1)"
    printf '%s\n' "${nextest_output}"
    local passed_count
    passed_count="$(printf '%s\n' "${nextest_output}" | grep -oE '[0-9]+ tests? run: [0-9]+ passed' | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' | tail -1)"
    if [ -z "${passed_count}" ] || [ "${passed_count}" -eq 0 ]; then
        printf 'ERROR: proxima nextest (%s) reported zero passed -- an empty filter is not a pass\n' "${label}" >&2
        exit 1
    fi
    printf 'nextest passed (%s): %s\n' "${label}" "${passed_count}"
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

printf '\n-- 6. determinism proof (requires http1-native; step 5 alone never compiles it) --\n'
nextest_filter_check "record::tests::determinism, http1-native" 'test(record::tests::determinism)' --features http1-native

printf '\n-- 7. doctests on default features (must be fully green) --\n'
doctest_check "default features"

printf '\n-- 8. doctests under %s (exercises the feature-gated examples) --\n' "${doctest_feats}"
doctest_check "${doctest_feats}" --features "${doctest_feats}"

printf '\n== prime-serve gate: PASS ==\n'
