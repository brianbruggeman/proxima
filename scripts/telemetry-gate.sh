#!/usr/bin/env bash
# telemetry-gate.sh
# Disciplined-component gate for proxima-telemetry. Re-proves the crate's
# correctness claims from scratch — the emit path (SpanSink, no per-span box),
# the lock-free MPSC ring (no-loss/no-tear stress test), the drain path, and the
# native-vs-OTLP byte-parity vectors — plus the feature tiers compile.
#
# usage: bash scripts/telemetry-gate.sh
#
# steps:
#   1. default build + clippy --all-targets -D warnings (pedantic, workspace lints)
#   2. default test suite (emit, MPSC+MPMC ring, drain, capture, native encode
#      parity, AND the lossless-backpressure proofs: elastic producer-assist
#      no-hang with no drainer, zero-drop under saturation, shutdown flush,
#      managed-drainer pump). a backpressure hang regression is caught by the
#      scoped nextest terminate-after in .config/nextest.toml.
#   3. otlp-http build + tests (adds the OTLP protobuf parity vectors)
#   4. feature-tier builds compile (tracing-init / macros / histogram / otlp-grpc)
#   5. no_std tier marker builds (--no-default-features)
#   6. doctests under otlp-http (the superset feature the crate's examples
#      need; nextest skips doctests by design, so this is the only place
#      they execute) — a run reporting zero passed doctests fails the
#      gate instead of looking like success
#
# this script never modifies the discipline log; sealing a row is a manual read
# of the bench output. the ring data-race proof (ThreadSanitizer, MPSC + MPMC)
# is a separate CI job — see .github/workflows/proxima-telemetry.yml — because it
# needs nightly + -Zbuild-std.

set -euo pipefail

crate="proxima-telemetry"

# runs `cargo test --doc -p <crate>` (plus any extra args), then fails loudly
# if the reported passed count is zero or absent — `cargo test --doc` exits 0
# on a vacuous "0 passed" run, which is exactly how a feature-gating mistake
# hides as green (see proxima's scripts/prime-serve-gate.sh for the proof).
doctest_check() {
    local label="$1"
    shift
    printf '\n-- doctests (%s) --\n' "${label}"
    local doctest_output
    doctest_output="$(cargo test --doc -p "${crate}" "$@" 2>&1)"
    printf '%s\n' "${doctest_output}"
    local passed_count
    passed_count="$(printf '%s\n' "${doctest_output}" | grep -oE '^test result: ok\. [0-9]+ passed' | grep -oE '[0-9]+' | tail -1)"
    if [ -z "${passed_count}" ] || [ "${passed_count}" -eq 0 ]; then
        printf 'ERROR: %s doctests (%s) reported zero passed -- an empty run is not a pass\n' "${crate}" "${label}" >&2
        exit 1
    fi
    printf 'doctests passed (%s): %s\n' "${label}" "${passed_count}"
}

printf '\n== telemetry gate ==\n'

printf '\n-- 1. default build + clippy --\n'
cargo build -p "${crate}"
cargo clippy -p "${crate}" --all-targets -- -D warnings

printf '\n-- 2. default test suite --\n'
cargo nextest run -p "${crate}"

printf '\n-- 3. otlp-http build + clippy + parity vectors --\n'
cargo build -p "${crate}" --features otlp-http
cargo clippy -p "${crate}" --features otlp-http --all-targets -- -D warnings
cargo nextest run -p "${crate}" --features otlp-http
# NOTE: the OTLP-over-the-wire E2E (transport-send keystone) lives in the UMBRELLA
# tests (../tests/otlp_send_prime_e2e.rs) — it composes the facade (proxima::test +
# proxima H1ClientUpstream/PrimeTcpUpstream + proxima::telemetry) so it belongs to
# the proxima crate's prime test lane, not this codec-only telemetry gate. Run:
#   cargo test -p proxima --features \
#     "test-prime,otlp-http,http-prime,runtime-prime-executor,runtime-prime-inbox-alloc,runtime-prime-reactor,runtime-prime-bgpool,tcp" \
#     --test otlp_send_prime_e2e

printf '\n-- 4. feature-tier builds compile --\n'
for feat in tracing-init macros histogram otlp-grpc tee-generic; do
    printf '   features=%s\n' "${feat}"
    cargo build -p "${crate}" --features "${feat}"
done

printf '\n-- 5. no_std tier marker builds --\n'
cargo build -p "${crate}" --no-default-features

printf '\n-- 6. doctests (otlp-http) --\n'
doctest_check "otlp-http" --features otlp-http

printf '\n== telemetry gate: PASS ==\n'
