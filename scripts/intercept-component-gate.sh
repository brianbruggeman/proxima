#!/usr/bin/env bash
# intercept-component-gate.sh — runs the disciplined-component gate per
# proxima-intercept component.
#
# Gate steps automated here (the rest are human-verified at sealing per
# docs/intercept-pipeline/discipline.md row evidence):
#   2. build clean under sub-flag
#   3. tests pass
#   4. clippy pedantic clean
#   5. micro-bench file builds
#
# usage: scripts/intercept-component-gate.sh <component>
#        e.g. scripts/intercept-component-gate.sh c9-capture

set -euo pipefail

component="${1:-}"
if [[ -z "${component}" ]]; then
    printf 'usage: %s <component>\n' "$0" >&2
    printf '       e.g. %s c9-capture\n' "$0" >&2
    printf 'components: c9-capture\n' >&2
    exit 2
fi

case "${component}" in
    c9-capture)
        package="proxima-intercept"
        feature="intercept-capture"
        test_filter='test(capture::) | test(compress::) | test(response_sniff_tests::) | test(request_sniff_tests::) | test(provider_tests::) | test(request_forward_tests::) | test(decode_tests::) | test(summarize::) | test(pump_streaming_tests::)'
        ;;
    c9-replay)
        package="proxima-intercept"
        feature="intercept-replay"
        # test(capture_replay_round_trip::) pulls in the cross-crate
        # capture → BinSink → replay integration test;
        # test(observe_paths_vendored_captures::) re-proves observe + swap
        # from the vendored spec/examples captures (C22).
        # all three are modules inside the single `integration` test binary
        # (proxima-intercept/tests/integration.rs via #[path] + mod), so they
        # are selected by module-qualified test() predicates, not binary().
        # all only compile under intercept-replay.
        test_filter='test(capture::) | test(compress::) | test(response_sniff_tests::) | test(request_sniff_tests::) | test(provider_tests::) | test(request_forward_tests::) | test(decode_tests::) | test(summarize::) | test(pump_streaming_tests::) | test(capture_replay_round_trip::) | test(observe_paths_vendored_captures::) | test(block_compression_disk_ratio::)'
        ;;
    c9-config)
        package="proxima-intercept"
        feature="intercept-config"
        test_filter='test(config::) | test(capture::) | test(compress::) | test(response_sniff_tests::) | test(request_sniff_tests::) | test(provider_tests::) | test(request_forward_tests::) | test(decode_tests::) | test(summarize::) | test(pump_streaming_tests::)'
        ;;
    *)
        printf 'error: unknown component %s\n' "${component}" >&2
        exit 2
        ;;
esac

printf '\n== intercept-gate %s (package=%s feature=%s) ==\n' \
    "${component}" "${package}" "${feature}"

printf '\n[1/4] build clean under sub-flag\n'
cargo build -p "${package}" --features "${feature}"

printf '\n[2/4] build clean WITHOUT sub-flag (firewall check)\n'
cargo build -p "${package}"

printf '\n[3/4] tests pass\n'
if command -v cargo-nextest >/dev/null 2>&1; then
    cargo nextest run -p "${package}" --features "${feature}" -E "${test_filter}"
else
    cargo test -p "${package}" --features "${feature}" -- capture::
fi

printf '\n[4/5] clippy pedantic clean (lib + tests, --no-deps so each component owns its own lint state)\n'
cargo clippy --no-deps -p "${package}" --features "${feature}" --lib --tests -- -D warnings

if [[ "${feature}" == *"intercept-capture"* ]] || [[ "${feature}" == "intercept-capture" ]] \
    || [[ "${feature}" == "intercept-replay" ]] || [[ "${feature}" == "intercept-config" ]]; then
    printf '\n[5/5] micro-bench file builds (no_run keeps host quiet)\n'
    cargo build -p "${package}" --features "${feature}" --bench bench_intercept_capture
else
    printf '\n[5/5] micro-bench skipped (feature does not bring in capture)\n'
fi

printf '\n== intercept-gate %s: green ==\n' "${component}"
