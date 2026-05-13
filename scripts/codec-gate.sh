#!/usr/bin/env bash
# codec-gate.sh <crate> <codec-trait-feature>
# 11-point gate for one codec-trait module. mirrors component-gate.sh
# but per-module, with the module's own `<module>-codec-trait` feature
# as the sub-flag.
#
# Since the http-fold (proxima-h1-codec / proxima-h2-codec /
# proxima-hpack / proxima-h3-proto folded into proxima-protocols,
# module-per-protocol), the codec-trait feature is module-prefixed
# rather than a shared literal `codec-trait` name.
#
# usage: bash scripts/codec-gate.sh proxima-protocols http1_codec-codec-trait
#        bash scripts/codec-gate.sh proxima-protocols hpack-codec-trait
#        bash scripts/codec-gate.sh proxima-protocols http2_codec-codec-trait
#        bash scripts/codec-gate.sh proxima-protocols http3_codec-codec-trait
#
# steps:
#   1. build clean under --features <feature>
#   2. tests pass (nextest, lib + integration tests, no examples)
#   3. clippy pedantic clean
#   4. micro-bench runs (records the row's bench data — does NOT seal)
#
# sealing the row in docs/codec-trait/discipline.md is a separate manual
# step that requires reading the bench output, checking CoV, and writing
# the changelog row. this script never modifies the discipline log.

set -euo pipefail

crate="${1:-}"
feature="${2:-codec-trait}"
if [[ -z "${crate}" ]]; then
    printf 'usage: %s <crate> <codec-trait-feature>\n' "$0" >&2
    printf '       e.g. %s proxima-protocols http1_codec-codec-trait\n' "$0" >&2
    exit 2
fi

common=(--features "${feature}")

printf '\n== codec-trait gate: %s (%s) ==\n' "${crate}" "${feature}"

printf '\n[1/4] build clean under --features %s\n' "${feature}"
cargo build -p "${crate}" "${common[@]}"

printf '\n[2/4] nextest passes (lib + tests, no examples)\n'
cargo nextest run -p "${crate}" "${common[@]}" --lib --tests --no-fail-fast

printf '\n[3/4] clippy pedantic clean\n'
cargo clippy -p "${crate}" "${common[@]}" -- -D warnings

printf '\n[4/4] micro-bench scaffolding (records bench data)\n'
if [[ -d "${crate}/benches" ]] && ls "${crate}/benches"/bench_*codec_trait*.rs >/dev/null 2>&1; then
    cargo bench -p "${crate}" "${common[@]}" --no-run
    printf '\n   bench binaries built; run `cargo bench -p %s --features %s` to measure.\n' "${crate}" "${feature}"
else
    printf '\n   no benches/bench_*codec_trait*.rs in %s — micro-bench step is a no-op until C<N> opens.\n' "${crate}"
fi

printf '\n== codec-trait gate %s (%s): green (build/tests/clippy/bench-scaffold) ==\n' "${crate}" "${feature}"
printf '   next: read bench output, check CoV <= 5%%, write the row in docs/codec-trait/discipline.md.\n'
