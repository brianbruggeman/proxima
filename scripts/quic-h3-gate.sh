#!/usr/bin/env bash
# quic-h3-gate.sh — discipline gate for the proxima-quic + proxima-http
# http3 rewrite.
#
# Verifies:
#   - tier-1 (no_std + alloc) and tier-3 (bare no_std + no alloc) build
#     matrices for the sans-IO proto modules (proxima-protocols' `quic`
#     module, folded from proxima-quic-proto, and its `http3_codec`
#     module), on host + thumbv7m-none-eabi;
#   - the always-on quinn-free invariant for proxima-http's http3-native
#     feature (`cargo tree -i quinn` must be empty when only `--features
#     http3-native` is requested) — guards the dual-surface decision
#     recorded in ai_docs/invariants.jsonl as
#     proxima.decision.quic_dual_surface_native_and_quinn;
#   - doctests for both proto crates so an indented protocol diagram
#     in a //! comment can't silently break the nextest-only suite.
#
# Not enforced (staged behind TOKIO_FREE_FACADE_ENFORCE=1):
#   - the tokio-free production-build gate. proxima-http's http3-native
#     facade today pulls tokio directly because the listener uses
#     tokio::net + tokio::spawn; proxima-quic native pulls tokio
#     transitively via prime's std feature (see
#     docs/proxima-quic/edges.md "Tokio transitive leak via prime's std
#     feature"). Flipping TOKIO_FREE_FACADE_ENFORCE=1 today fails by
#     design — both edges above need to land first.
#
# Usage:
#   bash scripts/quic-h3-gate.sh
#
# Exits 0 if clean, non-zero if any cell fails. Each cell prints its
# command + outcome in real time so a CI log shows exactly what ran.

set -euo pipefail

# Cells. Each is "label|cmd-line". The cmd-line runs verbatim under bash.

declare -a cells=(
    # tier-1: no_std + alloc on host target
    "proxima-protocols quic tier-1 (no_std + alloc)|cargo build -p proxima-protocols --no-default-features --features quic-alloc"
    "proxima-protocols http3_codec tier-1 (no_std + alloc)|cargo build -p proxima-protocols --no-default-features --features http3_codec-alloc"

    # tier-3: bare no_std + no alloc on host target
    "proxima-protocols quic tier-3 (no_std + no_alloc)|cargo build -p proxima-protocols --no-default-features --features quic"
    "proxima-protocols http3_codec tier-3 (no_std + no_alloc)|cargo build -p proxima-protocols --no-default-features --features http3_codec-no-alloc"

    # thumbv7m cliff: both tiers must compile on a real embedded target
    "proxima-protocols quic tier-1 on thumbv7m-none-eabi|cargo build -p proxima-protocols --no-default-features --features quic-alloc --target thumbv7m-none-eabi"
    "proxima-protocols http3_codec tier-1 on thumbv7m-none-eabi|cargo build -p proxima-protocols --no-default-features --features http3_codec-alloc --target thumbv7m-none-eabi"
    "proxima-protocols quic tier-3 on thumbv7m-none-eabi|cargo build -p proxima-protocols --no-default-features --features quic --target thumbv7m-none-eabi"
    "proxima-protocols http3_codec tier-3 on thumbv7m-none-eabi|cargo build -p proxima-protocols --no-default-features --features http3_codec-no-alloc --target thumbv7m-none-eabi"

    # std-tier production path (quic-std alias) + tests
    "proxima-protocols quic std-tier features|cargo build -p proxima-protocols --no-default-features --features quic-std"
    "proxima-protocols http3_codec default features|cargo build -p proxima-protocols --features http3_codec-alloc"
    "proxima-protocols quic nextest|cargo nextest run -p proxima-protocols --no-default-features --features quic-std"
    "proxima-protocols http3_codec nextest|cargo nextest run -p proxima-protocols --features http3_codec-alloc"

    # nextest skips doctests by design — but rustdoc reads every
    # indented block in //! comments as Rust unless wrapped in a
    # text fence. an unfenced protocol diagram in a module doc-
    # comment would silently break `cargo test --doc`.
    "proxima-protocols quic doctests|cargo test -p proxima-protocols --no-default-features --features quic-tls-rustls,quic-mock-tls,quic-codec-trait --doc"
    "proxima-protocols http3_codec doctests|cargo test -p proxima-protocols --features http3_codec-alloc,http3_codec-codec-trait,http3_codec-part-source --doc"

    # clippy pedantic clean
    "proxima-protocols quic clippy|cargo clippy -p proxima-protocols --no-default-features --features quic-tls-rustls,quic-mock-tls,quic-codec-trait --all-targets -- -D warnings"
    "proxima-protocols http3_codec clippy|cargo clippy -p proxima-protocols --features http3_codec-alloc,http3_codec-codec-trait,http3_codec-part-source --all-targets -- -D warnings"

    # profile-axis sanity — every profile must validate cleanly with the new
    # quic_impl + h3_impl axes
    "proxima-build profile axes|cargo nextest run -p proxima-build"
)

# Quinn-free native gate — enforced unconditionally. The `--features
# http3-native` build of proxima-http must NOT pull quinn anywhere in
# the tree (the legacy bridge stays gated behind the http3-quinn-compat
# feature).
cells+=(
    "proxima-http quinn-free (http3-native feature only)|cargo build -p proxima-http --no-default-features --features http3-native && (cargo tree -p proxima-http --no-default-features --features http3-native -e no-dev -i quinn 2>&1 | grep -Eq 'did not match any packages|nothing to print' || (echo 'quinn leaked into proxima-http http3-native build' >&2; exit 1))"
)

# Tokio-free facade gate — staged but not enforced today. proxima-http's
# `http3-native` listener uses tokio::net + tokio::spawn pending the
# prime Datagram reactor source; proxima-quic's `native` pulls prime
# which pulls tokio via prime's std feature (edges.md: "Tokio
# transitive leak via prime's std feature (C31)"). Flip
# TOKIO_FREE_FACADE_ENFORCE=1 after both issues are resolved.
if [ "${TOKIO_FREE_FACADE_ENFORCE:-0}" = "1" ]; then
    cells+=(
        "proxima-quic tokio-free (native feature only)|cargo build -p proxima-quic --no-default-features --features native && (cargo tree -p proxima-quic --no-default-features --features native -e no-dev -i tokio 2>&1 | grep -Eq 'did not match any packages|nothing to print' || (echo 'tokio leaked into proxima-quic production build' >&2; exit 1))"
        "proxima-http tokio-free (http3-native feature only)|cargo build -p proxima-http --no-default-features --features http3-native && (cargo tree -p proxima-http --no-default-features --features http3-native -e no-dev -i tokio 2>&1 | grep -Eq 'did not match any packages|nothing to print' || (echo 'tokio leaked into proxima-http http3-native build' >&2; exit 1))"
    )
fi

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

printf '\n== quic-h3-gate summary ==\n'
printf '   passed: %d\n' "$passed"
printf '   failed: %d\n' "$failed"

if [ "$failed" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "$label"
    done
    exit 1
fi

printf '\nquic-h3-gate: all green.\n'
