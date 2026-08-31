#!/usr/bin/env bash
# proxima-recording-gate.sh — feature-matrix gate for proxima-recording.
#
# Why it exists: the crate ships six features and `default = ["std"]` names
# one. `pipe` and `replay` are rescued by accident — the crate dev-depends on
# the `proxima` umbrella, which depends back on proxima-recording with
# `features = ["pipe", "replay"]`, and Cargo unifies features across the one
# compiled instance — but `durable-wal`, `pipe-config` and `replay-config` are
# turned on by NOTHING: not a sibling crate, not CI, not a script.
# Grep-verified 2026-08-05 across every Cargo.toml, workflow and script in the
# tree: the only mentions are proxima-recording's own manifest.
#
# What that cost, measured on the pre-audit tree (2026-08-05):
#   - `cargo clippy -p proxima-recording --lib --features replay-config`
#     exited 101 on clippy::duplicated_attributes. It had been failing since
#     the three-crate fold, recorded there as deferred debt precisely because
#     no gate ran it.
#   - `cargo nextest run -p proxima-recording --features durable-wal` ran 6
#     integration tests and 2 FAILED (`sync_returns_success_on_happy_path`,
#     `crash_mid_frame_yields_eof_not_corrupted_event`) — they call
#     `tokio::fs` from a `#[proxima::test]` body, which drives on a prime
#     shard where no tokio reactor exists.
#   - `cargo doc -p proxima-recording --no-deps` exited 101 on six broken or
#     redundant intra-doc links.
# Three classes of rot, none of them reachable by the crate's own default
# commands.
#
# Each config feature also gets a cell of its OWN rather than only a place in
# the --all-features union: `pipe-config` and `replay-config` share `bon` and
# `conflaguration`, so the union cannot show which one carries a break.
#
# The no_std tiers are NOT duplicated here — scripts/thumbv7m-cliff-gate.sh
# carries the `proxima-recording` and `proxima-recording-replay` cells
# (guiding-principle 1: one source, no forked copies). The host-side alloc
# cells below use `--lib` on purpose: any target that pulls dev-dependencies
# drags the `proxima` umbrella in and forces `std` straight back on, so a
# no-default-features cell with tests in it proves nothing about the tier.
#
# Usage:  bash scripts/proxima-recording-gate.sh
# Exits 0 if every cell passes, non-zero (after running them all) otherwise.

set -euo pipefail

# CI sets CARGO_TERM_COLOR=always; ANSI escapes wrapped around digits break
# any grep/awk that counts cargo/nextest summary output (the class
# proxima-tokenizer-gate.sh and proxima-test-gate.sh hit) -- force it off
# for every invocation in this script, present and future.
export CARGO_TERM_COLOR=never

declare -a cells=(
    # what the crate's own documented commands cover
    "default check|cargo check -p proxima-recording --all-targets"
    "default tests|cargo nextest run -p proxima-recording --no-fail-fast"
    "default clippy|cargo clippy -p proxima-recording --all-targets -- -D warnings"
    "default rustdoc|RUSTDOCFLAGS='-D warnings' cargo doc -p proxima-recording --no-deps"

    # the union — every feature compiled, tested and documented together
    "all-features check|cargo check -p proxima-recording --all-targets --all-features"
    "all-features tests|cargo nextest run -p proxima-recording --all-features --no-fail-fast"
    "all-features clippy|cargo clippy -p proxima-recording --all-targets --all-features -- -D warnings"
    "all-features rustdoc|RUSTDOCFLAGS='-D warnings' cargo doc -p proxima-recording --no-deps --all-features"

    # durable-wal ALONE — the offset-cursor integration tests are the only
    # thing that compiles tests/durable_wal.rs at all
    "durable-wal tests|cargo nextest run -p proxima-recording --features durable-wal --no-fail-fast"
    "durable-wal clippy|cargo clippy -p proxima-recording --all-targets --features durable-wal -- -D warnings"

    # each config feature ALONE — unification must not be what makes it build
    "pipe-config clippy|cargo clippy -p proxima-recording --all-targets --features pipe-config -- -D warnings"
    "pipe-config tests|cargo nextest run -p proxima-recording --features pipe-config --no-fail-fast"
    "replay-config clippy|cargo clippy -p proxima-recording --all-targets --features replay-config -- -D warnings"
    "replay-config tests|cargo nextest run -p proxima-recording --features replay-config --no-fail-fast"

    # the alloc floor on the host, lib-only (dev-deps would force std back on):
    # the sans-IO format tier the crate doc promises a bare-metal target
    "alloc floor clippy|cargo clippy -p proxima-recording --lib --no-default-features --features alloc -- -D warnings"
    "alloc floor rustdoc|RUSTDOCFLAGS='-D warnings' cargo doc -p proxima-recording --no-deps --no-default-features --features alloc"
    "alloc+replay floor clippy|cargo clippy -p proxima-recording --lib --no-default-features --features alloc,replay -- -D warnings"
)

passed=0
failed=0
declare -a failures

for cell in "${cells[@]}"; do
    label="${cell%%|*}"
    command="${cell#*|}"

    printf '\n== %s ==\n%s\n' "$label" "$command"

    if eval "$command"; then
        passed=$((passed + 1))
    else
        failed=$((failed + 1))
        failures+=("$label")
    fi
done

# nextest does NOT run doctests, and `cargo test --doc` exits 0 on an empty
# match — zero-passed is indistinguishable from all-passed by exit code
# (AGENTS.md), so the exit status alone proves nothing. Run it once and assert
# the PASSED COUNT. Before 2026-08-05 this crate had no doctest at all, so the
# command it would have run was the empty match itself. The one doctest is the
# `Format` round trip: frame a batch into the exact bytes a log appends, read
# one unit back.
printf '\n== doctests (--all-features), asserting a nonzero count ==\n'
doc_expected=1
if doc_output="$(cargo test -p proxima-recording --doc --all-features 2>&1)"; then
    printf '%s\n' "$doc_output"
    doc_passed="$(printf '%s\n' "$doc_output" \
        | awk '/^test result:/ { print $4; exit }')"
    if [ -n "$doc_passed" ] && [ "$doc_passed" -ge "$doc_expected" ]; then
        printf 'ok: %s doctests passed (expected at least %s)\n' \
            "$doc_passed" "$doc_expected"
        passed=$((passed + 1))
    else
        printf 'FAIL: expected at least %s doctests, got %s\n' \
            "$doc_expected" "${doc_passed:-none}" >&2
        failed=$((failed + 1))
        failures+=("doctests: passed count below $doc_expected")
    fi
else
    printf '%s\n' "$doc_output"
    failed=$((failed + 1))
    failures+=("doctests: cargo test --doc failed")
fi

printf '\n== proxima-recording-gate summary ==\n'
printf '   passed: %d\n' "$passed"
printf '   failed: %d\n' "$failed"

if [ "$failed" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "$label"
    done
    exit 1
fi

printf '\nproxima-recording-gate: all green.\n'
