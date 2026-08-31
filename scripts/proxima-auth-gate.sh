#!/usr/bin/env bash
# proxima-auth-gate.sh — feature-matrix gate for proxima-auth.
#
# Why it exists: the crate is `default = ["std"]` and its three form features —
# `signing` (AWS SigV4), `digest` (RFC 7616) and `negotiate` (RFC 4559 SPNEGO)
# — are all off there, so a crate-scoped command compiles ONE of the crate's
# four modules. Nothing in the dev graph turns them back on, and no workspace
# command reaches them either: the only consumers are proxima-pgwire (which
# takes `alloc` alone) and proxima-patterns (behind its own default-off
# `middleware` feature). Measured 2026-08-05 on the pre-audit tree:
# `cargo nextest run -p proxima-auth` ran 8 tests, `--all-features` ran 32.
# The 24 that never ran are the AWS `aws-sig-v4-test-suite` vectors, the RFC
# 7616 §3.9.1 digest vectors, and the whole negotiate-loop FSM — every locked
# parity vector this crate owns, unrun by any gate.
#
# Each feature also gets a cell of its OWN, not just a place in the
# --all-features union: a feature that only ever compiles unified with its
# siblings is a feature nobody has proven. `digest` and `signing` share `sha2`
# and `hex`, so the union cannot show which one carries the dependency.
#
# The rustdoc cells are here because nothing else ran rustdoc on this crate,
# and its module docs cross-reference items that only exist behind a feature —
# exactly the link that resolves in the union and breaks alone.
#
# The no_std tiers are NOT duplicated here — scripts/thumbv7m-cliff-gate.sh
# carries the `proxima-auth` and `proxima-auth-forms` cells
# (guiding-principle 1: one source, no forked copies).
#
# Usage:  bash scripts/proxima-auth-gate.sh
# Exits 0 if every cell passes, non-zero (after running them all) otherwise.

set -euo pipefail

# CI sets CARGO_TERM_COLOR=always; ANSI escapes wrapped around digits break
# any grep/awk that counts cargo/nextest summary output (the class
# proxima-tokenizer-gate.sh and proxima-test-gate.sh hit) -- force it off
# for every invocation in this script, present and future.
export CARGO_TERM_COLOR=never

declare -a cells=(
    # what the crate's own documented commands cover: token.rs alone
    "default check|cargo check -p proxima-auth --all-targets"
    "default tests|cargo nextest run -p proxima-auth --no-fail-fast"
    "default clippy|cargo clippy -p proxima-auth --all-targets -- -D warnings"
    "default rustdoc|RUSTDOCFLAGS='-D warnings' cargo doc -p proxima-auth --no-deps"

    # the union — every form compiled, tested and documented together
    "all-features check|cargo check -p proxima-auth --all-targets --all-features"
    "all-features tests|cargo nextest run -p proxima-auth --all-features --no-fail-fast"
    "all-features clippy|cargo clippy -p proxima-auth --all-targets --all-features -- -D warnings"
    "all-features rustdoc|RUSTDOCFLAGS='-D warnings' cargo doc -p proxima-auth --no-deps --all-features"

    # each form ALONE — unification must not be what makes it compile, and each
    # form's tests must run without its siblings' dependencies in the graph
    "signing alone tests|cargo nextest run -p proxima-auth --no-default-features --features std,signing --no-fail-fast"
    "signing alone clippy|cargo clippy -p proxima-auth --all-targets --no-default-features --features std,signing -- -D warnings"
    "digest alone tests|cargo nextest run -p proxima-auth --no-default-features --features std,digest --no-fail-fast"
    "digest alone clippy|cargo clippy -p proxima-auth --all-targets --no-default-features --features std,digest -- -D warnings"
    "negotiate alone tests|cargo nextest run -p proxima-auth --no-default-features --features std,negotiate --no-fail-fast"
    "negotiate alone clippy|cargo clippy -p proxima-auth --all-targets --no-default-features --features std,negotiate -- -D warnings"

    # the alloc floor on the host: every form declares `alloc`, so selecting one
    # without `std` must still resolve to a buildable tier
    "alloc floor clippy|cargo clippy -p proxima-auth --no-default-features --features alloc,signing,digest,negotiate -- -D warnings"
    "alloc floor rustdoc|RUSTDOCFLAGS='-D warnings' cargo doc -p proxima-auth --no-deps --no-default-features --features alloc,signing,digest,negotiate"
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
# the PASSED COUNT. The one doctest is `TokenLifecycle`'s worked example: the
# needs_fetch / set_token / poll protocol, whose whole point is that the two
# questions are asked in different places.
printf '\n== doctests (--all-features), asserting a nonzero count ==\n'
doc_expected=1
if doc_output="$(cargo test -p proxima-auth --doc --all-features 2>&1)"; then
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

printf '\n== proxima-auth-gate summary ==\n'
printf '   passed: %d\n' "$passed"
printf '   failed: %d\n' "$failed"

if [ "$failed" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "$label"
    done
    exit 1
fi

printf '\nproxima-auth-gate: all green.\n'
