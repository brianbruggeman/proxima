#!/usr/bin/env bash
# proxima-listen-gate.sh — feature-matrix gate for proxima-listen.
#
# Why it exists: the crate is `default = ["std"]`, and `stream` / `tokio` /
# `tls` / `framed-any` are all off there. Three of those four come back on by
# accident — proxima-listen dev-depends on the `proxima` umbrella, which
# enables them, so host feature unification hides the gap. `framed-any` does
# not: nothing in the dev graph turns it on. Measured 2026-08-05 on the
# pre-audit tree, `cargo nextest run -p proxima-listen` ran 124 tests and
# `--all-features` ran 130 — the six `any::framed_any` tests plus that
# module's doctest had never been run by a crate-scoped command, and neither
# `cargo check -p proxima-listen --all-targets` nor `cargo clippy` had ever
# compiled its 850 lines.
#
# Each feature also gets a cell of its OWN, not just a place in the
# --all-features union: a feature that only ever compiles unified with its
# siblings is a feature nobody has proven.
#
# The rustdoc cells are here because nothing else ran rustdoc on this crate.
# Measured 2026-08-05: eleven broken intra-doc links at default features and
# four more at --all-features, including two orphaned doc blocks glued to the
# wrong item (`ThreadLocalListenProtocol`'s onto `ServeBuilder`,
# `FramedListenProtocol`'s onto the `ConnTransform` type alias) that only the
# link checker could see.
#
# The two no_std tiers are NOT duplicated here — scripts/thumbv7m-cliff-gate.sh
# carries the `proxima-listen-bare-no-alloc` and `proxima-listen-alloc` cells
# (guiding-principle 1: one source, no forked copies). A host-side
# `--no-default-features` cell would be a false green anyway: the dev-dep
# unification described above puts `std` back.
#
# Usage:  bash scripts/proxima-listen-gate.sh
# Exits 0 if every cell passes, non-zero (after running them all) otherwise.

set -euo pipefail

declare -a cells=(
    # what the crate's own documented commands cover
    "default check|cargo check -p proxima-listen --all-targets"
    "default tests|cargo nextest run -p proxima-listen --no-fail-fast"
    "default clippy|cargo clippy -p proxima-listen --all-targets -- -D warnings"
    "default rustdoc|RUSTDOCFLAGS='-D warnings' cargo doc -p proxima-listen --no-deps"

    # the union — every folded module compiled, tested and documented together
    "all-features tests|cargo nextest run -p proxima-listen --all-features --no-fail-fast"
    "all-features clippy|cargo clippy -p proxima-listen --all-targets --all-features -- -D warnings"
    "all-features rustdoc|RUSTDOCFLAGS='-D warnings' cargo doc -p proxima-listen --no-deps --all-features"

    # each feature ALONE — unification must not be what makes it compile
    "alloc alone clippy|cargo clippy -p proxima-listen --all-targets --no-default-features --features alloc -- -D warnings"
    "std alone clippy|cargo clippy -p proxima-listen --all-targets --no-default-features --features std -- -D warnings"
    "tls alone clippy|cargo clippy -p proxima-listen --all-targets --no-default-features --features tls -- -D warnings"
    "tokio alone clippy|cargo clippy -p proxima-listen --all-targets --no-default-features --features tokio -- -D warnings"
    "stream alone clippy|cargo clippy -p proxima-listen --all-targets --no-default-features --features stream -- -D warnings"
    "framed-any alone clippy|cargo clippy -p proxima-listen --all-targets --no-default-features --features framed-any -- -D warnings"

    # rustdoc at the two no_std rungs: the `admission` + `preface` docs are
    # the only ones rendered there, and they are the ones most likely to link
    # at an item that is std- or alloc-gated away.
    "bare rustdoc|RUSTDOCFLAGS='-D warnings' cargo doc -p proxima-listen --no-deps --no-default-features"
    "alloc rustdoc|RUSTDOCFLAGS='-D warnings' cargo doc -p proxima-listen --no-deps --no-default-features --features alloc"
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
# (AGENTS.md), so the exit status alone proves nothing. Run it once and
# assert the PASSED COUNT. `--all-features` is what reaches
# `any::framed_any`'s worked example, which the default feature set cannot.
printf '\n== doctests (--all-features), asserting a nonzero count ==\n'
doc_expected=6
if doc_output="$(cargo test -p proxima-listen --doc --all-features 2>&1)"; then
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

printf '\n== proxima-listen-gate summary ==\n'
printf '   passed: %d\n' "$passed"
printf '   failed: %d\n' "$failed"

if [ "$failed" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "$label"
    done
    exit 1
fi

printf '\nproxima-listen-gate: all green.\n'
