#!/usr/bin/env bash
# proxima-pgwire-gate.sh
# Mechanical gate for the proxima-pgwire stack.
# Re-proves the stack's claims from the artifact alone, without anyone's
# memory (guiding-principle 16): the sans-IO codec + session FSM, the Pipe-native
# facade (driver / connection-pipe upgrade / auth / COPY / LISTEN-NOTIFY / portal
# suspension), and the two HARD invariants — the codec is untouched-green and the
# no-default-features graph carries zero tokio (bare-metal contract).
#
# usage: bash scripts/proxima-pgwire-gate.sh
#
# the last step only BUILDS the benches; measuring is a deliberate manual step
# so a noisy CI runner never gets to seal a perf claim.

set -euo pipefail

# CI sets CARGO_TERM_COLOR=always; ANSI escapes wrapped around digits break
# any grep/awk that counts cargo/nextest summary output (the class
# proxima-tokenizer-gate.sh and proxima-test-gate.sh hit) -- force it off
# for every invocation in this script, present and future.
export CARGO_TERM_COLOR=never

# proxima-pgwire-codec folded into proxima-protocols as the `pgwire_codec`
# feature (protocols-fold); the codec commands below now target that crate
# with the feature enabled instead of a standalone package.
codec="proxima-protocols"
codec_features="pgwire_codec"
facade="proxima-pgwire"

printf '\n== proxima-pgwire gate ==\n'

printf '\n[1/11] codec builds tier-3 (bare no_std + no-alloc) for a cortex-m target\n'
if rustup target list --installed | grep -q thumbv7em-none-eabihf; then
    cargo build -p "${codec}" --no-default-features --features "${codec_features}" --target thumbv7em-none-eabihf
else
    printf '   thumbv7em-none-eabihf not installed; skipping bare-metal build (run: rustup target add thumbv7em-none-eabihf)\n'
fi

printf '\n[2/11] facade builds clean (all features)\n'
cargo build -p "${facade}" --all-features

printf '\n[3/11] codec tests green (must stay untouched at 256)\n'
cargo nextest run -p "${codec}" --no-default-features --features "${codec_features}" --no-fail-fast

printf '\n[4/11] facade tests green (incl. psql / sqlx / tokio-postgres / prime e2e)\n'
cargo nextest run -p "${facade}" --all-features --no-fail-fast

printf '\n[5/11] codec zero-allocation hot-path proof\n'
cargo nextest run -p "${codec}" --no-default-features --features "${codec_features}" --no-fail-fast -E 'test(zero_allocations)'

printf '\n[6/11] clippy pedantic clean across the feature matrix\n'
cargo clippy -p "${codec}" --no-default-features --features "${codec_features}" --all-targets -- -D warnings
cargo clippy -p "${facade}" --all-targets --all-features -- -D warnings
cargo clippy -p "${facade}" --all-targets -- -D warnings
cargo clippy -p "${facade}" --lib --no-default-features -- -D warnings
for feat in listen tls scram md5-auth tokio-compat client; do
    printf '   clippy lib --no-default-features --features %s\n' "${feat}"
    cargo clippy -p "${facade}" --lib --no-default-features --features "${feat}" -- -D warnings
done

# nextest does not run doctests, and `cargo test --doc` exits 0 when it matched
# NOTHING — zero-passed is indistinguishable from all-passed by exit code — so
# the count is the assertion, not the status.
printf '\n[7/11] doctests run AND are non-vacuous\n'
doc_out="$(cargo test --doc -p "${facade}" --all-features 2>&1)"
printf '%s\n' "${doc_out}"
doc_passed="$(printf '%s' "${doc_out}" | sed -n 's/^test result: ok\. \([0-9][0-9]*\) passed.*/\1/p' | head -1)"
if [ -z "${doc_passed}" ] || [ "${doc_passed}" -eq 0 ]; then
    printf '   FAIL: no doctest actually ran; an exit-0 here proves nothing\n' >&2
    exit 1
fi
printf '   ok: %s doctest(s) ran\n' "${doc_passed}"

# `--all-targets` compiles examples but never RUNS them. The crate's one
# example needs a live PostgreSQL, so it runs here only when one is reachable;
# the workflow's realpg-differential job always has one and always runs it.
printf '\n[8/11] example RUNS when a postgres is reachable\n'
cargo build -p "${facade}" --example capture_realpg --features scram
if [ -n "${PGWIRE_REALPG_HOST:-}" ]; then
    cargo run -p "${facade}" --example capture_realpg --features scram -- "$(mktemp -d)"
    printf '   ok: capture_realpg ran against %s\n' "${PGWIRE_REALPG_HOST}"
else
    printf '   PGWIRE_REALPG_HOST unset; example built but NOT run here.\n'
    printf '   its executed proof is the realpg-differential workflow job.\n'
fi

# rustdoc resolves its own links; a broken one is denied, not warned. Nothing
# else in the gate builds docs, so an unresolved link survives every other step.
printf '\n[9/11] rustdoc resolves (both the default and the full feature set)\n'
cargo doc -p "${facade}" --no-deps
cargo doc -p "${facade}" --no-deps --all-features

printf '\n[10/11] TOKIO GATE — the bare facade graph must carry zero tokio\n'
leaked="$(cargo tree -p "${facade}" --no-default-features -e normal -i tokio 2>/dev/null || true)"
if printf '%s' "${leaked}" | grep -q '^tokio'; then
    printf '   FAIL: tokio leaked into the no-default-features graph:\n%s\n' "${leaked}" >&2
    exit 1
fi
printf '   ok: no tokio in the no-default-features graph\n'

printf '\n[11/11] bench scaffolding (records data, does NOT seal)\n'
if ls "${facade}"/benches/*.rs >/dev/null 2>&1; then
    cargo bench -p "${facade}" --all-features --no-run
    printf '   bench binaries built; run `cargo bench -p %s --all-features` to measure.\n' "${facade}"
else
    printf '   no benches in %s\n' "${facade}"
fi

printf '\n== proxima-pgwire gate: green ==\n'
printf '   next: read bench output and check CoV <= 5%% before quoting any number.\n'
