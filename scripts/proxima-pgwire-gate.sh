#!/usr/bin/env bash
# proxima-pgwire-gate.sh
# Mechanical gate for the proxima-pgwire stack (docs/proxima-pgwire/discipline.md).
# Re-proves every discipline-log row from the artifact alone, without anyone's
# memory (guiding-principle 16): the sans-IO codec + session FSM, the Pipe-native
# facade (driver / connection-pipe upgrade / auth / COPY / LISTEN-NOTIFY / portal
# suspension), and the two HARD invariants — the codec is untouched-green and the
# no-default-features graph carries zero tokio (bare-metal contract).
#
# usage: bash scripts/proxima-pgwire-gate.sh
#
# this script never modifies the discipline log; sealing a row is a separate
# manual step (read bench/CoV, write the changelog row).

set -euo pipefail

# proxima-pgwire-codec folded into proxima-protocols as the `pgwire_codec`
# feature (protocols-fold); the codec commands below now target that crate
# with the feature enabled instead of a standalone package.
codec="proxima-protocols"
codec_features="pgwire_codec"
facade="proxima-pgwire"

printf '\n== proxima-pgwire gate ==\n'

printf '\n[1/8] codec builds tier-3 (bare no_std + no-alloc) for a cortex-m target\n'
if rustup target list --installed | grep -q thumbv7em-none-eabihf; then
    cargo build -p "${codec}" --no-default-features --features "${codec_features}" --target thumbv7em-none-eabihf
else
    printf '   thumbv7em-none-eabihf not installed; skipping bare-metal build (run: rustup target add thumbv7em-none-eabihf)\n'
fi

printf '\n[2/8] facade builds clean (all features)\n'
cargo build -p "${facade}" --all-features

printf '\n[3/8] codec tests green (must stay untouched at 256)\n'
cargo nextest run -p "${codec}" --no-default-features --features "${codec_features}" --no-fail-fast

printf '\n[4/8] facade tests green (incl. psql / sqlx / tokio-postgres / prime e2e)\n'
cargo nextest run -p "${facade}" --all-features --no-fail-fast

printf '\n[5/8] codec zero-allocation hot-path proof\n'
cargo nextest run -p "${codec}" --no-default-features --features "${codec_features}" --no-fail-fast -E 'test(zero_allocations)'

printf '\n[6/8] clippy pedantic clean across the feature matrix\n'
cargo clippy -p "${codec}" --no-default-features --features "${codec_features}" --all-targets -- -D warnings
cargo clippy -p "${facade}" --all-targets --all-features -- -D warnings
cargo clippy -p "${facade}" --all-targets -- -D warnings
cargo clippy -p "${facade}" --lib --no-default-features -- -D warnings
for feat in listen tls scram md5-auth tokio-compat; do
    printf '   clippy lib --no-default-features --features %s\n' "${feat}"
    cargo clippy -p "${facade}" --lib --no-default-features --features "${feat}" -- -D warnings
done

printf '\n[7/8] TOKIO GATE — the bare facade graph must carry zero tokio\n'
leaked="$(cargo tree -p "${facade}" --no-default-features -e normal -i tokio 2>/dev/null || true)"
if printf '%s' "${leaked}" | grep -q '^tokio'; then
    printf '   FAIL: tokio leaked into the no-default-features graph:\n%s\n' "${leaked}" >&2
    exit 1
fi
printf '   ok: no tokio in the no-default-features graph\n'

printf '\n[8/8] bench scaffolding (records data, does NOT seal)\n'
if ls "${facade}"/benches/*.rs >/dev/null 2>&1; then
    cargo bench -p "${facade}" --all-features --no-run
    printf '   bench binaries built; run `cargo bench -p %s --all-features` to measure.\n' "${facade}"
else
    printf '   no benches in %s\n' "${facade}"
fi

printf '\n== proxima-pgwire gate: green ==\n'
printf '   next: read bench output, check CoV <= 5%%, write/seal rows in docs/proxima-pgwire/discipline.md.\n'
