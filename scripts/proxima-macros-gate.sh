#!/usr/bin/env bash
# proxima-macros-gate.sh
# Mechanical gate for proxima-macros (proc-macro support: #[proxima::test],
# runtime entrypoints, telemetry spans, schema description, error derives).
# No gate script existed for this crate before -- `cargo nextest run -p
# proxima-macros` and `cargo test --doc` both exit 0 whether they ran real
# work or nothing, so this asserts the nonzero count explicitly instead of
# trusting the exit code alone.
#
# usage: bash scripts/proxima-macros-gate.sh

set -euo pipefail

crate="proxima-macros"

printf '\n== %s gate ==\n' "${crate}"

printf '\n[1/5] proc-macro crate + trybuild fixtures build clean\n'
cargo build -p "${crate}" --all-targets

printf '\n[2/5] tests green (unit tests + trybuild pass/fail fixtures), count asserted\n'
nextest_output="$(cargo nextest run -p "${crate}" --no-fail-fast 2>&1)"
printf '%s\n' "${nextest_output}"
ran_count="$(printf '%s\n' "${nextest_output}" | grep -oE '[0-9]+ tests run:' | grep -oE '[0-9]+' | tail -1)"
if [ -z "${ran_count}" ] || [ "${ran_count}" -eq 0 ]; then
    printf 'ERROR: %s nextest reported zero tests run -- an empty run is not a pass\n' "${crate}" >&2
    exit 1
fi
printf '   tests run: %s\n' "${ran_count}"

printf '\n[3/5] clippy pedantic clean\n'
cargo clippy -p "${crate}" --all-targets -- -D warnings

printf '\n[4/5] rustdoc resolves\n'
cargo doc -p "${crate}" --no-deps

# a proc-macro crate cannot itself carry a runnable doctest for its own
# exported macros (the macro only expands in a DOWNSTREAM crate's token
# stream); its 28 ``` fences are illustrative usage blocks on the macro
# definitions. `cargo test --doc` on a `proc-macro = true` crate reports a
# real, structural zero -- assert that explicitly rather than silently
# skipping it, and rather than treating it as the same "nonzero required"
# shape every other crate in this fleet uses.
printf '\n[5/5] doctests: proc-macro crate, asserting the structural 0 expected, 0 found\n'
doctest_output="$(cargo test --doc -p "${crate}" 2>&1)"
printf '%s\n' "${doctest_output}"
if ! printf '%s\n' "${doctest_output}" | grep -qE 'doctests are not supported for crates|^test result: ok\. 0 passed'; then
    printf 'ERROR: %s doctest run no longer matches the expected proc-macro baseline -- update this gate\n' "${crate}" >&2
    exit 1
fi
printf '   doctests: 0 expected, 0 found (proc-macro crate, structural)\n'

printf '\n== %s gate: PASS ==\n' "${crate}"
