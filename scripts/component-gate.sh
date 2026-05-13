#!/usr/bin/env bash
# component-gate.sh <component-name>
# runs steps 2-4 of the 11-point disciplined-component gate per component:
#   2. build clean under sub-flag (default-features off, --features sub-flag)
#   3. tests pass
#   4. clippy pedantic clean
# usage: bash scripts/component-gate.sh c1-ring

set -euo pipefail

component="${1:-}"
if [[ -z "${component}" ]]; then
    printf 'usage: %s <component-name>\n' "$0" >&2
    printf '       e.g. %s c1-ring\n' "$0" >&2
    exit 2
fi

crate="proxima-telemetry"
common=(--no-default-features --features "${component}")

printf '\n== gate %s ==\n' "${component}"

printf '\n[1/3] build clean under sub-flag\n'
cargo build -p "${crate}" "${common[@]}"

printf '\n[2/3] tests pass\n'
cargo test -p "${crate}" "${common[@]}"

printf '\n[3/3] clippy pedantic clean\n'
cargo clippy -p "${crate}" "${common[@]}" -- -D warnings

printf '\n== gate %s: green ==\n' "${component}"
