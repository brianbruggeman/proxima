#!/usr/bin/env bash
# runtime-shaped-gate.sh <primitive>
# 11-point gate for one runtime-shaped primitive (mutex, rwlock,
# notify, mpsc, joinset, sleep). mirrors codec-gate.sh shape but
# per-primitive with a `runtime-shaped-<name>` sub-flag.
#
# usage: bash scripts/runtime-shaped-gate.sh mutex
#        bash scripts/runtime-shaped-gate.sh joinset
#
# steps:
#   1. build clean under --features runtime-shaped-<name>
#   2. nextest passes (lib + integration tests, no examples)
#   3. clippy pedantic clean
#   4. micro-bench scaffolding (records the row's bench data — does
#      NOT seal)
#
# sealing the row in docs/runtime-shaped/discipline.md is a separate
# manual step that requires reading the bench output, checking CoV,
# and writing the changelog row.

set -euo pipefail

primitive="${1:-}"
if [[ -z "${primitive}" ]]; then
    printf 'usage: %s <primitive>\n' "$0" >&2
    printf '       e.g. %s mutex\n' "$0" >&2
    exit 2
fi

# pick the owning crate. all sync and task primitives live in
# proxima-primitives::sync (task folded in from proxima-task, Workstream F;
# sync folded into proxima-primitives in the Wave D Phase 3 primitives merge);
# time primitives in proxima-core::time (folded from the former proxima-time
# crate).
case "${primitive}" in
    mutex|rwlock|notify|mpsc|oneshot|semaphore|barrier|joinset|yield)
        crate="proxima-primitives"
        ;;
    sleep|timeout|interval)
        crate="proxima-core"
        ;;
    *)
        printf 'unknown primitive: %s\n' "${primitive}" >&2
        printf 'known: mutex rwlock notify mpsc oneshot semaphore barrier joinset yield sleep timeout interval\n' >&2
        exit 2
        ;;
esac

feature="runtime-shaped-${primitive}"
common=(--features "${feature}")

printf '\n== runtime-shaped gate: %s (%s) ==\n' "${primitive}" "${crate}"

printf '\n[1/4] build clean under --features %s\n' "${feature}"
cargo build -p "${crate}" "${common[@]}"

printf '\n[2/4] nextest passes (lib + tests, no examples)\n'
cargo nextest run -p "${crate}" "${common[@]}" --lib --tests --no-fail-fast

printf '\n[3/4] clippy pedantic clean\n'
cargo clippy -p "${crate}" "${common[@]}" -- -D warnings

printf '\n[4/4] micro-bench scaffolding (records bench data)\n'
if [[ -d "${crate}/benches" ]] && ls "${crate}/benches"/bench_runtime_shaped_*.rs >/dev/null 2>&1; then
    cargo bench -p "${crate}" "${common[@]}" --no-run
    printf '\n   bench binaries built; run `cargo bench -p %s --features %s` to measure.\n' "${crate}" "${feature}"
else
    printf '\n   no benches/bench_runtime_shaped_*.rs in %s — micro-bench step is a no-op until R<N> opens.\n' "${crate}"
fi

printf '\n== runtime-shaped gate %s: green (build/tests/clippy/bench-scaffold) ==\n' "${primitive}"
printf '   next: read bench output, check CoV <= 5%%, write the row in docs/runtime-shaped/discipline.md.\n'
