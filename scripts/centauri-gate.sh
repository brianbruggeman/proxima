#!/usr/bin/env bash
# centauri-gate.sh — discipline gate for proxima-centauri.
#
# Re-proves, from scratch and without anyone's memory, every claim the
# centauri discipline log makes (guiding-principle 16): a DONE row is a
# contract CI can reverify, not a hypothesis.
#
# The three cells that exist because a claim here has previously been made
# on evidence that did not actually cover it:
#
#   - The bare-metal build IS the no-alloc proof. `cargo build` on the host
#     says nothing about allocation: with no allocator linked, a `Vec` fails
#     to compile, so thumbv7m-none-eabi is the assertion. An alloc-counter
#     test is weaker — it can only count allocations that were possible.
#   - The suite must RUN at --no-default-features, not merely compile. A test
#     suite that only executes with std cannot defend a no_std claim.
#   - The example must be EXECUTED. `cargo check --all-targets` compiles
#     examples and never runs them, which is how broken examples have reached
#     public main in this repo before.
#
# `cargo test --doc` exits 0 on a vacuous "0 passed" run, so the doctest cell
# greps for a nonzero count rather than trusting the exit code.
#
# Usage:  bash scripts/centauri-gate.sh
# Exits 0 if clean, non-zero if any cell fails; each cell prints its command.

set -euo pipefail

declare -a cells=(
    # formatting and lints across every target, including benches and the example
    "fmt|cargo fmt -p proxima-centauri -- --check"
    "clippy (all targets)|cargo clippy -p proxima-centauri --all-targets"

    # tier ladder. The last of these is the one that makes "no-alloc" a fact.
    "build std|cargo build -p proxima-centauri"
    "build no_std + no-alloc (host)|cargo build -p proxima-centauri --no-default-features"
    "build no_std + no-alloc on thumbv7m (bare metal, NO allocator)|cargo build -p proxima-centauri --no-default-features --lib --target thumbv7m-none-eabi"

    # the suite, at both tiers. The second is the load-bearing one: it proves
    # the tests themselves carry no alloc, so they can defend the tier claim.
    "nextest (std)|cargo nextest run -p proxima-centauri"
    "test suite RUNS at no_std + no-alloc|cargo test -p proxima-centauri --no-default-features"

    # take-once exclusivity, exhaustively rather than by sampling. Mutation
    # tested 2026-07-28: a check-then-act draw fails this cell.
    "loom model check (EntropyCell take-once)|cargo test -p proxima-centauri --features loom --test loom_entropy_cell"

    # principle 12: the sizing TOML actually drives the constants. A build
    # with an override must produce a differently-shaped replay bitmap and
    # still pass every test.
    "sized constants respond to an env override|PROXIMA_CENTAURI_REPLAY_WINDOW_PACKETS=512 cargo nextest run -p proxima-centauri"

    # gate point 12: the config surface and the fluent builder must construct
    # equivalent components, and the config must not carry key material.
    "config + API parity (gate point 12)|cargo nextest run -p proxima-centauri --features config -E 'test(config::)'"
    "clippy with config surface|cargo clippy -p proxima-centauri --features config --all-targets"

    # principle 11: the walkthrough is teaching surface only if it runs.
    "handshake walkthrough EXECUTES|cargo run -q -p proxima-centauri --example handshake_walkthrough"

    # doctests: nextest skips them, and a vacuous run exits 0, so assert a
    # nonzero pass count. tee keeps the raw output in the CI log.
    "doctests (nonzero)|cargo test --doc -p proxima-centauri 2>&1 | tee /dev/stderr | grep -qE 'test result: ok\\. [1-9][0-9]* passed'"

    # benches must keep compiling; they are the discipline log's evidence and
    # rot silently otherwise. --no-run so CI does not pay for a full sweep.
    "benches compile|cargo bench -p proxima-centauri --no-run"
)

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

printf '\n== centauri-gate summary ==\n'
printf '   passed: %d\n' "$passed"
printf '   failed: %d\n' "$failed"

if [ "$failed" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "$label"
    done
    exit 1
fi

printf '\ncentauri-gate: all green.\n'
