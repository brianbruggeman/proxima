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
    "build no_std + no-alloc, handshake only (no AEAD suite)|cargo build -p proxima-centauri --no-default-features"
    "build no_std + no-alloc + chacha suite|cargo build -p proxima-centauri --no-default-features --features aead-chacha20poly1305"
    "build on thumbv7m, handshake only (bare metal, NO allocator)|cargo build -p proxima-centauri --no-default-features --lib --target thumbv7m-none-eabi"
    "build on thumbv7m + chacha suite|cargo build -p proxima-centauri --no-default-features --features aead-chacha20poly1305 --lib --target thumbv7m-none-eabi"

    # The suite at both tiers. NOTE what the second cell does and does not
    # prove: it runs with the crate's `std` feature OFF, which is real, but
    # `alloc` remains reachable in a TEST binary because dev-dependencies link
    # it and inherent methods from `alloc` on primitives are then visible
    # without any `extern crate alloc` here. So a test that allocates would
    # still pass this cell. The bare-metal `--lib` build above is the binding
    # no-alloc proof; this one proves the suite does not need std.
    "nextest (std)|cargo nextest run -p proxima-centauri"
    "test suite RUNS with std off (see note: NOT a no-alloc proof)|cargo test -p proxima-centauri --no-default-features --features aead-chacha20poly1305"

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

    # AEAD suites are additive, so all three shapes must build and pass: each
    # suite alone, and both together with the choice made at runtime.
    "aes-gcm suite alone: builds|cargo build -p proxima-centauri --no-default-features --features aead-aes-gcm"
    "aes-gcm suite alone: tests|cargo nextest run -p proxima-centauri --no-default-features --features std,aead-aes-gcm"
    "both suites together: tests incl. cross-suite rejection|cargo nextest run -p proxima-centauri --features aead-aes-gcm"
    "aes-gcm suite on thumbv7m (bare metal)|cargo build -p proxima-centauri --no-default-features --features aead-aes-gcm --lib --target thumbv7m-none-eabi"

    # principle 11: the walkthrough is teaching surface only if it runs.
    "handshake walkthrough EXECUTES|cargo run -q -p proxima-centauri --example handshake_walkthrough"

    # doctests: nextest skips them, and a vacuous run exits 0, so assert a
    # nonzero pass count. tee keeps the raw output in the CI log.
    "doctests (nonzero)|cargo test --doc -p proxima-centauri 2>&1 | tee /dev/stderr | grep -qE 'test result: ok\\. [1-9][0-9]* passed'"

    # benches must keep compiling; they are the discipline log's evidence and
    # rot silently otherwise. --no-run so CI does not pay for a full sweep.
    "benches compile|cargo bench -p proxima-centauri --no-run"
)

# Second target. The curve backend and the atomic width both resolve per
# target, so a single-architecture green is a single-architecture claim.
# On an aarch64 host this runs x86_64 under emulation: correctness only —
# timings from an emulated target would be a lie, and are not taken here.
# NB: command substitution, not `rustc -vV | grep -q`. Under `set -o pipefail`
# grep -q exits on the first match, rustc takes SIGPIPE, and pipefail reports
# the pipeline as failed — so the condition is ALWAYS false and the cell is
# silently skipped while the gate still says green. Found the hard way.
RUSTC_HOST="$(rustc -vV | sed -n 's/^host: //p')"
case "$RUSTC_HOST" in
    aarch64-apple-darwin) SECOND_TARGET=x86_64-apple-darwin ;;
    x86_64-apple-darwin) SECOND_TARGET=aarch64-apple-darwin ;;
    x86_64-unknown-linux-gnu) SECOND_TARGET=aarch64-unknown-linux-gnu ;;
    *) SECOND_TARGET="" ;;
esac

if [ -n "$SECOND_TARGET" ] && rustup target list --installed | grep -E "^${SECOND_TARGET}$" >/dev/null; then
    cells+=(
        "second target ${SECOND_TARGET}: builds|cargo build -p proxima-centauri --target ${SECOND_TARGET}"
        "second target ${SECOND_TARGET}: tests pass|cargo test -p proxima-centauri --target ${SECOND_TARGET}"
        "second target ${SECOND_TARGET}: no-alloc tier|cargo test -p proxima-centauri --no-default-features --features aead-chacha20poly1305 --target ${SECOND_TARGET}"
    )
else
    printf 'SKIP: second-target cells — no installed second target for host %s\n' "$RUSTC_HOST"
fi

# Instruction counts. Deterministic where wall-clock is not, which matters
# because several deltas in the discipline log are smaller than this host's
# 1.1-1.5%% criterion spread. callgrind SIGSEGVs on aarch64 macOS even with
# valgrind installed, so probe by running it rather than by checking `which`.
if command -v valgrind >/dev/null 2>&1 \
    && command -v iai-callgrind-runner >/dev/null 2>&1 \
    && valgrind --tool=callgrind --callgrind-out-file=/dev/null /bin/true >/dev/null 2>&1; then
    cells+=("cycle counts (callgrind)|cargo bench -p proxima-centauri --bench bench_cycles")
else
    printf 'SKIP: cycle-count cell — no working callgrind on this host (runs on the Linux CI leg)\n'
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
