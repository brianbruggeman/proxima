#!/usr/bin/env bash
# proxima-autograd-gate.sh — feature/target matrix gate for proxima-autograd.
#
# Why it exists: `proxima-autograd` was a workspace member wired to ZERO CI
# jobs — the same gap `proxima-tensor-gate.sh` documented for proxima-tensor
# on 2026-08-19. This gate follows that script's structure: every feature
# gets its own cell (unification must not be what makes a feature compile),
# the doctest cell asserts a NONZERO count (`cargo test --doc` exits 0 on an
# empty match), and the portable-target cell is guarded rather than bare so a
# missing target/toolchain prints a named SKIP instead of silently exiting 0.
#
# `proxima-autograd`'s feature set (proxima-autograd/Cargo.toml) is smaller
# than proxima-tensor's: `default = ["std", "config"]`, plus `alloc`, `std`,
# `config`, `instrument`. No `test-support` feature of its own — its
# dev-dependency on `proxima` already carries that. `--all-features` unifies
# all four (alloc/std/config/instrument), none of which need an external
# asset (no vendored checkout, no multi-GiB model file), so unlike
# proxima-tensor's `ggml-bench` there is nothing to exclude from the union.
#
# Usage:  bash scripts/proxima-autograd-gate.sh
# Exits 0 if every cell passes, non-zero (after running them all) otherwise.

set -euo pipefail

# CI sets CARGO_TERM_COLOR=always; ANSI escapes wrapped around digits break
# any grep/awk that counts cargo/nextest summary output (the class
# proxima-tokenizer-gate.sh and proxima-test-gate.sh hit) -- force it off
# for every invocation in this script, present and future.
export CARGO_TERM_COLOR=never

# checked, never linked, so this runs from an arm64 host as well as from
# ubuntu CI where it is the native target.
PORTABLE_TARGET="x86_64-unknown-linux-gnu"

# `proxima-autograd`'s slowest tests are real Adam training-convergence runs
# over an actual corpus (`language_model::adam_training_overfits_...`,
# `model_search::rerunning_the_pipeline_reproduces_byte_identical_results`,
# `sparse_ffn_pruning::pruned_ffn_bands_degrade_loss_then_recover_...`) --
# measured 70-94s each, all under `#[proxima::test]`'s default 60s
# `body_timeout` (`proxima-test/src/harness.rs:446-448`). That default reads
# as a hang and fails the whole run; it is not one. Raised, not disabled --
# `PROXIMA_TEST_TIMEOUT_MS` is the harness's own env override, so a genuine
# hang still fails, just past a budget sized for this crate's real workload.
export PROXIMA_TEST_TIMEOUT_MS="${PROXIMA_TEST_TIMEOUT_MS:-180000}"

declare -a cells=(
    # what the crate's own documented commands cover
    "default check|cargo check -p proxima-autograd --all-targets"
    "default tests|cargo nextest run -p proxima-autograd --no-fail-fast"
    "default clippy|cargo clippy -p proxima-autograd --all-targets -- -D warnings"
    "default rustdoc|cargo doc -p proxima-autograd --no-deps"

    # alloc tier. graph construction only (adjoint transform, relu/softmax,
    # Adam expression builder) -- no evaluation, no config derive stack. See
    # the `alloc` feature's own doc comment in proxima-autograd/Cargo.toml.
    "alloc tier check|cargo check -p proxima-autograd --no-default-features --features alloc"
    "alloc tier clippy|cargo clippy -p proxima-autograd --no-default-features --features alloc -- -D warnings"

    # each feature ALONE — unification must not be what makes it compile.
    #
    # `std` and `instrument` are `--lib`-only here, not `--all-targets`: most
    # of this crate's integration tests build an `optimizer::AdamConfig`
    # (config-gated, since it is the bon-builder + conflaguration + serde
    # composition proxima-tensor's own `spec::ProgramSpec` already uses), so
    # the test *binaries* require `config` unconditionally — that is a
    # property of the tests, not of the library surface `std`/`instrument`
    # advertise alone. `config alone` below is `--all-targets` because it is
    # the one single-feature combination the test suite actually compiles
    # under (`config` implies `std`).
    "std alone clippy|cargo clippy -p proxima-autograd --lib --no-default-features --features std -- -D warnings"
    "config alone clippy|cargo clippy -p proxima-autograd --all-targets --no-default-features --features config -- -D warnings"
    "instrument clippy|cargo clippy -p proxima-autograd --lib --no-default-features --features instrument -- -D warnings"
    # no test in this crate is gated on `instrument` (nothing in src/ or
    # tests/ reads `feature = "instrument"` today), so proving it composes
    # with the tests that DO exist needs `config` unified alongside it.
    "instrument tests|cargo nextest run -p proxima-autograd --no-default-features --features std,config,instrument --no-fail-fast"

    # the default clippy cell above never compiles the full feature surface
    # unified together, which is how lint errors that only surface under
    # unification would sit unseen.
    "all-features clippy|cargo clippy -p proxima-autograd --all-targets --all-features -- -D warnings"

    # rustdoc gets the same unification treatment as clippy above: the
    # default rustdoc cell never compiles this feature set together. This
    # invocation has broken pushes to CI 4x before landing — it must pass.
    "all-features rustdoc|cargo doc -p proxima-autograd --no-deps --all-features"

    # the portable arm. lib only, deliberately: `--all-targets` pulls
    # dev-dependencies, and this crate's dev-deps include
    # proxima-tokenizer's `hf` feature, which is not guaranteed to
    # cross-compile from arm64-darwin. On CI this costs nothing — the runner
    # is ubuntu-x86_64, so PORTABLE_TARGET is the NATIVE target there and the
    # `default` cells above already compile every test, example and bench
    # against it.
    "portable check|cargo check -p proxima-autograd --target ${PORTABLE_TARGET}"

    # the cross-linker env is homebrew's mac-only naming
    # (`x86_64-unknown-linux-gnu-gcc`) and must NEVER be forced onto a host
    # where `${PORTABLE_TARGET}` is the NATIVE target -- that is exactly
    # ubuntu CI, whose own compiler is plain `cc` / `x86_64-linux-gnu-gcc`,
    # not the brew cross-binary. `rustup target list --installed` is true on
    # CI too (it's the native target), so that check alone is not enough:
    # the cell must also confirm it is genuinely CROSS-compiling (host triple
    # != target triple) before it goes looking for a cross toolchain, and
    # must probe for that toolchain with `command -v` rather than assume it.
    # On a native x86_64-linux host this is a clean, named SKIP -- the
    # `default clippy` cell above already clippies proxima-autograd natively,
    # so skipping here is redundant coverage, not lost coverage.
    "portable clippy cross-check|host_triple=\"\$(rustc -vV | awk '/^host:/ {print \$2}')\"; if [ \"\$host_triple\" = \"${PORTABLE_TARGET}\" ]; then printf 'SKIP: host is already %s (native) -- default clippy cell above already covers it natively; this cell only exists to cross-compile.\n' \"${PORTABLE_TARGET}\"; elif ! rustup target list --installed | grep -qx \"${PORTABLE_TARGET}\"; then printf 'SKIP: %s is not installed (rustup target add %s) -- this cell did NOT run and must not read as a pass.\n' \"${PORTABLE_TARGET}\" \"${PORTABLE_TARGET}\"; elif ! command -v \"${PORTABLE_TARGET}-gcc\" >/dev/null 2>&1; then printf 'SKIP: %s-gcc not found on PATH (brew install %s) -- this cell did NOT run and must not read as a pass.\n' \"${PORTABLE_TARGET}\" \"${PORTABLE_TARGET}\"; else env CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=${PORTABLE_TARGET}-gcc CC_x86_64_unknown_linux_gnu=${PORTABLE_TARGET}-gcc CXX_x86_64_unknown_linux_gnu=${PORTABLE_TARGET}-g++ AR_x86_64_unknown_linux_gnu=${PORTABLE_TARGET}-ar cargo clippy --target ${PORTABLE_TARGET} -p proxima-autograd --all-targets --all-features -- -D warnings; fi"
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

# doctests get their own cell because the count has to be asserted. nextest
# does not run doctests at all, so `cargo test --doc` is the only thing that
# reaches them, and it reports success on an empty match.
printf '\n== doctests (count asserted nonzero) ==\n'
doc_log="$(mktemp)"
if cargo test --doc -p proxima-autograd > "$doc_log" 2>&1; then
    doc_count="$(awk '/^test result:/ { total += $4 } END { print total + 0 }' "$doc_log")"
    printf 'doctests passed: %s\n' "$doc_count"
    if [ "$doc_count" -lt 1 ]; then
        printf 'RED: cargo test --doc ran %s doctests. an empty match exits 0 and reads as green.\n' "$doc_count"
        failed=$((failed + 1))
        failures+=("doctests ran zero")
    else
        passed=$((passed + 1))
    fi
else
    cat "$doc_log"
    failed=$((failed + 1))
    failures+=("doctests")
fi
rm -f "$doc_log"

printf '\n== proxima-autograd-gate summary ==\n'
printf '   passed: %d\n' "$passed"
printf '   failed: %d\n' "$failed"

if [ "$failed" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "$label"
    done
    exit 1
fi

printf '\nproxima-autograd-gate: all green.\n'
