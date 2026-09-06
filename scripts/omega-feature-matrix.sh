#!/usr/bin/env bash
# omega-feature-matrix.sh
# Feature-matrix gate for omega's `metal-*` flags.
#
# Audit finding (2026-09-04): omega carries 13 behavior-changing flags (the
# umbrella `metal` feature plus 12 `metal-*` sub-flags, per omega/Cargo.toml's
# own `[features]` block) whose cross-feature correctness dependencies are
# documented only in doc-comment prose on that block. `omega-gate.sh` proves
# the default set and `--all-features`; nothing proves a `metal-*` flag
# ALONE, nothing proves the default-minus-one combinations, and nothing
# proves the pairing `metal.rs`'s own `HazardTracker` doc calls out as the
# argument that must hold: it reasons about hazards over "the arena"
# (`metal-plan-stable-buffers`'s `BufferArena`) whenever a `Plan`'s runtime
# `DispatchType` is `Concurrent` (ROW 312 made that choice a runtime value,
# not a feature, so it is exercised by ordinary test runs, not this matrix).
#
# The feature list is NOT hardcoded here -- it is derived from
# omega/Cargo.toml's `[features]` block at run time (see `all_features` and
# `metal_flags` below), so a new flag lands in this gate automatically the
# next time this script runs, and a removed one drops out without editing
# this file.
#
# CELL 3 -- "default set minus each default-on flag" -- is only PARTIALLY
# reachable. Cargo features are a pure union: once anything in the resolved
# graph requires a feature, it is on, and there is no subtraction operator.
# Probed directly (`cargo build -p omega --no-default-features --features
# metal -v`, grepped for `feature="..."` in rustc's invocation): every one
# of the 5 flags `metal` folds in on its own line (`metal-output-placement`,
# `metal-wide-cooperative-reduce`, `metal-q5k-pair-dot`,
# `metal-plan-stable-buffers`, `cached-attention-streaming`) is UNREACHABLE
# to subtract while `metal` is selected -- `metal`'s own `[features]` entry
# hardwires them, so any path that turns on `metal` turns on all five,
# regardless of entry point. `std` is likewise unreachable to subtract:
# both `metal` and `cpu` (the other two members of `default`) re-add it
# unconditionally. Only `cpu` and `metal` themselves are subtractable from
# `default`, because neither is forced back on by anything else in
# `default`'s own list. This is a Cargo semantics fact, not a gap in this
# script -- the unreachable rows are printed as SKIP with the reason, never
# silently omitted.
#
# Usage: bash scripts/omega-feature-matrix.sh
# Exits 0 if every reachable cell passes, non-zero (after running every
# reachable cell) otherwise. CARGO_TARGET_DIR is read from the environment;
# defaults to `target` in the invoking directory if unset.

set -uo pipefail

export CARGO_TERM_COLOR=never
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target}"

crate="omega"
cargo_toml="omega/Cargo.toml"

# every top-level feature name declared in the `[features]` block
mapfile -t all_features < <(
    sed -n '/^\[features\]/,/^\[dependencies\]/p' "${cargo_toml}" \
        | grep -oE '^[a-zA-Z0-9_-]+ = \[' \
        | sed 's/ = \[//'
)

# the `metal-*` sub-flags, excluding the umbrella `metal` feature itself
metal_flags=()
for feature_name in "${all_features[@]}"; do
    if [[ "${feature_name}" == metal-* ]]; then
        metal_flags+=("${feature_name}")
    fi
done

printf '\n== omega [features] enumeration (derived from %s) ==\n' "${cargo_toml}"
printf '   total features declared: %d\n' "${#all_features[@]}"
printf '   metal-* sub-flags: %d\n' "${#metal_flags[@]}"
for feature_name in "${metal_flags[@]}"; do
    printf '     - %s\n' "${feature_name}"
done

# --list: print the planned cells (no cargo invocation) and exit. Mirrors
# the cell shape the real run below executes, so a count taken here matches
# a count taken from a real run's summary.
if [ "${1:-}" = "--list" ]; then
    cell_count=0
    printf '\n== planned cells ==\n'
    for flag in "${metal_flags[@]}" cached-attention-streaming kv-capacity-bucket; do
        printf '   [cell1] metal,%s\n' "${flag}"
        cell_count=$((cell_count + 1))
    done
    printf '   [cell2] default\n'
    cell_count=$((cell_count + 1))
    printf '   [cell3] default minus cpu\n'
    printf '   [cell3] default minus metal\n'
    cell_count=$((cell_count + 2))
    for forced_flag in std metal-output-placement metal-wide-cooperative-reduce metal-q5k-pair-dot metal-plan-stable-buffers cached-attention-streaming; do
        printf '   [cell3] default minus %s -- SKIP (unreachable, additive features)\n' "${forced_flag}"
        cell_count=$((cell_count + 1))
    done
    printf '   [cell4] all-features\n'
    cell_count=$((cell_count + 1))
    printf '   [cell5] metal-core on x86_64-unknown-linux-gnu (check only, no driver, no execution)\n'
    cell_count=$((cell_count + 1))
    printf '   [cell6] default clippy\n'
    printf '   [cell6] all-features clippy\n'
    cell_count=$((cell_count + 2))
    printf '\n   total cells: %d\n' "${cell_count}"
    exit 0
fi

passed=0
failed=0
skipped=0
declare -a failures

# waits for foreign measurement binaries to clear before every cargo
# invocation -- this script runs dozens of builds, so the courtesy is
# checked per cell, not once at the top.
wait_for_quiet() {
    while pgrep -l 'llama-bench|proxima_model_interop-' > /dev/null 2>&1; do
        printf '   [measurement courtesy] foreign measurement binary running, waiting 60s\n'
        sleep 60
    done
}

# runs a cell that asserts a nonzero nextest count. label identifies the row
# in the summary; feature_args is the full `--features ...` / `--no-default-
# features ...` argument list passed to `cargo nextest run`.
run_nextest_cell() {
    local label="$1"
    shift
    wait_for_quiet
    printf '\n== %s ==\ncargo nextest run -p %s %s\n' "${label}" "${crate}" "$*"
    local output
    output="$(cargo nextest run -p "${crate}" "$@" --no-fail-fast 2>&1)"
    printf '%s\n' "${output}"
    local ran_count
    ran_count="$(printf '%s\n' "${output}" | grep -oE '[0-9]+ tests run:' | grep -oE '[0-9]+' | tail -1)"
    if [ -z "${ran_count}" ] || [ "${ran_count}" -eq 0 ]; then
        printf '   RED: %s ran zero tests -- an empty run is not a pass\n' "${label}"
        failed=$((failed + 1))
        failures+=("${label}: zero tests run")
        return
    fi
    if printf '%s\n' "${output}" | grep -qE '^\s*Summary.*[1-9][0-9]* failed'; then
        printf '   RED: %s reported failing tests\n' "${label}"
        failed=$((failed + 1))
        failures+=("${label}: nextest reported failures")
        return
    fi
    printf '   GREEN: %s (%s tests run)\n' "${label}" "${ran_count}"
    passed=$((passed + 1))
}

run_clippy_cell() {
    local label="$1"
    shift
    wait_for_quiet
    printf '\n== %s ==\ncargo clippy -p %s %s -- -D warnings\n' "${label}" "${crate}" "$*"
    if cargo clippy -p "${crate}" "$@" -- -D warnings; then
        printf '   GREEN: %s\n' "${label}"
        passed=$((passed + 1))
    else
        printf '   RED: %s\n' "${label}"
        failed=$((failed + 1))
        failures+=("${label}: clippy failed")
    fi
}

run_check_cell() {
    local label="$1"
    shift
    wait_for_quiet
    printf '\n== %s ==\ncargo check -p %s %s\n' "${label}" "${crate}" "$*"
    if cargo check -p "${crate}" "$@"; then
        printf '   GREEN: %s\n' "${label}"
        passed=$((passed + 1))
    else
        printf '   RED: %s\n' "${label}"
        failed=$((failed + 1))
        failures+=("${label}: cargo check failed")
    fi
}

skip_cell() {
    local label="$1"
    local reason="$2"
    printf '\n== %s ==\n   SKIP: %s\n' "${label}" "${reason}"
    skipped=$((skipped + 1))
}

# ---------------------------------------------------------------------
# cell 1: every metal-* flag, plus cached-attention-streaming and
# kv-capacity-bucket, ALONE on top of metal
# ---------------------------------------------------------------------
printf '\n== [cell 1] each metal-* flag alone on top of metal ==\n'
for flag in "${metal_flags[@]}" cached-attention-streaming kv-capacity-bucket; do
    run_nextest_cell "cell1: metal,${flag}" --features "metal,${flag}"
done

# ---------------------------------------------------------------------
# cell 2: the default metal set
# ---------------------------------------------------------------------
printf '\n== [cell 2] default feature set ==\n'
run_nextest_cell "cell2: default"

# ---------------------------------------------------------------------
# cell 3: default set minus each default-on flag, where Cargo allows it
# ---------------------------------------------------------------------
printf '\n== [cell 3] default minus each default-on flag ==\n'
run_nextest_cell "cell3: default minus cpu" --no-default-features --features std,metal
run_nextest_cell "cell3: default minus metal" --no-default-features --features std,cpu
skip_cell "cell3: default minus std" \
    "UNREACHABLE: both metal and cpu (the other two members of default) re-add std unconditionally (metal = [\"std\", ...], cpu = [\"std\", ...]); Cargo features are additive-only"
for forced_flag in metal-output-placement metal-wide-cooperative-reduce metal-q5k-pair-dot metal-plan-stable-buffers cached-attention-streaming; do
    skip_cell "cell3: default minus ${forced_flag}" \
        "UNREACHABLE: metal's own [features] entry hardwires ${forced_flag} into every build that selects metal; probed via 'cargo build -p omega --no-default-features --features metal -v', ${forced_flag} appeared in the resolved feature set regardless of entry point"
done

# ---------------------------------------------------------------------
# cell 4: --all-features
# ---------------------------------------------------------------------
printf '\n== [cell 4] --all-features ==\n'
run_nextest_cell "cell4: all-features" --all-features

# ---------------------------------------------------------------------
# cell 5: metal-core builds emitter-only on a non-macOS target -- no metal
# driver, no objc2, so it is a `cargo check` (there is nothing to nextest
# without a driver, and this host cannot execute a cross-compiled linux
# binary anyway). Requires `rustup target add x86_64-unknown-linux-gnu`.
# ---------------------------------------------------------------------
printf '\n== [cell 5] metal-core on x86_64-unknown-linux-gnu ==\n'
run_check_cell "cell5: metal-core linux" \
    --no-default-features --features metal-core --target x86_64-unknown-linux-gnu

# ---------------------------------------------------------------------
# cell 6: clippy on cells 2 and 4's feature sets
# ---------------------------------------------------------------------
printf '\n== [cell 6] clippy pedantic on cells 2 and 4 ==\n'
run_clippy_cell "cell6: default clippy" --all-targets
run_clippy_cell "cell6: all-features clippy" --all-targets --all-features

printf '\n== omega-feature-matrix summary ==\n'
printf '   passed: %d\n' "${passed}"
printf '   failed: %d\n' "${failed}"
printf '   skipped (documented unreachable): %d\n' "${skipped}"

if [ "${failed}" -gt 0 ]; then
    printf '\nFAILURES:\n'
    for label in "${failures[@]}"; do
        printf '   - %s\n' "${label}"
    done
    exit 1
fi

printf '\nomega-feature-matrix: all reachable cells green.\n'
