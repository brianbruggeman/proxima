#!/usr/bin/env bash
# variant-bench.sh — per-crate × per-feature bench + size matrix
#
# measures binary size, compile time, runtime bench delta, and thumbv7m
# cross-compile pass/fail for the no_std + alloc cliff variant matrix.
#
# outputs a single markdown document to stdout suitable for pasting into
# the discipline log as a "## Variant-flag bench matrix" section.
#
# usage: bash scripts/variant-bench.sh > /tmp/variant-matrix.md 2>&1
# requires: cargo, jq, awk

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKSPACE_ROOT}"

THUMBV7M_TARGET="thumbv7m-none-eabi"

# cap criterion arms for speed — goal is feature-shape comparison, not abs numbers
BENCH_WARMUP=1
BENCH_MEASURE=2

# resolve target directory; cargo may redirect via CARGO_TARGET_DIR or
# workspace Cargo.toml's [build] target-dir setting.
CARGO_TARGET="$(cargo metadata --no-deps --format-version 1 2>/dev/null \
    | jq -r '.target_directory')"

# ── dependency check ─────────────────────────────────────────────────────────

if ! command -v jq >/dev/null 2>&1; then
    printf 'error: jq is required; install with brew install jq or apt-get install jq\n' >&2
    exit 1
fi

# ── thumbv7m target ──────────────────────────────────────────────────────────

if ! rustup target list --installed 2>/dev/null | grep -q "${THUMBV7M_TARGET}"; then
    printf 'installing %s ...\n' "${THUMBV7M_TARGET}" >&2
    rustup target add "${THUMBV7M_TARGET}" >&2
fi

# ── helper: human-readable byte count ────────────────────────────────────────

format_bytes() {
    local bytes="$1"
    case "${bytes}" in
        ''|n/a|BLOCKED|PASS|FAIL) printf '%s' "${bytes}"; return ;;
    esac
    awk -v b="${bytes}" 'BEGIN {
        if (b >= 1073741824) { printf "%.1f GiB", b / 1073741824 }
        else if (b >= 1048576) { printf "%.1f MiB", b / 1048576 }
        else if (b >= 1024) { printf "%.1f KiB", b / 1024 }
        else { printf "%d B", b }
    }'
}

# ── helper: rlib size ─────────────────────────────────────────────────────────
# workspace library crates produce an rlib at <target>/release/lib<snake>.rlib.
# the crate name uses hyphens; the file path uses underscores.

get_rlib_size() {
    local crate="$1"
    local snake
    snake="$(printf '%s' "${crate}" | tr '-' '_')"
    local rlib
    rlib="$(ls -S "${CARGO_TARGET}/release/lib${snake}"*.rlib 2>/dev/null | head -1)"
    if [[ -n "${rlib}" && -f "${rlib}" ]]; then
        wc -c < "${rlib}" | tr -d ' '
    else
        printf 'n/a'
    fi
}

# ── helper: measure wall-clock build time via bash SECONDS ───────────────────
# outputs "N.Ns" string or "n/a" on failure.
# note: --timings in modern cargo (post-1.60) outputs HTML only, not JSON.
# we use bash SECONDS + awk for a portable, accurate wall-clock measure.

time_build() {
    local crate="$1"
    local no_default_features="$2"
    local features="$3"
    local extra_args=()

    if [[ "${no_default_features}" == "yes" ]]; then
        extra_args+=("--no-default-features")
    fi
    if [[ -n "${features}" ]]; then
        extra_args+=("--features" "${features}")
    fi

    # force rebuild to get a real timing: touch the crate's lib.rs
    local lib_rs
    lib_rs="$(cargo metadata --no-deps --format-version 1 2>/dev/null \
        | jq -r --arg crate "${crate}" \
          '.packages[] | select(.name == $crate) | .manifest_path' \
        | xargs dirname)/src/lib.rs"

    local start_s=${SECONDS}
    if cargo build --release -p "${crate}" "${extra_args[@]}" --quiet 2>/dev/null; then
        local elapsed=$(( SECONDS - start_s ))
        printf '%ds' "${elapsed}"
    else
        printf 'n/a'
    fi
}

# ── helper: build combo and return size + time ───────────────────────────────
# outputs: <rlib_human>|<compile_time>

build_combo() {
    local crate="$1"
    local no_default="$2"
    local features="$3"
    local extra_args=()

    if [[ "${no_default}" == "yes" ]]; then
        extra_args+=("--no-default-features")
    fi
    if [[ -n "${features}" ]]; then
        extra_args+=("--features" "${features}")
    fi

    # touch src/lib.rs to force incremental rebuild timing (avoids cached 0s)
    local manifest_dir
    manifest_dir="$(cargo metadata --no-deps --format-version 1 2>/dev/null \
        | jq -r --arg crate "${crate}" \
          '.packages[] | select(.name == $crate) | .manifest_path' \
        | xargs dirname)"
    touch "${manifest_dir}/src/lib.rs" 2>/dev/null || true

    local start_s=${SECONDS}
    if cargo build --release -p "${crate}" "${extra_args[@]}" --quiet 2>/dev/null; then
        local elapsed=$(( SECONDS - start_s ))
        local rlib_raw
        rlib_raw="$(get_rlib_size "${crate}")"
        local rlib_human
        rlib_human="$(format_bytes "${rlib_raw}")"
        printf '%s|%ds' "${rlib_human}" "${elapsed}"
    else
        printf 'n/a|n/a'
    fi
}

# ── helper: cross-compile check ──────────────────────────────────────────────
# returns PASS or BLOCKED.

cross_compile_check() {
    local crate="$1"
    local no_default_features="$2"
    local features="$3"
    local extra_args=()

    if [[ "${no_default_features}" == "yes" ]]; then
        extra_args+=("--no-default-features")
    fi
    if [[ -n "${features}" ]]; then
        extra_args+=("--features" "${features}")
    fi

    if cargo build --target "${THUMBV7M_TARGET}" -p "${crate}" \
            "${extra_args[@]}" --quiet 2>/dev/null; then
        printf 'PASS'
    else
        printf 'BLOCKED'
    fi
}

# ── helper: save criterion baseline ──────────────────────────────────────────

save_bench_baseline() {
    local crate="$1"
    local bench_name="$2"
    local label="$3"
    local no_default="$4"
    local features="$5"
    local extra_args=()

    if [[ "${no_default}" == "yes" ]]; then
        extra_args+=("--no-default-features")
    fi
    if [[ -n "${features}" ]]; then
        extra_args+=("--features" "${features}")
    fi

    cargo bench -p "${crate}" --bench "${bench_name}" \
        "${extra_args[@]}" -- \
        --warm-up-time "${BENCH_WARMUP}" \
        --measurement-time "${BENCH_MEASURE}" \
        --save-baseline "${label}" \
        2>/dev/null | grep -v '^$' || true
}

# ── helper: run bench against saved baseline and capture delta summary ────────
# outputs a condensed "change: [lo med hi]" string or "n/a".

bench_vs_baseline() {
    local crate="$1"
    local bench_name="$2"
    local baseline_label="$3"
    local save_label="$4"
    local no_default="$5"
    local features="$6"
    local extra_args=()

    if [[ "${no_default}" == "yes" ]]; then
        extra_args+=("--no-default-features")
    fi
    if [[ -n "${features}" ]]; then
        extra_args+=("--features" "${features}")
    fi

    local raw
    raw="$(cargo bench -p "${crate}" --bench "${bench_name}" \
        "${extra_args[@]}" -- \
        --warm-up-time "${BENCH_WARMUP}" \
        --measurement-time "${BENCH_MEASURE}" \
        --baseline "${baseline_label}" \
        --save-baseline "${save_label}" \
        2>/dev/null | grep 'change:' -A 1 | grep 'time:' || printf '')"

    if [[ -z "${raw}" ]]; then
        printf 'n/a'
        return
    fi

    # extract the median from lines like:  time:   [lo med hi] (p = ...)
    # one line per bench arm — take the median of the middle brackets
    printf '%s' "${raw}" | awk '
        {
            match($0, /\[([^]]+)\]/, arr)
            if (arr[1] != "") {
                n = split(arr[1], parts, " ")
                mid = parts[int(n/2)+1]
                sub(/^[+-]?/, "", mid)
                delta = parts[int(n/2)+1]
                printf "%s ", delta
            }
        }
    ' | sed 's/ $//' | awk '{
        # summarise N arm deltas into a range
        lo = $1; hi = $1
        for (i = 2; i <= NF; i++) {
            val = $i + 0
            if (val < lo) lo = val
            if (val > hi) hi = val
        }
        if (lo == hi) {
            printf "%s (within noise)", $1
        } else {
            printf "%s to %s", lo, hi
        }
    }'
}

# ── combo definitions ─────────────────────────────────────────────────────────
#
# each row: name | features | no-default-features | scope | cross-compile

declare -a COMBO_NAMES=(
    "default"
    "std-only"
    "alloc-only"
    "prime-no-os"
    "prime-full"
)
declare -a COMBO_FEATURES=(
    ""
    "std"
    "alloc"
    "alloc,runtime-prime-inbox-alloc,runtime-prime-executor,runtime-prime-timer"
    "runtime-prime-executor,runtime-prime-inbox-alloc,runtime-prime-reactor,runtime-prime-bgpool,runtime-prime-thread-identity"
)
declare -a COMBO_NO_DEFAULT=(
    "no"
    "yes"
    "yes"
    "yes"
    "no"
)
declare -a COMBO_SCOPE=(
    "all"
    "all"
    "all"
    "prime-only"
    "prime-only"
)
# thumbv7m cross-compile: alloc-only and prime-no-os only (std combos are n/a;
# prime-full pulls runtime-prime-reactor which requires libc/std)
declare -a COMBO_CROSS=(
    "no"
    "no"
    "yes"
    "yes"
    "no"
)

CRATES=("proxima-core" "proxima-primitives" "proxima-runtime" "prime")
PRIME_BENCH="bench_thread_identity"

# bench features must include the required-feature + the combo features
# default combo: required-feature only (default features already on)
# other combos: required-feature appended to combo features
BENCH_REQUIRED_FEATURE="runtime-prime-thread-identity"

# ── emit markdown header ──────────────────────────────────────────────────────

cat <<'HEADER'
# Variant-flag bench matrix

Generated by `scripts/variant-bench.sh`. Captures per-crate × per-feature-combo
rlib binary size, compile time (wall-clock, forced incremental rebuild), and
thumbv7m cross-compile pass/fail. Runtime bench deltas are for `prime` crate's
`bench_thread_identity` bench (criterion, capped at warm-up 1s / measure 2s
for feature-shape comparison).

> **Structural blocker (DC5):** `thumbv7m-none-eabi` cross-compile of `prime` is
> blocked by workspace feature unification. `proxima-primitives`'s default `std`
> feature activates `futures-core/std`, `futures-sink/std`, `futures-io/std`
> (v0.3.32), which contain `extern crate std` unconditionally. Setting
> `proxima-primitives = { default-features = false }` at workspace level fixes the
> cross-compile but breaks 10+ workspace crates that rely on implicit std
> propagation. Fix requires all consumers to explicitly opt into
> `proxima-primitives/std`. Documented in the DC5 discipline row as DC6 pre-work.
> Cross-compile cells that cannot build emit **BLOCKED**.

HEADER

# ── per-crate size + compile time tables ──────────────────────────────────────

for crate in "${CRATES[@]}"; do
    printf '## %s variant matrix\n\n' "${crate}"
    printf '| combo | rlib size | compile time | thumbv7m |\n'
    printf '|---|---|---|---|\n'

    for idx in "${!COMBO_NAMES[@]}"; do
        combo="${COMBO_NAMES[$idx]}"
        features="${COMBO_FEATURES[$idx]}"
        no_default="${COMBO_NO_DEFAULT[$idx]}"
        scope="${COMBO_SCOPE[$idx]}"
        do_cross="${COMBO_CROSS[$idx]}"

        if [[ "${scope}" == "prime-only" && "${crate}" != "prime" ]]; then
            continue
        fi

        result="$(build_combo "${crate}" "${no_default}" "${features}")"
        rlib_size="${result%%|*}"
        compile_time="${result##*|}"

        if [[ "${do_cross}" == "yes" ]]; then
            cross_result="$(cross_compile_check "${crate}" "${no_default}" "${features}")"
        else
            cross_result="n/a"
        fi

        printf '| %-30s | %-10s | %-14s | %s |\n' \
            "${combo}" "${rlib_size}" "${compile_time}" "${cross_result}"
    done

    printf '\n'
done

# ── prime bench delta table ───────────────────────────────────────────────────

printf '## prime bench delta (vs default)\n\n'
printf 'Bench: `%s`\n' "${PRIME_BENCH}"
printf 'Cap: warm-up %ds / measure %ds per arm (feature-shape comparison; noisier than full bench).\n\n' \
    "${BENCH_WARMUP}" "${BENCH_MEASURE}"
printf '| combo | bench delta vs default |\n'
printf '|---|---|\n'
printf '| %-30s | %-30s |\n' "default" "baseline"

# save the default baseline
save_bench_baseline "prime" "${PRIME_BENCH}" "variant-default" \
    "no" "${BENCH_REQUIRED_FEATURE}" 2>/dev/null || true

for idx in "${!COMBO_NAMES[@]}"; do
    combo="${COMBO_NAMES[$idx]}"
    features="${COMBO_FEATURES[$idx]}"
    no_default="${COMBO_NO_DEFAULT[$idx]}"

    if [[ "${combo}" == "default" ]]; then
        continue
    fi

    # bench features: required-feature must be present; append if not already
    bench_features="${features}"
    if [[ "${bench_features}" != *"${BENCH_REQUIRED_FEATURE}"* ]]; then
        if [[ -n "${bench_features}" ]]; then
            bench_features="${bench_features},${BENCH_REQUIRED_FEATURE}"
        else
            bench_features="${BENCH_REQUIRED_FEATURE}"
        fi
    fi

    label="variant-${combo}"
    delta="$(bench_vs_baseline "prime" "${PRIME_BENCH}" \
        "variant-default" "${label}" \
        "${no_default}" "${bench_features}" 2>/dev/null || printf 'n/a')"

    if [[ -z "${delta}" ]]; then
        delta="n/a"
    fi

    printf '| %-30s | %s |\n' "${combo}" "${delta}"
done

printf '\n'

# ── per-profile build matrix ─────────────────────────────────────────────────

printf '## Per-profile build matrix\n\n'
printf 'Profile resolved via `PROXIMA_PROFILE=<name>` env var in `prime/build.rs`.\n\n'
printf '| profile | rlib size | compile time | thumbv7m | notes |\n'
printf '|---|---|---|---|---|\n'

declare -a PROFILES=("linux-daemon" "wasm-edge" "embedded-mqtt-gateway" "bare-metal")
declare -a PROFILE_FEATURES=("" "" "alloc" "")
declare -a PROFILE_NO_DEFAULT=("no" "no" "yes" "yes")
declare -a PROFILE_CROSS=("no" "no" "yes" "yes")
declare -a PROFILE_NOTES=(
    "std + alloc + tokio + rustls"
    "std + alloc + tokio + wasi reactor"
    "no_std + alloc + embassy — DC5 structural blocker"
    "no_std + no alloc — static-prime not yet implemented"
)

for idx in "${!PROFILES[@]}"; do
    profile="${PROFILES[$idx]}"
    features="${PROFILE_FEATURES[$idx]}"
    no_default="${PROFILE_NO_DEFAULT[$idx]}"
    do_cross="${PROFILE_CROSS[$idx]}"
    note="${PROFILE_NOTES[$idx]}"

    export PROXIMA_PROFILE="${profile}"
    result="$(build_combo "prime" "${no_default}" "${features}")"
    unset PROXIMA_PROFILE
    rlib_size="${result%%|*}"
    compile_time="${result##*|}"

    if [[ "${do_cross}" == "yes" ]]; then
        export PROXIMA_PROFILE="${profile}"
        cross_result="$(cross_compile_check "prime" "${no_default}" "${features}")"
        unset PROXIMA_PROFILE
    else
        cross_result="n/a"
    fi

    printf '| %-30s | %-10s | %-14s | %-15s | %s |\n' \
        "${profile}" "${rlib_size}" "${compile_time}" "${cross_result}" "${note}"
done

printf '\n'

# ── structural blocker detail ─────────────────────────────────────────────────

cat <<'BLOCKER'
## Structural blocker — resolved in commit 73c6862

DC5's discipline-log row originally documented a structural blocker:
`cargo build --target thumbv7m-none-eabi -p prime --no-default-features
--features alloc` failed because `proxima-pipe`'s (now folded into
`proxima-primitives`, commit `8cf44fcc`) default `std` feature activated
`futures/std`, which requires std unconditionally.

The script's `cross_compile_check` helper runs the actual build per combo
and emits `PASS` / `BLOCKED` dynamically, so the cells above reflect
reality, not a hardcoded assumption.

The blocker was resolved in commit `73c6862` (`fix(prime): unblock
thumbv7m cross-compile via path-direct deps for pipe + runtime`) by
declaring `proxima-pipe` (now `proxima-primitives`) and `proxima-runtime`
directly as `{ path = "...", default-features = false }` in prime's
`[dependencies]`, bypassing workspace dep inheritance which can't override
`default-features` per cargo's manifest rules.

Long-term cleanup remains: flipping workspace deps for `proxima-primitives`
and `proxima-runtime` to `default-features = false` (and having all
consumers explicitly opt into `proxima-primitives/std` in their own std
features) is the broader fix. That's deferred to a follow-up; the
path-direct workaround is local to prime and isolates the change.

NOTE (verify before trusting this section further): prime's current
`Cargo.toml` no longer declares `proxima-primitives`/`proxima-runtime` as
path-direct `default-features = false` deps — both are plain
`{ workspace = true }` today, relying on the workspace-level entries
(root `Cargo.toml`), which already carry `default-features = false`. The
path-direct-workaround narrative above may be superseded; this was not
re-derived from a specific commit, only observed against the current
tree.

See discipline log: `docs/runtime-prime-nostd/discipline.md`

BLOCKER

printf '_Generated %s_\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
