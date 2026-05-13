#!/usr/bin/env bash
# per-component micro-bench coverage map for proxima-telemetry.
# measures absolute performance of each telemetry primitive in isolation.
# cross-cutting end-to-end composition lives in bench-proxima-telemetry-e2e
# (not yet wired; that is a separate target).
#
# usage: scripts/bench-proxima-telemetry.sh
#
# outputs: benches/RESULTS_bench-proxima-telemetry_<platform>.md

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_bench-common.sh
source "${script_dir}/_bench-common.sh"

crate_dir="$(cd "$script_dir/.." && pwd)"
cd "$crate_dir"

PLATFORM="$(detect_platform)"

RESULTS="benches/RESULTS_bench-proxima-telemetry_${PLATFORM}.md"
CRITERION_DIR="target/criterion"
LOGS_DIR="/tmp/bench-proxima-telemetry-logs"
mkdir -p "$LOGS_DIR"

printf 'bench-proxima-telemetry: platform=%s\n' "$PLATFORM"
printf 'output: %s\n' "$RESULTS"

# extract_stats <estimates_json_path>
extract_stats() {
    local json_path="$1"
    local mean_ns cov formatted
    mean_ns="$(extract_mean_ns "$json_path")"
    cov="$(extract_cov "$json_path")"
    formatted="$(format_time_ns "$mean_ns")"
    printf '%s %s' "$formatted" "$cov"
}

# run_component <component_tag> <feature> <bench_name>
#   component_tag  — short label used in output (e.g. "c1-ring")
#   feature        — cargo feature flag (e.g. "c1-ring")
#   bench_name     — [[bench]] name from Cargo.toml (e.g. "bench_c1_ring")
#
# if the bench file is absent from proxima-telemetry/benches/, the function
# emits a PENDING row and returns immediately (no cargo invocation).
run_component() {
    local component_tag="$1"
    local feature="$2"
    local bench_name="$3"

    local bench_file="proxima-telemetry/benches/${bench_name}.rs"

    if [[ ! -f "$bench_file" ]]; then
        printf -- '--- %s: bench file absent — PENDING ---\n' "$component_tag"
        printf '| %s | mean | PENDING | PENDING | n/a | %s |\n' \
            "$component_tag" "$bench_name" >> "$RESULTS"
        return
    fi

    printf -- '--- %s / %s ---\n' "$component_tag" "$bench_name"

    cargo bench \
        -p proxima-telemetry \
        --no-default-features \
        --features "$feature" \
        --bench "$bench_name" \
        -- --save-baseline "bench-proxima-telemetry-${component_tag}" \
        2>&1 | tee "${LOGS_DIR}/${bench_name}.log" || true

    local bench_criterion_dir="${CRITERION_DIR}"
    if [[ ! -d "$bench_criterion_dir" ]]; then
        printf '| %s | (no criterion output) | pending | pending | n/a | %s |\n' \
            "$component_tag" "$bench_name" >> "$RESULTS"
        return
    fi

    local found=0
    while IFS= read -r -d '' estimates_json; do
        local arm_dir group_dir
        arm_dir="$(dirname "$estimates_json")"
        if [[ "$(basename "$arm_dir")" == "new" ]]; then
            arm_dir="$(dirname "$arm_dir")"
        fi
        group_dir="$(dirname "$arm_dir")"
        local group arm
        group="$(basename "$group_dir")"
        arm="$(basename "$arm_dir")"

        local formatted cov
        read -r formatted cov < <(extract_stats "$estimates_json")
        printf '| %s/%s | mean | %s | pending | %s | %s |\n' \
            "$group" "$arm" "$formatted" "$cov" "$bench_name" >> "$RESULTS"
        found=$((found + 1))
    done < <(find "$bench_criterion_dir" -name "estimates.json" -print0 2>/dev/null)

    if [[ $found -eq 0 ]]; then
        printf '| %s | mean | pending | pending | n/a | %s |\n' \
            "$component_tag" "$bench_name" >> "$RESULTS"
    fi
}

# ── initialise results file ──────────────────────────────────────────────────

cat > "$RESULTS" << HEADER
# proxima-telemetry bench results — ${PLATFORM}

Per-component micro-bench coverage map for proxima-telemetry primitives.
Each component is benched in isolation behind its own feature flag.
No comparison axis — this answers "is each primitive fast enough on its own?"

Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')

---

HEADER

# ── per-component benches ────────────────────────────────────────────────────

printf '## per-component\n\n' >> "$RESULTS"
printf '| component | metric | %s | linux | CoV | bench file |\n' "$PLATFORM" >> "$RESULTS"
printf '|-----------|--------|-----|-------|-----|------------|\n' >> "$RESULTS"

# sealed / gate-green today
run_component "c1-ring"  "c1-ring"  "bench_c1_ring"
run_component "c2-id"    "c2-id"    "bench_c2_id"
run_component "c3-level" "c3-level" "bench_c3_level"
run_component "c4-attr"  "c4-attr"  "bench_c4_attr"
run_component "c5-trace" "c5-trace" "bench_c5_trace"

# in flight — bench files not yet present
run_component "c6-metric-basic"     "c6-metric-basic"     "bench_c6_counter"
run_component "c7-metric-histogram" "c7-metric-histogram" "bench_c7_histogram"
run_component "c8-log"              "c8-log"              "bench_c8_log"

# pending implementation
run_component "c9-recorder"      "c9-recorder"      "bench_c9_recorder"
run_component "c10-out-otlp-http" "c10-out-otlp-http" "bench_c10_otlp_http"
run_component "c12-out-native"   "c12-out-native"   "bench_c12_native"

# ── done ──────────────────────────────────────────────────────────────────────

printf '\n---\nRun completed: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$RESULTS"
printf 'bench-proxima-telemetry complete. results: %s\n' "$RESULTS"
