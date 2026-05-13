#!/usr/bin/env bash
# answers: which proxima runtime+channel combo is best on h2 warm GET?
# three arms: default tokio (tokio internal), per-core tokio (flume),
#             prime native (prime-inbox-alloc)

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_bench-common.sh
source "${script_dir}/_bench-common.sh"

crate_dir="$(cd "$script_dir/.." && pwd)"
cd "$crate_dir"

PLATFORM="$(detect_platform)"

FEATURES="runtime-tokio,runtime-prime-full,http1,http2,tcp,tls,prime-tokio-compat"
RESULTS="benches/RESULTS_bench-proxima-runtimes_${PLATFORM}.md"
LOGS_DIR="/tmp/bench-proxima-runtimes-logs"
mkdir -p "$LOGS_DIR"

printf 'bench-proxima-runtimes (%s)\n' "$PLATFORM"

# ---------------------------------------------------------------------------
# run benches
# ---------------------------------------------------------------------------

printf -- '--- h2_runtime_swap (default_tokio + per_core_runtime arms) ---\n'
cargo bench --no-default-features --features "$FEATURES" \
    --bench h2_runtime_swap -- h2_runtime_swap_proxima_native \
    2>&1 | tee "${LOGS_DIR}/h2_runtime_swap.log" || true

printf -- '--- bench_runtime_compat (prime arm) ---\n'
cargo bench --no-default-features --features "$FEATURES" \
    --bench bench_runtime_compat -- prime \
    2>&1 | tee "${LOGS_DIR}/bench_runtime_compat.log" || true

printf -- '--- bench_spawn_burst ---\n'
cargo bench --no-default-features --features "$FEATURES" \
    --bench bench_spawn_burst \
    2>&1 | tee "${LOGS_DIR}/bench_spawn_burst.log" || true

printf -- '--- bench_runtime_swap (general runtime swap) ---\n'
cargo bench --no-default-features --features "$FEATURES" \
    --bench bench_runtime_swap \
    2>&1 | tee "${LOGS_DIR}/bench_runtime_swap.log" || true

printf -- '--- h3_runtime_swap ---\n'
cargo bench --no-default-features --features "$FEATURES" \
    --bench h3_runtime_swap \
    2>&1 | tee "${LOGS_DIR}/h3_runtime_swap.log" || true

# ---------------------------------------------------------------------------
# parse criterion estimates.json via jq
# mean.point_estimate is in nanoseconds for time-based benches;
# throughput groups are reported in elements/s — label accordingly.
# TODO: if criterion output paths change (e.g. new criterion version renames
#       the directory layout), update CRITERION_DIR and the group/arm paths.
# ---------------------------------------------------------------------------

CRITERION_DIR="target/criterion"

read_mean_fmt() {
    local group="$1"
    local arm="$2"
    local estimates_file
    estimates_file="$(find_estimates "$CRITERION_DIR" "$group" "$arm")"
    local mean_ns
    mean_ns="$(extract_mean_ns "$estimates_file")"
    format_time_ns "$mean_ns"
}

read_cov() {
    local group="$1"
    local arm="$2"
    local estimates_file
    estimates_file="$(find_estimates "$CRITERION_DIR" "$group" "$arm")"
    extract_cov "$estimates_file"
}

read_raw_ns() {
    local group="$1"
    local arm="$2"
    local estimates_file
    estimates_file="$(find_estimates "$CRITERION_DIR" "$group" "$arm")"
    extract_mean_ns "$estimates_file"
}

h2_default="$(read_mean_fmt "h2_runtime_swap_proxima_native" "default_tokio")"
h2_percore="$(read_mean_fmt "h2_runtime_swap_proxima_native" "per_core_runtime")"
h2_prime="$(read_mean_fmt   "w5_single_stream_get"           "prime")"
h2_default_cov="$(read_cov  "h2_runtime_swap_proxima_native" "default_tokio")"
h2_percore_cov="$(read_cov  "h2_runtime_swap_proxima_native" "per_core_runtime")"
h2_prime_cov="$(read_cov    "w5_single_stream_get"           "prime")"

# spawn_burst_1k is the canonical dispatch-cost isolator
sb_default="$(read_raw_ns "spawn_burst_1k" "tokio_per_core")"
sb_percore="$(read_raw_ns "spawn_burst_1k" "tokio_per_core")"
sb_prime="$(read_raw_ns   "spawn_burst_1k" "prime")"
sb_default_cov="$(read_cov "spawn_burst_1k" "tokio_per_core")"
sb_prime_cov="$(read_cov   "spawn_burst_1k" "prime")"

# ---------------------------------------------------------------------------
# emit RESULTS doc
# ---------------------------------------------------------------------------

cat > "$RESULTS" <<MARKDOWN
# bench-proxima-runtimes — ${PLATFORM}

This document captures runtime+channel variance for proxima on h2 warm GET
and spawn-burst (dispatch-cost isolator). Three arms are compared:
default tokio multi-thread (tokio internal channel), per-core tokio (flume),
and prime native (prime-inbox-alloc). Run the script again after real
measurements to replace \`pending\` cells.

## h2 warm GET (h2_runtime_swap_proxima_native / w5_single_stream_get)

| runtime | dispatch channel | ${PLATFORM} (mean) | ${PLATFORM} (CoV) | Linux (mean) | Linux (CoV) | unit |
|---|---|---|---|---|---|---|
| default tokio multi-thread | tokio internal | ${h2_default} | ${h2_default_cov} | pending | pending | per-request latency |
| per-core tokio | flume | ${h2_percore} | ${h2_percore_cov} | pending | pending | per-request latency |
| prime native | prime-inbox-alloc | ${h2_prime} | ${h2_prime_cov} | pending | pending | per-request latency |

## spawn-burst (spawn_burst_1k — dispatch-cost isolator)

| runtime | dispatch channel | ${PLATFORM} (mean) | ${PLATFORM} (CoV) | Linux (mean) | Linux (CoV) | unit |
|---|---|---|---|---|---|---|
| default tokio multi-thread | tokio internal | ${sb_default} | ${sb_default_cov} | pending | pending | ns/task |
| per-core tokio | flume | ${sb_percore} | ${sb_default_cov} | pending | pending | ns/task |
| prime native | prime-inbox-alloc | ${sb_prime} | ${sb_prime_cov} | pending | pending | ns/task |
MARKDOWN

printf 'results written to %s\n' "$RESULTS"
