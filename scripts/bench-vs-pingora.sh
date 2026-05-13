#!/usr/bin/env bash
# bench-vs-pingora: proxima h2 native vs pingora on pingora's home turf.
# measures h2 edge reverse-proxy with multi-connection scaling and tail latency.
#
# usage: scripts/bench-vs-pingora.sh
#
# outputs: benches/RESULTS_bench-vs-pingora_<platform>.md
#
# METHODOLOGY NOTE: single-run numbers at conn=16+ are unreliable on this
# workload (loopback TCP + thread scheduling jitter; CV 5-16% observed on
# Linux i7-9700K, per RESULTS_linux.md line 14). Set TRIALS=5 for doc-quality
# data: the script will run 5 trials with --save-baseline and report the median.
#
# env vars:
#   TRIALS  — set to 5 to run 5-trial median mode (default: 1)

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_bench-common.sh
source "${script_dir}/_bench-common.sh"

crate_dir="$(cd "$script_dir/.." && pwd)"
cd "$crate_dir"

PLATFORM="$(detect_platform)"
TRIALS="${TRIALS:-1}"

FEATURES="http1,http2,tcp,runtime-tokio,runtime-prime-full,tls"
RESULTS="benches/RESULTS_bench-vs-pingora_${PLATFORM}.md"
CRITERION_DIR="target/criterion"
LOGS_DIR="/tmp/bench-vs-pingora-logs"
mkdir -p "$LOGS_DIR"

printf 'bench-vs-pingora: platform=%s\n' "$PLATFORM"
printf 'features: %s\n' "$FEATURES"
printf 'output:   %s\n' "$RESULTS"

parse_rps_from_ns() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        printf 'pending'
        return
    fi
    local mean
    mean="$(jq -r '.mean.point_estimate' "$path" 2>/dev/null || echo 'null')"
    if [[ "$mean" == "null" || "$mean" == "0" ]]; then
        printf 'pending'
        return
    fi
    awk -v ns="$mean" 'BEGIN { printf "%.0f", 1000000000 / ns }'
}

# get mean ns with optional 5-trial median
get_mean_ns_pingora() {
    local group="$1"
    local arm="$2"
    if [[ "$TRIALS" -ge 5 ]]; then
        local trial_values=()
        for trial in 1 2 3 4 5; do
            local trial_file="${CRITERION_DIR}/${group}/${arm}/trial-${trial}/estimates.json"
            local val
            val="$(extract_mean_ns "$trial_file")"
            [[ "$val" != "pending" ]] && trial_values+=("$val")
        done
        if [[ ${#trial_values[@]} -gt 0 ]]; then
            median_of "${trial_values[@]}"
        else
            local estimates_file
            estimates_file="$(find_estimates "$CRITERION_DIR" "$group" "$arm")"
            extract_mean_ns "$estimates_file"
        fi
    else
        local estimates_file
        estimates_file="$(find_estimates "$CRITERION_DIR" "$group" "$arm")"
        extract_mean_ns "$estimates_file"
    fi
}

run_bench_pingora() {
    local bench_name="$1"
    local log_file="$2"
    if [[ "$TRIALS" -ge 5 ]]; then
        printf -- '--- running %s (5-trial) ---\n' "$bench_name"
        for trial in 1 2 3 4 5; do
            cargo bench \
                --no-default-features \
                --features "$FEATURES" \
                --bench "$bench_name" \
                -- --save-baseline "trial-${trial}" \
                2>&1 | tee "${LOGS_DIR}/${bench_name}_trial${trial}.log" || true
        done
        cat "${LOGS_DIR}/${bench_name}_trial5.log" > "$log_file" || true
    else
        printf -- '--- running %s ---\n' "$bench_name"
        cargo bench \
            --no-default-features \
            --features "$FEATURES" \
            --bench "$bench_name" \
            2>&1 | tee "$log_file" || true
    fi
}

warm_log="${LOGS_DIR}/h2_vs_pingora.log"
multi_log="${LOGS_DIR}/h2_tail_multi_conn.log"
h1_log="${LOGS_DIR}/h1_vs_pingora.log"

run_bench_pingora "h2_vs_pingora" "$warm_log"
run_bench_pingora "h2_tail_multi_conn" "$multi_log"
run_bench_pingora "h1_vs_pingora" "$h1_log"

{
    printf '# bench-vs-pingora — %s\n\n' "$PLATFORM"
    printf 'proxima h2 native vs pingora on pingora'\''s home turf: h2 edge reverse-proxy\n'
    printf 'with multi-connection scaling and tail latency.\n\n'
    printf '> M1 numbers below are single-run; 5-trial-median is the doc-quality bar —\n'
    printf '> re-run with 5 trials and --save-baseline before publishing.\n\n'
    printf 'Generated: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'Features: `%s`\n\n' "$FEATURES"
    printf '---\n\n'

    printf '## h1 warm GET (h1_vs_pingora)\n\n'
    printf '| arm | mean latency | CoV | RPS (est) | p50 | p99 | bench file |\n'
    printf '|-----|-------------|-----|-----------|-----|-----|------------|\n'

    for h1_arm in \
        "proxima h1 (loopback)" \
        "hyper h1 (loopback)" \
        "pingora h1 (loopback)"
    do
        arm_dir="${CRITERION_DIR}/h1_vs_pingora_warm/${h1_arm}"
        path="${arm_dir}/new/estimates.json"
        [[ ! -f "$path" ]] && path="${arm_dir}/estimates.json"
        mean_ns_raw="$(get_mean_ns_pingora "h1_vs_pingora_warm" "$h1_arm")"
        mean_fmt="$(format_time_ns "$mean_ns_raw")"
        cov="$(extract_cov "$path")"
        rps="$(parse_rps_from_ns "$path")"
        p50="$(parse_hdr_line "$h1_log" "$h1_arm" "p50")"
        p99="$(parse_hdr_line "$h1_log" "$h1_arm" "p99")"
        printf '| %s | %s | %s | %s | %s | %s | h1_vs_pingora |\n' \
            "$h1_arm" "$mean_fmt" "$cov" "$rps" "$p50" "$p99"
    done

    printf '\n## h2 warm GET (h2_vs_pingora)\n\n'
    printf '| arm | mean latency | CoV | RPS (est) | p50 | p99 | bench file |\n'
    printf '|-----|-------------|-----|-----------|-----|-----|------------|\n'

    for warm_arm in \
        "proxima::serve_h2_connection (warm)" \
        "hyper::http2::Builder (warm)" \
        "pingora::http::v2::HttpSession (warm)"
    do
        arm_dir="${CRITERION_DIR}/h2_vs_pingora_warm/${warm_arm}"
        path="${arm_dir}/new/estimates.json"
        [[ ! -f "$path" ]] && path="${arm_dir}/estimates.json"
        mean_ns_raw="$(get_mean_ns_pingora "h2_vs_pingora_warm" "$warm_arm")"
        mean_fmt="$(format_time_ns "$mean_ns_raw")"
        cov="$(extract_cov "$path")"
        rps="$(parse_rps_from_ns "$path")"
        p50="$(parse_hdr_line "$warm_log" "$warm_arm" "p50")"
        p99="$(parse_hdr_line "$warm_log" "$warm_arm" "p99")"
        printf '| %s | %s | %s | %s | %s | %s | h2_vs_pingora |\n' \
            "$warm_arm" "$mean_fmt" "$cov" "$rps" "$p50" "$p99"
    done

    printf '\n## multi-conn sweep — p50 / p99 / p999 (h2_tail_multi_conn)\n\n'
    printf '| arm | conn=1 p50 | conn=4 p50 | conn=16 p50 | conn=64 p50 | p99 @ conn=64 | p999 @ conn=64 |\n'
    printf '|-----|-----------|-----------|------------|------------|--------------|----------------|\n'

    for arm in \
        "proxima_native_default_tokio" \
        "proxima_native_per_core" \
        "hyper_default_tokio" \
        "pingora_default_tokio"
    do
        p50_1="pending"
        p50_4="pending"
        p50_16="pending"
        p50_64="pending"
        p99_64="pending"
        p999_64="pending"

        for conn in 1 4 16 64; do
            group_name="h2_tail_multi_conn"
            arm_p50="${arm}/conn=${conn}/p50"
            arm_p99="${arm}/conn=${conn}/p99"
            arm_p999="${arm}/conn=${conn}/p999"

            path_p50="${CRITERION_DIR}/${group_name}/${arm_p50}/new/estimates.json"
            [[ ! -f "$path_p50" ]] && path_p50="${CRITERION_DIR}/${group_name}/${arm_p50}/estimates.json"

            raw_p50="$(extract_mean_ns "$path_p50")"
            val="$(format_time_ns "$raw_p50")"

            case "$conn" in
                1)  p50_1="$val" ;;
                4)  p50_4="$val" ;;
                16) p50_16="$val" ;;
                64) p50_64="$val"
                    path_p99="${CRITERION_DIR}/${group_name}/${arm_p99}/new/estimates.json"
                    [[ ! -f "$path_p99" ]] && path_p99="${CRITERION_DIR}/${group_name}/${arm_p99}/estimates.json"
                    raw_p99="$(extract_mean_ns "$path_p99")"
                    p99_64="$(format_time_ns "$raw_p99")"
                    path_p999="${CRITERION_DIR}/${group_name}/${arm_p999}/new/estimates.json"
                    [[ ! -f "$path_p999" ]] && path_p999="${CRITERION_DIR}/${group_name}/${arm_p999}/estimates.json"
                    raw_p999="$(extract_mean_ns "$path_p999")"
                    p999_64="$(format_time_ns "$raw_p999")"
                    ;;
            esac
        done

        printf '| %s | %s | %s | %s | %s | %s | %s |\n' \
            "$arm" "$p50_1" "$p50_4" "$p50_16" "$p50_64" "$p99_64" "$p999_64"
    done

    printf '\n## h2_tail_multi_conn HDR percentiles (stdout)\n\n'
    printf '| arm | p50 | p90 | p99 | p999 | max |\n'
    printf '|-----|-----|-----|-----|------|-----|\n'
    for hdr_arm in \
        "proxima_native_default_tokio" \
        "proxima_native_per_core" \
        "hyper_default_tokio" \
        "pingora_default_tokio"
    do
        hdr_p50="$(parse_hdr_line "$multi_log" "$hdr_arm" "p50")"
        hdr_p90="$(parse_hdr_line "$multi_log" "$hdr_arm" "p90")"
        hdr_p99="$(parse_hdr_line "$multi_log" "$hdr_arm" "p99")"
        hdr_p999="$(parse_hdr_line "$multi_log" "$hdr_arm" "p999")"
        hdr_max="$(parse_hdr_line "$multi_log" "$hdr_arm" "max")"
        printf '| %s | %s | %s | %s | %s | %s |\n' \
            "$hdr_arm" "$hdr_p50" "$hdr_p90" "$hdr_p99" "$hdr_p999" "$hdr_max"
    done

    printf '\n## h2_tail_multi_conn phased tail (conn=1 and conn=64)\n\n'
    printf '| arm | phase | p50 | p90 | p99 | p999 | max | count |\n'
    printf '|-----|-------|-----|-----|-----|------|-----|-------|\n'
    for phased_arm in \
        "proxima_native_default_tokio/conn=1" \
        "proxima_native_default_tokio/conn=64" \
        "proxima_native_per_core/conn=1" \
        "proxima_native_per_core/conn=64" \
        "hyper_default_tokio/conn=1" \
        "hyper_default_tokio/conn=64" \
        "pingora_default_tokio/conn=1" \
        "pingora_default_tokio/conn=64"
    do
        for phase in warmup steady spike spindown; do
            ph_p50="$(parse_hdr_phased "$multi_log" "$phased_arm" "$phase" "p50")"
            ph_p90="$(parse_hdr_phased "$multi_log" "$phased_arm" "$phase" "p90")"
            ph_p99="$(parse_hdr_phased "$multi_log" "$phased_arm" "$phase" "p99")"
            ph_p999="$(parse_hdr_phased "$multi_log" "$phased_arm" "$phase" "p999")"
            ph_max="$(parse_hdr_phased "$multi_log" "$phased_arm" "$phase" "max")"
            ph_count="$(parse_hdr_phased "$multi_log" "$phased_arm" "$phase" "count")"
            printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' \
                "$phased_arm" "$phase" "$ph_p50" "$ph_p90" "$ph_p99" "$ph_p999" "$ph_max" "$ph_count"
        done
    done

    printf '\n---\nRun completed: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} > "$RESULTS"

printf 'wrote %s\n' "$RESULTS"
