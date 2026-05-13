#!/usr/bin/env bash
# shared helpers for all bench-*.sh orchestrators.
# source this file; do not execute it directly.

# returns m1 or linux
detect_platform() {
    case "$(uname)" in
        Darwin) printf 'm1' ;;
        Linux)  printf 'linux' ;;
        *)      printf 'unknown' ;;
    esac
}

# find_estimates <criterion_dir> <group> <arm>
# tries new/ layout first, falls back to flat layout
find_estimates() {
    local criterion_dir="$1"
    local group="$2"
    local arm="$3"
    local new_path="${criterion_dir}/${group}/${arm}/new/estimates.json"
    local flat_path="${criterion_dir}/${group}/${arm}/estimates.json"
    if [[ -f "$new_path" ]]; then
        printf '%s' "$new_path"
    elif [[ -f "$flat_path" ]]; then
        printf '%s' "$flat_path"
    else
        printf ''
    fi
}

# extract_mean_ns <estimates_file>
# prints the mean point_estimate in nanoseconds, or "pending"
extract_mean_ns() {
    local estimates_file="$1"
    if [[ -z "$estimates_file" || ! -f "$estimates_file" ]]; then
        printf 'pending'
        return
    fi
    local mean
    mean="$(jq -r '
        if .slope.point_estimate then .slope.point_estimate
        else .mean.point_estimate
        end' "$estimates_file" 2>/dev/null || printf '')"
    if [[ -z "$mean" || "$mean" == "null" || "$mean" == "0" ]]; then
        printf 'pending'
        return
    fi
    awk -v m="$mean" 'BEGIN { printf "%d", m }'
}

# extract_cov <estimates_file>
# prints CoV = std_dev / mean * 100, 1 decimal place, or "n/a"
extract_cov() {
    local estimates_file="$1"
    if [[ -z "$estimates_file" || ! -f "$estimates_file" ]]; then
        printf 'n/a'
        return
    fi
    local mean std_dev
    mean="$(jq -r '.mean.point_estimate' "$estimates_file" 2>/dev/null || printf '')"
    std_dev="$(jq -r '.std_dev.point_estimate' "$estimates_file" 2>/dev/null || printf '')"
    if [[ -z "$mean" || "$mean" == "null" || "$mean" == "0" || -z "$std_dev" || "$std_dev" == "null" ]]; then
        printf 'n/a'
        return
    fi
    awk -v m="$mean" -v s="$std_dev" 'BEGIN { printf "%.1f%%", s / m * 100 }'
}

# format_time_ns <ns_integer_or_pending_or_n/a>
# normalises nanoseconds to a human-readable string
format_time_ns() {
    local ns="$1"
    case "$ns" in
        pending|n/a|'') printf '%s' "$ns"; return ;;
    esac
    awk -v ns="$ns" 'BEGIN {
        if (ns >= 1000000000) {
            printf "%.2f s", ns / 1000000000
        } else if (ns >= 1000000) {
            printf "%.2f ms", ns / 1000000
        } else if (ns >= 1000) {
            printf "%.2f µs", ns / 1000
        } else {
            printf "%d ns", ns
        }
    }'
}

# parse_hdr_line <log_file> <arm_label> <percentile>
# percentile: p50 p90 p99 p999 max
# arm_label is matched as a substring of the line
# supports formats:
#   [hdr] LABEL  p50=VALUE  ...
#   LABEL  rps=... mean=... p50=VALUE ...
#   LABEL count=... mean=... p50=VALUE ...
parse_hdr_line() {
    local log_file="$1"
    local arm_label="$2"
    local percentile="$3"
    if [[ ! -f "$log_file" ]]; then
        printf 'pending'
        return
    fi
    local value
    value="$(grep -F "$arm_label" "$log_file" | grep "${percentile}=" | tail -1 | \
        sed -n "s/.*${percentile}=\([^ ]*\).*/\1/p")"
    if [[ -z "$value" ]]; then
        printf 'pending'
    else
        printf '%s' "$value"
    fi
}

# parse_hdr_phased(log_file, arm_label, phase, percentile)
# matches lines like: arm=<arm_label> phase=<phase> p50=Xns ...
# returns the raw value field (e.g. "412ns") or "pending"
parse_hdr_phased() {
    local log_file="$1"
    local arm="$2"
    local phase="$3"
    local pct="$4"
    if [[ ! -f "$log_file" ]]; then
        printf 'pending'
        return
    fi
    local value
    value="$(grep "arm=${arm}" "$log_file" 2>/dev/null \
        | grep "phase=${phase}" \
        | sed -nE "s/.*${pct}=([^ ]+).*/\1/p" \
        | head -1)"
    if [[ -z "$value" ]]; then
        printf 'pending'
    else
        printf '%s' "$value"
    fi
}

# median_of <v1> <v2> <v3> <v4> <v5> ...
# numeric values only; prints the middle value after sorting
median_of() {
    local sorted
    sorted="$(printf '%s\n' "$@" | sort -n)"
    local count
    count="$(printf '%s\n' "$@" | wc -l | tr -d ' ')"
    local mid
    mid=$(( (count + 1) / 2 ))
    printf '%s\n' "$sorted" | sed -n "${mid}p"
}
