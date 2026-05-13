#!/usr/bin/env bash
# answers: does prime+compat cost 15-25% on multi-conn h2 vs TokioPerCoreRuntime?
# runs bench_compat_libraries (library × runtime matrix) and bench_runtime_compat
# (proxima internal cost of compat), then emits a results table.
#
# usage: scripts/bench-tokio-compat.sh
# output: benches/RESULTS_bench-tokio-compat_<platform>.md

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_bench-common.sh
source "${script_dir}/_bench-common.sh"

crate_dir="$(cd "$script_dir/.." && pwd)"
cd "$crate_dir"

PLATFORM="$(detect_platform)"

FEATURES="http2,tcp,http1,runtime-tokio,runtime-prime-full,prime-tokio-compat,tls"
RESULTS="benches/RESULTS_bench-tokio-compat_${PLATFORM}.md"
CRITERION_DIR="target/criterion"
LOGS_DIR="/tmp/bench-tokio-compat-logs"
mkdir -p "$LOGS_DIR"

printf 'bench-tokio-compat (%s)\n' "$PLATFORM"

# ---------------------------------------------------------------------------
# run benches
# ---------------------------------------------------------------------------

printf -- '--- bench_compat_libraries (library × runtime matrix) ---\n'
cargo bench --no-default-features --features "$FEATURES" \
    --bench bench_compat_libraries \
    2>&1 | tee "${LOGS_DIR}/bench_compat_libraries.log" || true

printf -- '--- bench_runtime_compat (proxima internal compat cost) ---\n'
cargo bench --no-default-features --features "$FEATURES" \
    --bench bench_runtime_compat \
    2>&1 | tee "${LOGS_DIR}/bench_runtime_compat.log" || true

# ---------------------------------------------------------------------------
# parse criterion estimates.json via jq
# ---------------------------------------------------------------------------

read_mean_fmt() {
    local group="$1"
    local arm="$2"
    local estimates_file
    estimates_file="$(find_estimates "$CRITERION_DIR" "$group" "$arm")"
    local mean_ns
    mean_ns="$(extract_mean_ns "$estimates_file")"
    format_time_ns "$mean_ns"
}

read_mean_raw() {
    local group="$1"
    local arm="$2"
    local estimates_file
    estimates_file="$(find_estimates "$CRITERION_DIR" "$group" "$arm")"
    extract_mean_ns "$estimates_file"
}

ratio() {
    local compat_ns="$1"
    local percore_ns="$2"
    if [[ "$compat_ns" == "pending" || "$percore_ns" == "pending" || "$percore_ns" == "0" ]]; then
        printf 'pending'
    else
        printf '%.2fx' "$(awk -v c="$compat_ns" -v p="$percore_ns" 'BEGIN { printf "%.4f", c/p }')"
    fi
}

# ---------------------------------------------------------------------------
# read per-arm means for each group
# ---------------------------------------------------------------------------

# single_stream group
ss_hyper_ct="$(read_mean_fmt    "compat_libs_single_stream" "hyper_current_thread")"
ss_hyper_mt="$(read_mean_fmt    "compat_libs_single_stream" "hyper_default_tokio")"
ss_hyper_pc="$(read_mean_fmt    "compat_libs_single_stream" "hyper_per_core")"
ss_hyper_compat="$(read_mean_fmt "compat_libs_single_stream" "hyper_prime_compat")"
ss_pingora_ct="$(read_mean_fmt   "compat_libs_single_stream" "pingora_current_thread")"
ss_pingora_mt="$(read_mean_fmt   "compat_libs_single_stream" "pingora_default_tokio")"
ss_pingora_pc="$(read_mean_fmt   "compat_libs_single_stream" "pingora_per_core")"
ss_pingora_compat="$(read_mean_fmt "compat_libs_single_stream" "pingora_prime_compat")"
ss_proxima_ct="$(read_mean_fmt   "compat_libs_single_stream" "proxima_current_thread")"
ss_proxima_mt="$(read_mean_fmt   "compat_libs_single_stream" "proxima_default_tokio")"
ss_proxima_pc="$(read_mean_fmt   "compat_libs_single_stream" "proxima_per_core")"
ss_proxima_compat="$(read_mean_fmt "compat_libs_single_stream" "proxima_prime_compat")"
ss_hyper_ratio="$(ratio "$(read_mean_raw "compat_libs_single_stream" "hyper_prime_compat")" \
    "$(read_mean_raw "compat_libs_single_stream" "hyper_per_core")")"
ss_pingora_ratio="$(ratio "$(read_mean_raw "compat_libs_single_stream" "pingora_prime_compat")" \
    "$(read_mean_raw "compat_libs_single_stream" "pingora_per_core")")"
ss_proxima_ratio="$(ratio "$(read_mean_raw "compat_libs_single_stream" "proxima_prime_compat")" \
    "$(read_mean_raw "compat_libs_single_stream" "proxima_per_core")")"

# h2_fanin group
fi_hyper_ct="$(read_mean_fmt    "compat_libs_h2_fanin" "hyper_current_thread")"
fi_hyper_mt="$(read_mean_fmt    "compat_libs_h2_fanin" "hyper_default_tokio")"
fi_hyper_pc="$(read_mean_fmt    "compat_libs_h2_fanin" "hyper_per_core")"
fi_hyper_compat="$(read_mean_fmt "compat_libs_h2_fanin" "hyper_prime_compat")"
fi_pingora_ct="$(read_mean_fmt   "compat_libs_h2_fanin" "pingora_current_thread")"
fi_pingora_mt="$(read_mean_fmt   "compat_libs_h2_fanin" "pingora_default_tokio")"
fi_pingora_pc="$(read_mean_fmt   "compat_libs_h2_fanin" "pingora_per_core")"
fi_pingora_compat="$(read_mean_fmt "compat_libs_h2_fanin" "pingora_prime_compat")"
fi_proxima_ct="$(read_mean_fmt   "compat_libs_h2_fanin" "proxima_current_thread")"
fi_proxima_mt="$(read_mean_fmt   "compat_libs_h2_fanin" "proxima_default_tokio")"
fi_proxima_pc="$(read_mean_fmt   "compat_libs_h2_fanin" "proxima_per_core")"
fi_proxima_compat="$(read_mean_fmt "compat_libs_h2_fanin" "proxima_prime_compat")"
fi_hyper_ratio="$(ratio "$(read_mean_raw "compat_libs_h2_fanin" "hyper_prime_compat")" \
    "$(read_mean_raw "compat_libs_h2_fanin" "hyper_per_core")")"
fi_pingora_ratio="$(ratio "$(read_mean_raw "compat_libs_h2_fanin" "pingora_prime_compat")" \
    "$(read_mean_raw "compat_libs_h2_fanin" "pingora_per_core")")"
fi_proxima_ratio="$(ratio "$(read_mean_raw "compat_libs_h2_fanin" "proxima_prime_compat")" \
    "$(read_mean_raw "compat_libs_h2_fanin" "proxima_per_core")")"

# multicore_fanin group
mc_hyper_mt="$(read_mean_fmt    "compat_libs_multicore_fanin" "hyper_multi_thread")"
mc_hyper_pc="$(read_mean_fmt    "compat_libs_multicore_fanin" "hyper_per_core")"
mc_hyper_compat="$(read_mean_fmt "compat_libs_multicore_fanin" "hyper_prime_compat")"
mc_pingora_mt="$(read_mean_fmt   "compat_libs_multicore_fanin" "pingora_multi_thread")"
mc_pingora_pc="$(read_mean_fmt   "compat_libs_multicore_fanin" "pingora_per_core")"
mc_pingora_compat="$(read_mean_fmt "compat_libs_multicore_fanin" "pingora_prime_compat")"
mc_proxima_mt="$(read_mean_fmt   "compat_libs_multicore_fanin" "proxima_multi_thread")"
mc_proxima_pc="$(read_mean_fmt   "compat_libs_multicore_fanin" "proxima_per_core")"
mc_proxima_compat="$(read_mean_fmt "compat_libs_multicore_fanin" "proxima_prime_compat")"
mc_hyper_ratio="$(ratio "$(read_mean_raw "compat_libs_multicore_fanin" "hyper_prime_compat")" \
    "$(read_mean_raw "compat_libs_multicore_fanin" "hyper_per_core")")"
mc_pingora_ratio="$(ratio "$(read_mean_raw "compat_libs_multicore_fanin" "pingora_prime_compat")" \
    "$(read_mean_raw "compat_libs_multicore_fanin" "pingora_per_core")")"
mc_proxima_ratio="$(ratio "$(read_mean_raw "compat_libs_multicore_fanin" "proxima_prime_compat")" \
    "$(read_mean_raw "compat_libs_multicore_fanin" "proxima_per_core")")"

# HDR percentiles from bench_compat_libraries stdout
compat_log="${LOGS_DIR}/bench_compat_libraries.log"

# ---------------------------------------------------------------------------
# emit RESULTS doc
# ---------------------------------------------------------------------------

cat > "$RESULTS" << MARKDOWN
# bench-tokio-compat — ${PLATFORM}

Measures whether \`prime+compat\` costs 15-25% on multi-conn h2 vs
\`TokioPerCoreRuntime\` (the apples-to-apples baseline). Three libraries
(hyper, pingora, proxima) × three workloads (single_stream, h2_fanin,
multicore_fanin). \`per_core\` column = TokioPerCoreRuntime; \`compat/per_core\`
ratio > 1.0x means compat is slower by that factor.

Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')

---

## single_stream (1 req/iter, 1 TCP connection)

| library | current_thread | multi_thread | per_core | prime_compat | compat/per_core ratio |
|---------|---------------|-------------|----------|-------------|----------------------|
| hyper   | ${ss_hyper_ct} | ${ss_hyper_mt} | ${ss_hyper_pc} | ${ss_hyper_compat} | ${ss_hyper_ratio} |
| pingora | ${ss_pingora_ct} | ${ss_pingora_mt} | ${ss_pingora_pc} | ${ss_pingora_compat} | ${ss_pingora_ratio} |
| proxima | ${ss_proxima_ct} | ${ss_proxima_mt} | ${ss_proxima_pc} | ${ss_proxima_compat} | ${ss_proxima_ratio} |

### single_stream HDR percentiles (bench_compat_libraries stdout)

| arm | p50 | p90 | p99 | p999 | max |
|-----|-----|-----|-----|------|-----|
| hyper_prime_compat | $(parse_hdr_line "$compat_log" "hyper_prime_compat" "p50") | $(parse_hdr_line "$compat_log" "hyper_prime_compat" "p90") | $(parse_hdr_line "$compat_log" "hyper_prime_compat" "p99") | $(parse_hdr_line "$compat_log" "hyper_prime_compat" "p999") | $(parse_hdr_line "$compat_log" "hyper_prime_compat" "max") |
| proxima_prime_compat | $(parse_hdr_line "$compat_log" "proxima_prime_compat" "p50") | $(parse_hdr_line "$compat_log" "proxima_prime_compat" "p90") | $(parse_hdr_line "$compat_log" "proxima_prime_compat" "p99") | $(parse_hdr_line "$compat_log" "proxima_prime_compat" "p999") | $(parse_hdr_line "$compat_log" "proxima_prime_compat" "max") |

---

## h2_fanin (${FANIN_STREAMS:-32} concurrent streams, 1 TCP connection)

| library | current_thread | multi_thread | per_core | prime_compat | compat/per_core ratio |
|---------|---------------|-------------|----------|-------------|----------------------|
| hyper   | ${fi_hyper_ct} | ${fi_hyper_mt} | ${fi_hyper_pc} | ${fi_hyper_compat} | ${fi_hyper_ratio} |
| pingora | ${fi_pingora_ct} | ${fi_pingora_mt} | ${fi_pingora_pc} | ${fi_pingora_compat} | ${fi_pingora_ratio} |
| proxima | ${fi_proxima_ct} | ${fi_proxima_mt} | ${fi_proxima_pc} | ${fi_proxima_compat} | ${fi_proxima_ratio} |

---

## multicore_fanin (CORES ports × PER_CONN_STREAMS streams — the key cost-model workload)

| library | multi_thread | per_core | prime_compat | compat/per_core ratio |
|---------|-------------|----------|-------------|----------------------|
| hyper   | ${mc_hyper_mt} | ${mc_hyper_pc} | ${mc_hyper_compat} | ${mc_hyper_ratio} |
| pingora | ${mc_pingora_mt} | ${mc_pingora_pc} | ${mc_pingora_compat} | ${mc_pingora_ratio} |
| proxima | ${mc_proxima_mt} | ${mc_proxima_pc} | ${mc_proxima_compat} | ${mc_proxima_ratio} |

---

_Run \`scripts/bench-tokio-compat.sh\` to replace \`pending\` values with measured results._
MARKDOWN

printf 'results written to %s\n' "$RESULTS"
