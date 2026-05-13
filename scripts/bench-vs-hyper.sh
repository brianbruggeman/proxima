#!/usr/bin/env bash
# bench-vs-hyper: proxima h1/h2 native server vs hyper on hyper's home turf.
#
# Three arms per workload:
#   proxima(prime)              — proxima h2/h1 listener on prime native runtime
#   proxima(per-core tokio)     — proxima h2/h1 listener on TokioPerCoreRuntime
#   hyper(tokio multi_thread)   — hyper http1/http2::Builder on default tokio
#
# Four workloads:
#   h1_vs_hyper           — h1 warm GET head-to-head
#   h2_vs_hyper           — h2 warm GET head-to-head
#   h1_streaming          — h1 streaming (proxima + hyper arm, hdrhistogram)
#   h2_streaming_responses— h2 streaming (proxima + hyper arm, hdrhistogram)
#
# usage: scripts/bench-vs-hyper.sh
# output: benches/RESULTS_bench-vs-hyper_<platform>.md
#
# env vars:
#   TRIALS  — set to 5 to run 5-trial median mode (default: 1)

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_bench-common.sh
source "${script_dir}/_bench-common.sh"

FEATURES="http1,http2,tcp,runtime-tokio,runtime-prime-full,tls"
TRIALS="${TRIALS:-1}"

crate_dir="$(cd "$script_dir/.." && pwd)"
cd "$crate_dir"

PLATFORM="$(detect_platform)"

RESULTS="benches/RESULTS_bench-vs-hyper_${PLATFORM}.md"
CRITERION_DIR="target/criterion"
LOGS_DIR="/tmp/bench-vs-hyper-logs"
mkdir -p "$LOGS_DIR"

printf 'bench-vs-hyper: platform=%s\n' "$PLATFORM"
printf 'features: %s\n' "$FEATURES"
printf 'output:   %s\n' "$RESULTS"

run_bench_single() {
  local bench_name="$1"
  printf -- '--- bench: %s ---\n' "$bench_name"
  cargo bench \
    --no-default-features \
    --features "$FEATURES" \
    --bench "$bench_name" \
    2>&1 | tee "${LOGS_DIR}/${bench_name}.log" || true
}

run_bench_trials() {
  local bench_name="$1"
  printf -- '--- bench (5-trial): %s ---\n' "$bench_name"
  for trial in 1 2 3 4 5; do
    cargo bench \
      --no-default-features \
      --features "$FEATURES" \
      --bench "$bench_name" \
      -- --save-baseline "trial-${trial}" \
      2>&1 | tee "${LOGS_DIR}/${bench_name}_trial${trial}.log" || true
  done
  cat "${LOGS_DIR}/${bench_name}_trial5.log" > "${LOGS_DIR}/${bench_name}.log" || true
}

run_bench() {
  local bench_name="$1"
  if [[ "$TRIALS" -ge 5 ]]; then
    run_bench_trials "$bench_name"
  else
    run_bench_single "$bench_name"
  fi
}

# get mean ns for an arm, computing 5-trial median when TRIALS>=5
get_mean_ns() {
  local group="$1"
  local arm="$2"
  if [[ "$TRIALS" -ge 5 ]]; then
    local trial_values=()
    for trial in 1 2 3 4 5; do
      local estimates_file
      estimates_file="${CRITERION_DIR}/${group}/${arm}/trial-${trial}/estimates.json"
      [[ ! -f "$estimates_file" ]] && \
        estimates_file="${CRITERION_DIR}/${group}/${arm}/new/estimates.json"
      [[ ! -f "$estimates_file" ]] && \
        estimates_file="${CRITERION_DIR}/${group}/${arm}/estimates.json"
      local val
      val="$(extract_mean_ns "$estimates_file")"
      [[ "$val" != "pending" ]] && trial_values+=("$val")
    done
    if [[ ${#trial_values[@]} -gt 0 ]]; then
      median_of "${trial_values[@]}"
    else
      printf 'pending'
    fi
  else
    local estimates_file
    estimates_file="$(find_estimates "$CRITERION_DIR" "$group" "$arm")"
    extract_mean_ns "$estimates_file"
  fi
}

# initialise results file
cat > "$RESULTS" << HEADER
# bench-vs-hyper results — ${PLATFORM}

proxima h1/h2 native server vs hyper on hyper's home turf.

Arms: proxima(prime), proxima(per-core tokio), hyper(tokio multi_thread)

Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')

Features: \`${FEATURES}\`

---

HEADER

# h1 warm GET
printf '## h1 warm GET\n\n' >> "$RESULTS"
printf '| arm | %s mean | %s CoV | linux mean | unit |\n' "$PLATFORM" "$PLATFORM" >> "$RESULTS"
printf '|-----|---------|--------|------------|------|\n' >> "$RESULTS"

run_bench "h1_vs_hyper"

while IFS= read -r -d '' estimates_json; do
  local_arm_dir="$(dirname "$estimates_json")"
  [[ "$(basename "$local_arm_dir")" == "new" ]] && local_arm_dir="$(dirname "$local_arm_dir")"
  local_arm="$(basename "$local_arm_dir")"
  local_group="$(basename "$(dirname "$local_arm_dir")")"
  mean_ns="$(get_mean_ns "$local_group" "$local_arm")"
  formatted="$(format_time_ns "$mean_ns")"
  cov="$(extract_cov "$estimates_json")"
  printf '| %s | %s | %s | pending | ns |\n' "$local_arm" "$formatted" "$cov" >> "$RESULTS"
done < <(find "${CRITERION_DIR}/h1_end_to_end" -name "estimates.json" -print0 2>/dev/null || true)

# h2 warm GET
printf '\n## h2 warm GET\n\n' >> "$RESULTS"
printf '| arm | %s mean | %s CoV | linux mean | unit |\n' "$PLATFORM" "$PLATFORM" >> "$RESULTS"
printf '|-----|---------|--------|------------|------|\n' >> "$RESULTS"

run_bench "h2_vs_hyper"

while IFS= read -r -d '' estimates_json; do
  local_arm_dir="$(dirname "$estimates_json")"
  [[ "$(basename "$local_arm_dir")" == "new" ]] && local_arm_dir="$(dirname "$local_arm_dir")"
  local_arm="$(basename "$local_arm_dir")"
  local_group="$(basename "$(dirname "$local_arm_dir")")"
  mean_ns="$(get_mean_ns "$local_group" "$local_arm")"
  formatted="$(format_time_ns "$mean_ns")"
  cov="$(extract_cov "$estimates_json")"
  printf '| %s | %s | %s | pending | ns |\n' "$local_arm" "$formatted" "$cov" >> "$RESULTS"
done < <(find "${CRITERION_DIR}/h2_end_to_end_warm" -name "estimates.json" -print0 2>/dev/null || true)

# h1 streaming
printf '\n## h1 streaming\n\n' >> "$RESULTS"
printf '| arm | %s mean | %s p50 | %s p99 | %s p999 | linux mean | linux p50 | linux p99 | linux p999 | unit |\n' \
  "$PLATFORM" "$PLATFORM" "$PLATFORM" "$PLATFORM" >> "$RESULTS"
printf '|-----|---------|--------|--------|---------|------------|-----------|-----------|------------|------|\n' >> "$RESULTS"

run_bench "h1_streaming"

h1_stream_log="${LOGS_DIR}/h1_streaming.log"
while IFS= read -r -d '' estimates_json; do
  local_arm_dir="$(dirname "$estimates_json")"
  [[ "$(basename "$local_arm_dir")" == "new" ]] && local_arm_dir="$(dirname "$local_arm_dir")"
  local_arm="$(basename "$local_arm_dir")"
  local_group="$(basename "$(dirname "$local_arm_dir")")"
  mean_ns="$(get_mean_ns "$local_group" "$local_arm")"
  formatted="$(format_time_ns "$mean_ns")"
  p50="$(parse_hdr_line "$h1_stream_log" "$local_arm" "p50")"
  p99="$(parse_hdr_line "$h1_stream_log" "$local_arm" "p99")"
  p999="$(parse_hdr_line "$h1_stream_log" "$local_arm" "p999")"
  printf '| %s | %s | %s | %s | %s | pending | pending | pending | pending | ns |\n' \
    "$local_arm" "$formatted" "$p50" "$p99" "$p999" >> "$RESULTS"
done < <(find "${CRITERION_DIR}/h1_streaming_vs_hyper" -name "estimates.json" -print0 2>/dev/null || true)

printf '\n### h1 streaming — phased tail\n\n' >> "$RESULTS"
printf '| arm | phase | p50 | p90 | p99 | p999 | max | count |\n' >> "$RESULTS"
printf '|-----|-------|-----|-----|-----|------|-----|-------|\n' >> "$RESULTS"
for h1_phased_arm in "hyper::http1 streaming" "proxima::Connection streaming"; do
  for phase in warmup steady spike spindown; do
    ph_p50="$(parse_hdr_phased "$h1_stream_log" "$h1_phased_arm" "$phase" "p50")"
    ph_p90="$(parse_hdr_phased "$h1_stream_log" "$h1_phased_arm" "$phase" "p90")"
    ph_p99="$(parse_hdr_phased "$h1_stream_log" "$h1_phased_arm" "$phase" "p99")"
    ph_p999="$(parse_hdr_phased "$h1_stream_log" "$h1_phased_arm" "$phase" "p999")"
    ph_max="$(parse_hdr_phased "$h1_stream_log" "$h1_phased_arm" "$phase" "max")"
    ph_count="$(parse_hdr_phased "$h1_stream_log" "$h1_phased_arm" "$phase" "count")"
    printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' \
      "$h1_phased_arm" "$phase" "$ph_p50" "$ph_p90" "$ph_p99" "$ph_p999" "$ph_max" "$ph_count" >> "$RESULTS"
  done
done

# h2 streaming responses (standalone binary — no criterion output, HDR only)
printf '\n## h2 streaming responses\n\n' >> "$RESULTS"
printf '| arm | %s mean | %s p50 | %s p99 | %s p999 | linux mean | linux p50 | linux p99 | linux p999 | unit |\n' \
  "$PLATFORM" "$PLATFORM" "$PLATFORM" "$PLATFORM" >> "$RESULTS"
printf '|-----|---------|--------|--------|---------|------------|-----------|-----------|------------|------|\n' >> "$RESULTS"

run_bench "h2_streaming_responses"

h2_resp_log="${LOGS_DIR}/h2_streaming_responses.log"
for stream_arm in "proxima_native (default tokio)" "hyper (default tokio)" "pingora (default tokio)"; do
  p50="$(parse_hdr_line "$h2_resp_log" "$stream_arm" "p50")"
  p99="$(parse_hdr_line "$h2_resp_log" "$stream_arm" "p99")"
  p999="$(parse_hdr_line "$h2_resp_log" "$stream_arm" "p999")"
  printf '| %s | pending | %s | %s | %s | pending | pending | pending | pending | ns |\n' \
    "$stream_arm" "$p50" "$p99" "$p999" >> "$RESULTS"
done

printf '\n### h2 streaming responses — phased tail\n\n' >> "$RESULTS"
printf '| arm | phase | p50 | p90 | p99 | p999 | max | count |\n' >> "$RESULTS"
printf '|-----|-------|-----|-----|-----|------|-----|-------|\n' >> "$RESULTS"
for h2_phased_arm in "proxima_native (default tokio)" "hyper (default tokio)" "pingora (default tokio)"; do
  for phase in warmup steady spike spindown; do
    ph_p50="$(parse_hdr_phased "$h2_resp_log" "$h2_phased_arm" "$phase" "p50")"
    ph_p90="$(parse_hdr_phased "$h2_resp_log" "$h2_phased_arm" "$phase" "p90")"
    ph_p99="$(parse_hdr_phased "$h2_resp_log" "$h2_phased_arm" "$phase" "p99")"
    ph_p999="$(parse_hdr_phased "$h2_resp_log" "$h2_phased_arm" "$phase" "p999")"
    ph_max="$(parse_hdr_phased "$h2_resp_log" "$h2_phased_arm" "$phase" "max")"
    ph_count="$(parse_hdr_phased "$h2_resp_log" "$h2_phased_arm" "$phase" "count")"
    printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' \
      "$h2_phased_arm" "$phase" "$ph_p50" "$ph_p90" "$ph_p99" "$ph_p999" "$ph_max" "$ph_count" >> "$RESULTS"
  done
done

printf '\n---\nRun completed: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$RESULTS"
printf 'bench-vs-hyper complete. results: %s\n' "$RESULTS"
