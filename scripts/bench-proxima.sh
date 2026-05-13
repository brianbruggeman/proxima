#!/usr/bin/env bash
# produces a per-workload absolute-perf coverage map for proxima across every
# bench category. comparison vs alternative implementations lives in
# bench-vs-{hyper,pingora,rayon}. runtime variance lives in bench-proxima-runtimes.
#
# usage: scripts/bench-proxima.sh
#
# outputs: benches/RESULTS_bench-proxima_<platform>.md

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_bench-common.sh
source "${script_dir}/_bench-common.sh"

FEATURES="runtime-prime-full,runtime-tokio,http,tls,websocket,websocket-frame,websocket-upstream,redis-listener,memcached-listener,mqtt-listener,amqp-listener,kafka-listener,grpc-framing,protobuf-wire,dns-substrate,h3-upstream,runtime-prime-bgpool-rayon,runtime-prime-bgpool-par,runtime-prime-bgpool-async,rayon,prime-tokio-compat"

crate_dir="$(cd "$script_dir/.." && pwd)"
cd "$crate_dir"

PLATFORM="$(detect_platform)"

RESULTS="benches/RESULTS_bench-proxima_${PLATFORM}.md"
CRITERION_DIR="target/criterion"
LOGS_DIR="/tmp/bench-proxima-logs"
mkdir -p "$LOGS_DIR"

printf 'bench-proxima: platform=%s\n' "$PLATFORM"
printf 'features: %s\n' "$FEATURES"
printf 'output:   %s\n' "$RESULTS"

# extract_stats <estimates_json_path>
# prints: "<formatted_time> <cov_pct>" or "pending n/a" if file absent
extract_stats() {
  local json_path="$1"
  local mean_ns cov formatted
  mean_ns="$(extract_mean_ns "$json_path")"
  cov="$(extract_cov "$json_path")"
  formatted="$(format_time_ns "$mean_ns")"
  printf '%s %s' "$formatted" "$cov"
}

# run one bench and append its result rows to RESULTS
# usage: run_bench <category> <bench_name> [filter]
run_bench() {
  local category="$1"
  local bench_name="$2"
  local filter="${3:-}"

  printf -- '--- %s / %s%s ---\n' "$category" "$bench_name" "${filter:+ (filter: $filter)}"

  if [[ -n "$filter" ]]; then
    cargo bench \
      --no-default-features \
      --features "$FEATURES" \
      --bench "$bench_name" \
      -- "$filter" 2>&1 | tee "${LOGS_DIR}/${bench_name}.log" || true
  else
    cargo bench \
      --no-default-features \
      --features "$FEATURES" \
      --bench "$bench_name" \
      2>&1 | tee "${LOGS_DIR}/${bench_name}.log" || true
  fi

  # walk criterion output for this bench and emit table rows
  local bench_criterion_dir="${CRITERION_DIR}"
  if [[ ! -d "$bench_criterion_dir" ]]; then
    printf '| %s | (no criterion output) | pending | pending | ns | %s |\n' \
      "$bench_name" "$bench_name" >> "$RESULTS"
    return
  fi

  # criterion writes: target/criterion/<group>/<arm>[/new]/estimates.json
  # group names come from the bench source; walk all subdirs two levels deep
  local found=0
  while IFS= read -r -d '' estimates_json; do
    local arm_dir group_dir
    arm_dir="$(dirname "$estimates_json")"
    # handle new/ layout: arm_dir may be the new/ dir; step up if so
    if [[ "$(basename "$arm_dir")" == "new" ]]; then
      arm_dir="$(dirname "$arm_dir")"
    fi
    group_dir="$(dirname "$arm_dir")"
    local group arm
    group="$(basename "$group_dir")"
    arm="$(basename "$arm_dir")"

    # skip arms that don't match filter when filter is set
    if [[ -n "$filter" ]]; then
      if [[ "$arm" != *"$filter"* && "$group" != *"$filter"* ]]; then
        continue
      fi
    fi

    local formatted cov
    read -r formatted cov < <(extract_stats "$estimates_json")
    printf '| %s/%s | mean | %s | pending | %s | %s |\n' \
      "$group" "$arm" "$formatted" "$cov" "$bench_name" >> "$RESULTS"
    found=$((found + 1))
  done < <(find "$bench_criterion_dir" -name "estimates.json" -print0 2>/dev/null)

  if [[ $found -eq 0 ]]; then
    printf '| %s | mean | pending | pending | n/a | %s |\n' \
      "$bench_name" "$bench_name" >> "$RESULTS"
  fi
}

# ── initialise results file ──────────────────────────────────────────────────

cat > "$RESULTS" << HEADER
# proxima bench results — ${PLATFORM}

Absolute proxima performance across every workload category using the
canonical ship feature set. No comparison axis — see bench-vs-{hyper,pingora,rayon}
for competitive numbers.

Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')

Features: \`${FEATURES}\`

---

HEADER

# ── HTTP wire ────────────────────────────────────────────────────────────────

printf '## HTTP wire\n\n' >> "$RESULTS"
printf '| workload | metric | %s | linux | unit | bench file |\n' "$PLATFORM" >> "$RESULTS"
printf '|----------|--------|-----|-------|------|------------|\n' >> "$RESULTS"

for bench in h1_dispatch h2_dispatch h3_dispatch \
             h2_native_vs_h2_crate h2_native_vs_h2_crate_e2e h2_native_vs_h2_crate_alloc; do
  run_bench "http-wire" "$bench"
done

# streaming benches: p50 / p99 / p999 columns
printf '\n### HTTP streaming (p50 / p99 / p999)\n\n' >> "$RESULTS"
printf '| workload | p50 (%s) | p99 (%s) | p999 (%s) | p50 linux | p99 linux | p999 linux | unit | bench file |\n' \
  "$PLATFORM" "$PLATFORM" "$PLATFORM" >> "$RESULTS"
printf '|----------|----------|----------|-----------|-----------|-----------|------------|------|------------|\n' >> "$RESULTS"

for bench in h1_streaming h2_streaming h2_streaming_responses h2_tail_scaling h3_streaming \
             h2_native_vs_h2_crate_tail h3_streaming_responses h3_tail_scaling h3_tail_multi_conn; do
  run_bench "http-wire/streaming" "$bench"
done

# ── sans-IO parsers ──────────────────────────────────────────────────────────

printf '\n## sans-IO parsers\n\n' >> "$RESULTS"
printf '| workload | metric | %s | linux | unit | bench file |\n' "$PLATFORM" >> "$RESULTS"
printf '|----------|--------|-----|-------|------|------------|\n' >> "$RESULTS"

for bench in hpack_block hpack_huffman hpack_integer hpack_static_table \
             h2_native_frame bench_websocket_frame bench_dns \
             bench_protobuf_wire proxy_protocol_parse simd_json_decode; do
  run_bench "sans-io" "$bench"
done

# ── state protocols ──────────────────────────────────────────────────────────

printf '\n## state protocols\n\n' >> "$RESULTS"
printf '| workload | metric | %s | linux | unit | bench file |\n' "$PLATFORM" >> "$RESULTS"
printf '|----------|--------|-----|-------|------|------------|\n' >> "$RESULTS"

for bench in bench_redis bench_memcached bench_mqtt bench_amqp bench_kafka bench_grpc_framing \
             bench_ws_upstream bench_h3_upstream; do
  run_bench "state-protocols" "$bench"
done

# ── scheduling ───────────────────────────────────────────────────────────────

printf '\n## scheduling\n\n' >> "$RESULTS"
printf '| workload | metric | %s | linux | unit | bench file |\n' "$PLATFORM" >> "$RESULTS"
printf '|----------|--------|-----|-------|------|------------|\n' >> "$RESULTS"

for bench in bench_spawn_burst bench_open_loop_driver bench_fairness_imbalanced \
             bench_timer bench_reactor bench_local_executor bench_h2_spawn_blocking; do
  run_bench "scheduling" "$bench"
done

# ── channels ─────────────────────────────────────────────────────────────────

printf '\n## channels\n\n' >> "$RESULTS"
printf '| workload | metric | %s | linux | unit | bench file |\n' "$PLATFORM" >> "$RESULTS"
printf '|----------|--------|-----|-------|------|------------|\n' >> "$RESULTS"

# bench_inbox has multiple arms — filter to proxima arm only
run_bench "channels" "bench_inbox" "proxima"

# ── bgpool ────────────────────────────────────────────────────────────────────

printf '\n## bgpool\n\n' >> "$RESULTS"
printf '| workload | metric | %s | linux | unit | bench file |\n' "$PLATFORM" >> "$RESULTS"
printf '|----------|--------|-----|-------|------|------------|\n' >> "$RESULTS"

# bench_background_pool has multiple arms — filter to proxima arm only
run_bench "bgpool" "bench_background_pool" "proxima"

# ── pipeline ────────────────────────────────────────────────────────────────

printf '\n## pipeline\n\n' >> "$RESULTS"
printf '| workload | metric | %s | linux | unit | bench file |\n' "$PLATFORM" >> "$RESULTS"
printf '|----------|--------|-----|-------|------|------------|\n' >> "$RESULTS"

for bench in tee_backpressure tee_sink_primitives substrate_dispatch \
             hot_apply_build swap_under_load; do
  run_bench "pipeline" "$bench"
done

# ── end-to-end ────────────────────────────────────────────────────────────────

printf '\n## end-to-end\n\n' >> "$RESULTS"
printf '| workload | metric | %s | linux | unit | bench file |\n' "$PLATFORM" >> "$RESULTS"
printf '|----------|--------|-----|-------|------|------------|\n' >> "$RESULTS"

for bench in request_path network_throughput per_core_vs_arcswap perf_audit; do
  run_bench "e2e" "$bench"
done

# ── done ──────────────────────────────────────────────────────────────────────

printf '\n---\nRun completed: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$RESULTS"
printf 'bench-proxima complete. results: %s\n' "$RESULTS"
