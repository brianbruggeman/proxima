# Host A M1 Full Bench Matrix — 2026-05-18

- **HEAD**: `fc2d820`
- **Date**: 2026-05-18
- **Wall-clock total**: ~45 min (02:20–03:05 CDT)
- **Machine**: Host A M1 (macOS aarch64)
- **Features**: `runtime-prime-full,runtime-tokio,http2` (+ rayon for bench_background_pool)
- **Criterion params**: `--warm-up-time 2 --measurement-time 4 --sample-size 20`

## Status

| bench | status |
|---|---|
| bench_inbox | ok |
| bench_timer | ok |
| bench_local_executor | ok |
| bench_spawn_burst | ok |
| bench_runtime_swap | ok |
| h2_runtime_swap | ok |
| bench_background_pool | ok |
| h2_native_vs_h2_crate_e2e | ok |
| h2_native_vs_h2_crate_tail | ok |
| h2_native_vs_h2_crate_alloc | ok |
| h2_tail_multi_conn | ok |
| h2_tail_scaling | ok |
| h2_vs_pingora | PARTIAL — duplicate bench ID panic after first group |
| h2_vs_hyper | ok |
| h2_streaming_responses | ok |

**14 clean / 1 partial / 0 hung**

h2_vs_pingora: `Benchmark IDs must be unique within a group` — pre-existing bench bug,
not a runtime regression. First proxima arm (54.5 µs) captured before panic.

## Headline metrics

- **spawn_burst_1k/prime_typed**: 127 µs (7.86 Melem/s) — typed fast-path
- **h2_load_5way/proxima_on_prime**: 61.0 µs (16.4 Kelem/s)
- **h2_load_5way/proxima_on_tokio**: 55.3 µs (18.1 Kelem/s)
- **h2_load_5way/proxima_on_flume**: 54.0 µs (18.5 Kelem/s)
- streaming: proxima 6715 rps vs hyper 4549 (+48%), vs pingora 4710 (+43%)
- tail concurrency=100: proxima p50=456 µs vs h2_crate p50=711 µs (−36%)

## Per-bench medians

### bench_inbox

| group / arm | median |
|---|---|
| inbox_spsc_throughput / proxima | 768 µs (26.0 Melem/s) |
| inbox_spsc_throughput / flume_unbounded | 668 µs (29.9 Melem/s) |
| inbox_spsc_throughput / std_sync_mpsc | 347 µs (57.6 Melem/s) |
| inbox_spsc_throughput / tokio_mpsc | 2.26 ms (8.8 Melem/s) |
| inbox_mpsc_fanin_4 / proxima | 494 µs (40.5 Melem/s) |
| inbox_mpsc_fanin_8 / proxima | 405 µs (49.4 Melem/s) |
| inbox_mpsc_fanin_16 / proxima | 372 µs (53.8 Melem/s) |
| inbox_mpsc_fanin_32 / proxima | 652 µs (30.7 Melem/s) |

### bench_timer

| group / arm | median |
|---|---|
| timer_register_throughput / proxima | 702 µs (14.2 Melem/s) |
| timer_register_throughput / tokio_time | 803 µs (12.5 Melem/s) |
| timer_drain_throughput / proxima | 601 µs (16.6 Melem/s) |
| timer_drain_throughput / tokio_time_joinall | 130.6 ms (76.6 Kelem/s) |
| timer_register_then_cancel / proxima | 781 µs (25.6 Melem/s) |

### bench_local_executor

| group / arm | median |
|---|---|
| local_exec_ready_throughput / proxima | 384 µs (26.1 Melem/s) |
| local_exec_ready_throughput / tokio_localset | 1.48 ms (6.7 Melem/s) |
| local_exec_yield_pingpong / proxima | 145.9 µs (68.5 Melem/s) |
| local_exec_yield_pingpong / tokio_localset | 1.07 ms (9.3 Melem/s) |

### bench_spawn_burst

| group / arm | median |
|---|---|
| spawn_burst_1k / prime_typed | 127 µs (7.86 Melem/s) |
| spawn_burst_1k / prime | 327 µs (3.05 Melem/s) |
| spawn_burst_1k / tokio_per_core | 330 µs (3.03 Melem/s) |
| spawn_burst_10k / prime_typed | 1.28 ms (7.81 Melem/s) |
| spawn_burst_10k / prime | 3.10 ms (3.22 Melem/s) |
| spawn_burst_10k / tokio_per_core | 3.35 ms (2.99 Melem/s) |

### bench_runtime_swap

| group / arm | median |
|---|---|
| runtime_swap_cross_core_spawn / proxima_runtime | 300 µs (3.33 Melem/s) |
| runtime_swap_cross_core_spawn / tokio_per_core | 456 µs (2.19 Melem/s) |

### h2_runtime_swap

| group / arm | median |
|---|---|
| h2_load_5way / proxima_on_prime | 61.0 µs (16.4 Kelem/s) |
| h2_load_5way / proxima_on_tokio | 55.3 µs (18.1 Kelem/s) |
| h2_load_5way / proxima_on_flume | 54.0 µs (18.5 Kelem/s) |
| h2_load_5way / tokio_hyper | 66.3 µs (15.1 Kelem/s) |
| h2_load_5way / pingora | 67.9 µs (14.7 Kelem/s) |
| h2_runtime_swap_proxima_native / default_tokio | 53.8 µs (18.6 Kelem/s) |
| h2_runtime_swap_proxima_native / per_core_runtime | 53.6 µs (18.6 Kelem/s) |
| h2_per_core_vs_hyper_pingora / proxima_native_per_core | 56.7 µs (17.6 Kelem/s) |
| h2_per_core_vs_hyper_pingora / hyper_default_tokio | 66.7 µs (15.0 Kelem/s) |
| h2_per_core_vs_hyper_pingora / pingora_default_tokio | 67.0 µs (14.9 Kelem/s) |

### bench_background_pool

| group / arm | median |
|---|---|
| bg_pool_tiny_jobs / proxima_rayon_backed | 746 µs (1.34 Melem/s) |
| bg_pool_tiny_jobs / proxima_typed | 1.29 ms (778 Kelem/s) |
| bg_pool_tiny_jobs / rayon | 1.26 ms (794 Kelem/s) |
| bg_pool_tiny_jobs / proxima_rayon_backed_dyn | 1.37 ms (731 Kelem/s) |
| bg_pool_tiny_jobs / tokio_spawn_blocking | 1.41 ms (707 Kelem/s) |
| bg_pool_tiny_jobs / proxima_dyn | 2.14 ms (468 Kelem/s) |

### h2_native_vs_h2_crate_e2e

| group / arm | median |
|---|---|
| h2_e2e_best_case_get_minimal / proxima_native | 45.9 µs (21.8 Kelem/s) |
| h2_e2e_best_case_get_minimal / h2_crate | 54.9 µs (18.2 Kelem/s) |
| h2_e2e_balanced_browser_get / proxima_native | 52.7 µs (19.0 Kelem/s) |
| h2_e2e_balanced_browser_get / h2_crate | 56.4 µs (17.7 Kelem/s) |
| h2_e2e_worst_case_post_echo_32kib / h2_crate | 83.4 µs (375 MiB/s) |
| h2_e2e_worst_case_post_echo_32kib / proxima_native | 98.1 µs (319 MiB/s) |

### h2_native_vs_h2_crate_tail (raw histogram)

concurrency=1: proxima_native p50=31 µs p99=72 µs; h2_crate p50=35 µs p99=71 µs
concurrency=10: proxima_native p50=84 µs p99=193 µs; h2_crate p50=93 µs p99=166 µs
concurrency=100: proxima_native p50=456 µs p99=960 µs; h2_crate p50=711 µs p99=1.20 ms

### h2_native_vs_h2_crate_alloc (raw)

proxima_native: rps=23116, allocs/req=20, bytes/req=2354
h2_crate: rps=18755, allocs/req=19, bytes/req=1473
hyper_http2: rps=19633, allocs/req=16, bytes/req=2006

### h2_tail_multi_conn (conn=1 medians)

| arm | median (conn=1) |
|---|---|
| proxima_native_default_tokio | 52.0 µs |
| proxima_native_per_core | 52.2 µs |
| hyper_default_tokio | 66.0 µs |
| pingora_default_tokio | 65.7 µs |

### h2_tail_scaling (streams=1 medians)

| arm | median (streams=1) |
|---|---|
| proxima_native_per_core | 66.2 µs |
| proxima_native_default_tokio | 69.3 µs |
| hyper_default_tokio | 75.6 µs |
| pingora_default_tokio | 78.0 µs |

### h2_vs_pingora

PARTIAL. First arm: proxima warm = 54.5 µs. Panicked on duplicate bench ID.

### h2_vs_hyper

| group / arm | median |
|---|---|
| h2_end_to_end_warm / proxima | 54.2 µs (18.4 Kelem/s) |
| h2_end_to_end_warm / hyper | 53.9 µs (18.5 Kelem/s) |

### h2_streaming_responses (raw)

proxima_native: rps=6715, p50=146 µs, p99=228 µs
hyper: rps=4549, p50=217 µs, p99=307 µs
pingora: rps=4710, p50=209 µs, p99=304 µs
