# Host B x86_64 Full Bench Matrix — 2026-05-18

**HEAD:** fc2d820  
**machine:** host-b (Linux x86_64)  
**source:** rsynced from Host A M1, fc2d820 (post spawn-cost-collapse + reactor-slab-pack)  
**total elapsed:** 28m37s (02:22:00–02:50:37)  
**benches clean:** 13/15  
**benches failed:** 2 (bench_background_pool: initial run — feature flag error; h2_vs_pingora: duplicate criterion ID bug)  
**bench_background_pool:** re-run with correct features — CLEAN  
**affinity panics:** none (core_affinity guard at 33dae4f working correctly)

---

## bench_inbox

| arm | median |
|-----|--------|
| inbox_spsc_throughput/proxima | 150.68 µs |
| inbox_spsc_throughput/flume_unbounded | 2.0075 ms |
| inbox_spsc_throughput/flume_bounded | 2.1874 ms |
| inbox_spsc_throughput/std_sync_mpsc | 387.49 µs |
| inbox_spsc_throughput/tokio_mpsc | 2.7998 ms |
| inbox_mpsc_fanin_4/proxima | 243.83 µs |
| inbox_mpsc_fanin_4/flume_bounded | 4.5601 ms |
| inbox_mpsc_fanin_4/tokio_mpsc | 3.6084 ms |
| inbox_mpsc_fanin_8/proxima | 502.51 µs |
| inbox_mpsc_fanin_8/flume_bounded | 8.4904 ms |
| inbox_mpsc_fanin_8/tokio_mpsc | 4.4316 ms |
| inbox_mpsc_fanin_16/proxima | 553.13 µs |
| inbox_mpsc_fanin_16/flume_bounded | 14.787 ms |
| inbox_mpsc_fanin_16/tokio_mpsc | 5.2242 ms |
| inbox_mpsc_fanin_32/proxima | 1.0008 ms |
| inbox_mpsc_fanin_32/flume_bounded | 23.073 ms |
| inbox_mpsc_fanin_32/tokio_mpsc | 7.9517 ms |

## bench_timer

| arm | median |
|-----|--------|
| timer_register_throughput/proxima | 668.42 µs |
| timer_register_throughput/tokio_time | 354.00 µs |
| timer_drain_throughput/proxima | 430.86 µs |
| timer_drain_throughput/tokio_time_joinall | 130.46 ms |
| timer_register_then_cancel/proxima | 716.08 µs |

## bench_local_executor

| arm | median |
|-----|--------|
| local_exec_ready_throughput/proxima | 283.99 µs |
| local_exec_ready_throughput/tokio_localset | 1.7741 ms |
| local_exec_yield_pingpong/proxima | 168.56 µs |
| local_exec_yield_pingpong/tokio_localset | 970.13 µs |

## bench_spawn_burst

| arm | median |
|-----|--------|
| spawn_burst_1k/tokio_per_core | 220.95 µs |
| spawn_burst_1k/prime | 152.60 µs |
| spawn_burst_1k/prime_typed | **69.762 µs** |
| spawn_burst_10k/tokio_per_core | 2.2061 ms |
| spawn_burst_10k/prime | 1.5397 ms |
| spawn_burst_10k/prime_typed | **728.45 µs** |

## bench_runtime_swap

| arm | median |
|-----|--------|
| runtime_swap_cross_core_spawn/tokio_per_core | 219.44 µs |
| runtime_swap_cross_core_spawn/proxima_runtime | 151.02 µs |

## h2_runtime_swap (h2_load_5way headline)

| arm | median |
|-----|--------|
| h2_load_5way/pingora | 47.001 µs |
| h2_load_5way/tokio_hyper | 43.053 µs |
| h2_load_5way/proxima_on_tokio | 35.152 µs |
| h2_load_5way/proxima_on_flume | 30.822 µs |
| h2_load_5way/proxima_on_prime | **33.710 µs** |
| h2_runtime_swap_proxima_native/default_tokio | 31.574 µs |
| h2_runtime_swap_proxima_native/per_core_runtime | 30.212 µs |
| h2_per_core_vs_hyper_pingora/proxima_native_per_core | 30.486 µs |
| h2_per_core_vs_hyper_pingora/hyper_default_tokio | 41.856 µs |
| h2_per_core_vs_hyper_pingora/pingora_default_tokio | 45.111 µs |

## bench_background_pool (rayon-backed variant)

| arm | median |
|-----|--------|
| bg_pool_tiny_jobs/proxima_dyn | 1.0935 ms |
| bg_pool_tiny_jobs/proxima_typed | 1.1261 ms |
| bg_pool_tiny_jobs/rayon | 450.25 µs |
| bg_pool_tiny_jobs/proxima_rayon_backed | **393.26 µs** |
| bg_pool_tiny_jobs/proxima_rayon_backed_dyn | 497.58 µs |
| bg_pool_tiny_jobs/tokio_spawn_blocking | 1.0703 ms |

## h2_native_vs_h2_crate_e2e

| arm | median |
|-----|--------|
| h2_e2e_best_case_get_minimal/h2_crate | 32.258 µs |
| h2_e2e_best_case_get_minimal/proxima_native | 32.433 µs |
| h2_e2e_balanced_browser_get/h2_crate | 40.746 µs |
| h2_e2e_balanced_browser_get/proxima_native | 35.149 µs |
| h2_e2e_worst_case_post_echo_32kib/h2_crate | 67.944 µs |
| h2_e2e_worst_case_post_echo_32kib/proxima_native | 64.718 µs |

## h2_native_vs_h2_crate_tail (hdrhistogram rps/latency)

| arm | count | p50 | p99 |
|-----|-------|-----|-----|
| single-stream h2_crate | 107843 | 27.47 µs | 32.02 µs |
| single-stream proxima_native | 126984 | 23.36 µs | 27.02 µs |
| multi-stream h2_crate (10) | 401140 | 78.46 µs | 119.87 µs |
| multi-stream proxima_native (10) | 370186 | 77.82 µs | 139.65 µs |
| bulk h2_crate | 418557 | 717.82 µs | 806.40 µs |
| bulk proxima_native | 564997 | 532.99 µs | 871.42 µs |

## h2_native_vs_h2_crate_alloc

| arm | rps | bytes/req | allocs/req |
|-----|-----|-----------|------------|
| proxima_h2_crate | 30808 | 1393 | 18.0 |
| proxima_native | 34337 | 2274 | 19.0 |
| hyper_http2 | 29241 | 2006 | 16.0 |

## h2_tail_multi_conn

| arm | conn=1 | conn=4 | conn=16 | conn=64 |
|-----|--------|--------|---------|---------|
| proxima_native_default_tokio | 34.308 µs | 121.11 µs | 153.56 µs | 477.22 µs |
| proxima_native_per_core | 30.044 µs | 66.906 µs | 155.10 µs | 421.67 µs |
| hyper_default_tokio | 42.918 µs | 118.30 µs | 195.13 µs | 619.90 µs |
| pingora_default_tokio | 47.354 µs | 126.69 µs | 196.31 µs | 604.42 µs |

## h2_tail_scaling

| arm | streams=1 | streams=10 | streams=100 |
|-----|-----------|------------|-------------|
| proxima_native_default_tokio | 35.118 µs | 113.25 µs | 627.64 µs |
| proxima_native_per_core | 32.615 µs | 116.55 µs | 630.30 µs |
| hyper_default_tokio | 42.479 µs | 189.42 µs | 1.0283 ms |
| pingora_default_tokio | 48.672 µs | 178.36 µs | 1.1985 ms |

## h2_vs_pingora

FAILED — duplicate criterion benchmark ID bug in bench harness. Not an infra issue.

## h2_vs_hyper

| arm | median |
|-----|--------|
| h2_end_to_end_warm/proxima | 35.822 µs |
| h2_end_to_end_warm/hyper | 34.354 µs |

## h2_streaming_responses

| arm | rps | p50 | p99 |
|-----|-----|-----|-----|
| proxima_native | 8255 | 117.25 µs | 191.23 µs |
| hyper | 4286 | 223.49 µs | 301.57 µs |
| pingora | 3462 | 281.60 µs | 367.10 µs |

---

## Headlines

- **spawn_burst_1k/prime_typed:** 69.762 µs (vs tokio 220.95 µs — **3.17× faster**)
- **h2_load_5way/proxima_on_prime:** 33.710 µs (vs hyper 43.053 µs — 22% faster; vs proxima_on_flume 30.822 µs — near-parity)
- **h2_streaming_responses:** proxima_native 8255 rps vs hyper 4286 rps — **1.93× faster**
- **bench_background_pool/proxima_rayon_backed:** 393.26 µs vs rayon raw 450.25 µs — 13% better (x86 rayon variant wins as expected)
- **affinity panics:** 0

## Failures

| bench | reason |
|-------|--------|
| h2_vs_pingora | duplicate criterion ID in bench harness (bench code bug, not infra) |
