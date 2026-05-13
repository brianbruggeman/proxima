# bench-proxima-runtimes — m1

This document captures runtime+channel variance for proxima on h2 warm GET
and spawn-burst (dispatch-cost isolator). Three arms are compared:
default tokio multi-thread (tokio internal channel), per-core tokio (flume),
and prime native (prime-inbox-alloc). Run the script again after real
measurements to replace `pending` cells.

## h2 warm GET (h2_runtime_swap_proxima_native / w5_single_stream_get)

| runtime | dispatch channel | M1 (mean) | M1 (CoV) | Linux (mean) | Linux (CoV) | unit |
|---|---|---|---|---|---|---|
| default tokio multi-thread | tokio internal | pending | pending | pending | pending | per-request latency |
| per-core tokio | flume | pending | pending | pending | pending | per-request latency |
| prime native | prime-inbox-alloc | pending | pending | pending | pending | per-request latency |

## spawn-burst (spawn_burst_1k — dispatch-cost isolator)

| runtime | dispatch channel | M1 (mean) | M1 (CoV) | Linux (mean) | Linux (CoV) | unit |
|---|---|---|---|---|---|---|
| default tokio multi-thread | tokio internal | pending | pending | pending | pending | ns/task |
| per-core tokio | flume | pending | pending | pending | pending | ns/task |
| prime native | prime-inbox-alloc | pending | pending | pending | pending | ns/task |

## general runtime swap (bench_runtime_swap)

| runtime | mean | CoV | unit |
|---|---|---|---|
| default tokio multi-thread | pending | pending | ns |
| per-core tokio | pending | pending | ns |
| prime native | pending | pending | ns |

## h3 runtime swap (h3_runtime_swap)

| runtime | mean | CoV | unit |
|---|---|---|---|
| default tokio multi-thread | pending | pending | ns |
| per-core tokio | pending | pending | ns |
| prime native | pending | pending | ns |
