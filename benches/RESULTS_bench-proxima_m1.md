# proxima bench results — m1

Absolute proxima performance across every workload category using the canonical
ship feature set. No comparison axis — see bench-vs-{hyper,pingora,rayon} for
competitive numbers. Linux column is populated by CI; M1 column is populated by
local developer runs. Run `scripts/bench-proxima.sh` to refresh.

Features: `runtime-prime-full,runtime-tokio,http,tls,websocket,websocket-frame,websocket-upstream,redis-listener,memcached-listener,mqtt-listener,amqp-listener,kafka-listener,grpc-framing,protobuf-wire,dns-substrate,h3-upstream,runtime-prime-bgpool-rayon,runtime-prime-bgpool-par,runtime-prime-bgpool-async,rayon,prime-tokio-compat`

---

## HTTP wire

| workload | metric | M1 | Linux | unit | bench file |
|----------|--------|-----|-------|------|------------|
| h1_dispatch | mean | pending | pending | ns | h1_dispatch |
| h2_dispatch | mean | pending | pending | ns | h2_dispatch |
| h3_dispatch | mean | pending | pending | ns | h3_dispatch |
| h2_native_vs_h2_crate | mean | pending | pending | ns | h2_native_vs_h2_crate |
| h2_native_vs_h2_crate_e2e | mean | pending | pending | ns | h2_native_vs_h2_crate_e2e |
| h2_native_vs_h2_crate_alloc | mean | pending | pending | ns | h2_native_vs_h2_crate_alloc |

### HTTP streaming (p50 / p99 / p999)

| workload | p50 (M1) | p99 (M1) | p999 (M1) | p50 linux | p99 linux | p999 linux | unit | bench file |
|----------|----------|----------|-----------|-----------|-----------|------------|------|------------|
| h1_streaming | pending | pending | pending | pending | pending | pending | ns | h1_streaming |
| h2_streaming | pending | pending | pending | pending | pending | pending | ns | h2_streaming |
| h2_streaming_responses | pending | pending | pending | pending | pending | pending | ns | h2_streaming_responses |
| h2_tail_scaling | pending | pending | pending | pending | pending | pending | ns | h2_tail_scaling |
| h3_streaming | pending | pending | pending | pending | pending | pending | ns | h3_streaming |
| h2_native_vs_h2_crate_tail | pending | pending | pending | pending | pending | pending | ns | h2_native_vs_h2_crate_tail |
| h3_streaming_responses | pending | pending | pending | pending | pending | pending | ns | h3_streaming_responses |
| h3_tail_scaling | pending | pending | pending | pending | pending | pending | ns | h3_tail_scaling |
| h3_tail_multi_conn | pending | pending | pending | pending | pending | pending | ns | h3_tail_multi_conn |

---

## sans-IO parsers

| workload | metric | M1 | Linux | unit | bench file |
|----------|--------|-----|-------|------|------------|
| hpack_block | mean | pending | pending | ns | hpack_block |
| hpack_huffman | mean | pending | pending | ns | hpack_huffman |
| hpack_integer | mean | pending | pending | ns | hpack_integer |
| hpack_static_table | mean | pending | pending | ns | hpack_static_table |
| h2_native_frame | mean | pending | pending | ns | h2_native_frame |
| bench_websocket_frame | mean | pending | pending | ns | bench_websocket_frame |
| bench_dns | mean | pending | pending | ns | bench_dns |
| bench_protobuf_wire | mean | pending | pending | ns | bench_protobuf_wire |
| proxy_protocol_parse | mean | pending | pending | ns | proxy_protocol_parse |
| simd_json_decode | mean | pending | pending | ns | simd_json_decode |

---

## state protocols

| workload | metric | M1 | Linux | unit | bench file |
|----------|--------|-----|-------|------|------------|
| bench_redis | mean | pending | pending | ns | bench_redis |
| bench_memcached | mean | pending | pending | ns | bench_memcached |
| bench_mqtt | mean | pending | pending | ns | bench_mqtt |
| bench_amqp | mean | pending | pending | ns | bench_amqp |
| bench_kafka | mean | pending | pending | ns | bench_kafka |
| bench_grpc_framing | mean | pending | pending | ns | bench_grpc_framing |
| bench_ws_upstream | mean | pending | pending | ns | bench_ws_upstream |
| bench_h3_upstream | mean | pending | pending | ns | bench_h3_upstream |

---

## scheduling

| workload | metric | M1 | Linux | unit | bench file |
|----------|--------|-----|-------|------|------------|
| bench_spawn_burst | mean | pending | pending | ns | bench_spawn_burst |
| bench_open_loop_driver | mean | pending | pending | ns | bench_open_loop_driver |
| bench_fairness_imbalanced | mean | pending | pending | ns | bench_fairness_imbalanced |
| bench_timer | mean | pending | pending | ns | bench_timer |
| bench_reactor | mean | pending | pending | ns | bench_reactor |
| bench_local_executor | mean | pending | pending | ns | bench_local_executor |
| bench_h2_spawn_blocking | mean | pending | pending | ns | bench_h2_spawn_blocking |

---

## channels

| workload | metric | M1 | Linux | unit | bench file |
|----------|--------|-----|-------|------|------------|
| bench_inbox/proxima | mean | pending | pending | ns | bench_inbox |

---

## bgpool

| workload | metric | M1 | Linux | unit | bench file |
|----------|--------|-----|-------|------|------------|
| bench_background_pool/proxima | mean | pending | pending | ns | bench_background_pool |

---

## pipeline

| workload | metric | m1 | linux | unit | bench file |
|----------|--------|----|----|------|------------|
| tee_backpressure | mean latency | pending | pending | ns | tee_backpressure |
| tee_sink_primitives | mean latency | pending | pending | ns | tee_sink_primitives |
| substrate_dispatch | mean latency | pending | pending | ns | substrate_dispatch |
| hot_apply_build | mean latency | pending | pending | ns | hot_apply_build |
| swap_under_load | mean latency | pending | pending | ns | swap_under_load |

---

## end-to-end

| workload | metric | M1 | Linux | unit | bench file |
|----------|--------|-----|-------|------|------------|
| request_path | mean | pending | pending | ns | request_path |
| network_throughput | mean | pending | pending | ns | network_throughput |
| per_core_vs_arcswap | mean | pending | pending | ns | per_core_vs_arcswap |
| perf_audit | mean | pending | pending | ns | perf_audit |

---

_Run `scripts/bench-proxima.sh` to replace `pending` values with measured results._
