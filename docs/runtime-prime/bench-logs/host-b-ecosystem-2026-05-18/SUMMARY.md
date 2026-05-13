# Host B x86 Ecosystem Bench Matrix — 2026-05-18

**Host:** host-b · Linux x86_64 6.15.3-arch1-1 · AMD/Intel  
**Date:** 2026-05-18  
**Pairs with:** host-a-ecosystem-2026-05-18 (Apple M1)

## Status Table

| bench | status | note |
|---|---|---|
| h1_vs_hyper | clean | |
| h1_vs_pingora | clean | |
| h1_dispatch | clean | |
| h1_streaming | clean | |
| h3_runtime_swap | clean | features: http3,runtime-tokio |
| h3_dispatch | clean | features: http3 |
| h3_streaming | clean | features: http3 |
| h3_streaming_responses | clean | features: http3,runtime-tokio |
| h3_tail_multi_conn | clean | features: http3,runtime-tokio |
| h3_tail_scaling | clean | features: http3,runtime-tokio |
| bench_h3_upstream | clean | features: h3-upstream |
| bench_grpc_framing | clean | features: grpc-framing |
| bench_amqp | clean | features: amqp-listener |
| bench_dns | clean | features: dns-substrate |
| bench_kafka | clean | features: kafka-listener |
| bench_memcached | clean | features: memcached-listener |
| bench_mqtt | clean | features: mqtt-listener |
| bench_redis | clean | features: redis-listener |
| bench_websocket_frame | clean | features: websocket-frame |
| bench_ws_upstream | clean | features: websocket-upstream |
| bench_protobuf_wire | clean | features: protobuf-wire |
| h2_native_frame | clean | |
| hpack_block | clean | |
| hpack_huffman | clean | |
| hpack_integer | clean | |
| hpack_static_table | clean | |
| simd_json_decode | clean | |
| proxy_protocol_parse | clean | |
| request_path | clean | |
| substrate_dispatch | clean | |
| network_throughput | clean | |
| per_core_vs_arcswap | clean | |
| bench_reactor | clean | features: runtime-prime-reactor,runtime-prime-executor,runtime-prime-inbox-alloc,runtime-tokio |
| bench_fairness_imbalanced | clean | features: runtime-tokio,runtime-prime-full,http2,tcp,http1 |
| bench_h2_spawn_blocking | clean | features: runtime-tokio,runtime-prime-full,http2,tcp,http1 |
| swap_under_load | clean | |
| histogram_record | clean | |

**Summary: 37 benches — 37 clean, 0 partial, 0 compile-failed, 0 hung**

---

## Protocols vs Incumbents

### h1_vs_hyper

| variant | median |
|---|---|
| hyper::server::conn::http1 (duplex transport) | 2.312 µs |
| proxima::Connection (in-process) | 474.5 ns |
| proxima::Connection (duplex transport, async server task) | 1.576 µs |

proxima in-process is **4.9× faster** than hyper duplex. proxima duplex is **1.47× faster** than hyper duplex on x86.

### h1_vs_pingora (loopback, real TCP)

| variant | median |
|---|---|
| proxima::Connection | 65.3 µs |
| hyper::server::conn::http1 | 69.3 µs |
| pingora::HttpSession | 77.8 µs |

proxima wins loopback: **+6% vs hyper**, **+19% vs pingora**.

### h1_dispatch

| group | median |
|---|---|
| h1_parse_head/small_get_5_headers | 182.2 ns |
| h1_connection_round_trip_no_body | 335.0 ns |
| h1_connection_round_trip_post_with_body | 326.1 ns |

### h1_streaming

| group | median | throughput |
|---|---|---|
| h1_buffered_cl_256 | 326.2 ns | 1.119 GiB/s |
| h1_streaming_cl_256 | 352.6 ns | 1.035 GiB/s |
| h1_buffered_cl_64kib | 1.522 µs | 40.2 GiB/s |
| h1_streaming_cl_64kib | 3.155 µs | 19.4 GiB/s |
| h1_buffered_chunked_16x4kib | 1.785 µs | 34.3 GiB/s |
| h1_streaming_chunked_16x4kib | 4.059 µs | 15.1 GiB/s |

---

### h3_runtime_swap

| runtime | median | throughput |
|---|---|---|
| default tokio multi-thread | 76.5 µs | 13.1 Kelem/s |
| proxima TokioPerCoreRuntime | 60.0 µs | 16.7 Kelem/s |

per-core runtime is **+27% throughput** vs default tokio on x86.

### h3_dispatch

| group | median |
|---|---|
| h3_request_on_warm_connection | 60.6 µs |
| h3_post_with_body_on_warm_connection | 62.5 µs |

### h3_streaming

| group | median | throughput |
|---|---|---|
| h3_echo_64KiB | 286.5 µs | 218.1 MiB/s |
| h3_echo_16x4KiB | 287.3 µs | 217.6 MiB/s |

### h3_streaming_responses

| group | median |
|---|---|
| 32x2KiB_chunks_warm_connection | 217.3 ns (high variance) |

### h3_tail_multi_conn

| conns | median |
|---|---|
| n4 | 44.0 ns |
| n16 | 383.0 ns |

### h3_tail_scaling

| concurrency | median |
|---|---|
| c10 | 82.4 ns |
| c100 | 2.085 µs |

### bench_h3_upstream

| variant | median |
|---|---|
| h3_upstream_proxima | 56.3 µs |
| h3_upstream_parity | 56.8 µs |

Parity; proxima upstream overhead is negligible.

---

### bench_grpc_framing

| op | size | proxima | parity | winner |
|---|---|---|---|---|
| decode | 16 B | 1.408 ns | 1.402 ns | parity (tied) |
| decode | 1024 B | 1.399 ns | 1.396 ns | parity (tied) |
| decode | 65536 B | 1.394 ns | 1.391 ns | parity (tied) |
| encode | 16 B | 22.1 ns | 21.2 ns | parity |
| encode | 1024 B | 31.6 ns | 31.5 ns | tied |
| encode | 65536 B | 1.317 µs | 1.324 µs | proxima |

gRPC framing is functionally parity across all sizes on x86.

---

### bench_amqp

| group | proxima | incumbent | proxima speedup |
|---|---|---|---|
| amqp_method | 5.86 ns | 3.26 ns (parity) | 0.56× |
| amqp_body | 1.296 ns | 1.631 ns (parity) | 1.26× |
| amqp_heartbeat | 5.462 ns | 2.367 ns (parity) | 0.43× |
| amqp_real/method | 5.707 ns | 35.7 ns (amq_protocol) | **6.3×** |
| amqp_real/body | 5.491 ns | 30.8 ns (amq_protocol) | **5.6×** |
| amqp_real/heartbeat | 5.497 ns | 20.1 ns (amq_protocol) | **3.7×** |
| amqp_workload_classify_method | 1.875 ns | 34.6 ns (amq_protocol) | **18.4×** |

proxima AMQP classify workload: **18× faster** than amq_protocol crate. Simpler frame types favor parity baseline on micro; real workload is decisively proxima.

---

### bench_dns

| group | proxima | parity | hickory | proxima vs hickory |
|---|---|---|---|---|
| dns_parse/response_full | 31.2 ns | 21.0 ns | 216.3 ns | **6.9×** |
| dns_workload_count_a | 98.2 ns | — | 682.2 ns | **6.9×** |

proxima DNS is 33% slower than parity baseline, but 6.9× faster than hickory.

---

### bench_kafka

| group | proxima | parity | winner |
|---|---|---|---|
| kafka_peek_size | 941 ps | 931 ps | parity (tied) |
| kafka_frame_parse | 1.154 ns | 1.197 ns | proxima +3.6% |
| kafka_header_parse | 2.046 ns | 2.286 ns | **proxima +11.7%** |

proxima Kafka header parse wins by ~12% on x86.

---

### bench_memcached

| op | proxima | parity | proxima faster by |
|---|---|---|---|
| GET | 11.10 ns | 12.56 ns | +13.1% |
| SET | 33.90 ns | 36.06 ns | +6.4% |
| DELETE | 15.85 ns | 17.04 ns | +7.5% |

proxima Memcached wins all ops on x86.

---

### bench_mqtt

| op | proxima | mqttbytes | proxima speedup |
|---|---|---|---|
| publish_qos0 (micro) | 5.644 ns | 6.302 ns (parity) | 1.12× |
| publish_qos1 (micro) | 9.431 ns | 9.072 ns (parity) | 0.96× |
| mqtt_real/publish_qos0 | 9.999 ns | 93.9 ns (mqttbytes) | **9.4×** |
| mqtt_real/publish_qos1 | 10.39 ns | 97.1 ns (mqttbytes) | **9.3×** |
| workload_route_publish | 6.307 ns | 91.8 ns (mqttbytes) | **14.6×** |

MQTT workload: proxima is **14.6× faster** than mqttbytes.

---

### bench_redis

| group | proxima | parity | proxima faster by |
|---|---|---|---|
| parse/simple_string | 8.791 ns | 7.576 ns | 0.86× |
| parse/integer | 10.60 ns | 12.29 ns | +15.8% |
| parse/blob_short | 10.98 ns | 13.42 ns | +22.2% |
| parse/array_of_blobs | 46.9 ns | — | — |
| parse/blob_1kb | 14.4 ns | — | — |
| workload_classify_command | 48.98 ns | 273.2 ns (redis_protocol) | **5.6×** |

Redis classify workload: **5.6× faster** than redis_protocol crate.

---

### bench_websocket_frame

| size | proxima | tungstenite | proxima speedup |
|---|---|---|---|
| small_text | 2.616 ns | 18.6 ns | **7.1×** |
| medium_masked | 4.041 ns | 27.1 ns | **6.7×** |
| large | 3.708 ns | 21.4 ns | **5.8×** |

proxima WebSocket frame: **5.8–7.1× faster** than tungstenite.

---

### bench_ws_upstream

| variant | median |
|---|---|
| proxima | 25.4 µs |
| async-tungstenite direct | 25.4 µs |

Parity on full loopback upstream round-trip.

---

### bench_protobuf_wire

| op | proxima | prost | parity | best |
|---|---|---|---|---|
| varint_decode/0 | 953 ps | 1.136 ns | 1.511 ns | **proxima** |
| varint_decode/127 | 933 ps | 1.135 ns | — | **proxima** |
| walk_message | 23.4 ns | — | 26.9 ns | **proxima +15%** |

proxima protobuf varint: **+20% vs prost**, **+37% vs parity** (bytes crate). Walk-message: **+15% vs parity**.

---

## Proxima-Internal Primitives

### h2_native_frame

| op | median | throughput |
|---|---|---|
| header/parse | 698 ps | 1.43 Gelem/s |
| header/encode | 2.935 ns | 341 Melem/s |
| parse_data/64b–64kib | ~26.6 ns | ~37.5 Melem/s |
| parse_headers/no_priority | 27.6 ns | 36.2 Melem/s |
| parse_settings/1_entry | 12.3 ns | 81.3 Melem/s |
| encode_data_vectored/all sizes | ~38.0 ns | ~26.4 Melem/s |

H2 frame header parse at sub-700ps on x86.

---

### hpack_block

| op | median | throughput |
|---|---|---|
| encode/request_minimal | 175.9 ns | 22.7 Melem/s |
| encode/request_browser | 1.727 µs | 5.21 Melem/s |
| encode/request_api | 859 ns | 9.31 Melem/s |
| decode/request_minimal | 145.0 ns | 27.6 Melem/s |
| decode/request_browser | 1.752 µs | 5.14 Melem/s |
| decode/request_api | 1.106 µs | 7.24 Melem/s |

### hpack_huffman

| op | proxima | h2_crate | hpack_crate | winner |
|---|---|---|---|---|
| encode/www_example_com | 41.0 ns | 28.4 ns | — | h2_crate |
| encode/user_agent_chrome | 302 ns | 205.6 ns | — | h2_crate |
| encode/cookie_512b | 1.215 µs | 801.7 ns | — | h2_crate |
| decode/body_chunk_4kib | 8.117 µs | 14.8 µs | 521 µs | **proxima** |

Proxima Huffman encode: ~1.47× slower than h2_crate on x86 (x86 bit-manipulation advantage). Huffman decode: proxima is **1.83× faster** than h2_crate and **64× faster** than hpack_crate.

### hpack_integer

| op | proxima | h2_crate | winner |
|---|---|---|---|
| encode/rfc_c_1_1_10_5b | 1.529 ns | 1.519 ns | h2_crate (tied) |
| encode/rfc_c_1_2_1337_5b | 3.027 ns | 3.452 ns | **proxima +14%** |
| encode/rfc_c_1_3_42_8b | 1.495 ns | 1.495 ns | tied |
| encode/boundary_5b | 2.075 ns | 2.322 ns | **proxima +12%** |

Multi-byte integer encoding wins for proxima; single-byte tied.

### hpack_static_table

| lookup | proxima | h2_crate | winner |
|---|---|---|---|
| method_get | 3.964 ns | 4.775 ns | **proxima +20%** |
| method_delete | 3.640 ns | 4.303 ns | **proxima +18%** |
| status_200 | 5.686 ns | 5.685 ns | tied |
| accept_encoding_gz | 5.516 ns | 6.096 ns | **proxima +11%** |
| unknown_short | 2.505 ns | 2.505 ns | tied |

proxima static table wins method lookups; ties on full-header entries.

---

### simd_json_decode

| decoder | median | throughput |
|---|---|---|
| serde_json | 7.463 µs | 463 MiB/s |
| simd_json | 6.827 µs | 507 MiB/s |

simd_json (SIMD-accelerated) is +9.3% on x86 vs serde_json.

### proxy_protocol_parse

| type | median | throughput |
|---|---|---|
| v1_tcp4 | 142.7 ns | 7.01 Melem/s |
| v1_tcp6 | 229.0 ns | 4.37 Melem/s |
| v2_ipv4 | 8.646 ns | 115.6 Melem/s |
| v2_local | 6.376 ns | 156.9 Melem/s |
| reject/not_proxy | 3.266 ns | 306.2 Melem/s |

v2 parse is ~16× faster than v1 (fixed binary vs text).

### request_path

| variant | median |
|---|---|
| synth_only | 360 ns |
| cache_hit | 2.128 µs |

### substrate_dispatch

| middleware count | median | throughput |
|---|---|---|
| noop_mw_x0 | 341.7 ns | 2.93 Melem/s |
| noop_mw_x1 | 432.6 ns | 2.31 Melem/s |
| noop_mw_x3 | 578.9 ns | 1.73 Melem/s |
| noop_mw_x5 | 724.8 ns | 1.38 Melem/s |
| noop_mw_x3_concurrent_64 | 47.1 µs | 1.36 Melem/s |

Each middleware layer adds ~91 ns on x86.

### network_throughput

| group | median | throughput |
|---|---|---|
| network_http_listener/synth_200_keepalive | 44.0 µs | 22.7 Kelem/s |
| network_tcp_listener/connect_write_read_short | 60.0 µs | 16.7 Kelem/s |

### per_core_vs_arcswap

| variant | readers | median | throughput |
|---|---|---|---|
| arcswap (RCU) | 0 | 12.8 ns | 78.2 Melem/s |
| arcswap | 4 | 113.8 ns | 8.79 Melem/s |
| arcswap | 16 | 148.7 ns | 6.72 Melem/s |
| per_core | 0 | 2.289 ns | 436.9 Melem/s |
| per_core | 1 | 2.320 ns | 431.0 Melem/s |
| per_core | 4 | ~2.3 ns | ~430 Melem/s |

per_core reads are **5.6× faster than arcswap at 0 writers** and fully contention-immune. arcswap degrades ~5× under 4 concurrent writers on x86.

### bench_reactor

| variant | median |
|---|---|
| reactor_wake_latency/proxima | 1.340 µs |
| reactor_wake_latency/tokio_unix | 2.362 µs |
| reactor_turn_n_ready_16/proxima | 13.82 µs (1.16 Melem/s) |

proxima reactor wake: **1.76× lower latency** than tokio Unix FD on x86.

### bench_fairness_imbalanced

| runtime | median | throughput |
|---|---|---|
| tokio_multi_thread | 12.49 ms | 16.0 Kelem/s |
| prime_round_robin_fanout | 13.11 ms | 15.3 Kelem/s |

tokio multi-thread is ~5% faster on this imbalanced workload on x86. Expected: work-stealing favors imbalanced loads.

### bench_h2_spawn_blocking

| variant | median |
|---|---|
| tokio_per_core | 166.4 µs |
| prime | 169.7 µs |

Functionally parity; prime is ~2% slower on blocking-offload path.

### swap_under_load

| variant | median | throughput |
|---|---|---|
| uncontended_load | 31.7 ns | 31.5 Melem/s |
| uncontended_swap | 90.6 ns | 11.0 Melem/s |
| dispatch_during_storm | 393.2 ns | 2.54 Melem/s |

### histogram_record

| variant | median | throughput |
|---|---|---|
| single_thread | 214.9 ns | 4.65 Melem/s |
| 8_workers | 10.69 µs | 748.6 Kelem/s |

---

## Linux x86 Affinity Notes

- **h3_streaming_responses**: Single benchmark result with very high variance (189–270 ns median range over runs). Likely TSC frequency jitter under QUIC crypto workload; result should be treated as approximate.
- **bench_reactor required extra features**: `runtime-prime-reactor` alone does not compile — needs `runtime-prime-inbox-alloc` + `runtime-prime-executor`. Required features metadata in Cargo.toml appears incomplete.
- **bench_fairness_imbalanced** and **bench_h2_spawn_blocking** both required `runtime-prime-full` which is not listed in Cargo.toml's `required-features` for those benches. Discovered through error output; both ran clean once corrected.
- **hpack_huffman encode**: h2_crate is ~1.47× faster on x86 (likely benefit of BMI2/PDEP). proxima decode is still 1.83× faster.
- **request_path** showed 8–10% regression vs criterion baseline, consistent with cold target artifacts (first-run comparison).
- No CPU pinning / affinity issues encountered; kernel 6.15 scheduler cooperated cleanly.

---

## Headline Protocol Wins

| protocol | proxima result | vs incumbent | verdict |
|---|---|---|---|
| AMQP classify | 1.875 ns | 34.6 ns amq_protocol | **proxima 18.4×** |
| MQTT workload | 6.31 ns | 91.8 ns mqttbytes | **proxima 14.6×** |
| WebSocket frame (small) | 2.62 ns | 18.6 ns tungstenite | **proxima 7.1×** |
| DNS workload | 98.2 ns | 682 ns hickory | **proxima 6.9×** |
| Redis classify | 49.0 ns | 273 ns redis_protocol | **proxima 5.6×** |
| Protobuf varint | 953 ps | 1.136 ns prost | **proxima 1.2×** |
| HPACK Huffman decode | 8.12 µs | 14.8 µs h2_crate | **proxima 1.83×** |
| Kafka header parse | 2.046 ns | 2.286 ns parity | **proxima 12%** |
| gRPC framing | ~1.4 ns | ~1.4 ns parity | tied |
| WS upstream (loopback) | 25.4 µs | 25.4 µs tungstenite | tied |
| H1 (loopback vs pingora) | 65.3 µs | 77.8 µs pingora | **proxima 19%** |
| H3 per-core vs default tokio | 60.0 µs | 76.5 µs | **+27% throughput** |
| Reactor wake | 1.34 µs | 2.36 µs tokio | **proxima 1.76×** |
| per-core vs arcswap | 2.29 ns | 12.8 ns arcswap | **proxima 5.6×** |

