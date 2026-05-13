# Host A M1 Ecosystem Bench Matrix — 2026-05-18

**Complement to**: `host-a-full-matrix-2026-05-18/` (runtime + h2 benches)
**HEAD**: `18a0bc09292653517305f045652613457e254d47`
**Date**: 2026-05-18
**Machine**: MacBook Pro M1 Max, 10 cores (8P+2E), 64 GB
**Total elapsed**: ~84 minutes (06:38 CDT → 08:03 CDT)

---

## Status Table

| Bench | Status | Notes |
|---|---|---|
| h1_vs_hyper | ok | proxima in-process 572 ns vs hyper duplex 16.2 µs |
| h1_vs_pingora | ok | proxima loopback 68 µs vs hyper 71 µs vs pingora 72 µs |
| h1_dispatch | ok | parse_head 149 ns, round-trip 222-251 ns |
| h1_streaming | ok | buffered 211 ns-1.5 µs; streaming 270 ns-9.9 µs |
| h3_runtime_swap | ok | default-tokio 98 µs, per-core 96 µs |
| h3_dispatch | ok | handshake 457 µs, warm GET 82 µs, warm POST 86 µs |
| h3_streaming | ok | 256B 83 µs, 64KiB 668 µs, 16x4KiB 658 µs |
| h3_streaming_responses | ok | 32x2KiB 629 ns (high variance) |
| h3_tail_multi_conn | ok | n1 31 ns, n4 84 ns, n16 837 ns (p50 inside bench) |
| h3_tail_scaling | ok | c1 32 ns, c10 174 ns, c100 4.5 µs (p50 inside bench) |
| bench_h3_upstream | ok | proxima 81.6 µs vs parity baseline 78.0 µs (+5%) |
| bench_grpc_framing | ok | proxima decode 864 ps; encode 30-43 ns; parity near-parity |
| bench_amqp | ok | proxima 2.5-2.0 ns vs amq-protocol 8.0-53 ns (7-25x faster) |
| bench_dns | ok | proxima 26 ns vs parity 14 ns vs hickory 250 ns |
| bench_kafka | ok | proxima ~= parity baseline (sub-1 ns peek, 689 ps frame parse) |
| bench_memcached | ok | proxima 9.3-24 ns vs parity 9.0-27 ns (near-parity; set wins) |
| bench_mqtt | ok | proxima 3.3-5.6 ns vs mqttbytes 142-205 ns (30-40x faster) |
| bench_redis | ok | proxima 5.0-43 ns vs redis-protocol 50-377 ns (7-13x faster) |
| bench_websocket_frame | ok | proxima 1.7-2.1 ns vs tungstenite 13-18 ns (8-9x faster) |
| bench_ws_upstream | ok | proxima 31.7 µs vs async-tungstenite 31.4 µs (parity) |
| bench_protobuf_wire | ok | proxima varint decode 504 ps vs prost 629 ps; walk 19 ns |
| h2_native_frame | ok | header parse 457 ps, encode 2.2 ns |
| hpack_block | ok | encode 211-761 ns; decode 129-1425 ns |
| hpack_huffman | ok | proxima decode 2-4x faster than h2_crate |
| hpack_integer | partial | h2_crate panics on u32max_8b decode (DecoderError); proxima ok |
| hpack_static_table | ok | proxima 1.0-3.7 ns; parity with h2_crate throughout |
| simd_json_decode | ok | simd-json 7.1 µs vs serde_json 8.3 µs (+17% faster) |
| proxy_protocol_parse | ok | v1-tcp4 107 ns, v2-ipv4 4.5 ns, v2-ipv6 6.9 ns |
| request_path | ok | synth 587 ns, cache_hit 3.0 µs |
| substrate_dispatch | ok | noop x1 678 ns, x10 1.54 µs; concurrent x64 100 µs |
| network_throughput | ok | HTTP listener keepalive 66 µs, TCP 88 µs |
| per_core_vs_arcswap | ok | per-core 2.5 ns vs arcswap 10-55 ns under contention |
| bench_reactor | ok | proxima wake 833 ns vs tokio_unix 1.77 µs (2.1x faster) |
| bench_fairness_imbalanced | ok | tokio 15.7 ms vs prime round-robin 15.3 ms (parity) |
| bench_h2_spawn_blocking | ok | tokio_per_core 211 µs vs prime 217 µs (parity) |
| swap_under_load | ok | dispatch 17 ns, swap 109 ns, storm dispatch 880 ns |
| histogram_record | ok | single_thread 461 ns, 8_workers 29.7 µs |

**Total**: 36 ok, 1 partial (hpack_integer — h2 crate bug, not proxima)
**Compile failures**: 0
**Hangs**: 0

---

## Headline Protocol Wins

### WebSocket Frame
proxima sans-IO vs tungstenite:
- small text: 1.7 ns vs 15.3 ns (9x faster)
- medium masked 192B: 2.1 ns vs 12.8 ns (6x)
- large 64KiB: 2.0 ns vs 18 ns (9x)

### MQTT
proxima vs mqttbytes:
- PUBLISH QoS0: 4.7 ns vs 142 ns (30x faster)
- PUBLISH QoS1: 5.0 ns vs 142 ns (28x)
- CONNECT: 5.6 ns vs 205 ns (37x)

### Redis RESP
proxima vs redis-protocol:
- simple_string: 5.0 ns vs 65.6 ns (13x faster)
- integer: 6.5 ns vs 49.8 ns (8x)
- array_of_blobs: 42.9 ns vs 376 ns (9x)
- blob_1kb: 9.5 ns vs 101 ns (11x)

### AMQP
proxima vs amq-protocol:
- method frame: 2.5 ns vs 18.5 ns (7x faster)
- body frame: 2.1 ns vs 53 ns (25x faster)
- heartbeat: 2.0 ns vs 8.0 ns (4x)

### DNS
proxima vs hickory-proto:
- response parse: 26 ns vs 250 ns (10x faster)
- workload count-A: 84 ns vs 558 ns (7x)

### HTTP/1.1 vs hyper
In-process (duplex, no TCP):
- proxima Connection: 572 ns
- hyper duplex: 16.2 µs (proxima 28x faster in-process)

Over loopback TCP (fair, h1_vs_pingora):
- proxima: 68 µs vs hyper: 71 µs vs pingora: 72 µs (+5% vs both)

### Reactor Wake Latency
- proxima Reactor: 833 ns
- tokio unix pipe: 1.77 µs (proxima 2.1x faster)

### Protobuf Varint
- proxima: 504 ps vs prost: 629 ps (+25%)

---

## Per-Bench Median Tables

### bench_amqp (proxima vs amq-protocol)

| frame | proxima | amq-protocol | ratio |
|---|---|---|---|
| method | 2.5 ns | 18.5 ns | 7x |
| body | 2.1 ns | 53 ns | 25x |
| heartbeat | 2.0 ns | 8.0 ns | 4x |

### bench_dns

| arm | median |
|---|---|
| proxima | 26.3 ns |
| parity baseline | 14.2 ns |
| hickory-proto | 250 ns |

### bench_kafka

| op | proxima | parity |
|---|---|---|
| peek_size | 504 ps | 503 ps |
| frame_parse | 689 ps | 690 ps |
| header_parse | 1.72 ns | 1.71 ns |

### bench_memcached

| cmd | proxima | parity |
|---|---|---|
| get | 9.29 ns | 8.95 ns |
| set | 24.5 ns | 26.5 ns |
| delete | 13.3 ns | 14.9 ns |

### bench_mqtt

| packet | proxima | mqttbytes |
|---|---|---|
| PUBLISH QoS0 | 4.72 ns | 142 ns |
| PUBLISH QoS1 | 5.02 ns | 142 ns |
| CONNECT | 5.62 ns | 205 ns |

### bench_redis

| type | proxima | parity | redis-protocol |
|---|---|---|---|
| simple_string | 5.0 ns | 4.0 ns | 65.6 ns |
| error | 12.9 ns | 11.9 ns | 82.1 ns |
| integer | 6.5 ns | 7.6 ns | 49.8 ns |
| array_of_blobs | 42.9 ns | 45.4 ns | 376 ns |
| blob_1kb | 9.5 ns | 11.9 ns | 101 ns |

### bench_websocket_frame

| frame | proxima | tungstenite | ratio |
|---|---|---|---|
| small text 7B | 1.66 ns | 15.3 ns | 9x |
| medium masked 192B | 2.11 ns | 12.8 ns | 6x |
| large 64KiB | 1.99 ns | 18.0 ns | 9x |

### bench_grpc_framing

| op | proxima | parity |
|---|---|---|
| decode 16B | 864 ps | 864 ps |
| decode 64KiB | 864 ps | 864 ps |
| encode 16B | 30.6 ns | 30.0 ns |
| encode 64KiB | 1.21 µs | 1.22 µs |

### bench_protobuf_wire

| op | proxima | parity | prost |
|---|---|---|---|
| varint decode 1-byte | 504 ps | 944 ps | 629 ps |
| walk_message | 19.2 ns | 27.3 ns | N/A |

### h2_native_frame

| op | median |
|---|---|
| header parse | 457 ps |
| header encode | 2.16 ns |
| data parse 64B | 15.9 ns |
| data parse 4KiB | 14.0 ns |
| data encode 64B | 5.78 ns |
| settings 6-entry | 12.7 ns |

### hpack_huffman (selected)

| value | proxima enc | h2 enc | proxima dec | h2 dec |
|---|---|---|---|---|
| www_example_com | 18.9 ns | 17.2 ns | 23.3 ns | 63.7 ns |
| no_cache | 8.52 ns | 10.0 ns | 10.5 ns | 35.0 ns |
| cookie_512b | 537 ns | 577 ns | 639 ns | 1.75 µs |
| body_chunk_4kib | 4.64 µs | 4.44 µs | 4.50 µs | 16.8 µs |

### proxy_protocol_parse

| format | median |
|---|---|
| v1 TCP4 | 107 ns |
| v1 TCP6 | 155.7 ns |
| v2 IPv4 | 4.53 ns |
| v2 IPv6 | 6.95 ns |
| reject (non-proxy) | 2.61 ns |

### bench_reactor

| arm | median |
|---|---|
| proxima Reactor wake | 833 ns |
| tokio unix pipe wake | 1.77 µs |
| proxima turn N=16 | 8.33 µs |

### per_core_vs_arcswap (0 writers)

| impl | median |
|---|---|
| per-core thread-local | 2.5 ns |
| arcswap | 10.3 ns |
| dashmap | 15.0 ns |

Under 16 writers: per-core stays ~44 ns; arcswap ~116 ns; dashmap ~112 ns.

---

## Notes

- **hpack_integer** (partial): proxima fully passes all cases. The h2 crate panics with
  "DecoderError" on `u32max_8b` decode — this is a known edge case in the h2 crate's
  integer decoder, not in proxima.
- **bench_reactor**: requires `runtime-prime-full` (not just `runtime-prime-reactor`) due
  to `core_shard` dependency in the OS net module.
- **bench_h3_upstream**: proxima +5% vs parity (thin wrapper overhead over h3-quinn). Expected.
- **DNS parity gap**: proxima parse 26 ns vs parity 14 ns — parity is minimal scope (less
  record decoding). Both are 10x ahead of hickory's 250 ns.
