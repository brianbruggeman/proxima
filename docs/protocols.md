# The protocol surface

Every wire protocol is a sans-IO codec in `proxima-protocols`, one
feature-gated module each, folded from nineteen single-purpose parser crates.
The codec is buffer-in / buffer-out; the client or listener that owns the
transport lives in its own crate. Every protocol with a `bench` entry has a
criterion harness even where no number is published yet.

Rendered from `proxima-protocols/src/inventory.rs`. Regenerate:

```sh
cargo run -p proxima-protocols --example protocol_inventory -- markdown > docs/protocols.md
cargo run -p proxima-protocols --example protocol_inventory -- jsonl > ai_docs/protocols.jsonl
```

`inventory::tests::every_pub_module_is_in_the_inventory` fails when a module is
added without a row.

| protocol | module | feature | transport | bench | status | last result |
|---|---|---|---|---|---|---|
| HTTP/1 | `http1_codec` | `http1_codec` | proxima-http | bench_http1_codec_trait | codec | 2026-07 |
| HTTP/2 | `http2_codec` | `http2_codec` | proxima-http | bench_http2_codec_trait | codec | 2026-07 |
| HPACK | `hpack` | `hpack` | proxima-http | hpack_block, hpack_decode_into, hpack_huffman, hpack_integer, hpack_static_table, bench_hpack_codec_trait | codec |  |
| HTTP/3 + QPACK | `http3_codec` | `http3_codec` | proxima-http | bench_http3_codec_trait | codec |  |
| QUIC | `quic` | `quic` | proxima-quic | quic_bench_c1..c15, quic_bench_codec_trait | codec |  |
| WebSocket frame | `websocket_frame` | `websocket_frame` | proxima-http | bench_websocket_frame, bench_websocket_frame_codec_trait | codec |  |
| WebSocket handshake | `websocket_handshake` | `websocket_handshake` | proxima-http |  | codec |  |
| gRPC framing | `grpc_framing` | `grpc_framing` | proxima-http | bench_grpc_framing, bench_grpc_framing_codec_trait | codec |  |
| protobuf wire | `protobuf_wire` | `protobuf_wire` |  | bench_protobuf_wire, bench_protobuf_wire_codec_trait | codec |  |
| JSON-RPC | `jsonrpc` | `jsonrpc` | proxima-http |  | codec |  |
| JSON framing | `json_framing` | `json_framing` | proxima-http |  | codec |  |
| Redis | `redis` | `redis` | proxima-redis | bench_redis | codec |  |
| Memcached | `memcached` | `memcached` | proxima-memcached | bench_memcached, bench_memcached_frame_ownership | codec |  |
| Kafka | `kafka` | `kafka` | proxima-kafka | bench_kafka | codec |  |
| MQTT | `mqtt` | `mqtt` | proxima-mqtt | bench_mqtt | codec |  |
| AMQP | `amqp` | `amqp` | proxima-amqp | bench_amqp | codec |  |
| PgWire | `pgwire_codec` | `pgwire_codec` | proxima-pgwire | bench_pgwire_codec | codec |  |
| DNS | `dns` | `dns` | proxima-dns | bench_dns | codec |  |
| Ethernet/IPv4/TCP | `inet` | `inet` | proxima-net |  | codec |  |
| TCP FSM | `tcp` | `tcp` | proxima-net |  | codec |  |
| NVMe | `nvme` | `nvme` |  | bench_nvme_codec | codec |  |
| virtio | `virtio` | `virtio` | tools/proxima-vm |  | codec |  |
| process protocol | `process` | `process` | proxima-process |  | codec |  |
| PROXY protocol | `proxy_protocol` | `proxy_protocol` | proxima-listen |  | codec |  |
| codec to Pipe adapter | `codec_pipe` | `codec-pipe` |  |  | support |  |
| QUIC/TCP time | `time` | `quic, tcp` |  |  | support |  |
