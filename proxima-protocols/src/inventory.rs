//! The protocol surface as data.
//!
//! One row per module in `lib.rs`, so a reader or an agent can enumerate the
//! breadth without grepping `pub mod` lines. `docs/protocols.md` and
//! `ai_docs/protocols.jsonl` are rendered from [`INVENTORY`] by
//! `examples/protocol_inventory.rs`; the test below fails when a module is
//! added without a row, which is what keeps the table true.

/// One row per module. `kind` is `"protocol"` for a wire protocol and
/// `"support"` for shared machinery. `bench` names the criterion file stems,
/// empty where none exists. `last_result` is empty until a number is recorded.
pub struct ProtocolEntry {
    pub module: &'static str,
    pub protocol: &'static str,
    pub kind: &'static str,
    pub feature: &'static str,
    pub transport: &'static str,
    pub bench: &'static str,
    pub status: &'static str,
    pub last_result: &'static str,
}

pub const INVENTORY: &[ProtocolEntry] = &[
    ProtocolEntry {
        module: "http1_codec",
        protocol: "HTTP/1",
        kind: "protocol",
        feature: "http1_codec",
        transport: "proxima-http",
        bench: "bench_http1_codec_trait",
        status: "codec",
        last_result: "2026-07",
    },
    ProtocolEntry {
        module: "http2_codec",
        protocol: "HTTP/2",
        kind: "protocol",
        feature: "http2_codec",
        transport: "proxima-http",
        bench: "bench_http2_codec_trait",
        status: "codec",
        last_result: "2026-07",
    },
    ProtocolEntry {
        module: "hpack",
        protocol: "HPACK",
        kind: "protocol",
        feature: "hpack",
        transport: "proxima-http",
        bench: "hpack_block, hpack_decode_into, hpack_huffman, hpack_integer, hpack_static_table, bench_hpack_codec_trait",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "http3_codec",
        protocol: "HTTP/3 + QPACK",
        kind: "protocol",
        feature: "http3_codec",
        transport: "proxima-http",
        bench: "bench_http3_codec_trait",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "quic",
        protocol: "QUIC",
        kind: "protocol",
        feature: "quic",
        transport: "proxima-quic",
        bench: "quic_bench_c1..c15, quic_bench_codec_trait",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "websocket_frame",
        protocol: "WebSocket frame",
        kind: "protocol",
        feature: "websocket_frame",
        transport: "proxima-http",
        bench: "bench_websocket_frame, bench_websocket_frame_codec_trait",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "websocket_handshake",
        protocol: "WebSocket handshake",
        kind: "protocol",
        feature: "websocket_handshake",
        transport: "proxima-http",
        bench: "",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "grpc_framing",
        protocol: "gRPC framing",
        kind: "protocol",
        feature: "grpc_framing",
        transport: "proxima-http",
        bench: "bench_grpc_framing, bench_grpc_framing_codec_trait",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "protobuf_wire",
        protocol: "protobuf wire",
        kind: "protocol",
        feature: "protobuf_wire",
        transport: "",
        bench: "bench_protobuf_wire, bench_protobuf_wire_codec_trait",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "jsonrpc",
        protocol: "JSON-RPC",
        kind: "protocol",
        feature: "jsonrpc",
        transport: "proxima-http",
        bench: "",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "json_framing",
        protocol: "JSON framing",
        kind: "protocol",
        feature: "json_framing",
        transport: "proxima-http",
        bench: "",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "redis",
        protocol: "Redis",
        kind: "protocol",
        feature: "redis",
        transport: "proxima-redis",
        bench: "bench_redis",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "memcached",
        protocol: "Memcached",
        kind: "protocol",
        feature: "memcached",
        transport: "proxima-memcached",
        bench: "bench_memcached, bench_memcached_frame_ownership",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "kafka",
        protocol: "Kafka",
        kind: "protocol",
        feature: "kafka",
        transport: "proxima-kafka",
        bench: "bench_kafka",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "mqtt",
        protocol: "MQTT",
        kind: "protocol",
        feature: "mqtt",
        transport: "proxima-mqtt",
        bench: "bench_mqtt",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "amqp",
        protocol: "AMQP",
        kind: "protocol",
        feature: "amqp",
        transport: "proxima-amqp",
        bench: "bench_amqp",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "pgwire_codec",
        protocol: "PgWire",
        kind: "protocol",
        feature: "pgwire_codec",
        transport: "proxima-pgwire",
        bench: "bench_pgwire_codec",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "dns",
        protocol: "DNS",
        kind: "protocol",
        feature: "dns",
        transport: "proxima-dns",
        bench: "bench_dns",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "inet",
        protocol: "Ethernet/IPv4/TCP",
        kind: "protocol",
        feature: "inet",
        transport: "proxima-net",
        bench: "",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "tcp",
        protocol: "TCP FSM",
        kind: "protocol",
        feature: "tcp",
        transport: "proxima-net",
        bench: "",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "nvme",
        protocol: "NVMe",
        kind: "protocol",
        feature: "nvme",
        transport: "",
        bench: "bench_nvme_codec",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "virtio",
        protocol: "virtio",
        kind: "protocol",
        feature: "virtio",
        transport: "tools/proxima-vm",
        bench: "",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "process",
        protocol: "process protocol",
        kind: "protocol",
        feature: "process",
        transport: "proxima-process",
        bench: "",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "proxy_protocol",
        protocol: "PROXY protocol",
        kind: "protocol",
        feature: "proxy_protocol",
        transport: "proxima-listen",
        bench: "",
        status: "codec",
        last_result: "",
    },
    ProtocolEntry {
        module: "codec_pipe",
        protocol: "codec to Pipe adapter",
        kind: "support",
        feature: "codec-pipe",
        transport: "",
        bench: "",
        status: "support",
        last_result: "",
    },
    ProtocolEntry {
        module: "time",
        protocol: "QUIC/TCP time",
        kind: "support",
        feature: "quic, tcp",
        transport: "",
        bench: "",
        status: "support",
        last_result: "",
    },
];

#[cfg(test)]
mod tests {
    use super::INVENTORY;

    #[test]
    fn every_pub_module_is_in_the_inventory() {
        let lib = include_str!("lib.rs");
        for line in lib.lines() {
            let Some(rest) = line.trim().strip_prefix("pub mod ") else {
                continue;
            };
            let Some(module) = rest.strip_suffix(';') else {
                continue;
            };
            // the inventory describes the modules; it cannot list itself
            if module == "inventory" {
                continue;
            }
            assert!(
                INVENTORY.iter().any(|entry| entry.module == module),
                "pub mod {module} has no inventory row; add it to INVENTORY",
            );
        }
    }
}
