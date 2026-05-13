#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! C11 of the codec-trait initiative — e2e composition bench.
//!
//! Chains the per-sub-crate codecs into one realistic end-to-end
//! shape: an HTTP/1.1 request head followed by N WebSocket frames
//! (the canonical upgrade-and-stream pattern streaming clients use).
//!
//! Two arms per workload:
//!
//! - `concrete_chain` — calls into `proxima_protocols::http1_codec::h1::parse_head`
//!   then `proxima_protocols::websocket_frame::parse_frame` directly (the
//!   workspace's established hot path).
//! - `trait_chain` — calls into `H1RequestCodec::parse_frame` +
//!   `WebSocketFrameCodec::parse_frame` via `FrameCodec`.
//!
//! Gate condition for the C11 row in
//! `docs/codec-trait/discipline.md`: the trait-routed chain meets-or-
//! beats the concrete-struct chain on each workload. When green, the
//! per-sub-crate `codec-trait` Cargo features flip default-off →
//! default-on.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use proxima_codec::FrameCodec;
use proxima_protocols::http1_codec::{H1RequestCodec, h1::parse_head as h1_parse_concrete};
use proxima_protocols::websocket_frame::{
    Opcode, WebSocketFrameCodec, encode_header, parse_frame as ws_parse_concrete,
};

fn build_upgrade_request() -> Vec<u8> {
    let mut buf = String::new();
    buf.push_str("GET /v1/messages HTTP/1.1\r\n");
    buf.push_str("Host: api.example.com\r\n");
    buf.push_str("Upgrade: websocket\r\n");
    buf.push_str("Connection: Upgrade\r\n");
    buf.push_str("Sec-WebSocket-Version: 13\r\n");
    buf.push_str("Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n");
    buf.push_str("\r\n");
    buf.into_bytes()
}

fn build_ws_frame_stream(frame_count: usize, payload_size: usize) -> Vec<u8> {
    let mut buf = Vec::new();
    let payload = vec![0xa5u8; payload_size];
    for _ in 0..frame_count {
        encode_header(true, Opcode::Text, payload.len(), None, &mut buf);
        buf.extend_from_slice(&payload);
    }
    buf
}

fn bench_chain(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("codec_e2e_chain");
    group.measurement_time(Duration::from_secs(5));

    // Workload: 1 HTTP head + N WS frames at the typical realtime size.
    let head = build_upgrade_request();
    for &frame_count in &[10usize, 100] {
        let body = build_ws_frame_stream(frame_count, 256);
        let total = head.len() + body.len();
        group.throughput(Throughput::Bytes(total as u64));

        group.bench_with_input(
            BenchmarkId::new("concrete", frame_count),
            &(head.clone(), body.clone()),
            |bencher, (head, body)| {
                bencher.iter(|| {
                    let head_status = h1_parse_concrete(std::hint::black_box(head)).unwrap();
                    std::hint::black_box(head_status);
                    let mut cursor = 0;
                    while cursor < body.len() {
                        let (frame, consumed) =
                            ws_parse_concrete(std::hint::black_box(&body[cursor..])).unwrap();
                        std::hint::black_box(frame.payload.len());
                        cursor += consumed;
                    }
                    std::hint::black_box(cursor);
                });
            },
        );

        let h1_codec = H1RequestCodec;
        let ws_codec = WebSocketFrameCodec;
        group.bench_with_input(
            BenchmarkId::new("trait", frame_count),
            &(head.clone(), body.clone()),
            |bencher, (head, body)| {
                bencher.iter(|| {
                    let (head_frame, _) = h1_codec.parse_frame(std::hint::black_box(head)).unwrap();
                    std::hint::black_box(head_frame.headers.len());
                    let mut cursor = 0;
                    while cursor < body.len() {
                        let (frame, consumed) = ws_codec
                            .parse_frame(std::hint::black_box(&body[cursor..]))
                            .unwrap();
                        std::hint::black_box(frame.payload.len());
                        cursor += consumed;
                    }
                    std::hint::black_box(cursor);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_chain);
criterion_main!(benches);
