#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! C10 of the codec-trait initiative — proxima-websocket-frame.
//!
//! Compares three arms on a frame size sweep:
//!
//! - `concrete_decode` / `concrete_encode` — the existing
//!   `proxima_protocols::websocket_frame::{parse_frame, encode_header}` (the
//!   workspace's established hot path).
//! - `trait_decode` / `trait_encode` — the new
//!   `WebSocketFrameCodec::{parse_frame, encode_frame}` `FrameCodec`-
//!   routed path. C11 e2e gate: trait arm must be within noise floor
//!   of concrete arm.
//! - `tungstenite_*` — DEFERRED, recorded in
//!   `docs/codec-trait/baselines.md`. tungstenite's `Frame` couples
//!   header parsing with payload allocation in a different shape than
//!   the sans-IO borrow-only API here; the apples-to-apples comparison
//!   needs a careful harness that strips the buffered-write differences.
//!
//! Frame sizes: 16 B (small text), 1 KiB (typical realtime message),
//! 64 KiB (bulk transfer).

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use proxima_codec::FrameCodec;
use proxima_protocols::websocket_frame::{
    Frame, Opcode, WebSocketFrameCodec, encode_header, parse_frame as concrete_parse,
};

const SIZES: &[usize] = &[16, 1024, 64 * 1024];

fn build_unmasked_text_frame(payload_size: usize) -> Vec<u8> {
    let payload = vec![0xa5u8; payload_size];
    let mut buf = Vec::new();
    encode_header(true, Opcode::Text, payload.len(), None, &mut buf);
    buf.extend_from_slice(&payload);
    buf
}

fn bench_decode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("ws_codec_trait_decode");
    group.measurement_time(Duration::from_secs(5));
    for &size in SIZES {
        let bytes = build_unmasked_text_frame(size);
        group.throughput(Throughput::Bytes(bytes.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("concrete", size),
            &bytes,
            |bencher, buf| {
                bencher.iter(|| {
                    let (frame, consumed) = concrete_parse(std::hint::black_box(buf)).unwrap();
                    std::hint::black_box((frame.payload.len(), consumed));
                });
            },
        );

        let codec = WebSocketFrameCodec;
        group.bench_with_input(BenchmarkId::new("trait", size), &bytes, |bencher, buf| {
            bencher.iter(|| {
                let (frame, consumed) = codec.parse_frame(std::hint::black_box(buf)).unwrap();
                std::hint::black_box((frame.payload.len(), consumed));
            });
        });
    }
    group.finish();
}

fn bench_encode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("ws_codec_trait_encode");
    group.measurement_time(Duration::from_secs(5));
    for &size in SIZES {
        let payload = vec![0xa5u8; size];
        // header is at most 10 bytes for unmasked frames (2 + 8 for 64-bit length).
        group.throughput(Throughput::Bytes((size + 10) as u64));

        group.bench_with_input(
            BenchmarkId::new("concrete", size),
            &payload,
            |bencher, payload| {
                let mut dest = Vec::with_capacity(payload.len() + 16);
                bencher.iter(|| {
                    dest.clear();
                    encode_header(true, Opcode::Text, payload.len(), None, &mut dest);
                    dest.extend_from_slice(std::hint::black_box(payload));
                    std::hint::black_box(dest.len());
                });
            },
        );

        let codec = WebSocketFrameCodec;
        group.bench_with_input(
            BenchmarkId::new("trait", size),
            &payload,
            |bencher, payload| {
                let mut dest = Vec::with_capacity(payload.len() + 16);
                let frame = Frame {
                    fin: true,
                    opcode: Opcode::Text,
                    compressed: false,
                    mask: None,
                    payload,
                };
                bencher.iter(|| {
                    dest.clear();
                    codec
                        .encode_frame(std::hint::black_box(&frame), &mut dest)
                        .unwrap();
                    std::hint::black_box(dest.len());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_decode, bench_encode);
criterion_main!(benches);
