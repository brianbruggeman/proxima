#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! C6 of the codec-trait initiative — proxima-quic-proto.
//!
//! Two arms on a frame-type sweep:
//!
//! - `concrete_parse` — existing `frame::parse` (the workspace's
//!   established hot path; already returns `(Frame<'_>, usize)`
//!   matching FrameCodec).
//! - `trait_parse` — new `QuicFrameCodec::parse_frame` (pure
//!   delegation to `frame::parse` plus error type wrapping).
//!
//! `quinn-proto::frame::Frame` is `pub(crate)` upstream — no
//! head-to-head parser bench exists at this layer. C3-frame's
//! existing bench documents the structural home-turf claim. recorded
//! in `docs/codec-trait/baselines.md`.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use proxima_codec::FrameCodec;
use proxima_protocols::quic::{
    QuicFrameCodec,
    frame::{self, Frame},
};

fn build_stream_frame(payload_size: usize) -> Vec<u8> {
    // STREAM frame: tag(varint) + stream_id(varint) + offset(varint) +
    // length(varint) + payload. We use the maximum-flags STREAM frame
    // (LEN bit set) for a realistic mid-size shape.
    let payload = vec![0xa5u8; payload_size];
    let frame = Frame::Stream {
        stream_id: 4,
        offset: 0,
        data: &payload,
        fin: false,
    };
    let mut buf = vec![0u8; payload.len() + 32];
    let written = frame.encode(&mut buf).unwrap();
    buf.truncate(written);
    buf
}

fn bench_parse(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("quic_codec_trait_parse");
    group.measurement_time(Duration::from_secs(5));
    for &size in &[16usize, 1024, 1200] {
        let bytes = build_stream_frame(size);
        group.throughput(Throughput::Bytes(bytes.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("concrete", size),
            &bytes,
            |bencher, buf| {
                bencher.iter(|| {
                    let (frame, consumed) = frame::parse(std::hint::black_box(buf)).unwrap();
                    std::hint::black_box((frame, consumed));
                });
            },
        );

        let codec = QuicFrameCodec;
        group.bench_with_input(BenchmarkId::new("trait", size), &bytes, |bencher, buf| {
            bencher.iter(|| {
                let (frame, consumed) = codec.parse_frame(std::hint::black_box(buf)).unwrap();
                std::hint::black_box((frame, consumed));
            });
        });
    }

    // PING frame: single byte, exercises the per-frame fixed cost.
    let ping = vec![0x01u8];
    group.throughput(Throughput::Bytes(ping.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("concrete", "ping"),
        &ping,
        |bencher, buf| {
            bencher.iter(|| {
                let (frame, consumed) = frame::parse(std::hint::black_box(buf)).unwrap();
                std::hint::black_box((frame, consumed));
            });
        },
    );
    let codec = QuicFrameCodec;
    group.bench_with_input(BenchmarkId::new("trait", "ping"), &ping, |bencher, buf| {
        bencher.iter(|| {
            let (frame, consumed) = codec.parse_frame(std::hint::black_box(buf)).unwrap();
            std::hint::black_box((frame, consumed));
        });
    });

    group.finish();
}

criterion_group!(benches, bench_parse);
criterion_main!(benches);
