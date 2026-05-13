#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! C5 of the codec-trait initiative — proxima-h3-proto.
//!
//! Two arms on a DATA frame size sweep:
//!
//! - `concrete_parse` / `concrete_encode` — existing
//!   `frame::parse` + `frame::encode` (the workspace's established
//!   hot path; already returns `(Frame<'_>, usize)` matching the
//!   FrameCodec contract verbatim).
//! - `trait_parse` / `trait_encode` — `H3FrameCodec::parse_frame` /
//!   `encode_frame` (the new FrameCodec-routed path; pure
//!   delegation to the concrete functions plus a Vec<u8> bridge
//!   on encode).
//!
//! The quinn-h3 frame parser is not publicly exposed at the same
//! abstraction layer (quinn-h3's H3 codec bundles framing with
//! per-stream state in a Tokio-friendly shape). No apples-to-apples
//! incumbent arm; recorded in `docs/codec-trait/baselines.md`.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use proxima_codec::FrameCodec;
use proxima_protocols::http3_codec::{
    H3FrameCodec,
    frame::{H3Frame, encode, parse},
};

const PAYLOAD_SIZES: &[usize] = &[16, 1024, 16 * 1024];

fn build_data_frame(payload: &[u8]) -> Vec<u8> {
    let mut buf = vec![0u8; payload.len() + 16];
    let written = encode(&H3Frame::Data { payload }, &mut buf).unwrap();
    buf.truncate(written);
    buf
}

fn bench_parse(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h3_codec_trait_parse");
    group.measurement_time(Duration::from_secs(5));
    for &size in PAYLOAD_SIZES {
        let payload = vec![0xa5u8; size];
        let bytes = build_data_frame(&payload);
        group.throughput(Throughput::Bytes(bytes.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("concrete", size),
            &bytes,
            |bencher, buf| {
                bencher.iter(|| {
                    let (frame, consumed) = parse(std::hint::black_box(buf)).unwrap();
                    std::hint::black_box((frame, consumed));
                });
            },
        );

        let codec = H3FrameCodec;
        group.bench_with_input(BenchmarkId::new("trait", size), &bytes, |bencher, buf| {
            bencher.iter(|| {
                let (frame, consumed) = codec.parse_frame(std::hint::black_box(buf)).unwrap();
                std::hint::black_box((frame, consumed));
            });
        });
    }
    group.finish();
}

fn bench_encode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h3_codec_trait_encode");
    group.measurement_time(Duration::from_secs(5));
    for &size in PAYLOAD_SIZES {
        let payload = vec![0xa5u8; size];
        group.throughput(Throughput::Bytes((size + 16) as u64));

        group.bench_with_input(
            BenchmarkId::new("concrete", size),
            &payload,
            |bencher, payload| {
                let mut dest = vec![0u8; payload.len() + 16];
                bencher.iter(|| {
                    let written = encode(
                        &H3Frame::Data {
                            payload: std::hint::black_box(payload),
                        },
                        &mut dest,
                    )
                    .unwrap();
                    std::hint::black_box(written);
                });
            },
        );

        let codec = H3FrameCodec;
        group.bench_with_input(
            BenchmarkId::new("trait", size),
            &payload,
            |bencher, payload| {
                let mut dest = Vec::with_capacity(payload.len() + 16);
                let frame = H3Frame::Data { payload };
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

criterion_group!(benches, bench_parse, bench_encode);
criterion_main!(benches);
