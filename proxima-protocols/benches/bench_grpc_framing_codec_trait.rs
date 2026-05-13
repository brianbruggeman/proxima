#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! C9 of the codec-trait initiative — proxima-grpc-framing.
//!
//! Compares three arms on the same payload size sweep:
//!
//! - `concrete_decode` / `concrete_encode` — the existing
//!   `proxima_protocols::grpc_framing::{parse, encode}` free functions (the
//!   workspace's established hot path).
//! - `trait_decode` / `trait_encode` — the new
//!   `GrpcFrameCodec::{parse_frame, encode_frame}` `FrameCodec`-routed
//!   path. The C11 e2e gate condition: trait arm must be within
//!   noise floor of concrete arm.
//! - `incumbent_*` — DEFERRED. gRPC's 5-byte length-prefix codec is
//!   not exposed publicly by `tonic` (its `Streaming` decoder bundles
//!   framing with HTTP/2). Recorded in
//!   `docs/codec-trait/baselines.md` for documentation; no
//!   incumbent arm in this file.
//!
//! Three payload sizes (16 B, 1 KiB, 64 KiB) so the per-frame fixed
//! cost is visible at small sizes and memcpy dominates at large.
//!
//! Bench discipline (per `docs/codec-trait/discipline.md`):
//! - record range (e.g. 12.1–13.4 M ops/s), not point estimates.
//! - CoV ≤ 5% per arm or rerun with more iterations.
//! - capture host loadout in baselines.md before sealing the C9
//!   Compare-bench column.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use proxima_codec::FrameCodec;
use proxima_protocols::grpc_framing::{
    Compression, Frame, GrpcFrameCodec, encode as concrete_encode, parse as concrete_parse,
};

const SIZES: &[usize] = &[16, 1024, 64 * 1024];

fn make_frame_bytes(payload_size: usize) -> Vec<u8> {
    let payload = vec![0xa5u8; payload_size];
    let mut buf = Vec::new();
    concrete_encode(&payload, Compression::None, &mut buf);
    buf
}

fn bench_decode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("grpc_codec_trait_decode");
    group.measurement_time(Duration::from_secs(5));
    for &size in SIZES {
        let bytes = make_frame_bytes(size);
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

        let codec = GrpcFrameCodec;
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
    let mut group = criterion.benchmark_group("grpc_codec_trait_encode");
    group.measurement_time(Duration::from_secs(5));
    for &size in SIZES {
        let payload = vec![0xa5u8; size];
        group.throughput(Throughput::Bytes((size + 5) as u64));

        group.bench_with_input(
            BenchmarkId::new("concrete", size),
            &payload,
            |bencher, payload| {
                let mut dest = Vec::with_capacity(payload.len() + 5);
                bencher.iter(|| {
                    dest.clear();
                    concrete_encode(std::hint::black_box(payload), Compression::None, &mut dest);
                    std::hint::black_box(dest.len());
                });
            },
        );

        let codec = GrpcFrameCodec;
        group.bench_with_input(
            BenchmarkId::new("trait", size),
            &payload,
            |bencher, payload| {
                let mut dest = Vec::with_capacity(payload.len() + 5);
                let frame = Frame {
                    compression: Compression::None,
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
