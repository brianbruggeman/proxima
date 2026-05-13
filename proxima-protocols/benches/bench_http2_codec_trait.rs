#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! C4 of the codec-trait initiative — proxima-h2-codec.
//!
//! Two arms on a frame-mix sweep:
//!
//! - `concrete_parse` — existing `FrameHeader::parse` + `parse_payload`
//!   (the workspace's established hot path).
//! - `trait_parse` — `H2FrameCodec::parse_frame` (the new FrameCodec-
//!   routed path; pays one `Bytes::copy_from_slice` on the &[u8] →
//!   &Bytes boundary; see codec_trait.rs documentation).
//!
//! `h2::frame` is NOT public on the `h2` crate (the framer is private
//! and bundled with the connection state machine); there is no
//! apples-to-apples incumbent at this layer. Documented in
//! `docs/codec-trait/baselines.md`. callers wanting an h2-internal
//! comparison have to bench through h2's `Client` / `Server` which is
//! at a different abstraction tier.
//!
//! Three frame sizes — the per-frame fixed cost (9-byte header) is
//! visible at small sizes; memcpy dominates at large.

use std::time::Duration;

use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use proxima_codec::FrameCodec;
use proxima_protocols::http2_codec::{
    H2FrameCodec,
    frame::{FRAME_HEADER_LEN, FrameHeader, FramePayload, FrameType, encode_frame, parse_payload},
};

const PAYLOAD_SIZES: &[usize] = &[16, 1024, 16 * 1024];

fn build_data_frame(payload_size: usize) -> Vec<u8> {
    let payload = Bytes::from(vec![0xa5u8; payload_size]);
    let mut buf = Vec::with_capacity(FRAME_HEADER_LEN + payload_size);
    encode_frame(
        FrameType::Data,
        0,
        1,
        &FramePayload::Data { data: payload },
        &mut buf,
    );
    buf
}

fn bench_parse(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h2_codec_trait_parse");
    group.measurement_time(Duration::from_secs(5));
    for &size in PAYLOAD_SIZES {
        let bytes = build_data_frame(size);
        group.throughput(Throughput::Bytes(bytes.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("concrete", size),
            &bytes,
            |bencher, buf| {
                bencher.iter(|| {
                    let header = FrameHeader::parse(std::hint::black_box(buf)).unwrap();
                    let payload_len = header.length as usize;
                    let payload = Bytes::copy_from_slice(
                        &buf[FRAME_HEADER_LEN..FRAME_HEADER_LEN + payload_len],
                    );
                    let parsed = parse_payload(&header, &payload).unwrap();
                    std::hint::black_box(parsed);
                });
            },
        );

        let codec = H2FrameCodec;
        group.bench_with_input(BenchmarkId::new("trait", size), &bytes, |bencher, buf| {
            bencher.iter(|| {
                let (frame, consumed) = codec.parse_frame(std::hint::black_box(buf)).unwrap();
                std::hint::black_box((frame.header.length, consumed));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_parse);
criterion_main!(benches);
