// bench fixtures legitimately fail-fast on encoder errors.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! C2 — packet header codec bench arms.
//!
//! Home-turf incumbent: `quinn-proto::packet::PartialDecode`. Quinn's
//! partial decoder pulls in `ConnectionIdParser`, `BytesMut`, supported-
//! versions slice and grease bits — it's a heavier surface than our pure-
//! slice [`parse_long`] / [`parse_short`]. The bench arms focus on:
//!
//! - **Initial parse** at the typical Initial datagram shape (8-byte DCID,
//!   0-byte SCID, 0..32 byte token, ~1200-byte payload).
//! - **Short parse** at 1-RTT shape (8-byte DCID, ~1200-byte payload).
//! - **Initial encode** writing into a caller-owned buffer.
//!
//! Multi-arch SIMD: codec is branch-light + slice-copy dominated; no
//! hand-rolled SIMD. iai-callgrind cycle counts deferred to Linux x86_64
//! CI per `docs/proxima-quic/edges.md`.

use bytes::BytesMut;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima_protocols::quic::packet::header::{self, Header, MAX_CID_LEN, RETRY_INTEGRITY_TAG_LEN};

fn build_initial_bytes() -> Vec<u8> {
    // realistic Initial: 8-byte DCID, 0-byte SCID, 8-byte token, 1200-byte payload
    let dcid: [u8; 8] = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];
    let scid: [u8; 0] = [];
    let token: [u8; 8] = [0xaa; 8];
    let payload = vec![0x42u8; 1200];
    let header = Header::Initial {
        version: 1,
        dcid: &dcid,
        scid: &scid,
        token: &token,
        length: payload.len() as u64,
        pn_and_payload: &payload,
    };
    let mut buffer = vec![0u8; 2048];
    let written = header.encode(&mut buffer).expect("encode");
    buffer.truncate(written);
    buffer
}

fn build_short_bytes(dcid_len: usize) -> Vec<u8> {
    let dcid = vec![0x42u8; dcid_len];
    let payload = vec![0xabu8; 1200];
    let header = Header::Short {
        first_byte: 0b0100_0000,
        dcid: &dcid,
        pn_and_payload: &payload,
    };
    let mut buffer = vec![0u8; 2048];
    let written = header.encode(&mut buffer).expect("encode");
    buffer.truncate(written);
    buffer
}

fn bench_parse_initial(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c2_parse_initial");
    let bytes = build_initial_bytes();
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("proxima_quic_proto", |bencher| {
        bencher.iter(|| {
            let header = header::parse_long(std::hint::black_box(&bytes)).expect("parse");
            std::hint::black_box(header);
        });
    });
    group.bench_function("quinn_proto_partial_decode", |bencher| {
        // quinn-proto's PartialDecode requires a CID parser; we use the
        // fixed-length parser since our test DCID is fixed at 8 bytes.
        let cid_parser = quinn_proto::FixedLengthConnectionIdParser::new(8);
        let supported = [1u32];
        bencher.iter(|| {
            let buf = BytesMut::from(&bytes[..]);
            let (decoded, _rest) = quinn_proto::PartialDecode::new(
                std::hint::black_box(buf),
                std::hint::black_box(&cid_parser),
                std::hint::black_box(&supported),
                false,
            )
            .expect("partial decode");
            std::hint::black_box(decoded);
        });
    });
    group.finish();
}

fn bench_parse_short(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c2_parse_short");
    let dcid_len = 8usize;
    let bytes = build_short_bytes(dcid_len);
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("proxima_quic_proto", |bencher| {
        bencher.iter(|| {
            let header =
                header::parse_short(std::hint::black_box(&bytes), dcid_len).expect("parse");
            std::hint::black_box(header);
        });
    });
    group.bench_function("quinn_proto_partial_decode", |bencher| {
        let cid_parser = quinn_proto::FixedLengthConnectionIdParser::new(dcid_len);
        let supported = [1u32];
        bencher.iter(|| {
            let buf = BytesMut::from(&bytes[..]);
            let (decoded, _rest) = quinn_proto::PartialDecode::new(
                std::hint::black_box(buf),
                std::hint::black_box(&cid_parser),
                std::hint::black_box(&supported),
                false,
            )
            .expect("partial decode");
            std::hint::black_box(decoded);
        });
    });
    group.finish();
}

fn bench_encode_initial(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c2_encode_initial");
    let dcid: [u8; 8] = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];
    let scid: [u8; 0] = [];
    let token: [u8; 8] = [0xaa; 8];
    let payload = vec![0x42u8; 1200];
    let header = Header::Initial {
        version: 1,
        dcid: &dcid,
        scid: &scid,
        token: &token,
        length: payload.len() as u64,
        pn_and_payload: &payload,
    };
    group.throughput(Throughput::Bytes(1220));
    group.bench_function("proxima_quic_proto", |bencher| {
        let mut buffer = vec![0u8; 2048];
        bencher.iter(|| {
            let written = header
                .encode(std::hint::black_box(&mut buffer))
                .expect("encode");
            std::hint::black_box(written);
        });
    });
    group.finish();
}

fn bench_round_trip_retry(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c2_round_trip_retry");
    let dcid: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let scid: [u8; 0] = [];
    let retry_token = vec![0xbcu8; 24];
    let integrity_tag = [0xeeu8; RETRY_INTEGRITY_TAG_LEN];
    let original = Header::Retry {
        version: 1,
        dcid: &dcid,
        scid: &scid,
        retry_token: &retry_token,
        integrity_tag: &integrity_tag,
    };
    let mut wire = vec![0u8; 128];
    let written = original.encode(&mut wire).expect("encode");
    wire.truncate(written);
    group.throughput(Throughput::Bytes(wire.len() as u64));
    group.bench_function("proxima_quic_proto", |bencher| {
        bencher.iter(|| {
            let parsed = header::parse_long(std::hint::black_box(&wire)).expect("parse");
            std::hint::black_box(parsed);
        });
    });
    group.finish();
}

const _MAX_CID: usize = MAX_CID_LEN;

criterion_group!(
    benches,
    bench_parse_initial,
    bench_parse_short,
    bench_encode_initial,
    bench_round_trip_retry,
);
criterion_main!(benches);
