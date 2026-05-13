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

//! C3 — QUIC frame codec bench arms.
//!
//! **Note on home-turf comparison**: `quinn-proto::frame::Frame` is
//! `pub(crate)` upstream, so no direct head-to-head parser bench exists.
//! The home-turf claim for C3 is structural — composition of C1 varint
//! (which beat quinn-proto's `VarInt` by 3-7× on encode) and the tier-3
//! pure-slice design avoids the `BytesMut` allocation that quinn's
//! `frame::Iter` carries. C1's home-turf bench is the load-bearing
//! number; C3's bench measures proxima's absolute throughput on
//! representative wire shapes.
//!
//! Arms:
//!
//! - **mixed datagram parse** — typical ACK + STREAM + PING mix.
//! - **ACK parse** with N range pairs (5, 50, 500).
//! - **STREAM parse** at 1200-byte payload.
//! - **STREAM encode** at 1200-byte payload.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use proxima_protocols::quic::frame::{self, EcnCounts, Frame};
use proxima_protocols::quic::varint;

fn build_ack(range_count: u64) -> Vec<u8> {
    // ranges_raw: range_count pairs of (gap, length) varints
    let mut ranges_raw = vec![0u8; range_count as usize * 16];
    let mut cursor = 0;
    for index in 0..range_count {
        cursor += varint::encode(index % 7, &mut ranges_raw[cursor..]).unwrap();
        cursor += varint::encode((index + 1) % 11, &mut ranges_raw[cursor..]).unwrap();
    }
    ranges_raw.truncate(cursor);
    let frame = Frame::Ack {
        largest: 1000,
        delay: 23,
        first_range: 5,
        ranges_raw: &ranges_raw,
        range_count,
        ecn: Some(EcnCounts {
            ect0: 100,
            ect1: 50,
            ecn_ce: 10,
        }),
    };
    let mut buffer = vec![0u8; 16 * 1024];
    let written = frame.encode(&mut buffer).expect("encode");
    buffer.truncate(written);
    buffer
}

fn build_stream(payload_len: usize) -> Vec<u8> {
    let payload = vec![0x42u8; payload_len];
    let frame = Frame::Stream {
        stream_id: 8,
        offset: 1024,
        data: &payload,
        fin: false,
    };
    let mut buffer = vec![0u8; payload_len + 32];
    let written = frame.encode(&mut buffer).expect("encode");
    buffer.truncate(written);
    buffer
}

fn build_mixed_datagram() -> Vec<u8> {
    // 100 padding + 1 PING + 1 ACK (5 ranges) + 1 STREAM (200 B) + HANDSHAKE_DONE
    let mut buffer = vec![0u8; 4096];
    let mut cursor = 0;
    // padding
    cursor += Frame::Padding { count: 100 }
        .encode(&mut buffer[cursor..])
        .unwrap();
    cursor += Frame::Ping.encode(&mut buffer[cursor..]).unwrap();
    let ack_bytes = build_ack(5);
    buffer[cursor..cursor + ack_bytes.len()].copy_from_slice(&ack_bytes);
    cursor += ack_bytes.len();
    let stream_bytes = build_stream(200);
    buffer[cursor..cursor + stream_bytes.len()].copy_from_slice(&stream_bytes);
    cursor += stream_bytes.len();
    cursor += Frame::HandshakeDone.encode(&mut buffer[cursor..]).unwrap();
    buffer.truncate(cursor);
    buffer
}

fn bench_parse_mixed(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c3_parse_mixed_datagram");
    let bytes = build_mixed_datagram();
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("proxima_quic_proto", |bencher| {
        bencher.iter(|| {
            let mut input = bytes.as_slice();
            let mut frames = 0u64;
            while !input.is_empty() {
                let (parsed, consumed) = frame::parse(std::hint::black_box(input)).expect("parse");
                std::hint::black_box(parsed);
                input = &input[consumed..];
                frames += 1;
            }
            std::hint::black_box(frames);
        });
    });
    group.finish();
}

fn bench_parse_ack(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c3_parse_ack");
    for &range_count in &[5u64, 50, 500] {
        let bytes = build_ack(range_count);
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("proxima_quic_proto", range_count),
            &bytes,
            |bencher, bytes| {
                bencher.iter(|| {
                    let (parsed, consumed) =
                        frame::parse(std::hint::black_box(bytes)).expect("parse");
                    std::hint::black_box((parsed, consumed));
                });
            },
        );
    }
    group.finish();
}

fn bench_parse_stream(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c3_parse_stream");
    for &payload_len in &[16usize, 1024, 1200, 8192] {
        let bytes = build_stream(payload_len);
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("proxima_quic_proto", payload_len),
            &bytes,
            |bencher, bytes| {
                bencher.iter(|| {
                    let (parsed, consumed) =
                        frame::parse(std::hint::black_box(bytes)).expect("parse");
                    std::hint::black_box((parsed, consumed));
                });
            },
        );
    }
    group.finish();
}

fn bench_encode_stream(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("c3_encode_stream");
    let payload = vec![0x42u8; 1200];
    let frame = Frame::Stream {
        stream_id: 8,
        offset: 1024,
        data: &payload,
        fin: false,
    };
    group.throughput(Throughput::Bytes(1200));
    group.bench_function("proxima_quic_proto", |bencher| {
        let mut buffer = vec![0u8; 2048];
        bencher.iter(|| {
            let written = frame
                .encode(std::hint::black_box(&mut buffer))
                .expect("encode");
            std::hint::black_box(written);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_parse_mixed,
    bench_parse_ack,
    bench_parse_stream,
    bench_encode_stream,
);
criterion_main!(benches);
