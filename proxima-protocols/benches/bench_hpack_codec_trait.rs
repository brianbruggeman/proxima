#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! C7 of the codec-trait initiative — proxima-hpack.
//!
//! Two arms on a realistic JSON request header set:
//!
//! - `concrete_encode` / `concrete_decode` — existing `encode_block` /
//!   `decode_block` free functions with a directly-owned
//!   `DynamicTable`.
//! - `trait_encode` / `trait_decode` — `HpackCodec::new_encoder` /
//!   `new_decoder` vending per-session encoder / decoder instances
//!   (StatefulCodec contract). The factory call itself is part of
//!   the measured cost so the bench captures any factory overhead.
//!
//! The unmaintained `hpack` crate is intentionally not used as a baseline.
//! The h2-backed HPACK benches provide the maintained incumbent comparison.

use std::time::Duration;

use bytes::{Bytes, BytesMut};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima_codec::StatefulCodec;
use proxima_protocols::hpack::{DynamicTable, HpackCodec, decode_block, encode_block};

fn realistic_headers() -> Vec<(Bytes, Bytes)> {
    vec![
        (Bytes::from_static(b":method"), Bytes::from_static(b"POST")),
        (Bytes::from_static(b":scheme"), Bytes::from_static(b"https")),
        (
            Bytes::from_static(b":path"),
            Bytes::from_static(b"/v1/messages"),
        ),
        (
            Bytes::from_static(b":authority"),
            Bytes::from_static(b"api.example.com"),
        ),
        (
            Bytes::from_static(b"x-api-key"),
            Bytes::from_static(b"sk-example-very-long-secret-key-padded-for-realistic-size"),
        ),
        (
            Bytes::from_static(b"x-api-version"),
            Bytes::from_static(b"2023-06-01"),
        ),
        (
            Bytes::from_static(b"content-type"),
            Bytes::from_static(b"application/json"),
        ),
        (
            Bytes::from_static(b"accept"),
            Bytes::from_static(b"application/json"),
        ),
        (
            Bytes::from_static(b"content-length"),
            Bytes::from_static(b"2048"),
        ),
        (
            Bytes::from_static(b"user-agent"),
            Bytes::from_static(b"example-sdk/0.42.0"),
        ),
    ]
}

fn pre_encoded(headers: &[(Bytes, Bytes)]) -> Bytes {
    let mut table = DynamicTable::new(4096);
    let mut buf = BytesMut::new();
    encode_block(headers.iter().cloned(), &mut table, &mut buf);
    buf.freeze()
}

fn bench_encode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("hpack_codec_trait_encode");
    group.measurement_time(Duration::from_secs(5));
    let headers = realistic_headers();
    group.throughput(Throughput::Elements(headers.len() as u64));

    group.bench_function("concrete", |bencher| {
        let mut table = DynamicTable::new(4096);
        let mut buf = BytesMut::with_capacity(1024);
        bencher.iter(|| {
            buf.clear();
            encode_block(
                std::hint::black_box(&headers).iter().cloned(),
                &mut table,
                &mut buf,
            );
            std::hint::black_box(buf.len());
        });
    });

    let codec = HpackCodec::new();
    group.bench_function("trait", |bencher| {
        let mut encoder = codec.new_encoder().unwrap();
        let mut buf = BytesMut::with_capacity(1024);
        bencher.iter(|| {
            buf.clear();
            encoder.encode_block(std::hint::black_box(&headers).iter().cloned(), &mut buf);
            std::hint::black_box(buf.len());
        });
    });

    group.finish();
}

fn bench_decode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("hpack_codec_trait_decode");
    group.measurement_time(Duration::from_secs(5));
    let headers = realistic_headers();
    let encoded = pre_encoded(&headers);
    group.throughput(Throughput::Bytes(encoded.len() as u64));

    group.bench_function("concrete", |bencher| {
        let mut table = DynamicTable::new(4096);
        bencher.iter(|| {
            let mut count = 0usize;
            decode_block(std::hint::black_box(&encoded), &mut table, 4096, |_, _| {
                count += 1
            })
            .unwrap();
            std::hint::black_box(count);
        });
    });

    let codec = HpackCodec::new();
    group.bench_function("trait", |bencher| {
        let mut decoder = codec.new_decoder().unwrap();
        bencher.iter(|| {
            let mut count = 0usize;
            decoder
                .decode_block(std::hint::black_box(&encoded), 4096, |_, _| count += 1)
                .unwrap();
            std::hint::black_box(count);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
