#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! HPACK header block encode/decode end-to-end microbench.
//!
//! Per-layer benches (`hpack_integer`, `hpack_huffman`, `hpack_static_table`)
//! cover the algorithmic head-to-head against h2-0.4.14. This bench
//! tracks the integrated cost — the encoder picking representations,
//! the decoder dispatching on the first byte, the dynamic table
//! mutating — so we catch regressions in the orchestration layer
//! itself.
//!
//! Workloads model realistic HTTP request shapes:
//! - `request_minimal`  : 4 pseudo-headers (`:method GET`, etc.) — every header is a static hit
//! - `request_browser`  : pseudo + common browser headers (accept, user-agent, cookie, ...)
//! - `request_api`      : pseudo + bearer auth + content-type + content-length + x-request-id
//! - `response_minimal` : `:status 200` + content-type + content-length
//! - `response_cors`    : `:status 200` + access-control-* + content-type
//!
//! Apples-to-apples vs h2 is deferred: h2 keeps hpack pub(crate) and
//! vendoring their full Encoder/Decoder requires their Header +
//! HeaderName + Table machinery (~1500 lines). Per-layer benches
//! already prove the algorithmic comparison.

use std::time::Duration;

use bytes::{Bytes, BytesMut};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima_protocols::hpack::{DynamicTable, decode_block, encode_block};

fn h(name: &'static [u8], value: &'static [u8]) -> (Bytes, Bytes) {
    (Bytes::from_static(name), Bytes::from_static(value))
}

struct Workload {
    label: &'static str,
    headers: Vec<(Bytes, Bytes)>,
}

fn workloads() -> Vec<Workload> {
    vec![
        Workload {
            label: "request_minimal",
            headers: vec![
                h(b":method", b"GET"),
                h(b":scheme", b"https"),
                h(b":path", b"/"),
                h(b":authority", b"example.com"),
            ],
        },
        Workload {
            label: "request_browser",
            headers: vec![
                h(b":method", b"GET"),
                h(b":scheme", b"https"),
                h(b":path", b"/index.html"),
                h(b":authority", b"www.example.com"),
                h(b"user-agent", b"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
                h(b"accept", b"text/html,application/xhtml+xml,application/xml;q=0.9"),
                h(b"accept-language", b"en-US,en;q=0.9"),
                h(b"accept-encoding", b"gzip, deflate"),
                h(b"cookie", b"session=abc123; user=42; locale=en_US"),
            ],
        },
        Workload {
            label: "request_api",
            headers: vec![
                h(b":method", b"POST"),
                h(b":scheme", b"https"),
                h(b":path", b"/api/v1/users"),
                h(b":authority", b"api.example.com"),
                h(b"content-type", b"application/json"),
                h(b"content-length", b"1024"),
                h(b"authorization", b"Bearer t0k3n-eyJhbGciOiJIUzI1NiJ9"),
                h(b"x-request-id", b"01HAB7PXY9X3K4M5N6P7Q8R9S0"),
            ],
        },
        Workload {
            label: "response_minimal",
            headers: vec![
                h(b":status", b"200"),
                h(b"content-type", b"application/json"),
                h(b"content-length", b"512"),
            ],
        },
        Workload {
            label: "response_cors",
            headers: vec![
                h(b":status", b"200"),
                h(b"content-type", b"application/json"),
                h(b"access-control-allow-origin", b"https://app.example.com"),
                h(b"vary", b"Origin"),
                h(b"server", b"proxima/0.1.0"),
            ],
        },
    ]
}

fn encode_bench(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("hpack_block_encode");
    group.measurement_time(Duration::from_secs(2));
    for workload in workloads() {
        group.throughput(Throughput::Elements(workload.headers.len() as u64));
        group.bench_function(workload.label, |bencher| {
            let mut buffer = BytesMut::with_capacity(512);
            let mut table = DynamicTable::new(4096);
            bencher.iter(|| {
                buffer.clear();
                encode_block(workload.headers.clone(), &mut table, &mut buffer);
                std::hint::black_box(buffer.len());
            });
        });
    }
    group.finish();
}

fn decode_bench(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("hpack_block_decode");
    group.measurement_time(Duration::from_secs(2));
    for workload in workloads() {
        // Pre-encode each block so the decode bench only times decode.
        let mut buffer = BytesMut::with_capacity(512);
        let mut encode_table = DynamicTable::new(4096);
        encode_block(workload.headers.clone(), &mut encode_table, &mut buffer);
        let encoded = buffer.freeze();
        group.throughput(Throughput::Elements(workload.headers.len() as u64));
        group.bench_function(workload.label, |bencher| {
            let mut decode_table = DynamicTable::new(4096);
            let mut count = 0usize;
            bencher.iter(|| {
                count = 0;
                decode_block(
                    std::hint::black_box(&encoded),
                    &mut decode_table,
                    4096,
                    |_name, _value| {
                        count += 1;
                    },
                )
                .expect("decode");
                std::hint::black_box(count);
            });
        });
    }
    group.finish();
}

fn roundtrip_bench(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("hpack_block_roundtrip");
    group.measurement_time(Duration::from_secs(2));
    for workload in workloads() {
        group.throughput(Throughput::Elements(workload.headers.len() as u64));
        group.bench_function(workload.label, |bencher| {
            let mut buffer = BytesMut::with_capacity(512);
            let mut encode_table = DynamicTable::new(4096);
            let mut decode_table = DynamicTable::new(4096);
            let mut count = 0usize;
            bencher.iter(|| {
                buffer.clear();
                encode_block(workload.headers.clone(), &mut encode_table, &mut buffer);
                let block = buffer.clone().freeze();
                count = 0;
                decode_block(
                    std::hint::black_box(&block),
                    &mut decode_table,
                    4096,
                    |_name, _value| {
                        count += 1;
                    },
                )
                .expect("decode");
                std::hint::black_box(count);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, encode_bench, decode_bench, roundtrip_bench);
criterion_main!(benches);
