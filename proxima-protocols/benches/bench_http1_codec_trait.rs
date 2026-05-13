#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! C3 of the codec-trait initiative — proxima-h1-codec.
//!
//! Three arms on the same HTTP/1.1 request head + body sweep:
//!
//! - `concrete_decode` — `proxima_protocols::http1_codec::h1::parse_head` (the
//!   workspace's established hot path; wraps httparse internally).
//! - `trait_decode` — `H1RequestCodec::parse_frame` (the new
//!   FrameCodec-routed path).
//! - `httparse_decode` — direct `httparse::Request::parse` — the
//!   incumbent's home turf (parse a request head into a borrowed
//!   slice of headers). Apples-to-apples scope: parse head, no body
//!   decoder.
//!
//! Payload sizes simulate realistic shapes:
//! - small: 256 B head (~6 headers, /path)
//! - typical: 4 KiB head (~30 headers, /v1/messages-shaped path)
//! - large: 16 KiB head (~80 headers, edge case)

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use proxima_codec::FrameCodec;
use proxima_protocols::http1_codec::{H1RequestCodec, h1::parse_head as concrete_parse};

fn build_request_head(header_count: usize) -> Vec<u8> {
    let mut buf = String::new();
    buf.push_str("POST /v1/messages HTTP/1.1\r\n");
    for index in 0..header_count {
        buf.push_str(&format!(
            "x-proxima-h{index:03}: value-padded-to-bench-width-{index:03}\r\n"
        ));
    }
    buf.push_str("\r\n");
    buf.into_bytes()
}

fn bench_decode(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h1_codec_trait_decode");
    group.measurement_time(Duration::from_secs(5));
    for &header_count in &[6usize, 30, 80] {
        let bytes = build_request_head(header_count);
        group.throughput(Throughput::Bytes(bytes.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("concrete", header_count),
            &bytes,
            |bencher, buf| {
                bencher.iter(|| {
                    let outcome = concrete_parse(std::hint::black_box(buf)).unwrap();
                    std::hint::black_box(outcome);
                });
            },
        );

        let codec = H1RequestCodec;
        group.bench_with_input(
            BenchmarkId::new("trait", header_count),
            &bytes,
            |bencher, buf| {
                bencher.iter(|| {
                    let (head, consumed) = codec.parse_frame(std::hint::black_box(buf)).unwrap();
                    std::hint::black_box((head.headers.len(), consumed));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("httparse", header_count),
            &bytes,
            |bencher, buf| {
                bencher.iter(|| {
                    let mut headers = [httparse::EMPTY_HEADER; 128];
                    let mut request = httparse::Request::new(&mut headers);
                    let status = request.parse(std::hint::black_box(buf)).unwrap();
                    std::hint::black_box(status);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
