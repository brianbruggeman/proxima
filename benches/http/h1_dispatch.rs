#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! H1 hot-path microbench. Three measurements:
//!
//! 1. `parse_head` standalone — the zero-copy slice scanner on a
//!    realistic head (request line + 5 typical headers).
//! 2. `connection_round_trip_no_body` — full Connection cycle for a
//!    GET / no body: feed bytes, poll, read head, begin/end response,
//!    reset. Captures the per-request allocation overhead.
//! 3. `connection_round_trip_post_with_body` — same cycle for a POST
//!    with Content-Length 5. Adds body decoding to the path.
//!
//! These exist so Tier 2-4 optimizations (SIMD CRLF, per-connection
//! arena, etc.) have a baseline to compare against. The architectural
//! claim is 5M+ req/sec/core; this measures how far the current
//! design is from that target on this machine.

use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use proxima::h1::{Status, parse_head};
use proxima::h1_body::BodyFraming;
use proxima::h1_connection::{Connection, Poll};

const SMALL_GET: &[u8] = b"GET /v1/items HTTP/1.1\r\n\
Host: api.example.com\r\n\
User-Agent: proxima-bench/0.1\r\n\
Accept: application/json\r\n\
Accept-Encoding: gzip\r\n\
Connection: keep-alive\r\n\
\r\n";

const SMALL_POST: &[u8] = b"POST /v1/items HTTP/1.1\r\n\
Host: api.example.com\r\n\
Content-Type: application/json\r\n\
Content-Length: 5\r\n\
Connection: keep-alive\r\n\
\r\n\
hello";

fn parse_head_only(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h1_parse_head");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    group.bench_function("small_get_5_headers", |bencher| {
        bencher.iter(|| {
            let status = parse_head(std::hint::black_box(SMALL_GET)).expect("parse");
            match status {
                Status::Complete { head, consumed } => {
                    std::hint::black_box(head.method);
                    std::hint::black_box(consumed);
                }
                Status::Partial => panic!("expected Complete"),
            }
        });
    });
    group.finish();
}

fn connection_round_trip_no_body(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h1_connection_round_trip_no_body");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    let response_headers = vec![("content-length".to_string(), "2".to_string())];
    group.bench_function("get_then_200_ok", |bencher| {
        let mut connection = Connection::new();
        // Output buffer reused across iterations — listeners allocate
        // one per connection, not per request.
        let mut out = Vec::with_capacity(512);
        bencher.iter(|| {
            connection.feed_bytes(std::hint::black_box(SMALL_GET));
            match connection.poll().expect("poll") {
                Poll::RequestReady => {}
                other => panic!("unexpected poll outcome: {other:?}"),
            }
            // Use direct accessors (no Vec allocation) instead of
            // head() which still allocates a headers Vec.
            std::hint::black_box(connection.path());
            std::hint::black_box(connection.body());
            out.clear();
            let writer = connection.begin_response(
                200,
                "OK",
                std::hint::black_box(&response_headers),
                BodyFraming::ContentLength(2),
                &mut out,
            );
            writer.write_chunk(b"ok", &mut out);
            writer.end_response(&mut out);
            std::hint::black_box(out.len());
            connection.reset_for_next_request();
        });
    });
    group.finish();
}

fn connection_round_trip_post_with_body(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h1_connection_round_trip_post_with_body");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));
    let response_headers = vec![("content-length".to_string(), "5".to_string())];
    let echo_body: &[u8] = b"hello";
    group.bench_function("post_with_5_byte_body_then_200_echo", |bencher| {
        let mut connection = Connection::new();
        let mut out = Vec::with_capacity(512);
        bencher.iter(|| {
            connection.feed_bytes(std::hint::black_box(SMALL_POST));
            match connection.poll().expect("poll") {
                Poll::RequestReady => {}
                other => panic!("unexpected poll outcome: {other:?}"),
            }
            std::hint::black_box(connection.path());
            std::hint::black_box(connection.body());
            out.clear();
            let writer = connection.begin_response(
                200,
                "OK",
                std::hint::black_box(&response_headers),
                BodyFraming::ContentLength(5),
                &mut out,
            );
            // Echo: write a fixed reference body to bypass the
            // immutable-borrow-then-mutable-borrow issue. The body
            // slice borrowed from connection has been consumed via
            // black_box above; we write a constant ref here.
            writer.write_chunk(echo_body, &mut out);
            writer.end_response(&mut out);
            std::hint::black_box(out.len());
            connection.reset_for_next_request();
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    parse_head_only,
    connection_round_trip_no_body,
    connection_round_trip_post_with_body
);
criterion_main!(benches);
