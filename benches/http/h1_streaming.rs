#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! Streaming-vs-buffered Connection microbench.
//!
//! Measures per-request overhead the streaming dispatch path adds on
//! top of the buffered path for the same body. Streaming pays for:
//! `Bytes::copy_from_slice` per chunk (vs one copy at request-build
//! time in buffered mode), `VecDeque::push/pop`, an extra poll cycle
//! per chunk, and the head_emitted/body_end_emitted state checks.
//!
//! Three body shapes:
//! - 256-byte content-length (small body — streaming overhead matters
//!   most here, this is the worst case)
//! - 64 KiB content-length (single-chunk medium body)
//! - 16x4 KiB chunked transfer-encoding (multi-chunk streaming case
//!   — the workload streaming is designed for)
//!
//! Each variant runs the full Connection cycle (feed → poll → take →
//! write response → reset) so we capture amortized per-request cost.
//!
//! Additionally, `h1_streaming_vs_hyper` bench adds a hyper arm
//! (http1::Builder over tokio::io::duplex) with hdrhistogram
//! p50/p90/p99/p999/max tail-latency reporting for both arms.

use std::convert::Infallible;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use http_body_util::StreamBody;

#[path = "../common/hdr_phased.rs"]
mod hdr_phased;
use hdr_phased::HdrQuartet;
use hyper::body::Frame;
use hyper::server::conn::http1;
use hyper::service::service_fn as pipe_fn;
use hyper_util::rt::TokioIo;
use proxima::h1_body::BodyFraming;
use proxima::h1_connection::{AutoStreamPolicy, Connection, Poll};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const STREAMING_CHUNK_COUNT: usize = 16;
const STREAMING_CHUNK_SIZE: usize = 4 * 1024;

fn build_request_with_content_length(body_bytes: usize) -> Vec<u8> {
    let mut request: Vec<u8> = Vec::with_capacity(body_bytes + 128);
    request.extend_from_slice(b"POST /v1/upload HTTP/1.1\r\n");
    request.extend_from_slice(b"Host: api.example.com\r\n");
    request.extend_from_slice(b"Content-Type: application/octet-stream\r\n");
    let _ = std::io::Write::write_fmt(
        &mut request,
        format_args!("Content-Length: {body_bytes}\r\n"),
    );
    request.extend_from_slice(b"Connection: keep-alive\r\n\r\n");
    request.resize(request.len() + body_bytes, b'A');
    request
}

fn build_chunked_request(chunks: &[&[u8]]) -> Vec<u8> {
    let mut request: Vec<u8> = Vec::with_capacity(8 * 1024);
    request.extend_from_slice(b"POST /v1/upload HTTP/1.1\r\n");
    request.extend_from_slice(b"Host: api.example.com\r\n");
    request.extend_from_slice(b"Transfer-Encoding: chunked\r\n");
    request.extend_from_slice(b"Connection: keep-alive\r\n\r\n");
    for chunk in chunks {
        let _ = std::io::Write::write_fmt(&mut request, format_args!("{:x}\r\n", chunk.len()));
        request.extend_from_slice(chunk);
        request.extend_from_slice(b"\r\n");
    }
    request.extend_from_slice(b"0\r\n\r\n");
    request
}

fn buffered_round_trip(criterion: &mut Criterion, label: &str, request: Vec<u8>) {
    let mut group = criterion.benchmark_group(format!("h1_buffered_{label}"));
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(request.len() as u64));
    let response_headers = vec![("content-length".to_string(), "2".to_string())];
    group.bench_function("request_to_response_to_reset", |bencher| {
        let mut connection = Connection::new();
        let mut out = Vec::with_capacity(512);
        bencher.iter(|| {
            connection.feed_bytes(std::hint::black_box(&request));
            match connection.poll().expect("poll") {
                Poll::RequestReady => {}
                other => panic!("unexpected: {other:?}"),
            }
            std::hint::black_box(connection.body().len());
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

fn streaming_round_trip(criterion: &mut Criterion, label: &str, request: Vec<u8>) {
    let mut group = criterion.benchmark_group(format!("h1_streaming_{label}"));
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(request.len() as u64));
    let response_headers = vec![("content-length".to_string(), "2".to_string())];
    // Tiny threshold so the policy fires for the small-body case
    // too (otherwise the 256-byte body would stay on the buffered
    // path and the bench would compare apples to apples).
    let policy = AutoStreamPolicy {
        content_length_threshold: 0,
        stream_chunked: true,
    };
    group.bench_function("request_to_response_to_reset", |bencher| {
        let mut connection = Connection::new();
        connection.set_auto_stream_policy(Some(policy));
        let mut out = Vec::with_capacity(512);
        bencher.iter(|| {
            connection.feed_bytes(std::hint::black_box(&request));
            // Drive the streaming state machine to completion. Drain
            // every chunk via take_body_chunk so the queue stays at
            // its design depth (≤1).
            loop {
                match connection.poll().expect("poll") {
                    Poll::HeadReady => {}
                    Poll::BodyChunk => {
                        let chunk = connection.take_body_chunk().expect("chunk");
                        std::hint::black_box(chunk.len());
                    }
                    Poll::BodyEnd => break,
                    Poll::NeedInput => panic!("need input mid-bench"),
                    other => panic!("unexpected: {other:?}"),
                }
            }
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

fn small_body_cl_256(criterion: &mut Criterion) {
    let request = build_request_with_content_length(256);
    buffered_round_trip(criterion, "cl_256", request.clone());
    streaming_round_trip(criterion, "cl_256", request);
}

fn medium_body_cl_64kib(criterion: &mut Criterion) {
    let request = build_request_with_content_length(64 * 1024);
    buffered_round_trip(criterion, "cl_64kib", request.clone());
    streaming_round_trip(criterion, "cl_64kib", request);
}

fn chunked_16x4kib(criterion: &mut Criterion) {
    let chunk = vec![b'A'; 4 * 1024];
    let chunk_refs: Vec<&[u8]> = (0..16).map(|_| chunk.as_slice()).collect();
    let request = build_chunked_request(&chunk_refs);
    // Buffered path runs on the same chunked wire format — Connection
    // decodes chunks into its buffer and returns RequestReady once
    // the terminator arrives.
    buffered_round_trip(criterion, "chunked_16x4kib", request.clone());
    streaming_round_trip(criterion, "chunked_16x4kib", request);
}

/// End-to-end h1 streaming bench: proxima Connection vs hyper http1::Builder,
/// both driven over tokio::io::duplex, serving STREAMING_CHUNK_COUNT chunks of
/// STREAMING_CHUNK_SIZE bytes via chunked transfer-encoding. Records per-iter
/// latency in an hdrhistogram and prints p50/p90/p99/p999/max at the end of
/// each measurement run.
fn h1_streaming_vs_hyper(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h1_streaming_vs_hyper");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Bytes(
        (STREAMING_CHUNK_COUNT * STREAMING_CHUNK_SIZE) as u64,
    ));

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime");

    let chunk_data: Vec<u8> = vec![b'A'; STREAMING_CHUNK_SIZE];

    group.bench_function("hyper::server::conn::http1 streaming (duplex)", |bencher| {
        let chunk_data = chunk_data.clone();
        let mut quartet = HdrQuartet::new();
        bencher.iter_custom(|iterations| {
            let chunk_data = chunk_data.clone();
            let mut total = Duration::ZERO;
            for idx in 0..iterations {
                let chunk_data = chunk_data.clone();
                let start = std::time::Instant::now();
                runtime.block_on(async move {
                    let (mut client, server) = tokio::io::duplex(128 * 1024);
                    let chunk_data_for_server = chunk_data.clone();
                    tokio::spawn(async move {
                        let io = TokioIo::new(server);
                        let _ = http1::Builder::new()
                            .serve_connection(
                                io,
                                pipe_fn(move |_req: hyper::Request<hyper::body::Incoming>| {
                                    let chunk_data = chunk_data_for_server.clone();
                                    async move {
                                        let frames: Vec<Result<Frame<Bytes>, Infallible>> = (0
                                            ..STREAMING_CHUNK_COUNT)
                                            .map(|_| {
                                                Ok(Frame::data(Bytes::copy_from_slice(&chunk_data)))
                                            })
                                            .collect();
                                        let body = StreamBody::new(futures::stream::iter(frames));
                                        Ok::<_, Infallible>(
                                            hyper::Response::builder()
                                                .status(200)
                                                .body(body)
                                                .expect("response"),
                                        )
                                    }
                                }),
                            )
                            .await;
                    });
                    let request = build_chunked_request(
                        &(0..STREAMING_CHUNK_COUNT)
                            .map(|_| chunk_data.as_slice())
                            .collect::<Vec<_>>(),
                    );
                    client.write_all(&request).await.expect("write request");
                    let mut buf = vec![0_u8; (STREAMING_CHUNK_COUNT * STREAMING_CHUNK_SIZE) + 4096];
                    let mut received = 0;
                    while received < STREAMING_CHUNK_COUNT * STREAMING_CHUNK_SIZE {
                        let n = client.read(&mut buf).await.expect("read");
                        if n == 0 {
                            break;
                        }
                        received += n;
                    }
                    std::hint::black_box(received);
                    drop(client);
                });
                let elapsed = start.elapsed();
                quartet.record(idx, elapsed.as_nanos() as u64);
                total += elapsed;
            }
            quartet.finalize(iterations);
            quartet.report("hyper::http1 streaming");
            total
        });
    });

    group.bench_function("proxima::Connection streaming (duplex)", |bencher| {
        let chunk_data = chunk_data.clone();
        let mut quartet = HdrQuartet::new();
        let policy = AutoStreamPolicy {
            content_length_threshold: 0,
            stream_chunked: true,
        };
        bencher.iter_custom(|iterations| {
            let chunk_data = chunk_data.clone();
            let mut total = Duration::ZERO;
            for idx in 0..iterations {
                let chunk_data = chunk_data.clone();
                let start = std::time::Instant::now();
                runtime.block_on(async move {
                    let (mut client, mut server) = tokio::io::duplex(128 * 1024);
                    let chunk_data_for_server = chunk_data.clone();
                    tokio::spawn(async move {
                        let chunk_data = chunk_data_for_server;
                        let response_headers =
                            vec![("transfer-encoding".to_string(), "chunked".to_string())];
                        let mut connection = Connection::new();
                        connection.set_auto_stream_policy(Some(policy));
                        let mut read_buf = [0_u8; 8192];
                        let mut out =
                            Vec::with_capacity(STREAMING_CHUNK_COUNT * STREAMING_CHUNK_SIZE + 512);
                        loop {
                            let n = server.read(&mut read_buf).await.expect("server read");
                            if n == 0 {
                                break;
                            }
                            connection.feed_bytes(&read_buf[..n]);
                            let mut head_seen = false;
                            loop {
                                match connection.poll().expect("poll") {
                                    Poll::HeadReady => {
                                        head_seen = true;
                                    }
                                    Poll::BodyChunk => {
                                        let chunk = connection.take_body_chunk().expect("chunk");
                                        std::hint::black_box(chunk.len());
                                    }
                                    Poll::BodyEnd => break,
                                    Poll::RequestReady => {
                                        head_seen = true;
                                        break;
                                    }
                                    Poll::NeedInput => break,
                                    Poll::Close => return,
                                    _ => {}
                                }
                            }
                            if head_seen {
                                out.clear();
                                let writer = connection.begin_response(
                                    200,
                                    "OK",
                                    &response_headers,
                                    BodyFraming::Chunked,
                                    &mut out,
                                );
                                for _ in 0..STREAMING_CHUNK_COUNT {
                                    writer.write_chunk(&chunk_data, &mut out);
                                }
                                writer.end_response(&mut out);
                                server.write_all(&out).await.expect("server write");
                                connection.reset_for_next_request();
                                return;
                            }
                        }
                    });
                    let request = build_chunked_request(
                        &(0..STREAMING_CHUNK_COUNT)
                            .map(|_| chunk_data.as_slice())
                            .collect::<Vec<_>>(),
                    );
                    client.write_all(&request).await.expect("write request");
                    let mut buf = vec![0_u8; (STREAMING_CHUNK_COUNT * STREAMING_CHUNK_SIZE) + 4096];
                    let mut received = 0;
                    while received < STREAMING_CHUNK_COUNT * STREAMING_CHUNK_SIZE {
                        let n = client.read(&mut buf).await.expect("read");
                        if n == 0 {
                            break;
                        }
                        received += n;
                    }
                    std::hint::black_box(received);
                    drop(client);
                });
                let elapsed = start.elapsed();
                quartet.record(idx, elapsed.as_nanos() as u64);
                total += elapsed;
            }
            quartet.finalize(iterations);
            quartet.report("proxima::Connection streaming");
            total
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    small_body_cl_256,
    medium_body_cl_64kib,
    chunked_16x4kib,
    h1_streaming_vs_hyper
);
criterion_main!(benches);
