#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::type_complexity,
    clippy::default_constructed_unit_structs
)]

//! Compare-bench: prime-native H1ClientUpstream vs hyper SharedHttpClient.
//!
//! Home-turf operation: one HTTP/1.1 GET request→response cycle over a
//! loopback TCP connection.
//!
//! Arms:
//!   - `design-favors: incumbent` — HttpUpstream / SharedHttpClient (hyper +
//!     TokioExecutor) against a minimal loopback echo server.
//!   - `design-favors: neutral`   — H1ClientUpstream<TokioTcpUpstream> against
//!     the same loopback echo server. Both arms run on the same tokio
//!     current-thread reactor — isolates client/codec cost, not runtime cost.
//!   - `codec_only`               — pure CPU: encode_request_head +
//!     parse_response_head + BodyDecoder::feed, no socket. Sweeps three
//!     response body sizes (64 B, 8 KB, 64 KB).
//!
//! Verdict context: the primary consumer (a downstream consumer's corpus download) is
//! network-bound (tens to thousands of ms per file). Per-request local CPU
//! is microseconds and NOT the bill-mover. The prime client's value is
//! removing hyper+tokio from the dependency closure, not raw throughput.

use std::net::{Ipv4Addr, SocketAddr};
use std::time::Duration;

use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use futures::stream;
use http_body_util::BodyExt as _;
use proxima_http::http1::hyper_body::StreamingHyperBody;
use proxima_http::http1::shared_http::SharedHttpClient;
use proxima_protocols::http1_codec::h1_body::BodyDecoder;
use proxima_protocols::http1_codec::h1_client::{
    ResponseStatus, encode_request_head, framing_from_response, parse_response_head,
};
use proxima_net::tokio::tokio_stream_upstream::TokioTcpUpstream;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::TcpListener;

const REQUEST_PATH: &str = "/bench";
const REQUEST_HOST: &str = "bench.local";

fn make_response(body_size: usize) -> Vec<u8> {
    let body = vec![b'x'; body_size];
    let mut resp =
        format!("HTTP/1.1 200 OK\r\ncontent-length: {body_size}\r\nconnection: keep-alive\r\n\r\n")
            .into_bytes();
    resp.extend_from_slice(&body);
    resp
}

fn build_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
}

async fn spawn_echo_server(body_size: usize) -> SocketAddr {
    let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .await
        .expect("bind echo server");
    let addr = listener.local_addr().expect("local_addr");
    let response = make_response(body_size);
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            let response = response.clone();
            tokio::spawn(async move {
                let mut buf = [0_u8; 4096];
                loop {
                    let n = socket.read(&mut buf).await.unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    if buf[..n].windows(4).any(|window| window == b"\r\n\r\n") {
                        socket.write_all(&response).await.ok();
                    }
                }
            });
        }
    });
    addr
}

fn empty_body() -> StreamingHyperBody {
    use proxima_core::ProximaError;
    let empty: proxima_primitives::pipe::body::ChunkStream =
        Box::pin(stream::empty::<Result<Bytes, ProximaError>>());
    StreamingHyperBody::new(empty)
}

fn bench_codec_only(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h1_client_codec_only");
    group.measurement_time(Duration::from_secs(5));

    for &body_size in &[64_usize, 8 * 1024, 64 * 1024] {
        let response_wire = make_response(body_size);
        let label = format!("{body_size}b");

        group.throughput(Throughput::Bytes((response_wire.len()) as u64));
        group.bench_with_input(
            BenchmarkId::new("encode+parse+decode", &label),
            &response_wire,
            |bencher, response| {
                bencher.iter(|| {
                    let mut out = Vec::with_capacity(128);
                    encode_request_head(
                        "GET",
                        REQUEST_PATH,
                        &[("host", REQUEST_HOST), ("connection", "keep-alive")],
                        &mut out,
                    );
                    std::hint::black_box(out.len());

                    let status =
                        parse_response_head(std::hint::black_box(response)).expect("parse");
                    let (head, body_offset) = match status {
                        ResponseStatus::Complete { head, body_offset } => (head, body_offset),
                        ResponseStatus::Partial => panic!("partial"),
                    };
                    let framing = framing_from_response(&head);
                    let mut decoder = BodyDecoder::new(framing);
                    let mut body_len = 0_usize;
                    let (_consumed, _end) = decoder
                        .feed(&response[body_offset..], |chunk| body_len += chunk.len())
                        .expect("feed");
                    std::hint::black_box(body_len);
                });
            },
        );
    }

    group.finish();
}

fn bench_codec_adversarial(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h1_client_codec_adversarial");
    group.measurement_time(Duration::from_secs(5));

    // partial head — parser must accumulate bytes and return Partial fast.
    // the bench measures the fast-reject path: how quickly does the parser
    // determine there is not enough data yet.
    let partial_head = b"HTTP/1.1 200 O".as_slice();
    group.throughput(Throughput::Bytes(partial_head.len() as u64));
    group.bench_function("partial_head", |bencher| {
        bencher.iter(|| {
            let result = parse_response_head(std::hint::black_box(partial_head)).expect("no error");
            assert!(matches!(result, ResponseStatus::Partial));
            std::hint::black_box(result);
        });
    });

    // malformed status line — parser must reject early without panic.
    let malformed_line = b"NOTHTTP 200 OK\r\n\r\n".as_slice();
    group.throughput(Throughput::Bytes(malformed_line.len() as u64));
    group.bench_function("malformed_status_line", |bencher| {
        bencher.iter(|| {
            let result = parse_response_head(std::hint::black_box(malformed_line));
            assert!(result.is_err());
            std::hint::black_box(result.is_err());
        });
    });

    // bad version — should be rejected before allocating headers.
    let bad_version = b"HTTP/2.0 200 OK\r\n\r\n".as_slice();
    group.throughput(Throughput::Bytes(bad_version.len() as u64));
    group.bench_function("bad_http_version", |bencher| {
        bencher.iter(|| {
            let result = parse_response_head(std::hint::black_box(bad_version));
            assert!(result.is_err());
            std::hint::black_box(result.is_err());
        });
    });

    group.finish();
}

fn bench_e2e(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h1_client_e2e");
    group.measurement_time(Duration::from_secs(8));

    let runtime = build_runtime();

    for &body_size in &[64_usize, 8 * 1024, 64 * 1024] {
        let label = format!("{body_size}b");
        group.throughput(Throughput::Elements(1));

        let addr = runtime.block_on(spawn_echo_server(body_size));
        let base_url = format!("http://{addr}{REQUEST_PATH}");

        // arm 1: design-favors incumbent (hyper SharedHttpClient)
        {
            let url = base_url.clone();
            let client = SharedHttpClient::new();
            group.bench_with_input(
                BenchmarkId::new("design-favors:incumbent/hyper_SharedHttpClient", &label),
                &url,
                |bencher, url| {
                    let url = url.clone();
                    let client = client.clone();
                    bencher.to_async(&runtime).iter(|| {
                        let url = url.clone();
                        let client = client.clone();
                        async move {
                            let uri: hyper::Uri = url.parse().expect("uri");
                            let req = hyper::Request::builder()
                                .method("GET")
                                .uri(uri)
                                .header("host", REQUEST_HOST)
                                .header("connection", "keep-alive")
                                .body(empty_body())
                                .expect("build req");
                            let resp = client.request(req).await.expect("hyper request");
                            let status = resp.status().as_u16();
                            let body = resp
                                .into_body()
                                .collect()
                                .await
                                .expect("collect body")
                                .to_bytes();
                            std::hint::black_box((status, body.len()));
                        }
                    });
                },
            );
        }

        // arm 2: design-favors neutral (prime H1ClientUpstream, Collect+All — default)
        {
            use proxima_http::http1::client::H1ClientUpstream;
            use proxima_primitives::pipe::SendPipe;
            use proxima_primitives::pipe::request::Request;

            let upstream = TokioTcpUpstream::new(addr);
            let client = H1ClientUpstream::new(upstream, REQUEST_HOST, "bench");

            group.bench_with_input(
                BenchmarkId::new("composition:collect_all/prime_H1ClientUpstream", &label),
                &addr,
                |bencher, _addr| {
                    bencher.to_async(&runtime).iter(|| {
                        let client = &client;
                        async move {
                            let request = Request::builder()
                                .method("GET")
                                .path(REQUEST_PATH)
                                .build()
                                .expect("request");
                            let resp = client.call(request).await.expect("call");
                            let status = resp.status;
                            // body now streams; collect to bench the full
                            // request/response the buffered path measured.
                            let body = resp.collect_body().await.expect("collect");
                            std::hint::black_box((status, body.len()));
                        }
                    });
                },
            );
        }

        // arm 3: Drain+Framing composition (load-gen profile)
        // same wire, same connection, zero Bytes allocation for headers or body.
        {
            use proxima_http::http1::client::{H1ClientUpstream, ResponseHandling};
            use proxima_primitives::pipe::SendPipe;
            use proxima_primitives::pipe::request::Request;

            let upstream = TokioTcpUpstream::new(addr);
            let client = H1ClientUpstream::new(upstream, REQUEST_HOST, "bench-discard")
                .with_response(ResponseHandling::Discard);

            group.bench_with_input(
                BenchmarkId::new("composition:drain_framing/prime_H1ClientUpstream", &label),
                &addr,
                |bencher, _addr| {
                    bencher.to_async(&runtime).iter(|| {
                        let client = &client;
                        async move {
                            let request = Request::builder()
                                .method("GET")
                                .path(REQUEST_PATH)
                                .build()
                                .expect("request");
                            let resp = client.call(request).await.expect("call");
                            // body is already drained; only status is meaningful.
                            std::hint::black_box(resp.status);
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_codec_only,
    bench_codec_adversarial,
    bench_e2e
);
criterion_main!(benches);
