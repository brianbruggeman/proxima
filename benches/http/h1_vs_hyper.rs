#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! End-to-end head-to-head: hyper's HTTP/1 server pipeline vs our
//! `Connection` orchestration. Both fed the SAME bytes via an
//! in-memory transport, both producing a fixed response. The
//! measurement is "feed N bytes, get response bytes" — no socket
//! syscalls, no TCP — so we're comparing the pure software pipeline.
//!
//! Why this matters: httparse (which our Connection now uses too) is
//! 2-5% of an end-to-end request. The connection state machine,
//! request/response struct allocation, and async-task scheduling are
//! the rest. This bench measures *those*.

use std::convert::Infallible;
use std::time::Duration;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn as pipe_fn;
use hyper_util::rt::TokioIo;
use proxima::h1_body::BodyFraming;
use proxima::h1_connection::{Connection, Poll};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

// Connection: close so hyper finishes the connection after one
// request — the bench is one-shot per iteration; keep-alive would
// leave the socket open and leak server-side state.
const SMALL_GET: &[u8] = b"GET /v1/items HTTP/1.1\r\n\
Host: api.example.com\r\n\
User-Agent: proxima-bench/0.1\r\n\
Accept: application/json\r\n\
Accept-Encoding: gzip\r\n\
Connection: close\r\n\
\r\n";

const RESPONSE_BODY: &[u8] = b"ok";

fn build_runtime() -> tokio::runtime::Runtime {
    // Single-threaded current-thread runtime mirrors our per-core
    // architectural model.
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
}

async fn hyper_server_echo(
    _request: hyper::Request<Incoming>,
) -> Result<hyper::Response<Full<Bytes>>, Infallible> {
    Ok(hyper::Response::builder()
        .status(200)
        .header("content-type", "text/plain")
        .header("content-length", RESPONSE_BODY.len().to_string())
        .body(Full::new(Bytes::from_static(RESPONSE_BODY)))
        .expect("response builds"))
}

fn hyper_end_to_end(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h1_end_to_end");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));

    let runtime = build_runtime();

    group.bench_function("hyper::server::conn::http1 (duplex transport)", |bencher| {
        bencher.to_async(&runtime).iter(|| async {
            // Fresh duplex pair per iteration so the connection state
            // doesn't accumulate. duplex_size = 4 KB matches a typical
            // socket buffer.
            let (mut client, server) = tokio::io::duplex(4 * 1024);
            let server_handle = tokio::spawn(async move {
                let io = TokioIo::new(server);
                let _ = http1::Builder::new()
                    .serve_connection(io, pipe_fn(hyper_server_echo))
                    .await;
            });
            client
                .write_all(std::hint::black_box(SMALL_GET))
                .await
                .expect("write request");
            // Read response. Hyper's response is small (~70 bytes
            // including body); a 256-byte buffer is plenty.
            let mut response = [0_u8; 256];
            let n = client.read(&mut response).await.expect("read response");
            std::hint::black_box(&response[..n]);
            drop(client);
            let _ = server_handle.await;
        });
    });

    let response_headers = vec![
        ("content-type".to_string(), "text/plain".to_string()),
        (
            "content-length".to_string(),
            RESPONSE_BODY.len().to_string(),
        ),
    ];

    // In-process Connection: state-machine cost only, no async / no
    // socket. Lower bound — measures what the connection layer adds
    // on top of httparse + body decoder.
    group.bench_function("proxima::Connection (in-process)", |bencher| {
        let response_headers = response_headers.clone();
        bencher.to_async(&runtime).iter(|| {
            let response_headers = response_headers.clone();
            async move {
                let mut connection = Connection::new();
                let mut out = Vec::with_capacity(256);
                connection.feed_bytes(std::hint::black_box(SMALL_GET));
                match connection.poll().expect("poll") {
                    Poll::RequestReady => {}
                    Poll::NeedInput
                    | Poll::Close
                    | Poll::HeadReady
                    | Poll::BodyChunk
                    | Poll::BodyEnd
                    | Poll::Expect100Continue => panic!("unexpected poll outcome"),
                }
                std::hint::black_box(connection.path());
                std::hint::black_box(connection.body());
                let writer = connection.begin_response(
                    200,
                    "OK",
                    std::hint::black_box(&response_headers),
                    BodyFraming::ContentLength(RESPONSE_BODY.len() as u64),
                    &mut out,
                );
                writer.write_chunk(RESPONSE_BODY, &mut out);
                writer.end_response(&mut out);
                std::hint::black_box(out.len());
            }
        });
    });

    // Fair comparison: Connection wired through the SAME duplex +
    // tokio::spawn pattern hyper pays in this bench. The server task
    // drives Connection over the server-side half of the duplex; the
    // client writes the request and reads the response. Same async
    // overhead, different connection-layer machinery.
    group.bench_function(
        "proxima::Connection (duplex transport, async server task)",
        |bencher| {
            let response_headers = response_headers.clone();
            bencher.to_async(&runtime).iter(|| {
                let response_headers = response_headers.clone();
                async move {
                    let (mut client, mut server) = tokio::io::duplex(4 * 1024);
                    #[allow(unused_variables)]
                    let server_handle = tokio::spawn(async move {
                        let mut connection = Connection::new();
                        let mut out = Vec::with_capacity(256);
                        let mut read_buf = [0_u8; 1024];
                        loop {
                            let n = server.read(&mut read_buf).await.expect("server read");
                            if n == 0 {
                                break;
                            }
                            connection.feed_bytes(&read_buf[..n]);
                            match connection.poll().expect("poll") {
                                Poll::NeedInput => continue,
                                Poll::Close => break,
                                Poll::RequestReady => {
                                    out.clear();
                                    let writer = connection.begin_response(
                                        200,
                                        "OK",
                                        &response_headers,
                                        BodyFraming::ContentLength(RESPONSE_BODY.len() as u64),
                                        &mut out,
                                    );
                                    writer.write_chunk(RESPONSE_BODY, &mut out);
                                    writer.end_response(&mut out);
                                    server.write_all(&out).await.expect("server write");
                                    // Connection: close request — exit loop.
                                    break;
                                }
                                Poll::HeadReady
                                | Poll::BodyChunk
                                | Poll::BodyEnd
                                | Poll::Expect100Continue => {
                                    // Variants not used in this bench:
                                    // - HeadReady/BodyChunk/BodyEnd require an
                                    //   auto-stream policy we don't set.
                                    // - Expect100Continue requires an Expect
                                    //   header which this bench's SMALL_GET
                                    //   doesn't carry.
                                    break;
                                }
                            }
                        }
                    });
                    client
                        .write_all(std::hint::black_box(SMALL_GET))
                        .await
                        .expect("write request");
                    let mut response = [0_u8; 256];
                    let n = client.read(&mut response).await.expect("read response");
                    std::hint::black_box(&response[..n]);
                    drop(client);
                    let _ = server_handle.await;
                }
            });
        },
    );

    group.finish();
}

criterion_group!(benches, hyper_end_to_end);
criterion_main!(benches);
