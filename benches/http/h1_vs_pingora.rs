#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! Head-to-head: proxima's `Connection` state machine vs Cloudflare's
//! Pingora vs hyper, all driving a small GET → 200 OK over an
//! actual loopback TCP socket. Same transport for all three so the
//! delta is connection-layer machinery, not transport.
//!
//! Why loopback (not duplex like `h1_vs_hyper`): pingora's `Stream`
//! type wraps `tokio::net::TcpStream` (via `From<TcpStream>` on
//! `l4::stream::Stream`), not `DuplexStream`. Going through a real
//! loopback socket for all three is the fairest apples-to-apples
//! comparison and matches a deployed scenario.
//!
//! Each variant binds its listener ONCE for the bench group (avoids
//! port exhaustion under criterion's iteration count) and the
//! accept-loop task lives for the duration of the group.

use std::sync::Arc;
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
use tokio::net::{TcpListener, TcpStream};

const SMALL_GET: &[u8] = b"GET /v1/items HTTP/1.1\r\n\
Host: api.example.com\r\n\
User-Agent: proxima-bench/0.1\r\n\
Accept: application/json\r\n\
Accept-Encoding: gzip\r\n\
Connection: close\r\n\
\r\n";

const RESPONSE_BODY: &[u8] = b"ok";

fn build_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime")
}

async fn hyper_handler(
    _request: hyper::Request<Incoming>,
) -> Result<hyper::Response<Full<Bytes>>, std::convert::Infallible> {
    Ok(hyper::Response::builder()
        .status(200)
        .header("content-type", "text/plain")
        .header("content-length", RESPONSE_BODY.len().to_string())
        .body(Full::new(Bytes::from_static(RESPONSE_BODY)))
        .expect("response builds"))
}

/// Issue one client roundtrip against the given address.
async fn client_roundtrip(addr: std::net::SocketAddr) {
    let mut client = TcpStream::connect(addr).await.expect("connect");
    let _ = client.set_nodelay(true);
    client
        .write_all(std::hint::black_box(SMALL_GET))
        .await
        .expect("write");
    let mut response = [0_u8; 256];
    let mut total = 0;
    while let Ok(n) = client.read(&mut response[total..]).await {
        if n == 0 {
            break;
        }
        total += n;
        if total == response.len() {
            break;
        }
    }
    std::hint::black_box(&response[..total]);
}

/// Spawn an accept loop driven by `handle`. Returns the bound addr
/// and a shutdown notifier (drop = stop accepting new conns).
fn start_proxima_server(
    runtime: &tokio::runtime::Runtime,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let (addr, listener) = runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, listener)
    });
    let response_headers: Arc<Vec<(String, String)>> = Arc::new(vec![
        ("content-type".to_string(), "text/plain".to_string()),
        (
            "content-length".to_string(),
            RESPONSE_BODY.len().to_string(),
        ),
    ]);
    let join = runtime.spawn(async move {
        loop {
            let (mut socket, _) = match listener.accept().await {
                Ok(value) => value,
                Err(_) => break,
            };
            let _ = socket.set_nodelay(true);
            let response_headers = response_headers.clone();
            tokio::spawn(async move {
                let mut connection = Connection::new();
                let mut out = Vec::with_capacity(256);
                let mut read_buf = [0_u8; 1024];
                while let Ok(n) = socket.read(&mut read_buf).await {
                    if n == 0 {
                        break;
                    }
                    connection.feed_bytes(&read_buf[..n]);
                    match connection.poll().expect("poll") {
                        Poll::NeedInput => continue,
                        Poll::RequestReady => {
                            let writer = connection.begin_response(
                                200,
                                "OK",
                                &response_headers,
                                BodyFraming::ContentLength(RESPONSE_BODY.len() as u64),
                                &mut out,
                            );
                            writer.write_chunk(RESPONSE_BODY, &mut out);
                            writer.end_response(&mut out);
                            let _ = socket.write_all(&out).await;
                            break;
                        }
                        _ => break,
                    }
                }
            });
        }
    });
    (addr, join)
}

fn start_hyper_server(
    runtime: &tokio::runtime::Runtime,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let (addr, listener) = runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, listener)
    });
    let join = runtime.spawn(async move {
        loop {
            let (socket, _) = match listener.accept().await {
                Ok(value) => value,
                Err(_) => break,
            };
            let _ = socket.set_nodelay(true);
            tokio::spawn(async move {
                let io = TokioIo::new(socket);
                let _ = http1::Builder::new()
                    .serve_connection(io, pipe_fn(hyper_handler))
                    .await;
            });
        }
    });
    (addr, join)
}

fn start_pingora_server(
    runtime: &tokio::runtime::Runtime,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    use pingora_core::protocols::http::v1::server::HttpSession;
    use pingora_http::ResponseHeader;

    let (addr, listener) = runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, listener)
    });
    let join = runtime.spawn(async move {
        loop {
            let (socket, _) = match listener.accept().await {
                Ok(value) => value,
                Err(_) => break,
            };
            let _ = socket.set_nodelay(true);
            tokio::spawn(async move {
                let l4_stream: pingora_core::protocols::l4::stream::Stream = socket.into();
                let pingora_stream: pingora_core::protocols::Stream = Box::new(l4_stream);
                let mut session = HttpSession::new(pingora_stream);
                if session.read_request().await.is_err() {
                    return;
                }
                let mut header = match ResponseHeader::build(200, None) {
                    Ok(value) => value,
                    Err(_) => return,
                };
                let _ = header.append_header("content-type", "text/plain");
                let _ = header.append_header("content-length", RESPONSE_BODY.len().to_string());
                if session
                    .write_response_header(Box::new(header))
                    .await
                    .is_err()
                {
                    return;
                }
                let _ = session.write_body(&Bytes::from_static(RESPONSE_BODY)).await;
                let _ = session.finish_body().await;
            });
        }
    });
    (addr, join)
}

fn h1_vs_pingora(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("h1_vs_pingora_loopback");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(1));

    let runtime = build_runtime();

    let (proxima_addr, _proxima_join) = start_proxima_server(&runtime);
    group.bench_function("proxima::Connection (loopback)", |bencher| {
        bencher
            .to_async(&runtime)
            .iter(|| client_roundtrip(proxima_addr));
    });

    let (hyper_addr, _hyper_join) = start_hyper_server(&runtime);
    group.bench_function("hyper::server::conn::http1 (loopback)", |bencher| {
        bencher
            .to_async(&runtime)
            .iter(|| client_roundtrip(hyper_addr));
    });

    let (pingora_addr, _pingora_join) = start_pingora_server(&runtime);
    group.bench_function("pingora::HttpSession (loopback)", |bencher| {
        bencher
            .to_async(&runtime)
            .iter(|| client_roundtrip(pingora_addr));
    });

    group.finish();
}

criterion_group!(benches, h1_vs_pingora);
criterion_main!(benches);
