#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! 80%-case tail-latency sweep on the INCUMBENTS' design point: proxima's
//! `Connection` state machine vs hyper's `http1::Builder` vs Cloudflare
//! Pingora's `HttpSession` (v1), all driven by the same keep-alive h1
//! clients over loopback at concurrency 1 / 10 / 100. HDR histogram →
//! p50/p90/p99/p999/max.
//!
//! Keep-alive (not connect-per-request) because at concurrency=100 over a
//! multi-second window, connect-per-request exhausts ephemeral ports /
//! TIME_WAIT on loopback. N persistent connections each issuing sequential
//! requests IS hyper's and pingora's design point — the realistic 80% h1
//! load — and mirrors the warm-client model of `h2_tail_vs_incumbents`.
//!
//! `cargo bench --bench h1_tail_vs_incumbents` (own loop, not criterion —
//! criterion reports mean/median, not tail percentiles).

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use hdrhistogram::Histogram;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn as pipe_fn;
use hyper_util::rt::TokioIo;
use proxima::h1_body::BodyFraming;
use proxima::h1_connection::{Connection, Poll};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Keep-alive GET — no `Connection: close`, so the socket is reused.
const KEEPALIVE_GET: &[u8] = b"GET /v1/items HTTP/1.1\r\n\
Host: api.example.com\r\n\
User-Agent: proxima-bench/0.1\r\n\
Accept: application/json\r\n\
Accept-Encoding: gzip\r\n\
\r\n";

const RESPONSE_BODY: &[u8] = b"ok";
/// How long each (impl × concurrency) combo runs.
const SAMPLE_WINDOW: Duration = Duration::from_secs(3);

fn fresh_histogram() -> Histogram<u64> {
    Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).expect("hdr bounds")
}

async fn spawn_native() -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let response_headers: Arc<Vec<(String, String)>> = Arc::new(vec![
        ("content-type".to_string(), "text/plain".to_string()),
        (
            "content-length".to_string(),
            RESPONSE_BODY.len().to_string(),
        ),
    ]);
    tokio::spawn(async move {
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
                'conn: while let Ok(n) = socket.read(&mut read_buf).await {
                    if n == 0 {
                        break;
                    }
                    connection.feed_bytes(&read_buf[..n]);
                    loop {
                        match connection.poll() {
                            Ok(Poll::NeedInput) => continue 'conn,
                            Ok(Poll::RequestReady) => {
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
                                if socket.write_all(&out).await.is_err() {
                                    break 'conn;
                                }
                                if !connection.keep_alive() {
                                    break 'conn;
                                }
                                connection.reset_for_next_request();
                            }
                            _ => break 'conn,
                        }
                    }
                }
            });
        }
    });
    addr
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

async fn spawn_hyper() -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
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
    addr
}

async fn spawn_pingora() -> std::net::SocketAddr {
    use pingora_core::protocols::http::v1::server::HttpSession;
    use pingora_http::ResponseHeader;

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
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
                loop {
                    match session.read_request().await {
                        Ok(Some(_)) => {}
                        _ => break,
                    }
                    let mut header = match ResponseHeader::build(200, None) {
                        Ok(value) => value,
                        Err(_) => break,
                    };
                    let _ = header.append_header("content-type", "text/plain");
                    let _ = header.append_header("content-length", RESPONSE_BODY.len().to_string());
                    if session
                        .write_response_header(Box::new(header))
                        .await
                        .is_err()
                    {
                        break;
                    }
                    if session
                        .write_body(&Bytes::from_static(RESPONSE_BODY))
                        .await
                        .is_err()
                    {
                        break;
                    }
                    if session.finish_body().await.is_err() {
                        break;
                    }
                }
            });
        }
    });
    addr
}

/// Read exactly one keep-alive response. The content-length=2 body "ok" is
/// the last thing on the wire (no trailing CRLF for content-length framing),
/// and lowercase "ok" never appears in the status line ("OK") or headers, so
/// it's a reliable single-response sentinel when requests aren't pipelined.
async fn read_one_response(stream: &mut TcpStream, scratch: &mut Vec<u8>) {
    scratch.clear();
    let mut chunk = [0_u8; 256];
    loop {
        let n = stream.read(&mut chunk).await.expect("read");
        if n == 0 {
            break;
        }
        scratch.extend_from_slice(&chunk[..n]);
        if scratch.ends_with(RESPONSE_BODY) && scratch.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }
    std::hint::black_box(&scratch[..]);
}

/// `concurrency` keep-alive connections, each issuing sequential requests and
/// recording its own latency; merged into one distribution at the end.
async fn run_workload(addr: std::net::SocketAddr, concurrency: usize) -> Histogram<u64> {
    let deadline = Instant::now() + SAMPLE_WINDOW;
    let mut tasks = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        tasks.push(tokio::spawn(async move {
            let mut stream = TcpStream::connect(addr).await.expect("connect");
            let _ = stream.set_nodelay(true);
            let mut scratch = Vec::with_capacity(256);
            for _ in 0..10 {
                stream.write_all(KEEPALIVE_GET).await.expect("warmup write");
                read_one_response(&mut stream, &mut scratch).await;
            }
            let mut histogram = fresh_histogram();
            while Instant::now() < deadline {
                let start = Instant::now();
                stream.write_all(KEEPALIVE_GET).await.expect("write");
                read_one_response(&mut stream, &mut scratch).await;
                let _ = histogram.record((start.elapsed().as_nanos() as u64).max(1));
            }
            histogram
        }));
    }
    let mut combined = fresh_histogram();
    for task in tasks {
        combined.add(task.await.expect("join task")).expect("merge");
    }
    combined
}

fn format_ns(value: u64) -> String {
    if value < 1_000 {
        format!("{value} ns")
    } else if value < 1_000_000 {
        format!("{:.2} us", value as f64 / 1_000.0)
    } else if value < 1_000_000_000 {
        format!("{:.2} ms", value as f64 / 1_000_000.0)
    } else {
        format!("{:.2} s", value as f64 / 1_000_000_000.0)
    }
}

fn report(label: &str, histogram: &Histogram<u64>) {
    println!(
        "{label:<18} count={:<8} mean={:<10} p50={:<10} p90={:<10} p99={:<10} p999={:<10} max={}",
        histogram.len(),
        format_ns(histogram.mean() as u64),
        format_ns(histogram.value_at_quantile(0.50)),
        format_ns(histogram.value_at_quantile(0.90)),
        format_ns(histogram.value_at_quantile(0.99)),
        format_ns(histogram.value_at_quantile(0.999)),
        format_ns(histogram.max()),
    );
}

fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime");

    println!(
        "h1 tail sweep — proxima vs hyper vs pingora, {}s/scenario, keep-alive conns, minimal GET",
        SAMPLE_WINDOW.as_secs(),
    );
    println!();

    let (native_addr, hyper_addr, pingora_addr) = runtime.block_on(async {
        (
            spawn_native().await,
            spawn_hyper().await,
            spawn_pingora().await,
        )
    });

    for concurrency in [1, 10, 100] {
        let native = runtime.block_on(run_workload(native_addr, concurrency));
        let hyper = runtime.block_on(run_workload(hyper_addr, concurrency));
        let pingora = runtime.block_on(run_workload(pingora_addr, concurrency));
        println!("concurrency={concurrency}:");
        report("  proxima", &native);
        report("  hyper", &hyper);
        report("  pingora", &pingora);
        println!();
    }
}
