#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(all(feature = "http2", feature = "tls"))]

//! 80%-case tail-latency sweep on the INCUMBENTS' design point: proxima
//! native h2 vs hyper's `http2::Builder` vs Cloudflare Pingora's
//! `HttpSession`, all driven by the same warm h2 client over loopback at
//! concurrency 1 / 10 / 100. HDR histogram → p50/p90/p99/p999/max.
//!
//! This is the home-turf comparison the single-shot warm benches lack:
//! hyper and pingora are built for many concurrent streams under load, so
//! tail latency at concurrency=100 is their real 80% case, not a single
//! warm round-trip. Mirrors `h2_native_vs_h2_crate_tail.rs` (same harness)
//! but swaps the incumbent arms to hyper + pingora.
//!
//! `cargo bench --bench h2_tail_vs_incumbents` (not criterion — own loop,
//! because criterion reports mean/median, not tail percentiles).

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::{Duration, Instant};

use bytes::Bytes;
use hdrhistogram::Histogram;
use http_body_util::Full;
use hyper::server::conn::http2;
use hyper::service::service_fn as pipe_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use proxima::error::ProximaError;
use proxima::h2::serve_h2_connection;
use proxima::listeners::http::QuiesceResponse;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima_primitives::pipe::SendPipe;
use tokio::net::{TcpListener, TcpStream};
use tokio_util::compat::TokioAsyncReadCompatExt;

const RESPONSE_BODY: &[u8] = b"ok";
/// How long each (impl × concurrency) combo runs.
const SAMPLE_WINDOW: Duration = Duration::from_secs(3);

fn fresh_histogram() -> Histogram<u64> {
    Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).expect("hdr bounds")
}

struct ConstantOk;

impl SendPipe for ConstantOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok(Bytes::from_static(RESPONSE_BODY))) }
    }
}


async fn spawn_native(dispatch: PipeHandle) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        loop {
            let (socket, _) = match listener.accept().await {
                Ok(value) => value,
                Err(_) => break,
            };
            let _ = socket.set_nodelay(true);
            let dispatch = dispatch.clone();
            tokio::spawn(async move {
                let in_flight = Arc::new(AtomicU64::new(0));
                let quiesce = Arc::new(QuiesceResponse {
                    status: 503,
                    retry_after: "1".into(),
                });
                let _ =
                    serve_h2_connection(socket.compat(), dispatch, in_flight, quiesce, None).await;
            });
        }
    });
    addr
}

async fn hyper_handler(
    _request: hyper::Request<hyper::body::Incoming>,
) -> Result<hyper::Response<Full<Bytes>>, Infallible> {
    Ok(hyper::Response::builder()
        .status(200)
        .header("content-type", "text/plain")
        .body(Full::new(Bytes::from_static(RESPONSE_BODY)))
        .expect("response"))
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
                let _ = http2::Builder::new(TokioExecutor::new())
                    .serve_connection(io, pipe_fn(hyper_handler))
                    .await;
            });
        }
    });
    addr
}

async fn spawn_pingora() -> std::net::SocketAddr {
    use pingora_core::protocols::Digest;
    use pingora_core::protocols::http::v2::server::{HttpSession, handshake as h2c_handshake};
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
                let mut h2_conn = match h2c_handshake(pingora_stream, None).await {
                    Ok(conn) => conn,
                    Err(_) => return,
                };
                let digest = Arc::new(Digest::default());
                loop {
                    let session =
                        match HttpSession::from_h2_conn(&mut h2_conn, digest.clone()).await {
                            Ok(Some(session)) => session,
                            Ok(None) | Err(_) => break,
                        };
                    tokio::spawn(async move {
                        let mut session = session;
                        let mut header = match ResponseHeader::build(200, None) {
                            Ok(value) => value,
                            Err(_) => return,
                        };
                        let _ = header.append_header("content-type", "text/plain");
                        if session
                            .write_response_header(Box::new(header), false)
                            .is_err()
                        {
                            return;
                        }
                        let _ = session
                            .write_body(Bytes::from_static(RESPONSE_BODY), true)
                            .await;
                        let _ = session.finish();
                    });
                }
            });
        }
    });
    addr
}

async fn warm_client(addr: std::net::SocketAddr) -> h2::client::SendRequest<Bytes> {
    let socket = TcpStream::connect(addr).await.expect("connect");
    let _ = socket.set_nodelay(true);
    let (h2_client, h2_conn) = h2::client::handshake(socket).await.expect("handshake");
    tokio::spawn(async move {
        let _ = h2_conn.await;
    });
    h2_client
}

async fn one_request(client: &mut h2::client::SendRequest<Bytes>) {
    let request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/")
        .body(())
        .expect("request");
    let (response_future, _) = client.send_request(request, true).expect("send_request");
    let response = response_future.await.expect("response");
    let mut body = response.into_body();
    while let Some(chunk) = body.data().await {
        let chunk = chunk.expect("chunk");
        let len = chunk.len();
        body.flow_control()
            .release_capacity(len)
            .expect("flow control");
        std::hint::black_box(chunk);
    }
}

/// `concurrency` parallel request loops on one warm client, each recording
/// its own latency; merged into one distribution at the end.
async fn run_workload(addr: std::net::SocketAddr, concurrency: usize) -> Histogram<u64> {
    let client = warm_client(addr).await;
    {
        let mut warmup = client.clone();
        for _ in 0..10 {
            one_request(&mut warmup).await;
        }
    }
    let deadline = Instant::now() + SAMPLE_WINDOW;
    let mut tasks = Vec::with_capacity(concurrency);
    for _ in 0..concurrency {
        let mut client = client.clone();
        tasks.push(tokio::spawn(async move {
            let mut histogram = fresh_histogram();
            while Instant::now() < deadline {
                let start = Instant::now();
                one_request(&mut client).await;
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
        "h2 tail sweep — proxima_native vs hyper vs pingora, {}s/scenario, warm client, minimal GET",
        SAMPLE_WINDOW.as_secs(),
    );
    println!();

    let (native_addr, hyper_addr, pingora_addr) = runtime.block_on(async {
        (
            spawn_native(into_handle(ConstantOk)).await,
            spawn_hyper().await,
            spawn_pingora().await,
        )
    });

    for concurrency in [1, 10, 100] {
        let native = runtime.block_on(run_workload(native_addr, concurrency));
        let hyper = runtime.block_on(run_workload(hyper_addr, concurrency));
        let pingora = runtime.block_on(run_workload(pingora_addr, concurrency));
        println!("concurrency={concurrency}:");
        report("  proxima_native", &native);
        report("  hyper", &hyper);
        report("  pingora", &pingora);
        println!();
    }
}
