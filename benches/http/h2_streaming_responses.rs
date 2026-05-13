#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(all(feature = "http2", feature = "runtime-tokio"))]

//! Streaming-response head-to-head: proxima native h2 vs hyper vs
//! pingora, all serving a response built by emitting 32 chunks of
//! 2 KiB each (64 KiB total). Exercises the per-chunk pump path
//! that proxima just shipped (FuturesUnordered-driven chunk-pulls
//! consuming Body::from_stream in the connection task).
//!
//! Workload: client opens one warm h2 connection, issues GET, reads
//! every DATA frame back, then issues the next GET. Sequential
//! request-response cycles for the measurement window. 5-run median.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::time::{Duration, Instant};

use bytes::Bytes;
use hdrhistogram::Histogram;

#[path = "../common/hdr_phased.rs"]
mod hdr_phased;
use hdr_phased::HdrQuartet;
use http_body_util::StreamBody;
use hyper::body::Frame;
use hyper::server::conn::http2;
use hyper::service::service_fn as pipe_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use proxima::ResponseStream;
use proxima::error::ProximaError;
use proxima::h2::serve_h2_connection;
use proxima::listeners::http::QuiesceResponse;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::request::{Request, Response};
use proxima_primitives::pipe::SendPipe;
use tokio::net::{TcpListener, TcpStream};
use tokio_util::compat::TokioAsyncReadCompatExt;

const CHUNK_COUNT: usize = 32;
const CHUNK_SIZE: usize = 2 * 1024;
const SAMPLE_WINDOW: Duration = Duration::from_secs(3);

fn fresh_histogram() -> Histogram<u64> {
    Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).expect("hdr bounds")
}

// Pre-allocated, ref-counted chunks. Each call returns a cheap Bytes
// clone instead of a fresh Vec — keeps allocator noise out of the
// h2-framing measurement we actually care about.
static CHUNK_POOL: OnceLock<Vec<Bytes>> = OnceLock::new();

fn chunk_bytes(index: usize) -> Bytes {
    let pool = CHUNK_POOL.get_or_init(|| {
        (0..CHUNK_COUNT)
            .map(|chunk_index| Bytes::from(vec![b'a' + (chunk_index as u8 % 26); CHUNK_SIZE]))
            .collect()
    });
    pool[index].clone()
}

struct ProximaStreamingPipe;

impl SendPipe for ProximaStreamingPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let chunks: Vec<Result<Bytes, ProximaError>> = (0..CHUNK_COUNT)
                .map(|index| Ok(chunk_bytes(index)))
                .collect();
            let stream = futures::stream::iter(chunks);
            Ok(Response::streamed(ResponseStream::new(stream)))
        }
    }
}


async fn hyper_handler(
    _request: hyper::Request<hyper::body::Incoming>,
) -> Result<
    hyper::Response<
        StreamBody<futures::stream::Iter<std::vec::IntoIter<Result<Frame<Bytes>, Infallible>>>>,
    >,
    Infallible,
> {
    let frames: Vec<Result<Frame<Bytes>, Infallible>> = (0..CHUNK_COUNT)
        .map(|index| Ok(Frame::data(chunk_bytes(index))))
        .collect();
    let stream = futures::stream::iter(frames);
    let body = StreamBody::new(stream);
    Ok(hyper::Response::builder()
        .status(200)
        .body(body)
        .expect("response"))
}

fn start_proxima_native_default() -> std::net::SocketAddr {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("server runtime");
    let (addr, listener) = runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, listener)
    });
    let dispatch: PipeHandle = into_handle(ProximaStreamingPipe);
    runtime.spawn(async move {
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
    std::mem::forget(runtime);
    addr
}

fn start_hyper() -> std::net::SocketAddr {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("hyper runtime");
    let (addr, listener) = runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, listener)
    });
    runtime.spawn(async move {
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
    std::mem::forget(runtime);
    addr
}

fn start_pingora() -> std::net::SocketAddr {
    use pingora_core::protocols::Digest;
    use pingora_core::protocols::http::v2::server::{HttpSession, handshake as h2c_handshake};
    use pingora_http::ResponseHeader;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("pingora runtime");
    let (addr, listener) = runtime.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        (addr, listener)
    });
    runtime.spawn(async move {
        loop {
            let (socket, _) = match listener.accept().await {
                Ok(value) => value,
                Err(_) => break,
            };
            let _ = socket.set_nodelay(true);
            tokio::spawn(async move {
                let l4: pingora_core::protocols::l4::stream::Stream = socket.into();
                let stream: pingora_core::protocols::Stream = Box::new(l4);
                let mut h2_conn = match h2c_handshake(stream, None).await {
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
                        let _ = header.append_header("content-type", "application/octet-stream");
                        if session
                            .write_response_header(Box::new(header), false)
                            .is_err()
                        {
                            return;
                        }
                        for index in 0..CHUNK_COUNT {
                            let last = index + 1 == CHUNK_COUNT;
                            if session.write_body(chunk_bytes(index), last).await.is_err() {
                                return;
                            }
                        }
                        let _ = session.finish();
                    });
                }
            });
        }
    });
    std::mem::forget(runtime);
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

async fn one_streaming_request(client: &mut h2::client::SendRequest<Bytes>) -> usize {
    let request = http::Request::builder()
        .method("GET")
        .uri("http://localhost/stream")
        .body(())
        .expect("request");
    let (response_future, _) = client.send_request(request, true).expect("send_request");
    let response = response_future.await.expect("response");
    assert_eq!(response.status(), 200);
    let mut body = response.into_body();
    let mut total = 0;
    while let Some(chunk) = body.data().await {
        let chunk = chunk.expect("chunk");
        let len = chunk.len();
        body.flow_control()
            .release_capacity(len)
            .expect("flow control");
        total += len;
    }
    total
}

async fn run_workload(addr: std::net::SocketAddr) -> (Histogram<u64>, HdrQuartet) {
    let mut client = warm_client(addr).await;
    for _ in 0..10 {
        one_streaming_request(&mut client).await;
    }
    let deadline = Instant::now() + SAMPLE_WINDOW;
    let mut histogram = fresh_histogram();
    let mut quartet = HdrQuartet::new();
    let mut idx: u64 = 0;
    while Instant::now() < deadline {
        let start = Instant::now();
        let received = one_streaming_request(&mut client).await;
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        assert_eq!(received, CHUNK_COUNT * CHUNK_SIZE);
        let _ = histogram.record(elapsed_ns.max(1));
        quartet.record(idx, elapsed_ns);
        idx += 1;
    }
    quartet.finalize(idx);
    (histogram, quartet)
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

fn report(label: &str, histogram: &Histogram<u64>, elapsed: Duration) {
    let count = histogram.len();
    let rps = count as f64 / elapsed.as_secs_f64();
    let mean = histogram.mean() as u64;
    let p50 = histogram.value_at_quantile(0.50);
    let p90 = histogram.value_at_quantile(0.90);
    let p99 = histogram.value_at_quantile(0.99);
    let p999 = histogram.value_at_quantile(0.999);
    let max = histogram.max();
    println!(
        "  {label:<32}  rps={rps:<8.0}  mean={:<10}  p50={:<10}  p90={:<10}  p99={:<10}  p999={:<10}  max={}",
        format_ns(mean),
        format_ns(p50),
        format_ns(p90),
        format_ns(p99),
        format_ns(p999),
        format_ns(max),
    );
}

fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("client runtime");

    println!(
        "streaming-response sweep — {CHUNK_COUNT} chunks of {} KiB each ({} KiB total per request), {}s window",
        CHUNK_SIZE / 1024,
        CHUNK_COUNT * CHUNK_SIZE / 1024,
        SAMPLE_WINDOW.as_secs(),
    );

    let proxima_addr = start_proxima_native_default();
    let hyper_addr = start_hyper();
    let pingora_addr = start_pingora();
    std::thread::sleep(Duration::from_millis(200));

    let (proxima_h, proxima_q) = runtime.block_on(run_workload(proxima_addr));
    let (hyper_h, hyper_q) = runtime.block_on(run_workload(hyper_addr));
    let (pingora_h, pingora_q) = runtime.block_on(run_workload(pingora_addr));

    println!();
    report("proxima_native (default tokio)", &proxima_h, SAMPLE_WINDOW);
    report("hyper (default tokio)", &hyper_h, SAMPLE_WINDOW);
    report("pingora (default tokio)", &pingora_h, SAMPLE_WINDOW);

    println!();
    proxima_q.report("proxima_native (default tokio)");
    hyper_q.report("hyper (default tokio)");
    pingora_q.report("pingora (default tokio)");
}
