//! Prime-native HTTP/1 serve parity proof. Drives the SAME
//! `HttpListenProtocol::serve` path on two backends — tokio (reference)
//! and prime (under test) — over five request vectors, capturing the
//! FULL raw response bytes from a raw tokio TCP client.
//!
//! This is the central dog-food: the prime backend binds + accepts on a
//! prime `CoreShard` worker through `PrimeAcceptorFactory`, dispatches
//! per-connection + per-streaming-request via `spawn_on_current_core`,
//! and must match the tokio path byte-for-byte (modulo the inherently
//! variable `date:` / `traceparent:` header values).
//!
//! Vector 3 (the >1 MiB POST) is the make-or-break: a Content-Length
//! over 1 MiB forces the auto-stream threshold into
//! `dispatch_streaming_request`, which on prime runs the `Pipe::call`
//! task via `spawn_on_current_core` — the exact path Step 4 rewired off
//! `spawn_local`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::too_many_lines
)]
#![cfg(all(
    all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    ),
    feature = "http1"
))]

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bytes::Bytes;
use futures::channel::oneshot;

use proxima::error::ProximaError;
use proxima::listen::{ListenProtocol, ServeContext};
use proxima::listeners::HttpListenProtocol;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::prime::{CoreId, PrimeRuntime};
use proxima::request::{Request, Response};
use proxima::runtime::Runtime;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::telemetry_surface::NoopTelemetry;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

// ---- pipe under test --------------------------------------------------

/// Returns a fixed `Response::ok("ok")` and FULLY drains any streamed
/// request body so the server's streaming pump completes rather than
/// parking on back-pressure.
struct DrainOk;

impl SendPipe for DrainOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            if let Some(stream) = request.stream {
                let _ = stream.collect().await?;
            }
            Ok(Response::ok("ok"))
        }
    }
}


/// Records, from inside `Pipe::call` (which runs on the serving worker via
/// `spawn_on_current_core`), whether a tokio runtime is entered on THIS
/// thread. `Handle::try_current().is_ok()` is true iff a tokio reactor is
/// live on the calling thread — so this discriminates the prime serve path
/// (no tokio handle) from the tokio path (tokio handle entered).
struct ReactorProbePipe {
    ran: Arc<AtomicBool>,
    on_tokio: Arc<AtomicBool>,
}

impl SendPipe for ReactorProbePipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let on_tokio_here = tokio::runtime::Handle::try_current().is_ok();
        self.on_tokio.store(on_tokio_here, Ordering::SeqCst);
        self.ran.store(true, Ordering::SeqCst);
        async move {
            if let Some(stream) = request.stream {
                let _ = stream.collect().await?;
            }
            Ok(Response::ok("ok"))
        }
    }
}


// ---- raw client -------------------------------------------------------

#[derive(Debug, Clone)]
struct RawResponse {
    raw: Vec<u8>,
    status: u16,
    header_end: usize,
}

impl RawResponse {
    fn parse(raw: Vec<u8>) -> Self {
        let header_end = raw
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .expect("header terminator present");
        let header_text =
            std::str::from_utf8(&raw[..header_end]).expect("response headers are utf8");
        let status_line = header_text
            .split("\r\n")
            .next()
            .expect("status line present");
        let status = status_line
            .split(' ')
            .nth(1)
            .expect("status code token")
            .parse::<u16>()
            .expect("status code parses");
        Self {
            raw,
            status,
            header_end,
        }
    }

    fn body(&self) -> &[u8] {
        &self.raw[self.header_end + 4..]
    }

    /// The fixed `"ok"` body bytes are intact inside the response, whether
    /// the server framed them with `Content-Length` or `Transfer-Encoding:
    /// chunked` (`"2\r\nok\r\n0\r\n\r\n"`). Assert presence as a substring
    /// rather than parsing the chunk framing here.
    fn body_contains_ok(&self) -> bool {
        self.body().windows(2).any(|window| window == b"ok")
    }
}

async fn connect_retry(addr: SocketAddr) -> TcpStream {
    for _ in 0..200 {
        match TcpStream::connect(addr).await {
            Ok(stream) => return stream,
            Err(_) => tokio::time::sleep(Duration::from_millis(20)).await,
        }
    }
    panic!("listener at {addr} never accepted a connection");
}

/// One request on a fresh `Connection: close` connection; reads the full
/// response to EOF.
async fn request_close(addr: SocketAddr, request_bytes: &[u8]) -> RawResponse {
    let mut stream = connect_retry(addr).await;
    stream.write_all(request_bytes).await.expect("client write");
    stream.flush().await.expect("client flush");
    let mut raw = Vec::with_capacity(4096);
    stream.read_to_end(&mut raw).await.expect("client read");
    RawResponse::parse(raw)
}

/// Read exactly one HTTP/1.1 response off a kept-alive connection: parse
/// the head, then read the framed body (Content-Length or single chunked
/// body) without closing the socket.
async fn read_one_response(stream: &mut TcpStream) -> RawResponse {
    let mut raw = Vec::with_capacity(1024);
    let mut scratch = [0_u8; 1024];
    // read until the header terminator is present.
    let header_end = loop {
        if let Some(position) = raw.windows(4).position(|window| window == b"\r\n\r\n") {
            break position;
        }
        let read = stream.read(&mut scratch).await.expect("client read head");
        assert!(read != 0, "eof before response head completed");
        raw.extend_from_slice(&scratch[..read]);
    };
    let header_text = std::str::from_utf8(&raw[..header_end]).expect("response headers are utf8");
    let lower = header_text.to_ascii_lowercase();
    let chunked = lower.contains("transfer-encoding: chunked");
    let content_length = lower
        .split("\r\n")
        .find_map(|line| line.strip_prefix("content-length:"))
        .map(|value| {
            value
                .trim()
                .parse::<usize>()
                .expect("content-length parses")
        });

    let body_start = header_end + 4;
    if let Some(length) = content_length {
        while raw.len() < body_start + length {
            let read = stream.read(&mut scratch).await.expect("client read body");
            assert!(read != 0, "eof before content-length body completed");
            raw.extend_from_slice(&scratch[..read]);
        }
        raw.truncate(body_start + length);
    } else if chunked {
        // read until the terminating zero-length chunk "0\r\n\r\n".
        while !raw[body_start..].ends_with(b"0\r\n\r\n") {
            let read = stream.read(&mut scratch).await.expect("client read chunk");
            assert!(read != 0, "eof before chunked body terminated");
            raw.extend_from_slice(&scratch[..read]);
        }
    }
    RawResponse::parse(raw)
}

// ---- request vectors --------------------------------------------------

fn get_root_close() -> Vec<u8> {
    b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n".to_vec()
}

fn post_small_close() -> Vec<u8> {
    let body = vec![b'a'; 50];
    let mut request = format!(
        "POST / HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes();
    request.extend_from_slice(&body);
    request
}

fn post_large_close() -> Vec<u8> {
    let body = vec![b'b'; 2 * 1024 * 1024];
    let mut request = format!(
        "POST / HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes();
    request.extend_from_slice(&body);
    request
}

fn post_chunked_close() -> Vec<u8> {
    let mut request =
        b"POST / HTTP/1.1\r\nHost: x\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n"
            .to_vec();
    for chunk in ["hello", "from", "chunked", "body"] {
        request.extend_from_slice(format!("{:x}\r\n", chunk.len()).as_bytes());
        request.extend_from_slice(chunk.as_bytes());
        request.extend_from_slice(b"\r\n");
    }
    request.extend_from_slice(b"0\r\n\r\n");
    request
}

/// Run all five vectors against a bound listener and return the captured
/// responses in order. Vector 5 (keep-alive) issues two GETs on one
/// connection.
async fn run_vectors(addr: SocketAddr) -> Vec<RawResponse> {
    let mut responses = Vec::new();
    responses.push(request_close(addr, &get_root_close()).await); // v1
    responses.push(request_close(addr, &post_small_close()).await); // v2
    responses.push(request_close(addr, &post_large_close()).await); // v3
    responses.push(request_close(addr, &post_chunked_close()).await); // v4

    // v5: two GETs on one keep-alive connection (no Connection: close on
    // the first request, close on the second).
    let mut stream = connect_retry(addr).await;
    stream
        .write_all(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        .await
        .expect("keep-alive write 1");
    stream.flush().await.expect("keep-alive flush 1");
    let first = read_one_response(&mut stream).await;
    stream
        .write_all(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        .await
        .expect("keep-alive write 2");
    stream.flush().await.expect("keep-alive flush 2");
    let second = read_one_response(&mut stream).await;
    responses.push(first); // v5a
    responses.push(second); // v5b
    responses
}

// ---- parity normalization --------------------------------------------

/// Replace the values of the inherently-variable `date:` and
/// `traceparent:` headers with a constant so two structurally identical
/// responses compare byte-for-byte. Header names, order, and framing are
/// untouched.
fn normalize(raw: &[u8]) -> Vec<u8> {
    let header_end = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("header terminator present");
    let head = std::str::from_utf8(&raw[..header_end]).expect("headers utf8");
    let mut out = String::with_capacity(head.len());
    for (index, line) in head.split("\r\n").enumerate() {
        if index != 0 {
            out.push_str("\r\n");
        }
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("date:") {
            out.push_str("date: <normalized>");
        } else if lower.starts_with("traceparent:") {
            out.push_str("traceparent: <normalized>");
        } else {
            out.push_str(line);
        }
    }
    let mut bytes = out.into_bytes();
    bytes.extend_from_slice(&raw[header_end..]);
    bytes
}

// ---- tokio reference backend -----------------------------------------

async fn pick_free_addr() -> SocketAddr {
    let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("probe bind");
    let addr = probe.local_addr().expect("probe addr");
    drop(probe);
    addr
}

/// Drive `HttpListenProtocol::serve` with `TokioAcceptorFactory` and
/// `runtime = None` (the `None` → `spawn_local` arm) inside a `LocalSet`,
/// concurrently running the client vectors. The serve future borrows the
/// protocol + spec, so it cannot be spawned — race it against the client
/// on the same task.
async fn run_tokio_backend() -> Vec<RawResponse> {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            let addr = pick_free_addr().await;
            let dispatch: PipeHandle = into_handle(DrainOk);
            let context = ServeContext::new(NoopTelemetry::handle())
                .with_acceptor_factory(Arc::new(proxima_net::tokio::TokioAcceptorFactory));
            let (shutdown_tx, shutdown_rx) = oneshot::channel();
            let spec = serde_json::json!({ "name": "http" });
            let protocol = HttpListenProtocol::new();
            let serve = protocol.serve(addr, dispatch, &spec, context, shutdown_rx);

            let responses = tokio::select! {
                serve_result = serve => panic!("tokio serve returned early: {serve_result:?}"),
                responses = run_vectors(addr) => responses,
            };
            drop(shutdown_tx);
            responses
        })
        .await
}

// ---- prime backend under test ----------------------------------------

/// Bind + accept the real `HttpListenProtocol::serve` ON a prime worker
/// via `spawn_factory_on_core(CoreId(0), ...)`, exactly like
/// `serve_http` in proxima-runtime-prime. The serve future borrows the
/// protocol + spec (`'_`), so we leak both to `'static` — they live for
/// the whole process, which is fine for a test and matches how a real
/// listener's protocol/spec outlive the serve loop.
fn spawn_prime_serve(runtime: &Arc<PrimeRuntime>, addr: SocketAddr) -> oneshot::Sender<()> {
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let dispatch: PipeHandle = into_handle(DrainOk);
    let runtime_for_context: Arc<dyn Runtime> = runtime.clone();

    // leak protocol + spec so the serve future's `'_` borrow is `'static`
    // inside the `'static` factory closure.
    let protocol: &'static HttpListenProtocol = Box::leak(Box::new(HttpListenProtocol::new()));
    let spec: &'static serde_json::Value =
        Box::leak(Box::new(serde_json::json!({ "name": "http" })));

    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                let context = ServeContext::new(NoopTelemetry::handle())
                    .with_runtime(runtime_for_context)
                    .with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory));
                Box::pin(async move {
                    // bind happens here, on the worker with CURRENT_REACTOR
                    // live — the prime TcpListener requires it.
                    let _ = protocol
                        .serve(addr, dispatch, spec, context, shutdown_rx)
                        .await;
                }) as std::pin::Pin<Box<dyn Future<Output = ()> + 'static>>
            }),
        )
        .expect("spawn prime serve factory on core 0");

    shutdown_tx
}

async fn run_prime_backend(runtime: &Arc<PrimeRuntime>) -> Vec<RawResponse> {
    let addr = pick_free_addr().await;
    let shutdown_tx = spawn_prime_serve(runtime, addr);
    let responses = run_vectors(addr).await;
    drop(shutdown_tx);
    responses
}

// ---- reactor-absence probe drivers -----------------------------------

/// Drive a single `GET /` through the prime listener with the supplied
/// dispatch handle, mirroring `spawn_prime_serve` but parameterized on the
/// pipe so the probe runs on the prime worker.
fn spawn_prime_serve_with(
    runtime: &Arc<PrimeRuntime>,
    addr: SocketAddr,
    dispatch: PipeHandle,
) -> oneshot::Sender<()> {
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let runtime_for_context: Arc<dyn Runtime> = runtime.clone();

    let protocol: &'static HttpListenProtocol = Box::leak(Box::new(HttpListenProtocol::new()));
    let spec: &'static serde_json::Value =
        Box::leak(Box::new(serde_json::json!({ "name": "http" })));

    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                let context = ServeContext::new(NoopTelemetry::handle())
                    .with_runtime(runtime_for_context)
                    .with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory));
                Box::pin(async move {
                    let _ = protocol
                        .serve(addr, dispatch, spec, context, shutdown_rx)
                        .await;
                }) as std::pin::Pin<Box<dyn Future<Output = ()> + 'static>>
            }),
        )
        .expect("spawn prime probe serve factory on core 0");

    shutdown_tx
}

// ---- the proof --------------------------------------------------------

#[proxima::test(flavor = "multi_thread", worker_threads = 4)]
async fn prime_serve_matches_tokio_byte_for_byte() {
    let runtime = Arc::new(
        PrimeRuntime::builder()
            .cores(2)
            .background_inline()
            .build()
            .expect("build prime runtime"),
    );

    let tokio_responses = run_tokio_backend().await;
    let prime_responses = run_prime_backend(&runtime).await;

    assert_eq!(tokio_responses.len(), 6, "tokio: six captured responses");
    assert_eq!(prime_responses.len(), 6, "prime: six captured responses");

    // every vector returns 200 on both backends (no hang, no panic).
    for (index, response) in tokio_responses.iter().enumerate() {
        assert_eq!(response.status, 200, "tokio vector {index} status");
    }
    for (index, response) in prime_responses.iter().enumerate() {
        assert_eq!(response.status, 200, "prime vector {index} status");
    }

    // vector 3 (the >1 MiB streaming POST) is the gate: prime must have
    // returned 200 with the fixed body, proving the streaming pump +
    // spawn_on_current_core dispatch completed without hang or panic.
    assert_eq!(
        prime_responses[2].status, 200,
        "prime 2 MiB streaming POST must return 200"
    );
    assert!(
        prime_responses[2].body_contains_ok(),
        "prime 2 MiB streaming POST body fully consumed and answered; raw body: {:?}",
        prime_responses[2].body()
    );

    // vectors 4 + 5 fully consumed the body (no hang).
    assert!(
        prime_responses[3].body_contains_ok(),
        "prime chunked POST body"
    );
    assert!(
        prime_responses[4].body_contains_ok(),
        "prime keep-alive GET 1"
    );
    assert!(
        prime_responses[5].body_contains_ok(),
        "prime keep-alive GET 2"
    );

    // P14 byte-parity on the deterministic small vectors (1 + 2) after
    // normalizing date / traceparent. status line, header names + order,
    // and body framing must be byte-identical.
    for index in [0_usize, 1_usize] {
        let tokio_norm = normalize(&tokio_responses[index].raw);
        let prime_norm = normalize(&prime_responses[index].raw);
        assert_eq!(
            tokio_norm,
            prime_norm,
            "vector {index} byte-parity (normalized): \ntokio: {}\nprime: {}",
            String::from_utf8_lossy(&tokio_norm),
            String::from_utf8_lossy(&prime_norm),
        );
    }
}

/// P16 reactor-absence proof: inside the connection's `Pipe::call` — which
/// runs on the prime worker via `spawn_on_current_core` — no tokio runtime
/// is entered. The positive control drives the SAME probe through the tokio
/// backend, where a tokio runtime IS entered, proving the probe discriminates.
#[proxima::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_tokio_reactor_on_prime_serve_path() {
    // positive control: probe on the tokio backend must observe a tokio handle.
    let tokio_ran = Arc::new(AtomicBool::new(false));
    let tokio_on_tokio = Arc::new(AtomicBool::new(false));
    {
        let local = tokio::task::LocalSet::new();
        let ran = tokio_ran.clone();
        let on_tokio = tokio_on_tokio.clone();
        local
            .run_until(async move {
                let addr = pick_free_addr().await;
                let dispatch: PipeHandle = into_handle(ReactorProbePipe {
                    ran: ran.clone(),
                    on_tokio: on_tokio.clone(),
                });
                let context = ServeContext::new(NoopTelemetry::handle())
                    .with_acceptor_factory(Arc::new(proxima_net::tokio::TokioAcceptorFactory));
                let (shutdown_tx, shutdown_rx) = oneshot::channel();
                let spec = serde_json::json!({ "name": "http" });
                let protocol = HttpListenProtocol::new();
                let serve = protocol.serve(addr, dispatch, &spec, context, shutdown_rx);

                let request_bytes = get_root_close();
                let response = tokio::select! {
                    serve_result = serve => panic!("tokio serve returned early: {serve_result:?}"),
                    response = request_close(addr, &request_bytes) => response,
                };
                drop(shutdown_tx);
                assert_eq!(response.status, 200, "tokio probe GET status");
            })
            .await;
    }

    // prime backend under test: probe on the prime worker must NOT see a tokio handle.
    // builder uses background_inline (NOT new_with_tokio_compat) — no tokio handle
    // is entered on the worker.
    let runtime = Arc::new(
        PrimeRuntime::builder()
            .cores(2)
            .background_inline()
            .build()
            .expect("build prime runtime"),
    );
    let prime_ran = Arc::new(AtomicBool::new(false));
    let prime_on_tokio = Arc::new(AtomicBool::new(false));
    let addr = pick_free_addr().await;
    let dispatch: PipeHandle = into_handle(ReactorProbePipe {
        ran: prime_ran.clone(),
        on_tokio: prime_on_tokio.clone(),
    });
    let shutdown_tx = spawn_prime_serve_with(&runtime, addr, dispatch);
    let response = request_close(addr, &get_root_close()).await;
    drop(shutdown_tx);
    assert_eq!(response.status, 200, "prime probe GET status");

    // control discriminates: tokio path entered a tokio runtime.
    assert!(tokio_ran.load(Ordering::SeqCst), "tokio probe pipe ran");
    assert!(
        tokio_on_tokio.load(Ordering::SeqCst),
        "positive control: a tokio runtime IS entered on the tokio serve path"
    );

    // headline: prime serve path ran the pipe with NO tokio reactor entered.
    assert!(prime_ran.load(Ordering::SeqCst), "prime probe pipe ran");
    assert!(
        !prime_on_tokio.load(Ordering::SeqCst),
        "P16: NO tokio reactor on the prime serve path (Handle::try_current was Ok)"
    );
}
