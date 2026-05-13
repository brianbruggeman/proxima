//! Integration test proving the `SpreadToPeers` handler dispatch path.
//!
//! Forces spread mode on any host via `PROXIMA_HTTP_HANDLER_SPREAD=1`, then
//! uses the agnostic `Listener::run_with_runtime` + `HttpListenProtocol` +
//! `PrimeAcceptorFactory` to stand up a real HTTP listener. Each assertion
//! targets a failure mode that the single-core accept loop masked.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_lines)]
#![cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool",
    feature = "http1"
))]

use std::future::Future;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Barrier};
use std::thread::ThreadId;
use std::time::Duration;

use bytes::Bytes;
use futures::channel::oneshot;

use proxima::ListenProtocol;
use proxima::error::ProximaError;
use proxima::listeners::HttpListenProtocol;
use proxima::pipe::{PipeHandle, into_handle};
use proxima::prime::{CoreId, PrimeRuntime};
use proxima::request::{Request, Response};
use proxima::runtime::Runtime;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::telemetry_surface::NoopTelemetry;

struct ConstantOk;

impl SendPipe for ConstantOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok("ok")) }
    }
}


/// Deterministic stand-in for a handler that blocks its calling thread
/// synchronously (the `std::thread::sleep` anti-pattern), without timing:
/// a request to `/hold` reports the OS thread it runs on and then parks
/// on a `Barrier` the test releases explicitly; a request to any other
/// path answers immediately. Holding N `/hold` requests open lets the
/// test saturate the runtime's executor-thread budget on command instead
/// of racing a sleep duration against a client read timeout.
struct HoldOrFast {
    thread_report: Sender<ThreadId>,
    release: Arc<Barrier>,
}

impl SendPipe for HoldOrFast {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let thread_report = self.thread_report.clone();
        let release = self.release.clone();
        let is_hold = request.path.as_ref() == b"/hold";
        async move {
            if is_hold {
                thread_report
                    .send(std::thread::current().id())
                    .expect("test receiver still alive");
                release.wait();
                Ok(Response::ok("held"))
            } else {
                Ok(Response::ok("fast"))
            }
        }
    }
}


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


fn build_runtime(cores: usize) -> Arc<PrimeRuntime> {
    Arc::new(
        PrimeRuntime::builder()
            .cores(cores)
            .background_inline()
            .build()
            .expect("build prime runtime"),
    )
}

fn reserve_port() -> SocketAddr {
    let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("probe bind");
    let addr = probe.local_addr().expect("probe addr");
    drop(probe);
    addr
}

fn spawn_spread_listener(
    runtime: &Arc<PrimeRuntime>,
    dispatch: PipeHandle,
) -> (SocketAddr, oneshot::Sender<()>) {
    let cores = runtime.num_cores().max(1);
    let addr = reserve_port();
    let runtime_arc: Arc<dyn Runtime> = runtime.clone();
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    let protocol: &'static HttpListenProtocol = Box::leak(Box::new(HttpListenProtocol::new()));
    let spec: &'static serde_json::Value =
        Box::leak(Box::new(serde_json::json!({ "name": "http" })));

    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                let dispatch = dispatch.clone();
                let runtime_for_context = runtime_arc.clone();
                Box::pin(async move {
                    let context = proxima_listen::ServeContext::new(NoopTelemetry::handle())
                        .with_runtime(runtime_for_context)
                        .with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory))
                        .with_handler_dispatch(proxima_listen::HandlerDispatch::SpreadToPeers {
                            num_cores: cores,
                        });

                    let _ = protocol
                        .serve(addr, dispatch, spec, context, shutdown_rx)
                        .await;
                }) as std::pin::Pin<Box<dyn Future<Output = ()> + 'static>>
            }),
        )
        .expect("spawn spread listener on core 0");

    (addr, shutdown_tx)
}

/// Same wiring as [`spawn_spread_listener`], plus a report of the OS
/// thread id backing the accept core (core 0), captured the moment the
/// listener factory runs there — before it ever touches the socket.
/// `spawn_factory_on_core`'s factory closure executes ON the destination
/// core to build its future, so `std::thread::current().id()` here is
/// exactly the accept lane's executor thread, not a guess.
fn spawn_probed_spread_listener(
    runtime: &Arc<PrimeRuntime>,
    dispatch: PipeHandle,
    accept_thread_report: Sender<ThreadId>,
) -> (SocketAddr, oneshot::Sender<()>) {
    let cores = runtime.num_cores().max(1);
    let addr = reserve_port();
    let runtime_arc: Arc<dyn Runtime> = runtime.clone();
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    let protocol: &'static HttpListenProtocol = Box::leak(Box::new(HttpListenProtocol::new()));
    let spec: &'static serde_json::Value =
        Box::leak(Box::new(serde_json::json!({ "name": "http" })));

    runtime
        .spawn_factory_on_core(
            CoreId(0),
            Box::new(move || {
                accept_thread_report
                    .send(std::thread::current().id())
                    .expect("test receiver still alive");
                let dispatch = dispatch.clone();
                let runtime_for_context = runtime_arc.clone();
                Box::pin(async move {
                    let context = proxima_listen::ServeContext::new(NoopTelemetry::handle())
                        .with_runtime(runtime_for_context)
                        .with_acceptor_factory(Arc::new(proxima_net::prime::PrimeAcceptorFactory))
                        .with_handler_dispatch(proxima_listen::HandlerDispatch::SpreadToPeers {
                            num_cores: cores,
                        });

                    let _ = protocol
                        .serve(addr, dispatch, spec, context, shutdown_rx)
                        .await;
                }) as std::pin::Pin<Box<dyn Future<Output = ()> + 'static>>
            }),
        )
        .expect("spawn spread listener on core 0");

    (addr, shutdown_tx)
}

fn connect_with_retry(addr: SocketAddr) -> Result<std::net::TcpStream, String> {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        match std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(50)) {
            Ok(stream) => return Ok(stream),
            Err(_) if std::time::Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(err) => return Err(format!("connect: {err}")),
        }
    }
}

fn one_request_sync(addr: SocketAddr) -> Result<Vec<u8>, String> {
    let mut stream = connect_with_retry(addr)?;
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|err| format!("set_timeout: {err}"))?;
    stream
        .write_all(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        .map_err(|err| format!("write: {err}"))?;
    stream.flush().map_err(|err| format!("flush: {err}"))?;
    let mut response = Vec::with_capacity(256);
    stream
        .read_to_end(&mut response)
        .map_err(|err| format!("read: {err}"))?;
    Ok(response)
}

/// GETs `path` on a fresh connection. No client-side read timeout: a
/// `/hold` request is expected to block until the test's release
/// `Barrier` fires, so a timeout here would turn a genuine isolation
/// regression (a hang) into a misleading "read timed out" error instead
/// of the deterministic thread-id proof the test is after. The nextest
/// override for this test scopes a slow-timeout so a real regression
/// still fails fast in CI.
fn request_path(addr: SocketAddr, path: &str) -> Result<Vec<u8>, String> {
    let mut stream = connect_with_retry(addr)?;
    let request = format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    stream
        .write_all(request.as_bytes())
        .map_err(|err| format!("write: {err}"))?;
    stream.flush().map_err(|err| format!("flush: {err}"))?;
    let mut response = Vec::with_capacity(256);
    stream
        .read_to_end(&mut response)
        .map_err(|err| format!("read: {err}"))?;
    Ok(response)
}

/// Deterministic proof that `SpreadToPeers` isolates a blocking handler
/// off the runtime's executor threads — no sleeps, no wall-clock
/// timeouts, no `available_parallelism()` race. `cores` is hardcoded at
/// 2 (the CI shape this guards) rather than probed from the host, so the
/// test exercises the same topology everywhere it runs.
///
/// Mechanism, entirely synchronization-based:
///   1. Probe and record the OS thread id backing each of the runtime's
///      2 registered executor cores (core 0 = accept lane, core 1 = the
///      lone peer core `SpreadToPeers` would round-robin onto).
///   2. Fire `hold_count` concurrent `/hold` requests. Each one reports
///      the thread its `Pipe::call` body actually ran on, then parks on
///      a shared `Barrier` the test controls — held open until released,
///      never resolved by a timer.
///   3. Once all `hold_count` reports arrive (proving every held request
///      is in flight and parked, not "probably" but confirmed by the
///      channel receive), fire one `/fast` request. If the blocking
///      requests were still occupying the executor cores instead of a
///      background pool, this would deadlock — the same starvation the
///      old core-round-robin dispatch exhibited when `hold_count` meets
///      or exceeds the peer-core count. A nextest override scopes a
///      slow-timeout so a real regression here fails CI fast instead of
///      hanging it, without the test itself encoding any duration.
///   4. Release the barrier, join the held requests, and confirm they
///      complete too.
///   5. Assert every held request's reported thread id is DISJOINT from
///      the two executor thread ids captured in step 1 — the direct,
///      non-timing structural proof that the blocking body ran on a
///      background-pool thread, not a hot executor worker.
#[test]
fn spread_blocking_handler_runs_off_executor_threads() {
    let cores = 2;
    let hold_count: usize = 2;

    let runtime = build_runtime(cores);

    let (core_probe_tx, core_probe_rx) = std::sync::mpsc::channel::<ThreadId>();
    let core1_probe_tx = core_probe_tx.clone();
    runtime
        .spawn_on_core(
            CoreId(1),
            Box::pin(async move {
                let _ = core1_probe_tx.send(std::thread::current().id());
            }),
        )
        .expect("probe core 1");
    let core1_thread = core_probe_rx.recv().expect("core 1 reported its thread id");

    let (thread_report_tx, thread_report_rx): (Sender<ThreadId>, Receiver<ThreadId>) =
        std::sync::mpsc::channel();
    let release = Arc::new(Barrier::new(hold_count + 1));
    let dispatch = into_handle(HoldOrFast {
        thread_report: thread_report_tx,
        release: release.clone(),
    });
    let (addr, _shutdown) = spawn_probed_spread_listener(&runtime, dispatch, core_probe_tx);
    let core0_thread = core_probe_rx.recv().expect("core 0 reported its thread id");

    let held_workers: Vec<_> = (0..hold_count)
        .map(|_| std::thread::spawn(move || request_path(addr, "/hold")))
        .collect();

    // Blocks until every held request has entered its `Pipe::call` and
    // reported in — a channel receive, not a sleep. Once all
    // `hold_count` reports are in, every held request is confirmed
    // parked on `release`, not merely "probably running by now".
    let held_threads: Vec<ThreadId> = (0..hold_count)
        .map(|_| {
            thread_report_rx
                .recv()
                .expect("held request reported its thread id")
        })
        .collect();

    // The deterministic isolation proof: with all `hold_count` handlers
    // parked, a fresh `/fast` request still completes. Under the old
    // core-round-robin dispatch this would deadlock outright (hold_count
    // >= the single peer core's capacity), not merely run slowly.
    let fast_response = request_path(addr, "/fast").expect("fast request completed while held");
    assert!(
        fast_response.starts_with(b"HTTP/1.1 200"),
        "fast request must succeed while blocking requests are held, got: {:?}",
        String::from_utf8_lossy(&fast_response[..fast_response.len().min(80)])
    );

    release.wait();
    for worker in held_workers {
        let response = worker
            .join()
            .expect("held client thread panicked")
            .expect("held request completed after release");
        assert!(
            response.starts_with(b"HTTP/1.1 200"),
            "held request must eventually succeed, got: {:?}",
            String::from_utf8_lossy(&response[..response.len().min(80)])
        );
    }

    eprintln!(
        "spread-isolation: core0_thread={core0_thread:?} core1_thread={core1_thread:?} \
         held_threads={held_threads:?}"
    );
    for held_thread in held_threads {
        assert_ne!(
            held_thread, core0_thread,
            "a held handler ran on the accept core's executor thread — SpreadToPeers failed to \
             isolate it on the background pool"
        );
        assert_ne!(
            held_thread, core1_thread,
            "a held handler ran on the peer core's executor thread — SpreadToPeers failed to \
             isolate it on the background pool"
        );
    }
}

#[test]
fn spread_h1_response_is_well_formed() {
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(2)
        .max(2);

    let runtime = build_runtime(cores);
    let dispatch = into_handle(ConstantOk);
    let (addr, _shutdown) = spawn_spread_listener(&runtime, dispatch);

    let raw = one_request_sync(addr).expect("request succeeded");
    assert!(
        raw.starts_with(b"HTTP/1.1 200"),
        "expected HTTP/1.1 200, got: {:?}",
        String::from_utf8_lossy(&raw[..raw.len().min(80)])
    );
    let body_start = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("header terminator present")
        + 4;
    let body = &raw[body_start..];
    assert!(
        body.windows(2).any(|w| w == b"ok"),
        "response body must contain 'ok', got: {:?}",
        String::from_utf8_lossy(body)
    );
}

#[test]
fn spread_large_body_round_trips_intact() {
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(2)
        .max(2);

    let runtime = build_runtime(cores);
    let dispatch = into_handle(DrainOk);
    let (addr, _shutdown) = spawn_spread_listener(&runtime, dispatch);

    let body = vec![b'z'; 2 * 1024 * 1024];
    let mut request = format!(
        "POST / HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes();
    request.extend_from_slice(&body);

    let mut stream = connect_with_retry(addr).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("set_timeout");
    stream.write_all(&request).expect("write");
    stream.flush().expect("flush");
    let mut response = Vec::with_capacity(256);
    stream.read_to_end(&mut response).expect("read");

    assert!(
        response.starts_with(b"HTTP/1.1 200"),
        "large body POST must return 200, got: {:?}",
        String::from_utf8_lossy(&response[..response.len().min(80)])
    );
    let body_start = response
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("header terminator")
        + 4;
    let resp_body = &response[body_start..];
    assert!(
        resp_body.windows(2).any(|w| w == b"ok"),
        "large body response must contain 'ok'"
    );
}
