//! Stage 6 — graceful shutdown across cores.
//!
//! Property under test: a Pipe that registers a per-core resource
//! drop hook via `register_per_core_resource` observes the hook
//! firing (a) on the owning OS thread, and (b) only after the
//! in-flight request has completed and the listener has drained.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(feature = "http1")]

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use bytes::Bytes;
use proxima::{
    App, ProximaError, Request, Response, RunConfig, Spec, into_handle,
    shutdown::register_per_core_resource,
};
use proxima_primitives::pipe::SendPipe;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Pipe that — on its first call — registers a per-core drop
/// hook that records (a) the thread that fired, and (b) whether
/// the in-flight request had finished by the time the hook ran.
struct Registers {
    registered: Arc<AtomicBool>,
    serve_thread: Arc<std::sync::Mutex<Option<u64>>>,
    drop_thread: Arc<std::sync::Mutex<Option<u64>>>,
    drop_observed_in_flight: Arc<AtomicU64>,
    in_flight_at_drop_time: Arc<AtomicU64>,
    request_done: Arc<AtomicBool>,
}

impl SendPipe for Registers {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let registered = self.registered.clone();
        let serve_thread = self.serve_thread.clone();
        let drop_thread = self.drop_thread.clone();
        let in_flight_at_drop_time = self.in_flight_at_drop_time.clone();
        let drop_observed_in_flight = self.drop_observed_in_flight.clone();
        let request_done = self.request_done.clone();
        async move {
            // capture the serve-side thread id so the test can assert
            // the drop hook fires on the same OS thread.
            let serve_tid = thread_id_u64();
            if !registered.swap(true, Ordering::SeqCst) {
                if let Ok(mut guard) = serve_thread.lock() {
                    *guard = Some(serve_tid);
                }
                let drop_thread_for_hook = drop_thread.clone();
                let drop_observed_in_flight_for_hook = drop_observed_in_flight.clone();
                let in_flight_at_drop_time_for_hook = in_flight_at_drop_time.clone();
                let request_done_for_hook = request_done.clone();
                register_per_core_resource(
                    "registers-test-resource",
                    Box::new(move || {
                        if let Ok(mut guard) = drop_thread_for_hook.lock() {
                            *guard = Some(thread_id_u64());
                        }
                        drop_observed_in_flight_for_hook.store(
                            if request_done_for_hook.load(Ordering::SeqCst) {
                                0
                            } else {
                                1
                            },
                            Ordering::SeqCst,
                        );
                        in_flight_at_drop_time_for_hook.store(0, Ordering::SeqCst);
                    }),
                );
            }
            tokio::time::sleep(Duration::from_millis(80)).await;
            request_done.store(true, Ordering::SeqCst);
            Ok(Response::ok(bytes::Bytes::from_static(b"ok")))
        }
    }
}


#[proxima::test]
async fn shutdown_barrier_drops_per_core_resource_after_in_flight_completes() {
    let registered = Arc::new(AtomicBool::new(false));
    let serve_thread = Arc::new(std::sync::Mutex::new(None));
    let drop_thread = Arc::new(std::sync::Mutex::new(None));
    let drop_observed_in_flight = Arc::new(AtomicU64::new(99));
    let in_flight_at_drop_time = Arc::new(AtomicU64::new(99));
    let request_done = Arc::new(AtomicBool::new(false));

    let pipe = Registers {
        registered: registered.clone(),
        serve_thread: serve_thread.clone(),
        drop_thread: drop_thread.clone(),
        drop_observed_in_flight: drop_observed_in_flight.clone(),
        in_flight_at_drop_time: in_flight_at_drop_time.clone(),
        request_done: request_done.clone(),
    };

    let mut app = App::new().expect("app");
    app.pipe("registers", Spec::Handle(into_handle(pipe)))
        .await
        .expect("seed");
    app.mount("/", "registers").expect("mount");

    let listener_addr = pick_free_addr().await;
    let shutdown = app
        .run_until_signal(RunConfig {
            bind: listener_addr,
            protocol: "http".into(),
            spec: json!({ "drain_timeout_ms": 5000 }),
        })
        .await
        .expect("run");

    // fire a request that takes 80ms; while it's in-flight, kick off
    // drain. drain MUST wait for the request to complete before
    // firing the per-core drop hook.
    let request_task = tokio::spawn(async move {
        let mut stream = tokio::net::TcpStream::connect(listener_addr)
            .await
            .expect("connect");
        stream
            .write_all(b"GET / HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n")
            .await
            .expect("write");
        stream.flush().await.expect("flush");
        let mut out = Vec::new();
        stream.read_to_end(&mut out).await.expect("read");
        out
    });
    // give the listener a moment to start the request and register the hook
    tokio::time::sleep(Duration::from_millis(20)).await;
    let report = shutdown.drain().await;
    let response = request_task.await.expect("join");

    assert!(
        std::str::from_utf8(&response)
            .map(|text| text.starts_with("HTTP/1.1 200"))
            .unwrap_or(false),
        "in-flight request should have completed during drain; got {response:?}"
    );
    assert!(registered.load(Ordering::SeqCst), "Pipe registered a hook");
    assert_eq!(
        report.cores_acked.min(1),
        1,
        "at least one core should ack (this default App pins to CoreId(0))"
    );
    assert!(
        report.hooks_drained >= 1,
        "barrier should have drained at least the one hook we registered"
    );
    let serve_tid = serve_thread.lock().unwrap().expect("serve thread captured");
    let drop_tid = drop_thread.lock().unwrap().expect("drop thread captured");
    assert_eq!(
        serve_tid, drop_tid,
        "per-core drop hook must fire on the same OS thread that served the request"
    );
    assert_eq!(
        drop_observed_in_flight.load(Ordering::SeqCst),
        0,
        "drop hook must observe request_done=true (drain completed before drop)"
    );
}

fn thread_id_u64() -> u64 {
    let id = std::thread::current().id();
    let text = format!("{id:?}");
    // ThreadId is opaque; hash the Debug string for a stable u64 across calls
    // on the same thread.
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

async fn pick_free_addr() -> std::net::SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("local addr");
    drop(listener);
    addr
}
