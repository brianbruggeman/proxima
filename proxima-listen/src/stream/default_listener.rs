//! `type = "stream"` ListenProtocol over any backend's `AcceptorFactory`
//! (prime, tokio, or another runtime-agnostic implementation). Frames every
//! accepted connection as `Request { method: "STREAM", path: "/", body:
//! stream }`.

use std::future::Future;
use std::future::poll_fn;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;

use futures::FutureExt;
use futures::SinkExt;
use futures::channel::mpsc;
use futures::channel::oneshot;
use futures::stream::StreamExt;
use proxima_telemetry::metric::Counter;
use proxima_telemetry::{debug, warn};
use serde_json::Value;

use super::handle_connection;
use crate::{
    Admission, ConnectionHandle, DispatchPolicy, DrainOutcome, ListenProtocol, ListenerCore,
    ServeContext,
};
use proxima_core::ProximaError;
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::stream::StreamConnection;
use proxima_primitives::sync::AsyncMutex;
use proxima_runtime::Runtime;

/// Counts every finishing connection that found the release channel
/// (`serve_via_factory`'s `release_tx`) full and had to pend on the bounded
/// `send` — the pressure observable for the `OverflowPolicy::Block` release
/// channel (see `spawn_handler`).
pub static RELEASE_CHANNEL_PRESSURE: Counter =
    Counter::new("proxima_listen.stream.release_channel_pressure");

pub struct StreamListenProtocol {
    label: String,
}

impl StreamListenProtocol {
    pub fn new() -> Self {
        Self {
            label: "stream".into(),
        }
    }
}

impl Default for StreamListenProtocol {
    fn default() -> Self {
        Self::new()
    }
}

impl ListenProtocol for StreamListenProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    fn serve(
        &self,
        bind: SocketAddr,
        dispatch: PipeHandle,
        spec: &Value,
        context: ServeContext,
        shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
        let method = spec
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or(super::sized::LISTENER_METHOD_DEFAULT)
            .to_string();
        let path = spec
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or(super::sized::LISTENER_PATH_DEFAULT)
            .to_string();
        let chunk_bytes = spec
            .get("chunk_bytes")
            .and_then(Value::as_u64)
            .map(|raw| raw.max(1) as usize)
            .unwrap_or(super::sized::LISTENER_CHUNK_BYTES_DEFAULT);
        let release_channel_capacity = spec
            .get("release_channel_capacity")
            .and_then(Value::as_u64)
            .map(|raw| raw.max(1) as usize)
            .unwrap_or(super::sized::LISTENER_RELEASE_CHANNEL_CAPACITY_DEFAULT);
        let label = self.label.clone();
        // futures-io serve path: an injected acceptor factory binds + accepts
        // boxed StreamConnections, runtime- and backend-agnostic.
        let ready_signal = context.ready_signal.clone();
        // installed App runtime (Prime or TokioPerCoreRuntime), threaded down
        // to spawn_handler so per-conn tasks dispatch through the runtime that
        // actually owns this thread, not a hardcoded tokio primitive — a Prime
        // worker never enters a tokio LocalSet, so `spawn_local` alone panics
        // there (see default_listener::spawn_handler).
        let runtime = context.runtime.clone();
        if let Some(factory) = context.acceptor_factory.clone() {
            return Box::pin(serve_via_factory(
                factory,
                bind,
                dispatch,
                method,
                path,
                chunk_bytes,
                release_channel_capacity,
                label,
                shutdown,
                ready_signal,
                runtime,
            ));
        }
        // No `AcceptorFactory` injected: there is no bind seam to use, on
        // any backend. Generic listener code must never reach for a
        // concrete runtime's bind directly (that was the previous
        // `TokioTcpListener::bind` fallback here, gone under the
        // additivity fix) — this is one explicit, feature-independent
        // configuration error instead.
        let _ = (
            dispatch,
            method,
            path,
            chunk_bytes,
            release_channel_capacity,
            ready_signal,
            shutdown,
            runtime,
        );
        Box::pin(async move {
            Err(ProximaError::Config(format!(
                "{label} listener requires an acceptor factory (none injected on \
                 ServeContext — install one via `App`'s runtime bundle or \
                 `ServeContext::with_acceptor_factory`)"
            )))
        })
    }
}

/// futures-io accept loop mirroring the legacy tokio handler over an
/// injected `AcceptorFactory`. Binds through the factory (prime- or
/// tokio-backed) and feeds each accepted boxed `StreamConnection` to the
/// same `handle_connection` as the legacy path.
// every argument is a distinct per-serve value read off the spec or the
// ServeContext seam; a params struct would carry no invariant the tuple
// does not, so it would be a relocation, not a type.
#[allow(clippy::too_many_arguments)]
async fn serve_via_factory(
    factory: Arc<dyn proxima_primitives::stream::AcceptorFactory>,
    bind: SocketAddr,
    dispatch: PipeHandle,
    method: String,
    path: String,
    chunk_bytes: usize,
    release_channel_capacity: usize,
    label: String,
    mut shutdown: oneshot::Receiver<()>,
    ready_signal: Option<crate::ReadySignal>,
    runtime: Option<Arc<dyn Runtime>>,
) -> Result<(), ProximaError> {
    let options = proxima_primitives::stream::TcpBindOptions::default();
    let mut acceptor = factory.bind(bind, options).map_err(ProximaError::Io)?;
    if let Some(sender) = ready_signal {
        let _ = sender.send(Ok(()));
    }
    debug!(label = %label, %bind, "stream listener bound (factory)");
    let mut core = ListenerCore::new(DispatchPolicy::Inline);
    // Bounded (OverflowPolicy::Block): one message per finishing connection.
    // The accept loop below is the sole consumer and always makes progress
    // (it never itself awaits this channel), so a full channel only pends
    // the finishing connection's own task briefly — never a deadlock —
    // while capping worst-case queue memory under a release burst.
    //
    // `futures::channel::mpsc` grants every `Sender` clone its own
    // guaranteed slot on top of the buffer, regardless of how full the
    // channel already is — so handing `spawn_handler` a fresh
    // `release_tx.clone()` per accepted connection (the shape this used to
    // be) defeats the bound entirely. There is exactly ONE live `Sender`
    // for this channel's whole life; every spawned handler borrows it
    // through `AsyncMutex` (a waker-based async gate — pends the task
    // instead of blocking a thread) instead of minting a new clone. Every
    // spawned handler task on this core is cooperatively driven on a
    // single OS thread (see `spawn_handler`'s own doc: `?Send` for life,
    // pinned rather than work-stolen), so contention on this mutex is
    // scheduling contention, never true parallel access — the gate exists
    // to share the one `Sender` across `'static` tasks, not to arbitrate
    // real concurrent execution.
    let (release_tx, mut release_rx) = mpsc::channel::<ConnectionHandle>(release_channel_capacity);
    let release_tx = Arc::new(AsyncMutex::new(release_tx));
    loop {
        futures::select_biased! {
            _ = (&mut shutdown).fuse() => match core.begin_drain() {
                DrainOutcome::ClosedImmediately => return Ok(()),
                DrainOutcome::Draining => break,
            },
            released = release_rx.next().fuse() => if let Some(handle) = released {
                core.release(handle);
            },
            accepted = poll_fn(|cx| acceptor.poll_accept(cx)).fuse() => match accepted {
                Ok(conn) => match core.admit(crate::peer_ip(conn.peer().as_ref())) {
                    Admission::Admit { handle, .. } => spawn_handler(
                        conn, handle, Arc::clone(&release_tx), dispatch.clone(),
                        method.clone(), path.clone(), chunk_bytes, label.clone(),
                        runtime.clone(),
                    ),
                    Admission::Shed { reason } => {
                        debug!(?reason, label = %label, "stream connection shed (factory)");
                        drop(conn);
                    }
                },
                Err(error) => warn!(?error, label = %label, "stream accept error (factory)"),
            },
        }
    }
    drain_connections(&mut core, &mut release_rx).await;
    Ok(())
}

/// Drain phase: no longer accepting, wait for in-flight connections to release
/// their admission slots until the core reports closed.
async fn drain_connections(
    core: &mut ListenerCore,
    release_rx: &mut mpsc::Receiver<ConnectionHandle>,
) {
    while !core.is_closed() {
        match release_rx.next().await {
            Some(handle) => {
                core.release(handle);
            }
            None => break,
        }
    }
}

// same reason as `serve_via_factory` above, plus the admission handle and
// its release channel; no invariant binds them into a type.
#[allow(clippy::too_many_arguments)]
fn spawn_handler<C: StreamConnection>(
    conn: C,
    handle: ConnectionHandle,
    release_tx: Arc<AsyncMutex<mpsc::Sender<ConnectionHandle>>>,
    dispatch: PipeHandle,
    method: String,
    path: String,
    chunk_bytes: usize,
    label: String,
    runtime: Option<Arc<dyn Runtime>>,
) {
    // per-conn future holds `?Send` Pipe::call for life, so it must stay
    // pinned to the accepting core rather than work-stolen.
    let future: Pin<Box<dyn Future<Output = ()> + 'static>> = Box::pin(async move {
        if let Err(error) = handle_connection(conn, dispatch, method, path, chunk_bytes).await {
            warn!(?error, label = %label, "stream connection error");
        }
        // Release the admission slot so the listener can drain / re-admit.
        // Bounded + Block: `try_send` first (the common case, capacity
        // available); on `Full` record the pressure observable, then fall
        // to the awaiting `send` — pends this connection's own task until
        // the accept loop drains a slot. Never a silent drop of a release.
        //
        // `lock().await` acquires the ONE live `Sender` shared by every
        // finishing connection (see its construction in `serve_via_factory`)
        // instead of minting a fresh clone — a fresh clone per connection is
        // exactly the defect this pends: `futures::channel::mpsc` gives
        // every clone its own guaranteed slot, so a clone-per-connection
        // producer could never actually observe `Full` and this pressure
        // counter could never engage.
        let mut sender = release_tx.lock().await;
        if let Err(err) = sender.try_send(handle)
            && err.is_full()
        {
            RELEASE_CHANNEL_PRESSURE.add(1, &[]);
            let _ = sender.send(handle).await;
        }
    });
    // dispatch through the installed Runtime (Prime's CoreShard or tokio's
    // per-core LocalSet) when one is set; a Prime worker never enters a
    // tokio LocalSet, so calling `spawn_local` directly there panics —
    // confirmed by reading `TokioPerCoreRuntime`'s worker (wraps
    // `LocalSet::run_until`) against `PrimeRuntime`'s (no tokio at all).
    // With no runtime injected, every build does the SAME explicit thing:
    // drop the connection and say why — spawning is a runtime capability
    // and belongs on the seam, never behind a feature cfg here.
    match runtime {
        Some(runtime) => runtime.spawn_on_current_core(future),
        None => {
            warn!(
                "stream connection dropped: no runtime injected onto ServeContext, so \
                 there is no executor to spawn the ?Send connection future onto"
            );
            drop(future);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use proxima_primitives::pipe::SendPipe;
    use proxima_primitives::pipe::handler::into_handle;
    use proxima_primitives::pipe::header_list::HeaderList;
    use proxima_primitives::pipe::request::Request;
    use proxima_primitives::pipe::request::Response as ProximaResponse;
    use proxima_primitives::pipe::telemetry_surface::NoopTelemetry;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    struct EchoPipe;

    impl SendPipe for EchoPipe {
        type In = Request<Bytes>;
        type Out = ProximaResponse<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<ProximaResponse<Bytes>, ProximaError>> + Send {
            async move {
                let (_, bytes) = request.body_bytes().await?;
                Ok(ProximaResponse {
                    status: 200,
                    metadata: HeaderList::new(),
                    payload: bytes,
                    stream: None,
                    upgrade: None,
                })
            }
        }
    }

    // test-only `Runtime` whose `spawn_on_current_core` forwards to
    // `tokio::task::spawn_local` — stands in for an installed runtime so
    // these tests exercise the seam (`Some(runtime)`) rather than the
    // no-runtime degradation. Must run inside a `LocalSet`.
    struct LocalSetRuntime;

    impl Runtime for LocalSetRuntime {
        fn spawn_on_current_core(&self, future: Pin<Box<dyn Future<Output = ()> + 'static>>) {
            tokio::task::spawn_local(future);
        }

        fn spawn_on_core(
            &self,
            _core_id: proxima_runtime::CoreId,
            _future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
        ) -> Result<(), proxima_runtime::SpawnError> {
            unreachable!("test never routes to a peer core")
        }

        fn spawn_factory_on_core(
            &self,
            _core_id: proxima_runtime::CoreId,
            _factory: Box<
                dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + 'static>> + Send + 'static,
            >,
        ) -> Result<(), proxima_runtime::SpawnError> {
            unreachable!("test never spawns a cross-core factory")
        }

        fn spawn_background_blocking(
            &self,
            _work: Box<dyn FnOnce() -> Result<Box<dyn std::any::Any + Send>, ProximaError> + Send>,
        ) -> proxima_runtime::BackgroundHandle<Box<dyn std::any::Any + Send>> {
            unreachable!("test never spawns background-blocking work")
        }

        fn timer_at(
            &self,
            _deadline: std::time::Instant,
        ) -> Pin<Box<dyn Future<Output = ()> + 'static>> {
            unreachable!("test never times out")
        }

        fn num_cores(&self) -> usize {
            1
        }

        fn current_core(&self) -> proxima_runtime::CoreId {
            proxima_runtime::CoreId(0)
        }
    }

    // proves the factory path binds, accepts, and round-trips bytes with a
    // runtime injected via the seam. `LocalSetRuntime` (above) backs the
    // per-conn task, so this runs inside a LocalSet on a current-thread
    // runtime. serve borrows `protocol` + `server_spec`, so it is driven
    // concurrently with the client rather than spawned.
    #[proxima::test(runtime = "tokio")]
    async fn factory_path_round_trips_bytes() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async move {
                let dispatch = into_handle(EchoPipe);
                let runtime: Arc<dyn Runtime> = Arc::new(LocalSetRuntime);
                let context = ServeContext::new(NoopTelemetry::handle())
                    .with_runtime(runtime)
                    .with_acceptor_factory(Arc::new(proxima_net::tokio::TokioAcceptorFactory));
                let (shutdown_tx, shutdown_rx) = oneshot::channel();

                let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("probe bind");
                let addr = probe.local_addr().expect("probe addr");
                drop(probe);

                let server_spec = serde_json::json!({ "chunk_bytes": 32 });
                let protocol = StreamListenProtocol::new();
                let serve = protocol.serve(addr, dispatch, &server_spec, context, shutdown_rx);

                let client_work = async {
                    let mut client = loop {
                        match tokio::net::TcpStream::connect(addr).await {
                            Ok(stream) => break stream,
                            Err(_) => tokio::task::yield_now().await,
                        }
                    };
                    client
                        .write_all(b"the quick brown fox")
                        .await
                        .expect("client write");
                    client.shutdown().await.expect("client shutdown write");
                    let mut response = Vec::new();
                    client
                        .read_to_end(&mut response)
                        .await
                        .expect("client read");
                    response
                };

                let response = tokio::select! {
                    serve_result = serve => panic!("serve returned early: {serve_result:?}"),
                    response = client_work => response,
                };
                assert_eq!(response, b"the quick brown fox");
                drop(shutdown_tx);
            })
            .await;
    }

    // `spawn_handler` used to clone `release_tx` fresh per accepted
    // connection. `futures::channel::mpsc` grants every `Sender` clone its
    // own guaranteed slot regardless of how full the channel already is
    // (each clone starts with its own `maybe_parked = false`, checked
    // purely locally by `try_send` — never consulting the shared queue
    // depth on that clone's first attempt) — so a clone-per-connection
    // producer could never actually observe `Full` and
    // `RELEASE_CHANNEL_PRESSURE` could never engage.
    //
    // The fix: every finishing connection now shares the SAME live
    // `Sender`, serialized through `AsyncMutex` (see `serve_via_factory`'s
    // construction of `release_tx`). This test proves the FIXED contract —
    // reproducing exactly `spawn_handler`'s release snippet for three
    // concurrently-finishing connections sharing a channel with a single
    // reserved slot (capacity 0): at least one of the three genuinely
    // contends and pends, every release still lands losslessly, and the
    // pressure counter records the real contention instead of staying
    // permanently at zero.
    #[proxima::test]
    async fn release_channel_pressure_engages_because_connections_now_share_one_sender() {
        let before = RELEASE_CHANNEL_PRESSURE.get();
        let (release_tx, mut release_rx) = mpsc::channel::<ConnectionHandle>(0);
        let release_tx = Arc::new(AsyncMutex::new(release_tx));

        let release_one = |slot: u32| {
            let release_tx = Arc::clone(&release_tx);
            async move {
                let handle = ConnectionHandle(slot);
                // The exact snippet `spawn_handler` runs when a connection ends.
                let mut sender = release_tx.lock().await;
                if let Err(err) = sender.try_send(handle)
                    && err.is_full()
                {
                    RELEASE_CHANNEL_PRESSURE.add(1, &[]);
                    let _ = sender.send(handle).await;
                }
            }
        };

        let releases = futures::future::join3(release_one(0), release_one(1), release_one(2));
        let drain = async {
            let mut released = Vec::new();
            while released.len() < 3 {
                match release_rx.next().await {
                    Some(handle) => released.push(handle),
                    None => break,
                }
            }
            released
        };
        let (_, released) = futures::join!(releases, drain);

        assert_eq!(
            released.len(),
            3,
            "all three releases land losslessly despite the channel's configured \
             capacity of 0 — the shared sender pends instead of dropping work"
        );
        assert!(
            RELEASE_CHANNEL_PRESSURE.get() > before,
            "with only one reserved slot shared by three concurrently-finishing \
             connections, at least one try_send must observe Full and increment \
             the pressure counter — the bound now actually engages"
        );
    }

    // with nothing in flight, firing shutdown drains immediately through the
    // ListenerCore and serve returns Ok — proves the admission core is wired
    // into the accept loop's drain path.
    #[proxima::test(runtime = "tokio")]
    async fn factory_path_returns_on_shutdown() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async move {
                let dispatch = into_handle(EchoPipe);
                let context = ServeContext::new(NoopTelemetry::handle())
                    .with_acceptor_factory(Arc::new(proxima_net::tokio::TokioAcceptorFactory));
                let (shutdown_tx, shutdown_rx) = oneshot::channel();

                let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("probe bind");
                let addr = probe.local_addr().expect("probe addr");
                drop(probe);

                let server_spec = serde_json::json!({ "chunk_bytes": 32 });
                let protocol = StreamListenProtocol::new();
                let serve = protocol.serve(addr, dispatch, &server_spec, context, shutdown_rx);

                drop(shutdown_tx);
                let result = serve.await;
                assert!(
                    result.is_ok(),
                    "serve should drain and return Ok, got {result:?}"
                );
            })
            .await;
    }
}
