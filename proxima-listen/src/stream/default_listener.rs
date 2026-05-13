//! `type = "stream"` ListenProtocol over tokio TCP. Frames every accepted
//! connection as `Request { method: "STREAM", path: "/", body: stream }`.

use std::future::Future;
use std::future::poll_fn;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;

use bytes::Bytes;
use futures::FutureExt;
use futures::channel::mpsc;
use futures::channel::oneshot;
use futures::io::{AsyncReadExt, AsyncWriteExt};
use futures::stream::StreamExt;
use serde_json::Value;
use tracing::{debug, warn};

use super::reader_to_byte_stream;
use proxima_core::ProximaError;
use crate::{
    Admission, ConnectionHandle, DispatchPolicy, DrainOutcome, ListenProtocol, ListenerCore,
    ServeContext,
};
#[cfg(feature = "tokio")]
use proxima_net::tokio::tokio_stream_listener::TokioTcpListener;
use proxima_primitives::pipe::Method;
use proxima_runtime::Runtime;
use proxima_primitives::pipe::body::RequestStream;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::pipe::request::{Request, RequestContext};
#[cfg(feature = "tokio")]
use proxima_primitives::stream::StreamListenerExt;
use proxima_primitives::stream::StreamConnection;

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

    #[cfg_attr(not(feature = "tokio"), allow(unused_mut))]
    fn serve(
        &self,
        bind: SocketAddr,
        dispatch: PipeHandle,
        spec: &Value,
        context: ServeContext,
        mut shutdown: oneshot::Receiver<()>,
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
        let label = self.label.clone();
        // futures-io serve path: an injected acceptor factory binds + accepts
        // boxed StreamConnections, fed to the same handler as the legacy path.
        // the tokio TokioTcpListener path below stays byte-identical without
        // a factory.
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
                label,
                shutdown,
                ready_signal,
                runtime,
            ));
        }
        #[cfg(feature = "tokio")]
        return Box::pin(async move {
            let listener = TokioTcpListener::bind(bind).await.map_err(|err| {
                ProximaError::Io(std::io::Error::other(format!("stream bind {bind}: {err}")))
            })?;
            if let Some(sender) = ready_signal {
                let _ = sender.send(());
            }
            debug!(label = %label, %bind, "stream listener bound");
            let mut core = ListenerCore::new(DispatchPolicy::Inline);
            let (release_tx, mut release_rx) = mpsc::unbounded::<ConnectionHandle>();
            loop {
                tokio::select! {
                    outcome = listener.accept() => match outcome {
                        Ok(conn) => match core.admit(crate::peer_ip(conn.peer().as_ref())) {
                            Admission::Admit { handle, .. } => spawn_handler(
                                conn, handle, release_tx.clone(), dispatch.clone(),
                                method.clone(), path.clone(), chunk_bytes, label.clone(),
                                runtime.clone(),
                            ),
                            Admission::Shed { reason } => {
                                debug!(?reason, label = %label, "stream connection shed");
                                drop(conn);
                            }
                        },
                        Err(error) => warn!(?error, label = %label, "stream accept error"),
                    },
                    released = release_rx.next() => if let Some(handle) = released {
                        core.release(handle);
                    },
                    _ = &mut shutdown => match core.begin_drain() {
                        DrainOutcome::ClosedImmediately => return Ok(()),
                        DrainOutcome::Draining => break,
                    },
                }
            }
            drain_connections(&mut core, &mut release_rx).await;
            Ok(())
        });
        // No factory and no tokio: there is no tokio-free bind path left to
        // fall back to (only the factory + tokio legacy arms exist above).
        #[cfg(not(feature = "tokio"))]
        {
            let _ = (
                bind,
                dispatch,
                method,
                path,
                chunk_bytes,
                label,
                ready_signal,
                shutdown,
                runtime,
            );
            Box::pin(async move {
                Err(ProximaError::Config(
                    "stream listener requires an acceptor factory (no factory injected and the \
                     `tokio` feature is off, so the legacy tokio bind path is unavailable)"
                        .into(),
                ))
            })
        }
    }
}

/// futures-io accept loop mirroring the legacy tokio handler over an
/// injected `AcceptorFactory`. Binds through the factory (prime- or
/// tokio-backed) and feeds each accepted boxed `StreamConnection` to the
/// same `handle_connection` as the legacy path.
#[allow(clippy::too_many_arguments)]
async fn serve_via_factory(
    factory: Arc<dyn proxima_primitives::stream::AcceptorFactory>,
    bind: SocketAddr,
    dispatch: PipeHandle,
    method: String,
    path: String,
    chunk_bytes: usize,
    label: String,
    mut shutdown: oneshot::Receiver<()>,
    ready_signal: Option<std::sync::mpsc::Sender<()>>,
    runtime: Option<Arc<dyn Runtime>>,
) -> Result<(), ProximaError> {
    let options = proxima_primitives::stream::TcpBindOptions::default();
    let mut acceptor = factory.bind(bind, options).map_err(ProximaError::Io)?;
    if let Some(sender) = ready_signal {
        let _ = sender.send(());
    }
    debug!(label = %label, %bind, "stream listener bound (factory)");
    let mut core = ListenerCore::new(DispatchPolicy::Inline);
    let (release_tx, mut release_rx) = mpsc::unbounded::<ConnectionHandle>();
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
                        conn, handle, release_tx.clone(), dispatch.clone(),
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
    release_rx: &mut mpsc::UnboundedReceiver<ConnectionHandle>,
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

#[allow(clippy::too_many_arguments)]
fn spawn_handler<C: StreamConnection>(
    conn: C,
    handle: ConnectionHandle,
    release_tx: mpsc::UnboundedSender<ConnectionHandle>,
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
        // release the admission slot so the listener can drain / re-admit.
        let _ = release_tx.unbounded_send(handle);
    });
    // dispatch through the installed Runtime (Prime's CoreShard or tokio's
    // per-core LocalSet) when one is set; a Prime worker never enters a
    // tokio LocalSet, so calling `spawn_local` directly there panics —
    // confirmed by reading `TokioPerCoreRuntime`'s worker (wraps
    // `LocalSet::run_until`) against `PrimeRuntime`'s (no tokio at all).
    // Falls back to `spawn_local` only for the plain-tokio default path
    // (no App runtime installed), where the surrounding serve loop is
    // already known to run inside a LocalSet.
    match runtime {
        Some(runtime) => runtime.spawn_on_current_core(future),
        #[cfg(feature = "tokio")]
        None => {
            tokio::task::spawn_local(future);
        }
        #[cfg(not(feature = "tokio"))]
        None => {
            warn!(
                "stream connection dropped: no runtime injected and the `tokio` \
                 feature is off, so there is no executor to spawn the ?Send \
                 connection future onto"
            );
            drop(future);
        }
    }
}

async fn handle_connection<C: StreamConnection>(
    conn: C,
    dispatch: PipeHandle,
    method: String,
    path: String,
    chunk_bytes: usize,
) -> Result<(), ProximaError> {
    let (read_half, mut write_half) = conn.split();
    let context = RequestContext::default();
    let cancel = context.child_signal();
    let cancel_guard = cancel.clone().guard();
    let stream = RequestStream::new(reader_to_byte_stream(read_half, chunk_bytes));
    let request = Request {
        method: Method::from(method.as_str()),
        path: Bytes::from(path),
        query: HeaderList::new(),
        metadata: HeaderList::new(),
        payload: Bytes::new(),
        stream: Some(stream),
        context: context.with_cancel(cancel),
    };
    let response = SendPipe::call(&dispatch, request).await?;
    let mut response_stream = response.into_chunk_stream();
    while let Some(chunk) = response_stream.next().await {
        let bytes = chunk.map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!("response body: {err}")))
        })?;
        write_half.write_all(&bytes).await.map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!("stream write: {err}")))
        })?;
    }
    write_half
        .close()
        .await
        .map_err(|err| ProximaError::Io(std::io::Error::other(format!("stream close: {err}"))))?;
    cancel_guard.disarm();
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_primitives::pipe::header_list::HeaderList;
    use proxima_primitives::pipe::handler::into_handle;
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


    // proves the factory path binds, accepts, and round-trips bytes through
    // the same handle_connection as the legacy tokio path. spawn_local backs
    // the per-conn task, so this runs inside a LocalSet on a current-thread
    // runtime. serve borrows `protocol` + `server_spec`, so it is driven
    // concurrently with the client rather than spawned.
    #[proxima::test(runtime = "tokio")]
    async fn factory_path_round_trips_bytes() {
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
