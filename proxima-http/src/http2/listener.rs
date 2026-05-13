//! HTTP/2 listener — strictly h2 prior-knowledge (h2c), no h1
//! fallthrough, no ALPN multiplex.
//!
//! Sibling of [`crate::listeners::h1`] and [`crate::listeners::h3`]:
//! one wire version per listener type, uniform surface.
//!
//! Real-world TLS + ALPN h2 lives in
//! [`crate::listeners::http::HttpListenProtocol`] (the combiner) where
//! it can multiplex with h1 on the same socket. This sibling is for
//! the h2c case — pipe-mesh sidecars, internal traffic, h2-only
//! gateways — where TLS is terminated upstream or not needed.

use std::future::Future;
use std::future::poll_fn;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use futures::FutureExt;
use futures::channel::oneshot;
use serde_json::Value;
#[cfg(feature = "http2")]
use tokio_util::compat::TokioAsyncReadCompatExt;
use tracing::{debug, warn};

use crate::http2::serve_h2_connection;
use proxima_core::ProximaError;
use proxima_listen::{ListenProtocol, ServeContext};
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::pipe::quiesce::QuiesceResponse;
#[cfg(feature = "http2")]
use proxima_primitives::stream::PeerInfo;
#[cfg(feature = "http2")]
use tokio::net::TcpListener;

pub struct H2ListenProtocol {
    label: String,
}

impl Default for H2ListenProtocol {
    fn default() -> Self {
        Self { label: "h2".into() }
    }
}

impl H2ListenProtocol {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_label(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }
}

impl ListenProtocol for H2ListenProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    fn serve(
        &self,
        bind: SocketAddr,
        dispatch: PipeHandle,
        _spec: &Value,
        context: ServeContext,
        shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
        let runtime_for_conns = context.runtime.clone();
        let handler_dispatch_for_conn = context.handler_dispatch;
        let quiesce = Arc::new(QuiesceResponse {
            status: 503,
            retry_after: "1".into(),
        });

        // futures-io serve path: a runtime-matched acceptor factory binds +
        // accepts boxed StreamConnections, each fed straight to the sans-IO
        // serve_h2_connection (no preface sniff — h2 prior-knowledge). The
        // legacy tokio path below stays byte-identical without a factory.
        let ready_signal = context.ready_signal.clone();
        if let Some(factory) = context.acceptor_factory.clone() {
            let handler_dispatch = context.handler_dispatch;
            return Box::pin(serve_via_factory(
                factory,
                bind,
                dispatch,
                quiesce,
                runtime_for_conns,
                handler_dispatch,
                shutdown,
                ready_signal,
            ));
        }

        #[cfg(feature = "http2")]
        {
            return Box::pin(async move {
                let listener = TcpListener::bind(bind).await.map_err(ProximaError::Io)?;
                if let Some(sender) = ready_signal {
                    let _ = sender.send(());
                }
                debug!(?bind, "h2 listener bound (h2c, prior-knowledge)");
                let mut shutdown = shutdown;
                let policy = handler_dispatch_for_conn.as_policy();
                let mut route_cursor: usize = 0;
                loop {
                    tokio::select! {
                        biased;
                        _ = &mut shutdown => break,
                        accepted = listener.accept() => match accepted {
                            Ok((socket, peer)) => {
                                let _ = socket.set_nodelay(true);
                                let dispatch = dispatch.clone();
                                let quiesce_for_conn = quiesce.clone();
                                let in_flight_for_conn = Arc::new(AtomicU64::new(0));
                                let peer_info = Some(PeerInfo::Tcp(peer));
                                let route = policy.route(&mut route_cursor);
                                let conn_future = async move {
                                    if let Err(error) = serve_h2_connection(
                                        socket.compat(),
                                        dispatch,
                                        in_flight_for_conn,
                                        quiesce_for_conn,
                                        peer_info,
                                    )
                                    .await
                                    {
                                        warn!(?error, "h2 connection error");
                                    }
                                };
                                proxima_listen::dispatch_handler(
                                    runtime_for_conns.as_ref(),
                                    route,
                                    Box::pin(conn_future),
                                );
                            }
                            Err(error) => warn!(?error, "h2 accept failed"),
                        }
                    }
                }
                Ok(())
            });
        }
        // tokio-free default: no `AcceptorFactory` means there's no bind
        // path at all — the factory arm above is the only default-build
        // way to serve h2. Restore the legacy tokio bind+accept loop with
        // `--features tokio`.
        #[cfg(not(feature = "http2"))]
        {
            let _ = (ready_signal, handler_dispatch_for_conn, runtime_for_conns);
            Box::pin(async move {
                Err(ProximaError::Config(
                    "h2 listener requires an AcceptorFactory (no `tokio` feature fallback in this build)".into(),
                ))
            })
        }
    }
}

/// futures-io accept loop mirroring the legacy tokio handler over an
/// injected `AcceptorFactory`. Binds through the factory (prime- or
/// tokio-backed) and feeds each accepted boxed `StreamConnection`
/// straight to the sans-IO `serve_h2_connection`. The boxed acceptor
/// already sets TCP_NODELAY, so this path does not.
// ready_signal is the readiness-gate wiring shared by every ListenProtocol serve path.
#[allow(clippy::too_many_arguments)]
async fn serve_via_factory(
    factory: Arc<dyn proxima_primitives::stream::AcceptorFactory>,
    bind: SocketAddr,
    dispatch: PipeHandle,
    quiesce: Arc<QuiesceResponse>,
    runtime: Option<Arc<dyn proxima_runtime::Runtime>>,
    handler_dispatch: proxima_listen::HandlerDispatch,
    mut shutdown: oneshot::Receiver<()>,
    ready_signal: Option<std::sync::mpsc::Sender<()>>,
) -> Result<(), ProximaError> {
    let options = proxima_primitives::stream::TcpBindOptions {
        backlog: proxima_primitives::stream::DEFAULT_LISTEN_BACKLOG,
        reuseport: false,
        tcp_fastopen: None,
    };
    let mut acceptor = factory.bind(bind, options).map_err(ProximaError::Io)?;
    if let Some(sender) = ready_signal {
        let _ = sender.send(());
    }
    debug!(?bind, "h2 listener bound (h2c, prior-knowledge, factory)");
    let policy = handler_dispatch.as_policy();
    let mut route_cursor: usize = 0;
    loop {
        futures::select_biased! {
            _ = (&mut shutdown).fuse() => break,
            accepted = poll_fn(|cx| acceptor.poll_accept(cx)).fuse() => match accepted {
                Ok(conn) => {
                    let dispatch_for_conn = dispatch.clone();
                    let quiesce_for_conn = quiesce.clone();
                    let in_flight_for_conn = Arc::new(AtomicU64::new(0));
                    let peer_info = conn.peer();
                    let route = policy.route(&mut route_cursor);
                    let conn_future = async move {
                        if let Err(error) = serve_h2_connection(
                            conn,
                            dispatch_for_conn,
                            in_flight_for_conn,
                            quiesce_for_conn,
                            peer_info,
                        )
                        .await
                        {
                            warn!(?error, "h2 connection error");
                        }
                    };
                    proxima_listen::dispatch_handler(runtime.as_ref(), route, Box::pin(conn_future));
                }
                Err(error) => warn!(?error, "h2 accept failed"),
            },
        }
    }
    Ok(())
}

// exercises the legacy tokio TcpListener via TokioAcceptorFactory + the
// tokio test harness — needs the `tokio` feature.
#[cfg(all(test, feature = "http2"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    use bytes::Bytes;
    use proxima_primitives::pipe::SendPipe;
    use proxima_primitives::pipe::handler::into_handle;
    use proxima_primitives::pipe::request::{Request, Response};
    use proxima_primitives::pipe::telemetry_surface::NoopTelemetry;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

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


    /// 24-byte h2 client connection preface (RFC 7540 §3.5).
    const H2_CLIENT_PREFACE: &[u8; 24] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

    // proves the factory path binds, accepts, and drives the accepted
    // connection through serve_h2_connection: a prior-knowledge h2 client
    // sends the preface + an empty SETTINGS frame and the connection
    // driver replies with its own SETTINGS frame (its first wire action).
    // hand-rolling a full HPACK request is heavier than this needs to be —
    // the SETTINGS exchange alone proves the accepted stream reached the
    // sans-IO driver. spawn_local backs the per-conn task, so this runs
    // inside a LocalSet on a current-thread runtime.
    #[proxima::test(runtime = "tokio")]
    async fn factory_path_drives_h2_connection() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async move {
                let dispatch = into_handle(ConstantOk);
                let context = ServeContext::new(NoopTelemetry::handle())
                    .with_acceptor_factory(Arc::new(proxima_net::tokio::TokioAcceptorFactory));
                let (shutdown_tx, shutdown_rx) = oneshot::channel();

                let probe = tokio::net::TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("probe bind");
                let addr = probe.local_addr().expect("probe addr");
                drop(probe);

                let server_spec = serde_json::json!({});
                let protocol = H2ListenProtocol::new();
                let serve = protocol.serve(addr, dispatch, &server_spec, context, shutdown_rx);

                let client_work = async {
                    let mut client = loop {
                        match tokio::net::TcpStream::connect(addr).await {
                            Ok(stream) => break stream,
                            Err(_) => tokio::task::yield_now().await,
                        }
                    };
                    client
                        .write_all(H2_CLIENT_PREFACE)
                        .await
                        .expect("client preface write");
                    // empty SETTINGS frame: 0 length, type 0x4, no flags, stream 0.
                    let settings_frame = [0u8, 0, 0, 0x4, 0, 0, 0, 0, 0];
                    client
                        .write_all(&settings_frame)
                        .await
                        .expect("client settings write");
                    client.flush().await.expect("client flush");
                    // the driver answers with its own SETTINGS frame; read the
                    // 9-byte frame header and assert the type byte is SETTINGS.
                    let mut header = [0u8; 9];
                    client
                        .read_exact(&mut header)
                        .await
                        .expect("client read settings header");
                    header[3]
                };

                let frame_type = tokio::select! {
                    serve_result = serve => panic!("serve returned early: {serve_result:?}"),
                    frame_type = client_work => frame_type,
                };
                assert_eq!(frame_type, 0x4, "expected server SETTINGS frame");
                drop(shutdown_tx);
            })
            .await;
    }
}
