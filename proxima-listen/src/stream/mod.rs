//! `StreamListener` → `ListenProtocol` adapter. Each connection becomes
//! a `Request` whose body streams the read half; the `Response` body
//! streams back into the write half.

use std::future::Future;
use std::future::poll_fn;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use futures::FutureExt;
use futures::channel::oneshot;
use futures::io::{AsyncReadExt, AsyncWriteExt};
use futures::stream::{Stream, StreamExt};
use serde_json::Value;
use tracing::{debug, warn};

use proxima_core::ProximaError;
use proxima_runtime::Runtime;
use crate::{ListenProtocol, ServeContext};
use proxima_primitives::pipe::Method;
use proxima_primitives::pipe::body::RequestStream;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::pipe::request::{Request, RequestContext};
use proxima_primitives::stream::{StreamConnection, StreamListener, StreamListenerExt};

mod config;

pub use config::{ListenerStreamConfig, ListenerStreamLayerBuilder};

/// Build-time sizing constants generated from `proxima-listeners-stream.toml`.
/// At no_std+no_alloc (once this crate lifts off its std-only deps) these
/// consts ARE the config; at std they seed [`ListenerStreamConfig`]'s
/// runtime defaults — never duplicated.
pub mod sized {
    include!(concat!(
        env!("OUT_DIR"),
        "/proxima_listeners_stream_sized.rs"
    ));
}

type ListenerFactoryFut<L> = Pin<Box<dyn Future<Output = std::io::Result<L>> + Send>>;
type ListenerFactory<L> = Arc<dyn Fn() -> ListenerFactoryFut<L> + Send + Sync>;

pub struct StreamListenerProtocol<L: StreamListener> {
    label: String,
    method: String,
    path: String,
    chunk_bytes: usize,
    factory: ListenerFactory<L>,
}

impl<L: StreamListener> StreamListenerProtocol<L> {
    /// Factory builds a fresh listener each `serve()` so the control
    /// plane can rebind on restart.
    pub fn with_factory<F, Fut>(label: impl Into<String>, factory: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = std::io::Result<L>> + Send + 'static,
    {
        let config = ListenerStreamConfig::default();
        Self {
            label: label.into(),
            method: config.method,
            path: config.path,
            chunk_bytes: config.chunk_bytes,
            factory: Arc::new(move || Box::pin(factory())),
        }
    }

    pub fn with_method(mut self, method: impl Into<String>) -> Self {
        self.method = method.into();
        self
    }

    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = path.into();
        self
    }

    pub fn with_chunk_bytes(mut self, chunk_bytes: usize) -> Self {
        self.chunk_bytes = chunk_bytes.max(1);
        self
    }
}

impl<L: StreamListener + 'static> ListenProtocol for StreamListenerProtocol<L> {
    fn name(&self) -> &str {
        &self.label
    }

    fn serve(
        &self,
        bind: SocketAddr,
        dispatch: PipeHandle,
        _spec: &Value,
        context: ServeContext,
        mut shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
        let factory = Arc::clone(&self.factory);
        let method = self.method.clone();
        let path = self.path.clone();
        let chunk_bytes = self.chunk_bytes;
        let label = self.label.clone();
        // futures-io serve path: an injected acceptor factory binds + accepts
        // boxed StreamConnections, fed to the same handler as the legacy path.
        // the StreamListener factory path below stays byte-identical without
        // an injected acceptor factory.
        let ready_signal = context.ready_signal.clone();
        // installed App runtime, threaded to spawn_handler — see
        // default_listener::spawn_handler for why a hardcoded `spawn_local`
        // panics on a Prime worker (no tokio LocalSet there).
        let runtime = context.runtime.clone();
        if let Some(acceptor_factory) = context.acceptor_factory.clone() {
            return Box::pin(serve_via_acceptor_factory(
                acceptor_factory,
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
        Box::pin(async move {
            let listener = factory().await.map_err(|err| {
                ProximaError::Io(std::io::Error::other(format!("{label} bind: {err}")))
            })?;
            if let Some(sender) = ready_signal {
                let _ = sender.send(());
            }
            debug!(label = %label, "stream listener bound");
            loop {
                futures::select_biased! {
                    _ = (&mut shutdown).fuse() => return Ok(()),
                    outcome = listener.accept().fuse() => match outcome {
                        Ok(conn) => spawn_handler(
                            conn, dispatch.clone(), method.clone(), path.clone(), chunk_bytes,
                            label.clone(), runtime.clone(),
                        ),
                        Err(error) => warn!(?error, label = %label, "stream listener accept error"),
                    },
                }
            }
        })
    }
}

/// futures-io accept loop mirroring the legacy `StreamListener` handler
/// over an injected `AcceptorFactory`. Binds through the factory (prime-
/// or tokio-backed) and feeds each accepted boxed `StreamConnection` to
/// the same `spawn_handler` as the legacy path.
#[allow(clippy::too_many_arguments)]
async fn serve_via_acceptor_factory(
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
    debug!(label = %label, "stream listener bound (factory)");
    loop {
        futures::select_biased! {
            _ = (&mut shutdown).fuse() => return Ok(()),
            accepted = poll_fn(|cx| acceptor.poll_accept(cx)).fuse() => match accepted {
                Ok(conn) => spawn_handler(
                    conn,
                    dispatch.clone(),
                    method.clone(),
                    path.clone(),
                    chunk_bytes,
                    label.clone(),
                    runtime.clone(),
                ),
                Err(error) => warn!(?error, label = %label, "stream listener accept error (factory)"),
            },
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_handler<C: StreamConnection>(
    conn: C,
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
    });
    // dispatch through the installed Runtime — see
    // default_listener::spawn_handler for why a bare `spawn_local` panics
    // on a Prime worker (no tokio LocalSet there). Falls back to
    // `spawn_local` only for the plain-tokio default path (no App runtime
    // installed), where the surrounding serve loop already runs inside one.
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
        write_half
            .write_all(&bytes)
            .await
            .map_err(|err| ProximaError::Io(std::io::Error::other(format!("write: {err}"))))?;
    }
    write_half
        .close()
        .await
        .map_err(|err| ProximaError::Io(std::io::Error::other(format!("close: {err}"))))?;
    cancel_guard.disarm();
    Ok(())
}

pub fn reader_to_byte_stream<R: futures::io::AsyncRead + Send + Unpin + 'static>(
    mut reader: R,
    chunk_bytes: usize,
) -> impl Stream<Item = Result<Bytes, ProximaError>> + Send {
    async_stream::try_stream! {
        let chunk_bytes = chunk_bytes.max(1);
        loop {
            let mut buf = BytesMut::zeroed(chunk_bytes);
            let read = reader
                .read(&mut buf[..])
                .await
                .map_err(|err| ProximaError::Body(format!("stream read: {err}")))?;
            if read == 0 {
                break;
            }
            buf.truncate(read);
            yield buf.freeze();
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_net::tokio::tokio_stream_listener::TokioTcpListener;
    use proxima_primitives::pipe::handler::into_handle;
    use proxima_primitives::pipe::request::Response as ProximaResponse;
    use std::net::Ipv4Addr;
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


    #[proxima::test]
    async fn echo_protocol_round_trips_bytes_from_tcp_listener() {
        let bind = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
        let listener = TokioTcpListener::bind(bind).await.expect("bind");
        let local = match listener.local_addr().expect("local_addr") {
            proxima_primitives::stream::BindAddr::Tcp(addr) => addr,
            _ => panic!("expected tcp"),
        };

        let dispatch = into_handle(EchoPipe);

        // drive client side concurrently with the connection handler.
        let client_task = tokio::spawn(async move {
            let mut stream = tokio::net::TcpStream::connect(local)
                .await
                .expect("client connect");
            stream
                .write_all(b"the quick brown fox")
                .await
                .expect("client write");
            stream.shutdown().await.expect("client shutdown write");
            let mut response = Vec::new();
            stream
                .read_to_end(&mut response)
                .await
                .expect("client read");
            response
        });

        let conn = listener.accept().await.expect("accept");
        handle_connection(conn, dispatch, "STREAM".into(), "/".into(), 32)
            .await
            .expect("server-side handler");

        let response = client_task.await.expect("client task");
        assert_eq!(response, b"the quick brown fox");
    }
}

pub mod default_listener;
pub use default_listener::StreamListenProtocol;

// `ConnTransform` bakes `TokioTcpConnection` directly into its public
// signature (a real API-shape dependency, not just an accept-loop detail
// like `default_listener`/`StreamListenerProtocol` above) — gated on
// `tokio` rather than generalized, since generalizing it changes the
// public type and is out of scope for this migration.
#[cfg(feature = "tokio")]
pub mod framed_listener;
#[cfg(feature = "tokio")]
pub use framed_listener::{ConnTransform, FramedListenProtocol};

pub mod datagram_listener;
pub use datagram_listener::DatagramListenProtocol;
