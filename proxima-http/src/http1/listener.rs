//! HTTP/1.1 listener — strictly h1, no ALPN multiplex, no h2 sniff.
//!
//! Sibling of the umbrella's `h2` and `h3` listeners: one wire version
//! per listener type. Each has the same surface (`new`, `with_label`,
//! `impl ListenProtocol`), so callers compose protocols uniformly.
//!
//! For TLS, ALPN-multiplexed h1+h2, UDS, SO_REUSEPORT, io_uring,
//! quiesce / drain, telemetry — use the umbrella's `HttpListenProtocol`
//! combiner. This sibling is intentionally small: plain TCP +
//! native h1 driver, nothing else.

use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;

use futures::channel::oneshot;
use serde_json::Value;
use tokio_util::compat::TokioAsyncReadCompatExt;
use tracing::{debug, warn};

use crate::http1::serve::serve_h1_connection;
use proxima_core::ProximaError;
use proxima_listen::{ListenProtocol, ServeContext};
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::stream::PeerInfo;
use tokio::net::TcpListener;

pub struct H1ListenProtocol {
    label: String,
}

impl Default for H1ListenProtocol {
    fn default() -> Self {
        Self { label: "h1".into() }
    }
}

impl H1ListenProtocol {
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

impl ListenProtocol for H1ListenProtocol {
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
        let max_body_bytes = spec
            .get("max_body_bytes")
            .and_then(Value::as_u64)
            .map(|raw| raw as usize);
        let runtime_for_conns = context.runtime.clone();
        let ready_signal = context.ready_signal.clone();

        Box::pin(async move {
            let listener = TcpListener::bind(bind).await.map_err(ProximaError::Io)?;
            if let Some(sender) = ready_signal {
                let _ = sender.send(());
            }
            debug!(?bind, "h1 listener bound");
            let mut shutdown = shutdown;
            loop {
                tokio::select! {
                    biased;
                    _ = &mut shutdown => break,
                    accepted = listener.accept() => match accepted {
                        Ok((socket, peer)) => {
                            let _ = socket.set_nodelay(true);
                            let dispatch = dispatch.clone();
                            // peer plumbing: the convenience
                            // `serve_h1_connection` builds its own
                            // RequestContext without peer for now.
                            // Once HttpListenerSpec carries peer the
                            // h1 sibling lights it up the same way
                            // HttpListenProtocol does.
                            let _peer = PeerInfo::Tcp(peer);
                            let runtime_for_conn = runtime_for_conns.clone();
                            let conn_future = async move {
                                if let Err(error) = serve_h1_connection(
                                    socket.compat(),
                                    dispatch,
                                    max_body_bytes,
                                    runtime_for_conn,
                                )
                                .await
                                {
                                    warn!(?error, "h1 connection error");
                                }
                            };
                            match &runtime_for_conns {
                                Some(rt) => rt.spawn_on_current_core(Box::pin(conn_future)),
                                None => {
                                    tokio::task::spawn_local(conn_future);
                                }
                            }
                        }
                        Err(error) => warn!(?error, "h1 accept failed"),
                    }
                }
            }
            Ok(())
        })
    }
}
