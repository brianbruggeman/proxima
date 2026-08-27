//! `PgWireListenProtocol` — mounts the pgwire connection `Pipe` into the
//! proxima listener registry.
//!
//! Composes the primitives directly: `proxima_listen::ListenProtocol`
//! for registry mounting, `ServeContext`'s runtime-matched
//! `AcceptorFactory` (`proxima_primitives::stream::TcpAcceptor`, prime or tokio
//! backing) for the accept loop, `proxima_tls::build_acceptor_futures_io`
//! for SSLRequest upgrades (TLS config rides the listener spec under
//! `proxima_tls::SPEC_KEY`, exactly like the HTTP listeners), and
//! [`crate::pipe::PgWireConnectionPipe`] for the per-connection drive.
//!
//! The connection is a real `Pipe`: on each accepted socket the listener
//! calls the connection pipe's `() -> UpgradeHandler` accept hook and
//! invokes the returned handler against the socket wrapped as a
//! `HijackedSocket`. The query engine is the [`PgPipeHandle`] supplied to
//! [`PgWireListenProtocol::new`] — the registry's untyped `dispatch` is
//! never a substitute, because a SQL engine is `QueryRequest -> PgReply`
//! and `dispatch` is the envelope-shaped `Request -> Response`.
//!
//! Without an acceptor factory the serve call fails with a config error
//! unless the `tokio-compat` feature provides the legacy tokio listener
//! path (off by default; prime over `proxima_primitives::stream` is the
//! first-class path).

use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;

use futures::channel::oneshot;
use serde_json::Value;

#[cfg(feature = "tokio-compat")]
use std::io;
#[cfg(feature = "tokio-compat")]
use proxima_telemetry::{debug, warn};

use proxima_core::ProximaError;
use proxima_listen::{ListenProtocol, ServeContext};
use proxima_primitives::pipe::alloc_tier;
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::pipe::upgrade::AcceptHandle;
use proxima_runtime::Runtime;

use crate::pipes::PgPipeHandle;
use proxima_primitives::stream::TcpBindOptions;

#[cfg(feature = "tokio-compat")]
use futures::FutureExt;

use crate::auth::PgAuth;
use crate::config::PgServerConfig;
use crate::connection::CancelRegistry;
use crate::pipe::PgWireConnectionPipe;
use crate::spec::{TlsAcceptor, resolve_config, resolve_tls};

/// PostgreSQL wire listener. Register on an `App` via
/// `with_listen_protocol`, or drive directly with
/// `ListenProtocolFluent::fluent()`. The `query` [`PgPipeHandle`] is the SQL
/// engine: a `Pipe` that matches on [`crate::pipe_contract::Verb`] and
/// returns [`crate::pipe_contract::PgReply`].
pub struct PgWireListenProtocol {
    label: String,
    query: PgPipeHandle,
    config: PgServerConfig,
    auth_override: Option<PgAuth>,
    registry: Arc<CancelRegistry>,
}

impl PgWireListenProtocol {
    #[must_use]
    pub fn new(label: impl Into<String>, query: PgPipeHandle) -> Self {
        Self {
            label: label.into(),
            query,
            config: PgServerConfig::default(),
            auth_override: None,
            registry: Arc::new(CancelRegistry::new()),
        }
    }

    /// Replaces the default [`PgServerConfig`]; a `pgwire` object in the
    /// listener spec still wins at serve time.
    #[must_use]
    pub fn with_config(mut self, config: PgServerConfig) -> Self {
        self.config = config;
        self
    }

    /// Installs an authentication policy directly (e.g. a custom
    /// [`crate::auth::PasswordVerifier`]), overriding the config's auth
    /// section.
    #[must_use]
    pub fn with_auth(mut self, auth: PgAuth) -> Self {
        self.auth_override = Some(auth);
        self
    }
}

impl ListenProtocol for PgWireListenProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    /// `_dispatch` is the registry's untyped `Request -> Response` handle;
    /// pgwire's engine is the typed `QueryRequest -> PgReply` pipe held on
    /// `self`, so there is nothing to fall back to.
    fn serve(
        &self,
        bind: SocketAddr,
        _dispatch: PipeHandle,
        spec: &Value,
        context: ServeContext,
        shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
        let label = self.label.clone();
        let config = match resolve_config(&self.config, spec) {
            Ok(config) => config,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        let auth = match &self.auth_override {
            Some(auth) => auth.clone(),
            None => match config.build_auth() {
                Ok(auth) => auth,
                Err(error) => {
                    return Box::pin(async move { Err(ProximaError::Config(error.to_string())) });
                }
            },
        };
        let tls = match resolve_tls(spec) {
            Ok(tls) => tls,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        let use_reuseport = spec
            .get(proxima_listen::handle::REUSEPORT_SPEC_KEY)
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let runtime = context.runtime.clone();
        let connection_pipe = build_connection_pipe(
            &label,
            self.query.clone(),
            auth,
            config,
            &self.registry,
            tls,
            runtime.clone(),
        );
        let factory = context.acceptor_factory.clone();
        let ready_signal = context.ready_signal.clone();
        let pipe: AcceptHandle = alloc_tier::into_handle(connection_pipe);

        Box::pin(async move {
            let Some(factory) = factory else {
                return serve_legacy(bind, pipe, label, shutdown, ready_signal).await;
            };
            let options = TcpBindOptions {
                reuseport: use_reuseport,
                ..TcpBindOptions::default()
            };
            proxima_listen::serve_pipe_upgrades(
                factory,
                bind,
                options,
                pipe,
                runtime,
                shutdown,
                &label,
                ready_signal,
            )
            .await
        })
    }
}

#[cfg(feature = "tls")]
fn build_connection_pipe(
    label: &str,
    query: PgPipeHandle,
    auth: PgAuth,
    config: PgServerConfig,
    registry: &Arc<CancelRegistry>,
    tls: Option<TlsAcceptor>,
    runtime: Option<Arc<dyn Runtime>>,
) -> Arc<PgWireConnectionPipe> {
    Arc::new(
        PgWireConnectionPipe::new(label, query, auth, Arc::new(config), Arc::clone(registry))
            .with_tls(tls)
            .with_runtime(runtime),
    )
}

#[cfg(not(feature = "tls"))]
fn build_connection_pipe(
    label: &str,
    query: PgPipeHandle,
    auth: PgAuth,
    config: PgServerConfig,
    registry: &Arc<CancelRegistry>,
    _tls: Option<TlsAcceptor>,
    runtime: Option<Arc<dyn Runtime>>,
) -> Arc<PgWireConnectionPipe> {
    Arc::new(
        PgWireConnectionPipe::new(label, query, auth, Arc::new(config), Arc::clone(registry))
            .with_runtime(runtime),
    )
}

#[cfg(feature = "tokio-compat")]
async fn serve_legacy(
    bind: SocketAddr,
    pipe: AcceptHandle,
    label: String,
    mut shutdown: oneshot::Receiver<()>,
    ready_signal: Option<proxima_listen::ReadySignal>,
) -> Result<(), ProximaError> {
    use proxima_net::tokio::tokio_stream_listener::TokioTcpListener;
    use proxima_primitives::stream::StreamListenerExt;

    let listener = TokioTcpListener::bind(bind).await.map_err(|error| {
        ProximaError::Io(io::Error::other(format!("{label} bind {bind}: {error}")))
    })?;
    if let Some(sender) = ready_signal {
        let _ = sender.send(Ok(()));
    }
    debug!(label = %label, %bind, "pgwire listener bound (legacy tokio)");
    loop {
        futures::select_biased! {
            _ = (&mut shutdown).fuse() => return Ok(()),
            accepted = listener.accept().fuse() => match accepted {
                Ok(conn) => {
                    let pipe = pipe.clone();
                    let label = label.clone();
                    tokio::task::spawn_local(async move {
                        if let Err(error) =
                            proxima_listen::serve_pipe::handle_connection(Box::new(conn), pipe)
                                .await
                        {
                            debug!(?error, label = %label, "pgwire connection ended");
                        }
                    });
                }
                Err(error) => warn!(?error, label = %label, "pgwire accept failed"),
            },
        }
    }
}

#[cfg(not(feature = "tokio-compat"))]
async fn serve_legacy(
    _bind: SocketAddr,
    _pipe: AcceptHandle,
    label: String,
    _shutdown: oneshot::Receiver<()>,
    _ready_signal: Option<proxima_listen::ReadySignal>,
) -> Result<(), ProximaError> {
    Err(ProximaError::Config(format!(
        "{label}: pgwire listener needs a runtime-matched acceptor factory \
         (ServeContext::with_acceptor_factory); enable the tokio-compat \
         feature for the legacy tokio listener path"
    )))
}
