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
//! calls the connection pipe with a `CONNECT` request and invokes the
//! returned `Response.upgrade` against the socket wrapped as a
//! `HijackedSocket`. The query engine is the `PipeHandle` supplied to
//! [`PgWireListenProtocol::new`]; the registry's `dispatch` is used as the
//! query pipe only when the constructor did not set one (so a bare
//! `App`-mounted pgwire listener still routes to the app's dispatch).
//!
//! Without an acceptor factory the serve call fails with a config error
//! unless the `tokio-compat` feature provides the legacy tokio listener
//! path (off by default; prime/proxima-stream is the first-class path).

use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;

use futures::channel::oneshot;
use serde_json::Value;

#[cfg(feature = "tokio-compat")]
use std::io;
#[cfg(feature = "tokio-compat")]
use tracing::{debug, warn};

use proxima_core::ProximaError;
use proxima_listen::{ListenProtocol, ServeContext};
use proxima_primitives::pipe::handler::{PipeHandle, into_handle};
use proxima_runtime::Runtime;

use crate::pipes::PgPipeHandle;
use proxima_primitives::stream::TcpBindOptions;

#[cfg(feature = "tokio-compat")]
use bytes::Bytes;
#[cfg(feature = "tokio-compat")]
use futures::FutureExt;
#[cfg(feature = "tokio-compat")]
use proxima_primitives::pipe::SendPipe;
#[cfg(feature = "tokio-compat")]
use proxima_primitives::pipe::header_list::HeaderList;
#[cfg(feature = "tokio-compat")]
use proxima_primitives::pipe::request::{Request, RequestContext};
#[cfg(feature = "tokio-compat")]
use proxima_primitives::pipe::upgrade::{HijackStream, HijackedSocket};
#[cfg(feature = "tokio-compat")]
use proxima_primitives::stream::StreamConnection;

use crate::auth::PgAuth;
use crate::config::PgServerConfig;
use crate::connection::CancelRegistry;
use crate::pipe::PgWireConnectionPipe;

#[cfg(feature = "tls")]
type TlsAcceptor = futures_rustls::TlsAcceptor;
#[cfg(not(feature = "tls"))]
type TlsAcceptor = ();

/// PostgreSQL wire listener. Register on an `App` via
/// `with_listen_protocol`, or drive directly with
/// `ListenProtocolFluent::fluent()`. The `query` `PipeHandle` is the SQL
/// engine: a `Pipe` that matches on [`verb`] verbs and returns
/// [`crate::pipe_contract::PgReply`].
pub struct PgWireListenProtocol {
    label: String,
    query: Option<PgPipeHandle>,
    config: PgServerConfig,
    auth_override: Option<PgAuth>,
    registry: Arc<CancelRegistry>,
}

impl PgWireListenProtocol {
    #[must_use]
    pub fn new(label: impl Into<String>, query: PgPipeHandle) -> Self {
        Self {
            label: label.into(),
            query: Some(query),
            config: PgServerConfig::default(),
            auth_override: None,
            registry: Arc::new(CancelRegistry::new()),
        }
    }

    /// Mounts without a constructor-supplied engine: the registry's
    /// `dispatch` becomes the query pipe at serve time.
    #[must_use]
    pub fn from_dispatch(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            query: None,
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

fn resolve_config(base: &PgServerConfig, spec: &Value) -> Result<PgServerConfig, ProximaError> {
    match spec.get("pgwire") {
        None => Ok(base.clone()),
        Some(overrides) => serde_json::from_value(overrides.clone())
            .map_err(|error| ProximaError::Config(format!("pgwire spec: {error}"))),
    }
}

#[cfg(feature = "tls")]
fn resolve_tls(spec: &Value) -> Result<Option<TlsAcceptor>, ProximaError> {
    let config = proxima_tls::config_from_spec_value(spec.get(proxima_tls::SPEC_KEY))?;
    config
        .map(|config| proxima_tls::build_acceptor_futures_io(&config))
        .transpose()
}

#[cfg(not(feature = "tls"))]
fn resolve_tls(_spec: &Value) -> Result<Option<TlsAcceptor>, ProximaError> {
    Ok(None)
}

#[cfg(feature = "tokio-compat")]
fn connect_request() -> Request<Bytes> {
    Request {
        method: proxima_primitives::pipe::method::Method::Connect,
        path: Bytes::new(),
        query: HeaderList::new(),
        metadata: HeaderList::new(),
        payload: Bytes::new(),
        stream: None,
        context: RequestContext::default(),
    }
}

/// Drives one accepted socket through the connection pipe: build a
/// `CONNECT` request, take the returned upgrade, and invoke it against the
/// socket wrapped as a `HijackedSocket` (the accepted connection is
/// already `futures::io`, so it boxes straight to `Box<dyn HijackStream>`).
/// Mirrors `proxima_listen::serve_pipe_upgrades`'s per-connection step;
/// retained for the legacy tokio acceptor path that does not flow through
/// the factory helper.
#[cfg(feature = "tokio-compat")]
async fn handle_connection(
    conn: Box<dyn StreamConnection>,
    pipe: Arc<PgWireConnectionPipe>,
) -> Result<(), ProximaError> {
    let response = SendPipe::call(pipe.as_ref(), connect_request()).await?;
    let Some(handler) = response.upgrade else {
        return Err(ProximaError::Upstream(
            "pgwire connection pipe did not return an upgrade".into(),
        ));
    };
    let stream: Box<dyn HijackStream> = Box::new(conn);
    let hijacked = HijackedSocket::new(stream, Bytes::new());
    handler.invoke(hijacked).await
}

impl ListenProtocol for PgWireListenProtocol {
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
        let Some(query) = self.query.clone() else {
            return Box::pin(async move {
                Err(ProximaError::Config(
                    "pgwire listener requires a typed PgPipeHandle query engine; \
                     use PgWireListenProtocol::new to supply one"
                        .into(),
                ))
            });
        };
        let _ = dispatch;
        let runtime = context.runtime.clone();
        let connection_pipe = build_connection_pipe(
            &label,
            query,
            auth,
            config,
            &self.registry,
            tls,
            runtime.clone(),
        );
        let factory = context.acceptor_factory.clone();
        let ready_signal = context.ready_signal.clone();

        Box::pin(async move {
            let Some(factory) = factory else {
                return serve_legacy(bind, connection_pipe, label, shutdown, ready_signal).await;
            };
            let options = TcpBindOptions {
                reuseport: use_reuseport,
                ..TcpBindOptions::default()
            };
            let pipe: PipeHandle = into_handle(connection_pipe);
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
    pipe: Arc<PgWireConnectionPipe>,
    label: String,
    mut shutdown: oneshot::Receiver<()>,
    ready_signal: Option<std::sync::mpsc::Sender<()>>,
) -> Result<(), ProximaError> {
    use proxima_net::tokio::tokio_stream_listener::TokioTcpListener;
    use proxima_primitives::stream::StreamListenerExt;

    let listener = TokioTcpListener::bind(bind).await.map_err(|error| {
        ProximaError::Io(io::Error::other(format!("{label} bind {bind}: {error}")))
    })?;
    if let Some(sender) = ready_signal {
        let _ = sender.send(());
    }
    debug!(label = %label, %bind, "pgwire listener bound (legacy tokio)");
    loop {
        futures::select_biased! {
            _ = (&mut shutdown).fuse() => return Ok(()),
            accepted = listener.accept().fuse() => match accepted {
                Ok(conn) => {
                    let pipe = Arc::clone(&pipe);
                    let label = label.clone();
                    tokio::task::spawn_local(async move {
                        if let Err(error) = handle_connection(Box::new(conn), pipe).await {
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
    _pipe: Arc<PgWireConnectionPipe>,
    label: String,
    _shutdown: oneshot::Receiver<()>,
    _ready_signal: Option<std::sync::mpsc::Sender<()>>,
) -> Result<(), ProximaError> {
    Err(ProximaError::Config(format!(
        "{label}: pgwire listener needs a runtime-matched acceptor factory \
         (ServeContext::with_acceptor_factory); enable the tokio-compat \
         feature for the legacy tokio listener path"
    )))
}
