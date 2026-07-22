//! HTTP listener: ALPN-multiplexed h1+h2 on the same TLS listener, with
//! optional HAProxy PROXY-protocol prelude, UDS, and SO_REUSEPORT per-core
//! lanes.
//!
//! # A listener is not a pipe — it runs one
//!
//! Everything in proxima that does work is a **pipe**: one async function,
//! `call`, that takes an `In` and returns a `Result<Out, Err>`. The trait is
//! [`Pipe`](proxima_primitives::pipe::Pipe), and its docs name the four
//! forms it can take, chosen entirely by what you pick for `In` and `Out`:
//! **transform** (`In -> Out`), **source** (`() -> Out`), **sink**
//! (`In -> ()`), **observe** (`In -> In`).
//!
//! A listener is none of them. It is the thing that *drives* a pipe: it owns
//! the socket, accepts connections, parses bytes into a request, calls your
//! pipe, and writes the answer back. The form is in the seam:
//!
//! ```text
//!    socket bytes                                        socket bytes
//!         │                                                    ▲
//!         ▼                                                    │
//!   ┌──────────────────────────────────────────────────────────────┐
//!   │ HttpListenProtocol                                           │
//!   │                                                              │
//!   │   accept ─► parse ─►  [ your pipe ]  ─► write ─► keep-alive  │
//!   │                    Request<Bytes>                            │
//!   │                          │                                   │
//!   │                          ▼                                   │
//!   │                    Response<Bytes>                           │
//!   └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! The pipe in the middle is a **transform**: a
//! [`SendPipe`](proxima_primitives::pipe::SendPipe) with
//! `In = `[`Request<Bytes>`](proxima_primitives::pipe::Request),
//! `Out = `[`Response<Bytes>`](proxima_primitives::pipe::Response),
//! `Err = `[`ProximaError`](proxima_core::ProximaError). Write those three
//! types and you have written everything a listener needs.
//!
//! [`serve`](proxima_listen::ListenProtocol::serve) takes that pipe as a
//! [`PipeHandle`](proxima_primitives::pipe::handler::PipeHandle) — the
//! *erased* form. A listener holds a pipe whose concrete type it cannot
//! know (it was written later, by you), so
//! [`into_handle`](proxima_primitives::pipe::handler::into_handle) puts it
//! behind a pointer. `PipeHandle` is what the pipe becomes once the compiler
//! is no longer allowed to know which one it is. Nothing else changes: it is
//! still the same transform, still callable.
//!
//! # Why this matters more than it looks
//!
//! [`H1ClientUpstream`](crate::http1::H1ClientUpstream) — the HTTP/1.1
//! *client* — is a `SendPipe` with those exact three types. So the client is
//! already, with no adapter, something this listener can dispatch into.
//!
//! A listener with a client mounted underneath it is a reverse proxy. No
//! proxy type exists in this crate, and none needs to: the thing that
//! answers requests and the thing that makes them are the same form, so one
//! composes into the other. That is what "everything is a pipe" buys.
//!
//! # Config as composition: the registry
//!
//! The reason [`serve`](proxima_listen::ListenProtocol::serve) takes an
//! untyped JSON `spec` instead of a typed config struct only makes sense
//! once you see the registry, so here it is.
//!
//! A [`ListenRegistry`](proxima_listen::ListenRegistry) is a map from a
//! **name** to a compiled listener: `Arc<dyn ListenProtocol>`, keyed by
//! whatever that protocol's [`name`](proxima_listen::ListenProtocol::name)
//! returns. `HttpListenProtocol::new()` names itself `"http"`. Registration
//! is the moment a piece of compiled Rust becomes reachable by a string.
//!
//! That string is the seam. A listener in config is just:
//!
//! ```toml
//! bind     = "0.0.0.0:8080"
//! protocol = "http"          # <- a registry key, resolved at startup
//! [spec]
//! max_body_bytes = 65536
//! ```
//!
//! which is the config form of `proxima`'s `RunConfig { bind, protocol, spec }`.
//! At startup the registry turns `protocol` back into the code, and hands it
//! `spec` and `bind`. So:
//!
//! - **a new listener** — a different port, a different spec, a different
//!   protocol, ten of them instead of one — is **config**. No recompile.
//! - **a new kind of listener** is code: implement
//!   [`ListenProtocol`](proxima_listen::ListenProtocol), `register` it once,
//!   and from then on it is reachable from config by name like every other.
//!
//! The compiled set grows only for a genuinely new protocol. Everything you
//! can express by *arranging* the existing ones is config. That is why the
//! `spec` is untyped: the registry cannot know the config type of a protocol
//! that was written after it, so the protocol parses its own `spec`. The
//! trade is real — a typo in a spec key is not a compile error — and it buys
//! the ability to add a listener kind without touching the registry.
//!
//! Registration and lookup need no socket, so this runs:
//!
//! ```
//! use std::sync::Arc;
//!
//! use proxima_http::listener::HttpListenProtocol;
//! use proxima_listen::{ListenProtocol, ListenRegistry};
//!
//! let registry = ListenRegistry::new();
//!
//! // compiled code goes in, keyed by its own name.
//! registry.register(Arc::new(HttpListenProtocol::new())).unwrap();
//! assert_eq!(registry.names(), vec!["http".to_string()]);
//!
//! // a config carrying `protocol = "http"` resolves back to that code.
//! let from_config = "http";
//! let protocol = registry.get(from_config).unwrap();
//! assert_eq!(protocol.name(), "http");
//!
//! // names are unique: registering the same key twice is an error, not a
//! // silent overwrite, so a config can never resolve ambiguously.
//! assert!(registry.register(Arc::new(HttpListenProtocol::new())).is_err());
//!
//! // and a config naming something nobody registered fails loudly at
//! // startup rather than at the first request.
//! assert!(registry.get("gopher").is_err());
//! ```
//!
//! # Creating one
//!
//! [`HttpListenProtocol`](crate::listener::HttpListenProtocol) has the worked example. Two knobs reach a running
//! listener, and they are not the same thing:
//!
//! - the per-listener JSON `spec` passed to [`serve`](proxima_listen::ListenProtocol::serve)
//!   — the only source that changes a *running* listener's policy today;
//! - [`HttpListenerConfig`](crate::listener::HttpListenerConfig), the typed config surface, whose defaults come
//!   from the [`sized`](crate::listener::sized) build-time constants. Read its docs before reaching
//!   for it: what it does and does not currently reach is documented there,
//!   and it is narrower than the type suggests.
//!
//! # Where the code lives
//!
//! Drives [`crate::http1::serve_connection`] for h1 dispatch and
//! `crate::http2::serve_h2_connection` for h2. The Linux + `io-uring` accept
//! variant lives in the umbrella (`listeners/http_uring.rs`); this module
//! ships the default-tokio accept loop.
//!
//! Folded from the former `proxima-listeners-http` satellite crate into
//! `proxima-http` as the `listener` module (Cargo feature
//! `http-listener`), since it drives this crate's own `http1`/`http2`
//! stacks and already depended on it directly.

use std::future::Future;
use std::future::poll_fn;
use std::net::SocketAddr;
#[cfg(feature = "http1")]
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use futures::FutureExt;
use futures::channel::mpsc;
use futures::channel::oneshot;
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite};
use futures::stream::StreamExt;
use proxima_core::io::{FromFutures, IntoFutures, Prepend};
#[cfg(feature = "http1")]
use proxima_core::io::{FromTokio, IntoTokio};
use serde_json::Value;
#[cfg(feature = "tls")]
use tokio_util::compat::FuturesAsyncReadCompatExt;
// tokio-io bridge for the legacy no-`AcceptorFactory` accept loop only —
// `serve_via_factory`/`dispatch_h1_or_h2` stay on `FromFutures`/`IntoFutures`
// (no tokio) regardless of this feature.
#[cfg(feature = "http1")]
use tokio_util::compat::TokioAsyncReadCompatExt;

#[cfg(feature = "http1")]
use tokio::net::TcpListener;
#[cfg(feature = "http1")]
use tokio::net::TcpStream as CompatTcpStream;
use tracing::{debug, warn};

use proxima_core::time::{sleep, sleep_until};

use proxima_core::ProximaError;
use crate::http1::serve::serve_connection;
use proxima_listen::{
    Admission, ConnectionHandle, DrainOutcome, ListenProtocol, ListenerCore, Route,
};
#[cfg(feature = "http1")]
use proxima_listen::DispatchPolicy;
use proxima_primitives::pipe::handler::PipeHandle;

// Re-exports so the umbrella's `pub use listeners::*` keeps working.
pub use crate::http1::serve::{HttpListenerSpec, serve_h1_connection};

mod config;
pub use config::{HttpListenerConfig, HttpListenerLayerBuilder};

/// Build-time sizing constants generated from `proxima-listeners-http.toml`.
/// At no_std+no_alloc (once this crate lifts off its std-only deps) these
/// consts ARE the config; at std they seed [`HttpListenerConfig`]'s runtime
/// defaults — never duplicated.
pub mod sized {
    include!(concat!(env!("OUT_DIR"), "/proxima_listeners_http_sized.rs"));
}

/// The HTTP listener. Accepts connections, turns bytes into a
/// [`Request<Bytes>`](proxima_primitives::pipe::Request), calls the
/// [`PipeHandle`] it was given, and writes the
/// [`Response<Bytes>`](proxima_primitives::pipe::Response) back. See the
/// [module docs](crate::listener) for why the thing in the middle is a
/// transform.
///
/// # Building the pipe it drives
///
/// This half needs no socket, so it is shown as a real, running example.
/// Write a [`SendPipe`](proxima_primitives::pipe::SendPipe) whose `In` is
/// `Request<Bytes>`, `Out` is `Response<Bytes>`, and `Err` is
/// [`ProximaError`], then
/// [`into_handle`](proxima_primitives::pipe::handler::into_handle) erases it
/// into the [`PipeHandle`] a listener dispatches into.
///
/// ```
/// use std::future::Future;
///
/// use bytes::Bytes;
/// use proxima_http::listener::HttpListenProtocol;
/// use proxima_listen::ListenProtocol;
/// use proxima_core::ProximaError;
/// use proxima_primitives::pipe::handler::into_handle;
/// use proxima_primitives::pipe::{Request, Response, SendPipe};
///
/// struct Greet;
///
/// impl SendPipe for Greet {
///     type In = Request<Bytes>;
///     type Out = Response<Bytes>;
///     type Err = ProximaError;
///
///     fn call(
///         &self,
///         request: Request<Bytes>,
///     ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
///         async move {
///             let path = String::from_utf8_lossy(&request.path).into_owned();
///             Ok(Response::ok(Bytes::from(format!("you asked for {path}"))))
///         }
///     }
/// }
///
/// // Erase it into the form a listener can hold. Nothing was added to make
/// // `Greet` eligible — being the right transform IS the eligibility.
/// let handle = into_handle(Greet);
///
/// // A `PipeHandle` is still callable. No socket has been opened; the
/// // handler is the same object the listener would drive.
/// let request = Request::builder().method("GET").path("/hello").build().unwrap();
/// let response = futures::executor::block_on(handle.call(request)).unwrap();
/// assert_eq!(response.status, 200);
/// assert_eq!(&response.payload[..], b"you asked for /hello");
///
/// // The listener itself constructs with no arguments and no I/O.
/// let protocol = HttpListenProtocol::new();
/// assert_eq!(protocol.name(), "http");
/// ```
///
/// # Serving it
///
/// [`serve`](proxima_listen::ListenProtocol::serve) is positional. The fluent surface is
/// [`fluent`](proxima_listen::ListenProtocolFluent::fluent), which is
/// blanket-implemented for every [`ListenProtocol`] and defaults every knob
/// it is not given — including the dispatch, which defaults to a handler
/// returning 503 so that forgetting to wire one up is loud rather than
/// silent.
///
/// This example is `no_run`: it is compiled and type-checked, but not
/// executed, because it binds a real TCP socket and then accepts until
/// shutdown. There is no in-memory listener transport to substitute, and
/// with `bind` port 0 there is no way to learn which port was chosen.
///
/// ```no_run
/// use proxima_http::listener::HttpListenProtocol;
/// use proxima_listen::ListenProtocolFluent;
/// # use proxima_primitives::pipe::handler::PipeHandle;
/// # async fn example(handle: PipeHandle) -> Result<(), proxima_core::ProximaError> {
/// HttpListenProtocol::new()
///     .fluent()
///     .bind("127.0.0.1:8080".parse().unwrap())
///     .dispatch(handle)
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct HttpListenProtocol {
    label: String,
}

impl Default for HttpListenProtocol {
    fn default() -> Self {
        Self {
            label: "http".into(),
        }
    }
}

impl HttpListenProtocol {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl ListenProtocol for HttpListenProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    fn serve(
        &self,
        bind: SocketAddr,
        dispatch: PipeHandle,
        spec: &Value,
        context: proxima_listen::ServeContext,
        shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
        // The Linux + `io-uring` path (`listeners/http_uring.rs`) stays
        // in the umbrella for now — it pulls prime's io_uring TcpListener
        // via `io_uring_compat` and can't extract cleanly until that
        // selector moves into a shared crate. The default-tokio path
        // below is what this module ships.
        self.serve_default(bind, dispatch, spec, context, shutdown)
    }
}

impl HttpListenProtocol {
    fn serve_default(
        &self,
        bind: SocketAddr,
        dispatch: PipeHandle,
        spec: &Value,
        context: proxima_listen::ServeContext,
        shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
        // UDS dispatch: `spec.path` set → bind a Unix domain socket
        // instead of TCP. Skips TLS/SO_REUSEPORT/TCP_NODELAY (none
        // apply on UDS) and the connection-level telemetry that
        // requires SocketAddr labels. h2 prior-knowledge dispatch on
        // UDS is a follow-on (preface sniff); today h1 only over UDS.
        if let Some(path) = spec.get("path").and_then(Value::as_str) {
            // UDS has no `AcceptorFactory`-driven arm yet — it always
            // rode the legacy tokio `UnixListener` path, so it stays
            // behind `http1` (the tokio-coupled legacy feature) rather
            // than gaining a half-built tokio-free stand-in here.
            #[cfg(feature = "http1")]
            {
                let path_buf = PathBuf::from(path);
                let mode = spec
                    .get("mode")
                    .and_then(Value::as_u64)
                    .map(|raw| raw as u32);
                let max_body_bytes = spec
                    .get("max_body_bytes")
                    .and_then(Value::as_u64)
                    .map(|raw| raw as usize);
                return Box::pin(serve_default_uds(
                    path_buf,
                    mode,
                    dispatch,
                    max_body_bytes,
                    context.ready_signal.clone(),
                    shutdown,
                ));
            }
            #[cfg(not(feature = "http1"))]
            {
                let _ = path;
                return Box::pin(async move {
                    Err(ProximaError::Config(
                        "http listener UDS bind requires the `http1` feature (no tokio-free UDS driver yet)"
                            .into(),
                    ))
                });
            }
        }
        let defaults = HttpListenerConfig::default();
        let max_body_bytes = spec
            .get("max_body_bytes")
            .and_then(Value::as_u64)
            .map(|raw| raw as usize);
        let drain_timeout_ms = spec
            .get("drain_timeout_ms")
            .and_then(Value::as_u64)
            .unwrap_or(defaults.drain_timeout_ms);
        let quiesce_duration_ms = spec.get("quiesce_duration_ms").and_then(Value::as_u64);
        let quiesce_status = spec
            .get("quiesce_status")
            .and_then(Value::as_u64)
            .map(|raw| raw as u16)
            .unwrap_or(defaults.quiesce_status);
        let quiesce_retry_after = spec
            .get("quiesce_retry_after")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or(defaults.quiesce_retry_after);
        let listener_spec = Arc::new(HttpListenerSpec { max_body_bytes });
        let in_flight = Arc::new(AtomicU64::new(0));
        let quiescing = Arc::new(AtomicBool::new(false));
        let quiesce_response = Arc::new(QuiesceResponse {
            status: quiesce_status,
            retry_after: quiesce_retry_after,
        });
        let connections_accepted = Arc::new(AtomicU64::new(0));
        let connections_active = Arc::new(AtomicU64::new(0));
        let listener_label: Arc<[u8]> = Arc::from(
            spec.get("name")
                .and_then(Value::as_str)
                .unwrap_or(defaults.name.as_str())
                .as_bytes(),
        );
        let context_for_close = context.clone_telemetry();
        let runtime_for_conns = context.runtime.clone();
        let handler_dispatch_for_conn = context.handler_dispatch;
        // Offload wraps the served Pipe ONCE, here at serve start — never
        // per request. `SpreadToPeers` isolates a synchronously-blocking
        // handler by running its `Pipe::call` on the runtime's background
        // pool instead of the executor thread driving the connection; see
        // `Offload`'s docs for the full contract. `handler_dispatch_for_conn`
        // below still separately decides which core drives the connection
        // FUTURE via `dispatch_handler` — that per-connection concern is
        // unrelated to this per-dispatch one.
        let dispatch = match (handler_dispatch_for_conn, runtime_for_conns.as_ref()) {
            (proxima_listen::HandlerDispatch::SpreadToPeers { .. }, Some(runtime)) => {
                proxima_primitives::pipe::handler::into_handle(proxima_listen::Offload::new(
                    dispatch,
                    runtime.clone(),
                ))
            }
            _ => dispatch,
        };
        // TLS termination is opt-in via `__proxima_tls` in the spec (set by
        // `Listener::run_with_runtime` when `with_tls(...)` was called).
        // Build the acceptor once at listener start so a bad cert / key
        // surfaces here instead of on the first connection.
        #[cfg(feature = "tls")]
        let tls_acceptor: Option<tokio_rustls::TlsAcceptor> =
            match proxima_tls::config_from_spec_value(spec.get(proxima_tls::SPEC_KEY)) {
                Ok(Some(config)) => match proxima_tls::build_acceptor(&config) {
                    Ok(acceptor) => Some(acceptor),
                    Err(error) => return Box::pin(async move { Err(error) }),
                },
                Ok(None) => None,
                Err(error) => return Box::pin(async move { Err(error) }),
            };
        // SO_REUSEPORT-aware bind: when the spec carries the marker
        // flag (set by `Listener::run_with_runtime` for per-core
        // dispatch), construct the socket via socket2 with
        // SO_REUSEADDR + SO_REUSEPORT before binding. Otherwise fall
        // back to plain `TcpListener::bind` (single-core path,
        // mainly tests).
        let use_reuseport = spec
            .get(proxima_listen::handle::REUSEPORT_SPEC_KEY)
            .and_then(Value::as_bool)
            .unwrap_or(false);
        // TCP Fast Open (RFC 7413). spec key carries the kernel's
        // passive-open queue depth on Linux; non-Linux platforms
        // accept the spec but it's a no-op. Operators also need
        // `sysctl net.ipv4.tcp_fastopen=2` for the kernel to
        // advertise TFO.
        let tcp_fastopen_queue = spec
            .get("tcp_fastopen")
            .and_then(Value::as_u64)
            .map(|raw| raw as u32);
        // PROXY protocol policy. `true` requires the header on every
        // accepted TCP connection (deployment behind AWS NLB / GCP
        // L4 / haproxy). When set, proxima reads + discards the v1
        // or v2 PROXY header before TLS / h1 sees a byte.
        let proxy_protocol_enabled = spec
            .get("proxy_protocol")
            .and_then(Value::as_bool)
            .unwrap_or(defaults.proxy_protocol_enabled);
        // futures-io serve path: when a runtime-matched acceptor factory
        // is injected, bind + accept through it (prime or tokio backing)
        // instead of the tokio `TcpListener` below. The legacy path stays
        // byte-identical when no factory is present.
        if let Some(factory) = context.acceptor_factory.clone() {
            let telemetry = context.telemetry.clone();
            let runtime = context.runtime.clone();
            let handler_dispatch = context.handler_dispatch;
            let ready_signal = context.ready_signal.clone();
            return Box::pin(serve_via_factory(
                factory,
                bind,
                dispatch,
                listener_spec,
                in_flight,
                quiescing,
                quiesce_response,
                connections_accepted,
                connections_active,
                listener_label,
                telemetry,
                runtime,
                handler_dispatch,
                use_reuseport,
                tcp_fastopen_queue,
                proxy_protocol_enabled,
                drain_timeout_ms,
                quiesce_duration_ms,
                #[cfg(feature = "tls")]
                tls_acceptor,
                ready_signal,
                shutdown,
            ));
        }
        // Legacy no-`AcceptorFactory` accept loop: tokio `TcpListener` +
        // `tokio::select!`. Gated behind `http1` (the tokio-coupled
        // legacy feature) — the tokio-free `http1-native` build only
        // reaches the factory branch above; see the `#[cfg(not(feature
        // = "http1"))]` arm below for that build's fallback.
        #[cfg(feature = "http1")]
        {
            // Local `mut` shadow: the outer `shutdown` parameter stays
            // immutable so the tokio-free build (which never takes
            // `&mut shutdown`) doesn't trip `unused_mut`.
            let mut shutdown = shutdown;
            Box::pin(async move {
                let listener = if use_reuseport {
                proxima_listen::handle::bind_reuseport_listener_with_options(
                    bind,
                    tcp_fastopen_queue,
                )
                .map_err(ProximaError::Io)?
            } else if let Some(queue) = tcp_fastopen_queue {
                // Plain (non-reuseport) bind also wants TFO — route
                // through the socket2 path that exposes the option.
                proxima_listen::handle::bind_reuseport_listener_with_options(bind, Some(queue))
                    .map_err(ProximaError::Io)?
            } else {
                TcpListener::bind(bind).await.map_err(ProximaError::Io)?
            };
            debug!(
                ?bind,
                use_reuseport,
                ?tcp_fastopen_queue,
                "http listener bound"
            );
            if let Some(sender) = context.ready_signal.clone() {
                let _ = sender.send(());
            }
            #[cfg(feature = "tls")]
            if tls_acceptor.is_some() {
                debug!(?bind, "http listener terminating TLS");
            }
            let mut core = ListenerCore::new(handler_dispatch_for_conn.as_policy());
            let (release_tx, mut release_rx) = mpsc::unbounded::<ConnectionHandle>();
            let spawn_handler =
                |socket: CompatTcpStream,
                 raw_peer: SocketAddr,
                 handle: ConnectionHandle,
                 route: Route,
                 release_tx: mpsc::UnboundedSender<ConnectionHandle>| {
                    let dispatch_for_conn = dispatch.clone();
                    let spec_for_conn = listener_spec.clone();
                    let in_flight_for_conn = in_flight.clone();
                    let quiescing_for_conn = quiescing.clone();
                    let quiesce_response_for_conn = quiesce_response.clone();
                    let connections_active_for_conn = connections_active.clone();
                    let telemetry_for_conn = context_for_close.clone();
                    let listener_label_for_conn = Arc::clone(&listener_label);
                    let runtime_for_conn = runtime_for_conns.clone();
                    #[cfg(feature = "tls")]
                    let acceptor_for_conn = tls_acceptor.clone();
                    let conn_future = async move {
                        // PROXY protocol: when enabled, consume the v1/v2
                        // header off the wire before TLS / h1 sees any
                        // bytes, then wrap any over-read application bytes
                        // back onto the stream so downstream readers don't
                        // miss them. When disabled the wrapper is empty
                        // and the cost is one extra branch per read.
                        let mut raw_socket = socket;
                        let mut peer_info: Option<proxima_primitives::stream::PeerInfo> =
                            Some(proxima_primitives::stream::PeerInfo::Tcp(raw_peer));
                        let leftover: Vec<u8> = if proxy_protocol_enabled {
                            match proxima_protocols::proxy_protocol::read_header_tokio(&mut raw_socket).await {
                                Ok((header, leftover)) => {
                                    tracing::debug!(?header, "proxy protocol header parsed");
                                    if let proxima_protocols::proxy_protocol::ProxyHeader::Tcp {
                                        src, ..
                                    } = header
                                    {
                                        peer_info = Some(proxima_primitives::stream::PeerInfo::Tcp(src));
                                    }
                                    leftover
                                }
                                Err(error) => {
                                    warn!(?error, "proxy protocol header rejected");
                                    return;
                                }
                            }
                        } else {
                            Vec::new()
                        };
                        let socket = IntoTokio(Prepend::new(leftover, FromTokio(raw_socket)));
                        // Wrap the raw TCP socket with the TLS acceptor when
                        // configured; the result implements AsyncRead+AsyncWrite
                        // and feeds hyper identically to the plaintext path.
                        #[cfg(feature = "tls")]
                        let outcome = match acceptor_for_conn {
                            Some(acceptor) => match acceptor.accept(socket).await {
                                Ok(tls_stream) => {
                                    // ALPN dispatch: if the handshake negotiated h2
                                    // AND the http2 feature is enabled, hand the
                                    // stream to the h2 driver; otherwise the existing
                                    // native h1 path runs.
                                    #[cfg(feature = "http2-native")]
                                    let negotiated_h2 = tls_stream
                                        .get_ref()
                                        .1
                                        .alpn_protocol()
                                        .map(|alpn| alpn == b"h2")
                                        .unwrap_or(false);
                                    #[cfg(not(feature = "http2-native"))]
                                    let negotiated_h2 = false;
                                    if negotiated_h2 {
                                        #[cfg(feature = "http2-native")]
                                        {
                                            crate::http2::serve_h2_connection(
                                                tls_stream.compat(),
                                                dispatch_for_conn,
                                                in_flight_for_conn,
                                                quiesce_response_for_conn,
                                                peer_info.clone(),
                                            )
                                            .await
                                        }
                                        #[cfg(not(feature = "http2-native"))]
                                        {
                                            unreachable!(
                                                "negotiated_h2 is false without http2 feature"
                                            );
                                        }
                                    } else {
                                        serve_connection(
                                            tls_stream.compat(),
                                            dispatch_for_conn,
                                            spec_for_conn,
                                            in_flight_for_conn,
                                            quiescing_for_conn,
                                            quiesce_response_for_conn,
                                            peer_info.clone(),
                                            runtime_for_conn.clone(),
                                        )
                                        .await
                                    }
                                }
                                Err(error) => {
                                    Err(ProximaError::Upstream(format!("tls handshake: {error}")))
                                }
                            },
                            None => {
                                // Plain TCP, no TLS, no ALPN — sniff the
                                // first bytes to dispatch h1 or h2 prior-
                                // knowledge.
                                dispatch_h1_or_h2(
                                    socket.compat(),
                                    dispatch_for_conn,
                                    spec_for_conn,
                                    in_flight_for_conn,
                                    quiescing_for_conn,
                                    quiesce_response_for_conn,
                                    peer_info.clone(),
                                    runtime_for_conn.clone(),
                                )
                                .await
                            }
                        };
                        #[cfg(not(feature = "tls"))]
                        let outcome = dispatch_h1_or_h2(
                            socket.compat(),
                            dispatch_for_conn,
                            spec_for_conn,
                            in_flight_for_conn,
                            quiescing_for_conn,
                            quiesce_response_for_conn,
                            peer_info.clone(),
                            runtime_for_conn.clone(),
                        )
                        .await;
                        if let Err(error) = outcome {
                            warn!(?error, "connection error");
                        }
                        let active_after =
                            connections_active_for_conn.fetch_sub(1, Ordering::Relaxed) - 1;
                        telemetry_for_conn.gauge_set(
                            "proxima.connections.active",
                            &proxima_primitives::pipe::telemetry_surface::Labels::from_pairs(&[(
                                "listener",
                                std::str::from_utf8(&listener_label_for_conn).unwrap_or(""),
                            )]),
                            active_after as i64,
                        );
                        // release the admission slot so the listener can drain / re-admit.
                        let _ = release_tx.unbounded_send(handle);
                    };
                    proxima_listen::dispatch_handler(
                        runtime_for_conns.as_ref(),
                        route,
                        Box::pin(conn_future),
                    );
                };
            let on_accept =
                |socket: CompatTcpStream,
                 peer: SocketAddr,
                 handle: ConnectionHandle,
                 route: Route,
                 release_tx: mpsc::UnboundedSender<ConnectionHandle>| {
                    // TCP_NODELAY: disable Nagle on the accepted socket so
                    // small responses (typical request/response sizes) go
                    // out immediately. Without this, Nagle + Linux's
                    // delayed-ack adds 40-200 ms per round trip when the
                    // client is also small-write — measured 24 rps on
                    // host-b (Linux) before this fix; macOS's less
                    // aggressive Nagle masked it.
                    let _ = socket.set_nodelay(true);
                    connections_accepted.fetch_add(1, Ordering::Relaxed);
                    let active_now = connections_active.fetch_add(1, Ordering::Relaxed) + 1;
                    debug!(?peer, "accepted connection");
                    let listener_labels = proxima_primitives::pipe::telemetry_surface::Labels::from_pairs(&[(
                        "listener",
                        std::str::from_utf8(&listener_label).unwrap_or(""),
                    )]);
                    context.telemetry.counter_inc(
                        "proxima.connections_accepted_total",
                        &listener_labels,
                        1,
                    );
                    context.telemetry.gauge_set(
                        "proxima.connections.active",
                        &listener_labels,
                        active_now as i64,
                    );
                    spawn_handler(socket, peer, handle, route, release_tx);
                };
            loop {
                tokio::select! {
                    _ = &mut shutdown => break,
                    released = release_rx.next() => if let Some(handle) = released {
                        core.release(handle);
                    },
                    accepted = listener.accept() => match accepted {
                        Ok((socket, peer)) => match core.admit(peer.ip()) {
                            Admission::Admit { handle, route } => {
                                on_accept(socket, peer, handle, route, release_tx.clone());
                            }
                            Admission::Shed { reason } => {
                                debug!(?reason, "http connection shed");
                                drop(socket);
                            }
                        },
                        Err(error) => warn!(?error, "accept failed"),
                    },
                }
            }
            if let Some(quiesce_ms) = quiesce_duration_ms
                && quiesce_ms > 0
            {
                quiescing.store(true, Ordering::Relaxed);
                debug!(quiesce_ms, "http listener entering quiesce window");
                let deadline = proxima_core::time::now() + std::time::Duration::from_millis(quiesce_ms);
                loop {
                    tokio::select! {
                        _ = sleep_until(deadline) => break,
                        released = release_rx.next() => if let Some(handle) = released {
                            core.release(handle);
                        },
                        accepted = listener.accept() => match accepted {
                            Ok((socket, peer)) => match core.admit(peer.ip()) {
                                Admission::Admit { handle, route } => {
                                    on_accept(socket, peer, handle, route, release_tx.clone());
                                }
                                Admission::Shed { reason } => {
                                    debug!(?reason, "http connection shed during quiesce");
                                    drop(socket);
                                }
                            },
                            Err(error) => warn!(?error, "accept during quiesce failed"),
                        },
                    }
                }
            }
            debug!(
                in_flight = in_flight.load(Ordering::Relaxed),
                "http listener draining"
            );
            drain_in_flight(
                &in_flight,
                std::time::Duration::from_millis(drain_timeout_ms),
            )
            .await;
            if let DrainOutcome::Draining = core.begin_drain() {
                drain_connections(
                    &mut core,
                    &mut release_rx,
                    std::time::Duration::from_millis(drain_timeout_ms),
                )
                .await;
            }
            Ok(())
        })
        }
        // tokio-free default: no `AcceptorFactory` and no `http1` legacy
        // fallback compiled in — there's no bind path left. Restore the
        // legacy tokio accept loop with `--features http1` (mirrors
        // `H2ListenProtocol`'s non-native arm).
        #[cfg(not(feature = "http1"))]
        {
            let _ = (context_for_close, runtime_for_conns, handler_dispatch_for_conn);
            Box::pin(async move {
                Err(ProximaError::Config(
                    "http listener requires an AcceptorFactory (no `http1` tokio fallback in this build)"
                        .into(),
                ))
            })
        }
    }
}

/// UDS-bound accept loop. Minimal counterpart to the TCP path: no
/// TLS (UDS is already local-trusted), no SO_REUSEPORT (single-
/// process), no TCP_NODELAY (different transport). The daemon's
/// control plane is the primary user.
///
/// Process connections serially — UDS sees one operator at a time
/// (CLI invocations are short-lived round trips), serial processing
/// keeps the accept loop simple and avoids spawning. Each connection
/// goes through the preface sniff so h1 and h2 prior-knowledge
/// clients (no ALPN without TLS) work on the same socket.
///
/// Tokio-only (`tokio::net::UnixListener` + `tokio::select!`) — only
/// reachable from `serve_default`'s `http1`-gated UDS arm.
#[cfg(feature = "http1")]
async fn serve_default_uds(
    path: PathBuf,
    mode: Option<u32>,
    dispatch: PipeHandle,
    max_body_bytes: Option<usize>,
    ready_signal: Option<std::sync::mpsc::Sender<()>>,
    mut shutdown: oneshot::Receiver<()>,
) -> Result<(), ProximaError> {
    if path.exists() {
        std::fs::remove_file(&path).map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!(
                "remove stale uds socket: {err}"
            )))
        })?;
    }
    let listener = tokio::net::UnixListener::bind(&path)
        .map_err(|err| ProximaError::Io(std::io::Error::other(format!("bind uds: {err}"))))?;
    if let Some(perm_bits) = mode {
        use std::os::unix::fs::PermissionsExt;
        let permissions = std::fs::Permissions::from_mode(perm_bits);
        std::fs::set_permissions(&path, permissions)
            .map_err(|err| ProximaError::Io(std::io::Error::other(format!("chmod uds: {err}"))))?;
    }
    debug!(?path, "http listener (uds) bound");
    if let Some(sender) = ready_signal {
        let _ = sender.send(());
    }
    let mut core = ListenerCore::new(DispatchPolicy::Inline);
    loop {
        tokio::select! {
            outcome = listener.accept() => match outcome {
                Ok((socket, _peer)) => match core.admit(proxima_listen::peer_ip(None)) {
                    Admission::Admit { handle, .. } => {
                        let spec = Arc::new(HttpListenerSpec { max_body_bytes });
                        let in_flight = Arc::new(AtomicU64::new(0));
                        let quiescing = Arc::new(AtomicBool::new(false));
                        let quiesce_response = Arc::new(QuiesceResponse {
                            status: 503,
                            retry_after: "1".into(),
                        });
                        if let Err(error) = dispatch_h1_or_h2(
                            socket.compat(),
                            dispatch.clone(),
                            spec,
                            in_flight,
                            quiescing,
                            quiesce_response,
                            Some(proxima_primitives::stream::PeerInfo::Unix(None)),
                            None,
                        )
                        .await
                        {
                            warn!(?error, "uds connection error");
                        }
                        core.release(handle);
                    }
                    Admission::Shed { reason } => {
                        debug!(?reason, "uds connection shed");
                        drop(socket);
                    }
                },
                Err(error) => warn!(?error, "uds accept failed"),
            },
            _ = &mut shutdown => {
                core.begin_drain();
                let _ = std::fs::remove_file(&path);
                return Ok(());
            }
        }
    }
}

/// futures-io accept loop, mirroring `serve_default`'s legacy tokio
/// handler over an injected `AcceptorFactory`. Binds through the
/// factory (prime- or tokio-backed) and accepts boxed
/// `StreamConnection`s via `poll_accept`. The boxed acceptor already
/// sets TCP_NODELAY, so this path does not.
#[allow(clippy::too_many_arguments)]
async fn serve_via_factory(
    factory: Arc<dyn proxima_primitives::stream::AcceptorFactory>,
    bind: SocketAddr,
    dispatch: PipeHandle,
    listener_spec: Arc<HttpListenerSpec>,
    in_flight: Arc<AtomicU64>,
    quiescing: Arc<AtomicBool>,
    quiesce_response: Arc<QuiesceResponse>,
    connections_accepted: Arc<AtomicU64>,
    connections_active: Arc<AtomicU64>,
    listener_label: Arc<[u8]>,
    telemetry: proxima_primitives::pipe::telemetry_surface::TelemetryHandle,
    runtime: Option<Arc<dyn proxima_runtime::Runtime>>,
    handler_dispatch: proxima_listen::HandlerDispatch,
    use_reuseport: bool,
    tcp_fastopen_queue: Option<u32>,
    proxy_protocol_enabled: bool,
    drain_timeout_ms: u64,
    quiesce_duration_ms: Option<u64>,
    #[cfg(feature = "tls")] tls_acceptor: Option<tokio_rustls::TlsAcceptor>,
    ready_signal: Option<std::sync::mpsc::Sender<()>>,
    mut shutdown: oneshot::Receiver<()>,
) -> Result<(), ProximaError> {
    let options = proxima_primitives::stream::TcpBindOptions {
        backlog: proxima_primitives::stream::DEFAULT_LISTEN_BACKLOG,
        reuseport: use_reuseport,
        tcp_fastopen: tcp_fastopen_queue,
    };
    let mut acceptor = factory.bind(bind, options).map_err(ProximaError::Io)?;
    if let Some(sender) = ready_signal {
        let _ = sender.send(());
    }
    debug!(
        ?bind,
        use_reuseport,
        ?tcp_fastopen_queue,
        "http listener bound (factory)"
    );
    #[cfg(feature = "tls")]
    if tls_acceptor.is_some() {
        debug!(?bind, "http listener terminating tls (factory)");
    }
    let mut core = ListenerCore::new(handler_dispatch.as_policy());
    let (release_tx, mut release_rx) = mpsc::unbounded::<ConnectionHandle>();
    let spawn_handler =
        |conn: Box<dyn proxima_primitives::stream::StreamConnection>,
         handle: ConnectionHandle,
         route: Route,
         release_tx: mpsc::UnboundedSender<ConnectionHandle>| {
            let dispatch_for_conn = dispatch.clone();
            let spec_for_conn = listener_spec.clone();
            let in_flight_for_conn = in_flight.clone();
            let quiescing_for_conn = quiescing.clone();
            let quiesce_response_for_conn = quiesce_response.clone();
            let connections_active_for_conn = connections_active.clone();
            let telemetry_for_conn = telemetry.clone();
            let listener_label_for_conn = Arc::clone(&listener_label);
            let runtime_for_conn = runtime.clone();
            #[cfg(feature = "tls")]
            let acceptor_for_conn = tls_acceptor.clone();
            let conn_future = async move {
                let mut raw_conn = conn;
                let mut peer_info: Option<proxima_primitives::stream::PeerInfo> = raw_conn.peer();
                let leftover: Vec<u8> = if proxy_protocol_enabled {
                    match proxima_protocols::proxy_protocol::read_header(&mut raw_conn).await {
                        Ok((header, leftover)) => {
                            tracing::debug!(?header, "proxy protocol header parsed");
                            if let proxima_protocols::proxy_protocol::ProxyHeader::Tcp { src, .. } = header {
                                peer_info = Some(proxima_primitives::stream::PeerInfo::Tcp(src));
                            }
                            leftover
                        }
                        Err(error) => {
                            warn!(?error, "proxy protocol header rejected");
                            return;
                        }
                    }
                } else {
                    Vec::new()
                };
                let stream = IntoFutures(Prepend::new(leftover, FromFutures(raw_conn)));
                #[cfg(feature = "tls")]
                let outcome = match acceptor_for_conn {
                    Some(acceptor) => match acceptor.accept(stream.compat()).await {
                        Ok(tls_stream) => {
                            #[cfg(feature = "http2-native")]
                            let negotiated_h2 = tls_stream
                                .get_ref()
                                .1
                                .alpn_protocol()
                                .map(|alpn| alpn == b"h2")
                                .unwrap_or(false);
                            #[cfg(not(feature = "http2-native"))]
                            let negotiated_h2 = false;
                            if negotiated_h2 {
                                #[cfg(feature = "http2-native")]
                                {
                                    crate::http2::serve_h2_connection(
                                        tls_stream.compat(),
                                        dispatch_for_conn,
                                        in_flight_for_conn,
                                        quiesce_response_for_conn,
                                        peer_info.clone(),
                                    )
                                    .await
                                }
                                #[cfg(not(feature = "http2-native"))]
                                {
                                    unreachable!("negotiated_h2 is false without http2 feature");
                                }
                            } else {
                                serve_connection(
                                    tls_stream.compat(),
                                    dispatch_for_conn,
                                    spec_for_conn,
                                    in_flight_for_conn,
                                    quiescing_for_conn,
                                    quiesce_response_for_conn,
                                    peer_info.clone(),
                                    runtime_for_conn.clone(),
                                )
                                .await
                            }
                        }
                        Err(error) => {
                            Err(ProximaError::Upstream(format!("tls handshake: {error}")))
                        }
                    },
                    None => {
                        dispatch_h1_or_h2(
                            stream,
                            dispatch_for_conn,
                            spec_for_conn,
                            in_flight_for_conn,
                            quiescing_for_conn,
                            quiesce_response_for_conn,
                            peer_info.clone(),
                            runtime_for_conn.clone(),
                        )
                        .await
                    }
                };
                #[cfg(not(feature = "tls"))]
                let outcome = dispatch_h1_or_h2(
                    stream,
                    dispatch_for_conn,
                    spec_for_conn,
                    in_flight_for_conn,
                    quiescing_for_conn,
                    quiesce_response_for_conn,
                    peer_info.clone(),
                    runtime_for_conn.clone(),
                )
                .await;
                if let Err(error) = outcome {
                    warn!(?error, "connection error");
                }
                let active_after = connections_active_for_conn.fetch_sub(1, Ordering::Relaxed) - 1;
                telemetry_for_conn.gauge_set(
                    "proxima.connections.active",
                    &proxima_primitives::pipe::telemetry_surface::Labels::from_pairs(&[(
                        "listener",
                        std::str::from_utf8(&listener_label_for_conn).unwrap_or(""),
                    )]),
                    active_after as i64,
                );
                // release the admission slot so the listener can drain / re-admit.
                let _ = release_tx.unbounded_send(handle);
            };
            proxima_listen::dispatch_handler(runtime.as_ref(), route, Box::pin(conn_future));
        };
    let on_accept = |conn: Box<dyn proxima_primitives::stream::StreamConnection>,
                     handle: ConnectionHandle,
                     route: Route,
                     release_tx: mpsc::UnboundedSender<ConnectionHandle>| {
        connections_accepted.fetch_add(1, Ordering::Relaxed);
        let active_now = connections_active.fetch_add(1, Ordering::Relaxed) + 1;
        debug!("accepted connection (factory)");
        let listener_labels = proxima_primitives::pipe::telemetry_surface::Labels::from_pairs(&[(
            "listener",
            std::str::from_utf8(&listener_label).unwrap_or(""),
        )]);
        telemetry.counter_inc("proxima.connections_accepted_total", &listener_labels, 1);
        telemetry.gauge_set(
            "proxima.connections.active",
            &listener_labels,
            active_now as i64,
        );
        spawn_handler(conn, handle, route, release_tx);
    };
    loop {
        futures::select_biased! {
            _ = (&mut shutdown).fuse() => break,
            released = release_rx.next().fuse() => if let Some(handle) = released {
                core.release(handle);
            },
            accepted = poll_fn(|cx| acceptor.poll_accept(cx)).fuse() => match accepted {
                Ok(conn) => match core.admit(proxima_listen::peer_ip(conn.peer().as_ref())) {
                    Admission::Admit { handle, route } => {
                        on_accept(conn, handle, route, release_tx.clone());
                    }
                    Admission::Shed { reason } => {
                        debug!(?reason, "http connection shed (factory)");
                        drop(conn);
                    }
                },
                Err(error) => warn!(?error, "accept failed"),
            },
        }
    }
    if let Some(quiesce_ms) = quiesce_duration_ms
        && quiesce_ms > 0
    {
        quiescing.store(true, Ordering::Relaxed);
        debug!(
            quiesce_ms,
            "http listener entering quiesce window (factory)"
        );
        let deadline = proxima_core::time::now() + std::time::Duration::from_millis(quiesce_ms);
        loop {
            futures::select_biased! {
                _ = sleep_until(deadline).fuse() => break,
                released = release_rx.next().fuse() => if let Some(handle) = released {
                    core.release(handle);
                },
                accepted = poll_fn(|cx| acceptor.poll_accept(cx)).fuse() => match accepted {
                    Ok(conn) => match core.admit(proxima_listen::peer_ip(conn.peer().as_ref())) {
                        Admission::Admit { handle, route } => {
                            on_accept(conn, handle, route, release_tx.clone());
                        }
                        Admission::Shed { reason } => {
                            debug!(?reason, "http connection shed during quiesce (factory)");
                            drop(conn);
                        }
                    },
                    Err(error) => warn!(?error, "accept during quiesce failed"),
                },
            }
        }
    }
    debug!(
        in_flight = in_flight.load(Ordering::Relaxed),
        "http listener draining (factory)"
    );
    drain_in_flight(
        &in_flight,
        std::time::Duration::from_millis(drain_timeout_ms),
    )
    .await;
    if let DrainOutcome::Draining = core.begin_drain() {
        drain_connections(
            &mut core,
            &mut release_rx,
            std::time::Duration::from_millis(drain_timeout_ms),
        )
        .await;
    }
    Ok(())
}

/// Drain phase: no longer accepting, wait for in-flight connections to
/// release their admission slots until the core reports closed or
/// `timeout` elapses. Mirrors [`drain_in_flight`]'s bounded-wait style so
/// a stuck connection can't hang shutdown indefinitely.
async fn drain_connections(
    core: &mut ListenerCore,
    release_rx: &mut mpsc::UnboundedReceiver<ConnectionHandle>,
    timeout: std::time::Duration,
) {
    let started = std::time::Instant::now();
    while !core.is_closed() {
        if started.elapsed() >= timeout {
            warn!(
                remaining = core.live(),
                "connection drain timeout exceeded; abandoning in-flight connections"
            );
            return;
        }
        futures::select_biased! {
            released = release_rx.next().fuse() => match released {
                Some(handle) => {
                    core.release(handle);
                }
                None => return,
            },
            () = sleep(std::time::Duration::from_millis(20)).fuse() => {}
        }
    }
}

pub use proxima_primitives::pipe::quiesce::QuiesceResponse;

async fn drain_in_flight(in_flight: &Arc<AtomicU64>, timeout: std::time::Duration) {
    let started = std::time::Instant::now();
    let poll_interval = std::time::Duration::from_millis(20);
    while in_flight.load(Ordering::Relaxed) > 0 {
        if started.elapsed() >= timeout {
            warn!(
                remaining = in_flight.load(Ordering::Relaxed),
                "drain timeout exceeded; aborting in-flight requests"
            );
            return;
        }
        sleep(poll_interval).await;
    }
}

/// Sniff the first bytes of a fresh connection to choose h1 or h2
/// dispatch, via the sans-IO [`proxima_listen::preface::classify_preface`]
/// primitive. h2 prior-knowledge clients (RFC 9113 §3.4) send a 24-byte
/// preface; h1 clients send a request line — a short leading-byte sniff
/// is enough to disambiguate before the full preface arrives.
///
/// Used on transports without ALPN (UDS, plain-TCP-without-TLS). The
/// TLS path doesn't need this — ALPN negotiates h1 vs h2 during the
/// handshake. Re-emits the sniffed bytes via `PrefixedStream` so the
/// chosen protocol driver sees the intact byte stream from byte zero.
// threads listener-scoped state plus the runtime handle for h1 streaming dispatch
#[allow(clippy::too_many_arguments)]
async fn dispatch_h1_or_h2<S>(
    mut socket: S,
    dispatch: PipeHandle,
    listener_spec: Arc<HttpListenerSpec>,
    in_flight: Arc<AtomicU64>,
    quiescing: Arc<AtomicBool>,
    quiesce_response: Arc<QuiesceResponse>,
    peer: Option<proxima_primitives::stream::PeerInfo>,
    runtime: Option<Arc<dyn proxima_runtime::Runtime>>,
) -> Result<(), ProximaError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let mut buf = [0_u8; proxima_listen::preface::H2_CLIENT_PREFACE_LEN];
    let mut filled = 0usize;
    // Read up to the route-sniff length to disambiguate. h1's smallest
    // method ("GET ") is 4 ASCII bytes; h2's preface starts with "PRI ".
    while filled < 4 {
        let read = socket.read(&mut buf[filled..4]).await.map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!("preface sniff: {err}")))
        })?;
        if read == 0 {
            return Ok(()); // peer closed before any data
        }
        filled += read;
    }
    #[cfg(feature = "http2-native")]
    let route = proxima_listen::preface::classify_preface(&buf[..filled]);
    #[cfg(not(feature = "http2-native"))]
    let route = proxima_listen::preface::PrefaceClass::Http1;

    if matches!(route, proxima_listen::preface::PrefaceClass::Http1) {
        // h1 path. Hand the sniffed bytes back via PrefixedStream.
        let stream = IntoFutures(Prepend::new(buf[..filled].to_vec(), FromFutures(socket)));
        return serve_connection(
            stream,
            dispatch,
            listener_spec,
            in_flight,
            quiescing,
            quiesce_response,
            peer,
            runtime,
        )
        .await;
    }
    #[cfg(feature = "http2-native")]
    {
        // h2 path. Keep reading until we've seen the full 24-byte
        // preface, then let the classifier confirm the final verdict.
        // The h2 connection state machine consumes the preface as its
        // first action, so we re-emit it via PrefixedStream.
        while filled < buf.len() {
            let read = socket.read(&mut buf[filled..]).await.map_err(|err| {
                ProximaError::Io(std::io::Error::other(format!("preface read: {err}")))
            })?;
            if read == 0 {
                return Err(ProximaError::Upstream("h2 preface truncated".into()));
            }
            filled += read;
        }
        match proxima_listen::preface::classify_preface(&buf[..filled]) {
            proxima_listen::preface::PrefaceClass::Http2PriorKnowledge => {
                let stream = IntoFutures(Prepend::new(buf.to_vec(), FromFutures(socket)));
                // h2 is sans-IO; no per-request spawn, so the runtime handle is unused here.
                let _ = runtime;
                crate::http2::serve_h2_connection(stream, dispatch, in_flight, quiesce_response, peer)
                    .await
            }
            // "PRI " committed the connection to h2 above; anything but
            // an exact 24-byte match is neither valid h1 nor valid h2.
            _ => Err(ProximaError::Upstream(
                "h2 preface bytes did not match RFC 9113 §3.4".into(),
            )),
        }
    }
    #[cfg(not(feature = "http2-native"))]
    {
        // Unreachable when http2 disabled — route is always Http1.
        let _ = (
            dispatch,
            in_flight,
            quiescing,
            quiesce_response,
            listener_spec,
            peer,
            runtime,
        );
        Ok(())
    }
}

// serve_h1_connection + serve_connection + helpers moved to
// proxima-h1::serve. percent_decode tests live with the function there.

// exercises the factory path via `TokioAcceptorFactory` + a tokio
// `LocalSet` test harness — needs the `http1` feature the same way
// `http2::listener`'s test module needs `http2`, even though the driver
// under test (`serve_via_factory` -> `serve_connection`) is itself
// tokio-free.
#[cfg(all(test, feature = "http1"))]
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


    // the serve loop spawns per-connection work via `tokio::task::spawn_local`
    // (runtime: None), so the whole exercise runs inside a LocalSet on a
    // current-thread runtime. a per-core runtime can't be passed here: its
    // `spawn_on_current_core` asserts it is on a worker thread, which the
    // test thread is not.
    #[proxima::test(runtime = "tokio")]
    async fn factory_path_serves_http1_get() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async move {
                let bind: SocketAddr = "127.0.0.1:0".parse().expect("parse bind addr");
                let dispatch = into_handle(ConstantOk);
                let context = proxima_listen::ServeContext::new(NoopTelemetry::handle())
                    .with_acceptor_factory(Arc::new(proxima_net::tokio::TokioAcceptorFactory));
                let (shutdown_tx, shutdown_rx) = oneshot::channel();

                // bind to a free port first, then drop it so serve can claim it.
                // a 0-port bind on serve would hide the ephemeral port from the
                // client; pinning it lets the client connect deterministically.
                let probe = tokio::net::TcpListener::bind(bind)
                    .await
                    .expect("probe bind");
                let addr = probe.local_addr().expect("probe addr");
                drop(probe);

                let server_spec = serde_json::json!({ "name": "http" });
                let protocol = HttpListenProtocol::new();
                // serve borrows `protocol` + `server_spec` (`'_` future), so it
                // can't be spawned; drive it concurrently with the client on the
                // same task. the LocalSet still backs the per-connection
                // `spawn_local` calls the serve loop makes.
                let serve = protocol.serve(addr, dispatch, &server_spec, context, shutdown_rx);

                let client_work = async {
                    // poll connect readiness — no sleeps, just retry the connect
                    // until the acceptor is listening.
                    let mut client = loop {
                        match tokio::net::TcpStream::connect(addr).await {
                            Ok(stream) => break stream,
                            Err(_) => tokio::task::yield_now().await,
                        }
                    };
                    client
                        .write_all(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
                        .await
                        .expect("client write");
                    client.flush().await.expect("client flush");
                    let mut response = Vec::with_capacity(256);
                    client
                        .read_to_end(&mut response)
                        .await
                        .expect("client read");
                    String::from_utf8(response).expect("response utf8")
                };

                let text = tokio::select! {
                    serve_result = serve => panic!("serve returned early: {serve_result:?}"),
                    text = client_work => text,
                };
                assert!(
                    text.starts_with("HTTP/1.1"),
                    "expected HTTP/1.1 response, got: {text:?}"
                );
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
                let bind: SocketAddr = "127.0.0.1:0".parse().expect("parse bind addr");
                let dispatch = into_handle(ConstantOk);
                let context = proxima_listen::ServeContext::new(NoopTelemetry::handle())
                    .with_acceptor_factory(Arc::new(proxima_net::tokio::TokioAcceptorFactory));
                let (shutdown_tx, shutdown_rx) = oneshot::channel();

                let probe = tokio::net::TcpListener::bind(bind)
                    .await
                    .expect("probe bind");
                let addr = probe.local_addr().expect("probe addr");
                drop(probe);

                let server_spec = serde_json::json!({ "name": "http" });
                let protocol = HttpListenProtocol::new();
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
