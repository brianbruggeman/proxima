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
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;

use futures::channel::oneshot;
use serde_json::Value;

use proxima_core::ProximaError;
use proxima_listen::ListenProtocol;
use proxima_listen::any::AnyProtocol;
use proxima_primitives::pipe::handler::PipeHandle;

use crate::any_listener::{AnyListenProtocol, H1AnyProtocol};
#[cfg(feature = "http2-native")]
use crate::any_listener::H2PriorKnowledgeAnyProtocol;

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

    /// Reshaped onto [`AnyListenProtocol`]: this combiner's bind/accept/TLS/
    /// SO_REUSEPORT/TCP_FASTOPEN/UDS/admission/drain plumbing collapsed onto
    /// the SAME machinery `.any()` drives — this type now only fixes the
    /// candidate set to `{h1, h2}` (both bound to the ONE `dispatch` this
    /// call was given, via `AnyListenProtocol::serve`'s dispatch-fallback —
    /// see that method's doc) and its own registry name (`"http"`, not
    /// `"any"`). The h1/h2 keep-alive courtesy-503 behavior survives
    /// unchanged: it now rides the request-admit `Shed` path every
    /// `AnyProtocol` candidate shares, rather than a triple of
    /// listener-local `Arc<AtomicU64>`/`Arc<AtomicBool>`/`QuiesceResponse`
    /// this combiner used to thread by hand.
    fn serve(
        &self,
        bind: SocketAddr,
        dispatch: PipeHandle,
        spec: &Value,
        context: proxima_listen::ServeContext,
        shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            #[cfg_attr(not(feature = "http2-native"), allow(unused_mut))]
            let mut candidates: Vec<Arc<dyn AnyProtocol>> = vec![Arc::new(H1AnyProtocol::new())];
            #[cfg(feature = "http2-native")]
            candidates.push(Arc::new(H2PriorKnowledgeAnyProtocol::new()));
            let protocol = AnyListenProtocol::from_candidates(
                Arc::from(candidates),
                Arc::new(std::collections::BTreeMap::new()),
            )
            .with_label("http");
            protocol.serve(bind, dispatch, &spec, context, shutdown).await
        })
    }
}

// The Linux + `io-uring` path (`listeners/http_uring.rs`) is unaffected by
// this reshape — it stays in the umbrella, pulling prime's io_uring
// TcpListener via `io_uring_compat` directly rather than through either
// `HttpListenProtocol` or `AnyListenProtocol`.

pub use proxima_primitives::pipe::quiesce::QuiesceResponse;

// serve_h1_connection + serve_connection + helpers live in
// proxima-h1::serve / crate::any_listener. percent_decode tests live with
// the function there.

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
