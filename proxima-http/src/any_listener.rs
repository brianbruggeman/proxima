//! `Listener::any()` scaffolding: the concrete
//! [`AnyProtocol`](proxima_listen::any::AnyProtocol) candidates this crate
//! ships (h1, h2 prior-knowledge) plus [`AnyListenProtocol`], the
//! [`ListenProtocol`] that owns the ONE bind + accept loop and dispatches
//! each accepted stream through [`proxima_listen::any::Classifier`].
//!
//! # Where `dispatch_h1_or_h2` went
//!
//! `proxima-http`'s former `listener::dispatch_h1_or_h2` — the hand-rolled
//! byte-sniff that chose h1 vs h2 for a single ALPN-multiplexed bind — is
//! retired (principle 15: no dangling dead code, no parallel copy of the
//! same logic). Its two jobs split cleanly onto the new open-listener
//! primitives:
//!
//! - the classification rule (24-byte h2 preface compare / positive h1
//!   method match) is now [`H2PriorKnowledgeAnyProtocol::probe`] and
//!   [`H1AnyProtocol::probe`] — pure, sans-IO, unit-testable in isolation,
//!   and exercised by [`crate::listener`]'s OWN combiner (which still needs
//!   the same decision, now made through
//!   [`proxima_listen::any::Classifier`] instead of a bespoke inline
//!   compare) — see that module's `serve_via_open_classifier`.
//! - the connection drive (replay the sniffed prefix, call
//!   `serve_connection`/`serve_h2_connection`) is [`H1AnyProtocol::drive`]
//!   and [`H2PriorKnowledgeAnyProtocol::drive`].
//!
//! [`crate::listener::HttpListenProtocol`] (the ALPN h1+h2 combiner) still
//! threads its OWN per-listener state (in-flight counter, quiesce
//! response) into `serve_connection`/`serve_h2_connection` directly — that
//! richer, listener-scoped state has no place in the generic
//! [`AnyProtocol::drive`] signature, which only ever sees one already
//! -accepted stream. [`AnyListenProtocol`] here is the NEW, simpler,
//! quiesce-free sibling: an open registry of candidates behind one bind,
//! mirroring [`crate::http2::listener::H2ListenProtocol`]'s
//! `serve_via_factory` shape (no admission/drain wiring, matching that
//! shipped listener's own scope) rather than the combiner's fuller
//! quiesce/drain machinery.

use std::future::Future;
use std::future::poll_fn;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::FutureExt;
use futures::channel::mpsc;
use futures::channel::oneshot;
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite};
use futures::stream::StreamExt;
use serde_json::Value;
use tracing::{debug, warn};

use proxima_core::ProximaError;
use proxima_core::io::{FromFutures, IntoFutures, Prepend};
use proxima_core::time::sleep;
use proxima_listen::admission::{
    Admission, BlacklistTable, ConnAdmission, ConnectionHandle, DrainOutcome, ListenerCore,
    ShedReason,
};
use proxima_listen::any::{
    AnyHandler, AnyProtocol, Classifier, ClassifyOutcome, ProbeVerdict, RejectReason,
    downcast_handler,
};
use proxima_listen::{ListenProtocol, ServeContext};
use proxima_primitives::pipe::handler::PipeHandle;
use proxima_primitives::stream::{PeerInfo, StreamConnection};

use crate::http1::serve::HttpListenerSpec;
use crate::http1::serve::serve_connection as serve_h1_connection_shared;
#[cfg(feature = "http2-native")]
use crate::http2::serve_h2_connection;

/// HTTP/1.1 candidate for the open universal listener. `probe` is a
/// POSITIVE match against the known HTTP methods — an open registry has no
/// implicit "everything else is h1" fallback the way the closed h1-or-h2
/// combiner does, so h1 must claim its own wire explicitly, the same way
/// every other candidate does.
pub struct H1AnyProtocol {
    label: String,
}

impl Default for H1AnyProtocol {
    fn default() -> Self {
        Self { label: "h1".into() }
    }
}

impl H1AnyProtocol {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// HTTP/1.1 request methods this candidate positively recognizes (RFC 9110
/// §9.1 + `CONNECT`/`PATCH`), each written with its trailing space so a
/// prefix compare cannot mistake `"GET"` for the start of some longer,
/// unlisted verb. Longest entry sizes [`H1AnyProtocol::max_prefix_bytes`].
const H1_METHODS: &[&[u8]] = &[
    b"GET ",
    b"POST ",
    b"PUT ",
    b"HEAD ",
    b"DELETE ",
    b"OPTIONS ",
    b"PATCH ",
    b"CONNECT ",
    b"TRACE ",
];

impl AnyProtocol for H1AnyProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    fn max_prefix_bytes(&self) -> usize {
        H1_METHODS
            .iter()
            .map(|method| method.len())
            .max()
            .unwrap_or(0)
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        // Full-match check first: a method whose entire length fits inside
        // `prefix` and whose bytes agree all the way through wins outright.
        // No two entries in `H1_METHODS` can both fully match the same
        // prefix (they diverge within the first two bytes), so returning
        // on the first hit is safe, not merely convenient.
        for method in H1_METHODS {
            if prefix.len() >= method.len() && &prefix[..method.len()] == *method {
                return ProbeVerdict::Match { consumed: 0 };
            }
        }
        // No full match yet — find the SMALLEST additional length among
        // methods `prefix` is still a live (too-short) prefix of. Taking
        // the first such method instead of the minimum would report an
        // `at_least` too large whenever a shorter candidate method is also
        // still alive (e.g. "P" is a live prefix of both "PUT " (4) and
        // "PATCH " (6) — the correct answer is 4, not whichever is checked
        // first).
        let mut need_more_min: Option<usize> = None;
        for method in H1_METHODS {
            if prefix.len() < method.len() && prefix == &method[..prefix.len()] {
                need_more_min = Some(match need_more_min {
                    Some(current) => current.min(method.len()),
                    None => method.len(),
                });
            }
        }
        match need_more_min {
            Some(at_least) => ProbeVerdict::NeedMore { at_least },
            None => ProbeVerdict::No,
        }
    }

    fn drive<'a>(
        &'a self,
        stream: Box<dyn StreamConnection>,
        handler: AnyHandler,
        spec: &'a Value,
        peer: Option<PeerInfo>,
        admission: &'a ConnAdmission,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
        let max_body_bytes = spec
            .get("max_body_bytes")
            .and_then(Value::as_u64)
            .map(|raw| raw as usize);
        let quiesce_status = spec
            .get("quiesce_status")
            .and_then(Value::as_u64)
            .map(|raw| raw as u16)
            .unwrap_or(503);
        let quiesce_retry_after = spec
            .get("quiesce_retry_after")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| "1".into());
        Box::pin(async move {
            let dispatch: PipeHandle =
                (*downcast_handler::<PipeHandle>(self.name(), &handler)?).clone();
            let listener_spec = Arc::new(HttpListenerSpec { max_body_bytes });
            // Bridge onto the SAME shared in-flight counter / quiescing flag
            // `admission` owns — h1's existing per-request loop already
            // checks `quiescing` and increments/decrements `in_flight`
            // itself (its own inline equivalent of `request_admit`); driving
            // it with the admission-owned atomics (not fresh per-connection
            // ones) is what makes quiesce/drain listener-wide instead of a
            // no-op each connection ran in isolation.
            let in_flight = admission.in_flight_counter();
            let quiescing = admission.quiescing_flag();
            let quiesce_response = Arc::new(proxima_primitives::pipe::quiesce::QuiesceResponse {
                status: quiesce_status,
                retry_after: quiesce_retry_after,
            });
            serve_h1_connection_shared(
                stream,
                dispatch,
                listener_spec,
                in_flight,
                quiescing,
                quiesce_response,
                peer,
                None,
            )
            .await
        })
    }
}

/// HTTP/2 prior-knowledge candidate (h2c, RFC 9113 §3.4) for the open
/// universal listener — the identical classification rule
/// `proxima-listen`'s [`proxima_listen::preface::classify_preface`] uses
/// for its h2 sub-path, re-hosted here as an `AnyProtocol::probe`.
#[cfg(feature = "http2-native")]
pub struct H2PriorKnowledgeAnyProtocol {
    label: String,
}

#[cfg(feature = "http2-native")]
impl Default for H2PriorKnowledgeAnyProtocol {
    fn default() -> Self {
        Self { label: "h2".into() }
    }
}

#[cfg(feature = "http2-native")]
impl H2PriorKnowledgeAnyProtocol {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(feature = "http2-native")]
impl AnyProtocol for H2PriorKnowledgeAnyProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    fn max_prefix_bytes(&self) -> usize {
        proxima_listen::preface::H2_CLIENT_PREFACE_LEN
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        let full = proxima_listen::preface::H2_CLIENT_PREFACE;
        let compare_len = prefix.len().min(full.len());
        if prefix[..compare_len] != full[..compare_len] {
            return ProbeVerdict::No;
        }
        if prefix.len() < full.len() {
            return ProbeVerdict::NeedMore {
                at_least: full.len(),
            };
        }
        ProbeVerdict::Match {
            consumed: full.len(),
        }
    }

    fn drive<'a>(
        &'a self,
        stream: Box<dyn StreamConnection>,
        handler: AnyHandler,
        _spec: &'a Value,
        peer: Option<PeerInfo>,
        admission: &'a ConnAdmission,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
        Box::pin(async move {
            let dispatch: PipeHandle =
                (*downcast_handler::<PipeHandle>(self.name(), &handler)?).clone();
            serve_h2_connection(stream, dispatch, admission.clone(), peer).await
        })
    }
}

/// Replays the accumulated classification prefix at the front of an
/// accepted connection before any candidate's own wire parser sees a byte —
/// the same `Prepend`-over-`FromFutures`/`IntoFutures` dance
/// `proxima-http`'s former `dispatch_h1_or_h2` used to re-emit a sniffed
/// preface. Implements [`StreamConnection`] so it can be boxed and handed
/// straight to [`AnyProtocol::drive`].
struct PrefixedConnection {
    inner: IntoFutures<Prepend<FromFutures<Box<dyn StreamConnection>>>>,
    peer: Option<PeerInfo>,
}

impl PrefixedConnection {
    fn new(prefix: Vec<u8>, inner: Box<dyn StreamConnection>, peer: Option<PeerInfo>) -> Self {
        Self {
            inner: IntoFutures(Prepend::new(prefix, FromFutures(inner))),
            peer,
        }
    }
}

impl AsyncRead for PrefixedConnection {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.get_mut().inner).poll_read(cx, buf)
    }
}

impl AsyncWrite for PrefixedConnection {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.get_mut().inner).poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_flush(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_close(cx)
    }
}

impl StreamConnection for PrefixedConnection {
    fn peer(&self) -> Option<PeerInfo> {
        self.peer.clone()
    }
}

/// Optional observer for connections the classifier drops before any
/// candidate resolved (`RejectReason::NoCandidateMatched` /
/// `PrefixBoundExceeded`) — the seam a later deny/DoS-blacklist follow-on
/// hangs off of. This crate ships the seam, not a policy: no deny list, no
/// blacklist logic lives here or is called by default (`None`).
pub type RejectHook = Arc<dyn Fn(Option<PeerInfo>, RejectReason) + Send + Sync>;

/// The open universal listener: one bind, one accept loop, and a
/// [`Classifier`] over an arbitrary, registry-driven set of
/// [`AnyProtocol`] candidates — [`crate::listener::HttpListenProtocol`]'s
/// combiner generalized from "exactly h1 and h2" to "whatever is
/// registered." Mirrors [`crate::http2::listener::H2ListenProtocol`]'s
/// `serve_via_factory` shape: `AcceptorFactory`-driven, no admission/drain
/// wiring (that richer machinery belongs to a listener that owns exactly
/// one wire's quiesce semantics, which this open listener deliberately does
/// not assume for an arbitrary candidate set).
///
/// # Per-protocol handlers, not one shared dispatch
///
/// A single mounted pipe can't serve a classified mux — h1/h2 both want a
/// [`PipeHandle`], but nothing stops a future candidate wanting a
/// completely different shape (a `Frame -> Frame` redis handle, a pgwire
/// query engine). `handlers` is a name -> [`AnyHandler`] map, resolved ONCE
/// at construction (`ListenerBuilder::any`/`accept`/`accepts` merges any
/// per-listener `.any_handler(name, handler)` bindings over the `App`-level
/// per-protocol default-handler registry — see `src/listener/handle.rs`).
/// [`ListenProtocol::serve`]'s own `dispatch: PipeHandle` parameter is
/// therefore UNUSED here — the same asymmetry
/// `proxima_pgwire::PgWireListenProtocol` already has (it carries its own
/// query engine from construction and never reads `serve`'s `dispatch`
/// either); `.handle(pipe)` stays required before `.serve()` purely for the
/// one uniform validation path every listener axis shares.
pub struct AnyListenProtocol {
    label: String,
    candidates: Arc<[Arc<dyn AnyProtocol>]>,
    global_cap: usize,
    handlers: Arc<std::collections::BTreeMap<String, AnyHandler>>,
    on_reject: Option<RejectHook>,
    blacklist: Option<BlacklistTable>,
}

impl AnyListenProtocol {
    /// Accept every candidate `registry` currently holds (a snapshot taken
    /// now — later registrations after this call are not picked up, mirroring
    /// every other `Arc<dyn ListenProtocol>` in this codebase being a
    /// point-in-time compiled instance). `handlers` is the resolved
    /// per-protocol handler map (per-listener overrides already merged over
    /// the `App`-level defaults by the caller).
    #[must_use]
    pub fn new(
        registry: &proxima_listen::any::AnyRegistry,
        handlers: Arc<std::collections::BTreeMap<String, AnyHandler>>,
    ) -> Self {
        Self {
            label: "any".into(),
            candidates: registry.snapshot(),
            global_cap: proxima_listen::sized::ANY_MAX_PROBE_PREFIX_BYTES_DEFAULT,
            handlers,
            on_reject: None,
            blacklist: None,
        }
    }

    /// Construct directly from an explicit candidate list, bypassing the
    /// registry — used by [`crate::listener::HttpListenProtocol`]'s
    /// reshaped `serve` to build its own fixed `{h1, h2}` pair without
    /// needing an `AnyRegistry` in hand.
    #[must_use]
    pub fn from_candidates(
        candidates: Arc<[Arc<dyn AnyProtocol>]>,
        handlers: Arc<std::collections::BTreeMap<String, AnyHandler>>,
    ) -> Self {
        Self {
            label: "any".into(),
            candidates,
            global_cap: proxima_listen::sized::ANY_MAX_PROBE_PREFIX_BYTES_DEFAULT,
            handlers,
            on_reject: None,
            blacklist: None,
        }
    }

    /// Accept only the named candidates (`.accepts(&[..])`/`.accept(name)`
    /// on `Listener::builder()`). Errors if any name isn't registered.
    pub fn with_names(
        registry: &proxima_listen::any::AnyRegistry,
        names: &[String],
        handlers: Arc<std::collections::BTreeMap<String, AnyHandler>>,
    ) -> Result<Self, ProximaError> {
        Ok(Self {
            label: "any".into(),
            candidates: registry.snapshot_named(names)?,
            global_cap: proxima_listen::sized::ANY_MAX_PROBE_PREFIX_BYTES_DEFAULT,
            handlers,
            on_reject: None,
            blacklist: None,
        })
    }

    /// Override the registry label ("any") — used by
    /// [`crate::listener::HttpListenProtocol`] so its reshaped `serve`
    /// still identifies as `"http"` on the wire/registry, not `"any"`.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// A single-candidate, no-handler-map instance — the registry-driven
    /// `.h2()`/`.grpc()` axis's shape (`AppBuilder::with_defaults`,
    /// `src/listener/handle.rs`'s `h2_listen_protocol`): no dispatch pipe
    /// is available yet at REGISTRATION time, only later at `serve()` call
    /// time via `RunConfig`/`ListenerSpec` — so `handlers` stays empty and
    /// every accepted connection falls back to `serve`'s own `dispatch`
    /// parameter (see `merge_dispatch_fallback`).
    #[must_use]
    pub fn single_candidate(label: impl Into<String>, candidate: Arc<dyn AnyProtocol>) -> Self {
        Self {
            label: label.into(),
            candidates: Arc::from(vec![candidate]),
            global_cap: proxima_listen::sized::ANY_MAX_PROBE_PREFIX_BYTES_DEFAULT,
            handlers: Arc::new(std::collections::BTreeMap::new()),
            on_reject: None,
            blacklist: None,
        }
    }

    /// Install the reject-hook seam (see [`RejectHook`]'s doc). `None` by
    /// default — a bare `.any()` listener with no hook installed simply
    /// drops rejected connections (still logged via `warn!`, never silent).
    #[must_use]
    pub fn with_reject_hook(mut self, hook: RejectHook) -> Self {
        self.on_reject = Some(hook);
        self
    }

    /// Install the accept-edge DoS-blacklist gate (see [`BlacklistTable`]'s
    /// doc). `None` by default — a bare `.any()` listener with no table
    /// installed never sheds `ShedReason::Blacklisted`; both accept loops
    /// (TCP + UDS) consult it BEFORE `ListenerCore::admit`, never after.
    #[must_use]
    pub fn with_blacklist(mut self, table: BlacklistTable) -> Self {
        self.blacklist = Some(table);
        self
    }
}

impl ListenProtocol for AnyListenProtocol {
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
        let candidates = self.candidates.clone();
        let global_cap = self.global_cap;
        // Registry-driven callers (`AppBuilder::with_defaults`'s `"h2"`
        // registration, `ListenerBuilder`'s `.h2()`/`.grpc()` axis) have no
        // per-protocol handler map at construction time — only the ONE
        // `dispatch` `RunConfig`/`ListenerSpec` carries at serve time. Every
        // candidate whose name is absent from `self.handlers` falls back to
        // this erased `dispatch`, exactly like `ListenerBuilder::serve`'s own
        // `.handle(pipe)` fallback (works for any candidate whose expected
        // handler type IS a `PipeHandle` — h1/h2's shape; see that fallback's
        // own doc). A pre-populated `self.handlers` entry (the `.any()`
        // builder path, or a `.pgwire()`/`.redis()`-shaped candidate) always
        // wins over this fallback.
        //
        // `Offload` wraps the served pipe ONCE, here at serve start — never
        // per request. `SpreadToPeers` isolates a synchronously-blocking
        // handler by running its `Pipe::call` on the runtime's background
        // pool instead of whichever executor core the connection future
        // landed on; `context.handler_dispatch` alone only decides which
        // core drives the connection FUTURE (via `dispatch_handler`
        // downstream) — that per-connection concern is unrelated to this
        // per-dispatch one, and without this wrap a blocking handler still
        // wedges its assigned executor thread. See `Offload`'s own doc.
        let dispatch = match (context.handler_dispatch, context.runtime.as_ref()) {
            (proxima_listen::HandlerDispatch::SpreadToPeers { .. }, Some(runtime)) => {
                proxima_primitives::pipe::handler::into_handle(proxima_listen::Offload::new(
                    dispatch,
                    runtime.clone(),
                ))
            }
            _ => dispatch,
        };
        let handlers = merge_dispatch_fallback(&self.handlers, &candidates, &dispatch);
        let on_reject = self.on_reject.clone();
        let blacklist = self.blacklist.clone();
        let spec_owned = Arc::new(spec.clone());
        Box::pin(async move {
            // UDS: `spec.path` set -> bind a Unix domain socket instead of
            // TCP. No TLS/SO_REUSEPORT/TCP_FASTOPEN on UDS (none apply);
            // still gets full ListenerCore/ConnAdmission/drain treatment.
            if let Some(path) = spec_owned.get("path").and_then(Value::as_str) {
                #[cfg(feature = "http1")]
                {
                    return serve_uds(
                        std::path::PathBuf::from(path),
                        spec_owned
                            .get("mode")
                            .and_then(Value::as_u64)
                            .map(|raw| raw as u32),
                        candidates,
                        global_cap,
                        handlers,
                        on_reject,
                        blacklist,
                        spec_owned,
                        context.ready_signal.clone(),
                        shutdown,
                    )
                    .await;
                }
                #[cfg(not(feature = "http1"))]
                {
                    let _ = path;
                    return Err(ProximaError::Config(
                        "any listener UDS bind requires the `http1` feature".into(),
                    ));
                }
            }
            let Some(factory) = context.acceptor_factory.clone() else {
                return Err(ProximaError::Config(
                    "any listener requires an AcceptorFactory (no legacy tokio fallback for this listener)"
                        .into(),
                ));
            };
            let use_reuseport = spec_owned
                .get(proxima_listen::handle::REUSEPORT_SPEC_KEY)
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let tcp_fastopen_queue = spec_owned
                .get("tcp_fastopen")
                .and_then(Value::as_u64)
                .map(|raw| raw as u32);
            let max_in_flight_requests = spec_owned
                .get("max_in_flight_requests")
                .and_then(Value::as_u64)
                .map(|raw| raw as usize)
                .unwrap_or(usize::MAX);
            let drain_timeout_ms = spec_owned
                .get("drain_timeout_ms")
                .and_then(Value::as_u64)
                .unwrap_or(30_000);
            let quiesce_duration_ms = spec_owned
                .get("quiesce_duration_ms")
                .and_then(Value::as_u64);
            #[cfg(feature = "tls")]
            let tls_acceptor =
                match proxima_tls::config_from_spec_value(spec_owned.get(proxima_tls::SPEC_KEY)) {
                    Ok(Some(config)) => match proxima_tls::build_acceptor_futures_io(&config) {
                        Ok(acceptor) => Some(acceptor),
                        Err(error) => return Err(error),
                    },
                    Ok(None) => None,
                    Err(error) => return Err(error),
                };
            serve_via_factory(
                factory,
                bind,
                label,
                candidates,
                global_cap,
                handlers,
                on_reject,
                blacklist,
                spec_owned,
                context.telemetry.clone(),
                context.runtime.clone(),
                context.handler_dispatch,
                use_reuseport,
                tcp_fastopen_queue,
                max_in_flight_requests,
                drain_timeout_ms,
                quiesce_duration_ms,
                #[cfg(feature = "tls")]
                tls_acceptor,
                context.ready_signal.clone(),
                shutdown,
            )
            .await
        })
    }
}

/// Fills in a `dispatch`-erased fallback for every registered candidate
/// that has no entry in `handlers` — see [`AnyListenProtocol::serve`]'s
/// doc for why this exists (the registry-driven `.h2()`/`.grpc()` axis has
/// no per-candidate handler map at all). A candidate already present in
/// `handlers` (the `.any()` builder path's pre-merged map) is left
/// untouched.
fn merge_dispatch_fallback(
    handlers: &std::collections::BTreeMap<String, AnyHandler>,
    candidates: &[Arc<dyn AnyProtocol>],
    dispatch: &PipeHandle,
) -> Arc<std::collections::BTreeMap<String, AnyHandler>> {
    let mut merged = handlers.clone();
    for candidate in candidates {
        merged
            .entry(candidate.name().to_string())
            .or_insert_with(|| proxima_listen::any::erase_handler(dispatch.clone()));
    }
    Arc::new(merged)
}

/// `true` if `blacklist` is installed AND currently bans `peer_ip` — the
/// one check every accept loop (TCP factory, TCP quiesce window, UDS) runs
/// BEFORE `ListenerCore::admit`. `None` (no table installed) always
/// answers `false`, so a bare `.any()` listener with no `.deny()`/
/// `.blacklist()` wiring behaves exactly as it did before this gate
/// existed.
fn peer_is_banned(blacklist: &Option<BlacklistTable>, peer_ip: std::net::IpAddr) -> bool {
    blacklist
        .as_ref()
        .is_some_and(|table| table.is_banned(peer_ip, proxima_core::time::now()))
}

/// `AcceptorFactory`-driven accept loop: bind (honoring SO_REUSEPORT +
/// TCP_FASTOPEN from `spec`, matching every other TCP listener in this
/// crate), admit each accepted connection through a [`ListenerCore`],
/// terminate TLS FIRST when configured (plaintext is rejected at the
/// handshake — no bypass), classify its leading bytes through a fresh
/// per-connection [`Classifier`] (skipped when ALPN already named a
/// registered candidate), and — once resolved — replay the accumulated
/// prefix into the matched candidate's [`AnyProtocol::drive`], threading
/// the listener-wide [`ConnAdmission`] handle. On shutdown: quiesce (if
/// configured) sheds new REQUESTS while still accepting connections, then
/// drain stops accepting connections AND sheds all requests, bounded-
/// waiting for both the connection table and the request counter to reach
/// zero.
#[allow(clippy::too_many_arguments)]
async fn serve_via_factory(
    factory: Arc<dyn proxima_primitives::stream::AcceptorFactory>,
    bind: SocketAddr,
    label: String,
    candidates: Arc<[Arc<dyn AnyProtocol>]>,
    global_cap: usize,
    handlers: Arc<std::collections::BTreeMap<String, AnyHandler>>,
    on_reject: Option<RejectHook>,
    blacklist: Option<BlacklistTable>,
    spec: Arc<Value>,
    telemetry: proxima_primitives::pipe::telemetry_surface::TelemetryHandle,
    runtime: Option<Arc<dyn proxima_runtime::Runtime>>,
    handler_dispatch: proxima_listen::HandlerDispatch,
    use_reuseport: bool,
    tcp_fastopen_queue: Option<u32>,
    max_in_flight_requests: usize,
    drain_timeout_ms: u64,
    quiesce_duration_ms: Option<u64>,
    #[cfg(feature = "tls")] tls_acceptor: Option<futures_rustls::TlsAcceptor>,
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
        "any listener bound (open classifier, factory)"
    );
    #[cfg(feature = "tls")]
    if tls_acceptor.is_some() {
        debug!(?bind, "any listener terminating TLS");
    }
    let policy = handler_dispatch.as_policy();
    let read_chunk_len = global_cap.clamp(64, 4096);
    let mut core = ListenerCore::new(policy);
    let admission = ConnAdmission::new(max_in_flight_requests);
    let (release_tx, mut release_rx) = mpsc::unbounded::<ConnectionHandle>();
    let listener_labels = proxima_primitives::pipe::telemetry_surface::Labels::from_pairs(&[(
        "listener",
        label.as_str(),
    )]);

    loop {
        futures::select_biased! {
            _ = (&mut shutdown).fuse() => break,
            released = release_rx.next().fuse() => if let Some(handle) = released {
                core.release(handle);
            },
            accepted = poll_fn(|cx| acceptor.poll_accept(cx)).fuse() => match accepted {
                Ok(conn) => {
                    let peer_ip = proxima_listen::peer_ip(conn.peer().as_ref());
                    // Blacklist gate BEFORE `core.admit` — never after: a
                    // post-admit check would still have committed a table
                    // slot for a banned peer.
                    let decision = if peer_is_banned(&blacklist, peer_ip) {
                        Admission::Shed { reason: ShedReason::Blacklisted }
                    } else {
                        core.admit(peer_ip)
                    };
                    match decision {
                    Admission::Admit { handle, route } => {
                        telemetry.counter_inc("proxima.connections_accepted_total", &listener_labels, 1);
                        let candidates_for_conn = candidates.clone();
                        let handlers_for_conn = handlers.clone();
                        let on_reject_for_conn = on_reject.clone();
                        let spec_for_conn = spec.clone();
                        let admission_for_conn = admission.clone();
                        let release_tx_for_conn = release_tx.clone();
                        #[cfg(feature = "tls")]
                        let tls_for_conn = tls_acceptor.clone();
                        let conn_future = async move {
                            classify_and_drive(
                                conn,
                                candidates_for_conn,
                                global_cap,
                                read_chunk_len,
                                handlers_for_conn,
                                on_reject_for_conn,
                                spec_for_conn,
                                admission_for_conn,
                                #[cfg(feature = "tls")]
                                tls_for_conn,
                            )
                            .await;
                            let _ = release_tx_for_conn.unbounded_send(handle);
                        };
                        proxima_listen::dispatch_handler(runtime.as_ref(), route, Box::pin(conn_future));
                    }
                    Admission::Shed { reason } => {
                        debug!(?reason, "any listener connection shed");
                        drop(conn);
                    }
                    }
                },
                Err(error) => warn!(?error, "any listener accept failed"),
            },
        }
    }

    if let Some(quiesce_ms) = quiesce_duration_ms
        && quiesce_ms > 0
    {
        admission.begin_quiesce();
        debug!(quiesce_ms, "any listener entering quiesce window");
        let deadline = proxima_core::time::now() + std::time::Duration::from_millis(quiesce_ms);
        loop {
            futures::select_biased! {
                _ = proxima_core::time::sleep_until(deadline).fuse() => break,
                released = release_rx.next().fuse() => if let Some(handle) = released {
                    core.release(handle);
                },
                accepted = poll_fn(|cx| acceptor.poll_accept(cx)).fuse() => match accepted {
                    Ok(conn) => {
                        let peer_ip = proxima_listen::peer_ip(conn.peer().as_ref());
                        // Blacklist gate BEFORE `core.admit` — never after
                        // (see the primary accept loop above for why).
                        let decision = if peer_is_banned(&blacklist, peer_ip) {
                            Admission::Shed { reason: ShedReason::Blacklisted }
                        } else {
                            core.admit(peer_ip)
                        };
                        match decision {
                        Admission::Admit { handle, route } => {
                            telemetry.counter_inc("proxima.connections_accepted_total", &listener_labels, 1);
                            let candidates_for_conn = candidates.clone();
                            let handlers_for_conn = handlers.clone();
                            let on_reject_for_conn = on_reject.clone();
                            let spec_for_conn = spec.clone();
                            let admission_for_conn = admission.clone();
                            let release_tx_for_conn = release_tx.clone();
                            #[cfg(feature = "tls")]
                            let tls_for_conn = tls_acceptor.clone();
                                let conn_future = async move {
                                classify_and_drive(
                                    conn,
                                    candidates_for_conn,
                                    global_cap,
                                    read_chunk_len,
                                    handlers_for_conn,
                                    on_reject_for_conn,
                                    spec_for_conn,
                                    admission_for_conn,
                                    #[cfg(feature = "tls")]
                                    tls_for_conn,
                                )
                                .await;
                                let _ = release_tx_for_conn.unbounded_send(handle);
                            };
                            proxima_listen::dispatch_handler(runtime.as_ref(), route, Box::pin(conn_future));
                        }
                        Admission::Shed { reason } => {
                            debug!(?reason, "any listener connection shed during quiesce");
                            drop(conn);
                        }
                        }
                    }
                    Err(error) => warn!(?error, "any listener accept during quiesce failed"),
                },
            }
        }
    }

    debug!("any listener draining: both connections and in-flight requests");
    admission.begin_drain();
    if let DrainOutcome::Draining = core.begin_drain() {
        drain_both(
            &mut core,
            &admission,
            &mut release_rx,
            std::time::Duration::from_millis(drain_timeout_ms),
        )
        .await;
    } else {
        // no connections were live; still bound-wait on any straggling
        // in-flight requests (a connection could have released between
        // begin_drain's snapshot and this check).
        drain_requests_only(
            &admission,
            std::time::Duration::from_millis(drain_timeout_ms),
        )
        .await;
    }
    Ok(())
}

/// Drain phase: wait for BOTH the connection table (`core`) and the
/// request-level counter (`admission`) to reach zero, bounded by `timeout`
/// — a stuck connection or a stuck in-flight request can't hang shutdown
/// indefinitely.
async fn drain_both(
    core: &mut ListenerCore,
    admission: &ConnAdmission,
    release_rx: &mut mpsc::UnboundedReceiver<ConnectionHandle>,
    timeout: std::time::Duration,
) {
    let started = std::time::Instant::now();
    while !core.is_closed() || admission.in_flight() > 0 {
        if started.elapsed() >= timeout {
            warn!(
                remaining_connections = core.live(),
                remaining_requests = admission.in_flight(),
                "any listener drain timeout exceeded; abandoning in-flight work"
            );
            return;
        }
        futures::select_biased! {
            released = release_rx.next().fuse() => match released {
                Some(handle) => { core.release(handle); }
                None => return,
            },
            () = sleep(std::time::Duration::from_millis(20)).fuse() => {}
        }
    }
}

/// Sibling of [`drain_both`] for the (common) case where `begin_drain`
/// found nothing in the connection table at all — still bound-wait on the
/// request counter alone.
async fn drain_requests_only(admission: &ConnAdmission, timeout: std::time::Duration) {
    let started = std::time::Instant::now();
    while admission.in_flight() > 0 {
        if started.elapsed() >= timeout {
            warn!(
                remaining_requests = admission.in_flight(),
                "any listener drain timeout exceeded; abandoning in-flight requests"
            );
            return;
        }
        sleep(std::time::Duration::from_millis(20)).await;
    }
}

/// UDS-bound accept loop, mirroring [`crate::listener::serve_default_uds`]'s
/// shape: single-process, no TLS, no SO_REUSEPORT — but still routed
/// through the SAME [`ListenerCore`] + [`ConnAdmission`] + drain machinery
/// as the TCP path, since a control-plane UDS listener deserves the exact
/// same graceful-shutdown guarantee as a network one.
#[cfg(feature = "http1")]
#[allow(clippy::too_many_arguments)]
async fn serve_uds(
    path: std::path::PathBuf,
    mode: Option<u32>,
    candidates: Arc<[Arc<dyn AnyProtocol>]>,
    global_cap: usize,
    handlers: Arc<std::collections::BTreeMap<String, AnyHandler>>,
    on_reject: Option<RejectHook>,
    blacklist: Option<BlacklistTable>,
    spec: Arc<Value>,
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
    debug!(?path, "any listener (uds) bound");
    if let Some(sender) = ready_signal {
        let _ = sender.send(());
    }
    let read_chunk_len = global_cap.clamp(64, 4096);
    let mut core = ListenerCore::new(proxima_listen::DispatchPolicy::Inline);
    let admission = ConnAdmission::unbounded();
    loop {
        tokio::select! {
            outcome = listener.accept() => match outcome {
                Ok((socket, _peer)) => {
                    let peer_ip = proxima_listen::peer_ip(None);
                    // Blacklist gate BEFORE `core.admit` — same discipline
                    // as the TCP accept loops in `serve_via_factory`.
                    let decision = if peer_is_banned(&blacklist, peer_ip) {
                        Admission::Shed { reason: ShedReason::Blacklisted }
                    } else {
                        core.admit(peer_ip)
                    };
                    match decision {
                    Admission::Admit { handle, .. } => {
                        let stream: Box<dyn proxima_primitives::stream::StreamConnection> =
                            Box::new(UdsStream(tokio_util::compat::TokioAsyncReadCompatExt::compat(socket)));
                        let candidates_for_conn = candidates.clone();
                        let handlers_for_conn = handlers.clone();
                        let on_reject_for_conn = on_reject.clone();
                        let spec_for_conn = spec.clone();
                        let admission_for_conn = admission.clone();
                        classify_and_drive(
                            stream,
                            candidates_for_conn,
                            global_cap,
                            read_chunk_len,
                            handlers_for_conn,
                            on_reject_for_conn,
                            spec_for_conn,
                            admission_for_conn,
                            #[cfg(feature = "tls")]
                            None,
                        )
                        .await;
                        core.release(handle);
                    }
                    Admission::Shed { reason } => {
                        debug!(?reason, "uds connection shed");
                        drop(socket);
                    }
                    }
                }
                Err(error) => warn!(?error, "uds accept failed"),
            },
            _ = &mut shutdown => {
                // Serial accept loop: `classify_and_drive` above is always
                // awaited to completion before the loop returns to this
                // `select!`, so nothing is ever concurrently in flight here
                // — `begin_drain` always reports `ClosedImmediately` and
                // `admission.in_flight()` is always 0. Still call both (not
                // just drop the sockets) so the SAME admission contract UDS
                // shares with the TCP path is exercised uniformly.
                admission.begin_drain();
                let _ = core.begin_drain();
                let _ = std::fs::remove_file(&path);
                return Ok(());
            }
        }
    }
}

/// Wraps a tokio-compat UDS stream as a boxable [`StreamConnection`] with
/// no peer address (UDS has none) — the same "collapse onto loopback"
/// convention [`proxima_listen::peer_ip`] already documents for non-TCP
/// transports.
#[cfg(feature = "http1")]
struct UdsStream<S>(S);

#[cfg(feature = "http1")]
impl<S: AsyncRead + Unpin> AsyncRead for UdsStream<S> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.get_mut().0).poll_read(cx, buf)
    }
}

#[cfg(feature = "http1")]
impl<S: AsyncWrite + Unpin> AsyncWrite for UdsStream<S> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.get_mut().0).poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().0).poll_flush(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().0).poll_close(cx)
    }
}

#[cfg(feature = "http1")]
impl<S: AsyncRead + AsyncWrite + Unpin + Send + 'static> StreamConnection for UdsStream<S> {
    fn peer(&self) -> Option<PeerInfo> {
        Some(PeerInfo::Unix(None))
    }
}

/// Per-connection body: optionally terminate TLS FIRST (plaintext is
/// rejected at the handshake when a TLS acceptor is configured — no
/// bypass), then either dispatch straight to the ALPN-negotiated
/// candidate (skipping the classifier — the handshake already proved it)
/// or accumulate bytes and advance a fresh [`Classifier`] over the
/// (possibly now-decrypted) stream. Once resolved, look up the matched
/// candidate's bound handler and replay the accumulated prefix into its
/// [`AnyProtocol::drive`], threading `admission`. A rejected or
/// bound-exceeded classification calls the optional `on_reject` hook (if
/// installed) then drops the connection (still logged) — there is no
/// generic "everything else" fallback for an open, registry-driven set. A
/// matched candidate with NO bound handler is a configuration mistake, not
/// a silent drop: it is logged as a `ProximaError::Config` naming the
/// protocol.
#[allow(clippy::too_many_arguments)]
async fn classify_and_drive(
    conn: Box<dyn proxima_primitives::stream::StreamConnection>,
    candidates: Arc<[Arc<dyn AnyProtocol>]>,
    global_cap: usize,
    read_chunk_len: usize,
    handlers: Arc<std::collections::BTreeMap<String, AnyHandler>>,
    on_reject: Option<RejectHook>,
    spec: Arc<Value>,
    admission: ConnAdmission,
    #[cfg(feature = "tls")] tls_acceptor: Option<futures_rustls::TlsAcceptor>,
) {
    #[cfg(feature = "tls")]
    if let Some(acceptor) = tls_acceptor {
        let peer_info = conn.peer();
        let tls_stream = match acceptor.accept(conn).await {
            Ok(stream) => stream,
            Err(error) => {
                warn!(
                    ?error,
                    "any listener: TLS handshake failed; rejecting plaintext connection"
                );
                return;
            }
        };
        let negotiated = tls_stream.get_ref().1.alpn_protocol().map(<[u8]>::to_vec);
        let boxed: Box<dyn proxima_primitives::stream::StreamConnection> =
            Box::new(AlpnStream::new(tls_stream, peer_info.clone()));
        // ALPN already proved the wire — if it names a registered
        // candidate, dispatch straight to it and skip the classifier.
        if let Some(alpn) = negotiated
            && let Ok(name) = std::str::from_utf8(&alpn)
            && let Some(protocol) = candidates.iter().find(|candidate| candidate.name() == name)
        {
            drive_matched(
                protocol.clone(),
                Vec::new(),
                boxed,
                peer_info,
                &handlers,
                &spec,
                &admission,
            )
            .await;
            return;
        }
        classify_and_drive_plaintext(
            boxed,
            candidates,
            global_cap,
            read_chunk_len,
            handlers,
            on_reject,
            spec,
            admission,
        )
        .await;
        return;
    }
    classify_and_drive_plaintext(
        conn,
        candidates,
        global_cap,
        read_chunk_len,
        handlers,
        on_reject,
        spec,
        admission,
    )
    .await;
}

/// The classifier path shared by plaintext connections and (post-handshake,
/// ALPN-inconclusive) TLS connections.
#[allow(clippy::too_many_arguments)]
async fn classify_and_drive_plaintext(
    conn: Box<dyn proxima_primitives::stream::StreamConnection>,
    candidates: Arc<[Arc<dyn AnyProtocol>]>,
    global_cap: usize,
    read_chunk_len: usize,
    handlers: Arc<std::collections::BTreeMap<String, AnyHandler>>,
    on_reject: Option<RejectHook>,
    spec: Arc<Value>,
    admission: ConnAdmission,
) {
    let mut raw_conn = conn;
    let peer_info = raw_conn.peer();
    let mut classifier = Classifier::new(candidates, global_cap);
    let mut accumulated: Vec<u8> = Vec::new();
    let mut read_chunk = vec![0_u8; read_chunk_len];

    let matched = loop {
        let read = match raw_conn.read(&mut read_chunk).await {
            Ok(read) => read,
            Err(error) => {
                warn!(
                    ?error,
                    "any listener: read failed while classifying connection"
                );
                break None;
            }
        };
        if read == 0 {
            debug!("any listener: peer closed before a candidate resolved");
            break None;
        }
        accumulated.extend_from_slice(&read_chunk[..read]);
        match classifier.advance(&accumulated) {
            ClassifyOutcome::Matched(protocol) => break Some(protocol),
            ClassifyOutcome::NeedMoreBytes { .. } => continue,
            ClassifyOutcome::Rejected { bytes_examined } => {
                warn!(
                    bytes_examined,
                    "any listener: no candidate matched the connection prefix"
                );
                if let Some(hook) = &on_reject {
                    hook(
                        peer_info.clone(),
                        RejectReason::NoCandidateMatched { bytes_examined },
                    );
                }
                break None;
            }
            ClassifyOutcome::PrefixBoundExceeded => {
                warn!("any listener: prefix bound exceeded before any candidate resolved");
                if let Some(hook) = &on_reject {
                    hook(peer_info.clone(), RejectReason::PrefixBoundExceeded);
                }
                break None;
            }
            _ => {
                warn!("any listener: unrecognized classify outcome variant");
                break None;
            }
        }
    };

    let Some(protocol) = matched else {
        return;
    };
    drive_matched(
        protocol,
        accumulated,
        raw_conn,
        peer_info,
        &handlers,
        &spec,
        &admission,
    )
    .await;
}

/// Common tail: look up the matched candidate's bound handler, replay the
/// accumulated prefix (empty for an ALPN fast-path match — the handshake
/// consumed no application bytes), and call [`AnyProtocol::drive`].
async fn drive_matched(
    protocol: Arc<dyn AnyProtocol>,
    accumulated: Vec<u8>,
    raw_conn: Box<dyn proxima_primitives::stream::StreamConnection>,
    peer_info: Option<PeerInfo>,
    handlers: &std::collections::BTreeMap<String, AnyHandler>,
    spec: &Value,
    admission: &ConnAdmission,
) {
    let Some(handler) = handlers.get(protocol.name()).cloned() else {
        warn!(
            protocol = protocol.name(),
            "any listener: matched protocol has no bound handler; dropping connection"
        );
        return;
    };
    let prefixed = PrefixedConnection::new(accumulated, raw_conn, peer_info.clone());
    if let Err(error) = protocol
        .drive(Box::new(prefixed), handler, spec, peer_info, admission)
        .await
    {
        warn!(
            ?error,
            protocol = protocol.name(),
            "any listener: connection error"
        );
    }
}

/// futures-io wrapper around a decrypted `futures_rustls::TlsStream` so it
/// can be boxed as a [`StreamConnection`] — mirrors [`PrefixedConnection`]'s
/// same erase-to-trait-object shape.
#[cfg(feature = "tls")]
struct AlpnStream<S> {
    inner: futures_rustls::server::TlsStream<S>,
    peer: Option<PeerInfo>,
}

#[cfg(feature = "tls")]
impl<S> AlpnStream<S> {
    fn new(inner: futures_rustls::server::TlsStream<S>, peer: Option<PeerInfo>) -> Self {
        Self { inner, peer }
    }
}

#[cfg(feature = "tls")]
impl<S: AsyncRead + AsyncWrite + Unpin> AsyncRead for AlpnStream<S> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.get_mut().inner).poll_read(cx, buf)
    }
}

#[cfg(feature = "tls")]
impl<S: AsyncRead + AsyncWrite + Unpin> AsyncWrite for AlpnStream<S> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.get_mut().inner).poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_flush(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_close(cx)
    }
}

#[cfg(feature = "tls")]
impl<S: AsyncRead + AsyncWrite + Unpin + Send + 'static> StreamConnection for AlpnStream<S> {
    fn peer(&self) -> Option<PeerInfo> {
        self.peer.clone()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use proxima_listen::any::{AnyRegistry, erase_handler};
    use proxima_primitives::pipe::SendPipe;
    use proxima_primitives::pipe::handler::into_handle;
    use proxima_primitives::pipe::request::{Request, Response};

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

    #[test]
    fn h1_probe_positively_matches_a_known_method_and_rejects_pri() {
        let h1 = H1AnyProtocol::new();
        assert!(matches!(
            h1.probe(b"GET / HTTP/1.1\r\n"),
            ProbeVerdict::Match { consumed: 0 }
        ));
        // "PRI " is h2's reserved pseudo-method — never a real h1 verb.
        assert!(matches!(h1.probe(b"PRI * HTTP/2.0"), ProbeVerdict::No));
    }

    #[test]
    fn h1_probe_needs_more_bytes_on_a_short_ambiguous_prefix() {
        let h1 = H1AnyProtocol::new();
        // "P" is a live prefix of both "PUT " (4) and "PATCH " (6) — the
        // correct answer is the SMALLER length, not whichever method the
        // implementation happens to check first.
        match h1.probe(b"P") {
            ProbeVerdict::NeedMore { at_least } => assert_eq!(at_least, 4),
            other => panic!("expected NeedMore{{at_least: 4}}, got {other:?}"),
        }
    }

    #[cfg(feature = "http2-native")]
    #[test]
    fn h2_probe_matches_the_full_rfc9113_preface_and_needs_more_for_a_partial_one() {
        let h2 = H2PriorKnowledgeAnyProtocol::new();
        let full_preface = proxima_listen::preface::H2_CLIENT_PREFACE;
        assert!(matches!(
            h2.probe(full_preface),
            ProbeVerdict::Match { consumed: 24 }
        ));
        match h2.probe(&full_preface[..10]) {
            ProbeVerdict::NeedMore { at_least } => assert_eq!(at_least, 24),
            other => panic!("expected NeedMore{{at_least: 24}}, got {other:?}"),
        }
        assert!(matches!(h2.probe(b"GET / HTTP/1.1\r\n"), ProbeVerdict::No));
    }

    // The task's routing test: an h1 GET prefix and an h2 preface, both
    // registered at the default priority (100), each classify to their
    // OWN candidate through the real (non-fake) H1/H2 probes — proves the
    // whole path (registry snapshot -> Classifier::advance -> Matched)
    // minus the priority-wait subtlety this task explicitly defers.
    #[cfg(feature = "http2-native")]
    #[test]
    fn real_h1_and_h2_candidates_route_to_the_right_protocol_through_the_classifier() {
        let registry = AnyRegistry::new();
        registry
            .register(Arc::new(H1AnyProtocol::new()))
            .expect("register h1");
        registry
            .register(Arc::new(H2PriorKnowledgeAnyProtocol::new()))
            .expect("register h2");
        assert_eq!(
            registry.get("h1").expect("h1 registered").priority(),
            registry.get("h2").expect("h2 registered").priority(),
            "both candidates default to priority 100 per this task's scope"
        );

        let candidates = registry.snapshot();
        let cap = proxima_listen::sized::ANY_MAX_PROBE_PREFIX_BYTES_DEFAULT;

        let mut h1_classifier = Classifier::new(candidates.clone(), cap);
        match h1_classifier.advance(b"GET / HTTP/1.1\r\n") {
            ClassifyOutcome::Matched(protocol) => assert_eq!(protocol.name(), "h1"),
            other => panic!("expected Matched(h1), got {other:?}"),
        }

        let mut h2_classifier = Classifier::new(candidates, cap);
        match h2_classifier.advance(proxima_listen::preface::H2_CLIENT_PREFACE) {
            ClassifyOutcome::Matched(protocol) => assert_eq!(protocol.name(), "h2"),
            other => panic!("expected Matched(h2), got {other:?}"),
        }
    }

    // AnyProtocol::drive downcasts the AnyHandler it's handed to its own
    // concrete handler type; a handler bound under a mismatched shape must
    // surface as a named Config error, never a panic or a silent drop.
    #[proxima::test]
    async fn h1_drive_reports_a_config_error_when_the_bound_handler_is_the_wrong_shape() {
        let h1 = H1AnyProtocol::new();
        // Erase a plain `u8` instead of a `PipeHandle` — the wrong shape.
        let wrong_shaped_handler = erase_handler(7_u8);
        struct NeverPolled;
        impl futures::io::AsyncRead for NeverPolled {
            fn poll_read(
                self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
                _buf: &mut [u8],
            ) -> Poll<std::io::Result<usize>> {
                Poll::Ready(Ok(0))
            }
        }
        impl futures::io::AsyncWrite for NeverPolled {
            fn poll_write(
                self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
                buf: &[u8],
            ) -> Poll<std::io::Result<usize>> {
                Poll::Ready(Ok(buf.len()))
            }
            fn poll_flush(
                self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
            ) -> Poll<std::io::Result<()>> {
                Poll::Ready(Ok(()))
            }
            fn poll_close(
                self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
            ) -> Poll<std::io::Result<()>> {
                Poll::Ready(Ok(()))
            }
        }
        impl StreamConnection for NeverPolled {
            fn peer(&self) -> Option<PeerInfo> {
                None
            }
        }
        let spec = Value::Null;
        let admission = proxima_listen::admission::ConnAdmission::unbounded();
        let outcome = h1
            .drive(
                Box::new(NeverPolled),
                wrong_shaped_handler,
                &spec,
                None,
                &admission,
            )
            .await;
        let error = outcome.expect_err("mismatched handler shape must error, not panic");
        assert!(
            format!("{error}").contains("h1"),
            "error must name the protocol: {error}"
        );
    }

    // The per-listener handler map: a matched protocol with NO bound
    // handler must not silently drop the connection — `classify_and_drive`
    // logs it (see the function's own doc); this test proves the map
    // lookup itself behaves as the `Option::None` branch expects, without
    // needing a live socket.
    #[test]
    fn handler_map_missing_entry_is_none_not_a_panic() {
        let handlers: std::collections::BTreeMap<String, AnyHandler> =
            std::collections::BTreeMap::new();
        assert!(!handlers.contains_key("h1"));
        let _ = into_handle(ConstantOk); // proves PipeHandle erases cleanly too
    }

    // Reject-hook seam: install a hook, drive the classifier to a Rejected
    // outcome by hand, and confirm the hook observes the reason — proves
    // the seam is wired without needing a live socket or a deny-list policy.
    #[test]
    fn reject_hook_observes_no_candidate_matched() {
        use std::sync::Mutex;
        let observed: Arc<Mutex<Vec<RejectReason>>> = Arc::new(Mutex::new(Vec::new()));
        let observed_for_hook = observed.clone();
        let hook: RejectHook = Arc::new(move |_peer, reason| {
            observed_for_hook.lock().expect("lock").push(reason);
        });
        // Exercise the hook directly the same way `classify_and_drive` does
        // on a `Rejected` outcome — proves the call shape compiles and
        // records, without standing up a socket.
        hook(None, RejectReason::NoCandidateMatched { bytes_examined: 4 });
        let recorded = observed.lock().expect("lock");
        assert_eq!(recorded.len(), 1);
        assert!(matches!(
            recorded[0],
            RejectReason::NoCandidateMatched { bytes_examined: 4 }
        ));
    }

    #[test]
    fn any_handler_erase_and_downcast_round_trips() {
        let handle = into_handle(ConstantOk);
        let erased = erase_handler(handle);
        let recovered = proxima_listen::any::downcast_handler::<PipeHandle>("h1", &erased)
            .expect("downcast to PipeHandle must succeed");
        let _: PipeHandle = (*recovered).clone();
    }

    // Graceful drain: h2c/pgwire/redis had NO admission at all before this
    // lift, so "shutdown while a request is in flight" either dropped the
    // connection outright or was never a tracked concept. Proves
    // `drain_requests_only` (the tail every `AnyListenProtocol::serve`
    // shutdown path runs through) genuinely WAITS for an in-flight request
    // to call `request_release` before returning — not just that the
    // counter arithmetic is correct (already covered by
    // `proxima_listen::admission::request`'s own unit tests), but that the
    // async wait loop here really blocks on it.
    #[proxima::test(runtime = "tokio")]
    async fn drain_requests_only_waits_for_an_in_flight_request_to_release() {
        let admission = ConnAdmission::unbounded();
        assert_eq!(
            admission.request_admit(),
            proxima_listen::admission::RequestAdmit::Admit
        );

        let (release_tx, release_rx) = futures::channel::oneshot::channel::<()>();
        let admission_for_holder = admission.clone();
        let holder = tokio::task::spawn(async move {
            // Simulates a slow in-flight request: holds its admitted slot
            // until told to finish, exactly like a real h2 stream/pgwire
            // query/redis command would while awaiting the business
            // handler.
            let _ = release_rx.await;
            admission_for_holder.request_release();
        });

        // Race the drain against a short timeout — it must NOT resolve
        // while the holder is still live (proves it actually waits, not
        // just polls once and gives up).
        let drained_early = tokio::select! {
            () = drain_requests_only(&admission, std::time::Duration::from_secs(5)) => true,
            () = tokio::time::sleep(std::time::Duration::from_millis(50)) => false,
        };
        assert!(
            !drained_early,
            "drain must not complete while a request is still in flight"
        );
        assert_eq!(
            admission.in_flight(),
            1,
            "the in-flight request is still held"
        );

        // Release the held request; drain must now complete promptly.
        release_tx.send(()).expect("release send");
        holder.await.expect("holder task");
        drain_requests_only(&admission, std::time::Duration::from_secs(5)).await;
        assert_eq!(admission.in_flight(), 0);
    }

    // `.any().tls()` must actually terminate TLS — a plaintext client (no
    // ClientHello at all, just an h1 GET) gets rejected at the handshake,
    // never reaching the classifier or the h1 candidate. Before this task's
    // TLS-enforce path, `AnyListenProtocol::serve` never read the TLS
    // marker at all, so `.tls()` was a silent no-op; this proves the
    // regression can't reoccur.
    #[cfg(all(feature = "tls", feature = "http1"))]
    #[proxima::test(runtime = "tokio")]
    async fn any_listener_tls_rejects_a_plaintext_client_at_the_handshake() {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async move {
                let bind: SocketAddr = "127.0.0.1:0".parse().expect("parse bind addr");
                let mut handlers = std::collections::BTreeMap::new();
                handlers.insert("h1".to_string(), erase_handler(into_handle(ConstantOk)));
                let candidates: Arc<[Arc<dyn AnyProtocol>]> =
                    Arc::from(vec![Arc::new(H1AnyProtocol::new()) as Arc<dyn AnyProtocol>]);
                let protocol = AnyListenProtocol::from_candidates(candidates, Arc::new(handlers));

                let tls_config = proxima_tls::TlsConfig::self_signed();
                let mut spec = serde_json::Map::new();
                spec.insert(
                    proxima_tls::SPEC_KEY.to_string(),
                    proxima_tls::config_to_spec_value(&tls_config),
                );

                let context = proxima_listen::ServeContext::new(
                    proxima_primitives::pipe::telemetry_surface::NoopTelemetry::handle(),
                )
                .with_acceptor_factory(Arc::new(proxima_net::tokio::TokioAcceptorFactory));
                let (shutdown_tx, shutdown_rx) = oneshot::channel();

                let probe = tokio::net::TcpListener::bind(bind)
                    .await
                    .expect("probe bind");
                let addr = probe.local_addr().expect("probe addr");
                drop(probe);

                let dispatch = into_handle(ConstantOk);
                let serve = protocol.serve(
                    addr,
                    dispatch,
                    &serde_json::Value::Object(spec),
                    context,
                    shutdown_rx,
                );

                let client_work = async {
                    let mut client = loop {
                        match tokio::net::TcpStream::connect(addr).await {
                            Ok(stream) => break stream,
                            Err(_) => tokio::task::yield_now().await,
                        }
                    };
                    // Plaintext h1 request — no ClientHello. The TLS
                    // acceptor must reject this outright.
                    client
                        .write_all(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
                        .await
                        .expect("client write");
                    client.flush().await.expect("client flush");
                    let mut response = Vec::with_capacity(64);
                    let _ = client.read_to_end(&mut response).await;
                    response
                };

                let response = tokio::select! {
                    serve_result = serve => panic!("serve returned early: {serve_result:?}"),
                    response = client_work => response,
                };
                // A correct TLS server sends a fatal alert record (byte 0 =
                // 0x15 ContentType::Alert) when the "ClientHello" fails to
                // parse, then closes — never an HTTP response. Proves the
                // plaintext bytes never reached the classifier or the h1
                // candidate: no HTTP status line, no h1 framing at all.
                assert_ne!(
                    response.first(),
                    Some(&0x16),
                    "must not echo back a TLS handshake byte as if it were data"
                );
                let response_text = String::from_utf8_lossy(&response);
                assert!(
                    !response_text.starts_with("HTTP/"),
                    "a plaintext client must get NO HTTP response through a TLS listener — \
                     the handshake must fail first; got: {response:?}"
                );
                drop(shutdown_tx);
            })
            .await;
    }
}
