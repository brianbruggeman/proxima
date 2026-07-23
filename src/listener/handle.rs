use std::net::SocketAddr;
use std::sync::Arc;
#[cfg(feature = "tls")]
use std::{future::Future, pin::Pin};

#[cfg(feature = "tls")]
use futures::channel::oneshot;
use serde_json::Value;

use proxima_listen::ListenProtocol;
#[cfg(feature = "tls")]
use proxima_listen::ServeContext;
use proxima_listen::handle::Listener;

use crate::app::{App, MountTarget, RunConfig};
use crate::error::ProximaError;
use crate::listener::protocol::ListenerProtocolExt;
use crate::pipe::PipeHandle;
use crate::server::Server;

/// Gives the real listen-side primitive — [`Listener`]
/// (`proxima-listen/src/handle.rs`, produced by `ListenerSpec::attach(dispatch)`,
/// run via `Listener::run_with_runtime`) — the `Listener::builder()` /
/// `Listener::http(bind)` entry points mirroring
/// [`Client::builder()`](crate::Client::builder) /
/// [`Client::http(url)`](crate::Client::http). `Listener` is defined in
/// `proxima-listen`, a crate this one depends on but that cannot depend back
/// on [`App`] / [`Server`]; Rust's orphan rule forbids an inherent `impl
/// Listener { fn builder() }` from this crate, so the entry points are a
/// local trait blanket-impl'd for the foreign type instead of a second
/// `Listener` type living here — same idiom as the TYPE-SPECIFIC
/// [`ListenerTransportExt`](crate::ListenerTransportExt) /
/// [`ListenerProtocolExt`] extension traits: import the trait to unlock the
/// static methods, exactly like those traits unlock `.tcp()`/`.http()`.
/// Bring it into scope with `use proxima::{Listener, ListenerBuilderEntry};`.
pub trait ListenerBuilderEntry {
    /// Fluent builder: `Listener::builder().bind(addr).tcp().handle(pipe).serve()`.
    #[must_use]
    fn builder() -> ListenerBuilder;

    /// One-liner mirroring [`Client::http(url)`](crate::Client::http): binds
    /// the `ListenerBuilder`'s typed `.bind(addr)` slot AND the
    /// [`ListenerProtocolExt::http`] spec key (`bind.to_string()`) in one
    /// call, so
    /// `Listener::http(bind).tls(cfg).handle(pipe).serve()` reads exactly
    /// like `Client::builder().http(url).tls().build()`. Still requires
    /// `.handle(pipe)` before `.serve()` — the one input a listener carries
    /// that a client never does.
    #[must_use]
    fn http(bind: SocketAddr) -> ListenerBuilder;
}

impl ListenerBuilderEntry for Listener {
    fn builder() -> ListenerBuilder {
        ListenerBuilder::default()
    }

    fn http(bind: SocketAddr) -> ListenerBuilder {
        ListenerBuilder::default().bind(bind).http(bind.to_string())
    }
}

/// Fluent builder for [`Listener`] — accumulates a spec `serde_json::Map` the
/// exact same way `ClientBuilder` does (see `crate::client::handle`), via
/// `impl SpecBuilder` below. [`ListenerTransportExt`](crate::ListenerTransportExt)
/// (`.tcp`/`.udp`/`.quic`) and [`ListenerProtocolExt`]
/// (`.http`/`.https`/`.grpc`/`.kafka`/…) are its OWN type-specific extension
/// traits (no blanket impl over every `SpecBuilder` — `ClientBuilder` gets
/// its own, separate `ClientTransportExt`/`ClientProtocolExt`, not these
/// same ones). A few axes are honestly asymmetric and shadow or extend the
/// trait method with an inherent one carrying more than a client ever
/// needs: `.tls(TlsConfig)` (real cert material, no trait minted at all —
/// see [`Self::tls`]), `.grpc()` (url-less — a listener dispatches to a
/// `.handle(pipe)` already on hand, it doesn't dial out), `.h2()` (inherent,
/// no client twin), and `.pgwire(query)` (real query engine — see its own
/// doc). `.proxy(url)` has no listener meaning at all (it lives only on
/// `ClientTransportExt`) but `.serve()` still hard-errors if a caller
/// reaches it through `.spec("proxy", ..)` directly — see
/// `reject_dead_axes`. `.bind()`/`.handle()` are the listener-specific axes,
/// where the client instead has a url baked into `.http(url)` plus `.auth()`.
#[derive(Default)]
pub struct ListenerBuilder {
    spec: serde_json::Map<String, Value>,
    bind: Option<SocketAddr>,
    dispatch: Option<PipeHandle>,
    /// Accumulated separately from `spec` — TLS composes as a
    /// `proxima_listen::TlsListenProtocol` DECORATOR at `.serve()` time
    /// (wrapping whichever protocol `resolve_listen_protocol` resolves),
    /// not a spec key or a field carried on `proxima_listen::ListenerSpec`.
    /// See [`Self::tls`].
    #[cfg(feature = "tls")]
    tls: Option<proxima_tls::TlsConfig>,
    /// The typed SQL engine `.pgwire(query)` carries — accumulated
    /// separately from `spec` for the same reason `tls` is: a
    /// `proxima_pgwire::PgPipeHandle` doesn't fit a `serde_json::Value` spec
    /// key, it's a real handle `.serve()` needs in hand. See
    /// [`Self::pgwire`].
    #[cfg(feature = "pgwire")]
    pgwire_query: Option<proxima_pgwire::PgPipeHandle>,
    /// The typed query handler `.dns(handler)` carries — accumulated
    /// separately from `spec` for the same reason `pgwire_query` is: a
    /// `proxima_dns::DnsPipeHandle` doesn't fit a `serde_json::Value` spec
    /// key. `.dns()` is the one dual-transport axis: `.serve()` branches on
    /// `spec["transport"]` to pick a TCP single-candidate `AnyListenProtocol`
    /// vs. a UDP `DatagramProtocolListenProtocol` — see
    /// [`ListenerProtocolExt::dns`].
    #[cfg(feature = "dns-listener")]
    dns_handler: Option<proxima_dns::DnsPipeHandle>,
    /// The reusable post-handshake handler `.websocket(handler)` carries —
    /// wraps `dispatch` at `.serve()` time (see
    /// `crate::listener::websocket::wrap_dispatch`) rather than resolving
    /// through `resolve_listen_protocol`/`.protocol()`: a WebSocket upgrade
    /// is ordinary H1 until the 101 response, not a peer wire protocol. See
    /// [`ListenerProtocolExt::websocket`].
    #[cfg(all(
        feature = "websocket-upgrade",
        any(feature = "http1", feature = "http1-native")
    ))]
    websocket_handler: Option<crate::listener::websocket::WebSocketHandler>,
    /// `.any()`/`.accepts(&[..])`/`.accept(name)` — which `AnyProtocol`
    /// candidates the open universal listener accepts. `None` means none of
    /// those were called (the builder resolves through the ordinary
    /// `resolve_listen_protocol` axes instead). See [`AnyMode`].
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    any_mode: Option<AnyMode>,
    /// Per-listener handler overrides for `.any()`, keyed by protocol name
    /// — `.any_handler(name, handler)` populates this; entries here win
    /// over the `App`-level default-handler registry
    /// (`App::any_default_handlers`). See
    /// [`proxima_listen::any::AnyHandler`]'s doc for why the value is
    /// type-erased.
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    any_handlers: std::collections::BTreeMap<String, proxima_listen::any::AnyHandler>,
    /// `.any_on_reject(hook)` — the reject-hook seam threaded onto the
    /// resolved `AnyListenProtocol` (see that type's `with_reject_hook`).
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    any_reject_hook: Option<crate::listeners::RejectHook>,
    /// `.deny(name, literal)`/`.denies([..])` — fixed malicious/scanner
    /// byte literals registered as `DenySignature` candidates ALONGSIDE
    /// whatever `.any()`/`.accepts()`/`.accept()` already selected. See
    /// [`Self::deny`].
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    deny_signatures: Vec<(String, Vec<u8>)>,
    /// `.protocol(impl AnyProtocol)` — externally-defined `AnyProtocol`
    /// candidates, registered into the `App`'s `AnyRegistry` at `.serve()`
    /// time ALONGSIDE whatever `.any()`/`.accepts()`/`.accept()` already
    /// selected — the listener-side mirror of
    /// [`ClientProtocol`](crate::client::handle::ClientProtocol)'s
    /// `.protocol(impl ClientProtocol)`
    /// ([`ClientBuilder::protocol`](crate::client::handle::ClientBuilder::protocol)).
    /// See [`Self::protocol`].
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    extra_protocols: Vec<Arc<dyn proxima_listen::any::AnyProtocol>>,
    /// `.blacklist(config)` — overrides the accept-edge DoS-blacklist's
    /// strike thresholds/window/ban duration. `None` still gets a
    /// default-config `BlacklistTable` at `.serve()` time (a deny needs
    /// somewhere to record even if this was never called). See
    /// [`Self::blacklist`].
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    blacklist_config: Option<proxima_listen::admission::BlacklistConfig>,
}

/// Which `AnyProtocol` candidates `.any()`/`.accepts()`/`.accept()` selected.
/// `All` accepts every candidate the registry currently holds; `Subset`
/// restricts to the named ones (`.accept(name)` is `Subset` with one entry).
#[cfg(any(feature = "http1", feature = "http1-native"))]
#[derive(Debug, Clone, PartialEq, Eq)]
enum AnyMode {
    All,
    Subset(Vec<String>),
}

impl ListenerBuilder {
    // `.tcp()`/`.udp()`/`.quic()` are `ListenerTransportExt`; `.http(bind)`/
    // `.https(bind)`/`.grpc()`/`.kafka()`/… are `ListenerProtocolExt` — both
    // TYPE-SPECIFIC extension traits impl'd below (no blanket over every
    // `SpecBuilder`). `resolve_listen_protocol` below reads the same
    // `transport`/`grpc` spec keys `load.rs` reads on the client side.
    // `.tls()`/`.h2()` stay inherent here (no trait at all for `.tls()` —
    // real cert material; `.h2()` has no client-side twin to fold into a
    // shared trait). `use proxima::{ListenerTransportExt,
    // ListenerProtocolExt};` brings the rest into scope.

    /// The socket address to listen on. Required before `.serve()` unless
    /// `.http(bind.to_string())` (or `Listener::http(bind)`, which calls it
    /// for you) already carried it — see [`bind_from_spec`].
    #[must_use]
    pub fn bind(mut self, addr: SocketAddr) -> Self {
        self.bind = Some(addr);
        self
    }

    /// The dispatch pipe every accepted request routes to — required before
    /// `.serve()`. The listener-side sibling of the client's upstream url:
    /// where `Client` dials OUT to a spec-selected upstream, `Listener`
    /// dispatches IN to this handle.
    #[must_use]
    pub fn handle(mut self, handle: impl Into<PipeHandle>) -> Self {
        self.dispatch = Some(handle.into());
        self
    }

    /// Select h2 (h2c, prior-knowledge, no ALPN, no TLS) as the listen
    /// protocol — the url-less counterpart of a client's transport pick: a
    /// listener dispatches to a `.handle(pipe)` already on hand rather than
    /// dialing an upstream url, so this is an inherent 0-arg method (there
    /// is no client-side url-carrying twin the way `.tls()`/`.quic()` do).
    /// Resolves to the exact same `h2` listen protocol `.grpc()`
    /// ([`ListenerProtocolExt::grpc`]) resolves to — `.grpc()` and `.h2()`
    /// are the same wire, self-registered onto the fresh `App` at
    /// `.serve()` time.
    #[must_use]
    pub fn h2(mut self) -> Self {
        self.spec.insert("h2".to_string(), Value::Bool(true));
        self
    }

    /// Select the open universal listener, accepting EVERY `AnyProtocol`
    /// candidate the `App`'s [`proxima_listen::any::AnyRegistry`] currently
    /// holds (h1, h2 prior-knowledge — see `App::new`'s doc). Each accepted
    /// connection routes to its own candidate's bound handler: a
    /// per-listener `.any_handler(name, handler)` override if present,
    /// else the `App`-level default (`App::register_any_default_handler`),
    /// else — for a candidate whose expected handler type happens to be a
    /// [`PipeHandle`](crate::pipe::PipeHandle) (h1/h2's shape) — the
    /// `.handle(pipe)` this builder already required, erased. A candidate
    /// with no handler resolvable ANY of those three ways logs a named
    /// config error per connection rather than silently dropping it — see
    /// `proxima_http::any_listener::classify_and_drive`'s doc.
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[must_use]
    pub fn any(mut self) -> Self {
        self.any_mode = Some(AnyMode::All);
        self
    }

    /// Restrict the open universal listener to a named subset of
    /// registered candidates — otherwise identical to [`Self::any`].
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[must_use]
    pub fn accepts(mut self, names: &[&str]) -> Self {
        self.any_mode = Some(AnyMode::Subset(
            names.iter().map(|name| (*name).to_string()).collect(),
        ));
        self
    }

    /// Restrict the open universal listener to exactly one registered
    /// candidate — sugar over [`Self::accepts`] with a single name.
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[must_use]
    pub fn accept(self, name: &str) -> Self {
        self.accepts(&[name])
    }

    /// Plug in an out-of-crate `AnyProtocol` candidate for the open
    /// universal listener — the listener-side mirror of
    /// [`ClientProtocol`](crate::client::handle::ClientProtocol)'s
    /// [`.protocol(impl ClientProtocol)`](crate::client::handle::ClientBuilder::protocol):
    /// the typed, no-import path to extend `Listener::builder()` with a
    /// kafka/mqtt/private-wire candidate a downstream crate defines, with no
    /// edit here. Wraps `protocol` in the `Arc<dyn AnyProtocol>`
    /// [`proxima_listen::any::AnyRegistry::register`] wants and adds its own
    /// [`AnyProtocol::name`](proxima_listen::any::AnyProtocol::name) to the
    /// selected set, composing with `.any()`/`.accepts()`/`.accept()`
    /// exactly like [`Self::deny`] does: `None` implicitly selects every
    /// currently-registered candidate PLUS this one (so a caller who only
    /// calls `.protocol(..)` doesn't also have to remember `.any()`), an
    /// existing `All` stays `All`, and an existing `Subset` only gains this
    /// name rather than being narrowed to it. The candidate itself is
    /// registered into `App::any_registry()` inside [`any_listen_protocol`]
    /// at `.serve()` time, the same place `.deny()`'s `DenySignature`
    /// candidates register — `ListenerBuilder` cannot register any earlier
    /// since the `App` (and its registry) doesn't exist until `.serve()`
    /// creates one.
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[must_use]
    pub fn protocol(mut self, protocol: impl proxima_listen::any::AnyProtocol) -> Self {
        let protocol: Arc<dyn proxima_listen::any::AnyProtocol> = Arc::new(protocol);
        let name = protocol.name().to_string();
        self.any_mode = Some(match self.any_mode.take() {
            None | Some(AnyMode::All) => AnyMode::All,
            Some(AnyMode::Subset(mut names)) => {
                if !names.contains(&name) {
                    names.push(name);
                }
                AnyMode::Subset(names)
            }
        });
        self.extra_protocols.push(protocol);
        self
    }

    /// Bind an explicit per-protocol handler for the open universal
    /// listener, by that protocol's registered name (`"h1"`, `"h2"`, or a
    /// future candidate's own name). `handler` is erased via
    /// [`proxima_listen::any::erase_handler`] and downcast back inside that
    /// SAME candidate's own `AnyProtocol::drive` — see that trait's doc for
    /// why the handler type isn't fixed to [`PipeHandle`](crate::pipe::PipeHandle).
    /// Calling this WITHOUT a prior `.any()`/`.accepts()`/`.accept()`
    /// implicitly restricts the listener to just the named protocol
    /// (mirroring `.pgwire(query)` carrying its own engine without a
    /// separate "select pgwire" call) — call `.any()` first if you want
    /// every candidate accepted with only some of them overridden.
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[must_use]
    pub fn any_handler<T: Send + Sync + 'static>(
        mut self,
        name: impl Into<String>,
        handler: T,
    ) -> Self {
        let name = name.into();
        self.any_mode = Some(match self.any_mode.take() {
            None => AnyMode::Subset(vec![name.clone()]),
            Some(AnyMode::Subset(mut names)) => {
                if !names.contains(&name) {
                    names.push(name.clone());
                }
                AnyMode::Subset(names)
            }
            Some(AnyMode::All) => AnyMode::All,
        });
        self.any_handlers
            .insert(name, proxima_listen::any::erase_handler(handler));
        self
    }

    /// Install the open universal listener's reject-hook seam (see
    /// [`proxima_http::any_listener::RejectHook`]'s doc) — observes a
    /// connection the classifier dropped before any candidate resolved.
    /// The seam only; no deny-list/blacklist policy is implemented here.
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[must_use]
    pub fn any_on_reject(mut self, hook: crate::listeners::RejectHook) -> Self {
        self.any_reject_hook = Some(hook);
        self
    }

    /// Register a fixed malicious/scanner byte literal as a `DenySignature`
    /// candidate — reviewed ALONGSIDE whatever legit candidates are
    /// selected, never narrowing the listener to just this one: implicitly
    /// selects `.any()` (every registered candidate) if no
    /// `.any()`/`.accepts()`/`.accept()` call has run yet (mirrors
    /// `.any_handler`'s implicit-select convenience), but if a `.accepts()`
    /// `Subset` is already in effect, this only ADDS the deny's own name to
    /// that subset — it never collapses an existing subset down to just
    /// the denies, which would stop the subset's legit candidates from
    /// being classified at all. A match records a `Strike::Deny` against
    /// the connecting peer and drops the connection — no handler dispatch.
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[must_use]
    pub fn deny(mut self, name: impl Into<String>, literal: impl Into<Vec<u8>>) -> Self {
        let name = name.into();
        self.any_mode = Some(match self.any_mode.take() {
            None | Some(AnyMode::All) => AnyMode::All,
            Some(AnyMode::Subset(mut names)) => {
                if !names.contains(&name) {
                    names.push(name.clone());
                }
                AnyMode::Subset(names)
            }
        });
        self.deny_signatures.push((name, literal.into()));
        self
    }

    /// Register several `DenySignature` candidates in one call — sugar over
    /// repeated [`Self::deny`] calls.
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[must_use]
    pub fn denies<Name, Literal>(
        mut self,
        signatures: impl IntoIterator<Item = (Name, Literal)>,
    ) -> Self
    where
        Name: Into<String>,
        Literal: Into<Vec<u8>>,
    {
        for (name, literal) in signatures {
            self = self.deny(name, literal);
        }
        self
    }

    /// Override the accept-edge DoS-blacklist's strike thresholds/window/
    /// ban duration (see [`proxima_listen::admission::BlacklistConfig`]'s
    /// doc). Not required before using [`Self::deny`] — `.serve()` builds a
    /// default-config table regardless, since a deny needs somewhere to
    /// record even when a caller never tunes it.
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[must_use]
    pub fn blacklist(mut self, config: proxima_listen::admission::BlacklistConfig) -> Self {
        self.blacklist_config = Some(config);
        self
    }

    /// Terminate TLS at this listener — the listener-inherent counterpart of
    /// the client's bare, url-less `ClientSecurityExt::tls()` (which only
    /// picks a wire scheme for a url the client already carries). A listener
    /// additionally needs cert material, so this takes the exact
    /// [`proxima_tls::TlsConfig`] type — deliberately NOT a trait method at
    /// all (no `ListenerSecurityExt` is minted; see the module doc). Composes
    /// at `.serve()` time as a `proxima_listen::TlsListenProtocol` DECORATOR
    /// wrapping whatever protocol `resolve_listen_protocol` resolves — TLS is
    /// NOT a spec key and NOT a field on `ListenerSpec`; on/off is the
    /// presence of that wrapper. No new TLS mechanism underneath: the
    /// decorator still stamps the identical `proxima_tls::SPEC_KEY`
    /// (`"__proxima_tls"`) marker `HttpListenProtocol::serve_default` reads
    /// (`proxima-http/src/listener/mod.rs:451`) to build its
    /// `tokio_rustls::TlsAcceptor` — only WHERE that marker gets attached
    /// moved, from this builder's spec map to the decorator's own `serve`.
    #[cfg(feature = "tls")]
    #[must_use]
    pub fn tls(mut self, tls: proxima_tls::TlsConfig) -> Self {
        self.tls = Some(tls);
        self
    }

    /// Merge an arbitrary spec key — the same escape hatch as
    /// `ClientBuilder::spec`.
    #[must_use]
    pub fn spec(mut self, key: impl Into<String>, value: Value) -> Self {
        self.spec.insert(key.into(), value);
        self
    }

    /// Terminal: resolve the accumulated spec to a `ListenProtocol` (via
    /// [`resolve_listen_protocol`] — the listen-side mirror of the client's
    /// `load(Spec)` factory dispatch), bind, and return the running
    /// `Server`. Composes `App::new` + `App::mount` + `App::serve` — the
    /// exact `into_handle(pipe) -> App::new()? -> app.mount(...)? ->
    /// app.serve(...)` idiom `examples/hello` teaches, just automated behind
    /// the builder. No new serve loop.
    ///
    /// Mounts at `"/{*path}"` (the catch-all convention `App::bind_listener`
    /// also uses at `src/app.rs:1055,1111`) so every path routes to
    /// `.handle(pipe)`, not just the literal root.
    ///
    /// Readiness race: unlike `proxima_listen::handle::Listener::run_with_runtime`
    /// (which blocks for a per-lane ready ack before returning), `App::serve`
    /// returns as soon as the listener lane is SPAWNED, before its `serve`
    /// future gets its first poll and runs the real `bind`/`listen` syscalls.
    /// A caller that dials immediately after `.serve()` resolves can race a
    /// not-yet-listening socket (`ECONNREFUSED`) — wiring the same ready-ack
    /// into `App::run_until_signal` is a change to shared, widely-used serve
    /// plumbing, out of scope here; callers needing a synchronization point
    /// today poll-connect with a bounded retry loop (see
    /// `tests/e2e/listener_client_interop.rs`'s `wait_until_listening`).
    pub async fn serve(self) -> Result<Server, ProximaError> {
        reject_dead_axes(&self.spec)?;
        reject_invalid_axis_combinations(&self)?;
        #[cfg(all(feature = "tls", feature = "pgwire"))]
        if self.tls.is_some() && self.pgwire_query.is_some() {
            return Err(ProximaError::Config(
                "Listener::builder(): .pgwire(query) manages its own TLS upgrade \
                 (proxima-pgwire's `listen` feature); .tls(config) would double-wrap it \
                 with the wrong (http) protocol underneath — drop one"
                    .into(),
            ));
        }
        let bind = self.bind.or_else(|| bind_from_spec(&self.spec)).ok_or_else(|| {
            ProximaError::Config(
                "Listener::builder(): .bind(addr) (or .http(bind.to_string())) is required before .serve()".into(),
            )
        })?;
        let dispatch = self.dispatch.ok_or_else(|| {
            ProximaError::Config(
                "Listener::builder(): .handle(pipe) is required before .serve()".into(),
            )
        })?;
        // `.websocket(handler)` wraps the ordinary dispatch pipe BEFORE
        // mount — every non-upgrade request still reaches `dispatch`
        // unchanged, a genuine handshake diverts to `handler` instead. See
        // `crate::listener::websocket`'s module doc.
        #[cfg(all(
            feature = "websocket-upgrade",
            any(feature = "http1", feature = "http1-native")
        ))]
        let websocket_handler = self.websocket_handler;
        #[cfg(all(
            feature = "websocket-upgrade",
            any(feature = "http1", feature = "http1-native")
        ))]
        let dispatch = match websocket_handler {
            Some(handler) => crate::listener::websocket::wrap_dispatch(dispatch, handler),
            None => dispatch,
        };
        #[cfg(feature = "pgwire")]
        let pgwire_query = self.pgwire_query;
        #[cfg(feature = "dns-listener")]
        let dns_handler = self.dns_handler;
        #[cfg(feature = "dns-listener")]
        let dns_transport = self.spec.get("transport").and_then(Value::as_str).map(str::to_string);
        // Built BEFORE protocol resolution (unlike every other axis, which
        // resolves against bare spec data) — `.any()`/`.accepts()`/
        // `.accept()` need `app.any_registry()` /
        // `app.any_default_handlers()` to resolve at all.
        let app = App::new()?;
        #[cfg(any(feature = "http1", feature = "http1-native"))]
        let (protocol, extra_protocol) = match &self.any_mode {
            Some(mode) => any_listen_protocol(
                &app,
                mode,
                &self.any_handlers,
                self.any_reject_hook.clone(),
                dispatch.clone(),
                AnyAxisConfig {
                    deny_signatures: &self.deny_signatures,
                    blacklist_config: self.blacklist_config.clone(),
                    extra_protocols: &self.extra_protocols,
                },
            )?,
            None => resolve_listen_protocol(&self.spec)?,
        };
        #[cfg(not(any(feature = "http1", feature = "http1-native")))]
        let (protocol, extra_protocol) = resolve_listen_protocol(&self.spec)?;
        #[cfg(feature = "tls")]
        let (protocol, extra_protocol) = compose_tls(self.tls, protocol, extra_protocol)?;
        if let Some(protocol) = extra_protocol {
            app.register_listen_protocol(protocol)?;
        }
        // `.pgwire(query)` carries a typed query engine `App::new()`'s
        // static registration set cannot know about ahead of time —
        // register a fresh instance carrying it now, before `.serve()`
        // resolves `protocol` ("pgwire") against the registry. Resolves
        // through `AnyListenProtocol` (a single-candidate `PgWireAnyProtocol`
        // mount, not the standalone `PgWireListenProtocol`) so `.pgwire(query)`
        // gets the SAME real `ListenerCore`/`ConnAdmission` admission and
        // graceful drain every other TCP-stream listener now has —
        // `PgWireListenProtocol` itself stays available (unretired) for
        // direct/registry construction outside this builder.
        #[cfg(feature = "pgwire")]
        if let Some(query) = pgwire_query {
            app.register_listen_protocol(Arc::new(
                crate::listeners::AnyListenProtocol::single_candidate(
                    "pgwire",
                    Arc::new(proxima_pgwire::PgWireAnyProtocol::new("pgwire", query)),
                ),
            ))?;
        }
        // `.dns(handler)` is the one dual-transport axis: `.serve()`
        // branches on `spec["transport"]` and registers a FRESH instance
        // carrying `handler` now, the same way `.pgwire(query)` does —
        // `.quic()` is already rejected by `reject_invalid_axis_combinations`
        // above, so only the tcp/udp arms are live here. `.tcp()` (default)
        // registers a single-candidate `AnyListenProtocol` wrapping
        // `DnsAnyProtocol` (DNS-over-TCP, RFC 1035 §4.2.2 framing); `.udp()`
        // registers a `DatagramProtocolListenProtocol` wrapping
        // `DnsDatagramProtocol`, self-registered the same way the native h3
        // listener is (`h3_native_listen_protocol`).
        #[cfg(feature = "dns-listener")]
        if let Some(handler) = dns_handler {
            match dns_transport.as_deref() {
                Some("udp") => {
                    app.register_listen_protocol(Arc::new(
                        proxima_dns::DnsDatagramProtocol::listen_protocol(
                            "dns",
                            handler,
                            proxima_dns::DnsServerConfig::default(),
                        ),
                    ))?;
                }
                _ => {
                    app.register_listen_protocol(Arc::new(
                        crate::listeners::AnyListenProtocol::single_candidate(
                            "dns",
                            Arc::new(proxima_dns::DnsAnyProtocol::new("dns", handler)),
                        ),
                    ))?;
                }
            }
        }
        app.mount("/{*path}", MountTarget::Handle(dispatch))?;
        let config = RunConfig {
            bind,
            protocol,
            spec: Value::Object(self.spec),
        };
        app.serve(config).await
    }
}

/// The impl lives here (not in `protocol.rs`, where the trait is defined)
/// because `.pgwire()`/`.dns()`/`.websocket()` accumulate onto PRIVATE
/// `ListenerBuilder` fields — Rust requires one coherent `impl Trait for
/// Type` block, so the whole trait impl stays where those fields are
/// declared. `.kafka()`/`.mqtt()`/`.amqp()`/`.memcached()`/`.redis()` only
/// need the already-`pub` `.protocol()` seam; `.http()`/`.https()`/`.grpc()`
/// only need the already-`pub` `.spec()` — those five COULD live in
/// `protocol.rs`, but splitting one trait's methods across two `impl`
/// blocks isn't legal Rust, so everything stays together.
impl ListenerProtocolExt for ListenerBuilder {
    fn http(self, bind: impl Into<String>) -> Self {
        self.spec("http", Value::String(bind.into()))
    }

    fn https(self, bind: impl Into<String>) -> Self {
        self.spec("http", Value::String(bind.into()))
    }

    fn grpc(mut self) -> Self {
        self.spec.insert("grpc".to_string(), Value::Bool(true));
        self
    }

    #[cfg(all(
        feature = "kafka-listener",
        any(feature = "http1", feature = "http1-native")
    ))]
    fn kafka(self, handler: proxima_kafka::KafkaPipeHandle) -> Self {
        self.protocol(proxima_kafka::KafkaAnyProtocol::new("kafka", handler))
    }

    #[cfg(all(
        feature = "mqtt-listener",
        any(feature = "http1", feature = "http1-native")
    ))]
    fn mqtt(self, handler: proxima_mqtt::MqttPipeHandle) -> Self {
        self.protocol(proxima_mqtt::MqttAnyProtocol::new("mqtt", handler))
    }

    #[cfg(all(
        feature = "amqp-listener",
        any(feature = "http1", feature = "http1-native")
    ))]
    fn amqp(self, handler: proxima_amqp::AmqpPipeHandle) -> Self {
        self.protocol(proxima_amqp::AmqpAnyProtocol::new("amqp", handler))
    }

    #[cfg(all(
        feature = "memcached-listener",
        any(feature = "http1", feature = "http1-native")
    ))]
    fn memcached(self, handler: proxima_memcached::MemcachedPipeHandle) -> Self {
        self.protocol(proxima_memcached::MemcachedAnyProtocol::new(
            "memcached", handler,
        ))
    }

    #[cfg(all(
        feature = "redis-listener",
        any(feature = "http1", feature = "http1-native")
    ))]
    fn redis(self, handler: proxima_redis::RedisPipeHandle) -> Self {
        self.protocol(proxima_redis::RedisAnyProtocol::new("redis", handler))
    }

    #[cfg(feature = "pgwire")]
    fn pgwire(mut self, query: proxima_pgwire::PgPipeHandle) -> Self {
        self.pgwire_query = Some(query);
        self.spec
            .insert("pgwire_axis".to_string(), Value::Bool(true));
        self
    }

    #[cfg(feature = "dns-listener")]
    fn dns(mut self, handler: proxima_dns::DnsPipeHandle) -> Self {
        self.dns_handler = Some(handler);
        self.spec.insert("dns_axis".to_string(), Value::Bool(true));
        self
    }

    #[cfg(all(
        feature = "websocket-upgrade",
        any(feature = "http1", feature = "http1-native")
    ))]
    fn websocket(mut self, handler: crate::listener::websocket::WebSocketHandler) -> Self {
        // "Implies `.tcp()`" is a semantic statement, not a spec write: the
        // absence of a `transport` key already resolves to the h1+h2 ALPN
        // combiner (TCP) — see `resolve_listen_protocol`'s default arm.
        // Deliberately NOT calling `.tcp()` here (which would overwrite an
        // EXISTING `.quic()` to "tcp"): `reject_invalid_axis_combinations`
        // needs to still see a caller's own `.quic()` in the spec to reject
        // `.websocket(handler).quic()` — silently downgrading it here would
        // make that combination unobservable instead of a config error.
        self.websocket_handler = Some(handler);
        self
    }
}

/// Runtime validation for invalid axis compositions (Section G of the
/// builder-sugar design) — a NAMED [`ProximaError::Config`], never a silent
/// degrade and never a panic. Typestate can't do this: `.protocol()` is an
/// open seam a third-party `.thrift()` extension joins without this crate
/// ever seeing the concrete type, so the check can only run against the
/// spec/field DATA a builder has accumulated by `.serve()` time — mirrors
/// `reject_dead_axes`, just for the axis PAIRS that `resolve_listen_protocol`
/// and the `.dns()`/`.websocket()` branches can't already refuse on their
/// own.
fn reject_invalid_axis_combinations(builder: &ListenerBuilder) -> Result<(), ProximaError> {
    let transport = builder.spec.get("transport").and_then(Value::as_str);
    let is_quic = transport == Some("quic");
    #[cfg(any(feature = "pgwire", feature = "http1", feature = "http1-native"))]
    let is_udp = transport == Some("udp");

    #[cfg(any(feature = "http1", feature = "http1-native"))]
    if builder.any_mode.is_some() && (is_quic || is_udp) {
        return Err(ProximaError::Config(
            "Listener::builder(): .kafka()/.mqtt()/.amqp()/.memcached()/.redis()/.any()/\
             .accept()/.protocol() are TCP-only (AnyProtocol::drive takes \
             Box<dyn StreamConnection>); combining with .quic()/.udp() has no meaning — \
             use .tcp() (the default)"
                .into(),
        ));
    }

    if (builder.spec.contains_key("grpc") || builder.spec.contains_key("h2")) && is_quic {
        return Err(ProximaError::Config(
            "Listener::builder(): .grpc()/.h2() + .quic(): gRPC rides h2, not QUIC; drop \
             .quic() (the default h1+h2 ALPN combiner already carries h2)"
                .into(),
        ));
    }

    #[cfg(feature = "pgwire")]
    if builder.pgwire_query.is_some() && (is_quic || is_udp) {
        return Err(ProximaError::Config(
            "Listener::builder(): .pgwire(query) is TCP-only (in-band SSLRequest upgrade \
             assumes a byte stream); combining with .quic()/.udp() has no meaning — use \
             .tcp() (the default)"
                .into(),
        ));
    }

    #[cfg(feature = "dns-listener")]
    if builder.dns_handler.is_some() && is_quic {
        return Err(ProximaError::Config(
            "Listener::builder(): .dns(handler) + .quic(): DNS-over-QUIC (DoQ) is \
             unimplemented; use .tcp() (DNS-over-TCP) or .udp() (classic DNS-over-UDP)"
                .into(),
        ));
    }

    #[cfg(all(
        feature = "websocket-upgrade",
        any(feature = "http1", feature = "http1-native")
    ))]
    if builder.websocket_handler.is_some() {
        if is_quic {
            return Err(ProximaError::Config(
                "Listener::builder(): .websocket(handler) + .quic(): RFC 9220 extended-CONNECT \
                 over h3 is unimplemented; use .tcp() (the default — `.websocket()` already \
                 implies it)"
                    .into(),
            ));
        }
        if builder.any_mode.is_some() {
            return Err(ProximaError::Config(
                "Listener::builder(): .websocket(handler) + .kafka()/.mqtt()/.amqp()/\
                 .memcached()/.redis()/.any()/.protocol(): both claim the same h1 \
                 connection — pick one"
                    .into(),
            ));
        }
    }

    Ok(())
}

/// Recover the bind address from the [`ListenerProtocolExt::http`]/`.https`
/// spec key (`bind.to_string()`, set by `Listener::http(bind)` or a caller's
/// own `.http(bind.to_string())`) when `.bind(addr)` was never called directly —
/// the "or needed an adapter" seam the client side doesn't need, since a
/// client's `.http(url)` value IS the dial target, while a listener's bind
/// address is otherwise a typed field the string spec key doesn't carry.
fn bind_from_spec(spec: &serde_json::Map<String, Value>) -> Option<SocketAddr> {
    spec.get("http")
        .and_then(Value::as_str)
        .and_then(|value| value.parse().ok())
}

/// The candidate-accumulation axes `any_listen_protocol` registers into
/// `app.any_registry()` before resolving `mode` — grouped into one struct
/// (rather than three more positional parameters) since all three are
/// exactly the `ListenerBuilder` fields that only ever exist to feed this
/// one function: `.deny()`/`.denies()` (`deny_signatures`), `.blacklist()`
/// (`blacklist_config`), and `.protocol()` (`extra_protocols`, see
/// [`ListenerBuilder::protocol`]).
#[cfg(any(feature = "http1", feature = "http1-native"))]
struct AnyAxisConfig<'a> {
    deny_signatures: &'a [(String, Vec<u8>)],
    blacklist_config: Option<proxima_listen::admission::BlacklistConfig>,
    extra_protocols: &'a [Arc<dyn proxima_listen::any::AnyProtocol>],
}

/// Resolve `.any()`/`.accepts()`/`.accept()` to a fresh
/// [`crate::listeners::AnyListenProtocol`] — the `.any()`-family sibling of
/// [`resolve_listen_protocol`]. Registers `axis.extra_protocols`
/// (`.protocol(impl AnyProtocol)` candidates, see [`ListenerBuilder::protocol`])
/// into `app.any_registry()` first, before `axis.deny_signatures` and before
/// `candidate_names` snapshots the registry — so an externally-defined
/// candidate is present under its own name in time to be selected by
/// `mode`, exactly like a first-party one `App::new()` pre-registered.
/// Builds the per-protocol handler map each candidate resolves through, in
/// override-precedence order:
///
/// 1. a per-listener `.any_handler(name, handler)` binding (`overrides`);
/// 2. the `App`-level default (`App::any_default_handlers`);
/// 3. the single `.handle(pipe)` this builder already required, erased —
///    works for any candidate whose expected handler type IS a
///    [`PipeHandle`](crate::pipe::PipeHandle) (h1/h2's shape); a
///    differently-shaped candidate must use step 1 or 2, since this
///    fallback would simply fail its own downcast otherwise (a config
///    error naming the protocol, never a panic).
///
/// Also builds the accept-edge DoS-blacklist wiring — UNCONDITIONALLY, even
/// when `axis.deny_signatures` is empty and `axis.blacklist_config` was
/// never set: registers each `DenySignature` into `app.any_registry()` (so
/// it is classified alongside every other candidate `mode` selects), threads
/// a `BlacklistTable` onto the resolved `AnyListenProtocol`, and COMPOSES the
/// reject-hook (`reject_hook`, if present, still runs — this never clobbers
/// it) so an unclassifiable reject also records a `Strike::Unclassifiable`.
///
/// Always self-registers a fresh `AnyListenProtocol` instance (never
/// reuses one across calls) since the merged handler map is per-call data
/// — the same reason `.pgwire(query)` builds fresh every time instead of
/// sharing `App::new()`'s static registration.
#[cfg(any(feature = "http1", feature = "http1-native"))]
fn any_listen_protocol(
    app: &App,
    mode: &AnyMode,
    overrides: &std::collections::BTreeMap<String, proxima_listen::any::AnyHandler>,
    reject_hook: Option<crate::listeners::RejectHook>,
    dispatch_fallback: PipeHandle,
    axis: AnyAxisConfig<'_>,
) -> Result<(String, Option<Arc<dyn ListenProtocol>>), ProximaError> {
    let registry = app.any_registry();

    // `.protocol(impl AnyProtocol)` candidates — registered before the
    // `deny_signatures` loop below so an external candidate's own name is
    // already in the registry by the time `candidate_names` snapshots it.
    for protocol in axis.extra_protocols {
        registry.register(protocol.clone())?;
    }

    let blacklist =
        proxima_listen::admission::BlacklistTable::new(axis.blacklist_config.unwrap_or_default());
    for (name, literal) in axis.deny_signatures {
        let candidate: Arc<dyn proxima_listen::any::AnyProtocol> =
            Arc::new(proxima_listen::any::DenySignature::new(
                name.clone(),
                literal.clone(),
                blacklist.clone(),
            ));
        registry.register(candidate)?;
    }

    let candidate_names: Vec<String> = match mode {
        AnyMode::All => registry.names(),
        AnyMode::Subset(names) => names.clone(),
    };
    let defaults = app.any_default_handlers();
    let mut merged: std::collections::BTreeMap<String, proxima_listen::any::AnyHandler> =
        std::collections::BTreeMap::new();
    for name in &candidate_names {
        let handler = overrides
            .get(name)
            .or_else(|| defaults.get(name))
            .cloned()
            .unwrap_or_else(|| proxima_listen::any::erase_handler(dispatch_fallback.clone()));
        merged.insert(name.clone(), handler);
    }
    let merged = Arc::new(merged);

    let mut protocol = match mode {
        AnyMode::All => crate::listeners::AnyListenProtocol::new(&registry, merged),
        AnyMode::Subset(names) => {
            crate::listeners::AnyListenProtocol::with_names(&registry, names, merged)?
        }
    };
    let composed_hook: crate::listeners::RejectHook = {
        let blacklist_for_hook = blacklist.clone();
        Arc::new(
            move |peer: Option<proxima_primitives::stream::PeerInfo>, reason| {
                let peer_ip = proxima_listen::peer_ip(peer.as_ref());
                blacklist_for_hook.record_strike(
                    peer_ip,
                    proxima_core::time::now(),
                    proxima_listen::admission::Strike::Unclassifiable,
                );
                if let Some(hook) = &reject_hook {
                    hook(peer, reason);
                }
            },
        )
    };
    protocol = protocol
        .with_reject_hook(composed_hook)
        .with_blacklist(blacklist);
    Ok(("any".to_string(), Some(Arc::new(protocol))))
}

/// Reject spec axes the listener side has no wiring for, instead of letting
/// them silently degrade to plaintext / connect-anyway. `ListenerBuilder`
/// does NOT implement `ClientTransportExt`/`ClientSecurityExt` (each
/// TYPE-SPECIFIC now, not the retired blanket `ProtocolSugar`/
/// `TransportSugar`), so a caller cannot actually reach the client's
/// `.proxy(url)` or bare `.tls()` methods on a `ListenerBuilder` value —
/// this guard exists for the ONE door still open regardless: the raw
/// `SpecBuilder::set`/`.spec(key, value)` escape hatch every `SpecBuilder`
/// still exposes (`builder.set("proxy", url)` compiles, since
/// `ListenerBuilder: SpecBuilder`). `.proxy(url)`'s spec key has no
/// listener-side wiring at all, so `.serve()` hard-errors rather than the
/// caller discovering an ignored-proxy listener at request time. The bare
/// `.tls()` marker check below catches the shape a hand-written `.spec("transport",
/// "tls")` call (or a `tls`-feature-off build, where the inherent
/// `.tls(TlsConfig)` override doesn't exist to shadow anything) would
/// otherwise leave silently unterminated.
fn reject_dead_axes(spec: &serde_json::Map<String, Value>) -> Result<(), ProximaError> {
    if spec.contains_key("proxy") {
        return Err(ProximaError::Config(
            "Listener::builder(): .proxy(url) is a client-side egress axis with no listener meaning; drop it".into(),
        ));
    }
    if spec.get("transport").and_then(Value::as_str) == Some("tls") && !tls_marker_present(spec) {
        return Err(ProximaError::Config(
            "Listener::builder(): bare .tls() only sets a marker key and terminates nothing; \
             call .tls(TlsConfig::self_signed() | TlsConfig::pem(..) | TlsConfig::files(..)?), \
             which requires the `tls` feature"
                .into(),
        ));
    }
    Ok(())
}

#[cfg(feature = "tls")]
fn tls_marker_present(spec: &serde_json::Map<String, Value>) -> bool {
    spec.contains_key(proxima_tls::SPEC_KEY)
}

#[cfg(not(feature = "tls"))]
fn tls_marker_present(_spec: &serde_json::Map<String, Value>) -> bool {
    false
}

/// The listen-side protocol pick — the mirror of `load.rs`'s
/// `value.get("http") ... else if value.get("grpc")` factory dispatch
/// (`src/load.rs:488,499`), extended with the `transport` axis so `.quic()`
/// resolves too (the client side doesn't need this extension: its own
/// `.quic()` dispatches through a DIFFERENT mechanism entirely — see
/// `load.rs`'s `canonical_h3` — not a shared ALPN-selected factory), and
/// with `.h2()`/`.pgwire(query)`/`.dns(handler)`, which have no (or a
/// differently-shaped) client-side twin. Returns the registry name to put in
/// `RunConfig::protocol`, and — when that name is one `App::new()` doesn't
/// register by default — the concrete protocol `ListenerBuilder::serve`
/// registers onto its fresh `App` first. `.pgwire(query)`/`.dns(handler)` are
/// checked FIRST (each carries a typed handle no other axis combination can
/// produce, so they always win) and never return an extra protocol here —
/// `.serve()` registers its own fresh instance directly, since it needs the
/// handle this spec-only function never sees. `.grpc()`/`.h2()` take
/// priority over `transport` next (mirrors `load.rs`'s `if http ... else if
/// grpc`, which never consults `transport` either): a listener that calls
/// both `.grpc()` and `.quic()` gets gRPC-over-h2, since no gRPC-over-h3
/// listen protocol exists.
fn resolve_listen_protocol(
    spec: &serde_json::Map<String, Value>,
) -> Result<(String, Option<Arc<dyn ListenProtocol>>), ProximaError> {
    if spec.contains_key("pgwire_axis") {
        return Ok(("pgwire".to_string(), None));
    }
    if spec.contains_key("dns_axis") {
        return Ok(("dns".to_string(), None));
    }
    if spec.contains_key("grpc") || spec.contains_key("h2") {
        return h2_listen_protocol();
    }
    if spec.get("transport").and_then(Value::as_str) == Some("quic") {
        return h3_native_listen_protocol();
    }
    // default / `.tcp()`: the ALPN h1+h2 combiner. `.tls()` composes as a
    // decorator OVER whatever this resolves — see `compose_tls` — so it
    // never changes what's resolved here. Already registered by
    // `App::new()` under the default feature set, so nothing extra to carry.
    Ok(("http".to_string(), None))
}

// `H2ListenProtocol` is retired onto `AnyListenProtocol`'s single bind +
// accept loop (real `ListenerCore`/`ConnAdmission` admission, graceful
// drain) — see `proxima_http::http2`'s module doc for why. `AnyListenProtocol`
// itself is gated on `http-listener` (which pulls `http1-native`) in
// proxima-http, since its H1 candidate is unconditional inside that module —
// so unlike the retired standalone listener, `.h2()`/`.grpc()` now ALSO needs
// `http1`/`http1-native` compiled in. This is a real, narrow regression
// (documented, not silent): the middle arm below is what a `http2`-only
// build (no http1 at all) now gets instead of a build failure.
#[cfg(all(feature = "http2", any(feature = "http1", feature = "http1-native")))]
fn h2_listen_protocol() -> Result<(String, Option<Arc<dyn ListenProtocol>>), ProximaError> {
    let protocol: Arc<dyn ListenProtocol> =
        Arc::new(crate::listeners::AnyListenProtocol::single_candidate(
            "h2",
            Arc::new(crate::listeners::H2PriorKnowledgeAnyProtocol::new()),
        ));
    Ok(("h2".to_string(), Some(protocol)))
}

#[cfg(all(
    feature = "http2",
    not(any(feature = "http1", feature = "http1-native"))
))]
fn h2_listen_protocol() -> Result<(String, Option<Arc<dyn ListenProtocol>>), ProximaError> {
    Err(ProximaError::Config(
        "Listener::builder(): .grpc()/.h2() needs `http1` or `http1-native` in addition to \
         `http2` since the AnyListenProtocol lift (H2ListenProtocol standalone is retired); \
         enable one"
            .into(),
    ))
}

#[cfg(not(feature = "http2"))]
fn h2_listen_protocol() -> Result<(String, Option<Arc<dyn ListenProtocol>>), ProximaError> {
    Err(ProximaError::Config(
        "Listener::builder(): .grpc() needs the `http2` feature (gRPC rides h2); none built".into(),
    ))
}

#[cfg(feature = "http3")]
fn h3_native_listen_protocol() -> Result<(String, Option<Arc<dyn ListenProtocol>>), ProximaError> {
    let protocol: Arc<dyn ListenProtocol> =
        Arc::new(crate::listeners::H3NativeListenProtocol::new());
    Ok(("h3-native".to_string(), Some(protocol)))
}

#[cfg(not(feature = "http3"))]
fn h3_native_listen_protocol() -> Result<(String, Option<Arc<dyn ListenProtocol>>), ProximaError> {
    Err(ProximaError::Config(
        "Listener::builder(): .quic() needs the `http3` feature; none built".into(),
    ))
}

/// Compose `.tls(config)` onto whatever `resolve_listen_protocol` already
/// resolved — the listener-side compositional seam: wraps the resolved
/// protocol in a `proxima_listen::TlsListenProtocol` decorator instead of
/// setting a spec key or a `ListenerSpec` field. Renames the registry key
/// (`"{name}+tls"`) so a TLS-wrapped `"http"` listener never collides with
/// the plain `HttpListenProtocol` `App::new()` already registered under
/// `"http"` in this SAME fresh `App` — irrelevant for `h2`/`h3-native`
/// (never pre-registered), but applied uniformly rather than special-cased.
#[cfg(feature = "tls")]
fn compose_tls(
    tls: Option<proxima_tls::TlsConfig>,
    name: String,
    protocol: Option<Arc<dyn ListenProtocol>>,
) -> Result<(String, Option<Arc<dyn ListenProtocol>>), ProximaError> {
    let Some(tls) = tls else {
        return Ok((name, protocol));
    };
    let inner = match protocol {
        Some(protocol) => protocol,
        None => http_listen_protocol_for_tls()?,
    };
    let registry_name = format!("{name}+tls");
    let wrapped: Arc<dyn ListenProtocol> =
        Arc::new(proxima_listen::TlsListenProtocol::new(inner, tls));
    // `ListenRegistry::register` derives its key from `protocol.name()`, and
    // `TlsListenProtocol::name()` deliberately delegates to the wrapped
    // protocol's name (so identity checks elsewhere, e.g.
    // `Listener::run_with_runtime`'s SO_REUSEPORT-spread pick, see straight
    // through the decorator) — which is exactly `"http"`, colliding with
    // the plain `HttpListenProtocol` `App::new()` already registered under
    // that name in this same fresh `App`. `NamedListenProtocol` overrides
    // ONLY the registry key for THIS registration; `.serve()` still runs
    // the real (TLS-terminating) decorator underneath.
    let named: Arc<dyn ListenProtocol> = Arc::new(NamedListenProtocol {
        name: registry_name.clone(),
        inner: wrapped,
    });
    Ok((registry_name, Some(named)))
}

/// Registry-key override for a composed protocol whose own `.name()` isn't
/// suitable as a registry key (see [`compose_tls`]) — delegates `serve`
/// unchanged, reports a caller-chosen name instead.
#[cfg(feature = "tls")]
struct NamedListenProtocol {
    name: String,
    inner: Arc<dyn ListenProtocol>,
}

#[cfg(feature = "tls")]
impl ListenProtocol for NamedListenProtocol {
    fn name(&self) -> &str {
        &self.name
    }

    fn serve(
        &self,
        bind: SocketAddr,
        dispatch: PipeHandle,
        spec: &Value,
        context: ServeContext,
        shutdown: oneshot::Receiver<()>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + '_>> {
        self.inner.serve(bind, dispatch, spec, context, shutdown)
    }
}

/// Build a fresh `HttpListenProtocol` to wrap in TLS — needed because the
/// non-TLS default path leaves `resolve_listen_protocol`'s `extra_protocol`
/// `None` (relying on `App::new()`'s pre-registration); `.tls()` needs a
/// concrete instance in hand to wrap regardless. `HttpListenProtocol` is
/// available under either `http1` (the legacy tokio-coupled build) or
/// `http1-native` (the tokio-free build) — see `listeners/mod.rs`'s
/// re-export gate — so this checks the same `any(...)` the rest of the
/// crate uses to reach it, not `http1` alone.
#[cfg(all(feature = "tls", any(feature = "http1", feature = "http1-native")))]
fn http_listen_protocol_for_tls() -> Result<Arc<dyn ListenProtocol>, ProximaError> {
    Ok(Arc::new(crate::listeners::HttpListenProtocol::new()))
}

#[cfg(all(feature = "tls", not(any(feature = "http1", feature = "http1-native"))))]
fn http_listen_protocol_for_tls() -> Result<Arc<dyn ListenProtocol>, ProximaError> {
    Err(ProximaError::Config(
        "Listener::builder(): .tls(config) on the default transport needs the `http1` or \
         `http1-native` feature; none built"
            .into(),
    ))
}

/// The base spec seam ([`proxima_config::sugar::SpecBuilder`]): identical
/// impl to `ClientBuilder`'s (`crate::client::handle`) — `set`/`push` are the
/// only two methods [`crate::listener::transport::ListenerTransportExt`] /
/// [`crate::listener::protocol::ListenerProtocolExt`] need underneath (each
/// its OWN type-specific trait, unlike the retired blanket `ProtocolSugar`/
/// `TransportSugar`). `set` reuses the existing `.spec()` so there is
/// exactly one write path (the same discipline `ClientBuilder::set` follows,
/// `src/client/handle.rs:644`).
impl proxima_config::sugar::SpecBuilder for ListenerBuilder {
    fn set(self, key: &str, value: impl Into<Value>) -> Self {
        self.spec(key, value.into())
    }

    fn push(mut self, key: &str, value: impl Into<Value>) -> Self {
        let entry = self
            .spec
            .entry(key.to_string())
            .or_insert_with(|| Value::Array(Vec::new()));
        if let Value::Array(array) = entry {
            array.push(value.into());
        }
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::listener::transport::ListenerTransportExt;
    use serde_json::json;

    // P4 config-as-mirror: the fluent builder and a literal config-shaped
    // `Value` produce the IDENTICAL spec map — same parity claim as
    // `ClientBuilder`'s `fluent.inner.spec == config.inner.spec` proof
    // (`src/client/handle.rs`'s `verbs_map_to_methods_and_builder_lowers_axes`).
    #[test]
    fn builder_spec_matches_listen_config_shape() {
        let fluent = ListenerBuilder::default()
            .tcp()
            .grpc()
            .spec("max_body_bytes", json!(1024));
        let config = json!({
            "transport": "tcp",
            "grpc": true,
            "max_body_bytes": 1024,
        });
        assert_eq!(Value::Object(fluent.spec), config);
    }

    #[test]
    fn resolve_listen_protocol_defaults_to_http_and_opts_into_grpc_via_the_same_key_load_reads() {
        let http_default = ListenerBuilder::default().tcp();
        let (name, extra) = resolve_listen_protocol(&http_default.spec).expect("tcp resolves");
        assert_eq!(name, "http");
        assert!(
            extra.is_none(),
            "\"http\" is already in App::new()'s default set"
        );

        let tls = ListenerBuilder::default().spec("transport", json!("tls"));
        let (name, extra) = resolve_listen_protocol(&tls.spec).expect("tls resolves");
        assert_eq!(
            name, "http",
            "TLS is spec data on the same combiner, not a new protocol"
        );
        assert!(extra.is_none());
    }

    #[cfg(all(feature = "http2", any(feature = "http1", feature = "http1-native")))]
    #[test]
    fn grpc_axis_resolves_to_h2_and_self_registers() {
        let grpc = ListenerBuilder::default().grpc();
        let (name, extra) = resolve_listen_protocol(&grpc.spec).expect("grpc resolves");
        assert_eq!(name, "h2");
        let carried = extra.expect(".grpc() must carry a protocol to self-register");
        assert_eq!(carried.name(), "h2");
    }

    #[cfg(feature = "http3")]
    #[test]
    fn quic_axis_resolves_to_h3_native_and_self_registers() {
        let quic = ListenerBuilder::default().quic();
        let (name, extra) = resolve_listen_protocol(&quic.spec).expect("quic resolves");
        assert_eq!(name, "h3-native");
        let carried = extra.expect(".quic() must carry a protocol to self-register");
        assert_eq!(carried.name(), "h3-native");
    }

    #[cfg(all(feature = "http2", any(feature = "http1", feature = "http1-native")))]
    #[test]
    fn h2_axis_resolves_to_the_same_shared_h2_protocol_as_grpc() {
        let h2 = ListenerBuilder::default().h2();
        let (name, extra) = resolve_listen_protocol(&h2.spec).expect("h2 resolves");
        assert_eq!(name, "h2");
        let carried = extra.expect(".h2() must carry a protocol to self-register");
        assert_eq!(carried.name(), "h2");
    }

    #[cfg(feature = "pgwire")]
    #[test]
    fn pgwire_axis_resolves_to_pgwire_and_carries_nothing_here() {
        // `.pgwire(query)` self-registers directly in `.serve()` (the query
        // handle never reaches `resolve_listen_protocol`, which only sees
        // the `pgwire_axis` marker key) — see `serve`'s own registration.
        let pgwire = ListenerBuilder::default().spec("pgwire_axis", json!(true));
        let (name, extra) = resolve_listen_protocol(&pgwire.spec).expect("pgwire resolves");
        assert_eq!(name, "pgwire");
        assert!(extra.is_none());
    }

    #[cfg(feature = "pgwire")]
    #[test]
    fn pgwire_axis_takes_priority_over_grpc_and_quic() {
        let mixed = ListenerBuilder::default()
            .spec("pgwire_axis", json!(true))
            .spec("grpc", json!(true))
            .spec("transport", json!("quic"));
        let (name, extra) = resolve_listen_protocol(&mixed.spec).expect("pgwire wins");
        assert_eq!(name, "pgwire");
        assert!(extra.is_none());
    }

    #[cfg(feature = "dns-listener")]
    #[test]
    fn dns_axis_resolves_to_dns_and_carries_nothing_here() {
        // `.dns(handler)` self-registers directly in `.serve()` (the handler
        // never reaches `resolve_listen_protocol`, which only sees the
        // `dns_axis` marker key) — see `serve`'s own registration.
        let dns = ListenerBuilder::default().spec("dns_axis", json!(true));
        let (name, extra) = resolve_listen_protocol(&dns.spec).expect("dns resolves");
        assert_eq!(name, "dns");
        assert!(extra.is_none());
    }

    #[test]
    fn listener_builder_mirrors_client_builder_axis_keys() {
        // The same axis method NAMES a `ClientBuilder` chain would make
        // (`.https(url).tls()`,
        // `src/client/handle.rs`'s `verbs_map_to_methods_and_builder_lowers_axes`)
        // lower to the IDENTICAL spec keys here — proof the listener rides the
        // shared sugar, not a forked DSL.
        let fluent = ListenerBuilder::default()
            .https("127.0.0.1:8080")
            .spec("transport", json!("tls"));
        assert_eq!(
            fluent.spec.get("http").and_then(Value::as_str),
            Some("127.0.0.1:8080")
        );
        assert_eq!(
            fluent.spec.get("transport").and_then(Value::as_str),
            Some("tls")
        );
    }

    #[test]
    fn listener_http_one_liner_carries_bind_through_the_spec_key() {
        let bind: SocketAddr = "127.0.0.1:8080".parse().expect("addr");
        let builder = Listener::http(bind);
        assert_eq!(builder.bind, Some(bind));
        assert_eq!(
            builder.spec.get("http").and_then(Value::as_str),
            Some(bind.to_string().as_str())
        );
    }

    #[test]
    fn bind_from_spec_recovers_the_address_when_bind_was_never_called_directly() {
        let bind: SocketAddr = "127.0.0.1:8080".parse().expect("addr");
        let builder = ListenerBuilder::default().http(bind.to_string());
        assert_eq!(builder.bind, None);
        assert_eq!(bind_from_spec(&builder.spec), Some(bind));
    }

    #[test]
    fn serve_without_bind_errors_before_touching_a_socket() {
        let outcome = futures::executor::block_on(ListenerBuilder::default().serve());
        let err = match outcome {
            Ok(_) => panic!("missing bind must error"),
            Err(err) => err,
        };
        assert!(format!("{err}").contains(".bind"), "got: {err}");
    }

    #[test]
    fn serve_without_handle_errors_before_touching_a_socket() {
        let bind: SocketAddr = "127.0.0.1:0".parse().expect("addr");
        let outcome = futures::executor::block_on(ListenerBuilder::default().bind(bind).serve());
        let err = match outcome {
            Ok(_) => panic!("missing handle must error"),
            Err(err) => err,
        };
        assert!(format!("{err}").contains(".handle"), "got: {err}");
    }

    #[test]
    fn proxy_axis_hard_errors_at_serve_instead_of_silently_ignoring_it() {
        // `.proxy(url)` lives ONLY on `ClientTransportExt` now (no listener
        // twin at all) — a caller reaching it via the raw `.spec()` escape
        // hatch (the only way a `proxy` key could land on a `ListenerBuilder`
        // today) must still hard-error at `.serve()`, not silently ignore it.
        let bind: SocketAddr = "127.0.0.1:0".parse().expect("addr");
        let builder = ListenerBuilder::default()
            .bind(bind)
            .spec("proxy", json!("http://127.0.0.1:9"));
        let err = match futures::executor::block_on(builder.serve()) {
            Ok(_) => panic!(".proxy(url) must not silently serve"),
            Err(err) => err,
        };
        assert!(format!("{err}").contains(".proxy"), "got: {err}");
    }

    #[cfg(feature = "tls")]
    #[test]
    fn tls_composes_a_decorator_instead_of_writing_a_spec_key() {
        let fluent = ListenerBuilder::default().tls(proxima_tls::TlsConfig::self_signed());
        assert!(
            fluent.tls.is_some(),
            ".tls(config) must accumulate on the builder, not the spec"
        );
        assert!(
            !fluent.spec.contains_key(proxima_tls::SPEC_KEY),
            "TLS must not be a spec key any more — it composes as a decorator at .serve() time"
        );
    }

    #[cfg(feature = "tls")]
    #[test]
    fn compose_tls_wraps_the_resolved_protocol_and_renames_the_registry_key() {
        let tls = proxima_tls::TlsConfig::self_signed();
        let (name, extra) = compose_tls(Some(tls), "http".to_string(), None).expect("compose_tls");
        assert_eq!(name, "http+tls");
        let wrapped = extra.expect("tls must carry a wrapped protocol");
        // `name` and `wrapped.name()` MUST agree — `App::register_listen_protocol`
        // derives the registry key from `protocol.name()`, and `.serve()`
        // separately routes lookups by `name`; a mismatch would either
        // collide with `App::new()`'s plain "http" entry (see
        // `TlsListenProtocol::name()`'s inner-delegating contract, proven
        // directly against `proxima_listen::TlsListenProtocol` in
        // `proxima-listen`'s own test suite) or make the two never resolve
        // to the same registration at all.
        assert_eq!(wrapped.name(), name);
    }

    #[cfg(feature = "tls")]
    #[test]
    fn compose_tls_is_a_noop_when_no_tls_was_requested() {
        let (name, extra) = compose_tls(None, "http".to_string(), None).expect("compose_tls");
        assert_eq!(name, "http");
        assert!(extra.is_none());
    }

    #[cfg(all(feature = "tls", feature = "pgwire"))]
    #[test]
    fn pgwire_and_tls_together_hard_error_instead_of_wrapping_the_wrong_protocol() {
        use proxima_primitives::pipe::SendPipe;

        struct NeverCalled;
        impl SendPipe for NeverCalled {
            type In = proxima_pgwire::PgRequest;
            type Out = proxima_pgwire::PgResponse;
            type Err = ProximaError;

            async fn call(&self, _request: Self::In) -> Result<Self::Out, ProximaError> {
                unreachable!("guard must reject before the query engine is ever dispatched to")
            }
        }

        let builder = ListenerBuilder::default()
            .tls(proxima_tls::TlsConfig::self_signed())
            .pgwire(proxima_pgwire::into_pg_handle(NeverCalled));
        let err = match futures::executor::block_on(builder.serve()) {
            Ok(_) => panic!(".pgwire(query) + .tls(config) must not silently compose"),
            Err(err) => err,
        };
        assert!(format!("{err}").contains(".pgwire"), "got: {err}");
        assert!(format!("{err}").contains(".tls"), "got: {err}");
    }

    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[test]
    fn any_mode_defaults_to_none() {
        assert_eq!(ListenerBuilder::default().any_mode, None);
    }

    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[test]
    fn dot_any_selects_every_registered_candidate() {
        let builder = ListenerBuilder::default().any();
        assert_eq!(builder.any_mode, Some(AnyMode::All));
    }

    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[test]
    fn dot_accepts_restricts_to_the_named_subset() {
        let builder = ListenerBuilder::default().accepts(&["h1", "h2"]);
        assert_eq!(
            builder.any_mode,
            Some(AnyMode::Subset(vec!["h1".to_string(), "h2".to_string()]))
        );
    }

    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[test]
    fn dot_accept_restricts_to_exactly_one_name() {
        let builder = ListenerBuilder::default().accept("h2");
        assert_eq!(
            builder.any_mode,
            Some(AnyMode::Subset(vec!["h2".to_string()]))
        );
    }

    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[test]
    fn dot_protocol_without_a_prior_any_call_implicitly_selects_all_like_deny_does() {
        let builder = ListenerBuilder::default().protocol(StubAnyProtocol::new("mini"));
        assert_eq!(
            builder.any_mode,
            Some(AnyMode::All),
            ".protocol() with no prior .any()/.accepts()/.accept() must select every \
             registered candidate, the same implicit-select .deny() uses — not narrow to \
             just the new one"
        );
        assert_eq!(builder.extra_protocols.len(), 1);
        assert_eq!(builder.extra_protocols[0].name(), "mini");
    }

    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[test]
    fn dot_protocol_after_accepts_subset_only_appends_its_own_name() {
        let builder = ListenerBuilder::default()
            .accepts(&["h1"])
            .protocol(StubAnyProtocol::new("mini"));
        assert_eq!(
            builder.any_mode,
            Some(AnyMode::Subset(vec!["h1".to_string(), "mini".to_string()]))
        );
    }

    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[test]
    fn dot_protocol_after_dot_any_keeps_all_mode() {
        let builder = ListenerBuilder::default()
            .any()
            .protocol(StubAnyProtocol::new("mini"));
        assert_eq!(builder.any_mode, Some(AnyMode::All));
    }

    // The functional counterpart of the three builder-state tests above:
    // `any_listen_protocol` actually REGISTERS an `extra_protocols` entry
    // into `app.any_registry()` and resolves it as a live candidate,
    // exercised directly against a real `App` (not through a socket) —
    // the same style as `any_listen_protocol_resolves_to_the_any_registry_name`.
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[proxima::test]
    async fn any_listen_protocol_registers_and_selects_an_extra_protocol() {
        let app = App::new().expect("App::new");
        let dispatch = crate::pipe::into_handle(EchoOk);
        let extra_protocols: Vec<Arc<dyn proxima_listen::any::AnyProtocol>> =
            vec![Arc::new(StubAnyProtocol::new("mini"))];
        let (name, extra) = any_listen_protocol(
            &app,
            &AnyMode::Subset(vec!["mini".to_string()]),
            &std::collections::BTreeMap::new(),
            None,
            dispatch,
            AnyAxisConfig {
                deny_signatures: &[],
                blacklist_config: None,
                extra_protocols: &extra_protocols,
            },
        )
        .expect("any_listen_protocol resolves a registered extra protocol");
        assert_eq!(name, "any");
        assert!(extra.is_some());
        assert!(
            app.any_registry().get("mini").is_ok(),
            "the extra protocol must land in the App's AnyRegistry, reachable the same way \
             a first-party candidate is"
        );
    }

    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[test]
    fn any_handler_without_a_prior_any_call_implicitly_selects_that_name() {
        let builder = ListenerBuilder::default().any_handler("h1", 7_u8);
        assert_eq!(
            builder.any_mode,
            Some(AnyMode::Subset(vec!["h1".to_string()]))
        );
        assert!(builder.any_handlers.contains_key("h1"));
    }

    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[test]
    fn any_handler_after_dot_any_keeps_all_mode_and_still_records_the_override() {
        let builder = ListenerBuilder::default().any().any_handler("h1", 7_u8);
        assert_eq!(builder.any_mode, Some(AnyMode::All));
        assert!(builder.any_handlers.contains_key("h1"));
    }

    // Task's builder-resolution requirement: `.any()`/`.accepts()`/`.accept()`
    // resolve correctly through `any_listen_protocol` — exercised directly
    // (not through a live socket) against a real `App`, whose `App::new()`
    // registers the real h1 (+ h2 when `http2` is compiled in) candidates.
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[proxima::test]
    async fn any_listen_protocol_resolves_to_the_any_registry_name() {
        let app = App::new().expect("App::new");
        let dispatch = crate::pipe::into_handle(EchoOk);
        let (name, extra) = any_listen_protocol(
            &app,
            &AnyMode::All,
            &std::collections::BTreeMap::new(),
            None,
            dispatch,
            AnyAxisConfig {
                deny_signatures: &[],
                blacklist_config: None,
                extra_protocols: &[],
            },
        )
        .expect("any_listen_protocol resolves");
        assert_eq!(name, "any");
        let protocol = extra.expect(".any() must carry a protocol to self-register");
        assert_eq!(protocol.name(), "any");
    }

    #[cfg(any(feature = "http1", feature = "http1-native"))]
    #[proxima::test]
    async fn any_listen_protocol_subset_rejects_an_unregistered_name() {
        let app = App::new().expect("App::new");
        let dispatch = crate::pipe::into_handle(EchoOk);
        let outcome = any_listen_protocol(
            &app,
            &AnyMode::Subset(vec!["not-a-real-protocol".to_string()]),
            &std::collections::BTreeMap::new(),
            None,
            dispatch,
            AnyAxisConfig {
                deny_signatures: &[],
                blacklist_config: None,
                extra_protocols: &[],
            },
        );
        assert!(
            outcome.is_err(),
            "an unregistered candidate name must error, not silently ignore"
        );
    }

    /// A minimal `AnyProtocol` stand-in for an externally-defined candidate
    /// — `.protocol(impl AnyProtocol)`'s tests only need a name and a
    /// never-matching probe, never a live drive.
    #[cfg(any(feature = "http1", feature = "http1-native"))]
    struct StubAnyProtocol {
        name: String,
    }

    #[cfg(any(feature = "http1", feature = "http1-native"))]
    impl StubAnyProtocol {
        fn new(name: impl Into<String>) -> Self {
            Self { name: name.into() }
        }
    }

    #[cfg(any(feature = "http1", feature = "http1-native"))]
    impl proxima_listen::any::AnyProtocol for StubAnyProtocol {
        fn name(&self) -> &str {
            &self.name
        }

        fn max_prefix_bytes(&self) -> usize {
            8
        }

        fn probe(&self, _prefix: &[u8]) -> proxima_listen::any::ProbeVerdict {
            proxima_listen::any::ProbeVerdict::No
        }

        fn drive<'a>(
            &'a self,
            _stream: Box<dyn proxima_primitives::stream::StreamConnection>,
            _handler: proxima_listen::any::AnyHandler,
            _spec: &'a Value,
            _peer: Option<proxima_primitives::stream::PeerInfo>,
            _admission: &'a proxima_listen::admission::ConnAdmission,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<(), ProximaError>> + Send + 'a>,
        > {
            Box::pin(async move { Ok(()) })
        }
    }

    #[cfg(any(feature = "http1", feature = "http1-native"))]
    struct EchoOk;

    #[cfg(any(feature = "http1", feature = "http1-native"))]
    impl proxima_primitives::pipe::SendPipe for EchoOk {
        type In = crate::request::Request<bytes::Bytes>;
        type Out = crate::request::Response<bytes::Bytes>;
        type Err = ProximaError;

        async fn call(&self, _request: Self::In) -> Result<Self::Out, ProximaError> {
            Ok(crate::request::Response::ok("ok"))
        }
    }
}
