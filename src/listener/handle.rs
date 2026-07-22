use std::net::SocketAddr;
use std::sync::Arc;
#[cfg(feature = "tls")]
use std::{future::Future, pin::Pin};

#[cfg(feature = "tls")]
use futures::channel::oneshot;
use serde_json::Value;

use proxima_config::sugar::ProtocolSugar;
use proxima_listen::ListenProtocol;
#[cfg(feature = "tls")]
use proxima_listen::ServeContext;
use proxima_listen::handle::Listener;

use crate::app::{App, MountTarget, RunConfig};
use crate::error::ProximaError;
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
/// `Listener` type living here — same idiom as
/// `proxima_config::sugar::{TransportSugar, ProtocolSugar}`: import the
/// trait to unlock the static methods, exactly like those traits unlock
/// `.tcp()`/`.http()`. Bring it into scope with
/// `use proxima::{Listener, ListenerBuilderEntry};`.
pub trait ListenerBuilderEntry {
    /// Fluent builder: `Listener::builder().bind(addr).tcp().handle(pipe).serve()`.
    #[must_use]
    fn builder() -> ListenerBuilder;

    /// One-liner mirroring [`Client::http(url)`](crate::Client::http): binds
    /// the `ListenerBuilder`'s typed `.bind(addr)` slot AND the shared
    /// [`ProtocolSugar::http`](proxima_config::sugar::ProtocolSugar::http)
    /// spec key (`bind.to_string()`) in one call, so
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
/// `impl SpecBuilder` below. That blanket impl is what gives this type the
/// SAME [`ProtocolSugar`](proxima_config::sugar::ProtocolSugar)
/// (`.http`/`.https`/`.grpc`) and
/// [`TransportSugar`](proxima_config::sugar::TransportSugar)
/// (`.auto`/`.tcp`/`.tls`/`.h3`/`.proxy`) axes `ClientBuilder` gets — no
/// listener-specific per-wire methods would fork the sugar rather than
/// mirror it, EXCEPT where the wire genuinely has no client-side twin to
/// mirror: `.h2()` (h2 has no url-carrying client axis the way `.tls()`/
/// `.h3()` do) and `.pgwire(query)` (a typed SQL engine, not a wire pick at
/// all). A few axes are honestly asymmetric and shadow or extend the
/// blanket method with an inherent one carrying more than a client ever
/// needs: `.tls(TlsConfig)` (real cert material), `.grpc()` (url-less — a
/// listener dispatches to a `.handle(pipe)` already on hand, it doesn't dial
/// out), `.h2()`, and `.pgwire(query)` (real query engine — see its own
/// doc). `.proxy(url)` remains reachable through the blanket import (no
/// negative impl exists to hide it) but `.serve()` hard-errors if it's
/// present — see `reject_dead_axes`. `.bind()`/`.handle()` are the
/// listener-specific axes, where the client instead has a url baked into
/// `.http(url)` plus `.auth()`.
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
    /// The typed command handler `.redis(handler)` carries — accumulated
    /// separately from `spec` for the same reason `pgwire_query` is: a
    /// `proxima_redis::RedisPipeHandle` doesn't fit a `serde_json::Value`
    /// spec key. See [`Self::redis`].
    #[cfg(feature = "redis-listener")]
    redis_handler: Option<proxima_redis::RedisPipeHandle>,
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
    // `.auto()`/`.tcp()`/`.h3()` (`TransportSugar`) and `.http(url)`/
    // `.https(url)` (`ProtocolSugar`) are real, unmodified blanket methods —
    // `resolve_listen_protocol` below reads the same `transport`/`grpc` spec
    // keys `load.rs` reads on the client side. `.tls()`/`.grpc()`/`.h2()`/
    // `.pgwire(query)` are NOT blanket sugar here: all are inherent (the
    // last two have no blanket twin at all) because a listener needs cert
    // material / a query engine / has no url
    // to dial. `use proxima::{ProtocolSugar, TransportSugar}` brings the rest
    // into scope.

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

    /// Select gRPC as the listen protocol — the url-less counterpart of
    /// [`ProtocolSugar::grpc(url)`](proxima_config::sugar::ProtocolSugar::grpc):
    /// a listener dispatches to a `.handle(pipe)` already on hand, it doesn't
    /// dial an upstream url, so this inherent 0-arg method shadows the
    /// blanket 1-arg trait method and just flips the same `"grpc"` marker key
    /// [`resolve_listen_protocol`] reads (mirroring `load.rs`'s
    /// `value.get("grpc")` dispatch, `src/load.rs:499`). Resolves to the
    /// `h2` listen protocol — gRPC rides h2.
    #[must_use]
    pub fn grpc(mut self) -> Self {
        self.spec.insert("grpc".to_string(), Value::Bool(true));
        self
    }

    /// Select h2 (h2c, prior-knowledge, no ALPN, no TLS) as the listen
    /// protocol — the url-less counterpart of a client's transport pick: a
    /// listener dispatches to a `.handle(pipe)` already on hand rather than
    /// dialing an upstream url, so this is an inherent 0-arg method (there
    /// is no blanket `TransportSugar::h2()` to shadow — h2 has no
    /// client-side url-carrying twin the way `.tls()`/`.h3()` do). Resolves
    /// to the exact same `h2` listen protocol `.grpc()` resolves to (see its
    /// doc) — `.grpc()` and `.h2()` are the same wire, self-registered onto
    /// the fresh `App` at `.serve()` time.
    #[must_use]
    pub fn h2(mut self) -> Self {
        self.spec.insert("h2".to_string(), Value::Bool(true));
        self
    }

    /// Select PostgreSQL wire protocol as the listen protocol, carrying the
    /// SQL engine directly — the one axis that genuinely needs more than a
    /// marker key: `query` is the same typed
    /// [`PgPipeHandle`](proxima_pgwire::PgPipeHandle)
    /// [`PgWireListenProtocol::new`](proxima_pgwire::PgWireListenProtocol::new)
    /// takes, matching a SQL verb `Request`/`Response` pair no generic
    /// `.handle(pipe)` (`Request<Bytes>`/`Response<Bytes>`) can carry. This
    /// is the same asymmetry `.tls(TlsConfig)` has against the client's bare
    /// `.tls()`: a listener needs real material a client-side axis never
    /// does. Unlike `.h2()`/`.h3()` (which resolve to one shared instance
    /// self-registered onto the fresh `App`), `.serve()` constructs a fresh
    /// `PgWireListenProtocol` carrying THIS exact `query` every call —
    /// `App::new()` cannot pre-register a protocol it doesn't yet have a
    /// query engine for. `.handle(pipe)` is still required before `.serve()`
    /// (the one validation path stays uniform across axes) even though
    /// `PgWireListenProtocol::serve` never calls it once a
    /// constructor-supplied `query` is present.
    #[cfg(feature = "pgwire")]
    #[must_use]
    pub fn pgwire(mut self, query: proxima_pgwire::PgPipeHandle) -> Self {
        self.pgwire_query = Some(query);
        self.spec.insert("pgwire_axis".to_string(), Value::Bool(true));
        self
    }

    /// Select the Redis/Valkey wire protocol as the listen protocol,
    /// carrying the command handler directly — the same asymmetry
    /// `.pgwire(query)` has: `handler` is the typed
    /// [`RedisPipeHandle`](proxima_redis::RedisPipeHandle)
    /// (`Request<RedisRequest>`/`Response<RespValue>`), which no generic
    /// `.handle(pipe)` can carry. Resolves through a single-candidate
    /// `AnyListenProtocol` wrapping [`proxima_redis::RedisAnyProtocol`] —
    /// there is no standalone `RedisListenProtocol`; redis's listen-side
    /// surface has always been an `AnyProtocol` candidate. `.tls(config)`
    /// composes normally as the generic decorator over this (RESP-over-TLS
    /// is whole-connection TLS from byte 0, unlike pgwire's in-band
    /// SSLRequest — no conflict to guard against).
    #[cfg(feature = "redis-listener")]
    #[must_use]
    pub fn redis(mut self, handler: proxima_redis::RedisPipeHandle) -> Self {
        self.redis_handler = Some(handler);
        self.spec.insert("redis_axis".to_string(), Value::Bool(true));
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

    /// Terminate TLS at this listener — the listener-inherent counterpart of
    /// the client's bare, url-less
    /// [`TransportSugar::tls()`](proxima_config::sugar::TransportSugar::tls)
    /// (which only picks a wire scheme for a url the client already
    /// carries). A listener additionally needs cert material, so this takes
    /// the exact [`proxima_tls::TlsConfig`] type. Composes at `.serve()` time
    /// as a `proxima_listen::TlsListenProtocol` DECORATOR wrapping whatever
    /// protocol `resolve_listen_protocol` resolves — TLS is NOT a spec key
    /// and NOT a field on `ListenerSpec`; on/off is the presence of that
    /// wrapper. No new TLS mechanism underneath: the decorator still stamps
    /// the identical `proxima_tls::SPEC_KEY` (`"__proxima_tls"`) marker
    /// `HttpListenProtocol::serve_default` reads
    /// (`proxima-http/src/listener/mod.rs:451`) to build its
    /// `tokio_rustls::TlsAcceptor` — only WHERE that marker gets attached
    /// moved, from this builder's spec map to the decorator's own `serve`.
    /// This inherent `(Self, TlsConfig)` signature shadows the blanket 0-arg
    /// `TransportSugar::tls()`, so a bare `.tls()` call is a compile error
    /// here, not a silent plaintext no-op.
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
        #[cfg(feature = "pgwire")]
        let pgwire_query = self.pgwire_query;
        #[cfg(feature = "redis-listener")]
        let redis_handler = self.redis_handler;
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
        // resolves `protocol` ("pgwire") against the registry.
        #[cfg(feature = "pgwire")]
        if let Some(query) = pgwire_query {
            app.register_listen_protocol(Arc::new(proxima_pgwire::PgWireListenProtocol::new(
                "pgwire", query,
            )))?;
        }
        // `.redis(handler)` carries a typed command handler the same way
        // `.pgwire(query)` carries its query engine — register a fresh
        // single-candidate `AnyListenProtocol` wrapping `RedisAnyProtocol`
        // now, before `.serve()` resolves `protocol` ("redis") against the
        // registry.
        #[cfg(feature = "redis-listener")]
        if let Some(handler) = redis_handler {
            app.register_listen_protocol(Arc::new(
                crate::listeners::AnyListenProtocol::single_candidate(
                    "redis",
                    Arc::new(proxima_redis::RedisAnyProtocol::new("redis", handler)),
                ),
            ))?;
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

/// Recover the bind address from the `ProtocolSugar::http`/`.https` spec key
/// (`bind.to_string()`, set by `Listener::http(bind)` or a caller's own
/// `.http(bind.to_string())`) when `.bind(addr)` was never called directly —
/// the "or needed an adapter" seam the client side doesn't need, since a
/// client's `.http(url)` value IS the dial target, while a listener's bind
/// address is otherwise a typed field the string spec key doesn't carry.
fn bind_from_spec(spec: &serde_json::Map<String, Value>) -> Option<SocketAddr> {
    spec.get("http")
        .and_then(Value::as_str)
        .and_then(|value| value.parse().ok())
}

/// Resolve `.any()`/`.accepts()`/`.accept()` to a fresh
/// [`crate::listeners::AnyListenProtocol`] — the `.any()`-family sibling of
/// [`resolve_listen_protocol`]. Builds the per-protocol handler map each
/// candidate resolves through, in override-precedence order:
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
) -> Result<(String, Option<Arc<dyn ListenProtocol>>), ProximaError> {
    let registry = app.any_registry();
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
    if let Some(hook) = reject_hook {
        protocol = protocol.with_reject_hook(hook);
    }
    Ok(("any".to_string(), Some(Arc::new(protocol))))
}

/// Reject spec axes the listener side has no wiring for, instead of letting
/// them silently degrade to plaintext / connect-anyway. `.proxy(url)` is
/// `TransportSugar`'s blanket method (unavoidably in scope on every
/// `SpecBuilder`, `ListenerBuilder` included — there is no negative impl to
/// remove it); it has no listener-side implementation, so `.serve()`
/// hard-errors rather than the caller discovering an ignored-proxy listener
/// at request time. A bare `.tls()` (blanket 0-arg) only reaches this check
/// when the `tls` feature is off and the inherent `.tls(TlsConfig)` override
/// above doesn't exist to shadow it.
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
/// (`src/load.rs:488,499`), extended with the `transport` axis so `.h3()`
/// resolves too (the client side doesn't need this extension: its `h3`
/// transport rides the SAME `http`-keyed factory, selected by ALPN, not a
/// different registry entry), and with `.h2()`/`.pgwire(query)`, which have
/// no client-side twin at all. Returns the registry name to put in
/// `RunConfig::protocol`, and — when that name is one `App::new()` doesn't
/// register by default — the concrete protocol `ListenerBuilder::serve`
/// registers onto its fresh `App` first. `.pgwire(query)` is checked FIRST
/// (it carries a typed query engine no other axis combination can produce,
/// so it always wins) and never returns an extra protocol here — `.serve()`
/// registers its own fresh `PgWireListenProtocol` directly, since it needs
/// the `query` handle this spec-only function never sees. `.grpc()`/`.h2()`
/// take priority over `transport` next (mirrors `load.rs`'s `if http ...
/// else if grpc`, which never consults `transport` either): a listener that
/// calls both `.grpc()` and `.h3()` gets gRPC-over-h2, since no
/// gRPC-over-h3 listen protocol exists.
fn resolve_listen_protocol(
    spec: &serde_json::Map<String, Value>,
) -> Result<(String, Option<Arc<dyn ListenProtocol>>), ProximaError> {
    if spec.contains_key("pgwire_axis") {
        return Ok(("pgwire".to_string(), None));
    }
    if spec.contains_key("redis_axis") {
        return Ok(("redis".to_string(), None));
    }
    if spec.contains_key("grpc") || spec.contains_key("h2") {
        return h2_listen_protocol();
    }
    if spec.get("transport").and_then(Value::as_str) == Some("h3") {
        return h3_native_listen_protocol();
    }
    // default / `.tcp()` / `.auto()`: the ALPN h1+h2 combiner. `.tls()`
    // composes as a decorator OVER whatever this resolves — see
    // `compose_tls` — so it never changes what's resolved here. Already
    // registered by `App::new()` under the default feature set, so nothing
    // extra to carry.
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
    let protocol: Arc<dyn ListenProtocol> = Arc::new(
        crate::listeners::AnyListenProtocol::single_candidate(
            "h2",
            Arc::new(crate::listeners::H2PriorKnowledgeAnyProtocol::new()),
        ),
    );
    Ok(("h2".to_string(), Some(protocol)))
}

#[cfg(all(feature = "http2", not(any(feature = "http1", feature = "http1-native"))))]
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
        "Listener::builder(): .grpc() needs the `http2` feature (gRPC rides h2); none built"
            .into(),
    ))
}

#[cfg(feature = "http3")]
fn h3_native_listen_protocol() -> Result<(String, Option<Arc<dyn ListenProtocol>>), ProximaError> {
    let protocol: Arc<dyn ListenProtocol> =
        Arc::new(crate::listeners::H3NativeListenProtocol::new());
    Ok(("h3-native".to_string(), Some(protocol)))
}

#[cfg(not(feature = "http3"))]
fn h3_native_listen_protocol() -> Result<(String, Option<Arc<dyn ListenProtocol>>), ProximaError>
{
    Err(ProximaError::Config(
        "Listener::builder(): .h3() needs the `http3` feature; none built".into(),
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
    let wrapped: Arc<dyn ListenProtocol> = Arc::new(proxima_listen::TlsListenProtocol::new(inner, tls));
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

#[cfg(all(
    feature = "tls",
    not(any(feature = "http1", feature = "http1-native"))
))]
fn http_listen_protocol_for_tls() -> Result<Arc<dyn ListenProtocol>, ProximaError> {
    Err(ProximaError::Config(
        "Listener::builder(): .tls(config) on the default transport needs the `http1` or \
         `http1-native` feature; none built"
            .into(),
    ))
}

/// The base spec seam ([`proxima_config::sugar::SpecBuilder`]): identical
/// impl to `ClientBuilder`'s (`crate::client::handle`) — `set`/`push` are the
/// only two methods the axis sugar needs, so a `use` of `ProtocolSugar` /
/// `TransportSugar` lights up the same methods here as on the client. `set`
/// reuses the existing `.spec()` so there is exactly one write path (the same
/// discipline `ClientBuilder::set` follows, `src/client/handle.rs:644`).
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
    use proxima_config::sugar::TransportSugar;
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
        let (name, extra) =
            resolve_listen_protocol(&http_default.spec).expect("tcp resolves");
        assert_eq!(name, "http");
        assert!(extra.is_none(), "\"http\" is already in App::new()'s default set");

        let tls = ListenerBuilder::default().spec("transport", json!("tls"));
        let (name, extra) = resolve_listen_protocol(&tls.spec).expect("tls resolves");
        assert_eq!(name, "http", "TLS is spec data on the same combiner, not a new protocol");
        assert!(extra.is_none());
    }

    #[cfg(feature = "http2")]
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
    fn h3_axis_resolves_to_h3_native_and_self_registers() {
        let h3 = ListenerBuilder::default().h3();
        let (name, extra) = resolve_listen_protocol(&h3.spec).expect("h3 resolves");
        assert_eq!(name, "h3-native");
        let carried = extra.expect(".h3() must carry a protocol to self-register");
        assert_eq!(carried.name(), "h3-native");
    }

    #[cfg(feature = "http2")]
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
    fn pgwire_axis_takes_priority_over_grpc_and_h3() {
        let mixed = ListenerBuilder::default()
            .spec("pgwire_axis", json!(true))
            .spec("grpc", json!(true))
            .spec("transport", json!("h3"));
        let (name, extra) = resolve_listen_protocol(&mixed.spec).expect("pgwire wins");
        assert_eq!(name, "pgwire");
        assert!(extra.is_none());
    }

    #[test]
    fn listener_builder_mirrors_client_builder_axis_keys() {
        // The same `ProtocolSugar`/`TransportSugar` method calls a
        // `ClientBuilder` chain would make (`.https(url).tls()`,
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
        let bind: SocketAddr = "127.0.0.1:0".parse().expect("addr");
        let builder = ListenerBuilder::default()
            .bind(bind)
            .proxy("http://127.0.0.1:9");
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
        )
        .expect("any_listen_protocol resolves");
        assert_eq!(name, "any");
        let protocol = extra.expect(".any() must carry a protocol to self-register");
        assert_eq!(protocol.name(), "any");
    }

    #[cfg(feature = "http2")]
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
        );
        assert!(
            outcome.is_err(),
            "an unregistered candidate name must error, not silently ignore"
        );
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
