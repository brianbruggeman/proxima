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
/// listener-specific per-wire methods (`.h1()`/`.h2()`/`.h3_native()`
/// would be a fork of the sugar, not a mirror of it). Two axes are honestly
/// asymmetric and shadow the blanket method with an inherent one carrying
/// more than a client ever needs: `.tls(TlsConfig)` (real cert material) and
/// `.grpc()` (url-less — a listener dispatches to a `.handle(pipe)` already
/// on hand, it doesn't dial out). `.proxy(url)` remains reachable through the
/// blanket import (no negative impl exists to hide it) but `.serve()`
/// hard-errors if it's present — see `reject_dead_axes`. `.bind()`/`.handle()`
/// are the listener-specific axes, where the client instead has a url baked
/// into `.http(url)` plus `.auth()`.
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
}

impl ListenerBuilder {
    // `.auto()`/`.tcp()`/`.h3()` (`TransportSugar`) and `.http(url)`/
    // `.https(url)` (`ProtocolSugar`) are real, unmodified blanket methods —
    // `resolve_listen_protocol` below reads the same `transport`/`grpc` spec
    // keys `load.rs` reads on the client side. `.tls()`/`.grpc()` are NOT
    // blanket sugar here: both are redefined below (inherent, shadowing the
    // blanket version) because a listener needs cert material / has no url
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
    /// `value.get("grpc")` dispatch, `src/load.rs:455`). Resolves to the
    /// `h2` listen protocol — gRPC rides h2.
    #[must_use]
    pub fn grpc(mut self) -> Self {
        self.spec.insert("grpc".to_string(), Value::Bool(true));
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
    /// (`proxima-http/src/listener/mod.rs:195`) to build its
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
    /// also uses at `src/app.rs:925,981`) so every path routes to
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
        let (protocol, extra_protocol) = resolve_listen_protocol(&self.spec)?;
        #[cfg(feature = "tls")]
        let (protocol, extra_protocol) = compose_tls(self.tls, protocol, extra_protocol)?;
        let app = App::new()?;
        if let Some(protocol) = extra_protocol {
            app.register_listen_protocol(protocol)?;
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
/// (`src/load.rs:455`), extended with the `transport` axis so `.h3()`
/// resolves too (the client side doesn't need this extension: its `h3`
/// transport rides the SAME `http`-keyed factory, selected by ALPN, not a
/// different registry entry). Returns the registry name to put in
/// `RunConfig::protocol`, and — when that name is one `App::new()` doesn't
/// register by default — the concrete protocol `ListenerBuilder::serve`
/// registers onto its fresh `App` first. `.grpc()` takes priority over
/// `transport` (mirrors `load.rs`'s `if http ... else if grpc`, which never
/// consults `transport` either): a listener that calls both `.grpc()` and
/// `.h3()` gets gRPC-over-h2, since no gRPC-over-h3 listen protocol exists.
fn resolve_listen_protocol(
    spec: &serde_json::Map<String, Value>,
) -> Result<(String, Option<Arc<dyn ListenProtocol>>), ProximaError> {
    if spec.contains_key("grpc") {
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

#[cfg(feature = "http2")]
fn h2_listen_protocol() -> Result<(String, Option<Arc<dyn ListenProtocol>>), ProximaError> {
    let protocol: Arc<dyn ListenProtocol> = Arc::new(crate::listeners::H2ListenProtocol::new());
    Ok(("h2".to_string(), Some(protocol)))
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
/// concrete instance in hand to wrap regardless.
#[cfg(all(feature = "tls", feature = "http1"))]
fn http_listen_protocol_for_tls() -> Result<Arc<dyn ListenProtocol>, ProximaError> {
    Ok(Arc::new(crate::listeners::HttpListenProtocol::new()))
}

#[cfg(all(feature = "tls", not(feature = "http1")))]
fn http_listen_protocol_for_tls() -> Result<Arc<dyn ListenProtocol>, ProximaError> {
    Err(ProximaError::Config(
        "Listener::builder(): .tls(config) on the default transport needs the `http1` feature; none built"
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
}
