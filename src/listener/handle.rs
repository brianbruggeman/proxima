use std::net::SocketAddr;

use serde_json::Value;

use proxima_listen::handle::Listener;

use crate::app::{App, MountTarget, RunConfig};
use crate::error::ProximaError;
use crate::pipe::PipeHandle;
use crate::server::Server;

/// Gives the real listen-side primitive — [`Listener`]
/// (`proxima-listen/src/handle.rs`, produced by `ListenerSpec::attach(dispatch)`,
/// run via `Listener::run_with_runtime`) — the `Listener::builder()` entry
/// point mirroring [`Client::builder()`](crate::Client::builder). `Listener`
/// is defined in `proxima-listen`, a crate this one depends on but that
/// cannot depend back on [`App`] / [`Server`]; Rust's orphan rule forbids an
/// inherent `impl Listener { fn builder() }` from this crate, so the entry
/// point is a local trait blanket-impl'd for the foreign type instead of a
/// second `Listener` type living here — same idiom as
/// `proxima_config::sugar::{TransportSugar, ProtocolSugar}`: import the
/// trait to unlock the static method, exactly like those traits unlock
/// `.tcp()`/`.http()`. Bring it into scope with
/// `use proxima::{Listener, ListenerBuilderEntry};`.
pub trait ListenerBuilderEntry {
    /// Fluent builder: `Listener::builder().bind(addr).tcp().handle(pipe).serve()`.
    #[must_use]
    fn builder() -> ListenerBuilder;
}

impl ListenerBuilderEntry for Listener {
    fn builder() -> ListenerBuilder {
        ListenerBuilder::default()
    }
}

/// Fluent builder for [`Listener`] — accumulates a spec `serde_json::Map` the
/// exact same way `ClientBuilder` does (see `crate::client::handle`), via
/// `impl SpecBuilder` below. Only the axes that actually terminate something
/// on the listen side are exposed: `.tcp()`
/// ([`TransportSugar`](crate::TransportSugar), plaintext, the honest default)
/// and the inherent `.tls(TlsConfig)` / `.grpc()` defined on this type (see
/// their docs — both shadow a same-named blanket trait method that doesn't
/// carry enough for a listener to act on). `.h3()` / `.proxy(url)` remain
/// reachable through the blanket `TransportSugar` import (no negative impl
/// exists to hide them) but `.serve()` hard-errors if either is present —
/// see `reject_dead_axes`. `.bind()`/`.handle()` are the listener-specific
/// axes, where the client instead has a url baked into `.http(url)` plus
/// `.auth()`.
#[derive(Default)]
pub struct ListenerBuilder {
    spec: serde_json::Map<String, Value>,
    bind: Option<SocketAddr>,
    dispatch: Option<PipeHandle>,
}

impl ListenerBuilder {
    // `.auto()`/`.tcp()` (`TransportSugar`) are real and stay — plaintext is
    // the honest default. `.tls()`/`.h3()`/`.proxy()` are NOT blanket sugar
    // here: `.tls` is redefined below (inherent, shadowing the blanket 0-arg
    // version) because a listener needs cert material the client-side axis
    // never carries; `.h3`/`.proxy` have no listener wiring at all today, so
    // `.serve()` hard-errors rather than silently accepting them (see
    // `reject_dead_axes`). `use proxima::TransportSugar` still brings
    // `.tcp()`/`.auto()` into scope.

    /// The socket address to listen on — required before `.serve()`.
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
    /// [`protocol_name`] reads (mirroring `load.rs`'s `value.get("grpc")`
    /// dispatch, `src/load.rs:455`). `.http()` needs no method at all —
    /// `protocol_name` already defaults to `"http"`, the only listen
    /// protocol `App::new()` registers under the default feature set.
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
    /// the exact [`proxima_tls::TlsConfig`] type
    /// `proxima_listen::handle::ListenerSpec::with_tls` takes, and lowers
    /// through the identical primitive: `proxima_tls::config_to_spec_value`
    /// keyed by `proxima_tls::SPEC_KEY` (`"__proxima_tls"`) — the same key
    /// `HttpListenProtocol::serve_default` reads
    /// (`proxima-http/src/listener/mod.rs:195`) to build its
    /// `tokio_rustls::TlsAcceptor`. No new TLS mechanism. This inherent
    /// `(Self, TlsConfig)` signature shadows the blanket 0-arg
    /// `TransportSugar::tls()`, so a bare `.tls()` call is a compile error
    /// here, not a silent plaintext no-op.
    #[cfg(feature = "tls")]
    #[must_use]
    pub fn tls(mut self, tls: proxima_tls::TlsConfig) -> Self {
        self.spec.insert(
            proxima_tls::SPEC_KEY.to_string(),
            proxima_tls::config_to_spec_value(&tls),
        );
        self
    }

    /// Merge an arbitrary spec key — the same escape hatch as
    /// `ClientBuilder::spec`.
    #[must_use]
    pub fn spec(mut self, key: impl Into<String>, value: Value) -> Self {
        self.spec.insert(key.into(), value);
        self
    }

    /// Terminal: resolve the accumulated spec to a `ListenProtocol` (via the
    /// `ListenRegistry` an `App` carries — the server-side mirror of the
    /// client's `load(Spec)`), bind, and return the running `Server`.
    /// Composes `App::new` + `App::mount` + `App::serve` — the exact
    /// `into_handle(pipe) -> App::new()? -> app.mount(...)? ->
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
        let bind = self.bind.ok_or_else(|| {
            ProximaError::Config(
                "Listener::builder(): .bind(addr) is required before .serve()".into(),
            )
        })?;
        let dispatch = self.dispatch.ok_or_else(|| {
            ProximaError::Config(
                "Listener::builder(): .handle(pipe) is required before .serve()".into(),
            )
        })?;
        let protocol = protocol_name(&self.spec).to_string();
        let app = App::new()?;
        app.mount("/{*path}", MountTarget::Handle(dispatch))?;
        let config = RunConfig {
            bind,
            protocol,
            spec: Value::Object(self.spec),
        };
        app.serve(config).await
    }
}

/// Reject spec axes the listener side has no wiring for, instead of letting
/// them silently degrade to plaintext / connect-anyway. `.h3()` and
/// `.proxy(url)` are `TransportSugar`'s blanket methods (unavoidably in scope
/// on every `SpecBuilder`, `ListenerBuilder` included — there is no negative
/// impl to remove them); neither has a listener-side implementation, so
/// `.serve()` hard-errors rather than the caller discovering a plaintext /
/// ignored-proxy listener at request time. A bare `.tls()` (blanket 0-arg)
/// only reaches this check when the `tls` feature is off and the inherent
/// `.tls(TlsConfig)` override above doesn't exist to shadow it.
fn reject_dead_axes(spec: &serde_json::Map<String, Value>) -> Result<(), ProximaError> {
    if spec.get("transport").and_then(Value::as_str) == Some("h3") {
        return Err(ProximaError::Config(
            "Listener::builder(): .h3() has no listener implementation yet; drop it".into(),
        ));
    }
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
/// (`src/load.rs:455`), but for a listener "http" is the default (the only
/// listen protocol registered by `App::new()` under the default feature
/// set): only `.grpc(..)` needs to opt out of it. Picking `"grpc"` today
/// surfaces as a clean registry-lookup error at `.serve()` (no "grpc" listen
/// protocol is registered yet) rather than silently falling back — honest,
/// not fabricated support.
fn protocol_name(spec: &serde_json::Map<String, Value>) -> &'static str {
    if spec.contains_key("grpc") {
        "grpc"
    } else {
        "http"
    }
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
    fn protocol_name_defaults_to_http_and_opts_into_grpc_via_the_same_key_load_reads() {
        let http_default = ListenerBuilder::default().tcp();
        assert_eq!(protocol_name(&http_default.spec), "http");

        let grpc = ListenerBuilder::default().grpc();
        assert_eq!(protocol_name(&grpc.spec), "grpc");
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
    fn h3_axis_hard_errors_at_serve_instead_of_silently_binding_tcp() {
        let bind: SocketAddr = "127.0.0.1:0".parse().expect("addr");
        let builder = ListenerBuilder::default().bind(bind).h3();
        let err = match futures::executor::block_on(builder.serve()) {
            Ok(_) => panic!(".h3() must not silently serve"),
            Err(err) => err,
        };
        assert!(format!("{err}").contains(".h3()"), "got: {err}");
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
    fn tls_writes_the_same_spec_key_the_http_listener_reads() {
        let fluent = ListenerBuilder::default().tls(proxima_tls::TlsConfig::self_signed());
        assert!(
            fluent.spec.contains_key(proxima_tls::SPEC_KEY),
            "expected {} in {:?}",
            proxima_tls::SPEC_KEY,
            fluent.spec
        );
    }
}
