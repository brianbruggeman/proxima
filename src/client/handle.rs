use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use serde_json::Value;
// tokio-free async-init cell (async_lock-backed) — this is the one async
// OnceCell site in the crate, and the prime hot path must not touch a tokio
// primitive. unconditional, not gated behind `sync-wrappers`.
use proxima_primitives::sync::OnceCell;

use proxima_primitives::pipe::SendPipe;

use crate::client::request::RequestBuilder;
use crate::error::ProximaError;
use crate::load::{LoadContext, Spec, load};
use crate::pipe::PipeHandle;
use crate::pipe_factory::PipeFactory;
#[cfg(any(
    feature = "tokio-runtime",
    all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    )
))]
use crate::runtime::Runtime;

/// Spec-driven client. The upstream is configured via the spec
/// (`from_value`) or the fluent builder (`Client::builder().http(..)`); the
/// execution runtime is optional — inject one with `.runtime(..)`, or the
/// client self-contains a shared prime runtime so `send()` works from any
/// thread whether or not the caller is already on a prime worker.
#[derive(Clone)]
pub struct Client {
    inner: Arc<Inner>,
}

struct Inner {
    spec: Value,
    handle: OnceCell<PipeHandle>,
    /// a pre-composed handle injected via [`Client::from_handle`]; when set,
    /// dispatch goes straight to it and the spec is never loaded (the in-process
    /// `from_pipe` test/embedding seam).
    injected: Option<PipeHandle>,
    /// factories registered on top of the default set, so a protocol defined
    /// OUTSIDE this crate (kafka, redis, a private wire) is reachable through
    /// the same `Client` — the extensibility half of "ergonomic AND extensible".
    extra_factories: Vec<Arc<dyn PipeFactory>>,
    /// runtime to dispatch `send()` onto when the caller is NOT already on a
    /// thread that can host the transport's reactor. prime fallback is a
    /// process-shared prime runtime; the tokio path requires this injected
    /// runtime (e.g. `TokioPerCoreRuntime::from_handle`), else it dials inline.
    #[cfg(any(
        feature = "tokio-runtime",
        all(
            feature = "runtime-prime-executor",
            feature = "runtime-prime-inbox-alloc",
            feature = "runtime-prime-reactor",
            feature = "runtime-prime-bgpool"
        )
    ))]
    runtime: Option<Arc<dyn Runtime>>,
}

impl Client {
    /// Build from an inline spec; handle is materialized lazily on first
    /// `send()` so this stays sync.
    pub fn from_value(spec: Value) -> Result<Self, ProximaError> {
        Ok(Self {
            inner: Arc::new(Inner {
                spec,
                handle: OnceCell::new(),
                injected: None,
                extra_factories: Vec::new(),
                #[cfg(any(
                    feature = "tokio-runtime",
                    all(
                        feature = "runtime-prime-executor",
                        feature = "runtime-prime-inbox-alloc",
                        feature = "runtime-prime-reactor",
                        feature = "runtime-prime-bgpool"
                    )
                ))]
                runtime: None,
            }),
        })
    }

    /// Build a client that dispatches straight to an in-process
    /// [`Handler`](crate::pipe::Handler) — a mounted App, a fake daemon, or any composed
    /// handle — with no spec resolution and no socket. The substrate-native way to
    /// drive a server with the real `Client` in one process (tests, embedding);
    /// pairs with [`into_handle`](crate::pipe::into_handle) on the serve side.
    pub fn from_pipe(
        pipe: impl crate::pipe::Handler<
            In = crate::request::Request<Bytes>,
            Out = crate::request::Response<Bytes>,
        > + 'static,
    ) -> Self {
        Self::from_handle(crate::pipe::into_handle(pipe))
    }

    /// Build a client bound to an already-composed [`PipeHandle`] — the
    /// lower-level half of [`from_pipe`](Self::from_pipe) for callers that already
    /// hold a handle. Dispatch calls the handle directly; no transport is loaded.
    pub fn from_handle(handle: PipeHandle) -> Self {
        Self {
            inner: Arc::new(Inner {
                spec: Value::Null,
                handle: OnceCell::new(),
                injected: Some(handle),
                extra_factories: Vec::new(),
                #[cfg(any(
                    feature = "tokio-runtime",
                    all(
                        feature = "runtime-prime-executor",
                        feature = "runtime-prime-inbox-alloc",
                        feature = "runtime-prime-reactor",
                        feature = "runtime-prime-bgpool"
                    )
                ))]
                runtime: None,
            }),
        }
    }

    /// Point a client straight at an HTTP(S) base url — the one-liner over
    /// `Client::builder().http(url).build()` / `from_value({"http": url})`.
    /// The backend (prime h1 by default) resolves through the same registry
    /// as every other spec, so the body streams lazily. Reach for `builder()`
    /// when you also need a proxy, retry, or an injected runtime.
    pub fn http(url: impl Into<String>) -> Result<Self, ProximaError> {
        let mut spec = serde_json::Map::new();
        spec.insert("http".to_string(), Value::String(url.into()));
        Self::from_value(Value::Object(spec))
    }

    /// Like [`http`](Self::http) but composes the `Discard` response preset:
    /// the prime h1 backend drains each response body to the keep-alive
    /// boundary and copies only framing headers — never materializing the
    /// payload. The one-liner for a load generator / liveness prober that
    /// cares about completion, not content. Maps to the spec
    /// `{"http": url, "response": {"body": "drain", "headers": "framing"}}`.
    pub fn http_discard(url: impl Into<String>) -> Result<Self, ProximaError> {
        let mut spec = serde_json::Map::new();
        spec.insert("http".to_string(), Value::String(url.into()));
        spec.insert(
            "response".to_string(),
            serde_json::json!({ "body": "drain", "headers": "framing" }),
        );
        Self::from_value(Value::Object(spec))
    }

    /// Fluent builder: `Client::builder().http(url).runtime(rt).build()`.
    /// `runtime` is optional.
    #[must_use]
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    pub fn from_sugar(value: Value) -> Result<Self, ProximaError> {
        Self::from_value(crate::sugar::desugar(value)?)
    }

    pub async fn from_path(path: impl AsRef<Path>) -> Result<Self, ProximaError> {
        // one-shot config load at construction time; blocking std::fs is
        // simplest and correct here (not a hot-path), and removes the only
        // reason this constructor needed tokio.
        let raw = std::fs::read_to_string(path.as_ref()).map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!("read client config: {err}")))
        })?;
        let value: toml::Value = toml::from_str(&raw)
            .map_err(|err| ProximaError::Config(format!("client config toml: {err}")))?;
        let json = toml_to_json(value);
        Self::from_value(json)
    }

    pub fn call(&self, method: impl Into<String>, path: impl Into<String>) -> RequestBuilder {
        RequestBuilder::new(self.clone(), method, path)
    }

    /// `GET path` — sugar over [`call`](Self::call).
    pub fn get(&self, path: impl Into<String>) -> RequestBuilder {
        self.call("GET", path)
    }

    /// `POST path` — sugar over [`call`](Self::call).
    pub fn post(&self, path: impl Into<String>) -> RequestBuilder {
        self.call("POST", path)
    }

    /// `PUT path` — sugar over [`call`](Self::call).
    pub fn put(&self, path: impl Into<String>) -> RequestBuilder {
        self.call("PUT", path)
    }

    /// `PATCH path` — sugar over [`call`](Self::call).
    pub fn patch(&self, path: impl Into<String>) -> RequestBuilder {
        self.call("PATCH", path)
    }

    /// `DELETE path` — sugar over [`call`](Self::call).
    pub fn delete(&self, path: impl Into<String>) -> RequestBuilder {
        self.call("DELETE", path)
    }

    /// `HEAD path` — sugar over [`call`](Self::call).
    pub fn head(&self, path: impl Into<String>) -> RequestBuilder {
        self.call("HEAD", path)
    }

    pub(crate) async fn handle(&self) -> Result<PipeHandle, ProximaError> {
        if let Some(handle) = &self.inner.injected {
            return Ok(handle.clone());
        }
        // peek the cache before cloning — once the handle is built, every send
        // returns here with a single Arc clone, never re-cloning the spec +
        // factories just to build a closure `get_or_try_init` won't run.
        if let Some(handle) = self.inner.handle.get() {
            return Ok(handle.clone());
        }
        let spec = self.inner.spec.clone();
        let extra_factories = self.inner.extra_factories.clone();
        let handle = self
            .inner
            .handle
            .get_or_try_init(|| async move {
                let context = LoadContext::with_default_registry()?;
                // layer caller-supplied factories on top of the defaults so an
                // out-of-crate protocol resolves through the same `load()`.
                for factory in extra_factories {
                    context.registry.register(factory)?;
                }
                load(Spec::Inline(spec), &context).await
            })
            .await?;
        Ok(handle.clone())
    }

    /// The single dispatch seam every request goes through — shared by
    /// [`RequestBuilder::send`](crate::client::RequestBuilder::send) and the
    /// [`Handler`](crate::pipe::Handler) impl. On a prime worker, call the composed
    /// handle directly; off a worker (no `CURRENT_REACTOR`, so proxima's
    /// `TcpStream` can't construct here) hop onto the client's runtime via
    /// [`call_on_worker`](Self::call_on_worker). Composing `Client` as a transport
    /// stage therefore carries its own runtime, callable from any thread.
    pub(crate) async fn dispatch(
        &self,
        request: crate::request::Request<Bytes>,
    ) -> Result<crate::request::Response<Bytes>, ProximaError> {
        let handle = self.handle().await?;

        // tokio transport — a `"wire":"tokio"` upstream (default is prime), or a
        // tokio-only build. construct the stream where a tokio reactor is driven:
        // poll inline if already on one, else hop onto the injected runtime
        // (`from_handle`) or a process-shared tokio sidecar. the tokio-compat path
        // for systems that stay on tokio. checks the SAME conjunction as the
        // prime branch below, not just the bare `runtime-prime` feature — a
        // build missing one of prime's sub-features has no working prime
        // dispatch path even though `runtime-prime` itself is on, so it must
        // fall into this branch unconditionally rather than through to the
        // ungated `SendPipe::call` past both branches.
        #[cfg(feature = "tokio")]
        if self.wire_is_tokio()
            || cfg!(not(all(
                feature = "runtime-prime-executor",
                feature = "runtime-prime-inbox-alloc",
                feature = "runtime-prime-reactor",
                feature = "runtime-prime-bgpool"
            )))
        {
            if tokio::runtime::Handle::try_current().is_ok() {
                return SendPipe::call(&handle, request).await;
            }
            if let Some(runtime) = self.tokio_hop_runtime()? {
                return Self::hop_onto(runtime, handle, request).await;
            }
            return SendPipe::call(&handle, request).await;
        }

        // prime transport (default). gate matches `call_on_worker`'s: this
        // branch also calls `current_core` (needs executor + reactor +
        // inbox-alloc-or-dynamic per `prime::os::core_shard`'s own gate),
        // and that's a subset of the executor+inbox-alloc+reactor+bgpool
        // this block's other call, `call_on_worker`, requires.
        #[cfg(all(
            feature = "runtime-prime-executor",
            feature = "runtime-prime-inbox-alloc",
            feature = "runtime-prime-reactor",
            feature = "runtime-prime-bgpool"
        ))]
        if prime::os::core_shard::current_core().is_none() {
            return self.call_on_worker(handle, request).await;
        }
        SendPipe::call(&handle, request).await
    }

    /// Whether this client's upstream selected the tokio-compat wire via
    /// `"wire": "tokio"` in its spec. Absent ⟹ prime (the default). Only
    /// called from the `"tokio"`-gated dispatch arm (that arm needs the real
    /// `tokio` crate for `tokio::runtime::Handle`, which the bare
    /// `tokio-runtime` marker feature does not link) — so this must match
    /// that gate, not the narrower marker.
    #[cfg(feature = "tokio")]
    fn wire_is_tokio(&self) -> bool {
        self.inner.spec.get("wire").and_then(Value::as_str) == Some("tokio")
    }

    /// The runtime a tokio-wire dial hops onto when off a tokio reactor: the
    /// injected one if present, else (in a prime-default build) a process-shared
    /// tokio sidecar. `None` ⟹ no runtime available; dial inline.
    #[cfg(feature = "tokio")]
    fn tokio_hop_runtime(&self) -> Result<Option<Arc<dyn Runtime>>, ProximaError> {
        if let Some(injected) = self.inner.runtime.clone() {
            return Ok(Some(injected));
        }
        // a prime-default build with the tokio runtime impl gets a shared sidecar
        // for `"wire":"tokio"` upstreams; otherwise dial inline.
        #[cfg(all(feature = "runtime-prime", feature = "runtime-tokio"))]
        {
            return Ok(Some(shared_tokio_runtime()?));
        }
        #[cfg(not(all(feature = "runtime-prime", feature = "runtime-tokio")))]
        Ok(None)
    }

    /// run `Handler::call` on a prime worker and return the response. used by
    /// `send()` when the caller is NOT already on a worker (no CURRENT_REACTOR
    /// → proxima's TcpStream can't construct). dispatches onto the injected
    /// runtime, or a process-shared prime runtime the client owns, then awaits
    /// the result over a oneshot.
    ///
    /// mirrors `shared_prime_runtime`'s gate: its `None` arm calls that
    /// function unconditionally, so this must exist under the same conjunction.
    #[cfg(all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    ))]
    pub(crate) async fn call_on_worker(
        &self,
        handle: PipeHandle,
        request: crate::request::Request<Bytes>,
    ) -> Result<crate::request::Response<Bytes>, ProximaError> {
        let runtime: Arc<dyn Runtime> = match self.inner.runtime.clone() {
            Some(runtime) => runtime,
            None => shared_prime_runtime()?,
        };
        Self::hop_onto(runtime, handle, request).await
    }

    /// Move `Handler::call` onto a runtime worker (`CoreId(0)`) and await the
    /// response over a oneshot — the shared hop both the prime off-worker path
    /// and the tokio-host path take. The runtime must host the reactor the
    /// transport needs (prime worker for a prime stream, tokio worker for a
    /// tokio stream); selecting that runtime is the caller's job.
    #[cfg(any(
        feature = "tokio-runtime",
        all(
            feature = "runtime-prime-executor",
            feature = "runtime-prime-inbox-alloc",
            feature = "runtime-prime-reactor",
            feature = "runtime-prime-bgpool"
        )
    ))]
    async fn hop_onto(
        runtime: Arc<dyn Runtime>,
        handle: PipeHandle,
        request: crate::request::Request<Bytes>,
    ) -> Result<crate::request::Response<Bytes>, ProximaError> {
        use crate::runtime::CoreId;
        let (sender, receiver) = futures::channel::oneshot::channel();
        let future = Box::pin(async move {
            let result = SendPipe::call(&handle, request).await;
            let _ = sender.send(result);
        });
        runtime
            .spawn_on_core(CoreId(0), future)
            .map_err(|err| ProximaError::Upstream(format!("client dispatch: {err}")))?;
        receiver
            .await
            .map_err(|_| ProximaError::Upstream("client dispatch cancelled".to_string()))?
    }
}

/// `Client` is itself a [`Handler`](crate::pipe::Handler): composing it as a transport
/// stage routes the request through [`dispatch`](Client::dispatch), so the
/// on/off-worker hop + self-owned runtime apply everywhere `Client` is used —
/// not just via the `call(..).send()` builder. This is the seam the OTLP exporter
/// (and any codec/middleware chain) composes so the wire send always goes through
/// the `proxima::Client` API rather than a hand-extracted transport handle.
impl SendPipe for Client {
    type In = crate::request::Request<Bytes>;
    type Out = crate::request::Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: crate::request::Request<Bytes>,
    ) -> impl core::future::Future<Output = Result<crate::request::Response<Bytes>, ProximaError>> + Send
    {
        let client = self.clone();
        async move { client.dispatch(request).await }
    }
}


/// The wire a [`Client`] speaks, the transport axis of protocol × transport ×
/// auth. Lowers to the spec `transport` key the factory registry resolves; the
/// app protocol (`.http`/`.grpc`/…) and this compose into the concrete upstream.
///
/// `Auto` (the default) negotiates over the scheme/ALPN — `https` → TLS with
/// h1/h2 by ALPN, `http` → plaintext h1. `Tcp`/`Tls` force the wire; `H3` is
/// HTTP/3 over QUIC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Transport {
    /// Negotiate from the URL scheme + ALPN (the default).
    #[default]
    Auto,
    /// Plaintext TCP.
    Tcp,
    /// TLS over TCP (h1/h2 by ALPN).
    Tls,
    /// HTTP/3 over QUIC.
    H3,
}

impl Transport {
    /// The spec-string this transport lowers to.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Transport::Auto => "auto",
            Transport::Tcp => "tcp",
            Transport::Tls => "tls",
            Transport::H3 => "h3",
        }
    }
}

/// Fluent builder for [`Client`]: `Client::builder().http(url).transport(t).build()`.
#[derive(Default)]
pub struct ClientBuilder {
    spec: serde_json::Map<String, Value>,
    extra_factories: Vec<Arc<dyn PipeFactory>>,
    #[cfg(any(
        feature = "tokio-runtime",
        all(
            feature = "runtime-prime-executor",
            feature = "runtime-prime-inbox-alloc",
            feature = "runtime-prime-reactor",
            feature = "runtime-prime-bgpool"
        )
    ))]
    runtime: Option<Arc<dyn Runtime>>,
}

/// A protocol defined outside this crate, plugged into `Client` (the
/// extensible half of the protocol surface). An impl supplies the spec it
/// lowers to AND the `PipeFactory` that resolves it — so a kafka/redis/private
/// crate adds a wire to `proxima::Client` with no edit here. Pair with a
/// per-crate extension trait (e.g. `FooClientExt::foo`) for fluent sugar.
pub trait ClientProtocol {
    /// The spec this protocol lowers to (e.g. `{"type":"foo", ...}`); merged
    /// into the builder's spec.
    fn spec(&self) -> Value;
    /// The factory that resolves the spec's terminal, registered on top of the
    /// defaults for this client.
    fn factory(&self) -> Arc<dyn PipeFactory>;
}

impl ClientBuilder {
    // protocol axis (`.http()`/`.https()`/`.grpc()`) and transport axis
    // (`.auto()`/`.tcp()`/`.tls()`/`.h3()`/`.proxy()`) are the `ProtocolSugar` /
    // `TransportSugar` traits, blanket-impl'd over the `SpecBuilder` below.
    // `use proxima::{ProtocolSugar, TransportSugar}` brings them into scope —
    // the method is on the page because you imported it. The typed `.transport(
    // Transport)` and `.auth(ClientAuth)` escape hatches stay inherent.

    /// Point the client at a PostgreSQL server by DSN
    /// (`postgres://user:pw@host:port/db`). Lowers to the `pgwire` protocol
    /// terminal (`{"type":"pgwire","dsn":...}`). The SQL reply is typed, not an
    /// HTTP body, so it rides the response `Carry`:
    /// `response.reply::<proxima_pgwire::PgReply>()`.
    #[cfg(feature = "pgwire-client")]
    #[must_use]
    pub fn pgwire(mut self, dsn: impl Into<String>) -> Self {
        self.spec
            .insert("type".to_string(), Value::String("pgwire".to_string()));
        self.spec
            .insert("dsn".to_string(), Value::String(dsn.into()));
        self
    }

    /// Point the client at a Redis server by DSN
    /// (`redis://[user:pass@]host[:port][/db]`). Lowers to the `redis` protocol
    /// terminal (`{"type":"redis","dsn":...}`). The reply is typed, not an HTTP
    /// body, so it rides the response `Carry`:
    /// `response.reply::<proxima_redis::RespValue>()`.
    #[cfg(feature = "redis-client")]
    #[must_use]
    pub fn redis(mut self, dsn: impl Into<String>) -> Self {
        self.spec
            .insert("type".to_string(), Value::String("redis".to_string()));
        self.spec
            .insert("dsn".to_string(), Value::String(dsn.into()));
        self
    }

    /// Point the client at a Valkey server by DSN — Valkey speaks the same RESP
    /// wire protocol as Redis, so this aliases [`redis`](Self::redis) onto the
    /// one `redis` factory (one codec, one client, one terminal cover both).
    #[cfg(feature = "redis-client")]
    #[must_use]
    pub fn valkey(self, dsn: impl Into<String>) -> Self {
        self.redis(dsn)
    }

    /// Select the wire ([`Transport`]) — the transport axis. Lowers to the spec
    /// `transport` key; the app-protocol factory (`http`/`grpc`) composes it.
    #[must_use]
    pub fn transport(mut self, transport: Transport) -> Self {
        self.spec.insert(
            "transport".to_string(),
            Value::String(transport.as_str().to_string()),
        );
        self
    }

    /// Merge an arbitrary spec key (retry, synth, name, …) — same shape as a
    /// `[[pipe]]` entry. Lets the builder express anything `from_value` can.
    #[must_use]
    pub fn spec(mut self, key: impl Into<String>, value: Value) -> Self {
        self.spec.insert(key.into(), value);
        self
    }

    /// Plug in an out-of-crate protocol: merge its spec and register its
    /// factory. The typed, no-import path — `Client::builder().protocol(Foo::dsn(..))`.
    #[must_use]
    pub fn protocol(mut self, protocol: impl ClientProtocol) -> Self {
        if let Value::Object(map) = protocol.spec() {
            for (key, value) in map {
                self.spec.insert(key, value);
            }
        }
        self.extra_factories.push(protocol.factory());
        self
    }

    /// Register a `PipeFactory` on top of the default set for this client — the
    /// escape hatch for a protocol/middleware factory this crate doesn't know.
    /// Pair with `.spec("type", json!("<name>"))` (or a fluent ext trait) to
    /// select it.
    #[must_use]
    pub fn factory(mut self, factory: Arc<dyn PipeFactory>) -> Self {
        self.extra_factories.push(factory);
        self
    }

    /// Inject the runtime `send()` dispatches onto when off-worker. Omit to let
    /// the client own a process-shared prime runtime (prime builds). On the tokio
    /// path, pass `TokioPerCoreRuntime::from_handle(host)` so a client called off
    /// the host runtime dials onto it instead of failing for want of a reactor.
    #[cfg(any(
        feature = "tokio-runtime",
        all(
            feature = "runtime-prime-executor",
            feature = "runtime-prime-inbox-alloc",
            feature = "runtime-prime-reactor",
            feature = "runtime-prime-bgpool"
        )
    ))]
    #[must_use]
    pub fn runtime(mut self, runtime: Arc<dyn Runtime>) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// Attach outbound authentication (the auth axis) — both first-class
    /// surfaces: `.auth(ClientAuth::bearer(t))` / `.auth(ClientAuth::basic(u,p))`
    /// or the fluent builder `.auth(OauthAuth::builder()…build())`. Lowers to a
    /// `client-auth` middleware wrapping the protocol terminal; the credential
    /// is injected per request by a pipe driving a `proxima-auth` FSM.
    #[must_use]
    pub fn auth(mut self, auth: impl Into<crate::settings::ClientAuth>) -> Self {
        let client_auth: crate::settings::ClientAuth = auth.into();
        let Spec::Inline(value) = client_auth.into() else {
            return self;
        };
        let entry = self
            .spec
            .entry("middleware".to_string())
            .or_insert_with(|| Value::Array(Vec::new()));
        if let Value::Array(array) = entry {
            array.push(value);
        }
        self
    }

    /// Static bearer token (the auth axis) — sugar for
    /// `.auth(ClientAuth::bearer(token))`. Lowers through the typed `ClientAuth`
    /// (the source of truth for the auth spec shape), so config and fluent agree.
    #[must_use]
    pub fn bearer_token(self, token: impl Into<String>) -> Self {
        self.auth(crate::settings::ClientAuth::bearer(token))
    }

    /// HTTP Basic credentials (the auth axis) — sugar for
    /// `.auth(ClientAuth::basic(user, password))`. Like `.bearer_token`, a static
    /// credential injected per request; the stateful schemes (oauth/scram) ride
    /// the same `.auth()` door but lower to an FSM-driven handshake pipe.
    #[must_use]
    pub fn login(self, user: impl Into<String>, password: impl Into<String>) -> Self {
        self.auth(crate::settings::ClientAuth::basic(user, password))
    }

    /// Build the immutable client.
    pub fn build(self) -> Result<Client, ProximaError> {
        Ok(Client {
            inner: Arc::new(Inner {
                spec: Value::Object(self.spec),
                handle: OnceCell::new(),
                injected: None,
                extra_factories: self.extra_factories,
                #[cfg(any(
                    feature = "tokio-runtime",
                    all(
                        feature = "runtime-prime-executor",
                        feature = "runtime-prime-inbox-alloc",
                        feature = "runtime-prime-reactor",
                        feature = "runtime-prime-bgpool"
                    )
                ))]
                runtime: self.runtime,
            }),
        })
    }
}

/// The base spec seam ([`proxima_config::sugar::SpecBuilder`]): the axis sugar
/// ([`ProtocolSugar`](proxima_config::sugar::ProtocolSugar) /
/// [`TransportSugar`](proxima_config::sugar::TransportSugar)) is blanket-impl'd over
/// this, so a `use` of an axis trait lights up its methods on `ClientBuilder`.
/// `set` reuses the existing `.spec()` so there is exactly one write path.
impl proxima_config::sugar::SpecBuilder for ClientBuilder {
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

/// the process-shared prime runtime for off-worker client calls. one 1-core
/// runtime, leaked, reused by every runtime-less client — the same pattern the
/// `#[proxima::test]` harness uses for its client dispatch.
///
/// `prime::os::runtime::PrimeRuntime` (the type this builds) has its own
/// module-level gate requiring executor + inbox-alloc + reactor + bgpool
/// together; the bare `runtime-prime` feature doesn't guarantee any of those.
#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
fn shared_prime_runtime() -> Result<Arc<dyn Runtime>, ProximaError> {
    use std::sync::OnceLock;

    use prime::os::runtime::PrimeRuntime;

    static SHARED: OnceLock<Arc<PrimeRuntime>> = OnceLock::new();
    if let Some(existing) = SHARED.get() {
        let runtime: Arc<dyn Runtime> = existing.clone();
        return Ok(runtime);
    }
    let built = Arc::new(PrimeRuntime::new(1).map_err(|err| {
        ProximaError::Upstream(format!("build shared prime client runtime: {err}"))
    })?);
    // a concurrent caller may have won the race; keep whichever landed first.
    let runtime: Arc<dyn Runtime> = SHARED.get_or_init(|| built).clone();
    Ok(runtime)
}

/// a process-shared tokio sidecar for `"wire":"tokio"` upstreams dialed from a
/// prime-default process — one 1-core `TokioPerCoreRuntime`, leaked and reused,
/// so a prime worker can dial a tokio-only upstream without the caller owning a
/// tokio runtime. the compatibility shim for systems that stay on tokio.
/// only reachable through `tokio_hop_runtime`, which itself requires the full
/// `tokio` feature — this gate must include it too, or it goes dead whenever
/// `runtime-prime` + `runtime-tokio` are on without `tokio`.
#[cfg(all(feature = "tokio", feature = "runtime-prime", feature = "runtime-tokio"))]
fn shared_tokio_runtime() -> Result<Arc<dyn Runtime>, ProximaError> {
    use std::sync::OnceLock;

    use crate::runtime::TokioPerCoreRuntime;

    static SHARED: OnceLock<Arc<TokioPerCoreRuntime>> = OnceLock::new();
    if let Some(existing) = SHARED.get() {
        let runtime: Arc<dyn Runtime> = existing.clone();
        return Ok(runtime);
    }
    let built = Arc::new(TokioPerCoreRuntime::new(1).map_err(|err| {
        ProximaError::Upstream(format!("build shared tokio client sidecar: {err}"))
    })?);
    let runtime: Arc<dyn Runtime> = SHARED.get_or_init(|| built).clone();
    Ok(runtime)
}

fn toml_to_json(value: toml::Value) -> Value {
    match value {
        toml::Value::String(text) => Value::String(text),
        toml::Value::Integer(number) => Value::Number(number.into()),
        toml::Value::Float(number) => serde_json::Number::from_f64(number)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        toml::Value::Boolean(flag) => Value::Bool(flag),
        toml::Value::Datetime(timestamp) => Value::String(timestamp.to_string()),
        toml::Value::Array(items) => Value::Array(items.into_iter().map(toml_to_json).collect()),
        toml::Value::Table(table) => Value::Object(
            table
                .into_iter()
                .map(|(key, value)| (key, toml_to_json(value)))
                .collect(),
        ),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_config::sugar::{ProtocolSugar, TransportSugar};
    use serde_json::json;

    /// A protocol defined "outside" this crate: a factory the default registry
    /// has never heard of, returning a Handler that does no I/O. Proves the
    /// extensibility seam — `.factory()` + `{"type":"mock"}` reaches it through
    /// the same `Client`, no edit to `load.rs`.
    #[allow(dead_code)]
    struct MockFactory;

    impl crate::pipe_factory::PipeFactory for MockFactory {
        fn name(&self) -> &str {
            "mock"
        }

        fn build(
            &self,
            _spec: &Value,
            _inner: Option<PipeHandle>,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>,
        > {
            Box::pin(async move { Ok(crate::pipe::into_handle(MockPipe)) })
        }
    }

    #[allow(dead_code)]
    struct MockPipe;

    impl SendPipe for MockPipe {
        type In = crate::request::Request<Bytes>;
        type Out = crate::request::Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: crate::request::Request<Bytes>,
        ) -> impl std::future::Future<
            Output = Result<crate::request::Response<Bytes>, ProximaError>,
        > + Send {
            async move { Ok(crate::request::Response::ok("mock-protocol-reply")) }
        }
    }

    #[cfg(feature = "runtime-prime")]
    #[test]
    fn external_factory_registered_via_builder_is_reachable() {
        // a factory the crate does not know, plugged in at the builder and
        // selected by `{"type":"mock"}` — the "extensible" half of the surface.
        let body = futures::executor::block_on(async {
            let client = Client::builder()
                .factory(Arc::new(MockFactory))
                .spec("type", json!("mock"))
                .build()
                .expect("build");
            let response = client.call("GET", "/").send().await.expect("send");
            response.text().await.expect("text")
        });
        assert_eq!(body, "mock-protocol-reply");
    }

    #[test]
    fn from_pipe_dispatches_to_an_in_process_pipe_with_no_spec() {
        // the in-process seam: a client built straight from a mounted pipe routes
        // every request to it, loading no spec and binding no socket — the shape a
        // test or embedding uses to drive a mounted App/fake with the real client.
        let body = futures::executor::block_on(async {
            let client = Client::from_pipe(MockPipe);
            let response = client.call("GET", "/").send().await.expect("send");
            response.text().await.expect("text")
        });
        assert_eq!(body, "mock-protocol-reply");
    }

    /// the self-contained client: a plain `#[test]` runs OFF a prime worker
    /// (no CURRENT_REACTOR, no tokio), yet `send()` succeeds — it auto-dispatches
    /// onto the shared prime runtime. this is the path that lets reqwest leave:
    /// any caller, any thread, can use `proxima::Client`. server + drive are
    /// std + `futures::executor::block_on` — zero tokio.
    // mirrors `shared_prime_runtime`'s gate: the bare `runtime-prime` flag
    // doesn't guarantee executor + inbox-alloc + reactor + bgpool are all
    // present, and this test exercises exactly that off-worker dispatch path.
    #[cfg(all(
        feature = "http-prime",
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    ))]
    #[test]
    fn client_send_auto_dispatches_when_off_worker() {
        use std::io::{Read as _, Write as _};
        use std::net::{Ipv4Addr, SocketAddr, TcpListener};
        use std::sync::mpsc;

        let (port_tx, port_rx) = mpsc::channel();
        let server = std::thread::spawn(move || {
            let listener =
                TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).expect("bind");
            port_tx
                .send(listener.local_addr().expect("addr").port())
                .expect("send port");
            let (mut socket, _) = listener.accept().expect("accept");
            let mut buffer = Vec::new();
            let mut scratch = [0_u8; 1024];
            loop {
                let read = socket.read(&mut scratch).expect("read");
                buffer.extend_from_slice(&scratch[..read]);
                if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello")
                .expect("write");
            socket.flush().expect("flush");
        });
        let port = port_rx.recv().expect("port");

        // this libtest thread is NOT a prime worker. drive the async send with
        // futures' executor (not tokio); send() auto-dispatches to the runtime.
        let body = futures::executor::block_on(async {
            let client = Client::from_value(json!({ "http": format!("http://127.0.0.1:{port}") }))
                .expect("build");
            let response = client
                .call("GET", "/path")
                .send()
                .await
                .expect("send off-worker");
            assert_eq!(response.status(), 200);
            response.bytes().await.expect("bytes")
        });
        assert_eq!(&body[..], b"hello");
        server.join().expect("server join");
    }

    /// Live oauth (#3) end-to-end through `proxima::Client` against real loopback
    /// servers: the `.auth(OauthAuth::builder()…)` builder lowers to a client-auth
    /// middleware, whose oauth pipe fetches a token from a real HTTP token
    /// endpoint, then injects `Bearer <token>` on the real HTTP backend call. No
    /// mocks — both sub-pipes are the prime HTTP client. Proves config/fluent →
    /// middleware → pipe → FSM → exchange-sub-pipe → inject, end to end.
    // mirrors `shared_prime_runtime`'s gate: the bare `runtime-prime` flag
    // doesn't guarantee executor + inbox-alloc + reactor + bgpool are all
    // present, and this test exercises exactly that off-worker dispatch path.
    #[cfg(all(
        feature = "http-prime",
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    ))]
    #[test]
    fn oauth_client_fetches_token_and_injects_bearer_against_real_servers() {
        use std::io::{Read as _, Write as _};
        use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
        use std::sync::mpsc;
        use std::sync::{Arc, Mutex};

        use crate::settings::OauthAuth;

        fn read_head(socket: &mut TcpStream) -> Vec<u8> {
            let mut buffer = Vec::new();
            let mut scratch = [0_u8; 1024];
            loop {
                let read = socket.read(&mut scratch).expect("read");
                buffer.extend_from_slice(&scratch[..read]);
                if read == 0 || buffer.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            buffer
        }

        // token endpoint: any POST -> a client-credentials token response.
        let (token_tx, token_rx) = mpsc::channel();
        let token_server = std::thread::spawn(move || {
            let listener =
                TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).expect("bind token");
            token_tx
                .send(listener.local_addr().expect("addr").port())
                .expect("send token port");
            let (mut socket, _) = listener.accept().expect("accept token");
            let _ = read_head(&mut socket);
            let body = b"{\"access_token\":\"at-xyz\",\"expires_in\":3600}";
            socket
                .write_all(
                    format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len()).as_bytes(),
                )
                .expect("write token head");
            socket.write_all(body).expect("write token body");
            socket.flush().expect("flush token");
        });
        let token_port = token_rx.recv().expect("token port");

        // backend: captures the Authorization header the oauth pipe injected.
        let (backend_tx, backend_rx) = mpsc::channel();
        let auth_seen: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let auth_for_server = auth_seen.clone();
        let backend_server = std::thread::spawn(move || {
            let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
                .expect("bind backend");
            backend_tx
                .send(listener.local_addr().expect("addr").port())
                .expect("send backend port");
            let (mut socket, _) = listener.accept().expect("accept backend");
            let head = read_head(&mut socket);
            let text = String::from_utf8_lossy(&head);
            let authorization = text
                .lines()
                .find(|line| line.to_ascii_lowercase().starts_with("authorization:"))
                .map(|line| line["authorization:".len()..].trim().to_string());
            *auth_for_server.lock().expect("auth lock") = authorization;
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                .expect("write backend");
            socket.flush().expect("flush backend");
        });
        let backend_port = backend_rx.recv().expect("backend port");

        let body = futures::executor::block_on(async {
            let client = Client::builder()
                .http(format!("http://127.0.0.1:{backend_port}"))
                .auth(
                    OauthAuth::builder()
                        .token_url(format!("http://127.0.0.1:{token_port}"))
                        .client_id("id")
                        .client_secret("secret")
                        .build(),
                )
                .build()
                .expect("build");
            let response = client.call("GET", "/").send().await.expect("send");
            assert_eq!(response.status(), 200);
            response.bytes().await.expect("bytes")
        });
        assert_eq!(&body[..], b"ok");
        assert_eq!(
            auth_seen.lock().expect("auth lock").as_deref(),
            Some("Bearer at-xyz")
        );
        token_server.join().expect("token join");
        backend_server.join().expect("backend join");
    }

    #[proxima::test]
    async fn from_value_with_http_spec_builds_lazy_handle_on_first_use() {
        let client = Client::from_value(json!({
            "synth": { "status": 200, "body": "hi" },
        }))
        .expect("build");
        assert!(client.inner.handle.get().is_none());
        let _handle = client.handle().await.expect("handle");
        assert!(client.inner.handle.get().is_some());
    }

    #[proxima::test]
    async fn call_returns_request_builder_for_method_path() {
        let client = Client::from_value(json!({
            "synth": { "status": 200, "body": "hi" },
        }))
        .expect("build");
        let request = client.call("GET", "/foo");
        assert_eq!(request.method(), "GET");
        assert_eq!(request.path(), "/foo");
    }

    #[test]
    fn verbs_map_to_methods_and_builder_lowers_axes() {
        let client =
            Client::from_value(json!({ "synth": { "status": 200, "body": "ok" } })).expect("build");
        assert_eq!(client.post("/x").method(), "POST");
        assert_eq!(client.get("/y").method(), "GET");
        assert_eq!(client.delete("/z").method(), "DELETE");
        assert_eq!(client.put("/p").method(), "PUT");
        assert_eq!(client.patch("/q").method(), "PATCH");
        assert_eq!(client.head("/h").method(), "HEAD");

        // protocol × transport × proxy lower to the spec keys the registry resolves.
        let built = Client::builder()
            .https("https://api.example.com")
            .transport(Transport::Tls)
            .proxy("http://127.0.0.1:8080")
            .build()
            .expect("builder build");
        assert_eq!(
            built.inner.spec.get("http").and_then(Value::as_str),
            Some("https://api.example.com")
        );
        assert_eq!(
            built.inner.spec.get("transport").and_then(Value::as_str),
            Some("tls")
        );
        assert_eq!(
            built.inner.spec.get("proxy").and_then(Value::as_str),
            Some("http://127.0.0.1:8080")
        );

        // the headline parity: the fluent builder and the config file lower to
        // the IDENTICAL spec (one door). `.tls()` is the TransportSugar twin of
        // `transport = "tls"`; both Clients carry the same `inner.spec`.
        let fluent = Client::builder()
            .http("http://api.example.com")
            .tls()
            .proxy("http://127.0.0.1:8080")
            .build()
            .expect("fluent build");
        let config = Client::from_value(json!({
            "http": "http://api.example.com",
            "transport": "tls",
            "proxy": "http://127.0.0.1:8080",
        }))
        .expect("config build");
        assert_eq!(fluent.inner.spec, config.inner.spec);

        // grpc protocol lowers to the `grpc` key (transport factory pending).
        let grpc = Client::builder()
            .grpc("https://collector:4317")
            .build()
            .expect("grpc build");
        assert_eq!(
            grpc.inner.spec.get("grpc").and_then(Value::as_str),
            Some("https://collector:4317")
        );
    }

    #[test]
    fn auth_lowers_both_surfaces_to_a_client_auth_middleware() {
        use crate::settings::{ClientAuth, OauthAuth};

        // surface 1 — the constructor sugar
        let bearer = Client::builder()
            .http("http://x")
            .auth(ClientAuth::bearer("tok"))
            .build()
            .expect("build");
        let mw = bearer
            .inner
            .spec
            .get("middleware")
            .and_then(Value::as_array)
            .expect("middleware");
        assert_eq!(mw.len(), 1);
        assert_eq!(
            mw[0].get("type").and_then(Value::as_str),
            Some("client-auth")
        );
        assert_eq!(mw[0].get("scheme").and_then(Value::as_str), Some("bearer"));
        assert_eq!(mw[0].get("token").and_then(Value::as_str), Some("tok"));

        // the `.bearer_token()` axis sugar must lower to the IDENTICAL spec as
        // the explicit `.auth(ClientAuth::bearer(..))` — same source of truth.
        let sugar = Client::builder()
            .http("http://x")
            .bearer_token("tok")
            .build()
            .expect("build");
        assert_eq!(sugar.inner.spec, bearer.inner.spec);

        // `.login()` is the Basic-credential axis sugar (ClientAuth::basic).
        let login = Client::builder()
            .http("http://x")
            .login("u", "p")
            .build()
            .expect("build");
        let lmw = login
            .inner
            .spec
            .get("middleware")
            .and_then(Value::as_array)
            .expect("middleware");
        assert_eq!(lmw[0].get("scheme").and_then(Value::as_str), Some("basic"));
        assert_eq!(lmw[0].get("username").and_then(Value::as_str), Some("u"));
        assert_eq!(lmw[0].get("password").and_then(Value::as_str), Some("p"));

        // surface 2 — the fluent builder (OauthAuth -> ClientAuth via Into)
        let oauth = Client::builder()
            .http("http://x")
            .auth(
                OauthAuth::builder()
                    .token_url("https://idp/token")
                    .client_id("id")
                    .client_secret("secret")
                    .build(),
            )
            .build()
            .expect("build");
        let mw = oauth
            .inner
            .spec
            .get("middleware")
            .and_then(Value::as_array)
            .expect("middleware");
        assert_eq!(mw[0].get("scheme").and_then(Value::as_str), Some("oauth"));
        assert_eq!(
            mw[0].get("token_url").and_then(Value::as_str),
            Some("https://idp/token")
        );

        // the new auth variants ride the SAME `.auth()` door (SigV4Auth /
        // DigestAuth -> ClientAuth via Into) — both surfaces, one source.
        use crate::settings::{DigestAuth, SigV4Auth};
        let sigv4 = Client::builder()
            .http("http://x")
            .auth(
                SigV4Auth::builder()
                    .access_key_id("AKID")
                    .secret_access_key("sk")
                    .region("us-east-1")
                    .service("s3")
                    .build(),
            )
            .build()
            .expect("build");
        let smw = sigv4
            .inner
            .spec
            .get("middleware")
            .and_then(Value::as_array)
            .expect("middleware");
        assert_eq!(smw[0].get("scheme").and_then(Value::as_str), Some("sigv4"));
        assert_eq!(
            smw[0].get("region").and_then(Value::as_str),
            Some("us-east-1")
        );
        assert_eq!(smw[0].get("service").and_then(Value::as_str), Some("s3"));

        let digest = Client::builder()
            .http("http://x")
            .auth(
                DigestAuth::builder()
                    .username("Mufasa")
                    .password("Circle of Life")
                    .build(),
            )
            .build()
            .expect("build");
        let dmw = digest
            .inner
            .spec
            .get("middleware")
            .and_then(Value::as_array)
            .expect("middleware");
        assert_eq!(dmw[0].get("scheme").and_then(Value::as_str), Some("digest"));
        assert_eq!(
            dmw[0].get("username").and_then(Value::as_str),
            Some("Mufasa")
        );
    }

    /// The existing `Client` requests-style API, driven on the prime
    /// backend: a loopback h1 server (tokio + std thread) and the
    /// `Client` resolved through the prime `PrimeHttpPipeFactory`
    /// (registered for `"http"` under `http-prime`). Proves the same
    /// `Client::from_value` / `.call().send()` surface dispatches over
    /// the prime stack — no `HttpClient`, no new client type.
    #[cfg(all(feature = "http-prime", feature = "runtime-prime", feature = "tokio"))]
    #[test]
    fn client_get_roundtrip_over_prime_backend() {
        use std::net::{Ipv4Addr, SocketAddr};
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Mutex};
        use std::time::{Duration, Instant};

        use crate::CoreId;
        use prime::os::core_shard;

        let port_slot: Arc<Mutex<Option<u16>>> = Arc::new(Mutex::new(None));
        let port_for_server = port_slot.clone();
        let server_ready = Arc::new(AtomicBool::new(false));
        let server_ready_clone = server_ready.clone();

        let server_thread = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("server runtime");
            runtime.block_on(async move {
                use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
                let listener =
                    tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
                        .await
                        .expect("bind");
                let local = listener.local_addr().expect("local_addr");
                *port_for_server.lock().expect("port lock") = Some(local.port());
                server_ready_clone.store(true, Ordering::Release);

                let (mut socket, _) = listener.accept().await.expect("accept");
                let mut scratch = [0_u8; 2048];
                let mut buffer = Vec::new();
                loop {
                    let read = socket.read(&mut scratch).await.expect("read req");
                    buffer.extend_from_slice(&scratch[..read]);
                    if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                assert!(
                    buffer.starts_with(b"GET /path HTTP/1.1\r\n"),
                    "request line: {buffer:?}"
                );
                socket
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello")
                    .await
                    .expect("write resp");
                socket.flush().await.expect("flush resp");
            });
        });

        let deadline = Instant::now() + Duration::from_secs(2);
        while !server_ready.load(Ordering::Acquire) {
            assert!(Instant::now() < deadline, "server never bound");
            std::thread::sleep(Duration::from_millis(5));
        }
        let port = port_slot.lock().expect("port lock").expect("port set");

        let handle = core_shard::launch_with_lanes(CoreId(0), None, 2, 16).expect("launch");
        let done = Arc::new(AtomicBool::new(false));
        let done_clone = done.clone();
        let result_slot: Arc<Mutex<Option<(u16, bytes::Bytes)>>> = Arc::new(Mutex::new(None));
        let result_for_factory = result_slot.clone();

        handle
            .dispatch_factory(Box::new(move || {
                let done = done_clone.clone();
                let result_handle = result_for_factory.clone();
                Box::pin(async move {
                    let client = Client::from_value(json!({
                        "http": format!("http://127.0.0.1:{port}"),
                    }))
                    .expect("build client");
                    let response = client.call("GET", "/path").send().await.expect("send");
                    let status = response.status();
                    let body = response.bytes().await.expect("bytes");
                    *result_handle.lock().expect("result lock") = Some((status, body));
                    done.store(true, Ordering::Release);
                })
                    as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'static>>
            }))
            .expect("dispatch_factory");

        let deadline = Instant::now() + Duration::from_secs(5);
        while !done.load(Ordering::Acquire) {
            assert!(
                Instant::now() < deadline,
                "client-on-prime roundtrip never completed"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        handle.shutdown_and_join().expect("shutdown");
        server_thread.join().expect("server join");

        let (status, body) = result_slot
            .lock()
            .expect("result lock")
            .take()
            .expect("result set");
        assert_eq!(status, 200);
        assert_eq!(&body[..], b"hello");
    }
}
