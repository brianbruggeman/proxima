use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use bytes::Bytes;
use serde_json::Value;

use crate::codec_factory::{
    BytesPassthroughCodecFactory, CodecRegistry, DynCodecFactory, JsonCodecFactory,
};
use crate::config_format::{ConfigFormatRegistry, default_config_format_registry};
use crate::error::ProximaError;
use crate::log_buffer::LogBufferRegistry;
use crate::middlewares::auth::AuthFactory;
use crate::middlewares::client_auth::ClientAuthFactory;
use crate::middlewares::isolate::IsolateFactory;
use crate::middlewares::rate_limit::RateLimitFactory;
use crate::middlewares::retry::RetryFactory;
use proxima_primitives::pipe::SendPipe;

use crate::middlewares::transform::TransformFactory;
use crate::middlewares::validate::ValidateFactory;
use crate::middlewares::write_back::{WriteBack, WriteBackTarget};
use crate::pipe::{PipeHandle, into_handle};
use crate::pipe_factory::PipeFactoryRegistry;
use crate::recording::factory::RecordingSourceRegistry;
use crate::recording::{BinSourceFactory, JsonlSourceFactory};
use crate::request::{Request, Response};
use crate::schema::SchemaRegistry;
#[cfg(feature = "http1")]
use crate::shared_http::SharedHttpClient;
use crate::telemetry::{Metrics, NoopTelemetry, TelemetryHandle};

/// The shared hyper client pool, when the `http1` feature (hyper-backed)
/// is built; a zero-sized placeholder otherwise. `LoadContext::http_client`
/// carries this so the struct shape doesn't fork on the feature — nothing
/// downstream reads it unless `http1` is also on (see
/// `default_pipe_factory_registry`'s hyper-gated registration arms).
#[cfg(feature = "http1")]
pub type HttpClientHandle = SharedHttpClient;
#[cfg(not(feature = "http1"))]
pub type HttpClientHandle = ();
use crate::upstreams::callback::CallbackPipeFactory;
use crate::upstreams::fs::FsPipeFactory;
// HttpPipeFactory (hyper backend) is referenced only by the `http-hyper`
// use-sites in default_pipe_factory_registry; gate the import to match, so a
// prime-runtime build without an http backend does not see it as unused.
#[cfg(feature = "http-hyper")]
use crate::upstreams::http::HttpPipeFactory;
use crate::upstreams::kv_cache::{KvCacheFactory, build_kv_cache};
use crate::upstreams::kv_file::{KvFileFactory, build_kv_file};
use crate::upstreams::kv_upstream::KvUpstream;
#[cfg(feature = "tokio")]
use crate::upstreams::process::ProcessPipeFactory;
#[cfg(feature = "tokio")]
use crate::upstreams::process_rpc::ProcessRpcPipeFactory;
use crate::upstreams::record::RecordPipeFactory;
use crate::upstreams::replay::ReplayPipeFactory;
#[cfg(any(feature = "tcp", feature = "unix"))]
use crate::upstreams::stream_passthrough::StreamPassthroughPipeFactory;
use crate::upstreams::synth::SynthPipeFactory;
use proxima_patterns::kv::KvHandle;

/// Registers an existing factory under a second name. Used to expose the hyper
/// `"http"` backend as the selectable `"http-tokio"` wire alongside the prime
/// default, so a `"wire":"tokio"` upstream resolves to hyper in a both-wires build.
#[cfg(all(
    feature = "http-hyper",
    unix,
    feature = "http-prime-deps",
    feature = "runtime-prime"
))]
struct AliasFactory {
    alias: &'static str,
    inner: Arc<dyn crate::pipe_factory::PipeFactory>,
}

#[cfg(all(
    feature = "http-hyper",
    unix,
    feature = "http-prime-deps",
    feature = "runtime-prime"
))]
impl crate::pipe_factory::PipeFactory for AliasFactory {
    fn name(&self) -> &str {
        self.alias
    }

    fn build(
        &self,
        spec: &Value,
        inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        self.inner.build(spec, inner)
    }
}
use proxima_patterns::balancer::selection::{
    Fallthrough, LeastConn, MissPolicy, RoundRobin, SelectionHandle, WeightedLeastConn,
    WeightedRoundRobin,
};
use proxima_patterns::balancer::upstream_ref::UpstreamRef;

#[derive(Clone)]
pub struct LoadContext {
    pub registry: Arc<PipeFactoryRegistry>,
    /// Producer (source) factories keyed by `type`, the `SourceFactory`
    /// analogue of `registry`. Empty by default — an embedding application
    /// registers `IntervalSourceFactory`/`ScheduledTriggerSourceFactory`/its
    /// own `SourceFactory` impls the same way it would register a
    /// `PipeFactory`.
    pub source_registry: Arc<proxima_primitives::pipe::SourceFactoryRegistry>,
    /// Shared spigot armed at serve; the `record` upstream's durable sink
    /// stays inert until then (C7 spigot model).
    pub recording_spigot: crate::recording::DeferredRuntime,
    pub recording_source_registry: Arc<RecordingSourceRegistry>,
    pub codec_registry: Arc<CodecRegistry>,
    pub config_formats: Arc<ConfigFormatRegistry>,
    pub schemas: Arc<SchemaRegistry>,
    pub log_buffers: Arc<LogBufferRegistry>,
    pub telemetry: TelemetryHandle,
    pub metrics: Option<Arc<Metrics>>,
    pub http_client: HttpClientHandle,
}

#[cfg(feature = "http1")]
pub(crate) fn new_http_client_handle() -> HttpClientHandle {
    SharedHttpClient::new()
}
#[cfg(not(feature = "http1"))]
pub(crate) fn new_http_client_handle() -> HttpClientHandle {}

impl LoadContext {
    pub fn with_default_registry() -> Result<Self, ProximaError> {
        // `HttpClientHandle` is `()` under the default (no `http1`) build —
        // clippy only sees one arm per build, so this looks like a
        // pointless unit binding there; it's `SharedHttpClient` under `http1`.
        #[allow(clippy::let_unit_value)]
        let http_client = new_http_client_handle();
        let log_buffers = Arc::new(LogBufferRegistry::new());
        let recording_spigot = crate::recording::deferred_runtime();
        let recording_source_registry = Arc::new(default_recording_source_registry()?);
        let codec_registry = Arc::new(default_codec_registry()?);
        let config_formats = Arc::new(default_config_format_registry()?);
        let schemas = Arc::new(SchemaRegistry::new());
        crate::schema::register_scenario_schemas(&schemas)?;
        let registry = Arc::new(default_pipe_factory_registry(
            &http_client,
            recording_source_registry.clone(),
            log_buffers.clone(),
            schemas.clone(),
        )?);
        // record upstream needs a Weak<PipeFactoryRegistry> to recursively
        // resolve its `inner` spec; register after the Arc is constructed.
        registry.register(Arc::new(RecordPipeFactory::new(
            Arc::downgrade(&registry),
            recording_spigot.clone(),
        )))?;
        // client-auth's oauth scheme resolves its token-endpoint sub-pipe
        // through this same registry, so register it with a weak handle.
        registry.register(Arc::new(ClientAuthFactory::new(Arc::downgrade(&registry))))?;
        let metrics = Arc::new(Metrics::default());
        Ok(Self {
            registry,
            source_registry: Arc::new(proxima_primitives::pipe::SourceFactoryRegistry::new()),
            recording_spigot,
            recording_source_registry,
            codec_registry,
            config_formats,
            schemas,
            log_buffers,
            telemetry: metrics.clone(),
            metrics: Some(metrics),
            http_client,
        })
    }

    pub fn with_noop_telemetry() -> Result<Self, ProximaError> {
        // `HttpClientHandle` is `()` under the default (no `http1`) build —
        // clippy only sees one arm per build, so this looks like a
        // pointless unit binding there; it's `SharedHttpClient` under `http1`.
        #[allow(clippy::let_unit_value)]
        let http_client = new_http_client_handle();
        let log_buffers = Arc::new(LogBufferRegistry::new());
        let recording_spigot = crate::recording::deferred_runtime();
        let recording_source_registry = Arc::new(default_recording_source_registry()?);
        let codec_registry = Arc::new(default_codec_registry()?);
        let config_formats = Arc::new(default_config_format_registry()?);
        let schemas = Arc::new(SchemaRegistry::new());
        crate::schema::register_scenario_schemas(&schemas)?;
        let registry = Arc::new(default_pipe_factory_registry(
            &http_client,
            recording_source_registry.clone(),
            log_buffers.clone(),
            schemas.clone(),
        )?);
        registry.register(Arc::new(RecordPipeFactory::new(
            Arc::downgrade(&registry),
            recording_spigot.clone(),
        )))?;
        // client-auth's oauth scheme resolves its token-endpoint sub-pipe
        // through this same registry, so register it with a weak handle.
        registry.register(Arc::new(ClientAuthFactory::new(Arc::downgrade(&registry))))?;
        Ok(Self {
            registry,
            source_registry: Arc::new(proxima_primitives::pipe::SourceFactoryRegistry::new()),
            recording_spigot,
            recording_source_registry,
            codec_registry,
            config_formats,
            schemas,
            log_buffers,
            telemetry: Arc::new(NoopTelemetry),
            metrics: None,
            http_client,
        })
    }

    pub fn register_recording_source(
        &self,
        factory: crate::recording::factory::DynRecordingSourceFactory,
    ) -> Result<(), ProximaError> {
        self.recording_source_registry.register(factory)
    }

    pub fn register_codec(&self, factory: DynCodecFactory) -> Result<(), ProximaError> {
        self.codec_registry.register(factory)
    }
}

fn default_pipe_factory_registry(
    http_client: &HttpClientHandle,
    sources: Arc<RecordingSourceRegistry>,
    log_buffers: Arc<LogBufferRegistry>,
    schemas: Arc<SchemaRegistry>,
) -> Result<PipeFactoryRegistry, ProximaError> {
    let registry = PipeFactoryRegistry::new();
    // http_client feeds only the cfg-gated http backends below; mark it used so
    // a build with neither the prime nor the hyper http backend (e.g. a prime
    // runtime without http-prime-deps) does not see the param as unused.
    let _ = &http_client;
    // log_buffers only feeds the tokio-gated process upstream below; mark it
    // used so a tokio-free default build doesn't see the param as unused.
    let _ = &log_buffers;
    registry.register(Arc::new(KvCacheFactory))?;
    registry.register(Arc::new(KvFileFactory))?;
    // `http` upstream backend: PRIME is the default whenever the prime
    // runtime is active on unix — the prime `TcpStream` registers with the
    // per-worker reactor (`CURRENT_REACTOR`), so it can only be driven on a
    // prime CoreShard worker, not a plain tokio runtime. Hyper is the
    // backend otherwise: on the tokio runtime, on non-unix (proxima-net-prime
    // is macOS/Linux only), and whenever the `http-hyper` opt-out is on.
    // The `Client` requests-style API resolves its pipe through this
    // registry, so the backend swaps without touching the client surface.
    // prime is the default `"http"` backend whenever it is available.
    #[cfg(all(unix, feature = "http-prime-deps", feature = "runtime-prime"))]
    registry.register(Arc::new(proxima_http::http1::PrimeHttpPipeFactory::new()))?;
    // hyper backs `"http"` directly only when prime is unavailable.
    #[cfg(all(
        feature = "http-hyper",
        not(all(unix, feature = "http-prime-deps", feature = "runtime-prime"))
    ))]
    registry.register(Arc::new(HttpPipeFactory::with_shared_client(
        http_client.clone(),
    )))?;
    // when BOTH wires are built, hyper is the selectable tokio-compat wire under
    // `"http-tokio"`, chosen per-upstream via `"wire":"tokio"` — default stays prime.
    #[cfg(all(
        feature = "http-hyper",
        unix,
        feature = "http-prime-deps",
        feature = "runtime-prime"
    ))]
    registry.register(Arc::new(AliasFactory {
        alias: "http-tokio",
        inner: Arc::new(HttpPipeFactory::with_shared_client(http_client.clone())),
    }))?;
    registry.register(Arc::new(SynthPipeFactory))?;
    // `grpc` upstream backend: gRPC-over-HTTP/2 via the native h2 client over the
    // prime transport. Same prime-only availability as the prime `http` backend.
    #[cfg(all(
        feature = "http-prime",
        feature = "http2",
        any(target_os = "linux", target_os = "macos")
    ))]
    registry.register(Arc::new(crate::upstreams::grpc_h2::GrpcH2PipeFactory::new()))?;
    // `pgwire` protocol terminal: reached via `{"type":"pgwire", ...}` or the
    // `.pgwire(dsn)` builder sugar — the PostgreSQL client over prime TCP.
    #[cfg(all(
        feature = "pgwire-client",
        any(target_os = "linux", target_os = "macos")
    ))]
    registry.register(Arc::new(crate::upstreams::pgwire::PgwirePipeFactory::new()))?;
    // `redis` protocol terminal: reached via `{"type":"redis", ...}` or the
    // `.redis(dsn)` / `.valkey(dsn)` builder sugar — the Redis/Valkey client
    // over prime TCP.
    #[cfg(all(
        feature = "redis-client",
        any(target_os = "linux", target_os = "macos")
    ))]
    registry.register(Arc::new(crate::upstreams::redis::RedisPipeFactory::new()))?;
    // `h3-native` protocol terminal: reached via `{"type":"h3-native", ...}`
    // or the `.h3_native(url)` builder sugar — HTTP/3 over the native sans-IO
    // QUIC + H3 stack (prime UDP), the dual-surface peer of the quinn
    // `Http3Upstream` (P7).
    #[cfg(all(
        feature = "h3-native-upstream",
        any(target_os = "linux", target_os = "macos")
    ))]
    registry.register(Arc::new(
        crate::upstreams::h3_native::H3NativeUpstreamFactory::new(),
    ))?;
    registry.register(Arc::new(FsPipeFactory))?;
    registry.register(Arc::new(ReplayPipeFactory::new(sources)))?;
    registry.register(Arc::new(CallbackPipeFactory))?;
    #[cfg(any(feature = "tcp", feature = "unix"))]
    registry.register(Arc::new(StreamPassthroughPipeFactory::new()))?;
    #[cfg(feature = "tokio")]
    registry.register(Arc::new(ProcessPipeFactory::with_log_buffer_registry(
        log_buffers,
    )))?;
    #[cfg(feature = "tokio")]
    registry.register(Arc::new(ProcessRpcPipeFactory))?;
    registry.register(Arc::new(RetryFactory))?;
    registry.register(Arc::new(RateLimitFactory))?;
    registry.register(Arc::new(TransformFactory))?;
    registry.register(Arc::new(AuthFactory))?;
    // client-auth is registered in the context constructors below — its oauth
    // scheme needs the registry Arc to resolve the token-endpoint sub-pipe.
    registry.register(Arc::new(IsolateFactory))?;
    registry.register(Arc::new(ValidateFactory::new(schemas)))?;
    Ok(registry)
}

fn default_recording_source_registry() -> Result<RecordingSourceRegistry, ProximaError> {
    let registry = RecordingSourceRegistry::new();
    registry.register(Arc::new(JsonlSourceFactory))?;
    registry.register(Arc::new(BinSourceFactory))?;
    Ok(registry)
}

fn default_codec_registry() -> Result<CodecRegistry, ProximaError> {
    let registry = CodecRegistry::new();
    registry.register(Arc::new(JsonCodecFactory))?;
    registry.register(Arc::new(BytesPassthroughCodecFactory))?;
    Ok(registry)
}

impl std::fmt::Debug for LoadContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LoadContext")
            .field("registered", &self.registry.names())
            .field("metrics", &self.metrics.is_some())
            .finish()
    }
}

pub async fn load(
    spec: impl Into<Spec>,
    context: &LoadContext,
) -> Result<PipeHandle, ProximaError> {
    let resolved = spec.into().resolve(&context.config_formats)?;
    let (handle, _) = build_pipe(&resolved, context).await?;
    Ok(handle)
}

pub enum Spec {
    Path(std::path::PathBuf),
    Inline(Value),
    /// Inline source text. `format` of `None` triggers sniff (try-parse in
    /// registration order); `Some("toml" | "json" | "yaml" | ...)` hints
    /// the format directly. Use this instead of `Spec::Toml` for new code.
    Raw {
        content: String,
        format: Option<String>,
    },
    /// Equivalent to `Spec::Raw { content, format: Some("toml") }`. Kept
    /// for backwards compatibility.
    Toml(String),
    Handle(PipeHandle),
}

impl Spec {
    pub fn resolve(self, formats: &ConfigFormatRegistry) -> Result<ResolvedSpec, ProximaError> {
        match self {
            Self::Path(path) => Ok(ResolvedSpec::Value(load_value_from_path(&path, formats)?)),
            Self::Inline(value) => Ok(ResolvedSpec::Value(value)),
            Self::Raw { content, format } => Ok(ResolvedSpec::Value(
                formats.parse_with_hint(&content, format.as_deref())?,
            )),
            Self::Toml(text) => Ok(ResolvedSpec::Value(
                formats.parse_with_hint(&text, Some("toml"))?,
            )),
            Self::Handle(handle) => Ok(ResolvedSpec::Handle(handle)),
        }
    }
}

pub enum ResolvedSpec {
    Value(Value),
    Handle(PipeHandle),
}

impl ResolvedSpec {
    pub fn as_value_mut(&mut self) -> Option<&mut Value> {
        match self {
            Self::Value(value) => Some(value),
            Self::Handle(_) => None,
        }
    }
}

impl From<&str> for Spec {
    fn from(path: &str) -> Self {
        Self::Path(std::path::PathBuf::from(path))
    }
}

impl From<String> for Spec {
    fn from(path: String) -> Self {
        Self::Path(std::path::PathBuf::from(path))
    }
}

impl From<&Path> for Spec {
    fn from(path: &Path) -> Self {
        Self::Path(path.to_path_buf())
    }
}

impl From<std::path::PathBuf> for Spec {
    fn from(path: std::path::PathBuf) -> Self {
        Self::Path(path)
    }
}

impl From<Value> for Spec {
    fn from(value: Value) -> Self {
        Self::Inline(value)
    }
}

fn load_value_from_path(
    path: &Path,
    formats: &ConfigFormatRegistry,
) -> Result<Value, ProximaError> {
    let text = std::fs::read_to_string(path).map_err(ProximaError::Io)?;
    let extension = path.extension().and_then(|ext| ext.to_str());
    match extension {
        Some(ext) => match formats.get_by_extension(ext) {
            Ok(factory) => factory.parse(&text),
            // unknown extension — fall through to sniff so callers can
            // ship `.config`, `.cfg`, or no extension and still load.
            Err(_) => formats.parse_sniff(&text),
        },
        None => formats.parse_sniff(&text),
    }
}

#[derive(Clone)]
struct BuiltUpstream {
    handle: PipeHandle,
    kv_backend: Option<Arc<dyn KvHandle>>,
    label: String,
}

type BuildResult = Result<(PipeHandle, Option<Arc<dyn KvHandle>>), ProximaError>;

fn build_pipe<'context>(
    spec: &'context ResolvedSpec,
    context: &'context LoadContext,
) -> Pin<Box<dyn Future<Output = BuildResult> + Send + 'context>> {
    Box::pin(async move {
        let value = match spec {
            ResolvedSpec::Handle(handle) => return Ok((handle.clone(), None)),
            ResolvedSpec::Value(value) => value,
        };

        // gather the inner handle + optional kv backend per spec shape,
        // then apply the middleware stack uniformly. without this single
        // exit point, single-upstream specs like `{"http": ..., "retry":
        // ...}` would silently bypass the middleware; only the
        // `upstreams` array path used to attach middleware via
        // `build_composed`. that was a real bug — porcelain callers
        // assumed retry attached on http shorthand and didn't.
        let (handle, kv_backend): (PipeHandle, Option<Arc<dyn KvHandle>>) =
            if let Some(http) = value.get("http") {
                let canonical = canonical_http(http, value)?;
                // default prime; `"wire":"tokio"` selects the tokio-compat (hyper)
                // wire when both are built — compatibility for tokio-only upstreams.
                // falls back to `"http"` when the tokio wire isn't registered.
                let key = match value.get("wire").and_then(Value::as_str) {
                    Some("tokio") if context.registry.get("http-tokio").is_ok() => "http-tokio",
                    _ => "http",
                };
                let factory = context.registry.get(key)?;
                (factory.build(&canonical, None).await?, None)
            } else if let Some(grpc) = value.get("grpc") {
                // gRPC transport: same `{url} + name/timeout/headers` canonical
                // shape as http; the `grpc` factory stacks the h2 client.
                let canonical = canonical_http(grpc, value)?;
                let factory = context.registry.get("grpc")?;
                (factory.build(&canonical, None).await?, None)
            } else if let Some(synth) = value.get("synth") {
                let factory = context.registry.get("synth")?;
                (factory.build(synth, None).await?, None)
            } else if let Some(replay) = value.get("replay") {
                let factory = context.registry.get("replay")?;
                (factory.build(replay, None).await?, None)
            } else if let Some(callback) = value.get("callback") {
                let factory = context.registry.get("callback")?;
                (factory.build(callback, None).await?, None)
            } else if let Some(process) = value.get("process") {
                let factory = context.registry.get("process")?;
                let canonical = canonical_process(process, value)?;
                (factory.build(&canonical, None).await?, None)
            } else if let Some(rpc) = value.get("process_rpc") {
                let factory = context.registry.get("process_rpc")?;
                let canonical = canonical_process(rpc, value)?;
                (factory.build(&canonical, None).await?, None)
            } else if let Some(fs) = value.get("fs") {
                let factory = context.registry.get("fs")?;
                (factory.build(fs, None).await?, None)
            } else if let Some(kv) = value.get("kv") {
                let backend_name = kv.as_str().ok_or_else(|| {
                    ProximaError::Config("`kv` shorthand must be a string backend".into())
                })?;
                let factory_key = format!("kv:{backend_name}");
                let list_mode = value
                    .get("list_mode")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                if backend_name == "cache" {
                    let canonical = canonical_kv(kv, value)?;
                    let backend = build_kv_cache(&canonical)?;
                    let upstream = KvUpstream::new(backend.clone()).with_list_mode(list_mode);
                    let handle: PipeHandle = into_handle(upstream);
                    let kv_handle: Arc<dyn KvHandle> = backend;
                    (handle, Some(kv_handle))
                } else if backend_name == "file" {
                    let canonical = canonical_kv(kv, value)?;
                    let backend = build_kv_file(&canonical)?;
                    let upstream = KvUpstream::new(backend.clone()).with_list_mode(list_mode);
                    let handle: PipeHandle = into_handle(upstream);
                    let kv_handle: Arc<dyn KvHandle> = backend;
                    (handle, Some(kv_handle))
                } else {
                    let factory = context.registry.get(&factory_key)?;
                    let canonical = canonical_kv(kv, value)?;
                    (factory.build(&canonical, None).await?, None)
                }
            } else if value.get("upstreams").is_some() {
                // build_composed handles its own middleware-apply + write-back.
                // skip the post-process step below to avoid double-wrapping.
                let handle = build_composed(value, context).await?;
                return Ok((handle, None));
            } else if let Some(type_field) = value.get("type").and_then(Value::as_str) {
                let factory = context.registry.get(type_field)?;
                (factory.build(value, None).await?, None)
            } else {
                return Err(ProximaError::Config(
                    "spec must include `upstreams`, `http`, `kv`, or a `type` discriminator".into(),
                ));
            };

        let wrapped = apply_middleware_stack(handle, value, context).await?;
        Ok((wrapped, kv_backend))
    })
}

fn canonical_http(http_field: &Value, full: &Value) -> Result<Value, ProximaError> {
    let url = http_field
        .as_str()
        .ok_or_else(|| ProximaError::Config("`http` shorthand must be a string url".into()))?;
    let mut spec = serde_json::Map::new();
    spec.insert("url".into(), Value::String(url.into()));
    for forwarded in ["name", "timeout", "method", "headers", "proxy", "response"] {
        if let Some(value) = full.get(forwarded) {
            spec.insert(forwarded.into(), value.clone());
        }
    }
    Ok(Value::Object(spec))
}

fn canonical_process(process_field: &Value, full: &Value) -> Result<Value, ProximaError> {
    let mut spec = if let Some(table) = process_field.as_object() {
        table.clone()
    } else {
        return Err(ProximaError::Config(
            "`process` block must be a table (e.g. [pipe.process])".into(),
        ));
    };
    if let Some(name) = full.get("name").and_then(Value::as_str) {
        spec.entry("name".to_string())
            .or_insert_with(|| Value::String(name.to_string()));
    }
    Ok(Value::Object(spec))
}

fn canonical_kv(_kv_field: &Value, full: &Value) -> Result<Value, ProximaError> {
    let mut spec = serde_json::Map::new();
    for forwarded in [
        "ttl",
        "max_entries",
        "max_bytes",
        "name",
        "eviction",
        "url",
        "version",
        "path",
    ] {
        if let Some(value) = full.get(forwarded) {
            spec.insert(forwarded.into(), value.clone());
        }
    }
    Ok(Value::Object(spec))
}

async fn build_composed(spec: &Value, context: &LoadContext) -> Result<PipeHandle, ProximaError> {
    let upstream_specs = spec
        .get("upstreams")
        .and_then(Value::as_array)
        .ok_or_else(|| ProximaError::Config("`upstreams` must be an array".into()))?;
    let mut built: Vec<BuiltUpstream> = Vec::with_capacity(upstream_specs.len());
    for entry in upstream_specs {
        let label = entry
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("upstream-{}", built.len()));
        let (handle, kv_backend) = build_pipe(&ResolvedSpec::Value(entry.clone()), context).await?;
        built.push(BuiltUpstream {
            handle,
            kv_backend,
            label,
        });
    }

    let outer_label = spec
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("composed")
        .to_string();

    let upstream_refs: Vec<UpstreamRef> = built
        .iter()
        .zip(upstream_specs.iter())
        .map(|(item, spec_entry)| {
            let weight = spec_entry
                .get("weight")
                .and_then(Value::as_u64)
                .unwrap_or(1) as u32;
            UpstreamRef::new(item.handle.clone(), item.label.clone(), weight)
        })
        .collect();
    let selection = build_selection(spec.get("select"))?;
    let dispatch = DispatchPipe::new(upstream_refs, selection, &outer_label);

    let write_back_targets = parse_write_back_targets(spec, &built)?;
    let mut handle: PipeHandle = if write_back_targets.is_empty() {
        into_handle(dispatch)
    } else {
        let write_back = WriteBack::new(into_handle(dispatch), write_back_targets);
        into_handle(write_back)
    };

    handle = apply_middleware_stack(handle, spec, context).await?;

    // `outer_label` was previously re-attached via a `Labeled` wrapper so
    // callers could recover it through `Pipe::name()`; that method (and the
    // wrapper's sole purpose) is gone — the pipe registry key IS the name
    // now (TARGET 3), so `outer_label`'s only remaining job is the
    // `DispatchPipe` label above.
    Ok(handle)
}

/// Wraps `handle` with every middleware declared on `spec`, in the canonical
/// outer-to-inner: rate_limit (gates ingress) > transform > retry >
/// `[[middleware]]` entries in declaration order > write_back. Plugin
/// middleware only loads via the `[[middleware]]` array.
async fn apply_middleware_stack(
    mut handle: PipeHandle,
    spec: &Value,
    context: &LoadContext,
) -> Result<PipeHandle, ProximaError> {
    if let Some(retry_spec) = spec.get("retry") {
        let factory = context.registry.get("retry")?;
        handle = factory.build(retry_spec, Some(handle)).await?;
    }
    if let Some(transform_spec) = spec.get("transform") {
        let factory = context.registry.get("transform")?;
        handle = factory.build(transform_spec, Some(handle)).await?;
    }
    if let Some(rate_limit_spec) = spec.get("rate_limit") {
        let factory = context.registry.get("rate_limit")?;
        handle = factory.build(rate_limit_spec, Some(handle)).await?;
    }
    if let Some(middlewares) = spec.get("middleware").and_then(Value::as_array) {
        for entry in middlewares.iter().rev() {
            let mw_type = entry.get("type").and_then(Value::as_str).ok_or_else(|| {
                ProximaError::Config("[[middleware]] entries require `type`".into())
            })?;
            let factory = context.registry.get(mw_type)?;
            handle = factory.build(entry, Some(handle)).await?;
        }
    }
    Ok(handle)
}

fn build_selection(spec: Option<&Value>) -> Result<SelectionHandle, ProximaError> {
    let Some(value) = spec else {
        return Ok(Arc::new(Fallthrough::miss_on_no_data()));
    };
    let algorithm = value
        .get("algorithm")
        .and_then(Value::as_str)
        .unwrap_or("fallthrough");
    match algorithm {
        "fallthrough" => {
            let policy = parse_miss_policy(value)?;
            Ok(Arc::new(Fallthrough::new(policy)))
        }
        "round_robin" => Ok(Arc::new(RoundRobin::default())),
        "least_conn" => Ok(Arc::new(LeastConn)),
        "weighted_round_robin" => Ok(Arc::new(WeightedRoundRobin::default())),
        "weighted_least_connections" | "weighted_least_conn" => Ok(Arc::new(WeightedLeastConn)),
        other => Err(ProximaError::Config(format!(
            "unknown selection algorithm '{other}'"
        ))),
    }
}

fn parse_miss_policy(spec: &Value) -> Result<MissPolicy, ProximaError> {
    let mut policy = MissPolicy::fallthrough_default();
    if let Some(miss_on) = spec.get("miss_on").and_then(Value::as_array) {
        policy.on_no_data = false;
        for entry in miss_on {
            let text = entry
                .as_str()
                .ok_or_else(|| ProximaError::Config("`miss_on` entries must be strings".into()))?;
            match text {
                "no_data" => policy.on_no_data = true,
                "5xx" => {
                    for status in 500..=599 {
                        policy.on_status.push(status);
                    }
                }
                "404" => policy.on_status.push(404),
                "timeout" | "connection_refused" | "connection_reset" => policy.on_error = true,
                other => {
                    if let Ok(status) = other.parse::<u16>() {
                        policy.on_status.push(status);
                        continue;
                    }
                    return Err(ProximaError::Config(format!(
                        "unknown miss_on value '{other}'"
                    )));
                }
            }
        }
    }
    Ok(policy)
}

fn parse_write_back_targets(
    spec: &Value,
    upstreams: &[BuiltUpstream],
) -> Result<Vec<WriteBackTarget>, ProximaError> {
    let mut targets = Vec::new();
    let Some(write_back) = spec.get("write_back").and_then(Value::as_array) else {
        return Ok(targets);
    };
    for entry in write_back {
        let target = parse_write_back_entry(entry, upstreams)?;
        targets.push(target);
    }
    Ok(targets)
}

fn parse_write_back_entry(
    entry: &Value,
    upstreams: &[BuiltUpstream],
) -> Result<WriteBackTarget, ProximaError> {
    let array = entry.as_array().ok_or_else(|| {
        ProximaError::Config("write_back entries must be `[from_index, to_index]` arrays".into())
    })?;
    if array.len() != 2 {
        return Err(ProximaError::Config(
            "write_back entries must have exactly two elements".into(),
        ));
    }
    let from_index = resolve_upstream_index(&array[0], upstreams)?;
    let to_index = resolve_upstream_index(&array[1], upstreams)?;
    let _ = from_index;
    let target = upstreams
        .get(to_index)
        .ok_or_else(|| ProximaError::Config("write_back index out of range".into()))?;
    let backend = target.kv_backend.clone().ok_or_else(|| {
        ProximaError::Config("write_back target must be a kv-typed upstream".into())
    })?;
    Ok(WriteBackTarget::new(backend, target.label.clone()))
}

fn resolve_upstream_index(
    value: &Value,
    upstreams: &[BuiltUpstream],
) -> Result<usize, ProximaError> {
    if let Some(index) = value.as_u64() {
        return Ok(index as usize);
    }
    if let Some(name) = value.as_str()
        && let Some(position) = upstreams.iter().position(|item| item.label == name)
    {
        return Ok(position);
    }
    Err(ProximaError::Config(format!(
        "write_back reference '{value}' must be an integer index or a known upstream name"
    )))
}

struct DispatchPipe {
    upstreams: Vec<UpstreamRef>,
    selection: SelectionHandle,
}

impl DispatchPipe {
    // `_parent_label` is no longer stored (TARGET 3 — served-Pipe naming
    // now lives at the mount-site label, not the handle); kept as a
    // parameter so call sites read the same at the construction site.
    fn new(upstreams: Vec<UpstreamRef>, selection: SelectionHandle, _parent_label: &str) -> Self {
        Self {
            upstreams,
            selection,
        }
    }
}

impl SendPipe for DispatchPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let selection = self.selection.clone();
        let upstreams = self.upstreams.clone();
        let telemetry = request.context.telemetry.clone();
        let context_labels = request.context.metric_labels(&[]);
        let deadline = request.context.deadline;
        async move {
            let dispatch_future = async {
                let outcome = selection.dispatch_dyn(&upstreams, request).await?;
                for reason in &outcome.fallthroughs {
                    let labels = with_extra(&context_labels, "reason", &format!("{reason:?}"));
                    telemetry.counter_inc("proxima.selection.fallthroughs_total", &labels, 1);
                }
                Ok::<_, ProximaError>(outcome.response)
            };
            match deadline {
                Some(deadline) => {
                    let now = proxima_core::time::now();
                    if deadline <= now {
                        return Err(ProximaError::Timeout(std::time::Duration::ZERO));
                    }
                    let remaining = deadline - now;
                    match proxima_core::time::timeout(remaining, dispatch_future).await {
                        Ok(result) => result,
                        Err(_) => Err(ProximaError::Timeout(remaining)),
                    }
                }
                None => dispatch_future.await,
            }
        }
    }
}


fn with_extra(base: &crate::telemetry::Labels, key: &str, value: &str) -> crate::telemetry::Labels {
    let mut pairs: Vec<(String, String)> = base.entries().to_vec();
    pairs.push((key.to_string(), value.to_string()));
    let pair_refs: Vec<(&str, &str)> = pairs
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .collect();
    crate::telemetry::Labels::from_pairs(&pair_refs)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[proxima::test]
    async fn load_inline_kv_cache_returns_kv_upstream_handle() {
        let context = LoadContext::with_default_registry().expect("ctx");
        let handle = load(
            json!({"kv": "cache", "ttl": "1h", "max_entries": 100, "name": "echo-cache"}),
            &context,
        )
        .await
        .expect("load");
        // the raw handle no longer carries a queryable name (TARGET 3);
        // a successful load is the behavioral proof.
        let _ = &handle;
    }

    // matches default_pipe_factory_registry's "http" factory availability:
    // prime on unix, or hyper as the fallback wire.
    #[cfg(any(
        all(unix, feature = "http-prime-deps", feature = "runtime-prime"),
        feature = "http-hyper"
    ))]
    #[proxima::test]
    async fn load_inline_http_returns_named_handle() {
        let context = LoadContext::with_default_registry().expect("ctx");
        let handle = load(
            json!({"http": "http://example.test", "name": "echo"}),
            &context,
        )
        .await
        .expect("load");
        // see load_inline_kv_cache_returns_kv_upstream_handle above.
        let _ = &handle;
    }

    #[cfg(any(
        all(unix, feature = "http-prime-deps", feature = "runtime-prime"),
        feature = "http-hyper"
    ))]
    #[proxima::test]
    async fn load_composed_with_cache_then_origin_returns_dispatch_pipe() {
        let context = LoadContext::with_default_registry().expect("ctx");
        let spec = json!({
            "name": "cached_test",
            "upstreams": [
                {"kv": "cache", "ttl": "1h", "max_entries": 100, "name": "cache"},
                {"http": "http://example.test", "name": "origin"},
            ],
            "select": {"algorithm": "fallthrough", "miss_on": ["no_data"]},
            "write_back": [["origin", "cache"]],
        });
        let handle = load(spec, &context).await.expect("load");
        // see load_inline_kv_cache_returns_kv_upstream_handle above.
        let _ = &handle;
    }

    #[cfg(any(
        all(unix, feature = "http-prime-deps", feature = "runtime-prime"),
        feature = "http-hyper"
    ))]
    #[proxima::test]
    async fn write_back_index_out_of_range_errors() {
        let context = LoadContext::with_default_registry().expect("ctx");
        let spec = json!({
            "upstreams": [{"kv": "cache", "max_entries": 100}, {"http": "http://example.test"}],
            "write_back": [[0, 5]],
        });
        let outcome = load(spec, &context).await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[cfg(any(
        all(unix, feature = "http-prime-deps", feature = "runtime-prime"),
        feature = "http-hyper"
    ))]
    #[proxima::test]
    async fn write_back_to_non_kv_target_errors() {
        let context = LoadContext::with_default_registry().expect("ctx");
        let spec = json!({
            "upstreams": [{"http": "http://a.test"}, {"http": "http://b.test"}],
            "write_back": [[0, 1]],
        });
        let outcome = load(spec, &context).await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[proxima::test]
    async fn declarative_middleware_array_dispatches_through_registry() {
        use crate::pipe::{PipeHandle, into_handle};
        use crate::pipe_factory::PipeFactory;
        use crate::request::{Request, Response};
        use std::pin::Pin;
        use std::sync::Arc;

        struct StampHeader {
            name: String,
        }

        impl PipeFactory for StampHeader {
            fn name(&self) -> &str {
                &self.name
            }

            fn build(
                &self,
                spec: &Value,
                inner: Option<PipeHandle>,
            ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>>
            {
                let header_name = spec
                    .get("header")
                    .and_then(Value::as_str)
                    .unwrap_or("x-stamp")
                    .to_string();
                let header_value = spec
                    .get("value")
                    .and_then(Value::as_str)
                    .unwrap_or("set")
                    .to_string();
                Box::pin(async move {
                    let inner = inner.ok_or_else(|| {
                        ProximaError::Config("stamp requires an inner pipe".into())
                    })?;
                    struct Stamper {
                        inner: PipeHandle,
                        header_name: String,
                        header_value: String,
                    }
                    impl SendPipe for Stamper {
                        type In = Request<Bytes>;
                        type Out = Response<Bytes>;
                        type Err = ProximaError;

                        fn call(
                            &self,
                            request: Request<Bytes>,
                        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>>
                        {
                            let inner = self.inner.clone();
                            let header_name = self.header_name.clone();
                            let header_value = self.header_value.clone();
                            async move {
                                let response = SendPipe::call(&inner, request).await?;
                                Ok(response.with_header(header_name, header_value))
                            }
                        }
                    }
                    Ok(into_handle(Stamper {
                        inner,
                        header_name,
                        header_value,
                    }))
                })
            }
        }

        let context = LoadContext::with_default_registry().expect("ctx");
        context
            .registry
            .register(Arc::new(StampHeader {
                name: "stamp".into(),
            }))
            .expect("register stamp");
        let spec = json!({
            "name": "stamped",
            "upstreams": [{"synth": {"status": 200, "body": "ok"}}],
            "middleware": [
                {"type": "stamp", "header": "x-outer", "value": "outer"},
                {"type": "stamp", "header": "x-inner", "value": "inner"},
            ],
        });
        let handle = load(spec, &context).await.expect("load");
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("request");
        let response = SendPipe::call(&handle, request).await.expect("call");
        assert_eq!(response.metadata.get_str("x-outer"), Some("outer"));
        assert_eq!(response.metadata.get_str("x-inner"), Some("inner"));
    }

    #[proxima::test]
    async fn unknown_middleware_type_returns_registry_error() {
        let context = LoadContext::with_default_registry().expect("ctx");
        let spec = json!({
            "name": "broken",
            "upstreams": [{"synth": {"status": 200, "body": "ok"}}],
            "middleware": [{"type": "definitely-not-registered"}],
        });
        let outcome = load(spec, &context).await;
        assert!(matches!(outcome, Err(ProximaError::Registry(_))));
    }

    #[cfg(any(
        all(unix, feature = "http-prime-deps", feature = "runtime-prime"),
        feature = "http-hyper"
    ))]
    #[proxima::test]
    async fn write_back_by_name_resolves_via_upstream_label() {
        let context = LoadContext::with_default_registry().expect("ctx");
        let spec = json!({
            "name": "by-name",
            "upstreams": [
                {"kv": "cache", "max_entries": 100, "name": "hot"},
                {"http": "http://example.test", "name": "origin"},
            ],
            "select": {"algorithm": "fallthrough"},
            "write_back": [["origin", "hot"]],
        });
        let handle = load(spec, &context).await.expect("load");
        // see load_inline_kv_cache_returns_kv_upstream_handle above.
        let _ = &handle;
    }
}
