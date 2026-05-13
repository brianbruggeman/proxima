use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::app::App;
use crate::codec_factory::{
    BytesPassthroughCodecFactory, CodecRegistry, DynCodecFactory, JsonCodecFactory,
};
use crate::config_format::{ConfigFormatRegistry, default_config_format_registry};
use crate::error::ProximaError;
#[cfg(feature = "http1")]
use crate::listeners::http::HttpListenProtocol;
#[cfg(feature = "tokio")]
use crate::listeners::mcp::McpListenProtocol;
use crate::load::LoadContext;
use crate::log_buffer::LogBufferRegistry;
use crate::middlewares::auth::AuthFactory;
use crate::middlewares::isolate::IsolateFactory;
use crate::middlewares::rate_limit::RateLimitFactory;
use crate::middlewares::retry::RetryFactory;
use crate::middlewares::transform::TransformFactory;
use crate::middlewares::validate::ValidateFactory;
use crate::mount::Router;
use crate::pipe_factory::{DynPipeFactory, PipeFactoryRegistry};
use crate::recording::factory::{DynRecordingSourceFactory, RecordingSourceRegistry};
use crate::recording::{BinSourceFactory, JsonlSourceFactory};
use crate::load::HttpClientHandle;
use crate::schema::SchemaRegistry;
#[cfg(feature = "http1")]
use crate::shared_http::SharedHttpClient;
use crate::telemetry::{Metrics, NoopTelemetry, TelemetryHandle};
use crate::upstreams::callback::CallbackPipeFactory;
#[cfg(any(
    not(all(unix, feature = "http-prime-deps", feature = "runtime-prime")),
    feature = "http-hyper"
))]
use crate::upstreams::http::HttpPipeFactory;
use crate::upstreams::kv_cache::KvCacheFactory;
use crate::upstreams::kv_file::KvFileFactory;
#[cfg(feature = "tokio")]
use crate::upstreams::process::ProcessPipeFactory;
#[cfg(feature = "tokio")]
use crate::upstreams::process_rpc::ProcessRpcPipeFactory;
use crate::upstreams::record::RecordPipeFactory;
use crate::upstreams::replay::ReplayPipeFactory;
#[cfg(any(feature = "tcp", feature = "unix"))]
use crate::upstreams::stream_passthrough::StreamPassthroughPipeFactory;
use crate::upstreams::synth::SynthPipeFactory;
use proxima_listen::{ListenProtocol, ListenRegistry};

/// Compose an `App` from built-in and plugin-registered factories.
/// Use `with_defaults()` for the full substrate; skip it to start clean.
pub struct AppBuilder {
    listen_registry: ListenRegistry,
    pipe_factory_registry: PipeFactoryRegistry,
    recording_spigot: crate::recording::DeferredRuntime,
    recording_source_registry: RecordingSourceRegistry,
    codec_registry: CodecRegistry,
    config_formats: ConfigFormatRegistry,
    schemas: SchemaRegistry,
    log_buffers: Arc<LogBufferRegistry>,
    telemetry: Option<TelemetryHandle>,
    metrics: Option<Arc<Metrics>>,
    http_client: Option<HttpClientHandle>,
    runtime_config: Option<crate::app_config::RuntimeConfig>,
    defaults_replay: bool,
    defaults_process: bool,
    defaults_record: bool,
    defaults_validate: bool,
}

impl Default for AppBuilder {
    fn default() -> Self {
        Self {
            listen_registry: ListenRegistry::new(),
            pipe_factory_registry: PipeFactoryRegistry::new(),
            recording_spigot: crate::recording::deferred_runtime(),
            recording_source_registry: RecordingSourceRegistry::new(),
            codec_registry: CodecRegistry::new(),
            config_formats: ConfigFormatRegistry::new(),
            schemas: SchemaRegistry::new(),
            log_buffers: Arc::new(LogBufferRegistry::new()),
            telemetry: None,
            metrics: None,
            http_client: None,
            runtime_config: None,
            defaults_replay: false,
            defaults_process: false,
            defaults_record: false,
            defaults_validate: false,
        }
    }
}

impl AppBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_defaults(mut self) -> Result<Self, ProximaError> {
        // listen protocols
        #[cfg(feature = "http1")]
        self.listen_registry
            .register(Arc::new(HttpListenProtocol::new()))?;
        #[cfg(feature = "tokio")]
        self.listen_registry
            .register(Arc::new(McpListenProtocol::new()))?;
        #[cfg(feature = "tcp")]
        self.listen_registry
            .register(Arc::new(crate::listeners::StreamListenProtocol::new()))?;

        // `HttpClientHandle` is `SharedHttpClient` (not `Copy`, needs `clone()`)
        // under `http1`, `()` (a unit let-binding, `Copy`) otherwise — clippy
        // only sees one arm per build, so both lints are feature-conditional.
        #[allow(clippy::clone_on_copy, clippy::let_unit_value)]
        let http_client = self.http_client.clone().unwrap_or_default();

        // recording first — replay holds an Arc<RecordingSourceRegistry>
        // and needs every source registered before it's built. (Sinks are
        // built directly from the spigot now — no sink registry.)
        self.recording_source_registry
            .register(Arc::new(JsonlSourceFactory))?;
        self.recording_source_registry
            .register(Arc::new(BinSourceFactory))?;

        // replay + process + record register in build() once registries
        // are finalized into Arcs.
        self.pipe_factory_registry
            .register(Arc::new(KvCacheFactory))?;
        self.pipe_factory_registry
            .register(Arc::new(KvFileFactory))?;
        // `http` upstream backend: PRIME is the default whenever the prime
        // runtime is active on unix (the prime `TcpStream` only drives on a
        // CoreShard worker's reactor); hyper is the backend on the tokio
        // runtime, on non-unix, and under the `http-hyper` opt-out. The
        // `Client` requests-style API resolves through this registry, so the
        // swap is backend-only.
        #[cfg(all(
            unix,
            feature = "http-prime-deps",
            feature = "runtime-prime",
            not(feature = "http-hyper")
        ))]
        self.pipe_factory_registry
            .register(Arc::new(proxima_http::http1::PrimeHttpPipeFactory::new()))?;
        #[cfg(any(
            not(all(unix, feature = "http-prime-deps", feature = "runtime-prime")),
            feature = "http-hyper"
        ))]
        self.pipe_factory_registry
            .register(Arc::new(HttpPipeFactory::with_shared_client(
                http_client.clone(),
            )))?;
        self.pipe_factory_registry
            .register(Arc::new(SynthPipeFactory))?;
        self.pipe_factory_registry
            .register(Arc::new(CallbackPipeFactory))?;
        #[cfg(any(feature = "tcp", feature = "unix"))]
        self.pipe_factory_registry
            .register(Arc::new(StreamPassthroughPipeFactory::new()))?;
        #[cfg(feature = "tokio")]
        self.pipe_factory_registry
            .register(Arc::new(ProcessRpcPipeFactory))?;
        self.defaults_replay = true;
        self.defaults_process = true;
        self.defaults_record = true;

        self.pipe_factory_registry
            .register(Arc::new(RetryFactory))?;
        self.pipe_factory_registry
            .register(Arc::new(RateLimitFactory))?;
        self.pipe_factory_registry
            .register(Arc::new(TransformFactory))?;
        self.pipe_factory_registry.register(Arc::new(AuthFactory))?;
        self.pipe_factory_registry
            .register(Arc::new(IsolateFactory))?;
        self.defaults_validate = true;

        self.codec_registry.register(Arc::new(JsonCodecFactory))?;
        self.codec_registry
            .register(Arc::new(BytesPassthroughCodecFactory))?;

        self.config_formats = default_config_format_registry()?;

        self.http_client = Some(http_client);
        Ok(self)
    }

    pub fn with_listen_protocol(
        self,
        protocol: Arc<dyn ListenProtocol>,
    ) -> Result<Self, ProximaError> {
        self.listen_registry.register(protocol)?;
        Ok(self)
    }

    pub fn with_upstream_factory(self, factory: DynPipeFactory) -> Result<Self, ProximaError> {
        self.pipe_factory_registry.register(factory)?;
        Ok(self)
    }
}

impl proxima_primitives::pipe::plugin::PluginRegistry for AppBuilder {
    fn with_upstream_factory(self, factory: DynPipeFactory) -> Result<Self, ProximaError> {
        AppBuilder::with_upstream_factory(self, factory)
    }
}

impl AppBuilder {
    pub fn with_recording_source_factory(
        self,
        factory: DynRecordingSourceFactory,
    ) -> Result<Self, ProximaError> {
        self.recording_source_registry.register(factory)?;
        Ok(self)
    }

    pub fn with_codec_factory(self, factory: DynCodecFactory) -> Result<Self, ProximaError> {
        self.codec_registry.register(factory)?;
        Ok(self)
    }

    #[must_use]
    pub fn with_telemetry(mut self, telemetry: TelemetryHandle) -> Self {
        self.telemetry = Some(telemetry);
        self
    }

    #[must_use]
    pub fn with_metrics(mut self, metrics: Arc<Metrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    #[cfg(feature = "http1")]
    #[must_use]
    pub fn with_http_client(mut self, client: SharedHttpClient) -> Self {
        self.http_client = Some(client);
        self
    }

    /// Size the App's fallback default runtime explicitly — no env var, no
    /// build-and-discard. Takes effect only when no runtime is already
    /// installed ambiently (a `#[proxima::main]`-booted runtime always wins;
    /// see `crate::runtime::installed_runtime`); otherwise this is the
    /// config `App::new()`'s raw `std::env::var("PROXIMA_RUNTIME_CORES")`
    /// read used to require a global env mutation to influence.
    ///
    /// ```no_run
    /// # use proxima::App;
    /// # use proxima::app_config::RuntimeConfig;
    /// let app = App::builder()
    ///     .with_runtime_config(RuntimeConfig::builder().cores(1).build())
    ///     .with_defaults()?
    ///     .build()?;
    /// # Ok::<(), proxima::ProximaError>(())
    /// ```
    #[must_use]
    pub fn with_runtime_config(mut self, config: crate::app_config::RuntimeConfig) -> Self {
        self.runtime_config = Some(config);
        self
    }

    /// Sugar for `.with_runtime_config(RuntimeConfig::builder().cores(cores).build())`.
    #[must_use]
    pub fn with_runtime_cores(self, cores: usize) -> Self {
        self.with_runtime_config(crate::app_config::RuntimeConfig::builder().cores(cores).build())
    }

    pub fn build(self) -> Result<App, ProximaError> {
        let recording_source_registry = Arc::new(self.recording_source_registry);
        let recording_spigot = self.recording_spigot;
        // replay/process/record close over the finalized Arc'd registries.
        if self.defaults_replay && self.pipe_factory_registry.get("replay").is_err() {
            self.pipe_factory_registry
                .register(Arc::new(ReplayPipeFactory::new(
                    recording_source_registry.clone(),
                )))?;
        }
        #[cfg(feature = "tokio")]
        if self.defaults_process && self.pipe_factory_registry.get("process").is_err() {
            self.pipe_factory_registry.register(Arc::new(
                ProcessPipeFactory::with_log_buffer_registry(self.log_buffers.clone()),
            ))?;
        }
        let pipe_factory_registry = Arc::new(self.pipe_factory_registry);
        if self.defaults_record && pipe_factory_registry.get("record").is_err() {
            pipe_factory_registry.register(Arc::new(RecordPipeFactory::new(
                Arc::downgrade(&pipe_factory_registry),
                recording_spigot.clone(),
            )))?;
        }
        // see the `with_defaults` binding above for why this is feature-conditional.
        #[allow(clippy::let_unit_value)]
        let http_client = self.http_client.unwrap_or_default();
        let metrics = self.metrics.clone();
        let telemetry: TelemetryHandle = self
            .telemetry
            .or_else(|| metrics.clone().map(|metric| metric as TelemetryHandle))
            .unwrap_or_else(|| Arc::new(NoopTelemetry));
        let schemas = Arc::new(self.schemas);
        if self.defaults_validate && pipe_factory_registry.get("validate").is_err() {
            pipe_factory_registry.register(Arc::new(ValidateFactory::new(schemas.clone())))?;
        }
        let load_context = LoadContext {
            registry: pipe_factory_registry,
            source_registry: Arc::new(proxima_primitives::pipe::SourceFactoryRegistry::new()),
            recording_spigot,
            recording_source_registry,
            codec_registry: Arc::new(self.codec_registry),
            config_formats: Arc::new(self.config_formats),
            schemas,
            log_buffers: self.log_buffers,
            telemetry,
            metrics,
            http_client,
        };
        let listen_registry = Arc::new(self.listen_registry);
        let cores_override = self.runtime_config.map(|config| config.resolved_cores());
        App::with_components(
            load_context,
            listen_registry,
            build_router_state(),
            cores_override,
        )
    }
}

fn build_router_state() -> Arc<ArcSwap<Router>> {
    Arc::new(ArcSwap::from_pointee(Router::new()))
}

impl App {
    pub fn with_components(
        load_context: LoadContext,
        listen_registry: Arc<ListenRegistry>,
        router: Arc<ArcSwap<Router>>,
        cores_override: Option<usize>,
    ) -> Result<Self, ProximaError> {
        Self::__internal_assemble(load_context, listen_registry, router, cores_override)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pipe::{PipeHandle, into_handle};
    use crate::pipe_factory::PipeFactory;
    use crate::request::{Request, Response};
    use bytes::Bytes;
    use proxima_primitives::pipe::SendPipe;
    use serde_json::{Value, json};
    use std::future::Future;
    use std::pin::Pin;

    struct CountingFactory {
        builds: Arc<std::sync::atomic::AtomicU32>,
    }

    struct CountingPipe {
        builds: Arc<std::sync::atomic::AtomicU32>,
    }

    impl SendPipe for CountingPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
            let _ = self.builds.clone();
            async move { Ok(Response::ok("counted")) }
        }
    }


    impl PipeFactory for CountingFactory {
        fn name(&self) -> &str {
            "counting"
        }

        fn build(
            &self,
            _spec: &Value,
            _inner: Option<PipeHandle>,
        ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
            let builds = self.builds.clone();
            Box::pin(async move {
                builds.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(into_handle(CountingPipe { builds }))
            })
        }
    }

    struct StampFactory;

    impl PipeFactory for StampFactory {
        fn name(&self) -> &str {
            "stamp"
        }

        fn build(
            &self,
            _spec: &Value,
            inner: Option<PipeHandle>,
        ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
            Box::pin(async move {
                let inner = inner
                    .ok_or_else(|| ProximaError::Config("stamp requires an inner pipe".into()))?;
                struct Wrapped {
                    inner: PipeHandle,
                }
                impl SendPipe for Wrapped {
                    type In = Request<Bytes>;
                    type Out = Response<Bytes>;
                    type Err = ProximaError;

                    fn call(
                        &self,
                        request: Request<Bytes>,
                    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>>
                    {
                        let inner = self.inner.clone();
                        async move {
                            let response = SendPipe::call(&inner, request).await?;
                            Ok(response.with_header("x-stamp", "yes"))
                        }
                    }
                }
                Ok(into_handle(Wrapped { inner }))
            })
        }
    }

    #[proxima::test]
    async fn empty_builder_yields_app_with_no_factories() {
        let app = AppBuilder::new().build().expect("build");
        let names = app.load_context().registry.names();
        assert!(names.is_empty(), "names: {names:?}");
    }

    #[proxima::test]
    async fn with_defaults_registers_every_built_in() {
        let app = AppBuilder::new()
            .with_defaults()
            .expect("defaults")
            .build()
            .expect("build");
        let upstreams = app.load_context().registry.names();
        assert!(upstreams.contains(&"http".to_string()));
        assert!(upstreams.contains(&"synth".to_string()));
        assert!(upstreams.contains(&"replay".to_string()));
        #[cfg(feature = "tokio")]
        assert!(upstreams.contains(&"process".to_string()));
        #[cfg(any(feature = "tcp", feature = "unix"))]
        assert!(upstreams.contains(&"stream".to_string()));
        let middleware = app.load_context().registry.names();
        assert!(middleware.contains(&"retry".to_string()));
        assert!(middleware.contains(&"rate_limit".to_string()));
        assert!(middleware.contains(&"transform".to_string()));
        let codecs = app.load_context().codec_registry.names();
        assert!(codecs.contains(&"json".to_string()));
        assert!(codecs.contains(&"bytes".to_string()));
    }

    #[proxima::test]
    async fn plugin_factories_compose_with_defaults() {
        let counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let app = AppBuilder::new()
            .with_defaults()
            .expect("defaults")
            .with_upstream_factory(Arc::new(CountingFactory {
                builds: counter.clone(),
            }))
            .expect("counting")
            .with_upstream_factory(Arc::new(StampFactory))
            .expect("stamp")
            .build()
            .expect("build");
        let context = app.load_context();
        let counting_factory = context
            .registry
            .get("counting")
            .expect("counting registered");
        let _ = counting_factory
            .build(&json!({}), None)
            .await
            .expect("build counting");
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert!(context.registry.get("stamp").is_ok());
    }

    // same condition as `app::default_runtime`/`app::runtime_cores`: this
    // test asserts a runtime got installed, which only happens when one is
    // compiled in.
    #[cfg(any(
        feature = "runtime-tokio",
        all(
            feature = "serve-prime",
            feature = "runtime-prime-reactor",
            any(target_os = "linux", target_os = "macos")
        )
    ))]
    #[proxima::test]
    async fn app_builder_build_installs_runtime() {
        let app = AppBuilder::new()
            .with_defaults()
            .expect("defaults")
            .build()
            .expect("build");
        // builder-constructed apps must carry the default runtime + acceptor
        // factory, not None — otherwise run_until_signal errors with "no
        // Runtime installed". this is the regression guard for the old
        // `runtime: None` bug in __internal_assemble.
        assert!(
            app.runtime().is_some(),
            "AppBuilder::build must install the default runtime"
        );
        assert!(
            app.acceptor_factory().is_some(),
            "AppBuilder::build must install the default acceptor factory"
        );
    }

    #[proxima::test]
    async fn duplicate_default_registration_fails_loudly() {
        let outcome = AppBuilder::new()
            .with_defaults()
            .expect("first defaults")
            .with_defaults();
        assert!(matches!(outcome, Err(ProximaError::Registry(_))));
    }
}
