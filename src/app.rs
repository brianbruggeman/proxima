use std::collections::BTreeMap;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;

use arc_swap::ArcSwap;
use bytes::Bytes;
use futures::channel::oneshot;
use serde_json::Value;
use tracing::warn;

use proxima_primitives::pipe::SendPipe;

use crate::context_inject::ContextInjector;
use crate::error::ProximaError;
#[cfg(feature = "http1")]
use crate::listeners::http::HttpListenProtocol;
#[cfg(feature = "tokio")]
use crate::listeners::mcp::McpListenProtocol;
use crate::load::{LoadContext, ResolvedSpec, Spec, load};
use crate::mount::{MethodFilter, Mount, Router};
use crate::pipe::{Handler, PipeHandle, into_handle};
use crate::request::{Request, Response};
use crate::runtime::Runtime;
use crate::telemetry::{Metrics, TelemetryHandle};
use proxima_listen::ListenRegistry;
use proxima_listen::handle::{ListenerHandle, ListenerSpec};
use proxima_primitives::stream::AcceptorFactory;

pub struct App {
    pipes: BTreeMap<String, PipeHandle>,
    /// Resolved source spec per registered pipe. Retained so
    /// [`pipe_builder_from_existing`] can pre-fill the builder with
    /// the existing spec for read-modify-rebuild flows. Entries are
    /// `Value`-form (post-resolve); pipes registered via a pre-built
    /// `PipeHandle` (`Spec::Handle`) have no entry here — the spec
    /// is not recoverable from the handle.
    pipe_specs: BTreeMap<String, Value>,
    /// Registered background producers (`Signal -> ()` sources — interval
    /// ticks, scheduled triggers). Unlike `pipes`, these are never mounted;
    /// `run_until_signal` spawns every entry here onto a fresh
    /// `ProducerLifecycle` once the listener is bound.
    sources: BTreeMap<String, proxima_primitives::pipe::SourceHandle>,
    router: Arc<ArcSwap<Router>>,
    load_context: LoadContext,
    listen_registry: Arc<ListenRegistry>,
    /// per-core runtime for chain dispatch. `None` means listener spawns
    /// fall back to ambient `tokio::spawn` (work-stealing). install a
    /// `TokioPerCoreRuntime` via `with_runtime` to opt into per-core
    /// dispatch with no work-stealing on the chain runtime.
    runtime: Option<Arc<dyn crate::runtime::Runtime>>,
    /// acceptor factory the listener serve path uses to bind + accept
    /// sockets. paired with `runtime`: `PrimeAcceptorFactory` for the prime
    /// runtime (default), `TokioAcceptorFactory` under `runtime-tokio`. `None`
    /// leaves the protocol to its built-in accept path.
    acceptor_factory: Option<Arc<dyn AcceptorFactory>>,
}

/// Pick the default serve+chain runtime and its matching acceptor factory.
///
/// cfg precedence: explicit `runtime-tokio` opt-in wins; otherwise the prime
/// runtime serves on unix targets where the reactor is available; otherwise
/// neither (e.g. non-unix with no prime), and listener spawn falls back to
/// ambient `tokio::spawn`.
///
/// note: the app/serve path is always std (it binds sockets), so cpu probing
/// via std-only `num_cpus::get()` is fine here — no no_std cpu probe needed.
/// Worker-core count for the default runtime, resolved through
/// [`crate::app_config::RuntimeConfig`] — conflaguration-backed defaults with
/// a `PROXIMA_RUNTIME_CORES` env layer — unless `explicit` names a count
/// directly (threaded from `AppBuilder::with_runtime_config`). Clamped to at
/// least 1. A single process running many `App`s (the test suite) sets
/// `PROXIMA_RUNTIME_CORES` low so it does not spin `num_cpus` per-core
/// runtimes per `App` and oversubscribe the box.
// mirrors default_runtime's two callers exactly (by `A or (B and not A) = A or
// B`): the prime-tokio-compat arm needs serve-prime + reactor + the right os;
// the plain-tokio arm needs runtime-tokio. `runtime-prime-reactor` alone
// (without serve-prime, or on an unsupported os) satisfies neither.
#[cfg(any(
    feature = "runtime-tokio",
    all(
        feature = "serve-prime",
        feature = "runtime-prime-reactor",
        any(target_os = "linux", target_os = "macos")
    )
))]
fn resolve_cores(explicit: Option<usize>) -> Result<usize, ProximaError> {
    if let Some(cores) = explicit {
        return Ok(cores.max(1));
    }
    Ok(crate::app_config::RuntimeConfig::resolve_from_env()?.resolved_cores())
}

/// The default runtime plus its matching acceptor factory. `None` when no
/// runtime backend is compiled in (e.g. non-unix without prime).
type RuntimeAndFactory = (Option<Arc<dyn Runtime>>, Option<Arc<dyn AcceptorFactory>>);

// cfg-arm returns: exactly one arm compiles per build, so the early `return`
// is that arm's natural exit. needless_return is a false positive under the
// cfg cascade (the other arms are compiled out).
#[allow(clippy::needless_return)]
fn default_runtime(cores_override: Option<usize>) -> Result<RuntimeAndFactory, ProximaError> {
    // `#[proxima::main]` (or any other `block_on*` driver) may have already
    // booted a runtime sized by its own `runtime = ...` / `cores = ...` /
    // `affinity = ...` args and published it — adopt that instead of
    // building an independent second one. Without this, `#[proxima::main(cores
    // = 1)]` boots a 1-core runtime to drive `main`, and `App::new()` inside
    // `main`'s body boots a SECOND, independent runtime at `num_cpus::get()`
    // — two runtimes, contradictory core counts, one process. See
    // `crate::runtime::install_runtime` / `installed_runtime`.
    if let Some(installed) = crate::runtime::installed_runtime() {
        return Ok((Some(installed.runtime), Some(installed.acceptor_factory)));
    }

    // PRIME-FIRST: when `serve-prime` is set (the default) and the prime reactor
    // is available, serve on prime even if `runtime-tokio` is ALSO linked.
    // `runtime-tokio` is pulled into many multi-crate builds by cargo feature-
    // unification (e.g. proxima's own dev-dep on proxima-h1 -> proxima-runtime-
    // tokio); letting its mere presence flip the App to a tokio runtime left the
    // prime `http` upstream's `TcpStream` polled off a reactor worker
    // (CURRENT_REACTOR null -> RetriesExhausted -> 502). Explicit tokio serve is
    // now: opt OUT of `serve-prime` AND opt INTO `runtime-tokio`.
    #[cfg(all(
        feature = "serve-prime",
        feature = "runtime-prime-reactor",
        feature = "tokio",
        any(target_os = "linux", target_os = "macos")
    ))]
    {
        // tokio-compat: the prime transport (accept/serve/codec) needs no
        // tokio reactor (serve_parity proves it with a non-compat runtime),
        // but user pipes may call tokio primitives (tokio::time, hyper
        // upstream). compat installs a tokio reactor handle on each prime
        // worker so those keep working; the transport simply never uses it.
        let runtime: Arc<dyn Runtime> = Arc::new(
            crate::runtime::PrimeRuntime::new_with_tokio_compat(resolve_cores(cores_override)?)?,
        );
        let factory: Arc<dyn AcceptorFactory> = Arc::new(proxima_net::prime::PrimeAcceptorFactory);
        return Ok((Some(runtime), Some(factory)));
    }
    // tokio-free default: same prime transport, no sister tokio Handle on
    // each worker (see the `tokio`-gated arm above for that variant). User
    // pipes calling `tokio::{time,sync,spawn}` need `--features tokio`.
    #[cfg(all(
        feature = "serve-prime",
        feature = "runtime-prime-reactor",
        not(feature = "tokio"),
        any(target_os = "linux", target_os = "macos")
    ))]
    {
        let runtime: Arc<dyn Runtime> = Arc::new(crate::runtime::PrimeRuntime::new(
            resolve_cores(cores_override)?,
        )?);
        let factory: Arc<dyn AcceptorFactory> = Arc::new(proxima_net::prime::PrimeAcceptorFactory);
        return Ok((Some(runtime), Some(factory)));
    }
    #[cfg(all(
        feature = "runtime-tokio",
        not(all(
            feature = "serve-prime",
            feature = "runtime-prime-reactor",
            any(target_os = "linux", target_os = "macos")
        ))
    ))]
    {
        let runtime: Arc<dyn Runtime> = Arc::new(crate::runtime::TokioPerCoreRuntime::new(
            resolve_cores(cores_override)?,
        )?);
        let factory: Arc<dyn AcceptorFactory> = Arc::new(proxima_net::tokio::TokioAcceptorFactory);
        return Ok((Some(runtime), Some(factory)));
    }
    #[cfg(all(
        not(feature = "runtime-tokio"),
        not(all(
            feature = "serve-prime",
            feature = "runtime-prime-reactor",
            any(target_os = "linux", target_os = "macos")
        ))
    ))]
    {
        let _ = cores_override;
        Ok((None, None))
    }
}

/// A standalone runtime for offline tooling (verify/replay walks, MCP
/// `verify_replay`): the recording sources offload their blocking file reads
/// onto its background pool. Prefers prime (the serve default), falls back to
/// tokio; errors only when neither runtime backend is linked.
// cfg-arm returns: exactly one arm compiles per build (same cfg cascade as
// default_runtime), so needless_return is a false positive here.
#[allow(clippy::needless_return)]
pub fn offline_runtime() -> Result<Arc<dyn Runtime>, ProximaError> {
    #[cfg(all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    ))]
    {
        return Ok(Arc::new(crate::runtime::PrimeRuntime::new(1)?));
    }
    #[cfg(all(
        feature = "runtime-tokio",
        not(all(
            feature = "runtime-prime-executor",
            feature = "runtime-prime-inbox-alloc",
            feature = "runtime-prime-reactor",
            feature = "runtime-prime-bgpool"
        ))
    ))]
    {
        return Ok(Arc::new(crate::runtime::TokioPerCoreRuntime::new(1)?));
    }
    #[cfg(all(
        not(feature = "runtime-tokio"),
        not(all(
            feature = "runtime-prime-executor",
            feature = "runtime-prime-inbox-alloc",
            feature = "runtime-prime-reactor",
            feature = "runtime-prime-bgpool"
        ))
    ))]
    {
        Err(ProximaError::Config(
            "offline_runtime: no runtime backend linked (enable runtime-prime or runtime-tokio)"
                .into(),
        ))
    }
}

impl App {
    pub fn new() -> Result<Self, ProximaError> {
        let listen_registry = Arc::new(ListenRegistry::new());
        #[cfg(feature = "http1")]
        listen_registry.register(Arc::new(HttpListenProtocol::new()))?;
        #[cfg(feature = "tokio")]
        listen_registry.register(Arc::new(McpListenProtocol::new()))?;
        #[cfg(feature = "tcp")]
        listen_registry.register(Arc::new(crate::listeners::StreamListenProtocol::new()))?;

        // Adopt whatever runtime `#[proxima::main]` already booted; otherwise
        // default to the prime per-core runtime + prime acceptor (the prime
        // `TcpStream` only drives on a CoreShard worker's reactor). Explicit
        // `runtime-tokio` flips both to the tokio runtime + tokio acceptor.
        // Every chain dispatch goes through the Runtime trait (no work-stealing
        // on the chain path). Users may override the runtime with
        // `.with_runtime(...)`, or size the fallback default with
        // `App::builder().with_runtime_config(...)`.
        let (runtime, acceptor_factory) = default_runtime(None)?;

        let load_context = LoadContext::with_default_registry()?;
        // arm the recording spigot with the App's runtime so a directly-called
        // `record` upstream pumps without a serve loop; files still open lazily
        // on first call (C7 spigot model).
        if let Some(rt) = &runtime {
            let _ = load_context.recording_spigot.set(rt.clone());
            // read-source factories offload their file I/O through this same
            // runtime; arm the registry so a resolved BinSource/JsonlSource pumps.
            load_context
                .recording_source_registry
                .set_runtime(rt.clone());
        }
        Ok(Self {
            pipes: BTreeMap::new(),
            pipe_specs: BTreeMap::new(),
            sources: BTreeMap::new(),
            router: Arc::new(ArcSwap::from_pointee(Router::new())),
            load_context,
            listen_registry,
            runtime,
            acceptor_factory,
        })
    }

    #[must_use]
    pub fn with_runtime(mut self, runtime: Arc<dyn crate::runtime::Runtime>) -> Self {
        self.runtime = Some(runtime);
        self
    }

    #[must_use]
    pub fn runtime(&self) -> Option<Arc<dyn crate::runtime::Runtime>> {
        self.runtime.clone()
    }

    #[must_use]
    pub fn acceptor_factory(&self) -> Option<Arc<dyn AcceptorFactory>> {
        self.acceptor_factory.clone()
    }

    #[must_use]
    pub fn with_acceptor_factory(mut self, factory: Arc<dyn AcceptorFactory>) -> Self {
        self.acceptor_factory = Some(factory);
        self
    }

    /// Fluent builder for registering and mounting a pipe in one
    /// chain. The native-API counterpart of the TOML
    /// `[pipes.<name>] chain = [...]` + `mount = "..."` shape.
    /// Both surfaces are first-class — config and API are
    /// isomorphic; pick the shape that fits the deployment.
    ///
    /// ```no_run
    /// # async fn doctest() -> proxima::ProximaResult<()> {
    /// use proxima::{App, BearerAuth, Composable, HttpUpstream};
    /// use proxima::settings::RateLimit;
    /// use std::time::Duration;
    ///
    /// let mut app = App::new()?;
    /// app.pipe_builder("api")
    ///     .dispatch(
    ///         BearerAuth::allow_tokens(["t-1"])
    ///             .then(RateLimit::token_bucket(100, 50))
    ///             .then(
    ///                 HttpUpstream::builder()
    ///                     .url("https://x")
    ///                     .timeout(Duration::from_secs(5))
    ///                     .build(),
    ///             ),
    ///     )
    ///     .mount("/api/{*path}")
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn pipe_builder(&mut self, name: impl Into<String>) -> AppPipeBuilder<'_> {
        AppPipeBuilder {
            app: self,
            name: name.into(),
            spec: None,
            mount: None,
            methods: None,
        }
    }

    /// Builder pre-filled with the existing registered pipe's spec,
    /// for read-modify-rebuild flows ("load from config, mod via
    /// API"). Returns `None` when no pipe is registered under that
    /// name, or when the pipe was registered via a pre-built
    /// `PipeHandle` (no recoverable spec).
    ///
    /// `build()` on the returned builder routes through
    /// [`App::update_pipe`] — in-flight requests on the old handle
    /// finish, new requests dispatch to the modified handle via
    /// `SwappablePipe` semantics.
    ///
    /// ```ignore
    /// // Load from config, then swap one upstream's URL via API.
    /// app.load_full(spec_path).await?;
    /// let current = app.lookup_pipe_spec("api").expect("registered").clone();
    /// let modified = mutate_spec(current);  // your serde_json::Value editing
    /// app.pipe_builder_from_existing("api")
    ///     .expect("present")
    ///     .dispatch(modified)
    ///     .build()
    ///     .await?;
    /// ```
    #[must_use]
    pub fn pipe_builder_from_existing(&mut self, name: &str) -> Option<AppPipeBuilder<'_>> {
        let existing_spec = self.pipe_specs.get(name).cloned()?;
        Some(AppPipeBuilder {
            app: self,
            name: name.to_string(),
            spec: Some(Spec::Inline(existing_spec)),
            mount: None,
            methods: None,
        })
    }

    pub async fn pipe(
        &mut self,
        name: impl Into<String>,
        spec: impl Into<Spec>,
    ) -> Result<PipeHandle, ProximaError> {
        let name = name.into();
        let mut resolved = spec.into().resolve(&self.load_context.config_formats)?;
        if let Some(value) = resolved.as_value_mut()
            && let Value::Object(table) = value
            && !table.contains_key("name")
        {
            table.insert("name".into(), Value::String(name.clone()));
        }
        let (handle, spec_for_storage) = match resolved {
            ResolvedSpec::Handle(handle) => (handle, None),
            ResolvedSpec::Value(value) => {
                let handle = load(value.clone(), &self.load_context).await?;
                (handle, Some(value))
            }
        };
        if let Some(spec_value) = spec_for_storage {
            self.pipe_specs.insert(name.clone(), spec_value);
        } else {
            // Handle-form: no recoverable spec; drop any prior spec
            // entry so `pipe_builder_from_existing` returns None.
            self.pipe_specs.remove(&name);
        }
        self.pipes.insert(name, handle.clone());
        Ok(handle)
    }

    /// Return the resolved `Value`-form spec of a registered pipe,
    /// if available. Pipes registered via a pre-built `PipeHandle`
    /// (i.e. `Spec::Handle`) return `None` — the spec is not
    /// recoverable from a handle.
    #[must_use]
    pub fn lookup_pipe_spec(&self, name: &str) -> Option<&Value> {
        self.pipe_specs.get(name)
    }

    pub fn pipes(&self) -> &BTreeMap<String, PipeHandle> {
        &self.pipes
    }

    pub fn lookup_pipe(&self, name: &str) -> Option<PipeHandle> {
        self.pipes.get(name).cloned()
    }

    /// Register a background producer under `name`. Unlike `pipe`/`mount`,
    /// a source is never mounted on a listener — `run_until_signal` spawns
    /// every registered source onto a fresh `ProducerLifecycle` once the
    /// listener is bound, and `Shutdown::drain` folds their drain report in.
    pub fn source(
        &mut self,
        name: impl Into<String>,
        source: impl Into<proxima_primitives::pipe::SourceHandle>,
    ) {
        self.sources.insert(name.into(), source.into());
    }

    /// Registered source names (in registration order is not guaranteed —
    /// `BTreeMap` iterates sorted by key).
    pub fn sources(&self) -> impl Iterator<Item = &str> {
        self.sources.keys().map(String::as_str)
    }

    /// Look up a registered source by name — the `SourceHandle` sibling of
    /// [`lookup_pipe`](App::lookup_pipe).
    #[must_use]
    pub fn lookup_source(&self, name: &str) -> Option<proxima_primitives::pipe::SourceHandle> {
        self.sources.get(name).cloned()
    }

    pub async fn update_pipe(
        &mut self,
        name: &str,
        spec: impl Into<Spec>,
    ) -> Result<PipeHandle, ProximaError> {
        let old_handle = self.pipes.get(name).cloned().ok_or_else(|| {
            ProximaError::NotFound(format!(
                "no pipe registered as '{name}'; use App::pipe to register first"
            ))
        })?;
        let mut resolved = spec.into().resolve(&self.load_context.config_formats)?;
        if let Some(value) = resolved.as_value_mut()
            && let Value::Object(table) = value
            && !table.contains_key("name")
        {
            table.insert("name".into(), Value::String(name.to_string()));
        }
        let (handle, spec_for_storage) = match resolved {
            ResolvedSpec::Handle(handle) => (handle, None),
            ResolvedSpec::Value(value) => {
                let handle = load(value.clone(), &self.load_context).await?;
                (handle, Some(value))
            }
        };
        match spec_for_storage {
            Some(spec_value) => {
                self.pipe_specs.insert(name.to_string(), spec_value);
            }
            None => {
                self.pipe_specs.remove(name);
            }
        }
        self.pipes.insert(name.to_string(), handle.clone());
        // Mount fields are Arc'd; rebuilding only bumps refcounts.
        let snapshot = self.router.load();
        let existing = snapshot.mounts();
        let mut new_router = Router::with_capacity(existing.len());
        for mount in existing {
            let pipe = if Arc::ptr_eq(&mount.pipe, &old_handle) {
                handle.clone()
            } else {
                mount.pipe.clone()
            };
            new_router.add(Mount {
                path: mount.path.clone(),
                pipe,
                methods: mount.methods.clone(),
                host: mount.host.clone(),
                label: Arc::clone(&mount.label),
                pipe_name: Arc::clone(&mount.pipe_name),
            });
        }
        drop(snapshot);
        self.router.store(Arc::new(new_router));
        Ok(handle)
    }

    pub fn remove_pipe(&mut self, name: &str) -> Result<(), ProximaError> {
        let old_handle = self
            .pipes
            .remove(name)
            .ok_or_else(|| ProximaError::NotFound(format!("no pipe registered as '{name}'")))?;
        let mut router = (*self.router.load_full()).clone();
        let labels: Vec<Arc<[u8]>> = router
            .mounts()
            .iter()
            .filter(|mount| Arc::ptr_eq(&mount.pipe, &old_handle))
            .map(|mount| Arc::clone(&mount.label))
            .collect();
        for label in labels {
            // labels are URL-pattern-shaped (ASCII) by construction.
            let label_str = std::str::from_utf8(&label).unwrap_or("");
            router.remove(label_str);
        }
        self.router.store(Arc::new(router));
        Ok(())
    }

    pub fn unmount(&self, path: &str) -> bool {
        let mut router = (*self.router.load_full()).clone();
        let removed = router.remove(path);
        if removed > 0 {
            self.router.store(Arc::new(router));
        }
        removed > 0
    }

    /// Mount a pipe at `path`. Accepts anything [`IntoMountTarget`] covers —
    /// a handler-shaped pipe (`H: Handler`, e.g. a `#[proxima::piped(send)]`
    /// struct or a [`PipeHandle`]), a bare request-shaped `async fn`, a
    /// registered pipe name (`&str`/`String`), or an already-built
    /// [`MountTarget`]. `Via` is inferred and never named at the call site —
    /// see [`IntoMountTarget`]'s doc for why the three shapes don't collide.
    pub fn mount<Target, Via>(&self, path: &str, target: Target) -> Result<(), ProximaError>
    where
        Target: IntoMountTarget<Via>,
    {
        let target = target.into_mount_target();
        let mount = match target {
            MountTarget::Handle(handle) => Mount::new(path, handle),
            MountTarget::Named(name) => {
                let pipe = self.lookup_pipe(&name).ok_or_else(|| {
                    ProximaError::NotFound(format!("no pipe registered as '{name}'"))
                })?;
                Mount::new(path, pipe).named(name.clone())
            }
        };
        let mut router = (*self.router.load_full()).clone();
        router.add(mount);
        self.router.store(Arc::new(router));
        Ok(())
    }

    pub fn mount_with_methods<Target, Via>(
        &self,
        path: &str,
        target: Target,
        methods: MethodFilter,
    ) -> Result<(), ProximaError>
    where
        Target: IntoMountTarget<Via>,
    {
        let target = target.into_mount_target();
        let mount = match target {
            MountTarget::Handle(handle) => Mount::new(path, handle).with_methods(methods),
            MountTarget::Named(name) => {
                let pipe = self.lookup_pipe(&name).ok_or_else(|| {
                    ProximaError::NotFound(format!("no pipe registered as '{name}'"))
                })?;
                Mount::new(path, pipe)
                    .with_methods(methods)
                    .named(name.clone())
            }
        };
        let mut router = (*self.router.load_full()).clone();
        router.add(mount);
        self.router.store(Arc::new(router));
        Ok(())
    }

    pub fn mount_full<Target, Via>(
        &self,
        path: &str,
        target: Target,
        methods: MethodFilter,
        host: crate::mount::HostFilter,
    ) -> Result<(), ProximaError>
    where
        Target: IntoMountTarget<Via>,
    {
        let target = target.into_mount_target();
        let mount = match target {
            MountTarget::Handle(handle) => Mount::new(path, handle)
                .with_methods(methods)
                .with_host(host),
            MountTarget::Named(name) => {
                let pipe = self.lookup_pipe(&name).ok_or_else(|| {
                    ProximaError::NotFound(format!("no pipe registered as '{name}'"))
                })?;
                Mount::new(path, pipe)
                    .with_methods(methods)
                    .with_host(host)
                    .named(name.clone())
            }
        };
        let mut router = (*self.router.load_full()).clone();
        router.add(mount);
        self.router.store(Arc::new(router));
        Ok(())
    }

    /// Internal — `AppBuilder::build` calls this with already-assembled
    /// state, bypassing the default-registry path of `App::new`. Installs the
    /// same default runtime + acceptor factory as `App::new` (ambient-adopt
    /// first, `cores_override` as the explicit fallback sizing — see
    /// `AppBuilder::with_runtime_config`) so a builder-constructed app can
    /// `run_until_signal` without a manual `.with_runtime(...)`.
    #[doc(hidden)]
    pub fn __internal_assemble(
        load_context: LoadContext,
        listen_registry: Arc<ListenRegistry>,
        router: Arc<ArcSwap<Router>>,
        cores_override: Option<usize>,
    ) -> Result<Self, ProximaError> {
        let (runtime, acceptor_factory) = default_runtime(cores_override)?;
        // arm the recording spigot at build (see App::new) — a builder-made App
        // whose `record` upstream is called directly still pumps.
        if let Some(rt) = &runtime {
            let _ = load_context.recording_spigot.set(rt.clone());
            load_context
                .recording_source_registry
                .set_runtime(rt.clone());
        }
        Ok(Self {
            pipes: BTreeMap::new(),
            pipe_specs: BTreeMap::new(),
            sources: BTreeMap::new(),
            router,
            load_context,
            listen_registry,
            runtime,
            acceptor_factory,
        })
    }

    #[must_use]
    pub fn load_context(&self) -> &LoadContext {
        &self.load_context
    }

    pub fn router_handle(&self) -> PipeHandle {
        into_handle(RouterDispatch {
            router: self.router.clone(),
        })
    }

    #[must_use]
    pub fn metrics(&self) -> Option<Arc<Metrics>> {
        self.load_context.metrics.clone()
    }

    #[must_use]
    pub fn telemetry(&self) -> TelemetryHandle {
        self.load_context.telemetry.clone()
    }

    #[must_use]
    pub fn builder() -> crate::app_builder::AppBuilder {
        crate::app_builder::AppBuilder::new()
    }

    pub async fn run_until_signal(&self, config: RunConfig) -> Result<Shutdown, ProximaError> {
        let protocol = self.listen_registry.get(&config.protocol)?;
        let dispatch: PipeHandle = into_handle(ContextInjector::new(
            self.router_handle(),
            self.load_context.telemetry.clone(),
        ));
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let bind = config.bind;
        let spec = config.spec;
        let telemetry = self.load_context.telemetry.clone();
        let runtime = self.runtime.clone().ok_or_else(|| {
            ProximaError::Config(
                "App has no Runtime installed: enable the `runtime-tokio` \
                 feature (default) or call `.with_runtime(...)` before \
                 `run_until_signal`. Pipe::call returns ?Send and needs a \
                 per-core executor"
                    .into(),
            )
        })?;
        // arm the recording spigot with the serve runtime: the `record`
        // upstream's durable sink stays inert until here (C7 spigot model).
        // set-once — a re-serve observes the already-armed spigot.
        let _ = self.load_context.recording_spigot.set(runtime.clone());
        self.load_context
            .recording_source_registry
            .set_runtime(runtime.clone());
        let runtime_for_factory = runtime.clone();
        let acceptor_factory = self.acceptor_factory.clone();
        // drain_notify: listener fires this when its serve() future
        // returns (after quiesce + drain). `Shutdown::drain` awaits
        // it BEFORE broadcast_drop so per-core drop hooks run only
        // after every in-flight request has completed (phase 2 → 3
        // ordering per the ShutdownBarrier plan).
        let drain_notify = std::sync::Arc::new(proxima_primitives::sync::Notify::new());
        let drain_notify_for_factory = drain_notify.clone();
        // mirrors `Listener::run_with_runtime`'s readiness gate (handle.rs):
        // the factory's spawn returning does not mean the socket is
        // listening yet — block on one ack from the serve future's real
        // bind/listen so a caller dialing immediately after this returns
        // never sees ECONNREFUSED.
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<()>();
        // ship a Send factory cross-core; serve future is constructed on the
        // target worker (where it can spawn_local `?Send` connection futures).
        // setup-time spawn: surface failure to the caller rather than
        // silently losing the listener.
        runtime
            .spawn_factory_on_core(
                crate::runtime::CoreId(0),
                Box::new(move || {
                    let mut context = proxima_listen::ServeContext::new(telemetry)
                        .with_runtime(runtime_for_factory)
                        .with_ready_signal(ready_tx);
                    if let Some(factory) = acceptor_factory {
                        context = context.with_acceptor_factory(factory);
                    }
                    Box::pin(async move {
                        if let Err(error) = protocol
                            .serve(bind, dispatch, &spec, context, shutdown_rx)
                            .await
                        {
                            warn!(?error, "listener exited with error");
                        }
                        drain_notify_for_factory.notify_waiters();
                    })
                }),
            )
            .map_err(|err| {
                crate::error::ProximaError::Config(format!(
                    "listener spawn on core 0 failed: {err}"
                ))
            })?;
        if ready_rx
            .recv_timeout(proxima_listen::handle::LISTENER_READY_TIMEOUT)
            .is_err()
        {
            return Err(crate::error::ProximaError::Config(format!(
                "listener did not become ready within {:?}",
                proxima_listen::handle::LISTENER_READY_TIMEOUT
            )));
        }
        // Sources drive unconditionally once registered (TARGET 4) — every
        // `App::source(...)` entry gets spawned onto a fresh
        // `ProducerLifecycle` here, after the listener is bound. `Shutdown`
        // owns the lifecycle so `stop`/`drain` can cancel + drain it
        // alongside the listener.
        let mut source_lifecycle = proxima_primitives::pipe::ProducerLifecycle::new();
        let source_cancel = source_lifecycle.cancel_signal();
        for (name, source) in &self.sources {
            source_lifecycle.spawn_from_source(name, source);
        }
        Ok(Shutdown {
            shutdown_tx: Some(shutdown_tx),
            drain_notify,
            _runtime: self.runtime.clone(),
            source_lifecycle: Some(source_lifecycle),
            source_cancel,
        })
    }

    /// Fluent terminal: spawn the listener and return a `Server`
    /// handle. Equivalent to `run_until_signal(config)` wrapped so the
    /// caller gets all three drive shapes (await, explicit method,
    /// clone-and-control) on a single type.
    ///
    /// The returned `Server` impls `ControlPlane` so callers can do
    /// `server.list_pipes().await?` in-process. Today the
    /// underlying impl is a read-only `StaticControlPlane`; richer
    /// impls (DaemonControlPlane integration) land with Phase 3b
    /// when the fluent accumulators come online.
    pub async fn serve(
        &self,
        config: impl Into<RunConfig>,
    ) -> Result<crate::server::Server, ProximaError> {
        let shutdown = self.run_until_signal(config.into()).await?;
        let control: crate::control_plane::DynControlPlane =
            Arc::new(crate::control_plane::StaticControlPlane::new(Vec::new()));
        Ok(crate::server::Server::new(shutdown, control))
    }

    /// Materialize the upstream + pipe sections of a typed
    /// `ProximaSettings` into this `App`. Each upstream registers as
    /// a `PipeHandle`; each pipe composes its declared chain
    /// (middlewares + leaf upstream by name) and registers the
    /// composed Pipe.
    ///
    /// Listeners are NOT materialized here — they're applied at
    /// `App::serve(...)` time. Middlewares are NOT registered as
    /// standalone Pipes (they require an inner); they're
    /// referenced by name from the pipe entries.
    ///
    /// Partial Phase 5 — full Settings -> App round-trip with the
    /// reverse `Settings::from_app` direction lands when App grows
    /// a "stored spec" sidecar. For now this is the forward path:
    /// load TOML, materialize, run.
    pub async fn apply_settings(
        &mut self,
        settings: &crate::settings::ProximaSettings,
    ) -> Result<(), ProximaError> {
        // Step 1: register each upstream by name as a Pipe.
        // The RegistryEntry's `spec` field is the JSON the factory
        // dispatches on; we hand it through unchanged.
        for (name, entry) in &settings.upstreams {
            let value = registry_entry_value(entry);
            self.pipe(name.clone(), Spec::Inline(value)).await?;
        }
        // Step 2: register each pipe. The chain references named
        // middlewares + upstreams; we resolve them here and emit the
        // composed spec for the factory.
        for (name, entry) in &settings.pipes {
            let composed = compose_pipe_spec(entry, settings)?;
            self.pipe(name.clone(), Spec::Inline(composed)).await?;
        }
        // Step 3 (proxima-notify S3): register each producer as a source.
        // Sources are self-starting `Signal -> ()` pipes (TARGET 4/5) that
        // `run_until_signal` spawns unconditionally once registered — like a
        // handler drives once mounted, there is no longer a feature gate or
        // a separate spawn step the caller must remember to invoke.
        for (name, entry) in &settings.producers {
            let value = registry_entry_value(entry);
            let type_key = entry.r#type.as_str();
            let factory = self.load_context.source_registry.get(type_key)?;
            let source = factory.build(&value)?;
            self.source(name.clone(), source);
        }
        Ok(())
    }

    pub fn build_listener(&self, spec: ListenerSpec) -> Result<ListenerHandle, ProximaError> {
        let dispatch: PipeHandle = into_handle(ContextInjector::new(
            self.router_handle(),
            self.load_context.telemetry.clone(),
        ));
        spec.attach(dispatch).run_with_runtime(
            &self.listen_registry,
            self.load_context.telemetry.clone(),
            self.runtime.clone(),
            self.acceptor_factory.clone(),
            None,
        )
    }

    /// Parse a multi-block config: build every `[[pipe]]`, attach
    /// every `[[listen]].mount[]`, bind every `[[listen]]`. Returns one
    /// `ListenerHandle` per listener.
    pub async fn load_full(
        &mut self,
        spec: impl Into<Spec>,
    ) -> Result<Vec<ListenerHandle>, ProximaError> {
        let resolved = spec.into().resolve(&self.load_context.config_formats)?;
        let value = match resolved {
            ResolvedSpec::Value(value) => value,
            ResolvedSpec::Handle(_) => {
                return Err(ProximaError::Config(
                    "load_full requires a Value spec, not a pre-built Handle".into(),
                ));
            }
        };
        let table = value.as_object().ok_or_else(|| {
            ProximaError::Config("load_full expects a top-level table / object".into())
        })?;

        // register named schemas before pipes build, so middleware that
        // references them by name (`validate`) can resolve at build time.
        if let Some(schema_blocks) = table.get("schema").and_then(Value::as_array) {
            for entry in schema_blocks {
                let name = entry
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| ProximaError::Config("[[schema]] requires `name`".into()))?
                    .to_string();
                let schema_value = entry.get("schema").cloned().ok_or_else(|| {
                    ProximaError::Config(format!(
                        "[[schema]] `{name}` requires a nested `schema = {{ type = \"...\", ... }}` block"
                    ))
                })?;
                let schema: crate::schema::Schema =
                    serde_json::from_value(schema_value).map_err(|err| {
                        ProximaError::Config(format!("[[schema]] `{name}` decode: {err}"))
                    })?;
                self.load_context.schemas.register(name, schema)?;
            }
        }

        // build all named pipes first so listeners can reference them.
        if let Some(pipes) = table.get("pipe").and_then(Value::as_array) {
            for entry in pipes {
                let name = entry
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| ProximaError::Config("[[pipe]] requires `name`".into()))?
                    .to_string();
                let mut spec_value = entry.clone();
                if let Some(obj) = spec_value.as_object_mut() {
                    obj.remove("name");
                    obj.insert("name".into(), Value::String(name.clone()));
                }
                let handle = load(spec_value, &self.load_context).await?;
                self.pipes.insert(name, handle);
            }
        }

        // bind each listener with its OWN per-listener router so
        // mounts on listener A don't leak to listener B.
        let mut handles = Vec::new();
        if let Some(listens) = table.get("listen").and_then(Value::as_array) {
            for listen in listens {
                let handle = self.bind_listener(listen)?;
                handles.push(handle);
            }
        } else {
            return Err(ProximaError::Config(
                "load_full found no `[[listen]]` blocks; use App::pipe / App::mount / App::run_until_signal for single-listener configs".into(),
            ));
        }
        Ok(handles)
    }

    fn bind_listener(&self, listen: &Value) -> Result<ListenerHandle, ProximaError> {
        let protocol_name = listen
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("http")
            .to_string();
        let bind_str = listen
            .get("bind")
            .and_then(Value::as_str)
            .ok_or_else(|| ProximaError::Config("[[listen]] requires `bind`".into()))?;
        let bind: SocketAddr = bind_str
            .parse()
            .map_err(|err| ProximaError::Config(format!("invalid bind '{bind_str}': {err}")))?;

        // build a per-listener router so mounts stay local to this
        // listener (nginx server { } block semantics).
        let router = self.build_listener_router(&protocol_name, listen)?;
        let dispatch: PipeHandle = into_handle(ContextInjector::new(
            into_handle(RouterDispatch {
                router: Arc::new(ArcSwap::from_pointee(router)),
            }),
            self.load_context.telemetry.clone(),
        ));

        let listener_spec = ListenerSpec {
            bind,
            protocol_name,
            shutdown: proxima_listen::handle::ShutdownPolicy::drain_30s(),
            spec: listen.clone(),
            #[cfg(feature = "tls")]
            tls: None,
        };
        listener_spec.attach(dispatch).run_with_runtime(
            &self.listen_registry,
            self.load_context.telemetry.clone(),
            self.runtime.clone(),
            self.acceptor_factory.clone(),
            None,
        )
    }

    fn build_listener_router(
        &self,
        protocol_name: &str,
        listen: &Value,
    ) -> Result<Router, ProximaError> {
        let mut router = Router::new();
        match protocol_name {
            "http" => {
                let mounts = match listen.get("mount") {
                    Some(Value::Array(arr)) => arr.clone(),
                    Some(_) => {
                        return Err(ProximaError::Config(
                            "[[listen.mount]] must be an array".into(),
                        ));
                    }
                    None => {
                        // shorthand: `pipe =` mounts at /{*path}
                        if let Some(name) = listen.get("pipe").and_then(Value::as_str) {
                            let target = self.lookup_pipe(name).ok_or_else(|| {
                                ProximaError::NotFound(format!(
                                    "[[listen]] references unknown pipe '{name}'"
                                ))
                            })?;
                            router.add(Mount::new("/{*path}", target));
                        }
                        return Ok(router);
                    }
                };
                for mount in &mounts {
                    let path = mount.get("path").and_then(Value::as_str).ok_or_else(|| {
                        ProximaError::Config("[[listen.mount]] requires `path`".into())
                    })?;
                    let pipe_name = mount.get("pipe").and_then(Value::as_str).ok_or_else(|| {
                        ProximaError::Config("[[listen.mount]] requires `pipe`".into())
                    })?;
                    let target = self.lookup_pipe(pipe_name).ok_or_else(|| {
                        ProximaError::NotFound(format!(
                            "[[listen.mount]] references unknown pipe '{pipe_name}'"
                        ))
                    })?;
                    let methods = mount
                        .get("methods")
                        .and_then(Value::as_array)
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(str::to_string))
                                .collect::<Vec<_>>()
                        })
                        .map(MethodFilter::only)
                        .unwrap_or_else(MethodFilter::any);
                    let host = mount
                        .get("host")
                        .and_then(Value::as_array)
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(str::to_string))
                                .collect::<Vec<_>>()
                        })
                        .map(crate::mount::HostFilter::only)
                        .unwrap_or_else(crate::mount::HostFilter::any);
                    router.add(
                        Mount::new(path, target)
                            .with_methods(methods)
                            .with_host(host),
                    );
                }
            }
            _ => {
                // non-http listener: single target via `pipe =`
                let pipe_name = listen.get("pipe").and_then(Value::as_str).ok_or_else(|| {
                    ProximaError::Config(format!(
                        "[[listen]] type='{protocol_name}' requires `pipe`"
                    ))
                })?;
                let target = self.lookup_pipe(pipe_name).ok_or_else(|| {
                    ProximaError::NotFound(format!(
                        "[[listen]] references unknown pipe '{pipe_name}'"
                    ))
                })?;
                router.add(Mount::new("/{*path}", target));
            }
        }
        Ok(router)
    }
}

/// Fluent builder returned by [`App::pipe_builder`]. Buffers the
/// pipe spec + mount path + optional method filter; `build()` calls
/// the existing `App::pipe` and `App::mount` machinery so config-
/// driven and API-driven pipes go through identical load paths.
///
/// The chain reads top-down: configure, then `.build().await?`.
#[must_use = "AppPipeBuilder is inert until .build().await is called"]
pub struct AppPipeBuilder<'app> {
    app: &'app mut App,
    name: String,
    spec: Option<Spec>,
    mount: Option<String>,
    methods: Option<crate::mount::MethodFilter>,
}

impl<'app> AppPipeBuilder<'app> {
    /// Set the pipe's spec (typically a `Composable::then` chain or a
    /// settings struct). Accepts the same `impl Into<Spec>` shape as
    /// `App::pipe(name, spec)`.
    pub fn dispatch(mut self, spec: impl Into<Spec>) -> Self {
        self.spec = Some(spec.into());
        self
    }

    /// Mount the pipe at `path`. If omitted, the pipe is registered
    /// in the App but not exposed on the router; downstream code can
    /// `app.lookup_pipe(name)` and mount later.
    pub fn mount(mut self, path: impl Into<String>) -> Self {
        self.mount = Some(path.into());
        self
    }

    /// Restrict the mount to specific HTTP methods. No-op when
    /// `.mount(...)` is not also set.
    pub fn methods(mut self, filter: crate::mount::MethodFilter) -> Self {
        self.methods = Some(filter);
        self
    }

    /// Register the pipe and mount it. Returns the `PipeHandle` so
    /// callers can pass it elsewhere (e.g., `SwappablePipe`).
    ///
    /// If a pipe with this name is already registered in the App,
    /// `build()` routes through [`App::update_pipe`] — in-flight
    /// requests on the old handle finish, new requests dispatch to
    /// the new handle. Otherwise routes through [`App::pipe`] for a
    /// fresh registration.
    pub async fn build(self) -> Result<PipeHandle, ProximaError> {
        let Self {
            app,
            name,
            spec,
            mount,
            methods,
        } = self;
        let spec = spec.ok_or_else(|| {
            ProximaError::Config(format!(
                "AppPipeBuilder('{name}'): dispatch(...) is required before build()"
            ))
        })?;
        let handle = if app.pipes.contains_key(&name) {
            app.update_pipe(&name, spec).await?
        } else {
            app.pipe(name.clone(), spec).await?
        };
        if let Some(path) = mount {
            let target = MountTarget::Named(name);
            match methods {
                Some(filter) => app.mount_with_methods(&path, target, filter)?,
                None => app.mount(&path, target)?,
            }
        }
        Ok(handle)
    }
}

pub enum MountTarget {
    Handle(PipeHandle),
    Named(String),
}

/// Disjoint-dispatch marker for [`IntoMountTarget`]'s registered-pipe-name
/// arm (`&str` / `String`).
pub struct ViaName;

/// Disjoint-dispatch marker for the handler-shaped-pipe arm (`H: Handler`) —
/// covers a `#[proxima::piped(send)]` struct, a [`PipeHandle`], or any other
/// hand-written `Handler` impl.
pub struct ViaPipe;

/// Disjoint-dispatch marker for the bare request-shaped `async fn` arm —
/// `Fn(Request<Bytes>) -> Fut`, no `#[proxima::piped]` involved.
pub struct ViaFn;

/// Disjoint-dispatch marker for an already-built [`MountTarget`] passed
/// through unchanged (`MountTarget::Handle(..)` / `MountTarget::Named(..)`
/// constructed directly, e.g. by the daemon control plane).
pub struct ViaTarget;

/// What [`App::mount`] (and its `_with_methods`/`_full` siblings) accept.
/// `Via` is a phantom marker parameter, never named by a caller — it exists
/// so the four input shapes below are DISJOINT trait instantiations
/// (`IntoMountTarget<ViaName>` vs. `IntoMountTarget<ViaPipe>` vs. ...) rather
/// than one blanket impl per shape competing for the same `Self`, which
/// would coherence-conflict (E0119) the moment a `PipeHandle` (itself a
/// `Handler`) tried to satisfy both a by-name and a by-handle blanket at
/// once. Because each `Via` is a distinct concrete type, the compiler never
/// needs to prove non-overlap between the impls below — it's true by
/// construction.
#[diagnostic::on_unimplemented(
    message = "`{Self}` can't be mounted: expected a request handler — an `async fn(Request<Bytes>) -> Result<Response<Bytes>, ProximaError>`, a handler-shaped pipe, or a registered pipe name",
    label = "not something `App::mount` can dispatch to"
)]
pub trait IntoMountTarget<Via> {
    fn into_mount_target(self) -> MountTarget;
}

impl IntoMountTarget<ViaName> for &str {
    fn into_mount_target(self) -> MountTarget {
        MountTarget::Named(self.to_string())
    }
}

impl IntoMountTarget<ViaName> for String {
    fn into_mount_target(self) -> MountTarget {
        MountTarget::Named(self)
    }
}

impl IntoMountTarget<ViaTarget> for MountTarget {
    fn into_mount_target(self) -> MountTarget {
        self
    }
}

impl<Implementor> IntoMountTarget<ViaPipe> for Implementor
where
    Implementor: Handler + 'static,
{
    fn into_mount_target(self) -> MountTarget {
        MountTarget::Handle(into_handle(self))
    }
}

/// Wraps a bare `Fn(Request<Bytes>) -> Fut` in the [`SendPipe`] shape so
/// [`IntoMountTarget`]'s `ViaFn` arm can erase it the same way `ViaPipe`
/// erases a handler-shaped pipe. The sanctioned narrow app-edge blanket this
/// design needs (never library machinery) — a plain fn item or closure has
/// no other way to reach `Handler` without a struct to carry it.
struct FnHandler<Implementor>(Implementor);

impl<Implementor, Fut> SendPipe for FnHandler<Implementor>
where
    Implementor: Fn(Request<Bytes>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Response<Bytes>, ProximaError>> + Send + 'static,
{
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        (self.0)(request)
    }
}

impl<Implementor, Fut> IntoMountTarget<ViaFn> for Implementor
where
    Implementor: Fn(Request<Bytes>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Response<Bytes>, ProximaError>> + Send + 'static,
{
    fn into_mount_target(self) -> MountTarget {
        MountTarget::Handle(into_handle(FnHandler(self)))
    }
}

/// Handle returned by `App::run_until_signal`. Calling `stop()` (or
/// letting Drop fire) signals the listener to break its accept loop.
/// Subsequent commits add `drain()` / `wait_for_drain()` /
/// `graceful_shutdown(timeout)` for two-phase drain semantics.
/// Grace period `Shutdown::drain` gives registered sources to observe
/// cancellation and return cleanly before their tasks are aborted.
const SOURCE_SHUTDOWN_GRACE: std::time::Duration = std::time::Duration::from_secs(5);

pub struct Shutdown {
    shutdown_tx: Option<oneshot::Sender<()>>,
    /// fires when the listener's serve() future returns (after quiesce + drain).
    /// `drain()` awaits this before broadcasting the per-core drop signal so
    /// drop hooks observe an empty in-flight set.
    drain_notify: Arc<proxima_primitives::sync::Notify>,
    /// keep the per-core runtime alive while this handle exists. Without
    /// this, dropping `App` would drop the only `Arc<Runtime>` ref and
    /// terminate the workers serving the listener.
    _runtime: Option<Arc<dyn crate::runtime::Runtime>>,
    /// owns every source `run_until_signal` spawned (TARGET 4). `None` for
    /// the test-only constructors, which never register sources.
    source_lifecycle: Option<proxima_primitives::pipe::ProducerLifecycle>,
    /// the same cancellation `Signal` `source_lifecycle` was spawned with —
    /// `stop()` fires this too, so `.stop()` cancels sources even though it
    /// doesn't await their drain (that's what `.drain()` is for).
    source_cancel: proxima_core::signal::Signal,
}

impl Shutdown {
    /// Signal the listener to stop accepting and break its loop, and fire
    /// the source cancellation signal. In-flight requests continue to
    /// completion (or are cancelled by their own cancel Signal if the
    /// connection-close guard fires); sources observe `source_cancel`
    /// cooperatively and return on their own schedule.
    pub fn stop(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        self.source_cancel.fire();
    }

    /// Test-only constructor. Builds a `Shutdown` whose `stop()` fires
    /// the supplied oneshot, with no live listener loop behind it.
    #[cfg(test)]
    pub(crate) fn for_test_with_tx(tx: oneshot::Sender<()>) -> Self {
        Self {
            shutdown_tx: Some(tx),
            drain_notify: Arc::new(proxima_primitives::sync::Notify::new()),
            _runtime: None,
            source_lifecycle: None,
            source_cancel: proxima_core::signal::Signal::new(),
        }
    }

    /// Test-only constructor when the caller only cares that a
    /// `Shutdown` exists (e.g. exercising `Server::clone` paths).
    /// Receiving end of the channel is provided to avoid leaking
    /// the unused sender as a permanent receive-error trigger.
    #[cfg(test)]
    pub(crate) fn for_test(_rx: oneshot::Receiver<()>) -> Self {
        let (tx, _rx_unused) = oneshot::channel();
        Self {
            shutdown_tx: Some(tx),
            drain_notify: Arc::new(proxima_primitives::sync::Notify::new()),
            _runtime: None,
            source_lifecycle: None,
            source_cancel: proxima_core::signal::Signal::new(),
        }
    }

    /// Full shutdown: signal the listener, then broadcast the per-core
    /// drop signal so Pipes holding `!Send` state in their core's
    /// thread_local cleanup stack (`register_per_core_resource`) fire
    /// their hooks in LIFO order, then drain every registered source.
    /// Returns a `ShutdownReport` with per-core ack counts, total hooks
    /// drained, and the source drain/abort counts folded in.
    ///
    /// When no `Runtime` is installed (non-`runtime-tokio` build),
    /// `drain` degenerates to `stop` and reports zero acks.
    pub async fn drain(mut self) -> crate::shutdown::ShutdownReport {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        // Phase 1 + 2: wait for the listener to finish quiesce +
        // drain. Returns when serve() completes (in-flight = 0,
        // listener dropped, port released).
        self.drain_notify.notified().await;
        // Phase 3: broadcast per-core drop. Now safe to drop !Send
        // resources because no in-flight request can still touch
        // them.
        let mut report = match self._runtime.take() {
            Some(runtime) => {
                let barrier = crate::shutdown::ShutdownBarrier::new(runtime);
                barrier.broadcast_drop().await
            }
            None => crate::shutdown::ShutdownReport {
                cores_acked: 0,
                hooks_drained: 0,
                sources_drained: 0,
                sources_aborted: 0,
            },
        };
        // Phase 4: drain every registered source under the same grace
        // window. Cooperative — each source observes `source_cancel`
        // (already fired above) and returns on its own.
        if let Some(source_lifecycle) = self.source_lifecycle.take() {
            let source_report = source_lifecycle.shutdown(SOURCE_SHUTDOWN_GRACE).await;
            report.sources_drained = source_report.drained;
            report.sources_aborted = source_report.aborted;
        }
        report
    }
}

impl Drop for Shutdown {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        self.source_cancel.fire();
    }
}

/// Flatten a `RegistryEntry { type, spec }` into a `Value` shape the
/// existing factory loader recognizes — `{ "type": "<tag>", ...spec
/// fields }`. The factory dispatch in `load.rs` keys off the `type`
/// field for the generic registry-entry path.
fn registry_entry_value(entry: &crate::settings::RegistryEntry) -> Value {
    let mut map = match &entry.spec {
        Value::Object(m) => m.clone(),
        other => {
            let mut m = serde_json::Map::new();
            m.insert("value".into(), other.clone());
            m
        }
    };
    map.insert("type".into(), Value::String(entry.r#type.clone()));
    Value::Object(map)
}

/// Compose a pipe entry's chain into the `{ <leaf_fields>,
/// middleware: [...] }` shape the existing `apply_middleware_stack`
/// dispatch understands. The `chain` field is an ordered list of
/// names; the LAST name is the leaf upstream, earlier names are
/// middlewares wrapping it (outer-first, matching the convention
/// established in Phase 4's `Chain`).
fn compose_pipe_spec(
    entry: &crate::settings::RegistryEntry,
    settings: &crate::settings::ProximaSettings,
) -> Result<Value, ProximaError> {
    let mut map = match &entry.spec {
        Value::Object(m) => m.clone(),
        _ => {
            return Err(ProximaError::Config(
                "pipe entry spec must be an object with `mount` + `chain`".into(),
            ));
        }
    };
    let chain_names: Vec<String> = map
        .get("chain")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    if chain_names.is_empty() {
        // No chain field — the entry stands alone. Return the spec as-is
        // with the type tag for the loader.
        map.insert("type".into(), Value::String(entry.r#type.clone()));
        return Ok(Value::Object(map));
    }
    // Resolve each name to its typed RegistryEntry from upstream or
    // middleware sections. Last name = leaf upstream; earlier names
    // = middlewares wrapping it.
    let (leaf_name, mw_names) = chain_names
        .split_last()
        .ok_or_else(|| ProximaError::Config("pipe chain must not be empty".into()))?;
    let leaf_entry = settings.upstreams.get(leaf_name).ok_or_else(|| {
        ProximaError::Config(format!(
            "pipe chain leaf `{leaf_name}` not found in [upstream.*]"
        ))
    })?;
    let mut leaf_map = match &leaf_entry.spec {
        Value::Object(m) => m.clone(),
        _ => serde_json::Map::new(),
    };
    leaf_map.insert("type".into(), Value::String(leaf_entry.r#type.clone()));
    let mw_array: Vec<Value> = mw_names
        .iter()
        .map(|name| {
            let mw_entry = settings.middlewares.get(name).ok_or_else(|| {
                ProximaError::Config(format!(
                    "pipe chain middleware `{name}` not found in [middleware.*]"
                ))
            })?;
            Ok(registry_entry_value(mw_entry))
        })
        .collect::<Result<_, ProximaError>>()?;
    if !mw_array.is_empty() {
        leaf_map.insert("middleware".into(), Value::Array(mw_array));
    }
    // Preserve any extra fields from the pipe entry that aren't
    // chain-related (e.g. mount, methods — those are consumed by the
    // caller after registration).
    map.remove("chain");
    for (k, v) in &map {
        leaf_map.entry(k.clone()).or_insert(v.clone());
    }
    Ok(Value::Object(leaf_map))
}

#[derive(Clone)]
pub struct RunConfig {
    pub bind: SocketAddr,
    pub protocol: String,
    pub spec: Value,
}

impl RunConfig {
    #[must_use]
    pub fn http(bind: SocketAddr) -> Self {
        Self {
            bind,
            protocol: "http".into(),
            spec: Value::Null,
        }
    }
}

struct RouterDispatch {
    router: Arc<ArcSwap<Router>>,
}

impl SendPipe for RouterDispatch {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        mut request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let router = self.router.load_full();
        async move {
            let routed = router
                .route_with_params(&request)
                .map(|(mount, params)| (mount.clone(), params));
            let Some((mount, params)) = routed else {
                let telemetry = request.context.telemetry.clone();
                let labels = crate::telemetry::Labels::from_pairs(&[
                    ("pipe", "__unrouted__"),
                    ("status_class", "4xx"),
                ]);
                telemetry.counter_inc("proxima.requests_total", &labels, 1);
                return Ok(Response::not_found().with_body("no route matched"));
            };
            request.context.path_params = params;
            if request.context.pipe_label.is_none() {
                request.context.pipe_label = Some(Arc::from(mount.pipe_name.as_bytes()));
            }
            let started = std::time::Instant::now();
            let telemetry = request.context.telemetry.clone();
            let labels = request.context.metric_labels(&[]);
            telemetry.counter_inc("proxima.requests_total", &labels, 1);
            let outcome = SendPipe::call(&mount.pipe, request).await;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
            telemetry.histogram_record("proxima.request.latency_ms", &labels, elapsed_ms);
            outcome
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn build_request(method: &str, path: &str) -> Request<Bytes> {
        Request::builder()
            .method(method)
            .path(path)
            .build()
            .expect("builder")
    }

    #[proxima::test]
    async fn registered_pipe_can_be_mounted_by_name() {
        let mut app = App::new().expect("app");
        app.pipe(
            "cache",
            json!({"kv": "cache", "ttl": "1h", "max_entries": 100}),
        )
        .await
        .expect("register");
        app.mount("/users/{id}", "cache").expect("mount");
        let dispatch = app.router_handle();
        let outcome = SendPipe::call(&dispatch, build_request("GET", "/users/42")).await;
        assert!(
            matches!(outcome, Err(ProximaError::NoData)),
            "kv miss propagates as NoData"
        );
    }

    #[proxima::test]
    async fn unmatched_path_returns_404() {
        let mut app = App::new().expect("app");
        app.pipe("cache", json!({"kv": "cache", "max_entries": 100}))
            .await
            .expect("register");
        app.mount("/foo", "cache").expect("mount");
        let dispatch = app.router_handle();
        let response = SendPipe::call(&dispatch, build_request("GET", "/bar"))
            .await
            .expect("dispatch");
        assert_eq!(response.status, 404);
    }

    #[proxima::test]
    async fn mount_by_unknown_name_returns_not_found() {
        let app = App::new().expect("app");
        let outcome = app.mount("/foo", "missing");
        assert!(matches!(outcome, Err(ProximaError::NotFound(_))));
    }

    // a hand-written `Handler`-shaped pipe (not macro-generated, not
    // pre-erased) — the `ViaPipe` arm's target shape.
    struct Echo;

    impl SendPipe for Echo {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl std::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send
        {
            async move {
                let (_, body) = request.body_bytes().await?;
                Ok(Response::ok(body))
            }
        }
    }

    #[proxima::test]
    async fn mount_erases_and_mounts_a_handler_shaped_pipe_in_one_call() {
        let app = App::new().expect("app");
        app.mount("/echo", Echo).expect("mount");

        let dispatch = app.router_handle();
        let request = Request::builder()
            .method("POST")
            .path("/echo")
            .body(Bytes::from_static(b"hello"))
            .build()
            .expect("builder");
        let response = SendPipe::call(&dispatch, request).await.expect("dispatch");

        assert_eq!(response.status, 200);
        let body = response.collect_body().await.expect("body");
        assert_eq!(&body[..], b"hello", "mount reaches the erased handler");
    }

    #[proxima::test]
    async fn mount_with_methods_restricts_a_handler_shaped_pipe() {
        let app = App::new().expect("app");
        app.mount_with_methods("/echo", Echo, MethodFilter::only(["POST".to_string()]))
            .expect("mount_with_methods");

        let dispatch = app.router_handle();
        let response = SendPipe::call(&dispatch, build_request("GET", "/echo"))
            .await
            .expect("dispatch");
        assert_eq!(
            response.status, 404,
            "GET is outside the mounted method filter"
        );
    }

    // bare request-shaped `async fn`, no `#[proxima::piped]` involved — the
    // `ViaFn` arm's target shape. Proves the same `mount` call that accepts
    // `Echo` (a pipe) and `"cache"` (a name, see the tests above) also
    // accepts this third, disjoint shape.
    async fn echo_fn(request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        let (_, body) = request.body_bytes().await?;
        Ok(Response::ok(body))
    }

    #[proxima::test]
    async fn mount_erases_and_mounts_a_bare_async_fn() {
        let app = App::new().expect("app");
        app.mount("/echo-fn", echo_fn).expect("mount");

        let dispatch = app.router_handle();
        let request = Request::builder()
            .method("POST")
            .path("/echo-fn")
            .body(Bytes::from_static(b"world"))
            .build()
            .expect("builder");
        let response = SendPipe::call(&dispatch, request).await.expect("dispatch");

        assert_eq!(response.status, 200);
        let body = response.collect_body().await.expect("body");
        assert_eq!(&body[..], b"world", "mount reaches the bare fn via ViaFn");
    }

    #[proxima::test]
    async fn pipe_builder_registers_and_mounts_in_one_chain() {
        let mut app = App::new().expect("app");
        let handle = app
            .pipe_builder("cache")
            .dispatch(json!({"kv": "cache", "max_entries": 100}))
            .mount("/u/{id}")
            .build()
            .await
            .expect("build");
        // the raw handle no longer carries a queryable name (TARGET 3 —
        // name lives at the registry key / mount-site label, not the pipe);
        // the registration below is the behavioral proof.
        let _ = &handle;
        assert!(app.lookup_pipe("cache").is_some());
        let dispatch = app.router_handle();
        let outcome = SendPipe::call(&dispatch, build_request("GET", "/u/42")).await;
        assert!(
            matches!(outcome, Err(ProximaError::NoData)),
            "mount-by-builder reaches the same kv path"
        );
    }

    #[proxima::test]
    async fn pipe_builder_dispatch_required() {
        let mut app = App::new().expect("app");
        let outcome = app.pipe_builder("api").mount("/api").build().await;
        let err = match outcome {
            Ok(_) => panic!("dispatch missing must error"),
            Err(err) => err,
        };
        let message = format!("{err}");
        assert!(message.contains("dispatch(...)"), "got: {message}");
    }

    #[proxima::test]
    async fn pipe_builder_without_mount_registers_only() {
        let mut app = App::new().expect("app");
        app.pipe_builder("unmounted")
            .dispatch(json!({"kv": "cache", "max_entries": 8}))
            .build()
            .await
            .expect("build");
        assert!(app.lookup_pipe("unmounted").is_some());
        // No mount: the router has no path for it.
        let dispatch = app.router_handle();
        let response = SendPipe::call(&dispatch, build_request("GET", "/unmounted"))
            .await
            .expect("dispatch");
        assert_eq!(response.status, 404);
    }

    #[proxima::test]
    async fn lookup_pipe_spec_returns_resolved_value_after_registration() {
        let mut app = App::new().expect("app");
        app.pipe("svc", json!({"kv": "cache", "max_entries": 8}))
            .await
            .expect("register");
        let spec = app.lookup_pipe_spec("svc").expect("spec stored");
        assert_eq!(spec.get("kv").and_then(Value::as_str), Some("cache"));
        assert_eq!(
            spec.get("name").and_then(Value::as_str),
            Some("svc"),
            "name field gets injected at register time",
        );
    }

    #[proxima::test]
    async fn lookup_pipe_spec_returns_none_for_unregistered_name() {
        let app = App::new().expect("app");
        assert!(app.lookup_pipe_spec("ghost").is_none());
    }

    #[proxima::test]
    async fn pipe_builder_from_existing_pre_fills_spec_and_routes_through_update() {
        let mut app = App::new().expect("app");
        // 1. register original
        let original = app
            .pipe("svc", json!({"kv": "cache", "max_entries": 4}))
            .await
            .expect("register");
        let original_pipes_count = app.pipes.len();

        // 2. modify via the builder; default dispatch reuses the
        //    pre-filled spec (no .dispatch() call needed)
        let modified = app
            .pipe_builder_from_existing("svc")
            .expect("svc is registered")
            .build()
            .await
            .expect("rebuild");

        // pipe count unchanged (update_pipe path, not pipe path)
        assert_eq!(app.pipes.len(), original_pipes_count);
        // the new handle is the one stored in App now
        assert!(Arc::ptr_eq(
            &modified,
            &app.lookup_pipe("svc").expect("still registered"),
        ));
        // _ = original; the old handle stays valid for in-flight users
        let _ = original;
    }

    #[proxima::test]
    async fn pipe_builder_from_existing_with_dispatch_swap_replaces_spec() {
        let mut app = App::new().expect("app");
        app.pipe("svc", json!({"kv": "cache", "max_entries": 4}))
            .await
            .expect("register");

        app.pipe_builder_from_existing("svc")
            .expect("registered")
            .dispatch(json!({"kv": "cache", "max_entries": 64}))
            .build()
            .await
            .expect("rebuild");

        let stored = app.lookup_pipe_spec("svc").expect("spec stored");
        assert_eq!(
            stored.get("max_entries").and_then(Value::as_u64),
            Some(64),
            "dispatch(...) on the builder overrode the pre-filled spec",
        );
    }

    #[proxima::test]
    async fn pipe_builder_from_existing_returns_none_for_unregistered_name() {
        let mut app = App::new().expect("app");
        assert!(app.pipe_builder_from_existing("ghost").is_none());
    }

    #[proxima::test]
    async fn pipe_inherits_registration_name_when_spec_omits_it() {
        let mut app = App::new().expect("app");
        let handle = app
            .pipe("echo-cached", json!({"kv": "cache", "max_entries": 10}))
            .await
            .expect("register");
        // same as above: the raw handle carries no name; registration under
        // "echo-cached" is the observable behavior.
        assert!(app.lookup_pipe("echo-cached").is_some());
        let _ = &handle;
    }

    // same condition as `default_runtime`/`runtime_cores`: this test asserts
    // a runtime got installed, which only happens when one is compiled in.
    // also needs `http1`: it binds a real HTTP listener via `RunConfig::http`.
    #[cfg(all(
        feature = "http1",
        any(
            feature = "runtime-tokio",
            all(
                feature = "serve-prime",
                feature = "runtime-prime-reactor",
                any(target_os = "linux", target_os = "macos")
            )
        )
    ))]
    #[proxima::test]
    async fn shutdown_drain_fires_per_core_resource_hooks() {
        use std::sync::atomic::{AtomicU64, Ordering};

        let app = App::new().expect("app");
        let runtime = app.runtime().expect("default runtime installed");
        let drops_observed = Arc::new(AtomicU64::new(0));

        // Register a per-core cleanup hook on every worker core by
        // dispatching a registration factory to each core. The closure
        // pushes onto that core's thread_local stack.
        for core_index in 0..runtime.num_cores() {
            let drops_for_core = drops_observed.clone();
            runtime
                .spawn_factory_on_core(
                    crate::runtime::CoreId(core_index),
                    Box::new(move || {
                        Box::pin(async move {
                            crate::shutdown::register_per_core_resource(
                                format!("core-{core_index}"),
                                Box::new(move || {
                                    drops_for_core.fetch_add(1, Ordering::SeqCst);
                                }),
                            );
                        })
                    }),
                )
                .expect("test-time spawn must succeed on a fresh runtime");
        }

        // Give the registration tasks a moment to land on their cores.
        proxima_core::time::sleep(std::time::Duration::from_millis(50)).await;

        // Bind a no-op TCP HTTP listener so run_until_signal has
        // something to stop. The drain test only cares about
        // shutdown plumbing — any registered listener satisfies it.
        let config = RunConfig::http("127.0.0.1:0".parse().expect("addr"));
        let shutdown = app.run_until_signal(config).await.expect("run");

        let report = shutdown.drain().await;

        // Every core's hook must have fired exactly once.
        let expected = runtime.num_cores();
        assert_eq!(report.cores_acked, expected, "every core must ack");
        assert_eq!(report.hooks_drained, expected, "every hook must fire");
        assert_eq!(
            drops_observed.load(Ordering::SeqCst) as usize,
            expected,
            "destructor instrumentation must observe one drop per core",
        );
    }
}
