// SpawnRequest now lives in proxima-runtime as a generic enum
// (`SpawnRequest<Inline = Infallible>`) so the prime crate can be
// extracted without proxima-runtime depending on prime. Tokio's
// impl uses `SpawnRequest<Infallible>`; prime uses
// `SpawnRequest<InlineTask>`. See proxima-runtime/src/lib.rs.

// this module IS the backend-naming layer: the `run*` family is the only place
// `run_prime`/`run_tokio` are legitimately named and adaptively composed. the
// `disallowed_methods` guardrail (clippy.toml) protects CALLERS of this module,
// not the drivers' own delegation, so it is allowed here.
#![allow(clippy::disallowed_methods)]

use std::future::Future;
use std::sync::{Arc, OnceLock};

use proxima_core::ProximaError;
use proxima_primitives::stream::AcceptorFactory;

#[cfg(feature = "runtime-prime")]
pub use prime;
#[cfg(feature = "runtime-tokio")]
pub use proxima_runtime::tokio as tokio_per_core;

#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
pub use prime::os::runtime::PrimeRuntime;
#[cfg(feature = "rayon")]
pub use proxima_runtime::RayonBackgroundPool;
#[cfg(feature = "runtime-tokio")]
pub use tokio_per_core::TokioPerCoreRuntime;

// Runtime trait + value types (CoreId, BackgroundPool, BackgroundHandle, SpawnError,
// SpawnRequest, spawn_on_core_blocking_with) re-exported from proxima-runtime
// so existing call sites using `crate::runtime::Runtime` etc. keep working.
pub use proxima_runtime::*;

// ---------------------------------------------------------------------------
// ambient runtime: the App-adoption seam
// ---------------------------------------------------------------------------
//
// `#[proxima::main]` is the control surface for which runtime backend boots
// and how many cores it gets (`runtime = "prime"|"tokio"`, `cores = N`,
// `affinity = "..."`). Before this seam, `App::new()` had no way to learn
// that choice — it unconditionally built its OWN runtime sized off a raw env
// var read, so `#[proxima::main(cores = 1)]` booted a 1-core runtime to drive
// main's body while `App::new()` inside that body booted a SECOND,
// independent runtime at `num_cpus::get()`. Two runtimes, contradictory core
// counts, one process.
//
// `install_runtime` publishes the runtime (and its matching acceptor
// factory) that a `run*` driver actually booted; `App::new()` adopts it
// via `installed_runtime()` instead of building an independent default.
// Set-once (a process boots one main): a second call is a no-op — the first
// runtime installed is the one main's body actually runs on, so it's the
// only one worth adopting.
static INSTALLED_RUNTIME: OnceLock<InstalledRuntime> = OnceLock::new();

/// A runtime + the acceptor factory that matches its transport (prime pairs
/// with `PrimeAcceptorFactory`, tokio with `TokioAcceptorFactory`) —
/// published together so an adopter never ends up with one backend's
/// runtime and another's acceptor.
#[derive(Clone)]
pub struct InstalledRuntime {
    pub runtime: Arc<dyn Runtime>,
    pub acceptor_factory: Arc<dyn AcceptorFactory>,
}

/// Publish the runtime `#[proxima::main]` (or any other `run*` driver)
/// booted, so `App::new()` can adopt it instead of building an independent
/// second runtime. Set-once — a later call is ignored.
pub fn install_runtime(runtime: Arc<dyn Runtime>, acceptor_factory: Arc<dyn AcceptorFactory>) {
    let _ = INSTALLED_RUNTIME.set(InstalledRuntime {
        runtime,
        acceptor_factory,
    });
}

/// The runtime installed by a `run*` driver, if one has run in this
/// process yet. `App::new()` adopts this when present; falls back to its own
/// config-resolved default otherwise (e.g. `App::new()` called outside a
/// `#[proxima::main]`-driven binary, or in a test).
#[must_use]
pub fn installed_runtime() -> Option<InstalledRuntime> {
    INSTALLED_RUNTIME.get().cloned()
}

// ---------------------------------------------------------------------------
// prime-native serve adapter
// ---------------------------------------------------------------------------
//
// Folded in from the former proxima-runtime-prime crate (FOLD 2 of the
// runtime backend consolidation) — it could not become a `proxima-runtime`
// feature the way the tokio backend did: prime/http/listen -> runtime would
// cycle back into `proxima-runtime` itself. The umbrella already depended on
// prime/http/listen/tls/net/primitives/runtime, so inlining here adds zero
// new edges.
//
// `PrimeServeExt` wraps `Listener::run_with_runtime` with `HttpListenProtocol`
// + `PrimeAcceptorFactory` so callers get the same one-liner
// `runtime.serve_http(addr, pipe)` surface as before, routed through the
// agnostic listener stack instead of a bespoke accept loop.

#[cfg(all(
    any(feature = "http1", feature = "http1-native"),
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
use std::net::SocketAddr;

#[cfg(all(
    any(feature = "http1", feature = "http1-native"),
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
use proxima_http::listener::HttpListenProtocol;
#[cfg(all(
    any(feature = "http1", feature = "http1-native"),
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
use proxima_listen::handle::ListenerHandle;
#[cfg(all(
    any(feature = "http1", feature = "http1-native"),
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
use proxima_listen::{ListenRegistry, ListenerSpec};
#[cfg(all(
    any(feature = "http1", feature = "http1-native"),
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
use proxima_primitives::pipe::handler::PipeHandle;
#[cfg(all(
    any(feature = "http1", feature = "http1-native"),
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
use proxima_primitives::pipe::telemetry_surface::NoopTelemetry;

#[cfg(all(
    any(feature = "http1", feature = "http1-native"),
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
pub trait PrimeServeExt {
    fn serve_http(
        self: &Arc<Self>,
        bind: SocketAddr,
        pipe: PipeHandle,
    ) -> Result<ListenerHandle, ProximaError>;

    #[cfg(feature = "tls")]
    fn serve_https(
        self: &Arc<Self>,
        bind: SocketAddr,
        cert: std::path::PathBuf,
        key: std::path::PathBuf,
        pipe: PipeHandle,
    ) -> Result<ListenerHandle, ProximaError>;

    #[cfg(feature = "tls")]
    fn serve_https_with_tls(
        self: &Arc<Self>,
        bind: SocketAddr,
        tls: proxima_tls::TlsConfig,
        pipe: PipeHandle,
    ) -> Result<ListenerHandle, ProximaError>;
}

#[cfg(all(
    any(feature = "http1", feature = "http1-native"),
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
fn make_http_registry() -> Result<ListenRegistry, ProximaError> {
    let registry = ListenRegistry::new();
    registry.register(Arc::new(HttpListenProtocol::new()))?;
    Ok(registry)
}

#[cfg(all(
    any(feature = "http1", feature = "http1-native"),
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
impl PrimeServeExt for PrimeRuntime {
    fn serve_http(
        self: &Arc<Self>,
        bind: SocketAddr,
        pipe: PipeHandle,
    ) -> Result<ListenerHandle, ProximaError> {
        let registry = make_http_registry()?;
        let runtime: Arc<dyn proxima_runtime::Runtime> = self.clone();
        let acceptor = Arc::new(proxima_net::prime::PrimeAcceptorFactory);
        ListenerSpec::http(bind).attach(pipe).run_with_runtime(
            &registry,
            NoopTelemetry::handle(),
            Some(runtime),
            Some(acceptor),
            None,
        )
    }

    #[cfg(feature = "tls")]
    fn serve_https(
        self: &Arc<Self>,
        bind: SocketAddr,
        cert: std::path::PathBuf,
        key: std::path::PathBuf,
        pipe: PipeHandle,
    ) -> Result<ListenerHandle, ProximaError> {
        let tls = proxima_tls::TlsConfig::files(cert, key)?;
        self.serve_https_with_tls(bind, tls, pipe)
    }

    #[cfg(feature = "tls")]
    fn serve_https_with_tls(
        self: &Arc<Self>,
        bind: SocketAddr,
        tls: proxima_tls::TlsConfig,
        pipe: PipeHandle,
    ) -> Result<ListenerHandle, ProximaError> {
        // TLS composes as a `TlsListenProtocol` decorator around the plain
        // `HttpListenProtocol`, carried via `ListenerSpec::protocol` — there
        // is no `ListenerSpec::with_tls`/`.tls` field. `make_http_registry`
        // is unused on this path (the wrapped protocol is carried, not
        // looked up by name) but `run_with_runtime` still takes a registry
        // parameter, so an empty one is passed.
        let registry = ListenRegistry::new();
        let runtime: Arc<dyn proxima_runtime::Runtime> = self.clone();
        let acceptor = Arc::new(proxima_net::prime::PrimeAcceptorFactory);
        let protocol: Arc<dyn proxima_listen::ListenProtocol> = Arc::new(
            proxima_listen::TlsListenProtocol::new(Arc::new(HttpListenProtocol::new()), tls),
        );
        ListenerSpec::protocol(bind, protocol)
            .attach(pipe)
            .run_with_runtime(
                &registry,
                NoopTelemetry::handle(),
                Some(runtime),
                Some(acceptor),
                None,
            )
    }
}

// ---------------------------------------------------------------------------
// production run-to-completion
// ---------------------------------------------------------------------------

/// `run_prime` — BOOT a one-core prime runtime, then drive `future` to
/// completion on it (the same drive verb as `PrimeRuntime::block_on` /
/// [`block_on`], but this edge OWNS the runtime it drives on). The `run*`
/// family is the only place a backend is named; everything below composes the
/// `block_on` drive. The production analog of the `#[proxima::test]` prime
/// driver (`proxima_test::drive_prime`), extracted generic over `F::Output`.
/// Composes `PrimeRuntime` + `spawn_on_core` + a `sync_channel` for the
/// thread-park / value-handoff — the same shape as [`block_on`], plus a
/// dedicated driver core so `future` never deadlocks a serving worker.
///
/// Prefer this for the proxima serve/chain path — it runs the future on the
/// same per-core runtime production serves on. See [`run_tokio`] for
/// hyper/axum/`TokioPerCoreRuntime`-shaped bins.
///
/// # Errors
/// Returns `ProximaError` if the prime runtime fails to build, the core 0
/// inbox rejects the dispatch, or the worker drops the completion channel
/// without producing a value.
#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
pub fn run_prime<F>(future: F) -> Result<F::Output, ProximaError>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    run_prime_with_cores(None, None, future)
}

/// Like [`run_prime`], but sizes the prime runtime's worker count and
/// placement explicitly. `cores` is the count (`None` resolves to
/// [`CoreSelection::Auto`](prime::config::CoreSelection::Auto) — all physical
/// cores, matching [`PrimeConfig::default()`](prime::config::PrimeConfig)).
/// `affinity` is the placement grammar `Affinity::from_str` parses
/// (`"float"`/`"packed"`/a bare offset/`"a,b,c"`/`"a-b"`; `None` is `Float` —
/// unpinned, the same default `Affinity` itself carries). An explicit
/// `Affinity::Cores` list also fixes the worker count — passing a `cores`
/// that disagrees with that list's length is an error, not a silent pick.
///
/// Installs the booted runtime ambiently (see `install_runtime` above) —
/// `App::new()` called from inside `future` adopts this exact runtime
/// instead of building an independent second one, which is what makes
/// `#[proxima::main(runtime = "prime", cores = N, affinity = "...")]` (and
/// the adaptive `Default` runtime, via [`run_with_cores`]) mean what
/// they read as: one runtime, sized and placed once.
///
/// Internally boots one EXTRA worker beyond the App-visible placement and
/// runs `future` there, disjoint from the App-visible workers — see
/// [`AdoptedRuntime`]'s doc for why (`Listener::run_with_runtime`'s readiness
/// gate is a genuine OS-thread-blocking wait; running `future` on the same
/// core a listener lane targets deadlocks the one thread that would need to
/// both block and drain that lane). The extra worker is invisible to `App` —
/// `num_cores()` and `cores_acked` report exactly the App-visible count.
///
/// # Errors
/// Same as [`run_prime`], plus `ProximaError` if `affinity` fails to
/// parse or if `cores` conflicts with an explicit `Affinity::Cores` list.
#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
pub fn run_prime_with_cores<F>(
    cores: Option<usize>,
    affinity: Option<&str>,
    future: F,
) -> Result<F::Output, ProximaError>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let placement_affinity = match affinity {
        Some(spec) => {
            <prime::config::Affinity as std::str::FromStr>::from_str(spec).map_err(|error| {
                prime_run_dispatch_error(&format!("invalid affinity `{spec}`: {error}"))
            })?
        }
        None => prime::config::Affinity::Float,
    };
    let visible_cores = match (cores, placement_affinity.fixed_count()) {
        (Some(cores_count), Some(affinity_count)) if cores_count != affinity_count => {
            return Err(prime_run_dispatch_error(&format!(
                "cores = {cores_count} conflicts with affinity {affinity:?}, which names {affinity_count} cores"
            )));
        }
        (_, Some(affinity_count)) => affinity_count,
        (Some(cores_count), None) => cores_count,
        (None, None) => prime::config::CoreSelection::Auto.resolve(),
    }
    .max(1);

    let driver_core = CoreId(visible_cores);
    let mut placement = placement_affinity.placement(visible_cores);
    // the driver core is an implementation detail `App` never sees
    // (`num_cores()` reports `visible_cores`) — reuse the last placed
    // physical core rather than widening a pinned placement by one, so an
    // exact `affinity = "a,b,c"` list stays satisfiable without demanding an
    // extra physical core the caller never listed.
    let driver_physical_core = placement.last().copied().unwrap_or(visible_cores);
    placement.push(driver_physical_core);
    let pin = !matches!(placement_affinity, prime::config::Affinity::Float);

    let inner = Arc::new(build_prime_run_runtime(placement, pin)?);
    install_prime_ambient(&inner, visible_cores);
    let (sender, receiver) = std::sync::mpsc::sync_channel::<F::Output>(1);

    let task = async move {
        let output = future.await;
        let _ = sender.send(output);
    };

    match inner.spawn_on_core(driver_core, Box::pin(task)) {
        Ok(()) => {}
        Err(SpawnError::InboxFull) => {
            return Err(prime_run_dispatch_error(
                "prime driver core inbox full on run_prime dispatch",
            ));
        }
        Err(SpawnError::Disconnected) => {
            return Err(prime_run_dispatch_error(
                "prime driver core disconnected on run_prime dispatch",
            ));
        }
    }

    receiver.recv().map_err(|_| {
        prime_run_dispatch_error("prime worker dropped the run_prime completion channel")
    })
}

#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
fn prime_run_dispatch_error(message: &str) -> ProximaError {
    ProximaError::Io(std::io::Error::other(message))
}

#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
fn build_prime_run_runtime(
    placement: Vec<usize>,
    pin: bool,
) -> Result<PrimeRuntime, ProximaError> {
    #[cfg(feature = "run-prime-tokio-compat")]
    {
        PrimeRuntime::new_inner_placed(placement, pin, true)
    }
    #[cfg(not(feature = "run-prime-tokio-compat"))]
    {
        PrimeRuntime::new_inner_placed(placement, pin, false)
    }
}

/// Wraps a [`PrimeRuntime`] that has one MORE real worker than
/// `visible_cores` — the extra core (index `visible_cores`) drives
/// `#[proxima::main]`'s own body; cores `0..visible_cores` are the pool
/// `App`'s listeners/dispatch address. `num_cores()` reports only
/// `visible_cores`, so `App`'s own core math (`ShutdownBarrier::cores_acked`,
/// `Listener::run_with_runtime`'s lane count) sees exactly the count
/// `#[proxima::main(cores = N)]` asked for — the driver core is an
/// implementation detail, not part of the pool `App` reasons about.
///
/// This split exists because `Listener::run_with_runtime`'s readiness gate
/// (`std::sync::mpsc::Receiver::recv_timeout`) is a genuine OS-thread block,
/// not an `.await` — calling it from a task running ON the SAME core a
/// listener lane also targets deadlocks that core's one OS thread (it can't
/// both block-wait and drain the lane's own inbox to produce the value being
/// waited on). Giving `main`'s body a disjoint core sidesteps the deadlock
/// without reintroducing two independent runtimes with different core
/// counts — App and `main` still share one `Arc<PrimeRuntime>`.
#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
struct AdoptedRuntime {
    inner: Arc<PrimeRuntime>,
    visible_cores: usize,
}

#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
impl Runtime for AdoptedRuntime {
    fn spawn_on_current_core(&self, future: std::pin::Pin<Box<dyn Future<Output = ()> + 'static>>) {
        self.inner.spawn_on_current_core(future);
    }

    fn spawn_on_core(
        &self,
        core_id: CoreId,
        future: std::pin::Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
    ) -> Result<(), SpawnError> {
        self.inner.spawn_on_core(core_id, future)
    }

    fn spawn_factory_on_core(
        &self,
        core_id: CoreId,
        factory: Box<
            dyn FnOnce() -> std::pin::Pin<Box<dyn Future<Output = ()> + 'static>> + Send + 'static,
        >,
    ) -> Result<(), SpawnError> {
        self.inner.spawn_factory_on_core(core_id, factory)
    }

    fn spawn_background_blocking(
        &self,
        work: Box<
            dyn FnOnce() -> Result<Box<dyn std::any::Any + Send>, ProximaError> + Send,
        >,
    ) -> BackgroundHandle<Box<dyn std::any::Any + Send>> {
        self.inner.spawn_background_blocking(work)
    }

    fn timer_at(
        &self,
        deadline: std::time::Instant,
    ) -> std::pin::Pin<Box<dyn Future<Output = ()> + 'static>> {
        self.inner.timer_at(deadline)
    }

    fn num_cores(&self) -> usize {
        self.visible_cores
    }

    fn current_core(&self) -> CoreId {
        self.inner.current_core()
    }
}

/// Publish `inner` (wrapped so `App` sees exactly `visible_cores`) + its
/// matching `PrimeAcceptorFactory` as the ambient runtime `App::new()`
/// adopts — only when the full `serve-prime` bundle (the same cfg
/// `default_runtime`'s prime arm in `app.rs` requires) is compiled in. A
/// prime runtime linked WITHOUT that bundle has no matching acceptor
/// factory available here; `App::new()` falls back to its own
/// config-resolved default in that configuration.
#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool",
    feature = "serve-prime",
    any(target_os = "linux", target_os = "macos")
))]
fn install_prime_ambient(inner: &Arc<PrimeRuntime>, visible_cores: usize) {
    let runtime: Arc<dyn Runtime> = Arc::new(AdoptedRuntime {
        inner: inner.clone(),
        visible_cores,
    });
    let acceptor_factory: Arc<dyn AcceptorFactory> = Arc::new(proxima_net::prime::PrimeAcceptorFactory);
    install_runtime(runtime, acceptor_factory);
}

#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool",
    not(all(feature = "serve-prime", any(target_os = "linux", target_os = "macos")))
))]
fn install_prime_ambient(_inner: &Arc<PrimeRuntime>, _visible_cores: usize) {}

/// `run_tokio` — build a tokio runtime and drive `future` to completion
/// on it, returning its output. Wraps
/// `tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(..)`
/// (or the current-thread builder when `multi_thread` is false). The
/// production analog of the `#[proxima::test]` tokio driver.
///
/// `worker_threads` is honored only for the multi-thread flavor (ignored on
/// current-thread). Prefer this for bins built around hyper/axum or a
/// `TokioPerCoreRuntime` they manage; see [`run_prime`] for the prime
/// serve path.
///
/// # Errors
/// Returns `ProximaError::Io` if the tokio runtime fails to build.
#[cfg(feature = "tokio")]
pub fn run_tokio<F>(
    multi_thread: bool,
    worker_threads: Option<usize>,
    future: F,
) -> Result<F::Output, ProximaError>
where
    F: Future,
{
    // `::` prefix: the crate root (proxima)'s own `pub use proxima_runtime::*;`
    // wildcard above brings in `proxima_runtime::tokio` (the tokio-backed
    // runtime module) whenever proxima-runtime's `tokio` feature is active
    // anywhere in the build graph, which glob-shadows the extern crate
    // `tokio` for a bare path in THIS module. The `::` anchors resolution at
    // the extern prelude, bypassing the shadow.
    let mut builder = if multi_thread {
        let mut builder = ::tokio::runtime::Builder::new_multi_thread();
        if let Some(count) = worker_threads {
            builder.worker_threads(count);
        }
        builder
    } else {
        ::tokio::runtime::Builder::new_current_thread()
    };
    builder.enable_all();
    let runtime = builder.build().map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!("build tokio runtime: {err}")))
    })?;
    Ok(runtime.block_on(future))
}

/// `run` — the adaptive default: drive `future` on prime when the prime
/// runtime is compiled in, else on a tokio multi-thread runtime. Mirrors the
/// `#[proxima::test]` `Default` runtime selection so `#[proxima::main]` picks
/// the same backend as `#[proxima::test]` for the same build.
///
/// # Errors
/// Propagates the backend's build/dispatch error (see [`run_prime`] /
/// [`run_tokio`]). The prime backend additionally requires `F::Output:
/// Send + 'static` (the value crosses the per-core channel); the tokio
/// backend does not.
#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
pub fn run<F>(future: F) -> Result<F::Output, ProximaError>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    run_prime(future)
}

/// `run` — adaptive default (tokio fallback when the prime runtime is
/// not compiled in). See the prime-enabled variant for the full contract.
///
/// # Errors
/// Returns `ProximaError::Io` if the tokio runtime fails to build.
#[cfg(not(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
)))]
pub fn run<F>(future: F) -> Result<F::Output, ProximaError>
where
    F: Future,
{
    run_tokio(true, None, future)
}

/// Like [`run`], but sizes and places whichever backend `Default`
/// resolves to. `#[proxima::main(cores = N, affinity = "...")]` (no explicit
/// `runtime = ...`) compiles down to this — cores/placement for THIS
/// runtime, not a forced backend switch, so a prime-first build stays on
/// prime instead of silently flipping to tokio the moment a core count is
/// named. Bare `#[proxima::main]` passes `cores = None`, which resolves to
/// [`CoreSelection::Auto`](prime::config::CoreSelection::Auto) here — ALL
/// physical cores, matching [`PrimeConfig::default()`](prime::config::PrimeConfig).
///
/// # Errors
/// Same as [`run`], plus the `affinity`/`cores` errors documented on
/// [`run_prime_with_cores`].
#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
pub fn run_with_cores<F>(
    cores: Option<usize>,
    affinity: Option<&str>,
    future: F,
) -> Result<F::Output, ProximaError>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    run_prime_with_cores(cores, affinity, future)
}

/// Like [`run_with_cores`] but for the tokio-fallback build (no prime
/// runtime compiled). `cores` maps onto tokio's `worker_threads`; `affinity`
/// has no tokio-fallback equivalent in this crate (no core-pinning surface
/// here) and is ignored — accepted only so the macro's generated call site
/// is identical regardless of which backend the final binary compiles in.
///
/// # Errors
/// Returns `ProximaError::Io` if the tokio runtime fails to build.
#[cfg(not(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
)))]
pub fn run_with_cores<F>(
    cores: Option<usize>,
    affinity: Option<&str>,
    future: F,
) -> Result<F::Output, ProximaError>
where
    F: Future,
{
    let _ = affinity;
    run_tokio(true, cores, future)
}

// nextest runs one test per process, so `INSTALLED_RUNTIME`'s set-once
// `OnceLock` never leaks state between these — each test observes only the
// runtime it booted itself.
#[cfg(all(
    test,
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // the one intentional behavior change this seam makes: bare
    // `#[proxima::main]` (`cores = None, affinity = None`) used to resolve to
    // `unwrap_or(1)` — a 1-core toy default — and now resolves to
    // `CoreSelection::Auto` (all physical cores), matching
    // `PrimeConfig::default()`.
    #[test]
    fn bare_cores_and_affinity_resolve_to_all_physical_cores() {
        let expected = prime::config::CoreSelection::Auto.resolve();
        let observed = run_with_cores(None, None, async {
            installed_runtime()
                .expect("run_with_cores installs a runtime ambiently")
                .runtime
                .num_cores()
        })
        .expect("run_with_cores");
        assert_eq!(observed, expected);
    }

    #[test]
    fn explicit_cores_overrides_the_auto_default() {
        let observed = run_with_cores(Some(2), None, async {
            installed_runtime()
                .expect("run_with_cores installs a runtime ambiently")
                .runtime
                .num_cores()
        })
        .expect("run_with_cores");
        assert_eq!(observed, 2);
    }

    #[test]
    fn affinity_cores_list_fixes_the_count_without_an_explicit_cores_arg() {
        let observed = run_with_cores(None, Some("4,5,6"), async {
            installed_runtime()
                .expect("run_with_cores installs a runtime ambiently")
                .runtime
                .num_cores()
        })
        .expect("run_with_cores");
        assert_eq!(observed, 3);
    }

    #[test]
    fn cores_conflicting_with_affinity_list_length_is_an_error() {
        let result = run_with_cores(Some(2), Some("4,5,6"), async {});
        assert!(
            result.is_err(),
            "cores = 2 must not silently win over a 3-entry affinity list"
        );
    }

    #[test]
    fn invalid_affinity_literal_is_an_error_not_a_panic() {
        let result = run_with_cores(None, Some("not-a-valid-spec"), async {});
        assert!(result.is_err());
    }
}
