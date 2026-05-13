//! PrimeRuntime — `impl Runtime` aggregating N CoreShard workers + an
//! optional `BackgroundPool` for cross-thread CPU-bound work. operators bind
//! this via `App::with_runtime(...)` once `runtime-prime-full` is on.
//!
//! futures spawned via `spawn_on_current_core` reach the per-core executor
//! through the thread-local pointer set by the worker; cross-core dispatch
//! routes through each core's per-producer SPSC inbox (C1).

#![cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use proxima_core::ProximaError;
use proxima_runtime::{BackgroundHandle, BackgroundPool, CoreId, Runtime, SpawnError};

use super::core_shard::{
    self, CoreShardHandle, current_core as core_shard_current_core, spawn_on_current_core, timer_at,
};

pub struct PrimeRuntime {
    cores: Arc<Vec<CoreShardHandle>>,
    background_pool: Option<Arc<dyn BackgroundPool>>,
    /// epoch in std::time::Instant for translating `timer_at(deadline: Instant)`
    /// into the timer's tick basis (ms since shard launch).
    epoch: Instant,
    /// P2 — sister tokio runtimes (one per core) that back compat mode.
    /// `None` for native prime; `Some` when the runtime was built with
    /// `Builder::tokio_compat()`. Kept alive for the life of `PrimeRuntime`
    /// so prime worker threads can hold static `EnterGuard` borrows into
    /// each sister's `Handle`.
    #[cfg(feature = "prime-tokio-compat")]
    tokio_compat_handles: Option<Arc<super::tokio_compat::TokioCompatHandles>>,
}

impl PrimeRuntime {
    /// Spawn `num_cores` worker threads, UNPINNED (the default — workers float,
    /// the OS schedules them). Pin to specific cores via the affinity surface
    /// ([`PrimeRuntime::builder`] / [`PrimeConfig`](super::super::config::PrimeConfig)),
    /// which is the only place placement/pinning composes. Floating retains the
    /// per-core shared-nothing architecture (the throughput win) and lets a
    /// worker dodge a noisy neighbour; a hard pin is set-once-never-migrate, so
    /// it stalls on a contended core — correct only for a dedicated box.
    pub fn new(num_cores: usize) -> Result<Self, ProximaError> {
        Self::new_inner(num_cores, false)
    }

    /// Inverted compat (design D2) — each prime worker OWNS its own sister
    /// tokio current-thread runtime and ticks the prime executor inside
    /// `sister.block_on(...)`, so raw `tokio::spawn` from a prime task takes
    /// tokio's LOCAL fast path (no per-spawn driver.unpark/kevent). Mirrors
    /// [`new_with_tokio_compat`](Self::new_with_tokio_compat) but uses the
    /// inverted worker construction path. Gated on
    /// `prime-tokio-compat-inverted`.
    ///
    /// minimal park; full Dekker-park fidelity tracked in
    /// discipline-inverted-compat.md.
    #[cfg(feature = "prime-tokio-compat-inverted")]
    pub fn new_with_tokio_compat_inverted(num_cores: usize) -> Result<Self, ProximaError> {
        Self::new_inverted_placed((0..num_cores.max(1)).collect(), false)
    }

    /// The single composable inverted-compat constructor the affinity surface
    /// resolves to: `placement[i]` is worker `i`'s physical core when `pin`.
    #[cfg(feature = "prime-tokio-compat-inverted")]
    pub(crate) fn new_inverted_placed(
        placement: Vec<usize>,
        pin: bool,
    ) -> Result<Self, ProximaError> {
        let num_cores = placement.len().max(1);
        let physical = core_affinity::get_core_ids().unwrap_or_default();
        let mut cores: Vec<CoreShardHandle> = Vec::with_capacity(num_cores);
        for (index, &physical_core) in placement.iter().enumerate().take(num_cores) {
            let core_id = CoreId(index);
            let affinity = if pin {
                physical.get(physical_core).copied()
            } else {
                None
            };
            let handle = core_shard::launch_inverted_with_lanes(
                core_id,
                affinity,
                super::super::core::sized::INBOX_CAPACITY,
                super::super::core::sized::INBOX_CAPACITY,
            )?;
            cores.push(handle);
        }
        Ok(Self {
            cores: Arc::new(cores),
            background_pool: None,
            epoch: Instant::now(),
            #[cfg(feature = "prime-tokio-compat")]
            tokio_compat_handles: None,
        })
    }

    /// P2 — native prime workers PLUS a sister tokio current-thread
    /// runtime per core. each prime worker enters its sister's
    /// `tokio::runtime::Handle` for life, so `tokio::spawn` /
    /// `tokio::sync::*` / `tokio::time::*` API calls from inside a
    /// prime task resolve against the sister runtime.
    ///
    /// Cost picture, ship-criteria, and the bench matrix live in
    /// `rust/docs/runtime-prime/discipline-prime-tokio-compat.md`.
    #[cfg(feature = "prime-tokio-compat")]
    pub fn new_with_tokio_compat(num_cores: usize) -> Result<Self, ProximaError> {
        Self::new_inner(num_cores, true)
    }

    /// Returns the sister tokio runtime handles when this runtime was
    /// built with [`Builder::tokio_compat`]. `None` for vanilla prime.
    /// Useful for diagnostics, benches, and tests that need to verify
    /// the compat plumbing is in place; user code rarely needs this
    /// (the EnterGuard on each worker covers the common path).
    #[cfg(feature = "prime-tokio-compat")]
    #[must_use]
    pub fn tokio_compat_handles(&self) -> Option<&Arc<super::tokio_compat::TokioCompatHandles>> {
        self.tokio_compat_handles.as_ref()
    }

    fn new_inner(num_cores: usize, enable_tokio_compat: bool) -> Result<Self, ProximaError> {
        // default = float (unpinned); the affinity surface opts into pinning.
        Self::new_inner_placed((0..num_cores.max(1)).collect(), false, enable_tokio_compat)
    }

    /// The single composable constructor the affinity surface resolves to:
    /// `placement[i]` is the physical core index worker `i` pins to when `pin`;
    /// `pin = false` leaves every worker unpinned (the float default). The
    /// `Builder` computes `(placement, pin)` from the `Affinity` knob and calls
    /// this — no per-mode public constructor.
    ///
    /// `pub` (not `pub(crate)`) because `proxima::runtime::run_prime_with_cores`
    /// calls it directly from the umbrella crate: it needs one extra, hidden
    /// worker beyond the App-visible placement (the core that drives
    /// `#[proxima::main]`'s own body), which `Builder::build()` has no shape for.
    pub fn new_inner_placed(
        placement: Vec<usize>,
        pin: bool,
        _enable_tokio_compat: bool,
    ) -> Result<Self, ProximaError> {
        let num_cores = placement.len().max(1);
        let physical = core_affinity::get_core_ids().unwrap_or_default();

        #[cfg(feature = "prime-tokio-compat")]
        let tokio_compat_handles = if _enable_tokio_compat {
            Some(super::tokio_compat::TokioCompatHandles::new(num_cores)?)
        } else {
            None
        };

        // inbox lanes = peak concurrent producer threads targeting a core (the
        // other workers via cross-core spawn + background pool + caller threads),
        // NOT the ring depth. lanes recycle on producer-thread exit, so this
        // bounds *concurrent* producers, not total. previously this passed
        // INBOX_CAPACITY (1024) for the lane count too, so every core eagerly
        // allocated 1024 lanes * 1024-slot rings * size_of::<task> (~72 MiB),
        // and num_cores of those dominated daemon RSS. lanes_per_core * N +
        // headroom (both from prime-runtime.toml) covers realistic fan-in with
        // recycling; the ring depth stays INBOX_CAPACITY.
        // static inbox: lanes are eager + fixed, so size for peak fan-in.
        #[cfg(not(feature = "runtime-prime-inbox-dynamic"))]
        let inbox_lanes = num_cores
            .saturating_mul(super::super::core::sized::INBOX_LANES_PER_CORE)
            .saturating_add(super::super::core::sized::INBOX_LANES_HEADROOM);
        // dynamic inbox: this is only the eager FLOOR (= N); lanes grow lazily
        // to the ceiling on demand, so no headroom is pre-paid.
        #[cfg(feature = "runtime-prime-inbox-dynamic")]
        let inbox_lanes = num_cores;

        let mut cores: Vec<CoreShardHandle> = Vec::with_capacity(num_cores);
        for (index, &physical_core) in placement.iter().enumerate().take(num_cores) {
            let core_id = CoreId(index);
            // logical core `index` pins to the placement's physical core; packed
            // placement reproduces the old `physical.get(index)` behaviour. When
            // `pin` is false the worker floats (no `set_for_current`).
            let affinity = if pin {
                physical.get(physical_core).copied()
            } else {
                None
            };

            #[cfg(feature = "prime-tokio-compat")]
            let setup = tokio_compat_handles
                .as_ref()
                .and_then(|handles| handles.worker_setup(core_id));
            #[cfg(not(feature = "prime-tokio-compat"))]
            let setup: Option<core_shard::WorkerSetup> = None;

            let handle = core_shard::launch_with_lanes_and_setup(
                core_id,
                affinity,
                inbox_lanes,
                super::super::core::sized::INBOX_CAPACITY,
                setup,
            )?;
            cores.push(handle);
        }
        Ok(Self {
            cores: Arc::new(cores),
            background_pool: None,
            epoch: Instant::now(),
            #[cfg(feature = "prime-tokio-compat")]
            tokio_compat_handles,
        })
    }

    /// plug in a `BackgroundPool` for CPU-bound cross-thread work.
    #[must_use]
    pub fn with_background_pool(mut self, pool: Arc<dyn BackgroundPool>) -> Self {
        self.background_pool = Some(pool);
        self
    }

    /// Fluent builder. Equivalent first-class entry alongside
    /// [`PrimeRuntime::from_config`]; pick whichever shape the call
    /// site prefers. Defaults: `cores = auto` (physical core count),
    /// `background_pool = rayon` (when the `rayon` feature is on,
    /// else `inline`).
    ///
    /// ```ignore
    /// use proxima::prime::PrimeRuntime;
    /// let runtime = PrimeRuntime::builder()
    ///     .cores(4)
    ///     .background_inline()  // tests / small workloads
    ///     .build()?;
    /// ```
    ///
    /// The builder accepts a [`PrimeConfig`](super::super::config::PrimeConfig)
    /// via [`from_config`](super::super::config::Builder::from_config) — use
    /// it when most of the deployment shape lives in config but a test
    /// harness or hot-fix script needs to pin one field locally:
    ///
    /// ```ignore
    /// let config = PrimeConfig::from_env()?;
    /// let runtime = PrimeRuntime::builder()
    ///     .from_config(&config)
    ///     .cores(2)  // override the config's `cores`
    ///     .build()?;
    /// ```
    #[must_use]
    pub fn builder() -> super::super::config::Builder {
        super::super::config::Builder::new()
    }

    /// Build a `PrimeRuntime` directly from a typed
    /// [`PrimeConfig`](super::super::config::PrimeConfig). Equivalent
    /// to `PrimeRuntime::builder().from_config(config).build()`.
    ///
    /// `PrimeConfig` derives `conflaguration::Settings`, so it loads
    /// from the environment with no extra plumbing:
    ///
    /// ```ignore
    /// // PRIME_CORES=8 PRIME_BACKGROUND_POOL=rayon
    /// let config = PrimeConfig::from_env()?;
    /// let runtime = PrimeRuntime::from_config(&config)?;
    /// ```
    ///
    /// The same fields parse from a TOML file via
    /// `conflaguration::builder()`; layering env-over-file lets ops
    /// ship a base `prime.toml` and override per-environment via
    /// `PRIME_*` env vars without rebuilding.
    pub fn from_config(config: &super::super::config::PrimeConfig) -> Result<Self, ProximaError> {
        Self::builder().from_config(config).build()
    }

    /// Drive `future` to completion on this runtime's core 0, returning its
    /// output — method-syntax sugar for the runtime-holding
    /// [`proxima_runtime::block_on`], which it forwards to verbatim. Same verb
    /// as the single-thread [`LocalExecutor::block_on`](super::super::core::local_executor::LocalExecutor::block_on)
    /// and the no-runtime [`proxima_primitives::block_on`]; this one runs the
    /// future on a real prime worker.
    ///
    /// FOREIGN-THREAD entry: call from a thread that is NOT a prime worker of
    /// this runtime, or you deadlock the core-0 worker — the edge `run_prime`
    /// driver sidesteps that with a dedicated driver core; this method does
    /// not. See [`proxima_runtime::block_on`]'s doc for the full contract.
    ///
    /// # Errors
    /// Propagates [`proxima_runtime::block_on`]'s dispatch errors.
    #[must_use = "block_on returns the future's output or the dispatch error"]
    pub fn block_on<F>(&self, future: F) -> Result<F::Output, ProximaError>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        proxima_runtime::block_on(self, future)
    }

    /// typed-task fast path. avoids the per-spawn `Pin<Box<dyn Future>>`
    /// fat-pointer allocation that the `Runtime` trait surface mandates;
    /// the caller hands over a concrete `F` and the runtime inlines it
    /// (when small) or single-Boxes it (when oversized) — either way no
    /// `dyn Future` indirection on the dispatch path.
    ///
    /// available only on the concrete `PrimeRuntime` type (not the
    /// `Runtime` trait) because trait dispatch through `Arc<dyn Runtime>`
    /// would re-introduce the very fat-pointer cost this method avoids.
    /// callers wanting the fast path must hold a typed `Arc<PrimeRuntime>`
    /// (or `&PrimeRuntime`) — load generators and bench harnesses
    /// usually do.
    ///
    /// returns `Err(SpawnError::InboxFull)` on transient lane saturation
    /// (future consumed); `Err(SpawnError::Disconnected)` on out-of-range
    /// core or shut-down worker.
    pub fn spawn_typed_on_core<F>(&self, core_id: CoreId, future: F) -> Result<(), SpawnError>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        match self.cores.get(core_id.0) {
            Some(slot) => slot.dispatch_send_inline(future),
            None => Err(SpawnError::Disconnected),
        }
    }
}

impl Runtime for PrimeRuntime {
    fn spawn_on_current_core(&self, future: Pin<Box<dyn Future<Output = ()> + 'static>>) {
        spawn_on_current_core(future);
    }

    fn spawn_on_core(
        &self,
        core_id: CoreId,
        future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
    ) -> Result<(), SpawnError> {
        match self.cores.get(core_id.0) {
            Some(slot) => slot.dispatch_send(future),
            None => Err(SpawnError::Disconnected),
        }
    }

    fn spawn_factory_on_core(
        &self,
        core_id: CoreId,
        factory: Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + 'static>> + Send + 'static>,
    ) -> Result<(), SpawnError> {
        match self.cores.get(core_id.0) {
            Some(slot) => slot.dispatch_factory(factory),
            None => Err(SpawnError::Disconnected),
        }
    }

    fn spawn_background_blocking(
        &self,
        work: Box<dyn FnOnce() -> Result<Box<dyn std::any::Any + Send>, ProximaError> + Send>,
    ) -> BackgroundHandle<Box<dyn std::any::Any + Send>> {
        if let Some(pool) = &self.background_pool {
            return pool.spawn(work);
        }
        // no background pool configured — run work inline on a spawned std
        // thread. simple fallback; the documented recommendation is to
        // attach a real pool via `with_background_pool`.
        let (sender, receiver) = futures::channel::oneshot::channel();
        std::thread::spawn(move || {
            let result = work();
            let _ = sender.send(result);
        });
        Box::pin(async move {
            receiver.await.unwrap_or_else(|_| {
                Err(ProximaError::Body(
                    "proxima inline background sender dropped".into(),
                ))
            })
        })
    }

    fn timer_at(&self, deadline: Instant) -> Pin<Box<dyn Future<Output = ()> + 'static>> {
        let elapsed = deadline.saturating_duration_since(self.epoch).as_millis();
        let tick: u64 = elapsed.min(u64::MAX as u128) as u64;
        Box::pin(timer_at(tick))
    }

    fn num_cores(&self) -> usize {
        self.cores.len()
    }

    fn current_core(&self) -> CoreId {
        match core_shard_current_core() {
            Some(id) => id,
            None => {
                panic!(
                    "current_core: called from outside a proxima worker thread — \
                     use spawn_on_core(N, ...) for cross-thread dispatch"
                )
            }
        }
    }
}

impl Drop for PrimeRuntime {
    fn drop(&mut self) {
        // each CoreShardHandle's Drop signals shutdown and joins; the Arc
        // makes the order indeterminate, but every handle gets dropped.
        // explicit shutdown_and_join() is available if the caller needs to
        // observe shutdown errors.
        let _ = self.cores.len();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn new_runtime_spawns_requested_number_of_workers() {
        let runtime = PrimeRuntime::new(2).expect("build runtime");
        assert_eq!(runtime.num_cores(), 2);
    }

    // EXPERIMENT E (transport-unification proposal §5): does an inline tokio poll
    // on a `tokio_compat` prime worker complete, or does it stall because the mio
    // reactor is driven on the sister thread? a successful inline `connect` proves
    // the sister reactor delivers readiness to a future polled on the prime worker
    // (Direct is sound); a stall confirms the silent-hang premise (HopTo required).
    // records the fact; no hard assert on completion — the algorithm ships the
    // sound hop regardless, E only licenses the Direct optimization.
    #[cfg(feature = "prime-tokio-compat")]
    #[test]
    fn experiment_e_inline_tokio_connect_on_compat_worker() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind echo");
        let addr = listener.local_addr().expect("addr");
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_server = stop.clone();
        let server = std::thread::spawn(move || {
            listener.set_nonblocking(true).ok();
            while !stop_server.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok(_conn) => {} // drop the connection; connect-completion is all we test
                    Err(_) => std::thread::sleep(Duration::from_millis(5)),
                }
            }
        });

        let runtime = PrimeRuntime::new_with_tokio_compat(1).expect("compat runtime");
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ok = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done_worker = done.clone();
        let ok_worker = ok.clone();

        runtime
            .spawn_on_core(
                CoreId(0),
                Box::pin(async move {
                    // inline construct+poll on the compat worker — the exact thing
                    // the algorithm's "Direct" would do; NO hop to the sister.
                    let connected = tokio::net::TcpStream::connect(addr).await.is_ok();
                    ok_worker.store(connected, Ordering::Release);
                    done_worker.store(true, Ordering::Release);
                }),
            )
            .expect("spawn on compat worker");

        let mut completed = false;
        for _ in 0..200 {
            if done.load(Ordering::Acquire) {
                completed = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        stop.store(true, Ordering::Release);
        let _ = server.join();

        println!(
            "EXPERIMENT_E_RESULT inline_compat_connect completed={completed} ok={}",
            ok.load(Ordering::Acquire)
        );
    }

    #[test]
    fn spawn_on_core_dispatches_to_target_worker() {
        let runtime = PrimeRuntime::new(2).expect("build runtime");
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_a = counter.clone();
        runtime
            .spawn_on_core(
                CoreId(0),
                Box::pin(async move {
                    counter_a.fetch_add(1, Ordering::AcqRel);
                }),
            )
            .expect("spawn on fresh runtime");
        let counter_b = counter.clone();
        runtime
            .spawn_on_core(
                CoreId(1),
                Box::pin(async move {
                    counter_b.fetch_add(1, Ordering::AcqRel);
                }),
            )
            .expect("spawn on fresh runtime");
        for _ in 0..50 {
            if counter.load(Ordering::Acquire) == 2 {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(counter.load(Ordering::Acquire), 2);
    }

    #[test]
    fn current_core_inside_dispatched_future_matches_target() {
        let runtime = PrimeRuntime::new(2).expect("build runtime");
        let observed: Arc<std::sync::Mutex<Option<CoreId>>> = Arc::new(std::sync::Mutex::new(None));
        let observed_for_task = observed.clone();
        runtime
            .spawn_on_core(
                CoreId(1),
                Box::pin(async move {
                    let id = core_shard_current_core();
                    *observed_for_task.lock().unwrap() = id;
                }),
            )
            .expect("spawn on fresh runtime");
        for _ in 0..50 {
            if observed.lock().unwrap().is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(*observed.lock().unwrap(), Some(CoreId(1)));
    }

    #[test]
    fn timer_at_resolves_after_deadline_passes() {
        let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(1).expect("build runtime"));
        let done = Arc::new(AtomicUsize::new(0));
        let done_for_factory = done.clone();
        let target = Instant::now() + Duration::from_millis(50);
        // timer_at returns a `!Send` future (thread-local timer access), so
        // we ship the *construction* via spawn_factory_on_core and the future
        // is built and awaited on the target core.
        let runtime_for_factory = runtime.clone();
        runtime
            .spawn_factory_on_core(
                CoreId(0),
                Box::new(move || {
                    let timer = runtime_for_factory.timer_at(target);
                    Box::pin(async move {
                        timer.await;
                        done_for_factory.fetch_add(1, Ordering::AcqRel);
                    })
                }),
            )
            .expect("spawn on fresh runtime");
        for _ in 0..100 {
            if done.load(Ordering::Acquire) == 1 {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(done.load(Ordering::Acquire), 1);
    }

    /// Race-close regression: worker park sequence used to recheck
    /// only executor ready queues (`tick()`) after `arm_wakeup`,
    /// missing inbox pushes that landed in the window between
    /// spin-loop end and arm_wakeup. Under a "dispatch 4 tasks,
    /// busy-wait for completion" pattern repeated many iters, the
    /// worker would occasionally park with a task still in its
    /// inbox, deadlocking that core. Pre-fix repro
    /// (`examples/w4_mutex_repro` at 100 iters) hung by iter ~50
    /// with `started per core = [1, 0, 1, 1]` — task body for one
    /// core never ran. Post-fix the recheck also drains the inbox,
    /// closing the race.
    ///
    /// Test shape mirrors the W4 mutex bench at unit scale so a
    /// regression fails in deterministic seconds instead of a
    /// 10-minute bench timeout. 200 iters × 4 tasks × 200 contended
    /// lock cycles per task — many opportunities for the worker to
    /// park between iters and trigger the race.
    #[test]
    fn worker_park_recheck_drains_inbox_under_repeated_dispatch() {
        const TASKS: usize = 4;
        const OPS_PER_TASK: usize = 200;
        const ITERS: usize = 200;

        let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(TASKS).expect("build runtime"));

        for iter in 0..ITERS {
            let lock = Arc::new(futures::lock::Mutex::new(0u64));
            let done = Arc::new(AtomicUsize::new(0));
            for thread_index in 0..TASKS {
                let lock_outer = lock.clone();
                let done_outer = done.clone();
                let core = CoreId(thread_index);
                let _ = proxima_runtime::spawn_on_core_blocking_with(
                    runtime.as_ref(),
                    core,
                    move || {
                        let lock_inner = lock_outer.clone();
                        let done_inner = done_outer.clone();
                        Box::pin(async move {
                            for _ in 0..OPS_PER_TASK {
                                let mut guard = lock_inner.lock().await;
                                *guard += 1;
                                drop(guard);
                            }
                            done_inner.fetch_add(1, Ordering::AcqRel);
                        })
                    },
                );
            }
            let deadline = Instant::now() + Duration::from_secs(5);
            while done.load(Ordering::Acquire) < TASKS && Instant::now() < deadline {
                std::hint::spin_loop();
            }
            assert_eq!(
                done.load(Ordering::Acquire),
                TASKS,
                "iter {iter}: worker parked with task in inbox — race-close regressed",
            );
        }
    }

    /// Bug B fix verification: multiple producer threads each dispatch
    /// through `Arc<dyn Runtime>::spawn_on_core` against the SAME
    /// `CoreShardHandle`'s Producer. Pre-fix this hung (concurrent
    /// `try_send` on a single SPSC lane). Post-fix every thread gets
    /// its own SPSC lane via the thread-local cache in `try_send_mpsc`.
    /// Mirrors the bench `spawn_fanin_{2,4,8}` shape at unit scale so
    /// a hang fails the test in deterministic seconds, not a bench
    /// timeout.
    #[test]
    fn fanin_4_threads_share_runtime_arc_no_deadlock() {
        let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(2).expect("build runtime"));
        let counter = Arc::new(AtomicUsize::new(0));
        const THREADS: usize = 4;
        const PER_THREAD: usize = 500;
        let mut handles = Vec::with_capacity(THREADS);
        for thread_index in 0..THREADS {
            let runtime = runtime.clone();
            let counter = counter.clone();
            handles.push(std::thread::spawn(move || {
                for spawn_index in 0..PER_THREAD {
                    let counter = counter.clone();
                    let core = CoreId((thread_index + spawn_index) % 2);
                    let _ = proxima_runtime::spawn_on_core_blocking_with(
                        runtime.as_ref(),
                        core,
                        move || {
                            let counter = counter.clone();
                            Box::pin(async move {
                                counter.fetch_add(1, Ordering::AcqRel);
                            })
                        },
                    );
                }
            }));
        }
        for handle in handles {
            handle.join().expect("producer thread");
        }
        let total = THREADS * PER_THREAD;
        let deadline = Instant::now() + Duration::from_secs(5);
        while counter.load(Ordering::Acquire) < total && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            counter.load(Ordering::Acquire),
            total,
            "expected {total} task bodies to run; got {}",
            counter.load(Ordering::Acquire),
        );
    }

    #[test]
    fn many_cross_core_spawns_preserve_count() {
        let runtime = PrimeRuntime::new(2).expect("build runtime");
        let counter = Arc::new(AtomicUsize::new(0));
        let total = 200_usize;
        for index in 0..total {
            let counter = counter.clone();
            let target = CoreId(index % 2);
            // many_cross_core_spawns: 200 tasks across 2 cores, well
            // under the default 1024-lane capacity — spawn succeeds.
            runtime
                .spawn_on_core(
                    target,
                    Box::pin(async move {
                        counter.fetch_add(1, Ordering::AcqRel);
                    }),
                )
                .expect("spawn on fresh runtime");
        }
        for _ in 0..200 {
            if counter.load(Ordering::Acquire) == total {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(counter.load(Ordering::Acquire), total);
    }

    /// Typed-path stress: 8 producer threads concurrently call
    /// `spawn_typed_on_core` against the same `Arc<PrimeRuntime>`,
    /// distributing 10_000 typed tasks across all cores. Exercises:
    ///   - concurrent `try_send_mpsc` against the cross-core inbox
    ///     under the new 4-way HOT_LANES associative cache
    ///   - the worker's eager-poll path under concurrent arrivals
    ///   - the sticky-cursor scan when multiple lanes have work
    ///
    /// no_loss invariant: every task body runs exactly once.
    /// no_hang invariant: completes within the deadline.
    #[test]
    fn typed_fanin_8_concurrent_threads_no_loss_no_hang() {
        let runtime: Arc<PrimeRuntime> = Arc::new(PrimeRuntime::new(2).expect("build runtime"));
        let counter = Arc::new(AtomicUsize::new(0));
        const THREADS: usize = 8;
        const PER_THREAD: usize = 1_250;
        let mut handles = Vec::with_capacity(THREADS);
        for thread_index in 0..THREADS {
            let runtime = runtime.clone();
            let counter = counter.clone();
            handles.push(std::thread::spawn(move || {
                for spawn_index in 0..PER_THREAD {
                    let counter = counter.clone();
                    let core = CoreId((thread_index + spawn_index) % 2);
                    loop {
                        let counter = counter.clone();
                        match runtime.spawn_typed_on_core(core, async move {
                            counter.fetch_add(1, Ordering::AcqRel);
                        }) {
                            Ok(()) => break,
                            Err(SpawnError::InboxFull) => std::thread::yield_now(),
                            Err(SpawnError::Disconnected) => panic!("disconnected"),
                        }
                    }
                }
            }));
        }
        for handle in handles {
            handle.join().expect("producer thread");
        }
        let total = THREADS * PER_THREAD;
        let deadline = Instant::now() + Duration::from_secs(10);
        while counter.load(Ordering::Acquire) < total && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            counter.load(Ordering::Acquire),
            total,
            "{total} typed task bodies expected; observed {}",
            counter.load(Ordering::Acquire),
        );
    }

    /// Cross-thread wake after eager-Pending install: a typed task
    /// returns Pending on the eager poll, gets installed in the slab,
    /// then a SEPARATE thread fires the captured waker. Worker must
    /// observe the wake and re-poll to Ready. This is the dragon:
    /// if the eager-poll path accidentally leaked the noop-waker as
    /// the task's canonical waker, the cross-thread wake would land
    /// in the void and the task would silently hang.
    #[test]
    fn typed_pending_then_cross_thread_wake_drives_to_ready() {
        use std::sync::Mutex;
        use std::sync::atomic::AtomicBool;
        use std::task::Waker;

        let runtime: Arc<PrimeRuntime> = Arc::new(PrimeRuntime::new(1).expect("build runtime"));
        let done = Arc::new(AtomicBool::new(false));
        let flag = Arc::new(AtomicBool::new(false));
        let waker_slot: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));

        let done_for_task = done.clone();
        let flag_for_task = flag.clone();
        let waker_for_task = waker_slot.clone();

        runtime
            .spawn_typed_on_core(CoreId(0), async move {
                struct FlagFut {
                    flag: Arc<AtomicBool>,
                    waker_slot: Arc<Mutex<Option<Waker>>>,
                }
                impl Future for FlagFut {
                    type Output = ();
                    fn poll(
                        self: std::pin::Pin<&mut Self>,
                        context: &mut std::task::Context<'_>,
                    ) -> std::task::Poll<()> {
                        if self.flag.load(Ordering::Acquire) {
                            return std::task::Poll::Ready(());
                        }
                        *self.waker_slot.lock().expect("waker") = Some(context.waker().clone());
                        std::task::Poll::Pending
                    }
                }
                FlagFut {
                    flag: flag_for_task,
                    waker_slot: waker_for_task,
                }
                .await;
                done_for_task.store(true, Ordering::Release);
            })
            .expect("spawn_typed");

        // Wait for the worker to install + re-poll the task — the
        // second poll captures the REAL per-slot waker (not the noop).
        let waker_deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if waker_slot.lock().expect("waker mutex").is_some() {
                break;
            }
            assert!(
                Instant::now() < waker_deadline,
                "worker never captured the real waker — eager-poll didn't re-poll after slab install",
            );
            std::thread::sleep(Duration::from_millis(5));
        }

        // Fire the captured waker from a separate thread (cross-
        // thread wake — must route through TaskWaker::do_wake's
        // remote_ready + reactor wakeup path).
        let flag_for_waker = flag.clone();
        let waker_for_waker = waker_slot.clone();
        let _waker_thread = std::thread::spawn(move || {
            flag_for_waker.store(true, Ordering::Release);
            let waker = waker_for_waker
                .lock()
                .expect("waker mutex")
                .take()
                .expect("waker captured by re-poll");
            waker.wake();
        });

        let deadline = Instant::now() + Duration::from_secs(2);
        while !done.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            done.load(Ordering::Acquire),
            "typed task with eager-Pending install must resume on cross-thread wake",
        );
    }
}
