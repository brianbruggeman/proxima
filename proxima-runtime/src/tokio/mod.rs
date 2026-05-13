//! Default `Runtime` implementation: N tokio current-thread runtimes pinned
//! one per CPU core. The Pingora pattern — no work-stealing on the chain
//! runtime, `?Send` futures supported via `tokio::task::spawn_local`, real
//! CPU pinning via `core_affinity`. The HTTP ecosystem (hyper, h2, h3,
//! quinn, tokio-rustls) keeps working unchanged because each per-core
//! worker still drives a tokio runtime — we just chose the threading model.

pub mod primitives;
pub use primitives::{TokioJoinSet, TokioMutex, TokioMutexGuard, TokioNotify, TokioSleep};

use std::cell::Cell;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use proxima_core::ProximaError;

use crate::{BackgroundHandle, CoreId, Runtime, SpawnError, SpawnRequest};

thread_local! {
    /// each worker thread sets this once at startup so `current_core()` and
    /// `spawn_on_current_core()` know which slot they're running on.
    static CURRENT_CORE: Cell<Option<CoreId>> = const { Cell::new(None) };
}

struct CoreSlot {
    spawn_tx: flume::Sender<SpawnRequest>,
    /// `Option` so `Drop` can `take` and `.join()` the handle.
    handle: Option<thread::JoinHandle<()>>,
}

/// Pinned-per-core executor backed by tokio current-thread runtimes.
///
/// Construction starts N OS threads, each pinned to a CPU core (where
/// `core_affinity` can determine the physical ids), each driving its own
/// `tokio::runtime::current_thread` + `LocalSet`. Cross-core spawn flows
/// through a per-core flume MPSC channel.
pub struct TokioPerCoreRuntime {
    /// shared so cloning the runtime handle is cheap; the actual threads are
    /// owned by this Arc and torn down on the final drop.
    cores: Arc<Vec<CoreSlot>>,
    /// optional override for cross-thread CPU-bound work. when None,
    /// falls back to `tokio::task::spawn_blocking`.
    background_pool: Option<Arc<dyn crate::BackgroundPool>>,
    /// when set, this runtime WRAPS an existing tokio runtime (the host's) and
    /// dispatches `Send` work onto it via `Handle::spawn` — it owns NO worker
    /// threads (`cores` is empty). the tokio-hosts-proxima seam: a host that
    /// already runs a tokio runtime hands its `Handle` here so `proxima::Client`
    /// rides it instead of spawning a second runtime. set by [`from_handle`](Self::from_handle).
    host: Option<tokio::runtime::Handle>,
}

impl TokioPerCoreRuntime {
    /// Spawn `num_cores` per-core worker threads. Pinning is best-effort:
    /// when `core_affinity` can't enumerate physical core ids (CI, restricted
    /// environments) workers run unpinned but still single-threaded.
    pub fn new(num_cores: usize) -> Result<Self, ProximaError> {
        let num_cores = num_cores.max(1);
        let physical = core_affinity::get_core_ids().unwrap_or_default();
        let mut cores: Vec<CoreSlot> = Vec::with_capacity(num_cores);
        for index in 0..num_cores {
            let core_id = CoreId(index);
            let affinity = physical.get(index).copied();
            let (spawn_tx, spawn_rx) = flume::unbounded();
            let handle = thread::Builder::new()
                .name(format!("proxima-core-{index}"))
                .spawn(move || worker(core_id, affinity, spawn_rx))
                .map_err(|err| {
                    ProximaError::Config(format!("spawn per-core worker thread: {err}"))
                })?;
            cores.push(CoreSlot {
                spawn_tx,
                handle: Some(handle),
            });
        }
        Ok(Self {
            cores: Arc::new(cores),
            background_pool: None,
            host: None,
        })
    }

    /// Wrap an existing tokio runtime (via its `Handle`) as a proxima `Runtime`
    /// WITHOUT spawning new threads — the tokio-hosts-proxima seam. `spawn_on_core`
    /// dispatches `Send` work straight onto the host runtime, which is the path
    /// `proxima::Client`'s off-worker hop takes; so a tokio-hosted application
    /// (a GUI event loop, an embedder) runs the client on its own runtime instead
    /// of a second, client-owned one. `?Send` per-core work (server listener
    /// loops) is out of scope for a wrapped handle — use [`new`](Self::new) for that.
    #[must_use]
    pub fn from_handle(handle: tokio::runtime::Handle) -> Self {
        Self {
            cores: Arc::new(Vec::new()),
            background_pool: None,
            host: Some(handle),
        }
    }

    /// Plug in a `BackgroundPool` for CPU-bound cross-thread work. Without
    /// this, `spawn_background_blocking` falls back to
    /// `tokio::task::spawn_blocking` (good for I/O-blocking work; sub-
    /// optimal for fork-join compute).
    #[must_use]
    pub fn with_background_pool(mut self, pool: Arc<dyn crate::BackgroundPool>) -> Self {
        self.background_pool = Some(pool);
        self
    }

    /// Drive `future` to completion on this runtime's core 0, returning its
    /// output — method-syntax sugar for the runtime-holding
    /// [`crate::block_on`], which it forwards to verbatim (same verb as
    /// `PrimeRuntime::block_on` and the no-runtime
    /// `proxima_primitives::block_on`, just on a tokio per-core worker).
    ///
    /// FOREIGN-THREAD entry: call from a thread that is NOT a worker of this
    /// runtime, or you deadlock the core-0 worker — the same rule as
    /// `tokio::runtime::Runtime::block_on`. See [`crate::block_on`] for the
    /// full contract.
    ///
    /// # Errors
    /// Propagates [`crate::block_on`]'s dispatch errors.
    #[must_use = "block_on returns the future's output or the dispatch error"]
    pub fn block_on<F>(&self, future: F) -> Result<F::Output, ProximaError>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        crate::block_on(self, future)
    }
}

fn worker(
    core_id: CoreId,
    affinity: Option<core_affinity::CoreId>,
    spawn_rx: flume::Receiver<SpawnRequest>,
) {
    if let Some(target) = affinity {
        // best-effort: ignore failure (e.g., sandboxed CI runners)
        let _ = core_affinity::set_for_current(target);
    }
    CURRENT_CORE.with(|cell| cell.set(Some(core_id)));

    run_event_loop(spawn_rx);
}

/// Drains spawn requests until the channel returns Shutdown or closes.
async fn drain_loop(spawn_rx: flume::Receiver<SpawnRequest>) {
    while let Ok(request) = spawn_rx.recv_async().await {
        match request {
            SpawnRequest::Send(future) => {
                tokio::task::spawn_local(future);
            }
            SpawnRequest::Factory(factory) => {
                let future = factory();
                tokio::task::spawn_local(future);
            }
            // `Inline` defaults to `Infallible` in tokio's channel — the
            // SendInline arm is unreachable but kept for exhaustiveness.
            // (A cross-runtime mix that funneled a prime InlineTask here
            // would no longer compile, because the channel's Inline type
            // would have to be InlineTask — and tokio's worker doesn't
            // know how to poll one.)
            SpawnRequest::SendInline(never) => match never {},
            SpawnRequest::Shutdown => break,
        }
    }
}

#[cfg(all(target_os = "linux", feature = "io-uring"))]
fn run_event_loop(spawn_rx: flume::Receiver<SpawnRequest>) {
    // tokio-uring drives its own current-thread runtime + LocalSet
    // backed by io_uring. Owned-buffer I/O, no epoll. The substrate
    // contract (LocalSet for ?Send tasks, current-thread tokio) is
    // preserved — only the I/O reactor differs.
    tokio_uring::start(drain_loop(spawn_rx));
}

#[cfg(not(all(target_os = "linux", feature = "io-uring")))]
fn run_event_loop(spawn_rx: flume::Receiver<SpawnRequest>) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(_) => return,
    };
    let local = tokio::task::LocalSet::new();
    runtime.block_on(local.run_until(drain_loop(spawn_rx)));
}

impl Runtime for TokioPerCoreRuntime {
    fn spawn_on_current_core(&self, future: Pin<Box<dyn Future<Output = ()> + 'static>>) {
        // host mode wraps a bare `Handle`, which has no per-core `LocalSet`; the
        // ?Send same-thread path is for server listener loops, not the client
        // dispatch a wrapped handle serves. spawn_local lands it on the current
        // thread's LocalSet (must be inside the host runtime).
        if self.host.is_some() {
            tokio::task::spawn_local(future);
            return;
        }
        CURRENT_CORE.with(|cell| {
            assert!(
                cell.get().is_some(),
                "spawn_on_current_core: not on a TokioPerCoreRuntime worker thread — \
                 use spawn_on_core(N, ...) for cross-core dispatch"
            );
        });
        tokio::task::spawn_local(future);
    }

    fn spawn_on_core(
        &self,
        core_id: CoreId,
        future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
    ) -> Result<(), SpawnError> {
        // wrapped host runtime: dispatch Send work straight onto it. `core_id` is
        // advisory — the host owns scheduling across its own workers. this is the
        // tokio-hosts-proxima path the client's off-worker hop takes.
        if let Some(host) = &self.host {
            host.spawn(future);
            return Ok(());
        }
        let Some(slot) = self.cores.get(core_id.0) else {
            return Err(SpawnError::Disconnected);
        };
        // flume::unbounded never returns Full — the only failure is the
        // receiver being dropped, which means the worker shut down.
        let request = SpawnRequest::Send(future);
        slot.spawn_tx
            .send(request)
            .map_err(|_| SpawnError::Disconnected)
    }

    fn spawn_factory_on_core(
        &self,
        core_id: CoreId,
        factory: Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + 'static>> + Send + 'static>,
    ) -> Result<(), SpawnError> {
        if self.host.is_some() {
            // build the ?Send future on the current thread and spawn_local it
            // (host-runtime LocalSet context required).
            tokio::task::spawn_local(factory());
            return Ok(());
        }
        let Some(slot) = self.cores.get(core_id.0) else {
            return Err(SpawnError::Disconnected);
        };
        let request = SpawnRequest::Factory(factory);
        slot.spawn_tx
            .send(request)
            .map_err(|_| SpawnError::Disconnected)
    }

    fn spawn_background_blocking(
        &self,
        work: Box<dyn FnOnce() -> Result<Box<dyn std::any::Any + Send>, ProximaError> + Send>,
    ) -> BackgroundHandle<Box<dyn std::any::Any + Send>> {
        if let Some(pool) = &self.background_pool {
            return pool.spawn(work);
        }
        let join = match &self.host {
            Some(host) => host.spawn_blocking(work),
            None => tokio::task::spawn_blocking(work),
        };
        Box::pin(async move {
            match join.await {
                Ok(Ok(value)) => Ok(value),
                Ok(Err(err)) => Err(err),
                Err(join_err) => Err(ProximaError::Body(format!(
                    "background task aborted: {join_err}"
                ))),
            }
        })
    }

    fn timer_at(&self, deadline: Instant) -> Pin<Box<dyn Future<Output = ()> + 'static>> {
        let tokio_deadline = tokio::time::Instant::from_std(deadline);
        Box::pin(tokio::time::sleep_until(tokio_deadline))
    }

    fn num_cores(&self) -> usize {
        self.cores.len()
    }

    fn current_core(&self) -> CoreId {
        CURRENT_CORE.with(|cell| match cell.get() {
            Some(id) => id,
            None => panic!("current_core: called from outside a TokioPerCoreRuntime worker thread"),
        })
    }
}

impl Drop for TokioPerCoreRuntime {
    fn drop(&mut self) {
        // Signal shutdown to every worker. If Drop runs on one of our
        // OWN worker threads (e.g., the last Arc<Runtime> ref was held
        // by a listener factory closure that lived on a per-core
        // LocalSet), we can't `join` that worker — `pthread_join` on
        // self returns EDEADLK and panics. Detect it via the
        // thread_local CURRENT_CORE and detach the self-thread's
        // handle instead. Other workers still get joined cleanly.
        let current_core = CURRENT_CORE.with(|cell| cell.get());
        if let Some(cores) = Arc::get_mut(&mut self.cores) {
            for slot in cores.iter() {
                let _ = slot.spawn_tx.send(SpawnRequest::Shutdown);
            }
            for (index, slot) in cores.iter_mut().enumerate() {
                if let Some(handle) = slot.handle.take() {
                    if Some(CoreId(index)) == current_core {
                        // detach — the worker will exit naturally
                        // once it processes the Shutdown we sent.
                        std::mem::forget(handle);
                    } else {
                        let _ = handle.join();
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn new_runtime_spawns_requested_number_of_workers() {
        let runtime = TokioPerCoreRuntime::new(2).expect("build runtime");
        assert_eq!(runtime.num_cores(), 2);
    }

    // P-TU slice 1: a wrapped host runtime dispatches Send work onto the host's
    // own threads, with no proxima-owned worker threads — the tokio-hosts-proxima
    // seam the client's off-worker hop rides.
    #[test]
    fn from_handle_dispatches_send_work_onto_the_host_with_no_new_threads() {
        use std::sync::atomic::AtomicBool;
        use std::time::Duration;

        // the "host": a tokio multi-thread runtime the application already owns.
        let host = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("host-tokio-worker")
            .enable_all()
            .build()
            .expect("host runtime");
        let runtime = TokioPerCoreRuntime::from_handle(host.handle().clone());

        // from a BARE thread (not a proxima worker), dispatch Send work — the
        // exact shape of the client's off-worker hop.
        let ran = Arc::new(AtomicBool::new(false));
        let on_host = Arc::new(AtomicBool::new(false));
        let ran_worker = ran.clone();
        let on_host_worker = on_host.clone();
        runtime
            .spawn_on_core(
                CoreId(0),
                Box::pin(async move {
                    let name = std::thread::current()
                        .name()
                        .unwrap_or_default()
                        .to_string();
                    on_host_worker.store(name.starts_with("host-tokio-worker"), Ordering::Release);
                    ran_worker.store(true, Ordering::Release);
                }),
            )
            .expect("spawn onto host runtime");

        for _ in 0..200 {
            if ran.load(Ordering::Acquire) {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            ran.load(Ordering::Acquire),
            "from_handle dispatched the future onto the host runtime"
        );
        assert!(
            on_host.load(Ordering::Acquire),
            "future ran on a host-tokio worker, not a new proxima thread"
        );
        assert!(
            runtime.cores.is_empty(),
            "from_handle owns no worker threads"
        );
    }

    #[test]
    fn spawn_on_core_dispatches_to_target_worker() {
        let runtime = TokioPerCoreRuntime::new(2).expect("build runtime");
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_core_0 = counter.clone();
        runtime
            .spawn_on_core(
                CoreId(0),
                Box::pin(async move {
                    counter_for_core_0.fetch_add(1, Ordering::SeqCst);
                }),
            )
            .expect("spawn on fresh runtime");
        let counter_for_core_1 = counter.clone();
        runtime
            .spawn_on_core(
                CoreId(1),
                Box::pin(async move {
                    counter_for_core_1.fetch_add(1, Ordering::SeqCst);
                }),
            )
            .expect("spawn on fresh runtime");
        // give the workers a moment to drain the spawn channel.
        for _ in 0..20 {
            if counter.load(Ordering::SeqCst) == 2 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn current_core_inside_worker_returns_dispatched_id() {
        let runtime = TokioPerCoreRuntime::new(2).expect("build runtime");
        let observed: Arc<std::sync::Mutex<Option<CoreId>>> = Arc::new(std::sync::Mutex::new(None));
        let observed_for_task = observed.clone();
        runtime
            .spawn_on_core(
                CoreId(1),
                Box::pin(async move {
                    let id = CURRENT_CORE.with(|cell| cell.get());
                    *observed_for_task.lock().unwrap() = id;
                }),
            )
            .expect("spawn on fresh runtime");
        for _ in 0..20 {
            if observed.lock().unwrap().is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(*observed.lock().unwrap(), Some(CoreId(1)));
    }
}
