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
    ///
    /// Returns only once every worker has reported its executor live. A worker
    /// that cannot build one used to exit quietly, leaving a slot whose sender
    /// still accepted work nothing would ever poll; the readiness handshake
    /// turns that into a construction error.
    ///
    /// # Errors
    /// `ProximaError::Config` if a worker thread cannot be spawned, cannot
    /// build its executor, or dies before reporting.
    pub fn new(num_cores: usize) -> Result<Self, ProximaError> {
        let num_cores = num_cores.max(1);
        let physical = core_affinity::get_core_ids().unwrap_or_default();
        let (ready_tx, ready_rx) = flume::unbounded::<Result<(), String>>();
        let mut cores: Vec<CoreSlot> = Vec::with_capacity(num_cores);
        for index in 0..num_cores {
            let core_id = CoreId(index);
            let affinity = physical.get(index).copied();
            let (spawn_tx, spawn_rx) = flume::unbounded();
            let ready = ready_tx.clone();
            let handle = thread::Builder::new()
                .name(format!("proxima-core-{index}"))
                .spawn(move || worker(core_id, affinity, spawn_rx, &ready))
                .map_err(|err| {
                    ProximaError::Config(format!("spawn per-core worker thread: {err}"))
                })?;
            cores.push(CoreSlot {
                spawn_tx,
                handle: Some(handle),
            });
        }
        drop(ready_tx);
        for _ in 0..num_cores {
            match ready_rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(reason)) => return Err(ProximaError::Config(reason)),
                Err(_) => {
                    return Err(ProximaError::Config(
                        "per-core worker died before reporting readiness".into(),
                    ));
                }
            }
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
    ready: &flume::Sender<Result<(), String>>,
) {
    if let Some(target) = affinity {
        // best-effort: ignore failure (e.g., sandboxed CI runners)
        let _ = core_affinity::set_for_current(target);
    }
    CURRENT_CORE.with(|cell| cell.set(Some(core_id)));

    run_event_loop(spawn_rx, ready);
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
fn run_event_loop(
    spawn_rx: flume::Receiver<SpawnRequest>,
    ready: &flume::Sender<Result<(), String>>,
) {
    // tokio-uring drives its own current-thread runtime + LocalSet
    // backed by io_uring. Owned-buffer I/O, no epoll. The runtime
    // contract (LocalSet for ?Send tasks, current-thread tokio) is
    // preserved — only the I/O reactor differs.
    let _ = ready.send(Ok(()));
    tokio_uring::start(drain_loop(spawn_rx));
}

#[cfg(not(all(target_os = "linux", feature = "io-uring")))]
fn run_event_loop(
    spawn_rx: flume::Receiver<SpawnRequest>,
    ready: &flume::Sender<Result<(), String>>,
) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            let _ = ready.send(Err(format!("build per-core tokio runtime: {err}")));
            return;
        }
    };
    let _ = ready.send(Ok(()));
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
    use std::sync::mpsc::{Receiver, channel};
    use std::time::Duration;

    use super::*;

    /// A worker signalling back is the only ordering these tests need, so they
    /// wait on the value rather than polling for a flag. The deadline exists so
    /// a regression fails with a message instead of hanging the suite; the happy
    /// path never waits on it.
    fn awaited<T>(receiver: &Receiver<T>, what: &str) -> T {
        receiver
            .recv_timeout(Duration::from_secs(10))
            .unwrap_or_else(|err| panic!("{what}: {err}"))
    }

    #[test]
    fn new_runtime_spawns_requested_number_of_workers() {
        let runtime = TokioPerCoreRuntime::new(2).expect("build runtime");
        assert_eq!(runtime.num_cores(), 2);
    }

    /// The readiness handshake: `new` does not return until every worker's
    /// executor is live, so a slot whose executor failed to build surfaces as a
    /// construction error rather than as a lane that silently swallows work.
    #[test]
    fn new_returns_only_once_every_worker_is_live() {
        let runtime = TokioPerCoreRuntime::new(4).expect("build runtime");
        let (sender, receiver) = channel();
        for index in 0..runtime.num_cores() {
            let sender = sender.clone();
            runtime
                .spawn_on_core(
                    CoreId(index),
                    Box::pin(async move {
                        let _ = sender.send(CURRENT_CORE.with(|cell| cell.get()));
                    }),
                )
                .expect("spawn on a live core");
        }
        let mut reported: Vec<CoreId> = (0..4)
            .map(|_| awaited(&receiver, "worker report").expect("worker knows its core"))
            .collect();
        reported.sort_by_key(CoreId::as_usize);
        assert_eq!(reported, vec![CoreId(0), CoreId(1), CoreId(2), CoreId(3)]);
    }

    // P-TU slice 1: a wrapped host runtime dispatches Send work onto the host's
    // own threads, with no proxima-owned worker threads — the tokio-hosts-proxima
    // seam the client's off-worker hop rides.
    #[test]
    fn from_handle_dispatches_send_work_onto_the_host_with_no_new_threads() {
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
        let (sender, receiver) = channel();
        runtime
            .spawn_on_core(
                CoreId(0),
                Box::pin(async move {
                    let name = std::thread::current()
                        .name()
                        .unwrap_or_default()
                        .to_string();
                    let _ = sender.send(name);
                }),
            )
            .expect("spawn onto host runtime");

        let thread_name = awaited(&receiver, "host runtime ran the future");
        assert!(
            thread_name.starts_with("host-tokio-worker"),
            "future ran on a host-tokio worker, not a new proxima thread; got {thread_name}"
        );
        assert!(
            runtime.cores.is_empty(),
            "from_handle owns no worker threads"
        );
    }

    #[test]
    fn spawn_on_core_dispatches_to_target_worker() {
        let runtime = TokioPerCoreRuntime::new(2).expect("build runtime");
        let (sender, receiver) = channel();
        for index in 0..2 {
            let sender = sender.clone();
            runtime
                .spawn_on_core(
                    CoreId(index),
                    Box::pin(async move {
                        let _ = sender.send(index);
                    }),
                )
                .expect("spawn on fresh runtime");
        }
        let mut dispatched = vec![
            awaited(&receiver, "first worker"),
            awaited(&receiver, "second worker"),
        ];
        dispatched.sort_unstable();
        assert_eq!(dispatched, vec![0, 1]);
    }

    #[test]
    fn current_core_inside_worker_returns_dispatched_id() {
        let runtime = TokioPerCoreRuntime::new(2).expect("build runtime");
        let (sender, receiver) = channel();
        runtime
            .spawn_on_core(
                CoreId(1),
                Box::pin(async move {
                    let _ = sender.send(CURRENT_CORE.with(|cell| cell.get()));
                }),
            )
            .expect("spawn on fresh runtime");
        assert_eq!(awaited(&receiver, "core 1 report"), Some(CoreId(1)));
    }
}
