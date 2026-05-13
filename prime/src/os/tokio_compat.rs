//! P2 — Prime+Tokio compat mode plumbing.
//!
//! Per [`prime`](super::super)'s design (see `../../mod.rs`), prime is per-
//! core-sharded with its own reactor and executor; it does NOT expose a
//! tokio runtime context. User code that imports `tokio::spawn`,
//! `tokio::sync::*`, or `tokio::time::*` panics at runtime when those
//! APIs look up `tokio::runtime::Handle::current()` from inside a prime
//! task — there is no current tokio runtime.
//!
//! Compat mode fixes this without changing the user's imports. The
//! mechanism:
//!
//! 1. Per prime core, build a `tokio::runtime::Builder::new_current_thread()
//!    .enable_all().build()`. Run it on its own dedicated OS thread
//!    (`tokio-compat-<core>-driver`) where it `block_on(future::pending())`,
//!    keeping its scheduler + reactor + timer driver alive for the prime
//!    runtime's lifetime.
//! 2. Stash each sister runtime's `Handle` on the matching prime worker
//!    via [`core_shard::launch_with_lanes_and_setup`]'s `WorkerSetup`
//!    hook. The worker leaks the `Handle` to get `&'static Handle`, calls
//!    `.enter()` to obtain `EnterGuard<'static>`, and holds the guard on
//!    its stack for the rest of its life.
//! 3. With the guard live, `tokio::runtime::Handle::current()` returns the
//!    sister's handle. `tokio::spawn(future)` dispatches `future` to the
//!    sister's executor (running on the sister OS thread); `tokio::time::*`
//!    drivers point at the sister's timer; `tokio::net::*` use the sister's
//!    mio reactor.
//!
//! Cost picture, documented honestly:
//!
//! - Per core: one extra OS thread (sister tokio driver), one tokio
//!   current-thread runtime (its scheduler + mio reactor + timer wheel).
//!   Compared to pure prime: ~1 MB RSS per core for the tokio runtime
//!   state, plus the sister thread's stack (~2 MB default).
//! - Per `tokio::spawn` from a prime task: the spawned future executes
//!   on the sister thread, not the prime thread. Cross-thread wake hops
//!   are inherent. Locality is preserved if the sister is core-pinned
//!   next to its prime worker (this module does best-effort pinning via
//!   `core_affinity` when the prime worker has affinity set).
//! - Reactor split: prime keeps its native reactor for proxima's accept
//!   loops + native I/O (`prime::os::net::TcpListener`); tokio I/O API
//!   calls (`tokio::net::*`) use tokio's mio. Two reactors per core —
//!   wasted syscall budget for users running pure-tokio I/O on compat.
//!
//! The bench harness at `rust/benches/bench_runtime_compat.rs` measures
//! the cost. The plan's P2 ship criteria (see
//! `rust/docs/runtime-prime/discipline-prime-tokio-compat.md`) decide
//! whether compat lands or parks.

#![cfg(feature = "prime-tokio-compat")]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use proxima_core::ProximaError;
use proxima_runtime::{CoreId, SpawnError};
use tokio::sync::mpsc;

use super::core_shard::WorkerSetup;

/// A `Send` task dispatched onto a sister runtime via the batched
/// [`TokioCompatHandles::spawn_on_core`] path.
type CompatTask = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

/// Per-core sister tokio runtimes that back compat mode.
///
/// Owns one `tokio::runtime::Runtime` per prime core (current-thread,
/// `enable_all`), each driven on a dedicated `tokio-compat-<core>-driver`
/// OS thread. Provides [`Self::worker_setup`] to hand each prime worker
/// its sister's `Handle` for thread-local re-entry.
///
/// Lives behind an `Arc` inside `PrimeRuntime`; drop tears down all
/// sister threads via [`Self::shutdown`].
pub struct TokioCompatHandles {
    cores: Vec<TokioCompatCore>,
}

struct TokioCompatCore {
    handle: tokio::runtime::Handle,
    /// batched dispatch channel into the sister's drain loop. tasks sent
    /// here are `tokio::task::spawn`ed by the drain loop ON the sister's
    /// own driver thread (the local fast path — no remote `unpark`
    /// syscall per task, unlike `Handle::spawn` from a foreign thread).
    /// also the liveness anchor: dropping every sender closes the channel,
    /// the drain loop's `recv` returns `None`, and the driver thread exits.
    task_tx: mpsc::UnboundedSender<CompatTask>,
    driver_thread: Option<JoinHandle<()>>,
}

impl TokioCompatHandles {
    /// Spawn `num_cores` sister tokio runtimes. Each runs on its own
    /// `tokio-compat-<core>-driver` OS thread, keeps a current-thread
    /// scheduler + mio reactor + timer driver alive, and is best-effort
    /// pinned to the same physical core as its prime worker.
    pub fn new(num_cores: usize) -> Result<Arc<Self>, ProximaError> {
        let physical = core_affinity::get_core_ids().unwrap_or_default();
        let mut cores = Vec::with_capacity(num_cores);
        for index in 0..num_cores {
            let affinity = physical.get(index).copied();
            cores.push(spawn_compat_core(CoreId(index), affinity)?);
        }
        Ok(Arc::new(Self { cores }))
    }

    /// Borrow the sister `Handle` for `core_id`. Used by P2-aware
    /// dispatch paths that need to publish the handle without entering
    /// it (rare; the common path is `worker_setup`).
    #[must_use]
    pub fn handle(&self, core_id: CoreId) -> Option<&tokio::runtime::Handle> {
        self.cores.get(core_id.0).map(|core| &core.handle)
    }

    /// Dispatch `future` onto the sister runtime for `core_id` via the
    /// batched channel. The sister's drain loop `tokio::task::spawn`s it
    /// locally, so a burst of N tasks costs ~one `unpark` (first task wakes
    /// the parked drain loop; the rest land while it is already runnable),
    /// not N — the difference that makes `Handle::spawn`-per-task pay a
    /// `kevent` syscall on every call from a foreign thread.
    ///
    /// `Disconnected` means the core is out of range or its driver thread
    /// has shut down; the future is dropped.
    pub fn spawn_on_core(&self, core_id: CoreId, future: CompatTask) -> Result<(), SpawnError> {
        let core = self.cores.get(core_id.0).ok_or(SpawnError::Disconnected)?;
        core.task_tx
            .send(future)
            .map_err(|_| SpawnError::Disconnected)
    }

    /// Build a [`WorkerSetup`] closure for `core_id`. Pass to
    /// [`core_shard::launch_with_lanes_and_setup`] so the prime worker
    /// thread enters the sister tokio handle for its lifetime.
    ///
    /// The closure leaks one `Box<tokio::runtime::Handle>` (a single
    /// `Arc` clone of the sister runtime — a few dozen bytes) so the
    /// `EnterGuard` can be `'static`. The runtime itself is owned by
    /// `TokioCompatHandles`, so the leaked handle stays valid until
    /// process exit.
    #[must_use]
    pub fn worker_setup(&self, core_id: CoreId) -> Option<WorkerSetup> {
        let handle = self.cores.get(core_id.0).map(|core| core.handle.clone())?;
        Some(Box::new(move || -> Box<dyn std::any::Any> {
            // leak the Handle clone so the EnterGuard's borrow is
            // 'static. one leak per prime worker; bounded by num_cores.
            // EnterGuard is !Send (manipulates a tokio thread-local),
            // which is the exact reason WorkerSetup's returned token
            // is not Send-bound.
            let leaked: &'static tokio::runtime::Handle = Box::leak(Box::new(handle));
            Box::new(leaked.enter())
        }))
    }
}

impl Drop for TokioCompatHandles {
    fn drop(&mut self) {
        // Drain each core: dropping the last `task_tx` closes the channel,
        // so the sister's drain loop `recv` returns `None` and the
        // `block_on` exits cleanly. Then join the driver thread.
        for core in std::mem::take(&mut self.cores) {
            drop(core.task_tx);
            if let Some(thread) = core.driver_thread {
                let _ = thread.join();
            }
        }
    }
}

fn spawn_compat_core(
    core_id: CoreId,
    affinity: Option<core_affinity::CoreId>,
) -> Result<TokioCompatCore, ProximaError> {
    let (handle_tx, handle_rx) = std::sync::mpsc::channel::<tokio::runtime::Handle>();
    let (task_tx, mut task_rx) = mpsc::unbounded_channel::<CompatTask>();

    let driver_thread = thread::Builder::new()
        .name(format!("tokio-compat-{}-driver", core_id.0))
        .spawn(move || {
            if let Some(target) = affinity {
                // best-effort pin to the same physical core as the prime
                // worker. preserves locality across the sister hop.
                let valid = core_affinity::get_core_ids()
                    .map(|ids| ids.iter().any(|cid| cid.id == target.id))
                    .unwrap_or(false);
                if valid {
                    let _ = core_affinity::set_for_current(target);
                }
            }
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .thread_name(format!("tokio-compat-{}", core_id.0))
                .build()
            {
                Ok(value) => value,
                Err(_) => return,
            };
            let _ = handle_tx.send(runtime.handle().clone());
            // drive the runtime on a drain loop: keeps the scheduler + mio
            // reactor + timer driver alive for `Handle::current()` calls
            // routed from prime workers via the EnterGuard, AND services the
            // batched dispatch channel. each task is `spawn`ed from this
            // (the sister's own) thread, so it takes tokio's local fast path
            // — no remote `unpark` per task. the loop exits when every
            // `task_tx` is dropped (channel closed → `recv` yields `None`).
            runtime.block_on(async move {
                while let Some(task) = task_rx.recv().await {
                    tokio::task::spawn(task);
                }
            });
        })
        .map_err(|err| {
            ProximaError::Config(format!("spawn tokio-compat-{}-driver: {err}", core_id.0))
        })?;

    let handle = handle_rx.recv().map_err(|err| {
        ProximaError::Config(format!(
            "tokio-compat-{} handle channel closed before driver started: {err}",
            core_id.0
        ))
    })?;

    Ok(TokioCompatCore {
        handle,
        task_tx,
        driver_thread: Some(driver_thread),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn new_spawns_one_runtime_per_core() {
        let handles = TokioCompatHandles::new(2).expect("build handles");
        assert!(handles.handle(CoreId(0)).is_some());
        assert!(handles.handle(CoreId(1)).is_some());
        assert!(handles.handle(CoreId(2)).is_none());
    }

    #[test]
    fn handle_block_on_runs_async_work() {
        let handles = TokioCompatHandles::new(1).expect("build handles");
        let handle = handles.handle(CoreId(0)).expect("core 0 handle").clone();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_task = counter.clone();
        // dispatch a tokio::spawn through the sister handle. completion is
        // observed via atomic; tokio::time::sleep is reachable via the
        // sister's timer driver because `enable_all()` is set.
        let join = handle.spawn(async move {
            tokio::time::sleep(Duration::from_millis(5)).await;
            counter_for_task.fetch_add(1, Ordering::AcqRel);
        });
        // block on the join from the test thread via a fresh runtime to
        // avoid coupling the test to any ambient tokio context.
        let fresh = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("fresh test runtime");
        fresh.block_on(async move {
            let _ = join.await;
        });
        assert_eq!(counter.load(Ordering::Acquire), 1);
    }

    #[test]
    fn drop_terminates_driver_threads() {
        // the previous handles instance must shut its driver threads down
        // when dropped; this test ensures no panic/leak in the teardown
        // path. observable: the test exits cleanly without leaking
        // threads (criterion / other tests are unaffected).
        {
            let _handles = TokioCompatHandles::new(2).expect("build handles");
        }
    }

    #[test]
    fn spawn_on_core_runs_every_batched_task() {
        let handles = TokioCompatHandles::new(2).expect("build handles");
        let total = 256_usize;
        // each task drops a sender clone after running; recv blocks until
        // a task runs or all clones are gone — deterministic, no sleep.
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        for index in 0..total {
            let done_tx = done_tx.clone();
            handles
                .spawn_on_core(
                    CoreId(index % 2),
                    Box::pin(async move {
                        let _ = done_tx.send(());
                    }),
                )
                .expect("dispatch to in-range core");
        }
        drop(done_tx);
        let mut seen = 0_usize;
        while done_rx.recv().is_ok() {
            seen += 1;
        }
        assert_eq!(seen, total, "every batched task must run on its sister");
    }

    #[test]
    fn spawn_on_core_rejects_out_of_range_core() {
        let handles = TokioCompatHandles::new(1).expect("build handles");
        assert_eq!(
            handles.spawn_on_core(CoreId(4), Box::pin(async {})),
            Err(SpawnError::Disconnected)
        );
    }
}
