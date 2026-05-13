//! Cross-core graceful shutdown with per-core resource ordering.
//!
//! Folded in from the former `proxima-shutdown` satellite crate. Pipes that
//! hold `!Send` per-core state (open file handles, GPU contexts, FFI
//! bindings, `Rc`/`RefCell` caches) register a drop closure here. The
//! `ShutdownBarrier` broadcasts a drop signal to every core and waits for
//! each core to drain its registered resources in LIFO order — newest
//! registration first, mirroring stack-unwind semantics.
//!
//! ## Tiers
//!
//! [`ResourceRegistry`] is the `no_std` + `alloc` primitive: a per-core LIFO
//! stack addressed by an explicit [`CoreId`], built on
//! [`proxima_core::per_core::PerCore`] — the same `slot(core_id)`/`count()`
//! shape `per_core` already establishes elsewhere in the workspace.
//! [`register_per_core_resource`]/[`drain_current_core`] are the `std`-only
//! ambient convenience on top (OS thread-local, no core id required at the
//! call site) — the same role `PerCore::local` plays for its own primitive.
//! `ShutdownBarrier` itself stays `std`-only: it needs [`crate::sync::Notify`].

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "std")]
use alloc::sync::Arc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
#[cfg(feature = "alloc")]
use core::cell::RefCell;
#[cfg(feature = "std")]
use core::future::Future;
#[cfg(feature = "std")]
use core::pin::Pin;

#[cfg(feature = "std")]
use portable_atomic::{AtomicUsize, Ordering};

#[cfg(feature = "alloc")]
use proxima_core::per_core::PerCore;

#[cfg(feature = "alloc")]
use proxima_runtime::CoreId;
#[cfg(feature = "std")]
use proxima_runtime::Runtime;

#[cfg(feature = "alloc")]
pub type DropHook = Box<dyn FnOnce()>;

// shared by the ambient std stack and `ResourceRegistry`'s explicit-index
// slots — one drain routine, two ways to name which stack to drain.
#[cfg(feature = "alloc")]
fn drain_stack(stack: &RefCell<Vec<(String, DropHook)>>) -> usize {
    let drained = core::mem::take(&mut *stack.borrow_mut());
    let count = drained.len();
    for (_name, hook) in drained.into_iter().rev() {
        hook();
    }
    count
}

/// `no_std` + `alloc` primitive: a per-core LIFO stack of cleanup hooks,
/// addressed by explicit [`CoreId`] rather than ambient thread identity.
/// Bare-metal or prime-style callers that already know their own core id
/// use this directly; [`register_per_core_resource`]/[`drain_current_core`]
/// are the `std`-only, TLS-routed convenience for callers with no core id
/// at hand.
#[cfg(feature = "alloc")]
pub struct ResourceRegistry {
    cores: PerCore<RefCell<Vec<(String, DropHook)>>>,
}

#[cfg(feature = "alloc")]
impl ResourceRegistry {
    #[must_use]
    pub fn new(count: usize) -> Self {
        Self {
            cores: PerCore::new_with(count, |_index| RefCell::new(Vec::new())),
        }
    }

    /// Register a cleanup hook for `core_id`. Fires on the matching
    /// [`Self::drain`]. Multiple registrations under the same `name` are
    /// allowed and all fire.
    pub fn register(&self, core_id: CoreId, name: impl Into<String>, on_drop: DropHook) {
        self.cores
            .slot(core_id.as_usize())
            .borrow_mut()
            .push((name.into(), on_drop));
    }

    /// Drain every hook registered for `core_id`, LIFO. Returns the number
    /// of hooks executed.
    pub fn drain(&self, core_id: CoreId) -> usize {
        drain_stack(self.cores.slot(core_id.as_usize()))
    }

    #[must_use]
    pub fn count(&self) -> usize {
        self.cores.count()
    }
}

#[cfg(feature = "std")]
std::thread_local! {
    /// Per-core LIFO stack of cleanup hooks, keyed by whichever OS thread is
    /// executing. Correct as long as the runtime pins one worker thread per
    /// core — the same invariant `ShutdownBarrier` relies on when it
    /// dispatches drain via `spawn_factory_on_core`. `RefCell` not `Mutex`:
    /// the stack is single-owner (one worker thread per core).
    static RESOURCES: RefCell<Vec<(String, DropHook)>> = const { RefCell::new(Vec::new()) };
}

/// Register a cleanup hook on the *current* core. The hook fires when the
/// barrier broadcasts a drop signal here. Pipe authors call this from
/// inside a `Pipe::call` future (or any per-core context) so the hook
/// captures `!Send` state without ever crossing threads.
///
/// `name` is for diagnostics only; multiple registrations with the same
/// name are allowed and all fire.
#[cfg(feature = "std")]
pub fn register_per_core_resource(name: impl Into<String>, on_drop: DropHook) {
    RESOURCES.with(|stack| stack.borrow_mut().push((name.into(), on_drop)));
}

/// Drain every registered hook on the current core, LIFO. Each hook runs
/// once and is dropped. Returns the number of hooks executed.
#[cfg(feature = "std")]
pub fn drain_current_core() -> usize {
    RESOURCES.with(drain_stack)
}

/// Coordinates a multi-phase shutdown across every core of a `Runtime`.
///
/// Phases:
/// 1. Caller signals listeners to stop accepting (separate path —
///    `ListenerHandle::shutdown_signal()` / `Shutdown::stop()`).
/// 2. Caller awaits in-flight drain via its own counters.
/// 3. Caller invokes `broadcast_drop` here: every core runs
///    `drain_current_core` in its own LocalSet so `!Send` hooks fire on
///    their owning core. The future resolves when every core acks.
/// 4. Caller drops the `Arc<Runtime>` to join worker threads.
#[cfg(feature = "std")]
pub struct ShutdownBarrier {
    runtime: Arc<dyn Runtime>,
}

#[cfg(feature = "std")]
impl ShutdownBarrier {
    #[must_use]
    pub fn new(runtime: Arc<dyn Runtime>) -> Self {
        Self { runtime }
    }

    /// Broadcasts a drop signal to every core and resolves once each
    /// core has finished draining its per-core resources.
    pub fn broadcast_drop(&self) -> Pin<Box<dyn Future<Output = ShutdownReport> + Send>> {
        let num_cores = self.runtime.num_cores();
        let acked = Arc::new(AtomicUsize::new(0));
        let drained_total = Arc::new(AtomicUsize::new(0));
        let notify = Arc::new(crate::sync::Notify::new());
        for core_index in 0..num_cores {
            let acked_for_core = acked.clone();
            let drained_for_core = drained_total.clone();
            let notify_for_core = notify.clone();
            // Send factory crosses the channel; the produced ?Send future
            // runs on the target worker's LocalSet, drains its thread_local
            // resources there, then bumps the ack counter.
            // Shutdown broadcast on inbox-full is a non-event: failing to
            // deliver the drain factory just means that core's resources
            // stay unrouted, the ack never bumps, and the barrier waits
            // out its deadline. The right response is to log and move on —
            // we cannot block startup of shutdown on every core having an
            // empty inbox lane.
            if let Err(err) = self.runtime.spawn_factory_on_core(
                CoreId(core_index),
                Box::new(move || {
                    Box::pin(async move {
                        let count = drain_current_core();
                        drained_for_core.fetch_add(count, Ordering::SeqCst);
                        if acked_for_core.fetch_add(1, Ordering::SeqCst) + 1 == num_cores {
                            notify_for_core.notify_one();
                        }
                    })
                }),
            ) {
                tracing::warn!(
                    core_index,
                    error = %err,
                    "shutdown: drain factory not delivered; core will time out",
                );
            }
        }
        let report_notify = notify;
        let report_acked = acked;
        let report_drained = drained_total;
        Box::pin(async move {
            // Re-check in case every ack already arrived before we started awaiting.
            if report_acked.load(Ordering::SeqCst) < num_cores {
                report_notify.notified().await;
            }
            ShutdownReport {
                cores_acked: report_acked.load(Ordering::SeqCst),
                hooks_drained: report_drained.load(Ordering::SeqCst),
                sources_drained: 0,
                sources_aborted: 0,
            }
        })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShutdownReport {
    /// Number of cores that acknowledged the drop signal.
    pub cores_acked: usize,
    /// Total cleanup hooks fired across all cores.
    pub hooks_drained: usize,
    /// Registered `proxima_primitives::pipe::SourcePipe`s that observed cancellation and
    /// returned within the drain grace window. Folded in by
    /// `proxima::app::Shutdown::drain`; always `0` for a bare
    /// `ShutdownBarrier::broadcast_drop` call, which knows nothing about
    /// sources.
    pub sources_drained: usize,
    /// Registered sources that did NOT return within the grace window and
    /// were aborted. See `sources_drained`.
    pub sources_aborted: usize,
}

#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    #[cfg(all(feature = "runtime-tokio", not(loom)))]
    use std::sync::atomic::AtomicU64;
    #[cfg(all(feature = "runtime-tokio", not(loom)))]
    use std::time::Duration;

    #[test]
    fn lifo_drain_order_matches_registration_reverse() {
        let log: Arc<std::sync::Mutex<Vec<&'static str>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let log_first = log.clone();
        let log_second = log.clone();
        let log_third = log.clone();
        register_per_core_resource(
            "first",
            Box::new(move || log_first.lock().unwrap().push("first")),
        );
        register_per_core_resource(
            "second",
            Box::new(move || log_second.lock().unwrap().push("second")),
        );
        register_per_core_resource(
            "third",
            Box::new(move || log_third.lock().unwrap().push("third")),
        );
        let count = drain_current_core();
        assert_eq!(count, 3);
        let order = log.lock().unwrap().clone();
        assert_eq!(
            order,
            vec!["third", "second", "first"],
            "LIFO order required"
        );
        assert_eq!(
            drain_current_core(),
            0,
            "second drain must observe empty stack"
        );
    }

    #[test]
    fn resource_registry_lifo_drain_order_matches_registration_reverse() {
        let registry = ResourceRegistry::new(2);
        let log: Arc<std::sync::Mutex<Vec<&'static str>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let log_first = log.clone();
        let log_second = log.clone();
        registry.register(
            CoreId(1),
            "first",
            Box::new(move || log_first.lock().unwrap().push("first")),
        );
        registry.register(
            CoreId(1),
            "second",
            Box::new(move || log_second.lock().unwrap().push("second")),
        );

        assert_eq!(registry.count(), 2);
        assert_eq!(registry.drain(CoreId(0)), 0, "core 0 got no registrations");
        assert_eq!(registry.drain(CoreId(1)), 2);
        assert_eq!(*log.lock().unwrap(), vec!["second", "first"]);
        assert_eq!(
            registry.drain(CoreId(1)),
            0,
            "second drain observes empty stack"
        );
    }

    #[cfg(all(feature = "runtime-tokio", not(loom)))]
    #[test]
    fn broadcast_drop_fires_hooks_on_every_core() {
        use proxima_runtime::tokio::TokioPerCoreRuntime;

        let runtime = Arc::new(TokioPerCoreRuntime::new(2).expect("build runtime"));
        let runtime_for_barrier: Arc<dyn Runtime> = runtime.clone();
        let drops_observed = Arc::new(AtomicU64::new(0));
        let registered = Arc::new(AtomicU64::new(0));
        let registered_notify = Arc::new(crate::sync::Notify::new());

        // Register one drop hook on each core's thread_local. We dispatch
        // a registration task to each core; the closure pushes onto its
        // own per-core stack. Each factory bumps `registered` on completion
        // so the test can wait for both registrations deterministically
        // instead of guessing at a sleep duration.
        for core_index in 0..2 {
            let drops_for_core = drops_observed.clone();
            let registered_for_core = registered.clone();
            let notify_for_core = registered_notify.clone();
            runtime
                .spawn_factory_on_core(
                    CoreId(core_index),
                    Box::new(move || {
                        Box::pin(async move {
                            register_per_core_resource(
                                format!("core-{core_index}"),
                                Box::new(move || {
                                    drops_for_core.fetch_add(1, Ordering::SeqCst);
                                }),
                            );
                            if registered_for_core.fetch_add(1, Ordering::SeqCst) + 1 == 2 {
                                notify_for_core.notify_one();
                            }
                        })
                    }),
                )
                .expect("test-time spawn must succeed on a fresh runtime");
        }

        futures::executor::block_on(async {
            if registered.load(Ordering::SeqCst) < 2 {
                proxima_core::time::timeout(Duration::from_secs(1), registered_notify.notified())
                    .await
                    .expect("registration hooks must complete within timeout");
            }
        });

        let barrier = ShutdownBarrier::new(runtime_for_barrier);
        let report = futures::executor::block_on(barrier.broadcast_drop());
        assert_eq!(report.cores_acked, 2);
        assert_eq!(report.hooks_drained, 2);
        assert_eq!(drops_observed.load(Ordering::SeqCst), 2);
    }
}
