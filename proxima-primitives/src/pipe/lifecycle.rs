//! Background-task lifecycle for [`SourcePipe`](crate::pipe::source::SourcePipe)s.
//!
//! A `SourcePipe` is a `Signal -> ()` pipe that runs an unbounded loop and
//! observes a cancellation [`Signal`] cooperatively — the signal it receives
//! IS the loop's cancellation arm, not a wrapper's. This module is the
//! runtime that *spawns* registered sources, owns the shared cancellation
//! tree, and drains the join set on shutdown. It lives alongside
//! [`SourcePipe`](crate::pipe::source::SourcePipe) so the lifecycle sits
//! adjacent to the trait it serves — no downstream crate has to reinvent
//! shutdown-grace or panic-propagation.
//!
//! # Composed primitives
//!
//! - [`crate::sync::task::JoinSet`] — task set (`spawn` / `join_next`).
//!   Under `runtime-tokio` a thin forwarder over `tokio::task::JoinSet`
//!   (identical behaviour to the pre-migration direct `tokio::task::JoinSet`
//!   use); otherwise a tokio-free OS-thread-per-task backing so this stays
//!   reachable from the default, tokio-free build.
//! - [`proxima_core::signal::Signal`] — cooperative cancellation
//!   tree. Parent tokens propagate to children via [`with_parent_signal`].
//! - [`proxima_core::time::timeout`] — runtime-agnostic grace deadline.
//! - [`crate::pipe::source::SourceHandle`] — the erased source a caller registers
//!   via `spawn_from_source`; the same [`Signal`] instance it receives is
//!   the one the caller fires from [`shutdown`].
//!
//! [`with_parent_signal`]: ProducerLifecycle::with_parent_signal

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::time::Duration;

use proxima_core::signal::Signal;
use tracing::{error, trace, warn};

use crate::pipe::SendPipe;
use crate::pipe::source::SourceHandle;
use crate::sync::task::JoinSet;

/// Manages background-task lifecycle for a set of [`SourceHandle`]s.
///
/// The lifecycle:
/// 1. Construct with [`ProducerLifecycle::new`] (or [`with_parent_signal`] to
///    inherit a shared cancellation tree).
/// 2. For each registered source, call [`spawn_from_source`] — the source's
///    own loop observes the lifecycle's [`Signal`] cooperatively and returns
///    once it has drained.
/// 3. Hand callers a clone of [`cancel_signal`] if they need to coordinate
///    their own shutdown observation.
/// 4. On shutdown, call [`shutdown`] with a grace [`Duration`]. The signal is
///    fired; tasks that finish within the grace count as `drained`;
///    tasks that don't are aborted and count as `aborted`.
///
/// [`with_parent_signal`]: ProducerLifecycle::with_parent_signal
/// [`spawn_from_source`]: ProducerLifecycle::spawn_from_source
/// [`cancel_signal`]: ProducerLifecycle::cancel_signal
/// [`shutdown`]: ProducerLifecycle::shutdown
pub struct ProducerLifecycle {
    tasks: JoinSet<()>,
    cancel: Signal,
    spawned_names: Vec<Arc<str>>,
}

impl ProducerLifecycle {
    /// Construct a new lifecycle with a fresh root cancellation token.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tasks: JoinSet::new(),
            cancel: Signal::new(),
            spawned_names: Vec::new(),
        }
    }

    /// Construct from a parent token; cancelling the parent cancels this
    /// lifecycle's child token too. Useful when a higher-level shutdown
    /// (e.g. the app-level oneshot signal) owns the root token.
    #[must_use]
    pub fn with_parent_signal(parent: &Signal) -> Self {
        Self {
            tasks: JoinSet::new(),
            cancel: parent.child(),
            spawned_names: Vec::new(),
        }
    }

    /// Return a clone of the cancellation token. Callers that need to observe
    /// shutdown alongside the spawned tasks (e.g. the listener serve loop)
    /// can race against this.
    #[must_use]
    pub fn cancel_signal(&self) -> Signal {
        self.cancel.clone()
    }

    /// Spawn `source` under this lifecycle's shared cancellation [`Signal`].
    ///
    /// Unlike a wrapper-cancelled task, the source's own loop receives the
    /// lifecycle's real `Signal` as its `call` input and observes it
    /// cooperatively (see `run_interval_loop`'s `select_biased!` against
    /// `cancel.fired()`) — cancellation is not drop-based, the source returns
    /// on its own once it has drained.
    pub fn spawn_from_source(&mut self, name: &str, source: &SourceHandle) {
        let label: Arc<str> = Arc::from(name);
        let cancel = self.cancel.clone();
        let source = source.clone();
        self.spawned_names.push(label.clone());
        self.tasks.spawn(async move {
            match SendPipe::call(&source, cancel).await {
                Ok(()) => trace!(source = %label, "producer-lifecycle source completed"),
                Err(err) => error!(?err, source = %label, "producer-lifecycle source failed"),
            }
        });
    }

    /// Number of tasks currently being managed.
    #[must_use]
    pub fn task_count(&self) -> usize {
        self.spawned_names.len()
    }

    /// Names of all spawned tasks (in spawn order), per [`spawn_from_source`].
    ///
    /// [`spawn_from_source`]: ProducerLifecycle::spawn_from_source
    pub fn spawned_task_names(&self) -> impl Iterator<Item = &str> {
        self.spawned_names.iter().map(AsRef::as_ref)
    }

    /// Initiate graceful shutdown.
    ///
    /// 1. Cancel the lifecycle's token so cooperating tasks observe cancellation.
    /// 2. Drain the join set with the given `grace` deadline. Tasks that exit
    ///    within the grace count as `drained`; tasks that don't are aborted
    ///    and count as `aborted`. Tasks that panicked are noted in
    ///    [`ShutdownReport::panics`] and count as `aborted` (the task did
    ///    not drain cleanly).
    ///
    /// Consumes `self`, so the lifecycle cannot be re-used after shutdown.
    pub async fn shutdown(mut self, grace: Duration) -> ShutdownReport {
        let total = self.spawned_names.len();
        self.cancel.fire();

        let mut drained = 0usize;
        let mut panics = 0usize;
        let drain_complete = proxima_core::time::timeout(grace, async {
            while let Some(result) = self.tasks.join_next().await {
                match result {
                    Ok(()) => drained += 1,
                    Err(err) if err.is_cancelled() => {
                        // cancelled via abort_all in the timeout branch
                    }
                    Err(err) if err.is_panic() => {
                        panics += 1;
                        error!(?err, "producer-lifecycle task panicked");
                    }
                    Err(err) => {
                        warn!(
                            ?err,
                            "producer-lifecycle task ended with non-panic, non-cancel JoinError"
                        );
                    }
                }
            }
        })
        .await;

        if drain_complete.is_err() {
            // grace exceeded — abort remaining tasks and reap their handles
            self.tasks.abort_all();
            while let Some(result) = self.tasks.join_next().await {
                if let Err(err) = result
                    && err.is_panic()
                {
                    panics += 1;
                    error!(?err, "producer-lifecycle task panicked during abort");
                }
            }
        }

        let aborted = total.saturating_sub(drained);
        ShutdownReport {
            total,
            drained,
            aborted,
            panics,
        }
    }
}

impl Default for ProducerLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for ProducerLifecycle {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ProducerLifecycle")
            .field("task_count", &self.spawned_names.len())
            .field("cancelled", &self.cancel.is_fired())
            .finish()
    }
}

/// Outcome of a [`ProducerLifecycle::shutdown`] call.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ShutdownReport {
    /// Total tasks the lifecycle was managing at shutdown time.
    pub total: usize,
    /// Tasks that exited cleanly (cooperatively, within the grace deadline).
    pub drained: usize,
    /// Tasks that did NOT exit within the grace deadline and were aborted.
    pub aborted: usize,
    /// Tasks that ended with a panic (counted as part of [`aborted`]).
    ///
    /// [`aborted`]: ShutdownReport::aborted
    pub panics: usize,
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]

    use alloc::sync::Arc;
    use core::future::Future;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use core::time::Duration;

    use futures::FutureExt;
    use proxima_core::ProximaError;

    use crate::pipe::source::{SourceHandle, into_source_handle};

    use super::*;

    struct NoopSource;

    impl SendPipe for NoopSource {
        type In = Signal;
        type Out = ();
        type Err = ProximaError;

        fn call(&self, _cancel: Signal) -> impl Future<Output = Result<(), ProximaError>> + Send {
            async { Ok(()) }
        }
    }

    /// Awaits `cancel.fired()` then records the observation — proves
    /// cooperative cancellation reaches the source's own loop.
    struct CooperativeSource {
        observed: Arc<AtomicUsize>,
    }

    impl SendPipe for CooperativeSource {
        type In = Signal;
        type Out = ();
        type Err = ProximaError;

        fn call(&self, cancel: Signal) -> impl Future<Output = Result<(), ProximaError>> + Send {
            let observed = self.observed.clone();
            async move {
                cancel.fired().await;
                observed.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }
    }

    /// Races a never-ending body against `cancel.fired()`, mirroring
    /// `run_interval_loop`'s own `select_biased!` — the source itself owns
    /// cancellation, not a lifecycle wrapper.
    struct LongRunningSource;

    impl SendPipe for LongRunningSource {
        type In = Signal;
        type Out = ();
        type Err = ProximaError;

        fn call(&self, cancel: Signal) -> impl Future<Output = Result<(), ProximaError>> + Send {
            async move {
                let body = core::future::pending::<()>().fuse();
                let cancel_fut = cancel.fired().fuse();
                futures::pin_mut!(body, cancel_fut);
                futures::select_biased! {
                    () = cancel_fut => Ok(()),
                    () = body => Ok(()),
                }
            }
        }
    }

    struct PanickingSource;

    impl SendPipe for PanickingSource {
        type In = Signal;
        type Out = ();
        type Err = ProximaError;

        fn call(&self, _cancel: Signal) -> impl Future<Output = Result<(), ProximaError>> + Send {
            async { panic!("intentional test panic") }
        }
    }

    #[proxima::test]
    async fn empty_lifecycle_shutdown_reports_zero_total() {
        let lifecycle = ProducerLifecycle::new();
        let report = lifecycle.shutdown(Duration::from_millis(100)).await;
        assert_eq!(
            report,
            ShutdownReport {
                total: 0,
                drained: 0,
                aborted: 0,
                panics: 0,
            }
        );
    }

    #[proxima::test]
    async fn spawned_source_completes_cleanly_drains_within_grace() {
        let source: SourceHandle = into_source_handle(NoopSource);

        let mut lifecycle = ProducerLifecycle::new();
        lifecycle.spawn_from_source("noop_producer", &source);
        assert_eq!(lifecycle.task_count(), 1);

        let report = lifecycle.shutdown(Duration::from_secs(1)).await;
        assert_eq!(report.total, 1);
        assert_eq!(report.drained, 1);
        assert_eq!(report.aborted, 0);
        assert_eq!(report.panics, 0);
    }

    #[proxima::test]
    async fn cooperative_source_respects_cancellation_token_drained_count_one() {
        let observed = Arc::new(AtomicUsize::new(0));
        let source: SourceHandle = into_source_handle(CooperativeSource {
            observed: observed.clone(),
        });

        let mut lifecycle = ProducerLifecycle::new();
        lifecycle.spawn_from_source("cooperating_producer", &source);

        let report = lifecycle.shutdown(Duration::from_secs(1)).await;
        assert_eq!(report.total, 1);
        assert_eq!(report.drained, 1);
        assert_eq!(report.aborted, 0);
        assert_eq!(observed.load(Ordering::SeqCst), 1);
    }

    #[proxima::test]
    async fn long_running_source_drained_promptly_by_its_own_select_on_cancel() {
        // The source races its own body against cancel.fired(). On shutdown
        // the cancel arm wins, so the source returns Ok(()) cleanly.
        // Drained == 1, aborted == 0 — a grace deadline is a safety net for
        // the pathological case where a source never observes cancel at all.
        let source: SourceHandle = into_source_handle(LongRunningSource);

        let mut lifecycle = ProducerLifecycle::new();
        lifecycle.spawn_from_source("long_running_producer", &source);

        let started = std::time::Instant::now();
        let report = lifecycle.shutdown(Duration::from_millis(500)).await;
        let elapsed = started.elapsed();

        assert!(
            elapsed < Duration::from_millis(500),
            "shutdown should drain via the source's own select! well before the grace deadline: {elapsed:?}"
        );
        assert_eq!(report.total, 1);
        assert_eq!(
            report.drained, 1,
            "the source exits cleanly on cancel via its own select!"
        );
        assert_eq!(report.aborted, 0);
        assert_eq!(report.panics, 0);
    }

    #[proxima::test]
    async fn multiple_sources_spawn_and_drain_separately() {
        let mut lifecycle = ProducerLifecycle::new();
        lifecycle.spawn_from_source("a", &into_source_handle(NoopSource));
        lifecycle.spawn_from_source("b", &into_source_handle(NoopSource));
        lifecycle.spawn_from_source("c", &into_source_handle(NoopSource));
        assert_eq!(lifecycle.task_count(), 3);

        let names: Vec<&str> = lifecycle.spawned_task_names().collect();
        assert_eq!(names, alloc::vec!["a", "b", "c"]);

        let report = lifecycle.shutdown(Duration::from_secs(1)).await;
        assert_eq!(report.drained, 3);
    }

    #[proxima::test]
    async fn source_panic_is_counted_but_does_not_crash_lifecycle() {
        let mut lifecycle = ProducerLifecycle::new();
        lifecycle.spawn_from_source("oops", &into_source_handle(PanickingSource));
        lifecycle.spawn_from_source("ok", &into_source_handle(NoopSource));

        let report = lifecycle.shutdown(Duration::from_secs(1)).await;
        assert_eq!(report.total, 2);
        assert_eq!(report.panics, 1, "one source should have panicked");
        assert_eq!(report.drained + report.aborted, 2);
    }

    #[proxima::test]
    async fn parent_token_cancellation_propagates_to_child_lifecycle() {
        let parent = Signal::new();
        let mut lifecycle = ProducerLifecycle::with_parent_signal(&parent);

        let observed = Arc::new(AtomicUsize::new(0));
        let source: SourceHandle = into_source_handle(CooperativeSource {
            observed: observed.clone(),
        });
        lifecycle.spawn_from_source("watcher_producer", &source);

        parent.fire();

        let report = lifecycle.shutdown(Duration::from_secs(1)).await;
        assert_eq!(report.drained, 1);
        assert_eq!(observed.load(Ordering::SeqCst), 1);
    }

    #[proxima::test]
    async fn debug_impl_reports_task_count_and_cancellation_state() {
        let mut lifecycle = ProducerLifecycle::new();
        lifecycle.spawn_from_source("dbg_producer", &into_source_handle(NoopSource));

        let pre_shutdown = alloc::format!("{lifecycle:?}");
        assert!(
            pre_shutdown.contains("task_count: 1"),
            "got: {pre_shutdown}"
        );
        assert!(pre_shutdown.contains("cancelled: false"));

        let _report = lifecycle.shutdown(Duration::from_secs(1)).await;
    }
}
