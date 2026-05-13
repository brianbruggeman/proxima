//! `proxima_primitives::sync::task` — runtime-agnostic task primitives shaped like
//! `tokio::task`. Folded from the former `proxima-task` crate (Workstream
//! F, RISC-dedup) so other proxima leaf crates (proxima-middleware,
//! proxima-listeners-*, prime, etc.) reach it via the same crate as the
//! rest of proxima's concurrency surface, without circling back through
//! the top-level proxima crate.
//!
//! Surface:
//!
//! - [`yield_now`] — cooperative scheduling hook. Pure re-export of
//!   `futures::task::yield_now`; runtime-agnostic.
//! - [`JoinSet`] — under `runtime-tokio`, a thin newtype over
//!   `tokio::task::JoinSet`. Without it (the default, tokio-free build),
//!   a portable backing: one OS thread per spawned task running
//!   `futures::executor::block_on`, with completions delivered through an
//!   mpsc channel + `Notify` so `join_next` stays a non-blocking `.await`.
//!   Real independent background progress without depending on the
//!   caller's async runtime — the same functional contract as tokio's
//!   `JoinSet`, minus true task abortion (an OS thread can't be forcibly
//!   stopped; `abort_all` stops *waiting* on outstanding tasks instead).

/// Cooperative yield. The returned future yields exactly once on
/// first poll (`Pending` + waker-poke), then resolves on the next
/// poll. Same semantic as `tokio::task::yield_now`, runtime-agnostic.
pub fn yield_now() -> YieldNow {
    YieldNow { yielded: false }
}

/// Future returned by [`yield_now`]. Named so callers can store it.
#[derive(Debug)]
pub struct YieldNow {
    yielded: bool,
}

impl core::future::Future for YieldNow {
    type Output = ();

    fn poll(
        mut self: core::pin::Pin<&mut Self>,
        context: &mut core::task::Context<'_>,
    ) -> core::task::Poll<()> {
        if self.yielded {
            core::task::Poll::Ready(())
        } else {
            self.yielded = true;
            context.waker().wake_by_ref();
            core::task::Poll::Pending
        }
    }
}

// `runtime-tokio` can be requested workspace-wide by a sibling crate
// (see the root Cargo.toml's `proxima-primitives = { features = ["runtime-tokio"] }`)
// independent of any `-p proxima-primitives` scoping; `not(loom)` keeps this
// tokio-backed module out of the loom build regardless, since its
// backing `dep:tokio` is itself `[target.'cfg(not(loom))'.dependencies]`.
#[cfg(all(feature = "runtime-tokio", not(loom)))]
pub use join_set::{JoinError, JoinSet};

#[cfg(all(feature = "std", any(not(feature = "runtime-tokio"), loom)))]
pub use portable_join_set::{AbortHandle, JoinError, JoinSet};

#[cfg(all(feature = "runtime-tokio", not(loom)))]
mod join_set {
    //! Newtype over `tokio::task::JoinSet` that forwards `spawn` /
    //! `join_next` / `abort_all` / `len` / `is_empty`. The whole point
    //! is to give callers a `proxima::task::JoinSet` import so they
    //! aren't reaching into `tokio::task::` directly — substrate
    //! coupling stays inside this module.

    use std::future::Future;

    pub use tokio::task::JoinError;

    /// Set of spawned tasks; pull completions in arrival order via
    /// `join_next`. Matches `tokio::task::JoinSet`'s shape.
    #[derive(Debug)]
    pub struct JoinSet<T>(tokio::task::JoinSet<T>);

    impl<T: 'static> JoinSet<T> {
        /// New empty set.
        #[must_use]
        pub fn new() -> Self {
            Self(tokio::task::JoinSet::new())
        }

        /// Spawn `future` into the set. Requires `T: Send + 'static`
        /// and `future: Send + 'static`.
        pub fn spawn<F>(&mut self, future: F) -> tokio::task::AbortHandle
        where
            F: Future<Output = T> + Send + 'static,
            T: Send,
        {
            self.0.spawn(future)
        }

        /// Await the next task to complete. Returns `None` when the
        /// set is empty.
        pub async fn join_next(&mut self) -> Option<Result<T, JoinError>> {
            self.0.join_next().await
        }

        /// Abort every spawned task. Pending `join_next` calls then
        /// receive `Err(JoinError)` with `is_cancelled() == true`.
        pub fn abort_all(&mut self) {
            self.0.abort_all();
        }

        /// Number of tasks still in flight.
        #[must_use]
        pub fn len(&self) -> usize {
            self.0.len()
        }

        /// `true` iff `len() == 0`.
        #[must_use]
        pub fn is_empty(&self) -> bool {
            self.0.is_empty()
        }
    }

    impl<T: 'static> Default for JoinSet<T> {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(test)]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    mod tests {
        use super::*;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;

        #[proxima::test(runtime = "tokio")]
        async fn spawn_then_join_next_returns_value() {
            let mut set: JoinSet<u32> = JoinSet::new();
            set.spawn(async { 42 });
            let outcome = set.join_next().await.expect("one in flight");
            assert_eq!(outcome.expect("not cancelled"), 42);
            assert!(set.is_empty());
        }

        #[proxima::test(runtime = "tokio")]
        async fn drop_aborts_running_tasks() {
            let counter = Arc::new(AtomicUsize::new(0));
            let observed_counts = {
                let mut set: JoinSet<()> = JoinSet::new();
                for _ in 0..3 {
                    let counter = counter.clone();
                    set.spawn(async move {
                        loop {
                            tokio::time::sleep(Duration::from_millis(1)).await;
                            counter.fetch_add(1, Ordering::Release);
                        }
                    });
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
                let count_before_drop = counter.load(Ordering::Acquire);
                drop(set);
                tokio::time::sleep(Duration::from_millis(20)).await;
                let count_after_grace = counter.load(Ordering::Acquire);
                (count_before_drop, count_after_grace)
            };
            let (before, after) = observed_counts;
            // counter incremented before drop; after drop + grace,
            // counter should not be growing further (allow +5 jitter
            // for whatever last sleeps were already-pending)
            assert!(before > 0, "tasks should run before drop");
            assert!(
                after <= before + 5,
                "tasks should stop within grace; before={before} after={after}"
            );
        }
    }
}

#[cfg(all(feature = "std", any(not(feature = "runtime-tokio"), loom)))]
mod portable_join_set {
    //! tokio-free `JoinSet`: one OS thread per spawned task running
    //! `futures::executor::block_on(future)`. Completions land in an mpsc
    //! channel; `Notify` (event-listener backed) wakes `join_next` without
    //! ever blocking the caller's executor thread. Cooperative sources
    //! (the only kind `ProducerLifecycle` spawns) already exit on their own
    //! once their `Signal` fires, so the OS thread naturally winds down —
    //! `abort_all` only stops *this side* from waiting on stragglers.

    use std::future::Future;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::Arc;
    use std::sync::mpsc;

    use crate::sync::notify::Notify;

    /// Mirrors `tokio::task::JoinError`'s query surface — the only part
    /// `ProducerLifecycle` reads.
    #[derive(Debug)]
    pub struct JoinError {
        panicked: bool,
        cancelled: bool,
    }

    impl JoinError {
        #[must_use]
        pub fn is_panic(&self) -> bool {
            self.panicked
        }

        #[must_use]
        pub fn is_cancelled(&self) -> bool {
            self.cancelled
        }
    }

    impl core::fmt::Display for JoinError {
        fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            if self.panicked {
                write!(formatter, "task panicked")
            } else if self.cancelled {
                write!(formatter, "task aborted")
            } else {
                write!(formatter, "task join error")
            }
        }
    }

    impl std::error::Error for JoinError {}

    /// No-op marker kept for call-site parity with `tokio::task::AbortHandle`
    /// — an OS thread cannot be forcibly stopped from the outside.
    #[derive(Debug, Clone, Copy)]
    pub struct AbortHandle;

    /// Tokio-free task set. See module docs for the execution model.
    pub struct JoinSet<T> {
        sender: mpsc::Sender<Result<T, JoinError>>,
        receiver: mpsc::Receiver<Result<T, JoinError>>,
        notify: Arc<Notify>,
        outstanding: usize,
    }

    impl<T: Send + 'static> JoinSet<T> {
        /// New empty set.
        #[must_use]
        pub fn new() -> Self {
            let (sender, receiver) = mpsc::channel();
            Self {
                sender,
                receiver,
                notify: Arc::new(Notify::new()),
                outstanding: 0,
            }
        }

        /// Spawn `future` onto its own OS thread. Requires `T: Send +
        /// 'static` and `future: Send + 'static` — same bound as tokio's
        /// `JoinSet::spawn`.
        pub fn spawn<F>(&mut self, future: F) -> AbortHandle
        where
            F: Future<Output = T> + Send + 'static,
        {
            self.outstanding += 1;
            let sender = self.sender.clone();
            let notify = self.notify.clone();
            std::thread::spawn(move || {
                let outcome = catch_unwind(AssertUnwindSafe(|| futures::executor::block_on(future)));
                let result = match outcome {
                    Ok(value) => Ok(value),
                    Err(_) => Err(JoinError {
                        panicked: true,
                        cancelled: false,
                    }),
                };
                // Send before notify: the channel's happens-before edge is
                // what guarantees a woken `join_next` observes this value.
                let _ = sender.send(result);
                notify.notify_one();
            });
            AbortHandle
        }

        /// Await the next task to complete. Returns `None` once every
        /// spawned task has been reaped.
        pub async fn join_next(&mut self) -> Option<Result<T, JoinError>> {
            loop {
                if self.outstanding == 0 {
                    return None;
                }
                match self.receiver.try_recv() {
                    Ok(result) => {
                        self.outstanding -= 1;
                        return Some(result);
                    }
                    Err(mpsc::TryRecvError::Empty) => self.notify.notified().await,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        self.outstanding = 0;
                        return None;
                    }
                }
            }
        }

        /// Stop waiting on outstanding tasks; each still-running OS thread
        /// keeps executing to completion in the background (detached) since
        /// it cannot be forcibly stopped, but subsequent `join_next` calls
        /// report the remainder as cancelled instead of blocking on them.
        pub fn abort_all(&mut self) {
            while self.outstanding > 0 {
                self.outstanding -= 1;
                let _ = self.sender.send(Err(JoinError {
                    panicked: false,
                    cancelled: true,
                }));
            }
            self.notify.notify_waiters();
        }

        /// Number of tasks still in flight.
        #[must_use]
        pub fn len(&self) -> usize {
            self.outstanding
        }

        /// `true` iff `len() == 0`.
        #[must_use]
        pub fn is_empty(&self) -> bool {
            self.outstanding == 0
        }
    }

    impl<T: Send + 'static> Default for JoinSet<T> {
        fn default() -> Self {
            Self::new()
        }
    }
}
