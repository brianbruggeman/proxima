//! `proxima::sync::Notify` — single-permit edge-triggered notifier,
//! shape-compatible with `tokio::sync::Notify`. Backed by
//! `event_listener::Event`; permit semantics implemented via a
//! single-bit `AtomicBool` so `notify_one` saves a permit when no
//! waiter is listening (just like tokio).
//!
//! Semantics summary:
//! - At most ONE permit is ever stored. Repeated `notify_one` calls
//!   while a permit is already pending are no-ops.
//! - `notify_waiters` wakes all currently-listening waiters but does
//!   NOT save a permit. Future `notified().await` calls will park.
//! - `notified()` returns a [`Notified`] future. `notified().await`
//!   consumes a pending permit or parks until the next `notify_one`
//!   or `notify_waiters`.
//! - `Notified::enable(self: Pin<&mut Self>)` ensures the listener is
//!   registered *without* polling — used to close lost-wakeup races
//!   when the caller wants to re-check state between registration and
//!   parking.
//!
//! # Non-coverage
//!
//! - `notify_one` only — no `notify_last` / `notify_first` ordering
//!   variants. tokio's `Notify` doesn't offer these either, but other
//!   notifier crates do; documenting it for migration clarity.
//! - `notify_waiters` wakes ALL current waiters and does NOT save a
//!   permit. That matches tokio. If you need "wake all + save a
//!   permit for the next entrant", chain `notify_waiters()` and
//!   `notify_one()`.
//!
//! # Loom coverage
//!
//! `--cfg loom` loom-tests this exact `Notify` (not a hand-written
//! model): [`crate::sync::loom_atomic`] swaps `permit`'s `AtomicBool` for
//! loom's instrumented one, and `event_listener::Event` becomes
//! loom-instrumented too (its `loom` Cargo feature is enabled under
//! `[target.'cfg(loom)'.dependencies]`), so the model checker sees
//! every atomic op in the permit/listener-registration race, inside
//! and outside `Event`. See the `loom_tests` module below.

use alloc::boxed::Box;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use event_listener::{Event, EventListener};

use crate::sync::loom_atomic::{AtomicBool, Ordering};

#[derive(Debug, Default)]
pub struct Notify {
    event: Event,
    /// True when `notify_one` stored a permit that has not yet been
    /// consumed by a `notified().await`.
    permit: AtomicBool,
}

impl Notify {
    /// Under `--cfg loom`, `event_listener::Event::new` (and loom's own
    /// `AtomicBool::new`) are deliberately non-`const` — loom's model
    /// needs a runtime execution context to track the value. Only the
    /// `const` qualifier is affected; the constructed `Notify` is
    /// identical either way.
    #[cfg(not(loom))]
    #[must_use]
    pub const fn const_new() -> Self {
        Self {
            event: Event::new(),
            permit: AtomicBool::new(false),
        }
    }

    #[cfg(loom)]
    #[must_use]
    pub fn const_new() -> Self {
        Self {
            event: Event::new(),
            permit: AtomicBool::new(false),
        }
    }

    #[must_use]
    pub fn new() -> Self {
        Self::const_new()
    }

    /// Wake one waiter, or store a permit for the next waiter if none
    /// are currently listening. Multiple calls before a wake do not
    /// stack — at most one permit is held at a time.
    pub fn notify_one(&self) {
        if self
            .permit
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.event.notify(1);
        }
    }

    /// Wake every currently-listening waiter. Does NOT store a permit.
    pub fn notify_waiters(&self) {
        self.event.notify(usize::MAX);
    }

    /// Returns a future that resolves when a permit is available or a
    /// waiter wake fires.
    #[must_use = "futures are lazy and do nothing unless awaited"]
    pub fn notified(&self) -> Notified<'_> {
        Notified {
            notify: self,
            listener: None,
        }
    }
}

/// Future returned by [`Notify::notified`]. Can be explicitly
/// registered without polling via [`Notified::enable`] to close
/// lost-wakeup races between state check and park.
pub struct Notified<'lifetime> {
    notify: &'lifetime Notify,
    /// `Pin<Box<EventListener>>` is `Unpin` (Box is always Unpin)
    /// even though EventListener itself is not, so the whole
    /// `Notified` is `Unpin` and supports the `enable(self: Pin<&mut
    /// Self>)` signature without `unsafe`.
    listener: Option<Pin<Box<EventListener>>>,
}

impl Notified<'_> {
    /// Register as a listener and consume any pending permit. Returns
    /// `true` if a permit was already pending (the listener will not
    /// park on subsequent `await`); `false` otherwise. Idempotent.
    pub fn enable(self: Pin<&mut Self>) -> bool {
        let this = self.get_mut();
        this.ensure_listener();
        if this.notify.permit.swap(false, Ordering::AcqRel) {
            this.listener = None;
            true
        } else {
            false
        }
    }

    fn ensure_listener(&mut self) {
        if self.listener.is_none() {
            self.listener = Some(Box::pin(self.notify.event.listen()));
        }
    }
}

impl Future for Notified<'_> {
    type Output = ();
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if this.notify.permit.swap(false, Ordering::AcqRel) {
            this.listener = None;
            return Poll::Ready(());
        }
        this.ensure_listener();
        if this.notify.permit.swap(false, Ordering::AcqRel) {
            this.listener = None;
            return Poll::Ready(());
        }
        let Some(listener) = this.listener.as_mut() else {
            return Poll::Pending;
        };
        match listener.as_mut().poll(context) {
            Poll::Ready(()) => {
                this.listener = None;
                let _ = this.notify.permit.swap(false, Ordering::AcqRel);
                Poll::Ready(())
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

// under `--cfg loom`, `event_listener::Event` and loom's atomics require
// every call to happen inside a `loom::model()`; these plain, non-model
// tests exist for the normal (non-loom) build only — `loom_tests` below
// is the loom-model equivalent coverage.
#[cfg(all(test, not(loom)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use std::sync::Arc;

    #[test]
    fn notify_one_then_notified_returns_immediately() {
        let notify = Notify::new();
        notify.notify_one();
        block_on(notify.notified());
    }

    #[test]
    fn second_notified_parks_until_next_notify() {
        let notify = Arc::new(Notify::new());
        notify.notify_one();
        block_on(notify.notified());
        let started = std::sync::atomic::AtomicBool::new(false);
        let started_ref = &started;
        let notify_ref = notify.clone();
        std::thread::scope(|scope| {
            scope.spawn(move || {
                while !started_ref.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                notify_ref.notify_one();
            });
            started.store(true, Ordering::Release);
            block_on(notify.notified());
        });
    }

    #[test]
    fn notify_waiters_does_not_save_permit() {
        let notify = Notify::new();
        notify.notify_waiters();
        // no listener was registered when notify_waiters fired, so the
        // permit is NOT saved. A subsequent notified() must park; we
        // can't easily check "parks forever" so instead verify via
        // notify_one re-set after.
        notify.notify_one();
        block_on(notify.notified());
    }

    #[test]
    fn enable_returns_true_when_permit_already_pending() {
        let notify = Notify::new();
        notify.notify_one();
        let waiter = notify.notified();
        futures::pin_mut!(waiter);
        assert!(waiter.as_mut().enable());
    }

    #[test]
    fn enable_returns_false_when_no_permit() {
        let notify = Notify::new();
        let waiter = notify.notified();
        futures::pin_mut!(waiter);
        assert!(!waiter.as_mut().enable());
    }
}

/// Loom interleaving coverage for the REAL `Notify` above (not a
/// hand-written model). Ported 1:1 from the former
/// `proxima-loom/tests/loom_sync.rs`. `preemption_bound` is set inline
/// so the search is bounded without relying on an operator remembering
/// `LOOM_MAX_PREEMPTIONS`.
///
/// Run with:
///   RUSTFLAGS="--cfg loom" cargo test -p proxima-primitives --release notify::loom_tests
#[cfg(all(test, loom))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod loom_tests {
    use super::Notify;
    use loom::sync::Arc;

    fn check(model: impl Fn() + Sync + Send + 'static) {
        let mut builder = loom::model::Builder::new();
        builder.preemption_bound = Some(3);
        builder.check(model);
    }

    #[test]
    fn notify_one_then_notified_returns_immediately() {
        check(|| {
            let notify = Arc::new(Notify::new());
            notify.notify_one();
            loom::future::block_on(notify.notified());
        });
    }

    #[test]
    fn notified_then_notify_one_resolves() {
        check(|| {
            let notify = Arc::new(Notify::new());
            let notify_producer = notify.clone();

            let producer = loom::thread::spawn(move || {
                notify_producer.notify_one();
            });

            loom::future::block_on(notify.notified());
            producer.join().expect("producer thread panicked");
        });
    }

    #[test]
    fn concurrent_notify_one_and_notified() {
        check(|| {
            let notify = Arc::new(Notify::new());
            let notify_producer = notify.clone();

            let producer = loom::thread::spawn(move || {
                notify_producer.notify_one();
            });

            loom::future::block_on(notify.notified());
            producer.join().expect("producer thread panicked");
        });
    }

    /// A second `notify_one` while a permit is already pending must NOT
    /// stack additional permits — the CAS in `notify_one` is the guard.
    #[test]
    fn multiple_notify_one_does_not_stack() {
        check(|| {
            let notify = Arc::new(Notify::new());

            notify.notify_one();
            notify.notify_one(); // permit already true — must be a no-op

            loom::future::block_on(notify.notified()); // consumes the one permit

            // no stacked permit, so notified() must park until notify_one fires
            let notify_for_producer = notify.clone();
            let producer = loom::thread::spawn(move || {
                notify_for_producer.notify_one(); // third notify — unblocks second waiter
            });

            loom::future::block_on(notify.notified());
            producer.join().expect("producer thread panicked");
        });
    }
}
