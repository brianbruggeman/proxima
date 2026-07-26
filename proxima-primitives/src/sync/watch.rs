//! `proxima::sync::watch` — single-producer, multi-consumer
//! latest-value channel. Shape-compatible with `tokio::sync::watch`:
//! each receiver tracks whether it has observed the current value;
//! slow receivers don't block fast ones; back-to-back sends collapse
//! to the last value if the consumer hasn't called `changed()` in
//! between.
//!
//! Hand-rolled over `event_listener::Event` + `std::sync::RwLock<T>` +
//! per-receiver version counter. Backing this with `async-broadcast`
//! cap=1 didn't fit — broadcast delivers every value to every
//! receiver, watch collapses to the latest, and the semantics of
//! "this receiver has seen this version" are watch-specific.
//!
//! # Non-coverage
//!
//! - `Receiver::wait_for(closure) -> Result<Ref<T>, RecvError>` —
//!   predicate-await helper that loops `changed().await` + read until
//!   `closure(&value)` returns true. Compose at the call site if
//!   needed; not exposed here because no internal caller uses it.
//! - `Receiver::mark_changed()` / `Receiver::mark_unchanged()` —
//!   manual version-cursor manipulation. The current callers only
//!   need `borrow` / `borrow_and_update` / `changed`.
//! - `Sender::send_replace(value) -> T` — set new value AND return
//!   the prior value atomically. Compose via `borrow()` clone +
//!   `send` at the call site.
//! - `Sender::send_modify(closure)` — read-modify-write under the
//!   write lock. Not exposed because the value type is typically a
//!   small Copy state machine; if a caller needs in-place mutation
//!   they can use a `Mutex<T>` next to the watch and only send a
//!   version-cursor through the watch.
//! - `Sender::send_if_modified(closure) -> bool` — same with a
//!   skip-if-unchanged shortcut.
//! - `Receiver::same_channel(&other)` — identity check across clones.
//!
//! # Loom coverage
//!
//! `--cfg loom` loom-tests this exact `Sender`/`Receiver` pair (not a
//! hand-written model): [`crate::sync::loom_atomic`] swaps `version` /
//! `senders` / `receivers` and the `Arc`/`RwLock` for loom's
//! instrumented equivalents, and `event_listener::Event` becomes
//! loom-instrumented too (its `loom` Cargo feature is enabled under
//! `[target.'cfg(loom)'.dependencies]`). See the `loom_tests` module
//! below.

use event_listener::Event;

use crate::sync::loom_atomic::{Arc, AtomicU64, AtomicUsize, Ordering, RwLock, RwLockReadGuard};

/// Returned by [`Sender::send`] when all receivers have been dropped.
#[derive(Debug, thiserror::Error)]
#[error("watch channel has no receivers")]
pub struct SendError<T>(pub T);

/// Returned by [`Receiver::changed`] / [`Receiver::has_changed`] when
/// the sender has been dropped.
#[derive(Debug, thiserror::Error)]
#[error("watch channel sender dropped")]
pub struct RecvError;

struct Inner<T> {
    value: RwLock<T>,
    /// Bumped on every successful `send`. Receivers compare against
    /// their last-seen version to detect a new value.
    version: AtomicU64,
    /// Wakes all receivers parked in `changed()`.
    event: Event,
    /// Live sender count. Goes to 0 when the last `Sender` drops,
    /// which signals receivers to see `RecvError` from `changed()`.
    senders: AtomicUsize,
    /// Live receiver count. Used by `Sender::is_closed`.
    receivers: AtomicUsize,
}

/// Producer side. Only one sender is supported per channel today; the
/// API matches tokio's single-producer semantics.
pub struct Sender<T> {
    inner: Arc<Inner<T>>,
}

impl<T> Sender<T> {
    /// Replace the current value, bumping the version and waking
    /// every receiver parked in `changed()`. Returns the value back
    /// via [`SendError`] if every receiver has been dropped.
    pub fn send(&self, value: T) -> Result<(), SendError<T>> {
        if self.inner.receivers.load(Ordering::Acquire) == 0 {
            return Err(SendError(value));
        }
        {
            let mut guard = self
                .inner
                .value
                .write()
                .unwrap_or_else(|err| err.into_inner());
            *guard = value;
        }
        self.inner.version.fetch_add(1, Ordering::AcqRel);
        self.inner.event.notify(usize::MAX);
        Ok(())
    }

    /// Returns true once every receiver has been dropped.
    pub fn is_closed(&self) -> bool {
        self.inner.receivers.load(Ordering::Acquire) == 0
    }

    /// Build a fresh receiver subscribed at the current version. The
    /// new receiver's first `changed()` call resolves only after a
    /// *future* `send`.
    pub fn subscribe(&self) -> Receiver<T> {
        self.inner.receivers.fetch_add(1, Ordering::AcqRel);
        Receiver {
            inner: self.inner.clone(),
            last_seen: self.inner.version.load(Ordering::Acquire),
        }
    }

    /// Read-only borrow of the current value. Does NOT update any
    /// per-receiver seen-state because the sender has no notion of
    /// per-receiver state.
    pub fn borrow(&self) -> Ref<'_, T> {
        Ref {
            guard: self
                .inner
                .value
                .read()
                .unwrap_or_else(|err| err.into_inner()),
        }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        if self.inner.senders.fetch_sub(1, Ordering::AcqRel) == 1 {
            // last sender — wake every receiver so changed() can
            // observe RecvError.
            self.inner.event.notify(usize::MAX);
        }
    }
}

/// Consumer side. Each receiver tracks its own seen-version, so two
/// receivers can be at different positions in the value stream and
/// neither blocks the other.
pub struct Receiver<T> {
    inner: Arc<Inner<T>>,
    last_seen: u64,
}

impl<T> Receiver<T> {
    /// Park until the current value differs from this receiver's
    /// last-seen version. Returns `Err(RecvError)` if the sender has
    /// been dropped and no further value can ever arrive. Does NOT
    /// update the seen-version — call `borrow_and_update` after to
    /// acknowledge.
    pub async fn changed(&mut self) -> Result<(), RecvError> {
        loop {
            let current = self.inner.version.load(Ordering::Acquire);
            if current != self.last_seen {
                return Ok(());
            }
            if self.inner.senders.load(Ordering::Acquire) == 0 {
                return Err(RecvError);
            }
            let listener = self.inner.event.listen();
            // re-check after registering to close the lost-wakeup window.
            let current = self.inner.version.load(Ordering::Acquire);
            if current != self.last_seen {
                return Ok(());
            }
            if self.inner.senders.load(Ordering::Acquire) == 0 {
                return Err(RecvError);
            }
            listener.await;
        }
    }

    /// Non-blocking check: has a new value arrived since the last
    /// `borrow_and_update`?
    pub fn has_changed(&self) -> Result<bool, RecvError> {
        let current = self.inner.version.load(Ordering::Acquire);
        if current != self.last_seen {
            return Ok(true);
        }
        if self.inner.senders.load(Ordering::Acquire) == 0 {
            return Err(RecvError);
        }
        Ok(false)
    }

    /// Read-only borrow of the current value without updating
    /// seen-state. A follow-up `changed()` call will still return
    /// `Ok(())` if the version changed before this `borrow`.
    pub fn borrow(&self) -> Ref<'_, T> {
        Ref {
            guard: self
                .inner
                .value
                .read()
                .unwrap_or_else(|err| err.into_inner()),
        }
    }

    /// Read the current value and mark it as seen so the next
    /// `changed()` call only resolves on a *subsequent* send.
    pub fn borrow_and_update(&mut self) -> Ref<'_, T> {
        self.last_seen = self.inner.version.load(Ordering::Acquire);
        self.borrow()
    }
}

impl<T> Clone for Receiver<T> {
    fn clone(&self) -> Self {
        self.inner.receivers.fetch_add(1, Ordering::AcqRel);
        Self {
            inner: self.inner.clone(),
            last_seen: self.last_seen,
        }
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        self.inner.receivers.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Read guard returned by `borrow` / `borrow_and_update`. Holds the
/// shared lock for the lifetime of the borrow.
pub struct Ref<'lifetime, T> {
    guard: RwLockReadGuard<'lifetime, T>,
}

impl<T> std::ops::Deref for Ref<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

/// Build a watch channel anchored at `initial`.
pub fn channel<T>(initial: T) -> (Sender<T>, Receiver<T>) {
    let inner = Arc::new(Inner {
        value: RwLock::new(initial),
        version: AtomicU64::new(0),
        event: Event::new(),
        senders: AtomicUsize::new(1),
        receivers: AtomicUsize::new(1),
    });
    (
        Sender {
            inner: inner.clone(),
        },
        Receiver {
            inner,
            last_seen: 0,
        },
    )
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

    #[test]
    fn initial_value_is_visible_via_borrow() {
        let (_sender, receiver) = channel(7_u32);
        assert_eq!(*receiver.borrow(), 7);
    }

    #[test]
    fn send_makes_changed_resolve_and_updates_value() {
        let (sender, mut receiver) = channel(0_u32);
        sender.send(42).expect("send");
        block_on(async {
            receiver.changed().await.expect("changed");
            assert_eq!(*receiver.borrow_and_update(), 42);
        });
    }

    #[test]
    fn back_to_back_sends_collapse_to_latest() {
        let (sender, mut receiver) = channel(0_u32);
        sender.send(1).expect("send 1");
        sender.send(2).expect("send 2");
        sender.send(3).expect("send 3");
        block_on(async {
            receiver.changed().await.expect("changed");
            assert_eq!(*receiver.borrow_and_update(), 3);
            // no further changes pending
            assert!(!receiver.has_changed().expect("has_changed"));
        });
    }

    #[test]
    fn dropping_sender_makes_changed_error() {
        let (sender, mut receiver) = channel(0_u32);
        drop(sender);
        block_on(async {
            let result = receiver.changed().await;
            assert!(result.is_err());
        });
    }

    #[test]
    fn dropping_all_receivers_makes_send_error() {
        let (sender, receiver) = channel(0_u32);
        drop(receiver);
        match sender.send(1) {
            Err(SendError(value)) => assert_eq!(value, 1),
            Ok(()) => panic!("send to receiver-less channel should error"),
        }
    }

    #[test]
    fn subscribe_starts_at_current_version_and_resolves_on_next_send() {
        let (sender, _receiver) = channel(0_u32);
        sender.send(1).expect("send 1");
        let mut newcomer = sender.subscribe();
        // newcomer has not observed any "change" yet (joined at v1)
        assert!(!newcomer.has_changed().expect("has_changed"));
        sender.send(2).expect("send 2");
        block_on(async {
            newcomer.changed().await.expect("changed");
            assert_eq!(*newcomer.borrow_and_update(), 2);
        });
    }
}

/// Loom interleaving coverage for the REAL `channel`/`Sender`/`Receiver`
/// above (not a hand-written model). Ported 1:1 from the former
/// `proxima-loom/tests/loom_sync.rs`, rewritten to drive `changed()`
/// (the real future) instead of poking the version counter directly —
/// that's the actual send/version-bump/listener-wake race this module
/// promises. `preemption_bound` is set inline so the search is bounded
/// without relying on an operator remembering `LOOM_MAX_PREEMPTIONS`.
///
/// Run with:
///   RUSTFLAGS="--cfg loom" cargo test -p proxima-primitives --release watch::loom_tests
#[cfg(all(test, loom))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod loom_tests {
    use super::channel;
    use loom::sync::Arc;

    fn check(model: impl Fn() + Sync + Send + 'static) {
        let mut builder = loom::model::Builder::new();
        builder.preemption_bound = Some(3);
        builder.check(model);
    }

    /// Producer sends before consumer checks. `changed()` must resolve
    /// without parking and observe the new value.
    #[test]
    fn send_then_changed_observes_new_value() {
        check(|| {
            let (sender, mut receiver) = channel(0_u32);
            sender.send(42).expect("send");
            loom::future::block_on(async {
                receiver.changed().await.expect("changed");
            });
            assert_eq!(*receiver.borrow_and_update(), 42);
        });
    }

    /// Consumer registers first (parks in `changed()`), then the
    /// producer thread sends. The listener must be woken and
    /// `changed()` must resolve with the new value.
    #[test]
    fn changed_then_send_resolves() {
        check(|| {
            let (sender, mut receiver) = channel(0_u32);
            let sender = Arc::new(sender);
            let sender_producer = sender.clone();

            let producer = loom::thread::spawn(move || {
                sender_producer.send(1).expect("send");
            });

            loom::future::block_on(async {
                receiver.changed().await.expect("changed");
            });
            producer.join().expect("producer thread panicked");

            assert_eq!(*receiver.borrow_and_update(), 1);
        });
    }

    /// Producer and consumer race on all interleavings up to the
    /// preemption bound. Under every schedule, the consumer's
    /// `changed()` must resolve and observe the sent value.
    #[test]
    fn concurrent_send_and_changed() {
        check(|| {
            let (sender, mut receiver) = channel(0_u32);
            let sender = Arc::new(sender);
            let sender_producer = sender.clone();

            let producer = loom::thread::spawn(move || {
                sender_producer.send(99).expect("send");
            });

            let consumer = loom::thread::spawn(move || {
                loom::future::block_on(async {
                    receiver.changed().await.expect("changed");
                });
                *receiver.borrow_and_update()
            });

            producer.join().expect("producer thread panicked");
            let observed = consumer.join().expect("consumer thread panicked");
            assert_eq!(observed, 99);
        });
    }
}
