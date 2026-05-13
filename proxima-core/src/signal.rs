//! Sticky one-shot level signal — the cancellation / shutdown scope
//! primitive.
//!
//! A [`Signal`] is a *level*, not an event: it transitions once
//! (unfired → fired) and stays fired, so a late observer polls an
//! already-fired signal and resolves immediately — the
//! notify-before-wait lost-wakeup class cannot happen by construction.
//!
//! Scopes nest by merge, not by tree: [`Signal::child`] observes every
//! ancestor level plus its own, so firing a parent cancels the whole
//! subtree while firing a child leaves the parent untouched. A handle
//! is a flat `Vec<Arc<Level>>` (depth = real scope nesting: process →
//! listener → connection → request), `is_fired` is one relaxed pass of
//! atomic loads, and there is no node graph to refcount.

// Only the racy primitives (the fired flag + the waiters mutex) swap to
// loom's instrumented equivalents under `--features loom`, matching
// `ring/mpsc.rs`'s own convention (see that module's doc comment): loom
// explores interleavings of the types it instruments, so `Arc` — plain
// refcounting, not part of the fire/register/drop race — stays `std::sync::Arc`
// unconditionally. Rewiring it to `loom::sync::Arc` would also break the
// `Arc<[Arc<Level>]>` unsized-slice construction (`Arc::from([..])`), which
// loom's mock `Arc` does not support (no `CoerceUnsized`; see
// `loom::sync::Arc::from_std`'s doc comment) — a real API gap, not a choice.
#[cfg(feature = "loom")]
use loom::sync::Mutex;
#[cfg(feature = "loom")]
use loom::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(feature = "loom"))]
use std::sync::Mutex;
#[cfg(not(feature = "loom"))]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use core::future::Future;
use core::pin::Pin;

use futures::task::AtomicWaker;
use smallvec::SmallVec;

/// One awaiting observer's waker cell. Each [`Fired`] future owns one slot
/// and registers it with every unfired level; re-polls update the slot in
/// place, so a select loop re-polling never grows the registries.
///
/// Backed by [`AtomicWaker`] (from `futures`, RISC reuse per principle 1)
/// rather than a hand-rolled `Mutex<Option<Waker>>`: it is lock-free (no
/// blocking mutex on the register/wake path, per principle 21) and its
/// `register`-then-caller-checks-again contract is the exact same
/// lost-wakeup-avoidance shape `Fired::poll` already implements.
struct WakerSlot(AtomicWaker);

impl WakerSlot {
    fn new() -> Self {
        Self(AtomicWaker::new())
    }

    fn register(&self, waker: &Waker) {
        self.0.register(waker);
    }

    fn wake(&self) {
        self.0.wake();
    }
}

/// A level's set of pending waiters. Inline-capacity `SmallVec` (cap sized by
/// `sized::SIGNAL_WAITERS_INLINE_CAP`, principle 12): the common case — a
/// handful of concurrent `Fired` futures awaiting one scope level — never
/// touches the allocator, in place of the previous unconditionally-heap Vec.
/// A burst past the inline cap spills to a heap-backed tail, same as Vec
/// always did — no waiter is ever refused registration.
type WaiterRegistry = SmallVec<[Arc<WakerSlot>; crate::sized::SIGNAL_WAITERS_INLINE_CAP]>;

struct Level {
    fired: AtomicBool,
    waiters: Mutex<WaiterRegistry>,
}

impl Level {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            fired: AtomicBool::new(false),
            waiters: Mutex::new(SmallVec::new()),
        })
    }

    fn fire(&self) {
        if self.fired.swap(true, Ordering::AcqRel) {
            return;
        }
        let drained = match self.waiters.lock() {
            Ok(mut guard) => core::mem::take(&mut *guard),
            Err(_) => return,
        };
        for slot in drained {
            slot.wake();
        }
    }

    fn is_fired(&self) -> bool {
        self.fired.load(Ordering::Acquire)
    }
}

/// Cloneable handle onto a scope's cancellation level. See the module
/// docs for the level/merge model.
///
/// The level list is a shared slice, not an owned `Vec`: clone (the
/// hot per-request/per-record operation) is one atomic increment and
/// zero allocation; only `new`/`child` (scope creation) allocate.
#[derive(Clone)]
pub struct Signal {
    levels: Arc<[Arc<Level>]>,
}

impl Signal {
    #[must_use]
    pub fn new() -> Self {
        Self {
            levels: Arc::from([Level::new()]),
        }
    }

    /// A nested scope: observes every ancestor level plus a fresh one
    /// of its own. Firing the child never touches the ancestors.
    #[must_use]
    pub fn child(&self) -> Self {
        let mut levels = Vec::with_capacity(self.levels.len() + 1);
        levels.extend(self.levels.iter().cloned());
        levels.push(Level::new());
        Self {
            levels: Arc::from(levels),
        }
    }

    /// Fire this handle's own scope level (idempotent). Ancestor
    /// levels are never fired through a child handle.
    pub fn fire(&self) {
        if let Some(own) = self.levels.last() {
            own.fire();
        }
    }

    /// True once this scope or any ancestor scope has fired.
    #[must_use]
    pub fn is_fired(&self) -> bool {
        self.levels.iter().any(|level| level.is_fired())
    }

    /// Resolves when this scope or any ancestor fires; resolves
    /// immediately if that already happened (sticky level).
    #[must_use]
    pub fn fired(&self) -> Fired {
        Fired {
            levels: self.levels.clone(),
            slot: None,
        }
    }

    /// Fire-on-drop guard for tying a scope's lifetime to a value
    /// (e.g. cancel a request's subtree when its driver unwinds).
    #[must_use]
    pub fn guard(self) -> SignalGuard {
        SignalGuard {
            signal: self,
            armed: true,
        }
    }
}

impl Default for Signal {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for Signal {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Signal")
            .field("depth", &self.levels.len())
            .field("fired", &self.is_fired())
            .finish()
    }
}

/// Future returned by [`Signal::fired`]. Owns its level handles, so it
/// is `'static` and can outlive the handle it came from.
pub struct Fired {
    levels: Arc<[Arc<Level>]>,
    slot: Option<Arc<WakerSlot>>,
}

impl Future for Fired {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        // register-then-check closes the lost-wakeup race: fire() sets
        // the flag before draining waiters, so a fire that slipped past
        // our registration is visible in the check below.
        if this.slot.is_none() {
            let slot = Arc::new(WakerSlot::new());
            for level in this.levels.iter() {
                if let Ok(mut waiters) = level.waiters.lock() {
                    waiters.push(slot.clone());
                }
            }
            this.slot = Some(slot);
        }
        if let Some(slot) = &this.slot {
            slot.register(cx.waker());
        }
        if this.levels.iter().any(|level| level.is_fired()) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

impl Drop for Fired {
    fn drop(&mut self) {
        let Some(slot) = self.slot.take() else {
            return;
        };
        for level in self.levels.iter() {
            if let Ok(mut waiters) = level.waiters.lock() {
                waiters.retain(|candidate| !Arc::ptr_eq(candidate, &slot));
            }
        }
    }
}

/// Fires the wrapped signal's own level on drop, unless disarmed.
pub struct SignalGuard {
    signal: Signal,
    armed: bool,
}

impl SignalGuard {
    #[must_use]
    pub fn signal(&self) -> &Signal {
        &self.signal
    }

    /// Consume the guard without firing — the success path, where the
    /// scope outlives the guard. Returns the signal untouched.
    pub fn disarm(mut self) -> Signal {
        self.armed = false;
        self.signal.clone()
    }
}

impl Drop for SignalGuard {
    fn drop(&mut self) {
        if self.armed {
            self.signal.fire();
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::task::Wake;

    use super::*;

    struct CountingWaker(AtomicUsize);

    impl Wake for CountingWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn counting_waker() -> (Arc<CountingWaker>, Waker) {
        let counter = Arc::new(CountingWaker(AtomicUsize::new(0)));
        let waker = Waker::from(counter.clone());
        (counter, waker)
    }

    #[test]
    fn fire_is_sticky_and_idempotent() {
        let signal = Signal::new();
        assert!(!signal.is_fired());
        signal.fire();
        signal.fire();
        assert!(signal.is_fired());
    }

    #[test]
    fn late_observer_resolves_on_first_poll() {
        let signal = Signal::new();
        signal.fire();
        let mut fired = signal.fired();
        let (_, waker) = counting_waker();
        let mut cx = Context::from_waker(&waker);
        assert_eq!(Pin::new(&mut fired).poll(&mut cx), Poll::Ready(()));
    }

    #[test]
    fn registered_waiter_is_woken_by_fire() {
        let signal = Signal::new();
        let mut fired = signal.fired();
        let (counter, waker) = counting_waker();
        let mut cx = Context::from_waker(&waker);
        assert_eq!(Pin::new(&mut fired).poll(&mut cx), Poll::Pending);
        signal.fire();
        assert_eq!(counter.0.load(Ordering::Relaxed), 1);
        assert_eq!(Pin::new(&mut fired).poll(&mut cx), Poll::Ready(()));
    }

    #[test]
    fn child_observes_parent_fire() {
        let parent = Signal::new();
        let child = parent.child();
        parent.fire();
        assert!(child.is_fired());
    }

    #[test]
    fn parent_is_untouched_by_child_fire() {
        let parent = Signal::new();
        let child = parent.child();
        child.fire();
        assert!(child.is_fired());
        assert!(!parent.is_fired());
    }

    #[test]
    fn grandchild_observes_root_through_the_merge() {
        let root = Signal::new();
        let grandchild = root.child().child();
        root.fire();
        assert!(grandchild.is_fired());
    }

    #[test]
    fn guard_fires_own_level_on_drop() {
        let signal = Signal::new();
        let observer = signal.clone();
        {
            let _guard = signal.guard();
            assert!(!observer.is_fired());
        }
        assert!(observer.is_fired());
    }

    #[test]
    fn disarmed_guard_does_not_fire_on_drop() {
        let signal = Signal::new();
        let observer = signal.clone();
        let guard = signal.guard();
        let returned = guard.disarm();
        assert!(!observer.is_fired());
        assert!(!returned.is_fired());
    }

    #[test]
    fn dropping_a_pending_waiter_unregisters_its_slot() {
        let signal = Signal::new();
        let (_, waker) = counting_waker();
        let mut cx = Context::from_waker(&waker);
        {
            let mut fired = signal.fired();
            assert_eq!(Pin::new(&mut fired).poll(&mut cx), Poll::Pending);
            assert_eq!(signal.levels[0].waiters.lock().unwrap().len(), 1);
        }
        assert_eq!(signal.levels[0].waiters.lock().unwrap().len(), 0);
    }

    #[test]
    fn repolling_does_not_grow_the_waiter_list() {
        let signal = Signal::new();
        let mut fired = signal.fired();
        let (_, waker) = counting_waker();
        let mut cx = Context::from_waker(&waker);
        for _ in 0..64 {
            assert_eq!(Pin::new(&mut fired).poll(&mut cx), Poll::Pending);
        }
        assert_eq!(signal.levels[0].waiters.lock().unwrap().len(), 1);
    }
}
