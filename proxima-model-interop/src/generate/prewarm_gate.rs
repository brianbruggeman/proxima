//! Who may use the device: a real request, or an anticipatory prefill
//! (`proxima-tensor/specs/prefix-cache-reuse/SPEC.md`).
//!
//! A prewarm holds [`PrewarmGate::try_begin`]'s slot for as long as it runs
//! and looks at [`PrewarmGate::request_waiting`] between chunks; a request
//! raises its pending count first and then passes through the slot, so it
//! waits at most for the chunk in flight. After that the prewarm has put its
//! partial entry in the prompt cache and let go, and the request's ordinary
//! lookup finds it. Nothing spins and no thread is parked inside async code:
//! the request entry points are synchronous, and the slot is the same
//! `proxima_primitives::sync::blocking::Mutex` tier-3 case
//! [`super::LoadedModel`] documents for its `expert_slab` (held across one
//! chunk, never across an `.await`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use proxima_primitives::sync::blocking::{Mutex, MutexGuard};

/// The device slot a prewarm holds while it runs and the pending-request
/// count it yields to.
pub(super) struct PrewarmGate {
    slot: Mutex<()>,
    pending: AtomicUsize,
}

/// A request that is pending until this drops, and how long it waited for a
/// running prewarm to yield before it could start.
pub(super) struct PendingRequest<'gate> {
    gate: &'gate PrewarmGate,
    waited: Duration,
}

impl PendingRequest<'_> {
    pub(super) const fn waited(&self) -> Duration {
        self.waited
    }
}

impl Drop for PendingRequest<'_> {
    fn drop(&mut self) {
        self.gate.pending.fetch_sub(1, Ordering::SeqCst);
    }
}

impl PrewarmGate {
    pub(super) const fn new() -> Self {
        Self {
            slot: Mutex::new(()),
            pending: AtomicUsize::new(0),
        }
    }

    /// Registers a request and waits for any running prewarm to yield. The
    /// count rises before the wait, so a prewarm that checks between chunks
    /// sees it and lets go.
    pub(super) fn enter_request(&self) -> PendingRequest<'_> {
        self.pending.fetch_add(1, Ordering::SeqCst);
        let started = Instant::now();
        drop(self.slot.lock());
        PendingRequest {
            gate: self,
            waited: started.elapsed(),
        }
    }

    /// The slot for a prewarm to run under, `None` while another prewarm
    /// holds it.
    pub(super) fn try_begin(&self) -> Option<MutexGuard<'_, ()>> {
        self.slot.try_lock()
    }

    /// Whether any request is pending, which a running prewarm yields to.
    pub(super) fn request_waiting(&self) -> bool {
        self.pending.load(Ordering::SeqCst) > 0
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::*;

    #[test]
    fn a_second_prewarm_cannot_begin_while_one_holds_the_slot() {
        let gate = PrewarmGate::new();

        let first = gate.try_begin();
        let second = gate.try_begin();
        drop(first);
        let third = gate.try_begin();

        assert!(second.is_none(), "one prewarm at a time");
        assert!(third.is_some(), "the slot comes back once released");
    }

    #[test]
    fn a_pending_request_is_visible_until_it_drops() {
        let gate = PrewarmGate::new();
        assert!(!gate.request_waiting());

        let request = gate.enter_request();
        assert!(gate.request_waiting());
        drop(request);

        assert!(!gate.request_waiting());
    }

    #[test]
    fn a_request_waits_for_the_prewarm_to_yield_and_the_prewarm_sees_it() {
        let gate = PrewarmGate::new();
        let holding = AtomicBool::new(false);

        std::thread::scope(|scope| {
            let prewarm = scope.spawn(|| {
                let slot = gate.try_begin().expect("the idle gate yields its slot");
                holding.store(true, Ordering::SeqCst);
                while !gate.request_waiting() {
                    std::hint::spin_loop();
                }
                drop(slot);
            });
            while !holding.load(Ordering::SeqCst) {
                std::hint::spin_loop();
            }
            let request = gate.enter_request();
            prewarm.join().expect("the prewarm thread yields");

            assert!(gate.request_waiting());
            drop(request);
        });

        assert!(!gate.request_waiting());
    }

    #[test]
    fn a_prewarm_that_panicked_holding_the_slot_does_not_block_requests() {
        let gate = PrewarmGate::new();
        let outcome = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let _slot = gate.try_begin();
                    panic!("prewarm died mid-chunk");
                })
                .join()
        });

        let request = gate.enter_request();

        assert!(outcome.is_err());
        assert!(gate.request_waiting());
        assert!(gate.try_begin().is_some());
        drop(request);
    }
}
