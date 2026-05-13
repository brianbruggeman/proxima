//! Per-core sharded storage: `count` slots, each emitting thread/core routed to
//! one to cut contention on a shared structure (e.g. one MPMC [`ring`] per core).
//!
//! The **primitive** is `slot(core_id)` + `count()` — pure, no_std, no ambient
//! state: the caller says which core it is. `local()` is a **std convenience**
//! (behind `feature = "std"`) that assigns a sticky per-thread slot via TLS; a
//! bare-metal caller instead passes an id from its runtime (prime's per-core id,
//! a hardware CPU id) to `slot`. Two storage tiers: heap-backed [`PerCore`]
//! (`Vec`, `feature = "alloc"`) and inline no-alloc [`StaticPerCore`] (`[T; N]`).
//!
//! [`ring`]: crate::ring

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use core::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "std")]
use std::cell::Cell;

#[cfg(feature = "std")]
std::thread_local! {
    // a per-thread monotonic ticket, NOT a pre-resolved slot index — modded by
    // `count` at access so one ticket maps into any `PerCore` regardless of its
    // slot count (caching `ticket % count` would index a smaller one out of
    // bounds). Sticky for cache locality: a thread keeps landing on the same slot.
    static CORE_TICKET: Cell<Option<usize>> = const { Cell::new(None) };
}

#[cfg(feature = "std")]
static NEXT_CORE: AtomicUsize = AtomicUsize::new(0);

// std-only routing: assign this thread a sticky ticket, map it into `count` slots.
#[cfg(feature = "std")]
fn local_index(count: usize) -> usize {
    let ticket = CORE_TICKET.with(|cell| {
        cell.get().unwrap_or_else(|| {
            let assigned = NEXT_CORE.fetch_add(1, Ordering::Relaxed);
            cell.set(Some(assigned));
            assigned
        })
    });
    ticket % count
}

/// Heap-backed per-core storage (`Vec`, runtime slot count). See the module docs
/// for the `slot`/`local` split; [`StaticPerCore`] is the inline no-alloc tier.
#[cfg(feature = "alloc")]
pub struct PerCore<T> {
    slots: Vec<T>,
    count: usize,
}

#[cfg(feature = "alloc")]
impl<T> PerCore<T> {
    pub fn new_with(count: usize, factory: impl FnMut(usize) -> T) -> Self {
        let slots = (0..count).map(factory).collect();
        Self { slots, count }
    }

    pub fn from_vec(slots: Vec<T>) -> Self {
        let count = slots.len();
        Self { slots, count }
    }

    /// The calling thread's slot (std only). Threads spread round-robin; when
    /// emitters outnumber slots, several share one — safe for a multi-producer
    /// structure like [`crate::ring::Ring`], just more contended.
    #[cfg(feature = "std")]
    pub fn local(&self) -> &T {
        &self.slots[local_index(self.count)]
    }

    pub fn slot(&self, index: usize) -> &T {
        &self.slots[index]
    }

    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Inline, no-alloc per-core storage (`[T; N]`) — the bare-metal tier of
/// [`PerCore`]. Same `slot`/`local` routing; the buffer lives inline (no heap).
pub struct StaticPerCore<T, const N: usize> {
    slots: [T; N],
}

impl<T, const N: usize> StaticPerCore<T, N> {
    pub fn new_with(factory: impl FnMut(usize) -> T) -> Self {
        Self {
            slots: core::array::from_fn(factory),
        }
    }

    /// The calling thread's slot (std only); bare-metal callers use [`slot`].
    ///
    /// [`slot`]: StaticPerCore::slot
    #[cfg(feature = "std")]
    pub fn local(&self) -> &T {
        &self.slots[local_index(N)]
    }

    pub fn slot(&self, index: usize) -> &T {
        &self.slots[index]
    }

    #[must_use]
    pub const fn count(&self) -> usize {
        N
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn static_per_core_routes_by_index() {
        let cores = StaticPerCore::<usize, 4>::new_with(|index| index * 10);
        assert_eq!(cores.count(), 4);
        assert_eq!(*cores.slot(0), 0);
        assert_eq!(*cores.slot(3), 30);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn per_core_new_with_and_from_vec() {
        let built = PerCore::new_with(3, |index| index + 1);
        assert_eq!(built.count(), 3);
        assert_eq!(*built.slot(2), 3);

        let from = PerCore::from_vec(alloc::vec![10usize, 20]);
        assert_eq!(from.count(), 2);
        assert_eq!(*from.slot(1), 20);
    }

    #[cfg(feature = "std")]
    #[test]
    fn local_is_sticky_within_a_thread() {
        let cores = StaticPerCore::<usize, 8>::new_with(|index| index);
        // a thread keeps landing on the same slot (cache locality) — the ticket
        // is assigned once and reused for every `local()` on this thread.
        let first = *cores.local();
        assert_eq!(*cores.local(), first);
        assert!(first < cores.count());
    }
}
