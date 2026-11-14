use core::sync::atomic::Ordering;

use portable_atomic::AtomicU64;

use crate::tag::Tag;

/// A monotonically increasing integer counter instrument.
///
/// All state lives in the struct — no heap allocation on `add`.
/// The single `AtomicU64` accumulator is the v1 baseline; per-attr-set
/// bucket sharding is the opt-sweep target for C9+.
pub struct Counter {
    pub name: &'static str,
    pub unit: &'static str,
    pub description: &'static str,
    value: AtomicU64,
}

impl Counter {
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            unit: "",
            description: "",
            value: AtomicU64::new(0),
        }
    }

    #[must_use]
    pub const fn unit(mut self, unit: &'static str) -> Self {
        self.unit = unit;
        self
    }

    #[must_use]
    pub const fn description(mut self, description: &'static str) -> Self {
        self.description = description;
        self
    }

    /// Increment the counter by `delta`.
    ///
    /// `tags` are accepted for API symmetry with the opt-sweep target;
    /// they are not accumulated in v1 (single-accumulator baseline).
    pub fn add(&self, delta: u64, _tags: &[Tag]) {
        self.value.fetch_add(delta, Ordering::Relaxed);
        #[cfg(feature = "scoped-counters")]
        scope::record(self, delta);
    }

    /// Run `work` and return its result plus the delta THIS thread added to
    /// this counter inside it. Adds from other threads are not counted, so a
    /// test can assert an exact delta while parallel tests move the same
    /// process-global counter. Compare `allocations_during` in
    /// `proxima-model-interop`'s `alloc_probe`, the same thread-local shape.
    ///
    /// Scopes nest; an outer scope also sees the inner scope's adds. Work that
    /// `work` offloads to another thread is not attributed to the scope.
    ///
    /// Behind the default-off `scoped-counters` feature (implies `std`: the
    /// tally is a thread-local). Feature off, `add` is exactly one atomic
    /// add. Feature on, a no-scope `add` also pays one thread-local
    /// `Cell<usize>` load; `Counter` does not grow. Enable it from a
    /// dev-dependency only.
    #[cfg(feature = "scoped-counters")]
    pub fn scoped<R>(&self, work: impl FnOnce() -> R) -> (R, u64) {
        let guard = scope::Guard::enter(self);
        let result = work();
        (result, guard.finish())
    }

    /// Return the current accumulated value.
    #[must_use]
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Load the current accumulated delta and reset to zero.
    ///
    /// Called by the drainer at snapshot time. The reset is a swap to 0 — any
    /// increments racing the swap appear in the NEXT snapshot window (not lost).
    #[must_use]
    pub fn snapshot_and_reset(&self) -> u64 {
        self.value.swap(0, Ordering::Relaxed)
    }
}

#[cfg(feature = "scoped-counters")]
mod scope {
    use core::cell::{Cell, RefCell};

    use super::Counter;

    struct Entry {
        address: usize,
        tally: u64,
    }

    thread_local! {
        // const-init Cell has no lazy init and no destructor, so the no-scope
        // path in `record` is a plain load.
        static ACTIVE: Cell<usize> = const { Cell::new(0) };
        static ENTRIES: RefCell<alloc::vec::Vec<Entry>> = const { RefCell::new(alloc::vec::Vec::new()) };
    }

    pub(super) fn record(counter: &Counter, delta: u64) {
        if ACTIVE.with(Cell::get) == 0 {
            return;
        }
        let address = core::ptr::from_ref(counter) as usize;
        ENTRIES.with(|entries| {
            for entry in entries.borrow_mut().iter_mut() {
                if entry.address == address {
                    entry.tally += delta;
                }
            }
        });
    }

    pub(super) struct Guard {
        depth: usize,
    }

    impl Guard {
        pub(super) fn enter(counter: &Counter) -> Self {
            let address = core::ptr::from_ref(counter) as usize;
            let depth = ENTRIES.with(|entries| {
                let mut entries = entries.borrow_mut();
                entries.push(Entry { address, tally: 0 });
                entries.len()
            });
            ACTIVE.with(|active| active.set(depth));
            Self { depth }
        }

        pub(super) fn finish(self) -> u64 {
            ENTRIES.with(|entries| entries.borrow().get(self.depth - 1).map_or(0, |entry| entry.tally))
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            let remaining = ENTRIES.with(|entries| {
                let mut entries = entries.borrow_mut();
                entries.truncate(self.depth - 1);
                entries.len()
            });
            ACTIVE.with(|active| active.set(remaining));
        }
    }
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

    use super::Counter;

    #[test]
    fn add_increments_value() {
        let counter = Counter::new("hits");
        counter.add(5, &[]);
        counter.add(3, &[]);
        assert_eq!(counter.get(), 8);
    }

    #[test]
    fn builder_fields_carried() {
        let counter = Counter::new("reqs").unit("1").description("total requests");
        assert_eq!(counter.name, "reqs");
        assert_eq!(counter.unit, "1");
        assert_eq!(counter.description, "total requests");
    }

    #[cfg(feature = "scoped-counters")]
    #[test]
    fn scope_counts_only_its_own_thread_while_another_thread_adds() {
        let counter = Counter::new("uploads");
        let ((), own_delta) = std::thread::scope(|threads| {
            counter.scoped(|| {
                counter.add(2, &[]);
                threads.spawn(|| counter.add(100, &[])).join().unwrap();
                counter.add(3, &[]);
            })
        });
        assert_eq!(own_delta, 5);
        assert_eq!(counter.get(), 105);
    }

    #[cfg(feature = "scoped-counters")]
    #[test]
    fn nested_scopes_inner_sees_inner_and_outer_sees_both() {
        let counter = Counter::new("uploads");
        let (inner, outer) = counter.scoped(|| {
            counter.add(1, &[]);
            let ((), inner) = counter.scoped(|| counter.add(4, &[]));
            counter.add(2, &[]);
            inner
        });
        assert_eq!(inner, 4);
        assert_eq!(outer, 7);
    }

    #[cfg(feature = "scoped-counters")]
    #[test]
    fn scope_after_an_add_outside_it_starts_at_zero() {
        let counter = Counter::new("uploads");
        counter.add(9, &[]);
        let ((), delta) = counter.scoped(|| {});
        assert_eq!(delta, 0);
        let other = Counter::new("other");
        let ((), isolated) = counter.scoped(|| other.add(5, &[]));
        assert_eq!(isolated, 0);
    }

    #[test]
    fn size_is_pinned() {
        // name(16) + unit(16) + description(16) + AtomicU64(8) = 56B
        assert_eq!(core::mem::size_of::<Counter>(), 56);
    }
}
