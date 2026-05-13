use core::sync::atomic::{AtomicU64, Ordering};

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

    #[test]
    fn size_is_pinned() {
        // name(16) + unit(16) + description(16) + AtomicU64(8) = 56B
        assert_eq!(core::mem::size_of::<Counter>(), 56);
    }
}
