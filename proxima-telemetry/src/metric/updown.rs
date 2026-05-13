use core::sync::atomic::{AtomicI64, Ordering};

use crate::tag::Tag;

/// An up-down counter — supports positive and negative deltas.
///
/// Backed by `AtomicI64`; relaxed ordering matches the single-accumulator
/// v1 contract. Per-attr-set sharding is the opt-sweep target.
pub struct UpDownCounter {
    pub name: &'static str,
    pub unit: &'static str,
    pub description: &'static str,
    value: AtomicI64,
    // last value reported to a drain, so a steady sum goes quiet (otherwise it
    // would re-publish every pass and spin the managed drainer's drain loop).
    last_value: AtomicI64,
}

impl UpDownCounter {
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            unit: "",
            description: "",
            value: AtomicI64::new(0),
            last_value: AtomicI64::new(0),
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

    /// Add a signed `delta` (positive increments, negative decrements).
    pub fn add(&self, delta: i64, _tags: &[Tag]) {
        self.value.fetch_add(delta, Ordering::Relaxed);
    }

    /// Return the current accumulated value.
    #[must_use]
    pub fn get(&self) -> i64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Drain-time snapshot: the cumulative value IF it changed since the last
    /// drain, else `None`. Report-on-change keeps a steady sum from re-publishing
    /// every pass (which would spin the managed drainer).
    #[must_use]
    pub fn snapshot_if_changed(&self) -> Option<i64> {
        let value = self.value.load(Ordering::Relaxed);
        if value == self.last_value.swap(value, Ordering::Relaxed) {
            return None;
        }
        Some(value)
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

    use super::UpDownCounter;

    #[test]
    fn add_positive_increments() {
        let counter = UpDownCounter::new("queue_depth");
        counter.add(10, &[]);
        assert_eq!(counter.get(), 10);
    }

    #[test]
    fn add_negative_decrements() {
        let counter = UpDownCounter::new("queue_depth");
        counter.add(10, &[]);
        counter.add(-4, &[]);
        assert_eq!(counter.get(), 6);
    }

    #[test]
    fn size_is_pinned() {
        // name+unit+desc(48) + value(8) + last_value(8) = 64B
        assert_eq!(core::mem::size_of::<UpDownCounter>(), 64);
    }

    #[test]
    fn snapshot_if_changed_quiets_when_steady() {
        let counter = UpDownCounter::new("c");
        counter.add(5, &[]);
        assert_eq!(counter.snapshot_if_changed(), Some(5));
        assert_eq!(counter.snapshot_if_changed(), None, "steady is quiet");
        counter.add(-2, &[]);
        assert_eq!(
            counter.snapshot_if_changed(),
            Some(3),
            "change reports the new total"
        );
    }
}
