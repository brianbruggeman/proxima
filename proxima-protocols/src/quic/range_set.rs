//! Sorted descending packet-number range set per [RFC 9000 §19.3.1].
//!
//! Used by the ACK scheduler to accumulate received packet numbers
//! and emit a wire ACK frame. Per the C13 design pass
//! ([docs/proxima-quic/c13-ack-scheduler-design.md]) the storage is
//! a fixed-cap `ArrayVec<RangeInclusive, MAX>` sorted by descending
//! `end` (RFC ACK encoding starts from the largest received packet
//! number and walks down via gap+length pairs).
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). The `arrayvec::ArrayVec` is
//! no-alloc; the algorithm uses only stack-local state.
//!
//! # Overflow policy
//!
//! When full and the inserted PN would create a new range (not extend
//! an existing one), the oldest range (smallest `end`) is dropped.
//! Rationale per [c13-ack-scheduler-design.md]: RFC 9000 §13.2.4 only
//! REQUIRES acknowledging the largest packet number; older ranges are
//! a recovery optimisation. Drop-oldest preserves the recent
//! recovery-critical info.
//!
//! [RFC 9000 §19.3.1]: https://www.rfc-editor.org/rfc/rfc9000#section-19.3.1

use arrayvec::ArrayVec;

/// Inclusive packet-number range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeInclusive {
    pub start: u64,
    pub end: u64,
}

impl RangeInclusive {
    /// Construct a singleton range `[pn, pn]`.
    #[must_use]
    pub const fn singleton(pn: u64) -> Self {
        Self { start: pn, end: pn }
    }

    /// Number of packet numbers in the range, inclusive. RangeInclusive
    /// always contains at least one packet number (start <= end at
    /// construction), so `len() >= 1`; no `is_empty` companion is needed.
    #[must_use]
    #[allow(clippy::len_without_is_empty)]
    pub const fn len(&self) -> u64 {
        self.end - self.start + 1
    }

    /// Does this range contain `pn`?
    #[must_use]
    pub const fn contains(&self, pn: u64) -> bool {
        self.start <= pn && pn <= self.end
    }
}

/// Insertion outcome — useful for tests and the scheduler reorder check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InsertOutcome {
    /// PN was already covered by an existing range; no mutation.
    Duplicate,
    /// PN extended the highest range upward (the new largest).
    ExtendedNewMax,
    /// PN extended a non-highest range (gap fill or new lower range).
    /// This indicates reorder / loss recovery from the receiver's
    /// perspective and should trigger the scheduler's immediate-ACK flag.
    InsertedReorder,
}

/// Sorted descending interval set with const-generic capacity.
#[derive(Debug, Clone)]
pub struct ArrayRangeSet<const MAX: usize> {
    ranges: ArrayVec<RangeInclusive, MAX>,
}

impl<const MAX: usize> Default for ArrayRangeSet<MAX> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX: usize> ArrayRangeSet<MAX> {
    /// Construct an empty range set.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ranges: ArrayVec::new_const(),
        }
    }

    /// Number of disjoint ranges.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// True if no ranges have been inserted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// True if at capacity (`MAX` ranges).
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.ranges.is_full()
    }

    /// Iterate over the ranges in descending-`end` order.
    pub fn iter(&self) -> impl Iterator<Item = &RangeInclusive> {
        self.ranges.iter()
    }

    /// Largest packet number recorded so far.
    #[must_use]
    pub fn largest(&self) -> Option<u64> {
        self.ranges.first().map(|range| range.end)
    }

    /// Insert `pn`. Returns the insertion outcome.
    ///
    /// Per the C13 paper proof:
    ///
    /// 1. Find the range that `pn` either extends-down (its `start - 1 == pn`),
    ///    extends-up (its `end + 1 == pn`), or merges (both conditions on
    ///    adjacent ranges).
    /// 2. Otherwise insert a new singleton at the sorted position.
    /// 3. On overflow, drop the oldest range (smallest `end`).
    ///
    /// `InsertOutcome::ExtendedNewMax` is returned iff `pn` is strictly
    /// greater than every previously-recorded packet number (i.e. `pn`
    /// is the new max). Otherwise (gap-fill, merge of non-top ranges,
    /// pure-reorder, pre-existing duplicate) the outcome reflects the
    /// reorder/gap-fill condition that the scheduler uses to trigger
    /// the RFC 9000 §13.2.1 immediate-ACK rule.
    pub fn insert(&mut self, pn: u64) -> InsertOutcome {
        let previous_max = self.largest();
        // Locate first range whose end < pn — that's where pn would naturally land.
        // Ranges are sorted DESC by end.
        let mut insert_at = self.ranges.len();
        for (index, range) in self.ranges.iter().enumerate() {
            if range.end < pn {
                insert_at = index;
                break;
            }
            if range.contains(pn) {
                return InsertOutcome::Duplicate;
            }
        }

        let extends_down = insert_at > 0
            && self
                .ranges
                .get(insert_at - 1)
                .map(|range| range.start == pn + 1)
                .unwrap_or(false);
        let extends_up = self
            .ranges
            .get(insert_at)
            .map(|range| range.end + 1 == pn)
            .unwrap_or(false);

        match (extends_down, extends_up) {
            (true, true) => {
                let lower_start = self.ranges[insert_at].start;
                self.ranges[insert_at - 1].start = lower_start;
                self.ranges.remove(insert_at);
            }
            (true, false) => {
                self.ranges[insert_at - 1].start = pn;
            }
            (false, true) => {
                self.ranges[insert_at].end = pn;
            }
            (false, false) => {
                let new_range = RangeInclusive::singleton(pn);
                if self.ranges.is_full() {
                    if insert_at == self.ranges.len() {
                        return InsertOutcome::Duplicate;
                    }
                    self.ranges.pop();
                }
                self.ranges.insert(insert_at, new_range);
            }
        }
        match previous_max {
            None => InsertOutcome::ExtendedNewMax,
            Some(prev) if pn > prev => InsertOutcome::ExtendedNewMax,
            Some(_) => InsertOutcome::InsertedReorder,
        }
    }

    /// Borrow the underlying range slice (for tests + bench).
    #[must_use]
    pub fn as_slice(&self) -> &[RangeInclusive] {
        &self.ranges
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    type SmallSet = ArrayRangeSet<8>;

    #[test]
    fn new_is_empty() {
        let set = SmallSet::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert_eq!(set.largest(), None);
    }

    #[test]
    fn single_insert_creates_singleton() {
        let mut set = SmallSet::new();
        let outcome = set.insert(42);
        assert_eq!(outcome, InsertOutcome::ExtendedNewMax);
        assert_eq!(set.len(), 1);
        assert_eq!(set.largest(), Some(42));
        assert_eq!(set.as_slice()[0], RangeInclusive::singleton(42));
    }

    #[test]
    fn duplicate_insert_is_no_op() {
        let mut set = SmallSet::new();
        set.insert(10);
        let outcome = set.insert(10);
        assert_eq!(outcome, InsertOutcome::Duplicate);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn ascending_inserts_merge_into_one_range() {
        let mut set = SmallSet::new();
        for pn in 100..=104 {
            set.insert(pn);
        }
        assert_eq!(set.len(), 1);
        assert_eq!(
            set.as_slice()[0],
            RangeInclusive {
                start: 100,
                end: 104
            }
        );
    }

    #[test]
    fn descending_inserts_merge_into_one_range() {
        let mut set = SmallSet::new();
        for pn in (100..=104).rev() {
            set.insert(pn);
        }
        assert_eq!(set.len(), 1);
        assert_eq!(
            set.as_slice()[0],
            RangeInclusive {
                start: 100,
                end: 104
            }
        );
    }

    #[test]
    fn gap_fill_merges_two_adjacent_ranges() {
        let mut set = SmallSet::new();
        set.insert(100);
        set.insert(101);
        set.insert(103);
        set.insert(104);
        assert_eq!(set.len(), 2);
        let outcome = set.insert(102);
        // After merge, all should collapse into one range. The outcome
        // is InsertedReorder because 102 < previous max (104) — gap-fill
        // is the RFC 9000 §13.2.1 immediate-ACK trigger.
        assert_eq!(outcome, InsertOutcome::InsertedReorder);
        assert_eq!(set.len(), 1);
        assert_eq!(
            set.as_slice()[0],
            RangeInclusive {
                start: 100,
                end: 104
            }
        );
    }

    #[test]
    fn out_of_order_insert_creates_reorder_range() {
        let mut set = SmallSet::new();
        set.insert(100);
        set.insert(101);
        set.insert(102);
        let outcome = set.insert(50);
        assert_eq!(outcome, InsertOutcome::InsertedReorder);
        assert_eq!(set.len(), 2);
        assert_eq!(
            set.as_slice()[0],
            RangeInclusive {
                start: 100,
                end: 102
            }
        );
        assert_eq!(set.as_slice()[1], RangeInclusive::singleton(50));
    }

    #[test]
    fn ranges_stay_sorted_descending() {
        let mut set = SmallSet::new();
        for pn in &[50u64, 100, 75, 200, 25, 150] {
            set.insert(*pn);
        }
        let ends: alloc::vec::Vec<u64> = set.iter().map(|r| r.end).collect();
        for window in ends.windows(2) {
            assert!(window[0] > window[1], "ends must be descending: {ends:?}");
        }
    }

    #[test]
    fn overflow_drops_oldest_range() {
        let mut set: ArrayRangeSet<3> = ArrayRangeSet::new();
        set.insert(100);
        set.insert(200);
        set.insert(300);
        assert!(set.is_full());
        // Insert 400: extends-up of 300? 300+1=301 != 400, so new singleton.
        // At capacity → drop oldest (100).
        set.insert(400);
        assert_eq!(set.len(), 3);
        assert_eq!(set.largest(), Some(400));
        assert_eq!(
            set.as_slice(),
            &[
                RangeInclusive::singleton(400),
                RangeInclusive::singleton(300),
                RangeInclusive::singleton(200),
            ]
        );
    }

    #[test]
    fn overflow_with_pn_smaller_than_all_is_no_op() {
        let mut set: ArrayRangeSet<2> = ArrayRangeSet::new();
        set.insert(100);
        set.insert(200);
        // 50 is older than all; at capacity → would be the dropped one
        // anyway, so the insert is a no-op.
        let outcome = set.insert(50);
        assert_eq!(outcome, InsertOutcome::Duplicate);
        assert_eq!(set.len(), 2);
        assert_eq!(set.largest(), Some(200));
    }

    extern crate alloc;
}
