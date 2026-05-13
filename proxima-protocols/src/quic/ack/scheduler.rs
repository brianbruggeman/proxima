//! Per-epoch ACK scheduler.
//!
//! Tracks received packet numbers + the "should I emit an ACK now?"
//! decision per the C13 paper proof. The scheduler is state-machine-
//! light: a single struct with five named methods that map 1:1 to
//! paragraphs in [docs/proxima-quic/c13-ack-scheduler-design.md].

use crate::quic::range_set::{ArrayRangeSet, InsertOutcome, RangeInclusive};
use crate::quic::sized;
use crate::quic::time::Instant;

/// Maximum disjoint ACK ranges per scheduler. Sourced from
/// `proxima-quic-proto.toml [ack].max_ranges` (override via
/// `PROXIMA_QUIC_PROTO_ACK_MAX_RANGES`). Drop-oldest on overflow.
pub const MAX_ACK_RANGES: usize = sized::ACK_MAX_RANGES;

/// Default `max_ack_delay` in microseconds (RFC 9000 §18.2 default
/// 25 ms). Sourced from
/// `proxima-quic-proto.toml [ack].default_max_ack_delay_micros`.
pub const DEFAULT_MAX_ACK_DELAY_MICROS: u64 = sized::ACK_DEFAULT_MAX_ACK_DELAY_MICROS;

/// One ACK scheduler per epoch.
#[derive(Debug, Clone)]
pub struct AckScheduler {
    /// Sorted descending range set of received packet numbers.
    ranges: ArrayRangeSet<MAX_ACK_RANGES>,
    /// Number of ack-eliciting packets received since the last ACK was
    /// emitted.
    ack_eliciting_since_last_ack: u32,
    /// When the first ack-eliciting packet of the current pending batch
    /// arrived. `None` once the most recent ACK has been emitted.
    pending_ack_deadline: Option<Instant>,
    /// Force "emit ACK on the next poll_transmit" — set by reorder /
    /// PING / HANDSHAKE_DONE / ECN-CE triggers per RFC 9000 §13.2.1.
    immediate_ack_requested: bool,
    /// Largest packet number we ack'd in a previously-sent ACK frame.
    /// Used by [`Self::has_pending`] to decide whether the next ACK
    /// would carry informationally-new content.
    largest_acked_sent: Option<u64>,
}

impl Default for AckScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl AckScheduler {
    /// Construct a fresh, empty scheduler. ACK emission is never delayed
    /// (see [`Self::should_emit`]); the advertised `max_ack_delay` transport
    /// parameter is the connection's concern (it feeds the peer's PTO via
    /// loss detection), not the ACK emitter's.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ranges: ArrayRangeSet::new(),
            ack_eliciting_since_last_ack: 0,
            pending_ack_deadline: None,
            immediate_ack_requested: false,
            largest_acked_sent: None,
        }
    }

    /// Record a received packet.
    ///
    /// - `pn` — the unprotected, full packet number.
    /// - `is_ack_eliciting` — RFC 9000 §1.2 — `true` for every packet that
    ///   carries at least one ack-eliciting frame; `false` for pure-ACK
    ///   packets and packets that carry only `PADDING` + `CONNECTION_CLOSE`.
    /// - `now` — caller's monotonic clock; used to start the
    ///   `pending_ack_deadline` timer.
    pub fn record_received(&mut self, pn: u64, is_ack_eliciting: bool, now: Instant) {
        let outcome = self.ranges.insert(pn);
        if matches!(outcome, InsertOutcome::InsertedReorder) {
            // RFC 9000 §13.2.1 — receipt of a packet that fills a gap or
            // arrives out-of-order MUST trigger an immediate ACK.
            self.immediate_ack_requested = true;
        }
        if is_ack_eliciting {
            self.ack_eliciting_since_last_ack = self.ack_eliciting_since_last_ack.saturating_add(1);
            if self.pending_ack_deadline.is_none() {
                // Never delay: a lone ack-eliciting packet is due on the next
                // transmit opportunity. The deadline is `now` (no hold timer);
                // it is retained only so next_deadline() can wake the I/O layer
                // to flush a lone ACK when the connection is otherwise idle.
                self.pending_ack_deadline = Some(now);
            }
        }
    }

    /// Explicitly request an immediate ACK on the next emission opportunity.
    /// Used by HANDSHAKE_DONE / PING receive paths per RFC 9000 §13.2.1.
    pub fn request_immediate(&mut self) {
        self.immediate_ack_requested = true;
    }

    /// Should the caller emit an ACK in the next outbound datagram?
    ///
    /// We never delay ACK emission. Two triggers (logical OR):
    /// 1. `immediate_ack_requested` (reorder / PING / HANDSHAKE_DONE).
    /// 2. there is an unacked ack-eliciting packet — `pending_ack_deadline`
    ///    is set to `now` at record time, so this fires on the next transmit
    ///    opportunity. Acking every packet trivially satisfies the RFC 9000
    ///    §13.2.2 every-other-packet SHOULD; holding a lone ACK for
    ///    `max_ack_delay` would only add request/response latency.
    #[must_use]
    pub fn should_emit(&self, now: Instant) -> bool {
        if self.immediate_ack_requested {
            return true;
        }
        if let Some(deadline) = self.pending_ack_deadline
            && now >= deadline
        {
            return true;
        }
        false
    }

    /// Called after the caller has actually emitted an ACK frame whose
    /// `largest_acknowledged` is `largest`. Resets the pending state.
    pub fn on_emitted(&mut self, largest: u64) {
        self.ack_eliciting_since_last_ack = 0;
        self.pending_ack_deadline = None;
        self.immediate_ack_requested = false;
        self.largest_acked_sent = Some(largest);
    }

    /// When the next ACK MUST be emitted (informational; the I/O facade
    /// uses this to schedule wake-ups).
    #[must_use]
    pub fn next_deadline(&self) -> Option<Instant> {
        self.pending_ack_deadline
    }

    /// Is there any packet number in the set that has NOT yet been
    /// included in a previously-emitted ACK frame? Used by
    /// `poll_transmit` to opportunistically coalesce an ACK with
    /// CRYPTO/STREAM data.
    #[must_use]
    pub fn has_pending(&self) -> bool {
        match (self.ranges.largest(), self.largest_acked_sent) {
            (None, _) => false,
            (Some(largest), Some(last_sent)) => largest != last_sent,
            (Some(_), None) => true,
        }
    }

    /// Borrow the underlying range set for ACK-frame encoding.
    #[must_use]
    pub fn ranges(&self) -> &ArrayRangeSet<MAX_ACK_RANGES> {
        &self.ranges
    }

    /// Largest received packet number, or `None` if the set is empty.
    #[must_use]
    pub fn largest_received(&self) -> Option<u64> {
        self.ranges.largest()
    }

    /// Stream of (gap, length) varint pairs per RFC 9000 §19.3.1 for
    /// wire encoding. Yields nothing if the set has zero or one ranges
    /// (single-range ACK has only first_range, no pairs).
    pub fn ack_range_pairs(&self) -> AckRangeIter<'_> {
        AckRangeIter {
            ranges: self.ranges.as_slice(),
            index: 1,
        }
    }

    /// The `first_range` field for the ACK frame: largest range's length
    /// minus 1 per RFC 9000 §19.3.1.
    #[must_use]
    pub fn first_range_length(&self) -> Option<u64> {
        self.ranges.iter().next().map(|range| range.len() - 1)
    }

    /// The largest packet number in the set, encoded as the ACK frame's
    /// `largest_acknowledged` field. `None` if the scheduler is empty.
    #[must_use]
    pub fn largest_for_frame(&self) -> Option<u64> {
        self.ranges.largest()
    }
}

/// Iterator over `(gap, ack_range_length)` pairs per RFC 9000 §19.3.1.
///
/// Each pair is computed from adjacent ranges in the descending-end-sorted
/// set:
///
/// ```text
///   gap_i    = ranges[i - 1].start - ranges[i].end - 2
///   length_i = ranges[i].end - ranges[i].start
/// ```
pub struct AckRangeIter<'a> {
    ranges: &'a [RangeInclusive],
    index: usize,
}

impl Iterator for AckRangeIter<'_> {
    type Item = AckRangePair;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.ranges.get(self.index)?;
        let previous = &self.ranges[self.index - 1];
        let gap = previous.start - current.end - 2;
        let length = current.end - current.start;
        self.index += 1;
        Some(AckRangePair { gap, length })
    }
}

/// One `(gap, length)` pair per RFC 9000 §19.3.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AckRangePair {
    pub gap: u64,
    pub length: u64,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn at(micros: u64) -> Instant {
        Instant::from_micros(micros)
    }

    #[test]
    fn new_scheduler_emits_nothing() {
        let scheduler = AckScheduler::new();
        assert!(!scheduler.should_emit(at(0)));
        assert!(!scheduler.has_pending());
        assert_eq!(scheduler.next_deadline(), None);
    }

    #[test]
    fn lone_ack_eliciting_emits_immediately() {
        // Never delay: a single in-order ack-eliciting packet is due on the
        // next transmit opportunity — deadline == now, should_emit at once.
        let mut scheduler = AckScheduler::new();
        scheduler.record_received(100, true, at(1_000_000));
        assert_eq!(scheduler.next_deadline(), Some(at(1_000_000)));
        assert!(scheduler.has_pending());
        assert!(scheduler.should_emit(at(1_000_000)));
    }

    #[test]
    fn record_two_ack_eliciting_triggers_emit() {
        let mut scheduler = AckScheduler::new();
        scheduler.record_received(100, true, at(1_000_000));
        scheduler.record_received(101, true, at(1_010_000));
        assert!(scheduler.should_emit(at(1_010_001)));
    }

    #[test]
    fn pure_ack_does_not_count_toward_eliciting_or_deadline() {
        let mut scheduler = AckScheduler::new();
        scheduler.record_received(100, false, at(1_000_000));
        assert_eq!(scheduler.next_deadline(), None);
        assert!(!scheduler.should_emit(at(1_999_999)));
        // The range is still recorded — has_pending is true because we
        // haven't ack'd 100 yet.
        assert!(scheduler.has_pending());
    }

    #[test]
    fn reorder_triggers_immediate_emit() {
        let mut scheduler = AckScheduler::new();
        scheduler.record_received(100, true, at(1_000_000));
        scheduler.record_received(102, true, at(1_001_000));
        // Skipped 101 → next 101 inserted should reorder-fire AND be
        // ack-eliciting, but the previous record(102) ALREADY fired the
        // immediate flag because at insertion time 102 was not adjacent
        // to 100. Actually with the descending sort, 102 first becomes
        // the new top — no reorder. Insert 101 next → it fills gap → reorder.
        scheduler.record_received(101, true, at(1_002_000));
        assert!(scheduler.immediate_ack_requested);
        assert!(scheduler.should_emit(at(1_002_001)));
    }

    #[test]
    fn lone_packet_does_not_wait_for_a_second() {
        // Never delay: a single ack-eliciting packet (count 1, < 2) is due
        // immediately — it does not wait for a 2nd packet or any timer.
        let mut scheduler = AckScheduler::new();
        scheduler.record_received(100, true, at(1_000_000));
        assert_eq!(scheduler.ack_eliciting_since_last_ack, 1);
        assert!(scheduler.should_emit(at(1_000_000)));
    }

    #[test]
    fn on_emitted_clears_pending_state() {
        let mut scheduler = AckScheduler::new();
        scheduler.record_received(100, true, at(1_000_000));
        scheduler.record_received(101, true, at(1_005_000));
        assert!(scheduler.should_emit(at(1_005_001)));
        scheduler.on_emitted(101);
        assert!(!scheduler.should_emit(at(2_000_000)));
        assert_eq!(scheduler.next_deadline(), None);
        // has_pending should now be false because largest_acked_sent
        // matches the largest range end.
        assert!(!scheduler.has_pending());
    }

    #[test]
    fn ack_range_pairs_for_two_disjoint_ranges() {
        let mut scheduler = AckScheduler::new();
        for pn in &[100u64, 101, 102] {
            scheduler.record_received(*pn, true, at(1_000_000));
        }
        for pn in &[104u64, 105] {
            scheduler.record_received(*pn, true, at(1_000_000));
        }
        // ranges (descending): [{104,105}, {100,102}]
        assert_eq!(scheduler.first_range_length(), Some(1));
        let pairs: alloc::vec::Vec<_> = scheduler.ack_range_pairs().collect();
        assert_eq!(pairs.len(), 1);
        // gap = 104 - 102 - 2 = 0; length = 102 - 100 = 2
        assert_eq!(pairs[0], AckRangePair { gap: 0, length: 2 });
    }

    #[test]
    fn worked_example_from_design_doc() {
        // Walks docs/proxima-quic/c13-ack-scheduler-design.md under the
        // never-delay policy: every ack-eliciting packet sets the deadline to
        // `now`, so it is emittable on the next transmit opportunity. The
        // eliciting counter still tracks unacked packets but no longer gates
        // emission; reorder is covered by reorder_triggers_immediate_emit.
        let mut scheduler = AckScheduler::new();

        // record(100, eliciting) → deadline now, emittable at once.
        scheduler.record_received(100, true, at(5_000_000));
        assert_eq!(scheduler.next_deadline(), Some(at(5_000_000)));
        assert_eq!(scheduler.ack_eliciting_since_last_ack, 1);
        assert!(scheduler.should_emit(at(5_000_000)));

        // record(101, eliciting) → counter 2, still emittable.
        scheduler.record_received(101, true, at(5_010_000));
        assert_eq!(scheduler.ack_eliciting_since_last_ack, 2);
        assert!(scheduler.should_emit(at(5_010_000)));

        // emit → pending state cleared.
        scheduler.on_emitted(101);
        assert!(!scheduler.should_emit(at(5_010_002)));
        assert_eq!(scheduler.next_deadline(), None);

        // record(102, eliciting) → fresh deadline now, emittable at once.
        scheduler.record_received(102, true, at(5_015_000));
        assert_eq!(scheduler.next_deadline(), Some(at(5_015_000)));
        assert!(scheduler.should_emit(at(5_015_000)));
        scheduler.on_emitted(102);

        // record(150, NON-eliciting) — pure-ACK: no counter bump, no deadline.
        scheduler.record_received(150, false, at(6_000_000));
        assert_eq!(scheduler.ack_eliciting_since_last_ack, 0);
        assert_eq!(scheduler.next_deadline(), None);

        // record(151, eliciting) → deadline now, emittable at once.
        scheduler.record_received(151, true, at(6_001_000));
        assert_eq!(scheduler.ack_eliciting_since_last_ack, 1);
        assert_eq!(scheduler.next_deadline(), Some(at(6_001_000)));
        assert!(scheduler.should_emit(at(6_001_000)));
    }

    extern crate alloc;
}
