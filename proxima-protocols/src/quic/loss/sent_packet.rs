//! Per-epoch sent-packet tracking per [RFC 9002 §A.6].
//!
//! [RFC 9002 §A.6]: https://www.rfc-editor.org/rfc/rfc9002#section-a.6

use arrayvec::ArrayVec;

use crate::quic::time::Instant;

/// One record per packet we've transmitted that hasn't yet been
/// declared acked, lost, or discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SentPacket {
    pub packet_number: u64,
    pub sent_time: Instant,
    pub size_bytes: u16,
    /// Whether the packet carried any ack-eliciting frames (CRYPTO,
    /// STREAM, PING, etc.). Pure-ACK packets are not ack-eliciting.
    pub is_ack_eliciting: bool,
    /// Whether the packet counts against the congestion window.
    /// Pure-ACK packets are not in flight; almost everything else is.
    pub in_flight: bool,
}

/// Sorted-ascending queue of sent packets per epoch.
///
/// Per the C14 paper proof, drop-oldest on overflow: at capacity, the
/// lowest-PN record is dropped to make room for the new packet. This
/// can produce slightly-optimistic loss detection if the dropped
/// record was still in flight; the trade-off is bounded memory at the
/// cost of an upper bound on how far back loss can be detected.
#[derive(Debug, Clone)]
pub struct SentPacketQueue<const MAX: usize> {
    packets: ArrayVec<SentPacket, MAX>,
}

impl<const MAX: usize> Default for SentPacketQueue<MAX> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX: usize> SentPacketQueue<MAX> {
    /// Construct an empty queue.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            packets: ArrayVec::new_const(),
        }
    }

    /// Number of records currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.packets.len()
    }

    /// True if no packets are currently tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.packets.is_empty()
    }

    /// Push a freshly-sent packet. Returns the record that was dropped
    /// on overflow, if any.
    pub fn push(&mut self, packet: SentPacket) -> Option<SentPacket> {
        let dropped = if self.packets.is_full() {
            Some(self.packets.remove(0))
        } else {
            None
        };
        // Invariant: packets are sorted ascending by PN. Caller MUST
        // push in ascending PN order; debug_assert guards.
        if let Some(last) = self.packets.last() {
            debug_assert!(
                packet.packet_number > last.packet_number,
                "sent packets must be pushed in ascending PN order ({last:?} → {packet:?})",
            );
        }
        let _ = self.packets.try_push(packet);
        dropped
    }

    /// Remove and return the record with `packet_number == pn` if any.
    /// Sorted-ascending invariant allows binary search.
    pub fn remove_by_pn(&mut self, pn: u64) -> Option<SentPacket> {
        let index = self
            .packets
            .binary_search_by_key(&pn, |entry| entry.packet_number)
            .ok()?;
        Some(self.packets.remove(index))
    }

    /// Borrow as a sorted-ascending slice.
    #[must_use]
    pub fn as_slice(&self) -> &[SentPacket] {
        &self.packets
    }

    /// Borrow mutably as a sorted-ascending slice — primarily for tests.
    pub fn as_mut_slice(&mut self) -> &mut [SentPacket] {
        &mut self.packets
    }

    /// Iterate through the queue in ascending PN order.
    pub fn iter(&self) -> impl Iterator<Item = &SentPacket> {
        self.packets.iter()
    }

    /// Retain only the records for which `predicate` returns `true`.
    pub fn retain<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&SentPacket) -> bool,
    {
        self.packets.retain(|record| predicate(record));
    }

    /// Largest packet number currently tracked.
    #[must_use]
    pub fn largest(&self) -> Option<u64> {
        self.packets.last().map(|p| p.packet_number)
    }

    /// Smallest packet number currently tracked.
    #[must_use]
    pub fn smallest(&self) -> Option<u64> {
        self.packets.first().map(|p| p.packet_number)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn sent(pn: u64) -> SentPacket {
        SentPacket {
            packet_number: pn,
            sent_time: Instant::from_micros(1_000_000 + pn * 1_000),
            size_bytes: 1200,
            is_ack_eliciting: true,
            in_flight: true,
        }
    }

    type Small = SentPacketQueue<4>;

    #[test]
    fn new_queue_is_empty() {
        let queue = Small::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        assert_eq!(queue.largest(), None);
    }

    #[test]
    fn push_records_in_ascending_pn_order() {
        let mut queue = Small::new();
        queue.push(sent(10));
        queue.push(sent(11));
        queue.push(sent(12));
        assert_eq!(queue.len(), 3);
        assert_eq!(queue.smallest(), Some(10));
        assert_eq!(queue.largest(), Some(12));
    }

    #[test]
    fn remove_by_pn_uses_binary_search() {
        let mut queue = Small::new();
        for pn in 10..14 {
            queue.push(sent(pn));
        }
        let removed = queue.remove_by_pn(12).expect("found");
        assert_eq!(removed.packet_number, 12);
        assert_eq!(queue.len(), 3);
        assert!(queue.remove_by_pn(99).is_none());
    }

    #[test]
    fn push_at_capacity_drops_oldest() {
        let mut queue = Small::new();
        for pn in 10..14 {
            assert!(queue.push(sent(pn)).is_none());
        }
        assert!(queue.packets.is_full());
        let dropped = queue.push(sent(14)).expect("oldest dropped");
        assert_eq!(dropped.packet_number, 10);
        assert_eq!(queue.smallest(), Some(11));
        assert_eq!(queue.largest(), Some(14));
    }

    #[test]
    fn retain_drops_matching_records() {
        let mut queue = Small::new();
        for pn in 10..14 {
            queue.push(sent(pn));
        }
        queue.retain(|record| record.packet_number != 11);
        assert_eq!(queue.len(), 3);
        assert!(queue.iter().all(|record| record.packet_number != 11));
    }
}
