use super::error::DecodeError;
use super::raw::read_u16;

/// Borrowed view over a split-virtqueue available ring (VIRTIO 1.2 spec
/// §2.7.6): le16 flags, le16 idx, then `queue_size` le16 ring entries. This
/// is the driver's producer ring — the device (this codec's caller) only
/// reads it, mirroring how NVMe's controller only reads the host's
/// submission ring. `used_event` (present only under `VIRTIO_F_EVENT_IDX`)
/// is out of scope for M6's virtio-console slice and is not parsed.
#[derive(Debug, Clone, Copy)]
pub struct AvailRing<'ring> {
    bytes: &'ring [u8],
    queue_size: u16,
}

impl<'ring> AvailRing<'ring> {
    pub fn parse(bytes: &'ring [u8], queue_size: u16) -> Result<Self, DecodeError> {
        let need = 4 + usize::from(queue_size) * 2;
        if bytes.len() < need {
            return Err(DecodeError::Truncated {
                need,
                got: bytes.len(),
            });
        }
        Ok(Self { bytes, queue_size })
    }

    #[must_use]
    pub fn flags(&self) -> u16 {
        read_u16(self.bytes, 0)
    }

    /// The driver's free-running publish counter — compare against a
    /// [`RingCursor`]'s `pending` to find newly published entries.
    #[must_use]
    pub fn idx(&self) -> u16 {
        read_u16(self.bytes, 2)
    }

    /// Descriptor-chain head index published at free-running position
    /// `position` (wraps mod `queue_size` into the ring array, per spec).
    #[must_use]
    pub fn ring_entry(&self, position: u16) -> u16 {
        let slot = usize::from(position % self.queue_size);
        read_u16(self.bytes, 4 + slot * 2)
    }
}

/// Device-side cursor over a free-running virtqueue index — the avail ring's
/// publish counter (device reading) or the used ring's publish counter
/// (device writing). Owns no memory, does no I/O; pure index arithmetic,
/// mirroring `super::nvme::queue`'s `SubmissionRing`/`CompletionRing` shape.
///
/// VIRTIO indices are free-running 16-bit counters that wrap at 65536
/// (`u16::wrapping_add` matches the spec exactly, §2.7.6/§2.7.8); the ring
/// array position is the counter modulo `queue_size`, which the spec
/// requires to be a nonzero power of two. This is the one place virtio's
/// ring arithmetic differs from NVMe's: NVMe's tail *is* the slot index, so
/// wrap is `% depth` on the cursor itself; virtio's `idx` is a counter
/// that outlives many wraps of the ring array, so wrap is `% queue_size`
/// only at the point of indexing, never on the counter.
#[derive(Debug, Clone, Copy)]
pub struct RingCursor {
    queue_size: u16,
    last_seen: u16,
}

impl RingCursor {
    pub fn new(queue_size: u16) -> Result<Self, DecodeError> {
        if queue_size == 0 || !queue_size.is_power_of_two() {
            return Err(DecodeError::BadQueueSize { queue_size });
        }
        Ok(Self {
            queue_size,
            last_seen: 0,
        })
    }

    /// Rebuild a cursor from a persisted free-running index — the inverse of
    /// reading [`RingCursor::position`] out, so a stateless engine can keep
    /// the cursor in an atomic and reconstitute the FSM per poll (mirrors
    /// `SubmissionRing::resume`/`CompletionRing::resume`).
    pub fn resume(queue_size: u16, last_seen: u16) -> Result<Self, DecodeError> {
        let mut cursor = Self::new(queue_size)?;
        cursor.last_seen = last_seen;
        Ok(cursor)
    }

    #[must_use]
    pub fn position(&self) -> u16 {
        self.last_seen
    }

    #[must_use]
    pub fn queue_size(&self) -> u16 {
        self.queue_size
    }

    /// How many entries the producer has published since this cursor last
    /// looked, given the producer's current free-running `published` index.
    /// Wrapping subtraction is correct here: both counters wrap at 65536 in
    /// lockstep, so the difference is always the true pending count as long
    /// as the consumer never falls behind by a full 65536 entries.
    #[must_use]
    pub fn pending(&self, published: u16) -> u16 {
        published.wrapping_sub(self.last_seen)
    }

    /// Consume one entry: advance and return the new free-running index —
    /// the value NVMe calls a doorbell and virtio calls `idx`.
    pub fn advance(&mut self) -> u16 {
        self.last_seen = self.last_seen.wrapping_add(1);
        self.last_seen
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    // Worked example: queue_size = 4, one entry published — head index 0 at
    // ring position 0. Byte layout per VIRTIO 1.2 spec §2.7.6.
    //   flags = 0        -> LE16: 00 00
    //   idx   = 1         -> LE16: 01 00
    //   ring[0] = 0 (head) -> LE16: 00 00
    //   ring[1..4] = 0 (unpublished slots) -> LE16 x3: 00 00 00 00 00 00
    fn worked_example_avail() -> [u8; 12] {
        [
            0x00, 0x00, // flags
            0x01, 0x00, // idx
            0x00, 0x00, // ring[0] = head 0
            0x00, 0x00, // ring[1]
            0x00, 0x00, // ring[2]
            0x00, 0x00, // ring[3]
        ]
    }

    #[test]
    fn parse_reads_flags_idx_and_published_head() {
        let bytes = worked_example_avail();
        let ring = AvailRing::parse(&bytes, 4).expect("well-formed avail ring");
        assert_eq!(ring.flags(), 0);
        assert_eq!(ring.idx(), 1);
        assert_eq!(ring.ring_entry(0), 0, "position 0 published head index 0");
    }

    #[test]
    fn ring_entry_wraps_position_modulo_queue_size() {
        let bytes = worked_example_avail();
        let ring = AvailRing::parse(&bytes, 4).expect("well-formed avail ring");
        // position 4 wraps back to slot 0, same as position 0.
        assert_eq!(ring.ring_entry(4), ring.ring_entry(0));
    }

    #[test]
    fn short_buffer_is_truncated() {
        let bytes = [0u8; 11];
        assert_eq!(
            AvailRing::parse(&bytes, 4).unwrap_err(),
            DecodeError::Truncated { need: 12, got: 11 }
        );
    }

    #[test]
    fn cursor_rejects_non_power_of_two_queue_size() {
        for bad in [0u16, 3, 5, 6, 100] {
            assert_eq!(
                RingCursor::new(bad).unwrap_err(),
                DecodeError::BadQueueSize { queue_size: bad }
            );
        }
        assert!(RingCursor::new(4).is_ok());
        assert!(RingCursor::new(256).is_ok());
    }

    #[test]
    fn pending_counts_entries_since_last_seen_with_wraparound() {
        let cursor = RingCursor::new(4).expect("power of two");
        assert_eq!(cursor.pending(0), 0, "nothing published yet");
        assert_eq!(cursor.pending(3), 3);

        // wraparound: last_seen = 65535, published = 1 -> 2 pending (65535, 0).
        let wrapped = RingCursor::resume(4, u16::MAX).expect("power of two");
        assert_eq!(wrapped.pending(1), 2);
    }

    #[test]
    fn advance_wraps_at_65536_matching_the_spec_free_running_counter() {
        let mut cursor = RingCursor::resume(4, u16::MAX).expect("power of two");
        assert_eq!(cursor.advance(), 0, "counter wraps past u16::MAX to 0");
        assert_eq!(cursor.position(), 0);
    }

    #[test]
    fn resume_round_trips_a_persisted_cursor() {
        let mut cursor = RingCursor::new(8).expect("power of two");
        cursor.advance();
        cursor.advance();
        let resumed = RingCursor::resume(8, cursor.position()).expect("power of two");
        assert_eq!(resumed.position(), 2);
    }
}
