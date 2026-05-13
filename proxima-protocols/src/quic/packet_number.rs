//! Packet number spaces per [RFC 9000 §12.3] + on-the-wire packet-number
//! truncation/expansion per [RFC 9000 §17.1] + [Appendix A].
//!
//! QUIC keeps three independent packet-number spaces during a connection:
//!
//! - **Initial** — for Initial packets (RFC 9000 §17.2.2).
//! - **Handshake** — for Handshake packets (RFC 9000 §17.2.4).
//! - **Application Data** — for 0-RTT and 1-RTT packets (RFC 9000
//!   §17.2.3 + §17.3).
//!
//! Each space has a monotonically-increasing send-side packet number
//! (never reused per `(key, direction)` — the AEAD-nonce invariant from
//! C6 depends on it) and a receive-side largest-PN tracker for ACK
//! generation + duplicate detection.
//!
//! Packet numbers on the wire are 1-, 2-, 3-, or 4-byte truncations of
//! the full 62-bit packet number. The sender picks the length per
//! Appendix A.2 (enough bits to distinguish from the largest-acked PN);
//! the receiver expands per Appendix A.3 using its largest-received PN.
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). The per-space state is plain POD
//! plus a small const-generic bitset for in-window duplicate detection.
//!
//! [RFC 9000 §12.3]: https://www.rfc-editor.org/rfc/rfc9000#section-12.3
//! [RFC 9000 §17.1]: https://www.rfc-editor.org/rfc/rfc9000#section-17.1
//! [Appendix A]: https://www.rfc-editor.org/rfc/rfc9000#appendix-A

/// Maximum representable packet number per RFC 9000 §17.1 (2^62 - 1).
pub const MAX_PACKET_NUMBER: u64 = (1u64 << 62) - 1;

/// One of QUIC's three packet-number spaces per RFC 9000 §12.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PacketNumberSpace {
    Initial,
    Handshake,
    ApplicationData,
}

impl PacketNumberSpace {
    /// Iterator over the three spaces in canonical order.
    pub fn all() -> [Self; 3] {
        [Self::Initial, Self::Handshake, Self::ApplicationData]
    }
}

/// Send-side state for one packet-number space.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SendSpace {
    /// Next packet number to assign. Starts at 0.
    next: u64,
    /// Largest packet number known to have been acked by the peer.
    /// `None` until the first ACK lands. Used to decide the on-the-wire
    /// truncation length per RFC 9000 Appendix A.2.
    largest_acked: Option<u64>,
}

impl SendSpace {
    /// New send-space at PN 0, nothing acked.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            next: 0,
            largest_acked: None,
        }
    }

    /// Peek the next packet number to assign (does not consume).
    #[must_use]
    pub fn peek_next(&self) -> u64 {
        self.next
    }

    /// Allocate the next packet number for an outbound packet. Increments
    /// the counter; the returned value is **never returned again** for
    /// this space.
    ///
    /// # Errors
    ///
    /// Returns [`PacketNumberError::Exhausted`] when the next PN would
    /// exceed [`MAX_PACKET_NUMBER`]. Per RFC 9000 §12.3 the connection
    /// MUST close before this happens (key updates rotate the AEAD key
    /// long before PN exhaustion in practice).
    pub fn assign(&mut self) -> Result<u64, PacketNumberError> {
        if self.next > MAX_PACKET_NUMBER {
            return Err(PacketNumberError::Exhausted);
        }
        let pn = self.next;
        self.next += 1;
        Ok(pn)
    }

    /// Record an ack for `pn`. Updates the largest-acked watermark.
    pub fn record_acked(&mut self, pn: u64) {
        match self.largest_acked {
            Some(current) if current >= pn => {}
            _ => self.largest_acked = Some(pn),
        }
    }

    /// Largest PN known acked, or `None` if no ACK has landed yet.
    #[must_use]
    pub fn largest_acked(&self) -> Option<u64> {
        self.largest_acked
    }
}

/// Receive-side state for one packet-number space.
///
/// The const-generic `WINDOW` parameter sizes the in-window duplicate-
/// detection bitset. Per RFC 9000 §13.2.3 the receiver MUST detect and
/// discard duplicates; a `WINDOW`-bit sliding window covers the most
/// recently observed packet numbers. Anything older than the window is
/// assumed already-acked/already-processed and is dropped.
///
/// Typical value: `WINDOW = 128` (16-byte bitset), comfortable for any
/// realistic ACK delay budget.
#[derive(Debug, Clone, Copy)]
pub struct RecvSpace<const WINDOW: usize> {
    /// Largest packet number seen so far in this space.
    largest_received: Option<u64>,
    /// Sliding bitset: bit `i` (for i in 0..WINDOW) tracks whether PN
    /// `largest_received - i` has been seen. Bit 0 is always set when
    /// `largest_received.is_some()`.
    bitmap: [u8; 64],
}

impl<const WINDOW: usize> Default for RecvSpace<WINDOW> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const WINDOW: usize> RecvSpace<WINDOW> {
    /// New empty recv-space.
    #[must_use]
    pub const fn new() -> Self {
        const {
            assert!(WINDOW <= 512, "RecvSpace WINDOW must be <= 512 bits");
        }
        Self {
            largest_received: None,
            bitmap: [0u8; 64],
        }
    }

    /// Largest packet number seen so far.
    #[must_use]
    pub fn largest_received(&self) -> Option<u64> {
        self.largest_received
    }

    /// Record an incoming packet number. Returns:
    ///
    /// - `Ok(true)` — new packet, not seen before.
    /// - `Ok(false)` — duplicate within the window (or older than window;
    ///   caller MUST silently drop per RFC 9000 §13.2.3).
    ///
    /// # Errors
    ///
    /// [`PacketNumberError::Exhausted`] when `pn > MAX_PACKET_NUMBER`.
    pub fn record_received(&mut self, pn: u64) -> Result<bool, PacketNumberError> {
        if pn > MAX_PACKET_NUMBER {
            return Err(PacketNumberError::Exhausted);
        }
        let largest = match self.largest_received {
            None => {
                // first packet — mark bit 0 + set largest.
                self.largest_received = Some(pn);
                self.bitmap[0] |= 1;
                return Ok(true);
            }
            Some(value) => value,
        };
        if pn > largest {
            // shift bitmap right by (pn - largest) bits, then set bit 0.
            let shift = pn - largest;
            shift_bitmap_right(&mut self.bitmap, shift, WINDOW);
            self.bitmap[0] |= 1;
            self.largest_received = Some(pn);
            Ok(true)
        } else {
            let distance = largest - pn;
            if distance >= WINDOW as u64 {
                // outside the window — treat as duplicate/old, drop.
                return Ok(false);
            }
            let byte_index = (distance / 8) as usize;
            let bit_index = (distance % 8) as u8;
            let bit_mask = 1u8 << bit_index;
            if self.bitmap[byte_index] & bit_mask != 0 {
                Ok(false)
            } else {
                self.bitmap[byte_index] |= bit_mask;
                Ok(true)
            }
        }
    }

    /// Whether `pn` has been observed (used by tests + ACK generation
    /// to query the in-window state).
    #[must_use]
    pub fn contains(&self, pn: u64) -> bool {
        let largest = match self.largest_received {
            Some(value) => value,
            None => return false,
        };
        if pn > largest {
            return false;
        }
        let distance = largest - pn;
        if distance >= WINDOW as u64 {
            return false;
        }
        let byte_index = (distance / 8) as usize;
        let bit_index = (distance % 8) as u8;
        self.bitmap[byte_index] & (1u8 << bit_index) != 0
    }
}

/// Shift `bitmap` right by `shift` bits (so bit `i` becomes bit `i + shift`),
/// keeping only the low `window` bits. Used when a new packet number larger
/// than the current largest_received arrives: the old bits represent older
/// distances and must move further from bit 0.
fn shift_bitmap_right(bitmap: &mut [u8; 64], shift: u64, window: usize) {
    if shift >= window as u64 {
        // shift past the entire window — clear everything.
        bitmap.fill(0);
        return;
    }
    let shift = shift as usize;
    let byte_shift = shift / 8;
    let bit_shift = shift % 8;
    let window_bytes = window.div_ceil(8);
    // operate right-to-left to avoid clobbering input
    let mut accumulator = [0u8; 64];
    if bit_shift == 0 {
        for (index, slot) in accumulator.iter_mut().enumerate().take(window_bytes) {
            let source = index.wrapping_sub(byte_shift);
            if source < window_bytes {
                *slot = bitmap[source];
            }
        }
    } else {
        for (index, slot) in accumulator.iter_mut().enumerate().take(window_bytes) {
            let mut byte = 0u8;
            let source_low = index.wrapping_sub(byte_shift);
            let source_high = index.wrapping_sub(byte_shift + 1);
            if source_low < window_bytes {
                byte |= bitmap[source_low] << bit_shift;
            }
            if source_high < window_bytes {
                byte |= bitmap[source_high] >> (8 - bit_shift);
            }
            *slot = byte;
        }
    }
    // mask off bits beyond the window in the highest byte
    let high_byte = (window - 1) / 8;
    let high_bit = (window - 1) % 8;
    if high_byte < window_bytes {
        // when high_bit == 7 the byte is fully used; avoid the 1u8 << 8 overflow.
        let mask = if high_bit == 7 {
            0xffu8
        } else {
            (1u8 << (high_bit + 1)) - 1
        };
        accumulator[high_byte] &= mask;
        for slot in accumulator.iter_mut().skip(high_byte + 1) {
            *slot = 0;
        }
    }
    *bitmap = accumulator;
}

/// Failures for packet-number operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PacketNumberError {
    /// Packet number reached or exceeded `2^62 - 1`. Per RFC 9000 §12.3
    /// the connection MUST close before this.
    Exhausted,
    /// On-the-wire truncated length exceeded 4 bytes (RFC 9000 §17.1).
    LengthOutOfRange,
}

/// Encode `full_pn` into the shortest representation per RFC 9000 §A.2.
///
/// `largest_acked` is the highest PN the peer has acked (or `None` if
/// none yet) — needed to compute the minimum truncation length.
///
/// Returns `(truncated_value, byte_length)`. Caller writes `byte_length`
/// big-endian bytes of `truncated_value` to the wire.
///
/// # Errors
///
/// [`PacketNumberError::Exhausted`] when `full_pn > MAX_PACKET_NUMBER`.
pub fn encode_packet_number(
    full_pn: u64,
    largest_acked: Option<u64>,
) -> Result<(u64, usize), PacketNumberError> {
    if full_pn > MAX_PACKET_NUMBER {
        return Err(PacketNumberError::Exhausted);
    }
    let num_unacked = match largest_acked {
        None => full_pn + 1,
        Some(la) => full_pn - la,
    };
    // min_bits = log2(num_unacked) + 1
    let min_bits = u64::BITS - num_unacked.leading_zeros();
    let num_bytes = min_bits.div_ceil(8).max(1) as usize;
    let num_bytes = num_bytes.min(4);
    let mask = if num_bytes == 8 {
        u64::MAX
    } else {
        (1u64 << (num_bytes * 8)) - 1
    };
    Ok((full_pn & mask, num_bytes))
}

/// Expand `truncated_pn` (`pn_nbits` bits) using `largest_pn` as the
/// reference per RFC 9000 §A.3.
///
/// # Errors
///
/// [`PacketNumberError::LengthOutOfRange`] when `pn_nbits` is 0 or > 32.
pub fn decode_packet_number(
    largest_pn: u64,
    truncated_pn: u64,
    pn_nbits: u32,
) -> Result<u64, PacketNumberError> {
    if pn_nbits == 0 || pn_nbits > 32 {
        return Err(PacketNumberError::LengthOutOfRange);
    }
    let expected_pn = largest_pn.wrapping_add(1);
    let pn_win: u64 = 1u64 << pn_nbits;
    let pn_hwin: u64 = pn_win / 2;
    let pn_mask: u64 = pn_win - 1;
    let candidate = (expected_pn & !pn_mask) | truncated_pn;

    if candidate + pn_hwin <= expected_pn && candidate + pn_win < (1u64 << 62) {
        return Ok(candidate + pn_win);
    }
    if candidate > expected_pn + pn_hwin && candidate >= pn_win {
        return Ok(candidate - pn_win);
    }
    Ok(candidate)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn space_all_returns_three_canonical_spaces() {
        let spaces = PacketNumberSpace::all();
        assert_eq!(spaces.len(), 3);
        assert_eq!(spaces[0], PacketNumberSpace::Initial);
        assert_eq!(spaces[1], PacketNumberSpace::Handshake);
        assert_eq!(spaces[2], PacketNumberSpace::ApplicationData);
    }

    #[test]
    fn send_space_assigns_monotonically_increasing_pns() {
        let mut space = SendSpace::new();
        assert_eq!(space.assign().unwrap(), 0);
        assert_eq!(space.assign().unwrap(), 1);
        assert_eq!(space.assign().unwrap(), 2);
        assert_eq!(space.peek_next(), 3);
    }

    #[test]
    fn send_space_largest_acked_updates_monotonically() {
        let mut space = SendSpace::new();
        space.record_acked(5);
        assert_eq!(space.largest_acked(), Some(5));
        space.record_acked(3); // older ack is no-op
        assert_eq!(space.largest_acked(), Some(5));
        space.record_acked(10);
        assert_eq!(space.largest_acked(), Some(10));
    }

    #[test]
    fn recv_space_first_packet_is_new() {
        let mut space: RecvSpace<128> = RecvSpace::new();
        assert!(space.record_received(0).unwrap());
        assert_eq!(space.largest_received(), Some(0));
        assert!(space.contains(0));
    }

    #[test]
    fn recv_space_in_order_packets_all_new() {
        let mut space: RecvSpace<128> = RecvSpace::new();
        for pn in 0..50 {
            assert!(space.record_received(pn).unwrap(), "PN {pn} should be new");
        }
        assert_eq!(space.largest_received(), Some(49));
    }

    #[test]
    fn recv_space_duplicate_in_window_rejected() {
        let mut space: RecvSpace<128> = RecvSpace::new();
        assert!(space.record_received(10).unwrap());
        assert!(!space.record_received(10).unwrap(), "duplicate");
    }

    #[test]
    fn recv_space_out_of_order_in_window_accepted() {
        let mut space: RecvSpace<128> = RecvSpace::new();
        space.record_received(10).unwrap();
        assert!(space.record_received(5).unwrap(), "older in-window is new");
        assert!(
            !space.record_received(5).unwrap(),
            "second time is duplicate"
        );
    }

    #[test]
    fn recv_space_below_window_dropped() {
        let mut space: RecvSpace<32> = RecvSpace::new();
        // window 32 - so PN 0 then 100 means 0 is now 100 bits behind, dropped
        space.record_received(0).unwrap();
        space.record_received(100).unwrap();
        // PN 50 is 50 behind largest_received=100 — also outside the window
        assert!(!space.record_received(50).unwrap());
    }

    #[test]
    fn recv_space_shift_preserves_recent_history() {
        let mut space: RecvSpace<128> = RecvSpace::new();
        // mark PN 10 and 20
        space.record_received(10).unwrap();
        space.record_received(20).unwrap();
        // jump to 25 — should still remember 20 and 10
        space.record_received(25).unwrap();
        assert!(space.contains(25));
        assert!(space.contains(20));
        assert!(space.contains(10));
        assert!(!space.contains(15)); // never received
    }

    #[test]
    fn encode_pn_no_ack_first_packet() {
        let (truncated, len) = encode_packet_number(0, None).unwrap();
        assert_eq!(len, 1);
        assert_eq!(truncated, 0);
    }

    #[test]
    fn encode_pn_small_diff_one_byte() {
        // largest_acked = 5, full_pn = 6 — diff = 1, fits in 1 byte
        let (truncated, len) = encode_packet_number(6, Some(5)).unwrap();
        assert_eq!(len, 1);
        assert_eq!(truncated, 6);
    }

    #[test]
    fn encode_pn_rfc_9000_a2_example() {
        // RFC 9000 §A.2: full_pn = 0xac5c02, largest_acked = 0xabe8b3
        // diff = 0xac5c02 - 0xabe8b3 = 0x7d4f → fits in 2 bytes
        let (truncated, len) = encode_packet_number(0xac5c02, Some(0xabe8b3)).unwrap();
        assert_eq!(len, 2);
        assert_eq!(truncated, 0xac5c02 & 0xffff);
    }

    #[test]
    fn decode_pn_rfc_9000_a3_example() {
        // RFC 9000 §A.3: largest_pn=0xa82f30ea (expected = 0xa82f30eb)
        //                truncated_pn=0x9b32 (2 bytes, 16 bits)
        //                → expected window: a82f30eb ± 2^15
        //                → result: 0xa82f9b32
        let result = decode_packet_number(0xa82f30ea, 0x9b32, 16).unwrap();
        assert_eq!(result, 0xa82f9b32);
    }

    #[test]
    fn decode_pn_round_trip_small_values() {
        let mut space = SendSpace::new();
        for full_pn in 0..1000u64 {
            let largest_acked = if full_pn > 0 { Some(full_pn - 1) } else { None };
            let (truncated, len) = encode_packet_number(full_pn, largest_acked).unwrap();
            // simulate receiver having seen largest_received = full_pn - 1
            let largest_received = full_pn.saturating_sub(1);
            let decoded =
                decode_packet_number(largest_received, truncated, (len * 8) as u32).unwrap();
            assert_eq!(decoded, full_pn, "round-trip failed at full_pn = {full_pn}");
            space.assign().unwrap();
        }
    }

    #[test]
    fn decode_pn_invalid_nbits_rejected() {
        assert_eq!(
            decode_packet_number(0, 0, 0),
            Err(PacketNumberError::LengthOutOfRange)
        );
        assert_eq!(
            decode_packet_number(0, 0, 33),
            Err(PacketNumberError::LengthOutOfRange)
        );
    }

    #[test]
    fn encode_pn_value_too_large_rejected() {
        assert_eq!(
            encode_packet_number(MAX_PACKET_NUMBER + 1, None),
            Err(PacketNumberError::Exhausted)
        );
    }
}
