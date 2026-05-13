//! RFC 1071 internet checksum: one's-complement sum of 16-bit big-endian
//! words, end-around carry folded back, then bit-inverted.
//!
//! The accumulator is incremental so a pseudo-header, a header, and a payload
//! can be summed across separate calls without concatenating them into one
//! buffer (no-alloc requirement).

/// Incremental one's-complement checksum accumulator (RFC 1071).
#[derive(Debug, Clone, Default)]
pub struct Checksum {
    sum: u32,
    pending: Option<u8>,
}

impl Checksum {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sum: 0,
            pending: None,
        }
    }

    /// Fold `data` into the running sum. A trailing odd byte is held and
    /// paired with the first byte of the next call — the words are defined
    /// over the whole concatenated message, not per-call, so the boundary
    /// must not pad mid-stream.
    pub fn add_bytes(&mut self, data: &[u8]) {
        let mut rest = data;
        if let Some(high) = self.pending.take() {
            match rest.split_first() {
                Some((low, tail)) => {
                    self.sum += u32::from(u16::from_be_bytes([high, *low]));
                    rest = tail;
                }
                None => {
                    self.pending = Some(high);
                    return;
                }
            }
        }
        let mut words = rest.chunks_exact(2);
        for word in &mut words {
            self.sum += u32::from(u16::from_be_bytes([word[0], word[1]]));
        }
        if let [last] = words.remainder() {
            self.pending = Some(*last);
        }
    }

    /// Fold carries and invert. A held odd byte pads with a zero low byte.
    #[must_use]
    pub fn finish(self) -> u16 {
        let mut sum = self.sum;
        if let Some(high) = self.pending {
            sum += u32::from(u16::from_be_bytes([high, 0]));
        }
        while (sum >> 16) != 0 {
            sum = (sum & 0xffff) + (sum >> 16);
        }
        !(sum as u16)
    }
}

/// One-shot RFC 1071 checksum over a single contiguous buffer.
#[must_use]
pub fn checksum(data: &[u8]) -> u16 {
    let mut accumulator = Checksum::new();
    accumulator.add_bytes(data);
    accumulator.finish()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use proptest::prelude::*;
    use rstest::rstest;

    // RFC 1071 §3 worked example: derived by hand in the discipline log.
    #[test]
    fn rfc1071_section3_example() {
        let bytes = [0x00, 0x01, 0xf2, 0x03, 0xf4, 0xf5, 0xf6, 0xf7];
        assert_eq!(checksum(&bytes), 0x220d);
    }

    // Canonical IPv4 header with the checksum field zeroed -> 0xb861.
    #[test]
    fn canonical_ipv4_header_example() {
        let header = [
            0x45, 0x00, 0x00, 0x73, 0x00, 0x00, 0x40, 0x00, 0x40, 0x11, 0x00, 0x00, 0xc0, 0xa8,
            0x00, 0x01, 0xc0, 0xa8, 0x00, 0xc7,
        ];
        assert_eq!(checksum(&header), 0xb861);
    }

    // Incremental summing across arbitrary split points must equal the
    // one-shot result, including splits that fall on an odd boundary.
    #[rstest]
    #[case::split_even(4)]
    #[case::split_odd(3)]
    #[case::split_one(1)]
    fn incremental_matches_oneshot(#[case] split: usize) {
        let bytes = [0x00, 0x01, 0xf2, 0x03, 0xf4, 0xf5, 0xf6, 0xf7];
        let (head, tail) = bytes.split_at(split);
        let mut accumulator = Checksum::new();
        accumulator.add_bytes(head);
        accumulator.add_bytes(tail);
        assert_eq!(accumulator.finish(), checksum(&bytes));
    }

    // A correct checksum folded back over the data sums to zero's complement.
    #[test]
    fn verify_round_trips_to_zero_complement() {
        let header = [
            0x45, 0x00, 0x00, 0x73, 0x00, 0x00, 0x40, 0x00, 0x40, 0x11, 0xb8, 0x61, 0xc0, 0xa8,
            0x00, 0x01, 0xc0, 0xa8, 0x00, 0xc7,
        ];
        assert_eq!(checksum(&header), 0x0000);
    }

    proptest! {
        // RFC 1071 §2: splitting a buffer at any byte offset and feeding each
        // half to the incremental accumulator must equal the one-shot result.
        #[test]
        fn incremental_at_any_split_matches_oneshot(
            data in prop::collection::vec(any::<u8>(), 0..256),
            split in any::<usize>(),
        ) {
            let split = if data.is_empty() { 0 } else { split % (data.len() + 1) };
            let (head, tail) = data.split_at(split);

            let mut accumulator = Checksum::new();
            accumulator.add_bytes(head);
            accumulator.add_bytes(tail);
            let incremental = accumulator.finish();

            prop_assert_eq!(incremental, checksum(&data),
                "incremental split at {} diverged from one-shot on {} bytes",
                split, data.len());
        }

        // RFC 1071 verification property: a buffer where the last two bytes
        // hold a valid checksum over the whole buffer folds to zero.
        //
        // The checksum field must sit on a 16-bit word boundary. We truncate
        // the generated prefix to an even length before embedding the checksum.
        #[test]
        fn data_with_embedded_checksum_folds_to_zero(
            raw_prefix in prop::collection::vec(any::<u8>(), 0..254),
        ) {
            // Truncate to even so the checksum word is 16-bit-aligned.
            let even_len = raw_prefix.len() & !1;
            let prefix = &raw_prefix[..even_len];

            // Build: [prefix | 0x00 0x00], compute checksum, embed it.
            let mut buf: Vec<u8> = prefix.to_vec();
            buf.push(0u8);
            buf.push(0u8);
            let computed = checksum(&buf);
            let last = buf.len();
            let sum_bytes = computed.to_be_bytes();
            buf[last - 2] = sum_bytes[0];
            buf[last - 1] = sum_bytes[1];

            // The buffer with the checksum field in place must fold to zero.
            prop_assert_eq!(checksum(&buf), 0u16,
                "verification fold did not yield zero for {}-byte prefix", even_len);
        }
    }
}
