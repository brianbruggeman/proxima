//! HPACK variable-length unsigned integer codec (RFC 7541 §5.1).
//!
//! Integers in HPACK are length-prefixed by `N` bits of a byte (the
//! upper `8 - N` bits of that byte are reserved for type flags that
//! distinguish indexed / literal / table-size-update / etc.). The
//! encoder fits `value` into the N-bit prefix when possible; otherwise
//! it stores `(2^N - 1)` in the prefix and follows with continuation
//! bytes carrying 7 bits each (high bit = "more follows").
//!
//! ```text
//!     0   1   2   3   4   5   6   7
//!   +---+---+---+---+---+---+---+---+
//!   | ? | ? | ? |       Value       |    <- if Value < 2^N - 1
//!   +---+---+---+-------------------+
//!
//!     0   1   2   3   4   5   6   7
//!   +---+---+---+---+---+---+---+---+
//!   | ? | ? | ? | 1   1   1   1   1 |    <- 2^N - 1 marker (N=5)
//!   +---+---+---+-------------------+
//!   | 1 |    Value-(2^N-1) low 7    |    <- continuation
//!   +---+---------------------------+
//!   | 1 |    Value-(2^N-1) mid 7    |
//!   +---+---------------------------+
//!   | 0 |    Value-(2^N-1) high 7   |    <- final (bit 7 = 0)
//!   +---+---------------------------+
//! ```
//!
//! No allocation: encoder appends to any `BufMut`; decoder reads a
//! `&[u8]` and returns `(value, bytes_consumed)`. Decoder validates
//! against `u32` overflow on continuation accumulation (§5.1 lets
//! integers grow arbitrarily large; we cap at u32 because every
//! HPACK use in RFC 7541 fits — table sizes, indices, string
//! lengths bounded by SETTINGS_MAX_HEADER_LIST_SIZE).

use bytes::BufMut;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HpackError {
    #[error("HPACK integer truncated: more continuation bytes expected")]
    IntegerTruncated,
    #[error("HPACK integer overflows u32 during decode")]
    IntegerOverflow,
    #[error("HPACK prefix_bits must be in 1..=8, got {got}")]
    InvalidPrefixBits { got: u8 },
}

/// Encode an unsigned integer per RFC 7541 §5.1.
///
/// - `value`: the integer to encode (0..=u32::MAX).
/// - `prefix_bits`: `N` in `1..=8`. Lower `N` bits of the first byte
///   carry the integer (or a continuation marker); upper `8 - N` bits
///   are reserved for the caller's type flags.
/// - `flags`: the upper `8 - N` bits of the first byte. The encoder
///   OR-merges them in. Must have its low `N` bits clear.
///
/// Panics in debug mode if `prefix_bits` is out of range or if
/// `flags` collides with the prefix region.
///
/// `inline(always)` is load-bearing: without it the cross-crate
/// call survives LTO (probably because of the generic `BufMut`
/// parameter), costing 2× per call vs the inlined form. Verified
/// with `hpack_integer` bench against h2-0.4.14.
#[inline(always)]
pub fn encode_integer<B: BufMut>(value: u32, prefix_bits: u8, flags: u8, dst: &mut B) {
    debug_assert!((1..=8).contains(&prefix_bits), "prefix_bits must be 1..=8");
    let max_prefix_value: u32 = (1u32 << prefix_bits) - 1;
    debug_assert_eq!(
        flags & (max_prefix_value as u8),
        0,
        "flags must not overlap prefix region"
    );
    if value < max_prefix_value {
        dst.put_u8(flags | (value as u8));
        return;
    }
    dst.put_u8(flags | (max_prefix_value as u8));
    let mut remaining = value - max_prefix_value;
    while remaining >= 128 {
        dst.put_u8(0x80 | (remaining as u8));
        remaining >>= 7;
    }
    dst.put_u8(remaining as u8);
}

/// Decode an unsigned integer per RFC 7541 §5.1.
///
/// Returns the decoded value and the number of bytes consumed (which
/// includes the prefix byte plus any continuation bytes).
///
/// `prefix_bits` is `N` in `1..=8`. The decoder reads the low `N`
/// bits of `buf[0]`; the caller is responsible for stripping or
/// interpreting the upper `8 - N` flag bits.
///
/// Errors:
/// - `IntegerTruncated`: the continuation chain didn't terminate
///   within `buf`.
/// - `IntegerOverflow`: the accumulated value exceeds `u32::MAX`.
#[inline(always)]
pub fn decode_integer(buf: &[u8], prefix_bits: u8) -> Result<(u32, usize), HpackError> {
    if !(1..=8).contains(&prefix_bits) {
        return Err(HpackError::InvalidPrefixBits { got: prefix_bits });
    }
    if buf.is_empty() {
        return Err(HpackError::IntegerTruncated);
    }
    let max_prefix_value: u32 = (1u32 << prefix_bits) - 1;
    let first = (buf[0] as u32) & max_prefix_value;
    if first < max_prefix_value {
        return Ok((first, 1));
    }
    // Continuation bytes: each contributes 7 bits, low-to-high. Bit 7
    // of each continuation = "more follows"; the byte with bit 7 = 0
    // is the last.
    let mut value: u32 = max_prefix_value;
    let mut shift: u32 = 0;
    for (offset, byte) in buf[1..].iter().enumerate() {
        let contribution = (*byte & 0x7f) as u32;
        // shift can be 0, 7, 14, 21, 28; at 28 the next would be 35
        // (would shift into nothingness). Detect overflow before
        // shifting via checked_shl.
        let shifted = contribution
            .checked_shl(shift)
            .ok_or(HpackError::IntegerOverflow)?;
        value = value
            .checked_add(shifted)
            .ok_or(HpackError::IntegerOverflow)?;
        if byte & 0x80 == 0 {
            return Ok((value, offset + 2)); // +2 = prefix byte + this byte
        }
        shift += 7;
        if shift > 28 {
            // Next contribution shift would overflow u32. The next
            // byte must either continue (overflow) or terminate; if
            // it terminates with bit 7 set we'd still overflow. Force
            // overflow check by attempting another shift.
            //
            // RFC §5.1 doesn't specify a max integer size, but every
            // HPACK use fits in u32 (table sizes, indices, lengths
            // bounded by SETTINGS_MAX_HEADER_LIST_SIZE which is u32).
            // Reject larger integers as a protocol error.
            return Err(HpackError::IntegerOverflow);
        }
    }
    Err(HpackError::IntegerTruncated)
}

#[cfg(all(test, not(feature = "hpack-no-alloc")))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// RFC 7541 §C.1.1: encoding the integer 10 with a 5-bit prefix.
    /// Expected wire: `00001010` (the prefix flags region is zero).
    #[test]
    fn rfc_c_1_1_encode_10_with_5_bit_prefix() {
        let mut out = Vec::new();
        encode_integer(10, 5, 0, &mut out);
        assert_eq!(out, vec![0b0000_1010]);
    }

    #[test]
    fn rfc_c_1_1_decode_10_with_5_bit_prefix() {
        let (value, consumed) = decode_integer(&[0b0000_1010], 5).unwrap();
        assert_eq!(value, 10);
        assert_eq!(consumed, 1);
    }

    /// RFC 7541 §C.1.2: encoding 1337 with a 5-bit prefix.
    /// Expected wire:
    ///   0001 1111  <- prefix (5 bits = 31, marker)
    ///   1001 1010  <- continuation: (1337 - 31) % 128 = 154, OR'd with 0x80
    ///   0000 1010  <- final:        (1337 - 31) / 128 = 10
    #[test]
    fn rfc_c_1_2_encode_1337_with_5_bit_prefix() {
        let mut out = Vec::new();
        encode_integer(1337, 5, 0, &mut out);
        assert_eq!(out, vec![0b0001_1111, 0b1001_1010, 0b0000_1010]);
    }

    #[test]
    fn rfc_c_1_2_decode_1337_with_5_bit_prefix() {
        let (value, consumed) =
            decode_integer(&[0b0001_1111, 0b1001_1010, 0b0000_1010], 5).unwrap();
        assert_eq!(value, 1337);
        assert_eq!(consumed, 3);
    }

    /// RFC 7541 §C.1.3: encoding 42 with an 8-bit prefix.
    #[test]
    fn rfc_c_1_3_encode_42_with_8_bit_prefix() {
        let mut out = Vec::new();
        encode_integer(42, 8, 0, &mut out);
        assert_eq!(out, vec![0b0010_1010]);
    }

    #[test]
    fn rfc_c_1_3_decode_42_with_8_bit_prefix() {
        let (value, consumed) = decode_integer(&[0b0010_1010], 8).unwrap();
        assert_eq!(value, 42);
        assert_eq!(consumed, 1);
    }

    /// Flags region: encoder must OR the flags into the upper bits of
    /// the first byte without disturbing the integer encoding.
    #[test]
    fn encoder_ors_flags_into_upper_bits() {
        let mut out = Vec::new();
        // 3-bit prefix means upper 5 bits are flags. Encode value 5
        // (< 7 = 2^3 - 1, fits in prefix) with flags 0b1010_0000.
        encode_integer(5, 3, 0b1010_0000, &mut out);
        assert_eq!(out, vec![0b1010_0000 | 0b0000_0101]);
    }

    /// Decoder must ignore the upper (8 - prefix_bits) flag bits of
    /// the first byte — only the lower N bits encode the integer.
    #[test]
    fn decoder_ignores_flag_bits_in_first_byte() {
        // Same wire as the previous test (flags + value 5 in 3-bit prefix).
        // Decoder should report value=5 regardless of flags.
        let (value, consumed) = decode_integer(&[0b1010_0000 | 0b0000_0101], 3).unwrap();
        assert_eq!(value, 5);
        assert_eq!(consumed, 1);
    }

    /// Round-trip across all prefix sizes for a sweep of values that
    /// straddle the prefix boundary.
    #[test]
    fn round_trip_across_prefix_sizes() {
        for prefix_bits in 1_u8..=8 {
            let max_inline = (1u32 << prefix_bits) - 1;
            // pick values around the boundary + a few large ones
            let cases: Vec<u32> = vec![
                0,
                1,
                max_inline.saturating_sub(1),
                max_inline,
                max_inline + 1,
                max_inline.saturating_add(127),
                max_inline.saturating_add(128),
                1_000_000,
                u32::MAX - 128,
            ];
            for value in cases {
                let mut buf = Vec::new();
                encode_integer(value, prefix_bits, 0, &mut buf);
                let (decoded, consumed) = decode_integer(&buf, prefix_bits).unwrap();
                assert_eq!(
                    decoded, value,
                    "round-trip @ prefix={prefix_bits} value={value}"
                );
                assert_eq!(consumed, buf.len(), "consumed != produced");
            }
        }
    }

    /// Encoding u32::MAX with an 8-bit prefix must round-trip.
    /// 2^32 - 1 = 4294967295. With 8-bit prefix, the prefix marker is
    /// 255 and the continuation chain encodes 4294967040.
    #[test]
    fn round_trip_u32_max_with_8_bit_prefix() {
        let mut buf = Vec::new();
        encode_integer(u32::MAX, 8, 0, &mut buf);
        let (decoded, consumed) = decode_integer(&buf, 8).unwrap();
        assert_eq!(decoded, u32::MAX);
        assert_eq!(consumed, buf.len());
    }

    /// Decoder must reject inputs whose continuation chain doesn't
    /// terminate within the supplied buffer.
    #[test]
    fn truncated_continuation_chain_is_typed_error() {
        // 5-bit prefix marker (31) followed by a continuation byte
        // with high bit set (more follows) — but no more bytes.
        let outcome = decode_integer(&[0b0001_1111, 0b1000_0000], 5);
        assert!(matches!(outcome, Err(HpackError::IntegerTruncated)));
    }

    /// Empty input is also a truncation error.
    #[test]
    fn empty_buffer_is_truncated_error() {
        let outcome = decode_integer(&[], 5);
        assert!(matches!(outcome, Err(HpackError::IntegerTruncated)));
    }

    /// Decoder must reject integers whose continuation chain would
    /// overflow u32.
    #[test]
    fn overflow_continuation_chain_is_typed_error() {
        // Construct a chain that would produce ~u64::MAX. After the
        // 5-bit prefix marker (=31), feed 10 continuation bytes all
        // with high bit set and value 0x7f. Each contributes 0x7f
        // shifted by 7*i bits.
        let mut wire = vec![0b0001_1111];
        wire.extend(core::iter::repeat_n(0b1111_1111_u8, 10));
        // Final byte without continuation flag.
        wire.push(0b0111_1111);
        let outcome = decode_integer(&wire, 5);
        assert!(matches!(outcome, Err(HpackError::IntegerOverflow)));
    }

    /// Invalid prefix_bits surfaces as a typed error from decode_integer
    /// (encode panics in debug).
    #[test]
    fn invalid_prefix_bits_is_typed_error_on_decode() {
        let outcome = decode_integer(&[0x00], 0);
        assert!(matches!(
            outcome,
            Err(HpackError::InvalidPrefixBits { got: 0 })
        ));
        let outcome = decode_integer(&[0x00], 9);
        assert!(matches!(
            outcome,
            Err(HpackError::InvalidPrefixBits { got: 9 })
        ));
    }

    /// Decoder returns `consumed` exactly equal to the number of
    /// bytes it read — important for the HPACK block parser which
    /// advances its cursor by `consumed`.
    #[test]
    fn consumed_count_matches_encoded_length() {
        for value in [0_u32, 1, 31, 128, 16_384, 1_000_000, u32::MAX - 1] {
            for prefix_bits in 1_u8..=8 {
                let mut buf = Vec::new();
                encode_integer(value, prefix_bits, 0, &mut buf);
                let (_, consumed) = decode_integer(&buf, prefix_bits).unwrap();
                assert_eq!(
                    consumed,
                    buf.len(),
                    "consumed mismatch @ prefix={prefix_bits} value={value}"
                );
            }
        }
    }
}
