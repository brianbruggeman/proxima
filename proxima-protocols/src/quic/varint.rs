//! RFC 9000 §16 variable-length integer codec.
//!
//! Encoding: the top 2 bits of the first byte give the length class
//! (`0b00` = 1 byte / `0b01` = 2 / `0b10` = 4 / `0b11` = 8). The remaining
//! bits hold the value, big-endian unsigned. Maximum encodable value is
//! `2^62 - 1` ([`MAX_VALUE`]).
//!
//! ```text
//!   1-byte: 00xxxxxx                                            value <= 63
//!   2-byte: 01xxxxxx xxxxxxxx                                   value <= 16383
//!   4-byte: 10xxxxxx xxxxxxxx xxxxxxxx xxxxxxxx                 value <= 1073741823
//!   8-byte: 11xxxxxx xxxxxxxx xxxxxxxx xxxxxxxx
//!           xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx                 value <= MAX_VALUE
//! ```
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). Caller owns all buffers. No
//! `Vec`, no `Bytes`, no I/O traits. The functions take `&[u8]` /
//! `&mut [u8]` and return value + bytes consumed / bytes written.
//!
//! # Composability
//!
//! Mirrors `quinn-proto::VarInt` (the named incumbent) on the wire,
//! per [RFC 9000 §16](https://www.rfc-editor.org/rfc/rfc9000#section-16).
//! The shape is sans-IO so the same module can drive an embedded MCU,
//! an in-process loopback test, or the std-tier [`Endpoint`] facade.
//!
//! [`Endpoint`]: crate

/// Largest representable value, `2^62 - 1`.
pub const MAX_VALUE: u64 = (1u64 << 62) - 1;

/// Largest encoded length, in bytes.
pub const MAX_ENCODED_LEN: usize = 8;

/// Encode failure modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// Value exceeds [`MAX_VALUE`].
    ValueTooLarge,
    /// Output buffer is smaller than [`encoded_len`] of the value.
    BufferTooSmall,
}

/// Decode failure modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    /// Input buffer is empty.
    Empty,
    /// Input buffer was truncated before the encoded value completed.
    Truncated,
}

/// Bytes needed to encode `value`. Returns `8` for any input that
/// exceeds [`MAX_VALUE`]; callers that require strict validation should
/// pair this with [`encode`], which returns [`EncodeError::ValueTooLarge`].
#[must_use]
pub const fn encoded_len(value: u64) -> usize {
    if value < (1 << 6) {
        1
    } else if value < (1 << 14) {
        2
    } else if value < (1 << 30) {
        4
    } else {
        8
    }
}

/// Encode `value` into `output`. Writes the canonical (shortest) form per
/// RFC 9000 §16. Returns the number of bytes written.
///
/// # Errors
///
/// - [`EncodeError::ValueTooLarge`] if `value > MAX_VALUE`.
/// - [`EncodeError::BufferTooSmall`] if `output.len() < encoded_len(value)`.
#[inline]
pub fn encode(value: u64, output: &mut [u8]) -> Result<usize, EncodeError> {
    if value > MAX_VALUE {
        return Err(EncodeError::ValueTooLarge);
    }
    let len = encoded_len(value);
    if output.len() < len {
        return Err(EncodeError::BufferTooSmall);
    }
    match len {
        1 => {
            output[0] = value as u8;
        }
        2 => {
            let tagged = (value as u16) | (0b01 << 14);
            output[..2].copy_from_slice(&tagged.to_be_bytes());
        }
        4 => {
            let tagged = (value as u32) | (0b10 << 30);
            output[..4].copy_from_slice(&tagged.to_be_bytes());
        }
        8 => {
            let tagged = value | (0b11 << 62);
            output[..8].copy_from_slice(&tagged.to_be_bytes());
        }
        // encoded_len returns only 1/2/4/8 for values <= MAX_VALUE,
        // and the ValueTooLarge gate above bars everything else.
        _ => return Err(EncodeError::ValueTooLarge),
    }
    Ok(len)
}

/// Decode a varint from `input`. Returns the value and the number of
/// bytes consumed. The decoder accepts any valid encoded form — including
/// non-canonical ones, e.g. value `37` encoded as `[0x40, 0x25]` (2-byte
/// long form) as well as `[0x25]` (1-byte canonical form). Per
/// [RFC 9000 §16](https://www.rfc-editor.org/rfc/rfc9000#section-16),
/// the long form is legal on the wire.
///
/// # Errors
///
/// - [`DecodeError::Empty`] if `input` is zero-length.
/// - [`DecodeError::Truncated`] if `input.len() < encoded_len(prefix)`.
#[inline]
pub fn decode(input: &[u8]) -> Result<(u64, usize), DecodeError> {
    let Some(&first) = input.first() else {
        return Err(DecodeError::Empty);
    };
    // dispatching on the 2-bit length tag (not on decoded length) gives
    // the compiler a fully exhaustive 4-arm match instead of a 4-arm
    // match-plus-fallback over `usize`; saves a branch on the hot path.
    let tag = first >> 6;
    match tag {
        0b00 => Ok((u64::from(first & 0b0011_1111), 1)),
        0b01 => {
            if input.len() < 2 {
                return Err(DecodeError::Truncated);
            }
            // a single bulk read + single mask beats per-byte indexing;
            // the compiler elides the bounds check on the slice-to-array
            // copy because the length is already proven >= 2 above.
            let mut bytes = [0u8; 2];
            bytes.copy_from_slice(&input[..2]);
            bytes[0] &= 0b0011_1111;
            Ok((u64::from(u16::from_be_bytes(bytes)), 2))
        }
        0b10 => {
            if input.len() < 4 {
                return Err(DecodeError::Truncated);
            }
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&input[..4]);
            bytes[0] &= 0b0011_1111;
            Ok((u64::from(u32::from_be_bytes(bytes)), 4))
        }
        // 0b11 by exhaustion of the 2-bit tag.
        _ => {
            if input.len() < 8 {
                return Err(DecodeError::Truncated);
            }
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&input[..8]);
            bytes[0] &= 0b0011_1111;
            Ok((u64::from_be_bytes(bytes), 8))
        }
    }
}

/// Total encoded length implied by the top 2 bits of the first byte.
/// Returns 1, 2, 4, or 8.
#[inline]
#[must_use]
pub const fn decoded_len_from_prefix(first_byte: u8) -> usize {
    // top 2 bits select 1 / 2 / 4 / 8 — table lookup keeps it branchless.
    const LOOKUP: [usize; 4] = [1, 2, 4, 8];
    LOOKUP[(first_byte >> 6) as usize]
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // RFC 9000 Appendix A.1 — canonical test vectors.
    // (encoded_bytes, decoded_value, encoded_len)
    const RFC_VECTORS: &[(&[u8], u64, usize)] = &[
        // 8-byte form, value 151288809941952652
        (
            &[0xc2, 0x19, 0x7c, 0x5e, 0xff, 0x14, 0xe8, 0x8c],
            151_288_809_941_952_652,
            8,
        ),
        // 4-byte form, value 494878333
        (&[0x9d, 0x7f, 0x3e, 0x7d], 494_878_333, 4),
        // 2-byte form, value 15293
        (&[0x7b, 0xbd], 15_293, 2),
        // 1-byte form, value 37
        (&[0x25], 37, 1),
    ];

    // RFC 9000 §A.1 also notes the 2-byte long form of value 37
    // ([0x40, 0x25]) is legal on the wire.
    const RFC_NONCANONICAL_VECTOR: (&[u8], u64, usize) = (&[0x40, 0x25], 37, 2);

    #[test]
    fn rfc_9000_appendix_a1_vectors_decode() {
        for &(bytes, value, len) in RFC_VECTORS {
            let (decoded_value, decoded_len) = decode(bytes).expect("RFC vector must decode");
            assert_eq!(decoded_value, value, "value mismatch for {bytes:?}");
            assert_eq!(decoded_len, len, "length mismatch for {bytes:?}");
        }
    }

    #[test]
    fn rfc_9000_appendix_a1_vectors_encode_canonical() {
        for &(bytes, value, len) in RFC_VECTORS {
            let mut output = [0u8; 8];
            let written = encode(value, &mut output).expect("encode must succeed");
            assert_eq!(written, len, "canonical length mismatch for value {value}");
            assert_eq!(
                &output[..written],
                bytes,
                "canonical bytes mismatch for value {value}"
            );
        }
    }

    #[test]
    fn noncanonical_long_form_decodes_to_same_value() {
        let (bytes, value, len) = RFC_NONCANONICAL_VECTOR;
        let (decoded_value, decoded_len) = decode(bytes).expect("decode");
        assert_eq!(decoded_value, value);
        assert_eq!(decoded_len, len);
    }

    #[test]
    fn boundaries_round_trip() {
        // Boundaries between length classes plus the maximum value.
        let boundaries = [
            0u64,
            63,            // last 1-byte
            64,            // first 2-byte
            16_383,        // last 2-byte
            16_384,        // first 4-byte
            1_073_741_823, // last 4-byte
            1_073_741_824, // first 8-byte
            MAX_VALUE,
        ];
        for value in boundaries {
            let mut output = [0u8; MAX_ENCODED_LEN];
            let written = encode(value, &mut output).expect("encode");
            let (decoded, consumed) = decode(&output[..written]).expect("decode");
            assert_eq!(decoded, value, "round-trip failed for {value}");
            assert_eq!(consumed, written, "length mismatch for {value}");
        }
    }

    #[test]
    fn encoded_len_matches_class_table() {
        assert_eq!(encoded_len(0), 1);
        assert_eq!(encoded_len(63), 1);
        assert_eq!(encoded_len(64), 2);
        assert_eq!(encoded_len(16_383), 2);
        assert_eq!(encoded_len(16_384), 4);
        assert_eq!(encoded_len(1_073_741_823), 4);
        assert_eq!(encoded_len(1_073_741_824), 8);
        assert_eq!(encoded_len(MAX_VALUE), 8);
    }

    #[test]
    fn value_too_large_rejected() {
        let mut output = [0u8; MAX_ENCODED_LEN];
        let oversize = MAX_VALUE + 1;
        assert_eq!(
            encode(oversize, &mut output),
            Err(EncodeError::ValueTooLarge)
        );
        assert_eq!(
            encode(u64::MAX, &mut output),
            Err(EncodeError::ValueTooLarge)
        );
    }

    #[test]
    fn buffer_too_small_rejected() {
        let mut output_short = [0u8; 1];
        // value 64 needs 2 bytes; output is only 1.
        assert_eq!(
            encode(64, &mut output_short),
            Err(EncodeError::BufferTooSmall)
        );
        let mut output_zero = [0u8; 0];
        assert_eq!(
            encode(0, &mut output_zero),
            Err(EncodeError::BufferTooSmall)
        );
    }

    #[test]
    fn empty_input_rejected_by_decode() {
        assert_eq!(decode(&[]), Err(DecodeError::Empty));
    }

    #[test]
    fn truncated_input_rejected_by_decode() {
        // 4-byte form prefix (0b10xxxxxx) but only 2 bytes available.
        assert_eq!(
            decode(&[0x80, 0x00]),
            Err(DecodeError::Truncated),
            "4-byte form needs 4 input bytes"
        );
        // 8-byte form prefix with only 7 bytes.
        assert_eq!(
            decode(&[0xc0, 0, 0, 0, 0, 0, 0]),
            Err(DecodeError::Truncated)
        );
    }

    #[test]
    fn decoded_len_from_prefix_classifies_all_four_buckets() {
        // 0b00xxxxxx
        assert_eq!(decoded_len_from_prefix(0x00), 1);
        assert_eq!(decoded_len_from_prefix(0x3f), 1);
        // 0b01xxxxxx
        assert_eq!(decoded_len_from_prefix(0x40), 2);
        assert_eq!(decoded_len_from_prefix(0x7f), 2);
        // 0b10xxxxxx
        assert_eq!(decoded_len_from_prefix(0x80), 4);
        assert_eq!(decoded_len_from_prefix(0xbf), 4);
        // 0b11xxxxxx
        assert_eq!(decoded_len_from_prefix(0xc0), 8);
        assert_eq!(decoded_len_from_prefix(0xff), 8);
    }

    #[test]
    fn round_trip_dense_sweep() {
        // Sample one value per bit position from 0..62 plus the value
        // immediately preceding each length-class boundary. Exhaustive
        // sweep is impractical; this gives every bit width coverage.
        for bit in 0..62 {
            let value = 1u64 << bit;
            let mut output = [0u8; MAX_ENCODED_LEN];
            let written = encode(value, &mut output).expect("encode");
            let (decoded, consumed) = decode(&output[..written]).expect("decode");
            assert_eq!(decoded, value, "1<<{bit} round-trip failed");
            assert_eq!(consumed, written);
        }
    }

    #[test]
    fn property_round_trip_pseudo_random() {
        // Linear-congruential generator (sticks to no_std, no rand dep);
        // sweeps 10k values across the full 62-bit range.
        let mut state: u64 = 0xdeadbeefcafebabe;
        for _iteration in 0..10_000 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let value = state & MAX_VALUE;
            let mut output = [0u8; MAX_ENCODED_LEN];
            let written = encode(value, &mut output).expect("encode");
            let (decoded, consumed) = decode(&output[..written]).expect("decode");
            assert_eq!(decoded, value);
            assert_eq!(consumed, written);
        }
    }
}
