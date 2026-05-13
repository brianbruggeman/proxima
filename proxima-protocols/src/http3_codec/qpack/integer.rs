//! QPACK / HPACK integer encoding per [RFC 7541 §5.1].
//!
//! Each integer encode is parameterized by a **prefix length `N`** (in
//! bits, range 1..=8). The first byte stores `min(value, 2^N - 1)` in
//! its low N bits; if `value` fit, encode is one byte. Otherwise the
//! low N bits are all-ones (`2^N - 1`) and the remainder (`value -
//! (2^N - 1)`) is appended as a sequence of 7-bit groups with the
//! high bit set on all but the last byte.
//!
//! Distinct from QUIC varints (RFC 9000 §16) — same crate ships both
//! since QPACK + the H3 frame codec sit side-by-side.
//!
//! [RFC 7541 §5.1]: https://www.rfc-editor.org/rfc/rfc7541#section-5.1
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). All operations are slice-based;
//! caller supplies the output buffer.

/// QPACK integer codec errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IntegerError {
    /// Prefix length `N` was outside 1..=8.
    InvalidPrefix,
    /// Input ran out mid-continuation.
    Truncated,
    /// Decoded value exceeded `u64::MAX`.
    Overflow,
    /// Output buffer too small.
    BufferTooSmall { needed: usize },
}

/// Decode one prefix-encoded integer starting at the first byte of
/// `input`. The caller supplies the prefix bit-length `N` and the
/// **leading bits** (the top `8-N` bits of `input[0]`) already
/// stripped — i.e. the first byte's value masked to `2^N - 1`.
///
/// Returns the decoded value + bytes consumed (always ≥ 1).
///
/// # Errors
///
/// See [`IntegerError`].
pub fn decode(input: &[u8], prefix_bits: u8) -> Result<(u64, usize), IntegerError> {
    if !(1..=8).contains(&prefix_bits) {
        return Err(IntegerError::InvalidPrefix);
    }
    if input.is_empty() {
        return Err(IntegerError::Truncated);
    }
    let mask: u8 = (1u16 << prefix_bits).saturating_sub(1) as u8;
    let mut value = u64::from(input[0] & mask);
    if value < u64::from(mask) {
        return Ok((value, 1));
    }
    let mut cursor = 1usize;
    let mut shift: u32 = 0;
    loop {
        if cursor >= input.len() {
            return Err(IntegerError::Truncated);
        }
        let byte = input[cursor];
        cursor += 1;
        let payload = u64::from(byte & 0x7F);
        let shifted = payload.checked_shl(shift).ok_or(IntegerError::Overflow)?;
        value = value.checked_add(shifted).ok_or(IntegerError::Overflow)?;
        if byte & 0x80 == 0 {
            return Ok((value, cursor));
        }
        shift = shift.checked_add(7).ok_or(IntegerError::Overflow)?;
        if shift >= 64 {
            return Err(IntegerError::Overflow);
        }
    }
}

/// Encode one prefix-encoded integer into `output`. Writes into
/// `output[0]`'s low `prefix_bits` bits, preserving the high
/// `8 - prefix_bits` bits as `prefix_high_bits`. Continuation bytes
/// follow at `output[1..]`.
///
/// Returns total bytes written (always ≥ 1).
///
/// # Errors
///
/// See [`IntegerError`].
pub fn encode(
    value: u64,
    prefix_bits: u8,
    prefix_high_bits: u8,
    output: &mut [u8],
) -> Result<usize, IntegerError> {
    if !(1..=8).contains(&prefix_bits) {
        return Err(IntegerError::InvalidPrefix);
    }
    if output.is_empty() {
        return Err(IntegerError::BufferTooSmall { needed: 1 });
    }
    let mask: u64 = (1u64 << prefix_bits) - 1;
    let high_mask: u8 = if prefix_bits == 8 {
        0
    } else {
        !((1u16 << prefix_bits).saturating_sub(1) as u8)
    };
    let prefix_high_bits = prefix_high_bits & high_mask;
    if value < mask {
        output[0] = prefix_high_bits | (value as u8);
        return Ok(1);
    }
    output[0] = prefix_high_bits | (mask as u8);
    let mut remaining = value - mask;
    let mut cursor = 1usize;
    while remaining >= 128 {
        if cursor >= output.len() {
            return Err(IntegerError::BufferTooSmall { needed: cursor + 2 });
        }
        output[cursor] = 0x80 | ((remaining & 0x7F) as u8);
        remaining >>= 7;
        cursor += 1;
    }
    if cursor >= output.len() {
        return Err(IntegerError::BufferTooSmall { needed: cursor + 1 });
    }
    output[cursor] = remaining as u8;
    Ok(cursor + 1)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn rfc_7541_c11_decoding_10_from_5bit_prefix() {
        // RFC 7541 §C.1.1 — value 10 in 5-bit prefix encodes to one byte
        // 0x0a. Decode strips the high 3 bits (which would be the
        // representation-type bits in HPACK).
        let bytes = [0x0Au8];
        let (value, consumed) = decode(&bytes, 5).expect("decode");
        assert_eq!(value, 10);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn rfc_7541_c12_decoding_1337_from_5bit_prefix() {
        // RFC 7541 §C.1.2 — value 1337 in 5-bit prefix encodes as
        // [0x1f, 0x9a, 0x0a] (1337 = 31 + 154 + 1024).
        let bytes = [0x1Fu8, 0x9A, 0x0A];
        let (value, consumed) = decode(&bytes, 5).expect("decode");
        assert_eq!(value, 1337);
        assert_eq!(consumed, 3);
    }

    #[test]
    fn rfc_7541_c13_decoding_42_from_8bit_prefix() {
        // RFC 7541 §C.1.3 — value 42 in 8-bit prefix encodes as 0x2A.
        let bytes = [0x2Au8];
        let (value, consumed) = decode(&bytes, 8).expect("decode");
        assert_eq!(value, 42);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn rfc_7541_c11_encode_10_in_5bit_prefix() {
        let mut buf = [0u8; 4];
        let written = encode(10, 5, 0xE0, &mut buf).expect("encode");
        assert_eq!(written, 1);
        // Low 5 bits = 10 (0x0A); high 3 bits preserved (0xE0).
        assert_eq!(buf[0], 0xEA);
    }

    #[test]
    fn rfc_7541_c12_encode_1337_in_5bit_prefix() {
        let mut buf = [0u8; 8];
        let written = encode(1337, 5, 0, &mut buf).expect("encode");
        assert_eq!(written, 3);
        assert_eq!(&buf[..written], &[0x1Fu8, 0x9A, 0x0A]);
    }

    #[test]
    fn encode_then_decode_roundtrip_sweep() {
        for value in [
            0u64, 1, 10, 30, 31, 100, 200, 1023, 1024, 16383, 65535, 1_048_576,
        ] {
            for prefix in 1u8..=8 {
                let mut buf = [0u8; 16];
                let written = encode(value, prefix, 0, &mut buf).expect("encode");
                let (decoded, consumed) = decode(&buf[..written], prefix).expect("decode");
                assert_eq!(decoded, value, "value={value} prefix={prefix}");
                assert_eq!(consumed, written);
            }
        }
    }

    #[test]
    fn invalid_prefix_rejected() {
        assert_eq!(decode(&[0], 0), Err(IntegerError::InvalidPrefix));
        assert_eq!(decode(&[0], 9), Err(IntegerError::InvalidPrefix));
        let mut buf = [0u8; 4];
        assert_eq!(encode(10, 0, 0, &mut buf), Err(IntegerError::InvalidPrefix));
    }

    #[test]
    fn truncated_continuation_rejected() {
        // 5-bit prefix value 31 means "use continuation bytes" but no
        // bytes follow.
        let bytes = [0x1Fu8];
        assert_eq!(decode(&bytes, 5), Err(IntegerError::Truncated));
    }

    #[test]
    fn encode_buffer_too_small_returns_needed() {
        let mut tiny = [];
        let err = encode(10, 5, 0, &mut tiny).unwrap_err();
        assert!(matches!(err, IntegerError::BufferTooSmall { needed: 1 }));
    }
}
