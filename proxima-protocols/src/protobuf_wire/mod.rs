//! Protobuf wire-format primitive (sans-IO).
//!
//! Tracked as P9b in `docs/protocol-gap/discipline.md`. The
//! protobuf wire format per the official spec: every field on the
//! wire is `tag + payload`, where `tag = (field_number << 3) | wire_type`
//! and `wire_type ∈ {0,1,2,5}` (groups 3/4 are deprecated). Both
//! `tag` and length prefixes are LEB128 varints (base-128, low-bit
//! continuation, max 10 bytes for a 64-bit value).
//!
//! This module is **schema-agnostic** — it walks bytes and yields
//! `(field_number, wire_type, payload_slice)` triples. Middleware
//! that wants to *route* or *inspect* protobuf messages by tag can
//! do so without compiling `.proto` files. Schema-aware decode
//! (typed message structs) is the caller's job and is typically
//! handled by `prost`.
//!
//! Sub-flag: `protobuf-wire` (default off).


use alloc::vec::Vec;

/// Wire types defined in the protobuf spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WireType {
    /// Varint: int32/int64/uint32/uint64/sint32/sint64/bool/enum.
    Varint = 0,
    /// 64-bit fixed: fixed64/sfixed64/double.
    I64 = 1,
    /// Length-delimited: string/bytes/embedded messages/packed repeated.
    Len = 2,
    /// 32-bit fixed: fixed32/sfixed32/float.
    I32 = 5,
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("buffer ended mid-field")]
    Short,
    #[error("varint overflowed 10 bytes")]
    VarintOverflow,
    #[error("wire type {0} is deprecated (group start/end) or undefined")]
    UnknownWireType(u8),
    #[error("declared length-delimited field of {0} bytes exceeds buffer")]
    LengthOverflow(u64),
}

impl WireType {
    #[inline]
    fn from_tag_bits(bits: u8) -> Result<Self, ParseError> {
        match bits {
            0 => Ok(Self::Varint),
            1 => Ok(Self::I64),
            2 => Ok(Self::Len),
            5 => Ok(Self::I32),
            other => Err(ParseError::UnknownWireType(other)),
        }
    }
}

/// Decode one varint (LEB128, low-bit continuation). Returns the
/// value and number of bytes consumed. Max 10 bytes for a 64-bit
/// value.
///
/// Fast path: if the first byte is `< 0x80` (single-byte varint —
/// covers all values 0..=127, including most tag values and small
/// integers), return without entering the loop. Mirrors `prost`'s
/// approach; the 1-byte case is the hot one in practice.
#[inline]
pub fn decode_varint(buf: &[u8]) -> Result<(u64, usize), ParseError> {
    if buf.is_empty() {
        return Err(ParseError::Short);
    }
    let first = buf[0];
    if first < 0x80 {
        return Ok((u64::from(first), 1));
    }
    decode_varint_multi(buf)
}

#[inline(never)]
fn decode_varint_multi(buf: &[u8]) -> Result<(u64, usize), ParseError> {
    let mut value: u64 = u64::from(buf[0] & 0x7F);
    let mut shift: u32 = 7;
    let mut cursor: usize = 1;
    while cursor < buf.len() && cursor < 10 {
        let byte = buf[cursor];
        cursor += 1;
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, cursor));
        }
        shift += 7;
    }
    if cursor == buf.len() && cursor < 10 {
        Err(ParseError::Short)
    } else {
        Err(ParseError::VarintOverflow)
    }
}

/// Encode `value` as a varint and append to `dest`.
#[inline]
pub fn encode_varint(mut value: u64, dest: &mut Vec<u8>) {
    while value >= 0x80 {
        dest.push((value as u8) | 0x80);
        value >>= 7;
    }
    dest.push(value as u8);
}

/// Decode one field tag. Returns `(field_number, wire_type)` and the
/// number of bytes consumed. Field numbers above 2^29 are illegal per
/// spec but this primitive doesn't enforce — middleware that needs
/// strict validation can check separately.
#[inline]
pub fn decode_tag(buf: &[u8]) -> Result<(u32, WireType, usize), ParseError> {
    let (raw, used) = decode_varint(buf)?;
    let wire = WireType::from_tag_bits((raw & 0x07) as u8)?;
    let field = (raw >> 3) as u32;
    Ok((field, wire, used))
}

/// One protobuf wire-format field. Payload is borrowed from the
/// source buffer for `Len` fields; the scalar variants carry the
/// raw bytes pre-decoded for `I32`/`I64` and the varint value for
/// `Varint`.
#[derive(Debug, Clone, Copy)]
pub enum Field<'a> {
    Varint { field: u32, value: u64 },
    I64 { field: u32, value: [u8; 8] },
    Len { field: u32, payload: &'a [u8] },
    I32 { field: u32, value: [u8; 4] },
}

impl Field<'_> {
    pub fn field_number(&self) -> u32 {
        match self {
            Field::Varint { field, .. }
            | Field::I64 { field, .. }
            | Field::Len { field, .. }
            | Field::I32 { field, .. } => *field,
        }
    }
}

/// Read one full field from the start of `buf`. Returns the field
/// plus bytes consumed. Use [`Fields`] for a full message walk.
#[inline]
pub fn parse_field(buf: &[u8]) -> Result<(Field<'_>, usize), ParseError> {
    let (field, wire, tag_used) = decode_tag(buf)?;
    let rest = &buf[tag_used..];
    match wire {
        WireType::Varint => {
            let (value, used) = decode_varint(rest)?;
            Ok((Field::Varint { field, value }, tag_used + used))
        }
        WireType::I64 => {
            if rest.len() < 8 {
                return Err(ParseError::Short);
            }
            let mut value = [0u8; 8];
            value.copy_from_slice(&rest[..8]);
            Ok((Field::I64 { field, value }, tag_used + 8))
        }
        WireType::Len => {
            let (len, len_used) = decode_varint(rest)?;
            let payload_start = tag_used + len_used;
            let len_usize = usize::try_from(len).map_err(|_| ParseError::LengthOverflow(len))?;
            let payload_end = payload_start
                .checked_add(len_usize)
                .ok_or(ParseError::LengthOverflow(len))?;
            if buf.len() < payload_end {
                return Err(ParseError::LengthOverflow(len));
            }
            Ok((
                Field::Len {
                    field,
                    payload: &buf[payload_start..payload_end],
                },
                payload_end,
            ))
        }
        WireType::I32 => {
            if rest.len() < 4 {
                return Err(ParseError::Short);
            }
            let mut value = [0u8; 4];
            value.copy_from_slice(&rest[..4]);
            Ok((Field::I32 { field, value }, tag_used + 4))
        }
    }
}

/// Iterator over fields in a protobuf message buffer. Yields each
/// field on demand; stops at end-of-buffer or first error.
pub struct Fields<'a> {
    buf: &'a [u8],
    cursor: usize,
}

impl<'a> Fields<'a> {
    #[must_use]
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, cursor: 0 }
    }
}

impl<'a> Iterator for Fields<'a> {
    type Item = Result<Field<'a>, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.buf.len() {
            return None;
        }
        match parse_field(&self.buf[self.cursor..]) {
            Ok((field, used)) => {
                self.cursor += used;
                Some(Ok(field))
            }
            Err(err) => {
                // bail iteration on error — caller decides whether
                // to ignore or propagate.
                self.cursor = self.buf.len();
                Some(Err(err))
            }
        }
    }
}

#[cfg(feature = "protobuf_wire-codec-trait")]
pub mod codec_trait;
#[cfg(feature = "protobuf_wire-codec-trait")]
pub use codec_trait::ProtobufWireCodec;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn varint_round_trip_small() {
        for value in [0u64, 1, 127, 128, 16_383, 16_384, 1_000_000] {
            let mut buf = Vec::new();
            encode_varint(value, &mut buf);
            let (decoded, used) = decode_varint(&buf).unwrap();
            assert_eq!(decoded, value);
            assert_eq!(used, buf.len());
        }
    }

    #[test]
    fn varint_max_u64() {
        let mut buf = Vec::new();
        encode_varint(u64::MAX, &mut buf);
        assert_eq!(buf.len(), 10);
        let (decoded, used) = decode_varint(&buf).unwrap();
        assert_eq!(decoded, u64::MAX);
        assert_eq!(used, 10);
    }

    #[test]
    fn varint_short_buffer_returns_short() {
        let buf = [0x80u8, 0x80]; // continuation set, truncated
        match decode_varint(&buf) {
            Err(ParseError::Short) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn varint_over_ten_bytes_returns_overflow() {
        let buf = [0x80u8; 11];
        match decode_varint(&buf) {
            Err(ParseError::VarintOverflow) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn tag_decode_packs_field_and_wire() {
        // field=5, wire=Len ⇒ tag = (5 << 3) | 2 = 42
        let buf = [42u8];
        let (field, wire, used) = decode_tag(&buf).unwrap();
        assert_eq!(field, 5);
        assert_eq!(wire, WireType::Len);
        assert_eq!(used, 1);
    }

    #[test]
    fn parse_field_varint() {
        // field 1, varint 150
        let buf = [0x08, 0x96, 0x01];
        let (field, used) = parse_field(&buf).unwrap();
        assert_eq!(used, 3);
        match field {
            Field::Varint { field, value } => {
                assert_eq!(field, 1);
                assert_eq!(value, 150);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_field_rejects_deprecated_group_wire_types() {
        for wire in [3u8, 4u8] {
            let buf = [(1 << 3) | wire];
            match parse_field(&buf) {
                Err(ParseError::UnknownWireType(found)) => assert_eq!(found, wire),
                other => panic!("unexpected for wire {wire}: {other:?}"),
            }
        }
    }

    #[test]
    fn parse_field_len_borrows_payload() {
        // field 2, len-delimited, "hello"
        let mut buf = vec![(2 << 3) | 2, 5];
        buf.extend_from_slice(b"hello");
        let (field, used) = parse_field(&buf).unwrap();
        assert_eq!(used, buf.len());
        match field {
            Field::Len { field, payload } => {
                assert_eq!(field, 2);
                assert_eq!(payload, b"hello");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn len_payload_is_not_recursively_decoded() {
        // Outer field 2 is length-delimited. Its payload starts with a deprecated
        // group-start tag, but the wire walker must only return the borrowed
        // payload; schema-aware recursive decoding is a caller decision.
        let buf = [(2 << 3) | 2, 1, (1 << 3) | 3];
        let (field, used) = parse_field(&buf).unwrap();
        assert_eq!(used, buf.len());
        match field {
            Field::Len { field, payload } => {
                assert_eq!(field, 2);
                assert_eq!(payload, &[(1 << 3) | 3]);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn length_delimited_field_must_fit_buffer() {
        let buf = [(2 << 3) | 2, 10, b'h', b'i'];
        match parse_field(&buf) {
            Err(ParseError::LengthOverflow(10)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn fields_iterator_walks_message() {
        // {1: 150 (varint), 2: "hi" (len), 3: I32(0xdeadbeef)}
        let mut buf = vec![0x08, 0x96, 0x01];
        buf.extend_from_slice(&[(2 << 3) | 2, 2]);
        buf.extend_from_slice(b"hi");
        buf.extend_from_slice(&[(3 << 3) | 5, 0xef, 0xbe, 0xad, 0xde]);
        let mut iter = Fields::new(&buf);
        let first = iter.next().expect("first").expect("ok");
        assert_eq!(first.field_number(), 1);
        let second = iter.next().expect("second").expect("ok");
        assert_eq!(second.field_number(), 2);
        let third = iter.next().expect("third").expect("ok");
        assert_eq!(third.field_number(), 3);
        assert!(iter.next().is_none());
    }
}
