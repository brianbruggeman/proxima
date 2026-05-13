//! QPACK encoder per [RFC 9204 §4.5] — encoded field section emit.
//!
//! v1 scope mirrors the decoder: static-table indexed + literal-with-
//! static-name + literal-with-literal-name. No dynamic table. No
//! Huffman (literals emitted with H=0).
//!
//! [RFC 9204 §4.5]: https://www.rfc-editor.org/rfc/rfc9204#section-4.5
//!
//! # Tier
//!
//! Tier-1 (alloc). Output written into a caller-supplied
//! `&mut Vec<u8>` so the encoded section can be wrapped in an
//! HEADERS frame and pushed onto a QUIC stream send buffer.

use alloc::vec::Vec;

use super::integer;
use super::static_table;

/// Encoder errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// Integer encode failed (output buffer too small for prefix int).
    Integer(integer::IntegerError),
}

impl From<integer::IntegerError> for EncodeError {
    fn from(err: integer::IntegerError) -> Self {
        Self::Integer(err)
    }
}

/// Encode a sequence of `(name, value)` header fields into `output`.
/// Appends to `output` (does NOT clear it first); the encoded section
/// is `output[start..output.len()]` after return, where `start` is
/// the length on entry.
///
/// Uses static-table indexed references when an exact (name, value)
/// match exists, name-only references when the name matches but value
/// differs, and full literals otherwise.
///
/// # Errors
///
/// See [`EncodeError`].
pub fn encode<I>(headers: I, output: &mut Vec<u8>) -> Result<(), EncodeError>
where
    I: IntoIterator<Item = (&'static [u8], Vec<u8>)>,
{
    encode_borrowed(
        headers
            .into_iter()
            .map(|(n, v)| (n, OwnedOrSlice::Owned(v))),
        output,
    )
}

/// Same shape as [`encode`] but accepts borrowed values. Convenient
/// when the caller already owns the value bytes elsewhere.
///
/// # Errors
///
/// See [`EncodeError`].
pub fn encode_refs<'a, I>(headers: I, output: &mut Vec<u8>) -> Result<(), EncodeError>
where
    I: IntoIterator<Item = (&'a [u8], &'a [u8])>,
{
    encode_borrowed(
        headers
            .into_iter()
            .map(|(n, v)| (n, OwnedOrSlice::Borrowed(v))),
        output,
    )
}

enum OwnedOrSlice<'a> {
    Owned(Vec<u8>),
    Borrowed(&'a [u8]),
}

impl OwnedOrSlice<'_> {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(vec) => vec.as_slice(),
            Self::Borrowed(slice) => slice,
        }
    }
}

fn encode_borrowed<'a, I>(headers: I, output: &mut Vec<u8>) -> Result<(), EncodeError>
where
    I: IntoIterator<Item = (&'a [u8], OwnedOrSlice<'a>)>,
{
    // Field section prefix: Required Insert Count = 0 (8-bit prefix
    // varint) + S=0 + Delta Base = 0 (7-bit prefix varint).
    // Both encode to a single 0x00 byte each.
    output.push(0);
    output.push(0);
    let mut scratch = [0u8; 16];
    for (name, value_holder) in headers {
        let value = value_holder.as_slice();
        if let Some(index) = static_table::find_exact(name, value) {
            // Indexed Field Line — static: `1Txxxxxx` with T=1 (static).
            // 6-bit prefix; high 2 bits = 0b11.
            let written = integer::encode(index as u64, 6, 0b1100_0000, &mut scratch)?;
            output.extend_from_slice(&scratch[..written]);
            continue;
        }
        if let Some(name_index) = static_table::find_name(name) {
            // Literal with Static Name Reference: `01NTxxxx` with T=1.
            // High 4 bits = 0b0101 (N=0, T=1). 4-bit name index prefix.
            let written = integer::encode(name_index as u64, 4, 0b0101_0000, &mut scratch)?;
            output.extend_from_slice(&scratch[..written]);
            // Value: H=0 + 7-bit length + value bytes.
            let value_len_written = integer::encode(value.len() as u64, 7, 0, &mut scratch)?;
            output.extend_from_slice(&scratch[..value_len_written]);
            output.extend_from_slice(value);
            continue;
        }
        // Literal with Literal Name: `001NHxxx` with N=0, H=0.
        // High 5 bits = 0b00100. 3-bit name length prefix.
        let written = integer::encode(name.len() as u64, 3, 0b0010_0000, &mut scratch)?;
        output.extend_from_slice(&scratch[..written]);
        output.extend_from_slice(name);
        let value_len_written = integer::encode(value.len() as u64, 7, 0, &mut scratch)?;
        output.extend_from_slice(&scratch[..value_len_written]);
        output.extend_from_slice(value);
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::decoder::{DecodedField, decode};
    use super::*;
    use alloc::vec;

    fn field(name: &[u8], value: &[u8]) -> DecodedField {
        DecodedField {
            name: name.to_vec(),
            value: value.to_vec(),
        }
    }

    #[test]
    fn roundtrip_one_indexed_field() {
        // ":method: GET" is static index 17 → 1-byte encoded reference
        // (after the 2-byte prefix).
        let mut out = Vec::new();
        encode_refs([(b":method".as_slice(), b"GET".as_slice())], &mut out).unwrap();
        assert_eq!(out.len(), 3, "prefix(2) + indexed(1) = 3");
        let decoded = decode(&out).unwrap();
        assert_eq!(decoded, vec![field(b":method", b"GET")]);
    }

    #[test]
    fn roundtrip_indexed_status_200() {
        let mut out = Vec::new();
        encode_refs([(b":status".as_slice(), b"200".as_slice())], &mut out).unwrap();
        let decoded = decode(&out).unwrap();
        assert_eq!(decoded, vec![field(b":status", b"200")]);
    }

    #[test]
    fn roundtrip_literal_with_static_name() {
        // ":status: 999" — name in static table, value isn't.
        let mut out = Vec::new();
        encode_refs([(b":status".as_slice(), b"999".as_slice())], &mut out).unwrap();
        let decoded = decode(&out).unwrap();
        assert_eq!(decoded, vec![field(b":status", b"999")]);
    }

    #[test]
    fn roundtrip_literal_with_literal_name() {
        let mut out = Vec::new();
        encode_refs([(b"x-custom".as_slice(), b"value".as_slice())], &mut out).unwrap();
        let decoded = decode(&out).unwrap();
        assert_eq!(decoded, vec![field(b"x-custom", b"value")]);
    }

    #[test]
    fn roundtrip_full_request_headers() {
        let mut out = Vec::new();
        encode_refs(
            [
                (b":method".as_slice(), b"GET".as_slice()),
                (b":scheme".as_slice(), b"https".as_slice()),
                (b":authority".as_slice(), b"example.com".as_slice()),
                (b":path".as_slice(), b"/api/v1/things".as_slice()),
                (b"user-agent".as_slice(), b"proxima/0.1".as_slice()),
                (b"accept".as_slice(), b"application/json".as_slice()),
            ],
            &mut out,
        )
        .unwrap();
        let decoded = decode(&out).unwrap();
        assert_eq!(decoded.len(), 6);
        assert_eq!(decoded[0], field(b":method", b"GET"));
        assert_eq!(decoded[1], field(b":scheme", b"https"));
        assert_eq!(decoded[2], field(b":authority", b"example.com"));
        assert_eq!(decoded[3], field(b":path", b"/api/v1/things"));
        assert_eq!(decoded[4], field(b"user-agent", b"proxima/0.1"));
        assert_eq!(decoded[5], field(b"accept", b"application/json"));
    }

    #[test]
    fn roundtrip_full_response_headers() {
        let mut out = Vec::new();
        encode_refs(
            [
                (b":status".as_slice(), b"200".as_slice()),
                (b"content-type".as_slice(), b"application/json".as_slice()),
                (b"content-length".as_slice(), b"1024".as_slice()),
                (b"server".as_slice(), b"proxima/0.1".as_slice()),
            ],
            &mut out,
        )
        .unwrap();
        let decoded = decode(&out).unwrap();
        assert_eq!(decoded.len(), 4);
    }
}
