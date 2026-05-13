//! HPACK header block encoder (RFC 7541 §6).
//!
//! Inverse of [`super::decoder`]. For each `(name, value)` header,
//! picks the smallest representation:
//!
//! 1. Full match in static or dynamic table → Indexed Header Field (§6.1).
//!    Typical case for `:method GET`, `:status 200`, etc. — one byte.
//! 2. Name match → Literal with Incremental Indexing + indexed name (§6.2.1),
//!    OR Literal Without Indexing for sensitive headers (§6.2.2).
//! 3. No match → fully literal (name + value as strings).
//!
//! String literals emit huffman-encoded payload iff it's shorter
//! than raw; the H bit in the length prefix signals which.
//!
//! ## Sensitive headers
//!
//! `set-cookie`, `authorization`, `cookie`, and `:path` are not
//! inserted into the dynamic table. Same defaults h2 ships — keeps
//! the dynamic table from filling up with per-request unique values
//! (`:path`) or session-scoped secrets (`set-cookie`, `authorization`).

use bytes::{BufMut, Bytes, BytesMut};

use crate::hpack::{
    DynamicEntry, DynamicTable, STATIC_TABLE_LAST_INDEX, encode_integer, huffman_encode,
    huffman_encoded_len, static_lookup,
};

/// Encode a sequence of headers into `dst`, mutating `dynamic` for
/// incremental-indexing entries.
pub fn encode<I>(headers: I, dynamic: &mut DynamicTable, dst: &mut BytesMut)
where
    I: IntoIterator<Item = (Bytes, Bytes)>,
{
    for (name, value) in headers {
        encode_header(name.as_ref(), value.as_ref(), dynamic, dst);
    }
}

fn encode_header(name: &[u8], value: &[u8], dynamic: &mut DynamicTable, dst: &mut BytesMut) {
    if let Some((index, matched)) = static_lookup(name, value) {
        if matched {
            encode_integer(u32::from(index), 7, 0x80, dst);
            return;
        }
        emit_with_name_index(u32::from(index), name, value, dynamic, dst);
        return;
    }
    if let Some((dynamic_index, matched)) = dynamic.lookup(name, value) {
        let absolute = absolute_index(dynamic_index);
        if matched {
            encode_integer(absolute, 7, 0x80, dst);
            return;
        }
        emit_with_name_index(absolute, name, value, dynamic, dst);
        return;
    }
    emit_fully_literal(name, value, dynamic, dst);
}

fn emit_with_name_index(
    name_index: u32,
    name: &[u8],
    value: &[u8],
    dynamic: &mut DynamicTable,
    dst: &mut BytesMut,
) {
    if should_index(name) {
        encode_integer(name_index, 6, 0x40, dst);
        encode_string(value, dst);
        dynamic.insert(DynamicEntry::new(
            Bytes::copy_from_slice(name),
            Bytes::copy_from_slice(value),
        ));
    } else {
        encode_integer(name_index, 4, 0x00, dst);
        encode_string(value, dst);
    }
}

fn emit_fully_literal(name: &[u8], value: &[u8], dynamic: &mut DynamicTable, dst: &mut BytesMut) {
    if should_index(name) {
        encode_integer(0, 6, 0x40, dst);
        encode_string(name, dst);
        encode_string(value, dst);
        dynamic.insert(DynamicEntry::new(
            Bytes::copy_from_slice(name),
            Bytes::copy_from_slice(value),
        ));
    } else {
        encode_integer(0, 4, 0x00, dst);
        encode_string(name, dst);
        encode_string(value, dst);
    }
}

fn encode_string(input: &[u8], dst: &mut BytesMut) {
    let huff_len = huffman_encoded_len(input);
    if huff_len < input.len() {
        encode_integer(huff_len as u32, 7, 0x80, dst);
        huffman_encode(input, dst);
    } else {
        encode_integer(input.len() as u32, 7, 0, dst);
        dst.put_slice(input);
    }
}

fn absolute_index(dynamic_index: usize) -> u32 {
    (STATIC_TABLE_LAST_INDEX + dynamic_index) as u32
}

fn should_index(name: &[u8]) -> bool {
    !matches!(
        name,
        b"set-cookie" | b"authorization" | b"cookie" | b":path"
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::hpack::{DynamicTable, decode_block};
    use alloc::vec;
    use alloc::vec::Vec;

    fn roundtrip(headers: Vec<(Bytes, Bytes)>) -> Vec<(Bytes, Bytes)> {
        let mut encoded = BytesMut::new();
        let mut encode_table = DynamicTable::new(4096);
        encode(headers.clone(), &mut encode_table, &mut encoded);
        let block = encoded.freeze();
        let mut decode_table = DynamicTable::new(4096);
        let mut decoded = Vec::new();
        decode_block(&block, &mut decode_table, 4096, |name, value| {
            decoded.push((name, value));
        })
        .expect("decode");
        decoded
    }

    fn h(name: &'static [u8], value: &'static [u8]) -> (Bytes, Bytes) {
        (Bytes::from_static(name), Bytes::from_static(value))
    }

    #[test]
    fn indexed_static_full_match_is_one_byte() {
        let mut dst = BytesMut::new();
        let mut table = DynamicTable::new(4096);
        encode(vec![h(b":method", b"GET")], &mut table, &mut dst);
        assert_eq!(&dst[..], &[0x82]);
    }

    #[test]
    fn indexed_static_full_match_status_200() {
        let mut dst = BytesMut::new();
        let mut table = DynamicTable::new(4096);
        encode(vec![h(b":status", b"200")], &mut table, &mut dst);
        assert_eq!(&dst[..], &[0x88]);
    }

    #[test]
    fn name_indexed_value_literal_inserts_into_dynamic() {
        let mut dst = BytesMut::new();
        let mut table = DynamicTable::new(4096);
        encode(vec![h(b":authority", b"example.com")], &mut table, &mut dst);
        assert_eq!(
            table.len(),
            1,
            "indexed name + literal value -> incremental"
        );
        let stored = table.get(1).unwrap();
        assert_eq!(stored.name.as_ref(), b":authority");
        assert_eq!(stored.value.as_ref(), b"example.com");
    }

    #[test]
    fn sensitive_header_not_indexed() {
        let mut dst = BytesMut::new();
        let mut table = DynamicTable::new(4096);
        encode(
            vec![h(b":path", b"/v1/users/42/posts")],
            &mut table,
            &mut dst,
        );
        assert_eq!(table.len(), 0, ":path is excluded from indexing");
    }

    #[test]
    fn set_cookie_not_indexed() {
        let mut dst = BytesMut::new();
        let mut table = DynamicTable::new(4096);
        encode(vec![h(b"set-cookie", b"session=abc")], &mut table, &mut dst);
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn fully_literal_name_and_value_indexed_by_default() {
        let mut dst = BytesMut::new();
        let mut table = DynamicTable::new(4096);
        encode(vec![h(b"x-custom", b"abc")], &mut table, &mut dst);
        assert_eq!(table.len(), 1);
        let stored = table.get(1).unwrap();
        assert_eq!(stored.name.as_ref(), b"x-custom");
        assert_eq!(stored.value.as_ref(), b"abc");
    }

    #[test]
    fn roundtrip_first_request_rfc_c_3_1() {
        let input = vec![
            h(b":method", b"GET"),
            h(b":scheme", b"http"),
            h(b":path", b"/"),
            h(b":authority", b"www.example.com"),
        ];
        let decoded = roundtrip(input.clone());
        assert_eq!(decoded, input);
    }

    #[test]
    fn roundtrip_mixed_indexed_and_literal() {
        let input = vec![
            h(b":method", b"POST"),
            h(b":scheme", b"https"),
            h(b":path", b"/api/v1/users"),
            h(b":authority", b"api.example.com"),
            h(b"content-type", b"application/json"),
            h(b"content-length", b"1024"),
            h(b"x-request-id", b"abc-123-def"),
            h(b"authorization", b"Bearer t0k3n"),
        ];
        let decoded = roundtrip(input.clone());
        assert_eq!(decoded, input);
    }

    #[test]
    fn dynamic_lookup_reuses_inserted_entry() {
        let mut dst = BytesMut::new();
        let mut table = DynamicTable::new(4096);
        encode(vec![h(b"x-trace-id", b"abc-123")], &mut table, &mut dst);
        let first_len = dst.len();
        encode(vec![h(b"x-trace-id", b"abc-123")], &mut table, &mut dst);
        let second_len = dst.len() - first_len;
        assert!(
            second_len < first_len,
            "second emit reused dynamic entry, first={first_len} second={second_len}"
        );
    }

    #[test]
    fn huffman_used_when_shorter() {
        let mut dst = BytesMut::new();
        let mut table = DynamicTable::new(4096);
        encode(
            vec![h(b":authority", b"www.example.com")],
            &mut table,
            &mut dst,
        );
        // Literal-with-incremental-indexing for index 1 (:authority)
        // + huffman-encoded "www.example.com" (12 bytes < 15 raw).
        let bytes = &dst[..];
        assert_eq!(
            bytes[0], 0x41,
            "0x40 | 1 = literal w/ incremental, name idx 1"
        );
        assert_eq!(bytes[1] & 0x80, 0x80, "huffman bit set on length prefix");
    }
}
