//! HPACK header block decoder (RFC 7541 §6).
//!
//! Consumes a HEADERS frame's header block fragment and emits each
//! decoded `(name, value)` pair via a callback. The dynamic table
//! is mutated in-place; callers maintain one per connection.
//!
//! ## Wire format dispatch
//!
//! Each header field representation starts with a byte whose high
//! bits identify the kind:
//! - `1xxxxxxx` — Indexed Header Field (§6.1)
//! - `01xxxxxx` — Literal with Incremental Indexing (§6.2.1)
//! - `001xxxxx` — Dynamic Table Size Update (§6.3)
//! - `0001xxxx` — Literal Never Indexed (§6.2.3)
//! - `0000xxxx` — Literal Without Indexing (§6.2.2)
//!
//! ## Zero-copy
//!
//! The decoder takes the header block as `&Bytes`. Non-huffman
//! literal names/values come out as `Bytes::slice` views — refcount
//! bump, no copy. Huffman-encoded literals must allocate since
//! the decoded bytes don't exist in the input.
//!
//! ## Two engines, deliberately NOT one (unlike QPACK's `decode_into`)
//!
//! [`decode`] (above) and [`decode_into`] (below) are BOTH first-class,
//! for a reason QPACK's decoder doesn't have: HPACK's dynamic table
//! already gives [`decode`] a cheap "owned" story — `Bytes::clone`/
//! `Bytes::slice` are O(1) refcount bumps, not copies, for anything
//! resolved from the static table, the dynamic table, or a raw
//! (non-huffman) literal in the block. A caller that needs to KEEP
//! the decoded fields (queue them past the call, as
//! `proxima-h2-codec::connection::complete_headers` does for its
//! event) is cheaper served by [`decode`]'s `Bytes`-typed callback
//! than by copying [`decode_into`]'s borrowed `&[u8]` views into
//! fresh owned buffers — a real `Bytes::copy_from_slice` per field,
//! which [`decode`] never pays for the raw/indexed cases.
//!
//! [`decode_into`] exists for the OTHER shape of caller: one that
//! only needs to observe fields inline (validate, forward, log)
//! without owning them past the call. For that caller [`decode_into`]
//! is strictly better than [`decode`] — it never touches `Bytes`
//! machinery at all for raw/indexed fields (a true `&[u8]` slice
//! into the input, not even the one-time `Bytes` shared-state
//! promotion `decode` can pay on a fresh `Vec`-backed block) and it
//! decodes Huffman literals into a caller-supplied `scratch: &mut
//! [u8]` instead of always allocating a fresh `BytesMut` per Huffman
//! string — a real win when most literals in a block aren't
//! incrementally indexed (RFC C.2.2 / C.2.3 shapes).
//!
//! Mirrors QPACK's `FieldSink` shape
//! (`proxima-h3-proto::qpack::decoder::FieldSink`) per teaching-surface
//! principle 2 — same trait shape, same "borrow first, own only at
//! a real ownership boundary" discipline — adapted to HPACK's dynamic
//! table rather than collapsed into one engine, because collapsing
//! would regress [`decode`]'s existing callers (see
//! `docs/proxima-h2/discipline.md`'s "h2 HPACK borrowing decode"
//! entry for the measured before/after).

use bytes::{Bytes, BytesMut};
use thiserror::Error;

use crate::hpack::{
    DynamicEntry, DynamicTable, STATIC_TABLE, STATIC_TABLE_LAST_INDEX, decode_integer,
    huffman_decode,
};

/// HPACK decode errors. Each maps to a HEADERS-frame-level
/// connection error per RFC 7541 §7.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    #[error("malformed integer encoding")]
    InvalidInteger,
    #[error("invalid index {0} (out of range)")]
    InvalidIndex(u32),
    #[error("truncated input")]
    Truncated,
    #[error("invalid huffman code")]
    InvalidHuffman,
    #[error("dynamic table size update {0} exceeds settings max {1}")]
    SizeUpdateOverMax(u32, usize),
    /// A Huffman literal's worst-case decoded length exceeds the
    /// caller-supplied `scratch` buffer passed to [`decode_into`].
    /// Returned BEFORE calling into the Huffman decoder — never a
    /// panic. [`decode`] never returns this variant (it always owns
    /// a fresh `BytesMut`, no caller-sized scratch involved).
    #[error(
        "huffman-decoded output exceeds scratch buffer: needed {needed}, available {available}"
    )]
    ScratchTooSmall { needed: usize, available: usize },
}

/// Decode every header field in `block`, invoking `on_header` for
/// each. Mutates `dynamic` per the wire's incremental-indexing and
/// size-update signals.
///
/// `settings_max` is the most recent `SETTINGS_HEADER_TABLE_SIZE`
/// advertised by *this* peer; a size update larger than this is a
/// protocol violation (RFC §6.3).
pub fn decode<F>(
    block: &Bytes,
    dynamic: &mut DynamicTable,
    settings_max: usize,
    mut on_header: F,
) -> Result<(), DecodeError>
where
    F: FnMut(Bytes, Bytes),
{
    let buf = block.as_ref();
    let mut cursor = 0;
    while cursor < buf.len() {
        let first = buf[cursor];
        if first & 0b1000_0000 != 0 {
            cursor = decode_indexed(block, cursor, dynamic, &mut on_header)?;
        } else if first & 0b0100_0000 != 0 {
            cursor = decode_literal(block, cursor, 6, dynamic, true, &mut on_header)?;
        } else if first & 0b0010_0000 != 0 {
            cursor = decode_size_update(block, cursor, dynamic, settings_max)?;
        } else {
            cursor = decode_literal(block, cursor, 4, dynamic, false, &mut on_header)?;
        }
    }
    Ok(())
}

fn decode_indexed<F>(
    block: &Bytes,
    cursor: usize,
    dynamic: &DynamicTable,
    on_header: &mut F,
) -> Result<usize, DecodeError>
where
    F: FnMut(Bytes, Bytes),
{
    let (index, consumed) =
        decode_integer(&block[cursor..], 7).map_err(|_| DecodeError::InvalidInteger)?;
    let (name, value) = resolve_index(index, dynamic)?;
    on_header(name, value);
    Ok(cursor + consumed)
}

fn decode_literal<F>(
    block: &Bytes,
    cursor: usize,
    prefix_bits: u8,
    dynamic: &mut DynamicTable,
    incremental: bool,
    on_header: &mut F,
) -> Result<usize, DecodeError>
where
    F: FnMut(Bytes, Bytes),
{
    let (index, consumed) =
        decode_integer(&block[cursor..], prefix_bits).map_err(|_| DecodeError::InvalidInteger)?;
    let mut next = cursor + consumed;
    let name = if index == 0 {
        let (name_bytes, name_consumed) = decode_string(block, next)?;
        next = name_consumed;
        name_bytes
    } else {
        resolve_name(index, dynamic)?
    };
    let (value, value_end) = decode_string(block, next)?;
    if incremental {
        dynamic.insert(DynamicEntry::new(name.clone(), value.clone()));
    }
    on_header(name, value);
    Ok(value_end)
}

fn decode_size_update(
    block: &Bytes,
    cursor: usize,
    dynamic: &mut DynamicTable,
    settings_max: usize,
) -> Result<usize, DecodeError> {
    let (new_max, consumed) =
        decode_integer(&block[cursor..], 5).map_err(|_| DecodeError::InvalidInteger)?;
    if (new_max as usize) > settings_max {
        return Err(DecodeError::SizeUpdateOverMax(new_max, settings_max));
    }
    dynamic.set_max_size(new_max as usize);
    Ok(cursor + consumed)
}

fn decode_string(block: &Bytes, cursor: usize) -> Result<(Bytes, usize), DecodeError> {
    let buf = block.as_ref();
    if cursor >= buf.len() {
        return Err(DecodeError::Truncated);
    }
    let huffman = buf[cursor] & 0b1000_0000 != 0;
    let (length, header_consumed) =
        decode_integer(&buf[cursor..], 7).map_err(|_| DecodeError::InvalidInteger)?;
    let payload_start = cursor + header_consumed;
    let payload_end = payload_start
        .checked_add(length as usize)
        .ok_or(DecodeError::Truncated)?;
    if payload_end > buf.len() {
        return Err(DecodeError::Truncated);
    }
    if huffman {
        let mut decoded = BytesMut::with_capacity(length as usize * 2);
        huffman_decode(&buf[payload_start..payload_end], &mut decoded)
            .map_err(|_| DecodeError::InvalidHuffman)?;
        Ok((decoded.freeze(), payload_end))
    } else {
        Ok((block.slice(payload_start..payload_end), payload_end))
    }
}

fn resolve_index(index: u32, dynamic: &DynamicTable) -> Result<(Bytes, Bytes), DecodeError> {
    if index == 0 {
        return Err(DecodeError::InvalidIndex(0));
    }
    let idx = index as usize;
    if idx <= STATIC_TABLE_LAST_INDEX {
        let (name, value) = STATIC_TABLE[idx];
        return Ok((Bytes::from_static(name), Bytes::from_static(value)));
    }
    let dynamic_index = idx - STATIC_TABLE_LAST_INDEX;
    let entry = dynamic
        .get(dynamic_index)
        .ok_or(DecodeError::InvalidIndex(index))?;
    Ok((entry.name.clone(), entry.value.clone()))
}

fn resolve_name(index: u32, dynamic: &DynamicTable) -> Result<Bytes, DecodeError> {
    Ok(resolve_index(index, dynamic)?.0)
}

/// Receives one decoded header field per call. `name`/`value` are
/// BORROWED — from the RFC 7541 Appendix A static table (`'static`),
/// from the caller's `block`, from the dynamic table (borrowed for
/// the duration of the call only), or from the caller's `scratch`
/// (Huffman output) — and do not outlive the call. See the module
/// docs for why this is a SEPARATE engine from [`decode`] rather than
/// [`decode`]'s replacement.
///
/// Implement this directly or pass a closure — the blanket `impl`
/// below covers `FnMut(&[u8], &[u8]) -> Result<(), DecodeError>`.
/// No `Box<dyn FieldSink>` — sans-IO proto crates forbid trait
/// objects (guiding-principles axiom D).
pub trait FieldSink {
    /// # Errors
    ///
    /// A sink MAY reject a field (e.g. a fixed-capacity caller-owned
    /// map is full); the error propagates out of [`decode_into`]
    /// verbatim — AFTER any dynamic-table mutation this field's wire
    /// bytes required, since table state must track the wire
    /// regardless of what the local consumer does with the value
    /// (skipping the mutation would desync this connection's table
    /// from the peer's, corrupting every later reference).
    fn field(&mut self, name: &[u8], value: &[u8]) -> Result<(), DecodeError>;
}

impl<F> FieldSink for F
where
    F: FnMut(&[u8], &[u8]) -> Result<(), DecodeError>,
{
    fn field(&mut self, name: &[u8], value: &[u8]) -> Result<(), DecodeError> {
        self(name, value)
    }
}

/// Decode one HPACK header block, streaming each decoded field to
/// `sink` instead of materialising owned `Bytes`. `scratch` backs
/// Huffman-literal output; a field's borrowed `name`/`value` is only
/// valid for the duration of that field's `sink.field(..)` call.
///
/// Use this when the caller only needs to OBSERVE fields (validate,
/// forward, log) — not keep them past the call. A caller that needs
/// to queue the decoded fields past this call (survive an event-queue
/// boundary) should use [`decode`] instead — see the module docs.
///
/// # Errors
///
/// See [`DecodeError`]; in particular
/// [`DecodeError::ScratchTooSmall`] when a Huffman literal's
/// worst-case decoded length exceeds `scratch`.
pub fn decode_into<S: FieldSink>(
    block: &Bytes,
    dynamic: &mut DynamicTable,
    settings_max: usize,
    scratch: &mut [u8],
    sink: &mut S,
) -> Result<(), DecodeError> {
    let buf = block.as_ref();
    let mut cursor = 0;
    while cursor < buf.len() {
        let first = buf[cursor];
        if first & 0b1000_0000 != 0 {
            cursor = decode_indexed_into(block, cursor, dynamic, sink)?;
        } else if first & 0b0100_0000 != 0 {
            cursor = decode_literal_into(block, cursor, 6, dynamic, true, scratch, sink)?;
        } else if first & 0b0010_0000 != 0 {
            cursor = decode_size_update(block, cursor, dynamic, settings_max)?;
        } else {
            cursor = decode_literal_into(block, cursor, 4, dynamic, false, scratch, sink)?;
        }
    }
    Ok(())
}

fn decode_indexed_into<S: FieldSink>(
    block: &Bytes,
    cursor: usize,
    dynamic: &DynamicTable,
    sink: &mut S,
) -> Result<usize, DecodeError> {
    let (index, consumed) =
        decode_integer(&block[cursor..], 7).map_err(|_| DecodeError::InvalidInteger)?;
    let (name, value) = resolve_index_borrowed(index, dynamic)?;
    sink.field(name, value)?;
    Ok(cursor + consumed)
}

/// Same resolution as [`resolve_index`] but returns borrowed views
/// instead of cloning into owned `Bytes` — the whole point of the
/// `decode_into` engine.
fn resolve_index_borrowed(
    index: u32,
    dynamic: &DynamicTable,
) -> Result<(&[u8], &[u8]), DecodeError> {
    if index == 0 {
        return Err(DecodeError::InvalidIndex(0));
    }
    let idx = index as usize;
    if idx <= STATIC_TABLE_LAST_INDEX {
        let (name, value) = STATIC_TABLE[idx];
        return Ok((name, value));
    }
    let dynamic_index = idx - STATIC_TABLE_LAST_INDEX;
    let entry = dynamic
        .get(dynamic_index)
        .ok_or(DecodeError::InvalidIndex(index))?;
    Ok((entry.name.as_ref(), entry.value.as_ref()))
}

fn resolve_name_borrowed(index: u32, dynamic: &DynamicTable) -> Result<&[u8], DecodeError> {
    Ok(resolve_index_borrowed(index, dynamic)?.0)
}

fn decode_literal_into<S: FieldSink>(
    block: &Bytes,
    cursor: usize,
    prefix_bits: u8,
    dynamic: &mut DynamicTable,
    incremental: bool,
    scratch: &mut [u8],
    sink: &mut S,
) -> Result<usize, DecodeError> {
    let buf = block.as_ref();
    let (index, consumed) =
        decode_integer(&buf[cursor..], prefix_bits).map_err(|_| DecodeError::InvalidInteger)?;
    let next = cursor + consumed;
    if index == 0 {
        decode_literal_with_literal_name(block, next, dynamic, incremental, scratch, sink)
    } else {
        decode_literal_with_indexed_name(block, next, index, dynamic, incremental, scratch, sink)
    }
}

/// A located-but-not-yet-resolved wire string: the length prefix has
/// been decoded, `start..end` is its RAW payload range within the
/// block, `huffman` says whether that payload needs decoding, `next`
/// is the cursor position immediately after it. Splitting "locate"
/// from "resolve" lets [`decode_literal_with_literal_name`] see BOTH
/// the name's and the value's Huffman flags before committing any
/// `scratch` capacity to either.
#[derive(Clone, Copy)]
struct RawString {
    huffman: bool,
    start: usize,
    end: usize,
    next: usize,
}

fn locate_string(buf: &[u8], cursor: usize) -> Result<RawString, DecodeError> {
    if cursor >= buf.len() {
        return Err(DecodeError::Truncated);
    }
    let huffman = buf[cursor] & 0b1000_0000 != 0;
    let (length, header_consumed) =
        decode_integer(&buf[cursor..], 7).map_err(|_| DecodeError::InvalidInteger)?;
    let payload_start = cursor + header_consumed;
    let payload_end = payload_start
        .checked_add(length as usize)
        .ok_or(DecodeError::Truncated)?;
    if payload_end > buf.len() {
        return Err(DecodeError::Truncated);
    }
    Ok(RawString {
        huffman,
        start: payload_start,
        end: payload_end,
        next: payload_end,
    })
}

/// Build the owned `Bytes` a dynamic-table insert needs from an
/// already-resolved view: a raw (non-huffman) literal is `block.slice`
/// (O(1) refcount bump — no copy); a Huffman literal must copy out of
/// `scratch` since `scratch` is reused by the very next field.
fn owned_copy_for_insert(block: &Bytes, raw: &RawString, view: &[u8]) -> Bytes {
    if raw.huffman {
        Bytes::copy_from_slice(view)
    } else {
        block.slice(raw.start..raw.end)
    }
}

/// RFC 7541 min Huffman code length is 5 bits, so decoded output is
/// at most `encoded_len * 8 / 5` bytes — the same bound QPACK's
/// decoder uses for its own scratch sizing.
fn huffman_worst_case_len(encoded_len: usize) -> usize {
    let bits = (encoded_len as u64).saturating_mul(8);
    usize::try_from(bits.div_ceil(5)).unwrap_or(usize::MAX)
}

fn huffman_decode_into<'scratch>(
    raw: &[u8],
    scratch: &'scratch mut [u8],
) -> Result<&'scratch [u8], DecodeError> {
    let max_len = huffman_worst_case_len(raw.len());
    if scratch.len() < max_len {
        return Err(DecodeError::ScratchTooSmall {
            needed: max_len,
            available: scratch.len(),
        });
    }
    let mut cursor: &mut [u8] = &mut *scratch;
    let written = huffman_decode(raw, &mut cursor).map_err(|_| DecodeError::InvalidHuffman)?;
    Ok(&scratch[..written])
}

/// Split `scratch` so a name AND value that are BOTH Huffman-encoded
/// can be decoded into disjoint regions and stay simultaneously alive
/// for the `sink.field(..)` call — mirrors QPACK's
/// `resolve_both_huffman`.
fn resolve_both_huffman<'scratch>(
    raw_name: &[u8],
    raw_value: &[u8],
    scratch: &'scratch mut [u8],
) -> Result<(&'scratch [u8], &'scratch [u8]), DecodeError> {
    let name_cap = huffman_worst_case_len(raw_name.len());
    if name_cap >= scratch.len() {
        return Err(DecodeError::ScratchTooSmall {
            needed: name_cap.saturating_add(huffman_worst_case_len(raw_value.len())),
            available: scratch.len(),
        });
    }
    let (name_scratch, value_scratch) = scratch.split_at_mut(name_cap);
    let name_view = huffman_decode_into(raw_name, name_scratch)?;
    let value_view = huffman_decode_into(raw_value, value_scratch)?;
    Ok((name_view, value_view))
}

/// Literal-with-literal-name (`001NHxxx` prefix=4, or the incremental
/// `01` form with a zero name-index — RFC §6.2.1/§6.2.2/§6.2.3 all
/// share this shape when the name itself isn't table-referenced).
fn decode_literal_with_literal_name<S: FieldSink>(
    block: &Bytes,
    cursor: usize,
    dynamic: &mut DynamicTable,
    incremental: bool,
    scratch: &mut [u8],
    sink: &mut S,
) -> Result<usize, DecodeError> {
    let buf = block.as_ref();
    let name_raw = locate_string(buf, cursor)?;
    let value_raw = locate_string(buf, name_raw.next)?;

    let (name_view, value_view): (&[u8], &[u8]) = match (name_raw.huffman, value_raw.huffman) {
        (false, false) => (
            &buf[name_raw.start..name_raw.end],
            &buf[value_raw.start..value_raw.end],
        ),
        (true, false) => {
            let name_view = huffman_decode_into(&buf[name_raw.start..name_raw.end], scratch)?;
            (name_view, &buf[value_raw.start..value_raw.end])
        }
        (false, true) => {
            let value_view = huffman_decode_into(&buf[value_raw.start..value_raw.end], scratch)?;
            (&buf[name_raw.start..name_raw.end], value_view)
        }
        (true, true) => resolve_both_huffman(
            &buf[name_raw.start..name_raw.end],
            &buf[value_raw.start..value_raw.end],
            scratch,
        )?,
    };

    // Table mutation happens regardless of the sink's verdict (see
    // FieldSink::field's doc) — capture the sink result and propagate
    // it AFTER the insert.
    let sink_result = sink.field(name_view, value_view);
    if incremental {
        let owned_name = owned_copy_for_insert(block, &name_raw, name_view);
        let owned_value = owned_copy_for_insert(block, &value_raw, value_view);
        dynamic.insert(DynamicEntry::new(owned_name, owned_value));
    }
    sink_result?;
    Ok(value_raw.next)
}

/// Literal-with-indexed-name (name comes from the static or dynamic
/// table; only the value is a literal on the wire).
fn decode_literal_with_indexed_name<S: FieldSink>(
    block: &Bytes,
    cursor: usize,
    index: u32,
    dynamic: &mut DynamicTable,
    incremental: bool,
    scratch: &mut [u8],
    sink: &mut S,
) -> Result<usize, DecodeError> {
    let buf = block.as_ref();
    let value_raw = locate_string(buf, cursor)?;
    let value_view: &[u8] = if value_raw.huffman {
        huffman_decode_into(&buf[value_raw.start..value_raw.end], scratch)?
    } else {
        &buf[value_raw.start..value_raw.end]
    };

    // `index` is only meaningful against the table's PRE-insertion
    // state — inserting shifts every existing dynamic index by one
    // (LIFO). Resolve the name (and, if indexing, the owned copy the
    // insert needs) BEFORE mutating the table.
    let name_view = resolve_name_borrowed(index, dynamic)?;
    let sink_result = sink.field(name_view, value_view);
    if incremental {
        let owned_name = resolve_name(index, dynamic)?;
        let owned_value = owned_copy_for_insert(block, &value_raw, value_view);
        dynamic.insert(DynamicEntry::new(owned_name, owned_value));
    }
    sink_result?;
    Ok(value_raw.next)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::hpack::ENTRY_OVERHEAD;
    use alloc::vec;
    use alloc::vec::Vec;

    fn collect(block: &Bytes, dynamic: &mut DynamicTable) -> Vec<(Bytes, Bytes)> {
        let mut out = Vec::new();
        decode(block, dynamic, 4096, |name, value| {
            out.push((name, value));
        })
        .expect("decode");
        out
    }

    #[test]
    fn rfc_c_2_1_literal_with_incremental_indexing() {
        let block = Bytes::from_static(&[
            0x40, 0x0a, b'c', b'u', b's', b't', b'o', b'm', b'-', b'k', b'e', b'y', 0x0d, b'c',
            b'u', b's', b't', b'o', b'm', b'-', b'h', b'e', b'a', b'd', b'e', b'r',
        ]);
        let mut dynamic = DynamicTable::new(4096);
        let headers = collect(&block, &mut dynamic);
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].0.as_ref(), b"custom-key");
        assert_eq!(headers[0].1.as_ref(), b"custom-header");
        assert_eq!(dynamic.len(), 1);
        let stored = dynamic.get(1).unwrap();
        assert_eq!(stored.name.as_ref(), b"custom-key");
        assert_eq!(stored.value.as_ref(), b"custom-header");
    }

    #[test]
    fn rfc_c_2_2_literal_without_indexing() {
        let block = Bytes::from_static(&[
            0x04, 0x0c, b'/', b's', b'a', b'm', b'p', b'l', b'e', b'/', b'p', b'a', b't', b'h',
        ]);
        let mut dynamic = DynamicTable::new(4096);
        let headers = collect(&block, &mut dynamic);
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].0.as_ref(), b":path");
        assert_eq!(headers[0].1.as_ref(), b"/sample/path");
        assert_eq!(dynamic.len(), 0, "no incremental indexing");
    }

    #[test]
    fn rfc_c_2_3_literal_never_indexed() {
        let block = Bytes::from_static(&[
            0x10, 0x08, b'p', b'a', b's', b's', b'w', b'o', b'r', b'd', 0x06, b's', b'e', b'c',
            b'r', b'e', b't',
        ]);
        let mut dynamic = DynamicTable::new(4096);
        let headers = collect(&block, &mut dynamic);
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].0.as_ref(), b"password");
        assert_eq!(headers[0].1.as_ref(), b"secret");
        assert_eq!(dynamic.len(), 0, "never indexed");
    }

    #[test]
    fn rfc_c_2_4_indexed_header_field() {
        let block = Bytes::from_static(&[0x82]);
        let mut dynamic = DynamicTable::new(4096);
        let headers = collect(&block, &mut dynamic);
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].0.as_ref(), b":method");
        assert_eq!(headers[0].1.as_ref(), b"GET");
    }

    #[test]
    fn rfc_c_3_1_first_request() {
        let block = Bytes::from_static(&[
            0x82, 0x86, 0x84, 0x41, 0x0f, b'w', b'w', b'w', b'.', b'e', b'x', b'a', b'm', b'p',
            b'l', b'e', b'.', b'c', b'o', b'm',
        ]);
        let mut dynamic = DynamicTable::new(4096);
        let headers = collect(&block, &mut dynamic);
        assert_eq!(headers.len(), 4);
        assert_eq!(
            headers[0],
            (Bytes::from_static(b":method"), Bytes::from_static(b"GET"))
        );
        assert_eq!(
            headers[1],
            (Bytes::from_static(b":scheme"), Bytes::from_static(b"http"))
        );
        assert_eq!(
            headers[2],
            (Bytes::from_static(b":path"), Bytes::from_static(b"/"))
        );
        assert_eq!(headers[3].0.as_ref(), b":authority");
        assert_eq!(headers[3].1.as_ref(), b"www.example.com");
        assert_eq!(dynamic.len(), 1);
        let cached = dynamic.get(1).unwrap();
        assert_eq!(cached.name.as_ref(), b":authority");
        assert_eq!(cached.value.as_ref(), b"www.example.com");
    }

    #[test]
    fn rfc_c_4_1_huffman_encoded_authority() {
        let block = Bytes::from_static(&[
            0x82, 0x86, 0x84, 0x41, 0x8c, 0xf1, 0xe3, 0xc2, 0xe5, 0xf2, 0x3a, 0x6b, 0xa0, 0xab,
            0x90, 0xf4, 0xff,
        ]);
        let mut dynamic = DynamicTable::new(4096);
        let headers = collect(&block, &mut dynamic);
        assert_eq!(headers.len(), 4);
        assert_eq!(headers[3].0.as_ref(), b":authority");
        assert_eq!(headers[3].1.as_ref(), b"www.example.com");
    }

    #[test]
    fn dynamic_size_update_evicts() {
        let insert = Bytes::from_static(&[
            0x40, 0x05, b'x', b'-', b'b', b'i', b'g', 0x07, b'p', b'a', b'y', b'l', b'o', b'a',
            b'd',
        ]);
        let mut dynamic = DynamicTable::new(4096);
        let _ = collect(&insert, &mut dynamic);
        assert_eq!(dynamic.len(), 1);
        assert_eq!(dynamic.size(), 5 + 7 + ENTRY_OVERHEAD);

        let update_zero = Bytes::from_static(&[0x20]);
        decode(&update_zero, &mut dynamic, 4096, |_, _| {}).unwrap();
        assert_eq!(dynamic.len(), 0);
        assert_eq!(dynamic.max_size(), 0);
    }

    #[test]
    fn size_update_over_settings_max_errors() {
        // 0x3f 0xe1 0x1f → 4096; settings_max=1024 → 4096 > 1024
        let block = Bytes::from_static(&[0x3f, 0xe1, 0x1f]);
        let mut dynamic = DynamicTable::new(4096);
        let err = decode(&block, &mut dynamic, 1024, |_, _| {}).unwrap_err();
        assert!(matches!(err, DecodeError::SizeUpdateOverMax(_, _)));
    }

    #[test]
    fn truncated_payload_errors() {
        let block = Bytes::from_static(&[0x40, 0x0a, b'a', b'b', b'c']);
        let mut dynamic = DynamicTable::new(4096);
        let err = decode(&block, &mut dynamic, 4096, |_, _| {}).unwrap_err();
        assert_eq!(err, DecodeError::Truncated);
    }

    #[test]
    fn invalid_index_zero_errors() {
        let block = Bytes::from_static(&[0x80]);
        let mut dynamic = DynamicTable::new(4096);
        let err = decode(&block, &mut dynamic, 4096, |_, _| {}).unwrap_err();
        assert_eq!(err, DecodeError::InvalidIndex(0));
    }

    type OwnedFieldPairs = Vec<(Vec<u8>, Vec<u8>)>;

    /// Collects every [`FieldSink::field`] call into owned pairs — used
    /// by tests that want to inspect [`decode_into`]'s output without
    /// reaching for an alloc-counting probe.
    fn collect_into(
        block: &Bytes,
        dynamic: &mut DynamicTable,
        settings_max: usize,
        scratch: &mut [u8],
    ) -> Result<OwnedFieldPairs, DecodeError> {
        let mut collected: OwnedFieldPairs = Vec::new();
        let mut sink = |name: &[u8], value: &[u8]| -> Result<(), DecodeError> {
            collected.push((name.to_vec(), value.to_vec()));
            Ok(())
        };
        decode_into(block, dynamic, settings_max, scratch, &mut sink)?;
        Ok(collected)
    }

    #[test]
    fn decode_into_rfc_c_2_1_literal_with_incremental_indexing() {
        let block = Bytes::from_static(&[
            0x40, 0x0a, b'c', b'u', b's', b't', b'o', b'm', b'-', b'k', b'e', b'y', 0x0d, b'c',
            b'u', b's', b't', b'o', b'm', b'-', b'h', b'e', b'a', b'd', b'e', b'r',
        ]);
        let mut dynamic = DynamicTable::new(4096);
        let mut scratch = [0u8; 64];
        let headers = collect_into(&block, &mut dynamic, 4096, &mut scratch).expect("decode_into");
        assert_eq!(
            headers,
            vec![(b"custom-key".to_vec(), b"custom-header".to_vec())]
        );
        assert_eq!(
            dynamic.len(),
            1,
            "incremental indexing inserts into the table"
        );
        let stored = dynamic.get(1).expect("entry");
        assert_eq!(stored.name.as_ref(), b"custom-key");
        assert_eq!(stored.value.as_ref(), b"custom-header");
    }

    #[test]
    fn decode_into_rfc_c_2_2_literal_without_indexing() {
        let block = Bytes::from_static(&[
            0x04, 0x0c, b'/', b's', b'a', b'm', b'p', b'l', b'e', b'/', b'p', b'a', b't', b'h',
        ]);
        let mut dynamic = DynamicTable::new(4096);
        let mut scratch = [0u8; 64];
        let headers = collect_into(&block, &mut dynamic, 4096, &mut scratch).expect("decode_into");
        assert_eq!(headers, vec![(b":path".to_vec(), b"/sample/path".to_vec())]);
        assert_eq!(dynamic.len(), 0, "no incremental indexing");
    }

    #[test]
    fn decode_into_rfc_c_2_3_literal_never_indexed() {
        let block = Bytes::from_static(&[
            0x10, 0x08, b'p', b'a', b's', b's', b'w', b'o', b'r', b'd', 0x06, b's', b'e', b'c',
            b'r', b'e', b't',
        ]);
        let mut dynamic = DynamicTable::new(4096);
        let mut scratch = [0u8; 64];
        let headers = collect_into(&block, &mut dynamic, 4096, &mut scratch).expect("decode_into");
        assert_eq!(headers, vec![(b"password".to_vec(), b"secret".to_vec())]);
        assert_eq!(dynamic.len(), 0, "never indexed");
    }

    #[test]
    fn decode_into_rfc_c_2_4_indexed_header_field() {
        let block = Bytes::from_static(&[0x82]);
        let mut dynamic = DynamicTable::new(4096);
        let mut scratch = [0u8; 64];
        let headers = collect_into(&block, &mut dynamic, 4096, &mut scratch).expect("decode_into");
        assert_eq!(headers, vec![(b":method".to_vec(), b"GET".to_vec())]);
    }

    #[test]
    fn decode_into_rfc_c_3_1_first_request() {
        let block = Bytes::from_static(&[
            0x82, 0x86, 0x84, 0x41, 0x0f, b'w', b'w', b'w', b'.', b'e', b'x', b'a', b'm', b'p',
            b'l', b'e', b'.', b'c', b'o', b'm',
        ]);
        let mut dynamic = DynamicTable::new(4096);
        let mut scratch = [0u8; 64];
        let headers = collect_into(&block, &mut dynamic, 4096, &mut scratch).expect("decode_into");
        assert_eq!(
            headers,
            vec![
                (b":method".to_vec(), b"GET".to_vec()),
                (b":scheme".to_vec(), b"http".to_vec()),
                (b":path".to_vec(), b"/".to_vec()),
                (b":authority".to_vec(), b"www.example.com".to_vec()),
            ]
        );
        assert_eq!(dynamic.len(), 1);
        let cached = dynamic.get(1).expect("entry");
        assert_eq!(cached.name.as_ref(), b":authority");
        assert_eq!(cached.value.as_ref(), b"www.example.com");
    }

    /// RFC 7541 §C.4.1 — indexed `:authority` name (index 1, incremental
    /// prefix=6) with a Huffman-encoded literal value.
    #[test]
    fn decode_into_rfc_c_4_1_huffman_encoded_authority() {
        let block = Bytes::from_static(&[
            0x82, 0x86, 0x84, 0x41, 0x8c, 0xf1, 0xe3, 0xc2, 0xe5, 0xf2, 0x3a, 0x6b, 0xa0, 0xab,
            0x90, 0xf4, 0xff,
        ]);
        let mut dynamic = DynamicTable::new(4096);
        let mut scratch = [0u8; 64];
        let headers = collect_into(&block, &mut dynamic, 4096, &mut scratch).expect("decode_into");
        assert_eq!(headers.len(), 4);
        assert_eq!(headers[3].0, b":authority");
        assert_eq!(headers[3].1, b"www.example.com");
        // Huffman literal WAS incrementally indexed (prefix=6, index=1
        // means the NAME is indexed but the wire byte 0x41 = 0b0100_0001
        // has the incremental-indexing bit set) — the owned copy for
        // the table must byte-match what the sink observed.
        assert_eq!(dynamic.len(), 1);
        assert_eq!(
            dynamic.get(1).expect("entry").value.as_ref(),
            b"www.example.com"
        );
    }

    /// Direct unit test of the split-scratch primitive
    /// [`resolve_both_huffman`] — exercised via `decode_into` end-to-end
    /// in [`decode_and_decode_into_agree_on_browser_shaped_request`] and
    /// friends would require hand-crafting a wire fixture with BOTH a
    /// Huffman name and Huffman value (fragile bit-twiddling principle
    /// 9 warns against); testing the primitive directly with the RFC
    /// §C.4.2 "no-cache" vector (reused for both halves — the vector's
    /// job here is exercising the split, not testing Huffman itself,
    /// which the huffman module's own RFC tests already cover) proves
    /// the two decoded views stay simultaneously valid and correct.
    #[test]
    fn resolve_both_huffman_splits_scratch_for_simultaneous_views() {
        let huffman_no_cache = [0xa8u8, 0xeb, 0x10, 0x64, 0x9c, 0xbf];
        let mut scratch = [0u8; 64];
        let (name_view, value_view) =
            resolve_both_huffman(&huffman_no_cache, &huffman_no_cache, &mut scratch)
                .expect("split-scratch decode");
        assert_eq!(name_view, b"no-cache");
        assert_eq!(value_view, b"no-cache");
    }

    #[test]
    fn resolve_both_huffman_errors_when_scratch_cannot_fit_even_the_name() {
        let huffman_no_cache = [0xa8u8, 0xeb, 0x10, 0x64, 0x9c, 0xbf];
        let mut tiny_scratch = [0u8; 4]; // needs ceil(6*8/5) = 10 for the name alone
        let err = resolve_both_huffman(&huffman_no_cache, &huffman_no_cache, &mut tiny_scratch)
            .expect_err("scratch too small for even the name half");
        assert!(matches!(err, DecodeError::ScratchTooSmall { .. }));
    }

    #[test]
    fn decode_into_dynamic_size_update_evicts() {
        let insert = Bytes::from_static(&[
            0x40, 0x05, b'x', b'-', b'b', b'i', b'g', 0x07, b'p', b'a', b'y', b'l', b'o', b'a',
            b'd',
        ]);
        let mut dynamic = DynamicTable::new(4096);
        let mut scratch = [0u8; 64];
        let _ = collect_into(&insert, &mut dynamic, 4096, &mut scratch).expect("decode_into");
        assert_eq!(dynamic.len(), 1);
        assert_eq!(dynamic.size(), 5 + 7 + ENTRY_OVERHEAD);

        let update_zero = Bytes::from_static(&[0x20]);
        let mut no_op = |_: &[u8], _: &[u8]| -> Result<(), DecodeError> { Ok(()) };
        decode_into(&update_zero, &mut dynamic, 4096, &mut scratch, &mut no_op)
            .expect("size update");
        assert_eq!(dynamic.len(), 0);
        assert_eq!(dynamic.max_size(), 0);
    }

    #[test]
    fn decode_into_rejects_truncated_input_not_panics() {
        let block = Bytes::from_static(&[0x40, 0x0a, b'a', b'b', b'c']);
        let mut dynamic = DynamicTable::new(4096);
        let mut scratch = [0u8; 64];
        let err = collect_into(&block, &mut dynamic, 4096, &mut scratch).unwrap_err();
        assert_eq!(err, DecodeError::Truncated);
    }

    #[test]
    fn decode_into_rejects_invalid_index_zero() {
        let block = Bytes::from_static(&[0x80]);
        let mut dynamic = DynamicTable::new(4096);
        let mut scratch = [0u8; 64];
        let err = collect_into(&block, &mut dynamic, 4096, &mut scratch).unwrap_err();
        assert_eq!(err, DecodeError::InvalidIndex(0));
    }

    /// Huffman output whose worst-case decoded length exceeds `scratch`
    /// must error BEFORE running the Huffman decoder — never panic or
    /// silently truncate. Same RFC C.4.2 vector as the huffman parity
    /// test above, decoded into a deliberately undersized 1-byte
    /// scratch (needs `ceil(6*8/5) = 10` bytes).
    #[test]
    fn decode_into_huffman_output_exceeding_scratch_errors_not_panics() {
        // Single field, isolated on purpose: literal with incremental
        // indexing, indexed name (`:authority`, index 1), Huffman value
        // = RFC §C.4.2 "no-cache" (6 bytes, needs ceil(6*8/5)=10 bytes
        // of scratch). A multi-field fixture would call the sink
        // successfully for the earlier (non-huffman) fields before
        // reaching this one, tripping the deliberately-panicking sink
        // below for the WRONG reason.
        let block = Bytes::from_static(&[0x41, 0x86, 0xa8, 0xeb, 0x10, 0x64, 0x9c, 0xbf]);
        let mut dynamic = DynamicTable::new(4096);
        let mut tiny_scratch = [0u8; 1];
        let mut sink = |_: &[u8], _: &[u8]| -> Result<(), DecodeError> {
            panic!("sink must not be called when scratch is too small")
        };
        let err = decode_into(&block, &mut dynamic, 4096, &mut tiny_scratch, &mut sink)
            .expect_err("scratch too small must error, not panic");
        assert!(matches!(err, DecodeError::ScratchTooSmall { .. }));
    }

    /// A hostile SETTINGS_HEADER_TABLE_SIZE size-update still gets
    /// rejected under `decode_into`, matching [`decode`]'s behavior —
    /// the dynamic-table-sizing rule doesn't change between engines.
    #[test]
    fn decode_into_size_update_over_settings_max_errors() {
        let block = Bytes::from_static(&[0x3f, 0xe1, 0x1f]);
        let mut dynamic = DynamicTable::new(4096);
        let mut scratch = [0u8; 64];
        let err = collect_into(&block, &mut dynamic, 1024, &mut scratch).unwrap_err();
        assert!(matches!(err, DecodeError::SizeUpdateOverMax(_, _)));
    }

    /// Synthesizes a browser-shaped request header set (RFC-realistic
    /// shape per guiding-principles principle 9) and asserts `decode`
    /// (owned `Bytes` callback) and `decode_into` (borrowing `FieldSink`)
    /// agree byte-for-byte on BOTH the emitted fields AND the resulting
    /// dynamic-table state — the two engines are independent codepaths,
    /// so parity is the correctness oracle, not either one alone (P14).
    fn browser_request_wire() -> Bytes {
        let mut block = alloc::vec::Vec::new();
        block.extend_from_slice(&[0x82]); // :method: GET (static index 2)
        block.extend_from_slice(&[0x86]); // :scheme: http (static index 6)
        // :path literal with incremental indexing, indexed name (index 4)
        block.extend_from_slice(&[0x44, 0x0b]);
        block.extend_from_slice(b"/index.html");
        // :authority literal with incremental indexing, indexed name (index 1)
        block.extend_from_slice(&[0x41, 0x0f]);
        block.extend_from_slice(b"www.example.com");
        // user-agent literal with incremental indexing, literal name
        block.extend_from_slice(&[0x40, 0x0a]);
        block.extend_from_slice(b"user-agent");
        block.extend_from_slice(&[0x09]);
        block.extend_from_slice(b"Mozilla/5");
        Bytes::from(block)
    }

    #[test]
    fn decode_and_decode_into_agree_on_browser_shaped_request() {
        let wire = browser_request_wire();
        let mut via_decode: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mut decode_table = DynamicTable::new(4096);
        decode(&wire, &mut decode_table, 4096, |name, value| {
            via_decode.push((name.to_vec(), value.to_vec()));
        })
        .expect("decode");

        let mut into_table = DynamicTable::new(4096);
        let mut scratch = [0u8; 256];
        let via_into =
            collect_into(&wire, &mut into_table, 4096, &mut scratch).expect("decode_into");

        assert_eq!(via_decode, via_into, "field-by-field parity");
        assert_eq!(
            decode_table.len(),
            into_table.len(),
            "dynamic table length parity"
        );
        for dynamic_index in 1..=decode_table.len() {
            let expected = decode_table.get(dynamic_index).expect("decode entry");
            let actual = into_table.get(dynamic_index).expect("decode_into entry");
            assert_eq!(
                actual.name, expected.name,
                "dynamic entry {dynamic_index} name"
            );
            assert_eq!(
                actual.value, expected.value,
                "dynamic entry {dynamic_index} value"
            );
        }
    }

    /// Static-indexed + literal-WITHOUT-indexing fields only — no
    /// dynamic-table mutation, no Huffman. Isolates `decode_into`'s
    /// core promise (true `&[u8]` borrows, no `Bytes` machinery at
    /// all) from the dynamic table's OWN allocation lifecycle (its
    /// backing `VecDeque` grows on its first insert regardless of
    /// which decode engine drives it — see
    /// `alloc_count_decode_into_incremental_indexing_pays_only_table_growth`
    /// below for that separate, honestly-attributed cost).
    #[cfg(feature = "std")]
    fn minimal_static_and_literal_no_indexing_wire() -> Bytes {
        let mut block = alloc::vec::Vec::new();
        block.extend_from_slice(&[0x82]); // :method: GET (static index 2)
        block.extend_from_slice(&[0x87]); // :scheme: https (static index 7)
        // :path literal WITHOUT indexing (prefix=4), indexed name=4, value "/"
        block.extend_from_slice(&[0x04, 0x01]);
        block.push(b'/');
        Bytes::from(block)
    }

    /// DC-H2-HPACK-DECODE-INTO-ALLOC — `decode_into` driven by a
    /// non-allocating counting sink performs 0 heap allocations when
    /// no field is Huffman-encoded and no field triggers a
    /// dynamic-table insert.
    #[cfg(feature = "std")]
    #[test]
    fn alloc_count_decode_into_zero_when_no_huffman_and_no_table_growth() {
        let wire = minimal_static_and_literal_no_indexing_wire();
        let mut dynamic = DynamicTable::new(4096);
        let mut scratch = [0u8; 256];
        // Prime the block's Bytes shared-state promotion OUTSIDE the
        // measured window — the FIRST `.slice()`/`.clone()` touch of a
        // freshly `Bytes::from(Vec<u8>)`-backed value promotes it to
        // `Arc`-refcounted storage (one `Box<Shared>` allocation);
        // every touch after that is a pure atomic-refcount bump. That
        // one-time cost is a property of `Bytes::from(Vec<u8>)`
        // construction, not of `decode_into`'s algorithm.
        let _ = wire.slice(0..1);

        // the process-global stats_alloc counter also ticks for stray
        // allocations on other runtime threads parked in-window on a loaded
        // CI runner (observed left:4); that noise is additive-only, so the MIN
        // delta across repeats is decode_into's true per-call cost. the wire is
        // non-indexing, so every iteration is idempotent (no table growth).
        let region = crate::alloc_test::exclusive_region();
        let mut min_allocations = usize::MAX;
        let mut field_count = 0usize;
        for _ in 0..8 {
            field_count = 0;
            let mut sink = |_: &[u8], _: &[u8]| -> Result<(), DecodeError> {
                field_count += 1;
                Ok(())
            };
            let before = region.change();
            decode_into(&wire, &mut dynamic, 4096, &mut scratch, &mut sink).expect("decode_into");
            let after = region.change();
            min_allocations = min_allocations.min(after.allocations - before.allocations);
        }

        assert_eq!(field_count, 3);
        assert_eq!(
            min_allocations, 0,
            "decode_into must perform 0 heap allocations with no huffman + no table growth"
        );
    }

    /// Attributes the ONE allocation a browser-shaped (mostly
    /// incrementally-indexed) request costs under `decode_into` to
    /// `DynamicTable`'s backing `VecDeque` growing from empty on its
    /// FIRST insert — a cost `decode` pays too (same `DynamicTable`
    /// type, same growth curve), NOT a `decode_into`-specific
    /// regression. Measured, not assumed (P18) — this row's number
    /// came from an initial hand-derived guess (0) that this very
    /// test caught as wrong.
    #[cfg(feature = "std")]
    #[test]
    fn alloc_count_decode_into_incremental_indexing_pays_only_table_growth() {
        let wire = browser_request_wire();
        let mut dynamic = DynamicTable::new(4096);
        let mut scratch = [0u8; 256];
        let _ = wire.slice(0..1); // one-time Bytes promotion, excluded (see above)

        let region = crate::alloc_test::exclusive_region();
        let before = region.change();
        let mut field_count = 0usize;
        let mut sink = |_: &[u8], _: &[u8]| -> Result<(), DecodeError> {
            field_count += 1;
            Ok(())
        };
        decode_into(&wire, &mut dynamic, 4096, &mut scratch, &mut sink).expect("decode_into");
        let after = region.change();

        assert_eq!(field_count, 5);
        assert_eq!(
            dynamic.len(),
            3,
            "3 of the 5 fields use incremental indexing"
        );
        assert_eq!(
            after.allocations - before.allocations,
            1,
            "the ONE allocation is DynamicTable's VecDeque growing from empty, not per-field cost"
        );
    }

    /// Contrast arm: an OWNED `Vec<(Bytes,Bytes)>` built from
    /// `decode_into`'s borrowed `&[u8]` views (the shape a caller
    /// would reach for if it tried to use `decode_into` as a
    /// drop-in replacement for `decode`) pays a REAL `Bytes::copy_from_slice`
    /// per name AND per value — on top of the SAME `DynamicTable`
    /// growth + `Vec` growth `decode` also pays. This is the
    /// documented reason `decode_into` is NOT a drop-in replacement
    /// for `decode` when the caller must own the result past the call
    /// (see the module docs) — recorded honestly, not buried.
    #[cfg(feature = "std")]
    #[test]
    fn alloc_count_owning_decode_into_wrapper_costs_more_than_decode() {
        let wire = browser_request_wire();
        let _ = wire.slice(0..1); // same one-time promotion priming as above

        let region = crate::alloc_test::exclusive_region();

        let before_decode = region.change();
        let mut decode_table = DynamicTable::new(4096);
        let mut via_decode: Vec<(Bytes, Bytes)> = Vec::with_capacity(16);
        decode(&wire, &mut decode_table, 4096, |name, value| {
            via_decode.push((name, value));
        })
        .expect("decode");
        let after_decode = region.change();
        let decode_allocs = after_decode.allocations - before_decode.allocations;

        let before_into = region.change();
        let mut into_table = DynamicTable::new(4096);
        let mut scratch = [0u8; 256];
        let mut via_into: Vec<(Bytes, Bytes)> = Vec::with_capacity(16);
        let mut sink = |name: &[u8], value: &[u8]| -> Result<(), DecodeError> {
            via_into.push((Bytes::copy_from_slice(name), Bytes::copy_from_slice(value)));
            Ok(())
        };
        decode_into(&wire, &mut into_table, 4096, &mut scratch, &mut sink).expect("decode_into");
        let after_into = region.change();
        let decode_into_wrapper_allocs = after_into.allocations - before_into.allocations;

        assert_eq!(via_decode.len(), via_into.len());
        assert!(
            decode_into_wrapper_allocs > decode_allocs,
            "owning wrapper over decode_into ({decode_into_wrapper_allocs} allocs) must cost MORE than decode ({decode_allocs} allocs) — \
             proves decode_into is the wrong tool when the caller must own the result"
        );
    }
}
