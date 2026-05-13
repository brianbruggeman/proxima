//! QPACK decoder per [RFC 9204 §4.5] — encoded field section decode.
//!
//! v1 scope: static-table indexed references + literal-with-static-name
//! references + literal-with-literal-name. Dynamic-table references
//! require post-handshake dynamic-table state (RFC 9204 §3.2); they
//! land alongside C35/C36 connection FSMs which own that state.
//!
//! Huffman-encoded literals (RFC 7541 §5.2; QPACK reuses HPACK's
//! static Huffman table per RFC 9204 §4.1.2) decode via
//! [`crate::hpack::huffman::decode`] into caller/engine-owned scratch.
//! `proxima-hpack::huffman` no longer needs a heap — its decode-state
//! tables are `const`-evaluated `.rodata` `static`s (`DC-HPACK-HUFFMAN-BOX`,
//! REDESIGNED 2026-07-01, see `docs/proxima-h2/alloc-budget.md`), and
//! the crate ships a `no-alloc` tier-3 feature exposing that module.
//! This crate's Huffman branch is STILL only reachable when this
//! crate's OWN `alloc` feature is on — a bare `no-alloc` build here
//! declines Huffman literals with [`DecodeError::HuffmanUnsupported`]
//! — RFC-permitted (either side may decline Huffman), but the
//! `proxima-hpack`-side reason this doc used to cite for that gate is
//! now stale. Wiring this crate's Huffman branch to `proxima-hpack`'s
//! tier-3 `huffman` module (dropping the `alloc` cfg on
//! `resolve_value`/`resolve_name`/`resolve_both_huffman` below) is
//! follow-up work, a separate component from this one.
//!
//! # Three surfaces, one engine (P1 RISC reuse)
//!
//! [`FieldSectionCursor`] is the engine: a per-field-resumable stepper
//! that yields one BORROWED name/value pair per [`FieldSectionCursor::next_field`]
//! call, writing Huffman output into a caller-supplied `scratch: &mut
//! [u8]` — zero heap allocations, no callback required. [`decode_into`]
//! drives the cursor to exhaustion against a caller-supplied
//! [`FieldSink`] (the visitor surface). [`decode_bounded`] (and its
//! unbounded alias [`decode`]) is a thin alloc-tier convenience wrapper
//! over `decode_into` that copies each borrowed field into an owned
//! [`DecodedField`]. There is exactly one decode engine; the visitor
//! and owned-Vec surfaces are wrappers over it, not second
//! implementations (closes `DC-H3-QPACK-DECODE-OWNS-VECS` — see
//! `docs/proxima-quic/alloc-budget.md`).
//!
//! [RFC 9204 §4.5]: https://www.rfc-editor.org/rfc/rfc9204#section-4.5
//!
//! # Tier
//!
//! [`FieldSink`] + [`decode_into`] are tier-3 (bare `no_std` + no
//! `alloc`) — the QPACK static table is const data (principle
//! `proxima.decision.quic_tier3_promotion_aspiration`), so a decode
//! that only ever borrows from `input` / `scratch` / the static table
//! needs no heap at all. [`DecodedField`], [`decode_bounded`], and
//! [`decode`] stay tier-1 (`alloc`) — they exist purely to hand the
//! caller an owned, request-lifetime-independent copy.

#[cfg(feature = "http3_codec-alloc")]
use alloc::vec::Vec;

use super::integer;
use super::static_table;

/// Decoded header field (owned). Returned by the alloc-tier
/// convenience wrapper [`decode_bounded`] / [`decode`]. The tier-3
/// engine [`decode_into`] never constructs this type — see
/// [`FieldSink`].
#[cfg(feature = "http3_codec-alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedField {
    pub name: Vec<u8>,
    pub value: Vec<u8>,
}

/// Receives one decoded header field per call. `name` and `value` are
/// BORROWED — from the RFC 9204 Appendix A static table (`'static`),
/// from the caller's `input` slice, or from the caller's `scratch`
/// slice (Huffman output) — and do not outlive the call. This is the
/// composition seam that lets [`decode_into`] stay tier-3: the engine
/// never decides how (or whether) to own the bytes.
///
/// Implement this directly (see [`decode_bounded`]'s internal
/// `VecFieldSink` for a worked example) or pass a closure — the
/// blanket `impl` below covers `FnMut(&[u8], &[u8]) -> Result<(),
/// DecodeError>` so a caller can write
/// `decode_into(input, cap, &mut scratch, &mut |name, value| { .. })`
/// without a new type. No `Box<dyn FieldSink>` — sans-IO proto crates
/// forbid trait objects (guiding-principles axiom D).
pub trait FieldSink {
    /// # Errors
    ///
    /// A sink MAY reject a field (e.g. a fixed-capacity caller-owned
    /// map is full); the error propagates out of [`decode_into`]
    /// verbatim.
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

/// Decoder errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    /// Input ended mid-field-line.
    Truncated,
    /// Integer-prefix decode failed.
    Integer(integer::IntegerError),
    /// Indexed reference points past the static-table end. Dynamic
    /// references not supported in v1.
    UnknownStaticIndex { index: u64 },
    /// Field section prefix declared a non-zero Required Insert Count
    /// but dynamic-table state isn't wired yet (v1 limitation — lands
    /// with C35/C36 connection FSMs).
    DynamicTableRequired,
    /// Encoded literal used Huffman compression and either (a) this
    /// build lacks the `alloc` feature (`proxima-hpack`'s decode
    /// tables always need a heap — see the module docs), or (b) the
    /// underlying `crate::hpack::huffman::decode` rejected the code
    /// (invalid code / EOS-in-data / bad padding per RFC 7541 §5.2).
    HuffmanUnsupported,
    /// A Huffman literal's worst-case decoded length exceeds the
    /// caller-supplied `scratch` buffer. Returned BEFORE calling into
    /// the Huffman decoder — never a panic. Raise `scratch`'s size or
    /// route this field through a caller-owned larger buffer.
    ScratchTooSmall { needed: usize, available: usize },
    /// Decoded header section exceeds the
    /// `SETTINGS_MAX_FIELD_SECTION_SIZE` advertised by this endpoint
    /// (RFC 9114 §4.1.1.3 — calculated as sum of name.len() +
    /// value.len() + 32 per field). The size is checked against the
    /// declared per-field length BEFORE decoding the value, so a
    /// hostile peer cannot force work past the cap.
    ExceedsMaxFieldSectionSize { cap: u64, observed: u64 },
    /// A downstream [`FieldSink`] consumer's OWN fixed-capacity storage was
    /// exceeded — `decode_into` itself never runs out of room (it borrows,
    /// never owns). Currently raised only by
    /// [`part_source::HeaderBlockPartSource`](super::part_source::HeaderBlockPartSource)'s
    /// sink when a field doesn't fit its inline arena or header-slot count.
    /// Purely additive to this enum — `decode_into`'s own decode logic is
    /// unchanged; this variant exists so a sink can report "I'm full"
    /// through the same error type `decode_into` already propagates,
    /// instead of a second ad hoc error type.
    #[cfg(feature = "http3_codec-part-source")]
    SinkCapacityExceeded { needed: usize, available: usize },
}

impl From<integer::IntegerError> for DecodeError {
    fn from(err: integer::IntegerError) -> Self {
        Self::Integer(err)
    }
}

/// Per-field overhead from RFC 9114 §4.1.1.3 — every header line costs
/// `name.len() + value.len() + FIELD_OVERHEAD_BYTES` against the cap.
pub const FIELD_OVERHEAD_BYTES: u64 = 32;

/// Decode one encoded field section under the
/// `SETTINGS_MAX_FIELD_SECTION_SIZE` cap that this endpoint advertised,
/// streaming each decoded field to `sink` instead of materialising a
/// `Vec`. `cap` is the **decoded** size limit per RFC 9114 §4.1.1.3 —
/// sum of `name.len() + value.len() + 32` across all fields. `scratch`
/// backs Huffman-literal output; it is reused (overwritten) per field,
/// so a field's borrowed `value`/`name` is only valid for the duration
/// of that field's `sink.field(..)` call.
///
/// Both the literal-length varint AND the running total are checked
/// **before** decoding the value, so a hostile peer cannot force work
/// past the cap.
///
/// # Errors
///
/// See [`DecodeError`]; in particular
/// [`DecodeError::ExceedsMaxFieldSectionSize`] when the limit would be
/// exceeded, and [`DecodeError::ScratchTooSmall`] when a Huffman
/// literal's worst-case output exceeds `scratch`.
pub fn decode_into<S: FieldSink>(
    input: &[u8],
    cap: u64,
    scratch: &mut [u8],
    sink: &mut S,
) -> Result<(), DecodeError> {
    let mut cursor = FieldSectionCursor::new(input, cap)?;
    while let Some((name, value)) = cursor.next_field(&mut *scratch)? {
        sink.field(name, value)?;
    }
    Ok(())
}

/// One decoded field line — `(name, value)`, both borrowed. See
/// [`FieldSectionCursor::next_field`] for the borrow provenance (static
/// table / input / scratch).
pub type FieldLine<'field> = (&'field [u8], &'field [u8]);

/// Per-field-resumable decode of one encoded field section (RFC 9204
/// §4.5) — the engine [`decode_into`] drives. Where `decode_into` pushes
/// every field into a [`FieldSink`] in one call, the cursor lets the
/// CALLER step: each [`Self::next_field`] decodes exactly one field line
/// and returns it borrowed, so a lending iterator (e.g. a
/// `proxima_primitives::pipe::part::PartSource` yielding one `Part` per step) can be
/// built directly on the decode with no intermediate arena or owned
/// copy. Tier-3: `core`-only, borrows from `input` / the RFC 9204
/// Appendix A static table / the caller's `scratch`, never allocates.
///
/// Construction decodes + validates the field section prefix (§4.5.1);
/// the `SETTINGS_MAX_FIELD_SECTION_SIZE` `cap` is enforced cumulatively
/// across `next_field` calls, checked BEFORE any Huffman/copy work per
/// field, exactly as `decode_into` always has.
#[derive(Debug)]
pub struct FieldSectionCursor<'input> {
    input: &'input [u8],
    cursor: usize,
    cap: u64,
    accumulated: u64,
}

impl<'input> FieldSectionCursor<'input> {
    /// Decode the field section prefix per §4.5.1:
    /// `<Required Insert Count>` (8-bit prefix) then `<S | Delta Base>`
    /// (S=1 bit, Delta Base=7-bit prefix).
    ///
    /// # Errors
    ///
    /// [`DecodeError::Truncated`] / [`DecodeError::Integer`] on a
    /// malformed prefix; [`DecodeError::DynamicTableRequired`] when the
    /// section declares a non-zero Required Insert Count (dynamic-table
    /// state is unwired in v1).
    #[inline]
    pub fn new(input: &'input [u8], cap: u64) -> Result<Self, DecodeError> {
        let mut cursor = 0usize;
        if input.is_empty() {
            return Err(DecodeError::Truncated);
        }
        let (required_insert_count, ric_len) = integer::decode(input, 8)?;
        cursor += ric_len;
        if required_insert_count != 0 {
            return Err(DecodeError::DynamicTableRequired);
        }
        if input.len() < cursor + 1 {
            return Err(DecodeError::Truncated);
        }
        let (delta_base, db_len) = integer::decode(&input[cursor..], 7)?;
        cursor += db_len;
        let _ = delta_base; // unused without dynamic table
        Ok(Self {
            input,
            cursor,
            cap,
            accumulated: 0,
        })
    }

    /// Decode the next field line, or `Ok(None)` once the section is
    /// exhausted. `name`/`value` are BORROWED — from the static table
    /// (`'static`), from `input`, or from `scratch` (Huffman output,
    /// overwritten by the next call) — and are valid until the next
    /// `next_field` call at most.
    ///
    /// # Errors
    ///
    /// See [`DecodeError`]; a cursor that has returned an error is not
    /// resumable (the wire position of the failed field is undefined).
    #[inline(always)]
    pub fn next_field<'field>(
        &mut self,
        scratch: &'field mut [u8],
    ) -> Result<Option<FieldLine<'field>>, DecodeError>
    where
        'input: 'field,
    {
        let input: &'input [u8] = self.input;
        if self.cursor >= input.len() {
            return Ok(None);
        }
        let field_input = &input[self.cursor..];
        let first = field_input[0];
        let field = if first & 0b1000_0000 != 0 {
            decode_indexed(
                field_input,
                &mut self.cursor,
                self.cap,
                &mut self.accumulated,
            )?
        } else if first & 0b0100_0000 != 0 {
            decode_literal_with_name_ref(
                field_input,
                &mut self.cursor,
                self.cap,
                &mut self.accumulated,
                scratch,
            )?
        } else if first & 0b0010_0000 != 0 {
            decode_literal_with_literal_name(
                field_input,
                &mut self.cursor,
                self.cap,
                &mut self.accumulated,
                scratch,
            )?
        } else {
            // Patterns 0b000xxxxx (post-base indexed) + 0b0001xxxx
            // (literal with post-base name) require dynamic-table
            // post-base state per §4.5.4-6 — unwired in v1.
            return Err(DecodeError::DynamicTableRequired);
        };
        Ok(Some(field))
    }
}

#[inline(always)]
fn decode_indexed<'field>(
    input: &'field [u8],
    global_cursor: &mut usize,
    cap: u64,
    accumulated: &mut u64,
) -> Result<(&'field [u8], &'field [u8]), DecodeError> {
    // Pattern `1Txxxxxx` — T=1 static, T=0 dynamic. 6-bit prefix.
    let first = input[0];
    let is_static = first & 0b0100_0000 != 0;
    let (index, consumed) = integer::decode(input, 6)?;
    *global_cursor += consumed;
    if !is_static {
        return Err(DecodeError::DynamicTableRequired);
    }
    let entry =
        static_table::get(index as usize).ok_or(DecodeError::UnknownStaticIndex { index })?;
    let field_cost = (entry.name.len() as u64)
        .saturating_add(entry.value.len() as u64)
        .saturating_add(FIELD_OVERHEAD_BYTES);
    *accumulated = accumulated.saturating_add(field_cost);
    if *accumulated > cap {
        return Err(DecodeError::ExceedsMaxFieldSectionSize {
            cap,
            observed: *accumulated,
        });
    }
    Ok((entry.name, entry.value))
}

#[inline(always)]
fn decode_literal_with_name_ref<'field>(
    input: &'field [u8],
    global_cursor: &mut usize,
    cap: u64,
    accumulated: &mut u64,
    scratch: &'field mut [u8],
) -> Result<(&'field [u8], &'field [u8]), DecodeError> {
    // Pattern `01NTxxxx` — T=1 static, N=never-indexed (passes through
    // proxies untouched), 4-bit prefix for the name index.
    let first = input[0];
    let is_static = first & 0b0001_0000 != 0;
    let _never_index = first & 0b0010_0000 != 0;
    let (name_index, name_consumed) = integer::decode(input, 4)?;
    *global_cursor += name_consumed;
    let value_input = &input[name_consumed..];
    if value_input.is_empty() {
        return Err(DecodeError::Truncated);
    }
    let huffman = value_input[0] & 0b1000_0000 != 0;
    let (value_len, value_len_consumed) = integer::decode(value_input, 7)?;
    *global_cursor += value_len_consumed;
    if !is_static {
        return Err(DecodeError::DynamicTableRequired);
    }
    let name_entry = static_table::get(name_index as usize)
        .ok_or(DecodeError::UnknownStaticIndex { index: name_index })?;
    // Pre-flight: refuse to do any Huffman/copy work on the value if
    // doing so would push the running header-section size past the
    // advertised cap. Huffman expands at most 8/5 ≈ 1.6× (the maximum
    // ratio for the RFC 7541 static table); we estimate the WORST-case
    // decoded value length and clamp early to keep the bound
    // conservative.
    let max_value_decoded = if huffman {
        value_len.saturating_mul(8).div_ceil(5)
    } else {
        value_len
    };
    let projected = accumulated
        .saturating_add(name_entry.name.len() as u64)
        .saturating_add(max_value_decoded)
        .saturating_add(FIELD_OVERHEAD_BYTES);
    if projected > cap {
        return Err(DecodeError::ExceedsMaxFieldSectionSize {
            cap,
            observed: projected,
        });
    }
    let value_start = name_consumed + value_len_consumed;
    let value_end = value_start
        .checked_add(value_len as usize)
        .ok_or(DecodeError::Truncated)?;
    if input.len() < value_end {
        return Err(DecodeError::Truncated);
    }
    *global_cursor += value_len as usize;
    let raw_value = &input[value_start..value_end];
    let value = resolve_value(raw_value, huffman, scratch)?;
    let field_cost = (name_entry.name.len() as u64)
        .saturating_add(value.len() as u64)
        .saturating_add(FIELD_OVERHEAD_BYTES);
    *accumulated = accumulated.saturating_add(field_cost);
    if *accumulated > cap {
        return Err(DecodeError::ExceedsMaxFieldSectionSize {
            cap,
            observed: *accumulated,
        });
    }
    Ok((name_entry.name, value))
}

#[inline(always)]
fn decode_literal_with_literal_name<'field>(
    input: &'field [u8],
    global_cursor: &mut usize,
    cap: u64,
    accumulated: &mut u64,
    scratch: &'field mut [u8],
) -> Result<(&'field [u8], &'field [u8]), DecodeError> {
    // Pattern `001NHxxx` — N=never-indexed, H=huffman (name only),
    // 3-bit prefix for the name length.
    let first = input[0];
    let name_huffman = first & 0b0000_1000 != 0;
    let (name_len, name_len_consumed) = integer::decode(input, 3)?;
    *global_cursor += name_len_consumed;
    let name_start = name_len_consumed;
    let name_end = name_start
        .checked_add(name_len as usize)
        .ok_or(DecodeError::Truncated)?;
    if input.len() < name_end {
        return Err(DecodeError::Truncated);
    }
    *global_cursor += name_len as usize;
    let value_input = &input[name_end..];
    if value_input.is_empty() {
        return Err(DecodeError::Truncated);
    }
    let value_huffman = value_input[0] & 0b1000_0000 != 0;
    let (value_len, value_len_consumed) = integer::decode(value_input, 7)?;
    *global_cursor += value_len_consumed;
    // Pre-flight check both name + value against the cap before doing
    // any Huffman/copy work. Worst-case Huffman expansion bound = ×8/5.
    let max_name_decoded = if name_huffman {
        name_len.saturating_mul(8).div_ceil(5)
    } else {
        name_len
    };
    let max_value_decoded = if value_huffman {
        value_len.saturating_mul(8).div_ceil(5)
    } else {
        value_len
    };
    let projected = accumulated
        .saturating_add(max_name_decoded)
        .saturating_add(max_value_decoded)
        .saturating_add(FIELD_OVERHEAD_BYTES);
    if projected > cap {
        return Err(DecodeError::ExceedsMaxFieldSectionSize {
            cap,
            observed: projected,
        });
    }
    let value_start = name_end + value_len_consumed;
    let value_end = value_start
        .checked_add(value_len as usize)
        .ok_or(DecodeError::Truncated)?;
    if input.len() < value_end {
        return Err(DecodeError::Truncated);
    }
    *global_cursor += value_len as usize;
    let raw_name = &input[name_start..name_end];
    let raw_value = &input[value_start..value_end];

    let (name, value): (&[u8], &[u8]) = match (name_huffman, value_huffman) {
        (false, false) => (raw_name, raw_value),
        (true, false) => (resolve_value(raw_name, true, scratch)?, raw_value),
        (false, true) => (raw_name, resolve_value(raw_value, true, scratch)?),
        (true, true) => resolve_both_huffman(raw_name, raw_value, max_name_decoded, scratch)?,
    };

    let field_cost = (name.len() as u64)
        .saturating_add(value.len() as u64)
        .saturating_add(FIELD_OVERHEAD_BYTES);
    *accumulated = accumulated.saturating_add(field_cost);
    if *accumulated > cap {
        return Err(DecodeError::ExceedsMaxFieldSectionSize {
            cap,
            observed: *accumulated,
        });
    }
    Ok((name, value))
}

/// Resolve one literal string value: pass `raw` through untouched when
/// it's not Huffman-encoded, else decode it into `scratch`. Split into
/// its own function (rather than an inline `if`) so the `alloc` /
/// `no-alloc` cfg split below can select a whole function body — a
/// bare `#[cfg]` on an expression isn't valid Rust.
#[cfg(feature = "http3_codec-alloc")]
fn resolve_value<'scratch>(
    raw: &'scratch [u8],
    huffman: bool,
    scratch: &'scratch mut [u8],
) -> Result<&'scratch [u8], DecodeError> {
    if huffman {
        huffman_decode_into(raw, scratch)
    } else {
        Ok(raw)
    }
}

#[cfg(not(feature = "http3_codec-alloc"))]
fn resolve_value<'scratch>(
    raw: &'scratch [u8],
    huffman: bool,
    _scratch: &'scratch mut [u8],
) -> Result<&'scratch [u8], DecodeError> {
    if huffman {
        Err(DecodeError::HuffmanUnsupported)
    } else {
        Ok(raw)
    }
}

/// Resolve a literal-with-literal-name field whose name AND value are
/// BOTH Huffman-encoded: `scratch` must be split into two disjoint
/// regions (`split_at_mut`) so both decoded views can be alive
/// simultaneously when handed to [`FieldSink::field`].
#[cfg(feature = "http3_codec-alloc")]
fn resolve_both_huffman<'scratch>(
    raw_name: &[u8],
    raw_value: &[u8],
    max_name_decoded: u64,
    scratch: &'scratch mut [u8],
) -> Result<(&'scratch [u8], &'scratch [u8]), DecodeError> {
    let name_cap = saturating_usize(max_name_decoded);
    if scratch.len() < name_cap {
        return Err(DecodeError::ScratchTooSmall {
            needed: name_cap,
            available: scratch.len(),
        });
    }
    let (name_scratch, value_scratch) = scratch.split_at_mut(name_cap);
    let name = huffman_decode_into(raw_name, name_scratch)?;
    let value = huffman_decode_into(raw_value, value_scratch)?;
    Ok((name, value))
}

#[cfg(not(feature = "http3_codec-alloc"))]
fn resolve_both_huffman<'scratch>(
    _raw_name: &[u8],
    _raw_value: &[u8],
    _max_name_decoded: u64,
    _scratch: &'scratch mut [u8],
) -> Result<(&'scratch [u8], &'scratch [u8]), DecodeError> {
    Err(DecodeError::HuffmanUnsupported)
}

/// `usize::try_from` with a saturating fallback — used only to compare
/// against `scratch.len()`, so clamping an overflowing `u64` estimate
/// to `usize::MAX` just makes the "too small" check stricter, never
/// unsound.
#[cfg(feature = "http3_codec-alloc")]
fn saturating_usize(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

/// Huffman-decode `raw` into `scratch`, returning the written prefix.
/// Errors BEFORE calling the decoder (never mid-decode, never a panic)
/// when `raw`'s worst-case expansion (RFC 7541 min code length 5 bits
/// ⇒ ≤8/5 expansion) would exceed `scratch`.
///
/// Composes [`crate::hpack::huffman::decode`] (the primitive) —
/// `&mut [u8]` implements `bytes::BufMut` (a reborrowed `&mut &mut
/// [u8]` tracks the write cursor), so decode writes directly into the
/// caller/engine-owned buffer with no intermediate `Vec`.
#[cfg(feature = "http3_codec-alloc")]
fn huffman_decode_into<'scratch>(
    raw: &[u8],
    scratch: &'scratch mut [u8],
) -> Result<&'scratch [u8], DecodeError> {
    let raw_len_bits = (raw.len() as u64).saturating_mul(8);
    let max_len = saturating_usize(raw_len_bits.div_ceil(5));
    if scratch.len() < max_len {
        return Err(DecodeError::ScratchTooSmall {
            needed: max_len,
            available: scratch.len(),
        });
    }
    let mut cursor: &mut [u8] = &mut *scratch;
    let written = crate::hpack::huffman::decode(raw, &mut cursor)
        .map_err(|_| DecodeError::HuffmanUnsupported)?;
    Ok(&scratch[..written])
}

/// `VecFieldSink` adapts the borrowing [`decode_into`] engine to the
/// owned-`Vec` surface — the ONLY place this crate materialises
/// [`DecodedField`]. A worked example of implementing [`FieldSink`]
/// directly (vs. the blanket closure impl) for a caller that wants a
/// named, reusable sink type.
#[cfg(feature = "http3_codec-alloc")]
struct VecFieldSink<'a> {
    fields: &'a mut Vec<DecodedField>,
}

#[cfg(feature = "http3_codec-alloc")]
impl FieldSink for VecFieldSink<'_> {
    fn field(&mut self, name: &[u8], value: &[u8]) -> Result<(), DecodeError> {
        self.fields.push(DecodedField {
            name: name.to_vec(),
            value: value.to_vec(),
        });
        Ok(())
    }
}

/// Decode one encoded field section under the
/// `SETTINGS_MAX_FIELD_SECTION_SIZE` cap that this endpoint advertised.
/// `cap` is the **decoded** size limit per RFC 9114 §4.1.1.3 — sum of
/// `name.len() + value.len() + 32` across all fields.
///
/// Thin alloc-tier wrapper over [`decode_into`] (P1 RISC reuse — one
/// decode engine, two surfaces): drives it with a fixed-size internal
/// scratch (`crate::sized::PROXIMA_PROTOCOLS_HTTP3_CODEC_QPACK_DECODE_BOUNDED_SCRATCH_LEN`
/// bytes, per-crate build-time tunable) and a [`VecFieldSink`] that
/// copies each borrowed field into an owned [`DecodedField`]. Use
/// [`decode_into`] directly when the caller can supply pre-allocated
/// storage instead (the tier-3 / 0-alloc path).
///
/// # Errors
///
/// See [`DecodeError`]; in particular
/// [`DecodeError::ExceedsMaxFieldSectionSize`] when the limit would be
/// exceeded, and [`DecodeError::ScratchTooSmall`] when a single
/// field's Huffman-decoded value would exceed the internal scratch
/// buffer (raise it via `PROXIMA_PROTOCOLS_HTTP3_CODEC_QPACK_DECODE_BOUNDED_SCRATCH_LEN`
/// or call [`decode_into`] directly with a larger caller-owned buffer).
#[cfg(feature = "http3_codec-alloc")]
pub fn decode_bounded(input: &[u8], cap: u64) -> Result<Vec<DecodedField>, DecodeError> {
    let mut fields = Vec::new();
    let mut scratch = [0u8; crate::sized::PROXIMA_PROTOCOLS_HTTP3_CODEC_QPACK_DECODE_BOUNDED_SCRATCH_LEN];
    let mut sink = VecFieldSink {
        fields: &mut fields,
    };
    decode_into(input, cap, &mut scratch, &mut sink)?;
    Ok(fields)
}

/// Decode without enforcing any header-section size cap. Convenience
/// wrapper around [`decode_bounded`] with `cap = u64::MAX`.
/// Production callers (server / client connection FSMs) MUST use
/// [`decode_bounded`] with their advertised
/// `SETTINGS_MAX_FIELD_SECTION_SIZE` — otherwise a hostile peer can
/// force unbounded processing per RFC 9114 §4.1.1.
///
/// # Errors
///
/// See [`DecodeError`].
#[cfg(feature = "http3_codec-alloc")]
pub fn decode(input: &[u8]) -> Result<Vec<DecodedField>, DecodeError> {
    decode_bounded(input, u64::MAX)
}

#[cfg(all(test, feature = "http3_codec-alloc"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::http3_codec::qpack::encoder;
    use alloc::vec;

    fn encode_literal_field(name: &[u8], value: &[u8]) -> Vec<u8> {
        // encode_refs writes its own field section prefix; caller does
        // NOT need to push the prefix bytes manually.
        let mut out: Vec<u8> = Vec::new();
        encoder::encode_refs([(name, value)].iter().copied(), &mut out)
            .expect("encode small literal field");
        out
    }

    type OwnedFields = Vec<(Vec<u8>, Vec<u8>)>;

    /// Collects every [`FieldSink::field`] call into `(name, value)`
    /// owned pairs — used by tests that want to inspect `decode_into`'s
    /// output without reaching for [`decode_bounded`].
    fn collect_into(
        input: &[u8],
        cap: u64,
        scratch: &mut [u8],
    ) -> Result<OwnedFields, DecodeError> {
        let mut collected: OwnedFields = Vec::new();
        let mut sink = |name: &[u8], value: &[u8]| -> Result<(), DecodeError> {
            collected.push((name.to_vec(), value.to_vec()));
            Ok(())
        };
        decode_into(input, cap, scratch, &mut sink)?;
        Ok(collected)
    }

    #[test]
    fn decode_bounded_rejects_section_exceeding_cap() {
        let big_value = alloc::vec![b'a'; 4096];
        let encoded = encode_literal_field(b":custom", &big_value);
        // cap = 64 bytes — far smaller than (name + value + 32).
        let err = decode_bounded(&encoded, 64).expect_err("expected cap rejection");
        match err {
            DecodeError::ExceedsMaxFieldSectionSize { cap, observed } => {
                assert_eq!(cap, 64);
                assert!(observed > cap, "observed must exceed cap; got {observed}");
            }
            other => panic!("expected ExceedsMaxFieldSectionSize, got {other:?}"),
        }
    }

    #[test]
    fn decode_bounded_accepts_section_at_cap() {
        let encoded = encode_literal_field(b":custom", b"v");
        // 7 (name) + 1 (value) + 32 (overhead) = 40 bytes per field.
        // The prefix decoded fields land in the static table or as
        // literal-with-literal-name; either way the cap math runs.
        let decoded = decode_bounded(&encoded, 4096).expect("decode under cap");
        assert!(!decoded.is_empty());
    }

    #[test]
    fn decode_alias_keeps_unbounded_behavior_for_back_compat() {
        let big_value = alloc::vec![b'a'; 4096];
        let encoded = encode_literal_field(b":custom", &big_value);
        // decode() == decode_bounded(_, u64::MAX) — exists for tests
        // and fuzz harnesses. Production callers must use
        // decode_bounded with their SETTINGS_MAX_FIELD_SECTION_SIZE.
        let _ = decode(&encoded).expect("unbounded decode succeeds");
    }

    /// RFC 9204 Appendix A index 25 = `:status`/`200`; RFC 9204 §4.5.2
    /// Indexed Field Line = `1` `T`(=1 static) + 6-bit prefix integer.
    /// 25 < 2^6-1=63 so the index fits in one byte: `0b11_011001` =
    /// 0xD9. Prefix bytes (RIC=0, S=0/DeltaBase=0) are one zero byte
    /// each per §4.5.1.
    #[test]
    fn rfc_9204_appendix_a_static_indexed_status_200_via_decode_into() {
        let wire = [0x00u8, 0x00, 0xD9];
        let mut scratch = [0u8; 64];
        let decoded = collect_into(&wire, u64::MAX, &mut scratch).expect("decode indexed field");
        assert_eq!(decoded, vec![(b":status".to_vec(), b"200".to_vec())]);
    }

    #[test]
    fn literal_with_static_name_raw_value_decodes_via_decode_into() {
        // ":status: 999" — name hits the static table (":status" first
        // appears at index 24), value doesn't, so this is a
        // Literal-with-Name-Reference (RFC 9204 §4.5.4) with a raw
        // (non-Huffman) value.
        let encoded = encode_literal_field(b":status", b"999");
        let mut scratch = [0u8; 64];
        let decoded = collect_into(&encoded, u64::MAX, &mut scratch).expect("decode literal");
        assert_eq!(decoded, vec![(b":status".to_vec(), b"999".to_vec())]);
    }

    /// RFC 7541 §C.4.2 — Huffman-encoding "no-cache" yields the 6-byte
    /// sequence `a8 eb 10 64 9c bf` (bit-exact, cross-checked against
    /// `proxima-hpack`'s own `rfc_c_4_2_decode_no_cache` test). Built
    /// here as a QPACK Literal-with-Name-Reference (RFC 9204 §4.5.4,
    /// pattern `01NTxxxx`) whose name is a static-table reference
    /// (`cache-control`, looked up at runtime via `find_name` — never
    /// a hand-guessed index) and whose value is that Huffman byte
    /// sequence with H=1.
    #[cfg(feature = "std")]
    #[test]
    fn rfc_7541_c_4_2_huffman_literal_value_decodes_0_alloc_via_decode_into() {
        let name_index =
            static_table::find_name(b"cache-control").expect("cache-control in static table");
        let mut wire = alloc::vec![0x00u8, 0x00]; // field-section prefix
        let mut name_ref = [0u8; 4];
        let name_written = integer::encode(name_index as u64, 4, 0b0101_0000, &mut name_ref)
            .expect("encode 4-bit name index");
        wire.extend_from_slice(&name_ref[..name_written]);
        let huffman_no_cache = [0xa8u8, 0xeb, 0x10, 0x64, 0x9c, 0xbf];
        let mut value_len = [0u8; 2];
        let value_len_written = integer::encode(
            huffman_no_cache.len() as u64,
            7,
            0b1000_0000,
            &mut value_len,
        )
        .expect("encode 7-bit H=1 value length");
        wire.extend_from_slice(&value_len[..value_len_written]);
        wire.extend_from_slice(&huffman_no_cache);

        let mut scratch = [0u8; 32];
        let region = crate::alloc_test::exclusive_region();
        let decoded = collect_into(&wire, u64::MAX, &mut scratch).expect("decode huffman literal");
        assert_eq!(
            decoded,
            vec![(b"cache-control".to_vec(), b"no-cache".to_vec())]
        );
        // collect_into's Vec pushes DO allocate — isolate decode_into
        // itself via a non-allocating sink for the load-bearing claim.
        let mut field_count = 0usize;
        let mut probe = |name: &[u8], value: &[u8]| -> Result<(), DecodeError> {
            assert_eq!(name, b"cache-control");
            assert_eq!(value, b"no-cache");
            field_count += 1;
            Ok(())
        };
        let before = region.change();
        decode_into(&wire, u64::MAX, &mut scratch, &mut probe).expect("re-decode via probe sink");
        let after = region.change();
        assert_eq!(field_count, 1);
        assert_eq!(
            after.allocations - before.allocations,
            0,
            "decode_into must perform 0 heap allocations on the Huffman path"
        );
    }

    /// Synthesizes an nginx-shaped `200` response header set via this
    /// crate's own canonical encoder (RFC 9204 §4.5 conformant emit) —
    /// per guiding-principles principle 9's "truly impossible" real-
    /// data exception: standing up a live nginx-h3 endpoint + capturing
    /// its wire bytes is external infra this pass doesn't have (the H3
    /// e2e bench that unblocked this redesign lives in a sibling
    /// worktree's `examples/rekt_h3_load.rs`, not here). All five
    /// values are non-empty on purpose (see the alloc-count test below
    /// — an empty value skips the heap entirely and would corrupt the
    /// `1 + 2*field_count` formula).
    fn nginx_like_response_wire() -> Vec<u8> {
        let mut out = Vec::new();
        encoder::encode_refs(
            [
                (b":status".as_slice(), b"200".as_slice()),
                (b"server".as_slice(), b"nginx/1.27.0".as_slice()),
                (
                    b"date".as_slice(),
                    b"Tue, 30 Jun 2026 00:00:00 GMT".as_slice(),
                ),
                (b"content-type".as_slice(), b"text/html".as_slice()),
                (b"content-length".as_slice(), b"612".as_slice()),
            ],
            &mut out,
        )
        .expect("encode nginx-shaped response header set");
        out
    }

    #[test]
    fn decode_into_and_decode_bounded_agree_on_nginx_shaped_response() {
        let wire = nginx_like_response_wire();
        let mut scratch = [0u8; 256];
        let via_into = collect_into(&wire, u64::MAX, &mut scratch).expect("decode_into");
        let via_bounded = decode_bounded(&wire, u64::MAX).expect("decode_bounded");
        assert_eq!(via_into.len(), via_bounded.len());
        for (index, (name, value)) in via_into.iter().enumerate() {
            assert_eq!(*name, via_bounded[index].name, "field {index} name");
            assert_eq!(*value, via_bounded[index].value, "field {index} value");
        }
    }

    /// DC-H3-ALLOC-TEST-WIRE — the meta-row's per-exception assertion
    /// for `DC-H3-QPACK-DECODE-OWNS-VECS`. `decode_into` (driven by a
    /// non-allocating counting sink) performs 0 heap allocations over
    /// the captured block; `decode_bounded` performs exactly `1 + 2 *
    /// field_count` (1 for the outer `Vec<DecodedField>`'s first push,
    /// 2 per field for `name.to_vec()` + `value.to_vec()`). Vec's
    /// later amortized-growth reallocations are tracked by
    /// `stats_alloc` as `reallocations`, a separate counter — this
    /// assertion is intentionally scoped to fresh `alloc()` calls, the
    /// dimension the redesign actually closes.
    #[cfg(feature = "std")]
    #[test]
    fn alloc_count_decode_into_zero_decode_bounded_one_plus_two_per_field() {
        let wire = nginx_like_response_wire();
        let field_count = 5usize;
        let mut scratch = [0u8; 256];

        let region = crate::alloc_test::exclusive_region();
        let before = region.change();
        let mut probe_count = 0usize;
        let mut probe = |_name: &[u8], _value: &[u8]| -> Result<(), DecodeError> {
            probe_count += 1;
            Ok(())
        };
        decode_into(&wire, u64::MAX, &mut scratch, &mut probe).expect("decode_into");
        let after_into = region.change();
        assert_eq!(probe_count, field_count);
        assert_eq!(
            after_into.allocations - before.allocations,
            0,
            "decode_into must perform 0 heap allocations"
        );

        let before_bounded = region.change();
        let decoded = decode_bounded(&wire, u64::MAX).expect("decode_bounded");
        let after_bounded = region.change();
        assert_eq!(decoded.len(), field_count);
        assert_eq!(
            after_bounded.allocations - before_bounded.allocations,
            1 + 2 * field_count,
            "decode_bounded must perform exactly 1 + 2*field_count allocations"
        );
    }

    /// `FieldSectionCursor` is the engine `decode_into` drives — stepping
    /// it field-by-field must reproduce exactly the fields the one-call
    /// sink surface yields, on the same wire, in the same order.
    #[test]
    fn field_section_cursor_stepwise_matches_decode_into_on_nginx_shaped_response() {
        let wire = nginx_like_response_wire();
        let mut scratch = [0u8; 256];
        let via_sink = collect_into(&wire, u64::MAX, &mut scratch).expect("decode_into");

        let mut cursor = FieldSectionCursor::new(&wire, u64::MAX).expect("prefix decode");
        let mut via_cursor: OwnedFields = Vec::new();
        while let Some((name, value)) = cursor
            .next_field(&mut scratch)
            .expect("stepwise field decode")
        {
            via_cursor.push((name.to_vec(), value.to_vec()));
        }
        assert_eq!(via_cursor, via_sink);
    }

    #[test]
    fn field_section_cursor_enforces_cap_cumulatively_across_steps() {
        let wire = nginx_like_response_wire();
        // 5 fields at ~40-70 decoded bytes each — a cap of 100 admits the
        // first field and must reject a later step, not the construction.
        let mut cursor = FieldSectionCursor::new(&wire, 100).expect("prefix decode");
        let mut scratch = [0u8; 256];
        let mut steps = 0usize;
        let err = loop {
            match cursor.next_field(&mut scratch) {
                Ok(Some(_)) => steps += 1,
                Ok(None) => panic!("cap must reject before the section completes"),
                Err(err) => break err,
            }
        };
        assert!(steps >= 1, "first small field fits under the 100-byte cap");
        assert!(matches!(
            err,
            DecodeError::ExceedsMaxFieldSectionSize { cap: 100, .. }
        ));
    }

    #[test]
    fn field_section_cursor_rejects_truncated_prefix_at_construction() {
        let err = FieldSectionCursor::new(&[0x00], u64::MAX)
            .expect_err("one-byte input truncates the two-varint prefix");
        assert_eq!(err, DecodeError::Truncated);
    }

    #[rstest]
    #[case::truncated_mid_indexed_field(&[0x00, 0x00, 0xFF])]
    #[case::truncated_before_field_section_prefix(&[0x00])]
    fn decode_into_rejects_truncated_input(#[case] wire: &[u8]) {
        let mut scratch = [0u8; 16];
        let mut sink = |_: &[u8], _: &[u8]| -> Result<(), DecodeError> { Ok(()) };
        let err = decode_into(wire, u64::MAX, &mut scratch, &mut sink)
            .expect_err("truncated input must error");
        assert!(
            matches!(err, DecodeError::Truncated | DecodeError::Integer(_)),
            "expected a truncation-shaped error, got {err:?}"
        );
    }

    #[test]
    fn decode_into_rejects_section_exceeding_cap_before_huffman_work() {
        let big_value = alloc::vec![b'a'; 4096];
        let encoded = encode_literal_field(b":custom", &big_value);
        let mut scratch = [0u8; 16];
        let mut sink = |_: &[u8], _: &[u8]| -> Result<(), DecodeError> {
            panic!("sink must not be called once the pre-flight cap check rejects the field")
        };
        let err =
            decode_into(&encoded, 64, &mut scratch, &mut sink).expect_err("expected cap rejection");
        assert!(matches!(
            err,
            DecodeError::ExceedsMaxFieldSectionSize { cap: 64, .. }
        ));
    }

    #[test]
    fn decode_into_rejects_nonzero_required_insert_count() {
        // Required Insert Count = 1 (8-bit prefix, one byte, no
        // dynamic-table state wired in v1) per RFC 9204 §4.5.1.
        let wire = [0x01u8];
        let mut scratch = [0u8; 16];
        let mut sink = |_: &[u8], _: &[u8]| -> Result<(), DecodeError> { Ok(()) };
        let err = decode_into(&wire, u64::MAX, &mut scratch, &mut sink)
            .expect_err("non-zero RIC must be rejected");
        assert_eq!(err, DecodeError::DynamicTableRequired);
    }

    #[test]
    fn huffman_output_exceeding_scratch_errors_not_panics() {
        // Same RFC 7541 §C.4.2 "no-cache" vector as the happy-path
        // test above, decoded into a 1-byte scratch (needs 8: 6 bytes
        // * 8/5 rounded up) — must return ScratchTooSmall, not panic
        // or silently truncate.
        let name_index =
            static_table::find_name(b"cache-control").expect("cache-control in static table");
        let mut wire = alloc::vec![0x00u8, 0x00];
        let mut name_ref = [0u8; 4];
        let name_written = integer::encode(name_index as u64, 4, 0b0101_0000, &mut name_ref)
            .expect("encode 4-bit name index");
        wire.extend_from_slice(&name_ref[..name_written]);
        let huffman_no_cache = [0xa8u8, 0xeb, 0x10, 0x64, 0x9c, 0xbf];
        let mut value_len = [0u8; 2];
        let value_len_written = integer::encode(
            huffman_no_cache.len() as u64,
            7,
            0b1000_0000,
            &mut value_len,
        )
        .expect("encode 7-bit H=1 value length");
        wire.extend_from_slice(&value_len[..value_len_written]);
        wire.extend_from_slice(&huffman_no_cache);

        let mut tiny_scratch = [0u8; 1];
        let mut sink = |_: &[u8], _: &[u8]| -> Result<(), DecodeError> {
            panic!("sink must not be called when scratch is too small")
        };
        let err = decode_into(&wire, u64::MAX, &mut tiny_scratch, &mut sink)
            .expect_err("scratch too small must error, not panic");
        assert!(matches!(err, DecodeError::ScratchTooSmall { .. }));
    }
}
