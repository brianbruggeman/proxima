//! A numeric format resolved once into the two slice conversions that carry it.
//!
//! Teaching pointer: [`crate::float_layout_codec`] holds the scalar slice loop (`encode_body`,
//! `decode_body`) and [`crate::block_codec`] the block loops (`encode_run`, `decode_run`).
//! [`FormatCodec::scalar`] instantiates the scalar loop once per row of the named-layout table
//! ([`crate::float_layout::named_layout_table`]) and [`FormatCodec::block`] the block loops once per
//! row of the block table ([`crate::block_layout::named_block_table`]), each with the layout as a
//! literal constant, so every shift, mask, plane address and magic number is folded from the
//! layout's fields. Resolve at allocation time and keep the `Copy` value: a call is then one
//! indirect jump, where [`crate::encode::encode_blocks`] re-matches the format on every call. A
//! scalar layout without a name resolves to the same loop with the layout as data, so every
//! supported scalar layout has a codec; a block layout outside the table has none and keeps
//! [`crate::encode::encode_blocks`] and [`crate::decode::decode_blocks`].
//!
//! One window contract serves every caller. [`FormatCodec::encode`] writes the packed form of
//! `values` into the front of `out`, [`FormatCodec::decode`] reads the packed range for `out.len()`
//! elements from the front of `data`, and each returns the byte count. The codec derives the range
//! from the element count, so a caller holding a byte window over a whole allocation passes it
//! as is and sizes nothing itself; an exact-size buffer is the window that ends at the range. A
//! window shorter than the range, or an element count that is not whole blocks, is an error and
//! nothing is written; bytes past the range are untouched by encode and unread by decode. A caller
//! that needs the buffer to be exactly the range asserts the returned count.

use core::mem::MaybeUninit;

use crate::block_codec::{DecodeFn, EncodeFn};
use crate::block_layout::BlockLayout;
use crate::error::NumericError;
use crate::float_layout::{FloatLayout, named_layout_table};
use crate::float_layout_codec::{
    SOURCE_LAYOUT, decode_body, dispatch_named_layout, element_bytes, encode_body, is_storable,
};
use crate::format::NumericFormat;

const UNNAMED: &str = "unnamed";

// the element count up to which an identity decode copies inline: a vector loop beats the libc memcpy call by 5-11% at
// 256 and 512 elements and ties at 2048 and 8192, then loses 3-9% from 32768 up, where the out-of-line memcpy wins
const IDENTITY_INLINE_ELEMENTS: usize = 8192;

#[derive(Clone, Copy)]
enum Route {
    Scalar {
        encode: Option<EncodeFn>,
        decode: DecodeFn,
    },
    Block {
        encode: Option<EncodeFn>,
        decode: DecodeFn,
    },
    Identity,
    Layout(FloatLayout),
}

/// The encode and decode of one numeric format: plain function pointers to table instantiations
/// with the layout as a literal (a named scalar layout or a table block layout), or the scalar
/// layout as data for any other storable layout. No allocation, no trait object.
///
/// A scalar is a block of one element, so one type and one window check serve both: a call sizes
/// the packed range from the element count, checks the window holds it, then makes one indirect
/// call. The scalar and identity routes size the range inline by `len * width`; the rest take the
/// out-of-line path.
///
/// Example: a 16-bit layout storing 512 values takes 1 KiB of output; `Q4_0` storing 64 values
/// takes 36 bytes.
#[derive(Clone, Copy)]
pub struct FormatCodec {
    route: Route,
    elements: usize,
    block_bytes: usize,
    format: &'static str,
}

impl FormatCodec {
    pub(crate) const fn table(
        encode: Option<EncodeFn>,
        decode: DecodeFn,
        elements: usize,
        block_bytes: usize,
    ) -> Self {
        let route = if elements == 1 {
            Route::Scalar { encode, decode }
        } else {
            Route::Block { encode, decode }
        };
        Self {
            route,
            elements,
            block_bytes,
            format: UNNAMED,
        }
    }

    const fn identity() -> Self {
        Self {
            route: Route::Identity,
            elements: 1,
            block_bytes: element_bytes(&SOURCE_LAYOUT),
            format: UNNAMED,
        }
    }

    fn named_as(mut self, format: &NumericFormat) -> Self {
        self.format = format.name().unwrap_or(UNNAMED);
        self
    }

    /// The codec for a scalar layout the slice codec can execute: the named instantiation when the
    /// layout has a name, the layout-as-data loop otherwise; `None` for a layout wider than 32 bits
    /// or empty.
    #[must_use]
    pub fn scalar(layout: &FloatLayout) -> Option<Self> {
        if layout.is_identity_of(&SOURCE_LAYOUT) {
            return Some(Self::identity().named_as(&NumericFormat::Scalar(*layout)));
        }
        let codec = named_layout_table!(dispatch_named_layout!(
            *layout,
            NAMED => Some(Self::table(
                Some(|values, out| encode_body(NAMED, values, out)),
                |data, out| {
                    decode_body(NAMED, data, out, |slot, value| {
                        slot.write(value);
                    });
                },
                1,
                element_bytes(&NAMED),
            )),
            else is_storable(layout).then(|| Self {
                route: Route::Layout(*layout),
                elements: 1,
                block_bytes: element_bytes(layout),
                format: UNNAMED,
            }),
        ));
        codec.map(|resolved| resolved.named_as(&NumericFormat::Scalar(*layout)))
    }

    /// The instantiation for a layout in the block table, `None` for any other.
    #[must_use]
    pub fn block(layout: &BlockLayout) -> Option<Self> {
        crate::block_codec::resolve_block(layout)
            .map(|resolved| resolved.named_as(&NumericFormat::Block(*layout)))
    }

    /// The codec for a [`NumericFormat`]: a scalar layout, a block format, or a linear-ladder level
    /// format over a table layout. `None` for codebooks, explicit tables and any layout the table
    /// does not hold. Resolving a format says nothing about which consumers can read it.
    #[must_use]
    pub fn for_format(format: &NumericFormat) -> Option<Self> {
        match format {
            NumericFormat::Scalar(layout) => Self::scalar(layout),
            NumericFormat::Block(_) | NumericFormat::Levels(_) => {
                crate::block_codec::block_for_format(format)
            }
        }
        .map(|resolved| resolved.named_as(format))
    }

    /// Elements per block; 1 for a scalar.
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }

    /// Stored bytes per block; the element width for a scalar.
    #[must_use]
    pub const fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    /// The format's name for error text, `"unnamed"` when the format has none.
    #[must_use]
    pub const fn format_name(&self) -> &'static str {
        self.format
    }

    /// Whether this format has an encode instantiation.
    #[must_use]
    pub const fn can_encode(&self) -> bool {
        !matches!(self.route, Route::Scalar { encode: None, .. } | Route::Block { encode: None, .. })
    }

    /// Packed bytes for `elements` values, `None` when `elements` is not whole blocks. A scalar
    /// multiplies by its element width only; the block divide runs for blocks alone.
    #[must_use]
    #[inline]
    pub fn packed_len(&self, elements: usize) -> Option<usize> {
        if self.elements == 1 {
            return Some(elements * self.block_bytes);
        }
        if !elements.is_multiple_of(self.elements) {
            return None;
        }
        Some(elements / self.elements * self.block_bytes)
    }

    /// Encodes `values` into the front of `out`, little-endian, and returns the packed byte count.
    /// The packed range comes from `values`, so a caller holding a byte window over a whole
    /// allocation sizes nothing itself and the codec sizes and checks once; bytes past the range
    /// are untouched. An exact-size `out` is the window that ends at the range.
    ///
    /// # Errors
    /// [`NumericError::UnsupportedEncode`] for a layout with no encode instantiation,
    /// [`NumericError::InputNotElementMultiple`] when `values` is not whole blocks,
    /// [`NumericError::OutputSizeMismatch`] when `out` is shorter than the packed byte count; `out`
    /// is unwritten in each case.
    #[must_use = "an error leaves the output unwritten"]
    #[inline(always)]
    pub fn encode(&self, values: &[f32], out: &mut [u8]) -> Result<usize, NumericError> {
        // a scalar route's width is at most four bytes, so the product of a slice length and the
        // width cannot wrap
        match &self.route {
            Route::Scalar { encode: Some(run), .. } => {
                let bytes = values.len() * self.block_bytes;
                let Some(window) = out.get_mut(..bytes) else {
                    return Err(self.encode_fault(values.len(), out.len()));
                };
                run(values, window);
                Ok(bytes)
            }
            Route::Identity => {
                let bytes = values.len() * element_bytes(&SOURCE_LAYOUT);
                let Some(window) = out.get_mut(..bytes) else {
                    return Err(self.encode_fault(values.len(), out.len()));
                };
                encode_body(SOURCE_LAYOUT, values, window);
                Ok(bytes)
            }
            _ => self.encode_general(values, out),
        }
    }

    #[inline(never)]
    fn encode_general(&self, values: &[f32], out: &mut [u8]) -> Result<usize, NumericError> {
        if !self.can_encode() {
            return Err(self.unsupported_encode());
        }
        let Some(bytes) = self.packed_len(values.len()) else {
            return Err(self.encode_fault(values.len(), out.len()));
        };
        let Some(window) = out.get_mut(..bytes) else {
            return Err(self.encode_fault(values.len(), out.len()));
        };
        match &self.route {
            Route::Scalar { encode: Some(run), .. } | Route::Block { encode: Some(run), .. } => run(values, window),
            Route::Layout(layout) => encode_body(*layout, values, window),
            Route::Identity => encode_body(SOURCE_LAYOUT, values, window),
            Route::Scalar { encode: None, .. } | Route::Block { encode: None, .. } => {
                return Err(self.unsupported_encode());
            }
        }
        Ok(bytes)
    }

    /// Decodes the packed range for `out.len()` elements from the front of `data` into the
    /// caller-owned, possibly uninitialised `out`, initialising every slot, and returns the byte
    /// count read. The range comes from `out`, so a caller holding a byte window over a whole
    /// allocation sizes nothing itself and the codec sizes and checks once; bytes past the range
    /// are unread. An exact-size `data` is the window that ends at the range.
    ///
    /// # Errors
    /// [`NumericError::InputNotElementMultiple`] when `out` is not whole blocks,
    /// [`NumericError::InputNotBlockMultiple`] or [`NumericError::OutputSizeMismatch`] when `data`
    /// is shorter than the packed byte count; nothing is written in either case.
    #[must_use = "an error leaves the output uninitialised"]
    #[inline(always)]
    pub fn decode(&self, data: &[u8], out: &mut [MaybeUninit<f32>]) -> Result<usize, NumericError> {
        match &self.route {
            Route::Scalar { decode, .. } => {
                let bytes = out.len() * self.block_bytes;
                let Some(window) = data.get(..bytes) else {
                    return Err(self.decode_fault(data.len(), out.len()));
                };
                decode(window, out);
                Ok(bytes)
            }
            Route::Identity => {
                let bytes = out.len() * element_bytes(&SOURCE_LAYOUT);
                let Some(window) = data.get(..bytes) else {
                    return Err(self.decode_fault(data.len(), out.len()));
                };
                if out.len() <= IDENTITY_INLINE_ELEMENTS {
                    Self::decode_identity_copy(window, out);
                } else {
                    Self::decode_identity(window, out);
                }
                Ok(bytes)
            }
            _ => self.decode_general(data, out),
        }
    }

    #[inline(never)]
    fn decode_general(&self, data: &[u8], out: &mut [MaybeUninit<f32>]) -> Result<usize, NumericError> {
        let Some(bytes) = self.packed_len(out.len()) else {
            return Err(self.decode_fault(data.len(), out.len()));
        };
        let Some(window) = data.get(..bytes) else {
            return Err(self.decode_fault(data.len(), out.len()));
        };
        match &self.route {
            Route::Scalar { decode, .. } | Route::Block { decode, .. } => decode(window, out),
            Route::Layout(layout) => Self::decode_layout(layout, window, out),
            Route::Identity => Self::decode_identity(window, out),
        }
        Ok(bytes)
    }

    #[inline(always)]
    fn decode_identity_copy(data: &[u8], out: &mut [MaybeUninit<f32>]) {
        decode_body(SOURCE_LAYOUT, data, out, |slot, value| {
            slot.write(value);
        });
    }

    // out of line so the copy loop keeps its noalias arguments and lowers to one memcpy; inlined, the
    // optimiser cannot prove the packed bytes and the slots are disjoint and emits a checked vector loop
    #[inline(never)]
    fn decode_identity(data: &[u8], out: &mut [MaybeUninit<f32>]) {
        Self::decode_identity_copy(data, out);
    }

    // out of line so the layout-as-data loop's register pressure stays out of the inlined named path
    #[inline(never)]
    fn decode_layout(layout: &FloatLayout, data: &[u8], out: &mut [MaybeUninit<f32>]) {
        decode_body(*layout, data, out, |slot, value| {
            slot.write(value);
        });
    }

    #[cold]
    #[inline(never)]
    fn unsupported_encode(&self) -> NumericError {
        NumericError::UnsupportedEncode {
            format: self.format,
            reason: "no compile-time encode for this layout's fit",
        }
    }

    #[cold]
    #[inline(never)]
    fn encode_fault(&self, values: usize, out: usize) -> NumericError {
        match self.packed_len(values) {
            Some(expected) => NumericError::OutputSizeMismatch { found: out, expected },
            None => NumericError::InputNotElementMultiple {
                format: self.format,
                found: values,
                elements: self.elements,
            },
        }
    }

    #[cold]
    #[inline(never)]
    fn decode_fault(&self, data: usize, slots: usize) -> NumericError {
        if self.packed_len(slots).is_none() {
            return NumericError::InputNotElementMultiple {
                format: self.format,
                found: slots,
                elements: self.elements,
            };
        }
        if self.elements != 1 && !data.is_multiple_of(self.block_bytes) {
            return NumericError::InputNotBlockMultiple {
                format: self.format,
                found: data,
                block_bytes: self.block_bytes,
            };
        }
        NumericError::OutputSizeMismatch {
            found: slots,
            expected: data / self.block_bytes * self.elements,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::encode_blocks;

    fn kv_values(count: usize) -> Vec<f32> {
        let mut state = 0x9e37_79b9_u32;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f32::from_bits(state ^ (state << 13))
            })
            .collect()
    }

    #[test]
    fn resolved_codec_matches_encode_blocks_bit_for_bit_for_every_named_scalar() {
        let values = kv_values(4099);
        for layout in [
            FloatLayout::BF32,
            FloatLayout::F16,
            FloatLayout::BF16,
            FloatLayout::E5M2_SATURATING,
            FloatLayout::E5M2,
            FloatLayout::E4M3FN,
            FloatLayout::E3M2,
            FloatLayout::E2M1,
        ] {
            let format = NumericFormat::Scalar(layout);
            let codec = FormatCodec::scalar(&layout).expect("a named layout resolves");
            let mut expected = vec![0u8; values.len() * codec.block_bytes()];
            let mut resolved = vec![0u8; expected.len()];

            encode_blocks(&format, &values, &mut expected).expect("encode_blocks encodes");
            codec.encode(&values, &mut resolved).expect("sizes match");

            assert_eq!(resolved, expected, "{layout:?}");
        }
    }

    #[test]
    fn resolved_decode_reads_only_the_range_its_slots_name_and_refuses_more_slots_than_data() {
        let values = kv_values(1000);
        let codec = FormatCodec::scalar(&FloatLayout::F16).expect("f16 resolves");
        let mut packed = vec![0u8; values.len() * 2];
        codec.encode(&values, &mut packed).expect("sizes match");
        let mut fewer = vec![MaybeUninit::<f32>::uninit(); 999];
        let mut more = vec![MaybeUninit::<f32>::uninit(); 1001];

        let read = codec.decode(&packed, &mut fewer);
        let refused = codec.decode(&packed, &mut more);

        assert_eq!(read, Ok(1998));
        assert_eq!(refused, Err(NumericError::OutputSizeMismatch { found: 1001, expected: 1000 }));
    }

    #[test]
    fn a_layout_with_no_name_resolves_to_the_same_bytes_as_its_named_spelling() {
        let values = kv_values(4099);
        let unnamed = FloatLayout {
            bias_override: Some(FloatLayout::F16.bias()),
            ..FloatLayout::F16
        };
        let named = FormatCodec::scalar(&FloatLayout::F16).expect("f16 resolves");
        let respelled = FormatCodec::scalar(&unnamed).expect("an unnamed supported layout resolves");
        let mut from_named = vec![0u8; values.len() * 2];
        let mut from_respelled = vec![0u8; values.len() * 2];

        named.encode(&values, &mut from_named).expect("sizes match");
        respelled.encode(&values, &mut from_respelled).expect("sizes match");

        assert_eq!(from_respelled, from_named);
        assert_eq!(respelled.format_name(), UNNAMED);
    }

    #[test]
    fn every_named_scalar_layout_that_resolves_gets_the_compile_time_instantiation() {
        let mut resolved = Vec::new();
        for (layout, name) in crate::float_layout::NAMED_LAYOUTS {
            if let Some(codec) = FormatCodec::scalar(&layout) {
                assert!(
                    matches!(codec.route, Route::Scalar { .. } | Route::Identity),
                    "{name} must take a compile-time route"
                );
                resolved.push(name);
            }
        }

        assert!(resolved.len() >= 8, "the scan reached only {resolved:?}");
        assert!(resolved.contains(&"bfloat16") && resolved.contains(&"float8_e4m3fn"));
    }

    #[test]
    fn a_layout_that_is_the_source_layout_encodes_as_a_byte_copy_and_keeps_a_signalling_nan_payload() {
        let codec = FormatCodec::scalar(&SOURCE_LAYOUT).expect("the source layout resolves");
        let signalling = f32::from_bits(0x7fa0_1234);
        let mut packed = [0u8; 8];

        codec.encode(&[1.5, signalling], &mut packed).expect("sizes match");

        assert!(matches!(codec.route, Route::Identity));
        assert_eq!(packed[..4], 1.5_f32.to_le_bytes());
        assert_eq!(packed[4..], 0x7fa0_1234_u32.to_le_bytes());
    }

    fn route_codecs() -> Vec<(&'static str, FormatCodec)> {
        let respelled = FloatLayout {
            bias_override: Some(FloatLayout::F16.bias()),
            ..FloatLayout::F16
        };
        vec![
            ("identity", FormatCodec::scalar(&SOURCE_LAYOUT).expect("the source layout resolves")),
            ("named scalar", FormatCodec::scalar(&FloatLayout::BF16).expect("bf16 resolves")),
            ("layout as data", FormatCodec::scalar(&respelled).expect("a respelled f16 resolves")),
            ("block", FormatCodec::block(&BlockLayout::Q4_0).expect("q4_0 resolves")),
        ]
    }

    #[test]
    fn a_longer_window_gets_the_packed_range_and_the_count_and_keeps_its_tail() {
        let values = kv_values(256);
        for (route, codec) in route_codecs() {
            let packed_len = codec.packed_len(values.len()).expect("whole blocks");
            let mut exact = vec![0u8; packed_len];
            let mut window = vec![0xAB_u8; packed_len + 17];
            let exact_written = codec.encode(&values, &mut exact).expect("the buffer is the packed range");

            let window_written = codec.encode(&values, &mut window).expect("the window holds the packed range");

            assert_eq!((exact_written, window_written), (packed_len, packed_len), "{route} encode count");
            assert_eq!(window[..packed_len], exact, "{route} encode bytes");
            assert!(window[packed_len..].iter().all(|byte| *byte == 0xAB), "{route} encode tail");
            let mut from_exact = vec![MaybeUninit::<f32>::uninit(); values.len()];
            let mut from_window = vec![MaybeUninit::<f32>::uninit(); values.len()];

            let exact_read = codec.decode(&exact, &mut from_exact).expect("the buffer is the packed range");
            let window_read = codec.decode(&window, &mut from_window).expect("the window holds the packed range");

            assert_eq!((exact_read, window_read), (packed_len, packed_len), "{route} decode count");
        }
    }

    #[test]
    fn a_window_shorter_than_the_packed_range_is_refused_and_written_nowhere() {
        let values = kv_values(256);
        for (route, codec) in route_codecs() {
            let packed_len = codec.packed_len(values.len()).expect("whole blocks");
            let mut window = vec![0xAB_u8; packed_len - 1];

            let encode_refused = codec.encode(&values, &mut window);
            let mut slots = vec![MaybeUninit::<f32>::uninit(); values.len()];
            let decode_refused = codec.decode(&window, &mut slots);

            assert!(
                matches!(encode_refused, Err(NumericError::OutputSizeMismatch { found, expected }) if found == packed_len - 1 && expected == packed_len),
                "{route} encode: {encode_refused:?}"
            );
            assert!(window.iter().all(|byte| *byte == 0xAB), "{route} wrote into a refused window");
            assert!(decode_refused.is_err(), "{route} decode: {decode_refused:?}");
        }
    }

    #[test]
    fn a_count_that_is_not_whole_blocks_is_refused_over_any_window() {
        let codec = FormatCodec::block(&BlockLayout::Q4_0).expect("q4_0 resolves");
        let values = kv_values(33);
        let mut window = vec![0u8; 1024];
        let mut slots = vec![MaybeUninit::<f32>::uninit(); 33];

        let encode_refused = codec.encode(&values, &mut window);
        let decode_refused = codec.decode(&window, &mut slots);

        assert!(matches!(encode_refused, Err(NumericError::InputNotElementMultiple { found: 33, elements: 32, .. })));
        assert!(matches!(decode_refused, Err(NumericError::InputNotElementMultiple { found: 33, elements: 32, .. })));
    }

    #[test]
    fn a_respelled_layout_takes_the_layout_as_data_route() {
        let respelled = FloatLayout {
            bias_override: Some(FloatLayout::F16.bias()),
            ..FloatLayout::F16
        };

        let codec = FormatCodec::scalar(&respelled).expect("a respelled f16 resolves");

        assert!(matches!(codec.route, Route::Layout(_)));
    }

    #[test]
    fn a_layout_the_slice_codec_cannot_execute_does_not_resolve() {
        assert!(FormatCodec::scalar(&FloatLayout::INT64).is_none());
    }

    #[test]
    fn a_scalar_is_a_block_of_one_element_and_a_block_format_packs_whole_blocks() {
        let scalar = FormatCodec::scalar(&FloatLayout::BF16).expect("bf16 resolves");
        let block = FormatCodec::for_format(&NumericFormat::Q4_0).expect("q4_0 resolves");

        assert_eq!((scalar.elements(), scalar.block_bytes()), (1, 2));
        assert_eq!(scalar.packed_len(7), Some(14));
        assert_eq!((block.elements(), block.block_bytes()), (32, 18));
        assert_eq!(block.packed_len(64), Some(36));
        assert_eq!(block.packed_len(33), None);
    }

    #[test]
    fn a_block_error_names_the_format_that_raised_it() {
        let codec = FormatCodec::for_format(&NumericFormat::Q4_0).expect("q4_0 resolves");
        let mut out = [0u8; 18];

        let refused = codec.encode(&kv_values(33), &mut out);

        assert_eq!(
            refused,
            Err(NumericError::InputNotElementMultiple { format: "q4_0", found: 33, elements: 32 })
        );
        assert_eq!(codec.format_name(), "q4_0");
    }

    #[test]
    fn a_block_format_with_no_encode_names_itself_in_the_unsupported_encode_error() {
        let codec = FormatCodec::for_format(&NumericFormat::Mxfp4).expect("mxfp4 resolves");
        let mut out = [0u8; 17];

        let refused = codec.encode(&kv_values(32), &mut out);

        assert!(!codec.can_encode(), "mxfp4 resolves for decode only");
        assert_eq!(
            refused,
            Err(NumericError::UnsupportedEncode {
                format: "mxfp4",
                reason: "no compile-time encode for this layout's fit",
            })
        );
    }

    #[test]
    fn decoding_a_partial_block_names_the_format_that_raised_it() {
        let codec = FormatCodec::for_format(&NumericFormat::Q4_0).expect("q4_0 resolves");
        let mut slots = [MaybeUninit::<f32>::uninit(); 32];

        let refused = codec.decode(&[0u8; 17], &mut slots);

        assert_eq!(
            refused,
            Err(NumericError::InputNotBlockMultiple { format: "q4_0", found: 17, block_bytes: 18 })
        );
    }

    const FORMAT_NAMES: [&str; 4] = ["Float32", "Float16", "BFloat16", "BFloat8"];

    include!("../tests/support/production_text.rs");

    fn identifiers_after<'text>(text: &'text str, prefix: &str) -> Vec<&'text str> {
        text.match_indices(prefix)
            .map(|(offset, _)| {
                let tail = &text[offset + prefix.len()..];
                let end = tail
                    .find(|character: char| !character.is_alphanumeric() && character != '_')
                    .unwrap_or(tail.len());
                &tail[..end]
            })
            .collect()
    }

    fn format_name_tokens(text: &str) -> Vec<String> {
        ["NumericFormat::", "Codec::"]
            .into_iter()
            .flat_map(|prefix| {
                identifiers_after(text, prefix)
                    .into_iter()
                    .filter(|name| FORMAT_NAMES.contains(name))
                    .map(move |name| format!("{prefix}{name}"))
            })
            .collect()
    }

    fn layout_constants(text: &str) -> Vec<&str> {
        identifiers_after(text, "FloatLayout::")
            .into_iter()
            .filter(|name| name.starts_with(|first: char| first.is_ascii_uppercase()))
            .collect()
    }

    #[test]
    fn the_format_codec_and_the_layout_codec_name_no_format_and_one_source_layout() {
        let codec = production_of(include_str!("float_layout_codec.rs"));
        let bridge = production_of(include_str!("format_codec.rs"));

        assert_eq!(format_name_tokens(&codec), Vec::<String>::new());
        assert_eq!(format_name_tokens(&bridge), Vec::<String>::new());
        assert_eq!(
            layout_constants(&codec),
            ["BF32", "F16"],
            "the documented source layout, SOURCE_LAYOUT, and the one the cpu converts natively, NATIVE_HALF_LAYOUT"
        );
        assert_eq!(layout_constants(&bridge), Vec::<&str>::new());
    }

    #[test]
    fn the_scan_sees_a_format_name_when_one_is_present() {
        let planted = "fn f(format: &Codec) -> bool { *format == Codec::BFloat16 || NumericFormat::Float32 == *format }";

        assert_eq!(format_name_tokens(planted), ["NumericFormat::Float32", "Codec::BFloat16"]);
        assert_eq!(layout_constants("FloatLayout::E5M2 FloatLayout::from_name"), ["E5M2"]);
    }

    #[test]
    fn a_wrong_sized_output_is_an_error_not_a_partial_write() {
        let codec = FormatCodec::scalar(&FloatLayout::BF16).expect("bf16 resolves");
        let mut out = [0u8; 3];

        assert!(codec.encode(&[1.0, 2.0], &mut out).is_err());
        assert_eq!(out, [0, 0, 0]);
    }
}
