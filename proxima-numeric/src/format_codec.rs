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

use core::mem::MaybeUninit;

use crate::block_codec::{DecodeFn, EncodeFn};
use crate::block_layout::BlockLayout;
use crate::error::NumericError;
use crate::float_layout::{FloatLayout, named_layout_table};
use crate::float_layout_codec::{
    decode_body, dispatch_named_layout, element_bytes, encode_body, is_storable,
};
use crate::format::NumericFormat;

const UNNAMED: &str = "unnamed";

#[derive(Clone, Copy)]
enum Route {
    Named {
        encode: Option<EncodeFn>,
        decode: DecodeFn,
    },
    Layout(FloatLayout),
}

/// The encode and decode of one numeric format: plain function pointers to table instantiations
/// with the layout as a literal (a named scalar layout or a table block layout), or the scalar
/// layout as data for any other storable layout. No allocation, no trait object.
///
/// A scalar is a block of one element, so one type and one size check serve both: a call checks
/// whole blocks and exact sizes, then makes one indirect call.
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
        Self {
            route: Route::Named { encode, decode },
            elements,
            block_bytes,
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
        !matches!(self.route, Route::Named { encode: None, .. })
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

    /// Encodes whole blocks of `values` into `out`, little-endian.
    ///
    /// # Errors
    /// [`NumericError::UnsupportedEncode`] for a layout with no encode instantiation,
    /// [`NumericError::InputNotElementMultiple`] when `values` is not whole blocks,
    /// [`NumericError::OutputSizeMismatch`] when `out` is not exactly the packed byte count; the
    /// output is unwritten in each case.
    #[must_use = "an error leaves the output unwritten"]
    pub fn encode(&self, values: &[f32], out: &mut [u8]) -> Result<(), NumericError> {
        match self.route {
            Route::Named { encode: Some(run), .. } => {
                self.check_encode(values.len(), out.len())?;
                run(values, out);
            }
            Route::Named { encode: None, .. } => {
                return Err(NumericError::UnsupportedEncode {
                    format: self.format,
                    reason: "no compile-time encode for this layout's fit",
                });
            }
            Route::Layout(layout) => {
                self.check_encode(values.len(), out.len())?;
                encode_body(layout, values, out);
            }
        }
        Ok(())
    }

    /// Decodes whole blocks of `data` into the caller-owned, possibly uninitialised `out`,
    /// initialising every slot.
    ///
    /// # Errors
    /// [`NumericError::InputNotBlockMultiple`] when `data` is not whole blocks,
    /// [`NumericError::OutputSizeMismatch`] when `out` is not one slot per element; nothing is
    /// written in either case.
    #[must_use = "an error leaves the output uninitialised"]
    pub fn decode(&self, data: &[u8], out: &mut [MaybeUninit<f32>]) -> Result<(), NumericError> {
        self.check_decode(data.len(), out.len())?;
        match self.route {
            Route::Named { decode, .. } => decode(data, out),
            Route::Layout(layout) => decode_body(layout, data, out, |slot, value| {
                slot.write(value);
            }),
        }
        Ok(())
    }

    #[inline(always)]
    fn check_encode(&self, values: usize, out: usize) -> Result<(), NumericError> {
        let Some(expected) = self.packed_len(values) else {
            return Err(NumericError::InputNotElementMultiple {
                format: self.format,
                found: values,
                elements: self.elements,
            });
        };
        if out != expected {
            return Err(NumericError::OutputSizeMismatch { found: out, expected });
        }
        Ok(())
    }

    #[inline(always)]
    fn check_decode(&self, data: usize, out: usize) -> Result<(), NumericError> {
        if self.elements == 1 {
            return if data == out * self.block_bytes {
                Ok(())
            } else {
                Err(self.decode_mismatch(data, out))
            };
        }
        if !data.is_multiple_of(self.block_bytes) {
            return Err(NumericError::InputNotBlockMultiple {
                format: self.format,
                found: data,
                block_bytes: self.block_bytes,
            });
        }
        if out == data / self.block_bytes * self.elements {
            Ok(())
        } else {
            Err(self.decode_mismatch(data, out))
        }
    }

    #[cold]
    fn decode_mismatch(&self, data: usize, slots: usize) -> NumericError {
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
    fn resolved_decode_refuses_a_slot_count_that_is_not_the_element_count() {
        let values = kv_values(1000);
        let codec = FormatCodec::scalar(&FloatLayout::F16).expect("f16 resolves");
        let mut packed = vec![0u8; values.len() * 2];
        codec.encode(&values, &mut packed).expect("sizes match");
        let mut slots = vec![MaybeUninit::<f32>::uninit(); 999];

        let refused = codec.decode(&packed, &mut slots);

        assert_eq!(refused, Err(NumericError::OutputSizeMismatch { found: 999, expected: 1000 }));
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
                assert!(matches!(codec.route, Route::Named { .. }), "{name} must take the named route");
                resolved.push(name);
            }
        }

        assert!(resolved.len() >= 8, "the scan reached only {resolved:?}");
        assert!(resolved.contains(&"bfloat16") && resolved.contains(&"float8_e4m3fn"));
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
        assert_eq!(layout_constants(&codec), ["BF32"], "the documented source layout, SOURCE_LAYOUT");
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
