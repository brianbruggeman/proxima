//! A scalar format resolved once into the two slice conversions that carry it.
//!
//! Teaching pointer: [`crate::float_layout_codec`] holds the one slice loop (`encode_body`,
//! `decode_body`); [`ScalarCodec::resolve`] instantiates it once per row of the named-layout table
//! ([`crate::float_layout::named_layout_table`]) with the layout as a literal constant, so every
//! shift, mask and magic number in the instantiation is folded from the layout's fields. Resolve at
//! allocation time and keep the `Copy` value: a call is then one indirect jump, where
//! [`crate::encode::encode_blocks`] re-matches the format on every call. A layout without a name
//! resolves to the same loop with the layout as data, so every supported layout has a codec.

use core::mem::MaybeUninit;

use crate::error::NumericError;
use crate::float_layout::{FloatLayout, named_layout_table};
use crate::float_layout_codec::{
    decode_body, dispatch_named_layout, element_bytes, encode_body, is_storable,
};
use crate::format::NumericFormat;

type EncodeFn = fn(&[f32], &mut [u8]);
type DecodeFn = fn(&[u8], &mut [MaybeUninit<f32>]);

#[derive(Clone, Copy)]
enum Route {
    Named { encode: EncodeFn, decode: DecodeFn },
    Layout(FloatLayout),
}

/// The encode and decode of one scalar layout: plain function pointers to instantiations with the
/// layout as a literal for a named layout, the layout as data for any other. No allocation, no trait
/// object.
///
/// Example: a 16-bit layout storing 512 values takes 1 KiB of output.
#[derive(Clone, Copy)]
pub struct ScalarCodec {
    route: Route,
    element_bytes: usize,
}

impl ScalarCodec {
    /// The codec for a layout the slice codec can execute: the named instantiation when the layout
    /// has a name, the layout-as-data loop otherwise; `None` for a layout wider than 32 bits or empty.
    #[must_use]
    pub fn resolve(layout: &FloatLayout) -> Option<Self> {
        named_layout_table!(dispatch_named_layout!(
            *layout,
            NAMED => Some(Self {
                route: Route::Named {
                    encode: |values, out| encode_body(NAMED, values, out),
                    decode: |data, out| {
                        decode_body(NAMED, data, out, |slot, value| {
                            slot.write(value);
                        });
                    },
                },
                element_bytes: element_bytes(&NAMED),
            }),
            else is_storable(layout).then(|| Self {
                route: Route::Layout(*layout),
                element_bytes: element_bytes(layout),
            }),
        ))
    }

    /// The codec for a scalar [`NumericFormat`], `None` for a block or level format.
    #[must_use]
    pub fn for_format(format: &NumericFormat) -> Option<Self> {
        match format {
            NumericFormat::Scalar(layout) => Self::resolve(layout),
            _ => None,
        }
    }

    /// Stored bytes per element.
    #[must_use]
    pub const fn element_bytes(&self) -> usize {
        self.element_bytes
    }

    /// Encodes `values` into `out`, little-endian, one element per value.
    ///
    /// # Errors
    /// [`NumericError::OutputSizeMismatch`] when `out` is not `values.len()` elements.
    #[must_use = "a size mismatch leaves the output unwritten"]
    pub fn encode(&self, values: &[f32], out: &mut [u8]) -> Result<(), NumericError> {
        let expected = values.len() * self.element_bytes;
        if out.len() != expected {
            return Err(NumericError::OutputSizeMismatch {
                found: out.len(),
                expected,
            });
        }
        match self.route {
            Route::Named { encode, .. } => encode(values, out),
            Route::Layout(layout) => encode_body(layout, values, out),
        }
        Ok(())
    }

    /// Decodes `data` into the caller-owned, possibly uninitialised `out`, initialising every slot.
    ///
    /// # Errors
    /// [`NumericError::OutputSizeMismatch`] when `out` is not `data.len() / element_bytes` slots;
    /// nothing is written in that case.
    #[must_use = "a size mismatch leaves the output uninitialised"]
    pub fn decode(&self, data: &[u8], out: &mut [MaybeUninit<f32>]) -> Result<(), NumericError> {
        if data.len() != out.len() * self.element_bytes {
            return Err(self.decode_mismatch(data.len(), out.len()));
        }
        match self.route {
            Route::Named { decode, .. } => decode(data, out),
            Route::Layout(layout) => decode_body(layout, data, out, |slot, value| {
                slot.write(value);
            }),
        }
        Ok(())
    }

    #[cold]
    fn decode_mismatch(&self, data_len: usize, slots: usize) -> NumericError {
        NumericError::OutputSizeMismatch {
            found: slots,
            expected: data_len / self.element_bytes,
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
            let codec = ScalarCodec::resolve(&layout).expect("a named layout resolves");
            let mut expected = vec![0u8; values.len() * codec.element_bytes()];
            let mut resolved = vec![0u8; expected.len()];

            encode_blocks(&format, &values, &mut expected).expect("encode_blocks encodes");
            codec.encode(&values, &mut resolved).expect("sizes match");

            assert_eq!(resolved, expected, "{layout:?}");
        }
    }

    #[test]
    fn resolved_decode_refuses_a_slot_count_that_is_not_the_element_count() {
        let values = kv_values(1000);
        let codec = ScalarCodec::resolve(&FloatLayout::F16).expect("f16 resolves");
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
        let named = ScalarCodec::resolve(&FloatLayout::F16).expect("f16 resolves");
        let respelled = ScalarCodec::resolve(&unnamed).expect("an unnamed supported layout resolves");
        let mut from_named = vec![0u8; values.len() * 2];
        let mut from_respelled = vec![0u8; values.len() * 2];

        named.encode(&values, &mut from_named).expect("sizes match");
        respelled.encode(&values, &mut from_respelled).expect("sizes match");

        assert_eq!(from_respelled, from_named);
    }

    #[test]
    fn a_layout_the_slice_codec_cannot_execute_does_not_resolve() {
        assert!(ScalarCodec::resolve(&FloatLayout::INT64).is_none());
    }

    const FORMAT_NAMES: [&str; 4] = ["Float32", "Float16", "BFloat16", "BFloat8"];

    fn production_text(source: &str) -> &str {
        source.split("#[cfg(test)]").next().unwrap_or(source)
    }

    fn identifiers_after<'text>(text: &'text str, prefix: &str) -> Vec<&'text str> {
        text.match_indices(prefix)
            .map(|(at, _)| {
                let tail = &text[at + prefix.len()..];
                let end = tail.find(|c: char| !c.is_alphanumeric() && c != '_').unwrap_or(tail.len());
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
            .filter(|name| name.starts_with(|c: char| c.is_ascii_uppercase()))
            .collect()
    }

    #[test]
    fn the_scalar_codec_and_the_layout_codec_name_no_format_and_one_source_layout() {
        let codec = production_text(include_str!("float_layout_codec.rs"));
        let bridge = production_text(include_str!("scalar_codec.rs"));

        assert_eq!(format_name_tokens(codec), Vec::<String>::new());
        assert_eq!(format_name_tokens(bridge), Vec::<String>::new());
        assert_eq!(layout_constants(codec), ["BF32"], "the documented source layout, SOURCE_LAYOUT");
        assert_eq!(layout_constants(bridge), Vec::<&str>::new());
    }

    #[test]
    fn the_scan_sees_a_format_name_when_one_is_present() {
        let planted = "fn f(format: &Codec) -> bool { *format == Codec::BFloat16 || NumericFormat::Float32 == *format }";

        assert_eq!(format_name_tokens(planted), ["NumericFormat::Float32", "Codec::BFloat16"]);
        assert_eq!(layout_constants("FloatLayout::E5M2 FloatLayout::from_name"), ["E5M2"]);
    }

    #[test]
    fn a_wrong_sized_output_is_an_error_not_a_partial_write() {
        let codec = ScalarCodec::resolve(&FloatLayout::BF16).expect("bf16 resolves");
        let mut out = [0u8; 3];

        assert!(codec.encode(&[1.0, 2.0], &mut out).is_err());
        assert_eq!(out, [0, 0, 0]);
    }
}
