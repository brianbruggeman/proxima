//! The value the conversion surface carries: a [`NumericFormat`] and the wide numbers stored in it.
//!
//! A Rust primitive cannot be the value, because most of the formats this workspace stores are not
//! one number — a block is 32 to 256 codes under a scale, and a level-set format's codes index a
//! trained ladder. [`Numeric`] carries the format beside the wide values that format decodes to, so
//! one type spans every arm and either direction works:
//!
//! ```text
//! bytes --Numeric::decode--> Numeric --encode--> bytes        (the same format)
//! bytes --Numeric::decode--> Numeric --to--------> bytes        (any other format)
//! ```
//!
//! Zero allocation: the decoded values live in a caller-owned `&mut [f32]` and the packed bytes in
//! a caller-owned `&mut [u8]`, the same contract [`crate::decode`] and [`crate::encode`] already
//! keep. The wide values are exactly what the decoder produced for the format, so they are the
//! interchange between two formats; narrowing them is where [`Rounding`] and [`Overflow`] are
//! named, and a block target quantizes from them with its own fitted scale.
//!
//! Teaching pointer: [`Numeric::decode`] wraps [`crate::decode::decode_blocks`], [`Numeric::to`]
//! wraps [`crate::encode::encode_blocks`] and [`crate::convert::narrow_to_layout`]; those three
//! primitives are the whole surface, and this type exists so a caller holding "a number and the
//! format it is stored in" does not have to thread the format through by hand.

use crate::convert::{Overflow, Rounding};
use crate::decode::decode_blocks;
use crate::encode::encode_blocks;
use crate::error::NumericError;
use crate::float_layout::FloatLayout;
use crate::float_layout_codec;
use crate::format::NumericFormat;

/// A value and the format it is stored in.
///
/// `values` is the format's own numbers, widened exactly: one value for a scalar format, one
/// block's worth for a block or level-set format. It must be a whole number of blocks.
///
/// ```
/// use proxima_numeric::{Numeric, NumericFormat, Overflow, Rounding, decode_blocks, encode_blocks};
///
/// let source = NumericFormat::Q4_0;
/// let target = NumericFormat::Q8_0;
/// let weights: Vec<f32> = (0..32).map(|step| step as f32 / 8.0 - 2.0).collect();
///
/// let mut packed = vec![0u8; 18];
/// encode_blocks(&source, &weights, &mut packed).expect("32 values are one q4_0 block");
///
/// let mut wide = vec![0.0f32; 32];
/// let (value, blocks) =
///     Numeric::decode(&source, &packed, &mut wide).expect("the bytes decode");
/// assert_eq!(blocks, 1);
///
/// let mut finer = vec![0u8; 34];
/// value
///     .to(&target, Rounding::NearestTiesEven, Overflow::Format, &mut finer)
///     .expect("q8_0 takes every q4_0 decode");
///
/// let mut restored = vec![0.0f32; 32];
/// decode_blocks(&target, &finer, &mut restored).expect("q8_0 decodes");
/// assert!(restored
///     .iter()
///     .zip(&weights)
///     .all(|(back, from)| (back - from).abs() < 0.3));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Numeric<'a> {
    format: &'a NumericFormat,
    values: &'a [f32],
}

impl<'a> Numeric<'a> {
    /// Wrap values that are already in `format`, checking they are a whole number of blocks.
    ///
    /// # Errors
    /// [`NumericError::InputNotElementMultiple`] when `values` is not a whole number of blocks,
    /// [`NumericError::InvalidGeometry`] for a format with no elements or bytes. Whether the
    /// format has an encoder at all is [`Self::encode`]'s answer, not this constructor's: a sound
    /// block layout (Q4_K) can still be one no shared scale expresses.
    pub fn new(format: &'a NumericFormat, values: &'a [f32]) -> Result<Self, NumericError> {
        let elements = usize::from(format.elements_per_block());
        if elements == 0 || usize::from(format.bytes_per_block()) == 0 {
            return Err(NumericError::InvalidGeometry {
                format: format.name().unwrap_or("unnamed"),
                reason: "a format with no elements or bytes",
            });
        }
        if !values.len().is_multiple_of(elements) {
            return Err(NumericError::InputNotElementMultiple {
                format: format.name().unwrap_or("unnamed"),
                found: values.len(),
                elements,
            });
        }
        Ok(Self { format, values })
    }

    /// Wrap one scalar in a scalar format.
    ///
    /// # Errors
    /// [`NumericError::InvalidGeometry`] when `format` is not scalar: a block value is one block,
    /// so it needs [`Numeric::new`] with a whole block of values.
    pub fn one(format: &'a NumericFormat, value: &'a f32) -> Result<Self, NumericError> {
        if usize::from(format.elements_per_block()) != 1 {
            return Err(NumericError::InvalidGeometry {
                format: format.name().unwrap_or("unnamed"),
                reason: "one value is not a whole block",
            });
        }
        Self::new(format, core::slice::from_ref(value))
    }

    /// Decode a payload of `format` into `out`, then carry it.
    ///
    /// `out` must hold exactly one `f32` per element. Returns the value and the number of blocks;
    /// the value borrows `out` for as long as it lives.
    ///
    /// # Errors
    /// Whatever [`crate::decode::decode_blocks`] reports.
    pub fn decode(
        format: &'a NumericFormat,
        bytes: &[u8],
        out: &'a mut [f32],
    ) -> Result<(Self, usize), NumericError> {
        let blocks = decode_blocks(format, bytes, out)?;
        let values: &'a [f32] = out;
        Ok((Self { format, values }, blocks))
    }

    /// The format these values are stored in.
    #[must_use]
    pub const fn format(&self) -> &'a NumericFormat {
        self.format
    }

    /// The values themselves, exactly as the format's decoder produced them.
    #[must_use]
    pub const fn values(&self) -> &'a [f32] {
        self.values
    }

    /// How many elements the value covers.
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.values.len()
    }

    /// The one number a scalar value stands for.
    #[must_use]
    pub fn scalar(&self) -> Option<f32> {
        match self.values {
            [value] => Some(*value),
            _ => None,
        }
    }

    /// Write the value back in its own format; returns the block count.
    ///
    /// # Errors
    /// Whatever [`crate::encode::encode_blocks`] reports: a format with no encoder in this crate,
    /// a total value count that is not a whole number of blocks, or an `out` that is not the packed
    /// byte count.
    pub fn encode(&self, out: &mut [u8]) -> Result<usize, NumericError> {
        encode_blocks(self.format, self.values, out)
    }

    /// Write the value in `target`, under an explicit rounding and overflow policy.
    ///
    /// The wide values are exact in `self.format` and are narrowed once, into `target`. A scalar
    /// target narrows each value through [`crate::convert::narrow_to_layout`]'s arithmetic with the
    /// named policy; a block or level-set target quantizes a block at a time through
    /// [`crate::encode::encode_blocks`].
    ///
    /// # Errors
    /// [`NumericError::NonFinite`] when a `NaN` reaches a target format that cannot store one, and
    /// whatever the target format's encoder reports.
    pub fn to(
        &self,
        target: &NumericFormat,
        rounding: Rounding,
        overflow: Overflow,
        out: &mut [u8],
    ) -> Result<usize, NumericError> {
        match target {
            NumericFormat::Scalar(layout) => {
                write_scalars(self.values, layout, rounding, overflow, out)
            }
            NumericFormat::Block(_) | NumericFormat::Levels(_) => {
                encode_blocks(target, self.values, out)
            }
        }
    }
}

fn write_scalars(
    values: &[f32],
    layout: &FloatLayout,
    rounding: Rounding,
    overflow: Overflow,
    out: &mut [u8],
) -> Result<usize, NumericError> {
    float_layout_codec::layout_supported(layout)?;
    let bytes = usize::from(layout.width().div_ceil(8));
    let expected = values.len() * bytes;
    if out.len() != expected {
        return Err(NumericError::OutputSizeMismatch {
            found: out.len(),
            expected,
        });
    }
    let mut nan_count = 0u64;
    for (value, chunk) in values.iter().zip(out.chunks_exact_mut(bytes)) {
        let code = float_layout_codec::encode_with(*value, *layout, rounding, overflow, &mut nan_count);
        for (byte, slot) in chunk.iter_mut().enumerate() {
            *slot = (code >> (8 * byte)) as u8;
        }
    }
    if nan_count > 0 {
        return Err(NumericError::NonFinite {
            format: layout.name().unwrap_or("scalar"),
        });
    }
    Ok(values.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_layout::BlockLayout;
    use crate::encode::reach;

    fn values(count: usize, seed: u64) -> std::vec::Vec<f32> {
        let mut state = seed | 1;
        (0..count)
            .map(|index| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut mixed = state;
                mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                mixed ^= mixed >> 31;
                let unit = (mixed >> 11) as f32 / (1u64 << 53) as f32;
                let signed = unit * 2.0 - 1.0;
                2.0 * signed * (1.0 + (index % 5) as f32)
            })
            .collect()
    }

    #[test]
    fn a_q4_0_payload_round_trips_through_the_value() {
        let format = NumericFormat::Q4_0;
        let source = values(32 * 4, 0x5eed);
        let mut packed = std::vec![0u8; 4 * 18];
        encode_blocks(&format, &source, &mut packed).expect("q4_0 encodes");

        let mut wide = std::vec![0.0f32; 32 * 4];
        let (value, blocks) = Numeric::decode(&format, &packed, &mut wide).expect("q4_0 decodes");
        assert_eq!(blocks, 4);
        assert_eq!(value.format(), &format);
        assert_eq!(value.elements(), 128);

        let mut again = std::vec![0u8; 4 * 18];
        assert_eq!(value.encode(&mut again).expect("re-encodes"), 4);
        assert_eq!(again, packed, "a decode then encode is the same bytes");
    }

    #[test]
    fn the_value_converts_across_formats() {
        let from = NumericFormat::Q4_0;
        let to = NumericFormat::Q8_0;
        let source = values(32 * 3, 0xc0de);
        let mut packed = std::vec![0u8; 3 * 18];
        encode_blocks(&from, &source, &mut packed).expect("q4_0 encodes");

        let mut wide = std::vec![0.0f32; 32 * 3];
        let (value, _) = Numeric::decode(&from, &packed, &mut wide).expect("q4_0 decodes");

        let mut converted = std::vec![0u8; 3 * 34];
        let blocks = value
            .to(&to, Rounding::NearestTiesEven, Overflow::Format, &mut converted)
            .expect("q8_0 takes every q4_0 decode");
        assert_eq!(blocks, 3);

        let mut restored = std::vec![0.0f32; 32 * 3];
        crate::decode::decode_blocks(&to, &converted, &mut restored).expect("q8_0 decodes");
        let step = reach(&BlockLayout::Q8_0).expect("q8_0 is a shared-scale layout");
        for (source_block, restored_block) in wide.chunks(32).zip(restored.chunks(32)) {
            let peak = source_block
                .iter()
                .fold(0.0f32, |peak, number| peak.max(number.abs()));
            let bound = (peak / step) / 2.0 + 0.001 * peak;
            for (from, to) in source_block.iter().zip(restored_block) {
                assert!(
                    (from - to).abs() <= bound,
                    "{from} crossed to {to}, bound {bound}"
                );
            }
        }
    }

    #[test]
    fn scalar_values_become_a_block_in_another_format() {
        let scalars = NumericFormat::Scalar(FloatLayout::BF32);
        let source = values(32, 0xbeef);
        let value = Numeric::new(&scalars, &source).expect("32 scalars are 32 values");
        let mut converted = std::vec![0u8; 18];
        assert_eq!(
            value
                .to(
                    &NumericFormat::Q4_0,
                    Rounding::NearestTiesEven,
                    Overflow::Format,
                    &mut converted
                )
                .expect("one block of scalars is one q4_0 block"),
            1
        );
        let mut direct = std::vec![0u8; 18];
        encode_blocks(&NumericFormat::Q4_0, &source, &mut direct).expect("q4_0 encodes");
        assert_eq!(converted, direct, "the value is not a slower path");
    }

    #[test]
    fn a_scalar_value_narrows_under_the_rounding_it_names() {
        let format = NumericFormat::Scalar(FloatLayout::F16);
        let halfway = 1.0 + 1.0 / 2048.0;
        let value = Numeric::new(&format, std::slice::from_ref(&halfway)).expect("one scalar");

        let mut even = std::vec![0u8; 2];
        value
            .to(&format, Rounding::NearestTiesEven, Overflow::Format, &mut even)
            .expect("a tie rounds to the even code");
        assert_eq!(scalar_of(&format, &even), 1.0);

        let mut away = std::vec![0u8; 2];
        value
            .to(&format, Rounding::AwayFromZero, Overflow::Format, &mut away)
            .expect("a discarded bit rounds away from zero");
        assert_eq!(scalar_of(&format, &away), 1.0 + 1.0 / 1024.0);

        let mut toward = std::vec![0u8; 2];
        value
            .to(&format, Rounding::TowardZero, Overflow::Format, &mut toward)
            .expect("a discarded bit is dropped");
        assert_eq!(scalar_of(&format, &toward), 1.0);
    }

    #[test]
    fn a_value_outside_the_target_saturates_or_follows_the_format() {
        let format = NumericFormat::Scalar(FloatLayout::E5M2);
        let value = Numeric::new(&format, std::slice::from_ref(&1.0e6f32)).expect("one scalar");

        let mut saturated = std::vec![0u8; 1];
        value
            .to(
                &format,
                Rounding::NearestTiesEven,
                Overflow::Saturate,
                &mut saturated,
            )
            .expect("saturate clamps");
        assert_eq!(scalar_of(&format, &saturated), FloatLayout::E5M2.max_normal());

        let mut followed = std::vec![0u8; 1];
        value
            .to(
                &format,
                Rounding::NearestTiesEven,
                Overflow::Format,
                &mut followed,
            )
            .expect("the format has an infinity encoding");
        assert!(scalar_of(&format, &followed).is_infinite());
    }

    #[test]
    fn a_nan_into_a_layout_without_one_is_refused() {
        let format = NumericFormat::Scalar(FloatLayout::INT8);
        let value = Numeric::new(&format, std::slice::from_ref(&f32::NAN)).expect("one scalar");
        let error = value
            .to(
                &format,
                Rounding::NearestTiesEven,
                Overflow::Saturate,
                &mut std::vec![0u8; 1],
            )
            .expect_err("an integer layout cannot store a nan");
        assert_eq!(
            error,
            NumericError::NonFinite {
                format: "int8"
            }
        );
    }

    #[test]
    fn one_value_needs_a_scalar_and_a_partial_block_is_refused() {
        let value = 0.0f32;
        let error = Numeric::one(&NumericFormat::Q4_0, &value)
            .expect_err("a block format is one whole block, not one value");
        assert!(matches!(error, NumericError::InvalidGeometry { .. }), "{error:?}");

        let partial = values(31, 7);
        let error = Numeric::new(&NumericFormat::Q8_0, &partial)
            .expect_err("31 values is not one q8_0 block");
        assert_eq!(
            error,
            NumericError::InputNotElementMultiple {
                format: "q8_0",
                found: 31,
                elements: 32,
            }
        );
    }

    fn scalar_of(format: &NumericFormat, bytes: &[u8]) -> f32 {
        crate::decode::decode_scalar(format, bytes).expect("one scalar decodes")
    }
}
