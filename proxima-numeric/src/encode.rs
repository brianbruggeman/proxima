//! The inverse of [`crate::decode`]: `f32` values into any format this crate can describe.
//!
//! Reading is exact and total; writing is quantization, so every arm either produces bytes or
//! names what it cannot express. The matrix this module fills:
//!
//! - `Scalar(layout)` — one value per element, through [`crate::float_layout_codec::encode`],
//!   whose rounding and special-value policy are the format's own.
//! - `Block(layout)` — a run of codes under a shared absolute-maximum scale, fitted per run and
//!   written back through the layout's own planes. Ported from `proxima-gguf`'s
//!   `block_layout_encode`, which is now a thin adapter over this.
//! - `Levels` with a linear ladder (ternary) — the same machinery: a trit *is* a narrow integer
//!   code under the layout's bias and step.
//! - `Levels(Explicit)` — IQ4_NL's trained 16-entry ladder: the scale is fitted so the block peak
//!   lands on the largest table magnitude, and every code is the nearest table entry.
//!
//! Refused, with the reason: a block layout with sub-blocks or a stored min (one absolute-maximum
//! scale cannot express them), a mixed-radix packed code, a code wider than 12 bits, and a trained
//! grid codebook (quantizing into an IQ2_XS/IQ3_XXS grid needs a search over the codebook, and six
//! of those codebooks are not in this crate at all).

use crate::block_layout::{BlockLayout, Plane, PlaneCursor};
use crate::error::NumericError;
use crate::float_layout::FloatLayout;
use crate::float_layout_codec;
use crate::format::{LevelTable, NumericFormat};

const MAX_CODE_BITS: u8 = 12;
const SUM_BYTES: usize = 2;

/// A layout's code range and value mapping, read once per run of blocks.
#[derive(Debug, Clone, Copy)]
struct Linear {
    lowest: i32,
    highest: i32,
    reach: f32,
    multiplier: f32,
    bias: f32,
    sum_gap: Option<usize>,
}

fn code_bits(layout: &BlockLayout) -> u8 {
    layout.element.low.bits + layout.element.high.map_or(0, |plane| plane.bits)
}

fn code_range(layout: &BlockLayout) -> (i32, i32) {
    (0..1u32 << code_bits(layout)).fold((i32::MAX, i32::MIN), |(lowest, highest), raw| {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the point of the walk is the integer the code decodes to"
        )]
        let code = float_layout_codec::decode(raw, layout.element.code) as i32;
        (lowest.min(code), highest.max(code))
    })
}

fn check_linear(layout: &BlockLayout) -> Result<(), &'static str> {
    if layout.sub.is_some() {
        return Err("sub-blocks carry their own scales");
    }
    if layout.min.is_some() {
        return Err("a stored min is not an absolute-maximum scale");
    }
    if layout.element.packed_radix > 1 {
        return Err("mixed-radix packed codes");
    }
    if layout.element.code.exponent_bits != 0 || code_bits(layout) > MAX_CODE_BITS {
        return Err("element codes are not narrow integers");
    }
    Ok(())
}

fn touches(plane: &Plane, elements: usize, bytes: core::ops::Range<usize>) -> bool {
    (0..elements).any(|index| bytes.contains(&(plane.address(index) / 8)))
}

/// llama's `Q8_1` carries `s = d * sum(q)` in the two bytes after the scale; a layout whose planes
/// leave those bytes unused is the one place the consumer reads it.
fn sum_slot(layout: &BlockLayout) -> Option<usize> {
    let gap = usize::from(layout.scale.layout.width()) / 8;
    let elements = usize::from(layout.elements);
    let planes = [Some(layout.element.low), layout.element.high];
    let free = (0..layout.runs()).all(|run| {
        let start = layout.run_scale_byte(run) + gap;
        let bytes = start..start + SUM_BYTES;
        bytes.end <= usize::from(layout.block_bytes)
            && planes
                .iter()
                .flatten()
                .all(|plane| !touches(plane, elements, bytes.clone()))
    });
    free.then_some(gap)
}

fn linear(layout: &BlockLayout) -> Result<Linear, &'static str> {
    check_linear(layout)?;
    let (lowest, highest) = code_range(layout);
    let multiplier = f32::from(layout.multiplier);
    let bias = f32::from(layout.element.bias);
    let value_low = multiplier * lowest as f32 - bias;
    let value_high = multiplier * highest as f32 - bias;
    let reach = value_high.min(-value_low);
    if reach <= 0.0 {
        return Err("no code magnitude on both sides of zero");
    }
    Ok(Linear {
        lowest,
        highest,
        reach,
        multiplier,
        bias,
        sum_gap: sum_slot(layout),
    })
}

impl Linear {
    fn code_for(&self, value: f32, scale: f32) -> i32 {
        let scaled = if scale == 0.0 { 0.0 } else { value / scale };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the quotient is clamped into the code range immediately after"
        )]
        let code = libm::roundf((scaled + self.bias) / self.multiplier) as i32;
        code.clamp(self.lowest, self.highest)
    }

    fn value_of(&self, code: i32) -> f32 {
        self.multiplier * code as f32 - self.bias
    }
}

fn put_bits(plane: &Plane, cursor: &mut PlaneCursor, bits: u32, block: &mut [u8]) {
    let address = cursor.address();
    let mask = (1u32 << plane.bits) - 1;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a code plane is at most 12 bits and never straddles a byte it was not built to"
    )]
    {
        block[address / 8] |= ((bits & mask) << (address % 8)) as u8;
    }
    cursor.advance();
}

fn cursors_of(layout: &BlockLayout) -> [PlaneCursor; 2] {
    let mut cursors = [layout.element.low.cursor(), layout.element.low.cursor()];
    if let Some(high) = &layout.element.high {
        cursors[1] = high.cursor();
    }
    cursors
}

fn put_code(layout: &BlockLayout, cursors: &mut [PlaneCursor; 2], raw: u32, block: &mut [u8]) {
    let element = &layout.element;
    put_bits(&element.low, &mut cursors[0], raw, block);
    if let Some(high) = &element.high {
        put_bits(high, &mut cursors[1], raw >> element.low.bits, block);
    }
}

fn put_scale(layout: &BlockLayout, offset: usize, code: u32, block: &mut [u8]) {
    let bytes = usize::from(layout.scale.layout.width()) / 8;
    for byte in 0..bytes {
        block[offset + byte] = (code >> (8 * byte)) as u8;
    }
}

fn quantize_run(
    layout: &BlockLayout,
    shape: &Linear,
    run: u16,
    input: &[f32],
    cursors: &mut [PlaneCursor; 2],
    block: &mut [u8],
) {
    let peak = input
        .iter()
        .fold(0.0f32, |peak, value| peak.max(value.abs()));
    let scale_code = float_layout_codec::encode(peak / shape.reach, layout.scale.layout);
    let scale = float_layout_codec::decode(scale_code, layout.scale.layout);
    let scale_at = layout.run_scale_byte(run);
    put_scale(layout, scale_at, scale_code, block);
    let mut value_sum = 0.0f32;
    for value in input {
        let code = shape.code_for(*value, scale);
        put_code(
            layout,
            cursors,
            float_layout_codec::encode(code as f32, layout.element.code),
            block,
        );
        value_sum += shape.value_of(code);
    }
    if let Some(gap) = shape.sum_gap {
        let at = scale_at + gap;
        let code = float_layout_codec::encode(scale * value_sum, FloatLayout::F16);
        block[at..at + SUM_BYTES].copy_from_slice(&(code as u16).to_le_bytes());
    }
}

fn quantize_block(
    layout: &BlockLayout,
    shape: &Linear,
    input: &[f32],
    block: &mut [u8],
) -> Result<(), NumericError> {
    if input.iter().any(|value| !value.is_finite()) {
        return Err(NumericError::NonFinite {
            format: format_of(layout),
        });
    }
    block.fill(0);
    let mut cursors = cursors_of(layout);
    for (run, group) in input
        .chunks(usize::from(layout.run_elements()))
        .enumerate()
    {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a block holds at most 32 runs and each is at most 256 elements"
        )]
        quantize_run(layout, shape, run as u16, group, &mut cursors, block);
    }
    Ok(())
}

/// A block whose codes index a trained ladder: the scale places the block peak on `reach`, then
/// every code is the table entry nearest the scaled value.
fn quantize_ladder_block(
    layout: &BlockLayout,
    table: &[i8],
    reach: f32,
    input: &[f32],
    block: &mut [u8],
) -> Result<(), NumericError> {
    if input.iter().any(|value| !value.is_finite()) {
        return Err(NumericError::NonFinite {
            format: format_of(layout),
        });
    }
    block.fill(0);
    let mut cursors = cursors_of(layout);
    for (run, group) in input
        .chunks(usize::from(layout.run_elements()))
        .enumerate()
    {
        let peak = group
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()));
        let scale_code = float_layout_codec::encode(peak / reach, layout.scale.layout);
        let scale = float_layout_codec::decode(scale_code, layout.scale.layout);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a block holds at most 32 runs"
        )]
        let scale_at = layout.run_scale_byte(run as u16);
        put_scale(layout, scale_at, scale_code, block);
        for value in group {
            let target = if scale == 0.0 { 0.0 } else { value / scale };
            let code = nearest_level(table, target);
            put_code(layout, &mut cursors, code, block);
        }
    }
    Ok(())
}

fn format_of(layout: &BlockLayout) -> &'static str {
    NumericFormat::ALL
        .iter()
        .find(|format| format.block_layout().as_ref() == Some(layout))
        .and_then(NumericFormat::name)
        .unwrap_or("block")
}

/// The largest value magnitude `layout` reaches on both sides of zero, in units of its scale: the
/// divisor of the block maximum, and the test of whether this module encodes `layout` at all.
///
/// # Errors
/// [`NumericError::UnsupportedEncode`] naming what the layout has that a shared absolute-maximum
/// scale cannot express.
pub fn reach(layout: &BlockLayout) -> Result<f32, NumericError> {
    linear(layout)
        .map(|shape| shape.reach)
        .map_err(|reason| NumericError::UnsupportedEncode {
            format: format_of(layout),
            reason,
        })
}

/// Encode `values` into `format`, writing `bytes_per_block` bytes per `elements_per_block` values,
/// and returns the block count.
///
/// One value per block for a scalar format; a whole block of `f32` for every block and level-set
/// format whose geometry a shared absolute-maximum scale can express.
///
/// # Errors
/// [`NumericError::InputNotElementMultiple`] when `values` is not a whole number of blocks,
/// [`NumericError::OutputSizeMismatch`] when `out` is not exactly the packed byte count,
/// [`NumericError::UnsupportedEncode`] for a format this module cannot quantize, and
/// [`NumericError::NonFinite`] for a NaN or infinite block element.
pub fn encode_blocks(
    format: &NumericFormat,
    values: &[f32],
    out: &mut [u8],
) -> Result<usize, NumericError> {
    let elements = usize::from(format.elements_per_block());
    let block_bytes = usize::from(format.bytes_per_block());
    if elements == 0 || block_bytes == 0 {
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
    let block_count = values.len() / elements;
    let expected = block_count * block_bytes;
    if out.len() != expected {
        return Err(NumericError::OutputSizeMismatch {
            found: out.len(),
            expected,
        });
    }
    crate::decode::geometry_fits(format)?;
    match format {
        NumericFormat::Scalar(layout) => {
            float_layout_codec::layout_supported(layout)?;
            let bytes = usize::from(layout.width().div_ceil(8));
            for (value, chunk) in values.iter().zip(out.chunks_exact_mut(block_bytes)) {
                let code = float_layout_codec::encode(*value, *layout);
                for (byte, slot) in chunk.iter_mut().take(bytes).enumerate() {
                    *slot = (code >> (8 * byte)) as u8;
                }
            }
            Ok(block_count)
        }
        NumericFormat::Block(layout) => {
            let shape = linear(layout).map_err(|reason| NumericError::UnsupportedEncode {
                format: format.name().unwrap_or("block"),
                reason,
            })?;
            for (group, block) in values.chunks(elements).zip(out.chunks_mut(block_bytes)) {
                quantize_block(layout, &shape, group, block)?;
            }
            Ok(block_count)
        }
        NumericFormat::Levels(levels) => {
            let name = format.name().unwrap_or("levels");
            let Some(layout) = levels.packing() else {
                return Err(NumericError::UnsupportedEncode {
                    format: name,
                    reason: "a trained codebook needs a search over its table",
                });
            };
            match levels.levels {
                LevelTable::Linear { .. } => {
                    let shape = linear(&layout).map_err(|reason| NumericError::UnsupportedEncode {
                        format: name,
                        reason,
                    })?;
                    for (group, block) in values.chunks(elements).zip(out.chunks_mut(block_bytes))
                    {
                        quantize_block(&layout, &shape, group, block)?;
                    }
                    Ok(block_count)
                }
                LevelTable::Explicit(table) => {
                    let reach = ladder_reach(table);
                    for (group, block) in values.chunks(elements).zip(out.chunks_mut(block_bytes))
                    {
                        quantize_ladder_block(&layout, table, reach, group, block)?;
                    }
                    Ok(block_count)
                }
                LevelTable::Iq2XsGrid | LevelTable::Iq3XxsGrid | LevelTable::External => {
                    Err(NumericError::UnsupportedEncode {
                        format: name,
                        reason: "this codebook has no inverse in this crate",
                    })
                }
            }
        }
    }
}

/// The magnitude a trained ladder reaches on both sides of zero, in table units.
fn ladder_reach(values: &[i8]) -> f32 {
    let low = values.iter().copied().min().unwrap_or(0) as f32;
    let high = values.iter().copied().max().unwrap_or(0) as f32;
    high.min(-low).max(1.0)
}

fn nearest_level(values: &[i8], target: f32) -> u32 {
    let mut best = 0usize;
    let mut best_distance = f32::INFINITY;
    for (index, level) in values.iter().enumerate() {
        let distance = (f32::from(*level) - target).abs();
        if distance < best_distance {
            best_distance = distance;
            best = index;
        }
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a trained ladder has at most 256 entries"
    )]
    {
        best as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_layout::BlockLayout;
    use crate::decode::decode_blocks;

    /// half of a binary16 ulp at 1.0: every stored scale is a binary16, so the scale itself is
    /// only known to this much and every value inherits that relative error
    const F16_EPSILON: f32 = 1.0 / 1024.0;

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
                4.0 * signed * (1.0 + (index % 7) as f32)
            })
            .collect()
    }

    fn packed(format: &NumericFormat, input: &[f32]) -> std::vec::Vec<u8> {
        let elements = usize::from(format.elements_per_block());
        let bytes = usize::from(format.bytes_per_block());
        let mut out = std::vec![0u8; input.len() / elements * bytes];
        encode_blocks(format, input, &mut out).expect("a supported format encodes");
        out
    }

    fn decoded(format: &NumericFormat, packed: &[u8]) -> std::vec::Vec<f32> {
        let elements = usize::from(format.elements_per_block());
        let bytes = usize::from(format.bytes_per_block());
        let mut out = std::vec![0.0f32; packed.len() / bytes * elements];
        decode_blocks(format, packed, &mut out).expect("the bytes this encoder wrote decode");
        out
    }

    /// The scale the decoder reads back: the quotient stored as a binary16.
    fn stored_scale(peak: f32, reach: f32) -> f32 {
        let code = float_layout_codec::encode(peak / reach, FloatLayout::F16);
        float_layout_codec::decode(code, FloatLayout::F16)
    }

    fn peak_of(block: &[f32]) -> f32 {
        block.iter().fold(0.0f32, |peak, value| peak.max(value.abs()))
    }

    #[test]
    fn a_scalar_format_round_trips_every_code_its_layout_holds() {
        let format = NumericFormat::BFloat16;
        let input: std::vec::Vec<f32> = (0..512)
            .map(|step| crate::float_layout_codec::decode(step % 0x8000, FloatLayout::BF16))
            .collect();
        let bytes = packed(&format, &input);
        assert_eq!(bytes.len(), 512 * 2, "one binary16 per value");
        for (original, restore) in input.iter().zip(decoded(&format, &bytes)) {
            if original.is_nan() {
                assert!(restore.is_nan(), "a stored nan decodes to a nan");
            } else {
                assert_eq!(original.to_bits(), restore.to_bits());
            }
        }
    }

    #[test]
    fn q4_0_round_trips_within_half_a_code_of_its_block_peak() {
        let format = NumericFormat::Q4_0;
        let input = values(32 * 5, 0x5eed);
        let stated = reach(&BlockLayout::Q4_0).expect("q4_0 is a shared-scale layout");
        let restored = decoded(&format, &packed(&format, &input));
        for (originals, decodes) in input.chunks(32).zip(restored.chunks(32)) {
            let peak = peak_of(originals);
            let bound = (peak / stated) / 2.0 + 2.0 * peak * F16_EPSILON;
            for (original, restore) in originals.iter().zip(decodes) {
                assert!(
                    (original - restore).abs() <= bound,
                    "{original} decoded to {restore}, bound {bound}"
                );
            }
        }
    }

    #[test]
    fn encoding_a_decoded_q4_0_run_reproduces_its_own_bytes() {
        let format = NumericFormat::Q4_0;
        let first = packed(&format, &values(32 * 4, 0xfeed));
        let second = packed(&format, &decoded(&format, &first));
        assert_eq!(first, second, "the quantizer is idempotent on its own grid");
    }

    #[test]
    fn ternary_round_trips_onto_its_three_levels() {
        let format = NumericFormat::Tq20;
        let input = values(256 * 2, 0x7a1);
        let restored = decoded(&format, &packed(&format, &input));
        for (originals, decodes) in input.chunks(256).zip(restored.chunks(256)) {
            let peak = peak_of(originals);
            let bound = peak / 2.0 + 2.0 * peak * F16_EPSILON;
            let scale = stored_scale(peak, 1.0);
            for (original, restore) in originals.iter().zip(decodes) {
                assert!(
                    (original - restore).abs() <= bound,
                    "{original} decoded to {restore}, bound {bound}"
                );
                let units = restore / scale;
                assert!(
                    (units - units.round()).abs() < 1.0e-5,
                    "a trit lands on a whole level, not {units}"
                );
            }
        }
    }

    #[test]
    fn iq4_nl_round_trips_through_its_trained_ladder() {
        let format = NumericFormat::Iq4Nl;
        let input = values(32 * 4, 0x1a4);
        let table = crate::tables::KVALUES_IQ4_NL;
        let restored = decoded(&format, &packed(&format, &input));
        for (originals, decodes) in input.chunks(32).zip(restored.chunks(32)) {
            let peak = peak_of(originals);
            let scale = stored_scale(peak, ladder_reach(&table));
            let bound = 12.5 * scale + 2.0 * peak * F16_EPSILON;
            for (original, restore) in originals.iter().zip(decodes) {
                assert!(
                    (original - restore).abs() <= bound,
                    "{original} decoded to {restore}, bound {bound}"
                );
                let units = restore / scale;
                assert!(
                    table
                        .iter()
                        .any(|level| (f32::from(*level) - units).abs() < 1.0e-5),
                    "a code lands on a trained level, not {units}"
                );
            }
        }
    }

    #[test]
    fn layouts_a_shared_scale_cannot_express_are_a_typed_error() {
        for (layout, reason) in [
            (
                BlockLayout::Q4_1,
                "a stored min is not an absolute-maximum scale",
            ),
            (BlockLayout::Q4_K, "sub-blocks carry their own scales"),
        ] {
            let error = reach(&layout).expect_err("not a shared-scale linear layout");
            assert_eq!(
                error,
                NumericError::UnsupportedEncode {
                    format: format_of(&layout),
                    reason,
                }
            );
        }
    }

    #[test]
    fn a_trained_grid_has_no_encoder_in_this_crate() {
        for format in [NumericFormat::Iq2Xs, NumericFormat::Iq3Xxs] {
            let elements = usize::from(format.elements_per_block());
            let mut out = std::vec![0u8; usize::from(format.bytes_per_block())];
            let error = encode_blocks(&format, &values(elements, 3), &mut out)
                .expect_err("a grid code needs a search over its codebook");
            assert!(
                matches!(error, NumericError::UnsupportedEncode { .. }),
                "{error:?}"
            );
        }
    }

    #[test]
    fn a_non_finite_block_element_is_a_typed_error() {
        let format = NumericFormat::Q8_0;
        let mut input = values(32, 3);
        input[5] = f32::NAN;
        let error = encode_blocks(&format, &input, &mut std::vec![0u8; 34])
            .expect_err("nan has no code");
        assert_eq!(
            error,
            NumericError::NonFinite {
                format: format_of(&BlockLayout::Q8_0),
            }
        );
    }

    #[test]
    fn a_layout_that_cannot_address_its_own_block_is_refused_not_indexed() {
        let mut lying = BlockLayout::Q4_0;
        lying.block_bytes = 1;
        let format = NumericFormat::Block(lying);
        let error = encode_blocks(&format, &values(32, 11), &mut std::vec![0u8; 1])
            .expect_err("32 nibbles do not fit in one byte");
        assert!(
            matches!(error, NumericError::InvalidGeometry { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn a_partial_block_and_a_wrong_sized_output_are_typed() {
        let format = NumericFormat::Q8_0;
        let short = encode_blocks(&format, &values(31, 9), &mut std::vec![0u8; 34])
            .expect_err("31 values is not a whole block");
        assert_eq!(
            short,
            NumericError::InputNotElementMultiple {
                format: "q8_0",
                found: 31,
                elements: 32,
            }
        );
        let mis_sized = encode_blocks(&format, &values(32, 9), &mut std::vec![0u8; 33])
            .expect_err("q8_0 is 34 bytes per 32 values");
        assert_eq!(
            mis_sized,
            NumericError::OutputSizeMismatch {
                found: 33,
                expected: 34,
            }
        );
    }
}
