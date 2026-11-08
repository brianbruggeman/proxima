//! The encoder that inverts [`super::block_layout`]'s walker for the linear layouts: integer
//! element codes under one shared scale, no min and no sub-blocks.
//!
//! Teaching pointer: `proxima_numeric::block_layout::BlockLayout` is the data, the walker in
//! [`super::block_layout`] decodes it, and this module is its inverse, so a format that decodes
//! through the walker encodes here without a per-codec quantizer. The scale is the block's
//! absolute maximum over the largest code magnitude the layout represents on both sides of zero,
//! stored in the layout's scale field and read back before any code is chosen, so the codes
//! round against the scale the decoder will use.

use half::f16;
use proxima_numeric::block_layout::{BlockLayout, Plane, PlaneCursor};
use proxima_numeric::float_layout_codec::{decode, encode};

use crate::quant::QuantError;

const CODEC: &str = "block_layout_encode";
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

fn unsupported(reason: &'static str) -> QuantError {
    QuantError::UnsupportedBlockLayout { reason }
}

fn code_bits(layout: &BlockLayout) -> u8 {
    layout.element.low.bits + layout.element.high.map_or(0, |plane| plane.bits)
}

fn code_range(layout: &BlockLayout) -> (i32, i32) {
    (0..1u32 << code_bits(layout)).fold((i32::MAX, i32::MIN), |(lowest, highest), raw| {
        let code = decode(raw, layout.element.code) as i32;
        (lowest.min(code), highest.max(code))
    })
}

fn check_linear(layout: &BlockLayout) -> Result<(), QuantError> {
    if layout.sub.is_some() {
        return Err(unsupported("sub-blocks carry their own scales"));
    }
    if layout.min.is_some() {
        return Err(unsupported("a stored min is not an absolute-maximum scale"));
    }
    if layout.element.packed_radix > 1 {
        return Err(unsupported("mixed-radix packed codes"));
    }
    if layout.element.code.exponent_bits != 0 || code_bits(layout) > MAX_CODE_BITS {
        return Err(unsupported("element codes are not narrow integers"));
    }
    Ok(())
}

fn touches(plane: &Plane, elements: usize, bytes: core::ops::Range<usize>) -> bool {
    (0..elements).any(|index| bytes.contains(&(plane.address(index) / 8)))
}

/// llama's `Q8_1` carries `s = d * sum(q)` in the two bytes after the scale; a layout whose planes
/// leave those bytes unused (`BlockLayout::Q8_1`) is the one place the consumer reads it.
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

fn linear(layout: &BlockLayout) -> Result<Linear, QuantError> {
    check_linear(layout)?;
    let (lowest, highest) = code_range(layout);
    let multiplier = f32::from(layout.multiplier);
    let bias = f32::from(layout.element.bias);
    let value_low = multiplier * lowest as f32 - bias;
    let value_high = multiplier * highest as f32 - bias;
    let reach = value_high.min(-value_low);
    if reach <= 0.0 {
        return Err(unsupported("no code magnitude on both sides of zero"));
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
    block[address / 8] |= ((bits & mask) << (address % 8)) as u8;
    cursor.advance();
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
    let scale_code = encode(peak / shape.reach, layout.scale.layout);
    let scale = decode(scale_code, layout.scale.layout);
    let scale_at = layout.run_scale_byte(run);
    put_scale(layout, scale_at, scale_code, block);
    let mut value_sum = 0.0f32;
    for value in input {
        let code = shape.code_for(*value, scale);
        put_code(
            layout,
            cursors,
            encode(code as f32, layout.element.code),
            block,
        );
        value_sum += shape.value_of(code);
    }
    if let Some(gap) = shape.sum_gap {
        let at = scale_at + gap;
        block[at..at + SUM_BYTES].copy_from_slice(&f16::from_f32(scale * value_sum).to_le_bytes());
    }
}

fn quantize_block(
    layout: &BlockLayout,
    shape: &Linear,
    input: &[f32],
    block: &mut [u8],
) -> Result<(), QuantError> {
    if input.iter().any(|value| !value.is_finite()) {
        return Err(QuantError::NonFiniteInput { codec: CODEC });
    }
    block.fill(0);
    let mut cursors = [layout.element.low.cursor(), layout.element.low.cursor()];
    if let Some(high) = &layout.element.high {
        cursors[1] = high.cursor();
    }
    let runs = input.chunks(usize::from(layout.run_elements()));
    for (run, group) in runs.enumerate() {
        quantize_run(layout, shape, run as u16, group, &mut cursors, block);
    }
    Ok(())
}

/// The largest value magnitude `layout` reaches on both sides of zero, in units of its scale: the
/// divisor of the block maximum. Also the test of whether [`quantize`] encodes `layout` at all.
///
/// # Errors
/// [`QuantError::UnsupportedBlockLayout`] naming what the layout has that a shared absolute-maximum
/// scale cannot express.
pub fn reach(layout: &BlockLayout) -> Result<f32, QuantError> {
    linear(layout).map(|shape| shape.reach)
}

/// Encodes whole blocks of `f32` into `layout`'s packed bytes.
///
/// # Errors
/// [`QuantError::UnsupportedBlockLayout`] for a layout [`supports`] refuses;
/// [`QuantError::InputNotElementMultiple`] if `input` is not a whole number of blocks;
/// [`QuantError::OutputSizeMismatch`] if `output` is not exactly the packed byte count;
/// [`QuantError::NonFiniteInput`] for a NaN or infinite element.
pub fn quantize(layout: &BlockLayout, input: &[f32], output: &mut [u8]) -> Result<(), QuantError> {
    let shape = linear(layout)?;
    let (elements, block_bytes) = (
        usize::from(layout.elements),
        usize::from(layout.block_bytes),
    );
    if !input.len().is_multiple_of(elements) {
        return Err(QuantError::InputNotElementMultiple {
            codec: CODEC,
            unit: "block",
            found: input.len(),
            block_elements: elements,
        });
    }
    let expected = input.len() / elements * block_bytes;
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    for (group, block) in input.chunks(elements).zip(output.chunks_mut(block_bytes)) {
        quantize_block(layout, &shape, group, block)?;
    }
    Ok(())
}
