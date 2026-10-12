//! Scaled float blocks: 32 weights share one `f16` absmax scale and each weight is a
//! [`FloatLayout`] code of `layout.width()` bits. With [`FloatLayout::E3M2`] (OCP MX `s1e3m2`)
//! that is 26 bytes per 32 weights, 6.5 bits per weight.
//!
//! Teaching pointer: the per-element arithmetic is [`crate::quant::float_layout::encode`] and
//! [`crate::quant::float_layout::decode`]; this module only adds the block scale and the bit
//! packing, the same shape as [`crate::quant::q8_0`] with a float code in place of an `i8`.
//!
//! Block layout, 26 bytes:
//!
//! - bytes 0..2: the scale, `f16` little-endian, `f16(absmax / layout.max_normal())`, so the
//!   largest magnitude lands on the max normal code (28.0 for E3M2).
//! - bytes 2..26: the payload, 24 bytes holding 32 six-bit fields.
//!
//! Payload bit order, written once for every transcription (the omega decode kernel copies it):
//! read the 24 payload bytes as one little-endian integer, byte 0 least significant. Element `k`
//! is bits `6k .. 6k + 6` of that integer, so element `k` starts at payload bit `6k`, i.e. byte
//! `(6k) / 8`, bit `(6k) % 8` of that byte. Per 3-byte group `g` (bytes `3g .. 3g + 3`) let
//! `word = b0 | b1 << 8 | b2 << 16`; element `4g + j` is `(word >> 6j) & 0x3f` for `j` in 0..4.
//! Inside a field the code is the layout's own: sign at bit 5, exponent bits 4..2, mantissa
//! bits 1..0. The element is `scale_f16 * decode(field)`.

use proxima_primitives::codec::Codec;
use proxima_primitives::float_layout::FloatLayout;

use crate::quant::QuantError;
use crate::quant::float_layout::{decode, encode};

const CODEC: &str = "scaled_float";

pub const BLOCK_ELEMENTS: usize = Codec::FP6_E3M2_BLOCK_ELEMENTS;
pub const BLOCK_BYTES: usize = Codec::FP6_E3M2_BLOCK_BYTES;
pub const SCALE_BYTES: usize = 2;
pub const PAYLOAD_BYTES: usize = BLOCK_BYTES - SCALE_BYTES;
pub const FIELD_BITS: u32 = 6;

const GROUP_ELEMENTS: usize = 4;
const GROUP_BYTES: usize = 3;
const FIELD_MASK: u32 = (1 << FIELD_BITS) - 1;

const _: () = assert!(PAYLOAD_BYTES * 8 == BLOCK_ELEMENTS * FIELD_BITS as usize);
const _: () = assert!(GROUP_ELEMENTS * FIELD_BITS as usize == GROUP_BYTES * 8);

const fn check_width(layout: FloatLayout) -> Result<(), QuantError> {
    if layout.width() as u32 == FIELD_BITS {
        Ok(())
    } else {
        Err(QuantError::FieldWidthMismatch {
            codec: CODEC,
            found: layout.width() as u32,
            expected: FIELD_BITS,
        })
    }
}

fn absmax(values: &[f32; BLOCK_ELEMENTS]) -> Result<f32, QuantError> {
    values.iter().try_fold(0.0f32, |peak, value| {
        value
            .is_finite()
            .then(|| peak.max(value.abs()))
            .ok_or(QuantError::NonFiniteInput { codec: CODEC })
    })
}

fn block_scale(
    values: &[f32; BLOCK_ELEMENTS],
    layout: FloatLayout,
) -> Result<half::f16, QuantError> {
    let scale = half::f16::from_f32(absmax(values)? / layout.max_normal());
    if scale.is_finite() {
        Ok(scale)
    } else {
        Err(QuantError::NonFiniteInput { codec: CODEC })
    }
}

fn encode_scaled(value: f32, scale: f32, layout: FloatLayout) -> u32 {
    if scale == 0.0 {
        0
    } else {
        encode(value / scale, layout)
    }
}

fn pack_group(
    group: &[f32; GROUP_ELEMENTS],
    scale: f32,
    layout: FloatLayout,
    bytes: &mut [u8; GROUP_BYTES],
) {
    let word = group.iter().enumerate().fold(0u32, |word, (slot, &value)| {
        word | (encode_scaled(value, scale, layout) & FIELD_MASK) << (FIELD_BITS * slot as u32)
    });
    bytes.copy_from_slice(&word.to_le_bytes()[..GROUP_BYTES]);
}

/// Quantizes 32 weights into one block: `f16` absmax scale, then 32 packed six-bit codes.
///
/// # Errors
/// [`QuantError::FieldWidthMismatch`] if `layout` is not six bits wide;
/// [`QuantError::NonFiniteInput`] if a weight is NaN or infinite, or the scale overflows `f16`.
#[must_use = "the block is only written when this returns Ok"]
pub fn quantize_block(
    values: &[f32; BLOCK_ELEMENTS],
    layout: FloatLayout,
    out: &mut [u8; BLOCK_BYTES],
) -> Result<(), QuantError> {
    check_width(layout)?;
    let scale = block_scale(values, layout)?;
    let (scale_bytes, payload) = out.split_at_mut(SCALE_BYTES);
    scale_bytes.copy_from_slice(&scale.to_le_bytes());
    values
        .as_chunks::<GROUP_ELEMENTS>()
        .0
        .iter()
        .zip(payload.as_chunks_mut::<GROUP_BYTES>().0)
        .for_each(|(group, bytes)| pack_group(group, scale.to_f32(), layout, bytes));
    Ok(())
}

fn unpack_group(
    bytes: &[u8; GROUP_BYTES],
    scale: f32,
    layout: FloatLayout,
    out: &mut [f32; GROUP_ELEMENTS],
) {
    let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]);
    out.iter_mut().enumerate().for_each(|(slot, element)| {
        *element = scale * decode((word >> (FIELD_BITS * slot as u32)) & FIELD_MASK, layout);
    });
}

/// Dequantizes one block: each element is `scale_f16 * decode(field)`.
///
/// # Errors
/// [`QuantError::FieldWidthMismatch`] if `layout` is not six bits wide.
#[must_use = "the output is only written when this returns Ok"]
pub fn dequantize_block(
    block: &[u8; BLOCK_BYTES],
    layout: FloatLayout,
    out: &mut [f32; BLOCK_ELEMENTS],
) -> Result<(), QuantError> {
    check_width(layout)?;
    let (scale_bytes, payload) = block.split_at(SCALE_BYTES);
    let scale = half::f16::from_le_bytes([scale_bytes[0], scale_bytes[1]]).to_f32();
    payload
        .as_chunks::<GROUP_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<GROUP_ELEMENTS>().0)
        .for_each(|(bytes, group)| unpack_group(bytes, scale, layout, group));
    Ok(())
}

#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::expect_used)]
    extern crate std;

    use alloc::vec::Vec;
    use std::path::Path;

    use proxima_telemetry::debug;

    use super::*;
    use crate::quant::q8_0;
    use crate::types::GgmlType;

    const GRANITE_DEFAULT: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
         sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80";
    const MAX_NORMAL_RATIO: f32 = 28.0;
    const SUBNORMAL_HALF_SPACING: f32 = 0.03125;
    const FLOAT_SLACK_ULPS: f32 = 4.0;

    fn granite_q8_0_rows_1024x512() -> Vec<f32> {
        let path = std::env::var("PROXIMA_ARCH_GRANITE_MOE_GGUF")
            .unwrap_or_else(|_| GRANITE_DEFAULT.into());
        assert!(
            Path::new(&path).exists(),
            "no granite gguf at {path}: set PROXIMA_ARCH_GRANITE_MOE_GGUF or stage the blob"
        );
        let (parsed, bytes) =
            crate::edge::read_file(Path::new(&path)).expect("granite blob must parse as gguf");
        let rows: Vec<f32> = parsed
            .tensors
            .iter()
            .filter(|tensor| {
                tensor.ggml_type == GgmlType::Q8_0 && tensor.dims.as_slice() == [1024, 512]
            })
            .flat_map(|tensor| {
                let range = parsed
                    .tensor_data_range(tensor, bytes.len() as u64)
                    .expect("tensor range must fit the file");
                let mut tensor_rows = alloc::vec![0.0f32; tensor.element_count() as usize];
                q8_0::dequantize(
                    &bytes[range.start as usize..range.end as usize],
                    &mut tensor_rows,
                )
                .expect("well-formed q8_0 tensor");
                tensor_rows
            })
            .collect();
        assert!(
            !rows.is_empty(),
            "granite blob must hold Q8_0 tensors of 1024x512 elements"
        );
        rows
    }

    fn half_spacing(ratio: f32) -> f32 {
        if ratio < 0.25 {
            SUBNORMAL_HALF_SPACING
        } else {
            libm::exp2f(libm::floorf(libm::log2f(ratio)) - 3.0)
        }
    }

    fn bound_ratio(source: &[f32; BLOCK_ELEMENTS], block: &[u8; BLOCK_BYTES]) -> f32 {
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let mut restored = [0.0f32; BLOCK_ELEMENTS];
        dequantize_block(block, FloatLayout::E3M2, &mut restored).expect("six bit layout");
        source
            .iter()
            .zip(restored)
            .filter(|(value, _)| value.abs() / scale <= MAX_NORMAL_RATIO)
            .map(|(value, got)| {
                let allowed = scale * half_spacing(value.abs() / scale);
                let slack = FLOAT_SLACK_ULPS * f32::EPSILON * value.abs().max(got.abs());
                (value - got).abs() / (allowed + slack)
            })
            .fold(0.0f32, f32::max)
    }

    fn quantized(source: &[f32; BLOCK_ELEMENTS]) -> [u8; BLOCK_BYTES] {
        let mut block = [0u8; BLOCK_BYTES];
        quantize_block(source, FloatLayout::E3M2, &mut block).expect("finite block");
        block
    }

    #[test]
    fn scaled_e3m2_block_round_trip_error_bound_1024x512() {
        let rows = granite_q8_0_rows_1024x512();
        assert_eq!(
            rows.len() % (1024 * 512),
            0,
            "rows are whole 1024x512 tensors"
        );

        let worst = rows
            .as_chunks::<BLOCK_ELEMENTS>()
            .0
            .iter()
            .map(|source| bound_ratio(source, &quantized(source)))
            .fold(0.0f32, f32::max);

        debug!(
            worst,
            "quant.scaled_float worst error over bound, granite 1024x512"
        );
        assert!(worst <= 1.0, "worst error is {worst} times the h(q) bound");
    }

    #[test]
    fn scaled_e3m2_block_saturates_above_28() {
        let mut source = [0.0f32; BLOCK_ELEMENTS];
        let absmax = (1..2000u32)
            .map(|step| step as f32 * 0.001)
            .find(|&peak| {
                half::f16::from_f32(peak / MAX_NORMAL_RATIO).to_f32() * MAX_NORMAL_RATIO < peak
            })
            .expect("some absmax must round its f16 scale down");
        source[0] = absmax;
        source[1] = -absmax;
        let scale = half::f16::from_f32(absmax / MAX_NORMAL_RATIO).to_f32();
        assert!(
            absmax / scale > MAX_NORMAL_RATIO,
            "arranged ratio must exceed 28"
        );

        let mut restored = [0.0f32; BLOCK_ELEMENTS];
        dequantize_block(&quantized(&source), FloatLayout::E3M2, &mut restored)
            .expect("six bit layout");

        assert_eq!(restored[0], MAX_NORMAL_RATIO * scale);
        assert_eq!(restored[1], -MAX_NORMAL_RATIO * scale);
    }

    #[test]
    fn scaled_e3m2_block_control_perturbed_code_exceeds_bound() {
        let rows = granite_q8_0_rows_1024x512();
        let source = rows
            .as_chunks::<BLOCK_ELEMENTS>()
            .0
            .iter()
            .find(|source| source[0].abs() > 0.0 && bound_ratio(source, &quantized(source)) <= 1.0)
            .expect("a block with a non-zero first weight within the bound");
        let mut block = quantized(source);
        assert!(
            bound_ratio(source, &block) <= 1.0,
            "unperturbed block must satisfy the bound"
        );

        block[SCALE_BYTES] ^= 1 << (FIELD_BITS - 1);

        assert!(
            bound_ratio(source, &block) > 1.0,
            "flipping the sign of element 0 must break the bound"
        );
    }

    const OCP_E3M2_BIAS: i32 = 3;
    const OCP_E3M2_MAGNITUDE_CODES: u32 = 32;
    const BLOCK_STRIDE: usize = 16;

    fn ocp_table4_magnitude(code: u32) -> f32 {
        let exponent = (code >> 2) as i32;
        let mantissa = (code & 3) as f32;
        if exponent == 0 {
            libm::exp2f((1 - OCP_E3M2_BIAS) as f32) * (mantissa / 4.0)
        } else {
            libm::exp2f((exponent - OCP_E3M2_BIAS) as f32) * (1.0 + mantissa / 4.0)
        }
    }

    fn ocp_nearest_code(ratio: f32) -> u32 {
        let sign = u32::from(ratio.is_sign_negative()) << 5;
        let target = ratio.abs();
        let magnitude = (0..OCP_E3M2_MAGNITUDE_CODES).fold(0u32, |best, code| {
            let distance = (ocp_table4_magnitude(code) - target).abs();
            let best_distance = (ocp_table4_magnitude(best) - target).abs();
            if distance < best_distance || (distance == best_distance && code % 2 == 0) {
                code
            } else {
                best
            }
        });
        sign | magnitude
    }

    fn payload_field(payload: &[u8], element: usize) -> u32 {
        (0..FIELD_BITS as usize).fold(0u32, |field, bit| {
            let position = element * FIELD_BITS as usize + bit;
            let value = (payload[position / 8] >> (position % 8)) & 1;
            field | u32::from(value) << bit
        })
    }

    fn mismatched_codes(source: &[f32; BLOCK_ELEMENTS], block: &[u8; BLOCK_BYTES]) -> usize {
        let scale = half::f16::from_f32(
            source.iter().fold(0.0f32, |peak, value| peak.max(value.abs())) / MAX_NORMAL_RATIO,
        )
        .to_f32();
        assert_eq!(
            block[..SCALE_BYTES],
            half::f16::from_f32(scale).to_le_bytes(),
            "block scale must be f16(absmax / 28)"
        );
        (0..BLOCK_ELEMENTS)
            .filter(|&element| {
                let expected = if scale == 0.0 {
                    0
                } else {
                    ocp_nearest_code(source[element] / scale)
                };
                payload_field(&block[SCALE_BYTES..], element) != expected
            })
            .count()
    }

    #[test]
    fn scaled_e3m2_codes_match_independent_ocp_table4_encode_on_granite_rows() {
        let rows = granite_q8_0_rows_1024x512();

        let (blocks, mismatches) = rows
            .as_chunks::<BLOCK_ELEMENTS>()
            .0
            .iter()
            .step_by(BLOCK_STRIDE)
            .fold((0usize, 0usize), |(blocks, mismatches), source| {
                (blocks + 1, mismatches + mismatched_codes(source, &quantized(source)))
            });

        debug!(
            blocks,
            mismatches, "quant.scaled_float codes versus independent ocp table 4 encode"
        );
        assert!(blocks > 0, "the stride must sample at least one block");
        assert_eq!(mismatches, 0, "codes differing from the OCP table 4 nearest-even encode");
    }

    #[test]
    fn scaled_e3m2_independent_ocp_encode_control_flipped_code_is_detected() {
        let rows = granite_q8_0_rows_1024x512();
        let source = rows
            .as_chunks::<BLOCK_ELEMENTS>()
            .0
            .iter()
            .find(|source| source[0].abs() > 0.0)
            .expect("a block with a non-zero first weight");
        let mut block = quantized(source);
        assert_eq!(mismatched_codes(source, &block), 0, "unperturbed block agrees");

        block[SCALE_BYTES] ^= 1 << (FIELD_BITS - 2);

        assert_eq!(
            mismatched_codes(source, &block),
            1,
            "flipping one exponent bit of element 0 must register as one mismatch"
        );
    }
}
