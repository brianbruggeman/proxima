//! Caller-buffer transforms for cold, packed low-precision tensor payloads.

use proxima_primitives::float_layout::{FloatLayout, SignEncoding, Specials};

use crate::quant::{QuantError, float_layout};

const CODEC: &str = "cold_float";
const BF16_BYTES: usize = 2;

/// Converts little-endian BF16 tensor bytes into caller-owned packed output.
///
/// The input remains borrowed, the output is caller-sized, and the transform
/// allocates no intermediate tensor. Values use least-significant-bit-first
/// packing at the layout's declared width. Padding bits in the final byte are zero.
pub fn quantize_bf16(
    layout: FloatLayout,
    input: &[u8],
    output: &mut [u8],
) -> Result<(), QuantError> {
    if !input.len().is_multiple_of(BF16_BYTES) {
        return Err(QuantError::InputNotBlockMultiple {
            codec: CODEC,
            found: input.len(),
            block_bytes: BF16_BYTES,
        });
    }
    let element_count = input.len() / BF16_BYTES;
    let values = input
        .as_chunks::<BF16_BYTES>()
        .0
        .iter()
        .map(|chunk| half::bf16::from_le_bytes(*chunk).to_f32());
    quantize_values(layout, element_count, values, output)
}

/// Converts caller-owned F32 tensor values into a packed cold representation.
pub fn quantize_f32(
    layout: FloatLayout,
    input: &[f32],
    output: &mut [u8],
) -> Result<(), QuantError> {
    quantize_values(layout, input.len(), input.iter().copied(), output)
}

fn quantize_values(
    layout: FloatLayout,
    element_count: usize,
    values: impl Iterator<Item = f32>,
    output: &mut [u8],
) -> Result<(), QuantError> {
    check_layout(layout)?;
    let width = layout.width() as usize;
    let expected = element_count.saturating_mul(width).div_ceil(8);
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    output.fill(0);
    for (index, value) in values.enumerate() {
        let encoded = encode_value(layout, value)?;
        write_bits(output, index * width, width, encoded);
    }
    Ok(())
}

/// Converts a packed artifact into caller-owned F32 values.
pub fn dequantize(layout: FloatLayout, input: &[u8], output: &mut [f32]) -> Result<(), QuantError> {
    check_layout(layout)?;
    let width = layout.width() as usize;
    let expected = output.len().saturating_mul(width).div_ceil(8);
    if input.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: input.len(),
            expected,
        });
    }
    for (index, value) in output.iter_mut().enumerate() {
        let encoded = read_bits(input, index * width, width);
        *value = float_layout::decode(encoded.into(), layout);
    }
    Ok(())
}

fn check_layout(layout: FloatLayout) -> Result<(), QuantError> {
    let sign_bits = match layout.sign {
        SignEncoding::Absent => 0,
        SignEncoding::Magnitude | SignEncoding::TwosComplement => 1,
    };
    let width = u16::from(layout.exponent_bits) + u16::from(layout.mantissa_bits) + sign_bits;
    if (1..=32).contains(&width) && layout.exponent_bits <= 8 && layout.mantissa_bits <= 23 {
        Ok(())
    } else {
        Err(QuantError::FieldWidthOutOfRange {
            codec: CODEC,
            found: width as u32,
            min: 1,
            max: 32,
        })
    }
}

fn encode_value(layout: FloatLayout, value: f32) -> Result<u32, QuantError> {
    if !value.is_finite() && layout.specials == Specials::None {
        return Err(QuantError::NonFiniteInput { codec: CODEC });
    }
    Ok(float_layout::encode(value, layout))
}

fn write_bits(output: &mut [u8], offset: usize, width: usize, value: u32) {
    let byte_index = offset / 8;
    let bit_offset = offset % 8;
    let shifted_value = u64::from(value) << bit_offset;
    let byte_count = (bit_offset + width).div_ceil(8);
    for byte_offset in 0..byte_count {
        output[byte_index + byte_offset] |= (shifted_value >> (byte_offset * 8)) as u8;
    }
}

fn read_bits(input: &[u8], offset: usize, width: usize) -> u32 {
    let byte_index = offset / 8;
    let bit_offset = offset % 8;
    let byte_count = (bit_offset + width).div_ceil(8);
    let mut packed = 0_u64;
    for byte_offset in 0..byte_count {
        packed |= u64::from(input[byte_index + byte_offset]) << (byte_offset * 8);
    }
    let mask = if width == 32 {
        u32::MAX
    } else {
        (1_u32 << width) - 1
    };
    ((packed >> bit_offset) as u32) & mask
}

#[cfg(test)]
mod tests {
    use super::{dequantize, quantize_bf16, quantize_f32};
    use crate::quant::QuantError;
    use proxima_primitives::float_layout::FloatLayout;

    #[test]
    fn bf16_tensor_bytes_convert_to_each_packed_cold_float_format() {
        let input = [0x80, 0x3f, 0x00, 0xc0, 0x00, 0x3e, 0x00, 0x00];
        let expected = [1.0, -2.0, 0.125, 0.0];
        for (layout, expected_output) in [
            (FloatLayout::E5M2_SATURATING, expected),
            (FloatLayout::E3M2, expected),
            (FloatLayout::E2M2, [1.0, -2.0, 0.0, 0.0]),
        ] {
            let width = layout.width() as usize;
            let packed_len = expected.len() * width / 8
                + usize::from(!expected.len().saturating_mul(width).is_multiple_of(8));
            let mut packed = alloc::vec![0_u8; packed_len];
            quantize_bf16(layout, &input, &mut packed)
                .expect("four BF16 values fit the selected packed format");
            let mut output = [0.0_f32; 4];
            dequantize(layout, &packed, &mut output)
                .expect("packed payload contains four elements");
            assert_eq!(output, expected_output, "layout {layout:?}");
        }
    }

    #[test]
    fn cold_float_widths_and_ranges_come_from_the_float_layout() {
        let layouts = [
            FloatLayout::E5M2_SATURATING,
            FloatLayout::E3M2,
            FloatLayout::E2M2,
        ];
        let widths = layouts.map(|layout| layout.width());
        assert_eq!(widths, [8, 6, 5]);
        let ranges = layouts.map(|layout| layout.max_normal());
        assert_eq!(ranges, [57_344.0, 28.0, 7.0]);
    }

    #[test]
    fn bf16_tensor_conversion_rejects_partial_input_and_wrong_output_size() {
        assert_eq!(
            quantize_bf16(FloatLayout::E2M2, &[0x80], &mut []),
            Err(QuantError::InputNotBlockMultiple {
                codec: "cold_float",
                found: 1,
                block_bytes: 2,
            })
        );
        assert_eq!(
            quantize_bf16(FloatLayout::E2M2, &[0x80, 0x3f], &mut []),
            Err(QuantError::OutputSizeMismatch {
                found: 0,
                expected: 1,
            })
        );
    }

    #[test]
    fn packed_cold_floats_cross_byte_boundaries_least_significant_bit_first() {
        let mut e2m2 = [0_u8; 2];
        quantize_f32(FloatLayout::E2M2, &[1.0, 1.0], &mut e2m2)
            .expect("two E2M2 values fit 2 bytes");
        assert_eq!(e2m2, [0x84, 0x00]);
        let mut e3m2 = [0_u8; 3];
        quantize_f32(FloatLayout::E3M2, &[1.0, 1.0, -1.0, 0.0], &mut e3m2)
            .expect("four E3M2 values fit 3 bytes");
        assert_eq!(e3m2, [0x0c, 0xc3, 0x02]);
        let mut unpacked = [0.0_f32; 4];
        dequantize(FloatLayout::E3M2, &e3m2, &mut unpacked)
            .expect("packed bytes contain four E3M2 values");
        assert_eq!(unpacked, [1.0, 1.0, -1.0, 0.0]);
    }

    #[test]
    fn f32_tensor_values_convert_to_packed_e5m2() {
        let input = [1.0_f32, -2.0, 0.125, 0.0];
        let mut packed = [0_u8; 4];
        quantize_f32(FloatLayout::E5M2_SATURATING, &input, &mut packed)
            .expect("four F32 values fit the E5M2 payload");
        let mut output = [0.0_f32; 4];
        dequantize(FloatLayout::E5M2_SATURATING, &packed, &mut output)
            .expect("E5M2 payload contains four values");
        assert_eq!(output, input);
    }

    #[test]
    fn same_packed_path_round_trips_a_structured_32_bit_layout() {
        let input = [1.0_f32, -2.0, 0.125, -0.0];
        let mut packed = [0_u8; 16];
        quantize_f32(FloatLayout::BF32, &input, &mut packed)
            .expect("four binary32 values fit the generic packed path");
        let mut output = [0.0_f32; 4];
        dequantize(FloatLayout::BF32, &packed, &mut output)
            .expect("the stored layout describes the packed values");
        assert_eq!(output.map(f32::to_bits), input.map(f32::to_bits));
    }

    #[test]
    fn one_packer_handles_a_layout_without_a_named_format_variant() {
        let layout = FloatLayout::E2M3;
        let input = [1.0_f32, -2.0, 0.125, 0.0];
        let mut packed = [0_u8; 3];
        quantize_f32(layout, &input, &mut packed)
            .expect("four six-bit values fit the shared packed path");
        let mut output = [0.0_f32; 4];
        dequantize(layout, &packed, &mut output)
            .expect("the same structural layout decodes the packed values");
        assert_eq!(output, [1.0, -2.0, 0.125, 0.0]);
    }

    #[test]
    fn bf16_and_f32_sources_share_identical_packed_storage() {
        let layout = FloatLayout::E3M2;
        let values = [1.0_f32, -2.0, 0.125, 0.375];
        let input = values
            .map(|value| half::bf16::from_f32(value).to_le_bytes())
            .concat();
        let mut packed_bf16 = [0_u8; 3];
        let mut packed_f32 = [0_u8; 3];
        quantize_bf16(layout, &input, &mut packed_bf16)
            .expect("BF16 source uses the shared packer");
        quantize_f32(layout, &values, &mut packed_f32).expect("F32 source uses the shared packer");
        assert_eq!(packed_bf16, packed_f32);
    }

    #[test]
    fn layouts_without_non_finite_codes_reject_non_finite_values() {
        let mut output = [0_u8; 1];
        assert_eq!(
            quantize_f32(FloatLayout::E3M2, &[f32::INFINITY], &mut output),
            Err(QuantError::NonFiniteInput {
                codec: "cold_float"
            })
        );
    }
}
