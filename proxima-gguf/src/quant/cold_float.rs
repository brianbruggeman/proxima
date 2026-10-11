//! Caller-buffer transforms for cold, packed low-precision tensor payloads.

use proxima_primitives::float_layout::FloatLayout;

use crate::quant::{QuantError, bf8_e5m2, fp5_e2m2, fp6_e3m2};

const CODEC: &str = "cold_float";
const BF16_BYTES: usize = 2;

/// Packed floating-point representation for a cold tensor artifact.
///
/// This descriptor is separate from [`crate::types::GgmlType`] because these
/// formats do not have a byte-aligned, fixed-width scalar element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColdFloatFormat {
    /// IEEE-like sign(1), exponent(5), mantissa(2) bytes.
    E5M2,
    /// Finite sign(1), exponent(3), mantissa(2) values packed at six bits.
    E3M2,
    /// Proxima sign(1), exponent(2), mantissa(2), bias 1 values packed at five bits.
    E2M2,
}

impl ColdFloatFormat {
    /// The [`FloatLayout`] this format instantiates; the one source of its widths and range.
    #[must_use]
    pub const fn layout(self) -> FloatLayout {
        match self {
            Self::E5M2 => FloatLayout::E5M2_SATURATING,
            Self::E3M2 => FloatLayout::E3M2,
            Self::E2M2 => FloatLayout::E2M2,
        }
    }

    /// Number of bits stored per tensor element.
    #[must_use]
    pub const fn bits_per_element(self) -> usize {
        self.layout().width() as usize
    }

    /// Stable machine name used in manifests and command-line selectors.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::E5M2 => "e5m2",
            Self::E3M2 => "e3m2",
            Self::E2M2 => "e2m2",
        }
    }
}

/// Converts little-endian BF16 tensor bytes into caller-owned packed output.
///
/// The input remains borrowed, the output is caller-sized, and the transform
/// allocates no intermediate tensor. E3M2/E2M2 use least-significant-bit-first
/// packing. Padding bits in the final byte are zero.
pub fn quantize_bf16(
    format: ColdFloatFormat,
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
    let expected = element_count
        .saturating_mul(format.bits_per_element())
        .div_ceil(8);
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    if format == ColdFloatFormat::E5M2 {
        for (output_byte, input_chunk) in output.iter_mut().zip(input.as_chunks::<BF16_BYTES>().0) {
            *output_byte = bf8_e5m2::encode(half::bf16::from_le_bytes(*input_chunk).to_f32());
        }
        return Ok(());
    }
    output.fill(0);
    for (index, chunk) in input.as_chunks::<BF16_BYTES>().0.iter().enumerate() {
        let value = half::bf16::from_le_bytes(*chunk).to_f32();
        let encoded = encode_value(format, value)?;
        write_bits(
            output,
            index * format.bits_per_element(),
            format.bits_per_element(),
            encoded,
        );
    }
    Ok(())
}

/// Converts caller-owned F32 tensor values into a packed cold representation.
pub fn quantize_f32(
    format: ColdFloatFormat,
    input: &[f32],
    output: &mut [u8],
) -> Result<(), QuantError> {
    let expected = input
        .len()
        .saturating_mul(format.bits_per_element())
        .div_ceil(8);
    if output.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: output.len(),
            expected,
        });
    }
    if format == ColdFloatFormat::E5M2 {
        for (output_byte, value) in output.iter_mut().zip(input) {
            *output_byte = bf8_e5m2::encode(*value);
        }
        return Ok(());
    }
    output.fill(0);
    for (index, value) in input.iter().enumerate() {
        let encoded = encode_value(format, *value)?;
        write_bits(
            output,
            index * format.bits_per_element(),
            format.bits_per_element(),
            encoded,
        );
    }
    Ok(())
}

/// Converts a packed artifact into caller-owned F32 values.
pub fn dequantize(
    format: ColdFloatFormat,
    input: &[u8],
    output: &mut [f32],
) -> Result<(), QuantError> {
    let expected = output
        .len()
        .saturating_mul(format.bits_per_element())
        .div_ceil(8);
    if input.len() != expected {
        return Err(QuantError::OutputSizeMismatch {
            found: input.len(),
            expected,
        });
    }
    for (index, value) in output.iter_mut().enumerate() {
        let encoded = read_bits(
            input,
            index * format.bits_per_element(),
            format.bits_per_element(),
        );
        *value = match format {
            ColdFloatFormat::E5M2 => bf8_e5m2::decode(encoded),
            ColdFloatFormat::E3M2 => fp6_e3m2::decode(encoded),
            ColdFloatFormat::E2M2 => fp5_e2m2::decode(encoded),
        };
    }
    Ok(())
}

fn encode_value(format: ColdFloatFormat, value: f32) -> Result<u8, QuantError> {
    match format {
        ColdFloatFormat::E5M2 => Ok(bf8_e5m2::encode(value)),
        ColdFloatFormat::E3M2 => fp6_e3m2::encode(value),
        ColdFloatFormat::E2M2 => fp5_e2m2::encode(value),
    }
}

fn write_bits(output: &mut [u8], offset: usize, width: usize, value: u8) {
    let byte_index = offset / 8;
    let bit_offset = offset % 8;
    let shifted_value = u16::from(value) << bit_offset;
    output[byte_index] |= shifted_value as u8;
    if bit_offset + width > 8 {
        output[byte_index + 1] |= (shifted_value >> 8) as u8;
    }
}

fn read_bits(input: &[u8], offset: usize, width: usize) -> u8 {
    let mut value = 0;
    for bit in 0..width {
        let absolute = offset + bit;
        value |= ((input[absolute / 8] >> (absolute % 8)) & 1) << bit;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::{ColdFloatFormat, dequantize, quantize_bf16, quantize_f32};
    use crate::quant::QuantError;

    #[test]
    fn bf16_tensor_bytes_convert_to_each_packed_cold_float_format() {
        let input = [0x80, 0x3f, 0x00, 0xc0, 0x00, 0x3e, 0x00, 0x00];
        let expected = [1.0, -2.0, 0.125, 0.0];
        for format in [
            ColdFloatFormat::E5M2,
            ColdFloatFormat::E3M2,
            ColdFloatFormat::E2M2,
        ] {
            let packed_len = expected.len() * format.bits_per_element() / 8
                + usize::from(
                    !expected
                        .len()
                        .saturating_mul(format.bits_per_element())
                        .is_multiple_of(8),
                );
            let mut packed = alloc::vec![0_u8; packed_len];
            quantize_bf16(format, &input, &mut packed)
                .expect("four BF16 values fit the selected packed format");
            let mut output = [0.0_f32; 4];
            dequantize(format, &packed, &mut output)
                .expect("packed payload contains four elements");
            let expected_output = match format {
                ColdFloatFormat::E2M2 => [1.0, -2.0, 0.0, 0.0],
                ColdFloatFormat::E5M2 | ColdFloatFormat::E3M2 => expected,
            };
            assert_eq!(output, expected_output, "format {format:?}");
        }
    }

    #[test]
    fn cold_float_widths_and_ranges_come_from_the_float_layout() {
        let widths = [ColdFloatFormat::E5M2, ColdFloatFormat::E3M2, ColdFloatFormat::E2M2]
            .map(ColdFloatFormat::bits_per_element);
        assert_eq!(widths, [8, 6, 5]);
        let ranges = [ColdFloatFormat::E5M2, ColdFloatFormat::E3M2, ColdFloatFormat::E2M2]
            .map(|format| format.layout().max_normal());
        assert_eq!(ranges, [57_344.0, 28.0, 7.0]);
    }

    #[test]
    fn bf16_tensor_conversion_rejects_partial_input_and_wrong_output_size() {
        assert_eq!(
            quantize_bf16(ColdFloatFormat::E2M2, &[0x80], &mut []),
            Err(QuantError::InputNotBlockMultiple {
                codec: "cold_float",
                found: 1,
                block_bytes: 2,
            })
        );
        assert_eq!(
            quantize_bf16(ColdFloatFormat::E2M2, &[0x80, 0x3f], &mut []),
            Err(QuantError::OutputSizeMismatch {
                found: 0,
                expected: 1,
            })
        );
    }

    #[test]
    fn packed_cold_floats_cross_byte_boundaries_least_significant_bit_first() {
        let mut e2m2 = [0_u8; 2];
        quantize_f32(ColdFloatFormat::E2M2, &[1.0, 1.0], &mut e2m2)
            .expect("two E2M2 values fit 2 bytes");
        assert_eq!(e2m2, [0x84, 0x00]);
        let mut e3m2 = [0_u8; 3];
        quantize_f32(ColdFloatFormat::E3M2, &[1.0, 1.0, -1.0, 0.0], &mut e3m2)
            .expect("four E3M2 values fit 3 bytes");
        assert_eq!(e3m2, [0x0c, 0xc3, 0x02]);
        let mut unpacked = [0.0_f32; 4];
        dequantize(ColdFloatFormat::E3M2, &e3m2, &mut unpacked)
            .expect("packed bytes contain four E3M2 values");
        assert_eq!(unpacked, [1.0, 1.0, -1.0, 0.0]);
    }

    #[test]
    fn f32_tensor_values_convert_to_packed_e5m2() {
        let input = [1.0_f32, -2.0, 0.125, 0.0];
        let mut packed = [0_u8; 4];
        quantize_f32(ColdFloatFormat::E5M2, &input, &mut packed)
            .expect("four F32 values fit the E5M2 payload");
        let mut output = [0.0_f32; 4];
        dequantize(ColdFloatFormat::E5M2, &packed, &mut output)
            .expect("E5M2 payload contains four values");
        assert_eq!(output, input);
    }
}
