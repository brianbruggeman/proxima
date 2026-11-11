//! Rust transcription of three llama.cpp reference routines, kept as the external incumbent for
//! `tq2_0` and `mxfp4`, formats proxima-gguf has no routine of its own for (principle 14).
//!
//! Source: llama.cpp commit `f1ea206218210afb913ae2f5d2c51faed35915da`.
//! - `quantize_row_tq2_0_ref`: `ggml/src/ggml-quants.c:2382-2411`, layout `block_tq2_0` at
//!   `ggml/src/ggml-common.h:284-288` (`qs[64]` then little-endian f16 `d`, 66 bytes).
//! - `dequantize_row_tq2_0`: `ggml/src/ggml-quants.c:2467-2483`.
//! - `dequantize_row_mxfp4`: `ggml/src/ggml-quants.c:569-585`, layout `block_mxfp4` at
//!   `ggml/src/ggml-common.h:214-219` (`e` then `qs[16]`, 17 bytes), `kvalues_fp4` at
//!   `ggml/src/ggml-common.h:1126-1128`.
//! - helpers: `MAX` `ggml/src/ggml-impl.h:40`, `ggml_compute_fp16_to_fp32` / `_fp32_to_fp16`
//!   `ggml/src/ggml-impl.h:395-445`, `ggml_e8m0_to_fp32_half` `ggml/src/ggml-impl.h:487-510`.
//!
//! Loop structure and operation order follow the C line by line so a reader can diff them.

extern crate std;

const QK_K: usize = 256;
const TQ2_0_QS_BYTES: usize = QK_K / 4;
const TQ2_0_BLOCK_BYTES: usize = TQ2_0_QS_BYTES + 2;
const QK_MXFP4: usize = 32;
const MXFP4_BLOCK_BYTES: usize = 1 + QK_MXFP4 / 2;

const KVALUES_FP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

// ggml-impl.h:40, a NaN second operand wins and a NaN first operand loses
fn max_macro(first: f32, second: f32) -> f32 {
    if first > second { first } else { second }
}

fn ggml_compute_fp16_to_fp32(half: u16) -> f32 {
    let word = (half as u32) << 16;
    let sign = word & 0x8000_0000;
    let two_w = word.wrapping_add(word);

    let exp_offset: u32 = 0xE0 << 23;
    let exp_scale = f32::from_bits(0x0780_0000);
    let normalized_value = f32::from_bits((two_w >> 4).wrapping_add(exp_offset)) * exp_scale;

    let magic_mask: u32 = 126 << 23;
    let magic_bias = 0.5f32;
    let denormalized_value = f32::from_bits((two_w >> 17) | magic_mask) - magic_bias;

    let denormalized_cutoff: u32 = 1 << 27;
    let result = sign
        | if two_w < denormalized_cutoff {
            denormalized_value.to_bits()
        } else {
            normalized_value.to_bits()
        };
    f32::from_bits(result)
}

fn ggml_compute_fp32_to_fp16(value: f32) -> u16 {
    let scale_to_inf = f32::from_bits(0x7780_0000);
    let scale_to_zero = f32::from_bits(0x0880_0000);
    let mut base = (libm::fabsf(value) * scale_to_inf) * scale_to_zero;

    let word = value.to_bits();
    let shl1_w = word.wrapping_add(word);
    let sign = word & 0x8000_0000;
    let mut bias = shl1_w & 0xFF00_0000;
    if bias < 0x7100_0000 {
        bias = 0x7100_0000;
    }

    base += f32::from_bits((bias >> 1).wrapping_add(0x0780_0000));
    let bits = base.to_bits();
    let exp_bits = (bits >> 13) & 0x0000_7C00;
    let mantissa_bits = bits & 0x0000_0FFF;
    let nonsign = exp_bits + mantissa_bits;
    ((sign >> 16) | if shl1_w > 0xFF00_0000 { 0x7E00 } else { nonsign }) as u16
}

fn ggml_e8m0_to_fp32_half(exponent: u8) -> f32 {
    let bits: u32 = if exponent < 2 {
        0x0020_0000u32 << exponent
    } else {
        (exponent as u32 - 1) << 23
    };
    f32::from_bits(bits)
}

/// `quantize_row_tq2_0_ref`: `values.len()` is a multiple of 256, `out` holds 66 bytes per block.
pub(crate) fn quantize_row_tq2_0_ref(values: &[f32], out: &mut [u8]) {
    assert!(values.len().is_multiple_of(QK_K));
    let block_count = values.len() / QK_K;
    assert_eq!(out.len(), block_count * TQ2_0_BLOCK_BYTES);

    for block_index in 0..block_count {
        let block = &mut out[block_index * TQ2_0_BLOCK_BYTES..][..TQ2_0_BLOCK_BYTES];
        let mut source = &values[block_index * QK_K..];
        let mut amax = 0.0f32;

        for &value in &source[..QK_K] {
            amax = max_macro(amax, libm::fabsf(value));
        }

        // `d` and `id` in the C
        let scale = amax;
        let inverse = if scale != 0.0 { 1.0f32 / scale } else { 0.0f32 };

        block[TQ2_0_QS_BYTES..].copy_from_slice(&ggml_compute_fp32_to_fp16(scale).to_le_bytes());

        let mut byte_base = 0;
        while byte_base < TQ2_0_QS_BYTES {
            for lane in 0..32 {
                let mut packed: u8 = 0;
                for slot in 0..4 {
                    // lroundf of NaN or of an infinity (a subnormal block maximum makes the inverse infinite) is
                    // unspecified in C; the saturating cast and add keep it total and platform-independent here
                    let code = (libm::roundf(source[lane + slot * 32] * inverse) as i64).saturating_add(1) as i32;
                    packed = packed.wrapping_add(((code & 3) << (2 * slot)) as u8);
                }
                block[byte_base + lane] = packed;
            }
            source = &source[4 * 32..];
            byte_base += 32;
        }
    }
}

/// `dequantize_row_tq2_0`: `data` holds 66 bytes per block, `out` 256 floats per block.
pub(crate) fn dequantize_row_tq2_0(data: &[u8], out: &mut [f32]) {
    assert!(out.len().is_multiple_of(QK_K));
    let block_count = out.len() / QK_K;
    assert_eq!(data.len(), block_count * TQ2_0_BLOCK_BYTES);

    let mut cursor = 0;
    for block_index in 0..block_count {
        let block = &data[block_index * TQ2_0_BLOCK_BYTES..][..TQ2_0_BLOCK_BYTES];
        // `d` in the C
        let scale = ggml_compute_fp16_to_fp32(u16::from_le_bytes([block[TQ2_0_QS_BYTES], block[TQ2_0_QS_BYTES + 1]]));

        let mut byte_base = 0;
        while byte_base < TQ2_0_QS_BYTES {
            for shift_slot in 0..4 {
                for lane in 0..32 {
                    let code = ((block[byte_base + lane] >> (shift_slot * 2)) & 3) as i8;
                    out[cursor] = (code - 1) as f32 * scale;
                    cursor += 1;
                }
            }
            byte_base += 32;
        }
    }
}

/// `dequantize_row_mxfp4`: `data` holds 17 bytes per block, `out` 32 floats per block.
pub(crate) fn dequantize_row_mxfp4(data: &[u8], out: &mut [f32]) {
    assert!(out.len().is_multiple_of(QK_MXFP4));
    let block_count = out.len() / QK_MXFP4;
    assert_eq!(data.len(), block_count * MXFP4_BLOCK_BYTES);

    for block_index in 0..block_count {
        let block = &data[block_index * MXFP4_BLOCK_BYTES..][..MXFP4_BLOCK_BYTES];
        // `d` in the C
        let scale = ggml_e8m0_to_fp32_half(block[0]);

        for pair in 0..QK_MXFP4 / 2 {
            let packed = block[1 + pair];
            let low_value = KVALUES_FP4[(packed & 0x0F) as usize];
            let high_value = KVALUES_FP4[(packed >> 4) as usize];

            out[block_index * QK_MXFP4 + pair] = low_value as f32 * scale;
            out[block_index * QK_MXFP4 + pair + QK_MXFP4 / 2] = high_value as f32 * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec;
    use alloc::vec::Vec;

    fn scale_of(block: &[u8]) -> f32 {
        ggml_compute_fp16_to_fp32(u16::from_le_bytes([block[64], block[65]]))
    }

    #[test]
    fn tq2_0_round_trips_a_ternary_block_and_packs_four_elements_per_byte() {
        // value(i) = ((i % 3) - 1) * 0.5, so amax = 0.5, d = f16 0x3800, id = 2.0, codes = i % 3
        // byte 0 holds elements 0,32,64,96 = codes 0,2,1,0 -> 0 + 2<<2 + 1<<4 + 0<<6 = 0x18
        // byte 1 holds elements 1,33,65,97 = codes 1,0,2,1 -> 1 + 0 + 2<<4 + 1<<6 = 0x61
        // byte 32 holds elements 128,160,192,224 = codes 2,1,0,2 -> 2 + 1<<2 + 0 + 2<<6 = 0x86
        let input: Vec<f32> = (0..256).map(|index| ((index % 3) as f32 - 1.0) * 0.5).collect();
        let mut block = vec![0u8; 66];

        quantize_row_tq2_0_ref(&input, &mut block);

        assert_eq!(&block[64..66], &[0x00, 0x38], "f16 0.5 little-endian at the d offset");
        assert_eq!(scale_of(&block), 0.5);
        assert_eq!((block[0], block[1], block[32]), (0x18, 0x61, 0x86));
        let mut decoded = vec![0f32; 256];
        dequantize_row_tq2_0(&block, &mut decoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn tq2_0_a_block_of_the_smallest_subnormal_has_an_infinite_inverse_and_still_quantizes() {
        // amax = f32 bits 1, so id = 1 / 1.4e-45 overflows to +inf and every element is inf; the
        // saturated rounding is i64::MAX, whose low 32 bits are all ones, so every code is 3 and every
        // byte 0xFF; the scale 1.4e-45 underflows to f16 zero
        let input = vec![f32::from_bits(1); 256];
        let mut block = vec![0u8; 66];

        quantize_row_tq2_0_ref(&input, &mut block);

        assert_eq!(&block[64..66], &[0x00, 0x00], "f16 of the subnormal scale is zero");
        assert!(block[..64].iter().all(|byte| *byte == 0xFF), "every element saturates to code 3");
    }

    #[test]
    fn tq2_0_rounds_half_away_from_zero_so_plus_and_minus_half_take_codes_2_and_0() {
        // amax = 2.0, id = 0.5: 1.0 * id = +0.5 -> lroundf 1 -> code 2; -1.0 * id = -0.5 -> -1 -> code 0
        // element 0 = 2.0 (code 2), element 1 = 1.0 (code 2), element 2 = -1.0 (code 0), rest 0 (code 1)
        // byte 0: 2 + 1<<2 + 1<<4 + 1<<6 = 0x56, byte 1 = 0x56, byte 2: 0 + 4 + 16 + 64 = 0x54
        let mut input = vec![0f32; 256];
        input[0] = 2.0;
        input[1] = 1.0;
        input[2] = -1.0;
        let mut block = vec![0u8; 66];

        quantize_row_tq2_0_ref(&input, &mut block);

        assert_eq!(&block[64..66], &[0x00, 0x40], "f16 2.0 little-endian");
        assert_eq!((block[0], block[1], block[2]), (0x56, 0x56, 0x54));
        assert_eq!(block[3], 0x55, "all-zero elements pack code 1 in every slot");
    }

    #[test]
    fn mxfp4_block_with_exponent_128_decodes_kvalues_fp4_at_unit_scale() {
        // e = 128 = 2^1, half scale 1.0: bits (128-1)<<23 = 0x3F800000
        // qs[j] = (15-j)<<4 | j: out[j] = kvalues_fp4[j], out[j+16] = kvalues_fp4[15-j]
        let mut block = vec![128u8];
        block.extend((0..16u8).map(|pair| ((15 - pair) << 4) | pair));
        let mut decoded = vec![0f32; 32];

        dequantize_row_mxfp4(&block, &mut decoded);

        let table: Vec<f32> = KVALUES_FP4.iter().map(|&value| value as f32).collect();
        let reversed: Vec<f32> = table.iter().rev().copied().collect();
        assert_eq!(&decoded[..16], table.as_slice());
        assert_eq!(&decoded[16..], reversed.as_slice());
        assert_eq!(decoded[7], 12.0);
        assert_eq!(decoded[31], 0.0);
        assert_eq!(decoded[17], -8.0);
    }

    #[test]
    fn tq2_0_nan_mid_block_is_dropped_by_max_macro_but_last_element_nan_poisons_the_scale() {
        // MAX(a, b) = (a > b) ? a : b: any comparison with NaN is false, so the second operand wins.
        // NaN at index 0: MAX(0, NaN) = NaN, then MAX(NaN, 0.5) = 0.5, so the scale recovers to 0.5.
        let mut mid = vec![0.5f32; 256];
        mid[0] = f32::NAN;
        let mut block = vec![0u8; 66];
        quantize_row_tq2_0_ref(&mid, &mut block);
        assert_eq!(scale_of(&block), 0.5);
        assert_eq!(block[0] & 3, 1, "NaN * id is NaN, lround 0 under the saturating cast, code 1");

        // NaN at index 255: MAX(0.5, NaN) = NaN is final, d = NaN (truthy), id = NaN, f16 NaN = 0x7E00
        let mut last = vec![0.5f32; 256];
        last[255] = f32::NAN;
        quantize_row_tq2_0_ref(&last, &mut block);
        assert_eq!(&block[64..66], &[0x00, 0x7E]);
        assert!(block[..64].iter().all(|&byte| byte == 0x55), "every element is NaN * id, code 1");
    }
}
