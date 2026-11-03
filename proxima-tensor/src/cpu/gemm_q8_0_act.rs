use super::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, _mm_add_epi32, _mm_and_si128, _mm_cvtph_ps, _mm_cvtsi32_si128, _mm_cvtsi128_si32,
    _mm_cvtss_f32, _mm_loadu_si128, _mm_set1_epi8, _mm_shuffle_epi32, _mm_srli_epi16,
    _mm_unpackhi_epi64, _mm256_and_si256, _mm256_andnot_si256, _mm256_castsi128_si256,
    _mm256_castsi256_si128, _mm256_cmpeq_epi8, _mm256_extracti128_si256, _mm256_inserti128_si256,
    _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi8,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi8, _mm256_shuffle_epi8, _mm256_sign_epi8,
    _mm256_sub_epi8,
};

use proxima_gguf::quant::{q5_0, q8_0};

pub(super) const Q8_0_ACT_BLOCK_BYTES: usize = q8_0::BLOCK_BYTES;
const BLOCK_ELEMENTS: usize = q8_0::QK8_0;

pub(super) type ActivationDotFn = fn(&[u8], &[u8]) -> Result<f32, TensorError>;

pub(super) fn quantize_row_q8_0_activation(
    activation: &[f32],
    output: &mut [u8],
) -> Result<(), TensorError> {
    q8_0::quantize(activation, output).map_err(|_| TensorError::QuantizedShapeMismatch {
        reason: "q8_0 activation length is not a whole multiple of the q8_0 block size",
    })
}

fn f16_scale(block: &[u8]) -> f32 {
    half::f16::from_le_bytes([block[0], block[1]]).to_f32()
}

pub(super) fn block_sum_q8_0_scalar(weight_block: &[u8], activation_block: &[u8]) -> i32 {
    let weight_codes = &weight_block[2..2 + BLOCK_ELEMENTS];
    let activation_codes = &activation_block[2..2 + BLOCK_ELEMENTS];
    weight_codes
        .iter()
        .zip(activation_codes)
        .map(|(&weight, &activation)| {
            i32::from(weight.cast_signed()) * i32::from(activation.cast_signed())
        })
        .sum()
}

pub(super) fn block_sum_q5_0_scalar(weight_block: &[u8], activation_block: &[u8]) -> i32 {
    let high_bits = u32::from_le_bytes([
        weight_block[2],
        weight_block[3],
        weight_block[4],
        weight_block[5],
    ]);
    let nibbles = &weight_block[6..6 + BLOCK_ELEMENTS / 2];
    let activation_codes = &activation_block[2..2 + BLOCK_ELEMENTS];
    let mut low_sum = 0i32;
    let mut high_sum = 0i32;
    for (index, &packed) in nibbles.iter().enumerate() {
        let high_low = ((high_bits & (1u32 << index)) >> index) << 4;
        let high_high = (high_bits & (1u32 << (index + 16))) >> (index + 12);
        let low = (i32::from(packed & 0x0f) | high_low as i32) - 16;
        let high = (i32::from(packed >> 4) | high_high as i32) - 16;
        low_sum += low * i32::from(activation_codes[index].cast_signed());
        high_sum += high * i32::from(activation_codes[index + BLOCK_ELEMENTS / 2].cast_signed());
    }
    low_sum + high_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum(products: __m256i) -> i32 {
    let ones = _mm256_set1_epi16(1);
    let widened = _mm256_madd_epi16(products, ones);
    let sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(widened),
        _mm256_extracti128_si256(widened, 1),
    );
    let sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b01));
    _mm_cvtsi128_si32(sum32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn signed_dot(weights: __m256i, activations: __m256i) -> i32 {
    let magnitudes = _mm256_sign_epi8(weights, weights);
    let signed_activations = _mm256_sign_epi8(activations, weights);
    // SAFETY: avx2 is enabled on this function.
    unsafe { horizontal_sum(_mm256_maddubs_epi16(magnitudes, signed_activations)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn block_sum_q8_0_avx2(weight_block: &[u8], activation_block: &[u8]) -> i32 {
    let weight_codes = &weight_block[2..2 + BLOCK_ELEMENTS];
    let activation_codes = &activation_block[2..2 + BLOCK_ELEMENTS];
    // SAFETY: both slices are exactly 32 readable bytes; loadu needs no alignment.
    unsafe {
        let weights = _mm256_loadu_si256(weight_codes.as_ptr().cast());
        let activations = _mm256_loadu_si256(activation_codes.as_ptr().cast());
        signed_dot(weights, activations)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn block_sum_q5_0_avx2(weight_block: &[u8], activation_block: &[u8]) -> i32 {
    let high_bits = i32::from_le_bytes([
        weight_block[2],
        weight_block[3],
        weight_block[4],
        weight_block[5],
    ]);
    let nibbles = &weight_block[6..6 + BLOCK_ELEMENTS / 2];
    let activation_codes = &activation_block[2..2 + BLOCK_ELEMENTS];
    let spread = _mm256_shuffle_epi8(
        _mm256_set1_epi32(high_bits),
        _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
            3, 3, 3,
        ),
    );
    let bit_masks = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
        1, 2, 4, 8, 16, 32, 64, -128,
    );
    let bit_set = _mm256_cmpeq_epi8(_mm256_and_si256(spread, bit_masks), bit_masks);
    let missing_fifth_bit = _mm256_andnot_si256(bit_set, _mm256_set1_epi8(16));
    // SAFETY: `nibbles` is 16 readable bytes and `activation_codes` 32; loadu needs no alignment.
    unsafe {
        let packed = _mm_loadu_si128(nibbles.as_ptr().cast());
        let low_mask = _mm_set1_epi8(0x0f);
        let low = _mm_and_si128(packed, low_mask);
        let high = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);
        let codes = _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
        let weights = _mm256_sub_epi8(codes, missing_fifth_bit);
        let activations = _mm256_loadu_si256(activation_codes.as_ptr().cast());
        signed_dot(weights, activations)
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
fn avx2_available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn scale_f16c(block: &[u8]) -> f32 {
    let bits = i32::from(u16::from_le_bytes([block[0], block[1]]));
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn row_q8_0_avx2(weight_row: &[u8], activation_q8_0: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    for (weight_block, activation_block) in weight_row
        .as_chunks::<{ q8_0::BLOCK_BYTES }>()
        .0
        .iter()
        .zip(activation_q8_0.as_chunks::<Q8_0_ACT_BLOCK_BYTES>().0.iter())
    {
        // SAFETY: avx2 and f16c are enabled on this function; blocks are exact-size arrays.
        let (block_sum, weight_scale, activation_scale) = unsafe {
            (
                block_sum_q8_0_avx2(weight_block, activation_block),
                scale_f16c(weight_block),
                scale_f16c(activation_block),
            )
        };
        sum += (weight_scale * activation_scale) * block_sum as f32;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn row_q5_0_avx2(weight_row: &[u8], activation_q8_0: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    for (weight_block, activation_block) in weight_row
        .as_chunks::<{ q5_0::BLOCK_BYTES }>()
        .0
        .iter()
        .zip(activation_q8_0.as_chunks::<Q8_0_ACT_BLOCK_BYTES>().0.iter())
    {
        // SAFETY: avx2 and f16c are enabled on this function; blocks are exact-size arrays.
        let (block_sum, weight_scale, activation_scale) = unsafe {
            (
                block_sum_q5_0_avx2(weight_block, activation_block),
                scale_f16c(weight_block),
                scale_f16c(activation_block),
            )
        };
        sum += (weight_scale * activation_scale) * block_sum as f32;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
fn row_kernels_available() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c")
}

fn block_dot_row(
    weight_row: &[u8],
    activation_q8_0: &[u8],
    weight_block_bytes: usize,
    scalar: fn(&[u8], &[u8]) -> i32,
    #[cfg(target_arch = "x86_64")] vectorized_row: unsafe fn(&[u8], &[u8]) -> f32,
) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(weight_block_bytes) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of its codec's block size",
        });
    }
    let block_count = weight_row.len() / weight_block_bytes;
    if activation_q8_0.len() != block_count * Q8_0_ACT_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_0 activation length does not match the weight row's block count",
        });
    }
    #[cfg(target_arch = "x86_64")]
    if row_kernels_available() {
        // SAFETY: avx2 and f16c were detected at runtime.
        return Ok(unsafe { vectorized_row(weight_row, activation_q8_0) });
    }
    let mut sum = 0.0f32;
    for (weight_block, activation_block) in weight_row
        .chunks_exact(weight_block_bytes)
        .zip(activation_q8_0.as_chunks::<Q8_0_ACT_BLOCK_BYTES>().0.iter())
    {
        #[cfg(target_arch = "x86_64")]
        let block_sum = scalar(weight_block, activation_block);
        sum += (f16_scale(weight_block) * f16_scale(activation_block)) * block_sum as f32;
    }
    Ok(sum)
}

#[cfg(target_arch = "x86_64")]
pub(super) fn dot_q8_0_q8_0(weight_row: &[u8], activation_q8_0: &[u8]) -> Result<f32, TensorError> {
    block_dot_row(
        weight_row,
        activation_q8_0,
        q8_0::BLOCK_BYTES,
        block_sum_q8_0_scalar,
        row_q8_0_avx2,
    )
}

#[cfg(not(target_arch = "x86_64"))]
pub(super) fn dot_q8_0_q8_0(weight_row: &[u8], activation_q8_0: &[u8]) -> Result<f32, TensorError> {
    block_dot_row(
        weight_row,
        activation_q8_0,
        q8_0::BLOCK_BYTES,
        block_sum_q8_0_scalar,
    )
}

#[cfg(target_arch = "x86_64")]
pub(super) fn dot_q5_0_q8_0(weight_row: &[u8], activation_q8_0: &[u8]) -> Result<f32, TensorError> {
    block_dot_row(
        weight_row,
        activation_q8_0,
        q5_0::BLOCK_BYTES,
        block_sum_q5_0_scalar,
        row_q5_0_avx2,
    )
}

#[cfg(not(target_arch = "x86_64"))]
pub(super) fn dot_q5_0_q8_0(weight_row: &[u8], activation_q8_0: &[u8]) -> Result<f32, TensorError> {
    block_dot_row(
        weight_row,
        activation_q8_0,
        q5_0::BLOCK_BYTES,
        block_sum_q5_0_scalar,
    )
}

pub(super) fn activation_dot_for(codec: Codec) -> ActivationDotFn {
    match codec {
        Codec::Q5_0 => dot_q5_0_q8_0,
        _ => dot_q8_0_q8_0,
    }
}

pub(super) fn matmul_q8_0_activation_impl(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
    leading_total: usize,
    session: Option<&MatmulSession<'_>>,
    dot: ActivationDotFn,
) -> Result<Vec<f32>, TensorError> {
    if rows == 0 || !weights.len().is_multiple_of(rows) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight byte length is not a whole multiple of the row count",
        });
    }
    if leading_total == 0 || !activation.len().is_multiple_of(leading_total) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the position count",
        });
    }
    let k = activation.len() / leading_total;
    if !k.is_multiple_of(BLOCK_ELEMENTS) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the q8_0 block size",
        });
    }
    let quantized_row_bytes = (k / BLOCK_ELEMENTS) * Q8_0_ACT_BLOCK_BYTES;
    let mut quantized = vec![0u8; leading_total * quantized_row_bytes];
    quantize_row_q8_0_activation(activation, &mut quantized)?;

    let row_bytes = weights.len() / rows;
    match quantized_matmul_workers(rows, activation.len()) {
        Some(workers) => {
            matmul_rows_threaded(rows, leading_total, workers, session, k, |row, slot| {
                let weight_row = &weights[row * row_bytes..(row + 1) * row_bytes];
                for (position, output_slot) in slot.iter_mut().enumerate() {
                    let start = position * quantized_row_bytes;
                    *output_slot = dot(weight_row, &quantized[start..start + quantized_row_bytes])?;
                }
                Ok(())
            })
        }
        None => weights.chunks_exact(row_bytes).try_fold(
            Vec::with_capacity(rows * leading_total),
            |mut output, weight_row| {
                for position in 0..leading_total {
                    let start = position * quantized_row_bytes;
                    output.push(dot(
                        weight_row,
                        &quantized[start..start + quantized_row_bytes],
                    )?);
                }
                Ok::<Vec<f32>, TensorError>(output)
            },
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseError;

    fn reference_nearest(value: f32) -> f32 {
        libm::roundf(value)
    }

    fn reference_quantize(x: &[f32]) -> (f32, Vec<i8>) {
        let amax = x.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        let d = amax / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        let levels = x
            .iter()
            .map(|&value| reference_nearest(value * id) as i8)
            .collect();
        (half::f16::from_f32(d).to_f32(), levels)
    }

    fn reference_q5_0_codes(block: &[u8]) -> Vec<i32> {
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let mut codes = vec![0i32; 32];
        for j in 0..16 {
            let xh_0 = ((qh & (1 << j)) >> j) << 4;
            let xh_1 = (qh & (1 << (j + 16))) >> (j + 12);
            codes[j] = (i32::from(block[6 + j] & 0x0f) | xh_0 as i32) - 16;
            codes[j + 16] = (i32::from(block[6 + j] >> 4) | xh_1 as i32) - 16;
        }
        codes
    }

    fn finite_scale(mut bytes: Vec<u8>) -> Vec<u8> {
        let scale = half::f16::from_f32(f32::from(bytes[0]) / 64.0 + 0.01).to_le_bytes();
        bytes[0] = scale[0];
        bytes[1] = scale[1];
        bytes
    }

    fn activation_strategy() -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(
            prop_oneof![8 => -3.0f32..3.0f32, 1 => -200.0f32..200.0f32, 1 => Just(0.0f32)],
            32,
        )
    }

    proptest! {
        #[test]
        fn activation_quantize_matches_llama_reference(x in activation_strategy()) {
            let mut ours = vec![0u8; Q8_0_ACT_BLOCK_BYTES];
            prop_assert!(quantize_row_q8_0_activation(&x, &mut ours).is_ok());
            let (reference_d, reference_levels) = reference_quantize(&x);
            prop_assert_eq!(f16_scale(&ours).to_bits(), reference_d.to_bits());
            let ours_levels: Vec<i8> = ours[2..].iter().map(|&byte| byte.cast_signed()).collect();
            prop_assert_eq!(ours_levels, reference_levels);
        }

        #[test]
        fn q8_0_dot_matches_llama_reference_and_avx2_is_identical(
            weights in prop::collection::vec(any::<u8>(), q8_0::BLOCK_BYTES),
            x in activation_strategy(),
        ) {
            let block = finite_scale(weights);
            let mut packed = vec![0u8; Q8_0_ACT_BLOCK_BYTES];
            prop_assert!(quantize_row_q8_0_activation(&x, &mut packed).is_ok());
            let (activation_d, levels) = reference_quantize(&x);
            let expected_sum: i32 = block[2..].iter().zip(&levels).map(|(&w, &a)| i32::from(w.cast_signed()) * i32::from(a)).sum();
            let expected = f64::from(expected_sum) * f64::from(f16_scale(&block) * activation_d);
            let ours = dot_q8_0_q8_0(&block, &packed).map_err(|error| TestCaseError::fail(format!("{error:?}")))?;
            prop_assert_eq!(block_sum_q8_0_scalar(&block, &packed), expected_sum);
            prop_assert!((f64::from(ours) - expected).abs() <= 2e-6 * expected.abs().max(1e-6));
            #[cfg(target_arch = "x86_64")]
            if avx2_available() {
                // SAFETY: avx2 detected above.
                prop_assert_eq!(unsafe { block_sum_q8_0_avx2(&block, &packed) }, expected_sum);
            }
        }

        #[test]
        fn q5_0_dot_matches_llama_reference_and_avx2_is_identical(
            weights in prop::collection::vec(any::<u8>(), q5_0::BLOCK_BYTES),
            x in activation_strategy(),
        ) {
            let block = finite_scale(weights);
            let mut packed = vec![0u8; Q8_0_ACT_BLOCK_BYTES];
            prop_assert!(quantize_row_q8_0_activation(&x, &mut packed).is_ok());
            let (activation_d, levels) = reference_quantize(&x);
            let codes = reference_q5_0_codes(&block);
            let expected_sum: i32 = codes.iter().zip(&levels).map(|(&w, &a)| w * i32::from(a)).sum();
            let expected = f64::from(expected_sum) * f64::from(f16_scale(&block) * activation_d);
            let ours = dot_q5_0_q8_0(&block, &packed).map_err(|error| TestCaseError::fail(format!("{error:?}")))?;
            prop_assert_eq!(block_sum_q5_0_scalar(&block, &packed), expected_sum);
            prop_assert!((f64::from(ours) - expected).abs() <= 2e-6 * expected.abs().max(1e-6));
            #[cfg(target_arch = "x86_64")]
            if avx2_available() {
                // SAFETY: avx2 detected above.
                prop_assert_eq!(unsafe { block_sum_q5_0_avx2(&block, &packed) }, expected_sum);
            }
        }
    }
}
