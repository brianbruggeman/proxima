use super::*;

use core::arch::x86_64::{
    __m256i, _mm_add_epi32, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_set1_epi16,
    _mm_shuffle_epi32, _mm_unpackhi_epi64, _mm256_add_epi32, _mm256_and_si256,
    _mm256_castsi128_si256, _mm256_castsi256_si128, _mm256_extracti128_si256,
    _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
    _mm256_or_si256, _mm256_set1_epi8, _mm256_setzero_si256, _mm256_sll_epi16, _mm256_srl_epi16,
};

#[target_feature(enable = "avx2")]
unsafe fn lane_levels(ql_pair: __m256i, high_pairs: __m256i, lane: usize) -> __m256i {
    let low_mask = _mm256_set1_epi8(0x0f);
    let pair_mask = _mm256_set1_epi8(0x03);
    let low_shift = _mm_cvtsi32_si128(if lane < 2 { 0 } else { 4 });
    let high_shift = _mm_cvtsi32_si128(2 * lane as i32);
    let nibble = _mm256_and_si256(_mm256_srl_epi16(ql_pair, low_shift), low_mask);
    let pair = _mm256_and_si256(_mm256_srl_epi16(high_pairs, high_shift), pair_mask);
    _mm256_or_si256(nibble, _mm256_sll_epi16(pair, _mm_cvtsi32_si128(4)))
}

#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum(lanes: __m256i) -> i32 {
    let sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(lanes),
        _mm256_extracti128_si256(lanes, 1),
    );
    let sum64 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, sum128));
    _mm_cvtsi128_si32(_mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b01)))
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn dot_q6k_q8k_block_avx2(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q6K_D_OFFSET);
    let scales = &weight_block[Q6K_SCALES_OFFSET..Q6K_SCALES_OFFSET + 16];
    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);
    let bsums = &q8k_block[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];

    let mut bias = 0i32;
    for (sub_block, &scale) in scales.iter().enumerate() {
        let bsum = i16::from_le_bytes([bsums[sub_block * 2], bsums[sub_block * 2 + 1]]);
        bias += i32::from(scale.cast_signed()) * i32::from(bsum);
    }

    let mut accumulator = _mm256_setzero_si256();
    for half in 0..2 {
        // SAFETY: every offset below stays inside the 210-byte weight block and the
        // 292-byte q8_k block; loadu needs no alignment.
        unsafe {
            let ql_base = weight_block[Q6K_QL_OFFSET + half * 64..].as_ptr();
            let ql_near = _mm256_loadu_si256(ql_base.cast());
            let ql_far = _mm256_loadu_si256(ql_base.add(32).cast());
            let high_pairs =
                _mm256_loadu_si256(weight_block[Q6K_QH_OFFSET + half * 32..].as_ptr().cast());
            for lane in 0..4 {
                let ql_pair = if lane % 2 == 0 { ql_near } else { ql_far };
                let weights = lane_levels(ql_pair, high_pairs, lane);
                let activations = _mm256_loadu_si256(
                    q8k_block[Q8K_QS_OFFSET + half * 128 + lane * 32..]
                        .as_ptr()
                        .cast(),
                );
                let first = i16::from(scales[half * 8 + lane * 2].cast_signed());
                let second = i16::from(scales[half * 8 + lane * 2 + 1].cast_signed());
                let scale_vector = _mm256_inserti128_si256(
                    _mm256_castsi128_si256(_mm_set1_epi16(first)),
                    _mm_set1_epi16(second),
                    1,
                );
                let products = _mm256_maddubs_epi16(weights, activations);
                accumulator =
                    _mm256_add_epi32(accumulator, _mm256_madd_epi16(products, scale_vector));
            }
        }
    }
    // SAFETY: avx2 is enabled on this function.
    let sumi = unsafe { horizontal_sum(accumulator) } - 32 * bias;
    (activation_scale * d_weight) * sumi as f32
}
