use core::arch::x86_64::{
    __m128i, __m256, __m256i, _mm_and_si128, _mm_loadl_epi64, _mm_or_si128, _mm_set1_epi8,
    _mm_slli_epi16, _mm_srli_epi16, _mm256_add_epi32, _mm256_and_si256, _mm256_cmpeq_epi32,
    _mm256_cvtepi8_epi32, _mm256_cvtepi32_ps, _mm256_cvtepu8_epi32, _mm256_mul_ps,
    _mm256_set1_epi32, _mm256_set1_ps, _mm256_setr_epi32, _mm256_storeu_ps, _mm256_sub_epi32,
    _mm256_sub_ps,
};

use proxima_gguf::quant::{q4_k, q5_0, q6_k, q8_0};

fn avx2_available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

fn f16_scale(block: &[u8], offset: usize) -> f32 {
    half::f16::from_le_bytes([block[offset], block[offset + 1]]).to_f32()
}

#[target_feature(enable = "avx2")]
unsafe fn load_eight_bytes(source: &[u8]) -> __m128i {
    let eight = &source[..8];
    // SAFETY: `eight` is exactly 8 readable bytes; loadl reads 8 and needs no alignment.
    unsafe { _mm_loadl_epi64(eight.as_ptr().cast()) }
}

#[target_feature(enable = "avx2")]
unsafe fn store_eight(destination: &mut [f32], value: __m256) {
    let eight = &mut destination[..8];
    // SAFETY: `eight` is exactly 8 writable f32; storeu needs no alignment.
    unsafe { _mm256_storeu_ps(eight.as_mut_ptr(), value) };
}

#[target_feature(enable = "avx2")]
unsafe fn signed_codes_to_f32(bytes: __m128i) -> __m256 {
    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes))
}

#[target_feature(enable = "avx2")]
unsafe fn unsigned_codes_to_f32(bytes: __m128i) -> __m256 {
    _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes))
}

#[target_feature(enable = "avx2")]
unsafe fn dequantize_q8_0_avx2(block: &[u8], output: &mut [f32]) {
    let scale = _mm256_set1_ps(f16_scale(block, 0));
    let codes = &block[2..2 + q8_0::QK8_0];
    for chunk in 0..q8_0::QK8_0 / 8 {
        // SAFETY: avx2 is enabled on this function; each slice below is at least 8 elements.
        unsafe {
            let value = signed_codes_to_f32(load_eight_bytes(&codes[chunk * 8..]));
            store_eight(&mut output[chunk * 8..], _mm256_mul_ps(value, scale));
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn fifth_bit_offsets(high_bits: u32, first_element: usize) -> __m256i {
    let selected = _mm256_set1_epi32(((high_bits >> first_element) & 0xff) as i32);
    let lane_bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    let present = _mm256_cmpeq_epi32(_mm256_and_si256(selected, lane_bits), lane_bits);
    _mm256_and_si256(present, _mm256_set1_epi32(16))
}

#[target_feature(enable = "avx2")]
unsafe fn dequantize_q5_0_avx2(block: &[u8], output: &mut [f32]) {
    let scale = _mm256_set1_ps(f16_scale(block, 0));
    let high_bits = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let nibbles = &block[6..22];
    let low_mask = _mm_set1_epi8(0x0f);
    let sixteen = _mm256_set1_epi32(16);
    for chunk in 0..4 {
        let half = chunk / 2;
        let first_element = chunk * 8;
        // SAFETY: avx2 is enabled on this function; every slice below is at least 8 elements.
        unsafe {
            let bytes = load_eight_bytes(&nibbles[(chunk % 2) * 8..]);
            let nibble = if half == 0 {
                _mm_and_si128(bytes, low_mask)
            } else {
                _mm_and_si128(_mm_srli_epi16(bytes, 4), low_mask)
            };
            let code = _mm256_add_epi32(
                _mm256_cvtepu8_epi32(nibble),
                fifth_bit_offsets(high_bits, first_element),
            );
            let value = _mm256_cvtepi32_ps(_mm256_sub_epi32(code, sixteen));
            store_eight(&mut output[first_element..], _mm256_mul_ps(value, scale));
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn dequantize_q4_k_avx2(block: &[u8], output: &mut [f32]) {
    let d = f16_scale(block, 0);
    let dmin = f16_scale(block, 2);
    let mut scales = [0u8; q4_k::K_SCALE_SIZE];
    scales.copy_from_slice(&block[4..4 + q4_k::K_SCALE_SIZE]);
    let nibbles = &block[4 + q4_k::K_SCALE_SIZE..];
    let low_mask = _mm_set1_epi8(0x0f);
    for group in 0..q4_k::QK_K / 64 {
        let (scale_low, min_low) = q4_k::get_scale_min_k4(group * 2, &scales);
        let (scale_high, min_high) = q4_k::get_scale_min_k4(group * 2 + 1, &scales);
        let low_scale = _mm256_set1_ps(d * f32::from(scale_low));
        let low_min = _mm256_set1_ps(dmin * f32::from(min_low));
        let high_scale = _mm256_set1_ps(d * f32::from(scale_high));
        let high_min = _mm256_set1_ps(dmin * f32::from(min_high));
        for chunk in 0..4 {
            // SAFETY: avx2 is enabled on this function; every slice below is at least 8 elements.
            unsafe {
                let bytes = load_eight_bytes(&nibbles[group * 32 + chunk * 8..]);
                let low = unsigned_codes_to_f32(_mm_and_si128(bytes, low_mask));
                let high = unsigned_codes_to_f32(_mm_and_si128(_mm_srli_epi16(bytes, 4), low_mask));
                let base = group * 64 + chunk * 8;
                store_eight(
                    &mut output[base..],
                    _mm256_sub_ps(_mm256_mul_ps(low_scale, low), low_min),
                );
                store_eight(
                    &mut output[base + 32..],
                    _mm256_sub_ps(_mm256_mul_ps(high_scale, high), high_min),
                );
            }
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn q6_k_lane_levels(low_nibbles: __m128i, high_pairs: __m128i, lane: usize) -> __m128i {
    let low_mask = _mm_set1_epi8(0x0f);
    let pair_mask = _mm_set1_epi8(0x03);
    let low_part = if lane < 2 {
        _mm_and_si128(low_nibbles, low_mask)
    } else {
        _mm_and_si128(_mm_srli_epi16(low_nibbles, 4), low_mask)
    };
    let shifted = match lane {
        0 => high_pairs,
        1 => _mm_srli_epi16(high_pairs, 2),
        2 => _mm_srli_epi16(high_pairs, 4),
        _ => _mm_srli_epi16(high_pairs, 6),
    };
    let top = _mm_slli_epi16(_mm_and_si128(shifted, pair_mask), 4);
    _mm_or_si128(low_part, top)
}

#[target_feature(enable = "avx2")]
unsafe fn dequantize_q6_k_avx2(block: &[u8], output: &mut [f32]) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_scale(block, 208);
    let bias = _mm256_set1_ps(32.0);
    for half in 0..2 {
        for chunk in 0..4 {
            let position = chunk * 8;
            let sub_block = position / q6_k::SUB_BLOCK_ELEMENTS;
            // SAFETY: avx2 is enabled on this function; every slice below is at least 8 elements.
            unsafe {
                let ql_near = load_eight_bytes(&ql[half * 64 + position..]);
                let ql_far = load_eight_bytes(&ql[half * 64 + position + 32..]);
                let high_pairs = load_eight_bytes(&qh[half * 32 + position..]);
                for lane in 0..4 {
                    let low_nibbles = if lane % 2 == 0 { ql_near } else { ql_far };
                    let levels = q6_k_lane_levels(low_nibbles, high_pairs, lane);
                    let quant = _mm256_sub_ps(unsigned_codes_to_f32(levels), bias);
                    let scale = f32::from(scales[half * 8 + sub_block + lane * 2] as i8);
                    let factor = _mm256_set1_ps(d * scale);
                    store_eight(
                        &mut output[half * 128 + lane * 32 + position..],
                        _mm256_mul_ps(factor, quant),
                    );
                }
            }
        }
    }
}

pub(super) fn dequantize_block_q8_0(block: &[u8], output: &mut [f32]) {
    if avx2_available() && block.len() >= q8_0::BLOCK_BYTES && output.len() >= q8_0::QK8_0 {
        // SAFETY: avx2 detected; lengths checked above.
        return unsafe { dequantize_q8_0_avx2(block, output) };
    }
    q8_0::dequantize_block(block, output);
}

pub(super) fn dequantize_block_q5_0(block: &[u8], output: &mut [f32]) {
    if avx2_available() && block.len() >= q5_0::BLOCK_BYTES && output.len() >= q5_0::QK5_0 {
        // SAFETY: avx2 detected; lengths checked above.
        return unsafe { dequantize_q5_0_avx2(block, output) };
    }
    q5_0::dequantize_block(block, output);
}

pub(super) fn dequantize_block_q4_k(block: &[u8], output: &mut [f32]) {
    if avx2_available() && block.len() >= q4_k::BLOCK_BYTES && output.len() >= q4_k::QK_K {
        // SAFETY: avx2 detected; lengths checked above.
        return unsafe { dequantize_q4_k_avx2(block, output) };
    }
    q4_k::dequantize_block(block, output);
}

pub(super) fn dequantize_block_q6_k(block: &[u8], output: &mut [f32]) {
    if avx2_available() && block.len() >= q6_k::BLOCK_BYTES && output.len() >= q6_k::QK_K {
        // SAFETY: avx2 detected; lengths checked above.
        return unsafe { dequantize_q6_k_avx2(block, output) };
    }
    q6_k::dequantize_block(block, output);
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn block_bytes(length: usize) -> impl Strategy<Value = Vec<u8>> {
        prop::collection::vec(any::<u8>(), length)
    }

    fn finite_scale_block(mut bytes: Vec<u8>, scale_offsets: &[usize]) -> Vec<u8> {
        for &offset in scale_offsets {
            let scale = half::f16::from_f32(f32::from(bytes[offset]) / 16.0 - 8.0);
            let encoded = scale.to_le_bytes();
            bytes[offset] = encoded[0];
            bytes[offset + 1] = encoded[1];
        }
        bytes
    }

    fn assert_bits_equal(vectorized: &[f32], scalar: &[f32]) {
        let left: Vec<u32> = vectorized.iter().map(|value| value.to_bits()).collect();
        let right: Vec<u32> = scalar.iter().map(|value| value.to_bits()).collect();
        assert_eq!(left, right);
    }

    proptest! {
        #[test]
        fn q8_0_avx2_matches_scalar(raw in block_bytes(q8_0::BLOCK_BYTES)) {
            prop_assume!(avx2_available());
            let block = finite_scale_block(raw, &[0]);
            let mut scalar = [0.0f32; q8_0::QK8_0];
            let mut vectorized = [0.0f32; q8_0::QK8_0];
            q8_0::dequantize_block(&block, &mut scalar);
            // SAFETY: avx2 checked by prop_assume above.
            unsafe { dequantize_q8_0_avx2(&block, &mut vectorized) };
            assert_bits_equal(&vectorized, &scalar);
        }

        #[test]
        fn q5_0_avx2_matches_scalar(raw in block_bytes(q5_0::BLOCK_BYTES)) {
            prop_assume!(avx2_available());
            let block = finite_scale_block(raw, &[0]);
            let mut scalar = [0.0f32; q5_0::QK5_0];
            let mut vectorized = [0.0f32; q5_0::QK5_0];
            q5_0::dequantize_block(&block, &mut scalar);
            // SAFETY: avx2 checked by prop_assume above.
            unsafe { dequantize_q5_0_avx2(&block, &mut vectorized) };
            assert_bits_equal(&vectorized, &scalar);
        }

        #[test]
        fn q4_k_avx2_matches_scalar(raw in block_bytes(q4_k::BLOCK_BYTES)) {
            prop_assume!(avx2_available());
            let block = finite_scale_block(raw, &[0, 2]);
            let mut scalar = [0.0f32; q4_k::QK_K];
            let mut vectorized = [0.0f32; q4_k::QK_K];
            q4_k::dequantize_block(&block, &mut scalar);
            // SAFETY: avx2 checked by prop_assume above.
            unsafe { dequantize_q4_k_avx2(&block, &mut vectorized) };
            assert_bits_equal(&vectorized, &scalar);
        }

        #[test]
        fn q6_k_avx2_matches_scalar(raw in block_bytes(q6_k::BLOCK_BYTES)) {
            prop_assume!(avx2_available());
            let block = finite_scale_block(raw, &[208]);
            let mut scalar = [0.0f32; q6_k::QK_K];
            let mut vectorized = [0.0f32; q6_k::QK_K];
            q6_k::dequantize_block(&block, &mut scalar);
            // SAFETY: avx2 checked by prop_assume above.
            unsafe { dequantize_q6_k_avx2(&block, &mut vectorized) };
            assert_bits_equal(&vectorized, &scalar);
        }
    }

    #[test]
    fn dispatch_wrappers_agree_with_scalar_on_a_fixed_block() {
        let block: Vec<u8> = (0..q6_k::BLOCK_BYTES as u32)
            .map(|index| (index * 31 % 251) as u8)
            .collect();
        let block = finite_scale_block(block, &[208]);
        let mut scalar = [0.0f32; q6_k::QK_K];
        let mut wrapped = [0.0f32; q6_k::QK_K];
        q6_k::dequantize_block(&block, &mut scalar);
        dequantize_block_q6_k(&block, &mut wrapped);
        assert_bits_equal(&wrapped, &scalar);
    }
}
