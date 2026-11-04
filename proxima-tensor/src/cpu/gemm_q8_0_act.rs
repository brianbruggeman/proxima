use super::*;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256, __m256i, _mm_add_ps, _mm_add_ss, _mm_and_si128, _mm_cvtph_ps, _mm_cvtsi32_si128,
    _mm_cvtss_f32, _mm_loadu_si128, _mm_movehdup_ps, _mm_movehl_ps, _mm_mul_ss, _mm_set1_epi8,
    _mm_shuffle_ps, _mm_srli_epi16, _mm256_and_si256, _mm256_andnot_si256, _mm256_castps256_ps128,
    _mm256_castsi128_si256, _mm256_cmpeq_epi8, _mm256_cvtepi32_ps, _mm256_extractf128_ps,
    _mm256_fmadd_ps, _mm256_inserti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_set1_epi8, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_ps,
    _mm256_setr_epi8, _mm256_setzero_ps, _mm256_shuffle_epi8, _mm256_storeu_si256, _mm256_loadu_ps, _mm256_andnot_ps, _mm256_max_ps, _mm256_cvtps_epi32, _mm256_packs_epi16, _mm256_packs_epi32, _mm256_permutevar8x32_epi32, _mm256_setr_epi32, _mm_max_ps, _mm_max_ss, _mm256_sign_epi8, _mm256_sub_epi8, _mm256_mul_ps,
};

#[cfg(all(test, target_arch = "x86_64"))]
use core::arch::x86_64::{
    _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_unpackhi_epi64,
    _mm256_castsi256_si128, _mm256_extracti128_si256,
};

use proxima_gguf::quant::{q5_0, q8_0};

pub(super) const Q8_0_ACT_BLOCK_BYTES: usize = q8_0::BLOCK_BYTES;
const BLOCK_ELEMENTS: usize = q8_0::QK8_0;

pub(super) type ActivationDotFn = fn(&[u8], &[u8]) -> Result<f32, TensorError>;

// ggml's cpu activation quantizer, not `quantize_row_q8_0_ref`: on x86 it scales by
// `127 / amax` and rounds half to even (`ggml-cpu/arch/x86/quants.c`
// `quantize_row_q8_0`, `_mm256_round_ps(.., _MM_ROUND_NEAREST)`), which flips
// single codes against the reference's `round(x * (1 / d))`.
pub(super) fn quantize_row_q8_0_activation(
    activation: &[f32],
    output: &mut [u8],
) -> Result<(), TensorError> {
    let shape_error = || TensorError::QuantizedShapeMismatch {
        reason: "q8_0 activation length is not a whole multiple of the q8_0 block size",
    };
    if !activation.len().is_multiple_of(BLOCK_ELEMENTS)
        || output.len() != activation.len() / BLOCK_ELEMENTS * Q8_0_ACT_BLOCK_BYTES
    {
        return Err(shape_error());
    }
    for (block, packed) in activation
        .as_chunks::<BLOCK_ELEMENTS>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<Q8_0_ACT_BLOCK_BYTES>().0.iter_mut())
    {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 was detected at runtime.
            unsafe { quantize_block_avx2(block, packed) };
            continue;
        }
        quantize_activation_block(block, packed);
    }
    Ok(())
}

// ggml's two simd paths disagree on the reciprocal: x86 multiplies by
// `127 / amax` (`arch/x86/quants.c:333`), neon by `1 / (amax / 127)`
// (`arch/arm/quants.c:63-64`); both round half to even.
#[cfg(target_arch = "aarch64")]
fn activation_reciprocal(amax: f32) -> f32 {
    let scale = amax / 127.0;
    if scale == 0.0 { 0.0 } else { 1.0 / scale }
}

#[cfg(not(target_arch = "aarch64"))]
fn activation_reciprocal(amax: f32) -> f32 {
    if amax == 0.0 { 0.0 } else { 127.0 / amax }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_max_f32(lanes: __m256) -> f32 {
    let folded = _mm_max_ps(_mm256_extractf128_ps(lanes, 1), _mm256_castps256_ps128(lanes));
    let folded = _mm_max_ps(folded, _mm_movehl_ps(folded, folded));
    _mm_cvtss_f32(_mm_max_ss(folded, _mm_movehdup_ps(folded)))
}

// same arithmetic as `quantize_activation_block`: |x| max, `x * inverse`, round half to
// even (`cvtps_epi32` under the default rounding mode), narrowed with saturation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_block_avx2(block: &[f32; BLOCK_ELEMENTS], packed: &mut [u8; Q8_0_ACT_BLOCK_BYTES]) {
    // SAFETY: avx2 is enabled; `block` is 32 readable floats and `packed[2..]` 32 writable bytes.
    unsafe {
        let sign_mask = _mm256_set1_ps(-0.0);
        let lanes = [
            _mm256_loadu_ps(block.as_ptr()),
            _mm256_loadu_ps(block.as_ptr().add(8)),
            _mm256_loadu_ps(block.as_ptr().add(16)),
            _mm256_loadu_ps(block.as_ptr().add(24)),
        ];
        let magnitudes = lanes.map(|lane| _mm256_andnot_ps(sign_mask, lane));
        let widest = _mm256_max_ps(
            _mm256_max_ps(magnitudes[0], magnitudes[1]),
            _mm256_max_ps(magnitudes[2], magnitudes[3]),
        );
        let amax = horizontal_max_f32(widest);
        let inverse = _mm256_set1_ps(activation_reciprocal(amax));
        let codes = lanes.map(|lane| _mm256_cvtps_epi32(_mm256_mul_ps(lane, inverse)));
        let narrowed = _mm256_packs_epi16(
            _mm256_packs_epi32(codes[0], codes[1]),
            _mm256_packs_epi32(codes[2], codes[3]),
        );
        let ordered = _mm256_permutevar8x32_epi32(
            narrowed,
            _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7),
        );
        _mm256_storeu_si256(packed[2..].as_mut_ptr().cast(), ordered);
        packed[..2].copy_from_slice(&half::f16::from_f32(amax / 127.0).to_le_bytes());
    }
}


fn quantize_activation_block(
    block: &[f32; BLOCK_ELEMENTS],
    packed: &mut [u8; Q8_0_ACT_BLOCK_BYTES],
) {
    let amax = block.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
    let inverse = activation_reciprocal(amax);
    packed[..2].copy_from_slice(&half::f16::from_f32(amax / 127.0).to_le_bytes());
    for (code, value) in packed[2..].iter_mut().zip(block) {
        *code = ((value * inverse).round_ties_even() as i8).to_le_bytes()[0];
    }
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

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum(widened: __m256i) -> i32 {
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
unsafe fn signed_dot(weights: __m256i, activations: __m256i) -> __m256i {
    let magnitudes = _mm256_sign_epi8(weights, weights);
    let signed_activations = _mm256_sign_epi8(activations, weights);
    _mm256_madd_epi16(
        _mm256_maddubs_epi16(magnitudes, signed_activations),
        _mm256_set1_epi16(1),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unpack_q8_0_codes(weight_block: &[u8]) -> __m256i {
    let weight_codes = &weight_block[2..2 + BLOCK_ELEMENTS];
    // SAFETY: the slice is exactly 32 readable bytes; loadu needs no alignment.
    unsafe { _mm256_loadu_si256(weight_codes.as_ptr().cast()) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn block_lanes_q8_0_avx2(weight_block: &[u8], activation_block: &[u8]) -> __m256i {
    let activation_codes = &activation_block[2..2 + BLOCK_ELEMENTS];
    // SAFETY: avx2 is enabled; the slice is exactly 32 readable bytes.
    unsafe {
        let weights = unpack_q8_0_codes(weight_block);
        let activations = _mm256_loadu_si256(activation_codes.as_ptr().cast());
        signed_dot(weights, activations)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn block_lanes_q5_0_avx2(weight_block: &[u8], activation_block: &[u8]) -> __m256i {
    let activation_codes = &activation_block[2..2 + BLOCK_ELEMENTS];
    // SAFETY: avx2 is enabled; the slice is exactly 32 readable bytes.
    unsafe {
        let weights = unpack_q5_0_codes(weight_block);
        let activations = _mm256_loadu_si256(activation_codes.as_ptr().cast());
        signed_dot(weights, activations)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unpack_q5_0_codes(weight_block: &[u8]) -> __m256i {
    let high_bits = i32::from_le_bytes([
        weight_block[2],
        weight_block[3],
        weight_block[4],
        weight_block[5],
    ]);
    let nibbles = &weight_block[6..6 + BLOCK_ELEMENTS / 2];
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
    // SAFETY: `nibbles` is 16 readable bytes; loadu needs no alignment.
    unsafe {
        let packed = _mm_loadu_si128(nibbles.as_ptr().cast());
        let low_mask = _mm_set1_epi8(0x0f);
        let low = _mm_and_si128(packed, low_mask);
        let high = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);
        let codes = _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
        _mm256_sub_epi8(codes, missing_fifth_bit)
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn block_sum_q8_0_avx2(weight_block: &[u8], activation_block: &[u8]) -> i32 {
    // SAFETY: avx2 is enabled on this function.
    unsafe { horizontal_sum(block_lanes_q8_0_avx2(weight_block, activation_block)) }
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn block_sum_q5_0_avx2(weight_block: &[u8], activation_block: &[u8]) -> i32 {
    // SAFETY: avx2 is enabled on this function.
    unsafe { horizontal_sum(block_lanes_q5_0_avx2(weight_block, activation_block)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_f32(lanes: __m256) -> f32 {
    let high = _mm256_extractf128_ps(lanes, 1);
    let folded = _mm_add_ps(high, _mm256_castps256_ps128(lanes));
    let folded = _mm_add_ps(folded, _mm_movehl_ps(folded, folded));
    _mm_cvtss_f32(_mm_add_ss(folded, _mm_movehdup_ps(folded)))
}

#[cfg(all(test, target_arch = "x86_64"))]
fn avx2_available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn scale_product_f16c(weight_block: &[u8], activation_block: &[u8]) -> f32 {
    let weight_bits = u32::from(u16::from_le_bytes([weight_block[0], weight_block[1]]));
    let activation_bits = u32::from(u16::from_le_bytes([
        activation_block[0],
        activation_block[1],
    ]));
    let both = _mm_cvtph_ps(_mm_cvtsi32_si128(
        (weight_bits | (activation_bits << 16)) as i32,
    ));
    _mm_cvtss_f32(_mm_mul_ss(both, _mm_shuffle_ps(both, both, 0b01)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn row_q8_0_avx2(weight_row: &[u8], activation_q8_0: &[u8]) -> f32 {
    let mut accumulator = _mm256_setzero_ps();
    for (weight_block, activation_block) in weight_row
        .as_chunks::<{ q8_0::BLOCK_BYTES }>()
        .0
        .iter()
        .zip(activation_q8_0.as_chunks::<Q8_0_ACT_BLOCK_BYTES>().0.iter())
    {
        // SAFETY: avx2, f16c and fma are enabled on this function; blocks are exact-size arrays.
        unsafe {
            let lanes = _mm256_cvtepi32_ps(block_lanes_q8_0_avx2(weight_block, activation_block));
            let scale = _mm256_set1_ps(scale_product_f16c(weight_block, activation_block));
            accumulator = _mm256_fmadd_ps(scale, lanes, accumulator);
        }
    }
    // SAFETY: avx2 is enabled on this function.
    unsafe { horizontal_sum_f32(accumulator) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn row_q5_0_avx2(weight_row: &[u8], activation_q8_0: &[u8]) -> f32 {
    let mut accumulator = _mm256_setzero_ps();
    for (weight_block, activation_block) in weight_row
        .as_chunks::<{ q5_0::BLOCK_BYTES }>()
        .0
        .iter()
        .zip(activation_q8_0.as_chunks::<Q8_0_ACT_BLOCK_BYTES>().0.iter())
    {
        // SAFETY: avx2, f16c and fma are enabled on this function; blocks are exact-size arrays.
        unsafe {
            let lanes = _mm256_cvtepi32_ps(block_lanes_q5_0_avx2(weight_block, activation_block));
            let scale = _mm256_set1_ps(scale_product_f16c(weight_block, activation_block));
            accumulator = _mm256_fmadd_ps(scale, lanes, accumulator);
        }
    }
    // SAFETY: avx2 is enabled on this function.
    unsafe { horizontal_sum_f32(accumulator) }
}

#[cfg(target_arch = "x86_64")]
fn row_kernels_available() -> bool {
    std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("f16c")
        && std::is_x86_feature_detected!("fma")
}

fn block_dot_row<const WEIGHT_BLOCK_BYTES: usize>(
    weight_row: &[u8],
    activation_q8_0: &[u8],
    scalar: fn(&[u8], &[u8]) -> i32,
    #[cfg(target_arch = "x86_64")] vectorized_row: unsafe fn(&[u8], &[u8]) -> f32,
) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(WEIGHT_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of its codec's block size",
        });
    }
    let block_count = weight_row.len() / WEIGHT_BLOCK_BYTES;
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
        .as_chunks::<WEIGHT_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation_q8_0.as_chunks::<Q8_0_ACT_BLOCK_BYTES>().0.iter())
    {
        let block_sum = scalar(weight_block, activation_block);
        sum += (f16_scale(weight_block) * f16_scale(activation_block)) * block_sum as f32;
    }
    Ok(sum)
}

#[cfg(target_arch = "x86_64")]
pub(super) fn dot_q8_0_q8_0(weight_row: &[u8], activation_q8_0: &[u8]) -> Result<f32, TensorError> {
    block_dot_row::<{ q8_0::BLOCK_BYTES }>(
        weight_row,
        activation_q8_0,
        block_sum_q8_0_scalar,
        row_q8_0_avx2,
    )
}

#[cfg(not(target_arch = "x86_64"))]
pub(super) fn dot_q8_0_q8_0(weight_row: &[u8], activation_q8_0: &[u8]) -> Result<f32, TensorError> {
    block_dot_row::<{ q8_0::BLOCK_BYTES }>(weight_row, activation_q8_0, block_sum_q8_0_scalar)
}

#[cfg(target_arch = "x86_64")]
pub(super) fn dot_q5_0_q8_0(weight_row: &[u8], activation_q8_0: &[u8]) -> Result<f32, TensorError> {
    block_dot_row::<{ q5_0::BLOCK_BYTES }>(
        weight_row,
        activation_q8_0,
        block_sum_q5_0_scalar,
        row_q5_0_avx2,
    )
}

#[cfg(not(target_arch = "x86_64"))]
pub(super) fn dot_q5_0_q8_0(weight_row: &[u8], activation_q8_0: &[u8]) -> Result<f32, TensorError> {
    block_dot_row::<{ q5_0::BLOCK_BYTES }>(weight_row, activation_q8_0, block_sum_q5_0_scalar)
}

pub(super) type ActivationRowsFn = fn(&[u8], usize, &[u8], &mut [f32]) -> Result<(), TensorError>;



pub(super) type ActivationKernels = (ActivationDotFn, ActivationRowsFn, ActivationUnitsFn);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn rows_q8_0_avx2(weights: &[u8], row_bytes: usize, activation: &[u8], out: &mut [f32]) {
    for (slot, row) in out.iter_mut().zip(weights.chunks_exact(row_bytes)) {
        // SAFETY: avx2, f16c and fma are enabled on this function.
        *slot = unsafe { row_q8_0_avx2(row, activation) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn rows_q5_0_avx2(weights: &[u8], row_bytes: usize, activation: &[u8], out: &mut [f32]) {
    for (slot, row) in out.iter_mut().zip(weights.chunks_exact(row_bytes)) {
        // SAFETY: avx2, f16c and fma are enabled on this function.
        *slot = unsafe { row_q5_0_avx2(row, activation) };
    }
}

pub(super) struct UnitOperands<'a> {
    pub(super) weight_rows: &'a [u8],
    pub(super) row_bytes: usize,
    pub(super) activations: &'a [u8],
    pub(super) activation_row_bytes: usize,
    pub(super) activation_scales: &'a [f32],
    pub(super) positions: usize,
}

impl UnitOperands<'_> {
    fn rows(&self) -> usize {
        self.weight_rows.len() / self.row_bytes
    }

    fn blocks(&self) -> usize {
        self.activation_row_bytes / Q8_0_ACT_BLOCK_BYTES
    }

    fn is_consistent(&self, weight_block_bytes: usize, out_len: usize) -> bool {
        self.row_bytes != 0
            && self.row_bytes.is_multiple_of(weight_block_bytes)
            && self.weight_rows.len().is_multiple_of(self.row_bytes)
            && self.activation_row_bytes == self.row_bytes / weight_block_bytes * Q8_0_ACT_BLOCK_BYTES
            && self.activations.len() == self.positions * self.activation_row_bytes
            && self.activation_scales.len() == self.positions * self.blocks()
            && out_len == self.rows() * self.positions
    }
}

pub(super) type ActivationUnitsFn = fn(&UnitOperands<'_>, &mut [f32]) -> Result<(), TensorError>;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unpack_weight_codes<const FIVE_BIT: bool>(weight_block: &[u8]) -> __m256i {
    // SAFETY: avx2 is enabled on this function.
    unsafe {
        if FIVE_BIT {
            unpack_q5_0_codes(weight_block)
        } else {
            unpack_q8_0_codes(weight_block)
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn f16_bits_to_f32(bytes: &[u8]) -> f32 {
    let bits = i32::from(u16::from_le_bytes([bytes[0], bytes[1]]));
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)))
}

// weight rows unpacked once per unit (not once per activation row): sign-extended codes,
// their magnitudes, and the f16 scales widened to f32, so the tile loop below is loads + dots.
#[cfg(target_arch = "x86_64")]
struct UnpackedRows {
    codes: Vec<u8>,
    magnitudes: Vec<u8>,
    scales: Vec<f32>,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn unpack_rows<const FIVE_BIT: bool>(
    operands: &UnitOperands<'_>,
    rows: usize,
    unpacked: &mut UnpackedRows,
) {
    let weight_block_bytes = if FIVE_BIT { q5_0::BLOCK_BYTES } else { q8_0::BLOCK_BYTES };
    let weight_rows = operands.weight_rows;
    let blocks = operands.blocks();
    for row in 0..rows {
        for block in 0..blocks {
            let start = row * operands.row_bytes + block * weight_block_bytes;
            let weight_block = &weight_rows[start..start + weight_block_bytes];
            let destination = (row * blocks + block) * BLOCK_ELEMENTS;
            // SAFETY: avx2 and f16c are enabled on this function; destinations are 32-byte windows.
            unsafe {
                let codes = unpack_weight_codes::<FIVE_BIT>(weight_block);
                let magnitudes = _mm256_sign_epi8(codes, codes);
                _mm256_storeu_si256(unpacked.codes[destination..destination + BLOCK_ELEMENTS].as_mut_ptr().cast(), codes);
                _mm256_storeu_si256(unpacked.magnitudes[destination..destination + BLOCK_ELEMENTS].as_mut_ptr().cast(), magnitudes);
                unpacked.scales[row * blocks + block] = f16_bits_to_f32(weight_block);
            }
        }
    }
}

// ROWS weight rows x TILE activation rows (ggml's gemm4xN register tile): every output
// keeps `row_*_avx2`'s per-block lane order and single fma chain, so results are
// bit-identical to the row dots.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn tile_avx2<const ROWS: usize, const TILE: usize>(
    operands: &UnitOperands<'_>,
    unpacked: &UnpackedRows,
    first_row: usize,
    position: usize,
    out: &mut [f32],
) {
    let blocks = operands.blocks();
    let mut accumulators = [[_mm256_setzero_ps(); ROWS]; TILE];
    let ones = _mm256_set1_epi16(1);
    for block in 0..blocks {
        for (lane, lane_accumulators) in accumulators.iter_mut().enumerate() {
            let tile_position = position + lane;
            let codes_start = tile_position * operands.activation_row_bytes
                + block * Q8_0_ACT_BLOCK_BYTES
                + 2;
            let activation_codes = &operands.activations[codes_start..codes_start + BLOCK_ELEMENTS];
            // SAFETY: avx2 and fma are enabled on this function; every pointer below addresses a
            // whole 32-byte block inside buffers sized by `UnitOperands::is_consistent` and `unpack_rows`.
            unsafe {
                let activations = _mm256_loadu_si256(activation_codes.as_ptr().cast());
                let activation_scale = _mm256_set1_ps(
                    *operands.activation_scales.get_unchecked(tile_position * blocks + block),
                );
                for (row, accumulator) in lane_accumulators.iter_mut().enumerate() {
                    let row_block = (first_row + row) * blocks + block;
                    let offset = row_block * BLOCK_ELEMENTS;
                    let codes = _mm256_loadu_si256(unpacked.codes.as_ptr().add(offset).cast());
                    let magnitudes = _mm256_loadu_si256(unpacked.magnitudes.as_ptr().add(offset).cast());
                    let signed = _mm256_sign_epi8(activations, codes);
                    let lanes = _mm256_cvtepi32_ps(_mm256_madd_epi16(
                        _mm256_maddubs_epi16(magnitudes, signed),
                        ones,
                    ));
                    let weight_scale = _mm256_set1_ps(*unpacked.scales.get_unchecked(row_block));
                    let scale = _mm256_mul_ps(weight_scale, activation_scale);
                    *accumulator = _mm256_fmadd_ps(scale, lanes, *accumulator);
                }
            }
        }
    }
    for (lane, lane_accumulators) in accumulators.into_iter().enumerate() {
        for (row, accumulator) in lane_accumulators.into_iter().enumerate() {
            // SAFETY: avx2 is enabled on this function.
            out[(first_row + row) * operands.positions + position + lane] =
                unsafe { horizontal_sum_f32(accumulator) };
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn group_avx2<const ROWS: usize>(
    operands: &UnitOperands<'_>,
    unpacked: &UnpackedRows,
    first_row: usize,
    positions: core::ops::Range<usize>,
    out: &mut [f32],
) {
    let mut position = positions.start;
    while position + TILE_POSITIONS <= positions.end {
        // SAFETY: avx2, f16c and fma are enabled on this function.
        unsafe { tile_avx2::<ROWS, TILE_POSITIONS>(operands, unpacked, first_row, position, out) };
        position += TILE_POSITIONS;
    }
    while position < positions.end {
        // SAFETY: avx2, f16c and fma are enabled on this function.
        unsafe { tile_avx2::<ROWS, 1>(operands, unpacked, first_row, position, out) };
        position += 1;
    }
}

// cache blocking: a position block's activation rows stay in L2 while every row group of
// the unit sweeps them, so each activation byte crosses L3 once per unit, not once per group
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
unsafe fn units_avx2<const FIVE_BIT: bool>(operands: &UnitOperands<'_>, out: &mut [f32]) {
    let rows = operands.rows();
    let blocks = operands.blocks();
    let mut unpacked = UnpackedRows {
        codes: vec![0; rows * blocks * BLOCK_ELEMENTS],
        magnitudes: vec![0; rows * blocks * BLOCK_ELEMENTS],
        scales: vec![0.0; rows * blocks],
    };
    // SAFETY: avx2, f16c and fma are enabled on this function.
    unsafe { unpack_rows::<FIVE_BIT>(operands, rows, &mut unpacked) };
    let grouped = rows / TILE_ROWS * TILE_ROWS;
    for block_start in (0..operands.positions).step_by(POSITION_BLOCK) {
        let block_range = block_start..(block_start + POSITION_BLOCK).min(operands.positions);
        for first_row in (0..grouped).step_by(TILE_ROWS) {
            // SAFETY: avx2, f16c and fma are enabled on this function.
            unsafe { group_avx2::<TILE_ROWS>(operands, &unpacked, first_row, block_range.clone(), out) };
        }
        for first_row in grouped..rows {
            // SAFETY: avx2, f16c and fma are enabled on this function.
            unsafe { group_avx2::<1>(operands, &unpacked, first_row, block_range.clone(), out) };
        }
    }
}

#[cfg(target_arch = "x86_64")]
const TILE_POSITIONS: usize = 2;
#[cfg(target_arch = "x86_64")]
const TILE_ROWS: usize = 4;
#[cfg(target_arch = "x86_64")]
const POSITION_BLOCK: usize = 64;
pub(super) const UNIT_ROWS: usize = 16;

fn units_of<const WEIGHT_BLOCK_BYTES: usize>(
    operands: &UnitOperands<'_>,
    out: &mut [f32],
    dot_row: ActivationDotFn,
    #[cfg(target_arch = "x86_64")] vectorized: unsafe fn(&UnitOperands<'_>, &mut [f32]),
) -> Result<(), TensorError> {
    if !operands.is_consistent(WEIGHT_BLOCK_BYTES, out.len()) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight rows or q8_0 activation do not match the codec's block layout",
        });
    }
    #[cfg(target_arch = "x86_64")]
    if row_kernels_available() {
        // SAFETY: avx2, f16c and fma were detected at runtime; shapes were checked above.
        unsafe { vectorized(operands, out) };
        return Ok(());
    }
    for (row, weight_row) in operands.weight_rows.chunks_exact(operands.row_bytes).enumerate() {
        for position in 0..operands.positions {
            let start = position * operands.activation_row_bytes;
            out[row * operands.positions + position] = dot_row(
                weight_row,
                &operands.activations[start..start + operands.activation_row_bytes],
            )?;
        }
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn units_q8_0_q8_0(operands: &UnitOperands<'_>, out: &mut [f32]) -> Result<(), TensorError> {
    units_of::<{ q8_0::BLOCK_BYTES }>(operands, out, dot_q8_0_q8_0, units_avx2::<false>)
}

#[cfg(not(target_arch = "x86_64"))]
fn units_q8_0_q8_0(operands: &UnitOperands<'_>, out: &mut [f32]) -> Result<(), TensorError> {
    units_of::<{ q8_0::BLOCK_BYTES }>(operands, out, dot_q8_0_q8_0)
}

#[cfg(target_arch = "x86_64")]
fn units_q5_0_q8_0(operands: &UnitOperands<'_>, out: &mut [f32]) -> Result<(), TensorError> {
    units_of::<{ q5_0::BLOCK_BYTES }>(operands, out, dot_q5_0_q8_0, units_avx2::<true>)
}

#[cfg(not(target_arch = "x86_64"))]
fn units_q5_0_q8_0(operands: &UnitOperands<'_>, out: &mut [f32]) -> Result<(), TensorError> {
    units_of::<{ q5_0::BLOCK_BYTES }>(operands, out, dot_q5_0_q8_0)
}

fn block_dot_rows<const WEIGHT_BLOCK_BYTES: usize>(
    weights: &[u8],
    row_bytes: usize,
    activation_q8_0: &[u8],
    out: &mut [f32],
    dot_row: ActivationDotFn,
    #[cfg(target_arch = "x86_64")] vectorized_rows: unsafe fn(&[u8], usize, &[u8], &mut [f32]),
) -> Result<(), TensorError> {
    let block_count = row_bytes / WEIGHT_BLOCK_BYTES;
    if row_bytes == 0
        || !row_bytes.is_multiple_of(WEIGHT_BLOCK_BYTES)
        || out.len().checked_mul(row_bytes) != Some(weights.len())
        || activation_q8_0.len() != block_count * Q8_0_ACT_BLOCK_BYTES
    {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight rows or q8_0 activation do not match the codec's block layout",
        });
    }
    #[cfg(target_arch = "x86_64")]
    if row_kernels_available() {
        // SAFETY: avx2, f16c and fma were detected at runtime; shapes were checked above.
        unsafe { vectorized_rows(weights, row_bytes, activation_q8_0, out) };
        return Ok(());
    }
    for (slot, row) in out.iter_mut().zip(weights.chunks_exact(row_bytes)) {
        *slot = dot_row(row, activation_q8_0)?;
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn rows_q8_0_q8_0(
    weights: &[u8],
    row_bytes: usize,
    activation_q8_0: &[u8],
    out: &mut [f32],
) -> Result<(), TensorError> {
    block_dot_rows::<{ q8_0::BLOCK_BYTES }>(
        weights,
        row_bytes,
        activation_q8_0,
        out,
        dot_q8_0_q8_0,
        rows_q8_0_avx2,
    )
}

#[cfg(not(target_arch = "x86_64"))]
fn rows_q8_0_q8_0(
    weights: &[u8],
    row_bytes: usize,
    activation_q8_0: &[u8],
    out: &mut [f32],
) -> Result<(), TensorError> {
    block_dot_rows::<{ q8_0::BLOCK_BYTES }>(weights, row_bytes, activation_q8_0, out, dot_q8_0_q8_0)
}

#[cfg(target_arch = "x86_64")]
fn rows_q5_0_q8_0(
    weights: &[u8],
    row_bytes: usize,
    activation_q8_0: &[u8],
    out: &mut [f32],
) -> Result<(), TensorError> {
    block_dot_rows::<{ q5_0::BLOCK_BYTES }>(
        weights,
        row_bytes,
        activation_q8_0,
        out,
        dot_q5_0_q8_0,
        rows_q5_0_avx2,
    )
}

#[cfg(not(target_arch = "x86_64"))]
fn rows_q5_0_q8_0(
    weights: &[u8],
    row_bytes: usize,
    activation_q8_0: &[u8],
    out: &mut [f32],
) -> Result<(), TensorError> {
    block_dot_rows::<{ q5_0::BLOCK_BYTES }>(weights, row_bytes, activation_q8_0, out, dot_q5_0_q8_0)
}
pub(super) fn activation_dot_for(codec: Codec) -> ActivationKernels {
    match codec {
        Codec::Q5_0 => (dot_q5_0_q8_0, rows_q5_0_q8_0, units_q5_0_q8_0),
        _ => (dot_q8_0_q8_0, rows_q8_0_q8_0, units_q8_0_q8_0),
    }
}

pub fn matmul_q8_0_activation_f32(
    codec: Codec,
    weights: &[u8],
    rows: usize,
    activation: &[f32],
    leading_total: usize,
) -> Result<Vec<f32>, TensorError> {
    matmul_q8_0_activation_impl(
        weights,
        rows,
        activation,
        leading_total,
        None,
        activation_dot_for(codec),
    )
}

pub(super) fn matmul_q8_0_activation_impl(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
    leading_total: usize,
    session: Option<&MatmulSession<'_>>,
    (dot, rows_kernel, units_kernel): ActivationKernels,
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
    if leading_total > 1 {
        let activation_scales: Vec<f32> = quantized
            .as_chunks::<Q8_0_ACT_BLOCK_BYTES>()
            .0
            .iter()
            .map(|block| f16_scale(block))
            .collect();
        let run_group = |first_row: usize, group_rows: usize, out: &mut [f32]| {
            let operands = UnitOperands {
                weight_rows: &weights[first_row * row_bytes..(first_row + group_rows) * row_bytes],
                row_bytes,
                activations: &quantized,
                activation_row_bytes: quantized_row_bytes,
                activation_scales: &activation_scales,
                positions: leading_total,
            };
            units_kernel(&operands, &mut out[..group_rows * leading_total])
        };
        return match quantized_matmul_workers(rows, activation.len()) {
            Some(workers) => {
                let units = rows.div_ceil(UNIT_ROWS);
                let mut output = matmul_rows_threaded(
                    units,
                    UNIT_ROWS * leading_total,
                    workers,
                    session,
                    k,
                    |unit, slot| {
                        let first_row = unit * UNIT_ROWS;
                        run_group(first_row, UNIT_ROWS.min(rows - first_row), slot)
                    },
                )?;
                output.truncate(rows * leading_total);
                Ok(output)
            }
            None => {
                let mut output = vec![0.0f32; rows * leading_total];
                for first_row in (0..rows).step_by(UNIT_ROWS) {
                    let group_rows = UNIT_ROWS.min(rows - first_row);
                    run_group(
                        first_row,
                        group_rows,
                        &mut output[first_row * leading_total..],
                    )?;
                }
                Ok(output)
            }
        };
    }
    match quantized_matmul_workers(rows, activation.len()) {
        Some(workers) => matmul_rows_threaded(rows, 1, workers, session, k, |row, slot| {
            let weight_row = &weights[row * row_bytes..(row + 1) * row_bytes];
            slot[0] = dot(weight_row, &quantized)?;
            Ok(())
        }),
        None => {
            let mut output = vec![0.0f32; rows];
            rows_kernel(weights, row_bytes, &quantized, &mut output)?;
            Ok(output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseError;

    fn writer_reference_quantize(x: &[f32]) -> (f32, Vec<i8>) {
        let amax = x.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        let d = amax / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        let levels = x
            .iter()
            .map(|&value| libm::roundf(value * id) as i8)
            .collect();
        (half::f16::from_f32(d).to_f32(), levels)
    }

    #[cfg(target_arch = "aarch64")]
    fn ggml_simd_reciprocal(amax: f32) -> f32 {
        let d = amax / 127.0;
        if d == 0.0 { 0.0 } else { 1.0 / d }
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn ggml_simd_reciprocal(amax: f32) -> f32 {
        if amax == 0.0 { 0.0 } else { 127.0 / amax }
    }

    fn reference_quantize(x: &[f32]) -> (f32, Vec<i8>) {
        let amax = x.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        let id = ggml_simd_reciprocal(amax);
        let levels = x
            .iter()
            .map(|&value| (value * id).round_ties_even() as i8)
            .collect();
        (half::f16::from_f32(amax / 127.0).to_f32(), levels)
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

    #[test]
    fn activation_ties_round_to_even_where_the_gguf_writer_rounds_away() {
        let mut block = [0.0f32; 32];
        block[0] = 127.0;
        block[1] = 2.5;
        block[2] = -3.5;
        let mut activation = vec![0u8; Q8_0_ACT_BLOCK_BYTES];
        let mut written = vec![0u8; q8_0::BLOCK_BYTES];
        quantize_row_q8_0_activation(&block, &mut activation).expect("activation quantizes");
        q8_0::quantize(&block, &mut written).expect("writer quantizes");
        assert_eq!(
            (activation[3].cast_signed(), activation[4].cast_signed()),
            (2, -4),
            "ggml's cpu quantizer rounds ties to even"
        );
        assert_eq!(
            (written[3].cast_signed(), written[4].cast_signed()),
            (3, -4),
            "quantize_row_q8_0_ref rounds ties away from zero"
        );
    }

    proptest! {
        #[test]
        fn activation_quantize_matches_ggml_simd_quantize_row_q8_0(x in activation_strategy()) {
            let mut ours = vec![0u8; Q8_0_ACT_BLOCK_BYTES];
            prop_assert!(quantize_row_q8_0_activation(&x, &mut ours).is_ok());
            let (reference_d, reference_levels) = reference_quantize(&x);
            prop_assert_eq!(f16_scale(&ours).to_bits(), reference_d.to_bits());
            let ours_levels: Vec<i8> = ours[2..].iter().map(|&byte| byte.cast_signed()).collect();
            prop_assert_eq!(ours_levels, reference_levels);
        }

        #[test]
        fn gguf_writer_quantize_stays_quantize_row_q8_0_ref(x in activation_strategy()) {
            let mut written = vec![0u8; q8_0::BLOCK_BYTES];
            prop_assert!(q8_0::quantize(&x, &mut written).is_ok());
            let (reference_d, reference_levels) = writer_reference_quantize(&x);
            prop_assert_eq!(f16_scale(&written).to_bits(), reference_d.to_bits());
            let written_levels: Vec<i8> = written[2..].iter().map(|&byte| byte.cast_signed()).collect();
            prop_assert_eq!(written_levels, reference_levels);
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
            let abs_sum: i32 = block[2..].iter().zip(&levels).map(|(&w, &a)| (i32::from(w.cast_signed()) * i32::from(a)).abs()).sum();
            let magnitude = f64::from(abs_sum) * f64::from(f16_scale(&block) * activation_d).abs();
            prop_assert!((f64::from(ours) - expected).abs() <= 2e-6 * magnitude.max(1e-6));
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
            let abs_sum: i32 = codes.iter().zip(&levels).map(|(&w, &a)| (w * i32::from(a)).abs()).sum();
            let magnitude = f64::from(abs_sum) * f64::from(f16_scale(&block) * activation_d).abs();
            prop_assert!((f64::from(ours) - expected).abs() <= 2e-6 * magnitude.max(1e-6));
            #[cfg(target_arch = "x86_64")]
            if avx2_available() {
                // SAFETY: avx2 detected above.
                prop_assert_eq!(unsafe { block_sum_q5_0_avx2(&block, &packed) }, expected_sum);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn ggml_avx2_row_reference(blocks: &[(f32, [i32; 32])], activation: &[u8]) -> f32 {
        let mut accumulator = [0.0f32; 8];
        for (index, (weight_scale, codes)) in blocks.iter().enumerate() {
            let block = &activation[index * Q8_0_ACT_BLOCK_BYTES..];
            let activation_scale = f16_scale(block);
            let scale = weight_scale * activation_scale;
            for lane in 0..8 {
                let lane_sum: i32 = (0..4)
                    .map(|offset| {
                        codes[lane * 4 + offset]
                            * i32::from(block[2 + lane * 4 + offset].cast_signed())
                    })
                    .sum();
                accumulator[lane] = scale.mul_add(lane_sum as f32, accumulator[lane]);
            }
        }
        let folded: Vec<f32> = (0..4)
            .map(|lane| accumulator[lane + 4] + accumulator[lane])
            .collect();
        let pairs = [folded[0] + folded[2], folded[1] + folded[3]];
        pairs[0] + pairs[1]
    }

    fn lcg_bytes(seed: u64, length: usize) -> Vec<u8> {
        let mut state = seed;
        (0..length)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                (state >> 33) as u8
            })
            .collect()
    }

    fn lcg_activation(seed: u64, length: usize) -> Vec<f32> {
        lcg_bytes(seed, length)
            .into_iter()
            .map(|byte| (f32::from(byte) - 127.5) / 40.0)
            .collect()
    }

    fn weight_rows_of(codec: Codec, rows: usize, blocks: usize, seed: u64) -> Vec<u8> {
        let block_bytes = if codec == Codec::Q5_0 { q5_0::BLOCK_BYTES } else { q8_0::BLOCK_BYTES };
        lcg_bytes(seed, rows * blocks * block_bytes)
            .chunks(block_bytes)
            .flat_map(|block| finite_scale(block.to_vec()))
            .collect()
    }

    #[test]
    fn multi_row_tile_is_bit_identical_to_the_per_position_row_dots() {
        let shapes = [
            (1usize, 2usize),
            (3, 5),
            (4, 2),
            (5, 3),
            (16, 64),
            (17, 66),
            (9, 129),
            (64, 130),
        ];
        for codec in [Codec::Q5_0, Codec::Q8_0] {
            for (rows, positions) in shapes {
                let blocks = 7;
                let k = blocks * BLOCK_ELEMENTS;
                let weights = weight_rows_of(codec, rows, blocks, 11 + rows as u64);
                let activation = lcg_activation(29 + positions as u64, positions * k);
                let (dot, _, _) = activation_dot_for(codec);
                let wide = matmul_q8_0_activation_impl(&weights, rows, &activation, positions, None, activation_dot_for(codec))
                    .expect("multi-row tile runs");
                let mut quantized = vec![0u8; positions * blocks * Q8_0_ACT_BLOCK_BYTES];
                quantize_row_q8_0_activation(&activation, &mut quantized).expect("activation quantizes");
                let row_bytes = weights.len() / rows;
                let activation_row_bytes = blocks * Q8_0_ACT_BLOCK_BYTES;
                for row in 0..rows {
                    for position in 0..positions {
                        let expected = dot(
                            &weights[row * row_bytes..(row + 1) * row_bytes],
                            &quantized[position * activation_row_bytes..(position + 1) * activation_row_bytes],
                        )
                        .expect("row dot runs");
                        assert_eq!(
                            wide[row * positions + position].to_bits(),
                            expected.to_bits(),
                            "{codec:?} rows={rows} positions={positions} row={row} position={position}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn multi_row_tile_rejects_an_activation_that_is_not_a_whole_block_multiple() {
        let weights = weight_rows_of(Codec::Q5_0, 4, 3, 7);
        let activation = lcg_activation(3, 2 * (3 * BLOCK_ELEMENTS + 1));
        assert!(
            matmul_q8_0_activation_impl(&weights, 4, &activation, 2, None, activation_dot_for(Codec::Q5_0)).is_err()
        );
    }

    #[test]
    fn avx2_activation_quantizer_matches_the_scalar_reference_on_real_range_activations() {
        let activation = lcg_activation(41, 16 * BLOCK_ELEMENTS);
        let mut vectorized = vec![0u8; 16 * Q8_0_ACT_BLOCK_BYTES];
        quantize_row_q8_0_activation(&activation, &mut vectorized).expect("activation quantizes");
        for (block, packed) in activation.as_chunks::<BLOCK_ELEMENTS>().0.iter().zip(vectorized.as_chunks::<Q8_0_ACT_BLOCK_BYTES>().0) {
            let mut scalar = [0u8; Q8_0_ACT_BLOCK_BYTES];
            quantize_activation_block(block, &mut scalar);
            assert_eq!(&scalar, packed);
        }
    }

    #[cfg(target_arch = "x86_64")]
    proptest! {
        #[test]
        fn rows_equal_the_ggml_avx2_lane_order_bit_for_bit(
            raw in prop::collection::vec(any::<u8>(), 8 * q5_0::BLOCK_BYTES),
            raw_q8 in prop::collection::vec(any::<u8>(), 8 * q8_0::BLOCK_BYTES),
            x in prop::collection::vec(-4.0f32..4.0f32, 8 * 32),
            block_count in 1usize..=8,
        ) {
            prop_assume!(
                std::is_x86_feature_detected!("avx2")
                    && std::is_x86_feature_detected!("f16c")
                    && std::is_x86_feature_detected!("fma")
            );
            let mut activation = vec![0u8; block_count * Q8_0_ACT_BLOCK_BYTES];
            prop_assert!(quantize_row_q8_0_activation(&x[..block_count * 32], &mut activation).is_ok());

            let q5_row: Vec<u8> = raw
                .chunks(q5_0::BLOCK_BYTES)
                .take(block_count)
                .flat_map(|block| finite_scale(block.to_vec()))
                .collect();
            let q5_blocks: Vec<(f32, [i32; 32])> = q5_row
                .chunks(q5_0::BLOCK_BYTES)
                .map(|block| {
                    let codes = reference_q5_0_codes(block);
                    let mut array = [0i32; 32];
                    array.copy_from_slice(&codes);
                    (f16_scale(block), array)
                })
                .collect();
            let q5_expected = ggml_avx2_row_reference(&q5_blocks, &activation);
            let q5_ours = dot_q5_0_q8_0(&q5_row, &activation)
                .map_err(|error| TestCaseError::fail(format!("{error:?}")))?;
            prop_assert_eq!(q5_ours.to_bits(), q5_expected.to_bits());

            let q8_row: Vec<u8> = raw_q8
                .chunks(q8_0::BLOCK_BYTES)
                .take(block_count)
                .flat_map(|block| finite_scale(block.to_vec()))
                .collect();
            let q8_blocks: Vec<(f32, [i32; 32])> = q8_row
                .chunks(q8_0::BLOCK_BYTES)
                .map(|block| {
                    let mut array = [0i32; 32];
                    for (slot, &byte) in array.iter_mut().zip(&block[2..]) {
                        *slot = i32::from(byte.cast_signed());
                    }
                    (f16_scale(block), array)
                })
                .collect();
            let q8_expected = ggml_avx2_row_reference(&q8_blocks, &activation);
            let q8_ours = dot_q8_0_q8_0(&q8_row, &activation)
                .map_err(|error| TestCaseError::fail(format!("{error:?}")))?;
            prop_assert_eq!(q8_ours.to_bits(), q8_expected.to_bits());
        }
    }

    proptest! {
        #[test]
        fn batched_rows_equal_the_per_row_dots_bit_for_bit(
            raw in prop::collection::vec(any::<u8>(), 6 * 5 * q5_0::BLOCK_BYTES),
            raw_q8 in prop::collection::vec(any::<u8>(), 6 * 5 * q8_0::BLOCK_BYTES),
            x in prop::collection::vec(-4.0f32..4.0f32, 5 * 32),
        ) {
            let mut activation = vec![0u8; 5 * Q8_0_ACT_BLOCK_BYTES];
            prop_assert!(quantize_row_q8_0_activation(&x, &mut activation).is_ok());

            let q5_weights: Vec<u8> = raw
                .chunks(q5_0::BLOCK_BYTES)
                .flat_map(|block| finite_scale(block.to_vec()))
                .collect();
            let q5_row_bytes = 5 * q5_0::BLOCK_BYTES;
            let mut q5_batched = vec![0.0f32; 6];
            prop_assert!(rows_q5_0_q8_0(&q5_weights, q5_row_bytes, &activation, &mut q5_batched).is_ok());
            for (row, batched) in q5_weights.chunks(q5_row_bytes).zip(&q5_batched) {
                let single = dot_q5_0_q8_0(row, &activation)
                    .map_err(|error| TestCaseError::fail(format!("{error:?}")))?;
                prop_assert_eq!(single.to_bits(), batched.to_bits());
            }

            let q8_weights: Vec<u8> = raw_q8
                .chunks(q8_0::BLOCK_BYTES)
                .flat_map(|block| finite_scale(block.to_vec()))
                .collect();
            let q8_row_bytes = 5 * q8_0::BLOCK_BYTES;
            let mut q8_batched = vec![0.0f32; 6];
            prop_assert!(rows_q8_0_q8_0(&q8_weights, q8_row_bytes, &activation, &mut q8_batched).is_ok());
            for (row, batched) in q8_weights.chunks(q8_row_bytes).zip(&q8_batched) {
                let single = dot_q8_0_q8_0(row, &activation)
                    .map_err(|error| TestCaseError::fail(format!("{error:?}")))?;
                prop_assert_eq!(single.to_bits(), batched.to_bits());
            }
        }

        #[test]
        fn batched_rows_reject_shapes_that_do_not_match_the_block_layout(rows in 1usize..4) {
            let weights = vec![0u8; rows * 3 * q5_0::BLOCK_BYTES];
            let activation = vec![0u8; 2 * Q8_0_ACT_BLOCK_BYTES];
            let mut out = vec![0.0f32; rows];
            prop_assert!(rows_q5_0_q8_0(&weights, 3 * q5_0::BLOCK_BYTES, &activation, &mut out).is_err());
        }
    }
}
