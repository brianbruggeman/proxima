//! llama.cpp's default CPU attention, `ggml_compute_forward_flash_attn_ext_f16_one_chunk`
//! (`ggml/src/ggml-cpu/ops.cpp:8614-8850`), for one bound cached-attention op.
//!
//! Every line that mirrors the incumbent cites the line it mirrors. Q, K and V are f16, the score is
//! `ggml_vec_dot_f16`, the softmax is online with `expf`, the value accumulator lives in f16 and is
//! rounded after every scale and every multiply-add, and the running sum is f32. The vector kernels
//! are the AVX2 layouts ggml builds with: `GGML_F16_STEP` 32, `GGML_F16_EPR` 8, four accumulators
//! (`simd-mappings.h:635-683`), loads `_mm256_cvtph_ps`, stores `_mm256_cvtps_ph(.., 0)`
//! (round to nearest even). The scalar twin computes the same values operation for operation.

use half::f16;

use super::{AttentionExtents, CachedAttentionRotary, CachedAttentionScore};

const STEP: usize = 32;
const LANES: usize = 8;
const ARRAYS: usize = 4;

fn half_bits(value: f32) -> u16 {
    f16::from_f32(value).to_bits()
}

fn half_value(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

/// Which instruction set the half-precision kernels run on, decided once per call.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum HalfKernels {
    Scalar,
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    Avx2,
}

impl HalfKernels {
    pub(super) fn detect() -> Self {
        #[cfg(all(feature = "std", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
        {
            return Self::Avx2;
        }
        Self::Scalar
    }

    /// f32 to f16 bits, `ggml_fp32_to_fp16_row` / `_mm256_cvtps_ph(.., 0)`.
    fn convert(self, source: &[f32], destination: &mut [u16]) {
        match self {
            #[cfg(all(feature = "std", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { x86::convert(source, destination) },
            Self::Scalar => scalar::convert(source, destination),
        }
    }

    /// `ggml_vec_dot_f16` (`vec.cpp:345-367`).
    fn dot(self, key: &[u16], query: &[u16]) -> f32 {
        match self {
            #[cfg(all(feature = "std", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { x86::dot(key, query) },
            Self::Scalar => scalar::dot(key, query),
        }
    }

    /// `ggml_vec_scale_f16`.
    fn scale(self, accumulator: &mut [u16], factor: f32) {
        match self {
            #[cfg(all(feature = "std", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { x86::scale(accumulator, factor) },
            Self::Scalar => scalar::scale(accumulator, factor),
        }
    }

    /// `ggml_vec_mad_f16`.
    fn mad(self, accumulator: &mut [u16], row: &[u16], factor: f32) {
        match self {
            #[cfg(all(feature = "std", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { x86::mad(accumulator, row, factor) },
            Self::Scalar => scalar::mad(accumulator, row, factor),
        }
    }
}

pub(super) mod scalar {
    use super::{ARRAYS, LANES, STEP, half_bits, half_value};

    pub(in crate::physical) fn convert(source: &[f32], destination: &mut [u16]) {
        for (slot, value) in destination.iter_mut().zip(source) {
            *slot = half_bits(*value);
        }
    }

    /// Four eight-lane fma accumulators stepping 32, the `GGML_F32x8_REDUCE` tree
    /// (`simd-mappings.h:602-620`), then the leftovers in double (`vec.cpp:365-367`).
    pub(in crate::physical) fn dot(key: &[u16], query: &[u16]) -> f32 {
        let main = key.len() & !(STEP - 1);
        let mut sum = [[0f32; LANES]; ARRAYS];
        for step in (0..main).step_by(STEP) {
            for (array, lanes) in sum.iter_mut().enumerate() {
                for (lane, accumulator) in lanes.iter_mut().enumerate() {
                    let index = step + array * LANES + lane;
                    *accumulator = libm::fmaf(half_value(key[index]), half_value(query[index]), *accumulator);
                }
            }
        }
        let lanes: [f32; LANES] =
            core::array::from_fn(|lane| (sum[0][lane] + sum[2][lane]) + (sum[1][lane] + sum[3][lane]));
        let folded: [f32; 4] = core::array::from_fn(|lane| lanes[lane] + lanes[lane + 4]);
        let mut total = f64::from((folded[0] + folded[1]) + (folded[2] + folded[3]));
        for index in main..key.len() {
            total += f64::from(half_value(key[index]) * half_value(query[index]));
        }
        total as f32
    }

    pub(in crate::physical) fn scale(accumulator: &mut [u16], factor: f32) {
        for value in accumulator {
            *value = half_bits(half_value(*value) * factor);
        }
    }

    /// The first `n & ~31` lanes fuse, the leftover lanes multiply then add (`vec.h` mad_f16).
    pub(in crate::physical) fn mad(accumulator: &mut [u16], row: &[u16], factor: f32) {
        let main = accumulator.len() & !(STEP - 1);
        for (index, value) in accumulator.iter_mut().enumerate() {
            let updated = if index < main {
                libm::fmaf(half_value(row[index]), factor, half_value(*value))
            } else {
                half_value(*value) + half_value(row[index]) * factor
            };
            *value = half_bits(updated);
        }
    }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
mod x86 {
    use core::arch::x86_64::{
        __m128i, __m256, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps, _mm_loadu_si128, _mm_storeu_si128,
        _mm256_add_ps, _mm256_castps256_ps128, _mm256_cvtph_ps, _mm256_cvtps_ph, _mm256_extractf128_ps,
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_setzero_ps,
    };

    use super::{ARRAYS, LANES, STEP, half_bits, half_value};

    const NEAREST_EVEN: i32 = 0;

    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn load_half(pointer: *const u16) -> __m256 {
        unsafe { _mm256_cvtph_ps(_mm_loadu_si128(pointer.cast::<__m128i>())) }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn store_half(pointer: *mut u16, value: __m256) {
        unsafe { _mm_storeu_si128(pointer.cast::<__m128i>(), _mm256_cvtps_ph::<NEAREST_EVEN>(value)) }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn convert(source: &[f32], destination: &mut [u16]) {
        let main = source.len() & !(LANES - 1);
        for index in (0..main).step_by(LANES) {
            unsafe {
                store_half(destination.as_mut_ptr().add(index), _mm256_loadu_ps(source.as_ptr().add(index)));
            }
        }
        for index in main..source.len() {
            destination[index] = half_bits(source[index]);
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn dot(key: &[u16], query: &[u16]) -> f32 {
        let main = key.len() & !(STEP - 1);
        let mut sum = [_mm256_setzero_ps(); ARRAYS];
        for step in (0..main).step_by(STEP) {
            for (array, accumulator) in sum.iter_mut().enumerate() {
                let offset = step + array * LANES;
                unsafe {
                    let left = load_half(key.as_ptr().add(offset));
                    let right = load_half(query.as_ptr().add(offset));
                    *accumulator = _mm256_fmadd_ps(left, right, *accumulator);
                }
            }
        }
        sum[0] = _mm256_add_ps(sum[0], sum[2]);
        sum[1] = _mm256_add_ps(sum[1], sum[3]);
        sum[0] = _mm256_add_ps(sum[0], sum[1]);
        let folded = _mm_add_ps(_mm256_castps256_ps128(sum[0]), _mm256_extractf128_ps::<1>(sum[0]));
        let pairs = _mm_hadd_ps(folded, folded);
        let mut total = f64::from(_mm_cvtss_f32(_mm_hadd_ps(pairs, pairs)));
        for index in main..key.len() {
            total += f64::from(half_value(key[index]) * half_value(query[index]));
        }
        total as f32
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn scale(accumulator: &mut [u16], factor: f32) {
        let main = accumulator.len() & !(LANES - 1);
        let factor_lanes = _mm256_set1_ps(factor);
        for index in (0..main).step_by(LANES) {
            unsafe {
                let pointer = accumulator.as_mut_ptr().add(index);
                store_half(pointer, _mm256_mul_ps(load_half(pointer), factor_lanes));
            }
        }
        for value in &mut accumulator[main..] {
            *value = half_bits(half_value(*value) * factor);
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn mad(accumulator: &mut [u16], row: &[u16], factor: f32) {
        let fused = accumulator.len() & !(STEP - 1);
        let factor_lanes = _mm256_set1_ps(factor);
        for index in (0..fused).step_by(LANES) {
            unsafe {
                let pointer = accumulator.as_mut_ptr().add(index);
                let updated = _mm256_fmadd_ps(load_half(row.as_ptr().add(index)), factor_lanes, load_half(pointer));
                store_half(pointer, updated);
            }
        }
        for index in fused..accumulator.len() {
            accumulator[index] = half_bits(half_value(accumulator[index]) + half_value(row[index]) * factor);
        }
    }
}

/// Per-call working set, kept per thread so a steady-state call allocates nothing.
#[derive(Default)]
struct Scratch {
    keys: alloc::vec::Vec<u16>,
    values: alloc::vec::Vec<u16>,
    query: alloc::vec::Vec<u16>,
    accumulator: alloc::vec::Vec<u16>,
    staging: alloc::vec::Vec<f32>,
}

#[cfg(feature = "std")]
std::thread_local! {
    static SCRATCH: core::cell::RefCell<Scratch> = core::cell::RefCell::new(Scratch::default());
}

struct Range<'buffer> {
    keys: [&'buffer [f32]; 2],
    pass_keys: Option<&'buffer [f32]>,
    values: &'buffer [f32],
    rows: usize,
    is_cached: bool,
}

fn resized<T: Clone + Default>(buffer: &mut alloc::vec::Vec<T>, length: usize) {
    buffer.clear();
    buffer.resize(length, T::default());
}

/// [`super::stream_cached_attention_split_gqa`]'s contract and layouts, computed the way llama.cpp's
/// default CPU attention computes them.
#[must_use]
pub fn stream_cached_attention_llama_flash(
    queries: [&[f32]; 2],
    keys: [[&[f32]; 2]; 2],
    values: [&[f32]; 2],
    output: &mut [f32],
    extents: &AttentionExtents,
    rotary: &CachedAttentionRotary<'_>,
    score: &CachedAttentionScore,
) -> bool {
    #[cfg(feature = "std")]
    {
        SCRATCH.with(|cell| match cell.try_borrow_mut() {
            Ok(mut scratch) => run(&mut scratch, queries, keys, values, output, extents, rotary, score),
            Err(_) => run(&mut Scratch::default(), queries, keys, values, output, extents, rotary, score),
        })
    }
    #[cfg(not(feature = "std"))]
    {
        run(&mut Scratch::default(), queries, keys, values, output, extents, rotary, score)
    }
}

#[expect(clippy::too_many_arguments, reason = "the layout bundle of the public entry, passed through")]
fn run(
    scratch: &mut Scratch,
    queries: [&[f32]; 2],
    keys: [[&[f32]; 2]; 2],
    values: [&[f32]; 2],
    output: &mut [f32],
    extents: &AttentionExtents,
    rotary: &CachedAttentionRotary<'_>,
    score: &CachedAttentionScore,
) -> bool {
    let pair_dim = (rotary.rotary_dim / 2) as usize;
    let pass_dim = (extents.head_dim - rotary.rotary_dim) as usize;
    let head_dim = extents.head_dim as usize;
    let kv_heads = extents.kv_heads as usize;
    let groups = extents.query_groups as usize;
    let query_rows = extents.query_rows as usize;
    let cached_rows = extents.cached_key_rows as usize;
    let new_rows = extents.new_key_rows as usize;
    if head_dim == 0 || pair_dim == 0 || kv_heads == 0 || groups == 0 || rotary.rotary_dim > extents.head_dim {
        return false;
    }
    let query_width = query_rows * kv_heads * groups * pair_dim;
    if queries.iter().any(|plane| plane.len() != query_width)
        || output.len() != query_rows * kv_heads * groups * head_dim
        || (cached_rows > 0 && values[0].len() != cached_rows * kv_heads * head_dim)
        || values[1].len() != new_rows * kv_heads * head_dim
    {
        return false;
    }
    let kernels = HalfKernels::detect();
    let ranges = [
        Range {
            keys: keys[0],
            pass_keys: rotary.pass.as_ref().map(|plane| plane.cached_key),
            values: values[0],
            rows: cached_rows,
            is_cached: true,
        },
        Range {
            keys: keys[1],
            pass_keys: rotary.pass.as_ref().map(|plane| plane.new_key),
            values: values[1],
            rows: new_rows,
            is_cached: false,
        },
    ];
    let row_width = kv_heads * head_dim;
    resized(&mut scratch.keys, (cached_rows + new_rows) * row_width);
    resized(&mut scratch.values, (cached_rows + new_rows) * row_width);
    resized(&mut scratch.query, head_dim);
    resized(&mut scratch.accumulator, head_dim);
    resized(&mut scratch.staging, head_dim);
    let mut base = 0;
    for range in &ranges {
        for key_row in 0..range.rows {
            for kv_head in 0..kv_heads {
                let source = (key_row * kv_heads + kv_head) * pair_dim;
                let pass_source = (key_row * kv_heads + kv_head) * pass_dim;
                let target = (base + key_row) * row_width + kv_head * head_dim;
                let destination = &mut scratch.keys[target..target + head_dim];
                kernels.convert(&range.keys[0][source..source + pair_dim], &mut destination[..pair_dim]);
                kernels.convert(&range.keys[1][source..source + pair_dim], &mut destination[pair_dim..2 * pair_dim]);
                if let Some(pass_keys) = range.pass_keys {
                    kernels.convert(&pass_keys[pass_source..pass_source + pass_dim], &mut destination[2 * pair_dim..]);
                }
                let value_source = (key_row * kv_heads + kv_head) * head_dim;
                kernels.convert(
                    &range.values[value_source..value_source + head_dim],
                    &mut scratch.values[target..target + head_dim],
                );
            }
        }
        base += range.rows;
    }
    for query_row in 0..query_rows {
        for kv_head in 0..kv_heads {
            for group in 0..groups {
                let query_head = kv_head * groups + group;
                let query_start = (query_row * kv_heads * groups + query_head) * pair_dim;
                let pass_start = (query_row * kv_heads * groups + query_head) * pass_dim;
                scratch.staging[..pair_dim].copy_from_slice(&queries[0][query_start..query_start + pair_dim]);
                scratch.staging[pair_dim..2 * pair_dim].copy_from_slice(&queries[1][query_start..query_start + pair_dim]);
                if let Some(plane) = rotary.pass.as_ref() {
                    scratch.staging[2 * pair_dim..].copy_from_slice(&plane.query[pass_start..pass_start + pass_dim]);
                }
                kernels.convert(&scratch.staging, &mut scratch.query);
                scratch.accumulator.fill(0);
                let mut running_sum = 0.0f32;
                let mut running_max = f32::NEG_INFINITY;
                let mut base = 0;
                for (range_index, range) in ranges.iter().enumerate() {
                    let band = &score.bands[range_index];
                    for key_row in 0..range.rows {
                        let key_position = if range.is_cached {
                            key_row as i64 - extents.cached_key_rows as i64
                        } else {
                            key_row as i64
                        };
                        let relative = key_position - query_row as i64;
                        if relative < band.lower_inclusive || relative > band.upper_inclusive {
                            continue;
                        }
                        let target = (base + key_row) * row_width + kv_head * head_dim;
                        let weighted = kernels.dot(&scratch.keys[target..target + head_dim], &scratch.query) * score.scale;
                        let previous_max = running_max;
                        let mut rescale = 1.0f32;
                        let mut weight = 1.0f32;
                        if weighted > running_max {
                            running_max = weighted;
                            rescale = libm::expf(previous_max - running_max);
                            kernels.scale(&mut scratch.accumulator, rescale);
                        } else {
                            weight = libm::expf(weighted - running_max);
                        }
                        kernels.mad(&mut scratch.accumulator, &scratch.values[target..target + head_dim], weight);
                        running_sum = running_sum * rescale + weight;
                    }
                    base += range.rows;
                }
                let inverse = if running_sum == 0.0 { 0.0 } else { 1.0 / running_sum };
                let output_start = (query_row * kv_heads * groups + query_head) * head_dim;
                for (slot, bits) in output[output_start..output_start + head_dim].iter_mut().zip(&scratch.accumulator) {
                    *slot = half_value(*bits) * inverse;
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f32 / (1u64 << 31) as f32) * 8.0 - 4.0
    }

    fn halves(state: &mut u64, length: usize) -> alloc::vec::Vec<u16> {
        (0..length).map(|_| half_bits(lcg(state))).collect()
    }

    #[test]
    fn avx2_kernels_match_the_scalar_twin_bit_for_bit_on_random_data() {
        let kernels = HalfKernels::detect();
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for length in [32usize, 64, 96, 128, 80, 40, 7, 1] {
            for _ in 0..200 {
                let key = halves(&mut state, length);
                let query = halves(&mut state, length);
                assert_eq!(kernels.dot(&key, &query).to_bits(), scalar::dot(&key, &query).to_bits(), "dot n={length}");
                let factor = lcg(&mut state);
                let mut simd = halves(&mut state, length);
                let mut reference = simd.clone();
                kernels.scale(&mut simd, factor);
                scalar::scale(&mut reference, factor);
                assert_eq!(simd, reference, "scale n={length}");
                let row = halves(&mut state, length);
                kernels.mad(&mut simd, &row, factor);
                scalar::mad(&mut reference, &row, factor);
                assert_eq!(simd, reference, "mad n={length}");
                let source: alloc::vec::Vec<f32> = (0..length).map(|_| lcg(&mut state)).collect();
                let mut converted = alloc::vec![0u16; length];
                let mut expected = alloc::vec![0u16; length];
                kernels.convert(&source, &mut converted);
                scalar::convert(&source, &mut expected);
                assert_eq!(converted, expected, "convert n={length}");
            }
        }
    }
}
