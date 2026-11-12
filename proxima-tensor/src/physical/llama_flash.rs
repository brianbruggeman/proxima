//! llama.cpp's default CPU attention, `ggml_compute_forward_flash_attn_ext_f16_one_chunk`
//! (`ggml/src/ggml-cpu/ops.cpp:8614-8850`), for one bound cached-attention op.
//!
//! Every line that mirrors the incumbent cites the line it mirrors. Q, K and V are f16, the score is
//! `ggml_vec_dot_f16`, the softmax is online with `expf`, the value accumulator lives in f16 and is
//! rounded after every scale and every multiply-add, and the running sum is f32. The vector kernels
//! are the AVX2 layouts ggml builds with: `GGML_F16_STEP` 32, `GGML_F16_EPR` 8, four accumulators
//! (`simd-mappings.h:635-683`), loads `_mm256_cvtph_ps`, stores `_mm256_cvtps_ph(.., 0)`
//! (round to nearest even). The scalar twin computes the same values operation for operation.

mod tiled;

use half::f16;

use super::{AttentionExtents, CachedAttentionRotary, CachedAttentionScore};

#[cfg(any(not(target_arch = "aarch64"), test))]
const STEP: usize = 32;
#[cfg(any(not(target_arch = "aarch64"), test))]
const LANES: usize = 8;
#[cfg(any(not(target_arch = "aarch64"), test))]
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
    /// The AVX2 order of ggml, computed operation for operation without vector instructions.
    #[cfg(any(not(target_arch = "aarch64"), test))]
    Scalar,
    /// ggml built with `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` (`simd-mappings.h:263-302`): the dot and the
    /// accumulator updates run in f16 arithmetic.
    #[cfg(any(target_arch = "aarch64", test))]
    NeonFp16,
    /// ggml built for NEON without f16 vector arithmetic (`simd-mappings.h:307-328`): f32 four-lane
    /// accumulators stepping 16, f16 storage.
    #[cfg(any(target_arch = "aarch64", test))]
    NeonF32,
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
        #[cfg(target_arch = "aarch64")]
        {
            #[cfg(feature = "std")]
            return Self::for_aarch64(std::arch::is_aarch64_feature_detected!("fp16"));
            #[cfg(not(feature = "std"))]
            return Self::for_aarch64(cfg!(target_feature = "fp16"));
        }
        #[cfg(not(target_arch = "aarch64"))]
        Self::Scalar
    }

    /// The kernel family an aarch64 host runs, by whether it has f16 vector arithmetic.
    #[cfg(any(target_arch = "aarch64", test))]
    pub(super) const fn for_aarch64(has_fp16_vectors: bool) -> Self {
        if has_fp16_vectors { Self::NeonFp16 } else { Self::NeonF32 }
    }

    /// f32 to f16 bits, `ggml_fp32_to_fp16_row` / `_mm256_cvtps_ph(.., 0)`.
    fn convert(self, source: &[f32], destination: &mut [u16]) {
        match self {
            #[cfg(all(feature = "std", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { x86::convert(source, destination) },
            #[cfg(any(not(target_arch = "aarch64"), test))]
            Self::Scalar => scalar::convert(source, destination),
            #[cfg(any(target_arch = "aarch64", test))]
            Self::NeonFp16 | Self::NeonF32 => scalar::convert(source, destination),
        }
    }

    /// `ggml_vec_dot_f16` (`vec.cpp:345-367`).
    fn dot(self, key: &[u16], query: &[u16]) -> f32 {
        match self {
            #[cfg(all(feature = "std", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { x86::dot(key, query) },
            #[cfg(any(not(target_arch = "aarch64"), test))]
            Self::Scalar => scalar::dot(key, query),
            #[cfg(any(target_arch = "aarch64", test))]
            Self::NeonFp16 => neon_twin::dot_fp16(key, query),
            #[cfg(any(target_arch = "aarch64", test))]
            Self::NeonF32 => neon_twin::dot_f32(key, query),
        }
    }

    /// `ggml_vec_scale_f16`.
    fn scale(self, accumulator: &mut [u16], factor: f32) {
        match self {
            #[cfg(all(feature = "std", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { x86::scale(accumulator, factor) },
            #[cfg(any(not(target_arch = "aarch64"), test))]
            Self::Scalar => scalar::scale(accumulator, factor),
            #[cfg(any(target_arch = "aarch64", test))]
            Self::NeonFp16 => neon_twin::scale_fp16(accumulator, factor),
            #[cfg(any(target_arch = "aarch64", test))]
            Self::NeonF32 => neon_twin::scale_f32(accumulator, factor),
        }
    }

    /// `ggml_vec_mad_f16`.
    fn mad(self, accumulator: &mut [u16], row: &[u16], factor: f32) {
        match self {
            #[cfg(all(feature = "std", target_arch = "x86_64"))]
            Self::Avx2 => unsafe { x86::mad(accumulator, row, factor) },
            #[cfg(any(not(target_arch = "aarch64"), test))]
            Self::Scalar => scalar::mad(accumulator, row, factor),
            #[cfg(any(target_arch = "aarch64", test))]
            Self::NeonFp16 => neon_twin::mad_fp16(accumulator, row, factor),
            #[cfg(any(target_arch = "aarch64", test))]
            Self::NeonF32 => neon_twin::mad_f32(accumulator, row, factor),
        }
    }
}

#[cfg(any(target_arch = "aarch64", test))]
pub(super) mod neon_twin {
    //! Scalar twins of ggml's aarch64 half-precision kernels, expanded from the macros in
    //! `ggml-cpu/simd-mappings.h`; real NEON intrinsics can replace them once there is hardware to verify on.

    use half::f16;

    use super::{half_bits, half_value};

    const FP16_STEP: usize = 32;
    const FP16_LANES: usize = 8;
    const F32_STEP: usize = 16;
    const F32_LANES: usize = 4;

    /// One f16 operation: the exact result rounded to nearest even, `vfmaq_f16` and `vaddq_f16`.
    fn round_half(value: f64) -> f32 {
        f16::from_f64(value).to_f32()
    }

    /// `GGML_F16x8_FMA` over eight lanes in f16: `vfmaq_f16(acc, a, b)` rounds `acc + a*b` once.
    fn fma_half(accumulator: f32, left: f32, right: f32) -> f32 {
        round_half(f64::from(left) * f64::from(right) + f64::from(accumulator))
    }

    /// `ggml_vec_dot_f16` with `GGML_F16x8_*` (`vec.cpp:345-367`, `simd-mappings.h:275-292`).
    pub(in crate::physical) fn dot_fp16(key: &[u16], query: &[u16]) -> f32 {
        let main = key.len() & !(FP16_STEP - 1);
        let mut sum = [[0f32; FP16_LANES]; 4];
        for step in (0..main).step_by(FP16_STEP) {
            for (array, lanes) in sum.iter_mut().enumerate() {
                for (lane, accumulator) in lanes.iter_mut().enumerate() {
                    let index = step + array * FP16_LANES + lane;
                    *accumulator = fma_half(*accumulator, half_value(key[index]), half_value(query[index]));
                }
            }
        }
        let add_half = |left: f32, right: f32| round_half(f64::from(left + right));
        let first: [f32; FP16_LANES] = core::array::from_fn(|lane| add_half(sum[0][lane], sum[2][lane]));
        let second: [f32; FP16_LANES] = core::array::from_fn(|lane| add_half(sum[1][lane], sum[3][lane]));
        let folded: [f32; FP16_LANES] = core::array::from_fn(|lane| add_half(first[lane], second[lane]));
        let lanes: [f32; 4] = core::array::from_fn(|lane| folded[lane] + folded[lane + 4]);
        let mut total = f64::from((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]));
        for index in main..key.len() {
            total += f64::from(half_value(key[index]) * half_value(query[index]));
        }
        total as f32
    }

    /// `ggml_vec_scale_f16`: the factor becomes an f16 (`vdupq_n_f16`), the vector lanes multiply in f16.
    pub(in crate::physical) fn scale_fp16(accumulator: &mut [u16], factor: f32) {
        let main = accumulator.len() & !(FP16_STEP - 1);
        let factor_half = half_value(half_bits(factor));
        for (index, value) in accumulator.iter_mut().enumerate() {
            let multiplier = if index < main { factor_half } else { factor };
            *value = half_bits(half_value(*value) * multiplier);
        }
    }

    /// `ggml_vec_mad_f16`: vector lanes `vfmaq_f16(acc, x, f16(v))`, leftovers multiply then add in f32.
    pub(in crate::physical) fn mad_fp16(accumulator: &mut [u16], row: &[u16], factor: f32) {
        let main = accumulator.len() & !(FP16_STEP - 1);
        let factor_half = half_value(half_bits(factor));
        for (index, value) in accumulator.iter_mut().enumerate() {
            *value = if index < main {
                half_bits(fma_half(half_value(*value), half_value(row[index]), factor_half))
            } else {
                half_bits(half_value(*value) + half_value(row[index]) * factor)
            };
        }
    }

    /// `ggml_vec_dot_f16` with `GGML_F32Cx4_*` (`simd-mappings.h:307-328`, `GGML_F32x4_REDUCE`).
    pub(in crate::physical) fn dot_f32(key: &[u16], query: &[u16]) -> f32 {
        let main = key.len() & !(F32_STEP - 1);
        let mut sum = [[0f32; F32_LANES]; 4];
        for step in (0..main).step_by(F32_STEP) {
            for (array, lanes) in sum.iter_mut().enumerate() {
                for (lane, accumulator) in lanes.iter_mut().enumerate() {
                    let index = step + array * F32_LANES + lane;
                    *accumulator = libm::fmaf(half_value(key[index]), half_value(query[index]), *accumulator);
                }
            }
        }
        let lanes: [f32; F32_LANES] =
            core::array::from_fn(|lane| (sum[0][lane] + sum[2][lane]) + (sum[1][lane] + sum[3][lane]));
        let mut total = f64::from((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]));
        for index in main..key.len() {
            total += f64::from(half_value(key[index]) * half_value(query[index]));
        }
        total as f32
    }

    pub(in crate::physical) fn scale_f32(accumulator: &mut [u16], factor: f32) {
        for value in accumulator {
            *value = half_bits(half_value(*value) * factor);
        }
    }

    /// `GGML_F32Cx4_FMA` is `vfmaq_f32` on widened values, the store rounds to f16; step 16 (`:307`).
    pub(in crate::physical) fn mad_f32(accumulator: &mut [u16], row: &[u16], factor: f32) {
        let main = accumulator.len() & !(F32_STEP - 1);
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

pub(super) mod scalar {
    #[cfg(any(not(target_arch = "aarch64"), test))]
    use super::{ARRAYS, LANES, STEP, half_value};
    use super::half_bits;

    pub(in crate::physical) fn convert(source: &[f32], destination: &mut [u16]) {
        for (slot, value) in destination.iter_mut().zip(source) {
            *slot = half_bits(*value);
        }
    }

    /// Four eight-lane fma accumulators stepping 32, the `GGML_F32x8_REDUCE` tree
    /// (`simd-mappings.h:602-620`), then the leftovers in double (`vec.cpp:365-367`).
    #[cfg(any(not(target_arch = "aarch64"), test))]
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

    #[cfg(any(not(target_arch = "aarch64"), test))]
    pub(in crate::physical) fn scale(accumulator: &mut [u16], factor: f32) {
        for value in accumulator {
            *value = half_bits(half_value(*value) * factor);
        }
    }

    /// The first `n & ~31` lanes fuse, the leftover lanes multiply then add (`vec.h` mad_f16).
    #[cfg(any(not(target_arch = "aarch64"), test))]
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
/// Per-call choices that are not part of the data layout.
#[derive(Clone, Copy)]
struct Options {
    workers: Option<usize>,
    interleaved: bool,
}

/// The head vector as ggml lays it out: split halves or interleaved pairs, then the pass-through tail.
fn assemble_head(target: &mut [f32], even: &[f32], odd: &[f32], pass: Option<&[f32]>, interleaved: bool) {
    let pair = even.len();
    if interleaved {
        for index in 0..pair {
            target[2 * index] = even[index];
            target[2 * index + 1] = odd[index];
        }
    } else {
        target[..pair].copy_from_slice(even);
        target[pair..2 * pair].copy_from_slice(odd);
    }
    if let Some(tail) = pass {
        target[2 * pair..2 * pair + tail.len()].copy_from_slice(tail);
    }
}

#[derive(Default)]
struct Scratch {
    keys: alloc::vec::Vec<u16>,
    values: alloc::vec::Vec<u16>,
    stage: alloc::vec::Vec<f32>,
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
}

fn resized<T: Clone + Default>(buffer: &mut alloc::vec::Vec<T>, length: usize) {
    if buffer.len() < length {
        buffer.resize(length, T::default());
    }
}

/// `expf` of the platform math library, the function ggml calls (`ops.cpp:8768`, `:8774`).
fn expf(value: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        value.exp()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::expf(value)
    }
}

/// [`super::stream_cached_attention_split_gqa`]'s contract and layouts, computed the way llama.cpp's
/// default CPU attention computes them.
#[must_use]
#[expect(clippy::too_many_arguments, reason = "the layout bundle of the sibling online kernel plus the head-vector order")]
pub fn stream_cached_attention_llama_flash(
    queries: [&[f32]; 2],
    keys: [[&[f32]; 2]; 2],
    values: [&[f32]; 2],
    output: &mut [f32],
    extents: &AttentionExtents,
    rotary: &CachedAttentionRotary<'_>,
    score: &CachedAttentionScore<'_>,
    interleaved: bool,
) -> bool {
    #[cfg(feature = "std")]
    {
        SCRATCH.with(|cell| match cell.try_borrow_mut() {
            Ok(mut scratch) => run(&mut scratch, Options { workers: None, interleaved }, queries, keys, values, output, extents, rotary, score),
            Err(_) => run(&mut Scratch::default(), Options { workers: None, interleaved }, queries, keys, values, output, extents, rotary, score),
        })
    }
    #[cfg(not(feature = "std"))]
    {
        run(&mut Scratch::default(), Options { workers: None, interleaved }, queries, keys, values, output, extents, rotary, score)
    }
}

#[expect(clippy::too_many_arguments, reason = "the layout bundle of the public entry, passed through")]
fn run(
    scratch: &mut Scratch,
    options: Options,
    queries: [&[f32]; 2],
    keys: [[&[f32]; 2]; 2],
    values: [&[f32]; 2],
    output: &mut [f32],
    extents: &AttentionExtents,
    rotary: &CachedAttentionRotary<'_>,
    score: &CachedAttentionScore<'_>,
) -> bool {
    let pair_dim = (rotary.rotary_dim / 2) as usize;
    let pass_dim = (extents.head_dim - rotary.rotary_dim) as usize;
    let head_dim = extents.head_dim as usize;
    let kv_heads = extents.kv_heads as usize;
    let groups = extents.query_groups as usize;
    let query_rows = extents.query_rows as usize;
    let cached_rows = extents.cached_key_rows as usize;
    let new_rows = extents.new_key_rows as usize;
    if head_dim == 0
        || pair_dim == 0
        || kv_heads == 0
        || groups == 0
        || rotary.rotary_dim > extents.head_dim
        || !score.sinks_fit(extents)
    {
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
        },
        Range {
            keys: keys[1],
            pass_keys: rotary.pass.as_ref().map(|plane| plane.new_key),
            values: values[1],
            rows: new_rows,
        },
    ];
    let row_width = kv_heads * head_dim;
    resized(&mut scratch.keys, (cached_rows + new_rows) * row_width);
    resized(&mut scratch.values, (cached_rows + new_rows) * row_width);
    resized(&mut scratch.stage, head_dim);
    let mut base = 0;
    for range in &ranges {
        for key_row in 0..range.rows {
            for kv_head in 0..kv_heads {
                let source = (key_row * kv_heads + kv_head) * pair_dim;
                let pass_source = (key_row * kv_heads + kv_head) * pass_dim;
                let target = (base + key_row) * row_width + kv_head * head_dim;
                let destination = &mut scratch.keys[target..target + head_dim];
                assemble_head(
                    &mut scratch.stage[..head_dim],
                    &range.keys[0][source..source + pair_dim],
                    &range.keys[1][source..source + pair_dim],
                    range.pass_keys.map(|pass_keys| &pass_keys[pass_source..pass_source + pass_dim]),
                    options.interleaved,
                );
                kernels.convert(&scratch.stage[..head_dim], destination);
                let value_source = (key_row * kv_heads + kv_head) * head_dim;
                kernels.convert(
                    &range.values[value_source..value_source + head_dim],
                    &mut scratch.values[target..target + head_dim],
                );
            }
        }
        base += range.rows;
    }
    let shared = Shared {
        kernels,
        queries,
        pass_query: rotary.pass.as_ref().map(|plane| plane.query),
        keys: &scratch.keys,
        values: &scratch.values,
        score,
        cached_rows,
        new_rows,
        kv_heads,
        groups,
        pair_dim,
        pass_dim,
        head_dim,
        row_width,
        interleaved: options.interleaved,
    };
    let tiled = (query_rows >= tiled::TILE_Q).then(|| tiled::TiledData::build(&shared));
    let tiled = tiled.as_ref();
    let row_len = kv_heads * groups * head_dim;
    let workers = options.workers.unwrap_or_else(|| worker_count(query_rows, cached_rows + new_rows, row_len));
    if workers <= 1 {
        let mut local = Local::new(head_dim);
        for (query_row, out) in output.chunks_mut(row_len).enumerate() {
            shared.attend(tiled, query_row, out, &mut local);
        }
        return true;
    }
    #[cfg(feature = "std")]
    {
        let shared = &shared;
        let mut assignments: alloc::vec::Vec<alloc::vec::Vec<(usize, &mut [f32])>> =
            (0..workers).map(|_| alloc::vec::Vec::new()).collect();
        for (query_row, out) in output.chunks_mut(row_len).enumerate() {
            assignments[query_row % workers].push((query_row, out));
        }
        std::thread::scope(|scope| {
            for assignment in assignments {
                scope.spawn(move || {
                    let mut local = Local::new(head_dim);
                    for (query_row, out) in assignment {
                        shared.attend(tiled, query_row, out, &mut local);
                    }
                });
            }
        });
    }
    true
}

/// Threads only pay for multi-row calls (prefill, verify); each worker owns whole query rows, so the
/// output blocks are disjoint and the result is the single-threaded result bit for bit.
fn worker_count(query_rows: usize, key_rows: usize, row_len: usize) -> usize {
    #[cfg(feature = "std")]
    {
        const MIN_WORK: usize = 1 << 21;
        if query_rows < 2 || query_rows * key_rows * row_len < MIN_WORK {
            return 1;
        }
        std::thread::available_parallelism().map_or(1, usize::from).min(query_rows)
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = (query_rows, key_rows, row_len);
        1
    }
}

struct Local {
    query: alloc::vec::Vec<u16>,
    accumulator: alloc::vec::Vec<u16>,
    staging: alloc::vec::Vec<f32>,
    acc32: alloc::vec::Vec<f32>,
}

impl Local {
    fn new(head_dim: usize) -> Self {
        Self {
            query: alloc::vec![0; head_dim],
            accumulator: alloc::vec![0; head_dim],
            staging: alloc::vec![0.0; head_dim],
            acc32: alloc::vec![0.0; head_dim],
        }
    }
}

struct Shared<'call> {
    kernels: HalfKernels,
    queries: [&'call [f32]; 2],
    pass_query: Option<&'call [f32]>,
    keys: &'call [u16],
    values: &'call [u16],
    score: &'call CachedAttentionScore<'call>,
    cached_rows: usize,
    new_rows: usize,
    kv_heads: usize,
    groups: usize,
    pair_dim: usize,
    pass_dim: usize,
    head_dim: usize,
    row_width: usize,
    interleaved: bool,
}

impl Shared<'_> {
    /// One query row across every head: `ggml_compute_forward_flash_attn_ext_f16_one_chunk`'s outer
    /// loop body (`ops.cpp:8698-8848`).
    fn attend_row(&self, query_row: usize, out: &mut [f32], local: &mut Local) {
        let Self {
            kernels,
            queries,
            pass_query,
            keys,
            values,
            score,
            cached_rows,
            new_rows,
            kv_heads,
            groups,
            pair_dim,
            pass_dim,
            head_dim,
            row_width,
            interleaved,
        } = *self;
        for kv_head in 0..kv_heads {
            for group in 0..groups {
                let query_head = kv_head * groups + group;
                let query_start = (query_row * kv_heads * groups + query_head) * pair_dim;
                let pass_start = (query_row * kv_heads * groups + query_head) * pass_dim;
                assemble_head(
                    &mut local.staging[..head_dim],
                    &queries[0][query_start..query_start + pair_dim],
                    &queries[1][query_start..query_start + pair_dim],
                    pass_query.map(|plane| &plane[pass_start..pass_start + pass_dim]),
                    interleaved,
                );
                kernels.convert(&local.staging[..head_dim], &mut local.query[..head_dim]);
                local.accumulator[..head_dim].fill(0);
                let mut running_sum = 0.0f32;
                let mut running_max = f32::NEG_INFINITY;
                for (range_index, (base, rows)) in [(0, cached_rows), (cached_rows, new_rows)].into_iter().enumerate() {
                    let band = &score.bands[range_index];
                    for key_row in 0..rows {
                        let key_position = if range_index == 0 {
                            key_row as i64 - cached_rows as i64
                        } else {
                            key_row as i64
                        };
                        let relative = key_position - query_row as i64;
                        if relative < band.lower_inclusive || relative > band.upper_inclusive {
                            continue;
                        }
                        let target = (base + key_row) * row_width + kv_head * head_dim;
                        let weighted = kernels.dot(&keys[target..target + head_dim], &local.query[..head_dim]) * score.scale;
                        let previous_max = running_max;
                        let mut rescale = 1.0f32;
                        let mut weight = 1.0f32;
                        if weighted > running_max {
                            running_max = weighted;
                            rescale = expf(previous_max - running_max);
                            kernels.scale(&mut local.accumulator[..head_dim], rescale);
                        } else {
                            weight = expf(weighted - running_max);
                        }
                        kernels.mad(&mut local.accumulator[..head_dim], &values[target..target + head_dim], weight);
                        running_sum = running_sum * rescale + weight;
                    }
                }
                if let Some(sink) = score.sink_of(query_head) {
                    if sink > running_max {
                        kernels.scale(&mut local.accumulator[..head_dim], expf(running_max - sink));
                        running_sum = running_sum * expf(running_max - sink) + 1.0;
                    } else {
                        running_sum += expf(sink - running_max);
                    }
                }
                let inverse = if running_sum == 0.0 { 0.0 } else { 1.0 / running_sum };
                let output_start = query_head * head_dim;
                for (slot, bits) in out[output_start..output_start + head_dim].iter_mut().zip(&local.accumulator[..head_dim]) {
                    *slot = half_value(*bits) * inverse;
                }
            }
        }
    }
}


impl Shared<'_> {
    fn attend(&self, tiled: Option<&tiled::TiledData>, query_row: usize, out: &mut [f32], local: &mut Local) {
        match tiled {
            Some(data) => self.attend_row_tiled(data, query_row, out, local),
            None => self.attend_row(query_row, out, local),
        }
    }
}

// The AVX2 arm is the only one that runs a different implementation from the
// scalar twin it is compared against, so this module exists where that arm can
// be selected; the aarch64 twins are checked against their hand-expanded macro
// sequences in `aarch64_twin_tests` instead.
#[cfg(all(test, not(target_arch = "aarch64")))]
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

#[cfg(test)]
mod threading_tests {
    use super::super::{CausalBand, CachedAttentionScore};
    use super::*;

    fn random(state: &mut u64, length: usize) -> alloc::vec::Vec<f32> {
        (0..length)
            .map(|_| {
                *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((*state >> 33) as f32 / (1u64 << 31) as f32) * 6.0 - 3.0
            })
            .collect()
    }

    #[test]
    fn worker_threads_reproduce_the_single_thread_result_bit_for_bit() {
        let (rows, cached, kv_heads, groups, head_dim) = (24usize, 40usize, 2usize, 3usize, 64usize);
        let pair = head_dim / 2;
        let mut state = 0x1234_5678_9abc_def0u64;
        let query_even = random(&mut state, rows * kv_heads * groups * pair);
        let query_odd = random(&mut state, rows * kv_heads * groups * pair);
        let cached_even = random(&mut state, cached * kv_heads * pair);
        let cached_odd = random(&mut state, cached * kv_heads * pair);
        let new_even = random(&mut state, rows * kv_heads * pair);
        let new_odd = random(&mut state, rows * kv_heads * pair);
        let cached_value = random(&mut state, cached * kv_heads * head_dim);
        let new_value = random(&mut state, rows * kv_heads * head_dim);
        let extents = AttentionExtents {
            query_rows: rows as u64,
            cached_key_rows: cached as u64,
            new_key_rows: rows as u64,
            kv_heads: kv_heads as u64,
            query_groups: groups as u64,
            head_dim: head_dim as u64,
        };
        let rotary = CachedAttentionRotary { rotary_dim: head_dim as u64, pass: None };
        let score = CachedAttentionScore {
            scale: 0.125,
            sinks: None,
            bands: [
                CausalBand { lower_inclusive: i64::MIN, upper_inclusive: i64::MAX },
                CausalBand { lower_inclusive: i64::MIN, upper_inclusive: 0 },
            ],
        };
        let mut outputs = alloc::vec::Vec::new();
        for workers in [1usize, 2, 5] {
            let mut output = alloc::vec![0f32; rows * kv_heads * groups * head_dim];
            assert!(run(
                &mut Scratch::default(),
                Options { workers: Some(workers), interleaved: false },
                [&query_even, &query_odd],
                [[&cached_even, &cached_odd], [&new_even, &new_odd]],
                [&cached_value, &new_value],
                &mut output,
                &extents,
                &rotary,
                &score,
            ));
            outputs.push(output.iter().map(|value| value.to_bits()).collect::<alloc::vec::Vec<u32>>());
        }
        assert_eq!(outputs[0], outputs[1]);
        assert_eq!(outputs[0], outputs[2]);
    }
}

#[cfg(test)]
mod aarch64_twin_tests {
    use half::f16;

    use super::*;

    fn bits(values: &[f32]) -> alloc::vec::Vec<u16> {
        values.iter().map(|value| half_bits(*value)).collect()
    }

    fn fixed_key() -> alloc::vec::Vec<u16> {
        bits(&(0..64).map(|index| ((index * 7 % 13) as f32 - 6.0) * 0.375).collect::<alloc::vec::Vec<f32>>())
    }

    fn fixed_query() -> alloc::vec::Vec<u16> {
        bits(&(0..64).map(|index| ((index * 5 % 11) as f32 - 5.0) * 0.5).collect::<alloc::vec::Vec<f32>>())
    }

    fn half(value: f64) -> f32 {
        f16::from_f64(value).to_f32()
    }

    /// `GGML_F16x8_*` expanded by hand: four `float16x8_t` registers, `vfmaq_f16` per step, the
    /// `vaddq_f16` tree, `vcvt_f32_f16` of the low and high halves, `vaddvq_f32`.
    fn hand_expanded_fp16_dot(key: &[u16], query: &[u16]) -> f32 {
        let mut x0 = [0f32; 8];
        let mut x1 = [0f32; 8];
        let mut x2 = [0f32; 8];
        let mut x3 = [0f32; 8];
        for step in [0usize, 32] {
            for lane in 0..8 {
                let product = |register: usize| {
                    let index = step + register * 8 + lane;
                    f64::from(half_value(key[index])) * f64::from(half_value(query[index]))
                };
                x0[lane] = half(product(0) + f64::from(x0[lane]));
                x1[lane] = half(product(1) + f64::from(x1[lane]));
                x2[lane] = half(product(2) + f64::from(x2[lane]));
                x3[lane] = half(product(3) + f64::from(x3[lane]));
            }
        }
        for lane in 0..8 {
            x0[lane] = half(f64::from(x0[lane]) + f64::from(x2[lane]));
            x1[lane] = half(f64::from(x1[lane]) + f64::from(x3[lane]));
            x0[lane] = half(f64::from(x0[lane]) + f64::from(x1[lane]));
        }
        let t0 = [x0[0], x0[1], x0[2], x0[3]];
        let t1 = [x0[4], x0[5], x0[6], x0[7]];
        let summed = [t0[0] + t1[0], t0[1] + t1[1], t0[2] + t1[2], t0[3] + t1[3]];
        ((summed[0] + summed[1]) + (summed[2] + summed[3])) as f64 as f32
    }

    /// `GGML_F32Cx4_*` expanded by hand: four `float32x4_t` registers stepping 16, `vfmaq_f32`,
    /// `GGML_F32x4_REDUCE`.
    fn hand_expanded_f32_dot(key: &[u16], query: &[u16]) -> f32 {
        let mut registers = [[0f32; 4]; 4];
        for step in [0usize, 16, 32, 48] {
            for (register, lanes) in registers.iter_mut().enumerate() {
                for (lane, slot) in lanes.iter_mut().enumerate() {
                    let index = step + register * 4 + lane;
                    *slot = half_value(key[index]).mul_add(half_value(query[index]), *slot);
                }
            }
        }
        let mut folded = [0f32; 4];
        for lane in 0..4 {
            let low = registers[0][lane] + registers[2][lane];
            let high = registers[1][lane] + registers[3][lane];
            folded[lane] = low + high;
        }
        (folded[0] + folded[1]) + (folded[2] + folded[3])
    }

    #[test]
    fn the_fp16_twin_matches_the_hand_expanded_macro_sequence() {
        let (key, query) = (fixed_key(), fixed_query());
        assert_eq!(neon_twin::dot_fp16(&key, &query).to_bits(), hand_expanded_fp16_dot(&key, &query).to_bits());
    }

    #[test]
    fn the_f32_twin_matches_the_hand_expanded_macro_sequence() {
        let (key, query) = (fixed_key(), fixed_query());
        assert_eq!(neon_twin::dot_f32(&key, &query).to_bits(), hand_expanded_f32_dot(&key, &query).to_bits());
    }

    #[test]
    fn the_fp16_accumulator_updates_round_the_factor_and_every_product_to_f16() {
        let mut accumulator = bits(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8].repeat(8));
        let row = bits(&[1.5, -2.25, 3.125, 0.0625, -0.5, 4.0, -1.75, 0.3].repeat(8));
        let factor = 0.333_333_34f32;
        let expected: alloc::vec::Vec<u16> = accumulator
            .iter()
            .zip(&row)
            .map(|(value, x)| half_bits(half(f64::from(half_value(*x)) * f64::from(half(f64::from(factor))) + f64::from(half_value(*value)))))
            .collect();
        neon_twin::mad_fp16(&mut accumulator, &row, factor);
        assert_eq!(accumulator, expected);
        let mut scaled = bits(&[1.0, 2.0, 3.0, 4.0].repeat(16));
        neon_twin::scale_fp16(&mut scaled, factor);
        let expected_scaled: alloc::vec::Vec<u16> = bits(&[1.0, 2.0, 3.0, 4.0].repeat(16))
            .iter()
            .map(|value| half_bits(half_value(*value) * half(f64::from(factor))))
            .collect();
        assert_eq!(scaled, expected_scaled);
    }

    #[test]
    fn an_aarch64_host_selects_the_twin_its_f16_vector_support_names() {
        assert_eq!(HalfKernels::for_aarch64(true), HalfKernels::NeonFp16);
        assert_eq!(HalfKernels::for_aarch64(false), HalfKernels::NeonF32);
    }

    #[test]
    fn the_three_families_disagree_on_a_dot_that_f16_accumulation_rounds() {
        let mut values = alloc::vec![1024.0f32; 32];
        values.extend(core::iter::repeat_n(0.75f32, 32));
        let key = bits(&values);
        let query = bits(&[1.0; 64]);
        let avx = HalfKernels::Scalar.dot(&key, &query);
        let fp16 = neon_twin::dot_fp16(&key, &query);
        assert_ne!(avx.to_bits(), fp16.to_bits(), "the f16 accumulating dot must visibly differ from the f32 accumulating one");
    }
}

#[cfg(test)]
mod pairing_order_tests {
    use super::super::{CachedAttentionScore, CausalBand};
    use super::*;

    fn random(state: &mut u64, length: usize) -> alloc::vec::Vec<f32> {
        (0..length)
            .map(|_| {
                *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((*state >> 33) as f32 / (1u64 << 31) as f32) * 16.0 - 8.0
            })
            .collect()
    }

    #[test]
    fn a_head_vector_lies_in_memory_as_the_pairing_names() {
        let even = [10.0f32, 11.0, 12.0];
        let odd = [20.0f32, 21.0, 22.0];
        let tail = [30.0f32, 31.0];
        let mut split = [0f32; 8];
        let mut interleaved = [0f32; 8];
        assemble_head(&mut split, &even, &odd, Some(&tail), false);
        assemble_head(&mut interleaved, &even, &odd, Some(&tail), true);
        assert_eq!(split, [10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0]);
        assert_eq!(interleaved, [10.0, 20.0, 11.0, 21.0, 12.0, 22.0, 30.0, 31.0]);
    }

    /// One query row, one kv head, every key visible: the kernel and a memory-order reference built
    /// from the scalar twins must agree for each pairing, and the two pairings must not agree.
    #[test]
    fn the_dot_sums_in_memory_order_for_both_pairings() {
        let (keys, head_dim) = (6usize, 64usize);
        let pair = head_dim / 2;
        let mut state = 0xfeed_beef_1234_5678u64;
        let query_even = random(&mut state, pair);
        let query_odd = random(&mut state, pair);
        let key_even = random(&mut state, keys * pair);
        let key_odd = random(&mut state, keys * pair);
        let values = random(&mut state, keys * head_dim);
        let extents = AttentionExtents {
            query_rows: 1,
            cached_key_rows: 0,
            new_key_rows: keys as u64,
            kv_heads: 1,
            query_groups: 1,
            head_dim: head_dim as u64,
        };
        let rotary = CachedAttentionRotary { rotary_dim: head_dim as u64, pass: None };
        let score = CachedAttentionScore {
            scale: 0.125,
            sinks: None,
            bands: [
                CausalBand { lower_inclusive: i64::MIN, upper_inclusive: i64::MAX },
                CausalBand { lower_inclusive: i64::MIN, upper_inclusive: i64::MAX },
            ],
        };
        // the reference runs the same kernel family the op does, so this
        // comparison isolates memory order from accumulation precision
        let kernels = HalfKernels::detect();
        let reference = |interleaved: bool| -> alloc::vec::Vec<u32> {
            let mut query = alloc::vec![0f32; head_dim];
            assemble_head(&mut query, &query_even, &query_odd, None, interleaved);
            let query: alloc::vec::Vec<u16> = query.iter().map(|value| half_bits(*value)).collect();
            let mut accumulator = alloc::vec![0u16; head_dim];
            let (mut max, mut sum) = (f32::NEG_INFINITY, 0.0f32);
            for key in 0..keys {
                let mut row = alloc::vec![0f32; head_dim];
                assemble_head(&mut row, &key_even[key * pair..][..pair], &key_odd[key * pair..][..pair], None, interleaved);
                let row: alloc::vec::Vec<u16> = row.iter().map(|value| half_bits(*value)).collect();
                let weighted = kernels.dot(&row, &query) * 0.125;
                let (mut rescale, mut weight) = (1.0f32, 1.0f32);
                if weighted > max {
                    rescale = expf(max - weighted);
                    max = weighted;
                    kernels.scale(&mut accumulator, rescale);
                } else {
                    weight = expf(weighted - max);
                }
                let value: alloc::vec::Vec<u16> = values[key * head_dim..][..head_dim].iter().map(|raw| half_bits(*raw)).collect();
                kernels.mad(&mut accumulator, &value, weight);
                sum = sum * rescale + weight;
            }
            accumulator.iter().map(|bits| (half_value(*bits) * (1.0 / sum)).to_bits()).collect()
        };
        let mut results = alloc::vec::Vec::new();
        for interleaved in [false, true] {
            let mut output = alloc::vec![0f32; head_dim];
            assert!(stream_cached_attention_llama_flash(
                [&query_even, &query_odd],
                [[&[], &[]], [&key_even, &key_odd]],
                [&[], &values],
                &mut output,
                &extents,
                &rotary,
                &score,
                interleaved,
            ));
            let bits: alloc::vec::Vec<u32> = output.iter().map(|value| value.to_bits()).collect();
            assert_eq!(bits, reference(interleaved), "kernel vs memory-order reference, interleaved={interleaved}");
            results.push(bits);
        }
        let dot_in = |interleaved: bool, key: usize| {
            let mut query = alloc::vec![0f32; head_dim];
            assemble_head(&mut query, &query_even, &query_odd, None, interleaved);
            let mut row = alloc::vec![0f32; head_dim];
            assemble_head(&mut row, &key_even[key * pair..][..pair], &key_odd[key * pair..][..pair], None, interleaved);
            let to_bits = |values: &[f32]| values.iter().map(|value| half_bits(*value)).collect::<alloc::vec::Vec<u16>>();
            kernels.dot(&to_bits(&row), &to_bits(&query)).to_bits()
        };
        let differing_keys = (0..keys).filter(|key| dot_in(false, *key) != dot_in(true, *key)).count();
        assert!(differing_keys > 0, "summing the same products in the two memory orders must round differently for some key");
    }
}
