//! llama.cpp's prefill attention, `ggml_compute_forward_flash_attn_ext_tiled`
//! (`ggml/src/ggml-cpu/ops.cpp:8852-9138`), taken when a call carries at least `GGML_FA_TILE_Q` (64,
//! `ggml-cpu/common.h:9`) query rows (`ops.cpp:9318-9322`). It is a different arithmetic from the
//! one-chunk path: the query stays f32, keys and values widen from f16 to f32, scores come from an f32
//! fused-multiply-add GEMM over `GGML_FA_TILE_KV` (64) keys at a time, the softmax is the vector
//! `ggml_v_expf` (`vec.h:1215-1252`), and the value accumulator is f32.
//!
//! Every row is independent in every step (the GEMM accumulates each output element on its own, the
//! max, the sum and the rescale are per row), so the result does not depend on how rows are grouped
//! into 64-row tiles; the key tiles do matter, and they start at key zero exactly as in ggml.

use super::{Local, Shared, expf, half_value};

pub(super) const TILE_Q: usize = 64;
const TILE_KV: usize = 64;

/// `0x1.xxxxxxp<exp>` literals of `ggml_v_expf`: six hex digits are 24 bits, the f32 mantissa keeps 23.
const fn hex_float(digits: u32, exponent: i32) -> f32 {
    f32::from_bits((((exponent + 127) as u32) << 23) | (digits >> 1))
}

const ROUND: f32 = 12582912.0;
const LOG2_E: f32 = hex_float(0x715476, 0);
const LN2_LOW: f32 = hex_float(0x7f7d1c, -20);
const LN2_HIGH: f32 = hex_float(0x62e400, -1);
const P0: f32 = hex_float(0x0e4020, -7);
const P1: f32 = hex_float(0x573e2e, -5);
const P2: f32 = hex_float(0x555e66, -3);
const P3: f32 = hex_float(0xfffdb6, -2);
const P4: f32 = hex_float(0xffffec, -1);

fn fma(left: f32, right: f32, addend: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        left.mul_add(right, addend)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::fmaf(left, right, addend)
    }
}

/// One lane of `ggml_v_expf` for AVX2 (`vec.h:1215-1252`), operation for operation.
pub(super) fn v_expf_lane(x: f32) -> f32 {
    let z = fma(x, LOG2_E, ROUND);
    let n = z - ROUND;
    let b = fma(-n, LN2_LOW, fma(-n, LN2_HIGH, x));
    let e = z.to_bits() << 23;
    let k = f32::from_bits(e.wrapping_add(1.0f32.to_bits()));
    let big = n.abs() > 126.0;
    let u = b * b;
    let j = fma(
        fma(fma(P0, b, P1), u, fma(P2, b, P3)),
        u,
        P4 * b,
    );
    if !big {
        return fma(j, k, k);
    }
    let g = if n <= 0.0 { 0x8200_0000u32 } else { 0 };
    let s1 = f32::from_bits(g.wrapping_add(0x7f00_0000));
    let s2 = f32::from_bits(e.wrapping_sub(g));
    if n.abs() > 192.0 {
        s1 * s1
    } else {
        fma(s2, j, s2) * s1
    }
}

/// `ggml_vec_soft_max_f32` on AVX2 (`vec.cpp:541-551`): eight-lane blocks, each summed
/// `(l0+l4)+(l2+l6)` then `+ ((l1+l5)+(l3+l7))`, the block sums added in double.
fn soft_max(values: &mut [f32; TILE_KV], max: f32) -> f64 {
    let mut sum = 0.0f64;
    for block in values.as_chunks_mut::<8>().0 {
        for lane in block.iter_mut() {
            *lane = v_expf_lane(*lane - max);
        }
        let low: [f32; 4] = core::array::from_fn(|lane| block[lane + 4] + block[lane]);
        let moved = [low[0] + low[2], low[1] + low[3]];
        sum += f64::from(moved[0] + moved[1]);
    }
    sum
}

/// `simd_gemm` row of the score GEMM (`simd-gemm.h:23-57`): every output column accumulates
/// `fma(key[dk][column], query[dk], acc)` over `dk` in order, starting from zero.
fn score_row(kernels: Kernels, query: &[f32], keys: &[f32], out: &mut [f32; TILE_KV]) {
    match kernels {
        #[cfg(all(feature = "std", target_arch = "x86_64"))]
        Kernels::Avx2 => unsafe { x86::score_row(query, keys, out) },
        Kernels::Scalar => scalar_score_row(query, keys, out),
    }
}

fn scalar_score_row(query: &[f32], keys: &[f32], out: &mut [f32; TILE_KV]) {
    out.fill(0.0);
    for (dimension, weight) in query.iter().enumerate() {
        let row = &keys[dimension * TILE_KV..(dimension + 1) * TILE_KV];
        for (slot, key) in out.iter_mut().zip(row) {
            *slot = fma(*key, *weight, *slot);
        }
    }
}

/// `simd_gemm` row of the value GEMM: `acc[d] = fma(value[tk][d], weight[tk], acc[d])` over the tile's keys.
fn value_row(kernels: Kernels, weights: &[f32; TILE_KV], values: &[f32], accumulator: &mut [f32]) {
    match kernels {
        #[cfg(all(feature = "std", target_arch = "x86_64"))]
        Kernels::Avx2 => unsafe { x86::value_row(weights, values, accumulator) },
        Kernels::Scalar => scalar_value_row(weights, values, accumulator),
    }
}

fn scalar_value_row(weights: &[f32; TILE_KV], values: &[f32], accumulator: &mut [f32]) {
    let width = accumulator.len();
    for (key, weight) in weights.iter().enumerate() {
        let row = &values[key * width..(key + 1) * width];
        for (slot, value) in accumulator.iter_mut().zip(row) {
            *slot = fma(*value, *weight, *slot);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Kernels {
    Scalar,
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    Avx2,
}

impl Kernels {
    pub(super) fn detect() -> Self {
        #[cfg(all(feature = "std", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return Self::Avx2;
        }
        Self::Scalar
    }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
mod x86 {
    use core::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps};

    use super::TILE_KV;

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn score_row(query: &[f32], keys: &[f32], out: &mut [f32; TILE_KV]) {
        let mut accumulators = [_mm256_setzero_ps(); TILE_KV / 8];
        for (dimension, weight) in query.iter().enumerate() {
            let broadcast = _mm256_set1_ps(*weight);
            for (block, accumulator) in accumulators.iter_mut().enumerate() {
                unsafe {
                    let key = _mm256_loadu_ps(keys.as_ptr().add(dimension * TILE_KV + block * 8));
                    *accumulator = _mm256_fmadd_ps(key, broadcast, *accumulator);
                }
            }
        }
        for (block, accumulator) in accumulators.iter().enumerate() {
            unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(block * 8), *accumulator) };
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn value_row(weights: &[f32; TILE_KV], values: &[f32], accumulator: &mut [f32]) {
        let width = accumulator.len();
        let blocks = width / 8;
        for block in 0..blocks {
            unsafe {
                let mut sum = _mm256_loadu_ps(accumulator.as_ptr().add(block * 8));
                for (key, weight) in weights.iter().enumerate() {
                    let value = _mm256_loadu_ps(values.as_ptr().add(key * width + block * 8));
                    sum = _mm256_fmadd_ps(value, _mm256_set1_ps(*weight), sum);
                }
                _mm256_storeu_ps(accumulator.as_mut_ptr().add(block * 8), sum);
            }
        }
        for lane in blocks * 8..width {
            let mut sum = accumulator[lane];
            for (key, weight) in weights.iter().enumerate() {
                sum = values[key * width + lane].mul_add(*weight, sum);
            }
            accumulator[lane] = sum;
        }
    }
}

/// Keys transposed per key tile (`ops.cpp:9020-9035`) and values widened (`:9087-9094`), built once per call.
pub(super) struct TiledData {
    keys: alloc::vec::Vec<f32>,
    values: alloc::vec::Vec<f32>,
    tiles: usize,
    padded_rows: usize,
}

impl TiledData {
    pub(super) fn build(shared: &Shared<'_>) -> Self {
        let total = shared.cached_rows + shared.new_rows;
        let tiles = total.div_ceil(TILE_KV);
        let padded_rows = tiles * TILE_KV;
        let head_dim = shared.head_dim;
        let mut keys = alloc::vec![0f32; shared.kv_heads * tiles * head_dim * TILE_KV];
        let mut values = alloc::vec![0f32; shared.kv_heads * padded_rows * head_dim];
        for kv_head in 0..shared.kv_heads {
            for row in 0..total {
                let source = row * shared.row_width + kv_head * head_dim;
                let (tile, column) = (row / TILE_KV, row % TILE_KV);
                for dimension in 0..head_dim {
                    let target = ((kv_head * tiles + tile) * head_dim + dimension) * TILE_KV + column;
                    keys[target] = half_value(shared.keys[source + dimension]);
                    values[(kv_head * padded_rows + row) * head_dim + dimension] =
                        half_value(shared.values[source + dimension]);
                }
            }
        }
        Self {
            keys,
            values,
            tiles,
            padded_rows,
        }
    }
}

impl Shared<'_> {
    /// One query row across every head, `ggml_compute_forward_flash_attn_ext_tiled`'s per-row work.
    pub(super) fn attend_row_tiled(&self, data: &TiledData, query_row: usize, out: &mut [f32], local: &mut Local) {
        let kernels = Kernels::detect();
        let head_dim = self.head_dim;
        let total = self.cached_rows + self.new_rows;
        for kv_head in 0..self.kv_heads {
            for group in 0..self.groups {
                let query_head = kv_head * self.groups + group;
                let query_start = (query_row * self.kv_heads * self.groups + query_head) * self.pair_dim;
                let pass_start = (query_row * self.kv_heads * self.groups + query_head) * self.pass_dim;
                let query = &mut local.staging[..head_dim];
                super::assemble_head(
                    query,
                    &self.queries[0][query_start..query_start + self.pair_dim],
                    &self.queries[1][query_start..query_start + self.pair_dim],
                    self.pass_query.map(|plane| &plane[pass_start..pass_start + self.pass_dim]),
                    self.interleaved,
                );
                let accumulator = &mut local.acc32[..head_dim];
                accumulator.fill(0.0);
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;
                for tile in 0..data.tiles {
                    let first = tile * TILE_KV;
                    let in_tile = TILE_KV.min(total - first);
                    let mut mask = [f32::NEG_INFINITY; TILE_KV];
                    let mut any_visible = false;
                    for (column, slot) in mask.iter_mut().take(in_tile).enumerate() {
                        let global = first + column;
                        let (range_index, key_row, key_position) = if global < self.cached_rows {
                            (0, global, global as i64 - self.cached_rows as i64)
                        } else {
                            (1, global - self.cached_rows, (global - self.cached_rows) as i64)
                        };
                        let _ = key_row;
                        let band = &self.score.bands[range_index];
                        let relative = key_position - query_row as i64;
                        if relative >= band.lower_inclusive && relative <= band.upper_inclusive {
                            *slot = 0.0;
                            any_visible = true;
                        }
                    }
                    if !any_visible {
                        continue;
                    }
                    let keys = &data.keys[(kv_head * data.tiles + tile) * head_dim * TILE_KV..][..head_dim * TILE_KV];
                    let mut scores = [0f32; TILE_KV];
                    score_row(kernels, query, keys, &mut scores);
                    for (score, visible) in scores.iter_mut().zip(&mask) {
                        *score *= self.score.scale;
                        *score += *visible;
                    }
                    let tile_max = scores.iter().fold(f32::NEG_INFINITY, |max, value| if *value > max { *value } else { max });
                    if tile_max == f32::NEG_INFINITY {
                        continue;
                    }
                    let new_max = if running_max > tile_max { running_max } else { tile_max };
                    if new_max > running_max {
                        let rescale = expf(running_max - new_max);
                        for value in accumulator.iter_mut() {
                            *value *= rescale;
                        }
                        running_sum *= rescale;
                    }
                    running_max = new_max;
                    running_sum = (f64::from(running_sum) + soft_max(&mut scores, new_max)) as f32;
                    let values = &data.values[(kv_head * data.padded_rows + first) * head_dim..][..TILE_KV * head_dim];
                    value_row(kernels, &scores, values, accumulator);
                }
                let inverse = if running_sum == 0.0 { 0.0 } else { 1.0 / running_sum };
                let output_start = query_head * head_dim;
                for (slot, value) in out[output_start..output_start + head_dim].iter_mut().zip(accumulator.iter()) {
                    *slot = *value * inverse;
                }
            }
        }
    }
}

#[cfg(all(test, feature = "std", target_arch = "x86_64"))]
mod expf_tests {
    use core::arch::x86_64::{
        __m256, __m256i, _CMP_GT_OQ, _CMP_LE_OQ, _mm256_add_epi32, _mm256_and_ps, _mm256_and_si256, _mm256_andnot_ps,
        _mm256_castps_si256, _mm256_castsi256_ps, _mm256_cmp_ps, _mm256_fmadd_ps, _mm256_fnmadd_ps, _mm256_loadu_ps,
        _mm256_movemask_ps, _mm256_mul_ps, _mm256_or_ps, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_slli_epi32, _mm256_storeu_ps, _mm256_sub_epi32, _mm256_sub_ps,
    };

    use super::{LN2_HIGH, LN2_LOW, LOG2_E, P0, P1, P2, P3, P4, ROUND, v_expf_lane};

    /// `ggml_v_expf` for AVX2, transcribed from `vec.h:1215-1252`.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn v_expf(x: __m256) -> __m256 {
        let r = _mm256_set1_ps(ROUND);
        let z = _mm256_fmadd_ps(x, _mm256_set1_ps(LOG2_E), r);
        let n = _mm256_sub_ps(z, r);
        let b = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_LOW), _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_HIGH), x));
        let e = _mm256_slli_epi32::<23>(_mm256_castps_si256(z));
        let k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1.0))));
        let c = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GT_OQ>(
            _mm256_andnot_ps(_mm256_set1_ps(-0.0), n),
            _mm256_set1_ps(126.0),
        ));
        let u = _mm256_mul_ps(b, b);
        let j = _mm256_fmadd_ps(
            _mm256_fmadd_ps(
                _mm256_fmadd_ps(_mm256_set1_ps(P0), b, _mm256_set1_ps(P1)),
                u,
                _mm256_fmadd_ps(_mm256_set1_ps(P2), b, _mm256_set1_ps(P3)),
            ),
            u,
            _mm256_mul_ps(_mm256_set1_ps(P4), b),
        );
        if _mm256_movemask_ps(_mm256_castsi256_ps(c)) == 0 {
            return _mm256_fmadd_ps(j, k, k);
        }
        let g = _mm256_and_si256(
            _mm256_castps_si256(_mm256_cmp_ps::<_CMP_LE_OQ>(n, _mm256_setzero_ps())),
            _mm256_set1_epi32(0x8200_0000u32 as i32),
        );
        let s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f00_0000)));
        let s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
        let d = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GT_OQ>(
            _mm256_andnot_ps(_mm256_set1_ps(-0.0), n),
            _mm256_set1_ps(192.0),
        ));
        let _: __m256i = d;
        _mm256_or_ps(
            _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
            _mm256_andnot_ps(
                _mm256_castsi256_ps(d),
                _mm256_or_ps(
                    _mm256_and_ps(_mm256_castsi256_ps(c), _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                    _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)),
                ),
            ),
        )
    }

    #[test]
    fn the_scalar_lane_is_the_avx2_ggml_v_expf_bit_for_bit() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        let mut state = 0x0123_4567_89ab_cdefu64;
        let mut inputs: alloc::vec::Vec<f32> = alloc::vec![0.0, -0.0, -1.0, -87.0, -88.5, -100.0, -103.9, -126.4, -200.0, f32::NEG_INFINITY, 5.0, 88.0, 89.0];
        for _ in 0..200_000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
            inputs.push(unit * -130.0);
        }
        inputs.resize(inputs.len().next_multiple_of(8), 0.0);
        for chunk in inputs.as_chunks::<8>().0 {
            let mut lanes = [0f32; 8];
            unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), v_expf(_mm256_loadu_ps(chunk.as_ptr()))) };
            for (input, lane) in chunk.iter().zip(lanes) {
                assert_eq!(lane.to_bits(), v_expf_lane(*input).to_bits(), "x = {input}");
            }
        }
    }
}
