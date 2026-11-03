//! llama.cpp's default CPU attention, `ggml_compute_forward_flash_attn_ext_f16_one_chunk`
//! (`ggml/src/ggml-cpu/ops.cpp:8614-8850`), for one bound cached-attention op.
//!
//! Every line that mirrors the incumbent cites the line it mirrors. The query, key and value are
//! rounded to f16, the score is `ggml_vec_dot_f16` (`ggml/src/ggml-cpu/vec.cpp:264-378`, AVX2
//! `simd-mappings.h:602-620`), the softmax is online with `expf`, the value accumulator is stored in
//! f16 after every scale and every multiply-add, and the running sum is f32.

use half::f16;

use super::{AttentionExtents, CachedAttentionRotary, CachedAttentionScore};

const DOT_STEP: usize = 32;
const DOT_LANES: usize = 8;
const DOT_ARRAYS: usize = 4;

fn to_half(value: f32) -> f32 {
    f16::from_f32(value).to_f32()
}

/// `ggml_vec_dot_f16` on the AVX2 build: four eight-lane fma accumulators stepping 32, the
/// `GGML_F32x8_REDUCE` tree (`simd-mappings.h:602-620`), then the leftovers in double (`vec.cpp:365-367`).
fn vec_dot_half(left: &[f32], right: &[f32]) -> f32 {
    let main = left.len() & !(DOT_STEP - 1);
    let mut sum = [[0f32; DOT_LANES]; DOT_ARRAYS];
    for step in (0..main).step_by(DOT_STEP) {
        for (array, lanes) in sum.iter_mut().enumerate() {
            for (lane, accumulator) in lanes.iter_mut().enumerate() {
                let index = step + array * DOT_LANES + lane;
                *accumulator = libm::fmaf(left[index], right[index], *accumulator);
            }
        }
    }
    let lanes: [f32; DOT_LANES] =
        core::array::from_fn(|lane| (sum[0][lane] + sum[2][lane]) + (sum[1][lane] + sum[3][lane]));
    let folded: [f32; 4] = core::array::from_fn(|lane| lanes[lane] + lanes[lane + 4]);
    let reduced = (folded[0] + folded[1]) + (folded[2] + folded[3]);
    let mut total = f64::from(reduced);
    for index in main..left.len() {
        total += f64::from(left[index] * right[index]);
    }
    total as f32
}

/// `ggml_vec_scale_f16` (`vec.h`): multiply in f32, store f16.
fn scale_half(accumulator: &mut [f32], factor: f32) {
    for value in accumulator {
        *value = to_half(*value * factor);
    }
}

/// `ggml_vec_mad_f16`: the first `n & ~31` lanes fuse, the leftover lanes multiply then add.
fn mad_half(accumulator: &mut [f32], row: &[f32], factor: f32) {
    let main = accumulator.len() & !(DOT_STEP - 1);
    for (index, value) in accumulator.iter_mut().enumerate() {
        let updated = if index < main {
            libm::fmaf(row[index], factor, *value)
        } else {
            *value + row[index] * factor
        };
        *value = to_half(updated);
    }
}

struct Range<'buffer> {
    keys: [&'buffer [f32]; 2],
    pass_keys: Option<&'buffer [f32]>,
    values: &'buffer [f32],
    rows: usize,
    is_cached: bool,
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
    let pair_dim = (rotary.rotary_dim / 2) as usize;
    let pass_dim = (extents.head_dim - rotary.rotary_dim) as usize;
    let head_dim = extents.head_dim as usize;
    let kv_heads = extents.kv_heads as usize;
    let groups = extents.query_groups as usize;
    let query_rows = extents.query_rows as usize;
    let cached_rows = extents.cached_key_rows as usize;
    let new_rows = extents.new_key_rows as usize;
    if head_dim == 0 || pair_dim == 0 || kv_heads == 0 || groups == 0 || rotary.rotary_dim > extents.head_dim
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
    let mut query_half = alloc::vec![0f32; head_dim];
    let mut key_half = alloc::vec![0f32; head_dim];
    let mut value_half = alloc::vec![0f32; head_dim];
    let mut accumulator = alloc::vec![0f32; head_dim];
    for query_row in 0..query_rows {
        for kv_head in 0..kv_heads {
            for group in 0..groups {
                let query_head = kv_head * groups + group;
                let query_start = (query_row * kv_heads * groups + query_head) * pair_dim;
                let pass_start = (query_row * kv_heads * groups + query_head) * pass_dim;
                for dimension in 0..pair_dim {
                    query_half[dimension] = to_half(queries[0][query_start + dimension]);
                    query_half[pair_dim + dimension] = to_half(queries[1][query_start + dimension]);
                }
                if let Some(plane) = rotary.pass.as_ref() {
                    for dimension in 0..pass_dim {
                        query_half[2 * pair_dim + dimension] = to_half(plane.query[pass_start + dimension]);
                    }
                }
                accumulator.fill(0.0);
                let mut running_sum = 0.0f32;
                let mut running_max = f32::NEG_INFINITY;
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
                        let key_start = (key_row * kv_heads + kv_head) * pair_dim;
                        for dimension in 0..pair_dim {
                            key_half[dimension] = to_half(range.keys[0][key_start + dimension]);
                            key_half[pair_dim + dimension] = to_half(range.keys[1][key_start + dimension]);
                        }
                        if let Some(pass_keys) = range.pass_keys {
                            let pass_key_start = (key_row * kv_heads + kv_head) * pass_dim;
                            for dimension in 0..pass_dim {
                                key_half[2 * pair_dim + dimension] = to_half(pass_keys[pass_key_start + dimension]);
                            }
                        }
                        let dot = vec_dot_half(&key_half, &query_half);
                        let weighted = dot * score.scale;
                        let previous_max = running_max;
                        let mut rescale = 1.0f32;
                        let mut weight = 1.0f32;
                        if weighted > running_max {
                            running_max = weighted;
                            rescale = libm::expf(previous_max - running_max);
                            scale_half(&mut accumulator, rescale);
                        } else {
                            weight = libm::expf(weighted - running_max);
                        }
                        let value_start = (key_row * kv_heads + kv_head) * head_dim;
                        for (slot, raw) in value_half.iter_mut().zip(&range.values[value_start..value_start + head_dim]) {
                            *slot = to_half(*raw);
                        }
                        mad_half(&mut accumulator, &value_half, weight);
                        running_sum = running_sum * rescale + weight;
                    }
                }
                let inverse = if running_sum == 0.0 { 0.0 } else { 1.0 / running_sum };
                let output_start = (query_row * kv_heads * groups + query_head) * head_dim;
                for (slot, value) in output[output_start..output_start + head_dim].iter_mut().zip(&accumulator) {
                    *slot = *value * inverse;
                }
            }
        }
    }
    true
}
