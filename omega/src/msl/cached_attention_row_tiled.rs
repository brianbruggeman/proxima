use super::*;

/// The split-KV partial for [`CachedAttentionForm::TwoRangeRowTiled`]: one
/// threadgroup per `(kv_head, row tile, split)` of `simdgroups` simdgroups,
/// for `K = query_rows = new_key_rows` rows, from a verify of two rows to a
/// whole prompt. Where
/// [`render_cached_attention_decode_split`](super::render_cached_attention_decode_split)
/// gives every `(query row, query head)` its own threadgroup and so re-reads
/// every K and V byte once per row, this walks the keys in `keys_per_block`
/// blocks and scores each block against a whole tile of query vectors.
///
/// Each simdgroup owns a slice of the K block's 8-key fragments for Q.K^T (even
/// and odd rotary plane as two accumulations into one score fragment), an online
/// softmax runs per query vector with the per-row window mask, and P.V
/// accumulates into 8x8 output fragments that each simdgroup keeps in registers
/// for the output columns it owns, rescaled in place by the softmax's per-row
/// factor (the lane-to-row map of an 8x8 fragment is the one MLX's steel
/// attention uses). A vector block is eight heads of one query row when the query
/// groups fill whole blocks, and eight consecutive rows of one head otherwise
/// (a query-group size of two), so every group size shares one kernel. The cached
/// range, then the new range, are walked with the same fragment code; the keys of
/// the new range past its last whole 8-key fragment run on a scalar tail, so no
/// fragment reads past the in-graph new keys. A block no row of the tile can
/// see, above its causal diagonal or below its window, is never visited, and the
/// heaviest tiles are dispatched first. `u.splits == 1` stores the normalized
/// rows into `out`; above that the unnormalized partial plus `(max, sum)` go to
/// the interleaved scratch layout [`super::render_cached_attention_merge`] reads
/// back, the same layout the decode split writes.
///
/// Composes `simdgroup_matrix` MMA (Metal), llama.cpp's non-vec flash attention
/// block structure (`kernel_flash_attn_ext`, `fa_common.metal`), the decode
/// split's scalar pass for the tail, and the shared merge. The row count, the
/// live cached rows and the split count are read at run time, so one compiled
/// kernel serves every K and every bucket.
pub(super) fn render_cached_attention_row_tiled(
    resolved: &BoundOp,
    entry: &str,
    rows_per_threadgroup: u64,
    simdgroups: u64,
) -> Result<String, EmitError> {
    let BoundOpKind::CachedAttention {
        kv_heads,
        query_groups,
        head_dim,
        scale,
        cached_lower_inclusive,
        new_upper_inclusive,
        ..
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "cached_attention_row_tiled",
            found: resolved.kind.name(),
        });
    };
    validate_attention_split_limit(resolved.node, crate::sized::ATTENTION_SPLIT_MAX)?;
    if resolved.dtype != DType::Float32 {
        return Err(EmitError::UnsupportedDType {
            node: resolved.node,
            dtype: resolved.dtype,
        });
    }
    let cached_lower = if *cached_lower_inclusive == i64::MIN {
        "-9223372036854775807L".to_string()
    } else {
        format!("{cached_lower_inclusive}L")
    };
    let substitutions = [
        ("@ENTRY@", entry.to_string()),
        ("@KV_HEADS@", kv_heads.to_string()),
        ("@QUERY_GROUPS@", query_groups.to_string()),
        ("@HEAD_DIM@", head_dim.to_string()),
        ("@SCALE@", msl_literal(*scale)),
        ("@CACHED_LOWER@", cached_lower),
        ("@NEW_UPPER@", format!("{new_upper_inclusive}L")),
        ("@TILE_ROWS@", rows_per_threadgroup.to_string()),
        ("@SIMDGROUPS@", simdgroups.to_string()),
        ("@BLOCK@", row_tiled_block(*head_dim).to_string()),
        (
            "@SPLIT_KEYS@",
            crate::sized::ATTENTION_ROWS_KEYS_PER_SPLIT.to_string(),
        ),
    ];
    let mut source = String::new();
    preamble(&mut source, None);
    let mut body = ROW_TILED_KERNEL.to_string();
    for (token, value) in &substitutions {
        body = body.replace(token, value);
    }
    source.push_str(&body);
    Ok(source)
}

const ROW_TILED_KERNEL: &str = r#"struct Uniforms { long total_elements; long splits; };

#define FOR_UNROLL _Pragma("clang loop unroll(full)")

kernel void @ENTRY@(device const float* in0 [[buffer(0)]], device const float* in1 [[buffer(1)]], device const float* in2 [[buffer(2)]], device const float* in3 [[buffer(3)]], device const float* in4 [[buffer(4)]], device const float* in5 [[buffer(5)]], device const float* in6 [[buffer(6)]], device const float* in7 [[buffer(7)]], device const float* in8 [[buffer(8)]], device float* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]], uint tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]], ushort simdgroup_slot [[simdgroup_index_in_threadgroup]]) {
    constexpr long kv_heads = @KV_HEADS@; constexpr long query_groups = @QUERY_GROUPS@; constexpr long head_dim = @HEAD_DIM@; constexpr long half_dim = head_dim / 2; constexpr float scale = @SCALE@; constexpr long cached_lower = @CACHED_LOWER@; constexpr long new_upper = @NEW_UPPER@;
    constexpr long tile_rows = @TILE_ROWS@; constexpr long simdgroups = @SIMDGROUPS@; constexpr long block = @BLOCK@; constexpr long split_keys = @SPLIT_KEYS@;
    constexpr bool rows_in_fragment = (query_groups % 8) != 0;
    constexpr long groups_per_row = rows_in_fragment ? 1 : query_groups / 8;
    constexpr long tile_blocks = rows_in_fragment ? (tile_rows / 8) * query_groups : (tile_rows * query_groups) / 8;
    constexpr long tile_vectors = tile_blocks * 8; constexpr long threads = simdgroups * 32;
    constexpr long dims_per_group = head_dim / 8 / simdgroups; constexpr long depth_unroll = ((half_dim / 8) % 2 == 0) ? 2 : 1;
    constexpr long key_tiles_per_group = (block / 8) / simdgroups;
    constexpr long query_stride = rows_in_fragment ? kv_heads * query_groups * half_dim : half_dim;
    long splits = u.splits;
    long total_rows = u.total_elements / (kv_heads * query_groups);
    long tiles = (total_rows + tile_rows - 1L) / tile_rows;
    if ((long)tgid >= kv_heads * tiles * splits) { return; }
    long split = (long)tgid % splits;
    long tile = tiles - 1L - ((long)tgid / splits) % tiles;
    long kv_head = (long)tgid / (splits * tiles);
    long row0 = tile * tile_rows;
    long rows_here = min(tile_rows, total_rows - row0);
    long live = (long)in8[0];
    long thread_id = (long)simdgroup_slot * 32L + (long)lane;
    short quad = (short)(lane / 4); short fragment_row = (short)((quad & 4) + ((lane / 2) % 4)); short fragment_column = (short)((quad & 2) * 2 + (lane % 2) * 2);
    threadgroup float score_tile[tile_vectors * block]; threadgroup float row_maximum[tile_vectors]; threadgroup float row_sum[tile_vectors]; threadgroup float rescale_tile[tile_vectors];
    threadgroup int vector_row[tile_vectors]; threadgroup int vector_head[tile_vectors]; threadgroup int vector_live[tile_vectors];
    for (long index = thread_id; index < tile_vectors; index += threads) {
        long block_index = index / 8L; long within = index % 8L;
        long row; long head; long owned;
        if (rows_in_fragment) {
            long row_block = block_index / query_groups;
            long owned_from = row0 + row_block * 8L;
            row = min(owned_from, total_rows - 8L) + within; head = block_index % query_groups;
            owned = (row >= owned_from && row < total_rows) ? 1L : 0L;
        } else {
            row = row0 + index / query_groups; head = index % query_groups;
            owned = (row < total_rows) ? 1L : 0L;
        }
        vector_row[index] = (int)row; vector_head[index] = (int)head; vector_live[index] = (int)owned;
        row_maximum[index] = -INFINITY; row_sum[index] = 0.0f; rescale_tile[index] = 0.0f;
    }
    long block_row[tile_blocks]; long block_head[tile_blocks];
    FOR_UNROLL for (long vector_block = 0L; vector_block < tile_blocks; vector_block++) {
        if (rows_in_fragment) {
            block_row[vector_block] = min(row0 + (vector_block / query_groups) * 8L, total_rows - 8L); block_head[vector_block] = vector_block % query_groups;
        } else {
            block_row[vector_block] = min(row0 + vector_block / groups_per_row, total_rows - 1L); block_head[vector_block] = (vector_block % groups_per_row) * 8L;
        }
    }
    simdgroup_float8x8 accumulated[dims_per_group][tile_blocks];
    FOR_UNROLL for (long slot = 0L; slot < dims_per_group; slot++) { FOR_UNROLL for (long vector_block = 0L; vector_block < tile_blocks; vector_block++) { accumulated[slot][vector_block] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); } }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    long last_row = row0 + rows_here - 1L;
    long first_key = max(0L, live + cached_lower + row0) & ~7L;
    long band = max(0L, live - first_key);
    long slice = ((((band + splits - 1L) / splits) + split_keys - 1L) / split_keys) * split_keys;
    long slice_start = first_key + split * slice;
    long slice_end = min(slice_start + slice, live);
    long cached_blocks = slice_start < slice_end ? (slice_end - slice_start + block - 1L) / block : 0L;
    long total_aligned = total_rows & ~7L;
    long new_first = max(0L, row0 + cached_lower);
    long new_end = min(total_rows, last_row + new_upper + 1L);
    long new_start = new_first & ~7L;
    long mma_end = min(new_end, total_aligned);
    bool last_split = split == splits - 1L;
    long new_blocks = (last_split && mma_end > new_start) ? (mma_end - new_start + block - 1L) / block : 0L;
    long tail_steps = (last_split && new_end > total_aligned && new_first < total_rows) ? 1L : 0L;
    for (long step = 0L; step < cached_blocks + new_blocks + tail_steps; step++) {
        long mode = step < cached_blocks ? 0L : (step < cached_blocks + new_blocks ? 1L : 2L);
        long key0 = mode == 0L ? slice_start + step * block : (mode == 1L ? new_start + (step - cached_blocks) * block : total_aligned);
        long columns = mode == 0L ? min(block, slice_end - key0) : (mode == 1L ? min(block, mma_end - key0) : total_rows - total_aligned);
        if (mode != 2L) {
            device const float* key_even = mode == 0L ? in2 : in4;
            device const float* key_odd = mode == 0L ? in3 : in5;
            int fragments = (int)((columns + 7L) / 8L);
            simdgroup_float8x8 scores[key_tiles_per_group][tile_blocks];
            FOR_UNROLL for (int group = 0; group < (int)key_tiles_per_group; group++) { FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) { scores[group][vector_block] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); } }
            for (int depth = 0; depth < (int)half_dim; depth += 8 * (int)depth_unroll) {
                simdgroup_float8x8 key_even_tile[key_tiles_per_group][depth_unroll]; simdgroup_float8x8 key_odd_tile[key_tiles_per_group][depth_unroll];
                FOR_UNROLL for (int group = 0; group < (int)key_tiles_per_group; group++) {
                    int key_tile = (int)simdgroup_slot + group * (int)simdgroups;
                    if (key_tile < fragments) {
                        device const float* key_even_ptr = key_even + (key0 + (long)key_tile * 8L) * (kv_heads * half_dim) + kv_head * half_dim;
                        device const float* key_odd_ptr = key_odd + (key0 + (long)key_tile * 8L) * (kv_heads * half_dim) + kv_head * half_dim;
                        FOR_UNROLL for (int step_index = 0; step_index < (int)depth_unroll; step_index++) {
                            simdgroup_load(key_even_tile[group][step_index], key_even_ptr + depth + 8 * step_index, (ulong)(kv_heads * half_dim), ulong2(0, 0), true);
                            simdgroup_load(key_odd_tile[group][step_index], key_odd_ptr + depth + 8 * step_index, (ulong)(kv_heads * half_dim), ulong2(0, 0), true);
                        }
                    }
                }
                FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) {
                    long query_offset = (block_row[vector_block] * (kv_heads * query_groups) + kv_head * query_groups + block_head[vector_block]) * half_dim + depth;
                    simdgroup_float8x8 query_even[depth_unroll]; simdgroup_float8x8 query_odd[depth_unroll];
                    FOR_UNROLL for (int step_index = 0; step_index < (int)depth_unroll; step_index++) {
                        simdgroup_load(query_even[step_index], in0 + query_offset + 8 * step_index, (ulong)query_stride);
                        simdgroup_load(query_odd[step_index], in1 + query_offset + 8 * step_index, (ulong)query_stride);
                    }
                    FOR_UNROLL for (int group = 0; group < (int)key_tiles_per_group; group++) {
                        if ((int)simdgroup_slot + group * (int)simdgroups < fragments) {
                            FOR_UNROLL for (int step_index = 0; step_index < (int)depth_unroll; step_index++) {
                                simdgroup_multiply_accumulate(scores[group][vector_block], query_even[step_index], key_even_tile[group][step_index], scores[group][vector_block]);
                                simdgroup_multiply_accumulate(scores[group][vector_block], query_odd[step_index], key_odd_tile[group][step_index], scores[group][vector_block]);
                            }
                        }
                    }
                }
            }
            FOR_UNROLL for (int group = 0; group < (int)key_tiles_per_group; group++) {
                int key_tile = (int)simdgroup_slot + group * (int)simdgroups;
                if (key_tile < fragments) {
                    FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) { simdgroup_store(scores[group][vector_block], score_tile + vector_block * 8 * (int)block + key_tile * 8, (ulong)block); }
                }
            }
        } else {
            long pairs = tile_vectors * columns;
            short ty = (short)(lane / 8); short tx = (short)(lane % 8);
            for (long group = (long)simdgroup_slot; group * 4L < pairs; group += simdgroups) {
                long pair = group * 4L + (long)ty;
                long vector = pair / columns;
                long column = pair % columns;
                long query_row = (long)vector_row[vector];
                long relative = key0 + column - query_row;
                bool valid = pair < pairs && vector_live[vector] != 0 && relative <= new_upper && relative >= cached_lower;
                float partial_score = 0.0f;
                if (valid) {
                    long query_index = query_row * (kv_heads * query_groups) + kv_head * query_groups + (long)vector_head[vector];
                    long key_offset = (key0 + column) * (kv_heads * half_dim) + kv_head * half_dim;
                    device const float4* query_even4 = (device const float4*)(in0 + query_index * half_dim);
                    device const float4* query_odd4 = (device const float4*)(in1 + query_index * half_dim);
                    device const float4* key_even4 = (device const float4*)(in4 + key_offset);
                    device const float4* key_odd4 = (device const float4*)(in5 + key_offset);
                    for (short index = tx; index < (short)(half_dim / 4L); index += 8) {
                        partial_score += dot(key_even4[index], query_even4[index]);
                        partial_score += dot(key_odd4[index], query_odd4[index]);
                    }
                }
                partial_score += simd_shuffle_down(partial_score, 4);
                partial_score += simd_shuffle_down(partial_score, 2);
                partial_score += simd_shuffle_down(partial_score, 1);
                if (tx == 0 && pair < pairs) { score_tile[vector * block + column] = valid ? partial_score * scale : -INFINITY; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (long vector = (long)simdgroup_slot; vector < tile_vectors; vector += simdgroups) {
            long query_row = (long)vector_row[vector];
            float local_scores[block / 32];
            float block_maximum = -INFINITY;
            FOR_UNROLL for (long item = 0L; item < block / 32L; item++) {
                long column = (long)lane + item * 32L;
                float raw_score = -INFINITY;
                if (item * 32L < columns) {
                    if (mode == 2L) {
                        if (column < columns) { raw_score = score_tile[vector * block + column]; }
                    } else if (mode == 1L) {
                        long relative = key0 + column - query_row;
                        if (column < columns && relative <= new_upper && relative >= cached_lower) { raw_score = score_tile[vector * block + column] * scale; }
                    } else {
                        long key = key0 + column;
                        if (key < slice_end && (key - live - query_row) >= cached_lower) { raw_score = score_tile[vector * block + column] * scale; }
                    }
                }
                local_scores[item] = raw_score;
                block_maximum = max(block_maximum, raw_score);
            }
            block_maximum = simd_max(block_maximum);
            float previous_maximum = row_maximum[vector];
            float next_maximum = max(previous_maximum, block_maximum);
            float rescale = (previous_maximum == -INFINITY) ? 0.0f : exp(previous_maximum - next_maximum);
            float block_sum = 0.0f;
            FOR_UNROLL for (long item = 0L; item < block / 32L; item++) {
                if (item * 32L < columns) {
                    float weight = (local_scores[item] == -INFINITY) ? 0.0f : exp(local_scores[item] - next_maximum);
                    score_tile[vector * block + (long)lane + item * 32L] = weight;
                    block_sum += weight;
                }
            }
            block_sum = simd_sum(block_sum);
            if (lane == 0) { row_maximum[vector] = next_maximum; row_sum[vector] = row_sum[vector] * rescale + block_sum; rescale_tile[vector] = rescale; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) {
            float row_scale = rescale_tile[vector_block * 8 + fragment_row];
            FOR_UNROLL for (int slot = 0; slot < (int)dims_per_group; slot++) {
                accumulated[slot][vector_block].thread_elements()[0] *= row_scale;
                accumulated[slot][vector_block].thread_elements()[1] *= row_scale;
            }
        }
        if (mode != 2L) {
            device const float* value_ptr = (mode == 0L ? in6 : in7) + key0 * (kv_heads * head_dim) + kv_head * head_dim;
            int fragments = (int)((columns + 7L) / 8L);
            for (int key_tile = 0; key_tile < fragments; key_tile++) {
                simdgroup_float8x8 weights[tile_blocks];
                FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) { simdgroup_load(weights[vector_block], score_tile + vector_block * 8 * (int)block + key_tile * 8, (ulong)block); }
                FOR_UNROLL for (int slot = 0; slot < (int)dims_per_group; slot++) {
                    int dimension_block = (int)simdgroup_slot + slot * (int)simdgroups;
                    simdgroup_float8x8 value;
                    simdgroup_load(value, value_ptr + (long)key_tile * 8L * (kv_heads * head_dim) + dimension_block * 8, (ulong)(kv_heads * head_dim));
                    FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) { simdgroup_multiply_accumulate(accumulated[slot][vector_block], weights[vector_block], value, accumulated[slot][vector_block]); }
                }
            }
        } else {
            FOR_UNROLL for (int slot = 0; slot < (int)dims_per_group; slot++) {
                long dimension = ((long)simdgroup_slot + (long)slot * simdgroups) * 8L + (long)fragment_column;
                FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) {
                    float sum_even = 0.0f; float sum_odd = 0.0f;
                    for (long column = 0L; column < columns; column++) {
                        float weight = score_tile[(vector_block * 8 + fragment_row) * block + column];
                        long value_offset = (key0 + column) * (kv_heads * head_dim) + kv_head * head_dim + dimension;
                        sum_even += weight * in7[value_offset];
                        sum_odd += weight * in7[value_offset + 1L];
                    }
                    accumulated[slot][vector_block].thread_elements()[0] += sum_even;
                    accumulated[slot][vector_block].thread_elements()[1] += sum_odd;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    FOR_UNROLL for (int slot = 0; slot < (int)dims_per_group; slot++) {
        long dimension = ((long)simdgroup_slot + (long)slot * simdgroups) * 8L + (long)fragment_column;
        FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) {
            long vector = (long)vector_block * 8L + (long)fragment_row;
            if (vector_live[vector] != 0) {
                long query_index = (long)vector_row[vector] * (kv_heads * query_groups) + kv_head * query_groups + (long)vector_head[vector];
                if (splits == 1L) {
                    float sum = row_sum[vector];
                    float inverse = sum == 0.0f ? 0.0f : 1.0f / sum;
                    out[query_index * head_dim + dimension] = accumulated[slot][vector_block].thread_elements()[0] * inverse;
                    out[query_index * head_dim + dimension + 1L] = accumulated[slot][vector_block].thread_elements()[1] * inverse;
                } else {
                    device float* attn_scratch = out;
                    long base = ((query_index * (head_dim / 4L) + (dimension >> 2)) * splits + split) * 4L + (dimension & 3L);
                    attn_scratch[base] = accumulated[slot][vector_block].thread_elements()[0];
                    attn_scratch[base + 1L] = accumulated[slot][vector_block].thread_elements()[1];
                }
            }
        }
    }
    if (splits > 1L) {
        device float* attn_scratch = out;
        for (long vector = thread_id; vector < tile_vectors; vector += threads) {
            if (vector_live[vector] != 0) {
                long query_index = (long)vector_row[vector] * (kv_heads * query_groups) + kv_head * query_groups + (long)vector_head[vector];
                long stats_index = u.total_elements * head_dim * splits + (query_index * splits + split) * 2L;
                attn_scratch[stats_index] = row_maximum[vector];
                attn_scratch[stats_index + 1L] = row_sum[vector];
            }
        }
    }
}
"#;
