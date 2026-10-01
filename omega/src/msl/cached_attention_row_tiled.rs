use super::*;

/// The split-KV partial for [`CachedAttentionForm::TwoRangeRowTiled`]: one
/// threadgroup per `(kv_head, row tile, split)` of `simdgroups` simdgroups,
/// for `K = query_rows = new_key_rows` rows. Where
/// [`render_cached_attention_decode_split`](super::render_cached_attention_decode_split)
/// gives every `(query row, query head)` its own threadgroup and so re-reads
/// every K and V byte once per row, this stages a 64-key block of the cache
/// once per threadgroup and scores it against a whole tile of query vectors.
///
/// The 8 rows of each `simdgroup_matrix` tile are the 8 query heads of one
/// query row (GQA-packed), so every tile has one causal and one window
/// boundary and its Q rows are contiguous in memory: Q is `simdgroup_load`ed
/// straight from device memory with no staging. Per block, each simdgroup
/// owns key columns of the Q.K^T product (even and odd rotary plane as two
/// accumulations into one key_tile), an online softmax runs per query vector
/// with the per-row window mask, and P.V accumulates into an f32 output tile
/// in threadgroup memory that each simdgroup owns the output columns of. The
/// new range is scalar, in the last split only: a key_tile over the in-graph
/// new keys would read past their K rows. `u.splits == 1` stores the
/// normalized rows into `out`; above that the unnormalized partial plus
/// `(max, sum)` go to the interleaved scratch layout
/// [`super::render_cached_attention_merge`] reads back, the same layout the
/// decode split writes.
///
/// Composes `simdgroup_matrix` MMA (Metal), llama.cpp's non-vec flash
/// attention block structure (`kernel_flash_attn_ext`, `fa.metal:471-700`),
/// the decode split's per-key scalar pass for the new range, and the shared
/// merge. The row count, the live cached rows and the split count are read at
/// run time, so one compiled kernel serves every K and every bucket.
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
        (
            "@BLOCK@",
            crate::sized::ATTENTION_ROWS_KEYS_PER_BLOCK.to_string(),
        ),
    ];
    let mut source = String::new();
    preamble(&mut source, false);
    let mut body = ROW_TILED_KERNEL.to_string();
    for (token, value) in &substitutions {
        body = body.replace(token, value);
    }
    source.push_str(&body);
    Ok(source)
}

const ROW_TILED_KERNEL: &str = r#"struct Uniforms { long total_elements; long splits; };

kernel void @ENTRY@(device const float* in0 [[buffer(0)]], device const float* in1 [[buffer(1)]], device const float* in2 [[buffer(2)]], device const float* in3 [[buffer(3)]], device const float* in4 [[buffer(4)]], device const float* in5 [[buffer(5)]], device const float* in6 [[buffer(6)]], device const float* in7 [[buffer(7)]], device const float* in8 [[buffer(8)]], device float* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]], uint tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]], ushort simdgroup_slot [[simdgroup_index_in_threadgroup]]) {
    constexpr long kv_heads = @KV_HEADS@; constexpr long query_groups = @QUERY_GROUPS@; constexpr long head_dim = @HEAD_DIM@; constexpr long half_dim = head_dim / 2; constexpr float scale = @SCALE@; constexpr long cached_lower = @CACHED_LOWER@; constexpr long new_upper = @NEW_UPPER@;
    constexpr long tile_rows = @TILE_ROWS@; constexpr long simdgroups = @SIMDGROUPS@; constexpr long block = @BLOCK@;
    constexpr long groups_per_row = query_groups / 8; constexpr long tile_vectors = tile_rows * query_groups; constexpr long tile_blocks = tile_vectors / 8; constexpr long threads = simdgroups * 32;
    long splits = u.splits;
    long total_rows = u.total_elements / (kv_heads * query_groups);
    long tiles = (total_rows + tile_rows - 1L) / tile_rows;
    if ((long)tgid >= kv_heads * tiles * splits) { return; }
    long split = (long)tgid % splits;
    long tile = ((long)tgid / splits) % tiles;
    long kv_head = (long)tgid / (splits * tiles);
    long row0 = tile * tile_rows;
    long rows_here = min(tile_rows, total_rows - row0);
    long vectors = rows_here * query_groups;
    long blocks_here = vectors / 8L;
    long live = (long)in8[0];
    long thread_id = (long)simdgroup_slot * 32L + (long)lane;
    threadgroup float output_tile[tile_vectors * head_dim]; threadgroup float score_tile[tile_vectors * block]; threadgroup float row_maximum[tile_vectors]; threadgroup float row_sum[tile_vectors];
    for (long index = thread_id; index < vectors * head_dim; index += threads) { output_tile[index] = 0.0f; }
    for (long index = thread_id; index < vectors; index += threads) { row_maximum[index] = -INFINITY; row_sum[index] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    long first_key = max(0L, live + cached_lower + row0) & ~7L;
    long band = max(0L, live - first_key);
    long slice = ((((band + splits - 1L) / splits) + block - 1L) / block) * block;
    long slice_start = first_key + split * slice;
    long slice_end = min(slice_start + slice, live);
    long cached_blocks = slice_start < slice_end ? (slice_end - slice_start + block - 1L) / block : 0L;
    long new_blocks = (split == splits - 1L) ? (total_rows + block - 1L) / block : 0L;
    for (long step = 0L; step < cached_blocks + new_blocks; step++) {
        bool new_block = step >= cached_blocks;
        long key0 = new_block ? (step - cached_blocks) * block : slice_start + step * block;
        long columns = new_block ? min(block, total_rows - key0) : min(block, slice_end - key0);
        if (!new_block) {
            long fragments = (columns + 7L) / 8L;
            for (long key_tile = (long)simdgroup_slot; key_tile < fragments; key_tile += simdgroups) {
                long key_offset = (key0 + key_tile * 8L) * (kv_heads * half_dim) + kv_head * half_dim;
                simdgroup_float8x8 scores[tile_blocks];
                for (long vector_block = 0L; vector_block < tile_blocks; vector_block++) { scores[vector_block] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); }
                for (long depth = 0L; depth < half_dim; depth += 8L) {
                    simdgroup_float8x8 key_even; simdgroup_float8x8 key_odd;
                    simdgroup_load(key_even, in2 + key_offset + depth, (ulong)(kv_heads * half_dim), ulong2(0, 0), true);
                    simdgroup_load(key_odd, in3 + key_offset + depth, (ulong)(kv_heads * half_dim), ulong2(0, 0), true);
                    for (long vector_block = 0L; vector_block < tile_blocks; vector_block++) {
                        if (vector_block < blocks_here) {
                            long query_offset = ((row0 + vector_block / groups_per_row) * (kv_heads * query_groups) + kv_head * query_groups + (vector_block % groups_per_row) * 8L) * half_dim + depth;
                            simdgroup_float8x8 query_even; simdgroup_float8x8 query_odd;
                            simdgroup_load(query_even, in0 + query_offset, (ulong)half_dim);
                            simdgroup_load(query_odd, in1 + query_offset, (ulong)half_dim);
                            simdgroup_multiply_accumulate(scores[vector_block], query_even, key_even, scores[vector_block]);
                            simdgroup_multiply_accumulate(scores[vector_block], query_odd, key_odd, scores[vector_block]);
                        }
                    }
                }
                for (long vector_block = 0L; vector_block < tile_blocks; vector_block++) {
                    if (vector_block < blocks_here) { simdgroup_store(scores[vector_block], score_tile + vector_block * 8L * block + key_tile * 8L, (ulong)block); }
                }
            }
        } else {
            long pairs = vectors * columns;
            short ty = (short)(lane / 8); short tx = (short)(lane % 8);
            for (long group = (long)simdgroup_slot; group * 4L < pairs; group += simdgroups) {
                long pair = group * 4L + (long)ty;
                long vector = pair / columns;
                long column = pair % columns;
                long query_row = row0 + vector / query_groups;
                bool valid = pair < pairs && (key0 + column - query_row) <= new_upper;
                float partial_score = 0.0f;
                if (valid) {
                    long query_index = query_row * (kv_heads * query_groups) + kv_head * query_groups + vector % query_groups;
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
        for (long vector = (long)simdgroup_slot; vector < vectors; vector += simdgroups) {
            long query_row = row0 + vector / query_groups;
            float block_maximum = -INFINITY;
            for (long column = (long)lane; column < block; column += 32L) {
                float raw_score = -INFINITY;
                if (new_block) {
                    if (column < columns) { raw_score = score_tile[vector * block + column]; }
                } else {
                    long key = key0 + column;
                    if (key < slice_end && (key - live - query_row) >= cached_lower) { raw_score = score_tile[vector * block + column] * scale; }
                }
                score_tile[vector * block + column] = raw_score;
                block_maximum = max(block_maximum, raw_score);
            }
            block_maximum = simd_max(block_maximum);
            float previous_maximum = row_maximum[vector];
            float next_maximum = max(previous_maximum, block_maximum);
            float rescale = (previous_maximum == -INFINITY) ? 0.0f : exp(previous_maximum - next_maximum);
            float block_sum = 0.0f;
            for (long column = (long)lane; column < block; column += 32L) {
                float raw_score = score_tile[vector * block + column];
                float weight = (raw_score == -INFINITY) ? 0.0f : exp(raw_score - next_maximum);
                score_tile[vector * block + column] = weight;
                block_sum += weight;
            }
            block_sum = simd_sum(block_sum);
            if (lane == 0) { row_maximum[vector] = next_maximum; row_sum[vector] = row_sum[vector] * rescale + block_sum; }
            for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) { output_tile[vector * head_dim + dimension] *= rescale; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!new_block) {
            long fragments = (columns + 7L) / 8L;
            for (long dimension_block = (long)simdgroup_slot; dimension_block < head_dim / 8L; dimension_block += simdgroups) {
                simdgroup_float8x8 accumulated[tile_blocks];
                for (long vector_block = 0L; vector_block < tile_blocks; vector_block++) {
                    if (vector_block < blocks_here) { simdgroup_load(accumulated[vector_block], output_tile + vector_block * 8L * head_dim + dimension_block * 8L, (ulong)head_dim); }
                }
                for (long key_tile = 0L; key_tile < fragments; key_tile++) {
                    simdgroup_float8x8 value;
                    simdgroup_load(value, in6 + (key0 + key_tile * 8L) * (kv_heads * head_dim) + kv_head * head_dim + dimension_block * 8L, (ulong)(kv_heads * head_dim));
                    for (long vector_block = 0L; vector_block < tile_blocks; vector_block++) {
                        if (vector_block < blocks_here) {
                            simdgroup_float8x8 weights;
                            simdgroup_load(weights, score_tile + vector_block * 8L * block + key_tile * 8L, (ulong)block);
                            simdgroup_multiply_accumulate(accumulated[vector_block], weights, value, accumulated[vector_block]);
                        }
                    }
                }
                for (long vector_block = 0L; vector_block < tile_blocks; vector_block++) {
                    if (vector_block < blocks_here) { simdgroup_store(accumulated[vector_block], output_tile + vector_block * 8L * head_dim + dimension_block * 8L, (ulong)head_dim); }
                }
            }
        } else {
            for (long dimension = thread_id; dimension < head_dim; dimension += threads) {
                float accumulated[tile_vectors];
                for (long vector = 0L; vector < tile_vectors; vector++) { accumulated[vector] = 0.0f; }
                for (long column = 0L; column < columns; column++) {
                    float value = in7[(key0 + column) * (kv_heads * head_dim) + kv_head * head_dim + dimension];
                    for (long vector = 0L; vector < tile_vectors; vector++) {
                        if (vector < vectors) { accumulated[vector] += score_tile[vector * block + column] * value; }
                    }
                }
                for (long vector = 0L; vector < tile_vectors; vector++) {
                    if (vector < vectors) { output_tile[vector * head_dim + dimension] += accumulated[vector]; }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (long index = thread_id; index < vectors * head_dim; index += threads) {
        long vector = index / head_dim;
        long dimension = index % head_dim;
        long query_index = (row0 + vector / query_groups) * (kv_heads * query_groups) + kv_head * query_groups + vector % query_groups;
        if (splits == 1L) {
            float sum = row_sum[vector];
            out[query_index * head_dim + dimension] = (sum == 0.0f ? 0.0f : output_tile[index] / sum);
        } else {
            device float* attn_scratch = out;
            attn_scratch[((query_index * (head_dim / 4L) + (dimension >> 2)) * splits + split) * 4L + (dimension & 3L)] = output_tile[index];
        }
    }
    if (splits > 1L) {
        device float* attn_scratch = out;
        for (long vector = thread_id; vector < vectors; vector += threads) {
            long query_index = (row0 + vector / query_groups) * (kv_heads * query_groups) + kv_head * query_groups + vector % query_groups;
            long stats_index = u.total_elements * head_dim * splits + (query_index * splits + split) * 2L;
            attn_scratch[stats_index] = row_maximum[vector];
            attn_scratch[stats_index + 1L] = row_sum[vector];
        }
    }
}
"#;
