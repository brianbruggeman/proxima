use super::signature_tokens_prelude::{QUERY_STAGE_PAD, query_tile_staged};
use super::*;

const BF16_CACHE_HELPER: &str = r#"
inline float omega_bf16_to_float(ushort value) { return as_type<float>((uint)value << 16); }
inline void omega_bf16_load_matrix(thread simdgroup_float8x8& destination, device const ushort* source, ulong stride, ushort lane, bool transposed) {
    short quad = (short)(lane / 4); short row = (short)((quad & 4) + ((lane / 2) % 4)); short column = (short)((quad & 2) * 2 + (lane % 2) * 2);
    if (transposed) {
        destination.thread_elements()[0] = omega_bf16_to_float(source[(long)(column) * stride + row]);
        destination.thread_elements()[1] = omega_bf16_to_float(source[(long)(column + 1) * stride + row]);
    } else {
        destination.thread_elements()[0] = omega_bf16_to_float(source[(long)row * stride + column]);
        destination.thread_elements()[1] = omega_bf16_to_float(source[(long)row * stride + column + 1]);
    }
}
inline void omega_bf8_load_matrix(thread simdgroup_float8x8& destination, device const uchar* source, ulong stride, ushort lane, bool transposed) {
    short quad = (short)(lane / 4); short row = (short)((quad & 4) + ((lane / 2) % 4)); short column = (short)((quad & 2) * 2 + (lane % 2) * 2);
    if (transposed) {
        destination.thread_elements()[0] = omega_bf8_to_float(source[(long)(column) * stride + row]);
        destination.thread_elements()[1] = omega_bf8_to_float(source[(long)(column + 1) * stride + row]);
    } else {
        destination.thread_elements()[0] = omega_bf8_to_float(source[(long)row * stride + column]);
        destination.thread_elements()[1] = omega_bf8_to_float(source[(long)row * stride + column + 1]);
    }
}
inline void omega_zero_padded_value_rows(thread simdgroup_float8x8& value, long key_start, long valid_end, ushort lane) {
    short quad = (short)(lane / 4); short row = (short)((quad & 4) + ((lane / 2) % 4));
    if (key_start + row >= valid_end) {
        value.thread_elements()[0] = 0.0f;
        value.thread_elements()[1] = 0.0f;
    }
}
"#;

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
///
/// The operand precision of the two matrix multiplies is `[attention_rows].mma_precision`:
/// `float` multiplies `simdgroup_float8x8` fragments; `half` narrows each fragment to
/// `simdgroup_half8x8` as it is loaded (the `simdgroup_load` overloads in
/// [`HALF_OPERAND_HELPERS`]) and still accumulates into `simdgroup_float8x8`. Cached
/// K/V storage is independent: F32 loads directly, while BF16 and BF8 widen into
/// float fragments before the selected MMA operand conversion. New-range buffers,
/// softmax and output accumulation remain F32. This is not llama.cpp's F16 K/V path.
pub(super) fn render_cached_attention_row_tiled(
    resolved: &BoundOp,
    entry: &str,
    rows_per_threadgroup: u64,
    simdgroups: u64,
    cached_kv_codec: Option<Codec>,
    mma_selection: AttentionMmaSelection,
    row_schedule: AttentionRowSchedule,
) -> Result<String, EmitError> {
    let half_operands = match mma_selection {
        AttentionMmaSelection::Legacy => crate::sized::ATTENTION_ROWS_MMA_HALF,
        #[cfg(feature = "metal-attn-variants")]
        AttentionMmaSelection::F32 => false,
        #[cfg(feature = "metal-attn-variants")]
        AttentionMmaSelection::F16 => true,
    };
    render_cached_attention_row_tiled_with(
        resolved,
        entry,
        rows_per_threadgroup,
        simdgroups,
        half_operands,
        cached_kv_codec,
        row_schedule,
    )
}

pub(super) fn render_cached_attention_row_tiled_with(
    resolved: &BoundOp,
    entry: &str,
    rows_per_threadgroup: u64,
    simdgroups: u64,
    half_operands: bool,
    cached_kv_codec: Option<Codec>,
    row_schedule: AttentionRowSchedule,
) -> Result<String, EmitError> {
    let BoundOpKind::CachedAttention {
        query_rows,
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
    let stages_query = query_tile_staged(
        rows_per_threadgroup,
        *query_groups,
        *head_dim,
        row_tiled_block(*head_dim),
        crate::sized::CACHED_ATTENTION_THREADGROUP_MEMORY_BYTES,
        crate::sized::ATTENTION_ROWS_MAX_STAGED_QUERY_BYTES,
    );
    #[cfg(feature = "metal-attn-variants")]
    let shared_k = row_schedule.has_shared_k();
    #[cfg(feature = "metal-attn-variants")]
    let shared_v = row_schedule.is_shared_kv();
    #[cfg(feature = "metal-attn-variants")]
    let query_parallel_rows = row_schedule.is_simdgroup_rows();
    #[cfg(feature = "metal-attn-variants")]
    let prefetch_next_block = row_schedule.is_prefetch_next_block();
    #[cfg(feature = "metal-attn-variants")]
    let simd_per_head = row_schedule.is_per_head_topology();
    #[cfg(not(feature = "metal-attn-variants"))]
    let shared_k = false;
    #[cfg(not(feature = "metal-attn-variants"))]
    let shared_v = false;
    #[cfg(not(feature = "metal-attn-variants"))]
    let query_parallel_rows = false;
    #[cfg(not(feature = "metal-attn-variants"))]
    let prefetch_next_block = false;
    #[cfg(not(feature = "metal-attn-variants"))]
    let simd_per_head = false;
    #[cfg(not(feature = "metal-attn-variants"))]
    let _ = row_schedule;
    let block = row_tiled_block(*head_dim);
    let half_dim = *head_dim / 2;
    let depth_unroll = if (half_dim / 8).is_multiple_of(2) { 2 } else { 1 };
    let query_blocks = if !(*query_groups).is_multiple_of(8) {
        (rows_per_threadgroup / 8) * *query_groups
    } else {
        (rows_per_threadgroup * *query_groups) / 8
    };
    let query_owner_rows = shared_v || query_parallel_rows || prefetch_next_block;
    let score_vectors = if shared_k || query_owner_rows {
        query_blocks.div_ceil(simdgroups)
    } else {
        query_blocks
    };
    let query_owner_accumulator_fragments = (*head_dim / 8) * score_vectors;
    if query_owner_rows
        && query_owner_accumulator_fragments > crate::sized::ATTENTION_ROWS_ACCUMULATOR_FRAGMENTS
    {
        if query_parallel_rows {
            return Err(EmitError::CachedAttentionQueryParallelismNotSupported {
                node: resolved.node,
                query_rows: *query_rows,
                reason: "simdgroup row ownership exceeds the configured accumulator-fragment budget",
            });
        }
        return Err(EmitError::CachedAttentionKvReuseNotSupported {
            node: resolved.node,
            reason: "shared K/V query ownership exceeds the configured accumulator-fragment budget",
        });
    }
    let shared_k_bytes = if shared_k {
        2 * block * 8 * depth_unroll * if half_operands { 2 } else { 4 }
    } else {
        0
    };
    let operand_bytes = if half_operands { 2 } else { 4 };
    let prefetch_bytes = if prefetch_next_block {
        2 * block * *head_dim * operand_bytes
    } else {
        0
    };
    let base_threadgroup_bytes = row_tile_bytes(rows_per_threadgroup, *query_groups, block)
        + if stages_query {
            query_stage_bytes(rows_per_threadgroup, *query_groups, *head_dim)
        } else {
            0
        };
    let shared_kv_bytes = base_threadgroup_bytes + shared_k_bytes;
    let value_fragment_bytes = 64 * if half_operands { 2 } else { 4 };
    let shared_v_dimension_blocks = if shared_v {
        let available = crate::sized::CACHED_ATTENTION_THREADGROUP_MEMORY_BYTES
            .saturating_sub(shared_kv_bytes);
        (available / value_fragment_bytes).min(*head_dim / 8)
    } else {
        1
    };
    let shared_v_bytes = if shared_v {
        shared_v_dimension_blocks * value_fragment_bytes
    } else {
        0
    };
    let threadgroup_bytes = shared_kv_bytes + shared_v_bytes + prefetch_bytes;
    if prefetch_next_block
        && threadgroup_bytes > crate::sized::CACHED_ATTENTION_THREADGROUP_MEMORY_BYTES
    {
        return Err(EmitError::CachedAttentionPrefetchNotSupported {
            node: resolved.node,
            required_bytes: threadgroup_bytes,
            available_bytes: crate::sized::CACHED_ATTENTION_THREADGROUP_MEMORY_BYTES,
            reason: "double-buffered K/V staging exceeds the configured threadgroup-memory budget",
        });
    }
    if shared_v && shared_v_dimension_blocks == 0 {
        return Err(EmitError::CachedAttentionKvReuseNotSupported {
            node: resolved.node,
            reason: "shared K/V staging exceeds the configured threadgroup-memory budget",
        });
    }
    if threadgroup_bytes > crate::sized::CACHED_ATTENTION_THREADGROUP_MEMORY_BYTES {
        return Err(EmitError::CachedAttentionKvReuseNotSupported {
            node: resolved.node,
            reason: "shared K staging exceeds the configured threadgroup-memory budget",
        });
    }
    let cached_type = match cached_kv_codec {
        None => "float",
        Some(Codec::BFloat16) => "ushort",
        Some(Codec::BFloat8) => "uchar",
        Some(_) => {
            return Err(EmitError::CachedAttentionKvCodecNotSupported {
                node: resolved.node,
                reason: "row-tiled attention reads cached K/V as F32, BFloat16, or BFloat8",
            });
        }
    };
    let substitutions = [
        ("@KV_TYPE@", cached_type.to_string()),
        (
            "@NARROW_TO_OPERAND_EVEN@",
            if half_operands {
                "narrow_fragment(even_float)"
            } else {
                "even_float"
            }
            .to_string(),
        ),
        (
            "@NARROW_TO_OPERAND_ODD@",
            if half_operands {
                "narrow_fragment(odd_float)"
            } else {
                "odd_float"
            }
            .to_string(),
        ),
        (
            "@NARROW_TO_OPERAND_VALUE@",
            if half_operands {
                "narrow_fragment(value_float)"
            } else {
                "value_float"
            }
            .to_string(),
        ),
        (
            "@BF16_CACHE@",
            (cached_kv_codec == Some(Codec::BFloat16)).to_string(),
        ),
        (
            "@BF8_CACHE@",
            (cached_kv_codec == Some(Codec::BFloat8)).to_string(),
        ),
        (
            "@MMA_HELPERS@",
            if half_operands {
                HALF_OPERAND_HELPERS
            } else {
                ""
            }
            .to_string(),
        ),
        (
            "@OPERAND@",
            if half_operands {
                "simdgroup_half8x8"
            } else {
                "simdgroup_float8x8"
            }
            .to_string(),
        ),
        (
            "@OPERAND_SCALAR@",
            if half_operands { "half" } else { "float" }.to_string(),
        ),
        ("@KV_REUSE_SHARED_K@", shared_k.to_string()),
        ("@KV_REUSE_SHARED_V@", shared_v.to_string()),
        ("@QUERY_PARALLEL_ROWS@", query_parallel_rows.to_string()),
        ("@QUERY_OWNER_ROWS@", query_owner_rows.to_string()),
        ("@PREFETCH_NEXT_BLOCK@", prefetch_next_block.to_string()),
        ("@SIMD_TOPOLOGY_PER_HEAD@", simd_per_head.to_string()),
        (
            "@PREFETCH_KEY_ELEMENTS@",
            if prefetch_next_block {
                (block * half_dim).to_string()
            } else {
                "1".to_string()
            },
        ),
        (
            "@PREFETCH_VALUE_ELEMENTS@",
            if prefetch_next_block {
                (block * *head_dim).to_string()
            } else {
                "1".to_string()
            },
        ),
        (
            "@SHARED_K_LOAD@",
            if half_operands {
                "omega_load_shared_half"
            } else {
                "omega_load_shared_float"
            }
            .to_string(),
        ),
        (
            "@SHARED_K_ELEMENTS@",
            if shared_k {
                (block * 8 * depth_unroll).to_string()
            } else {
                "1".to_string()
            },
        ),
        (
            "@SHARED_V_LOAD@",
            if half_operands {
                "omega_load_shared_half"
            } else {
                "omega_load_shared_float"
            }
            .to_string(),
        ),
        (
            "@PREFETCH_LOAD@",
            if half_operands {
                "omega_load_prefetched_half"
            } else {
                "omega_load_shared_float"
            }
            .to_string(),
        ),
        (
            "@SHARED_V_ELEMENTS@",
            (64 * shared_v_dimension_blocks).to_string(),
        ),
        (
            "@SHARED_V_DIMENSION_BLOCKS@",
            shared_v_dimension_blocks.to_string(),
        ),
        ("@ENTRY@", entry.to_string()),
        ("@KV_HEADS@", kv_heads.to_string()),
        ("@QUERY_GROUPS@", query_groups.to_string()),
        ("@HEAD_DIM@", head_dim.to_string()),
        ("@SCALE@", msl_literal(*scale)),
        ("@CACHED_LOWER@", cached_lower),
        ("@NEW_UPPER@", format!("{new_upper_inclusive}L")),
        ("@TILE_ROWS@", rows_per_threadgroup.to_string()),
        ("@STAGE_QUERY@", stages_query.to_string()),
        ("@QUERY_STAGE_PAD@", QUERY_STAGE_PAD.to_string()),
        ("@SIMDGROUPS@", simdgroups.to_string()),
        ("@BLOCK@", block.to_string()),
        (
            "@SPLIT_KEYS@",
            crate::sized::ATTENTION_ROWS_KEYS_PER_SPLIT.to_string(),
        ),
    ];
    let mut source = String::new();
    preamble(&mut source, None);
    source.push_str(super::cached_attention_render::BF8_VECTOR_HELPER);
    source.push_str(BF16_CACHE_HELPER);
    if !half_operands {
        source.push_str(SHARED_K_FLOAT_HELPER);
    }
    let mut body = ROW_TILED_KERNEL.to_string();
    for (token, value) in &substitutions {
        body = body.replace(token, value);
    }
    source.push_str(&body);
    Ok(source)
}

pub(super) const HALF_OPERAND_HELPERS: &str = r#"
inline simdgroup_half8x8 narrow_fragment(simdgroup_float8x8 wide) { simdgroup_half8x8 narrowed; narrowed.thread_elements()[0] = half(wide.thread_elements()[0]); narrowed.thread_elements()[1] = half(wide.thread_elements()[1]); return narrowed; }
inline void simdgroup_load(thread simdgroup_half8x8& destination, device const float* source, ulong stride, ulong2 origin, bool transposed) { simdgroup_float8x8 loaded; simdgroup_load(loaded, source, stride, origin, transposed); destination = narrow_fragment(loaded); }
inline void simdgroup_load(thread simdgroup_half8x8& destination, device const float* source, ulong stride) { simdgroup_float8x8 loaded; simdgroup_load(loaded, source, stride); destination = narrow_fragment(loaded); }
inline void simdgroup_load(thread simdgroup_half8x8& destination, threadgroup const float* source, ulong stride) { simdgroup_float8x8 loaded; simdgroup_load(loaded, source, stride); destination = narrow_fragment(loaded); }
inline void omega_load_shared_half(thread simdgroup_half8x8& destination, threadgroup const half* source, ulong stride, ushort lane) { short quad = (short)(lane / 4); short row = (short)((quad & 4) + ((lane / 2) % 4)); short column = (short)((quad & 2) * 2 + (lane % 2) * 2); destination.thread_elements()[0] = source[(long)row * stride + column]; destination.thread_elements()[1] = source[(long)row * stride + column + 1]; }
inline void omega_load_prefetched_half(thread simdgroup_float8x8& destination, threadgroup const half* source, ulong stride, ushort lane) { short quad = (short)(lane / 4); short row = (short)((quad & 4) + ((lane / 2) % 4)); short column = (short)((quad & 2) * 2 + (lane % 2) * 2); destination.thread_elements()[0] = float(source[(long)row * stride + column]); destination.thread_elements()[1] = float(source[(long)row * stride + column + 1]); }
"#;

pub(super) const SHARED_K_FLOAT_HELPER: &str = r#"
inline void omega_load_shared_float(thread simdgroup_float8x8& destination, threadgroup const float* source, ulong stride, ushort lane) { (void)lane; simdgroup_load(destination, source, stride); }
"#;

const ROW_TILED_KERNEL: &str = r#"struct Uniforms { long total_elements; long splits; };@MMA_HELPERS@

#define FOR_UNROLL _Pragma("clang loop unroll(full)")

kernel void @ENTRY@(device const float* in0 [[buffer(0)]], device const float* in1 [[buffer(1)]], device const @KV_TYPE@* in2 [[buffer(2)]], device const @KV_TYPE@* in3 [[buffer(3)]], device const float* in4 [[buffer(4)]], device const float* in5 [[buffer(5)]], device const @KV_TYPE@* in6 [[buffer(6)]], device const float* in7 [[buffer(7)]], device const float* in8 [[buffer(8)]], device float* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]], uint tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]], ushort simdgroup_slot [[simdgroup_index_in_threadgroup]]) {
    constexpr long kv_heads = @KV_HEADS@; constexpr long query_groups = @QUERY_GROUPS@; constexpr long head_dim = @HEAD_DIM@; constexpr long half_dim = head_dim / 2; constexpr float scale = @SCALE@; constexpr long cached_lower = @CACHED_LOWER@; constexpr long new_upper = @NEW_UPPER@;
    constexpr long tile_rows = @TILE_ROWS@; constexpr long simdgroups = @SIMDGROUPS@; constexpr long block = @BLOCK@; constexpr long split_keys = @SPLIT_KEYS@;
    constexpr bool rows_in_fragment = (query_groups % 8) != 0;
    constexpr long groups_per_row = rows_in_fragment ? 1 : query_groups / 8;
    constexpr long tile_blocks = rows_in_fragment ? (tile_rows / 8) * query_groups : (tile_rows * query_groups) / 8;
    constexpr long tile_vectors = tile_blocks * 8; constexpr long threads = simdgroups * 32;
    constexpr long dims_per_group = head_dim / 8 / simdgroups; constexpr long depth_unroll = ((half_dim / 8) % 2 == 0) ? 2 : 1;
    constexpr bool shared_k = @KV_REUSE_SHARED_K@; constexpr bool shared_v = @KV_REUSE_SHARED_V@; constexpr bool query_parallel_rows = @QUERY_PARALLEL_ROWS@; constexpr bool prefetch_next_block = @PREFETCH_NEXT_BLOCK@; constexpr bool simd_per_head = @SIMD_TOPOLOGY_PER_HEAD@; constexpr bool query_owner_rows = @QUERY_OWNER_ROWS@;
    constexpr long key_tiles_per_group = (block / 8) / simdgroups;
    constexpr long score_key_tiles = (shared_k || query_owner_rows) ? (block / 8) : key_tiles_per_group;
    constexpr long key_tiles_per_simdgroup = (!shared_k && query_owner_rows) ? (block / 8) : key_tiles_per_group;
    constexpr long score_vectors_per_simdgroup = query_owner_rows ? (tile_blocks + simdgroups - 1) / simdgroups : tile_blocks;
    constexpr long accumulator_dimensions = query_owner_rows ? (head_dim / 8) : dims_per_group;
    constexpr long accumulator_vectors = query_owner_rows ? score_vectors_per_simdgroup : tile_blocks;
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
    threadgroup @OPERAND_SCALAR@ shared_key_even[@SHARED_K_ELEMENTS@]; threadgroup @OPERAND_SCALAR@ shared_key_odd[@SHARED_K_ELEMENTS@];
    threadgroup @OPERAND_SCALAR@ shared_value[@SHARED_V_ELEMENTS@];
    threadgroup @OPERAND_SCALAR@ prefetched_key_even[@PREFETCH_KEY_ELEMENTS@];
    threadgroup @OPERAND_SCALAR@ prefetched_key_odd[@PREFETCH_KEY_ELEMENTS@];
    threadgroup @OPERAND_SCALAR@ prefetched_value[@PREFETCH_VALUE_ELEMENTS@];
    constexpr long shared_v_dimension_blocks = @SHARED_V_DIMENSION_BLOCKS@;
    constexpr bool stage_query = @STAGE_QUERY@; constexpr long query_stage_stride = half_dim + @QUERY_STAGE_PAD@L;
    threadgroup float query_stage[stage_query ? tile_blocks * 2L * 8L * query_stage_stride : 1L];
    for (long index = thread_id; index < tile_vectors; index += threads) {
        long block_index = index / 8L; long within = index % 8L;
        long row; long head; long owned;
        if (rows_in_fragment) {
            long row_block = simd_per_head ? block_index % (tile_rows / 8L) : block_index / query_groups;
            long owned_from = row0 + row_block * 8L;
            row = min(owned_from, total_rows - 8L) + within;
            head = simd_per_head ? block_index / (tile_rows / 8L) : block_index % query_groups;
            owned = (row >= owned_from && row < total_rows) ? 1L : 0L;
        } else {
            row = simd_per_head ? row0 + (index / 8L) % tile_rows : row0 + index / query_groups;
            head = simd_per_head ? (index / (tile_rows * 8L)) * 8L + index % 8L : index % query_groups;
            owned = (row < total_rows) ? 1L : 0L;
        }
        vector_row[index] = (int)row; vector_head[index] = (int)head; vector_live[index] = (int)owned;
        row_maximum[index] = -INFINITY; row_sum[index] = 0.0f; rescale_tile[index] = 0.0f;
    }
    long block_row[tile_blocks]; long block_head[tile_blocks];
    FOR_UNROLL for (long vector_block = 0L; vector_block < tile_blocks; vector_block++) {
        if (rows_in_fragment) {
            long row_block = simd_per_head ? vector_block % (tile_rows / 8L) : vector_block / query_groups;
            block_row[vector_block] = min(row0 + row_block * 8L, total_rows - 8L);
            block_head[vector_block] = simd_per_head ? vector_block / (tile_rows / 8L) : vector_block % query_groups;
        } else {
            block_row[vector_block] = simd_per_head ? min(row0 + vector_block % tile_rows, total_rows - 1L) : min(row0 + vector_block / groups_per_row, total_rows - 1L);
            block_head[vector_block] = simd_per_head ? (vector_block / tile_rows) * 8L : (vector_block % groups_per_row) * 8L;
        }
    }
    simdgroup_float8x8 accumulated[accumulator_dimensions][accumulator_vectors];
    FOR_UNROLL for (long slot = 0L; slot < accumulator_dimensions; slot++) { FOR_UNROLL for (long vector_block = 0L; vector_block < accumulator_vectors; vector_block++) { accumulated[slot][vector_block] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); } }
    if (stage_query) {
        FOR_UNROLL for (long vector_block = 0L; vector_block < tile_blocks; vector_block++) {
            long query_base = (block_row[vector_block] * (kv_heads * query_groups) + kv_head * query_groups + block_head[vector_block]) * half_dim;
            for (long index = thread_id; index < 8L * half_dim; index += threads) {
                long stage_row = index / half_dim; long stage_column = index % half_dim;
                long source = query_base + stage_row * query_stride + stage_column;
                query_stage[(vector_block * 16L + stage_row) * query_stage_stride + stage_column] = in0[source];
                query_stage[(vector_block * 16L + 8L + stage_row) * query_stage_stride + stage_column] = in1[source];
            }
        }
    }
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
            device const float* key_even = mode == 0L ? (device const float*)in2 : in4;
            device const float* key_odd = mode == 0L ? (device const float*)in3 : in5;
            int fragments = (int)((columns + 7L) / 8L);
            simdgroup_float8x8 scores[score_key_tiles][score_vectors_per_simdgroup];
            FOR_UNROLL for (int key_slot = 0; key_slot < (int)score_key_tiles; key_slot++) { FOR_UNROLL for (int vector_slot = 0; vector_slot < (int)score_vectors_per_simdgroup; vector_slot++) { scores[key_slot][vector_slot] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); } }
            for (int depth = 0; depth < (int)half_dim; depth += 8 * (int)depth_unroll) {
                @OPERAND@ key_even_tile[key_tiles_per_simdgroup][depth_unroll]; @OPERAND@ key_odd_tile[key_tiles_per_simdgroup][depth_unroll];
                FOR_UNROLL for (int group = 0; group < (int)key_tiles_per_simdgroup; group++) {
                    int key_tile = (!shared_k && query_parallel_rows) ? group : ((int)simdgroup_slot + group * (int)simdgroups);
                    if (key_tile < fragments) {
                        device const float* key_even_ptr = key_even + (key0 + (long)key_tile * 8L) * (kv_heads * half_dim) + kv_head * half_dim;
                        device const float* key_odd_ptr = key_odd + (key0 + (long)key_tile * 8L) * (kv_heads * half_dim) + kv_head * half_dim;
                        FOR_UNROLL for (int step_index = 0; step_index < (int)depth_unroll; step_index++) {
                            if (prefetch_next_block && mode == 0L && step > 0L) {
                                long fragment_offset = ((long)key_tile * (half_dim / 8L) + (depth / 8L) + step_index) * 64L;
                                @SHARED_K_LOAD@(key_even_tile[group][step_index], prefetched_key_even + fragment_offset, 8, lane);
                                @SHARED_K_LOAD@(key_odd_tile[group][step_index], prefetched_key_odd + fragment_offset, 8, lane);
                            } else if (@BF16_CACHE@ && mode == 0L) {
                                device const ushort* cached_even = (device const ushort*)in2 + (key0 + (long)key_tile * 8L) * (kv_heads * half_dim) + kv_head * half_dim + depth + 8 * step_index;
                                device const ushort* cached_odd = (device const ushort*)in3 + (key0 + (long)key_tile * 8L) * (kv_heads * half_dim) + kv_head * half_dim + depth + 8 * step_index;
                                simdgroup_float8x8 even_float; simdgroup_float8x8 odd_float;
                                omega_bf16_load_matrix(even_float, cached_even, (ulong)(kv_heads * half_dim), lane, true);
                                omega_bf16_load_matrix(odd_float, cached_odd, (ulong)(kv_heads * half_dim), lane, true);
                                key_even_tile[group][step_index] = @NARROW_TO_OPERAND_EVEN@;
                                key_odd_tile[group][step_index] = @NARROW_TO_OPERAND_ODD@;
                            } else if (@BF8_CACHE@ && mode == 0L) {
                                device const uchar* cached_even = (device const uchar*)in2 + (key0 + (long)key_tile * 8L) * (kv_heads * half_dim) + kv_head * half_dim + depth + 8 * step_index;
                                device const uchar* cached_odd = (device const uchar*)in3 + (key0 + (long)key_tile * 8L) * (kv_heads * half_dim) + kv_head * half_dim + depth + 8 * step_index;
                                simdgroup_float8x8 even_float; simdgroup_float8x8 odd_float;
                                omega_bf8_load_matrix(even_float, cached_even, (ulong)(kv_heads * half_dim), lane, true);
                                omega_bf8_load_matrix(odd_float, cached_odd, (ulong)(kv_heads * half_dim), lane, true);
                                key_even_tile[group][step_index] = @NARROW_TO_OPERAND_EVEN@;
                                key_odd_tile[group][step_index] = @NARROW_TO_OPERAND_ODD@;
                            } else {
                                simdgroup_load(key_even_tile[group][step_index], key_even_ptr + depth + 8 * step_index, (ulong)(kv_heads * half_dim), ulong2(0, 0), true);
                                simdgroup_load(key_odd_tile[group][step_index], key_odd_ptr + depth + 8 * step_index, (ulong)(kv_heads * half_dim), ulong2(0, 0), true);
                            }
                        }
                    }
                }
                if (@KV_REUSE_SHARED_K@) {
                    FOR_UNROLL for (int group = 0; group < (int)key_tiles_per_group; group++) {
                        int key_tile = (int)simdgroup_slot + group * (int)simdgroups;
                        if (key_tile < fragments) {
                            FOR_UNROLL for (int step_index = 0; step_index < (int)depth_unroll; step_index++) {
                                int shared_index = (key_tile * (int)depth_unroll + step_index) * 64;
                                simdgroup_store(key_even_tile[group][step_index], shared_key_even + shared_index, 8);
                                simdgroup_store(key_odd_tile[group][step_index], shared_key_odd + shared_index, 8);
                            }
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (@KV_REUSE_SHARED_K@) {
                    FOR_UNROLL for (int key_tile = 0; key_tile < fragments; key_tile++) {
                        int shared_index = (key_tile * (int)depth_unroll) * 64;
                        @OPERAND@ shared_even[depth_unroll]; @OPERAND@ shared_odd[depth_unroll];
                        FOR_UNROLL for (int step_index = 0; step_index < (int)depth_unroll; step_index++) {
                            int fragment_index = shared_index + step_index * 64;
                            @SHARED_K_LOAD@(shared_even[step_index], shared_key_even + fragment_index, 8, lane);
                            @SHARED_K_LOAD@(shared_odd[step_index], shared_key_odd + fragment_index, 8, lane);
                        }
                        for (int vector_block = (int)simdgroup_slot; vector_block < (int)tile_blocks; vector_block += (int)simdgroups) {
                            int vector_slot = vector_block / (int)simdgroups;
                            long query_offset = (block_row[vector_block] * (kv_heads * query_groups) + kv_head * query_groups + block_head[vector_block]) * half_dim + depth;
                            @OPERAND@ query_even[depth_unroll]; @OPERAND@ query_odd[depth_unroll];
                            FOR_UNROLL for (int step_index = 0; step_index < (int)depth_unroll; step_index++) {
                                if (stage_query) {
                                    threadgroup const float* stage_even = query_stage + (long)vector_block * 16L * query_stage_stride + depth + 8 * step_index;
                                    simdgroup_load(query_even[step_index], stage_even, (ulong)query_stage_stride);
                                    simdgroup_load(query_odd[step_index], stage_even + 8L * query_stage_stride, (ulong)query_stage_stride);
                                } else {
                                    simdgroup_load(query_even[step_index], in0 + query_offset + 8 * step_index, (ulong)query_stride);
                                    simdgroup_load(query_odd[step_index], in1 + query_offset + 8 * step_index, (ulong)query_stride);
                                }
                                simdgroup_multiply_accumulate(scores[key_tile][vector_slot], query_even[step_index], shared_even[step_index], scores[key_tile][vector_slot]);
                                simdgroup_multiply_accumulate(scores[key_tile][vector_slot], query_odd[step_index], shared_odd[step_index], scores[key_tile][vector_slot]);
                            }
                        }
                    }
                } else {
                    FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) {
                        if (query_owner_rows && vector_block % (int)simdgroups != (int)simdgroup_slot) { continue; }
                        int vector_slot = query_owner_rows ? vector_block / (int)simdgroups : vector_block;
                        long query_offset = (block_row[vector_block] * (kv_heads * query_groups) + kv_head * query_groups + block_head[vector_block]) * half_dim + depth;
                        @OPERAND@ query_even[depth_unroll]; @OPERAND@ query_odd[depth_unroll];
                        FOR_UNROLL for (int step_index = 0; step_index < (int)depth_unroll; step_index++) {
                            if (stage_query) {
                                threadgroup const float* stage_even = query_stage + (long)vector_block * 16L * query_stage_stride + depth + 8 * step_index;
                                simdgroup_load(query_even[step_index], stage_even, (ulong)query_stage_stride);
                                simdgroup_load(query_odd[step_index], stage_even + 8L * query_stage_stride, (ulong)query_stage_stride);
                            } else {
                                simdgroup_load(query_even[step_index], in0 + query_offset + 8 * step_index, (ulong)query_stride);
                                simdgroup_load(query_odd[step_index], in1 + query_offset + 8 * step_index, (ulong)query_stride);
                            }
                        }
                        FOR_UNROLL for (int group = 0; group < (int)score_key_tiles; group++) {
                            int key_tile = query_owner_rows ? group : ((int)simdgroup_slot + group * (int)simdgroups);
                            if (key_tile < fragments) {
                                FOR_UNROLL for (int step_index = 0; step_index < (int)depth_unroll; step_index++) {
                                    simdgroup_multiply_accumulate(scores[group][vector_slot], query_even[step_index], key_even_tile[group][step_index], scores[group][vector_slot]);
                                    simdgroup_multiply_accumulate(scores[group][vector_slot], query_odd[step_index], key_odd_tile[group][step_index], scores[group][vector_slot]);
                                }
                            }
                        }
                    }
                }
                if (@KV_REUSE_SHARED_K@) { threadgroup_barrier(mem_flags::mem_threadgroup); }
            }
            if (@KV_REUSE_SHARED_K@) {
                for (int key_tile = 0; key_tile < fragments; key_tile++) {
                    for (int vector_block = (int)simdgroup_slot; vector_block < (int)tile_blocks; vector_block += (int)simdgroups) {
                        int vector_slot = vector_block / (int)simdgroups;
                        simdgroup_store(scores[key_tile][vector_slot], score_tile + vector_block * 8 * (int)block + key_tile * 8, (ulong)block);
                    }
                }
            } else if (query_owner_rows) {
                for (int key_tile = 0; key_tile < fragments; key_tile++) {
                    for (int vector_block = (int)simdgroup_slot; vector_block < (int)tile_blocks; vector_block += (int)simdgroups) {
                        int vector_slot = vector_block / (int)simdgroups;
                        simdgroup_store(scores[key_tile][vector_slot], score_tile + vector_block * 8 * (int)block + key_tile * 8, (ulong)block);
                    }
                }
            } else {
                FOR_UNROLL for (int group = 0; group < (int)key_tiles_per_group; group++) {
                    int key_tile = (int)simdgroup_slot + group * (int)simdgroups;
                    if (key_tile < fragments) {
                        FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) { simdgroup_store(scores[group][vector_block], score_tile + vector_block * 8 * (int)block + key_tile * 8, (ulong)block); }
                    }
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
        if (@QUERY_OWNER_ROWS@) {
            for (int vector_block = (int)simdgroup_slot; vector_block < (int)tile_blocks; vector_block += (int)simdgroups) {
                int vector_slot = vector_block / (int)simdgroups;
                float row_scale = rescale_tile[vector_block * 8 + fragment_row];
                FOR_UNROLL for (int dimension_block = 0; dimension_block < (int)(head_dim / 8); dimension_block++) {
                    accumulated[dimension_block][vector_slot].thread_elements()[0] *= row_scale;
                    accumulated[dimension_block][vector_slot].thread_elements()[1] *= row_scale;
                }
            }
        } else {
            FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) {
                float row_scale = rescale_tile[vector_block * 8 + fragment_row];
                FOR_UNROLL for (int slot = 0; slot < (int)dims_per_group; slot++) {
                    accumulated[slot][vector_block].thread_elements()[0] *= row_scale;
                    accumulated[slot][vector_block].thread_elements()[1] *= row_scale;
                }
            }
        }
        if (mode != 2L) {
            device const float* value_base = mode == 0L ? (device const float*)in6 : in7;
            device const float* value_ptr = value_base + key0 * (kv_heads * head_dim) + kv_head * head_dim;
            int fragments = (int)((columns + 7L) / 8L);
            if (@KV_REUSE_SHARED_V@) {
                for (int key_tile = 0; key_tile < fragments; key_tile++) {
                    @OPERAND@ weights[score_vectors_per_simdgroup];
                    for (int vector_block = (int)simdgroup_slot; vector_block < (int)tile_blocks; vector_block += (int)simdgroups) {
                        int vector_slot = vector_block / (int)simdgroups;
                        simdgroup_load(weights[vector_slot], score_tile + vector_block * 8 * (int)block + key_tile * 8, (ulong)block);
                    }
                    for (int dimension_base = 0; dimension_base < (int)(head_dim / 8); dimension_base += (int)shared_v_dimension_blocks) {
                        int dimension_count = min((int)shared_v_dimension_blocks, (int)(head_dim / 8) - dimension_base);
                        for (int slab_slot = 0; slab_slot < dimension_count; slab_slot++) {
                            int dimension_block = dimension_base + slab_slot;
                            if (dimension_block % (int)simdgroups == (int)simdgroup_slot) {
                                simdgroup_float8x8 value_float;
                                if (prefetch_next_block && mode == 0L && step > 0L) {
                                    long fragment_offset = ((long)key_tile * (head_dim / 8L) + dimension_block) * 64L;
                                    @PREFETCH_LOAD@(value_float, prefetched_value + fragment_offset, 8, lane);
                                } else if (@BF16_CACHE@ && mode == 0L) {
                                    device const ushort* cached_value = (device const ushort*)in6 + key0 * (kv_heads * head_dim) + kv_head * head_dim + (long)key_tile * 8L * (kv_heads * head_dim) + dimension_block * 8;
                                    omega_bf16_load_matrix(value_float, cached_value, (ulong)(kv_heads * head_dim), lane, false);
                                } else if (@BF8_CACHE@ && mode == 0L) {
                                    device const uchar* cached_value = (device const uchar*)in6 + key0 * (kv_heads * head_dim) + kv_head * head_dim + (long)key_tile * 8L * (kv_heads * head_dim) + dimension_block * 8;
                                    omega_bf8_load_matrix(value_float, cached_value, (ulong)(kv_heads * head_dim), lane, false);
                                } else {
                                    simdgroup_load(value_float, value_ptr + (long)key_tile * 8L * (kv_heads * head_dim) + dimension_block * 8, (ulong)(kv_heads * head_dim));
                                }
                                omega_zero_padded_value_rows(value_float, key0 + (long)key_tile * 8L, mode == 0L ? slice_end : mma_end, lane);
                                @OPERAND@ value_operand = @NARROW_TO_OPERAND_VALUE@;
                                simdgroup_store(value_operand, shared_value + slab_slot * 64, 8);
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        for (int slab_slot = 0; slab_slot < dimension_count; slab_slot++) {
                            int dimension_block = dimension_base + slab_slot;
                            @OPERAND@ value_operand;
                            @SHARED_V_LOAD@(value_operand, shared_value + slab_slot * 64, 8, lane);
                            for (int vector_block = (int)simdgroup_slot; vector_block < (int)tile_blocks; vector_block += (int)simdgroups) {
                                int vector_slot = vector_block / (int)simdgroups;
                                simdgroup_multiply_accumulate(accumulated[dimension_block][vector_slot], weights[vector_slot], value_operand, accumulated[dimension_block][vector_slot]);
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                }
            } else if (@QUERY_PARALLEL_ROWS@) {
                for (int key_tile = 0; key_tile < fragments; key_tile++) {
                    @OPERAND@ weights[score_vectors_per_simdgroup];
                    for (int vector_block = (int)simdgroup_slot; vector_block < (int)tile_blocks; vector_block += (int)simdgroups) {
                        int vector_slot = vector_block / (int)simdgroups;
                        simdgroup_load(weights[vector_slot], score_tile + vector_block * 8 * (int)block + key_tile * 8, (ulong)block);
                    }
                    FOR_UNROLL for (int dimension_block = 0; dimension_block < (int)(head_dim / 8); dimension_block++) {
                        simdgroup_float8x8 value_float;
                        if (prefetch_next_block && mode == 0L && step > 0L) {
                            long fragment_offset = ((long)key_tile * (head_dim / 8L) + dimension_block) * 64L;
                            @PREFETCH_LOAD@(value_float, prefetched_value + fragment_offset, 8, lane);
                        } else if (@BF16_CACHE@ && mode == 0L) {
                            device const ushort* cached_value = (device const ushort*)in6 + key0 * (kv_heads * head_dim) + kv_head * head_dim + (long)key_tile * 8L * (kv_heads * head_dim) + dimension_block * 8;
                            omega_bf16_load_matrix(value_float, cached_value, (ulong)(kv_heads * head_dim), lane, false);
                        } else if (@BF8_CACHE@ && mode == 0L) {
                            device const uchar* cached_value = (device const uchar*)in6 + key0 * (kv_heads * head_dim) + kv_head * head_dim + (long)key_tile * 8L * (kv_heads * head_dim) + dimension_block * 8;
                            omega_bf8_load_matrix(value_float, cached_value, (ulong)(kv_heads * head_dim), lane, false);
                        } else {
                            simdgroup_load(value_float, value_ptr + (long)key_tile * 8L * (kv_heads * head_dim) + dimension_block * 8, (ulong)(kv_heads * head_dim));
                        }
                        omega_zero_padded_value_rows(value_float, key0 + (long)key_tile * 8L, mode == 0L ? slice_end : mma_end, lane);
                        @OPERAND@ value = @NARROW_TO_OPERAND_VALUE@;
                        for (int vector_block = (int)simdgroup_slot; vector_block < (int)tile_blocks; vector_block += (int)simdgroups) {
                            int vector_slot = vector_block / (int)simdgroups;
                            simdgroup_multiply_accumulate(accumulated[dimension_block][vector_slot], weights[vector_slot], value, accumulated[dimension_block][vector_slot]);
                        }
                    }
                }
            } else {
                for (int key_tile = 0; key_tile < fragments; key_tile++) {
                    @OPERAND@ weights[tile_blocks];
                    FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) { simdgroup_load(weights[vector_block], score_tile + vector_block * 8 * (int)block + key_tile * 8, (ulong)block); }
                    FOR_UNROLL for (int slot = 0; slot < (int)dims_per_group; slot++) {
                        int dimension_block = (int)simdgroup_slot + slot * (int)simdgroups;
                        simdgroup_float8x8 value_float;
                        if (prefetch_next_block && mode == 0L && step > 0L) {
                            long fragment_offset = ((long)key_tile * (head_dim / 8L) + dimension_block) * 64L;
                            @PREFETCH_LOAD@(value_float, prefetched_value + fragment_offset, 8, lane);
                        } else if (@BF16_CACHE@ && mode == 0L) {
                            device const ushort* cached_value = (device const ushort*)in6 + key0 * (kv_heads * head_dim) + kv_head * head_dim + (long)key_tile * 8L * (kv_heads * head_dim) + dimension_block * 8;
                            omega_bf16_load_matrix(value_float, cached_value, (ulong)(kv_heads * head_dim), lane, false);
                        } else if (@BF8_CACHE@ && mode == 0L) {
                            device const uchar* cached_value = (device const uchar*)in6 + key0 * (kv_heads * head_dim) + kv_head * head_dim + (long)key_tile * 8L * (kv_heads * head_dim) + dimension_block * 8;
                            omega_bf8_load_matrix(value_float, cached_value, (ulong)(kv_heads * head_dim), lane, false);
                        } else {
                            simdgroup_load(value_float, value_ptr + (long)key_tile * 8L * (kv_heads * head_dim) + dimension_block * 8, (ulong)(kv_heads * head_dim));
                        }
                        omega_zero_padded_value_rows(value_float, key0 + (long)key_tile * 8L, mode == 0L ? slice_end : mma_end, lane);
                        @OPERAND@ value = @NARROW_TO_OPERAND_VALUE@;
                        FOR_UNROLL for (int vector_block = 0; vector_block < (int)tile_blocks; vector_block++) { simdgroup_multiply_accumulate(accumulated[slot][vector_block], weights[vector_block], value, accumulated[slot][vector_block]); }
                    }
                }
            }
        } else if (@QUERY_OWNER_ROWS@) {
            FOR_UNROLL for (int dimension_block = 0; dimension_block < (int)(head_dim / 8); dimension_block++) {
                long dimension = (long)dimension_block * 8L + (long)fragment_column;
                for (int vector_block = (int)simdgroup_slot; vector_block < (int)tile_blocks; vector_block += (int)simdgroups) {
                    int vector_slot = vector_block / (int)simdgroups;
                    float sum_even = 0.0f; float sum_odd = 0.0f;
                    for (long column = 0L; column < columns; column++) {
                        float weight = score_tile[(vector_block * 8 + fragment_row) * block + column];
                        long value_offset = (key0 + column) * (kv_heads * head_dim) + kv_head * head_dim + dimension;
                        sum_even += weight * in7[value_offset];
                        sum_odd += weight * in7[value_offset + 1L];
                    }
                    accumulated[dimension_block][vector_slot].thread_elements()[0] += sum_even;
                    accumulated[dimension_block][vector_slot].thread_elements()[1] += sum_odd;
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
        if (prefetch_next_block && mode == 0L && step + 1L < cached_blocks) {
            long next_key0 = key0 + block;
            long next_columns = min(block, slice_end - next_key0);
            int next_fragments = (int)((next_columns + 7L) / 8L);
            int depth_fragments = (int)(half_dim / 8L);
            int value_fragments = (int)(head_dim / 8L);
            for (int fragment_index = (int)simdgroup_slot; fragment_index < next_fragments * depth_fragments; fragment_index += (int)simdgroups) {
                int key_tile = fragment_index / depth_fragments;
                int depth_tile = fragment_index % depth_fragments;
                long fragment_offset = (long)fragment_index * 64L;
                device const float* key_even_ptr = (device const float*)in2 + (next_key0 + (long)key_tile * 8L) * (kv_heads * half_dim) + kv_head * half_dim + depth_tile * 8L;
                device const float* key_odd_ptr = (device const float*)in3 + (next_key0 + (long)key_tile * 8L) * (kv_heads * half_dim) + kv_head * half_dim + depth_tile * 8L;
                simdgroup_float8x8 even_float;
                simdgroup_float8x8 odd_float;
                if (@BF16_CACHE@) {
                    device const ushort* cached_even = (device const ushort*)in2 + (next_key0 + (long)key_tile * 8L) * (kv_heads * half_dim) + kv_head * half_dim + depth_tile * 8L;
                    device const ushort* cached_odd = (device const ushort*)in3 + (next_key0 + (long)key_tile * 8L) * (kv_heads * half_dim) + kv_head * half_dim + depth_tile * 8L;
                    omega_bf16_load_matrix(even_float, cached_even, (ulong)(kv_heads * half_dim), lane, true);
                    omega_bf16_load_matrix(odd_float, cached_odd, (ulong)(kv_heads * half_dim), lane, true);
                } else if (@BF8_CACHE@) {
                    device const uchar* cached_even = (device const uchar*)in2 + (next_key0 + (long)key_tile * 8L) * (kv_heads * half_dim) + kv_head * half_dim + depth_tile * 8L;
                    device const uchar* cached_odd = (device const uchar*)in3 + (next_key0 + (long)key_tile * 8L) * (kv_heads * half_dim) + kv_head * half_dim + depth_tile * 8L;
                    omega_bf8_load_matrix(even_float, cached_even, (ulong)(kv_heads * half_dim), lane, true);
                    omega_bf8_load_matrix(odd_float, cached_odd, (ulong)(kv_heads * half_dim), lane, true);
                } else {
                    simdgroup_load(even_float, key_even_ptr, (ulong)(kv_heads * half_dim), ulong2(0, 0), true);
                    simdgroup_load(odd_float, key_odd_ptr, (ulong)(kv_heads * half_dim), ulong2(0, 0), true);
                }
                @OPERAND@ even_operand = @NARROW_TO_OPERAND_EVEN@;
                @OPERAND@ odd_operand = @NARROW_TO_OPERAND_ODD@;
                simdgroup_store(even_operand, prefetched_key_even + fragment_offset, 8);
                simdgroup_store(odd_operand, prefetched_key_odd + fragment_offset, 8);
            }
            for (int fragment_index = (int)simdgroup_slot; fragment_index < next_fragments * value_fragments; fragment_index += (int)simdgroups) {
                int key_tile = fragment_index / value_fragments;
                int dimension_block = fragment_index % value_fragments;
                long fragment_offset = (long)fragment_index * 64L;
                device const float* value_ptr = (device const float*)in6 + next_key0 * (kv_heads * head_dim) + kv_head * head_dim + (long)key_tile * 8L * (kv_heads * head_dim) + dimension_block * 8L;
                simdgroup_float8x8 value_float;
                if (@BF16_CACHE@) {
                    device const ushort* cached_value = (device const ushort*)in6 + next_key0 * (kv_heads * head_dim) + kv_head * head_dim + (long)key_tile * 8L * (kv_heads * head_dim) + dimension_block * 8L;
                    omega_bf16_load_matrix(value_float, cached_value, (ulong)(kv_heads * head_dim), lane, false);
                } else if (@BF8_CACHE@) {
                    device const uchar* cached_value = (device const uchar*)in6 + next_key0 * (kv_heads * head_dim) + kv_head * head_dim + (long)key_tile * 8L * (kv_heads * head_dim) + dimension_block * 8L;
                    omega_bf8_load_matrix(value_float, cached_value, (ulong)(kv_heads * head_dim), lane, false);
                } else {
                    simdgroup_load(value_float, value_ptr, (ulong)(kv_heads * head_dim));
                }
                omega_zero_padded_value_rows(value_float, next_key0 + (long)key_tile * 8L, slice_end, lane);
                @OPERAND@ value_operand = @NARROW_TO_OPERAND_VALUE@;
                simdgroup_store(value_operand, prefetched_value + fragment_offset, 8);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (@QUERY_OWNER_ROWS@) {
        for (int vector_block = (int)simdgroup_slot; vector_block < (int)tile_blocks; vector_block += (int)simdgroups) {
            int vector_slot = vector_block / (int)simdgroups;
            long vector = (long)vector_block * 8L + (long)fragment_row;
            if (vector_live[vector] != 0) {
                long query_index = (long)vector_row[vector] * (kv_heads * query_groups) + kv_head * query_groups + (long)vector_head[vector];
                float sum = row_sum[vector];
                float inverse = sum == 0.0f ? 0.0f : 1.0f / sum;
                FOR_UNROLL for (int dimension_block = 0; dimension_block < (int)(head_dim / 8); dimension_block++) {
                    long dimension = (long)dimension_block * 8L + (long)fragment_column;
                    if (splits == 1L) {
                        out[query_index * head_dim + dimension] = accumulated[dimension_block][vector_slot].thread_elements()[0] * inverse;
                        out[query_index * head_dim + dimension + 1L] = accumulated[dimension_block][vector_slot].thread_elements()[1] * inverse;
                    } else {
                        device float* attn_scratch = out;
                        long base = ((query_index * (head_dim / 4L) + (dimension >> 2)) * splits + split) * 4L + (dimension & 3L);
                        attn_scratch[base] = accumulated[dimension_block][vector_slot].thread_elements()[0];
                        attn_scratch[base + 1L] = accumulated[dimension_block][vector_slot].thread_elements()[1];
                    }
                }
            }
        }
    } else {
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
