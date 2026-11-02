use proxima_tensor::{BoundOpKind, NumericPolicy};

const ATTENTION_TILE_ITEMS: u64 = 32;

const ATTENTION_TILE_MIN_ROWS: u64 = 16;

const ATTENTION_TILE_HEAD_DIM: u64 = 128;

pub(crate) fn cached_attention_uses_mma(kind: &BoundOpKind, numeric_policy: NumericPolicy) -> bool {
    numeric_policy.activation_quantization && cached_attention_tile_rows(kind).is_some()
}

pub(crate) fn cached_attention_tile_rows(kind: &BoundOpKind) -> Option<u64> {
    let BoundOpKind::CachedAttention {
        query_rows,
        query_groups,
        head_dim,
        rotary_dim,
        ..
    } = kind
    else {
        return None;
    };
    let vector_planes = rotary_dim % 8 == 0 && (head_dim - rotary_dim) % 4 == 0;
    let fits = (1..=ATTENTION_TILE_ITEMS).contains(query_groups)
        && *head_dim == ATTENTION_TILE_HEAD_DIM
        && vector_planes;
    (fits && *query_rows >= ATTENTION_TILE_MIN_ROWS).then(|| ATTENTION_TILE_ITEMS / query_groups)
}

pub(crate) const TILED_ATTENTION_BODY: &str = "\
    constexpr int warps = 4;
    constexpr int per_warp = 8;
    constexpr int items = tile_rows * (int)query_groups;
    constexpr int key_tile = 32;
    constexpr int key_stride = 132;
    constexpr int dim_chunks = 32;
    const float negative_infinity = -__int_as_float(0x7f800000);
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const long query_rows = u_ptr->total_elements / (kv_heads * query_groups);
    const long kv_head = (long)blockIdx.x % kv_heads;
    const long first_row = ((long)blockIdx.x / kv_heads) * tile_rows;
    const long row_count = (query_rows - first_row < (long)tile_rows) ? query_rows - first_row : (long)tile_rows;
    const long last_row = first_row + row_count - 1L;
    const long last_key = cached_key_rows + new_key_rows - 1L;
    auto allowed = [&](long key, long row) -> bool {
        const long relative = key - cached_key_rows - row;
        return key < cached_key_rows ? relative >= cached_lower : relative <= new_upper;
    };
    __shared__ __align__(16) float tile_query[warps * per_warp * 128];
    __shared__ __align__(16) float tile_stage[key_tile * key_stride];
    __shared__ __align__(16) float tile_weight[warps * per_warp * key_tile];
    for (int index = tid; index < items * dim_chunks; index += (int)blockDim.x) {
        const int item = index / dim_chunks;
        const int dimension = (index % dim_chunks) * 4;
        const long row = first_row + item / (int)query_groups;
        const long head = kv_head * query_groups + item % (int)query_groups;
        const long base = row * (kv_heads * query_groups * pair_dim) + head * pair_dim;
        const long pass_base = row * (kv_heads * query_groups * pass_dim) + head * pass_dim;
        float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (row <= last_row) {
            value = dimension < pair_dim ? *(const float4*)&in0[base + dimension]
                : dimension < 2 * pair_dim ? *(const float4*)&in1[base + dimension - pair_dim]
                : *(const float4*)&pass_query[pass_base + dimension - rotary_dim];
        }
        *(float4*)&tile_query[item * 128 + dimension] = value;
    }
    long item_row[per_warp];
    float running_max[per_warp];
    float running_sum[per_warp];
    float4 accumulator[per_warp];
    for (int slot = 0; slot < per_warp; slot++) {
        const int item = warp * per_warp + slot;
        item_row[slot] = (item < items) ? first_row + item / (int)query_groups : -1L;
        running_max[slot] = negative_infinity;
        running_sum[slot] = 0.0f;
        accumulator[slot] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (long tile_start = 0; tile_start <= last_key; tile_start += key_tile) {
        bool any_allowed = false;
        for (int offset = 0; offset < key_tile && tile_start + offset <= last_key; offset++) {
            const long key = tile_start + offset;
            any_allowed = any_allowed || allowed(key, key < cached_key_rows ? first_row : last_row);
        }
        if (!any_allowed) { continue; }
        for (int index = tid; index < key_tile * dim_chunks; index += (int)blockDim.x) {
            const int offset = index / dim_chunks;
            const int dimension = (index % dim_chunks) * 4;
            const long key = tile_start + offset;
            float4 key_value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (key <= last_key) {
                const bool cached = key < cached_key_rows;
                const long key_row = cached ? key : key - cached_key_rows;
                const long kbase = key_row * (kv_heads * pair_dim) + kv_head * pair_dim;
                const long kpass = key_row * (kv_heads * pass_dim) + kv_head * pass_dim;
                key_value = dimension < pair_dim ? *(const float4*)&(cached ? in2 : in4)[kbase + dimension]
                    : dimension < 2 * pair_dim ? *(const float4*)&(cached ? in3 : in5)[kbase + dimension - pair_dim]
                    : *(const float4*)&(cached ? pass_cached_key : pass_new_key)[kpass + dimension - rotary_dim];
            }
            *(float4*)&tile_stage[offset * key_stride + dimension] = key_value;
        }
        __syncthreads();
        float score[per_warp];
        for (int slot = 0; slot < per_warp; slot++) { score[slot] = 0.0f; }
        #pragma unroll 4
        for (int chunk = 0; chunk < dim_chunks; chunk++) {
            const float4 key_chunk = *(const float4*)&tile_stage[lane * key_stride + chunk * 4];
            #pragma unroll
            for (int slot = 0; slot < per_warp; slot++) {
                const float4 query_chunk = *(const float4*)&tile_query[(warp * per_warp + slot) * 128 + chunk * 4];
                score[slot] += query_chunk.x * key_chunk.x + query_chunk.y * key_chunk.y
                    + query_chunk.z * key_chunk.z + query_chunk.w * key_chunk.w;
            }
        }
        const long key = tile_start + lane;
        #pragma unroll
        for (int slot = 0; slot < per_warp; slot++) {
            const bool live = item_row[slot] >= 0 && key <= last_key && allowed(key, item_row[slot]);
            const float masked = live ? score[slot] * scale : negative_infinity;
            float tile_max = masked;
            for (int offset = 16; offset > 0; offset >>= 1) {
                tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset));
            }
            const float next_max = fmaxf(running_max[slot], tile_max);
            const float rescale = (running_max[slot] == negative_infinity) ? 0.0f : expf(running_max[slot] - next_max);
            const float weight = (masked == negative_infinity) ? 0.0f : expf(masked - next_max);
            running_sum[slot] = running_sum[slot] * rescale + weight;
            accumulator[slot].x *= rescale;
            accumulator[slot].y *= rescale;
            accumulator[slot].z *= rescale;
            accumulator[slot].w *= rescale;
            running_max[slot] = next_max;
            tile_weight[(warp * per_warp + slot) * key_tile + lane] = weight;
        }
        __syncthreads();
        for (int index = tid; index < key_tile * dim_chunks; index += (int)blockDim.x) {
            const int offset = index / dim_chunks;
            const int dimension = (index % dim_chunks) * 4;
            const long key_index = tile_start + offset;
            float4 value_chunk = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (key_index <= last_key) {
                const bool cached = key_index < cached_key_rows;
                const long key_row = cached ? key_index : key_index - cached_key_rows;
                const long vbase = key_row * (kv_heads * head_dim) + kv_head * head_dim;
                value_chunk = *(const float4*)&(cached ? in6 : in7)[vbase + dimension];
            }
            *(float4*)&tile_stage[offset * key_stride + dimension] = value_chunk;
        }
        __syncthreads();
        #pragma unroll 2
        for (int group = 0; group < key_tile / 4; group++) {
            float4 value_rows[4];
            #pragma unroll
            for (int step = 0; step < 4; step++) {
                value_rows[step] = *(const float4*)&tile_stage[(group * 4 + step) * key_stride + lane * 4];
            }
            #pragma unroll
            for (int slot = 0; slot < per_warp; slot++) {
                const float4 weights = *(const float4*)&tile_weight[(warp * per_warp + slot) * key_tile + group * 4];
                const float weight_steps[4] = {weights.x, weights.y, weights.z, weights.w};
                #pragma unroll
                for (int step = 0; step < 4; step++) {
                    accumulator[slot].x += weight_steps[step] * value_rows[step].x;
                    accumulator[slot].y += weight_steps[step] * value_rows[step].y;
                    accumulator[slot].z += weight_steps[step] * value_rows[step].z;
                    accumulator[slot].w += weight_steps[step] * value_rows[step].w;
                }
            }
        }
        __syncthreads();
    }
    for (int slot = 0; slot < per_warp; slot++) {
        float total = running_sum[slot];
        for (int offset = 16; offset > 0; offset >>= 1) { total += __shfl_xor_sync(0xffffffffu, total, offset); }
        if (item_row[slot] < 0) { continue; }
        const int item = warp * per_warp + slot;
        const long head = kv_head * query_groups + item % (int)query_groups;
        const long vector_index = item_row[slot] * (kv_heads * query_groups) + head;
        const float inverse = (total == 0.0f) ? 0.0f : 1.0f / total;
        *(float4*)&out[vector_index * head_dim + lane * 4] = make_float4(
            accumulator[slot].x * inverse, accumulator[slot].y * inverse,
            accumulator[slot].z * inverse, accumulator[slot].w * inverse);
    }
}
";
