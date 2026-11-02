pub(crate) const MMA_ATTENTION_HELPERS: &str = r#"
__device__ __forceinline__ unsigned int attention_pack(float first, float second) {
    const __half2 packed = __floats2half2_rn(first, second);
    return *reinterpret_cast<const unsigned int*>(&packed);
}
__device__ __forceinline__ unsigned int attention_pack_halves(__half first, __half second) {
    const __half2 packed = __halves2half2(first, second);
    return *reinterpret_cast<const unsigned int*>(&packed);
}
__device__ __forceinline__ void attention_split(float first, float second, unsigned int *high, unsigned int *low) {
    const __half2 head = __floats2half2_rn(first, second);
    const float2 back = __half22float2(head);
    *high = *reinterpret_cast<const unsigned int*>(&head);
    *low = attention_pack(first - back.x, second - back.y);
}
"#;

pub(crate) const MMA_ATTENTION_BODY: &str = "\
    constexpr int warps = 4;
    constexpr int items = tile_rows * (int)query_groups;
    constexpr int key_tile = 32;
    constexpr int key_tiles = key_tile / 8;
    constexpr int dim_chunks = (int)head_dim / 4;
    constexpr int dim_steps = (int)head_dim / 8;
    constexpr int dim_tiles = (int)head_dim / 16;
    constexpr int query_stride = (int)head_dim + 8;
    constexpr int loads = key_tile / warps;
    static_assert(head_dim == 128);
    const float negative_infinity = -__int_as_float(0x7f800000);
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group_id = lane >> 2;
    const int quad = lane & 3;
    const int block_item = (warp >> 1) * 16;
    const int dim_base = (warp & 1) * ((int)head_dim / 2);
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
    __shared__ __align__(16) __half tile_query[32 * query_stride];
    __shared__ __align__(16) __half tile_key[key_tile * query_stride];
    __shared__ __align__(16) __half tile_value[key_tile * query_stride];
    for (int index = tid; index < 32 * dim_chunks; index += (int)blockDim.x) {
        const int item = index / dim_chunks;
        const int dimension = (index % dim_chunks) * 4;
        const long row = first_row + item / (int)query_groups;
        const long head = kv_head * query_groups + item % (int)query_groups;
        const long base = row * (kv_heads * query_groups * pair_dim) + head * pair_dim;
        const long pass_base = row * (kv_heads * query_groups * pass_dim) + head * pass_dim;
        float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (item < items && row <= last_row) {
            value = dimension < pair_dim ? *(const float4*)&in0[base + dimension]
                : dimension < 2 * pair_dim ? *(const float4*)&in1[base + dimension - pair_dim]
                : *(const float4*)&pass_query[pass_base + dimension - rotary_dim];
        }
        *(unsigned int*)&tile_query[item * query_stride + dimension] = attention_pack(value.x, value.y);
        *(unsigned int*)&tile_query[item * query_stride + dimension + 2] = attention_pack(value.z, value.w);
    }
    long item_row[2];
    float running_max[2];
    float running_sum[2];
    for (int slot = 0; slot < 2; slot++) {
        const int item = block_item + group_id + 8 * slot;
        const long row = first_row + item / (int)query_groups;
        item_row[slot] = (item < items && row <= last_row) ? row : -1L;
        running_max[slot] = negative_infinity;
        running_sum[slot] = 0.0f;
    }
    float accumulator[dim_tiles][4];
    #pragma unroll
    for (int tile = 0; tile < dim_tiles; tile++) {
        accumulator[tile][0] = 0.0f; accumulator[tile][1] = 0.0f; accumulator[tile][2] = 0.0f; accumulator[tile][3] = 0.0f;
    }
    auto tile_open = [&](long start) -> bool {
        const long end = (start + key_tile - 1 < last_key) ? start + key_tile - 1 : last_key;
        const long cached_end = (end < cached_key_rows - 1L) ? end : cached_key_rows - 1L;
        const bool cached_open = start < cached_key_rows && cached_end >= cached_lower + cached_key_rows + first_row;
        const long new_start = (start > cached_key_rows) ? start : cached_key_rows;
        const bool new_open = end >= cached_key_rows && new_start <= new_upper + cached_key_rows + last_row;
        return cached_open || new_open;
    };
    auto next_open = [&](long start) -> long {
        while (start <= last_key && !tile_open(start)) { start += key_tile; }
        return start;
    };
    float4 key_regs[loads];
    float4 value_regs[loads];
    auto fetch = [&](long tile_start) {
        const int dimension = lane * 4;
        #pragma unroll
        for (int load = 0; load < loads; load++) {
            const long key = tile_start + warp + warps * load;
            float4 key_value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 value_value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (key <= last_key) {
                const bool cached = key < cached_key_rows;
                const long key_row = cached ? key : key - cached_key_rows;
                const long kbase = key_row * (kv_heads * pair_dim) + kv_head * pair_dim;
                const long kpass = key_row * (kv_heads * pass_dim) + kv_head * pass_dim;
                key_value = dimension < pair_dim ? *(const float4*)&(cached ? in2 : in4)[kbase + dimension]
                    : dimension < 2 * pair_dim ? *(const float4*)&(cached ? in3 : in5)[kbase + dimension - pair_dim]
                    : *(const float4*)&(cached ? pass_cached_key : pass_new_key)[kpass + dimension - rotary_dim];
                value_value = *(const float4*)&(cached ? in6 : in7)[key_row * (kv_heads * head_dim) + kv_head * head_dim + dimension];
            }
            key_regs[load] = key_value;
            value_regs[load] = value_value;
        }
    };
    auto stage = [&]() {
        const int dimension = lane * 4;
        #pragma unroll
        for (int load = 0; load < loads; load++) {
            const int offset = warp + warps * load;
            *(unsigned int*)&tile_key[offset * query_stride + dimension] = attention_pack(key_regs[load].x, key_regs[load].y);
            *(unsigned int*)&tile_key[offset * query_stride + dimension + 2] = attention_pack(key_regs[load].z, key_regs[load].w);
            *(unsigned int*)&tile_value[offset * query_stride + dimension] = attention_pack(value_regs[load].x, value_regs[load].y);
            *(unsigned int*)&tile_value[offset * query_stride + dimension + 2] = attention_pack(value_regs[load].z, value_regs[load].w);
        }
    };
    __syncthreads();
    long tile_start = next_open(0L);
    if (tile_start <= last_key) { fetch(tile_start); }
    while (tile_start <= last_key) {
        stage();
        __syncthreads();
        const long following = next_open(tile_start + key_tile);
        if (following <= last_key) { fetch(following); }
        float score[key_tiles][4];
        #pragma unroll
        for (int tile = 0; tile < key_tiles; tile++) {
            score[tile][0] = 0.0f; score[tile][1] = 0.0f; score[tile][2] = 0.0f; score[tile][3] = 0.0f;
        }
        #pragma unroll
        for (int step = 0; step < dim_steps; step++) {
            const unsigned int query_low = *(const unsigned int*)&tile_query[(block_item + group_id) * query_stride + step * 8 + quad * 2];
            const unsigned int query_high = *(const unsigned int*)&tile_query[(block_item + group_id + 8) * query_stride + step * 8 + quad * 2];
            #pragma unroll
            for (int tile = 0; tile < key_tiles; tile++) {
                const unsigned int key_pair = *(const unsigned int*)&tile_key[(tile * 8 + group_id) * query_stride + step * 8 + quad * 2];
                mma_m16n8k8(score[tile], query_low, query_high, key_pair);
            }
        }
        #pragma unroll
        for (int slot = 0; slot < 2; slot++) {
            float tile_max = negative_infinity;
            #pragma unroll
            for (int tile = 0; tile < key_tiles; tile++) {
                #pragma unroll
                for (int element = 0; element < 2; element++) {
                    const long key = tile_start + tile * 8 + quad * 2 + element;
                    const bool live = item_row[slot] >= 0 && key <= last_key && allowed(key, item_row[slot]);
                    const float masked = live ? score[tile][slot * 2 + element] * scale : negative_infinity;
                    score[tile][slot * 2 + element] = masked;
                    tile_max = fmaxf(tile_max, masked);
                }
            }
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 1));
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, 2));
            const float next_max = fmaxf(running_max[slot], tile_max);
            const float rescale = (running_max[slot] == negative_infinity) ? 0.0f : expf(running_max[slot] - next_max);
            float tile_sum = 0.0f;
            #pragma unroll
            for (int tile = 0; tile < key_tiles; tile++) {
                #pragma unroll
                for (int element = 0; element < 2; element++) {
                    const float masked = score[tile][slot * 2 + element];
                    const float weight = (masked == negative_infinity) ? 0.0f : expf(masked - next_max);
                    score[tile][slot * 2 + element] = weight;
                    tile_sum += weight;
                }
            }
            running_sum[slot] = running_sum[slot] * rescale + tile_sum;
            running_max[slot] = next_max;
            #pragma unroll
            for (int tile = 0; tile < dim_tiles; tile++) {
                accumulator[tile][slot * 2] *= rescale;
                accumulator[tile][slot * 2 + 1] *= rescale;
            }
        }
        #pragma unroll
        for (int step = 0; step < key_tiles; step++) {
            unsigned int top_high, bottom_high, top_low, bottom_low;
            attention_split(score[step][0], score[step][1], &top_high, &top_low);
            attention_split(score[step][2], score[step][3], &bottom_high, &bottom_low);
            #pragma unroll
            for (int tile = 0; tile < dim_tiles; tile++) {
                const int dimension = dim_base + tile * 8 + group_id;
                const int key_index = step * 8 + quad * 2;
                const unsigned int value_pair = attention_pack_halves(tile_value[key_index * query_stride + dimension], tile_value[(key_index + 1) * query_stride + dimension]);
                mma_m16n8k8(accumulator[tile], top_high, bottom_high, value_pair);
                mma_m16n8k8(accumulator[tile], top_low, bottom_low, value_pair);
            }
        }
        __syncthreads();
        tile_start = following;
    }
    #pragma unroll
    for (int slot = 0; slot < 2; slot++) {
        float total = running_sum[slot];
        total += __shfl_xor_sync(0xffffffffu, total, 1);
        total += __shfl_xor_sync(0xffffffffu, total, 2);
        if (item_row[slot] < 0) { continue; }
        const int item = block_item + group_id + 8 * slot;
        const long head = kv_head * query_groups + item % (int)query_groups;
        const long vector_index = item_row[slot] * (kv_heads * query_groups) + head;
        const float inverse = (total == 0.0f) ? 0.0f : 1.0f / total;
        #pragma unroll
        for (int tile = 0; tile < dim_tiles; tile++) {
            *(float2*)&out[vector_index * head_dim + dim_base + tile * 8 + quad * 2] = make_float2(
                accumulator[tile][slot * 2] * inverse, accumulator[tile][slot * 2 + 1] * inverse);
        }
    }
}
";
