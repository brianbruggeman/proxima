use super::*;

/// The split-KV partial for [`CachedAttentionForm::TwoRangeDecodeSplit`]: one
/// threadgroup per `(query_row, query_head, split)` with `chunks` simdgroups,
/// llama.cpp's `kernel_flash_attn_ext_vec` grid (`(iq1, iq2, iwg)` at
/// `fa.metal:1224-1228`) for `n_q = 1`. Each threadgroup walks its own slice
/// of the live key band in 32-key blocks, reading the even/odd K planes
/// straight from device memory (`float4` loads, an 8-lane shuffle reduce per
/// key), folds an online softmax, and merges its simdgroups in threadgroup
/// memory. `u.splits == 1` stores the normalized row into `out`; above that
/// it stores the unnormalized partial plus `(max, sum)` into the interleaved
/// scratch layout [`super::render_cached_attention_merge`] reads back.
///
/// Composes the block-staged body `render_cached_attention`'s single-range
/// path renders (`TreeReduce`, `NumericRewrite` admitted by the form's
/// classification), with `query_groups` moved out of the threadgroup into the
/// grid -- the per-query-head regime `cached_attention_per_query_head_grid`
/// already names below its knee. `splits` and `chunks` are runtime uniforms,
/// so one compiled kernel serves every `kv-capacity-bucket`.
pub(super) fn render_cached_attention_decode_split(
    resolved: &BoundOp,
    entry: &str,
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
            expected: "cached_attention_decode_split",
            found: resolved.kind.name(),
        });
    };
    validate_attention_split_limit(resolved.node, crate::sized::ATTENTION_SPLIT_MAX)?;
    let element_type = type_token(resolved.node, resolved.dtype)?;
    let cached_lower = if *cached_lower_inclusive == i64::MIN {
        "-9223372036854775807L".to_string()
    } else {
        format!("{cached_lower_inclusive}L")
    };
    let substitutions = [
        ("@ENTRY@", entry.to_string()),
        ("@T@", element_type.to_string()),
        ("@KV_HEADS@", kv_heads.to_string()),
        ("@QUERY_GROUPS@", query_groups.to_string()),
        ("@HEAD_DIM@", head_dim.to_string()),
        ("@V_REGISTERS@", (head_dim / 8 / 4).to_string()),
        ("@SCALE@", msl_literal(*scale)),
        ("@CACHED_LOWER@", cached_lower),
        ("@NEW_UPPER@", format!("{new_upper_inclusive}L")),
        (
            "@CAP@",
            effective_context_chunk_cap(1, *head_dim).to_string(),
        ),
        (
            "@BLOCK_WIDTH@",
            crate::sized::ATTENTION_BLOCK_WIDTH.to_string(),
        ),
    ];
    let mut source = String::new();
    preamble(&mut source, false);
    let mut body = DECODE_SPLIT_KERNEL.to_string();
    for (token, value) in &substitutions {
        body = body.replace(token, value);
    }
    source.push_str(&body);
    Ok(source)
}

const DECODE_SPLIT_KERNEL: &str = r#"struct Uniforms { long total_elements; long context_chunks; long splits; };

kernel void @ENTRY@(device const @T@* in0 [[buffer(0)]], device const @T@* in1 [[buffer(1)]], device const @T@* in2 [[buffer(2)]], device const @T@* in3 [[buffer(3)]], device const @T@* in4 [[buffer(4)]], device const @T@* in5 [[buffer(5)]], device const @T@* in6 [[buffer(6)]], device const @T@* in7 [[buffer(7)]], device const @T@* in8 [[buffer(8)]], device @T@* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]], uint tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]], ushort simdgroup_slot [[simdgroup_index_in_threadgroup]]) {
    long splits = u.splits;
    if ((long)tgid >= u.total_elements * splits) { return; }
    constexpr long kv_heads = @KV_HEADS@; constexpr long query_groups = @QUERY_GROUPS@; constexpr long head_dim = @HEAD_DIM@; constexpr float scale = @SCALE@; constexpr long cached_lower = @CACHED_LOWER@; constexpr long new_upper = @NEW_UPPER@; constexpr long new_key_rows = 1; constexpr long cap = @CAP@; constexpr long block_width = @BLOCK_WIDTH@; constexpr long v_registers = @V_REGISTERS@;
    long cached_key_rows = (long)in8[0];
    long chunks = u.context_chunks;
    long chunk = (long)simdgroup_slot;
    long split = (long)tgid % splits;
    long query_head = ((long)tgid / splits) % (kv_heads * query_groups);
    long query_row = (long)tgid / (splits * kv_heads * query_groups);
    long kv_head = query_head / query_groups;
    long query_index = query_row * (kv_heads * query_groups) + query_head;
    long qbase = query_index * (head_dim / 2);
    float maximum = -INFINITY; float sum = 0.0f; float weighted[(head_dim + 31) / 32];
    for (long slot = 0; slot < (head_dim + 31) / 32; slot++) { weighted[slot] = 0.0f; }
    long last_key = cached_key_rows + new_key_rows - 1L;
    long first_key = max(0L, cached_key_rows + cached_lower + query_row);
    long band = last_key + 1L - first_key;
    long slice_len = (band + splits - 1L) / splits;
    long lo = first_key + split * slice_len;
    long hi = min(lo + slice_len, last_key + 1L);
    threadgroup float shared_m[cap]; threadgroup float shared_l[cap]; threadgroup float shared_o[cap * head_dim]; threadgroup float ss[cap * block_width];
    short ty = (short)(lane / 8); short tx = (short)(lane % 8);
    if (chunk < chunks) {
    long slice_start = lo + chunk;
    long num_local_keys = (slice_start < hi) ? ((hi - 1L - slice_start) / chunks) + 1L : 0L;
    for (long block_start = 0L; block_start < num_local_keys; block_start += block_width) {
        for (long cc = 0L; cc < block_width / 4L; cc++) {
            long local_index = block_start + 4L * cc + (long)ty;
            bool valid = local_index < num_local_keys;
            long key = slice_start + local_index * chunks;
            bool cached = key < cached_key_rows; long new_index = key - cached_key_rows;
            if (valid) {
                long relative = (cached ? key - cached_key_rows : new_index) - query_row;
                if (cached && relative < cached_lower) { valid = false; }
                if (!cached && relative > new_upper) { valid = false; }
            }
            long kbase = (cached ? key : new_index) * (kv_heads * (head_dim / 2)) + kv_head * (head_dim / 2);
            float partial_score = 0.0f;
            if (valid) {
                device const @T@4* qr4 = (device const @T@4*)(in0 + qbase);
                device const @T@4* qi4 = (device const @T@4*)(in1 + qbase);
                device const @T@4* kr4 = (device const @T@4*)((cached ? in2 : in4) + kbase);
                device const @T@4* ki4 = (device const @T@4*)((cached ? in3 : in5) + kbase);
                for (short index = tx; index < (short)((head_dim / 2) / 4L); index += 8) {
                    partial_score += dot(kr4[index], qr4[index]);
                    partial_score += dot(ki4[index], qi4[index]);
                }
            }
            partial_score += simd_shuffle_down(partial_score, 4);
            partial_score += simd_shuffle_down(partial_score, 2);
            partial_score += simd_shuffle_down(partial_score, 1);
            if (tx == 0) { ss[chunk * block_width + 4L * cc + (long)ty] = valid ? partial_score * scale : -INFINITY; }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (long sub = 0L; sub < block_width; sub += 32L) {
            long lane_index = sub + (long)lane;
            float raw_score = ss[chunk * block_width + lane_index];
            float next_max = simd_max(max(maximum, raw_score));
            float rescale = (maximum == -INFINITY) ? 0.0f : exp(maximum - next_max);
            float weight = exp(raw_score - next_max);
            sum = sum * rescale + simd_sum(weight);
            ss[chunk * block_width + lane_index] = weight;
            for (long slot = 0L; slot < (head_dim + 31) / 32; slot++) { weighted[slot] *= rescale; }
            maximum = next_max;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            {
                float4 v_acc[v_registers];
                for (long register_index = 0L; register_index < v_registers; register_index++) { v_acc[register_index] = float4(0.0f); }
                for (long cc4 = 0L; cc4 < 8L; cc4++) {
                    long local_index = block_start + sub + 4L * cc4 + (long)ty;
                    bool valid = local_index < min(block_start + sub + 32L, num_local_keys);
                    long key = slice_start + local_index * chunks;
                    bool cached = key < cached_key_rows; long new_index = key - cached_key_rows;
                    long kbase = (cached ? key : new_index) * (kv_heads * (head_dim / 2)) + kv_head * (head_dim / 2);
                    float key_weight = valid ? ss[chunk * block_width + (local_index - block_start)] : 0.0f;
                    device const @T@4* v4 = (device const @T@4*)((cached ? in6 : in7) + kbase * 2);
                    for (long register_index = 0L; register_index < v_registers; register_index++) {
                        v_acc[register_index] += valid ? float4(v4[(long)tx + 8L * register_index]) * key_weight : float4(0.0f);
                    }
                }
                for (long register_index = 0L; register_index < v_registers; register_index++) {
                    v_acc[register_index] += simd_shuffle_xor(v_acc[register_index], 8);
                    v_acc[register_index] += simd_shuffle_xor(v_acc[register_index], 16);
                }
                if (ty == 0) {
                    threadgroup float4* shared_o4 = (threadgroup float4*)(shared_o + chunk * head_dim);
                    for (long register_index = 0L; register_index < v_registers; register_index++) { shared_o4[(long)tx + 8L * register_index] = v_acc[register_index]; }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {
                    weighted[dimension / 32L] += shared_o[chunk * head_dim + dimension];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    }
    if (lane == 0) { shared_m[chunk] = maximum; shared_l[chunk] = sum; }
    for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) { shared_o[chunk * head_dim + dimension] = weighted[dimension / 32L]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (chunk == 0L) {
        float merged_max = -INFINITY;
        for (long c = 0; c < chunks; c++) { merged_max = max(merged_max, shared_m[c]); }
        float merged_sum = 0.0f;
        for (long c = 0; c < chunks; c++) {
            float partial_max = shared_m[c];
            float rescale = (partial_max == -INFINITY) ? 0.0f : exp(partial_max - merged_max);
            merged_sum += shared_l[c] * rescale;
        }
        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {
            float acc = 0.0f;
            for (long c = 0; c < chunks; c++) {
                float partial_max = shared_m[c];
                float rescale = (partial_max == -INFINITY) ? 0.0f : exp(partial_max - merged_max);
                acc += shared_o[c * head_dim + dimension] * rescale;
            }
            weighted[dimension / 32L] = acc;
        }
        if (splits == 1L) {
            for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {
                out[query_index * head_dim + dimension] = (@T@)(merged_sum == 0.0f ? 0.0f : weighted[dimension / 32L] / merged_sum);
            }
        } else {
            device float* attn_scratch = (device float*)out;
            long value_base = query_index * (head_dim / 4L);
            long stats_index = u.total_elements * head_dim * splits + (query_index * splits + split) * 2L;
            if (lane == 0) { attn_scratch[stats_index] = merged_max; attn_scratch[stats_index + 1L] = merged_sum; }
            for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {
                attn_scratch[((value_base + (dimension >> 2)) * splits + split) * 4L + (dimension & 3L)] = weighted[dimension / 32L];
            }
        }
    }
}
"#;
