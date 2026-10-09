use super::*;

/// The split-KV partial for [`CachedAttentionForm::TwoRangeDecodeSplit`]: one
/// threadgroup per `(query_row, query_head, split)` with `chunks` simdgroups,
/// llama.cpp's `kernel_flash_attn_ext_vec` grid (`(iq1, iq2, iwg)` at
/// `fa.metal:1224-1228`) for `n_q = 1`. Each simdgroup strides its split's
/// slice of the live key band and keeps a running `(max, sum, weighted V)` in
/// registers; the simdgroups merge in threadgroup memory. `u.splits == 1`
/// stores the normalized row into `out`; above that it stores the unnormalized
/// partial plus `(max, sum)` into the interleaved scratch layout
/// [`super::render_cached_attention_merge`] reads back.
///
/// The lane layout is llama.cpp's `NE`/`NL` split of a simdgroup
/// (`[attention_decode].keys_in_flight`): `32 / keys_in_flight` lanes span one
/// key's head dim, so a lane owns `head_dim / 8 / lanes` float4 of each K and
/// Q plane row and `head_dim / 4 / lanes` float4 of each V row. At the default
/// of one key in flight the whole simdgroup holds one score, so the online
/// softmax state is uniform: no per-block `simd_max`/`simd_sum`, no staged
/// scores, no per-block threadgroup round trip of the V accumulator, and each
/// K/V load instruction reads one contiguous 512 B line. Q is read from
/// device memory once per simdgroup into registers; the loads of
/// `keys_per_batch` keys (K planes and V row) are issued before any of them is
/// consumed, so a simdgroup pays one memory round trip per batch instead of
/// one per K-plane loop trip.
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
    cached_kv_codec: Option<Codec>,
) -> Result<String, EmitError> {
    let BoundOpKind::CachedAttention {
        kv_heads,
        query_groups,
        head_dim,
        scale,
        cached_lower_inclusive,
        new_upper_inclusive,
        new_key_rows,
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
    let (cached_kv_type, kv_pointers, key_load, value_load) = match cached_kv_codec {
        None => (element_type, PLAIN_KV_POINTERS, PLAIN_KEY_LOAD, PLAIN_VALUE_LOAD),
        Some(Codec::Float16) => ("half", HALF_KV_POINTERS, HALF_KEY_LOAD, HALF_VALUE_LOAD),
        Some(Codec::BFloat16) => ("ushort", BF16_KV_POINTERS, BF16_KEY_LOAD, BF16_VALUE_LOAD),
        Some(_) => {
            return Err(EmitError::CachedAttentionKvCodecNotSupported {
                node: resolved.node,
                reason: "the decode split reads cached K/V that is plain, Float16, or BFloat16",
            });
        }
    };
    let substitutions = [
        ("@KV_POINTERS@", kv_pointers.to_string()),
        ("@KEY_LOAD@", key_load.to_string()),
        ("@VALUE_LOAD@", value_load.to_string()),
        ("@KV@", cached_kv_type.to_string()),
        ("@ENTRY@", entry.to_string()),
        ("@T@", element_type.to_string()),
        ("@KV_HEADS@", kv_heads.to_string()),
        ("@QUERY_GROUPS@", query_groups.to_string()),
        ("@HEAD_DIM@", head_dim.to_string()),
        ("@SCALE@", msl_literal(*scale)),
        ("@CACHED_LOWER@", cached_lower),
        ("@NEW_UPPER@", format!("{new_upper_inclusive}L")),
        ("@NEW_KEY_ROWS@", new_key_rows.to_string()),
        ("@CAP@", decode_simdgroup_cap(*head_dim).to_string()),
        ("@LANES_PER_KEY@", decode_lanes_per_key().to_string()),
        (
            "@KEYS_PER_BATCH@",
            crate::sized::ATTENTION_DECODE_KEYS_PER_BATCH.to_string(),
        ),
    ];
    let mut source = String::new();
    preamble(&mut source, None);
    if cached_kv_codec == Some(Codec::BFloat16) {
        source.push_str(BF16_VECTOR_HELPER);
    }
    let mut body = DECODE_SPLIT_KERNEL.to_string();
    for (token, value) in &substitutions {
        body = body.replace(token, value);
    }
    source.push_str(&body);
    Ok(source)
}

const PLAIN_KV_POINTERS: &str = "device const @T@4* kr4 = (device const @T@4*)((cached ? in2 : in4) + kbase);
                device const @T@4* ki4 = (device const @T@4*)((cached ? in3 : in5) + kbase);
                device const @T@4* v4 = (device const @T@4*)((cached ? in6 : in7) + kbase * 2);";

const PLAIN_KEY_LOAD: &str =
    "key_real[step][slot] = float4(kr4[index]); key_imag[step][slot] = float4(ki4[index]);";

const PLAIN_VALUE_LOAD: &str = "value_row[step][slot] = float4(v4[index]);";

// the cached range is half-width and the new range is the op's own element
// type, so one pointer cannot select between them as the plain form does
const HALF_KV_POINTERS: &str = "device const @KV@4* kr4_cached = (device const @KV@4*)(in2 + kbase);
                device const @KV@4* ki4_cached = (device const @KV@4*)(in3 + kbase);
                device const @KV@4* v4_cached = (device const @KV@4*)(in6 + kbase * 2);
                device const @T@4* kr4_new = (device const @T@4*)(in4 + kbase);
                device const @T@4* ki4_new = (device const @T@4*)(in5 + kbase);
                device const @T@4* v4_new = (device const @T@4*)(in7 + kbase * 2);";

const HALF_KEY_LOAD: &str = "if (cached) { key_real[step][slot] = float4(kr4_cached[index]); key_imag[step][slot] = float4(ki4_cached[index]); } else { key_real[step][slot] = float4(kr4_new[index]); key_imag[step][slot] = float4(ki4_new[index]); }";

const HALF_VALUE_LOAD: &str =
    "if (cached) { value_row[step][slot] = float4(v4_cached[index]); } else { value_row[step][slot] = float4(v4_new[index]); }";

const BF16_KV_POINTERS: &str = "device const ushort4* kr4_cached = (device const ushort4*)(in2 + kbase);
                device const ushort4* ki4_cached = (device const ushort4*)(in3 + kbase);
                device const ushort4* v4_cached = (device const ushort4*)(in6 + kbase * 2);
                device const @T@4* kr4_new = (device const @T@4*)(in4 + kbase);
                device const @T@4* ki4_new = (device const @T@4*)(in5 + kbase);
                device const @T@4* v4_new = (device const @T@4*)(in7 + kbase * 2);";

const BF16_KEY_LOAD: &str = "if (cached) { key_real[step][slot] = omega_bf16x4_to_float4(kr4_cached[index]); key_imag[step][slot] = omega_bf16x4_to_float4(ki4_cached[index]); } else { key_real[step][slot] = float4(kr4_new[index]); key_imag[step][slot] = float4(ki4_new[index]); }";

const BF16_VALUE_LOAD: &str = "if (cached) { value_row[step][slot] = omega_bf16x4_to_float4(v4_cached[index]); } else { value_row[step][slot] = float4(v4_new[index]); }";

const BF16_VECTOR_HELPER: &str = "static inline float4 omega_bf16x4_to_float4(ushort4 value) { return float4(as_type<float>((uint)value.x << 16u), as_type<float>((uint)value.y << 16u), as_type<float>((uint)value.z << 16u), as_type<float>((uint)value.w << 16u)); }\n";

const DECODE_SPLIT_KERNEL: &str =r#"struct Uniforms { long total_elements; long context_chunks; long splits; };

#define OMEGA_UNROLL _Pragma("clang loop unroll(full)")

kernel void @ENTRY@(device const @T@* in0 [[buffer(0)]], device const @T@* in1 [[buffer(1)]], device const @KV@* in2 [[buffer(2)]], device const @KV@* in3 [[buffer(3)]], device const @T@* in4 [[buffer(4)]], device const @T@* in5 [[buffer(5)]], device const @KV@* in6 [[buffer(6)]], device const @T@* in7 [[buffer(7)]], device const @T@* in8 [[buffer(8)]], device @T@* out [[buffer(9)]], constant Uniforms& u [[buffer(10)]], uint tgid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]], ushort simdgroup_slot [[simdgroup_index_in_threadgroup]]) {
    long splits = u.splits;
    if ((long)tgid >= u.total_elements * splits) { return; }
    constexpr long kv_heads = @KV_HEADS@; constexpr long query_groups = @QUERY_GROUPS@; constexpr long head_dim = @HEAD_DIM@; constexpr float scale = @SCALE@; constexpr long cached_lower = @CACHED_LOWER@; constexpr long new_upper = @NEW_UPPER@; constexpr long new_key_rows = @NEW_KEY_ROWS@; constexpr long cap = @CAP@;
    constexpr short lanes_per_key = @LANES_PER_KEY@; constexpr short keys_in_flight = 32 / lanes_per_key; constexpr short batch = @KEYS_PER_BATCH@;
    constexpr short plane_vectors = head_dim / 8; constexpr short row_vectors = head_dim / 4;
    constexpr short plane_slots = (plane_vectors + lanes_per_key - 1) / lanes_per_key; constexpr short row_slots = (row_vectors + lanes_per_key - 1) / lanes_per_key;
    long cached_key_rows = (long)in8[0];
    long chunks = u.context_chunks;
    long chunk = (long)simdgroup_slot;
    long split = (long)tgid % splits;
    long query_head = ((long)tgid / splits) % (kv_heads * query_groups);
    long query_row = (long)tgid / (splits * kv_heads * query_groups);
    long kv_head = query_head / query_groups;
    long query_index = query_row * (kv_heads * query_groups) + query_head;
    long qbase = query_index * (head_dim / 2);
    long last_key = cached_key_rows + new_key_rows - 1L;
    long first_key = max(0L, cached_key_rows + cached_lower + query_row);
    long band = last_key + 1L - first_key;
    long slice_len = (band + splits - 1L) / splits;
    long lo = first_key + split * slice_len;
    long hi = min(lo + slice_len, last_key + 1L);
    threadgroup float shared_m[cap]; threadgroup float shared_l[cap]; threadgroup float4 shared_o[cap * row_vectors];
    short ty = (short)(lane / lanes_per_key); short tx = (short)(lane % lanes_per_key);
    float maximum = -INFINITY; float sum = 0.0f;
    float4 weighted[row_slots];
    OMEGA_UNROLL for (short slot = 0; slot < row_slots; slot++) { weighted[slot] = float4(0.0f); }
    if (chunk < chunks) {
        device const @T@4* qr4 = (device const @T@4*)(in0 + qbase);
        device const @T@4* qi4 = (device const @T@4*)(in1 + qbase);
        float4 query_real[plane_slots]; float4 query_imag[plane_slots];
        OMEGA_UNROLL for (short slot = 0; slot < plane_slots; slot++) {
            short index = tx + slot * lanes_per_key;
            query_real[slot] = float4(0.0f); query_imag[slot] = float4(0.0f);
            if (index < plane_vectors) { query_real[slot] = float4(qr4[index]); query_imag[slot] = float4(qi4[index]); }
        }
        long slice_start = lo + chunk;
        long num_local_keys = (slice_start < hi) ? ((hi - 1L - slice_start) / chunks) + 1L : 0L;
        for (long batch_start = 0L; batch_start < num_local_keys; batch_start += (long)batch * (long)keys_in_flight) {
            float4 key_real[batch][plane_slots]; float4 key_imag[batch][plane_slots]; float4 value_row[batch][row_slots];
            bool valid[batch];
            OMEGA_UNROLL for (short step = 0; step < batch; step++) {
                long local_index = batch_start + (long)step * (long)keys_in_flight + (long)ty;
                long key = slice_start + local_index * chunks;
                bool cached = key < cached_key_rows; long new_index = key - cached_key_rows;
                bool live = local_index < num_local_keys;
                if (live) {
                    long relative = (cached ? key - cached_key_rows : new_index) - query_row;
                    if (cached && relative < cached_lower) { live = false; }
                    if (!cached && relative > new_upper) { live = false; }
                }
                valid[step] = live;
                long kbase = (cached ? key : new_index) * (kv_heads * (head_dim / 2)) + kv_head * (head_dim / 2);
                @KV_POINTERS@
                OMEGA_UNROLL for (short slot = 0; slot < plane_slots; slot++) {
                    short index = tx + slot * lanes_per_key;
                    key_real[step][slot] = float4(0.0f); key_imag[step][slot] = float4(0.0f);
                    if (live && index < plane_vectors) { @KEY_LOAD@ }
                }
                OMEGA_UNROLL for (short slot = 0; slot < row_slots; slot++) {
                    short index = tx + slot * lanes_per_key;
                    value_row[step][slot] = float4(0.0f);
                    if (live && index < row_vectors) { @VALUE_LOAD@ }
                }
            }
            float score[batch];
            OMEGA_UNROLL for (short step = 0; step < batch; step++) {
                float partial_score = 0.0f;
                OMEGA_UNROLL for (short slot = 0; slot < plane_slots; slot++) {
                    partial_score += dot(key_real[step][slot], query_real[slot]);
                    partial_score += dot(key_imag[step][slot], query_imag[slot]);
                }
                if (16 < lanes_per_key) { partial_score += simd_shuffle_xor(partial_score, (ushort)16); }
                if (8 < lanes_per_key) { partial_score += simd_shuffle_xor(partial_score, (ushort)8); }
                if (4 < lanes_per_key) { partial_score += simd_shuffle_xor(partial_score, (ushort)4); }
                if (2 < lanes_per_key) { partial_score += simd_shuffle_xor(partial_score, (ushort)2); }
                if (1 < lanes_per_key) { partial_score += simd_shuffle_xor(partial_score, (ushort)1); }
                score[step] = valid[step] ? partial_score * scale : -INFINITY;
            }
            float batch_max = score[0];
            OMEGA_UNROLL for (short step = 1; step < batch; step++) { batch_max = max(batch_max, score[step]); }
            if (keys_in_flight > 1) { batch_max = simd_max(batch_max); }
            float next_max = max(maximum, batch_max);
            float rescale = (maximum == -INFINITY) ? 0.0f : exp(maximum - next_max);
            sum *= rescale;
            OMEGA_UNROLL for (short slot = 0; slot < row_slots; slot++) { weighted[slot] *= rescale; }
            OMEGA_UNROLL for (short step = 0; step < batch; step++) {
                float weight = (score[step] == -INFINITY) ? 0.0f : exp(score[step] - next_max);
                sum += weight;
                OMEGA_UNROLL for (short slot = 0; slot < row_slots; slot++) { weighted[slot] += value_row[step][slot] * weight; }
            }
            maximum = next_max;
        }
        if (lanes_per_key <= 8) {
            sum += simd_shuffle_xor(sum, (ushort)8);
            OMEGA_UNROLL for (short slot = 0; slot < row_slots; slot++) { weighted[slot] += simd_shuffle_xor(weighted[slot], (ushort)8); }
        }
        if (lanes_per_key <= 16) {
            sum += simd_shuffle_xor(sum, (ushort)16);
            OMEGA_UNROLL for (short slot = 0; slot < row_slots; slot++) { weighted[slot] += simd_shuffle_xor(weighted[slot], (ushort)16); }
        }
    }
    if (lane == 0) { shared_m[chunk] = maximum; shared_l[chunk] = sum; }
    if (ty == 0) {
        OMEGA_UNROLL for (short slot = 0; slot < row_slots; slot++) {
            short index = tx + slot * lanes_per_key;
            if (index < row_vectors) { shared_o[chunk * row_vectors + index] = weighted[slot]; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (chunk == 0L) {
        float merged_max = -INFINITY;
        for (long c = 0; c < chunks; c++) { merged_max = max(merged_max, shared_m[c]); }
        float chunk_scale[cap];
        float merged_sum = 0.0f;
        OMEGA_UNROLL for (long c = 0; c < cap; c++) {
            bool live_chunk = c < chunks && shared_m[c] != -INFINITY;
            chunk_scale[c] = live_chunk ? exp(shared_m[c] - merged_max) : 0.0f;
            merged_sum += live_chunk ? shared_l[c] * chunk_scale[c] : 0.0f;
        }
        device float* attn_scratch = (device float*)out;
        long value_base = query_index * row_vectors;
        long stats_index = u.total_elements * head_dim * splits + (query_index * splits + split) * 2L;
        if (splits != 1L && lane == 0) { attn_scratch[stats_index] = merged_max; attn_scratch[stats_index + 1L] = merged_sum; }
        if (ty == 0) {
            OMEGA_UNROLL for (short slot = 0; slot < row_slots; slot++) {
                short index = tx + slot * lanes_per_key;
                if (index < row_vectors) {
                    float4 merged = float4(0.0f);
                    OMEGA_UNROLL for (long c = 0; c < cap; c++) { merged += (c < chunks) ? shared_o[c * row_vectors + index] * chunk_scale[c] : float4(0.0f); }
                    if (splits == 1L) {
                        ((device @T@4*)(out + query_index * head_dim))[index] = @T@4(merged_sum == 0.0f ? float4(0.0f) : merged / merged_sum);
                    } else {
                        ((device float4*)attn_scratch)[(value_base + index) * splits + split] = merged;
                    }
                }
            }
        }
    }
}
"#;
