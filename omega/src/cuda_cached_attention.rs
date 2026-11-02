use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_tensor::{BoundOp, BoundOpKind, DType, NumericPolicy};

use crate::cuda::mma_gemm::MMA_INSTRUCTION;
use crate::cuda::{
    CudaAttentionClass, cuda_cached_attention_class, cuda_float_literal, preamble,
};
use crate::cuda_cached_attention_mma::{MMA_ATTENTION_BODY, MMA_ATTENTION_HELPERS};
use crate::cuda_cached_attention_tiled::{
    TILED_ATTENTION_BODY, cached_attention_tile_rows, cached_attention_uses_mma,
};
use crate::error::EmitError;

pub(crate) fn cached_attention_entry_name(
    resolved: &BoundOp,
    class: CudaAttentionClass,
    split_chunks: Option<u64>,
    numeric_policy: NumericPolicy,
) -> Result<String, EmitError> {
    let BoundOpKind::CachedAttention {
        kv_heads,
        query_groups,
        head_dim,
        rotary_dim,
        new_key_rows,
        scale,
        cached_lower_inclusive,
        new_upper_inclusive,
        ..
    } = &resolved.kind
    else {
        return Err(kind_mismatch(resolved));
    };
    let rotary_token = if rotary_dim < head_dim {
        format!("_r{rotary_dim}")
    } else {
        String::new()
    };
    let mma_token = if cached_attention_uses_mma(&resolved.kind, numeric_policy) { "_mma16x8x8h" } else { "" };
    let tile_token = cached_attention_tile_rows(&resolved.kind)
        .map_or_else(String::new, |tile_rows| format!("_tq{tile_rows}{mma_token}"));
    let split_token = split_chunks.map_or_else(String::new, |chunks| format!("_sk{chunks}"));
    Ok(format!(
        "omega_cuda_cached_attention_{}_h{kv_heads}_g{query_groups}_d{head_dim}{rotary_token}_n{new_key_rows}{tile_token}{split_token}_s{:08x}_l{}_u{}",
        class_tag(class),
        scale.to_bits(),
        signed_name_part(*cached_lower_inclusive),
        signed_name_part(*new_upper_inclusive),
    ))
}

pub(crate) fn render_cached_attention_cuda(
    resolved: &BoundOp,
    entry: &str,
    numeric_policy: NumericPolicy,
) -> Result<String, EmitError> {
    let helpers = if cached_attention_uses_mma(&resolved.kind, numeric_policy) {
        format!("{MMA_INSTRUCTION}{MMA_ATTENTION_HELPERS}")
    } else {
        String::new()
    };
    render_attention(resolved, entry, "", &unsplit_tail(resolved, numeric_policy), &helpers)
}

fn unsplit_tail(resolved: &BoundOp, numeric_policy: NumericPolicy) -> String {
    let body = if cached_attention_uses_mma(&resolved.kind, numeric_policy) { MMA_ATTENTION_BODY } else { TILED_ATTENTION_BODY };
    match cached_attention_tile_rows(&resolved.kind) {
        Some(tile_rows) => format!("    constexpr int tile_rows = {tile_rows};\n{body}"),
        None => String::from(CACHED_ATTENTION_BODY),
    }
}

pub(crate) fn render_attention(
    resolved: &BoundOp,
    entry: &str,
    extra_param: &str,
    tail: &str,
    helpers: &str,
) -> Result<String, EmitError> {
    let Some(class) = cuda_cached_attention_class(&resolved.kind) else {
        return Err(EmitError::CudaUnsupportedOpKind {
            node: resolved.node,
            kind: "cached_attention",
        });
    };
    let BoundOpKind::CachedAttention {
        operands,
        cached_key_rows,
        new_key_rows,
        kv_heads,
        query_groups,
        head_dim,
        rotary_dim,
        scale,
        cached_lower_inclusive,
        new_upper_inclusive,
        ..
    } = &resolved.kind
    else {
        return Err(kind_mismatch(resolved));
    };
    if resolved.dtype != DType::FLOAT32 {
        return Err(EmitError::CudaUnsupportedOpKind {
            node: resolved.node,
            kind: "cached_attention_non_f32",
        });
    }
    let params: String = (0..operands.len())
        .map(|index| format!("const float* __restrict__ in{index}"))
        .collect::<Vec<String>>()
        .join(",\n    ");
    let cached_rows_decl = match class {
        CudaAttentionClass::CachedBound => String::from("const long cached_key_rows = (long)in8[0];"),
        CudaAttentionClass::Static => format!("constexpr long cached_key_rows = {cached_key_rows}L;"),
    };
    let cached_lower = match *cached_lower_inclusive {
        i64::MIN => String::from("-9223372036854775807L"),
        bound => format!("{bound}L"),
    };
    let pass_planes = if rotary_dim < head_dim {
        let first = match class {
            CudaAttentionClass::CachedBound => 9,
            CudaAttentionClass::Static => 8,
        };
        format!(
            "const float* __restrict__ pass_query = in{first};\n\
             \x20   const float* __restrict__ pass_cached_key = in{};\n\
             \x20   const float* __restrict__ pass_new_key = in{};\n",
            first + 1,
            first + 2
        )
    } else {
        String::from(
            "const float* __restrict__ pass_query = in0;\n\
             \x20   const float* __restrict__ pass_cached_key = in0;\n\
             \x20   const float* __restrict__ pass_new_key = in0;\n",
        )
    };
    let mut source = String::new();
    preamble(&mut source, "float");
    source.push_str(helpers);
    source.push_str("struct Uniforms { long total_elements; };\n\n");
    source.push_str(&format!(
        "extern \"C\" __global__ void {entry}(\n    {params},\n    float* __restrict__ out,\n    const Uniforms* __restrict__ u_ptr{extra_param}) {{\n"
    ));
    source.push_str(&format!(
        "    constexpr long kv_heads = {kv_heads};\n\
         \x20   constexpr long query_groups = {query_groups};\n\
         \x20   constexpr long head_dim = {head_dim};\n\
         \x20   constexpr long rotary_dim = {rotary_dim};\n\
         \x20   constexpr long pair_dim = rotary_dim / 2;\n\
         \x20   constexpr long pass_dim = head_dim - rotary_dim;\n\
         \x20   constexpr long new_key_rows = {new_key_rows}L;\n\
         \x20   constexpr long new_upper = {new_upper_inclusive}L;\n\
         \x20   constexpr long cached_lower = {cached_lower};\n\
         \x20   constexpr float scale = {scale};\n\
         \x20   {cached_rows_decl}\n\
         \x20   {pass_planes}",
        scale = cuda_float_literal(*scale),
    ));
    source.push_str(tail);
    Ok(source)
}

const CACHED_ATTENTION_BODY: &str = "\
    const float negative_infinity = -__int_as_float(0x7f800000);
    constexpr int warps = 4;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5u;
    const long vector_index = (long)blockIdx.x;
    if (vector_index >= u_ptr->total_elements) { return; }
    const long query_row = vector_index / (kv_heads * query_groups);
    const long query_head = vector_index % (kv_heads * query_groups);
    const long kv_head = query_head / query_groups;
    const long qbase = query_row * (kv_heads * query_groups * pair_dim) + query_head * pair_dim;
    const long qpass = query_row * (kv_heads * query_groups * pass_dim) + query_head * pass_dim;
    const long last_key = cached_key_rows + new_key_rows - 1L;
    float maximum = negative_infinity;
    float sum = 0.0f;
    float weighted[(head_dim + 31) / 32];
    for (long slot = 0; slot < (head_dim + 31) / 32; slot++) { weighted[slot] = 0.0f; }
    for (long key = (long)warp; key <= last_key; key += warps) {
        const bool cached = key < cached_key_rows;
        const long new_index = key - cached_key_rows;
        const long relative = (cached ? key - cached_key_rows : new_index) - query_row;
        if (cached && relative < cached_lower) { continue; }
        if (!cached && relative > new_upper) { continue; }
        const long key_row = cached ? key : new_index;
        const long kbase = key_row * (kv_heads * pair_dim) + kv_head * pair_dim;
        const long kpass = key_row * (kv_heads * pass_dim) + kv_head * pass_dim;
        const long vbase = key_row * (kv_heads * head_dim) + kv_head * head_dim;
        float partial = 0.0f;
        for (long pair = (long)lane; pair < pair_dim; pair += 32L) {
            partial += in0[qbase + pair] * (cached ? in2[kbase + pair] : in4[kbase + pair]);
            partial += in1[qbase + pair] * (cached ? in3[kbase + pair] : in5[kbase + pair]);
        }
        for (long column = (long)lane; column < pass_dim; column += 32L) {
            partial += pass_query[qpass + column] * (cached ? pass_cached_key[kpass + column] : pass_new_key[kpass + column]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) { partial += __shfl_xor_sync(0xffffffffu, partial, offset); }
        const float score = partial * scale;
        const float next_max = fmaxf(maximum, score);
        const float weight = expf(score - next_max);
        const float rescale = (maximum == negative_infinity) ? 0.0f : expf(maximum - next_max);
        sum = sum * rescale + weight;
        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {
            const long local_dimension = dimension / 32L;
            weighted[local_dimension] = weighted[local_dimension] * rescale
                + weight * (cached ? in6[vbase + dimension] : in7[vbase + dimension]);
        }
        maximum = next_max;
    }
    __shared__ float shared_maximum[warps];
    __shared__ float shared_sum[warps];
    __shared__ float shared_weighted[warps * head_dim];
    for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {
        shared_weighted[warp * head_dim + dimension] = weighted[dimension / 32L];
    }
    if (lane == 0u) {
        shared_maximum[warp] = maximum;
        shared_sum[warp] = sum;
    }
    __syncthreads();
    float global_maximum = negative_infinity;
    for (int source = 0; source < warps; source++) { global_maximum = fmaxf(global_maximum, shared_maximum[source]); }
    float total = 0.0f;
    for (int source = 0; source < warps; source++) {
        total += (shared_maximum[source] == negative_infinity) ? 0.0f : shared_sum[source] * expf(shared_maximum[source] - global_maximum);
    }
    for (long dimension = (long)threadIdx.x; dimension < head_dim; dimension += (long)blockDim.x) {
        float merged = 0.0f;
        for (int source = 0; source < warps; source++) {
            merged += (shared_maximum[source] == negative_infinity) ? 0.0f : shared_weighted[source * head_dim + dimension] * expf(shared_maximum[source] - global_maximum);
        }
        out[vector_index * head_dim + dimension] = (total == 0.0f) ? 0.0f : merged / total;
    }
}
";

fn kind_mismatch(resolved: &BoundOp) -> EmitError {
    EmitError::RenderKindMismatch {
        node: resolved.node,
        expected: "cached_attention",
        found: resolved.kind.name(),
    }
}

const fn class_tag(class: CudaAttentionClass) -> &'static str {
    match class {
        CudaAttentionClass::Static => "static",
        CudaAttentionClass::CachedBound => "cached_bound",
    }
}

fn signed_name_part(value: i64) -> String {
    if value < 0 {
        format!("m{}", value.unsigned_abs())
    } else {
        format!("{value}")
    }
}
