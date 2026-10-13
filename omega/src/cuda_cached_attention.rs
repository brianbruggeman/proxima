use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_tensor::{BoundOp, BoundOpKind, DType};

use crate::cuda::{
    CudaAttentionClass, cuda_cached_attention_class, cuda_float_literal, preamble,
};
use crate::error::EmitError;

pub(crate) fn cached_attention_entry_name(
    resolved: &BoundOp,
    class: CudaAttentionClass,
) -> Result<String, EmitError> {
    let BoundOpKind::CachedAttention {
        kv_heads,
        query_groups,
        head_dim,
        new_key_rows,
        scale,
        cached_lower_inclusive,
        new_upper_inclusive,
        ..
    } = &resolved.kind
    else {
        return Err(kind_mismatch(resolved));
    };
    Ok(format!(
        "omega_cuda_cached_attention_{}_h{kv_heads}_g{query_groups}_d{head_dim}_n{new_key_rows}_s{:08x}_l{}_u{}",
        class_tag(class),
        scale.to_bits(),
        signed_name_part(*cached_lower_inclusive),
        signed_name_part(*new_upper_inclusive),
    ))
}

pub(crate) fn render_cached_attention_cuda(
    resolved: &BoundOp,
    entry: &str,
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
        scale,
        cached_lower_inclusive,
        new_upper_inclusive,
        ..
    } = &resolved.kind
    else {
        return Err(kind_mismatch(resolved));
    };
    if resolved.dtype != DType::Float32 {
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
    let mut source = String::new();
    preamble(&mut source, "float");
    source.push_str("struct Uniforms { long total_elements; };\n\n");
    source.push_str(&format!(
        "extern \"C\" __global__ void {entry}(\n    {params},\n    float* __restrict__ out,\n    const Uniforms* __restrict__ u_ptr) {{\n"
    ));
    source.push_str(&format!(
        "    constexpr long kv_heads = {kv_heads};\n\
         \x20   constexpr long query_groups = {query_groups};\n\
         \x20   constexpr long head_dim = {head_dim};\n\
         \x20   constexpr long pair_dim = head_dim / 2;\n\
         \x20   constexpr long new_key_rows = {new_key_rows}L;\n\
         \x20   constexpr long new_upper = {new_upper_inclusive}L;\n\
         \x20   constexpr long cached_lower = {cached_lower};\n\
         \x20   constexpr float scale = {scale};\n\
         \x20   {cached_rows_decl}\n",
        scale = cuda_float_literal(*scale),
    ));
    source.push_str(CACHED_ATTENTION_BODY);
    Ok(source)
}

const CACHED_ATTENTION_BODY: &str = "\
    const float negative_infinity = -__int_as_float(0x7f800000);
    const unsigned int lane = threadIdx.x & 31u;
    const long vector_index = ((long)blockIdx.x * blockDim.x + threadIdx.x) / 32L;
    if (vector_index >= u_ptr->total_elements) { return; }
    const long query_row = vector_index / (kv_heads * query_groups);
    const long query_head = vector_index % (kv_heads * query_groups);
    const long kv_head = query_head / query_groups;
    const long qbase = query_row * (kv_heads * query_groups * pair_dim) + query_head * pair_dim;
    const long last_key = cached_key_rows + new_key_rows - 1L;
    float maximum = negative_infinity;
    float sum = 0.0f;
    float weighted[(head_dim + 31) / 32];
    for (long slot = 0; slot < (head_dim + 31) / 32; slot++) { weighted[slot] = 0.0f; }
    for (long key = 0; key <= last_key; key++) {
        const bool cached = key < cached_key_rows;
        const long new_index = key - cached_key_rows;
        const long relative = (cached ? key - cached_key_rows : new_index) - query_row;
        if (cached && relative < cached_lower) { continue; }
        if (!cached && relative > new_upper) { continue; }
        const long kbase = (cached ? key : new_index) * (kv_heads * pair_dim) + kv_head * pair_dim;
        float partial = 0.0f;
        for (long pair = (long)lane; pair < pair_dim; pair += 32L) {
            partial += in0[qbase + pair] * (cached ? in2[kbase + pair] : in4[kbase + pair]);
            partial += in1[qbase + pair] * (cached ? in3[kbase + pair] : in5[kbase + pair]);
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
                + weight * (cached ? in6[kbase * 2 + dimension] : in7[kbase * 2 + dimension]);
        }
        maximum = next_max;
    }
    for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {
        out[vector_index * head_dim + dimension] = (sum == 0.0f) ? 0.0f : weighted[dimension / 32L] / sum;
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
