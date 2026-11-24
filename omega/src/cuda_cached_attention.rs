use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_tensor::{BoundOp, BoundOpKind};
use proxima_numeric::codec::Codec;
use proxima_numeric::{DType, NumericPolicy};

use crate::cuda_layout::layout_load_cuda;
use crate::identity::description_fingerprint;
use crate::msl::layout_render::readable_scalar_cache;

use crate::cuda::mma_gemm::MMA_INSTRUCTION;
use crate::cuda::{
    CudaAttentionClass, WORD_C, cuda_cached_attention_class, cuda_float_literal, preamble, type_token,
};
use crate::cuda_cached_attention_mma::{MMA_ATTENTION_BODY, MMA_ATTENTION_HELPERS};
use crate::cuda_cached_attention_tiled::{
    TILED_ATTENTION_BODY, cached_attention_tile_rows, cached_attention_uses_mma,
};
use crate::error::EmitError;

/// Whether the kernel reads `new_key_rows`, `new_upper` and `cached_lower` from its uniforms: every
/// multi-row form, so one kernel serves every chunk width instead of one per width.
pub(crate) fn cached_attention_runtime_bounds(kind: &BoundOpKind) -> bool {
    matches!(kind, BoundOpKind::CachedAttention { query_rows, .. } if *query_rows > 1)
}

pub(crate) fn cached_attention_entry_name(
    resolved: &BoundOp,
    class: CudaAttentionClass,
    split_chunks: Option<u64>,
    numeric_policy: NumericPolicy,
    quantized: &[Option<Codec>],
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
    let kv_token = kv_entry_token(&cached_attention_kv_reads(resolved, quantized)?);
    let rotary_token = if rotary_dim < head_dim {
        format!("_r{rotary_dim}")
    } else {
        String::new()
    };
    let mma_token = if cached_attention_uses_mma(&resolved.kind, numeric_policy) { "_mma16x8x8h" } else { "" };
    let tile_token = cached_attention_tile_rows(&resolved.kind, numeric_policy)
        .map_or_else(String::new, |tile_rows| format!("_tq{tile_rows}{mma_token}"));
    let split_token = split_chunks.map_or_else(String::new, |chunks| format!("_sk{chunks}"));
    let bounds_token = if cached_attention_runtime_bounds(&resolved.kind) {
        String::from("_rtb")
    } else {
        format!(
            "_n{new_key_rows}_l{}_u{}",
            signed_name_part(*cached_lower_inclusive),
            signed_name_part(*new_upper_inclusive),
        )
    };
    Ok(format!(
        "omega_cuda_cached_attention_{}_h{kv_heads}_g{query_groups}_d{head_dim}{rotary_token}{kv_token}{tile_token}{split_token}_s{:08x}{bounds_token}",
        class_tag(class),
        scale.to_bits(),
    ))
}

pub(crate) fn render_cached_attention_cuda(
    resolved: &BoundOp,
    entry: &str,
    numeric_policy: NumericPolicy,
    quantized: &[Option<Codec>],
) -> Result<String, EmitError> {
    let helpers = if cached_attention_uses_mma(&resolved.kind, numeric_policy) {
        format!("{MMA_INSTRUCTION}{MMA_ATTENTION_HELPERS}")
    } else {
        String::new()
    };
    render_attention(resolved, entry, "", &unsplit_tail(resolved, numeric_policy), &helpers, quantized)
}

fn unsplit_tail(resolved: &BoundOp, numeric_policy: NumericPolicy) -> String {
    let body = if cached_attention_uses_mma(&resolved.kind, numeric_policy) { MMA_ATTENTION_BODY } else { TILED_ATTENTION_BODY };
    match cached_attention_tile_rows(&resolved.kind, numeric_policy) {
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
    quantized: &[Option<Codec>],
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
    let reads = cached_attention_kv_reads(resolved, quantized)?;
    let params: String = (0..operands.len())
        .map(|index| {
            let element = match index {
                2 | 3 => reads.key.element,
                6 => reads.value.element,
                _ => reads.compute,
            };
            format!("const {element}* __restrict__ in{index}")
        })
        .collect::<Vec<String>>()
        .join(",\n    ");
    let cached_rows_decl = match class {
        CudaAttentionClass::CachedBound => String::from("const long long cached_key_rows = (long long)in8[0];"),
        CudaAttentionClass::Static => format!("constexpr long long cached_key_rows = {cached_key_rows}LL;"),
    };
    let cached_lower = match *cached_lower_inclusive {
        i64::MIN => String::from("-9223372036854775807LL"),
        bound => format!("{bound}LL"),
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
    preamble(crate::cuda::target::Target::CUDA, &mut source, "float");
    source.push_str(&cache_helpers(&reads));
    source.push_str(helpers);
    let runtime_bounds = cached_attention_runtime_bounds(&resolved.kind);
    let (uniform_fields, bounds_decl) = if runtime_bounds {
        (
            format!(
                " {WORD_C} new_key_rows; {WORD_C} new_upper; {WORD_C} cached_lower;"
            ),
            String::from(
                "const long long new_key_rows = u_ptr->new_key_rows;\n\
                 \x20   const long long new_upper = u_ptr->new_upper;\n\
                 \x20   const long long cached_lower = u_ptr->cached_lower;",
            ),
        )
    } else {
        (
            String::new(),
            format!(
                "constexpr long long new_key_rows = {new_key_rows}LL;\n\
                 \x20   constexpr long long new_upper = {new_upper_inclusive}LL;\n\
                 \x20   constexpr long long cached_lower = {cached_lower};"
            ),
        )
    };
    source.push_str(&format!(
        "struct Uniforms {{ {WORD_C} total_elements;{uniform_fields} }};\n\n"
    ));
    source.push_str(&format!(
        "extern \"C\" __global__ void {entry}(\n    {params},\n    float* __restrict__ out,\n    const Uniforms* __restrict__ u_ptr{extra_param}) {{\n"
    ));
    source.push_str(&format!(
        "    constexpr long long kv_heads = {kv_heads};\n\
         \x20   constexpr long long query_groups = {query_groups};\n\
         \x20   constexpr long long head_dim = {head_dim};\n\
         \x20   constexpr long long rotary_dim = {rotary_dim};\n\
         \x20   constexpr long long pair_dim = rotary_dim / 2;\n\
         \x20   constexpr long long pass_dim = head_dim - rotary_dim;\n\
         \x20   {bounds_decl}\n\
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
    const long long vector_index = (long long)blockIdx.x;
    if (vector_index >= u_ptr->total_elements) { return; }
    const long long query_row = vector_index / (kv_heads * query_groups);
    const long long query_head = vector_index % (kv_heads * query_groups);
    const long long kv_head = query_head / query_groups;
    const long long qbase = query_row * (kv_heads * query_groups * pair_dim) + query_head * pair_dim;
    const long long qpass = query_row * (kv_heads * query_groups * pass_dim) + query_head * pass_dim;
    const long long last_key = cached_key_rows + new_key_rows - 1LL;
    float maximum = negative_infinity;
    float sum = 0.0f;
    float weighted[(head_dim + 31) / 32];
    for (long long slot = 0; slot < (head_dim + 31) / 32; slot++) { weighted[slot] = 0.0f; }
    for (long long key = (long long)warp; key <= last_key; key += warps) {
        const bool cached = key < cached_key_rows;
        const long long new_index = key - cached_key_rows;
        const long long relative = (cached ? key - cached_key_rows : new_index) - query_row;
        if (cached && relative < cached_lower) { continue; }
        if (!cached && relative > new_upper) { continue; }
        const long long key_row = cached ? key : new_index;
        const long long kbase = key_row * (kv_heads * pair_dim) + kv_head * pair_dim;
        const long long kpass = key_row * (kv_heads * pass_dim) + kv_head * pass_dim;
        const long long vbase = key_row * (kv_heads * head_dim) + kv_head * head_dim;
        float partial = 0.0f;
        for (long long pair = (long long)lane; pair < pair_dim; pair += 32LL) {
            partial += in0[qbase + pair] * (cached ? cached_key_at(in2, kbase + pair) : in4[kbase + pair]);
            partial += in1[qbase + pair] * (cached ? cached_key_at(in3, kbase + pair) : in5[kbase + pair]);
        }
        for (long long column = (long long)lane; column < pass_dim; column += 32LL) {
            partial += pass_query[qpass + column] * (cached ? pass_cached_key[kpass + column] : pass_new_key[kpass + column]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) { partial += __shfl_xor_sync(PROXIMA_ALL_LANES, partial, offset); }
        const float score = partial * scale;
        const float next_max = fmaxf(maximum, score);
        const float weight = expf(score - next_max);
        const float rescale = (maximum == negative_infinity) ? 0.0f : expf(maximum - next_max);
        sum = sum * rescale + weight;
        for (long long dimension = (long long)lane; dimension < head_dim; dimension += 32LL) {
            const long long local_dimension = dimension / 32LL;
            weighted[local_dimension] = weighted[local_dimension] * rescale
                + weight * (cached ? cached_value_at(in6, vbase + dimension) : in7[vbase + dimension]);
        }
        maximum = next_max;
    }
    __shared__ float shared_maximum[warps];
    __shared__ float shared_sum[warps];
    __shared__ float shared_weighted[warps * head_dim];
    for (long long dimension = (long long)lane; dimension < head_dim; dimension += 32LL) {
        shared_weighted[warp * head_dim + dimension] = weighted[dimension / 32LL];
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
    for (long long dimension = (long long)threadIdx.x; dimension < head_dim; dimension += (long long)blockDim.x) {
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

const CACHED_KV_OPERANDS: [usize; 3] = [2, 3, 6];

struct CacheRead {
    element: &'static str,
    format: Option<Codec>,
}

pub(crate) struct KvReads {
    compute: &'static str,
    key: CacheRead,
    value: CacheRead,
}

fn cache_read(format: Option<&Codec>, compute: &'static str) -> Option<CacheRead> {
    let Some(format) = format else {
        return Some(CacheRead { element: compute, format: None });
    };
    readable_scalar_cache(format).map(|(_, element)| CacheRead { element, format: Some(*format) })
}

fn load_name(layout: &impl core::fmt::Debug) -> String {
    format!("omega_kv_scalar_{}", description_fingerprint(layout))
}

/// Whether the cached-attention render reads a K cache stored as `key` and a V cache stored as `value`.
///
/// # Errors
/// A message naming the format the render cannot read.
pub(crate) fn cached_attention_kv_admits(key: &Codec, value: &Codec) -> Result<(), String> {
    [key, value].into_iter().try_for_each(|format| {
        let plain = format.is_identity_of(&Codec::default());
        if plain || readable_scalar_cache(format).is_some() {
            Ok(())
        } else {
            Err(format!("a {format:?} cache is not a whole-byte scalar layout the typed load reads"))
        }
    })
}

pub(crate) fn cached_attention_kv_reads(
    resolved: &BoundOp,
    quantized: &[Option<Codec>],
) -> Result<KvReads, EmitError> {
    let packed_elsewhere = quantized
        .iter()
        .enumerate()
        .any(|(index, format)| format.is_some() && !CACHED_KV_OPERANDS.contains(&index));
    let [even, odd, value] = CACHED_KV_OPERANDS.map(|index| quantized.get(index).and_then(Option::as_ref));
    let compute = type_token(crate::cuda::target::Target::CUDA, resolved.node, resolved.dtype)?;
    let reads = (even == odd)
        .then(|| Some(KvReads { compute, key: cache_read(even, compute)?, value: cache_read(value, compute)? }))
        .flatten();
    match reads {
        Some(reads) if !packed_elsewhere => Ok(reads),
        _ => Err(EmitError::CachedAttentionKvCodecNotSupported {
            node: resolved.node,
            reason: "only the cached K even/odd planes and V rows (operands 2, 3, 6) may be packed, each as a whole-byte scalar layout",
        }),
    }
}

fn cache_helpers(reads: &KvReads) -> String {
    let mut text = String::new();
    let layouts: Vec<_> = [&reads.key, &reads.value]
        .into_iter()
        .filter_map(|read| read.format.as_ref().and_then(readable_scalar_cache))
        .map(|(layout, _)| layout)
        .collect();
    if !layouts.is_empty() {
        text.push_str(&format!("#include <{}>\n", crate::cuda::target::Target::CUDA.bf16_header));
        text.push_str("typedef unsigned char uchar;\ntypedef unsigned short ushort;\ntypedef unsigned int uint;\n");
    }
    let mut named: Vec<String> = Vec::new();
    for layout in layouts {
        let name = load_name(&layout);
        if !named.contains(&name) {
            text.push_str(&layout_load_cuda(layout, &name));
            named.push(name);
        }
    }
    text.push_str(&role_helpers("key", &reads.key, reads.compute));
    text.push_str(&role_helpers("value", &reads.value, reads.compute));
    text
}

fn role_helpers(role: &str, read: &CacheRead, compute: &str) -> String {
    let element = read.element;
    let layout = read.format.as_ref().and_then(readable_scalar_cache).map(|(layout, _)| layout);
    let (scalar, quad) = match &layout {
        None => (String::from("p[i]"), format!("*(const {compute}4*)&p[i]")),
        Some(layout) => {
            let load = load_name(layout);
            let at = |offset: &str| format!("{load}((unsigned int)p[i + {offset}])");
            (at("0"), format!("make_{compute}4({}, {}, {}, {})", at("0"), at("1"), at("2"), at("3")))
        }
    };
    format!(
        "__device__ __forceinline__ {compute} cached_{role}_at(const {element}* p, long long i) {{ return {scalar}; }}\n\
         __device__ __forceinline__ {compute}4 cached_{role}_at4(const {element}* p, long long i) {{ return {quad}; }}\n"
    )
}

fn kv_entry_token(reads: &KvReads) -> String {
    let part = |tag: &str, read: &CacheRead| {
        read.format.as_ref().and_then(readable_scalar_cache).map_or_else(String::new, |(layout, _)| format!("_{tag}{}", description_fingerprint(&layout)))
    };
    format!("{}{}", part("k", &reads.key), part("v", &reads.value))
}
