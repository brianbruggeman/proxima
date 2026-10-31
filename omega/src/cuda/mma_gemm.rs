//! Tensor-core prefill GEMM for packed weights: `mma.sync.m16n8k8` (sm_75).
//!
//! A packed codec stores `value = scale * q - minimum` with `q` a small
//! integer, so `q` is exact in fp16 and the per-block `scale`/`minimum` stay
//! outside the multiply-accumulate. Each 16-element half-stage accumulates
//! `sum(q * activation)` into a scratch fragment that is folded into the f32
//! accumulator as `scale * scratch - minimum * activation_sum`. The activation
//! is split into an fp16 high and low part so the product keeps f32 accuracy.
//! Block shape and the integer-times-scale split follow llama.cpp's MMQ design
//! (ggml-cuda/mmq.cuh, MIT, ggml authors); this body is an independent
//! rendering for omega's `BoundOp` descriptor.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_tensor::{
    BoundOp, BoundOpKind, DType, Keep, NumericPolicy, ReduceInit, ScalarOp, StepArg,
};

use super::{
    Codec, EmitError, PQ2_0_BLOCK_BYTES, PQ2_0_BLOCK_ELEMENTS, Q1_0_BLOCK_BYTES,
    Q1_0_BLOCK_ELEMENTS, Q2_0_BLOCK_BYTES, Q2_0_BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES,
    Q4_0_BLOCK_ELEMENTS, Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES,
    Q5_0_BLOCK_ELEMENTS, Q5K_BLOCK_BYTES, Q6K_BLOCK_BYTES, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS,
    gather_count, logical_packed_address, push_reduce_epilogue_write, reduce_is_cooperative,
    reduction_dims,
};

pub(super) const FEATURE_TILE: usize = 64;
pub(super) const TOKEN_TILE: usize = 128;
const ACT_REPS: usize = TOKEN_TILE / FEATURE_TILE;
const WARP_TOKENS: usize = TOKEN_TILE / 2;
const WARP_TILES: usize = WARP_TOKENS / 8;
const _: () = assert!(WARP_TILES == 8 && WARP_TOKENS == 64 && ACT_REPS == 2);
pub(super) const GEMM_THREADS: u64 = 256;
const STAGE_K: usize = 32;
const STAGE_PAD: usize = 40;

/// The structure a packed matmul must have for the tensor-core body: which
/// operand is the packed weight and which output positions it owns.
pub(super) struct MmaGemm {
    pub(super) weight: usize,
    pub(super) activation: usize,
    codec: Codec,
    pub(super) feature_positions: Vec<usize>,
    pub(super) token_positions: Vec<usize>,
}

struct CodecShape {
    decode: &'static str,
    block_bytes: usize,
    block_elements: usize,
    has_minimum: bool,
}

fn codec_shape(codec: Codec) -> Option<CodecShape> {
    let (decode, block_bytes, block_elements, has_minimum) = match codec {
        Codec::Q8_0 => ("mma_decode_q8_0", Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS, false),
        Codec::Q4_0 => ("mma_decode_q4_0", Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS, false),
        Codec::Q5_0 => ("mma_decode_q5_0", Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMENTS, false),
        Codec::Q4K => ("mma_decode_q4k", Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS, true),
        Codec::Q5K => ("mma_decode_q5k", Q5K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS, true),
        Codec::Q6K => ("mma_decode_q6k", Q6K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS, false),
        Codec::Q1_0 => ("mma_decode_q1_0", Q1_0_BLOCK_BYTES, Q1_0_BLOCK_ELEMENTS, false),
        Codec::Q2_0 => ("mma_decode_q2_0", Q2_0_BLOCK_BYTES, Q2_0_BLOCK_ELEMENTS, false),
        Codec::Pq2_0 => ("mma_decode_q2_0", PQ2_0_BLOCK_BYTES, PQ2_0_BLOCK_ELEMENTS, false),
        _ => return None,
    };
    Some(CodecShape {
        decode,
        block_bytes,
        block_elements,
        has_minimum,
    })
}

fn is_plain_product(resolved: &BoundOp) -> bool {
    let steps = &resolved.element_body().steps;
    let [only] = steps.as_slice() else {
        return false;
    };
    only.op == ScalarOp::Multiply
        && (only.args == [StepArg::Operand(0), StepArg::Operand(1)]
            || only.args == [StepArg::Operand(1), StepArg::Operand(0)])
}

/// `Some` when this reduce is a plain `weight * activation` sum with more than
/// one token row. A single token row keeps the matvec body.
pub(super) fn mma_gemm_plan(
    resolved: &BoundOp,
    quantized: &[Option<Codec>],
    numeric_policy: NumericPolicy,
) -> Option<MmaGemm> {
    let BoundOpKind::Reduce {
        keep: Keep::Reduce,
        reduce_op: ScalarOp::Add,
        init: ReduceInit::Zero,
        output_axes,
        epilogue_broadcast_axes,
        ..
    } = &resolved.kind
    else {
        return None;
    };
    let [first, second] = quantized else {
        return None;
    };
    let (weight, codec) = match (first, second) {
        (Some(codec), None) => (0, *codec),
        (None, Some(codec)) => (1, *codec),
        _ => return None,
    };
    let shape = codec_shape(codec)?;
    let activation = 1 - weight;
    let operands = resolved.operands();
    let reduce_extent: u64 = reduction_dims(resolved, output_axes)
        .iter()
        .map(|&dim| resolved.extents[dim as usize])
        .product();
    let weight_layout = &operands[weight].1;
    let eligible = resolved.dtype == DType::FLOAT32
        && epilogue_broadcast_axes.is_empty()
        && gather_count(resolved) == 0
        && is_plain_product(resolved)
        && !logical_packed_address()
        && reduce_is_cooperative(resolved, quantized, numeric_policy)
        && !reduction_dims(resolved, output_axes).is_empty()
        && reduce_extent % STAGE_K as u64 == 0
        && reduce_extent % shape.block_elements.min(STAGE_K) as u64 == 0
        && weight_layout.base % 256 == 0;
    if !eligible {
        return None;
    }
    let activation_layout = &operands[activation].1;
    let mut feature_positions = Vec::new();
    let mut token_positions = Vec::new();
    for (position, &axis) in output_axes.iter().enumerate() {
        match (
            weight_layout.stride(axis) != 0,
            activation_layout.stride(axis) != 0,
        ) {
            (true, false) => feature_positions.push(position),
            (false, true) => token_positions.push(position),
            _ => return None,
        }
    }
    let token_total: u64 = token_positions
        .iter()
        .map(|&position| resolved.extents[output_axes[position] as usize])
        .product();
    (!feature_positions.is_empty() && !token_positions.is_empty() && token_total > 1).then_some(
        MmaGemm {
            weight,
            activation,
            codec,
            feature_positions,
            token_positions,
        },
    )
}

fn axes_product(resolved: &BoundOp, output_axes: &[u16], positions: &[usize]) -> u64 {
    positions
        .iter()
        .map(|&position| resolved.extents[output_axes[position] as usize])
        .product()
}

/// Thread count for the launch: one 256-thread block per 64-feature by
/// 64-token tile.
pub(super) fn mma_gemm_threads(resolved: &BoundOp, plan: &MmaGemm) -> u64 {
    let BoundOpKind::Reduce { output_axes, .. } = &resolved.kind else {
        return 0;
    };
    let feature_tiles = axes_product(resolved, output_axes, &plan.feature_positions)
        .div_ceil(FEATURE_TILE as u64);
    let token_tiles =
        axes_product(resolved, output_axes, &plan.token_positions).div_ceil(TOKEN_TILE as u64);
    feature_tiles * token_tiles * GEMM_THREADS
}

pub(super) const MMA_GEMM_HELPERS: &str = r#"
__device__ __forceinline__ void mma_m16n8k8(float *c, unsigned int a0, unsigned int a1, unsigned int b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ float mma_half_scale(const unsigned char *bytes) {
    return __half2float(__ushort_as_half((unsigned short)((unsigned int)bytes[0] | ((unsigned int)bytes[1] << 8))));
}
__device__ __forceinline__ __half2 mma_pair(int low, int high) {
    return __halves2half2(__int2half_rn(low), __int2half_rn(high));
}
__device__ __forceinline__ void mma_decode_q8_0(const unsigned char *block, unsigned int within, __half2 *out, float *scale, float *minimum) {
    *scale = mma_half_scale(block);
    *minimum = 0.0f;
    const signed char *quants = (const signed char *)(block + 2u + within);
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) { out[pair] = mma_pair((int)quants[2 * pair], (int)quants[2 * pair + 1]); }
}
__device__ __forceinline__ void mma_decode_q4_0(const unsigned char *block, unsigned int within, __half2 *out, float *scale, float *minimum) {
    *scale = mma_half_scale(block);
    *minimum = 0.0f;
    const unsigned char *bytes = block + 2u + (within & 15u);
    bool high = within >= 16u;
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        unsigned int low_byte = bytes[2 * pair];
        unsigned int high_byte = bytes[2 * pair + 1];
        int low_q = (int)(high ? (low_byte >> 4u) : (low_byte & 0x0Fu)) - 8;
        int high_q = (int)(high ? (high_byte >> 4u) : (high_byte & 0x0Fu)) - 8;
        out[pair] = mma_pair(low_q, high_q);
    }
}
__device__ __forceinline__ int mma_q5_0_level(const unsigned char *block, unsigned int qh, unsigned int index) {
    bool high_half = index >= 16u;
    unsigned int local = high_half ? (index - 16u) : index;
    unsigned int byte = (unsigned int)block[6u + local];
    return high_half ? (int)((byte >> 4u) | ((qh >> (local + 12u)) & 0x10u)) : (int)((byte & 0x0Fu) | (((qh >> local) << 4u) & 0x10u));
}
__device__ __forceinline__ void mma_decode_q5_0(const unsigned char *block, unsigned int within, __half2 *out, float *scale, float *minimum) {
    *scale = mma_half_scale(block);
    *minimum = 0.0f;
    unsigned int qh = (unsigned int)block[2] | ((unsigned int)block[3] << 8) | ((unsigned int)block[4] << 16) | ((unsigned int)block[5] << 24);
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        out[pair] = mma_pair(mma_q5_0_level(block, qh, within + 2 * pair) - 16, mma_q5_0_level(block, qh, within + 2 * pair + 1) - 16);
    }
}
__device__ __forceinline__ void mma_decode_q4k(const unsigned char *block, unsigned int within, __half2 *out, float *scale, float *minimum) {
    float d = mma_half_scale(block);
    float dmin = mma_half_scale(block + 2);
    const unsigned char *scales = block + 4;
    const unsigned char *qs = block + 16;
    unsigned int group = within / 64u;
    unsigned int within_group = within % 64u;
    bool low = within_group < 32u;
    unsigned int sub_block = 2u * group + (low ? 0u : 1u);
    *scale = d * (float)q4k_scale(scales, sub_block);
    *minimum = dmin * (float)q4k_min(scales, sub_block);
    const unsigned char *bytes = qs + group * 32u + (within_group % 32u);
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        unsigned int low_byte = bytes[2 * pair];
        unsigned int high_byte = bytes[2 * pair + 1];
        out[pair] = mma_pair((int)(low ? (low_byte & 0x0Fu) : (low_byte >> 4u)), (int)(low ? (high_byte & 0x0Fu) : (high_byte >> 4u)));
    }
}
__device__ __forceinline__ void mma_decode_q5k(const unsigned char *block, unsigned int within, __half2 *out, float *scale, float *minimum) {
    float d = mma_half_scale(block);
    float dmin = mma_half_scale(block + 2);
    const unsigned char *scales = block + 4;
    const unsigned char *qh = block + 16;
    const unsigned char *qs = block + 48;
    unsigned int chunk = within / 64u;
    unsigned int within_chunk = within % 64u;
    bool low = within_chunk < 32u;
    unsigned int sub_block = 2u * chunk + (low ? 0u : 1u);
    unsigned int offset = within_chunk % 32u;
    *scale = d * (float)q4k_scale(scales, sub_block);
    *minimum = dmin * (float)q4k_min(scales, sub_block);
    unsigned int mask = low ? (1u << (2u * chunk)) : (2u << (2u * chunk));
    int levels[8];
    #pragma unroll
    for (int slot = 0; slot < 8; slot++) {
        unsigned int qs_byte = qs[chunk * 32u + offset + slot];
        int nibble = (int)(low ? (qs_byte & 0x0Fu) : (qs_byte >> 4u));
        levels[slot] = nibble + (((unsigned int)qh[offset + slot] & mask) != 0u ? 16 : 0);
    }
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) { out[pair] = mma_pair(levels[2 * pair], levels[2 * pair + 1]); }
}
__device__ __forceinline__ void mma_decode_q6k(const unsigned char *block, unsigned int within, __half2 *out, float *scale, float *minimum) {
    float d = mma_half_scale(block + 208);
    unsigned int half_index = within / 128u;
    unsigned int local = within % 128u;
    unsigned int l = local % 32u;
    unsigned int lane = local / 32u;
    const unsigned char *ql = block + half_index * 64u;
    const unsigned char *qh = block + 128u + half_index * 32u;
    *scale = d * (float)(signed char)block[192u + half_index * 8u + (l / 16u) + lane * 2u];
    *minimum = 0.0f;
    int levels[8];
    #pragma unroll
    for (int slot = 0; slot < 8; slot++) {
        unsigned int ql_byte = (lane % 2u == 0u) ? ql[l + slot] : ql[l + slot + 32u];
        unsigned int nibble = (lane < 2u) ? (ql_byte & 0x0Fu) : (ql_byte >> 4u);
        unsigned int high2 = ((unsigned int)qh[l + slot] >> (lane * 2u)) & 0x03u;
        levels[slot] = (int)(nibble | (high2 << 4u)) - 32;
    }
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) { out[pair] = mma_pair(levels[2 * pair], levels[2 * pair + 1]); }
}
__device__ __forceinline__ void mma_decode_q1_0(const unsigned char *block, unsigned int within, __half2 *out, float *scale, float *minimum) {
    *scale = mma_half_scale(block);
    *minimum = 0.0f;
    unsigned int bits = (unsigned int)block[2u + (within >> 3u)];
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        out[pair] = mma_pair(((bits >> (2 * pair)) & 1u) ? 1 : -1, ((bits >> (2 * pair + 1)) & 1u) ? 1 : -1);
    }
}
__device__ __forceinline__ void mma_decode_q2_0(const unsigned char *block, unsigned int within, __half2 *out, float *scale, float *minimum) {
    *scale = mma_half_scale(block);
    *minimum = 0.0f;
    unsigned int codes = (unsigned int)block[2u + (within >> 2u)] | ((unsigned int)block[3u + (within >> 2u)] << 8u);
    #pragma unroll
    for (int pair = 0; pair < 4; pair++) {
        out[pair] = mma_pair((int)((codes >> (4 * pair)) & 3u) - 1, (int)((codes >> (4 * pair + 2)) & 3u) - 1);
    }
}
"#;

pub(super) fn push_mma_gemm_body(
    source: &mut String,
    resolved: &BoundOp,
    plan: &MmaGemm,
    element_type: &str,
) -> Result<(), EmitError> {
    let BoundOpKind::Reduce {
        output_axes,
        epilogue_body,
        epilogue_operands,
        ..
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "keep::reduce fold",
            found: resolved.kind.name(),
        });
    };
    let shape = codec_shape(plan.codec).ok_or(EmitError::CudaUnsupportedCodec {
        node: resolved.node,
    })?;
    let weight = plan.weight;
    let activation = plan.activation;
    let reduce_dims = reduction_dims(resolved, output_axes);
    let reduce_rank = reduce_dims.len();
    let output_rank = output_axes.len();
    let has_minimum = shape.has_minimum;
    let (block_bytes, block_elements) = (shape.block_bytes, shape.block_elements);
    let decode = shape.decode;

    source.push_str("    const int tid = (int)threadIdx.x;\n");
    source.push_str("    const int warp = tid >> 5;\n");
    source.push_str("    const int lane = tid & 31;\n");
    source.push_str("    const int group = lane >> 2;\n");
    source.push_str("    const int quad = lane & 3;\n");
    source.push_str("    const int warp_feature = warp & 3;\n");
    source.push_str("    const int warp_token = warp >> 2;\n");
    source.push_str("    long feature_total = 1;\n");
    for &position in &plan.feature_positions {
        source.push_str(&format!(
            "    feature_total *= u.output_extents[{position}];\n"
        ));
    }
    source.push_str("    long token_total = 1;\n");
    for &position in &plan.token_positions {
        source.push_str(&format!(
            "    token_total *= u.output_extents[{position}];\n"
        ));
    }
    source.push_str(&format!(
        "    const long feature_tiles = (feature_total + {FEATURE_TILE} - 1) / {FEATURE_TILE};\n"
    ));
    source.push_str("    const long feature_first = ((long)blockIdx.x % feature_tiles) * ");
    source.push_str(&format!("{FEATURE_TILE};\n"));
    source.push_str(&format!(
        "    const long token_first = ((long)blockIdx.x / feature_tiles) * {TOKEN_TILE};\n"
    ));
    source.push_str("    const int load_row = tid >> 2;
");
    source.push_str("    const int load_segment = (tid & 3) * 8;
");
    source.push_str("    const long load_feature = feature_first + load_row;
");
    source.push_str("    const bool feature_ok = load_feature < feature_total;
");
    for rep in 0..ACT_REPS {
        source.push_str(&format!(
            "    const long load_token{rep} = token_first + load_row + {};
    const bool token_ok{rep} = load_token{rep} < token_total;
    long token_offset{rep} = u.operand_base[{activation}];
    if (token_ok{rep}) {{
        long token_rest = load_token{rep};
",
            rep * FEATURE_TILE
        ));
        for &position in plan.token_positions.iter().rev() {
            let axis = output_axes[position];
            source.push_str(&format!(
                "        token_offset{rep} += (token_rest % u.output_extents[{position}]) * u.operand_strides[{activation}][{axis}]; token_rest /= u.output_extents[{position}];
"
            ));
        }
        source.push_str("    }
");
    }
    source.push_str(&format!(
        "    const long weight_row = u.operand_base[{weight}] + load_feature * u.reduction_total;\n"
    ));

    source.push_str(&format!(
        "    __shared__ __align__(16) __half weight_q[{FEATURE_TILE}][{STAGE_PAD}];\n"
    ));
    source.push_str(&format!(
        "    __shared__ __align__(16) __half act_high[{TOKEN_TILE}][{STAGE_PAD}];\n"
    ));
    source.push_str(&format!(
        "    __shared__ __align__(16) __half act_low[{TOKEN_TILE}][{STAGE_PAD}];\n"
    ));
    source.push_str(&format!(
        "    __shared__ float weight_scale[{FEATURE_TILE}][2];\n"
    ));
    if has_minimum {
        source.push_str(&format!(
            "    __shared__ float weight_minimum[{FEATURE_TILE}][2];\n"
        ));
        source.push_str(&format!(
            "    __shared__ float act_sum[{TOKEN_TILE}][2];\n"
        ));
    }
    source.push_str("    float acc[8][4];\n");
    source.push_str("    #pragma unroll\n    for (int tile = 0; tile < 8; tile++) {\n");
    source.push_str("        #pragma unroll\n        for (int slot = 0; slot < 4; slot++) { acc[tile][slot] = 0.0f; }\n    }\n");

    source.push_str(&format!(
        "    for (long stage_first = 0; stage_first < u.reduction_total; stage_first += {STAGE_K}) {{\n"
    ));
    source.push_str("        float scale = 0.0f;\n        float minimum = 0.0f;\n");
    source.push_str("        __align__(16) __half2 weight_pairs[4];\n");
    source.push_str("        #pragma unroll\n        for (int pair = 0; pair < 4; pair++) { weight_pairs[pair] = __float2half2_rn(0.0f); }\n");
    source.push_str("        if (feature_ok) {\n");
    source.push_str("            long element = weight_row + stage_first + load_segment;\n");
    source.push_str(&format!(
        "            const unsigned char *block = in{weight} + (element / {block_elements}) * {block_bytes};\n"
    ));
    source.push_str(&format!(
        "            {decode}(block, (unsigned int)(element % {block_elements}), weight_pairs, &scale, &minimum);\n"
    ));
    source.push_str("        }\n");
    source.push_str("        *(uint4 *)&weight_q[load_row][load_segment] = *(const uint4 *)weight_pairs;\n");
    source.push_str("        if ((load_segment & 15) == 0) {\n");
    source.push_str("            weight_scale[load_row][load_segment >> 4] = scale;\n");
    if has_minimum {
        source.push_str("            weight_minimum[load_row][load_segment >> 4] = minimum;\n");
    }
    source.push_str("        }\n");

    for rep in 0..ACT_REPS {
        let row = format!("load_row + {}", rep * FEATURE_TILE);
        source.push_str("        {
");
        source.push_str("        float act_values[8];
");
        source.push_str("        #pragma unroll
        for (int slot = 0; slot < 8; slot++) { act_values[slot] = 0.0f; }
");
        source.push_str(&format!("        if (token_ok{rep}) {{
"));
        source.push_str("            unsigned int reduce_rest = (unsigned int)(stage_first + load_segment);
");
        source.push_str(&format!("            long reduce_coord[{reduce_rank}];
"));
        source.push_str("            long reduce_offset = 0;
");
        for (index, dim) in reduce_dims.iter().enumerate().rev() {
            source.push_str(&format!(
                "            reduce_coord[{index}] = (long)(reduce_rest % (unsigned int)u.reduction_extents[{index}]); reduce_rest /= (unsigned int)u.reduction_extents[{index}];
            reduce_offset += reduce_coord[{index}] * u.operand_strides[{activation}][{dim}];
"
            ));
        }
        source.push_str("            #pragma unroll
            for (int slot = 0; slot < 8; slot++) {
");
        source.push_str(&format!(
            "                act_values[slot] = in{activation}[token_offset{rep} + reduce_offset];
"
        ));
        let last = reduce_rank - 1;
        source.push_str(&format!(
            "                reduce_coord[{last}]++; reduce_offset += u.operand_strides[{activation}][{}];
",
            reduce_dims[last]
        ));
        for index in (1..reduce_rank).rev() {
            source.push_str(&format!(
                "                if (reduce_coord[{index}] == u.reduction_extents[{index}]) {{ reduce_offset -= reduce_coord[{index}] * u.operand_strides[{activation}][{}]; reduce_coord[{index}] = 0; reduce_coord[{prev}]++; reduce_offset += u.operand_strides[{activation}][{}]; }}
",
                reduce_dims[index],
                reduce_dims[index - 1],
                prev = index - 1
            ));
        }
        source.push_str("            }
        }
");
        source.push_str("        __align__(16) __half2 high_pairs[4];
        __align__(16) __half2 low_pairs[4];
");
        if has_minimum {
            source.push_str("        float half_sum = 0.0f;
");
        }
        source.push_str("        #pragma unroll
        for (int pair = 0; pair < 4; pair++) {
");
        source.push_str("            __half high_a = __float2half_rn(act_values[2 * pair]);
");
        source.push_str("            __half high_b = __float2half_rn(act_values[2 * pair + 1]);
");
        source.push_str("            high_pairs[pair] = __halves2half2(high_a, high_b);
");
        source.push_str("            low_pairs[pair] = __halves2half2(__float2half_rn(act_values[2 * pair] - __half2float(high_a)), __float2half_rn(act_values[2 * pair + 1] - __half2float(high_b)));
");
        if has_minimum {
            source.push_str("            half_sum += act_values[2 * pair] + act_values[2 * pair + 1];
");
        }
        source.push_str("        }
");
        source.push_str(&format!(
            "        *(uint4 *)&act_high[{row}][load_segment] = *(const uint4 *)high_pairs;
"
        ));
        source.push_str(&format!(
            "        *(uint4 *)&act_low[{row}][load_segment] = *(const uint4 *)low_pairs;
"
        ));
        if has_minimum {
            source.push_str("        half_sum += __shfl_xor_sync(0xffffffffu, half_sum, 1);
");
            source.push_str(&format!(
                "        if ((load_segment & 15) == 0) {{ act_sum[{row}][load_segment >> 4] = half_sum; }}
"
            ));
        }
        source.push_str("        }
");
    }
    source.push_str("        __syncthreads();\n");

    source.push_str("        #pragma unroll\n        for (int half_index = 0; half_index < 2; half_index++) {\n");
    source.push_str("            float scratch[8][4];\n");
    source.push_str("            #pragma unroll\n            for (int tile = 0; tile < 8; tile++) {\n");
    source.push_str("                #pragma unroll\n                for (int slot = 0; slot < 4; slot++) { scratch[tile][slot] = 0.0f; }\n            }\n");
    source.push_str("            #pragma unroll\n            for (int step = 0; step < 2; step++) {\n");
    source.push_str("                const int k_base = half_index * 16 + step * 8 + quad * 2;\n");
    source.push_str("                const unsigned int a_low = *(const unsigned int *)&weight_q[warp_feature * 16 + group][k_base];\n");
    source.push_str("                const unsigned int a_high = *(const unsigned int *)&weight_q[warp_feature * 16 + group + 8][k_base];\n");
    source.push_str("                #pragma unroll\n                for (int tile = 0; tile < 8; tile++) {\n");
    source.push_str("                    if (token_first + warp_token * 64 + tile * 8 >= token_total) { continue; }\n");
    source.push_str("                    const int token_row = warp_token * 64 + tile * 8 + group;\n");
    source.push_str("                    mma_m16n8k8(scratch[tile], a_low, a_high, *(const unsigned int *)&act_high[token_row][k_base]);\n");
    source.push_str("                    mma_m16n8k8(scratch[tile], a_low, a_high, *(const unsigned int *)&act_low[token_row][k_base]);\n");
    source.push_str("                }\n            }\n");
    source.push_str("            const float scale_low = weight_scale[warp_feature * 16 + group][half_index];\n");
    source.push_str("            const float scale_high = weight_scale[warp_feature * 16 + group + 8][half_index];\n");
    if has_minimum {
        source.push_str("            const float minimum_low = weight_minimum[warp_feature * 16 + group][half_index];\n");
        source.push_str("            const float minimum_high = weight_minimum[warp_feature * 16 + group + 8][half_index];\n");
    }
    source.push_str("            #pragma unroll\n            for (int tile = 0; tile < 8; tile++) {\n");
    source.push_str("                if (token_first + warp_token * 64 + tile * 8 >= token_total) { continue; }\n");
    if has_minimum {
        source.push_str("                const int token_column = warp_token * 64 + tile * 8 + quad * 2;\n");
        source.push_str("                const float sum_a = act_sum[token_column][half_index];\n");
        source.push_str("                const float sum_b = act_sum[token_column + 1][half_index];\n");
        source.push_str("                acc[tile][0] += scale_low * scratch[tile][0] - minimum_low * sum_a;\n");
        source.push_str("                acc[tile][1] += scale_low * scratch[tile][1] - minimum_low * sum_b;\n");
        source.push_str("                acc[tile][2] += scale_high * scratch[tile][2] - minimum_high * sum_a;\n");
        source.push_str("                acc[tile][3] += scale_high * scratch[tile][3] - minimum_high * sum_b;\n");
    } else {
        source.push_str("                acc[tile][0] += scale_low * scratch[tile][0];\n");
        source.push_str("                acc[tile][1] += scale_low * scratch[tile][1];\n");
        source.push_str("                acc[tile][2] += scale_high * scratch[tile][2];\n");
        source.push_str("                acc[tile][3] += scale_high * scratch[tile][3];\n");
    }
    source.push_str("            }\n        }\n");
    source.push_str("        __syncthreads();\n    }\n");

    source.push_str("    #pragma unroll\n    for (int tile = 0; tile < 8; tile++) {\n");
    source.push_str("        #pragma unroll\n        for (int slot = 0; slot < 4; slot++) {\n");
    source.push_str("            const long feature = feature_first + warp_feature * 16 + group + (slot >> 1) * 8;\n");
    source.push_str("            const long token = token_first + warp_token * 64 + tile * 8 + quad * 2 + (slot & 1);\n");
    source.push_str("            if (feature >= feature_total || token >= token_total) { continue; }\n");
    source.push_str(&format!("            long output_coord[{output_rank}];\n"));
    source.push_str("            long feature_rest = feature;\n            long token_rest = token;\n");
    for &position in plan.feature_positions.iter().rev() {
        source.push_str(&format!(
            "            output_coord[{position}] = feature_rest % u.output_extents[{position}]; feature_rest /= u.output_extents[{position}];\n"
        ));
    }
    for &position in plan.token_positions.iter().rev() {
        source.push_str(&format!(
            "            output_coord[{position}] = token_rest % u.output_extents[{position}]; token_rest /= u.output_extents[{position}];\n"
        ));
    }
    source.push_str("            long out_offset = u.out_base;\n");
    for (position, axis) in output_axes.iter().enumerate() {
        source.push_str(&format!(
            "            out_offset += output_coord[{position}] * u.out_strides[{axis}];\n"
        ));
    }
    push_reduce_epilogue_write(
        source,
        epilogue_body,
        epilogue_operands,
        output_rank,
        element_type,
        "            ",
        |dim| format!("output_coord[{dim}]"),
        "acc[tile][slot]",
        "out_offset",
    );
    source.push_str("        }\n    }\n");
    Ok(())
}
