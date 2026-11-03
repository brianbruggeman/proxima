//! Int8 tensor-core prefill GEMM: `mma.sync.m8n8k16.s32.s8.s8.s32` over a packed weight and an
//! activation a `BlockQuantize` node already holds as `block_q8_1` bytes (one f16 scale, one f16
//! scaled sum, 32 int8 codes), read as an ordinary input binding. The weight codes are unpacked
//! to int8 into shared memory by the layout's [`BlockDot`] helpers, each 32-element chunk
//! accumulates exactly in int32 and is folded into the f32 accumulator as
//! `weight_scale * activation_scale * sum`.

use alloc::format;
use alloc::string::String;

use proxima_primitives::block_layout::BlockLayout;
use proxima_tensor::{BoundOp, Layout, NumericPolicy};

use super::block_dot::{BlockDot, DOT_ELEMENTS, block_dot_i8};
use super::mma_gemm::{MmaGemm, push_output_write, push_tile_origin};
use super::{Codec, EmitError, reduction_dims};

pub(super) const INT8_FEATURE_TILE: usize = 128;
pub(super) const INT8_TOKEN_TILE: usize = 64;
const INT8_THREADS: u64 = 256;
const ROW_PAD: usize = 16;

pub(super) const INT8_MMA_INSTRUCTION: &str = r#"
__device__ __forceinline__ void mma_s8(int *acc, int fragment_a, int fragment_b) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
        : "+r"(acc[0]), "+r"(acc[1]) : "r"(fragment_a), "r"(fragment_b));
}
"#;

/// The weight layout's int8 unpack helpers and how many 32-element chunks one staged slab spans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Int8Gemm {
    pub(super) dot: BlockDot,
    pub(super) chunks: usize,
    pub(super) activation: BlockLayout,
}

fn chunks_per_slab(reduction: u64) -> usize {
    [4usize, 2]
        .into_iter()
        .find(|chunks| reduction.is_multiple_of((chunks * DOT_ELEMENTS) as u64))
        .unwrap_or(1)
}

/// The int8 gemm serves only an activation some `BlockQuantize` already produced (`prequantized`,
/// held as `block_q8_1` bytes from the start of its buffer); any other activation takes the
/// fp16 gemm.
pub(super) fn int8_gemm_plan(
    resolved: &BoundOp,
    codec: Codec,
    numeric_policy: NumericPolicy,
    activation_layout: &Layout,
    output_axes: &[u16],
    prequantized: Option<Codec>,
) -> Option<Int8Gemm> {
    if !numeric_policy.activation_quantization || activation_layout.base != 0 {
        return None;
    }
    let activation = prequantized?.block_layout()?;
    let dot = int8_dot(codec)?;
    let reduction: u64 = reduction_dims(resolved, output_axes)
        .iter()
        .map(|&dim| resolved.extents[usize::from(dim)])
        .product();
    Some(Int8Gemm {
        chunks: chunks_per_slab(reduction),
        dot,
        activation,
    })
}

fn int8_dot(codec: Codec) -> Option<BlockDot> {
    let dot = block_dot_i8(&codec.block_layout()?)?;
    (dot.centering.is_none() && dot.block_bytes.is_multiple_of(2)).then_some(dot)
}

/// Whether the int8 gemm unpacks a weight of this codec.
#[must_use]
pub fn is_int8_weight(codec: Codec) -> bool {
    int8_dot(codec).is_some()
}

fn run_term(layout: &BlockLayout, group: &str, base: usize, next: usize) -> String {
    match layout.runs() {
        1 => format!("{base}"),
        runs => format!("{base} + (({group}) % {runs}) * {}", next - base),
    }
}

/// C expression for the byte offset of the block holding 32-element group `group`.
pub(super) fn activation_block_offset(layout: &BlockLayout, group: &str) -> String {
    format!("(({group}) / {}) * {}", layout.runs(), layout.block_bytes)
}

/// C expression for group `group`'s scale header, relative to its block.
pub(super) fn activation_scale_offset(layout: &BlockLayout, group: &str) -> String {
    run_term(layout, group, layout.run_scale_byte(0), layout.run_scale_byte(layout.runs().min(2) - 1))
}

/// C expression for group `group`'s first code, relative to its block.
pub(super) fn activation_code_offset(layout: &BlockLayout, group: &str) -> String {
    run_term(layout, group, layout.run_code_byte(0), layout.run_code_byte(layout.runs().min(2) - 1))
}

/// Whether every group's 32 codes start on a 16-byte boundary, so two `uint4` loads read them.
pub(super) fn activation_codes_are_16_byte_aligned(layout: &BlockLayout) -> bool {
    layout.block_bytes.is_multiple_of(16) && (0..layout.runs()).all(|run| layout.run_code_byte(run).is_multiple_of(16))
}

/// C statements declaring `int4 activation_low, activation_high`: the 32 int8 codes at `codes`.
pub(super) fn activation_code_words(layout: &BlockLayout, codes: &str, indent: &str) -> String {
    if activation_codes_are_16_byte_aligned(layout) {
        return format!(
            "{indent}const uint4 activation_low = ((const uint4 *){codes})[0];\n{indent}const uint4 activation_high = ((const uint4 *){codes})[1];\n"
        );
    }
    format!(
        "{indent}const unsigned int *activation_words = (const unsigned int *){codes};\n\
         {indent}const uint4 activation_low = make_uint4(activation_words[0], activation_words[1], activation_words[2], activation_words[3]);\n\
         {indent}const uint4 activation_high = make_uint4(activation_words[4], activation_words[5], activation_words[6], activation_words[7]);\n"
    )
}

fn push_declarations(source: &mut String, chunks: usize) {
    let row = chunks * DOT_ELEMENTS + ROW_PAD;
    source.push_str(&format!(
        "    __shared__ __align__(16) signed char weight_tile[{INT8_FEATURE_TILE}][{row}];\n\
         \x20   __shared__ __align__(16) signed char act_tile[{INT8_TOKEN_TILE}][{row}];\n\
         \x20   __shared__ float weight_scales[{chunks}][{INT8_FEATURE_TILE}];\n\
         \x20   __shared__ float act_scales[{INT8_TOKEN_TILE}][{chunks}];\n\
         \x20   float flat[32];\n\
         \x20   #pragma unroll\n\
         \x20   for (int index = 0; index < 32; index++) {{ flat[index] = 0.0f; }}\n"
    ));
}

fn push_stage_weight(source: &mut String, plan: &MmaGemm, int8: &Int8Gemm) {
    let weight = plan.weight;
    let (name, elements, block_bytes) = (&int8.dot.name, int8.dot.elements, int8.dot.block_bytes);
    let (chunks, items) = (int8.chunks, int8.chunks * INT8_FEATURE_TILE);
    source.push_str(&format!(
        "        for (int item = tid; item < {items}; item += {INT8_THREADS}) {{\n\
         \x20           const int feature_local = item / {chunks};\n\
         \x20           const int chunk = item % {chunks};\n\
         \x20           const long feature = feature_first + feature_local;\n\
         \x20           int words[8] = {{0, 0, 0, 0, 0, 0, 0, 0}};\n\
         \x20           float block_scale = 0.0f;\n\
         \x20           if (feature < feature_total) {{\n\
         \x20               const long element = u.operand_base[{weight}] + feature * u.reduction_total + slab_first + chunk * 32;\n\
         \x20               const unsigned char *block = in{weight} + (element / {elements}) * {block_bytes};\n\
         \x20               {name}_i8x32(block, (unsigned int)(element % {elements}) / 32u, words);\n\
         \x20               block_scale = {name}_scale(block);\n\
         \x20           }}\n\
         \x20           uint4 *destination = (uint4 *)&weight_tile[feature_local][chunk * 32];\n\
         \x20           destination[0] = make_uint4((unsigned int)words[0], (unsigned int)words[1], (unsigned int)words[2], (unsigned int)words[3]);\n\
         \x20           destination[1] = make_uint4((unsigned int)words[4], (unsigned int)words[5], (unsigned int)words[6], (unsigned int)words[7]);\n\
         \x20           weight_scales[chunk][feature_local] = block_scale;\n\
         \x20       }}\n"
    ));
}

fn push_stage_activation(source: &mut String, activation: usize, int8: &Int8Gemm) {
    let (chunks, layout) = (int8.chunks, &int8.activation);
    let items = chunks * INT8_TOKEN_TILE;
    let block = activation_block_offset(layout, "group");
    let scale = activation_scale_offset(layout, "group");
    let codes = activation_code_words(layout, &format!("(bytes + {})", activation_code_offset(layout, "group")), "                ");
    source.push_str(&format!(
        "        for (int item = tid; item < {items}; item += {INT8_THREADS}) {{\n\
         \x20           const int token_local = item / {chunks};\n\
         \x20           const int chunk = item % {chunks};\n\
         \x20           const long token = token_first + token_local;\n\
         \x20           uint4 low = make_uint4(0u, 0u, 0u, 0u);\n\
         \x20           uint4 high = make_uint4(0u, 0u, 0u, 0u);\n\
         \x20           float step = 0.0f;\n\
         \x20           if (token < token_total) {{\n\
         \x20               const long group = token * (u.reduction_total / 32) + slab_first / 32 + chunk;\n\
         \x20               const unsigned char *bytes = in{activation} + {block};\n\
         {codes}\
         \x20               low = activation_low;\n\
         \x20               high = activation_high;\n\
         \x20               step = mma_half_scale(bytes + {scale});\n\
         \x20           }}\n\
         \x20           uint4 *destination = (uint4 *)&act_tile[token_local][chunk * 32];\n\
         \x20           destination[0] = low;\n\
         \x20           destination[1] = high;\n\
         \x20           act_scales[token_local][chunk] = step;\n\
         \x20       }}\n"
    ));
}

fn push_compute(source: &mut String, chunks: usize) {
    source.push_str(&format!(
        "        #pragma unroll\n\
         \x20       for (int chunk = 0; chunk < {chunks}; chunk++) {{\n\
         \x20           int accumulators[4][4][2] = {{}};\n\
         \x20           #pragma unroll\n\
         \x20           for (int step = 0; step < 2; step++) {{\n\
         \x20               const int depth_at = chunk * 32 + step * 16 + quad * 4;\n\
         \x20               int fragment_a[4];\n\
         \x20               int fragment_b[4];\n\
         \x20               #pragma unroll\n\
         \x20               for (int tile = 0; tile < 4; tile++) {{\n\
         \x20                   fragment_a[tile] = *(const int *)&weight_tile[warp_feature * 32 + tile * 8 + group][depth_at];\n\
         \x20                   fragment_b[tile] = *(const int *)&act_tile[warp_token * 32 + tile * 8 + group][depth_at];\n\
         \x20               }}\n\
         \x20               #pragma unroll\n\
         \x20               for (int row = 0; row < 4; row++) {{\n\
         \x20                   #pragma unroll\n\
         \x20                   for (int col = 0; col < 4; col++) {{ mma_s8(accumulators[row][col], fragment_a[row], fragment_b[col]); }}\n\
         \x20               }}\n\
         \x20           }}\n\
         \x20           #pragma unroll\n\
         \x20           for (int row = 0; row < 4; row++) {{\n\
         \x20               const float weight_step = weight_scales[chunk][warp_feature * 32 + row * 8 + group];\n\
         \x20               #pragma unroll\n\
         \x20               for (int col = 0; col < 4; col++) {{\n\
         \x20                   #pragma unroll\n\
         \x20                   for (int pair = 0; pair < 2; pair++) {{\n\
         \x20                       const float scale = weight_step * act_scales[warp_token * 32 + col * 8 + quad * 2 + pair][chunk];\n\
         \x20                       flat[(row * 4 + col) * 2 + pair] += scale * (float)accumulators[row][col][pair];\n\
         \x20                   }}\n\
         \x20               }}\n\
         \x20           }}\n\
         \x20       }}\n"
    ));
}

pub(super) fn push_int8_gemm_body(
    source: &mut String,
    resolved: &BoundOp,
    plan: &MmaGemm,
    int8: &Int8Gemm,
    element_type: &str,
) -> Result<(), EmitError> {
    push_tile_origin(source, plan, INT8_FEATURE_TILE, INT8_TOKEN_TILE);
    push_declarations(source, int8.chunks);
    let slab = int8.chunks * DOT_ELEMENTS;
    source.push_str(&format!(
        "    for (long slab_first = 0; slab_first < u.reduction_total; slab_first += {slab}) {{\n"
    ));
    push_stage_weight(source, plan, int8);
    push_stage_activation(source, plan.activation, int8);
    source.push_str("        __syncthreads();\n");
    push_compute(source, int8.chunks);
    source.push_str("        __syncthreads();\n    }\n");
    push_output_write(
        source,
        resolved,
        plan,
        element_type,
        "        const int row = index >> 3;\n        const int col = (index >> 1) & 3;\n        const int pair = index & 1;\n",
        "feature_first + warp_feature * 32 + row * 8 + group",
        "token_first + warp_token * 32 + col * 8 + quad * 2 + pair",
    )
}
