use alloc::borrow::ToOwned;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use proxima_numeric::block_layout::BlockLayout;
use proxima_tensor::{BoundOp, BoundOpKind, Keep, Layout, NodeId, ReduceInit, ScalarOp, StepArg};
use proxima_numeric::{DType, NumericPolicy};

use super::block_dot::{BlockDot, ChunkTerms, DOT_ELEMENTS, block_dot_i8};
use super::int8_gemm::{activation_block_offset, activation_code_offset, activation_code_words, activation_scale_offset};
use super::{
    EmitDevice,
    Codec, CudaGridSpec, CudaKernel, EmitError, WARP_SIZE, flat_packed_rows, gather_count, push_reduce_epilogue_write,
    reduction_dims,
};
use crate::lowering::Binding;

/// Warps in one block of a staged matvec.
pub(super) const STAGED_WARPS: usize = 8;


/// Threads in one block of a staged matvec.
pub(super) const STAGED_THREADS: u64 = (STAGED_WARPS * 32) as u64;

/// Largest reduction a block can stage: int8 values plus one f32 scale per
/// 32 elements must fit the 48 KiB static shared-memory limit.
const MAX_STAGED_REDUCTION: u64 = 32_768;

/// Blocks per multiprocessor a small output should still launch so the device stays fed.
const TARGET_WAVES: u64 = 3;

fn shaped_rows(preferred: usize, output_total: u64, multiprocessors: u32) -> usize {
    let wanted = u64::from(multiprocessors) * TARGET_WAVES;
    let blocks = |rows: usize| output_total.div_ceil((STAGED_WARPS * rows) as u64);
    (1..=preferred)
        .rev()
        .find(|rows| blocks(*rows) >= wanted)
        .unwrap_or(1)
}

/// Below this many bytes of int8 plus scales a block stages in shared memory: the extra
/// launch of a global pre-pass costs more than the smem it would free.
const SMEM_STAGE_LIMIT_BYTES: u64 = 4096;

/// Global quantized-activation words are int8 groups of 32 followed by one f32 scale each.
pub(super) fn staged_scratch_len(reduction: u64) -> usize {
    (reduction / 4 + reduction / DOT_ELEMENTS as u64) as usize
}

/// The arena slot of a node's quantized activation; counted down from the top so it cannot
/// meet a program node, which are numbered up from zero.
pub(super) fn staged_scratch_node(node: NodeId) -> NodeId {
    NodeId(u32::MAX - node.0)
}

/// `sum(packed * shared)` lowered as: the operand every output row reads at the
/// same coordinates is staged once per block in shared memory, rounded to int8
/// groups of 32, and each warp folds `BlockDot::rows_per_warp` rows against it with
/// `dp4a`. The packed operand's half comes from its [`BlockDot`], which is
/// derived from the codec's block layout, so the rule names no format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct StagedMatvec {
    pub weight_slot: usize,
    pub activation_slot: usize,
    pub reduce_dims: Vec<u16>,
    pub reduce_extents: Vec<u64>,
    pub reduction: u64,
    /// The reduced elements are one unit-stride run starting on a 16-byte boundary.
    pub contiguous_activation: bool,
    pub dot: BlockDot,
    /// Output rows one warp folds: the codec's preferred count, reduced until the launch spans
    /// `TARGET_WAVES` waves of the device's multiprocessors.
    pub rows: usize,
    /// The activation is quantized into a global scratch by a pre-pass launch instead of
    /// being staged in every block's shared memory.
    pub scratch: bool,
    /// The block layout of an activation a `BlockQuantize` already produced, so the matvec
    /// neither stages nor pre-passes it.
    pub external: Option<BlockLayout>,
}

impl StagedMatvec {
    fn rows_in_shared(&self) -> bool {
        !self.scratch && self.external.is_none()
    }
}

fn is_contiguous_run(layout: &Layout, reduce_dims: &[u16], reduce_extents: &[u64]) -> bool {
    let mut expected = 1i64;
    for (dim, extent) in reduce_dims.iter().zip(reduce_extents).rev() {
        if layout.stride(*dim) != expected {
            return false;
        }
        expected *= i64::try_from(*extent).unwrap_or(i64::MAX);
    }
    layout.base.rem_euclid(4) == 0
}

pub(super) fn staged_matvec_plan(
    resolved: &BoundOp,
    quantized: &[Option<Codec>],
    numeric_policy: NumericPolicy,
    device: EmitDevice,
) -> Option<StagedMatvec> {
    if !numeric_policy.activation_quantization || resolved.dtype != DType::FLOAT32 {
        return None;
    }
    let BoundOpKind::Reduce {
        reduce_op: ScalarOp::Add,
        init: ReduceInit::Zero,
        keep: Keep::Reduce,
        output_axes,
        epilogue_broadcast_axes,
        operands,
        ..
    } = &resolved.kind
    else {
        return None;
    };
    let [only_step] = resolved.element_body().steps.as_slice() else {
        return None;
    };
    let is_product = only_step.op == ScalarOp::Multiply
        && only_step.args == [StepArg::Operand(0), StepArg::Operand(1)];
    let (weight_slot, activation_slot, codec, external) = match quantized {
        [Some(codec), None] => (0, 1, *codec, None),
        [None, Some(codec)] => (1, 0, *codec, None),
        [Some(codec), Some(activation)] if activation.is_quantized_activation() && !codec.is_quantized_activation() => {
            (0, 1, *codec, Some(activation.block_layout()?))
        }
        [Some(activation), Some(codec)] if activation.is_quantized_activation() && !codec.is_quantized_activation() => {
            (1, 0, *codec, Some(activation.block_layout()?))
        }
        _ => return None,
    };
    let dot = block_dot_i8(&codec.block_layout()?)?;
    let reduce_dims = reduction_dims(resolved, output_axes);
    if reduce_dims.is_empty() {
        return None;
    }
    let reduce_extents: Vec<u64> = reduce_dims
        .iter()
        .map(|dim| resolved.extents.get(usize::from(*dim)).copied())
        .collect::<Option<_>>()?;
    let reduction: u64 = reduce_extents.iter().product();
    let (_, activation_layout, _) = operands.get(activation_slot)?;
    let shared_by_every_row = output_axes
        .iter()
        .all(|axis| activation_layout.stride(*axis) == 0 || resolved.extents[usize::from(*axis)] == 1);
    let contiguous_activation = is_contiguous_run(activation_layout, &reduce_dims, &reduce_extents);
    let readable_activation = external.is_none() || (contiguous_activation && activation_layout.base == 0);
    let weight_only: Vec<Option<Codec>> = quantized
        .iter()
        .enumerate()
        .map(|(slot, codec)| if slot == activation_slot { None } else { *codec })
        .collect();
    let output_total: u64 = output_axes.iter().map(|axis| resolved.extents[usize::from(*axis)]).product();
    let stageable = reduction > 0
        && output_total <= u64::from(u32::MAX)
        && reduction <= MAX_STAGED_REDUCTION
        && reduction.is_multiple_of(dot.elements as u64);
    (is_product
        && stageable
        && shared_by_every_row
        && epilogue_broadcast_axes.is_empty()
        && readable_activation
        && gather_count(resolved) == 0
        && flat_packed_rows(resolved, &weight_only))
    .then_some(StagedMatvec {
        weight_slot,
        activation_slot,
        reduce_dims,
        contiguous_activation,
        reduce_extents,
        reduction,
        scratch: external.is_none() && reduction + reduction / DOT_ELEMENTS as u64 * 4 > SMEM_STAGE_LIMIT_BYTES,
        external,
        rows: shaped_rows(dot.rows_per_warp, output_total, device.multiprocessors),
        dot,
    })
}

/// Threads to launch: one block per `STAGED_WARPS * rows_per_warp` output rows.
pub(super) fn staged_matvec_threads(resolved: &BoundOp, plan: &StagedMatvec) -> u64 {
    let output_total: u64 = match &resolved.kind {
        BoundOpKind::Reduce { output_axes, .. } => output_axes
            .iter()
            .map(|axis| resolved.extents[usize::from(*axis)])
            .product(),
        _ => 0,
    };
    let rows_per_block = (STAGED_WARPS * plan.rows) as u64;
    output_total.div_ceil(rows_per_block) * STAGED_THREADS
}

/// The kernel body after `kernel_signature`'s opening lines.
pub(super) fn push_staged_matvec_body(
    source: &mut String,
    resolved: &BoundOp,
    plan: &StagedMatvec,
) -> Result<(), EmitError> {
    let BoundOpKind::Reduce { output_axes, .. } = &resolved.kind else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "keep::reduce fold",
            found: resolved.kind.name(),
        });
    };
    source.push_str("    const unsigned int lane = threadIdx.x & 31u;\n");
    source.push_str("    const unsigned int warp = threadIdx.x >> 5u;\n");
    push_row_setup(source, resolved, plan, output_axes);
    push_stage(source, plan);
    push_direct_fold(source, plan);
    push_write(source, resolved, output_axes.len(), plan.rows);
    Ok(())
}

/// Groups of 32 one warp loads before it reduces any, so their loads overlap.
const STAGE_BATCH: usize = 8;

/// `long act_offset = ...` for flat reduction index `flat`, walking every reduced axis.
fn activation_offset(plan: &StagedMatvec, flat: &str) -> String {
    let activation_slot = plan.activation_slot;
    let mut text = format!(
        "            long act_offset = u.operand_base[{activation_slot}];\n            unsigned int act_rest = (unsigned int)({flat});\n"
    );
    // the extents are literals so the walk is multiply-shift, not a 64-bit divide per element
    for (dim, extent) in plan.reduce_dims.iter().zip(&plan.reduce_extents).rev() {
        text.push_str(&format!(
            "            act_offset += (long)(act_rest % {extent}u) * u.operand_strides[{activation_slot}][{dim}];\n\
             \x20           act_rest /= {extent}u;\n"
        ));
    }
    text
}

/// Eight lanes own one group of 32 as four `float4` loads' worth, so a warp quantizes four
/// groups per pass with three shuffles each instead of one element per lane and five.
fn push_contiguous_stage(source: &mut String, plan: &StagedMatvec) {
    let activation_slot = plan.activation_slot;
    let groups = plan.reduction / DOT_ELEMENTS as u64;
    let groups_per_pass = STAGED_WARPS * 4;
    source.push_str(&format!(
        "    __shared__ __align__(16) int staged_q[{words}];\n\
         \x20   __shared__ float staged_d[{groups}];\n\
         \x20   {{\n\
         \x20       const float4 *activation_quads = (const float4 *)(in{activation_slot} + u.operand_base[{activation_slot}]);\n\
         \x20       const unsigned int quad_in_group = lane & 7u;\n\
         \x20       #pragma unroll 4\n\
         \x20       for (unsigned int first = warp * 4u; first < {groups}u; first += {groups_per_pass}u) {{\n\
         \x20           const unsigned int group = first + (lane >> 3u);\n\
         \x20           const float4 value = group < {groups}u ? activation_quads[group * 8u + quad_in_group] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);\n\
         \x20           float amax = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)), fmaxf(fabsf(value.z), fabsf(value.w)));\n\
         \x20           amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));\n\
         \x20           amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));\n\
         \x20           amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));\n\
         \x20           if (group >= {groups}u) {{ continue; }}\n\
         \x20           const float scale = amax / 127.0f;\n\
         \x20           const float inverse = amax == 0.0f ? 0.0f : 127.0f / amax;\n\
         \x20           const int packed = (__float2int_rn(value.x * inverse) & 0xff)\n\
         \x20               | ((__float2int_rn(value.y * inverse) & 0xff) << 8)\n\
         \x20               | ((__float2int_rn(value.z * inverse) & 0xff) << 16)\n\
         \x20               | (__float2int_rn(value.w * inverse) << 24);\n\
         \x20           staged_q[group * 8u + quad_in_group] = packed;\n\
         \x20           if (quad_in_group == 0u) {{ staged_d[group] = scale; }}\n\
         \x20       }}\n\
         \x20   }}\n\
         \x20   __syncthreads();\n",
        words = plan.reduction / 4,
    ));
}

fn push_scratch_stage(source: &mut String, plan: &StagedMatvec) {
    source.push_str(&format!(
        "    const int *staged_q = staged_scratch;\n    const float *staged_d = (const float *)(staged_scratch + {});\n",
        plan.reduction / 4
    ));
}

fn push_stage(source: &mut String, plan: &StagedMatvec) {
    if plan.external.is_some() {
        source.push_str(&format!("    const unsigned char *staged_blocks = in{};\n", plan.activation_slot));
        return;
    }
    if plan.scratch {
        push_scratch_stage(source, plan);
        return;
    }
    if plan.contiguous_activation {
        push_contiguous_stage(source, plan);
        return;
    }
    let activation_slot = plan.activation_slot;
    let groups = plan.reduction / DOT_ELEMENTS as u64;
    let stride = STAGED_WARPS * STAGE_BATCH;
    let offset = activation_offset(plan, "(long)group * 32 + lane");
    source.push_str(&format!(
        "    __shared__ __align__(16) int staged_q[{}];\n    __shared__ float staged_d[{groups}];\n",
        plan.reduction / 4
    ));
    source.push_str(&format!(
        "    for (unsigned int first = warp; first < {groups}u; first += {stride}u) {{\n\
         \x20       float values[{STAGE_BATCH}];\n\
         \x20       #pragma unroll\n\
         \x20       for (int batch = 0; batch < {STAGE_BATCH}; batch++) {{\n\
         \x20           const unsigned int group = first + batch * {STAGED_WARPS}u;\n\
         \x20           values[batch] = 0.0f;\n\
         \x20           if (group >= {groups}u) {{ continue; }}\n\
         {offset}\
         \x20           values[batch] = in{activation_slot}[act_offset];\n\
         \x20       }}\n\
         \x20       #pragma unroll\n\
         \x20       for (int batch = 0; batch < {STAGE_BATCH}; batch++) {{\n\
         \x20           const unsigned int group = first + batch * {STAGED_WARPS}u;\n\
         \x20           if (group >= {groups}u) {{ continue; }}\n\
         \x20           float amax = fabsf(values[batch]);\n\
         \x20           #pragma unroll\n\
         \x20           for (int shift = 16; shift > 0; shift >>= 1) {{ amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, shift)); }}\n\
         \x20           float scale = amax / 127.0f;\n\
         \x20           int level = amax == 0.0f ? 0 : __float2int_rn(values[batch] / scale);\n\
         \x20           ((signed char *)staged_q)[group * 32u + lane] = (signed char)level;\n\
         \x20           if (lane == 0u) {{ staged_d[group] = scale; }}\n\
         \x20       }}\n\
         \x20   }}\n\
         \x20   __syncthreads();\n"
    ));
}

fn push_row_setup(source: &mut String, resolved: &BoundOp, plan: &StagedMatvec, output_axes: &[u16]) {
    let rows = plan.rows;
    let weight_slot = plan.weight_slot;
    let (_, weight_layout, _) = &resolved.operands()[weight_slot];
    for index in 0..output_axes.len() {
        source.push_str(&format!("    long coord{index}_rows[{rows}];\n"));
    }
    if plan.rows_in_shared() {
        source.push_str(&format!("    __shared__ long block_row_byte[{}];\n", STAGED_WARPS * rows));
    }
    source.push_str(&format!(
        "    const long first_row = ((long)blockIdx.x * {STAGED_WARPS} + (long)warp) * {rows};\n\
         \x20   long row_block[{rows}];\n\
         \x20   long out_offset[{rows}];\n\
         \x20   #pragma unroll\n\
         \x20   for (int row = 0; row < {rows}; row++) {{\n\
         \x20       unsigned int remaining = (unsigned int)(first_row + row);\n\
         \x20       long packed_row = 0;\n\
         \x20       out_offset[row] = u.out_base;\n"
    ));
    for (index, axis) in output_axes.iter().enumerate().rev() {
        let extent = resolved.extents[usize::from(*axis)];
        source.push_str(&format!(
            "        long coord{index} = (long)(remaining % {extent}u); remaining /= {extent}u;\n\
             \x20       out_offset[row] += coord{index} * u.out_strides[{axis}];\n\
             \x20       coord{index}_rows[row] = coord{index};\n"
        ));
    }
    for (index, axis) in output_axes.iter().enumerate() {
        if weight_layout.stride(*axis) != 0 {
            source.push_str(&format!(
                "        packed_row = packed_row * {} + coord{index};\n",
                resolved.extents[usize::from(*axis)]
            ));
        }
    }
    let elements = plan.dot.elements;
    let block_bytes = plan.dot.block_bytes;
    source.push_str(&format!(
        "        const long row_element = u.operand_base[{weight_slot}] + packed_row * {};\n\
         \x20       if (row_element % {elements} != 0) {{ __trap(); }}\n\
         \x20       row_block[row] = row_element / {elements};\n\
         {publish}\
         \x20   }}\n",
        plan.reduction,
        publish = if plan.rows_in_shared() {
            format!("        if (lane == 0u) {{ block_row_byte[warp * {rows} + row] = first_row + row < u.output_total ? row_block[row] * {block_bytes} : -1; }}\n")
        } else {
            String::new()
        }
    ));
    if plan.rows_in_shared() {
        source.push_str("    __syncthreads();\n");
    }
}

fn staged_activation_load() -> String {
    "        const int4 low = *(const int4 *)(staged_q + chunk * 8u);\n\
             const int4 high = *(const int4 *)(staged_q + chunk * 8u + 4u);\n\
             const int activation[8] = { low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w };\n\
             const float activation_scale = staged_d[chunk];\n"
        .to_owned()
}

fn external_activation_load(layout: &BlockLayout) -> String {
    let block = activation_block_offset(layout, "chunk");
    let scale = activation_scale_offset(layout, "chunk");
    let codes = activation_code_words(layout, &format!("(activation_block + {})", activation_code_offset(layout, "chunk")), "        ");
    format!(
        "        const unsigned char *activation_block = staged_blocks + {block};\n\
         {codes}\
         \x20       const int activation[8] = {{ (int)activation_low.x, (int)activation_low.y, (int)activation_low.z, (int)activation_low.w, (int)activation_high.x, (int)activation_high.y, (int)activation_high.z, (int)activation_high.w }};\n\
         \x20       const float activation_scale = __half2float(__ushort_as_half(*(const unsigned short *)(activation_block + {scale})));\n"
    )
}

/// Each lane folds chunks `lane, lane + 32, ...` of its warp's rows straight from global memory:
/// a warp step covers 32 consecutive chunks of every row, so the loads coalesce without a shared
/// tile, and no barrier sits between the weight stream and the dot.
fn push_direct_fold(source: &mut String, plan: &StagedMatvec) {
    let weight_slot = plan.weight_slot;
    let BlockDot {
        name,
        elements,
        block_bytes,
        centering,
        chunk_terms,
        ..
    } = &plan.dot;
    let rows = plan.rows;
    let chunks_per_block = elements / DOT_ELEMENTS;
    let total_chunks = plan.reduction / DOT_ELEMENTS as u64;
    let (activation_sum, centered) = match chunk_terms {
        Some(terms) => (term_activation_sum(*terms), String::new()),
        None => centering_text(*centering),
    };
    let row_base = if plan.rows_in_shared() {
        format!("block_row_byte[warp * {rows} + row]")
    } else {
        format!("first_row + row < u.output_total ? row_block[row] * {block_bytes} : -1")
    };
    let activation_load = plan.external.as_ref().map_or_else(staged_activation_load, external_activation_load);
    let position = if 32 % chunks_per_block == 0 { format!("lane % {chunks_per_block}u") } else { format!("chunk % {chunks_per_block}u") };
    let row_fold = row_fold_text(name, &position, *chunk_terms, &centered);
    source.push_str(&format!(
        "    const unsigned char *row_start[{rows}];\n\
         \x20   float accumulator[{rows}];\n\
         \x20   #pragma unroll\n\
         \x20   for (int row = 0; row < {rows}; row++) {{\n\
         \x20       const long base = {row_base};\n\
         \x20       row_start[row] = base < 0 ? (const unsigned char *)0 : in{weight_slot} + base;\n\
         \x20       accumulator[row] = 0.0f;\n\
         \x20   }}\n\
         \x20   #pragma unroll 2\n\
         \x20   for (unsigned int chunk = lane; chunk < {total_chunks}u; chunk += 32u) {{\n\
{activation_load}\
         {activation_sum}\
         \x20       const unsigned int block_offset = (chunk / {chunks_per_block}u) * {block_bytes}u;\n\
         \x20       #pragma unroll\n\
         \x20       for (int row = 0; row < {rows}; row++) {{\n\
         \x20           if (row_start[row] == 0) {{ continue; }}\n\
         \x20           const unsigned char *block = row_start[row] + block_offset;\n\
         \x20           int words[8];\n\
         \x20           {name}_i8x32(block, {position}, words);\n\
{row_fold}\
         \x20       }}\n\
         \x20   }}\n"
    ));
}

const DP4A_ROW: &str = "            #pragma unroll\n            for (int word = 0; word < 8; word++) { dot = __dp4a(words[word], activation[word], dot); }\n";

/// The exact int32 sum of the chunk's activation codes, which the min term multiplies by the
/// activation scale.
fn term_activation_sum(terms: ChunkTerms) -> String {
    if !terms.min {
        return String::new();
    }
    "        int activation_sum = 0;\n        #pragma unroll\n        for (int word = 0; word < 8; word++) { activation_sum = __dp4a(0x01010101, activation[word], activation_sum); }\n".to_owned()
}

fn term_row_fold(name: &str, position: &str, terms: ChunkTerms) -> String {
    let scale = |part: usize| format!("{name}_chunk_scale(block, {position}, {part}u)");
    let mut text = if terms.parts == 1 {
        format!("            int dot = 0;\n{DP4A_ROW}            float contribution = {} * (float)dot;\n", scale(0))
    } else {
        format!(
            "            int dot_low = 0;\n            int dot_high = 0;\n            #pragma unroll\n            for (int word = 0; word < 4; word++) {{ dot_low = __dp4a(words[word], activation[word], dot_low); dot_high = __dp4a(words[4 + word], activation[4 + word], dot_high); }}\n            float contribution = {} * (float)dot_low + {} * (float)dot_high;\n",
            scale(0),
            scale(1)
        )
    };
    text.push_str("            accumulator[row] += activation_scale * contribution;\n");
    if terms.min {
        text.push_str(&format!(
            "            accumulator[row] += {name}_chunk_min(block, {position}) * activation_scale * (float)activation_sum;\n"
        ));
    }
    text
}

fn row_fold_text(name: &str, position: &str, chunk_terms: Option<ChunkTerms>, centered: &str) -> String {
    match chunk_terms {
        Some(terms) => term_row_fold(name, position, terms),
        None => format!(
            "            int dot = 0;\n{DP4A_ROW}{centered}            accumulator[row] += {name}_scale(block) * activation_scale * (float)dot;\n"
        ),
    }
}

fn prepass_warps(plan: &StagedMatvec) -> u64 {
    let groups = plan.reduction / DOT_ELEMENTS as u64;
    if plan.contiguous_activation { groups.div_ceil(4) } else { groups }
}

fn contiguous_prepass_body(plan: &StagedMatvec) -> String {
    let activation_slot = plan.activation_slot;
    let groups = plan.reduction / DOT_ELEMENTS as u64;
    format!(
        "    const unsigned int lane = threadIdx.x & 31u;\n\
         \x20   const unsigned int quad_in_group = lane & 7u;\n\
         \x20   const unsigned int group = (blockIdx.x * {STAGED_WARPS}u + (threadIdx.x >> 5u)) * 4u + (lane >> 3u);\n\
         \x20   const float4 *activation_quads = (const float4 *)(in{activation_slot} + u.operand_base[{activation_slot}]);\n\
         \x20   const float4 value = group < {groups}u ? activation_quads[group * 8u + quad_in_group] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);\n\
         \x20   float amax = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)), fmaxf(fabsf(value.z), fabsf(value.w)));\n\
         \x20   amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));\n\
         \x20   amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));\n\
         \x20   amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));\n\
         \x20   if (group >= {groups}u) {{ return; }}\n\
         \x20   const float scale = amax / 127.0f;\n\
         \x20   const float inverse = amax == 0.0f ? 0.0f : 127.0f / amax;\n\
         \x20   const int packed = (__float2int_rn(value.x * inverse) & 0xff)\n\
         \x20       | ((__float2int_rn(value.y * inverse) & 0xff) << 8)\n\
         \x20       | ((__float2int_rn(value.z * inverse) & 0xff) << 16)\n\
         \x20       | (__float2int_rn(value.w * inverse) << 24);\n\
         \x20   staged_scratch[group * 8u + quad_in_group] = packed;\n\
         \x20   if (quad_in_group == 0u) {{ ((float *)staged_scratch)[{words}u + group] = scale; }}\n",
        words = plan.reduction / 4,
    )
}

fn strided_prepass_body(plan: &StagedMatvec) -> String {
    let activation_slot = plan.activation_slot;
    let groups = plan.reduction / DOT_ELEMENTS as u64;
    let offset = activation_offset(plan, "(long)group * 32 + lane");
    format!(
        "    const unsigned int lane = threadIdx.x & 31u;\n\
         \x20   const unsigned int group = blockIdx.x * {STAGED_WARPS}u + (threadIdx.x >> 5u);\n\
         \x20   float value = 0.0f;\n\
         \x20   if (group < {groups}u) {{\n\
         {offset}\
         \x20       value = in{activation_slot}[act_offset];\n\
         \x20   }}\n\
         \x20   float amax = fabsf(value);\n\
         \x20   #pragma unroll\n\
         \x20   for (int shift = 16; shift > 0; shift >>= 1) {{ amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, shift)); }}\n\
         \x20   if (group >= {groups}u) {{ return; }}\n\
         \x20   const float scale = amax / 127.0f;\n\
         \x20   const int level = amax == 0.0f ? 0 : __float2int_rn(value / scale);\n\
         \x20   ((signed char *)staged_scratch)[group * 32u + lane] = (signed char)level;\n\
         \x20   if (lane == 0u) {{ ((float *)staged_scratch)[{words}u + group] = scale; }}\n",
        words = plan.reduction / 4,
    )
}

pub(super) fn staged_prepass_grid(plan: &StagedMatvec) -> CudaGridSpec {
    CudaGridSpec { threads: prepass_warps(plan) * WARP_SIZE, block_width: Some(STAGED_THREADS) }
}

/// The launch that fills the global quantized activation before the matvec reads it. It shares
/// the matvec's `Uniforms` and helper preamble, taken from `main_source` up to its signature.
pub(super) fn staged_prepass(
    resolved: &BoundOp,
    plan: &StagedMatvec,
    entry: &str,
    main_source: &str,
) -> Result<CudaKernel, EmitError> {
    let prefix_end = main_source.find("extern \"C\"").ok_or(EmitError::RenderKindMismatch {
        node: resolved.node,
        expected: "kernel signature",
        found: "source without an extern entry",
    })?;
    let (activation_node, _, _) = &resolved.operands()[plan.activation_slot];
    let activation_slot = plan.activation_slot;
    let body = if plan.contiguous_activation { contiguous_prepass_body(plan) } else { strided_prepass_body(plan) };
    let source = format!(
        "{}extern \"C\" __global__ void __launch_bounds__({STAGED_THREADS}) {entry}_prepass(\n\
         \x20   const float* __restrict__ in{activation_slot},\n\
         \x20   int* __restrict__ staged_scratch,\n\
         \x20   const Uniforms* __restrict__ u_ptr\n\
         ) {{\n\
         \x20   const Uniforms u = *u_ptr;\n\
         {body}}}\n",
        &main_source[..prefix_end]
    );
    Ok(CudaKernel {
        source,
        entry: format!("{entry}_prepass"),
        bindings: vec![
            Binding::Input(*activation_node),
            Binding::Output(staged_scratch_node(resolved.node)),
            Binding::Uniforms,
        ],
        grid: staged_prepass_grid(plan),
        extra_outputs: Vec::new(),
        prologue: Vec::new(),
    })
}

// words carry raw codes so recentering is one subtraction per chunk, not a per-word swar sub
fn centering_text(centering: Option<(i16, i16)>) -> (String, String) {
    let Some((multiplier, bias)) = centering else {
        return (String::new(), String::new());
    };
    let sum = if bias == 0 {
        String::new()
    } else {
        "        int activation_sum = 0;\n        #pragma unroll\n        for (int word = 0; word < 8; word++) { activation_sum = __dp4a(0x01010101, activation[word], activation_sum); }\n".to_owned()
    };
    let centered = if bias == 0 {
        format!("            dot = {multiplier} * dot;\n")
    } else {
        format!("            dot = {multiplier} * dot - {bias} * activation_sum;\n")
    };
    (sum, centered)
}

fn push_write(source: &mut String, resolved: &BoundOp, output_rank: usize, rows: usize) {
    source.push_str(&format!(
        "    #pragma unroll\n\
         \x20   for (int row = 0; row < {rows}; row++) {{\n\
         \x20       float total = accumulator[row];\n\
         \x20       #pragma unroll\n\
         \x20       for (int shift = 16; shift > 0; shift >>= 1) {{ total += __shfl_down_sync(0xffffffffu, total, shift); }}\n\
         \x20       if (lane == 0u && first_row + row < u.output_total) {{\n"
    ));
    if let BoundOpKind::Reduce {
        epilogue_body,
        epilogue_operands,
        ..
    } = &resolved.kind
    {
        push_reduce_epilogue_write(
            source,
            epilogue_body,
            epilogue_operands,
            output_rank,
            "float",
            "            ",
            |dim| format!("coord{dim}_rows[row]"),
            "total",
            "out_offset[row]",
        );
    }
    source.push_str("        }\n    }\n");
}
