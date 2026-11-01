use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_tensor::{BoundOp, BoundOpKind, DType, Keep, NumericPolicy, ReduceInit, ScalarOp, StepArg};

use super::block_dot::{BlockDot, DOT_ELEMENTS, block_dot_i8};
use super::{Codec, EmitError, gather_count, push_reduce_epilogue_write, reduction_dims};

/// Warps in one block of a staged matvec.
pub(super) const STAGED_WARPS: usize = 8;

/// Output rows one warp folds, each against the same staged operand.
pub(super) const STAGED_ROWS_PER_WARP: usize = 4;

/// Threads in one block of a staged matvec.
pub(super) const STAGED_THREADS: u64 = (STAGED_WARPS * 32) as u64;

/// Largest reduction a block can stage: int8 values plus one f32 scale per
/// 32 elements must fit the 48 KiB static shared-memory limit.
const MAX_STAGED_REDUCTION: u64 = 32_768;

/// `sum(packed * shared)` lowered as: the operand every output row reads at the
/// same coordinates is staged once per block in shared memory, rounded to int8
/// groups of 32, and each warp folds `STAGED_ROWS_PER_WARP` rows against it with
/// `dp4a`. The packed operand's half comes from its [`BlockDot`], which is
/// derived from the codec's block layout, so the rule names no format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct StagedMatvec {
    pub weight_slot: usize,
    pub activation_slot: usize,
    pub reduce_dims: Vec<u16>,
    pub reduction: u64,
    pub dot: BlockDot,
}

pub(super) fn staged_matvec_plan(
    resolved: &BoundOp,
    quantized: &[Option<Codec>],
    numeric_policy: NumericPolicy,
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
    let (weight_slot, activation_slot, codec) = match quantized {
        [Some(codec), None] => (0, 1, *codec),
        [None, Some(codec)] => (1, 0, *codec),
        _ => return None,
    };
    let dot = block_dot_i8(&codec.block_layout()?)?;
    let reduce_dims = reduction_dims(resolved, output_axes);
    if reduce_dims.is_empty() {
        return None;
    }
    let reduction: u64 = reduce_dims
        .iter()
        .map(|dim| resolved.extents.get(usize::from(*dim)).copied())
        .product::<Option<u64>>()?;
    let (_, activation_layout, _) = operands.get(activation_slot)?;
    let shared_by_every_row = output_axes
        .iter()
        .all(|axis| activation_layout.stride(*axis) == 0 || resolved.extents[usize::from(*axis)] == 1);
    let stageable = reduction > 0
        && reduction <= MAX_STAGED_REDUCTION
        && reduction.is_multiple_of(dot.elements as u64);
    (is_product
        && stageable
        && shared_by_every_row
        && epilogue_broadcast_axes.is_empty()
        && gather_count(resolved) == 0)
    .then_some(StagedMatvec {
        weight_slot,
        activation_slot,
        reduce_dims,
        reduction,
        dot,
    })
}

/// Threads to launch: one block per `STAGED_WARPS * STAGED_ROWS_PER_WARP` output rows.
pub(super) fn staged_matvec_threads(resolved: &BoundOp) -> u64 {
    let output_total: u64 = match &resolved.kind {
        BoundOpKind::Reduce { output_axes, .. } => output_axes
            .iter()
            .map(|axis| resolved.extents[usize::from(*axis)])
            .product(),
        _ => 0,
    };
    let rows_per_block = (STAGED_WARPS * STAGED_ROWS_PER_WARP) as u64;
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
    push_fold_prologue(source, plan);
    push_stage(source, plan);
    push_fold(source, plan);
    push_write(source, resolved, output_axes.len());
    Ok(())
}

/// Groups of 32 one warp loads before it reduces any, so their loads overlap.
const STAGE_BATCH: usize = 8;

/// `long act_offset = ...` for flat reduction index `flat`, walking every reduced axis.
fn activation_offset(plan: &StagedMatvec, flat: &str) -> String {
    let activation_slot = plan.activation_slot;
    let mut text = format!(
        "            long act_offset = u.operand_base[{activation_slot}];\n            long act_rest = {flat};\n"
    );
    for (index, dim) in plan.reduce_dims.iter().enumerate().rev() {
        text.push_str(&format!(
            "            act_offset += (act_rest % u.reduction_extents[{index}]) * u.operand_strides[{activation_slot}][{dim}];\n\
             \x20           act_rest /= u.reduction_extents[{index}];\n"
        ));
    }
    text
}

fn push_stage(source: &mut String, plan: &StagedMatvec) {
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
    let weight_slot = plan.weight_slot;
    let (_, weight_layout, _) = &resolved.operands()[weight_slot];
    for index in 0..output_axes.len() {
        source.push_str(&format!("    long coord{index}_rows[{STAGED_ROWS_PER_WARP}];\n"));
    }
    source.push_str(&format!("    __shared__ long block_row_byte[{}];\n", STAGED_WARPS * STAGED_ROWS_PER_WARP));
    source.push_str(&format!(
        "    const long first_row = ((long)blockIdx.x * {STAGED_WARPS} + (long)warp) * {STAGED_ROWS_PER_WARP};\n\
         \x20   long row_block[{STAGED_ROWS_PER_WARP}];\n\
         \x20   long out_offset[{STAGED_ROWS_PER_WARP}];\n\
         \x20   #pragma unroll\n\
         \x20   for (int row = 0; row < {STAGED_ROWS_PER_WARP}; row++) {{\n\
         \x20       long remaining = first_row + row;\n\
         \x20       long packed_row = 0;\n\
         \x20       out_offset[row] = u.out_base;\n"
    ));
    for (index, axis) in output_axes.iter().enumerate().rev() {
        source.push_str(&format!(
            "        long coord{index} = remaining % u.output_extents[{index}]; remaining /= u.output_extents[{index}];\n\
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
         \x20       if (lane == 0u) {{ block_row_byte[warp * {STAGED_ROWS_PER_WARP} + row] = first_row + row < u.output_total ? row_block[row] * {block_bytes} : -1; }}\n\
         \x20   }}\n",
        plan.reduction
    ));
    source.push_str("    __syncthreads();\n");
}

/// Byte budget for one row's share of a weight tile: small enough that 32 rows
/// fit beside the staged operand with several blocks resident per SM.
const TILE_ROW_BYTES: usize = 320;

/// How the reduction is cut into weight tiles: `blocks` packed blocks per tile,
/// each row's tile copied as `segments` aligned 16-byte pieces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TilePlan {
    blocks: usize,
    tiles: usize,
    total_blocks: usize,
    segments: usize,
    loads_per_thread: usize,
}

fn tile_plan(plan: &StagedMatvec) -> TilePlan {
    let dot = &plan.dot;
    let chunks_per_block = dot.elements / DOT_ELEMENTS;
    let mut blocks = (32 / chunks_per_block).max(1);
    while blocks > 1 && blocks * dot.block_bytes > TILE_ROW_BYTES {
        blocks /= 2;
    }
    let total_blocks = usize::try_from(plan.reduction).unwrap_or(usize::MAX) / dot.elements;
    let segments = (blocks * dot.block_bytes).div_ceil(16) + 1;
    let rows = STAGED_WARPS * STAGED_ROWS_PER_WARP;
    TilePlan {
        blocks,
        tiles: total_blocks.div_ceil(blocks),
        total_blocks,
        segments,
        loads_per_thread: (rows * segments).div_ceil(STAGED_THREADS as usize),
    }
}

/// Issues the global loads of tile `tile_expr` into the `prefetch` registers.
fn push_tile_load(source: &mut String, plan: &StagedMatvec, tiles: &TilePlan, tile_expr: &str) {
    let weight_slot = plan.weight_slot;
    let tile_bytes = tiles.blocks * plan.dot.block_bytes;
    let row_bytes = tiles.total_blocks * plan.dot.block_bytes;
    let TilePlan {
        segments,
        loads_per_thread,
        ..
    } = *tiles;
    let slots = STAGED_WARPS * STAGED_ROWS_PER_WARP * segments;
    source.push_str(&format!(
        "        #pragma unroll\n\
         \x20       for (int load = 0; load < {loads_per_thread}; load++) {{\n\
         \x20           const unsigned int slot = threadIdx.x + load * {STAGED_THREADS}u;\n\
         \x20           prefetch[load] = make_uint4(0u, 0u, 0u, 0u);\n\
         \x20           if (slot >= {slots}u) {{ continue; }}\n\
         \x20           const long base = block_row_byte[slot / {segments}u];\n\
         \x20           if (base < 0) {{ continue; }}\n\
         \x20           const long start = base + (long)({tile_expr}) * {tile_bytes};\n\
         \x20           const long aligned = start & ~15L;\n\
         \x20           const long left_bytes = (long){row_bytes} - (long)({tile_expr}) * {tile_bytes};\n\
         \x20           const long remaining_bytes = left_bytes < {tile_bytes} ? left_bytes : (long){tile_bytes};\n\
         \x20           const unsigned int segment = slot % {segments}u;\n\
         \x20           if ((long)segment * 16 < (start - aligned) + remaining_bytes) {{\n\
         \x20               prefetch[load] = __ldg((const uint4 *)(in{weight_slot} + aligned) + segment);\n\
         \x20           }}\n\
         \x20       }}\n"
    ));
}

/// Declares the tile buffers and issues tile 0's loads, so they are in flight
/// while the shared operand is staged.
fn push_fold_prologue(source: &mut String, plan: &StagedMatvec) {
    let tiles = tile_plan(plan);
    let rows = STAGED_WARPS * STAGED_ROWS_PER_WARP;
    source.push_str(&format!(
        "    __shared__ uint4 weight_tile[{}];\n\
         \x20   uint4 prefetch[{}];\n\
         \x20   {{\n",
        rows * tiles.segments,
        tiles.loads_per_thread
    ));
    push_tile_load(source, plan, &tiles, "0");
    source.push_str("    }\n");
}

fn push_fold(source: &mut String, plan: &StagedMatvec) {
    let BlockDot {
        name,
        elements,
        block_bytes,
        ..
    } = &plan.dot;
    let chunks_per_block = elements / DOT_ELEMENTS;
    let tiles = tile_plan(plan);
    let TilePlan {
        blocks,
        tiles: tile_count,
        total_blocks,
        segments,
        loads_per_thread,
    } = tiles;
    let rows = STAGED_WARPS * STAGED_ROWS_PER_WARP;
    let tile_bytes = blocks * block_bytes;
    source.push_str(&format!(
        "    float accumulator[{STAGED_ROWS_PER_WARP}];\n\
         \x20   #pragma unroll\n\
         \x20   for (int row = 0; row < {STAGED_ROWS_PER_WARP}; row++) {{ accumulator[row] = 0.0f; }}\n\
         \x20   for (int tile = 0; tile < {tile_count}; tile++) {{\n\
         \x20       #pragma unroll\n\
         \x20       for (int load = 0; load < {loads_per_thread}; load++) {{\n\
         \x20           const unsigned int slot = threadIdx.x + load * {STAGED_THREADS}u;\n\
         \x20           if (slot < {}u) {{ weight_tile[slot] = prefetch[load]; }}\n\
         \x20       }}\n\
         \x20       __syncthreads();\n\
         \x20       if (tile + 1 < {tile_count}) {{\n",
        rows * segments
    ));
    push_tile_load(source, plan, &tiles, "tile + 1");
    source.push_str(&format!(
        "        }}\n\
         \x20       const int left_blocks = {total_blocks} - tile * {blocks};\n\
         \x20       const unsigned int tile_chunks = (unsigned int)(left_blocks < {blocks} ? left_blocks : {blocks}) * {chunks_per_block}u;\n\
         \x20       for (unsigned int local = lane; local < tile_chunks; local += 32u) {{\n\
         \x20           const unsigned int chunk = (unsigned int)tile * {blocks}u * {chunks_per_block}u + local;\n\
         \x20           const int4 low = *(const int4 *)(staged_q + chunk * 8u);\n\
         \x20           const int4 high = *(const int4 *)(staged_q + chunk * 8u + 4u);\n\
         \x20           const int activation[8] = {{ low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w }};\n\
         \x20           const float activation_scale = staged_d[chunk];\n\
         \x20           #pragma unroll\n\
         \x20           for (int row = 0; row < {STAGED_ROWS_PER_WARP}; row++) {{\n\
         \x20               const int local_row = (int)warp * {STAGED_ROWS_PER_WARP} + row;\n\
         \x20               const long base = block_row_byte[local_row];\n\
         \x20               if (base < 0) {{ continue; }}\n\
         \x20               const unsigned int shift = (unsigned int)((base + (long)tile * {tile_bytes}) & 15L);\n\
         \x20               const unsigned char *block = (const unsigned char *)(weight_tile + local_row * {segments}) + shift + (local / {chunks_per_block}u) * {block_bytes}u;\n\
         \x20               int words[8];\n\
         \x20               {name}_i8x32(block, local % {chunks_per_block}u, words);\n\
         \x20               int dot = 0;\n\
         \x20               #pragma unroll\n\
         \x20               for (int word = 0; word < 8; word++) {{ dot = __dp4a(words[word], activation[word], dot); }}\n\
         \x20               accumulator[row] += {name}_scale(block) * activation_scale * (float)dot;\n\
         \x20           }}\n\
         \x20       }}\n\
         \x20       __syncthreads();\n\
         \x20   }}\n"
    ));
}

fn push_write(source: &mut String, resolved: &BoundOp, output_rank: usize) {
    source.push_str(&format!(
        "    #pragma unroll\n\
         \x20   for (int row = 0; row < {STAGED_ROWS_PER_WARP}; row++) {{\n\
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
