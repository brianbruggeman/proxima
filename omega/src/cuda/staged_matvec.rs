use alloc::format;
use alloc::string::String;

use proxima_tensor::{BoundOp, BoundOpKind, DType, Keep, NumericPolicy, ReduceInit, ScalarOp, StepArg};

use super::block_dot::{BlockDot, DOT_ELEMENTS, block_dot_i8};
use super::{Codec, EmitError, gather_count, reduction_dims};
use crate::identity::reduce_epilogue_is_identity;

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
    pub reduce_dim: u16,
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
        epilogue_body,
        epilogue_operands,
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
    let [reduce_dim] = reduce_dims.as_slice() else {
        return None;
    };
    let reduction = *resolved.extents.get(usize::from(*reduce_dim))?;
    let (_, activation_layout, _) = operands.get(activation_slot)?;
    let shared_by_every_row = output_axes
        .iter()
        .all(|axis| activation_layout.stride(*axis) == 0);
    let stageable = reduction > 0
        && reduction <= MAX_STAGED_REDUCTION
        && reduction.is_multiple_of(dot.elements as u64);
    (is_product
        && stageable
        && shared_by_every_row
        && epilogue_broadcast_axes.is_empty()
        && gather_count(resolved) == 0
        && reduce_epilogue_is_identity(epilogue_body, epilogue_operands))
    .then_some(StagedMatvec {
        weight_slot,
        activation_slot,
        reduce_dim: *reduce_dim,
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
    push_stage(source, plan);
    push_row_setup(source, resolved, plan, output_axes);
    push_fold(source, plan);
    push_write(source);
    Ok(())
}

fn push_stage(source: &mut String, plan: &StagedMatvec) {
    let activation_slot = plan.activation_slot;
    let reduce_dim = plan.reduce_dim;
    let groups = plan.reduction / DOT_ELEMENTS as u64;
    source.push_str(&format!(
        "    __shared__ __align__(16) int staged_q[{}];\n    __shared__ float staged_d[{groups}];\n",
        plan.reduction / 4
    ));
    source.push_str("    const unsigned int lane = threadIdx.x & 31u;\n");
    source.push_str("    const unsigned int warp = threadIdx.x >> 5u;\n");
    source.push_str(&format!(
        "    const long act_base = u.operand_base[{activation_slot}];\n\
         \x20   const long act_step = u.operand_strides[{activation_slot}][{reduce_dim}];\n\
         \x20   for (unsigned int group = warp; group < {groups}u; group += {STAGED_WARPS}u) {{\n\
         \x20       float value = in{activation_slot}[act_base + (long)(group * 32u + lane) * act_step];\n\
         \x20       float amax = fabsf(value);\n\
         \x20       #pragma unroll\n\
         \x20       for (int shift = 16; shift > 0; shift >>= 1) {{ amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, shift)); }}\n\
         \x20       float scale = amax / 127.0f;\n\
         \x20       int level = amax == 0.0f ? 0 : __float2int_rn(value / scale);\n\
         \x20       ((signed char *)staged_q)[group * 32u + lane] = (signed char)level;\n\
         \x20       if (lane == 0u) {{ staged_d[group] = scale; }}\n\
         \x20   }}\n\
         \x20   __syncthreads();\n"
    ));
}

fn push_row_setup(source: &mut String, resolved: &BoundOp, plan: &StagedMatvec, output_axes: &[u16]) {
    let weight_slot = plan.weight_slot;
    let (_, weight_layout, _) = &resolved.operands()[weight_slot];
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
             \x20       out_offset[row] += coord{index} * u.out_strides[{axis}];\n"
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
    source.push_str(&format!(
        "        const long row_element = u.operand_base[{weight_slot}] + packed_row * {};\n\
         \x20       if (row_element % {elements} != 0) {{ __trap(); }}\n\
         \x20       row_block[row] = row_element / {elements};\n\
         \x20   }}\n",
        plan.reduction
    ));
}

fn push_fold(source: &mut String, plan: &StagedMatvec) {
    let weight_slot = plan.weight_slot;
    let BlockDot {
        name,
        elements,
        block_bytes,
        ..
    } = &plan.dot;
    let chunks = plan.reduction / DOT_ELEMENTS as u64;
    let chunks_per_block = elements / DOT_ELEMENTS;
    source.push_str(&format!(
        "    float accumulator[{STAGED_ROWS_PER_WARP}];\n\
         \x20   #pragma unroll\n\
         \x20   for (int row = 0; row < {STAGED_ROWS_PER_WARP}; row++) {{ accumulator[row] = 0.0f; }}\n\
         \x20   for (unsigned int chunk = lane; chunk < {chunks}u; chunk += 32u) {{\n\
         \x20       const int4 low = *(const int4 *)(staged_q + chunk * 8u);\n\
         \x20       const int4 high = *(const int4 *)(staged_q + chunk * 8u + 4u);\n\
         \x20       const int activation[8] = {{ low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w }};\n\
         \x20       const float activation_scale = staged_d[chunk];\n\
         \x20       #pragma unroll\n\
         \x20       for (int row = 0; row < {STAGED_ROWS_PER_WARP}; row++) {{\n\
         \x20           if (first_row + row >= u.output_total) {{ continue; }}\n\
         \x20           const unsigned char *block = in{weight_slot} + (row_block[row] + chunk / {chunks_per_block}u) * {block_bytes};\n\
         \x20           int words[8];\n\
         \x20           {name}_i8x32(block, chunk % {chunks_per_block}u, words);\n\
         \x20           int dot = 0;\n\
         \x20           #pragma unroll\n\
         \x20           for (int word = 0; word < 8; word++) {{ dot = __dp4a(words[word], activation[word], dot); }}\n\
         \x20           accumulator[row] += {name}_scale(block) * activation_scale * (float)dot;\n\
         \x20       }}\n\
         \x20   }}\n"
    ));
}

fn push_write(source: &mut String) {
    source.push_str(&format!(
        "    #pragma unroll\n\
         \x20   for (int row = 0; row < {STAGED_ROWS_PER_WARP}; row++) {{\n\
         \x20       float total = accumulator[row];\n\
         \x20       #pragma unroll\n\
         \x20       for (int shift = 16; shift > 0; shift >>= 1) {{ total += __shfl_down_sync(0xffffffffu, total, shift); }}\n\
         \x20       if (lane == 0u && first_row + row < u.output_total) {{ out[out_offset[row]] = total; }}\n\
         \x20   }}\n"
    ));
}
