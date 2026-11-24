//! The expert-grouped contraction on CUDA: a route prepass and a grouped tensor-core kernel.
//!
//! The prepass is one block. It compacts, per expert, the flat token rows routed to it into the
//! buffer [`crate::route_compaction`] lays out; the grouped kernel is `push_mma_gemm_body` with
//! its token rows read through that list and its weight read at the tile's expert. CUDA has no
//! indirect dispatch, so the grouped kernel launches at the worst-case tile count and a block
//! whose tile index is past the device-side tile total leaves at once.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_numeric::NumericPolicy;
use proxima_tensor::{BoundOp, BoundOpKind, ContractionForm, Executes, product_contraction};

use super::mma_gemm::{
    FEATURE_TILE, GEMM_THREADS, MmaGemm, TOKEN_TILE, grouped_contraction_grant, mma_gemm_plan_for, weight_slot,
};
use super::{Codec, CudaGridSpec, CudaKernel, EmitError, gather_slots};
use crate::lowering::Binding;
use crate::route_compaction::{
    COUNT_HEADER_WORDS, ROUTE_COMPACTION_MISMATCH_FAULT, compaction_words, device_bases, tile_upper_bound,
};

pub(super) const GROUPED_TOKEN: &str = "_grp1";

pub(super) struct GroupedRoute {
    pub(super) slot: usize,
}

pub(super) struct GroupedGemm {
    pub(super) gemm: MmaGemm,
    pub(super) route: GroupedRoute,
    pub(super) experts: u64,
    pub(super) tokens: u64,
}

pub(super) fn grouped_entry(entry: &str) -> String {
    format!("{entry}{GROUPED_TOKEN}")
}

pub(super) fn prepass_entry(entry: &str) -> String {
    format!("{entry}_route_prepass")
}

/// `Some` when the grouped emitter plans this op: a gathered weight whose route follows the token
/// axes, at least `grouped_min_tokens` routed rows, and a codec the tensor-core body decodes.
pub(super) fn grouped_gemm_plan(
    resolved: &BoundOp,
    quantized: &[Option<Codec>],
    numeric_policy: NumericPolicy,
) -> Option<GroupedGemm> {
    let [first, second] = quantized else {
        return None;
    };
    let (weight, _, _) = weight_slot(*first, *second)?;
    let contraction = product_contraction(resolved, weight)?;
    let grant = grouped_contraction_grant(Executes::BASE);
    let gemm = mma_gemm_plan_for(resolved, quantized, numeric_policy, &grant, ContractionForm::Grouped)?;
    let lookup = resolved.operands()[weight].2.as_ref()?;
    let slot = gather_slots(resolved)[weight]?;
    Some(GroupedGemm {
        gemm,
        route: GroupedRoute { slot },
        experts: lookup.extent.max(1),
        tokens: contraction.token_total,
    })
}

/// The tile lookup at the head of the grouped kernel: which expert this block's token tile
/// belongs to, how many rows it carries and where its slice of the compacted list starts.
pub(super) fn push_grouped_tile_origin(source: &mut String, route: &GroupedRoute, tile_tokens: usize) {
    let slot = route.slot;
    let header = COUNT_HEADER_WORDS;
    source.push_str(&format!("    const long long experts = u.gather_extent[{slot}];\n"));
    source.push_str(&device_bases(header, "experts", "token_total"));
    source.push_str(&format!(
        "    if ((long long)route_compaction[0] != token_total) {{\n        if (tid == 0) {{ atomicMax(&fault[{slot}], {ROUTE_COMPACTION_MISMATCH_FAULT}u); }}\n        return;\n    }}\n"
    ));
    source.push_str(&format!(
        "    const long long tile_index = token_first / {tile_tokens};\n    const long long last_expert = experts - 1;\n    const long long tile_total = (long long)route_compaction[tile_start_base + last_expert] + ((long long)route_compaction[{header}ll + last_expert] + {tile_tokens} - 1) / {tile_tokens};\n    if (tile_index >= tile_total) {{ return; }}\n"
    ));
    source.push_str(&format!(
        "    const long long grouped_expert = (long long)route_compaction[tile_expert_base + tile_index];\n    const long long expert_tile = tile_index - (long long)route_compaction[tile_start_base + grouped_expert];\n    const long long expert_rows = (long long)route_compaction[{header}ll + grouped_expert];\n    const long long tile_count = min((long long){tile_tokens}, expert_rows - expert_tile * {tile_tokens});\n    const unsigned int *tile_list = route_compaction + list_base + route_compaction[{header}ll + experts + grouped_expert] + expert_tile * {tile_tokens};\n"
    ));
}

/// The prepass kernel: one block counts each expert's rows, prefix-sums them into offsets and
/// tiles, then places the flat token ids expert-major in route order.
pub(super) fn prepass_source(resolved: &BoundOp, plan: &GroupedGemm, entry: &str, index_type: &str) -> Result<String, EmitError> {
    let BoundOpKind::Reduce { output_axes, .. } = &resolved.kind else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "keep::reduce fold",
            found: resolved.kind.name(),
        });
    };
    let slot = plan.route.slot;
    let weight = plan.gemm.weight;
    let header = COUNT_HEADER_WORDS;
    let mut fetch = String::new();
    fetch.push_str(&format!(
        "            long long rest = entry;\n            long long gather_off = u.gather_index_base[{slot}];\n"
    ));
    for &position in plan.gemm.token_positions.iter().rev() {
        let axis = output_axes[position];
        fetch.push_str(&format!(
            "            gather_off += (rest % u.output_extents[{position}]) * u.gather_index_strides[{slot}][{axis}]; rest /= u.output_extents[{position}];\n"
        ));
    }
    fetch.push_str(&format!(
        "            long long fetched{weight} = live ? (long long)gather_idx{slot}[gather_off] : -1ll;\n"
    ));
    let mut source = String::new();
    source.push_str(&format!(
        "\nextern \"C\" __global__ void {entry}(\n    const {index_type}* __restrict__ gather_idx{slot},\n    const Uniforms* __restrict__ u_ptr,\n    unsigned int* __restrict__ fault,\n    unsigned int* __restrict__ route_compaction\n) {{\n    const Uniforms u = *u_ptr;\n    const int tid = (int)threadIdx.x;\n    const int warp = tid >> 5;\n    const int lane = tid & 31;\n    const int warps = (int)(blockDim.x >> 5);\n"
    ));
    source.push_str("    long long tokens = 1;\n");
    for &position in &plan.gemm.token_positions {
        source.push_str(&format!("    tokens *= u.output_extents[{position}];\n"));
    }
    source.push_str(&format!("    const long long experts = u.gather_extent[{slot}];\n"));
    source.push_str(&device_bases(header, "experts", "tokens"));
    source.push_str(&format!(
        "    for (long long expert = warp; expert < experts; expert += warps) {{\n        unsigned int tally = 0u;\n        for (long long base = 0; base < tokens; base += 32) {{\n            const long long entry = base + lane;\n            const bool live = entry < tokens;\n{fetch}"
    ));
    source.push_str("            if (expert == 0 && live) {\n");
    push_fault(&mut source, weight, slot);
    source.push_str(&format!(
        "            }}\n            fetched{weight} = live ? max((long long)0, min(fetched{weight}, u.gather_extent[{slot}] - 1)) : -1ll;\n            tally += __popc(__ballot_sync(PROXIMA_ALL_LANES, fetched{weight} == expert));\n        }}\n        if (lane == 0) {{ route_compaction[{header}ll + expert] = tally; }}\n    }}\n    __syncthreads();\n"
    ));
    source.push_str(&format!(
        "    if (tid == 0) {{\n        unsigned int running = 0u;\n        unsigned int tile_running = 0u;\n        for (long long expert = 0; expert < experts; ++expert) {{\n            const unsigned int rows = route_compaction[{header}ll + expert];\n            const unsigned int expert_tiles = (rows + {TOKEN_TILE}u - 1u) / {TOKEN_TILE}u;\n            route_compaction[{header}ll + experts + expert] = running;\n            route_compaction[tile_start_base + expert] = tile_running;\n            for (unsigned int tile = 0u; tile < expert_tiles; ++tile) {{ route_compaction[tile_expert_base + tile_running + tile] = (unsigned int)expert; }}\n            tile_running += expert_tiles;\n            running += rows;\n        }}\n        route_compaction[0] = running;\n    }}\n    __syncthreads();\n"
    ));
    source.push_str(&format!(
        "    for (long long expert = warp; expert < experts; expert += warps) {{\n        unsigned int placed = 0u;\n        const unsigned int offset = route_compaction[{header}ll + experts + expert];\n        for (long long base = 0; base < tokens; base += 32) {{\n            const long long entry = base + lane;\n            const bool live = entry < tokens;\n{fetch}            fetched{weight} = live ? max((long long)0, min(fetched{weight}, u.gather_extent[{slot}] - 1)) : -1ll;\n            const bool hit = fetched{weight} == expert;\n            const unsigned int votes = __ballot_sync(PROXIMA_ALL_LANES, hit);\n            if (hit) {{ route_compaction[list_base + offset + placed + __popc(votes & ((1u << lane) - 1u))] = (unsigned int)entry; }}\n            placed += __popc(votes);\n        }}\n    }}\n}}\n"
    ));
    Ok(source)
}

fn push_fault(source: &mut String, weight: usize, slot: usize) {
    super::push_gather_fault_check(source, weight, slot, "                ");
}

/// Words of the compaction buffer the host binds after the kernel's own bindings.
pub(super) fn compaction_buffer_words(plan: &GroupedGemm) -> usize {
    compaction_words(COUNT_HEADER_WORDS, plan.tokens, plan.experts, TOKEN_TILE as u64, 0) as usize
}

pub(super) fn grouped_threads(resolved: &BoundOp, plan: &GroupedGemm) -> u64 {
    let BoundOpKind::Reduce { output_axes, .. } = &resolved.kind else {
        return 0;
    };
    let features: u64 = plan.gemm.feature_positions.iter().map(|&position| resolved.extents[output_axes[position] as usize]).product();
    features.div_ceil(FEATURE_TILE as u64) * tile_upper_bound(plan.tokens, plan.experts, TOKEN_TILE as u64) * GEMM_THREADS
}

pub(super) fn prepass_grid(resolved: &BoundOp) -> Result<CudaGridSpec, EmitError> {
    CudaGridSpec::launch(resolved.node, GEMM_THREADS, Some(GEMM_THREADS), resolved.kind.name())
}

pub(super) fn grouped_grid(resolved: &BoundOp, plan: &GroupedGemm) -> Result<CudaGridSpec, EmitError> {
    CudaGridSpec::launch(resolved.node, grouped_threads(resolved, plan), Some(GEMM_THREADS), resolved.kind.name())
}

pub(super) fn pair(source: String, entries: [String; 2], resolved: &BoundOp, plan: &GroupedGemm, bindings: Vec<Binding>) -> Result<[CudaKernel; 2], EmitError> {
    let [prepass, grouped] = entries;
    let route_indices = bindings.iter().filter(|binding| matches!(binding, Binding::Indices(_))).nth(plan.route.slot).cloned();
    let prepass_bindings: Vec<Binding> = route_indices.into_iter().chain([Binding::Uniforms, Binding::Fault]).collect();
    let kernel = |entry: String, grid: CudaGridSpec, bindings: Vec<Binding>| CudaKernel {
        source: source.clone(),
        entry,
        bindings,
        grid,
        extra_outputs: Vec::new(),
        prologue: Vec::new(),
        route_compaction_words: 0,
    };
    Ok([
        kernel(prepass, prepass_grid(resolved)?, prepass_bindings),
        kernel(grouped, grouped_grid(resolved, plan)?, bindings),
    ])
}
