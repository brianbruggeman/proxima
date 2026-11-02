use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_tensor::{BoundOp, BoundOpKind, NodeId};

use super::{
    ATTENTION_BLOCK_WIDTH, Binding, CudaAttentionClass, CudaGridSpec, CudaKernel, EmitDevice,
    EmitError, cuda_cached_attention_class,
};
use crate::cuda_cached_attention::render_attention;
use crate::lowering::dispatch_bindings;

/// Blocks per multiprocessor the split pass should launch so the device stays fed.
const SPLIT_WAVES: u64 = 4;

const MAX_SPLIT_CHUNKS: u64 = 32;

/// Chunks of the key range a decode vector is split across, or `None` when the
/// kernel keeps one block per vector. Only a single-row, runtime-length query on a
/// known device splits: the grid is then too small to fill the multiprocessors.
pub(super) fn split_chunks(kind: &BoundOpKind, device: EmitDevice) -> Option<u64> {
    let BoundOpKind::CachedAttention {
        query_rows,
        kv_heads,
        query_groups,
        ..
    } = kind
    else {
        return None;
    };
    let splittable = *query_rows == 1
        && device.multiprocessors > 0
        && cuda_cached_attention_class(kind) == Some(CudaAttentionClass::CachedBound);
    let vectors = kv_heads * query_groups;
    let chunks = (u64::from(device.multiprocessors) * SPLIT_WAVES)
        .div_ceil(vectors.max(1))
        .min(MAX_SPLIT_CHUNKS);
    (splittable && chunks >= 2).then_some(chunks)
}

pub(super) fn split_scratch_node(node: NodeId) -> NodeId {
    NodeId(u32::MAX - node.0)
}

fn partial_record_len(head_dim: u64) -> u64 {
    head_dim + 2
}

pub(super) fn split_scratch_len(kind: &BoundOpKind, chunks: u64) -> usize {
    match kind {
        BoundOpKind::CachedAttention {
            kv_heads,
            query_groups,
            head_dim,
            ..
        } => (kv_heads * query_groups * chunks * partial_record_len(*head_dim)) as usize,
        _ => 0,
    }
}

pub(super) fn render_combine(
    resolved: &BoundOp,
    entry: &str,
    chunks: u64,
) -> Result<String, EmitError> {
    let tail = format!("    constexpr long chunks = {chunks}L;\n{COMBINE_BODY}");
    render_attention(
        resolved,
        entry,
        ",\n    const float* __restrict__ partials",
        &tail,
        "",
    )
}

pub(super) fn split_pass(
    resolved: &BoundOp,
    entry: &str,
    chunks: u64,
) -> Result<CudaKernel, EmitError> {
    let split_entry = format!("{entry}_split");
    let tail = format!("    constexpr long chunks = {chunks}L;\n{SPLIT_BODY}");
    let source = render_attention(resolved, &split_entry, "", &tail, "")?;
    let scratch = split_scratch_node(resolved.node);
    let bindings: Vec<Binding> = dispatch_bindings(resolved)
        .into_iter()
        .map(|binding| match binding {
            Binding::Output(_) => Binding::Output(scratch),
            other => other,
        })
        .collect();
    let vectors = split_vector_count(&resolved.kind);
    Ok(CudaKernel {
        source,
        entry: split_entry,
        bindings,
        grid: CudaGridSpec {
            threads: vectors * chunks * ATTENTION_BLOCK_WIDTH,
            block_width: Some(ATTENTION_BLOCK_WIDTH),
        },
        extra_outputs: Vec::new(),
        prologue: Vec::new(),
    })
}

fn split_vector_count(kind: &BoundOpKind) -> u64 {
    match kind {
        BoundOpKind::CachedAttention {
            query_rows,
            kv_heads,
            query_groups,
            ..
        } => query_rows * kv_heads * query_groups,
        _ => 0,
    }
}

const SPLIT_BODY: &str = "\
    constexpr long min_chunk_keys = 16L;
    const float negative_infinity = -__int_as_float(0x7f800000);
    constexpr int warps = 4;
    constexpr long record = head_dim + 2;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5u;
    const long vector_index = (long)blockIdx.x / chunks;
    const long chunk = (long)blockIdx.x % chunks;
    if (vector_index >= u_ptr->total_elements) { return; }
    const long query_row = vector_index / (kv_heads * query_groups);
    const long query_head = vector_index % (kv_heads * query_groups);
    const long kv_head = query_head / query_groups;
    const long qbase = query_row * (kv_heads * query_groups * pair_dim) + query_head * pair_dim;
    const long qpass = query_row * (kv_heads * query_groups * pass_dim) + query_head * pass_dim;
    const long total_keys = cached_key_rows + new_key_rows;
    const long per_chunk = max(min_chunk_keys, (total_keys + chunks - 1L) / chunks);
    const long first_key = chunk * per_chunk;
    const long end_key = min(total_keys, first_key + per_chunk);
    float maximum = negative_infinity;
    float sum = 0.0f;
    float weighted[(head_dim + 31) / 32];
    for (long slot = 0; slot < (head_dim + 31) / 32; slot++) { weighted[slot] = 0.0f; }
    for (long key = first_key + (long)warp; key < end_key; key += warps) {
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
    float* record_out = out + (vector_index * chunks + chunk) * record;
    if (threadIdx.x == 0u) {
        record_out[0] = global_maximum;
        record_out[1] = total;
    }
    for (long dimension = (long)threadIdx.x; dimension < head_dim; dimension += (long)blockDim.x) {
        float merged = 0.0f;
        for (int source = 0; source < warps; source++) {
            merged += (shared_maximum[source] == negative_infinity) ? 0.0f : shared_weighted[source * head_dim + dimension] * expf(shared_maximum[source] - global_maximum);
        }
        record_out[2 + dimension] = merged;
    }
}
";

const COMBINE_BODY: &str = "\
    const float negative_infinity = -__int_as_float(0x7f800000);
    constexpr long record = head_dim + 2;
    const long vector_index = (long)blockIdx.x;
    if (vector_index >= u_ptr->total_elements) { return; }
    const float* vector_partials = partials + vector_index * chunks * record;
    float global_maximum = negative_infinity;
    for (long chunk = 0; chunk < chunks; chunk++) { global_maximum = fmaxf(global_maximum, vector_partials[chunk * record]); }
    float total = 0.0f;
    for (long chunk = 0; chunk < chunks; chunk++) {
        const float chunk_maximum = vector_partials[chunk * record];
        total += (chunk_maximum == negative_infinity) ? 0.0f : vector_partials[chunk * record + 1] * expf(chunk_maximum - global_maximum);
    }
    for (long dimension = (long)threadIdx.x; dimension < head_dim; dimension += (long)blockDim.x) {
        float merged = 0.0f;
        for (long chunk = 0; chunk < chunks; chunk++) {
            const float chunk_maximum = vector_partials[chunk * record];
            merged += (chunk_maximum == negative_infinity) ? 0.0f : vector_partials[chunk * record + 2 + dimension] * expf(chunk_maximum - global_maximum);
        }
        out[vector_index * head_dim + dimension] = (total == 0.0f) ? 0.0f : merged / total;
    }
}
";
