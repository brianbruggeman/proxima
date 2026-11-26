use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_numeric::codec::Codec;
use proxima_tensor::{BoundOp, BoundOpKind, NodeId, ScorePlanes};

use super::{
    ATTENTION_BLOCK_WIDTH, Binding, CudaAttentionClass, CudaGridSpec, CudaKernel, EmitDevice,
    EmitError, cuda_cached_attention_class,
};
use crate::cuda_cached_attention::render_attention;
use crate::cuda_cached_attention_tiled::ATTENTION_TILE_MIN_ROWS;
use crate::lowering::dispatch_bindings;

/// Blocks per multiprocessor the split pass should launch so the device stays fed.
const SPLIT_WAVES: u64 = 4;

const MAX_SPLIT_CHUNKS: u64 = 32;

/// Keys one chunk holds at least, so a chunk is never split finer than a block's four warps can use.
const MIN_CHUNK_KEYS: u64 = 16;

const ODD_GUARD_TOKEN: &str = "@ODD_GUARD@";

const MIN_CHUNK_KEYS_TOKEN: &str = "@MIN_CHUNK_KEYS@";

/// Chunks of the key range a decode vector is split across, or `None` when the
/// kernel keeps one block per vector. Only a runtime-length single-row query, or a
/// whole-plane op below the row-tiling threshold, on a known device splits: the
/// grid is then too small to fill the multiprocessors. A whole plane has no cached
/// range, so its chunks are bounded by how many [`MIN_CHUNK_KEYS`] its keys fill.
pub(super) fn split_chunks(kind: &BoundOpKind, device: EmitDevice) -> Option<u64> {
    let BoundOpKind::CachedAttention {
        query_rows,
        new_key_rows,
        kv_heads,
        query_groups,
        planes,
        ..
    } = kind
    else {
        return None;
    };
    let ceiling = match cuda_cached_attention_class(kind) {
        Some(CudaAttentionClass::CachedBound) if *query_rows == 1 => MAX_SPLIT_CHUNKS,
        Some(CudaAttentionClass::Static)
            if *planes == ScorePlanes::Whole && *query_rows < ATTENTION_TILE_MIN_ROWS =>
        {
            new_key_rows.div_ceil(MIN_CHUNK_KEYS).min(MAX_SPLIT_CHUNKS)
        }
        _ => return None,
    };
    if device.multiprocessors == 0 {
        return None;
    }
    let vectors = query_rows * kv_heads * query_groups;
    let chunks = (u64::from(device.multiprocessors) * SPLIT_WAVES)
        .div_ceil(vectors.max(1))
        .min(ceiling);
    (chunks >= 2).then_some(chunks)
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
            query_rows,
            kv_heads,
            query_groups,
            head_dim,
            ..
        } => (query_rows * kv_heads * query_groups * chunks * partial_record_len(*head_dim)) as usize,
        _ => 0,
    }
}

pub(super) fn render_combine(
    resolved: &BoundOp,
    entry: &str,
    chunks: u64,
    quantized: &[Option<Codec>],
) -> Result<String, EmitError> {
    let tail = format!("    constexpr long long chunks = {chunks}LL;\n{COMBINE_BODY}");
    render_attention(
        resolved,
        entry,
        ",\n    const float* __restrict__ partials",
        &tail,
        "",
        quantized,
    )
}

pub(super) fn split_pass(
    resolved: &BoundOp,
    entry: &str,
    chunks: u64,
    quantized: &[Option<Codec>],
) -> Result<CudaKernel, EmitError> {
    let split_entry = format!("{entry}_split");
    let tail = format!("    constexpr long long chunks = {chunks}LL;\n{}", split_body(&resolved.kind));
    let source = render_attention(resolved, &split_entry, "", &tail, "", quantized)?;
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
        grid: CudaGridSpec::launch(
            resolved.node,
            vectors * chunks * ATTENTION_BLOCK_WIDTH,
            Some(ATTENTION_BLOCK_WIDTH),
            "cached_attention_split",
        )?,
        extra_outputs: Vec::new(),
        prologue: Vec::new(),
        route_compaction_words: 0,
        reduce_path: "",
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

/// The partial pass text for one op: a paired op keeps the template with its markers cleared; a
/// whole plane drops every line that reads the odd half plane or the pass triple.
fn split_body(kind: &BoundOpKind) -> String {
    let whole = matches!(kind, BoundOpKind::CachedAttention { planes: ScorePlanes::Whole, .. });
    let body: String = if whole {
        SPLIT_BODY
            .split_inclusive('\n')
            .filter(|line| !line.contains(ODD_GUARD_TOKEN))
            .collect()
    } else {
        SPLIT_BODY.replace(ODD_GUARD_TOKEN, "")
    };
    body.replace(MIN_CHUNK_KEYS_TOKEN, &MIN_CHUNK_KEYS.to_string())
}

const SPLIT_BODY: &str = "\
    constexpr long long min_chunk_keys = @MIN_CHUNK_KEYS@LL;
    const float negative_infinity = -__int_as_float(0x7f800000);
    constexpr int warps = 4;
    constexpr long long record = head_dim + 2;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5u;
    const long long vector_index = (long long)blockIdx.x / chunks;
    const long long chunk = (long long)blockIdx.x % chunks;
    if (vector_index >= u_ptr->total_elements) { return; }
    const long long query_row = vector_index / (kv_heads * query_groups);
    const long long query_head = vector_index % (kv_heads * query_groups);
    const long long kv_head = query_head / query_groups;
    const long long qbase = query_row * (kv_heads * query_groups * pair_dim) + query_head * pair_dim;
    const long long qpass = query_row * (kv_heads * query_groups * pass_dim) + query_head * pass_dim;
    const long long total_keys = cached_key_rows + new_key_rows;
    const long long per_chunk = max(min_chunk_keys, (total_keys + chunks - 1LL) / chunks);
    const long long first_key = chunk * per_chunk;
    const long long end_key = min(total_keys, first_key + per_chunk);
    float maximum = negative_infinity;
    float sum = 0.0f;
    float weighted[(head_dim + 31) / 32];
    for (long long slot = 0; slot < (head_dim + 31) / 32; slot++) { weighted[slot] = 0.0f; }
    for (long long key = first_key + (long long)warp; key < end_key; key += warps) {
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
            partial += in1[qbase + pair] * (cached ? cached_key_at(in3, kbase + pair) : in5[kbase + pair]);@ODD_GUARD@
        }
        for (long long column = (long long)lane; column < pass_dim; column += 32LL) {@ODD_GUARD@
            partial += pass_query[qpass + column] * (cached ? pass_cached_key[kpass + column] : pass_new_key[kpass + column]);@ODD_GUARD@
        }@ODD_GUARD@
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
    float* record_out = out + (vector_index * chunks + chunk) * record;
    if (threadIdx.x == 0u) {
        record_out[0] = global_maximum;
        record_out[1] = total;
    }
    for (long long dimension = (long long)threadIdx.x; dimension < head_dim; dimension += (long long)blockDim.x) {
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
    constexpr long long record = head_dim + 2;
    const long long vector_index = (long long)blockIdx.x;
    if (vector_index >= u_ptr->total_elements) { return; }
    const float* vector_partials = partials + vector_index * chunks * record;
    float global_maximum = negative_infinity;
    for (long long chunk = 0; chunk < chunks; chunk++) { global_maximum = fmaxf(global_maximum, vector_partials[chunk * record]); }
    float total = 0.0f;
    for (long long chunk = 0; chunk < chunks; chunk++) {
        const float chunk_maximum = vector_partials[chunk * record];
        total += (chunk_maximum == negative_infinity) ? 0.0f : vector_partials[chunk * record + 1] * expf(chunk_maximum - global_maximum);
    }
    for (long long dimension = (long long)threadIdx.x; dimension < head_dim; dimension += (long long)blockDim.x) {
        float merged = 0.0f;
        for (long long chunk = 0; chunk < chunks; chunk++) {
            const float chunk_maximum = vector_partials[chunk * record];
            merged += (chunk_maximum == negative_infinity) ? 0.0f : vector_partials[chunk * record + 2 + dimension] * expf(chunk_maximum - global_maximum);
        }
        out[vector_index * head_dim + dimension] = (total == 0.0f) ? 0.0f : merged / total;
    }
}
";
