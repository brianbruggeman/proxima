//! The Metal execution driver: runs an [`omega::emit`](crate::emit)-produced
//! [`Kernel`] on a real GPU and proves it agrees with
//! [`proxima_tensor::cpu::evaluate`] — the piece `msl.rs`'s own doc says is
//! "the device driver's job, composed on top."
//!
//! # Prepare pipeline
//!
//! [`execute`] mirrors [`proxima_tensor::cpu::evaluate`]'s semantics
//! exactly, over the same public API that function itself is built from:
//! [`proxima_tensor::infer`] resolves shapes and symbols, [`proxima_tensor::lower`]
//! produces the flat [`Nest`] sequence (with `Fold(Zip)` fusion already
//! decided), and per-nest buffer retirement is recomputed from that sequence
//! the same way `cpu::evaluate`'s own `nest_retirement` does — a node's
//! device buffer is freed the moment nothing later in the sequence reads it.
//! What differs is only the last mile: instead of interpreting a `Nest` with
//! nested loops, each one is emitted to MSL, compiled (or reused from cache),
//! and dispatched.
//!
//! # Uniforms packing
//!
//! `msl.rs` never bakes a `Nest`'s concrete extents/strides/bases into
//! source text — they are read at kernel runtime out of a `constant
//! Uniforms&` buffer whose MSL struct layout is rendered field-by-field in
//! [`crate::msl::render_elementwise`], [`render_reduce`](crate::msl::render_reduce)
//! and [`render_scan`](crate::msl::render_scan). Every field in all three is
//! MSL `long` (an 8-byte, 8-byte-aligned integer) or an array of `long`, so
//! there is no interior struct padding to reason about: packing is a flat
//! concatenation of `i64`s in the exact field order those functions emit.
//! [`pack_elementwise_uniforms`], [`pack_reduce_uniforms`] and
//! [`pack_scan_uniforms`] each carry a comment pointing at the struct
//! declaration they mirror, byte for byte.
//!
//! # Execution model
//!
//! One `MTLCommandBuffer` per nest (correctness first, per the module's v1
//! stance — batching multiple nests into one command buffer is a later
//! optimization, not a correctness requirement). Every `MTLBuffer` is
//! `storageModeShared`: on Apple Silicon's unified memory, that makes
//! reading a result back a plain pointer read, no blit pass. Compiled
//! `MTLLibrary`/`MTLComputePipelineState` pairs are cached by kernel source
//! text within one [`execute`] call, since `msl.rs`'s own module doc proves
//! two structurally-identical `Nest`s emit byte-identical source.
//! `MTLCompileOptions::mathMode` is pinned to `Safe`, never the default —
//! parity against the CPU interpreter demands IEEE behavior, not whatever
//! Metal's fast-math would substitute.
//!
//! # Gather fault reporting
//!
//! `cpu::evaluate` returns `TensorError::GatherIndexOutOfRange` when a
//! fetched index falls outside its dim's extent; a GPU kernel cannot
//! propagate a `Result`, so `msl.rs` clamps for memory safety but also
//! `atomic_fetch_max`s the offending index into a per-gather-slot `Fault`
//! buffer (see that module's doc). [`dispatch_nest`] allocates and
//! zero-fills that buffer before every dispatch that gathers, and after
//! `waitUntilCompleted` reads it back and — via [`check_gather_fault`] —
//! turns any nonzero slot into the identical `TensorError` `cpu.rs` would
//! report for the same fetched index, wired through [`MetalError`]'s
//! `#[from]` so [`execute`] and `cpu::evaluate` produce `assert_eq!`-equal
//! errors.

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::ffi::c_void;
use core::mem::{size_of, size_of_val};
use core::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};

use proxima_tensor::{
    DType, Expr, GatherAccess, IndexMap, Keep, Nest, NodeId, Reduction, Shapes, TensorError, infer,
    lower,
};

use crate::error::EmitError;
use crate::msl::{gather_count, reduction_dims};
use crate::{Binding, Kernel, emit};

/// Everything [`execute`] can fail with: a missing device, any device
/// operation that returned a Metal-side failure (compiling source, creating
/// a pipeline, or one of the handful of `Option`-returning calls that are
/// only ever `None` on a broken host), or a `Nest`/program-shaped fault
/// [`proxima_tensor`] or [`crate::msl`] already have a name for.
#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    #[error("no Metal device available on this host")]
    NoDevice,
    #[error("metal driver error: {log}")]
    CompileFailed { log: String },
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Emit(#[from] EmitError),
}

/// The result of [`execute`]: every requested output's data and shape,
/// already read back from device memory. Same shape as
/// [`proxima_tensor::cpu::Evaluated`], on purpose — a parity test diffs the
/// two directly.
#[derive(Debug)]
pub struct Executed {
    root: NodeId,
    results: Vec<(NodeId, Vec<u64>, Vec<f32>)>,
}

impl Executed {
    #[must_use]
    pub fn root(&self) -> &[f32] {
        self.get(self.root).map_or(&[], |(data, _)| data)
    }

    #[must_use]
    pub fn shape(&self) -> &[u64] {
        self.get(self.root).map_or(&[], |(_, shape)| shape)
    }

    /// The data and shape of a specific requested output, or `None` if
    /// `node` was not in the `outputs` passed to [`execute`].
    #[must_use]
    pub fn get(&self, node: NodeId) -> Option<(&[f32], &[u64])> {
        self.results
            .iter()
            .find(|(candidate, _, _)| *candidate == node)
            .map(|(_, shape, data)| (data.as_slice(), shape.as_slice()))
    }
}

/// Runs a tensor program on the system's default Metal device.
///
/// Same contract as [`proxima_tensor::cpu::evaluate`]: `blocks` binds
/// [`Expr::Block`] inputs positionally, `outputs` selects which nodes to
/// return data for (the root only, if empty).
pub fn execute(
    program: &[Expr],
    symbols: &[u64],
    blocks: &[&[f32]],
    outputs: &[NodeId],
) -> Result<Executed, MetalError> {
    let prepared = prepare(program, symbols, blocks, outputs)?;

    let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
    let queue = device
        .newCommandQueue()
        .ok_or_else(|| MetalError::CompileFailed {
            log: "device refused to create a command queue".to_string(),
        })?;

    let mut device_buffers: BTreeMap<NodeId, Retained<ProtocolObject<dyn MTLBuffer>>> =
        BTreeMap::new();
    for (node, data) in prepared.block_nodes.iter().zip(blocks.iter()) {
        device_buffers.insert(*node, upload_block(&device, data)?);
    }

    let mut pipeline_cache: BTreeMap<
        String,
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    > = BTreeMap::new();
    for (position, nest) in prepared.nests.iter().enumerate() {
        dispatch_nest(
            &device,
            &queue,
            &mut pipeline_cache,
            &mut device_buffers,
            nest,
        )?;
        for retired in &prepared.retires[position] {
            device_buffers.remove(retired);
        }
    }

    finish(
        &prepared.shapes,
        &prepared.effective_outputs,
        &device_buffers,
        prepared.root,
    )
}

/// Everything [`execute`] needs before touching a device — the same
/// judgments [`proxima_tensor::cpu::evaluate`]'s own `prepare` makes, rebuilt
/// here over the public API since that one is private to `cpu.rs`.
struct Prepared {
    root: NodeId,
    shapes: Shapes,
    effective_outputs: Vec<NodeId>,
    block_nodes: Vec<NodeId>,
    nests: Vec<Nest>,
    retires: Vec<Vec<NodeId>>,
}

fn prepare(
    program: &[Expr],
    symbols: &[u64],
    blocks: &[&[f32]],
    outputs: &[NodeId],
) -> Result<Prepared, MetalError> {
    let shapes = infer(program, symbols)?;
    reject_non_float32(program)?;

    let root = program
        .len()
        .checked_sub(1)
        .map(|last| NodeId(last as u32))
        .ok_or(TensorError::Empty)?;
    for output in outputs {
        if output.0 as usize >= program.len() {
            return Err(TensorError::UnknownOutput(*output).into());
        }
    }
    let effective_outputs = if outputs.is_empty() {
        alloc::vec![root]
    } else {
        outputs.to_vec()
    };

    let block_nodes = block_node_ids(program);
    if blocks.len() != block_nodes.len() {
        return Err(TensorError::BlockCountMismatch {
            expected: block_nodes.len(),
            found: blocks.len(),
        }
        .into());
    }
    for (node, data) in block_nodes.iter().zip(blocks.iter()) {
        let expected = element_count(shapes.of(*node));
        if data.len() != expected {
            return Err(TensorError::BlockSizeMismatch {
                node: *node,
                expected,
                found: data.len(),
            }
            .into());
        }
    }

    let nests = lower(program, &shapes, &effective_outputs)?;
    let retires = nest_retirement(&nests, &effective_outputs);

    Ok(Prepared {
        root,
        shapes,
        effective_outputs,
        block_nodes,
        nests,
        retires,
    })
}

// mirrors `proxima_tensor::cpu::reject_non_float32`: every device buffer
// this driver uploads is f32 (see `upload_block`), indices included — an
// index value is an exact integer carried as f32 — so a gather's `indices`
// node is the one deliberate exception, exactly as on the CPU path.
fn reject_non_float32(program: &[Expr]) -> Result<(), TensorError> {
    let index_nodes = index_node_ids(program);
    for (position, expr) in program.iter().enumerate() {
        let node = NodeId(position as u32);
        if expr.dtype() != DType::Float32 && !index_nodes.contains(&node) {
            return Err(TensorError::NotLowerable {
                node,
                reason: "metal execution is f32-only in v1, except for a gather's indices",
            });
        }
    }
    Ok(())
}

/// Every node referenced as a gather's `indices` anywhere in `program` —
/// mirrors `proxima_tensor::cpu::index_node_ids`.
fn index_node_ids(program: &[Expr]) -> BTreeSet<NodeId> {
    let mut nodes = BTreeSet::new();
    for expr in program {
        match expr {
            Expr::Block { .. } => {}
            Expr::Zip { operands, .. } => {
                for (_, map) in operands {
                    push_indices_node(map, &mut nodes);
                }
            }
            Expr::Fold(fold) => {
                push_indices_node(&fold.in_map, &mut nodes);
                push_indices_node(&fold.out_map, &mut nodes);
            }
        }
    }
    nodes
}

fn push_indices_node(map: &IndexMap, nodes: &mut BTreeSet<NodeId>) {
    if let IndexMap::Computed { indices, .. } = map {
        nodes.insert(*indices);
    }
}

fn block_node_ids(program: &[Expr]) -> Vec<NodeId> {
    program
        .iter()
        .enumerate()
        .filter(|(_, expr)| matches!(expr, Expr::Block { .. }))
        .map(|(position, _)| NodeId(position as u32))
        .collect()
}

fn element_count(shape: &[u64]) -> usize {
    shape.iter().product::<u64>() as usize
}

/// Per-nest retire sets over the emitted nest sequence: `result[p]` is every
/// node whose last read is `nests[p]`. Mirrors `cpu::evaluate`'s own
/// (private) `nest_retirement` exactly, over the same public `Nest.operands`
/// field.
fn nest_retirement(nests: &[Nest], outputs: &[NodeId]) -> Vec<Vec<NodeId>> {
    let outputs: BTreeSet<NodeId> = outputs.iter().copied().collect();
    let mut last_use: BTreeMap<NodeId, usize> = BTreeMap::new();
    for (position, nest) in nests.iter().enumerate() {
        for (source, _, gather) in &nest.operands {
            last_use.insert(*source, position);
            if let Some(gather_access) = gather {
                last_use.insert(gather_access.indices, position);
            }
        }
    }

    let mut retires = alloc::vec![Vec::new(); nests.len()];
    for (node, position) in last_use {
        if !outputs.contains(&node) {
            retires[position].push(node);
        }
    }
    retires
}

/// The output length a nest needs allocated: the reduced (product of
/// surviving dims) length for a `Keep::Last` fold, or the full iteration
/// space otherwise (elementwise and `Keep::All` both write one value per
/// coordinate). Deliberately independent of [`Kernel::grid`]'s thread count —
/// a `Keep::All` scan dispatches one thread per *line* but writes
/// `inner_len` values per thread, so grid threads and output length diverge
/// there.
fn nest_output_len(nest: &Nest) -> usize {
    match &nest.reduction {
        Some(reduction) if reduction.keep == Keep::Last => reduction
            .output_dims
            .iter()
            .map(|dim| nest.extents[*dim as usize] as usize)
            .product(),
        _ => nest.extents.iter().map(|extent| *extent as usize).product(),
    }
}

fn push_i64(bytes: &mut Vec<u8>, value: i64) {
    bytes.extend_from_slice(&value.to_ne_bytes());
}

/// Pushes `values` as a fixed-`width` MSL array, zero-padding any slot
/// `values` does not fill — the only case that happens is a rank-0 nest,
/// where the declared array width is `max(rank, 1)` but there is no real
/// dim to supply, and that padding slot is never read by the generated
/// source (see each `render_*`'s `if rank > 0` / `.saturating_sub(1)` guards).
fn push_i64_row(bytes: &mut Vec<u8>, values: &[i64], width: usize) {
    for slot in 0..width {
        push_i64(bytes, values.get(slot).copied().unwrap_or(0));
    }
}

/// Appends the four gather arrays every `Uniforms` struct declares last (via
/// `crate::msl::push_gather_uniform_fields`) when `nest` has at least one
/// gathered operand: `gather_index_base`, `gather_index_strides`,
/// `gather_element_stride`, `gather_extent` — each one array of length
/// `gather_count`. `crate::msl::gather_slots` numbers gathered operands by
/// encounter order over `nest.operands`, so filtering in that same order
/// (below) reproduces the identical numbering without needing to re-derive
/// or look up the slot indices themselves. A no-op when `nest` has no
/// gather, matching `push_gather_uniform_fields`'s own empty-array early
/// return.
fn push_gather_uniforms(bytes: &mut Vec<u8>, nest: &Nest, rank_len: usize) {
    let ordered: Vec<&GatherAccess> = nest
        .operands
        .iter()
        .filter_map(|(_, _, gather)| gather.as_ref())
        .collect();
    if ordered.is_empty() {
        return;
    }

    for gather in &ordered {
        push_i64(bytes, gather.index_view.base);
    }
    for gather in &ordered {
        push_i64_row(bytes, &gather.index_view.strides, rank_len);
    }
    for gather in &ordered {
        push_i64(bytes, gather.element_stride);
    }
    for gather in &ordered {
        push_i64(bytes, gather.extent as i64);
    }
}

fn pack_uniforms(nest: &Nest) -> Vec<u8> {
    match &nest.reduction {
        None => pack_elementwise_uniforms(nest),
        Some(reduction) if reduction.keep == Keep::Last => pack_reduce_uniforms(nest, reduction),
        Some(reduction) => pack_scan_uniforms(nest, reduction),
    }
}

/// Mirrors the `Uniforms` struct `crate::msl::render_elementwise` declares
/// at `omega/src/msl.rs:328-335`: `total_elements`, `extents[rank_len]`,
/// `operand_base[operand_count]`, `operand_strides[operand_count][rank_len]`,
/// then — only when `nest` has a gathered operand — the four
/// `push_gather_uniform_fields` arrays [`push_gather_uniforms`] appends, in
/// that order — every field `long`, so a flat `i64` concatenation is the
/// struct's byte layout.
fn pack_elementwise_uniforms(nest: &Nest) -> Vec<u8> {
    let rank_len = nest.extents.len().max(1);
    let extents: Vec<i64> = nest.extents.iter().map(|extent| *extent as i64).collect();

    let mut bytes = Vec::new();
    push_i64(&mut bytes, extents.iter().product());
    push_i64_row(&mut bytes, &extents, rank_len);
    for (_, view, _) in &nest.operands {
        push_i64(&mut bytes, view.base);
    }
    for (_, view, _) in &nest.operands {
        push_i64_row(&mut bytes, &view.strides, rank_len);
    }
    push_gather_uniforms(&mut bytes, nest, rank_len);
    bytes
}

/// Mirrors the `Uniforms` struct `crate::msl::render_reduce` declares at
/// `omega/src/msl.rs:386-397`: `output_total`, `reduction_total`,
/// `output_extents[output_rank_len]`, `reduction_extents[reduce_rank_len]`,
/// `operand_base[operand_count]`,
/// `operand_strides[operand_count][rank_len]`, `out_base`,
/// `out_strides[rank_len]`, then the gather arrays (see
/// [`pack_elementwise_uniforms`]'s doc), in that order.
fn pack_reduce_uniforms(nest: &Nest, reduction: &Reduction) -> Vec<u8> {
    let rank_len = nest.extents.len().max(1);
    let output_dims = &reduction.output_dims;
    let output_rank_len = output_dims.len().max(1);
    let reduce_dims = reduction_dims(nest, output_dims);
    let reduce_rank_len = reduce_dims.len().max(1);

    let output_extents: Vec<i64> = output_dims
        .iter()
        .map(|dim| nest.extents[*dim as usize] as i64)
        .collect();
    let reduction_extents: Vec<i64> = reduce_dims
        .iter()
        .map(|dim| nest.extents[*dim as usize] as i64)
        .collect();

    let mut bytes = Vec::new();
    push_i64(&mut bytes, output_extents.iter().product());
    push_i64(&mut bytes, reduction_extents.iter().product());
    push_i64_row(&mut bytes, &output_extents, output_rank_len);
    push_i64_row(&mut bytes, &reduction_extents, reduce_rank_len);
    for (_, view, _) in &nest.operands {
        push_i64(&mut bytes, view.base);
    }
    for (_, view, _) in &nest.operands {
        push_i64_row(&mut bytes, &view.strides, rank_len);
    }
    push_i64(&mut bytes, reduction.out_view.base);
    push_i64_row(&mut bytes, &reduction.out_view.strides, rank_len);
    push_gather_uniforms(&mut bytes, nest, rank_len);
    bytes
}

/// Mirrors the `Uniforms` struct `crate::msl::render_scan` declares at
/// `omega/src/msl.rs:493-503`: `outer_total`, `inner_len`,
/// `outer_extents[outer_rank_len]`, `operand_base[operand_count]`,
/// `operand_strides[operand_count][rank_len]`, `out_base`,
/// `out_strides[rank_len]`, then the gather arrays (see
/// [`pack_elementwise_uniforms`]'s doc), in that order. `crate::msl::validate`
/// already rejected a rank-0 scan before `emit` (and therefore this) ever
/// runs, so `nest.extents` is never empty here.
fn pack_scan_uniforms(nest: &Nest, reduction: &Reduction) -> Vec<u8> {
    let rank = nest.extents.len();
    let rank_len = rank.max(1);
    let outer_rank = rank.saturating_sub(1);
    let outer_rank_len = outer_rank.max(1);

    let outer_extents: Vec<i64> = nest.extents[..outer_rank]
        .iter()
        .map(|extent| *extent as i64)
        .collect();
    let inner_len = nest.extents.last().copied().unwrap_or(1) as i64;

    let mut bytes = Vec::new();
    push_i64(&mut bytes, outer_extents.iter().product());
    push_i64(&mut bytes, inner_len);
    push_i64_row(&mut bytes, &outer_extents, outer_rank_len);
    for (_, view, _) in &nest.operands {
        push_i64(&mut bytes, view.base);
    }
    for (_, view, _) in &nest.operands {
        push_i64_row(&mut bytes, &view.strides, rank_len);
    }
    push_i64(&mut bytes, reduction.out_view.base);
    push_i64_row(&mut bytes, &reduction.out_view.strides, rank_len);
    push_gather_uniforms(&mut bytes, nest, rank_len);
    bytes
}

fn nserror_description(error: &NSError) -> String {
    error.localizedDescription().to_string()
}

fn compile_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    kernel: &Kernel,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
    let options = MTLCompileOptions::new();
    // parity demands IEEE-safe math, never the fast-math Metal defaults to.
    options.setMathMode(MTLMathMode::Safe);

    let source = NSString::from_str(&kernel.source);
    let library = device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .map_err(|error| MetalError::CompileFailed {
            log: nserror_description(&error),
        })?;

    let entry = NSString::from_str(&kernel.entry);
    let function =
        library
            .newFunctionWithName(&entry)
            .ok_or_else(|| MetalError::CompileFailed {
                log: format!(
                    "kernel entry `{}` missing from its own compiled library",
                    kernel.entry
                ),
            })?;

    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| MetalError::CompileFailed {
            log: nserror_description(&error),
        })
}

fn pipeline_for(
    device: &ProtocolObject<dyn MTLDevice>,
    pipeline_cache: &mut BTreeMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    kernel: &Kernel,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
    if let Some(pipeline) = pipeline_cache.get(&kernel.source) {
        return Ok(pipeline.clone());
    }
    let pipeline = compile_pipeline(device, kernel)?;
    pipeline_cache.insert(kernel.source.clone(), pipeline.clone());
    Ok(pipeline)
}

fn allocate_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    element_count: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let byte_length = element_count.max(1) * size_of::<f32>();
    device
        .newBufferWithLength_options(byte_length, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| MetalError::CompileFailed {
            log: "device refused to allocate a shared buffer".to_string(),
        })
}

fn upload_block(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[f32],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    if data.is_empty() {
        return allocate_buffer(device, 0);
    }
    let byte_length = size_of_val(data);
    // SAFETY: `data` is a live, non-empty `&[f32]` for the duration of this
    // call, so its first element's address is a valid, non-null pointer that
    // stays valid while `newBufferWithBytes_length_options` copies from it.
    let pointer = unsafe { NonNull::new_unchecked(data.as_ptr() as *mut c_void) };
    unsafe {
        device.newBufferWithBytes_length_options(
            pointer,
            byte_length,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| MetalError::CompileFailed {
        log: "device refused to allocate a shared buffer for a block input".to_string(),
    })
}

/// Allocates a `gather_count`-long `uint` buffer for a dispatch's gather
/// faults and zero-fills it — a freshly allocated `MTLBuffer`'s contents are
/// undefined, and a slot left as garbage would read as a spurious fault.
fn allocate_fault_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    gather_count: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let byte_length = gather_count.max(1) * size_of::<u32>();
    let buffer = device
        .newBufferWithLength_options(byte_length, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| MetalError::CompileFailed {
            log: "device refused to allocate the gather fault buffer".to_string(),
        })?;
    zero_fault_buffer(&buffer, gather_count);
    Ok(buffer)
}

fn zero_fault_buffer(buffer: &ProtocolObject<dyn MTLBuffer>, gather_count: usize) {
    let pointer = buffer.contents();
    // SAFETY: `buffer` is `storageModeShared` and was sized to at least
    // `gather_count` `u32`s by `allocate_fault_buffer`, so this is a valid,
    // CPU-visible, mutable slice for the duration of this call.
    let slots = unsafe {
        core::slice::from_raw_parts_mut(pointer.as_ptr().cast::<u32>(), gather_count.max(1))
    };
    slots.fill(0);
}

fn upload_uniforms(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    // SAFETY: `bytes` is always non-empty (every `Uniforms` struct has at
    // least two `long` fields), so its first byte's address is valid and
    // stays valid while this call copies from it.
    let pointer = unsafe { NonNull::new_unchecked(bytes.as_ptr() as *mut c_void) };
    unsafe {
        device.newBufferWithBytes_length_options(
            pointer,
            bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| MetalError::CompileFailed {
        log: "device refused to allocate the uniforms buffer".to_string(),
    })
}

fn buffer_for(
    device_buffers: &BTreeMap<NodeId, Retained<ProtocolObject<dyn MTLBuffer>>>,
    node: NodeId,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    device_buffers.get(&node).cloned().ok_or_else(|| {
        TensorError::NotLowerable {
            node,
            reason: "operand buffer missing at execution time",
        }
        .into()
    })
}

fn bind_buffers(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    kernel: &Kernel,
    device_buffers: &BTreeMap<NodeId, Retained<ProtocolObject<dyn MTLBuffer>>>,
    output: &Retained<ProtocolObject<dyn MTLBuffer>>,
    uniforms: &Retained<ProtocolObject<dyn MTLBuffer>>,
    fault: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
) -> Result<(), MetalError> {
    for (index, binding) in kernel.bindings.iter().enumerate() {
        let buffer = match binding {
            Binding::Input(node) | Binding::Indices(node) => buffer_for(device_buffers, *node)?,
            Binding::Output(_) => output.clone(),
            Binding::Uniforms => uniforms.clone(),
            Binding::Fault => fault.cloned().ok_or_else(|| MetalError::CompileFailed {
                log: "kernel binds a fault buffer but none was allocated".to_string(),
            })?,
        };
        // SAFETY: `buffer`'s length was sized from the same nest this
        // kernel was emitted from, so every byte the kernel indexes through
        // this binding is in bounds.
        unsafe { encoder.setBuffer_offset_atIndex(Some(&buffer), 0, index) };
    }
    Ok(())
}

fn dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    threads: u64,
) {
    if threads == 0 {
        return;
    }
    let max_threadgroup = pipeline.maxTotalThreadsPerThreadgroup();
    let threadgroup_width = (threads as usize).min(max_threadgroup).max(1);
    let grid = MTLSize {
        width: threads as usize,
        height: 1,
        depth: 1,
    };
    let threadgroup = MTLSize {
        width: threadgroup_width,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
}

fn dispatch_nest(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline_cache: &mut BTreeMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    device_buffers: &mut BTreeMap<NodeId, Retained<ProtocolObject<dyn MTLBuffer>>>,
    nest: &Nest,
) -> Result<(), MetalError> {
    let kernel = emit(nest)?;
    let pipeline = pipeline_for(device, pipeline_cache, &kernel)?;
    let output = allocate_buffer(device, nest_output_len(nest))?;
    let uniforms = upload_uniforms(device, &pack_uniforms(nest))?;
    let gathers = gather_count(nest);
    let fault = (gathers > 0)
        .then(|| allocate_fault_buffer(device, gathers))
        .transpose()?;

    let command_buffer = queue
        .commandBuffer()
        .ok_or_else(|| MetalError::CompileFailed {
            log: "command queue refused to hand out a command buffer".to_string(),
        })?;
    let encoder =
        command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::CompileFailed {
                log: "command buffer refused to hand out a compute encoder".to_string(),
            })?;

    encoder.setComputePipelineState(&pipeline);
    bind_buffers(
        &encoder,
        &kernel,
        device_buffers,
        &output,
        &uniforms,
        fault.as_ref(),
    )?;
    dispatch(&encoder, &pipeline, kernel.grid.threads);
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    if let Some(fault_buffer) = &fault {
        check_gather_fault(nest, fault_buffer, gathers)?;
    }

    device_buffers.insert(nest.node, output);
    Ok(())
}

/// Reads back a dispatch's fault buffer and, if any slot recorded a fault,
/// turns it into the same `TensorError::GatherIndexOutOfRange`
/// `cpu::evaluate` reports for the identical fetched index. Slot order
/// matches `nest.operands`' gather order — the same numbering
/// `crate::msl::gather_slots` and `push_gather_uniforms` both use — so the
/// first faulted slot's own `GatherAccess` supplies the extent to report.
fn check_gather_fault(
    nest: &Nest,
    fault_buffer: &ProtocolObject<dyn MTLBuffer>,
    gather_count: usize,
) -> Result<(), MetalError> {
    let slots = read_fault_slots(fault_buffer, gather_count);
    let gathers: Vec<&GatherAccess> = nest
        .operands
        .iter()
        .filter_map(|(_, _, gather)| gather.as_ref())
        .collect();
    for (slot, recorded) in slots.iter().enumerate() {
        if *recorded != 0 {
            return Err(TensorError::GatherIndexOutOfRange {
                node: nest.node,
                index: i64::from(*recorded - 1),
                extent: gathers[slot].extent,
            }
            .into());
        }
    }
    Ok(())
}

fn read_fault_slots(buffer: &ProtocolObject<dyn MTLBuffer>, gather_count: usize) -> Vec<u32> {
    let pointer = buffer.contents();
    // SAFETY: allocated and sized to at least `gather_count` `u32`s by
    // `allocate_fault_buffer`, `storageModeShared` so CPU-visible now that
    // `waitUntilCompleted` has returned.
    unsafe { core::slice::from_raw_parts(pointer.as_ptr().cast::<u32>(), gather_count.max(1)) }
        .to_vec()
}

fn read_back(buffer: &ProtocolObject<dyn MTLBuffer>, element_count: usize) -> Vec<f32> {
    if element_count == 0 {
        return Vec::new();
    }
    let pointer = buffer.contents();
    // SAFETY: `buffer` is `storageModeShared`, so `contents()` is a
    // CPU-visible pointer to at least `element_count` initialized `f32`s —
    // every output buffer this driver allocates is sized to at least that
    // many elements (see `allocate_buffer`'s caller, `dispatch_nest`) before
    // this point is reached.
    unsafe { core::slice::from_raw_parts(pointer.as_ptr().cast::<f32>(), element_count) }.to_vec()
}

fn finish(
    shapes: &Shapes,
    effective_outputs: &[NodeId],
    device_buffers: &BTreeMap<NodeId, Retained<ProtocolObject<dyn MTLBuffer>>>,
    root: NodeId,
) -> Result<Executed, MetalError> {
    let mut results = Vec::with_capacity(effective_outputs.len());
    for node in effective_outputs {
        let shape = shapes.of(*node).to_vec();
        let data = match device_buffers.get(node) {
            Some(buffer) => read_back(buffer, element_count(&shape)),
            None => Vec::new(),
        };
        results.push((*node, shape, data));
    }
    Ok(Executed { root, results })
}
