//! Portable GPU execution driver over `wgpu`/WGSL — the same emit-then-drive
//! split `crate::metal` implements for Metal, one abstraction layer over:
//! [`crate::wgsl::emit_wgsl`] is the "how" (kernel source, buffer-index ->
//! data mapping, thread count), this module is "run it on a real device."
//!
//! # Execution model
//!
//! One [`wgpu::Device`]/[`wgpu::Queue`] pair acquired ONCE per [`WgpuPlan`]
//! (see [`plan`]), never per call — the same "device/queue setup is not free"
//! stance `crate::metal::device_and_queue`'s own doc measures. One
//! [`wgpu::CommandEncoder`] per [`execute_plan`] call, one dispatch per
//! [`proxima_tensor::BoundOp`] in program order, one `queue.submit`, one
//! blocking wait on the mapped readback buffer — mirroring `crate::metal`'s
//! own "one command buffer, one `commit`, one `waitUntilCompleted`" posture
//! (`crate::metal`'s module doc, "Execution model").
//!
//! # The async edge, confined
//!
//! `wgpu`'s device/adapter request and buffer-map calls are genuinely async
//! (they cross a process boundary on some backends). [`pollster::block_on`]
//! is used in exactly two places: [`plan`]'s one-time adapter/device
//! acquisition, and [`execute_plan`]'s end-of-call readback map. Every other
//! function in this module is synchronous — planning and dispatch never
//! await anything, matching this crate's box-free/no-async-runtime-dependency
//! stance for the rest of the emit-then-drive split.
//!
//! # v1 scope
//!
//! Only [`proxima_tensor::QuantizedBlock::Float32`] blocks upload — every
//! packed/narrow codec is rejected with [`WgpuError::UnsupportedBlock`]
//! rather than dequantized on the host, matching [`crate::wgsl`]'s own v1
//! scope (no [`crate::msl::PackedCodec`] table exists on this path).

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::mem::size_of;
use std::sync::mpsc;

use proxima_tensor::{
    BoundOp, BoundOpKind, DType, Evaluated, Keep, Lookup, NodeId, Op, QuantizedBlock, Shapes,
    TensorError, bind_with_fusion, infer, prune_dead, resolve_named_blocks,
};

use crate::error::EmitError;
use crate::msl::{Binding, PackedCodec, PackedOperands, gather_count};
use crate::wgsl::{WgslCaps, WgslKernel, emit_wgsl};

/// Everything the wgpu driver can fail with.
#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    #[error("no wgpu adapter available on this host")]
    NoAdapter,
    #[error("wgpu device request failed: {0}")]
    NoDevice(String),
    #[error("wgpu driver error: {0}")]
    Driver(String),
    /// v1 uploads `Float32` blocks only — see the module doc.
    #[error(
        "node {node} is bound to a {codec} block, which the wgpu v1 driver does not upload (float32 only)"
    )]
    UnsupportedBlock { node: NodeId, codec: &'static str },
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Emit(#[from] EmitError),
    /// A fused kernel's `var<storage, ...>` binding count exceeds what this
    /// device actually supports (`wgpu::Limits::max_storage_buffers_per_shader_stage`,
    /// now requested at the adapter's real ceiling in `acquire_device` --
    /// see that function's doc). Caught here, before `pipeline_for` calls
    /// `create_compute_pipeline`, because `wgpu-core` validates that call
    /// with no error scope around it: an unchecked over-limit pipeline is an
    /// uncaught panic, not a `Result::Err` (the finding
    /// `training_step_parity.rs` named against the pre-fix driver).
    #[error(
        "node {node} needs {needed} storage buffer bindings but this device supports at most {limit} per shader stage"
    )]
    TooManyStorageBuffers {
        node: NodeId,
        needed: u32,
        limit: u32,
    },
}

/// A resolved, reusable program bound to one live `wgpu` device — the
/// counterpart of [`crate::metal::Plan`]. Owns its device/queue rather than
/// reaching for a thread-local cache: `wgpu::Device`/`wgpu::Queue` are
/// `Send + Sync` (unlike an `objc2` `Retained<_>`), so there is no
/// correctness reason to hide them behind thread-local state, and owning
/// them here is what lets [`plan`] be the one place `pollster::block_on`
/// pays for adapter/device acquisition (see the module doc).
pub struct WgpuPlan {
    device: wgpu::Device,
    queue: wgpu::Queue,
    program: Vec<Op>,
    shapes: Shapes,
    resolved: Vec<BoundOp>,
    effective_outputs: Vec<NodeId>,
    block_nodes: Vec<NodeId>,
    /// Compiled pipelines keyed by [`WgslKernel::entry`] (already a
    /// structural fingerprint, see that field's own doc) — populated lazily
    /// on first dispatch of each distinct kernel shape, reused across every
    /// later [`execute_plan`] call on this plan.
    pipelines: BTreeMap<String, wgpu::ComputePipeline>,
    /// What [`acquire_device`] found this adapter/device pair actually
    /// supports — threaded into every [`emit_wgsl`] call so a `Float16` node
    /// renders through `enable f16;` exactly when the device can run it, and
    /// fails with a named [`EmitError::UnsupportedDType`] otherwise (see
    /// `crate::wgsl`'s own "f16 compute" doc for why this is never a silent
    /// `f32` fallback).
    caps: WgslCaps,
    /// Which packed codec each block-bound node's bytes are — derived once,
    /// at plan time, from the concrete [`QuantizedBlock`] variant the caller
    /// planned against (see `packed_operands_of`), the same "codec is a
    /// property of the weight, decided once" stance `crate::metal::plan`
    /// takes for its own `Plan::packed_operands` field. Threaded into every
    /// [`emit_wgsl`] call so the pipeline cache (keyed by [`WgslKernel::entry`],
    /// which does not itself encode a codec choice) can never reuse a kernel
    /// compiled for one codec against a node now holding another.
    packed_operands: PackedOperands,
}

/// Which packed codec each of `block_nodes`' [`QuantizedBlock`] carries —
/// the WGSL driver's counterpart of `crate::metal::packed_operands_of`.
/// `Float32` maps to `None` (no codec, a plain `array<f32>` operand).
fn packed_operands_of(block_nodes: &[NodeId], blocks: &[QuantizedBlock<'_>]) -> PackedOperands {
    block_nodes
        .iter()
        .zip(blocks.iter())
        .filter_map(|(node, block)| match block {
            QuantizedBlock::Q4K(_) => Some((*node, PackedCodec::Q4K)),
            QuantizedBlock::Q5K(_) => Some((*node, PackedCodec::Q5K)),
            QuantizedBlock::Q6K(_) => Some((*node, PackedCodec::Q6K)),
            QuantizedBlock::Q8_0(_) => Some((*node, PackedCodec::Q8_0)),
            QuantizedBlock::Q4_0(_) => Some((*node, PackedCodec::Q4_0)),
            QuantizedBlock::Float16(_) => Some((*node, PackedCodec::Float16)),
            QuantizedBlock::BFloat16(_) => Some((*node, PackedCodec::BFloat16)),
            // `PackedCodec::Q3K` exists (Metal has a real unpack kernel for
            // it) but `crate::wgsl` does not -- `None` here routes a `Q3_K`
            // node through `execute_plan`'s existing
            // `WgpuError::UnsupportedBlock` path, the same "codec has no
            // wgpu entry" rejection `emit_wgsl`'s own
            // `EmitError::UnsupportedPackedCodec` raises for a caller who
            // reaches it directly.
            QuantizedBlock::Q3K(_) | QuantizedBlock::Float32(_) => None,
        })
        .collect()
}

fn block_node_ids(program: &[Op]) -> Vec<NodeId> {
    program
        .iter()
        .enumerate()
        .filter(|(_, expr)| matches!(expr, Op::Input { .. }))
        .map(|(position, _)| NodeId(position as u32))
        .collect()
}

fn element_count(shape: &[u64]) -> usize {
    shape.iter().product::<u64>() as usize
}

/// The raw packed bytes underneath any non-`Float32` [`QuantizedBlock`]
/// variant — every one of them wraps a `&[u8]` (see that type's own doc), so
/// this is a match, not a computation.
///
/// # Errors
/// [`EmitError::RenderKindMismatch`] if `block` is [`QuantizedBlock::Float32`]
/// — every caller here already branched on that case first, so this fires
/// only if that upstream guarantee itself broke.
fn packed_block_bytes_slice<'a>(
    node: NodeId,
    block: &QuantizedBlock<'a>,
) -> Result<&'a [u8], EmitError> {
    match block {
        QuantizedBlock::Q3K(bytes)
        | QuantizedBlock::Q4K(bytes)
        | QuantizedBlock::Q5K(bytes)
        | QuantizedBlock::Q6K(bytes)
        | QuantizedBlock::Q8_0(bytes)
        | QuantizedBlock::Q4_0(bytes)
        | QuantizedBlock::Float16(bytes)
        | QuantizedBlock::BFloat16(bytes) => Ok(bytes),
        QuantizedBlock::Float32(_) => Err(EmitError::RenderKindMismatch {
            node,
            expected: "a packed (non-float32) block",
            found: "float32",
        }),
    }
}

/// The exact packed byte length `elements` elements of `codec` occupy —
/// `crate::msl::PackedCodec::block_bytes`/`block_elements`'s own product,
/// rounded up to a whole block: a partial trailing block is never legal
/// GGUF, so `div_ceil` (not plain division) is what makes an off-by-one
/// undersized upload a hard [`TensorError::InputSizeMismatch`] instead of a
/// kernel silently reading past the buffer's end.
fn packed_expected_bytes(codec: PackedCodec, elements: usize) -> usize {
    elements.div_ceil(codec.block_elements()) * codec.block_bytes()
}

fn block_codec_name(block: &QuantizedBlock<'_>) -> &'static str {
    match block {
        QuantizedBlock::Float32(_) => "float32",
        QuantizedBlock::Q3K(_) => "q3_k",
        QuantizedBlock::Q4K(_) => "q4_k",
        QuantizedBlock::Q5K(_) => "q5_k",
        QuantizedBlock::Q6K(_) => "q6_k",
        QuantizedBlock::Q8_0(_) => "q8_0",
        QuantizedBlock::Q4_0(_) => "q4_0",
        QuantizedBlock::Float16(_) => "float16",
        QuantizedBlock::BFloat16(_) => "bfloat16",
    }
}

/// Acquires one adapter/device/queue triple, blocking only here (see the
/// module doc's "async edge" section). `request_adapter` prefers a
/// high-performance (discrete GPU) adapter, matching what a compute-bound
/// caller wants; on this box (arm64 macOS) that resolves to `wgpu`'s Metal
/// backend, same physical device `crate::metal` drives directly.
fn acquire_device() -> Result<(wgpu::Device, wgpu::Queue, WgslCaps), WgpuError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .map_err(|_| WgpuError::NoAdapter)?;
    // request every capability the adapter actually offers that `emit_wgsl`
    // knows how to use — requesting an unsupported feature is a hard error
    // at `request_device`, so this is gated on `adapter.features()` first,
    // never requested blind.
    let adapter_features = adapter.features();
    let requested_features =
        adapter_features & (wgpu::Features::SHADER_F16 | wgpu::Features::SUBGROUP);
    let adapter_info = adapter.get_info();
    // this driver owns its device exclusively (see the struct doc's "no
    // thread-local cache" stance) and never shares it with a swapchain, so
    // there is no competing consumer to protect from an over-large request --
    // `wgpu::Limits::default()` is the cross-vendor PORTABLE floor (caps
    // `max_storage_buffers_per_shader_stage` at 8), not this adapter's real
    // ceiling, and asking for less than the adapter offers is what turned a
    // 13-storage-buffer fused backward+Adam kernel into an uncaught
    // validation panic. Requesting `adapter.limits()` outright (wgpu 30's
    // documented way to ask for "everything this adapter supports") is
    // strictly `using_resolution`-equivalent-or-better here: that helper only
    // widens the three texture-dimension fields, never storage/buffer caps.
    let adapter_limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("omega-wgpu-plan"),
        required_features: requested_features,
        required_limits: adapter_limits.clone(),
        ..Default::default()
    }))
    .map_err(|error| WgpuError::NoDevice(error.to_string()))?;
    // a fixed, NONZERO subgroup width is required to bake `@workgroup_size`
    // to it (see `crate::wgsl::WgslCaps::subgroup_size`'s own doc) -- a
    // heterogeneous adapter (min != max) never takes the cooperative path,
    // it stays on the portable serial fold. `0` is also rejected even when
    // reported as "fixed" (min == max == 0): a paravirtualized adapter that
    // does not populate subgroup sizing at all would otherwise bake
    // `@workgroup_size(0)`, which every backend's shader validator rejects
    // outright -- that is a hard compile failure, not a named skip, so it is
    // caught here instead of being handed to the emitter.
    let subgroup_size = (device.features().contains(wgpu::Features::SUBGROUP)
        && adapter_info.subgroup_min_size == adapter_info.subgroup_max_size
        && adapter_info.subgroup_min_size > 0)
        .then_some(adapter_info.subgroup_min_size);
    let caps = WgslCaps {
        shader_f16: device.features().contains(wgpu::Features::SHADER_F16),
        subgroup_size,
    };
    Ok((device, queue, caps))
}

/// Resolves a program into a reusable [`WgpuPlan`], acquiring a device.
/// `blocks` decides each block-bound node's packed codec (see
/// `packed_operands_of`) — the same positional shape `crate::metal::plan`
/// takes, not just a shape-inference input.
///
/// # Errors
/// Propagates inference/binding failures and device acquisition failures.
pub fn plan(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock<'_>],
    outputs: &[NodeId],
) -> Result<WgpuPlan, WgpuError> {
    let shapes = infer(program, symbols)?;

    // ROW 327 (mirrors `crate::metal::prepare`'s own fix): `block_nodes[i]`
    // is the ONLY node `blocks[i]` may be attributed to -- this crate's
    // positional contract, identical to `crate::metal::execute`'s own doc.
    // Validating that pairing here, BEFORE `packed_operands_of` classifies
    // each block by codec, is what stops a caller's node/block order
    // mismatch from surfacing as an unrelated downstream rejection
    // (`WgpuError::UnsupportedBlock` naming the wrong node) instead of the
    // precise, node-carrying `InputCountMismatch`/`InputSizeMismatch` below.
    let block_nodes = block_node_ids(program);
    if blocks.len() != block_nodes.len() {
        return Err(TensorError::InputCountMismatch {
            expected: block_nodes.len(),
            found: blocks.len(),
        }
        .into());
    }
    for (node, block) in block_nodes.iter().zip(blocks.iter()) {
        let expected = element_count(shapes.of(*node));
        let found = block.element_count()?;
        if found != expected {
            return Err(TensorError::InputSizeMismatch {
                node: *node,
                expected,
                found,
            }
            .into());
        }
    }

    let packed_operands = packed_operands_of(&block_nodes, blocks);
    let root = program
        .len()
        .checked_sub(1)
        .map(|last| NodeId(last as u32))
        .ok_or(TensorError::Empty)?;
    let effective_outputs = if outputs.is_empty() {
        alloc::vec![root]
    } else {
        outputs.to_vec()
    };
    // no persistent arena to skip a dead slot inside between calls -- see
    // `proxima_tensor::prune_dead`'s own doc. `fuse_cached_attention: false`
    // because `crate::wgsl::emit_wgsl` has no renderer for
    // `BoundOpKind::CachedAttention` yet -- the fused rewrite is a
    // Metal/CPU-only optimization until wgpu grows one.
    let resolved = prune_dead(
        bind_with_fusion(program, &shapes, &effective_outputs, false)?,
        &effective_outputs,
    );
    let (device, queue, caps) = acquire_device()?;
    Ok(WgpuPlan {
        device,
        queue,
        program: program.to_vec(),
        shapes,
        resolved,
        effective_outputs,
        block_nodes,
        pipelines: BTreeMap::new(),
        caps,
        packed_operands,
    })
}

/// [`plan`] against a name-keyed block set.
///
/// # Errors
/// Propagates name resolution and planning failures.
pub fn plan_named(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
) -> Result<WgpuPlan, WgpuError> {
    let blocks = resolve_named_blocks(program, named)?;
    plan(program, symbols, &blocks, outputs)
}

impl WgpuPlan {
    /// The capabilities [`plan`] found this plan's acquired device actually
    /// supports — a diagnostic accessor for a caller (or a parity test) that
    /// wants to know, before or after a run, whether a `Keep::Reduce` fold
    /// took `crate::wgsl::reduce_is_cooperative`'s subgroup path or the
    /// portable serial fold.
    #[must_use]
    pub fn caps(&self) -> WgslCaps {
        self.caps
    }

    /// The acquired device's actual resource limits — a diagnostic accessor
    /// for a caller (or a parity test) that wants to know, BEFORE
    /// dispatching, whether a program's largest buffer fits this adapter's
    /// `max_buffer_size`/`max_storage_buffer_binding_size` rather than
    /// discovering a paravirtualized/constrained adapter's ceiling only as a
    /// device-lost failure mid-run.
    #[must_use]
    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }
}

fn shader_module(device: &wgpu::Device, kernel: &WgslKernel) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(kernel.entry.as_str()),
        source: wgpu::ShaderSource::Wgsl(kernel.source.as_str().into()),
    })
}

fn pipeline_for<'plan>(
    device: &wgpu::Device,
    pipelines: &'plan mut BTreeMap<String, wgpu::ComputePipeline>,
    kernel: &WgslKernel,
) -> &'plan wgpu::ComputePipeline {
    pipelines.entry(kernel.entry.clone()).or_insert_with(|| {
        let module = shader_module(device, kernel);
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(kernel.entry.as_str()),
            layout: None,
            module: &module,
            entry_point: Some(kernel.entry.as_str()),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    })
}

fn push_i32(bytes: &mut Vec<u8>, value: i32) {
    bytes.extend_from_slice(&value.to_ne_bytes());
}

fn push_i32_row(bytes: &mut Vec<u8>, values: &[i64], width: usize) {
    for slot in 0..width {
        push_i32(bytes, values.get(slot).copied().unwrap_or(0) as i32);
    }
}

fn reduction_dims(bound: &BoundOp, output_axes: &[u16]) -> Vec<u16> {
    (0..bound.extents.len() as u16)
        .filter(|dim| !output_axes.contains(dim))
        .collect()
}

/// Appends the four gather arrays [`crate::wgsl`]'s `push_gather_uniform_fields`
/// declares last, in the same operand order [`crate::msl::gather_slots`]
/// numbers — mirrors `crate::metal::push_gather_uniforms`, narrowed to `i32`.
/// A no-op when `bound` gathers nothing, matching that field-emission's own
/// empty-array early exit.
fn push_gather_uniforms(bytes: &mut Vec<u8>, bound: &BoundOp, rank_len: usize) {
    let ordered: Vec<&Lookup> = bound
        .operands()
        .iter()
        .filter_map(|(_, _, gather)| gather.as_ref())
        .collect();
    if ordered.is_empty() {
        return;
    }
    for gather in &ordered {
        push_i32(bytes, gather.index_layout.base as i32);
    }
    for gather in &ordered {
        push_i32_row(bytes, &gather.index_layout.strides, rank_len);
    }
    for gather in &ordered {
        push_i32(bytes, gather.element_stride as i32);
    }
    for gather in &ordered {
        push_i32(bytes, gather.extent as i32);
    }
}

/// Mirrors `crate::metal::pack_elementwise_uniforms`, narrowed to `i32` (see
/// `crate::wgsl`'s own doc on why WGSL fields are `i32` rather than `long`)
/// and with no gather fields (v1 has no gather).
fn pack_elementwise_uniforms(bound: &BoundOp) -> Vec<u8> {
    let rank_len = bound.extents.len().max(1);
    let extents: Vec<i64> = bound.extents.iter().map(|extent| *extent as i64).collect();
    let mut bytes = Vec::new();
    push_i32(&mut bytes, extents.iter().product::<i64>() as i32);
    push_i32_row(&mut bytes, &extents, rank_len);
    for (_, layout, _) in bound.operands() {
        push_i32(&mut bytes, layout.base as i32);
    }
    for (_, layout, _) in bound.operands() {
        push_i32_row(&mut bytes, &layout.strides, rank_len);
    }
    push_gather_uniforms(&mut bytes, bound, rank_len);
    bytes
}

fn pack_reduce_uniforms(bound: &BoundOp) -> Result<Vec<u8>, EmitError> {
    let BoundOpKind::Reduce {
        output_axes,
        out_layout,
        epilogue_operands,
        ..
    } = &bound.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: bound.node,
            expected: "keep::reduce fold",
            found: bound.kind.name(),
        });
    };
    let rank_len = bound.extents.len().max(1);
    let output_rank_len = output_axes.len().max(1);
    let reduce_axes = reduction_dims(bound, output_axes);
    let reduce_rank_len = reduce_axes.len().max(1);

    let output_extents: Vec<i64> = output_axes
        .iter()
        .map(|axis| bound.extents[*axis as usize] as i64)
        .collect();
    let reduction_extents: Vec<i64> = reduce_axes
        .iter()
        .map(|axis| bound.extents[*axis as usize] as i64)
        .collect();

    let mut bytes = Vec::new();
    push_i32(&mut bytes, output_extents.iter().product::<i64>() as i32);
    push_i32(&mut bytes, reduction_extents.iter().product::<i64>() as i32);
    push_i32_row(&mut bytes, &output_extents, output_rank_len);
    push_i32_row(&mut bytes, &reduction_extents, reduce_rank_len);
    for (_, layout, _) in bound.operands() {
        push_i32(&mut bytes, layout.base as i32);
    }
    for (_, layout, _) in bound.operands() {
        push_i32_row(&mut bytes, &layout.strides, rank_len);
    }
    push_i32(&mut bytes, out_layout.base as i32);
    push_i32_row(&mut bytes, &out_layout.strides, rank_len);
    // `crate::wgsl::render_reduce`/`render_reduce_cooperative`'s own
    // `Uniforms` struct declares these fields ONLY when `epilogue_operands`
    // is non-empty (byte-identical to before epilogue fusion existed
    // otherwise), so this must stay conditional on the exact same test --
    // mirrors `crate::metal::pack_reduce_uniforms`'s own epilogue block.
    if !epilogue_operands.is_empty() {
        for (_, layout, _) in epilogue_operands {
            push_i32(&mut bytes, layout.base as i32);
        }
        for (_, layout, _) in epilogue_operands {
            push_i32_row(&mut bytes, &layout.strides, output_rank_len);
        }
    }
    push_gather_uniforms(&mut bytes, bound, rank_len);
    Ok(bytes)
}

fn pack_scan_uniforms(bound: &BoundOp) -> Result<Vec<u8>, EmitError> {
    let BoundOpKind::Reduce { out_layout, .. } = &bound.kind else {
        return Err(EmitError::RenderKindMismatch {
            node: bound.node,
            expected: "keep::scan fold",
            found: bound.kind.name(),
        });
    };
    let rank = bound.extents.len();
    let rank_len = rank.max(1);
    let outer_rank = rank.saturating_sub(1);
    let outer_rank_len = outer_rank.max(1);

    let outer_extents: Vec<i64> = bound.extents[..outer_rank]
        .iter()
        .map(|extent| *extent as i64)
        .collect();
    let inner_len = bound.extents.last().copied().unwrap_or(1) as i64;

    let mut bytes = Vec::new();
    push_i32(&mut bytes, outer_extents.iter().product::<i64>() as i32);
    push_i32(&mut bytes, inner_len as i32);
    push_i32_row(&mut bytes, &outer_extents, outer_rank_len);
    for (_, layout, _) in bound.operands() {
        push_i32(&mut bytes, layout.base as i32);
    }
    for (_, layout, _) in bound.operands() {
        push_i32_row(&mut bytes, &layout.strides, rank_len);
    }
    push_i32(&mut bytes, out_layout.base as i32);
    push_i32_row(&mut bytes, &out_layout.strides, rank_len);
    Ok(bytes)
}

/// Mirrors `crate::metal::pack_leaf_uniforms`, narrowed to `i32` the same
/// way [`pack_elementwise_uniforms`] is: `render_iota`/`render_constant`
/// both declare `struct Uniforms { total_elements: i32 }` and gate every
/// thread on `gid >= u.total_elements` — an empty uniform buffer reads as
/// zero (wgpu zero-fills unwritten buffer contents), so every thread would
/// see `total_elements == 0` and return before writing anything.
fn pack_leaf_uniforms(bound: &BoundOp) -> Vec<u8> {
    let total: i64 = bound.extents.iter().map(|extent| *extent as i64).product();
    let mut bytes = Vec::new();
    push_i32(&mut bytes, total as i32);
    bytes
}

fn pack_uniforms(bound: &BoundOp) -> Result<Vec<u8>, EmitError> {
    match &bound.kind {
        BoundOpKind::Elementwise { .. } => Ok(pack_elementwise_uniforms(bound)),
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => pack_reduce_uniforms(bound),
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => pack_scan_uniforms(bound),
        // `CachedAttention` never reaches this function in practice --
        // `crate::wgsl::emit_wgsl` (called before a `BoundOp` is ever
        // dispatched through this driver) already returns
        // `EmitError::UnsupportedOpKind` for it. Grouped with
        // `Iota`/`Constant` only to satisfy exhaustiveness with a harmless
        // value, never a real uniform layout.
        BoundOpKind::Iota | BoundOpKind::Constant { .. } | BoundOpKind::CachedAttention { .. } => {
            Ok(pack_leaf_uniforms(bound))
        }
    }
}

/// The output length an op needs allocated — mirrors
/// `crate::metal::bound_output_len`.
fn bound_output_len(bound: &BoundOp) -> usize {
    match &bound.kind {
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            output_axes,
            ..
        } => output_axes
            .iter()
            .map(|axis| bound.extents[*axis as usize] as usize)
            .product(),
        _ => bound
            .extents
            .iter()
            .map(|extent| *extent as usize)
            .product(),
    }
}

/// WebGPU requires every buffer's size to be a multiple of 4 bytes — most
/// callers already satisfy this for free (`size_of::<f32>() * n` is always a
/// multiple of 4), but a packed codec's own block width need not be (`Q6_K`'s
/// 210-byte super-block is not), so this rounds up rather than trusting the
/// caller.
fn storage_buffer(
    device: &wgpu::Device,
    label: &str,
    len_bytes: usize,
    extra: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: len_bytes.max(4).div_ceil(4) as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE | extra,
        mapped_at_creation: false,
    })
}

/// Maps `buffer` for CPU read and copies its bytes out, blocking on exactly
/// one `poll`/`recv` pair — the shared tail [`execute_plan`]'s output and
/// fault readbacks both need, factored out so the two cannot drift on the
/// map/poll/unmap sequence.
fn map_read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<u8>, WgpuError> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| WgpuError::Driver(error.to_string()))?;
    receiver
        .recv()
        .map_err(|error| WgpuError::Driver(error.to_string()))?
        .map_err(|error| WgpuError::Driver(error.to_string()))?;
    let view = slice
        .get_mapped_range()
        .map_err(|error| WgpuError::Driver(error.to_string()))?;
    let bytes = view.to_vec();
    drop(view);
    buffer.unmap();
    Ok(bytes)
}

fn gpu_dtype(program: &[Op], node: NodeId) -> DType {
    program[node.0 as usize].dtype()
}

/// Runs an already-resolved [`WgpuPlan`] against fresh block data — the
/// serving-loop entry point. `&mut self` because dispatch may populate
/// `pipelines` on a cache miss.
///
/// # Errors
/// Propagates block-shape mismatches, unsupported (non-`Float32`) blocks,
/// and every WGSL emit/dispatch failure.
pub fn execute_plan(
    plan: &mut WgpuPlan,
    blocks: &[QuantizedBlock<'_>],
) -> Result<Evaluated, WgpuError> {
    if blocks.len() != plan.block_nodes.len() {
        return Err(TensorError::InputCountMismatch {
            expected: plan.block_nodes.len(),
            found: blocks.len(),
        }
        .into());
    }
    for (node, block) in plan.block_nodes.iter().zip(blocks.iter()) {
        let elements = element_count(plan.shapes.of(*node));
        let (found, expected) = match block {
            QuantizedBlock::Float32(data) => (data.len(), elements),
            _ => {
                let Some(codec) = plan.packed_operands.get(node).copied() else {
                    return Err(WgpuError::UnsupportedBlock {
                        node: *node,
                        codec: block_codec_name(block),
                    });
                };
                (
                    packed_block_bytes_slice(*node, block)?.len(),
                    packed_expected_bytes(codec, elements),
                )
            }
        };
        if found != expected {
            return Err(TensorError::InputSizeMismatch {
                node: *node,
                expected,
                found,
            }
            .into());
        }
    }

    let mut device_buffers: BTreeMap<NodeId, wgpu::Buffer> = BTreeMap::new();
    for (node, block) in plan.block_nodes.iter().zip(blocks.iter()) {
        let buffer = match block {
            QuantizedBlock::Float32(data) => {
                let buffer = storage_buffer(
                    &plan.device,
                    "omega-wgpu-input",
                    size_of::<f32>() * data.len().max(1),
                    wgpu::BufferUsages::COPY_DST,
                );
                plan.queue
                    .write_buffer(&buffer, 0, bytemuck::cast_slice(data));
                buffer
            }
            _ => {
                let bytes = packed_block_bytes_slice(*node, block)?;
                let buffer = storage_buffer(
                    &plan.device,
                    "omega-wgpu-packed-input",
                    bytes.len(),
                    wgpu::BufferUsages::COPY_DST,
                );
                plan.queue.write_buffer(&buffer, 0, bytes);
                buffer
            }
        };
        device_buffers.insert(*node, buffer);
    }

    for bound in &plan.resolved {
        let output_len = bound_output_len(bound);
        let buffer = storage_buffer(
            &plan.device,
            "omega-wgpu-output",
            size_of::<f32>() * output_len.max(1),
            wgpu::BufferUsages::COPY_SRC,
        );
        device_buffers.insert(bound.node, buffer);
    }

    let mut encoder = plan
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("omega-wgpu-encoder"),
        });

    // uniform buffers are created per dispatch (not cached) — cheap, small,
    // and the plan's device buffer table already owns every operand/output
    // buffer these dispatches bind; a `Vec` here just keeps every uniforms
    // buffer alive until `submit`, matching `wgpu`'s "buffer must outlive
    // the encoded pass that references it" contract.
    let mut uniform_buffers: Vec<wgpu::Buffer> = Vec::with_capacity(plan.resolved.len());
    // one fault buffer per dispatch that gathers -- (node, buffer, per-slot
    // extent, ordered the same way `push_gather_uniforms` numbers slots) so
    // a post-submit fault reports the right `Lookup`'s extent. Mirrors
    // `crate::metal::encode_op`'s own `pending_faults` accumulator.
    let mut pending_faults: Vec<(NodeId, wgpu::Buffer, Vec<u64>)> = Vec::new();
    let storage_buffer_limit = plan.device.limits().max_storage_buffers_per_shader_stage;
    for bound in &plan.resolved {
        let kernel = emit_wgsl(bound, plan.caps, &plan.packed_operands)?;
        // pre-validate against the device's real limit BEFORE
        // `pipeline_for` reaches `create_compute_pipeline` -- the binding
        // count is fully known here (every `Binding` is a `var<storage,
        // ...>`, see `crate::wgsl`'s own binding-emission doc), so this is
        // the sans-IO-shaped alternative to wrapping the pipeline call in an
        // async `push_error_scope`/`pop_error_scope` pair (see
        // [`WgpuError::TooManyStorageBuffers`]'s own doc).
        let needed_bindings = kernel.bindings.len() as u32;
        if needed_bindings > storage_buffer_limit {
            return Err(WgpuError::TooManyStorageBuffers {
                node: bound.node,
                needed: needed_bindings,
                limit: storage_buffer_limit,
            });
        }
        let uniform_bytes = pack_uniforms(bound)?;
        let uniform_buffer = plan.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("omega-wgpu-uniforms"),
            size: uniform_bytes.len().max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        plan.queue.write_buffer(&uniform_buffer, 0, &uniform_bytes);

        let gathers = gather_count(bound);
        let fault_buffer = (gathers > 0).then(|| {
            let buffer = storage_buffer(
                &plan.device,
                "omega-wgpu-fault",
                size_of::<u32>() * gathers,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            );
            plan.queue
                .write_buffer(&buffer, 0, &alloc::vec![0u8; size_of::<u32>() * gathers]);
            buffer
        });

        let pipeline = pipeline_for(&plan.device, &mut plan.pipelines, &kernel);
        let layout = pipeline.get_bind_group_layout(0);
        let mut entries: Vec<wgpu::BindGroupEntry<'_>> = Vec::with_capacity(kernel.bindings.len());
        for (index, binding) in kernel.bindings.iter().enumerate() {
            let resource = match binding {
                Binding::Input(node) | Binding::Output(node) | Binding::Indices(node) => {
                    device_buffers
                        .get(node)
                        .ok_or_else(|| {
                            WgpuError::Driver(alloc::format!("no device buffer for node {node}"))
                        })?
                        .as_entire_binding()
                }
                Binding::Uniforms => uniform_buffer.as_entire_binding(),
                Binding::Fault => fault_buffer
                    .as_ref()
                    .ok_or_else(|| {
                        WgpuError::Driver(
                            "gather kernel requested but no fault buffer allocated".into(),
                        )
                    })?
                    .as_entire_binding(),
            };
            entries.push(wgpu::BindGroupEntry {
                binding: index as u32,
                resource,
            });
        }
        let bind_group = plan.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(kernel.entry.as_str()),
            layout: &layout,
            entries: &entries,
        });

        let workgroups = kernel
            .threads
            .div_ceil(u64::from(kernel.workgroup_size))
            .max(1) as u32;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(kernel.entry.as_str()),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);
        uniform_buffers.push(uniform_buffer);
        if let Some(buffer) = fault_buffer {
            let extents: Vec<u64> = bound
                .operands()
                .iter()
                .filter_map(|(_, _, gather)| gather.as_ref().map(|lookup| lookup.extent))
                .collect();
            pending_faults.push((bound.node, buffer, extents));
        }
    }

    // readback staging buffers for every requested output, mapped after the
    // one submit/wait below — mirrors `crate::metal`'s own single
    // end-of-program wait (see the module doc).
    let mut staging: Vec<(NodeId, wgpu::Buffer)> = Vec::with_capacity(plan.effective_outputs.len());
    for node in &plan.effective_outputs {
        let Some(source) = device_buffers.get(node) else {
            continue;
        };
        let byte_len = source.size();
        let staged = plan.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("omega-wgpu-readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(source, 0, &staged, 0, byte_len);
        staging.push((*node, staged));
    }

    let mut fault_staging: Vec<(NodeId, wgpu::Buffer, Vec<u64>)> =
        Vec::with_capacity(pending_faults.len());
    for (node, source, extents) in &pending_faults {
        let byte_len = source.size();
        let staged = plan.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("omega-wgpu-fault-readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(source, 0, &staged, 0, byte_len);
        fault_staging.push((*node, staged, extents.clone()));
    }

    plan.queue.submit(core::iter::once(encoder.finish()));

    let mut results = Vec::with_capacity(staging.len());
    for (node, buffer) in &staging {
        let bytes = map_read(&plan.device, buffer)?;
        let data: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
        let shape = plan.shapes.of(*node).to_vec();
        results.push((*node, shape, data));
    }
    let _ = gpu_dtype; // reserved for a future non-f32 readback path

    // one fault check per gathering dispatch, in program order -- the first
    // recorded fault anywhere wins, matching `crate::metal::check_gather_fault`'s
    // own "return on first faulted slot" posture.
    for (node, buffer, extents) in &fault_staging {
        let bytes = map_read(&plan.device, buffer)?;
        let slots: &[u32] = bytemuck::cast_slice(&bytes);
        for (slot, recorded) in slots.iter().enumerate() {
            if *recorded != 0 {
                return Err(TensorError::GatherIndexOutOfRange {
                    node: *node,
                    index: i64::from(*recorded - 1),
                    extent: extents[slot],
                }
                .into());
            }
        }
    }

    let root = plan
        .program
        .len()
        .checked_sub(1)
        .map(|last| NodeId(last as u32))
        .unwrap_or(NodeId(0));
    Ok(Evaluated::from_parts(root, results, None))
}

/// [`execute_plan`] against a name-keyed block set.
///
/// # Errors
/// Propagates name resolution and execution failures.
pub fn execute_plan_named(
    plan: &mut WgpuPlan,
    named: &[(&str, QuantizedBlock<'_>)],
) -> Result<Evaluated, WgpuError> {
    let blocks = resolve_named_blocks(&plan.program, named)?;
    execute_plan(plan, &blocks)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    use proxima_tensor::{DType, Extent, IndexMap, ScalarOp, append, bind, map};

    use super::*;

    fn elementwise_tanh_op(extent: u32) -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("infer succeeds");
        let bound = bind(&program, &shapes, &[]).expect("bind succeeds");
        bound.into_iter().next().expect("one bound op")
    }

    #[test]
    fn pack_reduce_uniforms_rejects_an_elementwise_bound_op() {
        let bound = elementwise_tanh_op(8);
        let error = pack_reduce_uniforms(&bound)
            .expect_err("an elementwise chain is not a Reduce fold");
        assert!(matches!(
            error,
            EmitError::RenderKindMismatch {
                expected: "keep::reduce fold",
                found: "elementwise",
                ..
            }
        ));
    }

    #[test]
    fn pack_scan_uniforms_rejects_an_elementwise_bound_op() {
        let bound = elementwise_tanh_op(8);
        let error =
            pack_scan_uniforms(&bound).expect_err("an elementwise chain is not a Reduce fold");
        assert!(matches!(
            error,
            EmitError::RenderKindMismatch {
                expected: "keep::scan fold",
                found: "elementwise",
                ..
            }
        ));
    }

    #[test]
    fn packed_block_bytes_slice_rejects_a_float32_block() {
        let data = [1.0f32, 2.0, 3.0];
        let block = QuantizedBlock::Float32(&data);
        let error = packed_block_bytes_slice(NodeId(0), &block)
            .expect_err("a float32 block has no packed byte slice");
        assert!(matches!(
            error,
            EmitError::RenderKindMismatch {
                expected: "a packed (non-float32) block",
                found: "float32",
                ..
            }
        ));
    }
}

/// ROW 327: [`plan`] attributes each [`QuantizedBlock`] in `blocks` to the
/// node at the same position in [`block_node_ids`]'s output -- this crate's
/// documented positional contract, mirroring `crate::metal`'s own
/// `block_node_attribution_tests`. A caller whose `blocks` order disagrees
/// with its own program's declaration order used to have that mismatch
/// surface as an unrelated `WgpuError::UnsupportedBlock` (whichever node
/// lost its packed classification) instead of the precise, node-carrying
/// `InputSizeMismatch` these tests pin.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod block_node_attribution_tests {
    use alloc::vec;

    use proxima_tensor::{
        DType, Extent, IndexMap, Keep, Reduce, ReduceInit, ScalarOp, TensorError, append,
        projection,
    };

    use super::{NodeId, Op, QuantizedBlock, WgpuError, plan};

    /// `activation -> weight -> product -> sum`: the activation node is
    /// declared FIRST (`NodeId(0)`), the quantized weight node SECOND
    /// (`NodeId(1)`) -- the reverse of the order a caller who lists `blocks`
    /// weight-first (a natural "formula" reading order) would need. Returns
    /// the program and both nodes in DECLARATION order.
    fn activation_first_matmul_program(
        tokens: u32,
        out_dim: u32,
        in_dim: u32,
    ) -> (Vec<Op>, NodeId, NodeId) {
        let mut program = Vec::new();
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(tokens), Extent::Static(in_dim)],
                name: None,
            },
        );
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::UInt8,
                shape: vec![Extent::Static(out_dim), Extent::Static(in_dim)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(projection(3, &[1, 2]))),
                    (activation, IndexMap::Affine(projection(3, &[0, 2]))),
                ],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, activation, weight)
    }

    /// The ROW 327 repro: `blocks` lists the weight FIRST even though the
    /// program declares the activation first. The activation's declared
    /// shape (16x16=256 elements) is engineered to equal one Q6_K
    /// super-block's decode count (256), so the misattributed pair
    /// (activation node, weight's Q6_K block) passes the per-node shape
    /// check silently -- exactly the "silently attributed" half of ROW
    /// 327 -- and the mismatch only becomes visible at the SECOND pair
    /// (weight node, activation's Float32 block: declared 8x16=128 elements
    /// vs. the 256 actually handed).
    #[test]
    fn weight_first_blocks_against_activation_first_program_names_the_true_node() {
        let (program, _activation, weight) = activation_first_matmul_program(16, 8, 16);
        let activation_data = [0.0f32; 256]; // 16 tokens * 16 in_dim
        let packed_weight = [0u8; 210]; // one Q6_K super-block, decodes to 256 elements

        // MISORDERED: weight's block first, activation's block second --
        // `block_node_ids(&program)` is `[activation, weight]`, so this is
        // the opposite order.
        let blocks = [
            QuantizedBlock::Q6K(&packed_weight),
            QuantizedBlock::Float32(&activation_data),
        ];

        let error = match plan(&program, &[], &blocks, &[]) {
            Ok(_) => panic!("misordered blocks must never silently plan"),
            Err(error) => error,
        };

        match error {
            WgpuError::Tensor(TensorError::InputSizeMismatch {
                node,
                expected,
                found,
            }) => {
                assert_eq!(
                    node, weight,
                    "the weight node -- 8x16=128 declared elements -- must be the node \
                     named, not a downstream node the misattribution happened to also \
                     affect"
                );
                assert_eq!(expected, 128, "weight's own declared element count");
                assert_eq!(
                    found, 256,
                    "the activation's Float32 block landed on the weight node, carrying \
                     the ACTIVATION's element count"
                );
            }
            other => panic!(
                "expected InputSizeMismatch naming node {weight:?} once the shape check \
                 runs before packed classification; got {other:?} instead"
            ),
        }
    }

    /// Same shape as above, correctly ordered blocks (activation first,
    /// matching `block_node_ids`'s `[activation, weight]` declaration
    /// order): must plan cleanly, proving the fix only rejects genuine
    /// mismatches, never a correctly-ordered call.
    #[test]
    fn declaration_ordered_blocks_plan_cleanly() {
        let (program, _activation, _weight) = activation_first_matmul_program(16, 16, 16);
        let activation_data = [0.0f32; 256]; // 16 tokens * 16 in_dim
        let packed_weight = [0u8; 210]; // one Q6_K super-block, decodes to 256 = 16 * 16

        let blocks = [
            QuantizedBlock::Float32(&activation_data),
            QuantizedBlock::Q6K(&packed_weight),
        ];

        plan(&program, &[], &blocks, &[]).expect("declaration-ordered blocks must plan");
    }
}
