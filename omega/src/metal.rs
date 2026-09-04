//! The Metal execution driver: runs an [`omega::emit`](crate::emit)-produced
//! [`Kernel`] on a real GPU and proves it agrees with
//! [`proxima_tensor::cpu::evaluate`] — the piece `msl.rs`'s own doc says is
//! "the device driver's job, composed on top."
//!
//! # Prepare pipeline
//!
//! [`execute`] mirrors [`proxima_tensor::cpu::evaluate`]'s semantics
//! exactly, over the same public API that function itself is built from:
//! [`proxima_tensor::infer`] resolves shapes and symbols, [`proxima_tensor::bind()`]
//! produces the flat [`BoundOp`] sequence (with `Reduce(Elementwise)` fusion already
//! decided), and per-nest buffer retirement is recomputed from that sequence
//! the same way `cpu::evaluate` itself does, through the shared
//! [`proxima_tensor::node_retirement`] — a node's
//! device buffer is freed the moment nothing later in the sequence reads it.
//! What differs is only the last mile: instead of interpreting a `BoundOp` with
//! nested loops, each one is emitted to MSL, compiled (or reused from cache),
//! and dispatched.
//!
//! # Uniforms packing
//!
//! `msl.rs` never bakes a `BoundOp`'s concrete extents/strides/bases into
//! source text — they are read at kernel runtime out of a `constant
//! Uniforms&` buffer whose MSL struct layout is rendered field-by-field in
//! `crate::msl::render_elementwise`, `render_reduce`
//! and `render_scan`. Every field in all three is
//! MSL `long` (an 8-byte, 8-byte-aligned integer) or an array of `long`, so
//! there is no interior struct padding to reason about: packing is a flat
//! concatenation of `i64`s in the exact field order those functions emit.
//! `pack_elementwise_uniforms`, `pack_reduce_uniforms` and
//! `pack_scan_uniforms` each carry a comment pointing at the struct
//! declaration they mirror, byte for byte.
//!
//! # Execution model
//!
//! One `MTLCommandBuffer` per [`execute`] call, not per op: every `BoundOp`
//! in the program is encoded — its own `MTLComputeCommandEncoder`, ended
//! before the next op's encoder is opened — into that SAME command buffer,
//! and only then is it `commit()`ted and `waitUntilCompleted()` exactly
//! once, in [`execute`]. Every expression used to pay a full CPU<->GPU
//! round trip; batching means only the genuine program outputs
//! (`finish`'s `effective_outputs`) ever cross back to the host, and
//! intermediates never do (they already didn't — `device_buffers` keeps
//! them GPU-resident between ops; what changes here is that the CPU no
//! longer blocks between ops either).
//!
//! Ordering is guaranteed, not assumed: a later op reading a buffer an
//! earlier op wrote is correct because every buffer here comes from
//! `device.newBuffer*` (see `allocate_buffer`, `upload_block`) with
//! `MTLResourceOptions::StorageModeShared` only — never
//! `HazardTrackingModeUntracked` — and a buffer's `hazardTrackingMode` for
//! any resource created directly from a device (as opposed to a heap)
//! defaults to tracked (`objc2-metal-0.3.2`'s
//! `src/generated/MTLResource.rs:326-329`: "Resources created from heaps
//! are by default untracked, whereas resources created from the device are
//! by default tracked."). Metal's documented contract for a tracked
//! resource is that it inserts an implicit execution barrier between two
//! encoders in the *same* command buffer whenever the later one reads what
//! the earlier one wrote. That guarantee composes with [`execute`] encoding
//! `prepared.resolved` strictly in program order (the same order
//! `prepare`'s own [`proxima_tensor::node_retirement`] call already relies on for liveness), so
//! sequential encode order plus default hazard tracking is the mechanism —
//! not an assumption that the GPU happens to serialize. This holds equally
//! for the no-copy buffers `upload_block` hands out (see "Host buffer
//! upload" below): `newBufferWithBytesNoCopy_length_options_deallocator`
//! takes the same `MTLResourceOptions`, so its hazard mode is identical.
//!
//! Every `MTLBuffer` is `storageModeShared`: on Apple Silicon's unified
//! memory, that makes reading a result back a plain pointer read, no blit
//! pass. Compiled `MTLLibrary`/`MTLComputePipelineState` pairs are cached
//! by kernel source text within one [`execute`] call, since `msl.rs`'s own
//! module doc proves two structurally-identical `BoundOp`s emit
//! byte-identical source. `MTLCompileOptions::mathMode` is pinned to
//! `Safe`, never the default — parity against the CPU interpreter demands
//! IEEE behavior, not whatever Metal's fast-math would substitute.
//!
//! # Gather fault reporting
//!
//! `cpu::evaluate` returns `TensorError::GatherIndexOutOfRange` when a
//! fetched index falls outside its dim's extent; a GPU kernel cannot
//! propagate a `Result`, so `msl.rs` clamps for memory safety but also
//! `atomic_fetch_max`s the offending index into a per-gather-slot `Fault`
//! buffer (see that module's doc). `encode_op` allocates and zero-fills
//! that buffer before every dispatch that gathers, but a fault buffer is
//! only CPU-visible once the whole command buffer completes, so — unlike a
//! per-op wait — [`execute`] cannot check it until after its single
//! end-of-program `waitUntilCompleted`. It then walks every op that
//! gathered, in program order, and — via `check_gather_fault` — turns the
//! first nonzero slot into the identical `TensorError` `cpu.rs` would
//! report for the same fetched index, wired through [`MetalError`]'s
//! `#[from]` so [`execute`] and `cpu::evaluate` produce `assert_eq!`-equal
//! errors. Ops after the one that would have faulted still get encoded and
//! dispatched (clamping keeps that memory-safe) — but the `Err` [`execute`]
//! returns is unaffected: everything downstream of the fault is discarded
//! the moment that `Err` propagates, so it is exactly what a fail-fast
//! per-op wait would have reported.
//!
//! # Host buffer upload
//!
//! `upload_block` is the one call on the copy of a caller-owned `&[f32]`
//! into device memory (`upload_uniforms` copies too, but a *locally
//! packed* `Vec<u8>`, not caller data, so it is out of scope here). On
//! unified memory that copy is pointless for the `Float32` path — CPU and
//! GPU already address the same DRAM — so `upload_block_as_float` takes
//! the zero-copy `newBufferWithBytesNoCopy` path whenever `data`'s pointer
//! AND byte length are both a multiple of [`page_size`] (that API's hard
//! requirement), and otherwise falls back to the copying `newBufferWithBytes`
//! path used everywhere else in this file. A `Float16` node's buffer is
//! narrowed into a freshly allocated `Vec<f16>` first (see the dtype
//! section below); that allocation is local to `upload_block_as_half` and
//! drops when it returns, so it can never take the no-copy path — doing so
//! would hand Metal a dangling pointer the instant the function returns,
//! since no deallocator callback is wired to keep the `Vec` alive for the
//! GPU's sake. `Float16` uploads therefore always copy. Which path ran is
//! never silent: [`NOCOPY_BUFFER_UPLOADS`] / [`COPYING_BUFFER_UPLOADS`]
//! (`proxima_telemetry::metric::Counter`, the same instrument
//! `proxima_tensor::instrument` already uses) are incremented on every
//! real call, so a caller — or this driver's own test suite — can read back
//! what fraction of real uploads actually took the no-copy path instead of
//! assuming it from the code alone.
//!
//! # Resident blocks — the copying path's own cache
//!
//! "Copying uploads are deliberately not cached" (see `NOCOPY_BUFFERS`'s own
//! doc) is right for a block whose bytes genuinely change every call --
//! `ids`/`rope_cos`/`rope_sin`/the KV cache's own blocks -- because a stale
//! copy would silently serve last token's data forever. It is wrong for a
//! model's own weights: `proxima-model-interop`'s `BoundWeights::owned`/
//! `packed` are bound once at load and never mutated again, so their
//! `(pointer, len)` is stable for the caller's whole process, and re-copying
//! them every token moves ~5.84 GB/token for data that never changed
//! (`proxima-tensor/docs/discipline.md` ROW 82). That distinction -- which
//! names are the caller's own static weights versus which change every step
//! -- is known to the CALLER (`generate.rs` builds `named_blocks` from two
//! structurally different sources) and destroyed the moment they flatten
//! into one `&[(&str, QuantizedBlock)]`. [`Plan::mark_resident`] hands that
//! knowledge back in: a caller-supplied name set, checked once per [`Plan`]
//! build against the program's own declared [`Op::name`]s, never against raw
//! bytes -- see that method's own doc for why NAME is safe to classify
//! against here even though `NOCOPY_BUFFERS`-style caching must never be
//! keyed on name.
//!
//! # dtype and device-buffer marshalling
//!
//! `execute`'s own host contract stays f32 in and f32 out — `blocks:
//! &[&[f32]]`, [`Evaluated`] carries `Vec<f32>` — the same contract
//! `cpu::evaluate` has, so a caller compares the two directly. What varies
//! *underneath* that contract is the device buffer each node's own dtype
//! ([`Op::dtype`]) gets: a `Float32` node uploads/allocates/reads back
//! 4-byte-per-element buffers exactly as before, but a `Float16` node's
//! buffer is 2 bytes per element — `upload_block` narrows the caller's
//! `f32` host data to `half::f16` once, at the host/device boundary, and
//! `read_back` widens it back once, at the same boundary, on the way out.
//! Every byte a dispatch's kernel actually reads or writes in between —
//! every input, every intermediate `BoundOp` output, the final result
//! buffer — is genuinely half-width; the narrowing/widening is a one-time
//! host-boundary conversion, not a disguise for still moving 4 bytes per
//! element on the GPU-resident path this feature targets. A gather's
//! `indices` node is the one exemption, exactly as in
//! `reject_unsupported_gpu_dtype`: an index value stays f32-encoded
//! regardless of its own declared dtype, matching `cpu.rs`'s own stance.

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;
use core::ffi::c_void;
use core::mem::{size_of, size_of_val};
use core::ptr::NonNull;
#[cfg(feature = "metal-buffer-pool")]
use std::collections::HashMap;
use std::sync::OnceLock;

use half::f16;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};
use proxima_telemetry::counter;
use proxima_telemetry::metric::Counter;

#[cfg(feature = "instrument")]
use proxima_tensor::instrument::{elapsed_ticks, read_ticks};
use proxima_tensor::{
    BoundOp, BoundOpKind, DType, Evaluated, Keep, Lookup, NodeId, Op, QuantizedBlock, Shapes,
    TensorError, bind, block_node_ids, correct_packed_matmul_layouts, index_node_ids, infer,
    node_retirement, prune_dead, resolve_named_blocks,
};

use crate::error::EmitError;
#[cfg(feature = "instrument")]
use crate::msl::diagnose_packed_row_block;
use crate::msl::{gather_count, kernel_cache_key, kernel_dispatch_shape, reduction_dims};
use crate::{Binding, GridSpec, Kernel, PackedCodec, PackedOperands, emit};

/// A live Metal buffer handle — the shape every device-buffer table and
/// return value in this file traffics in.
type MetalBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
/// A device buffer plus this node's byte OFFSET into it -- most nodes own
/// their whole buffer (offset 0), but a tensor served by
/// [`checkpoint_mapping_offset`] shares one buffer across many nodes, each
/// at its own offset. Carrying the pair through `device_buffers` is what
/// lets [`bind_buffers`] bind the right slice with `setBuffer:offset:atIndex:`
/// instead of every binding assuming offset 0.
type DeviceBuffer = (MetalBuffer, usize);

/// One gathering op's deferred fault check: the op it came from, its fault
/// buffer, and how many gather slots that buffer holds. [`encode_op`]
/// produces these; [`execute`] checks them all after its single
/// end-of-program wait (see the module doc's "Gather fault reporting").
type PendingFault<'a> = (&'a BoundOp, MetalBuffer, usize);

/// Everything [`execute`] can fail with: a missing device, any device
/// operation that returned a Metal-side failure (compiling source, creating
/// a pipeline, or one of the handful of `Option`-returning calls that are
/// only ever `None` on a broken host), or a `BoundOp`/program-shaped fault
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
/// This thread's Metal device paired with its command queue — both created
/// once per thread rather than per [`execute`] call.
type DeviceAndQueue = (
    Retained<ProtocolObject<dyn MTLDevice>>,
    Retained<ProtocolObject<dyn MTLCommandQueue>>,
);

thread_local! {
    /// Compiled pipelines, keyed by [`kernel_cache_key`]'s cheap structural
    /// fingerprint (not [`Kernel::source`] — deriving that string is exactly
    /// the cost this cache exists to avoid on a hit), for the lifetime of
    /// the thread rather than of one [`execute`] call.
    ///
    /// This was per-call, which meant EVERY `execute` compiled every kernel
    /// from MSL source before dispatching it. A serving loop runs the same
    /// graph thousands of times, so that is thousands of redundant
    /// compiles — measured at 3.2 ms for a 2.36 MB matvec and 8.3 ms for a
    /// 9.44 MB one, against llama.cpp's 17.62 ms for an entire 7B token.
    /// `thread_local` rather than a process-wide `OnceLock`: `Retained<_>` of
    /// an `objc2` protocol object is not `Send`/`Sync`, and a per-thread
    /// cache needs no lock on the dispatch path anyway.
    static PIPELINE_CACHE: RefCell<BTreeMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>> =
        RefCell::new(BTreeMap::new());

    /// The device and its command queue, created once per thread. Both were
    /// also per-call; `MTLCreateSystemDefaultDevice` plus `newCommandQueue`
    /// is not free, and nothing about either depends on the program being
    /// run.
    static DEVICE_AND_QUEUE: RefCell<Option<DeviceAndQueue>> = const { RefCell::new(None) };
}

/// This thread's `MTLDevice::currentAllocatedSize` -- Metal's own count of
/// bytes it has allocated for every buffer, texture and heap this device
/// owns, read directly rather than summed from this driver's own caches, so
/// it also catches anything the caches under- or over-count. `None` before
/// any Metal call on this thread has created a device.
#[must_use]
pub fn current_allocated_size() -> Option<u64> {
    let (device, _queue) = device_and_queue().ok()?;
    Some(device.currentAllocatedSize() as u64)
}

/// This thread's Metal device and command queue, created on first use.
fn device_and_queue() -> Result<DeviceAndQueue, MetalError> {
    DEVICE_AND_QUEUE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some(existing) = slot.as_ref() {
            return Ok(existing.clone());
        }
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| MetalError::CompileFailed {
                log: "device refused to create a command queue".to_string(),
            })?;
        let pair = (device, queue);
        *slot = Some(pair.clone());
        Ok(pair)
    })
}

/// Runs a tensor program on the system's default Metal device.
///
/// Everything about a program that does not change between runs, resolved
/// ONCE so a serving loop stops re-deriving it per token.
///
/// [`execute`] re-ran `infer` + `bind` on every call, then re-derived which
/// operands are packed, allocated fresh device buffers, and read the result
/// back. Measured on this box (`omega/examples/q4k_matvec_probe.rs`, two
/// problem sizes so the intercept separates from the slope): **0.191 ms of
/// fixed cost per call on the f32 arm and 0.400 ms on the packed arm**. A
/// real forward is 1196 nodes, so at one `execute` per node that is 228-478
/// ms per forward of overhead against llama.cpp Metal's 17.62 ms for the
/// whole token (`proxima-tensor/docs/discipline.md` ROW 71).
///
/// What a caller can do with this that they could not do before: prepare a
/// program once and run it many times. That is the entire justification for
/// the type — the shapes, the bound ops, the retirement schedule and the
/// codec set are all functions of the PROGRAM, and a serving loop holds the
/// program fixed while the block DATA changes every token.
pub struct Plan {
    /// The plan owns its program: `finish` needs it for output dtypes, and a
    /// plan that borrowed it could not outlive the caller's buffer.
    program: Vec<Op>,
    prepared: Prepared,
    packed_operands: PackedOperands,
    block_dtypes: Vec<DType>,
    /// Every block-input node a caller has told this plan, via
    /// [`Plan::mark_resident`], is bound to data that never changes across
    /// calls -- empty until that method runs, since [`plan`] itself has no
    /// way to know a caller's residency intent from codecs/shapes alone.
    resident_nodes: BTreeSet<NodeId>,
}

impl Plan {
    /// Tells this plan which of its named block inputs are the CALLER's own
    /// static data -- model weights bound once at load and never mutated
    /// again -- so [`execute_plan`]'s upload loop may cache and reuse their
    /// device buffer on the copying path instead of re-copying every call.
    /// See the module doc's "Resident blocks" section for the full argument.
    ///
    /// `resident_names` is checked against each block-input node's own
    /// declared [`Op::name`] -- a one-time string compare over the program's
    /// declared inputs, off the per-token upload path -- never against the
    /// block's bytes or pointer. That is the necessary half of the
    /// invariant this driver's `NOCOPY_BUFFERS` cache cannot provide on its
    /// own: an address match alone cannot distinguish a weight buffer that
    /// never moves from a freshly reallocated `ids`/KV-cache buffer that
    /// coincidentally lands at a freed address of the same size. The NAME
    /// check happens here, once per plan, specifically so the per-token
    /// upload path never has to trust an address by itself.
    ///
    /// Safe and cheap to call every token even though [`plan`]/[`plan_named`]
    /// currently rebuild a fresh [`Plan`] every step (`plan_hits=0`,
    /// `proxima-tensor/docs/discipline.md` ROW 82): this is a scan over the
    /// block-input node list matching each node's name against
    /// `resident_names`, not a device operation.
    pub fn mark_resident(&mut self, resident_names: &BTreeSet<&str>) {
        self.resident_nodes = self
            .prepared
            .block_nodes
            .iter()
            .filter(|node| {
                self.program[node.0 as usize]
                    .name()
                    .is_some_and(|name| resident_names.contains(name))
            })
            .copied()
            .collect();
    }
}

/// Which of `block_nodes`' entries carry a codec [`crate::msl::emit`] has an
/// unpack kernel for (`Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `Q4_0`, `Float16`,
/// `BFloat16`), keyed to its [`PackedCodec`] — the single place this crate
/// decides "packed AND which codec," shared by [`plan`] and [`prepare`] so
/// the two cannot drift on it. `Float16` earns a codec slot despite needing
/// no unpack FUNCTION (see `msl::FLOAT16_BLOCK_BYTES`'s own doc) because its
/// buffer still needs a non-`float`, non-`uchar` binding type -- `None`
/// would route it through `Float32`'s plain-array path and bind it as the
/// kernel's own accumulator type, which is wrong the moment a `Float16`
/// weight multiplies an `f32` activation.
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
            QuantizedBlock::Float32(_) => None,
        })
        .collect()
}

/// Resolves a program into a reusable [`Plan`]. `blocks` is read for its
/// CODECS and shapes only — the data is not captured, so the same plan runs
/// against fresh block data every call.
///
/// # Errors
/// Propagates inference, binding, dtype-gate and block-shape failures.
pub fn plan(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock<'_>],
    outputs: &[NodeId],
) -> Result<Plan, MetalError> {
    #[cfg(feature = "instrument")]
    let prepare_started = read_ticks();
    let prepared = prepare(program, symbols, blocks, outputs)?;
    #[cfg(feature = "instrument")]
    {
        counter!(PREPARE_CALLS, 1);
        counter!(PREPARE_TICKS, elapsed_ticks(prepare_started));
    }
    let packed_operands = packed_operands_of(&prepared.block_nodes, blocks);
    let block_dtypes = prepared
        .block_nodes
        .iter()
        .map(|node| gpu_dtype(program, &prepared.index_nodes, *node))
        .collect();
    Ok(Plan {
        program: program.to_vec(),
        prepared,
        packed_operands,
        block_dtypes,
        resident_nodes: BTreeSet::new(),
    })
}

/// Same contract as [`proxima_tensor::cpu::evaluate`], and returns the same
/// [`Evaluated`] type — a CPU run and a Metal run report the identical
/// shape, so a parity test compares them directly with no adapter on either
/// side (see `Evaluated`'s own doc). `blocks` binds [`Op::Input`] inputs
/// positionally, `outputs` selects which nodes to return data for (the root
/// only, if empty).
pub fn execute(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock<'_>],
    outputs: &[NodeId],
) -> Result<Evaluated, MetalError> {
    let resolved_plan = plan(program, symbols, blocks, outputs)?;
    execute_plan(&resolved_plan, blocks)
}

/// Runs an already-resolved [`Plan`] against fresh block data. This is the
/// serving-loop entry point: the plan is built once, this is called per
/// token, and none of `infer`/`bind`/codec-resolution happens here.
///
/// # Errors
/// Propagates block-codec and Metal driver failures.
pub fn execute_plan(plan: &Plan, blocks: &[QuantizedBlock<'_>]) -> Result<Evaluated, MetalError> {
    let prepared = &plan.prepared;
    let packed_operands = &plan.packed_operands;

    let (device, queue) = device_and_queue()?;

    let mut device_buffers: BTreeMap<NodeId, DeviceBuffer> = BTreeMap::new();
    #[cfg(feature = "instrument")]
    let block_upload_started = read_ticks();
    for ((node, block), dtype) in prepared
        .block_nodes
        .iter()
        .zip(blocks.iter())
        .zip(plan.block_dtypes.iter())
    {
        #[cfg(feature = "instrument")]
        {
            counter!(BLOCK_UPLOAD_CALLS, 1);
            counter!(BLOCK_UPLOAD_BYTES, block_byte_len(block) as u64);
        }
        let resident = plan.resident_nodes.contains(node);
        let buffer = match block {
            QuantizedBlock::Float32(data) => upload_block(&device, data, *node, *dtype, resident)?,
            // `Float16`/`BFloat16` upload their bytes UNCHANGED, same as
            // every packed codec above -- there is no host-side narrowing
            // step (unlike `upload_block`'s `Float32 -> Float16` path,
            // which narrows a caller's `&[f32]`): a `Float16` weight's on-
            // disk bytes already ARE its device buffer's bytes (native
            // `half`), and a `BFloat16` weight's bytes are widened entirely
            // on the GPU at the read (`msl::BF16_UNPACK_MSL`), never on the
            // host.
            QuantizedBlock::Q4K(bytes)
            | QuantizedBlock::Q5K(bytes)
            | QuantizedBlock::Q6K(bytes)
            | QuantizedBlock::Q8_0(bytes)
            | QuantizedBlock::Q4_0(bytes)
            | QuantizedBlock::Float16(bytes)
            | QuantizedBlock::BFloat16(bytes) => upload_packed_bytes(&device, bytes, resident)?,
        };
        device_buffers.insert(*node, buffer);
    }
    #[cfg(feature = "instrument")]
    counter!(BLOCK_UPLOAD_TICKS, elapsed_ticks(block_upload_started));

    let command_buffer = queue
        .commandBuffer()
        .ok_or_else(|| MetalError::CompileFailed {
            log: "command queue refused to hand out a command buffer".to_string(),
        })?;

    // ONE `MTLComputeCommandEncoder` for the whole program, opened here and
    // `endEncoding()`d once below, after every op has been encoded into it --
    // not one per op. `computeCommandEncoder()` (no dispatch-type argument)
    // defaults to `MTLDispatchTypeSerial` (Apple's own documented default,
    // confirmed against `objc2-metal-0.3.2`'s own `MTLDispatchType` doc:
    // "Command encoder dispatches are executed in dispatched order"), the
    // IDENTICAL dispatch type every one of the former per-op encoders already
    // used -- this change narrows encoder COUNT, not dispatch semantics.
    // Ordering and hazard-tracked visibility between two dispatches in one
    // serial encoder is Metal's documented behavior for tracked resources
    // (every buffer here is `storageModeShared`, never
    // `HazardTrackingModeUntracked` -- see the module doc's "Execution
    // model"), so this is the SAME correctness argument that section already
    // makes for two encoders in one command buffer, just one level tighter:
    // one encoder's own dispatches were always ordered and hazard-tracked
    // relative to each other, encoder boundaries or not.
    let encoder =
        command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::CompileFailed {
                log: "command buffer refused to hand out a compute encoder".to_string(),
            })?;

    // `metal-buffer-pool` reclaim bookkeeping: which node ids are op OUTPUTS
    // (never block inputs) and their `(bucket, dtype)` pool key -- built once,
    // off `prepared.resolved`'s own node ids and the same `bound_output_len`/
    // `dtype` `allocate_buffer` sizes from inside `encode_op`, run through the
    // IDENTICAL `pool_bucket` function `allocate_buffer` itself uses. That
    // identity is what keeps a reclaimed buffer's stored key equal to its own
    // real Metal capacity -- see `OUTPUT_BUFFER_POOL`'s doc for the full
    // invariant and the liveness argument this feeds.
    #[cfg(feature = "metal-buffer-pool")]
    let output_meta: BTreeMap<NodeId, (usize, DType)> = prepared
        .resolved
        .iter()
        .map(|bound| {
            let byte_length = bound_output_len(bound).max(1) * bound.dtype.size_bytes();
            (bound.node, (pool_bucket(byte_length), bound.dtype))
        })
        .collect();
    #[cfg(feature = "metal-buffer-pool")]
    let mut reclaim_stash: Vec<(MetalBuffer, usize, DType)> = Vec::new();

    // pipelines live in this thread's `PIPELINE_CACHE`, not here: see that
    // static's own doc for why per-call was the defect.
    // (bound op, its fault buffer, gather count) for every op that gathered —
    // checked only after the single end-of-program wait below, since a fault
    // buffer is not CPU-visible until the command buffer it was written in
    // completes. See the module doc's "Gather fault reporting" section.
    let mut pending_faults: Vec<PendingFault<'_>> = Vec::new();
    for (position, bound) in prepared.resolved.iter().enumerate() {
        let fault = encode_op(
            &device,
            &encoder,
            &mut device_buffers,
            bound,
            packed_operands,
            None,
        )?;
        if let Some((fault_buffer, gathers)) = fault {
            pending_faults.push((bound, fault_buffer, gathers));
        }
        // `metal-buffer-pool` off: identical to this function before the
        // feature existed -- a retired buffer is looked up once and dropped.
        // `metal-buffer-pool` on: same lookup-and-remove, but an op-OUTPUT
        // buffer (per `output_meta`) is ALSO cloned into `reclaim_stash`
        // rather than only dropped -- the clone is not pushed into the pool
        // until after this call's `waitUntilCompleted` below, so nothing here
        // hands a still-pending buffer back out early.
        #[cfg(not(feature = "metal-buffer-pool"))]
        for retired in &prepared.retires[position] {
            device_buffers.remove(retired);
        }
        #[cfg(feature = "metal-buffer-pool")]
        for retired in &prepared.retires[position] {
            if let Some((buffer, _offset)) = device_buffers.remove(retired)
                && let Some(&(bucket, dtype)) = output_meta.get(retired)
            {
                reclaim_stash.push((buffer, bucket, dtype));
            }
        }
    }
    encoder.endEncoding();

    #[cfg(feature = "instrument")]
    let gpu_exec_started = read_ticks();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    #[cfg(feature = "instrument")]
    {
        counter!(GPU_EXEC_CALLS, 1);
        counter!(GPU_EXEC_TICKS, elapsed_ticks(gpu_exec_started));
    }

    for (bound, fault_buffer, gathers) in &pending_faults {
        check_gather_fault(bound, fault_buffer, *gathers)?;
    }

    // Everything still in `device_buffers` at this point (never removed by
    // the retirement loop above -- e.g. the root output and any other
    // `effective_outputs` node) is now also safe to reclaim: the single
    // command buffer this whole program ran in has already completed via
    // `waitUntilCompleted` above. Cloned, not moved, so `finish` below still
    // reads the same buffers to copy their bytes out.
    #[cfg(feature = "metal-buffer-pool")]
    for (node, (buffer, _offset)) in &device_buffers {
        if let Some(&(bucket, dtype)) = output_meta.get(node) {
            reclaim_stash.push((buffer.clone(), bucket, dtype));
        }
    }

    let evaluated = finish(
        &plan.program,
        &prepared.index_nodes,
        &prepared.shapes,
        &prepared.effective_outputs,
        &device_buffers,
        prepared.root,
    )?;

    // `OUTPUT_POOL_MAX_PER_BUCKET`: a buffer beyond the cap for its
    // `(bucket, dtype)` slot is dropped here (ordinary `Retained` drop, same
    // as the `metal-buffer-pool`-off arm always did for every buffer) rather
    // than retained -- see `OUTPUT_POOL_MAX_PER_BUCKET`'s own doc for why
    // this is a safety net, not the primary bound.
    #[cfg(feature = "metal-buffer-pool")]
    OUTPUT_BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        for (buffer, bucket, dtype) in reclaim_stash {
            let slot = pool.entry((bucket, dtype)).or_default();
            if slot.len() < OUTPUT_POOL_MAX_PER_BUCKET {
                slot.push(buffer);
            }
        }
    });

    Ok(evaluated)
}

/// A device buffer a CALLER allocates, owns, and keeps alive across multiple
/// [`execute_plan_with_placements`] calls — the type this module's other
/// buffers (`MetalBuffer`, private) never had to be public for, since
/// `execute_plan`/`execute_plan_op_timed` allocate and own every buffer
/// themselves. Get one from [`allocate_placed_buffer`]; read one back with
/// [`read_placed_buffer_f32`].
#[cfg(feature = "metal-output-placement")]
pub type PlacedBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;

/// Allocates a `storageModeShared` buffer of `byte_len` bytes that OUTLIVES
/// any one [`execute_plan_with_placements`] call — the caller holds it,
/// passes `&buffer` into as many calls as it likes, and only it decides when
/// the buffer is dropped. Mirrors `allocate_buffer`'s own device call,
/// public and un-sized-to-an-op because a placed buffer's size is the
/// caller's own layout decision (e.g. a whole KV-cache page), not one op's
/// `bound_output_len`.
///
/// # Errors
/// Propagates a Metal device/driver failure to allocate.
#[cfg(feature = "metal-output-placement")]
pub fn allocate_placed_buffer(byte_len: usize) -> Result<PlacedBuffer, MetalError> {
    let (device, _queue) = device_and_queue()?;
    device
        .newBufferWithLength_options(byte_len.max(1), MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| MetalError::CompileFailed {
            log: "device refused to allocate a placed buffer".to_string(),
        })
}

/// Reads `element_count` `f32`s back from `buffer` starting at `byte_offset`
/// — the read-back counterpart to a placed write, for a caller that wants to
/// inspect what an [`execute_plan_with_placements`] call wrote without going
/// through [`Evaluated`]'s own output set (a placed node need not be a named
/// output at all).
///
/// # Panics
/// Never panics; an out-of-bounds `byte_offset`/`element_count` pair is the
/// caller's own contract to keep (documented on [`execute_plan_with_placements`]),
/// the same trust boundary `bind_buffers`' SAFETY comment already states for
/// every kernel-side binding.
#[cfg(feature = "metal-output-placement")]
#[must_use]
pub fn read_placed_buffer_f32(
    buffer: &PlacedBuffer,
    byte_offset: usize,
    element_count: usize,
) -> Vec<f32> {
    let pointer = buffer.contents();
    // SAFETY: `storageModeShared` is CPU-visible once the command buffer
    // that wrote it has completed, and `execute_plan_with_placements` always
    // `waitUntilCompleted`s before returning. `byte_offset`/`element_count`
    // staying inside `buffer`'s allocated length is the caller's contract —
    // this function has no independent way to learn that length's meaning
    // (a caller-owned buffer may hold several placed nodes at once).
    unsafe {
        let base = pointer.as_ptr().cast::<u8>().add(byte_offset).cast::<f32>();
        core::slice::from_raw_parts(base, element_count)
    }
    .to_vec()
}

/// [`execute_plan`], plus the ability to route one or more nodes' outputs
/// into a buffer the CALLER owns (`output_placements`), and to bind one or
/// more [`Op::Input`] nodes DIRECTLY to a buffer
/// the caller already owns on the device (`input_placements`), skipping the
/// per-call `upload_block`/`upload_packed_bytes` host round trip entirely.
/// Both are `&[(node, buffer, byte_offset)]` — deliberately plain tuples
/// over a named struct or a single slice with a direction field: the call
/// site destructures each triple positionally either way (`for (node,
/// buffer, offset) in placements`), and TWO plain slices already carry the
/// direction in which parameter a placement is passed to, at zero added
/// type cost. Compare both shapes, written out, for one KV-cache-shaped
/// call (`kv` bound as this program's growing history, `row` the freshly
/// computed token this call is adding to it, both backed by ONE buffer):
///
/// ```text
/// // two plain slices (chosen) — no new type, direction = which parameter
/// execute_plan_with_placements(
///     &plan, &blocks,
///     &[(kv_input_node, &kv_buffer, 0)],
///     &[(row_output_node, &kv_buffer, cached_len * row_bytes)],
/// )?;
///
/// // one slice + a direction field (rejected) — an enum earns its keep only
/// // if some caller needs to build a MIXED, order-independent placement
/// // list at runtime; no caller here does, so it is a type with nothing to
/// // do beyond what two parameter names already say for free.
/// execute_plan_with_placements(&plan, &blocks, &[
///     Placement { node: kv_input_node, buffer: &kv_buffer, offset: 0, direction: Direction::Input },
///     Placement { node: row_output_node, buffer: &kv_buffer, offset: cached_len * row_bytes, direction: Direction::Output },
/// ])?;
/// ```
///
/// Three hazards this signature exists to name explicitly, not paper over:
///
/// - **Cross-invocation liveness.** `Prepared::retires` computes per-
///   program liveness only (`bound_op_retirement`'s own doc) — it has no
///   notion of a buffer surviving into the NEXT `execute_plan_with_placements`
///   call. Every placed node (input or output) is therefore excluded from
///   this call's own retire sweep below: even if something inside THIS
///   program reads or writes it after its nominal last use, this function
///   will not drop this call's reference to it. The caller's own
///   `PlacedBuffer` handle is what actually keeps the GPU allocation alive
///   across calls; this exclusion just stops this function's bookkeeping
///   map from disagreeing with that fact.
/// - **Cross-invocation aliasing.** A placed buffer written by one call may
///   be read (as an input placement) by a LATER call. That is safe because
///   every call `commit`s and `waitUntilCompleted`s its own command buffer
///   before returning (this function, like [`execute_plan`], never overlaps
///   two command buffers) — the next call's encoder cannot begin recording
///   real GPU work against the buffer until this call's write has completed
///   and is CPU/GPU-visible: a strict happens-before, stronger than same-
///   encoder hazard tracking needs to be.
/// - **Within-call aliasing: one op writes a placed buffer, a LATER op in
///   the SAME program reads it — the KV-cache shape this exists for.** This
///   is covered by the SAME mechanism [`execute_plan`]'s own opening comment
///   already establishes for two dispatches sharing one serial encoder:
///   `MTLDispatchTypeSerial` guarantees encode order is execution order, and
///   every buffer here is device-allocated `storageModeShared` (never
///   `HazardTrackingModeUntracked`), so Metal's automatic hazard tracking
///   inserts an implicit barrier before the later dispatch. That tracking
///   operates at whole-`MTLResource` granularity, not the sub-range named by
///   `setBuffer:offset:atIndex:` — a write at one offset and a read at a
///   DIFFERENT offset of the SAME `MTLBuffer` object are still the same
///   resource to the tracker, so the barrier applies regardless of which
///   byte ranges the two dispatches actually touch. Nothing here is new
///   relative to what every existing multi-op program already relies on
///   (op N's freshly allocated output, read by op N+1) — placement changes
///   WHERE the write lands, not whether ordering holds. The proof, not just
///   the argument, is `metal_output_placement.rs`'s
///   `a_program_reads_a_placed_write_from_a_later_op_in_the_same_call` test:
///   if hazard tracking did not cover this, that test would read stale
///   (pre-write) bytes instead of the fresh write, and it does not.
///
/// # Errors
/// Propagates block-codec and Metal driver failures, same as [`execute_plan`].
#[cfg(feature = "metal-output-placement")]
pub fn execute_plan_with_placements(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
) -> Result<Evaluated, MetalError> {
    let prepared = &plan.prepared;
    let packed_operands = &plan.packed_operands;
    let input_placed: BTreeMap<NodeId, (&PlacedBuffer, usize)> = input_placements
        .iter()
        .map(|(node, buffer, offset)| (*node, (*buffer, *offset)))
        .collect();
    let output_placed: BTreeMap<NodeId, (&PlacedBuffer, usize)> = output_placements
        .iter()
        .map(|(node, buffer, offset)| (*node, (*buffer, *offset)))
        .collect();
    let (device, queue) = device_and_queue()?;

    let mut device_buffers: BTreeMap<NodeId, DeviceBuffer> = BTreeMap::new();
    for ((node, block), dtype) in prepared
        .block_nodes
        .iter()
        .zip(blocks.iter())
        .zip(plan.block_dtypes.iter())
    {
        // an input-placed node skips the host round trip entirely: its
        // buffer is already on the device, owned by the caller, and this
        // call's own `block` entry for it (still required, positionally, to
        // keep `block_nodes.iter().zip(blocks.iter())` aligned) is unused.
        // The caller's own offset travels in the same `DeviceBuffer` tuple
        // every other node's buffer carries -- `buffer_for`/`bind_buffers`
        // read it generically, with no separate offset map required.
        if let Some((buffer, offset)) = input_placed.get(node) {
            device_buffers.insert(*node, ((*buffer).clone(), *offset));
            continue;
        }
        let resident = plan.resident_nodes.contains(node);
        let buffer = match block {
            QuantizedBlock::Float32(data) => upload_block(&device, data, *node, *dtype, resident)?,
            QuantizedBlock::Q4K(bytes)
            | QuantizedBlock::Q5K(bytes)
            | QuantizedBlock::Q6K(bytes)
            | QuantizedBlock::Q8_0(bytes)
            | QuantizedBlock::Q4_0(bytes)
            | QuantizedBlock::Float16(bytes)
            | QuantizedBlock::BFloat16(bytes) => upload_packed_bytes(&device, bytes, resident)?,
        };
        device_buffers.insert(*node, buffer);
    }

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

    // `PROXIMA_PLACEMENT_POSITION_DUMP` -- diagnostic-only, `instrument`-gated,
    // default-off, same convention as `PROXIMA_METAL_OP_PROFILE_STEP`
    // (`generate.rs`'s own doc): prints each output-placed node's WRITE
    // position and each of its aliased readers' own position, the direct
    // evidence a write-before-read ordering claim needs (this function's own
    // doc, "Within-call aliasing"). Unset in every production run.
    #[cfg(feature = "instrument")]
    let placement_dump = std::env::var("PROXIMA_PLACEMENT_POSITION_DUMP").is_ok();
    let mut pending_faults: Vec<PendingFault<'_>> = Vec::new();
    for (position, bound) in prepared.resolved.iter().enumerate() {
        #[cfg(feature = "instrument")]
        if placement_dump {
            if output_placed.contains_key(&bound.node) {
                std::eprintln!("write position={position} node={:?}", bound.node);
            }
            for (operand, _, _) in bound.operands() {
                if input_placed.contains_key(operand) {
                    std::eprintln!(
                        "read  position={position} node={:?} reads={:?}",
                        bound.node,
                        operand
                    );
                }
            }
        }
        let placement = output_placed.get(&bound.node).copied();
        let fault = encode_op(
            &device,
            &encoder,
            &mut device_buffers,
            bound,
            packed_operands,
            placement,
        )?;
        if let Some((fault_buffer, gathers)) = fault {
            pending_faults.push((bound, fault_buffer, gathers));
        }
        // explicit liveness exclusion (see this function's doc): a placed
        // node, input or output, is externally owned and always live, so it
        // is never dropped from this call's own bookkeeping map, regardless
        // of what `prepared.retires` (a per-program-only liveness sweep)
        // says.
        for retired in &prepared.retires[position] {
            if input_placed.contains_key(retired) || output_placed.contains_key(retired) {
                continue;
            }
            device_buffers.remove(retired);
        }
    }
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    for (bound, fault_buffer, gathers) in &pending_faults {
        check_gather_fault(bound, fault_buffer, *gathers)?;
    }

    finish(
        &plan.program,
        &prepared.index_nodes,
        &prepared.shapes,
        &prepared.effective_outputs,
        &device_buffers,
        prepared.root,
    )
}

/// [`plan`] against a name-keyed block set — the shape a model binds its
/// weights in. Resolution goes through
/// [`proxima_tensor::resolve_named_blocks`], the same function the CPU
/// evaluator uses, so the two backends cannot disagree about which name is
/// which position.
///
/// # Errors
/// Propagates name-resolution and planning failures.
pub fn plan_named(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
) -> Result<Plan, MetalError> {
    let blocks = resolve_named_blocks(program, named)?;
    plan(program, symbols, &blocks, outputs)
}

/// [`execute_plan`] against a name-keyed block set. The plan owns its
/// program, so the caller hands over only the per-call data.
///
/// # Errors
/// Propagates name-resolution and Metal driver failures.
pub fn execute_plan_named(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
) -> Result<Evaluated, MetalError> {
    let blocks = resolve_named_blocks(&plan.program, named)?;
    execute_plan(plan, &blocks)
}

/// [`execute_plan_with_placements`] against a name-keyed block set -- the
/// name-resolving sibling of [`execute_plan_named`], mirroring how that
/// function wraps [`execute_plan`]. `blocks` (positional, for the per-node
/// upload path) is resolved once via [`resolve_named_blocks`], then handed
/// to `execute_plan_with_placements` unchanged; input-placed nodes skip that
/// resolved entry's upload just as they do in the positional call.
///
/// # Errors
/// Propagates name-resolution and Metal driver failures.
#[cfg(feature = "metal-output-placement")]
pub fn execute_plan_named_with_placements(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
) -> Result<Evaluated, MetalError> {
    let blocks = resolve_named_blocks(&plan.program, named)?;
    execute_plan_with_placements(plan, &blocks, input_placements, output_placements)
}

/// One [`BoundOp`]'s GPU-only execution time and the operand bytes it read,
/// as gathered by [`execute_plan_op_timed`]. `weight_name` is
/// [`Op::name`](proxima_tensor::Op::name) off whichever operand is a named
/// block input (a model weight or the `ids`/`eps`/`rope_*`/KV-cache block
/// this program declares by name) -- `None` when every operand is itself a
/// computed node, which is the common case for a chained elementwise op.
#[cfg(feature = "instrument")]
#[derive(Debug, Clone)]
pub struct OpGpuTiming {
    pub node: NodeId,
    pub kind: &'static str,
    pub operand_bytes: u64,
    pub gpu_ns: u64,
    pub weight_name: Option<String>,
    /// `bound.operands().len()` -- surfaced so a diagnostic caller can tell
    /// a two-operand (weight, activation) reduce, the shape
    /// `crate::msl::packed_row_block` requires, from a fused reduce whose
    /// element body absorbed a third operand (e.g. a gate/up product ahead
    /// of a down-projection), which disqualifies the row-blocked kernel via
    /// that same function's `quantized.len() != 2` check.
    pub operand_count: usize,
    /// The packed codec carried by this op's named operand, when present.
    pub packed_codec: Option<PackedCodec>,
    /// The emitted packed-kernel body, kept separate from the broad op kind.
    pub packed_kernel_variant: &'static str,
    /// [`crate::msl::diagnose_packed_row_block`]'s own verdict on THIS op,
    /// against a REAL bound program rather than a synthetic symbolic one --
    /// `None` when the op is not a `Reduce { keep: Keep::Reduce, .. }` at
    /// all (the diagnosis does not apply), `Some("PASS")` when it took the
    /// row-blocked path, `Some(<rejection debug>)` otherwise. Exists
    /// because a synthetic probe's rejection table and a real production
    /// run's own `classify_kind` bucket disagreed on `ffn_down`/
    /// `output.weight` -- printing this against the REAL bound op is what
    /// settles which one was wrong, rather than trusting either by
    /// inference.
    pub packed_row_block_rejection: Option<String>,
}

/// Diagnostic-only counterpart of [`execute_plan`]: instead of sharing ONE
/// command buffer across the whole program and `commit`/`waitUntilCompleted`ing
/// it exactly once (see the module doc's "Execution model"), this commits
/// and waits on its OWN command buffer per [`BoundOp`], so each op's
/// `GPUStartTime`/`GPUEndTime` -- `MTLCommandBuffer`'s own documented
/// per-buffer GPU occupancy, not a CPU-side measurement -- can be read back
/// individually. It exists to answer exactly one question: does GPU time
/// track operand bytes, or is it flat per dispatch regardless of size. That
/// answer costs exactly what [`execute_plan`]'s own module doc says ROW 73
/// measured and removed -- one submission-boundary intercept per split, now
/// paid `prepared.resolved.len()` times instead of once -- which is why this
/// function is reachable only behind the `instrument` feature, from this
/// crate's own diagnostic call sites, and never from the serving loop
/// ([`execute_plan`]/[`execute_plan_named`] remain the only production
/// entry points and are completely unchanged by this function's existence).
///
/// # Errors
/// Same as [`execute_plan`].
#[cfg(feature = "instrument")]
pub fn execute_plan_op_timed(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
) -> Result<(Evaluated, Vec<OpGpuTiming>), MetalError> {
    let prepared = &plan.prepared;
    let packed_operands = &plan.packed_operands;

    let (device, queue) = device_and_queue()?;

    let mut device_buffers: BTreeMap<NodeId, DeviceBuffer> = BTreeMap::new();
    for ((node, block), dtype) in prepared
        .block_nodes
        .iter()
        .zip(blocks.iter())
        .zip(plan.block_dtypes.iter())
    {
        let resident = plan.resident_nodes.contains(node);
        let buffer = match block {
            QuantizedBlock::Float32(data) => upload_block(&device, data, *node, *dtype, resident)?,
            // `Float16`/`BFloat16` upload their bytes UNCHANGED, same as
            // every packed codec above -- there is no host-side narrowing
            // step (unlike `upload_block`'s `Float32 -> Float16` path,
            // which narrows a caller's `&[f32]`): a `Float16` weight's on-
            // disk bytes already ARE its device buffer's bytes (native
            // `half`), and a `BFloat16` weight's bytes are widened entirely
            // on the GPU at the read (`msl::BF16_UNPACK_MSL`), never on the
            // host.
            QuantizedBlock::Q4K(bytes)
            | QuantizedBlock::Q5K(bytes)
            | QuantizedBlock::Q6K(bytes)
            | QuantizedBlock::Q8_0(bytes)
            | QuantizedBlock::Q4_0(bytes)
            | QuantizedBlock::Float16(bytes)
            | QuantizedBlock::BFloat16(bytes) => upload_packed_bytes(&device, bytes, resident)?,
        };
        device_buffers.insert(*node, buffer);
    }

    let mut timings: Vec<OpGpuTiming> = Vec::with_capacity(prepared.resolved.len());
    for (position, bound) in prepared.resolved.iter().enumerate() {
        let operand_bytes: u64 = bound
            .operands()
            .iter()
            .map(|(source, _, _)| {
                device_buffers
                    .get(source)
                    .map(|(buffer, _offset)| buffer.length() as u64)
                    .unwrap_or(0)
            })
            .sum();
        let weight_name = bound
            .operands()
            .iter()
            .find_map(|(source, _, _)| plan.program[source.0 as usize].name())
            .map(ToString::to_string);
        let kind = classify_kind(bound, packed_operands);
        let packed_kernel_variant = classify_packed_kernel_variant(bound, packed_operands);
        let packed_row_block_rejection = diagnose_kind(bound, packed_operands);
        let packed_codec = bound
            .operands()
            .iter()
            .find_map(|(source, _, _)| packed_operands.get(source).copied());

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
        let fault = encode_op(
            &device,
            &encoder,
            &mut device_buffers,
            bound,
            packed_operands,
            None,
        )?;
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let gpu_ns =
            ((command_buffer.GPUEndTime() - command_buffer.GPUStartTime()) * 1e9).max(0.0) as u64;
        if let Some((fault_buffer, gathers)) = fault {
            check_gather_fault(bound, &fault_buffer, gathers)?;
        }
        for retired in &prepared.retires[position] {
            device_buffers.remove(retired);
        }
        timings.push(OpGpuTiming {
            node: bound.node,
            kind,
            operand_bytes,
            gpu_ns,
            weight_name,
            operand_count: bound.operands().len(),
            packed_codec,
            packed_kernel_variant,
            packed_row_block_rejection,
        });
    }

    let evaluated = finish(
        &plan.program,
        &prepared.index_nodes,
        &prepared.shapes,
        &prepared.effective_outputs,
        &device_buffers,
        prepared.root,
    )?;
    Ok((evaluated, timings))
}

/// [`execute_plan_op_timed`] against a name-keyed block set, mirroring
/// [`execute_plan_named`]'s own name resolution.
///
/// # Errors
/// Propagates name-resolution and Metal driver failures.
#[cfg(feature = "instrument")]
pub fn execute_plan_named_op_timed(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
) -> Result<(Evaluated, Vec<OpGpuTiming>), MetalError> {
    let blocks = resolve_named_blocks(&plan.program, named)?;
    execute_plan_op_timed(plan, &blocks)
}

/// Classifies one [`BoundOp`]'s emitted kernel body the same way
/// `omega/examples/real_forward_packed_probe.rs` (discipline log ROW 85)
/// does: by grepping the SOURCE for the tiled-GEMM simdgroup-matrix call
/// site vs the row-blocked call site vs the generic per-element scalar call
/// site vs a SIMD cooperative combine, since [`crate::msl::emit`] is the one
/// place that decides which of the four a given `Reduce` gets and none of
/// that decision is exposed as its own accessor.
#[cfg(feature = "instrument")]
fn classify_kind(bound: &BoundOp, packed_operands: &PackedOperands) -> &'static str {
    match &bound.kind {
        BoundOpKind::CachedAttention { .. } => "cached-attention",
        BoundOpKind::Elementwise { .. } => "elementwise",
        BoundOpKind::Iota => "iota",
        BoundOpKind::Constant { .. } => "constant",
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => "scan",
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => match emit(bound, packed_operands) {
            // Checked BEFORE the row-blocked arm below: ROW 113's
            // weight-staging fix made `push_tiled_gemm_body` call
            // `q4k_run8`/`q4k_header_for` too (the same amortized decode
            // `push_packed_row_blocked_body` already used), so the row-blocked
            // arm's own source markers must include both packed bodies --
            // `simdgroup_multiply_accumulate` only ever appears in
            // [`crate::msl::push_tiled_gemm_body`]'s emitted source, so it is
            // the one marker that still disambiguates tiled from packed.
            Ok(kernel) if kernel.source.contains("simdgroup_multiply_accumulate") => {
                "reduce-tiled-gemm"
            }
            Ok(kernel)
                if kernel.source.contains("q4k_pair_dot(blk")
                    || kernel.source.contains("q4k_run8(blk")
                    || kernel.source.contains("q5k_value(blk")
                    || kernel.source.contains("q6k_value(blk") =>
            {
                "reduce-packed-row-blocked"
            }
            Ok(kernel)
                if kernel.source.contains("simd_sum(")
                    || kernel.source.contains("simd_max(")
                    || kernel.source.contains("simd_min(")
                    || kernel.source.contains("simd_product(") =>
            {
                "reduce-cooperative"
            }
            Ok(_) => "reduce-generic-scalar",
            Err(_) => "reduce-unclassified",
        },
    }
}

#[cfg(feature = "instrument")]
fn classify_packed_kernel_variant(
    bound: &BoundOp,
    packed_operands: &PackedOperands,
) -> &'static str {
    let Ok(kernel) = emit(bound, packed_operands) else {
        return "unclassified";
    };
    if kernel.source.contains("q4k_pair_dot(blk") {
        "q4k-paired"
    } else if kernel.source.contains("q4k_run8(blk") {
        "q4k-run8"
    } else if kernel.source.contains("q5k_value(blk") {
        "q5k-scalar"
    } else if kernel.source.contains("q6k_value(blk") {
        "q6k-scalar"
    } else {
        "other"
    }
}

/// [`diagnose_packed_row_block`]'s verdict for THIS bound op, against the
/// REAL production layout rather than a synthetic symbolic probe --
/// `None` for anything that is not a `Reduce { keep: Keep::Reduce, .. }`
/// (the row-blocked kernel does not apply). `Some("PASS")` means it took
/// (or would take) the row-blocked path; `Some(<debug of the rejection>)`
/// names the exact gate that rejected it.
#[cfg(feature = "instrument")]
fn diagnose_kind(bound: &BoundOp, packed_operands: &PackedOperands) -> Option<String> {
    if !matches!(
        bound.kind,
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            ..
        }
    ) {
        return None;
    }
    let quantized: Vec<Option<PackedCodec>> = bound
        .operands()
        .iter()
        .map(|(node, _, _)| packed_operands.get(node).copied())
        .collect();
    Some(match diagnose_packed_row_block(bound, &quantized) {
        Ok(()) => "PASS".to_string(),
        Err(rejection) => format!("{rejection:?}"),
    })
}

/// Everything [`execute`] needs before touching a device — the same
/// judgments [`proxima_tensor::cpu::evaluate`]'s own `prepare` makes, rebuilt
/// here over the public API since that one is private to `cpu.rs`.
struct Prepared {
    root: NodeId,
    shapes: Shapes,
    effective_outputs: Vec<NodeId>,
    block_nodes: Vec<NodeId>,
    resolved: Vec<BoundOp>,
    retires: Vec<Vec<NodeId>>,
    /// Every node referenced as a gather's `indices` anywhere in the
    /// program — see [`gpu_dtype`]'s doc for why upload/read-back both
    /// need this set alongside a node's own declared dtype.
    index_nodes: BTreeSet<NodeId>,
}

/// Element count of one bound block, whatever codec carries it. The CPU
/// evaluator's own block table is [`QuantizedBlock`]; this driver now takes
/// the identical type rather than an `&[&[f32]]` of its own, so the two
/// evaluators cannot drift on what a block IS. A packed codec's element
/// count is derived from its own block geometry, never from `data.len()` —
/// packed bytes and elements are not the same unit. Infallible now that
/// every `QuantizedBlock` variant has a real Metal path (`Float16`/
/// `BFloat16` were the last two arms that could still fail here); kept
/// returning a `Result` regardless, so a future codec added without an
/// entry here is still a typed error rather than a silent miscount.
fn block_element_count(block: &QuantizedBlock<'_>) -> Result<usize, MetalError> {
    match block {
        QuantizedBlock::Float32(data) => Ok(data.len()),
        // packed bytes and elements are NOT the same unit: a `Q4_K`
        // super-block is 144 bytes carrying 256 elements, so the count the
        // shape check compares against comes from block geometry, never
        // from `bytes.len()`.
        QuantizedBlock::Q4K(bytes) => {
            Ok((bytes.len() / crate::msl::Q4K_BLOCK_BYTES) * crate::msl::Q4K_BLOCK_ELEMENTS)
        }
        // `Q6_K`'s super-block is a different byte width (210, not 144) but
        // the SAME element count per super-block (256) as `Q4_K`/`Q5_K` —
        // see `crate::msl::Q4K_BLOCK_ELEMENTS`'s own doc.
        QuantizedBlock::Q6K(bytes) => {
            Ok((bytes.len() / crate::msl::Q6K_BLOCK_BYTES) * crate::msl::Q4K_BLOCK_ELEMENTS)
        }
        // `Q5_K`'s super-block is yet another byte width (176) over the
        // same 256-element count.
        QuantizedBlock::Q5K(bytes) => {
            Ok((bytes.len() / crate::msl::Q5K_BLOCK_BYTES) * crate::msl::Q4K_BLOCK_ELEMENTS)
        }
        // `Q8_0`'s block is a different shape entirely (34 bytes carrying
        // 32 elements, no super-block) — its own constants, never
        // `Q4K_BLOCK_ELEMENTS`.
        QuantizedBlock::Q8_0(bytes) => {
            Ok((bytes.len() / crate::msl::Q8_0_BLOCK_BYTES) * crate::msl::Q8_0_BLOCK_ELEMENTS)
        }
        // `Q4_0`'s block is the same flat shape as `Q8_0` (18 bytes
        // carrying 32 elements, no super-block) but a different byte
        // width -- its own constants, never `Q8_0_BLOCK_BYTES`.
        QuantizedBlock::Q4_0(bytes) => {
            Ok((bytes.len() / crate::msl::Q4_0_BLOCK_BYTES) * crate::msl::Q4_0_BLOCK_ELEMENTS)
        }
        // Half-precision weights are one element per block -- no
        // super-block to divide out, unlike every quantized codec above.
        QuantizedBlock::Float16(bytes) => Ok(
            (bytes.len() / crate::msl::FLOAT16_BLOCK_BYTES) * crate::msl::FLOAT16_BLOCK_ELEMENTS
        ),
        QuantizedBlock::BFloat16(bytes) => {
            Ok((bytes.len() / crate::msl::BFLOAT16_BLOCK_BYTES)
                * crate::msl::BFLOAT16_BLOCK_ELEMENTS)
        }
    }
}

/// Raw host bytes one [`QuantizedBlock`] hands [`upload_block`]/
/// [`upload_packed_bytes`] — the split-4019 "block upload" term's byte count,
/// distinct from [`block_element_count`]'s element count (a `Q4_K`
/// super-block's bytes and elements are not the same unit either).
#[cfg(feature = "instrument")]
fn block_byte_len(block: &QuantizedBlock<'_>) -> usize {
    match block {
        QuantizedBlock::Float32(data) => size_of_val(*data),
        QuantizedBlock::Q4K(bytes)
        | QuantizedBlock::Q5K(bytes)
        | QuantizedBlock::Q6K(bytes)
        | QuantizedBlock::Q8_0(bytes)
        | QuantizedBlock::Q4_0(bytes)
        | QuantizedBlock::Float16(bytes)
        | QuantizedBlock::BFloat16(bytes) => bytes.len(),
    }
}

fn prepare(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock<'_>],
    outputs: &[NodeId],
) -> Result<Prepared, MetalError> {
    let shapes = infer(program, symbols)?;
    let packed_operands = packed_operands_of(&block_node_ids(program), blocks);
    // every packed codec's declared dtype is the "these are bytes" marker
    // `reject_unsupported_gpu_dtype`'s own doc already claims as its
    // exemption's rationale -- not just the codecs `packed_operands` above
    // has a kernel for, so any future codec added to `QuantizedBlock` before
    // it has an unpack kernel here still gets the right dtype exemption
    // rather than an unrelated "not float" rejection.
    let packed_operand_nodes: BTreeSet<NodeId> = block_node_ids(program)
        .iter()
        .zip(blocks.iter())
        .filter(|(_, block)| !matches!(block, QuantizedBlock::Float32(_)))
        .map(|(node, _)| *node)
        .collect();
    reject_unsupported_gpu_dtype(program, &packed_operand_nodes)?;

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
        return Err(TensorError::InputCountMismatch {
            expected: block_nodes.len(),
            found: blocks.len(),
        }
        .into());
    }
    for (node, block) in block_nodes.iter().zip(blocks.iter()) {
        let expected = element_count(shapes.of(*node));
        let found = block_element_count(block)?;
        if found != expected {
            return Err(TensorError::InputSizeMismatch {
                node: *node,
                expected,
                found,
            }
            .into());
        }
    }

    let mut resolved = bind(program, &shapes, &effective_outputs)?;
    // A stateless driver has no persistent arena to skip a dead slot inside
    // between calls (unlike `proxima_tensor::cpu::StaticArena`'s own
    // execution-time skip set) -- the only way to avoid dispatching a kernel
    // nobody reads is to drop it from `resolved` before it ever reaches a
    // dispatch list. See `prune_dead`'s own doc.
    resolved = prune_dead(resolved, &effective_outputs);
    // `BoundOpBuilder::finish` (`proxima-tensor`'s `bind.rs`) flushes every
    // held elementwise op -- requested output or not -- at the very END of
    // the walk, ascending by `NodeId` among themselves, regardless of where
    // its dependencies or its own program position sit. That is correct for
    // a plain [`execute_plan`] call, where "when a value is computed" never
    // matters to a caller that only reads it back after the whole command
    // buffer completes. It is WRONG for a placed *output*
    // ([`execute_plan_with_placements`]'s own doc) this same call also
    // aliases as a placed *input* elsewhere in the SAME program (the
    // KV-cache shape that doc's "Within-call aliasing" section describes):
    // Metal's hazard tracking only orders two dispatches by ENCODE order, so
    // a deferred write encoded after a consumer that reads the SAME buffer
    // through its aliased `Op::Input` leaves that consumer reading stale
    // bytes. This promotes every `effective_outputs` node forward to right
    // after the latest position its own real operands already occupy -- the
    // position it would have held had `finish` never deferred it -- without
    // touching `BoundOpBuilder`'s own push/finish policy, which stays
    // correct for ordinary (non-aliased) outputs and is unchanged here. A
    // node this call means to output-place but never declared as an
    // `outputs` entry is invisible to this pass AND to `prune_dead` above --
    // see [`execute_plan_with_placements`]'s own doc for why every placed
    // output must be a declared output.
    promote_output_placed_nodes(&mut resolved, &effective_outputs);
    // `bind`'s own `layout_of` assumes every operand is stored row-major in
    // its DECLARED axis order -- true for every f32 buffer this driver reads
    // (bound-time-transposed to match, `bind_matmul_weight`'s own doc), but
    // never true for a packed `Q4_K`/`Q5_K`/`Q6_K` weight, whose bytes are GGUF's
    // native `[out, in]` regardless of what the declared shape says. Left
    // uncorrected, every quantized matmul reads its weight through the wrong
    // stride -- see `correct_packed_matmul_layouts`'s own doc (already
    // codec-agnostic: it takes any `packed_operands` node set).
    correct_packed_matmul_layouts(&mut resolved, &packed_operands.keys().copied().collect());
    let retires = node_retirement(&resolved, &effective_outputs);
    let index_nodes = index_node_ids(program);

    Ok(Prepared {
        root,
        shapes,
        effective_outputs,
        block_nodes,
        resolved,
        retires,
        index_nodes,
    })
}

// mirrors `proxima_tensor::cpu::reject_non_float32`'s exemption (a gather's
// `indices` node is the one deliberate exception, since an index value is
// an exact integer carried as f32 regardless of its own declared dtype) but
// this driver's own dtype ceiling is wider than the CPU oracle's: `Float32`
// or `Float16` may reach a device buffer, since `msl.rs` now emits a
// `half`-typed kernel for a `Float16` node instead of assuming `float`
// unconditionally (see `msl.rs`'s own dtype doc). A `BFloat16` node clears
// this gate only via the `packed_nodes` exemption below (it is never a bare
// `Float32`/`Float16` node itself) -- `packed_operands_of` puts every
// `BFloat16` block-input node into that set. Any other dtype is still
// rejected exactly as before.
fn reject_unsupported_gpu_dtype(
    program: &[Op],
    packed_nodes: &BTreeSet<NodeId>,
) -> Result<(), TensorError> {
    let index_nodes = index_node_ids(program);
    for (position, expr) in program.iter().enumerate() {
        let node = NodeId(position as u32);
        // a gather's indices are exempt (see this function's doc); so is a
        // packed quantized weight, whose declared dtype is the marker for
        // "these are bytes" and never the element type the kernel computes
        // in — the same exemption `cpu::reject_non_float32` makes via
        // `is_quantized_matmul_operand`.
        if index_nodes.contains(&node) || packed_nodes.contains(&node) {
            continue;
        }
        if !matches!(expr.dtype(), DType::Float32 | DType::Float16) {
            return Err(TensorError::NotLowerable {
                node,
                reason: "metal execution supports float32 or float16 in v1, \
                         except for a gather's indices",
            });
        }
    }
    Ok(())
}

/// The dtype `node`'s own device buffer marshals as: `Float32` when `node`
/// is a gather's `indices` (an index value is an exact integer carried as
/// f32 regardless of its own declared dtype — see
/// [`reject_unsupported_gpu_dtype`]'s doc), otherwise `node`'s own declared
/// dtype straight off the program. `BoundOp::dtype` already carries this
/// same value for a computed node (it is built from the identical `Op`),
/// so callers that already have a `BoundOp` in hand read `bound.dtype`
/// directly instead of calling this — this exists for the two places that
/// only have a bare `NodeId`: uploading a block input and reading back a
/// requested output, either of which may name a plain `Op::Input` node
/// this driver never resolves into a `BoundOp` at all.
fn gpu_dtype(program: &[Op], index_nodes: &BTreeSet<NodeId>, node: NodeId) -> DType {
    if index_nodes.contains(&node) {
        DType::Float32
    } else {
        program[node.0 as usize].dtype()
    }
}

fn element_count(shape: &[u64]) -> usize {
    shape.iter().product::<u64>() as usize
}

/// Moves every node named in `effective_outputs` to just after the latest
/// position its own operands already occupy in `resolved` -- see this
/// function's own call site in [`prepare`] for why. A no-op for any node
/// already there (every `Reduce` root: `BoundOpBuilder::push` emits those
/// immediately, never defers them).
///
/// A node with NO operand present in `resolved` at all (every real operand
/// is an `Op::Input` leaf) is left exactly where `finish` put it, not
/// pulled to the front: this function only sees the declared tensor-graph
/// operands, never the OTHER kind of ordering constraint this whole module
/// exists for -- one node's placed OUTPUT and a different node's placed
/// INPUT aliasing the SAME `PlacedBuffer`, which is invisible to
/// `proxima-tensor`'s graph entirely (the caller supplies that aliasing at
/// execute time, not plan time). A zero-real-operand output can be the
/// aliased READER half of exactly that pair
/// (`omega/tests/metal_output_placement.rs`'s
/// `a_program_reads_a_placed_write_from_a_later_op_in_the_same_call` is
/// this shape), and moving it to position `0` would place it BEFORE the
/// write it depends on for correctness, the same bug class this function
/// exists to fix, just on the other node. Leaving it at `finish`'s own
/// position is always safe for this shape: every node `finish` flushes is
/// already in ascending `NodeId` order relative to every OTHER flushed
/// node (its own doc), so two zero-real-operand outputs still come out in
/// their aliasing-write-then-read order as long as neither is promoted.
///
/// `resolved` is already a valid topological order before this call
/// (`BoundOpBuilder`'s own invariant: a reference only ever points
/// backwards), so every candidate with a real dependency has a target
/// position provably `<=` its current one -- this only ever pulls a node
/// EARLIER, never later, so it cannot itself introduce a forward reference.
fn promote_output_placed_nodes(resolved: &mut Vec<BoundOp>, effective_outputs: &[NodeId]) {
    let outputs: BTreeSet<NodeId> = effective_outputs.iter().copied().collect();
    if outputs.is_empty() {
        return;
    }
    let candidates: Vec<NodeId> = resolved
        .iter()
        .filter(|bound| outputs.contains(&bound.node))
        .map(|bound| bound.node)
        .collect();

    for node in candidates {
        // Recomputed fresh every iteration, not cached across the loop: an
        // earlier candidate's own move can shift this node's (and its
        // operands') position by one, and `resolved`'s exact current order
        // is the only thing this function is allowed to trust.
        let index_of: BTreeMap<NodeId, usize> = resolved
            .iter()
            .enumerate()
            .map(|(index, bound)| (bound.node, index))
            .collect();
        let Some(current_index) = index_of.get(&node).copied() else {
            continue;
        };
        let dependency_index = resolved[current_index]
            .operands()
            .iter()
            .filter_map(|(source, _, gather)| {
                let mut latest = index_of.get(source).copied();
                if let Some(lookup) = gather {
                    latest = latest.max(index_of.get(&lookup.indices).copied());
                }
                latest
            })
            .max();
        let Some(dependency) = dependency_index else {
            continue;
        };
        let target_index = dependency + 1;
        if target_index < current_index {
            let bound = resolved.remove(current_index);
            resolved.insert(target_index, bound);
        }
    }
}

/// The output length an op needs allocated: the reduced (product of
/// surviving axes) length for a `Keep::Reduce` reduce, or the full
/// iteration space otherwise (elementwise and `Keep::Scan` both write one
/// value per coordinate). Deliberately independent of [`Kernel::grid`]'s
/// thread count — a `Keep::Scan` scan dispatches one thread per *line* but
/// writes `inner_len` values per thread, so grid threads and output length
/// diverge there.
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

fn push_i64(bytes: &mut Vec<u8>, value: i64) {
    bytes.extend_from_slice(&value.to_ne_bytes());
}

/// Pushes `values` as a fixed-`width` MSL array, zero-padding any slot
/// `values` does not fill — the only case that happens is a rank-0 op,
/// where the declared array width is `max(rank, 1)` but there is no real
/// axis to supply, and that padding slot is never read by the generated
/// source (see each `render_*`'s `if rank > 0` / `.saturating_sub(1)` guards).
fn push_i64_row(bytes: &mut Vec<u8>, values: &[i64], width: usize) {
    for slot in 0..width {
        push_i64(bytes, values.get(slot).copied().unwrap_or(0));
    }
}

/// Appends the four gather arrays every `Uniforms` struct declares last (via
/// `crate::msl::push_gather_uniform_fields`) when `bound` has at least one
/// gathered operand: `gather_index_base`, `gather_index_strides`,
/// `gather_element_stride`, `gather_extent` — each one array of length
/// `gather_count`. `crate::msl::gather_slots` numbers gathered operands by
/// encounter order over `bound.operands()`, so filtering in that same order
/// (below) reproduces the identical numbering without needing to re-derive
/// or look up the slot indices themselves. A no-op when `bound` has no
/// gather, matching `push_gather_uniform_fields`'s own empty-array early
/// return.
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
        push_i64(bytes, gather.index_layout.base);
    }
    for gather in &ordered {
        push_i64_row(bytes, &gather.index_layout.strides, rank_len);
    }
    for gather in &ordered {
        push_i64(bytes, gather.element_stride);
    }
    for gather in &ordered {
        push_i64(bytes, gather.extent as i64);
    }
}

fn pack_uniforms(bound: &BoundOp) -> Vec<u8> {
    match &bound.kind {
        BoundOpKind::CachedAttention { .. } => pack_cached_attention_uniforms(bound),
        BoundOpKind::Elementwise { .. } => pack_elementwise_uniforms(bound),
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => pack_reduce_uniforms(bound),
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => pack_scan_uniforms(bound),
        BoundOpKind::Iota | BoundOpKind::Constant { .. } => pack_leaf_uniforms(bound),
    }
}

fn pack_cached_attention_uniforms(bound: &BoundOp) -> Vec<u8> {
    let BoundOpKind::CachedAttention { head_dim, .. } = &bound.kind else {
        unreachable!("cached attention uniform packer only receives cached attention")
    };
    let total: i64 = bound.extents.iter().map(|extent| *extent as i64).product::<i64>()
        / *head_dim as i64;
    let mut bytes = Vec::new();
    push_i64(&mut bytes, total);
    bytes
}

/// Mirrors the `Uniforms` struct `crate::msl::render_iota` and
/// `crate::msl::render_constant` both declare: just `total_elements` —
/// neither leaf has operands, a per-axis extents array, or a gather, so
/// there is nothing else this struct needs to carry. `render_constant`
/// bakes its literal into the source instead of adding a field here, which
/// is what lets one packer serve both.
fn pack_leaf_uniforms(bound: &BoundOp) -> Vec<u8> {
    let total: i64 = bound.extents.iter().map(|extent| *extent as i64).product();
    let mut bytes = Vec::new();
    push_i64(&mut bytes, total);
    bytes
}

/// Mirrors the `Uniforms` struct `crate::msl::render_elementwise` declares
/// at `omega/src/msl.rs:328-335`: `total_elements`, `extents[rank_len]`,
/// `operand_base[operand_count]`, `operand_strides[operand_count][rank_len]`,
/// then — only when `bound` has a gathered operand — the four
/// `push_gather_uniform_fields` arrays [`push_gather_uniforms`] appends, in
/// that order — every field `long`, so a flat `i64` concatenation is the
/// struct's byte layout.
fn pack_elementwise_uniforms(bound: &BoundOp) -> Vec<u8> {
    let rank_len = bound.extents.len().max(1);
    let extents: Vec<i64> = bound.extents.iter().map(|extent| *extent as i64).collect();

    let mut bytes = Vec::new();
    push_i64(&mut bytes, extents.iter().product());
    push_i64_row(&mut bytes, &extents, rank_len);
    for (_, layout, _) in bound.operands() {
        push_i64(&mut bytes, layout.base);
    }
    for (_, layout, _) in bound.operands() {
        push_i64_row(&mut bytes, &layout.strides, rank_len);
    }
    push_gather_uniforms(&mut bytes, bound, rank_len);
    bytes
}

/// Mirrors the `Uniforms` struct `crate::msl::render_reduce` declares at
/// `omega/src/msl.rs:386-397`: `output_total`, `reduction_total`,
/// `output_extents[output_rank_len]`, `reduction_extents[reduce_rank_len]`,
/// `operand_base[operand_count]`,
/// `operand_strides[operand_count][rank_len]`, `out_base`,
/// `out_strides[rank_len]`, then the gather arrays (see
/// [`pack_elementwise_uniforms`]'s doc), in that order.
fn pack_reduce_uniforms(bound: &BoundOp) -> Vec<u8> {
    let BoundOpKind::Reduce {
        output_axes,
        out_layout,
        ..
    } = &bound.kind
    else {
        unreachable!("pack_reduce_uniforms is only called for a Keep::Reduce reduce")
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
    push_i64(&mut bytes, output_extents.iter().product());
    push_i64(&mut bytes, reduction_extents.iter().product());
    push_i64_row(&mut bytes, &output_extents, output_rank_len);
    push_i64_row(&mut bytes, &reduction_extents, reduce_rank_len);
    for (_, layout, _) in bound.operands() {
        push_i64(&mut bytes, layout.base);
    }
    for (_, layout, _) in bound.operands() {
        push_i64_row(&mut bytes, &layout.strides, rank_len);
    }
    push_i64(&mut bytes, out_layout.base);
    push_i64_row(&mut bytes, &out_layout.strides, rank_len);
    push_gather_uniforms(&mut bytes, bound, rank_len);
    bytes
}

/// Mirrors the `Uniforms` struct `crate::msl::render_scan` declares at
/// `omega/src/msl.rs:493-503`: `outer_total`, `inner_len`,
/// `outer_extents[outer_rank_len]`, `operand_base[operand_count]`,
/// `operand_strides[operand_count][rank_len]`, `out_base`,
/// `out_strides[rank_len]`, then the gather arrays (see
/// [`pack_elementwise_uniforms`]'s doc), in that order. `crate::msl::validate`
/// already rejected a rank-0 scan before `emit` (and therefore this) ever
/// runs, so `bound.extents` is never empty here.
fn pack_scan_uniforms(bound: &BoundOp) -> Vec<u8> {
    let BoundOpKind::Reduce { out_layout, .. } = &bound.kind else {
        unreachable!("pack_scan_uniforms is only called for a Keep::Scan reduce")
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
    push_i64(&mut bytes, outer_extents.iter().product());
    push_i64(&mut bytes, inner_len);
    push_i64_row(&mut bytes, &outer_extents, outer_rank_len);
    for (_, layout, _) in bound.operands() {
        push_i64(&mut bytes, layout.base);
    }
    for (_, layout, _) in bound.operands() {
        push_i64_row(&mut bytes, &layout.strides, rank_len);
    }
    push_i64(&mut bytes, out_layout.base);
    push_i64_row(&mut bytes, &out_layout.strides, rank_len);
    push_gather_uniforms(&mut bytes, bound, rank_len);
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

/// Resolves `bound`'s compiled pipeline against `cache_key`
/// ([`kernel_cache_key`]) rather than [`Kernel::source`] — a hit never builds
/// the MSL source text at all, only a genuine miss calls [`emit`] to render
/// it and compile. See this module's `ROW 92`/`ROW 93` discipline-log entries
/// for the measured cost `emit` paid on every call, hit or miss, before this
/// split existed.
fn pipeline_for(
    device: &ProtocolObject<dyn MTLDevice>,
    bound: &BoundOp,
    packed_operands: &PackedOperands,
    cache_key: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
    if let Some(pipeline) = PIPELINE_CACHE.with(|cache| cache.borrow().get(cache_key).cloned()) {
        #[cfg(feature = "instrument")]
        counter!(PIPELINE_HITS, 1);
        return Ok(pipeline);
    }
    #[cfg(feature = "instrument")]
    let compile_started = read_ticks();
    let kernel = emit(bound, packed_operands)?;
    let pipeline = compile_pipeline(device, &kernel)?;
    #[cfg(feature = "instrument")]
    {
        counter!(PIPELINE_MISSES, 1);
        counter!(PIPELINE_COMPILE_TICKS, elapsed_ticks(compile_started));
    }
    PIPELINE_CACHE.with(|cache| {
        cache
            .borrow_mut()
            .insert(cache_key.to_string(), pipeline.clone());
    });
    Ok(pipeline)
}

#[cfg(not(feature = "metal-buffer-pool"))]
fn allocate_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    element_count: usize,
    dtype: DType,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let byte_length = element_count.max(1) * dtype.size_bytes();
    device
        .newBufferWithLength_options(byte_length, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| MetalError::CompileFailed {
            log: "device refused to allocate a shared buffer".to_string(),
        })
}

/// Pool-backed counterpart of the `metal-buffer-pool`-off `allocate_buffer`
/// above -- identical contract (a buffer sized to hold `element_count`
/// `dtype` elements), but the ACTUAL Metal allocation, on both the pool-hit
/// and pool-miss paths, is always sized to `pool_bucket(byte_length)`, never
/// to the tight `byte_length`. See `pool_bucket`'s doc for why that is what
/// makes a bucketed pop sound, and `OUTPUT_BUFFER_POOL`'s doc for the
/// invariant this maintains across the pool's whole lifetime.
#[cfg(feature = "metal-buffer-pool")]
fn allocate_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    element_count: usize,
    dtype: DType,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let byte_length = element_count.max(1) * dtype.size_bytes();
    let bucket = pool_bucket(byte_length);
    let pooled = OUTPUT_BUFFER_POOL.with(|pool| {
        pool.borrow_mut()
            .get_mut(&(bucket, dtype))
            .and_then(Vec::pop)
    });
    if let Some(buffer) = pooled {
        counter!(OUTPUT_BUFFER_POOL_REUSES, 1);
        return Ok(buffer);
    }
    device
        .newBufferWithLength_options(bucket, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| MetalError::CompileFailed {
            log: "device refused to allocate a shared buffer".to_string(),
        })
}

/// Rounds `byte_length` up to the next power of two -- the pool's bucket
/// function. A pool entry's KEY is always this bucket, and by
/// [`allocate_buffer`]'s own invariant (every fresh allocation under this
/// feature is sized to exactly `pool_bucket(request)` bytes, never to the
/// tight request), a buffer stored under bucket `B` always has REAL Metal
/// capacity `>= B` -- in fact exactly `B`, since nothing ever shrinks a
/// buffer once allocated. So a pop from bucket `B` is sound for any request
/// whose own bucket is `B`, i.e. any request in `(B/2, B]` bytes: never
/// smaller than what was asked for.
#[cfg(feature = "metal-buffer-pool")]
fn pool_bucket(byte_length: usize) -> usize {
    byte_length.next_power_of_two()
}

/// Per-bucket cap on retained buffers: [`execute_plan`]'s reclaim drain drops
/// (never pools) any buffer beyond this many already resident in its
/// `(bucket, dtype)` slot. Population is already naturally bounded by the
/// program's peak CONCURRENT live-output count at that bucket -- steady
/// decode does not exceed a handful -- so this is a safety net against a
/// pathological program shape (many parallel same-size branches), not the
/// primary bound. `8` is a plain constant here, not yet wired through the
/// project's build-time sizing-config mechanism (see the guiding-principles
/// "no magic numbers" rule) -- a gap named explicitly, not hidden, and one
/// more reason this feature is not yet a default-on candidate.
#[cfg(feature = "metal-buffer-pool")]
const OUTPUT_POOL_MAX_PER_BUCKET: usize = 8;

#[cfg(feature = "metal-buffer-pool")]
thread_local! {
    /// Op-OUTPUT device buffers, reused across [`execute_plan`] calls instead
    /// of a fresh `newBufferWithLength_options` per op per token
    /// (`allocate_buffer` pops from here first). Keyed on `(bucket, dtype)`
    /// where `bucket` is [`pool_bucket`]'s power-of-two rounding of the
    /// requested byte length -- NOT the tight byte length itself. See
    /// `pool_bucket`'s doc for the size invariant this relies on: every
    /// buffer stored under bucket `B` has REAL Metal capacity exactly `B`,
    /// because [`allocate_buffer`] allocates fresh buffers at `B` too, never
    /// at the tight request. `dtype` stays part of the key alongside the
    /// bucket so the guarantee reads directly off the type at every call
    /// site.
    ///
    /// # Why bucketing (not exact-size keying)
    ///
    /// Decode's cached-attention extent grows every token, so an op whose
    /// output size tracks it (`bound_output_len` scales with the KV extent)
    /// would get a NEW exact key every token under exact-size keying, which
    /// (a) never gets reused again -- an unbounded, one-orphaned-buffer-per-
    /// token leak over a long decode session -- and (b) MISSES every time on
    /// exactly those ops, so they keep paying `newBufferWithLength` per token
    /// regardless. Bucketing by power of two means a growing extent walks
    /// through a BOUNDED number of buckets (`log2(max_extent_bytes)`, not one
    /// per token) and hits the bucket it shared with the previous token's
    /// request whenever the two requests round to the same power of two.
    ///
    /// # Liveness: why this cannot alias two live tensors
    ///
    /// A buffer is pushed back here ONLY by [`execute_plan`], and only after
    /// that call's single `command_buffer.waitUntilCompleted()` has returned
    /// -- i.e. after every dispatch that could read or write it has finished
    /// running on the GPU. Nothing is ever returned mid-program:
    /// `prepared.retires` still drives `device_buffers` removal exactly as
    /// before this feature existed (see the `metal-buffer-pool`-off arm right
    /// below the retirement loop in `execute_plan`), the retired buffer is
    /// simply ALSO cloned into a same-call `reclaim_stash` rather than only
    /// dropped, and that stash is not drained into this pool until after the
    /// wait. So within one `execute_plan` call, every buffer this pool hands
    /// out via `allocate_buffer` was either freshly allocated or was a buffer
    /// whose prior GPU work is already complete -- it can never be a buffer
    /// some still-pending dispatch in the SAME command buffer is about to
    /// read or write, because nothing enters this pool until that command
    /// buffer no longer exists to have pending dispatches. This is the "pool
    /// only across calls" fallback, chosen over intra-program reuse (which
    /// would need to reason about `MTLDispatchTypeSerial` hazard-tracked
    /// ordering across a retired buffer's last read and a later op's first
    /// write) to keep the liveness argument this simple.
    ///
    /// # Never touches block-input caching
    ///
    /// Only buffers `encode_op` allocated for an op's OUTPUT are ever pushed
    /// here -- `execute_plan` filters by `output_meta`, built from
    /// `prepared.resolved`'s own node ids, before cloning anything into
    /// `reclaim_stash`. `NOCOPY_BUFFERS` and the resident-copy cache (block
    /// INPUTS, wrapping the caller's own weight memory) are untouched by this
    /// feature; a block-input buffer removed by the same retirement loop is
    /// dropped exactly as it always was, never reaching this map.
    ///
    /// # An over-sized buffer is only ever consulted through the bound op's
    /// own shape, never through its own `.length()`
    ///
    /// The one hazard bucketing introduces: a popped buffer can be LARGER
    /// than the op that requested it strictly needs. Every dispatch-affecting
    /// consumer -- `dispatch`, `bind_buffers`, `read_back` -- sizes its work
    /// from the bound op's own shape/uniforms, never from buffer capacity, so
    /// bucketing carries no dispatch/readback correctness hazard.
    ///
    /// # Bounded, not unbounded: worst-case memory overhead
    ///
    /// A power-of-two bucket wastes strictly less than 2x the tight request
    /// (a request just over `B/2` rounds up to `B`, the worst case; a request
    /// of exactly a power of two wastes nothing). DERIVED, not measured.
    /// [`OUTPUT_POOL_MAX_PER_BUCKET`] caps retained-buffer growth per bucket
    /// on top of that.
    static OUTPUT_BUFFER_POOL: RefCell<HashMap<(usize, DType), Vec<MetalBuffer>>> =
        RefCell::new(HashMap::new());
}

/// Counts op-output buffers served from `OUTPUT_BUFFER_POOL` rather than
/// freshly allocated.
#[cfg(feature = "metal-buffer-pool")]
pub static OUTPUT_BUFFER_POOL_REUSES: Counter = Counter::new("omega.metal.output_pool.reuse");

/// Total buffers currently retained across every `(bucket, dtype)` slot in
/// `OUTPUT_BUFFER_POOL` on THIS thread. Test/diagnostic surface: proves
/// bucketing keeps the pool's retained-buffer count BOUNDED across many
/// distinct historical output sizes (a growing cached-attention extent)
/// rather than growing once per distinct size an exact-size-keyed pool
/// would.
#[cfg(feature = "metal-buffer-pool")]
#[must_use]
pub fn output_buffer_pool_len() -> usize {
    OUTPUT_BUFFER_POOL.with(|pool| pool.borrow().values().map(Vec::len).sum())
}

/// split-4019 per-token attribution counters — each is a (`_CALLS`, `_TICKS`)
/// pair over one named stage of [`execute_plan`], read back via `.get()`
/// (cumulative) or `.snapshot_and_reset()` (per-token delta) by a caller that
/// wants to attribute wall clock rather than assume it. Every wrap site notes
/// which term of the split-4019 table it feeds.
#[cfg(feature = "instrument")]
pub static PREPARE_CALLS: Counter = Counter::new("omega.metal.prepare_calls");
#[cfg(feature = "instrument")]
pub static PREPARE_TICKS: Counter = Counter::new("omega.metal.prepare_ticks");
#[cfg(feature = "instrument")]
pub static EMIT_CALLS: Counter = Counter::new("omega.metal.emit_calls");
#[cfg(feature = "instrument")]
pub static EMIT_TICKS: Counter = Counter::new("omega.metal.emit_ticks");
#[cfg(feature = "instrument")]
pub static PIPELINE_HITS: Counter = Counter::new("omega.metal.pipeline_hits");
#[cfg(feature = "instrument")]
pub static PIPELINE_MISSES: Counter = Counter::new("omega.metal.pipeline_misses");
#[cfg(feature = "instrument")]
pub static PIPELINE_COMPILE_TICKS: Counter = Counter::new("omega.metal.pipeline_compile_ticks");
#[cfg(feature = "instrument")]
pub static BLOCK_UPLOAD_CALLS: Counter = Counter::new("omega.metal.block_upload_calls");
#[cfg(feature = "instrument")]
pub static BLOCK_UPLOAD_TICKS: Counter = Counter::new("omega.metal.block_upload_ticks");
#[cfg(feature = "instrument")]
pub static BLOCK_UPLOAD_BYTES: Counter = Counter::new("omega.metal.block_upload_bytes");
#[cfg(feature = "instrument")]
pub static OP_SETUP_CALLS: Counter = Counter::new("omega.metal.op_setup_calls");
#[cfg(feature = "instrument")]
pub static OP_SETUP_TICKS: Counter = Counter::new("omega.metal.op_setup_ticks");
/// ROW 93's split of ROW 92's "inside-backend residual": the whole
/// `pipeline_for` call (cache lookup on a hit, lookup+compile on a miss),
/// distinct from `EMIT_TICKS` (now the cheap `kernel_cache_key`/
/// `kernel_dispatch_shape` pair, no MSL text) and from
/// `PIPELINE_COMPILE_TICKS` (the compile-only sub-span that fires on a miss).
#[cfg(feature = "instrument")]
pub static PIPELINE_LOOKUP_CALLS: Counter = Counter::new("omega.metal.pipeline_lookup_calls");
#[cfg(feature = "instrument")]
pub static PIPELINE_LOOKUP_TICKS: Counter = Counter::new("omega.metal.pipeline_lookup_ticks");
/// ROW 93's split of ROW 92's residual, part two: `setComputePipelineState`
/// + `bind_buffers` + `dispatch`, `encode_op`'s three remaining calls.
#[cfg(feature = "instrument")]
pub static ENCODE_DISPATCH_CALLS: Counter = Counter::new("omega.metal.encode_dispatch_calls");
#[cfg(feature = "instrument")]
pub static ENCODE_DISPATCH_TICKS: Counter = Counter::new("omega.metal.encode_dispatch_ticks");
#[cfg(feature = "instrument")]
pub static GPU_EXEC_CALLS: Counter = Counter::new("omega.metal.gpu_exec_calls");
#[cfg(feature = "instrument")]
pub static GPU_EXEC_TICKS: Counter = Counter::new("omega.metal.gpu_exec_ticks");
#[cfg(feature = "instrument")]
pub static READBACK_CALLS: Counter = Counter::new("omega.metal.readback_calls");
#[cfg(feature = "instrument")]
pub static READBACK_TICKS: Counter = Counter::new("omega.metal.readback_ticks");
#[cfg(feature = "instrument")]
pub static READBACK_BYTES: Counter = Counter::new("omega.metal.readback_bytes");

/// One [`execute_plan`] call's worth of the split-4019 counters above,
/// snapshot-and-reset so a caller (the metal decode test) can read a
/// PER-TOKEN delta rather than a cumulative mean over the whole run —
/// guiding-principle 19's "per-token records, not just a mean".
#[cfg(feature = "instrument")]
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalStageTotals {
    pub prepare_calls: u64,
    pub prepare_ticks: u64,
    pub emit_calls: u64,
    pub emit_ticks: u64,
    pub pipeline_hits: u64,
    pub pipeline_misses: u64,
    pub pipeline_compile_ticks: u64,
    pub block_upload_calls: u64,
    pub block_upload_ticks: u64,
    pub block_upload_bytes: u64,
    pub op_setup_calls: u64,
    pub op_setup_ticks: u64,
    pub pipeline_lookup_calls: u64,
    pub pipeline_lookup_ticks: u64,
    pub encode_dispatch_calls: u64,
    pub encode_dispatch_ticks: u64,
    pub gpu_exec_calls: u64,
    pub gpu_exec_ticks: u64,
    pub readback_calls: u64,
    pub readback_ticks: u64,
    pub readback_bytes: u64,
    /// Split of `block_upload_calls` above by host->device path — reads back
    /// the ALREADY-EXISTING [`NOCOPY_BUFFER_UPLOADS`]/[`COPYING_BUFFER_UPLOADS`]/
    /// [`NOCOPY_BUFFER_REUSES`] counters (see this module's "Host buffer
    /// upload" doc) rather than adding parallel ones, so a caller can tell
    /// whether the 380+ ms/token `block_upload_ticks` figure is genuine
    /// no-copy cache misses (a real `newBufferWithBytes*` driver call) or a
    /// growing count of small COPYING allocations (the KV cache's own
    /// re-`Vec`-allocated-every-append pointer, which can never take the
    /// no-copy path).
    pub nocopy_uploads: u64,
    pub copying_uploads: u64,
    pub nocopy_reuses: u64,
    /// How many of `block_upload_calls` were a caller-declared-static weight
    /// -- see [`Plan::mark_resident`] and `upload_resident_copy`. A resident
    /// upload happens once per distinct weight buffer; a resident reuse
    /// happens every token after that. `resident_reuses` growing while
    /// `resident_uploads` stays flat at the weight count is the direct
    /// witness that the ~5.84 GB/token copy `proxima-tensor/docs/discipline.md`
    /// ROW 82 measured moved exactly once, not every step.
    pub resident_uploads: u64,
    pub resident_reuses: u64,
    /// How many of `block_upload_calls` were served by
    /// `checkpoint_mapping_offset` -- addressed by offset into the ONE
    /// no-copy buffer spanning the whole checkpoint mapping, instead of
    /// falling to `resident_uploads`' per-tensor copy. `resident_uploads`
    /// staying at 0 while this climbs to the packed-weight count is the
    /// direct witness the 429,173,760-byte per-tensor copy this counter
    /// replaces never happens.
    pub mapping_offset_uploads: u64,
}

/// Reads and resets every split-4019 counter in one call — see
/// [`MetalStageTotals`]'s own doc for why snapshot-and-reset rather than
/// `.get()`.
#[cfg(feature = "instrument")]
pub fn metal_stage_totals() -> MetalStageTotals {
    MetalStageTotals {
        prepare_calls: PREPARE_CALLS.snapshot_and_reset(),
        prepare_ticks: PREPARE_TICKS.snapshot_and_reset(),
        emit_calls: EMIT_CALLS.snapshot_and_reset(),
        emit_ticks: EMIT_TICKS.snapshot_and_reset(),
        pipeline_hits: PIPELINE_HITS.snapshot_and_reset(),
        pipeline_misses: PIPELINE_MISSES.snapshot_and_reset(),
        pipeline_compile_ticks: PIPELINE_COMPILE_TICKS.snapshot_and_reset(),
        block_upload_calls: BLOCK_UPLOAD_CALLS.snapshot_and_reset(),
        block_upload_ticks: BLOCK_UPLOAD_TICKS.snapshot_and_reset(),
        block_upload_bytes: BLOCK_UPLOAD_BYTES.snapshot_and_reset(),
        op_setup_calls: OP_SETUP_CALLS.snapshot_and_reset(),
        op_setup_ticks: OP_SETUP_TICKS.snapshot_and_reset(),
        pipeline_lookup_calls: PIPELINE_LOOKUP_CALLS.snapshot_and_reset(),
        pipeline_lookup_ticks: PIPELINE_LOOKUP_TICKS.snapshot_and_reset(),
        encode_dispatch_calls: ENCODE_DISPATCH_CALLS.snapshot_and_reset(),
        encode_dispatch_ticks: ENCODE_DISPATCH_TICKS.snapshot_and_reset(),
        gpu_exec_calls: GPU_EXEC_CALLS.snapshot_and_reset(),
        gpu_exec_ticks: GPU_EXEC_TICKS.snapshot_and_reset(),
        readback_calls: READBACK_CALLS.snapshot_and_reset(),
        readback_ticks: READBACK_TICKS.snapshot_and_reset(),
        readback_bytes: READBACK_BYTES.snapshot_and_reset(),
        nocopy_uploads: NOCOPY_BUFFER_UPLOADS.snapshot_and_reset(),
        copying_uploads: COPYING_BUFFER_UPLOADS.snapshot_and_reset(),
        nocopy_reuses: NOCOPY_BUFFER_REUSES.snapshot_and_reset(),
        resident_uploads: RESIDENT_BUFFER_UPLOADS.snapshot_and_reset(),
        resident_reuses: RESIDENT_BUFFER_REUSES.snapshot_and_reset(),
        mapping_offset_uploads: MAPPING_OFFSET_UPLOADS.snapshot_and_reset(),
    }
}

/// How many real `upload_block` calls took each host->device path —
/// incremented once per call, never per byte, so a caller can read back the
/// no-copy hit rate after a run without an external profiler. See the
/// module doc's "Host buffer upload" section.
pub static NOCOPY_BUFFER_UPLOADS: Counter = Counter::new("omega.metal.upload_block.nocopy");
pub static COPYING_BUFFER_UPLOADS: Counter = Counter::new("omega.metal.upload_block.copy");

/// The host's page size, queried once and cached — the alignment unit
/// `newBufferWithBytesNoCopy` requires for both the pointer and the length
/// (16384 on Apple silicon, but this asks the OS rather than hard-coding
/// that). Public so a caller building block inputs (e.g.
/// `proxima_tensor::AlignedBuffer::new`) can size an allocation to this
/// exact host's page size instead of duplicating the sysconf call.
pub fn page_size() -> usize {
    static PAGE_SIZE: OnceLock<usize> = OnceLock::new();
    // SAFETY: `sysconf` takes a plain `c_int` name and has no preconditions;
    // `_SC_PAGESIZE` is POSIX-portable (macOS's `libc` crate has no
    // `getpagesize()` binding, unlike Linux's).
    *PAGE_SIZE.get_or_init(|| unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize })
}

/// `newBufferWithBytesNoCopy`'s hard requirement: `pointer` and `length`
/// must both land on a page boundary.
fn is_page_aligned(pointer: *const c_void, length: usize) -> bool {
    let page = page_size();
    (pointer as usize).is_multiple_of(page) && length.is_multiple_of(page)
}

/// Narrows the caller's f32 host data to `dtype`'s own width before
/// uploading — see this module's dtype doc for why that narrowing happens
/// exactly once, here, rather than the device buffer staying 4 bytes per
/// element regardless of `dtype`. `node` names the block input this upload
/// is for, used only to point an [`EmitError::UnsupportedDType`] at the
/// right place — [`reject_unsupported_gpu_dtype`] already keeps anything
/// but `Float32`/`Float16` from reaching this call inside [`execute`], so
/// the new arm below is a totality guard, not a path this driver's own
/// pipeline can actually hit.
fn upload_block(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[f32],
    node: NodeId,
    dtype: DType,
    resident: bool,
) -> Result<(MetalBuffer, usize), MetalError> {
    match dtype {
        // unreached by every program this driver compiles today (none
        // declares a `Float16` block input -- `proxima-tensor/src/spec.rs`'s
        // `mistral_cached_forward_program` is `Float32` throughout), so it is
        // not worth the residency cache's extra bookkeeping: the narrowed
        // `Vec<f16>` this allocates is dropped every call regardless.
        DType::Float16 => upload_block_as_half(device, data).map(|buffer| (buffer, 0)),
        DType::Float32
        | DType::BFloat16
        | DType::Bool
        | DType::Int8
        | DType::UInt8
        | DType::Int32
        | DType::UInt32 => upload_block_as_float(device, data, resident),
        DType::Int16
        | DType::UInt16
        | DType::Int64
        | DType::UInt64
        | DType::Int128
        | DType::UInt128
        | DType::Float64 => Err(EmitError::UnsupportedDType { node, dtype }.into()),
    }
}

/// The only path that can take the no-copy upload: the caller's own
/// `&[f32]` slice is borrowed for [`execute`]'s entire call, which
/// `waitUntilCompleted`s its single command buffer (every op's reads
/// included) before that borrow can end, so handing the GPU the caller's
/// own pointer is sound whenever it is page-aligned. See the module doc's
/// "Host buffer upload" section for why [`upload_block_as_half`] can never
/// take this path. `resident` is [`Plan::mark_resident`]'s classification of
/// this block's own node -- see the module doc's "Resident blocks" section
/// for why a misaligned RESIDENT block still gets a cache, just a different
/// one than the no-copy path's.
///
/// CACHING the wrapper this creates is gated on `resident`, not on
/// `is_page_aligned` alone: page alignment is a property of an ADDRESS, not
/// of a LIFETIME, and `(pointer, byte_length)` is exactly the key an
/// ephemeral, growing buffer (a KV-cache row, say) can reuse after a
/// realloc moves a DIFFERENT allocation onto the same range. Only
/// `mark_resident`'s "this address holds a model weight nothing overwrites
/// again" proof licenses remembering the wrapper past this one call -- see
/// `proxima-tensor/docs/discipline.md` ROW 70.
fn upload_block_as_float(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[f32],
    resident: bool,
) -> Result<(MetalBuffer, usize), MetalError> {
    if data.is_empty() {
        return allocate_buffer(device, 0, DType::Float32).map(|buffer| (buffer, 0));
    }
    let byte_length = size_of_val(data);
    let pointer = data.as_ptr().cast::<c_void>();
    if is_page_aligned(pointer, byte_length) {
        counter!(NOCOPY_BUFFER_UPLOADS, 1);
        if resident {
            return upload_block_no_copy(device, pointer, byte_length).map(|buffer| (buffer, 0));
        }
        return upload_block_no_copy_uncached(device, pointer, byte_length)
            .map(|buffer| (buffer, 0));
    }
    if let Some(result) = checkpoint_mapping_offset(device, pointer, byte_length) {
        return result;
    }
    if resident {
        return upload_resident_copy(device, pointer, byte_length).map(|buffer| (buffer, 0));
    }
    counter!(COPYING_BUFFER_UPLOADS, 1);
    upload_block_copy(device, pointer, byte_length).map(|buffer| (buffer, 0))
}

/// Uploads a packed quantized weight buffer as raw BYTES — no dequantize on
/// the host, which is the entire point. A 7B `Q4_K_S` checkpoint is 3.784 GB
/// packed against 14.5 GB as `f16`; decode is a weight sweep, so that 3.56x
/// in traffic IS the token rate. Reuses the same page-aligned no-copy path
/// [`upload_block_as_float`] uses, since a memory-mapped GGUF tensor is very
/// often already page-aligned. `resident` is the same "caller's own static
/// weight" classification [`upload_block_as_float`] takes; a packed weight
/// too misaligned for the no-copy path takes the same resident-copy cache.
fn upload_packed_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
    resident: bool,
) -> Result<(MetalBuffer, usize), MetalError> {
    if bytes.is_empty() {
        return allocate_buffer(device, 0, DType::Float32).map(|buffer| (buffer, 0));
    }
    let byte_length = bytes.len();
    let pointer = bytes.as_ptr().cast::<c_void>();
    if is_page_aligned(pointer, byte_length) {
        counter!(NOCOPY_BUFFER_UPLOADS, 1);
        if resident {
            return upload_block_no_copy(device, pointer, byte_length).map(|buffer| (buffer, 0));
        }
        return upload_block_no_copy_uncached(device, pointer, byte_length)
            .map(|buffer| (buffer, 0));
    }
    if let Some(result) = checkpoint_mapping_offset(device, pointer, byte_length) {
        return result;
    }
    if resident {
        return upload_resident_copy(device, pointer, byte_length).map(|buffer| (buffer, 0));
    }
    counter!(COPYING_BUFFER_UPLOADS, 1);
    upload_block_copy(device, pointer, byte_length).map(|buffer| (buffer, 0))
}

thread_local! {
    /// The page-aligned, process-lifetime checkpoint mapping registered by
    /// [`register_checkpoint_mapping`] -- `(base_pointer, byte_length)` of the
    /// caller's own mmap. `None` until a loader calls it. A tensor whose byte
    /// range falls entirely inside this span never needs its own device
    /// buffer: see [`checkpoint_mapping_offset`].
    static CHECKPOINT_MAPPING: RefCell<Option<(usize, usize)>> = const { RefCell::new(None) };
}

/// Registers the whole-checkpoint memory mapping backing every packed
/// tensor's borrowed bytes, so a tensor whose own byte offset inside that
/// mapping is misaligned for `is_page_aligned` can still reach the GPU
/// without a copy -- by address, into ONE no-copy buffer spanning the whole
/// mapping, instead of one buffer per tensor. See `checkpoint_mapping_offset`
/// for the containment check and `upload_packed_bytes`'s doc for why the
/// per-tensor page-alignment test this replaces fails for every packed
/// tensor in a real GGUF layout (tensors are packed back-to-back at their
/// natural sizes; only the mapping's OWN base is page-aligned).
///
/// `bytes` must stay mapped, unchanged, at this address for the rest of the
/// process -- the same precondition `NOCOPY_BUFFERS` already rests on for
/// a single tensor, extended here to the whole file. Calling this again
/// replaces the previous registration; callers load one checkpoint mapping
/// per process in every reachable path today.
pub fn register_checkpoint_mapping(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let base = bytes.as_ptr() as usize;
    CHECKPOINT_MAPPING.with(|mapping| *mapping.borrow_mut() = Some((base, bytes.len())));
}

/// Counts uploads served by addressing the shared checkpoint-mapping buffer
/// at an offset, instead of copying the tensor into its own buffer -- the
/// direct witness for the census this mechanism is meant to zero out.
pub static MAPPING_OFFSET_UPLOADS: Counter =
    Counter::new("omega.metal.upload_block.mapping_offset");

/// If `pointer..pointer+byte_length` falls entirely inside the registered
/// checkpoint mapping, returns the whole-mapping no-copy buffer (created
/// once, reused after) plus this tensor's byte OFFSET into it. `None` when
/// no mapping is registered or the range falls outside it -- the scratch and
/// KV-cache buffers `upload_block_as_float`'s `Float32` arm also uploads
/// through never live inside the checkpoint's own mmap, so they fall
/// through unchanged.
///
/// The mapping's total length is rounded UP to a page boundary before
/// `newBufferWithBytesNoCopy` sees it, since that call requires a
/// page-aligned length; that rounding is sound because `mmap` only ever
/// backs a file with whole pages, so every byte up to the next page
/// boundary past the file's own length is already resident, zero-filled,
/// mapped memory (never past the region the OS mapped for this file) --
/// reading it is safe, and no kernel this driver emits ever reads past a
/// tensor's own declared byte length regardless.
fn checkpoint_mapping_offset(
    device: &ProtocolObject<dyn MTLDevice>,
    pointer: *const c_void,
    byte_length: usize,
) -> Option<Result<(MetalBuffer, usize), MetalError>> {
    let (base, mapping_length) = CHECKPOINT_MAPPING.with(|mapping| *mapping.borrow())?;
    let address = pointer as usize;
    if address < base || address + byte_length > base + mapping_length {
        return None;
    }
    let page = page_size();
    let rounded_length = mapping_length.div_ceil(page) * page;
    counter!(MAPPING_OFFSET_UPLOADS, 1);
    Some(
        upload_block_no_copy(device, base as *const c_void, rounded_length)
            .map(|buffer| (buffer, address - base)),
    )
}

thread_local! {
    /// No-copy block buffers, keyed by the exact host range they wrap.
    ///
    /// `newBufferWithBytesNoCopy` does not copy, but it is NOT free: every
    /// call creates a fresh `MTLBuffer` and Metal has to wire those pages
    /// for GPU access. `execute` rebuilt every block buffer on every call,
    /// so a serving loop re-wired the entire weight set per token — a cost
    /// that scales with BYTES, which is exactly what made it invisible in a
    /// bytes-normalized probe.
    ///
    /// CALLER PRECONDITION, enforced at the call site via `resident`, not
    /// (yet) a type: a cached wrapper aliases the caller's pages and Metal
    /// does NOT own them, so the host range `(pointer, len)` must stay
    /// mapped, and must never be reused by a DIFFERENT allocation, for as
    /// long as this thread keeps using omega. `upload_block_as_float` and
    /// `upload_packed_bytes` only route into this cache when
    /// [`Plan::mark_resident`] already proved that for the node's own
    /// address -- true for mmap'd GGUF weights, false for an ephemeral,
    /// growing buffer (a KV-cache row, say) whose page-aligned address is a
    /// coincidence of a page-boundary crossing, not a lifetime proof. A
    /// page-aligned but non-resident block takes
    /// `upload_block_no_copy_uncached` instead: still zero-copy for this one
    /// `execute` call (sound for the same `waitUntilCompleted` reason), just
    /// never remembered past it. See `proxima-tensor/docs/discipline.md`
    /// ROW 70.
    ///
    /// Reuse is otherwise safe on the data-freshness axis precisely BECAUSE
    /// it is no-copy: writes through the caller's own slice are visible to
    /// the GPU, so a wrapper never goes stale. Copying uploads are
    /// deliberately NOT cached — those snapshot the data, and reuse would
    /// serve a stale snapshot.
    static NOCOPY_BUFFERS: RefCell<BTreeMap<(usize, usize), MetalBuffer>> =
        RefCell::new(BTreeMap::new());
}

/// Counts entries `NOCOPY_BUFFERS` actually holds right now — the direct
/// witness that gating the cache on `resident` stops it growing without
/// bound. A non-resident page-aligned block (a KV-cache row that happened to
/// cross a page boundary) never reaches this map at all.
#[must_use]
pub fn nocopy_cache_len() -> usize {
    NOCOPY_BUFFERS.with(|cache| cache.borrow().len())
}

/// Counts the no-copy wrappers this thread reused instead of recreating —
/// the direct witness that a serving loop stops re-wiring its weights.
pub static NOCOPY_BUFFER_REUSES: Counter = Counter::new("omega.metal.upload_block.nocopy_reuse");

/// The zero-copy path: shares `pointer`'s memory directly with the GPU
/// instead of duplicating it. Sound only because every caller of
/// [`upload_block`] binds the returned buffer to a `device const float*`
/// kernel argument (see `msl::kernel_signature`) — the GPU never writes
/// through it, matching the `&[f32]` (never `&mut`) the caller handed us —
/// and because [`execute`] `waitUntilCompleted`s the one command buffer
/// every op (including this buffer's reads) is encoded into before
/// [`upload_block_as_float`]'s caller-owned slice's borrow can end.
///
/// Cached in [`NOCOPY_BUFFERS`] — reachable ONLY when the caller already
/// classified this address `resident` (see the call sites in
/// [`upload_block_as_float`]/[`upload_packed_bytes`]); a non-resident
/// page-aligned block takes [`upload_block_no_copy_uncached`] instead, which
/// shares this function's Metal call but never remembers the wrapper.
fn upload_block_no_copy(
    device: &ProtocolObject<dyn MTLDevice>,
    pointer: *const c_void,
    byte_length: usize,
) -> Result<MetalBuffer, MetalError> {
    let key = (pointer as usize, byte_length);
    if let Some(existing) = NOCOPY_BUFFERS.with(|cache| cache.borrow().get(&key).cloned()) {
        counter!(NOCOPY_BUFFER_REUSES, 1);
        return Ok(existing);
    }
    let buffer = create_no_copy_buffer(device, pointer, byte_length)?;
    NOCOPY_BUFFERS.with(|cache| cache.borrow_mut().insert(key, buffer.clone()));
    Ok(buffer)
}

/// The uncached counterpart to [`upload_block_no_copy`]: still hands Metal
/// the caller's own pointer directly (zero-copy, sound for this one
/// `execute` call for the exact reason [`upload_block_no_copy`]'s doc
/// gives), but never inserts the wrapper into [`NOCOPY_BUFFERS`]. Taken
/// whenever a page-aligned block's node is NOT [`Plan::mark_resident`]-classified
/// static — an ephemeral, growing buffer (a KV-cache row that crossed a page
/// boundary) can cross that same alignment by coincidence on every call, and
/// caching it would key a permanent entry off an address whose CONTENTS,
/// and even whose owning allocation, changes underneath it.
fn upload_block_no_copy_uncached(
    device: &ProtocolObject<dyn MTLDevice>,
    pointer: *const c_void,
    byte_length: usize,
) -> Result<MetalBuffer, MetalError> {
    create_no_copy_buffer(device, pointer, byte_length)
}

/// The `newBufferWithBytesNoCopy` FFI call itself, shared by
/// [`upload_block_no_copy`] and [`upload_block_no_copy_uncached`] — caching
/// is entirely the callers' concern, not this function's.
fn create_no_copy_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    pointer: *const c_void,
    byte_length: usize,
) -> Result<MetalBuffer, MetalError> {
    // SAFETY: `pointer` is non-null (it comes from a non-empty slice) and,
    // per `is_page_aligned`, page-aligned with a page-aligned `byte_length`
    // — `newBufferWithBytesNoCopy`'s documented precondition. Passing `None`
    // as the deallocator tells Metal it never owns this memory, so it is
    // never freed or written out from under the caller.
    let pointer = unsafe { NonNull::new_unchecked(pointer as *mut c_void) };
    unsafe {
        device.newBufferWithBytesNoCopy_length_options_deallocator(
            pointer,
            byte_length,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
    .ok_or_else(|| MetalError::CompileFailed {
        log: "device refused a no-copy shared buffer for a page-aligned block input".to_string(),
    })
}

fn upload_block_copy(
    device: &ProtocolObject<dyn MTLDevice>,
    pointer: *const c_void,
    byte_length: usize,
) -> Result<MetalBuffer, MetalError> {
    // SAFETY: `pointer` is a live, non-null address for the duration of this
    // call (borrowed from the caller's own `&[f32]`, or a locally owned
    // narrowed `Vec<f16>` that outlives this call), so it stays valid while
    // `newBufferWithBytes_length_options` copies from it.
    let pointer = unsafe { NonNull::new_unchecked(pointer as *mut c_void) };
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

thread_local! {
    /// Copied device buffers for blocks [`Plan::mark_resident`] classified as
    /// the caller's own static weights -- a SEPARATE map from
    /// [`NOCOPY_BUFFERS`], not a shared one, because the two caches rest on
    /// different soundness arguments: a no-copy entry is safe to reuse
    /// unconditionally because it aliases whatever is CURRENTLY at that
    /// address (never stale by construction); this cache instead reuses a
    /// SNAPSHOT taken at first upload, which is sound only because the
    /// caller already proved -- by name, once, in `mark_resident` -- that
    /// the address holds a model weight nothing overwrites again. Keeping
    /// them apart keeps that distinction visible at the call site instead of
    /// folding two different proofs into one lookup.
    static RESIDENT_BUFFERS: RefCell<BTreeMap<(usize, usize), MetalBuffer>> =
        RefCell::new(BTreeMap::new());
}

/// How many resident (caller-declared-static) blocks took a real copy versus
/// how many were served from `RESIDENT_BUFFERS` instead -- the direct
/// witness that the ~5.84 GB/token `proxima-tensor/docs/discipline.md` ROW 82
/// measured moving through `upload_block_copy` on every step now moves
/// exactly once per distinct weight buffer, never once per token.
pub static RESIDENT_BUFFER_UPLOADS: Counter =
    Counter::new("omega.metal.upload_block.resident_upload");
pub static RESIDENT_BUFFER_REUSES: Counter =
    Counter::new("omega.metal.upload_block.resident_reuse");

/// The copy-path counterpart to [`upload_block_no_copy`]: called only for a
/// block [`upload_block_as_float`]/[`upload_packed_bytes`] already found
/// misaligned AND [`Plan::mark_resident`] already classified as static, so
/// unlike [`upload_block_copy`] this one is allowed to remember the buffer it
/// creates and hand the SAME one back next time the SAME `(pointer, len)`
/// shows up -- sound only because that classification, not an address guess,
/// is what proves the bytes behind `pointer` never change again.
fn upload_resident_copy(
    device: &ProtocolObject<dyn MTLDevice>,
    pointer: *const c_void,
    byte_length: usize,
) -> Result<MetalBuffer, MetalError> {
    let key = (pointer as usize, byte_length);
    if let Some(existing) = RESIDENT_BUFFERS.with(|cache| cache.borrow().get(&key).cloned()) {
        counter!(RESIDENT_BUFFER_REUSES, 1);
        return Ok(existing);
    }
    counter!(COPYING_BUFFER_UPLOADS, 1);
    counter!(RESIDENT_BUFFER_UPLOADS, 1);
    let buffer = upload_block_copy(device, pointer, byte_length)?;
    RESIDENT_BUFFERS.with(|cache| cache.borrow_mut().insert(key, buffer.clone()));
    Ok(buffer)
}

/// Always copies — see the module doc's "Host buffer upload" section for
/// why a freshly narrowed `Vec<f16>` can never take the no-copy path: it
/// drops the instant this function returns, so no-copy would hand Metal a
/// dangling pointer.
fn upload_block_as_half(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[f32],
) -> Result<MetalBuffer, MetalError> {
    if data.is_empty() {
        return allocate_buffer(device, 0, DType::Float16);
    }
    let narrowed: Vec<f16> = data.iter().map(|value| f16::from_f32(*value)).collect();
    let byte_length = size_of_val(narrowed.as_slice());
    let pointer = narrowed.as_ptr().cast::<c_void>();
    counter!(COPYING_BUFFER_UPLOADS, 1);
    upload_block_copy(device, pointer, byte_length)
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

thread_local! {
    /// Uniform blobs, keyed by their own bytes. A plan's uniforms are a
    /// function of the BOUND OP — extents, strides, bases — so they are
    /// byte-identical on every call, and `execute` was allocating a fresh
    /// `MTLBuffer` for each of them per op per call. Safe to share: the
    /// kernel binds them `constant` and never writes through them, and two
    /// ops with identical uniform bytes want identical contents by
    /// definition.
    static UNIFORM_BUFFERS: RefCell<BTreeMap<Vec<u8>, MetalBuffer>> =
        RefCell::new(BTreeMap::new());
}

/// Counts uniform buffers served from cache rather than allocated.
pub static UNIFORM_BUFFER_REUSES: Counter = Counter::new("omega.metal.uniforms.reuse");
fn upload_uniforms(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    if let Some(existing) = UNIFORM_BUFFERS.with(|cache| cache.borrow().get(bytes).cloned()) {
        counter!(UNIFORM_BUFFER_REUSES, 1);
        return Ok(existing);
    }
    // SAFETY: `bytes` is always non-empty (every `Uniforms` struct has at
    // least two `long` fields), so its first byte's address is valid and
    // stays valid while this call copies from it.
    let pointer = unsafe { NonNull::new_unchecked(bytes.as_ptr() as *mut c_void) };
    let buffer = unsafe {
        device.newBufferWithBytes_length_options(
            pointer,
            bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| MetalError::CompileFailed {
        log: "device refused to allocate the uniforms buffer".to_string(),
    })?;
    UNIFORM_BUFFERS.with(|cache| cache.borrow_mut().insert(bytes.to_vec(), buffer.clone()));
    Ok(buffer)
}

fn buffer_for(device_buffers: &BTreeMap<NodeId, DeviceBuffer>, node: NodeId) -> Result<DeviceBuffer, MetalError> {
    device_buffers.get(&node).cloned().ok_or_else(|| {
        TensorError::NotLowerable {
            node,
            reason: "operand buffer missing at execution time",
        }
        .into()
    })
}

/// `output` is `(buffer, byte_offset)` rather than two separate parameters
/// so this function stays under clippy's argument-count lint without a
/// `#[allow]` — the pair is always passed and used together, never
/// independently. `byte_offset` is always `0` for a fresh, op-sized buffer
/// (the shape every call site used before `metal-output-placement`
/// existed); it is non-zero only when `buffer` is a caller-owned
/// [`PlacedBuffer`] the op is writing into at an offset (see
/// [`execute_plan_with_placements`]). An `Input`/`Indices` binding's own
/// offset travels with it already, in `device_buffers`' own
/// [`DeviceBuffer`] pair -- the same tuple `checkpoint_mapping_offset`
/// (an input-only placement predating this feature) already relied on, so
/// an input-placed node needs no separate offset map: its offset is
/// whatever [`execute_plan_with_placements`] inserted into `device_buffers`
/// for it. Uniforms and the fault buffer are always read from their own
/// start — neither is ever placed.
fn bind_buffers(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    bindings: &[Binding],
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    output: (&Retained<ProtocolObject<dyn MTLBuffer>>, usize),
    uniforms: &Retained<ProtocolObject<dyn MTLBuffer>>,
    fault: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
) -> Result<(), MetalError> {
    let (output_buffer, output_offset) = output;
    for (index, binding) in bindings.iter().enumerate() {
        let (buffer, offset) = match binding {
            Binding::Input(node) | Binding::Indices(node) => buffer_for(device_buffers, *node)?,
            Binding::Output(_) => (output_buffer.clone(), output_offset),
            Binding::Uniforms => (uniforms.clone(), 0),
            Binding::Fault => (
                fault.cloned().ok_or_else(|| MetalError::CompileFailed {
                    log: "kernel binds a fault buffer but none was allocated".to_string(),
                })?,
                0,
            ),
        };
        // SAFETY: `buffer`'s length was sized from the same op this kernel
        // was emitted from (or, for a placed input/output, the caller
        // guaranteed `offset + <this binding's own element count> *
        // dtype.size_bytes()` fits inside it — see
        // `execute_plan_with_placements`'s own doc), so every byte the
        // kernel indexes through this binding is in bounds, starting at
        // `offset` -- 0 for every binding except a tensor
        // `checkpoint_mapping_offset` or `metal-output-placement` address.
        unsafe { encoder.setBuffer_offset_atIndex(Some(&buffer), offset, index) };
    }
    Ok(())
}

/// `grid.threadgroup_width`, when present, is not an occupancy hint — it is
/// a correctness requirement a cooperative-reduce kernel's own coordinate
/// math depends on (`gid / SIMD_WIDTH` as an output index; see
/// `crate::msl::push_cooperative_reduce_body`'s doc), so it is honored
/// exactly rather than folded into the generic `min(threads, max)` pick.
fn dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    grid: GridSpec,
) {
    if grid.threads == 0 {
        return;
    }
    let max_threadgroup = pipeline.maxTotalThreadsPerThreadgroup();
    let threadgroup_width = match grid.threadgroup_width {
        Some(width) => (width as usize).min(max_threadgroup).max(1),
        None => (grid.threads as usize).min(max_threadgroup).max(1),
    };
    let grid_size = MTLSize {
        width: grid.threads as usize,
        height: 1,
        depth: 1,
    };
    let threadgroup = MTLSize {
        width: threadgroup_width,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup);
}

/// Encodes one `BoundOp` as a compute dispatch into the CALLER's already-open
/// `encoder` — neither opened nor `endEncoding()`d here. [`execute_plan`]
/// opens exactly one `MTLComputeCommandEncoder` for the whole program and
/// ends it once, after every op has been encoded into it (see the module
/// doc's "Execution model"); [`execute_plan_op_timed`]'s diagnostic path
/// instead opens and ends one per call, one op at a time. Either way this
/// function only ever binds a pipeline, binds buffers, and dispatches —
/// exactly the same three calls regardless of how many other ops share the
/// encoder it was handed. Returns the op's fault buffer and gather count
/// when it gathers, so the caller can check it after its own wait instead
/// of here, where the buffer is not yet CPU-visible.
///
/// `placement` is `None` on every call site that predates
/// `metal-output-placement` (byte-identical to before: a fresh
/// `allocate_buffer` sized to this op's own iteration space). `Some((buffer,
/// offset))` skips that allocation and binds `bound`'s output straight into
/// the caller-owned `buffer` at `offset` instead -- that offset is then
/// carried forward in `device_buffers`' own [`DeviceBuffer`] entry for this
/// node, so a later op reading it back through
/// [`bind_buffers`]/[`buffer_for`] needs no separate offset map.
fn encode_op(
    device: &ProtocolObject<dyn MTLDevice>,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device_buffers: &mut BTreeMap<NodeId, DeviceBuffer>,
    bound: &BoundOp,
    packed_operands: &PackedOperands,
    placement: Option<(&MetalBuffer, usize)>,
) -> Result<Option<(MetalBuffer, usize)>, MetalError> {
    #[cfg(feature = "instrument")]
    let emit_started = read_ticks();
    // `kernel_cache_key`/`kernel_dispatch_shape` are the cheap halves of
    // `emit`'s work -- structural fingerprint, bindings, grid -- with no MSL
    // body text rendered. On a pipeline-cache HIT (the steady-decode case,
    // `plan_hits`/`gpu_exec`'s own row) `emit` itself is never called; only a
    // genuine miss inside `pipeline_for` pays for the full render + compile.
    let cache_key = kernel_cache_key(bound, packed_operands)?;
    let (bindings, grid) = kernel_dispatch_shape(bound, packed_operands)?;
    #[cfg(feature = "instrument")]
    {
        counter!(EMIT_CALLS, 1);
        counter!(EMIT_TICKS, elapsed_ticks(emit_started));
    }
    #[cfg(feature = "instrument")]
    let pipeline_started = read_ticks();
    let pipeline = pipeline_for(device, bound, packed_operands, &cache_key)?;
    #[cfg(feature = "instrument")]
    {
        counter!(PIPELINE_LOOKUP_CALLS, 1);
        counter!(PIPELINE_LOOKUP_TICKS, elapsed_ticks(pipeline_started));
    }
    #[cfg(feature = "instrument")]
    let op_setup_started = read_ticks();
    let (output, output_offset) = match placement {
        Some((buffer, offset)) => (buffer.clone(), offset),
        None => (
            allocate_buffer(device, bound_output_len(bound), bound.dtype)?,
            0,
        ),
    };
    let uniforms = upload_uniforms(device, &pack_uniforms(bound))?;
    let gathers = gather_count(bound);
    let fault = (gathers > 0)
        .then(|| allocate_fault_buffer(device, gathers))
        .transpose()?;
    #[cfg(feature = "instrument")]
    {
        counter!(OP_SETUP_CALLS, 1);
        counter!(OP_SETUP_TICKS, elapsed_ticks(op_setup_started));
    }

    // ROW 92 named this residual (`omega/src/metal.rs:1873-1908` at that
    // row's line numbers) as four uncounted Metal API calls: `pipeline_for`'s
    // own lookup (now `PIPELINE_LOOKUP_TICKS` above), `setComputePipelineState`,
    // `bind_buffers`, `dispatch` -- the three below, wrapped together since
    // none does per-op device work heavy enough to need its own counter, and
    // splitting them finer would cost more ticks reading the clock than the
    // calls themselves take.
    #[cfg(feature = "instrument")]
    let encode_dispatch_started = read_ticks();
    encoder.setComputePipelineState(&pipeline);
    bind_buffers(
        encoder,
        &bindings,
        device_buffers,
        (&output, output_offset),
        &uniforms,
        fault.as_ref(),
    )?;
    dispatch(encoder, &pipeline, grid);
    #[cfg(feature = "instrument")]
    {
        counter!(ENCODE_DISPATCH_CALLS, 1);
        counter!(
            ENCODE_DISPATCH_TICKS,
            elapsed_ticks(encode_dispatch_started)
        );
    }

    device_buffers.insert(bound.node, (output, output_offset));
    Ok(fault.map(|fault_buffer| (fault_buffer, gathers)))
}

/// Reads back a dispatch's fault buffer and, if any slot recorded a fault,
/// turns it into the same `TensorError::GatherIndexOutOfRange`
/// `cpu::evaluate` reports for the identical fetched index. Slot order
/// matches `bound.operands()`' gather order — the same numbering
/// `crate::msl::gather_slots` and `push_gather_uniforms` both use — so the
/// first faulted slot's own `Lookup` supplies the extent to report.
fn check_gather_fault(
    bound: &BoundOp,
    fault_buffer: &ProtocolObject<dyn MTLBuffer>,
    gather_count: usize,
) -> Result<(), MetalError> {
    let slots = read_fault_slots(fault_buffer, gather_count);
    let gathers: Vec<&Lookup> = bound
        .operands()
        .iter()
        .filter_map(|(_, _, gather)| gather.as_ref())
        .collect();
    for (slot, recorded) in slots.iter().enumerate() {
        if *recorded != 0 {
            return Err(TensorError::GatherIndexOutOfRange {
                node: bound.node,
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

/// Widens a device buffer back to the host's f32 contract — see this
/// module's dtype doc for why that widening happens exactly once, here,
/// mirroring the narrowing [`upload_block`] does on the way in. `node`
/// names the output this read-back is for, used only to point an
/// [`EmitError::UnsupportedDType`] at the right place — same totality-guard
/// stance as [`upload_block`]'s `node` parameter.
fn read_back(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    element_count: usize,
    node: NodeId,
    dtype: DType,
) -> Result<Vec<f32>, MetalError> {
    if element_count == 0 {
        return Ok(Vec::new());
    }
    match dtype {
        DType::Float16 => Ok(read_back_half(buffer, element_count)),
        DType::Float32
        | DType::BFloat16
        | DType::Bool
        | DType::Int8
        | DType::UInt8
        | DType::Int32
        | DType::UInt32 => Ok(read_back_float(buffer, element_count)),
        DType::Int16
        | DType::UInt16
        | DType::Int64
        | DType::UInt64
        | DType::Int128
        | DType::UInt128
        | DType::Float64 => Err(EmitError::UnsupportedDType { node, dtype }.into()),
    }
}

fn read_back_float(buffer: &ProtocolObject<dyn MTLBuffer>, element_count: usize) -> Vec<f32> {
    let pointer = buffer.contents();
    // SAFETY: `buffer` is `storageModeShared`, so `contents()` is a
    // CPU-visible pointer to at least `element_count` initialized `f32`s —
    // every output buffer this driver allocates is sized to at least that
    // many elements (see `allocate_buffer`'s caller, `dispatch_op`) before
    // this point is reached.
    unsafe { core::slice::from_raw_parts(pointer.as_ptr().cast::<f32>(), element_count) }.to_vec()
}

fn read_back_half(buffer: &ProtocolObject<dyn MTLBuffer>, element_count: usize) -> Vec<f32> {
    let pointer = buffer.contents();
    // SAFETY: `buffer` is `storageModeShared`, so `contents()` is a
    // CPU-visible pointer to at least `element_count` initialized `f16`s —
    // the same sizing guarantee `read_back_float` relies on, just over the
    // narrower element width `allocate_buffer` used for a `Float16` node.
    let narrow =
        unsafe { core::slice::from_raw_parts(pointer.as_ptr().cast::<f16>(), element_count) };
    narrow.iter().map(|value| value.to_f32()).collect()
}

fn finish(
    program: &[Op],
    index_nodes: &BTreeSet<NodeId>,
    shapes: &Shapes,
    effective_outputs: &[NodeId],
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    root: NodeId,
) -> Result<Evaluated, MetalError> {
    let mut results = Vec::with_capacity(effective_outputs.len());
    #[cfg(feature = "instrument")]
    let readback_started = read_ticks();
    for node in effective_outputs {
        let shape = shapes.of(*node).to_vec();
        let dtype = gpu_dtype(program, index_nodes, *node);
        let data = match device_buffers.get(node) {
            // an output node's buffer is always freshly allocated by
            // `encode_op` at offset 0 -- only a weight INPUT can carry a
            // nonzero offset, and a weight is never a program output -- so
            // reading from the buffer's own start is always correct here.
            Some((buffer, _offset)) => read_back(buffer, element_count(&shape), *node, dtype)?,
            None => Vec::new(),
        };
        #[cfg(feature = "instrument")]
        {
            counter!(READBACK_CALLS, 1);
            counter!(READBACK_BYTES, size_of_val(data.as_slice()) as u64);
        }
        results.push((*node, shape, data));
    }
    #[cfg(feature = "instrument")]
    counter!(READBACK_TICKS, elapsed_ticks(readback_started));
    // this backend's buffer lifetime is managed by Metal's own
    // retain/release, not counted the way `cpu::evaluate` counts its
    // `Vec<Option<Vec<f32>>>` table, so peak_live_buffers is not tracked
    // here — see `Evaluated`'s own doc for why `None` is the honest answer
    // rather than a number that would not mean the same thing.
    Ok(Evaluated::from_parts(root, results, None))
}
