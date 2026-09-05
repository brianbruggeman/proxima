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
//! One `MTLCommandBuffer` AND one `MTLComputeCommandEncoder` per [`execute`]
//! call: every `BoundOp` in `prepared.resolved` is encoded, in program
//! order, into that SAME encoder (see `encode_op`'s call site), the encoder
//! is `endEncoding()`d exactly once after the loop, and only then is the
//! command buffer `commit()`ted and `waitUntilCompleted()` exactly once, in
//! [`execute`]. Every expression used to pay a full CPU<->GPU round trip;
//! batching means only the genuine program outputs (`finish`'s
//! `effective_outputs`) ever cross back to the host, and intermediates
//! never do (they already didn't — `device_buffers` keeps them
//! GPU-resident between ops; what changes here is that the CPU no longer
//! blocks between ops either).
//!
//! Ordering is guaranteed, not assumed: `computeCommandEncoder()` (no
//! dispatch-type argument) defaults to `MTLDispatchTypeSerial`, so the
//! encoder's dispatches execute in encode order, and a later op reading a
//! buffer an earlier op wrote sees that write because every buffer here
//! comes from `device.newBuffer*` (see `allocate_buffer`, `upload_block`)
//! with `MTLResourceOptions::StorageModeShared` only — never
//! `HazardTrackingModeUntracked` — and a buffer's `hazardTrackingMode` for
//! any resource created directly from a device (as opposed to a heap)
//! defaults to tracked (`objc2-metal-0.3.2`'s
//! `src/generated/MTLResource.rs:326-329`: "Resources created from heaps
//! are by default untracked, whereas resources created from the device are
//! by default tracked."). That guarantee composes with [`execute`] encoding
//! `prepared.resolved` strictly in program order (the same order
//! `prepare`'s own [`proxima_tensor::node_retirement`] call already relies on for liveness), so
//! serial dispatch order plus default hazard tracking is the mechanism —
//! not an assumption that the GPU happens to serialize. This holds equally
//! for the no-copy buffers `upload_block` hands out (see "Host buffer
//! upload" below): `newBufferWithBytesNoCopy_length_options_deallocator`
//! takes the same `MTLResourceOptions`, so its hazard mode is identical.
//!
//! llama.cpp's `ggml-metal.m` at its default `n_cb=1` uses the identical
//! shape — one command buffer, one encoder per token — so encoder count is
//! not where the two runtimes' decode paths diverge: measured this
//! session, GPU kernel time is 83.2% of the decode step and orchestration
//! is 16.8%.
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
#[cfg(feature = "metal-concurrent-dispatch")]
use objc2_metal::{MTLBarrierScope, MTLDispatchType};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};
use proxima_telemetry::counter;
use proxima_telemetry::debug;
use proxima_telemetry::metric::Counter;
use proxima_telemetry::trace;

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
    #[error("hazard tracking: operand {node} has no resolved device buffer")]
    UnresolvedHazardOperand { node: NodeId },
    /// `build_buffer_arena`'s own reuse pass still needed more transient
    /// bytes live at once than `ARENA_TRANSIENT_CAP` budgets -- MG-3's
    /// kill condition, now a typed error a caller can act on rather than a
    /// stderr line beside a silently returned `Ok`.
    #[error("arena peak_bytes={peak_bytes} exceeds arena_transient_cap={cap}")]
    ArenaOverCap { peak_bytes: usize, cap: usize },
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
    /// CARD 6.5: whole-buffer, size-class-reused device output buffers for
    /// every position in `prepared.resolved`. Built lazily, on the first
    /// call that actually consults a placement (`arena_placement`) --
    /// [`plan`] itself no longer builds this eagerly, since the ordinary
    /// (unplaced) `execute_plan`/`execute_plan_op_timed` paths never read it
    /// and were paying its device allocation on every miss regardless. See
    /// [`BufferArena`]'s own doc.
    #[cfg(feature = "metal-plan-stable-buffers")]
    arena: core::cell::OnceCell<BufferArena>,
    /// CARD 6.5: one uniform buffer per plan position, written in place by
    /// `encode_op` instead of going through the content-keyed
    /// `UNIFORM_BUFFERS` cache. Lazily built alongside `arena`, for the same
    /// reason -- see [`PlanUniforms`]'s own doc.
    #[cfg(feature = "metal-plan-stable-buffers")]
    uniforms: core::cell::OnceCell<PlanUniforms>,
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

    /// The arena's live-bytes high-water mark reached while it was built --
    /// the direct witness `build_buffer_arena`'s own `debug!` event already
    /// emits before allocating, kept queryable afterward for a census
    /// line. `None` when this feature is off, or when this plan has never
    /// executed a placed call and so never built its arena (see
    /// `Self::arena`'s own doc).
    #[must_use]
    pub fn arena_peak_bytes(&self) -> Option<usize> {
        #[cfg(feature = "metal-plan-stable-buffers")]
        {
            Some(self.arena.get()?.peak_bytes)
        }
        #[cfg(not(feature = "metal-plan-stable-buffers"))]
        {
            None
        }
    }

    /// Physical device buffers the arena actually holds -- `< op_count`
    /// whenever `build_buffer_arena`'s free list reused a slot across two
    /// or more non-overlapping positions. `None` when this feature is off,
    /// or when this plan has never built its arena.
    #[must_use]
    pub fn arena_slot_count(&self) -> Option<usize> {
        #[cfg(feature = "metal-plan-stable-buffers")]
        {
            Some(self.arena.get()?.slot_count())
        }
        #[cfg(not(feature = "metal-plan-stable-buffers"))]
        {
            None
        }
    }

    /// The byte length of the physical slot backing `position`'s output --
    /// `None` when this feature is off, `position` is out of range, or this
    /// plan has never built its arena.
    #[must_use]
    pub fn arena_position_byte_len(&self, position: usize) -> Option<usize> {
        #[cfg(feature = "metal-plan-stable-buffers")]
        {
            let arena = self.arena.get()?;
            let slot = *arena.position_slot.get(position)?;
            Some(arena.slot_byte_len(slot))
        }
        #[cfg(not(feature = "metal-plan-stable-buffers"))]
        {
            let _ = position;
            None
        }
    }
}

/// Which of `block_nodes`' entries carry a codec [`crate::msl::emit`] has an
/// unpack kernel for (`Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `Q4_0`,
/// `Float16`, `BFloat16`), keyed to its [`PackedCodec`] — the single place this crate
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
            QuantizedBlock::Q3K(_) => Some((*node, PackedCodec::Q3K)),
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
        #[cfg(feature = "metal-plan-stable-buffers")]
        arena: core::cell::OnceCell::new(),
        #[cfg(feature = "metal-plan-stable-buffers")]
        uniforms: core::cell::OnceCell::new(),
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
            // fires unconditionally, before the upload-path match below --
            // this is what a block was OFFERED for upload, regardless of
            // which terminal path (no-copy bind, checkpoint-offset bind, or
            // a real copy) actually served it. See `BLOCK_COPIED_BYTES` /
            // `BLOCK_NOCOPY_BOUND_BYTES` / `BLOCK_OFFSET_BOUND_BYTES` for the
            // per-path split; `BLOCK_COPIED + BLOCK_NOCOPY_BOUND +
            // BLOCK_OFFSET_BOUND == BLOCK_OFFERED` on every step is the
            // partition identity that proves no path is uninstrumented.
            counter!(BLOCK_OFFERED_BYTES, block_byte_len(block) as u64);
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
            // `Q3_K` uploads its raw super-block bytes unchanged, same as
            // every other packed codec below -- `msl::PackedCodec::Q3K`'s
            // own unpack kernel (`q3k_element`) reads them at the GPU side.
            QuantizedBlock::Q3K(bytes)
            | QuantizedBlock::Q4K(bytes)
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
    // model"), so this is the SAME correctness argument that section makes
    // for this one encoder's dispatches: they were always ordered and
    // hazard-tracked relative to each other, encoder boundaries or not.
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
        &BTreeSet::new(),
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

/// Zero-fills the first `byte_len` bytes of `buffer`. A freshly allocated
/// `MTLBuffer`'s contents are undefined (`allocate_fault_buffer`'s own
/// doc), and [`allocate_placed_buffer`] never zero-fills -- most callers
/// only ever read positions they themselves already wrote. The
/// `kv-capacity-bucket` padded-tail read is the exception: a decode step
/// binds the KV `Op::Input` leaf to `bucket` rows (`bucket > merged_len`),
/// so `[merged_len, bucket)` is read even though this call never wrote it.
/// `causal_mask_merged` masks that range's attention SCORE to `-inf`
/// exactly (`ScalarOp::Select` picks the constant without reading the
/// garbage key), but the softmax weight it produces (`0.0`) still
/// multiplies the corresponding V row -- `0.0 * garbage` is `0.0` only if
/// the garbage is a normal float; an uninitialized `NaN`/`Inf` bit pattern
/// survives that multiply. `proxima-model-interop`'s
/// `run_decode_loop_placed_kv` calls this once per KV buffer at
/// allocation (not per token): every row a later step ever reads was
/// either zeroed here or overwritten by a REAL rotated key/value this same
/// call wrote first, since `cached_len` only grows.
#[cfg(feature = "metal-output-placement")]
pub fn zero_placed_buffer(buffer: &PlacedBuffer, byte_len: usize) {
    let pointer = buffer.contents();
    // SAFETY: `buffer` is `storageModeShared` (`allocate_placed_buffer`'s
    // own contract) and `byte_len` is the caller's own allocated length
    // for it (mirrors `read_placed_buffer_f32`'s own SAFETY comment), so
    // this is a valid, CPU-visible, mutable byte slice for the duration of
    // this call.
    let slots = unsafe { core::slice::from_raw_parts_mut(pointer.as_ptr().cast::<u8>(), byte_len) };
    slots.fill(0);
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

/// `metal-concurrent-dispatch`'s dataflow-hazard set, generic over the
/// identity type so this logic is testable without a real Metal device
/// (`Id = usize`/`&str` in tests, `Id = *const ProtocolObject<dyn MTLBuffer>`
/// — pointer identity from [`Retained::as_ptr`] — on the real driver path).
/// Tracks two sets since the same buffer position is bound differently on
/// each side of a hazard: `written` names every buffer some op since the
/// last barrier has WRITTEN (a RAW or WAW hazard for a later op that reads
/// or writes it); `read` names every buffer READ since the last barrier (a
/// WAR hazard for a later op that writes it — real here because
/// [`BufferArena`] reuses a whole retired slot's buffer object for a later
/// position). A fresh, never-before-seen buffer (this op's own freshly
/// [`allocate_buffer`]d output) triggers neither: nothing else has touched
/// that pointer yet.
#[cfg(feature = "metal-concurrent-dispatch")]
#[derive(Debug)]
struct HazardTracker<Id: Eq + core::hash::Hash + Copy> {
    written: std::collections::HashSet<Id>,
    read: std::collections::HashSet<Id>,
}

#[cfg(feature = "metal-concurrent-dispatch")]
impl<Id: Eq + core::hash::Hash + Copy> HazardTracker<Id> {
    fn new() -> Self {
        Self {
            written: std::collections::HashSet::new(),
            read: std::collections::HashSet::new(),
        }
    }

    /// True when encoding the next op without a barrier first would let a
    /// concurrent-dispatch-scheduled GPU thread race a still-in-flight one:
    /// RAW (an input was written since the last barrier), WAW (the output
    /// buffer was written since the last barrier), or WAR (the output
    /// buffer was read since the last barrier — arena slot reuse).
    fn needs_barrier(&self, inputs: &[Id], output: Option<Id>) -> bool {
        inputs.iter().any(|input| self.written.contains(input))
            || output.is_some_and(|out| self.written.contains(&out) || self.read.contains(&out))
    }

    /// Clears both sets — called immediately after a barrier is actually
    /// emitted, since the barrier is exactly the guarantee that every
    /// dispatch encoded before it has completed and is visible to every
    /// dispatch encoded after.
    fn reset(&mut self) {
        self.written.clear();
        self.read.clear();
    }

    /// Records this op's own effect, called once per op regardless of
    /// whether a barrier fired for it.
    fn record(&mut self, inputs: &[Id], output: Option<Id>) {
        if let Some(out) = output {
            self.written.insert(out);
        }
        self.read.extend(inputs.iter().copied());
    }

    /// Drops a retired buffer's identity from both sets. Metal's allocator
    /// (and, more aggressively, `metal-buffer-pool`'s own reuse) can hand a
    /// freed address straight back out to a later, unrelated `allocate_buffer`
    /// call — without this, that later buffer would inherit hazard state that
    /// belongs to whatever this address used to be (an ABA on the pointer
    /// identity), not to itself.
    fn forget(&mut self, id: Id) {
        self.written.remove(&id);
        self.read.remove(&id);
    }
}

/// Resolves every hazard-tracked identity for a bound op's operands from
/// `device_buffers`, erroring on the first operand with none instead of
/// silently dropping it from the hazard set (the previous `filter_map`
/// behavior) — an operand missing from `device_buffers` means the encode
/// this identity feeds is already wrong, so hiding it from the hazard check
/// only hides a real bug behind a missing barrier.
#[cfg(feature = "metal-concurrent-dispatch")]
fn resolve_hazard_inputs(
    operands: impl Iterator<Item = NodeId>,
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
) -> Result<Vec<*const ProtocolObject<dyn MTLBuffer>>, MetalError> {
    operands
        .map(|operand| {
            device_buffers
                .get(&operand)
                .map(|(buffer, _offset)| Retained::as_ptr(buffer))
                .ok_or(MetalError::UnresolvedHazardOperand { node: operand })
        })
        .collect()
}

/// The exact per-op hazard step [`execute_plan_with_placements`]'s loop
/// runs: check, barrier-and-reset only on a hazard, then always record —
/// factored out so the tests drive THIS function instead of a hand-written
/// mirror of the loop body that could silently drift from it. `output` is
/// the identity of the buffer this op is about to write, already resolved
/// (placement -> arena slot -> fresh allocation) by the caller before this
/// runs, since every bound op writes exactly one device buffer. Returns
/// whether the caller must emit a barrier before encoding this op.
#[cfg(feature = "metal-concurrent-dispatch")]
fn hazard_step<Id: Eq + core::hash::Hash + Copy>(
    tracker: &mut HazardTracker<Id>,
    inputs: &[Id],
    output: Id,
) -> bool {
    let needs_barrier = tracker.needs_barrier(inputs, Some(output));
    if needs_barrier {
        tracker.reset();
    }
    tracker.record(inputs, Some(output));
    needs_barrier
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
/// With `metal-concurrent-dispatch` on, the paragraph above no longer
/// applies as written: the encoder is opened with
/// `computeCommandEncoderWithDispatchType(Concurrent)` instead of the
/// dispatch-type-less `computeCommandEncoder()`, which turns OFF the
/// automatic whole-resource barrier between every pair of dispatches — two
/// independent ops (e.g. Q/K/V projected from one normed input) now execute
/// with no ordering between them at all unless something inserts one. This
/// function inserts that "something" itself, explicitly, via a private
/// `HazardTracker` walked once per op before it is encoded — see that
/// type's own doc, right above this function, for the three hazards
/// (RAW/WAW/WAR) it covers, and
/// [`MTLBarrierScope::Buffers`] is emitted only where the tracker finds one.
///
/// `PROXIMA_METAL_KIND_FILTER=<kind-substring>[,<kind-substring>]` (or
/// `!<kind-substring>[,<kind-substring>]` for the complement) --
/// `instrument`-gated, default-off, parsed once per
/// [`execute_plan_with_placements`] call by [`KindFilter::from_env`] and
/// consulted per op against [`classify_kind`]'s own return value: the same
/// string a caller already sees in `op_profile_bucket kind=...`, not a
/// second enum this module would have to keep in lockstep with
/// `classify_kind`'s match arms. Unset (`None`) in every production run,
/// which is the ROW's in-buffer ablation harness's own arm-selection knob --
/// see that row for the arm table.
#[cfg(feature = "instrument")]
struct KindFilter {
    substrings: Vec<String>,
    negate: bool,
}

#[cfg(feature = "instrument")]
impl KindFilter {
    fn from_env() -> Option<Self> {
        let raw = std::env::var("PROXIMA_METAL_KIND_FILTER").ok()?;
        let (negate, body) = match raw.strip_prefix('!') {
            Some(rest) => (true, rest),
            None => (false, raw.as_str()),
        };
        let substrings: Vec<String> = body
            .split(',')
            .map(str::trim)
            .filter(|substring| !substring.is_empty())
            .map(String::from)
            .collect();
        Some(Self { substrings, negate })
    }

    fn matches(&self, kind: &str) -> bool {
        let any = self
            .substrings
            .iter()
            .any(|substring| kind.contains(substring.as_str()));
        any != self.negate
    }
}

/// [`KindFilter`]-excluded counterpart of [`encode_op`]'s output-buffer half:
/// called INSTEAD of [`encode_op`] for an op the filter drops, so no
/// pipeline is bound, no buffer is bound to the encoder, and no dispatch is
/// issued -- this op's output buffer keeps whatever bytes it already held (a
/// prior decode step's write, under `metal-plan-stable-buffers`'s stable
/// per-position arena slot; an uninitialized fresh allocation otherwise).
/// `device_buffers` still needs an entry for `bound.node` regardless, or
/// every downstream op that reads it as an operand fails `buffer_for`'s
/// `NotLowerable` lookup before `gpu_exec_ms` is ever read -- the ablation
/// is measuring dispatch time with this op removed, not producing a correct
/// result, but the OTHER ops in the same buffer still need to run.
#[cfg(feature = "instrument")]
fn register_skipped_output(
    device: &ProtocolObject<dyn MTLDevice>,
    device_buffers: &mut BTreeMap<NodeId, DeviceBuffer>,
    bound: &BoundOp,
    placement: Option<(&MetalBuffer, usize)>,
) -> Result<(), MetalError> {
    let (buffer, offset) = match placement {
        Some((buffer, offset)) => (buffer.clone(), offset),
        None => (
            allocate_buffer(device, bound_output_len(bound), bound.dtype)?,
            0,
        ),
    };
    device_buffers.insert(bound.node, (buffer, offset));
    Ok(())
}

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
        // an input-placed node above never reaches here, so `BLOCK_OFFERED_BYTES`
        // fires only for a block that genuinely takes the host round trip
        // below -- the same "offered for upload" meaning `execute_plan`'s own
        // fire site carries, extended to this placed-buffer entry point so
        // the partition identity (`BLOCK_COPIED_BYTES + BLOCK_NOCOPY_BOUND_BYTES
        // + BLOCK_OFFSET_BOUND_BYTES == BLOCK_OFFERED_BYTES`) holds on this
        // loop too, not just `execute_plan`'s.
        #[cfg(feature = "instrument")]
        {
            counter!(BLOCK_UPLOAD_CALLS, 1);
            counter!(BLOCK_OFFERED_BYTES, block_byte_len(block) as u64);
        }
        let resident = plan.resident_nodes.contains(node);
        let buffer = match block {
            QuantizedBlock::Float32(data) => upload_block(&device, data, *node, *dtype, resident)?,
            // `Q3_K` uploads its raw super-block bytes unchanged, same as
            // every other packed codec below -- `msl::PackedCodec::Q3K`'s
            // own unpack kernel (`q3k_element`) reads them at the GPU side.
            QuantizedBlock::Q3K(bytes)
            | QuantizedBlock::Q4K(bytes)
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
    #[cfg(not(feature = "metal-concurrent-dispatch"))]
    let encoder =
        command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| MetalError::CompileFailed {
                log: "command buffer refused to hand out a compute encoder".to_string(),
            })?;
    // `Concurrent` lets independent dispatches (e.g. Q/K/V from one normed
    // input) overlap instead of draining the pipeline between every op --
    // see this function's own doc for why that requires [`HazardTracker`]
    // below to insert the barriers the `Serial` type used to give for free.
    #[cfg(feature = "metal-concurrent-dispatch")]
    let encoder = command_buffer
        .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
        .ok_or_else(|| MetalError::CompileFailed {
            log: "command buffer refused to hand out a concurrent compute encoder".to_string(),
        })?;
    #[cfg(feature = "metal-concurrent-dispatch")]
    let mut hazards: HazardTracker<*const ProtocolObject<dyn MTLBuffer>> = HazardTracker::new();

    // diagnostic-only, `instrument`-gated, always emitted at `trace` level
    // (default-off via the runtime filter, never a bespoke env var): each
    // output-placed node's WRITE position and each of its aliased readers'
    // own position, the direct evidence a write-before-read ordering claim
    // needs (this function's own doc, "Within-call aliasing"). Silent unless
    // a caller raises `RUST_LOG` to `trace` for this target.
    #[cfg(feature = "instrument")]
    let kind_filter = KindFilter::from_env();
    let mut pending_faults: Vec<PendingFault<'_>> = Vec::new();
    for (position, bound) in prepared.resolved.iter().enumerate() {
        #[cfg(feature = "instrument")]
        {
            if output_placed.contains_key(&bound.node) {
                trace!(position, node = ?bound.node, "output-placed node write");
            }
            for (operand, _, _) in bound.operands() {
                if input_placed.contains_key(operand) {
                    trace!(position, node = ?bound.node, reads = ?operand, "placed-input node read");
                }
            }
        }
        let placement = match output_placed.get(&bound.node).copied() {
            Some(placement) => Some(placement),
            None => arena_placement(plan, position)?,
        };
        // `ablation_skip` is `false` on every non-`instrument` build (the
        // `match` folds to the literal at compile time, so this costs
        // nothing in production) and `false` on every `instrument` build
        // where `PROXIMA_METAL_KIND_FILTER` is unset -- the only way into
        // the `register_skipped_output` arm below is an operator explicitly
        // setting that env var, which never happens outside this ablation's
        // own harness.
        #[cfg(feature = "instrument")]
        let ablation_skip = match &kind_filter {
            Some(filter) => !filter.matches(classify_kind(bound, packed_operands)),
            None => false,
        };
        #[cfg(not(feature = "instrument"))]
        let ablation_skip = false;

        if ablation_skip {
            // `ablation_skip` is always `false` on a non-`instrument` build
            // (see its own binding above), so this arm never runs there --
            // `register_skipped_output` itself is `instrument`-gated and
            // does not exist to call outside it.
            #[cfg(feature = "instrument")]
            register_skipped_output(&device, &mut device_buffers, bound, placement)?;
        } else {
            // A skipped op above never reaches this hazard tracking either
            // -- it neither reads nor writes a buffer THIS command buffer
            // touches, so the tracker must see it as absent: no
            // `hazard_inputs`/`hazard_output` collected, no
            // `needs_barrier`/`record` call, no barrier emitted on its
            // account.
            #[cfg(feature = "metal-concurrent-dispatch")]
            let hazard_inputs = resolve_hazard_inputs(
                bound.operands().iter().map(|(operand, _, _)| *operand),
                &device_buffers,
            )?;
            // resolved ONCE, before the hazard check, from the exact same
            // placement -> arena -> fresh-allocation chain `encode_op` would
            // otherwise pick independently below: `needs_barrier` and
            // `record` used to see two different identities for a fresh
            // allocation (the check saw `None`, since nothing is known
            // before `encode_op` allocates; the record afterward saw the
            // real pointer), so a WAW/WAR hazard against an address Metal
            // happens to reuse for that allocation could never be caught.
            // Allocating here and handing the SAME buffer into `encode_op`
            // via `placement` closes that gap.
            #[cfg(feature = "metal-concurrent-dispatch")]
            let resolved_output: DeviceBuffer = match placement {
                Some((buffer, offset)) => (buffer.clone(), offset),
                None => (allocate_buffer(&device, bound_output_len(bound), bound.dtype)?, 0),
            };
            #[cfg(feature = "metal-concurrent-dispatch")]
            let hazard_output = Retained::as_ptr(&resolved_output.0);
            #[cfg(feature = "metal-concurrent-dispatch")]
            if hazard_step(&mut hazards, &hazard_inputs, hazard_output) {
                encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                counter!(BARRIERS_EMITTED, 1);
            }
            #[cfg(feature = "metal-concurrent-dispatch")]
            let placement = Some((&resolved_output.0, resolved_output.1));
            let uniform_buffer = plan_uniform_buffer(plan, position)?;
            let fault = encode_op(
                &device,
                &encoder,
                &mut device_buffers,
                bound,
                packed_operands,
                placement,
                uniform_buffer,
            )?;
            if let Some((fault_buffer, gathers)) = fault {
                pending_faults.push((bound, fault_buffer, gathers));
            }
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
            // a retired buffer's identity must not outlive it in the
            // tracker: Metal (and `metal-buffer-pool` more aggressively) can
            // hand this exact address back out to a later, unrelated
            // `allocate_buffer` call, and that later buffer must start with
            // no hazard history -- see `HazardTracker::forget`'s own doc.
            #[cfg(feature = "metal-concurrent-dispatch")]
            if let Some((buffer, _offset)) = device_buffers.remove(retired) {
                hazards.forget(Retained::as_ptr(&buffer));
            }
            #[cfg(not(feature = "metal-concurrent-dispatch"))]
            device_buffers.remove(retired);
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

    let placed_output_nodes: BTreeSet<NodeId> = output_placed.keys().copied().collect();
    finish(
        &plan.program,
        &prepared.index_nodes,
        &prepared.shapes,
        &prepared.effective_outputs,
        &device_buffers,
        prepared.root,
        &placed_output_nodes,
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
    /// This operand's TENSOR bytes -- `element_count(shape) * bytes_per_element`,
    /// where `bytes_per_element` is the operand's own dtype width for a plain
    /// buffer, or its `PackedCodec::block_bytes`/block-elements ratio for a
    /// packed one. See `operand_tensor_bytes`. Distinct from
    /// [`Self::bound_buffer_bytes`]: since a checkpoint mapping upload binds
    /// ONE buffer spanning the whole mmap (`checkpoint_mapping_offset`), that
    /// buffer's own `length()` overstates every individual operand sharing it
    /// -- this field is what a per-tensor byte-share table needs instead.
    pub operand_bytes: u64,
    /// The device buffer's own `length()` this operand was bound against --
    /// the value `operand_bytes` used to report before it was corrected to
    /// the tensor's own byte count. Kept so a checkpoint-mapping-offset bind
    /// (one shared buffer, `bound_buffer_bytes` far larger than
    /// `operand_bytes`) stays observable rather than silently disappearing.
    pub bound_buffer_bytes: u64,
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

/// Shared per-op timing dispatch: [`execute_plan_op_timed`] and
/// [`execute_plan_with_placements_op_timed`] both need to encode ONE
/// `BoundOp` on its own command buffer, commit, wait, and read back
/// `GPUStartTime`/`GPUEndTime` -- factored here so the two op-timed
/// executors cannot drift on the [`OpGpuTiming`] fields or the format
/// `proxima-model-interop/src/generate.rs`'s `report_op_timings` prints
/// from them. `placement` mirrors [`encode_op`]'s own parameter -- `None`
/// for the plain (unplaced) op-timed path, `Some((buffer, offset))` for
/// the placed one. `always_live` names every node THIS call's own
/// placement maps hold (input- or output-placed): those are excluded from
/// this op's own retire sweep the same way
/// [`execute_plan_with_placements`]'s own loop excludes them, so the
/// unplaced caller passes an empty set and every `prepared.retires` entry
/// is dropped exactly as it always was.
#[cfg(feature = "instrument")]
#[allow(clippy::too_many_arguments)]
fn execute_op_timed(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    device_buffers: &mut BTreeMap<NodeId, DeviceBuffer>,
    prepared: &Prepared,
    packed_operands: &PackedOperands,
    program: &[Op],
    position: usize,
    bound: &BoundOp,
    placement: Option<(&MetalBuffer, usize)>,
    plan_uniform: Option<&MetalBuffer>,
    always_live: &BTreeSet<NodeId>,
) -> Result<OpGpuTiming, MetalError> {
    // this operand's own TENSOR bytes, not the shared buffer's `length()` --
    // see `operand_tensor_bytes`'s own doc: a checkpoint-mapping-offset bind
    // shares ONE buffer across every packed weight, so `buffer.length()`
    // (kept below as `bound_buffer_bytes`) overstates every individual
    // operand sharing it.
    let operand_bytes: u64 = bound
        .operands()
        .iter()
        .map(|(source, _, _)| {
            operand_tensor_bytes(
                program,
                &prepared.index_nodes,
                &prepared.shapes,
                packed_operands,
                *source,
            )
        })
        .sum();
    let bound_buffer_bytes: u64 = bound
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
        .find_map(|(source, _, _)| program[source.0 as usize].name())
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
        device,
        &encoder,
        device_buffers,
        bound,
        packed_operands,
        placement,
        plan_uniform,
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
        if always_live.contains(retired) {
            continue;
        }
        device_buffers.remove(retired);
    }
    Ok(OpGpuTiming {
        node: bound.node,
        kind,
        operand_bytes,
        bound_buffer_bytes,
        gpu_ns,
        weight_name,
        operand_count: bound.operands().len(),
        packed_codec,
        packed_kernel_variant,
        packed_row_block_rejection,
    })
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
            // `Q3_K` uploads its raw super-block bytes unchanged, same as
            // every other packed codec below -- `msl::PackedCodec::Q3K`'s
            // own unpack kernel (`q3k_element`) reads them at the GPU side.
            QuantizedBlock::Q3K(bytes)
            | QuantizedBlock::Q4K(bytes)
            | QuantizedBlock::Q5K(bytes)
            | QuantizedBlock::Q6K(bytes)
            | QuantizedBlock::Q8_0(bytes)
            | QuantizedBlock::Q4_0(bytes)
            | QuantizedBlock::Float16(bytes)
            | QuantizedBlock::BFloat16(bytes) => upload_packed_bytes(&device, bytes, resident)?,
        };
        device_buffers.insert(*node, buffer);
    }

    let no_placements: BTreeSet<NodeId> = BTreeSet::new();
    let mut timings: Vec<OpGpuTiming> = Vec::with_capacity(prepared.resolved.len());
    for (position, bound) in prepared.resolved.iter().enumerate() {
        let timing = execute_op_timed(
            &device,
            &queue,
            &mut device_buffers,
            prepared,
            packed_operands,
            &plan.program,
            position,
            bound,
            None,
            None,
            &no_placements,
        )?;
        timings.push(timing);
    }

    let evaluated = finish(
        &plan.program,
        &prepared.index_nodes,
        &prepared.shapes,
        &prepared.effective_outputs,
        &device_buffers,
        prepared.root,
        &BTreeSet::new(),
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

/// [`execute_plan_with_placements`]'s op-timed twin -- the default decode
/// path (`run_decode_loop_placed_kv` in
/// `proxima-model-interop/src/generate.rs`) routes through
/// `execute_plan_with_placements`, never `execute_plan`, so
/// [`execute_plan_op_timed`] alone cannot attribute GPU time on that path:
/// it always builds fresh output buffers (`placement: None` at its own call
/// site above) and so never exercises the placed-KV write/read shape at
/// all. This shares every input/output-placement resolution step
/// `execute_plan_with_placements` itself performs (input-placed nodes skip
/// the host upload identically; `always_live` reproduces that function's
/// own retirement exclusion) and swaps only its single shared-command-buffer
/// submission for this module's own shared per-op dispatch helper, same relationship
/// [`execute_plan_op_timed`] already has to [`execute_plan`].
///
/// # Errors
/// Same as [`execute_plan_with_placements`].
#[cfg(all(feature = "metal-output-placement", feature = "instrument"))]
pub fn execute_plan_with_placements_op_timed(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
) -> Result<(Evaluated, Vec<OpGpuTiming>), MetalError> {
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
    let always_live: BTreeSet<NodeId> = input_placed
        .keys()
        .chain(output_placed.keys())
        .copied()
        .collect();
    let (device, queue) = device_and_queue()?;

    let mut device_buffers: BTreeMap<NodeId, DeviceBuffer> = BTreeMap::new();
    for ((node, block), dtype) in prepared
        .block_nodes
        .iter()
        .zip(blocks.iter())
        .zip(plan.block_dtypes.iter())
    {
        if let Some((buffer, offset)) = input_placed.get(node) {
            device_buffers.insert(*node, ((*buffer).clone(), *offset));
            continue;
        }
        // same `BLOCK_OFFERED_BYTES`/`BLOCK_UPLOAD_CALLS` fire
        // `execute_plan_with_placements`'s own loop carries -- this
        // diagnostic op-timed twin duplicates that loop's upload shape, so
        // it must duplicate the census fire too, or the partition identity
        // (`BLOCK_COPIED_BYTES + BLOCK_NOCOPY_BOUND_BYTES +
        // BLOCK_OFFSET_BOUND_BYTES == BLOCK_OFFERED_BYTES`) goes untested on
        // this path even though the three per-path counters (fired inside
        // `upload_block_as_float`/`upload_packed_bytes` themselves) still
        // increment here regardless.
        #[cfg(feature = "instrument")]
        {
            counter!(BLOCK_UPLOAD_CALLS, 1);
            counter!(BLOCK_OFFERED_BYTES, block_byte_len(block) as u64);
        }
        let resident = plan.resident_nodes.contains(node);
        let buffer = match block {
            QuantizedBlock::Float32(data) => upload_block(&device, data, *node, *dtype, resident)?,
            // `Q3_K` uploads its raw super-block bytes unchanged, same as
            // every other packed codec below -- `msl::PackedCodec::Q3K`'s
            // own unpack kernel (`q3k_element`) reads them at the GPU side.
            QuantizedBlock::Q3K(bytes)
            | QuantizedBlock::Q4K(bytes)
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
        let placement = match output_placed.get(&bound.node).copied() {
            Some(placement) => Some(placement),
            None => arena_placement(plan, position)?,
        };
        let uniform_buffer = plan_uniform_buffer(plan, position)?;
        let timing = execute_op_timed(
            &device,
            &queue,
            &mut device_buffers,
            prepared,
            packed_operands,
            &plan.program,
            position,
            bound,
            placement,
            uniform_buffer,
            &always_live,
        )?;
        timings.push(timing);
    }

    let placed_output_nodes: BTreeSet<NodeId> = output_placed.keys().copied().collect();
    let evaluated = finish(
        &plan.program,
        &prepared.index_nodes,
        &prepared.shapes,
        &prepared.effective_outputs,
        &device_buffers,
        prepared.root,
        &placed_output_nodes,
    )?;
    Ok((evaluated, timings))
}

/// [`execute_plan_with_placements_op_timed`] against a name-keyed block
/// set, mirroring [`execute_plan_named_with_placements`]'s own name
/// resolution.
///
/// # Errors
/// Propagates name-resolution and Metal driver failures.
#[cfg(all(feature = "metal-output-placement", feature = "instrument"))]
pub fn execute_plan_named_with_placements_op_timed(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
) -> Result<(Evaluated, Vec<OpGpuTiming>), MetalError> {
    let blocks = resolve_named_blocks(&plan.program, named)?;
    execute_plan_with_placements_op_timed(plan, &blocks, input_placements, output_placements)
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
                    || kernel.source.contains("q5k_pair_dot(blk")
                    || kernel.source.contains("q5k_value(blk")
                    || kernel.source.contains("q6k_value(blk")
                    // `metal-q4k-ggml-port`'s own body (`push_q4k_ggml_port_body`)
                    // has none of the above markers -- it never calls this
                    // crate's own decode helpers, that is the whole point of
                    // the port -- and it DOES end in a `simd_sum(` combine
                    // like every other cooperative-reduce kernel, so without
                    // this arm it fell through to "reduce-cooperative" below
                    // and the op-profile bucket undercounted packed-row-blocked
                    // ops by exactly the ggml-port op count (found bake-off
                    // measuring this landing: `reduce-packed-row-blocked`
                    // dropped from 225 to 9 ops, `reduce-cooperative` grew by
                    // the same 216, with `packed_row_block`'s own per-family
                    // `row_blocked_count` unchanged at 32 per family --
                    // dispatch was always correct, only this profiler label
                    // was wrong). `acc1_0` is unique to that body's per-thread
                    // accumulator naming.
                    || kernel.source.contains("acc1_0") =>
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
    } else if kernel.source.contains("q5k_pair_dot(blk") {
        "q5k-paired"
    } else if kernel.source.contains("q5k_value(blk") {
        "q5k-scalar"
    } else if kernel.source.contains("q6k_value(blk") {
        "q6k-scalar"
    } else if kernel.source.contains("acc1_0") {
        // `metal-q4k-ggml-port`'s own body -- same marker and same reason
        // as `classify_kind`'s own arm above, so this variant name does not
        // fall into "other" alongside genuinely unclassified kernels.
        "q4k-ggml-port"
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
/// every `QuantizedBlock` variant has a real Metal path (`Q3_K` was the
/// last arm that could still fail here); kept returning a `Result`
/// regardless, so a future codec added without an entry here is still a
/// typed error rather than a silent miscount.
fn block_element_count(block: &QuantizedBlock<'_>) -> Result<usize, MetalError> {
    match block {
        QuantizedBlock::Float32(data) => Ok(data.len()),
        // `Q3_K`'s super-block is 110 bytes carrying the same 256-element
        // count as the rest of the K-quant family (`Q4_K`/`Q5_K`/`Q6_K`) --
        // see `crate::msl::Q4K_BLOCK_ELEMENTS`'s own doc.
        QuantizedBlock::Q3K(bytes) => {
            Ok((bytes.len() / crate::msl::Q3K_BLOCK_BYTES) * crate::msl::Q4K_BLOCK_ELEMENTS)
        }
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
        QuantizedBlock::Q3K(bytes)
        | QuantizedBlock::Q4K(bytes)
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

/// `source`'s own TENSOR byte count -- `element_count(shape) *
/// bytes_per_element`, where `bytes_per_element` is exact for a plain buffer
/// ([`DType::size_bytes`]) and a `block_bytes / block_elements` ratio for a
/// packed operand ([`PackedCodec::block_bytes`]/`block_elements`). This is
/// the value a per-operand byte-share table needs -- NOT
/// `device_buffers[source].0.length()`, which reports the shared checkpoint-
/// mapping buffer's own size for every tensor `checkpoint_mapping_offset`
/// binds into it (see [`OpGpuTiming::bound_buffer_bytes`]'s own doc).
#[cfg(feature = "instrument")]
fn operand_tensor_bytes(
    program: &[Op],
    index_nodes: &BTreeSet<NodeId>,
    shapes: &Shapes,
    packed_operands: &PackedOperands,
    source: NodeId,
) -> u64 {
    let elements = element_count(shapes.of(source)) as u64;
    match packed_operands.get(&source) {
        Some(codec) => elements * codec.block_bytes() as u64 / codec.block_elements() as u64,
        None => elements * gpu_dtype(program, index_nodes, source).size_bytes() as u64,
    }
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

/// Bytes [`push_gather_uniforms`] appends for `gather_count` gathered
/// operands at `rank_len` — `0` operands write nothing (the early return in
/// [`push_gather_uniforms`] itself), otherwise 3 scalar `i64` fields
/// (`gather_index_base`/`gather_element_stride`/`gather_extent`) plus one
/// `rank_len`-wide `i64` row (`gather_index_strides`) PER gathered operand.
fn gather_uniform_byte_len(gather_count: usize, rank_len: usize) -> usize {
    if gather_count == 0 {
        0
    } else {
        gather_count * (rank_len + 3) * size_of::<i64>()
    }
}

/// [`pack_uniforms`]'s own byte length, computed from `bound`'s static shape
/// (extents, operand count, gather count) rather than by actually building
/// the byte vector — every field [`pack_uniforms`]'s packers write is a
/// scalar `i64` or an `i64` row of fixed width ([`push_i64`]/[`push_i64_row`]
/// never branch on the VALUES, only on `rank_len`/`width`), so the total byte
/// count is a pure function of shape. [`build_buffer_arena`]'s own
/// `uniform_bytes` diagnostic sum used to call [`pack_uniforms`] once per op
/// just to read `.len()` off the result, allocating and filling a real byte
/// vector — for every op in the plan — purely to throw it away; this
/// mirrors [`pack_uniforms`]'s own match arms field-for-field instead.
fn pack_uniforms_byte_len(bound: &BoundOp) -> usize {
    const WORD: usize = size_of::<i64>();
    let rank_len = bound.extents.len().max(1);
    let operand_count = bound.operands().len();
    let gather = gather_count(bound);

    match &bound.kind {
        BoundOpKind::CachedAttention { .. } => WORD,
        BoundOpKind::Iota | BoundOpKind::Constant { .. } => WORD,
        BoundOpKind::Elementwise { .. } => {
            (1 + rank_len + operand_count + operand_count * rank_len) * WORD
                + gather_uniform_byte_len(gather, rank_len)
        }
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            output_axes,
            epilogue_operands,
            ..
        } => {
            let output_rank_len = output_axes.len().max(1);
            let reduce_rank_len = reduction_dims(bound, output_axes).len().max(1);
            let epilogue_len = if epilogue_operands.is_empty() {
                0
            } else {
                epilogue_operands.len() * (1 + output_rank_len)
            };
            (2 + output_rank_len
                + reduce_rank_len
                + operand_count
                + operand_count * rank_len
                + 1
                + rank_len
                + epilogue_len)
                * WORD
                + gather_uniform_byte_len(gather, rank_len)
        }
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => {
            let outer_rank_len = bound.extents.len().saturating_sub(1).max(1);
            (2 + outer_rank_len + operand_count + operand_count * rank_len + 1 + rank_len) * WORD
                + gather_uniform_byte_len(gather, rank_len)
        }
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod pack_uniforms_byte_len_tests {
    //! [`super::pack_uniforms_byte_len`] mirrors [`super::pack_uniforms`]'s
    //! match arms field-for-field rather than calling it -- these tests are
    //! the parity proof: for a real elementwise op and a real `Keep::Reduce`
    //! matmul-shaped op (the two arms `build_buffer_arena`'s hot loop
    //! actually walks on every real forward), the analytically-computed
    //! length must equal the real byte vector's `.len()` exactly.

    use alloc::vec;
    use alloc::vec::Vec;

    use proxima_tensor::{
        BoundOp, BoundOpKind, DType, Extent, IndexMap, Keep, Op, Reduce, ReduceInit, ScalarOp,
        append, bind, infer, map,
    };

    use super::{pack_uniforms, pack_uniforms_byte_len};

    fn elementwise_add_op(extent_a: u32, extent_b: u32) -> BoundOp {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent_a), Extent::Static(extent_b)],
                name: None,
            },
        );
        let rhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent_a), Extent::Static(extent_b)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: vec![
                    (lhs, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (rhs, IndexMap::Affine(map::projection(2, &[0, 1]))),
                ],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("elementwise infers");
        bind(&program, &shapes, &[])
            .expect("elementwise lowers")
            .into_iter()
            .next_back()
            .expect("one bound op emitted")
    }

    fn matmul_reduce_op(m: u32, k: u32, n: u32) -> BoundOp {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(m), Extent::Static(k)],
                name: None,
            },
        );
        let rhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(k), Extent::Static(n)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
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
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: Some("matmul".into()),
            }),
        );
        let shapes = infer(&program, &[]).expect("matmul infers");
        bind(&program, &shapes, &[])
            .expect("matmul lowers")
            .into_iter()
            .next_back()
            .expect("one fused bound op emitted")
    }

    #[test]
    fn elementwise_byte_len_matches_the_real_packed_bytes() {
        let bound = elementwise_add_op(4, 3);
        assert!(
            matches!(bound.kind, BoundOpKind::Elementwise { .. }),
            "fixture must actually lower to an Elementwise BoundOp"
        );
        assert_eq!(pack_uniforms_byte_len(&bound), pack_uniforms(&bound).len());
    }

    #[test]
    fn reduce_byte_len_matches_the_real_packed_bytes() {
        let bound = matmul_reduce_op(2, 5, 3);
        assert!(
            matches!(
                bound.kind,
                BoundOpKind::Reduce {
                    keep: Keep::Reduce,
                    ..
                }
            ),
            "fixture must actually lower to a Keep::Reduce BoundOp"
        );
        assert_eq!(pack_uniforms_byte_len(&bound), pack_uniforms(&bound).len());
    }
}

fn pack_cached_attention_uniforms(bound: &BoundOp) -> Vec<u8> {
    let BoundOpKind::CachedAttention { head_dim, .. } = &bound.kind else {
        unreachable!("cached attention uniform packer only receives cached attention")
    };
    let total: i64 = bound
        .extents
        .iter()
        .map(|extent| *extent as i64)
        .product::<i64>()
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
        epilogue_operands,
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
    // `crate::msl::render_reduce`'s own `Uniforms` struct declares these
    // fields ONLY when `epilogue_operands` is non-empty (byte-identical to
    // before epilogue fusion existed otherwise), so this must stay
    // conditional on the exact same test.
    if !epilogue_operands.is_empty() {
        for (_, layout, _) in epilogue_operands {
            push_i64(&mut bytes, layout.base);
        }
        for (_, layout, _) in epilogue_operands {
            push_i64_row(&mut bytes, &layout.strides, output_rank_len);
        }
    }
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
        trace!(cache_key = %cache_key, hit = true, "pipeline cache lookup");
        #[cfg(feature = "instrument")]
        counter!(PIPELINE_HITS, 1);
        return Ok(pipeline);
    }
    trace!(cache_key = %cache_key, hit = false, "pipeline cache lookup");
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
    counter!(OUTPUT_BUFFER_ALLOCATIONS, 1);
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
    counter!(OUTPUT_BUFFER_ALLOCATIONS, 1);
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
pub static BLOCK_OFFERED_BYTES: Counter = Counter::new("omega.metal.block_offered_bytes");
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
    /// Every block's own declared byte length, summed regardless of which
    /// terminal upload path served it -- see [`BLOCK_OFFERED_BYTES`]'s own
    /// doc. `block_copied_bytes + block_nocopy_bound_bytes +
    /// block_offset_bound_bytes == block_offered_bytes` on every step is the
    /// partition identity this card exists to prove.
    pub block_offered_bytes: u64,
    /// Bytes bound through a real host->device copy this step -- see
    /// [`BLOCK_COPIED_BYTES`]'s own doc.
    pub block_copied_bytes: u64,
    /// Bytes bound zero-copy (no-copy path, cached or uncached) this step --
    /// see [`BLOCK_NOCOPY_BOUND_BYTES`]'s own doc.
    pub block_nocopy_bound_bytes: u64,
    /// Bytes bound at an offset into the shared checkpoint-mapping buffer
    /// this step -- see [`BLOCK_OFFSET_BOUND_BYTES`]'s own doc.
    pub block_offset_bound_bytes: u64,
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
    /// CARD 6.5 census: [`OUTPUT_BUFFER_ALLOCATIONS`]'s own per-step delta --
    /// `op_count` every step with `metal-plan-stable-buffers` off, `op_count`
    /// only on the step that builds a plan (a plan-cache miss) and 0 on
    /// every following plan-cache-hit step with it on.
    pub output_buffer_allocations: u64,
    /// CARD 6.5 census: [`PLAN_UNIFORM_WRITES`]'s own per-step delta --
    /// `op_count` every step on the `metal-plan-stable-buffers` path
    /// (`encode_op` writes every position's uniforms in place every call),
    /// 0 always with the feature off.
    pub plan_uniform_writes: u64,
    /// [`BARRIERS_EMITTED`]'s own per-step delta -- 0 on the serial-dispatch
    /// arm (`metal-concurrent-dispatch` off), the count of dataflow hazards
    /// the private `HazardTracker` actually found on the concurrent-dispatch
    /// arm.
    pub barriers_emitted: u64,
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
        block_offered_bytes: BLOCK_OFFERED_BYTES.snapshot_and_reset(),
        block_copied_bytes: BLOCK_COPIED_BYTES.snapshot_and_reset(),
        block_nocopy_bound_bytes: BLOCK_NOCOPY_BOUND_BYTES.snapshot_and_reset(),
        block_offset_bound_bytes: BLOCK_OFFSET_BOUND_BYTES.snapshot_and_reset(),
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
        output_buffer_allocations: OUTPUT_BUFFER_ALLOCATIONS.snapshot_and_reset(),
        plan_uniform_writes: PLAN_UNIFORM_WRITES.snapshot_and_reset(),
        barriers_emitted: BARRIERS_EMITTED.snapshot_and_reset(),
    }
}

/// How many real `upload_block` calls took each host->device path —
/// incremented once per call, never per byte, so a caller can read back the
/// no-copy hit rate after a run without an external profiler. See the
/// module doc's "Host buffer upload" section.
pub static NOCOPY_BUFFER_UPLOADS: Counter = Counter::new("omega.metal.upload_block.nocopy");
pub static COPYING_BUFFER_UPLOADS: Counter = Counter::new("omega.metal.upload_block.copy");

/// Bytes bound through a real host->device copy this call
/// (`upload_block_copy` or `upload_resident_copy`) — the terminal-path
/// byte split of [`BLOCK_OFFERED_BYTES`], fired at the same granularity (once
/// per block, every call, cache hit or miss) so the three counters below sum
/// to `BLOCK_OFFERED_BYTES` on every step. See the module doc's "Host buffer
/// upload" section for why most weight bytes never reach this counter --
/// only a misaligned, non-resident block (the KV cache, which is deliberately
/// never cached: see `upload_block_copy`'s own doc) pays a real copy every
/// token.
pub static BLOCK_COPIED_BYTES: Counter = Counter::new("omega.metal.block_copied_bytes");
/// Bytes bound zero-copy, either `upload_block_no_copy` (cached, resident)
/// or `upload_block_no_copy_uncached` (uncached) — the other terminal-path
/// byte split of [`BLOCK_OFFERED_BYTES`].
pub static BLOCK_NOCOPY_BOUND_BYTES: Counter = Counter::new("omega.metal.block_nocopy_bound_bytes");
/// Bytes bound at an offset into the single whole-checkpoint no-copy buffer
/// (`checkpoint_mapping_offset`) — the third terminal-path byte split of
/// [`BLOCK_OFFERED_BYTES`], and the one that carries the bulk of a real
/// model's weight bytes once `register_checkpoint_mapping` is in effect.
pub static BLOCK_OFFSET_BOUND_BYTES: Counter = Counter::new("omega.metal.block_offset_bound_bytes");

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
        counter!(BLOCK_NOCOPY_BOUND_BYTES, byte_length as u64);
        if resident {
            return upload_block_no_copy(device, pointer, byte_length).map(|buffer| (buffer, 0));
        }
        return upload_block_no_copy_uncached(device, pointer, byte_length)
            .map(|buffer| (buffer, 0));
    }
    if let Some(result) = checkpoint_mapping_offset(device, pointer, byte_length) {
        return result;
    }
    counter!(BLOCK_COPIED_BYTES, byte_length as u64);
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
        counter!(BLOCK_NOCOPY_BOUND_BYTES, byte_length as u64);
        if resident {
            return upload_block_no_copy(device, pointer, byte_length).map(|buffer| (buffer, 0));
        }
        return upload_block_no_copy_uncached(device, pointer, byte_length)
            .map(|buffer| (buffer, 0));
    }
    if let Some(result) = checkpoint_mapping_offset(device, pointer, byte_length) {
        return result;
    }
    counter!(BLOCK_COPIED_BYTES, byte_length as u64);
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
    counter!(BLOCK_OFFSET_BOUND_BYTES, byte_length as u64);
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
    ///
    /// Bounded to `crate::sized::UNIFORM_CACHE_ENTRIES` with least-recently-
    /// used eviction (the `u64` tick alongside each buffer) rather than left
    /// to grow forever: a workload whose uniform bytes vary per call
    /// (different shapes, different `cached_len` without bucketing) would
    /// otherwise retain one `MTLBuffer` per distinct blob ever seen.
    static UNIFORM_BUFFERS: RefCell<BTreeMap<Vec<u8>, (MetalBuffer, u64)>> =
        RefCell::new(BTreeMap::new());

    /// Monotonic use counter driving LRU eviction -- incremented on every
    /// hit and every insert, so the entry with the smallest stored tick is
    /// always the one least recently touched.
    static UNIFORM_CACHE_CLOCK: RefCell<u64> = const { RefCell::new(0) };
}

/// Counts uniform buffers served from cache rather than allocated.
pub static UNIFORM_BUFFER_REUSES: Counter = Counter::new("omega.metal.uniforms.reuse");

/// CARD 6.5's census counter: every genuinely fresh device buffer
/// `allocate_buffer` hands out, on ANY path (the classic per-op-per-call
/// path below, or `build_buffer_arena`'s own size-class-miss path). Not
/// gated behind `metal-plan-stable-buffers` -- this counter's whole point is
/// to read the SAME number on both arms of the bake-off: `op_count` every
/// step with the feature off, `op_count` once (at the plan-cache miss that
/// builds the arena) and 0 on every following plan-cache-hit step with it on.
pub static OUTPUT_BUFFER_ALLOCATIONS: Counter =
    Counter::new("omega.metal.output_buffer.allocations");

/// CARD 6.5's census counter: every in-place write `encode_op` makes into a
/// `PlanUniforms` buffer, bypassing `upload_uniforms`/`UNIFORM_BUFFERS`
/// entirely. Fires only on the `metal-plan-stable-buffers` path; stays 0 with
/// the feature off (or on any position `execute_plan`'s non-placed path
/// dispatches, which never receives a plan-owned uniform buffer).
pub static PLAN_UNIFORM_WRITES: Counter = Counter::new("omega.metal.plan_uniforms.write");

/// `metal-concurrent-dispatch`'s own census: every
/// `memoryBarrierWithScope(Buffers)` the private `HazardTracker` actually
/// emitted this step. Not gated behind that feature at declaration -- same
/// convention as
/// [`OUTPUT_BUFFER_ALLOCATIONS`] above -- so it reads a stable 0 on the
/// serial-dispatch arm rather than not existing at all.
pub static BARRIERS_EMITTED: Counter = Counter::new("omega.metal.concurrent.barriers_emitted");

/// Entries `UNIFORM_BUFFERS` holds right now -- the direct witness for D6
/// (round-4 synth S2): a caller that wants to know whether the cache grows
/// across a decode run reads this once per step rather than inferring
/// growth from `nocopy_cache_len`'s unrelated bound. Growing after the
/// plan-cache warms (roughly `op_count` per distinct token position, since
/// every `Uniforms` blob carries `reduction_total`, itself a function of
/// `cached_len`) up to `crate::sized::UNIFORM_CACHE_ENTRIES` is the
/// pre-registered prediction this counter exists to check; it never exceeds
/// that capacity.
#[must_use]
pub fn uniform_cache_len() -> usize {
    UNIFORM_BUFFERS.with(|cache| cache.borrow().len())
}

/// Evicts the entry with the smallest use-tick, making room for one more
/// insert. The map is bounded to `crate::sized::UNIFORM_CACHE_ENTRIES`
/// entries by construction, so a linear scan over its current contents to
/// find the minimum tick is cheap -- no ordered secondary index is needed
/// for a map this small.
fn evict_least_recently_used(cache: &mut BTreeMap<Vec<u8>, (MetalBuffer, u64)>) {
    let Some(oldest_key) = cache
        .iter()
        .min_by_key(|(_, (_, tick))| *tick)
        .map(|(key, _)| key.clone())
    else {
        return;
    };
    cache.remove(&oldest_key);
}

fn upload_uniforms(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let next_tick = UNIFORM_CACHE_CLOCK.with(|clock| {
        let mut clock = clock.borrow_mut();
        *clock += 1;
        *clock
    });

    if let Some(existing) = UNIFORM_BUFFERS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let hit = cache.get(bytes).map(|(buffer, _)| buffer.clone());
        if let Some(buffer) = &hit {
            cache.insert(bytes.to_vec(), (buffer.clone(), next_tick));
        }
        hit
    }) {
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
    UNIFORM_BUFFERS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let capacity = crate::sized::UNIFORM_CACHE_ENTRIES as usize;
        if cache.len() >= capacity && !cache.contains_key(bytes) {
            evict_least_recently_used(&mut cache);
        }
        cache.insert(bytes.to_vec(), (buffer.clone(), next_tick));
    });
    Ok(buffer)
}

/// Test-only reset -- the default std test harness reuses threads across
/// tests in the same binary, and `UNIFORM_BUFFERS`/`UNIFORM_CACHE_CLOCK` are
/// thread-local, so a prior test's entries would otherwise leak into the
/// next one's capacity accounting.
#[cfg(test)]
fn reset_uniform_cache_for_test() {
    UNIFORM_BUFFERS.with(|cache| cache.borrow_mut().clear());
    UNIFORM_CACHE_CLOCK.with(|clock| *clock.borrow_mut() = 0);
}

fn buffer_for(
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    node: NodeId,
) -> Result<DeviceBuffer, MetalError> {
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

/// CARD 6.5's memory gate (MG-3): the plan's own transient-buffer cap, in
/// bytes. DERIVED, not measured against a live sizing config -- like
/// [`OUTPUT_POOL_MAX_PER_BUCKET`] above, this is a plain constant, not yet
/// wired through the project's build-time sizing-config mechanism (see the
/// guiding-principles "no magic numbers" rule), a gap named explicitly here
/// rather than hidden. [`build_buffer_arena`] prints its own peak against
/// this BEFORE returning, and a peak above it is this card's own KILL
/// condition regardless of the `op_setup` win.
#[cfg(feature = "metal-plan-stable-buffers")]
const ARENA_TRANSIENT_CAP: usize = 172_812_125;

/// CARD 6.5: whole-`MetalBuffer` device output arena, hung off the cached
/// [`Plan`] and built exactly once, in [`plan`], from
/// [`proxima_tensor::node_retirement`]'s own liveness ranges over
/// `prepared.resolved` -- never lazily, never per `execute_plan_with_placements`
/// call. `slots` holds one physical buffer per size class actually needed;
/// `position_slot` says which slot backs each plan POSITION's output.
///
/// # Whole-buffer sharing only (crit RS-3)
///
/// A slot is reused across two positions only when [`build_buffer_arena`]'s
/// single retirement-ordered pass has seen the first position's node retired
/// before assigning the slot to a later position needing the SAME byte
/// length -- never a sub-range of a larger slot. Sub-allocating would make
/// `encode_op`'s `device_buffers.insert(bound.node, (output, 0))` a lie, and
/// [`finish`]'s readback invariant ("an output node's buffer is always
/// freshly allocated ... at offset 0") would then read a co-resident node's
/// bytes instead of its own.
///
/// # Outputs are pinned by construction
///
/// `node_retirement` already excludes every `effective_outputs` node from
/// every position's retire list, so an output's slot is never handed back to
/// `free_by_size` by this struct's own build loop -- no separate output
/// check is needed here, only the `debug_assert!` in [`build_buffer_arena`]
/// that keeps that upstream invariant honest if it ever changes.
#[cfg(feature = "metal-plan-stable-buffers")]
struct BufferArena {
    slots: Vec<MetalBuffer>,
    slot_bytes: Vec<usize>,
    /// Parallel to `prepared.resolved`.
    position_slot: Vec<usize>,
    /// Live-bytes high-water mark reached while building -- MG-3's own
    /// witness against [`ARENA_TRANSIENT_CAP`].
    peak_bytes: usize,
}

#[cfg(feature = "metal-plan-stable-buffers")]
impl BufferArena {
    /// The `(buffer, offset)` pair [`encode_op`] binds a position's output
    /// to when the caller has not output-placed that position's node.
    /// Offset is always 0: see this struct's own "whole-buffer sharing
    /// only" doc.
    fn placement_for(&self, position: usize) -> (&MetalBuffer, usize) {
        (&self.slots[self.position_slot[position]], 0)
    }

    /// Physical slot count -- the direct witness of how much reuse
    /// [`build_buffer_arena`]'s free list actually achieved: `slots.len() <
    /// position_slot.len()` whenever two or more positions shared a slot.
    fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// A slot's own allocated byte length -- test/diagnostic surface for
    /// asserting a growing extent forces a genuinely new slot rather than
    /// silently reusing an undersized one.
    fn slot_byte_len(&self, slot: usize) -> usize {
        self.slot_bytes[slot]
    }
}

/// Builds [`BufferArena`] in one pass over `resolved`, in program order,
/// mirroring the retirement ordering [`execute_plan_with_placements`]'s own
/// dispatch loop already uses: assign THIS position's slot first (against
/// the free list as of every EARLIER position's retirements only), then
/// free whatever `retires[position]` names. `effective_outputs` is passed
/// through only for the `debug_assert!` below -- `node_retirement` itself is
/// what actually keeps an output out of `retires`.
///
/// Prints the naive (no-reuse) transient sum against [`ARENA_TRANSIENT_CAP`]
/// before allocating anything, per this card's memory gate.
#[cfg(feature = "metal-plan-stable-buffers")]
fn build_buffer_arena(
    device: &ProtocolObject<dyn MTLDevice>,
    resolved: &[BoundOp],
    retires: &[Vec<NodeId>],
    effective_outputs: &[NodeId],
) -> Result<BufferArena, MetalError> {
    let outputs: BTreeSet<NodeId> = effective_outputs.iter().copied().collect();
    let naive_transient_bytes: usize = resolved
        .iter()
        .map(|bound| bound_output_len(bound).max(1) * bound.dtype.size_bytes())
        .sum();
    let uniform_bytes: usize = resolved.iter().map(pack_uniforms_byte_len).sum();
    debug!(
        naive_transient_bytes = naive_transient_bytes as u64,
        uniform_bytes = uniform_bytes as u64,
        op_count = resolved.len() as u64,
        arena_transient_cap = ARENA_TRANSIENT_CAP as u64,
        "buffer arena sized against the naive (no-reuse) transient sum"
    );

    let mut free_by_size: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut slots: Vec<MetalBuffer> = Vec::new();
    let mut slot_bytes: Vec<usize> = Vec::new();
    let mut position_slot: Vec<usize> = Vec::with_capacity(resolved.len());
    let mut node_slot: BTreeMap<NodeId, usize> = BTreeMap::new();
    let mut live_bytes: usize = 0;
    let mut peak_bytes: usize = 0;

    for (position, bound) in resolved.iter().enumerate() {
        let byte_length = bound_output_len(bound).max(1) * bound.dtype.size_bytes();
        let slot = match free_by_size.get_mut(&byte_length).and_then(Vec::pop) {
            Some(reused) => reused,
            None => {
                let index = slots.len();
                slots.push(allocate_buffer(
                    device,
                    bound_output_len(bound),
                    bound.dtype,
                )?);
                slot_bytes.push(byte_length);
                index
            }
        };
        // every slot assignment re-occupies `byte_length` bytes, whether the
        // slot is freshly allocated or pulled back from the free list -- a
        // reused slot was subtracted out of `live_bytes` when its PREVIOUS
        // occupant retired, so skipping this on the reuse arm would double-
        // count that subtraction the next time this new occupant retires.
        live_bytes += byte_length;
        peak_bytes = peak_bytes.max(live_bytes);
        position_slot.push(slot);
        node_slot.insert(bound.node, slot);

        for retired in &retires[position] {
            debug_assert!(
                !outputs.contains(retired),
                "node_retirement must never retire an effective output"
            );
            if let Some(retired_slot) = node_slot.remove(retired) {
                live_bytes -= slot_bytes[retired_slot];
                free_by_size
                    .entry(slot_bytes[retired_slot])
                    .or_default()
                    .push(retired_slot);
            }
        }
    }

    let reuse_factor = naive_transient_bytes as f64 / peak_bytes.max(1) as f64;
    debug!(
        peak_bytes = peak_bytes as u64,
        reuse_factor, "buffer arena reached its steady-state peak"
    );
    if peak_bytes > ARENA_TRANSIENT_CAP {
        proxima_telemetry::error!(
            peak_bytes,
            cap = ARENA_TRANSIENT_CAP,
            "arena peak_bytes exceeds arena_transient_cap -- MG-3 kill condition"
        );
        return Err(MetalError::ArenaOverCap {
            peak_bytes,
            cap: ARENA_TRANSIENT_CAP,
        });
    }

    Ok(BufferArena {
        slots,
        slot_bytes,
        position_slot,
        peak_bytes,
    })
}

/// CARD 6.5: one uniform buffer per plan position, allocated once in
/// [`plan`] and written IN PLACE by [`encode_op`] on every call thereafter --
/// never through the content-keyed `UNIFORM_BUFFERS` cache. That cache is a
/// dedup map shared by BYTES across every op with identical uniform bytes
/// (see `UNIFORM_BUFFERS`'s own doc); writing through it in place would
/// corrupt every other op sharing the same key. A plan-owned buffer per
/// POSITION has no such sharing hazard: each position's buffer is used by
/// exactly that position, forever, for this plan's lifetime.
#[cfg(feature = "metal-plan-stable-buffers")]
struct PlanUniforms {
    /// Parallel to `prepared.resolved`.
    buffers: Vec<MetalBuffer>,
}

#[cfg(feature = "metal-plan-stable-buffers")]
fn build_plan_uniforms(
    device: &ProtocolObject<dyn MTLDevice>,
    resolved: &[BoundOp],
) -> Result<PlanUniforms, MetalError> {
    let mut buffers = Vec::with_capacity(resolved.len());
    for bound in resolved {
        let bytes = pack_uniforms(bound);
        let buffer = device
            .newBufferWithLength_options(bytes.len().max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::CompileFailed {
                log: "device refused to allocate a plan uniform buffer".to_string(),
            })?;
        write_plan_uniform_bytes(&buffer, &bytes);
        buffers.push(buffer);
    }
    Ok(PlanUniforms { buffers })
}

/// Overwrites `buffer`'s whole CPU-visible range with `bytes` -- the
/// in-place counterpart to `upload_uniforms`'s allocate-or-cache-hit path,
/// used only when a [`PlanUniforms`] buffer already exists for this
/// position and only its VALUES (never its identity or its byte length --
/// see [`PlanUniforms`]'s own doc) change from one call to the next. Does
/// NOT fire [`PLAN_UNIFORM_WRITES`] -- that counter is the PER-CALL census
/// [`encode_op`] owns; [`build_plan_uniforms`]'s own one-time seed write
/// uses this raw form so the first real step's count still reads exactly
/// `op_count`, not `2 * op_count`.
#[cfg(feature = "metal-plan-stable-buffers")]
fn write_plan_uniform_bytes(buffer: &ProtocolObject<dyn MTLBuffer>, bytes: &[u8]) {
    let pointer = buffer.contents();
    // SAFETY: `buffer` was allocated by `build_plan_uniforms` at exactly
    // `bytes.len().max(1)` bytes and is `storageModeShared`, so this is a
    // valid, CPU-visible, mutable byte range for the duration of this write;
    // `bytes.len()` never exceeds the buffer's own allocated length because
    // `pack_uniforms(bound)` is a pure function of `bound`'s own static
    // extents/rank/gather-count, unchanged across calls against the SAME
    // plan position.
    let destination =
        unsafe { core::slice::from_raw_parts_mut(pointer.as_ptr().cast::<u8>(), bytes.len()) };
    destination.copy_from_slice(bytes);
}

/// [`write_plan_uniform_bytes`]'s read-back counterpart -- test surface only,
/// proving a write actually landed rather than trusting the copy above.
#[cfg(all(test, feature = "metal-plan-stable-buffers"))]
fn read_back_uniform_bytes(buffer: &ProtocolObject<dyn MTLBuffer>, byte_len: usize) -> Vec<u8> {
    let pointer = buffer.contents();
    // SAFETY: same CPU-visible, `storageModeShared` argument as
    // `write_plan_uniform_bytes` above, read rather than written.
    let source = unsafe { core::slice::from_raw_parts(pointer.as_ptr().cast::<u8>(), byte_len) };
    source.to_vec()
}

/// [`encode_op`]'s arena lookup, split out so the call site reads the same
/// three lines regardless of whether `metal-plan-stable-buffers` is
/// compiled in -- `Ok(None)` with the feature off, matching `encode_op`'s
/// pre-existing "no placement, fresh `allocate_buffer`" behavior exactly.
///
/// Builds `plan.arena` on its first call for this `Plan` rather than
/// requiring [`plan`] to have built it already -- only
/// `execute_plan_with_placements`/`execute_plan_with_placements_op_timed`
/// ever call this, so `execute_plan`/`execute_plan_op_timed` (which never
/// do) cost this device allocation zero times, not once per miss.
#[cfg(feature = "metal-plan-stable-buffers")]
fn arena_placement(plan: &Plan, position: usize) -> Result<Option<(&MetalBuffer, usize)>, MetalError> {
    if plan.arena.get().is_none() {
        let (device, _queue) = device_and_queue()?;
        let arena = build_buffer_arena(
            &device,
            &plan.prepared.resolved,
            &plan.prepared.retires,
            &plan.prepared.effective_outputs,
        )?;
        // a fresh, still-empty `OnceCell` can only fail to accept this set
        // if another call already raced it in -- impossible here since
        // `plan` is `&Plan`, never shared across a concurrent write.
        let _ = plan.arena.set(arena);
    }
    Ok(plan.arena.get().map(|arena| arena.placement_for(position)))
}
#[cfg(not(feature = "metal-plan-stable-buffers"))]
fn arena_placement(_plan: &Plan, _position: usize) -> Result<Option<(&MetalBuffer, usize)>, MetalError> {
    Ok(None)
}

/// [`encode_op`]'s plan-owned-uniform lookup -- see [`arena_placement`]'s
/// own doc for why this is a free function rather than an inline `#[cfg]`,
/// and for why it builds `plan.uniforms` lazily on the same schedule.
#[cfg(feature = "metal-plan-stable-buffers")]
fn plan_uniform_buffer(plan: &Plan, position: usize) -> Result<Option<&MetalBuffer>, MetalError> {
    if plan.uniforms.get().is_none() {
        let (device, _queue) = device_and_queue()?;
        let uniforms = build_plan_uniforms(&device, &plan.prepared.resolved)?;
        let _ = plan.uniforms.set(uniforms);
    }
    Ok(plan.uniforms.get().map(|uniforms| &uniforms.buffers[position]))
}
#[cfg(not(feature = "metal-plan-stable-buffers"))]
fn plan_uniform_buffer(_plan: &Plan, _position: usize) -> Result<Option<&MetalBuffer>, MetalError> {
    Ok(None)
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
///
/// `plan_uniform` is `None` on every call site that predates
/// `metal-plan-stable-buffers` (byte-identical to before: `upload_uniforms`'s
/// allocate-or-content-cache-hit path). `Some(buffer)` skips that call
/// entirely and writes this call's fresh uniform bytes straight into the
/// plan-owned `buffer` in place instead -- see [`PlanUniforms`]'s own doc for
/// why that is sound only because the buffer is keyed by PLAN POSITION, never
/// by content.
fn encode_op(
    device: &ProtocolObject<dyn MTLDevice>,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device_buffers: &mut BTreeMap<NodeId, DeviceBuffer>,
    bound: &BoundOp,
    packed_operands: &PackedOperands,
    placement: Option<(&MetalBuffer, usize)>,
    plan_uniform: Option<&MetalBuffer>,
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
    let uniforms = match plan_uniform {
        Some(buffer) => {
            #[cfg(feature = "metal-plan-stable-buffers")]
            {
                write_plan_uniform_bytes(buffer, &pack_uniforms(bound));
                counter!(PLAN_UNIFORM_WRITES, 1);
            }
            buffer.clone()
        }
        None => upload_uniforms(device, &pack_uniforms(bound))?,
    };
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
/// stance as [`upload_block`]'s `node` parameter. `byte_offset` honors a
/// placed output's own non-zero start ([`execute_plan_with_placements`]'s
/// `output_placements`): an ordinary, non-placed node's buffer is still
/// always freshly allocated at offset `0`, so passing `0` here for that case
/// is unchanged behavior, not a special case this function has to know about.
fn read_back(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    byte_offset: usize,
    element_count: usize,
    node: NodeId,
    dtype: DType,
) -> Result<Vec<f32>, MetalError> {
    if element_count == 0 {
        return Ok(Vec::new());
    }
    match dtype {
        DType::Float16 => Ok(read_back_half(buffer, byte_offset, element_count)),
        DType::Float32
        | DType::BFloat16
        | DType::Bool
        | DType::Int8
        | DType::UInt8
        | DType::Int32
        | DType::UInt32 => Ok(read_back_as_device_f32(buffer, byte_offset, element_count)),
        DType::Int16
        | DType::UInt16
        | DType::Int64
        | DType::UInt64
        | DType::Int128
        | DType::UInt128
        | DType::Float64 => Err(EmitError::UnsupportedDType { node, dtype }.into()),
    }
}

/// Reads back an output whose device buffer holds 4-byte `float` elements
/// regardless of its logical [`DType`] — `msl::type_token` emits `"float"`
/// (never a narrower or integer MSL type) for every one of `Float32`,
/// `BFloat16`, `Bool`, `Int8`, `UInt8`, `Int32`, and `UInt32`, so the bytes
/// this reads are always IEEE-754 binary32 on the device side no matter
/// which of those logical dtypes the caller asked for. `Float16` is the one
/// dtype this backend narrows on-device (`read_back_half`); every other
/// dtype that reaches this function was upcast to `float` before dispatch
/// and is downcast back to its logical dtype by the caller after this
/// returns, not by this function.
fn read_back_as_device_f32(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    byte_offset: usize,
    element_count: usize,
) -> Vec<f32> {
    debug_assert!(
        buffer.length() >= byte_offset + element_count * size_of::<f32>(),
        "device buffer too small for a device-f32 read-back: buffer.length()={}, byte_offset={byte_offset}, element_count={element_count}",
        buffer.length(),
    );
    let pointer = buffer.contents();
    // SAFETY: `buffer` is `storageModeShared`, so `contents()` is a
    // CPU-visible pointer to at least `byte_offset + element_count * 4`
    // initialized bytes — every output buffer this driver allocates is
    // sized to at least that many bytes past a placed node's own offset
    // (see `allocate_buffer`'s caller, `dispatch_op`, and
    // `execute_plan_with_placements`'s caller contract for a placed one)
    // before this point is reached.
    unsafe {
        let base = pointer.as_ptr().cast::<u8>().add(byte_offset).cast::<f32>();
        core::slice::from_raw_parts(base, element_count)
    }
    .to_vec()
}

fn read_back_half(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    byte_offset: usize,
    element_count: usize,
) -> Vec<f32> {
    let pointer = buffer.contents();
    // SAFETY: `buffer` is `storageModeShared`, so `contents()` is a
    // CPU-visible pointer to at least `byte_offset + element_count * 2`
    // initialized bytes — the same sizing guarantee `read_back_as_device_f32`
    // relies on, just over the narrower element width `allocate_buffer`
    // used for a `Float16` node.
    let narrow = unsafe {
        let base = pointer.as_ptr().cast::<u8>().add(byte_offset).cast::<f16>();
        core::slice::from_raw_parts(base, element_count)
    };
    narrow.iter().map(|value| value.to_f32()).collect()
}

fn finish(
    program: &[Op],
    index_nodes: &BTreeSet<NodeId>,
    shapes: &Shapes,
    effective_outputs: &[NodeId],
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    root: NodeId,
    caller_owned: &BTreeSet<NodeId>,
) -> Result<Evaluated, MetalError> {
    let mut results = Vec::with_capacity(effective_outputs.len());
    let mut placed = BTreeSet::new();
    #[cfg(feature = "instrument")]
    let readback_started = read_ticks();
    for node in effective_outputs {
        // A caller-owned (placed) output's bytes already live in the
        // caller's own `PlacedBuffer` (`execute_plan_with_placements`'s own
        // doc) -- the KV-cache shape this exists for never reads them back
        // through `Evaluated` at all, it reads the placed buffer directly.
        // Copying them into a fresh `Vec` here would be dead work on the
        // decode critical path (a memcpy plus an allocation per node, after
        // `waitUntilCompleted`, for bytes nothing downstream consumes) --
        // `root` is the one exception: a caller always expects `.root()` to
        // resolve, so it is read back even if it happens to be placed.
        //
        // `placed` records the skip explicitly so `Evaluated::is_placed`
        // can tell a caller "this was placed, not missing" — see
        // `Evaluated::from_parts_with_placed`'s own doc.
        if *node != root && caller_owned.contains(node) {
            placed.insert(*node);
            continue;
        }
        let shape = shapes.of(*node).to_vec();
        let dtype = gpu_dtype(program, index_nodes, *node);
        let data = match device_buffers.get(node) {
            // a plain [`execute_plan`] output's buffer is freshly allocated
            // by `encode_op` at offset 0, but an [`execute_plan_with_placements`]
            // output lands in the CALLER's own buffer at whatever byte offset
            // that call chose (`output_placements`) -- so `offset` here is
            // read, never assumed to be `0`, or a placed node's read-back
            // would silently return its buffer's UNRELATED leading bytes
            // instead of the bytes this node actually wrote.
            Some((buffer, offset)) => {
                read_back(buffer, *offset, element_count(&shape), *node, dtype)?
            }
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
    Ok(Evaluated::from_parts_with_placed(root, results, None, placed))
}

#[cfg(all(test, feature = "instrument"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod operand_tensor_bytes_tests {
    //! CARD 0.2's own tests: `operand_tensor_bytes` is the fix for the
    //! defect verified verbatim in the card's `opens` -- `operand_bytes`
    //! summing `device_buffers[source].0.length()`, which reports the
    //! shared checkpoint-mapping buffer's own size for EVERY tensor that
    //! buffer serves, rather than that tensor's own byte count. Every case
    //! here is a real GGUF codec's declared block shape, read from
    //! `crate::msl`'s own block constants rather than hand-computed, so a
    //! constant drifting there fails this test instead of silently
    //! agreeing with a stale hand-copy.

    use alloc::string::String;
    use alloc::vec;
    use core::slice;

    use objc2_metal::MTLBuffer;
    use proxima_tensor::{AlignedBuffer, DType, Extent, Op, infer};

    use super::{
        BTreeSet, NodeId, PackedCodec, PackedOperands, device_and_queue, element_count,
        operand_tensor_bytes, page_size, register_checkpoint_mapping, upload_packed_bytes,
    };
    use crate::msl::{Q4K_BLOCK_BYTES, Q5K_BLOCK_BYTES};

    /// One `Op::Input` program, so `operand_tensor_bytes` can be exercised
    /// against a real [`proxima_tensor::Shapes`] the same way [`plan`]
    /// builds one, rather than a hand-rolled shape table `Shapes`'s own
    /// module keeps private.
    fn single_input_shapes(elements: u32) -> (Vec<Op>, proxima_tensor::Shapes) {
        let program = vec![Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(elements)],
            name: Some(String::from("w")),
        }];
        let shapes = infer(&program, &[]).expect("a single Input node always infers");
        (program, shapes)
    }

    #[test]
    fn q4k_operand_reports_rows_times_k_times_144_over_256() {
        let (program, shapes) = single_input_shapes(2 * 256);
        let mut packed_operands = PackedOperands::new();
        packed_operands.insert(NodeId(0), PackedCodec::Q4K);

        let bytes = operand_tensor_bytes(
            &program,
            &BTreeSet::new(),
            &shapes,
            &packed_operands,
            NodeId(0),
        );

        assert_eq!(bytes, 2 * Q4K_BLOCK_BYTES as u64);
    }

    #[test]
    fn q5k_operand_reports_rows_times_k_times_176_over_256() {
        let (program, shapes) = single_input_shapes(3 * 256);
        let mut packed_operands = PackedOperands::new();
        packed_operands.insert(NodeId(0), PackedCodec::Q5K);

        let bytes = operand_tensor_bytes(
            &program,
            &BTreeSet::new(),
            &shapes,
            &packed_operands,
            NodeId(0),
        );

        assert_eq!(bytes, 3 * Q5K_BLOCK_BYTES as u64);
    }

    #[test]
    fn f32_operand_reports_element_count_times_4() {
        let (program, shapes) = single_input_shapes(4096);
        let packed_operands = PackedOperands::new();

        let bytes = operand_tensor_bytes(
            &program,
            &BTreeSet::new(),
            &shapes,
            &packed_operands,
            NodeId(0),
        );

        assert_eq!(bytes, 4096 * 4);
    }

    /// The mechanism-level counterpart to the three pure-function cases
    /// above: a tensor served by [`checkpoint_mapping_offset`] binds a
    /// buffer spanning the WHOLE registered mapping (several pages here),
    /// at a nonzero byte offset -- `bound_buffer_bytes` (the old,
    /// defective `operand_bytes`) reports that whole-mapping length for
    /// every tensor sharing it, while `operand_tensor_bytes` reports only
    /// this one tensor's own declared byte length, unaffected by which
    /// buffer or offset backs it.
    #[test]
    fn operand_bound_at_a_nonzero_mapping_offset_reports_the_tensor_length_not_the_shared_buffer() {
        let Ok((device, _queue)) = device_and_queue() else {
            // no Metal device on this host (e.g. a headless CI runner) --
            // every other Metal-gated test in this crate skips the same
            // way, so this one does too rather than failing spuriously.
            return;
        };
        let page = page_size();
        // two pages of f32 headroom -- comfortably larger than the single
        // Q4_K block this test carves a sub-slice out of, so the mapping's
        // own length is provably larger than the tensor's.
        let mapping = AlignedBuffer::new(page / 4 * 2, page).expect("page-aligned test mapping");
        // SAFETY: `mapping` owns `mapping.len()` initialized `f32`s; a byte
        // view of the exact same live range is valid for as long as
        // `mapping` is (it outlives every use of `mapping_bytes` below).
        let mapping_bytes: &[u8] =
            unsafe { slice::from_raw_parts(mapping.as_ptr().cast::<u8>(), mapping.len() * 4) };
        register_checkpoint_mapping(mapping_bytes);

        // a one-block Q4_K tensor starting at byte 144 (one block in) --
        // nonzero AND not page-aligned, so it can only reach the GPU
        // through `checkpoint_mapping_offset`, never the plain no-copy path.
        let tensor_offset = Q4K_BLOCK_BYTES;
        let tensor_bytes = &mapping_bytes[tensor_offset..tensor_offset + Q4K_BLOCK_BYTES];

        let (buffer, bound_offset) =
            upload_packed_bytes(&device, tensor_bytes, false).expect("checkpoint-mapping upload");

        assert_eq!(
            bound_offset, tensor_offset,
            "checkpoint_mapping_offset must report this tensor's own byte offset"
        );
        let bound_buffer_bytes = buffer.length();
        assert!(
            bound_buffer_bytes > tensor_bytes.len(),
            "the bound buffer spans the whole mapping ({bound_buffer_bytes} bytes), \
             not this one tensor ({} bytes) -- this IS defect 1",
            tensor_bytes.len()
        );

        let (program, shapes) = single_input_shapes(256);
        let mut packed_operands = PackedOperands::new();
        packed_operands.insert(NodeId(0), PackedCodec::Q4K);
        let operand_bytes = operand_tensor_bytes(
            &program,
            &BTreeSet::new(),
            &shapes,
            &packed_operands,
            NodeId(0),
        );
        assert_eq!(
            operand_bytes,
            tensor_bytes.len() as u64,
            "operand_tensor_bytes must report the TENSOR's own length regardless of \
             which shared buffer or offset backs it"
        );
        assert_ne!(
            operand_bytes, bound_buffer_bytes as u64,
            "the fix's whole point: operand bytes and bound buffer bytes now differ"
        );

        // leave no mapping registered for whichever test this thread runs
        // next -- `CHECKPOINT_MAPPING` is thread-local, but the default
        // std test harness reuses threads across tests in the same binary.
        super::CHECKPOINT_MAPPING.with(|cell| *cell.borrow_mut() = None);
        let _ = element_count(shapes.of(NodeId(0)));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod uniform_cache_tests {
    //! `metal::UNIFORM_BUFFERS`'s LRU bound: no eviction meant a workload
    //! whose uniform bytes vary per call grew the cache without bound (a
    //! leak by construction). These tests exercise the real
    //! `upload_uniforms` upload path against a real Metal device, so they
    //! skip (rather than fail) on a headless host with no GPU -- the same
    //! convention every other Metal-gated test in this module follows.

    use super::{
        device_and_queue, reset_uniform_cache_for_test, uniform_cache_len, upload_uniforms,
    };
    use crate::sized::UNIFORM_CACHE_ENTRIES;

    /// A distinct, non-empty uniform blob per `index` -- `upload_uniforms`
    /// requires non-empty bytes (every real `Uniforms` struct has at least
    /// two `long` fields), and content-keying means two different indices
    /// must produce byte-distinct blobs.
    fn blob(index: u64) -> Vec<u8> {
        index.to_le_bytes().to_vec()
    }

    #[test]
    fn filling_past_capacity_evicts_the_least_recently_used_entry() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_uniform_cache_for_test();
        let capacity = UNIFORM_CACHE_ENTRIES as usize;

        for index in 0..capacity as u64 {
            upload_uniforms(&device, &blob(index)).expect("upload within capacity");
        }
        assert_eq!(uniform_cache_len(), capacity);

        // touch key 0 -- a cache hit that must refresh its tick, protecting
        // it from the eviction the next insert triggers.
        let reuses_before_touch = super::UNIFORM_BUFFER_REUSES.get();
        upload_uniforms(&device, &blob(0)).expect("touch key 0");
        assert_eq!(
            super::UNIFORM_BUFFER_REUSES.get(),
            reuses_before_touch + 1,
            "touching key 0 must be a cache hit"
        );

        // one more distinct blob forces an eviction; key 1 is now the
        // least recently used (key 0 was just touched, keys 2.. were
        // inserted after key 1).
        upload_uniforms(&device, &blob(capacity as u64)).expect("upload past capacity");
        assert_eq!(
            uniform_cache_len(),
            capacity,
            "cache must stay bounded at capacity after eviction"
        );

        let reuses_before_key_zero = super::UNIFORM_BUFFER_REUSES.get();
        upload_uniforms(&device, &blob(0)).expect("key 0 must still be resident");
        assert_eq!(
            super::UNIFORM_BUFFER_REUSES.get(),
            reuses_before_key_zero + 1,
            "key 0 (touched) must have survived the eviction"
        );

        let reuses_before_key_one = super::UNIFORM_BUFFER_REUSES.get();
        upload_uniforms(&device, &blob(1)).expect("key 1 re-upload after eviction");
        assert_eq!(
            super::UNIFORM_BUFFER_REUSES.get(),
            reuses_before_key_one,
            "key 1 was the least recently used entry and must have missed the cache"
        );
    }

    #[test]
    fn a_cache_hit_refreshes_the_use_tick() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_uniform_cache_for_test();

        upload_uniforms(&device, &blob(0)).expect("insert key 0");
        upload_uniforms(&device, &blob(1)).expect("insert key 1");

        let tick_of_key = |bytes: &[u8]| -> u64 {
            super::UNIFORM_BUFFERS.with(|cache| cache.borrow().get(bytes).expect("key present").1)
        };
        let tick_before = tick_of_key(&blob(0));

        upload_uniforms(&device, &blob(0)).expect("touch key 0 again");
        let tick_after = tick_of_key(&blob(0));

        assert!(
            tick_after > tick_before,
            "a cache hit must advance key 0's use-tick ({tick_before} -> {tick_after})"
        );
    }

    #[test]
    fn uniform_buffer_reuses_still_counts_hits() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_uniform_cache_for_test();

        let reuses_before = super::UNIFORM_BUFFER_REUSES.get();
        upload_uniforms(&device, &blob(0)).expect("first upload is a miss");
        assert_eq!(
            super::UNIFORM_BUFFER_REUSES.get(),
            reuses_before,
            "a fresh blob must not count as a reuse"
        );

        upload_uniforms(&device, &blob(0)).expect("second upload of the same bytes is a hit");
        assert_eq!(
            super::UNIFORM_BUFFER_REUSES.get(),
            reuses_before + 1,
            "re-uploading identical bytes must count exactly one reuse"
        );
    }
}

/// CARD 6.5's own soundness tests: [`BufferArena`]/[`PlanUniforms`] built by
/// [`plan`] and bound (never allocated) by [`encode_op`] on every call
/// against a plan already in the plan cache.
#[cfg(all(test, feature = "metal-plan-stable-buffers"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod arena_tests {
    use alloc::vec;

    use objc2_metal::MTLDevice;
    use proxima_tensor::{
        DType, Extent, IndexMap, NodeId, Op, QuantizedBlock, ScalarOp, append, cpu, projection,
    };

    use super::{
        arena_placement, device_and_queue, execute_plan, execute_plan_with_placements, plan,
        plan_uniform_buffer,
    };

    /// `Input(a, extent) -> Identity -> Identity -> Input(b, extent) ->
    /// Identity` -- three dispatched (non-`Input`) elementwise ops, `node[0]`
    /// consumed only by `node[1]` so it retires right after `node[1]` runs,
    /// freeing a slot a same-size later op COULD reuse. Returns the program
    /// and the three dispatched nodes in position order.
    fn three_stage_chain(extent_a: u32, extent_b: u32) -> (Vec<Op>, [NodeId; 3]) {
        let mut program = Vec::new();
        let input_a = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent_a)],
                name: None,
            },
        );
        let stage_zero = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: vec![(input_a, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
        let stage_one = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: vec![(stage_zero, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
        let input_b = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent_b)],
                name: None,
            },
        );
        let stage_two = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: vec![(input_b, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
        (program, [stage_zero, stage_one, stage_two])
    }

    /// A ten-stage `Identity` chain, every stage the same `extent`, only the
    /// LAST stage declared reachable by any test that wants a real forward
    /// -- the shape [`live_buffer_count_bounded_in_steady_state`] and
    /// [`a_pooled_slot_is_never_rebound_before_its_last_consumers_position`]
    /// both need: whatever the binder's own elementwise-fusion pass leaves
    /// dispatched, over a long enough chain to force at least one real
    /// arena slot reuse if reuse is happening at all.
    fn ten_stage_chain(extent: u32) -> (Vec<Op>, Vec<NodeId>) {
        let mut program = Vec::new();
        let mut previous = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent)],
                name: None,
            },
        );
        let mut nodes = Vec::new();
        for _ in 0..10 {
            previous = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Identity,
                    operands: vec![(previous, IndexMap::Affine(projection(1, &[0])))],
                    name: None,
                },
            );
            nodes.push(previous);
        }
        (program, nodes)
    }

    /// `shared = Negate(input)` fed to TWO consumers -- fan-out the
    /// binder's own single-consumer elementwise fusion cannot collapse
    /// (inlining `shared` into either consumer alone would still leave the
    /// other needing a materialized read, so `shared` stays its own
    /// dispatched `BoundOp`). Neither consumer is declared an output, so
    /// `shared` genuinely retires once both have run -- unlike a plain
    /// linear `Identity` chain (see `ten_stage_chain`'s own doc), which
    /// fuses down to a single dispatch and never exercises retirement at
    /// all. Appends onto the CALLER's `program` so several diamonds can
    /// share one program (and one plan), each with its own `shared` node of
    /// the SAME extent -- the shape this test needs to prove reuse across
    /// independent diamonds, not just within one.
    fn append_diamond(program: &mut Vec<Op>, extent: u32) -> (NodeId, NodeId, NodeId) {
        let input = append(
            program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent)],
                name: None,
            },
        );
        let shared = append(
            program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Negate,
                operands: vec![(input, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
        let consumer_a = append(
            program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: vec![(shared, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
        let consumer_b = append(
            program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Negate,
                operands: vec![(shared, IndexMap::Affine(projection(1, &[0])))],
                name: None,
            },
        );
        (shared, consumer_a, consumer_b)
    }

    #[test]
    fn pooled_output_equals_the_cpu_oracle_on_a_real_forward() {
        let (program, [stage_zero, stage_one, stage_two]) = three_stage_chain(4, 4);
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0, 30.0, 40.0];
        let outputs = [stage_zero, stage_one, stage_two];

        let cpu_oracle = cpu::evaluate(&program, &[], &[&a, &b], &outputs)
            .expect("the CPU oracle evaluates the same chain");

        let resolved_plan = plan(
            &program,
            &[],
            &[QuantizedBlock::Float32(&a), QuantizedBlock::Float32(&b)],
            &outputs,
        )
        .expect("plans the chain once -- its arena and plan uniforms build lazily below");
        let pooled = execute_plan_with_placements(
            &resolved_plan,
            &[QuantizedBlock::Float32(&a), QuantizedBlock::Float32(&b)],
            &[],
            &[],
        )
        .expect("runs the chain against the arena-bound, plan-owned-uniform path");

        for node in outputs {
            let (expected, _shape) = cpu_oracle.get(node).expect("oracle has this output");
            let (actual, _shape) = pooled.get(node).expect("pooled run has this output");
            assert_eq!(
                actual, expected,
                "arena-bound output for {node:?} must equal the CPU oracle's -- \
                 pooled and unpooled must agree on the real forward"
            );
        }
    }

    /// The direct witness for the fix this module carries: `execute_plan`
    /// never binds a placement (it always passes `None, None` into
    /// `encode_op`), so it must never trigger `arena_placement`/
    /// `plan_uniform_buffer`'s device allocation -- before this change,
    /// [`plan`] built the arena and plan uniforms unconditionally, so this
    /// same call sequence would have found both `OnceCell`s already full.
    #[test]
    fn execute_plan_never_builds_the_arena_or_uniforms_for_the_unplaced_path() {
        let (program, [stage_zero, stage_one, stage_two]) = three_stage_chain(4, 4);
        let a = [1.0f32; 4];
        let b = [2.0f32; 4];
        let outputs = [stage_zero, stage_one, stage_two];
        let blocks = [QuantizedBlock::Float32(&a), QuantizedBlock::Float32(&b)];
        let resolved_plan =
            plan(&program, &[], &blocks, &outputs).expect("plans the three-stage chain");

        execute_plan(&resolved_plan, &blocks).expect("runs the unplaced program");

        assert!(
            resolved_plan.arena.get().is_none(),
            "execute_plan took the unplaced path and must never have built the arena"
        );
        assert!(
            resolved_plan.uniforms.get().is_none(),
            "execute_plan took the unplaced path and must never have built plan uniforms"
        );
        assert_eq!(
            resolved_plan.arena_peak_bytes(),
            None,
            "the public accessor must agree with the private field: no placed \
             execution happened, so there is no arena peak to report"
        );
    }

    #[test]
    fn a_pooled_slot_is_never_rebound_before_its_last_consumers_position() {
        // five independent diamonds, same extent -- each diamond's `shared`
        // node is a genuine intermediate (never declared an output) that
        // retires once both its consumers have run, and every diamond
        // requests the SAME byte length, so the free list has every
        // opportunity to hand an earlier diamond's freed slot to a later
        // one.
        let mut program = Vec::new();
        let mut outputs = Vec::new();
        for _ in 0..5 {
            let (_shared, consumer_a, consumer_b) = append_diamond(&mut program, 4);
            outputs.push(consumer_a);
            outputs.push(consumer_b);
        }
        let a = [1.0f32; 4];
        let blocks: Vec<QuantizedBlock<'_>> =
            core::iter::repeat_n(QuantizedBlock::Float32(a.as_slice()), 5).collect();
        let resolved_plan =
            plan(&program, &[], &blocks, &outputs).expect("plans the five-diamond program");
        arena_placement(&resolved_plan, 0).expect("builds the arena on first placement lookup");

        let arena = resolved_plan.arena.get().expect("arena was just built above");
        let retires = &resolved_plan.prepared.retires;
        let resolved = &resolved_plan.prepared.resolved;

        // group every position by which physical slot it was bound to --
        // any slot with 2+ positions is a real reuse event `node_retirement`
        // must have authorized.
        let mut positions_by_slot: alloc::collections::BTreeMap<usize, Vec<usize>> =
            alloc::collections::BTreeMap::new();
        for (position, &slot) in arena.position_slot.iter().enumerate() {
            positions_by_slot.entry(slot).or_default().push(position);
        }

        let mut reuse_events_checked = 0;
        for occupants in positions_by_slot.values() {
            for window in occupants.windows(2) {
                let (earlier, later) = (window[0], window[1]);
                let earlier_node = resolved[earlier].node;
                let freed_before_reassignment = retires[..later]
                    .iter()
                    .any(|freed_here| freed_here.contains(&earlier_node));
                assert!(
                    freed_before_reassignment,
                    "slot reused at position {later} before its earlier occupant \
                     (position {earlier}, node {earlier_node:?}) was retired -- a pooled \
                     buffer must never be rebound before its last consumer's position"
                );
                reuse_events_checked += 1;
            }
        }
        assert!(
            reuse_events_checked > 0,
            "degenerate gate: five same-extent diamonds must produce at least one real \
             slot reuse, or this test proves nothing"
        );
    }

    #[test]
    fn a_programs_output_buffer_is_never_in_the_free_list() {
        // stage_zero (position 0, 4 elements) retires right after stage_one
        // consumes it (position 1). stage_two (position 2) requests the
        // SAME 4-element extent -- if stage_zero's own OUTPUT were ever
        // freed, this arm would prove nothing; declaring it a program
        // OUTPUT is what pins it, per `node_retirement`'s own exclusion.
        let (program, [stage_zero, stage_one, stage_two]) = three_stage_chain(4, 4);
        let a = [1.0f32; 4];
        let b = [2.0f32; 4];
        // stage_zero declared as an output pins it -- node_retirement never
        // retires a declared output, so its slot can never reach the free
        // list stage_two's identical-sized request would otherwise pull from.
        let outputs = [stage_zero, stage_one, stage_two];
        let blocks = [QuantizedBlock::Float32(&a), QuantizedBlock::Float32(&b)];
        let resolved_plan =
            plan(&program, &[], &blocks, &outputs).expect("plans the pinned-output chain");
        arena_placement(&resolved_plan, 0).expect("builds the arena on first placement lookup");
        let arena = resolved_plan.arena.get().expect("arena was just built above");

        let position_of = |node: NodeId| {
            resolved_plan
                .prepared
                .resolved
                .iter()
                .position(|bound| bound.node == node)
                .expect("node is dispatched")
        };
        let stage_zero_slot = arena.position_slot[position_of(stage_zero)];
        let stage_two_slot = arena.position_slot[position_of(stage_two)];
        assert_ne!(
            stage_zero_slot, stage_two_slot,
            "a pinned program output's slot must never be handed to a later same-size op"
        );
    }

    #[test]
    fn an_extent_change_forces_a_documented_realloc() {
        // stage_zero (4 elements, 16 bytes) and stage_two (8 elements, 32
        // bytes) are both declared outputs -- independent single-op leaves,
        // immune to elementwise fusion -- so this test isolates exactly one
        // thing: two differently-sized requests in the SAME plan must never
        // be conflated by the arena's size-class bookkeeping.
        let (program, [stage_zero, _stage_one, stage_two]) = three_stage_chain(4, 8);
        let a = [1.0f32; 4];
        let b = [2.0f32; 8];
        let outputs = [stage_zero, stage_two];
        let blocks = [QuantizedBlock::Float32(&a), QuantizedBlock::Float32(&b)];
        let resolved_plan =
            plan(&program, &[], &blocks, &outputs).expect("plans the size-mismatched chain");
        arena_placement(&resolved_plan, 0).expect("builds the arena on first placement lookup");
        let arena = resolved_plan.arena.get().expect("arena was just built above");

        let position_of = |node: NodeId| {
            resolved_plan
                .prepared
                .resolved
                .iter()
                .position(|bound| bound.node == node)
                .expect("node is dispatched")
        };
        let stage_zero_slot = arena.position_slot[position_of(stage_zero)];
        let stage_two_slot = arena.position_slot[position_of(stage_two)];
        assert_ne!(
            stage_zero_slot, stage_two_slot,
            "a byte-length mismatch must force a genuinely new slot, never a reused one"
        );
        assert_eq!(
            arena.slot_byte_len(stage_two_slot),
            8 * size_of::<f32>(),
            "the new slot must be sized to the LARGER extent's own byte length"
        );
    }

    #[test]
    fn uniform_contents_change_per_step_while_the_buffer_identity_does_not() {
        let (device, _queue) = match device_and_queue() {
            Ok(pair) => pair,
            Err(_) => return,
        };
        let buffer = device
            .newBufferWithLength_options(8, objc2_metal::MTLResourceOptions::StorageModeShared)
            .expect("allocates an 8-byte plan-uniform-shaped buffer");
        let identity_before = objc2::rc::Retained::as_ptr(&buffer);

        super::write_plan_uniform_bytes(&buffer, &[1, 2, 3, 4, 5, 6, 7, 8]);
        let first_write = super::read_back_uniform_bytes(&buffer, 8);
        super::write_plan_uniform_bytes(&buffer, &[9, 8, 7, 6, 5, 4, 3, 2]);
        let second_write = super::read_back_uniform_bytes(&buffer, 8);
        let identity_after = objc2::rc::Retained::as_ptr(&buffer);

        assert_ne!(
            first_write, second_write,
            "successive writes into the SAME plan-owned uniform buffer must change its contents"
        );
        assert_eq!(
            identity_before, identity_after,
            "the buffer's own identity must never change across writes -- only its bytes do"
        );
    }

    #[test]
    fn live_buffer_count_bounded_in_steady_state() {
        // A ten-stage chain, every stage the SAME extent, so every stage
        // after the first two retires its predecessor and the free list can
        // fully reuse a small, bounded set of slots instead of growing one
        // slot per stage.
        let (program, nodes) = ten_stage_chain(4);
        let a = [1.0f32; 4];
        let outputs = [*nodes.last().expect("ten stages were pushed")];
        let blocks = [QuantizedBlock::Float32(&a)];
        let resolved_plan =
            plan(&program, &[], &blocks, &outputs).expect("plans the ten-stage chain");
        arena_placement(&resolved_plan, 0).expect("builds the arena on first placement lookup");

        let slot_count = resolved_plan
            .arena
            .get()
            .expect("arena was just built above")
            .slot_count();
        assert!(
            slot_count <= 3,
            "ten same-size stages must reuse down to a small, bounded slot count via the \
             free list, not grow one slot per stage (got {slot_count})"
        );
    }

    #[test]
    fn two_ops_with_identical_uniform_bytes_get_distinct_plan_owned_buffers() {
        // stage_zero and stage_two are two INDEPENDENT, identically-shaped
        // Identity ops -- `pack_uniforms` packs the same bytes (rank,
        // extents, operand base/strides) for both, since a fresh `Input`'s
        // own operand base is 0 either way. The content-keyed
        // `UNIFORM_BUFFERS` cache would hand both the SAME buffer; plan-
        // owned uniforms must not.
        let (program, [stage_zero, _stage_one, stage_two]) = three_stage_chain(4, 4);
        let a = [1.0f32; 4];
        let b = [2.0f32; 4];
        let outputs = [stage_zero, stage_two];
        let blocks = [QuantizedBlock::Float32(&a), QuantizedBlock::Float32(&b)];
        let resolved_plan =
            plan(&program, &[], &blocks, &outputs).expect("plans the identical-uniform chain");
        plan_uniform_buffer(&resolved_plan, 0).expect("builds the uniforms on first lookup");

        let position_of = |node: NodeId| {
            resolved_plan
                .prepared
                .resolved
                .iter()
                .position(|bound| bound.node == node)
                .expect("node is dispatched")
        };
        let stage_zero_bound = &resolved_plan.prepared.resolved[position_of(stage_zero)];
        let stage_two_bound = &resolved_plan.prepared.resolved[position_of(stage_two)];
        assert_eq!(
            super::pack_uniforms(stage_zero_bound),
            super::pack_uniforms(stage_two_bound),
            "degenerate gate: the two ops must genuinely pack identical uniform bytes"
        );

        let uniforms = resolved_plan
            .uniforms
            .get()
            .expect("uniforms were just built above");
        let stage_zero_uniform =
            objc2::rc::Retained::as_ptr(&uniforms.buffers[position_of(stage_zero)]);
        let stage_two_uniform =
            objc2::rc::Retained::as_ptr(&uniforms.buffers[position_of(stage_two)]);
        assert_ne!(
            stage_zero_uniform, stage_two_uniform,
            "two ops with identical uniform bytes must still get DISTINCT plan-owned buffers"
        );
    }
}

/// [`HazardTracker`]'s pure dataflow logic, tested with plain `&str`
/// identities so no real Metal device is required -- the real driver path
/// (`execute_plan_with_placements`) instantiates the same type with
/// `Id = *const ProtocolObject<dyn MTLBuffer>`.
#[cfg(all(test, feature = "metal-concurrent-dispatch"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod hazard_tracker_tests {
    use std::collections::BTreeMap;

    use super::{DeviceBuffer, HazardTracker, MetalError, NodeId, hazard_step, resolve_hazard_inputs};

    /// `a -> b`, `a -> c` (independent, both only read `a`), then `b, c ->
    /// d` -- the shape this whole feature exists for (Q/K/V from one normed
    /// input, then a later op that needs all three). `b` and `c` share no
    /// hazard with each other (neither reads nor writes the other), so
    /// encoding `c` right after `b` must NOT barrier; `d` reads both `b` and
    /// `c`, both written since the last barrier, so encoding `d` MUST. Driven
    /// through [`hazard_step`] -- the exact function
    /// [`execute_plan_with_placements`]'s own loop calls -- rather than a
    /// hand-rolled mirror of it.
    #[test]
    fn independent_producers_share_no_barrier_but_their_joint_consumer_does() {
        let mut hazards: HazardTracker<&str> = HazardTracker::new();

        // encode `b = f(a)`: `a` is a fresh input, never written -> no hazard.
        assert!(!hazard_step(&mut hazards, &["a"], "b"));

        // encode `c = g(a)`: `a` was only READ (by `b`'s own encode), never
        // WRITTEN, and `c` is a fresh output nothing has touched -> no hazard.
        assert!(!hazard_step(&mut hazards, &["a"], "c"));

        // encode `d = h(b, c)`: both inputs were WRITTEN since the last
        // barrier (by the two steps above) -> a barrier is required.
        assert!(
            hazard_step(&mut hazards, &["b", "c"], "d"),
            "exactly one barrier is required, immediately before encoding d"
        );
    }

    /// `BufferArena` reuses a retired slot's whole buffer for a later
    /// position (`metal-plan-stable-buffers`' own doc) -- `x` writes buffer
    /// `slot0`, `y` (independent of `x`) reads `slot0` as `p`'s arena-chosen
    /// output buffer... modeled here directly: `p` reads buffer `slot0` (a
    /// WAR-eligible read), then a LATER op `q` is placed by the arena into
    /// that SAME `slot0` identity. Writing `q` into a buffer just READ from
    /// is a WAR hazard and must barrier even though `q` shares no operand
    /// with `p`.
    #[test]
    fn arena_slot_reuse_after_a_read_emits_a_war_barrier() {
        let mut hazards: HazardTracker<&str> = HazardTracker::new();

        // encode `p`, which reads `slot0` (some earlier op's live output).
        assert!(!hazard_step(&mut hazards, &["slot0"], "p_out"));

        // encode `q`, whose output the arena has placed into `slot0` itself
        // -- the exact retired-slot-reuse shape `BufferArena::assign_slot`
        // produces. `q` has no operand overlap with `p` at all; the hazard
        // is purely WAR on the output identity.
        assert!(
            hazard_step(&mut hazards, &[], "slot0"),
            "writing into a buffer read since the last barrier must be flagged WAR"
        );
    }

    /// Mirrors [`execute_plan_with_placements`]'s own output-identity
    /// resolution (placement -> arena slot -> fresh allocation) with plain
    /// `usize` identities instead of a real Metal buffer, so a 4-op program
    /// can be driven through [`hazard_step`] end to end with no device.
    fn resolve_test_output(
        node: usize,
        placement_table: &BTreeMap<usize, usize>,
        next_fresh: &mut usize,
    ) -> usize {
        if let Some(identity) = placement_table.get(&node) {
            return *identity;
        }
        let fresh = *next_fresh;
        *next_fresh += 1;
        fresh
    }

    /// The exact production sequence for a 4-op program: op0 is a plain
    /// fresh allocation with no operands, op1 does a RAW read of op0's own
    /// output, op2 is an arena WAR reuse of op0's now-retired identity, and
    /// op3 is a second fresh allocation with no overlap with anything the
    /// tracker still holds (op2's own barrier reset it). Barriers must fire
    /// immediately before op1 (RAW) and op2 (WAR), and nowhere else.
    #[test]
    fn production_sequence_barriers_before_the_raw_and_war_ops_only() {
        let mut hazards: HazardTracker<usize> = HazardTracker::new();
        let mut next_fresh = 100;
        let output0 = resolve_test_output(0, &BTreeMap::new(), &mut next_fresh);
        // op2's own output is arena-placed into op0's retired identity --
        // `arena_slot_reuse_after_a_read_emits_a_war_barrier` covers that
        // shape in isolation; here it sits inside a longer program alongside
        // a RAW hazard (op1) and a genuinely fresh allocation (op3).
        let placement_table = BTreeMap::from([(2, output0)]);

        let barrier0 = hazard_step(&mut hazards, &[], output0);

        let output1 = resolve_test_output(1, &placement_table, &mut next_fresh);
        let barrier1 = hazard_step(&mut hazards, &[output0], output1);

        let output2 = resolve_test_output(2, &placement_table, &mut next_fresh);
        assert_eq!(output2, output0, "degenerate gate: op2 must reuse op0's own identity");
        let barrier2 = hazard_step(&mut hazards, &[], output2);

        let output3 = resolve_test_output(3, &placement_table, &mut next_fresh);
        let barrier3 = hazard_step(&mut hazards, &[], output3);

        assert_eq!(
            [barrier0, barrier1, barrier2, barrier3],
            [false, true, true, false],
            "barriers fire before op1 (RAW) and op2 (WAR), never before op0 or op3"
        );
    }

    /// An operand missing from `device_buffers` is a driver bug -- the
    /// encode it feeds is already wrong -- so [`resolve_hazard_inputs`] must
    /// error, never silently drop it from the hazard set.
    #[test]
    fn a_missing_operand_buffer_is_an_error_not_a_dropped_hazard() {
        let device_buffers: BTreeMap<NodeId, DeviceBuffer> = BTreeMap::new();
        let missing = NodeId(7);

        let result = resolve_hazard_inputs(core::iter::once(missing), &device_buffers);

        match result {
            Err(MetalError::UnresolvedHazardOperand { node }) => assert_eq!(node, missing),
            other => panic!("expected UnresolvedHazardOperand, got {other:?}"),
        }
    }

    /// A retired buffer's address can come back from a later, unrelated
    /// `allocate_buffer` call (Metal's own allocator, or `metal-buffer-pool`
    /// reuse more aggressively) -- `forget` must erase that address's hazard
    /// history so the new buffer at the same address starts clean, not
    /// inheriting a stale WRITTEN/READ mark that belonged to whatever this
    /// address used to be.
    #[test]
    fn forgetting_a_retired_identity_clears_it_from_both_sets() {
        let mut hazards: HazardTracker<&str> = HazardTracker::new();
        hazards.record(&["read_only"], Some("written_and_read"));
        hazards.record(&["written_and_read"], Some("also_written"));

        hazards.forget("written_and_read");

        assert!(
            !hazards.needs_barrier(&[], Some("written_and_read")),
            "a forgotten identity must carry no hazard history for a later allocation"
        );
    }
}
