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
//! byte-identical source. `MTLCompileOptions::mathMode` is a per-[`Plan`]
//! runtime choice ([`MathMode`], set via [`Plan::set_math_mode`]), not a
//! fixed compile option: `proxima-tensor/docs/discipline.md` ROW 296
//! measured the packed-row Q4_K matvec kernel at 179.2 GB/s under `Safe`
//! and 240.9-247.3 GB/s under `Relaxed`/`Fast`, with 0-1.9e-6 parity in
//! every cell of the shape sweep either way — the "parity demands `Safe`"
//! assumption this module carried until then does not hold on the
//! evidence, and ROW 297's own bake-off (identical generated text and
//! quality across three interleaved rounds, `Relaxed` 1.20x faster
//! wall-clock steady-state) confirms it holds on the whole decode program,
//! not just one kernel -- so [`MathMode::default`] is `Relaxed`, and
//! `Safe` stays one call away for a program where it does not.
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
use core::ops::Deref;
use core::ptr::NonNull;
#[cfg(feature = "metal-buffer-pool")]
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Instant;

use half::f16;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
#[cfg(feature = "instrument")]
use objc2_foundation::NSUInteger;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{MTLBarrierScope, MTLDispatchType};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};
use proxima_telemetry::counter;
use proxima_telemetry::debug;
use proxima_telemetry::info;
use proxima_telemetry::metric::Counter;
use proxima_telemetry::trace;

#[cfg(feature = "instrument")]
use objc2_metal::{MTLCounterSampleBuffer, MTLCounterSet};
#[cfg(feature = "instrument")]
use proxima_tensor::instrument::{
    OpKind, elapsed_ticks, read_ticks, record_op_kind, ticks_to_nanos,
};
use proxima_tensor::{
    BoundOp, BoundOpKind, DType, Evaluated, Keep, Lookup, NodeId, NumericPolicy, Op,
    QuantizedBlock, Shapes, TensorError, bind, block_node_ids, correct_packed_matmul_layouts,
    index_node_ids, infer, node_retirement, prune_dead, resolve_named_blocks,
};

use crate::error::EmitError;
#[cfg(feature = "instrument")]
use crate::msl::diagnose_packed_row_block;
use crate::msl::{gather_count, kernel_cache_key, kernel_dispatch_shape, reduction_dims};
#[cfg(feature = "metal-plan-stable-buffers")]
use crate::sized::ARENA_TRANSIENT_CAP;
#[cfg(feature = "metal-buffer-pool")]
use crate::sized::OUTPUT_POOL_MAX_PER_BUCKET;
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

/// Owns a compute encoder's `endEncoding()` call so an early `?` return from
/// inside an op-encoding loop cannot leave it un-ended. Every `execute*`
/// entry point in this file used to open one encoder and call
/// `endEncoding()` exactly once, textually after its per-op loop (see the
/// module doc's "Execution model") — correct on the happy path, but any
/// error propagated with `?` from inside that loop skipped the call, and the
/// `Retained` handle's `dealloc` at autorelease-pool drain hit Metal's own
/// `-[_MTLCommandEncoder dealloc]: failed assertion 'Command encoder
/// released without endEncoding'`, a hard `SIGTRAP` that also swallowed the
/// real Rust `Err` that triggered it.
///
/// `Deref`s to the encoder so every existing call site (`&encoder`,
/// `encoder.memoryBarrierWithScope(..)`, passing it into [`encode_op`])
/// compiles unchanged. Call [`EncoderGuard::finish`] on the explicit success
/// path; anywhere else (including every `?`), [`Drop::drop`] ends it exactly
/// once instead.
struct EncoderGuard {
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    finished: bool,
}

#[cfg(all(test, feature = "instrument"))]
mod bounded_compare_tests {
    use super::{MetalError, compare_bound_f32};
    use proxima_tensor::op::NodeId;

    #[test]
    fn finite_equal_is_not_a_mismatch() {
        assert!(
            compare_bound_f32(NodeId(1), &[1.0, -2.0], &[1.0, -2.0])
                .expect("equal lengths")
                .is_none()
        );
    }

    #[test]
    fn finite_difference_reports_first_element_and_maximum_relative_error() {
        let mismatch = compare_bound_f32(NodeId(1), &[2.0, 4.0], &[1.0, 2.0])
            .expect("equal lengths")
            .expect("difference");
        assert_eq!(mismatch.0, 0);
        assert_eq!(mismatch.1, 2.0);
        assert_eq!(mismatch.2, 1.0);
        assert_eq!(mismatch.3, 1.0);
        assert_eq!(mismatch.4, 1.0);
    }

    #[test]
    fn nonfinite_difference_cannot_evade_tolerance() {
        let mismatch = compare_bound_f32(NodeId(1), &[f32::NAN], &[1.0])
            .expect("equal lengths")
            .expect("nonfinite difference");
        assert!(mismatch.3.is_infinite());
    }

    #[test]
    fn cpu_nonfinite_difference_is_reported() {
        let mismatch = compare_bound_f32(NodeId(1), &[1.0], &[f32::NAN])
            .expect("equal lengths")
            .expect("nonfinite difference");
        assert!(mismatch.3.is_infinite());
    }

    #[test]
    fn length_mismatch_is_typed() {
        let error = compare_bound_f32(NodeId(9), &[1.0], &[]).expect_err("length mismatch");
        assert!(matches!(
            error,
            MetalError::CpuBoundComparisonLengthMismatch {
                node: NodeId(9),
                metal_len: 1,
                cpu_len: 0,
            }
        ));
    }
}

impl EncoderGuard {
    fn new(encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) -> Self {
        Self {
            encoder,
            finished: false,
        }
    }

    /// The success-path close: ends encoding now rather than at `Drop`, so
    /// intent at the call site reads the same as the un-guarded code it
    /// replaces (`encoder.endEncoding()` becomes `encoder.finish()`).
    fn finish(mut self) {
        self.encoder.endEncoding();
        self.finished = true;
    }

    /// A cheap `Retained` clone of the underlying encoder for callers that
    /// need to alias it (e.g. hand it to [`encode_op`] by value) without
    /// taking over its `endEncoding()` obligation — only this guard's own
    /// `finish`/`Drop` ever closes the encoder. Only
    /// [`execute_plan_with_placements_dispatch_timed`]'s shared/stage-encoder
    /// reuse needs an alias rather than sole ownership, so this is gated
    /// identically to that function rather than left dead under every other
    /// feature combination.
    #[cfg(all(feature = "metal-output-placement", feature = "instrument"))]
    fn clone_inner(&self) -> Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> {
        self.encoder.clone()
    }
}

impl Deref for EncoderGuard {
    type Target = ProtocolObject<dyn MTLComputeCommandEncoder>;

    fn deref(&self) -> &Self::Target {
        &self.encoder
    }
}

impl Drop for EncoderGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.encoder.endEncoding();
        }
    }
}

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
    #[error("metal expert source table for node {node} is not a uniform packed codec: {reason}")]
    ExpertSourceUnsupported { node: NodeId, reason: &'static str },
    #[error("checkpoint mmap page discard failed with errno {errno}")]
    CheckpointMmapDiscardFailed { errno: i32 },
    #[error("hazard tracking: operand {node} has no resolved device buffer")]
    UnresolvedHazardOperand { node: NodeId },
    /// `build_buffer_arena`'s own reuse pass still needed more transient
    /// bytes live at once than `cap_bytes` budgets -- MG-3's kill condition,
    /// now a typed error a caller can act on rather than a stderr line
    /// beside a silently returned `Ok`. `cap_bytes` is
    /// `ARENA_TRANSIENT_CAP * max(query_rows, 1)` (see
    /// `plan_query_rows`'s own doc for why the cap scales this way, not a
    /// fixed decode-shaped constant): a prefill's transient outputs grow
    /// linearly with its own row count, so a per-call budget that ignores
    /// row count rejects a prefill that fits the device just as readily as
    /// one that would not. `device_limit` (`MTLDevice::
    /// recommendedMaxWorkingSetSize`) is carried for diagnosis only -- it is
    /// not part of the accept/reject decision here.
    #[error(
        "arena peak_bytes={peak_bytes} exceeds arena_transient_cap={cap_bytes} \
         at query_rows={query_rows} (device_limit={device_limit})"
    )]
    ArenaOverCap {
        peak_bytes: usize,
        cap_bytes: usize,
        query_rows: u64,
        device_limit: u64,
    },
    /// `PROXIMA_METAL_KIND_FILTER`'s `kind:` term named a string
    /// `classify_kind` never returns -- ROW 308 found a stale or
    /// misspelled substring silently dropped the WHOLE plan instead of
    /// isolating one kind (`KindFilter::matches` returns `false` for every
    /// op, so `ablation_skip` becomes `true` for everything). This is now a
    /// typed error at the same point that env var is first consulted,
    /// rather than a silently degenerate ablation run.
    #[error("kind filter term {term:?} matches no classify_kind arm")]
    UnknownKindFilterTerm { term: String },
    /// A `PROXIMA_METAL_KIND_FILTER` value that, applied against THIS plan's
    /// own dispatch sequence, removes zero ops -- the same silent-degenerate
    /// failure [`UnknownKindFilterTerm`](Self::UnknownKindFilterTerm) covers
    /// for a `kind:` term, extended to `family:` terms (a family name is
    /// data-dependent on the loaded checkpoint, so it cannot be validated
    /// against a static list the way a `kind:` term can -- a typo there
    /// only ever shows up as "removed nothing").
    #[error("kind filter {filter:?} matched zero dispatches in this plan")]
    KindFilterMatchesNothing { filter: String },
    /// `upload_resident_copy`'s own contract, made typed rather than
    /// silently violated: [`Plan::mark_resident`]'s doc says a resident
    /// NAME's host buffer is "bound once at load and never mutated again",
    /// so a legitimate caller only ever offers the same host pointer and
    /// byte length under a given name. A hit under `name` whose offered
    /// host pointer or byte length differs from what was cached is a
    /// caller bug (ROW 331/332's shape, reproduced at the driver level): a
    /// second, unrelated host allocation reused a name a serving loop had
    /// already marked resident.
    #[error(
        "resident name {name:?} rebound to a different host buffer (cached len={cached_len}, offered len={offered_len})"
    )]
    ResidentNameRebound {
        name: String,
        cached_len: usize,
        offered_len: usize,
    },
    /// [`Plan::check_numeric_policy`]/[`Plan::set_math_mode`]: `bound` is the
    /// [`NumericPolicy`] this plan's program was actually compiled under
    /// ([`plan`]/[`plan_named`]'s own argument), `requested` is what the
    /// caller asked to confirm or narrow into. `Plan` never retains `blocks`
    /// (see [`Plan::numeric_policy`]'s own doc), so a rebind is not
    /// implementable in place -- the caller's only correct response is a
    /// fresh [`plan`]/[`plan_named`] call with `requested`.
    /// `PROXIMA_METAL_NAN_CHECK`'s own stop condition -- [`check_op_output_finite`]
    /// found this op's own output buffer holds a NaN/Inf, so
    /// [`execute_op_timed`] stops the step here rather than running every
    /// remaining op past a value already known bad. Diagnostic-only, same
    /// `instrument`-gated reachability as [`check_op_output_finite`] itself.
    #[cfg(feature = "instrument")]
    #[error("nan_check: op node={node} kind={kind} produced a non-finite output")]
    NonFiniteOpOutput { node: NodeId, kind: String },
    /// `PROXIMA_METAL_COMPARE_CPU`'s own stop condition -- [`compare_op_output_to_cpu`]
    /// found this op's own Metal output disagrees with the same node's CPU
    /// reference value beyond the 1e-2 relative-diff floor, so
    /// [`execute_op_timed`] stops the step at the FIRST divergent node
    /// rather than running every remaining op past a value already known
    /// wrong. Diagnostic-only, same `instrument`-gated reachability as
    /// [`check_op_output_finite`].
    #[cfg(feature = "instrument")]
    #[error("cpu_compare: op node={node} kind={kind} max_rel_diff={max_rel_diff} exceeds 1e-2")]
    CpuMetalDivergence {
        node: NodeId,
        kind: String,
        max_rel_diff: f32,
    },
    #[cfg(feature = "instrument")]
    #[error("cpu_compare: bound node {node} needs {bytes} snapshot bytes, above cap {cap_bytes}")]
    CpuBoundComparisonOverCap {
        node: NodeId,
        bytes: usize,
        cap_bytes: usize,
    },
    #[cfg(feature = "instrument")]
    #[error(
        "cpu_compare: bound node {node} output length differs (metal={metal_len}, cpu={cpu_len})"
    )]
    CpuBoundComparisonLengthMismatch {
        node: NodeId,
        metal_len: usize,
        cpu_len: usize,
    },
    #[cfg(feature = "instrument")]
    #[error("cpu_compare: selected bound node {node} is not present in this resolved plan")]
    CpuBoundComparisonNodeNotFound { node: NodeId },
    #[error(
        "plan bound under {bound:?}, caller requested {requested:?} -- rebuild via plan()/plan_named()"
    )]
    NumericPolicyMismatch {
        bound: NumericPolicy,
        requested: NumericPolicy,
    },
}

/// The plan-time description of one expert payload in an [`ExpertSource`].
///
/// The descriptor is deliberately independent of a Metal buffer: HOBBIT can
/// keep the payload in an mmap and a later lowering can bind one buffer per
/// codec run without changing the residency FSM or the tensor graph.  The
/// byte range is relative to the codec run's source, not to the checkpoint's
/// global mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertPayloadDescriptor {
    /// Original routed expert index. The descriptor table stays sparse in
    /// payload bytes but dense in route space so the kernel can retain O(1)
    /// `descriptor[expert]` addressing.
    pub expert_index: u32,
    pub codec: PackedCodec,
    pub byte_offset: usize,
    pub byte_length: usize,
    pub out_dim: u32,
    pub in_dim: u32,
    pub epoch: u64,
}

/// The two device buffers a codec-tagged expert kernel needs for one source
/// node. They stay owned by the execution call through command completion;
/// the table decides bytes between steps, never while a kernel borrows them.
#[derive(Clone)]
struct ExpertSourceBuffers {
    node: NodeId,
    payloads: MetalBuffer,
    payload_offset: usize,
    descriptors: MetalBuffer,
    descriptor_records: Vec<ExpertPayloadDescriptor>,
}

/// Host staging that keeps a no-copy upload's borrowed bytes alive through
/// the command-buffer wait. A contiguous expert arena can later replace this
/// staging without changing the descriptor ABI or the emitted kernel.
struct StagedExpertSource {
    _payload_bytes: Option<Vec<u8>>,
    _descriptor_bytes: Option<Vec<u8>>,
    payload_alias_address: usize,
    descriptor_alias_address: usize,
    buffers: ExpertSourceBuffers,
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

    /// Emitted mixed-expert kernels persist beside their compiled pipeline.
    /// The source body depends only on the bound shape, codec policy, math
    /// mode, and source node; the resident payload bytes are bound separately
    /// per step, so re-rendering MSL on every routed gather is unnecessary.
    static MIXED_KERNEL_CACHE: RefCell<BTreeMap<String, crate::msl::Kernel>> =
        RefCell::new(BTreeMap::new());

    /// Staged HOBBIT payloads persist across token steps on this Metal
    /// execution thread. The signature is derived from codec/shape/epoch and
    /// selected IDs, so a promotion replaces a table while an unchanged
    /// snapshot reuses its device buffers without another host upload.
    static EXPERT_SOURCE_CACHE: RefCell<BTreeMap<NodeId, (u64, StagedExpertSource)>> =
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

/// A plain record of the facts a load-time budget decision needs about this
/// host and its Metal device -- no policy, no threshold, just what the
/// device and the OS report right now (guiding-principles principle 1:
/// this is payload, not a new abstraction; the fit decision itself lives in
/// `proxima-model-interop`, which reads these fields). Every field is a
/// direct pass-through of one `MTLDevice` accessor or one `sysctlbyname`
/// call -- see [`system_memory_facts`]'s own doc for which.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemMemoryFacts {
    /// `MTLDevice::recommendedMaxWorkingSetSize` -- the device's own advice
    /// for how many bytes of GPU-visible memory a well-behaved app should
    /// keep resident at once, in bytes.
    pub recommended_max_working_set_size: u64,
    /// `MTLDevice::maxBufferLength` -- the largest single `MTLBuffer` this
    /// device will create, in bytes.
    pub max_buffer_length: u64,
    /// `MTLDevice::hasUnifiedMemory` -- `true` on every Apple silicon Mac
    /// (the device and the host share one physical memory pool), `false` on
    /// a discrete GPU with its own VRAM.
    pub has_unified_memory: bool,
    /// `MTLDevice::currentAllocatedSize` at probe time -- bytes this device
    /// has already allocated for buffers/textures/heaps, before this load
    /// adds anything.
    pub current_allocated_size: u64,
    /// `sysctlbyname("hw.memsize")` -- the host's total physical memory, in
    /// bytes, independent of anything Metal reports.
    pub physical_memory_bytes: u64,
}

/// Reads [`SystemMemoryFacts`] off this thread's Metal device
/// (`device_and_queue`, the same lazily-created device/queue pair every
/// other driver call in this module shares) and the host's `sysctlbyname`,
/// and emits them as one structured `system_facts` telemetry event. Callers
/// building a load-time fit budget (`proxima-model-interop`'s own gate) call
/// this once, before any weight upload, and derive their own limit from the
/// fields it returns -- this function makes no fit decision itself.
///
/// # Errors
///
/// [`MetalError::NoDevice`] if this host has no Metal device.
pub fn system_memory_facts() -> Result<SystemMemoryFacts, MetalError> {
    let (device, _queue) = device_and_queue()?;
    let facts = SystemMemoryFacts {
        recommended_max_working_set_size: device.recommendedMaxWorkingSetSize(),
        max_buffer_length: device.maxBufferLength() as u64,
        has_unified_memory: device.hasUnifiedMemory(),
        current_allocated_size: device.currentAllocatedSize() as u64,
        physical_memory_bytes: physical_memory_bytes(),
    };
    info!(
        recommended_max_working_set_size = facts.recommended_max_working_set_size,
        max_buffer_length = facts.max_buffer_length,
        has_unified_memory = facts.has_unified_memory,
        current_allocated_size = facts.current_allocated_size,
        physical_memory_bytes = facts.physical_memory_bytes,
        "system_facts: host and device memory facts probed at load time"
    );
    Ok(facts)
}

/// The host's total physical memory, in bytes -- `sysctlbyname("hw.memsize")`
/// rather than `sysconf` (unlike [`page_size`]'s `_SC_PAGESIZE`, POSIX has no
/// portable name for "total RAM"; `hw.memsize` is the macOS-specific MIB
/// name, read the same way `page_size` already reads a host fact through
/// `libc`).
fn physical_memory_bytes() -> u64 {
    let mut value: u64 = 0;
    let mut size = core::mem::size_of::<u64>();
    // SAFETY: `name` is a NUL-terminated C string naming a real MIB entry;
    // `value`/`size` point at a live `u64` and its own length, exactly what
    // `sysctlbyname` requires for an output buffer; the two trailing
    // pointers are `None`/`0` since this call has nothing to write.
    unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&raw mut value).cast::<c_void>(),
            &raw mut size,
            core::ptr::null_mut(),
            0,
        );
    }
    value
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
    /// Which [`MTLCompileOptions::mathMode`] every kernel this plan compiles
    /// is compiled under -- projected from `numeric_policy` at construction
    /// ([`numeric_policy_as_metal_math_mode`]), narrowable afterward within
    /// that policy via [`Plan::set_math_mode`]. See [`MathMode`]'s own doc
    /// for the measured rationale.
    math_mode: MathMode,
    /// Which bit-changing rewrites this plan's bound program was
    /// constructed with -- [`plan`]/[`plan_named`]'s own `numeric_policy`
    /// argument, fixed for this plan's whole life (see
    /// [`Plan::numeric_policy`]'s own doc for why there is no setter).
    /// `msl::context_chunks_for` (the cross-simdgroup attention
    /// context-chunk merge) consults it via [`Plan::numeric_policy`].
    numeric_policy: NumericPolicy,
    /// Which [`MTLDispatchType`] [`execute_plan_with_placements`] opens its
    /// compute encoder with -- [`DispatchType::default`] (`Concurrent`)
    /// until a caller overrides it with [`Plan::set_dispatch_type`]. See
    /// [`DispatchType`]'s own doc for the measured rationale (ROW 311/312).
    dispatch_type: DispatchType,
    /// ROW 329 diagnostic: when `Some(position)`,
    /// [`execute_plan_with_placements_dispatch_timed`]'s stage-boundary
    /// fallback (the only branch this device's own `AtStageBoundary`-only
    /// support ever takes, ROW 309's own finding) ends its compute encoder
    /// immediately BEFORE this plan position and opens a second one for the
    /// remainder, instead of ROW 309's original one-encoder-per-position
    /// fallback -- exactly the "one encoder per kind group" degradation
    /// that row's own doc named, generalized from size-1 groups to a
    /// caller-chosen two-way split. `None` (the default) keeps that
    /// original per-position behavior. Never invalidates `resolved_steps`:
    /// which encoder a dispatch lands in does not change its compiled
    /// pipeline.
    #[cfg(feature = "instrument")]
    encoder_split_at: Option<usize>,
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
    /// One scratch buffer per plan position, `None` for every position
    /// other than a `CachedAttention` op -- redesign §4c
    /// ([`NumericRewrite::ContextSplitMerge`]): the split kernel's own
    /// write set, the merge kernel's own read set (`crate::msl::Binding::
    /// Scratch`'s own doc). Sized once, for the compiled MAXIMUM split
    /// count (`crate::sized::ATTENTION_SPLIT_MAX`) regardless of which
    /// `NumericPolicy` this call happens to run under -- the same "always
    /// allocate room for the compiled ceiling" stance [`BufferArena`]'s own
    /// output slots take, so a later call narrowing/widening the active
    /// policy never needs a resize. Deliberately plan-owned rather than
    /// folded into [`BufferArena`]'s own whole-buffer reuse: that arena's
    /// slot-sharing invariant ("whole-buffer sharing only", that struct's
    /// own doc) is unverified against a SECOND buffer per position in this
    /// pass -- see `attention-kernel-design.md` §4c risk 1. A future slice
    /// folding this into the arena's own reuse is free to do so without
    /// changing this field's read side ([`attention_scratch_buffer`]).
    #[cfg(feature = "metal-plan-stable-buffers")]
    attention_scratch: core::cell::OnceCell<Vec<Option<MetalBuffer>>>,
    /// Per-position `(pipeline, bindings, grid)` resolved once, lazily, on
    /// this plan's first [`execute_plan_with_placements`] call -- see
    /// [`resolve_steps`]'s own doc for why a hit no longer builds
    /// [`kernel_cache_key`]'s `String` or [`kernel_dispatch_shape`]'s `Vec`
    /// per step. `None` until that first call. Rebuilt whole when
    /// [`Plan::set_math_mode`] changes the mode a prior resolution used,
    /// since [`pipeline_for`] compiles one pipeline per mode. Populated and
    /// read only by the placement executors; every other executor leaves it
    /// `None` for this plan's whole life, which costs nothing beyond the
    /// one empty `RefCell`.
    resolved_steps: RefCell<Option<ResolvedSteps>>,
    /// [`HazardState`]'s own doc: [`execute_plan_with_placements`]'s hazard
    /// tracker and its per-step input-pointer scratch, reused call-to-call
    /// instead of rebuilt every call.
    hazard_state: RefCell<HazardState>,
    /// [`execute_plan_with_placements`]'s per-node buffer map -- plan-owned
    /// and NEVER rebuilt fresh (`BTreeMap::new()`) call-to-call, so its
    /// already-allocated tree nodes are reused for every call's `insert`/
    /// `remove` cycle instead of an empty map paying that allocation again
    /// (ROW 303's residual). Every call still writes a fresh entry for every
    /// block/position exactly as before this landing -- only WHERE the map
    /// lives changed, so content correctness is unaffected regardless of
    /// residency; [`Plan::block_identity`] layers a further, residency-gated
    /// skip of the upload itself on top.
    device_buffers: RefCell<BTreeMap<NodeId, DeviceBuffer>>,
    /// Per-[`Prepared::block_nodes`]-position `(pointer, byte_length)` this
    /// plan last saw for a RESIDENT block -- `None` for a position that is
    /// not in [`Plan::resident_nodes`], or that has not been uploaded yet.
    /// [`execute_plan_with_placements`] compares this against the CURRENT
    /// call's own block identity to decide whether that position's entry in
    /// [`Plan::device_buffers`] can be trusted as-is (no content hash: an
    /// address match alone would wrongly reuse a freshly reallocated,
    /// unrelated buffer that happens to land at a freed one's old address --
    /// the RESIDENT gate is the caller's own promise that this address never
    /// moves and never changes content, the same promise
    /// [`Plan::mark_resident`]'s no-copy upload path already trusts).
    block_identity: RefCell<Vec<Option<(usize, usize)>>>,
}

/// [`Plan::resolved_steps`]'s payload -- the [`MathMode`] it was built
/// under, so a later [`Plan::set_math_mode`] call is detected and triggers a
/// rebuild rather than silently serving stale pipelines for the old mode.
/// Keyed on `math_mode` alone, not `numeric_policy`: `numeric_policy` cannot
/// move post-construction ([`Plan::numeric_policy`]'s own doc), so
/// `math_mode` -- narrowable within the bound policy via
/// [`Plan::set_math_mode`] -- is the only axis that can still go stale here.
struct ResolvedSteps {
    math_mode: MathMode,
    steps: Vec<ResolvedStep>,
}

/// One [`Plan::prepared`] position's compiled pipeline plus the two other
/// per-op values [`encode_op`] needs to dispatch it -- resolved once by
/// [`resolve_steps`] instead of every step re-deriving [`kernel_cache_key`]
/// (a `String`) and [`kernel_dispatch_shape`] (a `Vec<Binding>`) just to
/// look the same pipeline up again.
struct ResolvedStep {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    bindings: Vec<Binding>,
    grid: GridSpec,
    /// Redesign §4c: `Some` exactly when [`crate::msl::
    /// cached_attention_merge_needed`] admits `ContextSplitMerge` for this
    /// position's `CachedAttention` op -- its own compiled pipeline (a
    /// SEPARATE `MTLLibrary`/pipeline-cache entry, keyed on the `_merge`
    /// entry name [`crate::msl::emit_cached_attention_merge`] builds), read
    /// back out of the scratch buffer `bindings`' own trailing
    /// `Binding::Scratch` slot wrote.
    merge: Option<ResolvedMergeStep>,
}

/// [`ResolvedStep::merge`]'s payload -- the SAME triple `ResolvedStep`
/// itself carries, so [`encode_op`]'s second dispatch binds and dispatches
/// it identically to the first, just against a different pipeline/bindings/
/// grid and a scratch-buffer placement instead of the position's real
/// output.
struct ResolvedMergeStep {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    bindings: Vec<Binding>,
    grid: GridSpec,
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

/// The name [`Plan::mark_resident`] proved this node's block input is bound
/// to for the life of the served model, or `None` when the node was never
/// classified resident. This is the identity [`upload_resident_copy`] caches
/// on -- see that function's own doc for why a host address cannot serve as
/// this identity instead.
fn resident_name(plan: &Plan, node: NodeId) -> Option<&str> {
    plan.resident_nodes
        .contains(&node)
        .then(|| plan.program[node.0 as usize].name())
        .flatten()
}

/// A block's own `(pointer, byte_length)` -- every [`QuantizedBlock`] variant
/// is a borrowed slice, so this is the address identity
/// [`block_buffer_reusable`] compares, never the bytes themselves.
fn block_identity_key(block: &QuantizedBlock<'_>) -> (usize, usize) {
    match block {
        QuantizedBlock::Float32(data) => (data.as_ptr().cast::<()>() as usize, size_of_val(*data)),
        QuantizedBlock::Int32(data) => (data.as_ptr().cast::<()>() as usize, size_of_val(*data)),
        QuantizedBlock::Q4K(bytes)
        | QuantizedBlock::Q5K(bytes)
        | QuantizedBlock::Q3K(bytes)
        | QuantizedBlock::Q6K(bytes)
        | QuantizedBlock::Q8_0(bytes)
        | QuantizedBlock::Q4_0(bytes)
        | QuantizedBlock::Q5_1(bytes)
        | QuantizedBlock::Q2K(bytes)
        | QuantizedBlock::Iq4Nl(bytes)
        | QuantizedBlock::Iq2Xs(bytes)
        | QuantizedBlock::Iq3Xxs(bytes)
        | QuantizedBlock::Float16(bytes)
        | QuantizedBlock::BFloat16(bytes) => (bytes.as_ptr().cast::<()>() as usize, bytes.len()),
    }
}

/// Whether a RESIDENT block-input position's existing
/// [`Plan::device_buffers`] entry can be trusted as-is this call, so
/// [`execute_plan_with_placements`] can skip re-uploading it entirely.
/// `resident` alone is not enough -- see [`Plan::block_identity`]'s own doc
/// for why an address match still requires the caller's own residency
/// promise before it is trusted, and why no content hash backs it up.
fn block_buffer_reusable(
    resident: bool,
    previous: Option<(usize, usize)>,
    current: (usize, usize),
) -> bool {
    resident && previous == Some(current)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod block_buffer_reusable_tests {
    //! Pure, GPU-free proof of the reuse decision
    //! [`execute_plan_with_placements`]'s block-upload loop relies on --
    //! `plan_pipeline_alloc_count.rs` proves the allocation count this
    //! decision buys end to end, on a real Metal device; this proves the
    //! decision itself, so a regression here fails fast without a GPU.

    use super::block_buffer_reusable;

    #[test]
    fn a_resident_block_at_the_same_address_and_length_is_reused() {
        let identity = (0x1000, 64);
        assert!(block_buffer_reusable(true, Some(identity), identity));
    }

    #[test]
    fn a_resident_block_whose_address_moved_is_rebuilt() {
        let previous = (0x1000, 64);
        let current = (0x2000, 64);
        assert!(!block_buffer_reusable(true, Some(previous), current));
    }

    #[test]
    fn a_resident_block_whose_length_changed_is_rebuilt() {
        let previous = (0x1000, 64);
        let current = (0x1000, 128);
        assert!(!block_buffer_reusable(true, Some(previous), current));
    }

    #[test]
    fn a_non_resident_block_is_never_reused_even_at_the_same_address() {
        let identity = (0x1000, 64);
        assert!(!block_buffer_reusable(false, Some(identity), identity));
    }

    #[test]
    fn a_first_call_with_no_recorded_identity_is_never_reused() {
        assert!(!block_buffer_reusable(true, None, (0x1000, 64)));
    }
}

impl Plan {
    /// Narrows this plan's compiled [`MathMode`] from [`MathMode::default`]
    /// (`Relaxed`) -- never widens it, and never touches
    /// [`Self::numeric_policy`]. Safe to call any time before an
    /// `execute_plan*` call -- `pipeline_for`'s cache key folds the mode in,
    /// so switching a plan's mode between calls never hands back a pipeline
    /// compiled for the other one.
    ///
    /// [`MathMode`] is a 3-rung compiler flag; [`NumericPolicy`] is the
    /// richer, orthogonal 5-permission set this plan's bound program was
    /// actually constructed under ([`plan`]/[`plan_named`]'s own argument,
    /// fixed for this plan's whole life -- see [`Self::numeric_policy`]'s
    /// own doc for why there is no setter for it). `math_mode` may only
    /// request what `numeric_policy` already grants
    /// (`metal_math_mode_as_numeric_policy`); requesting `Fast` on a plan
    /// bound under `bit_exact()` is a caller error, not a silent widening,
    /// so this returns [`MetalError::NumericPolicyMismatch`] instead of
    /// mutating `numeric_policy` the way this setter used to.
    ///
    /// # Errors
    /// [`MetalError::NumericPolicyMismatch`] when `math_mode` needs a
    /// permission `self.numeric_policy` does not grant.
    pub fn set_math_mode(&mut self, math_mode: MathMode) -> Result<(), MetalError> {
        let requested = metal_math_mode_as_numeric_policy(math_mode);
        if self.numeric_policy.grants(requested) {
            self.math_mode = math_mode;
            Ok(())
        } else {
            Err(MetalError::NumericPolicyMismatch {
                bound: self.numeric_policy,
                requested,
            })
        }
    }

    /// The [`NumericPolicy`] this plan's bound program was constructed under
    /// ([`plan`]/[`plan_named`]'s own `numeric_policy` argument) -- fixed
    /// for this plan's whole life. There is no setter: the bound program's
    /// topology (which identity eliminations fired, whether chain/reduce-
    /// epilogue fusion ran) is decided once, at bind time, and cannot be
    /// patched after without rebuilding it from `blocks`, which this type
    /// does not retain (the data is call-scoped, not plan-scoped -- see
    /// `Plan`'s own struct doc). Build a fresh [`Plan`] via [`plan`]/
    /// [`plan_named`] with a different policy instead; use
    /// [`Self::check_numeric_policy`] to confirm a `Plan` from elsewhere (a
    /// test fixture, a cached plan) matches before trusting it.
    #[must_use]
    pub fn numeric_policy(&self) -> NumericPolicy {
        self.numeric_policy
    }

    /// `Ok(())` when `desired` matches the policy this plan was bound
    /// under, `Err` otherwise. See [`Self::numeric_policy`]'s own doc for
    /// why there is no in-place rebind.
    ///
    /// # Errors
    /// [`MetalError::NumericPolicyMismatch`] when `desired` differs from
    /// [`Self::numeric_policy`].
    pub fn check_numeric_policy(&self, desired: NumericPolicy) -> Result<(), MetalError> {
        if self.numeric_policy == desired {
            Ok(())
        } else {
            Err(MetalError::NumericPolicyMismatch {
                bound: self.numeric_policy,
                requested: desired,
            })
        }
    }

    /// Overrides this plan's [`DispatchType`] from [`DispatchType::default`]
    /// (`Concurrent`). Safe to call any time before an `execute_plan*` call
    /// -- unlike [`Self::set_math_mode`], this never invalidates
    /// `resolved_steps`: a plan's compiled pipelines are the same regardless
    /// of which encoder dispatch mode runs them.
    pub fn set_dispatch_type(&mut self, dispatch_type: DispatchType) {
        self.dispatch_type = dispatch_type;
    }

    /// This plan's currently applied [`MathMode`] -- the read side of
    /// [`Self::set_math_mode`], added so a caller (or a test) can prove a
    /// build path actually applied a mode rather than only asserting that a
    /// setter was called somewhere upstream.
    #[must_use]
    pub fn math_mode(&self) -> MathMode {
        self.math_mode
    }

    /// This plan's currently applied [`DispatchType`] -- [`Self::math_mode`]'s
    /// counterpart for [`Self::set_dispatch_type`].
    #[must_use]
    pub fn dispatch_type(&self) -> DispatchType {
        self.dispatch_type
    }

    /// Overrides this plan's ROW 329 encoder-split position from `None`
    /// (one stage-sampled encoder per position, ROW 309's original
    /// fallback). Safe to call any time before an
    /// [`execute_plan_with_placements_dispatch_timed`] call; every other
    /// executor -- including this same function's `AtDispatchBoundary`
    /// branch, on a device that has it -- ignores this field entirely.
    #[cfg(feature = "instrument")]
    pub fn set_encoder_split_at(&mut self, position: Option<usize>) {
        self.encoder_split_at = position;
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

    /// The plan-cache key `resolve_steps` computes for each program
    /// position (`omega::msl::kernel_cache_key`'s own doc: the shared
    /// structural + compile-option identity of that position's resolved
    /// `BoundOp`, this plan's own [`MathMode`] included) -- exposed so a
    /// caller can assert exactly which compiled kernel variant a program
    /// resolves to (operand order, reduced axis, packed-row shape, math
    /// mode) without re-deriving `kernel_cache_key`'s own logic outside this
    /// crate.
    ///
    /// # Errors
    /// Propagates the same rejection `kernel_cache_key` raises for an
    /// unsupported dtype.
    pub fn kernel_keys(&self) -> Result<Vec<String>, MetalError> {
        self.prepared
            .resolved
            .iter()
            .map(|bound| {
                kernel_cache_key(bound, &self.packed_operands, self.numeric_policy)
                    .map(|mut key| {
                        key.push(self.math_mode.cache_token());
                        key
                    })
                    .map_err(MetalError::from)
            })
            .collect()
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
            QuantizedBlock::Q2K(_) => Some((*node, PackedCodec::Q2K)),
            QuantizedBlock::Q3K(_) => Some((*node, PackedCodec::Q3K)),
            QuantizedBlock::Q4K(_) => Some((*node, PackedCodec::Q4K)),
            QuantizedBlock::Q5K(_) => Some((*node, PackedCodec::Q5K)),
            QuantizedBlock::Q6K(_) => Some((*node, PackedCodec::Q6K)),
            QuantizedBlock::Q8_0(_) => Some((*node, PackedCodec::Q8_0)),
            QuantizedBlock::Q4_0(_) => Some((*node, PackedCodec::Q4_0)),
            QuantizedBlock::Float16(_) => Some((*node, PackedCodec::Float16)),
            QuantizedBlock::BFloat16(_) => Some((*node, PackedCodec::BFloat16)),
            // decode-only codecs so far (CPU-only, see `proxima_tensor::cpu`)
            // -- no `PackedCodec`/unpack-kernel entry exists yet, so these
            // fall out of `packed_operands` exactly like `Float32` and hit
            // `reject_unsupported_gpu_dtype`'s ordinary rejection rather
            // than a silent, wrong-shape upload.
            QuantizedBlock::Q5_1(_)
            | QuantizedBlock::Iq4Nl(_)
            | QuantizedBlock::Iq2Xs(_)
            | QuantizedBlock::Iq3Xxs(_)
            | QuantizedBlock::Float32(_)
            | QuantizedBlock::Int32(_) => None,
        })
        .collect()
}

/// Describes the borrowed payloads in one expert substitution table.
///
/// This is the lowering boundary for HOBBIT: the residency decision remains
/// a borrowed [`ExpertSource`] while Metal receives codec-aware byte spans.
/// Keeping the spans separate is important because Q2_K, Q4_K, and Q6_K
/// blocks do not have the same byte width; joining them into one packed operand would
/// make the second expert's address wrong.
///
/// # Errors
/// Returns the existing typed expert-source error when an entry is not a
/// packed codec. No bytes are copied.
pub fn expert_payload_descriptors(
    node: NodeId,
    source: &proxima_tensor::cpu::ExpertSource<'_>,
) -> Result<Vec<ExpertPayloadDescriptor>, MetalError> {
    let mut descriptors = Vec::with_capacity(source.entries().len());
    let mut byte_offset = 0usize;
    for entry in source.entries() {
        let codec = match entry.block {
            QuantizedBlock::Q2K(_) => PackedCodec::Q2K,
            QuantizedBlock::Q3K(_) => PackedCodec::Q3K,
            QuantizedBlock::Q4K(_) => PackedCodec::Q4K,
            QuantizedBlock::Q6K(_) => PackedCodec::Q6K,
            _ => {
                return Err(MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "mixed expert lowering only has Q2_K, Q3_K, Q4_K, and Q6_K decoders",
                });
            }
        };
        let bytes = entry
            .block
            .packed_bytes()
            .ok_or(MetalError::ExpertSourceUnsupported {
                node,
                reason: "expert entries must use packed bytes",
            })?;
        let byte_length = bytes.len();
        descriptors.push(ExpertPayloadDescriptor {
            expert_index: u32::try_from(descriptors.len()).map_err(|_| {
                MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "expert index exceeds the Metal descriptor ABI",
                }
            })?,
            codec,
            byte_offset,
            byte_length,
            out_dim: entry.out_dim,
            in_dim: entry.in_dim,
            epoch: entry.epoch,
        });
        byte_offset =
            byte_offset
                .checked_add(byte_length)
                .ok_or(MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "expert payload byte span overflowed",
                })?;
    }
    Ok(descriptors)
}

/// Builds the dense route descriptor table and a compact payload arena for a
/// source snapshot. The descriptor table keeps one record per original expert
/// index so the existing Metal gather ABI remains O(1); unselected records
/// carry no payload bytes and are never valid routed entries for this step.
pub fn selected_expert_payloads(
    node: NodeId,
    source: &proxima_tensor::cpu::ExpertSource<'_>,
) -> Result<(Vec<u8>, Vec<ExpertPayloadDescriptor>), MetalError> {
    let selected_ids = source.selected_expert_ids();
    if selected_ids.is_some_and(|ids| ids.is_empty()) {
        return Err(MetalError::ExpertSourceUnsupported {
            node,
            reason: "selected expert ID list is empty",
        });
    }
    let mut payload_bytes = Vec::new();
    let mut descriptors = Vec::with_capacity(source.entries().len());
    for (expert_index, entry) in source.entries().iter().enumerate() {
        let codec = match entry.block {
            QuantizedBlock::Q2K(_) => PackedCodec::Q2K,
            QuantizedBlock::Q3K(_) => PackedCodec::Q3K,
            QuantizedBlock::Q4K(_) => PackedCodec::Q4K,
            QuantizedBlock::Q6K(_) => PackedCodec::Q6K,
            _ => {
                return Err(MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "mixed expert lowering only has Q2_K, Q3_K, Q4_K, and Q6_K decoders",
                });
            }
        };
        let bytes = entry
            .block
            .packed_bytes()
            .ok_or(MetalError::ExpertSourceUnsupported {
                node,
                reason: "expert entries must use packed bytes",
            })?;
        let selected =
            selected_ids.is_none_or(|ids| ids.iter().any(|id| *id == expert_index as u32));
        let (byte_offset, byte_length) = if selected {
            let byte_offset = payload_bytes.len();
            payload_bytes.extend_from_slice(bytes);
            (byte_offset, bytes.len())
        } else {
            (0, 0)
        };
        descriptors.push(ExpertPayloadDescriptor {
            expert_index: u32::try_from(expert_index).map_err(|_| {
                MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "expert index exceeds the Metal descriptor ABI",
                }
            })?,
            codec,
            byte_offset,
            byte_length,
            out_dim: entry.out_dim,
            in_dim: entry.in_dim,
            epoch: entry.epoch,
        });
    }
    if let Some(ids) = selected_ids {
        for id in ids {
            if usize::try_from(*id)
                .ok()
                .is_none_or(|index| index >= source.entries().len())
            {
                return Err(MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "selected expert ID is outside the source table",
                });
            }
        }
    }
    Ok((payload_bytes, descriptors))
}

fn selected_expert_arena_descriptors(
    node: NodeId,
    source: &proxima_tensor::cpu::ExpertSource<'_>,
    arena: proxima_tensor::cpu::ExpertPayloadArena<'_>,
) -> Result<Vec<ExpertPayloadDescriptor>, MetalError> {
    let selected_ids = source.selected_expert_ids();
    let spans = arena.spans();
    let mut descriptors = Vec::with_capacity(source.entries().len());
    for (expert_index, entry) in source.entries().iter().enumerate() {
        let codec = match entry.block {
            QuantizedBlock::Q2K(_) => PackedCodec::Q2K,
            QuantizedBlock::Q3K(_) => PackedCodec::Q3K,
            QuantizedBlock::Q4K(_) => PackedCodec::Q4K,
            QuantizedBlock::Q6K(_) => PackedCodec::Q6K,
            _ => {
                return Err(MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "mixed expert lowering only has Q2_K, Q3_K, Q4_K, and Q6_K decoders",
                });
            }
        };
        let selected = selected_ids.is_none_or(|ids| {
            ids.iter()
                .any(|id| *id == u32::try_from(expert_index).unwrap_or(u32::MAX))
        });
        let span = spans.get(expert_index).and_then(|span| *span);
        let (byte_offset, byte_length) = if selected {
            let span = span.ok_or(MetalError::ExpertSourceUnsupported {
                node,
                reason: "selected expert has no arena span",
            })?;
            let end = usize::try_from(span.offset)
                .ok()
                .and_then(|offset| {
                    usize::try_from(span.length)
                        .ok()
                        .and_then(|length| offset.checked_add(length))
                })
                .ok_or(MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "expert arena span endpoint overflowed",
                })?;
            if end > arena.bytes().len() {
                return Err(MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "expert arena span exceeds payload bytes",
                });
            }
            (
                usize::try_from(span.offset).unwrap_or(usize::MAX),
                usize::try_from(span.length).unwrap_or(usize::MAX),
            )
        } else {
            (0, 0)
        };
        descriptors.push(ExpertPayloadDescriptor {
            expert_index: u32::try_from(expert_index).map_err(|_| {
                MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "expert index exceeds the Metal descriptor ABI",
                }
            })?,
            codec,
            byte_offset,
            byte_length,
            out_dim: entry.out_dim,
            in_dim: entry.in_dim,
            epoch: entry.epoch,
        });
    }
    Ok(descriptors)
}

/// Packs descriptor records into the byte layout consumed by the MSL
/// `ExpertPayloadDescriptor` struct. The caller owns the returned bytes and
/// may upload them beside the borrowed payload arena without touching the
/// tensor graph or changing the residency FSM.
pub fn pack_expert_payload_descriptors(
    node: NodeId,
    descriptors: &[ExpertPayloadDescriptor],
) -> Result<Vec<u8>, MetalError> {
    let mut bytes = Vec::with_capacity(descriptors.len() * 32);
    for descriptor in descriptors {
        let codec_tag: u32 = match descriptor.codec {
            PackedCodec::Q2K => 1,
            PackedCodec::Q3K => 4,
            PackedCodec::Q4K => 2,
            PackedCodec::Q6K => 3,
            _ => {
                return Err(MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "mixed expert descriptor has no MSL decoder",
                });
            }
        };
        let byte_offset = u32::try_from(descriptor.byte_offset).map_err(|_| {
            MetalError::ExpertSourceUnsupported {
                node,
                reason: "expert payload byte offset exceeds the Metal descriptor ABI",
            }
        })?;
        let byte_length = u32::try_from(descriptor.byte_length).map_err(|_| {
            MetalError::ExpertSourceUnsupported {
                node,
                reason: "expert payload byte length exceeds the Metal descriptor ABI",
            }
        })?;
        let epoch =
            u32::try_from(descriptor.epoch).map_err(|_| MetalError::ExpertSourceUnsupported {
                node,
                reason: "expert promotion epoch exceeds the Metal descriptor ABI",
            })?;
        bytes.extend_from_slice(&descriptor.expert_index.to_ne_bytes());
        bytes.extend_from_slice(&codec_tag.to_ne_bytes());
        bytes.extend_from_slice(&byte_offset.to_ne_bytes());
        bytes.extend_from_slice(&byte_length.to_ne_bytes());
        bytes.extend_from_slice(&descriptor.out_dim.to_ne_bytes());
        bytes.extend_from_slice(&descriptor.in_dim.to_ne_bytes());
        bytes.extend_from_slice(&epoch.to_ne_bytes());
        bytes.extend_from_slice(&0_u32.to_ne_bytes());
    }
    Ok(bytes)
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
    numeric_policy: NumericPolicy,
) -> Result<Plan, MetalError> {
    #[cfg(feature = "instrument")]
    let prepare_started = read_ticks();
    let prepared = prepare(program, symbols, blocks, outputs, numeric_policy)?;
    #[cfg(feature = "instrument")]
    {
        counter!(PREPARE_CALLS, 1);
        counter!(PREPARE_TICKS, elapsed_ticks(prepare_started));
    }
    // `prepare` already built this attribution once, off the same `blocks`
    // argument, checked count- and shape-consistent against `block_nodes`
    // (ROW 327) -- cloning it here instead of re-zipping `block_nodes`
    // against `blocks` a second time is what keeps this field and
    // `prepared`'s own from being two independent computations that could
    // silently drift apart.
    let packed_operands = prepared.packed_operands.clone();
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
        math_mode: numeric_policy_as_metal_math_mode(numeric_policy),
        numeric_policy,
        dispatch_type: DispatchType::default(),
        #[cfg(feature = "instrument")]
        encoder_split_at: None,
        #[cfg(feature = "metal-plan-stable-buffers")]
        arena: core::cell::OnceCell::new(),
        #[cfg(feature = "metal-plan-stable-buffers")]
        uniforms: core::cell::OnceCell::new(),
        #[cfg(feature = "metal-plan-stable-buffers")]
        attention_scratch: core::cell::OnceCell::new(),
        resolved_steps: RefCell::new(None),
        hazard_state: RefCell::new(HazardState::new()),
        device_buffers: RefCell::new(BTreeMap::new()),
        block_identity: RefCell::new(Vec::new()),
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
    numeric_policy: NumericPolicy,
) -> Result<Evaluated, MetalError> {
    let resolved_plan = plan(program, symbols, blocks, outputs, numeric_policy)?;
    execute_plan(&resolved_plan, blocks)
}

/// Runs an already-resolved [`Plan`] against fresh block data. This is the
/// serving-loop entry point: the plan is built once, this is called per
/// token, and none of `infer`/`bind`/codec-resolution happens here.
///
/// # Errors
/// Propagates block-codec and Metal driver failures.
pub fn execute_plan(plan: &Plan, blocks: &[QuantizedBlock<'_>]) -> Result<Evaluated, MetalError> {
    execute_plan_inner(plan, blocks, &BTreeMap::new())
}

fn execute_plan_inner(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    expert_buffers: &BTreeMap<NodeId, ExpertSourceBuffers>,
) -> Result<Evaluated, MetalError> {
    let debug_timing = std::env::var_os("PROXIMA_DEBUG_SEGMENT_HOST").is_some();
    let prepared = &plan.prepared;
    let packed_operands = &plan.packed_operands;

    // Partitioned routed programs may retain the same named expert input at
    // more than one dense position. Alias the staged descriptor by name at
    // the prepared-plan boundary so renumbering cannot leave the gather
    // reading an ordinary (unbound) input.
    let mut effective_expert_buffers = expert_buffers.clone();
    for (source_node, buffers) in expert_buffers {
        let Some(source_name) = plan.program[source_node.0 as usize].name() else {
            continue;
        };
        for (position, operation) in plan.program.iter().enumerate() {
            if operation.name() == Some(source_name) {
                effective_expert_buffers
                    .entry(NodeId(position as u32))
                    .or_insert_with(|| buffers.clone());
            }
        }
    }
    if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
        eprintln!(
            "effective expert source nodes={:?}",
            effective_expert_buffers.keys().collect::<Vec<_>>()
        );
        for index in 0..plan.program.len().min(24) {
            eprintln!("plan op {} name={:?}", index, plan.program[index].name());
        }
    }

    let (device, queue) = device_and_queue()?;

    let mut device_buffers: BTreeMap<NodeId, DeviceBuffer> = BTreeMap::new();
    #[cfg(feature = "instrument")]
    let mut ordinary_upload_bytes = 0usize;
    #[cfg(feature = "instrument")]
    let mut ordinary_upload_blocks = 0usize;
    #[cfg(feature = "instrument")]
    let block_upload_started = read_ticks();
    for (node, block, dtype) in
        ordinary_block_uploads(&prepared.block_nodes, blocks, &plan.block_dtypes, |node| {
            effective_expert_buffers.contains_key(&node)
        })
    {
        let resident_name = resident_name(plan, node);
        if let Some((buffer, offset)) = cross_plan_resident_reuse(
            resident_name,
            block_identity_key(&block).0 as *const c_void,
            block_identity_key(&block).1,
        )? {
            device_buffers.insert(node, (buffer, offset));
            continue;
        }
        #[cfg(feature = "instrument")]
        {
            ordinary_upload_bytes = ordinary_upload_bytes.saturating_add(block_byte_len(&block));
            ordinary_upload_blocks += 1;
        }
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
            counter!(BLOCK_OFFERED_BYTES, block_byte_len(&block) as u64);
        }
        let buffer = match block {
            QuantizedBlock::Float32(data) => upload_block(
                &device,
                data,
                node,
                dtype,
                plan.program[node.0 as usize].name(),
                resident_name,
            )?,
            QuantizedBlock::Int32(data) => {
                upload_block_int32_as_float(&device, data, resident_name)?
            }
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
            | QuantizedBlock::Q5_1(bytes)
            | QuantizedBlock::Q2K(bytes)
            | QuantizedBlock::Iq4Nl(bytes)
            | QuantizedBlock::Iq2Xs(bytes)
            | QuantizedBlock::Iq3Xxs(bytes)
            | QuantizedBlock::Float16(bytes)
            | QuantizedBlock::BFloat16(bytes) => {
                upload_packed_bytes(&device, bytes, resident_name)?
            }
        };
        device_buffers.insert(node, buffer);
    }
    #[cfg(feature = "instrument")]
    if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
        eprintln!(
            "metal ordinary uploads blocks={ordinary_upload_blocks} bytes={ordinary_upload_bytes}"
        );
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
    let encoder = EncoderGuard::new(command_buffer.computeCommandEncoder().ok_or_else(|| {
        MetalError::CompileFailed {
            log: "command buffer refused to hand out a compute encoder".to_string(),
        }
    })?);

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
    if debug_timing {
        eprintln!(
            "metal expert prepared_nodes={:?}",
            prepared
                .resolved
                .iter()
                .map(|bound| bound.node)
                .collect::<Vec<_>>()
        );
        for (position, retired) in prepared.retires.iter().enumerate() {
            if retired.contains(&NodeId(16))
                || retired.contains(&NodeId(18))
                || retired.contains(&NodeId(22))
            {
                eprintln!(
                    "metal expert retirement node16_or_18 position={position} nodes={retired:?}"
                );
            }
        }
        for (position, bound) in prepared.resolved.iter().enumerate() {
            if bound.all_read_sources().any(|(source, _, lookup)| {
                *source == NodeId(22)
                    || lookup
                        .as_ref()
                        .is_some_and(|item| item.indices == NodeId(22))
            }) {
                eprintln!(
                    "metal expert node22 read_at_position={position} bound={:?}",
                    bound.node
                );
            }
            if matches!(bound.node, NodeId(16) | NodeId(18)) {
                eprintln!(
                    "metal expert retirement position={} node={:?} retires={:?}",
                    position, bound.node, prepared.retires[position]
                );
            }
        }
    }
    for (position, bound) in prepared.resolved.iter().enumerate() {
        if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
            eprintln!(
                "resolved bound node={:?} kind={} operands={:?}",
                bound.node,
                bound.kind.name(),
                bound
                    .operands()
                    .iter()
                    .map(|(node, _, lookup)| (
                        *node,
                        lookup.is_some(),
                        plan.program.get(node.0 as usize).and_then(Op::name)
                    ))
                    .collect::<Vec<_>>()
            );
        }
        let expert_buffers = expert_buffers_for(bound, &effective_expert_buffers)?;
        let fault = encode_op(
            &device,
            &encoder,
            &mut device_buffers,
            bound,
            packed_operands,
            None,
            None,
            plan.math_mode,
            plan.numeric_policy,
            None,
            None,
            None,
            expert_buffers,
        )?;
        if let Some((fault_buffer, gathers)) = fault {
            pending_faults.push((bound, fault_buffer, gathers));
        }
        if debug_timing && matches!(bound.node, NodeId(22) | NodeId(30)) {
            eprintln!(
                "metal expert post_encode node={:?} has_buffer={} buffer_keys={:?}",
                bound.node,
                device_buffers.contains_key(&bound.node),
                device_buffers.keys().copied().collect::<Vec<_>>()
            );
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
            // gather index buffers are tiny, and keeping them until the
            // command completes avoids retiring a computed index before a
            // backend binding that references it through `Binding::Indices`.
            if prepared.index_nodes.contains(retired) {
                continue;
            }
            // Keep a value when a later bound op still names it. This is
            // required for segment plans whose resolved order differs from
            // source NodeId order (notably computed gather intermediates).
            if prepared.resolved[position + 1..].iter().any(|later| {
                later.all_read_sources().any(|(source, _, lookup)| {
                    *source == *retired
                        || lookup
                            .as_ref()
                            .is_some_and(|access| access.indices == *retired)
                })
            }) {
                continue;
            }
            // A fused lowering may issue more than one dispatch for the
            // current bound op.  Its generated binding list is the source of
            // truth for those internal reads, while the resolved successor
            // walk only sees later bound ops.  Keep a source named by this
            // op until the command buffer finishes so the second dispatch
            // cannot observe a retired input.
            if bound.all_read_sources().any(|(source, _, lookup)| {
                *source == *retired
                    || lookup
                        .as_ref()
                        .is_some_and(|access| access.indices == *retired)
            }) {
                continue;
            }
            device_buffers.remove(retired);
        }
        #[cfg(feature = "metal-buffer-pool")]
        for retired in &prepared.retires[position] {
            if prepared.index_nodes.contains(retired) {
                continue;
            }
            if let Some((buffer, _offset)) = device_buffers.remove(retired)
                && let Some(&(bucket, dtype)) = output_meta.get(retired)
            {
                reclaim_stash.push((buffer, bucket, dtype));
            }
        }
    }
    encoder.finish();

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

    let evaluated = finish(plan, &device_buffers, &BTreeSet::new(), None)?;

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

/// Produces exactly the named-block inputs that retain their ordinary device
/// buffer binding. A substituted expert source has its own mixed payload and
/// descriptor buffers, so retaining its original stack here would allocate a
/// second device copy that no emitted mixed-expert kernel can read.
fn ordinary_block_uploads<'plan, 'block, Substituted>(
    block_nodes: &'plan [NodeId],
    blocks: &'plan [QuantizedBlock<'block>],
    block_dtypes: &'plan [DType],
    is_substituted: Substituted,
) -> impl Iterator<Item = (NodeId, QuantizedBlock<'block>, DType)> + 'plan
where
    Substituted: Fn(NodeId) -> bool + 'plan,
{
    block_nodes
        .iter()
        .copied()
        .zip(blocks.iter().copied())
        .zip(block_dtypes.iter().copied())
        .filter_map(move |((node, block), dtype)| {
            (!is_substituted(node)).then_some((node, block, dtype))
        })
}

fn expert_buffers_for<'a>(
    bound: &BoundOp,
    expert_buffers: &'a BTreeMap<NodeId, ExpertSourceBuffers>,
) -> Result<Option<&'a ExpertSourceBuffers>, MetalError> {
    let mut matched = expert_buffers.iter().filter_map(|(node, buffers)| {
        bound
            .operands()
            .iter()
            .any(|(operand, _, _lookup)| *operand == *node)
            .then_some((*node, buffers))
    });
    let Some((first_node, first_buffers)) = matched.next() else {
        if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
            eprintln!(
                "expert source not matched bound={:?} kind={} source_nodes={:?}",
                bound.node,
                bound.kind.name(),
                expert_buffers.keys().collect::<Vec<_>>()
            );
            eprintln!(
                "expert source bound_operands={:?}",
                bound
                    .operands()
                    .iter()
                    .map(|(node, _, lookup)| (*node, lookup.is_some()))
                    .collect::<Vec<_>>()
            );
        }
        return Ok(None);
    };
    if matched.next().is_some() {
        return Err(MetalError::ExpertSourceUnsupported {
            node: first_node,
            reason: "one gathered operation cannot bind more than one expert source table",
        });
    }
    Ok(Some(first_buffers))
}

fn stage_expert_source_reusing(
    device: &ProtocolObject<dyn MTLDevice>,
    node: NodeId,
    source: &proxima_tensor::cpu::ExpertSource<'_>,
    previous: Option<StagedExpertSource>,
) -> Result<StagedExpertSource, MetalError> {
    #[cfg(feature = "instrument")]
    let stage_ticks_started = read_ticks();
    let stage_started = Instant::now();
    let packed_arena = source.packed_arena();
    let (owned_payload_bytes, descriptors) = match packed_arena {
        Some(arena) => (
            None,
            selected_expert_arena_descriptors(node, source, arena)?,
        ),
        None => {
            let (payload, descriptors) = selected_expert_payloads(node, source)?;
            (Some(payload), descriptors)
        }
    };
    let payload_bytes = owned_payload_bytes
        .as_deref()
        .or_else(|| packed_arena.map(|arena| arena.bytes()))
        .ok_or(MetalError::ExpertSourceUnsupported {
            node,
            reason: "expert source produced no payload bytes",
        })?;
    let descriptor_records = descriptors;
    if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
        eprintln!(
            "metal expert source node={node:?} entries={} selected_experts={} selected_payload_bytes={}",
            source.entries().len(),
            source
                .selected_expert_ids()
                .map_or(source.entries().len(), <[_]>::len),
            payload_bytes.len()
        );
        if let Some(selected) = source.selected_expert_ids() {
            eprintln!("metal selected expert ids={selected:?}");
            for expert in selected {
                if let Some(descriptor) = descriptor_records.get(*expert as usize) {
                    eprintln!(
                        "metal selected descriptor expert={} codec={:?} epoch={} offset={} length={}",
                        descriptor.expert_index,
                        descriptor.codec,
                        descriptor.epoch,
                        descriptor.byte_offset,
                        descriptor.byte_length
                    );
                }
            }
        }
    }
    let descriptor_bytes = pack_expert_payload_descriptors(node, &descriptor_records)?;
    let previous_payload = previous
        .as_ref()
        .map(|staged| (&staged.buffers.payloads, staged.payload_alias_address));
    let all_expert_arena = packed_arena.is_some() && source.selected_expert_ids().is_none();
    if all_expert_arena && !expert_mapping_contains(payload_bytes) {
        return Err(MetalError::ExpertSourceUnsupported {
            node,
            reason: "all-expert arena is not contained in the registered expert mmap",
        });
    }
    let (payloads, payload_offset) = if all_expert_arena {
        upload_packed_bytes(device, payload_bytes, None)?
    } else {
        reuse_or_upload_packed_bytes(device, payload_bytes, previous_payload)?
    };
    let (descriptors, _) = reuse_or_upload_packed_bytes(
        device,
        &descriptor_bytes,
        previous
            .as_ref()
            .map(|staged| (&staged.buffers.descriptors, staged.descriptor_alias_address)),
    )?;
    #[cfg(feature = "instrument")]
    {
        counter!(EXPERT_SOURCE_STAGE_CALLS, 1);
        counter!(EXPERT_SOURCE_STAGE_BYTES, payload_bytes.len() as u64);
        counter!(
            EXPERT_SOURCE_STAGE_TICKS,
            elapsed_ticks(stage_ticks_started)
        );
    }
    if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
        eprintln!(
            "metal expert source staged node={node:?} payload_bytes={} descriptor_bytes={} elapsed_us={}",
            payload_bytes.len(),
            descriptor_bytes.len(),
            stage_started.elapsed().as_micros()
        );
        for descriptor in descriptor_records.iter().take(8) {
            eprintln!(
                "metal expert descriptor expert={} codec={:?} offset={} length={} shape={}x{}",
                descriptor.expert_index,
                descriptor.codec,
                descriptor.byte_offset,
                descriptor.byte_length,
                descriptor.out_dim,
                descriptor.in_dim,
            );
        }
    }
    Ok(StagedExpertSource {
        payload_alias_address: payload_alias_address(&payload_bytes),
        descriptor_alias_address: payload_alias_address(&descriptor_bytes),
        _payload_bytes: owned_payload_bytes
            .map(|payload| host_bytes_if_aliased(payload, &payloads))
            .flatten(),
        _descriptor_bytes: host_bytes_if_aliased(descriptor_bytes, &descriptors),
        buffers: ExpertSourceBuffers {
            node,
            payloads,
            payload_offset,
            descriptors,
            descriptor_records: descriptor_records.to_vec(),
        },
    })
}

fn host_bytes_if_aliased(bytes: Vec<u8>, buffer: &MetalBuffer) -> Option<Vec<u8>> {
    (!bytes.is_empty() && buffer.contents().as_ptr() as *const u8 == bytes.as_ptr())
        .then_some(bytes)
}

fn reuse_or_upload_packed_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
    previous: Option<(&MetalBuffer, usize)>,
) -> Result<(MetalBuffer, usize), MetalError> {
    if let Some((buffer, previous_alias_address)) = previous
        && buffer.length() >= bytes.len()
        && buffer.contents().as_ptr() as usize != previous_alias_address
    {
        // The previous buffer came from a copying upload, so its storage is
        // independent of the host vector and can be overwritten after the
        // prior command buffer has completed.
        let destination = buffer.contents().as_ptr().cast::<u8>();
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len()) };
        #[cfg(feature = "instrument")]
        counter!(EXPERT_SOURCE_BUFFER_REUSES, 1);
        return Ok((buffer.clone(), 0));
    }
    upload_packed_bytes(device, bytes, None)
}

fn payload_alias_address(bytes: &[u8]) -> usize {
    if bytes.is_empty() {
        0
    } else {
        bytes.as_ptr() as usize
    }
}

fn expert_source_signature(source: &proxima_tensor::cpu::ExpertSource<'_>) -> u64 {
    let mut signature = 1469598103934665603_u64;
    let mut mix = |value: u64| {
        signature ^= value;
        signature = signature.wrapping_mul(1099511628211);
    };
    for entry in source.entries() {
        let codec = match entry.block {
            QuantizedBlock::Q2K(_) => 1_u64,
            QuantizedBlock::Q3K(_) => 2_u64,
            QuantizedBlock::Q4K(_) => 3_u64,
            QuantizedBlock::Q5K(_) => 4_u64,
            QuantizedBlock::Q6K(_) => 5_u64,
            _ => 0,
        };
        mix(codec);
        mix(entry.out_dim as u64);
        mix(entry.in_dim as u64);
        mix(entry.epoch);
        if let Some(bytes) = entry.block.packed_bytes() {
            mix(bytes.as_ptr() as usize as u64);
            mix(bytes.len() as u64);
        } else {
            mix(0);
        }
    }
    if let Some(selected) = source.selected_expert_ids() {
        mix(selected.len() as u64);
        for expert in selected {
            mix(u64::from(*expert));
        }
    } else {
        mix(u64::MAX);
    }
    signature
}

/// Refuses the transitional host-staging path when it would upload at least
/// as many expert bytes as the checkpoint block it replaces. Such a table is
/// not an offload: it only rebuilds the full expert stack in transient memory.
fn reject_non_reducing_expert_staging(
    node: NodeId,
    original: QuantizedBlock<'_>,
    source: &proxima_tensor::cpu::ExpertSource<'_>,
) -> Result<(), MetalError> {
    let original_bytes = original
        .packed_bytes()
        .ok_or(MetalError::ExpertSourceUnsupported {
            node,
            reason: "the replaced expert block must use packed bytes",
        })?
        .len();
    let entry_bytes = |expert_index: usize| {
        source
            .entries()
            .get(expert_index)
            .ok_or(MetalError::ExpertSourceUnsupported {
                node,
                reason: "selected expert ID is outside the source table",
            })?
            .block
            .packed_bytes()
            .map(<[u8]>::len)
            .ok_or(MetalError::ExpertSourceUnsupported {
                node,
                reason: "expert entries must use packed bytes",
            })
    };
    let staged_bytes = match source.selected_expert_ids() {
        Some(selected) => selected.iter().try_fold(0usize, |total, expert| {
            total.checked_add(entry_bytes(*expert as usize)?).ok_or(
                MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "expert payload byte span overflowed",
                },
            )
        })?,
        None => (0..source.entries().len()).try_fold(0usize, |total, expert_index| {
            total.checked_add(entry_bytes(expert_index)?).ok_or(
                MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "expert payload byte span overflowed",
                },
            )
        })?,
    };
    if source.selected_expert_ids().is_none()
        && source.packed_arena().is_none()
        && staged_bytes >= original_bytes
    {
        return Err(MetalError::ExpertSourceUnsupported {
            node,
            reason: "transient expert staging must reduce checkpoint-resident bytes",
        });
    }
    Ok(())
}

/// Executes a plan with a borrowed per-step expert table.
///
/// Ordinary plans retain [`execute_plan`]'s exact path. A supplied table
/// stages its packed bytes and descriptor table for the one command buffer,
/// then selected gathered reductions receive the codec-tagged kernel ABI.
/// This is an execution bridge, not HOBBIT's final residency arena: the
/// staging vectors preserve correctness for discontiguous entries while the
/// residency owner grows a persistent contiguous arena.
pub fn execute_plan_with_expert_sources(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    expert_sources: &BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
) -> Result<Evaluated, MetalError> {
    if expert_sources.is_empty() {
        return execute_plan(plan, blocks);
    }
    #[cfg(feature = "instrument")]
    let expert_stage_started = std::time::Instant::now();
    let buffers = stage_expert_sources(plan, blocks, expert_sources)?;
    #[cfg(feature = "instrument")]
    if std::env::var_os("PROXIMA_DEBUG_METAL_STAGES").is_some() {
        eprintln!(
            "expert_source_stage_ms={} source_nodes={} staged_nodes={}",
            expert_stage_started.elapsed().as_secs_f64() * 1e3,
            expert_sources.len(),
            buffers.len(),
        );
    }
    execute_plan_inner(plan, blocks, &buffers)
}

fn stage_expert_sources(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    expert_sources: &BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
) -> Result<BTreeMap<NodeId, ExpertSourceBuffers>, MetalError> {
    for (node, source) in expert_sources {
        let block_position = plan
            .prepared
            .block_nodes
            .iter()
            .position(|candidate| candidate == node)
            .ok_or(MetalError::ExpertSourceUnsupported {
                node: *node,
                reason: "node is not a bound block input",
            })?;
        reject_non_reducing_expert_staging(*node, blocks[block_position], source)?;
    }
    let (device, _queue) = device_and_queue()?;
    let mut buffers = BTreeMap::new();
    for (node, source) in expert_sources {
        if !plan.prepared.block_nodes.contains(node) {
            return Err(MetalError::ExpertSourceUnsupported {
                node: *node,
                reason: "node is not a bound block input",
            });
        }
        let entries = source.entries();
        entries.first().ok_or(MetalError::ExpertSourceUnsupported {
            node: *node,
            reason: "table is empty",
        })?;
        let signature = expert_source_signature(source);
        EXPERT_SOURCE_CACHE.with(|cache| -> Result<(), MetalError> {
            let mut cache = cache.try_borrow_mut().map_err(|_| {
                MetalError::ExpertSourceUnsupported {
                    node: *node,
                    reason: "expert source cache is already borrowed",
                }
            })?;
        let needs_stage = cache
            .get(node)
            .is_none_or(|(cached_signature, _)| *cached_signature != signature);
        if needs_stage {
            #[cfg(feature = "instrument")]
            counter!(EXPERT_SOURCE_CACHE_MISSES, 1);
            let previous = cache.remove(node).map(|(_, staged)| staged);
            let staged = stage_expert_source_reusing(&device, *node, source, previous)?;
            cache.insert(*node, (signature, staged));
            if std::env::var_os("PROXIMA_DEBUG_EXPERT_SOURCE_CACHE").is_some() {
                eprintln!(
                    "metal expert source cache miss node={node:?} signature={signature} payload_bytes={}",
                    cache
                        .get(node)
                        .map_or(0, |(_, staged)| staged.buffers.payloads.length())
                );
            }
        } else {
            #[cfg(feature = "instrument")]
            counter!(EXPERT_SOURCE_CACHE_HITS, 1);
            if std::env::var_os("PROXIMA_DEBUG_EXPERT_SOURCE_CACHE").is_some() {
                eprintln!(
                    "metal expert source cache hit node={node:?} signature={signature} payload_bytes={}",
                    cache
                        .get(node)
                        .map_or(0, |(_, staged)| staged.buffers.payloads.length())
                );
            }
        }
        let Some((_, staged)) = cache.get(node) else {
            return Err(MetalError::ExpertSourceUnsupported {
                node: *node,
                reason: "expert source cache insertion did not produce a table",
            });
        };
        buffers.insert(
            *node,
            ExpertSourceBuffers {
                node: *node,
                payloads: staged.buffers.payloads.clone(),
                payload_offset: staged.buffers.payload_offset,
                descriptors: staged.buffers.descriptors.clone(),
                descriptor_records: staged.buffers.descriptor_records.clone(),
            },
        );
            Ok(())
        })?;
    }
    Ok(buffers)
}

/// Named-block counterpart used by `omega::backend`.
pub fn execute_plan_named_with_expert_sources(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    expert_sources: &BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
) -> Result<Evaluated, MetalError> {
    let debug_timing = std::env::var_os("PROXIMA_DEBUG_SEGMENT_HOST").is_some();
    let resolve_started = std::time::Instant::now();
    // Keep the original checkpoint blocks for the reduction guard; the
    // execution path substitutes expert payloads only after this comparison.
    let blocks = proxima_tensor::cpu::resolve_named_blocks(&plan.program, named)?;
    let resolve_elapsed_us = resolve_started.elapsed().as_micros();
    let execute_started = std::time::Instant::now();
    let result = execute_plan_with_expert_sources(plan, &blocks, expert_sources);
    if debug_timing {
        eprintln!(
            "metal expert execute resolve_us={} execute_us={}",
            resolve_elapsed_us,
            execute_started.elapsed().as_micros(),
        );
    }
    result
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

/// [`DispatchType::Concurrent`]'s dataflow-hazard set, generic over the
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
/// that pointer yet. Only instantiated on [`DispatchType::Concurrent`]'s
/// path -- [`DispatchType::Serial`] never allocates one, since a serial
/// encoder already orders every dispatch for it.
#[derive(Debug)]
struct HazardTracker<Id: Eq + core::hash::Hash + Copy> {
    written: std::collections::HashSet<Id>,
    read: std::collections::HashSet<Id>,
}

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

/// [`execute_plan_with_placements`]'s per-call hazard bookkeeping, owned by
/// the [`Plan`] and reused call-to-call instead of rebuilt: [`HazardTracker`]'s
/// two `HashSet`s and [`resolve_hazard_inputs`]'s pointer list all allocate
/// on their first insert of a call. A fresh [`HazardTracker::new`] and a
/// fresh `Vec` every call paid that allocation every call, forever, even
/// though every call needs the exact same starting (empty) state -- ROW 303's
/// residual. [`HazardState::reset`] clears both without releasing their
/// capacity, so only the very first call after a plan's own construction
/// ever allocates.
struct HazardState {
    tracker: HazardTracker<*const ProtocolObject<dyn MTLBuffer>>,
    inputs: Vec<*const ProtocolObject<dyn MTLBuffer>>,
}

impl HazardState {
    fn new() -> Self {
        Self {
            tracker: HazardTracker::new(),
            inputs: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.tracker.reset();
        self.inputs.clear();
    }
}

/// The one adapter from a dispatch's read `NodeId`s (see
/// `crate::msl::hazard_read_nodes`, the caller's own source for `nodes` below)
/// to the hazard tracker's buffer-pointer identities, resolved from
/// `device_buffers` — erroring on the first node with none instead of
/// silently dropping it from the hazard set (the previous `filter_map`
/// behavior) — a node missing from `device_buffers` means the encode this
/// identity feeds is already wrong, so hiding it from the hazard check only
/// hides a real bug behind a missing barrier.
/// Test-only surface: production hazard resolution goes through
/// [`resolve_hazard_inputs_into`] against [`Plan::hazard_state`]'s reused
/// scratch list. Kept as an owned-`Vec` wrapper here since a unit test wants
/// a value to assert on, not a buffer to manage.
#[cfg(test)]
fn resolve_hazard_inputs(
    nodes: impl Iterator<Item = NodeId>,
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
) -> Result<Vec<*const ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let mut resolved = Vec::new();
    resolve_hazard_inputs_into(nodes, device_buffers, &mut resolved)?;
    Ok(resolved)
}

/// [`resolve_hazard_inputs`], writing into a caller-owned, reused buffer
/// instead of collecting a fresh `Vec` -- [`execute_plan_with_placements`]
/// calls this against [`Plan::hazard_state`]'s own scratch list every
/// position so a warm plan-hit step's hazard resolution allocates nothing
/// (ROW 303's residual).
fn resolve_hazard_inputs_into(
    nodes: impl Iterator<Item = NodeId>,
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    resolved: &mut Vec<*const ProtocolObject<dyn MTLBuffer>>,
) -> Result<(), MetalError> {
    resolved.clear();
    for operand in nodes {
        let pointer = device_buffers
            .get(&operand)
            .map(|(buffer, _offset)| Retained::as_ptr(buffer))
            .ok_or(MetalError::UnresolvedHazardOperand { node: operand })?;
        resolved.push(pointer);
    }
    Ok(())
}

/// The exact per-op hazard step [`execute_plan_with_placements`]'s loop
/// runs: check, barrier-and-reset only on a hazard, then always record —
/// factored out so the tests drive THIS function instead of a hand-written
/// mirror of the loop body that could silently drift from it. `output` is
/// the identity of the buffer this op is about to write, already resolved
/// (placement -> arena slot -> fresh allocation) by the caller before this
/// runs, since every bound op writes exactly one device buffer. Returns
/// whether the caller must emit a barrier before encoding this op.
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
/// On [`DispatchType::Concurrent`], the paragraph above no longer
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
/// `PROXIMA_METAL_KIND_FILTER=<term>[,<term>]` (or `!<term>[,<term>]` for
/// the complement) -- `instrument`-gated, default-off, parsed once per
/// [`execute_plan_with_placements`] call by [`KindFilter::from_env`]. Each
/// `<term>` is either a bare substring, matched against [`classify_kind`]'s
/// own return value (the same string a caller already sees in
/// `op_profile_bucket kind=...`), or `family:<substring>`, matched against
/// [`weight_family`]'s own return value for the op's first named operand
/// (the same aggregation `proxima-model-interop`'s `report_op_timings`
/// already reuses via that function, rather than each call site keeping its
/// own copy of the layer-index-stripping rule). Unset (`None`) in every
/// production run, which is the ROW's in-buffer ablation harness's own
/// arm-selection knob -- see that row for the arm table.
///
/// `classify_kind`'s live return values, as of the `BoundOpKind::name()`
/// delegation (`refactor(omega): classify_kind names the kind through the
/// type`): `cached_attention`, `elementwise`, `iota`, `constant`,
/// `keep::scan fold` (the four `BoundOpKind::name()`-delegated arms plus
/// `Reduce { keep: Keep::Scan, .. }`), and `reduce-tiled-gemm` /
/// `reduce-packed-row-blocked` / `reduce-cooperative` /
/// `reduce-generic-scalar` / `reduce-unclassified` for `Reduce { keep:
/// Keep::Reduce, .. }`. A bare `kind:`-shaped term that matches none of
/// those is rejected eagerly by [`KindFilter::from_env`] --
/// [`MetalError::UnknownKindFilterTerm`]. A `family:` term cannot be
/// validated the same way (family names are data-dependent on the loaded
/// checkpoint, not a fixed enum), so instead [`validate_kind_filter`]
/// checks, against THIS plan's own dispatch sequence, that the filter
/// removes at least one op and not every op --
/// [`MetalError::KindFilterMatchesNothing`] otherwise. Before these two
/// checks landed (ROW 308), a stale or misspelled term silently degenerated
/// to "every op skipped" or "no op skipped" instead of failing loudly.
#[cfg(feature = "instrument")]
enum FilterTerm {
    Kind(String),
    Family(String),
}

#[cfg(feature = "instrument")]
impl FilterTerm {
    fn parse(raw: &str) -> Result<Self, MetalError> {
        match raw.strip_prefix("family:") {
            Some(family) => Ok(Self::Family(family.to_string())),
            None => {
                if KNOWN_KIND_SUBSTRINGS
                    .iter()
                    .any(|known| known.contains(raw))
                {
                    Ok(Self::Kind(raw.to_string()))
                } else {
                    Err(MetalError::UnknownKindFilterTerm {
                        term: raw.to_string(),
                    })
                }
            }
        }
    }

    fn matches(&self, kind: &str, family: Option<&str>) -> bool {
        match self {
            Self::Kind(substring) => kind.contains(substring.as_str()),
            Self::Family(substring) => family.is_some_and(|name| name.contains(substring.as_str())),
        }
    }
}

/// [`classify_kind`]'s own live return-value vocabulary, restated here only
/// for [`FilterTerm::parse`]'s eager validation -- see [`KindFilter`]'s own
/// doc for why this list must be re-read from `classify_kind`, not
/// memorized, whenever that function gains or renames an arm.
#[cfg(feature = "instrument")]
const KNOWN_KIND_SUBSTRINGS: &[&str] = &[
    "cached_attention",
    "elementwise",
    "iota",
    "constant",
    "keep::scan fold",
    "reduce-tiled-gemm",
    "reduce-packed-row-blocked",
    "reduce-cooperative",
    "reduce-generic-scalar",
    "reduce-unclassified",
];

#[cfg(feature = "instrument")]
struct KindFilter {
    raw: String,
    terms: Vec<FilterTerm>,
    negate: bool,
}

#[cfg(feature = "instrument")]
impl KindFilter {
    fn from_env() -> Result<Option<Self>, MetalError> {
        let Ok(raw) = std::env::var("PROXIMA_METAL_KIND_FILTER") else {
            return Ok(None);
        };
        let (negate, body) = match raw.strip_prefix('!') {
            Some(rest) => (true, rest),
            None => (false, raw.as_str()),
        };
        let terms = body
            .split(',')
            .map(str::trim)
            .filter(|term| !term.is_empty())
            .map(FilterTerm::parse)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Self { raw, terms, negate }))
    }

    fn matches(&self, kind: &str, family: Option<&str>) -> bool {
        let any = self.terms.iter().any(|term| term.matches(kind, family));
        any != self.negate
    }
}

/// `blk.7.ffn_down.weight` -> `ffn_down.weight`: drops exactly one
/// `.`-delimited numeric segment (the layer index every `blk.N.*` weight
/// name carries) so a caller can sum one matmul KIND across all layers
/// instead of reporting one line per layer. The one place this transform is
/// written -- `proxima-model-interop`'s `report_op_timings` calls this
/// function rather than keeping its own copy, and `KindFilter`'s `family:`
/// term reuses it too, so a layer-count change or a naming convention
/// change only needs to land here.
#[cfg(feature = "instrument")]
#[must_use]
pub fn weight_family(name: &str) -> String {
    name.split('.')
        .filter(|segment| segment.parse::<u32>().is_err())
        .collect::<Vec<&str>>()
        .join(".")
}

/// [`weight_family`] applied to `bound`'s own first named operand -- the
/// same `find_map` [`execute_plan_op_timed`] already runs to populate
/// [`OpGpuTiming::weight_name`], reused here rather than restated so
/// [`KindFilter`]'s `family:` term and the op-timed profiler's family
/// aggregation can never read two different operands as "the" weight.
#[cfg(feature = "instrument")]
fn bound_weight_family(bound: &BoundOp, program: &[Op]) -> Option<String> {
    bound
        .operands()
        .iter()
        .find_map(|(source, _, _)| program[source.0 as usize].name())
        .map(weight_family)
}

/// Applies `filter` to every op in `prepared.resolved` the same way the
/// main dispatch loop below will, and rejects a filter that would remove
/// zero ops or every op -- either shape means the filter's own terms never
/// isolated anything in THIS plan, the silent-degenerate failure ROW 308
/// found (see [`KindFilter`]'s own doc).
#[cfg(feature = "instrument")]
fn validate_kind_filter(
    filter: &KindFilter,
    prepared: &Prepared,
    packed_operands: &PackedOperands,
    program: &[Op],
) -> Result<(), MetalError> {
    let total = prepared.resolved.len();
    let removed = prepared
        .resolved
        .iter()
        .filter(|bound| {
            !filter.matches(
                classify_kind(bound, packed_operands),
                bound_weight_family(bound, program).as_deref(),
            )
        })
        .count();
    if removed == 0 || removed == total {
        return Err(MetalError::KindFilterMatchesNothing {
            filter: filter.raw.clone(),
        });
    }
    Ok(())
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

/// `recycle` is the same pool [`proxima_tensor::cpu::evaluate_with_scratch`]
/// takes: a caller done reading a PREVIOUS call's `Evaluated` hands its
/// storage back with [`Evaluated::into_scratch`], and this call pops one
/// buffer from the pool (if any) to reuse for the root output's read-back
/// instead of allocating fresh. An empty pool (every call before this
/// landing was, implicitly) always allocates fresh, exactly as before -- see
/// `finish`'s own doc for the exact conditions a popped buffer is actually
/// reused under.
///
/// # Errors
/// Propagates block-codec and Metal driver failures, same as [`execute_plan`].
#[cfg(feature = "metal-output-placement")]
pub fn execute_plan_with_placements(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
    recycle: &mut Vec<Vec<f32>>,
) -> Result<Evaluated, MetalError> {
    execute_plan_with_placements_inner(
        plan,
        blocks,
        input_placements,
        output_placements,
        recycle,
        &BTreeMap::new(),
    )
}

#[cfg(feature = "metal-output-placement")]
fn execute_plan_with_placements_inner(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
    recycle: &mut Vec<Vec<f32>>,
    expert_buffers: &BTreeMap<NodeId, ExpertSourceBuffers>,
) -> Result<Evaluated, MetalError> {
    let prepared = &plan.prepared;
    let packed_operands = &plan.packed_operands;
    let mut effective_expert_buffers = expert_buffers.clone();
    for (source_node, buffers) in expert_buffers {
        let Some(source_name) = plan.program[source_node.0 as usize].name() else {
            continue;
        };
        for (position, operation) in plan.program.iter().enumerate() {
            if operation.name() == Some(source_name) {
                effective_expert_buffers
                    .entry(NodeId(position as u32))
                    .or_insert_with(|| buffers.clone());
            }
        }
    }
    let input_placed: BTreeMap<NodeId, (&PlacedBuffer, usize)> = input_placements
        .iter()
        .map(|(node, buffer, offset)| (*node, (*buffer, *offset)))
        .collect();
    if std::env::var_os("PROXIMA_DEBUG_PLACEMENT_KEYS").is_some() {
        eprintln!(
            "metal placement keys inputs={:?} blocks={:?}",
            input_placed.keys().collect::<Vec<_>>(),
            prepared.block_nodes,
        );
    }
    let output_placed: BTreeMap<NodeId, (&PlacedBuffer, usize)> = output_placements
        .iter()
        .map(|(node, buffer, offset)| (*node, (*buffer, *offset)))
        .collect();
    let (device, queue) = device_and_queue()?;
    // On a plan-cache HIT this is a no-op (`resolve_steps` checks staleness
    // and returns immediately): the per-position loop below indexes
    // `plan.resolved_steps` instead of every step re-deriving
    // `kernel_cache_key`/`kernel_dispatch_shape`/`pipeline_for`'s own cache
    // key. Only a genuine plan-cache MISS or a `set_math_mode` change pays
    // this once, up front, rather than once per op per step.
    resolve_steps(&device, plan)?;
    let resolved_steps = plan.resolved_steps.borrow();

    // plan-owned and never rebuilt fresh -- see `Plan::device_buffers`'s own
    // doc for why this alone (independent of the per-node reuse skip below)
    // already removes ROW 303's residual `BTreeMap::new()` allocation on a
    // warm call: every key this loop and the position loop below touch is
    // still written fresh every call exactly as before, only the map's own
    // heap nodes now outlive one call instead of being dropped with it.
    let mut device_buffers = plan.device_buffers.borrow_mut();
    for (node, buffers) in &effective_expert_buffers {
        device_buffers.insert(*node, (buffers.payloads.clone(), buffers.payload_offset));
    }
    let mut block_identity = plan.block_identity.borrow_mut();
    if block_identity.len() != prepared.block_nodes.len() {
        block_identity.resize(prepared.block_nodes.len(), None);
    }
    for (index, ((node, block), dtype)) in prepared
        .block_nodes
        .iter()
        .zip(blocks.iter())
        .zip(plan.block_dtypes.iter())
        .enumerate()
    {
        if effective_expert_buffers.contains_key(node) {
            block_identity[index] = None;
            continue;
        }
        // an input-placed node skips the host round trip entirely: its
        // buffer is already on the device, owned by the caller, and this
        // call's own `block` entry for it (still required, positionally, to
        // keep `block_nodes.iter().zip(blocks.iter())` aligned) is unused.
        // The caller's own offset travels in the same `DeviceBuffer` tuple
        // every other node's buffer carries -- `buffer_for`/`bind_buffers`
        // read it generically, with no separate offset map required.
        if let Some((buffer, offset)) = input_placed.get(node) {
            device_buffers.insert(*node, ((*buffer).clone(), *offset));
            block_identity[index] = None;
            continue;
        }
        let pre_gather_enabled = std::env::var_os("PROXIMA_QWEN35MOE_PRE_GATHER").is_some()
            || (std::env::var_os("PROXIMA_EXPERT_SIDECAR").is_some()
                && std::env::var("PROXIMA_QWEN35MOE_RESIDENCY_BUDGET_BYTES")
                    .ok()
                    .and_then(|value| value.parse::<u64>().ok())
                    .is_some_and(|value| value > 0));
        if pre_gather_enabled
            && plan.program[node.0 as usize]
                .name()
                .is_some_and(|name| name.contains("_exps.weight"))
        {
            // Router segments carry the full program input inventory, but
            // expert bytes are supplied only by the gather source table.
            // Do not bind the original packed stack on this placement path.
            block_identity[index] = None;
            continue;
        }
        let current_identity = block_identity_key(block);
        let resident = plan.resident_nodes.contains(node);
        if block_buffer_reusable(resident, block_identity[index], current_identity)
            && device_buffers.contains_key(node)
        {
            // this position's buffer is already correct in the plan-owned
            // map from a PRIOR call (never retired -- see the retirement
            // loop's own `resident_nodes` skip below) -- no upload, no
            // `device_buffers` write, this call's only cost for this node
            // is the identity comparison just made.
            continue;
        }
        block_identity[index] = Some(current_identity);
        let resident_name = resident_name(plan, *node);
        // a brand-new `Plan` (a KV-bucket-boundary reshape, say) starts with
        // an empty `device_buffers`/`block_identity` of its OWN, so the
        // fast-skip above always misses on plan 1 of a node's life even
        // though the GLOBAL, name-keyed caches
        // (`NOCOPY_BUFFERS`/`RESIDENT_BUFFERS`/the checkpoint mapping)
        // already hold this exact host range from the PRIOR plan. Consult
        // them here, before counting this node as "offered", so a reshape
        // does not re-walk every resident weight block through the upload
        // path just to have it resolve to the same cache hit it already was.
        if let Some((buffer, offset)) = cross_plan_resident_reuse(
            resident_name,
            current_identity.0 as *const c_void,
            current_identity.1,
        )? {
            device_buffers.insert(*node, (buffer, offset));
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
        let buffer = match block {
            QuantizedBlock::Float32(data) => upload_block(
                &device,
                data,
                *node,
                *dtype,
                plan.program[node.0 as usize].name(),
                resident_name,
            )?,
            QuantizedBlock::Int32(data) => {
                upload_block_int32_as_float(&device, data, resident_name)?
            }
            // `Q3_K` uploads its raw super-block bytes unchanged, same as
            // every other packed codec below -- `msl::PackedCodec::Q3K`'s
            // own unpack kernel (`q3k_element`) reads them at the GPU side.
            QuantizedBlock::Q3K(bytes)
            | QuantizedBlock::Q4K(bytes)
            | QuantizedBlock::Q5K(bytes)
            | QuantizedBlock::Q6K(bytes)
            | QuantizedBlock::Q8_0(bytes)
            | QuantizedBlock::Q4_0(bytes)
            | QuantizedBlock::Q5_1(bytes)
            | QuantizedBlock::Q2K(bytes)
            | QuantizedBlock::Iq4Nl(bytes)
            | QuantizedBlock::Iq2Xs(bytes)
            | QuantizedBlock::Iq3Xxs(bytes)
            | QuantizedBlock::Float16(bytes)
            | QuantizedBlock::BFloat16(bytes) => {
                upload_packed_bytes(&device, bytes, resident_name)?
            }
        };
        device_buffers.insert(*node, buffer);
    }

    #[cfg(feature = "instrument")]
    let kind_filter = KindFilter::from_env()?;
    #[cfg(feature = "instrument")]
    if let Some(filter) = &kind_filter {
        validate_kind_filter(filter, prepared, packed_operands, &plan.program)?;
    }

    let command_buffer = queue
        .commandBuffer()
        .ok_or_else(|| MetalError::CompileFailed {
            log: "command queue refused to hand out a command buffer".to_string(),
        })?;
    // `Concurrent` lets independent dispatches (e.g. Q/K/V from one normed
    // input) overlap instead of draining the pipeline between every op --
    // see [`DispatchType`]'s own doc for why that requires [`HazardTracker`]
    // below to insert the barriers `Serial` gives for free by never
    // overlapping any two dispatches in the first place.
    let dispatch_type = plan.dispatch_type;
    let encoder = EncoderGuard::new(
        command_buffer
            .computeCommandEncoderWithDispatchType(dispatch_type.as_mtl())
            .ok_or_else(|| MetalError::CompileFailed {
                log: "command buffer refused to hand out a compute encoder".to_string(),
            })?,
    );
    // plan-owned, reused across every call against this `Plan` (`HazardState`'s
    // own doc) -- `reset` clears both the tracker's sets and the input scratch
    // without releasing their capacity, so a warm call after the first never
    // pays their first-insert allocation again.
    let mut hazard_state = plan.hazard_state.borrow_mut();
    hazard_state.reset();
    // reborrowed as a plain `&mut` so `tracker`/`inputs` split into disjoint
    // field borrows below -- the borrow checker cannot see through
    // `RefMut`'s own `Deref`/`DerefMut` to know the two fields are disjoint.
    let hazard_state = &mut *hazard_state;

    // diagnostic-only, `instrument`-gated, always emitted at `trace` level
    // (default-off via the runtime filter, never a bespoke env var): each
    // output-placed node's WRITE position and each of its aliased readers'
    // own position, the direct evidence a write-before-read ordering claim
    // needs (this function's own doc, "Within-call aliasing"). Silent unless
    // a caller raises `RUST_LOG` to `trace` for this target.
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
            Some(filter) => !filter.matches(
                classify_kind(bound, packed_operands),
                bound_weight_family(bound, &plan.program).as_deref(),
            ),
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
            // `Serial` never overlaps two dispatches, so no hazard this
            // tracker catches could ever race -- skip it entirely rather
            // than pay the resolve/record bookkeeping for barriers that
            // would never fire.
            let resolved_step = resolved_steps
                .as_ref()
                .and_then(|resolved| resolved.steps.get(position));
            let resolved_output: Option<DeviceBuffer> = if dispatch_type == DispatchType::Concurrent
            {
                // The hazard read set is derived from `bindings` -- the exact
                // list `bind_buffers` will bind for this op -- rather than
                // enumerated separately from `bound.all_read_sources()`: ROW
                // 323 was exactly those two enumerations drifting apart when
                // `msl::bindings` grew a source (a fused epilogue operand)
                // this hazard check had not been taught to see. Reading
                // `bindings` itself makes that class of drift impossible --
                // there is only one list, and both the encoder bind loop
                // (`bind_buffers`) and this hazard walk read the same one.
                let owned_bindings: Vec<Binding>;
                let bindings_for_hazard: &[Binding] = match resolved_step {
                    Some(step) => step.bindings.as_slice(),
                    None => {
                        let (bindings, _grid) =
                            kernel_dispatch_shape(bound, packed_operands, plan.numeric_policy)?;
                        owned_bindings = bindings;
                        owned_bindings.as_slice()
                    }
                };
                resolve_hazard_inputs_into(
                    crate::msl::hazard_read_nodes(bindings_for_hazard),
                    &device_buffers,
                    &mut hazard_state.inputs,
                )?;
                // resolved ONCE, before the hazard check, from the exact same
                // placement -> arena -> fresh-allocation chain `encode_op`
                // would otherwise pick independently below: `needs_barrier`
                // and `record` used to see two different identities for a
                // fresh allocation (the check saw `None`, since nothing is
                // known before `encode_op` allocates; the record afterward
                // saw the real pointer), so a WAW/WAR hazard against an
                // address Metal happens to reuse for that allocation could
                // never be caught. Allocating here and handing the SAME
                // buffer into `encode_op` via `placement` closes that gap.
                let resolved: DeviceBuffer = match placement {
                    Some((buffer, offset)) => (buffer.clone(), offset),
                    None => (
                        allocate_buffer(&device, bound_output_len(bound), bound.dtype)?,
                        0,
                    ),
                };
                // the write side of the same "derived from bindings" guarantee:
                // whenever `bindings_for_hazard` names an explicit
                // `Binding::Output`, it must name THIS op's own node -- if it
                // ever didn't, the buffer the hazard tracker records as
                // written and the buffer `bind_buffers` actually binds as
                // this op's output would be two different things, which is a
                // worse bug than the one this refactor closes. Redesign §4c:
                // a `CachedAttention` split kernel under `ContextSplitMerge`
                // has NO `Binding::Output` at all (`Binding::Scratch`
                // replaces it -- that kernel writes scratch, never
                // `bound.node`'s own buffer) -- `hazard_write_node` returning
                // `None` there is the honest, by-design case, not a drift:
                // `resolved` below is still `bound.node`'s own real output
                // buffer (from `placement`, independent of `bindings`), and
                // the merge dispatch this position's `encode_op` call also
                // issues is what actually writes it, in the same encoder,
                // immediately after the split.
                debug_assert!(
                    matches!(
                        crate::msl::hazard_write_node(bindings_for_hazard),
                        Some(node) if node == bound.node
                    ) || crate::msl::hazard_write_node(bindings_for_hazard).is_none()
                );
                let hazard_output = Retained::as_ptr(&resolved.0);
                if hazard_step(
                    &mut hazard_state.tracker,
                    &hazard_state.inputs,
                    hazard_output,
                ) {
                    encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                    counter!(BARRIERS_EMITTED, 1);
                }
                Some(resolved)
            } else {
                None
            };
            let placement = match &resolved_output {
                Some((buffer, offset)) => Some((buffer, *offset)),
                None => placement,
            };
            let uniform_buffer = plan_uniform_buffer(plan, position)?;
            // Redesign §4c: the ONE call site that resolves a scratch
            // buffer for `encode_op`'s two-dispatch `CachedAttention` form
            // -- every other `encode_op` caller passes `None` and rejects a
            // `Binding::Scratch` kernel instead (that function's own doc).
            let attention_scratch =
                attention_scratch_buffer(plan, position)?.map(|buffer| (buffer, 0usize));
            // Only `Concurrent` needs the intra-op scratch write -> read edge
            // routed through the tracker (see `encode_op`'s own doc for
            // `hazard`) -- `Serial` orders the split before the merge for
            // free, the same reason `resolved_output` above is `None` there.
            let hazard =
                (dispatch_type == DispatchType::Concurrent).then_some(&mut hazard_state.tracker);
            let bound_expert_buffers = expert_buffers_for(bound, &effective_expert_buffers)?;
            let fault = encode_op(
                &device,
                &encoder,
                &mut device_buffers,
                bound,
                packed_operands,
                placement,
                uniform_buffer,
                plan.math_mode,
                plan.numeric_policy,
                resolved_step,
                attention_scratch,
                hazard,
                bound_expert_buffers,
            )?;
            if let Some((fault_buffer, gathers)) = fault {
                pending_faults.push((bound, fault_buffer, gathers));
            }
        }
        // explicit liveness exclusion (see this function's doc): a placed
        // node, input or output, is externally owned and always live, so it
        // is never dropped from this call's own bookkeeping map, regardless
        // of what `prepared.retires` (a per-program-only liveness sweep)
        // says. A RESIDENT block node joins that exclusion for the same
        // reason: [`Plan::mark_resident`]'s caller-owned promise means its
        // buffer is live for the plan's whole life, and leaving its entry in
        // [`Plan::device_buffers`] across calls is exactly what lets
        // `block_buffer_reusable` skip re-uploading it next call.
        for retired in &prepared.retires[position] {
            if input_placed.contains_key(retired)
                || output_placed.contains_key(retired)
                || plan.resident_nodes.contains(retired)
            {
                continue;
            }
            // a retired buffer's identity must not outlive it in the
            // tracker: Metal (and `metal-buffer-pool` more aggressively) can
            // hand this exact address back out to a later, unrelated
            // `allocate_buffer` call, and that later buffer must start with
            // no hazard history -- see `HazardTracker::forget`'s own doc.
            if dispatch_type == DispatchType::Concurrent {
                if let Some((buffer, _offset)) = device_buffers.remove(retired) {
                    hazard_state.tracker.forget(Retained::as_ptr(&buffer));
                }
            } else {
                device_buffers.remove(retired);
            }
        }
    }
    encoder.finish();

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
    finish(plan, &device_buffers, &placed_output_nodes, recycle.pop())
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
    numeric_policy: NumericPolicy,
) -> Result<Plan, MetalError> {
    let blocks = resolve_named_blocks(program, named)?;
    plan(program, symbols, &blocks, outputs, numeric_policy)
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
    let block_nodes = block_node_ids(&plan.program);
    let placed_inputs: BTreeSet<NodeId> =
        input_placements.iter().map(|(node, _, _)| *node).collect();
    let mut placeholders = Vec::new();
    let mut placeholder_nodes = BTreeMap::new();
    for node in &block_nodes {
        if named.iter().any(|(candidate, _)| {
            *candidate == plan.program[node.0 as usize].name().unwrap_or_default()
        }) {
            continue;
        }
        if placed_inputs.contains(node) {
            let placeholder_index = placeholders.len();
            placeholders.push(vec![0.0_f32; element_count(plan.prepared.shapes.of(*node))]);
            placeholder_nodes.insert(*node, placeholder_index);
        }
    }
    let mut blocks = Vec::with_capacity(block_nodes.len());
    for node in &block_nodes {
        let name = plan.program[node.0 as usize]
            .name()
            .ok_or(TensorError::UnnamedInput(*node))?;
        if let Some(block) = named.iter().find(|(candidate, _)| *candidate == name) {
            blocks.push(block.1);
        } else if let Some(index) = placeholder_nodes.get(node) {
            blocks.push(QuantizedBlock::Float32(placeholders[*index].as_slice()));
        } else {
            return Err(TensorError::UnboundInputName(String::from(name)).into());
        }
    }
    // no scratch pool for this name-keyed convenience wrapper -- a caller
    // wanting the recycle path calls `execute_plan_with_placements` directly.
    execute_plan_with_placements(
        plan,
        &blocks,
        input_placements,
        output_placements,
        &mut Vec::new(),
    )
}

/// Executes a named plan with both caller-owned buffers and per-step expert
/// substitutions. Routed recurrent models need both capabilities in the same
/// command buffer: placement keeps recurrent state on the device while the
/// expert table selects the codec and address for the current route.
///
/// # Errors
/// Propagates name resolution, expert-source validation, and Metal failures.
#[cfg(feature = "metal-output-placement")]
pub fn execute_plan_named_with_placements_and_expert_sources(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
    expert_sources: &BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
) -> Result<Evaluated, MetalError> {
    let block_nodes = block_node_ids(&plan.program);
    let placed_inputs: BTreeSet<NodeId> =
        input_placements.iter().map(|(node, _, _)| *node).collect();
    let mut placeholders = Vec::new();
    let mut placeholder_nodes = BTreeMap::new();
    for node in &block_nodes {
        let name = plan.program[node.0 as usize]
            .name()
            .ok_or(TensorError::UnnamedInput(*node))?;
        if named.iter().any(|(candidate, _)| *candidate == name) {
            continue;
        }
        if placed_inputs.contains(node) {
            let placeholder_index = placeholders.len();
            placeholders.push(vec![0.0_f32; element_count(plan.prepared.shapes.of(*node))]);
            placeholder_nodes.insert(*node, placeholder_index);
        }
    }
    let mut blocks = Vec::with_capacity(block_nodes.len());
    for node in &block_nodes {
        let name = plan.program[node.0 as usize]
            .name()
            .ok_or(TensorError::UnnamedInput(*node))?;
        if let Some(block) = named.iter().find(|(candidate, _)| *candidate == name) {
            blocks.push(block.1);
        } else if let Some(index) = placeholder_nodes.get(node) {
            blocks.push(QuantizedBlock::Float32(placeholders[*index].as_slice()));
        } else {
            return Err(TensorError::UnboundInputName(String::from(name)).into());
        }
    }
    #[cfg(feature = "instrument")]
    let expert_stage_started = std::time::Instant::now();
    let expert_buffers = stage_expert_sources(plan, &blocks, expert_sources)?;
    #[cfg(feature = "instrument")]
    if std::env::var_os("PROXIMA_DEBUG_METAL_STAGES").is_some() {
        eprintln!(
            "expert_source_stage_ms={} source_nodes={} staged_nodes={}",
            expert_stage_started.elapsed().as_secs_f64() * 1e3,
            expert_sources.len(),
            expert_buffers.len(),
        );
    }
    execute_plan_with_placements_inner(
        plan,
        &blocks,
        input_placements,
        output_placements,
        &mut Vec::new(),
        &expert_buffers,
    )
}

/// [`execute_plan`] with the WHOLE program's own single command buffer's
/// `GPUStartTime`/`GPUEndTime` read back once, instead of
/// [`execute_plan_op_timed`]'s one-command-buffer-per-op attribution. ROW
/// 375 measured a batch of independent same-kind dispatches with the
/// per-op instrument and got a flat ~705 us reading dominated by that
/// function's own per-buffer submit/wait floor, 13-20x the in-program
/// per-dispatch cost `rmsnorm_fused_epilogue_cost.rs`'s ROW 368/372 batched
/// harness measures through this same one-encoder, one-command-buffer
/// shape `execute_plan` already uses for production. This function exists
/// so a caller batching N independent dispatches into one plan (this
/// crate's own `plan_named`/`execute_plan_named`) can read that batch's
/// real GPU occupancy without re-deriving `execute_plan`'s encode loop.
///
/// # Errors
/// Same as [`execute_plan`].
#[cfg(feature = "instrument")]
pub fn execute_plan_timed(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
) -> Result<(Evaluated, u64), MetalError> {
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
        let resident_name = resident_name(plan, *node);
        let buffer = match block {
            QuantizedBlock::Float32(data) => upload_block(
                &device,
                data,
                *node,
                *dtype,
                plan.program[node.0 as usize].name(),
                resident_name,
            )?,
            QuantizedBlock::Int32(data) => {
                upload_block_int32_as_float(&device, data, resident_name)?
            }
            QuantizedBlock::Q3K(bytes)
            | QuantizedBlock::Q4K(bytes)
            | QuantizedBlock::Q5K(bytes)
            | QuantizedBlock::Q6K(bytes)
            | QuantizedBlock::Q8_0(bytes)
            | QuantizedBlock::Q4_0(bytes)
            | QuantizedBlock::Q5_1(bytes)
            | QuantizedBlock::Q2K(bytes)
            | QuantizedBlock::Iq4Nl(bytes)
            | QuantizedBlock::Iq2Xs(bytes)
            | QuantizedBlock::Iq3Xxs(bytes)
            | QuantizedBlock::Float16(bytes)
            | QuantizedBlock::BFloat16(bytes) => {
                upload_packed_bytes(&device, bytes, resident_name)?
            }
        };
        device_buffers.insert(*node, buffer);
    }

    let command_buffer = queue
        .commandBuffer()
        .ok_or_else(|| MetalError::CompileFailed {
            log: "command queue refused to hand out a command buffer".to_string(),
        })?;
    let encoder = EncoderGuard::new(command_buffer.computeCommandEncoder().ok_or_else(|| {
        MetalError::CompileFailed {
            log: "command buffer refused to hand out a compute encoder".to_string(),
        }
    })?);

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
            plan.math_mode,
            plan.numeric_policy,
            None,
            None,
            None,
            None,
        )?;
        if let Some((fault_buffer, gathers)) = fault {
            pending_faults.push((bound, fault_buffer, gathers));
        }
        for retired in &prepared.retires[position] {
            device_buffers.remove(retired);
        }
    }
    encoder.finish();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let gpu_ns =
        ((command_buffer.GPUEndTime() - command_buffer.GPUStartTime()) * 1e9).max(0.0) as u64;

    for (bound, fault_buffer, gathers) in &pending_faults {
        check_gather_fault(bound, fault_buffer, *gathers)?;
    }

    let evaluated = finish(plan, &device_buffers, &BTreeSet::new(), None)?;
    Ok((evaluated, gpu_ns))
}

/// [`execute_plan_timed`] against a name-keyed block set, mirroring
/// [`execute_plan_named`]'s own name resolution.
///
/// # Errors
/// Propagates name-resolution and Metal driver failures.
#[cfg(feature = "instrument")]
pub fn execute_plan_named_timed(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
) -> Result<(Evaluated, u64), MetalError> {
    let blocks = resolve_named_blocks(&plan.program, named)?;
    execute_plan_timed(plan, &blocks)
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
    /// Full iteration extents of the bound operation that was timed.
    pub extents: Vec<u64>,
    /// Surviving axes for a reduction; empty for non-reductions.
    pub output_axes: Vec<u16>,
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

/// `PROXIMA_METAL_NAN_CHECK`-gated NaN probe for exactly ONE op's own output
/// buffer, called from [`execute_op_timed`] right after `waitUntilCompleted`
/// and before that op's operands are retired -- the diagnostic this crate's
/// own `PROXIMA_METAL_OP_PROFILE_STEP` path lacked to bisect the 30B
/// qwen3moe Metal decode's all-zero logits down to the exact op that first
/// produced NaN data, rather than the whole step.
///
/// Checks `is_nan()`, deliberately NOT `!is_finite()` (which would also
/// catch `-inf`): `causal_mask` (`proxima-tensor/src/spec.rs`) bakes a
/// literal `f32::NEG_INFINITY` into the program as a `constant` op, fed
/// through a `Select` into every masked attention score BEFORE softmax --
/// the SAME literal a correct CPU decode also evaluates and zeroes out, so
/// a `-inf` reaching a masked score, or even an unmasked reduce that legally
/// saturates, is not evidence of anything. This codebase never constructs an
/// intentional `f32::NAN` scalar (grepped: zero production call sites), so a
/// NaN appearing on ANY node's output has no legitimate source and is
/// unambiguous evidence of the defect this diagnostic exists to find.
///
/// Reads the buffer back through [`read_back`] (the same narrow-to-f32 path
/// [`finish`] uses for every other output), which is why this is reachable
/// only behind `instrument`: it pays a device round trip per op, same cost
/// class [`execute_op_timed`]'s own doc already accepts for GPU-time
/// attribution. Returns `true` the first time this op's own output holds a
/// NaN, letting the caller stop the step at the FIRST offending op instead
/// of running every remaining op past a value already known bad.
#[cfg(feature = "instrument")]
fn check_op_output_finite(
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    prepared: &Prepared,
    program: &[Op],
    node: NodeId,
    kind: &str,
) -> Result<Option<(usize, Vec<u64>, alloc::vec::Vec<f32>)>, MetalError> {
    let Some((buffer, offset)) = device_buffers.get(&node) else {
        return Ok(None);
    };
    let shape = prepared.shapes.of(node).to_vec();
    let dtype = gpu_dtype(program, &prepared.index_nodes, node);
    let data = read_back(buffer, *offset, element_count(&shape), node, dtype)?;
    let nan_count = data.iter().filter(|value| value.is_nan()).count();
    if nan_count == 0 {
        return Ok(None);
    }
    let Some(first_index) = data.iter().position(|value| value.is_nan()) else {
        return Ok(None);
    };
    let first_values: alloc::vec::Vec<f32> = data.iter().copied().take(8).collect();
    let kind_owned = kind.to_string();
    debug!(
        node = node.0,
        kind = %kind_owned,
        shape = ?shape,
        first_values = ?first_values,
        nan_count = nan_count as u64,
        total_count = data.len() as u64,
        "nan_check: first nan op output found"
    );
    Ok(Some((first_index, shape, data)))
}

/// `PROXIMA_METAL_COMPARE_CPU`-gated per-node parity probe, called from
/// [`execute_op_timed`] alongside [`check_op_output_finite`] -- reads this
/// op's own Metal output back through [`read_back`] and diffs it against
/// `cpu_reference`'s entry for the SAME [`NodeId`], computed once up front by
/// evaluating the identical program/symbols/blocks on the CPU route for
/// every non-`Op::Input` node (`proxima-model-interop/src/generate.rs`'s
/// `evaluate_op_timed`, the one call site that builds `cpu_reference` and
/// has `program`/`symbols`/`named` all in scope to do it).
///
/// Relative diff per element uses a `1e-6` floor on the denominator so a
/// near-zero CPU reference value cannot inflate a genuinely tiny absolute
/// gap into a huge ratio. Returns the max relative diff across every element
/// (`None` when this node has no CPU reference entry -- weights and other
/// `Op::Input` nodes are never in `cpu_reference` by construction). The
/// caller stops the step at the first node whose max relative diff exceeds
/// `1e-2`, printing every field this diagnostic's own report needs: node,
/// kind, shape, and the first 8 values on both sides.
#[cfg(feature = "instrument")]
fn compare_op_output_to_cpu(
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    prepared: &Prepared,
    program: &[Op],
    node: NodeId,
    kind: &str,
    cpu_reference: &BTreeMap<NodeId, alloc::vec::Vec<f32>>,
) -> Result<Option<f32>, MetalError> {
    let Some(cpu_values) = cpu_reference.get(&node) else {
        return Ok(None);
    };
    let Some((buffer, offset)) = device_buffers.get(&node) else {
        return Ok(None);
    };
    let shape = prepared.shapes.of(node).to_vec();
    let dtype = gpu_dtype(program, &prepared.index_nodes, node);
    let metal_values = read_back(buffer, *offset, element_count(&shape), node, dtype)?;
    let max_rel_diff = metal_values
        .iter()
        .zip(cpu_values.iter())
        .map(|(metal_value, cpu_value)| (metal_value - cpu_value).abs() / cpu_value.abs().max(1e-6))
        .fold(0.0_f32, f32::max);
    if max_rel_diff > 1e-2 {
        let metal_first: alloc::vec::Vec<f32> = metal_values.iter().copied().take(8).collect();
        let cpu_first: alloc::vec::Vec<f32> = cpu_values.iter().copied().take(8).collect();
        let kind_owned = kind.to_string();
        debug!(
            node = node.0,
            kind = %kind_owned,
            shape = ?shape,
            metal_first = ?metal_first,
            cpu_first = ?cpu_first,
            max_rel_diff = max_rel_diff as f64,
            "cpu_compare: divergent op output found"
        );
    }
    Ok(Some(max_rel_diff))
}

#[cfg(feature = "instrument")]
const CPU_BOUND_COMPARE_CAP_BYTES: usize = 64 * 1024 * 1024;

#[cfg(feature = "instrument")]
fn validate_selected_expert_routes(
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    prepared: &Prepared,
    program: &[Op],
    bound: &BoundOp,
    expert_buffers: Option<&ExpertSourceBuffers>,
) -> Result<(), MetalError> {
    let Some(selected) = std::env::var("PROXIMA_METAL_COMPARE_BOUND_NODE")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .map(NodeId)
    else {
        return Ok(());
    };
    if selected != bound.node {
        return Ok(());
    }
    let Some(expert_buffers) = expert_buffers else {
        eprintln!("metal_expert_routes_missing_buffers node={:?}", bound.node);
        return Ok(());
    };
    eprintln!(
        "metal_expert_routes_begin node={:?} source_node={:?}",
        bound.node, expert_buffers.node
    );
    for (source, _, lookup) in bound.all_read_sources() {
        if *source != expert_buffers.node {
            continue;
        }
        let Some(lookup) = lookup else {
            continue;
        };
        let Some((lookup_buffer, lookup_offset)) = device_buffers.get(&lookup.indices) else {
            return Err(MetalError::UnresolvedHazardOperand {
                node: lookup.indices,
            });
        };
        let lookup_values = read_back(
            lookup_buffer,
            *lookup_offset,
            element_count(prepared.shapes.of(lookup.indices)),
            lookup.indices,
            gpu_dtype(program, &prepared.index_nodes, lookup.indices),
        )?;
        eprintln!(
            "metal_expert_lookup node={:?} values={:?}",
            lookup.indices, lookup_values
        );
        let mut seen = Vec::new();
        for value in lookup_values {
            let expert = value as u32;
            if value < 0.0 || value.fract() != 0.0 {
                return Err(MetalError::ExpertSourceUnsupported {
                    node: bound.node,
                    reason: "route index was not a non-negative integer",
                });
            }
            let Some(descriptor) = expert_buffers.descriptor_records.get(expert as usize) else {
                return Err(MetalError::ExpertSourceUnsupported {
                    node: bound.node,
                    reason: "route index exceeded the descriptor table",
                });
            };
            if descriptor.expert_index != expert
                || descriptor.byte_length == 0
                || descriptor.out_dim == 0
                || descriptor.in_dim == 0
            {
                eprintln!(
                    "metal_expert_route_invalid node={:?} expert={} descriptor={descriptor:?}",
                    bound.node, expert
                );
                return Err(MetalError::ExpertSourceUnsupported {
                    node: bound.node,
                    reason: "route selected an invalid expert descriptor",
                });
            }
            let descriptor_summary = (
                expert,
                descriptor.codec,
                descriptor.byte_offset,
                descriptor.byte_length,
            );
            if !seen.contains(&descriptor_summary) {
                seen.push(descriptor_summary);
            }
        }
        eprintln!(
            "metal_expert_routes node={:?} lookup={:?} routes={seen:?}",
            bound.node, lookup.indices
        );
    }
    Ok(())
}

/// Captures only the already-live dense operands of the selected bound op.
/// The snapshot happens before that op is encoded, so the CPU interpreter
/// receives the same payload the Metal kernel is about to read without
/// retaining or recursively evaluating any other graph node.
#[cfg(feature = "instrument")]
fn evaluate_selected_bound_cpu(
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    prepared: &Prepared,
    packed_operands: &PackedOperands,
    program: &[Op],
    bound: &BoundOp,
) -> Result<Option<alloc::vec::Vec<f32>>, MetalError> {
    let selected = std::env::var("PROXIMA_METAL_COMPARE_BOUND_NODE")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .map(NodeId);
    if selected != Some(bound.node) {
        return Ok(None);
    }

    for (source, _, lookup) in bound.all_read_sources() {
        if lookup.is_some() {
            return Err(TensorError::BoundF32GatherOperand {
                node: bound.node,
                operand: *source,
            }
            .into());
        }
        if packed_operands.contains_key(source) {
            return Err(TensorError::BoundF32PackedOperand {
                node: bound.node,
                operand: *source,
            }
            .into());
        }
    }

    let sources: BTreeSet<NodeId> = bound
        .all_read_sources()
        .map(|(source, _, _)| *source)
        .collect();
    let output_elements = element_count(prepared.shapes.of(bound.node));
    let snapshot_elements = sources.iter().try_fold(output_elements, |total, source| {
        total.checked_add(element_count(prepared.shapes.of(*source)))
    });
    let snapshot_bytes = snapshot_elements
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
        .unwrap_or(usize::MAX);
    if snapshot_bytes > CPU_BOUND_COMPARE_CAP_BYTES {
        return Err(MetalError::CpuBoundComparisonOverCap {
            node: bound.node,
            bytes: snapshot_bytes,
            cap_bytes: CPU_BOUND_COMPARE_CAP_BYTES,
        });
    }

    let mut owned_buffers: alloc::vec::Vec<Option<alloc::vec::Vec<f32>>> =
        alloc::vec![None; program.len()];
    for source in sources {
        let (buffer, offset) = device_buffers
            .get(&source)
            .ok_or(MetalError::UnresolvedHazardOperand { node: source })?;
        let shape = prepared.shapes.of(source);
        let dtype = gpu_dtype(program, &prepared.index_nodes, source);
        owned_buffers[source.0 as usize] = Some(read_back(
            buffer,
            *offset,
            element_count(shape),
            source,
            dtype,
        )?);
    }
    let borrowed_buffers: alloc::vec::Vec<Option<&[f32]>> = owned_buffers
        .iter()
        .map(|buffer| buffer.as_deref())
        .collect();
    let mut output = alloc::vec![0.0; output_elements];
    let packed_nodes: alloc::vec::Vec<NodeId> = packed_operands.keys().copied().collect();
    proxima_tensor::evaluate_bound_f32_into(bound, &borrowed_buffers, &packed_nodes, &mut output)?;
    Ok(Some(output))
}

#[cfg(feature = "instrument")]
fn compare_bound_f32(
    node: NodeId,
    metal_values: &[f32],
    cpu_values: &[f32],
) -> Result<Option<(usize, f32, f32, f32, f32)>, MetalError> {
    if metal_values.len() != cpu_values.len() {
        return Err(MetalError::CpuBoundComparisonLengthMismatch {
            node,
            metal_len: metal_values.len(),
            cpu_len: cpu_values.len(),
        });
    }
    let mut max_relative = 0.0_f32;
    let mut first_mismatch = None;
    for (element, (metal_value, cpu_value)) in metal_values.iter().zip(cpu_values).enumerate() {
        let finite = metal_value.is_finite() && cpu_value.is_finite();
        let absolute = (metal_value - cpu_value).abs();
        let relative = if finite {
            absolute / cpu_value.abs().max(1e-6)
        } else {
            f32::INFINITY
        };
        max_relative = max_relative.max(relative);
        if (!finite || relative > 1e-2) && first_mismatch.is_none() {
            first_mismatch = Some((element, *metal_value, *cpu_value, relative));
        }
    }
    Ok(first_mismatch.map(|(element, metal, cpu, relative)| {
        (element, metal, cpu, relative, max_relative.max(relative))
    }))
}

#[cfg(feature = "instrument")]
fn report_bound_operands(
    bound: &BoundOp,
    prepared: &Prepared,
    packed_operands: &PackedOperands,
    program: &[Op],
    expert_buffers: Option<&ExpertSourceBuffers>,
    device_buffers: Option<&BTreeMap<NodeId, DeviceBuffer>>,
    sample_element: Option<usize>,
) {
    let sample_coordinate = sample_element.map(|mut element| {
        let mut coordinate = alloc::vec![0_u64; prepared.shapes.of(bound.node).len()];
        for axis in (0..coordinate.len()).rev() {
            let extent = prepared.shapes.of(bound.node)[axis] as usize;
            coordinate[axis] = (element % extent) as u64;
            element /= extent;
        }
        coordinate
    });
    for (operand_index, (source, layout, lookup)) in bound.all_read_sources().enumerate() {
        eprintln!(
            "metal_bound_operand node={:?} operand={} source={source:?} source_name={:?} source_shape={:?} layout={layout:?} lookup={lookup:?} packed_codec={:?} expert_source={}",
            bound.node,
            operand_index,
            program[source.0 as usize].name(),
            prepared.shapes.of(*source),
            packed_operands.get(source),
            expert_buffers.is_some_and(|buffers| buffers.node == *source),
        );
        if let (Some(device_buffers), Some(coordinate)) =
            (device_buffers, sample_coordinate.as_deref())
            && let Some((buffer, offset)) = device_buffers.get(source)
            && let Ok(values) = read_back(
                buffer,
                *offset,
                element_count(prepared.shapes.of(*source)),
                *source,
                gpu_dtype(program, &prepared.index_nodes, *source),
            )
        {
            let source_offset = layout.offset_of(coordinate);
            let source_value = usize::try_from(source_offset)
                .ok()
                .and_then(|index| values.get(index).copied());
            eprintln!(
                "metal_bound_operand_sample node={:?} operand={} coordinate={coordinate:?} source_offset={source_offset} value={source_value:?}",
                bound.node, operand_index
            );
        }
    }
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
    math_mode: MathMode,
    numeric_policy: NumericPolicy,
    expert_buffers: Option<&ExpertSourceBuffers>,
    cpu_reference: Option<&BTreeMap<NodeId, alloc::vec::Vec<f32>>>,
) -> Result<OpGpuTiming, MetalError> {
    validate_selected_expert_routes(device_buffers, prepared, program, bound, expert_buffers)?;
    let bounded_cpu =
        evaluate_selected_bound_cpu(device_buffers, prepared, packed_operands, program, bound)?;
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
    let encoder = EncoderGuard::new(command_buffer.computeCommandEncoder().ok_or_else(|| {
        MetalError::CompileFailed {
            log: "command buffer refused to hand out a compute encoder".to_string(),
        }
    })?);
    let fault = encode_op(
        device,
        &encoder,
        device_buffers,
        bound,
        packed_operands,
        placement,
        plan_uniform,
        math_mode,
        numeric_policy,
        None,
        None,
        None,
        expert_buffers,
    )?;
    encoder.finish();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let gpu_ns =
        ((command_buffer.GPUEndTime() - command_buffer.GPUStartTime()) * 1e9).max(0.0) as u64;
    if let Some((fault_buffer, gathers)) = fault {
        check_gather_fault(bound, &fault_buffer, gathers)?;
    }
    if std::env::var_os("PROXIMA_METAL_NAN_CHECK").is_some()
        && let Some((first_index, shape, metal_values)) =
            check_op_output_finite(device_buffers, prepared, program, bound.node, kind)?
    {
        let cpu_value = bounded_cpu
            .as_ref()
            .or_else(|| cpu_reference.and_then(|reference| reference.get(&bound.node)))
            .and_then(|values| values.get(first_index))
            .copied();
        eprintln!(
            "metal_nan_bound node={:?} name={:?} kind={kind} element={first_index} metal={} cpu={cpu_value:?} shape={shape:?} bound={bound:?}",
            bound.node,
            program[bound.node.0 as usize].name(),
            metal_values[first_index],
        );
        report_bound_operands(
            bound,
            prepared,
            packed_operands,
            program,
            expert_buffers,
            Some(device_buffers),
            Some(first_index),
        );
        return Err(MetalError::NonFiniteOpOutput {
            node: bound.node,
            kind: kind.to_string(),
        });
    }
    if let Some(cpu_values) = bounded_cpu {
        let Some((buffer, offset)) = device_buffers.get(&bound.node) else {
            return Err(MetalError::UnresolvedHazardOperand { node: bound.node });
        };
        let shape = prepared.shapes.of(bound.node).to_vec();
        let dtype = gpu_dtype(program, &prepared.index_nodes, bound.node);
        let metal_values = read_back(buffer, *offset, element_count(&shape), bound.node, dtype)?;
        let first_mismatch = compare_bound_f32(bound.node, &metal_values, &cpu_values)?;
        if let Some((element, metal_value, cpu_value, relative, max_rel_diff)) = first_mismatch {
            let absolute = (metal_value - cpu_value).abs();
            eprintln!(
                "metal_bound_compare node={:?} name={:?} kind={kind} element={element} metal={metal_value} cpu={cpu_value} abs={absolute} rel={relative} max_rel={max_rel_diff} shape={shape:?} bound={bound:?}",
                bound.node,
                program[bound.node.0 as usize].name(),
            );
            report_bound_operands(
                bound,
                prepared,
                packed_operands,
                program,
                expert_buffers,
                Some(device_buffers),
                Some(element),
            );
            return Err(MetalError::CpuMetalDivergence {
                node: bound.node,
                kind: kind.to_string(),
                max_rel_diff,
            });
        }
        eprintln!(
            "metal_bound_compare node={:?} name={:?} kind={kind} result=within_tolerance elements={} shape={shape:?}",
            bound.node,
            program[bound.node.0 as usize].name(),
            metal_values.len(),
        );
    }
    if let Some(cpu_reference) = cpu_reference
        && let Some(max_rel_diff) = compare_op_output_to_cpu(
            device_buffers,
            prepared,
            program,
            bound.node,
            kind,
            cpu_reference,
        )?
        && max_rel_diff > 1e-2
    {
        return Err(MetalError::CpuMetalDivergence {
            node: bound.node,
            kind: kind.to_string(),
            max_rel_diff,
        });
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
        extents: bound.extents.clone(),
        output_axes: match &bound.kind {
            BoundOpKind::Reduce { output_axes, .. } => output_axes.to_vec(),
            _ => Vec::new(),
        },
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
    cpu_reference: Option<&BTreeMap<NodeId, alloc::vec::Vec<f32>>>,
) -> Result<(Evaluated, Vec<OpGpuTiming>), MetalError> {
    execute_plan_op_timed_inner(plan, blocks, &BTreeMap::new(), cpu_reference)
}

/// [`execute_plan_op_timed`]'s routed-expert counterpart. It stages the
/// same per-expert payload and descriptor buffers as
/// [`execute_plan_with_expert_sources`] before splitting the plan into one
/// command buffer per bound operation.
///
/// # Errors
/// Same as [`execute_plan_with_expert_sources`] and [`execute_plan_op_timed`].
#[cfg(feature = "instrument")]
pub fn execute_plan_op_timed_with_expert_sources(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    expert_sources: &BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
    cpu_reference: Option<&BTreeMap<NodeId, alloc::vec::Vec<f32>>>,
) -> Result<(Evaluated, Vec<OpGpuTiming>), MetalError> {
    if expert_sources.is_empty() {
        return execute_plan_op_timed(plan, blocks, cpu_reference);
    }
    let expert_buffers = stage_expert_sources(plan, blocks, expert_sources)?;
    execute_plan_op_timed_inner(plan, blocks, &expert_buffers, cpu_reference)
}

#[cfg(feature = "instrument")]
fn execute_plan_op_timed_inner(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    expert_buffers: &BTreeMap<NodeId, ExpertSourceBuffers>,
    cpu_reference: Option<&BTreeMap<NodeId, alloc::vec::Vec<f32>>>,
) -> Result<(Evaluated, Vec<OpGpuTiming>), MetalError> {
    let prepared = &plan.prepared;
    let packed_operands = &plan.packed_operands;
    let selected_bound = std::env::var("PROXIMA_METAL_COMPARE_BOUND_NODE")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .map(NodeId);
    if let Some(node) = selected_bound
        && !prepared.resolved.iter().any(|bound| bound.node == node)
    {
        eprintln!(
            "metal_bound_compare_missing node={node:?} available={:?}",
            prepared
                .resolved
                .iter()
                .map(|bound| (bound.node, bound.kind.name()))
                .collect::<Vec<_>>()
        );
        return Err(MetalError::CpuBoundComparisonNodeNotFound { node });
    }

    let mut effective_expert_buffers = expert_buffers.clone();
    for (source_node, buffers) in expert_buffers {
        let Some(source_name) = plan.program[source_node.0 as usize].name() else {
            continue;
        };
        for (position, operation) in plan.program.iter().enumerate() {
            if operation.name() == Some(source_name) {
                effective_expert_buffers
                    .entry(NodeId(position as u32))
                    .or_insert_with(|| buffers.clone());
            }
        }
    }

    let (device, queue) = device_and_queue()?;

    let mut device_buffers: BTreeMap<NodeId, DeviceBuffer> = BTreeMap::new();
    for ((node, block), dtype) in prepared
        .block_nodes
        .iter()
        .zip(blocks.iter())
        .zip(plan.block_dtypes.iter())
    {
        if let Some(buffers) = effective_expert_buffers.get(node) {
            device_buffers.insert(*node, (buffers.payloads.clone(), buffers.payload_offset));
            continue;
        }
        let resident_name = resident_name(plan, *node);
        let buffer = match block {
            QuantizedBlock::Float32(data) => upload_block(
                &device,
                data,
                *node,
                *dtype,
                plan.program[node.0 as usize].name(),
                resident_name,
            )?,
            QuantizedBlock::Int32(data) => upload_block_int32_as_float(&device, data, None)?,
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
            | QuantizedBlock::Q5_1(bytes)
            | QuantizedBlock::Q2K(bytes)
            | QuantizedBlock::Iq4Nl(bytes)
            | QuantizedBlock::Iq2Xs(bytes)
            | QuantizedBlock::Iq3Xxs(bytes)
            | QuantizedBlock::Float16(bytes)
            | QuantizedBlock::BFloat16(bytes) => {
                upload_packed_bytes(&device, bytes, resident_name)?
            }
        };
        device_buffers.insert(*node, buffer);
    }

    let no_placements: BTreeSet<NodeId> = BTreeSet::new();
    let mut timings: Vec<OpGpuTiming> = Vec::with_capacity(prepared.resolved.len());
    for (position, bound) in prepared.resolved.iter().enumerate() {
        let expert_buffers = expert_buffers_for(bound, &effective_expert_buffers)?;
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
            plan.math_mode,
            plan.numeric_policy,
            expert_buffers,
            cpu_reference,
        )?;
        timings.push(timing);
    }

    let evaluated = finish(plan, &device_buffers, &BTreeSet::new(), None)?;
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
    cpu_reference: Option<&BTreeMap<NodeId, alloc::vec::Vec<f32>>>,
) -> Result<(Evaluated, Vec<OpGpuTiming>), MetalError> {
    let blocks = resolve_named_blocks(&plan.program, named)?;
    execute_plan_op_timed(plan, &blocks, cpu_reference)
}

/// Name-keyed counterpart of
/// [`execute_plan_op_timed_with_expert_sources`].
///
/// # Errors
/// Propagates name-resolution, expert-source, and Metal driver failures.
#[cfg(feature = "instrument")]
pub fn execute_plan_named_op_timed_with_expert_sources(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    expert_sources: &BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
) -> Result<(Evaluated, Vec<OpGpuTiming>), MetalError> {
    let blocks = resolve_named_blocks(&plan.program, named)?;
    execute_plan_op_timed_with_expert_sources(plan, &blocks, expert_sources, None)
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
        let resident_name = resident_name(plan, *node);
        let buffer = match block {
            QuantizedBlock::Float32(data) => upload_block(
                &device,
                data,
                *node,
                *dtype,
                plan.program[node.0 as usize].name(),
                resident_name,
            )?,
            QuantizedBlock::Int32(data) => upload_block_int32_as_float(&device, data, None)?,
            // `Q3_K` uploads its raw super-block bytes unchanged, same as
            // every other packed codec below -- `msl::PackedCodec::Q3K`'s
            // own unpack kernel (`q3k_element`) reads them at the GPU side.
            QuantizedBlock::Q3K(bytes)
            | QuantizedBlock::Q4K(bytes)
            | QuantizedBlock::Q5K(bytes)
            | QuantizedBlock::Q6K(bytes)
            | QuantizedBlock::Q8_0(bytes)
            | QuantizedBlock::Q4_0(bytes)
            | QuantizedBlock::Q5_1(bytes)
            | QuantizedBlock::Q2K(bytes)
            | QuantizedBlock::Iq4Nl(bytes)
            | QuantizedBlock::Iq2Xs(bytes)
            | QuantizedBlock::Iq3Xxs(bytes)
            | QuantizedBlock::Float16(bytes)
            | QuantizedBlock::BFloat16(bytes) => {
                upload_packed_bytes(&device, bytes, resident_name)?
            }
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
            plan.math_mode,
            plan.numeric_policy,
            None,
            None,
        )?;
        timings.push(timing);
    }

    let placed_output_nodes: BTreeSet<NodeId> = output_placed.keys().copied().collect();
    let evaluated = finish(plan, &device_buffers, &placed_output_nodes, None)?;
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

/// Which [`MTLCounterSamplingPoint`] a device actually honors -- resolved
/// once per [`execute_plan_with_placements_dispatch_timed`] call (this is a
/// diagnostic-only path, never the hot loop, so re-querying every call
/// costs nothing that matters) rather than assumed: an M1 Max reports
/// `AtDispatchBoundary` unsupported and `AtStageBoundary` supported
/// (verified with a standalone Metal probe against this exact device, not
/// inferred from Apple's docs), so the dispatch-boundary branch below is
/// exercised only on hardware that actually has it.
#[cfg(feature = "instrument")]
fn counter_sampling_mode(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Option<objc2_metal::MTLCounterSamplingPoint> {
    if device.supportsCounterSampling(objc2_metal::MTLCounterSamplingPoint::AtDispatchBoundary) {
        Some(objc2_metal::MTLCounterSamplingPoint::AtDispatchBoundary)
    } else if device.supportsCounterSampling(objc2_metal::MTLCounterSamplingPoint::AtStageBoundary)
    {
        Some(objc2_metal::MTLCounterSamplingPoint::AtStageBoundary)
    } else {
        None
    }
}

/// The device's own `MTLCommonCounterSetTimestamp` counter set, by name --
/// `device.counterSets()` returns every set the hardware exposes; this
/// finds the one [`objc2_metal::MTLCommonCounterSetTimestamp`] names,
/// rather than assuming index `0`.
#[cfg(feature = "instrument")]
fn timestamp_counter_set(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Option<Retained<ProtocolObject<dyn objc2_metal::MTLCounterSet>>> {
    let sets = device.counterSets()?;
    // SAFETY: `MTLCommonCounterSetTimestamp` is a framework-provided
    // constant `NSString`, valid for the process lifetime.
    let timestamp_name = unsafe { objc2_metal::MTLCommonCounterSetTimestamp };
    sets.iter().find(|set| &*set.name() == timestamp_name)
}

/// [`execute_plan_with_placements`]'s per-dispatch GPU-timestamp twin.
/// Unlike [`execute_plan_with_placements_op_timed`] (one command buffer per
/// op -- discipline log ROW 298's own finding that this reproduces neither
/// the batched buffer's barrier/serialization cost nor its aggregate
/// total), this function submits the SAME single command buffer the
/// production path does and brackets every dispatch with Metal's
/// `MTLCounterSampleBuffer` against `MTLCommonCounterSetTimestamp`
/// (`sampleCountersInBuffer:atSampleIndex:withBarrier:` when the device
/// supports `AtDispatchBoundary`; one compute encoder per position, sampled
/// at `startOfEncoderSampleIndex`/`endOfEncoderSampleIndex`, when it only
/// supports `AtStageBoundary` -- `AtStageBoundary`'s own granularity is
/// per-ENCODER, so that branch degrades to "one encoder per dispatch"
/// rather than the coarser "one encoder per kind" grouping, since this
/// program's own dispatch sequence rarely repeats the same op kind AND
/// shape twice in a row -- see the module-level discipline row this
/// function's own commit lands for the measured device support and the
/// resulting per-shape table).
///
/// GPU tick counts are converted to nanoseconds by calibrating against this
/// SAME call's own CPU/GPU timestamp pair (`sampleTimestamps:gpuTimestamp:`
/// taken once before `commit` and once after `waitUntilCompleted`): the
/// CPU side of that pair is in the same `mach_absolute_time` domain
/// [`ticks_to_nanos`] already converts, so `cpu_ns_delta / gpu_tick_delta`
/// is this call's own nanoseconds-per-GPU-tick, applied uniformly to every
/// per-position delta. No new type is introduced: the per-position record
/// is [`OpGpuTiming`], the same type [`execute_plan_with_placements_op_timed`]
/// already returns.
///
/// [`execute_plan_with_placements_dispatch_timed`]/
/// [`execute_plan_named_with_placements_dispatch_timed`]'s own return
/// shape, named only because clippy's `type_complexity` lint requires it
/// for a four-element tuple -- not a new abstraction, the same
/// `DispatchMeta`-alias precedent ROW 309 already set for this function's
/// internals, just applied to its public return type too.
#[cfg(all(feature = "metal-output-placement", feature = "instrument"))]
pub type DispatchTimedOutcome = (
    Evaluated,
    Vec<OpGpuTiming>,
    &'static str,
    Option<(u64, u64)>,
);

/// Returns `(evaluated, per_position_timings, sampling_mode, encoder_split_ns)`,
/// where `sampling_mode` is `"dispatch-boundary"`, `"stage-boundary"`, or
/// `"unsupported"` -- in the last case every `gpu_ns` is `0`, never
/// fabricated -- and `encoder_split_ns` is `Some((encoder_1_ns,
/// encoder_2_ns))` exactly when [`Plan::set_encoder_split_at`] named a
/// split position AND the stage-boundary branch actually ran (ROW 329):
/// three `MTLCounterSampleBuffer` samples (buffer start, encoder-1 end,
/// encoder-2 end) replace `per_position_timings`' own per-position samples
/// for this call, so every `OpGpuTiming::gpu_ns` reads `0` when this is
/// `Some` -- the same "never fabricated" contract `"unsupported"` already
/// carries, extended to a case where per-op attribution genuinely was not
/// sampled, not just unsupported by the device.
///
/// # Errors
/// Same as [`execute_plan_with_placements`], plus a [`MetalError`] if the
/// device refuses to allocate the counter sample buffer.
#[cfg(all(feature = "metal-output-placement", feature = "instrument"))]
#[allow(clippy::too_many_lines)]
pub fn execute_plan_with_placements_dispatch_timed(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
) -> Result<DispatchTimedOutcome, MetalError> {
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

    let mode = counter_sampling_mode(&device);
    let counter_set = mode.and_then(|_| timestamp_counter_set(&device));

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
        #[cfg(feature = "instrument")]
        {
            counter!(BLOCK_UPLOAD_CALLS, 1);
            counter!(BLOCK_OFFERED_BYTES, block_byte_len(block) as u64);
        }
        let resident_name = resident_name(plan, *node);
        let buffer = match block {
            QuantizedBlock::Float32(data) => upload_block(
                &device,
                data,
                *node,
                *dtype,
                plan.program[node.0 as usize].name(),
                resident_name,
            )?,
            QuantizedBlock::Int32(data) => upload_block_int32_as_float(&device, data, None)?,
            QuantizedBlock::Q3K(bytes)
            | QuantizedBlock::Q4K(bytes)
            | QuantizedBlock::Q5K(bytes)
            | QuantizedBlock::Q6K(bytes)
            | QuantizedBlock::Q8_0(bytes)
            | QuantizedBlock::Q4_0(bytes)
            | QuantizedBlock::Q5_1(bytes)
            | QuantizedBlock::Q2K(bytes)
            | QuantizedBlock::Iq4Nl(bytes)
            | QuantizedBlock::Iq2Xs(bytes)
            | QuantizedBlock::Iq3Xxs(bytes)
            | QuantizedBlock::Float16(bytes)
            | QuantizedBlock::BFloat16(bytes) => {
                upload_packed_bytes(&device, bytes, resident_name)?
            }
        };
        device_buffers.insert(*node, buffer);
    }

    let position_count = prepared.resolved.len();
    let Some((sampling_point, counter_set)) = mode.zip(counter_set) else {
        // no counter set / sampling point at all -- fall through to the
        // production single-encoder submission with zero instrumentation
        // rather than refusing to run: `gpu_ns` stays `0` on every entry,
        // never fabricated, and `sampling_mode` names this plainly.
        let evaluated = execute_plan_with_placements(
            plan,
            blocks,
            input_placements,
            output_placements,
            &mut Vec::new(),
        )?;
        let timings: Vec<OpGpuTiming> = prepared
            .resolved
            .iter()
            .map(|bound| OpGpuTiming {
                node: bound.node,
                kind: classify_kind(bound, packed_operands),
                extents: bound.extents.clone(),
                output_axes: match &bound.kind {
                    BoundOpKind::Reduce { output_axes, .. } => output_axes.to_vec(),
                    _ => Vec::new(),
                },
                operand_bytes: 0,
                bound_buffer_bytes: 0,
                gpu_ns: 0,
                weight_name: None,
                operand_count: bound.operands().len(),
                packed_codec: None,
                packed_kernel_variant: classify_packed_kernel_variant(bound, packed_operands),
                packed_row_block_rejection: None,
            })
            .collect();
        return Ok((evaluated, timings, "unsupported", None));
    };

    let dispatch_boundary =
        sampling_point == objc2_metal::MTLCounterSamplingPoint::AtDispatchBoundary;
    // ROW 329: the split feature only applies to the stage-boundary
    // fallback this device (M1 Max) actually takes -- a device that
    // supports `AtDispatchBoundary` already gets per-DISPATCH granularity
    // from `shared_encoder`'s manual `sampleCountersInBuffer` calls below,
    // with no per-encoder overhead to economize on, so `encoder_split_at`
    // is ignored there rather than silently reinterpreted.
    let split_at = if dispatch_boundary {
        None
    } else {
        plan.encoder_split_at
    };

    let sample_descriptor = objc2_metal::MTLCounterSampleBufferDescriptor::new();
    sample_descriptor.setCounterSet(Some(&counter_set));
    // Three samples (buffer start, encoder-1 end, encoder-2 end) when
    // split, else the original one-pair-per-position sizing. SAFETY: both
    // counts are plain arithmetic, well under any device's
    // `maxBufferLength`-scale sample-buffer limits for the per-token
    // dispatch counts this workspace's own decode programs emit.
    let sample_count: usize = if split_at.is_some() {
        3
    } else {
        2 * position_count
    };
    unsafe { sample_descriptor.setSampleCount(sample_count as NSUInteger) };
    let sample_buffer = device
        .newCounterSampleBufferWithDescriptor_error(&sample_descriptor)
        .map_err(|error| MetalError::CompileFailed {
            log: format!("failed to allocate a counter sample buffer: {error}"),
        })?;

    let command_buffer = queue
        .commandBuffer()
        .ok_or_else(|| MetalError::CompileFailed {
            log: "command queue refused to hand out a command buffer".to_string(),
        })?;

    let shared_encoder = if dispatch_boundary {
        Some(EncoderGuard::new(
            command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| MetalError::CompileFailed {
                    log: "command buffer refused to hand out a compute encoder".to_string(),
                })?,
        ))
    } else {
        None
    };
    // ROW 329: the stage-boundary encoder currently open, carried across
    // loop iterations so a split group's encoder stays open for every
    // position inside it -- `None` until the loop's first iteration
    // creates one. Unused (stays `None` the whole call) when
    // `dispatch_boundary` is true, since `shared_encoder` covers that case.
    // `EncoderGuard`-owned (not a plain `Retained`) so a `?` from inside the
    // loop below -- `arena_placement`, `plan_uniform_buffer`, `encode_op`'s
    // own error sites -- ends whichever encoder is currently open at `Drop`
    // instead of leaking it into the assertion this guard exists to avoid.
    let mut stage_encoder: Option<EncoderGuard> = None;

    // one tuple per position: `(node, kind, operand_bytes, bound_buffer_bytes,
    // weight_name, operand_count, packed_codec, packed_kernel_variant)` --
    // every field [`OpGpuTiming`] already carries, gathered while `bound`
    // is still live so the counter-resolve pass below only zips it against
    // `gpu_ns`.
    type DispatchMeta = (
        NodeId,
        &'static str,
        u64,
        u64,
        Option<String>,
        usize,
        Option<PackedCodec>,
        &'static str,
    );
    let mut metas: Vec<DispatchMeta> = Vec::with_capacity(position_count);
    let mut pending_faults: Vec<PendingFault<'_>> = Vec::new();

    let cpu_gpu_start = sample_timestamps(&device);
    for (position, bound) in prepared.resolved.iter().enumerate() {
        let placement = match output_placed.get(&bound.node).copied() {
            Some(placement) => Some(placement),
            None => arena_placement(plan, position)?,
        };
        let uniform_buffer = plan_uniform_buffer(plan, position)?;
        let operand_bytes: u64 = bound
            .operands()
            .iter()
            .map(|(source, _, _)| {
                operand_tensor_bytes(
                    &plan.program,
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
            .find_map(|(source, _, _)| plan.program[source.0 as usize].name())
            .map(ToString::to_string);
        let packed_codec = bound
            .operands()
            .iter()
            .find_map(|(source, _, _)| packed_operands.get(source).copied());
        metas.push((
            bound.node,
            classify_kind(bound, packed_operands),
            operand_bytes,
            bound_buffer_bytes,
            weight_name,
            bound.operands().len(),
            packed_codec,
            classify_packed_kernel_variant(bound, packed_operands),
        ));

        let encoder = match &shared_encoder {
            Some(guard) => guard.clone_inner(),
            None => {
                // ROW 329: without a split, every position opens (and, below,
                // immediately closes) its own encoder -- ROW 309's original
                // fallback. With a split, a new encoder opens only at
                // position 0 and at `split_at` itself, and stays open for
                // every position in between (closed just before the NEXT
                // new encoder opens, or after the loop for the last group).
                let needs_new_encoder = match split_at {
                    None => true,
                    Some(split) => position == 0 || position == split,
                };
                // A group's non-boundary positions reuse the encoder a prior
                // iteration in this SAME group already opened -- `None` here
                // (rather than an `expect`) falls through to opening a fresh
                // one instead of panicking, which cannot happen given
                // `needs_new_encoder`'s own boundary check above but costs
                // nothing to make self-healing rather than load-bearing.
                if let Some(existing) = (!needs_new_encoder)
                    .then(|| stage_encoder.as_ref().map(EncoderGuard::clone_inner))
                    .flatten()
                {
                    existing
                } else {
                    if let Some(previous) = stage_encoder.take() {
                        previous.finish();
                    }
                    let descriptor = objc2_metal::MTLComputePassDescriptor::computePassDescriptor();
                    let attachment = unsafe {
                        descriptor
                            .sampleBufferAttachments()
                            .objectAtIndexedSubscript(0)
                    };
                    attachment.setSampleBuffer(Some(&sample_buffer));
                    let (start_index, end_index) = match split_at {
                        None => (2 * position, 2 * position + 1),
                        Some(split) if position == split => (objc2_metal::MTLCounterDontSample, 2),
                        Some(_) => (0, 1),
                    };
                    unsafe {
                        attachment.setStartOfEncoderSampleIndex(start_index as NSUInteger);
                        attachment.setEndOfEncoderSampleIndex(end_index as NSUInteger);
                    }
                    let opened = command_buffer
                        .computeCommandEncoderWithDescriptor(&descriptor)
                        .ok_or_else(|| MetalError::CompileFailed {
                            log:
                                "command buffer refused to hand out a stage-sampled compute encoder"
                                    .to_string(),
                        })?;
                    stage_encoder = Some(EncoderGuard::new(opened.clone()));
                    opened
                }
            }
        };

        if dispatch_boundary {
            unsafe {
                encoder.sampleCountersInBuffer_atSampleIndex_withBarrier(
                    &sample_buffer,
                    (2 * position) as NSUInteger,
                    true,
                );
            }
        }
        let fault = encode_op(
            &device,
            &encoder,
            &mut device_buffers,
            bound,
            packed_operands,
            placement,
            uniform_buffer,
            plan.math_mode,
            plan.numeric_policy,
            None,
            None,
            None,
            None,
        )?;
        if dispatch_boundary {
            unsafe {
                encoder.sampleCountersInBuffer_atSampleIndex_withBarrier(
                    &sample_buffer,
                    (2 * position + 1) as NSUInteger,
                    true,
                );
            }
        }
        // ROW 329: a non-split stage-boundary encoder still closes here,
        // right after its one dispatch -- `needs_new_encoder` was always
        // `true` for it above, so `stage_encoder` holds exactly this
        // encoder and the very next iteration's `previous.endEncoding()`
        // (or, on the last position, the post-loop drain below) would
        // otherwise be the only place it closes. Closing immediately
        // instead reproduces ROW 309's original per-position timing
        // exactly, rather than silently widening every reported encoder by
        // one dispatch's worth of retire bookkeeping.
        if shared_encoder.is_none()
            && split_at.is_none()
            && let Some(open) = stage_encoder.take()
        {
            open.finish();
        }
        if let Some((fault_buffer, gathers)) = fault {
            pending_faults.push((bound, fault_buffer, gathers));
        }
        for retired in &prepared.retires[position] {
            if always_live.contains(retired) {
                continue;
            }
            device_buffers.remove(retired);
        }
    }
    if let Some(encoder) = shared_encoder {
        encoder.finish();
    }
    // ROW 329: the last group's stage-boundary encoder (split mode's
    // encoder-2, or a non-split call that somehow left one open) never hit
    // the per-iteration close above.
    if let Some(open) = stage_encoder.take() {
        open.finish();
    }

    // Same CPU-wall-clock-around-commit-and-wait shape
    // `execute_plan_with_placements`'s own `GPU_EXEC_TICKS` counter uses --
    // ROW 309 found this diagnostic path left `gpu_exec_ms` unrecorded for
    // its own step (the production path's counter bump never runs when
    // this function replaces it), so a caller comparing this call's
    // encoder-split totals against `gpu_exec_ms` had no same-step number to
    // compare against. Wiring the identical counter here closes that gap
    // with no second timing mechanism: it is the same `cpu_gpu_start`/
    // `cpu_gpu_end` bracket this function already takes for its own
    // nanosecond calibration, just also fed to the shared counter.
    let gpu_exec_started = read_ticks();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    counter!(GPU_EXEC_CALLS, 1);
    counter!(GPU_EXEC_TICKS, elapsed_ticks(gpu_exec_started));
    let cpu_gpu_end = sample_timestamps(&device);

    for (bound, fault_buffer, gathers) in &pending_faults {
        check_gather_fault(bound, fault_buffer, *gathers)?;
    }

    let cpu_ns_delta = ticks_to_nanos(cpu_gpu_end.0.wrapping_sub(cpu_gpu_start.0));
    let gpu_tick_delta = cpu_gpu_end.1.saturating_sub(cpu_gpu_start.1).max(1);
    let ns_per_gpu_tick = cpu_ns_delta as f64 / gpu_tick_delta as f64;

    let range = objc2_foundation::NSRange {
        location: 0,
        length: sample_count as NSUInteger,
    };
    // SAFETY: `range` is bounds-checked by construction above (it spans
    // exactly the `sample_count` samples this call itself wrote);
    // `resolveCounterRange`'s own unsafety is the driver's undocumented
    // out-of-range behavior, which this range cannot trigger.
    let resolved = unsafe { sample_buffer.resolveCounterRange(range) }.ok_or_else(|| {
        MetalError::CompileFailed {
            log: "counter sample buffer resolved no data".to_string(),
        }
    })?;
    let raw = resolved.to_vec();
    // ROW 329: split mode's three samples describe two ENCODER-level spans,
    // not `position_count` per-position ones -- reading them as
    // `2 * position` pairs (the non-split layout) would read past index 2
    // for every position beyond the first and silently attribute garbage
    // `gpu_ns`. Every `OpGpuTiming::gpu_ns` stays `0` here instead, same as
    // `"unsupported"` -- never fabricated -- and `encoder_split_ns` below
    // carries the real, encoder-granularity numbers this call actually
    // sampled.
    let encoder_split_ns = split_at.map(|_| {
        let buffer_start = read_timestamp(&raw, 0);
        let encoder_one_end = read_timestamp(&raw, 1);
        let encoder_two_end = read_timestamp(&raw, 2);
        let encoder_one_ns = if buffer_start == u64::MAX || encoder_one_end == u64::MAX {
            0
        } else {
            (encoder_one_end.wrapping_sub(buffer_start) as f64 * ns_per_gpu_tick).max(0.0) as u64
        };
        let encoder_two_ns = if encoder_one_end == u64::MAX || encoder_two_end == u64::MAX {
            0
        } else {
            (encoder_two_end.wrapping_sub(encoder_one_end) as f64 * ns_per_gpu_tick).max(0.0) as u64
        };
        (encoder_one_ns, encoder_two_ns)
    });
    let mut timings = Vec::with_capacity(position_count);
    for (position, meta) in metas.into_iter().enumerate() {
        let gpu_ns = if split_at.is_some() {
            0
        } else {
            let start = read_timestamp(&raw, 2 * position);
            let end = read_timestamp(&raw, 2 * position + 1);
            if start == u64::MAX || end == u64::MAX {
                0
            } else {
                (end.wrapping_sub(start) as f64 * ns_per_gpu_tick).max(0.0) as u64
            }
        };
        let (
            node,
            kind,
            operand_bytes,
            bound_buffer_bytes,
            weight_name,
            operand_count,
            packed_codec,
            packed_kernel_variant,
        ) = meta;
        timings.push(OpGpuTiming {
            node,
            kind,
            extents: prepared.resolved[position].extents.clone(),
            output_axes: match &prepared.resolved[position].kind {
                BoundOpKind::Reduce { output_axes, .. } => output_axes.to_vec(),
                _ => Vec::new(),
            },
            operand_bytes,
            bound_buffer_bytes,
            gpu_ns,
            weight_name,
            operand_count,
            packed_codec,
            packed_kernel_variant,
            packed_row_block_rejection: None,
        });
    }

    let placed_output_nodes: BTreeSet<NodeId> = output_placed.keys().copied().collect();
    let evaluated = finish(plan, &device_buffers, &placed_output_nodes, None)?;
    let sampling_mode = if dispatch_boundary {
        "dispatch-boundary"
    } else {
        "stage-boundary"
    };
    Ok((evaluated, timings, sampling_mode, encoder_split_ns))
}

/// One `(cpuTimestamp, gpuTimestamp)` reading via
/// `MTLDevice::sampleTimestamps:gpuTimestamp:` -- the CPU side is in the
/// same `mach_absolute_time` domain [`ticks_to_nanos`] converts, which is
/// what lets [`execute_plan_with_placements_dispatch_timed`] calibrate GPU
/// ticks to nanoseconds without a second, unrelated conversion table.
#[cfg(feature = "instrument")]
fn sample_timestamps(device: &ProtocolObject<dyn MTLDevice>) -> (u64, u64) {
    let mut cpu_timestamp: objc2_metal::MTLTimestamp = 0;
    let mut gpu_timestamp: objc2_metal::MTLTimestamp = 0;
    // SAFETY: both out-pointers are valid, stack-local `u64`s.
    unsafe {
        device.sampleTimestamps_gpuTimestamp(
            core::ptr::NonNull::from(&mut cpu_timestamp),
            core::ptr::NonNull::from(&mut gpu_timestamp),
        );
    }
    (cpu_timestamp, gpu_timestamp)
}

/// One [`objc2_metal::MTLCounterResultTimestamp`]'s 8-byte little-endian
/// `timestamp` field out of `resolveCounterRange`'s raw `NSData` bytes --
/// `u64::MAX` (Metal's own `MTLCounterErrorValue`) when the driver could
/// not take that sample, which the caller treats as "no measurement", never
/// a real zero-length dispatch.
#[cfg(feature = "instrument")]
fn read_timestamp(raw: &[u8], sample_index: usize) -> u64 {
    let start = sample_index * core::mem::size_of::<u64>();
    raw.get(start..start + core::mem::size_of::<u64>())
        .and_then(|bytes| bytes.try_into().ok())
        .map(u64::from_ne_bytes)
        .unwrap_or(u64::MAX)
}

/// [`execute_plan_with_placements_dispatch_timed`] against a name-keyed
/// block set, mirroring [`execute_plan_named_with_placements_op_timed`]'s
/// own name resolution.
///
/// # Errors
/// Propagates name-resolution and Metal driver failures.
#[cfg(all(feature = "metal-output-placement", feature = "instrument"))]
pub fn execute_plan_named_with_placements_dispatch_timed(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
) -> Result<DispatchTimedOutcome, MetalError> {
    let blocks = resolve_named_blocks(&plan.program, named)?;
    execute_plan_with_placements_dispatch_timed(plan, &blocks, input_placements, output_placements)
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
        // `BoundOpKind::name()` is the one place these four (plus
        // `keep::scan fold` below) are spelled -- this arm never restates
        // its own copy, so a future variant or renamed arm cannot drift
        // between this profiler label and `RenderKindMismatch`'s own.
        BoundOpKind::CachedAttention { .. }
        | BoundOpKind::Elementwise { .. }
        | BoundOpKind::Iota
        | BoundOpKind::Constant { .. } => bound.kind.name(),
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => bound.kind.name(),
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => match emit(bound, packed_operands, NumericPolicy::default()) {
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
            // Consults `msl::PACKED_ROW_BODY_MARKERS` rather than restating
            // its own copy of the marker list: the restated copy is exactly
            // what went stale before (missing `q3k_pair_dot(blk`/
            // `q3k_element(blk`/`q6k_pair_dot(blk`, see that const's own
            // doc), undercounting every Q3_K row-blocked dispatch and every
            // plain-product Q6_K dispatch into `"reduce-cooperative"` below.
            Ok(kernel)
                if crate::msl::PACKED_ROW_BODY_MARKERS
                    .iter()
                    .any(|marker| kernel.source.contains(marker)) =>
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
    let Ok(kernel) = emit(bound, packed_operands, NumericPolicy::default()) else {
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

#[cfg(all(test, feature = "instrument"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod classify_kind_packed_row_marker_tests {
    //! `classify_kind` used to restate its own copy of
    //! [`crate::msl::PACKED_ROW_BODY_MARKERS`] and the copy went stale: it
    //! carried `q4k_pair_dot(blk`/`q5k_pair_dot(blk`/`q5k_value(blk`/
    //! `q6k_value(blk` but never `q3k_pair_dot(blk`/`q3k_element(blk`/
    //! `q6k_pair_dot(blk` (`msl.rs:5142`, `5163`, `5298`), so every Q3_K
    //! row-blocked dispatch and every plain-product Q6_K dispatch (the shape
    //! the openchat output head actually takes) fell through to
    //! `"reduce-cooperative"`. Renders ONE kernel per (codec, body) pair
    //! through the real `emit` path -- the plain-product pair-dot arm (an
    //! `Add`-reduce over a `Multiply` body) and the per-element scalar
    //! fallback arm (any other reduce op) -- for all four K-quant codecs.

    use alloc::collections::BTreeMap;
    use alloc::vec;

    use proxima_tensor::{
        BoundOp, DType, Extent, IndexMap, NumericPolicy, Op, Reduce, ReduceInit, ScalarOp, append,
        bind, infer, map,
    };

    use super::classify_kind;
    use crate::{PackedCodec, PackedOperands};

    fn matmul_op_with_reduce(m: u32, k: u32, n: u32, reduce_op: ScalarOp) -> BoundOp {
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
                body: reduce_op,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: proxima_tensor::Keep::Reduce,
                name: Some("matmul".into()),
            }),
        );
        let shapes = infer(&program, &[]).expect("matmul infers");
        bind(&program, &shapes, &[], NumericPolicy::default())
            .expect("matmul lowers")
            .into_iter()
            .next()
            .expect("one fused bound emitted")
    }

    fn packed_operands_for(bound: &BoundOp, codec: PackedCodec) -> PackedOperands {
        let mut codecs = BTreeMap::new();
        codecs.insert(bound.operands()[0].0, codec);
        codecs
    }

    #[test]
    fn every_k_quant_codec_classifies_as_packed_row_blocked_plain_product() {
        for codec in [
            PackedCodec::Q3K,
            PackedCodec::Q4K,
            PackedCodec::Q5K,
            PackedCodec::Q6K,
        ] {
            // Add-reduce over a plain `weight * activation` body selects the
            // `plain_product` pair-dot arm (`push_packed_row_blocked_body`'s
            // own `plain_product` gate) for every codec that supports it.
            let bound = matmul_op_with_reduce(4, 256, 5, ScalarOp::Add);
            let packed_operands = packed_operands_for(&bound, codec);
            assert_eq!(
                classify_kind(&bound, &packed_operands),
                "reduce-packed-row-blocked",
                "{codec:?} plain-product row-blocked dispatch must classify as \
                 reduce-packed-row-blocked, not fall through to reduce-cooperative"
            );
        }
    }

    #[test]
    fn every_k_quant_codec_classifies_as_packed_row_blocked_scalar_fallback() {
        for codec in [
            PackedCodec::Q3K,
            PackedCodec::Q4K,
            PackedCodec::Q5K,
            PackedCodec::Q6K,
        ] {
            // A non-Add reduce op takes `push_packed_row_blocked_body`'s
            // per-element scalar fallback arm instead of the pair-dot arm.
            let bound = matmul_op_with_reduce(4, 256, 5, ScalarOp::Maximum);
            let packed_operands = packed_operands_for(&bound, codec);
            assert_eq!(
                classify_kind(&bound, &packed_operands),
                "reduce-packed-row-blocked",
                "{codec:?} per-element scalar row-blocked dispatch must classify as \
                 reduce-packed-row-blocked, not fall through to reduce-cooperative"
            );
        }
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
    let BoundOpKind::Reduce {
        keep: Keep::Reduce,
        output_axes,
        ..
    } = &bound.kind
    else {
        return None;
    };
    let quantized: Vec<Option<PackedCodec>> = bound
        .operands()
        .iter()
        .map(|(node, _, _)| packed_operands.get(node).copied())
        .collect();
    Some(match diagnose_packed_row_block(bound, &quantized) {
        Ok(()) => "PASS".to_string(),
        Err(rejection) => {
            let operand_layouts: Vec<_> = bound
                .operands()
                .iter()
                .map(|(_, layout, _)| layout)
                .collect();
            format!(
                "{rejection:?}; extents={:?}; output_axes={output_axes:?}; operand_layouts={operand_layouts:?}",
                bound.extents,
            )
        }
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
    /// The single, ROW-327-fixed attribution of `block_nodes` to codecs --
    /// computed once here, by [`packed_operands_of`], off blocks already
    /// checked count- and shape-consistent against `block_nodes` (see this
    /// function's own doc). [`plan`] reuses this instead of recomputing it
    /// a second time against the same `blocks` argument.
    packed_operands: PackedOperands,
    resolved: Vec<BoundOp>,
    retires: Vec<Vec<NodeId>>,
    /// Every node referenced as a gather's `indices` anywhere in the
    /// program — see [`gpu_dtype`]'s doc for why upload/read-back both
    /// need this set alongside a node's own declared dtype.
    index_nodes: BTreeSet<NodeId>,
}

/// Raw host bytes one [`QuantizedBlock`] hands [`upload_block`]/
/// [`upload_packed_bytes`] — the split-4019 "block upload" term's byte count,
/// distinct from [`QuantizedBlock::element_count`]'s element count (a `Q4_K`
/// super-block's bytes and elements are not the same unit either).
#[cfg(feature = "instrument")]
fn block_byte_len(block: &QuantizedBlock<'_>) -> usize {
    match block {
        QuantizedBlock::Float32(data) => size_of_val(*data),
        QuantizedBlock::Int32(data) => size_of_val(*data),
        QuantizedBlock::Q3K(bytes)
        | QuantizedBlock::Q4K(bytes)
        | QuantizedBlock::Q5K(bytes)
        | QuantizedBlock::Q6K(bytes)
        | QuantizedBlock::Q8_0(bytes)
        | QuantizedBlock::Q4_0(bytes)
        | QuantizedBlock::Q5_1(bytes)
        | QuantizedBlock::Q2K(bytes)
        | QuantizedBlock::Iq4Nl(bytes)
        | QuantizedBlock::Iq2Xs(bytes)
        | QuantizedBlock::Iq3Xxs(bytes)
        | QuantizedBlock::Float16(bytes)
        | QuantizedBlock::BFloat16(bytes) => bytes.len(),
    }
}

fn prepare(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock<'_>],
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
) -> Result<Prepared, MetalError> {
    let shapes = infer(program, symbols)?;

    // ROW 327: `block_nodes[i]` is the ONLY node `blocks[i]` may be
    // attributed to -- this crate's positional contract, identical to
    // `proxima_tensor::cpu::evaluate`'s (see `execute`'s own doc). Every
    // classification below (`packed_operands_of`, the dtype gate, per-node
    // shape) reads off this ONE pairing; validating it here, before any of
    // them run, is what stops a caller's node/block order mismatch from
    // surfacing as an unrelated downstream rejection -- a Q6_K block bound
    // to the wrong node used to reach `reject_unsupported_gpu_dtype`
    // (`NotLowerable`, no mention of block order) instead of the precise,
    // node-carrying `InputCountMismatch`/`InputSizeMismatch` below. A
    // same-shape swap between two quantized nodes is still undetectable
    // from bytes alone -- callers with more than one packed input should
    // bind by name via [`plan_named`]/[`resolve_named_blocks`], which
    // resolves this pairing from `Op::name` instead of argument order.
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
    // every packed codec's declared dtype is the "these are bytes" marker
    // `reject_unsupported_gpu_dtype`'s own doc already claims as its
    // exemption's rationale -- not just the codecs `packed_operands` above
    // has a kernel for, so any future codec added to `QuantizedBlock` before
    // it has an unpack kernel here still gets the right dtype exemption
    // rather than an unrelated "not float" rejection. Reads
    // `packed_operands`'s own keys rather than re-zipping `block_nodes`
    // against `blocks` a second time, so this set can never disagree with
    // the codec table above on which nodes are packed.
    let packed_operand_nodes: BTreeSet<NodeId> = packed_operands.keys().copied().collect();
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

    let mut resolved = bind(program, &shapes, &effective_outputs, numeric_policy)?;
    #[cfg(feature = "instrument")]
    if effective_outputs.iter().any(|output| output.0 == 2) {
        if let Some(bound_embedding) = resolved.iter().find(|bound| bound.node.0 == 2) {
            if let Some((source, layout, gather)) = bound_embedding.operands().first() {
                debug!(
                    node = bound_embedding.node.0,
                    source = source.0,
                    layout_strides = ?layout.strides,
                    gather = ?gather,
                    packed_codec = ?packed_operands.get(source),
                    "prepared embedding gather layout"
                );
            }
        }
    }
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
        packed_operands,
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
        // `all_read_sources`, not `operands` -- a fused `Reduce`'s
        // `epilogue_operands` are real, materialized reads this op's own
        // dispatch needs bound (`crate::msl::bindings`'s own doc says the
        // same), so a candidate this loop promotes earlier must never land
        // ahead of an epilogue operand's own producer position, or that
        // producer's buffer has not been dispatched yet when this op reads
        // it. `operands()` alone missed exactly that class of dependency:
        // the MoE top-k combine's per-round softmax weight lives ONLY in
        // the final reduce's epilogue, so this promotion moved the combine
        // ahead of its own weight's dispatch and `buffer_for` failed at
        // execution time with "operand buffer missing".
        let dependency_index = resolved[current_index]
            .all_read_sources()
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
///
/// A non-empty `epilogue_broadcast_axes` (`bind::BoundOpKind::Reduce::
/// epilogue_broadcast_axes`'s own doc, `proxima_tensor::cpu`'s own
/// `node_output_len` mirrors this same widening) re-broadcasts the fold's
/// scalar back over the axes it reduced away, so the MATERIALIZED output is
/// the full `extents` product, not just `output_axes`'s smaller fold shape —
/// falls through to the same full-iteration-space arm every other kind
/// takes.
fn bound_output_len(bound: &BoundOp) -> usize {
    match &bound.kind {
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            output_axes,
            epilogue_broadcast_axes,
            ..
        } if epilogue_broadcast_axes.is_empty() => output_axes
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

/// [`push_i64_row`]'s `u64`-extents counterpart -- casts in place instead of
/// collecting `bound.extents` into a temporary `Vec<i64>` first, the
/// allocation ROW 303's residual named in [`pack_elementwise_uniforms`].
fn push_extent_row(bytes: &mut Vec<u8>, extents: &[u64], width: usize) {
    for slot in 0..width {
        let value = extents.get(slot).map(|extent| *extent as i64).unwrap_or(0);
        push_i64(bytes, value);
    }
}

/// [`push_extent_row`]'s gathered counterpart: `axes` indexes `extents` in
/// AXIS order (a reduce's `output_axes`/reduction axes are a subset of
/// `bound.extents`, not a contiguous prefix), so this reads `extents[axis]`
/// directly per slot instead of [`pack_reduce_uniforms`] collecting the
/// gather into a temporary `Vec<i64>` first (ROW 303's residual, this
/// landing's own removal). `width` zero-pads past `axes.len()`, same
/// contract as [`push_i64_row`].
fn push_gathered_extent_row(bytes: &mut Vec<u8>, extents: &[u64], axes: &[u16], width: usize) {
    for slot in 0..width {
        let value = axes
            .get(slot)
            .map(|axis| extents[*axis as usize] as i64)
            .unwrap_or(0);
        push_i64(bytes, value);
    }
}

/// The product [`push_gathered_extent_row`] would write, computed without
/// materializing the row -- both `output_extents.iter().product()` and
/// `reduction_extents.iter().product()` in [`pack_reduce_uniforms`] need
/// exactly this, ahead of the row itself.
fn gathered_extent_product(extents: &[u64], axes: &[u16]) -> i64 {
    axes.iter()
        .map(|axis| extents[*axis as usize] as i64)
        .product()
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
        BoundOpKind::CachedAttention {
            cached_key_rows, ..
        } => {
            // Mirrors `pack_cached_attention_uniforms`: the single-range
            // fused form (nine operands, `cached_key_rows == 0`) appends
            // `cached_key_rows`, `new_key_rows`, `context_chunks`, and
            // `splits` as runtime
            // uniform fields (`total_elements` plus these four = 5 words).
            // `two_range_cached_bound` (nine operands, `cached_key_rows !=
            // 0`) reads its runtime bound straight off `in8` in the kernel
            // body instead (`render_cached_attention`'s own doc) and keeps
            // the minimal one-word struct, so only the single-range form
            // widens the uniform blob.
            if operand_count == 9 && *cached_key_rows == 0 {
                5 * WORD
            } else {
                WORD
            }
        }
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

fn pack_uniforms(bound: &BoundOp, numeric_policy: NumericPolicy) -> Result<Vec<u8>, EmitError> {
    let mut bytes = Vec::new();
    pack_uniforms_into(bound, numeric_policy, &mut bytes)?;
    Ok(bytes)
}

/// [`pack_uniforms`], writing into caller-owned storage. Stable plans call
/// this once while initializing each per-position uniform buffer; the
/// unplaced path uses the owned wrapper above.
fn pack_uniforms_into(
    bound: &BoundOp,
    numeric_policy: NumericPolicy,
    scratch: &mut Vec<u8>,
) -> Result<(), EmitError> {
    scratch.clear();
    match &bound.kind {
        BoundOpKind::CachedAttention { .. } => {
            pack_cached_attention_uniforms(bound, numeric_policy, scratch)
        }
        BoundOpKind::Elementwise { .. } => {
            pack_elementwise_uniforms(bound, scratch);
            Ok(())
        }
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => pack_reduce_uniforms(bound, scratch),
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => pack_scan_uniforms(bound, scratch),
        BoundOpKind::Iota | BoundOpKind::Constant { .. } => {
            pack_leaf_uniforms(bound, scratch);
            Ok(())
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod numeric_policy_construction_tests {
    //! [`plan`]/[`plan_named`] now take `numeric_policy` as a required
    //! constructor argument and record it once, at bind time -- the fix for
    //! the two bugs this design closes (a conflated permission ladder, and
    //! a policy that never reached `bind`). These tests prove the round
    //! trip through [`numeric_policy_as_metal_math_mode`]/
    //! [`metal_math_mode_as_numeric_policy`], and that a [`Plan`] reports
    //! exactly the policy it was constructed under -- never a later
    //! default, and never silently narrowed by [`Plan::set_math_mode`].

    use alloc::vec;

    use proxima_tensor::{DType, Extent, IndexMap, NumericPolicy, Op, QuantizedBlock, append, map};

    use super::{MathMode, MetalError, metal_math_mode_as_numeric_policy, plan};
    #[cfg(feature = "instrument")]
    use super::{PIPELINE_MISSES, device_and_queue, resolve_steps};

    #[test]
    fn metal_math_mode_round_trips_through_numeric_policy() {
        for mode in [MathMode::Safe, MathMode::Relaxed, MathMode::Fast] {
            let policy = metal_math_mode_as_numeric_policy(mode);
            assert_eq!(
                super::numeric_policy_as_metal_math_mode(policy),
                mode,
                "mode {mode:?} must round-trip through its own exact permission set"
            );
        }
    }

    /// The fix for the two bugs ROW 4309 (this file) found: a
    /// contraction-only policy used to be rounded UP to `Relaxed` (which
    /// also grants `reassociation`), and a `signed_zero`-only policy up to
    /// `Fast` (which grants everything). Every one of the 32 permission
    /// subsets must instead map to the WIDEST mode whose own permission set
    /// ([`metal_math_mode_as_numeric_policy`]) is a SUBSET of what the
    /// policy actually grants ([`NumericPolicy::grants`]) -- never wider.
    #[test]
    fn numeric_policy_projection_never_grants_more_than_the_policy_allows() {
        for bits in 0u8..32 {
            let policy = NumericPolicy::bit_exact()
                .with_contraction(bits & 0b0000_0001 != 0)
                .with_reassociation(bits & 0b0000_0010 != 0)
                .with_nan_assumptions(bits & 0b0000_0100 != 0)
                .with_signed_zero(bits & 0b0000_1000 != 0)
                .with_approx_functions(bits & 0b0001_0000 != 0);
            let mode = super::numeric_policy_as_metal_math_mode(policy);
            let granted = metal_math_mode_as_numeric_policy(mode);
            assert!(
                policy.grants(granted),
                "subset {bits:05b} projected to {mode:?}, whose own permission set \
                 {granted:?} exceeds what the policy {policy:?} actually grants"
            );
        }
    }

    fn identity_program() -> (Vec<Op>, proxima_tensor::NodeId) {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(4)],
                name: None,
            },
        );
        let identity = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: proxima_tensor::ScalarOp::Identity,
                operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        (program, identity)
    }

    #[test]
    fn plan_records_the_policy_it_bound_under_not_a_later_default() {
        let (program, identity) = identity_program();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let blocks = [QuantizedBlock::Float32(&data)];
        let resolved_plan = plan(
            &program,
            &[],
            &blocks,
            &[identity],
            NumericPolicy::llama_relaxed(),
        )
        .expect("plans the identity program");
        assert_eq!(
            resolved_plan.numeric_policy(),
            NumericPolicy::llama_relaxed()
        );
        assert!(
            resolved_plan
                .check_numeric_policy(NumericPolicy::llama_relaxed())
                .is_ok()
        );
    }

    #[test]
    fn plan_refuses_a_mismatched_policy_instead_of_silently_narrowing() {
        let (program, identity) = identity_program();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let blocks = [QuantizedBlock::Float32(&data)];
        let resolved_plan = plan(
            &program,
            &[],
            &blocks,
            &[identity],
            NumericPolicy::bit_exact(),
        )
        .expect("plans the identity program");
        let error = resolved_plan
            .check_numeric_policy(NumericPolicy::fast())
            .expect_err("bit-exact-bound plan does not satisfy fast");
        let MetalError::NumericPolicyMismatch { bound, requested } = error else {
            panic!("expected NumericPolicyMismatch, got {error:?}");
        };
        assert_eq!(bound, NumericPolicy::bit_exact());
        assert_eq!(requested, NumericPolicy::fast());
    }

    #[test]
    fn set_math_mode_narrows_within_the_bound_policy_but_never_widens_it() {
        let (program, identity) = identity_program();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let blocks = [QuantizedBlock::Float32(&data)];
        let mut resolved_plan = plan(
            &program,
            &[],
            &blocks,
            &[identity],
            NumericPolicy::llama_relaxed(),
        )
        .expect("plans the identity program");
        resolved_plan
            .set_math_mode(MathMode::Safe)
            .expect("Safe needs nothing, llama_relaxed() grants it trivially");
        assert_eq!(resolved_plan.math_mode(), MathMode::Safe);
        assert_eq!(
            resolved_plan.numeric_policy(),
            NumericPolicy::llama_relaxed(),
            "narrowing math_mode must never change the bound numeric_policy"
        );
        let error = resolved_plan
            .set_math_mode(MathMode::Fast)
            .expect_err("Fast needs nan_assumptions/signed_zero/approx_functions, llama_relaxed() withholds all three");
        let MetalError::NumericPolicyMismatch { bound, requested } = error else {
            panic!("expected NumericPolicyMismatch, got {error:?}");
        };
        assert_eq!(bound, NumericPolicy::llama_relaxed());
        assert_eq!(requested, NumericPolicy::fast());
    }

    #[cfg(feature = "instrument")]
    #[test]
    fn narrowing_math_mode_forces_a_genuine_pipeline_cache_miss_not_a_stale_hit() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        let (program, identity) = identity_program();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let blocks = [QuantizedBlock::Float32(&data)];
        let mut resolved_plan = plan(
            &program,
            &[],
            &blocks,
            &[identity],
            NumericPolicy::llama_relaxed(),
        )
        .expect("plans the identity program");
        assert_eq!(
            resolved_plan.math_mode(),
            MathMode::Relaxed,
            "llama_relaxed() constructs Relaxed, not the Fast that the old \
             widest-mode-that-covers-any-permission bug would have produced"
        );

        let _ = PIPELINE_MISSES.snapshot_and_reset();
        resolve_steps(&device, &resolved_plan).expect("first resolution compiles under Relaxed");
        let first_misses = PIPELINE_MISSES.snapshot_and_reset();
        assert_eq!(
            first_misses, 1,
            "the first resolution of a fresh plan is always a miss"
        );
        let relaxed_key = resolved_plan
            .kernel_keys()
            .expect("kernel_keys reads the just-resolved identity")[0]
            .clone();

        resolved_plan
            .set_math_mode(MathMode::Safe)
            .expect("Safe needs nothing, llama_relaxed() grants it trivially");
        resolve_steps(&device, &resolved_plan).expect("second resolution compiles under Safe");
        let second_misses = PIPELINE_MISSES.snapshot_and_reset();
        assert_eq!(
            second_misses, 1,
            "set_math_mode(Safe) must force a genuine PIPELINE_CACHE miss, never reuse the \
             Relaxed-compiled pipeline cached under the SAME numeric_policy token"
        );
        let safe_key = resolved_plan
            .kernel_keys()
            .expect("kernel_keys after narrowing")[0]
            .clone();
        assert_ne!(
            relaxed_key, safe_key,
            "Relaxed's and Safe's cache keys must differ by the math-mode token even though \
             numeric_policy (and so emit's rendered source) never changed"
        );

        let resolved_math_mode = resolved_plan
            .resolved_steps
            .borrow()
            .as_ref()
            .expect("resolve_steps populated resolved_steps")
            .math_mode;
        assert_eq!(
            resolved_math_mode,
            MathMode::Safe,
            "resolved_steps must record the mode it just compiled under, not the plan's earlier one"
        );
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
        BoundOp, BoundOpKind, DType, Extent, IndexMap, Keep, NumericPolicy, Op, Reduce, ReduceInit,
        ScalarOp, append, bind, infer, map,
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
        bind(&program, &shapes, &[], NumericPolicy::default())
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
        bind(&program, &shapes, &[], NumericPolicy::default())
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
        assert_eq!(
            pack_uniforms_byte_len(&bound),
            pack_uniforms(&bound, NumericPolicy::default())
                .expect("packs uniforms")
                .len()
        );
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
        assert_eq!(
            pack_uniforms_byte_len(&bound),
            pack_uniforms(&bound, NumericPolicy::default())
                .expect("packs uniforms")
                .len()
        );
    }
}

fn pack_cached_attention_uniforms(
    bound: &BoundOp,
    numeric_policy: NumericPolicy,
    bytes: &mut Vec<u8>,
) -> Result<(), EmitError> {
    let BoundOpKind::CachedAttention {
        head_dim,
        cached_key_rows,
        new_key_rows,
        ..
    } = &bound.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: bound.node,
            expected: "cached_attention",
            found: bound.kind.name(),
        });
    };
    // Same `cached_key_rows == 0` discriminator as `crate::msl::render_
    // cached_attention` / `grid_threads` -- `two_range_cached_bound` (nine
    // operands, `cached_key_rows != 0`) reads its live bound off `in8`
    // inside the kernel body instead of widening the dispatch or the
    // uniforms struct, so it takes the same path as the eight-operand,
    // unbucketed case below.
    let dynamic_cached_len = bound.operands().len() == 9 && *cached_key_rows == 0;
    let context_length = *cached_key_rows + *new_key_rows;
    let chunks = crate::msl::context_chunks_for(context_length, numeric_policy) as i64;
    let splits = crate::msl::splits_for(context_length, numeric_policy) as i64;
    // Redesign §5 option 2: the single-range fused path always dispatches
    // the compiled MAXIMUM chunk count (`cap`) -- `grid_threads`'s own
    // `CachedAttention` arm (`omega/src/msl.rs`) -- so `total_elements` here
    // must agree with that grid width, never the live `chunks` value, which
    // travels separately as its own `Uniforms` field below. Redesign §4c:
    // the dispatch is ALSO widened by the compiled `ATTENTION_SPLIT_MAX`,
    // but only when `cached_attention_merge_needed` holds for this op's own
    // context length -- `grid_threads`'s own doc explains why a split, unlike
    // a chunk, cannot be widened unconditionally without racing the real
    // dispatch's output write; this must stay in lock-step with that gate.
    let dispatch_chunks = if dynamic_cached_len {
        i64::try_from(crate::sized::ATTENTION_CONTEXT_CHUNK_CAP).unwrap_or(chunks)
    } else {
        chunks
    };
    let dispatch_splits = if dynamic_cached_len
        && crate::msl::cached_attention_merge_needed(&bound.kind, numeric_policy)
    {
        i64::try_from(crate::sized::ATTENTION_SPLIT_MAX).unwrap_or(splits)
    } else {
        1
    };
    let total: i64 = bound
        .extents
        .iter()
        .map(|extent| *extent as i64)
        .product::<i64>()
        / *head_dim as i64
        * dispatch_chunks
        * dispatch_splits;
    push_i64(bytes, total);
    // Mirrors `render_cached_attention`'s `struct Uniforms` (`omega/src/msl.rs`):
    // the single-range fused form (nine operands) declares three extra
    // `long` fields so the kernel body reads the live row counts AND the
    // live chunk count off the uniform buffer instead of baking either as
    // `constexpr` at render time -- the row counts per ROW 369, the chunk
    // count per redesign §5 option 2, both keyed off the SAME `bound.kind`
    // this function already reads per dispatch, so a `kv-capacity-bucket`
    // crossing (a new `BoundOp` with new `cached_key_rows`/`new_key_rows`)
    // repacks fresh uniform values without recompiling the kernel.
    if dynamic_cached_len {
        push_i64(bytes, *cached_key_rows as i64);
        push_i64(bytes, *new_key_rows as i64);
        push_i64(bytes, chunks);
        // Redesign §4c: the cross-THREADGROUP sibling of `chunks` above, one
        // hardware level up -- see `crate::msl::splits_for`'s own doc. The
        // LIVE value (never the compiled maximum `dispatch_splits` widens the
        // grid by above), since this is what the kernel body's own
        // `slice_len`/`lo`/`hi` computation reads. `tgid`'s own decode into
        // `query_row`/`split` divides out `dispatch_splits`, NOT this field --
        // that grid-widening constant, not the live count, is what an idle
        // threadgroup beyond it was actually enumerated against. ROW 385:
        // when no merge dispatch exists, this MUST be `1`, never `splits_for`'s
        // own raw result -- the kernel body's `slice_len = ceil(live/splits)`
        // would otherwise slice off keys with no second dispatch left to
        // merge the slices back, an out-of-bounds-shaped undercount.
        let live_splits = if crate::msl::cached_attention_merge_needed(&bound.kind, numeric_policy)
        {
            splits
        } else {
            1
        };
        push_i64(bytes, live_splits);
    }
    Ok(())
}

/// Mirrors `crate::msl::render_cached_attention_merge`'s own `Uniforms`
/// struct: `total_elements` (the SAME `query_rows * heads` count
/// [`pack_cached_attention_uniforms`]'s own `total` computes before its
/// `dispatch_chunks` multiply -- the merge kernel dispatches one simdgroup
/// per output row, not per `(row, chunk)` pair) plus `splits`, the live
/// count the merge's own online-softmax combine reduces over -- the SAME
/// `crate::msl::splits_for` call [`pack_cached_attention_uniforms`] already
/// makes for the split kernel's own `Uniforms.splits`, so the two dispatches
/// can never disagree on how many partials the split kernel wrote and the
/// merge kernel reads back. Uploaded fresh every call (`upload_uniforms`,
/// not a `Plan`-owned buffer like the split kernel's own `plan_uniform`) --
/// this slice's own scope boundary, named in `encode_op`'s call site;
/// folding it into `PlanUniforms` is follow-up work, not a correctness gap.
fn pack_cached_attention_merge_uniforms(
    bound: &BoundOp,
    numeric_policy: NumericPolicy,
) -> Result<Vec<u8>, EmitError> {
    let BoundOpKind::CachedAttention {
        head_dim,
        cached_key_rows,
        new_key_rows,
        ..
    } = &bound.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: bound.node,
            expected: "cached_attention_merge",
            found: bound.kind.name(),
        });
    };
    let total: i64 = bound
        .extents
        .iter()
        .map(|extent| *extent as i64)
        .product::<i64>()
        / *head_dim as i64;
    let splits = crate::msl::splits_for(*cached_key_rows + *new_key_rows, numeric_policy) as i64;
    let mut bytes = Vec::with_capacity(16);
    push_i64(&mut bytes, total);
    push_i64(&mut bytes, splits);
    Ok(bytes)
}

/// Mirrors the `Uniforms` struct `crate::msl::render_iota` and
/// `crate::msl::render_constant` both declare: just `total_elements` —
/// neither leaf has operands, a per-axis extents array, or a gather, so
/// there is nothing else this struct needs to carry. `render_constant`
/// bakes its literal into the source instead of adding a field here, which
/// is what lets one packer serve both.
fn pack_leaf_uniforms(bound: &BoundOp, bytes: &mut Vec<u8>) {
    let total: i64 = bound.extents.iter().map(|extent| *extent as i64).product();
    push_i64(bytes, total);
}

/// Mirrors the `Uniforms` struct `crate::msl::render_elementwise` declares
/// at `omega/src/msl.rs:328-335`: `total_elements`, `extents[rank_len]`,
/// `operand_base[operand_count]`, `operand_strides[operand_count][rank_len]`,
/// then — only when `bound` has a gathered operand — the four
/// `push_gather_uniform_fields` arrays [`push_gather_uniforms`] appends, in
/// that order — every field `long`, so a flat `i64` concatenation is the
/// struct's byte layout.
fn pack_elementwise_uniforms(bound: &BoundOp, bytes: &mut Vec<u8>) {
    let rank_len = bound.extents.len().max(1);

    push_i64(
        bytes,
        bound.extents.iter().map(|extent| *extent as i64).product(),
    );
    push_extent_row(bytes, &bound.extents, rank_len);
    for (_, layout, _) in bound.operands() {
        push_i64(bytes, layout.base);
    }
    for (_, layout, _) in bound.operands() {
        push_i64_row(bytes, &layout.strides, rank_len);
    }
    push_gather_uniforms(bytes, bound, rank_len);
}

/// Mirrors the `Uniforms` struct `crate::msl::render_reduce` declares at
/// `omega/src/msl.rs:386-397`: `output_total`, `reduction_total`,
/// `output_extents[output_rank_len]`, `reduction_extents[reduce_rank_len]`,
/// `operand_base[operand_count]`,
/// `operand_strides[operand_count][rank_len]`, `out_base`,
/// `out_strides[rank_len]`, then the gather arrays (see
/// [`pack_elementwise_uniforms`]'s doc), in that order.
fn pack_reduce_uniforms(bound: &BoundOp, bytes: &mut Vec<u8>) -> Result<(), EmitError> {
    let BoundOpKind::Reduce {
        output_axes,
        out_layout,
        epilogue_operands,
        epilogue_broadcast_axes,
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
    // `crate::msl::render_reduce`'s own doc: a non-empty `epilogue_broadcast_
    // axes` widens `epilogue_operand_strides` from `output_rank_len` to the
    // full `rank_len` (`x`/`gamma` read at the whole `(s, d)` coordinate, not
    // just `s`), and needs a SEPARATE `broadcast_out_strides` row -- the
    // CONTIGUOUS row-major layout the wider materialized output buffer is
    // allocated with, never `out_layout.strides` (kept as-is above: that
    // stays the fold's own COMPACT addressing, stride `0` on every broadcast
    // axis, still needed to locate the fold's scalar).
    let is_broadcast_epilogue = !epilogue_broadcast_axes.is_empty();
    let epilogue_stride_rank_len = if is_broadcast_epilogue {
        rank_len
    } else {
        output_rank_len
    };

    // ROW 303 residual, removed by this landing: `output_extents`/
    // `reduction_extents` used to be temporary `Vec<i64>`s built by
    // gathering `bound.extents` through `output_axes`/`reduce_axes` --
    // [`push_gathered_extent_row`]/[`gathered_extent_product`] read that
    // same gather directly into `bytes` (or fold it into a product) without
    // ever materializing the intermediate row.
    push_i64(bytes, gathered_extent_product(&bound.extents, output_axes));
    push_i64(bytes, gathered_extent_product(&bound.extents, &reduce_axes));
    push_gathered_extent_row(bytes, &bound.extents, output_axes, output_rank_len);
    push_gathered_extent_row(bytes, &bound.extents, &reduce_axes, reduce_rank_len);
    for (_, layout, _) in bound.operands() {
        push_i64(bytes, layout.base);
    }
    for (_, layout, _) in bound.operands() {
        push_i64_row(bytes, &layout.strides, rank_len);
    }
    push_i64(bytes, out_layout.base);
    push_i64_row(bytes, &out_layout.strides, rank_len);
    // `crate::msl::render_reduce`'s own `Uniforms` struct declares these
    // fields ONLY when `epilogue_operands` is non-empty (byte-identical to
    // before epilogue fusion existed otherwise), so this must stay
    // conditional on the exact same test.
    if !epilogue_operands.is_empty() {
        for (_, layout, _) in epilogue_operands {
            push_i64(bytes, layout.base);
        }
        for (_, layout, _) in epilogue_operands {
            push_i64_row(bytes, &layout.strides, epilogue_stride_rank_len);
        }
    }
    if is_broadcast_epilogue {
        push_i64_row(bytes, &contiguous_strides(&bound.extents), rank_len);
    }
    push_gather_uniforms(bytes, bound, rank_len);
    Ok(())
}

/// The row-major (C-order) strides the FULL `extents` product buffer a
/// broadcast-reduce epilogue materializes is allocated with -- a genuinely
/// different address space from any [`Layout`]'s own strides (which may
/// carry a stride of `0` on a broadcast axis), computed fresh here rather
/// than read off any bound operand.
fn contiguous_strides(extents: &[u64]) -> Vec<i64> {
    let mut strides = vec![0i64; extents.len()];
    let mut running = 1i64;
    for (axis, extent) in extents.iter().enumerate().rev() {
        strides[axis] = running;
        running *= *extent as i64;
    }
    strides
}

/// Mirrors the `Uniforms` struct `crate::msl::render_scan` declares at
/// `omega/src/msl.rs:493-503`: `outer_total`, `inner_len`,
/// `outer_extents[outer_rank_len]`, `operand_base[operand_count]`,
/// `operand_strides[operand_count][rank_len]`, `out_base`,
/// `out_strides[rank_len]`, then the gather arrays (see
/// [`pack_elementwise_uniforms`]'s doc), in that order. `crate::msl::validate`
/// already rejected a rank-0 scan before `emit` (and therefore this) ever
/// runs, so `bound.extents` is never empty here.
fn pack_scan_uniforms(bound: &BoundOp, bytes: &mut Vec<u8>) -> Result<(), EmitError> {
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

    let outer_extents = &bound.extents[..outer_rank];
    let inner_len = bound.extents.last().copied().unwrap_or(1) as i64;

    push_i64(
        bytes,
        outer_extents.iter().map(|extent| *extent as i64).product(),
    );
    push_i64(bytes, inner_len);
    push_extent_row(bytes, outer_extents, outer_rank_len);
    for (_, layout, _) in bound.operands() {
        push_i64(bytes, layout.base);
    }
    for (_, layout, _) in bound.operands() {
        push_i64_row(bytes, &layout.strides, rank_len);
    }
    push_i64(bytes, out_layout.base);
    push_i64_row(bytes, &out_layout.strides, rank_len);
    push_gather_uniforms(bytes, bound, rank_len);
    Ok(())
}

fn nserror_description(error: &NSError) -> String {
    error.localizedDescription().to_string()
}

/// [`MTLCompileOptions::mathMode`], narrowed to the three values this
/// crate's kernels compile with. ROW 296 (`proxima-tensor/docs/discipline.md`)
/// measured `Fast` identical to `Relaxed` on that one kernel
/// (240.9-247.3 GB/s vs. 179.2 GB/s for `Safe`, same 0-1.9e-6 parity drift)
/// and dropped it as a knob nothing selected; ROW 338's best cell (nsg=4 +
/// fast) reopened the question on a different kernel, so `Fast` is back as a
/// selectable runtime value rather than a recompile.
///
/// A [`Plan`] carries one of these ([`Plan::set_math_mode`]); it feeds
/// `compile_pipeline` and folds into `pipeline_for`'s cache key so a
/// `Safe`-compiled kernel is never handed to a caller that asked for
/// `Relaxed` or `Fast`, or the reverse.
///
/// `objc2_metal::MTLMathMode`'s own doc (`MTLLibrary.rs`, upstreaming
/// [Apple's `MTLMathMode`](https://developer.apple.com/documentation/metal/mtlmathmode))
/// states the three rungs in these exact words: `Safe` "disables unsafe
/// floating-point optimizations"; `Relaxed` "allows aggressive, unsafe
/// floating-point optimizations but preserves infs and nans"; `Fast`
/// "allows aggressive, unsafe floating-point optimizations" with no such
/// preservation. `Relaxed`'s wording -- aggressive optimization, NaN/inf
/// preserved -- is exactly [`NumericPolicy::llama_relaxed`]: contraction and
/// reassociation are permitted, `nan_assumptions`/`signed_zero`/
/// `approx_functions` are withheld because Relaxed's own contract
/// explicitly preserves NaN/inf/zero behavior. See
/// `numeric_policy_as_metal_math_mode` and
/// `metal_math_mode_as_numeric_policy` for the full projection both
/// directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MathMode {
    /// IEEE-safe float math -- bit-parity with
    /// [`proxima_tensor::cpu::evaluate`], at 179.2 GB/s on the ROW 296
    /// packed-row Q4_K matvec kernel.
    Safe,
    /// Metal's relaxed-math kernels. ROW 296/297's measured default:
    /// 240.9-247.3 GB/s on the ROW 296 kernel (1.34x `Safe`), 28.19 vs.
    /// 33.82 ms/token steady-state on the whole ROW 297 decode program
    /// (1.20x), with generated text and quality metrics identical to
    /// `Safe`'s in both rows.
    #[default]
    Relaxed,
    /// Metal's fast-math kernels. Measured indistinguishable from
    /// `Relaxed` on the ROW 296 kernel; ROW 338's best cell (nsg=4 + fast)
    /// found a different kernel where it was not, so it is exposed here as
    /// a runtime choice rather than assumed identical everywhere.
    Fast,
}

impl MathMode {
    const fn as_mtl(self) -> MTLMathMode {
        match self {
            MathMode::Safe => MTLMathMode::Safe,
            MathMode::Relaxed => MTLMathMode::Relaxed,
            MathMode::Fast => MTLMathMode::Fast,
        }
    }

    /// One character folded onto [`kernel_cache_key`]'s own identity string
    /// by every `PIPELINE_CACHE` call site ([`resolve_steps`],
    /// [`encode_op`], [`Plan::kernel_keys`]) -- NOT a duplicate of
    /// `identity::MetalOnlyExtras::numeric_policy_token`. That token is
    /// strictly finer for the axes `numeric_policy` fixes for a `Plan`'s
    /// whole life (chunking, contraction, reassociation); it does NOT track
    /// [`Plan::set_math_mode`], which narrows `compile_pipeline`'s
    /// `MTLCompileOptions.mathMode` for an UNCHANGED `numeric_policy`. Two
    /// resolutions of the same `Plan` that only differ by a `set_math_mode`
    /// call render byte-identical MSL source (`compile_pipeline` never
    /// touches source text) but must never share a `PIPELINE_CACHE` entry,
    /// since they were compiled with different `mathMode` compile options --
    /// this token is what keeps them apart.
    const fn cache_token(self) -> char {
        match self {
            MathMode::Safe => 'S',
            MathMode::Relaxed => 'R',
            MathMode::Fast => 'F',
        }
    }
}

/// `MTLCompileOptions.mathMode` only distinguishes 3 rungs -- a compiler
/// flag governing how the SAME algebra compiles. [`NumericPolicy`] governs a
/// richer, orthogonal question: which algebra `bind`/the emitter are
/// permitted to choose in the first place (chunk count, contraction,
/// reduction order). [`MathMode`] is this narrower projection, not a
/// duplicate ladder. This selects the WIDEST mode whose own permission set
/// ([`metal_math_mode_as_numeric_policy`]) is a SUBSET of `policy` --
/// [`NumericPolicy::grants`] is the subset check, the same one
/// [`Plan::set_math_mode`] already uses to refuse a widening request. It
/// never rounds up past what `policy` actually grants: a `contraction`-only
/// policy does not grant `llama_relaxed()` (which also needs
/// `reassociation`), so it stays `Safe`; only a policy granting BOTH
/// `contraction` and `reassociation` gets `Relaxed`, and only one granting
/// every permission [`NumericPolicy::fast`] names gets `Fast`. See
/// [`metal_math_mode_as_numeric_policy`] for the inverse.
///
/// A free function, not an inherent `impl NumericPolicy` -- `NumericPolicy`
/// is defined in `proxima-tensor`, and the orphan rule forbids an inherent
/// `impl` for a foreign type from this crate.
#[must_use]
const fn numeric_policy_as_metal_math_mode(policy: NumericPolicy) -> MathMode {
    if policy.grants(NumericPolicy::fast()) {
        MathMode::Fast
    } else if policy.grants(NumericPolicy::llama_relaxed()) {
        MathMode::Relaxed
    } else {
        MathMode::Safe
    }
}

/// The inverse of [`numeric_policy_as_metal_math_mode`]: the exact
/// permission set Apple's own `MTLMathMode` doc commits to for `mode` --
/// `Safe` grants nothing ([`NumericPolicy::bit_exact`]), `Relaxed` grants
/// contraction+reassociation only, NaN/inf/zero preserved per the header
/// ([`NumericPolicy::llama_relaxed`]), `Fast` grants everything
/// ([`NumericPolicy::fast`]). Loses nothing: it round-trips through
/// [`numeric_policy_as_metal_math_mode`] for all three modes. What the
/// OTHER direction loses: every point in the 32-state permission space that
/// isn't one of these 3 Metal natively supports (e.g. `contraction` alone
/// without `reassociation` compiles identically to both granted, since
/// Metal has no finer compiler flag).
#[must_use]
const fn metal_math_mode_as_numeric_policy(mode: MathMode) -> NumericPolicy {
    match mode {
        MathMode::Safe => NumericPolicy::bit_exact(),
        MathMode::Relaxed => NumericPolicy::llama_relaxed(),
        MathMode::Fast => NumericPolicy::fast(),
    }
}

/// [`MTLComputeCommandEncoder`]'s dispatch-scheduling mode, narrowed to the
/// two values [`objc2_metal::MTLDispatchType`] exposes to a compute encoder.
/// A [`Plan`] carries one of these ([`Plan::set_dispatch_type`]); it decides
/// which encoder [`execute_plan_with_placements`] opens and whether its loop
/// runs `HazardTracker` at all.
///
/// ROW 311 (`proxima-tensor/docs/discipline.md`): llama.cpp encodes its whole
/// token on a serial compute encoder with zero explicit barriers.
/// `Concurrent` (this type's default) lets independent dispatches overlap
/// instead of draining the pipeline between every op, at the cost of the 323
/// per-token [`MTLBarrierScope::Buffers`] barriers `HazardTracker` inserts
/// to keep that overlap correct; `Serial` orders every dispatch for free and
/// emits none. See ROW 312 for the measured wall-clock comparison between
/// the two on the decode program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DispatchType {
    /// One dispatch completes before the next begins -- llama.cpp's own
    /// encoding shape, and the same guarantee an unmodified
    /// `computeCommandEncoder()` gives. No barrier is ever needed.
    Serial,
    /// Independent dispatches may overlap; `HazardTracker` inserts a
    /// [`MTLBarrierScope::Buffers`] barrier wherever a RAW/WAW/WAR hazard
    /// would otherwise let two overlapping dispatches race.
    #[default]
    Concurrent,
}

impl DispatchType {
    const fn as_mtl(self) -> MTLDispatchType {
        match self {
            DispatchType::Serial => MTLDispatchType::Serial,
            DispatchType::Concurrent => MTLDispatchType::Concurrent,
        }
    }
}

fn compile_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    kernel: &Kernel,
    math_mode: MathMode,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
    let options = MTLCompileOptions::new();
    // ROW 296 (`proxima-tensor/docs/discipline.md`): `Safe` and `Relaxed`
    // stream 179.2 vs. 240.9-247.3 GB/s with identical parity in every
    // shape-sweep cell, so the mode is a caller choice (`Plan::set_math_mode`),
    // not a fixed compile option.
    options.setMathMode(math_mode.as_mtl());

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
    math_mode: MathMode,
    numeric_policy: NumericPolicy,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
    // `cache_key` already carries BOTH axes `compile_pipeline` reads: the
    // numeric-policy token from `kernel_cache_key`
    // (`crate::identity::kernel_identity`, via
    // `MetalOnlyExtras::numeric_policy_token`) plus `math_mode`'s own
    // `MathMode::cache_token`, appended by every caller of this function
    // (`resolve_steps`, `encode_op`) before it gets here. Two BoundOps
    // agreeing on everything else but compiled under different policies OR
    // different math modes still never share a pipeline: `Safe`'s kernel
    // body is byte-identical to `Relaxed`'s (`compile_pipeline` never
    // touches source text, only `MTLCompileOptions`), so the key's tokens
    // are what keep them apart. The numeric-policy token alone is not
    // enough: `Plan::set_math_mode` narrows the compiled mode for an
    // UNCHANGED `numeric_policy`, so a `math_mode` token distinct from the
    // policy token is required too (see `MathMode::cache_token`'s own doc).
    if let Some(pipeline) = PIPELINE_CACHE.with(|cache| cache.borrow().get(cache_key).cloned()) {
        trace!(cache_key = %cache_key, hit = true, "pipeline cache lookup");
        #[cfg(feature = "instrument")]
        counter!(PIPELINE_HITS, 1);
        return Ok(pipeline);
    }
    trace!(cache_key = %cache_key, hit = false, "pipeline cache lookup");
    // fires only on a pipeline-cache MISS, before `compile_pipeline` runs --
    // evidence of the lowering choice this cache-key string encodes, never
    // of a later dispatch: a cache hit on the same key emits nothing here.
    #[cfg(feature = "instrument")]
    if let BoundOpKind::CachedAttention {
        cached_key_rows,
        new_key_rows,
        ..
    } = &bound.kind
    {
        let context_length = cached_key_rows + new_key_rows;
        debug!(
            context_length,
            context_chunks = crate::msl::context_chunks_for(context_length, numeric_policy),
            numeric_policy = ?numeric_policy,
            kernel_identity = %cache_key,
            "lowering selected attention chunking"
        );
    }
    #[cfg(feature = "instrument")]
    let compile_started = read_ticks();
    let kernel = emit(bound, packed_operands, numeric_policy)?;
    if std::env::var_os("PROXIMA_DEBUG_METAL_SOURCE").is_some()
        && std::env::var("PROXIMA_METAL_COMPARE_BOUND_NODE").ok() == Some(bound.node.0.to_string())
    {
        eprintln!(
            "metal_bound_source node={:?}\n{}",
            bound.node, kernel.source
        );
    }
    let pipeline = compile_pipeline(device, &kernel, math_mode)?;
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

/// [`pipeline_for`]'s counterpart for a [`Kernel`] already in hand (the
/// `CachedAttention` merge dispatch's own `emit_cached_attention_merge`
/// output) instead of one this function must `emit` itself from a `bound` --
/// its own pipeline-cache entry, keyed on `cache_key` (the split kernel's
/// own key plus a `_merge` suffix at the call site), so a merge kernel never
/// shares a compiled `MTLComputePipelineState` with its split sibling even
/// though both come from the SAME `BoundOp` position.
fn pipeline_for_kernel(
    device: &ProtocolObject<dyn MTLDevice>,
    kernel: &Kernel,
    cache_key: &str,
    math_mode: MathMode,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
    if let Some(pipeline) = PIPELINE_CACHE.with(|cache| cache.borrow().get(cache_key).cloned()) {
        return Ok(pipeline);
    }
    let pipeline = compile_pipeline(device, kernel, math_mode)?;
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
    counter!(OUTPUT_BUFFER_ALLOCATED_BYTES, byte_length as u64);
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
    counter!(OUTPUT_BUFFER_ALLOCATED_BYTES, bucket as u64);
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
    /// [`crate::sized::OUTPUT_POOL_MAX_PER_BUCKET`] caps retained-buffer growth per bucket
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
pub static EXPERT_SOURCE_STAGE_CALLS: Counter =
    Counter::new("omega.metal.expert_source_stage_calls");
#[cfg(feature = "instrument")]
pub static EXPERT_SOURCE_STAGE_BYTES: Counter =
    Counter::new("omega.metal.expert_source_stage_bytes");
#[cfg(feature = "instrument")]
pub static EXPERT_SOURCE_STAGE_TICKS: Counter =
    Counter::new("omega.metal.expert_source_stage_ticks");
#[cfg(feature = "instrument")]
pub static EXPERT_SOURCE_CACHE_HITS: Counter = Counter::new("omega.metal.expert_source_cache_hits");
#[cfg(feature = "instrument")]
pub static EXPERT_SOURCE_CACHE_MISSES: Counter =
    Counter::new("omega.metal.expert_source_cache_misses");
#[cfg(feature = "instrument")]
pub static EXPERT_SOURCE_BUFFER_REUSES: Counter =
    Counter::new("omega.metal.expert_source_buffer_reuses");
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
    /// block_offset_bound_bytes == block_offered_bytes` on every step EXCEPT
    /// one where a resident-copy cache hit served bytes for free -- see
    /// [`BLOCK_COPIED_BYTES`]'s own doc for why that hit contributes to
    /// neither term.
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
    /// Device bytes requested by those fresh output-buffer allocations.
    pub output_buffer_allocated_bytes: u64,
    /// [`PLAN_UNIFORM_WRITES`]'s own per-step delta -- `op_count` on the
    /// first execution that initializes a stable plan's uniforms and zero
    /// on every warm execution of that plan.
    pub plan_uniform_writes: u64,
    /// [`BARRIERS_EMITTED`]'s own per-step delta -- 0 on
    /// [`DispatchType::Serial`], the count of dataflow hazards the private
    /// `HazardTracker` actually found on [`DispatchType::Concurrent`].
    pub barriers_emitted: u64,
    pub expert_source_cache_hits: u64,
    pub expert_source_cache_misses: u64,
    pub expert_source_buffer_reuses: u64,
    pub plan_handoff_reuses: u64,
    pub expert_source_cache_entries: u64,
    pub nocopy_cache_entries: u64,
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
        output_buffer_allocated_bytes: OUTPUT_BUFFER_ALLOCATED_BYTES.snapshot_and_reset(),
        plan_uniform_writes: PLAN_UNIFORM_WRITES.snapshot_and_reset(),
        barriers_emitted: BARRIERS_EMITTED.snapshot_and_reset(),
        expert_source_cache_hits: EXPERT_SOURCE_CACHE_HITS.snapshot_and_reset(),
        expert_source_cache_misses: EXPERT_SOURCE_CACHE_MISSES.snapshot_and_reset(),
        expert_source_buffer_reuses: EXPERT_SOURCE_BUFFER_REUSES.snapshot_and_reset(),
        plan_handoff_reuses: PLAN_HANDOFF_REUSES.snapshot_and_reset(),
        expert_source_cache_entries: EXPERT_SOURCE_CACHE.with(|cache| cache.borrow().len() as u64),
        nocopy_cache_entries: NOCOPY_BUFFERS.with(|cache| cache.borrow().len() as u64),
    }
}

/// How many real `upload_block` calls took each host->device path —
/// incremented once per call, never per byte, so a caller can read back the
/// no-copy hit rate after a run without an external profiler. See the
/// module doc's "Host buffer upload" section.
pub static NOCOPY_BUFFER_UPLOADS: Counter = Counter::new("omega.metal.upload_block.nocopy");
pub static COPYING_BUFFER_UPLOADS: Counter = Counter::new("omega.metal.upload_block.copy");

/// Bytes bound through a real host->device copy this call
/// (`upload_block_copy`, or `upload_resident_copy` on a cache MISS only) --
/// fired at the point the bytes actually move, never before the lookup that
/// can avoid moving them. `upload_resident_copy`'s own cache HIT does not
/// increment this counter (no bytes moved, the same buffer is reused), so
/// `BLOCK_COPIED_BYTES + BLOCK_NOCOPY_BOUND_BYTES + BLOCK_OFFSET_BOUND_BYTES`
/// sums to `BLOCK_OFFERED_BYTES` (feature-gated behind `instrument`) minus
/// whatever a resident-copy cache hit
/// served for free that step. See the module doc's "Host buffer upload"
/// section for why most weight bytes never reach this counter -- only a
/// misaligned, non-resident block (the KV cache, which is deliberately never
/// cached: see `upload_block_copy`'s own doc) pays a real copy every token.
pub static BLOCK_COPIED_BYTES: Counter = Counter::new("omega.metal.block_copied_bytes");
/// Bytes bound zero-copy, either `upload_block_no_copy` (cached, resident)
/// or `upload_block_no_copy_uncached` (uncached) — the other terminal-path
/// byte split of `BLOCK_OFFERED_BYTES` (feature-gated behind `instrument`).
pub static BLOCK_NOCOPY_BOUND_BYTES: Counter = Counter::new("omega.metal.block_nocopy_bound_bytes");
/// Bytes bound at an offset into the single whole-checkpoint no-copy buffer
/// (`checkpoint_mapping_offset`) — the third terminal-path byte split of
/// `BLOCK_OFFERED_BYTES` (feature-gated behind `instrument`), and the one that carries the bulk of a real
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
    block_name: Option<&str>,
    resident_name: Option<&str>,
) -> Result<(MetalBuffer, usize), MetalError> {
    if std::env::var_os("PROXIMA_DEBUG_BLOCK_UPLOADS").is_some() && !data.is_empty() {
        eprintln!(
            "metal block upload node={node:?} name={block_name:?} dtype={dtype:?} bytes={} resident={resident_name:?}",
            size_of_val(data),
        );
    }
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
        | DType::UInt32 => upload_block_as_float(device, data, resident_name),
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
/// CACHING the wrapper this creates is gated on `resident_name`, not on
/// `is_page_aligned` alone: page alignment is a property of an ADDRESS, not
/// of a LIFETIME, and `(pointer, byte_length)` is exactly the key an
/// ephemeral, growing buffer (a KV-cache row, say) can reuse after a
/// realloc moves a DIFFERENT allocation onto the same range. Only
/// `mark_resident`'s own NAME proof licenses remembering a buffer past this
/// one call -- see [`upload_resident_copy`]'s own doc for why that cache
/// keys on the name itself, never the address.
fn upload_block_as_float(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[f32],
    resident_name: Option<&str>,
) -> Result<(MetalBuffer, usize), MetalError> {
    if data.is_empty() {
        return allocate_buffer(device, 0, DType::Float32).map(|buffer| (buffer, 0));
    }
    let byte_length = size_of_val(data);
    let pointer = data.as_ptr().cast::<c_void>();
    if is_page_aligned(pointer, byte_length) {
        counter!(NOCOPY_BUFFER_UPLOADS, 1);
        counter!(BLOCK_NOCOPY_BOUND_BYTES, byte_length as u64);
        if let Some(name) = resident_name {
            return upload_block_no_copy(device, name, pointer, byte_length)
                .map(|buffer| (buffer, 0));
        }
        return upload_block_no_copy_uncached(device, pointer, byte_length)
            .map(|buffer| (buffer, 0));
    }
    if let Some(result) = checkpoint_mapping_offset(device, pointer, byte_length) {
        return result;
    }
    if let Some(result) = expert_mapping_offset(device, pointer, byte_length) {
        return result;
    }
    if let Some(name) = resident_name {
        return upload_resident_copy(device, name, pointer, byte_length).map(|buffer| (buffer, 0));
    }
    counter!(BLOCK_COPIED_BYTES, byte_length as u64);
    counter!(COPYING_BUFFER_UPLOADS, 1);
    upload_block_copy(device, pointer, byte_length).map(|buffer| (buffer, 0))
}

fn upload_block_int32_as_float(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[i32],
    resident_name: Option<&str>,
) -> Result<(MetalBuffer, usize), MetalError> {
    let converted: Vec<f32> = data.iter().map(|value| *value as f32).collect();
    upload_block_as_float(device, converted.as_slice(), resident_name)
}

/// Uploads a packed quantized weight buffer as raw BYTES — no dequantize on
/// the host, which is the entire point. A 7B `Q4_K_S` checkpoint is 3.784 GB
/// packed against 14.5 GB as `f16`; decode is a weight sweep, so that 3.56x
/// in traffic IS the token rate. Reuses the same page-aligned no-copy path
/// [`upload_block_as_float`] uses, since a memory-mapped GGUF tensor is very
/// often already page-aligned. `resident_name` is the same "caller's own
/// static weight" classification [`upload_block_as_float`] takes; a packed
/// weight too misaligned for the no-copy path takes the same resident-copy
/// cache.
fn upload_packed_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
    resident_name: Option<&str>,
) -> Result<(MetalBuffer, usize), MetalError> {
    if bytes.len() > 100 * 1024 * 1024 && std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some()
    {
        eprintln!(
            "metal large packed upload bytes={} resident={resident_name:?}",
            bytes.len()
        );
    }
    if bytes.is_empty() {
        return allocate_buffer(device, 0, DType::Float32).map(|buffer| (buffer, 0));
    }
    let byte_length = bytes.len();
    let pointer = bytes.as_ptr().cast::<c_void>();
    if is_page_aligned(pointer, byte_length) {
        counter!(NOCOPY_BUFFER_UPLOADS, 1);
        counter!(BLOCK_NOCOPY_BOUND_BYTES, byte_length as u64);
        if let Some(name) = resident_name {
            return upload_block_no_copy(device, name, pointer, byte_length)
                .map(|buffer| (buffer, 0));
        }
        return upload_block_no_copy_uncached(device, pointer, byte_length)
            .map(|buffer| (buffer, 0));
    }
    if let Some(result) = checkpoint_mapping_offset(device, pointer, byte_length) {
        return result;
    }
    if let Some(result) = expert_mapping_offset(device, pointer, byte_length) {
        return result;
    }
    if let Some(name) = resident_name {
        return upload_resident_copy(device, name, pointer, byte_length).map(|buffer| (buffer, 0));
    }
    counter!(BLOCK_COPIED_BYTES, byte_length as u64);
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
    static EXPERT_MAPPING: RefCell<Option<(usize, usize)>> = const { RefCell::new(None) };
}

pub fn register_expert_mapping(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let identity = (bytes.as_ptr() as usize, bytes.len());
    let changed = EXPERT_MAPPING.with(|mapping| {
        let mut mapping = mapping.borrow_mut();
        if *mapping == Some(identity) {
            false
        } else {
            *mapping = Some(identity);
            true
        }
    });
    if changed {
        EXPERT_SOURCE_CACHE.with(|cache| cache.borrow_mut().clear());
        NOCOPY_BUFFERS.with(|cache| {
            cache.borrow_mut().remove(EXPERT_MAPPING_NOCOPY_NAME);
        });
    }
}

pub fn unregister_expert_mapping(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let identity = (bytes.as_ptr() as usize, bytes.len());
    let matched = EXPERT_MAPPING.with(|mapping| {
        let mut mapping = mapping.borrow_mut();
        if *mapping == Some(identity) {
            *mapping = None;
            true
        } else {
            false
        }
    });
    if matched {
        EXPERT_SOURCE_CACHE.with(|cache| cache.borrow_mut().clear());
        NOCOPY_BUFFERS.with(|cache| {
            cache.borrow_mut().remove(EXPERT_MAPPING_NOCOPY_NAME);
        });
    }
}

fn expert_mapping_contains(bytes: &[u8]) -> bool {
    let address = bytes.as_ptr() as usize;
    EXPERT_MAPPING.with(|mapping| {
        mapping.borrow().is_some_and(|(base, mapping_length)| {
            address >= base
                && address
                    .checked_add(bytes.len())
                    .is_some_and(|end| end <= base.saturating_add(mapping_length))
        })
    })
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
    // this call is the caller EXPLICITLY declaring a new resident identity
    // for `CHECKPOINT_MAPPING_NOCOPY_NAME` -- drop whatever `NOCOPY_BUFFERS`
    // cached under that name for the PRIOR registration so the next
    // `checkpoint_mapping_offset` upload is a fresh, correctly-checked
    // entry rather than a `ResidentNameRebound` error against a mapping
    // this function itself just superseded.
    NOCOPY_BUFFERS.with(|cache| {
        cache.borrow_mut().remove(CHECKPOINT_MAPPING_NOCOPY_NAME);
    });
}

/// Advises the kernel that pages covering a checkpoint byte range are not
/// needed after an expert has been demoted to its sidecar copy. The mapping
/// remains valid; a later high promotion may fault the bytes back in.
pub fn discard_checkpoint_mmap_range(bytes: &[u8]) -> Result<(), MetalError> {
    if bytes.is_empty() {
        return Ok(());
    }
    let page = page_size();
    let start = (bytes.as_ptr() as usize) & !(page - 1);
    let end = (bytes.as_ptr() as usize)
        .checked_add(bytes.len())
        .and_then(|value| value.checked_add(page - 1))
        .map(|value| value & !(page - 1))
        .ok_or(MetalError::CheckpointMmapDiscardFailed {
            errno: libc::EINVAL,
        })?;
    let length = end
        .checked_sub(start)
        .ok_or(MetalError::CheckpointMmapDiscardFailed {
            errno: libc::EINVAL,
        })?;
    let result = unsafe { libc::madvise(start as *mut libc::c_void, length, libc::MADV_FREE) };
    if result == 0 {
        Ok(())
    } else {
        Err(MetalError::CheckpointMmapDiscardFailed {
            errno: std::io::Error::last_os_error()
                .raw_os_error()
                .map_or(libc::EIO, |error| error),
        })
    }
}

/// Advises Darwin to reclaim mapped pages immediately when the caller is
/// enforcing a hard resident-memory ceiling. Unlike `MADV_FREE`, this does
/// not leave the pages in the process footprint until memory pressure arrives.
pub fn discard_checkpoint_mmap_range_immediate(bytes: &[u8]) -> Result<(), MetalError> {
    if bytes.is_empty() {
        return Ok(());
    }
    let page = page_size();
    let start = (bytes.as_ptr() as usize) & !(page - 1);
    let end = (bytes.as_ptr() as usize)
        .checked_add(bytes.len())
        .and_then(|value| value.checked_add(page - 1))
        .map(|value| value & !(page - 1))
        .ok_or(MetalError::CheckpointMmapDiscardFailed {
            errno: libc::EINVAL,
        })?;
    let length = end
        .checked_sub(start)
        .ok_or(MetalError::CheckpointMmapDiscardFailed {
            errno: libc::EINVAL,
        })?;
    let result = unsafe { libc::madvise(start as *mut libc::c_void, length, libc::MADV_DONTNEED) };
    if result == 0 {
        Ok(())
    } else {
        Err(MetalError::CheckpointMmapDiscardFailed {
            errno: std::io::Error::last_os_error()
                .raw_os_error()
                .map_or(libc::EIO, |error| error),
        })
    }
}

/// Counts resident pages covering a checkpoint range using Darwin's `mincore`.
/// This is diagnostic-only and lets the serving harness distinguish mmap page
/// residency from retained heap allocations after an expert eviction.
pub fn checkpoint_mmap_resident_pages(bytes: &[u8]) -> Result<usize, MetalError> {
    if bytes.is_empty() {
        return Ok(0);
    }
    let page = page_size();
    let start = (bytes.as_ptr() as usize) & !(page - 1);
    let end = (bytes.as_ptr() as usize)
        .checked_add(bytes.len())
        .and_then(|value| value.checked_add(page - 1))
        .map(|value| value & !(page - 1))
        .ok_or(MetalError::CheckpointMmapDiscardFailed {
            errno: libc::EINVAL,
        })?;
    let length = end
        .checked_sub(start)
        .ok_or(MetalError::CheckpointMmapDiscardFailed {
            errno: libc::EINVAL,
        })?;
    let page_count = length / page;
    let mut residency = vec![0_i8; page_count];
    let result = unsafe {
        libc::mincore(
            start as *const libc::c_void,
            length,
            residency.as_mut_ptr().cast::<libc::c_char>(),
        )
    };
    if result != 0 {
        return Err(MetalError::CheckpointMmapDiscardFailed {
            errno: std::io::Error::last_os_error()
                .raw_os_error()
                .map_or(libc::EIO, |error| error),
        });
    }
    Ok(residency
        .iter()
        .filter(|state| **state & libc::MINCORE_INCORE as i8 != 0)
        .count())
}

/// Unregisters the checkpoint mapping [`register_checkpoint_mapping`]
/// installed for `bytes`, evicting its whole-mapping no-copy buffer from
/// `NOCOPY_BUFFERS` -- but ONLY when `bytes` is still the currently
/// registered mapping. [`register_checkpoint_mapping`] already evicts a
/// PRIOR mapping's entry the moment a new one is registered (see that
/// function's own doc), so a dropped model racing behind a second model's
/// load must not clear the second model's live mapping -- comparing the
/// base pointer and length is what tells "this is still mine" apart from
/// "someone else already superseded this". A no-op when `bytes` is empty
/// (matching [`register_checkpoint_mapping`]'s own early return) or when no
/// mapping this identity matches is currently registered.
pub fn unregister_checkpoint_mapping(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let identity = (bytes.as_ptr() as usize, bytes.len());
    let matched = CHECKPOINT_MAPPING.with(|mapping| {
        let mut mapping = mapping.borrow_mut();
        if *mapping == Some(identity) {
            *mapping = None;
            true
        } else {
            false
        }
    });
    if matched {
        NOCOPY_BUFFERS.with(|cache| {
            cache.borrow_mut().remove(CHECKPOINT_MAPPING_NOCOPY_NAME);
        });
    }
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
        upload_block_no_copy(
            device,
            CHECKPOINT_MAPPING_NOCOPY_NAME,
            base as *const c_void,
            rounded_length,
        )
        .map(|buffer| (buffer, address - base)),
    )
}

fn expert_mapping_offset(
    device: &ProtocolObject<dyn MTLDevice>,
    pointer: *const c_void,
    byte_length: usize,
) -> Option<Result<(MetalBuffer, usize), MetalError>> {
    let (base, mapping_length) = EXPERT_MAPPING.with(|mapping| *mapping.borrow())?;
    let address = pointer as usize;
    let end = address.checked_add(byte_length)?;
    let mapping_end = base.checked_add(mapping_length)?;
    if address < base || end > mapping_end {
        return None;
    }
    let rounded_length = mapping_length.div_ceil(page_size()) * page_size();
    counter!(MAPPING_OFFSET_UPLOADS, 1);
    counter!(BLOCK_OFFSET_BOUND_BYTES, byte_length as u64);
    Some(
        upload_block_no_copy(
            device,
            EXPERT_MAPPING_NOCOPY_NAME,
            base as *const c_void,
            rounded_length,
        )
        .map(|buffer| (buffer, address - base)),
    )
}

/// The single [`NOCOPY_BUFFERS`] identity [`checkpoint_mapping_offset`]
/// caches under -- sound because [`CHECKPOINT_MAPPING`] itself is a single
/// thread-local slot (never more than one registration live at a time), so
/// this name never collides across two DIFFERENT live mappings the way a
/// per-tensor or per-weight name would need to. [`register_checkpoint_mapping`]
/// drops any stale entry under this name before installing a new mapping, so
/// a re-registration is a deliberate cache invalidation, never a
/// [`MetalError::ResidentNameRebound`].
const CHECKPOINT_MAPPING_NOCOPY_NAME: &str = "__checkpoint_mapping__";
const EXPERT_MAPPING_NOCOPY_NAME: &str = "__expert_mapping__";

/// Test-only reset -- the default std test harness reuses threads across
/// tests in the same binary (see [`reset_nocopy_cache_for_test`]'s own
/// doc), and [`CHECKPOINT_MAPPING`] is thread-local, so a prior test's
/// registration would otherwise leak into a later test on the same thread.
#[cfg(test)]
fn reset_checkpoint_mapping_for_test() {
    CHECKPOINT_MAPPING.with(|mapping| *mapping.borrow_mut() = None);
    EXPERT_MAPPING.with(|mapping| *mapping.borrow_mut() = None);
    NOCOPY_BUFFERS.with(|cache| {
        cache.borrow_mut().remove(CHECKPOINT_MAPPING_NOCOPY_NAME);
        cache.borrow_mut().remove(EXPERT_MAPPING_NOCOPY_NAME);
    });
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod checkpoint_mapping_release_tests {
    use proxima_tensor::AlignedBuffer;

    use super::{
        checkpoint_mapping_offset, device_and_queue, expert_mapping_offset,
        mapping_buffer_allocated_bytes, page_size, register_checkpoint_mapping,
        register_expert_mapping, reset_checkpoint_mapping_for_test, unregister_checkpoint_mapping,
        unregister_expert_mapping,
    };

    /// The exact drop-ordering hazard `LoadedModel::drop` is written
    /// against: model A loads (registers its mapping), model B loads
    /// afterward (its own `register_checkpoint_mapping` call supersedes
    /// A's, per that function's own doc), and only THEN does A's `Drop`
    /// run. A's stale `unregister_checkpoint_mapping(bytes_a)` must be a
    /// no-op -- it is no longer the current registration -- so B's mapping
    /// stays resolvable. Only unregistering the CURRENTLY registered
    /// identity actually clears it.
    #[test]
    fn unregister_only_clears_the_currently_registered_identity() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_checkpoint_mapping_for_test();

        let page = page_size();
        // page-aligned, one page each -- `newBufferWithBytesNoCopy`'s own
        // precondition (`create_no_copy_buffer`'s SAFETY doc), which
        // `checkpoint_mapping_offset` exercises via `upload_block_no_copy`.
        let model_a = AlignedBuffer::new(page / core::mem::size_of::<f32>(), page)
            .expect("page-aligned fixture for fake model A's checkpoint bytes");
        let model_b = AlignedBuffer::new(page / core::mem::size_of::<f32>(), page)
            .expect("page-aligned fixture for fake model B's checkpoint bytes");
        let model_a_bytes =
            unsafe { core::slice::from_raw_parts(model_a.as_ptr().cast::<u8>(), page) };
        let model_b_bytes =
            unsafe { core::slice::from_raw_parts(model_b.as_ptr().cast::<u8>(), page) };

        register_checkpoint_mapping(model_a_bytes);
        register_checkpoint_mapping(model_b_bytes);

        // model A's drop races behind model B's load -- releasing A's own
        // (now-stale) identity must not disturb B's live mapping.
        unregister_checkpoint_mapping(model_a_bytes);
        assert!(
            checkpoint_mapping_offset(&device, model_b_bytes.as_ptr().cast(), model_b_bytes.len())
                .is_some(),
            "model B's own mapping must survive a stale release of model A's superseded one"
        );

        // model B's own drop releases its own, still-current identity.
        unregister_checkpoint_mapping(model_b_bytes);
        assert!(
            checkpoint_mapping_offset(&device, model_b_bytes.as_ptr().cast(), model_b_bytes.len())
                .is_none(),
            "releasing the CURRENTLY registered identity must actually clear it"
        );
    }

    #[test]
    fn expert_mapping_resolves_a_misaligned_arena_slice_and_unregisters() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_checkpoint_mapping_for_test();

        let page = page_size();
        let mapping = AlignedBuffer::new(page / core::mem::size_of::<f32>(), page)
            .expect("page-aligned fixture for an expert sidecar mapping");
        let mapping_bytes =
            unsafe { core::slice::from_raw_parts(mapping.as_ptr().cast::<u8>(), page) };
        let arena = &mapping_bytes[17..101];

        register_expert_mapping(mapping_bytes);
        let (_buffer, offset) = expert_mapping_offset(&device, arena.as_ptr().cast(), arena.len())
            .expect("the arena lies inside the registered expert mapping")
            .expect("the whole expert mapping binds without copying");
        assert_eq!(offset, 17);
        assert_eq!(mapping_buffer_allocated_bytes(), (0, page as u64));

        unregister_expert_mapping(mapping_bytes);
        assert_eq!(mapping_buffer_allocated_bytes(), (0, 0));
        assert!(
            expert_mapping_offset(&device, arena.as_ptr().cast(), arena.len()).is_none(),
            "unregistering the current expert mapping removes the offset alias"
        );
    }
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
    /// Keyed on the resident NAME [`Plan::mark_resident`] proved static for
    /// this node, never on `(pointer, byte_length)` alone -- ROW 334's own
    /// shape: an address-only key let a freed arm's `Vec<u8>` and a LATER,
    /// unrelated arm's same-byte-size `Vec<u8>` collide at the identical
    /// address, serving the later arm 100% stale bytes. `upload_block_no_copy`
    /// checks the offered pointer and byte length against what this name was
    /// first uploaded with on every lookup and refuses to serve a mismatch
    /// (see that function's own doc) -- the same contract
    /// [`upload_resident_copy`]/[`RESIDENT_BUFFERS`] enforce for the copy
    /// path, applied here to the no-copy path. See
    /// `proxima-tensor/docs/discipline.md` ROW 70/334.
    ///
    /// `upload_block_as_float` and `upload_packed_bytes` only route into this
    /// cache when [`Plan::mark_resident`] already proved a name for the
    /// node's own address -- true for mmap'd GGUF weights, false for an
    /// ephemeral, growing buffer (a KV-cache row, say) whose page-aligned
    /// address is a coincidence of a page-boundary crossing, not a lifetime
    /// proof. A page-aligned but non-resident (unnamed) block takes
    /// `upload_block_no_copy_uncached` instead: still zero-copy for this one
    /// `execute` call (sound for the same `waitUntilCompleted` reason), just
    /// never remembered past it, and never inserted here.
    ///
    /// Reuse is otherwise safe on the data-freshness axis precisely BECAUSE
    /// it is no-copy: writes through the caller's own slice are visible to
    /// the GPU, so a wrapper never goes stale. Copying uploads are
    /// deliberately NOT cached — those snapshot the data, and reuse would
    /// serve a stale snapshot.
    static NOCOPY_BUFFERS: RefCell<BTreeMap<String, (usize, usize, MetalBuffer)>> =
        RefCell::new(BTreeMap::new());
}

/// Shared identity check both the no-copy and resident-copy caches enforce:
/// a `name` hit is served only when the OFFERED host pointer and byte length
/// match what this name was first cached with; any other name hit is
/// [`MetalError::ResidentNameRebound`], never a stale serve and never a
/// silent replace. Composes with [`RESIDENT_BUFFERS`]/[`upload_resident_copy`]
/// and [`NOCOPY_BUFFERS`]/[`upload_block_no_copy`], the two callers that own
/// their own separate maps (different soundness arguments -- see each map's
/// own doc) but share this one lookup rule.
fn resident_name_lookup(
    cache: &RefCell<BTreeMap<String, (usize, usize, MetalBuffer)>>,
    name: &str,
    pointer: *const c_void,
    byte_length: usize,
) -> Result<Option<MetalBuffer>, MetalError> {
    let offered_address = pointer as usize;
    let Some((cached_address, cached_length, buffer)) = cache.borrow().get(name).cloned() else {
        return Ok(None);
    };
    if cached_address == offered_address && cached_length == byte_length {
        return Ok(Some(buffer));
    }
    Err(MetalError::ResidentNameRebound {
        name: name.to_string(),
        cached_len: cached_length,
        offered_len: byte_length,
    })
}

/// Counts a fresh [`Plan`]'s per-node fast path resolving a resident block
/// straight from the process-global caches (`NOCOPY_BUFFERS`,
/// `RESIDENT_BUFFERS`, the checkpoint mapping) instead of taking the
/// upload path -- the direct witness that a KV-bucket-boundary reshape does
/// not re-walk the checkpoint's weight blocks through
/// `upload_block`/`upload_packed_bytes` just to re-derive a buffer those
/// caches already hold.
pub static PLAN_HANDOFF_REUSES: Counter = Counter::new("omega.metal.plan_handoff_reuses");

/// Resolves `(pointer, byte_length)` against the GLOBAL, name-keyed residency
/// caches without recording it as a fresh "offer" -- the fast path a brand
/// new [`Plan`] takes so its own empty `device_buffers`/`block_identity`
/// (see [`plan`]'s doc) never forces a resident weight block through the
/// host round trip a PRIOR plan already paid once. `Ok(None)` means none of
/// the three caches have this host range yet, so the caller falls through to
/// `upload_block`/`upload_packed_bytes`'s normal offer path unchanged.
///
/// Never creates a buffer and never mutates a cache -- a rebind (the offered
/// `(pointer, byte_length)` disagreeing with what a name already holds) is
/// [`MetalError::ResidentNameRebound`] via [`resident_name_lookup`], the same
/// contract [`upload_block_no_copy`]/[`upload_resident_copy`] enforce, never
/// a silent reuse of a different allocation under the same name.
fn cross_plan_resident_reuse(
    resident_name: Option<&str>,
    pointer: *const c_void,
    byte_length: usize,
) -> Result<Option<(MetalBuffer, usize)>, MetalError> {
    if let Some(name) = resident_name {
        if let Some(buffer) =
            NOCOPY_BUFFERS.with(|cache| resident_name_lookup(cache, name, pointer, byte_length))?
        {
            counter!(PLAN_HANDOFF_REUSES, 1);
            return Ok(Some((buffer, 0)));
        }
        if let Some(buffer) = RESIDENT_BUFFERS
            .with(|cache| resident_name_lookup(cache, name, pointer, byte_length))?
        {
            counter!(PLAN_HANDOFF_REUSES, 1);
            return Ok(Some((buffer, 0)));
        }
    }
    let Some((base, mapping_length)) = CHECKPOINT_MAPPING.with(|mapping| *mapping.borrow()) else {
        return Ok(None);
    };
    let address = pointer as usize;
    if address < base || address + byte_length > base + mapping_length {
        return Ok(None);
    }
    let rounded_length = mapping_length.div_ceil(page_size()) * page_size();
    let Some(buffer) = NOCOPY_BUFFERS.with(|cache| {
        resident_name_lookup(
            cache,
            CHECKPOINT_MAPPING_NOCOPY_NAME,
            base as *const c_void,
            rounded_length,
        )
    })?
    else {
        return Ok(None);
    };
    counter!(PLAN_HANDOFF_REUSES, 1);
    Ok(Some((buffer, address - base)))
}

/// Counts entries `NOCOPY_BUFFERS` actually holds right now — the direct
/// witness that gating the cache on `resident` stops it growing without
/// bound. A non-resident page-aligned block (a KV-cache row that happened to
/// cross a page boundary) never reaches this map at all.
#[must_use]
pub fn nocopy_cache_len() -> usize {
    NOCOPY_BUFFERS.with(|cache| cache.borrow().len())
}

/// metal-visible byte lengths of the whole-file checkpoint and expert mmap
/// wrappers currently retained by the no-copy cache. these are the driver's
/// actual `MTLBuffer::length()` values, not tensor-shape estimates.
#[must_use]
pub fn mapping_buffer_allocated_bytes() -> (u64, u64) {
    NOCOPY_BUFFERS.with(|cache| {
        let cache = cache.borrow();
        let buffer_length = |name: &str| {
            cache
                .get(name)
                .map_or(0, |(_, _, buffer)| buffer.length() as u64)
        };
        (
            buffer_length(CHECKPOINT_MAPPING_NOCOPY_NAME),
            buffer_length(EXPERT_MAPPING_NOCOPY_NAME),
        )
    })
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
/// classified this address `resident` under `name` (see the call sites in
/// [`upload_block_as_float`]/[`upload_packed_bytes`]/[`checkpoint_mapping_offset`]);
/// a non-resident page-aligned block takes [`upload_block_no_copy_uncached`]
/// instead, which shares this function's Metal call but never remembers the
/// wrapper -- an unnamed upload is never cached here. A `name` hit whose
/// offered pointer or byte length disagrees with what is cached is a caller
/// contract violation ([`Plan::mark_resident`]'s doc), not a fresher version
/// of the same buffer, so it is [`MetalError::ResidentNameRebound`], never a
/// stale hit and never a silent re-upload. See [`resident_name_lookup`] and
/// [`NOCOPY_BUFFERS`]'s own doc.
fn upload_block_no_copy(
    device: &ProtocolObject<dyn MTLDevice>,
    name: &str,
    pointer: *const c_void,
    byte_length: usize,
) -> Result<MetalBuffer, MetalError> {
    if let Some(existing) =
        NOCOPY_BUFFERS.with(|cache| resident_name_lookup(cache, name, pointer, byte_length))?
    {
        counter!(NOCOPY_BUFFER_REUSES, 1);
        return Ok(existing);
    }
    let buffer = create_no_copy_buffer(device, pointer, byte_length)?;
    NOCOPY_BUFFERS.with(|cache| {
        cache.borrow_mut().insert(
            name.to_string(),
            (pointer as usize, byte_length, buffer.clone()),
        )
    });
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
    /// the NAME holds a model weight nothing overwrites again. Keyed on that
    /// name, never on `(pointer, byte_length)` alone: a host address is a
    /// property of an allocation's LIFETIME, and a short-lived buffer (an
    /// activation vector, say) can be freed and a same-sized,
    /// differently-contented allocation can land at the identical address on
    /// a later call -- `mark_resident`'s own proof is about the NAME the
    /// caller declared static, not about any address that name's data
    /// happened to occupy once. See `proxima-tensor/docs/discipline.md` ROW 70.
    ///
    /// The stored `(usize, usize, MetalBuffer)` is the host pointer and byte
    /// length this entry was uploaded from, alongside the device copy --
    /// ROW 332's shape was a caller marking a DIFFERENT host buffer resident
    /// under a REUSED name, which a name-only lookup cannot distinguish from
    /// a legitimate cache hit. [`upload_resident_copy`] checks both against
    /// what the caller offers on every lookup and refuses to serve a
    /// mismatch -- see that function's own doc.
    static RESIDENT_BUFFERS: RefCell<BTreeMap<String, (usize, usize, MetalBuffer)>> =
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

/// Entries `RESIDENT_BUFFERS` holds right now -- the direct witness that
/// residency reuse tracks the caller's declared NAME set, not the address
/// space: this stays exactly the plan's resident-name count across repeated
/// `execute_plan` calls for the same names, regardless of how many times the
/// host allocator has reused an address underneath them.
#[must_use]
pub fn resident_cache_len() -> usize {
    RESIDENT_BUFFERS.with(|cache| cache.borrow().len())
}

/// The copy-path counterpart to [`upload_block_no_copy`]: called only for a
/// block [`upload_block_as_float`]/[`upload_packed_bytes`] already found
/// misaligned AND [`Plan::mark_resident`] already classified `name` as
/// static, so unlike [`upload_block_copy`] this one is allowed to remember
/// the buffer it creates and hand the SAME one back next time the SAME
/// `name` shows up -- sound only because that classification, not an
/// address guess, is what proves the bytes behind `name` never change
/// again.
///
/// [`Plan::mark_resident`]'s doc is the contract this enforces: under a
/// legitimate caller, a resident name's host pointer and byte length never
/// change once bound. A `name` hit whose OFFERED host pointer or byte length
/// disagrees with what is cached is therefore not a fresher version of the
/// same weight -- it is a different host allocation that happened to reuse a
/// name a serving loop already marked resident (ROW 331/332's shape). This
/// cache cannot tell "the weight changed" from "the caller has a bug" (the
/// module doc's residency precondition rules the former case out), so it
/// never guesses: it refuses to serve, and never silently replaces the
/// cached entry, via [`MetalError::ResidentNameRebound`].
fn upload_resident_copy(
    device: &ProtocolObject<dyn MTLDevice>,
    name: &str,
    pointer: *const c_void,
    byte_length: usize,
) -> Result<MetalBuffer, MetalError> {
    if let Some(existing) =
        RESIDENT_BUFFERS.with(|cache| resident_name_lookup(cache, name, pointer, byte_length))?
    {
        counter!(RESIDENT_BUFFER_REUSES, 1);
        return Ok(existing);
    }
    counter!(COPYING_BUFFER_UPLOADS, 1);
    counter!(RESIDENT_BUFFER_UPLOADS, 1);
    counter!(BLOCK_COPIED_BYTES, byte_length as u64);
    let buffer = upload_block_copy(device, pointer, byte_length)?;
    RESIDENT_BUFFERS.with(|cache| {
        cache.borrow_mut().insert(
            name.to_string(),
            (pointer as usize, byte_length, buffer.clone()),
        )
    });
    Ok(buffer)
}

/// Evicts every buffer a checkpoint's own resident weight names cached in
/// `NOCOPY_BUFFERS`/`RESIDENT_BUFFERS` -- the drop-time counterpart to
/// `upload_block_no_copy`/`upload_resident_copy`'s insert. Removing by
/// NAME, never a blanket clear, is what lets a second, unrelated model
/// loaded on this same thread keep its own entries live after the first
/// model drops -- see `LoadedModel`'s own `Drop` impl in
/// `proxima-model-interop` for the caller. A name absent from either cache
/// (a resident block this run never actually uploaded) is silently
/// skipped.
pub fn release_resident_names<'name>(names: impl IntoIterator<Item = &'name str>) {
    for name in names {
        NOCOPY_BUFFERS.with(|cache| cache.borrow_mut().remove(name));
        RESIDENT_BUFFERS.with(|cache| cache.borrow_mut().remove(name));
    }
}

#[cfg(test)]
fn reset_resident_cache_for_test() {
    RESIDENT_BUFFERS.with(|cache| cache.borrow_mut().clear());
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod resident_buffer_cache_tests {
    use core::mem::size_of_val;

    use super::{
        MetalBuffer, device_and_queue, read_back_as_device_f32, reset_resident_cache_for_test,
        resident_cache_len, upload_block_copy, upload_resident_copy,
    };

    fn pre_fix_address_keyed_lookup(
        cache: &mut alloc::collections::BTreeMap<(usize, usize), MetalBuffer>,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        pointer: *const core::ffi::c_void,
        byte_length: usize,
    ) -> MetalBuffer {
        let key = (pointer as usize, byte_length);
        if let Some(existing) = cache.get(&key) {
            return existing.clone();
        }
        let buffer = upload_block_copy(device, pointer, byte_length).expect("pre-fix copy upload");
        cache.insert(key, buffer.clone());
        buffer
    }

    #[test]
    fn pre_fix_address_keyed_cache_served_stale_content_across_names() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        let mut pre_fix_cache = alloc::collections::BTreeMap::new();

        let mut host_buffer = vec![1.0_f32; 4096];
        let pointer = host_buffer.as_ptr().cast();
        let byte_length = size_of_val(host_buffer.as_slice());

        let first = pre_fix_address_keyed_lookup(&mut pre_fix_cache, &device, pointer, byte_length);
        let first_content = read_back_as_device_f32(&first, 0, host_buffer.len());
        assert_eq!(first_content, vec![1.0_f32; 4096]);

        host_buffer.fill(2.0_f32);
        let second =
            pre_fix_address_keyed_lookup(&mut pre_fix_cache, &device, pointer, byte_length);
        let second_content = read_back_as_device_f32(&second, 0, host_buffer.len());
        assert_eq!(second_content, vec![1.0_f32; 4096]);
    }

    #[test]
    fn name_keyed_cache_never_serves_a_different_names_content() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_resident_cache_for_test();

        let mut host_buffer = vec![1.0_f32; 4096];
        let pointer = host_buffer.as_ptr().cast();
        let byte_length = size_of_val(host_buffer.as_slice());

        let first = upload_resident_copy(&device, "resident_weight_one", pointer, byte_length)
            .expect("first resident upload");
        assert_eq!(
            read_back_as_device_f32(&first, 0, host_buffer.len()),
            vec![1.0_f32; 4096]
        );

        host_buffer.fill(2.0_f32);
        let second = upload_resident_copy(&device, "resident_weight_two", pointer, byte_length)
            .expect("second resident upload under a different name");
        assert_eq!(
            read_back_as_device_f32(&second, 0, host_buffer.len()),
            vec![2.0_f32; 4096]
        );
        assert_eq!(resident_cache_len(), 2);
    }

    #[test]
    fn the_same_resident_name_still_hits_the_cache() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_resident_cache_for_test();

        let host_buffer = vec![3.0_f32; 4096];
        let pointer = host_buffer.as_ptr().cast();
        let byte_length = size_of_val(host_buffer.as_slice());

        upload_resident_copy(&device, "resident_weight_stable", pointer, byte_length)
            .expect("first upload for a stable resident name");
        assert_eq!(resident_cache_len(), 1);

        let reuses_before = super::RESIDENT_BUFFER_REUSES.get();
        upload_resident_copy(&device, "resident_weight_stable", pointer, byte_length)
            .expect("second upload of the same name must hit the cache");
        assert_eq!(super::RESIDENT_BUFFER_REUSES.get(), reuses_before + 1);
        assert_eq!(resident_cache_len(), 1);
    }

    #[test]
    fn a_resident_name_rebound_to_a_different_host_pointer_is_rejected() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_resident_cache_for_test();

        let first_host_buffer = vec![4.0_f32; 4096];
        let byte_length = size_of_val(first_host_buffer.as_slice());
        upload_resident_copy(
            &device,
            "resident_weight_rebound",
            first_host_buffer.as_ptr().cast(),
            byte_length,
        )
        .expect("first upload establishes the cached entry");

        // a second, unrelated host allocation of the SAME length reusing the
        // SAME name -- exactly ROW 331/332's shape, reproduced at the driver
        // level instead of relying on a harness never naming two arms alike.
        let second_host_buffer = vec![5.0_f32; 4096];
        let uploads_before = super::RESIDENT_BUFFER_UPLOADS.get();
        let reuses_before = super::RESIDENT_BUFFER_REUSES.get();
        let error = upload_resident_copy(
            &device,
            "resident_weight_rebound",
            second_host_buffer.as_ptr().cast(),
            byte_length,
        )
        .expect_err("a different host pointer under the same name must never be served");

        assert!(matches!(
            error,
            super::MetalError::ResidentNameRebound { .. }
        ));
        assert_eq!(super::RESIDENT_BUFFER_UPLOADS.get(), uploads_before);
        assert_eq!(super::RESIDENT_BUFFER_REUSES.get(), reuses_before);
        assert_eq!(resident_cache_len(), 1);
    }

    #[test]
    fn a_resident_name_rebound_to_a_different_length_is_rejected() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_resident_cache_for_test();

        let host_buffer = vec![6.0_f32; 4096];
        let pointer = host_buffer.as_ptr().cast();
        let full_length = size_of_val(host_buffer.as_slice());
        upload_resident_copy(&device, "resident_weight_grown", pointer, full_length)
            .expect("first upload establishes the cached entry");

        let uploads_before = super::RESIDENT_BUFFER_UPLOADS.get();
        let reuses_before = super::RESIDENT_BUFFER_REUSES.get();
        let error =
            upload_resident_copy(&device, "resident_weight_grown", pointer, full_length / 2)
                .expect_err("a different byte length under the same name must never be served");

        assert!(matches!(
            error,
            super::MetalError::ResidentNameRebound { .. }
        ));
        assert_eq!(super::RESIDENT_BUFFER_UPLOADS.get(), uploads_before);
        assert_eq!(super::RESIDENT_BUFFER_REUSES.get(), reuses_before);
        assert_eq!(resident_cache_len(), 1);
    }

    /// A cache HIT never moves bytes -- `BLOCK_COPIED_BYTES` must reflect
    /// that (ROW 369): the counter used to fire at the call site, before the
    /// lookup that can avoid the copy, so a resident-cache hit was double
    /// counted as if it had re-copied the whole block every time.
    #[test]
    fn a_resident_cache_hit_adds_no_bytes_to_the_copied_counter() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_resident_cache_for_test();
        let _ = super::BLOCK_COPIED_BYTES.snapshot_and_reset();

        let host_buffer = vec![7.0_f32; 4096];
        let pointer = host_buffer.as_ptr().cast();
        let byte_length = size_of_val(host_buffer.as_slice());

        let copied_before_miss = super::BLOCK_COPIED_BYTES.get();
        upload_resident_copy(
            &device,
            "resident_weight_copied_counter",
            pointer,
            byte_length,
        )
        .expect("first upload is a real copy (a cache miss)");
        assert!(
            super::BLOCK_COPIED_BYTES.get() >= copied_before_miss + byte_length as u64,
            "a genuine copy must add its own byte length"
        );

        let copied_before_hit = super::BLOCK_COPIED_BYTES.get();
        upload_resident_copy(
            &device,
            "resident_weight_copied_counter",
            pointer,
            byte_length,
        )
        .expect("second upload of the same name hits the cache");
        assert_eq!(
            super::BLOCK_COPIED_BYTES.get(),
            copied_before_hit,
            "a cache hit moves zero bytes, so it must add zero"
        );
    }

    /// [`super::release_resident_names`]'s copy-path counterpart to
    /// `nocopy_buffer_cache_tests`'s own
    /// `release_resident_names_evicts_only_the_named_entry` -- the same
    /// two-fake-model shape, this time through the misaligned/resident-copy
    /// cache a real GGUF's odd-byte-offset tensors actually take.
    #[test]
    fn release_resident_names_evicts_only_the_named_resident_copy() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_resident_cache_for_test();

        let model_a = vec![1.0_f32; 64];
        let model_b = vec![2.0_f32; 64];
        let byte_length = size_of_val(model_a.as_slice());

        upload_resident_copy(
            &device,
            "fake_model_a.weight",
            model_a.as_ptr().cast(),
            byte_length,
        )
        .expect("bind fake model A's resident copy");
        upload_resident_copy(
            &device,
            "fake_model_b.weight",
            model_b.as_ptr().cast(),
            byte_length,
        )
        .expect("bind fake model B's resident copy");
        assert_eq!(
            resident_cache_len(),
            2,
            "both fake models' copies are cached"
        );

        super::release_resident_names(["fake_model_a.weight"]);
        assert_eq!(
            resident_cache_len(),
            1,
            "releasing model A's own name must evict exactly one entry"
        );

        let reuses_before = super::RESIDENT_BUFFER_REUSES.get();
        upload_resident_copy(
            &device,
            "fake_model_b.weight",
            model_b.as_ptr().cast(),
            byte_length,
        )
        .expect("model B's own name must still resolve after model A's release");
        assert!(
            super::RESIDENT_BUFFER_REUSES.get() >= reuses_before + 1,
            "model B's entry must still be a cache HIT -- it was never released"
        );
    }
}

#[cfg(test)]
fn reset_nocopy_cache_for_test() {
    NOCOPY_BUFFERS.with(|cache| cache.borrow_mut().clear());
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod nocopy_buffer_cache_tests {
    use core::mem::size_of;

    use proxima_tensor::AlignedBuffer;

    use super::{
        device_and_queue, nocopy_cache_len, page_size, reset_nocopy_cache_for_test,
        upload_block_no_copy, upload_block_no_copy_uncached,
    };

    #[test]
    fn the_same_nocopy_name_still_hits_the_cache() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_nocopy_cache_for_test();

        let page = page_size();
        let buffer =
            AlignedBuffer::new(page / size_of::<f32>(), page).expect("page-aligned test buffer");
        let pointer = buffer.as_ptr().cast();
        let byte_length = buffer.len() * size_of::<f32>();

        upload_block_no_copy(&device, "nocopy_weight_stable", pointer, byte_length)
            .expect("first upload for a stable no-copy name");
        assert_eq!(nocopy_cache_len(), 1);

        let reuses_before = super::NOCOPY_BUFFER_REUSES.get();
        upload_block_no_copy(&device, "nocopy_weight_stable", pointer, byte_length)
            .expect("second upload of the same name must hit the cache");
        assert_eq!(super::NOCOPY_BUFFER_REUSES.get(), reuses_before + 1);
        assert_eq!(nocopy_cache_len(), 1);
    }

    #[test]
    fn a_nocopy_name_rebound_to_a_different_host_pointer_is_rejected() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_nocopy_cache_for_test();

        let page = page_size();
        let first_buffer =
            AlignedBuffer::new(page / size_of::<f32>(), page).expect("page-aligned test buffer");
        let byte_length = first_buffer.len() * size_of::<f32>();
        upload_block_no_copy(
            &device,
            "nocopy_weight_rebound",
            first_buffer.as_ptr().cast(),
            byte_length,
        )
        .expect("first upload establishes the cached entry");

        // a second, unrelated page-aligned host allocation of the SAME
        // length reusing the SAME name -- ROW 334's own shape (a freed
        // ladder arm's `Vec<u8>` reused by a later arm at the identical
        // address), reproduced at the driver level instead of relying on a
        // test harness never marking an ephemeral buffer resident under a
        // reused name.
        let second_buffer =
            AlignedBuffer::new(page / size_of::<f32>(), page).expect("page-aligned test buffer");
        let reuses_before = super::NOCOPY_BUFFER_REUSES.get();
        let error = upload_block_no_copy(
            &device,
            "nocopy_weight_rebound",
            second_buffer.as_ptr().cast(),
            byte_length,
        )
        .expect_err("a different host pointer under the same name must never be served");

        assert!(matches!(
            error,
            super::MetalError::ResidentNameRebound { .. }
        ));
        assert_eq!(super::NOCOPY_BUFFER_REUSES.get(), reuses_before);
        assert_eq!(nocopy_cache_len(), 1);
    }

    #[test]
    fn a_nocopy_name_rebound_to_a_different_length_is_rejected() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_nocopy_cache_for_test();

        let page = page_size();
        let buffer = AlignedBuffer::new(2 * page / size_of::<f32>(), page)
            .expect("page-aligned test buffer");
        let pointer = buffer.as_ptr().cast();
        let full_length = buffer.len() * size_of::<f32>();
        upload_block_no_copy(&device, "nocopy_weight_grown", pointer, full_length)
            .expect("first upload establishes the cached entry");

        let reuses_before = super::NOCOPY_BUFFER_REUSES.get();
        let error = upload_block_no_copy(&device, "nocopy_weight_grown", pointer, full_length / 2)
            .expect_err("a different byte length under the same name must never be served");

        assert!(matches!(
            error,
            super::MetalError::ResidentNameRebound { .. }
        ));
        assert_eq!(super::NOCOPY_BUFFER_REUSES.get(), reuses_before);
        assert_eq!(nocopy_cache_len(), 1);
    }

    #[test]
    fn an_unnamed_upload_is_never_cached() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_nocopy_cache_for_test();

        let page = page_size();
        let buffer =
            AlignedBuffer::new(page / size_of::<f32>(), page).expect("page-aligned test buffer");
        let pointer = buffer.as_ptr().cast();
        let byte_length = buffer.len() * size_of::<f32>();

        upload_block_no_copy_uncached(&device, pointer, byte_length)
            .expect("first uncached upload");
        upload_block_no_copy_uncached(&device, pointer, byte_length)
            .expect("second uncached upload of the identical range");

        assert_eq!(
            nocopy_cache_len(),
            0,
            "an unnamed (uncached) upload must never populate NOCOPY_BUFFERS"
        );
    }

    /// ROW: `LoadedModel::drop` releases exactly the dropped checkpoint's
    /// own resident names -- a SECOND, unrelated model's own no-copy entry
    /// (a different name, same thread, both live at once, the two-model
    /// shape the caller's `Drop` impl exists for) must survive untouched.
    #[test]
    fn release_resident_names_evicts_only_the_named_entry() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_nocopy_cache_for_test();

        let page = page_size();
        let model_a = AlignedBuffer::new(page / size_of::<f32>(), page)
            .expect("page-aligned fixture for fake model A");
        let model_b = AlignedBuffer::new(page / size_of::<f32>(), page)
            .expect("page-aligned fixture for fake model B");
        let byte_length = model_a.len() * size_of::<f32>();

        upload_block_no_copy(
            &device,
            "fake_model_a.weight",
            model_a.as_ptr().cast(),
            byte_length,
        )
        .expect("bind fake model A's weight");
        upload_block_no_copy(
            &device,
            "fake_model_b.weight",
            model_b.as_ptr().cast(),
            byte_length,
        )
        .expect("bind fake model B's weight");
        assert_eq!(
            nocopy_cache_len(),
            2,
            "both fake models' weights are cached"
        );

        super::release_resident_names(["fake_model_a.weight"]);
        assert_eq!(
            nocopy_cache_len(),
            1,
            "releasing model A's own name must evict exactly one entry"
        );

        let reuses_before = super::NOCOPY_BUFFER_REUSES.get();
        upload_block_no_copy(
            &device,
            "fake_model_b.weight",
            model_b.as_ptr().cast(),
            byte_length,
        )
        .expect("model B's own name must still resolve after model A's release");
        assert!(
            super::NOCOPY_BUFFER_REUSES.get() >= reuses_before + 1,
            "model B's entry must still be a cache HIT -- it was never released"
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod cross_plan_resident_reuse_tests {
    //! ROW 369: a KV-bucket-boundary reshape builds a brand-new [`Plan`]
    //! whose own `device_buffers`/`block_identity` start empty. These prove
    //! [`cross_plan_resident_reuse`] -- the fast path that new `Plan` takes
    //! against the process-global caches -- resolves a still-resident block
    //! WITHOUT re-registering it as an "offer", and still refuses a genuine
    //! rebind. `execute_plan_with_placements`'s own per-node loop is the
    //! caller; these test the decision it delegates, GPU-free of any real
    //! `Plan`/program construction.

    use core::mem::size_of_val;

    use super::{
        PLAN_HANDOFF_REUSES, cross_plan_resident_reuse, device_and_queue,
        register_checkpoint_mapping, reset_nocopy_cache_for_test, upload_packed_bytes,
    };

    #[test]
    fn a_checkpoint_mapped_block_is_resolved_from_the_global_cache_without_an_offer() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_nocopy_cache_for_test();

        let checkpoint = vec![0_u8; super::page_size() * 4];
        register_checkpoint_mapping(&checkpoint);
        // one real upload registers `CHECKPOINT_MAPPING_NOCOPY_NAME` in
        // `NOCOPY_BUFFERS` -- the state a PRIOR `Plan`'s own weight walk
        // would have already left behind by the time a reshape builds a new
        // one.
        let tensor_bytes = &checkpoint[64..128];
        upload_packed_bytes(&device, tensor_bytes, None).expect("first checkpoint-backed upload");

        let handoffs_before = PLAN_HANDOFF_REUSES.get();
        let pointer = tensor_bytes.as_ptr().cast();
        let byte_length = tensor_bytes.len();
        let resolved = cross_plan_resident_reuse(None, pointer, byte_length)
            .expect("a resident block already in the checkpoint mapping must resolve")
            .expect("the checkpoint mapping is registered, so this must be Some");

        assert_eq!(resolved.1, 64, "the offset into the shared mapping buffer");
        assert_eq!(
            PLAN_HANDOFF_REUSES.get(),
            handoffs_before + 1,
            "a global-cache hit must be counted as a handoff, not an offer"
        );
    }

    #[test]
    fn a_host_range_the_global_caches_have_never_seen_is_not_resolved() {
        reset_nocopy_cache_for_test();
        let never_registered = vec![1.0_f32; 16];
        let byte_length = size_of_val(never_registered.as_slice());
        let resolved =
            cross_plan_resident_reuse(None, never_registered.as_ptr().cast(), byte_length)
                .expect("a miss is Ok(None), never an error");
        assert!(
            resolved.is_none(),
            "the caller must fall through to the normal offer path on a miss"
        );
    }

    #[test]
    fn a_resident_name_rebound_to_different_bytes_is_rejected_not_silently_reused() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        super::reset_resident_cache_for_test();

        let first_host_buffer = vec![8.0_f32; 4096];
        let byte_length = size_of_val(first_host_buffer.as_slice());
        super::upload_resident_copy(
            &device,
            "resident_weight_cross_plan_rebound",
            first_host_buffer.as_ptr().cast(),
            byte_length,
        )
        .expect("first upload establishes the cached entry a later plan could try to reuse");

        // a later `Plan` offering the SAME resident name against a
        // DIFFERENT host allocation -- the exact contract violation
        // `cross_plan_resident_reuse` must never paper over with a silent
        // reuse of the wrong buffer.
        let second_host_buffer = vec![9.0_f32; 4096];
        let error = cross_plan_resident_reuse(
            Some("resident_weight_cross_plan_rebound"),
            second_host_buffer.as_ptr().cast(),
            byte_length,
        )
        .expect_err("a rebind under the same name must never resolve, even from this fast path");
        assert!(matches!(
            error,
            super::MetalError::ResidentNameRebound { .. }
        ));
    }
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
pub static OUTPUT_BUFFER_ALLOCATED_BYTES: Counter =
    Counter::new("omega.metal.output_buffer.allocated_bytes");

/// Every initialization write into a plan-owned uniform buffer. These bytes
/// are immutable for the plan's life; a warm stable execution therefore
/// reports zero writes.
pub static PLAN_UNIFORM_WRITES: Counter = Counter::new("omega.metal.plan_uniforms.write");

/// [`DispatchType::Concurrent`]'s own census: every
/// `memoryBarrierWithScope(Buffers)` the private `HazardTracker` actually
/// emitted this step. Unconditional at declaration -- same convention as
/// [`OUTPUT_BUFFER_ALLOCATIONS`] above -- so it reads a stable 0 on
/// [`DispatchType::Serial`] rather than not existing at all.
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
    if std::env::var_os("PROXIMA_DEBUG_SEGMENT_HOST").is_some()
        && !device_buffers.contains_key(&node)
    {
        eprintln!(
            "metal missing operand node={node:?} available={:?}",
            device_buffers.keys().collect::<Vec<_>>()
        );
    }
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
    // `Some` only for a `CachedAttention` position under `ContextSplitMerge`
    // -- redesign §4c: the split kernel's `Binding::Scratch` slot binds
    // THIS buffer (never `output`, unlike `Binding::Output`), and the merge
    // kernel's own `Binding::Scratch` slot reads it back. `None` for every
    // other op, and for any call site that never resolved
    // `crate::metal::attention_scratch_buffer` -- `Binding::Scratch`
    // appearing with `scratch == None` is exactly the "caller lacks the
    // merge dispatch this binding shape requires" gap `encode_op`'s own doc
    // names, and is rejected there before this function is ever reached.
    scratch: Option<(&Retained<ProtocolObject<dyn MTLBuffer>>, usize)>,
    uniforms: &Retained<ProtocolObject<dyn MTLBuffer>>,
    fault: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
    expert_source_node: Option<NodeId>,
    expert_payloads: Option<(&MetalBuffer, usize)>,
    expert_descriptors: Option<&MetalBuffer>,
) -> Result<(), MetalError> {
    let (output_buffer, output_offset) = output;
    for (index, binding) in bindings.iter().enumerate() {
        let (buffer, offset) = match binding {
            Binding::Input(node) if expert_source_node.is_some_and(|source| source == *node) => {
                expert_payloads
                    .map(|(buffer, offset)| (buffer.clone(), offset))
                    .ok_or_else(|| MetalError::CompileFailed {
                        log: "kernel replaces an expert input but no payload buffer was supplied"
                            .to_string(),
                    })?
            }
            Binding::Input(node) | Binding::Indices(node) => {
                if std::env::var_os("PROXIMA_DEBUG_SEGMENT_HOST").is_some()
                    && !device_buffers.contains_key(node)
                {
                    eprintln!("metal binding missing node={node:?} bindings={bindings:?}");
                }
                buffer_for(device_buffers, *node)?
            }
            Binding::ExpertPayloads(_) => expert_payloads
                .map(|(buffer, offset)| (buffer.clone(), offset))
                .ok_or_else(|| MetalError::CompileFailed {
                    log: "kernel binds expert payloads but none were supplied".to_string(),
                })?,
            Binding::ExpertDescriptors(_) => (
                expert_descriptors
                    .cloned()
                    .ok_or_else(|| MetalError::CompileFailed {
                        log: "kernel binds expert descriptors but none were supplied".to_string(),
                    })?,
                0,
            ),
            Binding::Output(_) => (output_buffer.clone(), output_offset),
            Binding::Scratch => {
                let (buffer, offset) = scratch.ok_or_else(|| MetalError::CompileFailed {
                    log: "kernel binds a scratch buffer but none was resolved".to_string(),
                })?;
                (buffer.clone(), offset)
            }
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
/// This plan's own row count -- the largest `CachedAttention` `query_rows`
/// among `resolved`, or `1` when the plan has no attention op at all (every
/// non-attention op's own transient extents already scale with the same row
/// count, so the maximum over attention ops is the plan-wide M). Decode
/// dispatches one query row at a time (`query_rows == 1` everywhere), so this
/// returns `1` there and [`build_buffer_arena`]'s cap collapses to the
/// unscaled `ARENA_TRANSIENT_CAP` -- the constant's own decode-sized default
/// is preserved exactly. A prefill sharing ONE dispatch across `M` rows
/// (ROW 391's 1100-row interactive-chat prompt) reports `query_rows == M`,
/// so the cap scales up with it instead of being sized once for decode and
/// applied unchanged to every M.
#[cfg(feature = "metal-plan-stable-buffers")]
fn plan_query_rows(resolved: &[BoundOp]) -> u64 {
    resolved
        .iter()
        .filter_map(|bound| match &bound.kind {
            BoundOpKind::CachedAttention { query_rows, .. } => Some(*query_rows),
            _ => None,
        })
        .max()
        .unwrap_or(1)
}

/// Prints the naive (no-reuse) transient sum against this plan's own
/// [`plan_query_rows`]-scaled cap before allocating anything, per this
/// card's memory gate.
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
    let query_rows = plan_query_rows(resolved);
    let cap_bytes = ARENA_TRANSIENT_CAP.saturating_mul(query_rows.max(1) as usize);
    let device_limit = device.recommendedMaxWorkingSetSize();
    debug!(
        naive_transient_bytes = naive_transient_bytes as u64,
        uniform_bytes = uniform_bytes as u64,
        op_count = resolved.len() as u64,
        query_rows,
        arena_transient_cap = ARENA_TRANSIENT_CAP as u64,
        cap_bytes = cap_bytes as u64,
        device_limit,
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
    if peak_bytes > cap_bytes {
        proxima_telemetry::error!(
            peak_bytes,
            cap_bytes,
            query_rows,
            device_limit,
            "arena peak_bytes exceeds arena_transient_cap -- MG-3 kill condition"
        );
        return Err(MetalError::ArenaOverCap {
            peak_bytes,
            cap_bytes,
            query_rows,
            device_limit,
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
    numeric_policy: NumericPolicy,
) -> Result<PlanUniforms, MetalError> {
    let mut buffers = Vec::with_capacity(resolved.len());
    for bound in resolved {
        let bytes = pack_uniforms(bound, numeric_policy)?;
        let buffer = device
            .newBufferWithLength_options(bytes.len().max(1), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::CompileFailed {
                log: "device refused to allocate a plan uniform buffer".to_string(),
            })?;
        write_plan_uniform_bytes(&buffer, &bytes);
        #[cfg(feature = "instrument")]
        counter!(PLAN_UNIFORM_WRITES, 1);
        buffers.push(buffer);
    }
    Ok(PlanUniforms { buffers })
}

/// Initializes one plan-owned uniform buffer. The bytes are a pure function
/// of the resolved `BoundOp` and the plan's fixed numeric policy, so warm
/// execution binds this buffer unchanged instead of repacking and rewriting
/// it per dispatch.
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
fn arena_placement(
    plan: &Plan,
    position: usize,
) -> Result<Option<(&MetalBuffer, usize)>, MetalError> {
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
fn arena_placement(
    _plan: &Plan,
    _position: usize,
) -> Result<Option<(&MetalBuffer, usize)>, MetalError> {
    Ok(None)
}

/// [`encode_op`]'s plan-owned-uniform lookup -- see [`arena_placement`]'s
/// own doc for why this is a free function rather than an inline `#[cfg]`,
/// and for why it builds `plan.uniforms` lazily on the same schedule.
#[cfg(feature = "metal-plan-stable-buffers")]
fn plan_uniform_buffer(plan: &Plan, position: usize) -> Result<Option<&MetalBuffer>, MetalError> {
    if plan.uniforms.get().is_none() {
        let (device, _queue) = device_and_queue()?;
        let uniforms = build_plan_uniforms(&device, &plan.prepared.resolved, plan.numeric_policy)?;
        let _ = plan.uniforms.set(uniforms);
    }
    Ok(plan
        .uniforms
        .get()
        .map(|uniforms| &uniforms.buffers[position]))
}
#[cfg(not(feature = "metal-plan-stable-buffers"))]
fn plan_uniform_buffer(_plan: &Plan, _position: usize) -> Result<Option<&MetalBuffer>, MetalError> {
    Ok(None)
}

/// Element count (f32) [`Plan::attention_scratch`] reserves for one
/// `CachedAttention` position -- `query_rows * heads * splits *
/// (2 + head_dim)`, redesign §4c's own formula: `2` for the running
/// `(max, sum)` pair, `head_dim` for the un-normalized weighted-value
/// accumulator, one such record per split. `query_rows * heads`
/// is `resolved.extents.product() / head_dim` -- the SAME `total_elements`
/// [`crate::msl::grid_threads`]'s `CachedAttention` arm already derives.
/// `None` for every other op kind. Used both by [`Plan::attention_scratch`]'s
/// lazy builder and by [`encode_op`]'s own cold-path fallback allocation
/// (no `Plan` to own a buffer, so a fresh one is sized with this exact
/// formula every call -- the same "no placement, fresh `allocate_buffer`"
/// shape `output` itself already falls back to).
///
/// `splits` is the compiled MAXIMUM (`crate::sized::ATTENTION_SPLIT_MAX`)
/// only at `query_rows == 1` (decode: one query row per dispatch) -- there,
/// reserving the max up front is what lets a `Plan` grow its own live
/// context length token by token, all the way to `ATTENTION_SPLIT_MAX`
/// splits, without this buffer ever needing to resize mid-stream. At
/// `query_rows > 1` (prefill: every row in the batch sharing ONE dispatch)
/// that same per-row headroom multiplies `query_rows` times the FULL
/// compiled max, not the [`crate::msl::splits_for`] value this call's own
/// (fixed, already-known) `context_length` will ever actually dispatch --
/// production crash: a 900-row prefill at `head_dim=128`/`heads=32`
/// requested `900 * 32 * 32 * 130 * 4` bytes (~479 MB) PER LAYER against
/// that always-max headroom (~17 GB across a 36-layer forward), which no
/// device honors -- `allocate_buffer`'s `newBufferWithLength_options`
/// returned `None`, surfaced as `MetalError::CompileFailed` once
/// [`EncoderGuard`] stopped that `Err` from crashing the process outright.
/// Prefill's context length cannot grow after this call the way decode's
/// does, so there is nothing to protect against by over-reserving: sizing
/// against the real split count is exact, not merely smaller.
fn cached_attention_scratch_len(bound: &BoundOp, numeric_policy: NumericPolicy) -> Option<u64> {
    let BoundOpKind::CachedAttention {
        head_dim,
        query_rows,
        cached_key_rows,
        new_key_rows,
        ..
    } = &bound.kind
    else {
        return None;
    };
    let total_elements = bound
        .extents
        .iter()
        .product::<u64>()
        .checked_div(*head_dim)?;
    let splits = if *query_rows > 1 {
        crate::msl::splits_for(cached_key_rows + new_key_rows, numeric_policy)
    } else {
        crate::sized::ATTENTION_SPLIT_MAX
    };
    Some(total_elements * splits * (2 + head_dim))
}

/// [`Plan::attention_scratch`]'s lazy builder, built alongside [`PlanUniforms`]
/// on the same schedule -- see [`plan_uniform_buffer`]'s own doc for why a
/// free function rather than an inline `#[cfg]`.
#[cfg(feature = "metal-plan-stable-buffers")]
fn attention_scratch_buffer(
    plan: &Plan,
    position: usize,
) -> Result<Option<&MetalBuffer>, MetalError> {
    if plan.attention_scratch.get().is_none() {
        let (device, _queue) = device_and_queue()?;
        let mut buffers = Vec::with_capacity(plan.prepared.resolved.len());
        for bound in &plan.prepared.resolved {
            let buffer = match cached_attention_scratch_len(bound, plan.numeric_policy) {
                Some(elements) => {
                    Some(allocate_buffer(&device, elements as usize, DType::Float32)?)
                }
                None => None,
            };
            buffers.push(buffer);
        }
        let _ = plan.attention_scratch.set(buffers);
    }
    Ok(plan
        .attention_scratch
        .get()
        .and_then(|buffers| buffers[position].as_ref()))
}
#[cfg(not(feature = "metal-plan-stable-buffers"))]
fn attention_scratch_buffer(
    _plan: &Plan,
    _position: usize,
) -> Result<Option<&MetalBuffer>, MetalError> {
    Ok(None)
}

/// Builds `plan.resolved_steps` on its first call, or when
/// [`Plan::set_math_mode`] moved the compiled mode since the last build --
/// every later call for the SAME mode is a no-op. `numeric_policy` cannot
/// move post-construction ([`Plan::numeric_policy`]'s own doc), so
/// `math_mode` -- the one axis that can still narrow after [`plan`] --  is
/// the only staleness key this needs. Called once per
/// [`execute_plan_with_placements`] invocation, before that function's own
/// per-position loop, so a plan-cache HIT never pays [`kernel_cache_key`] or
/// [`kernel_dispatch_shape`] again: the loop below indexes
/// `plan.resolved_steps` by position instead.
fn resolve_steps(device: &ProtocolObject<dyn MTLDevice>, plan: &Plan) -> Result<(), MetalError> {
    let stale = plan
        .resolved_steps
        .borrow()
        .as_ref()
        .is_none_or(|resolved| resolved.math_mode != plan.math_mode);
    if !stale {
        return Ok(());
    }
    let mut steps = Vec::with_capacity(plan.prepared.resolved.len());
    for bound in &plan.prepared.resolved {
        let mut cache_key = kernel_cache_key(bound, &plan.packed_operands, plan.numeric_policy)?;
        cache_key.push(plan.math_mode.cache_token());
        let (bindings, grid) =
            kernel_dispatch_shape(bound, &plan.packed_operands, plan.numeric_policy)?;
        let pipeline = pipeline_for(
            device,
            bound,
            &plan.packed_operands,
            &cache_key,
            plan.math_mode,
            plan.numeric_policy,
        )?;
        // Redesign §4c: a `CachedAttention` position under a policy that
        // admits `ContextSplitMerge` resolves a SECOND pipeline for the
        // merge dispatch, keyed on the split's own cache key plus `_merge`
        // so the two never collide in `PIPELINE_CACHE` even though they
        // share every other structural token.
        let merge = match crate::msl::emit_cached_attention_merge(bound, plan.numeric_policy)? {
            Some(merge_kernel) => {
                let merge_cache_key = format!("{cache_key}_merge");
                let merge_pipeline =
                    pipeline_for_kernel(device, &merge_kernel, &merge_cache_key, plan.math_mode)?;
                Some(ResolvedMergeStep {
                    pipeline: merge_pipeline,
                    bindings: merge_kernel.bindings,
                    grid: merge_kernel.grid,
                })
            }
            None => None,
        };
        steps.push(ResolvedStep {
            pipeline,
            bindings,
            grid,
            merge,
        });
    }
    *plan.resolved_steps.borrow_mut() = Some(ResolvedSteps {
        math_mode: plan.math_mode,
        steps,
    });
    Ok(())
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
#[allow(clippy::too_many_arguments)]
fn encode_op(
    device: &ProtocolObject<dyn MTLDevice>,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    device_buffers: &mut BTreeMap<NodeId, DeviceBuffer>,
    bound: &BoundOp,
    packed_operands: &PackedOperands,
    placement: Option<(&MetalBuffer, usize)>,
    plan_uniform: Option<&MetalBuffer>,
    math_mode: MathMode,
    numeric_policy: NumericPolicy,
    resolved: Option<&ResolvedStep>,
    // Redesign §4c: `Some` when `crate::metal::attention_scratch_buffer`
    // already resolved a PLAN-OWNED scratch buffer for this position
    // (`execute_plan_with_placements`, the one call site with a `Plan` to
    // own it). `None` on every other call site (`execute_plan`, the
    // `*_op_timed` diagnostics) -- when `bindings` still needs one (a
    // `Binding::Scratch` slot), this function allocates a throwaway one
    // below, the same "no placement, fresh `allocate_buffer`" fallback
    // `output` itself already has.
    scratch: Option<(&MetalBuffer, usize)>,
    // `Some` only from `execute_plan_with_placements` under
    // `DispatchType::Concurrent` -- the plan's own per-call `HazardState`,
    // reborrowed. `None` everywhere else (`execute_plan`, `execute_op_timed`,
    // every `DispatchType::Serial` call), since program order alone already
    // orders a split's write before its merge's read there (see the
    // `dispatch(encoder, &pipeline, grid)` call site's own doc below).
    hazard: Option<&mut HazardTracker<*const ProtocolObject<dyn MTLBuffer>>>,
    expert_buffers: Option<&ExpertSourceBuffers>,
) -> Result<Option<(MetalBuffer, usize)>, MetalError> {
    let expert_source_node = match expert_buffers {
        None => None,
        Some(expert_buffers) => bound
            .operands()
            .iter()
            .any(|(node, _, _lookup)| *node == expert_buffers.node)
            .then_some(expert_buffers.node)
            .ok_or(MetalError::ExpertSourceUnsupported {
                node: expert_buffers.node,
                reason: "expert buffers were supplied to an operation that does not gather this source",
            })
            .map(Some)?,
    };
    // `cpu::run_node_into`'s own `BoundOpKind::CachedAttention` arm
    // (`proxima-tensor/src/cpu.rs:5210-5215`) is the only place
    // `instrument::record_op_kind` was ever called -- the CPU evaluator's
    // per-node dispatch. `encode_op` is `omega::metal`'s own per-op dispatch
    // (once per bound op in a plan, on the actual backend production
    // decode runs), and it never touched that counter: a Metal build's
    // `path_totals().op_kind_cached_attention` read 0 on every step
    // regardless of whether the fused kernel ran. Recording it here, at the
    // one point every `CachedAttention` op passes through regardless of
    // cache-hit/miss or placement, makes the interop's per-step
    // `cached_attention_ops` field truthful on the backend it actually
    // reports for.
    #[cfg(feature = "instrument")]
    if matches!(bound.kind, BoundOpKind::CachedAttention { .. }) {
        record_op_kind(OpKind::CachedAttention);
    }
    // `gather_count(bound) > 0` on an `Elementwise`/`Reduce` op is exactly
    // `spec::gathered_expert_product`'s shape once bound (a `Computed`
    // gather map on one operand feeding the multiply-then-reduce chain
    // `append_moe_ffn` builds) -- `embedding_lookup`'s own gather is a bare
    // `Elementwise` with no following reduce, so this also fires there, but
    // no test asserts this counter on that path; it exists so a MoE
    // fixture's Metal run can prove the gather engaged instead of silently
    // taking a non-gathering fallback ("default-on ≠ reachable").
    #[cfg(feature = "instrument")]
    if matches!(
        bound.kind,
        BoundOpKind::Elementwise { .. } | BoundOpKind::Reduce { .. }
    ) && gather_count(bound) > 0
    {
        record_op_kind(OpKind::GatheredExpert);
    }
    // `resolved` is `Some` only from `execute_plan_with_placements`, once
    // `resolve_steps` has run: this whole block -- `kernel_cache_key`'s
    // `String`, `kernel_dispatch_shape`'s `Vec<Binding>`, and
    // `pipeline_for`'s own `format!` cache-key lookup -- is skipped on
    // every step of a plan-cache HIT, not merely made cheaper. `None` on
    // every other call site (`execute_plan`, the `*_op_timed` diagnostics),
    // byte-identical to this function's behavior before `resolved` existed.
    #[cfg(feature = "instrument")]
    let emit_started = read_ticks();
    let owned_bindings: Vec<Binding>;
    let owned_merge: Option<ResolvedMergeStep>;
    let (pipeline, bindings, grid, merge) = if let Some(step) =
        resolved.filter(|_| expert_buffers.is_none())
    {
        (
            step.pipeline.clone(),
            step.bindings.as_slice(),
            step.grid,
            step.merge.as_ref(),
        )
    } else if let Some(source_node) = expert_source_node {
        let mut cache_key = kernel_cache_key(bound, packed_operands, numeric_policy)?;
        cache_key.push(math_mode.cache_token());
        let uniform_codec = expert_buffers.and_then(|buffers| {
            let mut codecs = buffers
                .descriptor_records
                .iter()
                .filter(|descriptor| descriptor.byte_length != 0)
                .map(|descriptor| descriptor.codec);
            let first = codecs.next()?;
            (packed_operands.get(&source_node) == Some(&first)
                && codecs.all(|codec| codec == first))
            .then_some(first)
        });
        if let Some(codec) = uniform_codec {
            cache_key.push_str("_uniform_expert_");
            cache_key.push_str(codec.cache_token());
            if std::env::var_os("PROXIMA_DEBUG_EXPERT_EMIT").is_some() {
                eprintln!("expert lowering mode=uniform codec={codec:?} node={source_node:?}");
            }
        } else {
            cache_key.push_str("_mixed_expert");
            if std::env::var_os("PROXIMA_DEBUG_EXPERT_EMIT").is_some() {
                eprintln!("expert lowering mode=mixed node={source_node:?}");
            }
        }
        let (binding_identity, _) = kernel_dispatch_shape(bound, packed_operands, numeric_policy)?;
        cache_key.push_str(&format!("_{binding_identity:?}"));
        let kernel = MIXED_KERNEL_CACHE.with(|cache| cache.borrow().get(&cache_key).cloned());
        let kernel = match kernel {
            Some(kernel) => kernel,
            None => {
                let kernel = if let Some(codec) = uniform_codec {
                    crate::msl::emit_with_uniform_expert_source(
                        bound,
                        packed_operands,
                        numeric_policy,
                        source_node,
                        codec,
                    )?
                } else {
                    crate::msl::emit_with_expert_sources(
                        bound,
                        packed_operands,
                        numeric_policy,
                        source_node,
                    )?
                };
                MIXED_KERNEL_CACHE.with(|cache| {
                    cache.borrow_mut().insert(cache_key.clone(), kernel.clone());
                });
                kernel
            }
        };
        let pipeline = pipeline_for_kernel(device, &kernel, &cache_key, math_mode)?;
        owned_bindings = kernel.bindings;
        (pipeline, owned_bindings.as_slice(), kernel.grid, None)
    } else {
        // `kernel_cache_key`/`kernel_dispatch_shape` are the cheap halves of
        // `emit`'s work -- structural fingerprint, bindings, grid -- with no
        // MSL body text rendered. On a pipeline-cache HIT (the steady-decode
        // case, `plan_hits`/`gpu_exec`'s own row) `emit` itself is never
        // called; only a genuine miss inside `pipeline_for` pays for the
        // full render + compile.
        let mut cache_key = kernel_cache_key(bound, packed_operands, numeric_policy)?;
        cache_key.push(math_mode.cache_token());
        let (bindings, grid) = kernel_dispatch_shape(bound, packed_operands, numeric_policy)?;
        #[cfg(feature = "instrument")]
        {
            counter!(EMIT_CALLS, 1);
            counter!(EMIT_TICKS, elapsed_ticks(emit_started));
        }
        #[cfg(feature = "instrument")]
        let pipeline_started = read_ticks();
        let pipeline = pipeline_for(
            device,
            bound,
            packed_operands,
            &cache_key,
            math_mode,
            numeric_policy,
        )?;
        #[cfg(feature = "instrument")]
        {
            counter!(PIPELINE_LOOKUP_CALLS, 1);
            counter!(PIPELINE_LOOKUP_TICKS, elapsed_ticks(pipeline_started));
        }
        // This cold path (`resolved: None`: `execute_plan`, the
        // `*_op_timed` diagnostics) has no `Plan` to own a scratch buffer
        // or a resolved merge pipeline -- it resolves both itself, here,
        // exactly like `pipeline_for`/`kernel_dispatch_shape` just above,
        // rather than caching them plan-side. `None` (every non-
        // `CachedAttention` op, or a `CachedAttention` op under a policy
        // that withholds `ContextSplitMerge`) costs nothing extra: `emit_
        // cached_attention_merge` returns `None` before rendering anything.
        owned_merge = match crate::msl::emit_cached_attention_merge(bound, numeric_policy)? {
            Some(merge_kernel) => {
                let merge_cache_key = format!("{cache_key}_merge");
                let merge_pipeline =
                    pipeline_for_kernel(device, &merge_kernel, &merge_cache_key, math_mode)?;
                Some(ResolvedMergeStep {
                    pipeline: merge_pipeline,
                    bindings: merge_kernel.bindings,
                    grid: merge_kernel.grid,
                })
            }
            None => None,
        };
        owned_bindings = bindings;
        (
            pipeline,
            owned_bindings.as_slice(),
            grid,
            owned_merge.as_ref(),
        )
    };
    #[cfg(feature = "instrument")]
    let op_setup_started = read_ticks();
    let (output, output_offset) = match placement {
        Some((buffer, offset)) => (buffer.clone(), offset),
        None => (
            allocate_buffer(device, bound_output_len(bound), bound.dtype)?,
            0,
        ),
    };
    // `Some` only from `execute_plan_with_placements` (a `Plan`-owned,
    // call-to-call-reused buffer via `attention_scratch_buffer`). Every
    // other caller with a merge dispatch to satisfy falls back to a fresh,
    // throwaway one, sized by the SAME formula the plan-owned path uses
    // (`cached_attention_scratch_len`) -- correctness first, plan-owned
    // reuse is that call site's own optimization, not a requirement this
    // function imposes on every caller.
    let owned_scratch: Option<MetalBuffer> = if scratch.is_none() && merge.is_some() {
        Some(allocate_buffer(
            device,
            cached_attention_scratch_len(bound, numeric_policy).unwrap_or(0) as usize,
            DType::Float32,
        )?)
    } else {
        None
    };
    let scratch: Option<(&MetalBuffer, usize)> = match &owned_scratch {
        Some(buffer) => Some((buffer, 0)),
        None => scratch,
    };
    let uniforms = match plan_uniform {
        Some(buffer) => buffer.clone(),
        None => upload_uniforms(device, &pack_uniforms(bound, numeric_policy)?)?,
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
    if let Err(err) = bind_buffers(
        encoder,
        bindings,
        device_buffers,
        (&output, output_offset),
        scratch,
        &uniforms,
        fault.as_ref(),
        expert_source_node,
        expert_buffers.map(|buffers| (&buffers.payloads, buffers.payload_offset)),
        expert_buffers.map(|buffers| &buffers.descriptors),
    ) {
        debug!(
            node = bound.node.0,
            kind = bound.kind.name(),
            ?bindings,
            resident = device_buffers.len() as u64,
            "encode_op failed binding this op's operands"
        );
        return Err(err);
    }
    dispatch(encoder, &pipeline, grid);
    // Redesign §4c: the split kernel above wrote its partial into `scratch`
    // (its own `Binding::Scratch` slot, never `output`); this second
    // dispatch reads it back and writes the position's REAL output --
    // `ggml_metal_op_flash_attn_ext`'s own one-op-two-dispatches shape,
    // `attention-kernel-design.md` §4c. Both land in the SAME encoder, in
    // program order: under `DispatchType::Serial` (this plan's default)
    // that ordering alone is Metal's own dependency guarantee, the same
    // reason this module's `HazardTracker` is never instantiated on that
    // path at all (`HazardTracker`'s own doc). Under `DispatchType::Concurrent`
    // program order is NOT a guarantee, so the split's write and the merge's
    // read of `scratch` are routed through `hazard` below -- the SAME
    // `HazardTracker` mechanism every other buffer's hazard uses, since
    // `Binding::Scratch` maps to its own buffer identity exactly like
    // `Binding::Output`/`Binding::Input` do (see that variant's own doc).
    if let Some(merge) = merge {
        if let (Some(tracker), Some((scratch_buffer, _))) = (hazard, scratch) {
            let scratch_pointer = Retained::as_ptr(scratch_buffer);
            let output_pointer = Retained::as_ptr(&output);
            // the split dispatch above just wrote `scratch` -- record it,
            // then ask the tracker whether the merge's own read needs a
            // barrier first (always true: a write is never already visible
            // to a later read without one). Same "record, check, barrier"
            // shape `hazard_step` gives every other buffer, applied here to
            // the one edge that is entirely internal to this op.
            tracker.record(&[], Some(scratch_pointer));
            if tracker.needs_barrier(&[scratch_pointer], None) {
                encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                counter!(BARRIERS_EMITTED, 1);
                // a barrier is a full flush, so `reset` is correct -- but
                // the caller's OWN `hazard_step`, before this op was ever
                // encoded, already recorded this op's real output as
                // written; restore that record so a later sibling op's own
                // hazard check still sees it, exactly as if this intra-op
                // edge had never touched the tracker.
                tracker.reset();
                tracker.record(&[], Some(output_pointer));
            }
        }
        let merge_uniforms = upload_uniforms(
            device,
            &pack_cached_attention_merge_uniforms(bound, numeric_policy)?,
        )?;
        encoder.setComputePipelineState(&merge.pipeline);
        bind_buffers(
            encoder,
            &merge.bindings,
            device_buffers,
            (&output, output_offset),
            scratch,
            &merge_uniforms,
            None,
            None,
            None,
            None,
        )?;
        dispatch(encoder, &merge.pipeline, merge.grid);
    }
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

/// [`read_back_as_device_f32`], writing into a caller-owned, reused `target`
/// instead of returning a fresh `Vec` -- [`finish`]'s `recycle` parameter
/// feeds this a buffer popped from [`execute_plan_with_placements`]'s own
/// pool (a PREVIOUS `Evaluated`'s storage, given back via
/// [`Evaluated::into_scratch`]) so the root output's warm-step read-back
/// allocates nothing (ROW 303's residual). `target.clear()` keeps its
/// capacity, so only a call whose output GREW past that capacity allocates.
fn read_back_as_device_f32_into(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    byte_offset: usize,
    element_count: usize,
    target: &mut Vec<f32>,
) {
    debug_assert!(
        buffer.length() >= byte_offset + element_count * size_of::<f32>(),
        "device buffer too small for a device-f32 read-back: buffer.length()={}, byte_offset={byte_offset}, element_count={element_count}",
        buffer.length(),
    );
    let pointer = buffer.contents();
    // SAFETY: see `read_back_as_device_f32`'s own doc -- identical sizing
    // guarantee, just writing into `target` instead of collecting into a
    // fresh `Vec`.
    let slice = unsafe {
        let base = pointer.as_ptr().cast::<u8>().add(byte_offset).cast::<f32>();
        core::slice::from_raw_parts(base, element_count)
    };
    target.clear();
    target.extend_from_slice(slice);
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

/// `plan` supplies every field this used to take individually (`program`,
/// `prepared.index_nodes`/`shapes`/`effective_outputs`/`root`) -- one
/// argument instead of five, under clippy's `too_many_arguments` with
/// `recycle` added by this landing.
fn finish(
    plan: &Plan,
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    caller_owned: &BTreeSet<NodeId>,
    // A buffer popped from `execute_plan_with_placements`'s own recycle
    // pool, if one was available -- reused ONLY for `root`'s read-back (the
    // one output a caller always requests, `Evaluated::root`'s own doc) and
    // only on the `read_back_as_device_f32`-shaped dtype path; every other
    // output, and a `Float16` root, still allocates fresh exactly as before
    // this landing.
    recycle: Option<Vec<f32>>,
) -> Result<Evaluated, MetalError> {
    let program = &plan.program;
    let index_nodes = &plan.prepared.index_nodes;
    let shapes = &plan.prepared.shapes;
    let effective_outputs = &plan.prepared.effective_outputs;
    let root = plan.prepared.root;
    let mut results = Vec::with_capacity(effective_outputs.len());
    let mut placed = BTreeSet::new();
    let mut recycle = recycle;
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
                let recyclable_dtype = matches!(
                    dtype,
                    DType::Float32
                        | DType::BFloat16
                        | DType::Bool
                        | DType::Int8
                        | DType::UInt8
                        | DType::Int32
                        | DType::UInt32
                );
                if *node == root
                    && recyclable_dtype
                    && let Some(mut target) = recycle.take()
                {
                    read_back_as_device_f32_into(
                        buffer,
                        *offset,
                        element_count(&shape),
                        &mut target,
                    );
                    target
                } else {
                    read_back(buffer, *offset, element_count(&shape), *node, dtype)?
                }
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
    Ok(Evaluated::from_parts_with_placed(
        root, results, None, placed,
    ))
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
            upload_packed_bytes(&device, tensor_bytes, None).expect("checkpoint-mapping upload");

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
        DType, Extent, IndexMap, NodeId, NumericPolicy, Op, QuantizedBlock, ScalarOp, append, cpu,
        projection,
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
            NumericPolicy::default(),
        )
        .expect("plans the chain once -- its arena and plan uniforms build lazily below");
        let pooled = execute_plan_with_placements(
            &resolved_plan,
            &[QuantizedBlock::Float32(&a), QuantizedBlock::Float32(&b)],
            &[],
            &[],
            &mut Vec::new(),
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
        let resolved_plan = plan(&program, &[], &blocks, &outputs, NumericPolicy::default())
            .expect("plans the three-stage chain");

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
        let resolved_plan = plan(&program, &[], &blocks, &outputs, NumericPolicy::default())
            .expect("plans the five-diamond program");
        arena_placement(&resolved_plan, 0).expect("builds the arena on first placement lookup");

        let arena = resolved_plan
            .arena
            .get()
            .expect("arena was just built above");
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
        let resolved_plan = plan(&program, &[], &blocks, &outputs, NumericPolicy::default())
            .expect("plans the pinned-output chain");
        arena_placement(&resolved_plan, 0).expect("builds the arena on first placement lookup");
        let arena = resolved_plan
            .arena
            .get()
            .expect("arena was just built above");

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
        let resolved_plan = plan(&program, &[], &blocks, &outputs, NumericPolicy::default())
            .expect("plans the size-mismatched chain");
        arena_placement(&resolved_plan, 0).expect("builds the arena on first placement lookup");
        let arena = resolved_plan
            .arena
            .get()
            .expect("arena was just built above");

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
        let resolved_plan = plan(&program, &[], &blocks, &outputs, NumericPolicy::default())
            .expect("plans the ten-stage chain");
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
        let resolved_plan = plan(&program, &[], &blocks, &outputs, NumericPolicy::default())
            .expect("plans the identical-uniform chain");
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
            super::pack_uniforms(stage_zero_bound, NumericPolicy::default())
                .expect("packs uniforms"),
            super::pack_uniforms(stage_two_bound, NumericPolicy::default())
                .expect("packs uniforms"),
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

/// ROW 327: [`prepare`] attributes each [`QuantizedBlock`] in `blocks` to
/// the node at the same position in [`block_node_ids`]'s output -- this
/// crate's documented positional contract (`execute`'s own doc: "`blocks`
/// binds `Op::Input` inputs positionally"). A caller whose `blocks` order
/// disagrees with its own program's declaration order used to have that
/// mismatch surface as an unrelated `NotLowerable` from
/// `reject_unsupported_gpu_dtype` (whichever node lost its packed
/// classification), because the per-node shape check ran AFTER the packed
/// classification and the dtype gate. These tests pin the fix: the shape
/// check now runs first, so a mismatch reports the exact node and its
/// element-count disagreement directly.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod block_node_attribution_tests {
    use alloc::vec;

    use proxima_tensor::{
        DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce,
        ReduceInit, ScalarOp, TensorError, append, projection,
    };

    use super::{MetalError, plan};

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
    /// vs. the 256 actually handed). Before the fix that second pair was
    /// never reached with a useful diagnostic: `packed_operands_of` and
    /// `reject_unsupported_gpu_dtype` ran first and rejected the weight node
    /// with `NotLowerable` (a dtype complaint, not an ordering one).
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

        let error = match plan(&program, &[], &blocks, &[], NumericPolicy::default()) {
            Ok(_) => panic!("misordered blocks must never silently plan"),
            Err(error) => error,
        };

        match error {
            MetalError::Tensor(TensorError::InputSizeMismatch {
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

        plan(&program, &[], &blocks, &[], NumericPolicy::default())
            .expect("declaration-ordered blocks must plan");
    }
}

/// [`HazardTracker`]'s pure dataflow logic, tested with plain `&str`
/// identities so no real Metal device is required -- the real driver path
/// (`execute_plan_with_placements`) instantiates the same type with
/// `Id = *const ProtocolObject<dyn MTLBuffer>`.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod hazard_tracker_tests {
    use std::collections::BTreeMap;

    use proxima_tensor::{
        Extent, IndexMap, Keep, NumericPolicy, Op, Reduce, ReduceInit, ScalarOp, append, bind,
        infer, projection,
    };

    use super::{
        Binding, DeviceBuffer, HazardTracker, MetalError, NodeId, PackedOperands, hazard_step,
        kernel_dispatch_shape, resolve_hazard_inputs,
    };
    use crate::msl::{hazard_read_nodes, hazard_write_node};

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
        assert_eq!(
            output2, output0,
            "degenerate gate: op2 must reuse op0's own identity"
        );
        let barrier2 = hazard_step(&mut hazards, &[], output2);

        let output3 = resolve_test_output(3, &placement_table, &mut next_fresh);
        let barrier3 = hazard_step(&mut hazards, &[], output3);

        assert_eq!(
            [barrier0, barrier1, barrier2, barrier3],
            [false, true, true, false],
            "barriers fire before op1 (RAW) and op2 (WAR), never before op0 or op3"
        );
    }

    /// Redesign §4c's own gap: `Binding::Scratch` is written by a
    /// `CachedAttention` split kernel and read by its merge kernel, an edge
    /// `encode_op`'s own doc names as internal to one op. Modeled here at
    /// the pure `HazardTracker` level (`encode_op`'s real fix threads the
    /// SAME `scratch`/`output` identities through this exact tracker): the
    /// split's write to `scratch`, a SIBLING op's write to an entirely
    /// unrelated buffer in between (the concurrent-dispatch case this
    /// tracker exists for -- two independent ops both queued before either
    /// finishes), then the merge's read of `scratch`. The sibling's own
    /// write must not need a barrier (it shares no identity with anything
    /// written or read so far) and must not erase `scratch`'s own
    /// written-since-last-barrier status (`record` only adds, `reset` is the
    /// only thing that clears, and the sibling never triggers one) -- so the
    /// merge's read still sees `scratch` as written and still barriers.
    #[test]
    fn a_sibling_op_between_the_split_write_and_the_merge_read_does_not_hide_the_scratch_raw_hazard()
     {
        let mut hazards: HazardTracker<&str> = HazardTracker::new();

        // the split kernel writes its partial into scratch.
        assert!(
            !hazard_step(&mut hazards, &[], "scratch"),
            "scratch is a fresh identity nothing has touched yet"
        );

        // an independent sibling op, queued in between under
        // `DispatchType::Concurrent`, touches a buffer with no relation to
        // `scratch` at all.
        assert!(
            !hazard_step(&mut hazards, &[], "sibling_out"),
            "an unrelated sibling write must not spuriously barrier"
        );

        // the merge kernel reads scratch back -- the RAW hazard the sibling
        // must not have hidden.
        assert!(
            hazard_step(&mut hazards, &["scratch"], "real_output"),
            "the merge's read of scratch must still see it as written, sibling or not"
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

    /// `raw -> y` (a dispatched `Identity`), then `weights -> reduced`, fused
    /// with `Add(reduced, y)` into ONE `BoundOpKind::Reduce` whose
    /// `epilogue_operands` reads `y` -- the exact decode-shaped census ROW
    /// 323 named: a fused reduce's epilogue reads a SIBLING's just-written
    /// output, not one of its own compute-step `operands()`. `extra_y_use`
    /// (a second, independent consumer of `y`) keeps `y` from being inlined
    /// away by ordinary elementwise fusion before the reduce-epilogue fold
    /// ever runs. Returns the plan's dispatched `BoundOp`s in plan order,
    /// `y`'s own node, the fused reduce's own node (the surviving `consumer`
    /// NodeId), and `PackedOperands::new()` (no packed/quantized operand
    /// here, so an empty table is exactly right -- same as [`emit`]'s own
    /// doctest).
    fn epilogue_reads_sibling_output_fixture()
    -> (Vec<super::BoundOp>, NodeId, NodeId, PackedOperands) {
        let mut program = Vec::new();
        let raw = append(
            &mut program,
            Op::Input {
                dtype: super::DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let identity = || IndexMap::Affine(projection(1, &[0]));
        let y = append(
            &mut program,
            Op::Elementwise {
                dtype: super::DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(raw, identity())],
                name: None,
            },
        );
        let weights = append(
            &mut program,
            Op::Input {
                dtype: super::DType::Float32,
                shape: alloc::vec![Extent::Static(8), Extent::Static(4)],
                name: None,
            },
        );
        let reduced = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: super::DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: weights,
                in_map: IndexMap::Affine(projection(2, &[0, 1])),
                out_map: IndexMap::Affine(projection(2, &[1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let consumer = append(
            &mut program,
            Op::Elementwise {
                dtype: super::DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(reduced, identity()), (y, identity())],
                name: None,
            },
        );
        // a second, independent consumer of `y` -- without it, elementwise
        // fusion inlines `y`'s trivial `Identity` body straight into
        // `consumer`'s own composed body before `reduce-epilogue-fusion` ever
        // runs, collapsing the whole program to one op and defeating the
        // fixture's own point (a SIBLING dispatch's buffer read via the
        // epilogue). A second use forces `y` to materialize as its own
        // dispatched node, same trick `reduce_then_residual_add_program`
        // (`proxima-tensor/src/bind.rs`) uses for `x`.
        let extra_y_use = append(
            &mut program,
            Op::Elementwise {
                dtype: super::DType::Float32,
                body: ScalarOp::Negate,
                operands: alloc::vec![(y, identity())],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("epilogue fixture infers");
        let resolved = bind(
            &program,
            &shapes,
            &[consumer, extra_y_use],
            NumericPolicy::default(),
        )
        .expect("epilogue fixture binds");
        assert_eq!(
            resolved.len(),
            3,
            "degenerate gate: `y`, `extra_y_use`, and the fused reduce must be the only \
             dispatched ops -- `reduce-epilogue-fusion` folded `consumer` into `reduced`'s own \
             epilogue, and `extra_y_use` keeps `y` from being inlined away entirely"
        );
        let fused = resolved
            .iter()
            .find(|bound| bound.node == consumer)
            .expect("the consumer's NodeId now names the fused reduce+epilogue op");
        assert!(
            matches!(
                fused.kind,
                super::BoundOpKind::Reduce { ref epilogue_operands, .. }
                    if epilogue_operands.iter().any(|(node, _, _)| *node == y)
            ),
            "degenerate gate: the fused reduce's epilogue must read `y`, not just `weights`, got {:?}",
            fused.kind
        );
        (resolved, y, consumer, PackedOperands::new())
    }

    /// `bindings()`'s `Binding::Input` entries come 1:1 from
    /// `BoundOp::all_read_sources()` (`msl::bindings`'s own doc), so deriving
    /// the hazard read set from `bindings` instead of a parallel
    /// `all_read_sources()` enumeration cannot change which barriers fire for
    /// a program that already has no missing binding -- proven here by
    /// running the tracker over the census fixture's own `bindings`, in plan
    /// order: `y` writes fresh (no prior hazard, no barrier); the fused
    /// reduce reads `y` via its epilogue, and `y` was written since the last
    /// barrier, so a barrier is required immediately before it -- exactly
    /// the RAW ROW 323 named; `extra_y_use`'s own read of `y` then sees a
    /// tracker the fused reduce's own barrier already reset, so it does not
    /// barrier again.
    #[test]
    fn hazard_walk_over_bindings_matches_the_decode_shaped_fixtures_expected_trace() {
        let (resolved, y_node, fused_node, packed_operands) =
            epilogue_reads_sibling_output_fixture();
        let mut hazards: HazardTracker<NodeId> = HazardTracker::new();

        let mut barrier_by_node = std::collections::BTreeMap::new();
        for bound in &resolved {
            let (bindings, _grid) =
                kernel_dispatch_shape(bound, &packed_operands, NumericPolicy::default())
                    .expect("fixture ops emit a shape");
            let reads: Vec<NodeId> = hazard_read_nodes(&bindings).collect();
            let write = hazard_write_node(&bindings).expect("every bound op writes one node");
            barrier_by_node.insert(bound.node, hazard_step(&mut hazards, &reads, write));
        }

        assert!(
            !barrier_by_node[&y_node],
            "y's own dispatch reads only a plain Input, never tracked as written -- no barrier"
        );
        assert!(
            barrier_by_node[&fused_node],
            "the fused reduce reads y (via its epilogue binding) after y was just written -- \
             this is the exact RAW ROW 323 named, now caught because the read set is derived \
             from the same bindings the encoder binds"
        );
    }

    /// The class fix, isolated from any real `BoundOp`/fusion machinery: a
    /// dispatch's `bindings` list gains a read a hand-written `operands()`-only
    /// enumeration would never have seen -- exactly the shape a future fusion
    /// rule (or any new `Binding` producer) could add. Deriving the hazard
    /// read set from `bindings` itself, rather than from a second, parallel
    /// operand table, means that read is barriered by CONSTRUCTION: there is
    /// no second call site left to forget to update.
    #[test]
    fn a_binding_list_gaining_a_read_is_barriered_with_no_second_call_site_to_update() {
        let mut hazards: HazardTracker<NodeId> = HazardTracker::new();
        let unrelated_input = NodeId(0);
        let producer_output = NodeId(1);

        // op0: writes `producer_output`, reading only its own unrelated input.
        let bindings0 = [
            Binding::Input(unrelated_input),
            Binding::Output(producer_output),
        ];
        let barrier0 = hazard_step(
            &mut hazards,
            &hazard_read_nodes(&bindings0).collect::<Vec<_>>(),
            hazard_write_node(&bindings0).expect("bindings0 names one output"),
        );
        assert!(
            !barrier0,
            "a fresh write with no prior hazard history never barriers"
        );

        // op1's own compute-step operand is `op1_own_operand`; its `bindings`
        // gain an EXTRA read of `producer_output` -- the slot a fused
        // epilogue (or any future `Binding` producer) would occupy, never
        // named by a bare `operands()` walk.
        let op1_own_operand = NodeId(2);
        let op1_output = NodeId(3);
        let bindings1 = [
            Binding::Input(op1_own_operand),
            Binding::Input(producer_output),
            Binding::Output(op1_output),
        ];
        let barrier1 = hazard_step(
            &mut hazards,
            &hazard_read_nodes(&bindings1).collect::<Vec<_>>(),
            hazard_write_node(&bindings1).expect("bindings1 names one output"),
        );

        assert!(
            barrier1,
            "a RAW hazard against a buffer just written must barrier even when the read arrived \
             through a binding slot no separate operand table names"
        );
    }
}

/// Production crash (2026-09-07): `-[_MTLCommandEncoder dealloc]: failed
/// assertion 'Command encoder released without endEncoding'`, caught here at
/// its true root instead of the assertion it manifested as. A pure-CPU
/// module -- [`cached_attention_scratch_len`] never touches a device -- so
/// these run on any host, without `metal-plan-stable-buffers` or a GPU.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod attention_scratch_len_tests {
    use alloc::vec;

    use proxima_tensor::{BoundOpKind, DType, Layout, NodeId, NumericPolicy};

    use super::{BoundOp, cached_attention_scratch_len};

    /// The real openchat decode shape's own dims (`omega/tests/
    /// row_376_cached_attention_batched.rs`'s `QUERY_HEADS`/`KV_HEADS`/
    /// `HEAD_DIM`), at `query_rows` and `cached_key_rows + new_key_rows` set
    /// by the caller -- one dispatch's worth of `CachedAttention`.
    fn openchat_shaped_bound(query_rows: u64, cached_key_rows: u64, new_key_rows: u64) -> BoundOp {
        const KV_HEADS: u64 = 8;
        const QUERY_GROUPS: u64 = 4;
        const HEAD_DIM: u64 = 128;
        let operands = (0..8)
            .map(|index| {
                (
                    NodeId(index),
                    Layout {
                        base: 0,
                        strides: vec![1].into(),
                    },
                    None,
                )
            })
            .collect();
        BoundOp {
            node: NodeId(8),
            dtype: DType::Float32,
            extents: vec![query_rows, KV_HEADS, QUERY_GROUPS, HEAD_DIM],
            kind: BoundOpKind::CachedAttention {
                operands,
                query_rows,
                cached_key_rows,
                new_key_rows,
                kv_heads: KV_HEADS,
                query_groups: QUERY_GROUPS,
                head_dim: HEAD_DIM,
                scale: 0.5,
                cached_lower_inclusive: i64::MIN,
                new_upper_inclusive: 0,
            },
        }
    }

    /// ROW: a decode dispatch (`query_rows == 1`) still reserves the
    /// COMPILED MAXIMUM split count, unchanged from before this fix -- a
    /// live decode's own context length grows one key per token, all the
    /// way to `ATTENTION_SPLIT_MAX` splits, and this buffer must never need
    /// to resize mid-stream to keep up with it.
    #[test]
    fn decode_dispatch_still_reserves_the_compiled_split_maximum() {
        let bound = openchat_shaped_bound(1, 0, 4096);
        let elements = cached_attention_scratch_len(&bound, NumericPolicy::llama_relaxed())
            .expect("a CachedAttention bound always yields a scratch length");
        let expected_heads = 8 * 4;
        let expected = expected_heads * crate::sized::ATTENTION_SPLIT_MAX * (2 + 128);
        assert_eq!(
            elements, expected,
            "decode (query_rows == 1) must keep reserving ATTENTION_SPLIT_MAX splits"
        );
    }

    /// ROW: the production crash. Before this fix, a 900-row prefill at
    /// this exact shape requested `900 * 32 * 32 * 130 * 4` bytes (~479 MB)
    /// of scratch for ONE `CachedAttention` op -- ~17 GB across a 36-layer
    /// forward, which no device honors. This test uses 1024 rows (the
    /// owner's own re-prove shape) and asserts the byte length actually
    /// requested now, plus the exact 4x reduction the real split count
    /// (`splits_for(1024, ..) == 8`, vs. the compiled max `32`) predicts --
    /// tying the byte count to the mechanism, not just a smaller number.
    #[test]
    fn prefill_dispatch_sizes_scratch_by_the_real_split_count_not_the_compiled_max() {
        let context_length = 1024;
        let bound = openchat_shaped_bound(1024, 0, context_length);
        let policy = NumericPolicy::llama_relaxed();

        let after_elements = cached_attention_scratch_len(&bound, policy)
            .expect("a CachedAttention bound always yields a scratch length");
        let after_bytes = after_elements * 4;

        let real_splits = crate::msl::splits_for(context_length, policy);
        let total_elements = 1024 * 8 * 4;
        let expected_after = total_elements * real_splits * (2 + 128);
        let before_elements = total_elements * crate::sized::ATTENTION_SPLIT_MAX * (2 + 128);
        let before_bytes = before_elements * 4;

        std::println!(
            "cached_attention_scratch_len: context_length={context_length} \
             real_splits={real_splits} compiled_max={} \
             before_bytes={before_bytes} after_bytes={after_bytes} \
             reduction={:.1}x",
            crate::sized::ATTENTION_SPLIT_MAX,
            before_bytes as f64 / after_bytes as f64,
        );

        assert_eq!(
            after_elements, expected_after,
            "prefill (query_rows > 1) must size scratch off the REAL split count"
        );
        assert!(
            real_splits < crate::sized::ATTENTION_SPLIT_MAX,
            "this shape is only a meaningful regression check if the real split count is \
             actually smaller than the compiled max -- otherwise the fix and the pre-fix \
             formula agree by coincidence, not by the mechanism this test asserts"
        );
        assert!(
            after_bytes < before_bytes,
            "the fix must always request fewer bytes than the pre-fix always-max formula: \
             before={before_bytes} after={after_bytes}"
        );
    }
}

/// Production crash (2026-09-07): `provider error: generate: arena
/// peak_bytes=984110552 exceeds arena_transient_cap=172812125` -- a
/// ~1100-row interactive-chat prefill rejected by a cap sized once for
/// decode (`query_rows == 1`) and never scaled for a prefill's own row
/// count. Pure CPU -- [`plan_query_rows`] never touches a device -- so
/// these run on any host, without a GPU.
#[cfg(all(test, feature = "metal-plan-stable-buffers"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod plan_query_rows_tests {
    use alloc::vec;

    use proxima_tensor::{BoundOpKind, DType, Layout, NodeId};

    use super::{ARENA_TRANSIENT_CAP, BoundOp, plan_query_rows};

    /// Same real openchat decode/prefill shape `attention_scratch_len_tests`'
    /// own `openchat_shaped_bound` builds -- one `CachedAttention` dispatch
    /// at the caller's own `query_rows`.
    fn cached_attention_bound(query_rows: u64) -> BoundOp {
        const KV_HEADS: u64 = 8;
        const QUERY_GROUPS: u64 = 4;
        const HEAD_DIM: u64 = 128;
        let operands = (0..8)
            .map(|index| {
                (
                    NodeId(index),
                    Layout {
                        base: 0,
                        strides: vec![1].into(),
                    },
                    None,
                )
            })
            .collect();
        BoundOp {
            node: NodeId(8),
            dtype: DType::Float32,
            extents: vec![query_rows, KV_HEADS, QUERY_GROUPS, HEAD_DIM],
            kind: BoundOpKind::CachedAttention {
                operands,
                query_rows,
                cached_key_rows: 0,
                new_key_rows: query_rows,
                kv_heads: KV_HEADS,
                query_groups: QUERY_GROUPS,
                head_dim: HEAD_DIM,
                scale: 0.5,
                cached_lower_inclusive: i64::MIN,
                new_upper_inclusive: 0,
            },
        }
    }

    /// A plan with no `CachedAttention` op at all (a pure elementwise/matmul
    /// program) has no row count to read off an op, so [`plan_query_rows`]
    /// falls back to `1` -- the cap this plan sees is the unscaled
    /// `ARENA_TRANSIENT_CAP`, exactly today's behavior.
    #[test]
    fn no_attention_op_defaults_to_one_row() {
        let elementwise = BoundOp {
            node: NodeId(0),
            dtype: DType::Float32,
            extents: vec![4096],
            kind: BoundOpKind::Elementwise {
                body: proxima_tensor::ComposedBody::leaf(proxima_tensor::ScalarOp::Identity),
                operands: vec![(
                    NodeId(1),
                    Layout {
                        base: 0,
                        strides: vec![1].into(),
                    },
                    None,
                )],
            },
        };
        assert_eq!(plan_query_rows(&[elementwise]), 1);
    }

    /// `query_rows` in {1 (decode), 31 (a short prefix), 1100 (ROW's own
    /// interactive-chat reproduction)} -- [`plan_query_rows`] reads the ONE
    /// `CachedAttention` op's own field back unchanged, and the derived cap
    /// scales linearly with it (decode's `query_rows == 1` reproduces the
    /// pre-fix constant exactly).
    #[test]
    fn cap_scales_linearly_with_the_plans_own_query_rows() {
        for query_rows in [1_u64, 31, 1100] {
            let resolved = [cached_attention_bound(query_rows)];
            let observed = plan_query_rows(&resolved);
            assert_eq!(
                observed, query_rows,
                "plan_query_rows must read the plan's own CachedAttention row count back exactly"
            );

            let cap_bytes = ARENA_TRANSIENT_CAP.saturating_mul(observed.max(1) as usize);
            let expected = ARENA_TRANSIENT_CAP * query_rows as usize;
            assert_eq!(
                cap_bytes, expected,
                "cap_bytes must be exactly ARENA_TRANSIENT_CAP * query_rows at query_rows={query_rows}"
            );
        }
    }

    /// ROW: the production shape itself -- at `query_rows=1100` the naive
    /// 984 MB peak from the incident report is now well inside the
    /// per-row-scaled cap, where the old fixed `ARENA_TRANSIENT_CAP`
    /// (172_812_125 bytes) rejected it outright.
    #[test]
    fn thousand_row_prefill_shape_fits_the_scaled_cap() {
        let query_rows = 1100_u64;
        let cap_bytes = ARENA_TRANSIENT_CAP.saturating_mul(query_rows as usize);
        let production_peak_bytes = 984_110_552_usize;
        assert!(
            production_peak_bytes < cap_bytes,
            "the ROW 391 incident's own peak_bytes must fit under the scaled cap: \
             peak_bytes={production_peak_bytes} cap_bytes={cap_bytes}"
        );
        assert!(
            production_peak_bytes > ARENA_TRANSIENT_CAP,
            "this reproduction is only meaningful if the unscaled constant alone \
             would have rejected it, matching the original incident"
        );
    }
}

#[cfg(test)]
mod expert_payload_descriptor_tests {
    use super::{
        ExpertPayloadDescriptor, MetalError, PackedCodec, expert_payload_descriptors,
        ordinary_block_uploads, pack_expert_payload_descriptors,
        reject_non_reducing_expert_staging, selected_expert_arena_descriptors,
        selected_expert_payloads,
    };
    use proxima_tensor::cpu::{ExpertEntry, ExpertPayloadSpan, ExpertSource, QuantizedBlock};
    use proxima_tensor::{DType, NodeId};

    #[test]
    fn mixed_hobbit_entries_keep_codec_and_byte_spans_separate() {
        let low_bytes = [0_u8; 84];
        let high_bytes = [0_u8; 144];
        let entries = [
            ExpertEntry {
                block: QuantizedBlock::Q2K(&low_bytes),
                out_dim: 256,
                in_dim: 256,
                epoch: 11,
            },
            ExpertEntry {
                block: QuantizedBlock::Q4K(&high_bytes),
                out_dim: 256,
                in_dim: 256,
                epoch: 12,
            },
        ];
        let source = ExpertSource::new(&entries);

        let descriptors = expert_payload_descriptors(NodeId(7), &source)
            .expect("packed HOBBIT entries produce descriptors");

        assert_eq!(
            descriptors,
            vec![
                ExpertPayloadDescriptor {
                    expert_index: 0,
                    codec: PackedCodec::Q2K,
                    byte_offset: 0,
                    byte_length: 84,
                    out_dim: 256,
                    in_dim: 256,
                    epoch: 11,
                },
                ExpertPayloadDescriptor {
                    expert_index: 1,
                    codec: PackedCodec::Q4K,
                    byte_offset: 84,
                    byte_length: 144,
                    out_dim: 256,
                    in_dim: 256,
                    epoch: 12,
                },
            ]
        );
        let packed = pack_expert_payload_descriptors(NodeId(7), &descriptors)
            .expect("Q2_K/Q4_K descriptors fit the Metal ABI");
        assert_eq!(&packed[0..4], &0u32.to_ne_bytes());
        assert_eq!(&packed[4..8], &1u32.to_ne_bytes());
        assert_eq!(&packed[32..36], &1u32.to_ne_bytes());
        assert_eq!(&packed[36..40], &2u32.to_ne_bytes());
    }

    #[test]
    fn selected_hobbit_entries_compact_payload_but_keep_original_ids() {
        let low_bytes = [1_u8; 84];
        let high_bytes = [2_u8; 144];
        let entries = [
            ExpertEntry {
                block: QuantizedBlock::Q2K(&low_bytes),
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Q4K(&high_bytes),
                out_dim: 256,
                in_dim: 256,
                epoch: 2,
            },
        ];
        let selected = [1_u32];
        let source = ExpertSource::with_selected_expert_ids(&entries, &selected);
        let (payload, descriptors) = selected_expert_payloads(NodeId(9), &source)
            .expect("selected packed entries produce a compact table");
        assert_eq!(payload.len(), high_bytes.len());
        assert_eq!(descriptors[0].byte_length, 0);
        assert_eq!(descriptors[1].expert_index, 1);
        assert_eq!(descriptors[1].byte_length, high_bytes.len());
    }

    #[test]
    fn selected_hobbit_q6k_entry_keeps_codec_and_descriptor_tag() {
        let low_bytes = [1_u8; 144];
        let high_bytes = [2_u8; 210];
        let entries = [
            ExpertEntry {
                block: QuantizedBlock::Q4K(&low_bytes),
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Q6K(&high_bytes),
                out_dim: 256,
                in_dim: 256,
                epoch: 2,
            },
        ];
        let selected = [1_u32];
        let source = ExpertSource::with_selected_expert_ids(&entries, &selected);
        let (payload, descriptors) = selected_expert_payloads(NodeId(10), &source)
            .expect("selected Q6_K expert produces a compact table");

        assert_eq!(payload, high_bytes);
        assert_eq!(descriptors[1].codec, PackedCodec::Q6K);
        assert_eq!(descriptors[1].byte_length, high_bytes.len());
        let packed = pack_expert_payload_descriptors(NodeId(10), &descriptors)
            .expect("Q6_K descriptor fits the Metal ABI");
        assert_eq!(&packed[36..40], &3u32.to_ne_bytes());
    }

    #[test]
    fn selected_hobbit_descriptor_offsets_follow_compact_payload_order() {
        let first = [1_u8; 144];
        let second = [2_u8; 210];
        let entries = [
            ExpertEntry {
                block: QuantizedBlock::Q4K(&first),
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Q6K(&second),
                out_dim: 256,
                in_dim: 256,
                epoch: 2,
            },
        ];
        let selected = [0_u32, 1_u32];
        let source = ExpertSource::with_selected_expert_ids(&entries, &selected);
        let (payload, descriptors) = selected_expert_payloads(NodeId(11), &source)
            .expect("selected mixed entries produce a compact payload");

        assert_eq!(payload.len(), first.len() + second.len());
        assert_eq!(descriptors[0].byte_offset, 0);
        assert_eq!(descriptors[1].byte_offset, first.len());
        assert_eq!(&payload[..first.len()], &first);
        assert_eq!(&payload[first.len()..], &second);
    }

    #[test]
    fn all_expert_arena_descriptors_keep_mmap_relative_offsets() {
        let arena = [0_u8; 256];
        let entries = [
            ExpertEntry {
                block: QuantizedBlock::Q2K(&arena[7..91]),
                out_dim: 256,
                in_dim: 256,
                epoch: 3,
            },
            ExpertEntry {
                block: QuantizedBlock::Q2K(&arena[139..223]),
                out_dim: 256,
                in_dim: 256,
                epoch: 4,
            },
        ];
        let spans = [
            Some(ExpertPayloadSpan {
                offset: 7,
                length: 84,
            }),
            Some(ExpertPayloadSpan {
                offset: 139,
                length: 84,
            }),
        ];
        let source = ExpertSource::with_all_expert_arena(&entries, &arena, &spans)
            .expect("the dense packed arena validates");

        let descriptors = selected_expert_arena_descriptors(
            NodeId(13),
            &source,
            source.packed_arena().expect("the source carries its arena"),
        )
        .expect("all expert spans lower without a selected route list");

        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors[0].expert_index, 0);
        assert_eq!(descriptors[0].byte_offset, 7);
        assert_eq!(descriptors[0].byte_length, 84);
        assert_eq!(descriptors[1].expert_index, 1);
        assert_eq!(descriptors[1].byte_offset, 139);
        assert_eq!(descriptors[1].byte_length, 84);
    }

    #[test]
    fn selected_hobbit_preserves_sparse_original_expert_indices() {
        let first = [1_u8; 144];
        let second = [2_u8; 144];
        let third = [3_u8; 144];
        let entries = [
            ExpertEntry {
                block: QuantizedBlock::Q4K(&first),
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Q4K(&second),
                out_dim: 256,
                in_dim: 256,
                epoch: 2,
            },
            ExpertEntry {
                block: QuantizedBlock::Q4K(&third),
                out_dim: 256,
                in_dim: 256,
                epoch: 3,
            },
        ];
        let selected = [2_u32];
        let source = ExpertSource::with_selected_expert_ids(&entries, &selected);
        let (payload, descriptors) = selected_expert_payloads(NodeId(12), &source)
            .expect("a sparse route keeps the original expert index");

        assert_eq!(payload, vec![3_u8; 144]);
        assert_eq!(descriptors[0].byte_length, 0);
        assert_eq!(descriptors[1].byte_length, 0);
        assert_eq!(descriptors[2].expert_index, 2);
        assert_eq!(descriptors[2].byte_offset, 0);
        assert_eq!(descriptors[2].byte_length, 144);
    }

    #[test]
    fn mixed_hobbit_entries_accept_q3k_runtime_decoder() {
        let q3_bytes = [0_u8; 110];
        let entries = [ExpertEntry {
            block: QuantizedBlock::Q3K(&q3_bytes),
            out_dim: 256,
            in_dim: 256,
            epoch: 0,
        }];
        let source = ExpertSource::new(&entries);

        let descriptors = expert_payload_descriptors(NodeId(9), &source)
            .expect("Q3_K has a mixed-expert MSL decoder");
        assert_eq!(descriptors[0].codec, PackedCodec::Q3K);
        assert_eq!(descriptors[0].byte_length, q3_bytes.len());
        let packed = pack_expert_payload_descriptors(NodeId(9), &descriptors)
            .expect("Q3_K descriptor fits the Metal ABI");
        assert_eq!(&packed[4..8], &4u32.to_ne_bytes());
    }

    #[test]
    fn substituted_expert_stack_is_absent_from_ordinary_uploads() {
        let full_expert_stack = [0_u8; 4_096];
        let activation = [0.0_f32; 16];
        let blocks = [
            QuantizedBlock::Q4K(&full_expert_stack),
            QuantizedBlock::Float32(&activation),
        ];
        let ordinary = ordinary_block_uploads(
            &[NodeId(41), NodeId(42)],
            &blocks,
            &[DType::UInt8, DType::Float32],
            |node| node == NodeId(41),
        )
        .collect::<Vec<_>>();

        assert_eq!(
            ordinary
                .iter()
                .map(|(node, _, _)| *node)
                .collect::<Vec<_>>(),
            vec![NodeId(42)],
            "an ExpertSource replacement must remove the original expert node from ordinary upload"
        );
        assert_eq!(
            ordinary
                .iter()
                .map(|(_, block, _)| match block {
                    QuantizedBlock::Float32(values) => core::mem::size_of_val(*values),
                    QuantizedBlock::Q4K(bytes) => bytes.len(),
                    _ => 0,
                })
                .sum::<usize>(),
            core::mem::size_of_val(&activation),
            "ordinary uploads retain unrelated input bytes but none of the full expert stack"
        );
    }

    #[test]
    fn transient_staging_rejects_a_full_size_expert_stack_before_upload() {
        let original_bytes = [0_u8; 288];
        let first_expert = [0_u8; 144];
        let second_expert = [0_u8; 144];
        let entries = [
            ExpertEntry {
                block: QuantizedBlock::Q4K(&first_expert),
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Q4K(&second_expert),
                out_dim: 256,
                in_dim: 256,
                epoch: 2,
            },
        ];
        let source = ExpertSource::new(&entries);

        let error = reject_non_reducing_expert_staging(
            NodeId(51),
            QuantizedBlock::Q4K(&original_bytes),
            &source,
        )
        .expect_err("a full-size staging table is not a low-memory substitution");

        assert!(matches!(
            error,
            MetalError::ExpertSourceUnsupported {
                node: NodeId(51),
                reason: "transient expert staging must reduce checkpoint-resident bytes",
            }
        ));
    }

    #[test]
    fn transient_staging_accepts_a_smaller_mixed_codec_table() {
        let original_bytes = [0_u8; 288];
        let low_expert = [0_u8; 84];
        let high_expert = [0_u8; 144];
        let entries = [
            ExpertEntry {
                block: QuantizedBlock::Q2K(&low_expert),
                out_dim: 256,
                in_dim: 256,
                epoch: 1,
            },
            ExpertEntry {
                block: QuantizedBlock::Q4K(&high_expert),
                out_dim: 256,
                in_dim: 256,
                epoch: 2,
            },
        ];
        let source = ExpertSource::new(&entries);

        reject_non_reducing_expert_staging(
            NodeId(52),
            QuantizedBlock::Q4K(&original_bytes),
            &source,
        )
        .expect("a smaller mixed-codec table reduces the device upload");
    }
}
