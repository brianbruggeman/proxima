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
#[cfg(any(feature = "metal-buffer-pool", feature = "metal-horizontal-merge"))]
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    QuantizedBlock, Shapes, TensorError, bind_with_fusion, block_node_ids,
    correct_packed_matmul_layouts, index_node_ids, infer, node_retirement, prune_dead,
    resolve_named_blocks,
};
#[cfg(any(not(feature = "metal-buffer-pool"), test))]
use proxima_tensor::node_last_reader;

use crate::error::EmitError;
#[cfg(feature = "instrument")]
use crate::msl::diagnose_packed_row_block;
use crate::msl::{gather_count, kernel_cache_key, kernel_dispatch_shape, reduction_dims};
#[cfg(feature = "metal-plan-stable-buffers")]
use crate::sized::ARENA_TRANSIENT_CAP;
#[cfg(feature = "metal-buffer-pool")]
use crate::sized::OUTPUT_POOL_MAX_PER_BUCKET;
use crate::{Binding, GridSpec, Kernel, Codec, PackedOperands, emit};


#[macro_use]
mod device_buffers_arena_plan;
#[macro_use]
mod execute_and_hazards;
#[macro_use]
mod placements_execute_named;
#[macro_use]
mod dispatch_timed_and_classify;
#[macro_use]
mod prepare_uniforms_pack;
#[macro_use]
mod pipeline_buffers_upload;
#[macro_use]
mod resident_nocopy_cache;
#[macro_use]
mod arena_encode_dispatch_finish;
pub use device_buffers_arena_plan::*;
pub use execute_and_hazards::*;
pub use placements_execute_named::*;
#[cfg(feature = "instrument")]
pub use dispatch_timed_and_classify::*;
use prepare_uniforms_pack::*;
pub use pipeline_buffers_upload::*;
pub use resident_nocopy_cache::*;
use arena_encode_dispatch_finish::*;

/// Public wrapper around [`prepare_uniforms_pack::pack_uniforms`] (crate-private) --
/// the same packer [`execute`] uses per dispatch, exposed so a caller outside
/// this module (a probe/dump binary) can pack the exact bytes a real
/// dispatch would upload without duplicating the per-`BoundOpKind` match.
pub fn pack_uniforms_for(bound: &BoundOp, numeric_policy: NumericPolicy) -> Result<Vec<u8>, EmitError> {
    prepare_uniforms_pack::pack_uniforms(bound, numeric_policy)
}
