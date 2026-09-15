//! A CPU interpreter for [`BoundOp`] nodes: strided, f32-only, streaming its
//! buffers.
//!
//! This module owns none of the stride arithmetic — that lives in
//! [`mod@bind`], shared with any other backend. What is
//! CPU-specific and lives here: the f32-only restriction (a v1 limitation —
//! [`ScalarOp`]'s transcendental bodies need `libm`-grade math this crate
//! does not depend on, so a GPU backend targeting f16/bf16 natively is
//! unaffected by this choice), the loop nests that walk a `BoundOp` node's
//! iteration space, and buffer lifetime.
//!
//! `reject_non_float32`'s one exception is a gather's `indices` buffer: it
//! carries integer index values, but as f32 like everything else here, since
//! no separate integer-buffer kind exists yet. f32 represents every integer
//! up to `2^24` (16,777,216) exactly, so [`shape::infer`] rejects any
//! gathered axis wider than that before this module ever sees the program —
//! see [`crate::map::IndexMap::Computed`]'s docs for the full accounting.
//!
//! The inner loop of every walk below is a straight loop with a per-operand
//! running offset incremented by a precomputed stride each step — never a
//! per-element recomputation of the full coordinate — so the shape an
//! optimizing compiler needs to autovectorize is actually on the page.
//! [`crate::bind::BoundOp`] documents the one fusion decided ahead of
//! execution (`Reduce(Elementwise)` skipping the elementwise op's
//! O(iteration space) intermediate); this module additionally drops each
//! node's buffer the moment nothing in the emitted node sequence reads it
//! again, which is the other half of not paying for what a program does not
//! keep — see [`Evaluated::peak_live_buffers`].
//!
//! [`Interpreter`] is this module's [`Pipe`]
//! impl: `In = Vec<BoundOp>`, `Out = ()`. Its interior state is the buffer
//! table — caller-provided scratch borrowed for `Interpreter`'s lifetime,
//! exactly the same interior-mutability idiom [`shape::ShapeTable`] applies
//! to its resolved shapes and [`crate::bind::BoundOpBuilder`] applies to its
//! held elementwise ops. `In` is a batch because
//! [`crate::bind::BoundOpBuilder::push`] can ready zero, one, or two
//! [`BoundOp`] nodes per `Op` it is handed (its own doc: "may return more
//! than one" — flushing a previously-held elementwise op that turns out not
//! to fuse, alongside the current op's own node): `Interpreter` absorbing
//! that batch in one `call`, rather than the caller unpacking it into a
//! loop of single-record calls, is what lets the full three-stage chain
//! `shapes.and_then(builder).and_then(interpreter)` compose through
//! `AndThen` directly — `Second::In = First::Out` holds by construction
//! (`BoundOpBuilder::Out = Vec<BoundOp> = Interpreter::In`), no adapter, no
//! new type. `Interpreter::call` folds the batch internally the same way
//! the buffer table itself already folds per-node writes; a zero-element
//! batch is a no-op call, not a special case.
//! `run_node_into` is the primitive `Interpreter::call` (and
//! [`evaluate`]/[`evaluate_parallel`]'s own loops) all drive — it writes
//! into a caller-provided slice instead of allocating one, which is what
//! lets `Interpreter` reach into a no-alloc-at-the-write-site tier.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vfmaq_n_f32, vld1q_f32, vst1q_f32,
};
// `dot_q4k_q8k_block_neon_dotprod`'s own intrinsics -- a separate `use`
// block (rather than folded into the one above) so a default build (the
// `q4k-int8-dot` feature off) never imports symbols nothing references,
// which `-D unused-imports` (workspace lint) would otherwise reject. Shared
// with `dot_q5k_q8k_block_neon_dotprod`/`dot_q6k_q8k_block_neon_dotprod`
// (same intrinsics, same reasoning), so this gate covers all three int8-dot
// features rather than duplicating the `use` per format.
#[cfg(all(
    target_arch = "aarch64",
    any(
        feature = "q4k-int8-dot",
        feature = "q5k-int8-dot",
        feature = "q6k-int8-dot"
    )
))]
use core::arch::aarch64::{
    vaddvq_s32, vandq_u8, vdupq_n_s32, vdupq_n_u8, vld1q_s8, vld1q_u8, vreinterpretq_s8_u8,
    vshrq_n_u8,
};
// `dot_q4k_q8k_block_neon_dotprod`'s two-register paired loads
// (`ld1 {v,v}`) matching ggml's `ggml_vld1q_u8_x2`/`ggml_vld1q_s8_x2`
// (`arch/arm/quants.c:2408-2427`) -- one instruction issuing both halves
// of a 32-byte `q4`/`q8` chunk instead of two single-register `ldur`s.
#[cfg(all(target_arch = "aarch64", feature = "q4k-int8-dot"))]
use core::arch::aarch64::{vld1q_s8_x2, vld1q_u8_x2};
// `dot_q4k_q8k_block_neon_dotprod`'s mins-correction path: unpack the 6-bit
// scale/min codes once per super-block (`vld1_u32`/`vreinterpret_u8_u32`/
// `vmovl_u8`), then reduce `bsums . mins` with pairwise-add + widening
// multiply (`vpaddq_s16`/`vmull_s16`/`vget_low_s16`/`vget_high_s16`/
// `vaddq_s32`) instead of the auto-vectorized scalar loop this replaced.
#[cfg(all(target_arch = "aarch64", feature = "q4k-int8-dot"))]
use core::arch::aarch64::{
    vaddq_s32, vget_high_s16, vget_low_s16, vld1_u32, vld1q_s16, vmovl_u8, vmull_s16, vpaddq_s16,
    vreinterpret_u8_u32, vreinterpretq_s16_u16,
};
// `dot_q5k_q8k_block_neon_dotprod`/`dot_q6k_q8k_block_neon_dotprod`'s extra
// intrinsics beyond the `Q4_K` set above -- both need to OR a shifted
// high-bit plane into the low nibble, which `Q4_K` (no high-bit plane at
// all) never does.
#[cfg(all(
    target_arch = "aarch64",
    any(feature = "q5k-int8-dot", feature = "q6k-int8-dot")
))]
use core::arch::aarch64::{vorrq_u8, vshlq_n_u8};
// `dot_q6k_q8k_block_neon_dotprod`'s own extra intrinsics: `Q6_K`'s levels
// are biased by -32 (`x = d*sc*(q-32)`, `q6_k.rs`'s own module doc) before
// the dot, unlike `Q4_K`/`Q5_K` (unsigned nibble, no bias) -- `vsubq_s8`
// applies that bias in-register, `vdupq_n_s8` builds the constant it
// subtracts.
#[cfg(all(target_arch = "aarch64", feature = "q6k-int8-dot"))]
use core::arch::aarch64::{vdupq_n_s8, vsubq_s8};
// `dot_q4k_q8k_block_avx2`'s own intrinsics -- the x86 sibling of the
// aarch64 `use` block above, same reasoning: a separate cfg-gated block so
// a default build never imports symbols nothing references. Gated on
// `target_arch = "x86_64"` alone, NOT `q4k_avx2`: the kernel itself must
// compile on every x86_64 build (runtime dispatch needs it present
// regardless of `-C target-feature=+avx2`), the `#[target_feature(enable =
// "avx2")]` on the functions below is what keeps the actual instructions
// gated, not this import.
use core::any::TypeId;
#[cfg(all(target_arch = "x86_64", feature = "q4k-int8-dot"))]
use core::arch::x86_64::{
    __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_unpackhi_epi64,
    _mm256_and_si256, _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256,
    _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi8, _mm256_set1_epi16,
    _mm256_srli_epi16,
};
use core::cell::RefCell;
use core::future::Future;
use core::num::NonZeroUsize;
use core::ops::Deref;
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
use core::sync::atomic::AtomicU64;
// `StagedRound` (production once `cohort-staged-graph` is on, test-only
// scaffolding otherwise) plus `evaluate_parallel`'s ordering tests use
// `Ordering` regardless of `target_arch`/`instrument`, unlike `AtomicU64`
// above which only backs the aarch64+instrument tile counters -- gated on
// `any(test, .., feature = "cohort-staged-graph")` rather than left
// unconditional so a non-test, non-aarch64-instrument, feature-off build
// (nothing left to use it) does not pick up an unused-import warning under
// this workspace's deny(warnings).
#[cfg(any(
    test,
    all(target_arch = "aarch64", feature = "instrument"),
    feature = "cohort-staged-graph"
))]
use core::sync::atomic::Ordering;
#[cfg(feature = "epilogue-profile-probe")]
use core::sync::atomic::{
    AtomicU64 as EpilogueProfileAtomicU64, Ordering as EpilogueProfileOrdering,
};
// default-on since the ROW 186 promotion (`docs/discipline.md`) -- the
// counters and the `AtomicBool` bench/test escape valve both need this
// import unconditionally now, not only under a probe feature.
use core::sync::atomic::{
    AtomicBool as EpilogueFuseAtomicBool, AtomicU64 as EpilogueFuseAtomicU64,
    Ordering as EpilogueFuseOrdering,
};
use std::borrow::Cow;
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread;

use prime::os::background::ProximaBackgroundPool;
use prime::os::cohort::{ChunkIndex, CohortRound, CohortSession, ThreadCohort};
use proxima_primitives::block_on;
use proxima_primitives::pipe::Pipe;
use proxima_primitives::pipe::fan_in::Quorum;
#[cfg(feature = "instrument")]
use proxima_telemetry::counter;
#[cfg(feature = "instrument")]
use proxima_telemetry::debug;

use half::{bf16, f16};

// aliased against `half::{bf16, f16}` above: these are the per-codec on-disk
// modules `QuantizedBlock::element_count` composes down to, not the scalar
// float types.
use proxima_gguf::quant::{
    bf16 as gguf_bf16, f16 as gguf_f16, iq2_xs, iq3_xxs, iq4_nl, q2_k, q3_k, q4_0, q4_k, q5_1,
    q5_k, q6_k, q8_0,
};

use crate::bind::{
    self, BoundOp, BoundOpKind, ComposedBody, ReadyBatch, StepArg, block_node_ids,
    dead_resolved_nodes, index_node_ids, node_retirement, push_indices_node,
};
use crate::convert::{Convert, SimdConvert};
use crate::dtype::DType;
use crate::error::TensorError;
#[cfg(feature = "instrument")]
use crate::instrument;
#[cfg(feature = "instrument")]
use crate::instrument::{KernelCounters, Path};
use crate::map::IndexMap;
use crate::numeric::NumericPolicy;
use crate::op::{Keep, NodeId, Op, ReduceInit, ScalarOp};
use crate::shape;
use crate::sized::COHORT_SPIN_POLLS;


#[macro_use]
mod arena;
#[macro_use]
mod epilogue;
#[macro_use]
mod quantized_eval;
#[macro_use]
mod run_node;
#[macro_use]
mod elementwise_matmul;
#[macro_use]
mod run_reduce_scan;
#[macro_use]
mod gemm_tile;
#[macro_use]
mod gemm_dot_quant;
#[macro_use]
mod gemm_q8k;
#[macro_use]
mod width_kernels;
#[macro_use]
mod typed_eval;
pub use arena::*;
pub use epilogue::*;
pub use quantized_eval::*;
pub use run_node::*;
use elementwise_matmul::*;
pub use run_reduce_scan::*;
pub use gemm_tile::*;
pub use gemm_dot_quant::*;
pub use gemm_q8k::*;
use width_kernels::*;
pub use typed_eval::*;

#[cfg(test)]
mod tests;
