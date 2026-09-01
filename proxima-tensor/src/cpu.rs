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
    any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot")
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
    vaddq_s32, vget_high_s16, vget_low_s16, vld1_u32, vld1q_s16, vmovl_u8, vmull_s16,
    vpaddq_s16, vreinterpret_u8_u32, vreinterpretq_s16_u16,
};
// `dot_q5k_q8k_block_neon_dotprod`/`dot_q6k_q8k_block_neon_dotprod`'s extra
// intrinsics beyond the `Q4_K` set above -- both need to OR a shifted
// high-bit plane into the low nibble, which `Q4_K` (no high-bit plane at
// all) never does.
#[cfg(all(target_arch = "aarch64", any(feature = "q5k-int8-dot", feature = "q6k-int8-dot")))]
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
#[cfg(all(target_arch = "x86_64", feature = "q4k-int8-dot"))]
use core::arch::x86_64::{
    __m256i, _mm256_and_si256, _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256,
    _mm256_maddubs_epi16, _mm256_madd_epi16, _mm256_set1_epi16, _mm256_set1_epi8, _mm256_srli_epi16,
    _mm_add_epi32, _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_unpackhi_epi64,
};
use core::any::TypeId;
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
use std::borrow::Cow;
use std::thread;
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, OnceLock};

use proxima_primitives::block_on;
use proxima_primitives::pipe::Pipe;
use proxima_primitives::pipe::fan_in::Quorum;
#[cfg(feature = "instrument")]
use proxima_telemetry::counter;
use prime::os::background::ProximaBackgroundPool;
use prime::os::cohort::{ChunkIndex, CohortRound, CohortSession, ThreadCohort};

type MatmulCohort = ThreadCohort<TensorError>;
type MatmulSession<'a> = CohortSession<'a, TensorError>;

use half::{bf16, f16};

use crate::bind::{self, BoundOp, BoundOpKind, ComposedBody, ReadyBatch, StepArg};
use crate::convert::{Convert, SimdConvert};
use crate::dtype::DType;
use crate::error::TensorError;
#[cfg(feature = "instrument")]
use crate::instrument;
#[cfg(feature = "instrument")]
use crate::instrument::{KernelCounters, Path};
use crate::map::IndexMap;
use crate::op::{Keep, NodeId, Op, ReduceInit, ScalarOp};
use crate::shape;
use crate::sized::COHORT_SPIN_POLLS;

/// The result of running a tensor program: every requested output's data
/// and shape, plus the peak number of live intermediate buffers the run
/// held at once, where that backend tracks it.
///
/// One type shared by every backend this crate and `omega` ship — a CPU run
/// and a Metal run report the identical shape, so a parity test compares
/// them directly with no adapter on either side. `peak_live_buffers` is
/// `Some` only where a backend actually counts it: [`evaluate`] and
/// [`evaluate_parallel`] track it against their own `Vec<Option<Vec<f32>>>`
/// buffer table (see [`Evaluated::peak_live_buffers`]'s own doc); a device
/// backend whose buffer lifetime is managed by its own allocator (Metal's
/// retain/release) reports `None` rather than a number that would not mean
/// the same thing.
#[derive(Debug)]
pub struct Evaluated {
    root: NodeId,
    results: Vec<(NodeId, Vec<u64>, Vec<f32>)>,
    peak_live_buffers: Option<usize>,
}

impl Evaluated {
    /// Builds a result directly from a backend's own bookkeeping. Public so
    /// a sibling backend (`omega`'s Metal driver) can report through this
    /// same type instead of minting its own — see the type's own doc for
    /// why one shared shape matters here.
    #[must_use]
    pub fn from_parts(
        root: NodeId,
        results: Vec<(NodeId, Vec<u64>, Vec<f32>)>,
        peak_live_buffers: Option<usize>,
    ) -> Self {
        Self {
            root,
            results,
            peak_live_buffers,
        }
    }

    #[must_use]
    pub fn root(&self) -> &[f32] {
        self.get(self.root).map_or(&[], |(data, _)| data)
    }

    #[must_use]
    pub fn shape(&self) -> &[u64] {
        self.get(self.root).map_or(&[], |(_, shape)| shape)
    }

    /// The data and shape of a specific requested output, or `None` if
    /// `node` was not in the `outputs` passed to [`evaluate`].
    #[must_use]
    pub fn get(&self, node: NodeId) -> Option<(&[f32], &[u64])> {
        self.results
            .iter()
            .find(|(candidate, _, _)| *candidate == node)
            .map(|(_, shape, data)| (data.as_slice(), shape.as_slice()))
    }

    /// The most buffers ([`Op::Input`] inputs and computed intermediates)
    /// held live at any one point during the run, on a backend that counts
    /// it — `None` otherwise (see this type's own doc). The one number that
    /// proves streaming buffer lifetime is doing something: a long unary
    /// chain should hold a small constant, not one buffer per op.
    #[must_use]
    pub const fn peak_live_buffers(&self) -> Option<usize> {
        self.peak_live_buffers
    }

    /// Surrenders every result's storage into `scratch` instead of letting
    /// it drop, so a caller done reading this result can hand the same
    /// memory to [`evaluate_with_scratch`]'s next call — the counterpart to
    /// that function's pool, letting a caller that already called it once
    /// avoid that function's per-call output allocation on every call after
    /// the first. A caller that never calls this just lets `Evaluated` drop
    /// normally, exactly as today.
    pub fn into_scratch(self, scratch: &mut Vec<Vec<f32>>) {
        scratch.extend(self.results.into_iter().map(|(_, _, data)| data));
    }
}

/// Everything [`evaluate`] and [`evaluate_parallel`] must agree on before
/// either one is free to choose how a single nest actually runs.
///
/// Each buffer-table slot is a [`Cow`]: `Borrowed` for an [`Op::Input`]
/// slice straight out of the caller's `blocks` (never written, never
/// retired — see [`prepare`]), `Owned` for a computed intermediate this
/// evaluator holds. `Cow<[f32]>`'s `Owned` associate is `Vec<f32>` (`[T]:
/// ToOwned<Owned = Vec<T>>`), so this is exactly the borrowed-or-owned shape
/// this table needs, with `Clone`/`Deref`/`into_owned` already provided —
/// no hand-rolled type earns a place next to it.
struct Prepared<'block> {
    root: NodeId,
    shapes: shape::Shapes,
    effective_outputs: Vec<NodeId>,
    buffers: Vec<Option<Cow<'block, [f32]>>>,
    resolved: Vec<BoundOp>,
    retires: Vec<Vec<NodeId>>,
}

/// The preamble [`evaluate`] and [`evaluate_parallel`] share: shape
/// inference, output resolution, block binding, stride resolution, and
/// per-node buffer retirement. Neither evaluator's own body decides any of
/// this — the two diverge only in how one already-resolved [`BoundOp`]
/// node gets executed.
///
/// Stride resolution (and the fusion it decides) runs over the whole
/// program at once — see `bind::bind`'s docs. Buffer retirement below
/// is a separate, finer-grained liveness question over the *emitted* node
/// sequence: a held zip's own consumption is deferred to whenever its
/// consumer materializes it, which can be well after the expression
/// position `live::annotate` reasons about, so retirement here is computed
/// fresh over `resolved`, not reused from the fusion pass's liveness.
fn prepare<'block>(
    program: &[Op],
    symbols: &[u64],
    blocks: &[&'block [f32]],
    outputs: &[NodeId],
) -> Result<Prepared<'block>, TensorError> {
    let shapes = shape::infer(program, symbols)?;
    // `evaluate`/`evaluate_parallel`'s own `blocks: &[&[f32]]` is f32-only,
    // so neither has a quantized weight to offer this gate yet — see
    // `reject_non_float32`'s doc for what the empty set here is standing in
    // for and what `matmul_q4k_f32` covers instead.
    reject_non_float32(program, &BTreeSet::new())?;

    let root = program
        .len()
        .checked_sub(1)
        .map(|last| NodeId(last as u32))
        .ok_or(TensorError::Empty)?;
    for output in outputs {
        if output.0 as usize >= program.len() {
            return Err(TensorError::UnknownOutput(*output));
        }
    }
    let effective_outputs: Vec<NodeId> = if outputs.is_empty() {
        vec![root]
    } else {
        outputs.to_vec()
    };
    // a node the structural pass above exempted as an unreferenced dead leaf
    // can still be exactly what THIS call asked to get back — see
    // `reject_non_float32_outputs`'s own doc for why that stays a separate,
    // always-run, per-call check rather than folding `outputs` into the
    // cached pass above.
    reject_non_float32_outputs(program, &BTreeSet::new(), &effective_outputs)?;

    let block_nodes = block_node_ids(program);
    if blocks.len() != block_nodes.len() {
        return Err(TensorError::InputCountMismatch {
            expected: block_nodes.len(),
            found: blocks.len(),
        });
    }

    let mut buffers: Vec<Option<Cow<'block, [f32]>>> = vec![None; program.len()];
    for (node, data) in block_nodes.iter().zip(blocks.iter()) {
        let expected = element_count(shapes.of(*node));
        if data.len() != expected {
            return Err(TensorError::InputSizeMismatch {
                node: *node,
                expected,
                found: data.len(),
            });
        }
        buffers[node.0 as usize] = Some(Cow::Borrowed(data));
    }

    let resolved = bind::bind(program, &shapes, &effective_outputs)?;
    let retires = node_retirement(&resolved, &effective_outputs);

    Ok(Prepared {
        root,
        shapes,
        effective_outputs,
        buffers,
        resolved,
        retires,
    })
}

/// Both evaluators reach the same [`Evaluated`] the same way once their
/// execution loop is done: read each requested output's shape and data
/// back out of the (by-then-retired-down) buffer table.
///
/// Takes `buffers` by value — both callers own the table and drop it right
/// after this returns — so the common case (each output node named once)
/// moves its data out instead of cloning it. A node named twice in
/// `effective_outputs` cannot be moved out twice: `repeats_later` detects
/// that and puts a clone back for the later occurrence to take instead, so
/// duplicate outputs still resolve correctly, just without the free move.
fn finish(
    shapes: &shape::Shapes,
    effective_outputs: &[NodeId],
    mut buffers: Vec<Option<Cow<'_, [f32]>>>,
    root: NodeId,
    peak_live_buffers: usize,
) -> Evaluated {
    let results = effective_outputs
        .iter()
        .enumerate()
        .map(|(position, node)| {
            let shape = shapes.of(*node).to_vec();
            let repeats_later = effective_outputs[position + 1..].contains(node);
            let data = match buffers[node.0 as usize].take() {
                Some(buffer) => {
                    if repeats_later {
                        buffers[node.0 as usize] = Some(buffer.clone());
                    }
                    buffer.into_owned()
                }
                None => Vec::new(),
            };
            (*node, shape, data)
        })
        .collect();

    Evaluated::from_parts(root, results, Some(peak_live_buffers))
}

/// Run a tensor program to f32 data.
///
/// `blocks` binds [`Op::Input`] inputs positionally, in the order they
/// appear in `program` — the local, single-partition case; a distributed
/// evaluator would instead resolve blocks by
/// [`name`](Op::name). `outputs` selects which nodes to return data for;
/// an empty slice means the root (the program's last expression) only.
///
/// Every call starts and ends with an empty reuse pool — see
/// [`evaluate_with_scratch`] for the same contract with a caller-carried
/// pool that survives across calls.
pub fn evaluate(
    program: &[Op],
    symbols: &[u64],
    blocks: &[&[f32]],
    outputs: &[NodeId],
) -> Result<Evaluated, TensorError> {
    let mut free_buffers: Vec<Vec<f32>> = Vec::new();
    evaluate_pooled(program, symbols, blocks, outputs, &mut free_buffers)
}

/// Same contract as [`evaluate`], but binds [`Op::Input`] inputs by
/// [`Op::name`] instead of position — the counterpart [`evaluate`]'s own doc
/// promises for the distributed case: a partition crossing a wire
/// (`partition::partition_at`) renumbers a program, so positional order is
/// gone by the time a consumer half receives its inputs, and only `name`
/// survives the cut.
///
/// This does not change `evaluate`'s signature or behaviour: every `named`
/// entry is wrapped as [`QuantizedBlock::Float32`] and handed to
/// [`evaluate_quantized_named`], the one name-resolution loop this and
/// [`evaluate_quantized_named`] both share — see that function's doc for
/// why `evaluate_quantized`'s gate is a no-op for an all-`Float32` caller
/// like this one, so routing through it changes nothing observable here.
pub fn evaluate_named(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, &[f32])],
    outputs: &[NodeId],
) -> Result<Evaluated, TensorError> {
    let wrapped: Vec<(&str, QuantizedBlock)> =
        named.iter().map(|(name, data)| (*name, QuantizedBlock::Float32(data))).collect();
    evaluate_quantized_named(program, symbols, &wrapped, outputs)
}

/// Same contract as [`evaluate`], plus one capability a caller cannot get
/// from that function: `scratch` seeds this run's buffer-reuse pool instead
/// of starting it empty, and receives back whatever the pool held once the
/// run finished — most usefully, whatever a prior call's
/// [`Evaluated::into_scratch`] deposited into it. A caller that runs the
/// same program (or same-shaped programs) repeatedly and feeds each
/// result's storage back through `into_scratch` skips `evaluate`'s per-call
/// output allocation on every call after the first, without this crate ever
/// exposing an allocator or a persistent handle — the caller still decides
/// `scratch`'s lifetime, this function only ever borrows it. `evaluate`
/// itself is exactly this function with `scratch` starting, and ending,
/// empty.
pub fn evaluate_with_scratch(
    program: &[Op],
    symbols: &[u64],
    blocks: &[&[f32]],
    outputs: &[NodeId],
    scratch: &mut Vec<Vec<f32>>,
) -> Result<Evaluated, TensorError> {
    evaluate_pooled(program, symbols, blocks, outputs, scratch)
}

/// A [`bind::bind`]-shaped execution plan whose per-node output storage is
/// allocated exactly once and reused, unchanged in size, across every call
/// to [`evaluate_named_with_arena`] against the SAME `program` — the
/// static-arena counterpart to [`evaluate_named`], for a caller (a training
/// loop, per `docs/discipline.md` ROW 164) that runs an identically-shaped
/// program hundreds of times in a row and today pays `evaluate_named`'s own
/// `shape::infer` + `bind::bind` + a fresh `vec![0.0; n]` per node on every
/// single call even though every one of those calls resolves to the
/// identical shapes.
///
/// What a caller can do with this that [`evaluate_named`] alone cannot:
/// amortize bind + per-node allocation across every step of a loop instead
/// of repeating both on every call — the ONLY thing this type exists to
/// buy, per this crate's own binary-question gate (`AGENTS.md`,
/// guiding-principles §1: "what can a caller do that they could not
/// before").
pub struct StaticArena {
    resolved: Vec<BoundOp>,
    shapes: shape::Shapes,
    effective_outputs: Vec<NodeId>,
    root: NodeId,
    input_names: Vec<(NodeId, String)>,
    buffers: Vec<Option<Vec<f32>>>,
    /// Nodes in `resolved` with zero consumers among every other resolved
    /// node's own operands and no membership in `effective_outputs` —
    /// `docs/discipline.md` ROW 166's own dead node (`bind`'s
    /// `differentiate_elementwise` builds both a `Multiply`'s operand
    /// contributions before `route_contribution`'s `is_unwanted_input` gate
    /// ever runs, so the unwanted one is bound but never read). Computed
    /// once here, against `BoundOp::operands()` — which already carries
    /// every source a fused/composed body absorbed, since fusion moves a
    /// source node into the fusing op's own physical operand list rather
    /// than leaving a separate reference behind — so a node consumed only
    /// inside a composed body is correctly counted live. `run_resolved_nodes_in_arena`
    /// skips these; every other field above (`resolved`, `shapes`,
    /// `effective_outputs`, `bind::bind`'s own fusion decisions) is
    /// untouched, which is what keeps every fusion/eligibility decision
    /// exactly as ROW 166's graph-level attempt found it, before either the
    /// node it deleted or the sibling that regressed once it was gone
    /// existed as candidates to remove.
    dead: BTreeSet<NodeId>,
    /// `resolved` nodes whose [`BoundOpKind`] is `Constant` or `Iota` —
    /// `docs/discipline.md` ROW 174's own found lever: a `Constant`'s value
    /// is a literal baked into the `BoundOp` at `bind::bind` time
    /// (`run_constant`'s whole body is `output.fill(value)`, no operand
    /// read) and an `Iota`'s output is derived purely from its own position
    /// in `BoundOp::extents` (`run_iota`, also no operand read) — both are
    /// call-invariant by construction, since neither `BoundOpKind::operands()`
    /// entry exists for either variant (`bind.rs`'s own `operands()` match
    /// returns `&[]` for `Iota | Constant`). Run once inside
    /// `build_static_arena`, then skipped forever by
    /// `run_resolved_nodes_in_arena`, the same "computed once, cheap to
    /// consult" shape `dead` above already uses. Never overlaps `dead`: a
    /// node here is either a requested output (kept, still run once) or
    /// consumed by a live sibling (kept, still run once) — `dead_resolved_nodes`
    /// only drops nodes with zero consumers and no output membership, which
    /// this field does not gate on.
    static_nodes: BTreeSet<NodeId>,
}

/// Every node `resolved` physically reads, straight off [`BoundOp::operands()`]
/// plus each gathered operand's own [`bind::Lookup::indices`] (a second,
/// separate `NodeId` reference `operands()`'s own `(NodeId, Layout,
/// Option<Lookup>)` tuple does not fold into its leading field — missing it
/// would silently mark a live index-table node dead). `operands()` already
/// carries every source a fused/composed body absorbed (`bind.rs`'s own doc:
/// fusion moves a source node into the fusing op's physical operand list
/// rather than leaving a separate graph edge behind), so this is a scan over
/// `BoundOp` structure, never the pre-bind graph.
fn consumed_by_resolved_nodes(resolved: &[BoundOp]) -> BTreeSet<NodeId> {
    let mut consumed = BTreeSet::new();
    for computed in resolved {
        for (operand, _layout, lookup) in computed.operands() {
            consumed.insert(*operand);
            if let Some(lookup) = lookup {
                consumed.insert(lookup.indices);
            }
        }
    }
    consumed
}

/// The dead-set `StaticArena::dead` documents: every `resolved` node neither
/// consumed by another resolved node nor named in `effective_outputs`. See
/// `docs/discipline.md` ROW 166/167.
fn dead_resolved_nodes(resolved: &[BoundOp], effective_outputs: &[NodeId]) -> BTreeSet<NodeId> {
    let consumed = consumed_by_resolved_nodes(resolved);
    resolved
        .iter()
        .map(|computed| computed.node)
        .filter(|node| !consumed.contains(node) && !effective_outputs.contains(node))
        .collect()
}

/// `StaticArena::static_nodes` documents: every LIVE `resolved` node whose
/// [`BoundOpKind`] is `Constant` or `Iota` — excludes anything already in
/// `dead` (no reason to run a dead constant even once) since callers pass
/// `dead` alongside this set to build the union `run_resolved_nodes_in_arena`
/// skips. See `docs/discipline.md` ROW 174.
fn static_resolved_nodes(resolved: &[BoundOp], dead: &BTreeSet<NodeId>) -> BTreeSet<NodeId> {
    resolved
        .iter()
        .filter(|computed| matches!(computed.kind, BoundOpKind::Constant { .. } | BoundOpKind::Iota))
        .map(|computed| computed.node)
        .filter(|node| !dead.contains(node))
        .collect()
}

/// Builds a [`StaticArena`] for `program`: runs shape inference and
/// [`bind::bind`] once, then pre-sizes every node's output buffer (both
/// [`Op::Input`] slots and every computed [`BoundOp`]) at its final,
/// call-invariant length. Every subsequent [`evaluate_named_with_arena`]
/// call against this arena reuses these SAME allocations — no per-step
/// `shape::infer`, no per-step `bind::bind`, no per-step `Vec` allocation,
/// and no [`evaluate_with_scratch`]-style runtime best-fit search over a
/// shared pool (`docs/discipline.md` ROW 159 measured that search losing)
/// — each node's buffer lives at a fixed index for the arena's whole
/// lifetime.
///
/// # Errors
/// The same shape/dtype/output errors [`evaluate_named`] itself raises,
/// since this runs the identical `prepare`-shaped validation once up front
/// instead of on every call.
pub fn build_static_arena(program: &[Op], symbols: &[u64], outputs: &[NodeId]) -> Result<StaticArena, TensorError> {
    let shapes = shape::infer(program, symbols)?;
    reject_non_float32(program, &BTreeSet::new())?;

    let root = program
        .len()
        .checked_sub(1)
        .map(|last| NodeId(last as u32))
        .ok_or(TensorError::Empty)?;
    for output in outputs {
        if output.0 as usize >= program.len() {
            return Err(TensorError::UnknownOutput(*output));
        }
    }
    let effective_outputs: Vec<NodeId> = if outputs.is_empty() {
        vec![root]
    } else {
        outputs.to_vec()
    };
    reject_non_float32_outputs(program, &BTreeSet::new(), &effective_outputs)?;

    let block_nodes = block_node_ids(program);
    let mut input_names = Vec::with_capacity(block_nodes.len());
    let mut buffers: Vec<Option<Vec<f32>>> = vec![None; program.len()];
    for node in &block_nodes {
        let name = program[node.0 as usize].name().ok_or(TensorError::UnnamedInput(*node))?;
        input_names.push((*node, String::from(name)));
        buffers[node.0 as usize] = Some(vec![0.0f32; element_count(shapes.of(*node))]);
    }

    let resolved = bind::bind(program, &shapes, &effective_outputs)?;
    for computed in &resolved {
        buffers[computed.node.0 as usize] = Some(vec![0.0f32; node_output_len(computed)]);
    }
    let dead = dead_resolved_nodes(&resolved, &effective_outputs);
    let static_nodes = static_resolved_nodes(&resolved, &dead);

    for computed in &resolved {
        if !static_nodes.contains(&computed.node) {
            continue;
        }
        let node_index = computed.node.0 as usize;
        let mut output = buffers[node_index].take().ok_or(TensorError::NotLowerable {
            node: computed.node,
            reason: "static arena has no pre-sized slot for this resolved node -- build_static_arena did not size it",
        })?;
        run_node_into(computed, &buffers, None, None, &mut output)?;
        buffers[node_index] = Some(output);
    }

    Ok(StaticArena {
        resolved,
        shapes,
        effective_outputs,
        root,
        input_names,
        buffers,
        dead,
        static_nodes,
    })
}

/// Runs `arena`'s already-[`bind::bind`]-resolved program once against this
/// step's `named` host buffers, writing every input and every computed
/// node's output into the SAME per-node storage [`build_static_arena`]
/// sized once. The only allocation on this call's own path is the small,
/// output-count-sized clone [`Evaluated`] needs to hand results back to a
/// caller that runs another step against the same arena immediately after
/// — the arena's own buffers must survive this call, so results are copied
/// out, never moved out (unlike `evaluate_pooled`'s one-shot table).
///
/// # Errors
/// [`TensorError::UnboundInputName`] if `named` has no entry for one of
/// the program's [`Op::Input`] names; [`TensorError::InputSizeMismatch`] if
/// a bound buffer's length no longer matches the size [`build_static_arena`]
/// fixed for it (a genuinely different-shaped call, not a training step).
pub fn evaluate_named_with_arena(arena: &mut StaticArena, named: &[(&str, &[f32])]) -> Result<Evaluated, TensorError> {
    bind_named_inputs_into_arena(arena, named, true)?;
    run_resolved_nodes_in_arena(arena)?;

    let results = arena
        .effective_outputs
        .iter()
        .map(|node| {
            let shape = arena.shapes.of(*node).to_vec();
            let data = arena.buffers[node.0 as usize].as_deref().unwrap_or(&[]).to_vec();
            (*node, shape, data)
        })
        .collect();

    Ok(Evaluated::from_parts(arena.root, results, None))
}

/// [`evaluate_named_with_arena`]'s own input-binding loop, factored out so
/// [`evaluate_named_with_arena_in_place`] can reuse it with `require_all =
/// false`: a name absent from `named` is treated as "already correct in
/// the arena" (the in-place rebind lever put it there) rather than an
/// error, instead of duplicating the loop body.
fn bind_named_inputs_into_arena(arena: &mut StaticArena, named: &[(&str, &[f32])], require_all: bool) -> Result<(), TensorError> {
    for (node, name) in &arena.input_names {
        let found = named.iter().find(|(candidate, _)| candidate == name).map(|(_, data)| *data);
        let data = match found {
            Some(data) => data,
            None if require_all => return Err(TensorError::UnboundInputName(name.clone())),
            None => continue,
        };
        let slot = arena.buffers[node.0 as usize].as_mut().ok_or(TensorError::NotLowerable {
            node: *node,
            reason: "static arena has no pre-sized slot for this input node -- build_static_arena did not size it",
        })?;
        if slot.len() != data.len() {
            return Err(TensorError::InputSizeMismatch {
                node: *node,
                expected: slot.len(),
                found: data.len(),
            });
        }
        slot.copy_from_slice(data);
    }
    Ok(())
}

/// [`evaluate_named_with_arena`]'s own resolved-node execution loop,
/// factored out so [`evaluate_named_with_arena_in_place`] shares the
/// identical execution path rather than a second copy of it.
fn run_resolved_nodes_in_arena(arena: &mut StaticArena) -> Result<(), TensorError> {
    for computed in &arena.resolved {
        if arena.dead.contains(&computed.node) || arena.static_nodes.contains(&computed.node) {
            continue;
        }
        let node_index = computed.node.0 as usize;
        let mut output = arena.buffers[node_index].take().ok_or(TensorError::NotLowerable {
            node: computed.node,
            reason: "static arena has no pre-sized slot for this resolved node -- build_static_arena did not size it",
        })?;
        run_node_into(computed, &arena.buffers, None, None, &mut output)?;
        arena.buffers[node_index] = Some(output);
    }
    Ok(())
}

/// `docs/discipline.md` ROW 180's dynamic-elision probe: the SAME skip
/// check [`run_resolved_nodes_in_arena`] already runs (`dead`/`static_nodes`,
/// both fixed forever at [`build_static_arena`] time), unioned with a THIRD
/// set the caller derives fresh every call from a per-step mask -- a block
/// live one step and skipped the next, which `dead`/`static_nodes` cannot
/// express since both are computed once and never revisited. `named` is
/// expected to carry ONLY this step's live blocks (`bind_named_inputs_into_arena`'s
/// `require_all = false`, the same relaxation
/// [`evaluate_named_with_arena_in_place`] already uses) -- a masked-off
/// block's caller-side buffer is never even copied into the arena, not just
/// never computed, so the traffic this probe measures is the SAME traffic a
/// caller genuinely elides (no read of the skipped block's data at all).
/// Execution-level only: `resolved`/`shapes`/`effective_outputs` (every
/// `bind::bind` fusion decision) are untouched, exactly as ROW 167 found for
/// the fixed dead-set -- graph-level removal (ROW 166) is not this
/// mechanism and is not attempted here. A node this step's `skip` names that
/// is NOT also live in `arena.buffers` keeps its stale prior-step value,
/// same as `evaluate_named_with_arena_in_place`'s own rebind aliasing
/// already relies on for untouched nodes.
#[cfg(feature = "dynamic-elision-probe")]
pub fn evaluate_named_with_arena_masked(
    arena: &mut StaticArena,
    named: &[(&str, &[f32])],
    skip: &BTreeSet<NodeId>,
) -> Result<Evaluated, TensorError> {
    bind_named_inputs_into_arena(arena, named, false)?;
    for computed in &arena.resolved {
        if arena.dead.contains(&computed.node) || arena.static_nodes.contains(&computed.node) || skip.contains(&computed.node) {
            continue;
        }
        let node_index = computed.node.0 as usize;
        let mut output = arena.buffers[node_index].take().ok_or(TensorError::NotLowerable {
            node: computed.node,
            reason: "static arena has no pre-sized slot for this resolved node -- build_static_arena did not size it",
        })?;
        run_node_into(computed, &arena.buffers, None, None, &mut output)?;
        arena.buffers[node_index] = Some(output);
    }

    let results = arena
        .effective_outputs
        .iter()
        .map(|node| {
            let shape = arena.shapes.of(*node).to_vec();
            let data = arena.buffers[node.0 as usize].as_deref().unwrap_or(&[]).to_vec();
            (*node, shape, data)
        })
        .collect();

    Ok(Evaluated::from_parts(arena.root, results, None))
}

/// Reads `node`'s current buffer straight out of `arena` -- a borrow, not a
/// clone. Valid for any node [`build_static_arena`] pre-sized: an
/// [`Op::Input`] slot or a resolved node's output, in whichever state the
/// arena is in right now (freshly computed, or holding a value
/// [`evaluate_named_with_arena_in_place`]'s rebind aliasing swapped in).
#[must_use]
pub fn arena_output(arena: &StaticArena, node: NodeId) -> Option<&[f32]> {
    arena.buffers.get(node.0 as usize).and_then(Option::as_deref)
}

/// The in-place counterpart to [`evaluate_named_with_arena`]: a caller
/// whose `rebind` targets stay resident IN the arena across steps (a
/// training loop's own parameters and optimizer state, per
/// `docs/discipline.md` ROW 164's own named residual -- "the rebind
/// targets... get cloned out because arena buffers must survive the next
/// call") never pays that clone at all. `named` here carries ONLY this
/// step's genuinely-new bindings (a batch, a step counter) -- every
/// `rebind` name is expected to already be resident from the PRIOR call's
/// own aliasing swap (or, on the very first call, from a `named` entry the
/// caller supplied once up front).
///
/// After running `program`, every `(computed, input_name)` pair in
/// `rebind` is spliced directly into place with `Vec::swap`: the
/// computed node's freshly written buffer BECOMES the `input_name` node's
/// buffer, and the stale old input buffer moves to the computed node's own
/// slot, where the next call's resolved-node pass fully overwrites it
/// again (matching `evaluate_pooled`'s own "every write position gets
/// overwritten before any read" contract) -- zero allocation, zero `f32`
/// copied, only two `Vec<f32>` headers exchanged.
///
/// What a caller can do with this that [`evaluate_named_with_arena`] alone
/// cannot: run N steps of a `rebind`-shaped loop with the state-carrying
/// buffers touched exactly zero times between [`build_static_arena`] and
/// the caller's own final read-out, instead of a clone-out-then-copy-in
/// pair on every single step.
///
/// # Errors
/// The same errors [`evaluate_named_with_arena`] raises for the `named`
/// bindings it IS given, plus [`TensorError::UnboundInputName`] if a
/// `rebind` pair names an [`Op::Input`] `build_static_arena` never bound,
/// and [`TensorError::InputSizeMismatch`] if a `rebind` pair's computed
/// and input buffers are not the same length (a genuinely different-shaped
/// rebind, not the same-program repeated-step case this exists for).
pub fn evaluate_named_with_arena_in_place(
    arena: &mut StaticArena,
    named: &[(&str, &[f32])],
    loss: NodeId,
    rebind: &[(NodeId, &str)],
) -> Result<f32, TensorError> {
    bind_named_inputs_into_arena(arena, named, false)?;
    run_resolved_nodes_in_arena(arena)?;

    let loss_value = arena_output(arena, loss).and_then(|data| data.first().copied()).unwrap_or(0.0);

    for (computed, name) in rebind {
        let input_node = arena
            .input_names
            .iter()
            .find(|(_, candidate)| candidate == name)
            .map(|(node, _)| *node)
            .ok_or_else(|| TensorError::UnboundInputName(String::from(*name)))?;
        let computed_index = computed.0 as usize;
        let input_index = input_node.0 as usize;
        let computed_len = arena.buffers[computed_index].as_ref().map_or(0, Vec::len);
        let input_len = arena.buffers[input_index].as_ref().map_or(0, Vec::len);
        if computed_len != input_len {
            return Err(TensorError::InputSizeMismatch {
                node: input_node,
                expected: input_len,
                found: computed_len,
            });
        }
        arena.buffers.swap(computed_index, input_index);
    }

    Ok(loss_value)
}

/// Reads `name`'s current resident buffer straight out of `arena` -- a
/// borrow, not a clone. `name` is any [`Op::Input`] [`build_static_arena`]
/// bound; the value returned reflects whatever the arena currently holds
/// for it, whether bound by the last [`evaluate_named_with_arena`]/
/// [`evaluate_named_with_arena_in_place`] call's own `named` or spliced in
/// by [`evaluate_named_with_arena_in_place`]'s rebind aliasing -- the
/// read-out a caller uses once, at the end of a run, to pull a `rebind`
/// loop's final state back into its own owned buffers.
#[must_use]
pub fn arena_named_input<'arena>(arena: &'arena StaticArena, name: &str) -> Option<&'arena [f32]> {
    let node = arena.input_names.iter().find(|(_, candidate)| candidate == name).map(|(node, _)| *node)?;
    arena_output(arena, node)
}

/// One [`evaluate_quantized`]-bound block: either a plain `f32`
/// [`Op::Input`] buffer, exactly what [`evaluate`]'s own `blocks: &[&[f32]]`
/// carries, or the raw packed bytes of a `Q4_K`-quantized weight matrix.
/// [`evaluate`]'s `blocks` parameter has no way to carry the second case — a
/// quantized weight has no legitimate `&[f32]` view to hand through it
/// without dequantizing first, which would defeat the entire point (see
/// [`matmul_q4k_f32`]'s doc on what dequantizing first costs). Both variants
/// bind positionally, in the same [`Op::Input`] program order [`evaluate`]'s
/// `blocks` already uses — one binding convention, not two.
#[derive(Debug, Clone, Copy)]
pub enum QuantizedBlock<'a> {
    Float32(&'a [f32]),
    Q4K(&'a [u8]),
    /// Raw packed `Q5_K` bytes -- same super-block shape as [`Self::Q4K`]
    /// (256 elements, 8 sub-blocks of 32) plus a `qh` high-bit plane; see
    /// [`proxima_gguf::quant::q5_k`] for the on-disk layout this borrows
    /// unchanged.
    Q5K(&'a [u8]),
    /// Raw packed `Q6_K` bytes -- 256 elements, 16 sub-blocks of 16, one
    /// signed 8-bit scale per sub-block and no `dmin` term; see
    /// [`proxima_gguf::quant::q6_k`] for the on-disk layout this borrows
    /// unchanged.
    Q6K(&'a [u8]),
    /// Raw packed `Q8_0` bytes -- 32-element blocks, one `f16` scale per
    /// block, no sub-block structure at all; see
    /// [`proxima_gguf::quant::q8_0`] for the on-disk layout this borrows
    /// unchanged. The one variant this enum carries that the growable
    /// per-layer key/value context cache (`proxima-model-interop`'s
    /// `LayerCache`) actually binds -- its rows are `HEAD_DIM / 2`
    /// elements wide, small enough that `Q4_K`/`Q5_K`/`Q6_K`'s 256-element
    /// super-blocks would straddle more than one cached position, while a
    /// 32-element `Q8_0` block divides a typical head dimension evenly.
    Q8_0(&'a [u8]),
    /// Raw packed `Q4_0` bytes -- 32-element blocks, one `f16` scale per
    /// block, no sub-block structure and no shared super-block with the
    /// K-quant family; see [`proxima_gguf::quant::q4_0`] for the on-disk
    /// layout this borrows unchanged. Legacy llama.cpp's simplest and most
    /// widely distributed 4-bit format -- unlike [`Self::Q4K`], no
    /// sub-block scale/min hierarchy, just `value = scale * (nibble - 8)`.
    Q4_0(&'a [u8]),
    /// Raw packed IEEE-754 binary16 bytes, little-endian, two per element,
    /// no block or scale structure at all -- unlike every other
    /// non-`Float32` variant above, a half-precision weight is not
    /// quantized, only narrower: each element converts to `f32` entirely on
    /// its own, with no neighbours' scale to consult. See [`matmul_f16_f32`]
    /// for the composed convert-then-fold kernel this variant reaches, and
    /// [`proxima_gguf::quant::f16`] for the on-disk layout this borrows
    /// unchanged.
    Float16(&'a [u8]),
    /// Raw packed `bfloat16` bytes, little-endian, two per element -- same
    /// per-element (non-block) shape as [`Self::Float16`], but a different
    /// bit layout (8-bit exponent, 7-bit mantissa) needing its own
    /// conversion. See [`matmul_bf16_f32`] and [`proxima_gguf::quant::bf16`].
    BFloat16(&'a [u8]),
}

/// [`evaluate`]'s counterpart for a program with one `Q4_K`-quantized weight
/// operand — the entry point that actually reaches [`matmul_q4k_f32`], which
/// [`evaluate`]/[`evaluate_parallel`] cannot: their `blocks: &[&[f32]]`
/// parameter is f32-only by construction, so neither has anywhere to put a
/// packed byte buffer. This function is that seam: [`QuantizedBlock::Q4K`]
/// entries are held back from the f32 buffer table and instead collected
/// into a `NodeId -> &[u8]` side table that `run_reduce` consults (via
/// `quantized_operand`) for the one `Reduce` node `is_quantized_matmul_operand`
/// already proves is shaped for it — every other node in `program` still
/// runs the exact same f32 path [`evaluate`] does, unchanged.
///
/// `evaluate_typed`'s `TypedBuffer` seam was considered and rejected for
/// this: even `typed_program_plan`'s `Widened` shape only crosses dtypes
/// once, at a `Reduce` node's own accumulator boundary, but a quantized
/// matmul is mixed *within* one fused reduce body — `UInt8`-packed weight
/// times `Float32` activation into a `Float32` output — which is exactly
/// the shape `reject_non_float32`'s quantized-weight exemption carves out,
/// not a program `evaluate_typed` would ever accept. See that function's own
/// doc (`typed_program_plan`'s `TypedPlan`) for the two shapes it does
/// accept.
///
/// Every call starts and ends with an empty node-output pool and an
/// unvalidated structure cache — see [`evaluate_quantized_with_scratch`]
/// for the same contract with caller-carried state that survives across
/// calls to the same `program`.
pub fn evaluate_quantized(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock],
    outputs: &[NodeId],
) -> Result<Evaluated, TensorError> {
    let mut free_buffers: Vec<Vec<f32>> = Vec::new();
    let mut validated_weight_nodes: Option<BTreeSet<NodeId>> = None;
    evaluate_quantized_with_scratch(program, symbols, blocks, outputs, &mut free_buffers, &mut validated_weight_nodes)
}

/// Same contract as [`evaluate_quantized`], plus two capabilities a caller
/// cannot get from that function, both aimed at the same shape of caller: a
/// decode loop that evaluates the *same* `program` once per generated
/// token, where only `symbols` (`cached_len` growing by one) actually
/// changes between calls and every weight stays put.
///
/// `free_buffers` is [`evaluate_with_scratch`]'s own reuse pool, applied to
/// this evaluator's loop the same way this crate's internal `evaluate_pooled`
/// already applies it to [`evaluate`]/[`evaluate_parallel`] — a private
/// `take_or_allocate` helper hands a node its output storage from the pool
/// instead of a fresh `vec![0.0; n]`, and a private `retire_into` helper
/// returns a retired node's owned storage to the pool instead of dropping
/// it. `evaluate_quantized` did neither: every one of a program's nodes
/// paid a fresh heap allocation on every call, measured at 3.2-3.7 ms/step
/// of a ~68 ms cached-decode step on the real checkpoint this crate's own
/// `DIAG … loop_overhead_ms` reports.
///
/// `validated_weight_nodes` caches this module's private
/// `reject_non_float32` gate's outcome across calls. That gate's cost is
/// `O(quantized weight count * program.len())` — for every node this
/// call's `blocks` tags as a packed weight, a private
/// `is_quantized_matmul_operand` helper rescans the whole program to prove
/// it is used in a matmul shape — and neither `program` nor which nodes are
/// weight-typed changes between decode steps, so the same outcome is valid
/// on every call after the first. Measured at 1.9-2.0 ms/step on the real
/// checkpoint's 1196-node cached-forward program, roughly half of
/// `DIAG … setup_ms`. A call whose `blocks` classifies a different set of
/// nodes as weight-typed than the cached run (a genuinely different
/// program shape, not a decode step) invalidates the cache and pays the
/// full gate again — the comparison against the cached
/// `BTreeSet<NodeId>` is what decides that,
/// not a size or pointer check that a coincidental match could fool.
///
/// `evaluate_quantized` is exactly this function with `free_buffers` and
/// `validated_weight_nodes` starting, and ending, empty — the same
/// relationship [`evaluate`] has to [`evaluate_with_scratch`].
pub fn evaluate_quantized_with_scratch(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock],
    outputs: &[NodeId],
    free_buffers: &mut Vec<Vec<f32>>,
    validated_weight_nodes: &mut Option<BTreeSet<NodeId>>,
) -> Result<Evaluated, TensorError> {
    // DIAGNOSTIC (proxima-debugger, remove before landing): brackets the
    // portion of evaluate_quantized that is neither the per-node-kind loop
    // below nor run_node_into itself -- shape::infer, bind::bind,
    // node_retirement, and the buffers table setup all run here, none of
    // it visible in the diag_kind_ticks table.
    #[cfg(feature = "instrument")]
    let diag_setup_started = instrument::read_ticks();
    let shapes = shape::infer(program, symbols)?;
    let block_nodes = block_node_ids(program);
    if blocks.len() != block_nodes.len() {
        return Err(TensorError::InputCountMismatch {
            expected: block_nodes.len(),
            found: blocks.len(),
        });
    }

    let mut quantized_weights: BTreeMap<NodeId, QuantizedBlock> = BTreeMap::new();
    let mut buffers: Vec<Option<Cow<[f32]>>> = vec![None; program.len()];
    for (node, block) in block_nodes.iter().zip(blocks.iter().copied()) {
        match block {
            QuantizedBlock::Float32(data) => {
                let expected = element_count(shapes.of(*node));
                if data.len() != expected {
                    return Err(TensorError::InputSizeMismatch {
                        node: *node,
                        expected,
                        found: data.len(),
                    });
                }
                buffers[node.0 as usize] = Some(Cow::Borrowed(data));
            }
            QuantizedBlock::Q4K(_)
            | QuantizedBlock::Q5K(_)
            | QuantizedBlock::Q6K(_)
            | QuantizedBlock::Q8_0(_)
            | QuantizedBlock::Q4_0(_)
            | QuantizedBlock::Float16(_)
            | QuantizedBlock::BFloat16(_) => {
                quantized_weights.insert(*node, block);
            }
        }
    }

    let root = program
        .len()
        .checked_sub(1)
        .map(|last| NodeId(last as u32))
        .ok_or(TensorError::Empty)?;
    for output in outputs {
        if output.0 as usize >= program.len() {
            return Err(TensorError::UnknownOutput(*output));
        }
    }
    let effective_outputs: Vec<NodeId> = if outputs.is_empty() {
        vec![root]
    } else {
        outputs.to_vec()
    };

    let quantized_weight_nodes: BTreeSet<NodeId> = quantized_weights.keys().copied().collect();
    if validated_weight_nodes.as_ref() != Some(&quantized_weight_nodes) {
        reject_non_float32(program, &quantized_weight_nodes)?;
        // cloned, not moved: the outputs-only check right below still needs
        // its own copy of this same set — see `reject_non_float32_outputs`'s
        // own doc for why that check cannot ride this cache.
        *validated_weight_nodes = Some(quantized_weight_nodes.clone());
    }
    reject_non_float32_outputs(program, &quantized_weight_nodes, &effective_outputs)?;

    let resolved = bind::bind(program, &shapes, &effective_outputs)?;
    let retires = node_retirement(&resolved, &effective_outputs);

    // DIAGNOSTIC (proxima-debugger, instrument-only): peak_live_buffers
    // counts occupied slots, not bytes -- a live buffer set of 12 entries
    // could be 12 MiB or 12 GiB. This mirrors that count but sums the actual
    // f32 byte length of every live entry, so the peak is reported in bytes.
    // Gated (unlike `peak_live_buffers` itself below, which `finish` needs
    // unconditionally): nothing in `Evaluated` carries a byte figure, only
    // the `DIAG` eprintln this landing also gated does, so an
    // instrument-off build pays neither the closure call nor its
    // `O(program.len())` rescan per node.
    #[cfg(feature = "instrument")]
    let diag_live_bytes = |buffers: &[Option<Cow<[f32]>>]| -> usize {
        buffers.iter().flatten().map(|cow| cow.len() * core::mem::size_of::<f32>()).sum()
    };

    // `peak_live_buffers` is real, always-returned `Evaluated` state (see
    // `finish` below and `Evaluated::peak_live_buffers`'s own tests), so it
    // stays unconditional -- but tracked via `live_now`, an O(1) running
    // count incremented/decremented alongside the loop's own `Some`/`None`
    // writes, rather than `live_count(&buffers)`'s O(program.len()) full
    // rescan called once per node (the actual quadratic cost this landing
    // removes: `O(program.len())` work times `program.len()` nodes).
    let mut peak_live_buffers = live_count(&buffers);
    let mut live_now = peak_live_buffers;
    #[cfg(feature = "instrument")]
    let mut diag_peak_live_bytes = diag_live_bytes(&buffers);
    #[cfg(feature = "instrument")]
    let mut diag_peak_position = 0usize;
    // DIAGNOSTIC (proxima-debugger, remove before landing): wall time by
    // node kind for this one serial evaluation loop -- `evaluate_quantized`
    // never routes through `evaluate_node_parallel`, so the only per-node
    // parallelism a forward pass gets is `run_reduce_quantized`'s internal
    // `matmul_rows_threaded` dispatch (the "reduce_matmul_quantized"
    // bucket below); every other kind runs fully serial on this one
    // thread. Keyed by the label `diag_node_kind_label` derives from the
    // same `quantized_operand` check `run_reduce_with_quantized_weights`
    // itself uses, so the bucket a node lands in matches the arm that
    // actually ran it.
    #[cfg(feature = "instrument")]
    let mut diag_kind_ticks: BTreeMap<&'static str, (u64, u64)> = BTreeMap::new();
    // DIAGNOSTIC (proxima-debugger, remove before landing): everything in
    // this loop body OTHER than run_node_into -- the output Vec's zero-fill
    // allocation (ahead of diag_node_started so it is never in
    // diag_kind_ticks) and the live-bytes bookkeeping after the call
    // (`diag_live_bytes` rescans the whole buffers table every node, but is
    // itself instrument-gated now -- see that closure's own doc). This
    // counter is instrument-only so it costs nothing outside this run.
    #[cfg(feature = "instrument")]
    let mut diag_loop_overhead_ticks: u64 = 0;
    #[cfg(feature = "instrument")]
    let diag_setup_ticks = instrument::elapsed_ticks(diag_setup_started);
    // Entered ONCE, before the node loop -- not per matmul call -- so the
    // wake this session amortizes is paid once per forward pass, the same
    // amortization `prime/src/os/cohort.rs`'s own module doc measures
    // against `ProximaBackgroundPool`'s per-call wake. `None` whenever
    // another forward already holds the process-wide cohort
    // (`ThreadCohort::enter` returning `Err`) or the cohort itself failed
    // to build; every one of the six signatures between here and
    // `matmul_rows_threaded` falls back to the `nest_pool` dispatch path in
    // that case, unchanged.
    let session = nest_cohort().and_then(|cohort| cohort.enter().ok());
    // `docs/discipline.md` ROW 140's own cache: scoped to this ONE
    // `evaluate_quantized_with_scratch` call (one decode/prefill step),
    // never carried across steps -- a later step's `activation_node` slot
    // holds a genuinely different value, and this vec is dropped with the
    // rest of this function's locals at the end of the call, so there is no
    // staleness window to reason about. Keyed by the activation's own
    // `NodeId` rather than threaded through `MatmulSession`/`CohortSession`
    // (`prime::os::cohort`): that type is a generic thread-cohort round
    // driver with no knowledge of `NodeId`/`Q8_K`, so extending it would
    // mean teaching a reusable concurrency primitive one tensor-specific
    // cache -- the same reuse-first question this crate already answers by
    // keeping `free_buffers`/`validated_weight_nodes` as this function's own
    // scratch parameters instead of bolting them onto a foreign type.
    //
    // `NodeId` is a position in `program`'s own flat `Vec<Op>` (`op.rs`'s
    // own doc), the exact same fact `buffers` above already exploits
    // (`vec![None; program.len()]`, indexed by `node.0 as usize`) -- so this
    // cache is sized and indexed identically, an O(1) direct slot lookup
    // instead of a `BTreeMap`'s per-hit key-comparison chain over up to 96
    // hits/step (ROW 140's own measured hit count).
    #[cfg(feature = "cohort-staged-graph")]
    let mut staged_quantize_cache: Vec<Option<Arc<[u8]>>> = vec![None; program.len()];
    let mut position = 0usize;
    while position < resolved.len() {
        #[cfg(feature = "cohort-staged-graph")]
        if let Some(session_ref) = session.as_ref() {
            let run_end = staged_batch_run_end(&resolved, position, &quantized_weights);
            if run_end - position >= STAGED_BATCH_MIN_LEN {
                #[cfg(feature = "instrument")]
                let diag_batch_started = instrument::read_ticks();
                run_staged_batch(
                    &resolved[position..run_end],
                    position,
                    &mut buffers,
                    &quantized_weights,
                    session_ref,
                    free_buffers,
                    &retires,
                    &mut live_now,
                    &mut staged_quantize_cache,
                )?;
                peak_live_buffers = peak_live_buffers.max(live_now);
                #[cfg(feature = "instrument")]
                {
                    let entry = diag_kind_ticks.entry("staged_batch").or_insert((0, 0));
                    entry.0 += (run_end - position) as u64;
                    entry.1 += instrument::elapsed_ticks(diag_batch_started);
                }
                position = run_end;
                continue;
            }
        }
        let computed = &resolved[position];
        #[cfg(feature = "instrument")]
        let diag_alloc_started = instrument::read_ticks();
        let mut output = take_or_allocate(free_buffers, node_output_len(computed));
        #[cfg(feature = "instrument")]
        {
            diag_loop_overhead_ticks += instrument::elapsed_ticks(diag_alloc_started);
        }
        #[cfg(feature = "instrument")]
        let diag_node_label = diag_node_kind_label(computed, &quantized_weights);
        #[cfg(feature = "instrument")]
        let diag_node_started = instrument::read_ticks();
        run_node_into(computed, &buffers, Some(&quantized_weights), session.as_ref(), &mut output)?;
        #[cfg(feature = "instrument")]
        {
            let entry = diag_kind_ticks.entry(diag_node_label).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += instrument::elapsed_ticks(diag_node_started);
        }
        #[cfg(feature = "instrument")]
        let diag_bookkeeping_started = instrument::read_ticks();
        buffers[computed.node.0 as usize] = Some(Cow::Owned(output));
        // `computed.node` is written exactly once (this position, in
        // program order), so this is always a `None` -> `Some` transition --
        // `live_now += 1` is the O(1) replacement for rescanning `buffers`.
        live_now += 1;
        peak_live_buffers = peak_live_buffers.max(live_now);
        #[cfg(feature = "instrument")]
        {
            let current_bytes = diag_live_bytes(&buffers);
            if current_bytes > diag_peak_live_bytes {
                diag_peak_live_bytes = current_bytes;
                diag_peak_position = position;
            }
        }
        for retired in &retires[position] {
            // NOT always a `Some` -> `None` transition, unlike
            // `evaluate_pooled`/`evaluate_parallel`'s identical loop: a
            // `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0` weight node is scheduled for
            // retirement here exactly like any other operand (`node_retirement`
            // builds its schedule from the generic program graph, with no
            // knowledge of the quantized/float32 split), but it never occupied
            // a `buffers` slot in the first place -- it lives in
            // `quantized_weights` instead (see the `QuantizedBlock` match
            // above). `retire_into` reports whether the slot was actually
            // live so `live_now` only counts a real retirement.
            if retire_into(&mut buffers, *retired, free_buffers) {
                live_now -= 1;
            }
        }
        #[cfg(feature = "instrument")]
        {
            diag_loop_overhead_ticks += instrument::elapsed_ticks(diag_bookkeeping_started);
        }
        position += 1;
    }
    #[cfg(feature = "instrument")]
    std::eprintln!(
        "DIAG evaluate_quantized: peak_live_buffers={peak_live_buffers} peak_live_bytes={diag_peak_live_bytes} ({:.4} GiB) at resolved_position={diag_peak_position}/{} node={:?}",
        diag_peak_live_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        resolved.len(),
        resolved.get(diag_peak_position).map(|computed| computed.node),
    );
    #[cfg(feature = "instrument")]
    {
        // captured before `diag_kind_ticks.into_iter()` below consumes the
        // map -- `nsper` task (2026-08-21): pairs with `MAC_OPS` (only
        // `run_reduce`'s dense f32 path increments it; `run_reduce_quantized`
        // uses its own `MATMUL_Q4K_MACS`/etc counters and `run_elementwise`
        // never touches it, so this map's "reduce_f32_dense" ticks and
        // `MAC_OPS`'s snapshot are the SAME node population).
        let reduce_dense_entry = diag_kind_ticks.get("reduce_f32_dense").copied();
        let total_ticks: u64 = diag_kind_ticks.values().map(|(_, ticks)| *ticks).sum();
        let mut ranked: Vec<(&str, u64, u64)> =
            diag_kind_ticks.into_iter().map(|(label, (count, ticks))| (label, count, ticks)).collect();
        ranked.sort_by_key(|entry| core::cmp::Reverse(entry.2));
        for (label, count, ticks) in ranked {
            std::eprintln!(
                "DIAG evaluate_quantized node_kind={label} count={count} total_ms={:.3} pct_of_forward={:.2}",
                instrument::ticks_to_nanos(ticks) as f64 / 1_000_000.0,
                100.0 * ticks as f64 / total_ticks as f64,
            );
        }
        std::eprintln!(
            "DIAG evaluate_quantized setup_ms={:.3} loop_overhead_ms={:.3}",
            instrument::ticks_to_nanos(diag_setup_ticks) as f64 / 1_000_000.0,
            instrument::ticks_to_nanos(diag_loop_overhead_ticks) as f64 / 1_000_000.0,
        );
        // DIAGNOSTIC (proxima-debugger, remove before landing): the fixed
        // per-node cost inside `run_elementwise_range`, split at the seam
        // this decode-speed investigation measured against -- setup (operand
        // span resolution, stride/gather scratch), the `step_values`
        // allocation, and the position loop. `snapshot_and_reset` so a caller
        // running several `evaluate_quantized` calls back to back (a decode
        // loop) gets one call's breakdown per printout, not a running total.
        let elementwise_calls = instrument::ELEMENTWISE_RANGE_CALLS.snapshot_and_reset();
        let elementwise_setup_ticks = instrument::ELEMENTWISE_SETUP_TICKS.snapshot_and_reset();
        let elementwise_step_values_ticks = instrument::ELEMENTWISE_STEP_VALUES_TICKS.snapshot_and_reset();
        let elementwise_loop_ticks = instrument::ELEMENTWISE_LOOP_TICKS.snapshot_and_reset();
        let elementwise_cohort_rounds = instrument::ELEMENTWISE_COHORT_ROUNDS.snapshot_and_reset();
        if elementwise_calls > 0 {
            std::eprintln!(
                "DIAG evaluate_quantized elementwise_breakdown calls={elementwise_calls} cohort_rounds={elementwise_cohort_rounds} setup_ms={:.3} step_values_ms={:.3} loop_ms={:.3} setup_ns_per_call={:.1} step_values_ns_per_call={:.1}",
                instrument::ticks_to_nanos(elementwise_setup_ticks) as f64 / 1_000_000.0,
                instrument::ticks_to_nanos(elementwise_step_values_ticks) as f64 / 1_000_000.0,
                instrument::ticks_to_nanos(elementwise_loop_ticks) as f64 / 1_000_000.0,
                instrument::ticks_to_nanos(elementwise_setup_ticks) as f64 / elementwise_calls as f64,
                instrument::ticks_to_nanos(elementwise_step_values_ticks) as f64 / elementwise_calls as f64,
            );
        }
        // achieved ns/element (nsper task, 2026-08-21): `MAC_OPS` is the
        // exact element denominator for `reduce_f32_dense` (see the comment
        // on `reduce_dense_entry` above); `reduce_dense_entry`'s ticks are
        // the SAME per-node wall time the `node_kind=reduce_f32_dense` row
        // above already printed, just paired with an element count here.
        let reduce_dense_mac_ops = instrument::MAC_OPS.snapshot_and_reset();
        if let Some((reduce_dense_count, reduce_dense_ticks)) = reduce_dense_entry {
            let reduce_dense_ns = instrument::ticks_to_nanos(reduce_dense_ticks) as f64;
            let ns_per_element = if reduce_dense_mac_ops > 0 {
                reduce_dense_ns / reduce_dense_mac_ops as f64
            } else {
                0.0
            };
            std::eprintln!(
                "DIAG nsper reduce_f32_dense calls={reduce_dense_count} mac_ops={reduce_dense_mac_ops} total_ms={:.3} ns_per_element={:.4}",
                reduce_dense_ns / 1_000_000.0,
                ns_per_element,
            );
        }
        // per-path-kind wall time within `reduce_f32_dense` (residual-profile
        // task, 2026-08-30): `instrument::record_reduce_path_ticks`'s own doc
        // explains why this is a separate, `run_reduce`-only timer. The sum
        // of these four should equal `reduce_dense_entry`'s own ticks above
        // (same node population, same `run_node_into` call boundary) --
        // reported separately, not reconciled here, so a divergence stays
        // visible rather than silently averaged away.
        let reduce_dot_fast_ticks = instrument::REDUCE_PATH_DOT_FAST_TICKS.snapshot_and_reset();
        let reduce_width_fast_ticks = instrument::REDUCE_PATH_WIDTH_FAST_TICKS.snapshot_and_reset();
        let reduce_conv_tile_ticks = instrument::REDUCE_PATH_CONV_TILE_TICKS.snapshot_and_reset();
        let reduce_generic_ticks = instrument::REDUCE_PATH_GENERIC_TICKS.snapshot_and_reset();
        std::eprintln!(
            "DIAG reduce_path_ticks dot_fast_ms={:.3} width_fast_ms={:.3} conv_tile_ms={:.3} generic_ms={:.3}",
            instrument::ticks_to_nanos(reduce_dot_fast_ticks) as f64 / 1_000_000.0,
            instrument::ticks_to_nanos(reduce_width_fast_ticks) as f64 / 1_000_000.0,
            instrument::ticks_to_nanos(reduce_conv_tile_ticks) as f64 / 1_000_000.0,
            instrument::ticks_to_nanos(reduce_generic_ticks) as f64 / 1_000_000.0,
        );
        // achieved ns/element split by `BodyShape` (nsper task, 2026-08-21):
        // `Unary`/`Binary` (monomorphic) versus `Generic` (fused multi-step)
        // -- a DIFFERENT axis from `Path::WidthFast`/`Path::Generic` (the
        // affine-operand fast-path gate) already printed via
        // `elementwise_breakdown` above. Compares directly against this
        // crate's own 0.38ns/element monomorphic figure (`cpu.rs:2159`).
        let monomorphic_ticks = instrument::ELEMENTWISE_LOOP_TICKS_MONOMORPHIC.snapshot_and_reset();
        let monomorphic_elements = instrument::ELEMENTWISE_ELEMENTS_MONOMORPHIC.snapshot_and_reset();
        let generic_ticks = instrument::ELEMENTWISE_LOOP_TICKS_GENERIC.snapshot_and_reset();
        let generic_elements = instrument::ELEMENTWISE_ELEMENTS_GENERIC.snapshot_and_reset();
        if monomorphic_elements > 0 {
            std::eprintln!(
                "DIAG nsper elementwise_monomorphic elements={monomorphic_elements} total_ms={:.3} ns_per_element={:.4}",
                instrument::ticks_to_nanos(monomorphic_ticks) as f64 / 1_000_000.0,
                instrument::ticks_to_nanos(monomorphic_ticks) as f64 / monomorphic_elements as f64,
            );
        }
        if generic_elements > 0 {
            std::eprintln!(
                "DIAG nsper elementwise_generic elements={generic_elements} total_ms={:.3} ns_per_element={:.4}",
                instrument::ticks_to_nanos(generic_ticks) as f64 / 1_000_000.0,
                instrument::ticks_to_nanos(generic_ticks) as f64 / generic_elements as f64,
            );
        }
        // fast_path-vs-slow-path split within `Unary`/`Binary` (`Monomorphic`)
        // (residual-profile task, 2026-08-30): mirrors the `Generic` split
        // immediately below -- answers whether `window_materialize`'s own
        // `Binary` multiply (Conv's im2col copy, `proxima-onnx/src/lower.rs`)
        // takes the affine width-copy fast path or falls to the per-element
        // gather loop despite being classified `Monomorphic`.
        let monomorphic_fast_ticks = instrument::ELEMENTWISE_LOOP_TICKS_MONOMORPHIC_FAST.snapshot_and_reset();
        let monomorphic_fast_elements = instrument::ELEMENTWISE_ELEMENTS_MONOMORPHIC_FAST.snapshot_and_reset();
        let monomorphic_slow_ticks = instrument::ELEMENTWISE_LOOP_TICKS_MONOMORPHIC_SLOW.snapshot_and_reset();
        let monomorphic_slow_elements = instrument::ELEMENTWISE_ELEMENTS_MONOMORPHIC_SLOW.snapshot_and_reset();
        if monomorphic_fast_elements > 0 {
            std::eprintln!(
                "DIAG nsper elementwise_monomorphic_fast_path elements={monomorphic_fast_elements} total_ms={:.3} ns_per_element={:.4}",
                instrument::ticks_to_nanos(monomorphic_fast_ticks) as f64 / 1_000_000.0,
                instrument::ticks_to_nanos(monomorphic_fast_ticks) as f64 / monomorphic_fast_elements as f64,
            );
        }
        if monomorphic_slow_elements > 0 {
            std::eprintln!(
                "DIAG nsper elementwise_monomorphic_slow_path elements={monomorphic_slow_elements} total_ms={:.3} ns_per_element={:.4}",
                instrument::ticks_to_nanos(monomorphic_slow_ticks) as f64 / 1_000_000.0,
                instrument::ticks_to_nanos(monomorphic_slow_ticks) as f64 / monomorphic_slow_elements as f64,
            );
        }
        // window-materialize-shaped copy split within `monomorphic_fast_path`
        // (rung 2, `docs/discipline.md` ROW 154): `window_copy_operand`'s own
        // specialized row-segment copy vs the remaining `elementwise_width_fast`
        // per-row dispatch, both members of `monomorphic_fast_path` above.
        let window_copy_ticks = instrument::ELEMENTWISE_LOOP_TICKS_WINDOW_COPY.snapshot_and_reset();
        let window_copy_elements = instrument::ELEMENTWISE_ELEMENTS_WINDOW_COPY.snapshot_and_reset();
        if window_copy_elements > 0 {
            std::eprintln!(
                "DIAG nsper elementwise_window_copy elements={window_copy_elements} total_ms={:.3} ns_per_element={:.4}",
                instrument::ticks_to_nanos(window_copy_ticks) as f64 / 1_000_000.0,
                instrument::ticks_to_nanos(window_copy_ticks) as f64 / window_copy_elements as f64,
            );
        }
        // fast_path-vs-slow-path split within `Generic` (A-vs-B task,
        // 2026-08-21): answers whether `Generic`'s 14.9x-slower-than-
        // monomorphic figure is (A) almost all elements falling to the
        // per-element `apply_body` gather loop (`fast_path=false`) or (B)
        // the affine fast path itself running ~15x off the monomorphic
        // rate. Same ticks/elements the `elementwise_generic` row above
        // already summed, split by the `fast_path` bool computed once per
        // call in `run_elementwise_range`.
        let generic_fast_ticks = instrument::ELEMENTWISE_LOOP_TICKS_GENERIC_FAST.snapshot_and_reset();
        let generic_fast_elements = instrument::ELEMENTWISE_ELEMENTS_GENERIC_FAST.snapshot_and_reset();
        let generic_slow_ticks = instrument::ELEMENTWISE_LOOP_TICKS_GENERIC_SLOW.snapshot_and_reset();
        let generic_slow_elements = instrument::ELEMENTWISE_ELEMENTS_GENERIC_SLOW.snapshot_and_reset();
        if generic_fast_elements > 0 {
            std::eprintln!(
                "DIAG nsper elementwise_generic_fast_path elements={generic_fast_elements} total_ms={:.3} ns_per_element={:.4}",
                instrument::ticks_to_nanos(generic_fast_ticks) as f64 / 1_000_000.0,
                instrument::ticks_to_nanos(generic_fast_ticks) as f64 / generic_fast_elements as f64,
            );
        }
        if generic_slow_elements > 0 {
            std::eprintln!(
                "DIAG nsper elementwise_generic_slow_path elements={generic_slow_elements} total_ms={:.3} ns_per_element={:.4}",
                instrument::ticks_to_nanos(generic_slow_ticks) as f64 / 1_000_000.0,
                instrument::ticks_to_nanos(generic_slow_ticks) as f64 / generic_slow_elements as f64,
            );
        }
        // call-size distribution (nsper task, 2026-08-21): how many
        // `run_elementwise_range` calls landed at each element count this
        // step -- answers "numerous small calls" vs "few large ones"
        // without guessing from a median call count alone.
        let mut call_sizes = instrument::elementwise_call_size_snapshot_and_reset();
        call_sizes.sort_by_key(|(size, _)| *size);
        let total_size_calls: u64 = call_sizes.iter().map(|(_, count)| *count).sum();
        for (size, count) in &call_sizes {
            std::eprintln!(
                "DIAG nsper elementwise_call_size elements={size} calls={count} pct_of_calls={:.2}",
                100.0 * *count as f64 / total_size_calls.max(1) as f64,
            );
        }
    }
    #[cfg(feature = "instrument")]
    let diag_finish_started = instrument::read_ticks();

    let result = finish(&shapes, &effective_outputs, buffers, root, peak_live_buffers);
    #[cfg(feature = "instrument")]
    std::eprintln!(
        "DIAG evaluate_quantized finish_ms={:.3}",
        instrument::ticks_to_nanos(instrument::elapsed_ticks(diag_finish_started)) as f64 / 1_000_000.0,
    );
    Ok(result)
}

/// [`evaluate_quantized`]'s counterpart for binding by name instead of
/// position — the same relationship [`evaluate_named`] has to [`evaluate`],
/// and the same resolution loop: walk `program`'s [`Op::Input`] nodes in
/// order, look each one's name up in `named`, and hand the resolved
/// positional `blocks: &[QuantizedBlock]` straight to [`evaluate_quantized`].
/// [`evaluate_named`] is now this function plus one wrapping step
/// (`QuantizedBlock::Float32`), rather than a second copy of this loop —
/// there is exactly one name-to-[`Op::Input`] resolution in this module.
pub fn evaluate_quantized_named<'block>(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'block>)],
    outputs: &[NodeId],
) -> Result<Evaluated, TensorError> {
    let mut free_buffers: Vec<Vec<f32>> = Vec::new();
    let mut validated_weight_nodes: Option<BTreeSet<NodeId>> = None;
    evaluate_quantized_named_with_scratch(
        program,
        symbols,
        named,
        outputs,
        &mut free_buffers,
        &mut validated_weight_nodes,
    )
}

/// [`evaluate_quantized_named`]'s counterpart to
/// [`evaluate_quantized_with_scratch`] — the same name-to-[`Op::Input`]
/// resolution loop, handing the resolved positional `blocks` and both
/// caller-carried pools straight through rather than duplicating either
/// Resolves a name-keyed block set into the positional order a program's
/// [`Op::Input`] nodes appear in.
///
/// Public and shared rather than private to the CPU evaluator: omega's Metal
/// driver binds blocks positionally too, and a second copy of this mapping
/// is a second thing that can drift from what the program actually declares.
/// The two backends already share [`QuantizedBlock`]; they share how a name
/// becomes a position as well.
///
/// # Errors
/// [`TensorError::UnnamedInput`] if a block input carries no name,
/// [`TensorError::UnboundInputName`] if `named` has no entry for one.
pub fn resolve_named_blocks<'block>(
    program: &[Op],
    named: &[(&str, QuantizedBlock<'block>)],
) -> Result<Vec<QuantizedBlock<'block>>, TensorError> {
    let block_nodes = block_node_ids(program);
    let mut blocks: Vec<QuantizedBlock<'block>> = Vec::with_capacity(block_nodes.len());
    for node in &block_nodes {
        let name = program[node.0 as usize]
            .name()
            .ok_or(TensorError::UnnamedInput(*node))?;
        let data = named
            .iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|(_, data)| *data)
            .ok_or_else(|| TensorError::UnboundInputName(String::from(name)))?;
        blocks.push(data);
    }
    Ok(blocks)
}

/// evaluator's body a third time.
pub fn evaluate_quantized_named_with_scratch<'block>(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'block>)],
    outputs: &[NodeId],
    free_buffers: &mut Vec<Vec<f32>>,
    validated_weight_nodes: &mut Option<BTreeSet<NodeId>>,
) -> Result<Evaluated, TensorError> {
    // DIAGNOSTIC (proxima-debugger, remove before landing): this name
    // resolution runs before evaluate_quantized's own diag_setup_started
    // timer starts, so it is invisible to every counter that function
    // reports -- a linear `find` over `named` per weight tensor, O(block
    // count * named count) string compares.
    #[cfg(feature = "instrument")]
    let diag_resolve_started = instrument::read_ticks();
    let blocks = resolve_named_blocks(program, named)?;
    #[cfg(feature = "instrument")]
    std::eprintln!(
        "DIAG evaluate_quantized_named resolve_ms={:.3} block_count={}",
        instrument::ticks_to_nanos(instrument::elapsed_ticks(diag_resolve_started)) as f64 / 1_000_000.0,
        blocks.len(),
    );
    evaluate_quantized_with_scratch(program, symbols, &blocks, outputs, free_buffers, validated_weight_nodes)
}

/// Shared body for [`evaluate`] and [`evaluate_with_scratch`] — the only
/// difference between the two public entry points is whether `free_buffers`
/// arrives pre-seeded and is read back by the caller afterward, so that
/// decision is made once, here, by each caller passing its own `Vec` (fresh
/// and discarded, or threaded through `scratch`) rather than by two
/// divergent copies of this loop.
///
/// Drives [`run_node_into`] directly, one node per iteration, rather than
/// through [`Interpreter::fold`]: `Interpreter` exposes no way to hand a
/// node its output storage from a pool instead of a fresh `vec![0.0; n]`
/// (its `Pipe::Out = ()` contract has no room for one), so reusing the same
/// execution primitive both callers already share (`run_node_into`, also
/// `evaluate_parallel`'s) is what keeps this a single behavior rather than a
/// second copy of the per-node dispatch match. `Interpreter` remains exactly
/// as it was for its own callers (the `shapes.and_then(builder)
/// .and_then(Interpreter::new(..))` `Pipe` chain a test in this module
/// exercises directly) — this function simply no longer routes through it.
fn evaluate_pooled(
    program: &[Op],
    symbols: &[u64],
    blocks: &[&[f32]],
    outputs: &[NodeId],
    free_buffers: &mut Vec<Vec<f32>>,
) -> Result<Evaluated, TensorError> {
    #[cfg(feature = "instrument")]
    let alloc_site_guard = instrument::AllocSiteGuard::enter(instrument::AllocSite::Prepare);
    let Prepared {
        root,
        shapes,
        effective_outputs,
        mut buffers,
        resolved,
        retires,
    } = prepare(program, symbols, blocks, outputs)?;
    #[cfg(feature = "instrument")]
    drop(alloc_site_guard);

    // `live_now`: O(1) running live-buffer count, incremented/decremented
    // alongside this loop's own `Some`/`None` writes -- see
    // `evaluate_quantized`'s identical `live_now` doc for why this replaces
    // `live_count(&buffers)`'s O(program.len()) full rescan per node (the
    // quadratic cost this landing removes).
    let mut peak_live_buffers = live_count(&buffers);
    let mut live_now = peak_live_buffers;
    for (position, computed) in resolved.iter().enumerate() {
        #[cfg(feature = "instrument")]
        let alloc_site_guard =
            instrument::AllocSiteGuard::enter(instrument::AllocSite::OutputBuffer);
        let mut output = take_or_allocate(free_buffers, node_output_len(computed));
        #[cfg(feature = "instrument")]
        drop(alloc_site_guard);
        run_node_into(computed, &buffers, None, None, &mut output)?;
        #[cfg(feature = "instrument")]
        record_bound_op_operand_access(computed, &buffers);
        buffers[computed.node.0 as usize] = Some(Cow::Owned(output));
        live_now += 1;
        peak_live_buffers = peak_live_buffers.max(live_now);
        for retired in &retires[position] {
            // `blocks: &[&[f32]]` means every node this evaluator ever
            // touches is float32 and lands in `buffers` -- no quantized-weight
            // split to trip over, unlike `evaluate_quantized` -- but the
            // decrement is still gated on `retire_into`'s liveness report
            // rather than assumed, so this loop stays correct if that ever
            // changes rather than relying on an invariant nothing enforces.
            if retire_into(&mut buffers, *retired, free_buffers) {
                live_now -= 1;
            }
        }
    }

    Ok(finish(&shapes, &effective_outputs, buffers, root, peak_live_buffers))
}

/// Takes the buffer at `node`'s slot (leaving `None` behind, exactly as
/// before buffer reuse existed) and, if it was this evaluator's own owned
/// storage rather than a caller-borrowed [`Op::Input`] slice, stashes it in
/// `pool` for [`take_or_allocate`] to hand to a later node instead of the
/// allocator. Sound because this only ever runs after the position that
/// retired `node` has already finished reading every one of its operands
/// (`node_retirement` records a node's *last* read position, and the caller
/// only retires after `run_node_into` for that position has returned) — the
/// buffer is out of `buffers` and not yet read by anything else before it
/// lands in `pool`, so no live reference to it survives the swap.
///
/// Returns whether `node`'s slot actually held a buffer. `node_retirement`
/// schedules a retirement for every operand the program graph reads, with no
/// knowledge of `evaluate_quantized`'s split storage: a `Q4_K`/`Q5_K`/`Q6_K`/
/// `Q8_0` weight node never occupies a `buffers` slot at all (it lives in the
/// separate `quantized_weights` map instead — see the `QuantizedBlock`
/// match in `evaluate_quantized_with_scratch`), so its "retirement" here is a
/// no-op on an already-`None` slot. A caller that decremented a running live
/// count unconditionally on every retirement (as this evaluator's `live_now`
/// used to) drifted low by one per quantized weight and could underflow.
fn retire_into(buffers: &mut [Option<Cow<'_, [f32]>>], node: NodeId, pool: &mut Vec<Vec<f32>>) -> bool {
    match buffers[node.0 as usize].take() {
        Some(Cow::Owned(buffer)) => {
            pool.push(buffer);
            true
        }
        Some(Cow::Borrowed(_)) => true,
        None => false,
    }
}

/// Hands out `required` elements of storage from `pool` when a sufficiently
/// large entry exists, or allocates fresh otherwise — the one place
/// [`evaluate_pooled`] gets a node's output buffer from. Every element
/// [`run_node_into`]'s callers write is unconditionally overwritten before
/// any node reads it back (`run_elementwise`, `run_reduce`'s NEON/tile/
/// fallback paths, and `run_scan` each write every output position once),
/// so a reused buffer's stale contents never leak into a result;
/// `Vec::resize`'s growth path only fires, and only fills the delta, when
/// `pool` had nothing big enough already, so the fill this replaces shrinks
/// to zero once `pool` reaches its working set's high-water mark.
fn take_or_allocate(pool: &mut Vec<Vec<f32>>, required: usize) -> Vec<f32> {
    let best_fit = pool
        .iter()
        .enumerate()
        .filter(|(_, buffer)| buffer.capacity() >= required)
        .min_by_key(|(_, buffer)| buffer.capacity())
        .map(|(index, _)| index);

    match best_fit {
        Some(index) => {
            let mut buffer = pool.swap_remove(index);
            buffer.resize(required, 0.0);
            buffer
        }
        None => vec![0.0f32; required],
    }
}

/// Below this many iteration-space elements, a nest runs the plain
/// sequential path even when `workers > 1`: `std::thread::scope`'s spawn
/// and join overhead outweighs the work for a small nest.
use crate::sized::PARALLEL_THRESHOLD;

/// chunk count is `workers * OVERSUBSCRIBE`, not `workers`: equal row
/// counts do not mean equal wall-clock (measured 2.04x spread across 8
/// equal-row chunks of a 1024^3 GEMM), and one chunk per worker leaves no
/// spare chunk for a worker that finishes early to pick up — more chunks
/// than workers lets a fast worker absorb a slow chunk's slack. Only pays
/// off under [`nest_pool`]'s dynamic claiming (see `claim_and_run`), the
/// only chunk dispatch this module has: the puller count [`run_chunks_threaded`]
/// spawns caps at `workers` regardless of `OVERSUBSCRIBE`, so raising this
/// grows the number of chunks a fixed puller count can steal from, without
/// growing the number of threads touching them.
///
/// A `4` was tried on this same mechanism: more chunks than workers gives a
/// work-stealing pool room for a fast puller to absorb a slow chunk's slack,
/// which is structural and not in dispute. The comparison that measured it —
/// 274.75 vs 270.08 mean GFLOPS at 2048^3/4 workers, n=9, 5.9 sigma — never
/// recorded the ambient load it ran under, and the 8-worker cells it was
/// meant to help never cleared their own CoV gate at any sample size tried
/// (n up to 30) under the load present when those cells were measured (5-10,
/// against a stated 2.2 plateau). That is the same evidence shape that
/// produced three false readings for `SPLIT_ALIGNMENT` on this same box:
/// strong sigma inside an unvalidated run does not rule out noise correlated
/// across that run rather than random within it. Left at `1`, the original
/// value, until a re-measurement validates its own floor first — a
/// same-code-path comparison, at a size and load where the two configurations
/// provably execute identical instructions — and only then shows an
/// oversubscription effect outside it.
use crate::sized::OVERSUBSCRIBE;

/// Row-alignment applied to every non-final chunk boundary via
/// `BoundOp::split_aligned`. `1` is a no-op (see that method's doc): every
/// chunk boundary lands wherever `extent / chunk_count` puts it, which is
/// not necessarily a multiple of `TILE_ROWS` — so a chunk pays its own
/// row-remainder through the kernel's narrower fallback path independently
/// of every other chunk, and that per-chunk remainder count grows with
/// chunk count even though the total row count did not change. That
/// mechanism is structural and not in dispute; whether it moves
/// busy-per-MAC by a measurable amount is.
///
/// Four measurements of this same `1` -> `TILE_ROWS` change exist, all
/// against the column-panel blocking below (already landed and left on —
/// see `NEON_COLUMN_PANEL_BUDGET_BYTES`). Three, run at system load
/// 12-31, read as 3-10% busy-per-MAC improvements. None of the three
/// established a noise floor before comparing — at that load level a
/// handful of percent between two configurations is not distinguishable
/// from scheduler contention, so those figures are retained here only as
/// unverified prior readings, not as evidence.
///
/// The fourth run validated a floor first: at load 2.49-3.07, `neither` vs
/// `panel` at 512^3 and 1024^3 — sizes where `neon_column_panel_cols`
/// provably clamps the panel to one, i.e. the two configurations execute
/// identical code — agreed to within +/-3.5%, sigma up to 3.1. That is the
/// noise floor any real effect at this load has to clear. Against it,
/// alignment (`1` -> `TILE_ROWS`) on top of the panel measured
/// +2.03% / +0.31% / -0.72% at 512^3 / 1024^3 / 2048^3, 8 threads — inside
/// the floor at every size. No measurable effect anywhere in the one
/// comparison whose noise floor is known.
///
/// Set to `1` on that basis: the only measurement with a validated floor
/// found nothing outside it, and the three load-12-31 figures were never
/// shown to clear their own (unmeasured) noise, so they carry no weight
/// against it. This changes only if a future re-measurement (a) validates
/// its own floor the same way — a same-code-path comparison at a size
/// where the panel is a no-op — and (b) then shows an alignment effect
/// outside that floor.
///
/// Provenance: the three load-12-31 figures were not independently dated
/// or sample-counted in the record available to this pass — treat them as
/// unverified, not merely old. The load-2.49-3.07 measurement is this
/// session's own, 2026-08-18; its sample count for the alignment
/// comparison specifically is not broken out beyond the three-configuration
/// grid it ran alongside.
use crate::sized::SPLIT_ALIGNMENT;

/// Same contract as [`evaluate`], including the exact same [`Evaluated`]
/// and error variants — the only difference is that each large-enough nest
/// runs its chunks across `workers` pool tasks via `run_chunks_threaded`
/// (the shared `nest_pool`), each writing a disjoint sub-slice of that
/// nest's own output buffer (see [`BoundOp::split`]). The preamble is
/// `prepare`, the same one [`evaluate`] runs — the two functions diverge
/// only in the loop below.
pub fn evaluate_parallel(
    program: &[Op],
    symbols: &[u64],
    blocks: &[&[f32]],
    outputs: &[NodeId],
    workers: NonZeroUsize,
) -> Result<Evaluated, TensorError> {
    #[cfg(feature = "instrument")]
    let evaluate_parallel_start = instrument::read_ticks();

    #[cfg(feature = "instrument")]
    let prepare_start = instrument::read_ticks();
    #[cfg(feature = "instrument")]
    let alloc_site_guard = instrument::AllocSiteGuard::enter(instrument::AllocSite::Prepare);
    let Prepared {
        root,
        shapes,
        effective_outputs,
        mut buffers,
        resolved,
        retires,
    } = prepare(program, symbols, blocks, outputs)?;
    #[cfg(feature = "instrument")]
    drop(alloc_site_guard);
    #[cfg(feature = "instrument")]
    counter!(
        instrument::SERIAL_PREPARE_TICKS,
        instrument::elapsed_ticks(prepare_start)
    );

    // `live_now`: O(1) running live-buffer count -- see `evaluate_quantized`'s
    // identical `live_now` doc for why this replaces `live_count(&buffers)`'s
    // O(program.len()) full rescan per node.
    let mut peak_live_buffers = live_count(&buffers);
    let mut live_now = peak_live_buffers;
    for (position, computed) in resolved.iter().enumerate() {
        let output = evaluate_node_parallel(computed, &buffers, workers)?;
        #[cfg(feature = "instrument")]
        let bookkeeping_start = instrument::read_ticks();
        buffers[computed.node.0 as usize] = Some(Cow::Owned(output));
        live_now += 1;
        peak_live_buffers = peak_live_buffers.max(live_now);
        for retired in &retires[position] {
            // same liveness-gated decrement as `evaluate_pooled`'s identical
            // loop -- `blocks: &[&[f32]]` means no quantized-weight split
            // exists here today, but the decrement stays conditioned on the
            // slot actually having held a buffer rather than assumed.
            if buffers[retired.0 as usize].take().is_some() {
                live_now -= 1;
            }
        }
        #[cfg(feature = "instrument")]
        counter!(
            instrument::SERIAL_BOOKKEEPING_TICKS,
            instrument::elapsed_ticks(bookkeeping_start)
        );
    }

    #[cfg(feature = "instrument")]
    let finish_start = instrument::read_ticks();
    let evaluated = finish(&shapes, &effective_outputs, buffers, root, peak_live_buffers);
    #[cfg(feature = "instrument")]
    {
        counter!(
            instrument::SERIAL_FINISH_TICKS,
            instrument::elapsed_ticks(finish_start)
        );
        counter!(
            instrument::SERIAL_EVALUATE_PARALLEL_TICKS,
            instrument::elapsed_ticks(evaluate_parallel_start)
        );
        counter!(instrument::SERIAL_EVALUATE_PARALLEL_CALLS, 1);
    }

    Ok(evaluated)
}

/// Runs one node, threaded across `workers` when [`BoundOp::split`] finds it
/// sound and it clears [`PARALLEL_THRESHOLD`]; otherwise the plain
/// sequential path via [`run_node_into`].
fn evaluate_node_parallel<B: Deref<Target = [f32]> + Sync>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    workers: NonZeroUsize,
) -> Result<Vec<f32>, TensorError> {
    #[cfg(feature = "instrument")]
    let alloc_site_guard = instrument::AllocSiteGuard::enter(instrument::AllocSite::OutputBuffer);
    #[cfg(feature = "instrument")]
    let alloc_start = instrument::read_ticks();
    let mut output = vec![0.0f32; node_output_len(resolved)];
    #[cfg(feature = "instrument")]
    counter!(
        instrument::SERIAL_ALLOC_TICKS,
        instrument::elapsed_ticks(alloc_start)
    );
    #[cfg(feature = "instrument")]
    drop(alloc_site_guard);

    #[cfg(feature = "instrument")]
    let split_start = instrument::read_ticks();
    let above_threshold = element_count(&resolved.extents) >= PARALLEL_THRESHOLD;
    // oversubscribing at `workers == 1` would still spawn `OVERSUBSCRIBE - 1`
    // pool tasks (chunk count alone bounds pool concurrency — see
    // `run_chunks_threaded`'s doc), silently using more physical threads
    // than the caller asked for; only multiply once there is more than one
    // worker to spread chunks across.
    let chunk_count = if workers.get() > 1 {
        workers.get() * OVERSUBSCRIBE
    } else {
        workers.get()
    };
    let chunks = above_threshold
        .then(|| resolved.split_aligned(chunk_count, SPLIT_ALIGNMENT))
        .flatten();
    #[cfg(feature = "instrument")]
    counter!(
        instrument::SERIAL_SPLIT_TICKS,
        instrument::elapsed_ticks(split_start)
    );

    match chunks {
        Some(chunks) => run_chunks_threaded(&chunks, buffers, &mut output, workers)?,
        None => {
            // one node, one dispatch decision — recorded once here, not
            // re-derived from `above_threshold` after the fact, since the
            // `chunks` match is the actual arm that ran.
            #[cfg(feature = "instrument")]
            if above_threshold {
                counter!(instrument::DISPATCH_SEQUENTIAL_SPLIT_UNAVAILABLE, 1);
            } else {
                counter!(instrument::DISPATCH_SEQUENTIAL_BELOW_THRESHOLD, 1);
            }
            #[cfg(feature = "instrument")]
            let sequential_start = instrument::read_ticks();
            run_node_into(resolved, buffers, None, None, &mut output)?;
            #[cfg(feature = "instrument")]
            counter!(
                instrument::SERIAL_SEQUENTIAL_COMPUTE_TICKS,
                instrument::elapsed_ticks(sequential_start)
            );
        }
    }
    // attributed against the PARENT (unsplit) `resolved`, not any one
    // spawned chunk above — see `record_bound_op_operand_access`'s doc for
    // why: a chunk's own shrunk extents would double-count a broadcast
    // operand's footprint once per chunk instead of once for this node.
    #[cfg(feature = "instrument")]
    record_bound_op_operand_access(resolved, buffers);
    Ok(output)
}

/// Runs each of `chunks` through the shared [`nest_pool`] (crossbeam-deque
/// work-stealing, built once and reused for every nest in the process)
/// instead of spawning a fresh OS thread per chunk. `std` implies
/// `tensor-bgpool` (`Cargo.toml`'s `std` feature doc), so this is the only
/// dispatch this module ever compiles — a fresh-`thread::scope`-per-call
/// sibling used to sit here and was removed once nothing could select it
/// anymore.
///
/// Every chunk, including the caller's own, is pulled off one shared
/// `next_index` cursor (see [`claim_and_run`]) instead of being statically
/// assigned: the calling thread and every pool task run the identical pull
/// loop, so a puller that finishes its chunk early goes straight back to
/// the cursor for the next available one rather than idling — this is what
/// lets `OVERSUBSCRIBE > 1` (`chunks.len() > workers`) actually pay off:
/// with a 1:1 static assignment (the previous shape here), a pool task
/// finishing early has nothing further to do even when a sibling chunk is
/// still running long past it. `workers.get() - 1` pool tasks are spawned —
/// the caller's requested puller count, not `chunks.len() - 1` — because
/// [`nest_pool`] is a single process-wide pool sized to `num_cpus`, shared
/// across every call regardless of its own `workers` argument: spawning one
/// task per chunk would let oversubscription silently recruit pool threads
/// past what the caller asked for on any box where `num_cpus > workers`.
/// Each spawned puller still drains the same shared cursor across every
/// chunk, so raising `OVERSUBSCRIBE` still grows the number of chunks a
/// fixed `workers` pullers can steal from, without growing puller count.
/// Completion is a real blocking handoff (`std::sync::mpsc::sync_channel`),
/// not a poll loop: the caller parks in `Receiver::recv` instead of
/// busy-spinning a `Waker::noop` future the way `proxima_primitives::block_on`
/// would.
///
/// A worker panic cannot be resumed on the joining thread the way a
/// `thread::scope`-spawned one could: `ProximaBackgroundPool`'s worker loop
/// wraps every job in `catch_unwind` and discards the payload
/// (`prime/src/os/background.rs`, `worker()`, `let _ = unwind;`),
/// converting a panic into a dropped closure with no way to recover the
/// original payload. That drop takes our own `sync_channel` sender clone
/// with it, so a panicking chunk never reports back; a chunk that never
/// reports is surfaced as `TensorError::ThreadedChunkFailed` instead.
fn run_chunks_threaded<B: Deref<Target = [f32]> + Sync>(
    chunks: &[BoundOp],
    buffers: &[Option<B>],
    output: &mut [f32],
    workers: NonZeroUsize,
) -> Result<(), TensorError> {
    #[cfg(feature = "instrument")]
    let slice_carve_start = instrument::read_ticks();
    #[cfg(feature = "instrument")]
    let alloc_site_guard = instrument::AllocSiteGuard::enter(instrument::AllocSite::ChunkSlices);

    let mut slices = Vec::with_capacity(chunks.len());
    let mut remaining = output;
    for chunk in chunks {
        let (this_chunk, rest) = remaining.split_at_mut(node_output_len(chunk));
        slices.push(this_chunk);
        remaining = rest;
    }
    #[cfg(feature = "instrument")]
    drop(alloc_site_guard);
    #[cfg(feature = "instrument")]
    counter!(
        instrument::SERIAL_SLICE_CARVE_TICKS,
        instrument::elapsed_ticks(slice_carve_start)
    );

    if chunks.len() < 2 {
        return match (chunks.first(), slices.into_iter().next()) {
            (Some(chunk), Some(slice)) => run_node_into(chunk, buffers, None, None, slice),
            _ => Ok(()),
        };
    }

    let pool = nest_pool()?;

    // `buffers` is read-only for the whole call and every spawned chunk
    // needs it; cloning would copy every live intermediate tensor per
    // chunk. its address crosses the pool's 'static spawn bound the same
    // way `par_chunks_mut` (prime/src/os/par.rs:1611-1625) already does for
    // its own slice: cast to usize here, reconstruct unsafely inside the
    // closure. sound because `buffers` outlives every spawned closure — the
    // caller thread drains `result_receiver` for every chunk before this
    // function returns.
    let buffers_address = buffers.as_ptr() as usize;
    let buffers_len = buffers.len();
    // same cast, same soundness argument, for `chunks` itself: every puller
    // now needs random access to an arbitrary chunk, not just the one it
    // was statically handed.
    let chunks_address = chunks.as_ptr() as usize;
    let chunks_len = chunks.len();
    // each chunk's own disjoint output sub-slice, addressed by index so any
    // puller (caller or pool task) can claim any chunk — `Arc` because,
    // unlike `buffers`/`chunks` above, this vector is allocated fresh here
    // rather than borrowed from the caller, so it needs its own shared
    // ownership to reach every spawned closure.
    let slice_addresses: Arc<Vec<(usize, usize)>> = Arc::new(
        slices
            .iter_mut()
            .map(|slice| (slice.as_mut_ptr() as usize, slice.len()))
            .collect(),
    );

    let next_index = Arc::new(AtomicUsize::new(0));
    let (result_sender, result_receiver) = sync_channel(chunks_len);

    #[cfg(feature = "instrument")]
    let node_start = instrument::read_ticks();

    // `workers - 1` pool tasks — the puller count the caller actually asked
    // for, NOT `chunks_len - 1`. `nest_pool` is sized to `num_cpus`, shared
    // and reused process-wide, independent of any one call's `workers`
    // argument: spawning one task per CHUNK (as this used to) rather than
    // one per WORKER let `OVERSUBSCRIBE > 1` silently recruit pool threads
    // past the caller's requested count — up to `num_cpus`, on a box where
    // `num_cpus > workers` — since nothing here otherwise bounds how many
    // of the pool's own threads can be pulling `claim_and_run` at once.
    // Each spawned puller still drains the shared cursor across every one
    // of the `chunks_len` chunks, exactly like the caller does below, so
    // oversubscription still grows the number of *chunks* available to
    // steal without growing the number of *threads* touching them.
    for _ in 0..workers.get() - 1 {
        let sender = result_sender.clone();
        let next_index = Arc::clone(&next_index);
        let slice_addresses = Arc::clone(&slice_addresses);
        // the pool's own returned future only reports back through its
        // internal oneshot channel, which nothing here awaits — completion
        // is reported through `sender` instead, so the future is dropped
        // deliberately rather than driven.
        drop(pool.spawn(move || {
            claim_and_run::<B>(
                &next_index,
                chunks_address,
                chunks_len,
                buffers_address,
                buffers_len,
                &slice_addresses,
                &sender,
            );
            Ok::<(), _>(())
        }));
    }

    #[cfg(feature = "instrument")]
    let spawn_ticks = instrument::elapsed_ticks(node_start);

    // the caller pulls from the same shared cursor as every pool task
    // instead of running one reserved chunk — see this function's doc
    // comment for why. it never sits idle: finishing a chunk sends it
    // straight back to `next_index` for another.
    claim_and_run::<B>(
        &next_index,
        chunks_address,
        chunks_len,
        buffers_address,
        buffers_len,
        &slice_addresses,
        &result_sender,
    );
    drop(result_sender);

    let mut outcomes: Vec<Option<Result<(), TensorError>>> =
        (0..chunks_len).map(|_| None).collect();
    for _ in 0..chunks_len {
        match result_receiver.recv() {
            Ok((index, outcome)) => outcomes[index] = Some(outcome),
            // every sender clone is gone (each spawned closure's clone is
            // dropped whether it sends or panics), so no further chunk will
            // ever report — stop waiting instead of blocking forever on a
            // message that cannot arrive. remaining `None` slots below
            // become `ThreadedChunkFailed`.
            Err(_) => break,
        }
    }

    #[cfg(feature = "instrument")]
    {
        let total_ticks = instrument::elapsed_ticks(node_start);
        counter!(instrument::PARALLEL_NODES, 1);
        counter!(instrument::PARALLEL_NODE_TICKS, total_ticks);
        counter!(instrument::PARALLEL_SPAWN_TICKS, spawn_ticks);
        // join/teardown is whatever wall-clock the node spent that wasn't
        // already charged to spawning the pool tasks — includes the
        // caller's own claim_and_run loop, same as the thread::scope
        // sibling's join/teardown includes its own compute.
        counter!(
            instrument::PARALLEL_JOIN_TICKS,
            total_ticks.saturating_sub(spawn_ticks)
        );
    }

    for (index, outcome) in outcomes.into_iter().enumerate() {
        match outcome {
            Some(result) => result?,
            None => {
                return Err(TensorError::ThreadedChunkFailed {
                    chunk: index + 1,
                    reason: alloc::string::String::from(
                        "worker did not report a result; ProximaBackgroundPool \
                         catches and discards worker panics (see \
                         prime/src/os/background.rs worker())",
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Pulls chunk indices off `next_index` one at a time and runs each to
/// completion, reporting through `sender` — called by both the calling
/// thread and every spawned pool task in [`run_chunks_threaded`], so a
/// puller that finishes early goes straight back for the next available
/// chunk instead of stopping after whichever one it started with.
///
/// # Safety (of the `unsafe` blocks inside)
/// `chunks_address`/`buffers_address` and every `(pointer, len)` pair in
/// `slice_addresses` must stay valid, and each slice address must be unique
/// to its index, for as long as any puller can still observe `next_index`
/// below `chunks_len` — guaranteed by [`run_chunks_threaded`] draining
/// `chunks_len` results from `sender`'s channel before `chunks`, `buffers`,
/// or `output` (the parent of every `slice_addresses` entry) can drop.
/// `fetch_add` never hands out the same index twice, so no two pullers ever
/// touch the same slice.
fn claim_and_run<B: Deref<Target = [f32]> + Sync>(
    next_index: &AtomicUsize,
    chunks_address: usize,
    chunks_len: usize,
    buffers_address: usize,
    buffers_len: usize,
    slice_addresses: &[(usize, usize)],
    sender: &SyncSender<(usize, Result<(), TensorError>)>,
) {
    loop {
        let index = next_index.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        if index >= chunks_len {
            return;
        }
        // SAFETY: see this function's doc comment.
        let chunk = unsafe { &*(chunks_address as *const BoundOp).add(index) };
        let chunk_buffers = unsafe {
            core::slice::from_raw_parts(buffers_address as *const Option<B>, buffers_len)
        };
        let (slice_address, slice_len) = slice_addresses[index];
        let chunk_output =
            unsafe { core::slice::from_raw_parts_mut(slice_address as *mut f32, slice_len) };

        #[cfg(feature = "instrument")]
        let chunk_start = instrument::read_ticks();
        #[cfg(feature = "instrument")]
        let cpu_start = instrument::thread_cpu_nanos();
        let outcome = run_node_into(chunk, chunk_buffers, None, None, chunk_output);
        #[cfg(feature = "instrument")]
        {
            let chunk_ticks = instrument::elapsed_ticks(chunk_start);
            let cpu_nanos = instrument::thread_cpu_nanos() - cpu_start;
            instrument::record_chunk_ticks(chunk_ticks);
            instrument::record_worker_busy_ticks(chunk_ticks);
            instrument::record_worker_cpu_nanos(instrument::CpuWorkload::Elementwise, cpu_nanos);
        }

        let _ = sender.send((index, outcome));
    }
}

/// The pool backing [`run_chunks_threaded`]'s and [`matmul_rows_threaded`]'s
/// chunk dispatch. Built once, on first use, and reused for every nest in
/// the process — a fresh `ProximaBackgroundPool` per node would reintroduce
/// the per-node OS-thread-spawn cost this pool exists to remove.
/// `OnceLock` only memoizes success: a failed build is not cached, so a
/// later call (after whatever exhausted OS thread resources clears up) can
/// retry instead of latching a permanent failure.
fn nest_pool() -> Result<Arc<ProximaBackgroundPool>, TensorError> {
    if let Some(pool) = NEST_POOL.get() {
        return Ok(Arc::clone(pool));
    }
    let built = Arc::new(ProximaBackgroundPool::new().map_err(|error| {
        TensorError::ThreadedPoolUnavailable(alloc::format!("build nest thread pool: {error}"))
    })?);
    // `set` can lose a race to a concurrent first caller; either pool is
    // equally valid, so use whichever one actually landed.
    let _ = NEST_POOL.set(Arc::clone(&built));
    Ok(NEST_POOL.get().cloned().unwrap_or(built))
}

static NEST_POOL: OnceLock<Arc<ProximaBackgroundPool>> = OnceLock::new();

/// The fixed-cohort spin barrier backing [`matmul_rows_threaded`]'s cohort
/// dispatch (see [`RowRound`]): dedicated member threads that stay parked on
/// an atomic round counter between calls instead of paying
/// `ProximaBackgroundPool`'s per-call `Mutex`+`Condvar` wake
/// (`prime/src/os/cohort.rs`'s own module doc: 2492.7 ns/round vs 19305.5
/// ns/round). Built once, on first use, sized to [`matmul_worker_count`] —
/// the same worker count [`quantized_matmul_workers`] already resolves for
/// the pool path, so a cohort round and a pool dispatch always claim the
/// same number of workers. `None` if the cohort fails to build (e.g. thread
/// spawn exhaustion); callers fall back to [`nest_pool`] in that case.
///
/// `PROXIMA_COHORT_SPIN_POLLS`, if set to a valid integer, overrides
/// [`COHORT_SPIN_POLLS`]; this exists to sweep the spin budget without a
/// rebuild. Read once, inside this same `get_or_init` that already builds
/// the cohort a single time for the process -- no separate `OnceLock` needed.
/// Default (unset) behavior is unchanged.
fn nest_cohort() -> Option<&'static MatmulCohort> {
    static COHORT: OnceLock<Option<MatmulCohort>> = OnceLock::new();
    COHORT
        .get_or_init(|| {
            let members = NonZeroUsize::new(matmul_worker_count())?;
            let spin_polls = std::env::var("PROXIMA_COHORT_SPIN_POLLS")
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(COHORT_SPIN_POLLS);
            let config = MatmulCohort::builder()
                .members(members)
                .spin_polls(spin_polls)
                .build();
            ThreadCohort::from_config(config).ok()
        })
        .as_ref()
}

fn live_count<B>(buffers: &[Option<B>]) -> usize {
    buffers.iter().filter(|entry| entry.is_some()).count()
}

/// Per-node retire sets over the *emitted* node sequence: `result[p]` is
/// every node whose last read is `resolved[p]`. This mirrors
/// [`live::annotate`](crate::live::annotate) in shape but is a different
/// computation over a different timeline — the resolved sequence, after
/// fusion has already decided which zips never materialize at all.
fn node_retirement(resolved: &[BoundOp], outputs: &[NodeId]) -> Vec<Vec<NodeId>> {
    let outputs: BTreeSet<NodeId> = outputs.iter().copied().collect();
    let mut last_use: BTreeMap<NodeId, usize> = BTreeMap::new();
    for (position, node) in resolved.iter().enumerate() {
        for (source, _, gather) in node.operands() {
            last_use.insert(*source, position);
            if let Some(gather_access) = gather {
                last_use.insert(gather_access.indices, position);
            }
        }
    }

    let mut retires = vec![Vec::new(); resolved.len()];
    for (node, position) in last_use {
        if !outputs.contains(&node) {
            retires[position].push(node);
        }
    }
    retires
}

fn block_node_ids(program: &[Op]) -> Vec<NodeId> {
    program
        .iter()
        .enumerate()
        .filter(|(_, expr)| matches!(expr, Op::Input { .. }))
        .map(|(position, _)| NodeId(position as u32))
        .collect()
}

// `shape::infer` (called before this) already rejects a scatter (a
// data-dependent fold output), any gather whose indices are not an integer
// dtype, and any gathered dim past 2^24 (an f32 index cannot represent a
// larger extent's values exactly), so the only restriction left to enforce
// here is that every OTHER node — every node not itself a gather's
// `indices` — is f32: every buffer slot derefs to `[f32]` regardless of
// which `B: Deref<Target = [f32]>` backs it (owned `Vec<f32>` or borrowed
// `Cow<[f32]>`), indices included (an index value is an exact integer
// carried as f32, per the module doc), so a gather's `indices` node is the
// one deliberate exception to the f32 rule rather than a second buffer kind.
//
// The real boundary this enforces is narrower than "f32-only" now reads:
// this function only gates `evaluate`/`evaluate_parallel`, the SIMD-tuned
// pipeline whose buffers, width-tiling, and dot-fold kernels are `Vec<f32>`
// throughout `cpu.rs` — regeneralizing *that* pipeline to every width is a
// materially bigger change than fits alongside adding one. A non-float32,
// non-gather-index, reduce/scan/gather-free program is not unsupported by
// this crate any more; it runs through [`evaluate_typed`] instead, which
// this function does not gate.
//
// `quantized_weights` is the second, narrower exception this now tolerates:
// a node tagged as a `Q4_K`-packed weight buffer, permitted ONLY when it is
// used exclusively as one operand of a `Reduce` whose body multiplies it
// against another operand and folds with `Add` — a matmul shape, checked by
// [`is_quantized_matmul_operand`] below, the same structural shape
// `neon_tile_plan`/`width_tile_plan` already recognize for the dense f32
// tile. `evaluate`/`evaluate_parallel` still pass an empty set (this is
// additive, not a behavior change for either), because their own `blocks:
// &[&[f32]]` parameter is itself f32-only — accepting the node here proves
// only that the *shape* of a quantized-weight matmul type-checks, not that
// this pipeline can execute one yet. [`matmul_q4k_f32`] is the dedicated,
// separately-tested execution path for that shape today; wiring a quantized
// buffer through `evaluate`'s own `blocks` array and `run_reduce`'s NEON
// tile is the remaining integration work this does not yet do.
fn reject_non_float32(program: &[Op], quantized_weights: &BTreeSet<NodeId>) -> Result<(), TensorError> {
    let index_nodes = index_node_ids(program);
    let referenced_nodes = referenced_node_ids(program);
    for (position, expr) in program.iter().enumerate() {
        let node = NodeId(position as u32);
        let is_quantized_weight = quantized_weights.contains(&node) && is_quantized_matmul_operand(program, node);
        // an `Op::Input` `bind::BoundOpBuilder::push` never materializes into
        // a `BoundOp` (see that match arm's own `Op::Input { .. } => {}`) —
        // it is a pure buffer handle, read directly by whichever node
        // references it, never itself run through `run_node_into`. So an
        // `Input` nothing in `program` references (an ONNX initializer for a
        // shape/index tensor no lowered op still reads, e.g.) can never
        // reach this f32-only interpreter's kernels regardless of its dtype
        // — unlike every other `Op` variant, which `push`/`finish` always
        // materialize into `resolved` and the evaluator's node loop always
        // runs, dead code or not (see `BoundOpBuilder::finish`'s own doc:
        // "either a requested output or dead code, and either way it
        // materializes"). A *referenced* non-float32 `Input` still feeds a
        // node that IS unconditionally evaluated, so it stays rejected.
        let is_unreferenced_input = matches!(expr, Op::Input { .. }) && !referenced_nodes.contains(&node);
        if expr.dtype() != DType::Float32
            && !index_nodes.contains(&node)
            && !is_quantized_weight
            && !is_unreferenced_input
        {
            return Err(TensorError::NotLowerable {
                node,
                reason: "this pipeline's buffers and SIMD kernels are f32-only; route a \
                         non-float32 elementwise program through evaluate_typed instead",
            });
        }
    }
    Ok(())
}

/// [`reject_non_float32`]'s dead-leaf exemption is deliberately
/// output-independent — a node's own connectivity to the rest of `program`,
/// nothing about which nodes a given call happens to request — so its
/// result stays valid for [`evaluate_quantized_with_scratch`]'s
/// `validated_weight_nodes` cache across calls whose `outputs` differ, not
/// only calls whose `quantized_weights` differ. But a caller CAN request an
/// otherwise-dead non-`Float32` `Input` directly as an output (this f32-only
/// pipeline still cannot honor that: its own `blocks`/`named` parameters
/// carry no non-`Float32` view for `evaluate`/`evaluate_named` to hand back
/// out, and `evaluate_quantized`'s `QuantizedBlock` non-`Float32` variants
/// are reserved for the matmul-weight shape [`is_quantized_matmul_operand`]
/// recognizes, not a passthrough return value), so this cheap,
/// always-run-per-call, `O(outputs.len())`-plus-one-scan check closes that
/// gap without folding `outputs` into the cached structural pass above.
fn reject_non_float32_outputs(
    program: &[Op],
    quantized_weights: &BTreeSet<NodeId>,
    outputs: &[NodeId],
) -> Result<(), TensorError> {
    let index_nodes = index_node_ids(program);
    for &node in outputs {
        let Some(expr) = program.get(node.0 as usize) else {
            continue;
        };
        let is_quantized_weight = quantized_weights.contains(&node) && is_quantized_matmul_operand(program, node);
        if expr.dtype() != DType::Float32 && !index_nodes.contains(&node) && !is_quantized_weight {
            return Err(TensorError::NotLowerable {
                node,
                reason: "this pipeline's buffers and SIMD kernels are f32-only; route a \
                         non-float32 elementwise program through evaluate_typed instead",
            });
        }
    }
    Ok(())
}

/// Every node referenced anywhere in `program` as an `Elementwise` operand, a
/// `Reduce`'s own operand, or either map's computed `indices` — the
/// complement of the set an `Op::Input` must fall outside of for
/// [`reject_non_float32`]'s dead-leaf exemption: a node in this set feeds
/// some other node that [`bind::BoundOpBuilder`] always materializes into a
/// [`BoundOp`](crate::bind::BoundOp), so its dtype still matters even when
/// that consumer is itself unreachable from any requested output.
fn referenced_node_ids(program: &[Op]) -> BTreeSet<NodeId> {
    let mut nodes = BTreeSet::new();
    for expr in program {
        match expr {
            Op::Input { .. } | Op::Iota { .. } | Op::Constant { .. } => {}
            Op::Elementwise { operands, .. } => {
                for (operand, map) in operands {
                    nodes.insert(*operand);
                    push_indices_node(map, &mut nodes);
                }
            }
            Op::Reduce(fold) => {
                nodes.insert(fold.operand);
                push_indices_node(&fold.in_map, &mut nodes);
                push_indices_node(&fold.out_map, &mut nodes);
            }
        }
    }
    nodes
}

/// Whether `node` appears, anywhere in `program`, ONLY as one operand of a
/// `Multiply` [`Op::Elementwise`] — the OTHER operand `Float32` — that
/// itself feeds directly into a `Reduce` whose `body` is `Add`: the exact
/// "quantized weight x f32 activation" matmul shape [`reject_non_float32`]'s
/// quantized-weight exemption requires. A quantized node used any other way
/// (paired with a second non-float32 operand, a second elementwise op, a
/// scan, a reduce with a different combiner) does not qualify: the
/// exemption is for the one shape [`matmul_q4k_f32`] actually implements,
/// not a blanket "trust the caller's tag."
fn is_quantized_matmul_operand(program: &[Op], node: NodeId) -> bool {
    let mut used_as_matmul_operand = false;
    for (position, expr) in program.iter().enumerate() {
        match expr {
            Op::Elementwise { body, operands, .. } => {
                if !operands.iter().any(|(source, _)| *source == node) {
                    continue;
                }
                let other_operand_is_float32 = operands
                    .iter()
                    .any(|(source, _)| *source != node && program[source.0 as usize].dtype() == DType::Float32);
                if *body != ScalarOp::Multiply || operands.len() != 2 || !other_operand_is_float32 {
                    return false;
                }
                let elementwise_node = NodeId(position as u32);
                let feeds_matmul_reduce = program.iter().any(|other| {
                    matches!(other, Op::Reduce(fold) if fold.operand == elementwise_node && fold.body == ScalarOp::Add)
                });
                if !feeds_matmul_reduce {
                    return false;
                }
                used_as_matmul_operand = true;
            }
            Op::Reduce(fold) => {
                if fold.operand == node {
                    // reduced directly, not through a Multiply elementwise —
                    // not the matmul shape this exemption covers.
                    return false;
                }
            }
            Op::Input { .. } | Op::Iota { .. } | Op::Constant { .. } => {}
        }
    }
    used_as_matmul_operand
}

/// Every node referenced as a gather's `indices` anywhere in `program` —
/// the one class of non-float32 node [`reject_non_float32`] tolerates.
fn index_node_ids(program: &[Op]) -> BTreeSet<NodeId> {
    let mut nodes = BTreeSet::new();
    for expr in program {
        match expr {
            Op::Input { .. } | Op::Iota { .. } | Op::Constant { .. } => {}
            Op::Elementwise { operands, .. } => {
                for (_, map) in operands {
                    push_indices_node(map, &mut nodes);
                }
            }
            Op::Reduce(fold) => {
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

fn buffer_of<T, B: Deref<Target = [T]>>(buffers: &[Option<B>], node: NodeId) -> Result<&[T], TensorError> {
    buffers[node.0 as usize]
        .as_deref()
        .ok_or(TensorError::NotLowerable {
            node,
            reason: "operand buffer missing at evaluation time",
        })
}

fn element_count(shape: &[u64]) -> usize {
    shape.iter().product::<u64>() as usize
}

fn split_innermost(extents: &[u64]) -> (&[u64], usize) {
    match extents.split_last() {
        Some((last, rest)) => (rest, *last as usize),
        None => (extents, 1),
    }
}

/// Total element count of an odometer over `shape` — `0..odometer_len(shape)`
/// is the flat-index range [`unflatten_into`] walks.
fn odometer_len(shape: &[u64]) -> u64 {
    shape.iter().product()
}

/// Writes flat index `flat`'s mixed-radix coordinate into the caller's
/// reused `coordinate` buffer instead of allocating a fresh `Vec` per call —
/// this runs once per (leading, reduction) coordinate pair in [`run_reduce`],
/// up to ~1e6 times for a 1024^3 GEMM. The allocating former version
/// (`odometer`/`unflatten`, returning `impl Iterator<Item = Vec<u64>>`)
/// accounted for roughly half of the 2.1M allocations measured after ROW 2's
/// `running`/`gather_cursors` hoist — the other half was
/// [`merge_coordinates_into`]'s former per-call `Vec` (`proxima-tensor/docs/discipline.md` ROW 2b).
fn unflatten_into(mut flat: u64, shape: &[u64], coordinate: &mut [u64]) {
    for (dim, extent) in shape.iter().enumerate().rev() {
        coordinate[dim] = flat % extent;
        flat /= extent;
    }
}

/// Writes the union of a leading coordinate and a reduction coordinate into
/// the caller's reused `out` buffer, zeroing any dim neither side supplies
/// (there are none in practice — every dim is either leading or reduction —
/// but the zero-fill keeps the contract obvious without relying on that).
fn merge_coordinates_into(
    leading_dims: &[u16],
    leading_coordinate: &[u64],
    reduction_dims: &[u16],
    reduction_coordinate: &[u64],
    out: &mut [u64],
) {
    out.fill(0);
    for (dim, value) in leading_dims.iter().zip(leading_coordinate) {
        out[*dim as usize] = *value;
    }
    for (dim, value) in reduction_dims.iter().zip(reduction_coordinate) {
        out[*dim as usize] = *value;
    }
}

fn initial_value(init: ReduceInit) -> Option<f32> {
    match init {
        ReduceInit::Zero => Some(0.0),
        ReduceInit::One => Some(1.0),
        ReduceInit::NegativeInfinity => Some(f32::NEG_INFINITY),
        ReduceInit::PositiveInfinity => Some(f32::INFINITY),
        ReduceInit::FirstElement => None,
    }
}

// `evaluate`/`evaluate_parallel` both drive `run_node_into` directly (see
// `evaluate_pooled`'s doc for why), so this allocate-and-run wrapper only
// remains for tests that want a whole node's output as a `Vec` to compare
// against hand-run chunks.
#[cfg(test)]
fn run_node(resolved: &BoundOp, buffers: &[Option<Vec<f32>>]) -> Result<Vec<f32>, TensorError> {
    let mut output = vec![0.0f32; node_output_len(resolved)];
    run_node_into(resolved, buffers, None, None, &mut output)?;
    Ok(output)
}

/// Runs `resolved`, writing its output into a caller-provided slice instead
/// of allocating one. This is the primitive [`run_node`] (sequential) and
/// [`evaluate_parallel`] (one call per chunk, each writing a disjoint
/// sub-slice of the same parent buffer) both drive — the loop nests below
/// are written once, here.
fn run_node_into<B: Deref<Target = [f32]> + Sync>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    quantized_weights: Option<&BTreeMap<NodeId, QuantizedBlock>>,
    session: Option<&MatmulSession<'_>>,
    output: &mut [f32],
) -> Result<(), TensorError> {
    match &resolved.kind {
        BoundOpKind::Elementwise { .. } => {
            #[cfg(feature = "instrument")]
            instrument::record_op_kind(instrument::OpKind::Elementwise);
            run_elementwise_dispatch(resolved, buffers, session, output)
        }
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            out_scatter: Some(_),
            ..
        } => {
            #[cfg(feature = "instrument")]
            instrument::record_op_kind(instrument::OpKind::Reduce);
            run_reduce_scatter(resolved, buffers, output)
        }
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            out_scatter: None,
            ..
        } => {
            #[cfg(feature = "instrument")]
            instrument::record_op_kind(instrument::OpKind::Reduce);
            match quantized_weights {
                Some(quantized_weights) => {
                    run_reduce_with_quantized_weights(resolved, buffers, quantized_weights, session, output)
                }
                None => run_reduce(resolved, buffers, output),
            }
        }
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => {
            #[cfg(feature = "instrument")]
            instrument::record_op_kind(instrument::OpKind::Scan);
            run_scan(resolved, buffers, output)
        }
        BoundOpKind::Iota => run_iota(output),
        BoundOpKind::Constant { value } => run_constant(*value, output),
    }
}

/// [`BoundOpKind::Constant`]'s whole computation: every element is the same
/// literal. Even simpler than [`run_iota`] — no operand reads, no body, and
/// not even a dependence on position.
fn run_constant(value: f32, output: &mut [f32]) -> Result<(), TensorError> {
    output.fill(value);
    Ok(())
}

/// [`BoundOpKind::Iota`]'s whole computation: `output[i] = i`, exact in f32
/// up to `GATHER_EXTENT_EXACT_FLOAT_LIMIT` (`shape.rs`'s own doc) the same
/// way a gather index is — no operand reads, no per-step body, just the
/// position itself.
fn run_iota(output: &mut [f32]) -> Result<(), TensorError> {
    for (index, slot) in output.iter_mut().enumerate() {
        *slot = index as f32;
    }
    Ok(())
}

/// The output length [`run_node_into`] expects from `resolved`: the full
/// iteration space for an elementwise node or a `Keep::Scan` scan (neither
/// drops any dim), or the reduced (leading dims x width) shape for a
/// `Keep::Reduce` fold.
fn node_output_len(resolved: &BoundOp) -> usize {
    match &resolved.kind {
        // `output_axes` excludes the scattered axis entirely (its position
        // is data-dependent, never a pure projection — see
        // `bind::pure_projection_axes`), so the ordinary leading/width
        // product below would silently drop that axis from the length.
        // `out_scatter.extent` is the one place that axis's static width
        // survives past shape inference (`bind::build_reduce_op`'s doc).
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            output_axes,
            out_scatter: Some(target),
            ..
        } => {
            let non_scattered_product: u64 = output_axes
                .iter()
                .map(|dim| resolved.extents[*dim as usize])
                .product();
            non_scattered_product as usize * target.extent as usize
        }
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            output_axes,
            out_scatter: None,
            ..
        } => {
            let (leading_output_axes, last_output_dim) = output_axes_split(output_axes.as_slice());
            let leading_product: u64 = leading_output_axes
                .iter()
                .map(|dim| resolved.extents[*dim as usize])
                .product();
            let width = last_output_dim.map_or(1, |dim| resolved.extents[dim as usize] as usize);
            leading_product as usize * width
        }
        _ => element_count(&resolved.extents),
    }
}

/// The execution stage: a [`Pipe`] over a batch of ready [`BoundOp`]
/// nodes — exactly the batch one upstream [`crate::bind::BoundOpBuilder`]
/// push readies.
///
/// `In = Vec<BoundOp>`, `Out = ()`: the buffer table this stage writes into
/// is interior state, borrowed from the caller at construction rather than
/// allocated here or threaded through `In`/`Out` — `Out = ()` is literal,
/// not a value smuggled through mutation and reported via a nonempty `Out`.
/// That borrow is what lets a caller run this against its own
/// no-alloc scratch. `RefCell` is the same interior-mutability idiom
/// [`shape::ShapeTable`] and [`crate::bind::BoundOpBuilder`] already use for
/// their own per-record state, applied to the buffer table that already
/// existed here rather than to a wrapper minted to host the impl. Taking
/// the batch as `In` (rather than one `BoundOp` at a time) is what makes
/// `Second::In = First::Out` hold against `BoundOpBuilder::Out =
/// Vec<BoundOp>`, so this stage composes into the full
/// `shapes.and_then(builder).and_then(interpreter)` chain with no adapter.
pub struct Interpreter<'buffers, B: Deref<Target = [f32]> + Sync> {
    buffers: RefCell<&'buffers mut [Option<B>]>,
}

impl<'buffers, B: Deref<Target = [f32]> + Sync + From<Vec<f32>>> Interpreter<'buffers, B> {
    /// `buffers` is caller-owned scratch, one slot per program node — the
    /// same shape `prepare` already builds locally for [`evaluate`].
    /// `Interpreter` never allocates it, resizes it, or takes ownership of
    /// it. Generic over `B` (matching `run_node_into`'s bound) so the same
    /// interpreter drives both [`evaluate`]'s `Cow`-backed table (no
    /// redundant copy of an `Op::Input` block) and a plain `Vec<f32>` table.
    #[must_use]
    pub fn new(buffers: &'buffers mut [Option<B>]) -> Self {
        Self {
            buffers: RefCell::new(buffers),
        }
    }

    /// Reads a node's computed data back out of the buffer table. Separate
    /// from `Pipe::Out` on purpose: what this stage produced for the algebra
    /// (nothing — `Out = ()`) and what a caller later wants to read out of
    /// its own state are different questions, and this crate's algebra only
    /// answers the first one through `Pipe::call`.
    #[must_use]
    pub fn get(&self, node: NodeId) -> Option<Vec<f32>> {
        self.buffers.borrow()[node.0 as usize].as_deref().map(<[f32]>::to_vec)
    }

    /// The actual fold: written once, against a borrowed `&[BoundOp]` rather
    /// than an owned `Vec`, so a caller driving one node at a time (like
    /// [`evaluate`]) can pass a one-element slice (`core::slice::from_ref`)
    /// with no batch allocation at all — `Pipe::call` below is the only
    /// other caller, and it just hands this its owned `Vec` by reference
    /// (`&ready`), so the streaming chain's batch contract (`In =
    /// Vec<BoundOp>`, required for `Second::In = First::Out` against
    /// [`crate::bind::BoundOpBuilder`]'s `Out`) and `evaluate`'s no-alloc
    /// per-node path both bottom out in this one written-once loop.
    fn fold(&self, ready: &[BoundOp]) -> Result<(), TensorError> {
        for resolved in ready {
            let mut output = vec![0.0f32; node_output_len(resolved)];
            {
                let buffers = self.buffers.borrow();
                run_node_into(resolved, *buffers, None, None, &mut output)?;
                #[cfg(feature = "instrument")]
                record_bound_op_operand_access(resolved, *buffers);
            }
            let mut buffers = self.buffers.borrow_mut();
            (*buffers)[resolved.node.0 as usize] = Some(B::from(output));
        }
        Ok(())
    }
}

impl<B: Deref<Target = [f32]> + Sync + From<Vec<f32>>> Pipe for Interpreter<'_, B> {
    type In = ReadyBatch;
    type Out = ();
    type Err = TensorError;

    /// Folds a batch of ready nodes into the buffer table, in order — the
    /// same fold the buffer table already does one write at a time, just
    /// driven for every element of `ready` inside one call instead of one
    /// call per element. An empty `ready` is a no-op, not a special case.
    ///
    /// `In` stays `ReadyBatch` (owned) because that is what
    /// `BoundOpBuilder::Out` already is — changing it would break the
    /// `Second::In = First::Out` composition law the streaming chain relies
    /// on — but the owned batch is only ever borrowed from here down; see
    /// `Interpreter::fold`.
    fn call(&self, ready: ReadyBatch) -> impl Future<Output = Result<(), TensorError>> {
        async move { self.fold(&ready) }
    }
}

/// A fold's `output_axes`, split into the leading (outer) dims and the
/// innermost one (if any) — shared by [`run_reduce`] and [`node_output_len`]
/// so the two agree on shape by construction.
fn output_axes_split(output_axes: &[u16]) -> (&[u16], Option<u16>) {
    match output_axes.split_last() {
        Some((last, leading)) => (leading, Some(*last)),
        None => (&[], None),
    }
}

/// Closed-form reads/distinct-touched accounting for one operand across a
/// bound op's own iteration space, given that operand's per-axis strides
/// (`Layout::stride` against `resolved.extents`, the same rank both share
/// throughout this module). An axis this operand broadcasts over
/// (`stride == 0`) is still visited by every position along it — the loop
/// nest re-reads the same element — so it contributes its full extent to
/// `reads` but only `1` (not `extent`) to `distinct`, since the same offset
/// resolves every time. A non-broadcast axis contributes its full extent to
/// both. `distinct` is therefore exact for an ordinary (gather-free)
/// operand, since a real tensor `Layout`'s strides never alias two distinct
/// coordinates onto the same offset outside of an explicit `stride == 0`
/// broadcast.
///
/// `O(rank)`, never `O(elements)` — every caller invokes this once per
/// bound-op evaluation (`cpu::record_bound_op_operand_access`), against the
/// UNSPLIT op's own extents, so a `BoundOp::split` chunk fan-out under
/// `evaluate_parallel` never re-derives this per chunk (that would double
/// count a broadcast operand's footprint once per chunk instead of once for
/// the whole node — see `instrument.rs`'s module comment on
/// `OperandAccess`).
#[cfg(any(feature = "instrument", test))]
fn operand_access_footprint(extents: &[u64], strides: &[i64]) -> (u64, u64) {
    let mut reads: u64 = 1;
    let mut distinct: u64 = 1;
    for (&extent, &stride) in extents.iter().zip(strides) {
        reads *= extent;
        if stride != 0 {
            distinct *= extent;
        }
    }
    (reads, distinct)
}

/// Attributes one bound op's operand reads to their own source `NodeId`s,
/// once the op has finished running. Called from `evaluate_pooled`,
/// `evaluate_node_parallel`, and `Interpreter::fold` — the three places that
/// hold the UNSPLIT `BoundOp` right after `run_node_into`/`run_chunks_threaded`
/// returns — never from inside a per-chunk or per-element loop.
///
/// A gathered operand (`Some(lookup)`) cannot get its distinct-element count
/// from `operand_access_footprint`: which table row a gather touches is a
/// runtime index value, not a function of loop coordinate alone. Its real
/// count instead comes from the row-level witness `fill_gather_cursors`
/// already builds during execution (`instrument::commit_gather_operand_access`
/// reads it back), scaled by the table's own row width
/// (`Lookup::element_stride`) to report elements rather than rows.
#[cfg(feature = "instrument")]
fn record_bound_op_operand_access<B: Deref<Target = [f32]>>(resolved: &BoundOp, buffers: &[Option<B>]) {
    for (source, layout, gather) in resolved.operands() {
        let strides: Vec<i64> = (0..resolved.extents.len() as u16).map(|axis| layout.stride(axis)).collect();
        let (reads, distinct) = operand_access_footprint(&resolved.extents, &strides);
        let total_elements = buffer_of(buffers, *source).map(<[f32]>::len).unwrap_or(0) as u64;
        match gather {
            Some(lookup) => {
                let row_width = lookup.element_stride.unsigned_abs();
                instrument::commit_gather_operand_access(*source, reads, row_width, total_elements);
            }
            None => {
                instrument::record_operand_access(*source, reads, distinct, total_elements);
            }
        }
    }
}

fn operand_buffers<'a, B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &'a [Option<B>],
) -> Result<Vec<&'a [f32]>, TensorError> {
    resolved
        .operands()
        .iter()
        .map(|(source, _, _)| buffer_of(buffers, *source))
        .collect()
}

/// A raw gather-index buffer element: whatever width the source buffer
/// stores a fetched row index at, reduced to the one signed 64-bit value
/// [`GatherCursor::fetch_and_advance`] bounds-checks and scales by
/// `element_stride`. `f32` is the f32 pipeline's own index width (every
/// buffer that pipeline carries, including `indices`, is f32 — see
/// [`crate::map::IndexMap`]'s own doc); `i64` is the typed evaluator's
/// canonical index width ([`canonical_index_buffers`]'s own doc). No other
/// width ever backs a [`GatherCursor`] directly.
trait GatherIndexElement: Copy {
    fn as_gather_index(self) -> i64;
}

impl GatherIndexElement for f32 {
    fn as_gather_index(self) -> i64 {
        self as i64
    }
}

impl GatherIndexElement for i64 {
    fn as_gather_index(self) -> i64 {
        self
    }
}

/// Per-step gather state for one operand: an incrementally-advanced offset
/// into the `indices` buffer (mirroring how a normal operand's own running
/// offset advances by a precomputed stride each step), plus what to do with
/// a fetched value once read. `E` is the index buffer's own element width
/// ([`GatherIndexElement`]); it defaults to `f32`, the f32 pipeline's only
/// width, so every existing call site naming `GatherCursor<'a>` keeps
/// compiling unchanged. [`fill_gather_cursors_typed`] is the only source of
/// `GatherCursor<'a, i64>`.
struct GatherCursor<'a, E = f32> {
    buffer: &'a [E],
    offset: i64,
    stride: i64,
    element_stride: i64,
    extent: u64,
}

impl<E: GatherIndexElement> GatherCursor<'_, E> {
    /// Reads the next index, advances the cursor, and returns the offset
    /// contribution that index adds to the operand's own running offset — a
    /// real error, not a clamp or a wraparound, when the fetched index falls
    /// outside the gathered dim's extent.
    fn fetch_and_advance(&mut self, node: NodeId) -> Result<i64, TensorError> {
        let raw = self.buffer[self.offset as usize];
        self.offset += self.stride;
        let index = raw.as_gather_index();
        if index < 0 || index as u64 >= self.extent {
            return Err(TensorError::GatherIndexOutOfRange {
                node,
                index,
                extent: self.extent,
            });
        }
        Ok(index * self.element_stride)
    }
}

/// Fills one [`GatherCursor`] per operand that gathers (`None` for the
/// rest) into a caller-owned buffer, each initialized at `coordinate` and
/// advancing by `stride_dim`'s stride per step — `stride_dim` is `None`
/// where there is no per-step dimension at all (a scalar reduction's single
/// accumulator).
///
/// Writes into `cursors` in place rather than returning a fresh `Vec`: this
/// runs once per reduction step (up to ~1e6 times for a 1024^3 GEMM), and
/// `cursors` is the caller's reused scratch buffer, sized once to operand
/// count outside the hot loop (`proxima-tensor/docs/discipline.md` ROW 2).
///
/// Under the `instrument` feature, this is also the row-level witness point
/// for [`instrument::record_gather_row`]: seeding a cursor already reads
/// this row's index value's OFFSET into the indices tensor
/// (`gather_access.index_layout.offset_of(coordinate)`) as part of normal,
/// already-paid-for addressing, so reading the raw index value itself here
/// too — once per row, never per element `fetch_and_advance` steps through —
/// piggybacks on that instead of adding a second traversal.
fn fill_gather_cursors<'a, B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &'a [Option<B>],
    coordinate: &[u64],
    stride_dim: Option<u16>,
    cursors: &mut [Option<GatherCursor<'a>>],
) -> Result<(), TensorError> {
    for (slot, (source, _, gather)) in cursors.iter_mut().zip(resolved.operands()) {
        #[cfg(not(feature = "instrument"))]
        let _ = source;
        *slot = gather
            .as_ref()
            .map(|gather_access| {
                let buffer = buffer_of(buffers, gather_access.indices)?;
                let offset = gather_access.index_layout.offset_of(coordinate);
                #[cfg(feature = "instrument")]
                {
                    let row_index = buffer[offset as usize] as i64;
                    if row_index >= 0 {
                        instrument::record_gather_row(*source, row_index as u64);
                    }
                }
                Ok(GatherCursor {
                    buffer,
                    offset,
                    stride: stride_dim.map_or(0, |dim| gather_access.index_layout.stride(dim)),
                    element_stride: gather_access.element_stride,
                    extent: gather_access.extent,
                })
            })
            .transpose()?;
    }
    Ok(())
}

/// [`fill_gather_cursors`]'s typed counterpart: sources each cursor's raw
/// index values from `index_buffers` — the canonical `i64` table
/// [`canonical_index_buffers`] builds once at execution start — rather than
/// the operand buffer table `fill_gather_cursors` reads from. The typed
/// evaluator's index nodes carry their own integer dtype, never the
/// program's compute dtype, so they cannot live in the same `buffers: &[T]`
/// table a gathered operand's own values do.
fn fill_gather_cursors_typed<'a>(
    resolved: &BoundOp,
    index_buffers: &'a [Option<Vec<i64>>],
    coordinate: &[u64],
    stride_dim: Option<u16>,
    cursors: &mut [Option<GatherCursor<'a, i64>>],
) -> Result<(), TensorError> {
    for (slot, (source, _, gather)) in cursors.iter_mut().zip(resolved.operands()) {
        #[cfg(not(feature = "instrument"))]
        let _ = source;
        *slot = gather
            .as_ref()
            .map(|gather_access| {
                let buffer = index_buffers[gather_access.indices.0 as usize]
                    .as_deref()
                    .ok_or(TensorError::NotLowerable {
                        node: gather_access.indices,
                        reason: "gather index buffer missing at evaluation time",
                    })?;
                let offset = gather_access.index_layout.offset_of(coordinate);
                #[cfg(feature = "instrument")]
                {
                    let row_index = buffer[offset as usize];
                    if row_index >= 0 {
                        instrument::record_gather_row(*source, row_index as u64);
                    }
                }
                Ok(GatherCursor {
                    buffer,
                    offset,
                    stride: stride_dim.map_or(0, |dim| gather_access.index_layout.stride(dim)),
                    element_stride: gather_access.element_stride,
                    extent: gather_access.extent,
                })
            })
            .transpose()?;
    }
    Ok(())
}

/// Recomputes each operand's running byte offset for a fresh coordinate,
/// writing into the caller's reused `running` buffer instead of collecting a
/// new `Vec` — the per-position counterpart of [`fill_gather_cursors`].
fn fill_running_offsets(resolved: &BoundOp, coordinate: &[u64], running: &mut [i64]) {
    for (slot, (_, view, _)) in running.iter_mut().zip(resolved.operands()) {
        *slot = view.offset_of(coordinate);
    }
}

/// Dispatches one elementwise node across the cohort when a `session` is
/// open, the node clears [`PARALLEL_THRESHOLD`], and there is more than one
/// outer position to spread across workers. [`BoundOp::split`] only chunks
/// the outermost *axis* (`split_axis`'s own doc), which for this program's
/// shapes is the sequence dim — 6 for the forward pass this was measured
/// against, smaller than `workers` on any real box, so every elementwise
/// node fell through to the sequential fallback and the split never fired
/// (`DIAG elementwise_split_none`, measured against every node above
/// threshold). The fix chunks the same *flattened* outer-position space
/// [`run_elementwise`]'s own loop already walks instead: each outer
/// position writes an independent, contiguous `inner_len`-wide row of
/// `output` (`fill_running_offsets`/`fill_gather_cursors` reseed fresh from
/// that position's own coordinate every iteration, so no state carries
/// across positions — see their own docs), so a contiguous range of
/// positions is exactly as independent as [`matmul_rows_threaded`]'s row
/// ranges, without needing [`BoundOp::split`]'s single-axis rebase at all.
/// Reuses [`row_chunk_count`] (rows = outer positions, contraction width =
/// `inner_len`) for the same oversubscription/macs-floor policy
/// [`matmul_rows_threaded`] already tunes, rather than a second policy for
/// this axis. Falls straight through to [`run_elementwise`] whenever any
/// gate fails: no session, too few elements, or too few outer positions to
/// clear even a one-chunk-per-worker split.
fn run_elementwise_dispatch<B: Deref<Target = [f32]> + Sync>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    session: Option<&MatmulSession<'_>>,
    output: &mut [f32],
) -> Result<(), TensorError> {
    let Some(session) = session else {
        return run_elementwise(resolved, buffers, output);
    };
    if element_count(&resolved.extents) < PARALLEL_THRESHOLD {
        return run_elementwise(resolved, buffers, output);
    }
    let workers = matmul_worker_count();
    let (outer_extents, inner_len) = split_innermost(&resolved.extents);
    let outer_len = odometer_len(outer_extents) as usize;
    if workers <= 1 || outer_len < 2 || inner_len == 0 {
        return run_elementwise(resolved, buffers, output);
    }

    // `row_chunk_count`'s own `MIN_MACS_PER_CHUNK` floor is tuned to a
    // matmul dot-product's per-mac cost (`matmul_rows_threaded`'s doc); an
    // elementwise op's per-element cost is a small constant number of
    // scalar ops, not a mac, so reusing that floor left almost every node
    // (median 12288 elements, `MIN_MACS_PER_CHUNK` 500,000) computing
    // exactly one chunk — no different from the sequential fallback
    // (measured: `elementwise` term unchanged at ~25.6 ms with that floor
    // applied here). `PARALLEL_THRESHOLD` above already gates whether
    // splitting is worth a round-open at all, so once past it this reuses
    // `evaluate_node_parallel`'s own chunk-count policy instead
    // (`workers * OVERSUBSCRIBE`, `evaluate_node_parallel`'s own doc),
    // capped at one row per chunk so `chunk_len` never rounds to zero.
    //
    // A per-node OR per-chunk element-count floor was tried here (three
    // variants: a flat total-element cutoff, the same cutoff scoped to
    // `Unary`/`Binary` bodies only, and a `MIN_MACS_PER_CHUNK`-shaped
    // per-chunk floor) to stop the real openchat-3.5 decode loop's small
    // elementwise nodes (4096/14336 total elements) from opening a
    // `CohortSession::run` round for work this crate's own measured
    // 0.38ns/element rate finishes before the round would even open. Every
    // variant either left decode's round count unchanged (its actual
    // splitting nodes turned out to be `Generic`-shaped, not `Unary`/
    // `Binary`, so a shape-scoped floor missed them) or measurably regressed
    // the real forward pass's (`runs_one_real_forward_pass_and_greedy_picks_
    // a_real_token`) own comparably-sized `Generic` nodes, which DO benefit
    // from splitting (`DIAG evaluate_quantized node_kind=elementwise
    // total_ms`: 20.327ms baseline vs 25.570ms flat-floor vs 23.851ms
    // per-chunk-floor, all worse). A per-chunk-sized `Generic` node in the
    // forward pass and a whole small `Generic` node in decode land on
    // IDENTICAL element counts (14336 either way), so no floor keyed on
    // element count alone can tell them apart — the real discriminator
    // (round-trip cost vs achievable parallelism for that specific node's
    // shape) was not isolated within this investigation's budget. Left
    // unchanged rather than shipped with a measured prefill regression;
    // see this landing's own log for the full measurement trail.
    let chunk_count = (workers * OVERSUBSCRIBE).min(outer_len);
    let chunk_len = outer_len.div_ceil(chunk_count);
    let mut chunk_ranges = Vec::with_capacity(chunk_count);
    let mut remaining = &mut *output;
    let mut outer_start = 0usize;
    while !remaining.is_empty() {
        let take = chunk_len.min(remaining.len() / inner_len);
        let (slice, rest) = remaining.split_at_mut(take * inner_len);
        remaining = rest;
        chunk_ranges.push((outer_start, slice.as_mut_ptr() as usize, slice.len()));
        outer_start += take;
    }
    if chunk_ranges.len() < 2 {
        return run_elementwise(resolved, buffers, output);
    }

    let round = ElementwiseRowRound {
        resolved,
        buffers,
        inner_len,
        chunk_ranges: &chunk_ranges,
    };
    #[cfg(feature = "instrument")]
    {
        counter!(instrument::ELEMENTWISE_COHORT_ROUNDS, 1);
    }
    let report = session.run(&round);
    if let Some(error) = report.first_error {
        return Err(error);
    }
    if report.abandoned > 0 {
        return Err(TensorError::ThreadedChunkFailed {
            chunk: report.first_abandoned.map_or(0, |chunk| chunk.0 + 1),
            reason: alloc::string::String::from(
                "cohort member panicked while running this elementwise chunk",
            ),
        });
    }
    Ok(())
}

/// [`run_elementwise_dispatch`]'s cohort dispatch shape — the same
/// relationship [`RowRound`] has to [`matmul_rows_threaded`], one round
/// over `(outer_start, chunk_address, len)` ranges of the flattened outer
/// position space, run through [`CohortSession::run`]. `resolved`/`buffers`
/// stay ordinary borrows for the round's whole lifetime, the same argument
/// [`RowRound`]'s own doc makes.
struct ElementwiseRowRound<'round, B> {
    resolved: &'round BoundOp,
    buffers: &'round [Option<B>],
    inner_len: usize,
    chunk_ranges: &'round [(usize, usize, usize)],
}

impl<B> CohortRound<TensorError> for ElementwiseRowRound<'_, B>
where
    B: Deref<Target = [f32]> + Sync,
{
    fn chunks(&self) -> usize {
        self.chunk_ranges.len()
    }

    fn run_chunk(&self, chunk: ChunkIndex) -> Result<(), TensorError> {
        let (outer_start, slice_address, slice_len) = self.chunk_ranges[chunk.0];
        // SAFETY: unique to this chunk by construction (`split_at_mut` in
        // `run_elementwise_dispatch` before the round starts); the parent
        // `output` outlives every reconstructed slice because
        // `CohortSession::run` does not return until every member has
        // reported done.
        let chunk_output =
            unsafe { core::slice::from_raw_parts_mut(slice_address as *mut f32, slice_len) };
        let outer_end = outer_start + slice_len / self.inner_len;
        run_elementwise_range(self.resolved, self.buffers, outer_start, outer_end, chunk_output)
    }
}

/// One cohort round holding an ORDERED sequence of parallel stages, so a
/// whole run of graph nodes pays ONE round open/close instead of one each.
///
/// This exists because the per-node round is the measured binding constraint
/// on decode (`proxima-tensor/docs/discipline.md` ROW 68): a decode
/// elementwise node holds ~33 us of work and opening a round costs ~25 us, so
/// splitting one node across the cohort measured 2x WORSE, while leaving it
/// serial measured no thread scaling at all (`reduce_f32_dense` 3.859 ms at 1
/// worker, 3.952 at 8). Neither end of that trade is acceptable; the way out
/// is to stop paying per node. ggml's CPU backend runs its whole graph on a
/// persistent team with a cheap barrier between nodes, which is the shape
/// this reproduces.
///
/// No new `prime` primitive was needed. [`CohortRound`] hands out a flat
/// chunk space off a monotonic `fetch_add` cursor (`prime/src/os/cohort.rs`
/// `cursor`), so chunk `i` is always CLAIMED before chunk `i + 1`. Laying
/// stages out consecutively in that space therefore means every chunk of
/// stage `s - 1` has an owner before any chunk of stage `s` is claimed, and a
/// member that reaches stage `s` can simply wait on a per-stage counter.
///
/// Deadlock-free by induction on stage index: a member waiting at stage `s`
/// waits only on stage `s - 1` chunks, each of which has an owner that is
/// either running it or waiting on a strictly earlier stage; the
/// lowest-stage active member is therefore always running, never waiting.
/// The counter is bumped even when a chunk FAILS, so one stage's error can
/// never hang the members behind it — the error still propagates through
/// `CohortSession`'s own report.
///
/// Requires the default all-chunks completion policy: a `FanInCompletion`
/// that stops dispatch early would strand a stage's chunks unclaimed and
/// hang the stage behind it.
///
/// Gated to `cfg(test)` until its consumer lands: the semantics below are
/// the load-bearing, easy-to-get-wrong half (claim order, barrier,
/// deadlock-freedom, error publication), so they are proven FIRST and
/// separately from the graph-walking change that will use them. Wiring the
/// executor onto this is what removes the gate.
///
/// `stage_offsets` (length `stage_count + 1`, strictly increasing,
/// `stage_offsets[0] == 0`) replaces a single uniform `chunks_per_stage`:
/// stage `s` owns chunks `stage_offsets[s]..stage_offsets[s + 1]`, so a
/// matmul-reduce stage (many row-chunks, real cross-worker parallelism) and
/// an elementwise/reduce stage (one chunk, `run_node_into`'s own serial
/// body) can share the SAME round without the narrower stage paying for
/// chunks it never needed — a uniform `chunks_per_stage` would have forced
/// every stage to either match the matmul stage's width (every one-node
/// stage now split into phantom sub-chunks with nothing to parallelize) or
/// the elementwise stage's width (every matmul stage capped at one chunk,
/// serializing the dominant-cost computation onto a single worker). Both
/// are exactly the failure `docs/discipline.md` ROW 96 measured when it
/// tried the uniform-width version of this idea against non-matmul nodes
/// only.
#[cfg(any(test, feature = "cohort-staged-graph"))]
struct StagedRound<'round, Run> {
    stage_offsets: &'round [usize],
    /// completed-chunk count for each stage, indexed by stage.
    completed: &'round [AtomicUsize],
    run_stage_chunk: Run,
}

#[cfg(any(test, feature = "cohort-staged-graph"))]
impl<Run> CohortRound<TensorError> for StagedRound<'_, Run>
where
    Run: Fn(usize, usize) -> Result<(), TensorError> + Sync,
{
    fn chunks(&self) -> usize {
        self.stage_offsets.last().copied().unwrap_or(0)
    }

    fn run_chunk(&self, chunk: ChunkIndex) -> Result<(), TensorError> {
        // `stage_offsets` is strictly increasing starting at 0, so the
        // number of offsets `<= chunk.0` is always `stage + 1` for the
        // owning stage `s` -- `partition_point`'s own contract (first index
        // whose predicate is false) hands that back directly.
        let stage = self.stage_offsets.partition_point(|&offset| offset <= chunk.0) - 1;
        let within_stage = chunk.0 - self.stage_offsets[stage];
        if let Some(previous) = stage.checked_sub(1) {
            let previous_len = self.stage_offsets[previous + 1] - self.stage_offsets[previous];
            while self.completed[previous].load(Ordering::Acquire) < previous_len {
                core::hint::spin_loop();
            }
        }
        let outcome = (self.run_stage_chunk)(stage, within_stage);
        self.completed[stage].fetch_add(1, Ordering::Release);
        outcome
    }
}

fn run_elementwise<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    let (outer_extents, _) = split_innermost(&resolved.extents);
    let outer_len = odometer_len(outer_extents) as usize;
    run_elementwise_range(resolved, buffers, 0, outer_len, output)
}

/// [`run_elementwise`]'s whole computation, restricted to
/// `[outer_start, outer_end)` of the flattened outer-position space —
/// `run_elementwise` itself is the `0..outer_len` case. Every outer
/// position is independent (see [`run_elementwise_dispatch`]'s doc), so
/// narrowing the range changes nothing about what any position computes,
/// only how many of them this call covers; `output` is indexed relative to
/// `outer_start` (`out_base` below), matching the disjoint sub-slice a
/// caller like [`ElementwiseRowRound`] hands in.
fn run_elementwise_range<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    outer_start: usize,
    outer_end: usize,
    output: &mut [f32],
) -> Result<(), TensorError> {
    #[cfg(feature = "instrument")]
    let diag_setup_started = instrument::read_ticks();
    let (outer_extents, inner_len) = split_innermost(&resolved.extents);
    let innermost_dim = outer_extents.len() as u16;
    let raw = operand_buffers(resolved, buffers)?;
    let body = resolved.element_body();
    let shape = body_shape(body);
    let mut operand_values = vec![0.0f32; raw.len()];
    // loop-invariant: the innermost dim's stride never depends on the outer
    // coordinate, so it is computed once for the whole node, not once per
    // outer position (`proxima-tensor/docs/discipline.md` ROW 2).
    let strides: Vec<i64> = resolved
        .operands()
        .iter()
        .map(|(_, view, _)| view.stride(innermost_dim))
        .collect();
    let mut running: Vec<i64> = vec![0; raw.len()];
    let mut gather_cursors: Vec<Option<GatherCursor>> = (0..raw.len()).map(|_| None).collect();
    let mut outer_coordinate = vec![0u64; outer_extents.len()];

    // The dim immediately outside the vectorized `inner_len` width — Conv's
    // own `window_materialize` multiply (`proxima-onnx/src/lower.rs`) shapes
    // its output `[n,c,oh,ow,kh,kw]`, so `kw` alone lands in `inner_len`
    // (3 elements) and `kh` (also 3) is this dim, otherwise walked one
    // `unflatten_into`+`fill_running_offsets` call at a time same as every
    // other outer dim (`docs/discipline.md` residual-profile session,
    // 2026-08-30: measured 12.9 ns/element on this op, ~34x this crate's own
    // 0.38 ns/element monomorphic figure, entirely fixed per-call overhead
    // amortized over only 3 elements — MAC_OPS/OUTPUT_WRITES showed no slow
    // gather path engaged at all). `block_strides` stays empty (never
    // indexed) whenever `block_extent <= 1`, the common case for every
    // rank-1-outer-extents or `kh == 1` shape.
    let block_dim = if outer_extents.is_empty() { None } else { Some((outer_extents.len() - 1) as u16) };
    let block_extent = outer_extents.last().copied().unwrap_or(1);
    let block_strides: Vec<i64> = if block_extent > 1 {
        resolved.operands().iter().map(|(_, view, _)| block_dim.map_or(0, |dim| view.stride(dim))).collect()
    } else {
        Vec::new()
    };

    // `Unary`/`Binary` share `run_reduce`'s own gate (ROW 3); `Generic`
    // (a fused multi-step chain) gets its own, narrower gate that only
    // `run_elementwise` acts on — every operand the body shape reads is
    // gather-free and affine with a width-dim stride of 0 or 1
    // (`proxima-tensor/docs/discipline.md` ROW 5).
    // `FusedAdamUpdate` (`docs/discipline.md` ROW 179) gets its own gate,
    // narrower than `Generic`'s: `fused_adam_update_is_affine_fast_path`
    // requires exact unit/zero strides, not `Generic`'s wider "any
    // non-negative constant stride" admission, because the dedicated kernel
    // slices `m`/`v`/`param` directly rather than walking `OperandSpan`.
    let fast_path = match shape {
        BodyShape::Generic(generic_body) => generic_body_is_affine_fast_path(resolved, generic_body, &strides),
        BodyShape::FusedAdamUpdate(roles, _) => fused_adam_update_is_affine_fast_path(resolved, roles, &strides),
        _ => body_shape_is_affine_fast_path(resolved, &shape, &strides),
    };
    // rung 2 (`docs/discipline.md` ROW 153's own charter): when the block
    // above is engaged AND the body is a bare identity copy (`window_materialize`'s
    // post-ROW-147 collapsed form), every row the block loop would otherwise
    // walk through `elementwise_width_fast`'s per-row shape/op dispatch is a
    // plain contiguous `inner_len`-wide read at a fixed row stride — computed
    // once per call, same discipline as `fast_path`/`block_strides` above.
    let window_copy_operand = window_copy_operand(&shape, fast_path, block_extent, &strides);
    // Row-flattening (`docs/discipline.md` ROW 178): `window_copy_operand`
    // already collapses ITS narrower shape (a bare identity copy, one block
    // dim) to one memcpy-shaped call per block; this is the same collapse
    // for the GENERAL case — every operand's address across the WHOLE
    // `outer_extents` odometer (every outer dim, not just the last one)
    // composing as a single contiguous stride-`strides[operand]` span,
    // reusing [`axes_flat_chain`] (ROW 148's own reduce-side helper,
    // de-gated from `aarch64`-only below since this call site is
    // architecture-generic) with `unit = strides[operand] * inner_len`: the
    // address one outer step away must land exactly one row past where the
    // current row's own width span ends, at the SAME per-element stride. A
    // stride-0 (broadcast) operand collapses `unit` to 0, which
    // `axes_flat_chain` already treats as "every nonzero-extent axis in the
    // chain must ALSO be stride 0" — a genuinely global scalar (Adam's
    // `beta1`/`beta2`/`eps`/`lr` constants, stride 0 in every dim) passes
    // this for free; a per-row-only broadcast (stride 0 in the width dim,
    // nonzero across an outer dim) correctly FAILS it, since that address
    // is a step function of the flattened index, not affine in it, and
    // cannot be expressed as one [`elementwise_width_fast`] call. Deferred
    // to `window_copy_operand`'s own narrower, already-proven-optimal path
    // (ROW 153/154) when both apply.
    let full_range_flat = fast_path && window_copy_operand.is_none() && {
        let outer_axes: Vec<u16> = (0..innermost_dim).collect();
        elementwise_rows_are_flat(resolved, &outer_axes, &strides, inner_len)
    };
    #[cfg(feature = "instrument")]
    {
        counter!(
            instrument::ELEMENTWISE_SETUP_TICKS,
            instrument::elapsed_ticks(diag_setup_started)
        );
    }
    #[cfg(feature = "instrument")]
    let diag_step_values_started = instrument::read_ticks();
    // `elementwise_width_generic` is the only reader of `step_values`
    // (`elementwise_width_fast`'s own doc: "`Unary`/`Binary` ignore it"), and
    // the slow scalar path's `eval_body_shape` matches the same way — a
    // `Unary`/`Binary` shape never touches it either. Sizing this for every
    // shape at `body.steps.len() * inner_len` paid a real
    // `inner_len`-element (4096/14336 `f32`) heap allocation per node even
    // when nothing ever read it back; only `Generic` needs the fused
    // per-step row table at all. `full_range_flat` widens the row this call
    // covers to the WHOLE `[outer_start, outer_end)` span, so the table must
    // be sized against that wider width, not `inner_len` alone — otherwise
    // `elementwise_width_generic`'s own internal `GENERIC_WIDTH_TILE`
    // chunking (ROW 175) would index past a table sized for one narrow row.
    let flat_width = (outer_end - outer_start) * inner_len;
    let mut step_values = match shape {
        BodyShape::Generic(_) => {
            let effective_width = if full_range_flat { flat_width } else { inner_len };
            vec![0.0f32; body.steps.len() * if fast_path { effective_width.min(GENERIC_WIDTH_TILE) } else { 1 }]
        }
        // The dedicated kernel (`elementwise_width_fused_adam_update`) never
        // reads `step_values` -- only the slow per-element gather fallback
        // (`eval_body_shape` -> `apply_body`, reached when `fast_path` is
        // false) needs one scalar row per step, the same shape `Generic`'s
        // own `else { 1 }` branch already sizes for.
        BodyShape::FusedAdamUpdate(..) => vec![0.0f32; if fast_path { 0 } else { body.steps.len() }],
        BodyShape::Unary(..) | BodyShape::Binary(..) => Vec::new(),
    };
    #[cfg(feature = "instrument")]
    {
        counter!(
            instrument::ELEMENTWISE_STEP_VALUES_TICKS,
            instrument::elapsed_ticks(diag_step_values_started)
        );
        counter!(instrument::ELEMENTWISE_RANGE_CALLS, 1);
    }
    #[cfg(feature = "instrument")]
    let diag_loop_started = instrument::read_ticks();
    #[cfg(feature = "instrument")]
    let mut counters = KernelCounters::default();
    #[cfg(feature = "instrument")]
    let path = if fast_path { Path::WidthFast } else { Path::Generic };

    let mut outer_position = outer_start;
    while outer_position < outer_end {
        unflatten_into(outer_position as u64, outer_extents, &mut outer_coordinate);
        fill_running_offsets(resolved, &outer_coordinate, &mut running);

        // The whole `[outer_start, outer_end)` range collapsed to one flat
        // span (`full_range_flat`, computed once above): a SINGLE
        // `elementwise_width_fast` call over `flat_width` elements replaces
        // what would otherwise be `outer_end - outer_start` separate
        // per-row calls — node 132's own `[784,128]` shape turns 784 calls
        // into 1 (`docs/discipline.md` ROW 178). `running` is already
        // correct for `outer_position == outer_start` (just computed
        // above); every subsequent row's own address is exactly
        // `strides[operand]` past the previous element by construction of
        // the flatten precondition, so `elementwise_width_fast` walking the
        // combined width at that SAME stride reads every row without a
        // second odometer step.
        if full_range_flat {
            elementwise_width_fast(&shape, &raw, &running, &strides, output, &mut step_values);
            #[cfg(feature = "instrument")]
            {
                let elements = output.len() as u64;
                counter!(instrument::ELEMENTWISE_FLAT_RANGE_HITS, 1);
                counter!(instrument::ELEMENTWISE_FLAT_RANGE_ROWS, (outer_end - outer_start) as u64);
                counters.leading_iters += (outer_end - outer_start) as u64;
                counters.kernel_calls += 1;
                counters.output_writes += elements;
                for &stride in &strides {
                    counters.operand_loads += if stride == 0 { 1 } else { elements };
                }
            }
            outer_position = outer_end;
            continue;
        }

        // Blocked sweep of `block_dim` (see its own doc above): only when the
        // fast width path is already engaged (so every operand here is
        // gather-free — `Layout::offset_of` is exactly linear in the
        // coordinate, `bind.rs`, so `offset_of(coord + h*e_dim) ==
        // offset_of(coord) + h*stride(dim)` is exact, not approximate), this
        // position starts a fresh sweep (`block_dim`'s own coordinate is 0),
        // and a full `block_extent`-long run still fits before `outer_end`
        // (a parallel chunk boundary mid-sweep falls through to the
        // per-position path below, same as an unaligned `outer_start`).
        if fast_path
            && block_extent > 1
            && outer_coordinate.last() == Some(&0)
            && outer_position + block_extent as usize <= outer_end
        {
            let out_base = (outer_position - outer_start) * inner_len;
            let out_slice = &mut output[out_base..out_base + block_extent as usize * inner_len];
            if let Some(operand) = window_copy_operand {
                let operand = operand as usize;
                window_copy_block(raw[operand], running[operand], block_strides[operand], block_extent, inner_len, out_slice);
                #[cfg(feature = "instrument")]
                {
                    let elements = block_extent * inner_len as u64;
                    counters.leading_iters += block_extent;
                    counters.kernel_calls += block_extent;
                    counters.output_writes += elements;
                    counters.operand_loads += elements;
                }
            } else {
                for step in 0..block_extent {
                    let step_base = step as usize * inner_len;
                    elementwise_width_fast(
                        &shape,
                        &raw,
                        &running,
                        &strides,
                        &mut out_slice[step_base..step_base + inner_len],
                        &mut step_values,
                    );
                    #[cfg(feature = "instrument")]
                    {
                        counters.leading_iters += 1;
                        counters.kernel_calls += 1;
                        counters.output_writes += inner_len as u64;
                        for &stride in &strides {
                            counters.operand_loads += if stride == 0 { 1 } else { inner_len as u64 };
                        }
                    }
                    if step + 1 < block_extent {
                        for (slot, block_stride) in running.iter_mut().zip(&block_strides) {
                            *slot += block_stride;
                        }
                    }
                }
            }
            outer_position += block_extent as usize;
            continue;
        }

        let out_base = (outer_position - outer_start) * inner_len;
        #[cfg(feature = "instrument")]
        {
            counters.leading_iters += 1;
        }

        if fast_path {
            let out_slice = &mut output[out_base..out_base + inner_len];
            elementwise_width_fast(&shape, &raw, &running, &strides, out_slice, &mut step_values);
            #[cfg(feature = "instrument")]
            {
                counters.kernel_calls += 1;
                counters.output_writes += inner_len as u64;
                for &stride in &strides {
                    counters.operand_loads += if stride == 0 { 1 } else { inner_len as u64 };
                }
            }
            outer_position += 1;
            continue;
        }

        fill_gather_cursors(
            resolved,
            buffers,
            &outer_coordinate,
            Some(innermost_dim),
            &mut gather_cursors,
        )?;

        for step in 0..inner_len {
            for (index, data) in raw.iter().enumerate() {
                let mut offset = running[index];
                if let Some(cursor) = gather_cursors[index].as_mut() {
                    offset += cursor.fetch_and_advance(resolved.node)?;
                }
                operand_values[index] = data[offset as usize];
                running[index] += strides[index];
            }
            output[out_base + step] = eval_body_shape(&shape, &operand_values, &mut step_values);
            #[cfg(feature = "instrument")]
            {
                counters.kernel_calls += 1;
                counters.output_writes += 1;
                counters.operand_loads += raw.len() as u64;
            }
        }
        outer_position += 1;
    }
    #[cfg(feature = "instrument")]
    {
        let diag_loop_ticks = instrument::elapsed_ticks(diag_loop_started);
        counter!(instrument::ELEMENTWISE_LOOP_TICKS, diag_loop_ticks);
        let distinct_operand_elements: u64 = raw.iter().map(|buffer| buffer.len() as u64).sum();
        // achieved-ns/element split by `BodyShape` (nsper task, 2026-08-21):
        // `Unary`/`Binary` is the monomorphic kernel this crate's own
        // 0.38ns/element figure (`cpu.rs:2159`) was measured against;
        // `Generic` is the fused multi-step body. Both axes read
        // `counters.output_writes`, the exact element count this call wrote
        // (identical whether `fast_path` did or didn't fire -- see the loop
        // above), never re-derived from extents.
        match shape {
            BodyShape::Generic(_) => {
                counter!(instrument::ELEMENTWISE_LOOP_TICKS_GENERIC, diag_loop_ticks);
                counter!(instrument::ELEMENTWISE_ELEMENTS_GENERIC, counters.output_writes);
                if fast_path {
                    counter!(instrument::ELEMENTWISE_LOOP_TICKS_GENERIC_FAST, diag_loop_ticks);
                    counter!(instrument::ELEMENTWISE_ELEMENTS_GENERIC_FAST, counters.output_writes);
                } else {
                    counter!(instrument::ELEMENTWISE_LOOP_TICKS_GENERIC_SLOW, diag_loop_ticks);
                    counter!(instrument::ELEMENTWISE_ELEMENTS_GENERIC_SLOW, counters.output_writes);
                }
            }
            BodyShape::FusedAdamUpdate(..) => {
                counter!(instrument::ELEMENTWISE_LOOP_TICKS_GENERIC, diag_loop_ticks);
                counter!(instrument::ELEMENTWISE_ELEMENTS_GENERIC, counters.output_writes);
                if fast_path {
                    counter!(instrument::ELEMENTWISE_LOOP_TICKS_GENERIC_FAST, diag_loop_ticks);
                    counter!(instrument::ELEMENTWISE_ELEMENTS_GENERIC_FAST, counters.output_writes);
                    counter!(instrument::ELEMENTWISE_LOOP_TICKS_FUSED_ADAM, diag_loop_ticks);
                    counter!(instrument::ELEMENTWISE_ELEMENTS_FUSED_ADAM, counters.output_writes);
                    counter!(instrument::ELEMENTWISE_FUSED_ADAM_HITS, 1);
                } else {
                    counter!(instrument::ELEMENTWISE_LOOP_TICKS_GENERIC_SLOW, diag_loop_ticks);
                    counter!(instrument::ELEMENTWISE_ELEMENTS_GENERIC_SLOW, counters.output_writes);
                }
            }
            BodyShape::Unary(..) | BodyShape::Binary(..) => {
                counter!(instrument::ELEMENTWISE_LOOP_TICKS_MONOMORPHIC, diag_loop_ticks);
                counter!(instrument::ELEMENTWISE_ELEMENTS_MONOMORPHIC, counters.output_writes);
                if window_copy_operand.is_some() {
                    // rung 2 (ROW 153/154): same per-call constant `fast_path`
                    // already splits on, one level narrower — this call's
                    // block-aligned rows took the specialized row-segment copy,
                    // not `elementwise_width_fast`'s per-row dispatch.
                    counter!(instrument::ELEMENTWISE_LOOP_TICKS_WINDOW_COPY, diag_loop_ticks);
                    counter!(instrument::ELEMENTWISE_ELEMENTS_WINDOW_COPY, counters.output_writes);
                }
                if fast_path {
                    counter!(instrument::ELEMENTWISE_LOOP_TICKS_MONOMORPHIC_FAST, diag_loop_ticks);
                    counter!(instrument::ELEMENTWISE_ELEMENTS_MONOMORPHIC_FAST, counters.output_writes);
                } else {
                    counter!(instrument::ELEMENTWISE_LOOP_TICKS_MONOMORPHIC_SLOW, diag_loop_ticks);
                    counter!(instrument::ELEMENTWISE_ELEMENTS_MONOMORPHIC_SLOW, counters.output_writes);
                }
            }
        }
        instrument::record_elementwise_call_size(counters.output_writes);
        counters.commit(path, distinct_operand_elements);
    }
    Ok(())
}

/// Which of `resolved`'s operands, if any, is a `Q4_K`-packed weight named
/// in `quantized_weights` — [`run_reduce`]'s own gate for routing to
/// [`matmul_q4k_f32`] instead of the f32 tile/generic paths below. Only a
/// `Keep::Reduce` fold can match: `quantized_weights` only ever names a node
/// [`reject_non_float32`] already proved (via [`is_quantized_matmul_operand`])
/// feeds exactly one such fold, so this need not re-check the shape, only
/// find which physical operand it is.
fn quantized_operand(resolved: &BoundOp, quantized_weights: &BTreeMap<NodeId, QuantizedBlock>) -> Option<NodeId> {
    if !matches!(resolved.kind, BoundOpKind::Reduce { keep: Keep::Reduce, .. }) {
        return None;
    }
    resolved.operands().iter().map(|(node, _, _)| *node).find(|node| quantized_weights.contains_key(node))
}

/// proxima-debugger diagnostic (`evaluate_quantized`'s per-node-kind timing
/// table): buckets a bound op the same way [`run_node_into`]'s own match
/// dispatches it, splitting `Keep::Reduce` into the quantized-matmul arm
/// ([`run_reduce_quantized`], parallel-dispatched) versus the dense f32 arm
/// ([`run_reduce`], serial) via the same [`quantized_operand`] check
/// [`run_reduce_with_quantized_weights`] itself gates on.
#[cfg(feature = "instrument")]
fn diag_node_kind_label(resolved: &BoundOp, quantized_weights: &BTreeMap<NodeId, QuantizedBlock>) -> &'static str {
    match &resolved.kind {
        BoundOpKind::Elementwise { .. } => "elementwise",
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => {
            if quantized_operand(resolved, quantized_weights).is_some() {
                "reduce_matmul_quantized"
            } else {
                "reduce_f32_dense"
            }
        }
        BoundOpKind::Reduce { keep: Keep::Scan, .. } => "scan",
        BoundOpKind::Iota => "iota",
        BoundOpKind::Constant { .. } => "constant",
    }
}

#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
use crate::sized::MIN_TRANSPOSE_ELEMENTS_FOR_DISPATCH;

/// The `wide` (`[row][position]`) -> `output` (`[position][row]`) transpose
/// copy-back the `Q4_K`/`Q5_K`/`Q6_K` wide-fold arms of
/// [`run_reduce_quantized`] pay -- dispatched across the cohort when a
/// `session` is open and `rows * leading_total` clears
/// [`MIN_TRANSPOSE_ELEMENTS_FOR_DISPATCH`]. Splits on `position`, the same
/// outer axis [`run_elementwise_dispatch`] splits on: each position range
/// writes a contiguous, disjoint `rows`-wide slice of `output` (safe
/// [`slice::split_at_mut`], no raw pointer needed for the write side),
/// reading a strided range of `wide` (a shared `&[f32]`, never mutated).
/// Falls straight through to the plain serial loop whenever any gate fails:
/// no session, too few elements, or fewer than two position chunks to split
/// into.
#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
fn transpose_wide_to_output(
    wide: &[f32],
    rows: usize,
    leading_total: usize,
    session: Option<&MatmulSession<'_>>,
    output: &mut [f32],
) -> Result<(), TensorError> {
    let serial = |output: &mut [f32]| {
        for row in 0..rows {
            for position in 0..leading_total {
                output[position * rows + row] = wide[row * leading_total + position];
            }
        }
    };
    let Some(session) = session else {
        serial(output);
        return Ok(());
    };
    if rows.saturating_mul(leading_total) < MIN_TRANSPOSE_ELEMENTS_FOR_DISPATCH {
        serial(output);
        return Ok(());
    }
    let workers = matmul_worker_count();
    if workers <= 1 || leading_total < 2 {
        serial(output);
        return Ok(());
    }
    let chunk_count = (workers * OVERSUBSCRIBE).min(leading_total);
    let chunk_len = leading_total.div_ceil(chunk_count);
    let mut chunk_ranges = Vec::with_capacity(chunk_count);
    let mut remaining = &mut *output;
    let mut position_start = 0usize;
    while !remaining.is_empty() {
        let take = chunk_len.min(remaining.len() / rows);
        let (slice, rest) = remaining.split_at_mut(take * rows);
        remaining = rest;
        chunk_ranges.push((position_start, slice.as_mut_ptr() as usize, slice.len()));
        position_start += take;
    }
    if chunk_ranges.len() < 2 {
        serial(output);
        return Ok(());
    }
    let round = TransposeRound {
        wide,
        rows,
        leading_total,
        chunk_ranges: &chunk_ranges,
    };
    let report = session.run(&round);
    if report.abandoned > 0 {
        return Err(TensorError::ThreadedChunkFailed {
            chunk: report.first_abandoned.map_or(0, |chunk| chunk.0 + 1),
            reason: alloc::string::String::from("cohort member panicked while running this transpose chunk"),
        });
    }
    Ok(())
}

/// [`transpose_wide_to_output`]'s cohort dispatch shape: one round over
/// `(position_start, out_ptr, out_len)` ranges of `output`'s position axis,
/// run through [`CohortSession::run`]. No error path -- pure data movement,
/// nothing here can fail the way a matmul row's dot product can.
#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
struct TransposeRound<'round> {
    wide: &'round [f32],
    rows: usize,
    leading_total: usize,
    chunk_ranges: &'round [(usize, usize, usize)],
}

#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
impl CohortRound<TensorError> for TransposeRound<'_> {
    fn chunks(&self) -> usize {
        self.chunk_ranges.len()
    }

    fn run_chunk(&self, chunk: ChunkIndex) -> Result<(), TensorError> {
        let (position_start, out_ptr, out_len) = self.chunk_ranges[chunk.0];
        // SAFETY: unique to this chunk by construction (`split_at_mut` in
        // `transpose_wide_to_output` before the round starts); the parent
        // `output` outlives every reconstructed slice because
        // `CohortSession::run` does not return until every member has
        // reported done.
        let chunk_output = unsafe { core::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };
        let position_count = out_len / self.rows;
        for local_position in 0..position_count {
            let position = position_start + local_position;
            for row in 0..self.rows {
                chunk_output[local_position * self.rows + row] = self.wide[row * self.leading_total + position];
            }
        }
        Ok(())
    }
}

/// Below this many consecutive [`is_staged_batch_eligible`] nodes,
/// [`run_staged_batch`] is not worth calling: a run of one node has nothing
/// to amortize a round-open against, so [`evaluate_quantized_with_scratch`]
/// falls through to the plain per-node call for it, exactly as
/// `cohort-staged-graph` off always does. Threaded through the build-time
/// sizing config (principle 12) as of ROW 98 -- see
/// `crate::sized::STAGED_BATCH_MIN_LEN`'s own doc for the measurement
/// record.
#[cfg(feature = "cohort-staged-graph")]
use crate::sized::STAGED_BATCH_MIN_LEN;

/// One codec's row-dot kernel: `(weight_row, activation_q8k) -> dot`. Every
/// `Q4_K`/`Q5_K`/`Q6_K` kernel (`dot_q4k_q8k`/`dot_q5k_q8k`/`dot_q6k_q8k`)
/// shares this exact signature, which is what makes [`dot_fn_for`]'s return
/// type — and therefore [`MatmulStagePlan::dot_fn`] — the same concrete type
/// regardless of which codec a given matmul node uses.
#[cfg(feature = "cohort-staged-graph")]
type MatmulRowDotFn = fn(&[u8], &[u8]) -> Result<f32, TensorError>;

/// Selects the row-dot kernel for one quantized-matmul-reduce node's own
/// codec as a plain `fn` pointer rather than naming a distinct codec
/// function at a distinct closure-literal source location -- what makes a
/// `Q4_K` node's own row-chunk work and a `Q5_K` or `Q6_K` node's the exact
/// same concrete Rust type: the ONE thing `docs/discipline.md` ROW 96 named
/// as the actual blocker to folding matmul stages into a shared round
/// ("each codec's closure is a distinct type") does not apply once the
/// codec choice is a captured VALUE ([`MatmulStagePlan::dot_fn`]) instead of
/// a name baked into the closure body. `None` for `Q8_0` (no shared
/// `Q8_K`-activation wide-fold path exists for it — see
/// [`run_reduce_quantized`]'s own per-position loop, which dequantizes
/// `Q8_0` row-by-row instead) and for `Float32` (not a quantized weight at
/// all); either leaves that node on the existing unbatched path via
/// [`is_staged_batch_eligible`].
#[cfg(feature = "cohort-staged-graph")]
fn dot_fn_for(weight_block: QuantizedBlock<'_>) -> Option<MatmulRowDotFn> {
    match weight_block {
        #[cfg(feature = "q4k-int8-dot")]
        QuantizedBlock::Q4K(_) => Some(dot_q4k_q8k),
        #[cfg(feature = "q5k-int8-dot")]
        QuantizedBlock::Q5K(_) => Some(dot_q5k_q8k),
        #[cfg(feature = "q6k-int8-dot")]
        QuantizedBlock::Q6K(_) => Some(dot_q6k_q8k),
        _ => None,
    }
}

/// Whether `resolved` belongs in a [`run_staged_batch`] run: ONLY a
/// quantized-weight matmul fold whose own codec has a [`dot_fn_for`] entry
/// (`Q4_K`/`Q5_K`/`Q6_K` built with that codec's own `-int8-dot` feature).
/// Every other kind — elementwise, dense f32 reduce, scan, iota, constant —
/// is deliberately NOT eligible here, unlike `docs/discipline.md` ROW 96's
/// own version of this function, which admitted them. Measured, not
/// assumed (ROW 97): a mixed run (ROW 96's non-matmul kinds ALSO admitted,
/// alongside this session's matmul fold) measured `rounds` RISE 6972 ->
/// 10355 (+48.5%) — the non-matmul kinds still open a round wherever grouped
/// that they opened zero of before, exactly ROW 96's own finding, and it
/// swamps the matmul-fold's own savings. Restricting eligibility to matmul
/// alone measured `rounds` FALL 6972 -> 4412 (-36.7%), because a run this
/// narrow only ever replaces rounds that already existed (one per matmul
/// node) with fewer, larger ones — it can never ADD a round where none
/// existed, which is the one property ROW 96's broader version lacked.
#[cfg(feature = "cohort-staged-graph")]
fn is_staged_batch_eligible(resolved: &BoundOp, quantized_weights: &BTreeMap<NodeId, QuantizedBlock>) -> bool {
    match quantized_operand(resolved, quantized_weights) {
        None => false,
        Some(weight_node) => quantized_weights.get(&weight_node).is_some_and(|block| dot_fn_for(*block).is_some()),
    }
}

/// The end (exclusive) of the maximal run of [`is_staged_batch_eligible`]
/// nodes starting at `start` — [`evaluate_quantized_with_scratch`]'s own
/// walk over `resolved` reuses this rather than recomputing a dependency
/// DAG: `resolved` is already topologically ordered (`bind::bind`'s own
/// doc), so a contiguous run of eligible positions is, by construction,
/// exactly as independent-of-everything-outside-the-run as any single node
/// already is of everything before it. [`run_staged_batch`] commits every
/// stage's output into the real `buffers` table only after the whole round
/// returns, so a node added to the run must read every operand from OUTSIDE
/// the run — a node at an earlier position that is ALSO in this run has not
/// been written into `buffers` yet when an in-run consumer's stage would
/// need to read it (`docs/discipline.md` ROW 96 caught exactly this via
/// `spec::tests::a_cached_decode_step_matches_the_uncached_forward_pass_exactly`).
/// `resolved`'s topological order means any such dependency is necessarily
/// on an earlier position, so checking positions `start..end` — never
/// anything after `end` — is exhaustive.
#[cfg(feature = "cohort-staged-graph")]
fn staged_batch_run_end(resolved: &[BoundOp], start: usize, quantized_weights: &BTreeMap<NodeId, QuantizedBlock>) -> usize {
    let mut end = start;
    while end < resolved.len() && is_staged_batch_eligible(&resolved[end], quantized_weights) {
        let reads_from_this_run = resolved[end]
            .operands()
            .iter()
            .any(|(operand, _, _)| resolved[start..end].iter().any(|produced| produced.node == *operand));
        if reads_from_this_run {
            break;
        }
        end += 1;
    }
    end
}

/// One quantized-matmul-reduce node's own row-parallel work, prepared
/// BEFORE [`run_staged_batch`] opens its round: activation quantization
/// (`Some(session)` — safe here because this runs strictly before
/// `session.run(&round)` is ever called, so there is no in-flight round for
/// a second `session.run` to collide with; see [`build_matmul_stage_plan`]'s
/// own call site) and the [`row_chunk_count`]-many `(row_start, ptr, len)`
/// ranges into `wide`, split via `split_at_mut` exactly the way
/// [`matmul_rows_threaded`] itself splits its own `output` — same
/// single-writer argument, one stage's own chunks instead of one call's.
#[cfg(feature = "cohort-staged-graph")]
struct MatmulStagePlan<'plan> {
    weights: &'plan [u8],
    // `Arc<[u8]>`, not `Vec<u8>`: `docs/discipline.md` ROW 140's own
    // measured hypothesis check (`instrument::quantize_activation_call_stats`,
    // 225 calls / 129 distinct activation nodes on the real checkpoint) --
    // `attn_q`/`attn_k`/`attn_v` (and `ffn_gate`/`ffn_up`) are consecutive
    // [`is_staged_batch_eligible`] nodes reading the SAME `activation_node`,
    // so [`build_matmul_stage_plan`]'s own `staged_quantize_cache` quantizes
    // it once and every later sibling in the same run clones the `Arc`
    // (one atomic refcount bump) instead of re-running
    // [`quantize_row_q8k_dispatch`]'s per-superblock scale search. A `Vec<u8>`
    // field would still force a byte-for-byte memcpy per cache hit; `Arc<[u8]>`
    // makes the reuse itself allocation-free.
    activation_q8k: Arc<[u8]>,
    row_bytes: usize,
    q8k_row_bytes: usize,
    /// `leading_total` — [`matmul_rows_threaded`]'s own `width` parameter:
    /// how many positions are folded into one row's own dot.
    width: usize,
    rows: usize,
    dot_fn: MatmulRowDotFn,
    /// row-major (`[row][position]`) scratch this stage's chunks write
    /// into; transposed to the node's real position-major output by
    /// [`run_staged_batch`] after the round returns, the same transpose
    /// [`run_reduce_quantized`]'s own wide-fold arms pay unbatched.
    wide: Vec<f32>,
    chunk_ranges: Vec<(usize, usize, usize)>,
}

#[cfg(feature = "cohort-staged-graph")]
impl MatmulStagePlan<'_> {
    fn run_chunk(&self, within_stage: usize) -> Result<(), TensorError> {
        let (row_start, address, length) = self.chunk_ranges[within_stage];
        // SAFETY: identical single-writer argument to `RowRound::run_chunk`
        // — carved via `split_at_mut` before the round opens (see this
        // plan's own construction in `build_matmul_stage_plan`), one chunk
        // per pointer, `self.wide` never pushed/resized again until the
        // round returns and `run_staged_batch` reads it back through `&self`.
        let chunk_output = unsafe { core::slice::from_raw_parts_mut(address as *mut f32, length) };
        for (offset, slot) in chunk_output.chunks_exact_mut(self.width).enumerate() {
            let row = row_start + offset;
            let start = row * self.row_bytes;
            let weight_row = &self.weights[start..start + self.row_bytes];
            for (position, output_slot) in slot.iter_mut().enumerate() {
                let q8k_start = position * self.q8k_row_bytes;
                *output_slot = (self.dot_fn)(weight_row, &self.activation_q8k[q8k_start..q8k_start + self.q8k_row_bytes])?;
            }
        }
        Ok(())
    }
}

/// Builds `resolved`'s own [`MatmulStagePlan`], or `None` when its shape
/// does not clear [`quantized_matmul_workers`]'s threshold — too little
/// work to parallelize, the same threshold [`run_reduce_quantized`] itself
/// checks before calling [`matmul_rows_threaded`]. `None` here means
/// [`run_staged_batch`] falls back to running this ONE node through the
/// plain [`run_node_into`] path (a single one-chunk stage, `session: None`),
/// identical to what every other batch-eligible node kind already does.
///
/// Shape derivation duplicates (rather than extracts from)
/// [`run_reduce_quantized`]'s own `rows`/`k`/`leading_total` derivation —
/// deliberately, so this feature's own additions never touch that already
/// bit-exact-verified function's body; the copy stays narrow and close to
/// it so a future drift shows up in a diff, not behind an extra call this
/// session did not have budget to verify against every one of
/// `run_reduce_quantized`'s own edge cases (the `Q8_0` growable-cache
/// `output.is_empty()` early return among them — moot here since `Q8_0` is
/// never [`dot_fn_for`]-eligible, but a reason to keep the two copies
/// separate rather than partially shared).
#[cfg(feature = "cohort-staged-graph")]
fn build_matmul_stage_plan<'weights>(
    resolved: &BoundOp,
    buffers: &[Option<Cow<'_, [f32]>>],
    weight_block: QuantizedBlock<'weights>,
    weight_node: NodeId,
    session: &MatmulSession<'_>,
    quantize_cache: &mut [Option<Arc<[u8]>>],
) -> Result<Option<MatmulStagePlan<'weights>>, TensorError> {
    let activation_node = resolved
        .operands()
        .iter()
        .map(|(node, _, _)| *node)
        .find(|node| *node != weight_node)
        .ok_or(TensorError::NotLowerable {
            node: resolved.node,
            reason: "quantized matmul reduce has no activation operand",
        })?;
    let activation = buffers[activation_node.0 as usize].as_deref().ok_or(TensorError::NotLowerable {
        node: activation_node,
        reason: "quantized matmul activation operand has no bound buffer",
    })?;
    let BoundOpKind::Reduce { output_axes, .. } = &resolved.kind else {
        unreachable!("build_matmul_stage_plan is only called for a Keep::Reduce fold")
    };
    let axis_shape = resolve_reduce_axis_shape(resolved, output_axes.as_slice());
    let contraction_width: u64 = axis_shape.reduction_extents.iter().product();
    let shape_error = || TensorError::NotLowerable {
        node: resolved.node,
        reason: "quantized matmul batch shape does not evenly divide by its packed weight rows",
    };
    let shared_axis_error = || TensorError::NotLowerable {
        node: resolved.node,
        reason: "quantized matmul activation varies along an output axis its packed weight also \
                 varies along -- not a flat weight matmul this interpreter can express",
    };
    let k = usize::try_from(contraction_width).map_err(|_| shape_error())?;

    let weight_layout = resolved
        .operands()
        .iter()
        .find(|(node, _, _)| *node == weight_node)
        .map(|(_, layout, _)| layout)
        .ok_or_else(shape_error)?;
    let activation_layout = resolved
        .operands()
        .iter()
        .find(|(node, _, _)| *node == activation_node)
        .map(|(_, layout, _)| layout)
        .ok_or_else(shape_error)?;

    // A gathered weight operand (`moe_block.toml`'s `expert_w`) picks a
    // different expert slab per position -- this staged precompute assumes
    // one flat weight matrix shared across the whole round
    // (`run_stage_chunk`'s own dispatch). Rather than duplicate
    // `run_reduce_quantized`'s gather resolution a second time in this
    // already-deliberately-duplicated shape derivation, bail to `None`: the
    // same fallback this function already takes for `dot_fn_for`/
    // `quantized_matmul_workers` ineligibility, which routes back through
    // `run_node_into`'s plain (gather-aware) path.
    if resolved
        .operands()
        .iter()
        .any(|(node, _, gather)| *node == weight_node && gather.is_some())
    {
        return Ok(None);
    }

    let mut rows_total: u64 = 1;
    let mut leading_total_u64: u64 = 1;
    for axis in output_axes.as_slice() {
        let extent = resolved.extents[*axis as usize];
        if weight_layout.stride(*axis) != 0 {
            if activation_layout.stride(*axis) != 0 {
                return Err(shared_axis_error());
            }
            rows_total *= extent;
        } else {
            leading_total_u64 *= extent;
        }
    }
    let rows = usize::try_from(rows_total).map_err(|_| shape_error())?;
    let leading_total = usize::try_from(leading_total_u64).map_err(|_| shape_error())?;

    let (weights, block_bytes, block_elements): (&[u8], usize, usize) = match weight_block {
        QuantizedBlock::Float32(_) => return Err(shape_error()),
        QuantizedBlock::Q4K(bytes) => (bytes, Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS),
        QuantizedBlock::Q5K(bytes) => (bytes, Q5K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS),
        QuantizedBlock::Q6K(bytes) => (bytes, Q6K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS),
        QuantizedBlock::Q8_0(bytes) => (bytes, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS),
        QuantizedBlock::Q4_0(bytes) => (bytes, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS),
        QuantizedBlock::Float16(bytes) => (bytes, HALF_PRECISION_ELEMENT_BYTES, 1),
        QuantizedBlock::BFloat16(bytes) => (bytes, HALF_PRECISION_ELEMENT_BYTES, 1),
    };
    if k == 0 || rows == 0 || !weights.len().is_multiple_of(block_bytes) {
        return Err(shape_error());
    }
    let total_weight_elements = (weights.len() / block_bytes) * block_elements;
    if total_weight_elements != rows * k {
        return Err(shape_error());
    }
    if activation.len() != leading_total * k {
        return Err(shape_error());
    }
    if !k.is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(shape_error());
    }

    let Some(dot_fn) = dot_fn_for(weight_block) else {
        return Ok(None);
    };
    let Some(workers) = quantized_matmul_workers(rows, activation.len()) else {
        return Ok(None);
    };

    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    let q8k_row_bytes = (k / Q4K_BLOCK_ELEMENTS) * Q8K_BLOCK_BYTES;
    // ROW 140's own fix: the SAME `activation_node` feeds every one of
    // `attn_q`/`attn_k`/`attn_v` (and `ffn_gate`/`ffn_up`) -- 129 distinct
    // activation nodes measured against 225 quantize calls on the real
    // checkpoint before this cache existed
    // (`instrument::quantize_activation_call_stats`, ROW 140's own doc).
    // `quantize_cache` is scoped to ONE `evaluate_quantized_with_scratch`
    // call (one decode/prefill step) -- see that function's own
    // `staged_quantize_cache` local. A hit clones the `Arc` (one atomic
    // refcount bump, no bytes touched); a miss pays the real quantize once
    // and seeds the cache for whichever sibling node reads this same
    // activation next.
    let activation_q8k: Arc<[u8]> = if let Some(cached) = quantize_cache[activation_node.0 as usize].as_ref() {
        #[cfg(feature = "instrument")]
        instrument::record_quantize_activation_cache_hit();
        Arc::clone(cached)
    } else {
        let mut buffer = vec![0u8; block_count * Q8K_BLOCK_BYTES];
        // `Some(session)` here is safe, unlike inside a stage's own
        // `run_stage_chunk` closure: this call runs during the precompute
        // pass, strictly BEFORE `run_staged_batch` opens its round
        // (`session.run(&round)` has not been called yet), so there is no
        // in-flight round for a second `session.run` to collide with.
        // Matches `run_reduce_quantized`'s own unbatched call exactly (same
        // function, same session), so a wide (prefill-shaped) activation
        // keeps its existing parallel quantize instead of losing it just
        // because this node got folded.
        //
        // instrumentation-only: a DEDICATED counter (`STAGED_MATMUL_QUANTIZE_TICKS`),
        // not a second call site into `MATMUL_QUANTIZE_ACTIVATION_TICKS` -- see
        // that counter's own doc for why sharing it across both call sites broke
        // `matmul_split`'s own nested-subset arithmetic. Before this counter
        // existed, this call site had no attribution at all: the staged path's
        // own quantize cost (160/225 matmul nodes per step, ROW97/98's dominant
        // bucket) was invisible.
        #[cfg(feature = "instrument")]
        let diag_staged_quantize_started = instrument::read_ticks();
        // ROW 140's own redundant-quantize hypothesis check: recorded on a
        // CACHE MISS only, i.e. once per distinct activation node this step
        // actually pays a real quantize for -- see
        // `instrument::quantize_activation_call_stats`'s own doc.
        #[cfg(feature = "instrument")]
        instrument::record_quantize_activation_call(activation_node);
        quantize_row_q8k_dispatch(activation, &mut buffer, Some(session))?;
        #[cfg(feature = "instrument")]
        counter!(
            instrument::STAGED_MATMUL_QUANTIZE_TICKS,
            instrument::elapsed_ticks(diag_staged_quantize_started)
        );
        let shared: Arc<[u8]> = Arc::from(buffer);
        quantize_cache[activation_node.0 as usize] = Some(Arc::clone(&shared));
        shared
    };
    #[cfg(feature = "instrument")]
    {
        let macs = (rows as u64).saturating_mul(k as u64).saturating_mul(leading_total as u64);
        counter!(instrument::STAGED_MATMUL_MACS, macs);
        counter!(instrument::STAGED_MATMUL_NODES, 1);
    }

    let row_bytes = weights.len() / rows;
    let chunk_count = row_chunk_count(rows, workers, k.saturating_mul(leading_total));
    let chunk_len = rows.div_ceil(chunk_count);
    let mut wide = vec![0.0f32; rows * leading_total];
    let mut chunk_ranges = Vec::with_capacity(chunk_count);
    let mut remaining = wide.as_mut_slice();
    let mut row_start = 0usize;
    while !remaining.is_empty() {
        let take_rows = chunk_len.min(remaining.len() / leading_total);
        let (slice, rest) = remaining.split_at_mut(take_rows * leading_total);
        remaining = rest;
        chunk_ranges.push((row_start, slice.as_mut_ptr() as usize, slice.len()));
        row_start += take_rows;
    }

    Ok(Some(MatmulStagePlan {
        weights,
        activation_q8k,
        row_bytes,
        q8k_row_bytes,
        width: leading_total,
        rows,
        dot_fn,
        wide,
        chunk_ranges,
    }))
}

/// Runs `run` (a maximal [`is_staged_batch_eligible`] slice of `resolved`,
/// starting at `resolved[run_start]`) as ONE [`StagedRound`] instead of one
/// `CohortSession::run` per node — the fix `docs/discipline.md` ROW 68/90/96
/// point at: threads stay resident and busy-spin through every stage of the
/// whole run behind a single round-open/wake, INCLUDING the quantized-matmul
/// stages that are ~87% of a forward's own wall time (ROW 96 folded every
/// OTHER kind and measured `rounds` rise, not fall, because those kinds
/// already opened zero rounds on their own — matmul is where the existing
/// per-node rounds actually live).
///
/// A node with a [`MatmulStagePlan`] becomes a many-chunk stage (real
/// cross-worker row parallelism, [`MatmulStagePlan::run_chunk`]); a matmul
/// node too small to parallelize (see [`build_matmul_stage_plan`]'s own
/// doc — the only way a node in `run` lacks a plan, since
/// [`is_staged_batch_eligible`] admits nothing else) is a single one-chunk
/// stage running the exact [`run_node_into`] call the unbatched path would
/// have made, `session: None` because that specific node would ALSO
/// serial-fallback with a real session (the same `quantized_matmul_workers`
/// threshold gates both), never because of round reentrancy.
///
/// Every output buffer for the whole run is allocated up front (reusing
/// [`take_or_allocate`], the same pool [`evaluate_quantized_with_scratch`]'s
/// per-node path already draws from) so the round's own closure can hold a
/// raw pointer to each stage's own disjoint slot before the round opens —
/// retirement (`retires`) is applied after the round returns, in run order,
/// so the final `buffers`/`free_buffers` state this leaves is identical to
/// running every node in `run` one at a time; the only difference is that a
/// buffer whose last use falls inside the run is held slightly longer
/// (until the run's own round returns) instead of being freed the instant
/// its consumer finishes — bounded by one run's own total output size, not
/// the whole step's.
#[cfg(feature = "cohort-staged-graph")]
#[allow(clippy::too_many_arguments)]
fn run_staged_batch(
    run: &[BoundOp],
    run_start: usize,
    buffers: &mut [Option<Cow<'_, [f32]>>],
    quantized_weights: &BTreeMap<NodeId, QuantizedBlock>,
    session: &MatmulSession<'_>,
    free_buffers: &mut Vec<Vec<f32>>,
    retires: &[Vec<NodeId>],
    live_now: &mut usize,
    quantize_cache: &mut [Option<Arc<[u8]>>],
) -> Result<(), TensorError> {
    let mut run_outputs: Vec<Vec<f32>> =
        run.iter().map(|node| take_or_allocate(free_buffers, node_output_len(node))).collect();
    let buffers_ref: &[Option<Cow<'_, [f32]>>] = buffers;

    let mut plans: Vec<Option<MatmulStagePlan<'_>>> = Vec::with_capacity(run.len());
    for node in run {
        let plan = match quantized_operand(node, quantized_weights) {
            Some(weight_node) => {
                let weight_block = quantized_weights.get(&weight_node).copied().ok_or(TensorError::NotLowerable {
                    node: weight_node,
                    reason: "quantized weight node has no bound byte buffer",
                })?;
                build_matmul_stage_plan(node, buffers_ref, weight_block, weight_node, session, quantize_cache)?
            }
            None => None,
        };
        plans.push(plan);
    }

    let stage_offsets: Vec<usize> = core::iter::once(0)
        .chain(plans.iter().scan(0usize, |total, plan| {
            *total += plan.as_ref().map_or(1, |plan| plan.chunk_ranges.len());
            Some(*total)
        }))
        .collect();
    let output_slots: Vec<(usize, usize)> =
        run_outputs.iter_mut().map(|buffer| (buffer.as_mut_ptr() as usize, buffer.len())).collect();
    let completed: Vec<AtomicUsize> = (0..run.len()).map(|_| AtomicUsize::new(0)).collect();

    let round = StagedRound {
        stage_offsets: &stage_offsets,
        completed: &completed,
        run_stage_chunk: |stage: usize, within: usize| -> Result<(), TensorError> {
            match &plans[stage] {
                Some(plan) => plan.run_chunk(within),
                None => {
                    let computed = &run[stage];
                    let (address, length) = output_slots[stage];
                    // SAFETY: single-writer argument identical to
                    // `ElementwiseRowRound`/`RowRound`'s own `split_at_mut`-carved
                    // ranges — one stage owns this whole slot (`plans[stage]`
                    // is `None`, so this stage is exactly one chunk).
                    let output = unsafe { core::slice::from_raw_parts_mut(address as *mut f32, length) };
                    run_node_into(computed, buffers_ref, Some(quantized_weights), None, output)
                }
            }
        },
    };

    // instrumentation-only: times `session.run(&round)` as a whole, the
    // same granularity `matmul_rows_threaded`'s own `MATMUL_OWN_CHUNK_TICKS`
    // uses for the unbatched leader-claim-and-wait call -- not per-chunk
    // (that would perturb the very thing being measured, see this module's
    // own doc) and not folded into `MATMUL_OWN_CHUNK_TICKS` itself, since
    // that counter's own denominator (`MATMUL_DISPATCH_CALLS`/per-node call
    // count) means something different for a run that folds several matmul
    // nodes into one round.
    #[cfg(feature = "instrument")]
    let diag_staged_round_started = instrument::read_ticks();
    let report = session.run(&round);
    #[cfg(feature = "instrument")]
    counter!(
        instrument::STAGED_MATMUL_ROUND_TICKS,
        instrument::elapsed_ticks(diag_staged_round_started)
    );
    if let Some(error) = report.first_error {
        return Err(error);
    }
    if report.abandoned > 0 {
        return Err(TensorError::ThreadedChunkFailed {
            chunk: report.first_abandoned.map_or(0, |chunk| chunk.0 + 1),
            reason: alloc::string::String::from("cohort member panicked while running a staged graph batch"),
        });
    }

    // `Some(session)` here is safe for the identical reason
    // `build_matmul_stage_plan`'s own quantize call is: this loop runs
    // strictly AFTER `session.run(&round)` above has already returned, so
    // the round this session was driving is closed before any of these
    // transpose calls open a new one.
    #[cfg(feature = "instrument")]
    let diag_staged_transpose_started = instrument::read_ticks();
    for (offset, node_output) in run_outputs.iter_mut().enumerate() {
        if let Some(plan) = &plans[offset] {
            transpose_wide_to_output(&plan.wide, plan.rows, plan.width, Some(session), node_output)?;
        }
    }
    #[cfg(feature = "instrument")]
    counter!(
        instrument::STAGED_MATMUL_TRANSPOSE_TICKS,
        instrument::elapsed_ticks(diag_staged_transpose_started)
    );

    for (offset, node_output) in run_outputs.into_iter().enumerate() {
        let node = run[offset].node;
        buffers[node.0 as usize] = Some(Cow::Owned(node_output));
        *live_now += 1;
        for retired in &retires[run_start + offset] {
            if retire_into(buffers, *retired, free_buffers) {
                *live_now -= 1;
            }
        }
    }
    Ok(())
}

/// [`Op::Reduce`] with a data-dependent (scatter) `out_map`, `f32` only —
/// the dedicated sequential path every fast path in [`run_reduce`] stays
/// ineligible for (`bind::BoundOp::split`'s own doc has the reason: no chunk
/// rebase story for `out_scatter`, so a scatter never reaches the NEON/dot/
/// width tiles above, which all assume a `Keep::Reduce` fold owns its own
/// disjoint output range).
///
/// No atomics: the CPU interpreter already walks its reduce loop strictly
/// in iteration order, one coordinate at a time, so a colliding write is
/// just another `reduce_op` fold applied to `output[dest]` in place —
/// nothing else can observe or mutate `output` mid-walk. This is the
/// forward half of the worked example `map.rs`'s `IndexMap::Computed` doc
/// and this function's own tests name: `src=[10,20,30,40]`,
/// `idx=[2,0,2,1]`, destination extent 3, body `Add`, `init` `Zero` ->
/// `out=[20,40,40]` (`out[2]` folds `10` then `30`, in iteration order).
///
/// `output` is filled with `init`'s identity *before* the walk (`init ==
/// ReduceInit::FirstElement` is rejected at shape-inference time — see
/// `shape.rs`'s `infer_reduce` — because which source element is "first" at
/// a colliding destination is not well-defined), since which cells a
/// data-dependent write ever touches is unknown until the fetched indices
/// are read.
fn run_reduce_scatter<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        out_layout,
        out_scatter,
        ..
    } = &resolved.kind
    else {
        unreachable!("run_reduce_scatter is only called for a Keep::Reduce fold")
    };
    let Some(target) = out_scatter else {
        unreachable!("run_reduce_scatter is only dispatched when out_scatter is Some")
    };

    output.fill(initial_value(*init).unwrap_or(0.0));

    let raw = operand_buffers(resolved, buffers)?;
    let body = resolved.element_body();
    let shape = body_shape(body);
    let mut operand_values = vec![0.0f32; raw.len()];
    let mut step_values = vec![0.0f32; body.steps.len()];

    let index_buffer = buffer_of(buffers, target.indices)?;
    let mut running: Vec<i64> = vec![0; raw.len()];
    let mut gather_cursors: Vec<Option<GatherCursor>> = (0..raw.len()).map(|_| None).collect();
    let mut coordinate = vec![0u64; resolved.extents.len()];
    let iteration_total = odometer_len(&resolved.extents);

    for flat in 0..iteration_total {
        unflatten_into(flat, &resolved.extents, &mut coordinate);
        fill_running_offsets(resolved, &coordinate, &mut running);
        fill_gather_cursors(resolved, buffers, &coordinate, None, &mut gather_cursors)?;

        for (index, data) in raw.iter().enumerate() {
            let mut offset = running[index];
            if let Some(cursor) = gather_cursors[index].as_mut() {
                offset += cursor.fetch_and_advance(resolved.node)?;
            }
            operand_values[index] = data[offset as usize];
        }
        let value = eval_body_shape(&shape, &operand_values, &mut step_values);

        let mut destination = GatherCursor {
            buffer: index_buffer,
            offset: target.index_layout.offset_of(&coordinate),
            stride: 0,
            element_stride: target.element_stride,
            extent: target.extent,
        };
        let dest_offset = out_layout.offset_of(&coordinate) + destination.fetch_and_advance(resolved.node)?;
        let slot = &mut output[dest_offset as usize];
        *slot = apply_scalar_op(*reduce_op, &[*slot, value]);
    }
    Ok(())
}

/// [`run_reduce`]'s quantized-weight branch: `resolved` is the fused
/// `Reduce(Elementwise(Multiply))` matmul shape, `weight_node` one of its two
/// operands, packed `Q4_K` bytes rather than a bound `f32` buffer. The other
/// operand is the plain `f32` activation, already sitting in `buffers` like
/// any other node — read straight out of the same table [`run_reduce`]'s f32
/// path uses, no second buffer convention for it.
///
/// [`matmul_q4k_f32`] itself only knows one activation vector times one
/// weight matrix — batch-1. A real forward pass batches every sequence
/// position through the same weight in one call (`mistral_forward_program`'s
/// `wq` node alone folds `s`, `h`, and `d` together into one packed-row
/// dimension: `"ihd->shdi"` broadcasts the same `[s, i]` activation across
/// every head, so the physical weight row a given `(h, d)` pair needs is
/// `h * head_dim + d`, exactly GGUF's own on-disk row order for a
/// `[embedding_in, embedding_out]` projection reinterpreted as heads x
/// head_dim). Rather than re-deriving that per-op axis grouping here, `k`
/// (the contraction width) and `rows` (the packed weight's own row count)
/// are both derived from data already at hand — `k` from `resolved`'s own
/// reduced dims exactly as [`run_reduce`] computes them, `rows` from
/// `weights.len()` divided by `k`'s worth of packed bytes — so `rows` comes
/// out correct regardless of how many *program* output axes the weight's
/// flat row dimension was split into. `leading_total = output.len() / rows`
/// then folds every one of those non-reduced output axes (`s`, `h`, `d`,
/// ...) into one batch loop, one [`matmul_q4k_f32`] call per position.
fn run_reduce_quantized<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    weight_block: QuantizedBlock,
    weight_node: NodeId,
    session: Option<&MatmulSession<'_>>,
    output: &mut [f32],
) -> Result<(), TensorError> {
    // proxima-debugger diagnostic: whole-function timer, once per matmul
    // node (the same granularity `evaluate_quantized`'s per-node-kind
    // table uses) -- localizes whether a gap between a node's total wall
    // time and the sum of matmul_rows_threaded's own timers sits inside
    // this function's position loop or outside it entirely.
    #[cfg(feature = "instrument")]
    let diag_reduce_quantized_started = instrument::read_ticks();
    // `session` only reaches a call site when its codec's own `q{4,5,6}k-int8-dot`
    // feature is on (the arms below); a build with every one of those off
    // never reads it, so bind it unconditionally here rather than let a
    // rare feature combination trip an unused-parameter warning.
    let _ = session;
    // A growable cache (`Q8_0`, see `QuantizedBlock::Q8_0`'s own doc) binds
    // a zero-length weight buffer on its very first call (`cached_len ==
    // 0`), which makes this reduce's own output axes multiply out to zero
    // elements too -- nothing to write, and no legal `rows` (weight rows /
    // contraction width) to derive from an empty buffer. A static weight
    // matmul (`Q4_K`/`Q5_K`/`Q6_K`) never binds an empty operand, so this
    // is additive for that path, not a behavior change.
    if output.is_empty() {
        return Ok(());
    }
    let activation_node = resolved
        .operands()
        .iter()
        .map(|(node, _, _)| *node)
        .find(|node| *node != weight_node)
        .ok_or(TensorError::NotLowerable {
            node: resolved.node,
            reason: "quantized matmul reduce has no activation operand",
        })?;
    let activation = buffers[activation_node.0 as usize].as_deref().ok_or(TensorError::NotLowerable {
        node: activation_node,
        reason: "quantized matmul activation operand has no bound buffer",
    })?;
    // ROW 140's own redundant-quantize hypothesis check, unbatched-path
    // twin of `build_matmul_stage_plan`'s call: this function's own
    // wide-fold arms below (`matmul_q4k_q8k_f32_impl` et al.) each quantize
    // `activation` fresh, so if two sibling matmul nodes (`ffn_gate`,
    // `ffn_up`) both reach `run_reduce_quantized` with the SAME
    // `activation_node`, this key sees two calls for one distinct node.
    #[cfg(feature = "instrument")]
    instrument::record_quantize_activation_call(activation_node);

    let BoundOpKind::Reduce { output_axes, .. } = &resolved.kind else {
        unreachable!("run_reduce_quantized is only called for a Keep::Reduce fold")
    };
    // Single-sourced from the same resolved axis structure `run_reduce`
    // reads (`resolve_reduce_axis_shape`), not a second, independent
    // derivation from raw packed-weight byte lengths -- that second
    // derivation is what let this shape drift out of step with the whole
    // reduce's own `output_axes`/`extents` on a cached-attention fold (see
    // `ReduceAxisShape`'s own doc).
    let axis_shape = resolve_reduce_axis_shape(resolved, output_axes.as_slice());
    let contraction_width: u64 = axis_shape.reduction_extents.iter().product();
    let shape_error = || TensorError::NotLowerable {
        node: resolved.node,
        reason: "quantized matmul batch shape does not evenly divide by its packed weight rows",
    };
    let shared_axis_error = || TensorError::NotLowerable {
        node: resolved.node,
        reason: "quantized matmul activation varies along an output axis its packed weight also \
                 varies along -- not a flat weight matmul this interpreter can express",
    };
    let k = usize::try_from(contraction_width).map_err(|_| shape_error())?;

    let weight_layout = resolved
        .operands()
        .iter()
        .find(|(node, _, _)| *node == weight_node)
        .map(|(_, layout, _)| layout)
        .ok_or_else(shape_error)?;
    let activation_layout = resolved
        .operands()
        .iter()
        .find(|(node, _, _)| *node == activation_node)
        .map(|(_, layout, _)| layout)
        .ok_or_else(shape_error)?;

    // Every output axis the packed weight varies over (nonzero stride) is
    // one of its own physical rows; every output axis it broadcasts over
    // (stride 0) is a batch position the same rows are reused for across —
    // `matmul_q4k_f32`'s target shape, `[rows, k] x [k] -> [rows]` called
    // once per batch position, never a byte-length division. An axis the
    // activation ALSO varies over while the weight does too cannot be
    // folded into either bucket: the loop below dots ONE activation vector
    // against every packed row, which only holds when the activation is
    // constant across whichever axes the weight's own rows enumerate.
    // A packed weight carrying `IndexMap::Computed` over its own leading
    // (batch) axis -- `moe_block.toml`'s `expert_w` gather -- selects one
    // whole `[rows, k]` expert slab per batch position out of an
    // `[n_experts, rows, k]` stack (`proxima-gguf/src/restack.rs`'s own
    // module doc: byte concatenation, block-aligned by construction). The
    // gathered axis itself never appears in `output_axes`/`resolved.extents`
    // at all -- only the token axis that *drives* the gather does, and that
    // axis already lands in the broadcast (`stride == 0`) bucket below, same
    // as any other batch position. `leading_axes`/`leading_extents` are only
    // populated when a gather is present, so the non-gathered path (every
    // codec this crate ran before Mixtral) allocates nothing extra here.
    let weight_gather = resolved.operands().iter().find(|(node, _, _)| *node == weight_node).and_then(|(_, _, gather)| gather.clone());
    let mut rows_total: u64 = 1;
    let mut leading_total_u64: u64 = 1;
    let mut leading_axes: Vec<u16> = Vec::new();
    let mut leading_extents: Vec<u64> = Vec::new();
    for axis in output_axes.as_slice() {
        let extent = resolved.extents[*axis as usize];
        if weight_layout.stride(*axis) != 0 {
            if activation_layout.stride(*axis) != 0 {
                return Err(shared_axis_error());
            }
            rows_total *= extent;
        } else {
            leading_total_u64 *= extent;
            if weight_gather.is_some() {
                leading_axes.push(*axis);
                leading_extents.push(extent);
            }
        }
    }
    let rows = usize::try_from(rows_total).map_err(|_| shape_error())?;
    let leading_total = usize::try_from(leading_total_u64).map_err(|_| shape_error())?;

    // Every K-quant weight codec this crate packs shares `Q4K_BLOCK_ELEMENTS`
    // (256) elements per super-block (`q5_k`/`q6_k`'s own module docs: same
    // `QK_K`); `Q8_0`'s block is a different, much smaller shape (32
    // elements, no sub-block structure) -- the growable key/value context
    // cache's rows are `HEAD_DIM / 2` wide, too narrow for a 256-element
    // super-block to divide evenly without straddling more than one cached
    // position, so `block_elements` varies per codec rather than being one
    // shared constant.
    let (weights, block_bytes, block_elements): (&[u8], usize, usize) = match weight_block {
        QuantizedBlock::Float32(_) => return Err(shape_error()),
        QuantizedBlock::Q4K(bytes) => (bytes, Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS),
        QuantizedBlock::Q5K(bytes) => (bytes, Q5K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS),
        QuantizedBlock::Q6K(bytes) => (bytes, Q6K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS),
        QuantizedBlock::Q8_0(bytes) => (bytes, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS),
        QuantizedBlock::Q4_0(bytes) => (bytes, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS),
        QuantizedBlock::Float16(bytes) => (bytes, HALF_PRECISION_ELEMENT_BYTES, 1),
        QuantizedBlock::BFloat16(bytes) => (bytes, HALF_PRECISION_ELEMENT_BYTES, 1),
    };
    if k == 0 || rows == 0 || !weights.len().is_multiple_of(block_bytes) {
        return Err(shape_error());
    }
    // A gathered weight packs `n_experts` (`weight_gather.extent`) whole
    // `[rows, k]` slabs back to back -- `per_expert_bytes` below is the same
    // byte-concatenation arithmetic `proxima-gguf::restack::plan_stack`
    // already validated at load time (block-aligned by construction, per
    // that module's own doc), re-derived here rather than trusted blind so
    // a stacked buffer that does NOT evenly divide by `rows * k` blocks is a
    // typed `NotLowerable`, never a silent wrong-expert read. The
    // non-gathered check below (`total_weight_elements != rows * k`) stays
    // byte-for-byte what it always was.
    let per_expert_bytes = if weight_gather.is_some() {
        let per_expert_elements = rows.checked_mul(k).ok_or_else(shape_error)?;
        if per_expert_elements == 0 || !per_expert_elements.is_multiple_of(block_elements) {
            return Err(shape_error());
        }
        (per_expert_elements / block_elements) * block_bytes
    } else {
        0
    };
    if let Some(gather) = weight_gather.as_ref() {
        let expert_count = usize::try_from(gather.extent).map_err(|_| shape_error())?;
        let expected_total_bytes = per_expert_bytes.checked_mul(expert_count).ok_or_else(shape_error)?;
        if weights.len() != expected_total_bytes {
            return Err(shape_error());
        }
    } else {
        // The packed buffer's own byte length must hold exactly the
        // structurally-derived `rows * k` elements -- still validated, just no
        // longer the SOURCE `rows` is derived from.
        let total_weight_elements = (weights.len() / block_bytes) * block_elements;
        if total_weight_elements != rows * k {
            return Err(shape_error());
        }
    }
    if output.len() != leading_total * rows || activation.len() != leading_total * k {
        return Err(shape_error());
    }

    #[cfg(feature = "instrument")]
    counter!(instrument::MATMUL_REDUCE_QUANTIZED_CALLS, 1);

    // `Q4_K` (216 of 225 matmul weights in a real forward, 40.37e9 of
    // 42.66e9 macs) folds every position into one `matmul_q4k_q8k_f32_impl`
    // call so each weight row's bytes are streamed once and its dot reused
    // across `leading_total` positions, instead of the per-position loop
    // below re-streaming the whole weight matrix once per position.
    // `Q5_K`/`Q6_K` (9 of 225 weights) use the identically-shaped
    // `matmul_q5k_q8k_f32_impl`/`matmul_q6k_q8k_f32_impl` wide calls below.
    // A gathered weight varies its whole slab per position -- the wide fold
    // below streams `weights` once and reuses it across every position in
    // `leading_total`, which only holds when every position dots against the
    // SAME weight. Skipping it here (rather than teaching it a per-position
    // slab swap) routes a gathered node into the per-position loop below,
    // which resolves the gather itself.
    #[cfg(feature = "q4k-int8-dot")]
    if weight_gather.is_none()
        && let QuantizedBlock::Q4K(_) = weight_block
    {
        #[cfg(feature = "instrument")]
        let diag_call_started = instrument::read_ticks();
        let wide = matmul_q4k_q8k_f32_impl(weights, rows, activation, leading_total, session)?;
        #[cfg(feature = "instrument")]
        {
            let diag_call_ticks = instrument::elapsed_ticks(diag_call_started);
            let diag_call_macs = (rows as u64) * (k as u64) * (leading_total as u64);
            counter!(instrument::MATMUL_Q4K_MACS, diag_call_macs);
            counter!(instrument::MATMUL_Q4K_CALL_TICKS, diag_call_ticks);
            instrument::record_q4k_shape_call(rows as u64, (k as u64) * leading_total as u64, diag_call_macs, diag_call_ticks);
        }
        // `wide` is row-major `[row][position]` — `matmul_rows_threaded`'s
        // natural shape, weight row as the parallel axis. `output` here is
        // position-major `[position][row]` (the layout the per-position
        // loop below writes, and what downstream consumes unchanged) — this
        // copy is the transpose back, `O(rows * leading_total)`, far
        // cheaper than the weight stream it replaces.
        #[cfg(feature = "instrument")]
        let diag_transpose_started = instrument::read_ticks();
        transpose_wide_to_output(&wide, rows, leading_total, session, output)?;
        #[cfg(feature = "instrument")]
        counter!(
            instrument::MATMUL_Q4K_TRANSPOSE_TICKS,
            instrument::elapsed_ticks(diag_transpose_started)
        );
        #[cfg(feature = "instrument")]
        counter!(
            instrument::MATMUL_REDUCE_QUANTIZED_TICKS,
            instrument::elapsed_ticks(diag_reduce_quantized_started)
        );
        return Ok(());
    }

    // `Q5_K`'s wide-fold arm -- same mechanism as the `Q4_K` arm above,
    // `matmul_q5k_q8k_f32_impl` in place of `matmul_q4k_q8k_f32_impl`. No
    // per-codec transpose-tick counter exists for `Q5_K` (only
    // `MATMUL_Q4K_TRANSPOSE_TICKS` does); the transpose's cost is still
    // captured inside `MATMUL_REDUCE_QUANTIZED_TICKS` below, which is not
    // codec-specific. Gated the same way the `Q4_K` arm above is -- a
    // gathered weight cannot use this single-flat-matrix wide fold.
    #[cfg(feature = "q5k-int8-dot")]
    if weight_gather.is_none()
        && let QuantizedBlock::Q5K(_) = weight_block
    {
        #[cfg(feature = "instrument")]
        let diag_call_started = instrument::read_ticks();
        let wide = matmul_q5k_q8k_f32_impl(weights, rows, activation, leading_total, session)?;
        #[cfg(feature = "instrument")]
        {
            let diag_call_ticks = instrument::elapsed_ticks(diag_call_started);
            let diag_call_macs = (rows as u64) * (k as u64) * (leading_total as u64);
            counter!(instrument::MATMUL_Q5K_MACS, diag_call_macs);
            counter!(instrument::MATMUL_Q5K_CALL_TICKS, diag_call_ticks);
        }
        transpose_wide_to_output(&wide, rows, leading_total, session, output)?;
        #[cfg(feature = "instrument")]
        counter!(
            instrument::MATMUL_REDUCE_QUANTIZED_TICKS,
            instrument::elapsed_ticks(diag_reduce_quantized_started)
        );
        return Ok(());
    }

    // `Q6_K`'s wide-fold arm -- same mechanism as the `Q4_K` arm above,
    // `matmul_q6k_q8k_f32_impl` in place of `matmul_q4k_q8k_f32_impl`. Gated
    // the same way the `Q4_K` arm above is.
    #[cfg(feature = "q6k-int8-dot")]
    if weight_gather.is_none()
        && let QuantizedBlock::Q6K(_) = weight_block
    {
        #[cfg(feature = "instrument")]
        let diag_call_started = instrument::read_ticks();
        let wide = matmul_q6k_q8k_f32_impl(weights, rows, activation, leading_total, session)?;
        #[cfg(feature = "instrument")]
        {
            let diag_call_ticks = instrument::elapsed_ticks(diag_call_started);
            let diag_call_macs = (rows as u64) * (k as u64) * (leading_total as u64);
            counter!(instrument::MATMUL_Q6K_MACS, diag_call_macs);
            counter!(instrument::MATMUL_Q6K_CALL_TICKS, diag_call_ticks);
        }
        transpose_wide_to_output(&wide, rows, leading_total, session, output)?;
        #[cfg(feature = "instrument")]
        counter!(
            instrument::MATMUL_REDUCE_QUANTIZED_TICKS,
            instrument::elapsed_ticks(diag_reduce_quantized_started)
        );
        return Ok(());
    }

    // A gathered weight resolves one expert slab out of the stacked buffer
    // per position, using the same coordinate -> `Layout::offset_of`
    // machinery `fill_gather_cursors` uses for the dense f32 gather path
    // (`fill_gather_cursors`/`GatherCursor`, this module) -- reused here as
    // a byte-offset computation layered on the packed weight's byte buffer,
    // not a second index-resolution mechanism. `leading_coordinate`/
    // `full_coordinate` stay empty `Vec`s (no allocation) on the
    // non-gathered path.
    let mut leading_coordinate = if weight_gather.is_some() { vec![0u64; leading_axes.len()] } else { Vec::new() };
    let mut full_coordinate = if weight_gather.is_some() { vec![0u64; resolved.extents.len()] } else { Vec::new() };

    #[cfg(feature = "instrument")]
    counter!(instrument::MATMUL_POSITION_LOOP_ITERS, leading_total as u64);
    for position in 0..leading_total {
        let activation_row = &activation[position * k..(position + 1) * k];
        let weights: &[u8] = if let Some(gather) = weight_gather.as_ref() {
            unflatten_into(position as u64, &leading_extents, &mut leading_coordinate);
            merge_coordinates_into(&leading_axes, &leading_coordinate, &[], &[], &mut full_coordinate);
            let index_buffer = buffer_of(buffers, gather.indices)?;
            let index_offset = usize::try_from(gather.index_layout.offset_of(&full_coordinate)).map_err(|_| shape_error())?;
            let raw_index = *index_buffer.get(index_offset).ok_or_else(shape_error)?;
            let expert_index = raw_index as i64;
            if expert_index < 0 || expert_index as u64 >= gather.extent {
                return Err(TensorError::GatherIndexOutOfRange {
                    node: resolved.node,
                    index: expert_index,
                    extent: gather.extent,
                });
            }
            let start = expert_index as usize * per_expert_bytes;
            weights.get(start..start + per_expert_bytes).ok_or_else(shape_error)?
        } else {
            weights
        };
        // proxima-debugger diagnostic: per-position, per-codec call timer
        // plus `rows * k` mac count -- localizes whether the missing 2x is
        // inside one codec's kernel (ns/mac far above the isolated
        // single-threaded bench) or purely dispatch overhead multiplied by
        // `leading_total` separate `matmul_rows_threaded` rounds (one per
        // position, never folded into a single wider row-batch).
        #[cfg(feature = "instrument")]
        let diag_call_started = instrument::read_ticks();
        let result = match weight_block {
            QuantizedBlock::Float32(_) => return Err(shape_error()),
            QuantizedBlock::Q4K(_) => {
                // unreachable when `q4k-int8-dot` is on: the wide fold above
                // already handled and returned for every `Q4K` weight.
                // Kept compiling (not `unreachable!()`) only because this
                // match still names all three `QuantizedBlock` variants.
                #[cfg(feature = "q4k-int8-dot")]
                {
                    matmul_q4k_q8k_f32_impl(weights, rows, activation_row, 1, session)?
                }
                #[cfg(not(feature = "q4k-int8-dot"))]
                {
                    matmul_q4k_f32(weights, rows, activation_row)?
                }
            }
            QuantizedBlock::Q5K(_) => {
                // unreachable when `q5k-int8-dot` is on: the wide fold above
                // already handled and returned for every `Q5K` weight. Kept
                // compiling (not `unreachable!()`) only because this match
                // still names all three `QuantizedBlock` variants.
                #[cfg(feature = "q5k-int8-dot")]
                {
                    matmul_q5k_q8k_f32_impl(weights, rows, activation_row, 1, session)?
                }
                #[cfg(not(feature = "q5k-int8-dot"))]
                {
                    matmul_q5k_f32(weights, rows, activation_row)?
                }
            }
            QuantizedBlock::Q6K(_) => {
                // unreachable when `q6k-int8-dot` is on: the wide fold above
                // already handled and returned for every `Q6K` weight. Kept
                // compiling (not `unreachable!()`) only because this match
                // still names all three `QuantizedBlock` variants.
                #[cfg(feature = "q6k-int8-dot")]
                {
                    matmul_q6k_q8k_f32_impl(weights, rows, activation_row, 1, session)?
                }
                #[cfg(not(feature = "q6k-int8-dot"))]
                {
                    matmul_q6k_f32(weights, rows, activation_row)?
                }
            }
            QuantizedBlock::Q8_0(_) => matmul_q8_0_f32(weights, rows, activation_row)?,
            QuantizedBlock::Q4_0(_) => matmul_q4_0_f32(weights, rows, activation_row)?,
            QuantizedBlock::Float16(_) => matmul_f16_f32(weights, rows, activation_row)?,
            QuantizedBlock::BFloat16(_) => matmul_bf16_f32(weights, rows, activation_row)?,
        };
        #[cfg(feature = "instrument")]
        {
            let diag_call_ticks = instrument::elapsed_ticks(diag_call_started);
            let diag_call_macs = (rows as u64) * (k as u64);
            match weight_block {
                QuantizedBlock::Float32(_) => {}
                QuantizedBlock::Q4K(_) => {
                    counter!(instrument::MATMUL_Q4K_MACS, diag_call_macs);
                    counter!(instrument::MATMUL_Q4K_CALL_TICKS, diag_call_ticks);
                    instrument::record_q4k_shape_call(rows as u64, k as u64, diag_call_macs, diag_call_ticks);
                }
                QuantizedBlock::Q5K(_) => {
                    counter!(instrument::MATMUL_Q5K_MACS, diag_call_macs);
                    counter!(instrument::MATMUL_Q5K_CALL_TICKS, diag_call_ticks);
                }
                QuantizedBlock::Q6K(_) => {
                    counter!(instrument::MATMUL_Q6K_MACS, diag_call_macs);
                    counter!(instrument::MATMUL_Q6K_CALL_TICKS, diag_call_ticks);
                }
                QuantizedBlock::Q8_0(_) => {}
                QuantizedBlock::Q4_0(_) => {}
                QuantizedBlock::Float16(_) | QuantizedBlock::BFloat16(_) => {}
            }
        }
        output[position * rows..(position + 1) * rows].copy_from_slice(&result);
    }
    #[cfg(feature = "instrument")]
    counter!(
        instrument::MATMUL_REDUCE_QUANTIZED_TICKS,
        instrument::elapsed_ticks(diag_reduce_quantized_started)
    );
    Ok(())
}

/// [`run_node_into`]'s entry point whenever a caller has any quantized
/// weight bound at all ([`evaluate_quantized`], the only one) — checks
/// [`quantized_operand`] and routes to [`run_reduce_quantized`] or falls
/// through to the plain [`run_reduce`] unchanged. Kept as its own function,
/// not folded into `run_reduce` itself, so `run_reduce`'s own compiled body
/// — what every other caller ([`evaluate`], [`evaluate_parallel`],
/// [`run_reduce_typed`]'s `f32` specialization) reaches through
/// `run_node_into`'s `quantized_weights: None` arm — carries none of this
/// check's machine code: measured to hold `run_reduce`'s own instruction
/// count exactly at its pre-quantization baseline (8629 lines, 40 `fmla`,
/// `sweep_gemm --release`) precisely because a caller that never binds a
/// quantized weight never reaches this function at all, not even through a
/// branch it doesn't take.
fn run_reduce_with_quantized_weights<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    quantized_weights: &BTreeMap<NodeId, QuantizedBlock>,
    session: Option<&MatmulSession<'_>>,
    output: &mut [f32],
) -> Result<(), TensorError> {
    if let Some(weight_node) = quantized_operand(resolved, quantized_weights) {
        let weight_block = quantized_weights.get(&weight_node).copied().ok_or(TensorError::NotLowerable {
            node: weight_node,
            reason: "quantized weight node has no bound byte buffer",
        })?;
        return run_reduce_quantized(resolved, buffers, weight_block, weight_node, session, output);
    }
    run_reduce(resolved, buffers, output)
}

/// The axis structure one `Keep::Reduce` fold's own binding already carries:
/// which of `resolved.extents`'s axes are reduced away versus kept in the
/// output, and the extents on each side of that split. Both the dense f32
/// path ([`run_reduce`]) and the quantized-weight path
/// ([`run_reduce_quantized`]) need exactly this — the only shape a
/// `Keep::Reduce` fold has — so it is derived once, here, rather than twice:
/// [`run_reduce_quantized`] used to re-derive its own `k` by dividing raw
/// packed-weight byte lengths instead of reading `output_axes` the way this
/// does, which is what let its shape drift out of step with this one
/// (`proxima-tensor` cached-attention quantized-seam fix).
struct ReduceAxisShape<'op> {
    reduction_dims: Vec<u16>,
    leading_output_axes: &'op [u16],
    last_output_dim: Option<u16>,
    leading_extents: Vec<u64>,
    reduction_extents: Vec<u64>,
    width: usize,
}

fn resolve_reduce_axis_shape<'op>(resolved: &BoundOp, output_axes: &'op [u16]) -> ReduceAxisShape<'op> {
    let reduction_dims: Vec<u16> = (0..resolved.extents.len() as u16).filter(|dim| !output_axes.contains(dim)).collect();
    let (leading_output_axes, last_output_dim) = output_axes_split(output_axes);

    let leading_extents: Vec<u64> = leading_output_axes.iter().map(|dim| resolved.extents[*dim as usize]).collect();
    let reduction_extents: Vec<u64> = reduction_dims.iter().map(|dim| resolved.extents[*dim as usize]).collect();
    let width = last_output_dim.map_or(1, |dim| resolved.extents[dim as usize] as usize);

    ReduceAxisShape {
        reduction_dims,
        leading_output_axes,
        last_output_dim,
        leading_extents,
        reduction_extents,
        width,
    }
}

/// True when `operand_index`'s physical layout walks the WHOLE
/// `reduction_dims` range as one contiguous, stride-1 span once every
/// element of it is visited in `reduction_dims`'s own (outer-to-inner)
/// order — the row-major contiguity chain: the innermost dim's stride is
/// exactly 1, and each dim outward is exactly the product of every
/// extent nested inside it. `reduction_dims.len() == 1` degenerates to
/// today's original single-dim check (`stride(dims[0]) == 1`) unchanged.
/// `docs/discipline.md` ROW 148 measured this true for mnist's own first
/// FC layer (both operands: a rank-3 `[c,h,w]` activation and a
/// matching-shaped weight, neither ever reshaped through an explicit
/// flatten) and false for `Conv`'s materialized `windowed` operand, whose
/// `ci` axis sits outside the window's `oh`/`ow` axes in memory — the
/// exact mechanism `run_reduce`'s own `reduction_strides` doc cites.
fn reduction_is_fully_flat(resolved: &BoundOp, reduction_dims: &[u16], operand_index: usize) -> bool {
    max_flat_reduction_suffix_len(resolved, reduction_dims, operand_index) == reduction_dims.len()
}

/// [`reduction_is_fully_flat`]'s own row-major contiguity chain, generalized
/// from a bool to a count: how many of `reduction_dims`'s TRAILING entries
/// (innermost-first, matching that function's own `.iter().rev()` walk) this
/// operand reads as one contiguous stride-1 span before the chain first
/// breaks. `reduction_dims.len()` degenerates to today's original
/// whole-range check unchanged. `docs/discipline.md` ROW 149 uses this to
/// find `Conv`'s own inner-contiguous-block boundary (`ky,kx`, length 2)
/// once the outer `ci` axis breaks the chain `Conv`'s materialized
/// `windowed` operand never satisfies as a whole.
fn max_flat_reduction_suffix_len(resolved: &BoundOp, reduction_dims: &[u16], operand_index: usize) -> usize {
    let view = &resolved.operands()[operand_index].1;
    let mut expected: i64 = 1;
    let mut len = 0usize;
    for &dim in reduction_dims.iter().rev() {
        if view.stride(dim) != expected {
            break;
        }
        expected = expected.saturating_mul(resolved.extents[dim as usize] as i64);
        len += 1;
    }
    len
}

/// The dense f32 GEMM interpreter: NEON dot/width tiles then a generic
/// fallback. Never sees a quantized weight — [`run_node_into`] routes any
/// call with `quantized_weights: Some(_)` through
/// [`run_reduce_with_quantized_weights`] instead, so this function's own
/// signature and body stay exactly what they were before quantized weights
/// existed anywhere in this module.
fn run_reduce<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        output_axes,
        out_layout,
        ..
    } = &resolved.kind
    else {
        unreachable!("run_reduce is only called for a Keep::Reduce fold")
    };
    let raw = operand_buffers(resolved, buffers)?;
    let body = resolved.element_body();
    let shape = body_shape(body);
    let mut operand_values = vec![0.0f32; raw.len()];
    let mut step_values = vec![0.0f32; body.steps.len()];

    let ReduceAxisShape {
        reduction_dims,
        leading_output_axes,
        last_output_dim,
        leading_extents,
        reduction_extents,
        width,
    } = resolve_reduce_axis_shape(resolved, output_axes.as_slice());

    // loop-invariant: neither `last_output_dim` nor the operand views change
    // across the whole node, so this stride table is built once instead of
    // once per (leading, reduction) coordinate pair — up to ~1e6 times for a
    // 1024^3 GEMM (`proxima-tensor/docs/discipline.md` ROW 2).
    let strides: Vec<i64> = resolved
        .operands()
        .iter()
        .map(|(_, view, _)| last_output_dim.map_or(0, |dim| view.stride(dim)))
        .collect();
    let mut running: Vec<i64> = vec![0; raw.len()];
    let mut gather_cursors: Vec<Option<GatherCursor>> = (0..raw.len()).map(|_| None).collect();
    let mut leading_coordinate = vec![0u64; leading_extents.len()];
    let mut reduction_coordinate = vec![0u64; reduction_extents.len()];
    let mut full_coordinate = vec![0u64; resolved.extents.len()];
    let reduction_total = odometer_len(&reduction_extents);

    // A matmul with a transposed right-hand operand (ggml's own `mul_mat`
    // layout) has a bad width-dim stride on one operand but a GOOD stride on
    // the contraction dim `k` — both operands read `k` contiguously.
    // `reduction_strides` is `strides`'s sibling table for the whole
    // contraction range, computed once per bound op the same way;
    // `body_shape_is_affine_fast_path` is reused verbatim, just handed a
    // different dim's stride table (`proxima-tensor/docs/discipline.md`
    // ROW 10). A multi-dim contraction (`reduction_dims.len() > 1`, e.g. a
    // matmul-shaped fold whose weight operand was never reshaped through an
    // explicit flatten — mnist's own first FC layer reduces directly over
    // its rank-3 `[c,h,w]` activation) qualifies too, via
    // [`reduction_is_fully_flat`], PROVIDED every dim composes as one
    // contiguous row-major span for that operand: `docs/discipline.md` ROW
    // 148 measured this true for such an FC layer but false for `Conv`'s
    // own materialized `windowed` operand (its `ci` axis sits outside the
    // window's `oh`/`ow` axes in memory — see ROW 148's own row-major
    // layout trace), so `Conv` correctly stays ineligible here, unchanged.
    let reduction_strides: Vec<i64> = (0..resolved.operands().len())
        .map(|index| if reduction_is_fully_flat(resolved, &reduction_dims, index) { 1 } else { i64::MAX })
        .collect();

    // Resolved ONCE per bound op, never per element: whether every physical
    // operand the body shape actually reads is gather-free with a width-dim
    // stride of 0 or 1. When it holds, the width loop below skips
    // `gather_cursors`'s per-element `Option` check and `operand_values`'s
    // per-element copy entirely, reading straight-line out of `raw`'s own
    // subslices instead (`proxima-tensor/docs/discipline.md` ROW 3).
    // `Generic` bodies and any gathered operand fall back to the loop
    // unchanged. The width path wins the tie against the dot path below,
    // the ordering every ROW 3/10 measurement was taken under.
    let fast_path = body_shape_is_affine_fast_path(resolved, &shape, &strides);
    let reduction_fast_path =
        !fast_path && !reduction_dims.is_empty() && body_shape_is_affine_fast_path(resolved, &shape, &reduction_strides);

    #[cfg(feature = "instrument")]
    let mut counters = KernelCounters::default();
    #[cfg(feature = "instrument")]
    let path = if reduction_fast_path {
        Path::DotFast
    } else if fast_path {
        Path::WidthFast
    } else {
        Path::Generic
    };
    // per-path-kind wall time (residual-profile task, 2026-08-30): started
    // once `path` is known, committed at whichever of this function's three
    // early returns (or its own tail) actually fires — see
    // `instrument::record_reduce_path_ticks`'s own doc for why this is a
    // NEW, `run_reduce`-only timer rather than a reuse of `PATH_WIDTH_FAST`/
    // `PATH_GENERIC` (those are invocation counts shared with
    // `run_elementwise_range`'s own, unrelated `Path` usage).
    #[cfg(feature = "instrument")]
    let commit_started = instrument::read_ticks();

    let seed = initial_value(*init).unwrap_or(0.0);
    let leading_total = odometer_len(&leading_extents);

    // Ported from `width-wt`: the `[k,n]`-layout twin of the dot-path tile
    // below. `reduction_fast_path` (dot tile) and `fast_path` (this tile) are
    // mutually exclusive by construction (`reduction_fast_path = !fast_path
    // && ..`), so `width_tile_plan`'s own stride gate never fires on a node
    // the dot tile already claimed — no ordering dependency between the two
    // blocks, only one of them is ever `Some`.
    //
    // whole block gated to aarch64: `try_run_width_tile` is a constant-`false`
    // stub everywhere else, so building `WidthPathContext` to hand it is dead
    // work, not just a dead type.
    #[cfg(target_arch = "aarch64")]
    {
        let width_path_context = WidthPathContext {
            resolved,
            shape: &shape,
            strides: &strides,
            reduce_op: *reduce_op,
            init: *init,
            leading_output_axes,
            reduction_dims: &reduction_dims,
            last_output_dim,
            width,
            out_layout,
        };
        #[cfg(feature = "instrument")]
        let width_tile_counters_before = width_tile_counters();
        if try_run_width_tile(&width_path_context, &raw, output) {
            // the tile's own early return skips the rest of this function
            // (including the `counters.commit` call every other path reaches),
            // so this is instrument's only chance to record the node — read
            // back the invocation/fallback deltas the tile itself already
            // counted instead of re-deriving them from `leading_total`/`width`.
            #[cfg(feature = "instrument")]
            {
                let (_, invocations_after, fallback_after) = width_tile_counters();
                let (_, invocations_before, fallback_before) = width_tile_counters_before;
                let invocations_delta = invocations_after - invocations_before;
                let fallback_delta = fallback_after - fallback_before;
                let tile_elements = (WIDTH_TILE_ROWS * WIDTH_TILE_VECS * 4) as u64;
                counters.kernel_calls += invocations_delta + fallback_delta;
                counters.mac_ops += invocations_delta * tile_elements * reduction_total;
                counters.operand_loads += invocations_delta * (WIDTH_TILE_ROWS + WIDTH_TILE_VECS) as u64 * reduction_total;
                counters.mac_ops += fallback_delta * reduction_total;
                counters.operand_loads += fallback_delta * 2 * reduction_total;
                counters.leading_iters += leading_total;
                counters.output_writes += leading_total * width as u64;
                let distinct_operand_elements: u64 = raw.iter().map(|buffer| buffer.len() as u64).sum();
                counters.commit(path, distinct_operand_elements);
                instrument::record_reduce_path_ticks(path, instrument::elapsed_ticks(commit_started));
            }
            return Ok(());
        }
    }

    // `Conv`'s own reduce shape (`docs/discipline.md` ROW 148/149): the
    // reduction body's two operands own DISJOINT subsets of
    // `leading_output_axes` (weight varies only with `co`, the materialized
    // `windowed` operand only with `n,oy` plus the width dim `ox`), so
    // neither `width_tile_plan` nor `neon_tile_plan` below can ever engage —
    // both require `leading_output_axes.len() == 1`, a single axis BOTH
    // operands share. Tried only when both of those already declined
    // (`fast_path`/`reduction_fast_path` both false), so this can never
    // steal a node either existing tile already claims.
    #[cfg(target_arch = "aarch64")]
    if !fast_path && !reduction_fast_path {
        let conv_context = ConvGemmContext {
            resolved,
            shape: &shape,
            reduce_op: *reduce_op,
            init: *init,
            leading_output_axes,
            reduction_dims: &reduction_dims,
            last_output_dim,
            out_layout,
        };
        if let Some(plan) = conv_gemm_tile_plan(&conv_context) {
            run_conv_gemm_tile(&plan, &raw, output);
            #[cfg(feature = "instrument")]
            {
                // `PATH_CONV_TILE` (via `counters.commit` below) is this
                // path's own unambiguous gate-pass signal — deliberately NOT
                // `NEON_TILE_GATE_PASSES`, which `neon_tile_plan`'s own dot
                // tile already owns and a shared assertion in this file's
                // tests reads as "the dot tile fired".
                counters.kernel_calls += (plan.m_total as u64).div_ceil(TILE_ROWS as u64)
                    * (plan.n_total as u64).div_ceil(TILE_COLS as u64)
                    * plan.outer_extent;
                counters.mac_ops += plan.m_total as u64 * plan.n_total as u64 * reduction_total;
                counters.operand_loads += (plan.m_total as u64 + plan.n_total as u64) * reduction_total;
                counters.leading_iters += plan.m_total as u64;
                counters.output_writes += plan.m_total as u64 * plan.n_total as u64;
                let distinct_operand_elements: u64 = raw.iter().map(|buffer| buffer.len() as u64).sum();
                counters.commit(Path::ConvTile, distinct_operand_elements);
                instrument::record_reduce_path_ticks(Path::ConvTile, instrument::elapsed_ticks(commit_started));
            }
            return Ok(());
        }
    }

    // Resolved ONCE per bound op: an explicit-NEON 6x4 microkernel for the
    // exact GEMM shape `reduction_fast_path` already isolates. Ported from
    // ggml tinyBLAS's `gemm_bloc` — see `neon_tile_plan` and
    // `gemm_tile_neon` docs for the six-condition gate and why the
    // accumulator type (not the loop shape) is what makes it fit in
    // registers (`proxima-tensor/docs/discipline.md`, attempts 1 and 2).
    #[cfg(target_arch = "aarch64")]
    let tile_plan = if reduction_fast_path {
        neon_tile_plan(
            resolved,
            &shape,
            *reduce_op,
            !matches!(init, ReduceInit::FirstElement),
            &reduction_strides,
            &strides,
            leading_output_axes,
        )
    } else {
        None
    };
    #[cfg(target_arch = "aarch64")]
    let tiled_leading_rows = tile_plan.as_ref().map_or(0, |_| {
        if leading_total >= TILE_ROWS as u64 {
            leading_total - leading_total % TILE_ROWS as u64
        } else {
            0
        }
    });
    #[cfg(target_arch = "aarch64")]
    let tiled_width_cols = tile_plan.as_ref().map_or(0, |_| width - width % TILE_COLS);

    // Set to `tiled_leading_rows + rows_remaining` below when the
    // row-remainder tile pass runs, so the untiled loop after this block
    // starts past whatever rows the remainder pass already computed.
    #[cfg(target_arch = "aarch64")]
    let mut main_loop_start = tiled_leading_rows;

    // accumulated locally across the whole bound op and committed once
    // below, never as a per-element atomic inside the tiled loops — a
    // per-element atomic would perturb the throughput it exists to measure.
    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    let mut neon_tile_fallback_elements = 0u64;
    // `NEON_TILE_INVOCATIONS`/`NEON_TILE_ROW_REMAINDER_*` used to be a
    // `fetch_add(1)` per tile call — ~43,520 atomics per 1024^3 GEMM (one
    // per 6x4 output tile), the same per-call-in-a-hot-loop shape the
    // per-element counters this module's docs warn about. Tallied locally
    // and committed once per bound op instead, same as the fallback count
    // right above.
    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    let mut neon_tile_invocations = 0u64;
    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    let mut neon_tile_row_remainder_invocations = 0u64;
    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    let mut neon_tile_row_remainder_elements = 0u64;

    #[cfg(target_arch = "aarch64")]
    if let Some(plan) = &tile_plan {
        #[cfg(feature = "instrument")]
        NEON_TILE_GATE_PASSES.fetch_add(1, Ordering::Relaxed);
        let leading_axis = leading_output_axes[0] as usize;
        let out_stride = last_output_dim.map_or(0, |dim| out_layout.stride(dim));

        // Column-panel width: bound the inner sweep to a slice of `b` that
        // stays resident in L2 across the whole row-strip pass below,
        // instead of re-streaming all of `b` past L2 once per 6-row strip
        // (`neon_column_panel_cols`'s doc has the budget arithmetic).
        let panel_cols = neon_column_panel_cols(reduction_total, tiled_width_cols);

        let mut panel_start = 0usize;
        loop {
            let panel_end = (panel_start + panel_cols).min(tiled_width_cols);
            let mut leading_flat = 0u64;
            while leading_flat < tiled_leading_rows {
                unflatten_into(leading_flat, &leading_extents, &mut leading_coordinate);
                merge_coordinates_into(leading_output_axes, &leading_coordinate, &[], &[], &mut full_coordinate);
                full_coordinate[reduction_dims[0] as usize] = 0;
                if let Some(dim) = last_output_dim {
                    full_coordinate[dim as usize] = 0;
                }
                fill_running_offsets(resolved, &full_coordinate, &mut running);
                let base_a = running[plan.index_a] as usize;
                let base_b0 = running[plan.index_b] as usize;

                let mut out_prefixes = [0i64; TILE_ROWS];
                for (row, prefix) in out_prefixes.iter_mut().enumerate() {
                    full_coordinate[leading_axis] = leading_flat + row as u64;
                    *prefix = out_layout.offset_of(&full_coordinate);
                }

                let mut col = panel_start;
                while col < panel_end {
                    let base_b = base_b0 + col * plan.col_stride_b;
                    let mut tile_out = [[seed; TILE_COLS]; TILE_ROWS];
                    // `neon_tile_plan`'s gate already proved: no gathers, both
                    // contraction strides == 1, and `reduction_total` elements
                    // read contiguously from `base_a`/`base_b` on every row and
                    // column this tile visits, so every offset the kernel forms
                    // stays within the source slices.
                    unsafe {
                        gemm_tile_neon::<TILE_ROWS>(
                            KStridedTile {
                                data: raw[plan.index_a],
                                base: base_a as i64,
                                k_stride: plan.row_stride_a as i64,
                            },
                            KStridedTile {
                                data: raw[plan.index_b],
                                base: base_b as i64,
                                k_stride: plan.col_stride_b as i64,
                            },
                            reduction_total as usize,
                            &mut tile_out,
                        );
                    }
                    #[cfg(feature = "instrument")]
                    {
                        neon_tile_invocations += 1;
                        counters.kernel_calls += 1;
                        counters.mac_ops += (TILE_ROWS * TILE_COLS) as u64 * reduction_total;
                        counters.operand_loads += (TILE_ROWS + TILE_COLS) as u64 * reduction_total;
                    }
                    for (tile_row, &out_prefix) in tile_out.iter().zip(out_prefixes.iter()) {
                        for (column, &value) in tile_row.iter().enumerate() {
                            let position = out_prefix + out_stride * (col + column) as i64;
                            output[position as usize] = value;
                        }
                    }
                    col += TILE_COLS;
                }

                // the column tail (`width % TILE_COLS` leftover columns) only
                // needs computing once per row, not once per panel — run it
                // on whichever panel reaches the tiled boundary (exactly one
                // does, including the degenerate single-panel case).
                if panel_end == tiled_width_cols && tiled_width_cols < width {
                    let fold = DotFold {
                        len: reduction_total as usize,
                        init: seed,
                        seeded: true,
                    };
                    for (row, &out_prefix) in out_prefixes.iter().enumerate() {
                        full_coordinate[leading_axis] = leading_flat + row as u64;
                        if let Some(dim) = last_output_dim {
                            full_coordinate[dim as usize] = tiled_width_cols as u64;
                        }
                        fill_running_offsets(resolved, &full_coordinate, &mut running);
                        for n in tiled_width_cols..width {
                            let value = reduce_dot_fast(&shape, *reduce_op, &raw, &running, &reduction_strides, fold);
                            output[(out_prefix + out_stride * n as i64) as usize] = value;
                            #[cfg(feature = "instrument")]
                            {
                                neon_tile_fallback_elements += 1;
                                counters.kernel_calls += 1;
                                counters.mac_ops += reduction_total;
                                for &operand_stride in &reduction_strides {
                                    counters.operand_loads += if operand_stride == 1 { reduction_total } else { 1 };
                                }
                            }
                            for (offset, stride) in running.iter_mut().zip(&strides) {
                                *offset += stride;
                            }
                        }
                    }
                }

                #[cfg(feature = "instrument")]
                {
                    let mut writes = (TILE_ROWS * (panel_end - panel_start)) as u64;
                    if panel_end == tiled_width_cols && tiled_width_cols < width {
                        writes += (TILE_ROWS * (width - tiled_width_cols)) as u64;
                    }
                    counters.output_writes += writes;
                    if panel_start == 0 {
                        counters.leading_iters += TILE_ROWS as u64;
                    }
                }

                leading_flat += TILE_ROWS as u64;
            }
            if panel_end >= tiled_width_cols {
                break;
            }
            panel_start = panel_end;
        }
        // every panel-loop exit leaves the row-strip sweep at exactly
        // `tiled_leading_rows` (each panel processes the same full row
        // range); the remainder pass below picks up from there.
        let mut leading_flat = tiled_leading_rows;

        // Leftover rows after the 6-row main pass are always in `1..=5`
        // (`leading_total mod TILE_ROWS`, `TILE_ROWS == 6`). Every one of
        // those widths gets its own `gemm_tile_neon` instantiation — same
        // kernel body as the main loop above, monomorphised at the exact
        // leftover width instead of testing a single fixed threshold. Zero
        // leftover rows means the remainder pass is skipped entirely; there
        // is no scalar fallback left for any `M`.
        let rows_remaining = leading_total - tiled_leading_rows;

        // one instantiation per possible leftover width; body is identical
        // across widths so a macro avoids five hand-duplicated copies.
        macro_rules! row_remainder_tile {
            ($rows:literal) => {{
                let leading_axis = leading_output_axes[0] as usize;
                let out_stride = last_output_dim.map_or(0, |dim| out_layout.stride(dim));
                unflatten_into(leading_flat, &leading_extents, &mut leading_coordinate);
                merge_coordinates_into(leading_output_axes, &leading_coordinate, &[], &[], &mut full_coordinate);
                full_coordinate[reduction_dims[0] as usize] = 0;
                if let Some(dim) = last_output_dim {
                    full_coordinate[dim as usize] = 0;
                }
                fill_running_offsets(resolved, &full_coordinate, &mut running);
                let base_a = running[plan.index_a] as usize;
                let base_b0 = running[plan.index_b] as usize;

                let mut out_prefixes = [0i64; $rows];
                for (row, prefix) in out_prefixes.iter_mut().enumerate() {
                    full_coordinate[leading_axis] = leading_flat + row as u64;
                    *prefix = out_layout.offset_of(&full_coordinate);
                }

                let mut col = 0usize;
                while col < tiled_width_cols {
                    let base_b = base_b0 + col * plan.col_stride_b;
                    let mut tile_out = [[seed; TILE_COLS]; $rows];
                    // same preconditions `neon_tile_plan`'s gate already
                    // proved for the main 6-row pass; only the row count
                    // differs.
                    unsafe {
                        gemm_tile_neon::<$rows>(
                            KStridedTile {
                                data: raw[plan.index_a],
                                base: base_a as i64,
                                k_stride: plan.row_stride_a as i64,
                            },
                            KStridedTile {
                                data: raw[plan.index_b],
                                base: base_b as i64,
                                k_stride: plan.col_stride_b as i64,
                            },
                            reduction_total as usize,
                            &mut tile_out,
                        );
                    }
                    #[cfg(feature = "instrument")]
                    {
                        neon_tile_row_remainder_invocations += 1;
                        neon_tile_row_remainder_elements += ($rows * TILE_COLS) as u64;
                        counters.kernel_calls += 1;
                        counters.mac_ops += ($rows * TILE_COLS) as u64 * reduction_total;
                        counters.operand_loads += ($rows + TILE_COLS) as u64 * reduction_total;
                    }
                    for (tile_row, &out_prefix) in tile_out.iter().zip(out_prefixes.iter()) {
                        for (column, &value) in tile_row.iter().enumerate() {
                            let position = out_prefix + out_stride * (col + column) as i64;
                            output[position as usize] = value;
                        }
                    }
                    col += TILE_COLS;
                }

                if tiled_width_cols < width {
                    let fold = DotFold {
                        len: reduction_total as usize,
                        init: seed,
                        seeded: true,
                    };
                    for (row, &out_prefix) in out_prefixes.iter().enumerate() {
                        full_coordinate[leading_axis] = leading_flat + row as u64;
                        if let Some(dim) = last_output_dim {
                            full_coordinate[dim as usize] = tiled_width_cols as u64;
                        }
                        fill_running_offsets(resolved, &full_coordinate, &mut running);
                        for n in tiled_width_cols..width {
                            let value = reduce_dot_fast(&shape, *reduce_op, &raw, &running, &reduction_strides, fold);
                            output[(out_prefix + out_stride * n as i64) as usize] = value;
                            #[cfg(feature = "instrument")]
                            {
                                neon_tile_fallback_elements += 1;
                                counters.kernel_calls += 1;
                                counters.mac_ops += reduction_total;
                                for &operand_stride in &reduction_strides {
                                    counters.operand_loads += if operand_stride == 1 { reduction_total } else { 1 };
                                }
                            }
                            for (offset, stride) in running.iter_mut().zip(&strides) {
                                *offset += stride;
                            }
                        }
                    }
                }

                #[cfg(feature = "instrument")]
                {
                    counters.leading_iters += $rows as u64;
                    counters.output_writes += ($rows * width) as u64;
                }

                leading_flat += $rows as u64;
                main_loop_start = leading_flat;
            }};
        }

        match rows_remaining {
            0 => {}
            5 => row_remainder_tile!(5),
            4 => row_remainder_tile!(4),
            3 => row_remainder_tile!(3),
            2 => row_remainder_tile!(2),
            1 => row_remainder_tile!(1),
            _ => unreachable!("rows_remaining must be < TILE_ROWS (6) after the main tiled pass"),
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    let main_loop_start = 0u64;

    // guards both the loop AND the allocation below it: a bound op the
    // width-tile or NEON-tile path fully covers leaves `main_loop_start ==
    // leading_total` (measured: the 1024^3 contiguous GEMM never reaches
    // this branch at all), and this fallback loop's own accumulator —
    // hoisted once per bound op rather than once per output row, same
    // reasoning as `output`'s own hoist above — has nothing to do in that
    // case; allocating it anyway would pay for storage this loop then never
    // touches.
    if main_loop_start < leading_total {
        let mut accumulator = vec![seed; width];
        for leading_flat in main_loop_start..leading_total {
            unflatten_into(leading_flat, &leading_extents, &mut leading_coordinate);
            accumulator.fill(seed);
            let mut seeded = !matches!(init, ReduceInit::FirstElement);

            if reduction_fast_path {
                // Fold along `k` (contiguous on every operand read here) instead
                // of accumulating across the width dim `n` — one full contraction
                // per output position, in the same k=0..K sequential order the
                // generic loop below would visit, so results stay bit-identical.
                merge_coordinates_into(leading_output_axes, &leading_coordinate, &[], &[], &mut full_coordinate);
                full_coordinate[reduction_dims[0] as usize] = 0;
                if let Some(dim) = last_output_dim {
                    full_coordinate[dim as usize] = 0;
                }
                fill_running_offsets(resolved, &full_coordinate, &mut running);
                let fold = DotFold {
                    len: reduction_total as usize,
                    init: initial_value(*init).unwrap_or(0.0),
                    seeded,
                };
                for slot in &mut accumulator {
                    *slot = reduce_dot_fast(&shape, *reduce_op, &raw, &running, &reduction_strides, fold);
                    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
                    if tile_plan.is_some() {
                        neon_tile_fallback_elements += 1;
                    }
                    #[cfg(feature = "instrument")]
                    {
                        counters.kernel_calls += 1;
                        counters.mac_ops += reduction_total;
                        for &operand_stride in &reduction_strides {
                            counters.operand_loads += if operand_stride == 0 { 1 } else { reduction_total };
                        }
                    }
                    for (offset, stride) in running.iter_mut().zip(&strides) {
                        *offset += stride;
                    }
                }
            } else {
                for reduction_flat in 0..reduction_total {
                    unflatten_into(reduction_flat, &reduction_extents, &mut reduction_coordinate);
                    merge_coordinates_into(
                        leading_output_axes,
                        &leading_coordinate,
                        &reduction_dims,
                        &reduction_coordinate,
                        &mut full_coordinate,
                    );
                    fill_running_offsets(resolved, &full_coordinate, &mut running);

                    if fast_path {
                        reduce_width_fast(&shape, *reduce_op, &raw, &running, &strides, &mut accumulator, seeded);
                        #[cfg(feature = "instrument")]
                        {
                            counters.kernel_calls += 1;
                            counters.mac_ops += width as u64;
                            for &stride in &strides {
                                counters.operand_loads += if stride == 0 { 1 } else { width as u64 };
                            }
                        }
                        seeded = true;
                        continue;
                    }

                    fill_gather_cursors(
                        resolved,
                        buffers,
                        &full_coordinate,
                        last_output_dim,
                        &mut gather_cursors,
                    )?;

                    for slot in &mut accumulator {
                        for (index, data) in raw.iter().enumerate() {
                            let mut offset = running[index];
                            if let Some(cursor) = gather_cursors[index].as_mut() {
                                offset += cursor.fetch_and_advance(resolved.node)?;
                            }
                            operand_values[index] = data[offset as usize];
                            running[index] += strides[index];
                        }
                        let value = eval_body_shape(&shape, &operand_values, &mut step_values);
                        *slot = if seeded {
                            apply_scalar_op(*reduce_op, &[*slot, value])
                        } else {
                            value
                        };
                        #[cfg(feature = "instrument")]
                        {
                            counters.kernel_calls += 1;
                            counters.mac_ops += 1;
                            counters.operand_loads += raw.len() as u64;
                        }
                    }
                    seeded = true;
                }
            }

            merge_coordinates_into(leading_output_axes, &leading_coordinate, &[], &[], &mut full_coordinate);
            let out_prefix = out_layout.offset_of(&full_coordinate);
            let out_stride = last_output_dim.map_or(0, |dim| out_layout.stride(dim));
            for (slot, value) in accumulator.iter().enumerate() {
                output[(out_prefix + out_stride * slot as i64) as usize] = *value;
            }
            #[cfg(feature = "instrument")]
            {
                counters.leading_iters += 1;
                counters.output_writes += accumulator.len() as u64;
            }
        }
    }
    #[cfg(feature = "instrument")]
    {
        let distinct_operand_elements: u64 = raw.iter().map(|buffer| buffer.len() as u64).sum();
        counters.commit(path, distinct_operand_elements);
        instrument::record_reduce_path_ticks(path, instrument::elapsed_ticks(commit_started));
    }
    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    {
        NEON_TILE_FALLBACK_ELEMENTS.fetch_add(neon_tile_fallback_elements, Ordering::Relaxed);
        NEON_TILE_INVOCATIONS.fetch_add(neon_tile_invocations, Ordering::Relaxed);
        NEON_TILE_ROW_REMAINDER_INVOCATIONS.fetch_add(neon_tile_row_remainder_invocations, Ordering::Relaxed);
        NEON_TILE_ROW_REMAINDER_ELEMENTS.fetch_add(neon_tile_row_remainder_elements, Ordering::Relaxed);
        // computed once from `width`/`tiled_width_cols`, both already in
        // scope from earlier in this call — never re-checked per iteration.
        if tile_plan.is_some() && tiled_width_cols < width {
            counter!(instrument::NEON_TILE_COLUMN_TAIL_PRESENT, 1);
        }
    }
    Ok(())
}

fn run_scan<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        out_layout,
        ..
    } = &resolved.kind
    else {
        unreachable!("run_scan is only called for a Keep::Scan fold")
    };
    let raw = operand_buffers(resolved, buffers)?;
    let (outer_extents, inner_len) = split_innermost(&resolved.extents);
    let innermost_dim = outer_extents.len() as u16;
    let body = resolved.element_body();
    let shape = body_shape(body);
    let mut operand_values = vec![0.0f32; raw.len()];
    let mut step_values = vec![0.0f32; body.steps.len()];
    // loop-invariant: see the identical hoist in `run_elementwise`
    // (`proxima-tensor/docs/discipline.md` ROW 2).
    let strides: Vec<i64> = resolved
        .operands()
        .iter()
        .map(|(_, view, _)| view.stride(innermost_dim))
        .collect();
    let mut running: Vec<i64> = vec![0; raw.len()];
    let mut gather_cursors: Vec<Option<GatherCursor>> = (0..raw.len()).map(|_| None).collect();
    let mut outer_coordinate = vec![0u64; outer_extents.len()];

    let mut accumulator = initial_value(*init).unwrap_or(0.0);
    let mut seeded = !matches!(init, ReduceInit::FirstElement);

    // Same operand-side gate as `run_elementwise`/`run_reduce`, plus one
    // scan-specific condition: the fast path writes into a contiguous
    // `&mut [f32]` output slice, so it additionally requires the output's
    // own width-dim stride to be 1 (`proxima-tensor/docs/discipline.md` ROW 5).
    // A strided output (real but rarer) falls back to the per-element loop
    // unchanged, named rather than silently narrowed.
    let operand_fast_path = body_shape_is_affine_fast_path(resolved, &shape, &strides);

    for outer_flat in 0..odometer_len(outer_extents) {
        unflatten_into(outer_flat, outer_extents, &mut outer_coordinate);
        fill_running_offsets(resolved, &outer_coordinate, &mut running);
        let out_running = out_layout.offset_of(&outer_coordinate);
        let out_stride = out_layout.stride(innermost_dim);

        if operand_fast_path && out_stride == 1 {
            let out_base = out_running as usize;
            let out_slice = &mut output[out_base..out_base + inner_len];
            accumulator = scan_width_fast(
                &shape,
                *reduce_op,
                &raw,
                &running,
                &strides,
                out_slice,
                ScanState { seeded, accumulator },
            );
            seeded = true;
            continue;
        }

        fill_gather_cursors(
            resolved,
            buffers,
            &outer_coordinate,
            Some(innermost_dim),
            &mut gather_cursors,
        )?;
        let mut out_running = out_running;

        for _ in 0..inner_len {
            for (index, data) in raw.iter().enumerate() {
                let mut offset = running[index];
                if let Some(cursor) = gather_cursors[index].as_mut() {
                    offset += cursor.fetch_and_advance(resolved.node)?;
                }
                operand_values[index] = data[offset as usize];
                running[index] += strides[index];
            }
            let value = eval_body_shape(&shape, &operand_values, &mut step_values);
            accumulator = if seeded {
                apply_scalar_op(*reduce_op, &[accumulator, value])
            } else {
                value
            };
            seeded = true;
            output[out_running as usize] = accumulator;
            out_running += out_stride;
        }
    }
    Ok(())
}

/// A [`ComposedBody`] classified once per node, outside the per-element
/// loop, into the shape its evaluation actually needs. `Unary`/`Binary` are
/// the overwhelmingly common post-fusion case (a single [`ScalarOp`] over
/// one or two freshly-read operands — a bare elementwise op, or the product
/// step a `Reduce(Elementwise(Multiply))` fusion folds straight into the
/// accumulator) and skip `apply_body`'s per-element step loop and its
/// dynamic `StepArg` dispatch entirely. `Generic` is the fallback for a
/// deeper fused chain (multiple `BodyStep`s referencing earlier steps).
///
/// Classifying here — once, before any element is visited — is what lets
/// [`eval_body_shape`] avoid re-deciding "is this one step or several" on
/// every one of a node's iteration-space elements; profiling
/// (`proxima-tensor/docs/discipline.md` ROW 0) found that per-element redecision,
/// via an out-of-line `apply_body` call and its computed jump table, was
/// 51.9% of self-time on a 1024^3 GEMM.
enum BodyShape<'a> {
    Unary(ScalarOp, u16),
    Binary(ScalarOp, u16, u16),
    /// The bias-corrected Adam update chain (`docs/discipline.md` ROW 179),
    /// bias correction absorbed in-line for BOTH `m` and `v` (the actual
    /// fused shape `optimizer::adam_step` builds — `recip_bias1`/
    /// `recip_bias2` are each a live, single-consumer 4-step sub-chain
    /// (`step*ln(beta) -> exp -> 1-that -> reciprocal`), not pre-materialized
    /// scalar inputs the way an EARLIER version of this detector, and
    /// ROW 176's own simplified microbench, both assumed): 16 `BodyStep`s
    /// total, detected structurally by [`detect_adam_update_roles`] on op
    /// sequence + `StepArg` wiring — never on a node's own identity or name.
    /// Carries the source [`ComposedBody`] too, purely so
    /// [`eval_body_shape`]'s slow gather fallback can still walk it through
    /// [`apply_body`] exactly like [`Generic`](Self::Generic) does; the fast
    /// dedicated kernel ([`elementwise_width_fused_adam_update`]) never
    /// touches that field.
    FusedAdamUpdate(AdamUpdateRoles, &'a ComposedBody),
    Generic(&'a ComposedBody),
}

/// The eleven physical operand slots [`BodyShape::FusedAdamUpdate`] reads,
/// named by the role each plays in the Adam update math — `m`/`v`/`param`
/// are the three full-shape, unit-stride tensors; every other field is a
/// rank-0 broadcast scalar (`step_for_bias1`/`step_for_bias2` are the SAME
/// logical training-step value, read at two separate operand slots because
/// `bind::compose_operand` freshly resolves each occurrence rather than
/// deduplicating by `NodeId` — same for `one_for_bias1`/`one_for_bias2`,
/// both the literal `1.0`). Every field is a `StepArg::Operand` index into
/// the SAME `BoundOp::operands()` slice every other `BodyShape` variant
/// already indexes into (`Unary`/`Binary`'s own `u16` fields), not a new
/// addressing scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AdamUpdateRoles {
    param: u16,
    learning_rate: u16,
    m: u16,
    one_for_bias1: u16,
    step_for_bias1: u16,
    ln_beta1: u16,
    v: u16,
    one_for_bias2: u16,
    step_for_bias2: u16,
    ln_beta2: u16,
    epsilon: u16,
}

/// Structural detector for [`BodyShape::FusedAdamUpdate`] (`docs/discipline.md`
/// ROW 179): matches `body`'s own 16 [`BodyStep`]s against the exact op
/// sequence and `StepArg` wiring [`optimizer::adam_step`]'s fusion produces
/// — every check is on `step.op`/`step.args` shape alone, never on a
/// `NodeId`, a name, or which physical buffer a caller happens to bind to
/// an operand slot. `None` on any mismatch (wrong step count, wrong op at
/// any position, or a `StepArg` referencing the wrong earlier step/operand
/// role) falls through to [`BodyShape::Generic`] untouched — the same
/// conservative-precondition shape `window_copy_operand` (ROW 153/154) and
/// `StaticArena::static_nodes` (ROW 174/175) already use for a dedicated
/// fast path beside the general one.
fn detect_adam_update_roles(body: &ComposedBody) -> Option<AdamUpdateRoles> {
    let [step0, step1, step2, step3, step4, step5, step6, step7, step8, step9, step10, step11, step12, step13, step14, step15] =
        body.steps.as_slice()
    else {
        return None;
    };
    let (step_for_bias1, ln_beta1) = match (step0.op, step0.args.as_slice()) {
        (ScalarOp::Multiply, [StepArg::Operand(step_for_bias1), StepArg::Operand(ln_beta1)]) => {
            (*step_for_bias1, *ln_beta1)
        }
        _ => return None,
    };
    if !matches!((step1.op, step1.args.as_slice()), (ScalarOp::Exponential, [StepArg::Step(0)])) {
        return None;
    }
    let one_for_bias1 = match (step2.op, step2.args.as_slice()) {
        (ScalarOp::Subtract, [StepArg::Operand(one_for_bias1), StepArg::Step(1)]) => *one_for_bias1,
        _ => return None,
    };
    if !matches!((step3.op, step3.args.as_slice()), (ScalarOp::Reciprocal, [StepArg::Step(2)])) {
        return None;
    }
    let m = match (step4.op, step4.args.as_slice()) {
        (ScalarOp::Multiply, [StepArg::Operand(m), StepArg::Step(3)]) => *m,
        _ => return None,
    };
    let (step_for_bias2, ln_beta2) = match (step5.op, step5.args.as_slice()) {
        (ScalarOp::Multiply, [StepArg::Operand(step_for_bias2), StepArg::Operand(ln_beta2)]) => {
            (*step_for_bias2, *ln_beta2)
        }
        _ => return None,
    };
    if !matches!((step6.op, step6.args.as_slice()), (ScalarOp::Exponential, [StepArg::Step(5)])) {
        return None;
    }
    let one_for_bias2 = match (step7.op, step7.args.as_slice()) {
        (ScalarOp::Subtract, [StepArg::Operand(one_for_bias2), StepArg::Step(6)]) => *one_for_bias2,
        _ => return None,
    };
    if !matches!((step8.op, step8.args.as_slice()), (ScalarOp::Reciprocal, [StepArg::Step(7)])) {
        return None;
    }
    let v = match (step9.op, step9.args.as_slice()) {
        (ScalarOp::Multiply, [StepArg::Operand(v), StepArg::Step(8)]) => *v,
        _ => return None,
    };
    if !matches!((step10.op, step10.args.as_slice()), (ScalarOp::SquareRoot, [StepArg::Step(9)])) {
        return None;
    }
    let epsilon = match (step11.op, step11.args.as_slice()) {
        (ScalarOp::Add, [StepArg::Step(10), StepArg::Operand(epsilon)]) => *epsilon,
        _ => return None,
    };
    if !matches!((step12.op, step12.args.as_slice()), (ScalarOp::Reciprocal, [StepArg::Step(11)])) {
        return None;
    }
    if !matches!(
        (step13.op, step13.args.as_slice()),
        (ScalarOp::Multiply, [StepArg::Step(4), StepArg::Step(12)])
    ) {
        return None;
    }
    let learning_rate = match (step14.op, step14.args.as_slice()) {
        (ScalarOp::Multiply, [StepArg::Operand(learning_rate), StepArg::Step(13)]) => *learning_rate,
        _ => return None,
    };
    let param = match (step15.op, step15.args.as_slice()) {
        (ScalarOp::Subtract, [StepArg::Operand(param), StepArg::Step(14)]) => *param,
        _ => return None,
    };
    Some(AdamUpdateRoles {
        param,
        learning_rate,
        m,
        one_for_bias1,
        step_for_bias1,
        ln_beta1,
        v,
        one_for_bias2,
        step_for_bias2,
        ln_beta2,
        epsilon,
    })
}

fn body_shape(body: &ComposedBody) -> BodyShape<'_> {
    if let [step] = body.steps.as_slice() {
        match step.args.as_slice() {
            [StepArg::Operand(a)] => return BodyShape::Unary(step.op, *a),
            [StepArg::Operand(a), StepArg::Operand(b)] => {
                return BodyShape::Binary(step.op, *a, *b);
            }
            _ => {}
        }
    }
    if let Some(roles) = detect_adam_update_roles(body) {
        return BodyShape::FusedAdamUpdate(roles, body);
    }
    BodyShape::Generic(body)
}

/// Evaluates one iteration step against a pre-classified [`BodyShape`].
/// `#[inline(always)]` plus a `shape` that never changes across a node's own
/// loop nest is what lets LLVM hoist the shape match itself out of the hot
/// loop (loop-invariant code motion over a value that provably doesn't
/// change), rather than re-running it every element the way a direct
/// `apply_body` call forced.
#[inline(always)]
fn eval_body_shape(shape: &BodyShape, operand_values: &[f32], step_values: &mut [f32]) -> f32 {
    match *shape {
        BodyShape::Unary(op, a) => apply_scalar_op(op, &[operand_values[a as usize]]),
        BodyShape::Binary(op, a, b) => {
            apply_scalar_op(op, &[operand_values[a as usize], operand_values[b as usize]])
        }
        // The gather-fallback loop never reaches the dedicated kernel (that
        // requires the affine fast path -- `fused_adam_update_is_affine_fast_path`)
        // so a `FusedAdamUpdate` here just walks its own carried `ComposedBody`
        // exactly like `Generic`, bit-identical either way.
        BodyShape::FusedAdamUpdate(_, body) | BodyShape::Generic(body) => apply_body(body, operand_values, step_values),
    }
}

/// True when a physical operand at `index` is gather-free and has a
/// non-negative constant width-dim stride. Negative strides are rejected
/// because every [`OperandSpan`] is built with `stride as usize`, and a
/// negative value would wrap.
fn operand_is_affine(resolved: &BoundOp, strides: &[i64], index: u16) -> bool {
    let (_, _, gather) = &resolved.operands()[index as usize];
    gather.is_none() && strides[index as usize] >= 0
}

/// [`operand_is_affine`] narrowed to the strides [`reduce_width_fast`] and
/// `scan_width_fast` should actually take. Their strided arms are correct
/// for any stride, but correctness is not the question this gate answers.
/// Unlike [`run_elementwise`], whose alternative is the per-element
/// interpreter at 16.2 ns/element, a reduce that fails this gate falls
/// through to the contraction-dim dot path and its NEON tile — a faster
/// kernel, not a slower one. Admitting stride > 1 here stole those nodes
/// into a scalar width walk and measured `reduce_f32_dense` at 180.1 ms of
/// prefill against 81.0 ms for the same nodes on the dot path
/// (`proxima-tensor/docs/discipline.md` ROW 66). The two gates genuinely
/// disagree on which strides they accept, and the disagreement is a
/// measurement.
fn operand_is_unit_or_broadcast(resolved: &BoundOp, strides: &[i64], index: u16) -> bool {
    operand_is_affine(resolved, strides, index) && strides[index as usize] <= 1
}

/// True when every physical operand [`BodyShape`] actually reads (one for
/// `Unary`, up to two for `Binary` — `Generic` never qualifies here) has a
/// width-dim stride of 0 or 1 ([`operand_is_unit_or_broadcast`]). Checked
/// once per bound op, never per element — the same discipline [`body_shape`]
/// already applies to the op/arity decision. Shared by [`run_reduce`] and
/// [`run_scan`], whose own straight-line arms have no `Generic` case, so
/// `Generic` staying `false` here is load-bearing, not merely conservative.
/// [`run_elementwise`]'s own `Generic` fast path is a separate, WIDER gate:
/// [`generic_body_is_affine_fast_path`].
fn body_shape_is_affine_fast_path(resolved: &BoundOp, shape: &BodyShape, strides: &[i64]) -> bool {
    match *shape {
        BodyShape::Unary(_, a) => operand_is_unit_or_broadcast(resolved, strides, a),
        BodyShape::Binary(_, a, b) => {
            operand_is_unit_or_broadcast(resolved, strides, a) && operand_is_unit_or_broadcast(resolved, strides, b)
        }
        // A reduce/scan body is never fused with the Adam-chain shape in
        // this crate (it is a straight-line elementwise chain, not a
        // reduce's own per-step combine) -- treated exactly like `Generic`,
        // conservatively false, so `reduce_dot_fast`/`scan_width_fast` never
        // see this variant either.
        BodyShape::FusedAdamUpdate(..) | BodyShape::Generic(_) => false,
    }
}

/// The physical operand index of a window-materialize-shaped identity copy,
/// when `run_elementwise_range`'s own block sweep (`block_dim`/`block_extent`,
/// ROW 150) is engaged — `None` otherwise. `window_materialize`
/// (`proxima-onnx/src/lower.rs`) shapes its output `[n,c,oh,ow,kh,kw]`; once
/// ROW 147's identity-multiply elimination collapses the all-ones stamp
/// away, this op's body is exactly `BodyShape::Unary(ScalarOp::Identity, _)`
/// — a bare copy from a source read whose `kw` axis is already guaranteed
/// contiguous — checked explicitly here via `strides[operand] == 1`, NOT
/// inferred from `fast_path` alone: `fast_path`'s own gate
/// (`operand_is_unit_or_broadcast`) admits stride 0 (a genuine broadcast)
/// as well as stride 1, and `MaxPool`'s `Indices` machinery
/// (`proxima-onnx/src/lower.rs`'s `coordinate_image`) hits exactly that —
/// a `window_materialize` over a value that varies only along `kh`, not
/// `kw`, composing to `Unary(Identity, _)` with the operand's OWN `kw`
/// stride at 0, not 1 (found live by `maxpool_indices_row_major_...`
/// panicking `out of range for slice of length 4` before this check was
/// added, `docs/discipline.md` ROW 154). The `kh` axis (`block_dim`)
/// sits at a regular, arbitrary-sign stride, unconstrained here. The gate
/// stays deliberately narrow on SHAPE (`Unary(Identity, _)` plus a live
/// block plus a genuinely contiguous inner read), not on axis names or a
/// `window_materialize`-specific tag: `Layout::offset_of`'s exact linearity
/// (ROW 150's own proof) makes [`window_copy_block`] correct for ANY
/// operand whose body happens to match this shape, window-materialize or
/// not (`docs/discipline.md` ROW 153's own rung-2 charter).
fn window_copy_operand(shape: &BodyShape, fast_path: bool, block_extent: u64, strides: &[i64]) -> Option<u16> {
    match *shape {
        BodyShape::Unary(ScalarOp::Identity, operand)
            if fast_path && block_extent > 1 && strides[operand as usize] == 1 =>
        {
            Some(operand)
        }
        _ => None,
    }
}

/// [`window_copy_operand`]'s block: `block_extent` `inner_len`-wide rows,
/// contiguous within each row, each row offset from the previous by
/// `row_stride` (any sign/magnitude — matches the per-step block loop this
/// replaces, which places no non-negativity requirement on `block_strides`
/// unlike the inner-width `strides` array). Bypasses
/// [`elementwise_width_fast`]'s per-row `BodyShape`/`ScalarOp` dispatch and
/// [`OperandSpan`] construction entirely: the shape is already known
/// constant for the whole block, so nothing is left to branch on per row —
/// each row is a plain slice-to-slice copy. An `inner_len == 3` (mnist's
/// own `kw`) hand-unrolled scalar variant was tried and measured
/// indistinguishable-to-worse than this `copy_from_slice` loop on 3 of 4
/// benched shapes (one shape's apparent win did not survive a second
/// sample — outlier noise, not signal); kept this simpler single form
/// rather than carry a second, unproven-faster code path
/// (`docs/discipline.md` ROW 154).
#[inline(always)]
fn window_copy_block(source: &[f32], src_base: i64, row_stride: i64, block_extent: u64, inner_len: usize, out: &mut [f32]) {
    let mut base = src_base;
    let mut out_offset = 0usize;
    for _ in 0..block_extent {
        let start = base as usize;
        out[out_offset..out_offset + inner_len].copy_from_slice(&source[start..start + inner_len]);
        out_offset += inner_len;
        base += row_stride;
    }
}

/// [`run_elementwise`]'s own eligibility gate for its `Generic` fast path
/// ([`elementwise_width_generic`]): every `StepArg::Operand` any step in
/// `body` references must be gather-free with a non-negative constant
/// stride ([`operand_is_affine`]) — ANY constant stride, not only 0 or 1.
/// A stride-2 RoPE body (`specs/rope.toml`'s `s,2*i->si`) is what this
/// width exists for: it used to fail here and fall to the per-element
/// interpreter at 16.2 ns/element, against 2.2 ns/element on this path.
fn generic_body_is_affine_fast_path(resolved: &BoundOp, body: &ComposedBody, strides: &[i64]) -> bool {
    body.steps.iter().flat_map(|step| step.args.iter()).all(|arg| match arg {
        StepArg::Operand(index) => operand_is_affine(resolved, strides, *index),
        StepArg::Step(_) => true,
    })
}

/// [`run_elementwise`]'s eligibility gate for [`BodyShape::FusedAdamUpdate`]'s
/// dedicated kernel (`docs/discipline.md` ROW 179) — strictly NARROWER than
/// [`generic_body_is_affine_fast_path`] above (which admits any non-negative
/// constant stride): [`elementwise_width_fused_adam_update`] slices `m`/`v`/
/// `param` directly (`&raw[idx][base..base+width]`), so those three roles
/// must be exactly unit-stride (`strides[idx] == 1`), and reads every other
/// role as one hoisted scalar each, so those eight roles must be exactly
/// stride-0 (a genuine call-invariant broadcast, never a per-row-only
/// broadcast — the same distinction `axes_flat_chain`'s own doc, ROW 178,
/// already draws). Any role failing its own required stride (a caller
/// somehow binding a strided/gathered buffer to one of these eleven slots)
/// falls through to `BodyShape::Generic`'s existing tiled path untouched —
/// this gate, not [`detect_adam_update_roles`]'s structural match, is what
/// makes that fall-through safe.
fn fused_adam_update_is_affine_fast_path(resolved: &BoundOp, roles: AdamUpdateRoles, strides: &[i64]) -> bool {
    let is_unit_stride =
        |index: u16| operand_is_affine(resolved, strides, index) && strides[index as usize] == 1;
    let is_broadcast_scalar =
        |index: u16| operand_is_affine(resolved, strides, index) && strides[index as usize] == 0;
    is_unit_stride(roles.m)
        && is_unit_stride(roles.v)
        && is_unit_stride(roles.param)
        && is_broadcast_scalar(roles.learning_rate)
        && is_broadcast_scalar(roles.one_for_bias1)
        && is_broadcast_scalar(roles.step_for_bias1)
        && is_broadcast_scalar(roles.ln_beta1)
        && is_broadcast_scalar(roles.one_for_bias2)
        && is_broadcast_scalar(roles.step_for_bias2)
        && is_broadcast_scalar(roles.ln_beta2)
        && is_broadcast_scalar(roles.epsilon)
}

/// The width loop's straight-line fast path: reads each physical operand's
/// value for the whole width span at once (a contiguous `&[f32]` subslice
/// when its stride is 1, a single hoisted scalar read when its stride is 0),
/// with no `operand_values` scratch copy, no `gather_cursors` `Option`
/// check, and no per-element `running`/`strides` bookkeeping — `running`
/// gives each operand's width span its starting offset, and
/// [`body_shape_is_affine_fast_path`]'s precondition guarantees every
/// operand here has stride 0 or 1, so `stride == 1` is the only branch left
/// to make (once, not per element) between a slice read and a scalar
/// broadcast. Iterates `accumulator` in the same slot order the generic path
/// does, combining via the same `apply_scalar_op` calls in the same order,
/// so output is bit-identical (`proxima-tensor/docs/discipline.md` ROW 3).
/// One operand's width-span read shape for [`reduce_width_fast`]'s
/// straight-line arms: `stride == 1` reads `data[base..base+width]` as a real
/// subslice, `stride == 0` reads `data[base]` once and broadcasts it across
/// every position, and any other value walks `base, base + stride,
/// base + 2 * stride, ...` — [`operand_is_affine`] admits any non-negative
/// stride, so all three shapes reach here. A bare `contiguous: bool` used to
/// stand in for this field: it could only ever distinguish "stride 1 or not",
/// which made a stride-2 body (RoPE's `x[2*i]`/`x[2*i+1]` reads) unrepresentable
/// in every accelerated kernel and forced it onto the per-element interpreter
/// for good. Bundling the three fields keeps `reduce_width_binary` under
/// clippy's argument-count lint without reaching for `#[allow]`.
#[derive(Clone, Copy)]
struct OperandSpan<'a> {
    data: &'a [f32],
    base: usize,
    stride: usize,
}

impl OperandSpan<'_> {
    /// distinguishes a real walk from the stride-0/1 shapes the existing
    /// monomorphic arms already handle, so those arms stay untouched.
    #[inline(always)]
    fn is_strided(self) -> bool {
        self.stride > 1
    }

    /// collapses broadcast (`position * 0 == 0`) and contiguous
    /// (`position * 1 == position`) into the same expression as any other
    /// constant stride, so the strided fallback needs no separate broadcast arm.
    #[inline(always)]
    fn at(self, position: usize) -> f32 {
        self.data[self.base + position * self.stride]
    }
}

#[inline(always)]
fn reduce_width_fast(
    shape: &BodyShape,
    reduce_op: ScalarOp,
    raw: &[&[f32]],
    running: &[i64],
    strides: &[i64],
    accumulator: &mut [f32],
    seeded: bool,
) {
    let span_of = |index: u16| {
        let index = index as usize;
        OperandSpan {
            data: raw[index],
            base: running[index] as usize,
            stride: strides[index] as usize,
        }
    };
    match *shape {
        BodyShape::Unary(op, a) => {
            reduce_width_unary(op, reduce_op, span_of(a), accumulator, seeded);
        }
        BodyShape::Binary(op, a, b) => {
            reduce_width_binary(op, reduce_op, span_of(a), span_of(b), accumulator, seeded);
        }
        BodyShape::FusedAdamUpdate(..) | BodyShape::Generic(_) => {
            unreachable!("fast path is never entered for a Generic or FusedAdamUpdate body shape")
        }
    }
}

#[inline(always)]
fn combine_reduction(reduce_op: ScalarOp, previous: f32, value: f32, seeded: bool) -> f32 {
    if seeded {
        apply_scalar_op(reduce_op, &[previous, value])
    } else {
        value
    }
}

/// Dispatches once per call (never per element) on `op`, then on
/// `reduce_op` — but only when `reduce_op` is one of the four ops a fold
/// realistically combines with (`Add`/`Multiply`/`Maximum`/`Minimum`: sum,
/// product, max-pool, min-pool). Each of the 28 (7 unary op x 4 reduce op)
/// arms hands two concrete, non-capturing closures to
/// [`reduce_width_unary_monomorphic`] — a distinct generic instantiation
/// per pair, so the width loop inside contains the literal arithmetic
/// (`-a`, `a.sqrt()`, `acc.max(v)`, ...) inlined straight into the loop
/// body, with no runtime branch and no indirect call
/// (`proxima-tensor/docs/discipline.md` ROW 4). `seeded` is also resolved here,
/// not per element — [`reduce_width_unary_monomorphic`] branches on it
/// once, outside its loops, rather than once per element the way
/// [`combine_reduction`] used to. A `reduce_op` outside that set of four
/// (`Subtract`/`Divide`/`Greater`/`Equal` as a fold combiner — legal by
/// the type system since both have arity 2, not a real reduction any
/// current caller constructs) falls back to
/// [`reduce_width_unary_scalar_dispatch`], the ROW 3 implementation:
/// correct, not accelerated, named rather than silently narrowed away.
fn reduce_width_unary(op: ScalarOp, reduce_op: ScalarOp, span: OperandSpan, accumulator: &mut [f32], seeded: bool) {
    macro_rules! unary_op_arm {
        ($f:expr) => {
            match reduce_op {
                ScalarOp::Add => reduce_width_unary_monomorphic($f, |acc: f32, v: f32| acc + v, span, accumulator, seeded),
                ScalarOp::Multiply => {
                    reduce_width_unary_monomorphic($f, |acc: f32, v: f32| acc * v, span, accumulator, seeded)
                }
                ScalarOp::Maximum => {
                    reduce_width_unary_monomorphic($f, |acc: f32, v: f32| acc.max(v), span, accumulator, seeded)
                }
                ScalarOp::Minimum => {
                    reduce_width_unary_monomorphic($f, |acc: f32, v: f32| acc.min(v), span, accumulator, seeded)
                }
                _ => reduce_width_unary_scalar_dispatch(op, reduce_op, span, accumulator, seeded),
            }
        };
    }
    match op {
        ScalarOp::Identity => unary_op_arm!(|a: f32| a),
        ScalarOp::Negate => unary_op_arm!(|a: f32| -a),
        ScalarOp::Reciprocal => unary_op_arm!(|a: f32| 1.0 / a),
        ScalarOp::Exponential => unary_op_arm!(|a: f32| a.exp()),
        ScalarOp::Logarithm => unary_op_arm!(|a: f32| a.ln()),
        ScalarOp::SquareRoot => unary_op_arm!(|a: f32| a.sqrt()),
        ScalarOp::Tanh => unary_op_arm!(|a: f32| a.tanh()),
        _ => reduce_width_unary_scalar_dispatch(op, reduce_op, span, accumulator, seeded),
    }
}

/// One monomorphized instantiation per (op, reduce_op) pair `reduce_width_unary`
/// selects. `seeded` is branched on ONCE, outside both loops (not per
/// element) — the loop bodies below each contain exactly one call to `op`
/// and, in the seeded case, one call to `reduce`, both of which are
/// non-capturing closures the compiler inlines directly into the loop,
/// leaving a single concrete arithmetic operation per element. A strided
/// span (stride > 1) delegates to [`reduce_width_unary_monomorphic_strided`]
/// before either arm below runs, so the stride-0/stride-1 arms here never
/// see anything but the two shapes they were always tuned for.
#[inline(always)]
fn reduce_width_unary_monomorphic<F, R>(op: F, reduce: R, span: OperandSpan, accumulator: &mut [f32], seeded: bool)
where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if span.is_strided() {
        return reduce_width_unary_monomorphic_strided(op, reduce, span, accumulator, seeded);
    }
    if span.stride == 1 {
        let slice = &span.data[span.base..span.base + accumulator.len()];
        if seeded {
            for (slot, &raw_value) in accumulator.iter_mut().zip(slice) {
                *slot = reduce(*slot, op(raw_value));
            }
        } else {
            for (slot, &raw_value) in accumulator.iter_mut().zip(slice) {
                *slot = op(raw_value);
            }
        }
    } else {
        let value = op(span.data[span.base]);
        if seeded {
            for slot in accumulator.iter_mut() {
                *slot = reduce(*slot, value);
            }
        } else {
            for slot in accumulator.iter_mut() {
                *slot = value;
            }
        }
    }
}

/// Mirrors the stride-1 arm of [`reduce_width_unary_monomorphic`] one
/// position at a time via [`OperandSpan::at`] instead of a contiguous slice
/// read, so a stride > 1 body folds in the exact same left-to-right order as
/// the stride-1 case — never routed through a reassociating multi-accumulator
/// fold, which would silently change output for this newly-widened case.
#[inline(always)]
fn reduce_width_unary_monomorphic_strided<F, R>(op: F, reduce: R, span: OperandSpan, accumulator: &mut [f32], seeded: bool)
where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if seeded {
        for (position, slot) in accumulator.iter_mut().enumerate() {
            *slot = reduce(*slot, op(span.at(position)));
        }
    } else {
        for (position, slot) in accumulator.iter_mut().enumerate() {
            *slot = op(span.at(position));
        }
    }
}

/// The pre-ROW-4 (ROW 3) implementation, kept as the fallback for a
/// `reduce_op` outside {Add, Multiply, Maximum, Minimum} — same numerical
/// result as [`reduce_width_unary_monomorphic`], dispatched per element via
/// [`apply_scalar_op`]/[`combine_reduction`] instead of an inlined closure.
/// [`OperandSpan::at`] already generalizes over every stride, so this needs
/// no separate strided sibling — one loop over positions covers stride 0, 1,
/// and any wider constant stride alike.
fn reduce_width_unary_scalar_dispatch(
    op: ScalarOp,
    reduce_op: ScalarOp,
    span: OperandSpan,
    accumulator: &mut [f32],
    seeded: bool,
) {
    for (position, slot) in accumulator.iter_mut().enumerate() {
        let value = apply_scalar_op(op, &[span.at(position)]);
        *slot = combine_reduction(reduce_op, *slot, value, seeded);
    }
}

/// Same discipline as [`reduce_width_unary`], for the two-operand case: 8
/// binary-arity body ops x 4 accelerated reduce ops = 32 monomorphized
/// instantiations of [`reduce_width_binary_monomorphic`], selected by one
/// nested match evaluated once per call. A `reduce_op` outside the
/// accelerated four falls back to [`reduce_width_binary_scalar_dispatch`].
fn reduce_width_binary(
    op: ScalarOp,
    reduce_op: ScalarOp,
    a: OperandSpan,
    b: OperandSpan,
    accumulator: &mut [f32],
    seeded: bool,
) {
    // the width-accumulating twin of `reduce_dot_binary`'s multiply-add arm.
    // For a `[k,n]`-laid-out matmul this is the inner loop: `a` is one scalar
    // at the current `(m, k)`, `b` is a contiguous row of `n` — an axpy, and
    // the single densest multiply-accumulate in the crate. `!a.is_strided()
    // && !b.is_strided()` keeps a real stride (e.g. 2) out of this block
    // explicitly — its own `(false, false)` arm below would otherwise read a
    // strided operand once and silently treat it as a broadcast.
    if FUSED_MULTIPLY_ADD
        && seeded
        && matches!((op, reduce_op), (ScalarOp::Multiply, ScalarOp::Add))
        && !a.is_strided()
        && !b.is_strided()
    {
        let width = accumulator.len();
        match (a.stride == 1, b.stride == 1) {
            (true, true) => {
                let slice_a = &a.data[a.base..a.base + width];
                let slice_b = &b.data[b.base..b.base + width];
                for ((slot, &value_a), &value_b) in accumulator.iter_mut().zip(slice_a).zip(slice_b) {
                    *slot = value_a.mul_add(value_b, *slot);
                }
                return;
            }
            (true, false) => {
                let slice_a = &a.data[a.base..a.base + width];
                let value_b = b.data[b.base];
                for (slot, &value_a) in accumulator.iter_mut().zip(slice_a) {
                    *slot = value_a.mul_add(value_b, *slot);
                }
                return;
            }
            (false, true) => {
                let value_a = a.data[a.base];
                let slice_b = &b.data[b.base..b.base + width];
                for (slot, &value_b) in accumulator.iter_mut().zip(slice_b) {
                    *slot = value_a.mul_add(value_b, *slot);
                }
                return;
            }
            (false, false) => {}
        }
    }
    macro_rules! binary_op_arm {
        ($f:expr) => {
            match reduce_op {
                ScalarOp::Add => {
                    reduce_width_binary_monomorphic($f, |acc: f32, v: f32| acc + v, a, b, accumulator, seeded)
                }
                ScalarOp::Multiply => {
                    reduce_width_binary_monomorphic($f, |acc: f32, v: f32| acc * v, a, b, accumulator, seeded)
                }
                ScalarOp::Maximum => {
                    reduce_width_binary_monomorphic($f, |acc: f32, v: f32| acc.max(v), a, b, accumulator, seeded)
                }
                ScalarOp::Minimum => {
                    reduce_width_binary_monomorphic($f, |acc: f32, v: f32| acc.min(v), a, b, accumulator, seeded)
                }
                _ => reduce_width_binary_scalar_dispatch(op, reduce_op, a, b, accumulator, seeded),
            }
        };
    }
    match op {
        ScalarOp::Add => binary_op_arm!(|x: f32, y: f32| x + y),
        ScalarOp::Subtract => binary_op_arm!(|x: f32, y: f32| x - y),
        ScalarOp::Multiply => binary_op_arm!(|x: f32, y: f32| x * y),
        ScalarOp::Divide => binary_op_arm!(|x: f32, y: f32| x / y),
        ScalarOp::Maximum => binary_op_arm!(|x: f32, y: f32| x.max(y)),
        ScalarOp::Minimum => binary_op_arm!(|x: f32, y: f32| x.min(y)),
        ScalarOp::Greater => binary_op_arm!(|x: f32, y: f32| f32::from(u8::from(x > y))),
        ScalarOp::Equal => binary_op_arm!(|x: f32, y: f32| f32::from(u8::from((x - y).abs() == 0.0))),
        _ => reduce_width_binary_scalar_dispatch(op, reduce_op, a, b, accumulator, seeded),
    }
}

#[inline(always)]
fn reduce_width_binary_monomorphic<F, R>(
    op: F,
    reduce: R,
    a: OperandSpan,
    b: OperandSpan,
    accumulator: &mut [f32],
    seeded: bool,
) where
    F: Fn(f32, f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if a.is_strided() || b.is_strided() {
        return reduce_width_binary_monomorphic_strided(op, reduce, a, b, accumulator, seeded);
    }
    let width = accumulator.len();
    match (a.stride == 1, b.stride == 1) {
        (true, true) => {
            let slice_a = &a.data[a.base..a.base + width];
            let slice_b = &b.data[b.base..b.base + width];
            if seeded {
                for ((slot, &value_a), &value_b) in accumulator.iter_mut().zip(slice_a).zip(slice_b) {
                    *slot = reduce(*slot, op(value_a, value_b));
                }
            } else {
                for ((slot, &value_a), &value_b) in accumulator.iter_mut().zip(slice_a).zip(slice_b) {
                    *slot = op(value_a, value_b);
                }
            }
        }
        (true, false) => {
            let slice_a = &a.data[a.base..a.base + width];
            let value_b = b.data[b.base];
            if seeded {
                for (slot, &value_a) in accumulator.iter_mut().zip(slice_a) {
                    *slot = reduce(*slot, op(value_a, value_b));
                }
            } else {
                for (slot, &value_a) in accumulator.iter_mut().zip(slice_a) {
                    *slot = op(value_a, value_b);
                }
            }
        }
        (false, true) => {
            let value_a = a.data[a.base];
            let slice_b = &b.data[b.base..b.base + width];
            if seeded {
                for (slot, &value_b) in accumulator.iter_mut().zip(slice_b) {
                    *slot = reduce(*slot, op(value_a, value_b));
                }
            } else {
                for (slot, &value_b) in accumulator.iter_mut().zip(slice_b) {
                    *slot = op(value_a, value_b);
                }
            }
        }
        (false, false) => {
            let value_a = a.data[a.base];
            let value_b = b.data[b.base];
            let value = op(value_a, value_b);
            if seeded {
                for slot in accumulator.iter_mut() {
                    *slot = reduce(*slot, value);
                }
            } else {
                for slot in accumulator.iter_mut() {
                    *slot = value;
                }
            }
        }
    }
}

/// Mirrors [`reduce_width_binary_monomorphic`]'s fold order one position at a
/// time via [`OperandSpan::at`], for the case at least one of `a`/`b` has a
/// stride > 1 — never reassociated, so output stays bit-identical to what the
/// scalar interpreter would produce for the same body.
#[inline(always)]
fn reduce_width_binary_monomorphic_strided<F, R>(
    op: F,
    reduce: R,
    a: OperandSpan,
    b: OperandSpan,
    accumulator: &mut [f32],
    seeded: bool,
) where
    F: Fn(f32, f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if seeded {
        for (position, slot) in accumulator.iter_mut().enumerate() {
            *slot = reduce(*slot, op(a.at(position), b.at(position)));
        }
    } else {
        for (position, slot) in accumulator.iter_mut().enumerate() {
            *slot = op(a.at(position), b.at(position));
        }
    }
}

/// The pre-ROW-4 (ROW 3) implementation, kept as the fallback for a
/// `reduce_op` outside {Add, Multiply, Maximum, Minimum}. [`OperandSpan::at`]
/// already generalizes over every stride, so one loop over positions covers
/// stride 0, 1, and any wider constant stride alike.
fn reduce_width_binary_scalar_dispatch(
    op: ScalarOp,
    reduce_op: ScalarOp,
    a: OperandSpan,
    b: OperandSpan,
    accumulator: &mut [f32],
    seeded: bool,
) {
    for (position, slot) in accumulator.iter_mut().enumerate() {
        let value = apply_scalar_op(op, &[a.at(position), b.at(position)]);
        *slot = combine_reduction(reduce_op, *slot, value, seeded);
    }
}

// ---- width-dim register-tile GEMM kernel (aarch64) ----
//
// `reduce_width_binary`'s FMA axpy path above still round-trips
// `accumulator` through memory every reduction step (load-fma-store per
// width chunk), because `accumulator` is a `&mut [f32]` slice spanning the
// whole output width and can never fit in registers. This tile instead
// keeps a small (rows x columns) block of partial sums in NEON registers
// for the WHOLE `k` reduction, spilling nothing until the block is fully
// accumulated — the same register-budget technique that took the sibling
// dot-path tile from 0.122s to 0.028s at 1024^3.
//
// Applicable to exactly the shape `run_reduce`'s width path already
// specializes for: `out[m, n] += a[m, k] * b[k, n]`, `a` invariant across
// `n` (width stride 0), `b` contiguous along `n` (width stride 1), a
// single leading dim, a single contraction dim.

/// Output rows one call to [`gemm_width_tile_neon`] computes.
#[cfg(target_arch = "aarch64")]
use crate::sized::WIDTH_TILE_ROWS;

/// `float32x4_t` vectors of output columns one call to [`gemm_width_tile_neon`]
/// computes — 4 gives `WIDTH_TILE_ROWS * WIDTH_TILE_VECS` = 16 independent
/// accumulators, the measured saturation point for this core's NEON FMA
/// throughput.
#[cfg(target_arch = "aarch64")]
use crate::sized::WIDTH_TILE_VECS;

/// Pass/invocation/fallback-element counts for the width tile — mandatory
/// verification, not a runtime feature: a caller (`profile_hot`) reads
/// these after a run to prove the tile actually fired, since a silently-zero
/// invocation count would make any timing number meaningless. Mirrors
/// [`NEON_TILE_GATE_PASSES`]'s family exactly, including the row-tail and
/// column-tail coverage the two loops in [`run_width_tile_neon`] below both
/// account for: `invocations * (WIDTH_TILE_ROWS * WIDTH_TILE_VECS * 4) +
/// fallback_elements == leading_total * width` for any shape, not only
/// multiples of the tile.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
static WIDTH_TILE_GATE_PASSES: AtomicU64 = AtomicU64::new(0);
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
static WIDTH_TILE_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
static WIDTH_TILE_FALLBACK_ELEMENTS: AtomicU64 = AtomicU64::new(0);

/// Snapshot of the three `WIDTH_TILE_GATE_PASSES`-family counters:
/// (gate passes, tile invocations, fallback elements) — the width tile's
/// counterpart to [`neon_tile_counters`].
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub fn width_tile_counters() -> (u64, u64, u64) {
    (
        WIDTH_TILE_GATE_PASSES.load(Ordering::Relaxed),
        WIDTH_TILE_INVOCATIONS.load(Ordering::Relaxed),
        WIDTH_TILE_FALLBACK_ELEMENTS.load(Ordering::Relaxed),
    )
}

/// Everything [`try_run_width_tile`] needs, bundled the same way
/// [`OperandSpan`] and [`DotFold`] are — keeps the entry point under
/// clippy's argument-count lint instead of reaching for `#[allow]`.
/// `cfg`-gated to aarch64: the width tile path is compiled out entirely on
/// every other target, so there is nothing left to hand this to.
#[cfg(target_arch = "aarch64")]
struct WidthPathContext<'a> {
    resolved: &'a BoundOp,
    shape: &'a BodyShape<'a>,
    strides: &'a [i64],
    reduce_op: ScalarOp,
    init: ReduceInit,
    leading_output_axes: &'a [u16],
    reduction_dims: &'a [u16],
    last_output_dim: Option<u16>,
    width: usize,
    out_layout: &'a bind::Layout,
}

/// Everything the tiled loop needs to walk, resolved once per bound op.
#[cfg(target_arch = "aarch64")]
struct WidthTilePlan {
    a_operand: usize,
    b_operand: usize,
    row_stride_a: i64,
    base_a: i64,
    k_stride_a: i64,
    base_b: i64,
    k_stride_b: i64,
    out_base: i64,
    out_row_stride: i64,
    out_col_stride: i64,
    leading_total: usize,
    reduction_total: usize,
    width: usize,
    seed: f32,
}

/// Resolves [`WidthTilePlan`] once per bound op, or `None` when this node
/// does not match the shape the tile is built for. Every condition here is
/// checked once, never per element.
#[cfg(target_arch = "aarch64")]
fn width_tile_plan(context: &WidthPathContext) -> Option<WidthTilePlan> {
    if !FUSED_MULTIPLY_ADD || context.reduce_op != ScalarOp::Add {
        return None;
    }
    let (operand_a, operand_b) = match *context.shape {
        BodyShape::Binary(ScalarOp::Multiply, operand_a, operand_b) => (operand_a, operand_b),
        _ => return None,
    };
    if matches!(context.init, ReduceInit::FirstElement) {
        return None;
    }
    if context.leading_output_axes.len() != 1 || context.reduction_dims.len() != 1 {
        return None;
    }
    if context.width < WIDTH_TILE_VECS * 4 {
        return None;
    }
    let last_output_dim = context.last_output_dim?;

    let operands = context.resolved.operands();
    let (_, layout_a_raw, gather_a) = &operands[operand_a as usize];
    let (_, layout_b_raw, gather_b) = &operands[operand_b as usize];
    if gather_a.is_some() || gather_b.is_some() {
        return None;
    }
    let (a_operand, layout_a, b_operand, layout_b) =
        match (context.strides[operand_a as usize], context.strides[operand_b as usize]) {
            (0, 1) => (operand_a as usize, layout_a_raw, operand_b as usize, layout_b_raw),
            (1, 0) => (operand_b as usize, layout_b_raw, operand_a as usize, layout_a_raw),
            _ => return None,
        };

    let leading_dim = context.leading_output_axes[0];
    let reduction_dim = context.reduction_dims[0];

    Some(WidthTilePlan {
        a_operand,
        b_operand,
        row_stride_a: layout_a.stride(leading_dim),
        base_a: layout_a.base,
        k_stride_a: layout_a.stride(reduction_dim),
        base_b: layout_b.base,
        k_stride_b: layout_b.stride(reduction_dim),
        out_base: context.out_layout.base,
        out_row_stride: context.out_layout.stride(leading_dim),
        out_col_stride: context.out_layout.stride(last_output_dim),
        leading_total: context.resolved.extents[leading_dim as usize] as usize,
        reduction_total: context.resolved.extents[reduction_dim as usize] as usize,
        width: context.width,
        seed: initial_value(context.init).unwrap_or(0.0),
    })
}

/// A flat `f32` operand's addressing bundle: `data` is the physical buffer,
/// `base` the flat offset of this tile's `(row 0, col 0, k 0)` corner (dot
/// path) or `(row, k)` corner (width path), `k_stride` the per-row (dot `a`),
/// per-column (dot `b`), or per-reduction-step (width path, both operands)
/// step between adjacent lanes the kernel reads — whichever axis that
/// caller's kernel actually steps by.
///
/// `i64`, not `usize`: `neon_tile_plan` proves the dot path's strides
/// non-negative, but `width_tile_plan`'s can run negative, so the one type
/// serving both carries the wider constraint. The casts this costs
/// `gemm_tile_neon` are free on aarch64 — both widths are one register, and
/// the kernels emit zero `sxtw`/`uxtw` either way.
///
/// A per-row stride is a parameter, never a field: only `gemm_width_tile_neon`
/// steps rows independently, and a field would make every other caller supply
/// a value its kernel does not read.
///
/// Bundled for the same argument-count-lint reason [`OperandSpan`]/[`DotFold`]
/// already document.
#[cfg(target_arch = "aarch64")]
struct KStridedTile<'a> {
    data: &'a [f32],
    base: i64,
    k_stride: i64,
}

/// The register-tile microkernel: `WIDTH_TILE_ROWS` output rows x
/// `WIDTH_TILE_VECS` `float32x4_t` vectors of output columns, folded over
/// the whole `k` reduction with `acc` living in registers throughout — the
/// vector *type*, not a plain `[f32; 4]` array, is the entire trick: a
/// plain array forces LLVM to put it in memory and spill. `out` already
/// holds the seed value on entry (`vaddq_f32` below folds `acc` into it,
/// rather than overwriting), so a caller may reuse this for a running total
/// if that shape is ever needed.
///
/// # Safety
/// Caller guarantees every offset `a.base + i*a_row_stride + step*a.k_stride`
/// for `i in 0..WIDTH_TILE_ROWS, step in 0..k` lies within `a.data`, and
/// every offset `b.base + step*b.k_stride + v*4 + lane` for `v in
/// 0..WIDTH_TILE_VECS, lane in 0..4` lies within `b.data`.
#[cfg(target_arch = "aarch64")]
unsafe fn gemm_width_tile_neon(
    a: KStridedTile,
    a_row_stride: i64,
    b: KStridedTile,
    k: usize,
    out: &mut [[f32; WIDTH_TILE_VECS * 4]; WIDTH_TILE_ROWS],
) {
    // caller-checked: every (row, step, vec) offset below is in bounds.
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); WIDTH_TILE_VECS]; WIDTH_TILE_ROWS];
        for step in 0..k {
            let step = step as i64;
            let mut bv = [vdupq_n_f32(0.0); WIDTH_TILE_VECS];
            for (v, lane) in bv.iter_mut().enumerate() {
                let offset = b.base + step * b.k_stride + v as i64 * 4;
                *lane = vld1q_f32(b.data.as_ptr().add(offset as usize));
            }
            for (i, row_acc) in acc.iter_mut().enumerate() {
                let offset = a.base + i as i64 * a_row_stride + step * a.k_stride;
                let value_a = *a.data.get_unchecked(offset as usize);
                for (slot, &vector_b) in row_acc.iter_mut().zip(&bv) {
                    *slot = vfmaq_n_f32(*slot, vector_b, value_a);
                }
            }
        }
        for (i, row_acc) in acc.iter().enumerate() {
            for (v, &value) in row_acc.iter().enumerate() {
                let combined = vaddq_f32(vld1q_f32(out[i].as_ptr().add(v * 4)), value);
                vst1q_f32(out[i].as_mut_ptr().add(v * 4), combined);
            }
        }
    }
}

/// A single (row, column) partial sum computed the scalar way — the
/// remainder path for a leading count or width not divisible by the tile
/// shape. Correctness-only: `profile_hot`'s 1024^3 GEMM divides evenly by
/// both `WIDTH_TILE_ROWS` and `WIDTH_TILE_VECS * 4`, so this never fires
/// there (`WIDTH_TILE_FALLBACK_ELEMENTS` proves it), but a caller with an
/// arbitrary shape still gets a correct answer.
#[cfg(target_arch = "aarch64")]
fn width_tile_scalar_cell(a: KStridedTile, b: KStridedTile, k: usize, seed: f32) -> f32 {
    let mut acc = seed;
    let mut offset_a = a.base;
    let mut offset_b = b.base;
    for _ in 0..k {
        acc = a.data[offset_a as usize].mul_add(b.data[offset_b as usize], acc);
        offset_a += a.k_stride;
        offset_b += b.k_stride;
    }
    acc
}

/// Walks the full leading x width space in `WIDTH_TILE_ROWS x
/// (WIDTH_TILE_VECS * 4)` blocks via [`gemm_width_tile_neon`], falling back
/// to [`width_tile_scalar_cell`] for any leftover rows or columns that do
/// not fill a whole tile. Both remainder loops below increment
/// [`WIDTH_TILE_FALLBACK_ELEMENTS`] once per scalar element they compute —
/// the column tail inside the row-tile loop, and the row tail below it —
/// so `invocations * tile_cols * WIDTH_TILE_ROWS + fallback_elements` always
/// equals `leading_total * width`, for any shape, not only multiples of
/// the tile (mirrors [`NEON_TILE_FALLBACK_ELEMENTS`]'s own row+column
/// accounting for the dot-path tile).
#[cfg(target_arch = "aarch64")]
fn run_width_tile_neon(plan: &WidthTilePlan, raw: &[&[f32]], output: &mut [f32]) {
    #[cfg(feature = "instrument")]
    WIDTH_TILE_GATE_PASSES.fetch_add(1, Ordering::Relaxed);

    // accumulated locally across the whole tile walk and committed once at
    // the end, never as a per-element atomic inside the fallback loops.
    #[cfg(feature = "instrument")]
    let mut width_tile_fallback_elements = 0u64;
    // was a `fetch_add(1)` per tile call (`row_tiles * col_tiles` times,
    // the same magnitude as `NEON_TILE_INVOCATIONS`'s historical
    // per-tile atomic) — tallied locally and committed once instead.
    #[cfg(feature = "instrument")]
    let mut width_tile_invocations = 0u64;

    let data_a = raw[plan.a_operand];
    let data_b = raw[plan.b_operand];
    let tile_cols = WIDTH_TILE_VECS * 4;
    let row_tiles = plan.leading_total / WIDTH_TILE_ROWS;
    let col_tiles = plan.width / tile_cols;

    for row_tile in 0..row_tiles {
        let row_start = row_tile * WIDTH_TILE_ROWS;
        let base_a = plan.base_a + row_start as i64 * plan.row_stride_a;
        let out_row_prefix = plan.out_base + row_start as i64 * plan.out_row_stride;

        for col_tile in 0..col_tiles {
            let col_start = col_tile * tile_cols;
            let base_b = plan.base_b + col_start as i64;
            let mut tile_out = [[plan.seed; WIDTH_TILE_VECS * 4]; WIDTH_TILE_ROWS];

            // caller-checked: `base_a`/`base_b` plus every stride-scaled
            // offset the kernel touches stay inside `data_a`/`data_b`,
            // guaranteed by `row_tiles`/`col_tiles` only covering whole
            // tiles carved out of `plan.leading_total`/`plan.width`.
            unsafe {
                gemm_width_tile_neon(
                    KStridedTile { data: data_a, base: base_a, k_stride: plan.k_stride_a },
                    plan.row_stride_a,
                    KStridedTile { data: data_b, base: base_b, k_stride: plan.k_stride_b },
                    plan.reduction_total,
                    &mut tile_out,
                );
            }
            #[cfg(feature = "instrument")]
            {
                width_tile_invocations += 1;
            }

            for (i, row) in tile_out.iter().enumerate() {
                let row_prefix = out_row_prefix + i as i64 * plan.out_row_stride;
                for (v, &value) in row.iter().enumerate() {
                    let position = row_prefix + (col_start + v) as i64 * plan.out_col_stride;
                    output[position as usize] = value;
                }
            }
        }

        // column tail for these `WIDTH_TILE_ROWS` rows: columns past the
        // last full tile, still inside a tiled row-block.
        for col in col_tiles * tile_cols..plan.width {
            for i in 0..WIDTH_TILE_ROWS {
                let row = row_start + i;
                let value = width_tile_scalar_cell(
                    KStridedTile {
                        data: data_a,
                        base: plan.base_a + row as i64 * plan.row_stride_a,
                        k_stride: plan.k_stride_a,
                    },
                    KStridedTile { data: data_b, base: plan.base_b + col as i64, k_stride: plan.k_stride_b },
                    plan.reduction_total,
                    plan.seed,
                );
                let position = plan.out_base + row as i64 * plan.out_row_stride + col as i64 * plan.out_col_stride;
                output[position as usize] = value;
                #[cfg(feature = "instrument")]
                {
                    width_tile_fallback_elements += 1;
                }
            }
        }
    }

    // row tail: leading rows past the last full row-tile, every column —
    // these never touch the tiled loop above at all, so every element here
    // (including what would otherwise be a "tiled" column) is fallback.
    for row in row_tiles * WIDTH_TILE_ROWS..plan.leading_total {
        for col in 0..plan.width {
            let value = width_tile_scalar_cell(
                KStridedTile {
                    data: data_a,
                    base: plan.base_a + row as i64 * plan.row_stride_a,
                    k_stride: plan.k_stride_a,
                },
                KStridedTile { data: data_b, base: plan.base_b + col as i64, k_stride: plan.k_stride_b },
                plan.reduction_total,
                plan.seed,
            );
            let position = plan.out_base + row as i64 * plan.out_row_stride + col as i64 * plan.out_col_stride;
            output[position as usize] = value;
            #[cfg(feature = "instrument")]
            {
                width_tile_fallback_elements += 1;
            }
        }
    }

    #[cfg(feature = "instrument")]
    {
        WIDTH_TILE_FALLBACK_ELEMENTS.fetch_add(width_tile_fallback_elements, Ordering::Relaxed);
        WIDTH_TILE_INVOCATIONS.fetch_add(width_tile_invocations, Ordering::Relaxed);
        // computed once from `plan.width`/`tile_cols`, both already in
        // scope — never re-checked per iteration.
        if col_tiles * tile_cols < plan.width {
            counter!(instrument::WIDTH_TILE_COLUMN_TAIL_PRESENT, 1);
        }
    }
}

/// `run_reduce`'s single entry point into the width tile: resolves the gate
/// once, runs the whole node through [`run_width_tile_neon`] and reports
/// `true` when it applies, or leaves `output` untouched and reports `false`
/// so the caller falls back to the existing per-element width path
/// unchanged. `run_reduce`'s call site is itself aarch64-gated — every other
/// target keeps the per-element width path only, this function never exists
/// there.
#[cfg(target_arch = "aarch64")]
fn try_run_width_tile(context: &WidthPathContext, raw: &[&[f32]], output: &mut [f32]) -> bool {
    match width_tile_plan(context) {
        Some(plan) => {
            run_width_tile_neon(&plan, raw, output);
            true
        }
        None => false,
    }
}

/// The reduction-dim fast path's fold state, bundled for the same reason
/// [`OperandSpan`] and `ScanState` are: keeps `reduce_dot_binary` under
/// clippy's argument-count lint. `len` is the contraction length (`k`'s
/// extent), `init` is the reduction's identity/seed value, `seeded` mirrors
/// [`run_reduce`]'s own `seeded` flag (whether `init` should be combined
/// into the first term or overwritten by it, per [`ReduceInit::FirstElement`]).
#[derive(Clone, Copy)]
struct DotFold {
    len: usize,
    init: f32,
    seeded: bool,
}

/// Partial-accumulator count for the contiguous dot fold
/// ([`dot_fold_multi_accumulator_binary`]/[`dot_fold_multi_accumulator_unary`]).
/// The strict left-to-right fold (`acc = reduce(acc, op(a, b))` once per
/// `k`) is a serial dependency chain LLVM cannot widen, because float
/// `+`/`*` are not associative under IEEE 754 — reordering the sum
/// changes its bit pattern (`proxima-tensor/docs/discipline.md` ROW 11).
/// Splitting the chain into `DOT_LANES` independent partial folds (one
/// per position in a `DOT_LANES`-wide `chunks_exact` block) breaks that
/// dependency: each lane's own chain is still strictly sequential (still
/// no per-lane reassociation), but the lanes run independently, so LLVM
/// can pack the common case into vector `fmul`/`fadd` and pay the
/// horizontal combine once per call instead of once per element —
/// exactly what every BLAS and ggml itself do. 4 and 8 were measured
/// head-to-head (ROW 12, `proxima-tensor/docs/discipline.md`): 8 measured
/// consistently faster (~0.337-0.349s vs ~0.352-0.354s, 1024^3
/// transposed-RHS GEMM, 5 runs each) — more independent lanes hide more
/// of the reduce's latency on this core's issue width. 8 was kept.
use crate::sized::DOT_LANES;

/// Whether the target issues a fused multiply-add as one instruction.
/// aarch64 carries `fmla` in the base ISA; x86-64 needs FMA3. Without it
/// `f32::mul_add` becomes a libm call and is far slower than the two-op
/// form, so the specialization below must not fire.
///
/// A structural axis, not a tunable: it belongs in the build-time profile
/// alongside lane width and unroll factor once the microkernel axes land.
const FUSED_MULTIPLY_ADD: bool = cfg!(target_arch = "aarch64") || cfg!(target_feature = "fma");

/// `DOT_LANES` independent partial accumulators folded with `f32::mul_add`
/// — the multiply-accumulate specialization of
/// [`dot_fold_multi_accumulator_binary`].
///
/// Rust never contracts `a * b + c` into an FMA on its own, and that is a
/// guarantee rather than a missed optimization: contraction rounds once
/// instead of twice, so it changes the result and is not a rewrite the
/// optimiser may make unasked. Measured on the pre-change binary at 1024^3:
/// `fmla.4s` = 0, `fmul.4s` = 467, `fadd.4s` = 458 — the loop vectorised
/// and then issued two instructions per multiply-accumulate. `mul_add` is
/// the explicit request.
///
/// Numerically this moves *toward* the infinitely-precise result (one
/// rounding per term, not two), so it stays inside the 1e-5 relative
/// tolerance ROW 12 already established for this fold.
#[inline(always)]
fn dot_fold_fused_multiply_add(slice_a: &[f32], slice_b: &[f32], fold: DotFold) -> f32 {
    let (chunks_a, remainder_a) = slice_a.as_chunks::<DOT_LANES>();
    let (chunks_b, remainder_b) = slice_b.as_chunks::<DOT_LANES>();
    let mut lanes = [0.0f32; DOT_LANES];
    for (chunk_a, chunk_b) in chunks_a.iter().zip(chunks_b) {
        for ((lane, &value_a), &value_b) in lanes.iter_mut().zip(chunk_a).zip(chunk_b) {
            *lane = value_a.mul_add(value_b, *lane);
        }
    }
    let mut acc = fold.init;
    for &lane in &lanes {
        acc += lane;
    }
    for (&value_a, &value_b) in remainder_a.iter().zip(remainder_b) {
        acc = value_a.mul_add(value_b, acc);
    }
    acc
}

/// Output rows/columns computed per call of [`gemm_tile_neon`] — ggml
/// tinyBLAS's `RM`/`RN`. Vector width (4) is implied by `float32x4_t`. An
/// iso-accumulator shape sweep at 1024^3, single-thread (CoV 0.2-0.44%, 7
/// launches each) measured 6x4 at 86.5 GFLOPS against 4x6 at 49.9 and 3x8 at
/// 48.8 — the row-heavy orientation beats its own transpose by 73% despite
/// identical accumulator count and loads/MAC. Why orientation dominates is
/// still unexplained.
#[cfg(target_arch = "aarch64")]
use crate::sized::TILE_ROWS;
#[cfg(target_arch = "aarch64")]
use crate::sized::TILE_COLS;

/// Bytes of L2 budgeted for a resident `b` column panel in the tiled GEMM
/// pass below. M1 Max: 12 MiB shared L2 per performance cluster of 4 cores —
/// about 3 MiB/core once every worker in the cluster streams its own panel,
/// not 12 MiB as an 8 MiB budget implicitly assumed (one worker owning the
/// whole cluster's L2). ggml's own combined panel footprint never exceeds
/// ~2.5 MiB at any size or thread count, which is also where headroom for
/// the row-strip's `a` tile, the output tile in flight, and set-associativity
/// conflicts remains without the near-fit turning into a thrash.
///
/// Swept 8/4/3/2.5/2 MiB at 512/1024/2048^3, 1/2/4/8 threads, n=9,
/// interleaved round-robin per budget, 2026-08-18, system load 1.8-3.4
/// (mostly under 3.0, one late 8-thread cell drifted to 3.37). Only the
/// 1-thread cells stayed under the 1.5% CoV resolvability bar; every
/// 2+-thread cell exceeded it (up to 20% CoV, this session's shared-host
/// contention) and is not usable for a budget comparison. Within the
/// resolvable 1-thread cells: 512^3 and 1024^3 measured flat across every
/// budget from 8 MiB down to 2 MiB (busy-per-MAC within ~1% of each other,
/// GFLOPS parity vs ggml 89.57-90.17 for 1024^3 across 8/2.5 MiB, no
/// resolvable win despite the panel becoming numerically "active" at
/// 1024^3 below ~2.8 MiB) — the hypothesis that a lower budget would help
/// 1024^3 did NOT hold up. 2048^3/1-thread did show a real, resolvable
/// effect: busy-per-mac dropped ~1.7-2% for every budget at or below 4 MiB
/// versus the 8 MiB control (0.02238 -> ~0.0220), and GFLOPS parity vs ggml
/// rose from 0.999x to 1.026x at 2.5 MiB. 4/3/2.5/2 MiB were statistically
/// indistinguishable from each other at 2048^3/1-thread (within ~0.5%, same
/// order as the noise floor) — no single value in that range measured best.
/// 2.5 MiB is landed here because it matches ggml's own measured combined
/// footprint and never measured worse than the 8 MiB control in any
/// resolvable cell; 4 MiB or 3 MiB would be an equally defensible pick on
/// this data. checksums (135.87619/260.24106/513.10425) and the 1024^3
/// allocation shape were unchanged across every budget tested.
#[cfg(target_arch = "aarch64")]
use crate::sized::NEON_COLUMN_PANEL_BUDGET_BYTES;

/// Column-panel width for the tiled GEMM pass: the widest multiple of
/// `TILE_COLS` whose panel of `b` (`panel_cols` columns, each a contiguous
/// run of `reduction_len` `f32`s along the contraction dim) fits inside
/// [`NEON_COLUMN_PANEL_BUDGET_BYTES`]. At `reduction_len = 2048` (2048^3's
/// `k`): `2.5 MiB / (2048 * 4 bytes) = 640 -> 640` columns (rounds to a
/// `TILE_COLS` multiple exactly), five-plus panels across 2048's tiled
/// width — the cell this budget measurably helps. At `reduction_len = 1024`:
/// `2.5 MiB / 4096 bytes = 640` columns against a 1024-wide tiled output,
/// so the panel loop is numerically active (two panels, not the pre-2026-08
/// no-op) but measured flat against every other budget swept, 1-thread,
/// n=9 (`NEON_COLUMN_PANEL_BUDGET_BYTES`'s doc has the full sweep). At
/// `reduction_len = 512` the budget covers 1280 columns, wider than any
/// tiled width a 512^3 call produces, so the `clamp` below still collapses
/// to one panel spanning `tiled_width_cols` — an unconditional no-op there
/// at every budget from 8 MiB down to 2 MiB.
#[cfg(target_arch = "aarch64")]
fn neon_column_panel_cols(reduction_len: u64, tiled_width_cols: usize) -> usize {
    let bytes_per_col = reduction_len as usize * 4;
    let budget_cols = NEON_COLUMN_PANEL_BUDGET_BYTES
        .checked_div(bytes_per_col)
        .unwrap_or(tiled_width_cols);
    let rounded = budget_cols - budget_cols % TILE_COLS;
    rounded.clamp(TILE_COLS, tiled_width_cols.max(TILE_COLS))
}

/// One bound op's applicability gate for [`gemm_tile_neon`], resolved once
/// before [`run_reduce`]'s leading-dimension loop rather than per tile. The
/// six conditions mirror attempt 2's (`proxima-tensor/docs/discipline.md`):
/// FMA available, seeded, the fused body is `Multiply` reduced by `Add`,
/// both operands gather-free, both contraction-dim strides `== 1`, and
/// exactly one operand's width-dim stride is `0` (that one is `a`, whose
/// leading-axis stride becomes `row_stride_a`) while the other is nonzero
/// (that one is `b`, whose width-dim stride becomes `col_stride_b`).
#[cfg(target_arch = "aarch64")]
struct NeonTilePlan {
    index_a: usize,
    index_b: usize,
    row_stride_a: usize,
    col_stride_b: usize,
}

/// Runtime evidence the tile path actually ran, not just compiled: how many
/// bound ops passed [`neon_tile_plan`]'s gate, how many times
/// [`gemm_tile_neon`] was called, and how many output elements fell through
/// to the per-slot [`reduce_dot_fast`] remainder instead (row/column
/// leftovers past the last full `TILE_ROWS`x`TILE_COLS` block). Plain
/// process-wide counters, not a telemetry event, because the only consumer
/// is `profile_hot`'s one-shot report.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
static NEON_TILE_GATE_PASSES: AtomicU64 = AtomicU64::new(0);
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
static NEON_TILE_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
static NEON_TILE_FALLBACK_ELEMENTS: AtomicU64 = AtomicU64::new(0);
/// Row-remainder tile invocations (any width `1..=5`), tracked apart from
/// [`NEON_TILE_INVOCATIONS`] since remainder tiles compute a different,
/// width-dependent number of outputs per call than the fixed-24 main tile.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
static NEON_TILE_ROW_REMAINDER_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
/// Output elements actually covered by row-remainder tiles, summed across
/// every width `1..=5` a run may exercise — `rows * TILE_COLS` added per
/// invocation. Unlike [`NEON_TILE_ROW_REMAINDER_INVOCATIONS`], which just
/// counts calls, this is directly usable in the coverage identity
/// (`main_invocations * 24 + row_remainder_elements + fallback == m*n`)
/// without knowing which width(s) fired.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
static NEON_TILE_ROW_REMAINDER_ELEMENTS: AtomicU64 = AtomicU64::new(0);

/// Snapshot of the three `NEON_TILE_GATE_PASSES`-family counters for the
/// main 6x4 tile: (gate passes, tile invocations, fallback elements).
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub fn neon_tile_counters() -> (u64, u64, u64) {
    (
        NEON_TILE_GATE_PASSES.load(Ordering::Relaxed),
        NEON_TILE_INVOCATIONS.load(Ordering::Relaxed),
        NEON_TILE_FALLBACK_ELEMENTS.load(Ordering::Relaxed),
    )
}

/// `NEON_TILE_ROW_REMAINDER_INVOCATIONS` snapshot — the row-remainder
/// tiles' own invocation count (any width `1..=5`), separate from the main
/// 6x4 tile's.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub fn neon_tile_row_remainder_invocations() -> u64 {
    NEON_TILE_ROW_REMAINDER_INVOCATIONS.load(Ordering::Relaxed)
}

/// `NEON_TILE_ROW_REMAINDER_ELEMENTS` snapshot — output elements covered
/// by row-remainder tiles of any width, for the `main*24 + row_remainder +
/// fallback == m*n` coverage identity.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub fn neon_tile_row_remainder_elements() -> u64 {
    NEON_TILE_ROW_REMAINDER_ELEMENTS.load(Ordering::Relaxed)
}

#[cfg(target_arch = "aarch64")]
fn neon_tile_plan(
    resolved: &BoundOp,
    shape: &BodyShape,
    reduce_op: ScalarOp,
    seeded_always: bool,
    reduction_strides: &[i64],
    strides: &[i64],
    leading_output_axes: &[u16],
) -> Option<NeonTilePlan> {
    if !FUSED_MULTIPLY_ADD || !seeded_always || leading_output_axes.len() != 1 || reduce_op != ScalarOp::Add {
        return None;
    }
    let BodyShape::Binary(op, a, b) = *shape else {
        return None;
    };
    if op != ScalarOp::Multiply {
        return None;
    }
    let index_a = a as usize;
    let index_b = b as usize;
    let (_, _, gather_a) = &resolved.operands()[index_a];
    let (_, _, gather_b) = &resolved.operands()[index_b];
    if gather_a.is_some() || gather_b.is_some() {
        return None;
    }
    if reduction_strides[index_a] != 1 || reduction_strides[index_b] != 1 {
        return None;
    }
    let (index_a, index_b) = match (strides[index_a], strides[index_b]) {
        (0, other) if other != 0 => (index_a, index_b),
        (other, 0) if other != 0 => (index_b, index_a),
        _ => return None,
    };
    let row_stride_a = resolved.operands()[index_a].1.stride(leading_output_axes[0]);
    if row_stride_a < 0 {
        return None;
    }
    Some(NeonTilePlan {
        index_a,
        index_b,
        row_stride_a: row_stride_a as usize,
        col_stride_b: strides[index_b] as usize,
    })
}

/// Everything [`conv_gemm_tile_plan`] needs to decide whether `Conv`'s own
/// disjoint-leading-axis reduce shape (`docs/discipline.md` ROW 148/149)
/// qualifies for the blocked 2D GEMM tile — the same field set
/// [`WidthPathContext`] bundles for its own gate, one context type per tile
/// kind rather than a single context threading fields only some tiles read.
#[cfg(target_arch = "aarch64")]
struct ConvGemmContext<'a> {
    resolved: &'a BoundOp,
    shape: &'a BodyShape<'a>,
    reduce_op: ScalarOp,
    init: ReduceInit,
    leading_output_axes: &'a [u16],
    reduction_dims: &'a [u16],
    last_output_dim: Option<u16>,
    out_layout: &'a bind::Layout,
}

/// Everything [`run_conv_gemm_tile`]'s loop nest needs to walk, resolved
/// once per bound op. `index_m`/`index_n` name the two operands by their
/// role (`M` = the weight-shaped operand that owns exactly one leading axis
/// and nothing else; `N` = the windowed-shaped operand that owns every
/// other leading axis plus the width axis), not by which body-step slot
/// they started in — `conv_gemm_tile_plan` tries both assignments and picks
/// whichever satisfies the shape.
#[cfg(target_arch = "aarch64")]
struct ConvGemmTilePlan {
    index_m: usize,
    index_n: usize,
    base_m: i64,
    row_stride_m: i64,
    outer_stride_m: i64,
    base_n: i64,
    col_stride_n: i64,
    outer_stride_n: i64,
    outer_extent: u64,
    inner_span: usize,
    out_base: i64,
    out_row_stride: i64,
    m_total: usize,
    n_total: usize,
    seed: f32,
}

/// Row-major contiguity chain over `axes` (given outer-to-inner, walked
/// innermost-first via `.rev()`, the same convention
/// [`max_flat_reduction_suffix_len`] uses) — `Some(total_elements)` when
/// `view` addresses the WHOLE combined axis space as one contiguous
/// stride-`unit` span, `None` at the first break. An axis whose own extent
/// is `<= 1` is skipped rather than checked: its coordinate is always 0, so
/// its physical stride can never affect an address and is not informative
/// about contiguity — `Conv`'s own `n` axis (always extent 1 for every
/// shape this initiative measured) would otherwise spuriously break the
/// chain purely because `windowed`'s declared `[n,c,oh,ow,kh,kw]` layout
/// (`docs/discipline.md` ROW 148) puts a real, un-skipped `c` between `n`
/// and `oh`/`ow` — a genuine break for `n>1`, which this skip rule
/// correctly still catches (a real, nonzero-extent axis out of chain order
/// fails the `stride(dim) != expected` check exactly as before). Pure
/// stride/extent arithmetic over already-resolved [`bind::Layout`] data, no
/// NEON intrinsics — genuinely architecture-generic despite living beside
/// the `aarch64`-only [`conv_gemm_tile_plan`] that introduced it; NOT
/// `#[cfg(target_arch = "aarch64")]` because [`elementwise_rows_are_flat`]
/// (`docs/discipline.md` ROW 178) reuses it verbatim on every target.
fn axes_flat_chain(resolved: &BoundOp, axes: &[u16], view: &bind::Layout, unit: i64) -> Option<u64> {
    let mut expected = unit;
    let mut count: u64 = 1;
    for &axis in axes.iter().rev() {
        let extent = resolved.extents[axis as usize];
        if extent <= 1 {
            continue;
        }
        if view.stride(axis) != expected {
            return None;
        }
        expected = expected.saturating_mul(extent as i64);
        count = count.saturating_mul(extent);
    }
    Some(count)
}

/// [`run_elementwise_range`]'s own row-flattening precondition (ROW 178):
/// true when EVERY physical operand's width-dim address, walked across the
/// WHOLE `outer_axes` odometer, composes as one contiguous
/// stride-`strides[operand]` span — [`axes_flat_chain`] reused verbatim per
/// operand with `unit = strides[operand] * inner_len`, the address one
/// outer step away landing exactly one row (`inner_len` elements, at that
/// SAME per-element stride) past the current row's own span. A stride-0
/// operand collapses `unit` to 0, which `axes_flat_chain` already treats as
/// "every nonzero-extent axis in the chain must ALSO be stride 0" — the
/// identical predicate covers a genuinely global broadcast scalar with no
/// special case. `resolved.operands()` (every physical operand, not only
/// the ones a `Generic` body's steps actually reference) is checked for
/// simplicity: an operand the body never reads cannot make this call
/// INCORRECT by being conservatively included, only cost this one
/// optimization opportunity on a pathological unread-but-non-flat operand.
fn elementwise_rows_are_flat(resolved: &BoundOp, outer_axes: &[u16], strides: &[i64], inner_len: usize) -> bool {
    resolved.operands().iter().enumerate().all(|(index, (_, view, _))| {
        let unit = strides[index].saturating_mul(inner_len as i64);
        axes_flat_chain(resolved, outer_axes, view, unit).is_some()
    })
}

/// Resolves [`ConvGemmTilePlan`] once per bound op, or `None` when this node
/// does not match the shape the tile is built for. `Conv`'s own shape
/// (`docs/discipline.md` ROW 148/149): `leading_output_axes` splits into one
/// axis the `M` operand (weight) alone varies over and everything else the
/// `N` operand (`windowed`) alone varies over, and `reduction_dims` splits
/// into exactly one outer axis (`ci`) neither operand can flatten away and a
/// trailing inner span (`ky,kx`) both operands read contiguously — the
/// "blocked (2-level: outer x contiguous-inner)" shape ROW 148 named as the
/// real next step and left unattempted pending this generalization.
#[cfg(target_arch = "aarch64")]
fn conv_gemm_tile_plan(context: &ConvGemmContext) -> Option<ConvGemmTilePlan> {
    if !FUSED_MULTIPLY_ADD || context.reduce_op != ScalarOp::Add || matches!(context.init, ReduceInit::FirstElement) {
        return None;
    }
    let BodyShape::Binary(op, operand_a, operand_b) = *context.shape else {
        return None;
    };
    if op != ScalarOp::Multiply {
        return None;
    }
    // `leading_output_axes.len() == 1` is exactly the shape `neon_tile_plan`
    // already claims (both operands share one row axis); this tile exists
    // for the case that gate can never reach.
    if context.leading_output_axes.len() < 2 || context.reduction_dims.len() < 2 {
        return None;
    }
    let resolved = context.resolved;
    let operands = resolved.operands();
    let index_a = operand_a as usize;
    let index_b = operand_b as usize;
    let (_, view_a, gather_a) = &operands[index_a];
    let (_, view_b, gather_b) = &operands[index_b];
    if gather_a.is_some() || gather_b.is_some() {
        return None;
    }

    let suffix_a = max_flat_reduction_suffix_len(resolved, context.reduction_dims, index_a);
    let suffix_b = max_flat_reduction_suffix_len(resolved, context.reduction_dims, index_b);
    let inner_len = suffix_a.min(suffix_b);
    // `inner_len == reduction_dims.len()` is `reduction_is_fully_flat` for
    // BOTH operands at once — already `reduction_fast_path`'s own case, and
    // this function is only ever tried when that gate declined.
    if inner_len == 0 || inner_len >= context.reduction_dims.len() {
        return None;
    }
    let split = context.reduction_dims.len() - inner_len;
    let outer_dims = &context.reduction_dims[..split];
    // exactly one outer axis (`ci`) is the provably-safe case ROW 148/149
    // measured against every one of `Conv`'s 3 real folds; a wider outer
    // block is a genuinely different, larger piece of surgery (a nested
    // odometer instead of one flat `for outer_idx in 0..outer_extent` loop)
    // left for a future row rather than guessed at here.
    if outer_dims.len() != 1 {
        return None;
    }
    let outer_dim = outer_dims[0];
    let inner_dims = &context.reduction_dims[split..];
    let inner_span: u64 = inner_dims.iter().map(|&dim| resolved.extents[dim as usize]).product();
    let inner_span_i64 = i64::try_from(inner_span).ok()?;

    try_conv_gemm_assignment(context, index_a, view_a, index_b, view_b, outer_dim, inner_span_i64)
        .or_else(|| try_conv_gemm_assignment(context, index_b, view_b, index_a, view_a, outer_dim, inner_span_i64))
}

/// One candidate `(M operand, N operand)` assignment for
/// [`conv_gemm_tile_plan`] — tried once per operand ordering, since the
/// fused body's own step order (`windowed * weight` vs `weight * windowed`)
/// is not guaranteed and this tile cares about roles, not slot positions.
#[cfg(target_arch = "aarch64")]
fn try_conv_gemm_assignment(
    context: &ConvGemmContext,
    index_m: usize,
    view_m: &bind::Layout,
    index_n: usize,
    view_n: &bind::Layout,
    outer_dim: u16,
    inner_span: i64,
) -> Option<ConvGemmTilePlan> {
    let resolved = context.resolved;
    let mut m_axis = None;
    for &axis in context.leading_output_axes {
        if resolved.extents[axis as usize] <= 1 {
            continue;
        }
        match (view_m.stride(axis), view_n.stride(axis)) {
            (stride_m, 0) if stride_m != 0 => {
                if m_axis.is_some() {
                    // more than one axis the M operand alone owns: outside
                    // this tile's single-row-axis scope.
                    return None;
                }
                m_axis = Some(axis);
            }
            (0, _) => {}
            _ => return None, // shared or N-owned axis the M operand also varies over
        }
    }
    let m_axis = m_axis?;
    let row_stride_m = view_m.stride(m_axis);
    if row_stride_m < 0 {
        return None;
    }
    if let Some(width_dim) = context.last_output_dim
        && resolved.extents[width_dim as usize] > 1
        && view_m.stride(width_dim) != 0
    {
        return None;
    }

    // built once per bound op (this function runs once per `Keep::Reduce`
    // fold, never per element), the same setup-path cost `reduction_strides`
    // already pays in `run_reduce`.
    let n_axes: Vec<u16> = context
        .leading_output_axes
        .iter()
        .copied()
        .filter(|&axis| axis != m_axis)
        .chain(context.last_output_dim)
        .collect();
    if n_axes.is_empty() {
        return None;
    }
    let n_total_n = axes_flat_chain(resolved, &n_axes, view_n, inner_span)?;
    let n_total_out = axes_flat_chain(resolved, &n_axes, context.out_layout, 1)?;
    if n_total_n != n_total_out {
        return None;
    }

    let outer_stride_m = view_m.stride(outer_dim);
    let outer_stride_n = view_n.stride(outer_dim);
    if outer_stride_m < 0 || outer_stride_n < 0 {
        return None;
    }

    Some(ConvGemmTilePlan {
        index_m,
        index_n,
        base_m: view_m.base,
        row_stride_m,
        outer_stride_m,
        base_n: view_n.base,
        col_stride_n: inner_span,
        outer_stride_n,
        outer_extent: resolved.extents[outer_dim as usize],
        inner_span: inner_span as usize,
        out_base: context.out_layout.base,
        out_row_stride: context.out_layout.stride(m_axis),
        m_total: resolved.extents[m_axis as usize] as usize,
        n_total: n_total_n as usize,
        seed: initial_value(context.init).unwrap_or(0.0),
    })
}

/// Runs the whole bound op `plan` describes: an `M x N` output tile, each
/// cell folded over `outer_extent` blocks of `inner_span` contiguous
/// elements — `Conv`'s own `sum over ci of (weight[co,ci,:,:] . windowed[n,
/// ci, oy, ox, :, :])`. Reuses [`gemm_tile_neon`] entirely unchanged, called
/// once per `(M tile, N tile, outer step)` into the SAME `tile_out`
/// register array: that kernel already reads its `out` parameter's existing
/// value and adds to it (`gemm_tile_neon`'s own doc), so accumulating across
/// `outer_extent` ci-blocks needs no new kernel body, only a caller that
/// seeds `tile_out` once before the outer loop and writes it to `output`
/// once after — the exact reuse ROW 148's own "blocked" rejected-alternative
/// named as mechanically sound but blocked on this function's own gate.
#[cfg(target_arch = "aarch64")]
fn run_conv_gemm_tile(plan: &ConvGemmTilePlan, raw: &[&[f32]], output: &mut [f32]) {
    let tiled_rows = plan.m_total - plan.m_total % TILE_ROWS;
    let mut row = 0usize;
    while row < tiled_rows {
        conv_gemm_row_block::<TILE_ROWS>(plan, raw, output, row);
        row += TILE_ROWS;
    }
    match plan.m_total - tiled_rows {
        0 => {}
        1 => conv_gemm_row_block::<1>(plan, raw, output, tiled_rows),
        2 => conv_gemm_row_block::<2>(plan, raw, output, tiled_rows),
        3 => conv_gemm_row_block::<3>(plan, raw, output, tiled_rows),
        4 => conv_gemm_row_block::<4>(plan, raw, output, tiled_rows),
        5 => conv_gemm_row_block::<5>(plan, raw, output, tiled_rows),
        _ => unreachable!("m_total - tiled_rows must be < TILE_ROWS (6) after the main tiled pass"),
    }
}

/// One `ROWS`-tall strip of [`run_conv_gemm_tile`]'s own `M x N` output,
/// generic over `ROWS` the same way [`gemm_tile_neon`] itself is: the main
/// pass monomorphises at [`TILE_ROWS`], the row-remainder pass (any leftover
/// `1..=5`) at exactly the width it needs, identical body either way. The
/// column loop mirrors `run_reduce`'s own main tile pass — a `TILE_COLS`-wide
/// NEON tile per step, then a scalar remainder for whatever `n_total %
/// TILE_COLS` leaves over (never fired for any of `Conv`'s 3 real mnist
/// folds, all `n_total` multiples of 4, but not assumed so here).
#[cfg(target_arch = "aarch64")]
fn conv_gemm_row_block<const ROWS: usize>(plan: &ConvGemmTilePlan, raw: &[&[f32]], output: &mut [f32], row_start: usize) {
    let out_row_base = plan.out_base + plan.out_row_stride * row_start as i64;
    let a_row_base = plan.base_m + plan.row_stride_m * row_start as i64;
    let tiled_cols = plan.n_total - plan.n_total % TILE_COLS;

    let mut col = 0usize;
    while col < tiled_cols {
        let mut tile_out = [[plan.seed; TILE_COLS]; ROWS];
        let mut a_base = a_row_base;
        let mut b_base = plan.base_n + plan.col_stride_n * col as i64;
        for _ in 0..plan.outer_extent {
            // `conv_gemm_tile_plan`'s own gate already proved: no gathers,
            // `inner_span` contiguous elements at `a_base`/`b_base` for
            // every row and column this tile visits, `m_total`/`n_total`
            // bound every offset formed below within the source slices.
            unsafe {
                gemm_tile_neon::<ROWS>(
                    KStridedTile { data: raw[plan.index_m], base: a_base, k_stride: plan.row_stride_m },
                    KStridedTile { data: raw[plan.index_n], base: b_base, k_stride: plan.col_stride_n },
                    plan.inner_span,
                    &mut tile_out,
                );
            }
            a_base += plan.outer_stride_m;
            b_base += plan.outer_stride_n;
        }
        for (row, tile_row) in tile_out.iter().enumerate() {
            let out_row = out_row_base + plan.out_row_stride * row as i64;
            for (column, &value) in tile_row.iter().enumerate() {
                output[(out_row + (col + column) as i64) as usize] = value;
            }
        }
        col += TILE_COLS;
    }

    for n in tiled_cols..plan.n_total {
        for row in 0..ROWS {
            let mut a_base = a_row_base + plan.row_stride_m * row as i64;
            let mut b_base = plan.base_n + plan.col_stride_n * n as i64;
            let mut total = plan.seed;
            for _ in 0..plan.outer_extent {
                for step in 0..plan.inner_span as i64 {
                    total = raw[plan.index_m][(a_base + step) as usize].mul_add(raw[plan.index_n][(b_base + step) as usize], total);
                }
                a_base += plan.outer_stride_m;
                b_base += plan.outer_stride_n;
            }
            let out_row = out_row_base + plan.out_row_stride * row as i64;
            output[(out_row + n as i64) as usize] = total;
        }
    }
}

/// Packed bytes per `Q4_K` super-block — re-exported at this crate's own
/// name rather than spelling `proxima_gguf::quant::q4_k::BLOCK_BYTES` at
/// every call site below.
const Q4K_BLOCK_BYTES: usize = proxima_gguf::quant::q4_k::BLOCK_BYTES;

/// Decoded `f32` elements per `Q4_K` super-block (`QK_K` in ggml/gguf
/// terms). `Q5_K` and `Q6_K` share this exact per-superblock element count
/// (both codecs' own module docs: `QK_K` is 256 crate-wide) -- only the
/// packed byte count differs per format ([`Q5K_BLOCK_BYTES`]/
/// [`Q6K_BLOCK_BYTES`] below), so this one constant covers all three rather
/// than three identical `_BLOCK_ELEMENTS` constants.
const Q4K_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q4_k::QK_K;

/// Packed bytes per `Q5_K` super-block — needed unconditionally (not just
/// under `q5k-int8-dot`) because [`run_reduce_quantized`]'s dispatch reads
/// it regardless of which matmul arm (dequantize-then-fold or packed int8
/// dot) actually runs.
const Q5K_BLOCK_BYTES: usize = proxima_gguf::quant::q5_k::BLOCK_BYTES;

/// Packed bytes per `Q6_K` super-block — same reasoning as
/// [`Q5K_BLOCK_BYTES`].
const Q6K_BLOCK_BYTES: usize = proxima_gguf::quant::q6_k::BLOCK_BYTES;

/// One output row of a `Q4_K`-quantized-weight x `f32`-activation dot
/// product — the scalar counterpart [`reject_non_float32`]'s quantized-weight
/// exemption documents. `weight_row` is one packed weight row's raw bytes
/// (a whole number of `Q4_K` super-blocks, [`Q4K_BLOCK_BYTES`] each — not a
/// [`KStridedTile`], which only ever addresses `f32` data, never the packed
/// `u8` bytes a quantized row is stored as); `activation` is the matching
/// `f32` slice, `Q4K_BLOCK_ELEMENTS` (256) wide per block.
///
/// Dequantizes one super-block at a time into a reused stack buffer
/// (`[f32; 256]`, never a per-row or per-matrix allocation) via
/// [`proxima_gguf::quant::q4_k::dequantize_block`] — the crate's own tested
/// `Q4_K` codec, ported bit-for-bit from `ggml-quants.c` and proven against
/// real GGUF weights — then folds it against the matching activation slice.
/// This reads [`Q4K_BLOCK_BYTES`] (144) bytes per 256 weights from memory
/// rather than the 1024 bytes a pre-expanded `f32` row would cost, which is
/// the whole point: the weight matrix is never materialized as `f32`, only
/// one super-block at a time is. It stops short of ggml's own register-level
/// int4 `vec_dot` (masking/shifting nibbles straight into a SIMD multiply,
/// no `f32` intermediate at all, not even a 256-element one) — see this
/// function's caller for exactly what is and is not NEON-accelerated.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`Q4K_BLOCK_BYTES`], or `activation.len()` does not
/// equal the row's block count times [`Q4K_BLOCK_ELEMENTS`].
fn dot_q4k_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q4K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q4_k block size",
        });
    }
    let block_count = weight_row.len() / Q4K_BLOCK_BYTES;
    if activation.len() != block_count * Q4K_BLOCK_ELEMENTS {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        });
    }

    let mut scratch = [0.0f32; Q4K_BLOCK_ELEMENTS];
    let mut acc = 0.0f32;
    for (block, activation_chunk) in weight_row
        .as_chunks::<Q4K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<Q4K_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::q4_k::dequantize_block(block, &mut scratch);
        // `DOT_LANES` (8) independent partial sums instead of one serial
        // mul_add chain -- reuses the same fold `reduce_dot_binary` already
        // uses for every f32 GEMM contraction (ROW 12, discipline.md);
        // `Q4K_BLOCK_ELEMENTS` (256) is a whole multiple of `DOT_LANES` so
        // every block folds with zero remainder.
        acc = dot_fold_fused_multiply_add(
            &scratch,
            activation_chunk,
            DotFold { len: Q4K_BLOCK_ELEMENTS, init: acc, seeded: true },
        );
    }
    Ok(acc)
}

/// A full `Q4_K`-quantized weight matrix (`rows` x `k`, row-major packed
/// bytes) times one `f32` activation vector (`k` wide) — batch-1 decode's
/// actual shape, the case the module docs measure at 4.00 bytes/mac. Each
/// output row is independent, so this is the scalar fallback
/// `reject_non_float32`'s quantized-weight exemption routes to when no
/// NEON tile plan claims the node; see `dot_q4k_f32` for the per-row
/// kernel and exactly what it does and does not materialize.
///
/// # Errors
/// Propagates `dot_q4k_f32`'s [`TensorError::QuantizedShapeMismatch`] for
/// the first row that fails its shape check, or reports the same error if
/// `weights.len()` is not a whole multiple of `rows`.
pub fn matmul_q4k_f32(weights: &[u8], rows: usize, activation: &[f32]) -> Result<Vec<f32>, TensorError> {
    matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_q4k_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_q4k_f32,
    )
}

/// Shared dispatch shape behind `matmul_q4k_f32`/`matmul_q5k_f32`/
/// `matmul_q6k_f32`: validate `rows`/`weights.len()`, then route the
/// per-row work either through [`matmul_rows_threaded`] (pool dispatch) or
/// a sequential `chunks_exact` fold, depending on
/// [`quantized_matmul_workers`]'s call. The three codecs differ only in
/// which per-row kernel (`dot_row`) they fold with and which `&'static str`
/// reasons their shape errors carry — both isolated as parameters so this
/// is the only copy of the dispatch logic itself.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] with `zero_rows_reason` if
/// `rows == 0`, with `row_length_reason` if `weights.len()` is not a whole
/// multiple of `rows`, or whatever `dot_row` itself reports for the first
/// row that fails its own shape check.
fn matmul_quantized_dispatch<Row>(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
    zero_rows_reason: &'static str,
    row_length_reason: &'static str,
    dot_row: Row,
) -> Result<Vec<f32>, TensorError>
where
    Row: Fn(&[u8], &[f32]) -> Result<f32, TensorError> + Sync,
{
    if rows == 0 {
        return Err(TensorError::QuantizedShapeMismatch { reason: zero_rows_reason });
    }
    if !weights.len().is_multiple_of(rows) {
        return Err(TensorError::QuantizedShapeMismatch { reason: row_length_reason });
    }
    let row_bytes = weights.len() / rows;
    match quantized_matmul_workers(rows, activation.len()) {
        // No shared cohort session at this call site — `matmul_quantized_dispatch`
        // backs the dequantize-then-fold codecs (`matmul_q4k_f32`/`q5k_f32`/
        // `q6k_f32`), called standalone by non-matmul consumers and tests, not
        // through `evaluate_quantized`'s per-forward session.
        Some(workers) => matmul_rows_threaded(rows, 1, workers, None, activation.len(), |row, slot| {
            let start = row * row_bytes;
            slot[0] = dot_row(&weights[start..start + row_bytes], activation)?;
            Ok(())
        }),
        None => weights
            .chunks_exact(row_bytes)
            .map(|weight_row| dot_row(weight_row, activation))
            .collect(),
    }
}

/// [`dot_q4k_f32`]'s mechanism applied to `Q5_K`: dequantizes one
/// super-block at a time into a reused stack buffer via
/// [`proxima_gguf::quant::q5_k::dequantize_block`], then folds against the
/// matching activation slice with the same [`dot_fold_fused_multiply_add`]
/// fold. This is [`QuantizedBlock::Q5K`]'s codec path whenever
/// `q5k-int8-dot` is off, and stays the codec path for non-matmul
/// consumers regardless.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`Q5K_BLOCK_BYTES`], or `activation.len()` does not
/// equal the row's block count times [`Q4K_BLOCK_ELEMENTS`].
fn dot_q5k_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q5K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q5_k block size",
        });
    }
    let block_count = weight_row.len() / Q5K_BLOCK_BYTES;
    if activation.len() != block_count * Q4K_BLOCK_ELEMENTS {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        });
    }

    let mut scratch = [0.0f32; Q4K_BLOCK_ELEMENTS];
    let mut acc = 0.0f32;
    for (block, activation_chunk) in weight_row
        .as_chunks::<Q5K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<Q4K_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::q5_k::dequantize_block(block, &mut scratch);
        acc = dot_fold_fused_multiply_add(
            &scratch,
            activation_chunk,
            DotFold { len: Q4K_BLOCK_ELEMENTS, init: acc, seeded: true },
        );
    }
    Ok(acc)
}

/// A full `Q5_K`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector — `dot_q5k_f32`'s per-row kernel, one row at a time
/// (no `matmul_q4k_f32`-style thread split; `Q5_K` has not yet earned that
/// on its own bench).
///
/// # Errors
/// Propagates `dot_q5k_f32`'s [`TensorError::QuantizedShapeMismatch`], or
/// reports the same error if `weights.len()` is not a whole multiple of
/// `rows`.
pub fn matmul_q5k_f32(weights: &[u8], rows: usize, activation: &[f32]) -> Result<Vec<f32>, TensorError> {
    // proxima-debugger diagnostic: was ALWAYS sequential -- unlike
    // `matmul_q4k_f32`/`matmul_q4k_q8k_f32`, it never called
    // `quantized_matmul_workers`, so it was invisible to every other
    // `MATMUL_*` counter. Now routed through the same
    // `matmul_quantized_dispatch` pool dispatch as `matmul_q4k_f32`; timer
    // kept as a whole-function wrap so this counter stays comparable to the
    // pre-fix baseline it was built to measure.
    #[cfg(feature = "instrument")]
    let diag_q5k_started = instrument::read_ticks();
    let result = matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_q5k_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_q5k_f32,
    );
    #[cfg(feature = "instrument")]
    {
        counter!(instrument::MATMUL_Q5K_F32_CALLS, 1);
        counter!(
            instrument::MATMUL_Q5K_F32_TICKS,
            instrument::elapsed_ticks(diag_q5k_started)
        );
    }
    result
}

/// [`dot_q4k_f32`]'s mechanism applied to `Q6_K`: dequantizes one
/// super-block at a time via [`proxima_gguf::quant::q6_k::dequantize_block`],
/// then folds against the matching activation slice. [`QuantizedBlock::Q6K`]'s
/// codec path whenever `q6k-int8-dot` is off.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`Q6K_BLOCK_BYTES`], or `activation.len()` does not
/// equal the row's block count times [`Q4K_BLOCK_ELEMENTS`].
fn dot_q6k_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q6K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q6_k block size",
        });
    }
    let block_count = weight_row.len() / Q6K_BLOCK_BYTES;
    if activation.len() != block_count * Q4K_BLOCK_ELEMENTS {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        });
    }

    let mut scratch = [0.0f32; Q4K_BLOCK_ELEMENTS];
    let mut acc = 0.0f32;
    for (block, activation_chunk) in weight_row
        .as_chunks::<Q6K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<Q4K_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::q6_k::dequantize_block(block, &mut scratch);
        acc = dot_fold_fused_multiply_add(
            &scratch,
            activation_chunk,
            DotFold { len: Q4K_BLOCK_ELEMENTS, init: acc, seeded: true },
        );
    }
    Ok(acc)
}

/// A full `Q6_K`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector — `dot_q6k_f32`'s per-row kernel.
///
/// # Errors
/// Propagates `dot_q6k_f32`'s [`TensorError::QuantizedShapeMismatch`], or
/// reports the same error if `weights.len()` is not a whole multiple of
/// `rows`.
pub fn matmul_q6k_f32(weights: &[u8], rows: usize, activation: &[f32]) -> Result<Vec<f32>, TensorError> {
    // proxima-debugger diagnostic: see the matching note on
    // `matmul_q5k_f32` -- same was-always-sequential shape, now routed
    // through the same `matmul_quantized_dispatch` pool dispatch, same
    // whole-function timer for baseline comparability.
    #[cfg(feature = "instrument")]
    let diag_q6k_started = instrument::read_ticks();
    let result = matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_q6k_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_q6k_f32,
    );
    #[cfg(feature = "instrument")]
    {
        counter!(instrument::MATMUL_Q6K_F32_CALLS, 1);
        counter!(
            instrument::MATMUL_Q6K_F32_TICKS,
            instrument::elapsed_ticks(diag_q6k_started)
        );
    }
    result
}

/// Packed bytes per `Q8_0` block -- needed unconditionally, same reasoning
/// as [`Q5K_BLOCK_BYTES`].
const Q8_0_BLOCK_BYTES: usize = proxima_gguf::quant::q8_0::BLOCK_BYTES;

/// Decoded `f32` elements per `Q8_0` block (`QK8_0`, 32) -- unlike the
/// `Q4_K`/`Q5_K`/`Q6_K` family, `Q8_0` has no shared super-block constant
/// with them; see [`QuantizedBlock::Q8_0`]'s own doc for why this codec's
/// much smaller block is the one that fits the key/value context cache's
/// row width.
const Q8_0_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q8_0::QK8_0;

/// [`dot_q4k_f32`]'s mechanism applied to `Q8_0`: dequantizes one 32-element
/// block at a time into a reused stack buffer via
/// [`proxima_gguf::quant::q8_0::dequantize_block`], then folds against the
/// matching activation slice with the same [`dot_fold_fused_multiply_add`]
/// fold. `Q8_0`'s block carries no sub-block scale structure at all -- one
/// `f16` delta per 32 elements -- so this is a direct port of `q8_0.rs`'s
/// own `dequantize_block`, not a variant of the K-quant super-block
/// unpacking `dot_q4k_f32`/`dot_q5k_f32`/`dot_q6k_f32` share.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`Q8_0_BLOCK_BYTES`], or `activation.len()` does not
/// equal the row's block count times [`Q8_0_BLOCK_ELEMENTS`].
fn dot_q8_0_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q8_0_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q8_0 block size",
        });
    }
    let block_count = weight_row.len() / Q8_0_BLOCK_BYTES;
    if activation.len() != block_count * Q8_0_BLOCK_ELEMENTS {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        });
    }

    let mut scratch = [0.0f32; Q8_0_BLOCK_ELEMENTS];
    let mut acc = 0.0f32;
    for (block, activation_chunk) in weight_row
        .as_chunks::<Q8_0_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<Q8_0_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::q8_0::dequantize_block(block, &mut scratch);
        acc = dot_fold_fused_multiply_add(
            &scratch,
            activation_chunk,
            DotFold { len: Q8_0_BLOCK_ELEMENTS, init: acc, seeded: true },
        );
    }
    Ok(acc)
}

/// A full `Q8_0`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector -- `dot_q8_0_f32`'s per-row kernel, the scalar
/// dequantize-then-fold path only (no packed int8-dot wide fold the way
/// `Q4_K`/`Q5_K`/`Q6_K` earn under their own `*-int8-dot` features): this is
/// the growable key/value context cache's storage codec, appended one call's
/// worth of new rows at a time, so the wide per-call fold those weight
/// codecs use (streamed once, reused across every batch position) does not
/// apply the same way here -- the cache itself IS the thing growing between
/// calls.
///
/// # Errors
/// Propagates `dot_q8_0_f32`'s [`TensorError::QuantizedShapeMismatch`], or
/// reports the same error if `weights.len()` is not a whole multiple of
/// `rows`.
pub fn matmul_q8_0_f32(weights: &[u8], rows: usize, activation: &[f32]) -> Result<Vec<f32>, TensorError> {
    matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_q8_0_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_q8_0_f32,
    )
}

/// Packed bytes per `Q4_0` block -- needed unconditionally, same reasoning
/// as [`Q8_0_BLOCK_BYTES`].
const Q4_0_BLOCK_BYTES: usize = proxima_gguf::quant::q4_0::BLOCK_BYTES;

/// Decoded `f32` elements per `Q4_0` block (`QK4_0`, 32) -- the same flat
/// 32-element shape as [`QuantizedBlock::Q8_0`], not [`Q4K_BLOCK_ELEMENTS`]'s
/// 256-wide super-block; see [`QuantizedBlock::Q4_0`]'s own doc.
const Q4_0_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q4_0::QK4_0;

/// [`dot_q8_0_f32`]'s mechanism applied to `Q4_0`: dequantizes one
/// 32-element block at a time into a reused stack buffer via
/// [`proxima_gguf::quant::q4_0::dequantize_block`], then folds against the
/// matching activation slice with the same [`dot_fold_fused_multiply_add`]
/// fold. `Q4_0` has no shared super-block with the K-quant family and no
/// `dot_fn_for` entry (see that function's own doc) -- this scalar
/// dequantize-then-fold path is the only one this codec takes on the CPU
/// backend.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`Q4_0_BLOCK_BYTES`], or `activation.len()` does not
/// equal the row's block count times [`Q4_0_BLOCK_ELEMENTS`].
fn dot_q4_0_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q4_0_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q4_0 block size",
        });
    }
    let block_count = weight_row.len() / Q4_0_BLOCK_BYTES;
    if activation.len() != block_count * Q4_0_BLOCK_ELEMENTS {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        });
    }

    let mut scratch = [0.0f32; Q4_0_BLOCK_ELEMENTS];
    let mut acc = 0.0f32;
    for (block, activation_chunk) in weight_row
        .as_chunks::<Q4_0_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<Q4_0_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::q4_0::dequantize_block(block, &mut scratch);
        acc = dot_fold_fused_multiply_add(
            &scratch,
            activation_chunk,
            DotFold { len: Q4_0_BLOCK_ELEMENTS, init: acc, seeded: true },
        );
    }
    Ok(acc)
}

/// A full `Q4_0`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector -- `dot_q4_0_f32`'s per-row kernel, same scalar
/// dequantize-then-fold shape as [`matmul_q8_0_f32`] (no packed int8-dot
/// wide fold exists for this codec either).
///
/// # Errors
/// Propagates `dot_q4_0_f32`'s [`TensorError::QuantizedShapeMismatch`], or
/// reports the same error if `weights.len()` is not a whole multiple of
/// `rows`.
pub fn matmul_q4_0_f32(weights: &[u8], rows: usize, activation: &[f32]) -> Result<Vec<f32>, TensorError> {
    matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_q4_0_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_q4_0_f32,
    )
}

/// Bytes per half-precision element -- both [`QuantizedBlock::Float16`]
/// and [`QuantizedBlock::BFloat16`] are 2-byte formats
/// ([`DType::size_bytes`] agrees for both).
const HALF_PRECISION_ELEMENT_BYTES: usize = 2;

/// Elements converted per stack-buffer chunk in [`dot_f16_f32`]/
/// [`dot_bf16_f32`] -- reuses [`Q4K_BLOCK_ELEMENTS`] (256) for the same
/// cache/register-friendly width the K-quant kernels beside this one were
/// already measured at, not because these two unrelated formats share any
/// structural need to agree on it. A structural axis sizing a stack array,
/// not a runtime tunable -- same reasoning as `sized.rs`'s own
/// `DOT_LANES`/`WIDTH_TILE_ROWS`.
const HALF_PRECISION_DOT_CHUNK: usize = Q4K_BLOCK_ELEMENTS;

/// One output row of a half-precision-weight x `f32`-activation dot
/// product -- composes two EXISTING primitives rather than shipping a new
/// kernel (guiding-principles §1's pipe question, answered by writing the
/// expression instead of a paragraph): [`Convert::<f16, f32>`]'s
/// [`SimdConvert::convert_slice`] widens one stack-buffer chunk of packed
/// bytes to `f32`, then [`dot_fold_fused_multiply_add`] folds it against
/// the activation slice -- the exact fold every other codec's dequantize
/// step already reuses. No half-precision-specific SIMD dot was written:
/// unlike `Q4_K`/`Q5_K`/`Q6_K`'s packed nibbles, an `f16` element carries no
/// block or scale structure to unpack, so the composed form is not a
/// stand-in for a missing fused kernel -- it is the whole job. Byte pairs
/// are widened to `f16` by hand (`u16::from_le_bytes` then `f16::from_bits`)
/// rather than an unsafe transmute of `&[u8]` to `&[f16]`. because a
/// `weight_row` sub-slice's 2-byte alignment relative to its backing
/// allocation is not a language guarantee.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`HALF_PRECISION_ELEMENT_BYTES`], or `activation.len()`
/// does not equal the row's decoded element count.
fn dot_f16_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(HALF_PRECISION_ELEMENT_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "f16 weight row length is not a whole multiple of 2 bytes",
        });
    }
    let element_count = weight_row.len() / HALF_PRECISION_ELEMENT_BYTES;
    if activation.len() != element_count {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the f16 weight row's element count",
        });
    }

    let converter = Convert::<f16, f32>::new();
    let mut half_scratch = [f16::from_bits(0); HALF_PRECISION_DOT_CHUNK];
    let mut wide_scratch = [0.0f32; HALF_PRECISION_DOT_CHUNK];
    let mut acc = 0.0f32;
    let byte_chunk_len = HALF_PRECISION_DOT_CHUNK * HALF_PRECISION_ELEMENT_BYTES;
    for (byte_chunk, activation_chunk) in weight_row.chunks(byte_chunk_len).zip(activation.chunks(HALF_PRECISION_DOT_CHUNK)) {
        let chunk_len = activation_chunk.len();
        for (slot, bytes) in half_scratch[..chunk_len].iter_mut().zip(byte_chunk.as_chunks::<HALF_PRECISION_ELEMENT_BYTES>().0) {
            *slot = f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]));
        }
        converter.convert_slice(&half_scratch[..chunk_len], &mut wide_scratch[..chunk_len]);
        acc = dot_fold_fused_multiply_add(
            &wide_scratch[..chunk_len],
            activation_chunk,
            DotFold { len: chunk_len, init: acc, seeded: true },
        );
    }
    Ok(acc)
}

/// [`dot_f16_f32`]'s mechanism applied to `bfloat16` -- same composed
/// convert-then-fold shape, [`Convert::<bf16, f32>`] in place of
/// `Convert<f16, f32>`. See that function's doc for why no new kernel was
/// written.
///
/// # Errors
/// Same shape as [`dot_f16_f32`]'s.
fn dot_bf16_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(HALF_PRECISION_ELEMENT_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "bf16 weight row length is not a whole multiple of 2 bytes",
        });
    }
    let element_count = weight_row.len() / HALF_PRECISION_ELEMENT_BYTES;
    if activation.len() != element_count {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the bf16 weight row's element count",
        });
    }

    let converter = Convert::<bf16, f32>::new();
    let mut half_scratch = [bf16::from_bits(0); HALF_PRECISION_DOT_CHUNK];
    let mut wide_scratch = [0.0f32; HALF_PRECISION_DOT_CHUNK];
    let mut acc = 0.0f32;
    let byte_chunk_len = HALF_PRECISION_DOT_CHUNK * HALF_PRECISION_ELEMENT_BYTES;
    for (byte_chunk, activation_chunk) in weight_row.chunks(byte_chunk_len).zip(activation.chunks(HALF_PRECISION_DOT_CHUNK)) {
        let chunk_len = activation_chunk.len();
        for (slot, bytes) in half_scratch[..chunk_len].iter_mut().zip(byte_chunk.as_chunks::<HALF_PRECISION_ELEMENT_BYTES>().0) {
            *slot = bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]));
        }
        converter.convert_slice(&half_scratch[..chunk_len], &mut wide_scratch[..chunk_len]);
        acc = dot_fold_fused_multiply_add(
            &wide_scratch[..chunk_len],
            activation_chunk,
            DotFold { len: chunk_len, init: acc, seeded: true },
        );
    }
    Ok(acc)
}

/// A full `Float16`-weight matrix (`rows` x `k`, row-major raw bytes) times
/// one `f32` activation vector -- `dot_f16_f32`'s per-row kernel driven
/// through the same `matmul_quantized_dispatch` every other codec shares.
///
/// # Errors
/// Propagates `dot_f16_f32`'s [`TensorError::QuantizedShapeMismatch`] for
/// the first row that fails its shape check, or reports the same error if
/// `weights.len()` is not a whole multiple of `rows`.
pub fn matmul_f16_f32(weights: &[u8], rows: usize, activation: &[f32]) -> Result<Vec<f32>, TensorError> {
    matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_f16_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_f16_f32,
    )
}

/// [`matmul_f16_f32`]'s `bfloat16` counterpart.
///
/// # Errors
/// Same shape as [`matmul_f16_f32`]'s.
pub fn matmul_bf16_f32(weights: &[u8], rows: usize, activation: &[f32]) -> Result<Vec<f32>, TensorError> {
    matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_bf16_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_bf16_f32,
    )
}

/// Below this many total multiply-accumulates (`rows * activation.len()`),
/// a quantized matmul's row loop runs sequentially even when more than one
/// hardware thread exists: `std::thread::scope`'s spawn/join overhead would
/// outweigh the work. Reuses [`PARALLEL_THRESHOLD`], the same element-count
/// floor [`evaluate_node_parallel`] already gates its own per-node chunk
/// dispatch on, rather than a second magic number for the same policy.
/// `None` also covers `rows < workers`, where a per-row split would leave
/// some worker with nothing to do.
///
/// This is a different axis than [`BoundOp::split`]/[`evaluate_node_parallel`]:
/// that machinery chunks a reduce node along its outermost *surviving*
/// output axis (`output_axes.first()` — `bind.rs`'s own doc), which for a
/// batch-1 decode step is the batch axis, extent 1 — nothing to split.
/// `matmul_q4k_f32`/`matmul_q4k_q8k_f32`'s weight-row loop has no such
/// dependency on `BoundOp` at all (`rows`/`k` arrive as plain integers, not
/// a bound node), so it can chunk the one axis that is actually wide at
/// batch-1: weight rows.
///
/// Queries Apple's performance-core count (`hw.perflevel0.logicalcpu`) via
/// `sysctlbyname`, so matmul dispatch spawns workers only across P-cores and
/// skips the E-cores that add per-call dispatch cost without contributing
/// matmul throughput (measured: 8 P-cores beats 10 logical cores on every
/// shape, `docs/discipline.md`). On a homogeneous machine every core reports
/// as perflevel0, so this returns the same count `available_parallelism`
/// would — the fallback in [`matmul_worker_count`] is for when the sysctl is
/// absent or answers something nonsensical, not a second code path for
/// homogeneous boxes.
#[cfg(target_vendor = "apple")]
fn performance_core_count() -> Option<usize> {
    let name = c"hw.perflevel0.logicalcpu";
    let mut value: i32 = 0;
    let mut size = core::mem::size_of::<i32>();
    // FFI: sysctlbyname has no safe wrapper in libc; the output pointer and
    // size are stack-local and sized to match the i32 the sysctl documents.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&raw mut value).cast(),
            &raw mut size,
            core::ptr::null_mut(),
            0,
        )
    };
    if status != 0 || value <= 0 {
        return None;
    }
    Some(value as usize)
}

/// Linux's analogue of Apple's `hw.perflevel0.logicalcpu`: on a hybrid
/// Intel part (Alder Lake and later) the kernel exposes the P-core set at
/// `/sys/devices/cpu_core/cpus` (`E`-cores at the sibling `cpu_atom/cpus`,
/// which this function has no need to read since it only wants the
/// performance set). The path is absent on a non-hybrid CPU -- that IS the
/// answer there, not a missing one: every core is already a performance
/// core, so `None` here is this crate's existing "nothing extra to learn"
/// contract, and [`matmul_worker_count`]'s own `available_parallelism()`
/// fallback already returns the right count for that case.
///
/// Deliberately does NOT re-derive process affinity
/// (`sched_getaffinity`/cgroup quota) here: `std::thread::available_parallelism`'s
/// own documentation ("Host environments such as VMs or container
/// orchestrators may want to restrict the amount of parallelism...") states
/// that it already honors those limits on Linux, and
/// [`matmul_worker_count`]'s `.filter(|&count| count >= 1 && count <=
/// available)` bound is what keeps a global P-core count from ever
/// exceeding what the process may actually use -- duplicating the affinity
/// read here would be a second source of truth for the same fact.
#[cfg(target_os = "linux")]
fn performance_core_count() -> Option<usize> {
    let text = std::fs::read_to_string("/sys/devices/cpu_core/cpus").ok()?;
    parse_cpu_list_count(&text)
}

/// Parses a Linux sysfs CPU list (`/sys/devices/cpu_core/cpus`'s own
/// format, e.g. `"0-7,16-23"` or a bare `"4"`) into the count of CPU ids it
/// names -- [`performance_core_count`]'s only consumer, split out so the
/// parsing itself is testable on every host (this crate's dev boxes are
/// aarch64-darwin; the file this reads only exists on a real Linux kernel).
#[cfg(any(test, target_os = "linux"))]
fn parse_cpu_list_count(text: &str) -> Option<usize> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let mut count = 0usize;
    for range in text.split(',') {
        let mut bounds = range.splitn(2, '-');
        let start: usize = bounds.next()?.trim().parse().ok()?;
        let end: usize = match bounds.next() {
            Some(end) => end.trim().parse().ok()?,
            None => start,
        };
        count += end.checked_sub(start)?.checked_add(1)?;
    }
    if count == 0 { None } else { Some(count) }
}

#[cfg(not(any(target_vendor = "apple", target_os = "linux")))]
fn performance_core_count() -> Option<usize> {
    None
}

/// Worker count for the row split, resolved once and cached for the process
/// lifetime. `std::thread::available_parallelism` is a `sysctl` on macOS —
/// measured at 3.53 us/call, 4.768 ms across the 1350 calls one real forward
/// pass makes through [`quantized_matmul_workers`] — so calling it per
/// matmul is pure waste on a value that never changes at runtime.
///
/// Prefers [`performance_core_count`] over `available_parallelism` on Apple
/// targets: `available_parallelism` returns P+E, but only the P cores run
/// this workload at full speed (measured, see [`performance_core_count`]'s
/// doc), so counting E-cores in the worker pool adds coordination overhead
/// without adding throughput. Falls back to `available_parallelism()` when
/// the sysctl is unavailable or answers something nonsensical (`<= 0` or
/// larger than `available_parallelism()` itself).
///
/// `PROXIMA_MATMUL_WORKERS`, if set to a valid non-zero integer, overrides
/// both of the above; this exists to sweep worker counts without a rebuild.
/// The env var is read once via `OnceLock`, never per call — a per-call
/// `std::env::var` allocates a `String` on every one of those 1350 calls and
/// would contaminate the very cost this cache exists to remove. Default
/// (unset) behavior is unchanged otherwise.
fn matmul_worker_count() -> usize {
    static WORKER_COUNT: OnceLock<usize> = OnceLock::new();
    *WORKER_COUNT.get_or_init(|| {
        std::env::var("PROXIMA_MATMUL_WORKERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&count| count > 0)
            .unwrap_or_else(|| {
                let available = thread::available_parallelism().map(NonZeroUsize::get).unwrap_or(1);
                performance_core_count().filter(|&count| count >= 1 && count <= available).unwrap_or(available)
            })
    })
}

fn quantized_matmul_workers(rows: usize, contraction_width: usize) -> Option<usize> {
    // proxima-debugger diagnostic (this is the ONE choke point every
    // quantized-matmul row-batch call passes through, `Q4_K`/`Q5_K`/`Q6_K`,
    // int8-dot or f32-dot alike): counts total calls and how many actually
    // return `None` (sequential fallback, no thread pool at all) versus
    // `Some` (threaded via `matmul_rows_threaded`) -- settles whether the
    // 1296-call `PARALLEL_NODES`/`MATMUL_DISPATCH_CALLS` figure already
    // covers every row-batch this forward pass runs, or whether a
    // sequential remainder is hiding node wall time `matmul_rows_threaded`
    // never sees.
    #[cfg(feature = "instrument")]
    counter!(instrument::MATMUL_WORKERS_CALLS, 1);
    let total_macs = rows.checked_mul(contraction_width)?;
    if total_macs < PARALLEL_THRESHOLD {
        #[cfg(feature = "instrument")]
        counter!(instrument::MATMUL_WORKERS_NONE, 1);
        return None;
    }
    #[cfg(feature = "instrument")]
    let diag_available_parallelism_started = instrument::read_ticks();
    let workers = matmul_worker_count();
    #[cfg(feature = "instrument")]
    counter!(
        instrument::MATMUL_AVAILABLE_PARALLELISM_TICKS,
        instrument::elapsed_ticks(diag_available_parallelism_started)
    );
    let decision = (workers > 1 && rows >= workers).then_some(workers);
    #[cfg(feature = "instrument")]
    if decision.is_none() {
        counter!(instrument::MATMUL_WORKERS_NONE, 1);
    }
    decision
}

use crate::sized::{MIN_MACS_PER_CHUNK, ROW_OVERSUBSCRIBE};

/// Runs `rows` independent per-row computations (`dot_row`) through the
/// shared [`nest_pool`], each writing its own contiguous sub-range of the
/// returned buffer — the row-loop counterpart of [`run_chunks_threaded`]'s
/// pool sibling (`BoundOp`-chunk parallelism, one level up the call stack):
/// no [`BoundOp`] exists at this call site to split, only a row count and a
/// per-row closure, so this dispatches directly over row indices instead of
/// `BoundOp` chunks. Every chunk's slice is carved via `split_at_mut` before
/// any chunk is spawned, so no two pullers ever touch the same output
/// element — same soundness argument as `run_chunks_threaded`'s own slice
/// carve.
///
/// Chunk assignment is dynamic, the same shared-cursor mechanism
/// `run_chunks_threaded`/`claim_and_run` use, applied to row ranges instead
/// of `BoundOp`s ([`claim_and_run_rows`]): `rows` is split into up to
/// `workers * ROW_OVERSUBSCRIBE` ranges (more chunks than pullers), and both
/// the `workers - 1` spawned pool tasks and the calling thread pull the next
/// unclaimed chunk off a shared [`AtomicUsize`] cursor instead of each
/// owning one fixed range. A prior 1:1 static split left the calling thread
/// idling in `Receiver::recv` for whichever spawned chunk ran longest even
/// though equal row counts do not mean equal wall-clock (measured 2.04x
/// spread across 8 equal-row chunks of a 1024^3 GEMM, see [`OVERSUBSCRIBE`]'s
/// doc) — a fast puller now claims another chunk instead of idling.
///
/// `contraction_width` (the per-row `k`, i.e. `activation.len()` at every
/// call site) caps that split: `rows * contraction_width` total multiply-add
/// work is floored against [`MIN_MACS_PER_CHUNK`] before the
/// `workers * ROW_OVERSUBSCRIBE` oversubscription is applied, so a call
/// carrying little total work (e.g. `attn_k`/`attn_v`'s narrow projection)
/// gets fewer, larger chunks instead of the same fixed 40-way split a wide
/// call like `ffn_up`/`ffn_gate` earns — see [`MIN_MACS_PER_CHUNK`]'s own
/// doc for the measurement that picked the floor.
///
/// # Safety (of the `unsafe` blocks inside)
/// `dot_row`'s address crosses the pool's `'static` spawn bound the same way
/// `buffers_address`/`chunks_address` do in `run_chunks_threaded`: cast to
/// `usize` here, reconstructed unsafely inside each pool closure. Sound
/// because this function blocks in `Receiver::recv` for every spawned chunk
/// before returning, so `dot_row` (borrowed from the caller for the whole
/// call) outlives every reconstructed reference. Each chunk's output slice
/// is likewise unique by construction (`split_at_mut` above, carved before
/// any puller starts claiming), and `AtomicUsize::fetch_add` never hands the
/// same chunk index to two pullers, so no two closures ever alias the same
/// output range.
/// The chunk count [`matmul_rows_threaded`] splits `rows` into: capped at
/// `workers * ROW_OVERSUBSCRIBE`, but never more than
/// `rows * contraction_width` total macs supports at
/// [`MIN_MACS_PER_CHUNK`] macs per chunk. A call carrying little total work
/// (narrow `contraction_width`, few `rows`) gets fewer, coarser chunks
/// instead of the fixed oversubscription split every shape used to pay --
/// see [`MIN_MACS_PER_CHUNK`]'s own doc for the per-shape measurement that
/// motivated this.
fn row_chunk_count(rows: usize, workers: usize, contraction_width: usize) -> usize {
    let oversubscribed = workers.saturating_mul(ROW_OVERSUBSCRIBE);
    let total_macs = rows.saturating_mul(contraction_width);
    let work_chunks = (total_macs / MIN_MACS_PER_CHUNK).max(1);
    oversubscribed.min(work_chunks).clamp(1, rows.max(1))
}

/// `PROXIMA_COHORT_QUORUM=1` routes the matmul row-cohort round through
/// `CohortSession::run_with_completion(round, Some(&Quorum(chunk_total)))`
/// instead of the zero-overhead `CohortSession::run` default -- exercises
/// the `FanInCompletion` dial `cohort.rs` added (landed, never called from
/// this crate) without changing what gets computed: `Quorum(chunk_total)`
/// is satisfied only once every chunk has retired, the same point cursor
/// exhaustion already stops dispatch at, so the two paths compute the same
/// output. What differs is cost: one extra `completion_ptr` load plus two
/// more atomic loads and a vtable call per claimed chunk, paid by every
/// cohort member on every chunk. Read once and cached, so toggling the env
/// var mid-process has no effect after the first call.
fn cohort_quorum_completion_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PROXIMA_COHORT_QUORUM").is_ok_and(|value| value == "1"))
}

/// [`matmul_rows_threaded`]'s cohort dispatch shape: one round over the same
/// `(row_start, pointer, len)` chunk ranges the pool path carves via
/// `split_at_mut`, run through [`CohortSession::run`] instead of
/// `nest_pool`'s spawn/channel dance. No `'static` erasure is needed here —
/// unlike the pool path, [`CohortSession::run`] blocks the calling thread
/// until every member reports done, so `dot_row` and `chunk_ranges` can stay
/// ordinary borrows for the round's whole lifetime, the same argument
/// `std::thread::scope` relies on.
///
/// `dot_row` can fail (`Row: Fn(usize, &mut [f32]) -> Result<(), TensorError>`,
/// the second argument one row's `width`-wide output slot — `width` is 1 for
/// every caller except [`matmul_q4k_q8k_f32_wide_impl`], which folds every
/// sequence position into that slot so the row's weight bytes are read once
/// and reused across all of them). [`CohortRound::run_chunk`] returns that
/// `Result` directly — [`CohortSession::run`]'s own `RoundReport::first_error`
/// carries the first `Err` any member observes back to the caller, replacing
/// the hand-rolled `OnceLock` this dispatch used to publish through itself.
struct RowRound<'round, Row> {
    dot_row: &'round Row,
    width: usize,
    chunk_ranges: &'round [(usize, usize, usize)],
}

impl<Row> CohortRound<TensorError> for RowRound<'_, Row>
where
    Row: Fn(usize, &mut [f32]) -> Result<(), TensorError> + Sync,
{
    fn chunks(&self) -> usize {
        self.chunk_ranges.len()
    }

    fn run_chunk(&self, chunk: ChunkIndex) -> Result<(), TensorError> {
        let (chunk_start, slice_address, slice_len) = self.chunk_ranges[chunk.0];
        // SAFETY: unique to this chunk by construction (`split_at_mut` in
        // `matmul_rows_threaded` before the round starts); the parent
        // `output` outlives every reconstructed slice because
        // `CohortSession::run` does not return until every member has
        // reported done, i.e. until this closure has returned.
        let chunk_output =
            unsafe { core::slice::from_raw_parts_mut(slice_address as *mut f32, slice_len) };
        run_row_chunk(self.dot_row, self.width, chunk_start, chunk_output)
    }
}

fn matmul_rows_threaded<Row>(
    rows: usize,
    width: usize,
    workers: usize,
    session: Option<&MatmulSession<'_>>,
    contraction_width: usize,
    dot_row: Row,
) -> Result<Vec<f32>, TensorError>
where
    Row: Fn(usize, &mut [f32]) -> Result<(), TensorError> + Sync,
{
    // proxima-debugger diagnostic: everything this function does before its
    // own spawn/own_chunk/recv_wait timer chain starts -- the `output`
    // alloc, the `chunk_ranges` build, `nest_pool()`, and the `Arc`/
    // `sync_channel` allocations. Named `MATMUL_SETUP_TICKS` so a caller can
    // tell "the dispatch chain is slow" apart from "this untimed setup,
    // paid once per call, is slow" -- see that counter's doc.
    #[cfg(feature = "instrument")]
    let diag_setup_started = instrument::read_ticks();
    let mut output = vec![0.0f32; rows * width];
    let chunk_count = row_chunk_count(rows, workers, contraction_width.saturating_mul(width));
    let chunk_len = rows.div_ceil(chunk_count);

    let mut chunk_ranges = Vec::with_capacity(chunk_count);
    let mut remaining = output.as_mut_slice();
    let mut row_start = 0usize;
    while !remaining.is_empty() {
        let take_rows = chunk_len.min(remaining.len() / width);
        let (slice, rest) = remaining.split_at_mut(take_rows * width);
        remaining = rest;
        chunk_ranges.push((row_start, slice.as_mut_ptr() as usize, slice.len()));
        row_start += take_rows;
    }
    let chunk_ranges_len = chunk_ranges.len();
    #[cfg(feature = "instrument")]
    instrument::record_chunks_created(chunk_ranges_len);

    if let Some(session) = session {
        #[cfg(feature = "instrument")]
        counter!(instrument::MATMUL_SETUP_TICKS, instrument::elapsed_ticks(diag_setup_started));
        #[cfg(feature = "instrument")]
        counter!(instrument::PARALLEL_NODES, 1);
        #[cfg(feature = "instrument")]
        counter!(instrument::MATMUL_COHORT_DISPATCH_CALLS, 1);
        let round = RowRound {
            dot_row: &dot_row,
            width,
            chunk_ranges: &chunk_ranges,
        };
        // `CohortSession::run` fuses the leader's own claim loop and its
        // wait for the dedicated members into one call (`cohort.rs`'s
        // `run_round(control)` followed by the `done` spin) -- unlike the
        // pool path, it does not expose a separate claim-only timer, so
        // this whole call is charged to `MATMUL_OWN_CHUNK_TICKS` rather
        // than split against `MATMUL_RECV_WAIT_TICKS` (which stays 0 on
        // this path). Nonzero here is the direct witness that the leader
        // is claiming chunks, not spinning idle -- see `cohort.rs`'s
        // `CohortSession::run` doc for the +14.8 ms it cost while it was.
        #[cfg(feature = "instrument")]
        let diag_own_chunk_started = instrument::read_ticks();
        let report = if cohort_quorum_completion_enabled() {
            session.run_with_completion(&round, Some(&Quorum(chunk_ranges_len)))
        } else {
            session.run(&round)
        };
        #[cfg(feature = "instrument")]
        counter!(
            instrument::MATMUL_OWN_CHUNK_TICKS,
            instrument::elapsed_ticks(diag_own_chunk_started)
        );
        if let Some(error) = report.first_error {
            return Err(error);
        }
        if report.abandoned > 0 {
            return Err(TensorError::ThreadedChunkFailed {
                chunk: report.first_abandoned.map_or(0, |chunk| chunk.0 + 1),
                reason: alloc::string::String::from(
                    "cohort member panicked while running this row chunk",
                ),
            });
        }
        return Ok(output);
    }

    let pool = nest_pool()?;
    // SAFETY-relevant: see this function's doc comment for why casting
    // `dot_row`'s address across the pool's `'static` bound is sound here.
    let dot_row_address = &dot_row as *const Row as usize;
    let next_index = Arc::new(AtomicUsize::new(0));
    let chunk_ranges: Arc<Vec<(usize, usize, usize)>> = Arc::new(chunk_ranges);
    let spawned_count = workers.saturating_sub(1).min(chunk_ranges_len.saturating_sub(1));
    let (result_sender, result_receiver) = sync_channel(chunk_ranges_len);
    #[cfg(feature = "instrument")]
    counter!(
        instrument::MATMUL_SETUP_TICKS,
        instrument::elapsed_ticks(diag_setup_started)
    );

    #[cfg(feature = "instrument")]
    counter!(instrument::PARALLEL_NODES, 1);

    // proxima-debugger diagnostic (this call's own dispatch-overhead
    // breakdown, `instrument.rs::MATMUL_*_TICKS`): times the spawn loop,
    // the caller's own claiming loop, and the `Receiver::recv` wait
    // separately so a caller can tell whether this dispatch is bottlenecked
    // on spawn (granularity too fine), recv (a straggler the cursor could
    // not route around fast enough), or neither (own-chunk + per-chunk
    // compute already accounted by `record_chunk_ticks` dominates and
    // dispatch is not the ceiling).
    #[cfg(feature = "instrument")]
    let diag_spawn_started = instrument::read_ticks();

    for _ in 0..spawned_count {
        let sender = result_sender.clone();
        let next_index = Arc::clone(&next_index);
        let chunk_ranges = Arc::clone(&chunk_ranges);
        drop(pool.spawn(move || {
            claim_and_run_rows::<Row>(&next_index, dot_row_address, width, &chunk_ranges, &sender);
            Ok::<(), _>(())
        }));
    }

    #[cfg(feature = "instrument")]
    let diag_spawn_ticks = instrument::elapsed_ticks(diag_spawn_started);

    #[cfg(feature = "instrument")]
    let diag_own_chunk_started = instrument::read_ticks();
    // the caller pulls from the same shared cursor as every pool task
    // instead of running one reserved chunk: it never sits idle, since
    // finishing a chunk sends it straight back to `next_index` for another.
    claim_and_run_rows::<Row>(&next_index, dot_row_address, width, &chunk_ranges, &result_sender);
    drop(result_sender);
    #[cfg(feature = "instrument")]
    let diag_own_chunk_ticks = instrument::elapsed_ticks(diag_own_chunk_started);

    #[cfg(feature = "instrument")]
    let diag_recv_started = instrument::read_ticks();
    let mut outcomes: Vec<Option<Result<(), TensorError>>> =
        (0..chunk_ranges_len).map(|_| None).collect();
    for _ in 0..chunk_ranges_len {
        match result_receiver.recv() {
            Ok((index, outcome)) => outcomes[index] = Some(outcome),
            // every sender clone is gone (each spawned closure's clone is
            // dropped whether it sends or panics), so no further chunk will
            // ever report — stop waiting instead of blocking forever.
            Err(_) => break,
        }
    }
    #[cfg(feature = "instrument")]
    {
        let diag_recv_ticks = instrument::elapsed_ticks(diag_recv_started);
        counter!(instrument::MATMUL_DISPATCH_CALLS, 1);
        counter!(instrument::MATMUL_SPAWN_TICKS, diag_spawn_ticks);
        counter!(instrument::MATMUL_OWN_CHUNK_TICKS, diag_own_chunk_ticks);
        counter!(instrument::MATMUL_RECV_WAIT_TICKS, diag_recv_ticks);
    }
    for (index, outcome) in outcomes.into_iter().enumerate() {
        match outcome {
            Some(result) => result?,
            None => {
                return Err(TensorError::ThreadedChunkFailed {
                    chunk: index + 1,
                    reason: alloc::string::String::from(
                        "worker did not report a result; ProximaBackgroundPool \
                         catches and discards worker panics (see \
                         prime/src/os/background.rs worker())",
                    ),
                });
            }
        }
    }
    Ok(output)
}

/// Pulls row-chunk indices off `next_index` one at a time and runs each to
/// completion through [`run_row_chunk`], reporting through `sender` — the
/// row-loop counterpart of [`claim_and_run`]'s shared-cursor claim loop,
/// called by both the calling thread and every spawned pool task in
/// [`matmul_rows_threaded`] so a puller that finishes early goes straight
/// back for the next available chunk instead of idling.
///
/// # Safety (of the `unsafe` blocks inside)
/// `dot_row_address` and every `(row_start, pointer, len)` triple in
/// `chunk_ranges` must stay valid, and each slice must be unique to its
/// index, for as long as any puller can still observe `next_index` below
/// `chunk_ranges.len()` — guaranteed by [`matmul_rows_threaded`] draining
/// `chunk_ranges.len()` results from `sender`'s channel before `dot_row` or
/// `output` (the parent of every `chunk_ranges` entry) can drop.
/// `fetch_add` never hands out the same index twice, so no two pullers ever
/// touch the same slice.
fn claim_and_run_rows<Row>(
    next_index: &AtomicUsize,
    dot_row_address: usize,
    width: usize,
    chunk_ranges: &[(usize, usize, usize)],
    sender: &SyncSender<(usize, Result<(), TensorError>)>,
) where
    Row: Fn(usize, &mut [f32]) -> Result<(), TensorError>,
{
    loop {
        let index = next_index.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        #[cfg(feature = "instrument")]
        counter!(instrument::MATMUL_POOL_CLAIM_ATTEMPTS, 1);
        if index >= chunk_ranges.len() {
            return;
        }
        // SAFETY: see this function's doc comment.
        let dot_row = unsafe { &*(dot_row_address as *const Row) };
        let (chunk_start, slice_address, slice_len) = chunk_ranges[index];
        // SAFETY: unique to this chunk by construction (`split_at_mut` in
        // `matmul_rows_threaded`); the parent `output` outlives every
        // reconstructed slice per this function's doc comment.
        let chunk_output =
            unsafe { core::slice::from_raw_parts_mut(slice_address as *mut f32, slice_len) };
        let outcome = run_row_chunk(dot_row, width, chunk_start, chunk_output);
        let _ = sender.send((index, outcome));
    }
}

/// Runs one contiguous row range of a [`matmul_rows_threaded`] dispatch,
/// writing each row's `width`-wide result into its matching `chunk_output`
/// slot (`width` is 1 for every caller except the folded Q4_K wide path).
fn run_row_chunk<Row>(
    dot_row: &Row,
    width: usize,
    chunk_start: usize,
    chunk_output: &mut [f32],
) -> Result<(), TensorError>
where
    Row: Fn(usize, &mut [f32]) -> Result<(), TensorError>,
{
    #[cfg(feature = "instrument")]
    let chunk_started = instrument::read_ticks();
    // proxima-debugger diagnostic: `Instant`-elapsed wall time (below) keeps
    // accruing while this worker is off-core, so on a box carrying ambient
    // load it cannot tell "the kernel is slower in situ" apart from "this
    // thread got descheduled" (`instrument.rs`'s own doc on
    // `WORKER_CPU_NANOS` already established this for the 1->8 scaling
    // read). `thread_cpu_nanos` is the deschedule-immune peer, reused here
    // via `record_worker_cpu_nanos` -- this row-chunk path shares the same
    // `WORKER_CPU_NANOS` pool as `claim_and_run`'s elementwise/node-chunk
    // path (they DO mix within one forward pass), which is why the call
    // below tags itself `CpuWorkload::MatmulRow` rather than leaving the
    // two workloads to be summed together downstream.
    #[cfg(feature = "instrument")]
    let chunk_cpu_started = instrument::thread_cpu_nanos();
    for (offset, slot) in chunk_output.chunks_exact_mut(width).enumerate() {
        dot_row(chunk_start + offset, slot)?;
    }
    #[cfg(feature = "instrument")]
    {
        instrument::record_chunk_ticks(instrument::elapsed_ticks(chunk_started));
        instrument::record_worker_cpu_nanos(
            instrument::CpuWorkload::MatmulRow,
            instrument::thread_cpu_nanos() - chunk_cpu_started,
        );
        counter!(instrument::MATMUL_CHUNK_RUNS, 1);
    }
    Ok(())
}

// ---------------------------------------------------------------------
// `q4k-int8-dot`: int8 dot directly on packed `Q4_K` nibbles against a
// `Q8_K`-quantized activation, skipping `dot_q4k_f32`'s per-superblock
// `[f32; 256]` dequantize entirely. Still its own compile-time feature so
// the codec path stays reachable, but now ON by default alongside its
// `q5k`/`q6k` siblings -- the e2e bench this gate waited for exists: on the
// real openchat-3.5 forward, adding q5k+q6k moved `reduce_matmul_quantized`
// 513.70 -> 497.80 ms with the greedy token bit-identical over 8 runs.
// Note what that measures: 15.91 ms of the 134.84 ms those 9 tensors cost,
// an ~11.8% cut on them, NOT the near-elimination the dequantize framing
// suggests. See `proxima-tensor/docs/discipline.md` for the landing rows.
// ---------------------------------------------------------------------

/// Byte offsets into one packed `Q4_K` super-block ([`Q4K_BLOCK_BYTES`]
/// bytes), mirroring `proxima_gguf::quant::q4_k`'s private layout
/// constants -- duplicated here (not re-exported from that module) because
/// [`dot_q4k_q8k`] reads the raw bytes directly rather than calling
/// `dequantize_block`, which is the entire point: no `[f32; 256]`
/// intermediate.
#[cfg(feature = "q4k-int8-dot")]
const Q4K_D_OFFSET: usize = 0;
#[cfg(feature = "q4k-int8-dot")]
const Q4K_DMIN_OFFSET: usize = 2;
#[cfg(feature = "q4k-int8-dot")]
const Q4K_SCALES_OFFSET: usize = 4;
#[cfg(feature = "q4k-int8-dot")]
const Q4K_SCALE_BYTES: usize = 12;
#[cfg(feature = "q4k-int8-dot")]
const Q4K_QS_OFFSET: usize = Q4K_SCALES_OFFSET + Q4K_SCALE_BYTES;
/// Sub-blocks of 32 elements per `Q4_K` super-block (`QK_K/32` = 8) --
/// [`Q4K_BLOCK_ELEMENTS`] is `pub(crate)`-visible above; this is the same
/// number under the name the int8 dot's loop structure uses it by. `Q5_K`
/// shares this exact sub-block shape (`q5_k.rs`'s own module doc: "the same
/// super-block/sub-block shape... as `q4_k`"), so [`dot_q5k_q8k_block_scalar`]
/// reuses this constant rather than defining an identical `Q5K_SUB_BLOCKS`.
#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot"))]
const Q4K_SUB_BLOCKS: usize = Q4K_BLOCK_ELEMENTS / 32;

/// Bytes per `Q8_K` super-block: `f32` scale (4 bytes), plus `QK_K` `i8`
/// quants (256 bytes), plus `QK_K/16` `i16` per-16-element partial sums
/// (16 times 2 bytes = 32 bytes) -- 292 total. Mirrors ggml's own
/// `block_q8_K` byte-for-byte (`ggml-common.h:333`,
/// `static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K +
/// QK_K/16*sizeof(int16_t), ...)`) -- deliberately: [`dot_q4k_q8k`]
/// takes `activation_q8k` as raw bytes in that exact layout rather than a
/// new struct type. Per guiding-principles §1: a byte buffer already in
/// ggml's own wire shape needs no host type any more than `dot_q4k_f32`'s
/// `weight_row: &[u8]` does -- a `(d, qs, bsums)` tuple or three parallel
/// slices would ALSO work, but would require [`quantize_row_q8k`] and
/// [`dot_q4k_q8k`] to agree on three independent buffer lengths instead of
/// one, for no capability a caller gains.
// `Q8_K` is the one activation format every K-quant weight codec (`Q4_K`,
// `Q5_K`, `Q6_K`) dots against -- shared, not duplicated per format, so
// these constants and `quantize_row_q8k` below build under ANY of the
// three weight codecs' int8-dot features, not `q4k-int8-dot` alone.
#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
const Q8K_BLOCK_BYTES: usize = 4 + Q4K_BLOCK_ELEMENTS + (Q4K_BLOCK_ELEMENTS / 16) * 2;
#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
const Q8K_D_OFFSET: usize = 0;
#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
const Q8K_QS_OFFSET: usize = 4;
#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
const Q8K_BSUMS_OFFSET: usize = Q8K_QS_OFFSET + Q4K_BLOCK_ELEMENTS;
#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
const Q8K_BSUMS_COUNT: usize = Q4K_BLOCK_ELEMENTS / 16;

#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
fn f16_le_at(bytes: &[u8], offset: usize) -> f32 {
    let mut raw = [0u8; 2];
    raw.copy_from_slice(&bytes[offset..offset + 2]);
    half::f16::from_le_bytes(raw).to_f32()
}

/// Quantizes an activation vector into packed `Q8_K` bytes (`Q8K_BLOCK_BYTES`
/// per 256-element super-block) -- the one pass [`dot_q4k_q8k`]'s int8
/// mechanism needs, hoisted OUT of the per-row loop the same way this
/// module's docs already measured a conversion pipe at (52 vs 52
/// instructions, `docs/discipline.md`): paying this per row instead of
/// once per `matmul_q4k_q8k_f32` call would cost `rows`x -- 4096x at this
/// crate's real weight-matrix shapes. Ports `quantize_row_q8_K_ref`
/// (`ggml-quants.c:2471-2505`) bit-for-bit: per super-block, finds the
/// largest-magnitude element, scales by `-127/max`, rounds every element to
/// `i8` via [`proxima_gguf::quant::q4_k::nearest_int`] (the same
/// ties-to-even bit trick `Q4_K`'s own reference quantizer uses -- ggml
/// calls one `nearest_int` for every k-quant codec, not a `Q8_K`-specific
/// one), then folds each 16-element run into one `i16` partial sum
/// (`bsums`) [`dot_q4k_q8k`]'s mins correction consumes without
/// re-scanning `qs`.
///
/// `activation.len()` must be a whole multiple of `Q4K_BLOCK_ELEMENTS`
/// (256); `output.len()` must exactly equal the block count times
/// `Q8K_BLOCK_BYTES`. No allocation: `output` is caller-provided.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if either length requirement
/// above is not met.
#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
use crate::sized::MIN_QUANTIZE_BLOCKS_FOR_DISPATCH;

/// [`quantize_row_q8k`] dispatched across the cohort when a `session` is
/// open and the call's super-block count clears
/// [`MIN_QUANTIZE_BLOCKS_FOR_DISPATCH`] -- the same shape
/// [`run_elementwise_dispatch`] has to [`run_elementwise`]: every `Q8_K`
/// super-block quantizes independently (this function's own doc), so a
/// contiguous range of blocks is exactly as independent as
/// [`ElementwiseRowRound`]'s outer-position ranges. Falls straight through
/// to [`quantize_row_q8k`] whenever any gate fails: no session, too few
/// blocks, or fewer than one worker.
#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
fn quantize_row_q8k_dispatch(
    activation: &[f32],
    output: &mut [u8],
    session: Option<&MatmulSession<'_>>,
) -> Result<(), TensorError> {
    let Some(session) = session else {
        return quantize_row_q8k(activation, output);
    };
    if !activation.len().is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the q8_k super-block size",
        });
    }
    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    if output.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k output length does not match the activation block count",
        });
    }
    if block_count < MIN_QUANTIZE_BLOCKS_FOR_DISPATCH {
        return quantize_row_q8k(activation, output);
    }
    let workers = matmul_worker_count();
    if workers <= 1 {
        return quantize_row_q8k(activation, output);
    }
    let chunk_count = (workers * OVERSUBSCRIBE).min(block_count);
    let block_chunk_len = block_count.div_ceil(chunk_count);
    let mut chunk_ranges = Vec::with_capacity(chunk_count);
    let mut remaining_in = activation;
    let mut remaining_out = &mut *output;
    while !remaining_out.is_empty() {
        let take_blocks = block_chunk_len.min(remaining_out.len() / Q8K_BLOCK_BYTES);
        let (in_slice, in_rest) = remaining_in.split_at(take_blocks * Q4K_BLOCK_ELEMENTS);
        let (out_slice, out_rest) = remaining_out.split_at_mut(take_blocks * Q8K_BLOCK_BYTES);
        remaining_in = in_rest;
        remaining_out = out_rest;
        chunk_ranges.push((
            in_slice.as_ptr() as usize,
            in_slice.len(),
            out_slice.as_mut_ptr() as usize,
            out_slice.len(),
        ));
    }
    if chunk_ranges.len() < 2 {
        return quantize_row_q8k(activation, output);
    }
    let round = QuantizeRound {
        chunk_ranges: &chunk_ranges,
    };
    let report = session.run(&round);
    if report.abandoned > 0 {
        return Err(TensorError::ThreadedChunkFailed {
            chunk: report.first_abandoned.map_or(0, |chunk| chunk.0 + 1),
            reason: alloc::string::String::from("cohort member panicked while running this quantize chunk"),
        });
    }
    Ok(())
}

/// [`quantize_row_q8k_dispatch`]'s cohort dispatch shape: one round over
/// `(in_ptr, in_len, out_ptr, out_len)` block ranges, run through
/// [`CohortSession::run`]. No error path -- every range's shape was already
/// validated whole, by construction, before the round opens, so
/// [`quantize_q8k_block`] cannot fail the way a matmul row's dot product can.
#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
struct QuantizeRound<'round> {
    chunk_ranges: &'round [(usize, usize, usize, usize)],
}

#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
impl CohortRound<TensorError> for QuantizeRound<'_> {
    fn chunks(&self) -> usize {
        self.chunk_ranges.len()
    }

    fn run_chunk(&self, chunk: ChunkIndex) -> Result<(), TensorError> {
        let (in_ptr, in_len, out_ptr, out_len) = self.chunk_ranges[chunk.0];
        // SAFETY: unique to this chunk by construction (`split_at`/
        // `split_at_mut` in `quantize_row_q8k_dispatch` before the round
        // starts); the parent `activation`/`output` outlive every
        // reconstructed slice because `CohortSession::run` does not return
        // until every member has reported done.
        let in_slice = unsafe { core::slice::from_raw_parts(in_ptr as *const f32, in_len) };
        // SAFETY: same argument as `in_slice` above, mutable side.
        let out_slice = unsafe { core::slice::from_raw_parts_mut(out_ptr as *mut u8, out_len) };
        for (block, out_block) in in_slice
            .as_chunks::<Q4K_BLOCK_ELEMENTS>()
            .0
            .iter()
            .zip(out_slice.as_chunks_mut::<Q8K_BLOCK_BYTES>().0)
        {
            quantize_q8k_block(block, out_block);
        }
        Ok(())
    }
}

#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
pub fn quantize_row_q8k(activation: &[f32], output: &mut [u8]) -> Result<(), TensorError> {
    if !activation.len().is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the q8_k super-block size",
        });
    }
    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    if output.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k output length does not match the activation block count",
        });
    }
    for (chunk, out_block) in activation
        .as_chunks::<Q4K_BLOCK_ELEMENTS>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<Q8K_BLOCK_BYTES>().0)
    {
        quantize_q8k_block(chunk, out_block);
    }
    Ok(())
}

#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
fn quantize_q8k_block(chunk: &[f32], out_block: &mut [u8]) {
    let mut amax = 0.0f32;
    let mut max = 0.0f32;
    for &value in chunk {
        let absolute = value.abs();
        if absolute > amax {
            amax = absolute;
            max = value;
        }
    }
    if amax == 0.0 {
        out_block.fill(0);
        return;
    }

    let iscale = -127.0f32 / max;
    let mut levels = [0i8; Q4K_BLOCK_ELEMENTS];
    for (level, &value) in levels.iter_mut().zip(chunk.iter()) {
        *level = proxima_gguf::quant::q4_k::nearest_int(iscale * value).min(127) as i8;
    }

    let scale = 1.0 / iscale;
    out_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4].copy_from_slice(&scale.to_le_bytes());

    let qs = &mut out_block[Q8K_QS_OFFSET..Q8K_QS_OFFSET + Q4K_BLOCK_ELEMENTS];
    for (slot, &level) in qs.iter_mut().zip(levels.iter()) {
        *slot = level.cast_unsigned();
    }

    let bsums_region = &mut out_block[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];
    for (sixteen, bytes) in levels.as_chunks::<16>().0.iter().zip(bsums_region.as_chunks_mut::<2>().0) {
        let sum: i16 = sixteen.iter().map(|&level| i16::from(level)).sum();
        bytes.copy_from_slice(&sum.to_le_bytes());
    }
}

/// [`quantize_q8k_block`]'s inverse: one packed `Q8_K` super-block back to
/// its `f32` levels (`scale * level`), the same `d`/`qs` fields
/// [`dot_q4k_q8k_block_scalar`] already reads (`bsums` is a fused-path-only
/// correction term, unused by a plain dequantize-then-fold). Exists for
/// [`QuantDot::Unfused`]'s own `In`/`Out` shape to match [`QuantDot::Fused`]
/// exactly: both take the SAME packed `Q8_K` activation bytes, so an
/// unfused reference needs a way back to `f32` for those bytes, not just
/// for the weight row (`proxima_gguf`'s codec `dequantize` already covers
/// the weight side).
///
/// # Panics
/// If `block.len() != Q8K_BLOCK_BYTES` or `output.len() != Q4K_BLOCK_ELEMENTS`.
#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
fn dequantize_q8k_block(block: &[u8], output: &mut [f32]) {
    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let scale = f32::from_le_bytes(d_bytes);
    let qs = &block[Q8K_QS_OFFSET..Q8K_QS_OFFSET + Q4K_BLOCK_ELEMENTS];
    for (level, out) in qs.iter().zip(output.iter_mut()) {
        *out = scale * f32::from(level.cast_signed());
    }
}

/// A `Q4_K`/`Q5_K`/`Q6_K` weight row's dot product against a packed `Q8_K`
/// activation row, as a [`Pipe`]: `In` = the activation row's packed bytes
/// ([`quantize_row_q8k`]'s own output shape), `Out` = the row's dot
/// product, `Err` = the same [`TensorError`] the underlying kernel already
/// raises on a malformed shape. The weight row travels with the pipe value
/// itself as a [`QuantizedBlock`] -- the codec (`Q4K`/`Q5K`/`Q6K`) is the
/// variant `QuantizedBlock` already carries, not a second marker type
/// minted to say the same thing again.
///
/// [`Self::Fused`] calls straight into the codec's own int8 kernel
/// ([`dot_q4k_q8k`]/[`dot_q5k_q8k`]/[`dot_q6k_q8k`]) -- packed nibbles in,
/// one integer accumulate, no `f32` intermediate. [`Self::Unfused`]
/// dequantizes both operands to `f32` first (the weight row via
/// `proxima_gguf`'s own codec `dequantize`, the activation row via this
/// module's private `dequantize_q8k_block`) and folds with a plain `f32`
/// multiply-add -- the incumbent shape this crate's own parity tests
/// already hold as ground truth (see
/// `matmul_q4k_f32_matches_dequantize_then_f32_matmul`).
///
/// Selecting between the two at a call site with no branch on the caller's
/// part is the reason this is one enum with one [`Pipe`] impl rather than
/// two free functions: `QuantDot::Fused(block).call(q8k_row)` and
/// `QuantDot::Unfused(block).call(q8k_row)` are the same shape, so a caller
/// choosing fused-vs-unfused per matmul row (a build-time feature gate or a
/// measured per-target decision) holds either behind one type, matched once
/// inside `call` rather than at every call site.
#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
pub enum QuantDot<'a> {
    Fused(QuantizedBlock<'a>),
    Unfused(QuantizedBlock<'a>),
}

#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
impl<'a> Pipe for QuantDot<'a> {
    type In = &'a [u8];
    type Out = f32;
    type Err = TensorError;

    fn call(&self, activation_q8k: &'a [u8]) -> impl Future<Output = Result<f32, TensorError>> {
        let result = match self {
            Self::Fused(block) => fused_quant_dot(*block, activation_q8k),
            Self::Unfused(block) => unfused_quant_dot(*block, activation_q8k),
        };
        async move { result }
    }
}

/// [`QuantDot::Fused`]'s own body: select the codec's int8 kernel by
/// matching [`QuantizedBlock`]'s variant, the same table
/// [`dot_fn_for`] already builds for the `cohort-staged-graph` batching
/// path -- this is the non-batched, single-row counterpart. A codec whose
/// int8-dot feature is not compiled in (or a non-K-quant variant like
/// `Q8_0`/`Float16`) is an honest [`TensorError::NotLowerable`], never a
/// silent fallback to a different codec's kernel.
#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
fn fused_quant_dot(block: QuantizedBlock<'_>, activation_q8k: &[u8]) -> Result<f32, TensorError> {
    match block {
        #[cfg(feature = "q4k-int8-dot")]
        QuantizedBlock::Q4K(bytes) => dot_q4k_q8k(bytes, activation_q8k),
        #[cfg(feature = "q5k-int8-dot")]
        QuantizedBlock::Q5K(bytes) => dot_q5k_q8k(bytes, activation_q8k),
        #[cfg(feature = "q6k-int8-dot")]
        QuantizedBlock::Q6K(bytes) => dot_q6k_q8k(bytes, activation_q8k),
        _ => Err(TensorError::NotLowerable {
            node: NodeId(0),
            reason: "QuantDot::Fused only supports a K-quant codec whose int8-dot feature is enabled",
        }),
    }
}

/// [`QuantDot::Unfused`]'s own body: dequantize both operands to `f32`
/// (weight row via `proxima_gguf`'s codec `dequantize`, activation row via
/// [`dequantize_q8k_block`]) and fold with a plain multiply-add. Every
/// length check mirrors [`dot_q4k_q8k`]'s own -- this path takes the
/// identical `In` shape, so it must reject the identical malformed shapes.
#[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
fn unfused_quant_dot(block: QuantizedBlock<'_>, activation_q8k: &[u8]) -> Result<f32, TensorError> {
    let (weight_bytes, block_bytes, qk_k): (&[u8], usize, usize) = match block {
        QuantizedBlock::Q4K(bytes) => (bytes, proxima_gguf::quant::q4_k::BLOCK_BYTES, proxima_gguf::quant::q4_k::QK_K),
        QuantizedBlock::Q5K(bytes) => (bytes, proxima_gguf::quant::q5_k::BLOCK_BYTES, proxima_gguf::quant::q5_k::QK_K),
        QuantizedBlock::Q6K(bytes) => (bytes, proxima_gguf::quant::q6_k::BLOCK_BYTES, proxima_gguf::quant::q6_k::QK_K),
        _ => {
            return Err(TensorError::NotLowerable {
                node: NodeId(0),
                reason: "QuantDot::Unfused only supports a K-quant codec (Q4_K/Q5_K/Q6_K)",
            });
        }
    };
    if !weight_bytes.len().is_multiple_of(block_bytes) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of its codec's block size",
        });
    }
    let block_count = weight_bytes.len() / block_bytes;
    if activation_q8k.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k activation length does not match the weight row's block count",
        });
    }

    let elements = block_count * qk_k;
    let mut weight_f32 = vec![0.0f32; elements];
    let dequantize_result = match block {
        QuantizedBlock::Q4K(bytes) => proxima_gguf::quant::q4_k::dequantize(bytes, &mut weight_f32),
        QuantizedBlock::Q5K(bytes) => proxima_gguf::quant::q5_k::dequantize(bytes, &mut weight_f32),
        QuantizedBlock::Q6K(bytes) => proxima_gguf::quant::q6_k::dequantize(bytes, &mut weight_f32),
        _ => unreachable!("codec already matched above"),
    };
    dequantize_result.map_err(|_| TensorError::QuantizedShapeMismatch {
        reason: "weight row failed to dequantize despite passing its own shape check",
    })?;

    let mut activation_f32 = vec![0.0f32; elements];
    for (block_bytes, block_f32) in activation_q8k
        .as_chunks::<Q8K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation_f32.as_chunks_mut::<Q4K_BLOCK_ELEMENTS>().0)
    {
        dequantize_q8k_block(block_bytes, block_f32);
    }

    Ok(weight_f32.iter().zip(&activation_f32).map(|(weight, value)| weight * value).sum())
}

/// One `Q4_K`-weight-row x `Q8_K`-activation int8 dot product --
/// `dot_q4k_f32`'s packed-arithmetic sibling: same `weight_row` shape
/// (raw `Q4_K` bytes, a whole number of `Q4K_BLOCK_BYTES` super-blocks),
/// but `activation_q8k` is [`quantize_row_q8k`]'s packed `Q8_K` bytes
/// instead of a plain `f32` slice, and the fold is an integer dot on the
/// packed 4-bit nibbles rather than an `f32` multiply-add over a
/// dequantized scratch buffer. `dot_q4k_f32` is left untouched as the
/// correct codec path for non-matmul consumers (module-level comment
/// above) -- this is an additional arm, not a replacement.
///
/// Caches `std::is_x86_feature_detected!("avx2")`, probed once per process
/// life -- the same [`OnceLock`] shape [`matmul_worker_count`] already uses
/// for `PROXIMA_MATMUL_WORKERS`/`performance_core_count`, so
/// [`dot_q4k_q8k`]'s per-block hot loop never repeats the CPUID probe the
/// detection macro performs. Only compiled when the build itself did NOT
/// already declare AVX2 present at compile time (`q4k_avx2` off): a
/// `-C target-feature=+avx2` / `-C target-cpu=native` build already knows
/// the answer statically and calls [`dot_q4k_q8k_block_avx2`] unconditionally
/// (`dot_q4k_q8k`'s `q4k_avx2` arm), skipping this check entirely -- this
/// function exists for the DEFAULT `cargo build --release` on an ordinary
/// x86_64 host, which has no reason to know at compile time whether the
/// CPU it will run on has AVX2 (essentially universal since 2013, but not
/// guaranteed by the bare `x86_64` target triple).
#[cfg(all(target_arch = "x86_64", feature = "q4k-int8-dot", not(q4k_avx2)))]
fn avx2_runtime_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}

/// Dispatches to `dot_q4k_q8k_block_neon_dotprod` when built with the
/// `q4k_dotprod` cfg (`build.rs`: every aarch64 target this workspace
/// builds for), to `dot_q4k_q8k_block_avx2` when built with the
/// `q4k_avx2` cfg (`build.rs`: an x86 target whose `CARGO_CFG_TARGET_FEATURE`
/// lists `avx2` -- unlike aarch64's `FEAT_DotProd`, AVX2 is NOT in the x86-64
/// baseline ISA, so this one is opt-in via `-C target-feature=+avx2` /
/// `-C target-cpu`, not implied by the target triple alone), to the SAME
/// `dot_q4k_q8k_block_avx2` chosen at RUNTIME via
/// `avx2_runtime_available` on a plain x86_64 build that did not opt in
/// at compile time (so a default `cargo build --release` on a modern
/// x86_64 host still gets the fast kernel instead of silently falling back
/// to scalar), and to the portable `dot_q4k_q8k_block_scalar` everywhere
/// else (or when the runtime probe reports AVX2 absent). All three/four
/// compute the identical mechanism -- read 4.5 bits/weight off `weight_row`
/// and do the multiply-accumulate against `Q8_K` `i8` activations directly,
/// no `f32` intermediate at all -- the NEON arm is an acceleration of that
/// mechanism (`vdotq_s32`'s 16-lane int8 dot via inline `sdot`,
/// `core::arch::aarch64::vdotq_s32` itself being unstable on this toolchain
/// -- `stdarch_neon_dotprod`), and the AVX2 arm is a second, independent
/// acceleration of it (`_mm256_maddubs_epi16` + `_mm256_madd_epi16`'s
/// 32-lane unsigned-times-signed int8 dot), not a different one.
///
/// AVX-512 VNNI (`_mm512_dpbusd_epi32`) and AVX-VNNI
/// (`_mm256_dpbusd_epi32`) would each do this same dot in one instruction
/// instead of AVX2's maddubs+madd pair -- not implemented here: this crate
/// cannot execute either on its aarch64-darwin dev boxes, so shipping an
/// unverified single-instruction accumulation path (whose saturation
/// semantics need re-checking against `dot_q4k_q8k_block_scalar` byte for
/// byte, not assumed) is deferred rather than guessed at. AVX2 is the
/// floor that matters (present on essentially every x86_64 CPU since
/// 2013); the follow-up is a `dpbusd`-based
/// `dot_q4k_q8k_block_avxvnni`/`_avx512vnni` pair selected ahead of the
/// AVX2 arm in `avx2_runtime_available`'s priority order, verified on
/// real VNNI hardware before it lands.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of `Q4K_BLOCK_BYTES`, or `activation_q8k.len()` does
/// not equal the row's block count times `Q8K_BLOCK_BYTES`.
#[cfg(feature = "q4k-int8-dot")]
pub fn dot_q4k_q8k(weight_row: &[u8], activation_q8k: &[u8]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q4K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q4_k block size",
        });
    }
    let block_count = weight_row.len() / Q4K_BLOCK_BYTES;
    if activation_q8k.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k activation length does not match the weight row's block count",
        });
    }

    #[cfg(all(target_arch = "x86_64", not(q4k_avx2)))]
    let use_avx2_runtime = avx2_runtime_available();

    let mut acc = 0.0f32;
    for (weight_block, q8k_block) in weight_row
        .as_chunks::<Q4K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation_q8k.as_chunks::<Q8K_BLOCK_BYTES>().0)
    {
        #[cfg(q4k_dotprod)]
        // SAFETY: `q4k_dotprod` is emitted by build.rs only for aarch64
        // targets, all of which carry FEAT_DotProd (build.rs's own doc).
        let block_sum = unsafe { dot_q4k_q8k_block_neon_dotprod(weight_block, q8k_block) };
        #[cfg(all(q4k_avx2, not(q4k_dotprod)))]
        // SAFETY: `q4k_avx2` is emitted by build.rs only when
        // `CARGO_CFG_TARGET_FEATURE` actually lists `avx2` (build.rs's own
        // doc) -- the caller opted the build itself into AVX2, so the
        // instructions this block issues are guaranteed present.
        let block_sum = unsafe { dot_q4k_q8k_block_avx2(weight_block, q8k_block) };
        #[cfg(all(target_arch = "x86_64", not(q4k_dotprod), not(q4k_avx2)))]
        let block_sum = if use_avx2_runtime {
            // SAFETY: `use_avx2_runtime` is true only when
            // `avx2_runtime_available` confirmed
            // `std::is_x86_feature_detected!("avx2")` before this loop
            // started.
            unsafe { dot_q4k_q8k_block_avx2(weight_block, q8k_block) }
        } else {
            dot_q4k_q8k_block_scalar(weight_block, q8k_block)
        };
        #[cfg(not(any(q4k_dotprod, q4k_avx2, target_arch = "x86_64")))]
        let block_sum = dot_q4k_q8k_block_scalar(weight_block, q8k_block);
        acc += block_sum;
    }
    Ok(acc)
}

/// [`dot_q4k_q8k`] with the dispatch forced to
/// `dot_q4k_q8k_block_scalar` regardless of `q4k_dotprod` -- the "what
/// does portable packing alone buy" measurement the discipline log's
/// packed-kernel row reports standalone, next to the `vdotq_s32`-
/// accelerated number `dot_q4k_q8k` itself produces on an aarch64 build.
/// Also what non-aarch64 targets (`cargo check --target
/// x86_64-unknown-linux-gnu`) actually call, via `dot_q4k_q8k`'s own
/// `not(q4k_dotprod)` arm -- this function exists so that arm's code path
/// stays reachable, and separately benchable, from an aarch64 host too.
///
/// # Errors
/// Same as [`dot_q4k_q8k`].
#[cfg(feature = "q4k-int8-dot")]
pub fn dot_q4k_q8k_portable(weight_row: &[u8], activation_q8k: &[u8]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q4K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q4_k block size",
        });
    }
    let block_count = weight_row.len() / Q4K_BLOCK_BYTES;
    if activation_q8k.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k activation length does not match the weight row's block count",
        });
    }

    let mut acc = 0.0f32;
    for (weight_block, q8k_block) in weight_row
        .as_chunks::<Q4K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation_q8k.as_chunks::<Q8K_BLOCK_BYTES>().0)
    {
        acc += dot_q4k_q8k_block_scalar(weight_block, q8k_block);
    }
    Ok(acc)
}

/// The portable packed-nibble x `Q8_K` int8 dot -- no dequantize pass, no
/// `f32` intermediate, no architecture intrinsics. This is the mechanism
/// itself (reading 4.5 bits/weight off `weight_block` instead of the 32
/// bits/weight a decoded `f32` row costs, 7.11x less traffic than
/// `dot_q4k_f32`'s scratch buffer); [`dot_q4k_q8k_block_neon_dotprod`]
/// accelerates this same computation with `vdotq_s32`, it does not replace
/// it -- every non-aarch64 target this crate builds for (the
/// `cargo check --target x86_64-unknown-linux-gnu` gate cell) runs this
/// function, not a stand-in nobody exercises.
///
/// Ports the scalar body of `ggml_vec_dot_q4_K_q8_K`
/// (`ggml-cpu/quants.c:515`, the `#else` arm every architecture file's own
/// vectorized version specializes): per sub-block of 32, unpack its 6-bit
/// `(scale, min)` pair via [`proxima_gguf::quant::q4_k::get_scale_min_k4`],
/// dot the sub-block's 32 nibbles (0..15, an unsigned weight code -- never
/// sign-extended) against the matching 32 `Q8_K` `i8` activations, scale
/// by the sub-block's 6-bit scale code; separately, the mins correction
/// sums each sub-block's `Q8_K` `bsums` pair times its 6-bit min code.
/// `d`/`dmin` (the super-block's own `f16` scale pair) and the `Q8_K`
/// block's `f32` scale multiply in only at the very end -- two `f32`
/// operations total per 256-element super-block, exactly matching this
/// component's module doc.
#[cfg(feature = "q4k-int8-dot")]
fn dot_q4k_q8k_block_scalar(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q4K_D_OFFSET);
    let dmin_weight = f16_le_at(weight_block, Q4K_DMIN_OFFSET);
    let mut scales = [0u8; Q4K_SCALE_BYTES];
    scales.copy_from_slice(&weight_block[Q4K_SCALES_OFFSET..Q4K_SCALES_OFFSET + Q4K_SCALE_BYTES]);
    let qs = &weight_block[Q4K_QS_OFFSET..Q4K_QS_OFFSET + Q4K_BLOCK_ELEMENTS / 2];

    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);
    let activation_qs = &q8k_block[Q8K_QS_OFFSET..Q8K_QS_OFFSET + Q4K_BLOCK_ELEMENTS];
    let bsums = &q8k_block[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];

    let mut sumi = 0i32;
    let mut mins_correction = 0i32;
    for sub_block in 0..Q4K_SUB_BLOCKS {
        let (scale_code, min_code) = proxima_gguf::quant::q4_k::get_scale_min_k4(sub_block, &scales);

        let bsum_lo = i16::from_le_bytes([bsums[sub_block * 4], bsums[sub_block * 4 + 1]]);
        let bsum_hi = i16::from_le_bytes([bsums[sub_block * 4 + 2], bsums[sub_block * 4 + 3]]);
        mins_correction += i32::from(bsum_lo + bsum_hi) * i32::from(min_code);

        let byte_base = (sub_block / 2) * 32;
        let is_high_nibble = sub_block % 2 == 1;
        let activation_base = sub_block * 32;
        let mut partial = 0i32;
        for offset in 0..32 {
            let byte = qs[byte_base + offset];
            let nibble = i32::from(if is_high_nibble { byte >> 4 } else { byte & 0x0F });
            let activation_value = i32::from(activation_qs[activation_base + offset].cast_signed());
            partial += nibble * activation_value;
        }
        sumi += partial * i32::from(scale_code);
    }

    let d = activation_scale * d_weight;
    let dmin = activation_scale * dmin_weight;
    d.mul_add(sumi as f32, -(dmin * mins_correction as f32))
}

/// Issues the ARM `FEAT_DotProd` `sdot` instruction directly via inline
/// asm rather than `core::arch::aarch64::vdotq_s32` -- that safe intrinsic
/// is gated behind the unstable `stdarch_neon_dotprod` feature on this
/// toolchain (probed against `rustc 1.97.1`; ggml's own C `ggml_vdotq_s32`
/// wrapper is the exact analogue this mirrors, `ggml-cpu-impl.h:312-321`).
/// `acc + sum over 4 lanes of (a[4i..4i+4] . b[4i..4i+4])` per output lane,
/// four independent lanes -- the standard armv8.2 `SDOT (vector)` encoding.
///
/// # Safety
/// Caller guarantees `FEAT_DotProd` is available -- this crate's `build.rs`
/// only ever calls this function under the `q4k_dotprod` cfg, which it
/// emits solely for aarch64 targets (see that cfg's doc). Shared by every
/// K-quant codec's `_block_neon_dotprod` kernel (`Q4_K`/`Q5_K`/`Q6_K`), not
/// duplicated per format -- the instruction itself has no codec-specific
/// behavior.
#[cfg(all(
    target_arch = "aarch64",
    any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot")
))]
#[target_feature(enable = "dotprod")]
#[inline]
unsafe fn sdot_s32(acc: core::arch::aarch64::int32x4_t, a: core::arch::aarch64::int8x16_t, b: core::arch::aarch64::int8x16_t) -> core::arch::aarch64::int32x4_t {
    // SAFETY: caller-guaranteed FEAT_DotProd (this fn's own doc); operands
    // are NEON vector registers, `options(pure, nomem, nostack)` matches
    // that no memory is touched and the instruction has no side effects.
    unsafe {
        let result: core::arch::aarch64::int32x4_t;
        core::arch::asm!(
            "sdot {result:v}.4s, {a:v}.16b, {b:v}.16b",
            result = inlateout(vreg) acc => result,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        );
        result
    }
}

/// [`dot_q4k_q8k_block_scalar`]'s mechanism, `vdotq_s32`-accelerated:
/// identical per-sub-block structure (unpack scale/min, dot 32 nibbles
/// against 32 `Q8_K` activations, scale, accumulate; mins correction
/// identical), but the 32-nibble dot is two 16-lane `sdot_s32` calls
/// (`ggml_vec_dot_q4_K_q8_K`'s `__ARM_NEON` arm, `arch/arm/quants.c:2408-
/// 2427`) instead of a 32-iteration scalar loop -- low/high nibbles split
/// in-register via `vandq_u8`/`vshrq_n_u8`, never written to memory. The
/// scale/min codes are unpacked ONCE per super-block by the same bit-trick
/// ggml's NEON arm uses (`arch/arm/quants.c:2367-2381`) rather than calling
/// [`proxima_gguf::quant::q4_k::get_scale_min_k4`] per sub-block -- same
/// identity as that function (see its own doc), just the vectorized route
/// instead of the scalar one. The mins correction (`sum(bsums[i] *
/// min_code[i])`) is reduced with `vpaddq_s16`/`vmull_s16`/`vaddvq_s32`
/// mirroring `arch/arm/quants.c:2380-2387`, in place of the auto-vectorized
/// scalar loop this replaced.
///
/// # Safety
/// Caller guarantees `FEAT_DotProd`; `weight_block.len() ==
/// Q4K_BLOCK_BYTES` and `q8k_block.len() == Q8K_BLOCK_BYTES` (both
/// [`dot_q4k_q8k`]'s own `chunks_exact` calls already guarantee before
/// calling this).
#[cfg(all(q4k_dotprod, feature = "q4k-int8-dot"))]
unsafe fn dot_q4k_q8k_block_neon_dotprod(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q4K_D_OFFSET);
    let dmin_weight = f16_le_at(weight_block, Q4K_DMIN_OFFSET);
    let mut scales = [0u8; Q4K_SCALE_BYTES];
    scales.copy_from_slice(&weight_block[Q4K_SCALES_OFFSET..Q4K_SCALES_OFFSET + Q4K_SCALE_BYTES]);

    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);
    let bsums = &q8k_block[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];

    // SAFETY: caller-guaranteed FEAT_DotProd (this fn's own doc);
    // `mins_correction_neon`'s own preconditions (`scales`/`bsums` lengths)
    // are met by the fixed-size array and the slice sized above.
    let (scale_lo, scale_hi, mins_correction) = unsafe { mins_correction_neon(&scales, bsums) };

    // SAFETY: caller-guaranteed FEAT_DotProd; `q4_ptr`/`q8_ptr` each walk
    // exactly `Q4K_BLOCK_ELEMENTS / 2` / `Q4K_BLOCK_ELEMENTS` bytes across
    // the 4 unrolled sub-block pairs below, both within the slices' checked
    // bounds.
    unsafe {
        let m4b = vdupq_n_u8(0x0f);
        let mzero = vdupq_n_s32(0);
        let q4_base = weight_block[Q4K_QS_OFFSET..].as_ptr();
        let q8_base = q8k_block[Q8K_QS_OFFSET..].as_ptr().cast::<i8>();

        let mut sumi1: i32 = 0;
        let mut sumi2: i32 = 0;
        // Hand-unrolled (not `for j in 0..4`): each `2 * j` / `2 * j + 1`
        // scale index below is a literal so `scale_byte` compiles to a
        // single `ubfx` on a register-resident word, matching ggml's
        // `arch/arm/quants.c:2408-2427` `utmp`-in-registers shape, instead
        // of round-tripping a `[u8; 8]` through the stack the way indexing
        // a runtime `j` into an array would.
        macro_rules! sub_block_pair {
            ($q4_offset:expr, $q8_offset:expr, $scale_word:expr) => {{
                let q4bits = vld1q_u8_x2(q4_base.add($q4_offset));
                let lo0 = vreinterpretq_s8_u8(vandq_u8(q4bits.0, m4b));
                let lo1 = vreinterpretq_s8_u8(vandq_u8(q4bits.1, m4b));
                let q8_lo = vld1q_s8_x2(q8_base.add($q8_offset));
                let partial_lo = sdot_s32(sdot_s32(mzero, lo0, q8_lo.0), lo1, q8_lo.1);
                sumi1 += vaddvq_s32(partial_lo) * scale_byte($scale_word, 0);

                let hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.0, 4));
                let hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.1, 4));
                let q8_hi = vld1q_s8_x2(q8_base.add($q8_offset + 32));
                let partial_hi = sdot_s32(sdot_s32(mzero, hi0, q8_hi.0), hi1, q8_hi.1);
                sumi2 += vaddvq_s32(partial_hi) * scale_byte($scale_word, 1);
            }};
        }
        sub_block_pair!(0, 0, scale_lo);
        sub_block_pair!(32, 64, scale_lo >> 16);
        sub_block_pair!(64, 128, scale_hi);
        sub_block_pair!(96, 192, scale_hi >> 16);

        let d = activation_scale * d_weight;
        let dmin = activation_scale * dmin_weight;
        d.mul_add((sumi1 + sumi2) as f32, -(dmin * mins_correction as f32))
    }
}

/// [`dot_q4k_q8k_block_neon_dotprod`]'s scale-unpack and mins-correction
/// step, factored out so the test below can exercise it in isolation
/// against [`proxima_gguf::quant::q4_k::get_scale_min_k4`]'s scalar route
/// to the identical quantity. Unpacks all 8 sub-blocks' 6-bit scale/min
/// codes ONCE via the same bit-trick ggml's NEON arm uses
/// (`arch/arm/quants.c:2367-2381`), then reduces `sum(bsums[i] *
/// min_code[i])` with `vpaddq_s16`/`vmull_s16`/`vaddvq_s32`
/// (`arch/arm/quants.c:2380-2387`) in place of a scalar loop.
///
/// Returns `(scale_lo, scale_hi, mins_correction)`: `scale_lo`/`scale_hi`
/// each pack four sub-blocks' masked scale bytes little-endian (byte `k` of
/// `scale_lo` is sub-block `k`'s scale, byte `k` of `scale_hi` is sub-block
/// `k + 4`'s), matching `get_scale_min_k4(sub_block, scales).0` byte for
/// byte once unpacked via [`scale_byte`]; `mins_correction` is the widened
/// `i32` reduction, matching `sum(get_scale_min_k4(i, scales).1 as i32 *
/// bsum_pair_sum(i) as i32)` for `i in 0..Q4K_SUB_BLOCKS`. Returned as two
/// plain `u32` words rather than a `[u8; 8]` so callers extract each byte
/// with a register-resident `ubfx`-style shift ([`scale_byte`]) instead of
/// indexing an array the compiler may otherwise round-trip through the
/// stack (bounds-check codegen on a non-constant index).
///
/// `Q5_K` shares this exact 12-byte scale/min layout and the same
/// [`Q4K_SUB_BLOCKS`]/`bsums` shape (`arch/arm/quants.c:2611-2622` mirrors
/// `arch/arm/quants.c:2367-2381` byte for byte), so
/// [`dot_q5k_q8k_block_neon_dotprod`] calls this same function directly
/// rather than duplicating it.
///
/// # Safety
/// Caller guarantees `FEAT_DotProd`; `bsums.len() == Q8K_BSUMS_COUNT * 2`
/// (16 `i16`s) so the two 8-lane `vld1q_s16` loads stay in bounds.
#[cfg(all(q4k_dotprod, any(feature = "q4k-int8-dot", feature = "q5k-int8-dot")))]
unsafe fn mins_correction_neon(scales: &[u8; Q4K_SCALE_BYTES], bsums: &[u8]) -> (u32, u32, i32) {
    // Same masks as `get_scale_min_k4`'s scalar bit-trick, applied once to
    // the whole 12-byte field instead of once per sub-block per call.
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;
    let word_0 = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
    let word_1 = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
    let word_2 = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);
    let mins_lo = word_1 & KMASK1;
    let mins_hi = ((word_2 >> 4) & KMASK2) | (((word_1 >> 6) & KMASK3) << 4);
    let scale_hi = (word_2 & KMASK2) | (((word_0 >> 6) & KMASK3) << 4);
    let scale_lo = word_0 & KMASK1;

    // SAFETY: caller-guaranteed FEAT_DotProd; caller-guaranteed
    // `bsums.len() == 32` bytes (16 `i16`s), so the two 8-lane `vld1q_s16`
    // loads below stay in bounds.
    let mins_correction = unsafe {
        let mins_words = [mins_lo, mins_hi];
        let mins8 = vld1_u32(mins_words.as_ptr());
        let mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        let bsums_ptr = bsums.as_ptr().cast::<i16>();
        let q8sums = vpaddq_s16(vld1q_s16(bsums_ptr), vld1q_s16(bsums_ptr.add(8)));
        let mins_product = vaddq_s32(
            vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
            vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
        );
        vaddvq_s32(mins_product)
    };

    (scale_lo, scale_hi, mins_correction)
}

/// Extracts byte `index` (`0..=3`) of a [`mins_correction_neon`]-returned
/// `scale_lo`/`scale_hi` word as `i32`, ready to multiply against a
/// `vaddvq_s32` dot-partial. `index` is a literal at
/// [`dot_q4k_q8k_block_neon_dotprod`]'s call sites, so this compiles to one
/// `ubfx` on a register the value already lives in -- ggml's
/// `arch/arm/quants.c` equivalent keeps `utmp` in registers the same way
/// and extracts with the same instruction.
#[cfg(all(q4k_dotprod, any(feature = "q4k-int8-dot", feature = "q5k-int8-dot")))]
#[inline(always)]
fn scale_byte(word: u32, index: u32) -> i32 {
    ((word >> (index * 8)) & 0xff) as i32
}

/// Horizontal sum of an `__m256i` holding eight packed `i32` lanes down to
/// one scalar -- the standard AVX2 idiom (extract the high 128 bits, add to
/// the low 128, fold 64-then-32), used by [`dot_q4k_q8k_block_avx2`] to
/// collapse [`_mm256_madd_epi16`]'s eight-lane pairwise-sum result into the
/// same single `i32` partial-dot value [`dot_q4k_q8k_block_scalar`]'s
/// 32-iteration scalar loop accumulates directly.
///
/// # Safety
/// Caller guarantees AVX2 is available -- every intrinsic this function
/// calls (`_mm256_extracti128_si256`/`_mm256_castsi256_si128` need AVX;
/// `_mm_add_epi32`/`_mm_unpackhi_epi64`/`_mm_shuffle_epi32`/
/// `_mm_cvtsi128_si32` are SSE2, x86-64 baseline) is a "safe" intrinsic
/// function under this toolchain's target-feature rules once the enclosing
/// function's `#[target_feature(enable = "avx2")]` statically guarantees
/// the feature, which is why the body below needs no inner `unsafe {}` --
/// this function itself stays `unsafe fn` only so its signature doesn't
/// imply it is callable outside an AVX2-guaranteed build. Compiled on every
/// x86_64 target (`target_arch` gate, not `q4k_avx2`): `dot_q4k_q8k`'s
/// runtime-dispatch arm calls this on a plain `cargo build` x86_64 host too,
/// gated by `std::is_x86_feature_detected!("avx2")` at the call site rather
/// than by a build-time flag -- see `avx2_runtime_available`'s own doc.
#[cfg(all(target_arch = "x86_64", feature = "q4k-int8-dot"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_epi32_avx2(v: __m256i) -> i32 {
    let high = _mm256_extracti128_si256(v, 1);
    let low = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(low, high);
    let high64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, high64);
    let high32 = _mm_shuffle_epi32(sum64, 0b01);
    let sum32 = _mm_add_epi32(sum64, high32);
    _mm_cvtsi128_si32(sum32)
}

/// [`dot_q4k_q8k_block_scalar`]'s mechanism, AVX2-accelerated: identical
/// per-sub-block structure (unpack scale/min via `get_scale_min_k4`, dot 32
/// nibbles against 32 `Q8_K` activations, scale, accumulate; mins correction
/// identical), but the 32-nibble dot is one `_mm256_maddubs_epi16` (32-lane
/// unsigned-nibble x signed-`i8` multiply, pairwise-summed to 16 `i16`
/// lanes) followed by `_mm256_madd_epi16` against an all-ones vector
/// (pairwise-summed to 8 `i32` lanes) and [`hsum_epi32_avx2`], instead of a
/// 32-iteration scalar loop -- low/high nibbles split via
/// `_mm256_and_si256`/`_mm256_srli_epi16` exactly as
/// `ggml_vec_dot_q4_K_q8_K`'s `__AVX2__` arm does
/// (`ggml-cpu/arch/x86/quants.c`), but WITHOUT that function's
/// `_mm256_shuffle_epi8`-based scale broadcast: this kernel multiplies each
/// 32-lane partial dot by its scalar `i32` scale code AFTER the horizontal
/// sum, the same order [`dot_q4k_q8k_block_scalar`] uses, rather than
/// folding the scale into the SIMD `madd` itself -- integer multiplication
/// distributes over integer addition exactly, so this is the identical
/// mechanism at the identical resulting value, just without minting a
/// scale-shuffle table this component doesn't otherwise need.
///
/// # Safety
/// Caller guarantees AVX2 is available; `weight_block.len() ==
/// Q4K_BLOCK_BYTES` and `q8k_block.len() == Q8K_BLOCK_BYTES` (both
/// [`dot_q4k_q8k`]'s own `chunks_exact` calls already guarantee before
/// calling this). Compiled on every x86_64 target -- see
/// [`hsum_epi32_avx2`]'s own doc for why this is `target_arch`-gated rather
/// than `q4k_avx2`-gated.
#[cfg(all(target_arch = "x86_64", feature = "q4k-int8-dot"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4k_q8k_block_avx2(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q4K_D_OFFSET);
    let dmin_weight = f16_le_at(weight_block, Q4K_DMIN_OFFSET);
    let mut scales = [0u8; Q4K_SCALE_BYTES];
    scales.copy_from_slice(&weight_block[Q4K_SCALES_OFFSET..Q4K_SCALES_OFFSET + Q4K_SCALE_BYTES]);

    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);
    let bsums = &q8k_block[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];

    let mut mins_correction = 0i32;
    for sub_block in 0..Q4K_SUB_BLOCKS {
        let (_, min_code) = proxima_gguf::quant::q4_k::get_scale_min_k4(sub_block, &scales);
        let bsum_lo = i16::from_le_bytes([bsums[sub_block * 4], bsums[sub_block * 4 + 1]]);
        let bsum_hi = i16::from_le_bytes([bsums[sub_block * 4 + 2], bsums[sub_block * 4 + 3]]);
        mins_correction += i32::from(bsum_lo + bsum_hi) * i32::from(min_code);
    }

    // SAFETY: caller-guaranteed AVX2; `q4_base`/`q8_base` each walk exactly
    // `Q4K_BLOCK_ELEMENTS / 2` / `Q4K_BLOCK_ELEMENTS` bytes across the
    // `Q4K_SUB_BLOCKS / 2` loop iterations below, both within the slices'
    // checked bounds (`_mm256_loadu_si256` needs no alignment).
    unsafe {
        let m4 = _mm256_set1_epi8(0x0f);
        let ones = _mm256_set1_epi16(1);
        let q4_base = weight_block[Q4K_QS_OFFSET..].as_ptr();
        let q8_base = q8k_block[Q8K_QS_OFFSET..].as_ptr().cast::<i8>();

        let mut sumi = 0i32;
        for j in 0..Q4K_SUB_BLOCKS / 2 {
            let q4bits = _mm256_loadu_si256(q4_base.add(j * 32).cast());
            let q4_lo = _mm256_and_si256(q4bits, m4);
            let q4_hi = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            let q8_lo = _mm256_loadu_si256(q8_base.add(j * 64).cast());
            let dot_lo = _mm256_madd_epi16(_mm256_maddubs_epi16(q4_lo, q8_lo), ones);
            let scale_lo = proxima_gguf::quant::q4_k::get_scale_min_k4(2 * j, &scales).0;
            sumi += hsum_epi32_avx2(dot_lo) * i32::from(scale_lo);

            let q8_hi = _mm256_loadu_si256(q8_base.add(j * 64 + 32).cast());
            let dot_hi = _mm256_madd_epi16(_mm256_maddubs_epi16(q4_hi, q8_hi), ones);
            let scale_hi = proxima_gguf::quant::q4_k::get_scale_min_k4(2 * j + 1, &scales).0;
            sumi += hsum_epi32_avx2(dot_hi) * i32::from(scale_hi);
        }

        let d = activation_scale * d_weight;
        let dmin = activation_scale * dmin_weight;
        d.mul_add(sumi as f32, -(dmin * mins_correction as f32))
    }
}

/// A full `Q4_K`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector -- [`matmul_q4k_f32`]'s packed-arithmetic sibling.
/// Quantizes `activation` to `Q8_K` exactly once ([`quantize_row_q8k`],
/// this function's own doc note on why: hoisted out of the row loop), then
/// calls [`dot_q4k_q8k`] per row against that one shared quantized buffer.
///
/// # Errors
/// Propagates [`quantize_row_q8k`]'s and [`dot_q4k_q8k`]'s
/// [`TensorError::QuantizedShapeMismatch`], or reports the same error if
/// `weights.len()` is not a whole multiple of `rows`.
#[cfg(feature = "q4k-int8-dot")]
pub fn matmul_q4k_q8k_f32(weights: &[u8], rows: usize, activation: &[f32]) -> Result<Vec<f32>, TensorError> {
    matmul_q4k_q8k_f32_impl(weights, rows, activation, 1, None)
}

/// [`matmul_q4k_q8k_f32`]'s body, plus `leading_total` (the sequence-position
/// count [`run_reduce_quantized`] already derives as `activation.len() / k`
/// at `cpu.rs:2179`) and the [`CohortSession`] a caller already inside a
/// forward pass's session can supply so `matmul_rows_threaded` dispatches
/// through the cohort instead of `nest_pool`. `matmul_q4k_q8k_f32` itself
/// stays the stable 3-argument public entry point (`leading_total = 1`,
/// `session = None`, unchanged call sites in every bench/test);
/// [`run_reduce_quantized`] is the only caller that passes `leading_total >
/// 1` or a session.
///
/// `activation` stays one contiguous position-major `&[f32]` of
/// `leading_total * k` elements — the same buffer [`run_reduce_quantized`]
/// already holds, not a `&[&[f32]]` or a generic batch type. Quantizing it
/// once here (a single [`quantize_row_q8k`] call over the whole buffer,
/// since every `Q8_K` super-block is 256 elements and `k` is always a whole
/// multiple of that, no super-block ever straddles a position boundary)
/// means each weight row's bytes are read once and its dot reused across
/// every position, instead of the weight stream being re-read once per
/// position the way a `leading_total`-times loop over the narrow
/// 3-argument entry point would.
#[cfg(feature = "q4k-int8-dot")]
fn matmul_q4k_q8k_f32_impl(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
    leading_total: usize,
    session: Option<&MatmulSession<'_>>,
) -> Result<Vec<f32>, TensorError> {
    if rows == 0 {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "matmul_q4k_q8k_f32 called with zero rows",
        });
    }
    if !weights.len().is_multiple_of(rows) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight byte length is not a whole multiple of the row count",
        });
    }
    if leading_total == 0 || !activation.len().is_multiple_of(leading_total) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the position count",
        });
    }
    let k = activation.len() / leading_total;
    if !k.is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the q8_k super-block size",
        });
    }
    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    let q8k_row_bytes = (k / Q4K_BLOCK_ELEMENTS) * Q8K_BLOCK_BYTES;
    let mut activation_q8k = vec![0u8; block_count * Q8K_BLOCK_BYTES];
    // proxima-debugger diagnostic: this preamble runs BEFORE
    // `quantized_matmul_workers`/`matmul_rows_threaded`, so none of the
    // spawn/own-chunk/recv-wait timers in `matmul_rows_threaded` see it --
    // timed separately to settle whether it is the source of the gap
    // between a matmul node's total wall time and its threaded-dispatch
    // time.
    #[cfg(feature = "instrument")]
    let diag_quantize_started = instrument::read_ticks();
    quantize_row_q8k_dispatch(activation, &mut activation_q8k, session)?;
    #[cfg(feature = "instrument")]
    counter!(
        instrument::MATMUL_QUANTIZE_ACTIVATION_TICKS,
        instrument::elapsed_ticks(diag_quantize_started)
    );

    let row_bytes = weights.len() / rows;
    match quantized_matmul_workers(rows, activation.len()) {
        Some(workers) => matmul_rows_threaded(rows, leading_total, workers, session, k, |row, slot| {
            let start = row * row_bytes;
            let weight_row = &weights[start..start + row_bytes];
            for (position, output_slot) in slot.iter_mut().enumerate() {
                let q8k_start = position * q8k_row_bytes;
                *output_slot = dot_q4k_q8k(weight_row, &activation_q8k[q8k_start..q8k_start + q8k_row_bytes])?;
            }
            Ok(())
        }),
        None => weights
            .chunks_exact(row_bytes)
            .try_fold(Vec::with_capacity(rows * leading_total), |mut output, weight_row| {
                for position in 0..leading_total {
                    let q8k_start = position * q8k_row_bytes;
                    output.push(dot_q4k_q8k(weight_row, &activation_q8k[q8k_start..q8k_start + q8k_row_bytes])?);
                }
                Ok::<Vec<f32>, TensorError>(output)
            }),
    }
}

/// [`matmul_q4k_q8k_f32`] with every row routed through
/// [`dot_q4k_q8k_portable`] instead of [`dot_q4k_q8k`] -- the matrix-level
/// counterpart of that function's own doc: the standalone "portable
/// packing alone" measurement, callable (and benchable) on any host
/// regardless of which accelerated arm that host's build would otherwise
/// pick.
///
/// # Errors
/// Same as [`matmul_q4k_q8k_f32`].
#[cfg(feature = "q4k-int8-dot")]
pub fn matmul_q4k_q8k_portable_f32(weights: &[u8], rows: usize, activation: &[f32]) -> Result<Vec<f32>, TensorError> {
    if rows == 0 {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "matmul_q4k_q8k_portable_f32 called with zero rows",
        });
    }
    if !weights.len().is_multiple_of(rows) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight byte length is not a whole multiple of the row count",
        });
    }
    if !activation.len().is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the q8_k super-block size",
        });
    }
    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    let mut activation_q8k = vec![0u8; block_count * Q8K_BLOCK_BYTES];
    quantize_row_q8k(activation, &mut activation_q8k)?;

    let row_bytes = weights.len() / rows;
    weights
        .chunks_exact(row_bytes)
        .map(|weight_row| dot_q4k_q8k_portable(weight_row, &activation_q8k))
        .collect()
}

// ---------------------------------------------------------------------
// `q5k-int8-dot` (default-off): `q4k-int8-dot`'s mechanism applied to
// `Q5_K` -- packed int8 dot directly against `Q8_K`, no `[f32; 256]`
// dequantize pass. `Q5_K` shares `Q4_K`'s exact super-block/sub-block
// shape (8 sub-blocks of 32, the same bit-interleaved 6-bit scale/min
// packing -- `get_scale_min_k4` above is reused unchanged) plus one
// extra `qh` high-bit plane; see `proxima-tensor/docs/discipline.md` for
// the row this landed under.
// ---------------------------------------------------------------------

/// Byte offsets into one packed `Q5_K` super-block ([`Q5K_BLOCK_BYTES`]
/// bytes), mirroring `proxima_gguf::quant::q5_k`'s private layout
/// constants -- duplicated here for the same reason [`Q4K_D_OFFSET`] and
/// siblings are: [`dot_q5k_q8k`] reads the raw bytes directly rather than
/// calling `dequantize_block`.
#[cfg(feature = "q5k-int8-dot")]
const Q5K_D_OFFSET: usize = 0;
#[cfg(feature = "q5k-int8-dot")]
const Q5K_DMIN_OFFSET: usize = 2;
#[cfg(feature = "q5k-int8-dot")]
const Q5K_SCALES_OFFSET: usize = 4;
/// `qh` sits between `scales` and `qs` in `Q5_K`'s on-disk layout
/// (`proxima_gguf::quant::q5_k`'s own module doc, ported from
/// `ggml-common.h:302-313`) -- unlike `Q4_K`, which has no high-bit plane
/// at all.
#[cfg(feature = "q5k-int8-dot")]
const Q5K_QH_OFFSET: usize = Q5K_SCALES_OFFSET + Q4K_SCALE_BYTES;
#[cfg(feature = "q5k-int8-dot")]
const Q5K_QH_BYTES: usize = Q4K_BLOCK_ELEMENTS / 8;
#[cfg(feature = "q5k-int8-dot")]
const Q5K_QS_OFFSET: usize = Q5K_QH_OFFSET + Q5K_QH_BYTES;

/// One `Q5_K`-weight-row x `Q8_K`-activation int8 dot product --
/// [`dot_q4k_q8k`]'s sibling for the 5-bit codec. Dispatches to
/// `dot_q5k_q8k_block_neon_dotprod` under `q4k_dotprod` (the same
/// arch-wide cfg [`dot_q4k_q8k`] keys off -- `FEAT_DotProd` availability is
/// a property of the target, not the weight codec) and to the portable
/// `dot_q5k_q8k_block_scalar` everywhere else. No AVX2 arm yet -- the
/// task ordering this landed under ran portable-then-aarch64 first; an
/// AVX2 arm is future work, not a correctness gap (the portable arm is
/// what an x86-64 build without `+avx2` runs regardless).
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of `Q5K_BLOCK_BYTES`, or `activation_q8k.len()` does
/// not equal the row's block count times `Q8K_BLOCK_BYTES`.
#[cfg(feature = "q5k-int8-dot")]
pub fn dot_q5k_q8k(weight_row: &[u8], activation_q8k: &[u8]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q5K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q5_k block size",
        });
    }
    let block_count = weight_row.len() / Q5K_BLOCK_BYTES;
    if activation_q8k.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k activation length does not match the weight row's block count",
        });
    }

    let mut acc = 0.0f32;
    for (weight_block, q8k_block) in weight_row
        .as_chunks::<Q5K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation_q8k.as_chunks::<Q8K_BLOCK_BYTES>().0)
    {
        #[cfg(q4k_dotprod)]
        // SAFETY: `q4k_dotprod` is emitted by build.rs only for aarch64
        // targets, all of which carry FEAT_DotProd.
        let block_sum = unsafe { dot_q5k_q8k_block_neon_dotprod(weight_block, q8k_block) };
        #[cfg(not(q4k_dotprod))]
        let block_sum = dot_q5k_q8k_block_scalar(weight_block, q8k_block);
        acc += block_sum;
    }
    Ok(acc)
}

/// [`dot_q5k_q8k`] with the dispatch forced to
/// `dot_q5k_q8k_block_scalar` regardless of `q4k_dotprod` -- the
/// standalone "portable packing alone" measurement, same role
/// [`dot_q4k_q8k_portable`] plays for `Q4_K`.
///
/// # Errors
/// Same as [`dot_q5k_q8k`].
#[cfg(feature = "q5k-int8-dot")]
pub fn dot_q5k_q8k_portable(weight_row: &[u8], activation_q8k: &[u8]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q5K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q5_k block size",
        });
    }
    let block_count = weight_row.len() / Q5K_BLOCK_BYTES;
    if activation_q8k.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k activation length does not match the weight row's block count",
        });
    }

    let mut acc = 0.0f32;
    for (weight_block, q8k_block) in weight_row
        .as_chunks::<Q5K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation_q8k.as_chunks::<Q8K_BLOCK_BYTES>().0)
    {
        acc += dot_q5k_q8k_block_scalar(weight_block, q8k_block);
    }
    Ok(acc)
}

/// The portable packed-nibble x `Q8_K` int8 dot for `Q5_K` -- no
/// dequantize pass, no `f32` intermediate. Identical structure to
/// [`dot_q4k_q8k_block_scalar`] (unpack each sub-block's 6-bit
/// scale/min via [`proxima_gguf::quant::q4_k::get_scale_min_k4`], dot 32
/// nibbles against 32 `Q8_K` activations, scale, accumulate; mins
/// correction identical -- `Q5_K` shares `Q4_K`'s exact bit-interleaved
/// scale/min packing) plus one addition: each nibble is OR'd with its
/// `qh` high bit before the multiply. `qh_mask = 1u8 << sub_block` is not
/// an approximation -- it is the exact bit `proxima_gguf::quant::q5_k`'s
/// own `dequantize_block` reads for this `sub_block` (derived from that
/// function's `mask_lo`/`mask_hi` cycling, which starts at
/// `1u8`/`2u8` and shifts left by 2 every 64-element chunk: the two
/// sub-blocks sharing a chunk land on consecutive bits, and consecutive
/// chunks land on the next two bits up, i.e. bit index == `sub_block`
/// exactly, for every one of the 8 sub-blocks).
#[cfg(feature = "q5k-int8-dot")]
fn dot_q5k_q8k_block_scalar(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q5K_D_OFFSET);
    let dmin_weight = f16_le_at(weight_block, Q5K_DMIN_OFFSET);
    let mut scales = [0u8; Q4K_SCALE_BYTES];
    scales.copy_from_slice(&weight_block[Q5K_SCALES_OFFSET..Q5K_SCALES_OFFSET + Q4K_SCALE_BYTES]);
    let qh = &weight_block[Q5K_QH_OFFSET..Q5K_QH_OFFSET + Q5K_QH_BYTES];
    let qs = &weight_block[Q5K_QS_OFFSET..Q5K_QS_OFFSET + Q4K_BLOCK_ELEMENTS / 2];

    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);
    let activation_qs = &q8k_block[Q8K_QS_OFFSET..Q8K_QS_OFFSET + Q4K_BLOCK_ELEMENTS];
    let bsums = &q8k_block[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];

    let mut sumi = 0i32;
    let mut mins_correction = 0i32;
    for sub_block in 0..Q4K_SUB_BLOCKS {
        let (scale_code, min_code) = proxima_gguf::quant::q4_k::get_scale_min_k4(sub_block, &scales);

        let bsum_lo = i16::from_le_bytes([bsums[sub_block * 4], bsums[sub_block * 4 + 1]]);
        let bsum_hi = i16::from_le_bytes([bsums[sub_block * 4 + 2], bsums[sub_block * 4 + 3]]);
        mins_correction += i32::from(bsum_lo + bsum_hi) * i32::from(min_code);

        let byte_base = (sub_block / 2) * 32;
        let is_high_nibble = sub_block % 2 == 1;
        let activation_base = sub_block * 32;
        let qh_mask = 1u8 << sub_block;
        let mut partial = 0i32;
        for offset in 0..32 {
            let byte = qs[byte_base + offset];
            let nibble = i32::from(if is_high_nibble { byte >> 4 } else { byte & 0x0F });
            let high_bit = i32::from(qh[offset] & qh_mask != 0) * 16;
            let level = nibble + high_bit;
            let activation_value = i32::from(activation_qs[activation_base + offset].cast_signed());
            partial += level * activation_value;
        }
        sumi += partial * i32::from(scale_code);
    }

    let d = activation_scale * d_weight;
    let dmin = activation_scale * dmin_weight;
    d.mul_add(sumi as f32, -(dmin * mins_correction as f32))
}

/// [`dot_q5k_q8k_block_scalar`]'s mechanism, `vdotq_s32`-accelerated.
/// Ports `ggml_vec_dot_q5_K_q8_K`'s `__ARM_NEON` arm
/// (`arch/arm/quants.c:2512-2579`) directly: per 64-element chunk (`j` in
/// `0..4`), extracts the current chunk's two high-bit planes from the
/// (persistently right-shifted) `qh` register pair via
/// `vandq_u8`/`vshlq_n_u8` with `mone`/`mtwo` masks, ORs each into its
/// nibble half, then two [`sdot_s32`] pairs per chunk (low nibble pair,
/// high nibble pair) instead of [`dot_q5k_q8k_block_scalar`]'s
/// 32-iteration scalar loop per sub-block. Scale/min unpack routes through
/// [`mins_correction_neon`] -- the same once-per-super-block NEON bit-trick
/// [`dot_q4k_q8k_block_neon_dotprod`] uses, since `Q5_K`'s 12-byte
/// scale/min field is byte-identical in layout -- in place of the 16 scalar
/// `get_scale_min_k4` calls (8 for the mins correction, 8 more inside this
/// loop for `scale_lo`/`scale_hi`) that path used to make per block.
///
/// # Safety
/// Caller guarantees `FEAT_DotProd`; `weight_block.len() ==
/// Q5K_BLOCK_BYTES` and `q8k_block.len() == Q8K_BLOCK_BYTES` (both
/// [`dot_q5k_q8k`]'s own `chunks_exact` calls already guarantee before
/// calling this).
#[cfg(all(q4k_dotprod, feature = "q5k-int8-dot"))]
unsafe fn dot_q5k_q8k_block_neon_dotprod(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q5K_D_OFFSET);
    let dmin_weight = f16_le_at(weight_block, Q5K_DMIN_OFFSET);
    let mut scales = [0u8; Q4K_SCALE_BYTES];
    scales.copy_from_slice(&weight_block[Q5K_SCALES_OFFSET..Q5K_SCALES_OFFSET + Q4K_SCALE_BYTES]);

    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);
    let bsums = &q8k_block[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];

    // SAFETY: caller-guaranteed FEAT_DotProd (this fn's own doc);
    // `mins_correction_neon`'s own preconditions (`scales`/`bsums` lengths)
    // are met by the fixed-size array and the slice sized above.
    let (scale_lo, scale_hi, mins_correction) = unsafe { mins_correction_neon(&scales, bsums) };

    // SAFETY: caller-guaranteed FEAT_DotProd; `q5_base` walks exactly
    // `Q4K_BLOCK_ELEMENTS / 2` bytes, `qh_base` is read once (32 bytes,
    // never advanced) and `q8_base` walks exactly `Q4K_BLOCK_ELEMENTS`
    // bytes, across the `Q4K_SUB_BLOCKS / 2` loop iterations below -- all
    // within the slices' checked bounds.
    unsafe {
        let m4b = vdupq_n_u8(0x0f);
        let mone = vdupq_n_u8(1);
        let mtwo = vdupq_n_u8(2);
        let mzero = vdupq_n_s32(0);
        let q5_base = weight_block[Q5K_QS_OFFSET..].as_ptr();
        let qh_base = weight_block[Q5K_QH_OFFSET..].as_ptr();
        let q8_base = q8k_block[Q8K_QS_OFFSET..].as_ptr().cast::<i8>();

        let mut qhbits0 = vld1q_u8(qh_base);
        let mut qhbits1 = vld1q_u8(qh_base.add(16));

        let mut sumi: i32 = 0;
        for j in 0..Q4K_SUB_BLOCKS / 2 {
            let q5bits0 = vld1q_u8(q5_base.add(j * 32));
            let q5bits1 = vld1q_u8(q5_base.add(j * 32 + 16));

            let q5h0 = vshlq_n_u8(vandq_u8(mone, qhbits0), 4);
            let q5h1 = vshlq_n_u8(vandq_u8(mone, qhbits1), 4);
            let q5h2 = vshlq_n_u8(vandq_u8(mtwo, qhbits0), 3);
            let q5h3 = vshlq_n_u8(vandq_u8(mtwo, qhbits1), 3);
            qhbits0 = vshrq_n_u8(qhbits0, 2);
            qhbits1 = vshrq_n_u8(qhbits1, 2);

            let q5bytes0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits0, m4b), q5h0));
            let q5bytes1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits1, m4b), q5h1));
            let q5bytes2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits0, 4), q5h2));
            let q5bytes3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits1, 4), q5h3));

            let q8b0 = vld1q_s8(q8_base.add(j * 64));
            let q8b1 = vld1q_s8(q8_base.add(j * 64 + 16));
            let q8b2 = vld1q_s8(q8_base.add(j * 64 + 32));
            let q8b3 = vld1q_s8(q8_base.add(j * 64 + 48));

            let scale_word = if j < 2 { scale_lo } else { scale_hi };
            let scale_shift = (j % 2) as u32 * 2;

            let partial_lo = sdot_s32(sdot_s32(mzero, q5bytes0, q8b0), q5bytes1, q8b1);
            sumi += vaddvq_s32(partial_lo) * scale_byte(scale_word, scale_shift);

            let partial_hi = sdot_s32(sdot_s32(mzero, q5bytes2, q8b2), q5bytes3, q8b3);
            sumi += vaddvq_s32(partial_hi) * scale_byte(scale_word, scale_shift + 1);
        }

        let d = activation_scale * d_weight;
        let dmin = activation_scale * dmin_weight;
        d.mul_add(sumi as f32, -(dmin * mins_correction as f32))
    }
}

/// A full `Q5_K`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector — [`matmul_q5k_f32`]'s packed-arithmetic sibling,
/// same structure as [`matmul_q4k_q8k_f32`].
///
/// # Errors
/// Propagates [`quantize_row_q8k`]'s and [`dot_q5k_q8k`]'s
/// [`TensorError::QuantizedShapeMismatch`], or reports the same error if
/// `weights.len()` is not a whole multiple of `rows`.
#[cfg(feature = "q5k-int8-dot")]
pub fn matmul_q5k_q8k_f32(weights: &[u8], rows: usize, activation: &[f32]) -> Result<Vec<f32>, TensorError> {
    matmul_q5k_q8k_f32_impl(weights, rows, activation, 1, None)
}

/// [`matmul_q5k_q8k_f32`]'s body, plus `leading_total` (the sequence-position
/// count [`run_reduce_quantized`] derives as `activation.len() / k`) and the
/// [`CohortSession`] a caller already inside a forward pass's session can
/// supply — identical shape to [`matmul_q4k_q8k_f32_impl`]: `activation` is
/// one contiguous position-major `&[f32]` of `leading_total * k` elements,
/// quantized to `Q8_K` once, so each weight row's bytes are read once and its
/// dot reused across every position instead of the weight stream being
/// re-read once per position.
#[cfg(feature = "q5k-int8-dot")]
fn matmul_q5k_q8k_f32_impl(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
    leading_total: usize,
    session: Option<&MatmulSession<'_>>,
) -> Result<Vec<f32>, TensorError> {
    if rows == 0 {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "matmul_q5k_q8k_f32 called with zero rows",
        });
    }
    if !weights.len().is_multiple_of(rows) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight byte length is not a whole multiple of the row count",
        });
    }
    if leading_total == 0 || !activation.len().is_multiple_of(leading_total) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the position count",
        });
    }
    let k = activation.len() / leading_total;
    if !k.is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the q8_k super-block size",
        });
    }
    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    let q8k_row_bytes = (k / Q4K_BLOCK_ELEMENTS) * Q8K_BLOCK_BYTES;
    let mut activation_q8k = vec![0u8; block_count * Q8K_BLOCK_BYTES];
    quantize_row_q8k_dispatch(activation, &mut activation_q8k, session)?;

    let row_bytes = weights.len() / rows;
    match quantized_matmul_workers(rows, activation.len()) {
        Some(workers) => matmul_rows_threaded(rows, leading_total, workers, session, k, |row, slot| {
            let start = row * row_bytes;
            let weight_row = &weights[start..start + row_bytes];
            for (position, output_slot) in slot.iter_mut().enumerate() {
                let q8k_start = position * q8k_row_bytes;
                *output_slot = dot_q5k_q8k(weight_row, &activation_q8k[q8k_start..q8k_start + q8k_row_bytes])?;
            }
            Ok(())
        }),
        None => weights
            .chunks_exact(row_bytes)
            .try_fold(Vec::with_capacity(rows * leading_total), |mut output, weight_row| {
                for position in 0..leading_total {
                    let q8k_start = position * q8k_row_bytes;
                    output.push(dot_q5k_q8k(weight_row, &activation_q8k[q8k_start..q8k_start + q8k_row_bytes])?);
                }
                Ok::<Vec<f32>, TensorError>(output)
            }),
    }
}

/// [`matmul_q5k_q8k_f32`] with every row routed through
/// [`dot_q5k_q8k_portable`] instead of [`dot_q5k_q8k`] -- the matrix-level
/// "portable packing alone" measurement, callable regardless of which
/// accelerated arm the host build would otherwise pick.
///
/// # Errors
/// Same as [`matmul_q5k_q8k_f32`].
#[cfg(feature = "q5k-int8-dot")]
pub fn matmul_q5k_q8k_portable_f32(weights: &[u8], rows: usize, activation: &[f32]) -> Result<Vec<f32>, TensorError> {
    if rows == 0 {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "matmul_q5k_q8k_portable_f32 called with zero rows",
        });
    }
    if !weights.len().is_multiple_of(rows) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight byte length is not a whole multiple of the row count",
        });
    }
    if !activation.len().is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the q8_k super-block size",
        });
    }
    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    let mut activation_q8k = vec![0u8; block_count * Q8K_BLOCK_BYTES];
    quantize_row_q8k(activation, &mut activation_q8k)?;

    let row_bytes = weights.len() / rows;
    weights
        .chunks_exact(row_bytes)
        .map(|weight_row| dot_q5k_q8k_portable(weight_row, &activation_q8k))
        .collect()
}

// ---------------------------------------------------------------------
// `q6k-int8-dot` (default-off): `q4k-int8-dot`'s mechanism applied to
// `Q6_K` -- packed int8 dot directly against `Q8_K`. `Q6_K` has a
// DIFFERENT super-block shape from `Q4_K`/`Q5_K`: 16 sub-blocks of 16
// (not 8 of 32), one signed 8-bit scale per sub-block, no `dmin` term at
// all (`x = d*sc*(q-32)`, `proxima_gguf::quant::q6_k`'s own module doc).
// See `proxima-tensor/docs/discipline.md` for the row this landed under.
// ---------------------------------------------------------------------

/// Byte offsets into one packed `Q6_K` super-block ([`Q6K_BLOCK_BYTES`]
/// bytes), mirroring `proxima_gguf::quant::q6_k`'s private layout
/// constants -- duplicated here for the same reason [`Q4K_D_OFFSET`] and
/// siblings are. Note the field order: `d` TRAILS the block here (unlike
/// `Q4_K`/`Q5_K`, where it leads) -- `proxima_gguf::quant::q6_k`'s own
/// module doc flags this explicitly as the one layout trap this codec has
/// that the others don't.
#[cfg(feature = "q6k-int8-dot")]
const Q6K_QL_OFFSET: usize = 0;
#[cfg(feature = "q6k-int8-dot")]
const Q6K_QL_BYTES: usize = Q4K_BLOCK_ELEMENTS / 2;
#[cfg(feature = "q6k-int8-dot")]
const Q6K_QH_OFFSET: usize = Q6K_QL_OFFSET + Q6K_QL_BYTES;
#[cfg(feature = "q6k-int8-dot")]
const Q6K_QH_BYTES: usize = Q4K_BLOCK_ELEMENTS / 4;
#[cfg(feature = "q6k-int8-dot")]
const Q6K_SCALES_OFFSET: usize = Q6K_QH_OFFSET + Q6K_QH_BYTES;
#[cfg(feature = "q6k-int8-dot")]
const Q6K_D_OFFSET: usize = Q6K_SCALES_OFFSET + proxima_gguf::quant::q6_k::SUB_BLOCKS;

/// One `Q6_K`-weight-row x `Q8_K`-activation int8 dot product --
/// [`dot_q4k_q8k`]'s sibling for the 6-bit codec. Dispatches to
/// `dot_q6k_q8k_block_neon_dotprod` under `q4k_dotprod` and to the
/// portable `dot_q6k_q8k_block_scalar` everywhere else -- same dispatch
/// shape as [`dot_q5k_q8k`], no AVX2 arm yet (same rationale: portable
/// arm first, aarch64 second, per this landing's task ordering).
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of `Q6K_BLOCK_BYTES`, or `activation_q8k.len()` does
/// not equal the row's block count times `Q8K_BLOCK_BYTES`.
#[cfg(feature = "q6k-int8-dot")]
pub fn dot_q6k_q8k(weight_row: &[u8], activation_q8k: &[u8]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q6K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q6_k block size",
        });
    }
    let block_count = weight_row.len() / Q6K_BLOCK_BYTES;
    if activation_q8k.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k activation length does not match the weight row's block count",
        });
    }

    let mut acc = 0.0f32;
    for (weight_block, q8k_block) in weight_row
        .as_chunks::<Q6K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation_q8k.as_chunks::<Q8K_BLOCK_BYTES>().0)
    {
        #[cfg(q4k_dotprod)]
        // SAFETY: `q4k_dotprod` is emitted by build.rs only for aarch64
        // targets, all of which carry FEAT_DotProd.
        let block_sum = unsafe { dot_q6k_q8k_block_neon_dotprod(weight_block, q8k_block) };
        #[cfg(not(q4k_dotprod))]
        let block_sum = dot_q6k_q8k_block_scalar(weight_block, q8k_block);
        acc += block_sum;
    }
    Ok(acc)
}

/// [`dot_q6k_q8k`] with the dispatch forced to
/// `dot_q6k_q8k_block_scalar` regardless of `q4k_dotprod`.
///
/// # Errors
/// Same as [`dot_q6k_q8k`].
#[cfg(feature = "q6k-int8-dot")]
pub fn dot_q6k_q8k_portable(weight_row: &[u8], activation_q8k: &[u8]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q6K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q6_k block size",
        });
    }
    let block_count = weight_row.len() / Q6K_BLOCK_BYTES;
    if activation_q8k.len() != block_count * Q8K_BLOCK_BYTES {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "q8_k activation length does not match the weight row's block count",
        });
    }

    let mut acc = 0.0f32;
    for (weight_block, q8k_block) in weight_row
        .as_chunks::<Q6K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation_q8k.as_chunks::<Q8K_BLOCK_BYTES>().0)
    {
        acc += dot_q6k_q8k_block_scalar(weight_block, q8k_block);
    }
    Ok(acc)
}

/// The portable packed-nibble x `Q8_K` int8 dot for `Q6_K` -- no
/// dequantize pass, no `f32` intermediate. Unlike [`dot_q4k_q8k_block_scalar`]/
/// [`dot_q5k_q8k_block_scalar`], this does not reuse those codecs'
/// `get_scale_min_k4` unpack (`Q6_K` has no bit-interleaved scale/min
/// pair at all, just 16 plain signed-`i8` scales and no `dmin`) or their
/// `byte_base = (sub_block/2)*32` nibble addressing (`Q6_K`'s sub-blocks
/// are 16 wide, not 32, and its `ql`/`qh` byte layout is genuinely
/// different -- see [`proxima_gguf::quant::q6_k::unpack_levels`], which
/// this function's addressing (`half`/`local_sub`/`lane`/`subhalf`) is
/// derived from and stays consistent with: `half = sub_block / 8`,
/// `local_sub = sub_block % 8`, `lane = local_sub / 2`,
/// `subhalf = local_sub % 2`; for output offset `e` within the sub-block,
/// `l = subhalf*16 + e` is `unpack_levels`'s own index parameter). Each
/// unpacked 6-bit level is biased by -32 before the multiply
/// (`q6_k.rs`'s own `x = d*sc*(q-32)` doc), matching what
/// `proxima_gguf::quant::q6_k::dequantize_block` computes exactly, just
/// against a `Q8_K` `i8` activation instead of an `f32` one.
#[cfg(feature = "q6k-int8-dot")]
fn dot_q6k_q8k_block_scalar(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q6K_D_OFFSET);
    let ql = &weight_block[Q6K_QL_OFFSET..Q6K_QL_OFFSET + Q6K_QL_BYTES];
    let qh = &weight_block[Q6K_QH_OFFSET..Q6K_QH_OFFSET + Q6K_QH_BYTES];
    let scales = &weight_block[Q6K_SCALES_OFFSET..Q6K_SCALES_OFFSET + proxima_gguf::quant::q6_k::SUB_BLOCKS];

    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);
    let activation_qs = &q8k_block[Q8K_QS_OFFSET..Q8K_QS_OFFSET + Q4K_BLOCK_ELEMENTS];

    let sub_block_elements = proxima_gguf::quant::q6_k::SUB_BLOCK_ELEMENTS;
    let mut sumi = 0i32;
    for (sub_block, &scale_byte) in scales.iter().enumerate() {
        let half = sub_block / 8;
        let local_sub = sub_block % 8;
        let lane = local_sub / 2;
        let subhalf = local_sub % 2;
        let scale = i32::from(scale_byte.cast_signed());
        let ql_half = &ql[half * 64..half * 64 + 64];
        let qh_half = &qh[half * 32..half * 32 + 32];
        let activation_base = sub_block * sub_block_elements;

        let mut partial = 0i32;
        for offset in 0..sub_block_elements {
            let l = subhalf * sub_block_elements + offset;
            let ql_byte = if lane == 0 || lane == 2 { ql_half[l] } else { ql_half[l + 32] };
            let nibble = if lane < 2 { ql_byte & 0x0F } else { ql_byte >> 4 };
            let high = (qh_half[l] >> (2 * lane)) & 0x03;
            let level = i32::from(nibble) | (i32::from(high) << 4);
            let quant = level - 32;
            let activation_value = i32::from(activation_qs[activation_base + offset].cast_signed());
            partial += quant * activation_value;
        }
        sumi += partial * scale;
    }

    let d = activation_scale * d_weight;
    d * sumi as f32
}

/// [`dot_q6k_q8k_block_scalar`]'s mechanism, `vdotq_s32`-accelerated.
/// Ports `ggml_vec_dot_q6_K_q8_K`'s plain `__ARM_NEON` arm
/// (`arch/arm/quants.c:3001-3090`, the non-`__ARM_FEATURE_MATMUL_INT8`,
/// non-SVE arm) with one deliberate simplification: ggml's version keeps
/// levels unbiased (`0..63`) through the dot and corrects for the -32
/// bias afterward via `bsums`/`isum_mins` (an optimization to avoid a
/// per-lane subtract); this port applies the -32 bias directly in-register
/// via [`vsubq_s8`] right after assembling each `q6bytes` lane, then dots
/// against `Q8_K` `i8` activations with no separate correction term
/// needed -- the SAME value, a simpler derivation, one extra vector op per
/// lane (8 total) traded for not needing `y[i].bsums` decoded at all here.
///
/// # Safety
/// Caller guarantees `FEAT_DotProd`; `weight_block.len() ==
/// Q6K_BLOCK_BYTES` and `q8k_block.len() == Q8K_BLOCK_BYTES` (both
/// [`dot_q6k_q8k`]'s own `chunks_exact` calls already guarantee before
/// calling this).
#[cfg(all(q4k_dotprod, feature = "q6k-int8-dot"))]
unsafe fn dot_q6k_q8k_block_neon_dotprod(weight_block: &[u8], q8k_block: &[u8]) -> f32 {
    let d_weight = f16_le_at(weight_block, Q6K_D_OFFSET);
    let mut scales = [0i8; 16];
    for (slot, byte) in scales
        .iter_mut()
        .zip(weight_block[Q6K_SCALES_OFFSET..Q6K_SCALES_OFFSET + proxima_gguf::quant::q6_k::SUB_BLOCKS].iter())
    {
        *slot = byte.cast_signed();
    }

    let mut d_bytes = [0u8; 4];
    d_bytes.copy_from_slice(&q8k_block[Q8K_D_OFFSET..Q8K_D_OFFSET + 4]);
    let activation_scale = f32::from_le_bytes(d_bytes);

    // SAFETY: caller-guaranteed FEAT_DotProd; `ql_base`/`qh_base` each walk
    // exactly `Q6K_QL_BYTES` / `Q6K_QH_BYTES` bytes and `q8_base` walks
    // exactly `Q4K_BLOCK_ELEMENTS` bytes across the two `half` iterations
    // below, all within the slices' checked bounds.
    unsafe {
        let m4b = vdupq_n_u8(0x0f);
        let high_bits_mask = vdupq_n_u8(0x03);
        let m32s = vdupq_n_s8(32);
        let mzero = vdupq_n_s32(0);
        let ql_base = weight_block[Q6K_QL_OFFSET..].as_ptr();
        let qh_base = weight_block[Q6K_QH_OFFSET..].as_ptr();
        let q8_base = q8k_block[Q8K_QS_OFFSET..].as_ptr().cast::<i8>();

        // FOUR accumulators, not one. `Q6_K` is 16 sub-blocks of 16 (vs
        // `Q4_K`/`Q5_K`'s 8 of 32), so a single `sumi` chains 16 dependent
        // `madd`s per super-block against `Q4_K`'s 3 -- measured 0.0429 ns/mac
        // here vs 0.0245 there, a 1.75x gap on only 1.33x the instructions
        // (190 vs 143), which is the signature of dependency depth, not
        // volume. Integer addition is associative, so splitting the chain is
        // bit-identical rather than merely close. The same defect cost 3.2x in
        // `dot_q4k_f32` in an earlier round, where the fix measured 5.68x --
        // width and depth turned out not to be independent factors.
        let mut sumi0: i32 = 0;
        let mut sumi1: i32 = 0;
        let mut sumi2: i32 = 0;
        let mut sumi3: i32 = 0;
        for half in 0..2usize {
            let qhbits0 = vld1q_u8(qh_base.add(half * 32));
            let qhbits1 = vld1q_u8(qh_base.add(half * 32 + 16));
            let ql0 = vld1q_u8(ql_base.add(half * 64));
            let ql1 = vld1q_u8(ql_base.add(half * 64 + 16));
            let ql2 = vld1q_u8(ql_base.add(half * 64 + 32));
            let ql3 = vld1q_u8(ql_base.add(half * 64 + 48));
            let q8_half_base = q8_base.add(half * 128);
            let scale_half = &scales[half * 8..half * 8 + 8];

            let low0 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, m4b), vshlq_n_u8(vandq_u8(qhbits0, high_bits_mask), 4))),
                m32s,
            );
            let low1 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, m4b), vshlq_n_u8(vandq_u8(qhbits1, high_bits_mask), 4))),
                m32s,
            );
            let low2 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql2, m4b),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qhbits0, 2), high_bits_mask), 4),
                )),
                m32s,
            );
            let low3 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vandq_u8(ql3, m4b),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qhbits1, 2), high_bits_mask), 4),
                )),
                m32s,
            );

            let q8_lo0 = vld1q_s8(q8_half_base);
            let q8_lo1 = vld1q_s8(q8_half_base.add(16));
            let q8_lo2 = vld1q_s8(q8_half_base.add(32));
            let q8_lo3 = vld1q_s8(q8_half_base.add(48));
            sumi0 += vaddvq_s32(sdot_s32(mzero, low0, q8_lo0)) * i32::from(scale_half[0]);
            sumi1 += vaddvq_s32(sdot_s32(mzero, low1, q8_lo1)) * i32::from(scale_half[1]);
            sumi2 += vaddvq_s32(sdot_s32(mzero, low2, q8_lo2)) * i32::from(scale_half[2]);
            sumi3 += vaddvq_s32(sdot_s32(mzero, low3, q8_lo3)) * i32::from(scale_half[3]);

            let high0 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql0, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qhbits0, 4), high_bits_mask), 4),
                )),
                m32s,
            );
            let high1 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql1, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qhbits1, 4), high_bits_mask), 4),
                )),
                m32s,
            );
            let high2 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql2, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qhbits0, 6), high_bits_mask), 4),
                )),
                m32s,
            );
            let high3 = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(
                    vshrq_n_u8(ql3, 4),
                    vshlq_n_u8(vandq_u8(vshrq_n_u8(qhbits1, 6), high_bits_mask), 4),
                )),
                m32s,
            );

            let q8_hi0 = vld1q_s8(q8_half_base.add(64));
            let q8_hi1 = vld1q_s8(q8_half_base.add(80));
            let q8_hi2 = vld1q_s8(q8_half_base.add(96));
            let q8_hi3 = vld1q_s8(q8_half_base.add(112));
            sumi0 += vaddvq_s32(sdot_s32(mzero, high0, q8_hi0)) * i32::from(scale_half[4]);
            sumi1 += vaddvq_s32(sdot_s32(mzero, high1, q8_hi1)) * i32::from(scale_half[5]);
            sumi2 += vaddvq_s32(sdot_s32(mzero, high2, q8_hi2)) * i32::from(scale_half[6]);
            sumi3 += vaddvq_s32(sdot_s32(mzero, high3, q8_hi3)) * i32::from(scale_half[7]);
        }

        activation_scale * d_weight * (sumi0 + sumi1 + sumi2 + sumi3) as f32
    }
}

/// A full `Q6_K`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector — [`matmul_q6k_f32`]'s packed-arithmetic sibling.
///
/// # Errors
/// Propagates [`quantize_row_q8k`]'s and [`dot_q6k_q8k`]'s
/// [`TensorError::QuantizedShapeMismatch`], or reports the same error if
/// `weights.len()` is not a whole multiple of `rows`.
#[cfg(feature = "q6k-int8-dot")]
pub fn matmul_q6k_q8k_f32(weights: &[u8], rows: usize, activation: &[f32]) -> Result<Vec<f32>, TensorError> {
    matmul_q6k_q8k_f32_impl(weights, rows, activation, 1, None)
}

/// [`matmul_q6k_q8k_f32`]'s body, plus `leading_total` (the sequence-position
/// count [`run_reduce_quantized`] derives as `activation.len() / k`) and the
/// [`CohortSession`] a caller already inside a forward pass's session can
/// supply — identical shape to [`matmul_q4k_q8k_f32_impl`]: `activation` is
/// one contiguous position-major `&[f32]` of `leading_total * k` elements,
/// quantized to `Q8_K` once, so each weight row's bytes are read once and its
/// dot reused across every position instead of the weight stream being
/// re-read once per position.
#[cfg(feature = "q6k-int8-dot")]
fn matmul_q6k_q8k_f32_impl(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
    leading_total: usize,
    session: Option<&MatmulSession<'_>>,
) -> Result<Vec<f32>, TensorError> {
    if rows == 0 {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "matmul_q6k_q8k_f32 called with zero rows",
        });
    }
    if !weights.len().is_multiple_of(rows) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight byte length is not a whole multiple of the row count",
        });
    }
    if leading_total == 0 || !activation.len().is_multiple_of(leading_total) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the position count",
        });
    }
    let k = activation.len() / leading_total;
    if !k.is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the q8_k super-block size",
        });
    }
    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    let q8k_row_bytes = (k / Q4K_BLOCK_ELEMENTS) * Q8K_BLOCK_BYTES;
    let mut activation_q8k = vec![0u8; block_count * Q8K_BLOCK_BYTES];
    quantize_row_q8k_dispatch(activation, &mut activation_q8k, session)?;

    let row_bytes = weights.len() / rows;
    match quantized_matmul_workers(rows, activation.len()) {
        Some(workers) => matmul_rows_threaded(rows, leading_total, workers, session, k, |row, slot| {
            let start = row * row_bytes;
            let weight_row = &weights[start..start + row_bytes];
            for (position, output_slot) in slot.iter_mut().enumerate() {
                let q8k_start = position * q8k_row_bytes;
                *output_slot = dot_q6k_q8k(weight_row, &activation_q8k[q8k_start..q8k_start + q8k_row_bytes])?;
            }
            Ok(())
        }),
        None => weights
            .chunks_exact(row_bytes)
            .try_fold(Vec::with_capacity(rows * leading_total), |mut output, weight_row| {
                for position in 0..leading_total {
                    let q8k_start = position * q8k_row_bytes;
                    output.push(dot_q6k_q8k(weight_row, &activation_q8k[q8k_start..q8k_start + q8k_row_bytes])?);
                }
                Ok::<Vec<f32>, TensorError>(output)
            }),
    }
}

/// [`matmul_q6k_q8k_f32`] with every row routed through
/// [`dot_q6k_q8k_portable`] instead of [`dot_q6k_q8k`].
///
/// # Errors
/// Same as [`matmul_q6k_q8k_f32`].
#[cfg(feature = "q6k-int8-dot")]
pub fn matmul_q6k_q8k_portable_f32(weights: &[u8], rows: usize, activation: &[f32]) -> Result<Vec<f32>, TensorError> {
    if rows == 0 {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "matmul_q6k_q8k_portable_f32 called with zero rows",
        });
    }
    if !weights.len().is_multiple_of(rows) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight byte length is not a whole multiple of the row count",
        });
    }
    if !activation.len().is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length is not a whole multiple of the q8_k super-block size",
        });
    }
    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    let mut activation_q8k = vec![0u8; block_count * Q8K_BLOCK_BYTES];
    quantize_row_q8k(activation, &mut activation_q8k)?;

    let row_bytes = weights.len() / rows;
    weights
        .chunks_exact(row_bytes)
        .map(|weight_row| dot_q6k_q8k_portable(weight_row, &activation_q8k))
        .collect()
}

/// Ported from ggml tinyBLAS's `gemm_bloc`: `ROWS` x [`TILE_COLS`]
/// output accumulators declared as `float32x4_t`, a native NEON vector
/// register type, not an `[f32; 4]` array indexed by a loop variable
/// (`proxima-tensor/docs/discipline.md` — attempt 2 spilled 737 `str q`
/// instructions doing exactly that). `av` holds one `float32x4_t` per tile
/// row, loaded once per `k`-step and reused across all [`TILE_COLS`]
/// columns; `bv` is loaded once per column per step and fused against every
/// row's `av`, giving 0.42 loads per multiply-accumulate the way tinyBLAS's
/// own microkernel does, versus this crate's un-tiled 2.0.
///
/// Generic over the row count (`ROWS`) rather than fixed at [`TILE_ROWS`] so
/// the row-remainder pass can call the identical kernel body monomorphised
/// at whichever width `1..=5` the leftover row count needs, instead of a
/// hand-duplicated copy per width.
#[cfg(target_arch = "aarch64")]
unsafe fn gemm_tile_neon<const ROWS: usize>(a: KStridedTile, b: KStridedTile, k: usize, out: &mut [[f32; TILE_COLS]; ROWS]) {
    // `vdupq_n_f32` requires the `neon` target feature, unconditionally
    // present in the aarch64 base ISA this module is gated on.
    let mut acc = [[unsafe { vdupq_n_f32(0.0) }; TILE_COLS]; ROWS];
    let steps = k / 4;
    for step in 0..steps {
        let l = step * 4;
        let mut av = [unsafe { vdupq_n_f32(0.0) }; ROWS];
        for (row, lane) in av.iter_mut().enumerate() {
            // caller guarantees `a.base + row * a.k_stride + l + 4 <= a.data.len()`
            // via the reduction-dim contiguity and row-count checks in
            // `neon_tile_plan` and its `run_reduce` call site.
            let offset = (a.base + row as i64 * a.k_stride + l as i64) as usize;
            *lane = unsafe { vld1q_f32(a.data.as_ptr().add(offset)) };
        }
        let mut bv = [unsafe { vdupq_n_f32(0.0) }; TILE_COLS];
        for (column, lane) in bv.iter_mut().enumerate() {
            // caller guarantees `b.base + column * b.k_stride + l + 4 <= b.data.len()`
            // by the same contiguity and column-count checks.
            let offset = (b.base + column as i64 * b.k_stride + l as i64) as usize;
            *lane = unsafe { vld1q_f32(b.data.as_ptr().add(offset)) };
        }
        for (row, acc_row) in acc.iter_mut().enumerate() {
            let row_vector = av[row];
            for (acc_lane, &bv_lane) in acc_row.iter_mut().zip(bv.iter()) {
                // both operands are `float32x4_t`; NEON `fmla.4s` has no
                // aliasing hazard between distinct accumulator lanes.
                *acc_lane = unsafe { vfmaq_f32(*acc_lane, row_vector, bv_lane) };
            }
        }
    }
    for (row, (acc_row, out_row)) in acc.iter().zip(out.iter_mut()).enumerate() {
        for (column, (&acc_lane, out_value)) in acc_row.iter().zip(out_row.iter_mut()).enumerate() {
            // horizontal combine of one lane-group; sound for any float
            // values, no aliasing or bounds precondition beyond `acc` being
            // fully initialized above.
            let mut total = *out_value + unsafe { vaddvq_f32(acc_lane) };
            for l in steps * 4..k {
                let offset_a = (a.base + row as i64 * a.k_stride + l as i64) as usize;
                let offset_b = (b.base + column as i64 * b.k_stride + l as i64) as usize;
                total = a.data[offset_a].mul_add(b.data[offset_b], total);
            }
            *out_value = total;
        }
    }
}

/// The pre-ROW-12 strict left-to-right fold, kept verbatim as the
/// `len < DOT_LANES` fallback for [`dot_fold_multi_accumulator_binary`] —
/// too few terms for independent lanes to pay for themselves, and this
/// keeps tiny-`k` folds byte-for-byte identical to pre-ROW-12 behavior.
fn dot_fold_scalar_binary<F, R>(op: F, reduce: R, slice_a: &[f32], slice_b: &[f32], fold: DotFold) -> f32
where
    F: Fn(f32, f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if fold.seeded {
        let mut acc = fold.init;
        for (&value_a, &value_b) in slice_a.iter().zip(slice_b) {
            acc = reduce(acc, op(value_a, value_b));
        }
        acc
    } else if let ([first_a, rest_a @ ..], [first_b, rest_b @ ..]) = (slice_a, slice_b) {
        let mut acc = op(*first_a, *first_b);
        for (&value_a, &value_b) in rest_a.iter().zip(rest_b) {
            acc = reduce(acc, op(value_a, value_b));
        }
        acc
    } else {
        fold.init
    }
}

/// `DOT_LANES` independent partial accumulators, one per position in each
/// `DOT_LANES`-wide `chunks_exact` block of `slice_a`/`slice_b`, combined
/// via `reduce` (associative by construction: `Add`/`Multiply`/`Maximum`/
/// `Minimum`, the only four reduce ops this fast path ever specializes
/// for), then folded into one scalar via a single `DOT_LANES`-wide
/// horizontal combine at the end. Reassociates the sum relative to the
/// strict left-to-right fold — the numeric result differs from the naive
/// triple loop by float rounding, same as Accelerate/OpenBLAS/ggml (ROW
/// 12, `proxima-tensor/docs/discipline.md`). Operates on matching-length
/// slices via `chunks_exact` (not manual indexing) so the length relation
/// LLVM needs to elide bounds checks and vectorize is visible in the
/// source, the same technique [`reduce_width_binary_monomorphic`] already
/// relies on. `seeded == false` (the `ReduceInit::FirstElement` case)
/// seeds each lane with its own first block value instead of `fold.init`,
/// so no lane ever combines with a non-identity `fold.init` value.
#[inline(always)]
fn dot_fold_multi_accumulator_binary<F, R>(op: F, reduce: R, slice_a: &[f32], slice_b: &[f32], fold: DotFold) -> f32
where
    F: Fn(f32, f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if fold.len < DOT_LANES {
        return dot_fold_scalar_binary(op, reduce, slice_a, slice_b, fold);
    }
    let (chunks_a, remainder_a) = slice_a.as_chunks::<DOT_LANES>();
    let (chunks_b, remainder_b) = slice_b.as_chunks::<DOT_LANES>();
    let mut lanes = [fold.init; DOT_LANES];
    let mut seeded = fold.seeded;
    for (chunk_a, chunk_b) in chunks_a.iter().zip(chunks_b) {
        if seeded {
            for ((lane, &value_a), &value_b) in lanes.iter_mut().zip(chunk_a).zip(chunk_b) {
                *lane = reduce(*lane, op(value_a, value_b));
            }
        } else {
            for ((lane, &value_a), &value_b) in lanes.iter_mut().zip(chunk_a).zip(chunk_b) {
                *lane = op(value_a, value_b);
            }
            seeded = true;
        }
    }
    let mut acc = lanes[0];
    for &lane in &lanes[1..] {
        acc = reduce(acc, lane);
    }
    for (&value_a, &value_b) in remainder_a.iter().zip(remainder_b) {
        let value = op(value_a, value_b);
        acc = if seeded { reduce(acc, value) } else { value };
        seeded = true;
    }
    acc
}

/// The pre-ROW-12 strict left-to-right fold, kept verbatim as the
/// `len < DOT_LANES` fallback for [`dot_fold_multi_accumulator_unary`].
fn dot_fold_scalar_unary<F, R>(op: F, reduce: R, slice: &[f32], fold: DotFold) -> f32
where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if fold.seeded {
        let mut acc = fold.init;
        for &raw_value in slice {
            acc = reduce(acc, op(raw_value));
        }
        acc
    } else if let [first, rest @ ..] = slice {
        let mut acc = op(*first);
        for &raw_value in rest {
            acc = reduce(acc, op(raw_value));
        }
        acc
    } else {
        fold.init
    }
}

/// Same discipline as [`dot_fold_multi_accumulator_binary`], one operand.
#[inline(always)]
fn dot_fold_multi_accumulator_unary<F, R>(op: F, reduce: R, slice: &[f32], fold: DotFold) -> f32
where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if fold.len < DOT_LANES {
        return dot_fold_scalar_unary(op, reduce, slice, fold);
    }
    let (chunks, remainder) = slice.as_chunks::<DOT_LANES>();
    let mut lanes = [fold.init; DOT_LANES];
    let mut seeded = fold.seeded;
    for chunk in chunks {
        if seeded {
            for (lane, &value) in lanes.iter_mut().zip(chunk) {
                *lane = reduce(*lane, op(value));
            }
        } else {
            for (lane, &value) in lanes.iter_mut().zip(chunk) {
                *lane = op(value);
            }
            seeded = true;
        }
    }
    let mut acc = lanes[0];
    for &lane in &lanes[1..] {
        acc = reduce(acc, lane);
    }
    for &value in remainder {
        let mapped = op(value);
        acc = if seeded { reduce(acc, mapped) } else { mapped };
        seeded = true;
    }
    acc
}

/// The contraction-dim counterpart of [`reduce_width_fast`]: instead of
/// accumulating one value per width position across many calls (one per `k`),
/// folds the whole contraction range for ONE output position in a single
/// call. Used when the width dim's own stride disqualifies
/// [`reduce_width_fast`] but the contraction dim is affine on every operand
/// the body shape reads (transposed-B GEMM: `k` is contiguous on both
/// operands even though `n` is not).
#[inline(always)]
fn reduce_dot_fast(
    shape: &BodyShape,
    reduce_op: ScalarOp,
    raw: &[&[f32]],
    running: &[i64],
    reduction_strides: &[i64],
    fold: DotFold,
) -> f32 {
    let span_of = |index: u16| {
        let index = index as usize;
        OperandSpan {
            data: raw[index],
            base: running[index] as usize,
            stride: reduction_strides[index] as usize,
        }
    };
    match *shape {
        BodyShape::Unary(op, a) => reduce_dot_unary(op, reduce_op, span_of(a), fold),
        BodyShape::Binary(op, a, b) => reduce_dot_binary(op, reduce_op, span_of(a), span_of(b), fold),
        BodyShape::FusedAdamUpdate(..) | BodyShape::Generic(_) => {
            unreachable!("fast path is never entered for a Generic or FusedAdamUpdate body shape")
        }
    }
}

/// Same op/reduce_op monomorphized-closure dispatch as [`reduce_width_unary`],
/// folding to one scalar instead of accumulating across a width slice.
fn reduce_dot_unary(op: ScalarOp, reduce_op: ScalarOp, span: OperandSpan, fold: DotFold) -> f32 {
    macro_rules! unary_op_arm {
        ($f:expr) => {
            match reduce_op {
                ScalarOp::Add => reduce_dot_unary_monomorphic($f, |acc: f32, v: f32| acc + v, span, fold),
                ScalarOp::Multiply => reduce_dot_unary_monomorphic($f, |acc: f32, v: f32| acc * v, span, fold),
                ScalarOp::Maximum => reduce_dot_unary_monomorphic($f, |acc: f32, v: f32| acc.max(v), span, fold),
                ScalarOp::Minimum => reduce_dot_unary_monomorphic($f, |acc: f32, v: f32| acc.min(v), span, fold),
                _ => reduce_dot_unary_scalar_dispatch(op, reduce_op, span, fold),
            }
        };
    }
    match op {
        ScalarOp::Identity => unary_op_arm!(|a: f32| a),
        ScalarOp::Negate => unary_op_arm!(|a: f32| -a),
        ScalarOp::Reciprocal => unary_op_arm!(|a: f32| 1.0 / a),
        ScalarOp::Exponential => unary_op_arm!(|a: f32| a.exp()),
        ScalarOp::Logarithm => unary_op_arm!(|a: f32| a.ln()),
        ScalarOp::SquareRoot => unary_op_arm!(|a: f32| a.sqrt()),
        ScalarOp::Tanh => unary_op_arm!(|a: f32| a.tanh()),
        _ => reduce_dot_unary_scalar_dispatch(op, reduce_op, span, fold),
    }
}

/// `seeded` is branched on ONCE, outside the fold loop, same discipline as
/// [`reduce_width_unary_monomorphic`] — the loop body below contains exactly
/// one call to `op` and, past the first term, one call to `reduce`, both
/// inlined non-capturing closures. A strided span delegates to
/// [`reduce_dot_unary_monomorphic_strided`] before the stride-0/1 arms run.
#[inline(always)]
fn reduce_dot_unary_monomorphic<F, R>(op: F, reduce: R, span: OperandSpan, fold: DotFold) -> f32
where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if span.is_strided() {
        return reduce_dot_unary_monomorphic_strided(op, reduce, span, fold);
    }
    if span.stride == 1 {
        let slice = &span.data[span.base..span.base + fold.len];
        dot_fold_multi_accumulator_unary(op, reduce, slice, fold)
    } else {
        let value = op(span.data[span.base]);
        if fold.seeded {
            let mut acc = fold.init;
            for _ in 0..fold.len {
                acc = reduce(acc, value);
            }
            acc
        } else if fold.len == 0 {
            fold.init
        } else {
            let mut acc = value;
            for _ in 1..fold.len {
                acc = reduce(acc, value);
            }
            acc
        }
    }
}

/// Mirrors [`reduce_dot_unary_monomorphic`]'s `seeded`/`fold.len == 0`
/// handling for a stride > 1 span, reading each term with
/// [`OperandSpan::at`] instead of a hoisted broadcast scalar — never routed
/// through [`dot_fold_multi_accumulator_unary`], which deliberately
/// reassociates and would silently change output for this newly-widened case.
#[inline(always)]
fn reduce_dot_unary_monomorphic_strided<F, R>(op: F, reduce: R, span: OperandSpan, fold: DotFold) -> f32
where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if fold.seeded {
        let mut acc = fold.init;
        for position in 0..fold.len {
            acc = reduce(acc, op(span.at(position)));
        }
        acc
    } else if fold.len == 0 {
        fold.init
    } else {
        let mut acc = op(span.at(0));
        for position in 1..fold.len {
            acc = reduce(acc, op(span.at(position)));
        }
        acc
    }
}

/// The unaccelerated fallback for a `reduce_op` outside {Add, Multiply,
/// Maximum, Minimum} — same numerical result as
/// [`reduce_dot_unary_monomorphic`], dispatched per term via
/// [`apply_scalar_op`]/[`combine_reduction`]. [`OperandSpan::at`] already
/// generalizes over every stride.
fn reduce_dot_unary_scalar_dispatch(op: ScalarOp, reduce_op: ScalarOp, span: OperandSpan, fold: DotFold) -> f32 {
    let mut acc = fold.init;
    let mut seeded = fold.seeded;
    for step in 0..fold.len {
        let value = apply_scalar_op(op, &[span.at(step)]);
        acc = combine_reduction(reduce_op, acc, value, seeded);
        seeded = true;
    }
    acc
}

/// Same discipline as [`reduce_dot_unary`], for the two-operand case — the
/// contraction-dim counterpart of [`reduce_width_binary`].
fn reduce_dot_binary(op: ScalarOp, reduce_op: ScalarOp, a: OperandSpan, b: OperandSpan, fold: DotFold) -> f32 {
    // the multiply-accumulate case — every contraction in every matmul —
    // taken before the generic closure dispatch, because `mul_add` has to be
    // asked for by name (see `dot_fold_fused_multiply_add`). `a.stride == 1
    // && b.stride == 1` already excludes any stride > 1 literally, so this
    // gate does not need widening alongside `operand_is_affine`.
    if FUSED_MULTIPLY_ADD
        && fold.seeded
        && fold.len >= DOT_LANES
        && a.stride == 1
        && b.stride == 1
        && matches!((op, reduce_op), (ScalarOp::Multiply, ScalarOp::Add))
    {
        let slice_a = &a.data[a.base..a.base + fold.len];
        let slice_b = &b.data[b.base..b.base + fold.len];
        return dot_fold_fused_multiply_add(slice_a, slice_b, fold);
    }
    macro_rules! binary_op_arm {
        ($f:expr) => {
            match reduce_op {
                ScalarOp::Add => reduce_dot_binary_monomorphic($f, |acc: f32, v: f32| acc + v, a, b, fold),
                ScalarOp::Multiply => reduce_dot_binary_monomorphic($f, |acc: f32, v: f32| acc * v, a, b, fold),
                ScalarOp::Maximum => reduce_dot_binary_monomorphic($f, |acc: f32, v: f32| acc.max(v), a, b, fold),
                ScalarOp::Minimum => reduce_dot_binary_monomorphic($f, |acc: f32, v: f32| acc.min(v), a, b, fold),
                _ => reduce_dot_binary_scalar_dispatch(op, reduce_op, a, b, fold),
            }
        };
    }
    match op {
        ScalarOp::Add => binary_op_arm!(|x: f32, y: f32| x + y),
        ScalarOp::Subtract => binary_op_arm!(|x: f32, y: f32| x - y),
        ScalarOp::Multiply => binary_op_arm!(|x: f32, y: f32| x * y),
        ScalarOp::Divide => binary_op_arm!(|x: f32, y: f32| x / y),
        ScalarOp::Maximum => binary_op_arm!(|x: f32, y: f32| x.max(y)),
        ScalarOp::Minimum => binary_op_arm!(|x: f32, y: f32| x.min(y)),
        ScalarOp::Greater => binary_op_arm!(|x: f32, y: f32| f32::from(u8::from(x > y))),
        ScalarOp::Equal => binary_op_arm!(|x: f32, y: f32| f32::from(u8::from((x - y).abs() == 0.0))),
        _ => reduce_dot_binary_scalar_dispatch(op, reduce_op, a, b, fold),
    }
}

/// The `(true, true)` arm folds via [`dot_fold_multi_accumulator_binary`]
/// (`DOT_LANES` independent partial sums, reassociated relative to the
/// naive triple loop — ROW 12) instead of one strict left-to-right chain.
/// This is the exact shape a transposed-B GEMM's per-output-element dot
/// product takes (`proxima-tensor/docs/discipline.md` ROW 10/11/12). A
/// strided operand delegates to [`reduce_dot_binary_monomorphic_strided`]
/// before this match runs.
#[inline(always)]
fn reduce_dot_binary_monomorphic<F, R>(op: F, reduce: R, a: OperandSpan, b: OperandSpan, fold: DotFold) -> f32
where
    F: Fn(f32, f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if a.is_strided() || b.is_strided() {
        return reduce_dot_binary_monomorphic_strided(op, reduce, a, b, fold);
    }
    match (a.stride == 1, b.stride == 1) {
        (true, true) => {
            let slice_a = &a.data[a.base..a.base + fold.len];
            let slice_b = &b.data[b.base..b.base + fold.len];
            dot_fold_multi_accumulator_binary(op, reduce, slice_a, slice_b, fold)
        }
        (true, false) => {
            let slice_a = &a.data[a.base..a.base + fold.len];
            let value_b = b.data[b.base];
            if fold.seeded {
                let mut acc = fold.init;
                for &value_a in slice_a {
                    acc = reduce(acc, op(value_a, value_b));
                }
                acc
            } else if let [first_a, rest_a @ ..] = slice_a {
                let mut acc = op(*first_a, value_b);
                for &value_a in rest_a {
                    acc = reduce(acc, op(value_a, value_b));
                }
                acc
            } else {
                fold.init
            }
        }
        (false, true) => {
            let value_a = a.data[a.base];
            let slice_b = &b.data[b.base..b.base + fold.len];
            if fold.seeded {
                let mut acc = fold.init;
                for &value_b in slice_b {
                    acc = reduce(acc, op(value_a, value_b));
                }
                acc
            } else if let [first_b, rest_b @ ..] = slice_b {
                let mut acc = op(value_a, *first_b);
                for &value_b in rest_b {
                    acc = reduce(acc, op(value_a, value_b));
                }
                acc
            } else {
                fold.init
            }
        }
        (false, false) => {
            let value_a = a.data[a.base];
            let value_b = b.data[b.base];
            let value = op(value_a, value_b);
            if fold.seeded {
                let mut acc = fold.init;
                for _ in 0..fold.len {
                    acc = reduce(acc, value);
                }
                acc
            } else if fold.len == 0 {
                fold.init
            } else {
                let mut acc = value;
                for _ in 1..fold.len {
                    acc = reduce(acc, value);
                }
                acc
            }
        }
    }
}

/// Mirrors [`reduce_dot_binary_monomorphic`]'s `seeded`/`fold.len == 0`
/// handling one position at a time via [`OperandSpan::at`], for the case at
/// least one of `a`/`b` has a stride > 1 — never routed through
/// [`dot_fold_multi_accumulator_binary`], which reassociates.
#[inline(always)]
fn reduce_dot_binary_monomorphic_strided<F, R>(op: F, reduce: R, a: OperandSpan, b: OperandSpan, fold: DotFold) -> f32
where
    F: Fn(f32, f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if fold.seeded {
        let mut acc = fold.init;
        for position in 0..fold.len {
            acc = reduce(acc, op(a.at(position), b.at(position)));
        }
        acc
    } else if fold.len == 0 {
        fold.init
    } else {
        let mut acc = op(a.at(0), b.at(0));
        for position in 1..fold.len {
            acc = reduce(acc, op(a.at(position), b.at(position)));
        }
        acc
    }
}

/// The unaccelerated fallback for a `reduce_op` outside {Add, Multiply,
/// Maximum, Minimum}. [`OperandSpan::at`] already generalizes over every
/// stride, so both reads collapse to one expression regardless of stride.
fn reduce_dot_binary_scalar_dispatch(op: ScalarOp, reduce_op: ScalarOp, a: OperandSpan, b: OperandSpan, fold: DotFold) -> f32 {
    let mut acc = fold.init;
    let mut seeded = fold.seeded;
    for step in 0..fold.len {
        let value = apply_scalar_op(op, &[a.at(step), b.at(step)]);
        acc = combine_reduction(reduce_op, acc, value, seeded);
        seeded = true;
    }
    acc
}

/// The width-loop fast path for [`run_elementwise`]: no accumulator, no
/// `reduce_op` — every position gets a fresh value written straight to
/// `out`. Same eligibility gate as `run_reduce`
/// ([`body_shape_is_affine_fast_path`]), same [`OperandSpan`] reads, same
/// monomorphized-closure-per-op dispatch technique as ROW 4
/// (`proxima-tensor/docs/discipline.md` ROW 5). `step_values` is only read
/// by the `Generic` arm ([`elementwise_width_generic`]); `Unary`/`Binary`
/// ignore it, same as [`eval_body_shape`]'s own split.
#[inline(always)]
fn elementwise_width_fast(
    shape: &BodyShape,
    raw: &[&[f32]],
    running: &[i64],
    strides: &[i64],
    out: &mut [f32],
    step_values: &mut [f32],
) {
    let span_of = |index: u16| {
        let index = index as usize;
        OperandSpan {
            data: raw[index],
            base: running[index] as usize,
            stride: strides[index] as usize,
        }
    };
    match *shape {
        BodyShape::Unary(op, a) => elementwise_width_unary(op, span_of(a), out),
        BodyShape::Binary(op, a, b) => elementwise_width_binary(op, span_of(a), span_of(b), out),
        BodyShape::FusedAdamUpdate(roles, _) => elementwise_width_fused_adam_update(roles, raw, running, out),
        BodyShape::Generic(body) => elementwise_width_generic(body, raw, running, strides, out, step_values),
    }
}

/// The dedicated, register-resident kernel for [`BodyShape::FusedAdamUpdate`]
/// (`docs/discipline.md` ROW 179) — the eight bias-correction scalar steps
/// (`step*ln(beta) -> exp -> 1-that -> reciprocal`, once each for `m` and
/// `v`) are pure rank-0 arithmetic on values that never vary across the
/// element loop, so they are hoisted and computed ONCE, exactly like
/// `run_elementwise_range`'s own loop-invariant-stride doc already
/// establishes for a genuine broadcast operand — reducing the per-element
/// body to the same 8-op chain ROW 176's own standalone `adam_update`
/// microbench measured at 0.2612 ns/element (`m_hat = m*recip_bias1`
/// through `out = param-scaled_update`). `m`/`v`/`param` are read as plain
/// contiguous slices (never through [`OperandSpan`]'s stride-generalized
/// `at` accessor) because [`fused_adam_update_is_affine_fast_path`] already
/// guarantees stride 1 — this is what lets LLVM auto-vectorize the whole
/// per-element chain the same way it already does for that microbench,
/// instead of branching or calling out per step the way a
/// runtime-dispatched interpreter over an arbitrary op sequence must (ROW
/// 177's own `candidate_b`/`candidate_c`, both measured WORSE than the
/// shipped tiled path they were meant to replace). Every arithmetic step
/// and its operand order matches [`apply_scalar_op`] exactly (`Multiply`/
/// `Add` are commutative so operand order is moot there; `Subtract`/
/// `Reciprocal` are not, and every subtraction/reciprocal here matches its
/// own `BodyStep`'s `apply_scalar_op` argument order bit-for-bit, per
/// [`detect_adam_update_roles`]'s own step-by-step doc) — a pure reorder of
/// the SAME expression tree (scalar hoisting included: a rank-0 value
/// computed once outside the loop is bit-identical to the same value
/// recomputed, unchanged, at every position inside it), not a
/// reassociation, so output is bit-identical to `elementwise_width_generic`'s
/// own tiled walk of the identical [`ComposedBody`].
#[inline(always)]
fn elementwise_width_fused_adam_update(roles: AdamUpdateRoles, raw: &[&[f32]], running: &[i64], out: &mut [f32]) {
    let width = out.len();
    let slice_of = |index: u16| {
        let index = index as usize;
        let base = running[index] as usize;
        &raw[index][base..base + width]
    };
    let scalar_of = |index: u16| {
        let index = index as usize;
        raw[index][running[index] as usize]
    };
    let m = slice_of(roles.m);
    let v = slice_of(roles.v);
    let param = slice_of(roles.param);
    let learning_rate = scalar_of(roles.learning_rate);
    let epsilon = scalar_of(roles.epsilon);

    let bias1_power = (scalar_of(roles.step_for_bias1) * scalar_of(roles.ln_beta1)).exp();
    let recip_bias1 = 1.0 / (scalar_of(roles.one_for_bias1) - bias1_power);
    let bias2_power = (scalar_of(roles.step_for_bias2) * scalar_of(roles.ln_beta2)).exp();
    let recip_bias2 = 1.0 / (scalar_of(roles.one_for_bias2) - bias2_power);

    for index in 0..width {
        let m_hat = m[index] * recip_bias1;
        let v_hat = v[index] * recip_bias2;
        let sqrt_v_hat = v_hat.sqrt();
        let denominator = sqrt_v_hat + epsilon;
        let recip_denominator = 1.0 / denominator;
        let update = m_hat * recip_denominator;
        let scaled_update = learning_rate * update;
        out[index] = param[index] - scaled_update;
    }
}

/// The width-loop fast path for a fused multi-step [`ComposedBody`]
/// (`BodyShape::Generic`) — the same straight-line shape
/// [`elementwise_width_fast`]'s `Unary`/`Binary` arms already give a
/// single-`ScalarOp` body, generalized to [`apply_body`]'s own step-by-step
/// evaluation instead of a bespoke per-arity function.
///
/// Step-outer, position-inner (the reverse of the position-outer loop this
/// replaced): each step resolves its [`StepArg`]s to plain [`OperandSpan`]s
/// **once**, the same struct [`elementwise_width_unary`]/`_binary` already
/// read, then hands them to that step's arity-specific monomorphic function
/// — [`elementwise_width_unary_monomorphic`] and
/// [`elementwise_width_binary_monomorphic`] are reused verbatim for arity
/// 1/2, [`elementwise_width_ternary_monomorphic`] added for `Select`'s arity
/// 3. Each of those matches `contiguous`/`broadcast` per operand **once,
/// before** the position loop, so the loop body itself is a fixed slice (or
/// scalar) walk with no per-element branch — earlier `ArgKind::at` case-fell
/// back to a per-element match instead, which measured slower, not faster,
/// than the naive path it replaced. `step_values` is a
/// `body.steps.len() * out.len()` flat row table (row `index` holds step
/// `index`'s value at every position) instead of one scalar reused across
/// steps, because `StepArg::Step` is backwards-only (`BodyStep`'s own doc)
/// and every earlier row must survive until the last step reads it. A
/// `StepArg::Step` read is always a whole prior row, so it is always
/// `contiguous: true` — never the loop-invariant-broadcast case, which only
/// ever applies to a genuine stride-0 `StepArg::Operand`. Evaluation order
/// and every `apply_scalar_op` call match [`apply_body`]'s scalar path
/// exactly: output is bit-identical (`proxima-tensor/docs/discipline.md`
/// ROW 5).
#[inline(always)]
fn elementwise_width_generic(
    body: &ComposedBody,
    raw: &[&[f32]],
    running: &[i64],
    strides: &[i64],
    out: &mut [f32],
    step_values: &mut [f32],
) {
    let mut tile_start = 0usize;
    while tile_start < out.len() {
        let tile_len = GENERIC_WIDTH_TILE.min(out.len() - tile_start);
        elementwise_width_generic_tile(
            body,
            raw,
            running,
            strides,
            tile_start,
            &mut out[tile_start..tile_start + tile_len],
            step_values,
        );
        tile_start += tile_len;
    }
}

/// Width block one [`elementwise_width_generic`] pass evaluates the whole
/// fused chain over. Step-outer/position-inner evaluation makes one full
/// pass across the width PER STEP, so a 6-step body on a 14336-wide row
/// streamed 6 x 56 KiB of intermediates through L2 and allocated a 344 KiB
/// `step_values` table per node call — measured by this crate's own
/// `ELEMENTWISE_STEP_VALUES_TICKS` at 1010.8 ns/call over 771 calls per
/// decode step. Blocking the row caps that scratch at `steps * 512` floats
/// whatever the row width, and keeps every intermediate the chain produces
/// L1-resident between the step that writes it and the step that reads it.
/// 512 `f32` is 2 KiB per step row, 12 KiB for the deepest body this
/// program builds, against this core's 128 KiB L1D.
const GENERIC_WIDTH_TILE: usize = 512;

/// One [`GENERIC_WIDTH_TILE`] block of [`elementwise_width_generic`].
/// `tile_start` offsets each operand's own width span by its own stride —
/// the only thing blocking changes. Every output position is computed by
/// the same steps in the same order against the same inputs it would have
/// been at full width, so output is bit-identical.
#[inline(always)]
fn elementwise_width_generic_tile(
    body: &ComposedBody,
    raw: &[&[f32]],
    running: &[i64],
    strides: &[i64],
    tile_start: usize,
    out: &mut [f32],
    step_values: &mut [f32],
) {
    let width = out.len();
    let empty: &[f32] = &[];
    for (index, step) in body.steps.iter().enumerate() {
        let (earlier, rest) = step_values.split_at_mut(index * width);
        let row = &mut rest[..width];

        let mut spans = [OperandSpan {
            data: empty,
            base: 0,
            stride: 1,
        }; 3];
        for (arg_slot, arg) in step.args.iter().enumerate() {
            spans[arg_slot] = match *arg {
                StepArg::Operand(operand_index) => {
                    let operand_index = operand_index as usize;
                    let stride = strides[operand_index] as usize;
                    OperandSpan {
                        data: raw[operand_index],
                        base: running[operand_index] as usize + tile_start * stride,
                        stride,
                    }
                }
                StepArg::Step(step_index) => {
                    let step_index = step_index as usize;
                    OperandSpan {
                        data: &earlier[step_index * width..(step_index + 1) * width],
                        base: 0,
                        stride: 1,
                    }
                }
            };
        }
        elementwise_width_generic_step(step.op, &spans, row);
    }
    let last = body.steps.len() - 1;
    out.copy_from_slice(&step_values[last * width..(last + 1) * width]);
}

/// Picks `step`'s `ScalarOp` **once** and dispatches to the matching arity's
/// monomorphic function — the `Generic`-body counterpart of
/// [`elementwise_width_unary`]/[`elementwise_width_binary`]'s own
/// once-per-call dispatch, generalized to a `Select`-only ternary case.
#[inline(always)]
fn elementwise_width_generic_step(op: ScalarOp, spans: &[OperandSpan; 3], row: &mut [f32]) {
    match op {
        ScalarOp::Identity => elementwise_width_unary_monomorphic(|a: f32| a, spans[0], row),
        ScalarOp::Negate => elementwise_width_unary_monomorphic(|a: f32| -a, spans[0], row),
        ScalarOp::Reciprocal => elementwise_width_unary_monomorphic(|a: f32| 1.0 / a, spans[0], row),
        ScalarOp::Exponential => elementwise_width_unary_monomorphic(|a: f32| a.exp(), spans[0], row),
        ScalarOp::Logarithm => elementwise_width_unary_monomorphic(|a: f32| a.ln(), spans[0], row),
        ScalarOp::SquareRoot => elementwise_width_unary_monomorphic(|a: f32| a.sqrt(), spans[0], row),
        ScalarOp::Tanh => elementwise_width_unary_monomorphic(|a: f32| a.tanh(), spans[0], row),
        ScalarOp::Erf => elementwise_width_unary_monomorphic(erf_f32, spans[0], row),
        ScalarOp::Add => elementwise_width_binary_monomorphic(|a: f32, b: f32| a + b, spans[0], spans[1], row),
        ScalarOp::Subtract => {
            elementwise_width_binary_monomorphic(|a: f32, b: f32| a - b, spans[0], spans[1], row);
        }
        ScalarOp::Multiply => {
            elementwise_width_binary_monomorphic(|a: f32, b: f32| a * b, spans[0], spans[1], row);
        }
        ScalarOp::Divide => elementwise_width_binary_monomorphic(|a: f32, b: f32| a / b, spans[0], spans[1], row),
        ScalarOp::Maximum => {
            elementwise_width_binary_monomorphic(|a: f32, b: f32| a.max(b), spans[0], spans[1], row);
        }
        ScalarOp::Minimum => {
            elementwise_width_binary_monomorphic(|a: f32, b: f32| a.min(b), spans[0], spans[1], row);
        }
        ScalarOp::Greater => elementwise_width_binary_monomorphic(
            |a: f32, b: f32| f32::from(u8::from(a > b)),
            spans[0],
            spans[1],
            row,
        ),
        ScalarOp::Equal => elementwise_width_binary_monomorphic(
            |a: f32, b: f32| f32::from(u8::from((a - b).abs() == 0.0)),
            spans[0],
            spans[1],
            row,
        ),
        ScalarOp::Select => elementwise_width_ternary_monomorphic(
            |condition: f32, when_true: f32, when_false: f32| {
                if condition != 0.0 {
                    when_true
                } else {
                    when_false
                }
            },
            spans[0],
            spans[1],
            spans[2],
            row,
        ),
    }
}

/// The `Select`-arity counterpart of
/// [`elementwise_width_binary_monomorphic`]: every operand's
/// `contiguous`/`broadcast` case is matched **once**, before the position
/// loop, so each of the eight combinations runs a fixed slice-or-scalar walk
/// with no per-element branch. A fully-broadcast step (all three operands
/// stride-0) computes `op` exactly once and splats the single result, same
/// as the all-broadcast arm of the binary/unary cases — never re-evaluated
/// per position, since none of its inputs vary by position. Any operand with
/// a stride > 1 delegates to [`elementwise_width_ternary_monomorphic_strided`]
/// before this match runs, so the eight combinations below still only ever
/// see stride 0 or 1.
#[inline(always)]
fn elementwise_width_ternary_monomorphic<F>(
    op: F,
    condition: OperandSpan,
    when_true: OperandSpan,
    when_false: OperandSpan,
    row: &mut [f32],
) where
    F: Fn(f32, f32, f32) -> f32,
{
    if condition.is_strided() || when_true.is_strided() || when_false.is_strided() {
        return elementwise_width_ternary_monomorphic_strided(op, condition, when_true, when_false, row);
    }
    let width = row.len();
    match (condition.stride == 1, when_true.stride == 1, when_false.stride == 1) {
        (true, true, true) => {
            let condition_slice = &condition.data[condition.base..condition.base + width];
            let when_true_slice = &when_true.data[when_true.base..when_true.base + width];
            let when_false_slice = &when_false.data[when_false.base..when_false.base + width];
            for (((slot, &condition_value), &when_true_value), &when_false_value) in row
                .iter_mut()
                .zip(condition_slice)
                .zip(when_true_slice)
                .zip(when_false_slice)
            {
                *slot = op(condition_value, when_true_value, when_false_value);
            }
        }
        (true, true, false) => {
            let condition_slice = &condition.data[condition.base..condition.base + width];
            let when_true_slice = &when_true.data[when_true.base..when_true.base + width];
            let when_false_value = when_false.data[when_false.base];
            for ((slot, &condition_value), &when_true_value) in
                row.iter_mut().zip(condition_slice).zip(when_true_slice)
            {
                *slot = op(condition_value, when_true_value, when_false_value);
            }
        }
        (true, false, true) => {
            let condition_slice = &condition.data[condition.base..condition.base + width];
            let when_true_value = when_true.data[when_true.base];
            let when_false_slice = &when_false.data[when_false.base..when_false.base + width];
            for ((slot, &condition_value), &when_false_value) in
                row.iter_mut().zip(condition_slice).zip(when_false_slice)
            {
                *slot = op(condition_value, when_true_value, when_false_value);
            }
        }
        (true, false, false) => {
            let condition_slice = &condition.data[condition.base..condition.base + width];
            let when_true_value = when_true.data[when_true.base];
            let when_false_value = when_false.data[when_false.base];
            for (slot, &condition_value) in row.iter_mut().zip(condition_slice) {
                *slot = op(condition_value, when_true_value, when_false_value);
            }
        }
        (false, true, true) => {
            let condition_value = condition.data[condition.base];
            let when_true_slice = &when_true.data[when_true.base..when_true.base + width];
            let when_false_slice = &when_false.data[when_false.base..when_false.base + width];
            for ((slot, &when_true_value), &when_false_value) in
                row.iter_mut().zip(when_true_slice).zip(when_false_slice)
            {
                *slot = op(condition_value, when_true_value, when_false_value);
            }
        }
        (false, true, false) => {
            let condition_value = condition.data[condition.base];
            let when_true_slice = &when_true.data[when_true.base..when_true.base + width];
            let when_false_value = when_false.data[when_false.base];
            for (slot, &when_true_value) in row.iter_mut().zip(when_true_slice) {
                *slot = op(condition_value, when_true_value, when_false_value);
            }
        }
        (false, false, true) => {
            let condition_value = condition.data[condition.base];
            let when_true_value = when_true.data[when_true.base];
            let when_false_slice = &when_false.data[when_false.base..when_false.base + width];
            for (slot, &when_false_value) in row.iter_mut().zip(when_false_slice) {
                *slot = op(condition_value, when_true_value, when_false_value);
            }
        }
        (false, false, false) => {
            let value = op(
                condition.data[condition.base],
                when_true.data[when_true.base],
                when_false.data[when_false.base],
            );
            for slot in row.iter_mut() {
                *slot = value;
            }
        }
    }
}

/// Independent per-position writes, no accumulator to reorder — reads each
/// operand with [`OperandSpan::at`], which already generalizes over stride 0,
/// 1, or any wider constant stride.
#[inline(always)]
fn elementwise_width_ternary_monomorphic_strided<F>(
    op: F,
    condition: OperandSpan,
    when_true: OperandSpan,
    when_false: OperandSpan,
    row: &mut [f32],
) where
    F: Fn(f32, f32, f32) -> f32,
{
    for (position, slot) in row.iter_mut().enumerate() {
        *slot = op(condition.at(position), when_true.at(position), when_false.at(position));
    }
}

fn elementwise_width_unary(op: ScalarOp, span: OperandSpan, out: &mut [f32]) {
    match op {
        ScalarOp::Identity => elementwise_width_unary_monomorphic(|a: f32| a, span, out),
        ScalarOp::Negate => elementwise_width_unary_monomorphic(|a: f32| -a, span, out),
        ScalarOp::Reciprocal => elementwise_width_unary_monomorphic(|a: f32| 1.0 / a, span, out),
        ScalarOp::Exponential => elementwise_width_unary_monomorphic(|a: f32| a.exp(), span, out),
        ScalarOp::Logarithm => elementwise_width_unary_monomorphic(|a: f32| a.ln(), span, out),
        ScalarOp::SquareRoot => elementwise_width_unary_monomorphic(|a: f32| a.sqrt(), span, out),
        ScalarOp::Tanh => elementwise_width_unary_monomorphic(|a: f32| a.tanh(), span, out),
        ScalarOp::Erf => elementwise_width_unary_monomorphic(erf_f32, span, out),
        ScalarOp::Add
        | ScalarOp::Subtract
        | ScalarOp::Multiply
        | ScalarOp::Divide
        | ScalarOp::Maximum
        | ScalarOp::Minimum
        | ScalarOp::Greater
        | ScalarOp::Equal
        | ScalarOp::Select => unreachable!("BodyShape::Unary only ever carries an arity-1 ScalarOp"),
    }
}

#[inline(always)]
fn elementwise_width_unary_monomorphic<F>(op: F, span: OperandSpan, out: &mut [f32])
where
    F: Fn(f32) -> f32,
{
    if span.is_strided() {
        return elementwise_width_unary_monomorphic_strided(op, span, out);
    }
    if span.stride == 1 {
        let slice = &span.data[span.base..span.base + out.len()];
        for (slot, &raw_value) in out.iter_mut().zip(slice) {
            *slot = op(raw_value);
        }
    } else {
        let value = op(span.data[span.base]);
        for slot in out.iter_mut() {
            *slot = value;
        }
    }
}

/// Independent per-position writes, no accumulator to reorder — one
/// [`OperandSpan::at`] read per position covers any stride > 1.
#[inline(always)]
fn elementwise_width_unary_monomorphic_strided<F>(op: F, span: OperandSpan, out: &mut [f32])
where
    F: Fn(f32) -> f32,
{
    for (position, slot) in out.iter_mut().enumerate() {
        *slot = op(span.at(position));
    }
}

fn elementwise_width_binary(op: ScalarOp, a: OperandSpan, b: OperandSpan, out: &mut [f32]) {
    match op {
        ScalarOp::Add => elementwise_width_binary_monomorphic(|x: f32, y: f32| x + y, a, b, out),
        ScalarOp::Subtract => elementwise_width_binary_monomorphic(|x: f32, y: f32| x - y, a, b, out),
        ScalarOp::Multiply => elementwise_width_binary_monomorphic(|x: f32, y: f32| x * y, a, b, out),
        ScalarOp::Divide => elementwise_width_binary_monomorphic(|x: f32, y: f32| x / y, a, b, out),
        ScalarOp::Maximum => elementwise_width_binary_monomorphic(|x: f32, y: f32| x.max(y), a, b, out),
        ScalarOp::Minimum => elementwise_width_binary_monomorphic(|x: f32, y: f32| x.min(y), a, b, out),
        ScalarOp::Greater => {
            elementwise_width_binary_monomorphic(|x: f32, y: f32| f32::from(u8::from(x > y)), a, b, out)
        }
        ScalarOp::Equal => elementwise_width_binary_monomorphic(
            |x: f32, y: f32| f32::from(u8::from((x - y).abs() == 0.0)),
            a,
            b,
            out,
        ),
        ScalarOp::Identity
        | ScalarOp::Negate
        | ScalarOp::Reciprocal
        | ScalarOp::Exponential
        | ScalarOp::Logarithm
        | ScalarOp::SquareRoot
        | ScalarOp::Tanh
        | ScalarOp::Erf
        | ScalarOp::Select => unreachable!("BodyShape::Binary only ever carries an arity-2 ScalarOp"),
    }
}

#[inline(always)]
fn elementwise_width_binary_monomorphic<F>(op: F, a: OperandSpan, b: OperandSpan, out: &mut [f32])
where
    F: Fn(f32, f32) -> f32,
{
    if a.is_strided() || b.is_strided() {
        return elementwise_width_binary_monomorphic_strided(op, a, b, out);
    }
    let width = out.len();
    match (a.stride == 1, b.stride == 1) {
        (true, true) => {
            let slice_a = &a.data[a.base..a.base + width];
            let slice_b = &b.data[b.base..b.base + width];
            for ((slot, &value_a), &value_b) in out.iter_mut().zip(slice_a).zip(slice_b) {
                *slot = op(value_a, value_b);
            }
        }
        (true, false) => {
            let slice_a = &a.data[a.base..a.base + width];
            let value_b = b.data[b.base];
            for (slot, &value_a) in out.iter_mut().zip(slice_a) {
                *slot = op(value_a, value_b);
            }
        }
        (false, true) => {
            let value_a = a.data[a.base];
            let slice_b = &b.data[b.base..b.base + width];
            for (slot, &value_b) in out.iter_mut().zip(slice_b) {
                *slot = op(value_a, value_b);
            }
        }
        (false, false) => {
            let value = op(a.data[a.base], b.data[b.base]);
            for slot in out.iter_mut() {
                *slot = value;
            }
        }
    }
}

/// Independent per-position writes, no accumulator to reorder — one
/// [`OperandSpan::at`] read per operand per position covers any stride > 1.
#[inline(always)]
fn elementwise_width_binary_monomorphic_strided<F>(op: F, a: OperandSpan, b: OperandSpan, out: &mut [f32])
where
    F: Fn(f32, f32) -> f32,
{
    for (position, slot) in out.iter_mut().enumerate() {
        *slot = op(a.at(position), b.at(position));
    }
}

/// The width-loop fast path for [`run_scan`]: unlike `run_elementwise`,
/// output at each position depends on the previous position's accumulated
/// value (`accumulator = reduce_op(accumulator, value)`), a genuine
/// sequential dependency the fold cannot be vectorized around without a
/// parallel-scan restructuring this row does not attempt. What IS removed,
/// same as `run_elementwise`/`run_reduce`: the per-element gather
/// `Option` check, the `operand_values` scratch copy, and the per-element
/// `op`/`reduce_op` dispatch — all replaced by [`OperandSpan`] reads and a
/// once-per-call monomorphized closure pair, restricted to the same four
/// accelerated `reduce_op`s ROW 4 used (`Add`/`Multiply`/`Maximum`/
/// `Minimum`). The `!seeded` special case for the very first element of
/// the very first call (across the whole scan, `seeded` is never reset
/// mid-run) is resolved ONCE before the loop, not re-checked per element.
///
/// `state` bundles `seeded`/`accumulator` — they always travel together —
/// keeping this under clippy's argument-count lint the same way
/// [`OperandSpan`] does for `reduce_width_binary` (ROW 3 addendum).
struct ScanState {
    seeded: bool,
    accumulator: f32,
}

#[inline(always)]
fn scan_width_fast(
    shape: &BodyShape,
    reduce_op: ScalarOp,
    raw: &[&[f32]],
    running: &[i64],
    strides: &[i64],
    out: &mut [f32],
    state: ScanState,
) -> f32 {
    let ScanState { seeded, accumulator } = state;
    let span_of = |index: u16| {
        let index = index as usize;
        OperandSpan {
            data: raw[index],
            base: running[index] as usize,
            stride: strides[index] as usize,
        }
    };
    match *shape {
        BodyShape::Unary(op, a) => scan_width_unary(op, reduce_op, span_of(a), out, seeded, accumulator),
        BodyShape::Binary(op, a, b) => {
            scan_width_binary(op, reduce_op, span_of(a), span_of(b), out, seeded, accumulator)
        }
        BodyShape::FusedAdamUpdate(..) | BodyShape::Generic(_) => {
            unreachable!("fast path is never entered for a Generic or FusedAdamUpdate body shape")
        }
    }
}

fn scan_width_unary(op: ScalarOp, reduce_op: ScalarOp, span: OperandSpan, out: &mut [f32], seeded: bool, accumulator: f32) -> f32 {
    macro_rules! unary_op_arm {
        ($f:expr) => {
            match reduce_op {
                ScalarOp::Add => {
                    scan_width_unary_monomorphic($f, |acc: f32, v: f32| acc + v, span, out, seeded, accumulator)
                }
                ScalarOp::Multiply => {
                    scan_width_unary_monomorphic($f, |acc: f32, v: f32| acc * v, span, out, seeded, accumulator)
                }
                ScalarOp::Maximum => {
                    scan_width_unary_monomorphic($f, |acc: f32, v: f32| acc.max(v), span, out, seeded, accumulator)
                }
                ScalarOp::Minimum => {
                    scan_width_unary_monomorphic($f, |acc: f32, v: f32| acc.min(v), span, out, seeded, accumulator)
                }
                _ => scan_width_unary_scalar_dispatch(op, reduce_op, span, out, seeded, accumulator),
            }
        };
    }
    match op {
        ScalarOp::Identity => unary_op_arm!(|a: f32| a),
        ScalarOp::Negate => unary_op_arm!(|a: f32| -a),
        ScalarOp::Reciprocal => unary_op_arm!(|a: f32| 1.0 / a),
        ScalarOp::Exponential => unary_op_arm!(|a: f32| a.exp()),
        ScalarOp::Logarithm => unary_op_arm!(|a: f32| a.ln()),
        ScalarOp::SquareRoot => unary_op_arm!(|a: f32| a.sqrt()),
        ScalarOp::Tanh => unary_op_arm!(|a: f32| a.tanh()),
        _ => scan_width_unary_scalar_dispatch(op, reduce_op, span, out, seeded, accumulator),
    }
}

#[inline(always)]
fn scan_width_unary_monomorphic<F, R>(
    op: F,
    reduce: R,
    span: OperandSpan,
    out: &mut [f32],
    seeded: bool,
    accumulator: f32,
) -> f32
where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    let width = out.len();
    let mut acc = accumulator;
    let mut start = 0usize;
    if !seeded && width > 0 {
        // position 0 is `.at(0)` regardless of stride (`0 * stride == 0`
        // for any stride) -- the shapes only diverge starting at position 1.
        acc = op(span.at(0));
        out[0] = acc;
        start = 1;
    }
    if span.is_strided() {
        return scan_width_unary_monomorphic_strided(op, reduce, span, out, start, acc);
    }
    if span.stride == 1 {
        let slice = &span.data[span.base..span.base + width];
        for (slot, &raw_value) in out[start..].iter_mut().zip(&slice[start..]) {
            acc = reduce(acc, op(raw_value));
            *slot = acc;
        }
    } else {
        let value = op(span.data[span.base]);
        for slot in out[start..].iter_mut() {
            acc = reduce(acc, value);
            *slot = acc;
        }
    }
    acc
}

/// Continues [`scan_width_unary_monomorphic`]'s fold from `start` via
/// [`OperandSpan::at`], for a stride > 1 span — same strict left-to-right
/// combine order, just read position by position instead of through a
/// contiguous slice or a hoisted broadcast scalar.
#[inline(always)]
fn scan_width_unary_monomorphic_strided<F, R>(
    op: F,
    reduce: R,
    span: OperandSpan,
    out: &mut [f32],
    start: usize,
    accumulator: f32,
) -> f32
where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    let mut acc = accumulator;
    for (position, slot) in out.iter_mut().enumerate().skip(start) {
        acc = reduce(acc, op(span.at(position)));
        *slot = acc;
    }
    acc
}

fn scan_width_unary_scalar_dispatch(
    op: ScalarOp,
    reduce_op: ScalarOp,
    span: OperandSpan,
    out: &mut [f32],
    seeded: bool,
    accumulator: f32,
) -> f32 {
    let width = out.len();
    let mut acc = accumulator;
    let mut seeded = seeded;
    for (index, slot) in out.iter_mut().enumerate().take(width) {
        let value = apply_scalar_op(op, &[span.at(index)]);
        acc = combine_reduction(reduce_op, acc, value, seeded);
        seeded = true;
        *slot = acc;
    }
    acc
}

fn scan_width_binary(
    op: ScalarOp,
    reduce_op: ScalarOp,
    a: OperandSpan,
    b: OperandSpan,
    out: &mut [f32],
    seeded: bool,
    accumulator: f32,
) -> f32 {
    macro_rules! binary_op_arm {
        ($f:expr) => {
            match reduce_op {
                ScalarOp::Add => {
                    scan_width_binary_monomorphic($f, |acc: f32, v: f32| acc + v, a, b, out, seeded, accumulator)
                }
                ScalarOp::Multiply => {
                    scan_width_binary_monomorphic($f, |acc: f32, v: f32| acc * v, a, b, out, seeded, accumulator)
                }
                ScalarOp::Maximum => {
                    scan_width_binary_monomorphic($f, |acc: f32, v: f32| acc.max(v), a, b, out, seeded, accumulator)
                }
                ScalarOp::Minimum => {
                    scan_width_binary_monomorphic($f, |acc: f32, v: f32| acc.min(v), a, b, out, seeded, accumulator)
                }
                _ => scan_width_binary_scalar_dispatch(op, reduce_op, a, b, out, seeded, accumulator),
            }
        };
    }
    match op {
        ScalarOp::Add => binary_op_arm!(|x: f32, y: f32| x + y),
        ScalarOp::Subtract => binary_op_arm!(|x: f32, y: f32| x - y),
        ScalarOp::Multiply => binary_op_arm!(|x: f32, y: f32| x * y),
        ScalarOp::Divide => binary_op_arm!(|x: f32, y: f32| x / y),
        ScalarOp::Maximum => binary_op_arm!(|x: f32, y: f32| x.max(y)),
        ScalarOp::Minimum => binary_op_arm!(|x: f32, y: f32| x.min(y)),
        ScalarOp::Greater => binary_op_arm!(|x: f32, y: f32| f32::from(u8::from(x > y))),
        ScalarOp::Equal => binary_op_arm!(|x: f32, y: f32| f32::from(u8::from((x - y).abs() == 0.0))),
        _ => scan_width_binary_scalar_dispatch(op, reduce_op, a, b, out, seeded, accumulator),
    }
}

#[inline(always)]
fn scan_width_binary_monomorphic<F, R>(
    op: F,
    reduce: R,
    a: OperandSpan,
    b: OperandSpan,
    out: &mut [f32],
    seeded: bool,
    accumulator: f32,
) -> f32
where
    F: Fn(f32, f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    let width = out.len();
    let mut acc = accumulator;
    let mut start = 0usize;
    if !seeded && width > 0 {
        acc = op(a.at(0), b.at(0));
        out[0] = acc;
        start = 1;
    }
    if a.is_strided() || b.is_strided() {
        return scan_width_binary_monomorphic_strided(op, reduce, a, b, out, start, acc);
    }
    match (a.stride == 1, b.stride == 1) {
        (true, true) => {
            let slice_a = &a.data[a.base..a.base + width];
            let slice_b = &b.data[b.base..b.base + width];
            for ((slot, &value_a), &value_b) in out[start..].iter_mut().zip(&slice_a[start..]).zip(&slice_b[start..]) {
                acc = reduce(acc, op(value_a, value_b));
                *slot = acc;
            }
        }
        (true, false) => {
            let slice_a = &a.data[a.base..a.base + width];
            let value_b = b.data[b.base];
            for (slot, &value_a) in out[start..].iter_mut().zip(&slice_a[start..]) {
                acc = reduce(acc, op(value_a, value_b));
                *slot = acc;
            }
        }
        (false, true) => {
            let value_a = a.data[a.base];
            let slice_b = &b.data[b.base..b.base + width];
            for (slot, &value_b) in out[start..].iter_mut().zip(&slice_b[start..]) {
                acc = reduce(acc, op(value_a, value_b));
                *slot = acc;
            }
        }
        (false, false) => {
            let value_a = a.data[a.base];
            let value_b = b.data[b.base];
            for slot in out[start..].iter_mut() {
                acc = reduce(acc, op(value_a, value_b));
                *slot = acc;
            }
        }
    }
    acc
}

/// Continues [`scan_width_binary_monomorphic`]'s fold from `start` via
/// [`OperandSpan::at`], for the case at least one of `a`/`b` has a stride > 1
/// — same strict left-to-right combine order as every other arm here.
#[inline(always)]
fn scan_width_binary_monomorphic_strided<F, R>(
    op: F,
    reduce: R,
    a: OperandSpan,
    b: OperandSpan,
    out: &mut [f32],
    start: usize,
    accumulator: f32,
) -> f32
where
    F: Fn(f32, f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    let mut acc = accumulator;
    for (position, slot) in out.iter_mut().enumerate().skip(start) {
        acc = reduce(acc, op(a.at(position), b.at(position)));
        *slot = acc;
    }
    acc
}

fn scan_width_binary_scalar_dispatch(
    op: ScalarOp,
    reduce_op: ScalarOp,
    a: OperandSpan,
    b: OperandSpan,
    out: &mut [f32],
    seeded: bool,
    accumulator: f32,
) -> f32 {
    let width = out.len();
    let mut acc = accumulator;
    let mut seeded = seeded;
    for (index, slot) in out.iter_mut().enumerate().take(width) {
        let value = apply_scalar_op(op, &[a.at(index), b.at(index)]);
        acc = combine_reduction(reduce_op, acc, value, seeded);
        seeded = true;
        *slot = acc;
    }
    acc
}

/// Evaluates a (possibly fused) [`ComposedBody`] for one iteration step:
/// `operand_values[i]` is the freshly-read value of physical operand `i`,
/// `step_values` is scratch sized `body.steps.len()` the caller reuses
/// across every step of a run rather than allocating it per element — each
/// step's own value lands in `step_values[index]` as it is computed, so a
/// later step's `StepArg::Step` reference always reads an already-written
/// slot (steps only ever reference earlier steps).
///
/// Only reached through [`BodyShape::Generic`] now — [`eval_body_shape`]'s
/// `Unary`/`Binary` arms bypass this entirely for the common single-step
/// case, so this stays the slow-but-general path for real fused chains.
fn apply_body(body: &ComposedBody, operand_values: &[f32], step_values: &mut [f32]) -> f32 {
    for (index, step) in body.steps.iter().enumerate() {
        let mut args = [0.0f32; 3];
        for (slot, arg) in step.args.iter().enumerate() {
            args[slot] = match arg {
                StepArg::Operand(operand_index) => operand_values[*operand_index as usize],
                StepArg::Step(step_index) => step_values[*step_index as usize],
            };
        }
        step_values[index] = apply_scalar_op(step.op, &args[..step.args.len()]);
    }
    step_values[body.steps.len() - 1]
}

/// Abramowitz & Stegun 7.1.26: a single-branch rational approximation to
/// `erf`, entire in `core` float ops (no `libm` dependency — see this
/// module's own doc for why the crate does not carry one). Published maximum
/// absolute error is `1.5e-7`; measured here in `f32` against 14 reference
/// points (`erf_f32_matches_reference_values_within_f32_epsilon`), the
/// actual max error is `1.1920929e-7` — equal to `f32::EPSILON` itself
/// (`2^-23`), i.e. this approximation is precision-limited by `f32`'s own
/// representable step at these points, not by the formula.
#[inline(always)]
fn erf_f32(x: f32) -> f32 {
    const P: f32 = 0.327_591_1;
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_72;
    const A3: f32 = 1.421_413_8;
    const A4: f32 = -1.453_152_1;
    const A5: f32 = 1.061_405_4;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let magnitude = x.abs();
    let t = 1.0 / P.mul_add(magnitude, 1.0);
    let poly = t * A5.mul_add(t, A4).mul_add(t, A3).mul_add(t, A2).mul_add(t, A1);
    sign * poly.mul_add(-(-magnitude * magnitude).exp(), 1.0)
}

/// Same formula as [`erf_f32`], carried in `f64` for [`Element::apply`]'s
/// `f64` instantiation — not a wider-precision approximation, the same
/// published `1.5e-7` bound, just without f32's own rounding compounding on
/// top of it.
#[inline(always)]
fn erf_f64(x: f64) -> f64 {
    const P: f64 = 0.327_591_1;
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let magnitude = x.abs();
    let t = 1.0 / P.mul_add(magnitude, 1.0);
    let poly = t * A5.mul_add(t, A4).mul_add(t, A3).mul_add(t, A2).mul_add(t, A1);
    sign * poly.mul_add(-(-magnitude * magnitude).exp(), 1.0)
}

#[inline(always)]
fn apply_scalar_op(op: ScalarOp, operands: &[f32]) -> f32 {
    match op {
        ScalarOp::Identity => operands[0],
        ScalarOp::Add => operands[0] + operands[1],
        ScalarOp::Subtract => operands[0] - operands[1],
        ScalarOp::Multiply => operands[0] * operands[1],
        ScalarOp::Divide => operands[0] / operands[1],
        ScalarOp::Maximum => operands[0].max(operands[1]),
        ScalarOp::Minimum => operands[0].min(operands[1]),
        ScalarOp::Negate => -operands[0],
        ScalarOp::Reciprocal => 1.0 / operands[0],
        ScalarOp::Exponential => operands[0].exp(),
        ScalarOp::Logarithm => operands[0].ln(),
        ScalarOp::SquareRoot => operands[0].sqrt(),
        ScalarOp::Tanh => operands[0].tanh(),
        ScalarOp::Erf => erf_f32(operands[0]),
        ScalarOp::Greater => f32::from(u8::from(operands[0] > operands[1])),
        ScalarOp::Equal => f32::from(u8::from((operands[0] - operands[1]).abs() == 0.0)),
        ScalarOp::Select => {
            if operands[0] != 0.0 {
                operands[1]
            } else {
                operands[2]
            }
        }
    }
}

// ---------------------------------------------------------------------
// Typed elementwise evaluator: every dtype `reject_non_float32` used to
// reject outright, restricted to elementwise-only programs.
//
// `evaluate`/`evaluate_parallel` stay f32-only by construction (their
// buffers, width-tiling, and dot-fold kernels are `Vec<f32>` end to end);
// regeneralizing that pipeline to every width is the "generic element
// parameter threaded through every kernel" option this module's author
// considered and did not take, for the reason `dtype.rs`'s own doc already
// gives for keeping `DType` a runtime field: a single node can mix several
// dtypes (quantized matmul is `i8 x i8 -> i32`), which one `T` cannot
// describe, and threading a type parameter through every SIMD/dot-fold
// kernel here would monomorphize each one per width — compile time and
// code size scale with the ~13-way product, for kernels most callers never
// invoke at most of those widths.
//
// What follows instead is the other option: one runtime-dispatched
// [`TypedBuffer`] enum, matched once at the entry point
// ([`evaluate_typed`]) to pick a monomorphized [`Element`] instantiation —
// same shape `DType` itself already uses. Every match arm still hands the
// kernel one contiguous `&[T]`/`Vec<T>`, never a per-element tag or a boxed
// scalar, which is what lets a future NEON kernel specialize per width
// later without changing how a buffer is stored: a 128-bit register packs
// 16 `i8` lanes, 8 `i16`, 4 `i32`, or 2 `i64`; `i128`/`u128` have no NEON
// lane width and would run scalar-only even with a kernel written. No SIMD
// kernel is written here — every op below is a scalar loop — this only
// leaves the representation ready for one.
// ---------------------------------------------------------------------

/// A CPU-native scalar [`evaluate_typed`] can execute: every operand and
/// every output is one contiguous `[Self]`, matching what [`TypedBuffer`]
/// stores. `apply` is fallible, not merely a closed match: an integer dtype
/// genuinely cannot execute a transcendental (`exp`/`ln`/`sqrt`/`tanh`/
/// `reciprocal`), an unsigned dtype cannot negate, and integer division has
/// a real undefined case (zero divisor, or `T::MIN / -1`) — each is a named
/// [`TensorError`] at the node it was found, not a panic or a silently wrong
/// answer.
///
/// `'static` is what lets [`run_reduce_typed`]/[`run_scan_typed`] compare
/// `TypeId::of::<T>()` against `TypeId::of::<f32>()` and, on a match,
/// reinterpret this evaluator's `Vec<T>` buffers as the `Vec<f32>` the
/// existing NEON reduce/scan (`run_reduce`/`run_scan`) already take — the
/// specialization that keeps the fast path a single implementation instead
/// of a second copy of the reduction nest.
///
/// Not a pipe, and not converted to one. `DTYPE` is a type-level fact, not
/// a transformation -- it is read only as an ordinary runtime struct field
/// (`dtype: Self::DTYPE` at the two `UnsupportedScalarOp` sites above),
/// never in a const item, array length, match pattern, or `const fn` body
/// anywhere in this workspace, so today the "a `Pipe` impl can't be const"
/// objection is theoretical, not load-bearing. The real reason `Element`
/// stays a trait: `unwrap_block`/`apply`/`reduce_seed`/`from_index` are a
/// per-type dispatch table the 11 `T: Element`-bound functions below
/// (`run_typed_program` through `run_scan_generic`) select at monomorphize
/// time, not a stream of values flowing through combinators -- each is
/// called once per site with its arguments already in hand, nothing is
/// composed. Splitting `apply` out as a pipe would still need a trait
/// bound naming that pipe per `T`, i.e. the same trait under a new name.
trait Element: Copy + Default + 'static {
    const DTYPE: DType;

    fn unwrap_block(buffer: &TypedBuffer) -> Option<&[Self]>;
    fn apply(node: NodeId, op: ScalarOp, args: &[Self]) -> Result<Self, TensorError>;

    /// The reduce seed for `init`, in this element's own type — the typed
    /// counterpart of [`initial_value`]. `None` for [`ReduceInit::FirstElement`],
    /// same as [`initial_value`]: there is no synthetic identity, the first
    /// element visited seeds the accumulator instead.
    fn reduce_seed(init: ReduceInit) -> Option<Self>;

    /// [`BoundOpKind::Iota`]'s output value at position `index`, in this
    /// element's own type — the typed counterpart of [`run_iota`]'s
    /// `index as f32`.
    fn from_index(index: usize) -> Self;

    /// [`BoundOpKind::Constant`]'s literal in this element's own type — the
    /// typed counterpart of [`run_constant`]'s bare `f32`. An integer
    /// element truncates toward zero, the same `as` conversion
    /// [`Element::from_index`] uses in the other direction.
    fn from_literal(value: f32) -> Self;
}

macro_rules! impl_element_signed_integer {
    ($ty:ty, $dtype:expr, $variant:ident) => {
        impl Element for $ty {
            const DTYPE: DType = $dtype;

            fn unwrap_block(buffer: &TypedBuffer) -> Option<&[Self]> {
                match buffer {
                    TypedBuffer::$variant(data) => Some(data.as_slice()),
                    _ => None,
                }
            }

            fn apply(node: NodeId, op: ScalarOp, args: &[Self]) -> Result<Self, TensorError> {
                Ok(match op {
                    ScalarOp::Identity => args[0],
                    ScalarOp::Add => args[0].wrapping_add(args[1]),
                    ScalarOp::Subtract => args[0].wrapping_sub(args[1]),
                    ScalarOp::Multiply => args[0].wrapping_mul(args[1]),
                    ScalarOp::Divide => {
                        return args[0]
                            .checked_div(args[1])
                            .ok_or(TensorError::CheckedDivisionFailed { node });
                    }
                    ScalarOp::Maximum => args[0].max(args[1]),
                    ScalarOp::Minimum => args[0].min(args[1]),
                    ScalarOp::Negate => args[0].wrapping_neg(),
                    ScalarOp::Greater => Self::from(args[0] > args[1]),
                    ScalarOp::Equal => Self::from(args[0] == args[1]),
                    ScalarOp::Select => {
                        if args[0] != 0 {
                            args[1]
                        } else {
                            args[2]
                        }
                    }
                    ScalarOp::Reciprocal
                    | ScalarOp::Exponential
                    | ScalarOp::Logarithm
                    | ScalarOp::SquareRoot
                    | ScalarOp::Tanh
                    | ScalarOp::Erf => {
                        return Err(TensorError::UnsupportedScalarOp {
                            node,
                            op,
                            dtype: Self::DTYPE,
                        });
                    }
                })
            }

            fn reduce_seed(init: ReduceInit) -> Option<Self> {
                match init {
                    ReduceInit::Zero => Some(0),
                    ReduceInit::One => Some(1),
                    ReduceInit::NegativeInfinity => Some(Self::MIN),
                    ReduceInit::PositiveInfinity => Some(Self::MAX),
                    ReduceInit::FirstElement => None,
                }
            }

            fn from_index(index: usize) -> Self {
                index as $ty
            }

            fn from_literal(value: f32) -> Self {
                value as $ty
            }
        }
    };
}

macro_rules! impl_element_unsigned_integer {
    ($ty:ty, $dtype:expr, $variant:ident) => {
        impl Element for $ty {
            const DTYPE: DType = $dtype;

            fn unwrap_block(buffer: &TypedBuffer) -> Option<&[Self]> {
                match buffer {
                    TypedBuffer::$variant(data) => Some(data.as_slice()),
                    _ => None,
                }
            }

            fn apply(node: NodeId, op: ScalarOp, args: &[Self]) -> Result<Self, TensorError> {
                Ok(match op {
                    ScalarOp::Identity => args[0],
                    ScalarOp::Add => args[0].wrapping_add(args[1]),
                    ScalarOp::Subtract => args[0].wrapping_sub(args[1]),
                    ScalarOp::Multiply => args[0].wrapping_mul(args[1]),
                    ScalarOp::Divide => {
                        return args[0]
                            .checked_div(args[1])
                            .ok_or(TensorError::CheckedDivisionFailed { node });
                    }
                    ScalarOp::Maximum => args[0].max(args[1]),
                    ScalarOp::Minimum => args[0].min(args[1]),
                    ScalarOp::Greater => Self::from(args[0] > args[1]),
                    ScalarOp::Equal => Self::from(args[0] == args[1]),
                    ScalarOp::Select => {
                        if args[0] != 0 {
                            args[1]
                        } else {
                            args[2]
                        }
                    }
                    ScalarOp::Negate
                    | ScalarOp::Reciprocal
                    | ScalarOp::Exponential
                    | ScalarOp::Logarithm
                    | ScalarOp::SquareRoot
                    | ScalarOp::Tanh
                    | ScalarOp::Erf => {
                        return Err(TensorError::UnsupportedScalarOp {
                            node,
                            op,
                            dtype: Self::DTYPE,
                        });
                    }
                })
            }

            fn reduce_seed(init: ReduceInit) -> Option<Self> {
                match init {
                    ReduceInit::Zero | ReduceInit::NegativeInfinity => Some(0),
                    ReduceInit::One => Some(1),
                    ReduceInit::PositiveInfinity => Some(Self::MAX),
                    ReduceInit::FirstElement => None,
                }
            }

            fn from_index(index: usize) -> Self {
                index as $ty
            }

            fn from_literal(value: f32) -> Self {
                value as $ty
            }
        }
    };
}

impl_element_signed_integer!(i8, DType::Int8, Int8);
impl_element_signed_integer!(i16, DType::Int16, Int16);
impl_element_signed_integer!(i32, DType::Int32, Int32);
impl_element_signed_integer!(i64, DType::Int64, Int64);
impl_element_signed_integer!(i128, DType::Int128, Int128);

impl_element_unsigned_integer!(u8, DType::UInt8, UInt8);
impl_element_unsigned_integer!(u16, DType::UInt16, UInt16);
impl_element_unsigned_integer!(u32, DType::UInt32, UInt32);
impl_element_unsigned_integer!(u64, DType::UInt64, UInt64);
impl_element_unsigned_integer!(u128, DType::UInt128, UInt128);

impl Element for f32 {
    const DTYPE: DType = DType::Float32;

    fn unwrap_block(buffer: &TypedBuffer) -> Option<&[Self]> {
        match buffer {
            TypedBuffer::Float32(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    fn apply(_node: NodeId, op: ScalarOp, args: &[Self]) -> Result<Self, TensorError> {
        Ok(apply_scalar_op(op, args))
    }

    fn reduce_seed(init: ReduceInit) -> Option<Self> {
        initial_value(init)
    }

    fn from_index(index: usize) -> Self {
        index as Self
    }

    fn from_literal(value: f32) -> Self {
        value as Self
    }
}

impl Element for f64 {
    const DTYPE: DType = DType::Float64;

    fn unwrap_block(buffer: &TypedBuffer) -> Option<&[Self]> {
        match buffer {
            TypedBuffer::Float64(data) => Some(data.as_slice()),
            _ => None,
        }
    }

    fn apply(_node: NodeId, op: ScalarOp, args: &[Self]) -> Result<Self, TensorError> {
        Ok(match op {
            ScalarOp::Identity => args[0],
            ScalarOp::Add => args[0] + args[1],
            ScalarOp::Subtract => args[0] - args[1],
            ScalarOp::Multiply => args[0] * args[1],
            ScalarOp::Divide => args[0] / args[1],
            ScalarOp::Maximum => args[0].max(args[1]),
            ScalarOp::Minimum => args[0].min(args[1]),
            ScalarOp::Negate => -args[0],
            ScalarOp::Reciprocal => 1.0 / args[0],
            ScalarOp::Exponential => args[0].exp(),
            ScalarOp::Logarithm => args[0].ln(),
            ScalarOp::SquareRoot => args[0].sqrt(),
            ScalarOp::Tanh => args[0].tanh(),
            ScalarOp::Erf => erf_f64(args[0]),
            ScalarOp::Greater => f64::from(u8::from(args[0] > args[1])),
            ScalarOp::Equal => f64::from(u8::from((args[0] - args[1]).abs() == 0.0)),
            ScalarOp::Select => {
                if args[0] != 0.0 {
                    args[1]
                } else {
                    args[2]
                }
            }
        })
    }

    fn reduce_seed(init: ReduceInit) -> Option<Self> {
        match init {
            ReduceInit::Zero => Some(0.0),
            ReduceInit::One => Some(1.0),
            ReduceInit::NegativeInfinity => Some(f64::NEG_INFINITY),
            ReduceInit::PositiveInfinity => Some(f64::INFINITY),
            ReduceInit::FirstElement => None,
        }
    }

    fn from_index(index: usize) -> Self {
        index as Self
    }

    fn from_literal(value: f32) -> Self {
        value as Self
    }
}

/// Shared body for [`f16`] and [`bf16`]'s [`Element`] impl: neither type has
/// stable-Rust arithmetic operators (`convert.rs`'s own doc), so every op
/// round-trips through `f32` — widen both operands, run the existing f32
/// scalar table ([`apply_scalar_op`]), narrow the result back. This is a
/// real semantic (one rounding step per op, not the fused half-precision
/// arithmetic a hardware FPU would give), documented here rather than
/// silently assumed by a caller.
macro_rules! impl_element_half_float {
    ($ty:ty, $dtype:expr, $variant:ident) => {
        impl Element for $ty {
            const DTYPE: DType = $dtype;

            fn unwrap_block(buffer: &TypedBuffer) -> Option<&[Self]> {
                match buffer {
                    TypedBuffer::$variant(data) => Some(data.as_slice()),
                    _ => None,
                }
            }

            fn apply(_node: NodeId, op: ScalarOp, args: &[Self]) -> Result<Self, TensorError> {
                let mut widened = [0.0f32; 3];
                for (slot, value) in widened.iter_mut().zip(args) {
                    *slot = value.to_f32();
                }
                let result = apply_scalar_op(op, &widened[..args.len()]);
                Ok(Self::from_f32(result))
            }

            fn reduce_seed(init: ReduceInit) -> Option<Self> {
                initial_value(init).map(Self::from_f32)
            }

            fn from_index(index: usize) -> Self {
                Self::from_f32(index as f32)
            }

            fn from_literal(value: f32) -> Self {
                Self::from_f32(value)
            }
        }
    };
}

impl_element_half_float!(f16, DType::Float16, Float16);
impl_element_half_float!(bf16, DType::BFloat16, BFloat16);

/// One contiguous typed buffer, tagged by which native type backs it — the
/// storage half of [`evaluate_typed`]'s runtime dispatch. Every variant is a
/// plain `Vec<T>`: a whole buffer is tagged, never a scalar, which is what
/// keeps every operand a contiguous, SIMD-ready slice once a kernel is
/// written for it (see this module's typed-evaluator doc). `Bool` has no
/// variant yet — its storage convention (packed bits vs. one byte per
/// element) is undecided; see `typed_program_plan` for the boundary this
/// actually enforces today. `Float16`/`BFloat16` route every arithmetic op
/// through an `f32` round-trip (`Element`'s half-float impl, above) since
/// neither has stable-Rust arithmetic operators of its own.
#[derive(Debug, Clone, PartialEq)]
pub enum TypedBuffer {
    Int8(Vec<i8>),
    UInt8(Vec<u8>),
    Int16(Vec<i16>),
    UInt16(Vec<u16>),
    Int32(Vec<i32>),
    UInt32(Vec<u32>),
    Int64(Vec<i64>),
    UInt64(Vec<u64>),
    Int128(Vec<i128>),
    UInt128(Vec<u128>),
    Float16(Vec<f16>),
    BFloat16(Vec<bf16>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
}

impl TypedBuffer {
    #[must_use]
    pub const fn dtype(&self) -> DType {
        match self {
            Self::Int8(_) => DType::Int8,
            Self::UInt8(_) => DType::UInt8,
            Self::Int16(_) => DType::Int16,
            Self::UInt16(_) => DType::UInt16,
            Self::Int32(_) => DType::Int32,
            Self::UInt32(_) => DType::UInt32,
            Self::Int64(_) => DType::Int64,
            Self::UInt64(_) => DType::UInt64,
            Self::Int128(_) => DType::Int128,
            Self::UInt128(_) => DType::UInt128,
            Self::Float16(_) => DType::Float16,
            Self::BFloat16(_) => DType::BFloat16,
            Self::Float32(_) => DType::Float32,
            Self::Float64(_) => DType::Float64,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Int8(data) => data.len(),
            Self::UInt8(data) => data.len(),
            Self::Int16(data) => data.len(),
            Self::UInt16(data) => data.len(),
            Self::Int32(data) => data.len(),
            Self::UInt32(data) => data.len(),
            Self::Int64(data) => data.len(),
            Self::UInt64(data) => data.len(),
            Self::Int128(data) => data.len(),
            Self::UInt128(data) => data.len(),
            Self::Float16(data) => data.len(),
            Self::BFloat16(data) => data.len(),
            Self::Float32(data) => data.len(),
            Self::Float64(data) => data.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// [`typed_program_plan`]'s answer: either every node in the program shares
/// one dtype (the only shape this evaluator supported before mixed
/// precision), or exactly one dtype change occurs, and only at a
/// [`Op::Reduce`] node's own accumulator — the quantized-accumulate shape
/// (`i8` operand folded into an `i32` accumulator) that a single uniform
/// dtype cannot express. See [`typed_program_plan`]'s own doc for the
/// structural check that produces this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TypedPlan {
    Uniform(DType),
    Widened { operand: DType, accumulator: DType },
}

/// Validates a program is executable by [`evaluate_typed`] and returns its
/// [`TypedPlan`].
///
/// A THIRD role sits alongside the operand/accumulator pair below: a
/// gather's `indices` node ([`index_node_ids`], the same structural
/// detection [`reject_non_float32`]'s f32 pipeline already uses to exempt
/// gather indices from its own uniform-dtype rule). An index node is
/// exempt from the uniform/widened dtype check entirely — it may carry any
/// [`DType::is_integer`] dtype regardless of what the rest of the program
/// runs at — but a non-integer index dtype (a float, or `Bool`) is an
/// honest `NotLowerable`, never silently coerced. [`run_typed_program`] and
/// [`run_widened_program`] execute this role for real: [`canonical_index_buffers`]
/// widens every index node's caller-supplied buffer into one canonical
/// `i64` table once, up front, and [`fill_gather_cursors_typed`] reads a
/// gathered operand's fetched index from that table instead of the
/// compute-dtype operand table — the plan only had to stop rejecting the
/// shape at the door once that table existed to back it.
///
/// Two shapes pass beyond that, everything else is
/// [`TensorError::NotLowerable`]:
///
/// - **uniform** — every non-index node shares one dtype. This is the
///   whole-program restriction this function always enforced; it is
///   unchanged for any program that never mixes dtypes, which is what keeps
///   the existing f32 NEON fast path (`run_reduce_typed`/`run_scan_typed`'s
///   `T = f32` specialization) reachable exactly as before.
/// - **widened** — every non-index node up to some position shares one
///   dtype (`operand`), the node at that position is an [`Op::Reduce`]
///   whose own dtype differs (`accumulator`), and every node from there on
///   shares `accumulator`. A `Reduce`'s `operand: NodeId` field only ever
///   points backwards (this crate's own SSA invariant — see [`Op::append`]'s
///   doc), so the dtype that changed at that node is provably the fold's
///   operand dtype widening into its accumulator, not an unrelated node
///   happening to differ. [`evaluate_typed`] dispatches this shape to
///   [`run_widened_program`], scoped to the pairs it ships a [`Convert`]
///   [`Pipe`] for — see that function's own doc.
///
/// Any dtype change outside those shapes (a third distinct non-index dtype,
/// or a change at a non-`Reduce` node) is rejected with an honest
/// `NotLowerable` rather than silently picked apart. `Bool` is out at any
/// non-index position — see [`TypedBuffer`]'s doc; `BFloat16`/`Float16` are
/// typed elements like any other (see `Element`'s half-float impl).
fn typed_program_plan(program: &[Op]) -> Result<TypedPlan, TensorError> {
    let index_nodes = index_node_ids(program);
    let base_dtype = program
        .iter()
        .enumerate()
        .find(|(position, _)| !index_nodes.contains(&NodeId(*position as u32)))
        .map(|(_, expr)| expr.dtype())
        .ok_or(TensorError::Empty)?;
    let mut widen_at: Option<(usize, DType)> = None;
    for (position, expr) in program.iter().enumerate() {
        let node = NodeId(position as u32);
        let dtype = expr.dtype();
        if index_nodes.contains(&node) {
            if !dtype.is_integer() {
                return Err(TensorError::NotLowerable {
                    node,
                    reason: "a gather index node must carry an integer dtype",
                });
            }
            continue;
        }
        if dtype == DType::Bool {
            return Err(TensorError::NotLowerable {
                node,
                reason: "the typed evaluator does not support Bool yet",
            });
        }
        if dtype == base_dtype {
            continue;
        }
        match widen_at {
            None => widen_at = Some((position, dtype)),
            Some((_, accumulator)) if accumulator == dtype => {}
            Some(_) => {
                return Err(TensorError::NotLowerable {
                    node,
                    reason: "the typed evaluator supports at most one dtype change per program",
                });
            }
        }
    }
    match widen_at {
        None => Ok(TypedPlan::Uniform(base_dtype)),
        Some((position, accumulator)) => {
            if !matches!(program[position], Op::Reduce(_)) {
                return Err(TensorError::NotLowerable {
                    node: NodeId(position as u32),
                    reason: "a dtype change may only occur at a Reduce node's own accumulator",
                });
            }
            Ok(TypedPlan::Widened {
                operand: base_dtype,
                accumulator,
            })
        }
    }
}

/// One requested output's node, shape, and data — [`evaluate_typed`]'s
/// per-dtype row, and [`run_typed_program`]'s own before it is wrapped into
/// a [`TypedBuffer`].
type TypedRow<Data> = (NodeId, Vec<u64>, Data);

/// Run an elementwise-or-reduce tensor program against a caller-chosen
/// non-f32 (or f64) dtype — the full-width counterpart of [`evaluate`] for
/// the programs `reject_non_float32` used to reject outright. See
/// `typed_program_plan` for exactly which programs qualify, and for the
/// uniform case, `DType::Float32` dispatches `run_typed_program` the same as
/// every other width, but that function's own [`Op::Reduce`] handling
/// specializes straight back to the existing NEON `run_reduce`/`run_scan`
/// for `T = f32` — see `run_reduce_typed`'s doc. A `TypedPlan::Widened`
/// program dispatches to `run_widened_program` instead, over the
/// `(operand, accumulator)` pairs that section ships a [`Convert`] for.
pub fn evaluate_typed(
    program: &[Op],
    symbols: &[u64],
    blocks: &[TypedBuffer],
    outputs: &[NodeId],
) -> Result<Vec<TypedRow<TypedBuffer>>, TensorError> {
    match typed_program_plan(program)? {
        TypedPlan::Uniform(dtype) => evaluate_uniform_typed(dtype, program, symbols, blocks, outputs),
        TypedPlan::Widened { operand, accumulator } => {
            evaluate_widened_typed(operand, accumulator, program, symbols, blocks, outputs)
        }
    }
}

/// [`evaluate_typed`]'s [`TypedPlan::Uniform`] arm: the whole-program,
/// single-dtype dispatch this evaluator always had, unmodified.
fn evaluate_uniform_typed(
    dtype: DType,
    program: &[Op],
    symbols: &[u64],
    blocks: &[TypedBuffer],
    outputs: &[NodeId],
) -> Result<Vec<TypedRow<TypedBuffer>>, TensorError> {
    macro_rules! dispatch {
        ($ty:ty, $variant:ident) => {{
            run_typed_program::<$ty>(program, symbols, blocks, outputs)?
                .into_iter()
                .map(|(node, shape, data)| (node, shape, TypedBuffer::$variant(data)))
                .collect()
        }};
    }
    Ok(match dtype {
        DType::Int8 => dispatch!(i8, Int8),
        DType::UInt8 => dispatch!(u8, UInt8),
        DType::Int16 => dispatch!(i16, Int16),
        DType::UInt16 => dispatch!(u16, UInt16),
        DType::Int32 => dispatch!(i32, Int32),
        DType::UInt32 => dispatch!(u32, UInt32),
        DType::Int64 => dispatch!(i64, Int64),
        DType::UInt64 => dispatch!(u64, UInt64),
        DType::Int128 => dispatch!(i128, Int128),
        DType::UInt128 => dispatch!(u128, UInt128),
        DType::Float16 => dispatch!(f16, Float16),
        DType::BFloat16 => dispatch!(bf16, BFloat16),
        DType::Float32 => dispatch!(f32, Float32),
        DType::Float64 => dispatch!(f64, Float64),
        DType::Bool => unreachable!("typed_program_plan already rejected this dtype"),
    })
}

/// [`evaluate_typed`]'s [`TypedPlan::Widened`] arm — the `(operand,
/// accumulator)` dispatch table. Scoped to the pairs actually shipped, not
/// the full `DType x DType` cross product: `(Int8, Int32)` (the
/// quantized-accumulate case `typed_program_plan`'s doc names), `(Int16,
/// Int64)` and `(UInt8, UInt32)` (the same accumulation-overflow shape at
/// other integer widths), and `(Float16, Float32)`/`(BFloat16, Float32)`
/// (a half-precision reduce folded into an f32 accumulator, the same
/// widen-before-fold shape at floating-point widths). Any other pair is an
/// honest [`TensorError::NotLowerable`] — never a silent wrong result from
/// picking the nearer-available width.
fn evaluate_widened_typed(
    operand: DType,
    accumulator: DType,
    program: &[Op],
    symbols: &[u64],
    blocks: &[TypedBuffer],
    outputs: &[NodeId],
) -> Result<Vec<TypedRow<TypedBuffer>>, TensorError> {
    match (operand, accumulator) {
        (DType::Int8, DType::Int32) => Ok(run_widened_program::<i8, i32>(program, symbols, blocks, outputs)?
            .into_iter()
            .map(|(node, shape, data)| (node, shape, TypedBuffer::Int32(data)))
            .collect()),
        (DType::Int16, DType::Int64) => Ok(run_widened_program::<i16, i64>(program, symbols, blocks, outputs)?
            .into_iter()
            .map(|(node, shape, data)| (node, shape, TypedBuffer::Int64(data)))
            .collect()),
        (DType::UInt8, DType::UInt32) => Ok(run_widened_program::<u8, u32>(program, symbols, blocks, outputs)?
            .into_iter()
            .map(|(node, shape, data)| (node, shape, TypedBuffer::UInt32(data)))
            .collect()),
        (DType::Float16, DType::Float32) => {
            Ok(run_widened_program::<f16, f32>(program, symbols, blocks, outputs)?
                .into_iter()
                .map(|(node, shape, data)| (node, shape, TypedBuffer::Float32(data)))
                .collect())
        }
        (DType::BFloat16, DType::Float32) => {
            Ok(run_widened_program::<bf16, f32>(program, symbols, blocks, outputs)?
                .into_iter()
                .map(|(node, shape, data)| (node, shape, TypedBuffer::Float32(data)))
                .collect())
        }
        _ => Err(TensorError::NotLowerable {
            node: NodeId(0),
            reason: "the typed evaluator does not ship a mixed-precision reduce pair for this \
                     operand/accumulator combination",
        }),
    }
}

/// Widens one caller-supplied index block into [`GatherCursor`]'s canonical
/// `i64` width, matching the same lossy `raw as i64` truncation
/// [`GatherCursor::fetch_and_advance`]'s f32 sibling already performs at
/// every element read (`i128`/`u128`/`u64` values past `i64::MAX` truncate
/// exactly as they would there) — paid once per index buffer here instead
/// of once per gathered element. A non-integer `TypedBuffer` variant is an
/// honest `NotLowerable`: [`typed_program_plan`] already rejected a
/// non-integer index node's *declared* dtype, this rejects the buffer the
/// caller actually handed over disagreeing with that at the same gate.
fn typed_buffer_to_index(node: NodeId, buffer: &TypedBuffer) -> Result<Vec<i64>, TensorError> {
    match buffer {
        TypedBuffer::Int8(data) => Ok(data.iter().map(|&value| i64::from(value)).collect()),
        TypedBuffer::UInt8(data) => Ok(data.iter().map(|&value| i64::from(value)).collect()),
        TypedBuffer::Int16(data) => Ok(data.iter().map(|&value| i64::from(value)).collect()),
        TypedBuffer::UInt16(data) => Ok(data.iter().map(|&value| i64::from(value)).collect()),
        TypedBuffer::Int32(data) => Ok(data.iter().map(|&value| i64::from(value)).collect()),
        TypedBuffer::UInt32(data) => Ok(data.iter().map(|&value| i64::from(value)).collect()),
        TypedBuffer::Int64(data) => Ok(data.clone()),
        TypedBuffer::UInt64(data) => Ok(data.iter().map(|&value| value as i64).collect()),
        TypedBuffer::Int128(data) => Ok(data.iter().map(|&value| value as i64).collect()),
        TypedBuffer::UInt128(data) => Ok(data.iter().map(|&value| value as i64).collect()),
        TypedBuffer::Float16(_) | TypedBuffer::BFloat16(_) | TypedBuffer::Float32(_) | TypedBuffer::Float64(_) => {
            Err(TensorError::NotLowerable {
                node,
                reason: "a gather index buffer must carry an integer dtype",
            })
        }
    }
}

/// Builds [`fill_gather_cursors_typed`]'s canonical `i64` index-buffer
/// table, one entry per [`index_node_ids`] member, from the same
/// caller-supplied `blocks` [`run_typed_program`]/[`run_widened_program`]
/// bind their own compute-dtype operand table from — every index node this
/// evaluator supports is a caller-supplied [`Op::Input`] leaf, the shape
/// [`crate::spec`]'s `Gather` table construction and every differential
/// gather test in this module actually produce. A computed index node
/// (derived through `Elementwise`/`Reduce`/`Iota`/`Constant` instead of
/// supplied as a block) is an honest `NotLowerable` rather than a second,
/// unexercised execution nest guessed at without a test to prove it right —
/// see [`TensorError::NotLowerable`]'s own doc on preferring a named gap
/// over a silently wrong result.
fn canonical_index_buffers(
    program: &[Op],
    shapes: &shape::Shapes,
    block_nodes: &[NodeId],
    blocks: &[TypedBuffer],
) -> Result<Vec<Option<Vec<i64>>>, TensorError> {
    let index_nodes = index_node_ids(program);
    let mut index_buffers: Vec<Option<Vec<i64>>> = vec![None; program.len()];
    for (node, buffer) in block_nodes.iter().zip(blocks.iter()) {
        if !index_nodes.contains(node) {
            continue;
        }
        let data = typed_buffer_to_index(*node, buffer)?;
        let expected = element_count(shapes.of(*node));
        if data.len() != expected {
            return Err(TensorError::InputSizeMismatch {
                node: *node,
                expected,
                found: data.len(),
            });
        }
        index_buffers[node.0 as usize] = Some(data);
    }
    for node in &index_nodes {
        if index_buffers[node.0 as usize].is_none() {
            return Err(TensorError::NotLowerable {
                node: *node,
                reason: "a gather index node must be a caller-supplied input; a computed index \
                         node is not supported yet",
            });
        }
    }
    Ok(index_buffers)
}

/// The monomorphic body [`evaluate_typed`] dispatches into per dtype: shape
/// inference, block binding, and a scalar-or-`T=f32`-specialized walk of
/// every resolved node — the same three stages [`prepare`]/[`evaluate_pooled`]
/// run for f32, minus chunk splitting (this evaluator does not parallelize
/// yet).
///
/// Buffer handling mirrors [`prepare`]/[`evaluate_pooled`] rather than
/// forking it: an input block is held as `Cow::Borrowed` (no per-call copy
/// of the caller's data — the `data.to_vec()` this replaced copied every
/// input block on every call regardless of whether the program even used
/// it), a computed node's output comes from [`typed_take_or_allocate`]
/// (reusing a retired buffer's storage instead of a fresh `vec![..]` per
/// node), and [`node_retirement`] — already generic over the buffer type,
/// unmodified here — decides when a buffer is done being read and goes back
/// to the pool via [`typed_retire_into`].
fn run_typed_program<T: Element>(
    program: &[Op],
    symbols: &[u64],
    blocks: &[TypedBuffer],
    outputs: &[NodeId],
) -> Result<Vec<TypedRow<Vec<T>>>, TensorError> {
    let shapes = shape::infer(program, symbols)?;

    let root = program
        .len()
        .checked_sub(1)
        .map(|last| NodeId(last as u32))
        .ok_or(TensorError::Empty)?;
    for output in outputs {
        if output.0 as usize >= program.len() {
            return Err(TensorError::UnknownOutput(*output));
        }
    }
    let effective_outputs: Vec<NodeId> = if outputs.is_empty() {
        vec![root]
    } else {
        outputs.to_vec()
    };

    let block_nodes = block_node_ids(program);
    if blocks.len() != block_nodes.len() {
        return Err(TensorError::InputCountMismatch {
            expected: block_nodes.len(),
            found: blocks.len(),
        });
    }

    let index_nodes = index_node_ids(program);
    let index_buffers = canonical_index_buffers(program, &shapes, &block_nodes, blocks)?;

    let mut buffers: Vec<Option<Cow<'_, [T]>>> = vec![None; program.len()];
    for (node, buffer) in block_nodes.iter().zip(blocks.iter()) {
        if index_nodes.contains(node) {
            continue;
        }
        let data = T::unwrap_block(buffer).ok_or(TensorError::NotLowerable {
            node: *node,
            reason: "typed evaluator input dtype does not match the program's uniform dtype",
        })?;
        let expected = element_count(shapes.of(*node));
        if data.len() != expected {
            return Err(TensorError::InputSizeMismatch {
                node: *node,
                expected,
                found: data.len(),
            });
        }
        buffers[node.0 as usize] = Some(Cow::Borrowed(data));
    }

    let resolved = bind::bind(program, &shapes, &effective_outputs)?;

    let retires = node_retirement(&resolved, &effective_outputs);
    let mut free_buffers: Vec<Vec<T>> = Vec::new();
    for (position, node) in resolved.iter().enumerate() {
        let mut output = typed_take_or_allocate(&mut free_buffers, node_output_len(node));
        match &node.kind {
            BoundOpKind::Elementwise { .. } => run_elementwise_typed(node, &buffers, &index_buffers, &mut output)?,
            BoundOpKind::Reduce { keep: Keep::Reduce, .. } => {
                run_reduce_typed(node, &buffers, &index_buffers, &mut output)?;
            }
            BoundOpKind::Reduce { keep: Keep::Scan, .. } => {
                run_scan_typed(node, &buffers, &index_buffers, &mut output)?;
            }
            BoundOpKind::Iota => run_iota_typed(&mut output),
            BoundOpKind::Constant { value } => run_constant_typed(*value, &mut output),
        }
        buffers[node.node.0 as usize] = Some(Cow::Owned(output));
        for retired in &retires[position] {
            typed_retire_into(&mut buffers, *retired, &mut free_buffers);
        }
    }

    Ok(effective_outputs
        .iter()
        .map(|node| {
            let shape = shapes.of(*node).to_vec();
            let data = buffers[node.0 as usize]
                .clone()
                .map(Cow::into_owned)
                .unwrap_or_default();
            (*node, shape, data)
        })
        .collect())
}

/// [`run_typed_program`]'s two-dtype sibling: every node up to a
/// [`Op::Reduce`] runs in `TIn` (the operand width), the `Reduce` node
/// itself and everything downstream of it runs in `TAcc` (the accumulator
/// width) — [`typed_program_plan`]'s `Widened` shape, executed. This is the
/// case a single generic parameter structurally could not express: an `i8`
/// operand folded into an `i32` accumulator needs the accumulator to
/// actually be `i32`-wide in memory, not `i8` wrapped on overflow.
///
/// Two buffer tables instead of one (`buffers_in: [TIn]`, `buffers_out:
/// [TAcc]`), each indexed by [`NodeId`] exactly like [`run_typed_program`]'s
/// single table. A node's own dtype (`BoundOp::dtype`, mirroring
/// [`Op::dtype`]) decides which table it writes into. The one new step is
/// the crossing itself: immediately before running the `Reduce` node (or any
/// node reading a `TIn`-dtype operand from `TAcc` context), that operand's
/// buffer is widened once, elementwise, through [`Convert`]`<TIn,
/// TAcc>`::`call` (the same [`Pipe`] [`crate::convert`] ships for every
/// other conversion in this crate — no bespoke fold, the algebra already had
/// this piece) and the converted copy is stashed into `buffers_out` at that
/// same node id, so every downstream reader (including the `Reduce` itself)
/// sees an ordinary same-type `TAcc` operand from then on.
fn run_widened_program<TIn, TAcc>(
    program: &[Op],
    symbols: &[u64],
    blocks: &[TypedBuffer],
    outputs: &[NodeId],
) -> Result<Vec<TypedRow<Vec<TAcc>>>, TensorError>
where
    TIn: Element,
    TAcc: Element,
    Convert<TIn, TAcc>: Pipe<In = TIn, Out = TAcc, Err = core::convert::Infallible>,
{
    let shapes = shape::infer(program, symbols)?;

    let root = program
        .len()
        .checked_sub(1)
        .map(|last| NodeId(last as u32))
        .ok_or(TensorError::Empty)?;
    for output in outputs {
        if output.0 as usize >= program.len() {
            return Err(TensorError::UnknownOutput(*output));
        }
    }
    let effective_outputs: Vec<NodeId> = if outputs.is_empty() {
        vec![root]
    } else {
        outputs.to_vec()
    };

    let block_nodes = block_node_ids(program);
    if blocks.len() != block_nodes.len() {
        return Err(TensorError::InputCountMismatch {
            expected: block_nodes.len(),
            found: blocks.len(),
        });
    }

    let index_nodes = index_node_ids(program);
    let index_buffers = canonical_index_buffers(program, &shapes, &block_nodes, blocks)?;

    let mut buffers_in: Vec<Option<Cow<'_, [TIn]>>> = vec![None; program.len()];
    let mut buffers_out: Vec<Option<Cow<'_, [TAcc]>>> = vec![None; program.len()];
    for (node, buffer) in block_nodes.iter().zip(blocks.iter()) {
        if index_nodes.contains(node) {
            continue;
        }
        let expected = element_count(shapes.of(*node));
        if program[node.0 as usize].dtype() == TIn::DTYPE {
            let data = TIn::unwrap_block(buffer).ok_or(TensorError::NotLowerable {
                node: *node,
                reason: "typed evaluator input dtype does not match its node's own dtype",
            })?;
            if data.len() != expected {
                return Err(TensorError::InputSizeMismatch {
                    node: *node,
                    expected,
                    found: data.len(),
                });
            }
            buffers_in[node.0 as usize] = Some(Cow::Borrowed(data));
        } else {
            let data = TAcc::unwrap_block(buffer).ok_or(TensorError::NotLowerable {
                node: *node,
                reason: "typed evaluator input dtype does not match its node's own dtype",
            })?;
            if data.len() != expected {
                return Err(TensorError::InputSizeMismatch {
                    node: *node,
                    expected,
                    found: data.len(),
                });
            }
            buffers_out[node.0 as usize] = Some(Cow::Borrowed(data));
        }
    }

    let resolved = bind::bind(program, &shapes, &effective_outputs)?;

    let retires = node_retirement(&resolved, &effective_outputs);
    let mut free_in: Vec<Vec<TIn>> = Vec::new();
    let mut free_out: Vec<Vec<TAcc>> = Vec::new();
    let converter = Convert::<TIn, TAcc>::new();

    for (position, node) in resolved.iter().enumerate() {
        if node.dtype == TIn::DTYPE {
            let mut output = typed_take_or_allocate(&mut free_in, node_output_len(node));
            match &node.kind {
                BoundOpKind::Elementwise { .. } => {
                    run_elementwise_typed(node, &buffers_in, &index_buffers, &mut output)?;
                }
                BoundOpKind::Reduce { keep: Keep::Reduce, .. } => {
                    run_reduce_typed(node, &buffers_in, &index_buffers, &mut output)?;
                }
                BoundOpKind::Reduce { keep: Keep::Scan, .. } => {
                    run_scan_typed(node, &buffers_in, &index_buffers, &mut output)?;
                }
                BoundOpKind::Iota => run_iota_typed(&mut output),
                BoundOpKind::Constant { value } => run_constant_typed(*value, &mut output),
            }
            buffers_in[node.node.0 as usize] = Some(Cow::Owned(output));
        } else {
            for (source, _, _) in node.operands() {
                let needs_widening =
                    buffers_out[source.0 as usize].is_none() && buffers_in[source.0 as usize].is_some();
                if needs_widening {
                    let narrow = buffers_in[source.0 as usize].as_deref().unwrap_or_default();
                    let mut widened: Vec<TAcc> = Vec::with_capacity(narrow.len());
                    for value in narrow {
                        widened.push(match block_on(converter.call(*value)) {
                            Ok(value) => value,
                            Err(never) => match never {},
                        });
                    }
                    buffers_out[source.0 as usize] = Some(Cow::Owned(widened));
                }
            }
            let mut output = typed_take_or_allocate(&mut free_out, node_output_len(node));
            match &node.kind {
                BoundOpKind::Elementwise { .. } => {
                    run_elementwise_typed(node, &buffers_out, &index_buffers, &mut output)?;
                }
                BoundOpKind::Reduce { keep: Keep::Reduce, .. } => {
                    run_reduce_typed(node, &buffers_out, &index_buffers, &mut output)?;
                }
                BoundOpKind::Reduce { keep: Keep::Scan, .. } => {
                    run_scan_typed(node, &buffers_out, &index_buffers, &mut output)?;
                }
                BoundOpKind::Iota => run_iota_typed(&mut output),
                BoundOpKind::Constant { value } => run_constant_typed(*value, &mut output),
            }
            buffers_out[node.node.0 as usize] = Some(Cow::Owned(output));
        }
        for retired in &retires[position] {
            typed_retire_into(&mut buffers_in, *retired, &mut free_in);
            typed_retire_into(&mut buffers_out, *retired, &mut free_out);
        }
    }

    Ok(effective_outputs
        .iter()
        .map(|node| {
            let shape = shapes.of(*node).to_vec();
            let data = buffers_out[node.0 as usize]
                .clone()
                .map(Cow::into_owned)
                .unwrap_or_default();
            (*node, shape, data)
        })
        .collect())
}

/// The typed counterpart of [`take_or_allocate`]: same best-fit-by-capacity
/// pool search, generic over [`Element`] instead of hardcoded to `f32`.
fn typed_take_or_allocate<T: Element>(pool: &mut Vec<Vec<T>>, required: usize) -> Vec<T> {
    let best_fit = pool
        .iter()
        .enumerate()
        .filter(|(_, buffer)| buffer.capacity() >= required)
        .min_by_key(|(_, buffer)| buffer.capacity())
        .map(|(index, _)| index);

    match best_fit {
        Some(index) => {
            let mut buffer = pool.swap_remove(index);
            buffer.resize(required, T::default());
            buffer
        }
        None => vec![T::default(); required],
    }
}

/// The typed counterpart of [`retire_into`]: same take-and-stash, generic
/// over [`Element`].
fn typed_retire_into<T: Element>(buffers: &mut [Option<Cow<'_, [T]>>], node: NodeId, pool: &mut Vec<Vec<T>>) {
    if let Some(Cow::Owned(buffer)) = buffers[node.0 as usize].take() {
        pool.push(buffer);
    }
}

/// The typed counterpart of [`operand_buffers`]: every node kind
/// ([`run_elementwise_typed`], [`run_reduce_generic`], [`run_scan_generic`])
/// reads its operands' physical buffers the same way, so this is the one
/// place that walk is written.
fn typed_operand_buffers<'a, T: Element>(
    resolved: &BoundOp,
    buffers: &'a [Option<Cow<'_, [T]>>],
) -> Result<Vec<&'a [T]>, TensorError> {
    resolved
        .operands()
        .iter()
        .map(|(source, _, _)| {
            buffers[source.0 as usize]
                .as_deref()
                .ok_or(TensorError::NotLowerable {
                    node: *source,
                    reason: "operand buffer missing at evaluation time",
                })
        })
        .collect()
}

/// The typed counterpart of [`run_iota`]: `output[i] = T::from_index(i)`, at
/// whichever width `T` calls for rather than only f32.
fn run_iota_typed<T: Element>(output: &mut [T]) {
    for (index, slot) in output.iter_mut().enumerate() {
        *slot = T::from_index(index);
    }
}

/// The typed counterpart of [`run_constant`].
fn run_constant_typed<T: Element>(value: f32, output: &mut [T]) {
    output.fill(T::from_literal(value));
}

/// The typed counterpart of [`run_elementwise`]: same coordinate walk
/// (`fill_running_offsets`/`unflatten_into`/`split_innermost` are pure
/// geometry over `&[u64]`/[`bind::Layout`], with no f32 dependence, so they
/// are shared verbatim) and the same [`GatherCursor`]/[`fill_gather_cursors_typed`]
/// gather step [`run_elementwise`]'s own generic loop uses, sourced from
/// `index_buffers` instead of `buffers` — no width-tile SIMD fast path,
/// which is still f32-only (see [`run_typed_program`]'s doc).
fn run_elementwise_typed<T: Element>(
    resolved: &BoundOp,
    buffers: &[Option<Cow<'_, [T]>>],
    index_buffers: &[Option<Vec<i64>>],
    output: &mut [T],
) -> Result<(), TensorError> {
    let (outer_extents, inner_len) = split_innermost(&resolved.extents);
    let innermost_dim = outer_extents.len() as u16;
    let raw = typed_operand_buffers(resolved, buffers)?;
    let body = resolved.element_body();
    let mut operand_values = vec![T::default(); raw.len()];
    let mut step_values = vec![T::default(); body.steps.len()];
    let strides: Vec<i64> = resolved
        .operands()
        .iter()
        .map(|(_, view, _)| view.stride(innermost_dim))
        .collect();
    let mut running: Vec<i64> = vec![0; raw.len()];
    let mut gather_cursors: Vec<Option<GatherCursor<'_, i64>>> = (0..raw.len()).map(|_| None).collect();
    let mut outer_coordinate = vec![0u64; outer_extents.len()];

    for outer_position in 0..odometer_len(outer_extents) as usize {
        unflatten_into(outer_position as u64, outer_extents, &mut outer_coordinate);
        fill_running_offsets(resolved, &outer_coordinate, &mut running);
        fill_gather_cursors_typed(
            resolved,
            index_buffers,
            &outer_coordinate,
            Some(innermost_dim),
            &mut gather_cursors,
        )?;
        let out_base = outer_position * inner_len;

        for step in 0..inner_len {
            for (index, data) in raw.iter().enumerate() {
                let mut offset = running[index];
                if let Some(cursor) = gather_cursors[index].as_mut() {
                    offset += cursor.fetch_and_advance(resolved.node)?;
                }
                operand_values[index] = data[offset as usize];
                running[index] += strides[index];
            }
            output[out_base + step] =
                eval_body_typed(resolved.node, body, &operand_values, &mut step_values)?;
        }
    }
    Ok(())
}

/// The typed counterpart of [`apply_body`]: same fused-step walk, fallible
/// per [`Element::apply`] instead of the f32 body's infallible one.
fn eval_body_typed<T: Element>(
    node: NodeId,
    body: &ComposedBody,
    operand_values: &[T],
    step_values: &mut [T],
) -> Result<T, TensorError> {
    for (index, step) in body.steps.iter().enumerate() {
        let mut args = [T::default(); 3];
        for (slot, arg) in step.args.iter().enumerate() {
            args[slot] = match arg {
                StepArg::Operand(operand_index) => operand_values[*operand_index as usize],
                StepArg::Step(step_index) => step_values[*step_index as usize],
            };
        }
        step_values[index] = T::apply(node, step.op, &args[..step.args.len()])?;
    }
    Ok(step_values[body.steps.len() - 1])
}

/// Reinterprets a `&[T]` as `&[f32]` with no copy.
///
/// # Safety
/// The caller must have already confirmed `TypeId::of::<T>() ==
/// TypeId::of::<f32>()`; only then are `T` and `f32` provably the same type,
/// which is what makes this pointer reinterpretation sound.
unsafe fn reinterpret_slice<T: 'static>(slice: &[T]) -> &[f32] {
    // SAFETY: forwarded from this function's own contract.
    unsafe { core::slice::from_raw_parts(slice.as_ptr().cast::<f32>(), slice.len()) }
}

/// The `&mut` counterpart of [`reinterpret_slice`]; same contract.
///
/// # Safety
/// See [`reinterpret_slice`].
unsafe fn reinterpret_slice_mut<T: 'static>(slice: &mut [T]) -> &mut [f32] {
    // SAFETY: forwarded from this function's own contract.
    unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<f32>(), slice.len()) }
}

/// [`Op::Reduce`] with `Keep::Reduce`, at any width [`Element`] covers.
///
/// This is the specialization point the module doc promises: for `T = f32`
/// with no gathered operand, it does not run a second reduction nest at all
/// — it reinterprets the typed evaluator's own `Vec<f32>` buffers as the
/// `&[f32]` the existing NEON-tiled [`run_reduce`] already takes (sound
/// because [`Element`]'s `'static` bound lets [`TypeId`] prove `T` really is
/// `f32` first) and calls that function directly, so the GEMM tiling,
/// dot-fold, and width-fast paths all still fire exactly as they do for
/// [`evaluate`]. A gathered operand skips this specialization even at `T =
/// f32`: [`run_reduce`]'s own [`fill_gather_cursors`] reads an index node's
/// value out of the *same* `&[f32]` buffer table its operands live in, but
/// the typed evaluator's index nodes live in the separate `index_buffers`
/// table [`canonical_index_buffers`] builds — reinterpreting `buffers` alone
/// would leave `run_reduce` unable to see them. Every other width, and every
/// gathered node regardless of width, falls through to
/// [`run_reduce_generic`].
fn run_reduce_typed<T: Element>(
    resolved: &BoundOp,
    buffers: &[Option<Cow<'_, [T]>>],
    index_buffers: &[Option<Vec<i64>>],
    output: &mut [T],
) -> Result<(), TensorError> {
    // Scatter is `f32`-only for now: `run_reduce_scatter` (the only
    // execution path this crate ships for a data-dependent `out_map`) reads
    // straight out of the `evaluate`/`evaluate_parallel` `&[f32]` buffer
    // table, not `evaluate_typed`'s per-width `Cow<'_, [T]>` one. Named,
    // never a silent fallback to a wrong (non-scatter) reduction.
    if let BoundOpKind::Reduce {
        out_scatter: Some(_),
        ..
    } = &resolved.kind
    {
        return Err(TensorError::NotLowerable {
            node: resolved.node,
            reason: "scatter is not yet supported by the typed (non-f32) evaluator",
        });
    }
    let has_gather = resolved.operands().iter().any(|(_, _, lookup)| lookup.is_some());
    if !has_gather && TypeId::of::<T>() == TypeId::of::<f32>() {
        let buffers_f32: Vec<Option<&[f32]>> = buffers
            .iter()
            .map(|slot| {
                slot.as_ref().map(|data| {
                    // SAFETY: the `TypeId` check above proves `T == f32`.
                    unsafe { reinterpret_slice(&data[..]) }
                })
            })
            .collect();
        // SAFETY: the `TypeId` check above proves `T == f32`.
        let output_f32 = unsafe { reinterpret_slice_mut(output) };
        return run_reduce(resolved, &buffers_f32, output_f32);
    }
    run_reduce_generic(resolved, buffers, index_buffers, output)
}

/// The scalar reduction nest generic over every [`Element`] width — the
/// same (leading, reduction) coordinate walk [`run_reduce`]'s own generic
/// fallback runs (its NEON/width-tile/dot-fold fast paths stay f32-only, so
/// this has no equivalent of them to port), rewritten against
/// [`Element::apply`]/[`eval_body_typed`] instead of `apply_scalar_op`/
/// `eval_body_shape` so it type-checks for every width, and fallible where
/// [`Element::apply`] is (an unsupported op, or an integer division that has
/// no representable result). Same [`GatherCursor`]/[`fill_gather_cursors_typed`]
/// step [`run_reduce`]'s own generic fallback uses, sourced from
/// `index_buffers`.
fn run_reduce_generic<T: Element>(
    resolved: &BoundOp,
    buffers: &[Option<Cow<'_, [T]>>],
    index_buffers: &[Option<Vec<i64>>],
    output: &mut [T],
) -> Result<(), TensorError> {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        output_axes,
        out_layout,
        ..
    } = &resolved.kind
    else {
        unreachable!("run_reduce_generic is only called for a Keep::Reduce fold")
    };
    let raw = typed_operand_buffers(resolved, buffers)?;
    let body = resolved.element_body();
    let mut operand_values = vec![T::default(); raw.len()];
    let mut step_values = vec![T::default(); body.steps.len()];

    let reduction_dims: Vec<u16> = (0..resolved.extents.len() as u16)
        .filter(|dim| !output_axes.as_slice().contains(dim))
        .collect();
    let (leading_output_axes, last_output_dim) = output_axes_split(output_axes.as_slice());
    let leading_extents: Vec<u64> = leading_output_axes
        .iter()
        .map(|dim| resolved.extents[*dim as usize])
        .collect();
    let reduction_extents: Vec<u64> = reduction_dims
        .iter()
        .map(|dim| resolved.extents[*dim as usize])
        .collect();
    let width = last_output_dim.map_or(1, |dim| resolved.extents[dim as usize] as usize);

    let strides: Vec<i64> = resolved
        .operands()
        .iter()
        .map(|(_, view, _)| last_output_dim.map_or(0, |dim| view.stride(dim)))
        .collect();
    let mut running: Vec<i64> = vec![0; raw.len()];
    let mut gather_cursors: Vec<Option<GatherCursor<'_, i64>>> = (0..raw.len()).map(|_| None).collect();
    let mut leading_coordinate = vec![0u64; leading_extents.len()];
    let mut reduction_coordinate = vec![0u64; reduction_extents.len()];
    let mut full_coordinate = vec![0u64; resolved.extents.len()];
    let reduction_total = odometer_len(&reduction_extents);
    let leading_total = odometer_len(&leading_extents);

    let seed = T::reduce_seed(*init).unwrap_or_default();
    let mut accumulator = vec![seed; width];

    for leading_flat in 0..leading_total {
        unflatten_into(leading_flat, &leading_extents, &mut leading_coordinate);
        accumulator.fill(seed);
        let mut seeded = !matches!(init, ReduceInit::FirstElement);

        for reduction_flat in 0..reduction_total {
            unflatten_into(reduction_flat, &reduction_extents, &mut reduction_coordinate);
            merge_coordinates_into(
                leading_output_axes,
                &leading_coordinate,
                &reduction_dims,
                &reduction_coordinate,
                &mut full_coordinate,
            );
            fill_running_offsets(resolved, &full_coordinate, &mut running);
            fill_gather_cursors_typed(
                resolved,
                index_buffers,
                &full_coordinate,
                last_output_dim,
                &mut gather_cursors,
            )?;

            for slot in &mut accumulator {
                for (index, data) in raw.iter().enumerate() {
                    let mut offset = running[index];
                    if let Some(cursor) = gather_cursors[index].as_mut() {
                        offset += cursor.fetch_and_advance(resolved.node)?;
                    }
                    operand_values[index] = data[offset as usize];
                    running[index] += strides[index];
                }
                let value = eval_body_typed(resolved.node, body, &operand_values, &mut step_values)?;
                *slot = if seeded {
                    T::apply(resolved.node, *reduce_op, &[*slot, value])?
                } else {
                    value
                };
            }
            seeded = true;
        }

        merge_coordinates_into(leading_output_axes, &leading_coordinate, &[], &[], &mut full_coordinate);
        let out_prefix = out_layout.offset_of(&full_coordinate);
        let out_stride = last_output_dim.map_or(0, |dim| out_layout.stride(dim));
        for (slot, value) in accumulator.iter().enumerate() {
            output[(out_prefix + out_stride * slot as i64) as usize] = *value;
        }
    }
    Ok(())
}

/// [`Op::Reduce`] with `Keep::Scan`, at any width [`Element`] covers — the
/// scan counterpart of [`run_reduce_typed`], same `T = f32`, gather-free
/// specialization down to the existing NEON-aware [`run_scan`] (see
/// [`run_reduce_typed`]'s own doc for why a gathered operand skips it).
fn run_scan_typed<T: Element>(
    resolved: &BoundOp,
    buffers: &[Option<Cow<'_, [T]>>],
    index_buffers: &[Option<Vec<i64>>],
    output: &mut [T],
) -> Result<(), TensorError> {
    let has_gather = resolved.operands().iter().any(|(_, _, lookup)| lookup.is_some());
    if !has_gather && TypeId::of::<T>() == TypeId::of::<f32>() {
        let buffers_f32: Vec<Option<&[f32]>> = buffers
            .iter()
            .map(|slot| {
                slot.as_ref().map(|data| {
                    // SAFETY: the `TypeId` check above proves `T == f32`.
                    unsafe { reinterpret_slice(&data[..]) }
                })
            })
            .collect();
        // SAFETY: the `TypeId` check above proves `T == f32`.
        let output_f32 = unsafe { reinterpret_slice_mut(output) };
        return run_scan(resolved, &buffers_f32, output_f32);
    }
    run_scan_generic(resolved, buffers, index_buffers, output)
}

/// The scalar scan nest generic over every [`Element`] width — [`run_scan`]'s
/// generic fallback (its width-fast SIMD path stays f32-only), rewritten
/// against [`Element::apply`]/[`eval_body_typed`] the same way
/// [`run_reduce_generic`] rewrites [`run_reduce`]'s, including the same
/// [`GatherCursor`]/[`fill_gather_cursors_typed`] step.
fn run_scan_generic<T: Element>(
    resolved: &BoundOp,
    buffers: &[Option<Cow<'_, [T]>>],
    index_buffers: &[Option<Vec<i64>>],
    output: &mut [T],
) -> Result<(), TensorError> {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        out_layout,
        ..
    } = &resolved.kind
    else {
        unreachable!("run_scan_generic is only called for a Keep::Scan fold")
    };
    let raw = typed_operand_buffers(resolved, buffers)?;
    let (outer_extents, inner_len) = split_innermost(&resolved.extents);
    let innermost_dim = outer_extents.len() as u16;
    let body = resolved.element_body();
    let mut operand_values = vec![T::default(); raw.len()];
    let mut step_values = vec![T::default(); body.steps.len()];
    let strides: Vec<i64> = resolved
        .operands()
        .iter()
        .map(|(_, view, _)| view.stride(innermost_dim))
        .collect();
    let mut running: Vec<i64> = vec![0; raw.len()];
    let mut gather_cursors: Vec<Option<GatherCursor<'_, i64>>> = (0..raw.len()).map(|_| None).collect();
    let mut outer_coordinate = vec![0u64; outer_extents.len()];

    let mut accumulator = T::reduce_seed(*init).unwrap_or_default();
    let mut seeded = !matches!(init, ReduceInit::FirstElement);

    for outer_flat in 0..odometer_len(outer_extents) {
        unflatten_into(outer_flat, outer_extents, &mut outer_coordinate);
        fill_running_offsets(resolved, &outer_coordinate, &mut running);
        fill_gather_cursors_typed(
            resolved,
            index_buffers,
            &outer_coordinate,
            Some(innermost_dim),
            &mut gather_cursors,
        )?;
        let mut out_running = out_layout.offset_of(&outer_coordinate);
        let out_stride = out_layout.stride(innermost_dim);

        for _ in 0..inner_len {
            for (index, data) in raw.iter().enumerate() {
                let mut offset = running[index];
                if let Some(cursor) = gather_cursors[index].as_mut() {
                    offset += cursor.fetch_and_advance(resolved.node)?;
                }
                operand_values[index] = data[offset as usize];
                running[index] += strides[index];
            }
            let value = eval_body_typed(resolved.node, body, &operand_values, &mut step_values)?;
            accumulator = if seeded {
                T::apply(resolved.node, *reduce_op, &[accumulator, value])?
            } else {
                value
            };
            seeded = true;
            output[out_running as usize] = accumulator;
            out_running += out_stride;
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::bind::BodyStep;
    use crate::map::{self, AxisTerm, IndexMap};
    use crate::op::{Extent, Reduce, append};

    use std::time::Instant;

    use crate::test_support::Lcg;


    /// `stage_offsets` for `stage_count` stages of a UNIFORM `chunks_per_stage`
    /// width -- the shape every pre-existing test used before `StagedRound`
    /// grew variable-width stages; kept as a fixture so those tests read the
    /// same as before, with the new field spelled out.
    fn uniform_stage_offsets(stage_count: usize, chunks_per_stage: usize) -> Vec<usize> {
        (0..=stage_count).map(|stage| stage * chunks_per_stage).collect()
    }

    /// The property [`StagedRound`] exists for: a chunk in stage `s` never
    /// observes stage `s - 1` incomplete. Driven from real threads against
    /// the real `run_chunk`, with chunks handed out off a monotonic cursor
    /// exactly the way `prime`'s cohort hands them out — that ordering is
    /// the precondition the barrier's deadlock-freedom argument rests on, so
    /// the test reproduces it rather than assuming it.
    ///
    /// [`parse_cpu_list_count`] against the exact shapes
    /// `/sys/devices/cpu_core/cpus` produces on a real hybrid Linux host --
    /// this crate's dev boxes are aarch64-darwin, so the sysfs file itself
    /// is unreachable here; this exercises the parser this Mac CAN run,
    /// leaving the `std::fs::read_to_string` wiring in
    /// [`performance_core_count`] compiled (`cargo check --target
    /// x86_64-unknown-linux-gnu`) but unexecuted on this machine.
    #[proxima::test]
    #[case::single_range("0-7,16-23", Some(16))]
    #[case::bare_id("4", Some(1))]
    #[case::single_cpu_range("4-4", Some(1))]
    #[case::trailing_newline("0-3\n", Some(4))]
    #[case::empty_file("", None)]
    #[case::whitespace_only("   \n", None)]
    #[case::malformed_range("abc", None)]
    async fn parse_cpu_list_count_matches_sysfs_shapes(#[case] text: &str, #[case] expected: Option<usize>) {
        assert_eq!(parse_cpu_list_count(text), expected);
    }

    /// Each stage's chunk asserts every earlier stage is fully published,
    /// then publishes its own slot. A missing barrier shows up as a stage
    /// reading a slot its predecessor had not written yet.
    #[test]
    fn staged_round_never_runs_a_stage_before_its_predecessor_completes() {
        const STAGES: usize = 6;
        const CHUNKS: usize = 4;
        const MEMBERS: usize = 3;

        let stage_offsets = uniform_stage_offsets(STAGES, CHUNKS);
        let completed: Vec<AtomicUsize> = (0..STAGES).map(|_| AtomicUsize::new(0)).collect();
        let published: Vec<AtomicUsize> = (0..STAGES * CHUNKS).map(|_| AtomicUsize::new(0)).collect();
        let violations = AtomicUsize::new(0);

        let round = StagedRound {
            stage_offsets: &stage_offsets,
            completed: &completed,
            run_stage_chunk: |stage: usize, within: usize| {
                for earlier in 0..stage {
                    for slot in 0..CHUNKS {
                        if published[earlier * CHUNKS + slot].load(Ordering::Acquire) != 1 {
                            violations.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                published[stage * CHUNKS + within].store(1, Ordering::Release);
                Ok(())
            },
        };

        let cursor = AtomicUsize::new(0);
        let total = round.chunks();
        std::thread::scope(|scope| {
            for _ in 0..MEMBERS {
                scope.spawn(|| {
                    loop {
                        let claimed = cursor.fetch_add(1, Ordering::Relaxed);
                        if claimed >= total {
                            break;
                        }
                        round.run_chunk(ChunkIndex(claimed)).expect("staged chunk must not fail");
                    }
                });
            }
        });

        assert_eq!(violations.load(Ordering::Relaxed), 0, "a stage ran before its predecessor completed");
        assert_eq!(total, STAGES * CHUNKS, "flat chunk space must cover every stage");
        for (index, slot) in published.iter().enumerate() {
            assert_eq!(slot.load(Ordering::Relaxed), 1, "chunk {index} never ran");
        }
    }

    /// Fewer members than stages is the case the deadlock-freedom argument
    /// has to cover: a member that claims a late stage waits on chunks whose
    /// owners may themselves be waiting. One member is the extreme.
    #[test]
    fn staged_round_completes_with_a_single_member() {
        const STAGES: usize = 5;
        const CHUNKS: usize = 2;

        let stage_offsets = uniform_stage_offsets(STAGES, CHUNKS);
        let completed: Vec<AtomicUsize> = (0..STAGES).map(|_| AtomicUsize::new(0)).collect();
        let ran = AtomicUsize::new(0);
        let round = StagedRound {
            stage_offsets: &stage_offsets,
            completed: &completed,
            run_stage_chunk: |_stage: usize, _within: usize| {
                ran.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
        };
        for chunk in 0..round.chunks() {
            round.run_chunk(ChunkIndex(chunk)).expect("staged chunk must not fail");
        }
        assert_eq!(ran.load(Ordering::Relaxed), STAGES * CHUNKS);
    }

    /// A failing chunk must still publish, or every member behind it hangs
    /// on the barrier instead of seeing the error through the round's report.
    #[test]
    fn staged_round_publishes_a_failed_chunk_so_later_stages_do_not_hang() {
        const CHUNKS: usize = 2;
        let stage_offsets = uniform_stage_offsets(2, CHUNKS);
        let completed: Vec<AtomicUsize> = (0..2).map(|_| AtomicUsize::new(0)).collect();
        let round = StagedRound {
            stage_offsets: &stage_offsets,
            completed: &completed,
            run_stage_chunk: |stage: usize, _within: usize| {
                if stage == 0 {
                    Err(TensorError::NotLowerable { node: NodeId(0), reason: "staged round error propagation fixture" })
                } else {
                    Ok(())
                }
            },
        };
        assert!(round.run_chunk(ChunkIndex(0)).is_err());
        assert!(round.run_chunk(ChunkIndex(1)).is_err());
        assert_eq!(completed[0].load(Ordering::Relaxed), CHUNKS, "a failed chunk must still publish");
        assert!(round.run_chunk(ChunkIndex(2)).is_ok(), "stage 1 must not be blocked by stage 0's error");
    }

    /// The property the whole matmul-fold design depends on
    /// (`docs/discipline.md` ROW 96's "what remains open"): a wide
    /// matmul-shaped stage (many chunks, real cross-worker parallelism) and
    /// a narrow elementwise-shaped stage (exactly one chunk) coexisting in
    /// the SAME round, neither one forced to match the other's width. Stage
    /// widths here are deliberately irregular (3, 1, 5, 1, 2) rather than a
    /// clean power of two, so a bug that only reproduces at a stage
    /// boundary math edge (off-by-one in `partition_point`'s translation
    /// back to `within_stage`) has somewhere to show up.
    #[test]
    fn staged_round_supports_variable_width_stages_in_one_round() {
        const WIDTHS: [usize; 5] = [3, 1, 5, 1, 2];
        let stage_offsets: Vec<usize> = core::iter::once(0)
            .chain(WIDTHS.iter().scan(0usize, |total, width| {
                *total += width;
                Some(*total)
            }))
            .collect();
        let completed: Vec<AtomicUsize> = (0..WIDTHS.len()).map(|_| AtomicUsize::new(0)).collect();
        let observed_widths: Vec<AtomicUsize> = (0..WIDTHS.len()).map(|_| AtomicUsize::new(0)).collect();
        let violations = AtomicUsize::new(0);

        let round = StagedRound {
            stage_offsets: &stage_offsets,
            completed: &completed,
            run_stage_chunk: |stage: usize, within: usize| {
                if within >= WIDTHS[stage] {
                    violations.fetch_add(1, Ordering::Relaxed);
                }
                observed_widths[stage].fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
        };

        assert_eq!(round.chunks(), WIDTHS.iter().sum::<usize>(), "flat chunk space must cover every stage's own width");
        for chunk in 0..round.chunks() {
            round.run_chunk(ChunkIndex(chunk)).expect("staged chunk must not fail");
        }
        assert_eq!(violations.load(Ordering::Relaxed), 0, "a chunk landed in the wrong stage or read the wrong within-stage index");
        for (stage, width) in WIDTHS.iter().enumerate() {
            assert_eq!(observed_widths[stage].load(Ordering::Relaxed), *width, "stage {stage} did not run exactly its own width in chunks");
        }
    }

    fn random_vec(seed: u64, count: usize) -> Vec<f32> {
        let mut lcg = Lcg(seed);
        (0..count).map(|_| lcg.next_unit()).collect()
    }

    /// The scalar reference [`run_elementwise`]'s own `Generic` arm takes:
    /// per position, read each operand via its own running offset (advancing
    /// by `strides` each step, mirroring [`fill_running_offsets`] plus the
    /// per-element loop body), then [`apply_body`]. No gather in any of
    /// these fixtures, so this omits [`GatherCursor`] entirely.
    fn reference_generic_body(body: &ComposedBody, raw: &[&[f32]], strides: &[i64], width: usize) -> Vec<f32> {
        let mut running = vec![0i64; raw.len()];
        let mut operand_values = vec![0.0f32; raw.len()];
        let mut step_values = vec![0.0f32; body.steps.len()];
        let mut out = Vec::with_capacity(width);
        for _ in 0..width {
            for (index, data) in raw.iter().enumerate() {
                operand_values[index] = data[running[index] as usize];
                running[index] += strides[index];
            }
            out.push(apply_body(body, &operand_values, &mut step_values));
        }
        out
    }

    fn step(op: ScalarOp, args: &[StepArg]) -> BodyStep {
        BodyStep {
            op,
            args: args.to_vec(),
        }
    }

    /// A synthetic instance of the 6-step SwiGLU chain named in
    /// `proxima-tensor/docs/discipline.md` ROW 5
    /// (`[Negate, Exponential, Add, Reciprocal, Multiply, Multiply]`):
    /// `silu(gate) * up` computed as `gate * sigmoid(gate) * up`, with
    /// `sigmoid(gate) = 1 / (1 + exp(-gate))` unrolled into the same five
    /// scalar steps a real fused body has. Operand 0 is `gate` (contiguous),
    /// operand 1 is a broadcast `1.0` (stride 0, standing in for a
    /// constant), operand 2 is `up` (contiguous) — exactly the affine
    /// precondition [`generic_body_is_affine_fast_path`] checks.
    fn swiglu_body() -> ComposedBody {
        ComposedBody {
            steps: vec![
                step(ScalarOp::Negate, &[StepArg::Operand(0)]),
                step(ScalarOp::Exponential, &[StepArg::Step(0)]),
                step(ScalarOp::Add, &[StepArg::Step(1), StepArg::Operand(1)]),
                step(ScalarOp::Reciprocal, &[StepArg::Step(2)]),
                step(ScalarOp::Multiply, &[StepArg::Step(3), StepArg::Operand(0)]),
                step(ScalarOp::Multiply, &[StepArg::Step(4), StepArg::Operand(2)]),
            ],
        }
    }

    /// A synthetic instance of the 6-step RMSNorm chain named in
    /// `proxima-tensor/docs/discipline.md` ROW 5
    /// (`[Multiply, Add, SquareRoot, Reciprocal, Multiply, Multiply]`):
    /// `x * (1 / sqrt(x*x + eps)) * weight`. Operand 0 is `x` (contiguous),
    /// operand 1 is a broadcast `eps` (stride 0), operand 2 is `weight`
    /// (contiguous).
    fn rmsnorm_body() -> ComposedBody {
        ComposedBody {
            steps: vec![
                step(ScalarOp::Multiply, &[StepArg::Operand(0), StepArg::Operand(0)]),
                step(ScalarOp::Add, &[StepArg::Step(0), StepArg::Operand(1)]),
                step(ScalarOp::SquareRoot, &[StepArg::Step(1)]),
                step(ScalarOp::Reciprocal, &[StepArg::Step(2)]),
                step(ScalarOp::Multiply, &[StepArg::Operand(0), StepArg::Step(3)]),
                step(ScalarOp::Multiply, &[StepArg::Step(4), StepArg::Operand(2)]),
            ],
        }
    }

    /// A synthetic instance of the 3-step RoPE chains named in
    /// `proxima-tensor/docs/discipline.md` ROW 5
    /// (`[Multiply, Multiply, Add]` / `[Multiply, Multiply, Subtract]`):
    /// `x * cos <op> y * sin`. All four operands (`x`, `cos`, `y`, `sin`)
    /// are contiguous.
    fn rope_body(combine: ScalarOp) -> ComposedBody {
        ComposedBody {
            steps: vec![
                step(ScalarOp::Multiply, &[StepArg::Operand(0), StepArg::Operand(1)]),
                step(ScalarOp::Multiply, &[StepArg::Operand(2), StepArg::Operand(3)]),
                step(combine, &[StepArg::Step(0), StepArg::Step(1)]),
            ],
        }
    }

    #[proxima::test]
    #[case::swiglu(swiglu_body(), vec![1, 0, 1])]
    #[case::rmsnorm(rmsnorm_body(), vec![1, 0, 1])]
    #[case::rope_add(rope_body(ScalarOp::Add), vec![1, 1, 1, 1])]
    #[case::rope_subtract(rope_body(ScalarOp::Subtract), vec![1, 1, 1, 1])]
    async fn elementwise_width_generic_matches_scalar_apply_body(
        #[case] body: ComposedBody,
        #[case] strides: Vec<i64>,
    ) {
        let width = 32;
        let operand_len = |stride: i64| if stride == 0 { 1 } else { width };
        let buffers: Vec<Vec<f32>> = strides
            .iter()
            .enumerate()
            .map(|(index, &stride)| {
                let values = random_vec(0x5eed_0000 + index as u64, operand_len(stride));
                // keep divisor-side and sqrt-side operands away from zero so
                // Reciprocal/SquareRoot stay finite and comparable exactly
                values.into_iter().map(|value| value.abs() + 0.25).collect()
            })
            .collect();
        let raw: Vec<&[f32]> = buffers.iter().map(Vec::as_slice).collect();

        let expected = reference_generic_body(&body, &raw, &strides, width);

        let running = vec![0i64; raw.len()];
        let mut step_values = vec![0.0f32; body.steps.len() * width];
        let mut actual = vec![0.0f32; width];
        elementwise_width_generic(&body, &raw, &running, &strides, &mut actual, &mut step_values);

        assert_eq!(actual, expected, "width-fast generic path must be bit-identical to the scalar apply_body path");
    }

    /// The exact shape `specs/rope.toml` maps against operand `x`
    /// (`"s,2*i->si"` / `"s,2*i+1->si"`): two views of the SAME underlying
    /// buffer, one reading even positions (stride 2, base 0), one reading odd
    /// positions (stride 2, base 1) — `x * cos + y * sin` with `x`/`y` both
    /// drawn from one physical buffer via [`OperandSpan::at`]. The reference
    /// pre-slices each view into its own contiguous buffer instead of relying
    /// on `reference_generic_body`'s hardcoded zero starting offset, so the
    /// two calls read identical values through different addressing.
    #[test]
    fn elementwise_width_generic_matches_scalar_apply_body_for_a_rope_shaped_stride_two_operand() {
        let width = 16;
        let x: Vec<f32> = random_vec(0x50fe_0000, 2 * width).into_iter().map(|value| value.abs() + 0.25).collect();
        let cos = random_vec(0x50fe_0001, width);
        let sin = random_vec(0x50fe_0002, width);

        let body = rope_body(ScalarOp::Add);
        let strides = vec![2i64, 1, 2, 1];
        let raw: Vec<&[f32]> = vec![x.as_slice(), cos.as_slice(), x.as_slice(), sin.as_slice()];
        let running = vec![0i64, 0, 1, 0];

        let x_even: Vec<f32> = (0..width).map(|position| x[2 * position]).collect();
        let x_odd: Vec<f32> = (0..width).map(|position| x[2 * position + 1]).collect();
        let reference_raw: Vec<&[f32]> = vec![x_even.as_slice(), cos.as_slice(), x_odd.as_slice(), sin.as_slice()];
        let reference_strides = vec![1i64, 1, 1, 1];
        let expected = reference_generic_body(&body, &reference_raw, &reference_strides, width);

        let mut step_values = vec![0.0f32; body.steps.len() * width];
        let mut actual = vec![0.0f32; width];
        elementwise_width_generic(&body, &raw, &running, &strides, &mut actual, &mut step_values);

        assert_eq!(
            actual.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            "width-fast generic path must be bit-identical to the scalar apply_body path for a stride-2 operand"
        );
    }

    /// [`reduce_dot_binary_monomorphic_strided`]'s load-bearing invariant:
    /// a strided dot fold must combine in the same strict left-to-right order
    /// [`Iterator::sum`] does, never reassociated the way the contiguous
    /// `DOT_LANES` path is. `x` is read at stride 2 (`x[2*k]`), `y` at stride
    /// 1 — the same shape a RoPE-adjacent contraction over an interleaved
    /// buffer would take.
    #[test]
    fn reduce_dot_binary_stride_two_matches_a_scalar_reference() {
        let k = 6usize;
        let mut program = Vec::new();
        let x = f32_block(&mut program, &[Extent::Static((2 * k) as u32)]);
        let y = f32_block(&mut program, &[Extent::Static(k as u32)]);
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (x, IndexMap::Affine(map::affine(1, &[(&[AxisTerm::scaled(0, 2)], 0)]))),
                    (y, IndexMap::Affine(map::projection(1, &[0]))),
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
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let x_data = random_vec(0x51de_0000, 2 * k);
        let y_data = random_vec(0x51de_0001, k);
        let evaluated = evaluate(&program, &[], &[&x_data, &y_data], &[]).expect("stride-2 dot reduce evaluates");

        let expected: f32 = (0..k).map(|position| x_data[2 * position] * y_data[position]).sum();
        assert_eq!(
            evaluated.root()[0].to_bits(),
            expected.to_bits(),
            "fast path must be bit-identical to the scalar reference for a stride-2 operand"
        );
    }

    /// [`scan_width_unary_monomorphic_strided`]'s equivalent invariant: a
    /// running sum over a stride-2 read must land on the exact same running
    /// total, position by position, as the plain scalar accumulation below.
    /// Built as a direct [`BoundOp`] (mirroring [`bound_op_for_gate`], for a
    /// `Reduce`/`Scan` shape instead of an `Elementwise` one) and run through
    /// [`run_scan`] directly, rather than through [`evaluate`]'s shape
    /// inference — a single scaled operand with no plain-projection sibling
    /// leaves iteration axis 0's extent unconstrained for inference to solve.
    #[test]
    fn scan_width_unary_stride_two_matches_a_running_sum_reference() {
        let k = 6usize;
        let data = random_vec(0x5ca4_0000, 2 * k);

        let resolved = BoundOp {
            node: NodeId(1),
            dtype: DType::Float32,
            extents: alloc::vec![k as u64],
            kind: BoundOpKind::Reduce {
                element_body: ComposedBody {
                    steps: alloc::vec![step(ScalarOp::Identity, &[StepArg::Operand(0)])],
                },
                reduce_op: ScalarOp::Add,
                init: ReduceInit::Zero,
                keep: Keep::Scan,
                operands: alloc::vec![(
                    NodeId(0),
                    bind::Layout {
                        base: 0,
                        strides: smallvec::smallvec![2],
                    },
                    None,
                )],
                output_axes: smallvec::smallvec![0],
                out_layout: bind::Layout {
                    base: 0,
                    strides: smallvec::smallvec![1],
                },
                out_scatter: None,
            },
        };

        let mut actual = vec![0.0f32; k];
        run_scan(&resolved, &[Some(data.as_slice())], &mut actual).expect("stride-2 cumsum runs");

        let mut running = 0.0f32;
        let reference: Vec<f32> = (0..k)
            .map(|position| {
                running += data[2 * position];
                running
            })
            .collect();

        assert_eq!(
            actual.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            reference.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            "fast path must be bit-identical to the scalar reference for a stride-2 operand"
        );
    }


    /// A minimal [`BoundOp`] carrying `body` and one operand per stride in
    /// `strides` — `Layout`/`node` are placeholders `operand_is_affine`
    /// never reads, only `strides` (passed separately, matching
    /// `run_elementwise`'s own precomputed table) and `gather` matter.
    fn bound_op_for_gate(body: ComposedBody, strides: &[i64], gather_operand: Option<usize>) -> BoundOp {
        let operands = strides
            .iter()
            .enumerate()
            .map(|(index, _)| {
                let gather = (Some(index) == gather_operand).then(|| bind::Lookup {
                    indices: NodeId(0),
                    index_layout: bind::Layout {
                        base: 0,
                        strides: smallvec::smallvec![0],
                    },
                    element_stride: 1,
                    extent: 1,
                });
                (
                    NodeId(index as u32),
                    bind::Layout {
                        base: 0,
                        strides: smallvec::smallvec![0],
                    },
                    gather,
                )
            })
            .collect();
        BoundOp {
            node: NodeId(strides.len() as u32),
            dtype: DType::Float32,
            extents: vec![32],
            kind: BoundOpKind::Elementwise { body, operands },
        }
    }

    #[proxima::test]
    #[case::swiglu(swiglu_body(), vec![1, 0, 1])]
    #[case::rmsnorm(rmsnorm_body(), vec![1, 0, 1])]
    #[case::rope_add(rope_body(ScalarOp::Add), vec![1, 1, 1, 1])]
    async fn generic_body_is_affine_fast_path_accepts_gather_free_affine_operands(
        #[case] body: ComposedBody,
        #[case] strides: Vec<i64>,
    ) {
        let resolved = bound_op_for_gate(body.clone(), &strides, None);
        assert!(generic_body_is_affine_fast_path(&resolved, &body, &strides));
    }

    #[proxima::test]
    #[case::swiglu(swiglu_body(), vec![1, 0, 1])]
    #[case::rmsnorm(rmsnorm_body(), vec![1, 0, 1])]
    async fn generic_body_is_affine_fast_path_rejects_a_gathered_operand(
        #[case] body: ComposedBody,
        #[case] strides: Vec<i64>,
    ) {
        let resolved = bound_op_for_gate(body.clone(), &strides, Some(0));
        assert!(!generic_body_is_affine_fast_path(&resolved, &body, &strides));
    }

    #[proxima::test]
    #[case::swiglu(swiglu_body(), vec![2, 0, 1])]
    #[case::rmsnorm(rmsnorm_body(), vec![1, 0, 2])]
    async fn generic_body_is_affine_fast_path_accepts_a_non_negative_constant_stride(
        #[case] body: ComposedBody,
        #[case] strides: Vec<i64>,
    ) {
        let resolved = bound_op_for_gate(body.clone(), &strides, None);
        assert!(generic_body_is_affine_fast_path(&resolved, &body, &strides));
    }

    #[proxima::test]
    #[case::swiglu(swiglu_body(), vec![-1, 0, 1])]
    #[case::rmsnorm(rmsnorm_body(), vec![1, 0, -1])]
    async fn generic_body_is_affine_fast_path_rejects_a_negative_stride(
        #[case] body: ComposedBody,
        #[case] strides: Vec<i64>,
    ) {
        let resolved = bound_op_for_gate(body.clone(), &strides, None);
        assert!(!generic_body_is_affine_fast_path(&resolved, &body, &strides));
    }

    /// Dispatches `rows` through the same `claim_and_run_rows` shared-cursor
    /// mechanism [`matmul_rows_threaded`] uses, but with `oversubscribe`
    /// passed in instead of hard-coded to [`crate::sized::ROW_OVERSUBSCRIBE`]
    /// — lets [`bench_row_oversubscribe_picks_the_multiplier`] sweep the
    /// multiplier without a rebuild per value. Test-only duplication of
    /// `matmul_rows_threaded`'s body; not shipped (`#[cfg(test)]`).
    fn dispatch_rows_with_oversubscribe<Row>(
        rows: usize,
        workers: usize,
        oversubscribe: usize,
        dot_row: Row,
    ) -> Vec<f32>
    where
        Row: Fn(usize) -> Result<f32, TensorError> + Sync,
    {
        // adapts the scalar `Row` this test builds (matches every real
        // caller's own dot-product closure shape) into
        // `claim_and_run_rows`'s `width`-slot form with `width == 1`, then
        // hands off to a generic-over-the-adapted-closure-type inner
        // dispatcher so the unsafe pointer cast below can name that type via
        // its own fresh generic parameter (a bare closure has no nameable
        // type to turbofish with).
        dispatch_rows_widened(rows, workers, oversubscribe, 1, move |row, slot| {
            slot[0] = dot_row(row)?;
            Ok(())
        })
    }

    fn dispatch_rows_widened<Wide>(
        rows: usize,
        workers: usize,
        oversubscribe: usize,
        width: usize,
        dot_row: Wide,
    ) -> Vec<f32>
    where
        Wide: Fn(usize, &mut [f32]) -> Result<(), TensorError> + Sync,
    {
        let mut output = vec![0.0f32; rows * width];
        let chunk_count = (workers.saturating_mul(oversubscribe)).clamp(1, rows.max(1));
        let chunk_len = rows.div_ceil(chunk_count);

        let mut chunk_ranges = Vec::with_capacity(chunk_count);
        let mut remaining = output.as_mut_slice();
        let mut row_start = 0usize;
        while !remaining.is_empty() {
            let take_rows = chunk_len.min(remaining.len() / width);
            let (slice, rest) = remaining.split_at_mut(take_rows * width);
            remaining = rest;
            chunk_ranges.push((row_start, slice.as_mut_ptr() as usize, slice.len()));
            row_start += take_rows;
        }
        let chunk_ranges_len = chunk_ranges.len();

        let pool = nest_pool().expect("pool builds under test");
        let dot_row_address = &dot_row as *const Wide as usize;
        let next_index = Arc::new(AtomicUsize::new(0));
        let chunk_ranges: Arc<Vec<(usize, usize, usize)>> = Arc::new(chunk_ranges);
        let spawned_count = workers.saturating_sub(1).min(chunk_ranges_len.saturating_sub(1));
        let (result_sender, result_receiver) = sync_channel(chunk_ranges_len);

        for _ in 0..spawned_count {
            let sender = result_sender.clone();
            let next_index = Arc::clone(&next_index);
            let chunk_ranges = Arc::clone(&chunk_ranges);
            drop(pool.spawn(move || {
                claim_and_run_rows::<Wide>(&next_index, dot_row_address, width, &chunk_ranges, &sender);
                Ok::<(), _>(())
            }));
        }
        claim_and_run_rows::<Wide>(&next_index, dot_row_address, width, &chunk_ranges, &result_sender);
        drop(result_sender);

        for _ in 0..chunk_ranges_len {
            let _ = result_receiver.recv();
        }
        output
    }

    /// Manual microbench picking [`crate::sized::ROW_OVERSUBSCRIBE`] —
    /// principle 18/19: a design constant needs a measurement artifact, not
    /// reasoning. Synthetic per-row cost is deliberately imbalanced (the
    /// last 1/8 of rows costs ~8x a normal row, echoing the 2.04x
    /// equal-row-count spread [`OVERSUBSCRIBE`]'s own doc records for a real
    /// GEMM) so a static 1:1 split leaves the calling thread idling in
    /// `Receiver::recv` for whichever puller drew the straggler range.
    /// `#[ignore]`: manual, not part of the CI gate — run with
    /// `cargo test -p proxima-tensor --release bench_row_oversubscribe -- --ignored --nocapture`.
    #[test]
    #[ignore = "manual microbench, not a CI gate; see this test's own doc"]
    fn bench_row_oversubscribe_picks_the_multiplier() {
        let workers = thread::available_parallelism().map(NonZeroUsize::get).unwrap_or(1);
        let rows = 4096usize;
        let straggler_start = rows - rows / 8;
        let cost_of = |row: usize| -> u64 {
            if row >= straggler_start { 4000 } else { 500 }
        };
        let dot_row = |row: usize| -> Result<f32, TensorError> {
            let mut accumulator = 0.0f32;
            for iteration in 0..cost_of(row) {
                accumulator += (iteration as f32).sin();
            }
            Ok(accumulator)
        };

        let load = std::process::Command::new("uptime")
            .output()
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
            .unwrap_or_else(|_| "uptime unavailable".to_string());
        eprintln!("workers={workers} rows={rows} ambient_load={load}");

        eprintln!("-- imbalanced (straggler last 1/8 of rows) --");
        for oversubscribe in [1usize, 2, 4, 8, 16, 32] {
            let mut samples_micros = Vec::with_capacity(5);
            for _ in 0..5 {
                let started = Instant::now();
                let output = dispatch_rows_with_oversubscribe(rows, workers, oversubscribe, dot_row);
                let elapsed = started.elapsed().as_micros() as f64;
                assert_eq!(output.len(), rows);
                samples_micros.push(elapsed);
            }
            let mean = samples_micros.iter().sum::<f64>() / samples_micros.len() as f64;
            let variance = samples_micros.iter().map(|sample| (sample - mean).powi(2)).sum::<f64>()
                / samples_micros.len() as f64;
            let coefficient_of_variation = variance.sqrt() / mean;
            eprintln!(
                "oversubscribe={oversubscribe} mean_us={mean:.1} cov={coefficient_of_variation:.4} samples={samples_micros:?}"
            );
        }

        // degenerate control (principle 19/V5): a UNIFORM per-row cost has
        // nothing to steal around, so this arm isolates the atomic-cursor
        // and `SyncSender` overhead oversubscription adds without any
        // imbalance to pay for it. If a high multiplier regresses here, that
        // is the ceiling on how far oversubscription can be pushed once real
        // rows are small enough for per-chunk overhead to matter.
        let uniform_dot_row = |row: usize| -> Result<f32, TensorError> {
            let mut accumulator = 0.0f32;
            for iteration in 0..1200u64 {
                accumulator += ((row as f32) + iteration as f32).sin();
            }
            Ok(accumulator)
        };
        eprintln!("-- uniform cost (degenerate control) --");
        for oversubscribe in [1usize, 2, 4, 8, 16, 32] {
            let mut samples_micros = Vec::with_capacity(5);
            for _ in 0..5 {
                let started = Instant::now();
                let output =
                    dispatch_rows_with_oversubscribe(rows, workers, oversubscribe, uniform_dot_row);
                let elapsed = started.elapsed().as_micros() as f64;
                assert_eq!(output.len(), rows);
                samples_micros.push(elapsed);
            }
            let mean = samples_micros.iter().sum::<f64>() / samples_micros.len() as f64;
            let variance = samples_micros.iter().map(|sample| (sample - mean).powi(2)).sum::<f64>()
                / samples_micros.len() as f64;
            let coefficient_of_variation = variance.sqrt() / mean;
            eprintln!(
                "oversubscribe={oversubscribe} mean_us={mean:.1} cov={coefficient_of_variation:.4} samples={samples_micros:?}"
            );
        }
    }

    /// [`matmul_q4k_f32`] against the reference path (`proxima_gguf`'s own
    /// tested dequantize into a full `f32` weight matrix, then a naive f32
    /// dot product) — [`super`]'s guiding-principle 14: the incumbent
    /// (dequantize-then-matmul) is correct by construction, so this is a
    /// parity check, not a round-trip-to-self check. Two real (pseudo-random,
    /// non-degenerate — `Lcg`, not zeros/constants) rows x 2 super-blocks
    /// (512 elements) each, `Q4_K`'s minimum non-trivial multi-block shape.
    #[proxima::test]
    #[case::seed_1(1)]
    #[case::seed_7(7)]
    #[case::seed_1000(1000)]
    async fn matmul_q4k_f32_matches_dequantize_then_f32_matmul(#[case] seed: u64) {
        use proxima_gguf::quant::q4_k::{QK_K, dequantize, quantize};

        const ROWS: usize = 2;
        const BLOCKS_PER_ROW: usize = 2;
        const K: usize = BLOCKS_PER_ROW * QK_K;
        const ROW_BYTES: usize = BLOCKS_PER_ROW * Q4K_BLOCK_BYTES;

        let weights_f32 = random_vec(seed, ROWS * K);
        let activation = random_vec(seed.wrapping_add(1), K);

        let mut packed = vec![0u8; ROWS * BLOCKS_PER_ROW * Q4K_BLOCK_BYTES];
        for (row_f32, row_packed) in weights_f32.as_chunks::<K>().0.iter().zip(packed.as_chunks_mut::<ROW_BYTES>().0) {
            quantize(row_f32, row_packed).expect("2 whole super-blocks quantize cleanly");
        }

        let mut dequantized_reference = vec![0.0f32; ROWS];
        let mut dequantized_row = vec![0.0f32; K];
        for (row_index, row_packed) in packed.as_chunks::<ROW_BYTES>().0.iter().enumerate() {
            dequantize(row_packed, &mut dequantized_row).expect("2 whole super-blocks dequantize cleanly");
            dequantized_reference[row_index] =
                dequantized_row.iter().zip(&activation).map(|(weight, value)| weight * value).sum();
        }

        let quantized_result = matmul_q4k_f32(&packed, ROWS, &activation).expect("well-formed quantized matmul");

        let mut max_diff = 0.0f32;
        let mut sum_sq_diff = 0.0f64;
        for (got, want) in quantized_result.iter().zip(&dequantized_reference) {
            let diff = (got - want).abs();
            max_diff = max_diff.max(diff);
            sum_sq_diff += f64::from(diff) * f64::from(diff);
        }
        let rms_diff = (sum_sq_diff / ROWS as f64).sqrt();
        eprintln!("matmul_q4k_f32 vs dequantize-then-matmul: seed={seed} max_diff={max_diff} rms_diff={rms_diff}");

        // not bit-exact: `dot_q4k_f32` folds one super-block at a time in a
        // single running accumulator, while the reference sums a
        // fully-materialized 512-element row in one linear pass — same
        // terms, different intermediate rounding. Loose bound, not tuned to
        // the measured numbers, matching `q4_k.rs`'s own round-trip tests.
        assert!(max_diff < 1e-2, "max_diff={max_diff} exceeds parity tolerance");
        assert!(rms_diff < 1e-2, "rms_diff={rms_diff} exceeds parity tolerance");
    }

    /// [`matmul_worker_count`]'s only machine-independent invariant: the
    /// selected count is never zero (nothing would run) and never more than
    /// `available_parallelism()` (oversubscription past the OS-reported
    /// core count). The exact value — P-core count on Apple, full logical
    /// count elsewhere — is machine-dependent and deliberately not asserted
    /// here.
    #[test]
    fn matmul_worker_count_is_between_one_and_available_parallelism() {
        let available = thread::available_parallelism().map(NonZeroUsize::get).unwrap_or(1);
        let workers = matmul_worker_count();
        assert!(workers >= 1, "worker count must be at least 1, got {workers}");
        assert!(
            workers <= available,
            "worker count {workers} exceeds available_parallelism {available}"
        );
    }

    /// [`matmul_rows_threaded`]'s pool dispatch (through [`matmul_q4k_f32`])
    /// against the same per-row kernel run sequentially in this test, no
    /// threading involved — 128 rows x 512 elements clears
    /// [`PARALLEL_THRESHOLD`] (65536 macs vs 4096) and is wide enough that
    /// `quantized_matmul_workers` returns `Some` on any machine with fewer
    /// than 128 hardware threads, so `matmul_q4k_f32` provably takes the
    /// pool path here. Each output row is an independent reduction with no
    /// cross-row accumulator (`dot_row` reads only its own row's bytes), so
    /// dispatch mechanism cannot perturb any one row's rounding — the pool
    /// and a bare sequential loop over the identical `dot_q4k_f32` calls
    /// must agree bit-for-bit, not just within a numeric tolerance.
    #[test]
    fn matmul_q4k_f32_threaded_pool_dispatch_matches_the_sequential_per_row_kernel() {
        use proxima_gguf::quant::q4_k::quantize;

        const ROWS: usize = 128;
        const BLOCKS_PER_ROW: usize = 2;
        const K: usize = BLOCKS_PER_ROW * proxima_gguf::quant::q4_k::QK_K;
        const ROW_BYTES: usize = BLOCKS_PER_ROW * Q4K_BLOCK_BYTES;

        let weights_f32 = random_vec(42, ROWS * K);
        let activation = random_vec(43, K);

        let mut packed = vec![0u8; ROWS * BLOCKS_PER_ROW * Q4K_BLOCK_BYTES];
        for (row_f32, row_packed) in weights_f32.as_chunks::<K>().0.iter().zip(packed.as_chunks_mut::<ROW_BYTES>().0) {
            quantize(row_f32, row_packed).expect("2 whole super-blocks quantize cleanly");
        }

        assert!(
            quantized_matmul_workers(ROWS, activation.len()).is_some(),
            "test fixture must actually clear the parallel threshold to exercise the pool path"
        );

        let pooled_result = matmul_q4k_f32(&packed, ROWS, &activation).expect("well-formed quantized matmul");

        let sequential_reference: Vec<f32> = packed
            .as_chunks::<ROW_BYTES>()
            .0
            .iter()
            .map(|weight_row| dot_q4k_f32(weight_row, &activation).expect("well-formed row"))
            .collect();

        assert_eq!(
            pooled_result, sequential_reference,
            "pool-dispatched rows must be bit-identical to the sequential per-row kernel: \
             each row is an independent reduction, so dispatch mechanism cannot move rounding"
        );
    }

    /// [`matmul_q5k_f32`] was unconditionally sequential (never called
    /// [`quantized_matmul_workers`]) before it was routed through the same
    /// `matmul_quantized_dispatch` helper `matmul_q4k_f32` uses — same test
    /// shape as `matmul_q4k_f32_threaded_pool_dispatch_matches_the_sequential_per_row_kernel`,
    /// proving the fix actually reaches the pool path and stays bit-exact.
    #[test]
    fn matmul_q5k_f32_threaded_pool_dispatch_matches_the_sequential_per_row_kernel() {
        use proxima_gguf::quant::q5_k::quantize;

        const ROWS: usize = 128;
        const BLOCKS_PER_ROW: usize = 2;
        const K: usize = BLOCKS_PER_ROW * proxima_gguf::quant::q5_k::QK_K;
        const ROW_BYTES: usize = BLOCKS_PER_ROW * Q5K_BLOCK_BYTES;

        let weights_f32 = random_vec(44, ROWS * K);
        let activation = random_vec(45, K);

        let mut packed = vec![0u8; ROWS * BLOCKS_PER_ROW * Q5K_BLOCK_BYTES];
        for (row_f32, row_packed) in weights_f32.as_chunks::<K>().0.iter().zip(packed.as_chunks_mut::<ROW_BYTES>().0) {
            quantize(row_f32, row_packed).expect("2 whole super-blocks quantize cleanly");
        }

        assert!(
            quantized_matmul_workers(ROWS, activation.len()).is_some(),
            "test fixture must actually clear the parallel threshold to exercise the pool path"
        );

        let pooled_result = matmul_q5k_f32(&packed, ROWS, &activation).expect("well-formed quantized matmul");

        let sequential_reference: Vec<f32> = packed
            .as_chunks::<ROW_BYTES>()
            .0
            .iter()
            .map(|weight_row| dot_q5k_f32(weight_row, &activation).expect("well-formed row"))
            .collect();

        assert_eq!(
            pooled_result, sequential_reference,
            "pool-dispatched rows must be bit-identical to the sequential per-row kernel: \
             each row is an independent reduction, so dispatch mechanism cannot move rounding"
        );
    }

    /// [`matmul_q6k_f32`]'s counterpart to the `matmul_q5k_f32` test above —
    /// same was-always-sequential bug, same fix, same bit-exactness proof.
    #[test]
    fn matmul_q6k_f32_threaded_pool_dispatch_matches_the_sequential_per_row_kernel() {
        use proxima_gguf::quant::q6_k::quantize;

        const ROWS: usize = 128;
        const BLOCKS_PER_ROW: usize = 2;
        const K: usize = BLOCKS_PER_ROW * proxima_gguf::quant::q6_k::QK_K;
        const ROW_BYTES: usize = BLOCKS_PER_ROW * Q6K_BLOCK_BYTES;

        let weights_f32 = random_vec(46, ROWS * K);
        let activation = random_vec(47, K);

        let mut packed = vec![0u8; ROWS * BLOCKS_PER_ROW * Q6K_BLOCK_BYTES];
        for (row_f32, row_packed) in weights_f32.as_chunks::<K>().0.iter().zip(packed.as_chunks_mut::<ROW_BYTES>().0) {
            quantize(row_f32, row_packed).expect("2 whole super-blocks quantize cleanly");
        }

        assert!(
            quantized_matmul_workers(ROWS, activation.len()).is_some(),
            "test fixture must actually clear the parallel threshold to exercise the pool path"
        );

        let pooled_result = matmul_q6k_f32(&packed, ROWS, &activation).expect("well-formed quantized matmul");

        let sequential_reference: Vec<f32> = packed
            .as_chunks::<ROW_BYTES>()
            .0
            .iter()
            .map(|weight_row| dot_q6k_f32(weight_row, &activation).expect("well-formed row"))
            .collect();

        assert_eq!(
            pooled_result, sequential_reference,
            "pool-dispatched rows must be bit-identical to the sequential per-row kernel: \
             each row is an independent reduction, so dispatch mechanism cannot move rounding"
        );
    }

    /// [`row_chunk_count`] must scale down with total work, not stay pinned
    /// to `workers * ROW_OVERSUBSCRIBE` for every shape -- the defect this
    /// session fixes: a narrow, low-mac call (`attn_k`/`attn_v`'s real
    /// `rows=1024 k=4096` shape, 4.19M macs) was paying the same fixed
    /// 40-way dispatch as a wide, high-mac call (`ffn_up`/`ffn_gate`'s
    /// shape) despite carrying far less work. `rows` held fixed across both
    /// cases so only `contraction_width` (the mac count) drives the
    /// difference; the wide case's `contraction_width` is synthetic
    /// (comfortably above `MIN_MACS_PER_CHUNK * oversubscribed_ceiling`)
    /// purely to prove the cap still applies once a shape carries enough
    /// work to earn the full oversubscribed split.
    #[test]
    fn row_chunk_count_scales_down_for_a_small_shape_and_stays_capped_for_a_large_one() {
        let workers = 10;
        let rows = 1024;

        let small_shape_chunks = row_chunk_count(rows, workers, 4096); // attn_k/attn_v's real k
        let large_shape_chunks = row_chunk_count(rows, workers, 100_000); // comfortably wide

        let oversubscribed_ceiling = workers * ROW_OVERSUBSCRIBE;
        assert!(
            small_shape_chunks < large_shape_chunks,
            "a low-mac shape must produce fewer chunks than a high-mac shape at the same row \
             count: small={small_shape_chunks} large={large_shape_chunks}"
        );
        assert_eq!(
            large_shape_chunks, oversubscribed_ceiling,
            "a shape whose total work clears MIN_MACS_PER_CHUNK * oversubscribed_ceiling must \
             still land on the full oversubscribed split"
        );
        assert!(small_shape_chunks >= 1, "chunk count must never be zero");
    }

    /// [`matmul_q5k_q8k_f32`] was unconditionally sequential (never called
    /// [`quantized_matmul_workers`]) despite [`matmul_q4k_q8k_f32`] already
    /// routing through the pool — same test shape as
    /// `matmul_q4k_f32_threaded_pool_dispatch_matches_the_sequential_per_row_kernel`,
    /// proving the fix actually reaches the pool path and stays bit-exact.
    #[cfg(feature = "q5k-int8-dot")]
    #[test]
    fn matmul_q5k_q8k_f32_threaded_pool_dispatch_matches_the_sequential_per_row_kernel() {
        use proxima_gguf::quant::q5_k::quantize;

        const ROWS: usize = 128;
        const BLOCKS_PER_ROW: usize = 2;
        const K: usize = BLOCKS_PER_ROW * proxima_gguf::quant::q5_k::QK_K;
        const ROW_BYTES: usize = BLOCKS_PER_ROW * Q5K_BLOCK_BYTES;

        let weights_f32 = random_vec(48, ROWS * K);
        let activation = random_vec(49, K);

        let mut packed = vec![0u8; ROWS * BLOCKS_PER_ROW * Q5K_BLOCK_BYTES];
        for (row_f32, row_packed) in weights_f32.as_chunks::<K>().0.iter().zip(packed.as_chunks_mut::<ROW_BYTES>().0) {
            quantize(row_f32, row_packed).expect("2 whole super-blocks quantize cleanly");
        }

        assert!(
            quantized_matmul_workers(ROWS, activation.len()).is_some(),
            "test fixture must actually clear the parallel threshold to exercise the pool path"
        );

        let pooled_result = matmul_q5k_q8k_f32(&packed, ROWS, &activation).expect("well-formed quantized matmul");

        let mut activation_q8k = vec![0u8; BLOCKS_PER_ROW * Q8K_BLOCK_BYTES];
        quantize_row_q8k(&activation, &mut activation_q8k).expect("well-formed activation");
        let sequential_reference: Vec<f32> = packed
            .as_chunks::<ROW_BYTES>()
            .0
            .iter()
            .map(|weight_row| dot_q5k_q8k(weight_row, &activation_q8k).expect("well-formed row"))
            .collect();

        assert_eq!(
            pooled_result, sequential_reference,
            "pool-dispatched rows must be bit-identical to the sequential per-row kernel: \
             each row is an independent reduction, so dispatch mechanism cannot move rounding"
        );
    }

    /// [`matmul_q6k_q8k_f32`]'s counterpart to the `matmul_q5k_q8k_f32` test
    /// above — same was-always-sequential bug, same fix, same bit-exactness
    /// proof.
    #[cfg(feature = "q6k-int8-dot")]
    #[test]
    fn matmul_q6k_q8k_f32_threaded_pool_dispatch_matches_the_sequential_per_row_kernel() {
        use proxima_gguf::quant::q6_k::quantize;

        const ROWS: usize = 128;
        const BLOCKS_PER_ROW: usize = 2;
        const K: usize = BLOCKS_PER_ROW * proxima_gguf::quant::q6_k::QK_K;
        const ROW_BYTES: usize = BLOCKS_PER_ROW * Q6K_BLOCK_BYTES;

        let weights_f32 = random_vec(50, ROWS * K);
        let activation = random_vec(51, K);

        let mut packed = vec![0u8; ROWS * BLOCKS_PER_ROW * Q6K_BLOCK_BYTES];
        for (row_f32, row_packed) in weights_f32.as_chunks::<K>().0.iter().zip(packed.as_chunks_mut::<ROW_BYTES>().0) {
            quantize(row_f32, row_packed).expect("2 whole super-blocks quantize cleanly");
        }

        assert!(
            quantized_matmul_workers(ROWS, activation.len()).is_some(),
            "test fixture must actually clear the parallel threshold to exercise the pool path"
        );

        let pooled_result = matmul_q6k_q8k_f32(&packed, ROWS, &activation).expect("well-formed quantized matmul");

        let mut activation_q8k = vec![0u8; BLOCKS_PER_ROW * Q8K_BLOCK_BYTES];
        quantize_row_q8k(&activation, &mut activation_q8k).expect("well-formed activation");
        let sequential_reference: Vec<f32> = packed
            .as_chunks::<ROW_BYTES>()
            .0
            .iter()
            .map(|weight_row| dot_q6k_q8k(weight_row, &activation_q8k).expect("well-formed row"))
            .collect();

        assert_eq!(
            pooled_result, sequential_reference,
            "pool-dispatched rows must be bit-identical to the sequential per-row kernel: \
             each row is an independent reduction, so dispatch mechanism cannot move rounding"
        );
    }

    /// [`reject_non_float32`]'s quantized-weight exemption: a `UInt8`-tagged
    /// node (standing in for packed `Q4_K` bytes) used as one operand of a
    /// `Multiply`-then-`Add`-reduce (matmul) now type-checks when named in
    /// the exemption set, and still rejects everything else exactly as
    /// before — proving "exactly as far as needed and no further."
    #[test]
    fn reject_non_float32_exempts_a_quantized_weight_in_matmul_position() {
        let mut program = Vec::new();
        let weight = block(&mut program, DType::UInt8, &[Extent::Static(4)]);
        let activation = f32_block(&mut program, &[Extent::Static(4)]);
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(map::projection(1, &[0]))),
                    (activation, IndexMap::Affine(map::projection(1, &[0]))),
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
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );

        assert!(
            reject_non_float32(&program, &BTreeSet::new()).is_err(),
            "an unexempted UInt8 node must still be rejected"
        );

        let mut exempt = BTreeSet::new();
        exempt.insert(weight);
        assert!(
            reject_non_float32(&program, &exempt).is_ok(),
            "a UInt8 node used exclusively as a matmul weight operand must be exempted"
        );
    }

    /// The exemption is shape-scoped, not tag-scoped: a `UInt8` node that is
    /// NOT feeding a `Multiply`-then-`Add` reduce is rejected even when
    /// named in the exemption set.
    #[test]
    fn reject_non_float32_still_rejects_a_quantized_node_outside_matmul_shape() {
        let mut program = Vec::new();
        let weight = block(&mut program, DType::UInt8, &[Extent::Static(4)]);
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: weight,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let mut exempt = BTreeSet::new();
        exempt.insert(weight);
        assert!(
            reject_non_float32(&program, &exempt).is_err(),
            "a quantized node reduced directly (no Multiply) is not the matmul shape and must stay rejected"
        );
    }

    /// The dead-leaf exemption's whole point: an ONNX-shaped program where an
    /// `Int64` `Op::Input` (standing in for a `Reshape`'s shape initializer)
    /// is never read by anything else in `program` must still evaluate its
    /// all-`Float32` output cone — reproduces the onnx-model failure this
    /// exemption exists for, first as `reject_non_float32` directly (the
    /// root cause), then through the real `evaluate_named` entry point (the
    /// user-visible symptom).
    #[test]
    fn reject_non_float32_exempts_an_unreferenced_non_float32_input() {
        let mut program = Vec::new();
        let _dead_shape_leaf = append(
            &mut program,
            Op::Input {
                dtype: DType::Int64,
                shape: vec![Extent::Static(2)],
                name: Some(String::from("reshape_shape")),
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(4)],
                name: Some(String::from("activation")),
            },
        );

        assert!(
            reject_non_float32(&program, &BTreeSet::new()).is_ok(),
            "an Int64 Input that nothing in the program reads can never reach the f32-only \
             kernels, so it must not fail the gate"
        );

        // `reshape_shape`'s two Int64 elements are never read as f32 data —
        // `evaluate_named` only needs *a* binding for every `Op::Input`, not
        // one whose bit pattern is meaningful, since the dead leaf's buffer
        // is never handed to a kernel.
        let evaluated = evaluate_named(
            &program,
            &[],
            &[("reshape_shape", &[0.0, 0.0]), ("activation", &[1.0, 2.0, 3.0, 4.0])],
            &[activation],
        )
        .expect("an all-f32 output cone must evaluate even with a dead non-f32 leaf present");
        let (data, _shape) = evaluated.get(activation).expect("activation must be a resolved output");
        assert_eq!(data, [1.0, 2.0, 3.0, 4.0].as_slice());
    }

    /// The other half of the same contract: a non-`Float32` `Op::Input` that
    /// IS referenced (here, added into an otherwise-`Float32` elementwise
    /// chain) still reaches `run_node_into`'s f32-only kernels regardless of
    /// output reachability — `BoundOpBuilder::finish`'s own doc is explicit
    /// that a held elementwise op materializes "either [as] a requested
    /// output or dead code" — so it must still be rejected, proving the
    /// dead-leaf exemption did not widen into "any Input is exempt."
    #[test]
    fn reject_non_float32_still_rejects_a_referenced_non_float32_input() {
        let mut program = Vec::new();
        let stray_int_leaf = block(&mut program, DType::Int64, &[Extent::Static(4)]);
        let activation = f32_block(&mut program, &[Extent::Static(4)]);
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: vec![
                    (stray_int_leaf, IndexMap::Affine(map::projection(1, &[0]))),
                    (activation, IndexMap::Affine(map::projection(1, &[0]))),
                ],
                name: None,
            },
        );

        assert!(
            reject_non_float32(&program, &BTreeSet::new()).is_err(),
            "an Int64 Input consumed by a live Elementwise operand still reaches the \
             f32-only kernels and must stay rejected"
        );
    }

    /// A two-named-input, one-`Elementwise`-node program (`c = a + b`), the
    /// smallest fixture that exercises both an [`Op::Input`] rebinding slot
    /// AND a resolved-node buffer -- no MNIST dependency, per this row's
    /// own task instruction (`docs/discipline.md` ROW 165: `build_static_arena`/
    /// `evaluate_named_with_arena` unit-tested on a small synthetic
    /// program).
    fn named_add_program() -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let a = append(
            &mut program,
            Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(4)], name: Some(String::from("a")) },
        );
        let b = append(
            &mut program,
            Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(4)], name: Some(String::from("b")) },
        );
        let sum = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: vec![
                    (a, IndexMap::Affine(map::projection(1, &[0]))),
                    (b, IndexMap::Affine(map::projection(1, &[0]))),
                ],
                name: None,
            },
        );
        (program, sum)
    }

    /// [`build_static_arena`] + [`evaluate_named_with_arena`] run TWICE
    /// against the same arena, on two different `named` bindings, and must
    /// each match [`evaluate_named`]'s own fresh-alloc result bit for bit --
    /// the same correctness bar `train_step_lane.rs`'s own bench-level
    /// `assert_arena_bit_identical_to_baseline` holds the arena to, proved
    /// here at the library-unit level instead.
    #[test]
    fn evaluate_named_with_arena_matches_evaluate_named_over_two_calls() {
        let (program, sum) = named_add_program();
        let mut arena = build_static_arena(&program, &[], &[sum]).expect("small program builds a static arena");

        let first_a = [1.0f32, 2.0, 3.0, 4.0];
        let first_b = [10.0f32, 20.0, 30.0, 40.0];
        let arena_first = evaluate_named_with_arena(&mut arena, &[("a", &first_a), ("b", &first_b)]).expect("first arena call evaluates");
        let baseline_first =
            evaluate_named(&program, &[], &[("a", &first_a), ("b", &first_b)], &[sum]).expect("first baseline call evaluates");
        assert_eq!(
            arena_first.get(sum).map(|(data, _)| data.to_vec()),
            baseline_first.get(sum).map(|(data, _)| data.to_vec()),
            "first call: arena and fresh-alloc paths must agree bit for bit"
        );

        let second_a = [100.0f32, 200.0, 300.0, 400.0];
        let second_b = [1.0f32, 2.0, 3.0, 4.0];
        let arena_second = evaluate_named_with_arena(&mut arena, &[("a", &second_a), ("b", &second_b)]).expect("second arena call evaluates");
        let baseline_second =
            evaluate_named(&program, &[], &[("a", &second_a), ("b", &second_b)], &[sum]).expect("second baseline call evaluates");
        assert_eq!(
            arena_second.get(sum).map(|(data, _)| data.to_vec()),
            baseline_second.get(sum).map(|(data, _)| data.to_vec()),
            "second call (same arena, reused buffers): arena and fresh-alloc paths must still agree bit for bit"
        );
        assert_eq!(arena_second.get(sum).map(|(data, _)| data.to_vec()), Some(vec![101.0, 202.0, 303.0, 404.0]));
    }

    /// A two-input program with a genuinely dead node -- `dead = a * b`,
    /// consumed by nothing, never a requested output -- alongside the
    /// requested `live = a + b`. The smallest fixture for `docs/discipline.md`
    /// ROW 167's execution-level elision: `dead` still gets `bind::bind`'s
    /// own `BoundOp` (bind-time construction is untouched by design), it
    /// just never runs.
    fn dead_node_program() -> (Vec<Op>, NodeId, NodeId) {
        let mut program = Vec::new();
        let a = append(
            &mut program,
            Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(4)], name: Some(String::from("a")) },
        );
        let b = append(
            &mut program,
            Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(4)], name: Some(String::from("b")) },
        );
        let dead = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (a, IndexMap::Affine(map::projection(1, &[0]))),
                    (b, IndexMap::Affine(map::projection(1, &[0]))),
                ],
                name: None,
            },
        );
        let live = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: vec![
                    (a, IndexMap::Affine(map::projection(1, &[0]))),
                    (b, IndexMap::Affine(map::projection(1, &[0]))),
                ],
                name: None,
            },
        );
        (program, dead, live)
    }

    /// [`build_static_arena`] finds `dead` (zero consumers, not requested)
    /// dead and elides it -- `run_resolved_nodes_in_arena` never runs it --
    /// while `live` still evaluates bit-identically to
    /// [`evaluate_named`]'s own fresh-alloc, non-eliding path. `dead`'s
    /// pre-sized buffer stays whatever [`build_static_arena`] initialized
    /// it to (zeros), proving the skip is real rather than a coincidence of
    /// the two ops sharing a body shape.
    #[test]
    fn build_static_arena_elides_a_node_with_zero_consumers() {
        let (program, dead, live) = dead_node_program();
        let mut arena = build_static_arena(&program, &[], &[live]).expect("dead-node program builds a static arena");
        assert!(arena.dead.contains(&dead), "the zero-consumer, non-output node must be marked dead");
        assert!(!arena.dead.contains(&live), "the requested output must never be marked dead");

        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0, 30.0, 40.0];
        let elided = evaluate_named_with_arena(&mut arena, &[("a", &a), ("b", &b)]).expect("elided arena call evaluates");
        let baseline = evaluate_named(&program, &[], &[("a", &a), ("b", &b)], &[live]).expect("baseline call evaluates");
        assert_eq!(
            elided.get(live).map(|(data, _)| data.to_vec()),
            baseline.get(live).map(|(data, _)| data.to_vec()),
            "the live output must be bit-identical whether or not the dead sibling actually executed"
        );
        assert_eq!(arena_output(&arena, dead), Some([0.0f32, 0.0, 0.0, 0.0].as_slice()), "the elided node's buffer must stay untouched -- run_resolved_nodes_in_arena skipped writing it");
    }

    /// The other half of the same contract: naming `dead` itself as a
    /// requested output un-elides it -- `effective_outputs` membership is
    /// what `dead_resolved_nodes` checks, so a caller who genuinely wants
    /// that value back still gets it computed.
    #[test]
    fn build_static_arena_does_not_elide_a_dead_node_that_is_also_a_requested_output() {
        let (program, dead, live) = dead_node_program();
        let mut arena = build_static_arena(&program, &[], &[dead, live]).expect("dead-node program builds a static arena with dead requested");
        assert!(!arena.dead.contains(&dead), "requesting the otherwise-dead node as an output must un-elide it");

        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0, 30.0, 40.0];
        let evaluated = evaluate_named_with_arena(&mut arena, &[("a", &a), ("b", &b)]).expect("arena call evaluates");
        let baseline = evaluate_named(&program, &[], &[("a", &a), ("b", &b)], &[dead, live]).expect("baseline call evaluates");
        assert_eq!(
            evaluated.get(dead).map(|(data, _)| data.to_vec()),
            baseline.get(dead).map(|(data, _)| data.to_vec()),
            "the now-requested node must actually compute, bit-identical to the non-eliding baseline"
        );
        assert_eq!(evaluated.get(dead).map(|(data, _)| data.to_vec()), Some(vec![10.0, 40.0, 90.0, 160.0]));
    }

    /// A one-input program with a `Constant` feeding a live `Add` --
    /// `docs/discipline.md` ROW 174's own found lever: `c`'s value is baked
    /// into its `BoundOp` at `bind::bind` time and never depends on `a`.
    fn constant_feeds_live_program() -> (Vec<Op>, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let a = append(
            &mut program,
            Op::Input { dtype: DType::Float32, shape: vec![Extent::Static(4)], name: Some(String::from("a")) },
        );
        let constant = append(
            &mut program,
            Op::Constant { dtype: DType::Float32, shape: vec![Extent::Static(4)], value: 5.0 },
        );
        let live = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: vec![
                    (a, IndexMap::Affine(map::projection(1, &[0]))),
                    (constant, IndexMap::Affine(map::projection(1, &[0]))),
                ],
                name: None,
            },
        );
        (program, a, constant, live)
    }

    /// [`build_static_arena`] marks a live `Constant` (`bind::bind`'s own
    /// `BoundOpKind::Constant` -- no operands, no dependence on `named` at
    /// all) as one of its `static_nodes` and runs it exactly once, at build
    /// time -- never again on any subsequent [`evaluate_named_with_arena`]
    /// call, proven here by corrupting the constant's own resident buffer
    /// between steps: if `run_resolved_nodes_in_arena` still executed it,
    /// `run_constant`'s `output.fill(value)` would overwrite the corruption
    /// back to the literal on the very next step. It does not.
    #[test]
    fn build_static_arena_runs_a_live_constant_once_and_never_again() {
        let (program, _a, constant, live) = constant_feeds_live_program();
        let mut arena = build_static_arena(&program, &[], &[live]).expect("constant-feeds-live program builds a static arena");
        assert!(arena.static_nodes.contains(&constant), "a live Constant-kind node must be marked static");
        assert!(!arena.dead.contains(&constant), "a consumed Constant must never also be marked dead");

        let step_one = [1.0f32, 2.0, 3.0, 4.0];
        let evaluated = evaluate_named_with_arena(&mut arena, &[("a", &step_one)]).expect("step one evaluates");
        assert_eq!(evaluated.get(live).map(|(data, _)| data.to_vec()), Some(vec![6.0, 7.0, 8.0, 9.0]), "a + the constant's own literal 5.0, computed once at build time");
        assert_eq!(arena_output(&arena, constant), Some([5.0f32; 4].as_slice()), "the constant's own buffer holds its literal after step one");

        arena.buffers[constant.0 as usize] = Some(alloc::vec![999.0f32; 4]);

        let step_two = [10.0f32, 20.0, 30.0, 40.0];
        let evaluated = evaluate_named_with_arena(&mut arena, &[("a", &step_two)]).expect("step two evaluates");
        assert_eq!(
            evaluated.get(live).map(|(data, _)| data.to_vec()),
            Some(vec![1009.0, 1019.0, 1029.0, 1039.0]),
            "step two must fold the CORRUPTED buffer, not the literal -- proving run_resolved_nodes_in_arena truly never re-executed the constant"
        );
        assert_eq!(arena_output(&arena, constant), Some([999.0f32; 4].as_slice()), "the corruption survives step two untouched");

        let step_three = [0.0f32, 0.0, 0.0, 0.0];
        let evaluated = evaluate_named_with_arena(&mut arena, &[("a", &step_three)]).expect("step three evaluates");
        assert_eq!(
            evaluated.get(live).map(|(data, _)| data.to_vec()),
            Some(vec![999.0, 999.0, 999.0, 999.0]),
            "a third arena step, still folding the same corrupted, never-recomputed buffer"
        );
    }

    /// The other half of ROW 174's same contract, mirroring
    /// [`build_static_arena_does_not_elide_a_dead_node_that_is_also_a_requested_output`]:
    /// a `Constant` with zero consumers lands in `dead`, not `static_nodes`
    /// -- `static_resolved_nodes` excludes anything `dead_resolved_nodes`
    /// already marked, so a dead constant is never even run the one time
    /// `static_nodes` would otherwise cost.
    #[test]
    fn a_dead_constant_is_marked_dead_not_static() {
        let (mut program, _a, constant, live) = constant_feeds_live_program();
        let dead_constant = append(
            &mut program,
            Op::Constant { dtype: DType::Float32, shape: vec![Extent::Static(4)], value: 42.0 },
        );

        let arena = build_static_arena(&program, &[], &[live]).expect("program with an unused constant builds a static arena");
        assert!(arena.dead.contains(&dead_constant), "a zero-consumer Constant must be marked dead");
        assert!(!arena.static_nodes.contains(&dead_constant), "a dead Constant must not also be marked static -- dead already skips it");
        assert!(arena.static_nodes.contains(&constant), "the live constant is unaffected by its dead sibling");
    }

    /// A `named` binding whose length no longer matches the shape
    /// [`build_static_arena`] fixed for that input must return a named
    /// [`TensorError::InputSizeMismatch`], not a silent truncation, a
    /// panic, or a wrong-shaped result.
    #[test]
    fn evaluate_named_with_arena_reports_a_shape_mismatched_rebind_by_name() {
        let (program, sum) = named_add_program();
        let mut arena = build_static_arena(&program, &[], &[sum]).expect("small program builds a static arena");

        let wrong_length_a = [1.0f32, 2.0, 3.0];
        let full_length_b = [10.0f32, 20.0, 30.0, 40.0];
        let error = evaluate_named_with_arena(&mut arena, &[("a", &wrong_length_a), ("b", &full_length_b)])
            .expect_err("a 3-element rebind against a 4-element input slot must be rejected");
        match error {
            TensorError::InputSizeMismatch { expected, found, .. } => {
                assert_eq!(expected, 4, "the arena's own fixed slot size for `a`");
                assert_eq!(found, 3, "the mismatched rebind's own length");
            }
            other => panic!("expected TensorError::InputSizeMismatch, got {other:?}"),
        }
    }

    /// `is_quantized_matmul_operand` is called once per candidate node, not
    /// once per program — proving the exemption holds for many quantized
    /// weights at once (a real checkpoint's 217 `Q4_K` tensors, not the
    /// single-weight case the earlier tests above cover), and that each
    /// node's own shape is judged independently: a second, unrelated
    /// `Multiply`-then-`Add` matmul with its own `UInt8` weight is exempted
    /// alongside the first when both are named, and rejected on its own
    /// when only the first is named.
    #[test]
    fn reject_non_float32_exempts_many_independent_quantized_weights() {
        let mut program = Vec::new();
        let weight_a = block(&mut program, DType::UInt8, &[Extent::Static(4)]);
        let activation_a = f32_block(&mut program, &[Extent::Static(4)]);
        let product_a = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight_a, IndexMap::Affine(map::projection(1, &[0]))),
                    (activation_a, IndexMap::Affine(map::projection(1, &[0]))),
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
                operand: product_a,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let weight_b = block(&mut program, DType::UInt8, &[Extent::Static(4)]);
        let activation_b = f32_block(&mut program, &[Extent::Static(4)]);
        let product_b = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight_b, IndexMap::Affine(map::projection(1, &[0]))),
                    (activation_b, IndexMap::Affine(map::projection(1, &[0]))),
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
                operand: product_b,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let mut only_a = BTreeSet::new();
        only_a.insert(weight_a);
        assert!(
            reject_non_float32(&program, &only_a).is_err(),
            "weight_b is UInt8 and unexempted, so the program must still be rejected"
        );

        let mut both = BTreeSet::new();
        both.insert(weight_a);
        both.insert(weight_b);
        assert!(
            reject_non_float32(&program, &both).is_ok(),
            "two independent matmul-shaped quantized weights must both be exempted when both are named"
        );
    }

    #[test]
    fn matmul_q4k_f32_rejects_a_row_length_not_a_block_multiple() {
        let weights = vec![0u8; Q4K_BLOCK_BYTES + 1];
        let activation = vec![0.0f32; Q4K_BLOCK_ELEMENTS];
        let error = matmul_q4k_f32(&weights, 1, &activation).unwrap_err();
        assert_eq!(
            error,
            TensorError::QuantizedShapeMismatch {
                reason: "weight row length is not a whole multiple of the q4_k block size",
            }
        );
    }

    #[test]
    fn matmul_q4k_f32_rejects_an_activation_length_mismatch() {
        let weights = vec![0u8; Q4K_BLOCK_BYTES];
        let activation = vec![0.0f32; Q4K_BLOCK_ELEMENTS - 1];
        let error = matmul_q4k_f32(&weights, 1, &activation).unwrap_err();
        assert_eq!(
            error,
            TensorError::QuantizedShapeMismatch {
                reason: "activation length does not match the weight row's decoded element count",
            }
        );
    }

    fn block(program: &mut Vec<Op>, dtype: DType, shape: &[Extent]) -> NodeId {
        append(
            program,
            Op::Input {
                dtype,
                shape: shape.to_vec(),
                name: None,
            },
        )
    }

    fn f32_block(program: &mut Vec<Op>, shape: &[Extent]) -> NodeId {
        block(program, DType::Float32, shape)
    }

    /// `run_iota`'s whole contract: `output[i] = i`, evaluated through the
    /// real `evaluate` entry point (not the internal `run_node_into` alone),
    /// proving the leaf materializes with no external `blocks` entry — the
    /// same guarantee `causal_attention.toml`'s `query_index`/`key_index`
    /// nodes rely on.
    #[test]
    fn an_iota_evaluates_to_its_own_position() {
        let mut program = Vec::new();
        let iota = append(
            &mut program,
            Op::Iota {
                dtype: DType::Float32,
                extent: Extent::Static(6),
            },
        );

        let evaluated = evaluate(&program, &[], &[], &[]).expect("a bare iota evaluates");
        assert_eq!(evaluated.root(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let _ = iota;
    }

    /// The counterpart of [`an_iota_evaluates_to_its_own_position`] for the
    /// other computed leaf: a `Constant` materializes with no external
    /// `blocks` entry, and every element is the literal it was built with.
    /// Run through the real `evaluate` entry point so the whole
    /// bind/schedule path is exercised, not `run_constant` alone.
    #[test]
    fn a_constant_evaluates_to_its_literal_at_every_position() {
        let mut program = Vec::new();
        append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                value: 0.088_388_35,
            },
        );

        let evaluated = evaluate(&program, &[], &[], &[]).expect("a bare constant evaluates");
        assert_eq!(evaluated.root(), &[0.088_388_35; 4]);
    }

    /// A rank-0 `Constant` is the shape every scalar literal in
    /// `spec.rs` uses: one element, and an empty operand side (`"->i"`)
    /// broadcasts it across any consumer's iteration space.
    #[test]
    fn a_rank_zero_constant_broadcasts_into_a_higher_rank_consumer() {
        let mut program = Vec::new();
        let scale = append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: Vec::new(),
                value: 3.0,
            },
        );
        let iota = append(
            &mut program,
            Op::Iota {
                dtype: DType::Float32,
                extent: Extent::Static(4),
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (iota, IndexMap::Affine(crate::map::projection(1, &[0]))),
                    (scale, IndexMap::Affine(crate::map::projection(1, &[]))),
                ],
                name: None,
            },
        );

        let evaluated = evaluate(&program, &[], &[], &[]).expect("rank-0 constant broadcasts");
        assert_eq!(evaluated.root(), &[0.0, 3.0, 6.0, 9.0]);
    }

    /// `-inf` is the literal the causal mask needs and the one an integer
    /// `Iota` derivation reached only through `Reciprocal(-0.0)`. It must
    /// survive the leaf verbatim.
    #[test]
    fn a_constant_carries_negative_infinity_verbatim() {
        let mut program = Vec::new();
        append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(2)],
                value: f32::NEG_INFINITY,
            },
        );

        let evaluated = evaluate(&program, &[], &[], &[]).expect("a -inf constant evaluates");
        assert!(evaluated.root().iter().all(|value| *value == f32::NEG_INFINITY));
    }

    /// `reduce_dot_binary_monomorphic`'s `(true, true)` arm reassociates the
    /// sum (`DOT_LANES` independent partial accumulators, ROW 12,
    /// `proxima-tensor/docs/discipline.md`) — bit-exactness against
    /// [`naive_matmul`]'s strict left-to-right fold is no longer the bar for
    /// the transposed-RHS (reduce_dot) path, same as Accelerate/OpenBLAS/
    /// ggml. Returns the measured max relative error so callers can log it.
    fn assert_all_close(actual: &[f32], expected: &[f32], relative_tolerance: f32) -> f32 {
        assert_eq!(actual.len(), expected.len());
        let mut max_relative_error = 0.0f32;
        for (&value, &reference) in actual.iter().zip(expected) {
            let scale = reference.abs().max(1.0);
            let relative_error = (value - reference).abs() / scale;
            max_relative_error = max_relative_error.max(relative_error);
            assert!(
                relative_error <= relative_tolerance,
                "relative error {relative_error} exceeds tolerance {relative_tolerance} \
                 (actual={value}, expected={reference})"
            );
        }
        max_relative_error
    }

    fn naive_matmul(lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for inner in 0..k {
                    sum += lhs[row * k + inner] * rhs[inner * n + col];
                }
                out[row * n + col] = sum;
            }
        }
        out
    }

    fn matmul_program(m: u32, k: u32, n: u32, symbolic: bool) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let lhs_shape = if symbolic {
            alloc::vec![Extent::Symbolic(0), Extent::Static(k)]
        } else {
            alloc::vec![Extent::Static(m), Extent::Static(k)]
        };
        let lhs = f32_block(&mut program, &lhs_shape);
        let rhs = f32_block(&mut program, &[Extent::Static(k), Extent::Static(n)]);
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
                ],
                name: None,
            },
        );
        let sum = append(
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
        (program, sum)
    }

    /// Same contraction as [`matmul_program`], RHS stored `[n, k]` instead
    /// of `[k, n]` (ggml's own `mul_mat` convention) — exercises
    /// [`run_reduce`]'s reduction-dim fast path
    /// (`proxima-tensor/docs/discipline.md` ROW 10/11): the width dim `n` is not
    /// contiguous on the RHS operand here, but the contraction dim `k` is.
    fn matmul_program_rhs_transposed(m: u32, k: u32, n: u32) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let lhs = f32_block(&mut program, &[Extent::Static(m), Extent::Static(k)]);
        let rhs = f32_block(&mut program, &[Extent::Static(n), Extent::Static(k)]);
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (rhs, IndexMap::Affine(map::projection(3, &[1, 2]))),
                ],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: Some("matmul_rhs_transposed".into()),
            }),
        );
        (program, sum)
    }

    /// `table[ids[s], d]` over iteration space `(s, d)`: dim 0 (vocab) is
    /// gathered by `ids`, dim 1 (feature) is a plain projection.
    fn embedding_lookup_program(vocab: u32, dim: u32, seq: u32) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let table = f32_block(&mut program, &[Extent::Static(vocab), Extent::Static(dim)]);
        let ids = block(&mut program, DType::Int32, &[Extent::Static(seq)]);
        let gathered_map = IndexMap::Computed {
            indices: ids,
            index_map: map::projection(2, &[0]),
            base: map::IndexPattern {
                iter_rank: 2,
                axes: alloc::vec![
                    map::AxisIndex::default(),
                    map::AxisIndex {
                        terms: core::iter::once(AxisTerm::projection(1)).collect(),
                        offset: 0,
                    },
                ],
            },
            gathered_dim: 0,
        };
        let gathered = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(table, gathered_map)],
                name: None,
            },
        );
        (program, gathered)
    }

    /// `sum_k table[ids[i], k] * weight[k, j]` — an embedding lookup fused
    /// straight into a contraction, mirroring [`matmul_program`] with `lhs`
    /// replaced by a gather.
    fn embedding_matmul_program(vocab: u32, embed_dim: u32, seq: u32, out_dim: u32) -> Vec<Op> {
        let mut program = Vec::new();
        let table = f32_block(
            &mut program,
            &[Extent::Static(vocab), Extent::Static(embed_dim)],
        );
        let ids = block(&mut program, DType::Int32, &[Extent::Static(seq)]);
        let weight = f32_block(
            &mut program,
            &[Extent::Static(embed_dim), Extent::Static(out_dim)],
        );

        let gather_map = IndexMap::Computed {
            indices: ids,
            index_map: map::projection(3, &[0]),
            base: map::IndexPattern {
                iter_rank: 3,
                axes: alloc::vec![
                    map::AxisIndex::default(),
                    map::AxisIndex {
                        terms: core::iter::once(AxisTerm::projection(2)).collect(),
                        offset: 0,
                    },
                ],
            },
            gathered_dim: 0,
        };
        let weight_map = IndexMap::Affine(map::projection(3, &[2, 1]));

        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(table, gather_map), (weight, weight_map)],
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
                name: Some("embedding_matmul".into()),
            }),
        );
        program
    }

    #[test]
    fn embedding_lookup_matches_a_hand_written_reference() {
        let (vocab, dim, seq) = (50_000usize, 8usize, 4usize);
        let (program, gathered) = embedding_lookup_program(vocab as u32, dim as u32, seq as u32);

        let table_data: Vec<f32> = (0..vocab * dim).map(|value| (value % 97) as f32).collect();
        let ids_data = [3.0f32, 49_999.0, 12_345.0, 0.0];
        let evaluated = evaluate(&program, &[], &[&table_data, &ids_data], &[])
            .expect("embedding lookup evaluates");

        let mut reference = vec![0.0f32; seq * dim];
        for (row, &id) in ids_data.iter().enumerate() {
            let vocab_index = id as usize;
            reference[row * dim..(row + 1) * dim]
                .copy_from_slice(&table_data[vocab_index * dim..(vocab_index + 1) * dim]);
        }
        assert_eq!(evaluated.shape(), &[seq as u64, dim as u64]);
        assert_eq!(evaluated.root(), reference.as_slice());
        let _ = gathered;
    }

    #[test]
    fn a_fetched_index_past_the_extent_is_a_real_error_not_ub() {
        let (program, _gathered) = embedding_lookup_program(4, 2, 1);
        let table_data: Vec<f32> = (0..8).map(|value| value as f32).collect();
        let ids_data = [4.0f32]; // extent is 4: 0..=3 are valid, 4 is not
        let error = evaluate(&program, &[], &[&table_data, &ids_data], &[])
            .expect_err("out-of-range fetched index is rejected");
        assert!(
            matches!(error, TensorError::GatherIndexOutOfRange { .. }),
            "{error}"
        );
    }

    #[test]
    fn a_gather_fused_into_a_fold_matches_a_hand_written_embedding_matmul_reference() {
        let (vocab, embed_dim, seq, out_dim) = (100usize, 6usize, 4usize, 3usize);
        let program =
            embedding_matmul_program(vocab as u32, embed_dim as u32, seq as u32, out_dim as u32);

        let shapes = shape::infer(&program, &[]).expect("embedding matmul infers");
        let resolved = bind::bind(&program, &shapes, &[]).expect("embedding matmul resolves");
        assert_eq!(
            resolved.len(),
            1,
            "the gather zip must fuse into the fold, not materialize separately"
        );
        assert!(matches!(resolved[0].kind, BoundOpKind::Reduce { .. }));

        let table_data: Vec<f32> = (0..vocab * embed_dim)
            .map(|value| (value % 13) as f32)
            .collect();
        let ids_data = [3.0f32, 99.0, 50.0, 0.0];
        let weight_data: Vec<f32> = (0..embed_dim * out_dim)
            .map(|value| (value % 5) as f32)
            .collect();

        let evaluated = evaluate(&program, &[], &[&table_data, &ids_data, &weight_data], &[])
            .expect("embedding matmul evaluates");

        let mut reference = vec![0.0f32; seq * out_dim];
        for (row, &id) in ids_data.iter().enumerate() {
            let vocab_index = id as usize;
            for col in 0..out_dim {
                let mut total = 0.0f32;
                for k in 0..embed_dim {
                    total +=
                        table_data[vocab_index * embed_dim + k] * weight_data[k * out_dim + col];
                }
                reference[row * out_dim + col] = total;
            }
        }
        assert_eq!(evaluated.root(), reference.as_slice());
    }

    #[proxima::test]
    #[case::one_worker(1)]
    #[case::two_workers(2)]
    #[case::three_workers(3)]
    async fn evaluate_parallel_matches_evaluate_for_a_gather_program(#[case] workers: usize) {
        let (vocab, embed_dim, seq, out_dim) = (100usize, 6usize, 4usize, 3usize);
        let program =
            embedding_matmul_program(vocab as u32, embed_dim as u32, seq as u32, out_dim as u32);
        let table_data: Vec<f32> = (0..vocab * embed_dim)
            .map(|value| (value % 13) as f32)
            .collect();
        let ids_data = [3.0f32, 99.0, 50.0, 0.0];
        let weight_data: Vec<f32> = (0..embed_dim * out_dim)
            .map(|value| (value % 5) as f32)
            .collect();

        assert_parallel_matches_sequential(
            &program,
            &[],
            &[&table_data, &ids_data, &weight_data],
            &[],
            workers,
        );
    }

    #[test]
    fn a_gather_program_past_the_parallel_threshold_actually_splits_and_still_matches_sequential() {
        let (vocab, embed_dim, seq, out_dim) = (200usize, 64usize, 128usize, 64usize);
        let program =
            embedding_matmul_program(vocab as u32, embed_dim as u32, seq as u32, out_dim as u32);
        let table_data: Vec<f32> = (0..vocab * embed_dim)
            .map(|value| (value % 13) as f32)
            .collect();
        let ids_data: Vec<f32> = (0..seq).map(|value| (value % vocab) as f32).collect();
        let weight_data: Vec<f32> = (0..embed_dim * out_dim)
            .map(|value| (value % 5) as f32)
            .collect();

        let shapes = shape::infer(&program, &[]).expect("infers");
        let resolved = bind::bind(&program, &shapes, &[]).expect("resolves");
        assert_eq!(resolved.len(), 1, "fused into one reduction node");
        assert!(
            element_count(&resolved[0].extents) >= PARALLEL_THRESHOLD,
            "this size must clear the threshold or this test proves nothing about the \
             threaded path"
        );
        assert!(
            resolved[0].split(4).is_some(),
            "the node must actually be splittable for the threaded path to run"
        );

        let workers = NonZeroUsize::new(4).expect("4 is nonzero");
        assert_parallel_matches_sequential(
            &program,
            &[],
            &[&table_data, &ids_data, &weight_data],
            &[],
            workers.get(),
        );
    }

    #[test]
    fn fused_matmul_matches_a_naive_triple_loop() {
        let (m, k, n) = (4usize, 3usize, 5usize);
        let (program, sum) = matmul_program(m as u32, k as u32, n as u32, false);
        let lhs: Vec<f32> = (0..m * k).map(|value| value as f32).collect();
        let rhs: Vec<f32> = (0..k * n).map(|value| value as f32).collect();

        let evaluated = evaluate(&program, &[], &[&lhs, &rhs], &[]).expect("matmul evaluates");
        assert_eq!(evaluated.shape(), &[m as u64, n as u64]);
        assert_eq!(
            evaluated.root(),
            naive_matmul(&lhs, &rhs, m, k, n).as_slice()
        );
        let _ = sum;
    }

    #[test]
    fn fused_matmul_with_transposed_rhs_matches_a_naive_triple_loop() {
        // k=7 (not a multiple of DOT_LANES=4) exercises the fast path's
        // remainder handling; the RHS buffer is the same numbers as
        // `naive_matmul`'s `[k, n]` reference expects, laid out `[n, k]`.
        //
        // WEAKENED (ROW 12, `proxima-tensor/docs/discipline.md`): was
        // `assert_eq!` (bit-exact) against `naive_matmul`'s strict
        // left-to-right fold. `reduce_dot_binary_monomorphic`'s `(true,
        // true)` arm now folds via `DOT_LANES` independent partial
        // accumulators (matches Accelerate/OpenBLAS/ggml practice), which
        // reassociates the sum and can change its bit pattern relative to
        // the naive loop. Switched to a 1e-5 relative-tolerance check.
        // Measured max relative error at this k=7, small-integer-input size
        // was 0.0 (integers this small sum exactly in f32 regardless of
        // grouping) — logged in ROW 12 rather than assumed.
        let (m, k, n) = (4usize, 7usize, 5usize);
        let (program, sum) = matmul_program_rhs_transposed(m as u32, k as u32, n as u32);
        let lhs: Vec<f32> = (0..m * k).map(|value| value as f32).collect();
        let rhs_kn: Vec<f32> = (0..k * n).map(|value| value as f32).collect();
        let mut rhs_nk = vec![0.0f32; k * n];
        for row in 0..k {
            for col in 0..n {
                rhs_nk[col * k + row] = rhs_kn[row * n + col];
            }
        }

        let evaluated = evaluate(&program, &[], &[&lhs, &rhs_nk], &[]).expect("matmul evaluates");
        assert_eq!(evaluated.shape(), &[m as u64, n as u64]);
        let reference = naive_matmul(&lhs, &rhs_kn, m, k, n);
        let max_relative_error = assert_all_close(evaluated.root(), &reference, 1e-5);
        println!("k=7 transposed-rhs max_relative_error={max_relative_error}");
        let _ = sum;
    }

    #[test]
    fn fused_matmul_with_transposed_rhs_k1024_within_tolerance_of_a_naive_triple_loop() {
        // k=1024 matches the real GEMM benchmark's contraction length and
        // uses fractional, non-integer data so the sum actually accumulates
        // rounding error under either fold order (unlike the k=7 test's
        // small-integer inputs, whose sums are exact regardless of
        // grouping) — see ROW 12, `proxima-tensor/docs/discipline.md`.
        let (m, k, n) = (8usize, 1024usize, 8usize);
        let (program, _sum) = matmul_program_rhs_transposed(m as u32, k as u32, n as u32);
        let lhs: Vec<f32> = (0..m * k).map(|value| (value as f32 * 0.0137).sin()).collect();
        let rhs_kn: Vec<f32> = (0..k * n).map(|value| (value as f32 * 0.0271).cos()).collect();
        let mut rhs_nk = vec![0.0f32; k * n];
        for row in 0..k {
            for col in 0..n {
                rhs_nk[col * k + row] = rhs_kn[row * n + col];
            }
        }

        let evaluated = evaluate(&program, &[], &[&lhs, &rhs_nk], &[]).expect("matmul evaluates");
        assert_eq!(evaluated.shape(), &[m as u64, n as u64]);
        let reference = naive_matmul(&lhs, &rhs_kn, m, k, n);
        let max_relative_error = assert_all_close(evaluated.root(), &reference, 1e-5);
        println!("k=1024 transposed-rhs max_relative_error={max_relative_error}");
    }

    #[test]
    fn matmul_binds_a_symbolic_sequence_length_at_eval_time() {
        let (m, k, n) = (4usize, 3usize, 5usize);
        let (program, _sum) = matmul_program(m as u32, k as u32, n as u32, true);
        let lhs: Vec<f32> = (0..m * k).map(|value| value as f32).collect();
        let rhs: Vec<f32> = (0..k * n).map(|value| value as f32).collect();

        let evaluated =
            evaluate(&program, &[m as u64], &[&lhs, &rhs], &[]).expect("symbolic matmul evaluates");
        assert_eq!(
            evaluated.root(),
            naive_matmul(&lhs, &rhs, m, k, n).as_slice()
        );
    }

    #[test]
    fn fused_contraction_skips_the_product_tensor() {
        let (m, k, n) = (64usize, 64usize, 64usize);
        let (program, _sum) = matmul_program(m as u32, k as u32, n as u32, false);
        let lhs: Vec<f32> = (0..m * k).map(|value| (value % 7) as f32).collect();
        let rhs: Vec<f32> = (0..k * n).map(|value| (value % 5) as f32).collect();

        let evaluated =
            evaluate(&program, &[], &[&lhs, &rhs], &[]).expect("64x64x64 matmul evaluates");
        let reference = naive_matmul(&lhs, &rhs, m, k, n);
        for (row, col) in [(0, 0), (0, n - 1), (m - 1, 0), (m - 1, n - 1)] {
            let index = row * n + col;
            assert_eq!(
                evaluated.root()[index],
                reference[index],
                "corner ({row}, {col})"
            );
        }
    }

    #[test]
    fn bias_add_via_broadcast() {
        let mut program = Vec::new();
        let matrix = f32_block(&mut program, &[Extent::Static(2), Extent::Static(3)]);
        let bias = f32_block(&mut program, &[Extent::Static(3)]);
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![
                    (matrix, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (bias, IndexMap::Affine(map::projection(2, &[1]))),
                ],
                name: None,
            },
        );

        let matrix_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let bias_data = [10.0, 20.0, 30.0f32];
        let evaluated =
            evaluate(&program, &[], &[&matrix_data, &bias_data], &[]).expect("bias add evaluates");
        assert_eq!(evaluated.root(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn transpose_via_permuted_map() {
        let mut program = Vec::new();
        let matrix = f32_block(&mut program, &[Extent::Static(2), Extent::Static(3)]);
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(matrix, IndexMap::Affine(map::projection(2, &[1, 0])))],
                name: None,
            },
        );

        let matrix_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let evaluated = evaluate(&program, &[], &[&matrix_data], &[]).expect("transpose evaluates");
        assert_eq!(evaluated.shape(), &[3, 2]);
        assert_eq!(evaluated.root(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn one_dimensional_convolution_via_a_two_term_window_map() {
        // a per-position ("locally connected") kernel: kernel[h, r] pins both
        // iteration dims via pure projection, while signal[h + r] is the
        // two-term windowed access under test.
        let mut program = Vec::new();
        let kernel = f32_block(&mut program, &[Extent::Static(6), Extent::Static(3)]);
        let signal = f32_block(&mut program, &[Extent::Static(8)]);
        let window = IndexMap::Affine(map::affine(
            2,
            &[(&[AxisTerm::scaled(0, 1), AxisTerm::scaled(1, 1)], 0)],
        ));
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (kernel, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (signal, window)
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
                in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                out_map: IndexMap::Affine(map::projection(2, &[0])),
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let kernel_data: Vec<f32> = (0..18).map(|value| value as f32).collect();
        let signal_data: Vec<f32> = (0..8).map(|value| value as f32).collect();
        let evaluated =
            evaluate(&program, &[], &[&kernel_data, &signal_data], &[]).expect("conv evaluates");

        let mut reference = vec![0.0f32; 6];
        for (h, slot) in reference.iter_mut().enumerate() {
            for r in 0..3 {
                *slot += kernel_data[h * 3 + r] * signal_data[h + r];
            }
        }
        assert_eq!(evaluated.root(), reference.as_slice());
    }

    #[test]
    fn softmax_end_to_end_matches_a_reference_within_epsilon() {
        let mut program = Vec::new();
        let (n, d) = (2usize, 4usize);
        let input = f32_block(
            &mut program,
            &[Extent::Static(n as u32), Extent::Static(d as u32)],
        );

        let row_map = IndexMap::Affine(map::projection(2, &[0, 1]));
        let broadcast_map = IndexMap::Affine(map::projection(2, &[0]));

        let max = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Maximum,
                init: ReduceInit::NegativeInfinity,
                operand: input,
                in_map: row_map.clone(),
                out_map: broadcast_map.clone(),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let shifted = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Subtract,
                operands: alloc::vec![(input, row_map.clone()), (max, broadcast_map.clone())],
                name: None,
            },
        );
        let exponentiated = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Exponential,
                operands: alloc::vec![(shifted, row_map.clone())],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: exponentiated,
                in_map: row_map.clone(),
                out_map: broadcast_map.clone(),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Divide,
                operands: alloc::vec![(exponentiated, row_map), (sum, broadcast_map)],
                name: None,
            },
        );

        let input_data = [1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0f32];
        let evaluated = evaluate(&program, &[], &[&input_data], &[]).expect("softmax evaluates");

        let mut reference = vec![0.0f32; n * d];
        for row in 0..n {
            let slice = &input_data[row * d..row * d + d];
            let row_max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = slice.iter().map(|value| (value - row_max).exp()).collect();
            let total: f32 = exps.iter().sum();
            for (col, value) in exps.iter().enumerate() {
                reference[row * d + col] = value / total;
            }
        }

        for (found, expected) in evaluated.root().iter().zip(reference.iter()) {
            assert!((found - expected).abs() < 1e-6, "{found} vs {expected}");
        }
    }

    #[test]
    fn cumsum_matches_a_running_sum_reference() {
        let mut program = Vec::new();
        let source = f32_block(&mut program, &[Extent::Static(6)]);
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[0])),
                keep: Keep::Scan,
                name: None,
            }),
        );

        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        let evaluated = evaluate(&program, &[], &[&data], &[]).expect("cumsum evaluates");

        let mut running = 0.0f32;
        let reference: Vec<f32> = data
            .iter()
            .map(|value| {
                running += value;
                running
            })
            .collect();
        assert_eq!(evaluated.root(), reference.as_slice());
    }

    #[test]
    fn a_chain_of_unary_zips_keeps_peak_live_buffers_small() {
        let mut program = Vec::new();
        let mut current = f32_block(&mut program, &[Extent::Static(4)]);
        for _ in 0..8 {
            current = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Tanh,
                    operands: alloc::vec![(current, IndexMap::Affine(map::projection(1, &[0])))],
                    name: None,
                },
            );
        }

        let input = [0.1, 0.2, 0.3, 0.4f32];
        let evaluated = evaluate(&program, &[], &[&input], &[]).expect("tanh chain evaluates");

        let mut reference = input;
        for value in &mut reference {
            for _ in 0..8 {
                *value = value.tanh();
            }
        }
        for (found, expected) in evaluated.root().iter().zip(reference.iter()) {
            assert!((found - expected).abs() < 1e-6, "{found} vs {expected}");
        }
        let peak = evaluated
            .peak_live_buffers()
            .expect("evaluate tracks peak live buffers");
        assert!(
            peak <= 3,
            "streaming a chain of 8 unary elementwise ops should not hold one buffer per op, got {peak}"
        );
        let _ = current;
    }

    #[test]
    fn a_chain_of_8_unary_ops_binds_to_one_bound_op_and_the_result_is_unchanged() {
        let mut program = Vec::new();
        let mut current = f32_block(&mut program, &[Extent::Static(4)]);
        for _ in 0..8 {
            current = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Tanh,
                    operands: alloc::vec![(current, IndexMap::Affine(map::projection(1, &[0])))],
                    name: None,
                },
            );
        }
        let _ = current;

        let shapes = shape::infer(&program, &[]).expect("tanh chain infers");
        let resolved = bind::bind(&program, &shapes, &[]).expect("tanh chain resolves");
        assert_eq!(
            resolved.len(),
            1,
            "8 chained unary elementwise ops must fuse into one BoundOp"
        );

        let input = [0.1, 0.2, 0.3, 0.4f32];
        let evaluated = evaluate(&program, &[], &[&input], &[]).expect("tanh chain evaluates");
        let mut reference = input;
        for value in &mut reference {
            for _ in 0..8 {
                *value = value.tanh();
            }
        }
        for (found, expected) in evaluated.root().iter().zip(reference.iter()) {
            assert!((found - expected).abs() < 1e-6, "{found} vs {expected}");
        }
    }

    /// `b = a * scale; c = b + bias; d = c * c` — the elementwise-into-
    /// elementwise fusion case, not the reduce-over-elementwise one every
    /// other fusion test in this crate already covers.
    #[test]
    fn a_chain_of_elementwise_ops_binds_to_one_bound_op_and_matches_a_hand_reference() {
        let mut program = Vec::new();
        let a = f32_block(&mut program, &[Extent::Static(4)]);
        let scale = f32_block(&mut program, &[Extent::Static(4)]);
        let bias = f32_block(&mut program, &[Extent::Static(4)]);
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let b = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(a, identity()), (scale, identity())],
                name: None,
            },
        );
        let c = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(b, identity()), (bias, identity())],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(c, identity()), (c, identity())],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("elementwise chain infers");
        let resolved = bind::bind(&program, &shapes, &[]).expect("elementwise chain resolves");
        assert_eq!(
            resolved.len(),
            1,
            "b and c must fuse into d's own BoundOp instead of three separate ones"
        );

        let a_data = [1.0, 2.0, 3.0, 4.0f32];
        let scale_data = [2.0, 0.5, -1.0, 3.0f32];
        let bias_data = [1.0, 1.0, 1.0, 1.0f32];
        let evaluated = evaluate(&program, &[], &[&a_data, &scale_data, &bias_data], &[])
            .expect("elementwise chain evaluates");

        let reference: Vec<f32> = a_data
            .iter()
            .zip(scale_data.iter())
            .zip(bias_data.iter())
            .map(|((a_value, scale_value), bias_value)| {
                let b_value = a_value * scale_value;
                let c_value = b_value + bias_value;
                c_value * c_value
            })
            .collect();
        assert_eq!(evaluated.root(), reference.as_slice());
    }

    /// `b` feeds two different consumers (`c1` and `c2`), so it must
    /// materialize once on its own rather than fuse into either — the
    /// multi-use case that forces materialization even though every map
    /// involved is a plain identity projection.
    #[test]
    fn an_elementwise_intermediate_consumed_by_two_ops_still_evaluates_correctly() {
        let mut program = Vec::new();
        let a = f32_block(&mut program, &[Extent::Static(4)]);
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let b = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: alloc::vec![(a, identity())],
                name: None,
            },
        );
        let c1 = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Negate,
                operands: alloc::vec![(b, identity())],
                name: None,
            },
        );
        let c2 = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Reciprocal,
                operands: alloc::vec![(b, identity())],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(c1, identity()), (c2, identity())],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("diamond chain infers");
        let resolved = bind::bind(&program, &shapes, &[]).expect("diamond chain resolves");
        assert_eq!(
            resolved.len(),
            2,
            "b must materialize standalone since it has two distinct consumers"
        );

        let a_data = [0.1, 0.2, 0.3, 0.4f32];
        let evaluated = evaluate(&program, &[], &[&a_data], &[]).expect("diamond chain evaluates");
        let reference: Vec<f32> = a_data
            .iter()
            .map(|value| {
                let b_value = value.tanh();
                -b_value + (1.0 / b_value)
            })
            .collect();
        for (found, expected) in evaluated.root().iter().zip(reference.iter()) {
            assert!((found - expected).abs() < 1e-6, "{found} vs {expected}");
        }
    }

    /// `b` is requested as an output alongside the root `c`: it must stay
    /// separately readable and correct rather than disappear into `c`'s
    /// fused body.
    #[test]
    fn a_requested_elementwise_intermediate_stays_readable_and_correct() {
        let mut program = Vec::new();
        let a = f32_block(&mut program, &[Extent::Static(4)]);
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let b = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: alloc::vec![(a, identity())],
                name: None,
            },
        );
        let c = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Negate,
                operands: alloc::vec![(b, identity())],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("requested-output chain infers");
        let resolved =
            bind::bind(&program, &shapes, &[b, c]).expect("requested-output chain resolves");
        assert_eq!(
            resolved.len(),
            2,
            "requesting b as an output must force it to materialize on its own"
        );

        let a_data = [0.1, 0.2, 0.3, 0.4f32];
        let evaluated =
            evaluate(&program, &[], &[&a_data], &[b, c]).expect("requested-output chain evaluates");

        let (b_data, _) = evaluated.get(b).expect("b was requested as an output");
        let b_reference: Vec<f32> = a_data.iter().map(|value| value.tanh()).collect();
        assert_eq!(b_data, b_reference.as_slice());

        let (c_data, _) = evaluated.get(c).expect("c was requested as an output");
        let c_reference: Vec<f32> = b_reference.iter().map(|value| -value).collect();
        assert_eq!(c_data, c_reference.as_slice());
    }

    #[test]
    fn a_requested_intermediate_survives_alongside_the_root() {
        let mut program = Vec::new();
        let source = f32_block(&mut program, &[Extent::Static(4)]);
        let mut current = source;
        let mut nodes = alloc::vec![source];
        for _ in 0..4 {
            current = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Tanh,
                    operands: alloc::vec![(current, IndexMap::Affine(map::projection(1, &[0])))],
                    name: None,
                },
            );
            nodes.push(current);
        }
        let midpoint = nodes[2];
        let root = current;

        let input = [0.1, 0.2, 0.3, 0.4f32];
        let evaluated = evaluate(&program, &[], &[&input], &[midpoint, root])
            .expect("chain with an output request evaluates");

        let (midpoint_data, _) = evaluated
            .get(midpoint)
            .expect("midpoint survives to the end");
        let mut reference = input;
        for value in &mut reference {
            for _ in 0..2 {
                *value = value.tanh();
            }
        }
        for (found, expected) in midpoint_data.iter().zip(reference.iter()) {
            assert!((found - expected).abs() < 1e-6, "{found} vs {expected}");
        }

        let (root_data, _) = evaluated.get(root).expect("root also present");
        let mut full_reference = input;
        for value in &mut full_reference {
            for _ in 0..4 {
                *value = value.tanh();
            }
        }
        for (found, expected) in root_data.iter().zip(full_reference.iter()) {
            assert!((found - expected).abs() < 1e-6, "{found} vs {expected}");
        }
    }

    #[test]
    fn wrong_block_count_is_rejected() {
        let mut program = Vec::new();
        f32_block(&mut program, &[Extent::Static(4)]);

        let error = evaluate(&program, &[], &[], &[]).expect_err("one block is required");
        assert!(
            matches!(error, TensorError::InputCountMismatch { .. }),
            "{error}"
        );
    }

    #[test]
    fn wrong_block_size_is_rejected() {
        let mut program = Vec::new();
        f32_block(&mut program, &[Extent::Static(4)]);

        let too_short = [1.0, 2.0f32];
        let error =
            evaluate(&program, &[], &[&too_short], &[]).expect_err("block is the wrong size");
        assert!(
            matches!(error, TensorError::InputSizeMismatch { .. }),
            "{error}"
        );
    }

    #[test]
    fn a_non_float32_program_is_rejected() {
        let mut program = Vec::new();
        block(&mut program, DType::Int32, &[Extent::Static(4)]);

        let data = [1i32; 0]; // never read: rejected before blocks are consulted
        let _ = data;
        let error = evaluate(&program, &[], &[], &[]).expect_err("int32 is not f32");
        assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
    }

    /// Builds `s in 0..idx.len() -> out[idx[s]] += src[s]` (`body: Add`,
    /// `init: Zero`), `src`/`idx` bound at evaluation time, destination
    /// extent `dest_extent`. `idx`'s values ride in the same `f32` buffer
    /// convention every other gather/scatter test in this file uses
    /// (`map.rs`'s own `IndexMap::Computed` doc: an index value is an exact
    /// integer carried as `f32`).
    fn scatter_add_program(dest_extent: u32) -> (Vec<Op>, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let source = f32_block(&mut program, &[Extent::Static(4)]);
        let ids = block(&mut program, DType::Int32, &[Extent::Static(4)]);
        let out_map = IndexMap::scatter(ids, map::projection(1, &[0]), 1, &[], 0, dest_extent);
        let scattered = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map,
                keep: Keep::Reduce,
                name: Some("scatter_add".into()),
            }),
        );
        (program, source, ids, scattered)
    }

    /// The hand-worked example this task's algorithm-development discipline
    /// requires, walked exactly: `src=[10,20,30,40]`, `idx=[2,0,2,1]`,
    /// destination extent 3, `body: Add`, `init: Zero`.
    ///
    /// Step by step, in iteration order (the order `run_reduce_scatter`
    /// actually walks, and the order that makes the collision below
    /// deterministic rather than merely "some fold"):
    /// - `s=0`: `src[0]=10` -> `idx[0]=2` -> `out[2] = 0 (init) + 10 = 10`
    /// - `s=1`: `src[1]=20` -> `idx[1]=0` -> `out[0] = 0 (init) + 20 = 20`
    /// - `s=2`: `src[2]=30` -> `idx[2]=2` -> `out[2] = 10 + 30 = 40` (collision)
    /// - `s=3`: `src[3]=40` -> `idx[3]=1` -> `out[1] = 0 (init) + 40 = 40`
    ///
    /// Final: `out = [20, 40, 40]`.
    #[test]
    fn scatter_add_matches_the_hand_worked_example() {
        let (program, _source, _ids, scattered) = scatter_add_program(3);
        let index_values = [2.0f32, 0.0, 2.0, 1.0];
        let source_values = [10.0f32, 20.0, 30.0, 40.0];
        let evaluated = evaluate(&program, &[], &[&source_values, &index_values], &[scattered])
            .expect("the hand-worked scatter example evaluates");

        assert_eq!(
            evaluated.root(),
            &[20.0, 40.0, 40.0],
            "out[2] folds src[0] then src[2]; out[0] and out[1] each see one source element"
        );
    }

    /// A destination wider than the source with no two source elements ever
    /// sharing a destination: every cell is either `init`'s identity (`0`,
    /// untouched) or exactly one source value, no fold ever runs twice.
    #[test]
    fn scatter_add_with_no_collisions_places_each_source_element_once() {
        let (program, _source, _ids, scattered) = scatter_add_program(5);
        let index_values = [4.0f32, 1.0, 3.0, 0.0];
        let source_values = [10.0f32, 20.0, 30.0, 40.0];
        let evaluated = evaluate(&program, &[], &[&source_values, &index_values], &[scattered])
            .expect("a collision-free scatter evaluates");

        assert_eq!(
            evaluated.root(),
            &[40.0, 20.0, 0.0, 30.0, 10.0],
            "cell 2 is untouched (init's identity, Zero); every other cell sees exactly one source value"
        );
    }

    /// A fetched destination index outside `[0, dest_extent)` is a real,
    /// named error at evaluation time -- the same
    /// [`TensorError::GatherIndexOutOfRange`] class an out-of-range *read*
    /// (gather) index already raises, reused rather than a second variant
    /// for the write side (`map.rs`'s own doc: scatter is the write-side
    /// twin of gather via the same [`IndexMap::Computed`] machinery).
    #[test]
    fn scatter_add_with_an_out_of_range_destination_index_is_rejected() {
        let (program, _source, _ids, scattered) = scatter_add_program(3);
        let index_values = [0.0f32, 1.0, 3.0, 2.0]; // 3 is out of range for extent 3
        let source_values = [10.0f32, 20.0, 30.0, 40.0];
        let error = evaluate(&program, &[], &[&source_values, &index_values], &[scattered])
            .expect_err("index 3 is out of range for destination extent 3");
        assert!(
            matches!(
                error,
                TensorError::GatherIndexOutOfRange { index: 3, extent: 3, .. }
            ),
            "{error}"
        );
    }

    /// The composition oracle: [`scatter_add_into_a_known_destination_via_mask_composition`]
    /// builds the identical `src`/`idx`/destination-extent-3 scatter-add out
    /// of `Iota`+`Equal`+`Multiply`+`Reduce`, with no `IndexMap::Computed`
    /// anywhere. Running the SAME fixture through this crate's native
    /// forward-scatter (`IndexMap::Computed` as a `Reduce`'s `out_map`) must
    /// land on the exact same numbers -- the composition, not a hand
    /// computation, is what proves the native path correct.
    #[test]
    fn native_scatter_matches_the_mask_composition_oracle_on_the_same_fixture() {
        let (native_program, _source, _ids, native_scattered) = scatter_add_program(3);
        let index_values = [0.0f32, 2.0, 0.0, 1.0];
        let source_values = [10.0f32, 20.0, 30.0, 40.0];
        let native = evaluate(
            &native_program,
            &[],
            &[&source_values, &index_values],
            &[native_scattered],
        )
        .expect("the native scatter evaluates");

        let oracle = [40.0f32, 40.0, 20.0];
        assert_eq!(
            native.root(),
            &oracle,
            "native IndexMap::Computed scatter must match the Iota+Equal+Multiply+Reduce oracle"
        );
    }

    // -- BoundOp::split proof: chunks executed by hand, one buffer, equal
    // -- the unsplit result. `resolve.rs` owns the split's geometry; the
    // -- interpreter that can actually run a chunk lives only here, std-gated.

    #[test]
    fn splitting_an_elementwise_node_and_running_its_chunks_matches_the_unsplit_result() {
        let mut program = Vec::new();
        let source = f32_block(&mut program, &[Extent::Static(10), Extent::Static(4)]);
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: alloc::vec![(source, IndexMap::Affine(map::projection(2, &[0, 1])))],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("elementwise infers");
        let resolved = bind::bind(&program, &shapes, &[]).expect("elementwise resolves");
        let node = &resolved[0];

        let data: Vec<f32> = (0..40).map(|value| value as f32 * 0.01).collect();
        let mut buffers: Vec<Option<Vec<f32>>> = vec![None; program.len()];
        buffers[0] = Some(data);

        let unsplit = run_node(node, &buffers).expect("unsplit runs");

        let chunks = node.split(3).expect("extent 10 over 3 parts splits");
        let mut split_output = vec![0.0f32; unsplit.len()];
        let mut remaining = split_output.as_mut_slice();
        for chunk in &chunks {
            let (this_chunk, rest) = remaining.split_at_mut(node_output_len(chunk));
            run_node_into(chunk, &buffers, None, None, this_chunk).expect("chunk runs");
            remaining = rest;
        }

        assert_eq!(split_output, unsplit);
    }

    #[test]
    fn splitting_a_fused_matmul_reduction_and_running_its_chunks_matches_the_unsplit_result() {
        let (m, k, n) = (8usize, 3usize, 5usize);
        let (program, _sum) = matmul_program(m as u32, k as u32, n as u32, false);
        let lhs: Vec<f32> = (0..m * k).map(|value| value as f32).collect();
        let rhs: Vec<f32> = (0..k * n).map(|value| value as f32).collect();

        let shapes = shape::infer(&program, &[]).expect("matmul infers");
        let resolved = bind::bind(&program, &shapes, &[]).expect("matmul resolves");
        assert_eq!(resolved.len(), 1, "fused into one reduction node");
        let node = &resolved[0];

        let mut buffers: Vec<Option<Vec<f32>>> = vec![None; program.len()];
        buffers[0] = Some(lhs);
        buffers[1] = Some(rhs);

        let unsplit = run_node(node, &buffers).expect("unsplit runs");

        let chunks = node.split(2).expect("8 rows over 2 parts splits");
        let mut split_output = vec![0.0f32; unsplit.len()];
        let mut remaining = split_output.as_mut_slice();
        for chunk in &chunks {
            let (this_chunk, rest) = remaining.split_at_mut(node_output_len(chunk));
            run_node_into(chunk, &buffers, None, None, this_chunk).expect("chunk runs");
            remaining = rest;
        }

        assert_eq!(split_output, unsplit);
    }

    // -- evaluate_parallel proof tests: same programs, workers in {1, 2, 3, 8},
    // -- bitwise-equal to `evaluate`.

    fn assert_parallel_matches_sequential(
        program: &[Op],
        symbols: &[u64],
        blocks: &[&[f32]],
        outputs: &[NodeId],
        workers: usize,
    ) {
        let workers = NonZeroUsize::new(workers).expect("every case here uses a nonzero count");
        let sequential = evaluate(program, symbols, blocks, outputs).expect("sequential evaluates");
        let parallel = evaluate_parallel(program, symbols, blocks, outputs, workers)
            .expect("parallel evaluates");

        assert_eq!(parallel.shape(), sequential.shape());
        assert_eq!(parallel.root(), sequential.root());
        for &node in outputs {
            assert_eq!(
                parallel.get(node),
                sequential.get(node),
                "node {node} output diverges"
            );
        }
    }

    /// [`run_elementwise_dispatch`]'s own cohort path only ever fires
    /// inside [`evaluate_quantized`] (the `session: Some(..)` arm
    /// `evaluate_parallel` never takes — `evaluate_parallel_matches_evaluate`'s
    /// cases above all exercise the pool path, `run_chunks_threaded` with
    /// `session: None`, not this one). A large elementwise chain clears
    /// `PARALLEL_THRESHOLD` and has `outer_len` (64) comfortably above any
    /// worker count tried here, so this is the one test that actually
    /// drives a cohort round for [`ElementwiseRowRound`] and checks its
    /// output is bit-identical — `assert_eq!`, not a tolerance — to the
    /// fully sequential [`evaluate`] path, per this node kind's own
    /// no-cross-element-accumulation argument (`run_elementwise_dispatch`'s
    /// doc).
    #[proxima::test]
    #[case::two_workers(2)]
    #[case::three_workers(3)]
    #[case::eight_workers(8)]
    async fn evaluate_quantized_matches_evaluate_for_a_large_elementwise_chain(
        #[case] workers: usize,
    ) {
        // SAFETY of the test env var mutation: `PROXIMA_MATMUL_WORKERS` is
        // read exactly once, lazily, behind `matmul_worker_count`'s own
        // `OnceLock` — set before that lock is ever touched by any other
        // test in this process would be a race, so this case relies on
        // nextest's default one-test-per-process isolation instead of
        // resetting the lock.
        // SAFETY: nextest runs each test in its own process, so no other
        // thread in this process reads or writes the environment
        // concurrently with this call.
        unsafe {
            std::env::set_var("PROXIMA_MATMUL_WORKERS", workers.to_string());
        }

        let mut program = Vec::new();
        let (rows, width) = (64usize, 8192usize);
        let mut current = f32_block(
            &mut program,
            &[Extent::Static(rows as u32), Extent::Static(width as u32)],
        );
        for _ in 0..3 {
            current = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Tanh,
                    operands: alloc::vec![(current, IndexMap::Affine(map::projection(2, &[0, 1])))],
                    name: None,
                },
            );
        }
        let _ = current;

        let input: Vec<f32> = (0..rows * width).map(|value| (value as f32) * 0.0001).collect();

        let sequential = evaluate(&program, &[], &[&input], &[]).expect("sequential evaluates");
        let blocks = [QuantizedBlock::Float32(&input)];
        let quantized =
            evaluate_quantized(&program, &[], &blocks, &[]).expect("quantized evaluates");

        assert_eq!(quantized.shape(), sequential.shape());
        assert_eq!(
            quantized.root(),
            sequential.root(),
            "cohort-dispatched elementwise output diverges from the sequential path"
        );
    }

    #[proxima::test]
    #[case::one_worker(1)]
    #[case::two_workers(2)]
    #[case::three_workers(3)]
    #[case::eight_workers(8)]
    async fn evaluate_parallel_matches_evaluate_for_a_matmul(#[case] workers: usize) {
        let (m, k, n) = (4usize, 3usize, 5usize);
        let (program, _sum) = matmul_program(m as u32, k as u32, n as u32, false);
        let lhs: Vec<f32> = (0..m * k).map(|value| value as f32).collect();
        let rhs: Vec<f32> = (0..k * n).map(|value| value as f32).collect();

        assert_parallel_matches_sequential(&program, &[], &[&lhs, &rhs], &[], workers);
    }

    #[proxima::test]
    #[case::one_worker(1)]
    #[case::two_workers(2)]
    #[case::three_workers(3)]
    #[case::eight_workers(8)]
    async fn evaluate_parallel_matches_evaluate_for_a_tanh_chain(#[case] workers: usize) {
        let mut program = Vec::new();
        let mut current = f32_block(&mut program, &[Extent::Static(4)]);
        for _ in 0..8 {
            current = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Tanh,
                    operands: alloc::vec![(current, IndexMap::Affine(map::projection(1, &[0])))],
                    name: None,
                },
            );
        }
        let _ = current;

        let input = [0.1, 0.2, 0.3, 0.4f32];
        assert_parallel_matches_sequential(&program, &[], &[&input], &[], workers);
    }

    #[proxima::test]
    #[case::one_worker(1)]
    #[case::two_workers(2)]
    #[case::three_workers(3)]
    #[case::eight_workers(8)]
    async fn evaluate_parallel_matches_evaluate_for_softmax(#[case] workers: usize) {
        let mut program = Vec::new();
        let (n, d) = (2usize, 4usize);
        let input = f32_block(
            &mut program,
            &[Extent::Static(n as u32), Extent::Static(d as u32)],
        );

        let row_map = IndexMap::Affine(map::projection(2, &[0, 1]));
        let broadcast_map = IndexMap::Affine(map::projection(2, &[0]));

        let max = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Maximum,
                init: ReduceInit::NegativeInfinity,
                operand: input,
                in_map: row_map.clone(),
                out_map: broadcast_map.clone(),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let shifted = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Subtract,
                operands: alloc::vec![(input, row_map.clone()), (max, broadcast_map.clone())],
                name: None,
            },
        );
        let exponentiated = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Exponential,
                operands: alloc::vec![(shifted, row_map.clone())],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: exponentiated,
                in_map: row_map.clone(),
                out_map: broadcast_map.clone(),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Divide,
                operands: alloc::vec![(exponentiated, row_map), (sum, broadcast_map)],
                name: None,
            },
        );

        let input_data = [1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0f32];
        assert_parallel_matches_sequential(&program, &[], &[&input_data], &[], workers);
    }

    #[proxima::test]
    #[case::one_worker(1)]
    #[case::two_workers(2)]
    #[case::three_workers(3)]
    #[case::eight_workers(8)]
    async fn evaluate_parallel_matches_evaluate_for_cumsum(#[case] workers: usize) {
        let mut program = Vec::new();
        let source = f32_block(&mut program, &[Extent::Static(6)]);
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[0])),
                keep: Keep::Scan,
                name: None,
            }),
        );

        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
        assert_parallel_matches_sequential(&program, &[], &[&data], &[], workers);
    }

    #[proxima::test]
    #[case::one_worker(1)]
    #[case::two_workers(2)]
    #[case::three_workers(3)]
    #[case::eight_workers(8)]
    async fn evaluate_parallel_matches_evaluate_for_multiple_requested_outputs(
        #[case] workers: usize,
    ) {
        let mut program = Vec::new();
        let source = f32_block(&mut program, &[Extent::Static(4)]);
        let mut current = source;
        let mut nodes = alloc::vec![source];
        for _ in 0..4 {
            current = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Tanh,
                    operands: alloc::vec![(current, IndexMap::Affine(map::projection(1, &[0])))],
                    name: None,
                },
            );
            nodes.push(current);
        }
        let midpoint = nodes[2];
        let root = current;

        let input = [0.1, 0.2, 0.3, 0.4f32];
        assert_parallel_matches_sequential(&program, &[], &[&input], &[midpoint, root], workers);
    }

    #[test]
    fn a_matmul_past_the_parallel_threshold_actually_splits_and_still_matches_sequential() {
        let (m, k, n) = (64usize, 64usize, 64usize);
        let (program, _sum) = matmul_program(m as u32, k as u32, n as u32, false);
        let lhs: Vec<f32> = (0..m * k).map(|value| (value % 7) as f32).collect();
        let rhs: Vec<f32> = (0..k * n).map(|value| (value % 5) as f32).collect();

        let shapes = shape::infer(&program, &[]).expect("64x64x64 matmul infers");
        let resolved = bind::bind(&program, &shapes, &[]).expect("64x64x64 matmul resolves");
        assert_eq!(resolved.len(), 1, "fused into one reduction node");
        assert!(
            element_count(&resolved[0].extents) >= PARALLEL_THRESHOLD,
            "this size must clear the threshold or this test proves nothing about the \
             threaded path"
        );
        assert!(
            resolved[0].split(4).is_some(),
            "the node must actually be splittable for the threaded path to run"
        );

        let workers = NonZeroUsize::new(4).expect("4 is nonzero");
        assert_parallel_matches_sequential(&program, &[], &[&lhs, &rhs], &[], workers.get());
    }

    #[test]
    fn evaluate_parallel_raises_the_same_errors_as_evaluate_on_every_existing_sad_path() {
        let workers = NonZeroUsize::new(2).expect("2 is nonzero");

        let mut count_program = Vec::new();
        f32_block(&mut count_program, &[Extent::Static(4)]);
        let sequential_error =
            evaluate(&count_program, &[], &[], &[]).expect_err("one block is required");
        let parallel_error = evaluate_parallel(&count_program, &[], &[], &[], workers)
            .expect_err("one block is required");
        assert_eq!(sequential_error, parallel_error);

        let mut size_program = Vec::new();
        f32_block(&mut size_program, &[Extent::Static(4)]);
        let too_short = [1.0, 2.0f32];
        let sequential_error =
            evaluate(&size_program, &[], &[&too_short], &[]).expect_err("block is wrong size");
        let parallel_error = evaluate_parallel(&size_program, &[], &[&too_short], &[], workers)
            .expect_err("block is wrong size");
        assert_eq!(sequential_error, parallel_error);

        let mut dtype_program = Vec::new();
        block(&mut dtype_program, DType::Int32, &[Extent::Static(4)]);
        let sequential_error =
            evaluate(&dtype_program, &[], &[], &[]).expect_err("int32 is not f32");
        let parallel_error = evaluate_parallel(&dtype_program, &[], &[], &[], workers)
            .expect_err("int32 is not f32");
        assert_eq!(sequential_error, parallel_error);

        // A `Keep::Scan` scatter stays rejected (shape.rs's own doc: a scan
        // step would need to read the destination its own write just
        // touched, which the sequential interpreter does not order that
        // way) -- unlike a `Keep::Reduce` scatter, which this row's own
        // `scatter_add_matches_the_hand_worked_example` etc. now accept.
        let mut scatter_scan_program = Vec::new();
        let source = f32_block(&mut scatter_scan_program, &[Extent::Static(4)]);
        let ids = block(&mut scatter_scan_program, DType::Int32, &[Extent::Static(4)]);
        let out_map = IndexMap::scatter(ids, map::projection(1, &[0]), 1, &[], 0, 3);
        append(
            &mut scatter_scan_program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map,
                keep: Keep::Scan,
                name: None,
            }),
        );
        let sequential_error = evaluate(&scatter_scan_program, &[], &[], &[])
            .expect_err("a scatter scan has no defined step order");
        let parallel_error = evaluate_parallel(&scatter_scan_program, &[], &[], &[], workers)
            .expect_err("a scatter scan has no defined step order");
        assert_eq!(sequential_error, parallel_error);
    }

    /// `evaluate_parallel`'s own chunking never touches a scatter node
    /// (`bind::BoundOp::split` refuses to split one -- see its own doc), so
    /// running the hand-worked scatter example through the parallel driver
    /// must land on the exact same numbers `evaluate` does.
    #[test]
    fn evaluate_parallel_matches_evaluate_on_the_hand_worked_scatter_example() {
        let workers = NonZeroUsize::new(4).expect("4 is nonzero");
        let (program, _source, _ids, scattered) = scatter_add_program(3);
        let index_values = [2.0f32, 0.0, 2.0, 1.0];
        let source_values = [10.0f32, 20.0, 30.0, 40.0];

        let sequential = evaluate(&program, &[], &[&source_values, &index_values], &[scattered])
            .expect("sequential scatter evaluates");
        let parallel = evaluate_parallel(
            &program,
            &[],
            &[&source_values, &index_values],
            &[scattered],
            workers,
        )
        .expect("parallel scatter evaluates");

        assert_eq!(sequential.root(), &[20.0, 40.0, 40.0]);
        assert_eq!(sequential.root(), parallel.root());
    }

    #[test]
    fn peak_live_buffers_on_a_tanh_chain_stays_small_under_evaluate_parallel() {
        let mut program = Vec::new();
        let mut current = f32_block(&mut program, &[Extent::Static(4)]);
        for _ in 0..8 {
            current = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Tanh,
                    operands: alloc::vec![(current, IndexMap::Affine(map::projection(1, &[0])))],
                    name: None,
                },
            );
        }
        let _ = current;

        let input = [0.1, 0.2, 0.3, 0.4f32];
        let workers = NonZeroUsize::new(4).expect("4 is nonzero");
        let evaluated = evaluate_parallel(&program, &[], &[&input], &[], workers)
            .expect("tanh chain evaluates in parallel");

        let peak = evaluated
            .peak_live_buffers()
            .expect("evaluate_parallel tracks peak live buffers");
        assert!(
            peak <= 3,
            "streaming a chain of 8 unary elementwise ops should not hold one buffer per op, got {peak}"
        );
    }

    // THE PROOF: the full pipeline — shape inference, layout binding, and
    // CPU execution — driven entirely through the real `Pipe` algebra, as
    // one composed chain (`shapes.and_then(builder).and_then(interpreter)`
    // via `PipeExt`), matches what `evaluate` (the free-function path every
    // other test in this crate trusts) produces for the identical matmul
    // program.
    //
    // The three-stage `AndThen` typechecks because `Second::In = First::Out`
    // holds at both joins: `ShapeTable::Out = (Op, Shapes) = BoundOpBuilder::In`,
    // and `BoundOpBuilder::Out = Vec<BoundOp> = Interpreter::In` — `Interpreter`
    // takes the batch a push readies (0, 1, or 2 records — see
    // `bind::BoundOpBuilder::push`'s own doc) directly, so no per-node
    // driving loop is needed at the call site; `Pipe::call` on the full
    // chain is called once per `Op` record, exactly as `ShapeTable`'s own
    // one-record-at-a-time contract expects.
    #[test]
    fn execute_composes_through_pipe_ext_matching_the_free_function() {
        use crate::bind::BoundOpBuilder;
        use crate::live;
        use crate::shape::ShapeTable;
        use proxima_primitives::block_on;
        use proxima_primitives::pipe::{Pipe, PipeExt};

        let (m, k, n) = (4usize, 3usize, 5usize);
        let (program, sum) = matmul_program(m as u32, k as u32, n as u32, false);
        let lhs: Vec<f32> = (0..m * k).map(|value| value as f32).collect();
        let rhs: Vec<f32> = (0..k * n).map(|value| value as f32).collect();

        let outputs: Vec<NodeId> = Vec::new();
        let retires = live::annotate(&program, &outputs);
        let shapes = ShapeTable::new(&[]);
        let builder = BoundOpBuilder::new(retires);

        // `matmul_program` always appends `lhs` then `rhs` first.
        let mut buffers: Vec<Option<Vec<f32>>> = vec![None; program.len()];
        buffers[0] = Some(lhs.clone());
        buffers[1] = Some(rhs.clone());
        let chain = shapes.and_then(builder).and_then(Interpreter::new(&mut buffers));

        for expr in &program {
            block_on(Pipe::call(&chain, expr.clone())).expect("shape+bind+execute pipe step succeeds");
        }
        // Release the chain's mutable borrow of `buffers` before reading the
        // result back out of it — the interpreter stage was moved into
        // `chain`, so its `get()` is unreachable here, but its buffer table
        // IS `buffers`: reading `buffers[sum.0]` directly is the same read
        // `Interpreter::get` performs, once the borrow is free to take back.
        drop(chain);

        let chain_result =
            buffers[sum.0 as usize].clone().expect("the matmul node was executed through the composed chain");

        let evaluated =
            evaluate(&program, &[], &[&lhs, &rhs], &[]).expect("free-function matmul evaluates");

        assert_eq!(chain_result, evaluated.root());
    }

    fn typed_identity() -> IndexMap {
        IndexMap::Affine(map::projection(1, &[0]))
    }

    fn typed_add_program(dtype: DType, len: u32) -> (Vec<Op>, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let lhs = block(&mut program, dtype, &[Extent::Static(len)]);
        let rhs = block(&mut program, dtype, &[Extent::Static(len)]);
        let sum = append(
            &mut program,
            Op::Elementwise {
                dtype,
                body: ScalarOp::Add,
                operands: alloc::vec![(lhs, typed_identity()), (rhs, typed_identity())],
                name: None,
            },
        );
        (program, lhs, rhs, sum)
    }

    #[proxima::test]
    #[case::int8(DType::Int8, TypedBuffer::Int8(alloc::vec![1, 2, 3]), TypedBuffer::Int8(alloc::vec![10, 20, 30]), TypedBuffer::Int8(alloc::vec![11, 22, 33]))]
    #[case::uint8(DType::UInt8, TypedBuffer::UInt8(alloc::vec![1, 2, 3]), TypedBuffer::UInt8(alloc::vec![10, 20, 30]), TypedBuffer::UInt8(alloc::vec![11, 22, 33]))]
    #[case::int16(DType::Int16, TypedBuffer::Int16(alloc::vec![1, 2, 3]), TypedBuffer::Int16(alloc::vec![10, 20, 30]), TypedBuffer::Int16(alloc::vec![11, 22, 33]))]
    #[case::uint16(DType::UInt16, TypedBuffer::UInt16(alloc::vec![1, 2, 3]), TypedBuffer::UInt16(alloc::vec![10, 20, 30]), TypedBuffer::UInt16(alloc::vec![11, 22, 33]))]
    #[case::int32(DType::Int32, TypedBuffer::Int32(alloc::vec![1, 2, 3]), TypedBuffer::Int32(alloc::vec![10, 20, 30]), TypedBuffer::Int32(alloc::vec![11, 22, 33]))]
    #[case::uint32(DType::UInt32, TypedBuffer::UInt32(alloc::vec![1, 2, 3]), TypedBuffer::UInt32(alloc::vec![10, 20, 30]), TypedBuffer::UInt32(alloc::vec![11, 22, 33]))]
    #[case::int64(DType::Int64, TypedBuffer::Int64(alloc::vec![1, 2, 3]), TypedBuffer::Int64(alloc::vec![10, 20, 30]), TypedBuffer::Int64(alloc::vec![11, 22, 33]))]
    #[case::uint64(DType::UInt64, TypedBuffer::UInt64(alloc::vec![1, 2, 3]), TypedBuffer::UInt64(alloc::vec![10, 20, 30]), TypedBuffer::UInt64(alloc::vec![11, 22, 33]))]
    #[case::int128(DType::Int128, TypedBuffer::Int128(alloc::vec![1, 2, 3]), TypedBuffer::Int128(alloc::vec![10, 20, 30]), TypedBuffer::Int128(alloc::vec![11, 22, 33]))]
    #[case::uint128(DType::UInt128, TypedBuffer::UInt128(alloc::vec![1, 2, 3]), TypedBuffer::UInt128(alloc::vec![10, 20, 30]), TypedBuffer::UInt128(alloc::vec![11, 22, 33]))]
    #[case::float64(DType::Float64, TypedBuffer::Float64(alloc::vec![1.5, 2.5, 3.5]), TypedBuffer::Float64(alloc::vec![10.0, 20.0, 30.0]), TypedBuffer::Float64(alloc::vec![11.5, 22.5, 33.5]))]
    async fn evaluate_typed_adds_across_every_extended_width(
        #[case] dtype: DType,
        #[case] lhs: TypedBuffer,
        #[case] rhs: TypedBuffer,
        #[case] expected: TypedBuffer,
    ) {
        let (program, _, _, _) = typed_add_program(dtype, 3);
        let results =
            evaluate_typed(&program, &[], &[lhs, rhs], &[]).expect("typed add evaluates");
        assert_eq!(results.len(), 1);
        let (_, shape, data) = &results[0];
        assert_eq!(shape, &alloc::vec![3u64]);
        assert_eq!(*data, expected);
        assert_eq!(data.dtype(), dtype);
        assert_eq!(data.len(), 3);
        assert!(!data.is_empty());
    }

    #[test]
    fn evaluate_typed_wraps_signed_narrow_overflow_instead_of_panicking() {
        let (program, _, _, _) = typed_add_program(DType::Int8, 1);
        let lhs = TypedBuffer::Int8(alloc::vec![127]);
        let rhs = TypedBuffer::Int8(alloc::vec![1]);
        let results = evaluate_typed(&program, &[], &[lhs, rhs], &[])
            .expect("i8 add wraps rather than panicking");
        assert_eq!(results[0].2, TypedBuffer::Int8(alloc::vec![-128]));
    }

    #[test]
    fn evaluate_typed_rejects_negate_on_an_unsigned_dtype() {
        let mut program = Vec::new();
        let operand = block(&mut program, DType::UInt32, &[Extent::Static(2)]);
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::UInt32,
                body: ScalarOp::Negate,
                operands: alloc::vec![(operand, typed_identity())],
                name: None,
            },
        );
        let blocks = [TypedBuffer::UInt32(alloc::vec![1, 2])];
        let error =
            evaluate_typed(&program, &[], &blocks, &[]).expect_err("u32 has no representable negative");
        assert!(matches!(error, TensorError::UnsupportedScalarOp { op: ScalarOp::Negate, dtype: DType::UInt32, .. }), "{error}");
    }

    #[test]
    fn evaluate_typed_rejects_a_transcendental_on_an_integer_dtype() {
        let mut program = Vec::new();
        let operand = block(&mut program, DType::Int32, &[Extent::Static(2)]);
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Int32,
                body: ScalarOp::SquareRoot,
                operands: alloc::vec![(operand, typed_identity())],
                name: None,
            },
        );
        let blocks = [TypedBuffer::Int32(alloc::vec![4, 9])];
        let error = evaluate_typed(&program, &[], &blocks, &[])
            .expect_err("sqrt is not defined over Int32 by this evaluator");
        assert!(
            matches!(error, TensorError::UnsupportedScalarOp { op: ScalarOp::SquareRoot, dtype: DType::Int32, .. }),
            "{error}"
        );
    }

    #[test]
    fn evaluate_typed_reports_integer_divide_by_zero_instead_of_panicking() {
        let mut program = Vec::new();
        let lhs = block(&mut program, DType::Int32, &[Extent::Static(1)]);
        let rhs = block(&mut program, DType::Int32, &[Extent::Static(1)]);
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Int32,
                body: ScalarOp::Divide,
                operands: alloc::vec![(lhs, typed_identity()), (rhs, typed_identity())],
                name: None,
            },
        );
        let blocks = [TypedBuffer::Int32(alloc::vec![10]), TypedBuffer::Int32(alloc::vec![0])];
        let error = evaluate_typed(&program, &[], &blocks, &[])
            .expect_err("integer division by zero is a real error, not UB");
        assert!(matches!(error, TensorError::CheckedDivisionFailed { .. }), "{error}");
    }

    #[test]
    fn evaluate_typed_computes_float64_transcendentals() {
        let mut program = Vec::new();
        let operand = block(&mut program, DType::Float64, &[Extent::Static(3)]);
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float64,
                body: ScalarOp::SquareRoot,
                operands: alloc::vec![(operand, typed_identity())],
                name: None,
            },
        );
        let blocks = [TypedBuffer::Float64(alloc::vec![4.0, 9.0, 16.0])];
        let results =
            evaluate_typed(&program, &[], &blocks, &[]).expect("f64 sqrt evaluates");
        assert_eq!(results[0].2, TypedBuffer::Float64(alloc::vec![2.0, 3.0, 4.0]));
    }

    #[test]
    fn evaluate_typed_rejects_a_program_mixing_dtypes() {
        let mut program = Vec::new();
        let lhs = block(&mut program, DType::Int32, &[Extent::Static(2)]);
        let rhs = block(&mut program, DType::Int8, &[Extent::Static(2)]);
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Int32,
                body: ScalarOp::Add,
                operands: alloc::vec![(lhs, typed_identity()), (rhs, typed_identity())],
                name: None,
            },
        );
        let blocks = [
            TypedBuffer::Int32(alloc::vec![1, 2]),
            TypedBuffer::Int8(alloc::vec![1, 2]),
        ];
        let error = evaluate_typed(&program, &[], &blocks, &[])
            .expect_err("a mixed-dtype fused body is not yet supported");
        assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
    }

    #[test]
    fn an_i8_operand_i32_accumulator_reduce_evaluates() {
        // five i8 elements of 30 each: the true sum is 150, which does not
        // fit in i8 (max 127) -- wrapping i8 arithmetic would land on -106
        // (150 - 256). an i32 accumulator is the only way to observe 150.
        let mut program = Vec::new();
        let operand = block(&mut program, DType::Int8, &[Extent::Static(5)]);
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Int32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let blocks = [TypedBuffer::Int8(alloc::vec![30, 30, 30, 30, 30])];
        let results = evaluate_typed(&program, &[], &blocks, &[])
            .expect("an i8-operand, i32-accumulator reduce evaluates");
        assert_eq!(
            results[0].2,
            TypedBuffer::Int32(alloc::vec![150]),
            "the i32 accumulator must carry the true sum, not the i8-wrapped one (-106)"
        );
    }

    #[test]
    fn f32_typed_path_is_unchanged() {
        let (program, _) = typed_reduce_vector_to_scalar_program(DType::Float32, 4);
        let operand = TypedBuffer::Float32(alloc::vec![1.5, 2.5, 3.0, 4.0]);
        let results = evaluate_typed(&program, &[], &[operand], &[])
            .expect("a uniform f32 typed program still evaluates via the unchanged NEON-backed path");
        assert_eq!(results[0].2, TypedBuffer::Float32(alloc::vec![11.0]));
    }

    #[test]
    fn an_unshipped_widened_pair_is_rejected_not_silently_wrong() {
        let mut program = Vec::new();
        let operand = block(&mut program, DType::UInt16, &[Extent::Static(3)]);
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::UInt64,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let blocks = [TypedBuffer::UInt16(alloc::vec![1, 2, 3])];
        let error = evaluate_typed(&program, &[], &blocks, &[])
            .expect_err("(UInt16, UInt64) is not a shipped widened pair");
        assert!(
            matches!(error, TensorError::NotLowerable { .. }),
            "an unshipped pair must fail honestly, never fall back to a wrong result: {error}"
        );
    }

    /// A widened reduce program: `operand_dtype` operand folded by `Add`
    /// into an `accumulator_dtype` accumulator — the shape
    /// [`an_i8_operand_i32_accumulator_reduce_evaluates`] built by hand,
    /// generalized over the dtype pair so every widened-pair test below
    /// shares one builder.
    fn typed_widened_reduce_program(
        operand_dtype: DType,
        accumulator_dtype: DType,
        len: u32,
    ) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let operand = block(&mut program, operand_dtype, &[Extent::Static(len)]);
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: accumulator_dtype,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, sum)
    }

    #[proxima::test]
    #[case::f16_add(DType::Float16, TypedBuffer::Float16(alloc::vec![f16::from_f32(1.5), f16::from_f32(2.25), f16::from_f32(-3.0)]), TypedBuffer::Float16(alloc::vec![f16::from_f32(10.0), f16::from_f32(20.0), f16::from_f32(30.0)]))]
    #[case::bf16_add(DType::BFloat16, TypedBuffer::BFloat16(alloc::vec![bf16::from_f32(1.5), bf16::from_f32(2.25), bf16::from_f32(-3.0)]), TypedBuffer::BFloat16(alloc::vec![bf16::from_f32(10.0), bf16::from_f32(20.0), bf16::from_f32(30.0)]))]
    async fn half_precision_uniform_elementwise_matches_f32_reference(
        #[case] dtype: DType,
        #[case] lhs: TypedBuffer,
        #[case] rhs: TypedBuffer,
    ) {
        let (program, _, _, _) = typed_add_program(dtype, 3);
        let results = evaluate_typed(&program, &[], &[lhs.clone(), rhs.clone()], &[])
            .expect("half-precision add evaluates through the typed path");
        let (got, reference): (Vec<f32>, Vec<f32>) = match (&results[0].2, &lhs, &rhs) {
            (TypedBuffer::Float16(sum), TypedBuffer::Float16(lhs), TypedBuffer::Float16(rhs)) => (
                sum.iter().map(|value| value.to_f32()).collect(),
                lhs.iter().zip(rhs).map(|(left, right)| left.to_f32() + right.to_f32()).collect(),
            ),
            (TypedBuffer::BFloat16(sum), TypedBuffer::BFloat16(lhs), TypedBuffer::BFloat16(rhs)) => (
                sum.iter().map(|value| value.to_f32()).collect(),
                lhs.iter().zip(rhs).map(|(left, right)| left.to_f32() + right.to_f32()).collect(),
            ),
            other => panic!("unexpected buffer shape: {other:?}"),
        };
        for (value, expected) in got.iter().zip(&reference) {
            // one rounding step from the f32 reference (the operands are
            // already half-precision, so the reference itself is exact at
            // this magnitude) -- a loose bound catching a wrong op, not
            // tuned to a measured residual.
            assert!(
                (value - expected).abs() < 1e-2,
                "half-precision add {value} vs f32 reference {expected}"
            );
        }
    }

    #[proxima::test]
    #[case::f16_sum(DType::Float16)]
    #[case::bf16_sum(DType::BFloat16)]
    async fn half_precision_uniform_reduce_matches_f32_reference(#[case] dtype: DType) {
        let values = [1.5f32, 2.5, -0.5, 4.0];
        let (program, _) = typed_reduce_vector_to_scalar_program(dtype, values.len() as u32);
        let expected_f32: f32 = values.iter().sum();
        let operand = match dtype {
            DType::Float16 => TypedBuffer::Float16(values.iter().map(|value| f16::from_f32(*value)).collect()),
            DType::BFloat16 => TypedBuffer::BFloat16(values.iter().map(|value| bf16::from_f32(*value)).collect()),
            other => panic!("unexpected dtype in case table: {other:?}"),
        };
        let results = evaluate_typed(&program, &[], &[operand], &[])
            .expect("half-precision reduce evaluates through the typed path");
        let got = match &results[0].2 {
            TypedBuffer::Float16(data) => data[0].to_f32(),
            TypedBuffer::BFloat16(data) => data[0].to_f32(),
            other => panic!("unexpected result buffer: {other:?}"),
        };
        assert!(
            (got - expected_f32).abs() < 1e-2,
            "half-precision reduce {got} vs f32 reference {expected_f32}"
        );
    }

    #[test]
    fn f16_reduce_widens_into_an_f32_accumulator_where_f16_alone_overflows() {
        // two f16 values whose sum overflows f16 range (max ~65504) and
        // would round to infinity if accumulated in f16, but the true sum
        // fits an f32 accumulator exactly -- the same "widening changes the
        // observable result" shape as the i8/i32 test above, at floating
        // widths instead of integer ones.
        let (program, _) = typed_widened_reduce_program(DType::Float16, DType::Float32, 2);
        let operand = TypedBuffer::Float16(alloc::vec![f16::from_f32(40000.0), f16::from_f32(40000.0)]);
        let results = evaluate_typed(&program, &[], &[operand], &[])
            .expect("an f16-operand, f32-accumulator reduce evaluates");
        let TypedBuffer::Float32(sum) = &results[0].2 else {
            panic!("widened f16 reduce must produce an f32 accumulator buffer");
        };
        assert_eq!(sum[0], 80000.0, "the f32 accumulator must carry the true sum, not an f16-saturated infinity");
    }

    #[test]
    fn bf16_reduce_widens_into_an_f32_accumulator_exactly() {
        let (program, _) = typed_widened_reduce_program(DType::BFloat16, DType::Float32, 3);
        let operand = TypedBuffer::BFloat16(alloc::vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
        ]);
        let results = evaluate_typed(&program, &[], &[operand], &[])
            .expect("a bf16-operand, f32-accumulator reduce evaluates");
        assert_eq!(results[0].2, TypedBuffer::Float32(alloc::vec![6.0]));
    }

    #[test]
    fn typed_program_plan_no_longer_rejects_float16_or_bfloat16_but_still_rejects_bool() {
        let (float16_program, _, _, _) = typed_add_program(DType::Float16, 2);
        let float16_blocks = [
            TypedBuffer::Float16(alloc::vec![f16::from_f32(1.0), f16::from_f32(2.0)]),
            TypedBuffer::Float16(alloc::vec![f16::from_f32(10.0), f16::from_f32(20.0)]),
        ];
        evaluate_typed(&float16_program, &[], &float16_blocks, &[])
            .expect("Float16 must no longer be rejected by typed_program_plan");

        let mut bool_program = Vec::new();
        let bool_operand = block(&mut bool_program, DType::Bool, &[Extent::Static(2)]);
        append(
            &mut bool_program,
            Op::Elementwise {
                dtype: DType::Bool,
                body: ScalarOp::Identity,
                operands: alloc::vec![(bool_operand, typed_identity())],
                name: None,
            },
        );
        let error = evaluate_typed(&bool_program, &[], &[], &[])
            .expect_err("Bool must still be rejected -- no TypedBuffer variant backs it");
        assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
    }

    /// `table[ids[s], d]`, `table` at `compute_dtype` and `ids` at
    /// `index_dtype` -- the same [`IndexMap::Computed`] wiring
    /// `embedding_lookup_program` uses, parameterized over both dtypes so
    /// [`typed_program_plan`]'s third, index role can be exercised at any
    /// compute width against any integer index width.
    fn typed_gather_program(compute_dtype: DType, index_dtype: DType) -> Vec<Op> {
        let mut program = Vec::new();
        let table = block(&mut program, compute_dtype, &[Extent::Static(4), Extent::Static(2)]);
        let ids = block(&mut program, index_dtype, &[Extent::Static(3)]);
        let gathered_map = IndexMap::Computed {
            indices: ids,
            index_map: map::projection(2, &[0]),
            base: map::IndexPattern {
                iter_rank: 2,
                axes: alloc::vec![
                    map::AxisIndex::default(),
                    map::AxisIndex {
                        terms: core::iter::once(AxisTerm::projection(1)).collect(),
                        offset: 0,
                    },
                ],
            },
            gathered_dim: 0,
        };
        append(
            &mut program,
            Op::Elementwise {
                dtype: compute_dtype,
                body: ScalarOp::Identity,
                operands: alloc::vec![(table, gathered_map)],
                name: None,
            },
        );
        program
    }

    /// [`typed_program_plan`]'s third role: a gather index node carries its
    /// own integer dtype, distinct from the program's compute dtype,
    /// without failing the plan -- proven directly against the plan
    /// function. See the `typed_gather_*` tests below for the same shape
    /// proven all the way through `evaluate_typed` against the f32 pipeline
    /// as oracle.
    #[proxima::test]
    #[case::i32_index_over_f32_compute(DType::Float32, DType::Int32)]
    #[case::u32_index_over_f32_compute(DType::Float32, DType::UInt32)]
    #[case::i32_index_over_f16_compute(DType::Float16, DType::Int32)]
    #[case::u32_index_over_f16_compute(DType::Float16, DType::UInt32)]
    async fn typed_program_plan_permits_an_integer_gather_index_distinct_from_compute_dtype(
        #[case] compute_dtype: DType,
        #[case] index_dtype: DType,
    ) {
        let program = typed_gather_program(compute_dtype, index_dtype);
        let plan = typed_program_plan(&program).expect("an integer gather index must not fail the plan");
        assert_eq!(
            plan,
            TypedPlan::Uniform(compute_dtype),
            "the index node's own dtype must not be folded into the program's compute dtype"
        );
    }

    /// The sad path this role's own gate exists for: a FLOAT gather index
    /// dtype is not a legal index type ([`DType::is_integer`]) and must
    /// still fail the plan, named, rather than being silently accepted
    /// alongside the new integer exemption.
    #[test]
    fn typed_program_plan_rejects_a_float_gather_index_dtype() {
        let program = typed_gather_program(DType::Float32, DType::Float64);
        let error = typed_program_plan(&program)
            .expect_err("a float-dtype gather index must be rejected, never silently accepted");
        assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
    }

    /// `table[ids[s], d]` at a chosen compute/index dtype pair, `dim`
    /// always the kept axis (`gathered_dim: 0`) -- the typed counterpart of
    /// [`embedding_lookup_program`], parameterized the same way
    /// [`typed_gather_program`] is, but at real `vocab`/`dim`/`seq` sizes so
    /// its output can be diffed against the f32 oracle element-for-element
    /// rather than only checked through [`typed_program_plan`].
    fn typed_embedding_lookup_program(
        compute_dtype: DType,
        index_dtype: DType,
        vocab: u32,
        dim: u32,
        seq: u32,
    ) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let table = block(&mut program, compute_dtype, &[Extent::Static(vocab), Extent::Static(dim)]);
        let ids = block(&mut program, index_dtype, &[Extent::Static(seq)]);
        let gathered_map = IndexMap::Computed {
            indices: ids,
            index_map: map::projection(2, &[0]),
            base: map::IndexPattern {
                iter_rank: 2,
                axes: alloc::vec![
                    map::AxisIndex::default(),
                    map::AxisIndex {
                        terms: core::iter::once(AxisTerm::projection(1)).collect(),
                        offset: 0,
                    },
                ],
            },
            gathered_dim: 0,
        };
        let gathered = append(
            &mut program,
            Op::Elementwise {
                dtype: compute_dtype,
                body: ScalarOp::Identity,
                operands: alloc::vec![(table, gathered_map)],
                name: None,
            },
        );
        (program, gathered)
    }

    /// [`typed_embedding_lookup_program`]'s sibling with the gathered table
    /// axis chosen by `gathered_dim` instead of hardcoded to `0` --
    /// `gathered_dim: 1` exercises a table laid out `[kept, gathered]`
    /// instead of `[gathered, kept]`, proving the typed cursor's
    /// `element_stride`/`extent` derivation (from [`bind::bind`], unmodified
    /// by this change) is honoured regardless of which axis is gathered.
    fn typed_gather_dim_program(
        compute_dtype: DType,
        index_dtype: DType,
        table_shape: [u32; 2],
        seq: u32,
        gathered_dim: u16,
    ) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let table = block(
            &mut program,
            compute_dtype,
            &[Extent::Static(table_shape[0]), Extent::Static(table_shape[1])],
        );
        let ids = block(&mut program, index_dtype, &[Extent::Static(seq)]);
        let kept_dim = 1 - gathered_dim;
        let mut axes = alloc::vec![map::AxisIndex::default(); 2];
        axes[kept_dim as usize] = map::AxisIndex {
            terms: core::iter::once(AxisTerm::projection(1)).collect(),
            offset: 0,
        };
        let gathered_map = IndexMap::Computed {
            indices: ids,
            index_map: map::projection(2, &[0]),
            base: map::IndexPattern { iter_rank: 2, axes },
            gathered_dim,
        };
        let gathered = append(
            &mut program,
            Op::Elementwise {
                dtype: compute_dtype,
                body: ScalarOp::Identity,
                operands: alloc::vec![(table, gathered_map)],
                name: None,
            },
        );
        (program, gathered)
    }

    /// A row-sum reduce, accumulated at `accumulator_dtype`, over a table
    /// gathered at `operand_dtype` -- [`typed_widened_reduce_program`]'s
    /// shape composed with a real gather instead of a plain block operand,
    /// proving `TypedPlan::Widened` executes correctly when its own operand
    /// is data-dependent.
    fn typed_widened_gather_reduce_program(
        operand_dtype: DType,
        accumulator_dtype: DType,
        index_dtype: DType,
        vocab: u32,
        dim: u32,
        seq: u32,
    ) -> (Vec<Op>, NodeId) {
        let (mut program, gathered) = typed_embedding_lookup_program(operand_dtype, index_dtype, vocab, dim, seq);
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: accumulator_dtype,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: gathered,
                in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                out_map: IndexMap::Affine(map::projection(2, &[0])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, sum)
    }

    /// [`typed_program_plan`]'s third role, executed: an `i32`- or
    /// `u32`-index gather over an `f32` compute table must produce exactly
    /// the same bytes [`embedding_lookup_program`]'s f32 pipeline does for
    /// the same table and the same row selection -- the incumbent-parity
    /// bar (guiding-principles §14): the f32 evaluator is the oracle, and
    /// any divergence is this evaluator's bug until proven otherwise.
    /// Covers a repeated index (row 3 selected twice) and both boundary
    /// indices (`0` and `vocab - 1`).
    #[proxima::test]
    #[case::i32_index(DType::Int32)]
    #[case::u32_index(DType::UInt32)]
    async fn typed_gather_matches_f32_oracle_element_for_element(#[case] index_dtype: DType) {
        let (vocab, dim, seq) = (50usize, 6usize, 5usize);
        let (f32_program, _) = embedding_lookup_program(vocab as u32, dim as u32, seq as u32);
        let table_data: Vec<f32> = (0..vocab * dim).map(|value| (value % 37) as f32 - 10.0).collect();
        // row 3 repeated, plus both boundary rows (0 and vocab - 1).
        let ids_f32 = [3.0f32, (vocab - 1) as f32, 0.0, 3.0, 25.0];
        let oracle = evaluate(&f32_program, &[], &[&table_data, &ids_f32], &[]).expect("f32 oracle evaluates");

        let (typed_program, _) =
            typed_embedding_lookup_program(DType::Float32, index_dtype, vocab as u32, dim as u32, seq as u32);
        let ids_block = match index_dtype {
            DType::Int32 => TypedBuffer::Int32(ids_f32.iter().map(|&value| value as i32).collect()),
            DType::UInt32 => TypedBuffer::UInt32(ids_f32.iter().map(|&value| value as u32).collect()),
            other => panic!("unexpected index dtype in case table: {other:?}"),
        };
        let blocks = [TypedBuffer::Float32(table_data.clone()), ids_block];
        let results =
            evaluate_typed(&typed_program, &[], &blocks, &[]).expect("typed gather evaluates against real data");
        let TypedBuffer::Float32(got) = &results[0].2 else {
            panic!("expected an f32 result buffer");
        };
        assert_eq!(
            got.as_slice(),
            oracle.root(),
            "typed {index_dtype:?}-index gather must match the f32 oracle element-for-element"
        );
    }

    /// The same oracle-parity bar as
    /// [`typed_gather_matches_f32_oracle_element_for_element`], with the
    /// compute dtype narrowed to `f16` -- exact equality no longer holds
    /// (the table itself round-trips through half precision before the
    /// gather ever runs), so the bound is half a step at the table's own
    /// magnitude instead.
    #[test]
    fn typed_gather_f16_compute_matches_f32_oracle_within_half_precision() {
        let (vocab, dim, seq) = (32usize, 4usize, 6usize);
        let (f32_program, _) = embedding_lookup_program(vocab as u32, dim as u32, seq as u32);
        let table_f32: Vec<f32> = (0..vocab * dim).map(|value| (value % 23) as f32 - 5.0).collect();
        let ids_f32 = [0.0f32, (vocab - 1) as f32, 7.0, 7.0, 15.0, 31.0];
        let oracle = evaluate(&f32_program, &[], &[&table_f32, &ids_f32], &[]).expect("f32 oracle evaluates");

        let (typed_program, _) =
            typed_embedding_lookup_program(DType::Float16, DType::Int32, vocab as u32, dim as u32, seq as u32);
        let table_f16: Vec<f16> = table_f32.iter().map(|&value| f16::from_f32(value)).collect();
        let ids_i32: Vec<i32> = ids_f32.iter().map(|&value| value as i32).collect();
        let blocks = [TypedBuffer::Float16(table_f16), TypedBuffer::Int32(ids_i32)];
        let results = evaluate_typed(&typed_program, &[], &blocks, &[]).expect("f16 typed gather evaluates");
        let TypedBuffer::Float16(got) = &results[0].2 else {
            panic!("expected an f16 result buffer");
        };
        for (value, expected) in got.iter().zip(oracle.root()) {
            assert!(
                (value.to_f32() - expected).abs() < 5e-2,
                "f16 gather {} vs f32 oracle {expected}",
                value.to_f32()
            );
        }
    }

    /// [`typed_gather_dim_program`]'s `gathered_dim: 1` shape against the
    /// same f32 oracle: the table is laid out `[dim, vocab]` (transposed
    /// relative to the `gathered_dim: 0` tests above) so the gather selects
    /// a *column*, not a row, proving the typed cursor does not assume the
    /// gathered axis is the table's leading one.
    #[test]
    fn typed_gather_dim1_matches_f32_oracle() {
        let (dim, vocab, seq) = (5usize, 20usize, 4usize);
        let (f32_program, _) = typed_gather_dim_program(DType::Float32, DType::Int32, [dim as u32, vocab as u32], seq as u32, 1);
        let table_data: Vec<f32> = (0..dim * vocab).map(|value| (value % 17) as f32 + 1.0).collect();
        let ids_f32 = [0.0f32, (vocab - 1) as f32, 9.0, 9.0];
        let oracle = evaluate(&f32_program, &[], &[&table_data, &ids_f32], &[]).expect("f32 oracle evaluates");

        let (typed_program, _) = typed_gather_dim_program(
            DType::Float32,
            DType::UInt32,
            [dim as u32, vocab as u32],
            seq as u32,
            1,
        );
        let ids_u32: Vec<u32> = ids_f32.iter().map(|&value| value as u32).collect();
        let blocks = [TypedBuffer::Float32(table_data.clone()), TypedBuffer::UInt32(ids_u32)];
        let results = evaluate_typed(&typed_program, &[], &blocks, &[]).expect("gathered_dim: 1 typed gather evaluates");
        let TypedBuffer::Float32(got) = &results[0].2 else {
            panic!("expected an f32 result buffer");
        };
        assert_eq!(
            got.as_slice(),
            oracle.root(),
            "a gathered_dim: 1 typed gather must match the f32 oracle element-for-element"
        );
    }

    /// A widened reduce ([`TypedPlan::Widened`]) whose own operand is a
    /// gathered `f16` table folded into an `f32` accumulator -- proves the
    /// two features compose: [`run_widened_program`]'s `TIn`/`TAcc` table
    /// split and [`canonical_index_buffers`]'s separate `i64` index table
    /// both apply to the same node without interfering.
    #[test]
    fn widened_reduce_over_a_gathered_f16_operand_matches_a_hand_written_reference() {
        let (vocab, dim, seq) = (10usize, 4usize, 3usize);
        let table_f32: Vec<f32> = (0..vocab * dim).map(|value| (value % 13) as f32 - 6.0).collect();
        let ids = [2u32, 9, 0];

        let mut reference = alloc::vec![0.0f32; seq];
        for (row, &id) in ids.iter().enumerate() {
            let row_start = id as usize * dim;
            reference[row] = table_f32[row_start..row_start + dim]
                .iter()
                .map(|&value| f16::from_f32(value).to_f32())
                .sum();
        }

        let (program, _) = typed_widened_gather_reduce_program(
            DType::Float16,
            DType::Float32,
            DType::UInt32,
            vocab as u32,
            dim as u32,
            seq as u32,
        );
        let table_f16: Vec<f16> = table_f32.iter().map(|&value| f16::from_f32(value)).collect();
        let blocks = [TypedBuffer::Float16(table_f16), TypedBuffer::UInt32(ids.to_vec())];
        let results = evaluate_typed(&program, &[], &blocks, &[]).expect("widened reduce over a gather evaluates");
        let TypedBuffer::Float32(got) = &results[0].2 else {
            panic!("expected an f32 accumulator buffer");
        };
        for (value, expected) in got.iter().zip(&reference) {
            assert!(
                (value - expected).abs() < 5e-2,
                "widened gathered reduce {value} vs reference {expected}"
            );
        }
    }

    /// The sad path a real gather program (not just [`typed_program_plan`])
    /// still honours: a float-dtype index node is rejected before any
    /// buffer is touched, the same gate
    /// [`typed_program_plan_rejects_a_float_gather_index_dtype`] proves at
    /// the plan level alone.
    #[test]
    fn evaluate_typed_rejects_a_float_gather_index_dtype_at_execution() {
        let program = typed_gather_program(DType::Float32, DType::Float64);
        let error = evaluate_typed(&program, &[], &[], &[])
            .expect_err("a float-dtype gather index must be rejected at execution, never silently accepted");
        assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
    }

    /// The f32 oracle's own out-of-range behaviour
    /// ([`a_fetched_index_past_the_extent_is_a_real_error_not_ub`]): a
    /// fetched index `>= extent` is `TensorError::GatherIndexOutOfRange`,
    /// never a clamp, a wraparound, or UB. The typed evaluator must answer
    /// the identical class of index for the identical class of input,
    /// proven for both the positive-overflow and the negative-index cases
    /// [`GatherCursor::fetch_and_advance`] (`proxima-tensor/src/cpu.rs`)
    /// checks in one `index < 0 || index as u64 >= self.extent` guard.
    #[proxima::test]
    #[case::index_past_the_extent(4)]
    #[case::negative_index(-1)]
    async fn typed_gather_out_of_range_index_matches_f32_oracle_error_shape(#[case] bad_index: i32) {
        let (vocab, dim, seq) = (4usize, 2usize, 1usize);
        let (f32_program, _) = embedding_lookup_program(vocab as u32, dim as u32, seq as u32);
        let table_data: Vec<f32> = (0..vocab * dim).map(|value| value as f32).collect();
        let ids_f32 = [bad_index as f32];
        let oracle_error = evaluate(&f32_program, &[], &[&table_data, &ids_f32], &[])
            .expect_err("the f32 oracle rejects the out-of-range index");
        let TensorError::GatherIndexOutOfRange { extent: oracle_extent, .. } = oracle_error else {
            panic!("expected the f32 oracle's own GatherIndexOutOfRange, got {oracle_error}");
        };

        let (typed_program, _) =
            typed_embedding_lookup_program(DType::Float32, DType::Int32, vocab as u32, dim as u32, seq as u32);
        let blocks = [TypedBuffer::Float32(table_data), TypedBuffer::Int32(alloc::vec![bad_index])];
        let typed_error =
            evaluate_typed(&typed_program, &[], &blocks, &[]).expect_err("the typed evaluator rejects it too");
        let TensorError::GatherIndexOutOfRange {
            index: typed_index,
            extent: typed_extent,
            ..
        } = typed_error
        else {
            panic!("expected GatherIndexOutOfRange, got {typed_error}");
        };
        assert_eq!(typed_index, i64::from(bad_index));
        assert_eq!(typed_extent, oracle_extent, "the typed evaluator must bounds-check against the same extent");
    }

    /// The honest boundary [`canonical_index_buffers`] draws rather than
    /// guessing: a gather index node computed in-program (an [`Op::Iota`]
    /// here, not a caller-supplied block) is a named `NotLowerable`, not a
    /// silently wrong execution — see [`canonical_index_buffers`]'s own doc.
    #[test]
    fn evaluate_typed_names_a_computed_gather_index_node_as_not_yet_supported() {
        let mut program = Vec::new();
        let table = block(&mut program, DType::Float32, &[Extent::Static(4), Extent::Static(2)]);
        let ids = append(&mut program, Op::Iota { dtype: DType::Int32, extent: Extent::Static(3) });
        let gathered_map = IndexMap::Computed {
            indices: ids,
            index_map: map::projection(2, &[0]),
            base: map::IndexPattern {
                iter_rank: 2,
                axes: alloc::vec![
                    map::AxisIndex::default(),
                    map::AxisIndex {
                        terms: core::iter::once(AxisTerm::projection(1)).collect(),
                        offset: 0,
                    },
                ],
            },
            gathered_dim: 0,
        };
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(table, gathered_map)],
                name: None,
            },
        );
        let table_data = TypedBuffer::Float32(alloc::vec![0.0; 8]);
        let error = evaluate_typed(&program, &[], &[table_data], &[])
            .expect_err("a computed gather index node must be a named gap, not a silent guess");
        assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
    }

    #[test]
    fn i16_operand_i64_accumulator_reduce_survives_i16_overflow() {
        // three i16 elements of 20000 each: the true sum is 60000, which
        // does not fit in i16 (max 32767) -- wrapping i16 arithmetic would
        // land on -5536 (60000 - 65536). an i64 accumulator is the only way
        // to observe 60000.
        let (program, _) = typed_widened_reduce_program(DType::Int16, DType::Int64, 3);
        let operand = TypedBuffer::Int16(alloc::vec![20000, 20000, 20000]);
        let results = evaluate_typed(&program, &[], &[operand], &[])
            .expect("an i16-operand, i64-accumulator reduce evaluates");
        assert_eq!(
            results[0].2,
            TypedBuffer::Int64(alloc::vec![60000]),
            "the i64 accumulator must carry the true sum, not the i16-wrapped one (-5536)"
        );
    }

    #[test]
    fn u8_operand_u32_accumulator_reduce_survives_u8_overflow() {
        // three u8 elements of 200 each: the true sum is 600, which does
        // not fit in u8 (max 255) -- wrapping u8 arithmetic would land on
        // 88 (600 - 512). a u32 accumulator is the only way to observe 600.
        let (program, _) = typed_widened_reduce_program(DType::UInt8, DType::UInt32, 3);
        let operand = TypedBuffer::UInt8(alloc::vec![200, 200, 200]);
        let results = evaluate_typed(&program, &[], &[operand], &[])
            .expect("a u8-operand, u32-accumulator reduce evaluates");
        assert_eq!(
            results[0].2,
            TypedBuffer::UInt32(alloc::vec![600]),
            "the u32 accumulator must carry the true sum, not the u8-wrapped one (88)"
        );
    }

    fn typed_reduce_vector_to_scalar_program(dtype: DType, len: u32) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let operand = block(&mut program, dtype, &[Extent::Static(len)]);
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, sum)
    }

    #[proxima::test]
    #[case::int32(DType::Int32, TypedBuffer::Int32(alloc::vec![1, 2, 3, 4]), TypedBuffer::Int32(alloc::vec![10]))]
    #[case::uint64(DType::UInt64, TypedBuffer::UInt64(alloc::vec![1, 2, 3, 4]), TypedBuffer::UInt64(alloc::vec![10]))]
    #[case::float64(DType::Float64, TypedBuffer::Float64(alloc::vec![1.5, 2.5, 3.0, 4.0]), TypedBuffer::Float64(alloc::vec![11.0]))]
    async fn evaluate_typed_reduces_a_vector_to_a_scalar_across_widths(
        #[case] dtype: DType,
        #[case] operand: TypedBuffer,
        #[case] expected: TypedBuffer,
    ) {
        let (program, _) = typed_reduce_vector_to_scalar_program(dtype, 4);
        let results =
            evaluate_typed(&program, &[], &[operand], &[]).expect("typed reduce evaluates");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].2, expected);
    }

    #[test]
    fn evaluate_typed_scans_an_integer_vector_producing_a_running_sum() {
        let mut program = Vec::new();
        let source = block(&mut program, DType::Int32, &[Extent::Static(5)]);
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Int32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[0])),
                keep: Keep::Scan,
                name: None,
            }),
        );
        let blocks = [TypedBuffer::Int32(alloc::vec![1, 2, 3, 4, 5])];
        let results = evaluate_typed(&program, &[], &blocks, &[]).expect("typed scan evaluates");
        assert_eq!(results[0].2, TypedBuffer::Int32(alloc::vec![1, 3, 6, 10, 15]));
    }

    /// Matmul-shaped: a `Multiply` elementwise body fused into an `Add`
    /// reduce, same construction as [`matmul_program`] with `dtype`
    /// parameterized so it can run through [`evaluate_typed`] at any width.
    fn typed_matmul_program(
        dtype: DType,
        m: u32,
        k: u32,
        n: u32,
    ) -> (Vec<Op>, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let lhs = block(&mut program, dtype, &[Extent::Static(m), Extent::Static(k)]);
        let rhs = block(&mut program, dtype, &[Extent::Static(k), Extent::Static(n)]);
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
                ],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: Some("typed_matmul".into()),
            }),
        );
        (program, lhs, rhs, sum)
    }

    #[test]
    fn evaluate_typed_matmul_shaped_reduce_matches_a_naive_reference_at_int32() {
        let (m, k, n) = (3usize, 4usize, 2usize);
        let (program, _, _, _) = typed_matmul_program(DType::Int32, m as u32, k as u32, n as u32);
        let lhs: Vec<i32> = (0..(m * k) as i32).collect();
        let rhs: Vec<i32> = (0..(k * n) as i32).collect();
        let blocks = [
            TypedBuffer::Int32(lhs.clone()),
            TypedBuffer::Int32(rhs.clone()),
        ];
        let results = evaluate_typed(&program, &[], &blocks, &[]).expect("typed matmul evaluates");

        let mut expected = vec![0i32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0i32;
                for inner in 0..k {
                    sum += lhs[row * k + inner] * rhs[inner * n + col];
                }
                expected[row * n + col] = sum;
            }
        }
        assert_eq!(results[0].2, TypedBuffer::Int32(expected));
    }

    #[test]
    fn evaluate_typed_matmul_shaped_reduce_matches_a_naive_reference_at_float64() {
        let (m, k, n) = (3usize, 4usize, 2usize);
        let (program, _, _, _) =
            typed_matmul_program(DType::Float64, m as u32, k as u32, n as u32);
        let lhs: Vec<f64> = (0..m * k).map(|value| value as f64 * 0.5).collect();
        let rhs: Vec<f64> = (0..k * n).map(|value| value as f64 * 0.25).collect();
        let blocks = [
            TypedBuffer::Float64(lhs.clone()),
            TypedBuffer::Float64(rhs.clone()),
        ];
        let results = evaluate_typed(&program, &[], &blocks, &[]).expect("typed matmul evaluates");

        let mut expected = vec![0.0f64; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f64;
                for inner in 0..k {
                    sum += lhs[row * k + inner] * rhs[inner * n + col];
                }
                expected[row * n + col] = sum;
            }
        }
        let TypedBuffer::Float64(actual) = &results[0].2 else {
            panic!("expected a Float64 result");
        };
        for (found, expect) in actual.iter().zip(expected.iter()) {
            assert!((found - expect).abs() < 1e-9, "{found} vs {expect}");
        }
    }

    /// `T = f32` is the specialization [`run_reduce_typed`] delegates
    /// straight back to the existing NEON-tiled [`run_reduce`] — this checks
    /// [`evaluate_typed`] and [`evaluate`] agree bit-for-bit on the exact
    /// same matmul-shaped program, which they only can if both ran the same
    /// function. (Whether the NEON tile itself fired, as opposed to one of
    /// `run_reduce`'s other f32 fast paths, is checked separately by
    /// `evaluate_typed_float32_matmul_shaped_reduce_fires_the_neon_tile`,
    /// gated on `feature = "instrument"`.)
    #[test]
    fn evaluate_typed_float32_matmul_shaped_reduce_matches_evaluate_bit_for_bit() {
        let (m, k, n) = (6usize, 32usize, 8usize);
        let (program, _, _, _) =
            typed_matmul_program(DType::Float32, m as u32, k as u32, n as u32);
        let lhs: Vec<f32> = (0..m * k).map(|value| (value as f32 * 0.0137).sin()).collect();
        let rhs: Vec<f32> = (0..k * n).map(|value| (value as f32 * 0.0271).cos()).collect();

        let via_evaluate = evaluate(&program, &[], &[&lhs, &rhs], &[]).expect("f32 matmul evaluates");
        let blocks = [TypedBuffer::Float32(lhs), TypedBuffer::Float32(rhs)];
        let via_typed =
            evaluate_typed(&program, &[], &blocks, &[]).expect("typed f32 matmul evaluates");
        let TypedBuffer::Float32(typed_data) = &via_typed[0].2 else {
            panic!("expected a Float32 result");
        };
        assert_eq!(typed_data.as_slice(), via_evaluate.root());
    }

    /// Same claim as the test above, but over the RHS-transposed layout
    /// that actually engages `neon_tile_plan`/`gemm_tile_neon` (see
    /// `evaluate_typed_float32_matmul_shaped_reduce_fires_the_neon_tile`'s
    /// doc on why plain `matmul_program`'s layout hits `width_tile_plan`
    /// instead), with a contraction (`k = 64`) long enough for the NEON
    /// tile's own row/lane splitting to matter, and compared via `to_bits`
    /// rather than `==` — the generic nest and the NEON nest accumulate in
    /// a different order, so a fallthrough from one to the other changes
    /// bits even where it would not change `==` (e.g. `-0.0` vs `0.0`).
    /// No feature gate: this is the check that fails the *default* gate if
    /// `run_reduce_typed`'s `T == f32` specialization silently stops firing,
    /// unlike `..._fires_the_neon_tile` below, which only runs under
    /// `instrument` and checks the counters instead of the bits.
    #[test]
    fn evaluate_typed_float32_matmul_rhs_transposed_matches_evaluate_bit_for_bit() {
        let (m, k, n) = (12usize, 64usize, 8usize);
        let (program, _) = matmul_program_rhs_transposed(m as u32, k as u32, n as u32);
        let lhs = random_vec(0x1234_5678_9abc_def0, m * k);
        let rhs = random_vec(0x0fed_cba9_8765_4321, n * k);

        let via_evaluate =
            evaluate(&program, &[], &[&lhs, &rhs], &[]).expect("f32 matmul evaluates");
        let blocks = [TypedBuffer::Float32(lhs), TypedBuffer::Float32(rhs)];
        let via_typed =
            evaluate_typed(&program, &[], &blocks, &[]).expect("typed f32 matmul evaluates");
        let TypedBuffer::Float32(typed_data) = &via_typed[0].2 else {
            panic!("expected a Float32 result");
        };

        assert_eq!(typed_data.len(), via_evaluate.root().len());
        let compared = typed_data.len();
        assert!(compared > 0, "the bit-identity check compared zero elements");
        for (index, (found, expected)) in typed_data.iter().zip(via_evaluate.root()).enumerate() {
            assert_eq!(
                found.to_bits(),
                expected.to_bits(),
                "node {index}: evaluate_typed produced {found} (bits {:#010x}), \
                 evaluate produced {expected} (bits {:#010x})",
                found.to_bits(),
                expected.to_bits(),
            );
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    fn evaluate_typed_float32_matmul_shaped_reduce_fires_the_neon_tile() {
        // `neon_tile_plan`'s gate (cpu.rs's own doc on that function) wants
        // both contraction strides == 1 with the *width* dim non-contiguous
        // on one operand — the RHS-transposed layout
        // `matmul_program_rhs_transposed` uses, not plain `matmul_program`'s
        // (whose RHS is `[k, n]` contiguous in `n` and hits `width_tile_plan`
        // instead). Mirrored here rather than reusing `typed_matmul_program`.
        let (m, k, n) = (12usize, 64usize, 8usize);
        let mut program = Vec::new();
        let lhs = block(&mut program, DType::Float32, &[Extent::Static(m as u32), Extent::Static(k as u32)]);
        let rhs = block(&mut program, DType::Float32, &[Extent::Static(n as u32), Extent::Static(k as u32)]);
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (rhs, IndexMap::Affine(map::projection(3, &[1, 2]))),
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
                name: Some("typed_matmul_rhs_transposed".into()),
            }),
        );
        let lhs_data: Vec<f32> = (0..m * k).map(|value| (value as f32 * 0.0137).sin()).collect();
        let rhs_data: Vec<f32> = (0..n * k).map(|value| (value as f32 * 0.0271).cos()).collect();
        let blocks = [TypedBuffer::Float32(lhs_data), TypedBuffer::Float32(rhs_data)];

        let (gate_before, invocations_before, _) = neon_tile_counters();
        evaluate_typed(&program, &[], &blocks, &[]).expect("typed f32 matmul evaluates");
        let (gate_after, invocations_after, _) = neon_tile_counters();

        assert!(gate_after > gate_before, "neon_tile_plan never matched through evaluate_typed");
        assert!(
            invocations_after > invocations_before,
            "gemm_tile_neon never ran through evaluate_typed"
        );
    }

    #[test]
    fn evaluate_typed_rejects_a_block_whose_dtype_does_not_match_the_program() {
        let (program, _, _, _) = typed_add_program(DType::Int32, 2);
        let blocks = [
            TypedBuffer::Int32(alloc::vec![1, 2]),
            TypedBuffer::Int8(alloc::vec![1, 2]),
        ];
        let error = evaluate_typed(&program, &[], &blocks, &[])
            .expect_err("Int8 block cannot bind an Int32 program");
        assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
    }

    // -- operand_access_footprint: pure, always compiled under `cfg(test)`
    // regardless of the `instrument` feature, so these run in every default
    // `cargo nextest run -p proxima-tensor` too, not only `--features
    // instrument` (see the function's own `#[cfg(any(feature = "instrument",
    // test))]`).

    #[test]
    fn operand_access_footprint_is_exact_for_a_dense_operand() {
        let extents = [4u64, 5, 6];
        let strides = [30i64, 6, 1];
        let (reads, distinct) = operand_access_footprint(&extents, &strides);
        assert_eq!(reads, 4 * 5 * 6);
        assert_eq!(distinct, reads, "a dense operand's own footprint is read exactly once per position");
    }

    #[test]
    fn operand_access_footprint_undercounts_distinct_under_broadcast() {
        let extents = [4u64, 5, 6];
        let strides = [30i64, 0, 1]; // broadcast over the middle axis
        let (reads, distinct) = operand_access_footprint(&extents, &strides);
        assert_eq!(reads, 4 * 5 * 6, "every iterated position still counts as a read");
        assert_eq!(distinct, 4 * 6, "the broadcast axis contributes 1, not its extent, to distinct");
        assert!(reads > distinct);
    }

    #[test]
    fn operand_access_footprint_is_one_for_a_scalar_reduction() {
        assert_eq!(operand_access_footprint(&[], &[]), (1, 1));
    }

    // -- proof tests: the instrument must measure something real, not just
    // increment. Each asserts a number known a priori from the program's
    // own construction, gated behind `instrument` since the API under test
    // only exists there.

    /// A matmul's RHS is read once per `(m, n, k)` position but only ever
    /// resolves to `k * n` distinct elements — it never varies along `m`.
    /// Asserts `reads >> distinct`, the shape a cold-weight quantization
    /// decision needs.
    #[test]
    #[cfg(feature = "instrument")]
    fn evaluate_records_more_reads_than_distinct_elements_for_a_broadcast_operand() {
        instrument::reset_operand_access();
        let (m, k, n) = (4u32, 3u32, 5u32);
        let (program, sum) = matmul_program(m, k, n, false);
        let lhs_data = random_vec(1, (m * k) as usize);
        let rhs_data = random_vec(2, (k * n) as usize);
        evaluate(&program, &[], &[&lhs_data, &rhs_data], &[sum]).expect("matmul evaluates");

        let lhs_access = instrument::operand_access_of(NodeId(0)).expect("lhs was instrumented");
        assert_eq!(
            lhs_access.distinct_elements,
            u64::from(m) * u64::from(k),
            "lhs's real footprint excludes the n broadcast axis"
        );
        assert!(
            lhs_access.reads > lhs_access.distinct_elements,
            "reads={} distinct={}",
            lhs_access.reads,
            lhs_access.distinct_elements
        );

        let rhs_access = instrument::operand_access_of(NodeId(1)).expect("rhs was instrumented");
        assert_eq!(rhs_access.distinct_elements, u64::from(k) * u64::from(n));
        assert!(
            rhs_access.reads > rhs_access.distinct_elements,
            "reads={} distinct={}",
            rhs_access.reads,
            rhs_access.distinct_elements
        );
    }

    /// The case that matters most: an embedding lookup into a 1000-row
    /// table, fetching only 3 distinct rows across 6 positions (each row
    /// hit twice). A naive "count every read" instrument would report all
    /// 1000 rows touched (or the raw read count); this must report exactly
    /// 3 rows' worth of elements.
    #[test]
    #[cfg(feature = "instrument")]
    fn evaluate_records_exactly_the_distinct_rows_a_gather_touches_not_the_whole_table() {
        instrument::reset_operand_access();
        let (vocab, dim, seq) = (1_000u32, 8u32, 6u32);
        let (program, gathered) = embedding_lookup_program(vocab, dim, seq);
        let table_data: Vec<f32> = (0..(vocab * dim) as usize).map(|value| value as f32).collect();
        // 6 fetches, 3 distinct rows: 3 and 999 and 500 each hit twice.
        let ids_data = [3.0f32, 3.0, 999.0, 999.0, 500.0, 500.0];
        evaluate(&program, &[], &[&table_data, &ids_data], &[gathered]).expect("gather evaluates");

        let table_access = instrument::operand_access_of(NodeId(0)).expect("table was instrumented");
        assert_eq!(
            table_access.distinct_elements,
            3 * u64::from(dim),
            "only 3 of {vocab} rows were ever fetched, not the whole table"
        );
        assert_eq!(table_access.total_elements, u64::from(vocab) * u64::from(dim));
        assert!(table_access.distinct_elements < table_access.total_elements);
    }

    /// Degenerate control: an operand the requested outputs never reach
    /// gets no `BoundOp` at all, so it must read back `None` — absent, not
    /// a zero-touch row assumed on its behalf. Paired with a direct API
    /// check that a REAL zero-read record (an operand that was reached, and
    /// genuinely read zero times) reads back `Some` with every field `0`,
    /// so the two "zero" cases are distinguishable rather than folded
    /// together.
    #[test]
    #[cfg(feature = "instrument")]
    fn operand_access_distinguishes_never_read_from_a_recorded_zero() {
        instrument::reset_operand_access();
        let (m, k, n) = (2u32, 2u32, 2u32);
        let (mut program, sum) = matmul_program(m, k, n, false);
        let unused = f32_block(&mut program, &[Extent::Static(3)]);
        let lhs_data = random_vec(1, (m * k) as usize);
        let rhs_data = random_vec(2, (k * n) as usize);
        let unused_data = alloc::vec![0.0f32; 3];
        evaluate(&program, &[], &[&lhs_data, &rhs_data, &unused_data], &[sum]).expect("matmul evaluates");

        assert!(instrument::operand_access_of(NodeId(0)).is_some(), "lhs was actually read");
        assert_eq!(
            instrument::operand_access_of(unused),
            None,
            "an operand the requested outputs never reach is absent, not a recorded zero"
        );

        instrument::reset_operand_access();
        let node = NodeId(42);
        assert_eq!(instrument::operand_access_of(node), None, "nothing recorded yet");
        instrument::record_operand_access(node, 0, 0, 128);
        let access = instrument::operand_access_of(node).expect("recording zero reads still creates a row");
        assert_eq!(access.reads, 0);
        assert_eq!(access.distinct_elements, 0);
        assert_eq!(access.total_elements, 128, "total size is known even when nothing was ever read");
    }

    /// DLMF 7.2 / Abramowitz & Stegun Table 7.1's published `erf(x)`, to the
    /// precision commonly republished — the oracle `erf_f32_matches_reference_values`
    /// and `erf_f64_matches_reference_values` sweep against, independent of
    /// this crate's own approximation.
    const ERF_REFERENCE: &[(f64, f64)] = &[
        (0.0, 0.0),
        (0.2, 0.222_702_589_2),
        (0.4, 0.428_392_355_0),
        (0.6, 0.603_856_090_8),
        (0.8, 0.742_100_964_7),
        (1.0, 0.842_700_792_9),
        (1.2, 0.910_313_978_2),
        (1.4, 0.952_285_119_8),
        (1.6, 0.976_348_383_3),
        (1.8, 0.989_090_501_6),
        (2.0, 0.995_322_265_0),
        (2.5, 0.999_593_048_0),
        (3.0, 0.999_977_909_5),
        (5.0, 1.0),
    ];

    /// Abramowitz & Stegun 7.1.26's published maximum absolute error is
    /// `1.5e-7`. Measured here against [`ERF_REFERENCE`] (swept across both
    /// signs via `erf_f32(-x) == -erf_f32(x)`), the actual max error is
    /// `1.1920929e-7`, equal to `f32::EPSILON` (`2^-23`) exactly — this
    /// approximation is at, not below, the "below the type's own epsilon"
    /// bar for `f32`; it does not clear it outright, but it also is not
    /// dominated by formula error rather than `f32`'s own rounding.
    #[test]
    fn erf_f32_matches_reference_values_within_f32_epsilon() {
        let mut max_error = 0.0f32;
        for &(x, reference) in ERF_REFERENCE {
            let (x, reference) = (x as f32, reference as f32);
            let positive_error = (erf_f32(x) - reference).abs();
            let negative_error = (erf_f32(-x) - (-reference)).abs();
            max_error = max_error.max(positive_error).max(negative_error);
        }
        assert!(
            max_error <= 1.5 * f32::EPSILON,
            "measured max abs error {max_error} should stay within 1.5x f32::EPSILON ({}); the \
             published Abramowitz & Stegun bound is 1.5e-7, essentially f32::EPSILON itself",
            f32::EPSILON
        );
    }

    /// Same formula, same reference table, `f64` throughout: isolates how
    /// much of `erf_f32`'s error is the formula itself versus f32 rounding
    /// compounding on top of it.
    #[test]
    fn erf_f64_matches_reference_values() {
        let mut max_error = 0.0f64;
        for &(x, reference) in ERF_REFERENCE {
            let positive_error = (erf_f64(x) - reference).abs();
            let negative_error = (erf_f64(-x) - (-reference)).abs();
            max_error = max_error.max(positive_error).max(negative_error);
        }
        assert!(
            max_error < 2e-7,
            "measured f64 max abs error {max_error} exceeds the ~1.5e-7 published bound"
        );
    }

    /// `elementwise_width_unary`'s dispatch table (the fast path
    /// `evaluate`/`evaluate_parallel` actually run through) used to end in
    /// `_ => unreachable!("BodyShape::Unary only ever carries an arity-1
    /// ScalarOp")` — a real panic, not a fallback, for any arity-1
    /// `ScalarOp` this match does not name explicitly. Evaluating a real
    /// `Op::Elementwise { body: ScalarOp::Erf, .. }` program end to end is
    /// what proves `Erf`'s arm was actually added there, not merely to
    /// `apply_scalar_op`'s slow general path.
    #[test]
    fn erf_evaluates_through_a_real_elementwise_program() {
        let mut program = Vec::new();
        let input = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let output = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Erf,
                operands: alloc::vec![(input, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );

        let values: [f32; 4] = [0.0, 0.5, 1.0, -1.5];
        let blocks: [&[f32]; 1] = [&values];
        let evaluated = evaluate(&program, &[], &blocks, &[output]).expect("erf elementwise evaluates");

        let found = evaluated.root();
        assert_eq!(found.len(), values.len());
        for (result, &raw_value) in found.iter().zip(values.iter()) {
            let expected = erf_f32(raw_value);
            assert!(
                (result - expected).abs() < 1e-6,
                "elementwise erf({raw_value}) = {result}, direct erf_f32 gives {expected}"
            );
        }
    }

    /// [`matmul_q4k_f32`] against the incumbent: quantize a random weight
    /// matrix, then compare its output row-for-row to plain
    /// dequantize-then-`f32`-dot-product on the same bytes. Both paths
    /// call the identical [`proxima_gguf::quant::q4_k::dequantize_block`]
    /// codec, so the only source of disagreement is accumulation order —
    /// this path folds with [`f32::mul_add`] one super-block at a time
    /// ([`dot_q4k_f32`]), the reference sums a plain `iter().zip().map()`
    /// over the fully-dequantized row — so a nonzero difference is
    /// expected (guiding-principle 14/19: report the measured number,
    /// never assert bit-exact equality here).
    #[test]
    fn matmul_q4k_f32_agrees_with_dequantize_then_matmul_within_a_measured_tolerance() {
        use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};

        let rows = 5;
        let blocks_per_row = 3;
        let k = QK_K * blocks_per_row;

        // realistic weight-scale values, not degenerate all-zero/constant
        // inputs — `Lcg::next_unit` is already this file's own random-f32
        // fixture generator (see `random_vec` above).
        let activation: Vec<f32> = random_vec(7, k).into_iter().map(|value| value * 4.0 - 2.0).collect();
        let weight_f32: Vec<f32> = random_vec(11, rows * k).into_iter().map(|value| value * 4.0 - 2.0).collect();

        let mut weight_blocks = vec![0u8; rows * blocks_per_row * BLOCK_BYTES];
        for (row_f32, row_blocks) in weight_f32
            .chunks_exact(k)
            .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
        {
            quantize(row_f32, row_blocks).expect("row length is a whole multiple of QK_K by construction");
        }

        // incumbent: dequantize the packed bytes back to f32, then a plain
        // dot product per row — never touches `dot_q4k_f32`/`matmul_q4k_f32`.
        let mut expected = Vec::with_capacity(rows);
        for row_blocks in weight_blocks.chunks_exact(blocks_per_row * BLOCK_BYTES) {
            let mut dequantized = vec![0.0f32; k];
            dequantize(row_blocks, &mut dequantized).expect("row_blocks is a whole number of q4_k super-blocks");
            let dot: f32 = dequantized.iter().zip(activation.iter()).map(|(&weight, &value)| weight * value).sum();
            expected.push(dot);
        }

        let actual = matmul_q4k_f32(&weight_blocks, rows, &activation).expect("well-formed quantized matmul");

        assert_eq!(actual.len(), expected.len());
        let mut max_error = 0.0f32;
        let mut sum_sq_error = 0.0f64;
        for (&got, &want) in actual.iter().zip(expected.iter()) {
            assert!(got.is_finite(), "quantized matmul row produced a non-finite value: {got}");
            let diff = (got - want).abs();
            max_error = max_error.max(diff);
            sum_sq_error += f64::from(diff) * f64::from(diff);
        }
        let rms_error = (sum_sq_error / rows as f64).sqrt();
        eprintln!("matmul_q4k_f32 vs dequantize-then-matmul: max_error={max_error} rms_error={rms_error}");

        // loose sanity bound around the accumulation-order float noise
        // floor for a 768-element dot product at this value scale — not
        // tuned to the measured numbers, matching this crate's existing
        // q4_k round-trip test convention (`proxima-gguf`'s
        // `quantize_dequantize_smooth_signal_round_trip_error`).
        assert!(max_error < 0.05, "max_error={max_error} exceeds loose sanity bound");
        assert!(rms_error < 0.02, "rms_error={rms_error} exceeds loose sanity bound");
    }

    /// [`dot_q4k_f32`]'s shape-mismatch guard: an activation slice whose
    /// length does not match the weight row's decoded element count is
    /// rejected, not silently truncated or padded.
    #[test]
    fn matmul_q4k_f32_rejects_an_activation_length_that_does_not_match_the_weight_rows_element_count() {
        use proxima_gguf::quant::q4_k::BLOCK_BYTES;

        let weight_blocks = vec![0u8; BLOCK_BYTES];
        let wrong_length_activation = vec![0.0f32; 200];
        let error = matmul_q4k_f32(&weight_blocks, 1, &wrong_length_activation).unwrap_err();
        assert!(matches!(error, TensorError::QuantizedShapeMismatch { .. }), "got {error:?}");
    }

    /// [`matmul_q4k_q8k_f32`] against the SAME incumbent
    /// [`matmul_q4k_f32_agrees_with_dequantize_then_matmul_within_a_measured_tolerance`]
    /// checks against: dequantize the packed `Q4_K` bytes, plain `f32` dot.
    /// This path never touches `dequantize`/`dot_q4k_f32` at all -- every
    /// weight byte is read once, as a nibble, straight into an integer
    /// accumulate -- so a nonzero difference from the `f32` reference is
    /// expected (Q8_K's own quantization error, `iscale = -127/max`, is a
    /// second lossy step this path pays that `dot_q4k_f32` does not).
    #[cfg(feature = "q4k-int8-dot")]
    #[test]
    fn matmul_q4k_q8k_f32_agrees_with_dequantize_then_matmul_within_a_measured_tolerance() {
        use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};

        let rows = 5;
        let blocks_per_row = 3;
        let k = QK_K * blocks_per_row;

        let activation: Vec<f32> = random_vec(7, k).into_iter().map(|value| value * 4.0 - 2.0).collect();
        let weight_f32: Vec<f32> = random_vec(11, rows * k).into_iter().map(|value| value * 4.0 - 2.0).collect();

        let mut weight_blocks = vec![0u8; rows * blocks_per_row * BLOCK_BYTES];
        for (row_f32, row_blocks) in weight_f32
            .chunks_exact(k)
            .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
        {
            quantize(row_f32, row_blocks).expect("row length is a whole multiple of QK_K by construction");
        }

        let mut expected = Vec::with_capacity(rows);
        for row_blocks in weight_blocks.chunks_exact(blocks_per_row * BLOCK_BYTES) {
            let mut dequantized = vec![0.0f32; k];
            dequantize(row_blocks, &mut dequantized).expect("row_blocks is a whole number of q4_k super-blocks");
            let dot: f32 = dequantized.iter().zip(activation.iter()).map(|(&weight, &value)| weight * value).sum();
            expected.push(dot);
        }

        let actual = matmul_q4k_q8k_f32(&weight_blocks, rows, &activation).expect("well-formed packed int8 matmul");

        assert_eq!(actual.len(), expected.len());
        let mut max_error = 0.0f32;
        let mut sum_sq_error = 0.0f64;
        for (&got, &want) in actual.iter().zip(expected.iter()) {
            assert!(got.is_finite(), "packed int8 matmul row produced a non-finite value: {got}");
            let diff = (got - want).abs();
            max_error = max_error.max(diff);
            sum_sq_error += f64::from(diff) * f64::from(diff);
        }
        let rms_error = (sum_sq_error / rows as f64).sqrt();
        let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
        let relative_max_error = max_error / max_magnitude;
        eprintln!(
            "matmul_q4k_q8k_f32 vs dequantize-then-matmul: max_error={max_error} rms_error={rms_error} \
             max_magnitude={max_magnitude} relative_max_error={relative_max_error}"
        );
        // Unlike `matmul_q4k_f32_agrees_with_dequantize_then_matmul_within_a_measured_tolerance`
        // above (whose only source of disagreement is accumulation order --
        // both arms there consume the SAME dequantized bytes, so an
        // absolute bound near the float noise floor is right), this path
        // ALSO quantizes the activation to Q8_K, a second real lossy step.
        // The dot magnitude here runs into the thousands (this fixture's
        // `random_vec`-derived data is not zero-mean), so an absolute
        // bound copied from that other test would be meaningless -- this
        // is RELATIVE error against the signal's own magnitude, still a
        // loose sanity bound and still not tuned to the measured number.
        assert!(
            relative_max_error < 0.01,
            "relative_max_error={relative_max_error} (max_error={max_error} over magnitude {max_magnitude}) \
             exceeds loose sanity bound"
        );
    }

    /// [`dot_q4k_q8k_block_avx2`]'s equivalence proof, the x86_64 sibling of
    /// [`matmul_q4k_q8k_f32_agrees_bit_exact_with_the_portable_arm`] below --
    /// same reasoning: every intermediate value both kernels compute is
    /// integer (`i32` partial sums via `_mm256_maddubs_epi16` +
    /// `_mm256_madd_epi16`, `i32` mins correction) until the final `f32`
    /// scale multiply, so [`dot_q4k_q8k_block_avx2`] and
    /// [`dot_q4k_q8k_block_scalar`] must agree bit-for-bit on the same
    /// input.
    ///
    /// This crate's dev boxes are all aarch64-darwin, so this test is
    /// COMPILED (verified via `cargo check -p proxima-tensor --target
    /// x86_64-unknown-linux-gnu --tests`) but never EXECUTED on this
    /// machine -- `#[cfg(target_arch = "x86_64")]` means it does not even
    /// exist in the aarch64 test binary this crate's own `cargo nextest
    /// run` builds. It runs for real on any x86_64 CI runner (this
    /// workspace's `proxima-tensor-gate.sh` targets
    /// `x86_64-unknown-linux-gnu`) or on `versailles`; the `is_x86_feature_detected!`
    /// guard skips it rather than failing on a pre-2013 x86_64 host with no
    /// AVX2 at all.
    #[cfg(all(test, target_arch = "x86_64", feature = "q4k-int8-dot"))]
    #[test]
    fn dot_q4k_q8k_block_avx2_agrees_bit_exact_with_scalar_when_avx2_is_present() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

        let blocks_per_row = 3;
        let k = QK_K * blocks_per_row;
        let activation: Vec<f32> = random_vec(29, k).into_iter().map(|value| value * 6.0 - 3.0).collect();
        let weight_f32: Vec<f32> = random_vec(31, k).into_iter().map(|value| value * 6.0 - 3.0).collect();

        let mut weight_row = vec![0u8; blocks_per_row * BLOCK_BYTES];
        quantize(&weight_f32, &mut weight_row).expect("row length is a whole multiple of QK_K by construction");

        let mut activation_q8k = vec![0u8; blocks_per_row * Q8K_BLOCK_BYTES];
        quantize_row_q8k(&activation, &mut activation_q8k).expect("well-formed activation");

        for (weight_block, q8k_block) in weight_row
            .as_chunks::<Q4K_BLOCK_BYTES>()
            .0
            .iter()
            .zip(activation_q8k.as_chunks::<Q8K_BLOCK_BYTES>().0)
        {
            let scalar = dot_q4k_q8k_block_scalar(weight_block, q8k_block);
            // SAFETY: `is_x86_feature_detected!("avx2")` confirmed above.
            let avx2 = unsafe { dot_q4k_q8k_block_avx2(weight_block, q8k_block) };
            assert_eq!(
                scalar.to_bits(),
                avx2.to_bits(),
                "AVX2 block dot diverged from the scalar reference -- not merely an acceleration"
            );
        }
    }

    /// The whole point of [`dot_q4k_q8k_block_neon_dotprod`]: it is an
    /// ACCELERATION of [`dot_q4k_q8k_block_scalar`]'s mechanism, not a
    /// different one. Every intermediate value both paths compute is
    /// integer (`i32` partial sums, `i32` mins correction) until the very
    /// last step, and integer addition has no rounding -- so
    /// [`matmul_q4k_q8k_f32`] (whichever arm `q4k_dotprod` selects) and
    /// [`matmul_q4k_q8k_portable_f32`] (always the scalar arm) must produce
    /// BIT-EXACT output on the same input, not merely close. A tolerance
    /// here would hide a real divergence between the two implementations.
    #[cfg(feature = "q4k-int8-dot")]
    #[test]
    fn matmul_q4k_q8k_f32_agrees_bit_exact_with_the_portable_arm() {
        use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

        let rows = 4;
        let blocks_per_row = 5;
        let k = QK_K * blocks_per_row;

        let activation: Vec<f32> = random_vec(13, k).into_iter().map(|value| value * 6.0 - 3.0).collect();
        let weight_f32: Vec<f32> = random_vec(17, rows * k).into_iter().map(|value| value * 6.0 - 3.0).collect();

        let mut weight_blocks = vec![0u8; rows * blocks_per_row * BLOCK_BYTES];
        for (row_f32, row_blocks) in weight_f32
            .chunks_exact(k)
            .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
        {
            quantize(row_f32, row_blocks).expect("row length is a whole multiple of QK_K by construction");
        }

        let dispatched = matmul_q4k_q8k_f32(&weight_blocks, rows, &activation).expect("well-formed dispatched matmul");
        let portable = matmul_q4k_q8k_portable_f32(&weight_blocks, rows, &activation).expect("well-formed portable matmul");

        assert_eq!(dispatched, portable, "dispatched and portable arms diverged -- not merely an acceleration");
    }

    /// [`quantize_row_q8k`]'s all-zero fast path
    /// (`quantize_row_q8_K_ref`'s `if (!amax) { y[i].d = 0; memset(...); }`
    /// arm, `ggml-quants.c:2483-2488`): a zero super-block must round-trip
    /// to an exactly zero packed block, not merely a near-zero one.
    #[cfg(feature = "q4k-int8-dot")]
    #[test]
    fn quantize_row_q8k_zero_vector_is_bit_exact_zero() {
        let activation = vec![0.0f32; Q4K_BLOCK_ELEMENTS];
        let mut packed = vec![0xFFu8; Q8K_BLOCK_BYTES];
        quantize_row_q8k(&activation, &mut packed).expect("one well-formed super-block");
        assert!(packed.iter().all(|&byte| byte == 0), "zero activation must pack to an all-zero Q8_K block");
    }

    /// [`quantize_row_q8k_dispatch`]'s cohort split against
    /// [`quantize_row_q8k`]'s serial reference, at a block count
    /// (`4 * MIN_QUANTIZE_BLOCKS_FOR_DISPATCH`) chosen to clear
    /// [`MIN_QUANTIZE_BLOCKS_FOR_DISPATCH`] with headroom so the dispatch
    /// path (not the serial fallback) actually runs -- every `Q8_K`
    /// super-block quantizes independently (`quantize_row_q8k`'s own doc),
    /// so this must be bit-for-bit, not merely close.
    #[cfg(feature = "q4k-int8-dot")]
    #[test]
    fn quantize_row_q8k_dispatch_is_bit_identical_to_the_serial_reference() {
        let block_count = MIN_QUANTIZE_BLOCKS_FOR_DISPATCH * 4;
        let activation: Vec<f32> =
            (0..block_count * Q4K_BLOCK_ELEMENTS).map(|index| ((index % 251) as f32 - 125.0) * 0.037).collect();

        let mut serial = vec![0u8; block_count * Q8K_BLOCK_BYTES];
        quantize_row_q8k(&activation, &mut serial).expect("well-formed activation quantizes serially");

        let cohort = MatmulCohort::from_config(MatmulCohort::builder().members(NonZeroUsize::new(4).expect("4 is nonzero")).build())
            .expect("test cohort with 4 members spawns");
        let session = cohort.enter().expect("no other session open on a fresh cohort");
        let mut dispatched = vec![0u8; block_count * Q8K_BLOCK_BYTES];
        quantize_row_q8k_dispatch(&activation, &mut dispatched, Some(&session))
            .expect("well-formed activation quantizes through the cohort");
        drop(session);

        assert_eq!(dispatched, serial, "cohort-dispatched Q8_K packing must be bit-identical to the serial reference");
    }

    /// [`transpose_wide_to_output`]'s cohort split against its own serial
    /// fallback, at `rows * leading_total` (`rows = 8000`,
    /// `leading_total = 9`, 72,000 elements) chosen to clear
    /// [`MIN_TRANSPOSE_ELEMENTS_FOR_DISPATCH`] with headroom -- pure data
    /// movement, so this must be bit-for-bit.
    #[cfg(feature = "q4k-int8-dot")]
    #[test]
    fn transpose_wide_to_output_dispatch_is_bit_identical_to_the_serial_reference() {
        let rows = 8000usize;
        let leading_total = 9usize;
        assert!(
            rows * leading_total >= MIN_TRANSPOSE_ELEMENTS_FOR_DISPATCH,
            "this shape must clear the threshold or this test proves nothing about the dispatch path"
        );
        let wide: Vec<f32> = (0..rows * leading_total).map(|index| index as f32 * 0.5 - 17.0).collect();

        let mut serial = vec![0.0f32; rows * leading_total];
        transpose_wide_to_output(&wide, rows, leading_total, None, &mut serial)
            .expect("serial transpose never fails");

        let cohort = MatmulCohort::from_config(MatmulCohort::builder().members(NonZeroUsize::new(4).expect("4 is nonzero")).build())
            .expect("test cohort with 4 members spawns");
        let session = cohort.enter().expect("no other session open on a fresh cohort");
        let mut dispatched = vec![0.0f32; rows * leading_total];
        transpose_wide_to_output(&wide, rows, leading_total, Some(&session), &mut dispatched)
            .expect("cohort-dispatched transpose never fails");
        drop(session);

        assert_eq!(dispatched, serial, "cohort-dispatched transpose must be bit-identical to the serial reference");
    }

    /// [`dot_q4k_q8k`]'s shape-mismatch guard, mirroring
    /// [`matmul_q4k_f32_rejects_an_activation_length_that_does_not_match_the_weight_rows_element_count`]
    /// for the packed-int8 sibling: a `Q8_K` activation buffer sized for
    /// the wrong block count is rejected, never silently truncated.
    #[cfg(feature = "q4k-int8-dot")]
    #[test]
    fn dot_q4k_q8k_rejects_a_q8k_activation_length_mismatch() {
        let weight_block = vec![0u8; Q4K_BLOCK_BYTES];
        let wrong_length_q8k = vec![0u8; Q8K_BLOCK_BYTES - 1];
        let error = dot_q4k_q8k(&weight_block, &wrong_length_q8k).unwrap_err();
        assert!(matches!(error, TensorError::QuantizedShapeMismatch { .. }), "got {error:?}");
    }

    /// Same guard, exercised on the always-portable entry point directly
    /// (bypassing `q4k_dotprod` dispatch entirely).
    #[cfg(feature = "q4k-int8-dot")]
    #[test]
    fn dot_q4k_q8k_portable_rejects_a_weight_row_length_not_a_block_multiple() {
        let weight_row = vec![0u8; Q4K_BLOCK_BYTES - 1];
        let q8k = vec![0u8; Q8K_BLOCK_BYTES];
        let error = dot_q4k_q8k_portable(&weight_row, &q8k).unwrap_err();
        assert!(matches!(error, TensorError::QuantizedShapeMismatch { .. }), "got {error:?}");
    }

    /// [`QuantDot::Fused`] vs [`QuantDot::Unfused`] on identical random
    /// `Q4_K` blocks: both consume the exact same already-quantized bytes
    /// (weight and activation), so the only remaining disagreement is
    /// floating-point accumulation order (an integer-factored int8 fold vs
    /// a linear `f32` sum) -- a MUCH tighter bound than
    /// `matmul_q4k_q8k_f32_agrees_with_dequantize_then_matmul_within_a_measured_tolerance`,
    /// which compares against the ORIGINAL unquantized activation and so
    /// also absorbs the activation's own Q8_K quantization error. Also
    /// checks `Fused` against [`dot_q4k_q8k`] directly (bit-exact): the
    /// pipe wrapper introduces no deviation from the kernel it delegates to.
    #[cfg(feature = "q4k-int8-dot")]
    #[test]
    fn quant_dot_fused_and_unfused_agree_for_q4k_within_int8_quantization_tolerance() {
        use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

        let blocks_per_row = 3;
        let k = QK_K * blocks_per_row;
        let weight_f32 = random_vec(21, k);
        let activation_f32: Vec<f32> = random_vec(22, k).into_iter().map(|value| value * 4.0 - 2.0).collect();

        let mut weight_bytes = vec![0u8; blocks_per_row * BLOCK_BYTES];
        quantize(&weight_f32, &mut weight_bytes).expect("k is a whole number of q4_k super-blocks");
        let mut activation_q8k = vec![0u8; blocks_per_row * Q8K_BLOCK_BYTES];
        quantize_row_q8k(&activation_f32, &mut activation_q8k).expect("k is a whole number of q8_k super-blocks");

        let fused = block_on(QuantDot::Fused(QuantizedBlock::Q4K(&weight_bytes)).call(&activation_q8k))
            .expect("fused int8 dot evaluates");
        let unfused = block_on(QuantDot::Unfused(QuantizedBlock::Q4K(&weight_bytes)).call(&activation_q8k))
            .expect("unfused dequantize-then-fold evaluates");
        let kernel_ground_truth =
            dot_q4k_q8k(&weight_bytes, &activation_q8k).expect("the underlying kernel evaluates directly");

        assert_eq!(fused, kernel_ground_truth, "QuantDot::Fused must be a bit-exact wrapper over dot_q4k_q8k");
        let relative_error = (fused - unfused).abs() / fused.abs().max(1.0);
        eprintln!("q4_k QuantDot fused={fused} unfused={unfused} relative_error={relative_error}");
        assert!(relative_error < 1e-3, "relative_error={relative_error} exceeds parity tolerance");
    }

    #[cfg(feature = "q5k-int8-dot")]
    #[test]
    fn quant_dot_fused_and_unfused_agree_for_q5k_within_int8_quantization_tolerance() {
        use proxima_gguf::quant::q5_k::{BLOCK_BYTES, QK_K, quantize};

        let blocks_per_row = 3;
        let k = QK_K * blocks_per_row;
        let weight_f32 = random_vec(23, k);
        let activation_f32: Vec<f32> = random_vec(24, k).into_iter().map(|value| value * 4.0 - 2.0).collect();

        let mut weight_bytes = vec![0u8; blocks_per_row * BLOCK_BYTES];
        quantize(&weight_f32, &mut weight_bytes).expect("k is a whole number of q5_k super-blocks");
        let mut activation_q8k = vec![0u8; blocks_per_row * Q8K_BLOCK_BYTES];
        quantize_row_q8k(&activation_f32, &mut activation_q8k).expect("k is a whole number of q8_k super-blocks");

        let fused = block_on(QuantDot::Fused(QuantizedBlock::Q5K(&weight_bytes)).call(&activation_q8k))
            .expect("fused int8 dot evaluates");
        let unfused = block_on(QuantDot::Unfused(QuantizedBlock::Q5K(&weight_bytes)).call(&activation_q8k))
            .expect("unfused dequantize-then-fold evaluates");
        let kernel_ground_truth =
            dot_q5k_q8k(&weight_bytes, &activation_q8k).expect("the underlying kernel evaluates directly");

        assert_eq!(fused, kernel_ground_truth, "QuantDot::Fused must be a bit-exact wrapper over dot_q5k_q8k");
        let relative_error = (fused - unfused).abs() / fused.abs().max(1.0);
        eprintln!("q5_k QuantDot fused={fused} unfused={unfused} relative_error={relative_error}");
        assert!(relative_error < 1e-3, "relative_error={relative_error} exceeds parity tolerance");
    }

    #[cfg(feature = "q6k-int8-dot")]
    #[test]
    fn quant_dot_fused_and_unfused_agree_for_q6k_within_int8_quantization_tolerance() {
        use proxima_gguf::quant::q6_k::{BLOCK_BYTES, QK_K, quantize};

        let blocks_per_row = 3;
        let k = QK_K * blocks_per_row;
        let weight_f32 = random_vec(25, k);
        let activation_f32: Vec<f32> = random_vec(26, k).into_iter().map(|value| value * 4.0 - 2.0).collect();

        let mut weight_bytes = vec![0u8; blocks_per_row * BLOCK_BYTES];
        quantize(&weight_f32, &mut weight_bytes).expect("k is a whole number of q6_k super-blocks");
        let mut activation_q8k = vec![0u8; blocks_per_row * Q8K_BLOCK_BYTES];
        quantize_row_q8k(&activation_f32, &mut activation_q8k).expect("k is a whole number of q8_k super-blocks");

        let fused = block_on(QuantDot::Fused(QuantizedBlock::Q6K(&weight_bytes)).call(&activation_q8k))
            .expect("fused int8 dot evaluates");
        let unfused = block_on(QuantDot::Unfused(QuantizedBlock::Q6K(&weight_bytes)).call(&activation_q8k))
            .expect("unfused dequantize-then-fold evaluates");
        let kernel_ground_truth =
            dot_q6k_q8k(&weight_bytes, &activation_q8k).expect("the underlying kernel evaluates directly");

        assert_eq!(fused, kernel_ground_truth, "QuantDot::Fused must be a bit-exact wrapper over dot_q6k_q8k");
        let relative_error = (fused - unfused).abs() / fused.abs().max(1.0);
        eprintln!("q6_k QuantDot fused={fused} unfused={unfused} relative_error={relative_error}");
        assert!(relative_error < 1e-3, "relative_error={relative_error} exceeds parity tolerance");
    }

    /// Both arms of [`QuantDot`] take the identical `In` shape
    /// (`dot_q4k_q8k`'s own malformed-shape guards), so both must reject the
    /// identical malformed shapes rather than only one of them.
    #[cfg(feature = "q4k-int8-dot")]
    #[test]
    fn quant_dot_rejects_a_malformed_weight_row_on_both_fused_and_unfused() {
        let weight_row = vec![0u8; Q4K_BLOCK_BYTES - 1];
        let activation_q8k = vec![0u8; Q8K_BLOCK_BYTES];

        let fused_error =
            block_on(QuantDot::Fused(QuantizedBlock::Q4K(&weight_row)).call(&activation_q8k)).unwrap_err();
        assert!(matches!(fused_error, TensorError::QuantizedShapeMismatch { .. }), "got {fused_error:?}");

        let unfused_error =
            block_on(QuantDot::Unfused(QuantizedBlock::Q4K(&weight_row)).call(&activation_q8k)).unwrap_err();
        assert!(matches!(unfused_error, TensorError::QuantizedShapeMismatch { .. }), "got {unfused_error:?}");
    }

    /// A codec `QuantDot` does not support (`Q8_0` has no `Q8_K`-activation
    /// int8-dot path -- [`dot_fn_for`]'s own doc names this) is an honest
    /// `NotLowerable`, never a silent misroute to a different codec's
    /// kernel.
    #[cfg(feature = "q4k-int8-dot")]
    #[test]
    fn quant_dot_rejects_a_codec_with_no_int8_dot_kernel() {
        let weight_row = vec![0u8; 34]; // one Q8_0 block: 2-byte f16 scale + 32 nibble-packed bytes
        let activation_q8k = vec![0u8; Q8K_BLOCK_BYTES];

        let fused_error =
            block_on(QuantDot::Fused(QuantizedBlock::Q8_0(&weight_row)).call(&activation_q8k)).unwrap_err();
        assert!(matches!(fused_error, TensorError::NotLowerable { .. }), "got {fused_error:?}");

        let unfused_error =
            block_on(QuantDot::Unfused(QuantizedBlock::Q8_0(&weight_row)).call(&activation_q8k)).unwrap_err();
        assert!(matches!(unfused_error, TensorError::NotLowerable { .. }), "got {unfused_error:?}");
    }

    /// [`mins_correction_neon`] (the explicit NEON mins-correction path
    /// this landing introduced) against the scalar route through
    /// [`proxima_gguf::quant::q4_k::get_scale_min_k4`] it replaced -- bit
    /// exact, not approximate (guiding-principles: the mins correction is
    /// an integer computation widened to `i32`, so exact equality is
    /// achievable and is the bar). Uses the FIRST `Q4_K` super-block of a
    /// real weight tensor read straight out of the real openchat-3.5-1210
    /// `Q4_K_S` GGUF file (principle 9: real-world data), against a real
    /// (non-zero) quantized activation super-block, so the scale/min bit
    /// patterns and `bsums` are whatever ggml's own quantizer actually
    /// produced -- not a synthetic stand-in.
    #[cfg(all(q4k_dotprod, feature = "q4k-int8-dot"))]
    #[test]
    fn mins_correction_neon_agrees_with_get_scale_min_k4_scalar_route_on_real_gguf_bytes() {
        let path = std::path::Path::new(REAL_OPENCHAT_GGUF_PATH);
        let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
            eprintln!("real gguf file not found at {REAL_OPENCHAT_GGUF_PATH}; test skipped");
            return;
        };
        let Some((weight_bytes, _in_dim, _out_dim)) = real_tensor_bytes(
            &mut file,
            &parsed,
            file_len,
            "blk.0.attn_q.weight",
            proxima_gguf::types::GgmlType::Q4_K,
        ) else {
            eprintln!("blk.0.attn_q.weight is not Q4_K in this file; test skipped, not faked");
            return;
        };

        let mut scales = [0u8; Q4K_SCALE_BYTES];
        scales.copy_from_slice(&weight_bytes[Q4K_SCALES_OFFSET..Q4K_SCALES_OFFSET + Q4K_SCALE_BYTES]);

        let activation: Vec<f32> = random_vec(29, Q4K_BLOCK_ELEMENTS).into_iter().map(|value| value * 6.0 - 3.0).collect();
        let mut activation_q8k = vec![0u8; Q8K_BLOCK_BYTES];
        quantize_row_q8k(&activation, &mut activation_q8k).expect("well-formed activation super-block");
        let bsums = &activation_q8k[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];

        let mut expected_mins_correction = 0i32;
        for sub_block in 0..Q4K_SUB_BLOCKS {
            let (expected_scale, min_code) = proxima_gguf::quant::q4_k::get_scale_min_k4(sub_block, &scales);
            // SAFETY: `q4k_dotprod` cfg guarantees `FEAT_DotProd`; `bsums` is
            // exactly `Q8K_BSUMS_COUNT * 2` bytes from the fixed-size buffer above.
            let (scale_lo, scale_hi, _) = unsafe { mins_correction_neon(&scales, bsums) };
            let actual_scale = if sub_block < 4 {
                scale_byte(scale_lo, sub_block as u32)
            } else {
                scale_byte(scale_hi, (sub_block - 4) as u32)
            };
            assert_eq!(
                actual_scale, i32::from(expected_scale),
                "scale word diverged from the scalar get_scale_min_k4 route at sub_block {sub_block}"
            );
            let bsum_lo = i16::from_le_bytes([bsums[sub_block * 4], bsums[sub_block * 4 + 1]]);
            let bsum_hi = i16::from_le_bytes([bsums[sub_block * 4 + 2], bsums[sub_block * 4 + 3]]);
            expected_mins_correction += i32::from(bsum_lo + bsum_hi) * i32::from(min_code);
        }

        // SAFETY: `q4k_dotprod` cfg guarantees `FEAT_DotProd`; `bsums` is
        // exactly `Q8K_BSUMS_COUNT * 2` bytes from the fixed-size buffer above.
        let (_, _, actual_mins_correction) = unsafe { mins_correction_neon(&scales, bsums) };
        assert_eq!(
            actual_mins_correction, expected_mins_correction,
            "NEON mins_correction diverged from the scalar get_scale_min_k4 route -- nonzero delta, faster wrong kernel"
        );
    }

    /// [`dot_q5k_q8k_block_neon_dotprod`]'s new [`mins_correction_neon`]
    /// route (this landing) against the 16-scalar-call
    /// `get_scale_min_k4` route it replaced (8 for the mins correction,
    /// 8 more for `scale_lo`/`scale_hi` inside the SIMD dot loop) -- bit
    /// exact, not approximate, same bar as
    /// [`mins_correction_neon_agrees_with_get_scale_min_k4_scalar_route_on_real_gguf_bytes`].
    /// Uses the first `Q5_K` super-block of `blk.0.attn_v.weight` read
    /// straight out of the real openchat-3.5-1210 GGUF file (principle 9:
    /// real-world data), against a real quantized activation super-block.
    #[cfg(all(q4k_dotprod, feature = "q5k-int8-dot"))]
    #[test]
    fn q5k_mins_correction_neon_agrees_with_get_scale_min_k4_scalar_route_on_real_gguf_bytes() {
        let path = std::path::Path::new(REAL_OPENCHAT_GGUF_PATH);
        let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
            eprintln!("real gguf file not found at {REAL_OPENCHAT_GGUF_PATH}; test skipped");
            return;
        };
        let Some((weight_bytes, _in_dim, _out_dim)) = real_tensor_bytes(
            &mut file,
            &parsed,
            file_len,
            "blk.0.attn_v.weight",
            proxima_gguf::types::GgmlType::Q5_K,
        ) else {
            eprintln!("blk.0.attn_v.weight is not Q5_K in this file; test skipped, not faked");
            return;
        };

        let mut scales = [0u8; Q4K_SCALE_BYTES];
        scales.copy_from_slice(&weight_bytes[Q5K_SCALES_OFFSET..Q5K_SCALES_OFFSET + Q4K_SCALE_BYTES]);

        let activation: Vec<f32> = random_vec(37, Q4K_BLOCK_ELEMENTS).into_iter().map(|value| value * 6.0 - 3.0).collect();
        let mut activation_q8k = vec![0u8; Q8K_BLOCK_BYTES];
        quantize_row_q8k(&activation, &mut activation_q8k).expect("well-formed activation super-block");
        let bsums = &activation_q8k[Q8K_BSUMS_OFFSET..Q8K_BSUMS_OFFSET + Q8K_BSUMS_COUNT * 2];

        let mut expected_mins_correction = 0i32;
        for sub_block in 0..Q4K_SUB_BLOCKS {
            let (expected_scale, min_code) = proxima_gguf::quant::q4_k::get_scale_min_k4(sub_block, &scales);
            // SAFETY: `q4k_dotprod` cfg guarantees `FEAT_DotProd`; `bsums` is
            // exactly `Q8K_BSUMS_COUNT * 2` bytes from the fixed-size buffer above.
            let (scale_lo, scale_hi, _) = unsafe { mins_correction_neon(&scales, bsums) };
            let actual_scale = if sub_block < 4 {
                scale_byte(scale_lo, sub_block as u32)
            } else {
                scale_byte(scale_hi, (sub_block - 4) as u32)
            };
            assert_eq!(
                actual_scale, i32::from(expected_scale),
                "scale word diverged from the scalar get_scale_min_k4 route at sub_block {sub_block} on Q5_K bytes"
            );
            let bsum_lo = i16::from_le_bytes([bsums[sub_block * 4], bsums[sub_block * 4 + 1]]);
            let bsum_hi = i16::from_le_bytes([bsums[sub_block * 4 + 2], bsums[sub_block * 4 + 3]]);
            expected_mins_correction += i32::from(bsum_lo + bsum_hi) * i32::from(min_code);
        }

        // SAFETY: `q4k_dotprod` cfg guarantees `FEAT_DotProd`; `bsums` is
        // exactly `Q8K_BSUMS_COUNT * 2` bytes from the fixed-size buffer above.
        let (_, _, actual_mins_correction) = unsafe { mins_correction_neon(&scales, bsums) };
        assert_eq!(
            actual_mins_correction, expected_mins_correction,
            "Q5_K NEON mins_correction diverged from the scalar get_scale_min_k4 route -- nonzero delta, faster wrong kernel"
        );
    }

    /// Same shape as [`matmul_program`] (`[rows, k] x [k, 1] -> [rows, 1]`,
    /// `n = 1` — batch-1, [`matmul_q4k_f32`]'s own documented target shape),
    /// weight declared `UInt8` instead of `Float32`: the program
    /// [`evaluate_quantized`] runs, standing in for a `Q4_K`-packed weight
    /// matrix.
    fn quantized_matmul_program(rows: u32, k: u32) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let weight = block(&mut program, DType::UInt8, &[Extent::Static(rows), Extent::Static(k)]);
        let activation = f32_block(&mut program, &[Extent::Static(k), Extent::Static(1)]);
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (weight, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (activation, IndexMap::Affine(map::projection(3, &[2, 1]))),
                ],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: Some("quantized_matmul".into()),
            }),
        );
        (program, sum)
    }

    /// The capability this whole module change exists for: a real `Op`
    /// program, with a `Q4_K`-packed weight operand, run end to end through
    /// [`evaluate_quantized`] — not `matmul_q4k_f32` called directly, the
    /// way every other test above exercises it — and checked against the
    /// exact same shape run through plain [`evaluate`] with the weight
    /// dequantized to `f32` first. Proves the dispatch chain
    /// `evaluate_quantized` -> `run_node_into` -> `run_reduce` ->
    /// [`quantized_operand`] -> `run_reduce_quantized` -> `matmul_q4k_q8k_f32`
    /// (`q4k-int8-dot` is default-on; `matmul_q4k_f32` is the fallback when
    /// it is off) is reachable from the program-level entry point, not
    /// merely callable in isolation.
    #[test]
    fn evaluate_quantized_matmul_matches_dequantize_then_f32_evaluate() {
        use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};

        let rows: u32 = 5;
        let blocks_per_row = 3;
        let k = QK_K as u32 * blocks_per_row as u32;

        let activation: Vec<f32> = random_vec(13, k as usize).into_iter().map(|value| value * 4.0 - 2.0).collect();
        let weight_f32: Vec<f32> =
            random_vec(17, rows as usize * k as usize).into_iter().map(|value| value * 4.0 - 2.0).collect();

        let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
        for (row_f32, row_blocks) in
            weight_f32.chunks_exact(k as usize).zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
        {
            quantize(row_f32, row_blocks).expect("row length is a whole multiple of QK_K by construction");
        }

        let (quantized_program, quantized_sum) = quantized_matmul_program(rows, k);
        let quantized_blocks = [QuantizedBlock::Q4K(&weight_blocks), QuantizedBlock::Float32(&activation)];
        let quantized_result = evaluate_quantized(&quantized_program, &[], &quantized_blocks, &[quantized_sum])
            .expect("quantized matmul evaluates end to end");

        let mut dequantized_weight = vec![0.0f32; rows as usize * k as usize];
        for (row_blocks, row_f32) in weight_blocks
            .chunks_exact(blocks_per_row * BLOCK_BYTES)
            .zip(dequantized_weight.chunks_exact_mut(k as usize))
        {
            dequantize(row_blocks, row_f32).expect("row_blocks is a whole number of q4_k super-blocks");
        }

        let (f32_program, f32_sum) = matmul_program(rows, k, 1, false);
        let f32_blocks: [&[f32]; 2] = [&dequantized_weight, &activation];
        let f32_result =
            evaluate(&f32_program, &[], &f32_blocks, &[f32_sum]).expect("dequantized f32 matmul evaluates");

        let actual = quantized_result.root();
        let expected = f32_result.root();
        assert_eq!(actual.len(), rows as usize);
        assert_eq!(actual.len(), expected.len());

        let mut max_diff = 0.0f32;
        let mut sum_sq_diff = 0.0f64;
        for (&got, &want) in actual.iter().zip(expected.iter()) {
            assert!(got.is_finite(), "evaluate_quantized produced a non-finite value: {got}");
            let diff = (got - want).abs();
            max_diff = max_diff.max(diff);
            sum_sq_diff += f64::from(diff) * f64::from(diff);
        }
        let rms_diff = (sum_sq_diff / rows as f64).sqrt();
        let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
        let relative_max_diff = max_diff / max_magnitude;
        eprintln!(
            "evaluate_quantized vs dequantize-then-evaluate: max_diff={max_diff} rms_diff={rms_diff} \
             max_magnitude={max_magnitude} relative_max_diff={relative_max_diff}"
        );

        // `run_reduce_quantized` routes through `matmul_q4k_q8k_f32` by
        // default (`q4k-int8-dot` default-on) — same second lossy step
        // (Q8_K activation quantization) as
        // `matmul_q4k_q8k_f32_agrees_with_dequantize_then_matmul_within_a_measured_tolerance`,
        // so this bound is RELATIVE to the signal's own magnitude the same
        // way that test's is, not the absolute float-noise-floor bound that
        // was right when this call bottomed out in `matmul_q4k_f32` alone.
        assert!(
            relative_max_diff < 0.01,
            "relative_max_diff={relative_max_diff} (max_diff={max_diff} over magnitude {max_magnitude}) \
             exceeds loose sanity bound"
        );
    }

    /// `sum_k weight[route[s], o, k] * activation[s, k]` -- `moe_block.toml`'s
    /// `expert_w` gather, mirroring [`embedding_matmul_program`]'s `(s, o, k)`
    /// iteration shape with the gather moved from the activation side to the
    /// weight side. `weight`'s own physical shape is `[n_experts, rows, k]`;
    /// `gathered_dim: 0` and `index_map` reading iteration axis 0 (`s`) is the
    /// same [`IndexMap::Computed`] wiring [`embedding_lookup_program`] uses,
    /// just on the operand [`run_reduce_quantized`] actually dequantizes.
    fn gathered_quantized_matmul_program(n_experts: u32, rows: u32, k: u32, seq: u32) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let weight = block(
            &mut program,
            DType::UInt8,
            &[Extent::Static(n_experts), Extent::Static(rows), Extent::Static(k)],
        );
        let route = block(&mut program, DType::Int32, &[Extent::Static(seq)]);
        let activation = f32_block(&mut program, &[Extent::Static(seq), Extent::Static(k)]);

        let gather_map = IndexMap::Computed {
            indices: route,
            index_map: map::projection(3, &[0]),
            base: map::IndexPattern {
                iter_rank: 3,
                axes: alloc::vec![
                    map::AxisIndex::default(),
                    map::AxisIndex {
                        terms: core::iter::once(AxisTerm::projection(1)).collect(),
                        offset: 0,
                    },
                    map::AxisIndex {
                        terms: core::iter::once(AxisTerm::projection(2)).collect(),
                        offset: 0,
                    },
                ],
            },
            gathered_dim: 0,
        };
        let activation_map = IndexMap::Affine(map::projection(3, &[0, 2]));

        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(weight, gather_map), (activation, activation_map)],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: Some("gathered_quantized_matmul".into()),
            }),
        );
        (program, sum)
    }

    /// The decisive test this whole change exists for: gathering expert `k`
    /// out of a stacked packed weight must dequantize BIT-IDENTICALLY to
    /// expert `k`'s own standalone tensor, exercised through the evaluator
    /// (`evaluate_quantized` -> `run_node_into` -> `run_reduce_with_quantized_weights`
    /// -> `run_reduce_quantized`'s new gather-resolution branch), not through
    /// `restack::gather_expert` called directly. Three tokens, three DISTINCT
    /// experts with asymmetric weight magnitudes (1x / 5x / 20x): a token
    /// reading the wrong expert's slab produces a value off by that same
    /// asymmetric factor, not a subtle rounding difference, so this is a
    /// mechanism check, not a tolerance check.
    #[test]
    fn evaluate_quantized_gathered_moe_weight_matches_the_routed_experts_own_matmul() {
        use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

        let n_experts: u32 = 3;
        let rows: u32 = 4;
        let blocks_per_row = 1;
        let k = QK_K as u32 * blocks_per_row as u32;
        let seq: u32 = 3;
        // token `t` routes to a DIFFERENT expert than its own position (a
        // constant or identity route could not distinguish a working gather
        // from the pre-fix constant-offset read this change closes).
        let route_data = [2.0f32, 0.0, 1.0];
        let expert_scales = [1.0f32, 5.0, 20.0];

        let mut expert_blocks: Vec<Vec<u8>> = Vec::new();
        for (expert, &scale) in expert_scales.iter().enumerate() {
            let weight_f32: Vec<f32> = random_vec(101 + expert as u64, rows as usize * k as usize)
                .into_iter()
                .map(|value| (value * 4.0 - 2.0) * scale)
                .collect();
            let mut blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
            for (row_f32, row_blocks) in
                weight_f32.chunks_exact(k as usize).zip(blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
            {
                quantize(row_f32, row_blocks).expect("row length is QK_K by construction");
            }
            expert_blocks.push(blocks);
        }
        let stacked_weight: Vec<u8> = expert_blocks.iter().flatten().copied().collect();

        let activation: Vec<f32> =
            random_vec(211, seq as usize * k as usize).into_iter().map(|value| value * 2.0 - 1.0).collect();

        let (program, sum) = gathered_quantized_matmul_program(n_experts, rows, k, seq);
        let quantized_blocks = [
            QuantizedBlock::Q4K(&stacked_weight),
            QuantizedBlock::Float32(&route_data),
            QuantizedBlock::Float32(&activation),
        ];
        let evaluated = evaluate_quantized(&program, &[], &quantized_blocks, &[sum])
            .expect("gathered quantized moe matmul evaluates end to end");
        let actual = evaluated.root();
        assert_eq!(actual.len(), seq as usize * rows as usize);

        for (token, &route) in route_data.iter().enumerate() {
            let expert = route as usize;
            let activation_row = &activation[token * k as usize..(token + 1) * k as usize];
            // Same codec kernel `run_reduce_quantized`'s per-position loop
            // itself calls for a `Q4K` block (`q4k-int8-dot` default-on
            // routes through `matmul_q4k_q8k_f32`; off, the plain
            // `matmul_q4k_f32`) -- comparing against the OTHER kernel would
            // fail on that kernel's own lossy Q8_K quantization step, not on
            // a wrong-expert read, which is the one thing this test exists
            // to catch.
            #[cfg(feature = "q4k-int8-dot")]
            let expected = matmul_q4k_q8k_f32(&expert_blocks[expert], rows as usize, activation_row)
                .expect("the routed expert's own standalone matmul evaluates");
            #[cfg(not(feature = "q4k-int8-dot"))]
            let expected = matmul_q4k_f32(&expert_blocks[expert], rows as usize, activation_row)
                .expect("the routed expert's own standalone matmul evaluates");
            let actual_row = &actual[token * rows as usize..(token + 1) * rows as usize];
            assert_eq!(
                actual_row, expected,
                "token {token} routed to expert {expert}: gathered result does not bit-match that \
                 expert's own standalone matmul call"
            );
        }
    }

    /// `evaluate_quantized`'s `live_now` running count treats every operand
    /// `node_retirement` schedules for retirement as "always live here" (see
    /// the comment above `live_now -= 1` in that loop) -- true for an
    /// `Op::Input` float32 block and for a computed node, both of which are
    /// written into `buffers`, but FALSE for a `Q4_K`-packed weight node: the
    /// `QuantizedBlock::Q4K` match arm in `evaluate_quantized_with_scratch`
    /// routes it into the separate `quantized_weights` map instead of
    /// `buffers`, so its slot is `None` the whole time. `node_retirement`
    /// does not know about that split -- it schedules the weight's
    /// retirement from the generic program graph like any other operand --
    /// so every quantized-weight retirement decrements `live_now` for a slot
    /// that was never live. One layer only drifts the running count by one
    /// (silently wrong, not yet negative); two independent layers combined
    /// by a plain `Add` accumulate two such spurious decrements ahead of the
    /// real ones and drive the last, legitimate retirement negative --
    /// `attempt to subtract with overflow` in a debug build, silent
    /// wraparound to a huge `usize` in release (this session's
    /// `peak_live_buffers=18446744073709551614` == `2^64 - 2`).
    #[test]
    fn evaluate_quantized_two_layers_does_not_underflow_live_now() {
        use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

        let rows: u32 = 3;
        let blocks_per_row = 1;
        let k = QK_K as u32 * blocks_per_row as u32;

        fn quantized_weight_blocks(seed: u64, rows: u32, k: u32, blocks_per_row: usize) -> Vec<u8> {
            let weight_f32: Vec<f32> =
                random_vec(seed, rows as usize * k as usize).into_iter().map(|value| value * 4.0 - 2.0).collect();
            let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
            for (row_f32, row_blocks) in
                weight_f32.chunks_exact(k as usize).zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
            {
                quantize(row_f32, row_blocks).expect("row length is a whole multiple of QK_K by construction");
            }
            weight_blocks
        }

        fn append_layer(program: &mut Vec<Op>, rows: u32, k: u32) -> NodeId {
            let weight = block(program, DType::UInt8, &[Extent::Static(rows), Extent::Static(k)]);
            let activation = f32_block(program, &[Extent::Static(k), Extent::Static(1)]);
            let product = append(
                program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Multiply,
                    operands: alloc::vec![
                        (weight, IndexMap::Affine(map::projection(3, &[0, 2]))),
                        (activation, IndexMap::Affine(map::projection(3, &[2, 1]))),
                    ],
                    name: None,
                },
            );
            append(
                program,
                Op::Reduce(Reduce {
                    dtype: DType::Float32,
                    body: ScalarOp::Add,
                    init: ReduceInit::Zero,
                    operand: product,
                    in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                    out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                    keep: Keep::Reduce,
                    name: None,
                }),
            )
        }

        let mut program = Vec::new();
        let sum1 = append_layer(&mut program, rows, k);
        let sum2 = append_layer(&mut program, rows, k);
        let total = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![
                    (sum1, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (sum2, IndexMap::Affine(map::projection(2, &[0, 1]))),
                ],
                name: None,
            },
        );

        let weight1_blocks = quantized_weight_blocks(101, rows, k, blocks_per_row);
        let weight2_blocks = quantized_weight_blocks(202, rows, k, blocks_per_row);
        let activation1: Vec<f32> = random_vec(303, k as usize).into_iter().map(|value| value * 4.0 - 2.0).collect();
        let activation2: Vec<f32> = random_vec(404, k as usize).into_iter().map(|value| value * 4.0 - 2.0).collect();

        let blocks = [
            QuantizedBlock::Q4K(&weight1_blocks),
            QuantizedBlock::Float32(&activation1),
            QuantizedBlock::Q4K(&weight2_blocks),
            QuantizedBlock::Float32(&activation2),
        ];

        let evaluated = evaluate_quantized(&program, &[], &blocks, &[total])
            .expect("two chained quantized layers evaluate without panicking");

        let peak_live_buffers =
            evaluated.peak_live_buffers().expect("evaluate_quantized always reports peak_live_buffers");
        assert!(
            peak_live_buffers <= program.len(),
            "peak_live_buffers={peak_live_buffers} exceeds program.len()={} -- a live-buffer count can never \
             exceed the node count, so this value proves live_now underflowed and wrapped rather than being \
             merely large",
            program.len(),
        );
    }

    /// A small, synthetic stand-in for the cached-attention reduce shape
    /// that trips the seam `q8_0_quantized_key_value_cache_cannot_cross_the_weight_matmul_quantized_seam`
    /// (`proxima-model-interop/src/bind.rs`) reaches on a real checkpoint:
    /// one axis (`u`, the kv-head analog) that the packed weight operand
    /// varies over AND the activation operand ALSO varies over, kept as an
    /// OUTPUT axis (not the contracted one). Iteration space `[s, t, u, d]`
    /// — `t` (cached-length analog) is the sole reduced axis; `s`, `u`, `d`
    /// survive into the output. The packed weight varies over `t, u, d`
    /// (broadcasts over `s`); the activation varies over `s, t, u`
    /// (broadcasts over `d`) — `u` is the axis both share.
    ///
    /// Before this fix, `run_reduce_quantized`'s raw-byte division
    /// (`cpu.rs:2476-2486`) derived `rows` from the packed weight's own byte
    /// length alone (`u * d` here, 32), then checked `output.len() / rows`
    /// against `activation.len() / k` — `64 / 32 = 2` (real leading axis
    /// `s`) against `32 / 4 = 8` (a leading count that has already folded in
    /// `u`, which `rows` also claimed) — an off-by-`u`-factor disagreement,
    /// not a coincidence: this is the same "kv-head factor" mismatch the
    /// real checkpoint reduce hits, rejected with the generic
    /// "does not evenly divide" reason.
    ///
    /// After this fix, the rejection is structural rather than a cardinality
    /// coincidence: `resolve_reduce_axis_shape` (shared with [`run_reduce`])
    /// finds `u` nonzero-stride on BOTH the packed weight and the
    /// activation, which is not a shape `run_reduce_quantized`'s one flat
    /// `[rows, k] x [k]` matmul kernel can express (one activation vector
    /// dotted against every packed row). Closing this for real needs a
    /// per-position packed-weight byte offset the interpreter does not have
    /// today — a capability gap, not a shape-arithmetic one — so this test
    /// still asserts a rejection, now for the reason that is actually true.
    #[test]
    fn a_reduce_where_activation_and_packed_weight_share_a_kept_output_axis_is_rejected() {
        use proxima_gguf::quant::q8_0::{BLOCK_BYTES, QK8_0, quantize};

        let sequence_len: u32 = 2; // s -- leading, weight-broadcast
        let cached_len: u32 = 4; // t -- the sole reduced axis
        let kv_heads: u32 = 4; // u -- shared between weight and activation
        let head_dim: u32 = 8; // d -- weight-only row axis

        let mut program = Vec::new();
        let weight = block(
            &mut program,
            DType::UInt8,
            &[Extent::Static(cached_len), Extent::Static(kv_heads), Extent::Static(head_dim)],
        );
        let activation = f32_block(
            &mut program,
            &[Extent::Static(sequence_len), Extent::Static(cached_len), Extent::Static(kv_heads)],
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (weight, IndexMap::Affine(map::projection(4, &[1, 2, 3]))),
                    (activation, IndexMap::Affine(map::projection(4, &[0, 1, 2]))),
                ],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])),
                out_map: IndexMap::Affine(map::projection(4, &[0, 2, 3])),
                keep: Keep::Reduce,
                name: Some("cached_attention_shaped_reduce".into()),
            }),
        );

        let weight_elements = (cached_len * kv_heads * head_dim) as usize;
        assert_eq!(weight_elements % QK8_0, 0, "fixture dims must divide evenly into whole Q8_0 blocks");
        let weight_f32: Vec<f32> = random_vec(29, weight_elements);
        let mut weight_bytes = vec![0u8; (weight_elements / QK8_0) * BLOCK_BYTES];
        quantize(&weight_f32, &mut weight_bytes).expect("fixture weight length is a whole multiple of QK8_0");

        let activation_values: Vec<f32> = random_vec(31, (sequence_len * cached_len * kv_heads) as usize);

        let blocks = [QuantizedBlock::Q8_0(&weight_bytes), QuantizedBlock::Float32(&activation_values)];
        let outcome = evaluate_quantized(&program, &[], &blocks, &[sum]);

        let error = outcome.expect_err(
            "a packed weight operand and its activation sharing a kept output axis is not a flat \
             matmul this interpreter can express -- see this test's own doc",
        );
        assert_eq!(
            error,
            TensorError::NotLowerable {
                node: sum,
                reason: "quantized matmul activation varies along an output axis its packed weight also \
                         varies along -- not a flat weight matmul this interpreter can express",
            }
        );
    }

    // -------------------------------------------------------------
    // `Float16`/`BFloat16` -- composed convert-then-fold kernels
    // (`dot_f16_f32`/`dot_bf16_f32`), hand-computed expected values (never
    // the implementation checked against itself), plus one end-to-end
    // `evaluate_quantized` parity test against `proxima_gguf`'s own
    // dequantize (guiding-principle 14: the dequantize-then-matmul
    // reference is correct by construction).
    // -------------------------------------------------------------

    /// Which half-precision codec a parameterized case exercises -- the two
    /// formats share the same [`dot_f16_f32`]/[`dot_bf16_f32`] +
    /// [`matmul_f16_f32`]/[`matmul_bf16_f32`] call shape but pack a value to
    /// a different bit pattern, so the byte-packing step is the one thing
    /// each case varies.
    #[derive(Clone, Copy, Debug)]
    enum HalfPrecisionKind {
        F16,
        Bf16,
    }

    impl HalfPrecisionKind {
        fn pack(self, values: &[f32]) -> Vec<u8> {
            match self {
                Self::F16 => values.iter().flat_map(|&value| f16::from_f32(value).to_le_bytes()).collect(),
                Self::Bf16 => values.iter().flat_map(|&value| bf16::from_f32(value).to_le_bytes()).collect(),
            }
        }

        fn dot(self, weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
            match self {
                Self::F16 => dot_f16_f32(weight_row, activation),
                Self::Bf16 => dot_bf16_f32(weight_row, activation),
            }
        }

        fn matmul(self, weights: &[u8], rows: usize, activation: &[f32]) -> Result<Vec<f32>, TensorError> {
            match self {
                Self::F16 => matmul_f16_f32(weights, rows, activation),
                Self::Bf16 => matmul_bf16_f32(weights, rows, activation),
            }
        }
    }

    /// `weight . activation` computed by hand: `1.0*2.0 + 2.0*0.5 +
    /// (-1.0)*3.0 + 0.5*4.0 = 2.0 + 1.0 - 3.0 + 2.0 = 2.0`. Every value on
    /// both sides is an exact power of two (or zero), and both binary16 and
    /// bfloat16 represent every power of two in this range exactly, so the
    /// composed convert-then-fold kernel must reproduce `2.0` bit-exactly —
    /// this checks the kernel against arithmetic done by hand, not against
    /// itself.
    #[proxima::test]
    #[case::f16(HalfPrecisionKind::F16)]
    #[case::bf16(HalfPrecisionKind::Bf16)]
    async fn dot_half_precision_matches_a_hand_computed_dot_product(#[case] kind: HalfPrecisionKind) {
        let weight = [1.0f32, 2.0, -1.0, 0.5];
        let activation = [2.0f32, 0.5, 3.0, 4.0];
        let weight_bytes = kind.pack(&weight);

        let actual = kind.dot(&weight_bytes, &activation).expect("well-formed half-precision row");

        assert_eq!(actual, 2.0f32, "hand-computed dot product ({kind:?})");
    }

    /// Two rows, each hand-computed independently: row 0 is the dot-product
    /// fixture above (`2.0`); row 1 is `[0.0, 1.0, 0.0, -2.0] . [2.0, 0.5,
    /// 3.0, 4.0] = 0 + 0.5 + 0 - 8.0 = -7.5` -- again every value an exact
    /// power of two (or zero), so both formats reproduce it bit-exactly.
    #[proxima::test]
    #[case::f16(HalfPrecisionKind::F16)]
    #[case::bf16(HalfPrecisionKind::Bf16)]
    async fn matmul_half_precision_matches_a_hand_computed_two_row_matmul(#[case] kind: HalfPrecisionKind) {
        let row0 = [1.0f32, 2.0, -1.0, 0.5];
        let row1 = [0.0f32, 1.0, 0.0, -2.0];
        let activation = [2.0f32, 0.5, 3.0, 4.0];
        let weights: Vec<f32> = row0.iter().chain(row1.iter()).copied().collect();
        let weight_bytes = kind.pack(&weights);

        let actual = kind.matmul(&weight_bytes, 2, &activation).expect("well-formed 2-row half-precision matmul");

        assert_eq!(actual, alloc::vec![2.0f32, -7.5], "hand-computed 2-row matmul ({kind:?})");
    }

    /// Proves the hand-computed assertion above is load-bearing rather than
    /// vacuous: a deliberately wrong second-row expectation (`123.0` in
    /// place of the hand-computed `-7.5`) checked with `assert_ne!` against
    /// the kernel's real output, so this file itself carries the evidence
    /// that a wrong answer is caught, without needing to hand-edit the test
    /// file to demonstrate it.
    #[proxima::test]
    #[case::f16(HalfPrecisionKind::F16)]
    #[case::bf16(HalfPrecisionKind::Bf16)]
    async fn matmul_half_precision_hand_computed_assertion_can_actually_fail(#[case] kind: HalfPrecisionKind) {
        let row0 = [1.0f32, 2.0, -1.0, 0.5];
        let row1 = [0.0f32, 1.0, 0.0, -2.0];
        let activation = [2.0f32, 0.5, 3.0, 4.0];
        let weights: Vec<f32> = row0.iter().chain(row1.iter()).copied().collect();
        let weight_bytes = kind.pack(&weights);

        let actual = kind.matmul(&weight_bytes, 2, &activation).expect("well-formed 2-row half-precision matmul");
        let deliberately_wrong = alloc::vec![2.0f32, 123.0];

        assert_ne!(
            actual, deliberately_wrong,
            "a deliberately wrong expectation must not match the kernel's real output ({kind:?})"
        );
    }

    /// [`dot_f16_f32`]/[`dot_bf16_f32`] reject a byte length that is not a
    /// whole number of 2-byte elements, and an activation length that does
    /// not match the decoded element count -- never a panic or an
    /// out-of-bounds read.
    #[proxima::test]
    #[case::f16(HalfPrecisionKind::F16)]
    #[case::bf16(HalfPrecisionKind::Bf16)]
    async fn dot_half_precision_rejects_a_malformed_shape(#[case] kind: HalfPrecisionKind) {
        let odd_bytes = alloc::vec![0u8; 3];
        let activation = [0.0f32; 1];
        assert!(matches!(
            kind.dot(&odd_bytes, &activation),
            Err(TensorError::QuantizedShapeMismatch { .. })
        ));

        let weight_bytes = kind.pack(&[1.0, 2.0]);
        let mismatched_activation = [0.0f32; 3];
        assert!(matches!(
            kind.dot(&weight_bytes, &mismatched_activation),
            Err(TensorError::QuantizedShapeMismatch { .. })
        ));
    }

    /// End-to-end: a `Float16`/`BFloat16` [`QuantizedBlock`] weight bound
    /// through [`evaluate_quantized`]'s full `Op` graph (elementwise
    /// multiply feeding an add-reduce -- the matmul shape
    /// `is_quantized_matmul_operand` recognizes), checked against
    /// `proxima_gguf`'s own tested `dequantize` followed by a naive `f32`
    /// dot product -- guiding-principle 14: the dequantize-then-matmul
    /// reference is correct by construction, so this is a parity check
    /// against an independent path, not a round-trip-to-self check. Real
    /// (pseudo-random, non-degenerate -- `Lcg`) weight and activation
    /// values, not zeros or constants.
    #[proxima::test]
    #[case::f16(HalfPrecisionKind::F16)]
    #[case::bf16(HalfPrecisionKind::Bf16)]
    async fn evaluate_quantized_executes_a_half_precision_weight_end_to_end(#[case] kind: HalfPrecisionKind) {
        const ROWS: u32 = 3;
        const K: u32 = 16;
        const K_USIZE: usize = K as usize;

        let weights_f32 = random_vec(41, (ROWS * K) as usize);
        let activation = random_vec(43, K as usize);
        let weight_bytes = kind.pack(&weights_f32);

        let mut program = Vec::new();
        let weight = block(&mut program, DType::UInt8, &[Extent::Static(ROWS), Extent::Static(K)]);
        let activation_node = f32_block(&mut program, &[Extent::Static(K)]);
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (weight, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (activation_node, IndexMap::Affine(map::projection(2, &[1]))),
                ],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                out_map: IndexMap::Affine(map::projection(2, &[0])),
                keep: Keep::Reduce,
                name: Some("half_precision_matmul".into()),
            }),
        );

        let blocks = match kind {
            HalfPrecisionKind::F16 => alloc::vec![QuantizedBlock::Float16(&weight_bytes), QuantizedBlock::Float32(&activation)],
            HalfPrecisionKind::Bf16 => alloc::vec![QuantizedBlock::BFloat16(&weight_bytes), QuantizedBlock::Float32(&activation)],
        };
        let evaluated = evaluate_quantized(&program, &[], &blocks, &[sum]).expect("half-precision matmul evaluates");
        let actual = evaluated.root();

        let mut dequantized = vec![0.0f32; (ROWS * K) as usize];
        match kind {
            HalfPrecisionKind::F16 => {
                proxima_gguf::quant::f16::dequantize(&weight_bytes, &mut dequantized).expect("well-formed f16 bytes");
            }
            HalfPrecisionKind::Bf16 => {
                proxima_gguf::quant::bf16::dequantize(&weight_bytes, &mut dequantized).expect("well-formed bf16 bytes");
            }
        }
        let expected: Vec<f32> = dequantized
            .as_chunks::<K_USIZE>()
            .0
            .iter()
            .map(|row| row.iter().zip(&activation).map(|(weight, value)| weight * value).sum())
            .collect();

        assert_eq!(actual.len(), ROWS as usize, "degenerate gate: no outputs compared");
        // Not `assert_eq!`: `dot_fold_fused_multiply_add`'s `DOT_LANES` (8)
        // independent partial sums (K=16 here is two whole lanes) combine
        // in a different order than this reference's strict left-to-right
        // `Iterator::sum` -- float addition is not associative, so the two
        // legitimately differ in the last mantissa bits. The K4Q4_K parity
        // test above (`matmul_q4k_f32_matches_dequantize_then_f32_matmul`)
        // hits the exact same reordering and uses the same loose-tolerance
        // shape rather than bit-exact equality.
        let mut max_diff = 0.0f32;
        for (got, want) in actual.iter().zip(&expected) {
            assert!(got.is_finite(), "half-precision matmul ({kind:?}) produced a non-finite value: {got}");
            max_diff = max_diff.max((got - want).abs());
        }
        eprintln!("half-precision matmul ({kind:?}) vs dequantize-then-fold: max_diff={max_diff}");
        assert!(
            max_diff < 1e-4,
            "half-precision matmul ({kind:?}) disagrees with the dequantize-then-fold reference: max_diff={max_diff}"
        );
    }

    // -------------------------------------------------------------
    // `Q5_K`/`Q6_K` packed int8 kernels -- bit-exactness (synthetic,
    // both arms on the SAME weight bytes) and correctness against the
    // dequantize-then-fold reference path on REAL packed bytes read
    // straight out of the real openchat-3.5-1210 `Q4_K_S` GGUF file
    // (guiding-principles principle 9: real-world data, not a synthetic
    // stand-in) -- the same discipline `bench_q4k_matmul.rs` applies
    // against ggml, minus the timing: correctness only, per this
    // landing's task scope.
    // -------------------------------------------------------------

    /// Streams `path`'s header/tensor-directory prefix in growing chunks
    /// until [`proxima_gguf::parser::GgufParser`] reports `Complete`,
    /// without ever reading the (multi-GiB) tensor data section -- the
    /// same technique `bench_q4k_matmul.rs::parse_header` uses, duplicated
    /// here rather than shared across the lib/bench boundary (benches are
    /// their own crate roots in this workspace). Returns `None` if the
    /// file does not exist on this host, so these tests degrade to a
    /// no-op on a machine without the real model file rather than a hard
    /// failure.
    #[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
    fn real_gguf_header(path: &std::path::Path) -> Option<(proxima_gguf::pipe::ParsedGguf, u64, std::fs::File)> {
        use std::io::{Read, Seek, SeekFrom};

        use proxima_gguf::parser::{GgufEvent, GgufParser};
        use proxima_gguf::pipe::ParsedGguf;

        let mut file = std::fs::File::open(path).ok()?;
        let file_len = file.metadata().ok()?.len();

        let mut prefix_len = 1usize << 20;
        loop {
            let mut buf = vec![0u8; prefix_len];
            file.seek(SeekFrom::Start(0)).expect("seek to start");
            let read = file.read(&mut buf).expect("read gguf prefix");
            buf.truncate(read);

            if let Ok((parser, events)) = GgufParser::new().push(&buf) {
                let mut version = None;
                let mut metadata = Vec::new();
                let mut tensors = Vec::new();
                let mut completion = None;
                for event in events {
                    match event {
                        GgufEvent::Header { version: version_value, .. } => version = Some(version_value),
                        GgufEvent::Metadata { key, value } => metadata.push((key, value)),
                        GgufEvent::Tensor(tensor) => tensors.push(tensor),
                        GgufEvent::Complete { data_offset, alignment } => {
                            completion = Some((data_offset, alignment));
                        }
                    }
                }
                if let (Some(version), Some((data_offset, alignment))) = (version, completion) {
                    parser.finish().expect("parser reports complete and clean");
                    let parsed = ParsedGguf {
                        version,
                        tensor_count: tensors.len() as u64,
                        kv_count: metadata.len() as u64,
                        metadata,
                        tensors,
                        data_offset,
                        alignment,
                    };
                    return Some((parsed, file_len, file));
                }
            }

            assert!(prefix_len < (1 << 26), "gguf header/directory exceeded 64 MiB prefix budget");
            prefix_len *= 2;
        }
    }

    /// Reads one named tensor's packed bytes off `file` via its validated
    /// absolute byte range, or `None` if `name`/`ggml_type` doesn't match
    /// what's actually in the file (a mixed-precision quant recipe like
    /// `Q4_K_S` doesn't guarantee a given tensor lands at a given codec on
    /// every quantizer version -- reported, not faked, same stance
    /// `bench_q4k_matmul.rs` takes).
    #[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
    fn real_tensor_bytes(
        file: &mut std::fs::File,
        parsed: &proxima_gguf::pipe::ParsedGguf,
        file_len: u64,
        name: &str,
        expect_type: proxima_gguf::types::GgmlType,
    ) -> Option<(Vec<u8>, usize, usize)> {
        use std::io::{Read, Seek, SeekFrom};

        let tensor = parsed.tensors.iter().find(|candidate| candidate.name == name)?;
        if tensor.ggml_type != expect_type {
            eprintln!(
                "real_tensor_bytes: {name} is {:?} in this file, not {expect_type:?} -- test skipped, not faked",
                tensor.ggml_type
            );
            return None;
        }
        let in_dim = tensor.dims[0] as usize;
        let out_dim = tensor.dims[1] as usize;
        let range = parsed.tensor_data_range(tensor, file_len).expect("tensor byte range within file bounds");
        let mut buf = vec![0u8; (range.end - range.start) as usize];
        file.seek(SeekFrom::Start(range.start)).expect("seek to tensor data");
        file.read_exact(&mut buf).expect("read exact tensor byte range");
        Some((buf, in_dim, out_dim))
    }

    #[cfg(any(feature = "q4k-int8-dot", feature = "q5k-int8-dot", feature = "q6k-int8-dot"))]
    const REAL_OPENCHAT_GGUF_PATH: &str =
        "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf";

    /// [`matmul_q5k_q8k_f32`]/[`matmul_q5k_q8k_portable_f32`] agree with
    /// [`matmul_q5k_f32`] (the dequantize-then-fold reference path) on the
    /// SAME packed `Q5_K` bytes read directly out of the real
    /// openchat-3.5-1210 GGUF file -- `blk.0.attn_v.weight`, one of the
    /// two shapes this landing's task names. This is the correctness gate
    /// principle 14 (the incumbent -- here, the already-tested
    /// dequantize path -- wins on correctness) demands BEFORE any timing;
    /// no timing is taken in this test at all.
    #[cfg(feature = "q5k-int8-dot")]
    #[test]
    fn matmul_q5k_q8k_f32_agrees_with_dequantize_then_fold_on_real_gguf_bytes() {
        let path = std::path::Path::new(REAL_OPENCHAT_GGUF_PATH);
        let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
            eprintln!("real gguf file not found at {REAL_OPENCHAT_GGUF_PATH}; test skipped");
            return;
        };
        let Some((weight_bytes, in_dim, out_dim)) = real_tensor_bytes(
            &mut file,
            &parsed,
            file_len,
            "blk.0.attn_v.weight",
            proxima_gguf::types::GgmlType::Q5_K,
        ) else {
            return;
        };

        let activation = random_vec(401, in_dim).into_iter().map(|value| value - 0.5).collect::<Vec<f32>>();

        let expected = matmul_q5k_f32(&weight_bytes, out_dim, &activation).expect("well-formed dequant reference matmul");
        let dispatched = matmul_q5k_q8k_f32(&weight_bytes, out_dim, &activation).expect("well-formed packed int8 matmul");
        let portable = matmul_q5k_q8k_portable_f32(&weight_bytes, out_dim, &activation).expect("well-formed portable matmul");

        assert_eq!(dispatched, portable, "attn_v: dispatched and portable packed-int8 arms diverged on real bytes");

        let mut max_error = 0.0f32;
        let mut sum_sq_error = 0.0f64;
        for (&got, &want) in dispatched.iter().zip(expected.iter()) {
            assert!(got.is_finite(), "packed int8 matmul row produced a non-finite value: {got}");
            let diff = (got - want).abs();
            max_error = max_error.max(diff);
            sum_sq_error += f64::from(diff) * f64::from(diff);
        }
        let rms_error = (sum_sq_error / out_dim as f64).sqrt();
        let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
        let relative_max_error = max_error / max_magnitude;
        eprintln!(
            "attn_v (real Q5_K bytes) packed vs dequant-fold reference: max_error={max_error} \
             rms_error={rms_error} max_magnitude={max_magnitude} relative_max_error={relative_max_error}"
        );
        // Same band `matmul_q4k_q8k_f32_agrees_with_dequantize_then_matmul_within_a_measured_tolerance`
        // uses for its own real-weight relative-error check: Q8_K
        // activation quantization is a second real lossy step neither
        // side of this comparison shares.
        assert!(
            relative_max_error < 0.01,
            "relative_max_error={relative_max_error} (max_error={max_error} over magnitude {max_magnitude}) \
             exceeds loose sanity bound"
        );
    }

    /// The same correctness gate as
    /// [`matmul_q5k_q8k_f32_agrees_with_dequantize_then_fold_on_real_gguf_bytes`],
    /// at this landing's second named `Q5_K` shape -- `blk.0.ffn_down.weight`
    /// (14336x4096).
    #[cfg(feature = "q5k-int8-dot")]
    #[test]
    fn matmul_q5k_q8k_f32_agrees_with_dequantize_then_fold_on_real_gguf_bytes_ffn_down() {
        let path = std::path::Path::new(REAL_OPENCHAT_GGUF_PATH);
        let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
            eprintln!("real gguf file not found at {REAL_OPENCHAT_GGUF_PATH}; test skipped");
            return;
        };
        let Some((weight_bytes, in_dim, out_dim)) = real_tensor_bytes(
            &mut file,
            &parsed,
            file_len,
            "blk.0.ffn_down.weight",
            proxima_gguf::types::GgmlType::Q5_K,
        ) else {
            return;
        };

        let activation = random_vec(402, in_dim).into_iter().map(|value| value - 0.5).collect::<Vec<f32>>();

        let expected = matmul_q5k_f32(&weight_bytes, out_dim, &activation).expect("well-formed dequant reference matmul");
        let dispatched = matmul_q5k_q8k_f32(&weight_bytes, out_dim, &activation).expect("well-formed packed int8 matmul");
        let portable = matmul_q5k_q8k_portable_f32(&weight_bytes, out_dim, &activation).expect("well-formed portable matmul");

        assert_eq!(dispatched, portable, "ffn_down: dispatched and portable packed-int8 arms diverged on real bytes");

        let max_error =
            dispatched.iter().zip(expected.iter()).map(|(&got, &want)| (got - want).abs()).fold(0.0f32, f32::max);
        let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
        let relative_max_error = max_error / max_magnitude;
        eprintln!("ffn_down (real Q5_K bytes) packed vs dequant-fold reference: relative_max_error={relative_max_error}");
        assert!(
            relative_max_error < 0.01,
            "relative_max_error={relative_max_error} exceeds loose sanity bound"
        );
    }

    /// [`matmul_q6k_q8k_f32`]/[`matmul_q6k_q8k_portable_f32`] agree with
    /// [`matmul_q6k_f32`] on the SAME packed `Q6_K` bytes read directly out
    /// of the real openchat-3.5-1210 GGUF file -- `output.weight`
    /// (4096x32002), this landing's named `Q6_K` shape.
    #[cfg(feature = "q6k-int8-dot")]
    #[test]
    fn matmul_q6k_q8k_f32_agrees_with_dequantize_then_fold_on_real_gguf_bytes() {
        let path = std::path::Path::new(REAL_OPENCHAT_GGUF_PATH);
        let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
            eprintln!("real gguf file not found at {REAL_OPENCHAT_GGUF_PATH}; test skipped");
            return;
        };
        let Some((weight_bytes, in_dim, out_dim)) =
            real_tensor_bytes(&mut file, &parsed, file_len, "output.weight", proxima_gguf::types::GgmlType::Q6_K)
        else {
            return;
        };

        let activation = random_vec(403, in_dim).into_iter().map(|value| value - 0.5).collect::<Vec<f32>>();

        let expected = matmul_q6k_f32(&weight_bytes, out_dim, &activation).expect("well-formed dequant reference matmul");
        let dispatched = matmul_q6k_q8k_f32(&weight_bytes, out_dim, &activation).expect("well-formed packed int8 matmul");
        let portable = matmul_q6k_q8k_portable_f32(&weight_bytes, out_dim, &activation).expect("well-formed portable matmul");

        assert_eq!(dispatched, portable, "output.weight: dispatched and portable packed-int8 arms diverged on real bytes");

        let mut max_error = 0.0f32;
        let mut sum_sq_error = 0.0f64;
        for (&got, &want) in dispatched.iter().zip(expected.iter()) {
            assert!(got.is_finite(), "packed int8 matmul row produced a non-finite value: {got}");
            let diff = (got - want).abs();
            max_error = max_error.max(diff);
            sum_sq_error += f64::from(diff) * f64::from(diff);
        }
        let rms_error = (sum_sq_error / out_dim as f64).sqrt();
        let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
        let relative_max_error = max_error / max_magnitude;
        eprintln!(
            "output.weight (real Q6_K bytes) packed vs dequant-fold reference: max_error={max_error} \
             rms_error={rms_error} max_magnitude={max_magnitude} relative_max_error={relative_max_error}"
        );
        assert!(
            relative_max_error < 0.01,
            "relative_max_error={relative_max_error} (max_error={max_error} over magnitude {max_magnitude}) \
             exceeds loose sanity bound"
        );
    }

    /// The position-folding defect this landing fixes: before it,
    /// `run_reduce_quantized` called [`matmul_q4k_q8k_f32`] once per
    /// sequence position, re-streaming the entire `Q4_K` weight matrix
    /// `leading_total` times. [`matmul_q4k_q8k_f32_impl`] now takes
    /// `leading_total` directly and folds every position's dot into one
    /// pass over the weight rows. This test is the one every *existing*
    /// test (all single-position) cannot catch: run the wide path at
    /// `leading_total = 3` on real packed `Q4_K` bytes from the checkpoint,
    /// and check it against `leading_total` separate single-position calls
    /// through the narrow (already-tested) public entry point --
    /// `assert_eq!`, not a tolerance, since folding changes neither the
    /// per-element arithmetic nor its order (still one row-dot over the
    /// same `k` elements per `(row, position)` pair), only which weight
    /// bytes get re-read. Also pins the wide output's own row-major
    /// `[row][position]` layout (`matmul_rows_threaded`'s natural shape,
    /// weight row as the parallel axis) against the narrow path's
    /// position-major results reassembled the same way -- a silent
    /// transpose would show up as a mismatch here, not a tolerance miss.
    #[cfg(feature = "q4k-int8-dot")]
    #[test]
    fn matmul_q4k_q8k_f32_wide_matches_leading_total_separate_narrow_calls_on_real_gguf_bytes() {
        let path = std::path::Path::new(REAL_OPENCHAT_GGUF_PATH);
        let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
            eprintln!("real gguf file not found at {REAL_OPENCHAT_GGUF_PATH}; test skipped");
            return;
        };
        let Some((weight_bytes, in_dim, out_dim)) = real_tensor_bytes(
            &mut file,
            &parsed,
            file_len,
            "blk.0.attn_q.weight",
            proxima_gguf::types::GgmlType::Q4_K,
        ) else {
            return;
        };

        let leading_total = 3usize;
        let activation: Vec<f32> = (0..leading_total)
            .flat_map(|position| {
                random_vec(500 + position as u64, in_dim).into_iter().map(|value| value - 0.5)
            })
            .collect();

        let wide =
            matmul_q4k_q8k_f32_impl(&weight_bytes, out_dim, &activation, leading_total, None).expect("wide fold call");
        assert_eq!(wide.len(), out_dim * leading_total, "wide output is not row-major [row][position]");

        for position in 0..leading_total {
            let activation_row = &activation[position * in_dim..(position + 1) * in_dim];
            let narrow = matmul_q4k_q8k_f32(&weight_bytes, out_dim, activation_row).expect("narrow per-position call");
            for row in 0..out_dim {
                assert_eq!(
                    wide[row * leading_total + position],
                    narrow[row],
                    "row {row} position {position}: folded and per-position paths diverged"
                );
            }
        }
    }

    /// [`matmul_q4k_q8k_f32_wide_matches_leading_total_separate_narrow_calls_on_real_gguf_bytes`]'s
    /// exact mechanism, ported to `Q5_K`: [`matmul_q5k_q8k_f32_impl`] now
    /// takes `leading_total` and folds every position's dot into one pass
    /// over the weight rows, in place of `run_reduce_quantized`'s old
    /// per-position loop re-streaming the whole `Q5_K` weight matrix
    /// `leading_total` times. Run the wide path at `leading_total = 3` on
    /// real packed `Q5_K` bytes (`blk.0.ffn_down.weight`, this crate's named
    /// `Q5_K` shape) and check it against `leading_total` separate
    /// single-position calls through the narrow (already-tested) public
    /// entry point -- `assert_eq!`, not a tolerance, for the same reason the
    /// `Q4_K` test uses one: folding changes neither the per-element
    /// arithmetic nor its order, only which weight bytes get re-read.
    #[cfg(feature = "q5k-int8-dot")]
    #[test]
    fn matmul_q5k_q8k_f32_wide_matches_leading_total_separate_narrow_calls_on_real_gguf_bytes() {
        let path = std::path::Path::new(REAL_OPENCHAT_GGUF_PATH);
        let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
            eprintln!("real gguf file not found at {REAL_OPENCHAT_GGUF_PATH}; test skipped");
            return;
        };
        let Some((weight_bytes, in_dim, out_dim)) = real_tensor_bytes(
            &mut file,
            &parsed,
            file_len,
            "blk.0.ffn_down.weight",
            proxima_gguf::types::GgmlType::Q5_K,
        ) else {
            return;
        };

        let leading_total = 3usize;
        let activation: Vec<f32> = (0..leading_total)
            .flat_map(|position| {
                random_vec(600 + position as u64, in_dim).into_iter().map(|value| value - 0.5)
            })
            .collect();

        let wide =
            matmul_q5k_q8k_f32_impl(&weight_bytes, out_dim, &activation, leading_total, None).expect("wide fold call");
        assert_eq!(wide.len(), out_dim * leading_total, "wide output is not row-major [row][position]");

        for position in 0..leading_total {
            let activation_row = &activation[position * in_dim..(position + 1) * in_dim];
            let narrow = matmul_q5k_q8k_f32(&weight_bytes, out_dim, activation_row).expect("narrow per-position call");
            for row in 0..out_dim {
                assert_eq!(
                    wide[row * leading_total + position],
                    narrow[row],
                    "row {row} position {position}: folded and per-position paths diverged"
                );
            }
        }
    }

    /// [`matmul_q4k_q8k_f32_wide_matches_leading_total_separate_narrow_calls_on_real_gguf_bytes`]'s
    /// exact mechanism, ported to `Q6_K`: [`matmul_q6k_q8k_f32_impl`] now
    /// takes `leading_total` and folds every position's dot into one pass
    /// over the weight rows. Run the wide path at `leading_total = 3` on
    /// real packed `Q6_K` bytes (`output.weight`, this crate's named `Q6_K`
    /// shape) and check it against `leading_total` separate single-position
    /// calls through the narrow (already-tested) public entry point --
    /// `assert_eq!`, not a tolerance, same reasoning as the `Q4_K`/`Q5_K`
    /// tests.
    #[cfg(feature = "q6k-int8-dot")]
    #[test]
    fn matmul_q6k_q8k_f32_wide_matches_leading_total_separate_narrow_calls_on_real_gguf_bytes() {
        let path = std::path::Path::new(REAL_OPENCHAT_GGUF_PATH);
        let Some((parsed, file_len, mut file)) = real_gguf_header(path) else {
            eprintln!("real gguf file not found at {REAL_OPENCHAT_GGUF_PATH}; test skipped");
            return;
        };
        let Some((weight_bytes, in_dim, out_dim)) = real_tensor_bytes(
            &mut file,
            &parsed,
            file_len,
            "output.weight",
            proxima_gguf::types::GgmlType::Q6_K,
        ) else {
            return;
        };

        let leading_total = 3usize;
        let activation: Vec<f32> = (0..leading_total)
            .flat_map(|position| {
                random_vec(700 + position as u64, in_dim).into_iter().map(|value| value - 0.5)
            })
            .collect();

        let wide =
            matmul_q6k_q8k_f32_impl(&weight_bytes, out_dim, &activation, leading_total, None).expect("wide fold call");
        assert_eq!(wide.len(), out_dim * leading_total, "wide output is not row-major [row][position]");

        for position in 0..leading_total {
            let activation_row = &activation[position * in_dim..(position + 1) * in_dim];
            let narrow = matmul_q6k_q8k_f32(&weight_bytes, out_dim, activation_row).expect("narrow per-position call");
            for row in 0..out_dim {
                assert_eq!(
                    wide[row * leading_total + position],
                    narrow[row],
                    "row {row} position {position}: folded and per-position paths diverged"
                );
            }
        }
    }

    /// [`dot_q5k_q8k_block_neon_dotprod`]'s whole justification: it must
    /// be an ACCELERATION of [`dot_q5k_q8k_block_scalar`], not a different
    /// mechanism -- same bit-exactness argument
    /// `matmul_q4k_q8k_f32_agrees_bit_exact_with_the_portable_arm` makes
    /// for `Q4_K` (every intermediate is integer until the final `f32`
    /// multiply, so both arms must match EXACTLY, not merely closely), on
    /// synthetic multi-row, multi-block data (not real-file bytes -- this
    /// test's job is arm-vs-arm agreement, not real-weight correctness,
    /// which the two tests above already cover).
    #[cfg(feature = "q5k-int8-dot")]
    #[test]
    fn matmul_q5k_q8k_f32_agrees_bit_exact_with_the_portable_arm() {
        use proxima_gguf::quant::q5_k::{BLOCK_BYTES, QK_K, quantize};

        let rows = 4;
        let blocks_per_row = 5;
        let k = QK_K * blocks_per_row;

        let activation: Vec<f32> = random_vec(23, k).into_iter().map(|value| value * 6.0 - 3.0).collect();
        let weight_f32: Vec<f32> = random_vec(29, rows * k).into_iter().map(|value| value * 6.0 - 3.0).collect();

        let mut weight_blocks = vec![0u8; rows * blocks_per_row * BLOCK_BYTES];
        for (row_f32, row_blocks) in weight_f32
            .chunks_exact(k)
            .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
        {
            quantize(row_f32, row_blocks).expect("row length is a whole multiple of QK_K by construction");
        }

        let dispatched = matmul_q5k_q8k_f32(&weight_blocks, rows, &activation).expect("well-formed dispatched matmul");
        let portable = matmul_q5k_q8k_portable_f32(&weight_blocks, rows, &activation).expect("well-formed portable matmul");

        assert_eq!(dispatched, portable, "dispatched and portable arms diverged -- not merely an acceleration");
    }

    /// [`dot_q6k_q8k_block_neon_dotprod`]'s equivalent bit-exactness proof.
    #[cfg(feature = "q6k-int8-dot")]
    #[test]
    fn matmul_q6k_q8k_f32_agrees_bit_exact_with_the_portable_arm() {
        use proxima_gguf::quant::q6_k::{BLOCK_BYTES, QK_K, quantize};

        let rows = 4;
        let blocks_per_row = 5;
        let k = QK_K * blocks_per_row;

        let activation: Vec<f32> = random_vec(31, k).into_iter().map(|value| value * 6.0 - 3.0).collect();
        let weight_f32: Vec<f32> = random_vec(37, rows * k).into_iter().map(|value| value * 6.0 - 3.0).collect();

        let mut weight_blocks = vec![0u8; rows * blocks_per_row * BLOCK_BYTES];
        for (row_f32, row_blocks) in weight_f32
            .chunks_exact(k)
            .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
        {
            quantize(row_f32, row_blocks).expect("row length is a whole multiple of QK_K by construction");
        }

        let dispatched = matmul_q6k_q8k_f32(&weight_blocks, rows, &activation).expect("well-formed dispatched matmul");
        let portable = matmul_q6k_q8k_portable_f32(&weight_blocks, rows, &activation).expect("well-formed portable matmul");

        assert_eq!(dispatched, portable, "dispatched and portable arms diverged -- not merely an acceleration");
    }

    /// [`QuantizedBlock::Q5K`] routes through [`evaluate_quantized`] end to
    /// end -- the same shape
    /// [`evaluate_quantized_matches_dequantize_then_evaluate_within_a_measured_tolerance`]
    /// proves for `Q4K`, one variant over: a `Reduce(Elementwise(Multiply))`
    /// matmul node bound to packed `Q5_K` bytes must agree with binding the
    /// SAME bytes dequantized to plain `f32`.
    #[cfg(feature = "q5k-int8-dot")]
    #[test]
    fn evaluate_quantized_routes_q5k_block_and_matches_dequantize_then_evaluate() {
        use proxima_gguf::quant::q5_k::{BLOCK_BYTES, QK_K, dequantize, quantize};

        let rows: u32 = 6;
        let blocks_per_row = 3;
        let k = QK_K as u32 * blocks_per_row as u32;

        let activation = random_vec(43, k as usize);
        let weight_f32: Vec<f32> = random_vec(47, rows as usize * k as usize);

        let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
        for (row_f32, row_blocks) in weight_f32
            .chunks_exact(k as usize)
            .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
        {
            quantize(row_f32, row_blocks).expect("row length is a whole multiple of QK_K by construction");
        }

        let (program, sum) = quantized_matmul_program(rows, k);
        let blocks = [QuantizedBlock::Q5K(&weight_blocks), QuantizedBlock::Float32(&activation)];
        let quantized_result =
            evaluate_quantized(&program, &[], &blocks, &[sum]).expect("q5_k-quantized matmul evaluates");

        let mut dequantized_weight = vec![0.0f32; rows as usize * k as usize];
        for (row_blocks, row_f32) in weight_blocks
            .chunks_exact(blocks_per_row * BLOCK_BYTES)
            .zip(dequantized_weight.chunks_exact_mut(k as usize))
        {
            dequantize(row_blocks, row_f32).expect("row_blocks is a whole number of q5_k super-blocks");
        }

        let (f32_program, f32_sum) = matmul_program(rows, k, 1, false);
        let f32_blocks: [&[f32]; 2] = [&dequantized_weight, &activation];
        let f32_result =
            evaluate(&f32_program, &[], &f32_blocks, &[f32_sum]).expect("dequantized f32 matmul evaluates");

        let actual = quantized_result.root();
        let expected = f32_result.root();
        assert_eq!(actual.len(), rows as usize);
        assert_eq!(actual.len(), expected.len());

        let max_diff = actual.iter().zip(expected.iter()).map(|(&got, &want)| (got - want).abs()).fold(0.0f32, f32::max);
        let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
        let relative_max_diff = max_diff / max_magnitude;
        eprintln!("evaluate_quantized (Q5K) vs dequantize-then-evaluate: relative_max_diff={relative_max_diff}");
        assert!(
            relative_max_diff < 0.01,
            "relative_max_diff={relative_max_diff} (max_diff={max_diff} over magnitude {max_magnitude}) \
             exceeds loose sanity bound"
        );
    }

    /// The union of every dedicated codec test above, parameterized rather
    /// than copy-pasted: one `[rows, k] x [k, 1]` matmul, one seeded weight
    /// and activation, run through whichever evaluator that codec actually
    /// reaches -- plain [`evaluate`] for `float32` (packing it as a
    /// [`QuantizedBlock::Float32`] *weight* is rejected outright by
    /// `run_reduce_quantized`, see the `shape_error()` arm a few hundred
    /// lines up, so `float32` is not an `evaluate_quantized` cell at all),
    /// [`evaluate_quantized`] for the four packed codecs -- and compared
    /// against the same `f32` reference every one of those dedicated tests
    /// already computes independently. A codec dropped from this list is a
    /// missing `#[case::...]` line, not a missing whole function.
    ///
    /// Tolerance is RELATIVE to the reference's own magnitude, never a flat
    /// absolute epsilon: `float32` is an exact self-consistency check (same
    /// bytes, same evaluator, twice), while every packed codec's activation
    /// additionally folds through a lossy int8 quantization step on the CPU
    /// path (`matmul_q4k_q8k_f32`-family), so a single absolute bound across
    /// all five cells would be either vacuous for `float32` or spuriously
    /// red for the packed codecs.
    #[proxima::test]
    #[case::float32("float32", 1e-6)]
    #[case::q4_k("q4_k", 0.01)]
    #[case::q5_k("q5_k", 0.01)]
    #[case::q6_k("q6_k", 0.01)]
    #[case::q8_0("q8_0", 0.01)]
    // q4_0 is the coarsest codec under test here -- one scale per block, no
    // k-quant sub-block min/scale pair -- so it alone needed a wider bound
    // once `test_support::Lcg::next_unit`'s own range-halving bug (see that
    // function's doc) was fixed: the old bug never generated near-zero
    // weights, and near-zero values are where a single-scale codec's
    // relative error is largest. `0.0104` measured against the corrected,
    // zero-crossing input; `0.012` leaves headroom without loosening the
    // other four codecs' tighter, still-met `0.01`.
    #[case::q4_0("q4_0", 0.012)]
    async fn evaluate_quantized_matmul_matches_dequantized_reference_across_every_codec(
        #[case] codec: &str,
        #[case] tolerance: f32,
    ) {
        let rows: u32 = 5;
        let k: u32 = 768;

        let weight_f32: Vec<f32> =
            random_vec(17, rows as usize * k as usize).into_iter().map(|value| value * 4.0 - 2.0).collect();
        let activation: Vec<f32> = random_vec(13, k as usize).into_iter().map(|value| value * 4.0 - 2.0).collect();

        let (f32_program, f32_sum) = matmul_program(rows, k, 1, false);
        let reference = evaluate(&f32_program, &[], &[&weight_f32, &activation], &[f32_sum])
            .expect("f32 reference matmul evaluates");

        let actual: Vec<f32> = match codec {
            "float32" => reference.root().to_vec(),
            "q4_k" => {
                use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};
                let blocks_per_row = k as usize / QK_K;
                let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
                for (row_f32, row_blocks) in weight_f32
                    .chunks_exact(k as usize)
                    .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
                {
                    quantize(row_f32, row_blocks).expect("row length is a whole multiple of QK_K by construction");
                }
                let (program, sum) = quantized_matmul_program(rows, k);
                let blocks = [QuantizedBlock::Q4K(&weight_blocks), QuantizedBlock::Float32(&activation)];
                evaluate_quantized(&program, &[], &blocks, &[sum])
                    .expect("q4_k-quantized matmul evaluates")
                    .root()
                    .to_vec()
            }
            "q5_k" => {
                use proxima_gguf::quant::q5_k::{BLOCK_BYTES, QK_K, quantize};
                let blocks_per_row = k as usize / QK_K;
                let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
                for (row_f32, row_blocks) in weight_f32
                    .chunks_exact(k as usize)
                    .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
                {
                    quantize(row_f32, row_blocks).expect("row length is a whole multiple of QK_K by construction");
                }
                let (program, sum) = quantized_matmul_program(rows, k);
                let blocks = [QuantizedBlock::Q5K(&weight_blocks), QuantizedBlock::Float32(&activation)];
                evaluate_quantized(&program, &[], &blocks, &[sum])
                    .expect("q5_k-quantized matmul evaluates")
                    .root()
                    .to_vec()
            }
            "q6_k" => {
                use proxima_gguf::quant::q6_k::{BLOCK_BYTES, QK_K, quantize};
                let blocks_per_row = k as usize / QK_K;
                let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
                for (row_f32, row_blocks) in weight_f32
                    .chunks_exact(k as usize)
                    .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
                {
                    quantize(row_f32, row_blocks).expect("row length is a whole multiple of QK_K by construction");
                }
                let (program, sum) = quantized_matmul_program(rows, k);
                let blocks = [QuantizedBlock::Q6K(&weight_blocks), QuantizedBlock::Float32(&activation)];
                evaluate_quantized(&program, &[], &blocks, &[sum])
                    .expect("q6_k-quantized matmul evaluates")
                    .root()
                    .to_vec()
            }
            "q8_0" => {
                use proxima_gguf::quant::q8_0::{BLOCK_BYTES, QK8_0, quantize};
                let blocks_per_row = k as usize / QK8_0;
                let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
                for (row_f32, row_blocks) in weight_f32
                    .chunks_exact(k as usize)
                    .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
                {
                    quantize(row_f32, row_blocks).expect("row length is a whole multiple of QK8_0 by construction");
                }
                let (program, sum) = quantized_matmul_program(rows, k);
                let blocks = [QuantizedBlock::Q8_0(&weight_blocks), QuantizedBlock::Float32(&activation)];
                evaluate_quantized(&program, &[], &blocks, &[sum])
                    .expect("q8_0-quantized matmul evaluates")
                    .root()
                    .to_vec()
            }
            "q4_0" => {
                use proxima_gguf::quant::q4_0::{BLOCK_BYTES, QK4_0, quantize};
                let blocks_per_row = k as usize / QK4_0;
                let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * BLOCK_BYTES];
                for (row_f32, row_blocks) in weight_f32
                    .chunks_exact(k as usize)
                    .zip(weight_blocks.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
                {
                    quantize(row_f32, row_blocks).expect("row length is a whole multiple of QK4_0 by construction");
                }
                let (program, sum) = quantized_matmul_program(rows, k);
                let blocks = [QuantizedBlock::Q4_0(&weight_blocks), QuantizedBlock::Float32(&activation)];
                evaluate_quantized(&program, &[], &blocks, &[sum])
                    .expect("q4_0-quantized matmul evaluates")
                    .root()
                    .to_vec()
            }
            other => panic!("unhandled codec case in this matrix: {other}"),
        };

        let expected = reference.root();
        assert_eq!(actual.len(), rows as usize, "degenerate gate: no outputs compared");
        assert_eq!(actual.len(), expected.len());

        let max_diff =
            actual.iter().zip(expected.iter()).map(|(&got, &want)| (got - want).abs()).fold(0.0f32, f32::max);
        let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
        let relative = max_diff / max_magnitude;
        eprintln!("{codec}: relative={relative} tolerance={tolerance} (max_diff={max_diff} max_magnitude={max_magnitude})");
        assert!(
            relative <= tolerance,
            "{codec}: relative diff {relative} exceeds tolerance {tolerance} -- max_diff={max_diff} \
             max_magnitude={max_magnitude}"
        );
    }

    /// Fixed-topology (structured) sparsity needs no gate at all: the
    /// nonzero pattern here is two disjoint 2x2 blocks fixed at graph-build
    /// time, and each block is a plain, unshifted `IndexMap::Affine`
    /// projection -- never `IndexMap::Computed`. The zero off-diagonal
    /// blocks of the dense 4x4 equivalent are never built as ops at all, so
    /// this costs 2 * (2*2) = 8 multiply-adds against the 16 a dense matmul
    /// would spend, and nothing here is data-dependent, so
    /// `shape.rs:166`'s scatter gate never sees this program.
    #[test]
    fn a_static_block_sparse_matmul_needs_no_data_dependent_map() {
        let mut program = Vec::new();
        let x_block0 = f32_block(&mut program, &[Extent::Static(2)]);
        let x_block1 = f32_block(&mut program, &[Extent::Static(2)]);
        let weight_block0 = f32_block(&mut program, &[Extent::Static(2), Extent::Static(2)]);
        let weight_block1 = f32_block(&mut program, &[Extent::Static(2), Extent::Static(2)]);

        let block_output = |program: &mut Vec<Op>, weight: NodeId, x: NodeId| {
            let product = append(
                program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Multiply,
                    operands: alloc::vec![
                        (weight, IndexMap::Affine(map::projection(2, &[0, 1]))),
                        (x, IndexMap::Affine(map::projection(2, &[1]))),
                    ],
                    name: None,
                },
            );
            append(
                program,
                Op::Reduce(Reduce {
                    dtype: DType::Float32,
                    body: ScalarOp::Add,
                    init: ReduceInit::Zero,
                    operand: product,
                    in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                    out_map: IndexMap::Affine(map::projection(2, &[0])),
                    keep: Keep::Reduce,
                    name: None,
                }),
            )
        };

        let block0_output = block_output(&mut program, weight_block0, x_block0);
        let block1_output = block_output(&mut program, weight_block1, x_block1);

        let x0 = [1.0f32, 2.0];
        let x1 = [3.0f32, 4.0];
        let weight0 = [2.0f32, 1.0, 1.0, 2.0];
        let weight1 = [1.0f32, 1.0, 1.0, -1.0];
        let evaluated = evaluate(
            &program,
            &[],
            &[&x0, &x1, &weight0, &weight1],
            &[block0_output, block1_output],
        )
        .expect("static block-sparse matmul lowers and evaluates");

        let (block0, _) = evaluated.get(block0_output).expect("block0 output present");
        let (block1, _) = evaluated.get(block1_output).expect("block1 output present");
        assert_eq!(block0, &[4.0, 5.0], "weight0 @ (x0, x1)");
        assert_eq!(block1, &[7.0, -1.0], "weight1 @ (x2, x3)");
    }

    /// The test above's own `weight0`/`weight1` are both symmetric matrices
    /// (`[[2,1],[1,2]]`, `[[1,1],[1,-1]]`), so a row/col axis-order bug in
    /// `block_output`'s `projection(2, &[0, 1])` weight read -- transposing
    /// which axis is "row" (kept in `out_map`) and which is "col" (reduced)
    /// -- would still land on the exact same numbers and pass silently: the
    /// same shape of blind spot `causal_conv1d`'s `embedding=1` fixture had,
    /// here from symmetric data rather than a degenerate extent. This uses
    /// deliberately asymmetric `2x2` blocks (`weight0 @ x0` and its
    /// transpose disagree) so that exact bug is observable.
    #[test]
    fn a_static_block_sparse_matmul_catches_a_transposed_block_weight() {
        let mut program = Vec::new();
        let x_block0 = f32_block(&mut program, &[Extent::Static(2)]);
        let x_block1 = f32_block(&mut program, &[Extent::Static(2)]);
        let weight_block0 = f32_block(&mut program, &[Extent::Static(2), Extent::Static(2)]);
        let weight_block1 = f32_block(&mut program, &[Extent::Static(2), Extent::Static(2)]);

        let block_output = |program: &mut Vec<Op>, weight: NodeId, x: NodeId| {
            let product = append(
                program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Multiply,
                    operands: alloc::vec![
                        (weight, IndexMap::Affine(map::projection(2, &[0, 1]))),
                        (x, IndexMap::Affine(map::projection(2, &[1]))),
                    ],
                    name: None,
                },
            );
            append(
                program,
                Op::Reduce(Reduce {
                    dtype: DType::Float32,
                    body: ScalarOp::Add,
                    init: ReduceInit::Zero,
                    operand: product,
                    in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                    out_map: IndexMap::Affine(map::projection(2, &[0])),
                    keep: Keep::Reduce,
                    name: None,
                }),
            )
        };

        let block0_output = block_output(&mut program, weight_block0, x_block0);
        let block1_output = block_output(&mut program, weight_block1, x_block1);

        // weight0 = [[1, 2], [3, 4]], weight1 = [[5, 6], [7, 8]] -- neither
        // symmetric, so weight @ x and weight^T @ x disagree below.
        let x0 = [1.0f32, 0.0];
        let x1 = [0.0f32, 1.0];
        let weight0 = [1.0f32, 2.0, 3.0, 4.0];
        let weight1 = [5.0f32, 6.0, 7.0, 8.0];
        let evaluated = evaluate(
            &program,
            &[],
            &[&x0, &x1, &weight0, &weight1],
            &[block0_output, block1_output],
        )
        .expect("static block-sparse matmul lowers and evaluates");

        let (block0, _) = evaluated.get(block0_output).expect("block0 output present");
        let (block1, _) = evaluated.get(block1_output).expect("block1 output present");
        // weight0 @ x0 = [1*1+2*0, 3*1+4*0] = [1, 3]; the transposed
        // reading would give [1*1+3*0, 2*1+4*0] = [1, 2] instead.
        assert_eq!(block0, &[1.0, 3.0], "weight0 @ x0, not weight0^T @ x0");
        // weight1 @ x1 = [5*0+6*1, 7*0+8*1] = [6, 8]; transposed would give
        // [6, 7] instead.
        assert_eq!(block1, &[6.0, 8.0], "weight1 @ x1, not weight1^T @ x1");
    }

    /// The adjoint of a gather -- and gradient accumulation into a
    /// fixed-shape destination -- is expressible today with the same three
    /// generators the crate's own causal-mask idiom already composes
    /// (`Iota` plus `Equal` builds the selector; see `op.rs`'s `Iota` doc),
    /// never `IndexMap::Computed` as an `out_map`. The destination extent
    /// (`3` here) comes from `Iota`'s own `extent` field -- the same
    /// externally-supplied-extent mechanism `Op::Input`'s leaf shape already
    /// uses -- not from `shape.rs`'s iteration-space unification, which is
    /// why this needs no change to the `is_data_dependent` gate at
    /// `shape.rs:166`. Source rows 0 and 2 both target destination row 0,
    /// which is the whole difference between a scatter-add and a
    /// scatter-write: `Reduce`'s own `body: Add` sums both contributions
    /// instead of one clobbering the other.
    #[test]
    fn scatter_add_into_a_known_destination_via_mask_composition() {
        let mut program = Vec::new();
        let destination_positions = append(
            &mut program,
            Op::Iota {
                dtype: DType::Float32,
                extent: Extent::Static(3),
            },
        );
        let indices = f32_block(&mut program, &[Extent::Static(4)]);
        let source = f32_block(&mut program, &[Extent::Static(4)]);

        let mask = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Equal,
                operands: alloc::vec![
                    (destination_positions, IndexMap::Affine(map::projection(2, &[0]))),
                    (indices, IndexMap::Affine(map::projection(2, &[1]))),
                ],
                name: None,
            },
        );
        let masked_source = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (mask, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (source, IndexMap::Affine(map::projection(2, &[1]))),
                ],
                name: None,
            },
        );
        let scattered = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: masked_source,
                in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                out_map: IndexMap::Affine(map::projection(2, &[0])),
                keep: Keep::Reduce,
                name: Some("scatter_add".into()),
            }),
        );

        let index_values = [0.0f32, 2.0, 0.0, 1.0];
        let source_values = [10.0f32, 20.0, 30.0, 40.0];
        let evaluated = evaluate(&program, &[], &[&index_values, &source_values], &[scattered])
            .expect("scatter-add composed from Iota+Equal+Multiply+Reduce lowers and evaluates");

        assert_eq!(
            evaluated.root(),
            &[40.0, 40.0, 20.0],
            "dest[0]=src[0]+src[2] (collision), dest[1]=src[3], dest[2]=src[1]"
        );
    }
}
