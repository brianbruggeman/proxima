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
use core::any::TypeId;
use core::cell::RefCell;
use core::future::Future;
use core::num::NonZeroUsize;
use core::ops::Deref;
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
use core::sync::atomic::{AtomicU64, Ordering};
use std::borrow::Cow;
#[cfg(not(feature = "tensor-bgpool"))]
use std::panic;
#[cfg(not(feature = "tensor-bgpool"))]
use std::thread;
#[cfg(feature = "instrument")]
use std::time::Instant;
#[cfg(feature = "tensor-bgpool")]
use std::sync::atomic::AtomicUsize;
#[cfg(feature = "tensor-bgpool")]
use std::sync::mpsc::{sync_channel, SyncSender};
#[cfg(feature = "tensor-bgpool")]
use std::sync::{Arc, OnceLock};

use proxima_primitives::pipe::Pipe;
#[cfg(feature = "instrument")]
use proxima_telemetry::counter;
#[cfg(feature = "tensor-bgpool")]
use prime::os::background::ProximaBackgroundPool;

use crate::bind::{self, BoundOp, BoundOpKind, ComposedBody, ReadyBatch, StepArg};
use crate::dtype::DType;
use crate::error::TensorError;
#[cfg(feature = "instrument")]
use crate::instrument;
#[cfg(feature = "instrument")]
use crate::instrument::{KernelCounters, Path};
use crate::map::IndexMap;
use crate::op::{Keep, NodeId, Op, ReduceInit, ScalarOp};
use crate::shape;

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
/// this: `typed_program_dtype` requires one uniform dtype across the
/// *whole* program, but a quantized matmul is deliberately mixed —
/// `UInt8`-packed weight times `Float32` activation into a `Float32`
/// output — which is exactly the shape `reject_non_float32`'s
/// quantized-weight exemption carves out, not a program `evaluate_typed`
/// would ever accept.
pub fn evaluate_quantized(
    program: &[Op],
    symbols: &[u64],
    blocks: &[QuantizedBlock],
    outputs: &[NodeId],
) -> Result<Evaluated, TensorError> {
    let shapes = shape::infer(program, symbols)?;
    let block_nodes = block_node_ids(program);
    if blocks.len() != block_nodes.len() {
        return Err(TensorError::InputCountMismatch {
            expected: block_nodes.len(),
            found: blocks.len(),
        });
    }

    let mut quantized_weights: BTreeMap<NodeId, &[u8]> = BTreeMap::new();
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
            QuantizedBlock::Q4K(bytes) => {
                quantized_weights.insert(*node, bytes);
            }
        }
    }

    let quantized_weight_nodes: BTreeSet<NodeId> = quantized_weights.keys().copied().collect();
    reject_non_float32(program, &quantized_weight_nodes)?;

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

    let resolved = bind::bind(program, &shapes, &effective_outputs)?;
    let retires = node_retirement(&resolved, &effective_outputs);

    let mut peak_live_buffers = live_count(&buffers);
    for (position, computed) in resolved.iter().enumerate() {
        let mut output = vec![0.0f32; node_output_len(computed)];
        run_node_into(computed, &buffers, Some(&quantized_weights), &mut output)?;
        buffers[computed.node.0 as usize] = Some(Cow::Owned(output));
        peak_live_buffers = peak_live_buffers.max(live_count(&buffers));
        for retired in &retires[position] {
            buffers[retired.0 as usize] = None;
        }
    }

    Ok(finish(&shapes, &effective_outputs, buffers, root, peak_live_buffers))
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

    let mut peak_live_buffers = live_count(&buffers);
    for (position, computed) in resolved.iter().enumerate() {
        #[cfg(feature = "instrument")]
        let alloc_site_guard =
            instrument::AllocSiteGuard::enter(instrument::AllocSite::OutputBuffer);
        let mut output = take_or_allocate(free_buffers, node_output_len(computed));
        #[cfg(feature = "instrument")]
        drop(alloc_site_guard);
        run_node_into(computed, &buffers, None, &mut output)?;
        #[cfg(feature = "instrument")]
        record_bound_op_operand_access(computed, &buffers);
        buffers[computed.node.0 as usize] = Some(Cow::Owned(output));
        peak_live_buffers = peak_live_buffers.max(live_count(&buffers));
        for retired in &retires[position] {
            retire_into(&mut buffers, *retired, free_buffers);
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
fn retire_into(buffers: &mut [Option<Cow<'_, [f32]>>], node: NodeId, pool: &mut Vec<Vec<f32>>) {
    if let Some(Cow::Owned(buffer)) = buffers[node.0 as usize].take() {
        pool.push(buffer);
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
/// off under `tensor-bgpool`'s dynamic claiming (see `claim_and_run`): the
/// default `thread::scope` sibling spawns one OS thread per chunk, so
/// raising this on that path spawns more threads for the same work rather
/// than letting any thread steal another's slack — the puller count that
/// path uses is fixed at `chunks.len()` by construction, unlike the pool
/// sibling below, whose puller count [`run_chunks_threaded`] caps at
/// `workers` regardless of `OVERSUBSCRIBE`.
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
/// oversubscription effect outside it. Gating to `1` outside `tensor-bgpool`
/// regardless: the default `thread::scope` sibling spawns one OS thread per
/// chunk, so raising this there would spawn more threads for the same work
/// rather than letting any of them steal another's slack (see this
/// constant's own first paragraph) — that argument does not depend on
/// whichever value the pool path settles on.
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
/// runs its chunks across `workers` OS threads via [`std::thread::scope`],
/// each writing a disjoint sub-slice of that nest's own output buffer (see
/// [`BoundOp::split`]). The preamble is `prepare`, the same one [`evaluate`]
/// runs — the two functions diverge only in the loop below.
pub fn evaluate_parallel(
    program: &[Op],
    symbols: &[u64],
    blocks: &[&[f32]],
    outputs: &[NodeId],
    workers: NonZeroUsize,
) -> Result<Evaluated, TensorError> {
    #[cfg(feature = "instrument")]
    let evaluate_parallel_start = Instant::now();

    #[cfg(feature = "instrument")]
    let prepare_start = Instant::now();
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
        instrument::SERIAL_PREPARE_NANOS,
        prepare_start.elapsed().as_nanos() as u64
    );

    let mut peak_live_buffers = live_count(&buffers);
    for (position, computed) in resolved.iter().enumerate() {
        let output = evaluate_node_parallel(computed, &buffers, workers)?;
        #[cfg(feature = "instrument")]
        let bookkeeping_start = Instant::now();
        buffers[computed.node.0 as usize] = Some(Cow::Owned(output));
        peak_live_buffers = peak_live_buffers.max(live_count(&buffers));
        for retired in &retires[position] {
            buffers[retired.0 as usize] = None;
        }
        #[cfg(feature = "instrument")]
        counter!(
            instrument::SERIAL_BOOKKEEPING_NANOS,
            bookkeeping_start.elapsed().as_nanos() as u64
        );
    }

    #[cfg(feature = "instrument")]
    let finish_start = Instant::now();
    let evaluated = finish(&shapes, &effective_outputs, buffers, root, peak_live_buffers);
    #[cfg(feature = "instrument")]
    {
        counter!(
            instrument::SERIAL_FINISH_NANOS,
            finish_start.elapsed().as_nanos() as u64
        );
        counter!(
            instrument::SERIAL_EVALUATE_PARALLEL_NANOS,
            evaluate_parallel_start.elapsed().as_nanos() as u64
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
    let alloc_start = Instant::now();
    let mut output = vec![0.0f32; node_output_len(resolved)];
    #[cfg(feature = "instrument")]
    counter!(
        instrument::SERIAL_ALLOC_NANOS,
        alloc_start.elapsed().as_nanos() as u64
    );
    #[cfg(feature = "instrument")]
    drop(alloc_site_guard);

    #[cfg(feature = "instrument")]
    let split_start = Instant::now();
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
        instrument::SERIAL_SPLIT_NANOS,
        split_start.elapsed().as_nanos() as u64
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
            let sequential_start = Instant::now();
            run_node_into(resolved, buffers, None, &mut output)?;
            #[cfg(feature = "instrument")]
            counter!(
                instrument::SERIAL_SEQUENTIAL_COMPUTE_NANOS,
                sequential_start.elapsed().as_nanos() as u64
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

/// Runs each of `chunks` on its own OS thread, writing into the matching
/// disjoint sub-slice of `output` — sound because [`BoundOp::split`] documents
/// (and this function relies on) chunk `k`'s output occupying a contiguous,
/// non-overlapping range of the parent buffer.
#[cfg(not(feature = "tensor-bgpool"))]
fn run_chunks_threaded<B: Deref<Target = [f32]> + Sync>(
    chunks: &[BoundOp],
    buffers: &[Option<B>],
    output: &mut [f32],
    // unused here: this path always spawns one OS thread per chunk, so
    // there is no separate "puller count" to cap — see the `tensor-bgpool`
    // sibling below, where the two counts genuinely diverge.
    _workers: NonZeroUsize,
) -> Result<(), TensorError> {
    #[cfg(feature = "instrument")]
    let slice_carve_start = Instant::now();
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
        instrument::SERIAL_SLICE_CARVE_NANOS,
        slice_carve_start.elapsed().as_nanos() as u64
    );
    // scoped to start here, after slice-carving: `PARALLEL_NODE_NANOS` below
    // is deliberately the thread::scope portion only (spawn + compute +
    // join), so it composes with `SERIAL_SLICE_CARVE_NANOS` above rather than
    // double-counting the carving loop inside both.
    #[cfg(feature = "instrument")]
    let node_start = Instant::now();

    thread::scope(|scope| {
        let handles: Vec<_> = chunks
            .iter()
            .zip(slices)
            .map(|(chunk, slice)| {
                scope.spawn(move || {
                    #[cfg(feature = "instrument")]
                    let chunk_start = Instant::now();
                    #[cfg(feature = "instrument")]
                    let cpu_start = instrument::thread_cpu_nanos();
                    let outcome = run_node_into(chunk, buffers, None, slice);
                    #[cfg(feature = "instrument")]
                    {
                        let chunk_nanos = chunk_start.elapsed().as_nanos() as u64;
                        let cpu_nanos = instrument::thread_cpu_nanos() - cpu_start;
                        instrument::record_chunk_nanos(chunk_nanos);
                        instrument::record_worker_busy_nanos(chunk_nanos);
                        instrument::record_worker_cpu_nanos(cpu_nanos);
                    }
                    outcome
                })
            })
            .collect();

        #[cfg(feature = "instrument")]
        let spawn_elapsed = node_start.elapsed();

        for handle in handles {
            // a worker panicking is a bug, not a new failure mode this
            // module introduces: resuming it on the joining thread matches
            // what would already happen if the same code path panicked
            // inside the sequential `evaluate`.
            handle
                .join()
                .unwrap_or_else(|panic_payload| panic::resume_unwind(panic_payload))?;
        }

        #[cfg(feature = "instrument")]
        {
            let total_elapsed = node_start.elapsed();
            let spawn_nanos = spawn_elapsed.as_nanos() as u64;
            let total_nanos = total_elapsed.as_nanos() as u64;
            counter!(instrument::PARALLEL_NODES, 1);
            counter!(instrument::PARALLEL_NODE_NANOS, total_nanos);
            counter!(instrument::PARALLEL_SPAWN_NANOS, spawn_nanos);
            // join/teardown is whatever wall-clock the node spent that
            // wasn't already charged to spawning the threads.
            counter!(
                instrument::PARALLEL_JOIN_NANOS,
                total_nanos.saturating_sub(spawn_nanos)
            );
        }

        Ok(())
    })
}

/// Pool-backed sibling of the `thread::scope` implementation above, gated
/// behind `tensor-bgpool` so the default build never depends on `prime`.
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
/// A worker panic cannot be resumed here the way the `thread::scope` sibling
/// does: `ProximaBackgroundPool`'s worker loop wraps every job in
/// `catch_unwind` and discards the payload (`prime/src/os/background.rs`,
/// `worker()`, `let _ = unwind;`), converting a panic into a dropped
/// closure with no way to recover the original payload. That drop takes our
/// own `sync_channel` sender clone with it, so a panicking chunk never
/// reports back; a chunk that never reports is surfaced as
/// `TensorError::ThreadedChunkFailed` instead.
#[cfg(feature = "tensor-bgpool")]
fn run_chunks_threaded<B: Deref<Target = [f32]> + Sync>(
    chunks: &[BoundOp],
    buffers: &[Option<B>],
    output: &mut [f32],
    workers: NonZeroUsize,
) -> Result<(), TensorError> {
    #[cfg(feature = "instrument")]
    let slice_carve_start = Instant::now();
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
        instrument::SERIAL_SLICE_CARVE_NANOS,
        slice_carve_start.elapsed().as_nanos() as u64
    );

    if chunks.len() < 2 {
        return match (chunks.first(), slices.into_iter().next()) {
            (Some(chunk), Some(slice)) => run_node_into(chunk, buffers, None, slice),
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
    let node_start = Instant::now();

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
    let spawn_elapsed = node_start.elapsed();

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
        let total_elapsed = node_start.elapsed();
        let spawn_nanos = spawn_elapsed.as_nanos() as u64;
        let total_nanos = total_elapsed.as_nanos() as u64;
        counter!(instrument::PARALLEL_NODES, 1);
        counter!(instrument::PARALLEL_NODE_NANOS, total_nanos);
        counter!(instrument::PARALLEL_SPAWN_NANOS, spawn_nanos);
        // join/teardown is whatever wall-clock the node spent that wasn't
        // already charged to spawning the pool tasks — includes the
        // caller's own claim_and_run loop, same as the thread::scope
        // sibling's join/teardown includes its own compute.
        counter!(
            instrument::PARALLEL_JOIN_NANOS,
            total_nanos.saturating_sub(spawn_nanos)
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
#[cfg(feature = "tensor-bgpool")]
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
        let chunk_start = Instant::now();
        #[cfg(feature = "instrument")]
        let cpu_start = instrument::thread_cpu_nanos();
        let outcome = run_node_into(chunk, chunk_buffers, None, chunk_output);
        #[cfg(feature = "instrument")]
        {
            let chunk_nanos = chunk_start.elapsed().as_nanos() as u64;
            let cpu_nanos = instrument::thread_cpu_nanos() - cpu_start;
            instrument::record_chunk_nanos(chunk_nanos);
            instrument::record_worker_busy_nanos(chunk_nanos);
            instrument::record_worker_cpu_nanos(cpu_nanos);
        }

        let _ = sender.send((index, outcome));
    }
}

/// The pool backing [`run_chunks_threaded`]'s chunk dispatch under
/// `tensor-bgpool`. Built once, on first use, and reused for every nest in
/// the process — a fresh `ProximaBackgroundPool` per node would reintroduce
/// the per-node OS-thread-spawn cost this feature exists to remove.
/// `OnceLock` only memoizes success: a failed build is not cached, so a
/// later call (after whatever exhausted OS thread resources clears up) can
/// retry instead of latching a permanent failure.
#[cfg(feature = "tensor-bgpool")]
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

#[cfg(feature = "tensor-bgpool")]
static NEST_POOL: OnceLock<Arc<ProximaBackgroundPool>> = OnceLock::new();

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
    for (position, expr) in program.iter().enumerate() {
        let node = NodeId(position as u32);
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
            Op::Input { .. } | Op::Iota { .. } => {}
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
            Op::Input { .. } | Op::Iota { .. } => {}
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

fn buffer_of<B: Deref<Target = [f32]>>(buffers: &[Option<B>], node: NodeId) -> Result<&[f32], TensorError> {
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
    run_node_into(resolved, buffers, None, &mut output)?;
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
    quantized_weights: Option<&BTreeMap<NodeId, &[u8]>>,
    output: &mut [f32],
) -> Result<(), TensorError> {
    match &resolved.kind {
        BoundOpKind::Elementwise { .. } => {
            #[cfg(feature = "instrument")]
            instrument::record_op_kind(instrument::OpKind::Elementwise);
            run_elementwise(resolved, buffers, output)
        }
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => {
            #[cfg(feature = "instrument")]
            instrument::record_op_kind(instrument::OpKind::Reduce);
            match quantized_weights {
                Some(quantized_weights) => {
                    run_reduce_with_quantized_weights(resolved, buffers, quantized_weights, output)
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
    }
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
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            output_axes,
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
                run_node_into(resolved, *buffers, None, &mut output)?;
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

/// Per-step gather state for one operand: an incrementally-advanced offset
/// into the `indices` buffer (mirroring how a normal operand's own running
/// offset advances by a precomputed stride each step), plus what to do with
/// a fetched value once read.
struct GatherCursor<'a> {
    buffer: &'a [f32],
    offset: i64,
    stride: i64,
    element_stride: i64,
    extent: u64,
}

impl GatherCursor<'_> {
    /// Reads the next index, advances the cursor, and returns the offset
    /// contribution that index adds to the operand's own running offset — a
    /// real error, not a clamp or a wraparound, when the fetched index falls
    /// outside the gathered dim's extent.
    fn fetch_and_advance(&mut self, node: NodeId) -> Result<i64, TensorError> {
        let raw = self.buffer[self.offset as usize];
        self.offset += self.stride;
        let index = raw as i64;
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

/// Recomputes each operand's running byte offset for a fresh coordinate,
/// writing into the caller's reused `running` buffer instead of collecting a
/// new `Vec` — the per-position counterpart of [`fill_gather_cursors`].
fn fill_running_offsets(resolved: &BoundOp, coordinate: &[u64], running: &mut [i64]) {
    for (slot, (_, view, _)) in running.iter_mut().zip(resolved.operands()) {
        *slot = view.offset_of(coordinate);
    }
}

fn run_elementwise<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    let (outer_extents, inner_len) = split_innermost(&resolved.extents);
    let innermost_dim = outer_extents.len() as u16;
    let raw = operand_buffers(resolved, buffers)?;
    let body = resolved.element_body();
    let shape = body_shape(body);
    let mut operand_values = vec![0.0f32; raw.len()];
    let mut step_values = vec![0.0f32; body.steps.len()];
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

    // Same fast-path gate `run_reduce` uses (ROW 3), reused verbatim here:
    // every operand the body shape reads is gather-free and affine with a
    // width-dim stride of 0 or 1 (`proxima-tensor/docs/discipline.md` ROW 5).
    let fast_path = body_shape_is_affine_fast_path(resolved, &shape, &strides);
    #[cfg(feature = "instrument")]
    let mut counters = KernelCounters::default();
    #[cfg(feature = "instrument")]
    let path = if fast_path { Path::WidthFast } else { Path::Generic };

    for outer_position in 0..odometer_len(outer_extents) as usize {
        unflatten_into(outer_position as u64, outer_extents, &mut outer_coordinate);
        fill_running_offsets(resolved, &outer_coordinate, &mut running);
        let out_base = outer_position * inner_len;
        #[cfg(feature = "instrument")]
        {
            counters.leading_iters += 1;
        }

        if fast_path {
            let out_slice = &mut output[out_base..out_base + inner_len];
            elementwise_width_fast(&shape, &raw, &running, &strides, out_slice);
            #[cfg(feature = "instrument")]
            {
                counters.kernel_calls += 1;
                counters.output_writes += inner_len as u64;
                for &stride in &strides {
                    counters.operand_loads += if stride == 1 { inner_len as u64 } else { 1 };
                }
            }
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
    }
    #[cfg(feature = "instrument")]
    {
        let distinct_operand_elements: u64 = raw.iter().map(|buffer| buffer.len() as u64).sum();
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
fn quantized_operand(resolved: &BoundOp, quantized_weights: &BTreeMap<NodeId, &[u8]>) -> Option<NodeId> {
    if !matches!(resolved.kind, BoundOpKind::Reduce { keep: Keep::Reduce, .. }) {
        return None;
    }
    resolved.operands().iter().map(|(node, _, _)| *node).find(|node| quantized_weights.contains_key(node))
}

/// [`run_reduce`]'s quantized-weight branch: `resolved` is the fused
/// `Reduce(Elementwise(Multiply))` matmul shape, `weight_node` one of its two
/// operands, packed `Q4_K` bytes rather than a bound `f32` buffer. The other
/// operand is the plain `f32` activation, already sitting in `buffers` like
/// any other node — read straight out of the same table [`run_reduce`]'s f32
/// path uses, no second buffer convention for it. Delegates the actual
/// per-row dot product to [`matmul_q4k_f32`], the dedicated, separately
/// parity-tested kernel; this function's whole job is locating that kernel's
/// two arguments inside a resolved `BoundOp`.
fn run_reduce_quantized<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    weights: &[u8],
    weight_node: NodeId,
    output: &mut [f32],
) -> Result<(), TensorError> {
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
    let rows = output.len();
    let result = matmul_q4k_f32(weights, rows, activation)?;
    output.copy_from_slice(&result);
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
    quantized_weights: &BTreeMap<NodeId, &[u8]>,
    output: &mut [f32],
) -> Result<(), TensorError> {
    if let Some(weight_node) = quantized_operand(resolved, quantized_weights) {
        let weights = quantized_weights.get(&weight_node).copied().ok_or(TensorError::NotLowerable {
            node: weight_node,
            reason: "quantized weight node has no bound byte buffer",
        })?;
        return run_reduce_quantized(resolved, buffers, weights, weight_node, output);
    }
    run_reduce(resolved, buffers, output)
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

    // Resolved ONCE per bound op, never per element: whether every physical
    // operand the body shape actually reads is gather-free and affine with
    // a width-dim stride of 0 (broadcast) or 1 (contiguous). When it holds,
    // the width loop below skips `gather_cursors`'s per-element `Option`
    // check and `operand_values`'s per-element copy entirely, reading
    // straight-line out of `raw`'s own contiguous subslices instead
    // (`proxima-tensor/docs/discipline.md` ROW 3). `Generic` bodies and any
    // non-unit stride or gathered operand fall back to the loop unchanged.
    let fast_path = body_shape_is_affine_fast_path(resolved, &shape, &strides);

    // A matmul with a transposed right-hand operand (ggml's own `mul_mat`
    // layout) has a bad width-dim stride on one operand (`fast_path` above
    // is false) but a GOOD stride on the contraction dim `k` — both
    // operands read `k` contiguously. `reduction_strides` is `strides`'s
    // sibling table for the single contraction dim, computed once per bound
    // op the same way; `body_shape_is_affine_fast_path` is reused verbatim,
    // just handed a different dim's stride table (`proxima-tensor/docs/discipline.md`
    // ROW 10). Scoped to exactly one contraction dim — a multi-dim
    // contraction falls back to the generic loop below unchanged.
    let reduction_strides: Vec<i64> = if reduction_dims.len() == 1 {
        let dim = reduction_dims[0];
        resolved.operands().iter().map(|(_, view, _)| view.stride(dim)).collect()
    } else {
        Vec::new()
    };
    let reduction_fast_path =
        !fast_path && reduction_dims.len() == 1 && body_shape_is_affine_fast_path(resolved, &shape, &reduction_strides);

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
                            counters.operand_loads += if operand_stride == 1 { reduction_total } else { 1 };
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
                                counters.operand_loads += if stride == 1 { width as u64 } else { 1 };
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
    Generic(&'a ComposedBody),
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
        BodyShape::Generic(body) => apply_body(body, operand_values, step_values),
    }
}

/// True when every physical operand [`BodyShape`] actually reads (one for
/// `Unary`, up to two for `Binary` — `Generic` never qualifies) is both
/// gather-free and affine with a width-dim stride of 0 (broadcast) or 1
/// (contiguous). Checked once per bound op, never per element — the same
/// discipline [`body_shape`] already applies to the op/arity decision, now
/// extended to the gather-vs-affine and stride-shape decision
/// [`reduce_width_fast`]'s straight-line arms depend on.
fn body_shape_is_affine_fast_path(resolved: &BoundOp, shape: &BodyShape, strides: &[i64]) -> bool {
    let operand_qualifies = |index: u16| {
        let (_, _, gather) = &resolved.operands()[index as usize];
        gather.is_none() && matches!(strides[index as usize], 0 | 1)
    };
    match *shape {
        BodyShape::Unary(_, a) => operand_qualifies(a),
        BodyShape::Binary(_, a, b) => operand_qualifies(a) && operand_qualifies(b),
        BodyShape::Generic(_) => false,
    }
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
/// straight-line arms: `contiguous` means `stride == 1` (read
/// `data[base..base+width]` as a real subslice), otherwise `stride == 0`
/// (read `data[base]` once and broadcast) — [`body_shape_is_affine_fast_path`]
/// already proved no other stride reaches here. Bundling the three fields
/// keeps `reduce_width_binary` under clippy's argument-count lint without
/// reaching for `#[allow]`.
struct OperandSpan<'a> {
    data: &'a [f32],
    base: usize,
    contiguous: bool,
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
            contiguous: strides[index] == 1,
        }
    };
    match *shape {
        BodyShape::Unary(op, a) => {
            reduce_width_unary(op, reduce_op, span_of(a), accumulator, seeded);
        }
        BodyShape::Binary(op, a, b) => {
            reduce_width_binary(op, reduce_op, span_of(a), span_of(b), accumulator, seeded);
        }
        BodyShape::Generic(_) => unreachable!("fast path is never entered for a Generic body shape"),
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
/// leaving a single concrete arithmetic operation per element.
#[inline(always)]
fn reduce_width_unary_monomorphic<F, R>(op: F, reduce: R, span: OperandSpan, accumulator: &mut [f32], seeded: bool)
where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if span.contiguous {
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

/// The pre-ROW-4 (ROW 3) implementation, kept as the fallback for a
/// `reduce_op` outside {Add, Multiply, Maximum, Minimum} — same numerical
/// result as [`reduce_width_unary_monomorphic`], dispatched per element via
/// [`apply_scalar_op`]/[`combine_reduction`] instead of an inlined closure.
fn reduce_width_unary_scalar_dispatch(
    op: ScalarOp,
    reduce_op: ScalarOp,
    span: OperandSpan,
    accumulator: &mut [f32],
    seeded: bool,
) {
    if span.contiguous {
        let slice = &span.data[span.base..span.base + accumulator.len()];
        for (slot, &raw_value) in accumulator.iter_mut().zip(slice) {
            let value = apply_scalar_op(op, &[raw_value]);
            *slot = combine_reduction(reduce_op, *slot, value, seeded);
        }
    } else {
        let raw_value = span.data[span.base];
        let value = apply_scalar_op(op, &[raw_value]);
        for slot in accumulator.iter_mut() {
            *slot = combine_reduction(reduce_op, *slot, value, seeded);
        }
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
    // the single densest multiply-accumulate in the crate.
    if FUSED_MULTIPLY_ADD
        && seeded
        && matches!((op, reduce_op), (ScalarOp::Multiply, ScalarOp::Add))
    {
        let width = accumulator.len();
        match (a.contiguous, b.contiguous) {
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
    let width = accumulator.len();
    match (a.contiguous, b.contiguous) {
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

/// The pre-ROW-4 (ROW 3) implementation, kept as the fallback for a
/// `reduce_op` outside {Add, Multiply, Maximum, Minimum}.
fn reduce_width_binary_scalar_dispatch(
    op: ScalarOp,
    reduce_op: ScalarOp,
    a: OperandSpan,
    b: OperandSpan,
    accumulator: &mut [f32],
    seeded: bool,
) {
    let width = accumulator.len();
    match (a.contiguous, b.contiguous) {
        (true, true) => {
            let slice_a = &a.data[a.base..a.base + width];
            let slice_b = &b.data[b.base..b.base + width];
            for ((slot, &value_a), &value_b) in accumulator.iter_mut().zip(slice_a).zip(slice_b) {
                let value = apply_scalar_op(op, &[value_a, value_b]);
                *slot = combine_reduction(reduce_op, *slot, value, seeded);
            }
        }
        (true, false) => {
            let slice_a = &a.data[a.base..a.base + width];
            let value_b = b.data[b.base];
            for (slot, &value_a) in accumulator.iter_mut().zip(slice_a) {
                let value = apply_scalar_op(op, &[value_a, value_b]);
                *slot = combine_reduction(reduce_op, *slot, value, seeded);
            }
        }
        (false, true) => {
            let value_a = a.data[a.base];
            let slice_b = &b.data[b.base..b.base + width];
            for (slot, &value_b) in accumulator.iter_mut().zip(slice_b) {
                let value = apply_scalar_op(op, &[value_a, value_b]);
                *slot = combine_reduction(reduce_op, *slot, value, seeded);
            }
        }
        (false, false) => {
            let value_a = a.data[a.base];
            let value_b = b.data[b.base];
            let value = apply_scalar_op(op, &[value_a, value_b]);
            for slot in accumulator.iter_mut() {
                *slot = combine_reduction(reduce_op, *slot, value, seeded);
            }
        }
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

/// Snapshot of the three [`WIDTH_TILE_GATE_PASSES`]-family counters:
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
    let mut chunks_a = slice_a.chunks_exact(DOT_LANES);
    let mut chunks_b = slice_b.chunks_exact(DOT_LANES);
    let mut lanes = [0.0f32; DOT_LANES];
    for (chunk_a, chunk_b) in (&mut chunks_a).zip(&mut chunks_b) {
        for ((lane, &value_a), &value_b) in lanes.iter_mut().zip(chunk_a).zip(chunk_b) {
            *lane = value_a.mul_add(value_b, *lane);
        }
    }
    let remainder_a = chunks_a.remainder();
    let remainder_b = chunks_b.remainder();
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

/// Snapshot of the three [`NEON_TILE_GATE_PASSES`]-family counters for the
/// main 6x4 tile: (gate passes, tile invocations, fallback elements).
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub fn neon_tile_counters() -> (u64, u64, u64) {
    (
        NEON_TILE_GATE_PASSES.load(Ordering::Relaxed),
        NEON_TILE_INVOCATIONS.load(Ordering::Relaxed),
        NEON_TILE_FALLBACK_ELEMENTS.load(Ordering::Relaxed),
    )
}

/// [`NEON_TILE_ROW_REMAINDER_INVOCATIONS`] snapshot — the row-remainder
/// tiles' own invocation count (any width `1..=5`), separate from the main
/// 6x4 tile's.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub fn neon_tile_row_remainder_invocations() -> u64 {
    NEON_TILE_ROW_REMAINDER_INVOCATIONS.load(Ordering::Relaxed)
}

/// [`NEON_TILE_ROW_REMAINDER_ELEMENTS`] snapshot — output elements covered
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

/// Packed bytes per `Q4_K` super-block — re-exported at this crate's own
/// name rather than spelling `proxima_gguf::quant::q4_k::BLOCK_BYTES` at
/// every call site below.
const Q4K_BLOCK_BYTES: usize = proxima_gguf::quant::q4_k::BLOCK_BYTES;

/// Decoded `f32` elements per `Q4_K` super-block (`QK_K` in ggml/gguf
/// terms).
const Q4K_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q4_k::QK_K;

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
        .chunks_exact(Q4K_BLOCK_BYTES)
        .zip(activation.chunks_exact(Q4K_BLOCK_ELEMENTS))
    {
        proxima_gguf::quant::q4_k::dequantize_block(block, &mut scratch);
        for (&weight, &value) in scratch.iter().zip(activation_chunk) {
            acc = weight.mul_add(value, acc);
        }
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
    if rows == 0 {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "matmul_q4k_f32 called with zero rows",
        });
    }
    if !weights.len().is_multiple_of(rows) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight byte length is not a whole multiple of the row count",
        });
    }
    let row_bytes = weights.len() / rows;
    weights
        .chunks_exact(row_bytes)
        .map(|weight_row| dot_q4k_f32(weight_row, activation))
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
    let mut chunks_a = slice_a.chunks_exact(DOT_LANES);
    let mut chunks_b = slice_b.chunks_exact(DOT_LANES);
    let mut lanes = [fold.init; DOT_LANES];
    let mut seeded = fold.seeded;
    for (chunk_a, chunk_b) in (&mut chunks_a).zip(&mut chunks_b) {
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
    let remainder_a = chunks_a.remainder();
    let remainder_b = chunks_b.remainder();
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
    let mut chunks = slice.chunks_exact(DOT_LANES);
    let mut lanes = [fold.init; DOT_LANES];
    let mut seeded = fold.seeded;
    for chunk in &mut chunks {
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
    let remainder = chunks.remainder();
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
            contiguous: reduction_strides[index] == 1,
        }
    };
    match *shape {
        BodyShape::Unary(op, a) => reduce_dot_unary(op, reduce_op, span_of(a), fold),
        BodyShape::Binary(op, a, b) => reduce_dot_binary(op, reduce_op, span_of(a), span_of(b), fold),
        BodyShape::Generic(_) => unreachable!("fast path is never entered for a Generic body shape"),
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
/// inlined non-capturing closures.
#[inline(always)]
fn reduce_dot_unary_monomorphic<F, R>(op: F, reduce: R, span: OperandSpan, fold: DotFold) -> f32
where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if span.contiguous {
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

/// The unaccelerated fallback for a `reduce_op` outside {Add, Multiply,
/// Maximum, Minimum} — same numerical result as
/// [`reduce_dot_unary_monomorphic`], dispatched per term via
/// [`apply_scalar_op`]/[`combine_reduction`].
fn reduce_dot_unary_scalar_dispatch(op: ScalarOp, reduce_op: ScalarOp, span: OperandSpan, fold: DotFold) -> f32 {
    let mut acc = fold.init;
    let mut seeded = fold.seeded;
    for step in 0..fold.len {
        let raw_value = if span.contiguous { span.data[span.base + step] } else { span.data[span.base] };
        let value = apply_scalar_op(op, &[raw_value]);
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
    // asked for by name (see `dot_fold_fused_multiply_add`).
    if FUSED_MULTIPLY_ADD
        && fold.seeded
        && fold.len >= DOT_LANES
        && a.contiguous
        && b.contiguous
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
/// product takes (`proxima-tensor/docs/discipline.md` ROW 10/11/12).
#[inline(always)]
fn reduce_dot_binary_monomorphic<F, R>(op: F, reduce: R, a: OperandSpan, b: OperandSpan, fold: DotFold) -> f32
where
    F: Fn(f32, f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    match (a.contiguous, b.contiguous) {
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

/// The unaccelerated fallback for a `reduce_op` outside {Add, Multiply,
/// Maximum, Minimum}.
fn reduce_dot_binary_scalar_dispatch(op: ScalarOp, reduce_op: ScalarOp, a: OperandSpan, b: OperandSpan, fold: DotFold) -> f32 {
    let mut acc = fold.init;
    let mut seeded = fold.seeded;
    for step in 0..fold.len {
        let value_a = if a.contiguous { a.data[a.base + step] } else { a.data[a.base] };
        let value_b = if b.contiguous { b.data[b.base + step] } else { b.data[b.base] };
        let value = apply_scalar_op(op, &[value_a, value_b]);
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
/// (`proxima-tensor/docs/discipline.md` ROW 5).
#[inline(always)]
fn elementwise_width_fast(shape: &BodyShape, raw: &[&[f32]], running: &[i64], strides: &[i64], out: &mut [f32]) {
    let span_of = |index: u16| {
        let index = index as usize;
        OperandSpan {
            data: raw[index],
            base: running[index] as usize,
            contiguous: strides[index] == 1,
        }
    };
    match *shape {
        BodyShape::Unary(op, a) => elementwise_width_unary(op, span_of(a), out),
        BodyShape::Binary(op, a, b) => elementwise_width_binary(op, span_of(a), span_of(b), out),
        BodyShape::Generic(_) => unreachable!("fast path is never entered for a Generic body shape"),
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
    if span.contiguous {
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
    let width = out.len();
    match (a.contiguous, b.contiguous) {
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
            contiguous: strides[index] == 1,
        }
    };
    match *shape {
        BodyShape::Unary(op, a) => scan_width_unary(op, reduce_op, span_of(a), out, seeded, accumulator),
        BodyShape::Binary(op, a, b) => {
            scan_width_binary(op, reduce_op, span_of(a), span_of(b), out, seeded, accumulator)
        }
        BodyShape::Generic(_) => unreachable!("fast path is never entered for a Generic body shape"),
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
        // position 0 reads `data[base]` whether the span is contiguous or
        // broadcast -- the two shapes only diverge starting at position 1.
        acc = op(span.data[span.base]);
        out[0] = acc;
        start = 1;
    }
    if span.contiguous {
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
        let raw_value = if span.contiguous {
            span.data[span.base + index]
        } else {
            span.data[span.base]
        };
        let value = apply_scalar_op(op, &[raw_value]);
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
    let read = |span: &OperandSpan, index: usize| {
        if span.contiguous {
            span.data[span.base + index]
        } else {
            span.data[span.base]
        }
    };
    if !seeded && width > 0 {
        acc = op(read(&a, 0), read(&b, 0));
        out[0] = acc;
        start = 1;
    }
    match (a.contiguous, b.contiguous) {
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
        let value_a = if a.contiguous { a.data[a.base + index] } else { a.data[a.base] };
        let value_b = if b.contiguous { b.data[b.base + index] } else { b.data[b.base] };
        let value = apply_scalar_op(op, &[value_a, value_b]);
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
}

/// One contiguous typed buffer, tagged by which native type backs it — the
/// storage half of [`evaluate_typed`]'s runtime dispatch. Every variant is a
/// plain `Vec<T>`: a whole buffer is tagged, never a scalar, which is what
/// keeps every operand a contiguous, SIMD-ready slice once a kernel is
/// written for it (see this module's typed-evaluator doc). `Bool`,
/// `BFloat16`, and `Float16` have no variant yet — `Bool`'s storage
/// convention (packed bits vs. one byte per element) is undecided, and the
/// two half-precision floats have no arithmetic on stable Rust; see
/// `typed_program_dtype` for the boundary this actually enforces today.
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
            Self::Float32(data) => data.len(),
            Self::Float64(data) => data.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Validates a program is executable by [`evaluate_typed`] and returns its
/// single uniform dtype.
///
/// One restriction narrows this evaluator below the full `Op` vocabulary,
/// because of what genuinely is not built yet rather than a semantic limit:
/// every node (bar a gather's `indices`, still carried as f32 — see
/// [`reject_non_float32`]'s doc) must share one dtype, since a mixed-dtype
/// fused body (quantized matmul's `i8 x i8 -> i32`) would need per-operand
/// element types this evaluator's single `T::apply` cannot express yet.
/// [`Op::Reduce`] (`Keep::Reduce` and `Keep::Scan` both) is supported at
/// every width this function accepts — see [`run_reduce_typed`]/
/// [`run_scan_typed`]. Gather is still rejected, in [`run_typed_program`]
/// rather than here, since it is a per-operand property of a bound node, not
/// a program-wide one. `Bool`/`BFloat16`/`Float16` are also out — see
/// [`TypedBuffer`]'s doc.
fn typed_program_dtype(program: &[Op]) -> Result<DType, TensorError> {
    let dtype = program.first().ok_or(TensorError::Empty)?.dtype();
    if matches!(dtype, DType::Bool | DType::BFloat16 | DType::Float16) {
        return Err(TensorError::NotLowerable {
            node: NodeId(0),
            reason: "the typed evaluator does not support Bool, BFloat16, or Float16 yet",
        });
    }
    for (position, expr) in program.iter().enumerate() {
        let node = NodeId(position as u32);
        if expr.dtype() != dtype {
            return Err(TensorError::NotLowerable {
                node,
                reason: "the typed evaluator requires one uniform dtype across the whole program",
            });
        }
    }
    Ok(dtype)
}

/// One requested output's node, shape, and data — [`evaluate_typed`]'s
/// per-dtype row, and [`run_typed_program`]'s own before it is wrapped into
/// a [`TypedBuffer`].
type TypedRow<Data> = (NodeId, Vec<u64>, Data);

/// Run an elementwise-or-reduce tensor program against a caller-chosen
/// non-f32 (or f64) dtype — the full-width counterpart of [`evaluate`] for
/// the programs `reject_non_float32` used to reject outright. See
/// `typed_program_dtype` for exactly which programs qualify. `DType::Float32`
/// dispatches `run_typed_program` the same as every other width, but that
/// function's own [`Op::Reduce`] handling specializes straight back to the
/// existing NEON `run_reduce`/`run_scan` for `T = f32` — see
/// `run_reduce_typed`'s doc.
pub fn evaluate_typed(
    program: &[Op],
    symbols: &[u64],
    blocks: &[TypedBuffer],
    outputs: &[NodeId],
) -> Result<Vec<TypedRow<TypedBuffer>>, TensorError> {
    let dtype = typed_program_dtype(program)?;
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
        DType::Float32 => dispatch!(f32, Float32),
        DType::Float64 => dispatch!(f64, Float64),
        DType::Bool | DType::BFloat16 | DType::Float16 => {
            unreachable!("typed_program_dtype already rejected this dtype")
        }
    })
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

    let mut buffers: Vec<Option<Cow<'_, [T]>>> = vec![None; program.len()];
    for (node, buffer) in block_nodes.iter().zip(blocks.iter()) {
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
    for node in &resolved {
        if node.operands().iter().any(|(_, _, lookup)| lookup.is_some()) {
            return Err(TensorError::NotLowerable {
                node: node.node,
                reason: "the typed evaluator does not support gather yet (indices stay f32-only)",
            });
        }
    }

    let retires = node_retirement(&resolved, &effective_outputs);
    let mut free_buffers: Vec<Vec<T>> = Vec::new();
    for (position, node) in resolved.iter().enumerate() {
        let mut output = typed_take_or_allocate(&mut free_buffers, node_output_len(node));
        match &node.kind {
            BoundOpKind::Elementwise { .. } => run_elementwise_typed(node, &buffers, &mut output)?,
            BoundOpKind::Reduce { keep: Keep::Reduce, .. } => {
                run_reduce_typed(node, &buffers, &mut output)?;
            }
            BoundOpKind::Reduce { keep: Keep::Scan, .. } => {
                run_scan_typed(node, &buffers, &mut output)?;
            }
            BoundOpKind::Iota => run_iota_typed(&mut output),
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

/// The typed counterpart of [`run_elementwise`]: same coordinate walk
/// (`fill_running_offsets`/`unflatten_into`/`split_innermost` are pure
/// geometry over `&[u64]`/[`bind::Layout`], with no f32 dependence, so they
/// are shared verbatim), but no width-tile SIMD fast path and no gather
/// cursor — see [`run_typed_program`]'s doc for why both are still only on
/// the f32 side.
fn run_elementwise_typed<T: Element>(
    resolved: &BoundOp,
    buffers: &[Option<Cow<'_, [T]>>],
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
    let mut outer_coordinate = vec![0u64; outer_extents.len()];

    for outer_position in 0..odometer_len(outer_extents) as usize {
        unflatten_into(outer_position as u64, outer_extents, &mut outer_coordinate);
        fill_running_offsets(resolved, &outer_coordinate, &mut running);
        let out_base = outer_position * inner_len;

        for step in 0..inner_len {
            for (index, data) in raw.iter().enumerate() {
                operand_values[index] = data[running[index] as usize];
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
/// it does not run a second reduction nest at all — it reinterprets the
/// typed evaluator's own `Vec<f32>` buffers as the `&[f32]` the existing
/// NEON-tiled [`run_reduce`] already takes (sound because [`Element`]'s
/// `'static` bound lets [`TypeId`] prove `T` really is `f32` first) and
/// calls that function directly, so the GEMM tiling, dot-fold, and
/// width-fast paths all still fire exactly as they do for [`evaluate`]. Only
/// every other width falls through to [`run_reduce_generic`], the one new
/// implementation this evaluator adds.
fn run_reduce_typed<T: Element>(
    resolved: &BoundOp,
    buffers: &[Option<Cow<'_, [T]>>],
    output: &mut [T],
) -> Result<(), TensorError> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
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
    run_reduce_generic(resolved, buffers, output)
}

/// The scalar reduction nest generic over every [`Element`] width — the
/// same (leading, reduction) coordinate walk [`run_reduce`]'s own generic
/// fallback runs (its NEON/width-tile/dot-fold fast paths stay f32-only, so
/// this has no equivalent of them to port), rewritten against
/// [`Element::apply`]/[`eval_body_typed`] instead of `apply_scalar_op`/
/// `eval_body_shape` so it type-checks for every width, and fallible where
/// [`Element::apply`] is (an unsupported op, or an integer division that has
/// no representable result). Gather is never reached here — the typed
/// evaluator already rejects any node with a gathered operand before this
/// runs, in [`run_typed_program`].
fn run_reduce_generic<T: Element>(
    resolved: &BoundOp,
    buffers: &[Option<Cow<'_, [T]>>],
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

            for slot in &mut accumulator {
                for (index, data) in raw.iter().enumerate() {
                    operand_values[index] = data[running[index] as usize];
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
/// scan counterpart of [`run_reduce_typed`], same `T = f32` specialization
/// down to the existing NEON-aware [`run_scan`].
fn run_scan_typed<T: Element>(
    resolved: &BoundOp,
    buffers: &[Option<Cow<'_, [T]>>],
    output: &mut [T],
) -> Result<(), TensorError> {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
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
    run_scan_generic(resolved, buffers, output)
}

/// The scalar scan nest generic over every [`Element`] width — [`run_scan`]'s
/// generic fallback (its width-fast SIMD path stays f32-only), rewritten
/// against [`Element::apply`]/[`eval_body_typed`] the same way
/// [`run_reduce_generic`] rewrites [`run_reduce`]'s.
fn run_scan_generic<T: Element>(
    resolved: &BoundOp,
    buffers: &[Option<Cow<'_, [T]>>],
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
    let mut outer_coordinate = vec![0u64; outer_extents.len()];

    let mut accumulator = T::reduce_seed(*init).unwrap_or_default();
    let mut seeded = !matches!(init, ReduceInit::FirstElement);

    for outer_flat in 0..odometer_len(outer_extents) {
        unflatten_into(outer_flat, outer_extents, &mut outer_coordinate);
        fill_running_offsets(resolved, &outer_coordinate, &mut running);
        let mut out_running = out_layout.offset_of(&outer_coordinate);
        let out_stride = out_layout.stride(innermost_dim);

        for _ in 0..inner_len {
            for (index, data) in raw.iter().enumerate() {
                operand_values[index] = data[running[index] as usize];
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
    use crate::map::{self, AxisTerm, IndexMap};
    use crate::op::{Extent, Reduce, append};
    use rstest::rstest;

    use crate::test_support::Lcg;

    fn random_vec(seed: u64, count: usize) -> Vec<f32> {
        let mut lcg = Lcg(seed);
        (0..count).map(|_| lcg.next_unit()).collect()
    }

    /// [`matmul_q4k_f32`] against the reference path (`proxima_gguf`'s own
    /// tested dequantize into a full `f32` weight matrix, then a naive f32
    /// dot product) — [`super`]'s guiding-principle 14: the incumbent
    /// (dequantize-then-matmul) is correct by construction, so this is a
    /// parity check, not a round-trip-to-self check. Two real (pseudo-random,
    /// non-degenerate — `Lcg`, not zeros/constants) rows x 2 super-blocks
    /// (512 elements) each, `Q4_K`'s minimum non-trivial multi-block shape.
    #[rstest]
    #[case::seed_1(1)]
    #[case::seed_7(7)]
    #[case::seed_1000(1000)]
    fn matmul_q4k_f32_matches_dequantize_then_f32_matmul(#[case] seed: u64) {
        use proxima_gguf::quant::q4_k::{QK_K, dequantize, quantize};

        const ROWS: usize = 2;
        const BLOCKS_PER_ROW: usize = 2;
        const K: usize = BLOCKS_PER_ROW * QK_K;

        let weights_f32 = random_vec(seed, ROWS * K);
        let activation = random_vec(seed.wrapping_add(1), K);

        let mut packed = vec![0u8; ROWS * BLOCKS_PER_ROW * Q4K_BLOCK_BYTES];
        for (row_f32, row_packed) in weights_f32.chunks_exact(K).zip(packed.chunks_exact_mut(BLOCKS_PER_ROW * Q4K_BLOCK_BYTES)) {
            quantize(row_f32, row_packed).expect("2 whole super-blocks quantize cleanly");
        }

        let mut dequantized_reference = vec![0.0f32; ROWS];
        let mut dequantized_row = vec![0.0f32; K];
        for (row_index, row_packed) in packed.chunks_exact(BLOCKS_PER_ROW * Q4K_BLOCK_BYTES).enumerate() {
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

    #[rstest]
    #[case::one_worker(1)]
    #[case::two_workers(2)]
    #[case::three_workers(3)]
    fn evaluate_parallel_matches_evaluate_for_a_gather_program(#[case] workers: usize) {
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

    #[test]
    fn a_scatter_data_dependent_fold_output_is_still_rejected_by_evaluate() {
        let mut program = Vec::new();
        let source = f32_block(&mut program, &[Extent::Static(4)]);
        let ids = block(&mut program, DType::Int32, &[Extent::Static(4)]);
        let out_map = IndexMap::Computed {
            indices: ids,
            index_map: map::projection(1, &[0]),
            base: map::IndexPattern {
                iter_rank: 1,
                axes: alloc::vec![map::AxisIndex::default()],
            },
            gathered_dim: 0,
        };
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map,
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let error = evaluate(&program, &[], &[], &[]).expect_err("scatter is not lowerable in v1");
        assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
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
            run_node_into(chunk, &buffers, None, this_chunk).expect("chunk runs");
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
            run_node_into(chunk, &buffers, None, this_chunk).expect("chunk runs");
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

    #[rstest]
    #[case::one_worker(1)]
    #[case::two_workers(2)]
    #[case::three_workers(3)]
    #[case::eight_workers(8)]
    fn evaluate_parallel_matches_evaluate_for_a_matmul(#[case] workers: usize) {
        let (m, k, n) = (4usize, 3usize, 5usize);
        let (program, _sum) = matmul_program(m as u32, k as u32, n as u32, false);
        let lhs: Vec<f32> = (0..m * k).map(|value| value as f32).collect();
        let rhs: Vec<f32> = (0..k * n).map(|value| value as f32).collect();

        assert_parallel_matches_sequential(&program, &[], &[&lhs, &rhs], &[], workers);
    }

    #[rstest]
    #[case::one_worker(1)]
    #[case::two_workers(2)]
    #[case::three_workers(3)]
    #[case::eight_workers(8)]
    fn evaluate_parallel_matches_evaluate_for_a_tanh_chain(#[case] workers: usize) {
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

    #[rstest]
    #[case::one_worker(1)]
    #[case::two_workers(2)]
    #[case::three_workers(3)]
    #[case::eight_workers(8)]
    fn evaluate_parallel_matches_evaluate_for_softmax(#[case] workers: usize) {
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

    #[rstest]
    #[case::one_worker(1)]
    #[case::two_workers(2)]
    #[case::three_workers(3)]
    #[case::eight_workers(8)]
    fn evaluate_parallel_matches_evaluate_for_cumsum(#[case] workers: usize) {
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

    #[rstest]
    #[case::one_worker(1)]
    #[case::two_workers(2)]
    #[case::three_workers(3)]
    #[case::eight_workers(8)]
    fn evaluate_parallel_matches_evaluate_for_multiple_requested_outputs(#[case] workers: usize) {
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

        let mut scatter_program = Vec::new();
        let source = f32_block(&mut scatter_program, &[Extent::Static(4)]);
        let ids = block(&mut scatter_program, DType::Int32, &[Extent::Static(4)]);
        let out_map = IndexMap::Computed {
            indices: ids,
            index_map: map::projection(1, &[0]),
            base: map::IndexPattern {
                iter_rank: 1,
                axes: alloc::vec![map::AxisIndex::default()],
            },
            gathered_dim: 0,
        };
        append(
            &mut scatter_program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map,
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let sequential_error =
            evaluate(&scatter_program, &[], &[], &[]).expect_err("scatter is not lowerable in v1");
        let parallel_error = evaluate_parallel(&scatter_program, &[], &[], &[], workers)
            .expect_err("scatter is not lowerable in v1");
        assert_eq!(sequential_error, parallel_error);
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

    #[rstest]
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
    fn evaluate_typed_adds_across_every_extended_width(
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

    #[rstest]
    #[case::int32(DType::Int32, TypedBuffer::Int32(alloc::vec![1, 2, 3, 4]), TypedBuffer::Int32(alloc::vec![10]))]
    #[case::uint64(DType::UInt64, TypedBuffer::UInt64(alloc::vec![1, 2, 3, 4]), TypedBuffer::UInt64(alloc::vec![10]))]
    #[case::float64(DType::Float64, TypedBuffer::Float64(alloc::vec![1.5, 2.5, 3.0, 4.0]), TypedBuffer::Float64(alloc::vec![11.0]))]
    fn evaluate_typed_reduces_a_vector_to_a_scalar_across_widths(
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
    /// [`quantized_operand`] -> `run_reduce_quantized` -> `matmul_q4k_f32`
    /// is reachable from the program-level entry point, not merely callable
    /// in isolation.
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
        eprintln!("evaluate_quantized vs dequantize-then-evaluate: max_diff={max_diff} rms_diff={rms_diff}");

        // same loose sanity bound as `matmul_q4k_f32_agrees_with_..`, which
        // this test's own inner kernel call bottoms out in — not tuned to
        // the measured numbers.
        assert!(max_diff < 0.05, "max_diff={max_diff} exceeds loose sanity bound");
        assert!(rms_diff < 0.02, "rms_diff={rms_diff} exceeds loose sanity bound");
    }
}
