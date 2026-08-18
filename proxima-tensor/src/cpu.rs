//! A CPU interpreter for [`BoundOp`] nodes: strided, f32-only, streaming its
//! buffers.
//!
//! This module owns none of the stride arithmetic — that lives in
//! [`bind`](crate::bind), shared with any other backend. What is
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
//! [`Interpreter`] is this module's [`Pipe`](proxima_primitives::pipe::Pipe)
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
//! [`run_node_into`] is the primitive `Interpreter::call` (and
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
    reject_non_float32(program)?;

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
        run_node_into(computed, &buffers, &mut output)?;
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
const PARALLEL_THRESHOLD: usize = 4096;

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
const OVERSUBSCRIBE: usize = 1;

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
const SPLIT_ALIGNMENT: u64 = 1;

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
            run_node_into(resolved, buffers, &mut output)?;
            #[cfg(feature = "instrument")]
            counter!(
                instrument::SERIAL_SEQUENTIAL_COMPUTE_NANOS,
                sequential_start.elapsed().as_nanos() as u64
            );
        }
    }
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
                    let outcome = run_node_into(chunk, buffers, slice);
                    #[cfg(feature = "instrument")]
                    {
                        let chunk_nanos = chunk_start.elapsed().as_nanos() as u64;
                        instrument::record_chunk_nanos(chunk_nanos);
                        instrument::record_worker_busy_nanos(chunk_nanos);
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
            (Some(chunk), Some(slice)) => run_node_into(chunk, buffers, slice),
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
        let outcome = run_node_into(chunk, chunk_buffers, chunk_output);
        #[cfg(feature = "instrument")]
        {
            let chunk_nanos = chunk_start.elapsed().as_nanos() as u64;
            instrument::record_chunk_nanos(chunk_nanos);
            instrument::record_worker_busy_nanos(chunk_nanos);
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
fn reject_non_float32(program: &[Op]) -> Result<(), TensorError> {
    let index_nodes = index_node_ids(program);
    for (position, expr) in program.iter().enumerate() {
        let node = NodeId(position as u32);
        if expr.dtype() != DType::Float32 && !index_nodes.contains(&node) {
            return Err(TensorError::NotLowerable {
                node,
                reason: "cpu evaluation is f32-only in v1, except for a gather's indices",
            });
        }
    }
    Ok(())
}

/// Every node referenced as a gather's `indices` anywhere in `program` —
/// the one class of non-float32 node [`reject_non_float32`] tolerates.
fn index_node_ids(program: &[Op]) -> BTreeSet<NodeId> {
    let mut nodes = BTreeSet::new();
    for expr in program {
        match expr {
            Op::Input { .. } => {}
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
/// [`merge_coordinates_into`]'s former per-call `Vec` (`scratchpad/opt/discipline.md` ROW 2b).
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
    run_node_into(resolved, buffers, &mut output)?;
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
            run_reduce(resolved, buffers, output)
        }
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => {
            #[cfg(feature = "instrument")]
            instrument::record_op_kind(instrument::OpKind::Scan);
            run_scan(resolved, buffers, output)
        }
    }
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
    /// same shape [`prepare`] already builds locally for [`evaluate`].
    /// `Interpreter` never allocates it, resizes it, or takes ownership of
    /// it. Generic over `B` (matching [`run_node_into`]'s bound) so the same
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
                run_node_into(resolved, *buffers, &mut output)?;
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
    /// [`Interpreter::fold`].
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
/// count outside the hot loop (`scratchpad/opt/discipline.md` ROW 2).
fn fill_gather_cursors<'a, B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &'a [Option<B>],
    coordinate: &[u64],
    stride_dim: Option<u16>,
    cursors: &mut [Option<GatherCursor<'a>>],
) -> Result<(), TensorError> {
    for (slot, (_, _, gather)) in cursors.iter_mut().zip(resolved.operands()) {
        *slot = gather
            .as_ref()
            .map(|gather_access| {
                let buffer = buffer_of(buffers, gather_access.indices)?;
                Ok(GatherCursor {
                    buffer,
                    offset: gather_access.index_layout.offset_of(coordinate),
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
    // outer position (`scratchpad/opt/discipline.md` ROW 2).
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
    // width-dim stride of 0 or 1 (`scratchpad/opt/discipline.md` ROW 5).
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
    // 1024^3 GEMM (`scratchpad/opt/discipline.md` ROW 2).
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
    // (`scratchpad/opt/discipline.md` ROW 3). `Generic` bodies and any
    // non-unit stride or gathered operand fall back to the loop unchanged.
    let fast_path = body_shape_is_affine_fast_path(resolved, &shape, &strides);

    // A matmul with a transposed right-hand operand (ggml's own `mul_mat`
    // layout) has a bad width-dim stride on one operand (`fast_path` above
    // is false) but a GOOD stride on the contraction dim `k` — both
    // operands read `k` contiguously. `reduction_strides` is `strides`'s
    // sibling table for the single contraction dim, computed once per bound
    // op the same way; `body_shape_is_affine_fast_path` is reused verbatim,
    // just handed a different dim's stride table (`scratchpad/opt/discipline.md`
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
    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    let width_tile_counters_before = width_tile_counters();
    if try_run_width_tile(&width_path_context, &raw, output) {
        // the tile's own early return skips the rest of this function
        // (including the `counters.commit` call every other path reaches),
        // so this is instrument's only chance to record the node — read
        // back the invocation/fallback deltas the tile itself already
        // counted instead of re-deriving them from `leading_total`/`width`.
        #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
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

    // Resolved ONCE per bound op: an explicit-NEON 6x4 microkernel for the
    // exact GEMM shape `reduction_fast_path` already isolates. Ported from
    // ggml tinyBLAS's `gemm_bloc` — see `neon_tile_plan` and
    // `gemm_tile_neon` docs for the six-condition gate and why the
    // accumulator type (not the loop shape) is what makes it fit in
    // registers (`scratchpad/opt/discipline.md`, attempts 1 and 2).
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
                            TileOperand {
                                data: raw[plan.index_a],
                                base: base_a,
                                stride: plan.row_stride_a,
                            },
                            TileOperand {
                                data: raw[plan.index_b],
                                base: base_b,
                                stride: plan.col_stride_b,
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
                            TileOperand {
                                data: raw[plan.index_a],
                                base: base_a,
                                stride: plan.row_stride_a,
                            },
                            TileOperand {
                                data: raw[plan.index_b],
                                base: base_b,
                                stride: plan.col_stride_b,
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
    // (`scratchpad/opt/discipline.md` ROW 2).
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
    // own width-dim stride to be 1 (`scratchpad/opt/discipline.md` ROW 5).
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
/// (`scratchpad/opt/discipline.md` ROW 0) found that per-element redecision,
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
/// so output is bit-identical (`scratchpad/opt/discipline.md` ROW 3).
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
/// (`scratchpad/opt/discipline.md` ROW 4). `seeded` is also resolved here,
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
const WIDTH_TILE_ROWS: usize = 4;

/// `float32x4_t` vectors of output columns one call to [`gemm_width_tile_neon`]
/// computes — 4 gives `WIDTH_TILE_ROWS * WIDTH_TILE_VECS` = 16 independent
/// accumulators, the measured saturation point for this core's NEON FMA
/// throughput.
#[cfg(target_arch = "aarch64")]
const WIDTH_TILE_VECS: usize = 4;

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
/// clippy's argument-count lint instead of reaching for `#[allow]`. Not
/// `cfg`-gated to aarch64: `run_reduce` calls [`try_run_width_tile`] on
/// every target, and the non-aarch64 stub still needs a type to accept.
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

/// The `a`-side (row-invariant-across-width) operand a tile call reads —
/// bundled with its `b`-side counterpart below for the same argument-count
/// reason [`OperandSpan`]/[`DotFold`] already document.
#[cfg(target_arch = "aarch64")]
struct TileOperandA<'a> {
    data: &'a [f32],
    base: i64,
    row_stride: i64,
    k_stride: i64,
}

/// The `b`-side (width-contiguous) operand a tile call reads.
#[cfg(target_arch = "aarch64")]
struct TileOperandB<'a> {
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
/// Caller guarantees every offset `a.base + i*a.row_stride + step*a.k_stride`
/// for `i in 0..WIDTH_TILE_ROWS, step in 0..k` lies within `a.data`, and
/// every offset `b.base + step*b.k_stride + v*4 + lane` for `v in
/// 0..WIDTH_TILE_VECS, lane in 0..4` lies within `b.data`.
#[cfg(target_arch = "aarch64")]
unsafe fn gemm_width_tile_neon(
    a: TileOperandA,
    b: TileOperandB,
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
                let offset = a.base + i as i64 * a.row_stride + step * a.k_stride;
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
fn width_tile_scalar_cell(a: TileOperandA, b: TileOperandB, k: usize, seed: f32) -> f32 {
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
                    TileOperandA { data: data_a, base: base_a, row_stride: plan.row_stride_a, k_stride: plan.k_stride_a },
                    TileOperandB { data: data_b, base: base_b, k_stride: plan.k_stride_b },
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
                    TileOperandA {
                        data: data_a,
                        base: plan.base_a + row as i64 * plan.row_stride_a,
                        row_stride: plan.row_stride_a,
                        k_stride: plan.k_stride_a,
                    },
                    TileOperandB { data: data_b, base: plan.base_b + col as i64, k_stride: plan.k_stride_b },
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
                TileOperandA {
                    data: data_a,
                    base: plan.base_a + row as i64 * plan.row_stride_a,
                    row_stride: plan.row_stride_a,
                    k_stride: plan.k_stride_a,
                },
                TileOperandB { data: data_b, base: plan.base_b + col as i64, k_stride: plan.k_stride_b },
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
/// unchanged. Non-aarch64 always reports `false` — the existing path is the
/// only one compiled there.
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

#[cfg(not(target_arch = "aarch64"))]
fn try_run_width_tile(_context: &WidthPathContext, _raw: &[&[f32]], _output: &mut [f32]) -> bool {
    false
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
/// changes its bit pattern (`scratchpad/opt/discipline.md` ROW 11).
/// Splitting the chain into `DOT_LANES` independent partial folds (one
/// per position in a `DOT_LANES`-wide `chunks_exact` block) breaks that
/// dependency: each lane's own chain is still strictly sequential (still
/// no per-lane reassociation), but the lanes run independently, so LLVM
/// can pack the common case into vector `fmul`/`fadd` and pay the
/// horizontal combine once per call instead of once per element —
/// exactly what every BLAS and ggml itself do. 4 and 8 were measured
/// head-to-head (ROW 12, `scratchpad/opt/discipline.md`): 8 measured
/// consistently faster (~0.337-0.349s vs ~0.352-0.354s, 1024^3
/// transposed-RHS GEMM, 5 runs each) — more independent lanes hide more
/// of the reduce's latency on this core's issue width. 8 was kept.
const DOT_LANES: usize = 8;

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
const TILE_ROWS: usize = 6;
#[cfg(target_arch = "aarch64")]
const TILE_COLS: usize = 4;

/// Bytes of L2 budgeted for a resident `b` column panel in the tiled GEMM
/// pass below. M1 Max: 12 MiB shared L2 per performance cluster. Budgeting
/// 8 MiB rather than the full 12 MiB leaves headroom for the row-strip's own
/// `a` tile, the output tile in flight, and set-associativity conflicts —
/// zero headroom is exactly the margin that turns a near-fit back into a
/// thrash.
#[cfg(target_arch = "aarch64")]
const NEON_COLUMN_PANEL_BUDGET_BYTES: usize = 8 * 1024 * 1024;

/// Column-panel width for the tiled GEMM pass: the widest multiple of
/// `TILE_COLS` whose panel of `b` (`panel_cols` columns, each a contiguous
/// run of `reduction_len` `f32`s along the contraction dim) fits inside
/// [`NEON_COLUMN_PANEL_BUDGET_BYTES`]. At `reduction_len = 2048` (2048^3's
/// `k`): `8 MiB / (2048 * 4 bytes) = 1024` columns, half of 2048's tiled
/// width — two panels. At `reduction_len = 1024` the budget already covers
/// 2048 columns, wider than any tiled width a 1024^3 or smaller call
/// produces, so the `clamp` below collapses the result to one panel
/// spanning `tiled_width_cols` — the panel loop becomes a no-op wrapper
/// around the original single sweep, unchanged behavior for 1024^3 and
/// smaller.
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
/// six conditions mirror attempt 2's (`scratchpad/opt/discipline.md`):
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

/// One operand's tile-relative addressing for [`gemm_tile_neon`]: `data` is
/// the physical buffer, `base` the flat offset of this tile's `(row 0, col
/// 0, k 0)` corner, `stride` the per-row (for `a`) or per-column (for `b`)
/// step between adjacent tile lanes. Bundled for the same reason
/// [`OperandSpan`] is — keeps the kernel under clippy's argument-count lint.
#[cfg(target_arch = "aarch64")]
struct TileOperand<'a> {
    data: &'a [f32],
    base: usize,
    stride: usize,
}

/// Ported from ggml tinyBLAS's `gemm_bloc`: `ROWS` x [`TILE_COLS`]
/// output accumulators declared as `float32x4_t`, a native NEON vector
/// register type, not an `[f32; 4]` array indexed by a loop variable
/// (`scratchpad/opt/discipline.md` — attempt 2 spilled 737 `str q`
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
unsafe fn gemm_tile_neon<const ROWS: usize>(a: TileOperand, b: TileOperand, k: usize, out: &mut [[f32; TILE_COLS]; ROWS]) {
    // `vdupq_n_f32` requires the `neon` target feature, unconditionally
    // present in the aarch64 base ISA this module is gated on.
    let mut acc = [[unsafe { vdupq_n_f32(0.0) }; TILE_COLS]; ROWS];
    let steps = k / 4;
    for step in 0..steps {
        let l = step * 4;
        let mut av = [unsafe { vdupq_n_f32(0.0) }; ROWS];
        for (row, lane) in av.iter_mut().enumerate() {
            // caller guarantees `a.base + row * a.stride + l + 4 <= a.data.len()`
            // via the reduction-dim contiguity and row-count checks in
            // `neon_tile_plan` and its `run_reduce` call site.
            *lane = unsafe { vld1q_f32(a.data.as_ptr().add(a.base + row * a.stride + l)) };
        }
        let mut bv = [unsafe { vdupq_n_f32(0.0) }; TILE_COLS];
        for (column, lane) in bv.iter_mut().enumerate() {
            // caller guarantees `b.base + column * b.stride + l + 4 <= b.data.len()`
            // by the same contiguity and column-count checks.
            *lane = unsafe { vld1q_f32(b.data.as_ptr().add(b.base + column * b.stride + l)) };
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
                total = a.data[a.base + row * a.stride + l].mul_add(b.data[b.base + column * b.stride + l], total);
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
/// 12, `scratchpad/opt/discipline.md`). Operates on matching-length
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
/// product takes (`scratchpad/opt/discipline.md` ROW 10/11/12).
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
/// (`scratchpad/opt/discipline.md` ROW 5).
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
        _ => unreachable!("BodyShape::Unary only ever carries an arity-1 ScalarOp"),
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
        _ => unreachable!("BodyShape::Binary only ever carries an arity-2 ScalarOp"),
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::map::{self, AxisTerm, IndexMap};
    use crate::op::{Extent, Reduce, append};
    use rstest::rstest;

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

    /// `reduce_dot_binary_monomorphic`'s `(true, true)` arm reassociates the
    /// sum (`DOT_LANES` independent partial accumulators, ROW 12,
    /// `scratchpad/opt/discipline.md`) — bit-exactness against
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
    /// (`scratchpad/opt/discipline.md` ROW 10/11): the width dim `n` is not
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
        // WEAKENED (ROW 12, `scratchpad/opt/discipline.md`): was
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
        // grouping) — see ROW 12, `scratchpad/opt/discipline.md`.
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
            run_node_into(chunk, &buffers, this_chunk).expect("chunk runs");
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
            run_node_into(chunk, &buffers, this_chunk).expect("chunk runs");
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
}
