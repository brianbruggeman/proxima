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
use core::cell::RefCell;
use core::future::Future;
use core::num::NonZeroUsize;
use std::panic;
use std::thread;

use proxima_primitives::pipe::Pipe;

use crate::bind::{self, BoundOp, BoundOpKind, ComposedBody, StepArg};
use crate::dtype::DType;
use crate::error::TensorError;
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
}

/// Everything [`evaluate`] and [`evaluate_parallel`] must agree on before
/// either one is free to choose how a single nest actually runs.
struct Prepared {
    root: NodeId,
    shapes: shape::Shapes,
    effective_outputs: Vec<NodeId>,
    buffers: Vec<Option<Vec<f32>>>,
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
fn prepare(
    program: &[Op],
    symbols: &[u64],
    blocks: &[&[f32]],
    outputs: &[NodeId],
) -> Result<Prepared, TensorError> {
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

    let mut buffers: Vec<Option<Vec<f32>>> = vec![None; program.len()];
    for (node, data) in block_nodes.iter().zip(blocks.iter()) {
        let expected = element_count(shapes.of(*node));
        if data.len() != expected {
            return Err(TensorError::InputSizeMismatch {
                node: *node,
                expected,
                found: data.len(),
            });
        }
        buffers[node.0 as usize] = Some((*data).to_vec());
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
fn finish(
    shapes: &shape::Shapes,
    effective_outputs: &[NodeId],
    buffers: &[Option<Vec<f32>>],
    root: NodeId,
    peak_live_buffers: usize,
) -> Evaluated {
    let results = effective_outputs
        .iter()
        .map(|node| {
            let shape = shapes.of(*node).to_vec();
            let data = buffers[node.0 as usize].clone().unwrap_or_default();
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
pub fn evaluate(
    program: &[Op],
    symbols: &[u64],
    blocks: &[&[f32]],
    outputs: &[NodeId],
) -> Result<Evaluated, TensorError> {
    let Prepared {
        root,
        shapes,
        effective_outputs,
        mut buffers,
        resolved,
        retires,
    } = prepare(program, symbols, blocks, outputs)?;

    let mut peak_live_buffers = live_count(&buffers);
    for (position, computed) in resolved.iter().enumerate() {
        let output = run_node(computed, &buffers)?;
        buffers[computed.node.0 as usize] = Some(output);
        peak_live_buffers = peak_live_buffers.max(live_count(&buffers));
        for retired in &retires[position] {
            buffers[retired.0 as usize] = None;
        }
    }

    Ok(finish(
        &shapes,
        &effective_outputs,
        &buffers,
        root,
        peak_live_buffers,
    ))
}

/// Below this many iteration-space elements, a nest runs the plain
/// sequential path even when `workers > 1`: `std::thread::scope`'s spawn
/// and join overhead outweighs the work for a small nest.
const PARALLEL_THRESHOLD: usize = 4096;

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
    let Prepared {
        root,
        shapes,
        effective_outputs,
        mut buffers,
        resolved,
        retires,
    } = prepare(program, symbols, blocks, outputs)?;

    let mut peak_live_buffers = live_count(&buffers);
    for (position, computed) in resolved.iter().enumerate() {
        let output = evaluate_node_parallel(computed, &buffers, workers)?;
        buffers[computed.node.0 as usize] = Some(output);
        peak_live_buffers = peak_live_buffers.max(live_count(&buffers));
        for retired in &retires[position] {
            buffers[retired.0 as usize] = None;
        }
    }

    Ok(finish(
        &shapes,
        &effective_outputs,
        &buffers,
        root,
        peak_live_buffers,
    ))
}

/// Runs one node, threaded across `workers` when [`BoundOp::split`] finds it
/// sound and it clears [`PARALLEL_THRESHOLD`]; otherwise the plain
/// sequential path via [`run_node_into`].
fn evaluate_node_parallel(
    resolved: &BoundOp,
    buffers: &[Option<Vec<f32>>],
    workers: NonZeroUsize,
) -> Result<Vec<f32>, TensorError> {
    let mut output = vec![0.0f32; node_output_len(resolved)];

    let chunks = (element_count(&resolved.extents) >= PARALLEL_THRESHOLD)
        .then(|| resolved.split(workers.get()))
        .flatten();

    match chunks {
        Some(chunks) => run_chunks_threaded(&chunks, buffers, &mut output)?,
        None => run_node_into(resolved, buffers, &mut output)?,
    }
    Ok(output)
}

/// Runs each of `chunks` on its own OS thread, writing into the matching
/// disjoint sub-slice of `output` — sound because [`BoundOp::split`] documents
/// (and this function relies on) chunk `k`'s output occupying a contiguous,
/// non-overlapping range of the parent buffer.
fn run_chunks_threaded(
    chunks: &[BoundOp],
    buffers: &[Option<Vec<f32>>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    let mut slices = Vec::with_capacity(chunks.len());
    let mut remaining = output;
    for chunk in chunks {
        let (this_chunk, rest) = remaining.split_at_mut(node_output_len(chunk));
        slices.push(this_chunk);
        remaining = rest;
    }

    thread::scope(|scope| {
        let handles: Vec<_> = chunks
            .iter()
            .zip(slices)
            .map(|(chunk, slice)| scope.spawn(move || run_node_into(chunk, buffers, slice)))
            .collect();
        for handle in handles {
            // a worker panicking is a bug, not a new failure mode this
            // module introduces: resuming it on the joining thread matches
            // what would already happen if the same code path panicked
            // inside the sequential `evaluate`.
            handle
                .join()
                .unwrap_or_else(|panic_payload| panic::resume_unwind(panic_payload))?;
        }
        Ok(())
    })
}

fn live_count(buffers: &[Option<Vec<f32>>]) -> usize {
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
// `indices` — is f32: this interpreter's buffers are `Vec<f32>` throughout,
// indices included (an index value is an exact integer carried as f32, per
// the module doc), so a gather's `indices` node is the one deliberate
// exception to the f32 rule rather than a second buffer kind.
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

fn buffer_of(buffers: &[Option<Vec<f32>>], node: NodeId) -> Result<&[f32], TensorError> {
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
fn run_node_into(
    resolved: &BoundOp,
    buffers: &[Option<Vec<f32>>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    match &resolved.kind {
        BoundOpKind::Elementwise { .. } => run_elementwise(resolved, buffers, output),
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => run_reduce(resolved, buffers, output),
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => run_scan(resolved, buffers, output),
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
            let (leading_output_axes, last_output_dim) = output_axes_split(output_axes);
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
pub struct Interpreter<'buffers> {
    buffers: RefCell<&'buffers mut [Option<Vec<f32>>]>,
}

impl<'buffers> Interpreter<'buffers> {
    /// `buffers` is caller-owned scratch, one slot per program node — the
    /// same shape [`prepare`] already builds locally for [`evaluate`].
    /// `Interpreter` never allocates it, resizes it, or takes ownership of
    /// it.
    #[must_use]
    pub fn new(buffers: &'buffers mut [Option<Vec<f32>>]) -> Self {
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
        self.buffers.borrow()[node.0 as usize].clone()
    }
}

impl Pipe for Interpreter<'_> {
    type In = Vec<BoundOp>;
    type Out = ();
    type Err = TensorError;

    /// Folds a batch of ready nodes into the buffer table, in order — the
    /// same fold the buffer table already does one write at a time, just
    /// driven for every element of `ready` inside one call instead of one
    /// call per element. An empty `ready` is a no-op, not a special case.
    fn call(&self, ready: Vec<BoundOp>) -> impl Future<Output = Result<(), TensorError>> {
        async move {
            for resolved in ready {
                let mut output = vec![0.0f32; node_output_len(&resolved)];
                {
                    let buffers = self.buffers.borrow();
                    run_node_into(&resolved, *buffers, &mut output)?;
                }
                let mut buffers = self.buffers.borrow_mut();
                (*buffers)[resolved.node.0 as usize] = Some(output);
            }
            Ok(())
        }
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

fn operand_buffers<'a>(
    resolved: &BoundOp,
    buffers: &'a [Option<Vec<f32>>],
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
fn fill_gather_cursors<'a>(
    resolved: &BoundOp,
    buffers: &'a [Option<Vec<f32>>],
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

fn run_elementwise(
    resolved: &BoundOp,
    buffers: &[Option<Vec<f32>>],
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

    for outer_position in 0..odometer_len(outer_extents) as usize {
        unflatten_into(outer_position as u64, outer_extents, &mut outer_coordinate);
        fill_running_offsets(resolved, &outer_coordinate, &mut running);
        let out_base = outer_position * inner_len;

        if fast_path {
            let out_slice = &mut output[out_base..out_base + inner_len];
            elementwise_width_fast(&shape, &raw, &running, &strides, out_slice);
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
        }
    }
    Ok(())
}

fn run_reduce(
    resolved: &BoundOp,
    buffers: &[Option<Vec<f32>>],
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
        .filter(|dim| !output_axes.contains(dim))
        .collect();
    let (leading_output_axes, last_output_dim) = output_axes_split(output_axes);

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

    for leading_flat in 0..odometer_len(&leading_extents) {
        unflatten_into(leading_flat, &leading_extents, &mut leading_coordinate);
        let mut accumulator = vec![initial_value(*init).unwrap_or(0.0); width];
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

            if fast_path {
                reduce_width_fast(&shape, *reduce_op, &raw, &running, &strides, &mut accumulator, seeded);
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

fn run_scan(
    resolved: &BoundOp,
    buffers: &[Option<Vec<f32>>],
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
                        terms: alloc::vec![AxisTerm::projection(1)],
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
                        terms: alloc::vec![AxisTerm::projection(2)],
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
