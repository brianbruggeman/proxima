//! A CPU interpreter for [`Nest`]s: strided, f32-only, streaming its buffers.
//!
//! This module owns none of the stride arithmetic — that lives in
//! [`nest`](crate::nest), shared with any other backend. What is CPU-specific
//! and lives here: the f32-only restriction (a v1 limitation — [`ScalarOp`]'s
//! transcendental bodies need `libm`-grade math this crate does not depend
//! on, so a GPU backend targeting f16/bf16 natively is unaffected by this
//! choice), the loop nests that walk a `Nest`'s iteration space, and buffer
//! lifetime.
//!
//! `reject_non_float32`'s one exception is a gather's `indices` buffer: it
//! carries integer index values, but as f32 like everything else here, since
//! no separate integer-buffer kind exists yet. f32 represents every integer
//! up to `2^24` (16,777,216) exactly, so [`shape::infer`] rejects any
//! gathered dim wider than that before this module ever sees the program —
//! see [`crate::map::IndexMap::Computed`]'s docs for the full accounting.
//!
//! The inner loop of every walk below is a straight loop with a per-operand
//! running offset incremented by a precomputed stride each step — never a
//! per-element recomputation of the full coordinate — so the shape an
//! optimizing compiler needs to autovectorize is actually on the page.
//! [`nest::Nest`] documents the one fusion decided ahead of execution
//! (`Fold(Zip)` skipping the zip's O(iteration space) intermediate); this
//! module additionally drops each node's buffer the moment nothing in the
//! emitted nest sequence reads it again, which is the other half of not
//! paying for what a program does not keep — see
//! [`Evaluated::peak_live_buffers`].

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use std::panic;
use std::thread;

use crate::dtype::DType;
use crate::error::TensorError;
use crate::expr::{Expr, FoldInit, Keep, NodeId, ScalarOp};
use crate::map::IndexMap;
use crate::nest::{self, Nest, Reduction};
use crate::shape;

/// The result of [`evaluate`]: every requested output's data and shape, plus
/// the peak number of live intermediate buffers the run held at once.
#[derive(Debug)]
pub struct Evaluated {
    root: NodeId,
    results: Vec<(NodeId, Vec<u64>, Vec<f32>)>,
    peak_live_buffers: usize,
}

impl Evaluated {
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

    /// The most buffers ([`Expr::Block`] inputs and computed intermediates)
    /// held live at any one point during the run. The one number that proves
    /// streaming buffer lifetime is doing something: a long unary chain
    /// should hold a small constant, not one buffer per expression.
    #[must_use]
    pub const fn peak_live_buffers(&self) -> usize {
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
    nests: Vec<Nest>,
    retires: Vec<Vec<NodeId>>,
}

/// The preamble [`evaluate`] and [`evaluate_parallel`] share: shape
/// inference, output resolution, block binding, lowering, and per-nest
/// buffer retirement. Neither evaluator's own body decides any of this —
/// the two diverge only in how one already-lowered [`Nest`] gets executed.
///
/// Lowering (and the fusion it decides) runs over the whole program at
/// once — see `nest::lower`'s docs. Buffer retirement below is a separate,
/// finer-grained liveness question over the *emitted* nest sequence: a
/// held zip's own consumption is deferred to whenever its consumer
/// materializes it, which can be well after the expression position
/// `live::annotate` reasons about, so retirement here is computed fresh
/// over `nests`, not reused from the fusion pass's liveness.
fn prepare(
    program: &[Expr],
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
        return Err(TensorError::BlockCountMismatch {
            expected: block_nodes.len(),
            found: blocks.len(),
        });
    }

    let mut buffers: Vec<Option<Vec<f32>>> = vec![None; program.len()];
    for (node, data) in block_nodes.iter().zip(blocks.iter()) {
        let expected = element_count(shapes.of(*node));
        if data.len() != expected {
            return Err(TensorError::BlockSizeMismatch {
                node: *node,
                expected,
                found: data.len(),
            });
        }
        buffers[node.0 as usize] = Some((*data).to_vec());
    }

    let nests = nest::lower(program, &shapes, &effective_outputs)?;
    let retires = nest_retirement(&nests, &effective_outputs);

    Ok(Prepared {
        root,
        shapes,
        effective_outputs,
        buffers,
        nests,
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

    Evaluated {
        root,
        results,
        peak_live_buffers,
    }
}

/// Run a tensor program to f32 data.
///
/// `blocks` binds [`Expr::Block`] inputs positionally, in the order they
/// appear in `program` — the local, single-partition case; a distributed
/// evaluator would instead resolve blocks by
/// [`name`](Expr::name). `outputs` selects which nodes to return data for;
/// an empty slice means the root (the program's last expression) only.
pub fn evaluate(
    program: &[Expr],
    symbols: &[u64],
    blocks: &[&[f32]],
    outputs: &[NodeId],
) -> Result<Evaluated, TensorError> {
    let Prepared {
        root,
        shapes,
        effective_outputs,
        mut buffers,
        nests,
        retires,
    } = prepare(program, symbols, blocks, outputs)?;

    let mut peak_live_buffers = live_count(&buffers);
    for (position, computed) in nests.iter().enumerate() {
        let output = run_nest(computed, &buffers)?;
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
/// [`Nest::split`]). The preamble is [`prepare`], the same one [`evaluate`]
/// runs — the two functions diverge only in the loop below.
pub fn evaluate_parallel(
    program: &[Expr],
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
        nests,
        retires,
    } = prepare(program, symbols, blocks, outputs)?;

    let mut peak_live_buffers = live_count(&buffers);
    for (position, computed) in nests.iter().enumerate() {
        let output = evaluate_nest_parallel(computed, &buffers, workers)?;
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

/// Runs one nest, threaded across `workers` when [`Nest::split`] finds it
/// sound and it clears [`PARALLEL_THRESHOLD`]; otherwise the plain
/// sequential path via [`run_nest_into`].
fn evaluate_nest_parallel(
    nest: &Nest,
    buffers: &[Option<Vec<f32>>],
    workers: NonZeroUsize,
) -> Result<Vec<f32>, TensorError> {
    let mut output = vec![0.0f32; nest_output_len(nest)];

    let chunks = (element_count(&nest.extents) >= PARALLEL_THRESHOLD)
        .then(|| nest.split(workers.get()))
        .flatten();

    match chunks {
        Some(chunks) => run_chunks_threaded(&chunks, buffers, &mut output)?,
        None => run_nest_into(nest, buffers, &mut output)?,
    }
    Ok(output)
}

/// Runs each of `chunks` on its own OS thread, writing into the matching
/// disjoint sub-slice of `output` — sound because [`Nest::split`] documents
/// (and this function relies on) chunk `k`'s output occupying a contiguous,
/// non-overlapping range of the parent buffer.
fn run_chunks_threaded(
    chunks: &[Nest],
    buffers: &[Option<Vec<f32>>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    let mut slices = Vec::with_capacity(chunks.len());
    let mut remaining = output;
    for chunk in chunks {
        let (this_chunk, rest) = remaining.split_at_mut(nest_output_len(chunk));
        slices.push(this_chunk);
        remaining = rest;
    }

    thread::scope(|scope| {
        let handles: Vec<_> = chunks
            .iter()
            .zip(slices)
            .map(|(chunk, slice)| scope.spawn(move || run_nest_into(chunk, buffers, slice)))
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

/// Per-nest retire sets over the *emitted* nest sequence: `result[p]` is
/// every node whose last read is `nests[p]`. This mirrors
/// [`live::annotate`](crate::live::annotate) in shape but is a different
/// computation over a different timeline — the nest sequence, after fusion
/// has already decided which zips never materialize at all.
fn nest_retirement(nests: &[Nest], outputs: &[NodeId]) -> Vec<Vec<NodeId>> {
    let outputs: BTreeSet<NodeId> = outputs.iter().copied().collect();
    let mut last_use: BTreeMap<NodeId, usize> = BTreeMap::new();
    for (position, nest) in nests.iter().enumerate() {
        for (source, _, gather) in &nest.operands {
            last_use.insert(*source, position);
            if let Some(gather_access) = gather {
                last_use.insert(gather_access.indices, position);
            }
        }
    }

    let mut retires = vec![Vec::new(); nests.len()];
    for (node, position) in last_use {
        if !outputs.contains(&node) {
            retires[position].push(node);
        }
    }
    retires
}

fn block_node_ids(program: &[Expr]) -> Vec<NodeId> {
    program
        .iter()
        .enumerate()
        .filter(|(_, expr)| matches!(expr, Expr::Block { .. }))
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
fn reject_non_float32(program: &[Expr]) -> Result<(), TensorError> {
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
fn index_node_ids(program: &[Expr]) -> BTreeSet<NodeId> {
    let mut nodes = BTreeSet::new();
    for expr in program {
        match expr {
            Expr::Block { .. } => {}
            Expr::Zip { operands, .. } => {
                for (_, map) in operands {
                    push_indices_node(map, &mut nodes);
                }
            }
            Expr::Fold(fold) => {
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

fn odometer(shape: &[u64]) -> impl Iterator<Item = Vec<u64>> + '_ {
    let total: u64 = shape.iter().product();
    (0..total).map(move |flat| unflatten(flat, shape))
}

fn unflatten(mut flat: u64, shape: &[u64]) -> Vec<u64> {
    let mut coordinate = vec![0u64; shape.len()];
    for (dim, extent) in shape.iter().enumerate().rev() {
        coordinate[dim] = flat % extent;
        flat /= extent;
    }
    coordinate
}

fn merge_coordinates(
    rank: usize,
    leading_dims: &[u16],
    leading_coordinate: &[u64],
    reduction_dims: &[u16],
    reduction_coordinate: &[u64],
) -> Vec<u64> {
    let mut coordinate = vec![0u64; rank];
    for (dim, value) in leading_dims.iter().zip(leading_coordinate) {
        coordinate[*dim as usize] = *value;
    }
    for (dim, value) in reduction_dims.iter().zip(reduction_coordinate) {
        coordinate[*dim as usize] = *value;
    }
    coordinate
}

fn initial_value(init: FoldInit) -> Option<f32> {
    match init {
        FoldInit::Zero => Some(0.0),
        FoldInit::One => Some(1.0),
        FoldInit::NegativeInfinity => Some(f32::NEG_INFINITY),
        FoldInit::PositiveInfinity => Some(f32::INFINITY),
        FoldInit::FirstElement => None,
    }
}

fn run_nest(nest: &Nest, buffers: &[Option<Vec<f32>>]) -> Result<Vec<f32>, TensorError> {
    let mut output = vec![0.0f32; nest_output_len(nest)];
    run_nest_into(nest, buffers, &mut output)?;
    Ok(output)
}

/// Runs `nest`, writing its output into a caller-provided slice instead of
/// allocating one. This is the primitive [`run_nest`] (sequential) and
/// [`evaluate_parallel`] (one call per chunk, each writing a disjoint
/// sub-slice of the same parent buffer) both drive — the loop nests below
/// are written once, here.
fn run_nest_into(
    nest: &Nest,
    buffers: &[Option<Vec<f32>>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    match &nest.reduction {
        None => run_elementwise(nest, buffers, output),
        Some(reduction) => match reduction.keep {
            Keep::Last => run_reduce(nest, reduction, buffers, output),
            Keep::All => run_scan(nest, reduction, buffers, output),
        },
    }
}

/// The output length [`run_nest_into`] expects from `nest`: the full
/// iteration space for an elementwise nest or a `Keep::All` scan (neither
/// drops any dim), or the reduced (leading dims x width) shape for a
/// `Keep::Last` fold.
fn nest_output_len(nest: &Nest) -> usize {
    match &nest.reduction {
        Some(reduction) if reduction.keep == Keep::Last => {
            let (leading_output_dims, last_output_dim) = output_dims_split(reduction);
            let leading_product: u64 = leading_output_dims
                .iter()
                .map(|dim| nest.extents[*dim as usize])
                .product();
            let width = last_output_dim.map_or(1, |dim| nest.extents[dim as usize] as usize);
            leading_product as usize * width
        }
        _ => element_count(&nest.extents),
    }
}

/// `reduction.output_dims`, split into the leading (outer) dims and the
/// innermost one (if any) — shared by [`run_reduce`] and [`nest_output_len`]
/// so the two agree on shape by construction.
fn output_dims_split(reduction: &Reduction) -> (&[u16], Option<u16>) {
    match reduction.output_dims.split_last() {
        Some((last, leading)) => (leading, Some(*last)),
        None => (&[], None),
    }
}

fn operand_buffers<'a>(
    nest: &Nest,
    buffers: &'a [Option<Vec<f32>>],
) -> Result<Vec<&'a [f32]>, TensorError> {
    nest.operands
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

/// Builds one [`GatherCursor`] per operand that gathers (`None` for the
/// rest), each initialized at `coordinate` and advancing by `stride_dim`'s
/// stride per step — `stride_dim` is `None` where there is no per-step
/// dimension at all (a scalar reduction's single accumulator).
fn build_gather_cursors<'a>(
    nest: &Nest,
    buffers: &'a [Option<Vec<f32>>],
    coordinate: &[u64],
    stride_dim: Option<u16>,
) -> Result<Vec<Option<GatherCursor<'a>>>, TensorError> {
    nest.operands
        .iter()
        .map(|(_, _, gather)| {
            gather
                .as_ref()
                .map(|gather_access| {
                    let buffer = buffer_of(buffers, gather_access.indices)?;
                    Ok(GatherCursor {
                        buffer,
                        offset: gather_access.index_view.offset_of(coordinate),
                        stride: stride_dim.map_or(0, |dim| gather_access.index_view.stride(dim)),
                        element_stride: gather_access.element_stride,
                        extent: gather_access.extent,
                    })
                })
                .transpose()
        })
        .collect()
}

fn run_elementwise(
    nest: &Nest,
    buffers: &[Option<Vec<f32>>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    let (outer_extents, inner_len) = split_innermost(&nest.extents);
    let innermost_dim = outer_extents.len() as u16;
    let raw = operand_buffers(nest, buffers)?;

    for (outer_position, outer_coordinate) in odometer(outer_extents).enumerate() {
        let mut running: Vec<i64> = nest
            .operands
            .iter()
            .map(|(_, view, _)| view.offset_of(&outer_coordinate))
            .collect();
        let strides: Vec<i64> = nest
            .operands
            .iter()
            .map(|(_, view, _)| view.stride(innermost_dim))
            .collect();
        let mut gather_cursors =
            build_gather_cursors(nest, buffers, &outer_coordinate, Some(innermost_dim))?;
        let out_base = outer_position * inner_len;

        for step in 0..inner_len {
            let mut scratch = [0.0f32; 3];
            for (index, data) in raw.iter().enumerate() {
                let mut offset = running[index];
                if let Some(cursor) = gather_cursors[index].as_mut() {
                    offset += cursor.fetch_and_advance(nest.node)?;
                }
                scratch[index] = data[offset as usize];
                running[index] += strides[index];
            }
            output[out_base + step] = apply_scalar_op(nest.body, &scratch[..raw.len()]);
        }
    }
    Ok(())
}

fn run_reduce(
    nest: &Nest,
    reduction: &Reduction,
    buffers: &[Option<Vec<f32>>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    let raw = operand_buffers(nest, buffers)?;

    let reduction_dims: Vec<u16> = (0..nest.extents.len() as u16)
        .filter(|dim| !reduction.output_dims.contains(dim))
        .collect();
    let (leading_output_dims, last_output_dim) = output_dims_split(reduction);

    let leading_extents: Vec<u64> = leading_output_dims
        .iter()
        .map(|dim| nest.extents[*dim as usize])
        .collect();
    let reduction_extents: Vec<u64> = reduction_dims
        .iter()
        .map(|dim| nest.extents[*dim as usize])
        .collect();
    let width = last_output_dim.map_or(1, |dim| nest.extents[dim as usize] as usize);

    for leading_coordinate in odometer(&leading_extents) {
        let mut accumulator = vec![initial_value(reduction.init).unwrap_or(0.0); width];
        let mut seeded = !matches!(reduction.init, FoldInit::FirstElement);

        for reduction_coordinate in odometer(&reduction_extents) {
            let full_coordinate = merge_coordinates(
                nest.extents.len(),
                leading_output_dims,
                &leading_coordinate,
                &reduction_dims,
                &reduction_coordinate,
            );
            let mut running: Vec<i64> = nest
                .operands
                .iter()
                .map(|(_, view, _)| view.offset_of(&full_coordinate))
                .collect();
            let strides: Vec<i64> = nest
                .operands
                .iter()
                .map(|(_, view, _)| last_output_dim.map_or(0, |dim| view.stride(dim)))
                .collect();
            let mut gather_cursors =
                build_gather_cursors(nest, buffers, &full_coordinate, last_output_dim)?;

            for slot in &mut accumulator {
                let mut scratch = [0.0f32; 3];
                for (index, data) in raw.iter().enumerate() {
                    let mut offset = running[index];
                    if let Some(cursor) = gather_cursors[index].as_mut() {
                        offset += cursor.fetch_and_advance(nest.node)?;
                    }
                    scratch[index] = data[offset as usize];
                    running[index] += strides[index];
                }
                let value = apply_scalar_op(nest.body, &scratch[..raw.len()]);
                *slot = if seeded {
                    apply_scalar_op(reduction.body, &[*slot, value])
                } else {
                    value
                };
            }
            seeded = true;
        }

        let out_full_coordinate = merge_coordinates(
            nest.extents.len(),
            leading_output_dims,
            &leading_coordinate,
            &[],
            &[],
        );
        let out_prefix = reduction.out_view.offset_of(&out_full_coordinate);
        let out_stride = last_output_dim.map_or(0, |dim| reduction.out_view.stride(dim));
        for (slot, value) in accumulator.iter().enumerate() {
            output[(out_prefix + out_stride * slot as i64) as usize] = *value;
        }
    }
    Ok(())
}

fn run_scan(
    nest: &Nest,
    reduction: &Reduction,
    buffers: &[Option<Vec<f32>>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    let raw = operand_buffers(nest, buffers)?;
    let (outer_extents, inner_len) = split_innermost(&nest.extents);
    let innermost_dim = outer_extents.len() as u16;

    let mut accumulator = initial_value(reduction.init).unwrap_or(0.0);
    let mut seeded = !matches!(reduction.init, FoldInit::FirstElement);

    for outer_coordinate in odometer(outer_extents) {
        let mut running: Vec<i64> = nest
            .operands
            .iter()
            .map(|(_, view, _)| view.offset_of(&outer_coordinate))
            .collect();
        let strides: Vec<i64> = nest
            .operands
            .iter()
            .map(|(_, view, _)| view.stride(innermost_dim))
            .collect();
        let mut gather_cursors =
            build_gather_cursors(nest, buffers, &outer_coordinate, Some(innermost_dim))?;
        let mut out_running = reduction.out_view.offset_of(&outer_coordinate);
        let out_stride = reduction.out_view.stride(innermost_dim);

        for _ in 0..inner_len {
            let mut scratch = [0.0f32; 3];
            for (index, data) in raw.iter().enumerate() {
                let mut offset = running[index];
                if let Some(cursor) = gather_cursors[index].as_mut() {
                    offset += cursor.fetch_and_advance(nest.node)?;
                }
                scratch[index] = data[offset as usize];
                running[index] += strides[index];
            }
            let value = apply_scalar_op(nest.body, &scratch[..raw.len()]);
            accumulator = if seeded {
                apply_scalar_op(reduction.body, &[accumulator, value])
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
    use crate::expr::{Extent, Fold, append};
    use crate::map::{self, AffineTerm, IndexMap};
    use rstest::rstest;

    fn block(program: &mut Vec<Expr>, dtype: DType, shape: &[Extent]) -> NodeId {
        append(
            program,
            Expr::Block {
                dtype,
                shape: shape.to_vec(),
                name: None,
            },
        )
    }

    fn f32_block(program: &mut Vec<Expr>, shape: &[Extent]) -> NodeId {
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

    fn matmul_program(m: u32, k: u32, n: u32, symbolic: bool) -> (Vec<Expr>, NodeId) {
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
            Expr::Zip {
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
            Expr::Fold(Fold {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: FoldInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Last,
                name: Some("matmul".into()),
            }),
        );
        (program, sum)
    }

    /// `table[ids[s], d]` over iteration space `(s, d)`: dim 0 (vocab) is
    /// gathered by `ids`, dim 1 (feature) is a plain projection.
    fn embedding_lookup_program(vocab: u32, dim: u32, seq: u32) -> (Vec<Expr>, NodeId) {
        let mut program = Vec::new();
        let table = f32_block(&mut program, &[Extent::Static(vocab), Extent::Static(dim)]);
        let ids = block(&mut program, DType::Int32, &[Extent::Static(seq)]);
        let gathered_map = IndexMap::Computed {
            indices: ids,
            index_map: map::projection(2, &[0]),
            base: map::AffineMap {
                iter_rank: 2,
                dims: alloc::vec![
                    map::DimExpr::default(),
                    map::DimExpr {
                        terms: alloc::vec![AffineTerm::projection(1)],
                        offset: 0,
                    },
                ],
            },
            gathered_dim: 0,
        };
        let gathered = append(
            &mut program,
            Expr::Zip {
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
    fn embedding_matmul_program(vocab: u32, embed_dim: u32, seq: u32, out_dim: u32) -> Vec<Expr> {
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
            base: map::AffineMap {
                iter_rank: 3,
                dims: alloc::vec![
                    map::DimExpr::default(),
                    map::DimExpr {
                        terms: alloc::vec![AffineTerm::projection(2)],
                        offset: 0,
                    },
                ],
            },
            gathered_dim: 0,
        };
        let weight_map = IndexMap::Affine(map::projection(3, &[2, 1]));

        let product = append(
            &mut program,
            Expr::Zip {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(table, gather_map), (weight, weight_map)],
                name: None,
            },
        );
        append(
            &mut program,
            Expr::Fold(Fold {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: FoldInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Last,
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
        let nests = nest::lower(&program, &shapes, &[]).expect("embedding matmul lowers");
        assert_eq!(
            nests.len(),
            1,
            "the gather zip must fuse into the fold, not materialize separately"
        );
        assert!(nests[0].reduction.is_some());

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
        let nests = nest::lower(&program, &shapes, &[]).expect("lowers");
        assert_eq!(nests.len(), 1, "fused into one reduction nest");
        assert!(
            element_count(&nests[0].extents) >= PARALLEL_THRESHOLD,
            "this size must clear the threshold or this test proves nothing about the \
             threaded path"
        );
        assert!(
            nests[0].split(4).is_some(),
            "the nest must actually be splittable for the threaded path to run"
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
            Expr::Zip {
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
            Expr::Zip {
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
            &[(&[AffineTerm::scaled(0, 1), AffineTerm::scaled(1, 1)], 0)],
        ));
        let product = append(
            &mut program,
            Expr::Zip {
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
            Expr::Fold(Fold {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: FoldInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
                out_map: IndexMap::Affine(map::projection(2, &[0])),
                keep: Keep::Last,
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
            Expr::Fold(Fold {
                dtype: DType::Float32,
                body: ScalarOp::Maximum,
                init: FoldInit::NegativeInfinity,
                operand: input,
                in_map: row_map.clone(),
                out_map: broadcast_map.clone(),
                keep: Keep::Last,
                name: None,
            }),
        );
        let shifted = append(
            &mut program,
            Expr::Zip {
                dtype: DType::Float32,
                body: ScalarOp::Subtract,
                operands: alloc::vec![(input, row_map.clone()), (max, broadcast_map.clone())],
                name: None,
            },
        );
        let exponentiated = append(
            &mut program,
            Expr::Zip {
                dtype: DType::Float32,
                body: ScalarOp::Exponential,
                operands: alloc::vec![(shifted, row_map.clone())],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Expr::Fold(Fold {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: FoldInit::Zero,
                operand: exponentiated,
                in_map: row_map.clone(),
                out_map: broadcast_map.clone(),
                keep: Keep::Last,
                name: None,
            }),
        );
        append(
            &mut program,
            Expr::Zip {
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
            Expr::Fold(Fold {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: FoldInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[0])),
                keep: Keep::All,
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
                Expr::Zip {
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
        assert!(
            evaluated.peak_live_buffers() <= 3,
            "streaming a chain of 8 unary zips should not hold one buffer per zip, got {}",
            evaluated.peak_live_buffers()
        );
        let _ = current;
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
                Expr::Zip {
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
            matches!(error, TensorError::BlockCountMismatch { .. }),
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
            matches!(error, TensorError::BlockSizeMismatch { .. }),
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
            base: map::AffineMap {
                iter_rank: 1,
                dims: alloc::vec![map::DimExpr::default()],
            },
            gathered_dim: 0,
        };
        append(
            &mut program,
            Expr::Fold(Fold {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: FoldInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map,
                keep: Keep::Last,
                name: None,
            }),
        );

        let error = evaluate(&program, &[], &[], &[]).expect_err("scatter is not lowerable in v1");
        assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
    }

    // -- Nest::split proof: chunks executed by hand, one buffer, equal the
    // -- unsplit result. `nest.rs` owns the split's geometry; the interpreter
    // -- that can actually run a chunk lives only here, std-gated.

    #[test]
    fn splitting_an_elementwise_nest_and_running_its_chunks_matches_the_unsplit_result() {
        let mut program = Vec::new();
        let source = f32_block(&mut program, &[Extent::Static(10), Extent::Static(4)]);
        append(
            &mut program,
            Expr::Zip {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: alloc::vec![(source, IndexMap::Affine(map::projection(2, &[0, 1])))],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("elementwise infers");
        let nests = nest::lower(&program, &shapes, &[]).expect("elementwise lowers");
        let nest = &nests[0];

        let data: Vec<f32> = (0..40).map(|value| value as f32 * 0.01).collect();
        let mut buffers: Vec<Option<Vec<f32>>> = vec![None; program.len()];
        buffers[0] = Some(data);

        let unsplit = run_nest(nest, &buffers).expect("unsplit runs");

        let chunks = nest.split(3).expect("extent 10 over 3 parts splits");
        let mut split_output = vec![0.0f32; unsplit.len()];
        let mut remaining = split_output.as_mut_slice();
        for chunk in &chunks {
            let (this_chunk, rest) = remaining.split_at_mut(nest_output_len(chunk));
            run_nest_into(chunk, &buffers, this_chunk).expect("chunk runs");
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
        let nests = nest::lower(&program, &shapes, &[]).expect("matmul lowers");
        assert_eq!(nests.len(), 1, "fused into one reduction nest");
        let nest = &nests[0];

        let mut buffers: Vec<Option<Vec<f32>>> = vec![None; program.len()];
        buffers[0] = Some(lhs);
        buffers[1] = Some(rhs);

        let unsplit = run_nest(nest, &buffers).expect("unsplit runs");

        let chunks = nest.split(2).expect("8 rows over 2 parts splits");
        let mut split_output = vec![0.0f32; unsplit.len()];
        let mut remaining = split_output.as_mut_slice();
        for chunk in &chunks {
            let (this_chunk, rest) = remaining.split_at_mut(nest_output_len(chunk));
            run_nest_into(chunk, &buffers, this_chunk).expect("chunk runs");
            remaining = rest;
        }

        assert_eq!(split_output, unsplit);
    }

    // -- evaluate_parallel proof tests: same programs, workers in {1, 2, 3, 8},
    // -- bitwise-equal to `evaluate`.

    fn assert_parallel_matches_sequential(
        program: &[Expr],
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
                Expr::Zip {
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
            Expr::Fold(Fold {
                dtype: DType::Float32,
                body: ScalarOp::Maximum,
                init: FoldInit::NegativeInfinity,
                operand: input,
                in_map: row_map.clone(),
                out_map: broadcast_map.clone(),
                keep: Keep::Last,
                name: None,
            }),
        );
        let shifted = append(
            &mut program,
            Expr::Zip {
                dtype: DType::Float32,
                body: ScalarOp::Subtract,
                operands: alloc::vec![(input, row_map.clone()), (max, broadcast_map.clone())],
                name: None,
            },
        );
        let exponentiated = append(
            &mut program,
            Expr::Zip {
                dtype: DType::Float32,
                body: ScalarOp::Exponential,
                operands: alloc::vec![(shifted, row_map.clone())],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Expr::Fold(Fold {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: FoldInit::Zero,
                operand: exponentiated,
                in_map: row_map.clone(),
                out_map: broadcast_map.clone(),
                keep: Keep::Last,
                name: None,
            }),
        );
        append(
            &mut program,
            Expr::Zip {
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
            Expr::Fold(Fold {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: FoldInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[0])),
                keep: Keep::All,
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
                Expr::Zip {
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
        let nests = nest::lower(&program, &shapes, &[]).expect("64x64x64 matmul lowers");
        assert_eq!(nests.len(), 1, "fused into one reduction nest");
        assert!(
            element_count(&nests[0].extents) >= PARALLEL_THRESHOLD,
            "this size must clear the threshold or this test proves nothing about the \
             threaded path"
        );
        assert!(
            nests[0].split(4).is_some(),
            "the nest must actually be splittable for the threaded path to run"
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
            base: map::AffineMap {
                iter_rank: 1,
                dims: alloc::vec![map::DimExpr::default()],
            },
            gathered_dim: 0,
        };
        append(
            &mut scatter_program,
            Expr::Fold(Fold {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: FoldInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map,
                keep: Keep::Last,
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
                Expr::Zip {
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

        assert!(
            evaluated.peak_live_buffers() <= 3,
            "streaming a chain of 8 unary zips should not hold one buffer per zip, got {}",
            evaluated.peak_live_buffers()
        );
    }
}
