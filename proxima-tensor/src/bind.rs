//! Tensor operations with their addressing resolved — the seam between the
//! symbolic program and any executor.
//!
//! There is no second intermediate representation here. An [`BoundOp`] is an
//! [`Op`] whose addressing has been worked out against one call's symbol
//! bindings — the same elementwise/reduce shape, with a resolved [`Layout`]
//! standing in for a symbolic [`IndexMap`] and resolved iteration extents
//! standing in for a symbolic [`Extent`] list. Nothing here is a competing
//! tree: [`bind`] still walks the program left to right, one `Op` at a
//! time, and every `BoundOp` it emits corresponds to exactly one `Op` that
//! actually computes (`Op::Input` never does — it is where data enters,
//! not where anything is derived).
//!
//! [`BoundOpBuilder`] does two things in one pass rather than two passes, and
//! that is a property of the algebra, not a corner cut: whether an
//! elementwise operand's layout is ever worth resolving on its own depends
//! entirely on whether the reduce consuming it can absorb that elementwise
//! op's body directly (the fusion decision). Splitting "resolve layout" from
//! "fuse" into separate stages would mean resolving a `Layout` for every
//! elementwise op and then throwing most of them away — real work with no
//! payoff — so the one `BoundOpBuilder` stage decides both at once, exactly the
//! way [`shape::ShapeTable`] decides shape *and* validity in one pass rather
//! than validating first and shaping second.
//!
//! Like [`shape::ShapeTable`], [`BoundOpBuilder`] is a sans-IO push state
//! machine: a program can arrive a step at a time, and op building must not
//! require the whole thing in hand. [`bind`] is the batch driver over it.
//! What *does* require the whole program in hand is liveness
//! ([`live::annotate`]) — computed once, upstream, and handed to
//! `BoundOpBuilder` as a plain kill-flag list it never has to guess at;
//! `BoundOpBuilder::new` takes that list up front for exactly this reason,
//! streamed or not.
//!
//! `BoundOpBuilder` also implements [`Pipe`]
//! (`In = (Op, Shapes)`, `Out = Vec<BoundOp>`) — the same state machine, not a
//! second type wrapping it. [`Pipe::call`] takes `&self`, so `held` below is
//! a `RefCell` and the node position a `Cell`, the same interior-mutability
//! idiom [`shape::ShapeTable`] uses for its own `Pipe` impl.
//!
//! The one optimization decided here: when a reduce's operand is an
//! elementwise op whose last use is that reduce (exact liveness, from
//! [`live::annotate`]), the elementwise op is never materialized — its body
//! is composed directly into the reduce's [`BoundOp`], which is the difference
//! between an O(extents) buffer and an O(iteration space) one for something
//! like matmul. An elementwise op whose last use is anything else (another
//! elementwise op, a non-fusable reduce, or nothing — a requested output or
//! dead code) materializes as its own `BoundOp`, emitted the moment that use is
//! seen (or, for dead code and outputs never referenced again, when
//! [`BoundOpBuilder::finish`] flushes it).

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};
use core::future::Future;

use proxima_primitives::pipe::Pipe;

use crate::error::TensorError;
use crate::live;
use crate::map::{IndexMap, IndexPattern};
use crate::op::{Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};
use crate::shape::{self, Shapes};

/// One operand's address into its own buffer, expressed directly in an
/// [`BoundOp`]'s iteration-axis space: `strides[axis]` is how far the linear
/// offset moves per step of iteration axis `axis`. An axis this operand
/// never varies along (broadcast) simply has stride 0.
///
/// This is the tensor-domain notion of memory layout (the same concept
/// `torch.Tensor.stride()` names) — the resolved counterpart of an
/// [`IndexPattern`], the same category as `IndexPattern` itself, not a rival
/// record kind, so it stays a small value type with its own accessors
/// rather than being folded away into bare tuple fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub base: i64,
    pub strides: Vec<i64>,
}

impl Layout {
    #[must_use]
    pub fn offset_of(&self, coordinate: &[u64]) -> i64 {
        self.base
            + coordinate
                .iter()
                .zip(&self.strides)
                .map(|(index, stride)| stride * (*index as i64))
                .sum::<i64>()
    }

    #[must_use]
    pub fn stride(&self, axis: u16) -> i64 {
        self.strides.get(axis as usize).copied().unwrap_or(0)
    }
}

/// The extra addressing a gathered operand needs on top of its [`Layout`]:
/// where to fetch the index from, and how to turn a fetched index into an
/// offset once it lands — the same shape an embedding table lookup needs.
/// `element_stride` is the operand's own per-element stride along the
/// gathered axis (the table's row stride, for an embedding lookup) — an
/// executor's runtime read offset is
/// `layout.offset_of(coord) + fetched_index * element_stride`. `extent` is
/// the gathered axis's size, carried so an executor can reject an
/// out-of-range fetched index instead of reading past the buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lookup {
    pub indices: NodeId,
    pub index_layout: Layout,
    pub element_stride: i64,
    pub extent: u64,
}

type BoundOperands = Vec<(NodeId, Layout, Option<Lookup>)>;

/// One tensor op with its addressing resolved: which buffers to read, at
/// what layout, combined by which scalar op — and, for a reduce, how the
/// reduction is shaped. Carries only what an executor needs, nothing about
/// *how* to execute it: [`cpu`](crate::cpu) interprets an `BoundOp` with nested
/// loops; a GPU backend could instead emit kernel source from the same
/// descriptor. Neither backend's shape belongs in this module.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundOp {
    pub node: NodeId,
    /// The iteration-space extents this op's loop walks. Equal to the
    /// output shape for an [`BoundOpKind::Elementwise`] or a `Keep::Scan` reduce
    /// (a scan drops no axis); wider than the output shape for a
    /// `Keep::Reduce` reduce, which walks the full pre-reduction space.
    pub extents: Vec<u64>,
    pub kind: BoundOpKind,
}

/// The resolved counterpart of [`Op`]'s `Elementwise`/`Reduce` variants —
/// `Input` has no counterpart because a leaf never computes anything to
/// resolve.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundOpKind {
    Elementwise {
        op: ScalarOp,
        operands: BoundOperands,
    },
    Reduce {
        /// The per-step combine of `operands` before reducing: the fused
        /// elementwise op's own body when one was absorbed,
        /// [`ScalarOp::Identity`] otherwise. Distinct from `reduce_op`,
        /// which is the reduce's own accumulation.
        element_op: ScalarOp,
        reduce_op: ScalarOp,
        init: ReduceInit,
        keep: Keep,
        operands: BoundOperands,
        /// Iteration axes that survive to the output, in the reduce's
        /// `out_map` operand-axis order. The last entry (if any) is the
        /// innermost loop.
        output_axes: Vec<u16>,
        out_layout: Layout,
    },
}

impl BoundOp {
    #[must_use]
    pub fn operands(&self) -> &[(NodeId, Layout, Option<Lookup>)] {
        match &self.kind {
            BoundOpKind::Elementwise { operands, .. } | BoundOpKind::Reduce { operands, .. } => {
                operands
            }
        }
    }

    /// The scalar op applied per step to build one combined value from
    /// `operands()`, before any reduction: an elementwise op's own body, or
    /// a fused reduce's absorbed body (`Identity` if nothing fused).
    #[must_use]
    pub fn element_op(&self) -> ScalarOp {
        match &self.kind {
            BoundOpKind::Elementwise { op, .. } => *op,
            BoundOpKind::Reduce { element_op, .. } => *element_op,
        }
    }

    /// Splits this op along its outermost output-iteration axis into
    /// `parts` contiguous chunks so each chunk can execute independently:
    /// for an `Elementwise` op that axis is `extents[0]`; for a
    /// `Keep::Reduce` reduce it is the first entry of `output_axes` (the
    /// reduce's outermost surviving axis, per that field's own doc). This
    /// split is backend-neutral on purpose: a CPU driver runs chunks on
    /// worker threads, a GPU backend would tile the identical axis into
    /// threadgroups — neither belongs in this module, only the geometry
    /// does.
    ///
    /// Returns `None` when splitting is not sound or not useful:
    /// - a scalar reduction (`output_axes` is empty — nothing to split, the
    ///   whole op is one accumulator),
    /// - a `Keep::Scan` scan (each step reads the previous step's output,
    ///   so the extent is a sequential dependency, not parallel work),
    /// - the split axis's extent is smaller than `parts` (some chunk would
    ///   be empty),
    /// - `parts < 2` (nothing to split into).
    ///
    /// Chunk `k`'s output occupies the contiguous range
    /// `[chunk_start * inner_size, chunk_start * inner_size + chunk_len *
    /// inner_size)` of the parent output buffer, where `inner_size` is the
    /// product of the extents after the split axis. A caller relies on this
    /// to hand each chunk its own disjoint `&mut` sub-slice — via successive
    /// `split_at_mut` of one parent buffer — and run every chunk
    /// concurrently with no further coordination.
    ///
    /// The two layout kinds are rebased differently, which is *why* that
    /// works:
    /// - every **operand** layout's base is shifted by
    ///   `chunk_start * stride(split_axis)`, because operands are read from
    ///   the one full, unsplit source buffer shared by every chunk.
    /// - a reduce's `out_layout` is **not** shifted at all: the interpreter
    ///   already derives every write offset from a loop that iterates the
    ///   split axis's own (already-shrunk) extent starting at 0, for every
    ///   chunk alike, so an out_layout carrying the parent's unmodified
    ///   base already produces exactly the 0-based offsets a `split_at_mut`
    ///   sub-slice expects. Rebasing it too would double-count the offset.
    #[must_use]
    pub fn split(&self, parts: usize) -> Option<Vec<BoundOp>> {
        if parts < 2 {
            return None;
        }
        let split_axis = self.split_axis()?;
        let extent = self.extents[split_axis as usize];
        if extent < parts as u64 {
            return None;
        }

        Some(
            chunk_ranges(extent, parts)
                .map(|(chunk_start, chunk_len)| {
                    self.rebase_chunk(split_axis, chunk_start, chunk_len)
                })
                .collect(),
        )
    }

    fn split_axis(&self) -> Option<u16> {
        match &self.kind {
            BoundOpKind::Elementwise { .. } => (!self.extents.is_empty()).then_some(0),
            BoundOpKind::Reduce {
                keep, output_axes, ..
            } => match keep {
                Keep::Scan => None,
                Keep::Reduce => output_axes.first().copied(),
            },
        }
    }

    fn rebase_chunk(&self, split_axis: u16, chunk_start: u64, chunk_len: u64) -> BoundOp {
        let mut extents = self.extents.clone();
        extents[split_axis as usize] = chunk_len;

        let kind = match &self.kind {
            BoundOpKind::Elementwise { op, operands } => BoundOpKind::Elementwise {
                op: *op,
                operands: rebase_operands(operands, split_axis, chunk_start),
            },
            BoundOpKind::Reduce {
                element_op,
                reduce_op,
                init,
                keep,
                operands,
                output_axes,
                out_layout,
            } => BoundOpKind::Reduce {
                element_op: *element_op,
                reduce_op: *reduce_op,
                init: *init,
                keep: *keep,
                operands: rebase_operands(operands, split_axis, chunk_start),
                output_axes: output_axes.clone(),
                // unchanged from the parent: see this method's doc for why
                // an unshifted out_layout already yields 0-based write
                // offsets.
                out_layout: out_layout.clone(),
            },
        };

        BoundOp {
            node: self.node,
            extents,
            kind,
        }
    }
}

fn rebase_operands(operands: &BoundOperands, split_axis: u16, chunk_start: u64) -> BoundOperands {
    operands
        .iter()
        .map(|(node, layout, lookup)| {
            let rebased_lookup = lookup.as_ref().map(|lookup| Lookup {
                indices: lookup.indices,
                index_layout: rebase_layout(&lookup.index_layout, split_axis, chunk_start),
                element_stride: lookup.element_stride,
                extent: lookup.extent,
            });
            (
                *node,
                rebase_layout(layout, split_axis, chunk_start),
                rebased_lookup,
            )
        })
        .collect()
}

/// `parts` contiguous `(start, len)` ranges covering `0..extent`: the first
/// `parts - 1` ranges are `extent / parts` wide, the last absorbs whatever
/// remains, so it is the only one that can be a different (ragged) size.
fn chunk_ranges(extent: u64, parts: usize) -> impl Iterator<Item = (u64, u64)> {
    let chunk_len = extent / parts as u64;
    (0..parts).scan(0u64, move |start, index| {
        let chunk_start = *start;
        let len = if index + 1 == parts {
            extent - chunk_start
        } else {
            chunk_len
        };
        *start += len;
        Some((chunk_start, len))
    })
}

fn rebase_layout(layout: &Layout, split_axis: u16, chunk_start: u64) -> Layout {
    Layout {
        base: layout.base + layout.stride(split_axis) * chunk_start as i64,
        strides: layout.strides.clone(),
    }
}

struct HeldElementwise {
    body: ScalarOp,
    operands: Vec<(NodeId, IndexMap)>,
}

/// The prefix state of op building: elementwise ops seen but not yet
/// materialized.
///
/// `retires` and `position` make [`BoundOpBuilder::push`] a single-argument-per-node
/// step (`expr`, `shapes`) rather than needing `node`/`retires` threaded in
/// by the caller on every call: `retires[i]` is node `i`'s kill-flag list
/// (see [`live::annotate`], computed once over the whole program before the
/// first push), and `position` is the node id the next push resolves to —
/// both pieces this type already needed to know, now carried as its own
/// state instead of repeated arguments.
pub struct BoundOpBuilder {
    held: RefCell<BTreeMap<NodeId, HeldElementwise>>,
    retires: Vec<Vec<NodeId>>,
    position: Cell<u32>,
}

impl BoundOpBuilder {
    /// `retires` is normally [`live::annotate`]`(program, outputs)`.
    #[must_use]
    pub fn new(retires: Vec<Vec<NodeId>>) -> Self {
        Self {
            held: RefCell::new(BTreeMap::new()),
            retires,
            position: Cell::new(0),
        }
    }

    /// Judge one expression: hold an elementwise op, or emit whatever is now
    /// ready.
    ///
    /// May return more than one [`BoundOp`]: consuming a held elementwise op
    /// that turns out not to fuse must materialize it before the current
    /// expression can read it, so a single push can ready both that
    /// standalone op and the current expression's own.
    pub fn push(&self, expr: &Op, shapes: &Shapes) -> Result<Vec<BoundOp>, TensorError> {
        let node = NodeId(self.position.get());
        self.position.set(self.position.get() + 1);
        let empty = Vec::new();
        let retires = self.retires.get(node.0 as usize).unwrap_or(&empty);

        let mut emitted = Vec::new();

        match expr {
            Op::Input { .. } => {}
            Op::Elementwise { body, operands, .. } => {
                for (operand_node, _) in operands {
                    self.materialize_if_held(*operand_node, shapes, &mut emitted);
                }
                self.held.borrow_mut().insert(
                    node,
                    HeldElementwise {
                        body: *body,
                        operands: operands.clone(),
                    },
                );
            }
            Op::Reduce(reduce) => {
                let fuses = retires.contains(&reduce.operand)
                    && is_identity_projection(&reduce.in_map)
                    && self.held.borrow().contains_key(&reduce.operand);

                let (element_op, operands) = if let Some(held) = fuses
                    .then(|| self.held.borrow_mut().remove(&reduce.operand))
                    .flatten()
                {
                    compose_fused_operands(shapes, &reduce.in_map, held.body, &held.operands)
                } else {
                    self.materialize_if_held(reduce.operand, shapes, &mut emitted);
                    let operand = build_operand(reduce.operand, &reduce.in_map, shapes);
                    (ScalarOp::Identity, vec![operand])
                };

                emitted.push(build_reduce_op(node, reduce, shapes, element_op, operands));
            }
        }

        Ok(emitted)
    }

    /// Flush every elementwise op still held: it was either a requested
    /// output or dead code, and either way it materializes as its own op.
    pub fn finish(self, shapes: &Shapes) -> Result<Vec<BoundOp>, TensorError> {
        Ok(self
            .held
            .into_inner()
            .into_iter()
            .map(|(node, held)| build_elementwise_op(node, shapes, held.body, &held.operands))
            .collect())
    }

    fn materialize_if_held(&self, node: NodeId, shapes: &Shapes, emitted: &mut Vec<BoundOp>) {
        if let Some(held) = self.held.borrow_mut().remove(&node) {
            emitted.push(build_elementwise_op(
                node,
                shapes,
                held.body,
                &held.operands,
            ));
        }
    }
}

/// `In = (Op, Shapes)` matches [`shape::ShapeTable`]'s own `Pipe::Out`
/// exactly, so `AndThen::new(ShapeTable, BoundOpBuilder)` (or
/// `shapes_instance.and_then(builder_instance)`) composes with no adapter:
/// shape resolution's snapshot travels alongside the `Op` it was resolved
/// for, and [`BoundOpBuilder::push`] reads both straight out of `Self::In`.
impl Pipe for BoundOpBuilder {
    type In = (Op, Shapes);
    type Out = Vec<BoundOp>;
    type Err = TensorError;

    fn call(
        &self,
        (expr, shapes): Self::In,
    ) -> impl Future<Output = Result<Vec<BoundOp>, TensorError>> {
        async move { self.push(&expr, &shapes) }
    }
}

fn build_elementwise_op(
    node: NodeId,
    shapes: &Shapes,
    op: ScalarOp,
    operands: &[(NodeId, IndexMap)],
) -> BoundOp {
    let extents = shapes.of(node).to_vec();
    let built_operands = operands
        .iter()
        .map(|(operand_node, map)| build_operand(*operand_node, map, shapes))
        .collect();
    BoundOp {
        node,
        extents,
        kind: BoundOpKind::Elementwise {
            op,
            operands: built_operands,
        },
    }
}

/// One operand's [`Layout`] (and, for a gather, its [`Lookup`]), built
/// directly from its [`IndexMap`] — the one place that decides how an
/// `Affine` vs a `Computed` map turns into what an executor reads.
fn build_operand(
    node: NodeId,
    map: &IndexMap,
    shapes: &Shapes,
) -> (NodeId, Layout, Option<Lookup>) {
    match map {
        IndexMap::Affine(pattern) => (node, layout_of(pattern, shapes.of(node)), None),
        IndexMap::Computed {
            indices,
            index_map,
            base,
            gathered_dim,
        } => {
            let operand_shape = shapes.of(node);
            let layout = layout_of(base, operand_shape);
            let index_layout = layout_of(index_map, shapes.of(*indices));
            let element_stride = row_major_strides(operand_shape)[*gathered_dim as usize];
            let extent = operand_shape[*gathered_dim as usize];
            let lookup = Lookup {
                indices: *indices,
                index_layout,
                element_stride,
                extent,
            };
            (node, layout, Some(lookup))
        }
    }
}

fn build_reduce_op(
    node: NodeId,
    reduce: &Reduce,
    shapes: &Shapes,
    element_op: ScalarOp,
    operands: BoundOperands,
) -> BoundOp {
    let out_pattern = reduce.out_map.affine();
    let out_layout = layout_of(out_pattern, shapes.of(node));
    let output_axes = pure_projection_axes(out_pattern);
    BoundOp {
        node,
        extents: shape::fold_iteration_extents(reduce, shapes),
        kind: BoundOpKind::Reduce {
            element_op,
            reduce_op: reduce.body,
            init: reduce.init,
            keep: reduce.keep,
            operands,
            output_axes,
            out_layout,
        },
    }
}

fn pure_projection_axes(pattern: &IndexPattern) -> Vec<u16> {
    pattern
        .axes
        .iter()
        .filter_map(|axis| match axis.terms.as_slice() {
            [term] if term.coeff == 1 => Some(term.axis),
            _ => None,
        })
        .collect()
}

/// A fusion can compose through: every axis a plain, unshifted projection.
/// Anything richer (a window, a slice, a stride) still resolves correctly,
/// it just materializes its operand instead of composing through it.
fn is_identity_projection(map: &IndexMap) -> bool {
    if map.is_data_dependent() {
        return false;
    }
    map.affine()
        .axes
        .iter()
        .all(|axis| axis.offset == 0 && matches!(axis.terms.as_slice(), [term] if term.coeff == 1))
}

fn compose_fused_operands(
    shapes: &Shapes,
    in_map: &IndexMap,
    elementwise_body: ScalarOp,
    elementwise_operands: &[(NodeId, IndexMap)],
) -> (ScalarOp, BoundOperands) {
    let in_pattern = in_map.affine();
    let iter_rank = in_pattern.iter_rank;
    let elementwise_axis_to_reduce_axis: Vec<u16> = in_pattern
        .axes
        .iter()
        .map(|axis| axis.terms[0].axis)
        .collect();

    let operands = elementwise_operands
        .iter()
        .map(|(operand_node, map)| {
            let (node, layout, lookup) = build_operand(*operand_node, map, shapes);
            let remapped_layout =
                remap_strides(&layout, &elementwise_axis_to_reduce_axis, iter_rank);
            let remapped_lookup = lookup.map(|lookup| Lookup {
                indices: lookup.indices,
                index_layout: remap_strides(
                    &lookup.index_layout,
                    &elementwise_axis_to_reduce_axis,
                    iter_rank,
                ),
                element_stride: lookup.element_stride,
                extent: lookup.extent,
            });
            (node, remapped_layout, remapped_lookup)
        })
        .collect();

    (elementwise_body, operands)
}

fn layout_of(pattern: &IndexPattern, operand_shape: &[u64]) -> Layout {
    let element_strides = row_major_strides(operand_shape);
    let mut strides = vec![0i64; pattern.iter_rank as usize];
    let mut base = 0i64;
    for (axis_index, axis) in pattern.axes.iter().enumerate() {
        let stride = element_strides[axis_index];
        base += i64::from(axis.offset) * stride;
        for term in &axis.terms {
            strides[term.axis as usize] += i64::from(term.coeff) * stride;
        }
    }
    Layout { base, strides }
}

fn row_major_strides(shape: &[u64]) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];
    let mut accumulator = 1i64;
    for (axis_index, extent) in shape.iter().enumerate().rev() {
        strides[axis_index] = accumulator;
        accumulator *= *extent as i64;
    }
    strides
}

fn remap_strides(layout: &Layout, axis_map: &[u16], iter_rank: u16) -> Layout {
    let mut strides = vec![0i64; iter_rank as usize];
    for (from_axis, to_axis) in axis_map.iter().enumerate() {
        strides[*to_axis as usize] += layout.stride(from_axis as u16);
    }
    Layout {
        base: layout.base,
        strides,
    }
}

/// Batch driver: computes liveness once, then streams every expression
/// through a fresh [`BoundOpBuilder`], flushing whatever remains held at the end.
pub fn bind(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
) -> Result<Vec<BoundOp>, TensorError> {
    let retires = live::annotate(program, outputs);
    let building = BoundOpBuilder::new(retires);
    let mut built = Vec::new();
    for expr in program {
        built.extend(building.push(expr, shapes)?);
    }
    built.extend(building.finish(shapes)?);
    Ok(built)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::map;
    use crate::op::{Extent, append};
    use rstest::rstest;

    fn matmul_program() -> (Vec<Op>, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Symbolic(0), Extent::Static(768)],
                name: None,
            },
        );
        let rhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(768), Extent::Static(3072)],
                name: None,
            },
        );
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
                init: crate::op::ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: Some("matmul".into()),
            }),
        );
        (program, product, sum, lhs)
    }

    #[test]
    fn matmul_resolves_to_one_fused_op_not_two() {
        let (program, product, sum, _lhs) = matmul_program();
        let shapes = shape::infer(&program, &[512]).expect("matmul infers");
        let built = bind(&program, &shapes, &[]).expect("matmul builds ops");

        assert_eq!(
            built.len(),
            1,
            "the elementwise op must not materialize separately"
        );
        assert_eq!(built[0].node, sum);
        assert!(matches!(built[0].kind, BoundOpKind::Reduce { .. }));
        assert_ne!(
            built[0].element_op(),
            ScalarOp::Identity,
            "the fused body is the elementwise op's multiply"
        );
        let _ = product;
    }

    #[test]
    fn requesting_the_intermediate_elementwise_op_as_an_output_prevents_fusion() {
        let (program, product, sum, _lhs) = matmul_program();
        let shapes = shape::infer(&program, &[512]).expect("matmul infers");
        let built =
            bind(&program, &shapes, &[product, sum]).expect("matmul builds ops with two outputs");

        assert_eq!(
            built.len(),
            2,
            "the requested-output elementwise op must materialize"
        );
        assert!(
            built
                .iter()
                .any(|op| op.node == product && matches!(op.kind, BoundOpKind::Elementwise { .. }))
        );
        assert!(
            built
                .iter()
                .any(|op| op.node == sum && matches!(op.kind, BoundOpKind::Reduce { .. }))
        );
    }

    #[test]
    fn a_broadcast_operand_has_stride_zero_in_the_broadcast_axis() {
        let mut program = Vec::new();
        let matrix = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4), Extent::Static(8)],
                name: None,
            },
        );
        let bias = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(8)],
                name: None,
            },
        );
        let sum = append(
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

        let shapes = shape::infer(&program, &[]).expect("broadcast infers");
        let built = bind(&program, &shapes, &[]).expect("broadcast builds ops");
        let op = built.iter().find(|op| op.node == sum).expect("sum emitted");
        assert_eq!(
            op.operands()[1].1.stride(0),
            0,
            "bias never varies over the batch axis"
        );
        assert_ne!(
            op.operands()[0].1.stride(0),
            0,
            "matrix does vary over the batch axis"
        );
    }

    #[test]
    fn a_conv_window_operand_folds_two_terms_into_one_stride_slot() {
        let mut program = Vec::new();
        let anchor = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4), Extent::Static(2)],
                name: None,
            },
        );
        let signal = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(8)],
                name: None,
            },
        );
        let window = IndexMap::Affine(map::affine(
            2,
            &[(
                &[
                    crate::map::AxisTerm::scaled(0, 2),
                    crate::map::AxisTerm::scaled(1, 1),
                ],
                0,
            )],
        ));
        let touched = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![
                    (anchor, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (signal, window)
                ],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("conv window infers");
        let built = bind(&program, &shapes, &[]).expect("conv window builds ops");
        let op = built
            .iter()
            .find(|op| op.node == touched)
            .expect("touched emitted");
        let signal_layout = &op.operands()[1].1;
        assert_eq!(
            signal_layout.strides.len(),
            2,
            "one stride slot per iteration axis"
        );
        assert_ne!(signal_layout.stride(0), 0, "stride term contributes");
        assert_ne!(signal_layout.stride(1), 0, "dilation term contributes");
    }

    #[test]
    fn transpose_layout_has_permuted_strides() {
        let mut program = Vec::new();
        let matrix = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(3), Extent::Static(5)],
                name: None,
            },
        );
        let transposed = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(matrix, IndexMap::Affine(map::projection(2, &[1, 0])))],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("transpose infers");
        let built = bind(&program, &shapes, &[]).expect("transpose builds ops");
        let op = built
            .iter()
            .find(|op| op.node == transposed)
            .expect("transposed emitted");
        let layout = &op.operands()[0].1;
        // matrix is row-major [3, 5]: elem strides are [5, 1]. axis 0 of the
        // operand (stride 5) projects iteration axis 1; axis 1 (stride 1)
        // projects iteration axis 0, so the strides land permuted relative
        // to iteration order.
        assert_eq!(layout.stride(0), 1);
        assert_eq!(layout.stride(1), 5);
    }

    fn elementwise_op() -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(10), Extent::Static(4)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(source, IndexMap::Affine(map::projection(2, &[0, 1])))],
                name: None,
            },
        );
        let shapes = shape::infer(&program, &[]).expect("elementwise infers");
        bind(&program, &shapes, &[])
            .expect("elementwise builds ops")
            .into_iter()
            .next()
            .expect("one op emitted")
    }

    fn scalar_reduction_op() -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(8)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let shapes = shape::infer(&program, &[]).expect("scalar reduction infers");
        bind(&program, &shapes, &[])
            .expect("scalar reduction builds ops")
            .into_iter()
            .next()
            .expect("one op emitted")
    }

    fn scan_op() -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(8)],
                name: None,
            },
        );
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
        let shapes = shape::infer(&program, &[]).expect("scan infers");
        bind(&program, &shapes, &[])
            .expect("scan builds ops")
            .into_iter()
            .next()
            .expect("one op emitted")
    }

    #[test]
    fn split_of_an_elementwise_op_yields_contiguous_chunks_with_a_ragged_last() {
        let op = elementwise_op();

        let chunks = op.split(3).expect("extent 10 over 3 parts splits");
        assert_eq!(chunks.len(), 3, "one chunk per part");

        let lengths: Vec<u64> = chunks.iter().map(|chunk| chunk.extents[0]).collect();
        assert_eq!(
            lengths,
            alloc::vec![3, 3, 4],
            "only the last chunk is ragged"
        );

        // axis 0's stride is 4 (row-major over [10, 4]): chunk k's operand
        // layout is rebased by chunk_start * stride(0), which is exactly
        // what lets a caller treat each chunk's output as a disjoint
        // sub-slice.
        let stride = op.operands()[0].1.stride(0);
        assert_eq!(chunks[0].operands()[0].1.base, op.operands()[0].1.base);
        assert_eq!(
            chunks[1].operands()[0].1.base,
            op.operands()[0].1.base + stride * 3
        );
        assert_eq!(
            chunks[2].operands()[0].1.base,
            op.operands()[0].1.base + stride * 6
        );
    }

    #[test]
    fn split_of_a_fused_matmul_reduction_rebases_operands_but_not_out_layout() {
        let (program, _product, sum, _lhs) = matmul_program();
        let shapes = shape::infer(&program, &[512]).expect("matmul infers");
        let op = bind(&program, &shapes, &[])
            .expect("matmul builds ops")
            .into_iter()
            .next()
            .expect("one fused op emitted");
        assert_eq!(op.node, sum);
        let BoundOpKind::Reduce { .. } = &op.kind else {
            panic!("the reduction fused with its elementwise op");
        };

        let chunks = op.split(2).expect("512 rows over 2 parts splits");
        assert_eq!(chunks.len(), 2, "one chunk per part");
        for chunk in &chunks {
            assert!(
                matches!(chunk.kind, BoundOpKind::Reduce { .. }),
                "each chunk is still a reduce"
            );
            // the contracted axis (k) is untouched by a split on the output
            // row axis: every chunk still walks the full contraction.
            assert_eq!(chunk.extents[2], op.extents[2]);
        }
        assert_eq!(chunks[0].extents[0], 256, "rows split evenly in half");
        assert_eq!(chunks[1].extents[0], 256);

        // the fused elementwise op's lhs/rhs operand reads are rebased:
        // chunk 1 starts reading lhs at row 256 (row stride = k = 768).
        let lhs_row_stride = op.operands()[0].1.stride(0);
        assert_eq!(
            chunks[1].operands()[0].1.base,
            op.operands()[0].1.base + lhs_row_stride * 256
        );

        // out_layout stays exactly as the parent's: the interpreter's own
        // per-chunk loop already starts each chunk's leading coordinate at
        // 0, so an unshifted out_layout already yields the 0-based write
        // offsets a `split_at_mut` sub-slice expects (see the `split` doc).
        let BoundOpKind::Reduce {
            out_layout: parent_out,
            ..
        } = &op.kind
        else {
            unreachable!("checked above")
        };
        for chunk in &chunks {
            let BoundOpKind::Reduce {
                out_layout: chunk_out,
                ..
            } = &chunk.kind
            else {
                panic!("chunk reduction");
            };
            assert_eq!(chunk_out, parent_out);
        }
    }

    #[rstest]
    #[case::scalar_reduction(scalar_reduction_op(), 2)]
    #[case::keep_scan_scan(scan_op(), 2)]
    #[case::too_few_parts(elementwise_op(), 1)]
    #[case::extent_smaller_than_parts(elementwise_op(), 999)]
    fn split_returns_none_when_unsound_or_unhelpful(#[case] op: BoundOp, #[case] parts: usize) {
        assert!(op.split(parts).is_none());
    }

    // THE PROOF: `ShapeTable` and `BoundOpBuilder` compose through the real
    // `PipeExt` surface (`.and_then`, not hand-sequenced calls dressed up as
    // composition), and the ops that composed chain produces for a matmul
    // program are byte-for-byte the same ops `shape::infer` + `bind::bind`
    // (the free-function path every other test in this crate trusts)
    // produce for the identical program.
    #[test]
    fn infer_and_then_build_ops_matches_the_free_function_pipeline() {
        use crate::shape::ShapeTable;
        use proxima_primitives::pipe::PipeExt;

        let (program, _product, sum, _lhs) = matmul_program();
        let outputs: Vec<NodeId> = Vec::new();
        let retires = live::annotate(&program, &outputs);

        let shape_table = ShapeTable::new(&[512]);
        let builder = BoundOpBuilder::new(retires);
        let chain = shape_table.and_then(builder);

        let mut built_via_pipe = Vec::new();
        for expr in &program {
            let batch = proxima_primitives::block_on(Pipe::call(&chain, expr.clone()))
                .expect("shape+op pipe step succeeds");
            built_via_pipe.extend(batch);
        }

        let shapes = shape::infer(&program, &[512]).expect("free-function infer succeeds");
        let built_via_free_function =
            bind(&program, &shapes, &outputs).expect("free-function op building succeeds");

        assert_eq!(built_via_pipe, built_via_free_function);
        assert_eq!(built_via_pipe.len(), 1, "matmul fuses into one op");
        assert_eq!(built_via_pipe[0].node, sum);
    }
}
