//! Shape inference — and, since it must touch every reference to do it, the
//! only structural validation a program gets.
//!
//! A tensor program is not validated on construction the way the old arena
//! was: [`Op`] is just data. [`ShapeTable`] is where "well-formed" is
//! actually checked — backwards references, elementwise arity, the
//! accumulator-widening rule, iteration-axis range, and the affine
//! unification that resolves every extent. An `Op` is fully judged the
//! moment it is pushed.
//!
//! [`ShapeTable`] is also this crate's sans-IO stance made concrete: a
//! tensor program is something that can arrive a step at a time (a
//! partition crossing a wire is a stream of `Op`s; compiling overlaps
//! transport), and `ShapeTable` is the core that judges each step against
//! everything before it, with no I/O of its own. [`infer`] is the batch
//! case, three lines over the stream; `ShapeTable` also implements [`Pipe`]
//! directly (`In = Op`, `Out = (Op, Shapes)`) — the same core, not a
//! second type wrapping it. [`Pipe::call`] takes `&self`, while judging a
//! node is inherently a mutation, so `dtypes`/`shapes` below are
//! `RefCell`s: the interior-mutability idiom
//! `proxima_primitives::pipe::isolate`'s module doc names for runtime-owned
//! `!Send` pipe state, applied to the state that was already here rather
//! than to a wrapper around it.

use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::future::Future;

use proxima_primitives::pipe::Pipe;

use crate::dtype::DType;
use crate::error::TensorError;
use crate::map::{AxisIndex, IndexMap, IndexPattern};
use crate::op::{Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};

/// The largest integer an f32 can represent exactly — its 24-bit mantissa's
/// width. Gather indices ride in f32 buffers (see
/// [`IndexMap::Computed`](crate::map::IndexMap::Computed) and `cpu.rs`'s
/// module docs), so a gathered axis wider than this could silently address
/// the wrong row once an index value loses precision.
const GATHER_EXTENT_EXACT_FLOAT_LIMIT: u64 = 1 << 24;

/// Every node's resolved output extents, in `u64` regardless of how the
/// program spelled them (`Extent::Static` or a bound `Extent::Symbolic`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Shapes {
    extents: Vec<Vec<u64>>,
}

impl Shapes {
    /// The resolved output shape of `node`.
    ///
    /// Panics if `node` was never pushed to the [`ShapeTable`] that built
    /// this `Shapes` — every valid `NodeId` a program can produce was, by
    /// the backwards-reference rule, resolved before anything that could
    /// ask for it.
    #[must_use]
    pub fn of(&self, node: NodeId) -> &[u64] {
        &self.extents[node.0 as usize]
    }

    fn push(&mut self, resolved: Vec<u64>) {
        self.extents.push(resolved);
    }
}

/// The prefix state of shape inference: every node judged so far.
pub struct ShapeTable {
    symbols: Vec<u64>,
    dtypes: RefCell<Vec<DType>>,
    shapes: RefCell<Shapes>,
}

impl ShapeTable {
    #[must_use]
    pub fn new(symbols: &[u64]) -> Self {
        Self {
            symbols: symbols.to_vec(),
            dtypes: RefCell::new(Vec::new()),
            shapes: RefCell::new(Shapes::default()),
        }
    }

    /// Judge one expression against everything pushed before it.
    pub fn push(&self, expr: &Op) -> Result<(), TensorError> {
        let node = NodeId(self.shapes.borrow().extents.len() as u32);
        let resolved = match expr {
            Op::Input { shape, .. } => resolve_leaf_shape(shape, &self.symbols)?,
            Op::Elementwise { body, operands, .. } => {
                self.infer_elementwise(node, *body, operands)?
            }
            Op::Reduce(reduce) => self.infer_reduce(node, reduce)?,
            Op::Iota { extent, .. } => {
                resolve_leaf_shape(core::slice::from_ref(extent), &self.symbols)?
            }
            Op::Constant { shape, .. } => resolve_leaf_shape(shape, &self.symbols)?,
        };
        self.dtypes.borrow_mut().push(expr.dtype());
        self.shapes.borrow_mut().push(resolved);
        Ok(())
    }

    /// A snapshot of every shape resolved so far — legal to call mid-stream.
    #[must_use]
    pub fn shapes(&self) -> Shapes {
        self.shapes.borrow().clone()
    }

    #[must_use]
    pub fn finish(self) -> Shapes {
        self.shapes.into_inner()
    }

    fn infer_elementwise(
        &self,
        here: NodeId,
        body: ScalarOp,
        operands: &[(NodeId, IndexMap)],
    ) -> Result<Vec<u64>, TensorError> {
        if operands.is_empty() {
            return Err(TensorError::EmptyElementwise(here));
        }
        if operands.len() != body.arity() {
            return Err(TensorError::ArityMismatch {
                node: here,
                found: operands.len(),
                expected: body.arity(),
            });
        }
        for (operand_node, map) in operands {
            check_backward(here, *operand_node)?;
            check_map(here, map)?;
            self.check_indices_dtype(here, map)?;
            self.check_gather_extent(here, *operand_node, map)?;
        }
        let iter_rank = operands
            .iter()
            .map(|(_, map)| combined_iter_rank(map))
            .max()
            .unwrap_or(0);
        let refs: Vec<(NodeId, &IndexMap)> =
            operands.iter().map(|(node, map)| (*node, map)).collect();
        self.unify_iteration_space(here, iter_rank, &refs)
    }

    fn infer_reduce(&self, here: NodeId, reduce: &Reduce) -> Result<Vec<u64>, TensorError> {
        check_backward(here, reduce.operand)?;
        check_map(here, &reduce.in_map)?;
        check_map(here, &reduce.out_map)?;
        self.check_indices_dtype(here, &reduce.in_map)?;
        self.check_indices_dtype(here, &reduce.out_map)?;
        self.check_gather_extent(here, reduce.operand, &reduce.in_map)?;

        if reduce.body.is_associative() && !reduce.dtype.accumulates_in_place() {
            return Err(TensorError::NarrowAccumulator {
                node: here,
                element: self.dtypes.borrow()[reduce.operand.0 as usize],
                accumulator: reduce.dtype,
            });
        }

        let iter_rank = combined_iter_rank(&reduce.in_map);
        let iter_extents =
            self.unify_iteration_space(here, iter_rank, &[(reduce.operand, &reduce.in_map)])?;

        // A scatter's output *shape* is static even though its *addressing*
        // is not: `scatter_output_shape` reads the destination extent
        // [`IndexMap::scatter_extent`] carries, never anything from `here`'s
        // own (not-yet-resolved) shape. Everything below this is the reason
        // a `Reduce`-wide field for that extent was rejected in favor of
        // reusing `IndexMap::Computed`'s existing (otherwise dead, for a
        // write) `base`-at-`gathered_dim` slot: a `Reduce { .. }` struct
        // literal is built at ~150 call sites across this workspace (`grep
        // -rn 'Reduce {'`, checked against this row's own worktree), every
        // one of which a new required field breaks; `IndexMap::Computed`'s
        // own construction sites number ~56, still real but a third the
        // size, and every one of *those* is already about gather/scatter
        // addressing rather than an unrelated majority paying for a feature
        // they never use — the same blast-radius judgment this file's
        // fused-QKV row (below) already applied to `AxisIndex::len`.
        if reduce.out_map.is_data_dependent() {
            if reduce.keep == Keep::Scan {
                return Err(TensorError::NotLowerable {
                    node: here,
                    reason: "a scatter (data-dependent reduce output) cannot be a Keep::Scan: \
                             each scan step would need the destination it just wrote to read \
                             its own predecessor, which the sequential CPU interpreter does not \
                             order that way",
                });
            }
            if reduce.init == ReduceInit::FirstElement {
                return Err(TensorError::NotLowerable {
                    node: here,
                    reason: "a scatter cannot use ReduceInit::FirstElement: which source element \
                             is \"first\" at a colliding destination depends on iteration order, \
                             not a well-defined identity",
                });
            }
            return scatter_output_shape(here, &reduce.out_map, &iter_extents);
        }

        match reduce.keep {
            Keep::Scan => Ok(iter_extents),
            Keep::Reduce => project_output_shape(here, &reduce.out_map, &iter_extents),
        }
    }

    fn unify_iteration_space(
        &self,
        here: NodeId,
        iter_rank: u16,
        operands: &[(NodeId, &IndexMap)],
    ) -> Result<Vec<u64>, TensorError> {
        let mut resolved: Vec<Option<u64>> = vec![None; iter_rank as usize];
        let refs = flatten_operand_maps(operands);
        let shapes = self.shapes.borrow();

        for entry in &refs {
            let operand_shape = shapes.of(entry.node);
            for (axis_index, axis) in entry.pattern.axes.iter().enumerate() {
                if entry.skip_axis == Some(axis_index as u16) {
                    continue;
                }
                if let [term] = axis.terms.as_slice()
                    && term.coeff == 1
                    && axis.offset == 0
                {
                    let extent = operand_shape[axis_index];
                    let slot = &mut resolved[term.axis as usize];
                    match *slot {
                        None => *slot = Some(extent),
                        Some(existing) if existing == extent => {}
                        Some(existing) => {
                            return Err(TensorError::ExtentMismatch {
                                node: here,
                                dim: term.axis,
                                left: existing,
                                right: extent,
                            });
                        }
                    }
                }
            }
        }

        let extents: Vec<u64> = resolved
            .into_iter()
            .enumerate()
            .map(|(axis, extent)| {
                extent.ok_or(TensorError::UnconstrainedDim {
                    node: here,
                    dim: axis as u16,
                })
            })
            .collect::<Result<_, _>>()?;

        for entry in &refs {
            let operand_shape = shapes.of(entry.node);
            for (axis_index, axis) in entry.pattern.axes.iter().enumerate() {
                if entry.skip_axis == Some(axis_index as u16) {
                    continue;
                }
                let is_pure_projection =
                    matches!(axis.terms.as_slice(), [term] if term.coeff == 1) && axis.offset == 0;
                if is_pure_projection {
                    continue;
                }
                bounds_check(
                    here,
                    axis_index as u16,
                    axis,
                    &extents,
                    operand_shape[axis_index],
                )?;
            }
        }

        Ok(extents)
    }

    /// A [`IndexMap::Computed`]'s `indices` must resolve to an integer
    /// [`DType`] — a float or bool cannot select a dimension.
    fn check_indices_dtype(&self, here: NodeId, map: &IndexMap) -> Result<(), TensorError> {
        if let IndexMap::Computed { indices, .. } = map {
            let dtype = self.dtypes.borrow()[indices.0 as usize];
            if !dtype.is_integer() {
                return Err(TensorError::NonIntegerIndices { node: here, dtype });
            }
        }
        Ok(())
    }

    /// A [`IndexMap::Computed`]'s `gathered_dim` extent, read off `operand`'s
    /// already-resolved shape, must fit in [`GATHER_EXTENT_EXACT_FLOAT_LIMIT`]
    /// — see that constant's docs for why. `check_map` has already confirmed
    /// `gathered_dim` is in range for `operand`'s rank before this runs.
    fn check_gather_extent(
        &self,
        here: NodeId,
        operand: NodeId,
        map: &IndexMap,
    ) -> Result<(), TensorError> {
        if let IndexMap::Computed { gathered_dim, .. } = map {
            let extent = self.shapes.borrow().of(operand)[*gathered_dim as usize];
            if extent > GATHER_EXTENT_EXACT_FLOAT_LIMIT {
                return Err(TensorError::GatherExtentExceedsExactFloat { node: here, extent });
            }
        }
        Ok(())
    }
}

/// `In = Op`, `Out = (Op, Shapes)`: an expression is fully judged the
/// moment [`ShapeTable::push`] resolves it, and the same `Op` is handed
/// back unchanged, paired with a snapshot of every shape known so far. The
/// pair (not `Out = ()` or a bare resolved shape) is what lets a downstream
/// stage ([`crate::bind::BoundOpBuilder`]'s own `Pipe` impl) compose with
/// `.and_then`: op building needs both the `Op` just judged *and* the
/// accumulated [`Shapes`] to build its [`crate::bind::BoundOp`], and `AndThen`
/// requires `Second::In = First::Out`, so both travel together in this one
/// `Out` rather than one of them being reconstructed by a shared handle on
/// the other side.
impl Pipe for ShapeTable {
    type In = Op;
    type Out = (Op, Shapes);
    type Err = TensorError;

    fn call(&self, input: Op) -> impl Future<Output = Result<(Op, Shapes), TensorError>> {
        async move {
            self.push(&input)?;
            let shapes = self.shapes();
            Ok((input, shapes))
        }
    }
}

/// One [`IndexPattern`] reference flattened out of an [`IndexMap`]: `Affine`
/// contributes one, `Computed` contributes two — `index_map` (addressing the
/// `indices` tensor) and `base` (addressing the operand, with `skip_axis`
/// marking the axis the fetch supplies instead of iteration). Unifying and
/// bounds-checking both `Affine` and `Computed` operands is then one loop
/// over this flat list rather than two special cases.
struct MapRef<'a> {
    node: NodeId,
    pattern: &'a IndexPattern,
    skip_axis: Option<u16>,
}

fn flatten_operand_maps<'a>(operands: &[(NodeId, &'a IndexMap)]) -> Vec<MapRef<'a>> {
    let mut refs = Vec::with_capacity(operands.len());
    for (node, map) in operands {
        match map {
            IndexMap::Affine(pattern) => refs.push(MapRef {
                node: *node,
                pattern,
                skip_axis: None,
            }),
            IndexMap::Computed {
                indices,
                index_map,
                base,
                gathered_dim,
            } => {
                refs.push(MapRef {
                    node: *indices,
                    pattern: index_map,
                    skip_axis: None,
                });
                refs.push(MapRef {
                    node: *node,
                    pattern: base,
                    skip_axis: Some(*gathered_dim),
                });
            }
        }
    }
    refs
}

/// The widest iteration rank a map touches: `Affine` only ever names one
/// [`IndexPattern`], but `Computed` names two (`index_map` and `base`), both
/// drawn from the same iteration space, so both must be considered when an
/// iteration space's rank is derived from its operands.
fn combined_iter_rank(map: &IndexMap) -> u16 {
    match map {
        IndexMap::Affine(pattern) => pattern.iter_rank,
        IndexMap::Computed {
            index_map, base, ..
        } => index_map.iter_rank.max(base.iter_rank),
    }
}

fn check_backward(here: NodeId, referenced: NodeId) -> Result<(), TensorError> {
    if referenced.0 == here.0 {
        return Err(TensorError::Cycle(here));
    }
    if referenced.0 > here.0 {
        return Err(TensorError::NodeOutOfRange(here, referenced));
    }
    Ok(())
}

fn check_map(here: NodeId, map: &IndexMap) -> Result<(), TensorError> {
    match map {
        IndexMap::Affine(pattern) => check_pattern_axes(here, pattern),
        IndexMap::Computed {
            indices,
            index_map,
            base,
            gathered_dim,
        } => {
            check_backward(here, *indices)?;
            check_pattern_axes(here, index_map)?;
            check_pattern_axes(here, base)?;
            if *gathered_dim as usize >= base.axes.len() {
                return Err(TensorError::GatheredDimOutOfRange {
                    node: here,
                    dim: *gathered_dim,
                });
            }
            Ok(())
        }
    }
}

fn check_pattern_axes(here: NodeId, pattern: &IndexPattern) -> Result<(), TensorError> {
    for axis in &pattern.axes {
        for term in &axis.terms {
            if term.axis >= pattern.iter_rank {
                return Err(TensorError::IterDimOutOfRange {
                    node: here,
                    dim: term.axis,
                });
            }
        }
    }
    Ok(())
}

fn bounds_check(
    here: NodeId,
    axis_index: u16,
    axis: &AxisIndex,
    extents: &[u64],
    operand_extent: u64,
) -> Result<(), TensorError> {
    let mut max_index = i64::from(axis.offset);
    let mut min_index = i64::from(axis.offset);
    for term in &axis.terms {
        let extent = extents[term.axis as usize];
        let max_step = extent.saturating_sub(1) as i64;
        let coeff = i64::from(term.coeff);
        if coeff >= 0 {
            max_index += coeff * max_step;
        } else {
            min_index += coeff * max_step;
        }
    }
    if min_index < 0 || max_index >= operand_extent as i64 {
        return Err(TensorError::IndexOutOfBounds {
            node: here,
            dim: axis_index,
        });
    }
    Ok(())
}

fn project_output_shape(
    here: NodeId,
    out_map: &IndexMap,
    iter_extents: &[u64],
) -> Result<Vec<u64>, TensorError> {
    out_map
        .affine()
        .axes
        .iter()
        .map(|axis| match axis.terms.as_slice() {
            [term] if term.coeff == 1 => Ok(iter_extents[term.axis as usize]),
            _ => Err(TensorError::NotLowerable {
                node: here,
                reason: "reduce output maps must be pure projections in v1",
            }),
        })
        .collect()
}

/// A scatter's output shape: every axis but `gathered_dim` is a pure
/// projection onto an already-resolved iteration axis, exactly like
/// [`project_output_shape`]'s ordinary (affine) case; `gathered_dim` itself
/// reads [`IndexMap::scatter_extent`] instead of `iter_extents`, since that
/// axis's *position* is data-dependent while its *width* is a static number
/// the caller supplied — the whole reason this is shape-inferable at all
/// (`map.rs`'s `IndexMap::Computed` doc has the full convention).
fn scatter_output_shape(
    here: NodeId,
    out_map: &IndexMap,
    iter_extents: &[u64],
) -> Result<Vec<u64>, TensorError> {
    let IndexMap::Computed {
        base, gathered_dim, ..
    } = out_map
    else {
        return Err(TensorError::NotLowerable {
            node: here,
            reason: "scatter_output_shape called on a non-data-dependent out_map",
        });
    };
    let extent = out_map.scatter_extent().ok_or(TensorError::NotLowerable {
        node: here,
        reason: "a scatter out_map's gathered_dim is out of range for its own base pattern",
    })?;
    if extent < 0 {
        return Err(TensorError::NotLowerable {
            node: here,
            reason: "a scatter's destination extent must be non-negative",
        });
    }
    let extent = extent as u64;
    if extent > GATHER_EXTENT_EXACT_FLOAT_LIMIT {
        return Err(TensorError::GatherExtentExceedsExactFloat { node: here, extent });
    }

    base.axes
        .iter()
        .enumerate()
        .map(|(axis_index, axis)| {
            if axis_index as u16 == *gathered_dim {
                return Ok(extent);
            }
            match axis.terms.as_slice() {
                [term] if term.coeff == 1 && axis.offset == 0 => {
                    Ok(iter_extents[term.axis as usize])
                }
                _ => Err(TensorError::NotLowerable {
                    node: here,
                    reason: "a scatter output map's non-scattered axes must be pure projections",
                }),
            }
        })
        .collect()
}

fn resolve_leaf_shape(
    shape: &[crate::op::Extent],
    symbols: &[u64],
) -> Result<Vec<u64>, TensorError> {
    shape
        .iter()
        .map(|extent| match extent {
            crate::op::Extent::Static(size) => Ok(u64::from(*size)),
            crate::op::Extent::Symbolic(symbol) => symbols
                .get(*symbol as usize)
                .copied()
                .ok_or(TensorError::UnboundSymbol { symbol: *symbol }),
        })
        .collect()
}

/// The full iteration-space extents a [`Reduce`] walks, re-derived from an
/// already-resolved [`Shapes`].
///
/// [`ShapeTable`] discards this once it has projected a reduce's *output*
/// shape (smaller than the iteration space for `Keep::Reduce`), but
/// [`bind::bind`](crate::bind::bind) needs the full space back to size its loop.
/// Re-deriving is a handful of lines over data [`ShapeTable`] already proved
/// valid, versus a second parallel store on [`Shapes`] whose only consumer
/// is op building.
#[must_use]
pub(crate) fn fold_iteration_extents(reduce: &Reduce, shapes: &Shapes) -> Vec<u64> {
    let pattern = reduce.in_map.affine();
    let mut resolved = vec![0u64; pattern.iter_rank as usize];
    let operand_shape = shapes.of(reduce.operand);
    for (axis_index, axis) in pattern.axes.iter().enumerate() {
        if let [term] = axis.terms.as_slice()
            && term.coeff == 1
        {
            resolved[term.axis as usize] = operand_shape[axis_index];
        }
    }
    resolved
}

/// Batch driver: `new` / `push` each expression / `finish`, over the whole
/// program at once.
pub fn infer(program: &[Op], symbols: &[u64]) -> Result<Shapes, TensorError> {
    let inference = ShapeTable::new(symbols);
    for expr in program {
        inference.push(expr)?;
    }
    Ok(inference.finish())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::map::{self, AxisTerm};
    use crate::op::{Extent, ReduceInit, ScalarOp, append};

    fn leaf(program: &mut Vec<Op>, shape: &[Extent]) -> NodeId {
        append(
            program,
            Op::Input {
                dtype: DType::Float32,
                shape: shape.to_vec(),
                name: None,
            },
        )
    }

    fn matmul_program() -> (Vec<Op>, NodeId, NodeId) {
        let mut program = Vec::new();
        let lhs = leaf(&mut program, &[Extent::Symbolic(0), Extent::Static(768)]);
        let rhs = leaf(&mut program, &[Extent::Static(768), Extent::Static(3072)]);

        let lhs_map = IndexMap::Affine(map::projection(3, &[0, 2]));
        let rhs_map = IndexMap::Affine(map::projection(3, &[2, 1]));
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(lhs, lhs_map), (rhs, rhs_map)],
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
        (program, product, sum)
    }

    #[test]
    fn matmul_shape_resolves_a_symbolic_sequence_length() {
        let (program, _product, sum) = matmul_program();
        let shapes = infer(&program, &[512]).expect("matmul infers");
        assert_eq!(shapes.of(sum), &[512, 3072]);
    }

    #[test]
    fn a_broadcast_bias_add_resolves_the_wider_shape() {
        let mut program = Vec::new();
        let matrix = leaf(&mut program, &[Extent::Static(4), Extent::Static(8)]);
        let bias = leaf(&mut program, &[Extent::Static(8)]);
        let matrix_map = IndexMap::Affine(map::projection(2, &[0, 1]));
        let bias_map = IndexMap::Affine(map::projection(2, &[1]));
        let sum = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(matrix, matrix_map), (bias, bias_map)],
                name: None,
            },
        );

        let shapes = infer(&program, &[]).expect("broadcast add infers");
        assert_eq!(shapes.of(sum), &[4, 8]);
    }

    #[test]
    fn transpose_reads_the_permuted_shape() {
        let mut program = Vec::new();
        let matrix = leaf(&mut program, &[Extent::Static(3), Extent::Static(5)]);
        let transposed_map = IndexMap::Affine(map::projection(2, &[1, 0]));
        let transposed = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(matrix, transposed_map)],
                name: None,
            },
        );

        let shapes = infer(&program, &[]).expect("transpose infers");
        assert_eq!(shapes.of(transposed), &[5, 3]);
    }

    #[test]
    fn a_conv_window_within_bounds_infers() {
        let mut program = Vec::new();
        let anchor = leaf(&mut program, &[Extent::Static(4), Extent::Static(2)]);
        let signal = leaf(&mut program, &[Extent::Static(8)]);
        let anchor_map = IndexMap::Affine(map::projection(2, &[0, 1]));
        // out[h, r] reads in[h*2 + r], h in 0..4, r in 0..2: max index 3*2+1=7 < 8.
        let window = IndexMap::Affine(map::affine(
            2,
            &[(&[AxisTerm::scaled(0, 2), AxisTerm::scaled(1, 1)], 0)],
        ));
        let touched = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(anchor, anchor_map), (signal, window)],
                name: None,
            },
        );

        let shapes = infer(&program, &[]).expect("windowed access within bounds infers");
        assert_eq!(shapes.of(touched), &[4, 2]);
    }

    #[test]
    fn a_conv_window_that_exceeds_bounds_is_rejected() {
        let mut program = Vec::new();
        let signal = leaf(&mut program, &[Extent::Static(8)]);
        let anchor = leaf(&mut program, &[Extent::Static(4), Extent::Static(2)]);
        let anchor_map = IndexMap::Affine(map::projection(2, &[0, 1]));
        // out[h] reads in[h*2 + r], h in 0..4, r in 0..2: max index 3*2+1=7,
        // but a +2 offset pushes the top read to 9, past the extent of 8.
        let window = IndexMap::Affine(map::affine(
            2,
            &[(&[AxisTerm::scaled(0, 2), AxisTerm::scaled(1, 1)], 2)],
        ));
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(anchor, anchor_map), (signal, window)],
                name: None,
            },
        );

        let error = infer(&program, &[]).expect_err("out-of-bounds window is rejected");
        assert!(
            matches!(error, TensorError::IndexOutOfBounds { .. }),
            "{error}"
        );
    }

    #[test]
    fn disagreeing_operand_extents_are_rejected() {
        let mut program = Vec::new();
        let left = leaf(&mut program, &[Extent::Static(4)]);
        let right = leaf(&mut program, &[Extent::Static(5)]);
        let left_map = IndexMap::Affine(map::projection(1, &[0]));
        let right_map = IndexMap::Affine(map::projection(1, &[0]));
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(left, left_map), (right, right_map)],
                name: None,
            },
        );

        let error = infer(&program, &[]).expect_err("4 vs 5 disagree");
        assert!(
            matches!(error, TensorError::ExtentMismatch { .. }),
            "{error}"
        );
    }

    #[proxima::test]
    #[case::static_extent(Extent::Static(4), &[], &[4])]
    #[case::symbolic_extent(Extent::Symbolic(0), &[16], &[16])]
    async fn an_iota_resolves_its_own_shape_from_its_extent(
        #[case] extent: Extent,
        #[case] symbols: &[u64],
        #[case] expected: &[u64],
    ) {
        let mut program = Vec::new();
        let iota = append(
            &mut program,
            Op::Iota {
                dtype: DType::Float32,
                extent,
            },
        );

        let shapes = infer(&program, symbols).expect("an iota leaf infers");
        assert_eq!(shapes.of(iota), expected);
    }

    #[test]
    fn an_iota_over_an_unbound_symbol_is_rejected() {
        let mut program = Vec::new();
        append(
            &mut program,
            Op::Iota {
                dtype: DType::Float32,
                extent: Extent::Symbolic(0),
            },
        );

        let error = infer(&program, &[]).expect_err("no symbols supplied");
        assert!(
            matches!(error, TensorError::UnboundSymbol { symbol: 0 }),
            "{error}"
        );
    }

    /// `Iota` broadcasts into a higher-rank consumer through the same
    /// [`IndexMap`] machinery any other 1-D leaf uses — the mechanism
    /// `causal_attention.toml` relies on to spread `query_index`/`key_index`
    /// across the `st` iteration space.
    #[test]
    fn an_iota_broadcasts_into_a_higher_rank_consumer() {
        let mut program = Vec::new();
        let index = append(
            &mut program,
            Op::Iota {
                dtype: DType::Float32,
                extent: Extent::Static(4),
            },
        );
        let other = leaf(&mut program, &[Extent::Static(4), Extent::Static(3)]);
        let broadcast = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![
                    (index, IndexMap::Affine(map::projection(2, &[0]))),
                    (other, IndexMap::Affine(map::projection(2, &[0, 1]))),
                ],
                name: None,
            },
        );

        let shapes = infer(&program, &[]).expect("iota broadcast infers");
        assert_eq!(shapes.of(broadcast), &[4, 3]);
    }

    #[test]
    fn an_unbound_symbol_is_rejected() {
        let mut program = Vec::new();
        leaf(&mut program, &[Extent::Symbolic(0)]);

        let error = infer(&program, &[]).expect_err("no symbols supplied");
        assert!(
            matches!(error, TensorError::UnboundSymbol { symbol: 0 }),
            "{error}"
        );
    }

    #[test]
    fn an_unconstrained_iteration_dim_is_rejected() {
        let mut program = Vec::new();
        let source = leaf(&mut program, &[Extent::Static(4)]);
        // iter_rank 2, but only axis 0 is ever addressed by a pure projection.
        let map = IndexMap::Affine(map::projection(2, &[0]));
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(source, map)],
                name: None,
            },
        );

        let error = infer(&program, &[]).expect_err("axis 1 is never constrained");
        assert!(
            matches!(error, TensorError::UnconstrainedDim { .. }),
            "{error}"
        );
    }

    #[test]
    fn scan_output_shape_equals_the_input_shape() {
        let mut program = Vec::new();
        let source = leaf(&mut program, &[Extent::Static(16)]);
        let in_map = IndexMap::Affine(map::projection(1, &[0]));
        let out_map = IndexMap::Affine(map::projection(1, &[0]));
        let scanned = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map,
                out_map,
                keep: Keep::Scan,
                name: None,
            }),
        );

        let shapes = infer(&program, &[]).expect("cumsum shape infers");
        assert_eq!(shapes.of(scanned), &[16]);
    }

    #[test]
    fn reduce_output_shape_drops_the_contracted_dim() {
        let mut program = Vec::new();
        let source = leaf(&mut program, &[Extent::Static(4), Extent::Static(16)]);
        let in_map = IndexMap::Affine(map::projection(2, &[0, 1]));
        let out_map = IndexMap::Affine(map::projection(2, &[0]));
        let reduced = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map,
                out_map,
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let shapes = infer(&program, &[]).expect("reduce shape infers");
        assert_eq!(shapes.of(reduced), &[4]);
    }

    #[proxima::test]
    #[case::forward_reference(NodeId(2))]
    #[case::self_reference(NodeId(1))]
    async fn a_non_backward_reference_is_rejected(#[case] referenced: NodeId) {
        let mut program = Vec::new();
        leaf(&mut program, &[Extent::Static(4)]);
        let map = IndexMap::Affine(map::projection(1, &[0]));
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(referenced, map)],
                name: None,
            },
        );

        let error = infer(&program, &[]).expect_err("cannot reference a node not yet built");
        assert!(
            matches!(
                error,
                TensorError::NodeOutOfRange(_, _) | TensorError::Cycle(_)
            ),
            "{error}"
        );
    }

    #[test]
    fn a_bad_expr_is_rejected_at_push_time_not_at_finish() {
        let mut program = Vec::new();
        leaf(&mut program, &[Extent::Static(4)]);
        // an empty elementwise expression is malformed the instant it is pushed.
        let bad = Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: Vec::new(),
            name: None,
        };

        let inference = ShapeTable::new(&[]);
        inference.push(&program[0]).expect("leaf pushes cleanly");
        let error = inference
            .push(&bad)
            .expect_err("empty elementwise expression rejected at push");
        assert!(matches!(error, TensorError::EmptyElementwise(_)), "{error}");
    }

    #[test]
    fn streaming_and_batch_inference_agree() {
        let (program, _product, sum) = matmul_program();

        let batch = infer(&program, &[512]).expect("batch infers");

        let streamed = ShapeTable::new(&[512]);
        for expr in &program {
            streamed.push(expr).expect("streamed push succeeds");
        }
        let streamed = streamed.finish();

        assert_eq!(batch.of(sum), streamed.of(sum));
    }

    #[test]
    fn summing_narrow_integers_in_place_is_rejected() {
        let mut program = Vec::new();
        let quantized = append(
            &mut program,
            Op::Input {
                dtype: DType::Int8,
                shape: alloc::vec![Extent::Static(64)],
                name: None,
            },
        );
        let in_map = IndexMap::Affine(map::projection(1, &[0]));
        let out_map = IndexMap::Affine(map::projection(1, &[]));
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Int8,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: quantized,
                in_map,
                out_map,
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let error = infer(&program, &[]).expect_err("i8 accumulator overflows");
        assert!(
            matches!(error, TensorError::NarrowAccumulator { .. }),
            "{error}"
        );
    }

    /// `table[ids[s], d]` over iteration space `(s, d)`: `ids` selects
    /// `table`'s axis 0 (vocab), `d` is a plain projection onto `table`'s
    /// axis 1 (feature). The worked example `map.rs` and this module's own
    /// docs both describe.
    fn embedding_lookup_program(vocab: u32, dim: u32, seq: u32) -> (Vec<Op>, NodeId, NodeId) {
        let mut program = Vec::new();
        let table = leaf(&mut program, &[Extent::Static(vocab), Extent::Static(dim)]);
        let ids = append(
            &mut program,
            Op::Input {
                dtype: DType::Int32,
                shape: alloc::vec![Extent::Static(seq)],
                name: None,
            },
        );
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
        (program, ids, gathered)
    }

    /// A minimal, fully-gathered rank-1 map: `table[ids[s]]`, iteration
    /// space is just `s`. Used by the sad-path tests below where the
    /// embedding-style 2D shape would only add noise.
    fn fully_gathered_map(indices: NodeId) -> IndexMap {
        IndexMap::Computed {
            indices,
            index_map: map::projection(1, &[0]),
            base: map::IndexPattern {
                iter_rank: 1,
                axes: alloc::vec![map::AxisIndex::default()],
            },
            gathered_dim: 0,
        }
    }

    #[test]
    fn an_embedding_lookup_infers_without_the_vocab_extent_leaking_into_iteration() {
        let (program, _ids, gathered) = embedding_lookup_program(50_000, 8, 4);
        let shapes = infer(&program, &[]).expect("embedding lookup infers");
        assert_eq!(
            shapes.of(gathered),
            &[4, 8],
            "seq x feature; vocab (50_000) never appears"
        );
    }

    #[test]
    fn a_gathered_extent_at_exactly_two_to_the_24_is_accepted() {
        let (program, _ids, gathered) = embedding_lookup_program(1 << 24, 2, 1);
        let shapes = infer(&program, &[]).expect("extent at the exact-float ceiling infers");
        assert_eq!(shapes.of(gathered), &[1, 2]);
    }

    #[test]
    fn a_gathered_extent_past_two_to_the_24_is_rejected() {
        let (program, _ids, _gathered) = embedding_lookup_program((1 << 24) + 1, 2, 1);
        let error = infer(&program, &[]).expect_err("extent past 2^24 is rejected");
        assert!(
            matches!(
                error,
                TensorError::GatherExtentExceedsExactFloat { extent, .. }
                    if extent == (1 << 24) + 1
            ),
            "{error}"
        );
    }

    #[test]
    fn non_integer_gather_indices_are_rejected() {
        let mut program = Vec::new();
        let table = leaf(&mut program, &[Extent::Static(4)]);
        let float_ids = leaf(&mut program, &[Extent::Static(3)]); // float32, not integer
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(table, fully_gathered_map(float_ids))],
                name: None,
            },
        );

        let error = infer(&program, &[]).expect_err("float32 indices are rejected");
        assert!(
            matches!(error, TensorError::NonIntegerIndices { .. }),
            "{error}"
        );
    }

    #[test]
    fn a_gathered_dim_out_of_range_for_the_operand_is_rejected() {
        let mut program = Vec::new();
        let table = leaf(&mut program, &[Extent::Static(4)]);
        let ids = append(
            &mut program,
            Op::Input {
                dtype: DType::Int32,
                shape: alloc::vec![Extent::Static(3)],
                name: None,
            },
        );
        let mut map = fully_gathered_map(ids);
        if let IndexMap::Computed { gathered_dim, .. } = &mut map {
            *gathered_dim = 5;
        }
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(table, map)],
                name: None,
            },
        );

        let error = infer(&program, &[]).expect_err("gathered_dim 5 exceeds the operand's rank");
        assert!(
            matches!(error, TensorError::GatheredDimOutOfRange { .. }),
            "{error}"
        );
    }

    /// Builds a rank-1 `s in 0..4 -> out[ids[s]]` scatter program: `source`
    /// and `ids` both extent 4, `ids` dtype `Int32`, destination extent
    /// `dest_extent`. The shared fixture behind the worked example in
    /// [`a_scatter_out_map_infers_the_hand_worked_destination_shape`] and
    /// every other scatter shape test below.
    fn scatter_program(dest_extent: u32) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let source = leaf(&mut program, &[Extent::Static(4)]);
        let ids = append(
            &mut program,
            Op::Input {
                dtype: DType::Int32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
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
        (program, scattered)
    }

    /// The hand-worked example this task's own algorithm-development
    /// discipline requires: `src=[10,20,30,40]`, `idx=[2,0,2,1]`, a
    /// destination narrower than the source (extent 3, not 4) -- proof the
    /// destination shape is genuinely independent of the source's, not
    /// silently reusing it. Values are checked in `cpu.rs`'s own worked-
    /// example test; this one is shape alone.
    #[test]
    fn a_scatter_out_map_infers_the_hand_worked_destination_shape() {
        let (program, scattered) = scatter_program(3);
        let shapes = infer(&program, &[]).expect("a well-formed scatter infers");
        assert_eq!(shapes.of(scattered), &[3]);
    }

    #[test]
    fn a_scatter_as_a_keep_scan_is_rejected() {
        let (mut program, _) = scatter_program(3);
        let last = program
            .pop()
            .expect("scatter_program pushes at least one reduce");
        let Op::Reduce(mut reduce) = last else {
            panic!("scatter_program's last op is a Reduce");
        };
        reduce.keep = Keep::Scan;
        append(&mut program, Op::Reduce(reduce));

        let error = infer(&program, &[]).expect_err("a scatter scan has no defined step order");
        assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
    }

    #[test]
    fn a_scatter_with_first_element_init_is_rejected() {
        let (mut program, _) = scatter_program(3);
        let last = program
            .pop()
            .expect("scatter_program pushes at least one reduce");
        let Op::Reduce(mut reduce) = last else {
            panic!("scatter_program's last op is a Reduce");
        };
        reduce.init = ReduceInit::FirstElement;
        append(&mut program, Op::Reduce(reduce));

        let error = infer(&program, &[])
            .expect_err("which source is \"first\" at a collision is undefined");
        assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
    }

    #[test]
    fn a_scatter_destination_extent_past_two_to_the_24_is_rejected() {
        let (program, _) = scatter_program((1 << 24) + 1);
        let error = infer(&program, &[]).expect_err("extent past 2^24 is rejected");
        assert!(
            matches!(
                error,
                TensorError::GatherExtentExceedsExactFloat { extent, .. }
                    if extent == (1 << 24) + 1
            ),
            "{error}"
        );
    }

    #[test]
    fn infer_as_a_pipe_matches_the_free_function() {
        let (program, _product, sum) = matmul_program();

        let inference = ShapeTable::new(&[512]);
        let mut last_shapes = None;
        for expr in &program {
            let (echoed, shapes) =
                proxima_primitives::block_on(Pipe::call(&inference, expr.clone()))
                    .expect("infer pipe step succeeds");
            assert_eq!(&echoed, expr, "the observe form hands the same Op back");
            last_shapes = Some(shapes);
        }

        let via_pipe = last_shapes.expect("matmul program is non-empty");
        let via_free_function = infer(&program, &[512]).expect("free-function infer succeeds");
        assert_eq!(via_pipe.of(sum), via_free_function.of(sum));
    }

    /// The literal complaint: a fused checkpoint tensor's on-disk width
    /// (`12`, standing in for a real `[2048, 6144]` fused QKV/BCx weight)
    /// cannot be narrowed by an offset alone, because a pure single-term
    /// axis's extent used to come from the *sliced operand's own shape*
    /// regardless of its offset (`unify_iteration_space`, before this row).
    /// `donor` supplies the true width (`4`) through a plain 0-offset
    /// projection onto the same iteration axis -- the same
    /// externally-supplied-extent mechanism `Op::Iota`'s own `extent` field
    /// and `Op::Input`'s own `shape` already use -- so `fused`'s axis-1 term
    /// (`coeff == 1`, `offset == 8`) is no longer treated as extent-defining
    /// and instead falls through to `bounds_check`, which already handles a
    /// nonzero offset correctly (it is the same arithmetic a convolution
    /// window's multi-term axis already exercises).
    #[test]
    fn a_nonzero_offset_slice_of_a_wider_operand_narrows_via_a_companion_donor() {
        let mut program = Vec::new();
        let fused = leaf(&mut program, &[Extent::Static(2), Extent::Static(12)]);
        let donor = leaf(&mut program, &[Extent::Static(2), Extent::Static(4)]);

        let fused_map = IndexMap::Affine(map::affine(
            2,
            &[
                (&[AxisTerm::projection(0)], 0),
                (&[AxisTerm::scaled(1, 1)], 8),
            ],
        ));
        let donor_map = IndexMap::Affine(map::projection(2, &[0, 1]));
        let sliced = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(fused, fused_map), (donor, donor_map)],
                name: None,
            },
        );

        let shapes = infer(&program, &[]).expect("an offset slice with a donor infers");
        assert_eq!(
            shapes.of(sliced),
            &[2, 4],
            "narrowed to the donor's width (4), not fused's on-disk width (12)"
        );
    }

    /// Perturbing the fix under test: reverting `unify_iteration_space` to
    /// ignore `axis.offset` (the pre-fix predicate) makes this test fail
    /// with `ExtentMismatch { dim: 1, left: 12, right: 4 }` -- the exact
    /// shape of the fused-checkpoint failure this row closes. Left in this
    /// comment rather than as a `#[should_panic]`, since the proof this row
    /// reports is the before/after `cargo nextest` transcript, not a second
    /// mechanism to keep green.
    #[test]
    fn a_nonzero_offset_slice_out_of_the_donors_bounds_is_still_rejected() {
        let mut program = Vec::new();
        let fused = leaf(&mut program, &[Extent::Static(2), Extent::Static(12)]);
        let donor = leaf(&mut program, &[Extent::Static(2), Extent::Static(4)]);

        // offset 9 + (donor extent 4 - 1) = 12, the first index past fused's
        // on-disk width of 12.
        let fused_map = IndexMap::Affine(map::affine(
            2,
            &[
                (&[AxisTerm::projection(0)], 0),
                (&[AxisTerm::scaled(1, 1)], 9),
            ],
        ));
        let donor_map = IndexMap::Affine(map::projection(2, &[0, 1]));
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(fused, fused_map), (donor, donor_map)],
                name: None,
            },
        );

        let error =
            infer(&program, &[]).expect_err("a slice window past the buffer end is rejected");
        assert!(
            matches!(error, TensorError::IndexOutOfBounds { .. }),
            "{error}"
        );
    }

    /// The residual limitation `unify_iteration_space` cannot lift from
    /// inside `shape.rs` alone: an `offset == 0` slice narrower than its
    /// operand is indistinguishable, from `AxisIndex` alone, from "read the
    /// whole axis" -- `AxisIndex` carries a start (`offset`) and a stride
    /// (`coeff`) but no length, so there is no bit anywhere in the map that
    /// says "stop at 4, not 12" when the window starts at the origin. The
    /// donor's independently-correct extent (4) collides with `fused`'s
    /// offset-0 axis reading its own full on-disk width (12) as an
    /// `ExtentMismatch`, not as a silently-narrowed shape -- ambiguous, not
    /// wrong, and rejected rather than guessed.
    #[test]
    fn an_offset_zero_slice_narrower_than_its_operand_is_still_ambiguous() {
        let mut program = Vec::new();
        let fused = leaf(&mut program, &[Extent::Static(2), Extent::Static(12)]);
        let donor = leaf(&mut program, &[Extent::Static(2), Extent::Static(4)]);

        let fused_map = IndexMap::Affine(map::projection(2, &[0, 1]));
        let donor_map = IndexMap::Affine(map::projection(2, &[0, 1]));
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(fused, fused_map), (donor, donor_map)],
                name: None,
            },
        );

        let error = infer(&program, &[]).expect_err(
            "an offset-0 slice narrower than its operand cannot be told apart from a full read",
        );
        assert!(
            matches!(error, TensorError::ExtentMismatch { .. }),
            "{error}"
        );
    }

    /// Row 131's own question -- "is this a precedence rule rather than a
    /// missing field?" -- tried directly against `unify_iteration_space`
    /// above, twice, each patch applied to this file, the suite run, then
    /// reverted; nothing below reflects either patch, only what the run
    /// showed:
    ///
    /// - *Narrower wins*: replace the `ExtentMismatch` arm (the `match *slot`
    ///   in `unify_iteration_space`, above) with
    ///   `Some(existing) => *slot = Some(existing.min(extent))`.
    ///   `disagreeing_operand_extents_are_rejected` FAILED:
    ///   `infer` returned `Ok(Shapes { extents: [[4], [5], [4]] })` -- a
    ///   genuine 4-vs-5 shape mismatch silently narrowed to 4.
    /// - *First operand wins*: ignore a later disagreement at an
    ///   already-resolved axis instead of erroring, and always run
    ///   `bounds_check` instead of skipping it for what looks like a pure
    ///   projection. Same test, same failure, same `Ok(Shapes { extents:
    ///   [[4], [5], [4]] })` -- because `bounds_check` only rejects an
    ///   operand *narrower* than the iteration space demands. Reading fewer
    ///   than all of a wider operand's elements is exactly what a legitimate
    ///   broadcast does everywhere else in this file, so `bounds_check`
    ///   cannot also be made to mean "these two operands were never
    ///   compatible."
    ///
    /// Both rules collapse two situations that must stay distinct -- "this
    /// narrower operand is the true donor" and "these two operands are
    /// simply the wrong sizes" -- into the same bit pattern: two 0-offset,
    /// `coeff == 1` pure projections onto the same axis that disagree.
    /// `unify_iteration_space` has no other information to tell them apart,
    /// so precedence is not the fix; the missing bit is real.
    ///
    /// That bit would need to live as a `len: Option<Extent>` on
    /// [`AxisIndex`] -- resolved the same way [`Op::Iota`]'s own `extent`
    /// field is, and, when present, always routed through `bounds_check`
    /// exactly like a nonzero offset already is, never treated as a pure
    /// projection. Its blast radius, by construction site
    /// (`grep -rn 'AxisIndex[[:space:]]*{'`, checked against this row's own
    /// worktree): 21 struct-literal sites across two crates need an added
    /// `len: None` to keep compiling. 16 sit in `proxima-tensor`: `map.rs`
    /// (x3) and this file (x1) are test-only and mine to change; `spec.rs`
    /// (x8), `cpu.rs` (x2), `benches/bench_vs_ggml.rs` (x1), and `bind.rs`
    /// (x1, inside `remap_pattern`'s operand-fusion rebuild of a remapped
    /// `AxisIndex`) are not. The other 5 sit in `omega` (`msl.rs` x1, two
    /// integration-test files x2 each), a separately-versioned downstream
    /// crate. `bind.rs`'s one site is the one this task's own scope forbids
    /// touching, and a Rust struct literal cannot omit a field
    /// conditionally: `cargo build --workspace --lib` would fail the moment
    /// `len` existed, independent of how many of the other 20 sites got
    /// fixed. The field is the right answer to the standalone,
    /// offset-0-alone case just above; landing it is not this session's to
    /// do.
    ///
    /// What follows is this row's acceptance criterion: a real fused-QKV
    /// checkpoint tensor, `[2048, 6144]`, read as three `[2048, 2048]`
    /// chunks at on-disk offsets 0, 2048, and 4096 -- values checked via
    /// [`crate::cpu::evaluate`], not shape alone.
    ///
    /// The three offsets are one iteration axis (`chunk`, extent 3) rather
    /// than three separately-offset ops, because that is the primitive this
    /// crate already has for exactly this shape of problem: `fused`'s wide
    /// axis reads as `chunk * 2048 + within`, a genuine two-term axis with
    /// real, non-zero coefficients -- the same mechanism
    /// `a_conv_window_within_bounds_infers` (above) already exercises for a
    /// convolution window, not a new one. Because the axis carries two
    /// terms, `unify_iteration_space` never treats it as a "pure
    /// projection" (that match only ever fires for a *single*, `coeff == 1`,
    /// 0-offset term), so it always goes through `bounds_check` and never
    /// competes with `donor` for `chunk`'s or `within`'s extent -- exactly
    /// where the standalone offset-0 case just above gets stuck, because
    /// there the axis genuinely is one term with nothing to keep it from
    /// looking like a full read. `donor` (a zero-valued [`Op::Constant`],
    /// exactly the "carries an extent a consumer cannot otherwise infer"
    /// role its own doc names) supplies `chunk`'s and `within`'s extents
    /// through a plain 0-offset projection, unambiguous because nothing
    /// else here offers a competing value for either axis.
    #[test]
    fn a_fused_qkv_split_evaluates_all_three_chunks_by_a_real_chunk_axis() {
        let rows: u32 = 2048;
        let fused_cols: u32 = 6144;
        let chunk_extent: u32 = 3;
        let chunk_width: u32 = 2048;

        let mut program = Vec::new();
        let fused = leaf(
            &mut program,
            &[Extent::Static(rows), Extent::Static(fused_cols)],
        );
        let donor = append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(chunk_extent), Extent::Static(chunk_width)],
                value: 0.0,
            },
        );

        let fused_map = IndexMap::Affine(map::affine(
            3,
            &[
                (&[AxisTerm::projection(0)], 0),
                (
                    &[
                        AxisTerm::scaled(1, chunk_width as i32),
                        AxisTerm::scaled(2, 1),
                    ],
                    0,
                ),
            ],
        ));
        let donor_map = IndexMap::Affine(map::projection(3, &[1, 2]));
        let split = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(fused, fused_map), (donor, donor_map)],
                name: None,
            },
        );

        let shapes = infer(&program, &[]).expect("the chunk-axis split infers");
        assert_eq!(
            shapes.of(split),
            &[
                u64::from(rows),
                u64::from(chunk_extent),
                u64::from(chunk_width)
            ],
            "row x chunk x within, not row x fused_cols"
        );

        let fused_data: Vec<f32> = (0..rows)
            .flat_map(|row| (0..fused_cols).map(move |col| (row * fused_cols + col) as f32))
            .collect();
        let evaluated = crate::cpu::evaluate(&program, &[], &[&fused_data], &[split])
            .expect("the chunk-axis split evaluates");
        let (data, shape) = evaluated.get(split).expect("split is a requested output");
        assert_eq!(
            shape,
            &[
                u64::from(rows),
                u64::from(chunk_extent),
                u64::from(chunk_width)
            ]
        );

        let at = |row: u32, chunk: u32, within: u32| {
            data[((row * chunk_extent + chunk) * chunk_width + within) as usize]
        };
        assert_eq!(
            at(0, 0, 0),
            0.0,
            "chunk 0 (offset 0) reads fused's first column"
        );
        assert_eq!(
            at(0, 1, 0),
            2048.0,
            "chunk 1 (offset 2048) reads fused's column 2048"
        );
        assert_eq!(
            at(0, 2, 0),
            4096.0,
            "chunk 2 (offset 4096) reads fused's column 4096"
        );
        assert_eq!(
            at(2047, 2, 2047),
            (2047 * fused_cols + 4096 + 2047) as f32,
            "the last row, last chunk, last column reads fused's final element"
        );
    }

    /// The literal per-chunk framing -- three independently offset-addressed
    /// ops, matching
    /// `a_nonzero_offset_slice_of_a_wider_operand_narrows_via_a_companion_donor`
    /// above -- already resolves shapes correctly for a nonzero offset; this
    /// checks it also evaluates the right *values*, for both nonzero chunks
    /// a real fused-QKV split needs (offsets `2048` and `4096`). The
    /// `0`-offset chunk is the one this row's remaining gap blocks -- see
    /// the doc above.
    #[proxima::test]
    #[case::second_chunk_offset_2048(2048)]
    #[case::third_chunk_offset_4096(4096)]
    async fn a_nonzero_offset_slice_evaluates_the_correct_chunk_values(#[case] offset: i32) {
        let mut program = Vec::new();
        let fused = leaf(&mut program, &[Extent::Static(2), Extent::Static(6144)]);
        let donor = leaf(&mut program, &[Extent::Static(2), Extent::Static(2048)]);

        let fused_map = IndexMap::Affine(map::affine(
            2,
            &[
                (&[AxisTerm::projection(0)], 0),
                (&[AxisTerm::scaled(1, 1)], offset),
            ],
        ));
        let donor_map = IndexMap::Affine(map::projection(2, &[0, 1]));
        let sliced = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(fused, fused_map), (donor, donor_map)],
                name: None,
            },
        );

        let fused_data: Vec<f32> = (0..2 * 6144).map(|index| index as f32).collect();
        let donor_data = alloc::vec![0.0f32; 2 * 2048];
        let evaluated = crate::cpu::evaluate(&program, &[], &[&fused_data, &donor_data], &[sliced])
            .expect("a nonzero-offset slice with a donor evaluates");
        let (data, shape) = evaluated.get(sliced).expect("sliced is a requested output");

        assert_eq!(shape, &[2, 2048]);
        let start = offset as usize;
        assert_eq!(data[0], fused_data[start], "row 0, column 0 of the chunk");
        assert_eq!(
            data[2047],
            fused_data[start + 2047],
            "row 0, last column of the chunk"
        );
        assert_eq!(
            data[2048],
            fused_data[6144 + start],
            "row 1, column 0 of the chunk"
        );
    }
}
