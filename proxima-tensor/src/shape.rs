//! Shape inference — and, since it must touch every reference to do it, the
//! only structural validation a program gets.
//!
//! A tensor program is not validated on construction the way the old arena
//! was: [`Expr`] is just data. [`Infer`] is where "well-formed" is actually
//! checked — backwards references, zip arity, the accumulator-widening rule,
//! iteration-dim range, and the affine unification that resolves every
//! extent. An `Expr` is fully judged the moment it is pushed.
//!
//! [`Infer`] is also this crate's sans-IO stance made concrete: a tensor
//! program is something that can arrive a step at a time (a partition
//! crossing a wire is a stream of `Expr`s; compiling overlaps transport), and
//! `Infer` is the core that judges each step against everything before it,
//! with no I/O of its own. [`infer`] is the batch case, three lines over the
//! stream; `Infer` also implements
//! [`Pipe`] directly (`In = Expr`,
//! `Out = (Expr, Shapes)`) — the same core, not a second type wrapping it.
//! [`Pipe::call`] takes `&self`, while judging a node is inherently a
//! mutation, so `dtypes`/`shapes` below are `RefCell`s: the interior-
//! mutability idiom `proxima_primitives::pipe::isolate`'s module doc names
//! for runtime-owned `!Send` pipe state, applied to the state that was
//! already here rather than to a wrapper around it.

use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::future::Future;

use proxima_primitives::pipe::Pipe;

use crate::dtype::DType;
use crate::error::TensorError;
use crate::expr::{Expr, Fold, Keep, NodeId, ScalarOp};
use crate::map::{AffineMap, DimExpr, IndexMap};

/// The largest integer an f32 can represent exactly — its 24-bit mantissa's
/// width. Gather indices ride in f32 buffers (see
/// [`IndexMap::Computed`](crate::map::IndexMap::Computed) and `cpu.rs`'s
/// module docs), so a gathered dim wider than this could silently address
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
    /// Panics if `node` was never pushed to the [`Infer`] that built this
    /// `Shapes` — every valid `NodeId` a program can produce was, by the
    /// backwards-reference rule, resolved before anything that could ask for
    /// it.
    #[must_use]
    pub fn of(&self, node: NodeId) -> &[u64] {
        &self.extents[node.0 as usize]
    }

    fn push(&mut self, resolved: Vec<u64>) {
        self.extents.push(resolved);
    }
}

/// The prefix state of shape inference: every node judged so far.
pub struct Infer {
    symbols: Vec<u64>,
    dtypes: RefCell<Vec<DType>>,
    shapes: RefCell<Shapes>,
}

impl Infer {
    #[must_use]
    pub fn new(symbols: &[u64]) -> Self {
        Self {
            symbols: symbols.to_vec(),
            dtypes: RefCell::new(Vec::new()),
            shapes: RefCell::new(Shapes::default()),
        }
    }

    /// Judge one expression against everything pushed before it.
    pub fn push(&self, expr: &Expr) -> Result<(), TensorError> {
        let node = NodeId(self.shapes.borrow().extents.len() as u32);
        let resolved = match expr {
            Expr::Block { shape, .. } => resolve_block_shape(shape, &self.symbols)?,
            Expr::Zip { body, operands, .. } => self.infer_zip(node, *body, operands)?,
            Expr::Fold(fold) => self.infer_fold(node, fold)?,
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

    fn infer_zip(
        &self,
        here: NodeId,
        body: ScalarOp,
        operands: &[(NodeId, IndexMap)],
    ) -> Result<Vec<u64>, TensorError> {
        if operands.is_empty() {
            return Err(TensorError::EmptyZip(here));
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

    fn infer_fold(&self, here: NodeId, fold: &Fold) -> Result<Vec<u64>, TensorError> {
        check_backward(here, fold.operand)?;
        check_map(here, &fold.in_map)?;
        check_map(here, &fold.out_map)?;
        self.check_indices_dtype(here, &fold.in_map)?;
        self.check_gather_extent(here, fold.operand, &fold.in_map)?;

        if fold.body.is_associative() && !fold.dtype.accumulates_in_place() {
            return Err(TensorError::NarrowAccumulator {
                node: here,
                element: self.dtypes.borrow()[fold.operand.0 as usize],
                accumulator: fold.dtype,
            });
        }

        let iter_rank = combined_iter_rank(&fold.in_map);
        let iter_extents =
            self.unify_iteration_space(here, iter_rank, &[(fold.operand, &fold.in_map)])?;

        if fold.out_map.is_data_dependent() {
            return Err(TensorError::NotLowerable {
                node: here,
                reason: "scatter (a data-dependent fold output) is not shape-inferable in v1",
            });
        }

        match fold.keep {
            Keep::All => Ok(iter_extents),
            Keep::Last => project_output_shape(here, &fold.out_map, &iter_extents),
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
            for (dim_index, dim) in entry.map.dims.iter().enumerate() {
                if entry.skip_dim == Some(dim_index as u16) {
                    continue;
                }
                if let [term] = dim.terms.as_slice()
                    && term.coeff == 1
                {
                    let extent = operand_shape[dim_index];
                    let slot = &mut resolved[term.iter_dim as usize];
                    match *slot {
                        None => *slot = Some(extent),
                        Some(existing) if existing == extent => {}
                        Some(existing) => {
                            return Err(TensorError::ExtentMismatch {
                                node: here,
                                dim: term.iter_dim,
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
            .map(|(dim, extent)| {
                extent.ok_or(TensorError::UnconstrainedDim {
                    node: here,
                    dim: dim as u16,
                })
            })
            .collect::<Result<_, _>>()?;

        for entry in &refs {
            let operand_shape = shapes.of(entry.node);
            for (dim_index, dim) in entry.map.dims.iter().enumerate() {
                if entry.skip_dim == Some(dim_index as u16) {
                    continue;
                }
                let is_pure_projection = matches!(dim.terms.as_slice(), [term] if term.coeff == 1);
                if is_pure_projection {
                    continue;
                }
                bounds_check(
                    here,
                    dim_index as u16,
                    dim,
                    &extents,
                    operand_shape[dim_index],
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

/// `In = Out = Expr`, the observe form: an expression is fully judged the
/// moment [`Infer::push`] resolves it, and the same `Expr` is handed back
/// unchanged, paired with a snapshot of every shape known so far. The pair
/// (not `Out = ()` or a bare resolved shape) is what lets a downstream stage
/// ([`crate::nest::Lower`]'s own `Pipe` impl) compose with `.and_then`:
/// lowering needs both the `Expr` just judged *and* the accumulated
/// [`Shapes`] to build its `Nest`, and `AndThen` requires `Second::In =
/// First::Out`, so both travel together in this one `Out` rather than one of
/// them being reconstructed by a shared handle on the other side.
impl Pipe for Infer {
    type In = Expr;
    type Out = (Expr, Shapes);
    type Err = TensorError;

    fn call(&self, input: Expr) -> impl Future<Output = Result<(Expr, Shapes), TensorError>> {
        async move {
            self.push(&input)?;
            let shapes = self.shapes();
            Ok((input, shapes))
        }
    }
}

/// One [`AffineMap`] reference flattened out of an [`IndexMap`]: `Affine`
/// contributes one, `Computed` contributes two — `index_map` (addressing the
/// `indices` tensor) and `base` (addressing the operand, with `skip_dim`
/// marking the dim the fetch supplies instead of iteration). Unifying and
/// bounds-checking both `Affine` and `Computed` operands is then one loop
/// over this flat list rather than two special cases.
struct MapRef<'a> {
    node: NodeId,
    map: &'a AffineMap,
    skip_dim: Option<u16>,
}

fn flatten_operand_maps<'a>(operands: &[(NodeId, &'a IndexMap)]) -> Vec<MapRef<'a>> {
    let mut refs = Vec::with_capacity(operands.len());
    for (node, map) in operands {
        match map {
            IndexMap::Affine(affine) => refs.push(MapRef {
                node: *node,
                map: affine,
                skip_dim: None,
            }),
            IndexMap::Computed {
                indices,
                index_map,
                base,
                gathered_dim,
            } => {
                refs.push(MapRef {
                    node: *indices,
                    map: index_map,
                    skip_dim: None,
                });
                refs.push(MapRef {
                    node: *node,
                    map: base,
                    skip_dim: Some(*gathered_dim),
                });
            }
        }
    }
    refs
}

/// The widest iteration rank a map touches: `Affine` only ever names one
/// [`AffineMap`], but `Computed` names two (`index_map` and `base`), both
/// drawn from the same iteration space, so both must be considered when an
/// iteration space's rank is derived from its operands.
fn combined_iter_rank(map: &IndexMap) -> u16 {
    match map {
        IndexMap::Affine(affine) => affine.iter_rank,
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
        IndexMap::Affine(affine) => check_affine_dims(here, affine),
        IndexMap::Computed {
            indices,
            index_map,
            base,
            gathered_dim,
        } => {
            check_backward(here, *indices)?;
            check_affine_dims(here, index_map)?;
            check_affine_dims(here, base)?;
            if *gathered_dim as usize >= base.dims.len() {
                return Err(TensorError::GatheredDimOutOfRange {
                    node: here,
                    dim: *gathered_dim,
                });
            }
            Ok(())
        }
    }
}

fn check_affine_dims(here: NodeId, affine: &AffineMap) -> Result<(), TensorError> {
    for dim in &affine.dims {
        for term in &dim.terms {
            if term.iter_dim >= affine.iter_rank {
                return Err(TensorError::IterDimOutOfRange {
                    node: here,
                    dim: term.iter_dim,
                });
            }
        }
    }
    Ok(())
}

fn bounds_check(
    here: NodeId,
    dim_index: u16,
    dim: &DimExpr,
    extents: &[u64],
    operand_extent: u64,
) -> Result<(), TensorError> {
    let mut max_index = i64::from(dim.offset);
    let mut min_index = i64::from(dim.offset);
    for term in &dim.terms {
        let extent = extents[term.iter_dim as usize];
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
            dim: dim_index,
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
        .dims
        .iter()
        .map(|dim| match dim.terms.as_slice() {
            [term] if term.coeff == 1 => Ok(iter_extents[term.iter_dim as usize]),
            _ => Err(TensorError::NotLowerable {
                node: here,
                reason: "fold output maps must be pure projections in v1",
            }),
        })
        .collect()
}

fn resolve_block_shape(
    shape: &[crate::expr::Extent],
    symbols: &[u64],
) -> Result<Vec<u64>, TensorError> {
    shape
        .iter()
        .map(|extent| match extent {
            crate::expr::Extent::Static(size) => Ok(u64::from(*size)),
            crate::expr::Extent::Symbolic(symbol) => symbols
                .get(*symbol as usize)
                .copied()
                .ok_or(TensorError::UnboundSymbol { symbol: *symbol }),
        })
        .collect()
}

/// The full iteration-space extents a [`Fold`] walks, re-derived from an
/// already-resolved [`Shapes`].
///
/// [`Infer`] discards this once it has projected a fold's *output* shape
/// (smaller than the iteration space for `Keep::Last`), but
/// [`nest::lower`](crate::nest::lower) needs the full space back to size its
/// loop nest. Re-deriving is a handful of lines over data [`Infer`] already
/// proved valid, versus a second parallel store on [`Shapes`] whose only
/// consumer is lowering.
#[must_use]
pub(crate) fn fold_iteration_extents(fold: &Fold, shapes: &Shapes) -> Vec<u64> {
    let affine = fold.in_map.affine();
    let mut resolved = vec![0u64; affine.iter_rank as usize];
    let operand_shape = shapes.of(fold.operand);
    for (dim_index, dim) in affine.dims.iter().enumerate() {
        if let [term] = dim.terms.as_slice()
            && term.coeff == 1
        {
            resolved[term.iter_dim as usize] = operand_shape[dim_index];
        }
    }
    resolved
}

/// Batch driver: `new` / `push` each expression / `finish`, over the whole
/// program at once.
pub fn infer(program: &[Expr], symbols: &[u64]) -> Result<Shapes, TensorError> {
    let inference = Infer::new(symbols);
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
    use crate::expr::{Extent, FoldInit, ScalarOp, append};
    use crate::map::{self, AffineTerm};
    use rstest::rstest;

    fn block(program: &mut Vec<Expr>, shape: &[Extent]) -> NodeId {
        append(
            program,
            Expr::Block {
                dtype: DType::Float32,
                shape: shape.to_vec(),
                name: None,
            },
        )
    }

    fn matmul_program() -> (Vec<Expr>, NodeId, NodeId) {
        let mut program = Vec::new();
        let lhs = block(&mut program, &[Extent::Symbolic(0), Extent::Static(768)]);
        let rhs = block(&mut program, &[Extent::Static(768), Extent::Static(3072)]);

        let lhs_map = IndexMap::Affine(map::projection(3, &[0, 2]));
        let rhs_map = IndexMap::Affine(map::projection(3, &[2, 1]));
        let product = append(
            &mut program,
            Expr::Zip {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(lhs, lhs_map), (rhs, rhs_map)],
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
        let matrix = block(&mut program, &[Extent::Static(4), Extent::Static(8)]);
        let bias = block(&mut program, &[Extent::Static(8)]);
        let matrix_map = IndexMap::Affine(map::projection(2, &[0, 1]));
        let bias_map = IndexMap::Affine(map::projection(2, &[1]));
        let sum = append(
            &mut program,
            Expr::Zip {
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
        let matrix = block(&mut program, &[Extent::Static(3), Extent::Static(5)]);
        let transposed_map = IndexMap::Affine(map::projection(2, &[1, 0]));
        let transposed = append(
            &mut program,
            Expr::Zip {
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
        let anchor = block(&mut program, &[Extent::Static(4), Extent::Static(2)]);
        let signal = block(&mut program, &[Extent::Static(8)]);
        let anchor_map = IndexMap::Affine(map::projection(2, &[0, 1]));
        // out[h, r] reads in[h*2 + r], h in 0..4, r in 0..2: max index 3*2+1=7 < 8.
        let window = IndexMap::Affine(map::affine(
            2,
            &[(&[AffineTerm::scaled(0, 2), AffineTerm::scaled(1, 1)], 0)],
        ));
        let touched = append(
            &mut program,
            Expr::Zip {
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
        let signal = block(&mut program, &[Extent::Static(8)]);
        let anchor = block(&mut program, &[Extent::Static(4), Extent::Static(2)]);
        let anchor_map = IndexMap::Affine(map::projection(2, &[0, 1]));
        // out[h] reads in[h*2 + r], h in 0..4, r in 0..2: max index 3*2+1=7,
        // but a +2 offset pushes the top read to 9, past the extent of 8.
        let window = IndexMap::Affine(map::affine(
            2,
            &[(&[AffineTerm::scaled(0, 2), AffineTerm::scaled(1, 1)], 2)],
        ));
        append(
            &mut program,
            Expr::Zip {
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
        let left = block(&mut program, &[Extent::Static(4)]);
        let right = block(&mut program, &[Extent::Static(5)]);
        let left_map = IndexMap::Affine(map::projection(1, &[0]));
        let right_map = IndexMap::Affine(map::projection(1, &[0]));
        append(
            &mut program,
            Expr::Zip {
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

    #[test]
    fn an_unbound_symbol_is_rejected() {
        let mut program = Vec::new();
        block(&mut program, &[Extent::Symbolic(0)]);

        let error = infer(&program, &[]).expect_err("no symbols supplied");
        assert!(
            matches!(error, TensorError::UnboundSymbol { symbol: 0 }),
            "{error}"
        );
    }

    #[test]
    fn an_unconstrained_iteration_dim_is_rejected() {
        let mut program = Vec::new();
        let source = block(&mut program, &[Extent::Static(4)]);
        // iter_rank 2, but only dim 0 is ever addressed by a pure projection.
        let map = IndexMap::Affine(map::projection(2, &[0]));
        append(
            &mut program,
            Expr::Zip {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(source, map)],
                name: None,
            },
        );

        let error = infer(&program, &[]).expect_err("dim 1 is never constrained");
        assert!(
            matches!(error, TensorError::UnconstrainedDim { .. }),
            "{error}"
        );
    }

    #[test]
    fn scan_output_shape_equals_the_input_shape() {
        let mut program = Vec::new();
        let source = block(&mut program, &[Extent::Static(16)]);
        let in_map = IndexMap::Affine(map::projection(1, &[0]));
        let out_map = IndexMap::Affine(map::projection(1, &[0]));
        let scanned = append(
            &mut program,
            Expr::Fold(Fold {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: FoldInit::Zero,
                operand: source,
                in_map,
                out_map,
                keep: Keep::All,
                name: None,
            }),
        );

        let shapes = infer(&program, &[]).expect("cumsum shape infers");
        assert_eq!(shapes.of(scanned), &[16]);
    }

    #[test]
    fn reduce_output_shape_drops_the_contracted_dim() {
        let mut program = Vec::new();
        let source = block(&mut program, &[Extent::Static(4), Extent::Static(16)]);
        let in_map = IndexMap::Affine(map::projection(2, &[0, 1]));
        let out_map = IndexMap::Affine(map::projection(2, &[0]));
        let reduced = append(
            &mut program,
            Expr::Fold(Fold {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: FoldInit::Zero,
                operand: source,
                in_map,
                out_map,
                keep: Keep::Last,
                name: None,
            }),
        );

        let shapes = infer(&program, &[]).expect("reduce shape infers");
        assert_eq!(shapes.of(reduced), &[4]);
    }

    #[rstest]
    #[case::forward_reference(NodeId(2))]
    #[case::self_reference(NodeId(1))]
    fn a_non_backward_reference_is_rejected(#[case] referenced: NodeId) {
        let mut program = Vec::new();
        block(&mut program, &[Extent::Static(4)]);
        let map = IndexMap::Affine(map::projection(1, &[0]));
        append(
            &mut program,
            Expr::Zip {
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
        block(&mut program, &[Extent::Static(4)]);
        // an empty zip is malformed the instant it is pushed.
        let bad = Expr::Zip {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: Vec::new(),
            name: None,
        };

        let inference = Infer::new(&[]);
        inference.push(&program[0]).expect("block pushes cleanly");
        let error = inference
            .push(&bad)
            .expect_err("empty zip rejected at push");
        assert!(matches!(error, TensorError::EmptyZip(_)), "{error}");
    }

    #[test]
    fn streaming_and_batch_inference_agree() {
        let (program, _product, sum) = matmul_program();

        let batch = infer(&program, &[512]).expect("batch infers");

        let streamed = Infer::new(&[512]);
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
            Expr::Block {
                dtype: DType::Int8,
                shape: alloc::vec![Extent::Static(64)],
                name: None,
            },
        );
        let in_map = IndexMap::Affine(map::projection(1, &[0]));
        let out_map = IndexMap::Affine(map::projection(1, &[]));
        append(
            &mut program,
            Expr::Fold(Fold {
                dtype: DType::Int8,
                body: ScalarOp::Add,
                init: FoldInit::Zero,
                operand: quantized,
                in_map,
                out_map,
                keep: Keep::Last,
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
    /// `table`'s dim 0 (vocab), `d` is a plain projection onto `table`'s
    /// dim 1 (feature). The worked example `map.rs` and this module's own
    /// docs both describe.
    fn embedding_lookup_program(vocab: u32, dim: u32, seq: u32) -> (Vec<Expr>, NodeId, NodeId) {
        let mut program = Vec::new();
        let table = block(&mut program, &[Extent::Static(vocab), Extent::Static(dim)]);
        let ids = append(
            &mut program,
            Expr::Block {
                dtype: DType::Int32,
                shape: alloc::vec![Extent::Static(seq)],
                name: None,
            },
        );
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
        (program, ids, gathered)
    }

    /// A minimal, fully-gathered rank-1 map: `table[ids[s]]`, iteration
    /// space is just `s`. Used by the sad-path tests below where the
    /// embedding-style 2D shape would only add noise.
    fn fully_gathered_map(indices: NodeId) -> IndexMap {
        IndexMap::Computed {
            indices,
            index_map: map::projection(1, &[0]),
            base: map::AffineMap {
                iter_rank: 1,
                dims: alloc::vec![map::DimExpr::default()],
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
        let table = block(&mut program, &[Extent::Static(4)]);
        let float_ids = block(&mut program, &[Extent::Static(3)]); // float32, not integer
        append(
            &mut program,
            Expr::Zip {
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
        let table = block(&mut program, &[Extent::Static(4)]);
        let ids = append(
            &mut program,
            Expr::Block {
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
            Expr::Zip {
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

    #[test]
    fn a_scatter_data_dependent_fold_output_is_still_rejected_in_v1() {
        let mut program = Vec::new();
        let source = block(&mut program, &[Extent::Static(4)]);
        let ids = append(
            &mut program,
            Expr::Block {
                dtype: DType::Int32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        append(
            &mut program,
            Expr::Fold(Fold {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: FoldInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: fully_gathered_map(ids),
                keep: Keep::Last,
                name: None,
            }),
        );

        let error = infer(&program, &[]).expect_err("scatter is not shape-inferable in v1");
        assert!(matches!(error, TensorError::NotLowerable { .. }), "{error}");
    }

    #[test]
    fn infer_as_a_pipe_matches_the_free_function() {
        let (program, _product, sum) = matmul_program();

        let inference = Infer::new(&[512]);
        let mut last_shapes = None;
        for expr in &program {
            let (echoed, shapes) =
                proxima_primitives::block_on(Pipe::call(&inference, expr.clone()))
                    .expect("infer pipe step succeeds");
            assert_eq!(&echoed, expr, "the observe form hands the same Expr back");
            last_shapes = Some(shapes);
        }

        let via_pipe = last_shapes.expect("matmul program is non-empty");
        let via_free_function = infer(&program, &[512]).expect("free-function infer succeeds");
        assert_eq!(via_pipe.of(sum), via_free_function.of(sum));
    }
}
