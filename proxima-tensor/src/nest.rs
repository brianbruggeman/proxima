//! The shared lowering seam between the tensor algebra and any executor.
//!
//! [`lower`] turns a validated program plus its [`Shapes`] into a flat list
//! of [`Nest`]s, one per expression that actually computes. A `Nest` says
//! only *what* to compute: which buffers to read, at what strides, combined
//! by which scalar op, and — for a fold — how the reduction is shaped. It
//! says nothing about *how*: [`cpu`](crate::cpu) interprets a `Nest` with
//! nested loops; a GPU backend could instead emit kernel source from the same
//! descriptor. Neither backend's shape belongs in this module.
//!
//! Like [`shape::Infer`](crate::shape::Infer), [`Lower`] is a sans-IO push
//! state machine: a program can arrive a step at a time, and lowering must
//! not require the whole thing in hand. [`lower`] is the batch driver over
//! it. What *does* require the whole program in hand is liveness
//! ([`live::annotate`](crate::live::annotate)) — computed once, upstream,
//! and handed to `Lower` as a plain kill-flag list it never has to guess at.
//!
//! The one optimization decided here: when a fold's operand is a zip whose
//! last use is that fold (exact liveness, from [`live::annotate`]), the zip
//! is never materialized — its body is composed directly into the fold's
//! `Nest`, which is the difference between an O(extents) buffer and an
//! O(iteration space) one for something like matmul. A zip whose last use is
//! anything else (another zip, a non-fusable fold, or nothing — a requested
//! output or dead code) materializes as its own `Nest`, emitted the moment
//! that use is seen (or, for dead code and outputs never referenced again,
//! when [`Lower::finish`] flushes it).

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::error::TensorError;
use crate::expr::{Expr, Fold, FoldInit, Keep, NodeId, ScalarOp};
use crate::live;
use crate::map::{AffineMap, IndexMap};
use crate::shape::{self, Shapes};

/// One operand's address into its own buffer, expressed directly in a
/// [`Nest`]'s iteration-dim space: `strides[d]` is how far the linear offset
/// moves per step of iteration dim `d`. A dim this operand never varies along
/// (broadcast) simply has stride 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StridedView {
    pub base: i64,
    pub strides: Vec<i64>,
}

impl StridedView {
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
    pub fn stride(&self, dim: u16) -> i64 {
        self.strides.get(dim as usize).copied().unwrap_or(0)
    }
}

/// The reduction half of a fold [`Nest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reduction {
    pub body: ScalarOp,
    pub init: FoldInit,
    pub keep: Keep,
    /// Iteration dims that survive to the output, in the fold's `out_map`
    /// operand-dim order. The last entry (if any) is the innermost loop.
    pub output_dims: Vec<u16>,
    pub out_view: StridedView,
}

/// One computed region: a loop nest over `extents`, combining `operands`
/// through `body`, optionally folded down by `reduction`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nest {
    pub node: NodeId,
    pub extents: Vec<u64>,
    pub body: ScalarOp,
    /// Each operand: which node's buffer to read, and at what strides over
    /// this nest's iteration space.
    pub operands: Vec<(NodeId, StridedView)>,
    pub reduction: Option<Reduction>,
}

struct HeldZip {
    body: ScalarOp,
    operands: Vec<(NodeId, IndexMap)>,
}

/// The prefix state of lowering: zips seen but not yet materialized.
pub struct Lower {
    held: BTreeMap<NodeId, HeldZip>,
}

impl Default for Lower {
    fn default() -> Self {
        Self::new()
    }
}

impl Lower {
    #[must_use]
    pub fn new() -> Self {
        Self {
            held: BTreeMap::new(),
        }
    }

    /// Judge one expression: hold a zip, or emit whatever is now ready.
    ///
    /// May return more than one [`Nest`]: consuming a held zip that turns out
    /// not to fuse must materialize it before the current expression can read
    /// it, so a single push can ready both that standalone nest and the
    /// current expression's own.
    pub fn push(
        &mut self,
        node: NodeId,
        expr: &Expr,
        shapes: &Shapes,
        retires: &[NodeId],
    ) -> Result<Vec<Nest>, TensorError> {
        let mut emitted = Vec::new();

        match expr {
            Expr::Block { .. } => {}
            Expr::Zip { body, operands, .. } => {
                for (operand_node, _) in operands {
                    self.materialize_if_held(*operand_node, shapes, &mut emitted);
                }
                self.held.insert(
                    node,
                    HeldZip {
                        body: *body,
                        operands: operands.clone(),
                    },
                );
            }
            Expr::Fold(fold) => {
                let fuses = retires.contains(&fold.operand)
                    && is_identity_projection(&fold.in_map)
                    && self.held.contains_key(&fold.operand);

                let (body, operands) =
                    if let Some(held) = fuses.then(|| self.held.remove(&fold.operand)).flatten() {
                        compose_fused_operands(shapes, &fold.in_map, held.body, &held.operands)
                    } else {
                        self.materialize_if_held(fold.operand, shapes, &mut emitted);
                        let view = strided_view(fold.in_map.affine(), shapes.of(fold.operand));
                        (ScalarOp::Identity, vec![(fold.operand, view)])
                    };

                emitted.push(lower_fold_nest(node, fold, shapes, body, operands));
            }
        }

        Ok(emitted)
    }

    /// Flush every zip still held: it was either a requested output or dead
    /// code, and either way it materializes as its own nest.
    pub fn finish(self, shapes: &Shapes) -> Result<Vec<Nest>, TensorError> {
        Ok(self
            .held
            .into_iter()
            .map(|(node, held)| lower_zip_nest(node, shapes, held.body, &held.operands))
            .collect())
    }

    fn materialize_if_held(&mut self, node: NodeId, shapes: &Shapes, emitted: &mut Vec<Nest>) {
        if let Some(held) = self.held.remove(&node) {
            emitted.push(lower_zip_nest(node, shapes, held.body, &held.operands));
        }
    }
}

fn lower_zip_nest(
    node: NodeId,
    shapes: &Shapes,
    body: ScalarOp,
    operands: &[(NodeId, IndexMap)],
) -> Nest {
    let extents = shapes.of(node).to_vec();
    let built_operands = operands
        .iter()
        .map(|(operand_node, map)| {
            (
                *operand_node,
                strided_view(map.affine(), shapes.of(*operand_node)),
            )
        })
        .collect();
    Nest {
        node,
        extents,
        body,
        operands: built_operands,
        reduction: None,
    }
}

fn lower_fold_nest(
    node: NodeId,
    fold: &Fold,
    shapes: &Shapes,
    body: ScalarOp,
    operands: Vec<(NodeId, StridedView)>,
) -> Nest {
    let out_affine = fold.out_map.affine();
    let out_view = strided_view(out_affine, shapes.of(node));
    let output_dims = pure_projection_dims(out_affine);
    Nest {
        node,
        extents: shape::fold_iteration_extents(fold, shapes),
        body,
        operands,
        reduction: Some(Reduction {
            body: fold.body,
            init: fold.init,
            keep: fold.keep,
            output_dims,
            out_view,
        }),
    }
}

fn pure_projection_dims(affine: &AffineMap) -> Vec<u16> {
    affine
        .dims
        .iter()
        .filter_map(|dim| match dim.terms.as_slice() {
            [term] if term.coeff == 1 => Some(term.iter_dim),
            _ => None,
        })
        .collect()
}

/// A map fusion can compose through: every dim a plain, unshifted projection.
/// Anything richer (a window, a slice, a stride) still lowers correctly, it
/// just materializes its operand instead of composing through it.
fn is_identity_projection(map: &IndexMap) -> bool {
    if map.is_data_dependent() {
        return false;
    }
    map.affine()
        .dims
        .iter()
        .all(|dim| dim.offset == 0 && matches!(dim.terms.as_slice(), [term] if term.coeff == 1))
}

fn compose_fused_operands(
    shapes: &Shapes,
    in_map: &IndexMap,
    zip_body: ScalarOp,
    zip_operands: &[(NodeId, IndexMap)],
) -> (ScalarOp, Vec<(NodeId, StridedView)>) {
    let in_affine = in_map.affine();
    let iter_rank = in_affine.iter_rank;
    let zip_dim_to_fold_dim: Vec<u16> = in_affine
        .dims
        .iter()
        .map(|dim| dim.terms[0].iter_dim)
        .collect();

    let operands = zip_operands
        .iter()
        .map(|(operand_node, map)| {
            let zip_view = strided_view(map.affine(), shapes.of(*operand_node));
            let view = remap_strides(&zip_view, &zip_dim_to_fold_dim, iter_rank);
            (*operand_node, view)
        })
        .collect();

    (zip_body, operands)
}

fn strided_view(affine: &AffineMap, operand_shape: &[u64]) -> StridedView {
    let element_strides = row_major_strides(operand_shape);
    let mut strides = vec![0i64; affine.iter_rank as usize];
    let mut base = 0i64;
    for (dim_index, dim) in affine.dims.iter().enumerate() {
        let stride = element_strides[dim_index];
        base += i64::from(dim.offset) * stride;
        for term in &dim.terms {
            strides[term.iter_dim as usize] += i64::from(term.coeff) * stride;
        }
    }
    StridedView { base, strides }
}

fn row_major_strides(shape: &[u64]) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];
    let mut accumulator = 1i64;
    for (dim_index, extent) in shape.iter().enumerate().rev() {
        strides[dim_index] = accumulator;
        accumulator *= *extent as i64;
    }
    strides
}

fn remap_strides(view: &StridedView, zip_dim_to_fold_dim: &[u16], iter_rank: u16) -> StridedView {
    let mut strides = vec![0i64; iter_rank as usize];
    for (zip_dim, fold_dim) in zip_dim_to_fold_dim.iter().enumerate() {
        strides[*fold_dim as usize] += view.stride(zip_dim as u16);
    }
    StridedView {
        base: view.base,
        strides,
    }
}

/// Batch driver: computes liveness once, then streams every expression
/// through a fresh [`Lower`], flushing whatever remains held at the end.
pub fn lower(
    program: &[Expr],
    shapes: &Shapes,
    outputs: &[NodeId],
) -> Result<Vec<Nest>, TensorError> {
    let retires = live::annotate(program, outputs);
    let mut lowering = Lower::new();
    let mut nests = Vec::new();
    for (position, expr) in program.iter().enumerate() {
        let node = NodeId(position as u32);
        nests.extend(lowering.push(node, expr, shapes, &retires[position])?);
    }
    nests.extend(lowering.finish(shapes)?);
    Ok(nests)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::expr::{Extent, append};
    use crate::map;

    fn matmul_program() -> (Vec<Expr>, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Expr::Block {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Symbolic(0), Extent::Static(768)],
                name: None,
            },
        );
        let rhs = append(
            &mut program,
            Expr::Block {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(768), Extent::Static(3072)],
                name: None,
            },
        );
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
                init: crate::expr::FoldInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Last,
                name: Some("matmul".into()),
            }),
        );
        (program, product, sum, lhs)
    }

    #[test]
    fn matmul_lowers_to_one_fused_nest_not_two() {
        let (program, product, sum, _lhs) = matmul_program();
        let shapes = shape::infer(&program, &[512]).expect("matmul infers");
        let nests = lower(&program, &shapes, &[]).expect("matmul lowers");

        assert_eq!(nests.len(), 1, "the zip must not materialize separately");
        assert_eq!(nests[0].node, sum);
        assert!(nests[0].reduction.is_some());
        assert_ne!(
            nests[0].body,
            ScalarOp::Identity,
            "the fused body is the zip's multiply"
        );
        let _ = product;
    }

    #[test]
    fn requesting_the_intermediate_zip_as_an_output_prevents_fusion() {
        let (program, product, sum, _lhs) = matmul_program();
        let shapes = shape::infer(&program, &[512]).expect("matmul infers");
        let nests =
            lower(&program, &shapes, &[product, sum]).expect("matmul lowers with two outputs");

        assert_eq!(nests.len(), 2, "the requested-output zip must materialize");
        assert!(
            nests
                .iter()
                .any(|nest| nest.node == product && nest.reduction.is_none())
        );
        assert!(
            nests
                .iter()
                .any(|nest| nest.node == sum && nest.reduction.is_some())
        );
    }

    #[test]
    fn a_broadcast_operand_has_stride_zero_in_the_broadcast_dim() {
        let mut program = Vec::new();
        let matrix = append(
            &mut program,
            Expr::Block {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4), Extent::Static(8)],
                name: None,
            },
        );
        let bias = append(
            &mut program,
            Expr::Block {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(8)],
                name: None,
            },
        );
        let sum = append(
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

        let shapes = shape::infer(&program, &[]).expect("broadcast infers");
        let nests = lower(&program, &shapes, &[]).expect("broadcast lowers");
        let nest = nests
            .iter()
            .find(|nest| nest.node == sum)
            .expect("sum emitted");
        assert_eq!(
            nest.operands[1].1.stride(0),
            0,
            "bias never varies over the batch dim"
        );
        assert_ne!(
            nest.operands[0].1.stride(0),
            0,
            "matrix does vary over the batch dim"
        );
    }

    #[test]
    fn a_conv_window_operand_folds_two_terms_into_one_stride_slot() {
        let mut program = Vec::new();
        let anchor = append(
            &mut program,
            Expr::Block {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4), Extent::Static(2)],
                name: None,
            },
        );
        let signal = append(
            &mut program,
            Expr::Block {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(8)],
                name: None,
            },
        );
        let window = IndexMap::Affine(map::affine(
            2,
            &[(
                &[
                    crate::map::AffineTerm::scaled(0, 2),
                    crate::map::AffineTerm::scaled(1, 1),
                ],
                0,
            )],
        ));
        let touched = append(
            &mut program,
            Expr::Zip {
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
        let nests = lower(&program, &shapes, &[]).expect("conv window lowers");
        let nest = nests
            .iter()
            .find(|nest| nest.node == touched)
            .expect("touched emitted");
        let signal_view = &nest.operands[1].1;
        assert_eq!(
            signal_view.strides.len(),
            2,
            "one stride slot per iteration dim"
        );
        assert_ne!(signal_view.stride(0), 0, "stride term contributes");
        assert_ne!(signal_view.stride(1), 0, "dilation term contributes");
    }

    #[test]
    fn transpose_view_has_permuted_strides() {
        let mut program = Vec::new();
        let matrix = append(
            &mut program,
            Expr::Block {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(3), Extent::Static(5)],
                name: None,
            },
        );
        let transposed = append(
            &mut program,
            Expr::Zip {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(matrix, IndexMap::Affine(map::projection(2, &[1, 0])))],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("transpose infers");
        let nests = lower(&program, &shapes, &[]).expect("transpose lowers");
        let nest = nests
            .iter()
            .find(|nest| nest.node == transposed)
            .expect("transposed emitted");
        let view = &nest.operands[0].1;
        // matrix is row-major [3, 5]: elem strides are [5, 1]. dim0 of the
        // operand (stride 5) projects iter_dim 1; dim1 (stride 1) projects
        // iter_dim 0, so the strides land permuted relative to iteration order.
        assert_eq!(view.stride(0), 1);
        assert_eq!(view.stride(1), 5);
    }
}
