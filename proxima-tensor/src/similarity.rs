//! Cosine top-k over a placed matrix.
//!
//! [`cosine_top_k`] composes exactly the two [`crate::op`] generators every
//! forward program in [`crate::spec`] already uses for a vocab projection
//! (`duplicate_head_reduce`'s own `sum_d(activation[d] * lm_head[d, v])`
//! shape) -- an [`Op::Elementwise`] multiply broadcasting the query row
//! across every candidate row, folded by an [`Op::Reduce`] over the shared
//! dimension -- evaluated through [`crate::cpu::evaluate`], the same
//! interpreter [`crate::cpu`]'s quantized dot kernels back onto for every
//! decode step. This is deliberately not a hand-rolled dot-product loop: the
//! whole point of routing it through [`Op`]/[`crate::cpu::evaluate`] is that
//! a caller placing `matrix` on Metal gets the identical computation run by
//! `omega`'s backend instead of a second, CPU-only implementation to keep in
//! sync.

use alloc::vec::Vec;

use crate::TensorError;
use crate::cpu::evaluate;
use crate::dtype::DType;
use crate::map::{self, IndexMap};
use crate::op::{self, Extent, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};

/// `sum_d(query[d] * matrix[row, d])` for every `row`, evaluated as one
/// [`Op::Elementwise`] multiply plus one [`Op::Reduce`] fold rather than a
/// per-row scalar loop -- see this module's own doc for why that is the
/// same shape [`crate::spec`]'s `duplicate_head_reduce` already runs for a
/// vocab projection, just with `matrix` playing `lm_head`'s role.
fn row_dot_products(query: &[f32], matrix: &[f32], dim: usize) -> Result<Vec<f32>, TensorError> {
    let row_count = matrix.len() / dim;
    let mut program: Vec<Op> = Vec::with_capacity(3);

    let query_node = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(dim as u32)],
            name: None,
        },
    );
    let matrix_node = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(row_count as u32), Extent::Static(dim as u32)],
            name: None,
        },
    );

    // iteration space is [row, dim]; query broadcasts over `row` (axis 0),
    // matrix reads both axes in its own on-disk order.
    let product = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (query_node, IndexMap::Affine(map::projection(2, &[1]))),
                (matrix_node, IndexMap::Affine(map::projection(2, &[0, 1]))),
            ],
            name: None,
        },
    );

    let dots = op::append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            // dropping axis 1 (`dim`) from the projected set is the fold:
            // every row keeps its own accumulator, `dim` collapses into it.
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: None,
        }),
    );

    let evaluated = evaluate(&program, &[], &[query, matrix], &[dots])?;
    let (data, _shape) = evaluated
        .get(dots)
        .ok_or(TensorError::NotLowerable {
            node: dots,
            reason: "cosine_top_k's own dot-product root was not evaluated",
        })?;
    Ok(data.to_vec())
}

fn l2_norm(vector: &[f32]) -> f32 {
    vector.iter().map(|value| value * value).sum::<f32>().sqrt()
}

/// Cosine similarity between `query` (`dim` elements) and every `dim`-wide
/// row of `matrix` (`matrix.len() / dim` rows, row-major), returning the top
/// `k` rows as `(row_index, cosine_similarity)`, highest similarity first.
///
/// The dot products backing every cosine score come from `row_dot_products`
/// -- one [`Op::Elementwise`]+[`Op::Reduce`] pair evaluated by
/// [`crate::cpu::evaluate`], not a per-row scalar loop. Row and query norms
/// are `sqrt(sum_d(x[d]^2))`, a plain `O(n * dim)` scalar fold with no
/// cross-row structure to route through the tensor evaluator profitably --
/// `l2_norm` is that fold, called once per row plus once for `query`.
///
/// # Errors
///
/// [`TensorError`] if `matrix.len()` is not a multiple of `dim`, or whatever
/// [`crate::cpu::evaluate`] can fail with while lowering the dot-product
/// program.
#[must_use = "the ranked rows are the entire result"]
pub fn cosine_top_k(
    query: &[f32],
    matrix: &[f32],
    dim: usize,
    k: usize,
) -> Result<Vec<(usize, f32)>, TensorError> {
    if dim == 0 || !matrix.len().is_multiple_of(dim) {
        return Err(TensorError::InputSizeMismatch {
            node: NodeId(0),
            expected: dim,
            found: matrix.len(),
        });
    }

    let query_norm = l2_norm(query);
    let dots = row_dot_products(query, matrix, dim)?;

    let mut scored: Vec<(usize, f32)> = dots
        .iter()
        .enumerate()
        .map(|(row, &dot)| {
            let row_slice = &matrix[row * dim..(row + 1) * dim];
            let denom = query_norm * l2_norm(row_slice);
            let cosine = if denom == 0.0 { 0.0 } else { dot / denom };
            (row, cosine)
        })
        .collect();

    scored.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    scored.truncate(k);
    Ok(scored)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::cosine_top_k;

    /// Three 2-d rows with a hand-computed ranking: `query = [1, 0]` is
    /// identical to row 0 (cosine 1.0), orthogonal to row 1 (cosine 0.0),
    /// and anti-parallel to row 2 (cosine -1.0) -- `k = 2` must return rows
    /// `[0, 1]` in that order, row 2 excluded.
    #[test]
    fn cosine_top_k_ranks_a_synthetic_matrix_by_hand_computed_cosine() {
        let query = [1.0_f32, 0.0];
        let matrix = [1.0_f32, 0.0, 0.0, 1.0, -1.0, 0.0];

        let ranked = cosine_top_k(&query, &matrix, 2, 2).expect("evaluates a tiny dot program");

        assert_eq!(
            ranked.iter().map(|(row, _)| *row).collect::<Vec<_>>(),
            alloc::vec![0, 1],
            "identical row must rank above the orthogonal row"
        );
        assert!(
            (ranked[0].1 - 1.0).abs() < 1e-6,
            "row 0 is query itself: cosine must be 1.0, got {}",
            ranked[0].1
        );
        assert!(
            ranked[1].1.abs() < 1e-6,
            "row 1 is orthogonal to query: cosine must be 0.0, got {}",
            ranked[1].1
        );
    }

    #[test]
    fn cosine_top_k_rejects_a_matrix_length_not_a_multiple_of_dim() {
        let query = [1.0_f32, 0.0, 0.0];
        let matrix = [1.0_f32, 0.0];

        let result = cosine_top_k(&query, &matrix, 3, 1);

        assert!(
            result.is_err(),
            "a 2-element matrix cannot hold whole dim=3 rows"
        );
    }
}
