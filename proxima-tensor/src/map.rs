//! The index-map grammar — where this crate's expressiveness actually lives.
//!
//! Three expression forms carry the algebra, but they only work because an
//! operand can be *related* to the iteration space by something richer than a
//! permutation. Every shape operation is a map, not a variant:
//!
//! | operation | map |
//! |---|---|
//! | transpose | permute the projected iteration dims |
//! | broadcast | project fewer dims than the iteration space has |
//! | slice | a non-zero `offset` |
//! | stride / dilation | a `coeff` other than 1 |
//! | convolution | two terms in one dim: `h*stride + r*dilation` |
//! | gather (read-side) | [`IndexMap::Computed`] — one dim's index comes from a node |
//!
//! Convolution is why a dim is a *sum* of terms rather than a single
//! projection. Without that, windowed access needs its own expression form,
//! and the three generators become four, then a dozen.
//!
//! A map owns its dims and terms directly — there is no interned, span-based
//! arena here, because [`Expr`](crate::expr::Expr) itself no longer lives in
//! one: a tensor program is a plain `Vec<Expr>`, and each `Expr` is
//! self-contained.

use alloc::vec::Vec;

use crate::expr::NodeId;

/// One `coeff * iter[iter_dim]` contribution to an operand index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct AffineTerm {
    pub iter_dim: u16,
    pub coeff: i32,
}

impl AffineTerm {
    #[must_use]
    pub const fn projection(iter_dim: u16) -> Self {
        Self { iter_dim, coeff: 1 }
    }

    #[must_use]
    pub const fn scaled(iter_dim: u16, coeff: i32) -> Self {
        Self { iter_dim, coeff }
    }
}

/// One operand dimension, as `sum(terms) + offset` over the iteration space.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct DimExpr {
    pub terms: Vec<AffineTerm>,
    pub offset: i32,
}

/// Relates an iteration space of rank `iter_rank` to an operand's index space.
///
/// `dims` holds one [`DimExpr`] per operand dimension, so the operand's rank
/// is `dims.len()` — which may be lower than `iter_rank` (a broadcast) or
/// reorder it (a transpose).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct AffineMap {
    pub iter_rank: u16,
    pub dims: Vec<DimExpr>,
}

/// How an operand is addressed. `Affine` is statically analysable and is what
/// shape inference and lowering reason about; `Computed` is a data-dependent
/// index and is the reason gather needs no expression form of its own.
///
/// A gather touches exactly one operand dimension — the one the fetched
/// index selects (`gathered_dim`) — while the rest stay affine. `index_map`
/// and `base` are both drawn from the *same* iteration space: `index_map`
/// says how to read the `indices` tensor from it, `base` says how the
/// operand's other dims read from it. `base`'s entry at `gathered_dim` is
/// never consulted (its `terms` are empty by construction) because that
/// dim's address comes from the fetched index at evaluation time, not from
/// iteration.
///
/// `index_map` is always a plain [`AffineMap`], never another `IndexMap` —
/// the indices tensor itself must be affinely addressed, so a gather cannot
/// nest inside another gather. This is enforced by the type, not a runtime
/// check.
///
/// `indices` is an integer [`crate::dtype::DType`] logically, but every
/// backend this crate ships (`cpu.rs`'s interpreter, `omega`'s Metal driver)
/// carries every buffer as f32 — including `indices` — rather than plumbing
/// a second integer-buffer kind through the stack for this one case. f32's
/// 24-bit mantissa represents every integer up to `2^24` (16,777,216)
/// exactly, so [`crate::shape::infer`] rejects any gathered dim wider than
/// that (`TensorError::GatherExtentExceedsExactFloat`) rather than let a
/// fetched index silently lose precision and select the wrong row. Lifting
/// the ceiling means adding real integer buffers, not raising a constant.
///
/// Scatter — a data-dependent *output* map on a [`crate::expr::Fold`] — is a
/// different, still-unsupported feature: it needs atomics for colliding
/// writes and stays out of scope here (see
/// [`TensorError::NotLowerable`](crate::error::TensorError::NotLowerable)).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum IndexMap {
    Affine(AffineMap),
    Computed {
        /// The node supplying fetched index values; must be a backwards
        /// reference with an integer [`crate::dtype::DType`].
        indices: NodeId,
        /// How the iteration space addresses the `indices` tensor.
        index_map: AffineMap,
        /// How the iteration space addresses the operand's non-gathered
        /// dims; the entry at `gathered_dim` is unused.
        base: AffineMap,
        /// Which operand dimension the fetched index selects.
        gathered_dim: u16,
    },
}

impl IndexMap {
    #[must_use]
    pub const fn affine(&self) -> &AffineMap {
        match self {
            Self::Affine(map) | Self::Computed { base: map, .. } => map,
        }
    }

    #[must_use]
    pub const fn is_data_dependent(&self) -> bool {
        matches!(self, Self::Computed { .. })
    }
}

/// A plain projection: operand dim `n` reads iteration dim `projected[n]`.
/// Covers transpose, broadcast, and identity — the overwhelming majority.
#[must_use]
pub fn projection(iter_rank: u16, projected: &[u16]) -> AffineMap {
    let dims = projected
        .iter()
        .map(|iter_dim| DimExpr {
            terms: alloc::vec![AffineTerm::projection(*iter_dim)],
            offset: 0,
        })
        .collect();
    AffineMap { iter_rank, dims }
}

/// A general affine map: one `(terms, offset)` pair per operand dimension.
/// What a projection cannot express — convolution windows, dilation, slices
/// with a stride — goes through this constructor instead.
#[must_use]
pub fn affine(iter_rank: u16, dims: &[(&[AffineTerm], i32)]) -> AffineMap {
    let dims = dims
        .iter()
        .map(|(terms, offset)| DimExpr {
            terms: terms.to_vec(),
            offset: *offset,
        })
        .collect();
    AffineMap { iter_rank, dims }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn projection_has_unit_coefficient_and_scaled_does_not() {
        assert_eq!(AffineTerm::projection(2).coeff, 1);
        assert_eq!(AffineTerm::scaled(2, 3).coeff, 3);
        assert_eq!(AffineTerm::scaled(2, 3).iter_dim, 2);
    }

    #[test]
    fn computed_and_affine_both_expose_a_base_map() {
        let base = AffineMap {
            iter_rank: 2,
            dims: alloc::vec![DimExpr {
                terms: alloc::vec![AffineTerm::projection(0)],
                offset: 0,
            }],
        };
        let index_map = projection(2, &[1]);
        let direct = IndexMap::Affine(base.clone());
        let gathered = IndexMap::Computed {
            indices: NodeId(7),
            index_map,
            base: base.clone(),
            gathered_dim: 0,
        };
        assert_eq!(*direct.affine(), base);
        assert_eq!(*gathered.affine(), base);
        assert!(!direct.is_data_dependent());
        assert!(gathered.is_data_dependent());
    }

    #[test]
    fn a_convolution_dim_is_two_terms() {
        // h*stride + r*dilation is the whole reason DimExpr sums terms.
        let window = affine(
            2,
            &[(&[AffineTerm::scaled(0, 2), AffineTerm::scaled(1, 1)], -1)],
        );
        assert_eq!(window.dims[0].terms.len(), 2);
        assert_eq!(window.dims[0].terms[0].coeff, 2, "stride");
        assert_eq!(window.dims[0].terms[1].coeff, 1, "dilation");
        assert_eq!(window.dims[0].offset, -1, "padding is the offset");
    }

    #[test]
    fn projection_reads_transpose_broadcast_and_identity() {
        let identity = projection(2, &[0, 1]);
        let transpose = projection(2, &[1, 0]);
        let broadcast = projection(2, &[1]);
        assert_eq!(identity.dims.len(), 2);
        assert_eq!(transpose.dims[0].terms[0].iter_dim, 1);
        assert_eq!(
            broadcast.dims.len(),
            1,
            "broadcast projects fewer dims than iter_rank"
        );
    }
}
