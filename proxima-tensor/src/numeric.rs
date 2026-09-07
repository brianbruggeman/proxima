//! Numerical permission ladder for bind-time and plan-time rewrites.
//!
//! [`op::is_associative`](crate::op::ScalarOp::is_associative) has exactly
//! one caller today ([`shape::ShapeTable`](crate::shape::ShapeTable)'s
//! accumulator-width check) — nothing consults it before reassociating a
//! fold. [`NumericPolicy`] and [`admit`] close that gap: a rewrite that
//! changes which bits come out (fusing a multiply and add into one FMA,
//! reordering an associative reduction, approximating a transcendental)
//! must be classified with [`NumericRewrite`] and cleared through [`admit`]
//! before it fires, instead of firing unconditionally.

use crate::error::TensorError;

/// What class of float-bit-changing rewrite a bind (or a GPU plan) may
/// apply. A total order: `BitExact < FusedNoReassociation <
/// ReassociationPermitted < FastMath`. The default is the most conservative
/// rung — every rewrite this crate ships unconditionally today (identity
/// elimination, chain fusion, reduce-epilogue fusion) is bit-exact by
/// construction and is admitted at the default, so nothing regresses; only
/// a rewrite that actually changes bits needs the caller to opt up.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum NumericPolicy {
    /// Bit-parity with [`crate::cpu::evaluate`] — no reordering, no fused
    /// rounding.
    #[default]
    BitExact,
    /// Same operand order and count; permits merging a multiply and an add
    /// into one hardware FMA (one rounding instead of two).
    FusedNoReassociation,
    /// Permits reordering an associative fold: tree-reduce, the simdgroup
    /// context-chunk merge, quantization-scale factoring across a block.
    ReassociationPermitted,
    /// Permits approximate transcendentals/reciprocals with a bounded
    /// relative error, on top of everything `ReassociationPermitted` allows.
    FastMath,
}

/// One rewrite class this crate or a GPU backend may apply, and the minimum
/// [`NumericPolicy`] it requires. `#[non_exhaustive]` — every future
/// bit-changing rewrite adds a variant and a [`NumericRewrite::minimum_level`]
/// arm before it may fire, never a second, parallel check.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericRewrite {
    /// [`crate::bind`]'s identity-elimination fold, restricted to the ONE
    /// sub-case that is bit-exact for every `f32` including NaN and signed
    /// zero: `x * 1.0`. Always admitted at [`NumericPolicy::BitExact`].
    IdentityElimination,
    /// [`crate::bind`]'s identity-elimination fold for `x + 0.0`,
    /// `max(x, -inf)`, and `min(x, +inf)` -- each collapses to `x` for every
    /// FINITE input, but not for every `f32`: `max(NaN, -inf)` evaluates to
    /// `-inf` (IEEE 754 `maxNum`/[`f32::max`]'s own "if one argument is NaN,
    /// return the other" rule), while eliminating the op returns the
    /// survivor `NaN` instead; `(-0.0) + 0.0` evaluates to `+0.0`, while
    /// eliminating the op returns `-0.0`. This changes bits without
    /// reassociating anything (no operand order changes, no operand count
    /// changes -- an op simply vanishes), so [`NumericPolicy::BitExact`]
    /// does NOT admit it; its floor is [`NumericPolicy::FusedNoReassociation`],
    /// the first rung this crate's ladder allows to change bits at all.
    IdentityEliminationSignedZeroNan,
    /// [`crate::bind`]'s elementwise/reduce chain fusion.
    ChainFusion,
    /// [`crate::bind`]'s reduce-epilogue fusion.
    ReduceEpilogueFusion,
    /// Merging a multiply and an add into one hardware FMA.
    FmaContraction,
    /// Reordering an associative reduction as a tree instead of a left fold.
    TreeReduce,
    /// The GPU cross-simdgroup context-chunk merge in an attention kernel.
    ContextChunkMerge,
    /// Factoring a dequantization scale across a block instead of per element.
    DequantScaleFactoring,
    /// An approximate transcendental/reciprocal with bounded relative error.
    FastMathApprox,
}

impl NumericRewrite {
    /// The lowest [`NumericPolicy`] under which this rewrite may fire.
    #[must_use]
    pub const fn minimum_level(self) -> NumericPolicy {
        match self {
            Self::IdentityElimination | Self::ChainFusion | Self::ReduceEpilogueFusion => {
                NumericPolicy::BitExact
            }
            Self::IdentityEliminationSignedZeroNan | Self::FmaContraction => {
                NumericPolicy::FusedNoReassociation
            }
            Self::TreeReduce | Self::ContextChunkMerge | Self::DequantScaleFactoring => {
                NumericPolicy::ReassociationPermitted
            }
            Self::FastMathApprox => NumericPolicy::FastMath,
        }
    }
}

/// The one admission check every rewrite site calls before firing — a typed
/// error, never a silent numerics change.
///
/// # Errors
/// [`TensorError::NumericPolicyTooStrict`] when `policy` is below
/// `rewrite`'s [`NumericRewrite::minimum_level`].
pub fn admit(policy: NumericPolicy, rewrite: NumericRewrite) -> Result<(), TensorError> {
    let minimum = rewrite.minimum_level();
    if policy < minimum {
        return Err(TensorError::NumericPolicyTooStrict {
            rewrite,
            minimum,
            granted: policy,
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn ladder_orders_bit_exact_below_fast_math() {
        assert!(NumericPolicy::BitExact < NumericPolicy::FusedNoReassociation);
        assert!(NumericPolicy::FusedNoReassociation < NumericPolicy::ReassociationPermitted);
        assert!(NumericPolicy::ReassociationPermitted < NumericPolicy::FastMath);
    }

    #[test]
    fn bit_exact_rewrites_admitted_at_the_default_policy() {
        let policy = NumericPolicy::default();
        assert_eq!(policy, NumericPolicy::BitExact);
        assert!(admit(policy, NumericRewrite::IdentityElimination).is_ok());
        assert!(admit(policy, NumericRewrite::ChainFusion).is_ok());
        assert!(admit(policy, NumericRewrite::ReduceEpilogueFusion).is_ok());
    }

    #[test]
    fn reassociating_rewrite_rejected_under_bit_exact_policy() {
        let error = admit(NumericPolicy::BitExact, NumericRewrite::ContextChunkMerge)
            .expect_err("context-chunk merge reassociates and needs ReassociationPermitted");
        assert_eq!(
            error,
            TensorError::NumericPolicyTooStrict {
                rewrite: NumericRewrite::ContextChunkMerge,
                minimum: NumericPolicy::ReassociationPermitted,
                granted: NumericPolicy::BitExact,
            }
        );
    }

    #[test]
    fn reassociating_rewrite_admitted_once_the_policy_opts_up() {
        assert!(admit(NumericPolicy::ReassociationPermitted, NumericRewrite::ContextChunkMerge).is_ok());
        assert!(admit(NumericPolicy::FastMath, NumericRewrite::ContextChunkMerge).is_ok());
    }

    /// The NaN/signed-zero-changing identity eliminations (`x+0`,
    /// `max(x,-inf)`, `min(x,+inf)`) are NOT admitted at the library
    /// default -- only `x*1` (plain [`NumericRewrite::IdentityElimination`])
    /// is bit-exact for every float and clears [`NumericPolicy::BitExact`].
    #[test]
    fn signed_zero_nan_identity_elimination_rejected_under_bit_exact_policy() {
        let policy = NumericPolicy::default();
        assert_eq!(policy, NumericPolicy::BitExact);
        assert!(admit(policy, NumericRewrite::IdentityElimination).is_ok());
        let error = admit(policy, NumericRewrite::IdentityEliminationSignedZeroNan)
            .expect_err("x+0/max(x,-inf)/min(x,+inf) change bits on NaN/signed-zero inputs");
        assert_eq!(
            error,
            TensorError::NumericPolicyTooStrict {
                rewrite: NumericRewrite::IdentityEliminationSignedZeroNan,
                minimum: NumericPolicy::FusedNoReassociation,
                granted: NumericPolicy::BitExact,
            }
        );
    }

    #[test]
    fn signed_zero_nan_identity_elimination_admitted_once_the_policy_opts_up() {
        assert!(
            admit(
                NumericPolicy::FusedNoReassociation,
                NumericRewrite::IdentityEliminationSignedZeroNan
            )
            .is_ok()
        );
        assert!(
            admit(
                NumericPolicy::ReassociationPermitted,
                NumericRewrite::IdentityEliminationSignedZeroNan
            )
            .is_ok()
        );
    }
}
